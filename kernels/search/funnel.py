"""One round of the funnel (decided 2026-10-22): volume through measured tiers, every cheap tier a measurement. A round
takes one parent (an MCTS expansion, `expand`) or several (one generation of the genetic search, `generation`,
kernels/search/generations.py; decided 2026-10-26).

From the parents:
  0. a parent with no ncu counters yet is profiled (stage 3): counters feed only the prompts of a parent's children,
     so a program is profiled when it is first chosen as a parent, not when it is examined (decided 2026-10-27)
  1. generate: for one parent, one judge call per slice of the facts (the measurements; measured winners elsewhere
     in the tree; the compiler's report; the PTX comparison); for several, one call per slot (a parent, an idea type
     and a quota, kernels/search/generations.py), a refine slot's slice rotated across the generation's parents, so
     every call reads one slice and every slice is read. Every judge call also gets the memory
     (kernels/search/memory.py: ideas that won on other tasks, every idea tried on this one). The judge calls cycle
     through `Models.judges` (3 Opus : 1 Sonnet by default), for diversity.
  2. dedup / merge: hypotheses within `SAME` cosine are one idea, merged by a cheap model keeping every detail
  3. past failures: an idea like an archived one that never beat its parent goes back to the judge with those records
     and survives only if the judge names what differs now
  4. the wide coder writes each idea; edits that do not apply or repeat a judged program are dropped; with
     `precheck`, what fails to build in a jail with no GPU is dropped too, and the compiler's report is kept
  5. correctness on the GPU (stage 1), `parallel` at a time; failures go to the repair queue (the strong coder)
  6. the quick speed test (the most-called shape only); programs within `RACE` of the cutoff are timed once more and
     ranked by their mean (successive halving's racing: repeats only where one timing cannot separate them); the
     fastest `keep` go on (with `keep_by`, the fastest of each idea type, so each type gets its share of full exams)
  7. the strong coder rewrites each of those from the idea and the passing version; the rewrite replaces the draft
     only if it builds, passes stage 1 and its quick time is faster (decided 2026-10-27: the strong model on the
     programs that reach the full exam, not on every correct draft)
  8. the full exam (stage 2) and the tree; every idea and how far it got go to the log, with the rank correlation
     between the quick test and the full exam over the programs that took both

Model calls run `workers` at a time (the first of each batch alone, so the rest read the prompt cache it writes); GPU
timings stay one at a time. Only programs that finish the full exam join the tree: a round's dozens of cheap failures
would otherwise close the parent. Facts only in every prompt; the instructions below say what to return, never what
to try."""
from __future__ import annotations

import difflib
import json
import math
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

from kernels.search import diff, tune
from kernels.search.llm import CACHE_SPLIT
from kernels.search.tree import Node, Tree, fingerprint

SAME = 0.92                  # cosine above which two hypotheses are one idea (as the archive's DISTINCT)
SLICES = ("measurements", "winners", "compiler", "ptx")
RACE = 0.05                  # quick scores within this fraction of the cutoff are timed once more before the cut

MERGE = """These hypotheses about one GPU kernel program say the same thing in different words. Answer with one JSON
object with the keys "bottleneck", "optimisation_method", "modification_plan" and "mechanism" (a list, possibly empty)
that keeps every specific detail any of them gives, and nothing else."""
PAST = """Each item below is a proposed change to the program shown, with archived records of similar changes and their
measured outcomes (none beat its parent). Answer with one JSON array and nothing else, one object per item:
{"index": <i>, "keep": true|false, "differs": "<what in this program or its measurements differs from the archived
records' context, max 30 words; empty when nothing does>"}. "keep" is true only when something differs."""


@dataclass
class Models:
    judge: Callable[[str, str], tuple[str, float]]          # the strong model that names hypotheses
    cheap: Callable[[str, str], tuple[str, float]]          # merges duplicates
    wide: Callable[[str, str], tuple[str, float]]           # the wide coder
    strong: Callable[[str, str], tuple[str, float]]         # repairs failures, rewrites the programs kept
    judges: tuple = ()                        # the hypothesis-naming calls cycle through these (empty: `judge` only)


@dataclass
class Funnel:
    models: Models
    stage: Callable[[str, int], dict]       # (code, stage) -> {metrics, artifacts}; 1, 2, 3 (ncu), 4 (quick), 6/7 (tune)
    archive: list = field(default_factory=list)            # kernels.archive entries for this task
    task: str | None = None
    library: list = field(default_factory=list)            # kernels.archive entries for every task (the memory's)
    memory: tuple = ("", "")                 # (general table, task table) in every judge call, kernels/search/memory.py
    embedder: object | None = None
    ptx: str | None = None                   # the PTX comparison's feature cards, when built
    per_call: int = 20
    keep: int = 6
    parallel: int = 4
    repairs: int = 3
    ncu: bool = False
    precheck: bool = False                   # the no-GPU build check (jailed stage 5) before stage 1
    usd_cap: float | None = None             # model spend at which a round stops, mid-round included
    gpu_s_cap: float | None = None           # GPU seconds at which a round stops (`gpu_used` reads the spend)
    gpu_used: Callable[[], float] | None = None
    workers: int = 8                         # model calls at once
    tune: bool = False                       # each program's grid tuned before its quick timing and its full exam
    size_expr: str | None = None             # the task's call size: the full exam's tuning may dispatch on it
    deadline: float | None = None            # time.monotonic() at which a round stops (the search's wall-clock cap)
    spent: float = 0.0
    calls: int = 0
    judge_calls: int = 0                     # hypothesis-naming calls so far: picks the judge from the mix
    tried: set = field(default_factory=set)  # every program written this run, judged or not: never written again
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def _ask(self, model, system: str, user: str) -> str:
        with self._lock:
            if self.usd_cap is not None and self.spent >= self.usd_cap:
                raise Spent("model budget")
        text, usd = model(system, user)
        with self._lock:
            self.spent, self.calls = self.spent + usd, self.calls + 1
        return text or ""

    def _judge(self):
        """The next hypothesis-naming model from the mix."""
        mix = self.models.judges or (self.models.judge,)
        with self._lock:
            m = mix[self.judge_calls % len(mix)]
            self.judge_calls += 1
        return m

    def run(self, code: str, stage: int) -> dict:
        """One evaluator stage, refused once the GPU budget or the wall-clock is spent (the precheck uses no GPU)."""
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise Spent("wall-clock")
        if stage != PRECHECK and self.gpu_s_cap is not None and self.gpu_used and self.gpu_used() >= self.gpu_s_cap:
            raise Spent("GPU budget")
        return self.stage(code, stage)

    def measure(self, code: str, stage: int) -> tuple[str, dict]:
        """A program's first measurement at a tier (4: the quick timing, 2: the full exam), its grid tuned on that tier's
        shapes first (kernels/search/tune.py): (the program measured, its result, with every grid point's row)."""
        rows = []
        if self.tune:
            code, rows = tune.tuned(self.run, code, quick=stage == 4, size_expr=self.size_expr)
        r = self.run(code, stage)
        if rows:
            r = {**r, "artifacts": {**r["artifacts"], "tune": json.dumps(rows)}}
        return code, r


class Spent(Exception):
    """A budget ran out mid-round: the round ends with what it has."""


def pmap(f: Funnel, fn: Callable, items) -> list:
    """`fn` over `items`, `f.workers` at a time, in order; the first alone, so the calls after it read the prompt cache
    it writes (a cache entry is readable only once its request has started)."""
    items = list(items)
    if len(items) <= 1 or f.workers <= 1:
        return [fn(x) for x in items]
    first = fn(items[0])
    with ThreadPoolExecutor(max_workers=f.workers) as ex:
        return [first, *ex.map(fn, items[1:])]


def slices(f: Funnel, tree: Tree, parent: Node) -> dict[str, str]:
    """The facts each generator call gets beyond the program, its measurements and the memory; a slice with no facts is
    skipped."""
    from kernels.search.loop import diff_from_root
    out = {"measurements": ""}
    mine = {n.id for n in tree.lineage(parent)}
    wins = sorted((n for n in tree.nodes.values() if n.id not in mine and n.parent and n.score > 0
                   and n.score > tree.nodes[n.parent].score), key=lambda n: -n.score)[:4]
    if wins:
        out["winners"] = "# Measured winners elsewhere in the tree (each beat its parent), as diffs from the start\n" + \
            "".join(f"## {n.id} (score {n.score:.4f})\n```diff\n{diff_from_root(tree, n)}```\n" for n in wins)
    if parent.artifacts.get("compile_report"):
        out["compiler"] = f"# The compiler's report for this program\n{parent.artifacts['compile_report']}"
    if f.ptx:
        out["ptx"] = f"# The PTX comparison: this program and other implementations, as feature cards\n{f.ptx}"
    return out


def judging(f: Funnel, system: str) -> str:
    """A judge call's system prompt: the setting with the general table, then the task table and the judge's role, each
    part cached on its own (the general table changes least, so it comes first)."""
    from kernels.search.loop import JUDGE
    general, mine = f.memory
    return system + (f"\n\n{general}" if general else "") + CACHE_SPLIT + "\n\n" + (f"{mine}\n\n" if mine else "") + JUDGE


def generate(f: Funnel, system: str, base: str, facts: dict[str, str], n: int | None = None,
             only: str | None = None, mode: str = "optimization", note: str = "", **kw) -> list[dict]:
    """One judge call per slice naming `per_call` hypotheses, or (`only`) one call on that slice naming `n`; `mode`
    (loop.MODES, formatted with `kw`) is the request that ends the prompt, after the slice and `note`."""
    from kernels.search.loop import MODES, hypotheses
    n = n or f.per_call
    spec, ask = judging(f, system), MODES[mode].format(n=n, **kw)
    names = [s for s in SLICES if s in facts and (only is None or s == only)]
    texts = pmap(f, lambda s: f._ask(f._judge(), spec, base + CACHE_SPLIT + "".join(
        f"\n\n{t}" for t in (facts[s], note, ask) if t)), names)
    return [{**h, "slice": s} for s, text in zip(names, texts) for h in hypotheses(text, n)]


def _text(h: dict) -> str:
    return "\n".join(str(h.get(k, "")) for k in ("bottleneck", "optimisation_method", "modification_plan")).strip() \
        or json.dumps(h, sort_keys=True)


def merge(f: Funnel, ideas: list[dict]) -> list[dict]:
    """Clusters of the same idea (cosine >= SAME, else identical text) become one, merged by the cheap model; a merged
    idea keeps the first member's parent and slice."""
    from kernels.search.loop import hypotheses
    from lab.search import _cosine
    texts = [_text(h) for h in ideas]
    vecs = f.embedder.passages(texts) if f.embedder and texts else None
    groups: list[list[int]] = []
    for i in range(len(ideas)):
        g = next((g for g in groups if texts[g[0]] == texts[i] or (vecs and _cosine(vecs[g[0]], vecs[i]) >= SAME)), None)
        (g.append(i) if g else groups.append([i]))

    def one(g: list[int]) -> dict:
        if len(g) == 1:
            return {**ideas[g[0]], "merged_from": 1}
        got = hypotheses(f._ask(f.models.cheap, MERGE, "\n\n".join(json.dumps(ideas[i]) for i in g)), 1)
        tags = {k: ideas[g[0]][k] for k in ("slice", "parent_id", "idea_id", "type", "with") if k in ideas[g[0]]}
        return {**(got[0] if got else ideas[g[0]]), **tags, "merged_from": len(g)}
    return pmap(f, one, groups)


def past_failures(f: Funnel, system: str, base: str, ideas: list[dict]) -> tuple[list[dict], list[dict]]:
    """(kept, dropped): ideas like archived ones that never beat their parent need the judge to name what differs."""
    from kernels import archive
    if not f.archive:
        return ideas, []
    flagged = {}
    vectors = f.embedder is not None and all(e.vector is not None for e in f.archive)   # else matches ranks by BM25
    for i, h in enumerate(ideas):
        like = [e for s, e in archive.matches(f.archive, _text(h), f.task, k=3, embedder=f.embedder)
                if (s >= SAME if vectors else e.text == _text(h))            # no embeddings: the same text only
                and not e.ever_improved() and e.ever_failed()]               # measured, and never beat its parent
        if like:
            flagged[i] = like
    if not flagged:
        return ideas, []
    items = "\n\n".join(f"## Item {i}\n{json.dumps({k: v for k, v in ideas[i].items() if k != 'slice'})}\n"
                        + "\n".join(json.dumps(archive.facts(e)) for e in like) for i, like in flagged.items())
    try:
        verdicts = json.loads(_json(f._ask(f.models.judge, system + CACHE_SPLIT + "\n\n" + PAST, base + CACHE_SPLIT + "\n\n" + items)))
    except ValueError:
        verdicts = []
    keep = {}
    for v in verdicts if isinstance(verdicts, list) else []:
        try:
            if isinstance(v, dict) and v.get("keep") is True:
                keep[int(v["index"])] = str(v.get("differs", ""))
        except (KeyError, TypeError, ValueError):
            continue
    kept, dropped = [], []
    for i, h in enumerate(ideas):
        if i not in flagged:
            kept.append(h)
        elif i in keep:
            kept.append({**h, "differs_from_archive": keep[i]})
        else:
            dropped.append({**h, "dropped": "like archived ideas that never beat their parent; nothing named as different"})
    return kept, dropped


def _json(text: str) -> str:
    a, b = text.find("["), text.rfind("]")
    return text[a:b + 1] if a >= 0 and b > a else "[]"


def write(f: Funnel, model, system: str, prompt: str, parent_code: str, seen: set) -> tuple[str | None, str]:
    """One coder call applied: (code, None) or (None, why)."""
    from kernels.search.loop import FORMAT
    text = f._ask(model, system + CACHE_SPLIT + "\n\n" + FORMAT, prompt)
    code, why = diff.apply(parent_code, text) if text else (None, "the model returned no text")
    if code is not None and fingerprint(code) in seen:
        code, why = None, "identical to a program already judged (comments and whitespace aside)"
    return code, why


def written(f: Funnel, results: list[tuple[str | None, str]], seen: set) -> list[tuple[str | None, str]]:
    """Coder results from one parallel batch, a program written twice in it refused the second time; the rest are
    added to `seen` and `f.tried`."""
    out = []
    for code, why in results:
        if code is not None and fingerprint(code) in seen:
            code, why = None, "identical to a program already judged (comments and whitespace aside)"
        if code is not None:
            seen.add(fingerprint(code))
            f.tried.add(fingerprint(code))
        out.append((code, why))
    return out


PRECHECK = 5                 # kernels.evolve.jailed.PRECHECK_STAGE


def correct(f: Funnel, codes: list[str], stage: int = 1) -> list[dict]:
    """Stage 1 (or the precheck) for each program, `parallel` jails at a time (neither needs the GPU to itself)."""
    with ThreadPoolExecutor(max_workers=max(1, f.parallel)) as ex:
        return list(ex.map(lambda c: f.run(c, stage), codes))


def counters(artifacts: dict) -> dict[str, float]:
    """Each ncu metric summed over the profiled shapes and their top kernels."""
    out: dict[str, float] = {}
    try:
        d = json.loads(artifacts.get("ncu") or "{}")
    except ValueError:
        return out
    for p in d.get("shapes") or [d]:
        for k in p.get("kernels") or []:
            for m, v in k.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    out[m] = out.get(m, 0.0) + v
    return out


def mechanism(idea: dict, parent: Node, child: Node) -> list[dict]:
    """The idea's predicted counter changes beside the measured ones (child against parent), or why one is missing."""
    before, after, out = counters(parent.artifacts), counters(child.artifacts), []
    for p in idea.get("mechanism") or []:
        if not isinstance(p, dict) or not isinstance(p.get("counter"), str):
            continue
        b, a = before.get(p["counter"]), after.get(p["counter"])
        out.append({**p, "actual_change_pct": (a - b) / b * 100 if b and a is not None else None}
                   | ({} if b and a is not None else {"missing": "no ncu value for the parent or the child"}))
    return out


def spearman(xs: list[float], ys: list[float]) -> float | None:
    """Spearman's rho between two rankings (ties get their mean rank); None under three pairs or with no spread."""
    if len(xs) < 3:
        return None

    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r, i = [0.0] * len(v), 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in order[i:j + 1]:
                r[k] = (i + j) / 2 + 1
            i = j + 1
        return r
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sx, sy = math.sqrt(sum((a - mx) ** 2 for a in rx)), math.sqrt(sum((b - my) ** 2 for b in ry))
    return sum((a - mx) * (b - my) for a, b in zip(rx, ry)) / (sx * sy) if sx and sy else None


def _score(r: dict) -> float:
    return float(r["metrics"].get("combined_score") or 0.0)


def race(f: Funnel, quick: list, keep: int) -> list:
    """`quick`: (item..., result) sorted fastest first. Programs whose quick score is within RACE of the cutoff (the
    slowest kept and the fastest dropped) are timed once more and the list re-sorted by each one's mean."""
    if len(quick) <= keep or keep <= 0:
        return quick
    cut = (_score(quick[keep - 1][-1]) + _score(quick[keep][-1])) / 2
    if cut <= 0:
        return quick
    out = []
    for row in quick:
        s = _score(row[-1])
        if abs(s - cut) <= RACE * cut and s > 0:
            again = f.run(row[-2], 4)
            m = {**row[-1]["metrics"], "combined_score": (s + _score(again)) / 2, "raced": [s, _score(again)]}
            row = (*row[:-1], {**row[-1], "metrics": m})
        out.append(row)
    out.sort(key=lambda x: -_score(x[-1]))
    return out


def profile_parents(f: Funnel, parents: list[Node]) -> None:
    """ncu counters (stage 3) for each parent that has none yet, into its artifacts; once the host refuses them, they
    are not asked for again."""
    from kernels.search.loop import REFUSED
    for p in parents:
        if not f.ncu or any(k in p.artifacts for k in ("ncu", "ncu_missing", "ncu_error")):
            continue
        a = f.run(p.code, 3)["artifacts"]
        if REFUSED in json.dumps(a):
            f.ncu = False
            a = {"ncu_missing": f"ncu counters unavailable on this host ({REFUSED}); not collected this run"}
        p.artifacts.update(a)


def expand(f: Funnel, tree: Tree, parent: Node, system: str, it: int, keep_node: Callable[[Node, float], None],
           log: Callable[[dict], None]) -> list[Node]:
    """One round from one parent (an MCTS expansion): one judge call per slice, `per_call` ideas each, the fastest
    `keep` to the full exam."""
    return generation(f, tree, [parent], system, it, keep_node, log, quota=f.per_call, keep=f.keep, rotate=False)


@dataclass
class Slot:
    """One judge call's assignment in a generation: name `n` ideas of type `kind` from `parent`."""
    parent: Node
    kind: str = "refine"         # start, refine, persist, fresh or crossover (kernels/search/generations.py)
    n: int = 1
    other: Node | None = None    # crossover: the other parent, shown as its diff from the start


def generation(f: Funnel, tree: Tree, parents: list[Node], system: str, it: int,
               keep_node: Callable[[Node, float], None], log: Callable[[dict], None], quota: int, keep: int,
               rotate: bool = True, crossover: int = 0, spread: bool = False, slots: list[Slot] | None = None,
               keep_by: dict[str, int] | None = None, extra: dict | None = None) -> list[Node]:
    """One round: each slot's judge call names its ideas. Without `slots`, each of `parents` is a refine slot of `quota`
    ideas (with `rotate`, one call on one slice, rotated across the parents; else one call per slice, `quota` each, or
    with `spread` the parent's `quota` split over its slices) and `crossover` slots of one idea each combine a parent
    with another parent's diff. A refine (or start) slot reads its slices as above; persist, fresh and crossover slots
    read the measurements. The fastest `keep` of the whole round (with `keep_by`, of each idea type) go to the full
    exam and join the tree under their own parent, each child's `kind` its idea's type. Every idea carries an
    `idea_id` and its `type`, and its log row the diff its program made. A budget that runs out mid-round (`Spent`)
    ends the round with what it has; the log says where. `extra` goes into the round's log row."""
    from kernels.search.loop import diff_from_root, prompt
    spent0, fates, children = f.spent, [], []
    counts = {"named": 0, "distinct": 0, "after_archive": 0, "written": 0, "correct": 0, "rewritten": 0}
    stopped, pairs, code_of = None, [], {}       # code_of: each written idea's latest program, for the log's diff
    if slots is None:
        slots = [Slot(p, "refine", quota) for p in parents] + [
            Slot(parents[k % len(parents)], "crossover", 1, parents[(k + 1) % len(parents)])
            for k in range(crossover if len(parents) > 1 else 0)]
    by_id = {s.parent.id: s.parent for s in slots}
    parents = list(by_id.values())

    def key(h):
        return h.get("idea_id")
    try:
        profile_parents(f, parents)
        need = [p for p in parents if f.precheck and "compile_report" not in p.artifacts]
        for p, r in zip(need, correct(f, [p.code for p in need], PRECHECK)):    # for the compiler slice
            if r["artifacts"].get("compile_report"):
                p.artifacts["compile_report"] = r["artifacts"]["compile_report"]
        bases = {p.id: prompt(tree, p, coder=False) for p in parents}
        refine = [s for s in slots if s.kind in ("start", "refine")]

        def name(s: Slot) -> list[dict]:
            p = s.parent
            if s.kind == "crossover":
                got = [{**h, "slice": "crossover", "with": s.other.id}
                       for h in _crossover(f, system, bases[p.id], tree, s.other, diff_from_root)]
            elif s.kind == "persist":            # the program's last change, to be taken further
                before = tree.nodes[p.parent].score if p.parent else 0.0
                got = generate(f, system, bases[p.id], {"measurements": ""}, n=s.n, mode="persistence",
                               note=f"# Its last change (from the judge)\n{p.judge}", before=before, after=p.score)
            elif s.kind == "fresh":
                got = generate(f, system, bases[p.id], {"measurements": ""}, n=s.n, mode="fresh")
            else:
                facts = slices(f, tree, p)
                if rotate:                       # one slice per call, every slice read across the generation
                    names = [x for x in SLICES if x in facts]
                    i = refine.index(s)
                    got = generate(f, system, bases[p.id], facts, n=s.n,
                                   only=names[(it * len(refine) + i) % len(names)])
                else:
                    got = generate(f, system, bases[p.id], facts,
                                   n=max(1, math.ceil(s.n / len(facts))) if spread else s.n)
            return [{**h, "parent_id": p.id, "type": s.kind, "idea_id": uuid.uuid4().hex[:8]} for h in got]
        ideas = [h for got in pmap(f, name, slots) for h in got]
        counts["named"] = len(ideas)
        ideas = merge(f, ideas)
        judged = pmap(f, lambda p: past_failures(f, system, bases[p.id], [h for h in ideas if h.get("parent_id") == p.id]),
                      parents)
        ideas, dropped = [h for k, _ in judged for h in k], [h for _, d in judged for h in d]
        counts["distinct"], counts["after_archive"] = len(ideas) + len(dropped), len(ideas)
        fates += [{"idea": h, "fate": h["dropped"]} for h in dropped]
        seen = set(tree.seen) | f.tried

        def plan_of(h):
            return json.dumps({k: v for k, v in h.items() if k in ("bottleneck", "optimisation_method",
                                                                   "modification_plan", "mechanism")})
        drafts = written(f, pmap(f, lambda h: write(f, f.models.wide, system, prompt(tree, by_id[h["parent_id"]],
                                                                                     plan_of(h)),
                                                    by_id[h["parent_id"]].code, set(seen)), ideas), seen)
        wrote = []
        for h, (code, why) in zip(ideas, drafts):
            if code is None:
                fates.append({"idea": h, "fate": f"not applied: {why}"})
            else:
                wrote.append((h, plan_of(h), code))
                code_of[key(h)] = code
        counts["written"] = len(wrote)
        reports = {}
        if f.precheck and wrote:                 # what fails to build never takes GPU time
            built = []
            for (h, plan, code), r in zip(wrote, correct(f, [c for _, _, c in wrote], PRECHECK)):
                if r["artifacts"].get("compile_report"):
                    reports[code] = r["artifacts"]["compile_report"]
                if r["metrics"].get("correct") == 0.0:
                    fates.append({"idea": h, "fate": "precheck: " + r["artifacts"].get("precheck", "")[:300]})
                    continue
                built.append((h, plan, code))
            wrote = built
        survivors, failed = [], []
        for (h, plan, code), r in zip(wrote, correct(f, [c for _, _, c in wrote])):
            (survivors if r["metrics"].get("correct") == 1.0 else failed).append((h, plan, code, r))
        counts["correct"] = len(survivors)
        for h, plan, code, r in failed[f.repairs:]:
            fates.append({"idea": h, "fate": "failed correctness", "error": str(r["artifacts"])[:300]})
        queue = failed[:f.repairs]                # the repair queue: one fix by the strong coder each
        fixes = written(f, pmap(f, lambda x: write(
            f, f.models.strong, system, prompt(tree, by_id[x[0]["parent_id"]], x[1] + "\n\n# A first version failed "
                                                "correctness\n" + json.dumps(x[3]["artifacts"])[:3000]),
            by_id[x[0]["parent_id"]].code, set(seen)), queue), seen)
        checked = correct(f, [c for c, _ in fixes if c])
        results = iter(checked)
        for (h, plan, code, r), (c2, why) in zip(queue, fixes):
            r2 = next(results) if c2 else {"metrics": {"correct": 0.0}, "artifacts": {"error": why}}
            if c2 and r2["metrics"].get("correct") == 1.0:
                survivors.append((h, plan, c2, r2))
                code_of[key(h)] = c2
            else:
                fates.append({"idea": h, "fate": "failed correctness, and its repair",
                              "error": str(r["artifacts"])[:300]})
        quick = [(h, plan, *f.measure(c, 4)) for h, plan, c, _ in survivors]   # GPU timings: one at a time
        for h, _, c, _ in quick:
            code_of[key(h)] = c
        quick.sort(key=lambda x: -_score(x[3]))
        groups: dict = {}                        # with keep_by, each idea type is cut on its own
        for row in quick:
            groups.setdefault(row[0].get("type") if keep_by else None, []).append(row)
        finalists = []
        for t, rows in groups.items():
            k, n = (keep_by.get(t, 0) if keep_by else keep), 0
            for h, plan, code, q in race(f, rows, k):
                if _score(q) <= 0:
                    fates.append({"idea": h, "fate": "quick speed test: scored 0", "quick": q["metrics"]})
                elif n >= k:
                    fates.append({"idea": h, "fate": "quick speed test: not among the fastest", "quick": q["metrics"]})
                else:
                    finalists.append((h, plan, code, q))
                    n += 1
        finalists = rewrite(f, tree, system, by_id, finalists, seen, reports, counts)
        for h, plan, code, q, how in finalists:
            p = by_id[h["parent_id"]]
            code, r = f.measure(code, 2)
            code_of[key(h)] = code
            if reports.get(code):
                r["artifacts"] = {**r["artifacts"], "compile_report": reports[code]}
            child = Node(uuid.uuid4().hex[:8], p.id, code, r["metrics"], r["artifacts"], it, "", plan, "optimization",
                         h.get("type", ""))
            keep_node(child, 0.0)
            children.append(child)
            pairs.append((_score(q), child.score))
            fates.append({"idea": h, "fate": "full exam", "child_id": child.id, "child_score": child.score,
                          "improved": child.score > p.score, "quick": q["metrics"], "written_by": how,
                          **({"mechanism": mechanism(h, p, child)} if h.get("mechanism") else {})})
    except Spent as e:
        stopped = str(e)
    for x in fates:                              # each written idea's program as a diff from its parent
        c, p = code_of.get(key(x["idea"])), by_id.get(x["idea"].get("parent_id"))
        if c and p:
            x["diff"] = "".join(difflib.unified_diff(p.code.splitlines(True), c.splitlines(True), n=2))[:4000]
    one = parents[0] if len(parents) == 1 else None
    log({"iteration": it, "parent_id": one.id if one else None, "parent_score": one.score if one else None,
         "parents": [{"id": p.id, "score": p.score} for p in by_id.values()], **counts, "examined": len(children),
         "improved": sum(c.score > by_id[c.parent].score for c in children),
         "quick_vs_full_spearman": spearman([a for a, _ in pairs], [b for _, b in pairs]) if children else None,
         "usd": round(f.spent - spent0, 4), **(extra or {}), **({"stopped": stopped} if stopped else {}),
         "ideas": fates})
    return children


def rewrite(f: Funnel, tree: Tree, system: str, by_id: dict, finalists: list, seen: set, reports: dict,
            counts: dict) -> list:
    """The strong coder rewrites each finalist (idea, plan, draft, quick result) from its idea and the passing draft;
    the rewrite replaces the draft when it builds, passes stage 1 and its quick time is faster. Returns (idea, plan,
    code, quick result, "rewrite" | "draft (why the rewrite was not kept)") for each."""
    from kernels.search.loop import prompt

    def ask(x):
        h, plan, code, _ = x
        p = by_id[h["parent_id"]]
        ref = plan + "\n\n# A version that passed correctness, as a diff from this program\n" + \
            "".join(difflib.unified_diff(p.code.splitlines(True), code.splitlines(True), n=2))[:6000]
        return write(f, f.models.strong, system, prompt(tree, p, ref), p.code, set(seen))
    asked = written(f, pmap(f, ask, finalists), seen)
    news = [c for c, _ in asked]
    ok = {}
    todo = [c for c in news if c]
    if f.precheck and todo:
        for c, r in zip(todo, correct(f, todo, PRECHECK)):
            if r["artifacts"].get("compile_report"):
                reports[c] = r["artifacts"]["compile_report"]
            if r["metrics"].get("correct") == 0.0:
                ok[c] = False
    todo = [c for c in todo if ok.get(c) is not False]
    for c, r in zip(todo, correct(f, todo)):
        ok[c] = r["metrics"].get("correct") == 1.0
    out = []
    for (h, plan, code, q), (c2, why) in zip(finalists, asked):
        if not c2:
            out.append((h, plan, code, q, f"draft (rewrite not applied: {why[:200]})"))
        elif not ok.get(c2):
            out.append((h, plan, code, q, "draft (rewrite failed " + ("the precheck)" if c2 in ok and ok[c2] is False
                                                                     and c2 not in todo else "stage 1)")))
        else:
            c2, q2 = f.measure(c2, 4)
            if _score(q2) > _score(q):
                counts["rewritten"] += 1
                out.append((h, plan, c2, q2, "rewrite"))
            else:
                out.append((h, plan, code, q, f"draft (rewrite quick {_score(q2):.4f} vs {_score(q):.4f})"))
    return out


def _crossover(f: Funnel, system: str, base: str, tree: Tree, other: Node, diff_from_root) -> list[dict]:
    """One judge call: this program beside another parent's diff from the start, naming one change that carries what
    the other does differently into this program."""
    from kernels.search.loop import MODES, hypotheses
    text = f._ask(f._judge(), judging(f, system), base + f"{CACHE_SPLIT}\n\n# Another program in this generation, "
                  f"{other.id} (score {other.score:.4f}), as a diff from the starting program\n```diff\n"
                  f"{diff_from_root(tree, other)}```\n\n" + MODES["crossover"].format(n=1))
    return hypotheses(text, 1)
