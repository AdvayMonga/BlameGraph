"""One round of the funnel (decided 2026-10-22): volume through measured tiers, every cheap tier a measurement. A round
takes one parent (an MCTS expansion, `expand`) or several (one generation of the genetic search, `generation`,
kernels/search/generations.py; decided 2026-10-26).

From the parents:
  1. generate: for one parent, one judge call per slice of the facts (the measurements; archive matches; measured
     winners elsewhere in the tree; the compiler's report; the PTX comparison); for several, one call per parent with
     the slices rotated across the generation's parents, so every call reads one slice and every slice is read. Each
     call names the parent's quota of hypotheses. The judge calls cycle through `Models.judges` (3 Opus : 1 Sonnet
     by default), for diversity.
  2. dedup / merge: hypotheses within `SAME` cosine are one idea, merged by a cheap model keeping every detail
  3. past failures: an idea like an archived one that never beat its parent goes back to the judge with those records
     and survives only if the judge names what differs now
  4. the wide coder writes each idea; edits that do not apply or repeat a judged program are dropped; with
     `precheck`, what fails to build in a jail with no GPU is dropped too, and the compiler's report is kept
  5. correctness on the GPU (stage 1), `parallel` at a time; failures go to the repair queue
  6. the strong coder rewrites each survivor from the idea and the passing version, checked again
  7. the quick speed test (the most-called shape only); programs within `RACE` of the cutoff are timed once more and
     ranked by their mean (successive halving's racing: repeats only where one timing cannot separate them); the
     fastest `keep` go on
  8. the full exam (stage 2, then ncu) and the tree; every idea and how far it got go to the log, with the rank
     correlation between the quick test and the full exam over the programs that took both

Only programs that finish the full exam join the tree: a round's dozens of cheap failures would otherwise close the
parent. Facts only in every prompt; the instructions below say what to return, never what to try."""
from __future__ import annotations

import difflib
import json
import math
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

from kernels.search import diff
from kernels.search.tree import Node, Tree, fingerprint

SAME = 0.92                  # cosine above which two hypotheses are one idea (as the archive's DISTINCT)
SLICES = ("measurements", "archive", "winners", "compiler", "ptx")
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
    strong: Callable[[str, str], tuple[str, float]]         # rewrites survivors
    judges: tuple = ()                        # the hypothesis-naming calls cycle through these (empty: `judge` only)


@dataclass
class Funnel:
    models: Models
    stage: Callable[[str, int], dict]       # (code, stage) -> {metrics, artifacts}; stage 1, 2, 3 (ncu), 4 (quick)
    archive: list = field(default_factory=list)            # kernels.archive entries for this task
    task: str | None = None
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
    spent: float = 0.0
    calls: int = 0
    judge_calls: int = 0                     # hypothesis-naming calls so far: picks the judge from the mix
    tried: set = field(default_factory=set)  # every program written this run, judged or not: never written again

    def _ask(self, model, system: str, user: str) -> str:
        if self.usd_cap is not None and self.spent >= self.usd_cap:
            raise Spent("model budget")
        text, usd = model(system, user)
        self.spent, self.calls = self.spent + usd, self.calls + 1
        return text or ""

    def _judge(self):
        """The next hypothesis-naming model from the mix."""
        mix = self.models.judges or (self.models.judge,)
        m = mix[self.judge_calls % len(mix)]
        self.judge_calls += 1
        return m

    def run(self, code: str, stage: int) -> dict:
        """One evaluator stage, refused once the GPU budget is spent (the precheck uses no GPU)."""
        if stage != PRECHECK and self.gpu_s_cap is not None and self.gpu_used and self.gpu_used() >= self.gpu_s_cap:
            raise Spent("GPU budget")
        return self.stage(code, stage)


class Spent(Exception):
    """A budget ran out mid-round: the round ends with what it has."""


def slices(f: Funnel, tree: Tree, parent: Node) -> dict[str, str]:
    """The facts each generator call gets beyond the program and its measurements; a slice with no facts is skipped."""
    from kernels import archive
    from kernels.search.loop import diff_from_root
    out = {"measurements": ""}
    q = parent.judge or (f.task or "")
    found = archive.matches(f.archive, str(q), f.task, k=8, embedder=f.embedder) if f.archive else []
    head = "# Similar ideas in the archive, with their measured outcomes"
    if f.archive and not found:                  # nothing like this program's idea: the task's best-measured ideas
        mine = [e for e in f.archive if e.task == f.task or f.task is None]
        found = [(None, e) for e in sorted(mine, key=lambda e: -((e.best() and e.best().child_score) or 0))[:8]]
        head = "# The task's best-measured ideas in the archive, with their measured outcomes"
    if found:
        out["archive"] = head + "\n" + "\n".join(json.dumps(archive.facts(e, s)) for s, e in found)
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


def generate(f: Funnel, system: str, base: str, facts: dict[str, str], n: int | None = None,
             only: str | None = None) -> list[dict]:
    """One judge call per slice naming `per_call` hypotheses, or (`only`) one call on that slice naming `n`."""
    from kernels.search.loop import JUDGE, MODES, hypotheses
    n = n or f.per_call
    spec = system + "\n\n" + JUDGE.format(mode_spec=MODES["optimization"].format(n=n))
    ideas = []
    for name in SLICES:
        if name in facts and (only is None or name == only):
            text = f._ask(f._judge(), spec, base + (f"\n\n{facts[name]}" if facts[name] else ""))
            ideas += [{**h, "slice": name} for h in hypotheses(text, n)]
    return ideas


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
    out = []
    for g in groups:
        if len(g) == 1:
            out.append({**ideas[g[0]], "merged_from": 1})
            continue
        got = hypotheses(f._ask(f.models.cheap, MERGE, "\n\n".join(json.dumps(ideas[i]) for i in g)), 1)
        tags = {k: ideas[g[0]][k] for k in ("slice", "parent_id") if k in ideas[g[0]]}
        out.append({**(got[0] if got else ideas[g[0]]), **tags, "merged_from": len(g)})
    return out


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
        verdicts = json.loads(_json(f._ask(f.models.judge, system + "\n\n" + PAST, base + "\n\n" + items)))
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
    text = f._ask(model, system + "\n\n" + FORMAT, prompt)
    code, why = diff.apply(parent_code, text) if text else (None, "the model returned no text")
    if code is not None and fingerprint(code) in seen:
        code, why = None, "identical to a program already judged (comments and whitespace aside)"
    return code, why


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


def expand(f: Funnel, tree: Tree, parent: Node, system: str, it: int, keep_node: Callable[[Node, float], None],
           log: Callable[[dict], None]) -> list[Node]:
    """One round from one parent (an MCTS expansion): one judge call per slice, `per_call` ideas each, the fastest
    `keep` to the full exam."""
    return generation(f, tree, [parent], system, it, keep_node, log, quota=f.per_call, keep=f.keep, rotate=False)


def generation(f: Funnel, tree: Tree, parents: list[Node], system: str, it: int,
               keep_node: Callable[[Node, float], None], log: Callable[[dict], None], quota: int, keep: int,
               rotate: bool = True, crossover: int = 0, spread: bool = False) -> list[Node]:
    """One round from `parents`: each parent's judge calls name `quota` ideas (with `rotate`, one call on one slice,
    rotated across the parents; else one call per slice, `quota` each, or with `spread` the parent's `quota` split over
    its slices), `crossover` further ideas each combine a parent with another parent's diff; the fastest `keep` of the
    whole round go to the full exam and join the tree under their own parent. A budget that runs out mid-round
    (`Spent`) ends the round with what it has; the log says where."""
    from kernels.search.loop import REFUSED, diff_from_root, prompt
    spent0, fates, children = f.spent, [], []
    counts = {"named": 0, "distinct": 0, "after_archive": 0, "written": 0, "correct": 0}
    stopped = None
    by_id = {p.id: p for p in parents}
    try:
        bases = {}
        for p in parents:
            if f.precheck and "compile_report" not in p.artifacts:   # the parent's own report, for the compiler slice
                a = f.run(p.code, PRECHECK)["artifacts"]
                if a.get("compile_report"):
                    p.artifacts["compile_report"] = a["compile_report"]
            bases[p.id] = prompt(tree, p, coder=False)
        ideas = []
        for i, p in enumerate(parents):
            facts = slices(f, tree, p)
            if rotate:                           # one slice per call, every slice read across the generation
                names = [s for s in SLICES if s in facts]
                got = generate(f, system, bases[p.id], facts, n=quota, only=names[(it * len(parents) + i) % len(names)])
            else:
                got = generate(f, system, bases[p.id], facts, n=max(1, math.ceil(quota / len(facts))) if spread else quota)
            ideas += [{**h, "parent_id": p.id} for h in got]
        for k in range(crossover if len(parents) > 1 else 0):     # a parent shown another parent's diff
            p, o = parents[k % len(parents)], parents[(k + 1) % len(parents)]
            ideas += [{**h, "parent_id": p.id, "slice": "crossover", "with": o.id}
                      for h in _crossover(f, system, bases[p.id], tree, o, diff_from_root)]
        counts["named"] = len(ideas)
        ideas = merge(f, ideas)
        kept, dropped = [], []
        for p in parents:
            mine = [h for h in ideas if h.get("parent_id") == p.id]
            k2, d2 = past_failures(f, system, bases[p.id], mine)
            kept += k2
            dropped += d2
        ideas = kept
        counts["distinct"], counts["after_archive"] = len(ideas) + len(dropped), len(ideas)
        fates += [{"idea": h, "fate": h["dropped"]} for h in dropped]
        seen = set(tree.seen) | f.tried
        written = []
        for h in ideas:
            p = by_id[h["parent_id"]]
            plan = json.dumps({k: v for k, v in h.items() if k in ("bottleneck", "optimisation_method",
                                                                   "modification_plan", "mechanism")})
            code, why = write(f, f.models.wide, system, prompt(tree, p, plan), p.code, seen)
            if code is None:
                fates.append({"idea": h, "fate": f"not applied: {why}"})
                continue
            seen.add(fingerprint(code))
            f.tried.add(fingerprint(code))
            written.append((h, plan, code))
        counts["written"] = len(written)
        reports = {}
        if f.precheck and written:               # what fails to build never takes GPU time
            built = []
            for (h, plan, code), r in zip(written, correct(f, [c for _, _, c in written], PRECHECK)):
                if r["metrics"].get("correct") == 0.0:
                    fates.append({"idea": h, "fate": "precheck: " + r["artifacts"].get("precheck", "")[:300]})
                    continue
                reports[code] = r["artifacts"].get("compile_report")
                built.append((h, plan, code))
            written = built
        survivors, failed = [], []
        for (h, plan, code), r in zip(written, correct(f, [c for _, _, c in written])):
            (survivors if r["metrics"].get("correct") == 1.0 else failed).append((h, plan, code, r))
        counts["correct"] = len(survivors)
        for h, plan, code, r in failed[f.repairs:]:
            fates.append({"idea": h, "fate": "failed correctness", "error": str(r["artifacts"])[:300]})
        for h, plan, code, r in failed[:f.repairs]:          # the repair queue: one fix by the strong coder each
            p = by_id[h["parent_id"]]
            fix = plan + "\n\n# A first version failed correctness\n" + json.dumps(r["artifacts"])[:3000]
            c2, why = write(f, f.models.strong, system, prompt(tree, p, fix), p.code, seen)
            if c2:
                seen.add(fingerprint(c2))
                f.tried.add(fingerprint(c2))
            r2 = f.run(c2, 1) if c2 else {"metrics": {"correct": 0.0}, "artifacts": {"error": why}}
            if c2 and r2["metrics"].get("correct") == 1.0:
                survivors.append((h, plan, c2, r2))
            else:
                fates.append({"idea": h, "fate": "failed correctness, and its repair",
                              "error": str(r["artifacts"])[:300]})
        polished = []
        for h, plan, code, r in survivors:                    # the strong coder rewrites from the idea + passing version
            p = by_id[h["parent_id"]]
            ref = plan + "\n\n# A version that passed correctness, as a diff from this program\n" + \
                "".join(difflib.unified_diff(p.code.splitlines(True), code.splitlines(True), n=2))[:6000]
            c2, _ = write(f, f.models.strong, system, prompt(tree, p, ref), p.code, seen)
            if c2:
                seen.add(fingerprint(c2))
                f.tried.add(fingerprint(c2))
            if c2 and f.run(c2, 1)["metrics"].get("correct") == 1.0:
                polished.append((h, plan, c2))
            else:
                polished.append((h, plan, code))              # the rewrite failed: the passing version goes on
        quick = []
        for h, plan, c in polished:
            quick.append((h, plan, c, f.run(c, 4)))
        quick.sort(key=lambda x: -_score(x[3]))
        quick = race(f, quick, keep)
        examined, pairs = 0, []
        for h, plan, code, q in quick:
            score = _score(q)
            if score <= 0:
                fates.append({"idea": h, "fate": "quick speed test: scored 0", "quick": q["metrics"]})
                continue
            if examined >= keep:
                fates.append({"idea": h, "fate": "quick speed test: not among the fastest", "quick": q["metrics"]})
                continue
            examined += 1
            p = by_id[h["parent_id"]]
            r = f.run(code, 2)
            if f.ncu and r["metrics"].get("correct") == 1.0 and r["metrics"].get("combined_score", 0) > 0:
                a = f.run(code, 3)["artifacts"]
                if REFUSED in json.dumps(a):                 # the host refuses counters: not asked for again
                    f.ncu = False
                    a = {"ncu_missing": f"ncu counters unavailable on this host ({REFUSED}); not collected this run"}
                r["artifacts"] = {**r["artifacts"], **a}
            if reports.get(code):
                r["artifacts"] = {**r["artifacts"], "compile_report": reports[code]}
            child = Node(uuid.uuid4().hex[:8], p.id, code, r["metrics"], r["artifacts"], it, "", plan, "optimization")
            keep_node(child, 0.0)
            children.append(child)
            pairs.append((score, child.score))
            fates.append({"idea": h, "fate": "full exam", "child_id": child.id, "child_score": child.score,
                          "improved": child.score > p.score, "quick": q["metrics"],
                          **({"mechanism": mechanism(h, p, child)} if h.get("mechanism") else {})})
    except Spent as e:
        stopped = str(e)
    one = parents[0] if len(parents) == 1 else None
    log({"iteration": it, "parent_id": one.id if one else None, "parent_score": one.score if one else None,
         "parents": [{"id": p.id, "score": p.score} for p in parents], **counts, "examined": len(children),
         "improved": sum(c.score > by_id[c.parent].score for c in children),
         "quick_vs_full_spearman": spearman([a for a, _ in pairs], [b for _, b in pairs]) if children else None,
         "usd": round(f.spent - spent0, 4), **({"stopped": stopped} if stopped else {}), "ideas": fates})
    return children


def _crossover(f: Funnel, system: str, base: str, tree: Tree, other: Node, diff_from_root) -> list[dict]:
    """One judge call: this program beside another parent's diff from the start, naming one change that carries what
    the other does differently into this program."""
    from kernels.search.loop import JUDGE, MODES, hypotheses
    spec = system + "\n\n" + JUDGE.format(mode_spec=MODES["crossover"].format(n=1))
    text = f._ask(f._judge(), spec, base + f"\n\n# Another program in this generation, {other.id} (score "
                  f"{other.score:.4f}), as a diff from the starting program\n```diff\n{diff_from_root(tree, other)}```")
    return hypotheses(text, 1)
