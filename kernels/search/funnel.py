"""One round of the search (decided 2026-10-22; one call per change and every correct program measured, decided
2026-10-30). A round takes one parent (an MCTS expansion, `expand`) or several (one generation of the genetic search,
`generation`, kernels/search/generations.py; decided 2026-10-26).

From the parents:
  0. a parent with no ncu counters yet is profiled (stage 3): counters feed only the prompts of a parent's children,
     so a program is profiled when it is first chosen as a parent (decided 2026-10-27)
  1. one model call per change: the call is shown the parent, its measurements, the starting program and the best
     programs off its lineage as full code, one slice of facts (the measurements; measured winners elsewhere in the
     tree; the compiler's report; the PTX comparison; rotated across a slot's calls) and the memory
     (kernels/search/memory.py), and answers with its hypothesis and the change as SEARCH/REPLACE blocks. A round's
     slots (a parent, an idea type and a count, kernels/search/generations.py) say how many calls each parent gets
     and of which type. The calls cycle through `Models.judges` (3 Opus : 1 Sonnet by default), for diversity.
  2. edits that do not apply, repeat a program already written, or whose change from the start is within NOVEL cosine
     of one already written (local embeddings, ShinkaEvolve's novelty rejection) are dropped; with `precheck`, what fails to build in
     a jail with no GPU is dropped too, and the compiler's report is kept
  3. correctness on the GPU (stage 1), `parallel` at a time
  4. every correct program takes the full exam (stage 2, its grid tuned first with `tune`) and joins the tree; every
     idea and how far it got go to the log

Model calls run `workers` at a time (the first of each batch alone, so the rest read the prompt cache it writes); GPU
timings stay one at a time. A program that fails correctness does not join the tree: a round's failures would
otherwise close the parent. Facts only in every prompt; the instructions say what to return, never what to try."""
from __future__ import annotations

import difflib
import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

from kernels.search import diff, tune
from kernels.search.llm import CACHE_SPLIT
from kernels.search.tree import Node, Tree, fingerprint

SLICES = ("measurements", "winners", "compiler", "ptx")
NOVEL = 0.95                 # cosine of two programs' changes from the start at which the later one is a near-duplicate
                             # (ShinkaEvolve's novelty rejection threshold)


@dataclass
class Models:
    judge: Callable[[str, str], tuple[str, float]]          # the model that makes each change
    judges: tuple = ()                        # the calls cycle through these (empty: `judge` only)


@dataclass
class Funnel:
    models: Models
    stage: Callable[[str, int], dict]       # (code, stage) -> {metrics, artifacts}; 1, 2, 3 (ncu), 5 (precheck), 6 (tune)
    task: str | None = None
    ptx: str | None = None                   # the PTX comparison's feature cards, when built
    per_call: int = 20                       # changes per MCTS expansion
    parallel: int = 4
    ncu: bool = False
    precheck: bool = False                   # the no-GPU build check (jailed stage 5) before stage 1
    workers: int = 8                         # model calls at once
    tune: bool = False                       # each program's grid tuned before its full exam
    size_expr: str | None = None             # the task's call size: the tuning may dispatch on it
    library: list = field(default_factory=list)            # kernels.archive entries for every task (the memory's)
    embedder: object | None = None           # lab.search's embedder: near-duplicate programs refused before any GPU time
    vectors: dict = field(default_factory=dict)            # each program's fingerprint -> (its id or "", its vector)
    spent: float = 0.0
    calls: int = 0
    judge_calls: int = 0                     # calls so far: picks the model from the mix
    tried: set = field(default_factory=set)  # every program written this run, judged or not: never written again
    memory: tuple = ("", "")                 # (general table, task table) in every call, kernels/search/memory.py
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def _ask(self, model, system: str, user: str) -> str:
        text, usd = model(system, user)
        with self._lock:
            self.spent, self.calls = self.spent + usd, self.calls + 1
        return text or ""

    def _judge(self):
        """The next model from the mix."""
        mix = self.models.judges or (self.models.judge,)
        with self._lock:
            m = mix[self.judge_calls % len(mix)]
            self.judge_calls += 1
        return m

    def measure(self, code: str) -> tuple[str, dict]:
        """A program's full exam, its grid tuned on every seen shape first (kernels/search/tune.py): (the program
        measured, its result, with every grid point's row)."""
        rows = []
        if self.tune:
            code, rows = tune.tuned(self.stage, code, quick=False, size_expr=self.size_expr)
        r = self.stage(code, 2)
        if rows:
            r = {**r, "artifacts": {**r["artifacts"], "tune": json.dumps(rows)}}
        return code, r


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
    """A call's system prompt: the setting with the general table, then the task table, the role and the answer
    format, each part cached on its own (the general table changes least, so it comes first)."""
    from kernels.search.loop import FORMAT, JUDGE
    general, mine = f.memory
    return (system + (f"\n\n{general}" if general else "") + CACHE_SPLIT + "\n\n" + (f"{mine}\n\n" if mine else "")
            + JUDGE + "\n\n" + FORMAT)


def change(f: Funnel, system: str, base: str, parent: Node, facts: str, mode: str, seen: set,
           **kw) -> tuple[dict, str | None, str]:
    """One call: (its hypothesis, the program it wrote or None, why not). `facts` (a slice, a note, another program)
    goes between the cached `base` and the request `mode` (loop.MODES, formatted with `kw`)."""
    from kernels.search.loop import MODES, hypotheses
    text = f._ask(f._judge(), judging(f, system), base + CACHE_SPLIT + "".join(
        f"\n\n{t}" for t in (facts, MODES[mode].format(**kw)) if t))
    head = text.split("<<<<<<< SEARCH")[0]
    got = hypotheses(head, 1)
    idea = got[0] if got else {}
    code, why = diff.apply(parent.code, text) if text else (None, "the model returned no text")
    if code is not None and fingerprint(code) in seen:
        code, why = None, "identical to a program already judged (comments and whitespace aside)"
    return idea, code, why


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


def change_text(tree: Tree, code: str) -> str:
    """A program's lines changed from the starting program: what its embedding compares."""
    return "".join(x for x in difflib.unified_diff(tree.root().code.splitlines(True), code.splitlines(True), n=0)
                   if x[:1] in "+-" and not x.startswith(("+++", "---")))


def near_duplicates(f: Funnel, tree: Tree, codes: list[str]) -> list[str | None]:
    """For each new program, why it repeats one already written (its change from the start within NOVEL cosine of
    that one's), else None; the novel ones join `f.vectors`, so a later program in the same batch is compared too."""
    from lab.search import _cosine
    if f.embedder is None:
        return [None] * len(codes)
    for n in tree.nodes.values():                # a program written earlier has since joined the tree: name it
        if f.vectors.get(fingerprint(n.code), ("x",))[0] == "":
            f.vectors[fingerprint(n.code)] = (n.id, f.vectors[fingerprint(n.code)][1])
    old = [n for n in tree.nodes.values() if n.parent and fingerprint(n.code) not in f.vectors]
    for n, v in zip(old, f.embedder.passages([change_text(tree, n.code) for n in old]) if old else []):
        f.vectors[fingerprint(n.code)] = (n.id, v)
    out = []
    for c, v in zip(codes, f.embedder.passages([change_text(tree, c) for c in codes]) if codes else []):
        best = max(((_cosine(v, w), i) for i, w in f.vectors.values()), default=(0.0, ""))
        if best[0] >= NOVEL:
            out.append(f"a near-duplicate of {'program ' + best[1] if best[1] else 'a program written this run'} "
                       f"(cosine {best[0]:.3f} between their changes from the start)")
        else:
            out.append(None)
            f.vectors[fingerprint(c)] = ("", v)
    return out


PRECHECK = 5                 # kernels.evolve.jailed.PRECHECK_STAGE


def correct(f: Funnel, codes: list[str], stage: int = 1) -> list[dict]:
    """Stage 1 (or the precheck) for each program, `parallel` jails at a time (neither needs the GPU to itself)."""
    with ThreadPoolExecutor(max_workers=max(1, f.parallel)) as ex:
        return list(ex.map(lambda c: f.stage(c, stage), codes))


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


def profile_parents(f: Funnel, parents: list[Node]) -> None:
    """ncu counters (stage 3) for each parent that has none yet, into its artifacts; once the host refuses them, they
    are not asked for again."""
    from kernels.search.loop import REFUSED
    for p in parents:
        if not f.ncu or any(k in p.artifacts for k in ("ncu", "ncu_missing", "ncu_error")):
            continue
        a = f.stage(p.code, 3)["artifacts"]
        if REFUSED in json.dumps(a):
            f.ncu = False
            a = {"ncu_missing": f"ncu counters unavailable on this host ({REFUSED}); not collected this run"}
        p.artifacts.update(a)


def expand(f: Funnel, tree: Tree, parent: Node, system: str, it: int, keep_node: Callable[[Node, float], None],
           log: Callable[[dict], None]) -> list[Node]:
    """One round from one parent (an MCTS expansion): `per_call` changes, the slices rotated across them."""
    return generation(f, tree, [Slot(parent, "refine", f.per_call)], system, it, keep_node, log)


@dataclass
class Slot:
    """One parent's share of a round: `n` calls of idea type `kind` from `parent`."""
    parent: Node
    kind: str = "refine"         # start, refine, persist, fresh or crossover (kernels/search/generations.py)
    n: int = 1
    other: Node | None = None    # crossover: the other parent, shown as full code


def generation(f: Funnel, tree: Tree, slots: list[Slot], system: str, it: int,
               keep_node: Callable[[Node, float], None], log: Callable[[dict], None],
               extra: dict | None = None) -> list[Node]:
    """One round: each slot's `n` calls each make one change; every program that builds and passes stage 1 takes the
    full exam and joins the tree under its own parent, its `kind` its idea's type. A refine (or start) slot's calls
    read the slices in turn; persist, fresh and crossover calls read the measurements. Every idea carries an `idea_id`
    and its `type`, and its log row the diff its program made. `extra` goes into the round's log row."""
    from kernels.search.loop import prompt
    spent0, fates, children = f.spent, [], []
    counts = {"named": 0, "written": 0, "correct": 0}
    code_of = {}                                 # each written idea's program, for the log's diff
    by_id = {s.parent.id: s.parent for s in slots}
    parents = list(by_id.values())

    def key(h):
        return h.get("idea_id")
    profile_parents(f, parents)
    need = [p for p in parents if f.precheck and "compile_report" not in p.artifacts]
    for p, r in zip(need, correct(f, [p.code for p in need], PRECHECK)):    # for the compiler slice
        if r["artifacts"].get("compile_report"):
            p.artifacts["compile_report"] = r["artifacts"]["compile_report"]
    bases = {p.id: prompt(tree, p) for p in parents}
    seen = set(tree.seen) | f.tried
    calls = []                               # (slot, the facts its call reads, the request, its format args)
    for i, s in enumerate(slots):
        p = s.parent
        for k in range(s.n):
            if s.kind == "crossover":
                calls.append((s, f"# Another program in this generation, {s.other.id} (score {s.other.score:.4f})"
                                 f"\n```python\n{s.other.code}\n```", "crossover", {}))
            elif s.kind == "persist":        # the program's last change, to be taken further
                before = tree.nodes[p.parent].score if p.parent else 0.0
                calls.append((s, f"# Its last change\n{p.judge}", "persistence",
                              {"before": before, "after": p.score}))
            elif s.kind == "fresh":
                calls.append((s, "", "fresh", {}))
            else:
                facts = slices(f, tree, p)
                names = [x for x in SLICES if x in facts]
                calls.append((s, facts[names[(it + i + k) % len(names)]], "optimization", {}))

    def one(c):
        s, facts, mode, kw = c
        idea, code, why = change(f, system, bases[s.parent.id], s.parent, facts, mode, set(seen), **kw)
        return {**idea, "parent_id": s.parent.id, "type": s.kind, "idea_id": uuid.uuid4().hex[:8],
                **({"with": s.other.id} if s.other else {})}, (code, why)
    made = pmap(f, one, calls)
    counts["named"] = len(made)
    ideas = [h for h, _ in made]
    drafts = written(f, [cw for _, cw in made], seen)
    near = iter(near_duplicates(f, tree, [c for c, _ in drafts if c is not None]))
    drafts = [(c, w) if c is None else ((None, why) if (why := next(near)) else (c, w)) for c, w in drafts]
    wrote = []
    for h, (code, why) in zip(ideas, drafts):
        if code is None:
            fates.append({"idea": h, "fate": f"not applied: {why}"})
        else:
            wrote.append((h, code))
            code_of[key(h)] = code
    counts["written"] = len(wrote)
    reports = {}
    if f.precheck and wrote:                 # what fails to build never takes GPU time
        built = []
        for (h, code), r in zip(wrote, correct(f, [c for _, c in wrote], PRECHECK)):
            if r["artifacts"].get("compile_report"):
                reports[code] = r["artifacts"]["compile_report"]
            if r["metrics"].get("correct") == 0.0:
                fates.append({"idea": h, "fate": "precheck: " + r["artifacts"].get("precheck", "")[:300]})
                continue
            built.append((h, code))
        wrote = built
    survivors = []
    for (h, code), r in zip(wrote, correct(f, [c for _, c in wrote])):
        if r["metrics"].get("correct") == 1.0:
            survivors.append((h, code))
        else:
            fates.append({"idea": h, "fate": "failed correctness", "error": str(r["artifacts"])[:300]})
    counts["correct"] = len(survivors)
    for h, code in survivors:                # GPU timings: one at a time
        p = by_id[h["parent_id"]]
        report = reports.get(code)
        code, r = f.measure(code)
        code_of[key(h)] = code
        if report:
            r["artifacts"] = {**r["artifacts"], "compile_report": report}
        plan = json.dumps({k: v for k, v in h.items() if k in ("bottleneck", "optimisation_method",
                                                               "modification_plan", "mechanism")})
        child = Node(uuid.uuid4().hex[:8], p.id, code, r["metrics"], r["artifacts"], it, "", plan, "optimization",
                     h.get("type", ""))
        keep_node(child, 0.0)
        children.append(child)
        fates.append({"idea": h, "fate": "full exam", "child_id": child.id, "child_score": child.score,
                      "improved": child.score > p.score,
                      **({"mechanism": mechanism(h, p, child)} if h.get("mechanism") else {})})
    for x in fates:                              # each written idea's program as a diff from its parent
        c, p = code_of.get(key(x["idea"])), by_id.get(x["idea"].get("parent_id"))
        if c and p:
            x["diff"] = "".join(difflib.unified_diff(p.code.splitlines(True), c.splitlines(True), n=2))[:4000]
    one_parent = parents[0] if len(parents) == 1 else None
    log({"iteration": it, "parent_id": one_parent.id if one_parent else None,
         "parent_score": one_parent.score if one_parent else None,
         "parents": [{"id": p.id, "score": p.score} for p in parents], **counts, "examined": len(children),
         "improved": sum(c.score > by_id[c.parent].score for c in children),
         "usd": round(f.spent - spent0, 4), **(extra or {}),
         "ideas": fates})
    return children
