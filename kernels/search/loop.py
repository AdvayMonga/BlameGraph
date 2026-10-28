"""The search: from the engine's kernel, repeatedly pick an open node of the tree, show the model that program and its
measurements (facts only: gates, per-shape paired timings, % of the roofline bound, error against the reference, the
lineage's scores, what its earlier children scored or why they failed), apply its SEARCH/REPLACE answer, and judge
the result with the cascade (stage 1, then stage 2). Lessons it builds on: serial refinement over parallel sampling
(Kevin), a tree with caps and pruning (STARK), keeping slower and failed nodes as context (Kernel Forge), no
conversation history, only the latest program and its facts (CudaForge), duplicates refused before any GPU time.
With a judge (a strong model), each expansion is one funnel round (kernels/search/funnel.py), whose judge calls use
`JUDGE` and `MODES` below; the genetic search (kernels/search/generations.py) runs the same rounds from several parents.

Every node goes to `<out>/trace.jsonl` in the fields `kernels.evolve.run.record` and `retime` read."""
from __future__ import annotations

import difflib
import json
import re
import tempfile
import uuid
from pathlib import Path
from typing import Callable

from kernels.search import diff
from kernels.search.llm import CACHE_SPLIT
from kernels.search.tree import UCT_C, Node, Tree, fingerprint

FORMAT = """Answer with SEARCH/REPLACE blocks against the program shown. Each SEARCH section must match the program
exactly once, or the answer is not applied. Any number of blocks:

<<<<<<< SEARCH
old lines
=======
new lines
>>>>>>> REPLACE"""

TIMING_KEYS = ("shape", "candidate_ms", "baseline_ms", "speedup", "speedup_p25", "speedup_p75", "beyond_noise",
               "pct_of_roofline", "bound", "pct_of_peak_bandwidth", "pct_of_peak_flops")


JUDGE = """You are the judge in a search over GPU kernel programs. You are shown one program and its measurements.
{mode_spec}"""
MODES = {
    "optimization": 'The program is correct. Name {n} distinct changes a coder could make, as one JSON array of {n} '
                    'objects and nothing else, each {{"bottleneck": "<max 30 words>", "optimisation_method": "<max 35 '
                    'words>", "modification_plan": "<max 35 words>", "mechanism": [{{"counter": "<an ncu metric name '
                    'from the measurements>", "change_pct": <predicted % change, summed over the program\'s kernels>}}]}}. '
                    '"mechanism" may be empty. Every one is written and measured.',
    "crossover": 'The program is correct. Below it is another program from the same search, as a diff. Name {n} '
                 'change(s) to this program that carry over what the other program does differently, as one JSON '
                 'array of {n} objects and nothing else, each {{"bottleneck": "<max 30 words>", "optimisation_method": '
                 '"<max 35 words>", "modification_plan": "<max 35 words>", "mechanism": []}}.',
}


def trace_row(node: Node, cost: float) -> dict:
    """A node as `<out>/trace.jsonl` keeps it (the fields `kernels.evolve.run.record` and `retime` read)."""
    return {"iteration": node.iteration, "parent_id": node.parent, "child_id": node.id, "child_metrics": node.metrics,
            "child_code": node.code, "mode": node.mode, "artifacts": node.artifacts, "judge": node.judge,
            "proposal": node.proposal, "cost_usd": cost}


def hypotheses(text: str, n: int) -> list[dict]:
    """The judge's JSON array (or one object), best effort; unparsable text becomes one hypothesis."""
    m = re.search(r"\[.*\]|\{.*\}", text or "", re.S)
    try:
        v = json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        v = None
    v = v if isinstance(v, list) else [v] if isinstance(v, dict) else [{"plan": text}] if text else []
    return [h for h in v if isinstance(h, dict)][:n]


def ncu_facts(node: Node) -> str:
    """The top kernels of one call with CudaForge's 24 counters, or why there are none."""
    a = node.artifacts
    if a.get("ncu"):
        d = json.loads(a["ncu"])
        per = d.get("shapes") or [{"shape": None, **d}]
        lines = []
        for p in per:
            head = f"shape {p['shape']}: " if p.get("shape") is not None else ""
            ks = p.get("kernels") or []
            lines += [head + json.dumps(k) for k in ks] or [head + (p.get("error") or "none recorded")]
        return "\n".join(lines)
    if a.get("ncu_missing"):
        return a["ncu_missing"]
    return "ncu failed: " + a["ncu_error"][-300:] if a.get("ncu_error") else ""


def facts(node: Node) -> str:
    """A node's measurements as the evaluator returned them, compacted: the gates (with reasons when failed), the
    per-shape timings, the error against the reference, or the error that stopped it."""
    a, out = node.artifacts, [f"metrics: {json.dumps(node.metrics)}"]
    if a.get("error"):
        out.append(f"error: {a['error'][-2000:]}")
    if a.get("gates"):
        g = json.loads(a["gates"])
        out.append("gates: " + "; ".join(f"{k} {'passed' if v['passed'] else 'FAILED: ' + v['reason']}"
                                         for k, v in g.items()))
    if a.get("timings"):
        out.append("timings (per seen shape, paired with the baseline):")
        out += [json.dumps({k: r.get(k) for k in TIMING_KEYS}) for r in json.loads(a["timings"])]
    if a.get("error_vs_reference"):
        out.append(f"error against the reference: {a['error_vs_reference']}")
    if ncu_facts(node):
        out.append("ncu counters (one call each on the smallest and the largest seen shape, longest kernels first):\n"
                   + ncu_facts(node))
    return "\n".join(out)


def diff_from_root(tree: Tree, node: Node, limit: int = 3000) -> str:
    root = tree.root()
    d = "".join(difflib.unified_diff(root.code.splitlines(keepends=True), node.code.splitlines(keepends=True),
                                     "start", node.id, n=2))
    return d if len(d) <= limit else d[:limit] + f"\n... ({len(d) - limit} more characters)\n"


def inspirations(tree: Tree, node: Node, k: int = 2) -> list[Node]:
    """The best judged programs off this program's lineage that beat the start (CodeEvolve: 2-3 inspirations)."""
    mine = {n.id for n in tree.lineage(node)}
    base = tree.root().score
    others = [n for n in tree.nodes.values() if n.id not in mine and n.metrics.get("correct") == 1.0 and n.score > base]
    return sorted(others, key=lambda n: -n.score)[:k]


def prompt(tree: Tree, node: Node, plan: str = "", coder: bool = True) -> str:
    path = " -> ".join(f"{n.id} ({n.score:.4f})" for n in tree.lineage(node))
    tried = [tree.nodes[c] for c in node.children]
    kids = "\n".join(f"- {c.id}: score {c.score:.4f}" + (f", {c.artifacts.get('error', '')[:300]}"
                                                          if c.score <= 0 else "") for c in tried) or "none"
    sol = f"; {node.pct_of_roofline:.1f}% of its roofline bound, call-weighted" if node.pct_of_roofline else ""
    others = [] if coder else inspirations(tree, node)       # the judge sees other branches; the coder its plan only
    other = "".join(f"## Program {o.id} (score {o.score:.4f}), its diff against the starting program\n"
                    f"```diff\n{diff_from_root(tree, o)}```\n" for o in others)
    return (f"# Program {node.id} (score {node.score:.4f}{sol}; best in the tree {tree.best().score:.4f})\n"
            f"```python\n{node.code}\n```\n\n# Its measurements\n{facts(node)}\n\n"
            f"# Lineage (node, score)\n{path}\n\n# Programs already derived from it\n{kids}\n\n"
            + (f"# Other programs in the tree\n{other}\n" if other else "")
            + (f"{CACHE_SPLIT}# The change to make (from the judge)\n{plan}\n\n" if plan else "")
            + (FORMAT if coder else ""))


REFUSED = "ERR_NVGPUCTRPERM"        # ncu: this host does not let the jail read GPU performance counters


def cascade(judge_stage: Callable[[str, int], dict], counters: bool = False) -> Callable[[str], dict]:
    """Stage 1, then stage 2 for a program that passes it, then (counters=True) its ncu counters: {metrics,
    artifacts}. Once the host refuses counters, they are not asked for again (two jailed runs per program, for an
    error: every profile of 2026-10-18/19)."""
    on = [counters]

    def judge(code: str) -> dict:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(code)
        try:
            r = judge_stage(f.name, 1)
            if r["metrics"].get("correct") != 1.0:
                return r
            r = judge_stage(f.name, 2)
            if on[0] and r["metrics"].get("correct") == 1.0 and r["metrics"].get("combined_score", 0) > 0:
                a = judge_stage(f.name, 3)["artifacts"]
                if REFUSED in json.dumps(a):
                    on[0] = False
                    a = {"ncu_missing": f"ncu counters unavailable on this host ({REFUSED}); not collected this run"}
                r["artifacts"] = {**r["artifacts"], **a}
            return r
        finally:
            Path(f.name).unlink()
    return judge


def search(initial: str, system: str, propose: Callable[[str, str], tuple[str, float]],
           judge: Callable[[str], dict], out: Path, evals: int, usd: float, seed: int = 0,
           stop_pct: float | None = None, patience: int | None = None, children: int = 4,
           uct_c: float = UCT_C, funnel=None, gpu_s: float | None = None,
           gpu_used: Callable[[], float] | None = None) -> tuple[Tree, dict]:
    """Search until `evals` programs are judged, `usd` is spent on model calls, nothing is left to try, or the best
    program reaches `stop_pct`% of its roofline bound, or `patience` programs have been judged since the best last
    improved (MANTIS, Hari et al.: within 25% of speed-of-light or 16 attempts without progress kept 96% of the geomean
    gain for 33% fewer tokens).

    Each expansion of an MCTS-chosen parent is one funnel round (kernels/search/funnel.py), every idea and what became
    of it in `<out>/hypotheses.jsonl`; without a funnel (no judge), the coder alone proposes `children` children.
    Returns the tree and the run's cost {usd, model_calls, evals, ...}."""
    out.mkdir(parents=True, exist_ok=True)
    tree, spent, calls, judged, it = Tree(seed, c=uct_c), 0.0, 0, 0, 0

    def gpu_out() -> bool:
        return bool(gpu_s is not None and gpu_used and gpu_used() >= gpu_s)
    if funnel is not None:                       # its rounds stop mid-way at the same caps
        funnel.usd_cap = usd if funnel.usd_cap is None else funnel.usd_cap
        funnel.gpu_s_cap, funnel.gpu_used = gpu_s, gpu_used
    with open(out / "trace.jsonl", "w") as trace, open(out / "hypotheses.jsonl", "w") as hyplog:
        def keep(node: Node, cost: float) -> None:
            tree.add(node)
            trace.write(json.dumps(trace_row(node, cost)) + "\n")
            trace.flush()

        def attempt(parent: Node) -> None:
            """One coder call, applied and judged (nothing when the budget ran out first)."""
            nonlocal spent, calls, judged
            if judged >= evals or spent >= usd or gpu_out():
                return
            text, c = propose(system + "\n\n" + FORMAT, prompt(tree, parent))
            spent, calls = spent + c, calls + 1
            code, why = diff.apply(parent.code, text) if text else (None, "the model returned no text")
            if code is not None and fingerprint(code) in tree.seen:
                code, why = None, "identical to a program already judged (comments and whitespace aside)"
            if code is None:                     # no GPU time spent: a failed child of its parent
                zero = {"combined_score": 0.0, "correct": 0.0, "stage": 0}
                keep(Node(uuid.uuid4().hex[:8], parent.id, parent.code, zero, {"error": why}, it, text), c)
                return
            r = judge(code)
            judged += 1
            keep(Node(uuid.uuid4().hex[:8], parent.id, code, r["metrics"], r["artifacts"], it, text), c)

        r = judge(initial)
        judged += 1
        keep(Node(uuid.uuid4().hex[:8], None, initial, r["metrics"], r["artifacts"], 0), 0.0)
        best_at, best_score = judged, tree.best().score
        while judged < evals and spent < usd and not gpu_out():
            if stop_pct and (tree.best().pct_of_roofline or 0) >= stop_pct:
                break
            if tree.best().score > best_score:
                best_at, best_score = judged, tree.best().score
            if patience and judged - best_at >= patience:
                break
            it += 1
            parent = tree.select()
            if parent is None:
                break
            if funnel is None:
                for _ in range(children):
                    attempt(parent)
                continue
            from kernels.search.funnel import expand
            before = funnel.spent
            kids = expand(funnel, tree, parent, system, it, keep,
                          lambda row: (hyplog.write(json.dumps(row, default=str) + "\n"), hyplog.flush()))
            spent, calls, judged = spent + funnel.spent - before, funnel.calls, judged + len(kids)
            if not kids:                         # a round that examined nothing counts as one failure
                parent.failures += 1
    near = bool(stop_pct and (tree.best().pct_of_roofline or 0) >= stop_pct)
    stalled = bool(patience and not near and judged < evals and spent < usd and judged - best_at >= patience)
    return tree, {"usd": round(spent, 4), "model_calls": calls, "evals": judged, "stopped_near_roofline": near,
                  "stopped_no_progress": stalled}
