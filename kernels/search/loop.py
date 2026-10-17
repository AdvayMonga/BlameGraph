"""The search: from the engine's kernel, repeatedly pick an open node of the tree, show the model that program and its
measurements (facts only: gates, per-shape paired timings, % of the roofline bound, error against the reference, the
lineage's scores, what its earlier children scored or why they failed), apply its SEARCH/REPLACE answer, and judge
the result with the cascade (stage 1, then stage 2). Lessons it builds on: serial refinement over parallel sampling
(Kevin), a tree with caps and pruning (STARK), keeping slower and failed nodes as context (Kernel Forge), no
conversation history, only the latest program and its facts (CudaForge), duplicates refused before any GPU time.
With a judge (a strong model), each step is two calls (CudaForge, KForge): the judge reads the facts, including
CudaForge's curated ncu counters, and names one change as JSON; the coder writes it. A judged child that fails gets one
repair (the judge in correction mode on it; Kernel Forge's bounded repair loop).

Every node goes to `<out>/trace.jsonl` in the fields `kernels.evolve.run.record` and `retime` read."""
from __future__ import annotations

import json
import tempfile
import uuid
from pathlib import Path
from typing import Callable

from kernels.search import diff
from kernels.search.tree import Node, Tree, fingerprint

FORMAT = """Answer with SEARCH/REPLACE blocks against the program shown. Each SEARCH section must match the program
exactly once, or the answer is not applied. Any number of blocks:

<<<<<<< SEARCH
old lines
=======
new lines
>>>>>>> REPLACE"""

TIMING_KEYS = ("shape", "candidate_ms", "baseline_ms", "speedup", "speedup_p25", "speedup_p75", "beyond_noise",
               "pct_of_roofline")


JUDGE = """You are the judge in a search over GPU kernel programs. You are shown one program and its measurements;
name one change for a coder to make, as one JSON object and nothing else.
{mode_spec}"""
MODES = {
    "optimization": 'The program is correct. Reply {"bottleneck": "<max 30 words>", "optimisation_method": "<max 35 '
                    'words>", "modification_plan": "<max 35 words>"}.',
    "correction": 'The program failed (see its measurements). Reply {"critical_issue": "<max 20 words>", '
                  '"why_it_matters": "<max 35 words>", "minimal_fix_hint": "<max 20 words>"}.',
}


def ncu_facts(node: Node) -> str:
    """The top kernels of one call with CudaForge's 24 counters, or why there are none."""
    a = node.artifacts
    if a.get("ncu"):
        return "\n".join(json.dumps(k) for k in json.loads(a["ncu"]).get("kernels", [])) or "none recorded"
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
        out.append("ncu counters (one call, most-called seen shape, longest kernels first):\n" + ncu_facts(node))
    return "\n".join(out)


def prompt(tree: Tree, node: Node, plan: str = "", coder: bool = True) -> str:
    path = " -> ".join(f"{n.id} ({n.score:.4f})" for n in tree.lineage(node))
    tried = [tree.nodes[c] for c in node.children]
    kids = "\n".join(f"- {c.id}: score {c.score:.4f}" + (f", {c.artifacts.get('error', '')[:300]}"
                                                          if c.score <= 0 else "") for c in tried) or "none"
    sol = f"; {node.pct_of_roofline:.1f}% of its roofline bound, call-weighted" if node.pct_of_roofline else ""
    return (f"# Program {node.id} (score {node.score:.4f}{sol}; best in the tree {tree.best().score:.4f})\n"
            f"```python\n{node.code}\n```\n\n# Its measurements\n{facts(node)}\n\n"
            f"# Lineage (node, score)\n{path}\n\n# Programs already derived from it\n{kids}\n\n"
            + (f"# The change to make (from the judge)\n{plan}\n\n" if plan else "") + (FORMAT if coder else ""))


def cascade(judge_stage: Callable[[str, int], dict], counters: bool = False) -> Callable[[str], dict]:
    """Stage 1, then stage 2 for a program that passes it, then (counters=True) its ncu counters: {metrics,
    artifacts}."""
    def judge(code: str) -> dict:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(code)
        try:
            r = judge_stage(f.name, 1)
            if r["metrics"].get("correct") != 1.0:
                return r
            r = judge_stage(f.name, 2)
            if counters and r["metrics"].get("correct") == 1.0 and r["metrics"].get("combined_score", 0) > 0:
                r["artifacts"] = {**r["artifacts"], **judge_stage(f.name, 3)["artifacts"]}
            return r
        finally:
            Path(f.name).unlink()
    return judge


def search(initial: str, system: str, propose: Callable[[str, str], tuple[str, float]],
           judge: Callable[[str], dict], out: Path, evals: int, usd: float, seed: int = 0,
           review: Callable[[str, str], tuple[str, float]] | None = None,
           stop_pct: float | None = None) -> tuple[Tree, dict]:
    """Search until `evals` programs are judged, `usd` is spent on model calls, nothing is left to try, or the best
    program is within `stop_pct`% of its roofline bound (Hari et al.: stopping near speed-of-light kept >= 95% of the
    gain for 19-43% fewer tokens).
    `propose` is the coder, `review` the judge (None: the coder alone). Returns the tree and the run's cost
    {usd, model_calls, evals}."""
    out.mkdir(parents=True, exist_ok=True)
    tree, spent, calls, judged, it, repairs = Tree(seed), 0.0, 0, 0, 0, []
    with open(out / "trace.jsonl", "w") as trace:
        def keep(node: Node, cost: float) -> None:
            tree.add(node)
            trace.write(json.dumps({"iteration": node.iteration, "parent_id": node.parent, "child_id": node.id,
                                    "child_metrics": node.metrics, "child_code": node.code, "mode": node.mode,
                                    "artifacts": node.artifacts, "judge": node.judge, "proposal": node.proposal,
                                    "cost_usd": cost}) + "\n")
            trace.flush()
        r = judge(initial)
        judged += 1
        keep(Node(uuid.uuid4().hex[:8], None, initial, r["metrics"], r["artifacts"], 0), 0.0)
        while judged < evals and spent < usd:
            if stop_pct and (tree.best().pct_of_roofline or 0) >= stop_pct:
                break
            if review and repairs:
                parent, mode = tree.nodes[repairs.pop(0)], "correction"
            else:
                parent, mode = tree.select(), "optimization"
            if parent is None:
                break
            it += 1
            plan, cost = review(system + "\n\n" + JUDGE.format(mode_spec=MODES[mode]), prompt(tree, parent, coder=False)) \
                if review else ("", 0.0)
            text, c2 = propose(system + "\n\n" + FORMAT, prompt(tree, parent, plan))
            cost += c2
            spent, calls = spent + cost, calls + (2 if review else 1)
            code, why = diff.apply(parent.code, text) if text else (None, "the model returned no text")
            if code is not None and fingerprint(code) in tree.seen:
                code, why = None, "identical to a program already judged (comments and whitespace aside)"
            if code is None:                     # no GPU time spent: a failed child of its parent
                zero = {"combined_score": 0.0, "correct": 0.0, "stage": 0}
                keep(Node(uuid.uuid4().hex[:8], parent.id, parent.code, zero, {"error": why}, it, text, plan, mode),
                     cost)
                continue
            r = judge(code)
            judged += 1
            child = Node(uuid.uuid4().hex[:8], parent.id, code, r["metrics"], r["artifacts"], it, text, plan, mode)
            keep(child, cost)
            if child.score <= 0 and mode == "optimization":
                repairs.append(child.id)         # one repair per failed child
    near = bool(stop_pct and (tree.best().pct_of_roofline or 0) >= stop_pct)
    return tree, {"usd": round(spent, 4), "model_calls": calls, "evals": judged, "stopped_near_roofline": near}
