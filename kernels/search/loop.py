"""The search: from the engine's kernel, repeatedly pick an open node of the tree, show the model that program and its
measurements (facts only: gates, per-shape paired timings, % of the roofline bound, error against the reference, the
lineage's scores, what its earlier children scored or why they failed), apply its SEARCH/REPLACE answer, and judge
the result with the cascade (stage 1, then stage 2). Lessons it builds on: serial refinement over parallel sampling
(Kevin), a tree with caps and pruning (STARK), keeping slower and failed nodes as context (Kernel Forge), no
conversation history, only the latest program and its facts (CudaForge), duplicates refused before any GPU time.

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
    return "\n".join(out)


def prompt(tree: Tree, node: Node) -> str:
    path = " -> ".join(f"{n.id} ({n.score:.4f})" for n in tree.lineage(node))
    tried = [tree.nodes[c] for c in node.children]
    kids = "\n".join(f"- {c.id}: score {c.score:.4f}" + (f", {c.artifacts.get('error', '')[:300]}"
                                                          if c.score <= 0 else "") for c in tried) or "none"
    return (f"# Program {node.id} (score {node.score:.4f}; best in the tree {tree.best().score:.4f})\n"
            f"```python\n{node.code}\n```\n\n# Its measurements\n{facts(node)}\n\n"
            f"# Lineage (node, score)\n{path}\n\n# Programs already derived from it\n{kids}\n\n{FORMAT}")


def cascade(judge_stage: Callable[[str, int], dict]) -> Callable[[str], dict]:
    """Stage 1, then stage 2 for a program that passes it: {metrics, artifacts}."""
    def judge(code: str) -> dict:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(code)
        try:
            r = judge_stage(f.name, 1)
            return judge_stage(f.name, 2) if r["metrics"].get("correct") == 1.0 else r
        finally:
            Path(f.name).unlink()
    return judge


def search(initial: str, system: str, propose: Callable[[str, str], tuple[str, float]],
           judge: Callable[[str], dict], out: Path, evals: int, usd: float, seed: int = 0) -> tuple[Tree, dict]:
    """Search until `evals` programs are judged, `usd` is spent on model calls, or no node is open.
    Returns the tree and the run's cost {usd, model_calls, evals}."""
    out.mkdir(parents=True, exist_ok=True)
    tree, spent, calls, judged, it = Tree(seed), 0.0, 0, 0, 0
    with open(out / "trace.jsonl", "w") as trace:
        def keep(node: Node, cost: float) -> None:
            tree.add(node)
            trace.write(json.dumps({"iteration": node.iteration, "parent_id": node.parent, "child_id": node.id,
                                    "child_metrics": node.metrics, "child_code": node.code,
                                    "artifacts": node.artifacts, "proposal": node.proposal, "cost_usd": cost}) + "\n")
            trace.flush()
        r = judge(initial)
        judged += 1
        keep(Node(uuid.uuid4().hex[:8], None, initial, r["metrics"], r["artifacts"], 0), 0.0)
        while judged < evals and spent < usd:
            parent = tree.select()
            if parent is None:
                break
            it += 1
            text, cost = propose(system, prompt(tree, parent))
            spent, calls = spent + cost, calls + 1
            code, why = diff.apply(parent.code, text) if text else (None, "the model returned no text")
            if code is not None and fingerprint(code) in tree.seen:
                code, why = None, "identical to a program already judged (comments and whitespace aside)"
            if code is None:                     # no GPU time spent: a failed child of its parent
                zero = {"combined_score": 0.0, "correct": 0.0, "stage": 0}
                keep(Node(uuid.uuid4().hex[:8], parent.id, parent.code, zero, {"error": why}, it, text), cost)
                continue
            r = judge(code)
            judged += 1
            keep(Node(uuid.uuid4().hex[:8], parent.id, code, r["metrics"], r["artifacts"], it, text), cost)
    return tree, {"usd": round(spent, 4), "model_calls": calls, "evals": judged}
