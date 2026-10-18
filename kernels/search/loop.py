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

import difflib
import json
import math
import re
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
               "pct_of_roofline", "bound", "pct_of_peak_bandwidth", "pct_of_peak_flops")


JUDGE = """You are the judge in a search over GPU kernel programs. You are shown one program and its measurements.
{mode_spec}"""
MODES = {
    "optimization": 'The program is correct. Name {n} distinct changes a coder could make, as one JSON array of {n} '
                    'objects and nothing else, each {{"bottleneck": "<max 30 words>", "optimisation_method": "<max 35 '
                    'words>", "modification_plan": "<max 35 words>", "expected_speedup": <your estimate, >= 1.0>, '
                    '"implementation_risk": <1 low to 3 high>, "performance_risk": <1 low to 3 high>}}.',
    "correction": 'The program failed (see its measurements). Name one fix, as one JSON object and nothing else: '
                  '{{"critical_issue": "<max 20 words>", "why_it_matters": "<max 35 words>", "minimal_fix_hint": '
                  '"<max 20 words>"}}.',
}


def hypotheses(text: str, n: int) -> list[dict]:
    """The judge's JSON array (or one object), best effort; unparsable text becomes one hypothesis."""
    m = re.search(r"\[.*\]|\{.*\}", text or "", re.S)
    try:
        v = json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        v = None
    v = v if isinstance(v, list) else [v] if isinstance(v, dict) else [{"plan": text}] if text else []
    return [h for h in v if isinstance(h, dict)][:n]


def roi(h: dict, pct_of_roofline: float | None) -> float:
    """MANTIS's triage score: S^(1 + max(0, log10(g / 5))) / (R_impl * R_perf), S = the judge's expected speedup,
    g = time over the speed-of-light bound (100 / % of roofline). Arithmetic on the judge's own estimates."""
    try:
        sp = max(float(h.get("expected_speedup", 1.0)), 1e-6)
        ri, rp = (min(max(float(h.get(k, 2)), 1.0), 3.0) for k in ("implementation_risk", "performance_risk"))
    except (TypeError, ValueError):
        return 0.0
    g = 100.0 / pct_of_roofline if pct_of_roofline else 1.0
    return sp ** (1 + max(0.0, math.log10(g / 5))) / (ri * rp)


def choose(scores: list[float], k: int, explore: int, rng) -> list[tuple[int, bool]]:
    """(index, chosen at random) for k of them: the top k - explore by score, then `explore` at random from the rest,
    so the measured outcomes include hypotheses the ranking did not favour (unbiased labels for a learned screener)."""
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    top = order[: max(0, k - explore)]
    rest = order[len(top):]
    picked = rng.sample(rest, min(explore, len(rest)))
    return [(i, False) for i in top] + [(i, True) for i in picked]


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


def history(tree: Tree, measured: list[dict], limit: int = 80) -> str:
    """Every hypothesis measured anywhere in the tree with what came of it, newest last: the judge sees what was
    tried on other branches too (in the 2026-10-18 A/B it re-proposed the same ~20 changes at every expansion)."""
    lines = []
    for m in measured[-limit:]:
        c = tree.nodes.get(m["child_id"])
        plan = str(m.get("plan") or "")[:200]
        why = f"; failed: {(c.artifacts.get('error') or '')[:120]}" if c is not None and c.score <= 0 else ""
        lines.append(f"- on {m['parent_id']} ({m['parent_score']:.4f}) -> {m['child_id']} {m['child_score']:.4f}{why}: {plan}")
    return "\n".join(lines) or "none"


def prompt(tree: Tree, node: Node, plan: str = "", coder: bool = True, measured: list[dict] | None = None) -> str:
    path = " -> ".join(f"{n.id} ({n.score:.4f})" for n in tree.lineage(node))
    tried = [tree.nodes[c] for c in node.children]
    kids = "\n".join(f"- {c.id}: score {c.score:.4f}" + (f", {c.artifacts.get('error', '')[:300]}"
                                                          if c.score <= 0 else "") for c in tried) or "none"
    sol = f"; {node.pct_of_roofline:.1f}% of its roofline bound, call-weighted" if node.pct_of_roofline else ""
    others = inspirations(tree, node)
    if coder:                                   # the coder sees another program only when told to merge it
        try:
            merge = json.loads(plan).get("merge_from") if plan.startswith("{") else None
        except (json.JSONDecodeError, AttributeError):
            merge = None
        others = [tree.nodes[merge]] if merge in tree.nodes else []
    other = "".join(f"## Program {o.id} (score {o.score:.4f}), its diff against the starting program\n"
                    f"```diff\n{diff_from_root(tree, o)}```\n" for o in others)
    return (f"# Program {node.id} (score {node.score:.4f}{sol}; best in the tree {tree.best().score:.4f})\n"
            f"```python\n{node.code}\n```\n\n# Its measurements\n{facts(node)}\n\n"
            f"# Lineage (node, score)\n{path}\n\n# Programs already derived from it\n{kids}\n\n"
            + (f"# Other programs in the tree\n{other}\n" if other else "")
            + (f"# Hypotheses already measured, anywhere in the tree (parent -> child score; plan)\n"
               f"{history(tree, measured)}\n\n" if measured is not None and not coder else "")
            + (f"# The change to make (from the judge)\n{plan}\n\n" if plan else "") + (FORMAT if coder else ""))


def merge_hypothesis(tree: Tree, node: Node) -> dict | None:
    """Combine this program with the best program on another branch (a crossover the beam can pick like any other
    hypothesis; its expected speedup is that program's measured score over this one's, so it outranks the judge's
    changes only where the other branch is really ahead; over the start's, as in the 2026-10-18 A/B, it was 8x and
    took a slot in every beam)."""
    o = inspirations(tree, node, 1)
    if not o:
        return None
    return {"bottleneck": "", "optimisation_method": "merge",
            "modification_plan": f"combine this program with program {o[0].id}: apply its changes (its diff against the "
                                 "starting program is shown) to this program",
            "merge_from": o[0].id, "expected_speedup": round(o[0].score / (node.score or 1.0), 4),
            "implementation_risk": 2, "performance_risk": 2}


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
           stop_pct: float | None = None, patience: int | None = None, n_hypotheses: int = 16, beam: int = 4,
           explore: int = 1, max_repairs: int = 3,
           screen: Callable[[str, str, list[dict]], tuple[list[float], float]] | None = None) -> tuple[Tree, dict]:
    """Search until `evals` programs are judged, `usd` is spent on model calls, nothing is left to try, or the best
    program reaches `stop_pct`% of its roofline bound, or `patience` programs have been judged since the best last
    improved (MANTIS, Hari et al.: within 25% of speed-of-light or 16 attempts without progress kept 96% of the geomean
    gain for 33% fewer tokens).

    Each expansion of an MCTS-chosen parent: the judge (`review`) names `n_hypotheses` changes with its own speedup
    and risk estimates; `screen` scores them (default: MANTIS's ROI on those estimates); the top `beam - explore` and
    `explore` random others are written by the coder (`propose`) and judged (Kevin's beam: 4x16, 1.56x vs best@16
    1.41x). A judged child that fails is repaired, one fix per round, up to `max_repairs` deep. Without a judge the
    coder alone proposes `beam` children. Every hypothesis, its score and what became of it go to
    `<out>/hypotheses.jsonl`. Returns the tree and the run's cost {usd, model_calls, evals, ...}."""
    out.mkdir(parents=True, exist_ok=True)
    tree, spent, calls, judged, it, repairs, tried = Tree(seed), 0.0, 0, 0, 0, [], []
    with open(out / "trace.jsonl", "w") as trace, open(out / "hypotheses.jsonl", "w") as hyplog:
        def keep(node: Node, cost: float) -> None:
            tree.add(node)
            trace.write(json.dumps({"iteration": node.iteration, "parent_id": node.parent, "child_id": node.id,
                                    "child_metrics": node.metrics, "child_code": node.code, "mode": node.mode,
                                    "artifacts": node.artifacts, "judge": node.judge, "proposal": node.proposal,
                                    "cost_usd": cost}) + "\n")
            trace.flush()

        def attempt(parent: Node, plan: str, mode: str, cost: float) -> Node | None:
            """One coder call on `plan`, applied and judged; the child (None when the budget ran out first)."""
            nonlocal spent, calls, judged
            if judged >= evals or spent >= usd:
                return None
            text, c = propose(system + "\n\n" + FORMAT, prompt(tree, parent, plan))
            spent, calls, cost = spent + c, calls + 1, cost + c
            code, why = diff.apply(parent.code, text) if text else (None, "the model returned no text")
            if code is not None and fingerprint(code) in tree.seen:
                code, why = None, "identical to a program already judged (comments and whitespace aside)"
            deep = parent.repairs + (mode == "correction")
            if code is None:                     # no GPU time spent: a failed child of its parent
                zero = {"combined_score": 0.0, "correct": 0.0, "stage": 0}
                child = Node(uuid.uuid4().hex[:8], parent.id, parent.code, zero, {"error": why}, it, text, plan, mode,
                             deep)
            else:
                r = judge(code)
                judged += 1
                child = Node(uuid.uuid4().hex[:8], parent.id, code, r["metrics"], r["artifacts"], it, text, plan,
                             mode, deep)
                if child.score <= 0 and review and deep < max_repairs:
                    repairs.append(child.id)
            keep(child, cost)
            return child

        r = judge(initial)
        judged += 1
        keep(Node(uuid.uuid4().hex[:8], None, initial, r["metrics"], r["artifacts"], 0), 0.0)
        best_at, best_score = judged, tree.best().score
        while judged < evals and spent < usd:
            if stop_pct and (tree.best().pct_of_roofline or 0) >= stop_pct:
                break
            if tree.best().score > best_score:
                best_at, best_score = judged, tree.best().score
            if patience and judged - best_at >= patience:
                break
            it += 1
            if review and repairs:               # a failed child first: one fix (CudaForge's correction mode)
                failed = tree.nodes[repairs.pop(0)]
                plan, cost = review(system + "\n\n" + JUDGE.format(mode_spec=MODES["correction"].format()),
                                    prompt(tree, failed, coder=False))
                spent, calls = spent + cost, calls + 1
                attempt(failed, plan, "correction", cost)
                continue
            parent = tree.select()
            if parent is None:
                break
            if not review:
                for _ in range(beam):
                    attempt(parent, "", "optimization", 0.0)
                continue
            text, cost = review(system + "\n\n" + JUDGE.format(mode_spec=MODES["optimization"].format(n=n_hypotheses)),
                                prompt(tree, parent, coder=False, measured=tried))
            spent, calls = spent + cost, calls + 1
            hs = hypotheses(text, n_hypotheses)
            mh = merge_hypothesis(tree, parent) if hs else None
            hs += [mh] if mh else []
            if not hs:
                keep(Node(uuid.uuid4().hex[:8], parent.id, parent.code, {"combined_score": 0.0, "correct": 0.0,
                                                                        "stage": 0},
                          {"error": "the judge named no change"}, it, "", text), cost)
                continue
            ranks = [roi(h, parent.pct_of_roofline) for h in hs]
            scores, c = screen(system, prompt(tree, parent, coder=False), hs) if screen else (ranks, 0.0)
            spent, calls = spent + c, calls + (1 if c else 0)   # a model screener's call; roi/random cost 0
            measured, best_before = [], tree.best().score
            for i, random_pick in choose(scores, beam, explore, tree.rng):
                child = attempt(parent, json.dumps(hs[i]), "optimization", cost if not measured else 0.0)
                if child is not None:                # the labels: did it beat its parent? the tree's best so far?
                    measured.append({"index": i, "random": random_pick, "child_id": child.id,
                                     "child_score": child.score, "improved": child.score > parent.score,
                                     "new_best": child.score > best_before})
                    tried.append({"parent_id": parent.id, "parent_score": parent.score, "child_id": child.id,
                                  "child_score": child.score, "plan": hs[i].get("modification_plan") or hs[i]})
            hyplog.write(json.dumps({"iteration": it, "parent_id": parent.id, "parent_score": parent.score,
                                     "hypotheses": hs, "roi": ranks, "scores": scores, "measured": measured,
                                     "screen": getattr(screen, "last", None)}, default=str) + "\n")
            hyplog.flush()
    near = bool(stop_pct and (tree.best().pct_of_roofline or 0) >= stop_pct)
    stalled = bool(patience and not near and judged < evals and spent < usd and judged - best_at >= patience)
    return tree, {"usd": round(spent, 4), "model_calls": calls, "evals": judged, "stopped_near_roofline": near,
                  "stopped_no_progress": stalled}
