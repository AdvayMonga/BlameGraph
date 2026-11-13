"""Context: everything a model call reads, built in one place, and how its answer is read back. Facts only."""
from __future__ import annotations

import json
import re

from generate.kernels.search.llm import CACHE_SPLIT
from generate.kernels.search.program import Idea, Program

ROLE = ("You are making changes to GPU kernel programs in a search. You are shown one program, its measurements and "
        "the starting program.")
FORMAT = """Answer with SEARCH/REPLACE blocks against the program shown. Each SEARCH section must match the program
exactly once, or the answer is not applied. Any number of blocks:

<<<<<<< SEARCH
old lines
=======
new lines
>>>>>>> REPLACE"""
REQUEST = ('The program is correct. Make one change to it. Answer with one JSON object {"bottleneck": "<max 30 words>", '
           '"optimisation_method": "<max 35 words>", "modification_plan": "<max 35 words>", "mechanism": [{"counter": '
           '"<an ncu metric name from the measurements>", "change_pct": <predicted % change, summed over the program\'s '
           'kernels>}]} ("mechanism" may be empty), then the change as SEARCH/REPLACE blocks against the program shown. '
           'It is applied and measured.')
TIMING_KEYS = ("shape", "candidate_ms", "baseline_ms", "speedup", "speedup_p25", "speedup_p75", "beyond_noise",
               "pct_of_roofline", "bound", "pct_of_peak_bandwidth", "pct_of_peak_flops")
INSPIRATIONS = 2             # the best programs off the parent's lineage, shown as full code
HISTORY_MAX = 400            # rows of the run's ideas shown; the rest are counted
TEXT = 240                   # characters of an idea's text shown


def build(setting: str, parent: Program, population: list[Program], ideas: list[Idea],
          proposed: list[dict]) -> tuple[str, str]:
    """(system, user) for one call on `parent`: the setting and the run's history, then the program with its
    measurements and the changes already proposed for it this generation. CACHE_SPLIT ends each stable part."""
    past = history(ideas, population)
    system = setting + CACHE_SPLIT + "\n\n" + (f"{past}\n\n" if past else "") + ROLE + "\n\n" + FORMAT
    return system, program(parent, population) + CACHE_SPLIT + "".join(
        f"\n\n{t}" for t in (siblings(proposed), REQUEST) if t)


def read(text: str) -> dict:
    """The answer's JSON object (the idea), best effort; text that does not parse is kept as its plan."""
    head = (text or "").split("<<<<<<< SEARCH")[0]
    m = re.search(r"\{.*\}", head, re.S)
    try:
        v = json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        v = None
    return v if isinstance(v, dict) else {"plan": head} if head.strip() else {}


def short(plan: dict) -> str:
    t = " / ".join(str(plan[k]) for k in ("optimisation_method", "modification_plan") if plan.get(k)) \
        or json.dumps(plan, sort_keys=True)
    return t if len(t) <= TEXT else t[:TEXT] + "..."


def history(ideas: list[Idea], population: list[Program]) -> str:
    """Every idea of the run so far with what was measured: its gain over its parent, or how far it got."""
    if not ideas:
        return ""
    score = {p.id: p.score for p in population}
    rows = [json.dumps({"generation": i.generation, "idea": short(i.plan),
                        **({"gain": round(i.child.score / score[i.parent], 4)} if i.child and score.get(i.parent)
                           else {"furthest": i.fate[:120]})}) for i in ideas]
    more = [f"({len(rows) - HISTORY_MAX} more ideas not shown)"] if len(rows) > HISTORY_MAX else []
    return "\n".join(["# Every idea tried in this search, with what was measured (gain = child score / parent score)",
                      *rows[:HISTORY_MAX], *more])


def siblings(proposed: list[dict]) -> str:
    rows = [f"- {short(p)}" for p in proposed if p.get("optimisation_method") or p.get("modification_plan")]
    return ("# Changes already proposed for this program in this generation (not measured yet)\n" + "\n".join(rows)
            + "\nYour change must differ from each of these.") if rows else ""


def lineage(node: Program, population: list[Program]) -> list[Program]:
    by, out = {p.id: p for p in population}, []
    while node:
        out.append(node)
        node = by.get(node.parent)
    return out[::-1]


def program(node: Program, population: list[Program]) -> str:
    """The program, its measurements, its lineage and children, the starting program and the best programs off its
    lineage as full code."""
    root, mine = population[0], lineage(node, population)
    path = " -> ".join(f"{n.id} ({n.score:.4f})" for n in mine)
    kids = "\n".join(f"- {c.id}: score {c.score:.4f}" for c in population if c.parent == node.id) or "none"
    start = "" if node is root else f"# The starting program {root.id} (score {root.score:.4f})\n```python\n{root.code}\n```\n\n"
    others = sorted((p for p in population if p not in mine and p.usable and p.score > root.score),
                    key=lambda p: -p.score)[:INSPIRATIONS]
    other = "".join(f"## Program {o.id} (score {o.score:.4f})\n```python\n{o.code}\n```\n" for o in others)
    best = max(p.score for p in population)
    return (f"# Program {node.id} (score {node.score:.4f}; best in the search {best:.4f})\n"
            f"```python\n{node.code}\n```\n\n# Its measurements\n{facts(node)}\n\n"
            f"# Lineage (program, score)\n{path}\n\n# Programs already derived from it\n{kids}\n\n" + start
            + (f"# Other programs in the search\n{other}\n" if other else ""))


def facts(node: Program) -> str:
    """A program's measurements as the evaluator returned them: gates, per-shape timings, error against the
    reference, ncu counters, the compiler's report."""
    a, out = node.artifacts, [f"metrics: {json.dumps(node.metrics)}"]
    if a.get("error"):
        out.append(f"error: {a['error'][-2000:]}")
    if a.get("gates"):
        out.append("gates: " + "; ".join(f"{k} {'passed' if v['passed'] else 'FAILED: ' + v['reason']}"
                                         for k, v in json.loads(a["gates"]).items()))
    if a.get("timings"):
        rows = json.loads(a["timings"])
        w = sum(r.get("weight") or 0 for r in rows if r.get("pct_of_roofline") is not None)
        if w:
            pct = sum((r.get("weight") or 0) * r["pct_of_roofline"] for r in rows
                      if r.get("pct_of_roofline") is not None) / w
            out.append(f"{pct:.1f}% of its roofline bound, call-weighted")
        out.append("timings (per seen shape, paired with the baseline):")
        out += [json.dumps({k: r.get(k) for k in TIMING_KEYS}) for r in rows]
    if a.get("error_vs_reference"):
        out.append(f"error against the reference: {a['error_vs_reference']}")
    if counters(node):
        out.append("ncu counters (one call each on the smallest and the largest seen shape, longest kernels first):\n"
                   + counters(node))
    if a.get("compile_report"):
        out.append(f"the compiler's report:\n{a['compile_report']}")
    return "\n".join(out)


def counters(node: Program) -> str:
    """The top kernels of one call with their ncu counters, or why there are none."""
    a = node.artifacts
    if a.get("ncu"):
        d, lines = json.loads(a["ncu"]), []
        for p in d.get("shapes") or [{"shape": None, **d}]:
            head = f"shape {p['shape']}: " if p.get("shape") is not None else ""
            lines += [head + json.dumps(k) for k in p.get("kernels") or []] or [head + (p.get("error") or "none recorded")]
        return "\n".join(lines)
    if a.get("ncu_missing"):
        return a["ncu_missing"]
    return "ncu failed: " + a["ncu_error"][-300:] if a.get("ncu_error") else ""
