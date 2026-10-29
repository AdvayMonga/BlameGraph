"""What the judge is shown of every idea already measured (decided 2026-10-29): two tables in every judge call.

The general table: ideas that beat their parent on other kernel tasks, each with its best measured gain (child score /
parent score), how often it was measured and improved, that task, and the bound type (memory or compute) of the
program it won on. The task table: every distinct idea tried on this task, past runs (the idea archive,
kernels/archive.py) and this run so far (its hypothesis log), each with how often it was tried, how far it got and
its measured gains; an archived idea that names a program of its own run is left out. Measurements only (the no-beliefs rule, 2026-10-09): an idea's text is the judge's, its outcomes
the evaluator's; no lesson is written down. They replace the funnel's archive slice, which showed only the ideas most
like one parent's to one call in five.

The general table changes only with the archive, so it sits in the prompt before the task table, which grows every
generation: a prompt cache matches up to the first block that changed."""
from __future__ import annotations

import json
import re

GENERAL_MAX, TASK_MAX = 150, 400     # rows shown; the rest are counted
TEXT = 240                           # characters of an idea's text shown
ELSEWHERE = re.compile(r"\bprogram [0-9a-f]{8}\b")   # an archived idea naming a program of its own run (the 2026-10-18
                                                        # merge hypothesis): that program is not in this search


def short(idea: dict) -> str:
    t = " / ".join(str(idea[k]) for k in ("optimisation_method", "modification_plan") if idea.get(k)) \
        or json.dumps(idea, sort_keys=True)
    return t if len(t) <= TEXT else t[:TEXT] + "..."


def gain(parent: float | None, child: float | None) -> float | None:
    return round(child / parent, 4) if parent and child is not None and parent > 0 else None


def _cut(rows: list[dict], cap: int, what: str) -> str:
    out = [json.dumps(r) for r in rows[:cap]]
    if len(rows) > cap:
        out.append(f"({len(rows) - cap} more {what} not shown)")
    return "\n".join(out)


def general(entries: list, task: str | None) -> str:
    """Ideas from the archive that beat their parent on a task other than `task`, best gain first."""
    rows = []
    for e in entries:
        if e.task == task or not e.ever_improved() or ELSEWHERE.search(e.text):
            continue
        won = [o for o in e.outcomes if o.improved]
        best = max(won, key=lambda o: gain(o.parent_score, o.child_score) or 0)
        rows.append({"task": e.task, "idea": short(e.idea), "best_gain": gain(best.parent_score, best.child_score),
                     "measured": sum(1 for o in e.outcomes if o.child_id), "improved": len(won),
                     **({"bound": best.bound} if best.bound else {})})
    if not rows:
        return ""
    rows.sort(key=lambda r: -(r["best_gain"] or 0))
    return ("# Kernel ideas that beat their parent on other tasks (gain = child score / parent score, measured)\n"
            + _cut(rows, GENERAL_MAX, "such ideas"))


def _furthest(fates: list[str]) -> str:
    """How far an idea got when it never reached the full exam: its last recorded fate."""
    return (fates[-1] if fates else "never written")[:120]


def task(entries: list, name: str | None, log: list[dict]) -> str:
    """Every idea tried on `name`: the archive's distinct ideas, then each idea of this run's generations so far."""
    past = []
    for e in entries:
        if e.task != name or ELSEWHERE.search(e.text):
            continue
        gains = [g for o in e.outcomes if (g := gain(o.parent_score, o.child_score)) is not None]
        row = {"idea": short(e.idea), "tried": len(e.outcomes), "full_exam": sum(1 for o in e.outcomes if o.child_id),
               "improved": sum(1 for o in e.outcomes if o.improved)}
        if gains:
            row |= {"best_gain": max(gains), "worst_gain": min(gains)}
        else:
            row["furthest"] = _furthest([o.error or "" for o in e.outcomes])
        past.append(row)
    past.sort(key=lambda r: -(r.get("best_gain") or 0))
    now = []
    for g in log:
        scores = {p["id"]: p["score"] for p in g.get("parents") or [] if isinstance(p, dict)}
        for x in g.get("ideas") or []:
            h = x.get("idea") or {}
            row = {"generation": g.get("iteration"), "idea": short(h), **({"type": h["type"]} if h.get("type") else {})}
            if x.get("child_id"):
                row["gain"] = gain(scores.get(h.get("parent_id"), g.get("parent_score")), x.get("child_score"))
            else:
                row["furthest"] = _furthest([x.get("fate") or ""])
            now.append(row)
    if not past and not now:
        return ""
    out = ["# Every idea tried on this task, with what was measured (gain = child score / parent score)"]
    if past:
        out += ["## Earlier runs (similar ideas folded into one row)", _cut(past, TASK_MAX, "ideas")]
    if now:
        out += ["## This run so far", _cut(now, TASK_MAX, "ideas")]
    return "\n".join(out)
