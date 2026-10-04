"""inference-server lab ledger (lab/ledger/ledger.jsonl) -> agent-facing facts and integrity verdict.

The lab changes engine source, so a workspace snapshot id plays the role of a config, and every bench launches a
fresh server. Records are lab/tools.py `_record` lines (kind, run, session, tool, args, snapshot, snapshot_blob,
patch, result, cost) plus `session` records. Fields are read from `result`, else from the record itself (the lab
README puts held-out `config`/`metrics` at top level). Assumed result shapes, every field optional:
  test    {"lint": bool, "tests": bool, ...}                         as lab/tools.py writes it
  equiv   {"passed": bool, "reasons": [str]}
  bench   {"verdict": "ok", "split": "seen", "metrics": {name: {"base", "new", "delta_pct", "band_pct", "verdict"}}}
  submit  {"verdict": "ok", "split": "heldout", "metrics": {...as bench...}}
{"verdict": "refused", "reason": str} means the tool did not run. A metric's verdict is "win", "loss" or
"within_band" (delta_pct is signed new vs base; "win" says which way is better); a bare number means {"new": n}.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

KINDS = {"test", "equiv", "bench", "profile", "submit", "finding", "session", "note"}
TOOL_KINDS = ("test", "equiv", "bench", "profile", "submit")


def ledger_file(p: Path) -> Path | None:
    """The lab ledger at `p` (the file, or a directory holding ledger.jsonl), else None."""
    f = p / "ledger.jsonl" if p.is_dir() else p
    if not f.is_file():
        return None
    with open(f) as fh:
        first = next((ln for ln in fh if ln.strip()), "")
    try:
        r = json.loads(first)
    except json.JSONDecodeError:
        return None
    return f if isinstance(r, dict) and r.get("kind") in KINDS and "schema" in r else None


def load(f: Path, run: str | None = None) -> tuple[str | None, list[dict]]:
    """Records of one run (default: the run of the last record naming one). A torn final line is skipped."""
    lines = [ln for ln in Path(f).read_text().splitlines() if ln.strip()]
    recs = []
    for i, ln in enumerate(lines):
        try:
            recs.append(json.loads(ln))
        except json.JSONDecodeError:
            if i < len(lines) - 1:
                raise
    run = run or next((r["run"] for r in reversed(recs) if r.get("run")), None)
    return run, [r for r in recs if run is None or r.get("run") == run]


def result(r: dict) -> dict:
    return r.get("result") or r


def refused(r: dict) -> bool:
    return result(r).get("verdict") == "refused"


def passed(r: dict) -> bool | None:
    """test: lint and tests passed; equiv: its verdict (None if absent); bench/submit: completed with metrics."""
    res = result(r)
    if r["kind"] == "test":
        return bool(res.get("lint") and res.get("tests"))
    if r["kind"] == "equiv":
        return res.get("passed")
    return res.get("verdict", "ok") == "ok" and bool(res.get("metrics"))


def split(r: dict) -> str:
    return result(r).get("split") or (r.get("config") or {}).get("split") or ("heldout" if r["kind"] == "submit" else "seen")


def metrics(r: dict) -> dict:
    return {k: m if isinstance(m, dict) else {"new": m} for k, m in (result(r).get("metrics") or {}).items()}


def gain(m: dict) -> float | None:
    """Improvement in percent from a metric's verdict and delta_pct; None when either is missing."""
    d, v = m.get("delta_pct"), m.get("verdict")
    if not isinstance(d, (int, float)) or not v:
        return None
    return abs(d) if v == "win" else -abs(d) if v == "loss" else 0.0


def submitted(recs: list[dict]) -> dict | None:
    """The last completed submit."""
    return next((r for r in reversed(recs) if r["kind"] == "submit" and not refused(r) and passed(r)), None)


def _best(snaps: dict, only: str | None = None) -> dict:
    best: dict[str, tuple] = {}
    for sid, s in snaps.items():
        for b in s["benches"] if only in (None, sid) else ():
            for name, m in b["metrics"].items():
                g = gain(m)
                if b["split"] == "seen" and g is not None and (name not in best or g > best[name][0]):
                    best[name] = (g, {"snapshot": sid, "record": b["record"], **m})
    return {k: v[1] for k, v in best.items()}


def facts(recs: list[dict]) -> dict:
    """Snapshots tried and what each tool recorded on them. No judgments."""
    calls: dict[str, Counter] = {}
    snaps: dict[str, dict] = {}
    for r in recs:
        if r.get("tool"):
            c = calls.setdefault(r["tool"], Counter(calls=0, refused=0))
            c["calls"] += 1
            c["refused"] += refused(r)
        if r["kind"] not in TOOL_KINDS or refused(r):
            continue
        s = snaps.setdefault(r.get("snapshot"), {"tests_passed": 0, "tests_failed": 0, "equiv_passed": 0,
                                                 "equiv_failed": 0, "benches": [], "benches_not_completed": 0})
        ok = passed(r)
        if r["kind"] in ("test", "equiv") and ok is not None:
            s[("tests_" if r["kind"] == "test" else "equiv_") + ("passed" if ok else "failed")] += 1
        elif r["kind"] == "bench" and ok:
            s["benches"].append({"record": r.get("id"), "split": split(r), "metrics": metrics(r)})
        elif r["kind"] == "bench":
            s["benches_not_completed"] += 1
    sub = submitted(recs)
    sid = sub.get("snapshot") if sub else None
    return {
        "sessions": sum(r["kind"] == "session" for r in recs),
        "tool_calls": {k: dict(v) for k, v in calls.items()},
        "snapshots": snaps,
        "submitted_snapshot": sid,
        "submitted_result": metrics(sub) if sub else None,
        "best_measured_seen": _best(snaps),
        "submitted_best_measured_seen": _best(snaps, sid) if sub else None,
    }


def integrity(recs: list[dict]) -> dict:
    """Valid iff a completed submit exists whose snapshot has a completed seen-split bench, passing tests, a
    passing equiv and no failing equiv, and no record reports a write-surface violation."""
    reasons = []
    viol = [r for r in recs if r.get("violation") or (r.get("result") or {}).get("violation")]
    if viol:
        reasons.append(f"write_surface: {len(viol)} record(s) report a write-surface violation")
    sub = submitted(recs)
    if sub is None:
        subs = [r for r in recs if r["kind"] == "submit"]
        reasons.append(f"submission_known: no completed submit ({len(subs)} submit call(s), "
                       f"{sum(map(refused, subs))} refused)")
        return {"valid": False, "reasons": reasons}
    sid = sub.get("snapshot")
    on = [r for r in recs if r.get("snapshot") == sid]
    ref = Counter(r["kind"] for r in on if refused(r))
    on = [r for r in on if not refused(r)]
    if not any(r["kind"] == "bench" and passed(r) and split(r) == "seen" for r in on):
        reasons.append(f"submission_benchmarked: submitted snapshot {sid} has no completed bench on the seen split "
                       f"({ref['bench']} refused)")
    eq = [passed(r) for r in on if r["kind"] == "equiv"]
    if True not in eq or False in eq:
        reasons.append(f"submission_equivalent: submitted snapshot {sid} has {eq.count(True)} passing and "
                       f"{eq.count(False)} failing equiv record(s) ({ref['equiv']} refused)")
    ts = [passed(r) for r in on if r["kind"] == "test"]
    if True not in ts:
        reasons.append(f"submission_tested: submitted snapshot {sid} has no passing test record ({ts.count(False)} failing)")
    return {"valid": not reasons, "reasons": reasons}
