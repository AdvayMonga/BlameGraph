"""Lab ledger -> feedback on synthetic inference-server ledgers. Run: python tests/test_lab_verdict.py"""
from __future__ import annotations

import itertools
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from feedback.report import feedback, render_for_agent  # noqa: E402

RUN = "run-20261002-120000-abc123"
A, B = "a" * 24, "b" * 24
NOT_WIRED = {"bench": "the eval harness is not connected yet; nothing can be scored",
             "equiv": "output equivalence is not wired yet",
             "submit": "submit refuses until the eval harness is wired; nothing can become a win yet"}
_ids = itertools.count(1)


def rec(kind: str, snapshot: str, result: dict, tool: str | None = None, run: str = RUN) -> dict:
    """A record as lab/tools.py `_record` writes it, stamped as lab/ledger.py `append` does."""
    return {"id": f"ev-20261002-{next(_ids):012x}", "at": "2026-10-02T12:00:00+00:00", "schema": 1,
            "kind": kind, "run": run, "session": f"{run}-s1", "tool": tool or kind, "args": {},
            "snapshot": snapshot, "snapshot_blob": f"blobs/{snapshot}", "patch": f"blobs/{snapshot}x.patch",
            "result": result, "cost": {"usd": 0.0}}


def session_rec(snapshot: str, note: str | None = None, run: str = RUN) -> dict:
    """A record as lab/session.py writes at session end."""
    return {"id": f"ev-20261002-{next(_ids):012x}", "at": "2026-10-02T12:30:00+00:00", "schema": 1,
            "kind": "session", "run": run, "session": f"{run}-s1", "provider": "claude", "model": "m",
            "snapshot": snapshot, "snapshot_blob": f"blobs/{snapshot}", "patch": "", "cost": {"usd": 1.0, "estimated": False},
            "turns": 10, "seconds": 60.0, "status": "stop", "error": None, "violation": None, "claim": {"note": note}}


def tested(passed: bool = True) -> dict:
    return {"lint": passed, "tests": passed, "returncode": 0 if passed else 1, "lint_output": "", "test_output": "",
            "changed": ["src/inference_server/x.py"], "scratch_left_out": []}


def bench_ok(delta: float, verdict: str) -> dict:
    return {"verdict": "ok", "split": "seen",
            "metrics": {"ttft_p50_ms": {"base": 800, "new": 800 * (1 + delta / 100), "delta_pct": delta, "band_pct": 3, "verdict": verdict}}}


def refused(kind: str, snapshot: str) -> dict:
    return rec(kind, snapshot, {"verdict": "refused", "reason": NOT_WIRED[kind]})


SUBMIT_OK = {"verdict": "ok", "split": "heldout",
             "metrics": {"ttft_p50_ms": {"base": 800, "new": 760, "delta_pct": -5.0, "band_pct": 3, "verdict": "win"}}}


def run_feedback(lines: list[dict], run: str | None = None) -> dict:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "ledger.jsonl"
        p.write_text("".join(json.dumps(r) + "\n" for r in lines))
        fb = feedback(p, run_id=run)
        render_for_agent(fb)
        assert "should" not in json.dumps(fb["for_agent"]).lower()
        return fb


def valid_session() -> list[dict]:
    return [rec("test", A, tested()), rec("equiv", A, {"passed": True, "reasons": []}),
            rec("bench", A, bench_ok(-5.0, "win")),
            rec("test", B, tested()), rec("bench", B, bench_ok(-12.0, "win")),
            rec("submit", A, SUBMIT_OK), session_rec(A)]


def test_valid_session():
    fb = run_feedback(valid_session())
    i, f = fb["for_agent"]["integrity"], fb["for_agent"]["facts"]
    assert i["valid"], i
    assert f["submitted_snapshot"] == A and f["snapshots"][A]["tests_passed"] == 1 and f["snapshots"][A]["equiv_passed"] == 1
    assert f["best_measured_seen"]["ttft_p50_ms"]["snapshot"] == B                    # a fact, not advice
    assert f["submitted_best_measured_seen"]["ttft_p50_ms"]["delta_pct"] == -5.0
    assert f["submitted_result"]["ttft_p50_ms"]["verdict"] == "win" and f["sessions"] == 1


def test_run_filter():
    other = [rec("test", A, tested(False), run="run-other")]
    fb = run_feedback(valid_session() + other, run=RUN)
    assert fb["run"] == RUN and fb["for_agent"]["integrity"]["valid"]
    fb = run_feedback(valid_session() + other)                                          # default: the latest run
    assert fb["run"] == "run-other" and not fb["for_agent"]["integrity"]["valid"]


def test_submission_never_benchmarked():
    lines = [rec("test", A, tested()), rec("equiv", A, {"passed": True, "reasons": []}),
             rec("bench", B, bench_ok(-12.0, "win")), refused("bench", A), rec("submit", A, SUBMIT_OK)]
    i = run_feedback(lines)["for_agent"]["integrity"]
    assert not i["valid"] and [r.split(":")[0] for r in i["reasons"]] == ["submission_benchmarked"], i
    assert "1 refused" in i["reasons"][0]


def test_submission_equiv_failed():
    lines = [rec("test", A, tested()), rec("equiv", A, {"passed": False, "reasons": ["kl_mean 0.2 > 0.05"]}),
             rec("bench", A, bench_ok(-5.0, "win")), rec("submit", A, SUBMIT_OK)]
    fb = run_feedback(lines)
    i = fb["for_agent"]["integrity"]
    assert not i["valid"] and [r.split(":")[0] for r in i["reasons"]] == ["submission_equivalent"], i
    assert fb["for_agent"]["facts"]["snapshots"][A]["equiv_failed"] == 1


def test_only_refused_tools():
    lines = [refused("bench", A), refused("equiv", A), refused("submit", A), session_rec(A)]
    fb = run_feedback(lines)
    i, f = fb["for_agent"]["integrity"], fb["for_agent"]["facts"]
    assert not i["valid"] and i["reasons"] == ["submission_known: no completed submit (1 submit call(s), 1 refused)"], i
    assert f["tool_calls"] == {k: {"calls": 1, "refused": 1} for k in ("bench", "equiv", "submit")}, f["tool_calls"]
    assert f["snapshots"] == {} and f["submitted_snapshot"] is None and f["best_measured_seen"] == {}


def test_session_ledger_is_not_lab_ledger():
    from feedback.lab_verdict import ledger_file
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "session.jsonl"
        p.write_text(json.dumps({"kind": "session_start", "t": 1.0, "minute": 0, "tool_version": "0.1.0"}) + "\n")
        assert ledger_file(p) is None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
