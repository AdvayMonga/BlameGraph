"""Validator flips and feedback on synthetic sessions. Run: python -m pytest tests  (or python tests/test_feedback.py)"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from blamegraph.feedback import feedback  # noqa: E402
from blamegraph.session import Ledger  # noqa: E402
from blamegraph.validate import validate  # noqa: E402

SERVER = {"pid": 1, "started_at": 1.0, "config_changed_after_start": False}


def metrics(ttft: float) -> dict:
    return {"profiles": {"burst": {"ttft": {"p50": ttft}, "tpot": {"p50": 0.02}, "request_throughput_req_per_s": 1.0,
                                   "failure_rate": 0.0, "generation_throughput_tokens_per_s": 100}}, "quality_check": {"pass": True}}


def session(td: Path, measure_shipped: bool = True, tamper: bool = False, impossible: bool = False) -> Ledger:
    cfg, grader = td / "start_server.sh", td / "evaluate.py"
    grader.write_text("# grader\n")
    led = Ledger.open(td / "session.jsonl")
    led.start_session({"scenario": "inference_scenario_a_input_heavy"}, [grader])
    cfg.write_text("vllm serve m --max-num-seqs 64\n")
    led.record_measure(cfg, "full", [], metrics(0.20), True, server=SERVER)          # v0 measured
    cfg.write_text("vllm serve m --max-num-seqs 256\n")
    if measure_shipped:
        led.record_measure(cfg, "full", [], metrics(0.001 if impossible else 0.30), True, server={**SERVER, "started_at": 2.0})
    if tamper:
        grader.write_text("# tampered\n")
    led.record_submission(cfg, [grader])
    return led


def test_valid_session():
    with tempfile.TemporaryDirectory() as d:
        led = session(Path(d))
        assert validate(led).valid
        fb = feedback(Path(d) / "session.jsonl")
        f = fb["for_agent"]["facts"]
        assert fb["for_agent"]["integrity"]["valid"] and f["measurements"] == 2 and f["shipped_comparable_measurements"] == 1
        assert f["best_measured_config"] == "v0" and f["shipped_config"] == "v1"   # shipped the slower one: a fact, not advice
        assert "should" not in json.dumps(fb["for_agent"]).lower()


def test_unmeasured_submission_is_invalid():
    with tempfile.TemporaryDirectory() as d:
        v = validate(session(Path(d), measure_shipped=False))
        assert not v.valid and any(r.startswith("submission") for r in v.reasons), v.reasons


def test_grader_edit_is_invalid():
    with tempfile.TemporaryDirectory() as d:
        v = validate(session(Path(d), tamper=True))
        assert not v.valid and any(r.startswith("grader_untouched") for r in v.reasons), v.reasons


def test_impossible_metrics_are_invalid():
    with tempfile.TemporaryDirectory() as d:
        v = validate(session(Path(d), impossible=True))
        assert not v.valid and any(r.startswith("metrics_physical") for r in v.reasons), v.reasons


def test_stale_measurement_does_not_count():
    with tempfile.TemporaryDirectory() as d:
        td = Path(d); cfg, grader = td / "start_server.sh", td / "evaluate.py"
        grader.write_text("# grader\n"); led = Ledger.open(td / "session.jsonl"); led.start_session({"scenario": "A"}, [grader])
        cfg.write_text("vllm serve m --max-num-seqs 64\n")
        led.record_measure(cfg, "full", [], metrics(0.2), True, server=SERVER)
        cfg.write_text("vllm serve m --max-num-seqs 256\n")   # edited, server not restarted
        led.record_measure(cfg, "full", [], metrics(0.2), True, server={**SERVER, "config_changed_after_start": True})
        led.record_submission(cfg, [grader])
        v = validate(led)
        assert not v.valid and any(r.startswith("submission_measured") for r in v.reasons), v.reasons


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
