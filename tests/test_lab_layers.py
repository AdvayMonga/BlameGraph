"""Per-layer facts from a profile bundle: shares of the window from the timeline and the kernel table, the decode step
against its floor, and a missing ceiling that says why. A bundle is written by hand; a real H200 one is used when
it is on this disk."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from lab import layers

REAL = Path(__file__).resolve().parents[1] / "data" / "engine-h200-20261017" / "session6" / "prof6-b1"


def bundle(tmp_path: Path, kernel_s: float = 0.4, with_steps: bool = True) -> Path:
    """A 2 s window: 10 decode steps of 0.15 s (batch 4), admit 0.02 s, evict 0.01 s, a prefill of 0.2 s; the rest
    (0.27 s) outside steps. One warmup step before the window, which must not count."""
    b = tmp_path / "b"
    b.mkdir(parents=True)
    t0 = 1000.0
    (b / "meta.json").write_text(json.dumps({
        "window": {"start": t0, "end": t0 + 2.0, "wall_s": 2.0}, "max_tokens": 64,
        "settings": {"max_batch_size": 8}, "workload": {"kind": "synthetic", "prompt_len": 500},
        "gpu": {"window_start": {"name": "NVIDIA H200", "driver_version": "580.173.02"}}}))
    (b / "kernels.json").write_text(json.dumps({"total_us": kernel_s * 1e6, "launches": 1234, "kernels": []}))
    (b / "stats.json").write_text(json.dumps({"window_delta": {"decode_steps": 10 if with_steps else 0, "kv_admit_blocked": 3,
                                                               "total_admitted": 4, "total_completed": 4}}))
    ev = [{"ts": t0 - 5, "kind": "step", "dur_s": 9.0, "active": 1, "pending": 0},            # warmup: outside
          {"ts": t0 - 5, "kind": "phase", "name": "decode", "dur_s": 9.0}]
    if with_steps:
        ev += [{"ts": t0 + 0.1, "kind": "enqueue", "prompt_tokens": 600, "max_tokens": 64},
               {"ts": t0 + 0.1, "kind": "enqueue", "prompt_tokens": 400, "max_tokens": 64},
               {"ts": t0 + 0.2, "kind": "phase", "name": "admit", "dur_s": 0.02},
               {"ts": t0 + 0.4, "kind": "phase", "name": "prefill_chunk", "dur_s": 0.2},
               {"ts": t0 + 0.4, "kind": "phase", "name": "prefill_mixed", "dur_s": 0.19},      # nested: not counted
               {"ts": t0 + 0.4, "kind": "phase", "name": "evict", "dur_s": 0.01},
               {"ts": t0 + 0.4, "kind": "step", "dur_s": 0.23, "active": 0, "prefilling": 2, "pending": 1}]
        for k in range(10):
            ts = t0 + 0.5 + 0.15 * (k + 1)
            ev += [{"ts": ts, "kind": "phase", "name": "decode", "dur_s": 0.15},
                   {"ts": ts, "kind": "step", "dur_s": 0.15, "active": 4, "prefilling": 0, "pending": 0}]
    (b / "events.jsonl").write_text("\n".join(json.dumps(e) for e in ev) + "\n")
    return b


def test_shares_come_from_the_window_only_and_add_up(tmp_path):
    out = layers.from_bundle(bundle(tmp_path))
    assert out["window_s"] == 2.0 and out["scheduler_steps"] == 11                     # the warmup step is out
    assert out["kernels"]["share"] == pytest.approx(0.2) and out["kernels"]["launches"] == 1234
    me = out["model_execution"]
    assert me["prefill_phase_s"] == pytest.approx(0.2) and me["decode_phase_s"] == pytest.approx(1.5)
    assert me["share"] == pytest.approx((0.2 + 1.5 - 0.4) / 2) and me["decode_step_mean_s"] == pytest.approx(0.15)
    rt = out["runtime"]
    assert rt["share"] == pytest.approx(0.015) and rt["gpu_idle_share"] == pytest.approx(0.8)
    assert rt["batch_mean"] == 4 and rt["batch_fill"] == 0.5 and rt["kv_admit_blocked"] == 3
    assert rt["steps_idle_with_pending"] == 0
    assert out["serving"]["share"] == pytest.approx((2.0 - 0.23 - 1.5) / 2)
    assert "ceiling_missing" in rt and "ceiling_missing" in out["serving"] and "basis" in out["kernels"]


def test_the_decode_floor_needs_a_target_and_peaks_and_says_so(tmp_path, monkeypatch):
    from lab import target
    b = bundle(tmp_path)
    out = layers.from_bundle(b)
    assert out["model_execution"]["pct_of_ceiling"] is None and "no target" in out["model_execution"]["ceiling_missing"]
    out = layers.from_bundle(b, target.load())
    me = out["model_execution"]
    if me["pct_of_ceiling"] is None:                      # no peaks file for the H200 in this checkout, or no HF config
        assert me["ceiling_missing"]
    else:
        assert 0 < me["pct_of_ceiling"] <= 100 and "batch 4" in me["ceiling_basis"] and "context 532" in me["ceiling_basis"]
    assert out["kernels"]["pct_of_ceiling"] is None and "gate" in out["kernels"]["ceiling_missing"]
    assert layers.from_bundle(bundle(tmp_path / "x", with_steps=False))["model_execution"]["ceiling_missing"] == "no decode steps in the window"


def test_kernel_distance_from_gates(tmp_path):
    gate = {"artifacts": {"timings": json.dumps([{"pct_of_roofline": 10.0, "weight": 3}, {"pct_of_roofline": 30.0, "weight": 1}])}}
    out = layers.from_bundle(bundle(tmp_path), gates={"paged_decode": gate, "other": {"artifacts": {}}})
    k = out["kernels"]
    assert k["pct_of_ceiling"] == pytest.approx(15.0) and k["pct_of_roofline_by_task"] == {"paged_decode": 15.0, "other": None}


def test_a_bundle_without_a_window_is_none(tmp_path):
    (tmp_path / "meta.json").write_text("{}")
    assert layers.from_bundle(tmp_path) is None
    assert layers.from_bundle(tmp_path / "nowhere") is None


@pytest.mark.skipif(not REAL.exists(), reason="the 2026-10-17 H200 bundle is local only")
def test_the_real_h200_bundle():
    b = next(REAL.iterdir())
    out = layers.from_bundle(b)
    shares = {k: out[k]["share"] for k in ("kernels", "model_execution", "runtime", "serving")}
    assert all(0 <= v <= 1 for v in shares.values()) and out["model_execution"]["decode_steps"] == 64
    assert out["runtime"]["batch_mean"] == 1 and out["step_share"] <= 1.0001
