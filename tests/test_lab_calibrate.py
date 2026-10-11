"""Calibrated results: ceilings and reference only under a matching fingerprint, kernel shares, and the baseline
record carrying its context. No GPU."""
from __future__ import annotations

import json

import pytest

from lab import calibrate, target
from tests.test_validity_ceiling import QWEN3_30B_A3B

GPU = {"name": "NVIDIA H200", "driver_version": "580.1", "clocks.sm": "1980"}


def rows(regime="single_stream", n=20, prompt=600, out=100):
    return [{"regime": regime, "status": "ok", "scheduled_s": i * 1.0, "e2e_s": 0.9, "prompt_tokens": prompt,
             "n_tokens": out} for i in range(n)]


def results(regime="single_stream", objective="tpot_p99_s", value=0.02, better="lower"):
    return [{"regime": regime, "objective": objective, "value": value, "better": better, "valid": True,
             "invalid_reasons": []}]


@pytest.fixture
def t(monkeypatch):
    monkeypatch.setattr(calibrate, "model_config", lambda model: QWEN3_30B_A3B)
    return target.load()


def test_results_get_ceiling_pct_and_shape(t):
    cal = calibrate.calibrate_results(results(), rows(), t, GPU, "short", "seen")["single_stream"]
    assert cal["value"] == 0.02 and cal["ceiling"] < 0.02 and 0 < cal["pct_of_ceiling"] < 100
    assert cal["bound"] == "memory" and cal["peaks_source"] == "spec" and cal["shape"]["prompt_tokens_mean"] == 600
    assert cal["fingerprint"]["gpu"] == "NVIDIA H200" and cal["fingerprint"]["w_dtype"] == "bf16"
    assert cal["reference"] is None and "no reference" in cal["reference_missing"]


def test_missing_inputs_are_named_not_guessed(t, monkeypatch):
    no_gpu = calibrate.calibrate_results(results(), rows(), t, None, "short", "seen")["single_stream"]
    assert no_gpu["ceiling"] is None and "peaks" in no_gpu["ceiling_missing"]
    no_rows = calibrate.calibrate_results(results(), [], t, GPU, "short", "seen")["single_stream"]
    assert no_rows["ceiling"] is None and "prompt token" in no_rows["ceiling_missing"]
    ratio = calibrate.calibrate_results(results("bursty", "speedup", 1.2, "higher"), rows("bursty"), t, GPU,
                                        "short", "seen")["bursty"]
    assert ratio["ceiling"] is None and "no physical bound" in ratio["ceiling_missing"]
    monkeypatch.setattr(calibrate, "model_config", lambda model: None)
    assert "model config" in calibrate.calibrate_results(results(), rows(), t, GPU, "short", "seen")["single_stream"]["ceiling_missing"]


def test_reference_only_under_the_same_fingerprint(t, tmp_path, monkeypatch):
    monkeypatch.setattr(calibrate, "REFERENCE_DIR", tmp_path)
    fp = calibrate.fingerprint(t, GPU, "short", "seen")
    calibrate.write_reference(t, "single_stream", results(value=0.015)[0], fp, rows())
    same = calibrate.calibrate_results(results(), rows(), t, GPU, "short", "seen")["single_stream"]
    assert same["reference"] == 0.015 and same["pct_of_reference"] == pytest.approx(75)
    other_gpu = calibrate.calibrate_results(results(), rows(), t, {**GPU, "name": "NVIDIA H100"}, "short", "seen")["single_stream"]
    assert other_gpu["reference"] is None and "gpu" in other_gpu["reference_missing"]
    full = calibrate.calibrate_results(results(), rows(), t, GPU, "full", "seen")["single_stream"]
    assert full["reference"] is None and "tier" in full["reference_missing"]
    stored = json.loads((tmp_path / t.name / "single_stream.json").read_text())
    assert stored["server"] == "reference" and stored["shape"]["output_tokens_mean"] == 100


def test_kernel_shares_and_ncu_rates(tmp_path):
    b = tmp_path / "b1"
    b.mkdir()
    (b / "meta.json").write_text(json.dumps({"window": {"wall_s": 2.0}}))
    (b / "kernels.json").write_text(json.dumps({"kernels": [
        {"name": "decode_attn", "count": 10, "total_us": 1_000_000}, {"name": "gemm", "count": 5, "total_us": 500_000}]}))
    k = calibrate.calibrate_kernels(b, GPU)
    assert k["kernels"][0]["share_of_window"] == pytest.approx(0.5) and k["kernel_share_total"] == pytest.approx(0.75)
    assert calibrate.calibrate_kernels(tmp_path, GPU) is None
    csvp = tmp_path / "metrics.csv"
    csvp.write_text('"ID","Kernel Name","gpu__time_duration.sum","dram__bytes.sum.per_second"\n'
                    '"","","us","Tbyte/s"\n"0","gemm","12.5","2.4"\n"1","gemm","12.0","2.0"\n')
    n = calibrate.calibrate_ncu(csvp, GPU)
    assert [round(x["pct_of_bandwidth_peak"]) for x in n["launches"]] == [50, 42] and n["launches"][0]["duration_unit"] == "us"
    assert calibrate.calibrate_ncu(csvp, None)["launches"][0]["pct_of_bandwidth_peak"] is None


def test_facts_for_the_brief(t):
    f = calibrate.facts(t, GPU)
    assert f["peaks"]["source"] == "spec" and f["model"]["active_params"] > 2.9e9
    assert 0.001 < f["floors_s"]["decode_step_batch1_ctx1k"] < 0.003
    assert calibrate.facts(t, None)["peaks"] is None
