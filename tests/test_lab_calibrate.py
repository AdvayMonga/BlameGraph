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
def t(monkeypatch, tmp_path):
    monkeypatch.setattr(calibrate, "model_config", lambda model: QWEN3_30B_A3B)
    monkeypatch.setattr(calibrate, "REFERENCE_DIR", tmp_path / "no-reference")      # the committed ones are real
    return target.load()


def test_results_get_ceiling_pct_and_shape(t):
    cal = calibrate.calibrate_results(results(), rows(), t, GPU, "short", "seen")["single_stream"]
    assert cal["value"] == 0.02 and cal["ceiling"] < 0.02 and 0 < cal["pct_of_ceiling"] < 100
    assert cal["bound"] == "memory" and cal["peaks_source"] == "spec" and cal["shape"]["prompt_tokens_mean"] == 600
    assert cal["fingerprint"]["gpu"] == "NVIDIA H200" and cal["fingerprint"]["w_dtype"] == "bf16"
    assert cal["reference"] is None and "no reference measurement" in cal["reference_missing"]


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


def test_nsys_kernel_and_api_time_against_the_window(tmp_path):
    (tmp_path / "bundle" / "b1").mkdir(parents=True)
    (tmp_path / "bundle" / "b1" / "meta.json").write_text(json.dumps({"window": {"wall_s": 2.0}}))
    (tmp_path / "stats_cuda_gpu_kern_sum.csv").write_text(
        "Time (%),Total Time (ns),Instances,Avg (ns),Name\n"
        "25.0,250000000,100,\"2,500,000\",small\n75.0,750000000,300,2500000,moe_gemm\n")
    (tmp_path / "stats_cuda_gpu_mem_time_sum.csv").write_text(
        "Time (%),Total Time (ns),Count,Operation\n100.0,100000000,4,[CUDA memcpy Host-to-Device]\n")
    (tmp_path / "stats_cuda_api_sum.csv").write_text(
        "Time (%),Total Time (ns),Num Calls,Name\n90.0,900000000,36,cudaStreamSynchronize\n")
    n = calibrate.calibrate_nsys(tmp_path, ranged=True)
    assert n["kernel_s"] == pytest.approx(1.0) and n["launches"] == 400 and n["kernel_share_of_window"] == pytest.approx(0.5)
    assert n["kernels"][0]["name"] == "moe_gemm" and n["kernels"][0]["share_of_kernel_time"] == pytest.approx(0.75)
    assert n["kernels"][0]["share_of_window"] == pytest.approx(0.375) and n["memcpy_s"] == pytest.approx(0.1)
    assert n["cuda_api"] == [{"name": "cudaStreamSynchronize", "total_s": pytest.approx(0.9), "count": 36}]
    assert calibrate.calibrate_nsys(tmp_path, ranged=True)["window_s"] == 2.0
    free = calibrate.calibrate_nsys(tmp_path, ranged=False)
    assert free["window_s"] is None and free["kernel_share_of_window"] is None and "capture-range" in free["window_missing"]
    assert calibrate.calibrate_nsys(tmp_path / "bundle", ranged=True) is None


def test_pyspy_samples_per_thread_and_function(tmp_path):
    (tmp_path / "bundle" / "b1").mkdir(parents=True)
    frames = [{"name": "run", "file": "s.py", "line": 1}, {"name": "step", "file": "s.py", "line": 9},
              {"name": "wait", "file": "t.py", "line": 3}]
    (tmp_path / "bundle" / "b1" / "pyspy.speedscope.json").write_text(json.dumps({"shared": {"frames": frames}, "profiles": [
        {"type": "sampled", "name": "Thread 1 \"MainThread\"", "samples": [[0, 1], [0, 1], [0, 2]], "weights": [1, 1, 1]},
        {"type": "sampled", "name": "Thread 2 \"scheduler\"", "samples": [[2]], "weights": [1]}]}))
    p = calibrate.calibrate_pyspy(tmp_path)
    assert p["samples"] == 4 and p["threads"][0] == {"name": "Thread 1 \"MainThread\"", "samples": 3, "share": 0.75}
    assert [(f["function"], f["share"]) for f in p["self"]] == [("step", 0.5), ("wait", 0.5)]
    assert p["total"][0]["function"] == "run" and p["total"][0]["share"] == 0.75
    assert calibrate.calibrate_pyspy(tmp_path / "bundle") is None


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


def test_perf_reference_drops_determinism_switches(t):
    import dataclasses
    srv = calibrate.perf_reference(t)
    assert "VLLM_BATCH_INVARIANT" not in srv.env and srv.launch == t.reference.launch
    sg = dataclasses.replace(t, reference=dataclasses.replace(t.reference, launch="python -m sglang.launch_server "
                                                              "--enable-deterministic-inference --port {port}"))
    assert "--enable-deterministic-inference" not in calibrate.perf_reference(sg).launch


def test_calibration_never_raises_and_invalid_values_say_why(t, monkeypatch):
    bad = calibrate.calibrate_results([{"regime": "x"}], [], t, GPU, "short", "seen")
    assert "calibration failed" in bad["error"]
    inv = calibrate.calibrate_results([{**results()[0], "valid": False, "invalid_reasons": ["3 silent drops"]}], rows(),
                                      t, GPU, "short", "seen")["single_stream"]
    assert inv["value"] is None and "3 silent drops" in inv["value_missing"]


def test_dollar_facts_at_the_gpu_price_or_why_not(t, monkeypatch):
    summary = {"good_req_per_s": 2.0, "good_output_tok_per_s": 200.0}
    sat = [{**results("saturated", "goodput_req_per_s", 2.0, "higher")[0], "summary": summary}]
    monkeypatch.delenv("LAB_GPU_USD_PER_HOUR", raising=False)
    none = calibrate.calibrate_results(sat, rows("saturated"), t, GPU, "short", "seen")["saturated"]
    assert "usd_per_request" not in none and "no GPU price" in none["cost_missing"]
    monkeypatch.setenv("LAB_GPU_USD_PER_HOUR", "3.6")
    cal = calibrate.calibrate_results(sat, rows("saturated"), t, GPU, "short", "seen")["saturated"]
    assert cal["usd_per_request"] == pytest.approx(0.0005) and cal["usd_per_m_output_tokens"] == pytest.approx(5.0)
    spm = [{**results("shared_prefix_multi_turn", "goodput_sessions_per_s", 0.5, "higher")[0], "summary": summary}]
    assert calibrate.calibrate_results(spm, [], t, GPU, "short", "seen")["shared_prefix_multi_turn"]["usd_per_session"] \
        == pytest.approx(0.002)
    batch = [{**results("batch", "usd_per_m_output_tokens", 5.0, "lower")[0], "summary": summary}]
    b = calibrate.calibrate_results(batch, rows("batch"), t, GPU, "short", "seen")["batch"]
    assert 0 < b["ceiling"] < 5.0 and 0 < b["pct_of_ceiling"] < 100 and "TPOT <= 0.25" in b["ceiling_basis"]
    monkeypatch.delenv("LAB_GPU_USD_PER_HOUR")
    b = calibrate.calibrate_results([{**batch[0], "value": None, "value_missing": "no GPU price: x"}], rows("batch"),
                                    t, GPU, "short", "seen")["batch"]
    assert b["value_missing"] == "no GPU price: x" and "no GPU price" in b["ceiling_missing"]


def test_ncu_units_and_odd_cells(tmp_path):
    p = tmp_path / "metrics.csv"
    p.write_text('"ID","Kernel Name","gpu__time_duration.sum","dram__bytes.sum.per_second"\n'
                 '"","","usecond","Gbyte/second"\n"0","gemm","1,250.5","2,400"\n"1","gemm","n/a",""\n')
    n = calibrate.calibrate_ncu(p, GPU)["launches"]
    assert n[0]["duration"] == 1250.5 and round(n[0]["pct_of_bandwidth_peak"]) == 50
    assert n[1]["duration"] is None and n[1]["pct_of_bandwidth_peak"] is None


def test_uncached_prompt_tokens_come_from_the_server_or_not_at_all():
    full = [{**r, "cached_tokens": 450} for r in rows()]
    assert calibrate.shape_from_rows(full).uncached_prompt_tokens_mean == 150
    partial = full[:5] + rows()[5:]
    assert calibrate.shape_from_rows(partial).uncached_prompt_tokens_mean is None


def test_heldout_aggregate_context_uses_the_seen_ceiling_only():
    from lab import ledger
    from lab.evaltools import aggregate, heldout_context
    seen = {"ceiling": 20.0}
    m = {**aggregate({"value": 5.0}, 10.0, "higher", 3.0), **heldout_context(seen, {"value": 5.0}, 10.0, "higher")}
    assert m["pct_of_ceiling_base"] == pytest.approx(25) and m["pct_of_ceiling_new"] == pytest.approx(27.5)
    assert "seen-split" in m["ceiling_from"] and "held-out" in m["reference_missing"]
    none = heldout_context({"ceiling_missing": "no peaks"}, {"value": 5.0}, None, "higher")
    assert none["ceiling"] is None and none["pct_of_ceiling_new"] is None and none["ceiling_from"] == "no peaks"
    import tempfile
    from pathlib import Path
    ledger.append({"kind": "submit", "run": "r", "config": {"split": "heldout"}, "metrics": {"saturated": m}},
                  Path(tempfile.mkdtemp()))                     # the held-out writer accepts the context fields


def test_brief_ceilings_need_a_measured_gpu(t, tmp_path):
    from lab import ledger
    from lab.evaltools import _ceiling_facts
    assert "no measurement" in _ceiling_facts(t, tmp_path)["missing"]
    ledger.append({"kind": "bench", "run": "r", "result": {"gpu": GPU}}, tmp_path)
    f = _ceiling_facts(t, tmp_path)
    assert f["peaks"]["gpu"] == "NVIDIA H200" and f["floors_s"]["decode_step_batch1_ctx1k"] > 0
