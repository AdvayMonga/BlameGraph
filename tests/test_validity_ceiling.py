"""Physical ceilings from a model config and hardware peaks: Qwen3-30B-A3B's known counts, floors below any real
measurement, and the regime bounds. No GPU."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from validity import ceiling, peaks  # noqa: E402
from validity.ceiling import ModelSpec, Shape  # noqa: E402

QWEN3_30B_A3B = {"num_hidden_layers": 48, "hidden_size": 2048, "num_attention_heads": 32, "num_key_value_heads": 4,
                 "head_dim": 128, "vocab_size": 151936, "intermediate_size": 6144, "num_experts": 128,
                 "num_experts_per_tok": 8, "moe_intermediate_size": 768, "tie_word_embeddings": False}
LLAMA_8B = {"num_hidden_layers": 32, "hidden_size": 4096, "num_attention_heads": 32, "num_key_value_heads": 8,
            "vocab_size": 128256, "intermediate_size": 14336}
H200 = peaks.Peaks("NVIDIA H200", 989e12, 4.8e12, 141e9, "spec")


def test_qwen3_moe_parameter_counts_match_the_model_card():
    m = ModelSpec.from_hf_config(QWEN3_30B_A3B, "qwen")
    assert 2.9e9 < m.active_params < 3.4e9            # "3B activated"
    assert 29e9 < m.total_params < 31e9               # "30B total"
    assert m.kv_bytes_per_token("bf16") == 2 * 48 * 4 * 128 * 2
    assert m.experts_touched(1) == 8 and 120 < m.experts_touched(1000) <= 128


def test_dense_llama_counts():
    m = ModelSpec.from_hf_config(LLAMA_8B, "llama")
    assert 7.5e9 < m.active_params == m.total_params - LLAMA_8B["vocab_size"] * 4096 < 8.5e9
    assert m.head_dim == 128 and m.n_experts == 0


def test_decode_floor_is_memory_bound_at_small_batch_and_grows_with_batch():
    m = ModelSpec.from_hf_config(QWEN3_30B_A3B)
    f1 = ceiling.decode_step_floor(m, H200, 1, 1024)
    f64 = ceiling.decode_step_floor(m, H200, 64, 1024)
    assert f1["bound"] == "memory" and 0.001 < f1["floor_s"] < 0.003     # ~6 GB of weights at 4.8 TB/s
    assert f64["floor_s"] > f1["floor_s"] and f64["floor_s"] < 64 * f1["floor_s"]


def test_prefill_floor_and_goodput_ceiling():
    m = ModelSpec.from_hf_config(QWEN3_30B_A3B)
    p = ceiling.prefill_floor(m, H200, 4096)
    assert p["floor_s"] > ceiling.prefill_floor(m, H200, 512)["floor_s"]
    b = ceiling.max_batch_under(m, H200, 0.1, 1024)
    assert b > 64 and ceiling.decode_step_floor(m, H200, b, 1024)["floor_s"] <= 0.1
    assert (b == ceiling.capacity_batch(m, H200, 1024)
            or ceiling.decode_step_floor(m, H200, b + 1, 1024)["floor_s"] > 0.1)
    assert ceiling.max_batch_under(m, H200, 1e-6, 1024) == 0 and ceiling.tokens_per_s_ceiling(m, H200, 1e-6, 1024) == 0


def test_regime_ceilings_by_objective():
    m = ModelSpec.from_hf_config(QWEN3_30B_A3B)
    shape = Shape(600, 2000, 200, concurrency=1)
    lim = {"ttft_s": 2.0, "tpot_s": 0.1}
    tpot = ceiling.regime_ceiling("tpot_p99_s", m, H200, shape, lim)
    ttft = ceiling.regime_ceiling("ttft_p99_s", m, H200, shape, lim)
    good = ceiling.regime_ceiling("goodput_req_per_s", m, H200, shape, lim)
    assert tpot["better"] == "lower" and tpot["ceiling"] < 0.01
    assert ttft["better"] == "lower" and "p99 prompt" in ttft["basis"]
    assert good["better"] == "higher" and good["ceiling"] > 10
    assert ceiling.regime_ceiling("speedup", m, H200, shape, lim) is None


def test_pct_of_and_ceiling_gain():
    assert math.isclose(ceiling.pct_of(0.004, 0.002, "lower"), 50)
    assert math.isclose(ceiling.pct_of(50, 200, "higher"), 25)
    assert ceiling.pct_of(None, 1, "lower") is None and ceiling.pct_of(1, 0, "higher") is None
    assert math.isclose(ceiling.ceiling_gain(0.4, 25), 0.3) and ceiling.ceiling_gain(0.4, None) is None
    assert ceiling.ceiling_gain(0.4, 150) == 0


def test_peaks_measured_file_wins_over_spec_and_driver_must_match():
    import tempfile
    tmp_path = Path(tempfile.mkdtemp())
    empty = Path(tempfile.mkdtemp())
    assert peaks.load("NVIDIA H200", peaks_dir=empty).source == "spec" and peaks.load("Radeon", peaks_dir=empty) is None
    p = peaks.Peaks("NVIDIA H200", 900e12, 4.5e12, 141e9, "measured", driver="580.1")
    out = peaks.write(p, tmp_path)
    assert out.name == "nvidia-h200-580-1.json" and json.loads(out.read_text())["bw_bytes_s"] == 4.5e12
    assert peaks.load("NVIDIA H200", "580.1", tmp_path).source == "measured"
    assert peaks.load("NVIDIA H200", None, tmp_path).source == "measured"
    assert peaks.load("NVIDIA H200", "581.0", tmp_path).source == "spec"   # another driver: back to the sheet


def test_measure_needs_cuda():
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        return                                   # the microbench itself is checked on the VM
    try:
        peaks.measure(0.1)
    except RuntimeError:
        return
    raise AssertionError("measure() must refuse without CUDA")


def test_capacity_caps_the_batch_and_goodput_pays_for_prefill():
    m = ModelSpec.from_hf_config(QWEN3_30B_A3B)
    cap = ceiling.capacity_batch(m, H200, 1024)
    assert 0 < cap < 1200                                   # ~80 GB free for KV at ~100 KB/token
    assert ceiling.max_batch_under(m, H200, 10.0, 1024) == cap   # a loose TPOT limit stops at HBM, not at 65k
    lim = {"ttft_s": 2.0, "tpot_s": 0.1}
    short_out = ceiling.regime_ceiling("goodput_req_per_s", m, H200, Shape(8000, 9000, 50, uncached_prompt_tokens_mean=8000), lim)
    long_out = ceiling.regime_ceiling("goodput_req_per_s", m, H200, Shape(8000, 9000, 2000, uncached_prompt_tokens_mean=8000), lim)
    no_prefill = ceiling.tokens_per_s_ceiling(m, H200, 0.1, 8025) / 50
    assert short_out["ceiling"] < no_prefill and short_out["bound"] == "prefill" and long_out["bound"] == "decode"
    unknown = ceiling.regime_ceiling("goodput_req_per_s", m, H200, Shape(8000, 9000, 50), lim)
    assert unknown["ceiling"] == pytest_approx(no_prefill) and "may have been cached" in unknown["basis"]
    half = ceiling.regime_ceiling("goodput_req_per_s", m, H200, Shape(8000, 9000, 50, uncached_prompt_tokens_mean=4000), lim)
    assert short_out["ceiling"] < half["ceiling"] < unknown["ceiling"]
    assert ceiling.regime_ceiling("goodput_sessions_per_s", m, H200, Shape(600, 900, 200), lim) is None


def pytest_approx(x, rel=1e-9):
    class A:
        def __eq__(self, y):
            return math.isclose(x, y, rel_tol=rel)
    return A()


def test_dtype_aliases():
    assert ceiling.dtype("bfloat16") == "bf16" and ceiling.dtype("FP8") == "fp8"
    try:
        ceiling.dtype("int3")
    except ValueError:
        return
    raise AssertionError("unknown dtype accepted")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
