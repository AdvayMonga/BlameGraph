"""Physical ceilings from a model config and hardware peaks: Qwen3-30B-A3B's known counts, floors below any real
measurement, and the regime bounds. No GPU."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from validity import ceiling, peaks
from validity.ceiling import ModelSpec, Shape

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
    assert ceiling.decode_step_floor(m, H200, b + 1, 1024)["floor_s"] > 0.1
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
    assert ceiling.pct_of(0.004, 0.002, "lower") == pytest.approx(50)
    assert ceiling.pct_of(50, 200, "higher") == pytest.approx(25)
    assert ceiling.pct_of(None, 1, "lower") is None and ceiling.pct_of(1, 0, "higher") is None
    assert ceiling.ceiling_gain(0.4, 25) == pytest.approx(0.3) and ceiling.ceiling_gain(0.4, None) is None
    assert ceiling.ceiling_gain(0.4, 150) == 0


def test_peaks_measured_file_wins_over_spec_and_driver_must_match(tmp_path):
    assert peaks.load("NVIDIA H200").source == "spec" and peaks.load("Radeon") is None
    p = peaks.Peaks("NVIDIA H200", 900e12, 4.5e12, 141e9, "measured", driver="580.1")
    out = peaks.write(p, tmp_path)
    assert out.name == "nvidia-h200.json" and json.loads(out.read_text())["bw_bytes_s"] == 4.5e12
    assert peaks.load("NVIDIA H200", "580.1", tmp_path).source == "measured"
    assert peaks.load("NVIDIA H200", None, tmp_path).source == "measured"
    assert peaks.load("NVIDIA H200", "581.0", tmp_path).source == "spec"   # another driver: back to the sheet


def test_measure_needs_cuda():
    pytest.importorskip("torch")
    import torch
    if torch.cuda.is_available():
        pytest.skip("a GPU is present; the microbench is checked on the VM")
    with pytest.raises(RuntimeError):
        peaks.measure(0.1)
    assert not (Path(peaks.PEAKS_DIR) / "radeon.json").exists()
