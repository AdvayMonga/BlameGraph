"""The competitor timings' and the crossover's host-side logic (kernels/competitors.py, kernels/crossover.py): a wrong
competitor is reported and never timed; SASS is filtered to the kernels that matter; step times by batch from a
timeline; the faster path per batch and prompt length. The GPU parts run on the GPU host."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json  # noqa: E402


from kernels import competitors, crossover  # noqa: E402
from kernels.tasks import load  # noqa: E402


def test_a_wrong_competitor_is_reported_not_timed():
    t = load("toy_rmsnorm")
    right = lambda *xs: t.reference(*xs)          # noqa: E731
    wrong = lambda *xs: t.reference(*xs) * 2      # noqa: E731
    assert competitors.check(t, right, [(8, 256)], "cpu") is None
    assert "differs from the reference" in competitors.check(t, wrong, [(8, 256)], "cpu")
    assert "raised" in competitors.check(t, lambda *xs: 1 / 0, [(8, 256)], "cpu")


def test_sass_is_filtered_to_the_named_kernels(tmp_path, monkeypatch):
    dump = ("\n\tcode for sm_90\n\t\tFunction : _Z15rms_norm_kernelIfEvv\n\t/*0000*/ MOV R1 ;\n"
            "\n\t\tFunction : _Z9otherKernv\n\t/*0000*/ BRA 0x0 ;\n")
    monkeypatch.setattr(competitors.subprocess, "run", lambda *a, **k: type("P", (), {"stdout": dump})())
    got = competitors.sass("lib.so", r"rms_norm_kernel")
    assert "rms_norm_kernel" in got and "otherKern" not in got
    dump2 = dump + "\n\t\tFunction : _Z15rms_norm_kernelIdEvv\n\t/*0000*/ NOP ;\n"

    def run(argv, *a, **k):
        if "cu++filt" in argv[0]:
            return type("P", (), {"stdout": "void rms_norm_kernel<float>()\nvoid rms_norm_kernel<double>()\n"})()
        return type("P", (), {"stdout": dump2})()
    monkeypatch.setattr(competitors.subprocess, "run", run)
    only = competitors.sass("lib.so", r"rms_norm_kernel", {"void rms_norm_kernel<float>()"})
    assert "IfEvv" in only and "IdEvv" not in only                   # only the instantiation that ran


def test_our_moe_routing_in_vllms_layout_reads_the_same_rows_and_experts():
    import torch
    from kernels.tasks.moe_gemm import _routing
    g = torch.Generator().manual_seed(0)
    T, E, k, M = 6, 8, 2, 12
    experts = torch.stack([torch.randperm(E, generator=g)[:k] for _ in range(T)])           # real routing: k per token
    order = torch.argsort(experts.flatten(), stable=True)
    tok = (order // k).to(torch.int32)                                                    # our pair i = sorted position
    _, sorted_ids, expert_ids = _routing(M, E, T, 1, g, sorted_e=experts.flatten()[order])
    a = torch.randn(T, 4)
    p, top_k, gather, v_ids, e_ids, padded = competitors._vllm_routing(a, tok, sorted_ids, expert_ids, E, 4)
    assert top_k == k and not gather and int(padded) % 4 == 0
    our_e = experts.flatten()[order]                                                      # expert of our pair i
    vrow_e = e_ids.long().repeat_interleave(4)[: int(padded)]
    for slot, vp in enumerate(v_ids[: int(padded)].tolist()):
        if vp < M:
            i = int((p == vp).nonzero())
            assert vp // top_k == int(tok[i]) and int(vrow_e[slot]) == int(our_e[i])     # same row, same expert
    tok2, s2, e2 = _routing(M, E, T, 1, g)                                                # tokens with uneven pairs
    if len(set(torch.bincount(tok2.long(), minlength=T).tolist())) > 1:
        assert competitors._vllm_routing(a, tok2, s2, e2, E, 4)[2]                       # rows gathered instead


def test_ptx_per_implementation_and_the_weighted_ratio(tmp_path):
    (tmp_path / "h1").mkdir()
    (tmp_path / "h1" / "fused_moe_kernel.ptx").write_text("THEIRS")
    (tmp_path / "h1" / "_moe_gemm_kernel.ptx").write_text("OURS")
    assert competitors.ptx(tmp_path, r"_moe_gemm_kernel") == "OURS"
    assert competitors.ptx(tmp_path, r"fused_moe_kernel") == "THEIRS"
    r = competitors.weighted([{"calls": 3, "ours_over_competitor": 2.0}, {"calls": 1, "ours_over_competitor": 0.5}])
    assert abs(r - 2 ** 0.5) < 1e-9


def test_the_fused_moe_port_applies_every_edit_once_or_refuses():
    kernel = ("@triton.jit\ndef fused_moe_kernel(\n    a_ptr,\n    b_ptr,\n    USE_TD: tl.constexpr = False,\n):\n"
              "    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)\n"
              "    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)\n"
              "    token_mask = offs_token < num_valid_tokens\n"
              "    x = offs_token[:, None] // top_k * stride_am\n    y = offs_token[None, :] // top_k * stride_am\n")
    prog = "# EVOLVE-BLOCK-START\ndef moe_gemm(): pass\n# EVOLVE-BLOCK-END\n\nkernel = moe_gemm\n"
    out = competitors.port_fused_moe(prog, kernel, "@triton.jit\ndef write_zeros_to_output(): pass\n",
                                     {"1": {"BLOCK_SIZE_N": 64}}, "0.32.0")
    assert "tok_ptr" in out and "offs_row[:, None]" in out and "_moe_gemm_engine = moe_gemm" in out
    assert out.index("def fused_moe_kernel") < out.index("# EVOLVE-BLOCK-END") < out.index("kernel = moe_gemm")
    compile(out.replace("@triton.jit", ""), "port", "exec")
    try:
        competitors.port_fused_moe(prog, kernel.replace("    b_ptr,", "    bb_ptr,"), "", {}, "x")
        raise AssertionError("a changed kernel was ported")
    except ValueError as e:
        assert "no longer applies" in str(e)


def test_step_times_by_batch_and_the_faster_path(tmp_path):
    runs = {}
    for path, ms in (("megakernel", {1: 4.1, 2: 5.4, 4: 7.0}), ("kernels", {1: 5.6, 2: 5.7, 4: 6.0})):
        for b, v in ms.items():
            d = tmp_path / f"{path}-512-b{b}" / "x"
            d.mkdir(parents=True)
            lines = [{"kind": "step", "active": b, "dur_s": v / 1e3} for _ in range(3)]
            lines += [{"kind": "step", "active": 1, "dur_s": 9.0},                      # the run's tail
                      {"kind": "step", "active": b, "prefilling": 1, "dur_s": 1.0}, {"kind": "phase", "dur_s": 9}]
            (d / "events.jsonl").write_text("\n".join(json.dumps(x) for x in lines))
            runs[(path, 512, b)] = crossover.steps(tmp_path / f"{path}-512-b{b}")
    t = crossover.table(runs)["512"]
    assert t["1"]["faster"] == "megakernel" and t["2"]["faster"] == "megakernel" and t["4"]["faster"] == "kernels"
    assert t["2"]["megakernel_ms"] == 5.4 and t["2"]["megakernel_steps"] == 3        # tail and prefill left out


if __name__ == "__main__":
    import inspect
    import tempfile
    for name, fn in list(globals().items()):
        if name.startswith("test_") and "monkeypatch" not in inspect.signature(fn).parameters:
            fn(*[Path(tempfile.mkdtemp()) for _ in inspect.signature(fn).parameters]); print("ok", name)
