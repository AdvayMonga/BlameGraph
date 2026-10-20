"""The MoE down projection as the example engine runs it (inference-server #112, fp8 by default since #122):
`moe_gemm_scatter(a, w, sorted_ids, expert_ids, M, tok, pair_w, out, w_scale=)` adds, per (token, expert) pair in
expert-sorted order, pair_w[i] * (a[i] @ dequant(w[expert(i)]).T) into out[tok[i]], `out` [T, N] fp32 zeroed by the
caller. The task's call zeroes a fresh `out` (same for the baseline and every candidate) and returns it.

Shape: (M pairs, K, N, E experts, T tokens). Inputs: a [M, K] bf16 (the pairs' activated rows), w [E, N, K] fp8 e4m3,
w_scale [E, N] fp32, sorted_ids / expert_ids from `align` (block_m 16), tok [M] int32, pair_w [M] fp32, out0 [T, N]
fp32 (the output's shape). Captured calls carry the engine's routing (sorted_ids, expert_ids, tok)."""
from __future__ import annotations

import torch

from kernels.tasks import Task, randn
from kernels.tasks.moe_gemm import _block_m, align
from kernels.tasks.moe_gemm_fp8 import dequantize, quantize


def _call(fn, a, w, w_scale, sorted_ids, expert_ids, tok, pair_w, out0):
    out = torch.zeros_like(out0)
    fn(a, w, sorted_ids, expert_ids, a.shape[0], tok, pair_w, out, block_m=_block_m(sorted_ids, expert_ids),
       w_scale=w_scale)
    return out


def reference(a, w, w_scale, sorted_ids, expert_ids, tok, pair_w, out0):
    """sum over pairs of pair_w * row @ W_e^T into each pair's token, in fp32."""
    M, E = a.shape[0], w.shape[0]
    bm = _block_m(sorted_ids, expert_ids)
    expert = torch.full((sorted_ids.numel(),), E, dtype=torch.long, device=a.device)
    expert[: expert_ids.numel() * bm] = expert_ids.long().repeat_interleave(bm)
    real = (sorted_ids < M) & (expert < E)
    ids, ex = sorted_ids[real].long(), expert[real]
    wd, out = dequantize(w, w_scale), torch.zeros_like(out0)
    for e in ex.unique().tolist():
        sel = ids[ex == e]
        out.index_add_(0, tok[sel].long(), pair_w[sel, None] * (a[sel].float() @ wd[e].T))
    return out


def _with_values(M, K, N, E, T, dtype, device, g, sorted_ids, expert_ids, tok):
    a = randn((M, K), g, device, dtype)
    q, s = quantize(randn((E, N, K), g, device, torch.float32, 0.05))
    pair_w = torch.rand(M, generator=g).to(device)
    return (a, q, s, sorted_ids.to(device), expert_ids.to(device), tok.to(device), pair_w,
            torch.zeros(T, N, dtype=torch.float32, device=device))


def make_inputs(shape, dtype, device, g):
    M, K, N, E, T = shape
    sorted_e = torch.sort(torch.randint(0, E, (M,), generator=g)).values
    tok = torch.randint(0, T, (M,), generator=g).to(torch.int32)
    return _with_values(M, K, N, E, T, dtype, device, g, *align(sorted_e, E), tok)


def make_real(shape, dtype, device, g, ints):
    """The captured call's routing (sorted_ids, expert_ids, tok), random rows, weights and routing weights."""
    M, K, N, E, T = shape
    sorted_ids, expert_ids, tok = (torch.tensor(x, dtype=torch.int32) for x in ints[:3])
    return _with_values(M, K, N, E, T, dtype, device, g, sorted_ids, expert_ids, tok)


def shape_of(call):
    """(M, K, N, E, T) from a call with fp8 weights; bf16 calls are skipped (no task for them)."""
    a, kw = call["args"], call.get("kwargs") or {}
    if len(a) < 8 or "shape" not in a[0] or a[1].get("dtype") != "float8_e4m3fn" or "w_scale" not in kw:
        return None
    M, K = a[0]["shape"]
    E, N, _ = a[1]["shape"]
    return (M, K, N, E, a[7]["shape"][0])


def cost(shape, dtype, inputs=None):
    """2·M·N·K FLOPs; bytes: the pairs' rows, the touched experts' fp8 weights and scales, the fp32 output read and
    written."""
    M, K, N, E, T = shape
    b = torch.finfo(dtype).bits // 8
    touched = int(inputs[4][inputs[4] < E].unique().numel()) if inputs is not None else min(E, M)
    return 2 * M * N * K, M * K * b + touched * (N * K + 4 * N) + 2 * T * N * 4 + M * (4 + 4)


def edge_shapes(shape):
    M, K, N, E, T = shape
    return [(1, K, N, E, 1), (3, K, N, E, 2), (E + 1, K, N, E, E + 1)]


# Qwen3-30B-A3B's down projection: K = 768 (expert width), N = 2048 (hidden), 128 experts, 8 pairs per token.
TASK = Task("moe_gemm_scatter_fp8", "inference_server.models.moe_kernel:moe_gemm_scatter", reference, make_inputs,
            shape_of, cost, call=_call, make_real=make_real, graphs=True, kernel_names=(r"_moe_gemm_scatter_kernel",),
            size_expr="args[4]", heldout_axes=(0,), edge_shapes=edge_shapes, dtypes=(torch.bfloat16,), layouts=(),
            seen_default=((8, 48, 64, 8, 1), (32, 48, 64, 8, 4), (64, 48, 64, 8, 8)),
            heldout_default=((128, 48, 64, 8, 16), (256, 48, 64, 8, 32)))
