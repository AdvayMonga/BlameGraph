"""The MoE grouped GEMM as the example engine runs it by default since inference-server #122: `moe_gemm` with fp8
e4m3 expert weights and a per-output-channel fp32 scale (`w_scale` [E, N]), here the gate_up projection (rows
gathered from the tokens). Shape and routing as `moe_gemm` (bf16 weights), which keeps the bf16 path.

Inputs: a [T, K] bf16, tok [M] int32, w [E, N, K] float8_e4m3fn, w_scale [E, N] fp32, sorted_ids / expert_ids."""
from __future__ import annotations

import torch

from kernels.tasks import Task, randn
from kernels.tasks.moe_gemm import _block_m, _pairs, _routing, edge_shapes
from kernels.tasks.moe_gemm import reference as _bf16_reference

FP8, FP8_MAX = torch.float8_e4m3fn, 448.0


def quantize(w):
    """Per output channel (the engine's scheme): w ~= q * scale[..., None], q fp8 e4m3, scale fp32 [E, N]."""
    scale = (w.float().abs().amax(-1) / FP8_MAX).clamp_min(1e-12)
    return (w.float() / scale[..., None]).to(FP8), scale


def dequantize(q, scale):
    return q.float() * scale[..., None]


def _call(fn, a, tok, w, w_scale, sorted_ids, expert_ids):
    M = _pairs(tok, a)
    out = fn(a, tok if tok.numel() else None, w, sorted_ids, expert_ids, M, block_m=_block_m(sorted_ids, expert_ids),
             w_scale=w_scale)
    return out if out.dtype == a.dtype else out.to(a.dtype)


def reference(a, tok, w, w_scale, sorted_ids, expert_ids):
    """Each pair's row times its expert's dequantized weights, in fp32."""
    return _bf16_reference(a, tok, dequantize(w, w_scale), sorted_ids, expert_ids)


def _with_values(T, K, N, E, dtype, device, g, tok, sorted_ids, expert_ids):
    a = randn((T, K), g, device, dtype)
    q, s = quantize(randn((E, N, K), g, device, torch.float32, 0.05))
    return a, tok.to(device), q, s, sorted_ids.to(device), expert_ids.to(device)


def make_inputs(shape, dtype, device, g):
    M, K, N, E, T, ft = shape
    return _with_values(T, K, N, E, dtype, device, g, *_routing(M, E, T, ft, g))


def make_real(shape, dtype, device, g, ints):
    M, K, N, E, T, ft = shape
    tok = torch.tensor(ints[0], dtype=torch.int32) if ft else torch.empty(0, dtype=torch.int32)
    sorted_ids, expert_ids = (torch.tensor(x, dtype=torch.int32) for x in ints[-2:])
    return _with_values(T, K, N, E, dtype, device, g, tok, sorted_ids, expert_ids)


def shape_of(call):
    """Calls with fp8 weights and their scale only (bf16 ones are `moe_gemm`'s)."""
    a, kw = call["args"], call.get("kwargs") or {}
    if len(a) < 6 or "shape" not in a[0] or a[2].get("dtype") != "float8_e4m3fn" or "w_scale" not in kw:
        return None
    T, K = a[0]["shape"]
    E, N, _ = a[2]["shape"]
    ft = int("shape" in a[1])
    return (a[1]["shape"][0] if ft else T, K, N, E, T, ft)


def cost(shape, dtype, inputs=None):
    """2·M·N·K FLOPs; bytes: the pairs' input rows, the touched experts' fp8 weights and scales, the output."""
    M, K, N, E, T, ft = shape
    b = torch.finfo(dtype).bits // 8
    touched = int(inputs[5][inputs[5] < E].unique().numel()) if inputs is not None else min(E, M)
    return 2 * M * N * K, M * K * b + touched * (N * K + 4 * N) + M * N * b


TASK = Task("moe_gemm_fp8", "inference_server.models.moe_kernel:moe_gemm", reference, make_inputs, shape_of, cost,
            call=_call, make_real=make_real, graphs=True, kernel_names=(r"_moe_gemm_kernel",), size_expr="args[5]",
            heldout_axes=(0,), edge_shapes=edge_shapes, dtypes=(torch.bfloat16,), layouts=(),
            seen_default=((8, 64, 96, 8, 4, 1), (16, 64, 96, 8, 16, 1), (32, 64, 96, 8, 16, 1)),
            heldout_default=((64, 64, 96, 8, 32, 1), (128, 64, 96, 8, 128, 1)))
