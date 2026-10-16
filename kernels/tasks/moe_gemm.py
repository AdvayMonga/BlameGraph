"""The MoE grouped GEMM, the example engine's `moe_gemm`: per (token, expert) pair in expert-sorted order, that
token's row times its expert's weights. Both expert projections call it: gate_up with rows gathered from the tokens
(`tok`), down with the pairs' rows given directly (`tok` empty here; None to the engine).

Shape: (M pairs, K, N, E experts, T rows of `a`, from_tokens 1|0). Inputs: a [T, K], tok [M] int64 (empty when not
from tokens), w [E, N, K], sorted_ids / expert_ids from `align` (block_m 16): the pair ids per padded row block and
each block's expert. Captured calls carry the engine's real routing (tok, sorted_ids, expert_ids by value)."""
from __future__ import annotations

import torch

from kernels.tasks import Task

BLOCK_M = 16


def align(sorted_experts, num_experts: int, block_m: int = BLOCK_M):
    """The engine's `align`: each expert's run of sorted pairs padded to a multiple of block_m; (pair id per padded
    row with M as padding, expert per row block with num_experts past the end)."""
    M, dev = sorted_experts.numel(), sorted_experts.device
    ar = torch.arange(num_experts, device=dev)
    ends, starts = torch.searchsorted(sorted_experts, ar, right=True), torch.searchsorted(sorted_experts, ar)
    padded = (ends - starts + block_m - 1) // block_m * block_m
    pstart = torch.cumsum(padded, 0) - padded
    max_pad = M + num_experts * (block_m - 1)
    sorted_ids = torch.full((max_pad,), M, dtype=torch.int32, device=dev)
    i = torch.arange(M, device=dev)
    sorted_ids[pstart[sorted_experts] + (i - starts[sorted_experts])] = i.to(torch.int32)
    block_starts = torch.arange(max_pad // block_m, device=dev) * block_m
    total = pstart[-1] + padded[-1]
    expert_ids = torch.searchsorted(pstart, block_starts, right=True) - 1
    expert_ids = torch.where(block_starts < total, expert_ids, torch.full_like(expert_ids, num_experts))
    return sorted_ids, expert_ids.to(torch.int32)


def _pairs(tok, a):
    return tok.numel() if tok.numel() else a.shape[0]


def _call(fn, a, tok, w, sorted_ids, expert_ids):
    M = _pairs(tok, a)
    block_m = (sorted_ids.numel() - M) // w.shape[0] + 1
    return fn(a, tok if tok.numel() else None, w, sorted_ids, expert_ids, M, block_m=block_m)


def reference(a, tok, w, sorted_ids, expert_ids):
    """Each pair's row times its expert's weights, in fp32."""
    M, (E, N, _) = _pairs(tok, a), w.shape
    bm = (sorted_ids.numel() - M) // E + 1
    expert = torch.full((sorted_ids.numel(),), E, dtype=torch.long, device=a.device)
    expert[: expert_ids.numel() * bm] = expert_ids.long().repeat_interleave(bm)   # rows past the last block: padding
    real = (sorted_ids < M) & (expert < E)
    ids, ex = sorted_ids[real].long(), expert[real]
    rows = a[tok[ids]] if tok.numel() else a[ids]
    out = torch.zeros(M, N, dtype=torch.float32, device=a.device)
    for e in ex.unique().tolist():
        sel = ex == e
        out[ids[sel]] = rows[sel].float() @ w[e].float().T
    return out.to(a.dtype)


def _routing(M, E, T, from_tokens, g, sorted_e=None):
    if sorted_e is None:
        sorted_e = torch.sort(torch.randint(0, E, (M,), generator=g)).values
    tok = torch.randint(0, T, (M,), generator=g) if from_tokens else torch.empty(0, dtype=torch.int64)
    return (tok, *align(sorted_e, E))


def make_inputs(shape, dtype, device, g):
    M, K, N, E, T, ft = shape
    tok, sorted_ids, expert_ids = _routing(M, E, T, ft, g)
    a = torch.randn(T, K, generator=g).to(dtype)
    w = (torch.randn(E, N, K, generator=g) * 0.05).to(dtype)
    return tuple(t.to(device) for t in (a, tok, w, sorted_ids, expert_ids))


def make_real(shape, dtype, device, g, ints):
    """The captured call's routing (tok when from tokens, sorted_ids, expert_ids), random a and w."""
    M, K, N, E, T, ft = shape
    tok = torch.tensor(ints[0], dtype=torch.int64) if ft else torch.empty(0, dtype=torch.int64)
    sorted_ids, expert_ids = (torch.tensor(x, dtype=torch.int32) for x in ints[-2:])
    a = torch.randn(T, K, generator=g).to(dtype)
    w = (torch.randn(E, N, K, generator=g) * 0.05).to(dtype)
    return tuple(t.to(device) for t in (a, tok, w, sorted_ids, expert_ids))


def shape_of(call):
    a = call["args"]
    if len(a) < 6 or "shape" not in a[0] or "shape" not in a[2]:
        return None
    T, K = a[0]["shape"]
    E, N, _ = a[2]["shape"]
    ft = int("shape" in a[1])
    M = a[1]["shape"][0] if ft else T
    return (M, K, N, E, T, ft)


def cost(shape, dtype, inputs=None):
    """2·M·N·K FLOPs; bytes: each pair's input row, the weights of every expert the pairs touch, the output."""
    M, K, N, E, T, ft = shape
    b = torch.finfo(dtype).bits // 8
    touched = int(inputs[4][inputs[4] < E].unique().numel()) if inputs is not None else min(E, M)
    return 2 * M * N * K, (M * K + touched * N * K + M * N) * b


def edge_shapes(shape):
    M, K, N, E, T, ft = shape
    return [(1, K, N, E, 1, ft), (3, K, N, E, 3, ft), (E + 1, K, N, E, E + 1, ft)]


# Qwen3-30B-A3B: 128 experts, hidden 2048, expert width 768 (gate_up N = 1536, K = 2048; down N = 2048, K = 768); the
# engine calls this kernel for up to 2048 pairs (CUSTOM_BACKEND_MOE_KERNEL_ROWS) inside its decode CUDA graphs.
TASK = Task("moe_gemm", "inference_server.models.moe_kernel:moe_gemm", reference, make_inputs, shape_of, cost,
            call=_call, make_real=make_real, graphs=True, kernel_names=(r"_moe_gemm_kernel",), heldout_axes=(0,),
            edge_shapes=edge_shapes, dtypes=(torch.bfloat16,), layouts=(),
            seen_default=((8, 64, 96, 8, 4, 1), (16, 64, 96, 8, 16, 0), (32, 64, 96, 8, 16, 1)),
            heldout_default=((64, 64, 96, 8, 32, 1), (128, 64, 96, 8, 128, 0)))
