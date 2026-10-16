"""Paged decode attention, the example engine's `paged_decode_attention`: one query token per sequence attends to
that sequence's keys and values, read through a block table from a shared paged pool.

Shape: (N sequences, Hq query heads, Hkv KV heads, D head dim, block_size, L longest context). Inputs:
q [N, Hq, D], k_pool / v_pool [blocks, Hkv, block_size, D], block_tables [N, max_blocks] int32, seq_lens [N]
int32 (one sequence at L, the rest drawn in [1, L]). The scale is 1/sqrt(D)."""
from __future__ import annotations

import math

import torch

from kernels.tasks import Task


def _call(fn, q, k_pool, v_pool, block_tables, seq_lens):
    return fn(q, k_pool, v_pool, block_tables, seq_lens, scale=1.0 / math.sqrt(q.shape[-1]))


def reference(q, k_pool, v_pool, block_tables, seq_lens):
    N, Hq, D = q.shape
    Hkv, bs = k_pool.shape[1], k_pool.shape[2]
    scale = 1.0 / math.sqrt(D)
    out = torch.empty(N, Hq, D, dtype=torch.float32, device=q.device)
    for i in range(N):
        n = int(seq_lens[i])
        blocks = block_tables[i, : (n + bs - 1) // bs].long()
        k = k_pool[blocks].permute(1, 0, 2, 3).reshape(Hkv, -1, D)[:, :n].float()     # [Hkv, n, D]
        v = v_pool[blocks].permute(1, 0, 2, 3).reshape(Hkv, -1, D)[:, :n].float()
        k, v = k.repeat_interleave(Hq // Hkv, 0), v.repeat_interleave(Hq // Hkv, 0)  # [Hq, n, D]
        w = torch.softmax((q[i].float().unsqueeze(1) @ k.transpose(1, 2)).squeeze(1) * scale, dim=-1)
        out[i] = (w.unsqueeze(1) @ v).squeeze(1)
    return out.to(q.dtype)


def make_inputs(shape, dtype, device, g, lens=None):
    N, Hq, Hkv, D, bs, L = shape
    if lens is None:
        lens = torch.randint(1, L + 1, (N,), generator=g, dtype=torch.int32)
        lens[0] = L
    per = (L + bs - 1) // bs
    blocks = N * per + 1
    table = torch.randperm(blocks, generator=g)[: N * per].view(N, per).to(torch.int32)
    q = torch.randn(N, Hq, D, generator=g).to(dtype)
    k = torch.randn(blocks, Hkv, bs, D, generator=g).to(dtype)
    v = torch.randn(blocks, Hkv, bs, D, generator=g).to(dtype)
    return tuple(t.to(device) for t in (q, k, v, table, lens))


def make_real(shape, dtype, device, g, ints):
    """The captured call's sequence lengths (its one 1-D int argument), random q/k/v and block tables."""
    return make_inputs(shape, dtype, device, g, torch.tensor(ints[0], dtype=torch.int32))


def shape_of(call):
    a = call["args"]
    if len(a) < 5 or "shape" not in a[0] or "values" not in a[4]:
        return None
    N, Hq, D = a[0]["shape"]
    _, Hkv, bs, _ = a[1]["shape"]
    return (N, Hq, Hkv, D, bs, max(a[4]["values"]))


def cost(shape, dtype, inputs=None):
    """Read q and every live K/V once, write the output; 4 FLOPs per (head, key, dim) for QK and PV. With inputs, the
    keys are their real sequence lengths; without, every sequence at L, which overstates the work."""
    N, Hq, Hkv, D, bs, L = shape
    b = torch.finfo(dtype).bits // 8
    keys = int(inputs[4].sum()) if inputs is not None else N * L
    return 4 * Hq * keys * D, (2 * Hkv * keys * D + 2 * N * Hq * D) * b


def edge_shapes(shape):
    N, Hq, Hkv, D, bs, L = shape
    return [(1, Hq, Hkv, D, bs, 1), (1, Hq, Hkv, D, bs, bs + 1), (N, Hq, Hkv, D, bs, bs), (3, Hq, Hkv, D, bs, 7)]


# Qwen3-30B-A3B: 32 query heads, 4 KV heads, head dim 128; the engine's block size is read from captures.
TASK = Task("paged_decode", "inference_server.models.paged_attention_kernel:paged_decode_attention", reference,
            make_inputs, shape_of, cost, call=_call, make_real=make_real, graphs=True,
            kernel_names=(r"_paged_decode(_splitk)?_kernel", r"_gqa_decode\w*_kernel", r"_splitk_combine(_vec)?_kernel"), heldout_axes=(0, 5), edge_shapes=edge_shapes,
            dtypes=(torch.bfloat16,), layouts=(),        # the engine passes contiguous tensors only
            seen_default=((1, 32, 4, 128, 16, 512), (8, 32, 4, 128, 16, 1024), (32, 32, 4, 128, 16, 2048)),
            heldout_default=((64, 32, 4, 128, 16, 4096), (128, 32, 4, 128, 16, 8192)))
