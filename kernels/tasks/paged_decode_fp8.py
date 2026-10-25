"""Paged decode attention on fp8 KV pools: the example engine's `paged_decode_attention` with KV_CACHE_DTYPE=fp8, its
pools float8_e4m3fn at a unit scale (no dequantization factor), q bf16; the kernel widens K/V per element today. Shapes,
block tables and sequence lengths as `paged_decode`; its calls are captured with KV_CACHE_DTYPE=fp8."""
from __future__ import annotations

import torch

from kernels.tasks import Task
from kernels.tasks import paged_decode as pd

FP8 = torch.float8_e4m3fn


def make_inputs(shape, dtype, device, g, lens=None, context_window=None):
    q, k, v, table, lens = pd.make_inputs(shape, dtype, device, g, lens, context_window)
    return q, k.to(FP8), v.to(FP8), table, lens


def make_real(shape, dtype, device, g, ints, context_window=None):
    return make_inputs(shape, dtype, device, g, torch.tensor(ints[0], dtype=torch.int32), context_window)


def shape_of(call):
    """Calls on fp8 pools only."""
    a = call["args"]
    if len(a) < 5 or a[1].get("dtype") != "float8_e4m3fn":
        return None
    return pd.shape_of({**call, "args": [a[0], {**a[1], "dtype": None}, *a[2:]]})


def cost(shape, dtype, inputs=None):
    """As `paged_decode`, with one byte per K/V element."""
    N, Hq, Hkv, D, bs, L = shape
    keys = int(inputs[4].sum()) if inputs is not None else N * L
    b = torch.finfo(dtype).bits // 8
    return 4 * Hq * keys * D, 2 * Hkv * keys * D + 2 * N * Hq * D * b


TASK = Task("paged_decode_fp8", pd.TASK.entry, pd.reference, make_inputs, shape_of, cost, call=pd._call,
            make_real=make_real, graphs=True, options=("context_window",), kernel_names=pd.TASK.kernel_names,
            size_expr=pd.TASK.size_expr, heldout_axes=pd.TASK.heldout_axes, edge_shapes=pd.edge_shapes,
            dtypes=(torch.bfloat16,), layouts=(), capture_env=(("KV_CACHE_DTYPE", "fp8"),),
            seen_default=pd.TASK.seen_default, heldout_default=pd.TASK.heldout_default)
