"""RMSNorm, the example engine's `fused.rmsnorm(x, weight, eps)`: x [rows, D] (the engine flattens leading dims),
weight [D]. Shape: (rows, D)."""
from __future__ import annotations

import torch

from kernels.tasks import Task, randn
from kernels.tasks._fused import EPS, HIDDEN, START_EDITS, rms, rows_cost, rows_edges, weight


def reference(x, w):
    return rms(x, w).to(x.dtype)


def make_inputs(shape, dtype, device, g):
    rows, d = shape
    return randn((rows, d), g, device, dtype), weight(d, g).to(dtype).to(device)


def shape_of(call):
    a = call["args"]
    if len(a) < 2 or "shape" not in a[0]:
        return None
    s = a[0]["shape"]
    rows = 1
    for v in s[:-1]:
        rows *= v
    return (rows, s[-1])


TASK = Task("rmsnorm", "inference_server.models.fused:rmsnorm", reference, make_inputs, shape_of,
            lambda shape, dtype, inputs=None: rows_cost(*shape, 1, 1, dtype), call=lambda fn, x, w: fn(x, w, EPS),
            start_edits=START_EDITS, graphs=True, kernel_names=(r"^_rmsnorm_kernel",), size_expr="args[0].numel() // args[0].shape[-1]", heldout_axes=(0,), edge_shapes=rows_edges,
            dtypes=(torch.bfloat16,), layouts=(), seen_default=((4, HIDDEN), (16, 128), (64, HIDDEN)),
            heldout_default=((256, HIDDEN), (1024, 128)))
