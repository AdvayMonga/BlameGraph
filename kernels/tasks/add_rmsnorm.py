"""Residual add + RMSNorm, the example engine's `fused.add_rmsnorm(h, a, weight, eps)` -> (h + a, rmsnorm(h + a)):
h, a [rows, D] bf16, weight [D]. Shape: (rows, D)."""
from __future__ import annotations

import torch

from kernels.tasks import Task, randn
from kernels.tasks._fused import EPS, HIDDEN, rms, rows_cost, rows_edges, weight


def reference(h, a, w):
    s = (h.float() + a.float()).to(h.dtype)
    return s, rms(s, w).to(h.dtype)


def make_inputs(shape, dtype, device, g):
    rows, d = shape
    return randn((rows, d), g, device, dtype), randn((rows, d), g, device, dtype), weight(d, g).to(dtype).to(device)


def shape_of(call):
    a = call["args"]
    if len(a) < 3 or "shape" not in a[0]:
        return None
    s = a[0]["shape"]
    rows = 1
    for v in s[:-1]:
        rows *= v
    return (rows, s[-1])


TASK = Task("add_rmsnorm", "inference_server.models.fused:add_rmsnorm", reference, make_inputs, shape_of,
            lambda shape, dtype, inputs=None: rows_cost(*shape, 2, 2, dtype),
            call=lambda fn, h, a, w: fn(h, a, w, EPS), graphs=True, kernel_names=(r"^_add_rmsnorm_kernel",),
            heldout_axes=(0,), edge_shapes=rows_edges, dtypes=(torch.bfloat16,), layouts=(),
            seen_default=((4, HIDDEN), (32, HIDDEN)), heldout_default=((512, HIDDEN),))
