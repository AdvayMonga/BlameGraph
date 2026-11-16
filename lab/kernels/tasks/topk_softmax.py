"""MoE routing, the example engine's `fused.topk_softmax(logits, k, renorm, fp32=True)`: softmax, top-k, renormalise.
Shape: (M, E). Returns (fp32 weights, int64 expert ids), both [M, k]."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from lab.kernels.tasks import Task, randn
from lab.kernels.tasks._fused import EXPERTS, START_EDITS, TOP_K


def reference(logits):
    """Ties go to the lowest expert index."""
    p = F.softmax(logits, dim=-1, dtype=torch.float)
    w, ids = torch.sort(p, dim=-1, descending=True, stable=True)
    w, ids = w[..., :TOP_K], ids[..., :TOP_K]
    w = w / w.sum(dim=-1, keepdim=True)
    bad = p.isnan().any(dim=-1, keepdim=True)        # a NaN row gives NaN weights and expert E - 1
    return torch.where(bad, float("nan"), w), torch.where(bad, logits.shape[-1] - 1, ids)


def make_inputs(shape, dtype, device, g):
    return (randn(shape, g, device, dtype),)


def shape_of(call):
    a = call["args"]
    return tuple(a[0]["shape"]) if a and len(a[0].get("shape") or []) == 2 else None


def cost(shape, dtype, inputs=None):
    M, E = shape
    b = torch.finfo(dtype).bits // 8
    return 5 * M * E, M * E * b + M * TOP_K * (4 + 8)


TASK = Task("topk_softmax", "inference_server.models.fused:topk_softmax", reference, make_inputs, shape_of, cost,
            call=lambda fn, logits: fn(logits, TOP_K, True, fp32=True), start_edits=START_EDITS, graphs=True, scales=(3.0, -1.0), kernel_names=(r"^_topk_softmax_kernel",),
            heldout_axes=(0,), edge_shapes=lambda s: [(1, s[1]), (3, s[1]), (s[0] + 1, s[1])],
            dtypes=(torch.bfloat16,), layouts=(), seen_default=((1, EXPERTS), (16, EXPERTS), (64, EXPERTS)),
            heldout_default=((512, EXPERTS),))
