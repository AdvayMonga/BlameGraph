"""MoE routing, the example engine's `fused.topk_softmax(logits, k, renorm, fp32=True)` -> (fp32 weights, expert ids
int64), both [M, k]: softmax over experts in fp32, top-k, renormalised (the engine's `route`, which keeps the weights
in fp32 since inference-server #122; Qwen3-30B-A3B: 128 experts, k = 8, renorm on). Shape: (M, E)."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from kernels.tasks import Task, randn
from kernels.tasks._fused import EXPERTS, START_EDITS, TOP_K


def reference(logits):
    w, ids = torch.topk(F.softmax(logits, dim=-1, dtype=torch.float), TOP_K, dim=-1)
    return w / w.sum(dim=-1, keepdim=True), ids


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
            size_expr="args[0].shape[0]", heldout_axes=(0,), edge_shapes=lambda s: [(1, s[1]), (3, s[1]), (s[0] + 1, s[1])],
            dtypes=(torch.bfloat16,), layouts=(), seen_default=((1, EXPERTS), (16, EXPERTS), (64, EXPERTS)),
            heldout_default=((512, EXPERTS),))
