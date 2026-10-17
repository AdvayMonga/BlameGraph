"""SiLU-gated product, the example engine's `fused.silu_mul(x)`: x [M, 2I] -> silu(x[:, :I]) * x[:, I:] (the MoE
gate_up output; I = 768 for Qwen3-30B-A3B). Shape: (M, 2I)."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from kernels.tasks import Task, randn


def reference(x):
    gate, up = x.float().chunk(2, dim=-1)
    return (F.silu(gate) * up).to(x.dtype)


def make_inputs(shape, dtype, device, g):
    return (randn(shape, g, device, dtype),)


def shape_of(call):
    a = call["args"]
    return tuple(a[0]["shape"]) if a and len(a[0].get("shape") or []) == 2 else None


def cost(shape, dtype, inputs=None):
    M, two_i = shape
    b = torch.finfo(dtype).bits // 8
    return 4 * M * two_i, (M * two_i + M * two_i // 2) * b


TASK = Task("silu_mul", "inference_server.models.fused:silu_mul", reference, make_inputs, shape_of, cost,
            graphs=True, kernel_names=(r"^_silu_mul_kernel",), heldout_axes=(0,),
            edge_shapes=lambda s: [(1, s[1]), (3, s[1]), (s[0] + 1, s[1])], dtypes=(torch.bfloat16,), layouts=(),
            seen_default=((8, 1536), (64, 1536), (512, 1536)), heldout_default=((2048, 1536),))
