"""Head-wise RMSNorm then rotary, the example engine's `fused.rmsnorm_rope(x, weight, cos, sin, eps)`: x [B, S, H, D],
weight [D], cos/sin [B, S, D] (HF layout: the two halves repeat). Called for queries (H = 32) and keys (H = 4).
Shape: (B, S, H, D); cos/sin from random positions at Qwen3's rope theta (1e6)."""
from __future__ import annotations

import torch

from kernels.tasks import Task, randn
from kernels.tasks._fused import EPS, HEAD_DIM, ROPE_THETA, rms, weight


def reference(x, w, cos, sin):
    n = rms(x, w).to(x.dtype).float()
    half = n.shape[-1] // 2
    rot = torch.cat((-n[..., half:], n[..., :half]), dim=-1)
    return (n * cos.float().unsqueeze(2) + rot * sin.float().unsqueeze(2)).to(x.dtype)


def _cos_sin(B, S, D, g, dtype):
    pos = torch.randint(0, 32768, (B, S), generator=g).float()
    inv = 1.0 / (ROPE_THETA ** (torch.arange(0, D, 2).float() / D))
    emb = torch.cat([pos[..., None] * inv] * 2, dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def make_inputs(shape, dtype, device, g):
    B, S, H, D = shape
    cos, sin = _cos_sin(B, S, D, g, dtype)
    return randn((B, S, H, D), g, device, dtype), weight(D, g).to(dtype).to(device), cos.to(device), sin.to(device)


def shape_of(call):
    a = call["args"]
    if len(a) < 4 or len(a[0].get("shape") or []) != 4:
        return None
    return tuple(a[0]["shape"])


def cost(shape, dtype, inputs=None):
    B, S, H, D = shape
    b = torch.finfo(dtype).bits // 8
    return 8 * B * S * H * D, (2 * B * S * H * D + 2 * B * S * D + D) * b


def edge_shapes(shape):
    B, S, H, D = shape
    return [(1, 1, H, D), (B + 1, 1, H, D), (1, S + 1, H, D)]


TASK = Task("rmsnorm_rope", "inference_server.models.fused:rmsnorm_rope", reference, make_inputs, shape_of, cost,
            call=lambda fn, x, w, c, s: fn(x, w, c, s, EPS), graphs=True, kernel_names=(r"^_rmsnorm_rope_kernel",),
            heldout_axes=(0, 1), edge_shapes=edge_shapes, dtypes=(torch.bfloat16,), layouts=(),
            seen_default=((4, 1, 32, HEAD_DIM), (8, 1, 4, HEAD_DIM), (1, 16, 32, HEAD_DIM)),
            heldout_default=((64, 1, 32, HEAD_DIM), (1, 256, 4, HEAD_DIM)))
