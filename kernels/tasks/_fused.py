"""Shared pieces of the tasks for the example engine's fused kernels (inference_server.models.fused): Qwen3-30B-A3B's
constants and the engine's own eager semantics as references (HF Qwen3 RMSNorm: normalise in fp32, cast back, then
scale; HF rotary: halves rotated)."""
from __future__ import annotations

import torch

EPS, HIDDEN, HEAD_DIM, ROPE_THETA, EXPERTS, TOP_K = 1e-6, 2048, 128, 1e6, 128, 8


def rms(x, w):
    """The engine's eager Qwen3RMSNorm in fp32, the cast to the input dtype kept where the engine has it."""
    x32 = x.float()
    n = (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + EPS)).to(x.dtype)
    return w.float() * n.float()


def rows_cost(rows: int, d: int, reads: int, writes: int, dtype) -> tuple[int, int]:
    b = torch.finfo(dtype).bits // 8
    return 4 * rows * d, (reads * rows * d + writes * rows * d + d) * b


def rows_edges(shape):
    return [(1, *shape[1:]), (3, *shape[1:]), (shape[0] + 1, *shape[1:])]


def weight(d, g):
    return 1 + 0.1 * torch.randn(d, generator=g)
