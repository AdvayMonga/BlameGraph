"""Shared pieces of the fused-kernel tasks: Qwen3-30B-A3B's constants and the engine's eager RMSNorm as a reference."""
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


# Edits that make models/fused.py import its kernels module directly.
START_EDITS = (
    ("\nimport importlib.util\n", "\ntry:                                    # its kernels module\n"
     "    from inference_server.models import fused_kernels as K\nexcept ImportError:                     # no triton\n"
     "    K = None\n"),
    ('importlib.util.find_spec("triton") is not None', "K is not None"),
    ('class _Lazy:\n    """The kernels module, imported on first use."""\n    def __getattr__(self, name):\n'
     "        from inference_server.models import fused_kernels\n        return getattr(fused_kernels, name)\n\n\n"
     "K = _Lazy()\n\n\n", "# K, the kernels module, is imported above\n\n\n"),
)
