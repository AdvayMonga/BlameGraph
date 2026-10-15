"""A toy task for exercising the loop anywhere (CPU, MPS): RMSNorm over the last dim, x [rows, dim], w [dim]."""
from __future__ import annotations

import torch

from kernels.tasks import Task

EPS = 1e-6


def reference(x, w):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS) * w.float()).to(x.dtype)


def make_inputs(shape, dtype, device, g):
    rows, dim = shape
    return (torch.randn(rows, dim, generator=g).to(dtype).to(device),
            (1 + 0.1 * torch.randn(dim, generator=g)).to(dtype).to(device))


def shape_of(call):
    return tuple(call["args"][0]["shape"])


def cost(shape, dtype, inputs=None):
    rows, dim = shape
    b = torch.finfo(dtype).bits // 8
    return 4 * rows * dim, (2 * rows * dim + dim) * b          # read x and w, write y


TASK = Task("toy_rmsnorm", None, reference, make_inputs, shape_of, cost, heldout_axes=(0,),
            dtypes=(torch.float32,), seen_default=((8, 256), (32, 512), (64, 1024)),
            heldout_default=((256, 1024), (512, 2048)))
