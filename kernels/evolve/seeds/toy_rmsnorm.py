# EVOLVE-BLOCK-START
import torch


def kernel(x, w):
    """RMSNorm over the last dim, written naively: one row at a time."""
    out = torch.empty_like(x)
    for i in range(x.shape[0]):
        r = x[i].float()
        out[i] = (r / torch.sqrt((r * r).mean() + 1e-6) * w.float()).to(x.dtype)
    return out
# EVOLVE-BLOCK-END
