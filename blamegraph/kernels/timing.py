"""Honest kernel timing: synchronized, rotated input buffers, L2 flushed on CUDA; plus a memoization check."""
from __future__ import annotations

import statistics
import time

import torch

from .check import _gen


def _sync(device):
    kind = torch.device(device).type
    if kind == "cuda":
        torch.cuda.synchronize(device)
    elif kind == "mps":
        torch.mps.synchronize()


def time_kernel(fn, make_inputs, shape, dtype, device="cpu", iters=50, warmup=5, n_buffers=4, fresh=False,
                seed=0) -> dict:
    """Median and IQR (ms) of fn(*inputs) over n_buffers rotated input sets; fresh=True refills them with new
    values before every call. Each call is synchronized before and after; L2 is flushed between calls on CUDA."""
    bufs = [make_inputs(shape, dtype, device, _gen(seed + i)) for i in range(n_buffers)]
    flush = torch.empty(256 * 2**20, dtype=torch.int8, device=device) if torch.device(device).type == "cuda" else None
    times = []
    for i in range(warmup + iters):
        inputs = bufs[i % n_buffers]
        if fresh:
            for t, n in zip(inputs, make_inputs(shape, dtype, device, _gen(seed + n_buffers + i))):
                t.copy_(n)
        if flush is not None:
            flush.zero_()
        _sync(device)
        t0 = time.perf_counter()
        fn(*inputs)
        _sync(device)
        if i >= warmup:
            times.append((time.perf_counter() - t0) * 1e3)
    p25, med, p75 = statistics.quantiles(times, n=4)
    return {"median_ms": med, "iqr_ms": p75 - p25, "p25_ms": p25, "p75_ms": p75, "iters": iters,
            "n_buffers": n_buffers, "fresh": fresh, "device": str(device)}


def check_memoization(fn, make_inputs, shape, dtype, device="cpu", iters=30, warmup=3, ratio=0.5) -> dict:
    """Gate {passed, reason, evidence}: fails if repeated identical inputs run faster than ratio x fresh inputs."""
    rep = time_kernel(fn, make_inputs, shape, dtype, device, iters, warmup, n_buffers=1)
    new = time_kernel(fn, make_inputs, shape, dtype, device, iters, warmup, fresh=True)
    r = rep["median_ms"] / new["median_ms"]
    reason = (f"median on repeated identical inputs {rep['median_ms']:.3g} ms = {r:.2f}x median on fresh inputs "
              f"{new['median_ms']:.3g} ms" + ("" if r >= ratio else f" (< {ratio})"))
    return {"passed": r >= ratio, "reason": reason, "evidence": {"repeated": rep, "fresh": new, "ratio": r}}
