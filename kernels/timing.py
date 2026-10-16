"""Honest kernel timing: synchronized, rotated input buffers, L2 flushed on CUDA; plus a memoization check."""
from __future__ import annotations

import random
import statistics
import time

import torch

from .check import _gen

# Bound at import, before any candidate loads: a candidate that patches the clock, the events or the syncs on their
# modules changes nothing the timer calls.
_perf_counter, _quantiles, _Random = time.perf_counter, statistics.quantiles, random.Random
_Event, _record, _elapsed = torch.cuda.Event, torch.cuda.Event.record, torch.cuda.Event.elapsed_time
_cuda_sync, _mps_sync = torch.cuda.synchronize, torch.mps.synchronize


def _sync(device):
    kind = torch.device(device).type
    if kind == "cuda":
        _cuda_sync(device)
    elif kind == "mps":
        _mps_sync()


def _one(fn, inputs, device, flush) -> float:
    """One synchronized call's time in ms: L2 flushed first on CUDA, where CUDA events time it on the device (host
    clock jitter swamps a ~10 µs kernel)."""
    if flush is not None:
        flush.zero_()
    _sync(device)
    if torch.device(device).type == "cuda":
        start, end = _Event(enable_timing=True), _Event(enable_timing=True)
        _record(start)
        fn(*inputs)
        _record(end)
        _sync(device)
        return _elapsed(start, end)
    t0 = _perf_counter()
    fn(*inputs)
    _sync(device)
    return (_perf_counter() - t0) * 1e3


def _stats(times: list[float]) -> dict:
    p25, med, p75 = _quantiles(times, n=4)
    return {"median_ms": med, "iqr_ms": p75 - p25, "p25_ms": p25, "p75_ms": p75}


def _flush(device):
    return torch.empty(256 * 2**20, dtype=torch.int8, device=device) if torch.device(device).type == "cuda" else None


def time_kernel(fn, make_inputs, shape, dtype, device="cpu", iters=50, warmup=5, n_buffers=4, fresh=False,
                seed=0) -> dict:
    """Median and IQR (ms) of fn(*inputs) over n_buffers rotated input sets; fresh=True refills them with new
    values before every call. Each call is synchronized before and after, L2 flushed between calls on CUDA."""
    bufs = [make_inputs(shape, dtype, device, _gen(seed + i)) for i in range(n_buffers)]
    flush, times = _flush(device), []
    for i in range(warmup + iters):
        inputs = bufs[i % n_buffers]
        if fresh:
            for t, n in zip(inputs, make_inputs(shape, dtype, device, _gen(seed + n_buffers + i))):
                t.copy_(n)
        ms = _one(fn, inputs, device, flush)
        if i >= warmup:
            times.append(ms)
    return {**_stats(times), "iters": iters, "n_buffers": n_buffers, "fresh": fresh, "device": str(device),
            "clock": "cuda_events" if torch.device(device).type == "cuda" else "host"}


def time_pair(a, b, make_inputs, shape, dtype, device="cpu", iters=50, warmup=5, n_buffers=4, seed=0) -> dict:
    """a and b timed in pairs on the same inputs, in a seeded random order per pair, so drift (clocks, heat, other
    load) lands on both: each one's stats plus the per-pair ratio b/a (above 1: a is faster)."""
    bufs = [make_inputs(shape, dtype, device, _gen(seed + i)) for i in range(n_buffers)]
    flush, order, ta, tb = _flush(device), _Random(seed), [], []
    for i in range(warmup + iters):
        inputs = bufs[i % n_buffers]
        first_a = order.random() < 0.5
        x = _one(a if first_a else b, inputs, device, flush)
        y = _one(b if first_a else a, inputs, device, flush)
        if i >= warmup:
            ta.append(x if first_a else y)
            tb.append(y if first_a else x)
    r25, rmed, r75 = _quantiles([y / x for x, y in zip(ta, tb) if x > 0], n=4)
    return {"a": _stats(ta), "b": _stats(tb), "ratio": {"median": rmed, "p25": r25, "p75": r75}, "iters": iters,
            "clock": "cuda_events" if torch.device(device).type == "cuda" else "host"}


def check_memoization(fn, make_inputs, shape, dtype, device="cpu", iters=30, warmup=3, ratio=0.5) -> dict:
    """Gate {passed, reason, evidence}: fails if repeated identical inputs run faster than ratio x fresh inputs."""
    rep = time_kernel(fn, make_inputs, shape, dtype, device, iters, warmup, n_buffers=1)
    new = time_kernel(fn, make_inputs, shape, dtype, device, iters, warmup, fresh=True)
    r = rep["median_ms"] / new["median_ms"]
    reason = (f"median on repeated identical inputs {rep['median_ms']:.3g} ms = {r:.2f}x median on fresh inputs "
              f"{new['median_ms']:.3g} ms" + ("" if r >= ratio else f" (< {ratio})"))
    return {"passed": r >= ratio, "reason": reason, "evidence": {"repeated": rep, "fresh": new, "ratio": r}}
