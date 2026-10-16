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
_CUDAGraph, _graph, _Stream, _stream, _current = (torch.cuda.CUDAGraph, torch.cuda.graph, torch.cuda.Stream,
                                                  torch.cuda.stream, torch.cuda.current_stream)
_sleep = torch.cuda._sleep
SPIN_CYCLES = 1_000_000          # ~0.5 ms of GPU spin queued ahead of a graph replay, so its launch is never timed


def _sync(device):
    kind = torch.device(device).type
    if kind == "cuda":
        _cuda_sync(device)
    elif kind == "mps":
        _mps_sync()


def _one(fn, inputs, device, flush, spin=False) -> tuple[float, object]:
    """One synchronized call's time in ms and its output: L2 flushed first on CUDA, where CUDA events time it on the
    device (host clock jitter swamps a ~10 µs kernel). spin=True queues a GPU spin before the start event, so the
    host's launch of fn happens while the GPU is busy and is not in the window (graph replays)."""
    if flush is not None:
        flush.zero_()
    _sync(device)
    if torch.device(device).type == "cuda":
        start, end = _Event(enable_timing=True), _Event(enable_timing=True)
        if spin:
            _sleep(SPIN_CYCLES)
        _record(start)
        out = fn(*inputs)
        _record(end)
        _sync(device)
        return _elapsed(start, end), out
    t0 = _perf_counter()
    out = fn(*inputs)
    _sync(device)
    return (_perf_counter() - t0) * 1e3, out


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
        ms, _ = _one(fn, inputs, device, flush)
        if i >= warmup:
            times.append(ms)
    return {**_stats(times), "iters": iters, "n_buffers": n_buffers, "fresh": fresh, "device": str(device),
            "clock": "cuda_events" if torch.device(device).type == "cuda" else "host"}


def graphed(fn, bufs: list, device) -> list:
    """One CUDA graph per input set, captured after a warm-up outside the capture (compiles, autotuning): each entry
    replays fn on that set, as an engine that decodes in graphs runs it, and returns the graph's static output (what
    the replay wrote). Raises if fn cannot be captured."""
    side = _Stream(device)
    side.wait_stream(_current(device))
    with _stream(side):
        for inputs in bufs:
            fn(*inputs)
    _current(device).wait_stream(side)
    out = []
    for inputs in bufs:
        g = _CUDAGraph()
        with _graph(g):
            static = fn(*inputs)
        out.append(lambda *_, g=g, static=static: (g.replay(), static)[1])
    return out


def time_pair(a, b, make_inputs, shape, dtype, device="cpu", iters=50, warmup=5, n_buffers=4, seed=0,
              graphs=False) -> dict:
    """a and b timed in pairs on the same inputs, in a seeded random order per pair, so drift (clocks, heat, other
    load) lands on both: each one's stats plus the per-pair ratio b/a (above 1: a is faster). graphs=True (CUDA):
    each is captured once per input set and timed as a graph replay, so host launch overhead is not counted. Returns a's
    outputs of its timed calls too: the caller checks them, since a kernel may behave differently when timed."""
    bufs = [make_inputs(shape, dtype, device, _gen(seed + i)) for i in range(n_buffers)]
    graphs = graphs and torch.device(device).type == "cuda"
    ga, gb = (graphed(a, bufs, device), graphed(b, bufs, device)) if graphs else (None, None)
    flush, order, ta, tb, outs = _flush(device), _Random(seed), [], [], [None] * n_buffers
    for i in range(warmup + iters):
        inputs = bufs[i % n_buffers]
        fa, fb = (ga[i % n_buffers], gb[i % n_buffers]) if graphs else (a, b)
        first_a = order.random() < 0.5
        x, ox = _one(fa if first_a else fb, inputs, device, flush, spin=graphs)
        y, oy = _one(fb if first_a else fa, inputs, device, flush, spin=graphs)
        outs[i % n_buffers] = ox if first_a else oy
        if i >= warmup:
            ta.append(x if first_a else y)
            tb.append(y if first_a else x)
    r25, rmed, r75 = _quantiles([y / x for x, y in zip(ta, tb) if x > 0], n=4)
    clock = "cuda_graph" if graphs else "cuda_events" if torch.device(device).type == "cuda" else "host"
    return {"a": _stats(ta), "b": _stats(tb), "ratio": {"median": rmed, "p25": r25, "p75": r75}, "iters": iters,
            "clock": clock, "inputs": bufs, "outputs_a": outs}     # a's output of its last timed call per input set


def check_memoization(fn, make_inputs, shape, dtype, device="cpu", iters=30, warmup=3, ratio=0.5) -> dict:
    """Gate {passed, reason, evidence}: fails if repeated identical inputs run faster than ratio x fresh inputs."""
    rep = time_kernel(fn, make_inputs, shape, dtype, device, iters, warmup, n_buffers=1)
    new = time_kernel(fn, make_inputs, shape, dtype, device, iters, warmup, fresh=True)
    r = rep["median_ms"] / new["median_ms"]
    reason = (f"median on repeated identical inputs {rep['median_ms']:.3g} ms = {r:.2f}x median on fresh inputs "
              f"{new['median_ms']:.3g} ms" + ("" if r >= ratio else f" (< {ratio})"))
    return {"passed": r >= ratio, "reason": reason, "evidence": {"repeated": rep, "fresh": new, "ratio": r}}
