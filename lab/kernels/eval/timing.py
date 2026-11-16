"""Kernel timing: synchronized, rotated input buffers, L2 flushed on CUDA; plus a memoization check."""
from __future__ import annotations

import random
import statistics
import time

import torch

from .check import _gen

# Bound at import, before any candidate loads.
_perf_counter, _quantiles, _Random = time.perf_counter, statistics.quantiles, random.Random
_Event, _record, _elapsed = torch.cuda.Event, torch.cuda.Event.record, torch.cuda.Event.elapsed_time
_cuda_sync, _mps_sync = torch.cuda.synchronize, torch.mps.synchronize
_CUDAGraph, _graph, _Stream, _stream, _current = (torch.cuda.CUDAGraph, torch.cuda.graph, torch.cuda.Stream,
                                                  torch.cuda.stream, torch.cuda.current_stream)
_sleep = torch.cuda._sleep
SPIN_CYCLES = 1_000_000          # ~0.5 ms of GPU spin queued ahead of a graph replay


def _sync(device):
    kind = torch.device(device).type
    if kind == "cuda":
        _cuda_sync(device)
    elif kind == "mps":
        _mps_sync()


def _one(fn, inputs, device, flush, spin=False) -> tuple[float, object]:
    """One synchronized call's time in ms and its output; on CUDA, L2 flushed first and timed by CUDA events.
    spin=True queues a GPU spin before the start event."""
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


def _fingerprint(inputs) -> torch.Tensor:
    """Every input tensor's bytes as order-sensitive sums of int words, computed on its device."""
    out = []
    for t in inputs:
        if not isinstance(t, torch.Tensor):
            continue
        b = t.detach().reshape(-1).contiguous().view(torch.uint8)
        n = b.numel() - b.numel() % 8
        w = b[:n].view(torch.int64)
        pos = torch.arange(1, w.numel() + 1, device=t.device, dtype=torch.int64)
        out.append(torch.stack([w.sum(), (w * pos).sum(), b[n:].to(torch.int64).sum()]))     # overflow wraps
    return torch.stack([x.cpu() for x in out]) if out else torch.zeros(0, 3, dtype=torch.int64)   # inputs may mix devices


def _unchanged(inputs, before: torch.Tensor, who: str, buffer: int) -> None:
    if not torch.equal(_fingerprint(inputs), before):
        raise ValueError(f"an input of timed buffer {buffer} changed during {who}'s call: a call must leave its inputs "
                         "as they were")


def _stats(times: list[float]) -> dict:
    p25, med, p75 = _quantiles(times, n=4)
    return {"median_ms": med, "iqr_ms": p75 - p25, "p25_ms": p25, "p75_ms": p75}


def _flush(device):
    return torch.empty(256 * 2**20, dtype=torch.int8, device=device) if torch.device(device).type == "cuda" else None


def time_kernel(fn, make_inputs, shape, dtype, device="cpu", iters=50, warmup=5, n_buffers=4, fresh=False,
                seed=0) -> dict:
    """Median and IQR (ms) of fn(*inputs) over n_buffers rotated input sets, each call synchronized, L2 flushed on CUDA.
    fresh=True refills the inputs with new values before every call."""
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
    """One CUDA graph per input set, captured after a warm-up; each entry replays fn and returns its static output.
    Raises if fn cannot be captured."""
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
    """a and b timed in pairs on the same inputs in a seeded random order: each one's stats and the per-pair ratio b/a.
    graphs=True (CUDA) times graph replays. Also returns inputs and a's timed outputs; raises if an input changes."""
    bufs = [make_inputs(shape, dtype, device, _gen(seed + i)) for i in range(n_buffers)]
    graphs = graphs and torch.device(device).type == "cuda"
    ga, gb = (graphed(a, bufs, device), graphed(b, bufs, device)) if graphs else (None, None)
    flush, order, ta, tb, outs = _flush(device), _Random(seed), [], [], [None] * n_buffers
    for i in range(warmup + iters):
        inputs = bufs[i % n_buffers]
        fa, fb = (ga[i % n_buffers], gb[i % n_buffers]) if graphs else (a, b)
        first_a = order.random() < 0.5
        before = _fingerprint(inputs)            # outside the timed window
        x, ox = _one(fa if first_a else fb, inputs, device, flush, spin=graphs)
        _unchanged(inputs, before, "the candidate" if first_a else "the baseline", i % n_buffers)
        y, oy = _one(fb if first_a else fa, inputs, device, flush, spin=graphs)
        _unchanged(inputs, before, "the baseline" if first_a else "the candidate", i % n_buffers)
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
