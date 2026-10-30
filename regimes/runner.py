"""Drive a server with a workload and summarize it against latency limits.

open loop    every request fires at its scheduled time whatever the server is doing (queues can grow)
closed loop  a fixed number of clients, each sending its next request when the last one finishes
goodput      the highest load at which >= 99% of requests meet both the TTFT and the TPOT limit
"""
from __future__ import annotations

import asyncio
import math
import multiprocessing as mp
import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable

from .client import Request, Row, asend, recount, send

MAX_CLIENT_LAG_S = 0.010     # p99 send lag above this means the client, not the server, was the bottleneck
MAX_CLIENT_CPU = 0.8         # a closed-loop client process busier than this (CPU / wall) may have read streams late
ATTAINMENT = 0.99            # goodput: share of requests that must meet both limits


@dataclass(frozen=True)
class Limits:
    ttft_s: float
    tpot_s: float


INTERACTIVE = Limits(0.5, 0.030)      # MLPerf Llama-3.1-8B interactive, checked at p99
CONVERSATIONAL = Limits(2.0, 0.100)   # MLPerf Llama-3.1-8B conversational, checked at p99


def run_open(url: str, model: str, requests: list[Request], timeout: float = 600, count_tokens=None,
             procs: int = 4, chat_kwargs: dict | None = None, cpus: list[int] | None = None) -> list[Row]:
    """Fire each request at its `at_s`; returns one row per request in input order. The schedule is dealt
    round-robin to `procs` client processes on one shared start time, so parsing many concurrent streams can't
    starve the sender of interpreter time (the per-request lag check would void the run). `cpus`: the cores the
    client processes run on (Linux), apart from the server's."""
    order = sorted(range(len(requests)), key=lambda i: requests[i].at_s)
    procs = max(1, min(procs, len(requests)))
    if procs == 1:
        _pin(cpus)
        rows = dict(zip(order, _fire_all(url, model, [requests[i] for i in order], timeout, time.perf_counter(),
                                         chat_kwargs=chat_kwargs)))
    else:
        slices = [order[k::procs] for k in range(procs)]
        parts = _spawn([(_fire_all, (url, model, [requests[i] for i in sl], timeout), {"chat_kwargs": chat_kwargs})
                        for sl in slices], cpus)
        rows = {i: row for sl, part in zip(slices, parts) for i, row in zip(sl, part)}
    return [recount(rows[i], count_tokens) for i in range(len(requests))]


def _pin(cpus: list[int] | None):
    if cpus and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, cpus)


def _spawn(jobs: list[tuple], cpus: list[int] | None) -> list[list[Row]]:
    """Each (fn, args, kw) in its own client process, all started together: fn(*args, t0, **kw) on one shared
    perf_counter start t0 (CLOCK_MONOTONIC is system-wide, so processes agree on it). Returns each job's rows."""
    ctx = mp.get_context("spawn")
    ready, results, go, t0 = ctx.Queue(), ctx.Queue(), ctx.Event(), ctx.Value("d", 0.0)
    workers = [ctx.Process(target=_worker, args=(k, job, cpus, ready, go, t0, results), daemon=True)
               for k, job in enumerate(jobs)]
    for w in workers:
        w.start()
    try:
        for _ in workers:
            ready.get(timeout=120)
    except queue.Empty:
        raise RuntimeError("client worker processes did not start (spawn needs an importable __main__)") from None
    t0.value = time.perf_counter() + 0.05
    go.set()
    parts: dict[int, list[Row]] = {}
    while len(parts) < len(workers):
        try:
            k, part = results.get(timeout=5)
        except queue.Empty:
            dead = [w.exitcode for w in workers if w.exitcode not in (None, 0)]
            if dead:
                raise RuntimeError(f"client worker exited with {dead[0]}") from None
            continue
        parts[k] = part
    for w in workers:
        w.join()
    return [parts[k] for k in range(len(workers))]


def _worker(k, job, cpus, ready, go, t0, results):
    fn, args, kw = job
    _pin(cpus)
    ready.put(k)
    go.wait()
    results.put((k, fn(*args, t0.value, **kw)))


def _fire_all(url, model, reqs: list[Request], timeout: float, t0: float, chat_kwargs: dict | None = None) -> list[Row]:
    """reqs sent at their `at_s` from one event loop (`client.asend`). t0 is the shared perf_counter start
    (CLOCK_MONOTONIC is system-wide, so processes agree on it)."""
    clock = lambda: time.perf_counter() - t0

    async def one(req):
        try:
            return await asend(url, model, req, clock, timeout, chat_kwargs=chat_kwargs)
        except Exception as e:                    # the client itself failed: still one row, never a lost request
            return Row(req.id, "crash", req.at_s, 0.0, detail=f"client {type(e).__name__}: {e}"[:200])

    async def every():
        return await asyncio.gather(*(one(r) for r in reqs))
    return list(asyncio.run(every()))


def run_closed(url: str, model: str, requests, concurrency: int, duration_s: float | None = None,
               timeout: float = 600, count_tokens=None, chat_kwargs: dict | None = None, procs: int = 1,
               cpus: list[int] | None = None) -> tuple[list[Row], float]:
    """`concurrency` clients take requests in order from `requests` (any iterable, consumed lazily) until it (or
    `duration_s`) runs out. Returns (rows, elapsed seconds). A request's scheduled time is when its client became free.
    With `procs` > 1 and a `requests` that splits (`part(k, n)`, e.g. `workload.Cycle`), the clients are dealt to that
    many processes on one shared start, so parsing many streams can't starve one interpreter; `cpus` as in run_open."""
    procs = min(procs, concurrency) if hasattr(requests, "part") else 1
    if procs == 1:
        _pin(cpus)
        rows = _closed_all(url, model, requests, concurrency, duration_s, timeout, time.perf_counter(),
                           chat_kwargs=chat_kwargs)
    else:
        sizes = [concurrency // procs + (k < concurrency % procs) for k in range(procs)]
        parts = _spawn([(_closed_all, (url, model, requests.part(k, procs), sizes[k], duration_s, timeout),
                         {"chat_kwargs": chat_kwargs}) for k in range(procs)], cpus)
        rows = [r for part in parts for r in part]
    return [recount(r, count_tokens) for r in rows], max((r.scheduled_s + (r.e2e_s or 0) for r in rows), default=0.0)


def _closed_all(url, model, requests, concurrency: int, duration_s: float | None, timeout: float, t0: float,
                chat_kwargs: dict | None = None) -> list[Row]:
    """One process's closed-loop clients on the shared start t0; every client stops once `duration_s` has passed.
    Each row carries the process's CPU share over the call: closed loop has no send schedule to lag behind, so a busy
    client process is the sign that it read its streams late."""
    clock, c0, w0 = (lambda: time.perf_counter() - t0), time.process_time(), time.perf_counter()
    lock, it, rows = threading.Lock(), iter(requests), []

    def client():
        while True:
            with lock:
                if duration_s is not None and clock() >= duration_s:
                    return
                req = next(it, None)
            if req is None:
                return
            try:
                row = send(url, model, req, clock(), clock, timeout, chat_kwargs=chat_kwargs)
            except Exception as e:
                row = Row(req.id, "crash", clock(), 0.0, detail=f"client {type(e).__name__}: {e}"[:200])
            with lock:
                rows.append(row)

    threads = [threading.Thread(target=client, daemon=True) for _ in range(concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    share = (time.process_time() - c0) / max(time.perf_counter() - w0, 1e-9)
    for r in rows:
        r.client_cpu = share
    return rows


def _q(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, math.ceil(q * len(xs)) - 1))]


def summarize(rows: list[Row], limits: Limits, elapsed_s: float | None = None) -> dict:
    """Latency percentiles, failures by kind, share meeting both limits, throughput, and whether the run is valid."""
    n = len(rows)
    ok = [r for r in rows if r.status == "ok"]
    ttft = [r.ttft_s for r in ok if r.ttft_s is not None]
    tpot = [r.tpot_s for r in ok if r.tpot_s is not None]
    meets = [r for r in ok if r.ttft_s is not None and r.ttft_s <= limits.ttft_s
             and r.tpot_s is not None and r.tpot_s <= limits.tpot_s]
    lags = [r.lag_s for r in rows]
    span = elapsed_s if elapsed_s is not None else max((r.scheduled_s + (r.e2e_s or 0) for r in rows), default=0.0)
    counts = {s: sum(r.status == s for r in rows) for s in ("ok", "error", "silent_drop", "truncated", "crash")}
    reasons = [] if n else ["no requests were sent"]
    if counts["silent_drop"] or counts["truncated"]:
        reasons.append(f"{counts['silent_drop']} silent drops and {counts['truncated']} truncated streams")
    if counts["crash"]:
        reasons.append(f"{counts['crash']} requests got no HTTP response (connection failed or timed out)")
    lag99 = _q(lags, 0.99)
    if lag99 is not None and lag99 > MAX_CLIENT_LAG_S:
        reasons.append(f"client lag p99 {lag99 * 1000:.1f} ms > {MAX_CLIENT_LAG_S * 1000:.0f} ms: the client was the bottleneck")
    cpu = max((r.client_cpu for r in rows if r.client_cpu is not None), default=None)
    if cpu is not None and cpu > MAX_CLIENT_CPU:
        reasons.append(f"a client process ran at {cpu:.0%} CPU > {MAX_CLIENT_CPU:.0%}: the client may have been the bottleneck")
    return {
        "n": n, **counts, "attainment": len(meets) / n if n else 0.0,
        "meets_limits": bool(n) and len(meets) / n >= ATTAINMENT,
        "ttft_p50_s": _q(ttft, 0.5), "ttft_p99_s": _q(ttft, 0.99), "tpot_p50_s": _q(tpot, 0.5), "tpot_p99_s": _q(tpot, 0.99),
        "output_tokens": sum(r.n_tokens for r in ok), "elapsed_s": span,
        "req_per_s": len(ok) / span if span else 0.0, "good_req_per_s": len(meets) / span if span else 0.0,
        "output_tok_per_s": sum(r.n_tokens for r in ok) / span if span else 0.0,
        "client_lag_p99_s": lag99, "client_cpu_max": cpu, "limits": {"ttft_s": limits.ttft_s, "tpot_s": limits.tpot_s},
        "valid": not reasons, "invalid_reasons": reasons,
    }


def find_goodput(probe: Callable[[float], dict], start: float, max_probes: int = 40, rel_tol: float = 0.1,
                 min_load: float | None = None, growth: float = 2.0) -> dict:
    """Highest load x with probe(x)["meets_limits"]: grow by `growth` from `start` until a probe fails (or shrink by
    it until one passes), the step squaring on each move up to 2x, then bisect until the bracket is within `rel_tol`.
    An invalid probe (client bottleneck, silent drops, crashes) stops the search, as does going below `min_load`
    (default start/16: a server that fails there has no goodput worth finding). Returns {goodput, fails_at (the lowest
    failing load, None if none failed), stopped (resolved | invalid | below_min | probe_budget), probes}."""
    min_load = start / 16 if min_load is None else min_load
    probes, lo, hi, x, step, stopped = [], 0.0, None, start, growth, "probe_budget"
    for _ in range(max_probes):
        s = probe(x)
        probes.append({"load": x, **s})
        if not s["valid"]:
            stopped = "invalid"
            break
        if s["meets_limits"]:
            lo = x
        else:
            hi = x
        if hi is None:
            x *= step
            step = min(step * step, 2.0)
        elif lo == 0.0:
            x = hi / step
            step = min(step * step, 2.0)
            if x < min_load:
                stopped = "below_min"
                break
        elif (hi - lo) / hi <= rel_tol:
            stopped = "resolved"
            break
        else:
            x = (lo + hi) / 2
    return {"goodput": lo or None, "fails_at": hi, "stopped": stopped, "probes": probes}
