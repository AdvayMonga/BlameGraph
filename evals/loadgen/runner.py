"""Drive a server with a workload, open loop (requests fire on schedule) or closed loop (a fixed number of clients),
and summarize it against latency limits; goodput is the highest load at which ATTAINMENT of requests meet both."""
from __future__ import annotations

import asyncio
import math
import multiprocessing as mp
import os
import queue
import resource
import threading
import time
from dataclasses import dataclass
from typing import Callable

from .client import Request, Row, asend, recount, send

MAX_CLIENT_LAG_S = 0.010     # p99 send lag limit, seconds
MAX_CLIENT_CPU = 0.8         # CPU / wall limit of a closed-loop client process
ATTAINMENT = 0.99            # goodput: share of requests that must meet both limits


@dataclass(frozen=True)
class Limits:
    ttft_s: float
    tpot_s: float


INTERACTIVE = Limits(0.5, 0.030)      # MLPerf interactive limits, checked at p99
CONVERSATIONAL = Limits(2.0, 0.100)   # MLPerf conversational limits, checked at p99


def run_open(url: str, model: str, requests: list[Request], timeout: float = 600, count_tokens=None,
             procs: int = 4, chat_kwargs: dict | None = None, cpus: list[int] | None = None) -> list[Row]:
    """Fire each request at its `at_s` from `procs` client processes on one shared start; rows are in input order.
    `cpus`: the cores the client processes run on (Linux)."""
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
    """Each (fn, args, kw) in its own process as fn(*args, t0, **kw) on one shared start t0; returns each job's rows."""
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
    """Send reqs at their `at_s` from one event loop; t0 is the shared perf_counter start."""
    clock = lambda: time.perf_counter() - t0

    async def one(req):
        try:
            return await asend(url, model, req, clock, timeout, chat_kwargs=chat_kwargs)
        except Exception as e:                    # the client itself failed: still one row
            return Row(req.id, "crash", req.at_s, 0.0, detail=f"client {type(e).__name__}: {e}"[:200])

    async def every():
        return await asyncio.gather(*(one(r) for r in reqs))
    return list(asyncio.run(every()))


def run_closed(url: str, model: str, requests, concurrency: int, duration_s: float | None = None,
               timeout: float = 600, count_tokens=None, chat_kwargs: dict | None = None, procs: int = 1,
               cpus: list[int] | None = None, min_requests: int = 0, max_s: float | None = None) -> tuple[list[Row], float]:
    """Closed loop: `concurrency` clients take requests in order until `requests` runs out or the time limits pass.
    Returns (rows, elapsed seconds). Stops at `duration_s` once `min_requests` were started, or at `max_s`."""
    procs = min(procs, concurrency) if hasattr(requests, "part") else 1
    if procs == 1:
        _pin(cpus)
        rows = _closed_all(url, model, requests, concurrency, duration_s, timeout, time.perf_counter(),
                           chat_kwargs=chat_kwargs, min_requests=min_requests, max_s=max_s)
    else:
        sizes = [concurrency // procs + (k < concurrency % procs) for k in range(procs)]
        parts = _spawn([(_closed_all, (url, model, requests.part(k, procs), sizes[k], duration_s, timeout),
                         {"chat_kwargs": chat_kwargs, "min_requests": -(-min_requests * sizes[k] // concurrency),
                          "max_s": max_s})
                        for k in range(procs)], cpus)
        rows = [r for part in parts for r in part]
    return [recount(r, count_tokens) for r in rows], max((r.scheduled_s + (r.e2e_s or 0) for r in rows), default=0.0)


def _nofile(n: int):
    """Raise this process's open-file soft limit to `n`."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < n:
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(n, hard) if hard != resource.RLIM_INFINITY else n, hard))


def _closed_all(url, model, requests, concurrency: int, duration_s: float | None, timeout: float, t0: float,
                chat_kwargs: dict | None = None, min_requests: int = 0, max_s: float | None = None) -> list[Row]:
    """One process's closed-loop clients on the shared start t0; each row carries the process's CPU share."""
    _nofile(concurrency + 64)
    clock, c0, w0 = (lambda: time.perf_counter() - t0), time.process_time(), time.perf_counter()
    lock, it, rows, started = threading.Lock(), iter(requests), [], 0

    def client():
        nonlocal started
        while True:
            with lock:
                if duration_s is not None and clock() >= duration_s and (started >= min_requests
                                                                        or (max_s is not None and clock() >= max_s)):
                    return
                req = next(it, None)
                started += 1
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
    """Highest load x with probe(x)["meets_limits"]: grow or shrink geometrically from `start`, then bisect.
    Returns {goodput, fails_at (lowest failing load), stopped: resolved|invalid|below_min|probe_budget, probes}."""
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
