"""Drive a server with a workload and summarize it against latency limits.

open loop    every request fires at its scheduled time whatever the server is doing (queues can grow)
closed loop  a fixed number of clients, each sending its next request when the last one finishes
goodput      the highest load at which >= 99% of requests meet both the TTFT and the TPOT limit
"""
from __future__ import annotations

import math
import multiprocessing as mp
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable

from .client import Request, Row, recount, send, wall_clock

MAX_CLIENT_LAG_S = 0.010     # p99 send lag above this means the client, not the server, was the bottleneck
ATTAINMENT = 0.99            # goodput: share of requests that must meet both limits


@dataclass(frozen=True)
class Limits:
    ttft_s: float
    tpot_s: float


INTERACTIVE = Limits(0.5, 0.030)      # MLPerf Llama-3.1-8B interactive, checked at p99
CONVERSATIONAL = Limits(2.0, 0.100)   # MLPerf Llama-3.1-8B conversational, checked at p99


def run_open(url: str, model: str, requests: list[Request], timeout: float = 600, count_tokens=None,
             procs: int = 4) -> list[Row]:
    """Fire each request at its `at_s`; returns one row per request in input order. The schedule is dealt
    round-robin to `procs` client processes on one shared start time, so parsing many concurrent streams can't
    starve the sender of interpreter time (the per-request lag check would void the run)."""
    order = sorted(range(len(requests)), key=lambda i: requests[i].at_s)
    procs = max(1, min(procs, len(requests)))
    if procs == 1:
        rows = dict(zip(order, _fire_all(url, model, [requests[i] for i in order], timeout, time.perf_counter())))
    else:
        ctx = mp.get_context("spawn")
        ready, results, go, t0 = ctx.Queue(), ctx.Queue(), ctx.Event(), ctx.Value("d", 0.0)
        slices = [order[k::procs] for k in range(procs)]
        workers = [ctx.Process(target=_worker, args=(k, url, model, [requests[i] for i in sl], timeout, ready, go,
                                                     t0, results), daemon=True) for k, sl in enumerate(slices)]
        for w in workers:
            w.start()
        try:
            for _ in workers:
                ready.get(timeout=120)
        except queue.Empty:
            raise RuntimeError("client worker processes did not start (spawn needs an importable __main__)") from None
        t0.value = time.perf_counter() + 0.05
        go.set()
        rows = {}
        while len(rows) < len(order):
            try:
                k, part = results.get(timeout=5)
            except queue.Empty:
                dead = [w.exitcode for w in workers if w.exitcode not in (None, 0)]
                if dead:
                    raise RuntimeError(f"client worker exited with {dead[0]}") from None
                continue
            rows.update(zip(slices[k], part))
        for w in workers:
            w.join()
    return [recount(rows[i], count_tokens) for i in range(len(requests))]


def _worker(k, url, model, reqs, timeout, ready, go, t0, results):
    ready.put(k)
    go.wait()
    results.put((k, _fire_all(url, model, reqs, timeout, t0.value)))


def _fire_all(url, model, reqs: list[Request], timeout: float, t0: float, lead_s: float = 0.005) -> list[Row]:
    """reqs in schedule order, each on its own thread started `lead_s` early so it is awake at its send time.
    t0 is the shared perf_counter start (CLOCK_MONOTONIC is system-wide, so processes agree on it)."""
    clock = lambda: time.perf_counter() - t0
    rows: list[Row | None] = [None] * len(reqs)

    def fire(i):
        _wait_until(clock, reqs[i].at_s)
        try:
            rows[i] = send(url, model, reqs[i], reqs[i].at_s, clock, timeout)
        except Exception as e:                    # the client itself failed: still one row, never a lost request
            rows[i] = Row(reqs[i].id, "crash", reqs[i].at_s, 0.0, detail=f"client {type(e).__name__}: {e}"[:200])

    threads = []
    for i, req in enumerate(reqs):
        _wait_until(clock, req.at_s - lead_s)
        t = threading.Thread(target=fire, args=(i,), daemon=True)
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    return rows


def _wait_until(clock, t: float, spin_s: float = 0.002):
    """Sleep until `spin_s` before t, then yield-spin: OS sleeps overshoot by milliseconds, which would show up as lag."""
    while (d := t - clock()) > 0:
        time.sleep(d - spin_s if d > spin_s + 0.001 else 0)


def run_closed(url: str, model: str, requests, concurrency: int, duration_s: float | None = None,
               timeout: float = 600, count_tokens=None) -> tuple[list[Row], float]:
    """`concurrency` clients take requests in order from `requests` (any iterable, consumed lazily) until it (or
    `duration_s`) runs out. Returns (rows, elapsed seconds). A request's scheduled time is when its client became free."""
    clock = wall_clock()
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
                row = send(url, model, req, clock(), clock, timeout, count_tokens)
            except Exception as e:
                row = Row(req.id, "crash", clock(), 0.0, detail=f"client {type(e).__name__}: {e}"[:200])
            with lock:
                rows.append(row)

    threads = [threading.Thread(target=client, daemon=True) for _ in range(concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return rows, clock()


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
    return {
        "n": n, **counts, "attainment": len(meets) / n if n else 0.0,
        "meets_limits": bool(n) and len(meets) / n >= ATTAINMENT,
        "ttft_p50_s": _q(ttft, 0.5), "ttft_p99_s": _q(ttft, 0.99), "tpot_p50_s": _q(tpot, 0.5), "tpot_p99_s": _q(tpot, 0.99),
        "output_tokens": sum(r.n_tokens for r in ok), "elapsed_s": span,
        "req_per_s": len(ok) / span if span else 0.0, "good_req_per_s": len(meets) / span if span else 0.0,
        "output_tok_per_s": sum(r.n_tokens for r in ok) / span if span else 0.0,
        "client_lag_p99_s": lag99, "limits": {"ttft_s": limits.ttft_s, "tpot_s": limits.tpot_s},
        "valid": not reasons, "invalid_reasons": reasons,
    }


def find_goodput(probe: Callable[[float], dict], start: float, max_probes: int = 10, rel_tol: float = 0.1,
                 min_load: float | None = None) -> dict:
    """Highest load x with probe(x)["meets_limits"]: double from `start` until a probe fails, then bisect.
    An invalid probe (client bottleneck, silent drops, crashes) stops the search, as does halving below
    `min_load` (default start/16: a server that fails there has no goodput worth finding). Returns {goodput, probes}."""
    min_load = start / 16 if min_load is None else min_load
    probes, lo, hi, x = [], 0.0, None, start
    for _ in range(max_probes):
        s = probe(x)
        probes.append({"load": x, **s})
        if not s["valid"]:
            break
        if s["meets_limits"]:
            lo = x
        else:
            hi = x
        if hi is None:
            x *= 2
        elif lo == 0.0:
            x = hi / 2
            if x < min_load:
                break
        elif (hi - lo) / hi <= rel_tol:
            break
        else:
            x = (lo + hi) / 2
    return {"goodput": lo or None, "probes": probes}
