"""The eight regimes. Each measures one way an inference server is used and reports one headline number.

  single_stream             one user at a time: p99 time per output token (interactive limits)
  saturated                 closed-loop concurrency sweep: goodput in req/s
  bursty                    the corpus burst trace, sped up until it breaks: goodput as a speed-up factor
  long_prompt_short_output  long-context prompts, 64-token answers, Poisson arrivals: goodput in req/s
  short_prompt_long_output  short prompts, answers up to 1024 tokens, Poisson arrivals: goodput in req/s
  shared_prefix_multi_turn  conversations resending their history under one shared system prompt: goodput in sessions/s
  overload                  Poisson at 2x the measured goodput: does it degrade honestly (explicit errors, no silent drops)
  cold_start                process launch to first correct token, empty compile caches, plus the warm figure

Everything except single_stream and cold_start is checked against the conversational limits (TTFT 2 s, TPOT 100 ms
at p99); goodput is the highest load at which >= 99% of requests meet both. Loads are found with short probes, then
confirmed for the tier's full duration.
"""
from __future__ import annotations

import os
import random
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import workload as W
from .client import Request, send, wall_clock
from .runner import CONVERSATIONAL, INTERACTIVE, Limits, find_goodput, run_closed, run_open, summarize

TIERS = {"short": {"probe_s": 30.0, "final_s": 60.0}, "full": {"probe_s": 60.0, "final_s": 600.0}}
MAX_STRETCH = 4.0


@dataclass
class Ctx:
    url: str
    model: str
    corpus: str | None = None          # a corpus/ dir; None = synthetic prompts (tests, smoke runs)
    workload: str | None = None        # a workloads/ JSONL file instead of the corpus: every regime draws from all of it
    split: str = "seen"
    tier: str = "short"
    seed: int = 0
    timeout: float = 600.0
    count_tokens: object = None        # reference tokenizer's encode, to re-count output tokens; None = server usage
    chat_kwargs: dict = field(default_factory=dict)   # the target's chat_template_kwargs
    min_requests: int = 100            # a probe runs until it holds this many: 99% of fewer means little
    interactive: Limits = INTERACTIVE  # MLPerf limits; probes must last well beyond the TTFT limit to see overload
    conversational: Limits = CONVERSATIONAL
    rows: list | None = None           # when a list, every measured row lands here (no text), tagged with its call
    _pools: dict = field(default_factory=dict)
    _calls: int = 0

    @property
    def probe_s(self) -> float:
        return TIERS[self.tier]["probe_s"]

    @property
    def final_s(self) -> float:
        return TIERS[self.tier]["final_s"]

    def pool(self, cls: str | None = None) -> list[dict]:
        """Corpus records of one class (None = every class) for this split."""
        if self.workload is not None:
            cls = "workload"
        if cls not in self._pools:
            if self.workload is not None:
                self._pools[cls] = W.workload_file(self.workload)
            elif self.corpus is None:
                self._pools[cls] = W.synthetic(100, seed=self.seed, prefix=cls or "mix")
            else:
                classes = [cls] if cls else W.corpus_classes(self.corpus)
                self._pools[cls] = [r for c in classes for r in W.corpus(self.corpus, c, self.split)]
        return self._pools[cls]

    def duration(self, rate: float, final: bool) -> float:
        """Tier length, stretched to hold `min_requests` at `rate` but never past `MAX_STRETCH` x (a server that
        needs longer than that is not worth measuring at that load)."""
        base = self.final_s if final else self.probe_s
        return min(max(base, self.min_requests / rate), base * MAX_STRETCH)

    def run_open(self, reqs):
        started = time.time()
        return self._keep(run_open(self.url, self.model, reqs, self.timeout, self.count_tokens,
                                   chat_kwargs=self.chat_kwargs), started)

    def run_closed(self, reqs, c, duration):
        started = time.time()
        rows, elapsed = run_closed(self.url, self.model, reqs, c, duration, self.timeout, self.count_tokens,
                                   chat_kwargs=self.chat_kwargs)
        return self._keep(rows, started), elapsed

    def _keep(self, rows, started: float):
        """Rows into the sink: `call` numbers the run_open/run_closed call, `call_started_at` is its epoch start."""
        if self.rows is not None:
            self.rows.extend({"call": self._calls, "call_started_at": started, **r.to_dict()} for r in rows)
            self._calls += 1
        return rows

    @classmethod
    def from_target(cls, t, url: str, **kw) -> "Ctx":
        """A Ctx for the target: its model, chat kwargs, corpus and limits; `kw` overrides (tier, split, seed...)."""
        from .runner import Limits
        return cls(url, t.model, str(t.corpus_dir) if t.corpus_dir.is_dir() else None, chat_kwargs=t.chat_kwargs,
                   interactive=Limits(t.interactive.ttft_s, t.interactive.tpot_s),
                   conversational=Limits(t.conversational.ttft_s, t.conversational.tpot_s), **kw)


def _cycle(order: list[dict], tag: str = ""):
    """Requests built one at a time, cycling the pool: closed-loop runs consume only what they send."""
    k = 0
    while True:
        for r in order:
            yield W.to_request(r, rid=f"{r['id']}#{tag}{k}")
            k += 1


def _warmup(ctx: Ctx, pool: list[dict], n: int = 8):
    run_closed(ctx.url, ctx.model, [W.to_request(r, max_tokens=32) for r in pool[:n]], 4, None, ctx.timeout,
               chat_kwargs=ctx.chat_kwargs)


def _result(name, objective, value, direction, summary, **extra) -> dict:
    return {"regime": name, "objective": objective, "value": value, "better": direction, "summary": summary,
            "valid": summary["valid"] if summary else False,
            "invalid_reasons": summary["invalid_reasons"] if summary else ["no measurement"], **extra}


def _search_and_confirm(measure, start: float) -> tuple[float | None, dict | None, list]:
    """find_goodput with measure(load, final=False), then confirm with measure(load, final=True), backing off 15%
    up to 5 times. Returns (confirmed load or None, final summary, probes)."""
    search = find_goodput(lambda x: measure(x, False), start)
    load, final = search["goodput"], None
    for _ in range(6):
        if load is None:
            break
        final = measure(load, True)
        if final["meets_limits"] or not final["valid"]:
            break
        load *= 0.85
    ok = final is not None and final["meets_limits"] and final["valid"]
    if final is None and search["probes"]:
        final = search["probes"][-1]
    return (load if ok else None), final, search["probes"]


def _goodput_regime(ctx: Ctx, name: str, objective: str, build, limits: Limits, per_unit: float = 1.0,
                    ramp_s: float = 0.0) -> dict:
    """Open-loop goodput: `build(load, duration)` makes the schedule; `per_unit` = requests per unit of load (turns
    per session for sessions/s). Probes last ctx.probe_s, or longer when needed to hold ctx.min_requests; the search
    starts at the load that fills one probe. With `ramp_s`, the schedule starts that much early and only requests
    scheduled inside [ramp_s, ramp_s + duration] are scored, so multi-turn load is at steady state."""
    def measure(load, final):
        duration = ctx.duration(load * per_unit, final)
        rows = ctx.run_open(build(load, duration + ramp_s))
        return summarize([r for r in rows if ramp_s <= r.scheduled_s < ramp_s + duration], limits, duration)
    load, final, probes = _search_and_confirm(measure, ctx.min_requests / (ctx.probe_s * per_unit))
    return _result(name, objective, load, "higher", final, probes=probes)


def single_stream(ctx: Ctx) -> dict:
    pool = ctx.pool("steady_interactive")
    _warmup(ctx, pool)
    order = random.Random(ctx.seed).sample(pool, len(pool))
    rows, elapsed = ctx.run_closed(_cycle(order), 1, ctx.final_s)
    s = summarize(rows, ctx.interactive, elapsed)
    return _result("single_stream", "tpot_p99_s", s["tpot_p99_s"], "lower", s, ttft_p99_s=s["ttft_p99_s"])


def saturated(ctx: Ctx, max_concurrency: int = 512) -> dict:
    """Doubling concurrency; goodput = good req/s at the best level whose requests meet the limits."""
    pool = ctx.pool(None)
    _warmup(ctx, pool)
    order = random.Random(ctx.seed).sample(pool, len(pool))
    levels, best, c, misses = [], None, 1, 0
    while c <= max_concurrency:
        rows, elapsed = ctx.run_closed(_cycle(order, f"c{c}"), c, ctx.probe_s)
        s = summarize(rows, ctx.conversational, elapsed)
        levels.append({"concurrency": c, **s})
        if not s["valid"]:
            break
        if s["meets_limits"] and (best is None or s["good_req_per_s"] > best["good_req_per_s"]):
            best, misses = levels[-1], 0
        elif not s["meets_limits"]:
            misses += 1
            if misses >= 2:
                break
        c *= 2
    final, best = None, None
    passing = sorted((lv for lv in levels if lv["meets_limits"]), key=lambda lv: -lv["good_req_per_s"])
    for cand in passing[:3]:                       # confirm at full length; a short probe can flatter a level
        rows, elapsed = ctx.run_closed(_cycle(order, f"f{cand['concurrency']}"), cand["concurrency"], ctx.final_s)
        final = summarize(rows, ctx.conversational, elapsed)
        if final["meets_limits"] or not final["valid"]:
            best = cand
            break
    ok = final is not None and final["meets_limits"] and final["valid"]
    return _result("saturated", "goodput_req_per_s", final["good_req_per_s"] if ok else None, "higher",
                   final or (levels[-1] if levels else None), concurrency=best and best["concurrency"], levels=levels)


def bursty(ctx: Ctx) -> dict:
    """The spike trace replayed at `speed` x its real rate (burst shape kept); short tier crops to the peak window."""
    if ctx.workload is not None:
        return {**_result("bursty", "speedup", None, "higher", None),
                "invalid_reasons": ["a --workload file has no arrival times to replay"]}
    trace = ctx.pool("spike")
    _warmup(ctx, trace)
    window = W.peak_window(trace) if ctx.tier == "short" else None

    def measure(speed, _final):                     # the trace is its own duration; confirming is a rerun
        return summarize(ctx.run_open(W.replay(trace, speed, window)), ctx.conversational)
    speed, final, probes = _search_and_confirm(measure, 1.0)
    return _result("bursty", "speedup", speed, "higher", final, probes=probes,
                   good_req_per_s=final and final["good_req_per_s"])


def long_prompt_short_output(ctx: Ctx) -> dict:
    pool = ctx.pool("long_context")
    _warmup(ctx, pool)
    build = lambda rate, dur: W.poisson(pool, rate, dur, ctx.seed, max_tokens=64)
    return _goodput_regime(ctx, "long_prompt_short_output", "goodput_req_per_s", build, ctx.conversational)


SHORT_PROMPT_TOKENS = 512
LONG_ANSWER = "\n\nAnswer in depth, with explanation and examples."


def short_prompt_long_output(ctx: Ctx) -> dict:
    allp = ctx.pool(None)
    pool = [r for r in allp if (r.get("build_prompt_tokens") or 0) <= SHORT_PROMPT_TOKENS and not r.get("messages")]
    pool = pool or sorted(allp, key=lambda r: r.get("build_prompt_tokens") or 0)[: max(1, len(allp) // 4)]
    _warmup(ctx, pool)
    build = lambda rate, dur: W.poisson(pool, rate, dur, ctx.seed, max_tokens=1024, suffix=LONG_ANSWER)
    return _goodput_regime(ctx, "short_prompt_long_output", "goodput_req_per_s", build, ctx.conversational)


THINK_S, MAX_TURNS = 3.0, 6


def shared_prefix_multi_turn(ctx: Ctx) -> dict:
    pool = ctx.pool(None)
    system = W.shared_system_prompt(seed=ctx.seed)
    _warmup(ctx, pool)
    if ctx.corpus is None and ctx.workload is None:   # synthetic records have no histories: make two-turn ones
        pool = [{**r, "messages": [{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": "ok"},
                                   {"role": "user", "content": "go on"}]} for r in pool]
    convs = W.conversations(pool, MAX_TURNS)
    if not convs:
        return {**_result("shared_prefix_multi_turn", "goodput_sessions_per_s", None, "higher", None),
                "invalid_reasons": ["no multi-turn conversations in the workload"]}
    turns = sum(len(c) for c in convs) / len(convs)
    build = lambda rate, dur: W.sessions(pool, rate, dur, ctx.seed, think_s=THINK_S, system=system, max_turns=MAX_TURNS)
    return _goodput_regime(ctx, "shared_prefix_multi_turn", "goodput_sessions_per_s", build, ctx.conversational, turns,
                           ramp_s=THINK_S * (MAX_TURNS - 1))


def overload(ctx: Ctx, factor: float = 2.0, rate: float | None = None) -> dict:
    """Poisson at `factor` x the steady mix's goodput (searched here unless `rate` is given). Valid only if every
    request either completes or is explicitly rejected: silent drops, truncated streams and no-response are invalid."""
    pool = ctx.pool("steady_interactive")
    _warmup(ctx, pool)
    def measure(r, final):
        duration = ctx.duration(r, final)
        return summarize(ctx.run_open(W.poisson(pool, r, duration, ctx.seed)), ctx.conversational, duration)
    found = None
    if rate is None:
        found = find_goodput(lambda r: measure(r, False), ctx.min_requests / ctx.probe_s)
        rate = found["goodput"]
    if rate is None:
        return _result("overload", "good_req_per_s", None, "higher", found["probes"][-1] if found else None)
    s = measure(rate * factor, True)
    return _result("overload", "good_req_per_s", s["good_req_per_s"], "higher", s, offered_req_per_s=rate * factor,
                   explicit_rejections=s["error"], base_goodput_req_per_s=rate)


PROBE = [{"role": "user", "content": "What is 17 + 25? Reply with the number only."}]


def cold_start(url: str, model: str, cmd: str, expect: str = "42", timeout_s: float = 1800.0, poll_s: float = 0.25,
               warm: bool = True, log_dir: str | None = None) -> dict:
    """Launch `cmd` with empty compile caches (fresh VLLM_CACHE_ROOT / TORCHINDUCTOR_CACHE_DIR / TRITON_CACHE_DIR),
    poll until a reply contains `expect`, and report seconds from launch to that first correct token. With `warm`,
    relaunch on the now-filled caches and report that too. Weights may already be on disk."""
    cache = tempfile.mkdtemp(prefix="bg-coldstart-")
    env = {**os.environ, "VLLM_CACHE_ROOT": f"{cache}/vllm", "TORCHINDUCTOR_CACHE_DIR": f"{cache}/inductor",
           "TRITON_CACHE_DIR": f"{cache}/triton"}
    out = {"regime": "cold_start", "objective": "first_correct_token_s", "better": "lower", "cmd": cmd}
    try:
        for label in (("cold", "warm") if warm else ("cold",)):
            out[label] = _launch_once(url, model, cmd, env, expect, timeout_s, poll_s, log_dir, label)
    finally:
        shutil.rmtree(cache, ignore_errors=True)
    out["value"] = out["cold"].get("first_correct_token_s")
    out["valid"] = out["value"] is not None
    out["invalid_reasons"] = [] if out["valid"] else [out["cold"].get("detail", "no correct token")]
    return out


def _launch_once(url, model, cmd, env, expect, timeout_s, poll_s, log_dir, label) -> dict:
    log = open(Path(log_dir) / f"cold_start_{label}.log", "w") if log_dir else subprocess.DEVNULL
    clock = wall_clock()
    proc = subprocess.Popen(cmd, shell=True, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    attempts, res = 0, {"first_correct_token_s": None}
    try:
        while clock() < timeout_s:
            if proc.poll() is not None:
                res["detail"] = f"process exited with {proc.returncode} before a correct token"
                break
            attempts += 1
            t = clock()
            row = send(url, model, Request(f"cold-{label}-{attempts}", PROBE, 8), t, clock, timeout=30)
            if row.status == "ok" and expect in row.text:
                res = {"first_correct_token_s": row.ttft_s + t, "attempts": attempts, "answer": row.text}
                break
            if row.status == "ok":
                res["detail"] = f"server answered {row.text!r}, expected {expect!r}"
            time.sleep(poll_s)
        else:
            last = res.get("detail")
            res["detail"] = f"no correct token within {timeout_s:.0f} s" + (f"; last: {last}" if last else "")
    finally:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=60)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if log_dir:
            log.close()
    return res


REGIMES = {"single_stream": single_stream, "saturated": saturated, "bursty": bursty,
           "long_prompt_short_output": long_prompt_short_output, "short_prompt_long_output": short_prompt_long_output,
           "shared_prefix_multi_turn": shared_prefix_multi_turn, "overload": overload}
