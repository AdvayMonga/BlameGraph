"""The scheduler simulator (layer 3): the engine's real `ContinuousBatchScheduler` driven one `step()` at a time on
a fake clock, with a backend that does no model work and instead charges a cost model's time for every prefill and
decode. A scheduling change (scheduler.py, scheduling_policy.py, admission, KV reservation) is tried in seconds on a
CPU, on the same traffic the regimes send; `bench` and `submit` stay the judge.

What is real: the scheduler code, its policy, its admission and KV arithmetic, the request objects and their
timestamps, TTFT/TPOT as the regimes define them. What is modelled: how long a prefill of N tokens and a decode step
of B rows over K KV tokens take, and how many tokens each request generates (`CostModel`; its defaults are stand-ins
until fitted from a GPU run's step telemetry: `fit` is phase 2).

Imports the engine, so it runs with the engine's python (`lab.engine.python()`), the engine's `src` on sys.path."""
from __future__ import annotations

import json
import math
import random
import threading
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

from regimes.client import Row

EOS = -1


# -- the clock ------------------------------------------------------------------------------
class Clock:
    """Simulated seconds. The scheduler module's `time` is replaced by this while a simulation runs."""

    def __init__(self, start: float = 1000.0):
        self.now = start

    def perf_counter(self) -> float:
        return self.now

    def monotonic(self) -> float:
        return self.now

    def time(self) -> float:
        return self.now

    def sleep(self, s: float) -> None:
        self.now += s

    def advance(self, s: float) -> None:
        self.now += max(s, 0.0)

    @contextmanager
    def at(self, t: float):
        """Briefly read `t` (an arrival that happened while a step ran), then back to now."""
        saved, self.now = self.now, t
        try:
            yield
        finally:
            self.now = saved


@contextmanager
def installed(clock: Clock):
    """`clock` as `time` for the scheduler and its metrics; restored on exit."""
    from inference_server import metrics, scheduler
    saved = (scheduler.time, metrics.time)
    scheduler.time, metrics.time = clock, clock
    try:
        yield
    finally:
        scheduler.time, metrics.time = saved


# -- the cost model -------------------------------------------------------------------------
@dataclass
class CostModel:
    """Seconds a backend call takes, and how long each answer is. Linear forms, to be fitted from step telemetry
    (phase 2); the defaults are stand-ins (an H200-ish decode step of ~20 ms at batch 1) and say so."""
    prefill_fixed_s: float = 0.010          # per prefill wave
    prefill_per_token_s: float = 0.00005    # per prompt token in the wave
    decode_fixed_s: float = 0.018           # per decode step
    decode_per_row_s: float = 0.0004        # per active row
    decode_per_kv_token_s: float = 2e-7     # per KV token read across the batch
    step_overhead_s: float = 0.0005         # host time per scheduler iteration, even an empty one
    out_tokens_default: int | None = None   # None: each request's `max_tokens` (an upper bound) when nothing better
    source: str = "defaults (not fitted)"

    def prefill_s(self, tokens: int) -> float:
        return self.prefill_fixed_s + self.prefill_per_token_s * tokens

    def decode_s(self, rows: int, kv_tokens: int) -> float:
        return self.decode_fixed_s + self.decode_per_row_s * rows + self.decode_per_kv_token_s * kv_tokens

    def out_tokens(self, rec: dict, rng: random.Random) -> int:
        n = rec.get("expected_output_tokens") or self.out_tokens_default or rec.get("max_tokens") or 1
        return max(1, min(int(n), int(rec.get("max_tokens") or n)))

    @classmethod
    def load(cls, path: str | Path) -> "CostModel":
        return cls(**json.loads(Path(path).read_text()))


# -- the backend ----------------------------------------------------------------------------
class _KV:
    """One row's state: how far it has decoded and when it must stop."""
    __slots__ = ("length", "remaining", "rid")

    def __init__(self, length: int, remaining: int, rid: str):
        self.length, self.remaining, self.rid = length, remaining, rid


class SimBackend:
    """What the scheduler touches on a backend (backends/base.py), charging `cost` to `clock` instead of running.
    KV capacity is a block budget reserved worst-case per request, as the custom backend's `kv_reserve` does."""
    needs_attention_mask = False
    device_str = "cpu"

    def __init__(self, clock: Clock, cost: CostModel, blocks: int = 4096, block_size: int = 16):
        self._lock = threading.Lock()
        self.clock, self.cost = clock, cost
        self.cache_adapter = None
        self.last_cache_hit_tokens = 0
        self.last_batch_cache_hit_tokens = 0
        self.blocks, self.block_size, self.reserved = blocks, block_size, 0
        self.plan: dict[str, int] = {}          # request id (its first token id) -> tokens to generate
        self.prefills: list[int] = []           # tokens per prefill wave, for the record
        self.decode_steps = 0

    # requests announce their answer length through their token ids: ids[0] is a key into `plan`
    def _rid(self, token_ids: list[int]) -> str:
        return str(token_ids[0]) if token_ids else ""

    def _row(self, token_ids: list[int]) -> tuple[_KV, int]:
        """The row after its prefill: its first token is out, `n - 1` more come from decode steps, then EOS."""
        n = self.plan.get(self._rid(token_ids), 1)
        return _KV(len(token_ids), n, self._rid(token_ids)), 1

    def prefill(self, token_ids, session_id="default"):
        self.clock.advance(self.cost.prefill_s(len(token_ids)))
        self.prefills.append(len(token_ids))
        kv, first = self._row(token_ids)
        return kv, first, kv.length

    def prefill_batch(self, prompts, session_ids=None):
        self.clock.advance(self.cost.prefill_s(sum(len(p) for p in prompts)))
        self.prefills.append(sum(len(p) for p in prompts))
        self.last_batch_cache_hit_tokens = 0
        out = []
        for p in prompts:
            kv, first = self._row(p)
            out.append((kv, first, kv.length))
        return out

    def decode_step_batched(self, current_tokens, batched_kv, attention_mask, position_ids, sampling_per_row=None):
        import torch
        self.clock.advance(self.cost.decode_s(len(batched_kv), sum(kv.length for kv in batched_kv)))
        self.decode_steps += 1
        toks = []
        for kv in batched_kv:
            kv.length += 1
            kv.remaining -= 1
            toks.append(EOS if kv.remaining <= 0 else 1)
        return torch.tensor(toks), batched_kv

    def splice_into_batched(self, batched_kv, new_kv, new_kv_len):
        return [new_kv] if batched_kv is None else batched_kv + [new_kv]

    def remove_row_from_cache(self, batched_kv, row_idx):
        return batched_kv[:row_idx] + batched_kv[row_idx + 1:]

    def is_eos(self, token_id) -> bool:
        return int(token_id) == EOS

    def _blocks(self, prompt_len: int, max_tokens: int) -> int:
        return math.ceil((prompt_len + max_tokens) / self.block_size)

    def kv_capacity_error(self, prompt_len: int, max_tokens: int) -> str | None:
        need = self._blocks(prompt_len, max_tokens)
        return None if need <= self.blocks else f"needs {need} blocks, the cache has {self.blocks}"

    def kv_reserve(self, prompt_len: int, max_tokens: int) -> bool:
        need = self._blocks(prompt_len, max_tokens)
        if self.reserved + need > self.blocks:
            return False
        self.reserved += need
        return True

    def kv_release(self, prompt_len: int, max_tokens: int) -> None:
        self.reserved = max(0, self.reserved - self._blocks(prompt_len, max_tokens))

    # the abstract methods the scheduler never calls
    def load_model(self, model_name): pass
    def generate(self, *a, **k): return []
    def generate_batch(self, *a, **k): return []
    def generate_step(self, *a, **k): return 0, None
    def stream(self, *a, **k):
        if False:
            yield 0


# -- the driver -----------------------------------------------------------------------------
class _Immediate:
    """The scheduler's event loop: results delivered on the spot."""

    def call_soon_threadsafe(self, fn, *args):
        fn(*args)


class _Future:
    def __init__(self):
        self.result, self.error, self._done = None, None, False

    def done(self):
        return self._done

    def set_result(self, r):
        self.result, self._done = r, True

    def set_exception(self, e):
        self.error, self._done = e, True


class _Queue:
    """Where streamed tokens land, each stamped with the simulated time it was produced."""

    def __init__(self, clock: Clock):
        self.clock, self.times = clock, []

    def put_nowait(self, tok):
        if tok is not None:
            self.times.append(self.clock.now)


@dataclass
class Settings:
    """The scheduler's knobs, as the engine's env names them (config.py), so a profile's env applies as is."""
    max_batch_size: int = 128
    max_queue_size: int = 1000
    max_active_kv_tokens: int = 0
    prefill_chunk_size: int = 0
    max_prefill_tokens: int = 32768
    prefill_step_tokens: int = 2048
    prefill_step_rows: int = 0
    wave_window_mult: int = 0
    max_queue_wait_s: float = 30.0
    scheduling_policy: str = "fair"
    kv_blocks: int = 4096
    kv_block_size: int = 16

    @classmethod
    def from_env(cls, env: dict) -> "Settings":
        names = {"MAX_BATCH_SIZE": "max_batch_size", "MAX_QUEUE_SIZE": "max_queue_size",
                 "MAX_ACTIVE_KV_TOKENS": "max_active_kv_tokens", "PREFILL_CHUNK_SIZE": "prefill_chunk_size",
                 "MAX_PREFILL_TOKENS": "max_prefill_tokens", "PREFILL_STEP_TOKENS": "prefill_step_tokens",
                 "PREFILL_STEP_ROWS": "prefill_step_rows", "WAVE_WINDOW_MULT": "wave_window_mult",
                 "MAX_QUEUE_WAIT_S": "max_queue_wait_s", "SCHEDULING_POLICY": "scheduling_policy",
                 "CUSTOM_BACKEND_BLOCKS": "kv_blocks", "CUSTOM_BACKEND_BLOCK_SIZE": "kv_block_size"}
        kw = {}
        for k, v in env.items():
            if k in names:
                typ = type(getattr(cls, names[k]))
                kw[names[k]] = typ(v)
        return cls(**kw)


def build(settings: Settings, backend) -> object:
    """The engine's scheduler under `settings`, not started: the driver calls `step()`."""
    from inference_server.scheduler import ContinuousBatchScheduler
    from inference_server.scheduling_policy import create_scheduling_policy
    s = ContinuousBatchScheduler(
        backend, max_batch_size=settings.max_batch_size, max_queue_size=settings.max_queue_size,
        max_active_kv_tokens=settings.max_active_kv_tokens, prefill_chunk_size=settings.prefill_chunk_size,
        max_prefill_tokens=settings.max_prefill_tokens, prefill_step_tokens=settings.prefill_step_tokens,
        prefill_step_rows=settings.prefill_step_rows, wave_window_mult=settings.wave_window_mult,
        max_queue_wait_s=settings.max_queue_wait_s, policy=create_scheduling_policy(settings.scheduling_policy))
    s._loop = _Immediate()
    return s


@dataclass
class Result:
    rows: list[Row]
    elapsed_s: float
    steps: int
    decode_steps: int
    prefill_waves: int
    settings: Settings
    cost: CostModel

    def to_dict(self) -> dict:
        return {"rows": [r.to_dict() for r in self.rows], "elapsed_s": self.elapsed_s, "steps": self.steps,
                "decode_steps": self.decode_steps, "prefill_waves": self.prefill_waves,
                "settings": asdict(self.settings), "cost": asdict(self.cost)}


def simulate(records: list[dict], settings: Settings | None = None, cost: CostModel | None = None, seed: int = 0,
             horizon_s: float | None = None) -> Result:
    """Run `records` (corpus-shaped: arrival_s, build_prompt_tokens, max_tokens, session_id, turn_index, id;
    `expected_output_tokens` when known) through the engine's scheduler on the fake clock. Returns a regimes-shaped
    row per request, so `regimes.runner.summarize` judges a simulation like a bench."""
    from inference_server.scheduler import QueueFullError, RequestTooLargeError, ScheduledRequest
    settings, cost = settings or Settings(), cost or CostModel()
    rng = random.Random(seed)
    clock = Clock()
    t0 = clock.now
    backend = SimBackend(clock, cost, settings.kv_blocks, settings.kv_block_size)
    sched = build(settings, backend)
    arrivals = sorted(records, key=lambda r: (float(r.get("arrival_s") or 0.0), str(r.get("id"))))
    rows: list[Row] = []
    live: list[tuple[dict, ScheduledRequest, _Future, _Queue, float]] = []
    steps = 0
    i = 0

    def enqueue(rec: dict, at: float) -> None:
        n_prompt = int(rec.get("build_prompt_tokens") or max(1, len(str(rec.get("prompt", "")).split())))
        key = 10 + len(backend.plan)                      # ids[0] names the row to the backend (never EOS)
        backend.plan[str(key)] = cost.out_tokens(rec, rng)
        fut, q = _Future(), _Queue(clock)
        req = ScheduledRequest(token_ids=[key] + [1] * (n_prompt - 1), max_tokens=int(rec.get("max_tokens") or 16),
                               session_id=str(rec.get("session_id") or rec.get("id")), future=fut, token_queue=q,
                               trace_id=str(rec.get("id")), turn_index=int(rec.get("turn_index") or 0))
        with clock.at(at):
            req.arrival_ts = clock.now
            try:
                sched.enqueue(req)
            except (QueueFullError, RequestTooLargeError) as e:      # an explicit rejection, as the server's 429/400
                rows.append(Row(id=str(rec.get("id")), status="error", scheduled_s=at - t0, lag_s=0.0,
                               prompt_tokens=n_prompt, http_status=429 if isinstance(e, QueueFullError) else 400,
                               detail=type(e).__name__))
                return
        live.append((rec, req, fut, q, at))

    def settle() -> None:
        for item in list(live):
            rec, req, fut, q, at = item
            if not fut.done():
                continue
            live.remove(item)
            n = len(q.times)
            if fut.error is not None or n == 0:
                rows.append(Row(id=str(rec.get("id")), status="error", scheduled_s=at - t0, lag_s=0.0,
                               prompt_tokens=len(req.token_ids), detail=type(fut.error).__name__ if fut.error else "no tokens"))
                continue
            ttft, e2e = q.times[0] - at, q.times[-1] - at
            rows.append(Row(id=str(rec.get("id")), status="ok", scheduled_s=at - t0, lag_s=0.0, ttft_s=ttft, e2e_s=e2e,
                           n_tokens=n, prompt_tokens=len(req.token_ids), tpot_s=(e2e - ttft) / (n - 1) if n > 1 else None,
                           finish_reason="stop" if n < req.max_tokens else "length"))

    with installed(clock):
        while True:
            while i < len(arrivals) and t0 + float(arrivals[i].get("arrival_s") or 0.0) <= clock.now:
                enqueue(arrivals[i], t0 + float(arrivals[i].get("arrival_s") or 0.0))
                i += 1
            busy = bool(sched._active or sched._prefilling or sched._pending_count)
            if not busy:
                if i >= len(arrivals):
                    break
                clock.now = max(clock.now, t0 + float(arrivals[i].get("arrival_s") or 0.0))
                continue
            if horizon_s is not None and clock.now - t0 > horizon_s:
                break
            before = clock.now
            sched.step("cpu")
            steps += 1
            if clock.now == before:
                clock.advance(cost.step_overhead_s)       # an iteration that ran nothing still took host time
            settle()
        settle()
    for rec, req, fut, q, at in live:                          # cut off by the horizon
        rows.append(Row(id=str(rec.get("id")), status="error", scheduled_s=at - t0, lag_s=0.0,
                       prompt_tokens=len(req.token_ids), detail="unfinished at horizon"))
    return Result(rows, clock.now - t0, steps, backend.decode_steps, len(backend.prefills), settings, cost)
