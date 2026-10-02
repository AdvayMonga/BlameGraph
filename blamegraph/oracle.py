"""Crowd-sourced oracle: pool every (config -> measured metric) observation across all runs into one landscape."""
from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass

from .experiment import ExperimentLog
from .exploration import is_parameterized, knob_values
from .noise import clean_observations
from .traces import BASELINE_METRIC, Run

KNOBS_FOR_DISTANCE = (  # knobs that define a "config" for the oracle (engine flags only, no paths/ports)
    "max_num_seqs", "max_num_batched_tokens", "gpu_memory_utilization", "block_size", "enable_chunked_prefill",
    "enable_prefix_caching", "enforce_eager", "quantization", "kv_cache_dtype", "attention_backend",
    "num_speculative_tokens", "max_model_len", "dtype", "performance_mode", "async_scheduling", "cuda_graph",
)


@dataclass
class Point:
    run_id: str
    agent: str
    scenario: str
    engine: str
    knobs: tuple          # tuple of (knob, value) sorted
    speedup: float        # metric / baseline
    quick: bool | None


def _key(cfg) -> tuple:
    kv = knob_values(cfg)
    return tuple(sorted((k, kv[k]) for k in KNOBS_FOR_DISTANCE if k in kv))


def collect(runs_logs: list[tuple[Run, ExperimentLog]], full_only: bool = True) -> list[Point]:
    pts = []
    for run, log in runs_logs:
        h = {c.idx: c for c in log.configs}
        for cfg_idx, v in clean_observations(log, run.scenario, full_only=full_only):
            c = h.get(cfg_idx)
            if c is None or c.engine not in ("vllm", "sglang") or is_parameterized(knob_values(c)):
                continue
            pts.append(Point(run.run_id, run.agent, run.scenario, c.engine, _key(c), v / BASELINE_METRIC[run.scenario], None))
    return pts


def distance(a: tuple, b: tuple) -> int:
    da, db = dict(a), dict(b)
    return sum(1 for k in set(da) | set(db) if da.get(k) != db.get(k))


def best_nearby(pts: list[Point], scenario: str, engine: str, knobs: tuple, max_dist: int, exclude_run: str | None = None) -> Point | None:
    cands = [p for p in pts if p.scenario == scenario and p.engine == engine and p.run_id != exclude_run
             and distance(p.knobs, knobs) <= max_dist]
    return max(cands, key=lambda p: p.speedup) if cands else None


def cross_run_noise(pts: list[Point]) -> dict:
    """Identical knob vectors measured in different runs: between-run CV of the landscape."""
    groups: dict[tuple, list[float]] = defaultdict(list)
    runs_in: dict[tuple, set] = defaultdict(set)
    for p in pts:
        k = (p.scenario, p.engine, p.knobs)
        groups[k].append(p.speedup); runs_in[k].add(p.run_id)
    cvs = [statistics.pstdev(v) / statistics.mean(v) for k, v in groups.items() if len(runs_in[k]) >= 2 and statistics.mean(v) > 0]
    cvs.sort()
    return {"n_configs_seen_in_2plus_runs": len(cvs), "cv_median": cvs[len(cvs) // 2] if cvs else None,
            "cv_p75": cvs[int(len(cvs) * .75)] if cvs else None}
