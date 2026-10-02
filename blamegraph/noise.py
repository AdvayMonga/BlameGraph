"""Noise floor: how repeatable is the benchmark, and did the agent act on differences smaller than that?"""
from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass, field

from .experiment import ExperimentLog, Observation
from .traces import BASELINE_METRIC

# 3x the best speedup the InferenceBench search baselines reached per scenario: anything above is not a
# comparable measurement (tiny request sets, warm caches, patched graders, or a cheating server)
PLAUSIBLE_MAX_SPEEDUP = {"A": 15.0, "B": 45.0, "C": 150.0, "D": 17.0}


def _primary(o: Observation, scenario: str) -> float | None:
    if scenario == "A" and o.ttft_p50:
        return 1 / o.ttft_p50
    if scenario == "B" and o.tpot_p50:
        return 1 / o.tpot_p50
    if scenario == "C" and (o.rps_geomean or o.rps):
        return o.rps_geomean or o.rps
    if scenario == "D" and o.rps and o.ttft_p50 and o.tpot_p50:
        return (1 / o.ttft_p50 * 1 / o.tpot_p50 * o.rps) ** (1 / 3)
    return None


def clean_observations(log: ExperimentLog, scenario: str, full_only: bool = True, standard_only: bool = True) -> list[tuple[int, float]]:
    """(config_idx, metric) for comparable observations: attributed to a config, no request failures,
    full eval if asked, standard harness invocation if asked (and never from a run that patched the grader),
    and within the plausible range for this hardware."""
    out = []
    if standard_only and log.eval_script_modified:
        return out
    cap = PLAUSIBLE_MAX_SPEEDUP[scenario] * BASELINE_METRIC[scenario]
    for o in log.observations:
        if o.config_idx is None or (o.failure_rate or 0) > 0 or (full_only and o.quick) or (standard_only and o.standard is False):
            continue
        v = _primary(o, scenario)
        if v and v <= cap:
            out.append((o.config_idx, v))
    return out


@dataclass
class NoiseStats:
    repeats: list[tuple[int, list[float]]] = field(default_factory=list)   # (config_idx, metrics observed for it)
    decisions: list[tuple[int, int, float]] = field(default_factory=list)  # (from_cfg, to_cfg, relative change)

    @property
    def within_config_cv(self) -> float | None:
        cvs = [statistics.pstdev(v) / statistics.mean(v) for _, v in self.repeats if len(v) >= 2 and statistics.mean(v) > 0]
        return statistics.median(cvs) if cvs else None


def analyze(log: ExperimentLog, scenario: str, full_only: bool = True) -> NoiseStats:
    obs = clean_observations(log, scenario, full_only)
    by_cfg: dict[int, list[float]] = defaultdict(list)
    for cfg, v in obs:
        by_cfg[cfg].append(v)
    ns = NoiseStats(repeats=[(c, v) for c, v in by_cfg.items()])
    seq: list[tuple[int, float]] = []
    for cfg, v in obs:
        if not seq or seq[-1][0] != cfg:
            seq.append((cfg, v))
    for (a, va), (b, vb) in zip(seq, seq[1:]):
        if va > 0:
            ns.decisions.append((a, b, (vb - va) / va))
    return ns


def corpus_noise(stats_by_run: dict[str, NoiseStats]) -> dict:
    """Pooled noise estimate: all within-config CVs across the corpus."""
    cvs = []
    for ns in stats_by_run.values():
        for _, v in ns.repeats:
            if len(v) >= 2 and statistics.mean(v) > 0:
                cvs.append(statistics.pstdev(v) / statistics.mean(v))
    if not cvs:
        return {}
    cvs.sort()
    return {"n_repeated_configs": len(cvs), "cv_median": cvs[len(cvs) // 2], "cv_p75": cvs[int(len(cvs) * .75)],
            "cv_p90": cvs[int(len(cvs) * .9)]}
