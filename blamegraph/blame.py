"""Model-free blame graph: where in a run did value get lost?

State value V(t) = best metric the agent has measured so far (speedup units). The run's final speedup
decomposes as  final = found x kept x executed:
  found    = best config it ever measured (search quality)
  kept     = shipped config's own in-run measurement / found (selection: did it keep the best?)
  executed = official final speedup / shipped config's in-run measurement (did the submission reproduce?)
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .experiment import ExperimentLog
from .noise import clean_observations
from .traces import BASELINE_METRIC, Run


@dataclass
class LossEvent:
    step_i: int
    kind: str          # abandoned_best | unmeasured_final | killed_eval | late_change | no_relaunch_check
    value_lost: float  # in speedup units (0 when unknown)
    note: str = ""


@dataclass
class Blame:
    run_id: str
    found: float | None          # best in-run measured speedup
    shipped_obs: float | None    # in-run measured speedup of the shipped config
    final: float                 # official speedup (1.0 floor)
    events: list[LossEvent] = field(default_factory=list)
    best_step: int | None = None
    best_cfg: int | None = None

    @property
    def kept(self) -> float | None:
        return self.shipped_obs / self.found if self.found and self.shipped_obs else None

    @property
    def executed(self) -> float | None:
        return self.final / self.shipped_obs if self.shipped_obs else None

    @property
    def phase_of_loss(self) -> str:
        """Which factor cost the most, relative to a perfect run."""
        k, e = self.kept, self.executed
        if self.found is None:
            return "no_measurement"
        losses = {"selection": 1 / k if k else 1.0, "execution": 1 / e if e else 1.0}
        worst = max(losses, key=losses.get)
        return worst if losses[worst] > 1.1 else "none"


def blame(run: Run, log: ExperimentLog, full_only: bool = False) -> Blame:
    base = BASELINE_METRIC[run.scenario]
    obs = clean_observations(log, run.scenario, full_only=full_only)   # (config_idx, metric)
    b = Blame(run_id=run.run_id, found=None, shipped_obs=None, final=run.speedup)
    if not obs:
        b.events.append(LossEvent(0, "no_measurement", 0.0, "no clean observation attributed to a config"))
        return b
    h = {c.idx: c for c in log.configs}
    # best observed, and when it was observed
    best_cfg, best_v = max(obs, key=lambda cv: cv[1])
    b.found = best_v / base
    b.best_cfg = best_cfg
    step_of_obs = {}
    for o in log.observations:
        if o.config_idx is not None:
            step_of_obs.setdefault(o.config_idx, o.step_i)
    b.best_step = step_of_obs.get(best_cfg)
    f = log.final_config
    if f is not None:
        same = [v for c, v in obs if h.get(c) and h[c].hash == f.hash]
        if same:
            b.shipped_obs = max(same) / base
        if h.get(best_cfg) and h[best_cfg].hash != f.hash:
            # the agent left its best config behind: blame the first config write after the best measurement
            after = [c for c in log.configs if b.best_step is not None and c.step_i > b.best_step]
            if after:
                lost = (b.found - (b.shipped_obs or 1.0))
                b.events.append(LossEvent(after[0].step_i, "abandoned_best", max(0.0, lost),
                                          f"best cfg{best_cfg} ({b.found:.2f}x) replaced by cfg{after[0].idx}"))
        if not log.final_config_measured:
            b.events.append(LossEvent(f.step_i, "unmeasured_final", 0.0, "shipped config never benchmarked"))
    for e in log.evals:
        if e.killed:
            b.events.append(LossEvent(e.step_i, "killed_eval", 0.0, "eval killed before result"))
    if f is not None and f.minute is not None and f.minute > 110:
        b.events.append(LossEvent(f.step_i, "late_change", 0.0, f"config changed at minute {f.minute:.0f}"))
    if b.shipped_obs and b.final < 0.8 * b.shipped_obs:
        b.events.append(LossEvent(log.evals[-1].step_i if log.evals else 0, "execution_gap", b.shipped_obs - b.final,
                                  f"measured {b.shipped_obs:.2f}x in-run, scored {b.final:.2f}x"))
    return b
