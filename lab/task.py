"""The task: what a run is trying to improve, stated up front so the win condition is known, never inferred.

    [task]
    goal = "cut p99 time per token under steady load without losing burst capacity"
    [objective]
    regimes = ["single_stream"]            # the headline(s) that count; several combine by `combine`
    combine = "min"                        # min | mean of the regimes' gains, in %
    [constraints]                          # "don't get worse than X": checked on every submit
    bursty = { max_regression_pct = 5.0 }

A win is a submit whose objective gain clears the noise band on every objective regime (verdict `improved`), with
no constraint regressed. `score()` computes that from a submit record's metrics; it is the task's definition, not a
judgment, so the agent sees it.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Task:
    goal: str
    regimes: tuple[str, ...]
    combine: str = "min"
    constraints: dict = field(default_factory=dict)      # regime -> {"max_regression_pct": float}
    path: Path | None = None

    @classmethod
    def parse(cls, path: str | Path) -> "Task":
        p = Path(path)
        raw = tomllib.loads(p.read_text())
        obj = raw.get("objective") or {}
        regimes = tuple(obj.get("regimes") or ())
        if not regimes:
            raise ValueError(f"{p}: [objective].regimes is empty")
        from regimes import suite
        bad = [r for r in regimes if r not in suite.REGIMES] + [r for r in (raw.get("constraints") or {}) if r not in suite.REGIMES]
        if bad:
            raise ValueError(f"{p}: unknown regime(s) {bad}; one of {list(suite.REGIMES)}")
        return cls(raw["task"]["goal"], regimes, obj.get("combine", "min"), dict(raw.get("constraints") or {}), p)

    def brief(self) -> str:
        lines = [f"Goal: {self.goal}", f"Objective: {', '.join(self.regimes)} ({self.combine} of their gains, in %); "
                 f"a win needs every objective regime improved beyond its noise band on a submit."]
        for r, c in self.constraints.items():
            lines.append(f"Constraint: {r} may not regress by more than {c.get('max_regression_pct', 0)}%.")
        return "\n".join(lines)


def gain_pct(agg: dict, better: str) -> float | None:
    """A held-out aggregate's gain in the metric's good direction."""
    d = agg.get("delta_pct")
    if d is None:
        return None
    return d if better == "higher" else -d


LOWER_IS_BETTER = ("single_stream", "cold_start")     # fallback when the caller cannot pass the measured direction


def score(metrics: dict, task: Task, better: dict | None = None) -> dict:
    """From a submit record's metrics ({regime: {base, new, delta_pct, band_pct, verdict}}) and each regime's
    measured direction (`better`, as submit reports it): per-regime gains, the combined objective, constraint
    checks, and whether this is a win."""
    from regimes import suite
    better = {**{n: ("lower" if n in LOWER_IS_BETTER else "higher") for n in suite.REGIMES}, **(better or {})}
    gains, verdicts, missing = {}, {}, []
    for r in task.regimes:
        agg = metrics.get(r)
        if agg is None:
            missing.append(r); continue
        gains[r] = gain_pct(agg, better.get(r, "higher"))
        verdicts[r] = agg.get("verdict")
    known = [g for g in gains.values() if g is not None]
    combined = (min(known) if task.combine == "min" else sum(known) / len(known)) if known and len(known) == len(task.regimes) else None
    constraints = {}
    for r, c in task.constraints.items():
        agg = metrics.get(r)
        g = gain_pct(agg, better.get(r, "higher")) if agg else None
        constraints[r] = {"gain_pct": g, "max_regression_pct": c.get("max_regression_pct", 0.0),
                          "ok": None if g is None else g >= -float(c.get("max_regression_pct", 0.0))}
    win = (not missing and all(v == "improved" for v in verdicts.values())
           and all(c["ok"] is True for c in constraints.values()))
    return {"gains_pct": gains, "verdicts": verdicts, "combined_gain_pct": combined, "combine": task.combine,
            "constraints": constraints, "missing_regimes": missing, "win": win}
