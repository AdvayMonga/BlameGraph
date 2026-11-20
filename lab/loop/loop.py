"""The run on the VM: calibrate once, then turns. A turn: the model picks one gap of the best measurement, a way makes
one candidate on the best tree, it is measured, and a correct candidate better than the best beyond the noise bands
is pushed and becomes the best. Every turn is recorded; the run finishes once, however it ends."""
from __future__ import annotations

import itertools
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from evals.targets import target
from lab.budget import Budget
from lab.evaltools import aggregate, delta_pct
from lab.loop import record as R
from lab.loop.calibrate import calibrate
from lab.loop.gaps import gaps
from lab.loop.measure import measure
from lab.loop.types import Calibration, Candidate, Gap, Measurement, Pushed
from lab.records import ledger
from lab.records.remote import Remote
from lab.task import Task, score


@dataclass(frozen=True)
class Plan:
    run: str
    turns: int | None                   # None: until the budget is spent
    budget_usd: float | None            # None: no cap
    gpu_usd_per_hour: float = 0.0


@dataclass(frozen=True)
class Parts:
    pick: Callable[[tuple[Gap, ...], Measurement, Budget], Gap]
    turn: Callable[[Gap, Candidate, Path, Budget], "Candidate | None"]
    push: Callable[[Candidate, str, str, str], Pushed]     # (candidate, body, branch, onto)
    stop: Callable[[], None]                               # this VM stops itself


def plan(run: str, n: str | None, budget_usd: float | None, gpu_usd_per_hour: float = 0.0) -> Plan:
    """`--n` and `--budget` as a Plan: no `--n` is one turn, `--n N` is N, `--n loop` runs until the budget is spent
    and needs one."""
    if n == "loop":
        if budget_usd is None:
            raise ValueError("--n loop needs --budget USD")
        return Plan(run, None, budget_usd, gpu_usd_per_hour)
    turns = 1 if n is None else int(n)
    if turns < 1:
        raise ValueError("--n is a number of turns, at least 1, or `loop`")
    return Plan(run, turns, budget_usd, gpu_usd_per_hour)


def compare(best: Measurement, m: Measurement, cal: Calibration, task: Task) -> dict:
    """The task's score of `m` against `best`: each regime's change judged against its noise band, and `win`."""
    better = {n: m.metrics[n]["better"] for n in cal.regimes}
    return score({n: aggregate(best.metrics[n], delta_pct(best.metrics[n], m.metrics[n]), better[n], cal.bands[n])
                  for n in cal.regimes}, task, better)


def body(m: Measurement, s: dict) -> str:
    """A win's pull request text: each regime's value and gain, and the gate's verdict."""
    rows = [f"- {n}: {m.metrics[n]['value']:g} ({g:+.1f}%)" for n, g in s["gains_pct"].items() if g is not None]
    return "\n".join(["Measured on the seen split against the best before it.", "", *rows, "",
                      f"Correctness gate: {m.equiv['verdict']}."])


def run(p: Plan, task: Task, t: target.Target, wk, parts: Parts, remote: Remote, work: Path, onto: str,
        root: Path = ledger.ROOT) -> list[dict]:
    """The run's records: the baseline, then one a turn. `onto` is the engine branch the first win's pull request
    targets; each later win targets the win before it."""
    budget = Budget(float("inf") if p.budget_usd is None else p.budget_usd)
    records, state, mark = [], {"state": "running", "stage": "calibrate", "turn": 0, "wins": 0}, time.monotonic()

    def spent() -> float:
        """Charge the GPU time since the last call; returns the budget's total."""
        nonlocal mark
        now = time.monotonic()
        budget.spent_usd += Budget.gpu_usd(now - mark, p.gpu_usd_per_hour)
        mark = now
        return budget.spent_usd

    try:
        R.record(remote, p.run, state, root)
        regimes = tuple(dict.fromkeys((*task.regimes, *task.constraints)))
        cal = calibrate(t, regimes, wk, work / "calibrate", root)
        records.append(R.baseline(cal, p.run, spent(), root))
        best, best_m = Candidate(work / "calibrate" / "base"), cal.baseline
        for n in itertools.count(1) if p.turns is None else range(1, p.turns + 1):
            if budget.remaining_usd <= 0:
                state["stopped"] = "budget spent"
                break
            state.update(stage="turn", turn=n, spent_usd=budget.spent_usd)
            R.record(remote, p.run, state, root)
            before, t0, result = budget.spent_usd, time.monotonic(), {}
            try:
                gap = parts.pick(gaps(best_m), best_m, budget)
                result["gap"] = asdict(gap)
                c = parts.turn(gap, best, work / f"turn-{n}", budget)
                if c is not None:
                    m = measure(c, cal, t, wk, work / f"turn-{n}" / "measure", root)
                    result.update(candidate=c.note, patch=c.patch, **R.measurement(m), win=False)
                    if m.correct:
                        result["score"] = compare(best_m, m, cal, task)
                        if result["score"]["win"]:
                            pr = parts.push(c, body(m, result["score"]), f"perf/{p.run}-{n}", onto)
                            result.update(win=True, pr=asdict(pr))
                            best, best_m, onto = c, m, pr.branch
                            state["wins"] += 1
            except Exception as e:               # noqa: BLE001  a failed turn is recorded; the run goes on
                result["error"] = f"{type(e).__name__}: {e}"[-2000:]
            records.append(R.turn(p.run, n, result, spent() - before, time.monotonic() - t0, root))
        state.update(state="done", stage="finish")
    except BaseException as e:                   # noqa: BLE001  the run finishes and the VM stops whatever happened
        state.update(state="failed", reason=f"{type(e).__name__}: {e}"[-2000:])
        raise
    finally:
        state["spent_usd"] = spent()
        R.finish(remote, p.run, state, records, parts.stop, root)
    return records
