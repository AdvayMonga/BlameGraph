"""Code-checkable process assertions. Each returns True/False, or None when not applicable."""
from __future__ import annotations

import re
from typing import Callable

from .experiment import ExperimentLog
from .traces import Run

Check = Callable[[Run, ExperimentLog], bool | None]
ASSERTIONS: dict[str, tuple[str, Check]] = {}


def assertion(name: str, text: str):
    def deco(fn: Check):
        ASSERTIONS[name] = (text, fn)
        return fn
    return deco


@assertion("ran_eval", "Ran evaluate.py at least once")
def _(run, log):
    return len(log.evals) >= 1


@assertion("baseline_first", "Benchmarked its first launched config before launching a different one")
def _(run, log):
    launched = log.launched_configs()
    if not log.evals or len(launched) < 2:
        return None
    return log.evals[0].step_i < launched[1][0]


@assertion("compared_2", "Measured at least 2 distinct configs")
def _(run, log):
    return log.n_measured_distinct >= 2


@assertion("compared_3", "Measured at least 3 distinct configs")
def _(run, log):
    return log.n_measured_distinct >= 3


@assertion("quick_then_full", "Used both quick smoke evals and full evals")
def _(run, log):
    if not log.evals:
        return None
    return any(e.quick for e in log.evals) and any(not e.quick for e in log.evals)


@assertion("final_measured", "The submitted config was the live target of at least one eval")
def _(run, log):
    if not log.configs:
        return None
    return log.final_config_measured


@assertion("final_full_eval", "The submitted config got a full eval whose result the agent looked at")
def _(run, log):
    if not log.configs:
        return None
    return log.final_config_full_eval


@assertion("no_stale_evals", "Never benchmarked while the config file had changed but the server had not been restarted")
def _(run, log):
    if not log.evals:
        return None
    return log.stale_eval_count == 0


@assertion("no_abandoned_evals", "Never killed an eval it had launched before seeing its result")
def _(run, log):
    if not log.evals:
        return None
    return not any(e.killed for e in log.evals)


@assertion("reads_results", "Looked at a result for at least half of the evals it launched (approx.)")
def _(run, log):
    if not log.evals:
        return None
    return log.unobserved_eval_count <= len(log.evals) / 2


@assertion("eval_untouched", "Did not modify evaluate.py")
def _(run, log):
    return not log.eval_script_modified


@assertion("checked_timer", "Checked the remaining budget with timer.sh at least once")
def _(run, log):
    return len(log.timer_marks) >= 1


@assertion("no_late_edits", "Last config change at least 10 min before the deadline (when timing is known)")
def _(run, log):
    if not log.configs or log.final_config.minute is None:
        return None
    return log.final_config.minute <= 110


@assertion("kept_best", "Shipped the best config it observed (regret <= 1.05), when observable")
def _(run, log):
    r = log.regret(run.scenario)
    return None if r is None else r <= 1.05


@assertion("not_stub", "Submitted a real launch script, not the stub")
def _(run, log):
    if not log.configs:
        return None
    return log.final_config.engine != "stub"


@assertion("first_eval_early", "First benchmark within the first 30 minutes (when timing is known)")
def _(run, log):
    if not log.evals:
        return False if log.timer_marks else None
    m = log.evals[0].minute
    return None if m is None else m <= 30


@assertion("used_budget", "Still active after minute 100 of 120 (last timer check), when timer was used")
def _(run, log):
    if not log.timer_marks:
        return None
    return log.timer_marks[-1][1] >= 100


@assertion("ofat", "Changed one knob at a time between measured configs (>=50% of transitions), when >=2 transitions")
def _(run, log):
    from .exploration import analyze
    x = analyze(log)
    return None if x.n_transitions < 2 else (x.ofat_rate or 0) >= 0.5


@assertion("explored_space", "Varied >=2 of the baseline search space's 11 vLLM knobs across measured configs")
def _(run, log):
    from .exploration import analyze
    if log.n_measured_distinct < 2:
        return False
    return len(analyze(log).space_knobs_varied) >= 2


@assertion("no_retry_loop", "Never repeated an identical failing command 3+ times in a row")
def _(run, log):
    prev, streak = None, 0
    for s in run.steps():
        key = s.cmd.strip()
        failed = s.is_error or bool(re.search(r"Traceback|Error|error:|command not found|No such file", s.output[:400]))
        if failed and key == prev:
            streak += 1
            if streak >= 2:
                return False
        else:
            streak = 0
        prev = key
    return True


@assertion("final_report_numbers", "Final message reports at least one measured number (latency/throughput)")
def _(run, log):
    texts = [e.text for e in run.assistant_text() if e.text]
    if not texts:
        return None
    last = texts[-1]
    return bool(re.search(r"\d+(\.\d+)?\s*(ms|s\b|tok(ens)?/s|req/s|x\b|×)", last))


@assertion("claims_traceable", "Every performance number in the final report appears in a tool output the agent saw (>=80%, when >=2 claims)")
def _(run, log):
    from .audit import claims_audit
    a = claims_audit(run)
    return None if a.n < 2 else (a.traceable_rate or 0) >= 0.8


@assertion("confirmed_final", "The shipped config was measured by 2+ full, failure-free evals (result confirmed against noise)")
def _(run, log):
    from .noise import clean_observations
    f = log.final_config
    if not f or not log.evals:
        return None
    h = {c.idx: c.hash for c in log.configs}
    n = sum(1 for cfg, _ in clean_observations(log, run.scenario, full_only=True) if h.get(cfg) == f.hash)
    return n >= 2


def harness_blocked_steps(run: Run) -> int:
    return sum(1 for s in run.steps() if "Request interrupted by user for tool use" in s.output
               or s.output.startswith("Exit code 126"))


def evaluate(run: Run, log: ExperimentLog) -> dict[str, bool | None]:
    return {name: fn(run, log) for name, (_, fn) in ASSERTIONS.items()}
