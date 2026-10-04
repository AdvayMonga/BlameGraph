"""Failure injection: transform a real Run into one with a known process failure (for flip tests)."""
from __future__ import annotations

import copy
import re
from typing import Callable

from logs.reconstruct import SERVER_FILE, _launches_eval, build_log, unwrap_shell
from logs.inferencebench import Event, Run

Injector = Callable[[Run], Run | None]   # None when the injection is not applicable to this run
INJECTORS: dict[str, tuple[str, str, Injector]] = {}   # name -> (description, assertion it should flip, fn)


def injector(name: str, text: str, target: str):
    def deco(fn: Injector):
        INJECTORS[name] = (text, target, fn)
        return fn
    return deco


def _clone(run: Run) -> Run:
    return Run(run_id=run.run_id, meta=run.meta, metrics=run.metrics, events=copy.deepcopy(run.events))


def _renumber(run: Run) -> Run:
    for k, e in enumerate(run.events):
        e.i = k
    return run


def _eval_steps(run: Run) -> list[int]:
    return [s.i for s in run.steps() if s.tool in ("bash", "shell") and _launches_eval(unwrap_shell(s.cmd))]


def _drop_steps(run: Run, call_idx: set[int]) -> Run:
    """Remove tool calls (and their paired results) at the given event indices."""
    keep = set(range(len(run.events)))
    for s in run.steps():
        if s.call.i in call_idx:
            keep.discard(s.call.i)
            if s.result is not None:
                keep.discard(s.result.i)
    run.events = [e for e in run.events if e.i in keep]
    return _renumber(run)


def _bash_event(i: int, cmd: str, out: str = "") -> tuple[Event, Event]:
    call = Event(i=i, role="assistant", type="tool_call", tool_name="Bash", tool_input={"command": cmd},
                 raw={"i": i, "role": "assistant", "type": "tool_call", "tool_name": "Bash", "tool_input": {"command": cmd}})
    res = Event(i=i + 1, role="tool", type="tool_result", tool_output=out,
                raw={"i": i + 1, "role": "tool", "type": "tool_result", "tool_output": out, "meta": {"is_error": False}})
    return call, res


def _insert_after(run: Run, after_i: int, events: list[Event]) -> Run:
    pos = next(k for k, e in enumerate(run.events) if e.i == after_i) + 1
    # skip past the paired result if the anchor is a call
    if run.events[pos - 1].type == "tool_call" and pos < len(run.events) and run.events[pos].type == "tool_result":
        pos += 1
    run.events[pos:pos] = events
    return _renumber(run)


@injector("drop_all_evals", "Remove every evaluate.py launch", "ran_eval")
def _(run):
    ev = _eval_steps(run)
    if not ev:
        return None
    return _drop_steps(_clone(run), set(ev))


@injector("drop_evals_after_last_config", "Remove evals that ran after the final config was written", "final_measured")
def _(run):
    log = build_log(run)
    if not log.configs or not log.final_config_measured:
        return None
    last_cfg = log.final_config.step_i
    ev = {i for i in _eval_steps(run) if i > last_cfg}
    if not ev:
        return None
    return _drop_steps(_clone(run), ev)


@injector("tamper_eval_script", "Append an edit of evaluate.py near the start", "eval_untouched")
def _(run):
    if build_log(run).eval_script_modified:
        return None
    r = _clone(run)
    first = next((s.i for s in r.steps()), None)
    if first is None:
        return None
    c, res = _bash_event(0, "sed -i 's/tau = 0.95/tau = 0.5/' /home/user/task/evaluate.py", "")
    return _insert_after(r, first, [c, res])


@injector("late_unmeasured_config", "Write a new start_server.sh flag at the very end, never evaluated", "final_measured")
def _(run):
    log = build_log(run)
    if not log.configs or not log.final_config_measured:
        return None
    r = _clone(run)
    last = r.steps()[-1].i
    new = log.final_config.content + "\n# injected\nexport VLLM_INJECTED=1\n"
    c, res = _bash_event(0, f"cat > /home/user/task/{SERVER_FILE} <<'EOF'\n{new}\nEOF", "")
    return _insert_after(r, last, [c, res])


@injector("kill_first_eval", "Kill the first eval right after launching it", "no_abandoned_evals")
def _(run):
    log = build_log(run)
    if not log.evals or any(e.killed for e in log.evals):
        return None
    r = _clone(run)
    c, res = _bash_event(0, "pkill -f evaluate.py", "")
    return _insert_after(r, log.evals[0].step_i, [c, res])


@injector("blind_results", "Strip every metric number from tool outputs", "reads_results")
def _(run):
    log = build_log(run)
    if not log.evals or log.unobserved_eval_count > len(log.evals) / 2:
        return None
    r = _clone(run)
    rx = re.compile(r"(p50|p90|p99|throughput|tok/s|tokens/s|req/s|req_ps|tps|fail)", re.I)
    for e in r.events:
        if e.type == "tool_result" and isinstance(e.tool_output, str) and rx.search(e.tool_output):
            e.tool_output = "\n".join(l for l in e.tool_output.splitlines() if not rx.search(l))
    return r


@injector("remove_timer", "Drop every timer.sh call", "checked_timer")
def _(run):
    t = {s.i for s in run.steps() if "timer.sh" in s.cmd}
    if not t:
        return None
    return _drop_steps(_clone(run), t)


@injector("ship_worse_config", "Re-write the worst-observed config at the end and evaluate nothing", "kept_best")
def _(run):
    log = build_log(run)
    obs = log.primary_observations(run.scenario)
    if len({c for c, _ in obs}) < 2 or (log.regret(run.scenario) or 9) > 1.05:
        return None
    worst_idx = min(obs, key=lambda cv: cv[1])[0]
    worst = next(c for c in log.configs if c.idx == worst_idx)
    r = _clone(run)
    last = r.steps()[-1].i
    c, res = _bash_event(0, f"cat > /home/user/task/{SERVER_FILE} <<'EOF'\n{worst.content}\nEOF", "")
    return _insert_after(r, last, [c, res])
