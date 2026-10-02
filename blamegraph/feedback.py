"""Feedback for one finished session.

Input: an InferenceBench-format run directory (trace.jsonl, run_meta.json, metrics.json) or a session ledger
(.jsonl in blamegraph.session format). Output has two halves:
  for_agent       integrity verdict + facts about what was measured and shipped. Facts only, no advice.
  for_researcher  assertion results and blame events (process diagnostics; never fed to the agent).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .adapter import log_from_ledger
from .assertions import evaluate
from .audit import claims_audit
from .blame import blame
from .experiment import ExperimentLog, build_log
from .noise import clean_observations
from .session import Ledger
from .traces import BASELINE_METRIC, load_run
from .validate import validate

SCENARIO_OF_TASK = {"inference_scenario_a_input_heavy": "A", "inference_scenario_b_output_heavy": "B",
                    "inference_scenario_c_high_load": "C", "inference_scenario_d_general": "D"}


@dataclass
class _SessionRun:
    """Minimal Run stand-in for ledger sessions (no agent text or tool steps)."""
    run_id: str
    scenario: str
    speedup: float = 1.0

    def steps(self):
        return []

    def assistant_text(self):
        return iter(())


def facts(log: ExperimentLog, scenario: str) -> dict:
    """What was measured and shipped, in the scenario's primary metric (higher is better). No judgments."""
    h = {c.idx: c for c in log.configs}
    obs = clean_observations(log, scenario, full_only=True)
    by_hash: dict[str, list[float]] = {}
    for idx, v in obs:
        if idx in h:
            by_hash.setdefault(h[idx].hash, []).append(v)
    f = log.final_config
    shipped_vals = by_hash.get(f.hash, []) if f else []
    best = max(((k, max(v)) for k, v in by_hash.items()), key=lambda kv: kv[1], default=(None, None))
    base = BASELINE_METRIC.get(scenario)
    to_x = (lambda v: round(v / base, 3)) if base else (lambda v: round(v, 6))
    return {
        "configs_written": len(log.configs),
        "measurements": len(log.evals),
        "measurements_result_read": sum(e.observed for e in log.evals),
        "measurements_on_stale_server": log.stale_eval_count,
        "measurements_nonstandard": sum(not e.standard for e in log.evals),
        "measurements_killed": sum(e.killed for e in log.evals),
        "measurements_with_failures_over_10pct": sum(1 for o in log.observations if (o.failure_rate or 0) > 0.1),
        "configs_with_comparable_measurement": len(by_hash),
        "shipped_config": f"v{f.idx}" if f else None,
        "shipped_comparable_measurements": len(shipped_vals),
        "shipped_best_measured": to_x(max(shipped_vals)) if shipped_vals else None,
        "best_measured_config": next((f"v{c.idx}" for c in log.configs if c.hash == best[0]), None) if best[0] else None,
        "best_measured": to_x(best[1]) if best[1] else None,
        "unit": "x baseline" if base else "primary metric",
        "last_activity_minute": log.timer_marks[-1][1] if log.timer_marks else None,
    }


def _trace_integrity(run, log: ExperimentLog) -> dict:
    reasons = []
    if log.eval_script_modified:
        reasons.append("grader_untouched: evaluate.py was modified during the session")
    if log.final_config is None:
        reasons.append("submission_known: no submitted config could be reconstructed")
    elif not log.final_config_measured:
        reasons.append("submission_measured: the shipped config was never the target of a benchmark run")
    if run.primary_metric[1] == "integrity_failed":
        reasons.append("metrics_physical: the final evaluation reports physically impossible numbers")
    if run.flagged:
        reasons.append("harness_flag: the benchmark's own audit flagged this run")
    ca = claims_audit(run)
    if ca.n >= 2 and (ca.traceable_rate or 0) < 0.8:
        reasons.append(f"claims_traceable: {ca.n - ca.n_traceable} of {ca.n} numbers in the final report appear in no tool output")
    return {"valid": not reasons, "reasons": reasons}


def feedback(path: str | Path, scenario: str | None = None) -> dict:
    p = Path(path)
    if (p / "trace.jsonl").exists():
        run = load_run(p)
        log = build_log(run)
        sc = scenario or run.scenario
        integrity = _trace_integrity(run, log)
    else:
        lp = p / "blamegraph_ledger.jsonl" if p.is_dir() else p
        led = Ledger.open(lp)
        task = next((e.get("task", {}) for e in led.events if e["kind"] == "session_start"), {})
        raw = str(scenario or task.get("scenario") or "?")
        sc = SCENARIO_OF_TASK.get(raw, raw)
        log = log_from_ledger(led, lp.parent.name)
        run = _SessionRun(run_id=lp.parent.name, scenario=sc)
        v = validate(led)
        integrity = {"valid": v.valid, "reasons": v.reasons}
    b = blame(run, log, full_only=True)
    return {
        "session": str(p),
        "scenario": sc,
        "for_agent": {"integrity": integrity, "facts": facts(log, sc)},
        "for_researcher": {
            "assertions": evaluate(run, log),
            "blame": {"found": b.found, "shipped_measured": b.shipped_obs, "kept": b.kept,
                      "events": [{"step": e.step_i, "kind": e.kind, "note": e.note} for e in b.events]},
        },
    }


def render_for_agent(fb: dict) -> str:
    """Plain-text rendering of the agent-facing half. Same facts, nothing added."""
    a = fb["for_agent"]; i = a["integrity"]; f = a["facts"]
    lines = [f"integrity: {'valid' if i['valid'] else 'INVALID'}"] + [f"  - {r}" for r in i["reasons"]]
    lines += [f"{k}: {v}" for k, v in f.items()]
    return "\n".join(lines)
