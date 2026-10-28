"""Feedback for one finished session of an inference-server lab ledger (see lab_verdict).
Output has two halves:
  for_agent       integrity verdict + the harness facts. Facts only, no advice.
  for_researcher  activity facts and process evals (never fed to the agent).
"""
from __future__ import annotations

import json
from pathlib import Path

from feedback import lab_verdict
from feedback.claim_facts import claim_facts
from diagnostics.lab_process import process


def feedback(path: str | Path, run_id: str | None = None) -> dict:
    p = Path(path)
    lab = lab_verdict.ledger_file(p)
    if not lab:
        raise ValueError(f"{p}: not a lab ledger")
    run_id, recs = lab_verdict.load(lab, run_id)
    return lab_feedback(recs, run_id, str(p))


def lab_feedback(recs: list[dict], run_id: str | None, source: str, harness: dict | None = None) -> dict:
    """The split for a lab ledger (decided 2026-10-06, "option c"): the agent gets the integrity verdict with its
    evidence, the raw ledger (it already has it, through the `ledger` tool) and only facts it cannot derive from
    that ledger: `harness`, the parameters the referee measured against. Everything derivable from the records
    (activity counts, best measured, claims vs evidence) is the researcher's."""
    integrity = lab_verdict.integrity(recs)
    activity = lab_verdict.facts(recs)
    activity["claims"] = claim_facts(recs)   # facts only: claims never decide the verdict (lab/README.md)
    try:                                     # researcher-only: never stops the session start that also calls this
        proc = process(recs)
    except Exception as e:                   # noqa: BLE001
        proc = {"error": f"{type(e).__name__}: {e}"}
    return {"session": source, "run": run_id,
            "for_agent": {"integrity": integrity, "facts": {"harness": harness or {}}},
            "for_researcher": {"activity": activity, "process": proc}}


def render_for_agent(fb: dict) -> str:
    """Plain-text rendering of the agent-facing half. Same facts, nothing added."""
    a = fb["for_agent"]; i = a["integrity"]; f = a["facts"]
    lines = [f"integrity: {'valid' if i['valid'] else 'INVALID'}"]
    ev = {e["rule"]: e for e in i.get("evidence", [])}
    for r in i["reasons"]:
        lines.append(f"  - {r}")
        e = ev.pop(r.split(":")[0], None)   # a rule's evidence goes under its first reason
        if e:
            lines += [f"      {e['detail']}"] + [f"      {k}: {json.dumps(v)}" for k, v in e.items() if k not in ("rule", "detail")]
    lines += [f"{k}: {json.dumps(v, default=str) if isinstance(v, (dict, list)) else v}" for k, v in f.items()]
    return "\n".join(lines)
