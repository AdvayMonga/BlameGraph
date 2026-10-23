"""Process evals on a lab ledger, researcher-only (never shown to the agent): how the agent's predictions held up
against what was measured, outcome and mechanism (lab/experiments.py); bundles whose hypotheses were never measured
apart; reflections checked against the run's recorded numbers. Facts derived from the records, never the verdict.

A `change_pct` outcome prediction "held" when it and the measurement have the same sign and the measurement is beyond
the record's noise band, or, for a prediction of 0, within it; with no band (only bench and submit have one) it is not
judged. A mechanism prediction is judged by sign alone (a prediction of 0 by an exact 0). `absolute` predictions have
no direction without a baseline and are reported by gap only."""
from __future__ import annotations

from feedback.claim_facts import claim_facts
from lab.experiments import involved

CASES = {(True, True): "worked as intended", (True, False): "outcome held, mechanism did not",
         (False, True): "mechanism held, outcome did not", (False, False): "neither held"}


def _sign(x: float) -> int:
    return (x > 0) - (x < 0)


def _outcome_held(actual: float, value: float, band: float | None) -> bool | None:
    if band is None:
        return None
    if value == 0:
        return abs(actual) <= band
    return _sign(actual) == _sign(value) and abs(actual) > band


def record_facts(r: dict) -> dict:
    """One measured record: its hypotheses, whether each prediction's direction held, and which of the four cases."""
    x = r["experiment"]
    e = x.get("expect") if isinstance(x.get("expect"), dict) else {}
    out = {"record": r.get("id"), "tool": r.get("tool"), "hypotheses": sorted(involved(r)),
           "bundle": bool(x.get("bundle"))}
    if not isinstance(x.get("actual"), (int, float)) or isinstance(x.get("actual"), bool):
        out["outcome_missing"] = x.get("actual_missing")
    else:
        out["outcome_gap_pct_of_expected"] = x.get("gap_pct_of_expected")
        if e.get("unit") == "change_pct" and isinstance(e.get("value"), (int, float)):
            held = _outcome_held(x["actual"], e["value"], x.get("noise_band_pct"))
            if held is None:
                out["outcome_direction_missing"] = "no noise band for this metric"
            else:
                out["outcome_direction_held"] = held
    mech = [p for p in x.get("mechanism") or [] if isinstance(p, dict)]
    if mech:
        got = [p for p in mech if isinstance(p.get("actual"), (int, float))]
        out["mechanism_measured"], out["mechanism_predicted"] = len(got), len(mech)
        signed = [p for p in got if p.get("unit") == "change_pct" and isinstance(p.get("value"), (int, float))]
        if signed:
            out["mechanism_direction_held"] = all(_sign(p["actual"]) == _sign(p["value"]) for p in signed)
    if "outcome_direction_held" in out and "mechanism_direction_held" in out:
        out["case"] = CASES[(out["outcome_direction_held"], out["mechanism_direction_held"])]
    return out


def process(recs: list[dict]) -> dict:
    """The run's process facts: per measured record, totals per case, unreflected and unseparated records, and the
    numbers in reflections that match nothing recorded."""
    measured = [r for r in recs if isinstance(r.get("experiment"), dict) and r["experiment"]]
    by_id = {r.get("id"): r for r in recs}
    reflected = {r.get("record") for r in recs if r.get("kind") == "reflection"}
    per = [record_facts(r) for r in measured]
    alone = {(r.get("tool"), r["experiment"].get("hypothesis")) for r in measured
             if not r["experiment"].get("bundle") and r["experiment"].get("actual") is not None}
    unseparated = []
    for r in measured:
        if r["experiment"].get("bundle"):
            left = sorted(h for h in involved(r) if (r.get("tool"), h) not in alone)
            if left:
                unseparated.append({"record": r.get("id"), "tool": r.get("tool"), "never_measured_alone": left})
    cases: dict[str, int] = {}
    for f in per:
        if "case" in f:
            cases[f["case"]] = cases.get(f["case"], 0) + 1

    def held(key):
        v = [f[key] for f in per if key in f]
        return {"held": sum(v), "of": len(v)}
    # reflections as claims about the record they reflect on; the experiment blocks' numbers count as recorded
    as_claims = [{**r, "snapshot": (by_id.get(r.get("record")) or {}).get("snapshot"),
                  "claim": "\n".join(t for t in (r.get("reflection"), r.get("alternative")) if isinstance(t, str))}
                 if r.get("kind") == "reflection" else {k: v for k, v in r.items() if k != "claim"} for r in recs]
    as_claims += [{"kind": "note", "result": r["experiment"]} for r in measured]
    return {"measurements": len(per), "unreflected": [r.get("id") for r in measured if r.get("id") not in reflected],
            "outcome_direction": held("outcome_direction_held"), "mechanism_direction": held("mechanism_direction_held"),
            "cases": cases, "bundles": sum(f["bundle"] for f in per), "unseparated_bundles": unseparated,
            "reflections_vs_records": claim_facts(as_claims), "records": per}
