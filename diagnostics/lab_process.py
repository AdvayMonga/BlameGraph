"""Process evals on a lab ledger, researcher-only (never shown to the agent): how the agent's predictions held up
against what was measured, outcome and mechanism (lab/experiments.py); bundles whose hypotheses were never measured
apart; reflections checked against the run's recorded numbers. Facts derived from the records, never the verdict.

A direction "held" when a `change_pct` prediction and its measurement have the same sign (the outcome also beyond the
record's noise band); `absolute` predictions have no direction without a baseline and are reported by gap only."""
from __future__ import annotations

from feedback.claim_facts import claim_facts
from lab.experiments import involved

CASES = {(True, True): "worked as intended", (True, False): "outcome held, mechanism did not",
         (False, True): "mechanism held, outcome did not", (False, False): "neither held"}


def _sign(x: float) -> int:
    return (x > 0) - (x < 0)


def record_facts(r: dict) -> dict:
    """One measured record: its hypotheses, whether each prediction's direction held, and which of the four cases."""
    x, e = r["experiment"], r["experiment"]["expect"]
    out = {"record": r["id"], "tool": r.get("tool"), "hypotheses": sorted(involved(r)), "bundle": bool(x.get("bundle"))}
    if x.get("actual") is None:
        out["outcome_missing"] = x.get("actual_missing")
    else:
        out["outcome_gap_pct_of_expected"] = x.get("gap_pct_of_expected")
        if e.get("unit") == "change_pct":
            out["outcome_direction_held"] = (_sign(x["actual"]) == _sign(e["value"])
                                             and abs(x["actual"]) > (x.get("noise_band_pct") or 0))
    mech = x.get("mechanism") or []
    if mech:
        got = [p for p in mech if p.get("actual") is not None]
        out["mechanism_measured"], out["mechanism_predicted"] = len(got), len(mech)
        signed = [p for p in got if p.get("unit") == "change_pct"]
        if signed:
            out["mechanism_direction_held"] = all(_sign(p["actual"]) == _sign(p["value"]) for p in signed)
    if "outcome_direction_held" in out and "mechanism_direction_held" in out:
        out["case"] = CASES[(out["outcome_direction_held"], out["mechanism_direction_held"])]
    return out


def process(recs: list[dict]) -> dict:
    """The run's process facts: per measured record, totals per case, unreflected and unseparated records, and the
    numbers in reflections that match nothing recorded."""
    measured = [r for r in recs if isinstance(r.get("experiment"), dict)]
    reflected = {r.get("record") for r in recs if r.get("kind") == "reflection"}
    per = [record_facts(r) for r in measured]
    alone = {r["experiment"].get("hypothesis") for r in measured if not r["experiment"].get("bundle")}
    unseparated = [{"record": r["id"], "never_measured_alone": sorted(involved(r) - alone)}
                   for r in measured if r["experiment"].get("bundle") and involved(r) - alone]
    cases: dict[str, int] = {}
    for f in per:
        if "case" in f:
            cases[f["case"]] = cases.get(f["case"], 0) + 1

    def held(key):
        v = [f[key] for f in per if key in f]
        return {"held": sum(v), "of": len(v)}
    as_claims = [{**r, "claim": r.get("reflection")} if r.get("kind") == "reflection"
                 else {k: v for k, v in r.items() if k != "claim"} for r in recs]
    return {"measurements": len(per), "unreflected": [r["id"] for r in measured if r["id"] not in reflected],
            "outcome_direction": held("outcome_direction_held"), "mechanism_direction": held("mechanism_direction_held"),
            "cases": cases, "bundles": sum(f["bundle"] for f in per), "unseparated_bundles": unseparated,
            "reflections_vs_records": claim_facts(as_claims), "records": per}
