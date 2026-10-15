"""Experiment records (decided 2026-10-11): no measured number without the agent's reasoning beside it, and no
reasoning without the number. A measurement names an open `hypothesis` record (the bottleneck and proposed change,
its rationale, optionally what result would end it) and carries an `expect`: the metric, the predicted value and
why. The lab computes the gap between expected and measured. Before the next measurement the agent files a
`reflection` on the last one: why it matched or differed, optionally an alternative hypothesis. All of it is the
agent's text, attributed and paired with measured outcomes; none of it is the referee's."""
from __future__ import annotations

import math
from pathlib import Path

from lab import ledger

MEASURED = ("bench", "equiv", "submit", "profile", "trace", "kernel", "hostprof")
UNITS = ("absolute", "change_pct")
# Runs made outside the lab by a researcher (`python -m lab.experiment`): their text is a person's, so no agent view
# (the `ledger` tool, the agent's SQL index) shows any of their records.
RESEARCHER = "researcher-"


def researcher(run: str | None) -> bool:
    return isinstance(run, str) and run.startswith(RESEARCHER)

EXPECT_SCHEMA = {
    "type": "object", "required": ["metric", "value", "unit", "why"],
    "properties": {
        "metric": {"type": "string", "description": "a key of this tool's result metrics (a regime for bench and submit)"},
        "value": {"type": "number"},
        "unit": {"type": "string", "enum": list(UNITS),
                 "description": "absolute: the metric's own value; change_pct: % change from the run's baseline"},
        "why": {"type": "string"}}}
MEASURED_PROPS = {"hypothesis_id": {"type": "string"}, "expect": EXPECT_SCHEMA}


def _text(args: dict, key: str, required: bool = True) -> str | None:
    v = args.get(key)
    if v is None and not required:
        return None
    if not isinstance(v, str) or not v.strip():
        raise ValueError(f"`{key}` must be non-empty text")
    return v.strip()


def hypotheses(root: Path, run: str) -> dict[str, dict]:
    return {r["id"]: r for r in ledger.records(root, run=run, kind="hypothesis")}


def measured(root: Path, run: str) -> list[dict]:
    """This run's records that carry an experiment block, in write order."""
    return [r for r in ledger.records(root, run=run) if r.get("experiment")]


def reflected(root: Path, run: str) -> set[str]:
    return {r["record"] for r in ledger.records(root, run=run, kind="reflection")}


def pending(root: Path, run: str) -> dict | None:
    """The latest measured record of the run if it has no reflection yet."""
    ms = measured(root, run)
    return ms[-1] if ms and ms[-1]["id"] not in reflected(root, run) else None


def check(root: Path, run: str, tool: str, args: dict) -> None:
    """Refuse a measured call that lacks its hypothesis or prediction, or that would leave a result unreflected."""
    if tool not in MEASURED:
        return
    p = pending(root, run)
    if p is not None:
        raise ValueError(f"record {p['id']} ({p.get('tool')}) has no reflection; `reflect` with record={p['id']} "
                         "comes before the next measured call")
    hid = args.get("hypothesis_id")
    if not isinstance(hid, str) or hid not in hypotheses(root, run):
        raise ValueError("`hypothesis_id` must be the id of a `hypothesis` record of this run")
    e = args.get("expect")
    if not isinstance(e, dict):
        raise ValueError("`expect` must be an object: metric, value, unit, why")
    for k in ("metric", "why"):
        if not isinstance(e.get(k), str) or not e[k].strip():
            raise ValueError(f"`expect.{k}` must be non-empty text")
    if isinstance(e.get("value"), bool) or not isinstance(e.get("value"), (int, float)) or not math.isfinite(e["value"]):
        raise ValueError("`expect.value` must be a finite number")
    if e.get("unit") not in UNITS:
        raise ValueError(f"`expect.unit` must be one of {list(UNITS)}")


def new_hypothesis(args: dict) -> dict:
    """The fields of a `hypothesis` record from the tool's arguments."""
    out = {"hypothesis": _text(args, "hypothesis"), "rationale": _text(args, "rationale")}
    for k in ("abandon_if", "from_record"):
        v = _text(args, k, required=False)
        if v is not None:
            out[k] = v
    return out


def new_reflection(root: Path, run: str, args: dict) -> dict:
    rid = args.get("record")
    ids = {r["id"] for r in measured(root, run)}
    if rid not in ids:
        raise ValueError("`record` must be the id of a measured record of this run")
    if rid in reflected(root, run):
        raise ValueError(f"record {rid} already has a reflection")
    out = {"record": rid, "reflection": _text(args, "reflection")}
    alt = _text(args, "alternative", required=False)
    if alt is not None:
        out["alternative"] = alt
    return out


# -- outcome --------------------------------------------------------------------------------
def _actual(tool: str, expect: dict, result: dict | None, metrics: dict | None, baseline: dict | None) -> tuple:
    """(actual, missing): the measured counterpart of `expect` in this tool's result, or why there is none."""
    name, unit = expect["metric"], expect["unit"]
    if tool == "submit":
        m = (metrics or {}).get(name)
        if not isinstance(m, dict):
            return None, f"`{name}` is not a regime of this submit"
        v = m.get("delta_pct") if unit == "change_pct" else m.get("new")
        return v, None if v is not None else "the held-out aggregate reported no value"
    got = ((result or {}).get("metrics") or {}).get(name)
    value = got.get("value") if isinstance(got, dict) else got
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None, f"`{name}` is not a numeric metric of this {tool} result"
    if unit == "absolute":
        return value, None
    base = ((baseline or {}).get(name) or {}).get("value") if isinstance((baseline or {}).get(name), dict) else None
    if not base:
        return None, f"no baseline value for `{name}` in this run"
    return (value - base) / base * 100, None


def outcome(tool: str, args: dict, result: dict | None = None, metrics: dict | None = None,
            baseline: dict | None = None, band_pct: float | None = None) -> dict:
    """The experiment block a measured record carries: hypothesis id, the prediction, and the computed gap."""
    e = args["expect"]
    actual, missing = _actual(tool, e, result, metrics, baseline)
    out = {"hypothesis": args["hypothesis_id"], "expect": e, "actual": actual}
    if missing:
        out["actual_missing"] = missing
    else:
        out["gap"] = actual - e["value"]
        out["gap_pct_of_expected"] = (actual - e["value"]) / abs(e["value"]) * 100 if e["value"] else None
    if band_pct is not None:
        out["noise_band_pct"] = band_pct
    return out


def summary(root: Path, run: str) -> list[dict]:
    """This run's hypotheses, each with its measurements (expected, actual, gap) and reflections, for the brief."""
    hs = hypotheses(root, run)
    refl = {r["record"]: r for r in ledger.records(root, run=run, kind="reflection")}
    out = {h: {"id": h, "hypothesis": r["hypothesis"], "rationale": r["rationale"],
               **({"abandon_if": r["abandon_if"]} if r.get("abandon_if") else {}), "measurements": []}
           for h, r in hs.items()}
    for m in measured(root, run):
        x = m["experiment"]
        if x["hypothesis"] in out:
            out[x["hypothesis"]]["measurements"].append(
                {"record": m["id"], "tool": m.get("tool"), "expect": x["expect"], "actual": x.get("actual"),
                 "gap": x.get("gap"), "reflection": (refl.get(m["id"]) or {}).get("reflection"),
                 "alternative": (refl.get(m["id"]) or {}).get("alternative")})
    return list(out.values())
