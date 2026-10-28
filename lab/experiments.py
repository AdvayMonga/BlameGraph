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

MEASURED = ("bench", "equiv", "submit", "profile", "trace", "kernel", "hostprof", "kernel_search")
PROFILERS = ("profile", "trace", "kernel", "hostprof")      # results with no `metrics`: predictions name a path
PROFILE_WORKLOAD = ("requests", "max_tokens", "prompt_len", "synthetic", "corpus_class", "seed", "capture", "profile")
UNITS = ("absolute", "change_pct")
LAYERS = ("kernels", "model_execution", "runtime", "serving")      # lab/layers.py's four layers
# Runs made outside the lab by a researcher (`python -m lab.experiment`): their text is a person's, so no agent view
# (the `ledger` tool, the agent's SQL index) shows any of their records.
RESEARCHER = "researcher-"


def researcher(run: str | None) -> bool:
    return isinstance(run, str) and run.startswith(RESEARCHER)

EXPECT_SCHEMA = {
    "type": "object", "required": ["metric", "value", "unit", "why"],
    "properties": {
        "metric": {"type": "string", "description": "a key of this tool's result metrics (a regime for bench and "
                   "submit); for profile, trace, kernel and hostprof, which have none, a dotted path to a number in "
                   "the result, a row of a list named by its name, e.g. calibrated.kernels.<kernel name>.total_s"},
        "value": {"type": "number"},
        "unit": {"type": "string", "enum": list(UNITS),
                 "description": "absolute: the metric's own value; change_pct: % change from the run's baseline"},
        "why": {"type": "string"},
        "mechanism": {"type": "array", "description": "optional: how the change should work, as predictions on other "
                      "numbers in this tool's result",
                      "items": {"type": "object", "required": ["metric", "value", "unit", "why"],
                                "properties": {
                                    "metric": {"type": "string", "description": "a dotted path to a number in this "
                                               "tool's result (a row of a list named by its name), e.g. "
                                               "calibrated.layers.runtime.gpu_idle_share"},
                                    "value": {"type": "number"},
                                    "unit": {"type": "string", "enum": list(UNITS),
                                             "description": "absolute: the number itself; change_pct: % change from "
                                                            "the same path in the run's baseline record"},
                                    "why": {"type": "string"},
                                    "hypothesis": {"type": "string", "description": "in a bundle: the hypothesis "
                                                   "this prediction is about"}}}}}}
MEASURED_PROPS = {"hypothesis_id": {"type": "string"}, "expect": EXPECT_SCHEMA,
                  "bundle": {"type": "array", "items": {"type": "string"},
                             "description": "optional: further hypothesis ids measured together in this call"},
                  "independence": {"type": "string", "description": "with `bundle`: why the changes do not affect "
                                   "each other"}}


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


def unreflected(root: Path, run: str) -> list[dict]:
    """Every measured record of the run that has no reflection yet, in write order."""
    done = reflected(root, run)
    return [r for r in measured(root, run) if r["id"] not in done]


def involved(rec: dict) -> set[str]:
    """The hypotheses a measured record measured: its own and its bundle's."""
    x = rec.get("experiment") or {}
    return {x.get("hypothesis"), *(x.get("bundle") or [])} - {None}


def open_batch(root: Path, run: str) -> None:
    """A new hypothesis starts the next batch: refused while a measured record of the run has no reflection."""
    waiting = unreflected(root, run)
    if waiting:
        raise ValueError(f"records {', '.join(r['id'] for r in waiting)} have no reflection; every measurement of "
                         "this batch is reflected on before a new hypothesis")


def check(root: Path, run: str, tool: str, args: dict) -> None:
    """Refuse a measured call that lacks its hypothesis or prediction, or that would leave a result unreflected."""
    if tool not in MEASURED:
        return
    hs = hypotheses(root, run)
    hid = args.get("hypothesis_id")
    if not isinstance(hid, str) or hid not in hs:
        raise ValueError("`hypothesis_id` must be the id of a `hypothesis` record of this run")
    bundle = args.get("bundle")
    if bundle is not None:
        if (not isinstance(bundle, list) or not bundle or len(set(bundle)) != len(bundle) or hid in bundle
                or not all(isinstance(h, str) and h in hs for h in bundle)):
            raise ValueError("`bundle` must list other `hypothesis` records of this run, each once")
        _text(args, "independence")
    ids = {hid, *(bundle or [])}
    for r in unreflected(root, run):
        if ids & involved(r):
            raise ValueError(f"record {r['id']} ({r.get('tool')}) measured {', '.join(sorted(ids & involved(r)))} and "
                             f"has no reflection; `reflect` with record={r['id']} comes before the next measurement "
                             "of that hypothesis")
    if tool == "kernel_search" and any(hs[h].get("layer") not in (None, "kernels") for h in ids):
        raise ValueError("`kernel_search` measures hypotheses of the kernels layer only")
    e = args.get("expect")
    _prediction(e, "expect", ("mechanism",))
    m = e.get("mechanism")
    if m is not None:
        if tool == "submit":
            raise ValueError("`expect.mechanism` is not taken by submit: its held-out record keeps one aggregate per "
                             "regime and nothing else")
        if not isinstance(m, list) or not m:
            raise ValueError("`expect.mechanism` must be a non-empty list of predictions")
        for i, p in enumerate(m):
            _prediction(p, f"expect.mechanism[{i}]", ("hypothesis",))
            if p.get("hypothesis") is not None and p["hypothesis"] not in ids:
                raise ValueError(f"`expect.mechanism[{i}].hypothesis` must be one of this call's hypotheses")


PREDICTION_KEYS = ("metric", "value", "unit", "why")


def _prediction(e, where: str, extra: tuple[str, ...] = ()) -> None:
    """One prediction: a metric, a finite number, its unit and why; no other keys (the lab writes the measured ones)."""
    if not isinstance(e, dict):
        raise ValueError(f"`{where}` must be an object: metric, value, unit, why")
    unknown = sorted(e.keys() - set(PREDICTION_KEYS) - set(extra))
    if unknown:
        raise ValueError(f"`{where}` takes only {', '.join(PREDICTION_KEYS + extra)}; not {', '.join(unknown)}")
    for k in ("metric", "why"):
        if not isinstance(e.get(k), str) or not e[k].strip():
            raise ValueError(f"`{where}.{k}` must be non-empty text")
    if isinstance(e.get("value"), bool) or not isinstance(e.get("value"), (int, float)) or not math.isfinite(e["value"]):
        raise ValueError(f"`{where}.value` must be a finite number")
    if e.get("unit") not in UNITS:
        raise ValueError(f"`{where}.unit` must be one of {list(UNITS)}")


def new_hypothesis(args: dict) -> dict:
    """The fields of a `hypothesis` record from the tool's arguments."""
    if args.get("layer") not in LAYERS:
        raise ValueError(f"`layer` must be one of {list(LAYERS)}")
    out = {"hypothesis": _text(args, "hypothesis"), "rationale": _text(args, "rationale"), "layer": args["layer"]}
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
def _actual(tool: str, expect: dict, result: dict | None, metrics: dict | None, baseline: dict | None,
            base_result: dict | None, base: str | None) -> tuple:
    """(actual, missing): the measured counterpart of `expect` in this tool's result, or why there is none."""
    name, unit = expect["metric"], expect["unit"]
    if tool in PROFILERS:
        m = _mechanism(expect, result, base_result, base)
        return m["actual"], m.get("actual_missing")
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


def _number(obj, path: str):
    """The number at a dotted path; in a list of rows a step is a row's `name` (or `function`), else its position."""
    for k in path.split("."):
        if isinstance(obj, list):
            obj = (obj[int(k)] if k.isdigit() and int(k) < len(obj) else
                   next((r for r in obj if isinstance(r, dict) and k in (r.get("name"), r.get("function"))), None))
        elif isinstance(obj, dict) and k in obj:
            obj = obj[k]
        else:
            return None
    return obj if isinstance(obj, (int, float)) and not isinstance(obj, bool) and math.isfinite(obj) else None


def _base(tool: str, args: dict) -> str | None:
    """Where a change_pct prediction finds its baseline in the run's baseline record: the same path for a measured
    tool; under `profiled` (torch at profile's defaults, general launch) for a profile measured the same way; none
    for another instrument or workload."""
    if tool not in PROFILERS:
        return ""
    return "profiled." if tool == "profile" and not any(k in args for k in PROFILE_WORKLOAD) else None


def _mechanism(p: dict, result: dict | None, base_result: dict | None, base: str | None) -> dict:
    """One mechanism prediction with the number at its path in this result (or why there is none) and the gap."""
    v = _number(result, p["metric"])
    if v is not None and p["unit"] == "change_pct":
        if base is None:
            return {**p, "actual": None, "actual_missing": "the run's baseline was not measured with this instrument "
                    "and workload (its after-profile is torch at profile's defaults)"}
        b = _number(base_result, f"{base}{p['metric']}")
        if not b:
            return {**p, "actual": None, "actual_missing": f"no baseline value at `{p['metric']}` in this run"}
        v = (v - b) / b * 100
    if v is None:
        return {**p, "actual": None, "actual_missing": f"`{p['metric']}` is not a number in this result"}
    return {**p, "actual": v, "gap": v - p["value"]}


def outcome(tool: str, args: dict, result: dict | None = None, metrics: dict | None = None,
            baseline: dict | None = None, band_pct: float | None = None, base_result: dict | None = None) -> dict:
    """The experiment block a measured record carries: hypothesis id, the predictions, and the computed gaps."""
    e = {k: v for k, v in args["expect"].items() if k != "mechanism"}
    base = _base(tool, args)
    actual, missing = _actual(tool, e, result, metrics, baseline, base_result, base)
    out = {"hypothesis": args["hypothesis_id"], "expect": e, "actual": actual}
    if args.get("bundle"):
        out["bundle"], out["independence"] = list(args["bundle"]), args["independence"].strip()
    if missing:
        out["actual_missing"] = missing
    else:
        out["gap"] = actual - e["value"]
        out["gap_pct_of_expected"] = (actual - e["value"]) / abs(e["value"]) * 100 if e["value"] else None
    if band_pct is not None:
        out["noise_band_pct"] = band_pct
    if args["expect"].get("mechanism"):
        out["mechanism"] = [_mechanism(p, result, base_result, base) for p in args["expect"]["mechanism"]]
    return out


def summary(root: Path, run: str) -> list[dict]:
    """This run's hypotheses, each with its measurements (expected, actual, gap) and reflections, for the brief; a
    bundled measurement is listed under each hypothesis it measured."""
    hs = hypotheses(root, run)
    refl = {r["record"]: r for r in ledger.records(root, run=run, kind="reflection")}
    out = {h: {"id": h, "hypothesis": r["hypothesis"], "rationale": r["rationale"],
               **{k: r[k] for k in ("layer", "abandon_if") if r.get(k)}, "measurements": []}
           for h, r in hs.items()}
    for m in measured(root, run):
        x = m["experiment"]
        for h in sorted(involved(m) & out.keys(), key=lambda h: h != x["hypothesis"]):
            out[h]["measurements"].append(
                {"record": m["id"], "tool": m.get("tool"), "expect": x["expect"], "actual": x.get("actual"),
                 "gap": x.get("gap"), **({"mechanism": x["mechanism"]} if x.get("mechanism") else {}),
                 **({k: x[k] for k in ("hypothesis", "bundle", "independence")} if x.get("bundle") else {}),
                 "reflection": (refl.get(m["id"]) or {}).get("reflection"),
                 "alternative": (refl.get(m["id"]) or {}).get("alternative")})
    return list(out.values())
