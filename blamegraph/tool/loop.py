"""Glue for an autoresearch loop: ingest finished sessions, accumulate the landscape, compare candidates across seeds.
Statistics only; nothing here tells the loop what to try next."""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path

from .landscape import Landscape, primary
from .ledger import Ledger

SCENARIO_OF_TASK = {"inference_scenario_a_input_heavy": "A", "inference_scenario_b_output_heavy": "B",
                    "inference_scenario_c_high_load": "C", "inference_scenario_d_general": "D"}


@dataclass
class SessionResult:
    session_id: str
    scenario: str
    valid: bool
    reasons: list[str]
    official_metric: float | None     # from the harness's own final eval (metrics.json), None if absent/unscorable
    in_run_metric_of_shipped: float | None
    n_measurements: int
    n_standard_full_fresh: int
    minutes: float | None

    def to_dict(self) -> dict:
        return asdict(self)


def ingest(eval_dir: Path, landscape: Landscape, session_id: str | None = None, scenario: str | None = None) -> SessionResult:
    """Read what the patched harness leaves in EVAL_DIR: blamegraph_ledger.jsonl, blamegraph_validation.json, metrics.json."""
    sid = session_id or eval_dir.name
    led = Ledger.open(eval_dir / "blamegraph_ledger.jsonl")
    task = next((e.get("task", {}) for e in led.events if e["kind"] == "session_start"), {})
    sc = scenario or SCENARIO_OF_TASK.get(str(task.get("scenario")), str(task.get("scenario") or "?"))
    vpath = eval_dir / "blamegraph_validation.json"
    v = json.loads(vpath.read_text()) if vpath.exists() and vpath.stat().st_size else {"valid": False, "reasons": ["no validation record"], "facts": {}}
    mpath = eval_dir / "metrics.json"
    official = primary(json.loads(mpath.read_text()), sc) if mpath.exists() else None
    sub = led.submission() or {}
    shipped = sub.get("config_hash")
    inrun = [primary(m.get("metrics"), sc) for m in led.measurements()
             if m.get("live_hash") == shipped and m.get("standard") and m.get("mode") == "full" and not m.get("stale")]
    inrun = [x for x in inrun if x]
    landscape.add_session(led, sc, sid)
    return SessionResult(session_id=sid, scenario=sc, valid=bool(v.get("valid")), reasons=v.get("reasons", []),
                         official_metric=official if v.get("valid") else None,
                         in_run_metric_of_shipped=(sum(inrun) / len(inrun)) if inrun else None,
                         n_measurements=len(led.measurements()), n_standard_full_fresh=v.get("facts", {}).get("n_standard_full_fresh", 0),
                         minutes=led.events[-1]["minute"] if led.events else None)


def paired_compare(candidate: list[float | None], incumbent: list[float | None], floor: float = 1.0, n_boot: int = 5000, seed: int = 0) -> dict:
    """Seed-paired comparison of two policies' official metrics (same seeds, same order). Invalid/unscored sessions
    enter at `floor` (the benchmark's 1.0x rule). Returns the geomean ratio with a bootstrap interval and the
    probability that the candidate is better — a statement about the data, not a decision."""
    pairs = [(a if a else floor, b if b else floor) for a, b in zip(candidate, incumbent)]
    if not pairs:
        return {"n": 0}
    logs = [math.log(a / b) for a, b in pairs]
    rng = random.Random(seed)
    boots = [sum(rng.choice(logs) for _ in logs) / len(logs) for _ in range(n_boot)]
    boots.sort()
    return {"n": len(pairs), "geomean_ratio": math.exp(sum(logs) / len(logs)),
            "ci95": [math.exp(boots[int(0.025 * n_boot)]), math.exp(boots[int(0.975 * n_boot)])],
            "p_candidate_better": sum(1 for b in boots if b > 0) / n_boot,
            "seeds_where_candidate_better": sum(1 for a, b in pairs if a > b)}
