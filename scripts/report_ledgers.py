"""Researcher-side report over native ledgers: run the code assertions on every EVAL_DIR under a root.
Usage: report_ledgers.py RESULTS_ROOT   (each subdir with blamegraph_ledger.jsonl is one session)"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from blamegraph.assertions import ASSERTIONS, evaluate  # noqa: E402
from blamegraph.tool.adapter import log_from_ledger  # noqa: E402
from blamegraph.tool.ledger import Ledger  # noqa: E402
from blamegraph.tool.loop import SCENARIO_OF_TASK  # noqa: E402


class _Run:  # minimal stand-in: ledger sessions have no agent text or tool steps
    def __init__(self, scenario): self.scenario = scenario
    def steps(self): return []
    def assistant_text(self): return iter(())


def main(root: str):
    rows = []
    for lp in sorted(Path(root).rglob("blamegraph_ledger.jsonl")):
        led = Ledger.open(lp)
        task = next((e.get("task", {}) for e in led.events if e["kind"] == "session_start"), {})
        sc = SCENARIO_OF_TASK.get(str(task.get("scenario")), str(task.get("scenario") or "?"))
        vp = lp.parent / "blamegraph_validation.json"
        v = json.loads(vp.read_text()) if vp.exists() and vp.stat().st_size else {}
        r = dict(session=lp.parent.name, scenario=sc, valid=v.get("valid"), reasons="; ".join(v.get("reasons", [])))
        r.update(evaluate(_Run(sc), log_from_ledger(led, lp.parent.name)))
        rows.append(r)
    if not rows:
        print("no ledgers under", root); return
    d = pd.DataFrame(rows)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print(d[["session", "scenario", "valid", "reasons"]].to_string(index=False))
    print("\nassertion pass rates:")
    for n in ASSERTIONS:
        col = d[n].dropna()
        if len(col):
            print(f"  {n:20s} {col.astype(float).mean():.2f} (n={len(col)})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
