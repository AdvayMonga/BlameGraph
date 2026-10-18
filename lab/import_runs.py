"""Manual measurements into the ledger as numbers only (decided 2026-10-18).

A run dir written by `python -m regimes run --out DIR` (`regimes.json`) or `python -m correctness candidate --out
DIR/equiv.json` becomes a `bench` or `equiv` record under a `manual-<date>` run: the engine commit and the launch env
as its config, the numbers as its result, the regimes' full results kept, no text. The agent sees it like any
measurement (`ledger`, `query`; `search` has nothing to index). `cost` is unknown for a manual run and stays null.

    python -m lab.import_runs --run manual-20261018 --commit 9c6b79a --env PREFILL_MODE=mixed --env ASYNC_DECODE=1 \\
        --label s9packed data/engine-h200-20261018/s9packed
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lab import ledger
from lab.evaltools import headline

SESSION = "manual"


def _env(pairs: list[str]) -> dict:
    out = {}
    for p in pairs:
        k, _, v = p.partition("=")
        out[k] = v
    return out


def bench_record(run: str, commit: str, env: dict, label: str, regimes_json: dict, gpu: str | None) -> dict:
    results = regimes_json["results"]
    return {"kind": "bench", "run": run, "session": SESSION, "tool": "bench",
            "args": {"regimes": [r["regime"] for r in results], "label": label, "manual": True},
            "snapshot": commit, "snapshot_blob": None, "patch": None,
            "config": {"split": regimes_json.get("split", "seen"), "tier": regimes_json.get("tier", "short"),
                       "seed": regimes_json.get("seed", 0), "commit": commit, "env": env},
            "result": {"verdict": "ok", "metrics": headline(results), "regimes": results,
                       "tier": regimes_json.get("tier", "short"), "gpu": gpu, "source": "manual",
                       "measured_at": regimes_json.get("at"), "corpus_version": regimes_json.get("corpus_version"),
                       "model": regimes_json.get("model")},
            "cost": {"usd": None, "seconds": None}}


def equiv_record(run: str, commit: str, env: dict, label: str, equiv_json: dict) -> dict:
    res = equiv_json["result"]
    passed = {"pass": True, "fail": False}.get(res["verdict"])
    return {"kind": "equiv", "run": run, "session": SESSION, "tool": "equiv",
            "args": {"label": label, "manual": True}, "snapshot": commit, "snapshot_blob": None, "patch": None,
            "config": {"split": "seen", "tier": "dev", "commit": commit, "env": env},
            "result": {"verdict": res["verdict"], "passed": passed, "reasons": res.get("reasons"),
                       "gates": res.get("gates"), "metrics": res.get("metrics"), "thresholds": res.get("thresholds"),
                       "tier": "dev", "source": "manual"},
            "cost": {"usd": None, "seconds": None}}


def import_dir(d: Path, run: str, commit: str, env: dict, label: str | None, gpu: str | None,
               root: Path = ledger.ROOT) -> list[dict]:
    """Every regimes.json / equiv.json under `d` (not recursive) → one record each; returns the stored records."""
    label = label or d.name
    stored = []
    rj = d / "regimes.json"
    if rj.exists():
        stored.append(ledger.append(bench_record(run, commit, env, label, json.loads(rj.read_text()), gpu), root))
    ej = d / "equiv.json"
    if ej.exists():
        stored.append(ledger.append(equiv_record(run, commit, env, label, json.loads(ej.read_text())), root))
    if not stored:
        raise FileNotFoundError(f"{d}: no regimes.json or equiv.json")
    return stored


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--run", required=True, help="run id, e.g. manual-20261018")
    ap.add_argument("--commit", required=True, help="the engine commit (or branch state) that was served")
    ap.add_argument("--env", action="append", default=[], help="launch env K=V (repeatable)")
    ap.add_argument("--label", help="record label; default: the dir name")
    ap.add_argument("--gpu", help='e.g. "NVIDIA H200"')
    a = ap.parse_args(argv)
    for d in a.dirs:
        for rec in import_dir(d, a.run, a.commit, _env(a.env), a.label, a.gpu):
            print(rec["id"], rec["kind"], rec["args"]["label"],
                  json.dumps(rec["result"]["metrics"] if rec["kind"] == "bench"
                             else rec["result"]["verdict"])[:160])


if __name__ == "__main__":
    main()
