"""Validate the harness on one target: the noise bands of its engine under the regimes.

  lab validate --target T [--noise-runs 3] [--regimes a,b] [--out DIR]

Exit status 0 only if every band was measured.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from lab.infra.runner import serve
from lab.records import ledger
from evals.targets import target
from lab.targets import engine
from lab.evaltools import BENCH_TIER


RESOLUTION_PCT = 5.0    # percent the noise runs' load searches resolve to; the floor of a searched regime's band


def band_pct(values: list[float], floor: float = 0.0) -> float | None:
    """Run-to-run band as a percent of the mean: 2 x sample sd / mean, at least `floor` (one value: None)."""
    vals = [v for v in values if v is not None]
    if len(vals) < 2 or not statistics.mean(vals):
        return None
    return max(2 * statistics.stdev(vals) / abs(statistics.mean(vals)) * 100, floor)


def write_noise(regime: str, values: list[float], t: target.Target, extra: dict | None = None,
                root: Path = ledger.ROOT, floor: float = 0.0) -> dict:
    """One regime's band as a `noise` ledger record; returns its fields."""
    from evals.loadgen.workload import corpus_version
    d = {"regime": regime, "band_pct": band_pct(values, floor), "values": values, "runs": len(values), "model": t.model,
         "target": t.name, "server": "engine", "corpus_version": corpus_version(t.corpus_dir) if t.corpus_dir.is_dir() else None,
         "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S"), **(extra or {})}
    ledger.append({"kind": "noise", "config": {"split": "seen", "tier": d.get("tier")}, "result": d}, root)
    return d


def validate(t: target.Target, noise_runs: int = 3, regimes: list[str] | None = None,
             out: Path | None = None, log_dir: Path | None = None, root: Path = ledger.ROOT) -> dict:
    from evals.loadgen import suite
    out = out or (target.ENV_ROOT / "local" / "validate" / t.name)
    out.mkdir(parents=True, exist_ok=True)
    log_dir = log_dir or out
    report: dict = {"target": t.name, "model": t.model, "noise": {}, "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    names = regimes or list(suite.REGIMES)
    if noise_runs > 0 and names:
        with serve.Served(t.engine_repo, t.engine, log=log_dir / "serve-noise.log", jailed=False) as srv:
            values: dict[str, list] = {n: [] for n in names}
            searched: set[str] = set()
            for k in range(noise_runs):
                ctx = suite.Ctx.from_target(t, srv.url, split="seen", tier=BENCH_TIER, seed=k, tol=RESOLUTION_PCT / 100)
                for n in names:
                    r = suite.REGIMES[n](ctx)
                    values[n].append(r["value"] if r["valid"] else None)
                    if r.get("probes") or r.get("levels"):
                        searched.add(n)
            srv.exclusive()
        for n in names:
            report["noise"][n] = write_noise(n, values[n], t, {"tier": BENCH_TIER, "split": "seen",
                                                                "engine_commit": engine.commit(t.engine_repo, clean=True)}, root,
                                             RESOLUTION_PCT if n in searched else 0.0)
    report["bands_measured"] = all(d["band_pct"] is not None for d in report["noise"].values()) if names and noise_runs else None
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str))
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="lab validate", description=__doc__.split("\n")[0])
    target.add_argument(ap)
    ap.add_argument("--noise-runs", type=int, default=3)
    ap.add_argument("--regimes", help="comma-separated (default: all)")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    t = target.load(a.target)
    rep = validate(t, a.noise_runs, a.regimes.split(",") if a.regimes else None, Path(a.out) if a.out else None)
    for n, d in rep["noise"].items():
        print(f"noise {n}: band {d['band_pct'] if d['band_pct'] is None else round(d['band_pct'], 2)}% over {d['runs']} runs")
    print("bands measured:", rep["bands_measured"], file=sys.stderr)
    return 0 if rep["bands_measured"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
