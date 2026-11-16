"""Validate the harness on one target: reference outputs, gate separation on known-good and -bad launches, noise bands.

  lab validate --target T [--tier dev] [--noise-runs 3] [--regimes a,b] [--out DIR]

The launches come from the target's [validation] `good` and `bad` tables; exit status 0 only if the gate separated
them and every band was measured.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from lab.infra.context import calibrate
from lab.profile import gpu
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


def validate(t: target.Target, tier: str = "dev", noise_runs: int = 3, regimes: list[str] | None = None,
             out: Path | None = None, encoder=None, log_dir: Path | None = None,
             reference_regimes: bool = True, root: Path = ledger.ROOT) -> dict:
    from lab.gate.correctness import client, run as crun, tasks
    from lab.gate.correctness.gate import Thresholds
    from evals.loadgen import suite
    out = out or (target.ENV_ROOT / "local" / "validate" / t.name)
    out.mkdir(parents=True, exist_ok=True)
    log_dir = log_dir or out
    enc = encoder or crun.HFEncoder(t.model, t.chat_kwargs)
    th = Thresholds(min_score_ratio=t.min_score_ratio, min_length_ratio=t.min_length_ratio)
    report: dict = {"target": t.name, "model": t.model, "tier": tier, "candidates": {}, "noise": {}, "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if t.reference is None or t.reference_dir is None:
        raise SystemExit("the target has no [reference] launch and dir")
    ref_dir = t.reference_dir_for(tier)
    # 1. reference
    if not target.reference_complete(ref_dir):
        with serve.Served(target.ENV_ROOT, t.reference, log=log_dir / "serve-reference.log", jailed=False) as srv:
            api = client.Api.from_target(t.reference, t)
            crun.reference(srv.url, t.model, ref_dir, tasks.load(t.tasks, tier=tier), enc, api=api)
    report["reference_dir"] = str(ref_dir)
    # 2. known-good and known-bad candidates
    v = t.raw.get("validation") or {}
    for kind in ("good", "bad"):
        for name, launch in (v.get(kind) or {}).items():
            spec = target.Server(launch, dict((v.get("env") or {}).get(name) or {}), t.reference.health, t.reference.api)
            with serve.Served(target.ENV_ROOT, spec, log=log_dir / f"serve-{name}.log", jailed=False) as srv:
                res = crun.candidate(ref_dir, srv.url, t.model, out / f"{kind}_{name}.json", enc, th,
                                     api=client.Api(spec.api, t.chat_kwargs), config={"tier": tier, "validation": kind})
            report["candidates"][name] = {"kind": kind, "verdict": res["verdict"], "reasons": res["reasons"],
                                          "accuracy": res["metrics"].get("accuracy")}
    goods = [c for c in report["candidates"].values() if c["kind"] == "good"]
    bads = [c for c in report["candidates"].values() if c["kind"] == "bad"]
    report["gate_separates"] = (bool(goods) and bool(bads) and all(c["verdict"] == "pass" for c in goods)
                                and all(c["verdict"] == "fail" for c in bads))
    # 3. noise bands from repeated runs of the engine
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
    # 4. the reference engine under the same regimes, once
    if reference_regimes and names:
        with serve.Served(target.ENV_ROOT, calibrate.perf_reference(t), log=log_dir / "serve-reference-regimes.log",
                          jailed=False) as srv:
            rows: list = []
            ctx = suite.Ctx.from_target(t, srv.url, split="seen", tier=BENCH_TIER, seed=0, rows=rows, tol=RESOLUTION_PCT / 100)
            fp = calibrate.fingerprint(t, gpu.query(), BENCH_TIER, "seen", 0)
            for n in names:
                k = len(rows)
                r = suite.REGIMES[n](ctx)
                calibrate.write_reference(t, n, r, fp, rows[k:])
                report.setdefault("reference_regimes", {})[n] = r["value"] if r["valid"] else None
            srv.exclusive()
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str))
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="lab validate", description=__doc__.split("\n")[0])
    target.add_argument(ap)
    ap.add_argument("--tier", choices=("dev", "full"), default="dev")
    ap.add_argument("--noise-runs", type=int, default=3)
    ap.add_argument("--no-reference-regimes", action="store_true", help="skip measuring the reference under the regimes")
    ap.add_argument("--regimes", help="comma-separated (default: all)")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    t = target.load(a.target)
    rep = validate(t, a.tier, a.noise_runs, a.regimes.split(",") if a.regimes else None, Path(a.out) if a.out else None,
                   reference_regimes=not a.no_reference_regimes)
    for name, c in rep["candidates"].items():
        print(f"{c['kind']:4s} {name}: {c['verdict'].upper()}" + (f"  {c['reasons']}" if c["reasons"] else ""))
    for n, d in rep["noise"].items():
        print(f"noise {n}: band {d['band_pct'] if d['band_pct'] is None else round(d['band_pct'], 2)}% over {d['runs']} runs")
    ok = rep["gate_separates"] and rep["bands_measured"] is not False
    print("gate separates good from bad:", rep["gate_separates"], "| bands measured:", rep["bands_measured"], file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
