"""Does the harness mean anything on *this* target? Run once per model + hardware, before trusting a verdict.

  python -m lab.validate --target T [--tier dev] [--noise-runs 3] [--regimes a,b] [--out DIR]

1. Serves the target's reference and writes the reference outputs (`correctness reference`) to reference.dir.
2. Serves each launch in the target's [validation] good and bad tables, judges it (`correctness candidate`), and
   checks the gate separates them: every good must PASS, every bad must FAIL (inconclusive counts as neither).
3. Serves the target's *engine* (the base commit, from its repo) and runs the chosen regimes `noise_runs` times on
   the seen split, short tier, to measure each regime's run-to-run band on the thing submit will judge; writes
   knowledge/noise/<regime>.json (tagged with the target and `server: engine`), which `submit` reads.
Exit status 0 only if the gate separated and every band was measured. Nothing here is jailed: these are the
operator's own launches, not the agent's.

    [validation]
    good = { fp8 = "vllm serve Qwen/Qwen3-30B-A3B-FP8 --served-model-name Qwen/Qwen3-30B-A3B --port {port}" }
    bad  = { int4 = "vllm serve Qwen/Qwen3-30B-A3B-GPTQ-Int4 --served-model-name Qwen/Qwen3-30B-A3B --port {port}" }
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from lab import serve, target


def band_pct(values: list[float]) -> float | None:
    """Run-to-run band as a percent of the mean: 2 x sample sd / mean (one value: None)."""
    vals = [v for v in values if v is not None]
    if len(vals) < 2 or not statistics.mean(vals):
        return None
    return 2 * statistics.stdev(vals) / abs(statistics.mean(vals)) * 100


def write_noise(regime: str, values: list[float], t: target.Target, extra: dict | None = None) -> dict:
    from regimes.workload import corpus_version
    target.noise_dir().mkdir(parents=True, exist_ok=True)
    d = {"regime": regime, "band_pct": band_pct(values), "values": values, "runs": len(values), "model": t.model,
         "target": t.name, "server": "engine", "corpus_version": corpus_version(t.corpus_dir) if t.corpus_dir.is_dir() else None,
         "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S"), **(extra or {})}
    (target.noise_dir() / f"{regime}.json").write_text(json.dumps(d, indent=1) + "\n")
    return d


def validate(t: target.Target, tier: str = "dev", noise_runs: int = 3, regimes: list[str] | None = None,
             out: Path | None = None, encoder=None, log_dir: Path | None = None) -> dict:
    from correctness import client, run as crun, tasks
    from correctness.gate import Thresholds
    from regimes import suite
    out = out or (target.ENV_ROOT / "data" / "validate" / t.name)
    out.mkdir(parents=True, exist_ok=True)
    log_dir = log_dir or out
    enc = encoder or crun.HFEncoder(t.model, t.chat_kwargs)
    th = Thresholds(min_score_ratio=t.min_score_ratio, min_length_ratio=t.min_length_ratio)
    report: dict = {"target": t.name, "model": t.model, "tier": tier, "candidates": {}, "noise": {}, "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if t.reference is None or t.reference_dir is None:
        raise SystemExit("the target has no [reference] launch and dir")
    ref_dir = t.reference_dir_for(tier)
    # 1. reference (meta.json is written last: its absence means an interrupted run, redone from the cached answers)
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
    # 3. noise bands from repeated unchanged runs of the engine itself (what submit compares against)
    names = regimes or list(suite.REGIMES)
    if noise_runs > 0 and names:
        with serve.Served(t.engine_repo, t.engine, log=log_dir / "serve-noise.log", jailed=False) as srv:
            values: dict[str, list] = {n: [] for n in names}
            for k in range(noise_runs):
                ctx = suite.Ctx.from_target(t, srv.url, split="seen", tier="short", seed=k)
                for n in names:
                    r = suite.REGIMES[n](ctx)
                    values[n].append(r["value"] if r["valid"] else None)
            srv.exclusive()                     # a band from a shared GPU would skew every submit
        for n in names:
            report["noise"][n] = write_noise(n, values[n], t, {"tier": "short", "split": "seen"})
    report["bands_measured"] = all(d["band_pct"] is not None for d in report["noise"].values()) if names and noise_runs else None
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str))
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m lab.validate", description=__doc__.split("\n")[0])
    target.add_argument(ap)
    ap.add_argument("--tier", choices=("dev", "full"), default="dev")
    ap.add_argument("--noise-runs", type=int, default=3)
    ap.add_argument("--regimes", help="comma-separated (default: all)")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    t = target.load(a.target)
    rep = validate(t, a.tier, a.noise_runs, a.regimes.split(",") if a.regimes else None, Path(a.out) if a.out else None)
    for name, c in rep["candidates"].items():
        print(f"{c['kind']:4s} {name}: {c['verdict'].upper()}" + (f"  {c['reasons']}" if c["reasons"] else ""))
    for n, d in rep["noise"].items():
        print(f"noise {n}: band {d['band_pct'] if d['band_pct'] is None else round(d['band_pct'], 2)}% over {d['runs']} runs")
    ok = rep["gate_separates"] and rep["bands_measured"] is not False
    print("gate separates good from bad:", rep["gate_separates"], "| bands measured:", rep["bands_measured"], file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
