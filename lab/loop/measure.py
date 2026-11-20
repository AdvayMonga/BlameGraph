"""One candidate measured: the gate's verdict, then the benchmark numbers and the profile, as worker jobs."""
from __future__ import annotations

import tempfile
import time
from pathlib import Path

from evals.targets import target
from lab.evaltools import BENCH_TIER, equiv_rows, headline, search_hints
from lab.infra.context import calibrate, layers
from lab.infra.runner import artifacts
from lab.loop.types import Calibration, Candidate, Measurement
from lab.records import ledger

SEED = 0
EQUIV_CONCURRENCY = 128
PROFILE_TIMEOUT_S = 900
EQUIV_KEYS = ("verdict", "passed", "reasons", "gates", "metrics", "thresholds", "artifacts")


def bench(wk, tree: Path, regimes, out: Path, seed: int = SEED, **search) -> dict:
    """The worker's `measure` job on the seen split, passive data into `out`; `search`: hints, bands, tol."""
    out.mkdir(parents=True, exist_ok=True)
    return wk.call("measure", {"tree": tree}, {"names": list(regimes), "split": "seen", "tier": BENCH_TIER,
                                               "seed": seed, "jailed": True, "passive": True, "workbench": None,
                                               **search}, out)


def equiv(wk, tree: Path, reference: Path, out: Path) -> dict:
    """The worker's `equiv` job against `reference`: the gate's result."""
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as prior:
        return wk.call("equiv", {"tree": tree, "reference": reference, "prior": Path(prior)},
                       {"concurrency": EQUIV_CONCURRENCY, "jailed": True, "profile": None, "workbench": None}, out)


def profile(wk, tree: Path, out: Path, t: target.Target, root: Path) -> dict:
    """The worker's `profile` job at its defaults: {bundle, layers, kernels}, or {error}."""
    from lab.tools import _drop_links
    out.mkdir(parents=True, exist_ok=True)
    try:
        r = wk.call("profile", {"tree": tree}, {"kind": "profile", "args": {}, "workbench": None,
                                                "timeout_s": PROFILE_TIMEOUT_S}, out)
    except Exception as e:                       # noqa: BLE001  a measurement stands without its profile
        return {"error": f"{type(e).__name__}: {e}"}
    files = out / "files"
    if r.get("refused") or r.get("returncode") or not (files / "meta.json").exists():
        return {"error": r.get("refused") or str(r.get("output") or "the profile wrote no bundle")[-2000:]}
    _drop_links(files)
    blob = ledger.put_blob(files, root)
    return {"bundle": blob, "layers": layers.from_bundle(root / blob, t, r.get("gpu")),
            "kernels": calibrate.calibrate_kernels(root / blob, r.get("gpu"))}


def measured(eq: dict, r: dict, bench_dir: Path, prof: dict, t: target.Target, root: Path,
             seconds: float) -> Measurement:
    """The Measurement of one gate result, one bench reply with its passive data in `bench_dir`, and one profile."""
    cal = calibrate.calibrate_results(r["results"], artifacts.read_rows(bench_dir), t, r.get("gpu"), BENCH_TIER,
                                      "seen", SEED)
    return Measurement(bool(eq["passed"]), {k: eq.get(k) for k in EQUIV_KEYS}, headline(r["results"]), cal, prof,
                       artifacts.store(bench_dir, root), r.get("gpu"), seconds)


def judged(wk, tree: Path, reference: Path, out: Path, root: Path) -> dict:
    """The gate's result on `tree`, its per-item rows kept as a ledger blob under `artifacts`."""
    eq = equiv(wk, tree, reference, out)
    artifacts.write_rows(out, equiv_rows(out / "equiv.json"))
    return {**eq, "artifacts": artifacts.store(out, root)}


def measure(c: Candidate, cal: Calibration, t: target.Target, wk, work: Path,
            root: Path = ledger.ROOT) -> Measurement:
    """`c` judged by the gate; when it passes, benchmarked on the calibration's regimes and profiled."""
    t0 = time.monotonic()
    eq = judged(wk, c.tree, cal.reference, work / "equiv", root)
    if not eq["passed"]:
        return Measurement(False, {k: eq.get(k) for k in EQUIV_KEYS}, seconds=time.monotonic() - t0)
    r = bench(wk, c.tree, cal.regimes, work / "bench", hints=search_hints(cal.baseline.metrics), bands=cal.bands)
    prof = profile(wk, c.tree, work / "profile", t, root)
    return measured(eq, r, work / "bench", prof, t, root, time.monotonic() - t0)
