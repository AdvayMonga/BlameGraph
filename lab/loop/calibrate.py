"""What a run measures once before its turns: the reference answers, each regime's noise band, the GPU's peaks, and
the base commit's measurement, which is the first noise run."""
from __future__ import annotations

import time
from pathlib import Path

from evals.targets import target
from lab.evaltools import BENCH_TIER, headline
from lab.gate.validate import RESOLUTION_PCT, write_noise
from lab.infra.context import peaks
from lab.loop import measure as M
from lab.loop.types import Calibration
from lab.records import ledger
from lab.targets import engine

NOISE_RUNS = 3


def calibrate(t: target.Target, regimes: tuple[str, ...], wk, work: Path, root: Path = ledger.ROOT,
              peaks_dir: Path | None = None) -> Calibration:
    """The engine's HEAD commit benchmarked NOISE_RUNS times (seeds 0..), judged and profiled; a `noise` record a
    regime. Raises when a band or the GPU's peaks are missing."""
    from lab.gate.correctness import hf
    from lab.infra.sandbox import grader
    t0 = time.monotonic()
    sha = engine.commit(t.engine_repo)
    tree = work / "base"
    grader.export(t.engine_repo, sha, tree)
    if not target.reference_complete(t.reference_dir):
        hf.build(t.reference_dir, t.tasks, wk)
    runs = [M.bench(wk, tree, regimes, work / f"noise-{k}", seed=k, tol=RESOLUTION_PCT / 100)
            for k in range(NOISE_RUNS)]
    bands = {}
    for n in regimes:
        heads = [headline(r["results"])[n] for r in runs]
        searched = any(x.get("probes") or x.get("levels") for r in runs for x in r["results"] if x["regime"] == n)
        bands[n] = write_noise(n, [h["value"] if h["valid"] else None for h in heads], t,
                               {"tier": BENCH_TIER, "split": "seen", "engine_commit": sha}, root,
                               RESOLUTION_PCT if searched else 0.0)["band_pct"]
    unmeasured = [n for n, b in bands.items() if b is None]
    if unmeasured:
        raise RuntimeError(f"no noise band for {unmeasured}: fewer than two valid runs of {NOISE_RUNS}")
    gpu = runs[0].get("gpu") or {}
    hw = peaks.load(gpu["name"], gpu.get("driver_version"), peaks_dir) if gpu.get("name") else None
    if hw is None:
        raise RuntimeError(f"no peaks for GPU {gpu.get('name')!r}: `lab peaks` measures them")
    eq = M.judged(wk, tree, t.reference_dir, work / "equiv", root)
    prof = M.profile(wk, tree, work / "profile", t, root)
    base = M.measured(eq, runs[0], work / "noise-0", prof, t, root, time.monotonic() - t0)
    return Calibration(tuple(regimes), bands, t.reference_dir, hw, sha, base)
