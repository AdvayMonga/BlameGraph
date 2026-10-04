"""Flip tests on real traces: inject each known failure and check its assertion flips. Skips without the dataset.
Run: python tests/test_flip.py [N_RUNS]"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from diagnostics.assertions import evaluate  # noqa: E402
from logs.reconstruct import build_log  # noqa: E402
from diagnostics.inject import INJECTORS  # noqa: E402
from logs.inferencebench import DATA_ROOT, iter_runs  # noqa: E402

# minimum share of applicable runs where the target assertion must flip
MIN_HIT = {"ship_worse_config": 0.3, "remove_timer": 0.6, "drop_evals_after_last_config": 0.7}


def test_flips(n_runs: int = 40):
    if not (DATA_ROOT / "runs").exists():
        print("skip: dataset not present"); return
    hits = {k: [0, 0] for k in INJECTORS}
    for run in iter_runs(limit=n_runs):
        base = evaluate(run, build_log(run))
        for name, (_, target, fn) in INJECTORS.items():
            mod = fn(run)
            if mod is None or base[target] is not True:
                continue
            hits[name][1] += 1
            hits[name][0] += evaluate(mod, build_log(mod))[target] is not True
    for name, (f, a) in hits.items():
        rate = f / a if a else None
        print(f"{name:30s} {f}/{a}")
        if a:
            assert rate >= MIN_HIT.get(name, 0.95), f"{name}: target flipped on only {f}/{a} runs"


if __name__ == "__main__":
    test_flips(int(sys.argv[1]) if len(sys.argv) > 1 else 40)
