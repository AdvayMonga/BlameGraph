"""Flip test: inject a known failure into real runs and check which assertions change answer."""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from blamegraph.assertions import ASSERTIONS, evaluate  # noqa: E402
from blamegraph.experiment import build_log  # noqa: E402
from blamegraph.inject import INJECTORS  # noqa: E402
from blamegraph.traces import iter_runs  # noqa: E402


def main(limit: int | None = None):
    names = list(ASSERTIONS)
    flips = {inj: defaultdict(lambda: [0, 0]) for inj in INJECTORS}   # inj -> assertion -> [flipped, applicable]
    n_app = defaultdict(int)
    for run in iter_runs(limit=limit):
        base = evaluate(run, build_log(run))
        for inj, (_, target, fn) in INJECTORS.items():
            mod = fn(run)
            if mod is None:
                continue
            n_app[inj] += 1
            after = evaluate(mod, build_log(mod))
            for n in names:
                if base[n] is True:
                    flips[inj][n][1] += 1
                    if after[n] is not True:
                        flips[inj][n][0] += 1
    rows = []
    for inj, (text, target, _) in INJECTORS.items():
        row = {"injection": inj, "n": n_app[inj], "target": target}
        for n in names:
            f, a = flips[inj][n]
            row[n] = f"{f}/{a}" if a else ""
        rows.append(row)
    pd.set_option("display.width", 300); pd.set_option("display.max_columns", 40)
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nTarget hit rate (assertion flipped when its failure was injected):")
    for inj, (text, target, _) in INJECTORS.items():
        f, a = flips[inj][target]
        print(f"  {inj:28s} -> {target:20s} {f}/{a}" + ("" if a else "   (never applicable)"))


if __name__ == "__main__":
    main(limit=int(sys.argv[1]) if len(sys.argv) > 1 else None)
