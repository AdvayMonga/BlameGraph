"""Run all code-checkable assertions over the corpus; report per-agent pass rates, discrimination, reliability."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from blamegraph.assertions import ASSERTIONS, evaluate, harness_blocked_steps  # noqa: E402
from blamegraph.experiment import build_log  # noqa: E402
from blamegraph.extract import obs_by_run  # noqa: E402
from blamegraph.traces import iter_runs  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "data" / "derived"


def main():
    llm = obs_by_run()
    print(f"extractor cache: {sum(len(v) for v in llm.values())} steps over {len(llm)} runs")
    rows = []
    for run in iter_runs():
        log = build_log(run, llm_obs=llm.get(run.run_id))
        r = dict(run_id=run.run_id, agent=run.agent, scenario=run.scenario, harness=run.harness,
                 scored=int(run.scored), speedup=run.speedup, blocked=harness_blocked_steps(run))
        r.update(evaluate(run, log))
        rows.append(r)
    d = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT / "assertions.csv", index=False)
    names = list(ASSERTIONS)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)

    print("== pass rate overall (n applicable)")
    for n in names:
        col = d[n].dropna()
        print(f"  {n:20s} {col.mean():5.2f}  (n={len(col):3d})   {ASSERTIONS[n][0]}")

    agg = d.groupby("agent").agg(speedup=("speedup", lambda s: float(np.exp(np.log(s).mean()))),
                                 scored=("scored", "mean"), blocked=("blocked", "median"))
    for n in names:
        agg[n] = d.groupby("agent")[n].mean()
    agg["process"] = agg[names].mean(axis=1)
    agg = agg.sort_values("speedup", ascending=False)
    print("\n== per-agent pass rates (sorted by geomean speedup)")
    print(agg.round(2).to_string())
    print("\nspearman(agent process score, agent speedup) =", round(agg.process.corr(agg.speedup, method="spearman"), 2))

    print("\n== discrimination per assertion (std of per-agent pass rate; higher = separates agents)")
    print(agg[names].std().sort_values(ascending=False).round(2).to_string())

    print("\n== seed-consistency: share of (agent,scenario) cells where all seeds give the same answer")
    cons = {n: d.dropna(subset=[n]).groupby(["agent", "scenario"])[n].agg(lambda s: s.nunique() == 1 and len(s) >= 2).mean()
            for n in names}
    print(pd.Series(cons).round(2).to_string())

    print("\n== speedup geomean when assertion passes vs fails (scored runs, log-ratio to scenario median)")
    s = d[d.scored == 1].copy()
    s["lr"] = np.log(s.speedup) - s.groupby("scenario").speedup.transform(lambda x: np.log(x).median())
    for n in names:
        g = s.dropna(subset=[n]).groupby(n).lr.agg(["mean", "count"])
        if len(g) == 2:
            print(f"  {n:20s} pass {np.exp(g.loc[True,'mean']):.2f}x (n={g.loc[True,'count']})  fail {np.exp(g.loc[False,'mean']):.2f}x (n={g.loc[False,'count']})")

    print("\n== outcome vs process disagreements")
    print("  scored but never evaluated final config:", int(((d.scored == 1) & (d.final_measured == False)).sum()))
    print("  scored with evaluate.py modified:", int(((d.scored == 1) & (d.eval_untouched == False)).sum()))
    print("  scored with zero evals:", int(((d.scored == 1) & (d.ran_eval == False)).sum()))


if __name__ == "__main__":
    main()
