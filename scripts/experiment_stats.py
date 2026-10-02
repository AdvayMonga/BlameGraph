"""Code-derived process metrics per run + how they relate to outcome. Writes data/derived/experiment.csv."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from blamegraph.experiment import build_log  # noqa: E402
from blamegraph.extract import obs_by_run  # noqa: E402
from blamegraph.traces import iter_runs  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "data" / "derived"


def row_for(run, llm=None) -> dict:
    log = build_log(run, llm_obs=llm)
    engines = [c.engine for c in log.configs]
    return dict(
        run_id=run.run_id, agent=run.agent, scenario=run.scenario, harness=run.harness,
        scored=int(run.scored), gate_pass=int(run.gate_passed), flagged=int(run.flagged),
        outcome=run.outcome, speedup=run.speedup,
        n_steps=len(run.steps()),
        n_configs=len(log.configs), n_distinct=log.n_distinct_configs,
        n_engines=len(set(engines) - {"stub", "unknown"}), final_engine=log.final_config.engine if log.final_config else None,
        n_starts=len(log.server_starts),
        n_evals=len(log.evals), n_quick=sum(e.quick for e in log.evals), n_full=sum(not e.quick for e in log.evals),
        n_stale=log.stale_eval_count, n_unobserved=log.unobserved_eval_count, n_killed=sum(e.killed for e in log.evals),
        n_measured_distinct=log.n_measured_distinct, n_observed_distinct=log.n_observed_distinct,
        n_obs=len(log.observations),
        final_measured=int(log.final_config_measured), final_full=int(log.final_config_full_eval),
        eval_modified=int(log.eval_script_modified),
        regret=log.regret(run.scenario),
        first_eval_min=next((e.minute for e in log.evals if e.minute is not None), None),
        last_cfg_min=log.final_config.minute if log.final_config else None,
        last_timer_min=log.timer_marks[-1][1] if log.timer_marks else None,
        n_timer=len(log.timer_marks),
    )


def main():
    llm = obs_by_run()
    rows = [row_for(r, llm.get(r.run_id)) for r in iter_runs()]
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "experiment.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    d = pd.DataFrame(rows)
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40)
    print(f"{len(d)} runs\n")
    cols = ["n_configs", "n_distinct", "n_engines", "n_starts", "n_evals", "n_quick", "n_full", "n_stale",
            "n_unobserved", "n_killed", "n_measured_distinct", "n_observed_distinct", "n_obs", "n_timer"]
    print("== quantiles"); print(d[cols].describe(percentiles=[.1, .5, .9]).T[["min", "10%", "50%", "90%", "max"]].round(1))
    print("\n== binary flags (share of runs)")
    print(d[["final_measured", "final_full", "eval_modified"]].mean().round(3).to_dict())
    print("regret available:", d.regret.notna().sum(), " regret>1.05:", (d.regret > 1.05).sum())
    print("\n== within-scenario rank correlation of process counters with outcome (scored runs only)")
    for c in ["n_distinct", "n_evals", "n_full", "n_measured_distinct", "n_observed_distinct", "n_stale", "n_unobserved", "n_starts"]:
        rs = [stats.spearmanr(g[c], g.outcome).correlation for _, g in d[d.scored == 1].groupby("scenario") if g[c].nunique() > 1]
        print(f"  {c:22s} mean spearman across scenarios = {np.mean(rs):+.2f}")
    print("\n== per-agent medians")
    g = d.groupby("agent").agg(n=("run_id", "count"), scored=("scored", "mean"), distinct=("n_distinct", "median"),
                              evals=("n_evals", "median"), full=("n_full", "median"), meas=("n_measured_distinct", "median"),
                              obs=("n_observed_distinct", "median"), unobs=("n_unobserved", "median"),
                              final_full=("final_full", "mean"), eval_mod=("eval_modified", "mean")).sort_values("scored", ascending=False)
    print(g.round(2))


if __name__ == "__main__":
    main()
