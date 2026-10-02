"""Early warning: do process signals observable by minute T predict whether the run will score at all?"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from blamegraph.experiment import build_log  # noqa: E402
from blamegraph.extract import obs_by_run  # noqa: E402
from blamegraph.traces import iter_runs  # noqa: E402


def features_at(run, log, T: float) -> dict | None:
    """Process counters using only events whose timer-anchored minute is <= T (runs without timer anchors are skipped)."""
    if not log.timer_marks:
        return None
    marks = log.timer_marks
    def minute(step_i):
        m = [t for si, t in marks if si <= step_i]
        return m[-1] if m else 0.0
    evals = [e for e in log.evals if minute(e.step_i) <= T]
    obs = [o for o in log.observations if minute(o.step_i) <= T]
    cfgs = [c for c in log.configs if minute(c.step_i) <= T]
    starts = [s for s, _ in log.server_starts if minute(s) <= T]
    return dict(
        n_evals=len(evals), n_full=sum(not e.quick for e in evals), n_observed=sum(e.observed for e in evals),
        n_killed=sum(e.killed for e in evals), n_cfgs=len(cfgs), n_starts=len(starts),
        any_healthy_obs=int(any((o.failure_rate or 0) == 0 and (o.ttft_p50 or o.tpot_p50 or o.rps) for o in obs)),
        any_failures=int(any((o.failure_rate or 0) > 0.1 for o in obs)),
        last_minute=max([minute(e.step_i) for e in evals], default=0.0),
    )


def main():
    llm = obs_by_run()
    rl = [(r, build_log(r, llm_obs=llm.get(r.run_id))) for r in iter_runs()]
    from sklearn.linear_model import LogisticRegression  # noqa: E402
    for T in (15, 30, 60, 90):
        rows = []
        for run, log in rl:
            f = features_at(run, log, T)
            if f:
                f.update(agent=run.agent, y=int(run.scored)); rows.append(f)
        d = pd.DataFrame(rows)
        X = d.drop(columns=["agent", "y"]).values.astype(float); y = d.y.values
        # leave-one-agent-out so the model cannot learn agent identity
        preds = np.zeros(len(d))
        for a in d.agent.unique():
            tr, te = d.agent != a, d.agent == a
            m = LogisticRegression(max_iter=2000, C=0.5).fit(X[tr], y[tr])
            preds[te] = m.predict_proba(X[te])[:, 1]
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y, preds)
        base = y.mean()
        # simple rule: "no healthy observation yet by minute T"
        rule = d.any_healthy_obs.values
        print(f"T={T:3d} min: n={len(d)} scored-rate={base:.2f}  LOAO logistic AUC={auc:.2f}  | rule 'has a healthy measurement by T': "
              f"scored-rate when yes {y[rule==1].mean():.2f} (n={int((rule==1).sum())}) vs no {y[rule==0].mean():.2f} (n={int((rule==0).sum())})")


if __name__ == "__main__":
    main()
