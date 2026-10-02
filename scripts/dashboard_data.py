"""Export everything the dashboard needs into one JSON (data/derived/dashboard.json)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from blamegraph.assertions import ASSERTIONS  # noqa: E402
from blamegraph.blame import blame  # noqa: E402
from blamegraph.experiment import build_log  # noqa: E402
from blamegraph.extract import obs_by_run  # noqa: E402
from blamegraph.judge import QUESTIONS  # noqa: E402
from blamegraph.noise import _primary  # noqa: E402
from blamegraph.traces import BASELINE_METRIC, iter_runs  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "data" / "derived"
SCORED = {"integrity": ["eval_untouched", "final_measured", "claims_traceable", "not_stub"],
          "self_consistency": ["reads_results", "kept_best", "no_abandoned_evals", "no_stale_evals", "final_full_eval"]}


def gm(s):
    s = np.asarray([x for x in s if x and x > 0], dtype=float)
    return float(np.exp(np.log(s).mean())) if len(s) else None


def nz(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return round(float(v), 4)
    return v


def main():
    d = pd.read_csv(OUT / "report_runs.csv")
    names = list(ASSERTIONS)
    for c in names:
        d[c] = d[c].map(lambda v: None if pd.isna(v) else bool(v))
    role = {n: ("integrity" if n in SCORED["integrity"] else "self_consistency" if n in SCORED["self_consistency"] else "descriptive") for n in names}
    rng = np.random.default_rng(0)
    agents = []
    for a, g in d.groupby("agent"):
        v = g.bg_score.dropna().values
        bs = [rng.choice(v, len(v)).mean() for _ in range(1000)]
        agents.append(dict(agent=a, harness=g.harness.iloc[0], n=len(g), speedup=nz(gm(g.speedup)), scored=nz(g.scored.mean()),
                           bg=nz(g.bg_score.mean()), bg_lo=nz(np.percentile(bs, 2.5)), bg_hi=nz(np.percentile(bs, 97.5)),
                           integrity=nz(g.integrity.mean()), self_consistency=nz(g.self_consistency.mean()),
                           judge=nz(g.judge_score.mean()) if "judge_score" in g else None,
                           blocked_runs=int((g.blocked > 20).sum()), last_min=nz(g.last_min.median()),
                           worst_seed=nz(gm(g.groupby("scenario").speedup.min())),
                           passes={n: nz(g[n].dropna().astype(float).mean()) if g[n].notna().any() else None for n in names}))
    agents.sort(key=lambda r: -(r["speedup"] or 0))
    # assertion metadata
    wide = d.set_index("run_id")[names].astype(float)
    items = []
    for n in names:
        col = wide[n].dropna(); rest = wide.drop(columns=[n]).mean(axis=1); m = wide[n].notna()
        disc = float(np.corrcoef(wide.loc[m, n], rest[m])[0, 1]) if m.sum() > 10 and wide.loc[m, n].std() > 0 else None
        cons = d.dropna(subset=[n]).groupby(["agent", "scenario"])[n].agg(lambda s: s.nunique() == 1 and len(s) >= 2).mean()
        items.append(dict(name=n, text=ASSERTIONS[n][0], role=role[n], pass_rate=nz(col.mean()), n=int(len(col)),
                          discrimination=nz(disc), seed_consistency=nz(cons), spread=nz(d.groupby("agent")[n].mean().std())))
    # judge findings
    jp = OUT / "judgments.csv"
    findings = []
    if jp.exists():
        j = pd.read_csv(jp)
        for _, r in j[j.applicable == 1].iterrows():
            findings.append(dict(run_id=r.run_id, agent=r.agent, question=r.question, answer=r.answer, bad=int(r.bad),
                                 confidence=nz(r.confidence), evidence=str(r.evidence)[:220], rationale=str(r.rationale)[:260]))
    # runs + timelines
    llm = obs_by_run(); runs = []; timelines = {}
    meta = d.set_index("run_id")
    for run in iter_runs():
        log = build_log(run, llm_obs=llm.get(run.run_id)); b = blame(run, log, full_only=True)
        base = BASELINE_METRIC[run.scenario]; m = meta.loc[run.run_id]
        ev = []
        marks = log.timer_marks
        def minute(si):
            mm = [t for s, t in marks if s <= si]; return mm[-1] if mm else None
        for c in log.configs:
            ev.append(dict(i=c.step_i, t="config", v=c.idx, eng=c.engine, min=minute(c.step_i)))
        for si, idx in log.server_starts:
            ev.append(dict(i=si, t="start", v=idx, min=minute(si)))
        for e in log.evals:
            ev.append(dict(i=e.step_i, t="eval", v=e.config_idx, quick=e.quick, stale=e.config_stale, killed=e.killed, obs=e.observed, std=e.standard, min=minute(e.step_i)))
        for o in log.observations:
            p = _primary(o, run.scenario)
            ev.append(dict(i=o.step_i, t="obs", v=o.config_idx, sp=nz(p / base) if p else None, fail=nz(o.failure_rate), min=minute(o.step_i)))
        ev.sort(key=lambda x: x["i"])
        timelines[run.run_id] = dict(events=ev, n_steps=len(run.steps()), final_cfg=log.final_config.idx if log.final_config else None,
                                     blame=[dict(i=e.step_i, kind=e.kind, note=e.note) for e in b.events], eval_modified=log.eval_script_modified)
        runs.append(dict(run_id=run.run_id, agent=run.agent, scenario=run.scenario, harness=run.harness, scored=bool(run.scored),
                         speedup=nz(run.speedup), bg=nz(m.bg_score), integrity=nz(m.integrity), self_consistency=nz(m.self_consistency),
                         regret=nz(m.regret), found=nz(b.found), shipped_obs=nz(b.shipped_obs), phase=b.phase_of_loss,
                         blocked=int(m.blocked), last_min=nz(m.last_min), n_evals=len(log.evals), n_configs=len(log.configs),
                         n_measured=log.n_measured_distinct, asserts={n: m[n] if m[n] is None or isinstance(m[n], bool) else bool(m[n]) for n in names}))
    payload = dict(generated="2026-10-02", n_runs=len(runs), agents=agents, assertions=items, runs=runs, timelines=timelines,
                   findings=findings, questions=QUESTIONS,
                   headline=dict(never_measured_final=int(((d.scored == 1) & (d.final_measured == False)).sum()),
                                 zero_evals=int(((d.scored == 1) & (d.ran_eval == False)).sum()),
                                 grader_edited=int(((d.scored == 1) & (d.eval_untouched == False)).sum()),
                                 regret_runs=int((d.regret > 1.05).sum()), regret_n=int(d.regret.notna().sum()),
                                 stable_pairs_speedup=0.49, stable_pairs_bg=0.64, icc_speedup=0.18, icc_selfc=0.34))
    (OUT / "dashboard.json").write_text(json.dumps(payload, default=nz))
    print("wrote dashboard.json", round((OUT / "dashboard.json").stat().st_size / 1e6, 2), "MB;", len(findings), "findings")


if __name__ == "__main__":
    main()
