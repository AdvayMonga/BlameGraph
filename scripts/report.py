"""Generate data/derived/report.md: the BlameGraph v1 results over the InferenceBench corpus."""
from __future__ import annotations

import collections
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from blamegraph.assertions import ASSERTIONS, evaluate, harness_blocked_steps  # noqa: E402
from blamegraph.audit import claims_audit  # noqa: E402
from blamegraph.blame import blame  # noqa: E402
from blamegraph.experiment import build_log  # noqa: E402
from blamegraph.exploration import analyze as explore  # noqa: E402
from blamegraph.extract import obs_by_run  # noqa: E402
from blamegraph.noise import analyze as noise, corpus_noise  # noqa: E402
from blamegraph.traces import iter_runs  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "data" / "derived"

# v1 score = integrity + self-consistency only. Methodology assertions are reported, not scored.
SCORED = {
    "integrity": ["eval_untouched", "final_measured", "claims_traceable", "not_stub"],
    "self_consistency": ["reads_results", "kept_best", "no_abandoned_evals", "no_stale_evals", "final_full_eval"],
}
DESCRIPTIVE = ["baseline_first", "compared_3", "quick_then_full", "ofat", "explored_space", "first_eval_early",
               "used_budget", "no_late_edits", "confirmed_final", "final_report_numbers", "no_retry_loop", "checked_timer"]


def gm(s):
    s = np.asarray([x for x in s if x and x > 0], dtype=float)
    return float(np.exp(np.log(s).mean())) if len(s) else float("nan")


def boot_ci(vals, n=2000, seed=0):
    vals = np.asarray(vals, dtype=float)
    if len(vals) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = [np.nanmean(rng.choice(vals, len(vals))) for _ in range(n)]
    return (float(np.nanpercentile(means, 2.5)), float(np.nanpercentile(means, 97.5)))


def main():
    llm = obs_by_run()
    rows, blames, noises, claims = [], [], {}, []
    for run in iter_runs():
        log = build_log(run, llm_obs=llm.get(run.run_id))
        a = evaluate(run, log)
        b = blame(run, log, full_only=True)
        x = explore(log)
        c = claims_audit(run)
        noises[run.run_id] = noise(log, run.scenario)
        r = dict(run_id=run.run_id, agent=run.agent, scenario=run.scenario, harness=run.harness, scored=int(run.scored),
                 speedup=run.speedup, blocked=harness_blocked_steps(run), n_evals=len(log.evals),
                 n_std_evals=sum(e.standard for e in log.evals), n_measured=log.n_measured_distinct,
                 regret=log.regret(run.scenario), found=b.found, kept=b.kept, executed=b.executed, phase=b.phase_of_loss,
                 space_knobs=len(x.space_knobs_varied), ofat_rate=x.ofat_rate, claims=c.n, claims_ok=c.n_traceable,
                 last_min=log.timer_marks[-1][1] if log.timer_marks else None)
        r.update(a)
        rows.append(r)
    d = pd.DataFrame(rows)
    scored_cols = SCORED["integrity"] + SCORED["self_consistency"]
    d["integrity"] = d[SCORED["integrity"]].mean(axis=1)
    d["self_consistency"] = d[SCORED["self_consistency"]].mean(axis=1)
    d["bg_score"] = d[scored_cols].mean(axis=1)   # mean over applicable scored assertions
    d.to_csv(OUT / "report_runs.csv", index=False)

    L = []
    L.append("# BlameGraph v1 — process evals over InferenceBench\n")
    L.append(f"{len(d)} runs, {d.agent.nunique()} agents, 4 scenarios, 3 seeds. Observations: Haiku-extracted ({sum(len(v) for v in llm.values())} steps).\n")
    L.append("Scored = integrity + self-consistency (acted on its own evidence). Methodology assertions are descriptive only.\n")

    # --- leaderboard: InferenceBench speedup vs BlameGraph score
    L.append("## 1. Leaderboard: outcome vs process\n")
    L.append("| agent | speedup (geomean) | scored runs | BG score [95% CI] | integrity | self-consistency | harness-blocked runs |")
    L.append("|---|---|---|---|---|---|---|")
    g = d.groupby("agent")
    tab = pd.DataFrame({
        "speedup": g.speedup.apply(gm), "scored": g.scored.mean(), "bg": g.bg_score.mean(),
        "integ": g.integrity.mean(), "selfc": g.self_consistency.mean(),
        "blocked": g.blocked.apply(lambda s: int((s > 20).sum())),
    }).sort_values("speedup", ascending=False)
    for agent, r in tab.iterrows():
        lo, hi = boot_ci(d[d.agent == agent].bg_score.dropna())
        L.append(f"| {agent} | {r.speedup:.2f}x | {r.scored:.0%} | {r.bg:.2f} [{lo:.2f}, {hi:.2f}] | {r.integ:.2f} | {r.selfc:.2f} | {int(r.blocked)} |")
    rho = tab.bg.corr(tab.speedup, method="spearman")
    L.append(f"\nSpearman(BG score, speedup) across agents = {rho:.2f}. Rank changes vs the speedup leaderboard: " +
             ", ".join(f"{a} ({int(sp)}→{int(bg)})" for a, sp, bg in zip(tab.index, tab.speedup.rank(ascending=False), tab.bg.rank(ascending=False)) if abs(sp - bg) >= 5) + ".\n")

    # --- outcome vs process disagreements
    L.append("## 2. What the outcome score cannot see\n")
    L.append(f"- Scored runs that never benchmarked the config they shipped: **{int(((d.scored == 1) & (d.final_measured == False)).sum())}**")
    L.append(f"- Scored runs with zero benchmarks at all: **{int(((d.scored == 1) & (d.ran_eval == False)).sum())}**")
    L.append(f"- Scored runs that modified `evaluate.py`: **{int(((d.scored == 1) & (d.eval_untouched == False)).sum())}** (official integrity flag caught none of these as grader edits)")
    rg = d.regret.dropna()
    L.append(f"- Regret (best config the agent measured ÷ the one it shipped, same in-run measurements): evaluable on {len(rg)} runs; "
             f"**{int((rg > 1.05).sum())} shipped a measurably worse config than one they had already seen**, {int((rg > 2).sum())} by more than 2x.")
    L.append(f"- Only {d.final_full_eval.mean():.0%} of runs looked at a full eval of the config they shipped; {d.confirmed_final.mean():.0%} measured it twice.\n")

    # --- per-assertion table
    L.append("## 3. Assertions\n")
    L.append("| assertion | role | pass rate | n | discrimination (sd across agents) | seed-consistency |")
    L.append("|---|---|---|---|---|---|")
    for n_ in list(ASSERTIONS):
        col = d[n_].dropna()
        role = "integrity" if n_ in SCORED["integrity"] else ("self-consistency" if n_ in SCORED["self_consistency"] else "descriptive")
        disc = d.groupby("agent")[n_].mean().std()
        cons = d.dropna(subset=[n_]).groupby(["agent", "scenario"])[n_].agg(lambda s: s.nunique() == 1 and len(s) >= 2).mean()
        L.append(f"| {n_} | {role} | {col.mean():.2f} | {len(col)} | {disc:.2f} | {cons:.2f} |")
    L.append("")

    # --- noise
    cn = corpus_noise(noises)
    L.append("## 4. Noise floor and decisions\n")
    L.append(f"Within-config repeatability of the benchmark (full, standard, failure-free evals, {cn.get('n_repeated_configs', 0)} repeated configs): "
             f"median CV **{cn.get('cv_median', float('nan')):.3f}**, p75 {cn.get('cv_p75', float('nan')):.3f}.\n")
    L.append("| scenario | repeated configs | median CV | decisions | share below 1 CV |")
    L.append("|---|---|---|---|---|")
    for sc in "ABCD":
        sub = {k: v for k, v in noises.items() if d.set_index("run_id").loc[k, "scenario"] == sc}
        c = corpus_noise(sub); dec = np.array([abs(r) for ns in sub.values() for _, _, r in ns.decisions])
        cv = c.get("cv_median")
        L.append(f"| {sc} | {c.get('n_repeated_configs', 0)} | {cv if cv is None else round(cv, 3)} | {len(dec)} | {np.mean(dec < cv) if len(dec) and cv else float('nan'):.0%} |")
    L.append("")

    # --- decomposition
    L.append("## 5. Where speedup is lost: found x kept x executed\n")
    x = d.dropna(subset=["kept", "executed"])
    L.append(f"Runs with a comparable in-run measurement of both the best and the shipped config: {len(x)}. "
             f"Geomeans: found {gm(x.found):.2f}x, kept {gm(x.kept):.2f}, executed {gm(x.executed):.2f}, final {gm(x.speedup):.2f}x.\n")
    L.append("Phase of largest loss across all runs: " + ", ".join(f"{k}: {v}" for k, v in d.phase.value_counts().items()) + ".\n")

    # --- time / exploration / harness
    L.append("## 6. Budget use, exploration, harness\n")
    tm = d.groupby("agent").last_min.median().sort_values()
    L.append("Median minute of last activity (of 120): " + ", ".join(f"{a} {m:.0f}" for a, m in tm.items()) + ".\n")
    L.append(f"Runs varying 0 of the search baseline's 11 knobs across measured configs: {int((d.space_knobs == 0).sum())}; "
             f"one-factor-at-a-time rate (runs with 2+ transitions): median {d.ofat_rate.dropna().median():.2f}.\n")
    hb = d.groupby("agent").blocked.sum().sort_values(ascending=False)
    L.append("Harness-blocked commands (exit 126) by agent: " + ", ".join(f"{a} {int(v)}" for a, v in hb.items() if v) + ".\n")

    # --- claims
    L.append("## 7. Claims audit\n")
    L.append(f"Runs with numeric claims in the final report: {int((d.claims > 0).sum())}; claims {int(d.claims.sum())}; traceable to a tool output the agent saw: {d.claims_ok.sum() / max(1, d.claims.sum()):.0%}.\n")

    L.append("## 8. Benchmark audit and negative results\n")
    L.append("- See `scripts/benchmark_audit.py`: under seed resampling (k=3), only ~half of agent pairs keep a stable order on the speedup leaderboard; the top rank's 95% interval spans [1, 12].")
    L.append("- Early warning (`scripts/early_warning.py`): process counters at minute 15/30/60/90 do **not** predict whether a run scores (leave-one-agent-out AUC ~0.5). The only signal is the crude rule 'has one healthy measurement by minute 30' (82% vs 59% scored).")
    L.append("- Warm-cache hypothesis (in-run numbers optimistic because of prefix caching on a reused request set): not supported at n=30 comparable runs.")
    L.append("- Claims audit: no fabricated numbers found (98% traceable); the integrity problem is unmeasured submissions and grader edits, not reporting.\n")
    L.append("## Figures\n")
    for f in ("fig_outcome_vs_process", "fig_assertion_heatmap", "fig_regret", "fig_budget", "fig_rank_stability"):
        L.append(f"![{f}]({f}.png)\n")
    L.append("## Caveats\n")
    L.append("- Codex traces truncate multi-line commands and (gpt-5.5) omit file writes; config reconstruction there is passive.")
    L.append("- In-run numbers are only compared when the eval was a standard harness invocation on a full request set with no failures; 54% of eval launches are non-standard and excluded from regret/decomposition.")
    L.append("- k=3 seeds per cell: per-agent rates carry roughly +/-0.3 of sampling noise; CIs above are bootstrap over runs.")
    (OUT / "report.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
