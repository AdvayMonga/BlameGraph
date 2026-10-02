"""Audit the outcome leaderboard itself: rank stability under seed resampling, reliability, harness normalization."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUT = Path(__file__).resolve().parent.parent / "data" / "derived"


def gm(s):
    s = np.asarray(s, dtype=float)
    return float(np.exp(np.log(s).mean())) if len(s) else float("nan")


def agg_speedup(d: pd.DataFrame) -> pd.Series:
    """InferenceBench aggregate: per-scenario mean over seeds, geomean over scenarios."""
    per = d.groupby(["agent", "scenario"]).speedup.mean().unstack()
    return per.apply(lambda r: gm(r.dropna()), axis=1)


def main():
    d = pd.read_csv(OUT / "report_runs.csv")
    agents = sorted(d.agent.unique())
    base = agg_speedup(d).sort_values(ascending=False)

    # 1) rank stability: resample seeds within each (agent, scenario) cell with replacement
    rng = np.random.default_rng(0)
    ranks = {a: [] for a in agents}
    B = 1000
    cells = {k: g.speedup.values for k, g in d.groupby(["agent", "scenario"])}
    for _ in range(B):
        rows = []
        for (a, sc), v in cells.items():
            rows.append((a, sc, rng.choice(v, len(v)).mean()))
        per = pd.DataFrame(rows, columns=["agent", "scenario", "m"]).pivot(index="agent", columns="scenario", values="m")
        agg = per.apply(lambda r: gm(r.dropna()), axis=1).rank(ascending=False)
        for a in agents:
            ranks[a].append(agg[a])
    print("== Speedup leaderboard rank with 95% interval under seed resampling (k=3 per cell)")
    print(f"{'agent':24s} {'speedup':>8s} {'rank':>5s}   95% rank interval")
    for a in base.index:
        r = np.array(ranks[a])
        print(f"{a:24s} {base[a]:8.2f} {int(base.rank(ascending=False)[a]):5d}   [{np.percentile(r, 2.5):.0f}, {np.percentile(r, 97.5):.0f}]")
    # pairwise: share of agent pairs whose order is stable in >=95% of resamples
    stable = 0; pairs = 0
    R = np.array([ranks[a] for a in agents])
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            pairs += 1
            p = np.mean(R[i] < R[j])
            stable += (p >= 0.95 or p <= 0.05)
    print(f"\nagent pairs with a stable order (>=95% of resamples): {stable}/{pairs} ({stable/pairs:.0%})")

    # 1b) same resampling for the BlameGraph score and its two halves: is process ranking more stable than outcome ranking?
    def pair_stability(col):
        cells_c = {k: g[col].dropna().values for k, g in d.groupby(["agent", "scenario"])}
        Rk = {a: [] for a in agents}
        for _ in range(B):
            rows = [(a, sc, rng.choice(v, len(v)).mean()) for (a, sc), v in cells_c.items() if len(v)]
            per = pd.DataFrame(rows, columns=["agent", "scenario", "m"]).pivot(index="agent", columns="scenario", values="m")
            rk = per.mean(axis=1).rank(ascending=False)
            for a in agents:
                Rk[a].append(rk.get(a, np.nan))
            M = np.array([Rk[a] for a in agents])
        st = 0
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                p = np.nanmean(M[i] < M[j]); st += (p >= 0.95 or p <= 0.05)
        return st / pairs
    for col in ("bg_score", "integrity", "self_consistency"):
        print(f"stable pairs for {col:17s}: {pair_stability(col):.0%}")

    # 2) reliability: all-seeds-scored cells, and worst-seed speedup
    print("\n== Reliability")
    rel = d.groupby(["agent", "scenario"]).scored.agg(lambda s: int(s.all())).groupby("agent").mean()
    worst = d.groupby(["agent", "scenario"]).speedup.min().unstack().apply(lambda r: gm(r.dropna()), axis=1)
    tab = pd.DataFrame({"speedup": base, "cells_all_scored": rel, "worst_seed_speedup": worst}).sort_values("speedup", ascending=False)
    print(tab.round(2).to_string())
    print("Spearman(mean speedup, worst-seed speedup) =", round(tab.speedup.corr(tab.worst_seed_speedup, method="spearman"), 2))

    # 3) harness normalization: drop runs with >20 blocked commands
    print("\n== Harness normalization (drop runs with >20 sandbox-blocked commands)")
    clean = d[d.blocked <= 20]
    dropped = d[d.blocked > 20].groupby("agent").size()
    b2 = agg_speedup(clean)
    bg2 = clean.groupby("agent").bg_score.mean()
    for a in dropped.index:
        print(f"  {a:24s} dropped {int(dropped[a])} runs: speedup {base[a]:.2f} -> {b2[a]:.2f}; BG {d[d.agent==a].bg_score.mean():.2f} -> {bg2[a]:.2f}")


if __name__ == "__main__":
    main()
