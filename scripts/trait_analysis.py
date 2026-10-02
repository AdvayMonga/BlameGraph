"""Is process a property of the agent? Cross-scenario consistency (ICC) of process vs speedup, and effort-level pairs."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
OUT = Path(__file__).resolve().parent.parent / "data" / "derived"


def icc1(frame: pd.DataFrame, col: str) -> float:
    """One-way random-effects ICC(1): share of variance in `col` explained by agent identity."""
    g = frame.dropna(subset=[col]).groupby("agent")[col]
    k = g.size(); n = len(k)
    if n < 2:
        return float("nan")
    grand = frame[col].mean()
    msb = sum(k_i * (m - grand) ** 2 for k_i, m in zip(k, g.mean())) / (n - 1)
    msw = sum(((frame.loc[frame.agent == a, col] - g.mean()[a]) ** 2).sum() for a in k.index) / (k.sum() - n)
    k0 = (k.sum() - (k ** 2).sum() / k.sum()) / (n - 1)
    return float((msb - msw) / (msb + (k0 - 1) * msw))


def main():
    d = pd.read_csv(OUT / "report_runs.csv")
    d["log_speedup"] = np.log(d.speedup)
    # within-scenario z-scores so scenarios with different speedup scales are comparable
    for c in ("log_speedup", "bg_score", "self_consistency", "integrity", "judge_score", "bg_plus"):
        if c in d:
            d[c + "_z"] = d.groupby("scenario")[c].transform(lambda s: (s - s.mean()) / (s.std() or 1))
    print("== ICC(1): how much of a run's score is explained by *which agent* ran it (higher = more trait-like)")
    for c in ("log_speedup_z", "bg_score_z", "self_consistency_z", "integrity_z", "judge_score_z", "bg_plus_z"):
        if c in d:
            print(f"  {c:20s} ICC = {icc1(d, c):.2f}")

    print("\n== split-half: rank agents on scenarios A+B vs C+D; Spearman between halves")
    for c in ("log_speedup", "bg_score", "self_consistency"):
        h1 = d[d.scenario.isin(["A", "B"])].groupby("agent")[c].mean()
        h2 = d[d.scenario.isin(["C", "D"])].groupby("agent")[c].mean()
        print(f"  {c:18s} rho = {h1.corr(h2, method='spearman'):.2f}")

    print("\n== same model, different effort / variant")
    pairs = [("claude-fable-5", "claude-fable-5-low"), ("gpt-5.5-xhigh", "gpt-5.5-high"), ("gpt-5.3-codex-high", "gpt-5.3-codex-med"),
             ("claude-opus-4-8-xhigh", "claude-opus-4-8"), ("gpt-5.2", "gpt-5.2-codex"), ("glm-5.2-max", "glm-5")]
    g = d.groupby("agent").agg(speedup=("speedup", lambda s: float(np.exp(np.log(s).mean()))), bg=("bg_score", "mean"),
                              selfc=("self_consistency", "mean"), evals=("n_evals", "median"), last_min=("last_min", "median"))
    for a, b in pairs:
        if a in g.index and b in g.index:
            print(f"  {a:22s} speedup {g.speedup[a]:.2f} bg {g.bg[a]:.2f} selfc {g.selfc[a]:.2f} evals {g.evals[a]:.0f} last_min {g.last_min[a]:.0f}")
            print(f"  {b:22s} speedup {g.speedup[b]:.2f} bg {g.bg[b]:.2f} selfc {g.selfc[b]:.2f} evals {g.evals[b]:.0f} last_min {g.last_min[b]:.0f}\n")


if __name__ == "__main__":
    main()
