"""Figures for the report -> data/derived/fig_*.png"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
OUT = Path(__file__).resolve().parent.parent / "data" / "derived"

ACCENT, MUTED, GRID = "#2563eb", "#64748b", "#e2e8f0"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True})


def gm(s):
    s = np.asarray([x for x in s if x and x > 0], dtype=float)
    return float(np.exp(np.log(s).mean())) if len(s) else np.nan


def main():
    d = pd.read_csv(OUT / "report_runs.csv")
    for c in ("bg_score", "self_consistency", "integrity", "regret", "last_min", "speedup"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    g = d.groupby("agent")
    agents = pd.DataFrame({"speedup": g.speedup.apply(gm), "bg": g.bg_score.mean(), "selfc": g.self_consistency.mean(),
                           "integ": g.integrity.mean(), "harness": g.harness.first()})
    # CIs for bg
    rng = np.random.default_rng(0)
    lo, hi = {}, {}
    for a, grp in g:
        v = grp.bg_score.dropna().values
        bs = [rng.choice(v, len(v)).mean() for _ in range(1000)]
        lo[a], hi[a] = np.percentile(bs, 2.5), np.percentile(bs, 97.5)
    agents["lo"], agents["hi"] = pd.Series(lo), pd.Series(hi)

    # Fig 1: outcome vs process
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"claude": "#2563eb", "codex": "#dc2626", "opencode": "#16a34a"}
    for a, r in agents.iterrows():
        ax.errorbar(r.speedup, r.bg, yerr=[[r.bg - r.lo], [r.hi - r.bg]], fmt="o", color=colors[r.harness], ecolor=GRID, capsize=2, ms=5)
        ax.annotate(a.replace("claude-", "").replace("gemini-", "gem-"), (r.speedup, r.bg), textcoords="offset points", xytext=(4, 3), fontsize=7, color=MUTED)
    ax.set_xscale("log"); ax.set_xlabel("InferenceBench speedup (geomean over scenarios, log)"); ax.set_ylabel("BlameGraph score (integrity + self-consistency)")
    ax.set_title("Outcome vs process, per agent (95% bootstrap CI over runs)")
    for h, c in colors.items():
        ax.plot([], [], "o", color=c, label=h)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(OUT / "fig_outcome_vs_process.png", dpi=160); plt.close(fig)

    # Fig 2: assertion heatmap (agents x assertions), sorted by speedup
    from blamegraph.assertions import ASSERTIONS
    items = [n for n in ASSERTIONS if n != "not_stub"]
    M = d[items].astype(float).groupby(d.agent).mean().loc[agents.sort_values("speedup", ascending=False).index]
    fig, ax = plt.subplots(figsize=(11, 7))
    im = ax.imshow(M.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(items))); ax.set_xticklabels(items, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(M))); ax.set_yticklabels([f"{a}  ({agents.speedup[a]:.1f}x)" for a in M.index], fontsize=7)
    ax.grid(False); fig.colorbar(im, ax=ax, fraction=0.02, label="pass rate")
    ax.set_title("Assertion pass rates per agent (rows sorted by speedup)")
    fig.tight_layout(); fig.savefig(OUT / "fig_assertion_heatmap.png", dpi=160); plt.close(fig)

    # Fig 3: regret distribution
    r = d.regret.dropna()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(np.clip(r, 1, 5), bins=np.linspace(1, 5, 33), color=ACCENT)
    ax.axvline(1.05, color=MUTED, ls="--", lw=1); ax.set_xlabel("regret = best measured / shipped (clipped at 5x)"); ax.set_ylabel("runs")
    ax.set_title(f"Regret on {len(r)} runs: {int((r > 1.05).sum())} shipped a config they had measured as worse")
    fig.tight_layout(); fig.savefig(OUT / "fig_regret.png", dpi=160); plt.close(fig)

    # Fig 4: last activity minute per agent
    lm = d.groupby("agent").last_min.median().loc[agents.sort_values("speedup", ascending=False).index]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(range(len(lm)), lm.values, color=[colors[agents.harness[a]] for a in lm.index])
    ax.set_yticks(range(len(lm))); ax.set_yticklabels(lm.index, fontsize=7); ax.invert_yaxis()
    ax.axvline(120, color=MUTED, ls="--", lw=1); ax.set_xlabel("median minute of last activity (budget = 120)")
    ax.set_title("Budget use (rows sorted by speedup)")
    fig.tight_layout(); fig.savefig(OUT / "fig_budget.png", dpi=160); plt.close(fig)

    # Fig 5: rank intervals of the speedup leaderboard
    cells = {k: grp.speedup.values for k, grp in d.groupby(["agent", "scenario"])}
    names = sorted(d.agent.unique()); ranks = {a: [] for a in names}
    for _ in range(1000):
        rows = [(a, sc, rng.choice(v, len(v)).mean()) for (a, sc), v in cells.items()]
        per = pd.DataFrame(rows, columns=["agent", "scenario", "m"]).pivot(index="agent", columns="scenario", values="m")
        rk = per.apply(lambda row: gm(row.dropna()), axis=1).rank(ascending=False)
        for a in names:
            ranks[a].append(rk[a])
    order = agents.sort_values("speedup", ascending=False).index
    fig, ax = plt.subplots(figsize=(7, 5))
    for y, a in enumerate(order):
        q = np.percentile(ranks[a], [2.5, 50, 97.5])
        ax.plot([q[0], q[2]], [y, y], color=GRID, lw=4); ax.plot(q[1], y, "o", color=ACCENT, ms=4)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=7); ax.invert_yaxis()
    ax.set_xlabel("leaderboard rank (95% interval under seed resampling, k=3)"); ax.set_title("How stable is the speedup leaderboard?")
    fig.tight_layout(); fig.savefig(OUT / "fig_rank_stability.png", dpi=160); plt.close(fig)
    print("wrote", sorted(p.name for p in OUT.glob("fig_*.png")))


if __name__ == "__main__":
    main()
