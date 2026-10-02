"""Rasch (1PL) fit of the run x assertion pass matrix: agent ability and assertion difficulty on one logit scale,
plus each assertion's discrimination (point-biserial with the rest-score) and infit-style misfit."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from blamegraph.assertions import ASSERTIONS  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "data" / "derived"


def long_format(d: pd.DataFrame, items: list[str]) -> pd.DataFrame:
    rows = []
    for _, r in d.iterrows():
        for it in items:
            v = r[it]
            if pd.notna(v):
                rows.append((r.agent, r.run_id, it, int(bool(v))))
    return pd.DataFrame(rows, columns=["agent", "run_id", "item", "y"])


def rasch(L: pd.DataFrame):
    """logit P(pass) = ability[agent] - difficulty[item], via one-hot logistic regression (ridge-regularized)."""
    A = pd.get_dummies(L.agent, prefix="a", dtype=float)
    I = pd.get_dummies(L.item, prefix="i", dtype=float)
    X = pd.concat([A, -I], axis=1).values
    m = LogisticRegression(C=5.0, max_iter=5000, fit_intercept=False).fit(X, L.y.values)
    coef = pd.Series(m.coef_[0], index=list(A.columns) + list(I.columns))
    ability = coef[[c for c in coef.index if c.startswith("a_")]].rename(lambda s: s[2:])
    difficulty = coef[[c for c in coef.index if c.startswith("i_")]].rename(lambda s: s[2:])
    # identify the scale: mean difficulty = 0
    shift = difficulty.mean()
    return ability - shift, difficulty - shift, m


def main():
    d = pd.read_csv(OUT / "report_runs.csv")
    items = [n for n in ASSERTIONS if n != "not_stub"]   # not_stub is constant
    L = long_format(d, items)
    ability, difficulty, m = rasch(L)

    # discrimination: point-biserial of item with the run's mean over the other items
    wide = d.set_index("run_id")[items].astype(float)
    disc = {}
    for it in items:
        rest = wide.drop(columns=[it]).mean(axis=1)
        mask = wide[it].notna()
        disc[it] = float(np.corrcoef(wide.loc[mask, it].astype(float), rest[mask])[0, 1]) if mask.sum() > 10 else np.nan

    print("== Assertion difficulty (logit; higher = harder to pass) and discrimination (r with rest-score)")
    tab = pd.DataFrame({"difficulty": difficulty, "pass_rate": wide.mean(), "discrimination": pd.Series(disc)}).sort_values("difficulty")
    print(tab.round(2).to_string())
    print("\nLow-discrimination items (r < 0.15) tell you little about the agent beyond the item itself:",
          ", ".join(tab[tab.discrimination < 0.15].index))

    print("\n== Agent ability (logit) with bootstrap 95% CI over runs")
    rng = np.random.default_rng(0)
    boots = {a: [] for a in ability.index}
    for _ in range(200):
        samp = pd.concat([g.sample(len(g), replace=True, random_state=int(rng.integers(1e9))) for _, g in d.groupby("agent")])
        ab, _, _ = rasch(long_format(samp, items))
        for a in ability.index:
            boots[a].append(ab.get(a, np.nan))
    spd = d.groupby("agent").speedup.apply(lambda s: float(np.exp(np.log(s).mean())))
    out = pd.DataFrame({"ability": ability, "lo": {a: np.nanpercentile(v, 2.5) for a, v in boots.items()},
                        "hi": {a: np.nanpercentile(v, 97.5) for a, v in boots.items()}, "speedup": spd}).sort_values("ability", ascending=False)
    print(out.round(2).to_string())
    print("\nSpearman(ability, speedup) =", round(out.ability.corr(out.speedup, method="spearman"), 2))
    out.to_csv(OUT / "irt_agents.csv"); tab.to_csv(OUT / "irt_items.csv")


if __name__ == "__main__":
    main()
