"""Analyze cached judgments: answer rates per question/agent, agreement with code assertions, and examples."""
from __future__ import annotations

import collections
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from blamegraph.judge import QUESTIONS, load_cache  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "data" / "derived"

# for each question, which answer indicates a process failure
BAD = {"noticed_failures": "no", "regression_investigated": "no", "abandon_reasoned": "no", "error_reacted": "no",
       "stale_aware": "yes", "headline_from_shipped": "no", "claims_supported": "yes"}


def main():
    cache = load_cache()
    runs = pd.read_csv(OUT / "report_runs.csv").set_index("run_id")
    rows = []
    for (rid, q, anchor), d in cache.items():
        a = d["data"]
        rows.append(dict(run_id=rid, agent=runs.loc[rid, "agent"] if rid in runs.index else "?", question=q, anchor=anchor,
                         answer=a["answer"], confidence=a.get("confidence"), evidence=a.get("evidence", "")[:160],
                         rationale=a.get("rationale", "")[:200], bad=int(a["answer"] == BAD[q]), applicable=int(a["answer"] != "not_applicable"),
                         tokens_in=d["in"], tokens_out=d["out"]))
    j = pd.DataFrame(rows)
    j.to_csv(OUT / "judgments.csv", index=False)
    cost = j.tokens_in.sum() / 1e6 * 2 + j.tokens_out.sum() / 1e6 * 10
    print(f"{len(j)} judgments over {j.run_id.nunique()} runs; cost ${cost:.2f}; mean confidence {j.confidence.mean():.2f}\n")
    print("== per question: applicable / failure rate among applicable")
    for q in QUESTIONS:
        s = j[j.question == q]
        app = s[s.applicable == 1]
        print(f"  {q:24s} n={len(s):3d} applicable={len(app):3d} failure-rate={app.bad.mean() if len(app) else float('nan'):.2f}  (bad answer = '{BAD[q]}')")
    print("\n== per agent: judged-failure rate across applicable questions (n applicable)")
    g = j[j.applicable == 1].groupby("agent").bad.agg(["mean", "count"]).sort_values("mean")
    print(g.round(2).to_string())
    # does the judge add information beyond code? compare with the code assertions on the same runs
    print("\n== judge vs code on the same runs")
    piv = j[j.applicable == 1].pivot_table(index="run_id", columns="question", values="bad", aggfunc="max")
    for q, code in (("noticed_failures", "reads_results"), ("headline_from_shipped", "final_measured"), ("claims_supported", "final_full_eval")):
        if q in piv:
            both = piv[[q]].join(runs[[code]], how="inner").dropna()
            if len(both):
                ct = pd.crosstab(both[q], both[code].astype(bool))
                print(f"  {q} (1=judged failure) vs {code} (True=code pass):\n{ct.to_string()}\n")
    print("== examples of judged failures (highest confidence)")
    for q in QUESTIONS:
        ex = j[(j.question == q) & (j.bad == 1)].sort_values("confidence", ascending=False).head(2)
        for _, r in ex.iterrows():
            print(f"  [{q}] {r.run_id} {r.agent} conf={r.confidence:.2f}\n     evidence: {r.evidence}\n     rationale: {r.rationale}")


if __name__ == "__main__":
    main()
