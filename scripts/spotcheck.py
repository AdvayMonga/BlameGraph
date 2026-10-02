"""Sample judgments for human labeling -> data/derived/spotcheck.csv (fill the `human` column with yes/no/not_applicable)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from blamegraph.experiment import build_log  # noqa: E402
from blamegraph.extract import obs_by_run  # noqa: E402
from blamegraph.judge import QUESTIONS, build_instances  # noqa: E402
from blamegraph.traces import DATA_ROOT, load_run  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "data" / "derived"


def main(n_per_cell: int = 4):
    j = pd.read_csv(OUT / "judgments.csv")
    j = j[j.applicable == 1]
    # stratify: each question x answer cell
    samp = pd.concat([g.sample(min(len(g), n_per_cell), random_state=0) for _, g in j.groupby(["question", "answer"])])
    llm = obs_by_run(); rows = []
    for rid, g in samp.groupby("run_id"):
        run = load_run(DATA_ROOT / "runs" / rid)
        inst = {(i.question, i.anchor): i.window for i in build_instances(run, build_log(run, llm_obs=llm.get(rid)))}
        for _, r in g.iterrows():
            rows.append(dict(run_id=rid, agent=r.agent, question=r.question, question_text=QUESTIONS[r.question], anchor=r.anchor,
                             judge_answer=r.answer, judge_confidence=r.confidence, judge_evidence=r.evidence, human="",
                             window=inst.get((r.question, r.anchor), "")))
    out = pd.DataFrame(rows).sample(frac=1, random_state=1)
    out.to_csv(OUT / "spotcheck.csv", index=False)
    print(f"wrote {len(out)} items to {OUT / 'spotcheck.csv'}; per question:", out.question.value_counts().to_dict())


if __name__ == "__main__":
    main()
