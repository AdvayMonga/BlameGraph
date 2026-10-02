"""Evaluate a Laya checkpoint (zero-shot or fine-tuned) on the System-One test sets as noul questions.
Usage: laya_eval.py [--model convaiinnovations/laya-typed-decisions] [--task extract|judge] [--split test] [--limit N]"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
DATA = Path(__file__).resolve().parent.parent / "data" / "laya"
OUT = Path(__file__).resolve().parent.parent / "data" / "derived"


def ece(p, y, bins=10):
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="convaiinnovations/laya-typed-decisions")
    ap.add_argument("--task", default="extract")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    import laya
    from sklearn.metrics import accuracy_score, roc_auc_score
    items = [json.loads(l) for l in (DATA / f"{a.task}_{a.split}.jsonl").read_text().splitlines()][: a.limit]
    agent = laya.load(a.model)
    states = [it["state"] for it in items]
    # one question per item; batch by identical question text to use predict_batch
    t0 = time.time(); probs = np.zeros(len(items))
    by_q: dict[str, list[int]] = {}
    for k, it in enumerate(items):
        by_q.setdefault(it["question"], []).append(k)
    for q, idxs in by_q.items():
        qs = {"q": {"type": "noul", "instructions": q}}
        for s in range(0, len(idxs), a.batch):
            chunk = idxs[s:s + a.batch]
            res = agent.predict_batch([states[k] for k in chunk], qs, batch_size=a.batch)
            for k, r in zip(chunk, res):
                ans = r["answers"]["q"] if isinstance(r, dict) else r.answers["q"]
                probs[k] = ans["noul"] if isinstance(ans, dict) else ans.noul
    dt = time.time() - t0
    y = np.array([int(it["label"]) for it in items])
    print(f"{a.model} on {a.task}/{a.split}: n={len(y)} pos={y.mean():.2f} | {dt/len(y)*1000:.0f} ms/item on this Mac")
    print(f"  acc={accuracy_score(y, probs > .5):.3f} auc={roc_auc_score(y, probs):.3f} ece={ece(probs, y):.3f} mean_p={probs.mean():.2f}  (majority acc={max(y.mean(), 1-y.mean()):.3f})")
    if a.task == "judge":
        qid = np.array([it.get("question_id", "") for it in items])
        for q in sorted(set(qid)):
            m = qid == q
            if m.sum() >= 5 and len(set(y[m])) > 1:
                print(f"  {q:24s} n={m.sum():3d} acc={accuracy_score(y[m], probs[m] > .5):.2f} auc={roc_auc_score(y[m], probs[m]):.2f}")
    tag = a.model.replace("/", "_")
    (OUT / f"laya_{a.task}_{a.split}_{tag}.json").write_text(json.dumps({"probs": probs.tolist(), "y": y.tolist()}))


if __name__ == "__main__":
    main()
