"""Cheap reference models for the System-One tasks: TF-IDF + logistic regression on the Laya datasets.
If a bag-of-words model already matches Haiku/Sonnet, a learned judge adds little for that task."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, roc_auc_score

DATA = Path(__file__).resolve().parent.parent / "data" / "laya"


def load(name):
    X, y, q = [], [], []
    for line in (DATA / f"{name}.jsonl").read_text().splitlines():
        d = json.loads(line)
        X.append(d["question"] + "\n" + d["state"]); y.append(int(d["label"])); q.append(d.get("question_id", ""))
    return X, np.array(y), q


def ece(p, y, bins=10):
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any()))


def run(task):
    Xtr, ytr, _ = load(f"{task}_train"); Xte, yte, qte = load(f"{task}_test")
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200000, sublinear_tf=True, token_pattern=r"[A-Za-z_][A-Za-z0-9_./-]*|\d+(?:\.\d+)?")
    A = vec.fit_transform(Xtr); B = vec.transform(Xte)
    m = LogisticRegression(C=4.0, max_iter=3000, class_weight="balanced").fit(A, ytr)
    p = m.predict_proba(B)[:, 1]
    print(f"== {task}: train n={len(ytr)} pos={ytr.mean():.2f} | test n={len(yte)} pos={yte.mean():.2f}")
    print(f"   acc={accuracy_score(yte, p > .5):.3f} f1={f1_score(yte, p > .5):.3f} auc={roc_auc_score(yte, p):.3f} brier={brier_score_loss(yte, p):.3f} ece={ece(p, yte):.3f}  (majority acc={max(yte.mean(), 1-yte.mean()):.3f})")
    if task == "judge":
        for qid in sorted(set(qte)):
            mask = np.array([x == qid for x in qte])
            if mask.sum() >= 5 and len(set(yte[mask])) > 1:
                print(f"   {qid:24s} n={mask.sum():3d} acc={accuracy_score(yte[mask], p[mask] > .5):.2f} auc={roc_auc_score(yte[mask], p[mask]):.2f}")


if __name__ == "__main__":
    for t in ("extract", "judge"):
        run(t)
