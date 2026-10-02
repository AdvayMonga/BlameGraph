"""Compare System-One judges on the held-out test sets: zero-shot Laya, fine-tuned Laya, TF-IDF; teacher = Haiku/Sonnet labels."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
OUT = Path(__file__).resolve().parent.parent / "data" / "derived"


def ece(p, y, bins=10):
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any()))


def metrics(p, y):
    from sklearn.metrics import roc_auc_score
    p, y = np.asarray(p), np.asarray(y)
    return dict(n=len(y), acc=float(np.mean((p > .5) == y)), auc=float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan"), ece=ece(p, y))


def main():
    rows = []
    for task in ("extract", "judge"):
        for f in sorted(OUT.glob(f"laya_{task}_test_*.json")):
            d = json.loads(f.read_text())
            tag = f.stem.replace(f"laya_{task}_test_", "")
            rows.append((task, tag, metrics(d["probs"], d["y"])))
    print(f"{'task':8s} {'model':55s} {'n':>5s} {'acc':>6s} {'auc':>6s} {'ece':>6s}")
    for task, tag, m in rows:
        print(f"{task:8s} {tag:55s} {m['n']:5d} {m['acc']:6.3f} {m['auc']:6.3f} {m['ece']:6.3f}")
    print("\nReference (same splits): TF-IDF+LR extract acc 0.943 auc 0.987 | judge acc 0.742 auc 0.814 (scripts/laya_baseline.py)")
    print("Teachers: Haiku 4.5 (extract gate, $15.61 for 8,876 steps), Sonnet 5.5 (judge, $2.74 for 571 windows). Laya: $0, ~0.2 s/item on an M-series Mac (MPS).")


if __name__ == "__main__":
    main()
