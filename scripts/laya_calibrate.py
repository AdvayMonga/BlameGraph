"""Fit one temperature on the val split's Laya probabilities and report test ECE/acc before and after.
Usage: laya_calibrate.py --tag <model tag as in laya_eval output filenames> --task extract"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent.parent / "data" / "derived"


def ece(p, y, bins=10):
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any()))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def nll(z, y):
    p = 1 / (1 + np.exp(-z))
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--task", default="extract"); a = ap.parse_args()
    val = json.loads((OUT / f"laya_{a.task}_val_{a.tag}.json").read_text()); test = json.loads((OUT / f"laya_{a.task}_test_{a.tag}.json").read_text())
    zv, yv = logit(np.array(val["probs"])), np.array(val["y"]); zt, yt = logit(np.array(test["probs"])), np.array(test["y"])
    # scalar temperature + bias, grid search on val NLL (small, robust)
    best = min(((nll(zv / T + b, yv), T, b) for T in np.linspace(0.2, 5, 97) for b in np.linspace(-3, 3, 61)))
    _, T, b = best
    pt0 = 1 / (1 + np.exp(-zt)); pt1 = 1 / (1 + np.exp(-(zt / T + b)))
    print(f"{a.tag} {a.task}: fitted T={T:.2f} bias={b:.2f} on val (n={len(yv)})")
    print(f"  test before: acc={np.mean((pt0 > .5) == yt):.3f} ece={ece(pt0, yt):.3f}   after: acc={np.mean((pt1 > .5) == yt):.3f} ece={ece(pt1, yt):.3f}")


if __name__ == "__main__":
    main()
