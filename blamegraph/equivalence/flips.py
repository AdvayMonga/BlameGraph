"""Paired flip test: same questions to reference and candidate, compared answer by answer.

Average scores hide behaviour changes (quantization can hold accuracy within 1-2% while flipping over 10% of
answers), and with a few hundred questions an average moves by about two points from chance alone. Pairing fixes
both: only discordant items carry information, and McNemar's test asks whether right->wrong flips outnumber
wrong->right flips by more than chance.
"""
from __future__ import annotations

import math


def _binom_tail_ge(k: int, n: int) -> float:
    """P(X >= k) for X ~ Binomial(n, 0.5)."""
    if n == 0:
        return 1.0
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n


def flip_test(ref_correct: list[bool], cand_correct: list[bool]) -> dict:
    """`ref_correct[i]` / `cand_correct[i]`: whether each server answered item i correctly (same item order)."""
    if len(ref_correct) != len(cand_correct):
        raise ValueError("answer lists must be paired item by item")
    n = len(ref_correct)
    lost = sum(1 for r, c in zip(ref_correct, cand_correct) if r and not c)    # right -> wrong
    gained = sum(1 for r, c in zip(ref_correct, cand_correct) if c and not r)  # wrong -> right
    ref_acc = sum(ref_correct) / n if n else float("nan")
    cand_acc = sum(cand_correct) / n if n else float("nan")
    return {
        "n": n, "lost": lost, "gained": gained,
        "flip_rate": (lost + gained) / n if n else float("nan"),
        "net_loss_rate": (lost - gained) / n if n else float("nan"),
        "ref_accuracy": ref_acc, "cand_accuracy": cand_acc,
        "score_ratio": cand_acc / ref_acc if ref_acc else float("nan"),
        # one-sided exact McNemar: probability of at least this many losses among discordant pairs if no degradation
        "p_degraded": _binom_tail_ge(lost, lost + gained),
    }
