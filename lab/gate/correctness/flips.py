"""Paired flips: same questions to reference and candidate, compared answer by answer."""
from __future__ import annotations


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
    }
