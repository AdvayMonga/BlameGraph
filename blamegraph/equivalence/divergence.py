"""Teacher-forced divergence between reference and candidate.

Both servers score the *same* token sequence (the reference's own output), so one early token difference cannot
cascade. Each position carries the logprob of the actual next token and the top-k alternatives. With full logits
(local scoring) KL is exact; with top-k (server APIs) it is computed over the reference's top-k support, with
tokens missing from the candidate's top-k assigned the candidate's smallest listed logprob (a conservative bound).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class Position:
    token: int | str                          # the reference's actual token at this position
    logprob: float                            # logprob the scorer assigned to that token
    top: dict[int | str, float] = field(default_factory=dict)   # top-k alternatives (token -> logprob)


def _kl_topk(ref: Position, cand: Position) -> float:
    support = dict(ref.top) or {ref.token: ref.logprob}
    support.setdefault(ref.token, ref.logprob)
    floor = min(cand.top.values()) if cand.top else cand.logprob
    cand_lp = {**cand.top, cand.token: cand.logprob}
    # renormalize both over the shared support so the bound is a proper KL between two distributions
    toks = list(support)
    p = [math.exp(support[t]) for t in toks]
    q = [math.exp(cand_lp.get(t, floor)) for t in toks]
    zp, zq = sum(p), sum(q)
    return sum(pi / zp * (math.log(pi / zp) - math.log(qi / zq)) for pi, qi in zip(p, q) if pi > 0)


def kl_full(ref_logits: list[float], cand_logits: list[float]) -> float:
    """Exact KL(ref || cand) from full-vocabulary logits at one position."""
    def log_softmax(xs):
        m = max(xs); s = math.log(sum(math.exp(x - m) for x in xs)) + m
        return [x - s for x in xs]
    lp, lq = log_softmax(ref_logits), log_softmax(cand_logits)
    return sum(math.exp(a) * (a - b) for a, b in zip(lp, lq))


def compare_positions(ref: list[Position], cand: list[Position]) -> list[dict]:
    """Per-position comparison of two scorings of the same token sequence."""
    if len(ref) != len(cand):
        raise ValueError(f"scorings cover different lengths ({len(ref)} vs {len(cand)}); they must score the same tokens")
    out = []
    for r, c in zip(ref, cand):
        if r.token != c.token:
            raise ValueError(f"token mismatch at a position ({r.token!r} vs {c.token!r}); align on the reference tokens")
        r_top1 = max(r.top, key=r.top.get) if r.top else r.token
        c_top1 = max(c.top, key=c.top.get) if c.top else c.token
        out.append({"kl": _kl_topk(r, c), "delta_logprob": c.logprob - r.logprob, "top1_agree": r_top1 == c_top1})
    return out


def _quantile(xs: list[float], q: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, math.ceil(q * len(s)) - 1))] if s else float("nan")


def divergence_summary(per_sequence: list[list[dict]]) -> dict:
    """Aggregate over sequences: mean and tail KL per token, worst sequence, top-1 agreement, ref-token logprob shift."""
    flat = [p for seq in per_sequence for p in seq]
    if not flat:
        return {"n_tokens": 0}
    kls = [p["kl"] for p in flat]
    seq_means = [sum(p["kl"] for p in seq) / len(seq) for seq in per_sequence if seq]
    return {
        "n_sequences": len(per_sequence), "n_tokens": len(flat),
        "kl_mean": sum(kls) / len(kls), "kl_p99": _quantile(kls, 0.99), "kl_seq_max": max(seq_means),
        "top1_agree": sum(p["top1_agree"] for p in flat) / len(flat),
        "delta_logprob_mean": sum(p["delta_logprob"] for p in flat) / len(flat),
        "delta_logprob_abs_p99": _quantile([abs(p["delta_logprob"]) for p in flat], 0.99),
    }
