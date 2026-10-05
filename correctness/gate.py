"""Combine the checks into one verdict under the MLPerf accuracy policy.

Quantization is allowed (MLPerf rule: purely mathematical, original weights). A candidate passes if it keeps >= 99%
of the reference's score, pooled over every task item, its outputs are >= 90% of the reference's length, and its
reported tokens match its text. Divergence and per-task flips are reported as facts, not gated.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from .flips import _binom_tail_ge


@dataclass
class Thresholds:
    min_score_ratio: float = 0.99    # MLPerf: >= 99% of the reference score (0.999 for its high-accuracy category)
    min_length_ratio: float = 0.90   # MLPerf: output length >= 90% of reference
    policy: str = "MLPerf 99%"


def pooled_accuracy(flips: dict[str, dict]) -> dict:
    """Reference vs candidate correct counts summed over all tasks; ratio = candidate / reference."""
    n = sum(f["n"] for f in flips.values())
    ref = sum(round(f["ref_accuracy"] * f["n"]) for f in flips.values())
    cand = sum(round(f["cand_accuracy"] * f["n"]) for f in flips.values())
    lost, gained = sum(f["lost"] for f in flips.values()), sum(f["gained"] for f in flips.values())
    return {"n": n, "ref_correct": ref, "cand_correct": cand, "score_ratio": cand / ref if ref else float("nan"),
            "lost": lost, "gained": gained, "p_degraded": _binom_tail_ge(lost, lost + gained)}


def evaluate(divergence: dict | None = None, flips: dict[str, dict] | None = None, length: dict | None = None,
             consistency: dict | None = None, th: Thresholds | None = None) -> dict:
    """Return {"passed", "reasons", "gates": {name: {passed, reason, evidence}}, "metrics", "thresholds"}.
    Gates: accuracy, length, consistency; each left as None is "not run" and fails the verdict. `metrics` also
    carries divergence and per-task flips as facts."""
    th = th or Thresholds()
    gates: dict[str, dict] = {}

    def gate(name, ok, reason, evidence):
        gates[name] = {"passed": ok, "reason": reason, "evidence": evidence}

    if not flips:
        gate("accuracy", None, "not run", None)
    else:
        acc = pooled_accuracy(flips)
        ok = acc["score_ratio"] >= th.min_score_ratio
        gate("accuracy", ok, "ok" if ok else f"pooled score ratio {acc['score_ratio']:.4f} < {th.min_score_ratio} "
             f"({acc['cand_correct']} vs {acc['ref_correct']} correct of {acc['n']})", acc)
    if length is None:
        gate("length", None, "not run", None)
    else:
        ok = length["ratio"] >= th.min_length_ratio
        gate("length", ok, "ok" if ok else f"output length ratio {length['ratio']:.3f} < {th.min_length_ratio}", length)
    if consistency is None:
        gate("consistency", None, "not run", None)
    else:
        gate("consistency", consistency["consistent"], "ok" if consistency["consistent"] else
             f"first-token mismatches {consistency['first_token_mismatch']}, token-count mismatches "
             f"{consistency['token_count_mismatch']}, text after EOS {consistency['text_after_eos']}", consistency)
    reasons = [f"{k}: {g['reason']}" for k, g in gates.items() if g["passed"] is not True]
    metrics = {k: g["evidence"] for k, g in gates.items() if g["evidence"] is not None}
    if divergence is not None:
        metrics["divergence"] = divergence
    metrics.update({f"flips:{t}": f for t, f in (flips or {}).items()})
    return {"passed": all(g["passed"] is True for g in gates.values()), "reasons": reasons, "gates": gates,
            "metrics": metrics, "thresholds": asdict(th)}


def to_ledger_record(result: dict, config: dict, base: str, change: str | None = None) -> dict:
    """Shape a verdict as an `equiv` record for the loop's lab ledger (writer adds id/at/schema)."""
    return {"kind": "equiv", "config": config, "base": base, "change": change, "metrics": result["metrics"],
            "gates": result["gates"], "raw": {"thresholds": result["thresholds"], "reasons": result["reasons"]}}
