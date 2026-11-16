"""Combine the checks into one pass/fail/inconclusive verdict under the MLPerf accuracy policy."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .flips import _binom_tail_ge


@dataclass
class Thresholds:
    min_score_ratio: float = 0.99    # candidate score / reference score
    min_length_ratio: float = 0.90   # candidate output length / reference
    alpha: float = 0.01              # significance level a net loss must reach to fail
    policy: str = "MLPerf 99%"


def pooled_accuracy(flips: dict[str, dict]) -> dict:
    """Reference vs candidate correct counts summed over all tasks; ratio = candidate / reference."""
    n = sum(f["n"] for f in flips.values())
    ref = sum(round(f["ref_accuracy"] * f["n"]) for f in flips.values())
    cand = sum(round(f["cand_accuracy"] * f["n"]) for f in flips.values())
    lost, gained = sum(f["lost"] for f in flips.values()), sum(f["gained"] for f in flips.values())
    half = 1.96 * math.sqrt(lost + gained) / ref if ref else float("nan")
    ratio = cand / ref if ref else float("nan")
    return {"n": n, "ref_correct": ref, "cand_correct": cand, "score_ratio": ratio,
            "score_ratio_ci95": [ratio - half, ratio + half],
            "lost": lost, "gained": gained, "p_degraded": _binom_tail_ge(lost, lost + gained)}


def evaluate(divergence: dict | None = None, flips: dict[str, dict] | None = None, length: dict | None = None,
             consistency: dict | None = None, th: Thresholds | None = None, unanswered: int = 0) -> dict:
    """Judge the gates (accuracy, length, consistency when given); `passed` is True only if every gate passed.
    Returns {verdict: pass|fail|inconclusive, passed, reasons, gates: {name: {passed, reason, evidence}}, metrics,
    thresholds}."""
    th = th or Thresholds()
    gates: dict[str, dict] = {}

    def gate(name, ok, reason, evidence):
        gates[name] = {"passed": ok, "reason": reason, "evidence": evidence}

    if not flips:
        gate("accuracy", None, "not run", None)
    else:
        acc = pooled_accuracy(flips)
        acc["unanswered"] = unanswered
        r, lo, hi = acc["score_ratio"], *acc["score_ratio_ci95"]
        where = f"pooled score ratio {r:.4f} (95% CI {lo:.4f}..{hi:.4f}; {acc['cand_correct']} vs {acc['ref_correct']} correct of {acc['n']})"
        if not acc["ref_correct"]:
            gate("accuracy", None, f"inconclusive: the reference answered 0 of {acc['n']} items correctly; "
                 "the task set says nothing about this model", acc)
        elif unanswered:
            gate("accuracy", None, f"inconclusive: {unanswered} item(s) unanswered (shed or errored after retries); "
                 f"{where} over the answered ones", acc)
        elif r >= th.min_score_ratio:
            gate("accuracy", True, "ok", acc)
        elif acc["p_degraded"] < th.alpha:
            gate("accuracy", False, f"{where} < {th.min_score_ratio}; net loss {acc['lost']} - {acc['gained']} is "
                 f"significant (p = {acc['p_degraded']:.2g})", acc)
        else:
            gate("accuracy", None, f"inconclusive: {where} < {th.min_score_ratio} but the net loss "
                 f"{acc['lost']} - {acc['gained']} is within run-to-run churn (p = {acc['p_degraded']:.2g}); "
                 f"run the full tier", acc)
    if length is None:
        gate("length", None, "not run", None)
    else:
        ok = length["ratio"] >= th.min_length_ratio
        gate("length", ok, "ok" if ok else f"output length ratio {length['ratio']:.3f} < {th.min_length_ratio}", length)
    if consistency is not None:
        gate("consistency", consistency["consistent"], "ok" if consistency["consistent"] else
             f"first-token mismatches {consistency['first_token_mismatch']}, token-count mismatches "
             f"{consistency['token_count_mismatch']}, text after EOS {consistency['text_after_eos']}", consistency)
    reasons = [f"{k}: {g['reason']}" for k, g in gates.items() if g["passed"] is not True]
    metrics = {k: g["evidence"] for k, g in gates.items() if g["evidence"] is not None}
    if divergence is not None:
        metrics["divergence"] = divergence
    metrics.update({f"flips:{t}": f for t, f in (flips or {}).items()})
    states = [g["passed"] for g in gates.values()]
    verdict = "fail" if False in states else ("pass" if all(x is True for x in states) else "inconclusive")
    return {"verdict": verdict, "passed": verdict == "pass", "reasons": reasons, "gates": gates,
            "metrics": metrics, "thresholds": asdict(th)}


def to_ledger_record(result: dict, config: dict, base: str, change: str | None = None) -> dict:
    """Shape a verdict as an `equiv` ledger record (the writer adds id/at/schema)."""
    return {"kind": "equiv", "config": config, "base": base, "change": change, "metrics": result["metrics"],
            "gates": result["gates"], "raw": {"thresholds": result["thresholds"], "reasons": result["reasons"]}}
