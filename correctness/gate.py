"""Combine the checks into one verdict under the MLPerf accuracy policy.

Quantization is allowed (MLPerf rule: purely mathematical, original weights). A candidate passes if it keeps >= 99%
of the reference's score, pooled over every task item, its outputs are >= 90% of the reference's length, and its
reported tokens match its text. Divergence and per-task flips are reported as facts, not gated.

The verdict is three-valued. An engine that is not run-to-run deterministic churns answers both ways; at the dev
tier that churn alone moves the pooled ratio by about a point. So a ratio under the line only FAILS when the net
loss (lost minus gained) is significant (one-sided McNemar p < alpha); otherwise it is INCONCLUSIVE, which never
counts as a pass and asks for the full tier. Unanswered items (the engine shed or errored after retries) also make
the verdict inconclusive: they are neither right nor wrong. A 95% interval on the ratio is reported as a fact.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .flips import _binom_tail_ge


@dataclass
class Thresholds:
    min_score_ratio: float = 0.99    # MLPerf: >= 99% of the reference score (0.999 for its high-accuracy category)
    min_length_ratio: float = 0.90   # MLPerf: output length >= 90% of reference
    alpha: float = 0.01              # a ratio under the line fails only if the net loss is significant at this level
    policy: str = "MLPerf 99%"


def pooled_accuracy(flips: dict[str, dict]) -> dict:
    """Reference vs candidate correct counts summed over all tasks; ratio = candidate / reference."""
    n = sum(f["n"] for f in flips.values())
    ref = sum(round(f["ref_accuracy"] * f["n"]) for f in flips.values())
    cand = sum(round(f["cand_accuracy"] * f["n"]) for f in flips.values())
    lost, gained = sum(f["lost"] for f in flips.values()), sum(f["gained"] for f in flips.values())
    half = 1.96 * math.sqrt(lost + gained) / ref if ref else float("nan")   # the discordant pairs carry the noise
    ratio = cand / ref if ref else float("nan")
    return {"n": n, "ref_correct": ref, "cand_correct": cand, "score_ratio": ratio,
            "score_ratio_ci95": [ratio - half, ratio + half],
            "lost": lost, "gained": gained, "p_degraded": _binom_tail_ge(lost, lost + gained)}


def evaluate(divergence: dict | None = None, flips: dict[str, dict] | None = None, length: dict | None = None,
             consistency: dict | None = None, th: Thresholds | None = None, unanswered: int = 0) -> dict:
    """Return {"verdict": pass|fail|inconclusive, "passed", "reasons", "gates": {name: {passed, reason, evidence}},
    "metrics", "thresholds"}. Gates: accuracy, length, consistency; a gate's `passed` is True, False or None
    (not run / inconclusive). `passed` is True only when every gate passed. `metrics` also carries divergence and
    per-task flips as facts."""
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
    if consistency is None:
        gate("consistency", None, "not run", None)
    else:
        gate("consistency", consistency["consistent"], "ok" if consistency["consistent"] else
             f"first-token mismatches {consistency['first_token_mismatch']}, token-count mismatches "
             f"{consistency['token_count_mismatch']}, text after EOS {consistency['text_after_eos']}", consistency)
    reasons = [f"{k}: {g['reason']}" for k, g in gates.items() if g["passed"] is not True]
    metrics = {k: g["evidence"] for k, g in gates.items() if g["evidence"] is not None}
    metrics["divergence"] = divergence if divergence is not None else {"not_run": "the engine exposes no logprobs or scoring failed"}
    metrics.update({f"flips:{t}": f for t, f in (flips or {}).items()})
    states = [g["passed"] for g in gates.values()]
    verdict = "fail" if False in states else ("pass" if all(x is True for x in states) else "inconclusive")
    return {"verdict": verdict, "passed": verdict == "pass", "reasons": reasons, "gates": gates,
            "metrics": metrics, "thresholds": asdict(th)}


def to_ledger_record(result: dict, config: dict, base: str, change: str | None = None) -> dict:
    """Shape a verdict as an `equiv` record for the loop's lab ledger (writer adds id/at/schema)."""
    return {"kind": "equiv", "config": config, "base": base, "change": change, "metrics": result["metrics"],
            "gates": result["gates"], "raw": {"thresholds": result["thresholds"], "reasons": result["reasons"]}}
