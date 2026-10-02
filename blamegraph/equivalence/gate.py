"""Combine the checks into one verdict, and calibrate its thresholds from known-good and known-bad candidates.

Quantization is allowed (MLPerf rule: purely mathematical, original weights). So the divergence and flip-rate
limits are not "zero": they are calibrated so acceptable quantizations (e.g. FP8, INT8) pass and a deliberately
degraded one fails. Task-score degradation fails only when it is both statistically real and material.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class Thresholds:
    kl_mean: float = 0.02            # placeholder until calibrated on the target model and hardware
    kl_p99: float = 0.5
    flip_rate: float = 0.10
    min_score_ratio: float = 0.99    # MLPerf: >= 99% of the reference task score
    alpha: float = 0.01              # McNemar significance for "degraded"
    min_length_ratio: float = 0.90   # MLPerf: output length >= 90% of reference
    calibrated_from: str = "defaults (uncalibrated)"


def evaluate(divergence: dict | None = None, flips: dict[str, dict] | None = None, length: dict | None = None,
             consistency: dict | None = None, th: Thresholds | None = None) -> dict:
    """Return {"passed", "reasons", "gates": {name: {passed, reason, evidence}}, "metrics", "thresholds"}.
    Any check left as None is reported as not run; the gate passes only if every check ran and passed."""
    th = th or Thresholds()
    gates: dict[str, dict] = {}

    def gate(name, ok, reason, evidence):
        gates[name] = {"passed": ok, "reason": reason, "evidence": evidence}

    if divergence is None:
        gate("divergence", None, "not run", None)
    else:
        bad = [f"kl_mean {divergence['kl_mean']:.4g} > {th.kl_mean:.4g}"] if divergence["kl_mean"] > th.kl_mean else []
        bad += [f"kl_p99 {divergence['kl_p99']:.4g} > {th.kl_p99:.4g}"] if divergence["kl_p99"] > th.kl_p99 else []
        gate("divergence", not bad, "; ".join(bad) or "within calibrated limits", divergence)
    if not flips:
        gate("flips", None, "not run", None)
    for task, f in (flips or {}).items():
        bad = []
        if f["flip_rate"] > th.flip_rate:
            bad.append(f"flip_rate {f['flip_rate']:.3f} > {th.flip_rate:.3f}")
        if f["p_degraded"] < th.alpha and f["score_ratio"] < th.min_score_ratio:
            bad.append(f"significant degradation: {f['lost']} lost vs {f['gained']} gained (p={f['p_degraded']:.2g}), "
                       f"score ratio {f['score_ratio']:.3f} < {th.min_score_ratio}")
        gate(f"flips:{task}", not bad, "; ".join(bad) or "no material, significant degradation", f)
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
    return {"passed": all(g["passed"] is True for g in gates.values()), "reasons": reasons, "gates": gates,
            "metrics": {k: g["evidence"] for k, g in gates.items() if g["evidence"] is not None}, "thresholds": asdict(th)}


def to_ledger_record(result: dict, config: dict, base: str, change: str | None = None) -> dict:
    """Shape a verdict as an `equiv` record for the loop's lab ledger (writer adds id/at/schema)."""
    return {"kind": "equiv", "config": config, "base": base, "change": change, "metrics": result["metrics"],
            "gates": result["gates"], "raw": {"thresholds": result["thresholds"], "reasons": result["reasons"]}}


def calibrate(good: list[dict], bad: list[dict], margin: float = 1.5, label: str = "") -> Thresholds:
    """Thresholds from divergence summaries / flip results of known-good candidates (must pass) and known-bad
    candidates (must fail). Each item: {"divergence": summary, "flips": {task: flip_test result}}.
    Raises if the good and bad sets cannot be separated (the gate would be meaningless)."""
    def worst(items, path):
        vals = [v for it in items for v in path(it)]
        return max(vals) if vals else None
    kl_mean = worst(good, lambda it: [it["divergence"]["kl_mean"]] if it.get("divergence") else [])
    kl_p99 = worst(good, lambda it: [it["divergence"]["kl_p99"]] if it.get("divergence") else [])
    flip = worst(good, lambda it: [f["flip_rate"] for f in (it.get("flips") or {}).values()])
    d = Thresholds()
    th = Thresholds(kl_mean=kl_mean * margin if kl_mean is not None else d.kl_mean,
                    kl_p99=kl_p99 * margin if kl_p99 is not None else d.kl_p99,
                    flip_rate=flip * margin if flip is not None else d.flip_rate,
                    calibrated_from=label or f"{len(good)} good / {len(bad)} bad")
    for it in bad:
        gates = evaluate(it.get("divergence"), it.get("flips"), th=th)["gates"]
        if not any(g["passed"] is False for k, g in gates.items() if k == "divergence" or k.startswith("flips:")):
            raise ValueError("a known-bad candidate passes the calibrated thresholds; the gate cannot separate good from bad")
    return th
