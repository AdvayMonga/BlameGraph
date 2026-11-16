"""Combine the checks into one pass/fail verdict under the MLPerf accuracy policy."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class Thresholds:
    min_score_ratio: float = 0.99    # candidate score / reference score, each task
    min_length_ratio: float = 0.90   # candidate output length / reference
    max_length_ratio: float = 1.10
    policy: str = "MLPerf 99%"


def task_accuracy(flips: dict[str, dict]) -> dict:
    """Per task: reference and candidate correct counts and their ratio (candidate / reference)."""
    out = {}
    for task, f in flips.items():
        ref, cand = round(f["ref_accuracy"] * f["n"]), round(f["cand_accuracy"] * f["n"])
        out[task] = {"n": f["n"], "ref_correct": ref, "cand_correct": cand,
                     "score_ratio": cand / ref if ref else float("nan")}
    return out


def evaluate(divergence: dict | None = None, flips: dict[str, dict] | None = None, length: dict | None = None,
             consistency: dict | None = None, th: Thresholds | None = None, unanswered: int = 0) -> dict:
    """Judge the gates (accuracy, length, consistency when given); `passed` is True only if every gate passed.
    Returns {verdict: pass|fail, passed, reasons, gates: {name: {passed, reason, evidence}}, metrics, thresholds}."""
    th = th or Thresholds()
    gates: dict[str, dict] = {}

    def gate(name, ok, reason, evidence):
        gates[name] = {"passed": ok, "reason": reason, "evidence": evidence}

    if not flips:
        gate("accuracy", None, "not run", None)
    else:
        tasks = task_accuracy(flips)
        acc = {"tasks": tasks, "unanswered": unanswered}
        low = [f"{t} {a['score_ratio']:.4f} ({a['cand_correct']} vs {a['ref_correct']} correct of {a['n']})"
               for t, a in tasks.items() if not a["score_ratio"] >= th.min_score_ratio]
        if unanswered:
            gate("accuracy", False, f"{unanswered} item(s) unanswered (shed or errored after retries)", acc)
        elif low:
            gate("accuracy", False, f"task score ratio < {th.min_score_ratio}: " + "; ".join(low), acc)
        else:
            gate("accuracy", True, "ok", acc)
    if length is None:
        gate("length", None, "not run", None)
    else:
        ok = th.min_length_ratio <= length["ratio"] <= th.max_length_ratio
        gate("length", ok, "ok" if ok else f"output length ratio {length['ratio']:.3f} outside "
             f"{th.min_length_ratio}..{th.max_length_ratio}", length)
    if consistency is not None:
        gate("consistency", consistency["consistent"], "ok" if consistency["consistent"] else
             f"first-token mismatches {consistency['first_token_mismatch']}, token-count mismatches "
             f"{consistency['token_count_mismatch']}, text after EOS {consistency['text_after_eos']}", consistency)
    reasons = [f"{k}: {g['reason']}" for k, g in gates.items() if g["passed"] is not True]
    metrics = {k: g["evidence"] for k, g in gates.items() if g["evidence"] is not None}
    if divergence is not None:
        metrics["divergence"] = divergence
    metrics.update({f"flips:{t}": f for t, f in (flips or {}).items()})
    verdict = "pass" if all(g["passed"] is True for g in gates.values()) else "fail"
    return {"verdict": verdict, "passed": verdict == "pass", "reasons": reasons, "gates": gates,
            "metrics": metrics, "thresholds": asdict(th)}


def to_ledger_record(result: dict, config: dict, base: str, change: str | None = None) -> dict:
    """Shape a verdict as an `equiv` ledger record (the writer adds id/at/schema)."""
    return {"kind": "equiv", "config": config, "base": base, "change": change, "metrics": result["metrics"],
            "gates": result["gates"], "raw": {"thresholds": result["thresholds"], "reasons": result["reasons"]}}
