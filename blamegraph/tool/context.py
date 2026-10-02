"""Context packer: the facts a loop may read. Ledger summary, measurement table with uncertainty, validator result,
and nearby prior measurements from the landscape. No scores, no judgments, no recommendations."""
from __future__ import annotations

import json
from pathlib import Path

from .landscape import Landscape, primary
from .ledger import Ledger
from .validate import validate


def pack(ledger: Ledger, config_path: Path, scenario: str, grader_paths: list[Path], landscape: Landscape | None = None,
         baseline_metric: float | None = None, max_rows: int = 60) -> dict:
    content = {c["hash"]: c for c in ledger.configs()}
    rows = []
    for m in ledger.measurements():
        v = primary(m.get("metrics"), scenario)
        rows.append({
            "id": m["id"], "minute": m["minute"], "config": m.get("live_hash"), "config_file_at_the_time": m.get("config_hash"),
            "server_state": "stale (file changed after server start)" if m.get("stale") else m.get("cache_state"),
            "mode": m["mode"], "standard_invocation": m["standard"], "flags": m.get("flags"),
            "metric": v, "speedup_vs_baseline": (v / baseline_metric) if (v and baseline_metric) else None,
            "failure_rate": m.get("failure_rate"), "quality_pass": m.get("quality_pass"), "error": m.get("error"),
        })
    # repeatability per config from this session's own standard full measurements
    by_cfg: dict[str, list[float]] = {}
    for r in rows:
        if r["metric"] and r["standard_invocation"] and r["mode"] == "full" and not str(r["server_state"]).startswith("stale"):
            by_cfg.setdefault(r["config"], []).append(r["metric"])
    repeat = {}
    for h, vals in by_cfg.items():
        if len(vals) >= 2:
            mean = sum(vals) / len(vals); sd = (sum((x - mean) ** 2 for x in vals) / len(vals)) ** 0.5
            repeat[h] = {"n": len(vals), "mean": mean, "cv": sd / mean if mean else None}
    verdict = validate(ledger, config_path, grader_paths)
    shipped = verdict.facts.get("shipped_config")
    ctx = {
        "tool_version": ledger.events[0].get("tool_version") if ledger.events else None,
        "session": {"minutes_elapsed": ledger.events[-1]["minute"] if ledger.events else 0,
                     "configs_seen": [{"hash": h, "minute": c["minute"], "size": c["size"]} for h, c in content.items()],
                     "current_config": ledger.current_config_hash(), "shipped_if_submitted_now": shipped},
        "measurements": rows[-max_rows:],
        "repeatability_this_session": repeat,
        "validation_if_submitted_now": verdict.to_dict(),
    }
    if landscape is not None and config_path.exists():
        ctx["prior_measurements_near_current_config"] = landscape.near(scenario, config_path.read_text())[:20]
        ctx["benchmark_noise_from_prior_sessions"] = landscape.noise(scenario)
    return ctx


def pack_text(ctx: dict) -> str:
    """Compact plain-text rendering for a prompt. Same facts as pack(); nothing added."""
    L = [f"minutes elapsed: {ctx['session']['minutes_elapsed']:.0f}; configs seen: {len(ctx['session']['configs_seen'])}; current config: {ctx['session']['current_config']}"]
    L.append("measurements (id, minute, config, server, mode, standard, metric, speedup, failure_rate, quality):")
    for r in ctx["measurements"]:
        L.append(f"  {r['id']} m{r['minute']:.0f} {r['config']} {r['server_state']} {r['mode']} std={r['standard_invocation']} "
                 f"metric={None if r['metric'] is None else round(r['metric'], 4)} speedup={None if r['speedup_vs_baseline'] is None else round(r['speedup_vs_baseline'], 2)} "
                 f"fail={r['failure_rate']} quality={r['quality_pass']}{' error=' + r['error'] if r['error'] else ''}")
    if ctx["repeatability_this_session"]:
        L.append("repeatability (same config, standard full, fresh server): " + "; ".join(f"{h}: n={d['n']} cv={d['cv']:.3f}" for h, d in ctx["repeatability_this_session"].items()))
    v = ctx["validation_if_submitted_now"]
    L.append(f"validation if submitted now: {'valid' if v['valid'] else 'INVALID'}" + ("" if v["valid"] else " — " + "; ".join(v["reasons"])))
    if ctx.get("prior_measurements_near_current_config"):
        L.append("prior sessions measured near this config (distance = knobs that differ):")
        for p in ctx["prior_measurements_near_current_config"]:
            L.append(f"  d={p['distance']} n={p['n']} mean_metric={p['mean']:.4f} cv={None if p['cv'] is None else round(p['cv'], 3)} knobs_diff={ {k: v for k, v in p['knobs'].items()} if p['distance'] else 'same config'}")
    if ctx.get("benchmark_noise_from_prior_sessions", {}).get("cv_median") is not None:
        n = ctx["benchmark_noise_from_prior_sessions"]
        L.append(f"benchmark repeatability across prior sessions: median cv {n['cv_median']:.3f} over {n['n_repeated_configs']} repeated configs")
    return "\n".join(L)
