"""The comparison's tables from a compare/run.py output dir: per regime and precision every engine's number (median over
repeats, its run-to-run spread, where its search failed), the winner, and whether the win is clear: beyond every
contender's spread and, for a load search, at a load the runner-up failed. Then the saturated curve (each concurrency's
output tokens/s per GPU against TPOT, with $ per million output tokens when an hourly price is given), equiv's verdicts,
cold start and the client check.

    python -m compare.report DIR [--gpu-usd-per-hour 3.5] [--out report.md]
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def _load(d: Path) -> dict:
    """{(engine, precision): {"regimes": {name: [result per repeat]}, "equiv": ..., "cold": ..., "launch": ...}}"""
    out = {}
    for cfg in sorted(p for p in d.iterdir() if p.is_dir() and "-" in p.name):
        engine, precision = cfg.name.rsplit("-", 1)
        regs: dict[str, list] = {}
        for f in sorted(cfg.glob("regimes-*.json")):
            for r in json.loads(f.read_text())["results"]:
                regs.setdefault(r["regime"], []).append(r)
        read = lambda name: json.loads((cfg / name).read_text()) if (cfg / name).exists() else None
        gpus = {json.loads(f.read_text()).get("gpu_uuid") for f in cfg.glob("facts-*.json")}
        out[(engine, precision)] = {"regimes": regs, "equiv": read("equiv.json"), "cold": read("cold.json"),
                                    "launch": read("launch.json"), "gpus": gpus - {None, ""}}
    return out


def summarize(runs: list[dict]) -> dict:
    """One configuration's repeats of one regime: median value, spread % (max - min over the median), first fail."""
    vals = [r["value"] for r in runs if r.get("valid") and r.get("value") is not None]
    med = statistics.median(vals) if vals else None
    spread = (max(vals) - min(vals)) / med * 100 if len(vals) > 1 and med else None
    fails = [r.get("fails_at") for r in runs if r.get("fails_at") is not None]
    return {"value": med, "n": len(vals), "of": len(runs), "spread_pct": spread, "fails_at": min(fails) if fails else None,
            "stopped": sorted({r.get("stopped") for r in runs if r.get("stopped")}), "better": runs[0]["better"],
            "invalid": [x for r in runs if not r.get("valid") for x in r.get("invalid_reasons") or []][:2]}


def verdict(rows: dict[str, dict], tol_pct: float) -> dict:
    """The winner among `rows` ({engine: summary}) and whether its lead is clear of noise and of the search's bracket."""
    ok = {e: s for e, s in rows.items() if s["value"] is not None}
    if not ok:
        return {"winner": None, "clear": False, "why": "no valid result"}
    hi = next(iter(ok.values()))["better"] == "higher"
    order = sorted(ok, key=lambda e: ok[e]["value"], reverse=hi)
    if len(order) == 1:
        return {"winner": order[0], "clear": False, "why": "no other valid result"}
    w, r = ok[order[0]], ok[order[1]]
    gap = abs(w["value"] - r["value"]) / r["value"] * 100
    noise = max(tol_pct, w["spread_pct"] or 0, r["spread_pct"] or 0)
    why = []
    if w["n"] < 2 or r["n"] < 2:
        why.append("fewer than 2 repeats: spread unknown")
    if gap <= noise:
        why.append(f"lead {gap:.1f}% within noise {noise:.1f}%")
    if hi and r["fails_at"] is not None and w["value"] < r["fails_at"]:
        why.append(f"inside the runner-up's bracket ({r['value']:.4g} passed, {r['fails_at']:.4g} failed)")
    if "max_concurrency" in w["stopped"] or "probe_budget" in w["stopped"]:
        why.append(f"the winner's search stopped on {w['stopped']}, not a failure")
    return {"winner": order[0], "runner_up": order[1], "lead_pct": gap, "clear": not why, "why": "; ".join(why)}


def _fmt(x, digits=4):
    return "-" if x is None else f"{x:.{digits}g}"


def render(d: Path, gpu_usd_per_hour: float | None = None, tol_pct: float = 2.0) -> str:
    data = _load(d)
    engines = sorted({e for e, _ in data})
    precisions = sorted({p for _, p in data})
    regimes = sorted({n for v in data.values() for n in v["regimes"]})
    lines = [f"# Engine comparison ({d.name})", ""]
    for prec in precisions:
        lines += [f"## {prec}", "", "| regime | " + " | ".join(engines) + " | winner | clear? |",
                  "|---|" + "---|" * (len(engines) + 2)]
        for name in regimes:
            rows = {e: summarize(data[(e, prec)]["regimes"][name]) for e in engines
                    if (e, prec) in data and name in data[(e, prec)]["regimes"]}
            cell = lambda s: ("invalid: " + "; ".join(s["invalid"])[:60] if s["value"] is None else
                              f"{_fmt(s['value'])}" + (f" ±{s['spread_pct']:.1f}%" if s["spread_pct"] is not None else "")
                              + (f" (fails {_fmt(s['fails_at'])})" if s["fails_at"] else "") + f" [{s['n']}/{s['of']}]")
            v = verdict(rows, tol_pct)
            lines.append(f"| {name} | " + " | ".join(cell(rows[e]) if e in rows else "-" for e in engines) +
                         f" | {v['winner'] or '-'}" + (f" +{v['lead_pct']:.1f}%" if v.get("lead_pct") is not None else "") +
                         f" | {'yes' if v['clear'] else 'no: ' + v['why']} |")
        cold = {e: (data[(e, prec)]["cold"] or {}) for e in engines if (e, prec) in data}
        lines += ["| cold_start (s) | " + " | ".join(_fmt((cold.get(e) or {}).get("value")) for e in engines) +
                  " | | |", ""]
    lines += ["## Saturated curve: output tokens/s per GPU against TPOT", "",
              "| engine | precision | concurrency | output tok/s | TPOT p50 ms | TPOT p99 ms | meets limits |"
              + (" $/M output tokens |" if gpu_usd_per_hour else ""),
              "|---|---|---|---|---|---|---|" + ("---|" if gpu_usd_per_hour else "")]
    for (e, p), v in sorted(data.items()):
        for run in v["regimes"].get("saturated", [])[:1]:
            for lv in run.get("levels") or []:
                tps = lv.get("output_tok_per_s") or 0
                cost = f" {gpu_usd_per_hour / (tps * 3600) * 1e6:.3f} |" if gpu_usd_per_hour and tps else (" - |" if gpu_usd_per_hour else "")
                lines.append(f"| {e} | {p} | {lv['concurrency']} | {tps:.0f} | {_fmt((lv.get('tpot_p50_s') or 0) * 1e3)} | "
                             f"{_fmt((lv.get('tpot_p99_s') or 0) * 1e3)} | {lv.get('meets_limits')} |{cost}")
    lines += ["", "## Correctness (equiv against the bf16 reference)", "", "| engine | precision | verdict | reasons |",
              "|---|---|---|---|"]
    for (e, p), v in sorted(data.items()):
        q = v["equiv"] or {}
        lines.append(f"| {e} | {p} | {q.get('verdict', '-')} | {'; '.join(q.get('reasons') or [])[:120]} |")
    gpus = {g for v in data.values() for g in v["gpus"]}
    if len(gpus) > 1:
        lines += ["", f"**Measured on {len(gpus)} different GPUs** (a spot VM restarted on another host): "
                  + "; ".join(f"{e}-{p}: {', '.join(sorted(v['gpus']))}" for (e, p), v in sorted(data.items())),
                  "Compare only configurations measured on the same GPU, or rerun on one."]
    check = d / "clientcheck.json"
    if check.exists():
        c = json.loads(check.read_text())
        lines += ["", f"Client check: clean up to concurrency {c['max_clean_concurrency']} closed loop and "
                  f"{c['max_clean_rate']} req/s open loop."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dir", type=Path)
    ap.add_argument("--gpu-usd-per-hour", type=float)
    ap.add_argument("--tol", type=float, default=2.0, help="the searches' resolution, %% (the noise floor of a lead)")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    text = render(a.dir, a.gpu_usd_per_hour, a.tol)
    print(text)
    if a.out:
        a.out.write_text(text)
