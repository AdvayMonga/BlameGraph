"""Each kernel task's share of the engine's profiled window, from a `profile` bundle (`lab/bundle.py`: kernels.json +
meta.json): the device kernels the task's entry launches (`Task.kernel_names`), their share of the window, and the
end-to-end speedup bound if they took no time (Amdahl). Facts for choosing the next task; the kernels no task covers
are listed by share."""
from __future__ import annotations

import json
import re
from pathlib import Path

from lab import calibrate


def shares(bundle: Path, tasks: list, unmatched: int = 15) -> dict | None:
    cal = calibrate.calibrate_kernels(Path(bundle), None, top=10**9)
    if cal is None:
        return None
    claimed: set[str] = set()
    out = {}
    for t in tasks:
        mine = [k for k in cal["kernels"] if any(re.search(p, k["name"]) for p in t.kernel_names)]
        claimed |= {k["name"] for k in mine}
        s = sum(k["share_of_window"] for k in mine)
        out[t.name] = {"share_of_window": s, "kernels": [k["name"] for k in mine],
                       "speedup_if_free": 1 / (1 - s) if s < 1 else None}
    rest = [{"name": k["name"], "share_of_window": k["share_of_window"], "count": k["count"]}
            for k in cal["kernels"] if k["name"] not in claimed][:unmatched]
    return {"window_s": cal["window_s"], "kernel_share_total": cal["kernel_share_total"], "tasks": out,
            "unmatched": rest}


def weighted_pct(gate: dict) -> float | None:
    """Call-weighted % of the roofline bound over the shapes a `gate` result timed (seen and held-out)."""
    rows = [r for k in ("timings", "heldout_timings") for r in json.loads((gate.get("artifacts") or {}).get(k) or "[]")
            if r.get("pct_of_roofline") is not None]
    w = sum(r.get("weight") or 1 for r in rows)
    return sum((r.get("weight") or 1) * r["pct_of_roofline"] for r in rows) / w if rows else None


def rank(bundles: list[Path], tasks: list, gates: dict | None = None) -> dict:
    """Each task's share of the window in every profile (one per corpus class or regime), their mean, its kernel's %
    of the roofline bound (from a `gate` result of the engine's own kernel, when given), and the fraction of the
    window that would disappear at the roofline: share x (1 - % of roofline), the Amdahl ceiling gain
    (validity.ceiling.ceiling_gain; GEAK ranks by GPU-time share x achievable speedup). Ranked by it, else by share."""
    from validity.ceiling import ceiling_gain
    per = [shares(b, tasks) for b in bundles]
    out = []
    for t in tasks:
        sh = [p["tasks"][t.name]["share_of_window"] for p in per if p]
        mean = sum(sh) / len(sh) if sh else 0.0
        pct = weighted_pct((gates or {})[t.name]) if t.name in (gates or {}) else None
        out.append({"task": t.name, "share_by_profile": sh, "share": mean, "pct_of_roofline": pct,
                    "gain_at_roofline": ceiling_gain(mean, pct), "speedup_if_free": 1 / (1 - mean) if mean < 1 else None})
    out.sort(key=lambda r: -(r["gain_at_roofline"] if r["gain_at_roofline"] is not None else r["share"]))
    return {"profiles": [str(b) for b in bundles], "tasks": out,
            "unmatched": per[0]["unmatched"] if per and per[0] else []}
