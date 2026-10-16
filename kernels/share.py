"""Each kernel task's share of the engine's profiled window, from a `profile` bundle (`lab/bundle.py`: kernels.json +
meta.json): the device kernels the task's entry launches (`Task.kernel_names`), their share of the window, and the
end-to-end speedup bound if they took no time (Amdahl). Facts for choosing the next task; the kernels no task covers
are listed by share."""
from __future__ import annotations

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
