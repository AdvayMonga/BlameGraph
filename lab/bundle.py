"""One directory per profile run: the raw outputs of every layer plus the provenance they were taken under (files: lab/README.md)."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
import platform
import resource
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import torch
from torch.autograd import DeviceType

FILES = ("events.jsonl", "trace.json", "ops.json", "memory.json", "stats.json", "meta.json")


def new_dir(root: str | Path) -> Path:
    d = Path(root) / f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    d.mkdir(parents=True, exist_ok=False)
    return d


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=1, default=str))


def git_sha() -> str | None:
    """LAB_GIT_SHA when lab.vm shipped it (the pushed tree has no .git), else git here, else None."""
    if os.environ.get("LAB_GIT_SHA"):
        return os.environ["LAB_GIT_SHA"]
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    except OSError:
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def workload_hash(prompts: list[list[int]], max_tokens: int) -> str:
    h = hashlib.sha256(json.dumps([prompts, max_tokens]).encode())
    return h.hexdigest()[:16]


def device_memory(device: str) -> dict:
    """Allocator view for the device the engine ran on, plus peak host RSS."""
    out: dict[str, Any] = {"device": device,
                           "peak_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                           * (1 if sys.platform == "darwin" else 1024)}
    if device.startswith("cuda") and torch.cuda.is_available():
        out["cuda"] = torch.cuda.memory_stats()
    elif device == "mps" and torch.backends.mps.is_available():
        out["mps"] = {"current_allocated": torch.mps.current_allocated_memory(),
                      "driver_allocated": torch.mps.driver_allocated_memory()}
    return out


def device_peaks(device: str) -> dict | None:
    """Allocator high-water marks since the last reset; CUDA only (MPS keeps no peak)."""
    if not (device.startswith("cuda") and torch.cuda.is_available()):
        return None
    s = torch.cuda.memory_stats()
    return {"allocated_bytes": s.get("allocated_bytes.all.peak", 0), "reserved_bytes": s.get("reserved_bytes.all.peak", 0)}


def reset_peaks(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def ops_table(prof: Any, top: int = 200) -> dict:
    """prof.key_averages() as rows (times in us), sorted by self device time then self CPU time; top N plus totals."""
    rows = [{"name": a.key, "count": a.count, "self_cpu_us": a.self_cpu_time_total, "cpu_us": a.cpu_time_total,
             "self_device_us": a.self_device_time_total, "device_us": a.device_time_total}
            for a in prof.key_averages()]
    rows.sort(key=lambda r: (r["self_device_us"], r["self_cpu_us"]), reverse=True)
    return {"sort": "self_device_us, self_cpu_us", "ops_total": len(rows), "top": rows[:top],
            "totals": {"self_cpu_us": sum(r["self_cpu_us"] for r in rows),
                       "self_device_us": sum(r["self_device_us"] for r in rows)}}


def kernels_table(events: Any) -> dict | None:
    """CUDA kernels aggregated by name (times in us), by total time; None when the trace has no CUDA activity.
    record_function ranges also appear on the device timeline: they are annotations, not kernels."""
    agg: dict[str, list[float]] = {}
    for e in events:
        if e.device_type == DeviceType.CUDA and not getattr(e, "is_user_annotation", False):
            agg.setdefault(e.name, []).append(e.time_range.elapsed_us())
    if not agg:
        return None
    rows = [{"name": k, "count": len(v), "total_us": sum(v), "mean_us": sum(v) / len(v), "max_us": max(v)}
            for k, v in agg.items()]
    rows.sort(key=lambda r: r["total_us"], reverse=True)
    return {"kernels": rows, "total_us": sum(r["total_us"] for r in rows), "launches": sum(r["count"] for r in rows)}


def meta(device: str, settings: Any, prompts: list[list[int]], max_tokens: int, t0: float,
         t1: float, **extra: Any) -> dict:
    """A number is a fact about a config, so the engine settings travel with every bundle."""
    return {"git_sha": git_sha(), "torch": torch.__version__, "python": sys.version.split()[0],
            "platform": platform.platform(), "device": device,
            "settings": asdict(settings),
            "workload_hash": workload_hash(prompts, max_tokens), "requests": len(prompts),
            "max_tokens": max_tokens, "window": {"start": t0, "end": t1, "wall_s": t1 - t0},
            **extra}
