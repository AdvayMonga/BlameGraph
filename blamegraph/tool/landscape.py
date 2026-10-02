"""Pooled landscape: every standard measurement from every session, keyed by (scenario, config hash), with the
knobs that config set. Queries return data points with counts and spread, never recommendations."""
from __future__ import annotations

import json
import math
import re
import statistics
from dataclasses import dataclass, asdict
from pathlib import Path

from .ledger import Ledger

KNOB_RE = re.compile(r"(--[a-zA-Z][\w-]*)(?:[= ]((?:\"[^\"]*\"|'[^']*'|[^\s\"'\\-][^\s\"'\\]*)))?")
ENV_RE = re.compile(r"^\s*(?:export\s+)?([A-Z][A-Z0-9_]*)=([^\s#]+)", re.M)


def knobs(script: str) -> dict[str, str]:
    body = re.sub(r"\\\n", " ", script)
    out = {m.group(1): (m.group(2) or "").strip("\"'") for m in KNOB_RE.finditer(body)}
    for k, v in ENV_RE.findall(body):
        if k.startswith(("VLLM_", "SGLANG_", "CUDA_", "NCCL_", "TORCH")):
            out[k] = v
    return {k: re.sub(r"\$\{\w+:-([^}]*)\}", r"\1", v) for k, v in out.items()}


def primary(metrics: dict, scenario: str) -> float | None:
    profiles = (metrics or {}).get("profiles") or {}
    b = profiles.get("burst") or {}
    try:
        if scenario == "A":
            return 1 / b["ttft"]["p50"]
        if scenario == "B":
            return 1 / b["tpot"]["p50"]
        if scenario == "C":
            vals = [profiles[n]["request_throughput_req_per_s"] for n in ("burst", "poisson", "constant")]
            return math.prod(vals) ** (1 / 3)
        if scenario == "D":
            return (1 / b["ttft"]["p50"] / b["tpot"]["p50"] * b["request_throughput_req_per_s"]) ** (1 / 3)
    except (KeyError, TypeError, ZeroDivisionError):
        return None
    return None


@dataclass
class Point:
    scenario: str
    config_hash: str
    knobs: dict
    values: list[float]           # primary metric per standard full fresh measurement
    sessions: list[str]

    @property
    def mean(self) -> float:
        return statistics.mean(self.values)

    @property
    def cv(self) -> float | None:
        return statistics.pstdev(self.values) / self.mean if len(self.values) >= 2 and self.mean > 0 else None

    def to_dict(self) -> dict:
        d = asdict(self); d.update(mean=self.mean, cv=self.cv, n=len(self.values)); return d


class Landscape:
    def __init__(self, path: Path):
        self.path = path
        self.points: dict[tuple[str, str], Point] = {}
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    d = json.loads(line)
                    self.points[(d["scenario"], d["config_hash"])] = Point(d["scenario"], d["config_hash"], d["knobs"], d["values"], d["sessions"])

    def add_session(self, ledger: Ledger, scenario: str, session_id: str) -> int:
        content = {c["hash"]: c["content"] for c in ledger.configs()}
        n = 0
        for m in ledger.measurements():
            if not (m.get("standard") and m.get("mode") == "full" and not m.get("stale") and not m.get("error")):
                continue
            v = primary(m.get("metrics"), scenario)
            if not v:
                continue
            key = (scenario, m["live_hash"])
            p = self.points.get(key) or Point(scenario, m["live_hash"], knobs(content.get(m["live_hash"], "")), [], [])
            p.values.append(v); p.sessions.append(session_id); self.points[key] = p; n += 1
        return n

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("\n".join(json.dumps({**asdict(p)}) for p in self.points.values()) + ("\n" if self.points else ""))

    def near(self, scenario: str, script: str, max_dist: int = 2) -> list[dict]:
        """Points whose knobs differ from `script` in at most max_dist keys, with distance; sorted by distance then mean."""
        k0 = knobs(script); out = []
        for p in self.points.values():
            if p.scenario != scenario:
                continue
            keys = set(k0) | set(p.knobs)
            dist = sum(1 for k in keys if k0.get(k) != p.knobs.get(k))
            if dist <= max_dist:
                out.append({**p.to_dict(), "distance": dist})
        return sorted(out, key=lambda d: (d["distance"], -d["mean"]))

    def noise(self, scenario: str) -> dict:
        cvs = sorted(p.cv for p in self.points.values() if p.scenario == scenario and p.cv is not None)
        return {"n_repeated_configs": len(cvs), "cv_median": cvs[len(cvs) // 2] if cvs else None, "cv_p75": cvs[int(len(cvs) * .75)] if cvs else None}
