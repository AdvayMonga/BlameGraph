"""Load InferenceBench trajectories into one Run object per directory."""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "inferencebench"

# Mirrors src/eval/inference/hpo_search_baselines.py in aisa-group/InferenceBench.
FAILURE_SCORE = 1e-6

# PyTorch-baseline primary metric per scenario, backed out from README leaderboard means
# (consensus across ~10 agents). Not published by the authors.
BASELINE_METRIC = {"A": 1.236, "B": 28.64, "C": 0.132, "D": 1.679}


@dataclass
class Event:
    i: int
    role: str
    type: str
    text: str | None = None
    tool_name: str | None = None
    tool_input: Any = None
    tool_output: Any = None
    raw: dict = field(default_factory=dict, repr=False)

    @property
    def tool_input_text(self) -> str:
        """Best-effort flatten of tool_input to the command/content string."""
        ti = self.tool_input
        if isinstance(ti, str):
            return ti
        if isinstance(ti, dict):
            for k in ("command", "cmd", "content", "new_string", "file_text"):
                if isinstance(ti.get(k), str):
                    return ti[k]
            return json.dumps(ti)
        return "" if ti is None else str(ti)

    @property
    def tool_output_text(self) -> str:
        to = self.tool_output
        if isinstance(to, str):
            return to
        if to is None:
            return ""
        return json.dumps(to)


@dataclass
class Run:
    run_id: str
    meta: dict
    events: list[Event]
    metrics: dict | None

    # ---- metadata ----
    @property
    def agent(self) -> str:
        return self.meta["agent"]

    @property
    def scenario(self) -> str:
        return self.meta["scenario"]

    @property
    def harness(self) -> str:
        return self.meta["harness"]

    @property
    def flagged(self) -> bool:
        return bool(self.meta.get("invalid_or_reward_hack"))

    # ---- outcome (mirrors InferenceBench scoring) ----
    @property
    def gate_passed(self) -> bool:
        m = self.metrics
        if not m or m.get("error") or m.get("score") == 0:
            return False
        qc = m.get("quality_check")
        return isinstance(qc, dict) and qc.get("pass") is True

    @property
    def primary_metric(self) -> tuple[float, str]:
        """Higher is better. (FAILURE_SCORE, reason) when unscorable."""
        if not self.metrics:
            return FAILURE_SCORE, "no_metrics"
        return primary_metric(self.metrics, self.scenario)

    @property
    def scored(self) -> bool:
        """True when the run would contribute a real (>1.0x-eligible) score."""
        return self.gate_passed and not self.flagged and self.primary_metric[0] > FAILURE_SCORE

    @property
    def outcome(self) -> float:
        """Primary metric if scored, else 0.0 (stands in for the 1.0x floor)."""
        return self.primary_metric[0] if self.scored else 0.0

    @property
    def speedup(self) -> float:
        """Estimated speedup over the PyTorch baseline (1.0 for unscored runs), as InferenceBench reports it."""
        return max(1.0, self.outcome / BASELINE_METRIC[self.scenario]) if self.scored else 1.0

    # ---- event helpers ----
    def tool_calls(self) -> Iterator[Event]:
        return (e for e in self.events if e.type == "tool_call")

    def tool_results(self) -> Iterator[Event]:
        return (e for e in self.events if e.type == "tool_result")

    def assistant_text(self) -> Iterator[Event]:
        return (e for e in self.events if e.role == "assistant" and e.type in ("text", "message"))

    def steps(self) -> list["Step"]:
        """Pair tool calls with results. Results are FIFO within one assistant turn;
        a turn boundary is any assistant text/thinking event. Unmatched calls get result=None."""
        out: list[Step] = []
        calls: list[Event] = []
        results: list[Event] = []

        def flush():
            for k, c in enumerate(calls):
                out.append(Step(call=c, result=results[k] if k < len(results) else None))
            calls.clear(); results.clear()

        for e in self.events:
            if e.role == "assistant" and e.type in ("text", "thinking", "message"):
                flush()
            elif e.type == "tool_call":
                calls.append(e)
            elif e.type == "tool_result":
                results.append(e)
        flush()
        return out


CODEX_DIFF_RE = re.compile(r"\nfile update:\n(diff --git .*)\Z", re.S)
TRAILING_DIFF_RE = re.compile(r"(?:^|\n)diff --git a/.*\Z", re.S)


@dataclass
class Step:
    call: Event
    result: Event | None

    @property
    def i(self) -> int:
        return self.call.i

    @property
    def tool(self) -> str:
        return (self.call.tool_name or "").lower()

    @property
    def cmd(self) -> str:
        return self.call.tool_input_text

    @property
    def output(self) -> str:
        """Tool output with the codex harness's trailing cumulative diff removed (the 'file update:' header
        sometimes lands in the command field instead, so any trailing `diff --git` block is cut too)."""
        if self.result is None:
            return ""
        out = CODEX_DIFF_RE.sub("", self.result.tool_output_text)
        if "git diff" not in self.cmd:
            m = TRAILING_DIFF_RE.search(out)
            if m:
                out = out[:m.start()]
        return re.sub(r"\n?file update:\s*\Z", "", out)

    @property
    def codex_diff(self) -> str | None:
        """The cumulative workspace diff codex appends to results, if present."""
        if self.result is None:
            return None
        m = CODEX_DIFF_RE.search(self.result.tool_output_text)
        return m.group(1) if m else None

    @property
    def is_error(self) -> bool:
        if self.result is None:
            return False
        meta = self.result.raw.get("meta") or {}
        return bool(meta.get("is_error")) or meta.get("status") == "failed"


def _safe_p50(profile: dict, group: str) -> float | None:
    v = (profile.get(group) or {}).get("p50")
    return float(v) if isinstance(v, (int, float)) else None


def profile_is_physical(profile: dict) -> bool:
    """InferenceBench's integrity floor: impossibly fast numbers mean the server cheated."""
    gen = profile.get("generation_throughput_tokens_per_s")
    if isinstance(gen, (int, float)) and gen > 20_000.0:
        return False
    tpot, ttft = _safe_p50(profile, "tpot"), _safe_p50(profile, "ttft")
    if tpot is not None and 0 < tpot < 1e-4:
        return False
    if ttft is not None and 0 < ttft < 0.010:
        return False
    return True


def primary_metric(metrics: dict, scenario: str) -> tuple[float, str]:
    profiles = metrics.get("profiles") or {}
    burst = profiles.get("burst")
    if not isinstance(burst, dict):
        return FAILURE_SCORE, "missing_burst_profile"
    if any(isinstance(p, dict) and not profile_is_physical(p) for p in profiles.values()):
        return FAILURE_SCORE, "integrity_failed"
    if scenario == "A":
        ttft = _safe_p50(burst, "ttft")
        return (1.0 / ttft, "1/ttft_p50") if ttft and ttft > 0 else (FAILURE_SCORE, "missing_ttft")
    if scenario == "B":
        tpot = _safe_p50(burst, "tpot")
        return (1.0 / tpot, "1/tpot_p50") if tpot and tpot > 0 else (FAILURE_SCORE, "missing_tpot")
    if scenario == "C":
        vals = []
        for name in ("burst", "poisson", "constant"):
            p = profiles.get(name)
            if not isinstance(p, dict):
                return FAILURE_SCORE, f"missing_{name}_profile"
            req = p.get("request_throughput_req_per_s")
            if not isinstance(req, (int, float)) or req <= 0:
                return FAILURE_SCORE, "missing_request_throughput"
            vals.append(float(req))
        return math.prod(vals) ** (1 / len(vals)), "geomean_req_per_s"
    if scenario == "D":
        ttft, tpot = _safe_p50(burst, "ttft"), _safe_p50(burst, "tpot")
        req = burst.get("request_throughput_req_per_s")
        if not ttft or not tpot or not isinstance(req, (int, float)) or req <= 0:
            return FAILURE_SCORE, "missing_geomean_component"
        return (1 / ttft * 1 / tpot * req) ** (1 / 3), "geomean_inv_latency_throughput"
    return FAILURE_SCORE, "unknown_scenario"


def load_run(run_dir: Path) -> Run:
    meta = json.loads((run_dir / "run_meta.json").read_text())
    events = []
    with open(run_dir / "trace.jsonl") as f:
        for line in f:
            d = json.loads(line)
            events.append(Event(
                i=d["i"], role=d["role"], type=d["type"], text=d.get("text"),
                tool_name=d.get("tool_name"), tool_input=d.get("tool_input"),
                tool_output=d.get("tool_output"), raw=d,
            ))
    mp = run_dir / "metrics.json"
    metrics = json.loads(mp.read_text()) if mp.exists() else None
    return Run(run_id=run_dir.name, meta=meta, events=events, metrics=metrics)


def iter_runs(root: Path = DATA_ROOT, limit: int | None = None) -> Iterator[Run]:
    dirs = sorted((root / "runs").iterdir())
    for d in dirs[:limit]:
        if (d / "trace.jsonl").exists():
            yield load_run(d)
