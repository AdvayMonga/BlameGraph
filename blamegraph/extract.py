"""LLM extractor: turn the benchmark numbers an agent printed (in any format) into structured observations.

Haiku reads each metric-looking tool output once and returns JSON. Results are cached in
data/derived/observations.jsonl keyed by (run_id, step_i) so reruns only pay for new steps.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import anthropic

from .experiment import Observation
from .traces import Run

MODEL = "claude-haiku-4-5"
CACHE = Path(__file__).resolve().parent.parent / "data" / "derived" / "observations.jsonl"

KW = re.compile(r"p50|ttft|tpot|throughput|tok(?:ens)?/s|req/s|req_ps|fail(?:ure)?[_ ]?rate|latency|mmlu|quality", re.I)
NOISE = re.compile(r"\(APIServer pid=\d+\)|Avg prompt throughput|Avg generation throughput|Loading weights|it/s\]")

SYSTEM = """You extract benchmark results from tool output produced while an AI agent optimized an LLM inference server.
The agent's benchmark (evaluate.py) reports, per load profile (burst/poisson/constant): TTFT p50/p90/p99 (seconds),
TPOT p50 (seconds), ITL, generation_throughput_tokens_per_s, request_throughput_req_per_s, failure_rate, and a
quality_check with an MMLU-Pro accuracy ratio vs baseline and pass=true/false.

Return ONLY benchmark results the agent actually observed in this output. Ignore: server log lines, plans,
code that would print values, help text, and numbers that are configuration (e.g. --max-num-seqs 256).
Convert milliseconds to seconds. If a value is shown as a percentage, express failure_rate as a fraction.
If several profiles are shown, report the burst profile in the top-level fields and list all in `profiles`.
If the output contains no observed benchmark result, set is_benchmark_result=false and leave fields null."""

SCHEMA = {
    "type": "object",
    "properties": {
        "is_benchmark_result": {"type": "boolean"},
        "ttft_p50_s": {"type": ["number", "null"]},
        "tpot_p50_s": {"type": ["number", "null"]},
        "request_throughput_rps": {"type": ["number", "null"]},
        "generation_tps": {"type": ["number", "null"]},
        "failure_rate": {"type": ["number", "null"]},
        "mmlu_ratio": {"type": ["number", "null"]},
        "quality_pass": {"type": ["boolean", "null"]},
        "profiles": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "ttft_p50_s": {"type": ["number", "null"]},
                    "tpot_p50_s": {"type": ["number", "null"]},
                    "request_throughput_rps": {"type": ["number", "null"]},
                    "failure_rate": {"type": ["number", "null"]},
                },
                "required": ["name", "ttft_p50_s", "tpot_p50_s", "request_throughput_rps", "failure_rate"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["is_benchmark_result", "ttft_p50_s", "tpot_p50_s", "request_throughput_rps", "generation_tps",
                 "failure_rate", "mmlu_ratio", "quality_pass", "profiles"],
    "additionalProperties": False,
}


def trim(out: str, cap: int = 5000) -> str:
    """Keep metric-bearing lines (+1 line of context each), drop server-log noise, cap length."""
    lines = out.splitlines()
    keep: set[int] = set()
    for k, l in enumerate(lines):
        if KW.search(l) and not NOISE.search(l):
            keep.update((k - 1, k, k + 1))
    return "\n".join(lines[k] for k in sorted(keep) if 0 <= k < len(lines))[:cap]


def candidates(run: Run) -> list[tuple[int, str]]:
    out = []
    for s in run.steps():
        if s.tool in ("bash", "shell", "taskoutput", "monitor", "read") and s.output and KW.search(s.output):
            t = trim(s.output)
            if t.strip():
                out.append((s.i, t))
    return out


@dataclass
class Extracted:
    run_id: str
    step_i: int
    data: dict
    input_tokens: int
    output_tokens: int


def load_cache() -> dict[tuple[str, int], dict]:
    if not CACHE.exists():
        return {}
    out = {}
    for line in CACHE.read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            out[(d["run_id"], d["step_i"])] = d
    return out


def obs_by_run() -> dict[str, dict[int, dict]]:
    """Cached extractor output grouped as run_id -> {step_i: data}, for build_log(run, llm_obs=...)."""
    out: dict[str, dict[int, dict]] = {}
    for (rid, i), d in load_cache().items():
        out.setdefault(rid, {})[i] = d["data"]
    return out


def to_observation(step_i: int, data: dict, config_idx: int | None) -> Observation | None:
    if not data.get("is_benchmark_result"):
        return None
    obs = Observation(step_i=step_i, config_idx=config_idx, ttft_p50=data.get("ttft_p50_s"),
                      tpot_p50=data.get("tpot_p50_s"), rps=data.get("request_throughput_rps"),
                      gen_tps=data.get("generation_tps"), failure_rate=data.get("failure_rate"),
                      mmlu_ratio=data.get("mmlu_ratio"), quality_pass=data.get("quality_pass"))
    # sanity: latencies in seconds, within physical range
    for f in ("ttft_p50", "tpot_p50"):
        v = getattr(obs, f)
        if v is not None and not (0 < v < 60):
            setattr(obs, f, None)
    if any(v is not None for v in (obs.ttft_p50, obs.tpot_p50, obs.rps, obs.gen_tps, obs.failure_rate)):
        return obs
    return None


async def _extract_one(client: anthropic.AsyncAnthropic, sem: asyncio.Semaphore, run_id: str, step_i: int, text: str) -> Extracted:
    async with sem:
        resp = await client.messages.create(
            model=MODEL, max_tokens=1024,
            system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": f"Tool output:\n```\n{text}\n```"}],
            output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
        )
        body = next((b.text for b in resp.content if b.type == "text"), "{}")
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            data = {"is_benchmark_result": False, "_parse_error": body[:200]}
        return Extracted(run_id, step_i, data, resp.usage.input_tokens, resp.usage.output_tokens)


async def extract_runs(runs: list[Run], concurrency: int = 8, limit_steps: int | None = None) -> dict:
    """Extract all uncached candidate steps for the given runs; append to the cache; return usage totals."""
    cache = load_cache()
    todo = [(r.run_id, i, t) for r in runs for i, t in candidates(r) if (r.run_id, i) not in cache]
    if limit_steps:
        todo = todo[:limit_steps]
    client = anthropic.AsyncAnthropic(default_headers=_headers())
    sem = asyncio.Semaphore(concurrency)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    tot_in = tot_out = done = errors = 0
    with open(CACHE, "a") as f:
        tasks = [asyncio.create_task(_extract_one(client, sem, rid, i, t)) for rid, i, t in todo]
        for fut in asyncio.as_completed(tasks):
            try:
                ex = await fut
            except anthropic.RateLimitError as e:
                errors += 1; print("rate limit:", e); continue
            except anthropic.APIStatusError as e:
                errors += 1; print("api error:", e.status_code, str(e)[:120]); continue
            except anthropic.APIConnectionError as e:
                errors += 1; print("connection error:", e); continue
            f.write(json.dumps({"run_id": ex.run_id, "step_i": ex.step_i, "data": ex.data,
                                "in": ex.input_tokens, "out": ex.output_tokens}) + "\n")
            f.flush()
            tot_in += ex.input_tokens; tot_out += ex.output_tokens; done += 1
            if done % 200 == 0:
                print(f"..{done}/{len(todo)} in={tot_in} out={tot_out} ~${tot_in/1e6*1 + tot_out/1e6*5:.2f}", flush=True)
    return {"todo": len(todo), "done": done, "errors": errors, "input_tokens": tot_in, "output_tokens": tot_out,
            "est_cost_usd": round(tot_in / 1e6 * 1.0 + tot_out / 1e6 * 5.0, 2)}


def _headers() -> dict[str, str]:
    """Org-level API keys must name a workspace; set ANTHROPIC_WORKSPACE_ID in .env."""
    ws = os.environ.get("ANTHROPIC_WORKSPACE_ID")
    return {"anthropic-workspace-id": ws} if ws else {}


def load_env():
    """Populate ANTHROPIC_API_KEY from the project's .env when the shell doesn't export it."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    env = Path(__file__).resolve().parent.parent / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))
