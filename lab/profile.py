"""profile: run the served engine in process under torch.profiler on a workload and write a raw bundle (usage: lab/README.md)."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import logging
import random
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity

from inference_server.backends.base import InferenceBackend
from inference_server.config import Settings, load_settings
from inference_server.scheduler import ContinuousBatchScheduler, ScheduledRequest
from inference_server.server import build_backend, build_scheduler
from inference_server.timeline import Timeline
from lab import bundle, gpu

logger = logging.getLogger(__name__)


COUNTERS = ("total_admitted", "total_completed", "total_rejected", "total_preempted", "total_expired",
            "total_iteration_errors", "decode_steps", "kv_admit_blocked", "prefill_chunks_processed")


def synthetic_prompts(n: int, length: int, vocab: int = 1000, seed: int = 0) -> list[list[int]]:
    rng = random.Random(seed)
    return [[rng.randrange(1, vocab) for _ in range(length)] for _ in range(n)]


def warmup_prompts(prompts: list[list[int]], n: int, vocab: int = 1000, seed: int = 1) -> list[list[int]]:
    """Random ids at the window's prompt lengths: the same shapes get compiled, nothing lands in the prefix cache."""
    rng = random.Random(seed)
    return [[rng.randrange(1, vocab) for _ in prompts[i % len(prompts)]] for i in range(n)]


def _profiler(device: str) -> tuple[torch.profiler.profile, dict]:
    """All threads: by default the profiler records only the thread that opened it, and the scheduler has its own."""
    acts = [ProfilerActivity.CPU]
    if device.startswith("cuda"):
        acts.append(ProfilerActivity.CUDA)
    config = {"activities": [a.name for a in acts], "all_threads": True}
    try:
        from torch._C._profiler import _ExperimentalConfig
        exp = _ExperimentalConfig(profile_all_threads=True)
    except (ImportError, TypeError):   # older torch: the trace will miss the scheduler thread
        logger.warning("torch %s cannot profile all threads; trace covers the main thread only",
                       torch.__version__)
        return torch.profiler.profile(activities=acts), {**config, "all_threads": False}
    return torch.profiler.profile(activities=acts, experimental_config=exp), config


async def _fire(sched: ContinuousBatchScheduler, prompts: list[list[int]], max_tokens: int,
                session: str) -> list:
    loop = asyncio.get_running_loop()
    reqs = [ScheduledRequest(token_ids=p, max_tokens=max_tokens, session_id=f"{session}-{i}",
                             future=loop.create_future()) for i, p in enumerate(prompts)]
    return await asyncio.gather(*(sched.submit(r) for r in reqs), return_exceptions=True)


def stats_delta(start: dict, end: dict) -> dict:
    """What the window itself did: counter differences, wave sizes included."""
    d = {k: end[k] - start[k] for k in COUNTERS if k in start and k in end}
    waves = {k: v - start.get("wave_sizes", {}).get(k, 0) for k, v in end.get("wave_sizes", {}).items()}
    d["wave_sizes"] = {k: v for k, v in waves.items() if v}
    return d


def outcomes(results: list, max_tokens: int) -> dict:
    """Per-request outcome counts and output lengths; `short` stopped before max_tokens (EOS or cut: not recorded)."""
    toks = [len(r) for r in results if not isinstance(r, BaseException)]
    return {"errors": len(results) - len(toks), "empty": toks.count(0),
            "short": sum(0 < n < max_tokens for n in toks), "full": toks.count(max_tokens),
            "tokens_out": ({"total": sum(toks), "min": min(toks), "mean": sum(toks) / len(toks), "max": max(toks)}
                           if toks else None)}


async def run(backend: InferenceBackend, settings: Settings, prompts: list[list[int]],
              max_tokens: int, out: Path, warmup: int | None = None, source: dict | None = None) -> Path:
    """Profile `prompts` through the served scheduler config on `backend`; return the bundle directory.
    `warmup` requests (default: as many as the window) are fired together first, outside the window."""
    out = bundle.new_dir(out)
    device = backend.device_str
    warmup = len(prompts) if warmup is None else warmup
    timeline = Timeline(out)
    sched = build_scheduler(backend, settings, timeline=timeline)
    sched.start()
    sampler = gpu.Sampler(out / "gpu.csv")
    prof, prof_config = _profiler(device)
    try:
        w0 = time.time()
        if warmup:
            await _fire(sched, warmup_prompts(prompts, warmup), max_tokens, "warmup")
        w1 = time.time()
        peak_before = bundle.device_peaks(device)
        bundle.reset_peaks(device)
        mem0, stats0, gpu0 = bundle.device_memory(device), sched.stats(), gpu.query()
        timeline.event("profile_window", state="begin")
        sampler.start()
        t0 = time.time()
        with prof:
            results = await _fire(sched, prompts, max_tokens, "profile")
        t1 = time.time()
        timeline.event("profile_window", state="end")
        mem1, stats1, gpu1 = bundle.device_memory(device), sched.stats(), gpu.query()
        peak_window = bundle.device_peaks(device)
    finally:
        sampler.stop()
        await sched.stop()
    prof.export_chrome_trace(str(out / "trace.json"))
    bundle.write_json(out / "ops.json", bundle.ops_table(prof))
    kernels = bundle.kernels_table(prof.events())
    if kernels is not None:
        bundle.write_json(out / "kernels.json", kernels)
    lifetime = (None if peak_window is None else
                {k: max(v, (peak_before or {}).get(k, 0)) for k, v in peak_window.items()})
    bundle.write_json(out / "memory.json", {
        "device": device, "window_peak": peak_window, "lifetime_peak": lifetime,
        "host_rss_peak_lifetime_bytes": mem1["peak_host_rss_bytes"],
        "at_window_start": mem0, "at_window_end": mem1})
    bundle.write_json(out / "stats.json", {"window_start": stats0, "window_end": stats1,
                                           "window_delta": stats_delta(stats0, stats1)})
    done = outcomes(results, max_tokens)
    bundle.write_json(out / "meta.json", bundle.meta(
        device, settings, prompts, max_tokens, t0, t1, failed=done["errors"] + done["empty"], outcomes=done,
        warmup={"requests": warmup, "concurrent": True, "prompts": "random ids at the window's prompt lengths",
                "max_tokens": max_tokens, "wall_s": w1 - w0},
        profiler_on=True, profiler=prof_config, gpu={"window_start": gpu0, "window_end": gpu1},
        timeline=timeline.stats(), workload=source))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("(")[0])
    ap.add_argument("--backend", help="override BACKEND, e.g. custom-mps")
    ap.add_argument("--model", help="override MODEL_NAME")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--corpus-class", help="prompts drawn from this corpus class (verified loader)")
    src.add_argument("--workload", help="JSON from lab.corpus.sample, for a tree without the corpus (the profile tool)")
    src.add_argument("--prompts", help="JSON list of prompt strings")
    src.add_argument("--synthetic", action="store_true", help="random token ids of --prompt-len")
    ap.add_argument("--split", default="seen", choices=("seen", "heldout"))
    ap.add_argument("--corpus-dir", help="default: this repo's corpus/")
    ap.add_argument("--enable-thinking", action="store_true", help="chat template thinking on, for corpus prompts")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--requests", type=int, default=8)
    ap.add_argument("--prompt-len", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--warmup", type=int, help="warmup requests, fired together; default: as many as the window")
    ap.add_argument("--out", default="lab/runs")
    args = ap.parse_args(argv)

    # The served configuration, from the same env the server reads; only the workload is ours.
    settings = load_settings()
    overrides = {k: v for k, v in (("backend_name", args.backend), ("model_name", args.model)) if v}
    settings = dataclasses.replace(settings, **overrides)
    backend, _ = build_backend(settings)
    if args.synthetic:
        prompts = synthetic_prompts(args.requests, args.prompt_len, seed=args.seed)
        source = {"kind": "synthetic", "prompt_len": args.prompt_len, "seed": args.seed}
    else:
        from inference_server.tokenizer import Tokenizer
        tok = Tokenizer(settings.model_name, settings.context_window)
        if args.prompts:
            prompts = [tok.encode_chat(t) for t in json.loads(Path(args.prompts).read_text())]
            source = {"kind": "prompts", "file": args.prompts}
        else:
            if args.workload:
                w = json.loads(Path(args.workload).read_text())
            else:
                from lab import corpus
                w = corpus.sample(args.corpus_class, args.split, args.seed, args.requests,
                                  Path(args.corpus_dir) if args.corpus_dir else corpus.CORPUS_DIR)
            prompts = [tok.encode_messages(m, thinking=args.enable_thinking) for m in w["messages"]]
            source = {**w["source"], "thinking": args.enable_thinking}
    out = asyncio.run(run(backend, settings, prompts, args.max_tokens, Path(args.out),
                          warmup=args.warmup, source=source))
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
