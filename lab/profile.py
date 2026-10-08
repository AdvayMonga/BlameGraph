"""profile: run the served engine in process under torch.profiler on a workload and write a raw bundle (usage: lab/README.md)."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import dataclasses
import json
import logging
import os
import random
import subprocess
import sys
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


def synthetic_prompts(n: int, length: int, vocab: int = 1000, seed: int = 0) -> list[list[int]]:
    rng = random.Random(seed)
    return [[rng.randrange(1, vocab) for _ in range(length)] for _ in range(n)]


def _profiler(device: str) -> torch.profiler.profile:
    """All threads: by default the profiler records only the thread that opened it, and the scheduler has its own."""
    acts = [ProfilerActivity.CPU]
    if device.startswith("cuda"):
        acts.append(ProfilerActivity.CUDA)
    try:
        from torch._C._profiler import _ExperimentalConfig
        config = _ExperimentalConfig(profile_all_threads=True)
    except (ImportError, TypeError):   # older torch: the trace will miss the scheduler thread
        logger.warning("torch %s cannot profile all threads; trace covers the main thread only",
                       torch.__version__)
        return torch.profiler.profile(activities=acts)
    return torch.profiler.profile(activities=acts, experimental_config=config)


@contextlib.contextmanager
def _cuda_range(device: str):
    """cudaProfilerStart/Stop around the window: nsys --capture-range=cudaProfilerApi and ncu --profile-from-start off
    record only inside it."""
    on = device.startswith("cuda")
    if on:
        torch.cuda.profiler.start()
    try:
        yield None
    finally:
        if on:
            torch.cuda.synchronize()
            torch.cuda.profiler.stop()


def _allow_ptrace() -> None:
    """Let any process of ours read this one: under Yama ptrace_scope 1 only ancestors may, and py-spy is a child."""
    if sys.platform.startswith("linux"):
        import ctypes
        ctypes.CDLL(None).prctl(0x59616D61, ctypes.c_ulong(-1), 0, 0, 0)    # PR_SET_PTRACER, PR_SET_PTRACER_ANY


async def _sampled(sched: ContinuousBatchScheduler, prompts: list[list[int]], max_tokens: int, out: Path,
                   pyspy: tuple[str, float, int]) -> tuple[list, int]:
    """Repeat the workload for `seconds` while py-spy samples this process; one stack dump halfway. (results, rounds)"""
    exe, seconds, rate = pyspy
    _allow_ptrace()
    pid = str(os.getpid())
    with open(out / "pyspy-record.log", "w") as log:
        rec = subprocess.Popen([exe, "record", "--pid", pid, "--duration", str(int(seconds)), "--rate", str(rate),
                                "--format", "speedscope", "--output", str(out / "pyspy.speedscope.json"),
                                "--nonblocking"], stdout=log, stderr=log)

    async def dump():
        await asyncio.sleep(seconds / 2)
        d = await asyncio.to_thread(subprocess.run, [exe, "dump", "--pid", pid, "--nonblocking"],
                                    capture_output=True, text=True)
        (out / "pyspy-dump.txt").write_text(d.stdout + d.stderr)
    dumper = asyncio.create_task(dump())
    results, rounds, end = [], 0, time.monotonic() + seconds
    while time.monotonic() < end:
        results += await _fire(sched, prompts, max_tokens, f"profile{rounds}")
        rounds += 1
    await dumper
    await asyncio.to_thread(rec.wait, seconds + 60)
    return results, rounds


async def _fire(sched: ContinuousBatchScheduler, prompts: list[list[int]], max_tokens: int,
                session: str) -> list:
    loop = asyncio.get_running_loop()
    reqs = [ScheduledRequest(token_ids=p, max_tokens=max_tokens, session_id=f"{session}-{i}",
                             future=loop.create_future()) for i, p in enumerate(prompts)]
    return await asyncio.gather(*(sched.submit(r) for r in reqs), return_exceptions=True)


async def run(backend: InferenceBackend, settings: Settings, prompts: list[list[int]],
              max_tokens: int, out: Path, warmup: int = 1, profiler: str = "torch",
              pyspy: tuple[str, float, int] | None = None) -> Path:
    """Profile `prompts` through the served scheduler config on `backend`; return the bundle directory.
    `profiler`: torch (torch.profiler trace), cuda-range (window marked for nsys/ncu), pyspy (`pyspy` = exe, s, Hz)."""
    out = bundle.new_dir(out)
    device = backend.device_str
    timeline = Timeline(out)
    sched = build_scheduler(backend, settings, timeline=timeline)
    sched.start()
    sampler = gpu.Sampler(out / "gpu.csv")
    try:
        if warmup:
            await _fire(sched, prompts[:warmup], max_tokens, "warmup")
        timeline.event("profile_window", state="begin")
        sampler.start()
        t0, rounds = time.time(), 1
        if profiler == "pyspy":
            results, rounds = await _sampled(sched, prompts, max_tokens, out, pyspy)
        else:
            with (_profiler(device) if profiler == "torch" else _cuda_range(device)) as prof:
                results = await _fire(sched, prompts, max_tokens, "profile")
        t1 = time.time()
        timeline.event("profile_window", state="end")
    finally:
        sampler.stop()
        await sched.stop()
    if profiler == "torch":
        prof.export_chrome_trace(str(out / "trace.json"))
    bundle.write_json(out / "memory.json", bundle.device_memory(device))
    bundle.write_json(out / "stats.json", sched.stats())
    failed = sum(isinstance(r, BaseException) for r in results)
    bundle.write_json(out / "meta.json", bundle.meta(device, settings, prompts, max_tokens, t0, t1,
                                                     failed=failed, warmup=warmup, profiler=profiler,
                                                     rounds=rounds))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("(")[0])
    ap.add_argument("--backend", help="override BACKEND, e.g. custom-mps")
    ap.add_argument("--model", help="override MODEL_NAME")
    ap.add_argument("--prompts", help="JSON list of prompt strings; synthetic token ids when absent")
    ap.add_argument("--requests", type=int, default=8)
    ap.add_argument("--prompt-len", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--out", default="lab/runs")
    ap.add_argument("--profiler", choices=("torch", "cuda-range", "pyspy"), default="torch",
                    help="torch: torch.profiler trace; cuda-range: mark the window for nsys/ncu; pyspy: sample with py-spy")
    ap.add_argument("--seconds", type=float, default=20.0, help="pyspy: how long to sample under load")
    ap.add_argument("--rate", type=int, default=100, help="pyspy: samples per second")
    ap.add_argument("--pyspy-bin", default="py-spy")
    args = ap.parse_args(argv)

    # The served configuration, from the same env the server reads; only the workload is ours.
    settings = load_settings()
    overrides = {k: v for k, v in (("backend_name", args.backend), ("model_name", args.model)) if v}
    settings = dataclasses.replace(settings, **overrides)
    backend, _ = build_backend(settings)
    if args.prompts:
        from inference_server.tokenizer import Tokenizer
        tok = Tokenizer(settings.model_name, settings.context_window)
        prompts = [tok.encode_chat(t) for t in json.loads(Path(args.prompts).read_text())]
    else:
        prompts = synthetic_prompts(args.requests, args.prompt_len)
    out = asyncio.run(run(backend, settings, prompts, args.max_tokens, Path(args.out), warmup=args.warmup,
                          profiler=args.profiler, pyspy=(args.pyspy_bin, args.seconds, args.rate)))
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
