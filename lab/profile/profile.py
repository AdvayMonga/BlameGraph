"""Profile the served engine: its own launch command (after an instrument's, if given) runs with lab/profile/hook on
PYTHONPATH, so every Python process it starts joins lab.profile.window; the load generator warms it, then sends the
window's requests while the window is open. The bundle gets each process's profiler files (procs/<pid>/) and their
kernel and op tables merged, the engine's counters at the window's ends, its telemetry, device samples and log, and
provenance.

  python -m lab.profile.profile (--corpus-class C | --synthetic) [--requests N] [--max-tokens N] [--out DIR] ...
"""

from __future__ import annotations

import argparse
import contextlib
import itertools
import json
import os
import platform
import random
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from evals.loadgen import runner
from evals.loadgen.client import Request
from evals.targets import target
from lab.infra.runner import serve
from lab.profile import bundle, gpu
from lab.targets import engine

WINDOW_S = 300           # the longest the engine may take to open or close the window
SETTLE_S = 1.0           # the window is open once no further process has joined it for this long
EXIT_S = 600             # the longest the engine, and an instrument writing its report, may take to end after it
CAPTURE_BYTES = 64 * 2**20


def synthetic(n: int, length: int, max_tokens: int, seed: int, vocab: int = 1000) -> list[Request]:
    rng = random.Random(seed)
    return [Request(f"s{seed}-{i}", [], max_tokens, ignore_eos=True,
                    prompt_ids=[rng.randrange(1, vocab) for _ in range(length)]) for i in range(n)]


def workload(n: int, max_tokens: int, seed: int = 0, corpus_class: str | None = None, prompt_len: int | None = None,
             warmup: int | None = None, corpus_dir: Path | None = None) -> tuple[list[Request], list[Request], dict]:
    """The window's `n` requests, `warmup` more (default n) drawn with the next seed, all run to `max_tokens`, and
    their source: random token ids of `prompt_len` if given, else seen-split requests of `corpus_class`."""
    warmup = n if warmup is None else warmup
    if prompt_len is not None:
        return (synthetic(n, prompt_len, max_tokens, seed), synthetic(warmup, prompt_len, max_tokens, seed + 1),
                {"kind": "synthetic", "prompt_len": prompt_len, "seed": seed})
    from evals import regimes as corpus
    d = corpus_dir or corpus.CORPUS_DIR

    def chats(seed, k):
        w = corpus.sample(corpus_class, "seen", seed, k, d)
        return [Request(f"c{seed}-{i}", m, max_tokens, ignore_eos=True) for i, m in enumerate(w["messages"])], w["source"]
    (window, source), (warm, _) = chats(seed, n), chats(seed + 1, warmup)
    return window, warm, source


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read())


def delta(start, end):
    """end - start for every number both hold, nested dicts included."""
    if isinstance(start, dict) and isinstance(end, dict):
        d = {k: delta(start[k], end[k]) for k in start.keys() & end.keys()}
        return {k: v for k, v in d.items() if v is not None}
    num = (int, float)
    if isinstance(start, num) and isinstance(end, num) and not isinstance(start, bool) and not isinstance(end, bool):
        return end - start
    return None


def _await(srv: serve.Served, out: Path, done, what: str) -> None:
    """Poll until `done()`; raise on a process's error.txt, the engine's exit or WINDOW_S."""
    deadline = time.monotonic() + WINDOW_S
    while not done():
        for e in out.glob("procs/*/error.txt"):
            raise RuntimeError(f"the profiling window failed in engine process {e.parent.name}:\n{e.read_text()[-2000:]}")
        if srv.proc.poll() is not None:
            raise RuntimeError(f"the engine exited with {srv.proc.returncode} while {what}")
        if time.monotonic() > deadline:
            raise TimeoutError(f"no engine process answered within {WINDOW_S}s while {what}")
        time.sleep(0.05)


def _open(srv: serve.Served, out: Path) -> list[Path]:
    """Open the window; the dirs of the processes taking part, once each has started and none has joined for
    SETTLE_S."""
    (out / "start").write_text(str(time.time()))
    seen = {"n": 0, "at": time.monotonic()}

    def settled():
        joined = list(out.glob("procs/*"))
        if len(joined) != seen["n"]:
            seen.update(n=len(joined), at=time.monotonic())
        return (joined and all((p / "started").exists() for p in joined)
                and time.monotonic() - seen["at"] >= SETTLE_S)
    _await(srv, out, settled, "opening the window")
    return sorted(p.parent for p in out.glob("procs/*/started"))


def _close(srv: serve.Served, out: Path, procs: list[Path]) -> None:
    (out / "stop").write_text(str(time.time()))
    _await(srv, out, lambda: all((p / "window.json").exists() for p in procs), "closing the window")


def outcomes(rows: list) -> dict:
    """Per-request outcome counts and output lengths."""
    toks = [r.n_tokens for r in rows if r.status == "ok"]
    return {"errors": len(rows) - len(toks), "tokens_out": ({"total": sum(toks), "min": min(toks),
                                                            "mean": sum(toks) / len(toks), "max": max(toks)}
                                                           if toks else None)}


def run(tree: Path, window: list[Request], warmup: list[Request], out: Path, *, source: dict, profiler: str = "torch",
        capture: list[str] = (), capture_bytes: int = CAPTURE_BYTES, pyspy: tuple[str, float, int, list[str]] | None = None,
        concurrency: int | None = None, prefix: list[str] = (), read: list[Path] = (), env: dict | None = None,
        harness: Path = engine.ENV_ROOT, jailed: bool = True, timeout_s: float = 3600) -> Path:
    """Serve `tree` under the profiler and return the bundle directory, made in `out`. `warmup` runs first, all at once;
    then the window's requests, closed loop at `concurrency` in flight (default all), repeated for py-spy's seconds.
    `profiler`: torch (torch.profiler trace), cuda-range (the window marked for nsys and ncu), pyspy (`pyspy` = exe,
    seconds, rate, extra flags). `capture`: "module:function" entries recorded in the window, or the engine's own
    region capture (region:FIRST:LAST, its CAPTURE_REGION) from launch on. `prefix`: an instrument's command before the
    engine's; `harness` is where lab.profile imports from in the engine."""
    t = target.load()
    out = bundle.new_dir(out)
    env = dict(env or {})
    cfg = {"dir": str(out), "profiler": profiler, "capture": list(capture), "capture_bytes": capture_bytes,
           "pyspy": dict(zip(("exe", "seconds", "rate", "extra"), pyspy)) if pyspy else None}
    path = os.pathsep.join([str(harness), str(Path(harness) / "lab" / "profile" / "hook"), str(Path(tree) / "src"),
                            str(tree)])
    load = {"timeout": timeout_s, "chat_kwargs": t.chat_kwargs}
    with serve.Served(tree, t.engine, log=out.parent / "engine.log", jailed=jailed, artifacts=out,
                      env_extra={**env, "PYTHONPATH": path, "LAB_PROFILE": json.dumps(cfg)}, prefix=prefix,
                      read=read) as srv:
        w0 = time.time()
        if warmup:
            runner.run_closed(srv.url, t.model, warmup, len(warmup), **load)
        w1 = time.time()
        stats0 = _get(srv.url + t.engine.stats) if t.engine.stats else None
        gpu0 = gpu.query()
        procs = _open(srv, out)
        t0 = time.time()
        lanes = concurrency or len(window)
        if pyspy:
            rows, _ = runner.run_closed(srv.url, t.model, itertools.cycle(window), lanes, duration_s=pyspy[1], **load)
        else:
            rows, _ = runner.run_closed(srv.url, t.model, window, lanes, **load)
        t1 = time.time()
        stats1 = _get(srv.url + t.engine.stats) if t.engine.stats else None
        gpu1 = gpu.query()
        _close(srv, out, procs)
        (out / "end").write_text(str(time.time()))
        with contextlib.suppress(subprocess.TimeoutExpired):
            srv.proc.wait(EXIT_S)
        srv.exclusive()
        ready_s = srv.ready_s
    for f in ("start", "stop", "end", *(p / "started" for p in procs)):
        (out / f).unlink(missing_ok=True)
    for name, merge in (("kernels.json", bundle.merge_kernels), ("ops.json", bundle.merge_ops)):
        tables = [json.loads(f.read_text()) for f in sorted(out.glob(f"procs/*/{name}"))]
        merged = merge(tables) if tables else None
        if merged is not None:
            bundle.write_json(out / name, merged)
    if stats0 is not None:
        bundle.write_json(out / "stats.json", {"window_start": stats0, "window_end": stats1,
                                               "window_delta": delta(stats0, stats1)})
    done = outcomes(rows)
    wl = [r.prompt_ids if r.prompt_ids is not None else r.messages for r in window]
    bundle.write_json(out / "meta.json", {
        "git_sha": bundle.git_sha(), "python": sys.version.split()[0], "platform": platform.platform(),
        "engine": {"launch": t.engine.launch, "env": {**t.engine.env, **env}, "ready_s": ready_s},
        "profiler": {"kind": profiler, "processes": [json.loads((p / "window.json").read_text()) for p in procs]},
        "workload": source, "workload_hash": bundle.workload_hash(wl, window[0].max_tokens if window else 0),
        "requests": len(window), "max_tokens": window[0].max_tokens if window else 0, "concurrency": lanes,
        "rounds": len(rows) / len(window) if window else 0,
        "window": {"start": t0, "end": t1, "wall_s": t1 - t0}, "failed": done["errors"], "outcomes": done,
        "warmup": {"requests": len(warmup), "concurrent": True, "wall_s": w1 - w0},
        "gpu": {"window_start": gpu0, "window_end": gpu1}})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--corpus-class", help="seen-split requests drawn from this corpus class")
    src.add_argument("--synthetic", action="store_true", help="random token ids of --prompt-len")
    ap.add_argument("--corpus-dir", help="default: this repo's evals/regimes/")
    ap.add_argument("--tree", help="the engine tree to serve; default the target's engine repo")
    ap.add_argument("--jailed", action="store_true", help="serve it in the jail")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--requests", type=int, default=8)
    ap.add_argument("--concurrency", type=int, help="the window's requests closed loop at this many in flight; "
                                                    "default: all at once")
    ap.add_argument("--prompt-len", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--warmup", type=int, help="warmup requests, sent together; default: as many as the window")
    ap.add_argument("--out", default="local/runs/profile")
    ap.add_argument("--profiler", choices=("torch", "cuda-range", "pyspy"), default="torch",
                    help="torch: torch.profiler trace; cuda-range: mark the window for nsys/ncu; pyspy: sample with py-spy")
    ap.add_argument("--seconds", type=float, default=20.0, help="pyspy: how long to sample under load")
    ap.add_argument("--rate", type=int, default=100, help="pyspy: samples per second")
    ap.add_argument("--pyspy-bin", default="py-spy")
    ap.add_argument("--pyspy-arg", action="append", default=[], help="pyspy: extra `py-spy record` flag (repeatable)")
    ap.add_argument("--capture", action="append", default=[], metavar="MODULE:FUNCTION",
                    help="record every call's argument shapes in the window and the first call per shape whole "
                         "(repeatable); region:FIRST:LAST records that stretch of the engine's decode stations (its "
                         "CAPTURE_REGION) from launch on; it synchronizes, so the window's times are not representative")
    ap.add_argument("--capture-bytes", type=int, default=CAPTURE_BYTES, help="largest call kept whole as a sample")
    a = ap.parse_args(argv)
    window, warm, source = workload(a.requests, a.max_tokens, a.seed, a.corpus_class,
                                    a.prompt_len if a.synthetic else None, a.warmup,
                                    Path(a.corpus_dir) if a.corpus_dir else None)
    print(run(Path(a.tree) if a.tree else target.load().engine_repo, window, warm, Path(a.out), source=source,
              profiler=a.profiler, capture=a.capture,
              capture_bytes=a.capture_bytes, pyspy=(a.pyspy_bin, a.seconds, a.rate, a.pyspy_arg) if a.profiler == "pyspy"
              else None, concurrency=a.concurrency, jailed=a.jailed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
