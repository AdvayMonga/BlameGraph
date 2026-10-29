"""Where each engine's time goes, at fixed load points: every engine of engines.toml served as the comparison serves it,
under nsys (`nsys launch`, kernels inside CUDA graphs traced), driven by the comparison's client at each point of
POINTS; at each point a WINDOW_S nsys capture once the load is steady, then WINDOW_S of py-spy over the server's
processes, then the client's own numbers for the point. Run on the GPU VM, from the BlameGraph checkout with the
engine's python, with nothing else on the GPU; `python -m compare.gap DIR` reads the result.

    python -m compare.profile --out ~/gap [--engines ours,vllm,trtllm] [--precision fp8] [--points decode1,sat256]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from compare import run as R

WARM_S, WINDOW_S = 30.0, 10.0
NSYS_TRACE = "--trace=cuda,nvtx --sample=none --cpuctxsw=none --cuda-graph-trace=node"

# name: (loop, pool class or None for every class, concurrency or req/s, fixed answer length or None for the corpus's)
POINTS = {
    "decode1": ("closed", "steady_interactive", 1, None),        # single stream
    "sat256": ("closed", None, 256, None),                     # the saturated mix
    "long512": ("closed", "short", 512, 1024),                 # short prompts, 1,024-token answers: large-batch decode
    "lp8": ("open", "long_context", 8.0, 64),                  # long prompts, short answers
}


def _pool(ctx, cls: str | None, length: int | None) -> list[dict]:
    from regimes import suite
    if cls == "short":
        allp = ctx.pool(None)
        pool = [r for r in allp if (r.get("build_prompt_tokens") or 0) <= suite.SHORT_PROMPT_TOKENS and not r.get("messages")]
    else:
        pool = ctx.pool(cls)
    pool = random.Random(0).sample(pool, len(pool))
    return [{**r, "expected_output_tokens": length} for r in pool] if length else pool


def _drive(ctx, point, duration_s: float, rows: list):
    from regimes import workload as W
    loop, cls, load, length = POINTS[point]
    pool = _pool(ctx, cls, length)
    if loop == "closed":
        rows.extend(ctx.run_closed(W.Cycle(pool, point), int(load), duration_s)[0])
    else:
        rows.extend(ctx.run_open(W.poisson(pool, load, duration_s, 0, max_tokens=length)))


def _nsys() -> tuple[str, str]:
    """The host's nsys and the install dir that holds it (mounted at the same path into a container)."""
    exe = os.path.realpath(shutil.which("nsys") or "/usr/local/cuda/bin/nsys")
    return exe, str(Path(exe).parents[1])


def _server_pid(engine: str, srv) -> int:
    if engine in ("ours",) or not R._sh("docker", "ps", "-q", "-f", f"name=bgcmp-{engine}"):
        return srv.proc.pid
    return int(R._sh("docker", "inspect", "-f", "{{.State.Pid}}", f"bgcmp-{engine}"))


def profile(out: Path, engines: list[str], precision: str, points: list[str]) -> None:
    from lab import target
    from regimes import runner, suite
    from regimes.__main__ import _cpus
    spec, t = R.load(), target.load()
    c = spec["common"]
    exe, root = _nsys()
    for engine in engines:
        d = out / f"{engine}-{precision}"
        d.mkdir(parents=True, exist_ok=True)
        if all((d / f"{p}.json").exists() for p in points):
            continue
        docker = spec["engines"][engine]["kind"] == "docker"
        launch = f"{exe} launch --session-new=bg {NSYS_TRACE}"
        ctl = (lambda *a: R._sh("docker", "exec", f"bgcmp-{engine}", exe, *a)) if docker else (lambda *a: R._sh(exe, *a))
        where = "/prof" if docker else str(d)
        try:
            with R.Server(spec, engine, precision, d, prefix=launch, mounts=f"-v {root}:{root}:ro -v {d}:/prof ") as srv:
                url = f"http://127.0.0.1:{R.PORT}"
                ctx = suite.Ctx.from_target(t, url, client_procs=8, client_cpus=_cpus(c["client_cpus"]))
                for point in points:
                    if (d / f"{point}.json").exists():
                        continue
                    rows: list = []
                    total = WARM_S + 2 * WINDOW_S + 5
                    load = threading.Thread(target=_drive, args=(ctx, point, total, rows), daemon=True)
                    t0 = time.time()
                    load.start()
                    time.sleep(WARM_S)
                    ctl("start", "--session=bg", f"--output={where}/{point}", "--force-overwrite=true")
                    time.sleep(WINDOW_S)
                    ctl("stop", "--session=bg")
                    pid = _server_pid(engine, srv)
                    subprocess.run(["sudo", shutil.which("py-spy") or str(Path(sys.executable).with_name("py-spy")), "record", "--pid", str(pid),
                                    "--subprocesses", "--nonblocking", "--duration", str(int(WINDOW_S)),
                                    "--rate", "100", "--format", "speedscope", "-o", str(d / f"{point}.pyspy.json")],
                                   capture_output=True, timeout=WINDOW_S + 120)
                    load.join()
                    summary = runner.summarize(rows, ctx.conversational, time.time() - t0)
                    (d / f"{point}.json").write_text(json.dumps({"point": point, "def": POINTS[point],
                                                                 "summary": summary, "facts": R.facts()}) + "\n")
                    print(f"{engine}-{precision} {point}: {summary['output_tok_per_s']:.0f} tok/s, TPOT p50 "
                          f"{(summary['tpot_p50_s'] or 0) * 1e3:.1f} ms, ok {summary['ok']}/{summary['n']}", flush=True)
        except RuntimeError as err:
            (d / "error.txt").write_text(f"{err}\n")
            print(f"{engine}-{precision}: {err}", file=sys.stderr)
        for rep in sorted(d.glob("*.nsys-rep")):
            if not rep.with_suffix(".sqlite").exists():
                R._sh(exe, "export", "--type=sqlite", f"--output={rep.with_suffix('.sqlite')}", str(rep))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--engines", default="ours,vllm,trtllm")
    ap.add_argument("--precision", default="fp8")
    ap.add_argument("--points", default=",".join(POINTS))
    a = ap.parse_args()
    profile(a.out, a.engines.split(","), a.precision, a.points.split(","))
