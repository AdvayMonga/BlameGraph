"""The engine comparison, on the GPU VM: every (engine, precision) of engines.toml served one at a time on the same GPU
and cores, each measured by the same client: the client check once, then per configuration a launch check (the model
answers, a fixed-length answer runs to its length), the regimes, equiv once, and cold start. Repeats run the
configurations in a rotated order, so drift over the session falls on every engine alike. Every stage writes one file
and is skipped when that file exists, so a rerun resumes. Run from the BlameGraph checkout with the engine's python:

    python -m compare.run --out ~/compare [--engines ours,vllm,sglang,trtllm] [--precisions bf16,fp8]
                          [--stages clientcheck,launch,regimes,equiv,cold] [--repeats 1] [--regimes all] [--tol 0.02]
    # more repeats of the contested regimes, numbered after the first pass's:
    python -m compare.run --out ~/compare --stages launch,regimes --regimes bursty,overload --repeats 2 --first-repeat 1
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import tomllib
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PORT = 8100
STAGES = ("clientcheck", "launch", "regimes", "equiv", "cold")


def load(path: Path = HERE / "engines.toml") -> dict:
    return tomllib.loads(path.read_text())


def command(spec: dict, engine: str, precision: str, out: Path, cache: str) -> str:
    """The shell command that serves `engine` at `precision`; `cache` is the dir the engine keeps compile caches in."""
    c, e = spec["common"], spec["engines"][engine]
    env = e["precision"][precision]
    model = e.get("checkpoint") or spec["precision"][precision]["checkpoint"]
    fill = dict(c, model=model, port=PORT, env=env, cfg="/cfg/llm_api.yaml")
    if e["kind"] == "venv":
        engine_dir = os.environ.get("LAB_ENGINE_REPO", os.path.expanduser("~/inference-server"))
        return f"cd {engine_dir} && exec taskset -c {c['engine_cpus']} env " + e["cmd"].format(**fill)
    args = e["cmd"].format(**fill)
    cfg = ""
    if "{cfg}" in e["cmd"]:
        (out / "llm_api.yaml").write_text(env.strip() + "\n")
        args = e["cmd"].format(**{**fill, "env": ""})
        cfg = f"-v {out}:/cfg:ro "
    hf = os.path.expanduser("~/.cache/huggingface")
    return (f"exec docker run --rm --init --name bgcmp-{engine} --gpus all --network host --ipc host "
            f"--cpuset-cpus {c['engine_cpus']} -e HF_HUB_OFFLINE=1 -e TRITON_CACHE_DIR=/root/.cache/triton "
            f"-v {cache}:/root/.cache -v {hf}:/root/.cache/huggingface {cfg}{e['image']} {args}")


def _sh(*args: str) -> str:
    """A command's stdout; empty where the tool is missing (a laptop has no nvidia-smi)."""
    try:
        return subprocess.run(args, capture_output=True, text=True).stdout.strip()
    except FileNotFoundError:
        return ""


def _get(url: str, timeout: float = 5.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.status, r.read()


def gpu_free(timeout_s: float = 300.0) -> bool:
    """Until no process holds GPU memory (a stopped container can take a while to let go)."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        used = _sh("nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits").split()
        if used and int(used[0]) < 1024:
            return True
        time.sleep(3)
    return False


def facts() -> dict:
    """What the GPU and host were during a configuration's runs."""
    return {"gpu": _sh("nvidia-smi", "--query-gpu=name,driver_version,clocks.sm,clocks.max.sm,clocks.mem,power.limit,"
                     "temperature.gpu,clocks_throttle_reasons.active", "--format=csv,noheader"),
            "gpu_uuid": _sh("nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"),   # a spot VM can move hosts
            "cpus": os.cpu_count(), "kernel": _sh("uname", "-r"), "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


class Server:
    def __init__(self, spec, engine, precision, out: Path):
        self.engine, self.out = engine, out
        cache = Path(os.path.expanduser(f"~/compare-cache/{engine}-{precision}"))
        cache.mkdir(parents=True, exist_ok=True)
        self.cmd = command(spec, engine, precision, out, str(cache))
        self.health = f"http://127.0.0.1:{PORT}{spec['engines'][engine]['health']}"
        self.proc = None

    def __enter__(self):
        if not gpu_free():
            raise RuntimeError("the GPU is still held by another process")
        log = open(self.out / "server.log", "a")
        self.started = time.time()
        self.proc = subprocess.Popen(self.cmd, shell=True, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        while time.time() - self.started < 1800:
            if self.proc.poll() is not None:
                raise RuntimeError(f"{self.engine} exited with {self.proc.returncode}; see {self.out / 'server.log'}")
            try:
                if _get(self.health)[0] == 200:
                    self.ready_s = time.time() - self.started
                    return self
            except OSError:
                pass
            time.sleep(2)
        raise RuntimeError(f"{self.engine} was not healthy within 30 min")

    def __exit__(self, *exc):
        if self.proc and self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
        _sh("docker", "rm", "-f", f"bgcmp-{self.engine}")
        gpu_free()


def launch_check(url: str, model: str, chat_kwargs: dict) -> dict:
    """The model answers "17 + 25" with 42, and a fixed-length answer runs to exactly its length with ignore_eos."""
    from regimes.client import Request, send, wall_clock
    from regimes.suite import PROBE
    clock = wall_clock()
    probe = send(url, model, Request("check-42", PROBE, 16), clock(), clock, chat_kwargs=chat_kwargs)
    fixed = send(url, model, Request("check-len", [{"role": "user", "content": "Say hi."}], 300, ignore_eos=True),
                 clock(), clock, chat_kwargs=chat_kwargs)
    ok = probe.status == "ok" and "42" in probe.text and fixed.status == "ok" and fixed.n_tokens == 300
    return {"ok": ok, "answer": probe.text, "probe": probe.status, "fixed": fixed.status, "fixed_tokens": fixed.n_tokens,
            "fixed_finish": fixed.finish_reason, "detail": probe.detail or fixed.detail}


def _python(*args, env=None) -> int:
    return subprocess.run([sys.executable, "-m", *args], cwd=ROOT, env={**os.environ, **(env or {})}).returncode


def run(out: Path, engines: list[str], precisions: list[str], stages: list[str], repeats: int, regimes: str,
        tol: float, spec: dict | None = None, first_repeat: int = 0) -> None:
    spec = spec or load()
    c = spec["common"]
    from lab import target
    t = target.load()
    out.mkdir(parents=True, exist_ok=True)
    if "clientcheck" in stages and not (out / "clientcheck.json").exists():
        _python("regimes.clientcheck", "--client-cpus", c["client_cpus"], "--server-cpus", c["engine_cpus"],
                "--out", str(out / "clientcheck.json"))
    configs = [(e, p) for e in engines for p in precisions]
    for rep in range(first_repeat, first_repeat + repeats):
        order = configs[rep % len(configs):] + configs[:rep % len(configs)]
        for engine, precision in order:
            d = out / f"{engine}-{precision}"
            d.mkdir(exist_ok=True)
            todo = [s for s in ("launch", "regimes", "equiv") if s in stages and not _done(d, s, rep)]
            if todo:
                try:
                    with Server(spec, engine, precision, d) as srv:
                        url = f"http://127.0.0.1:{PORT}"
                        if "launch" in todo:
                            res = {**launch_check(url, c["served_name"], t.chat_kwargs), "ready_s": srv.ready_s,
                                   "cmd": srv.cmd, "facts": facts(), "spec": spec["engines"][engine]}
                            if engine == "ours":
                                res["engine_commit"] = _sh("git", "-C", os.environ.get(
                                    "LAB_ENGINE_REPO", os.path.expanduser("~/inference-server")), "rev-parse", "HEAD")
                            (d / "launch.json").write_text(json.dumps(res, indent=1) + "\n")
                            if not res["ok"]:
                                print(f"{engine}-{precision}: launch check failed: {res}", file=sys.stderr)
                                continue
                        if "regimes" in todo:
                            _python("regimes", "run", regimes, "--url", url, "--tier", "short", "--tol", str(tol),
                                    "--client-procs", "8", "--client-cpus", c["client_cpus"],
                                    *(["--tokenizer", c["tokenizer"]] if c.get("tokenizer") else []),
                                    "--out", str(d / f"regimes-{rep}.json"))
                            (d / f"facts-{rep}.json").write_text(json.dumps(facts()) + "\n")
                        if "equiv" in todo:
                            _python("correctness", "candidate", "--url", url, "--api", spec["engines"][engine]["api"],
                                    "--concurrency", "32", "--out", str(d / "equiv.json"))
                except RuntimeError as err:
                    (d / f"error-{rep}.txt").write_text(f"{err}\n")
                    print(f"{engine}-{precision}: {err}", file=sys.stderr)
                    continue
            if "cold" in stages and rep == first_repeat and not (d / "cold.json").exists():
                subprocess.run("sync && echo 3 | sudo tee /proc/sys/vm/drop_caches", shell=True, capture_output=True)
                if gpu_free():
                    cmd = command(spec, engine, precision, d, '"$(dirname "$VLLM_CACHE_ROOT")"')
                    _python("regimes", "cold-start", "--url", f"http://127.0.0.1:{PORT}", "--cmd", cmd,
                            "--log-dir", str(d), "--out", str(d / "cold.json"))
                    _sh("docker", "rm", "-f", f"bgcmp-{engine}")


def _done(d: Path, stage: str, rep: int) -> bool:
    if stage == "launch":
        return (d / "launch.json").exists() and json.loads((d / "launch.json").read_text())["ok"]
    return (d / {"regimes": f"regimes-{rep}.json", "equiv": "equiv.json"}[stage]).exists()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--engines", default="ours,vllm,sglang,trtllm")
    ap.add_argument("--precisions", default="bf16,fp8")
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--first-repeat", type=int, default=0, help="number new repeats from here (more of a contested regime)")
    ap.add_argument("--regimes", default="single_stream,saturated,bursty,long_prompt_short_output,"
                    "short_prompt_long_output,shared_prefix_multi_turn,overload")
    ap.add_argument("--tol", type=float, default=0.02)
    a = ap.parse_args()
    run(a.out, a.engines.split(","), a.precisions.split(","), a.stages.split(","), a.repeats, a.regimes, a.tol,
        first_repeat=a.first_repeat)
