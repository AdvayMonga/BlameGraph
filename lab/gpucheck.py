"""What lab/RUNTIME.md's jails need from a real GPU host, checked on one: run on the GPU VM, as root the way the remote
worker runs (`sudo`), with the engine's venv first on PATH. Each check prints PASS or FAIL with its evidence; the exit
status is the number of failures. `--only NAME,...` runs a subset; `--skip-build` leaves out the dependency build.

    sudo -n env HOME="$HOME" PATH="$HOME/inference-server/.venv/bin:$PATH" LAB_ENGINE_REPO="$HOME/inference-server" \\
        LAB_TARGET=targets/inference-server.toml python -m lab.gpucheck
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

from lab import engine, gpu, serve, worker
from lab.safety import container, grader

HOLD = "import torch, time; x = torch.empty(2**30, dtype=torch.uint8, device='cuda'); print('holding', flush=True); time.sleep(600)"
MEASURE = {"names": ["single_stream"], "split": "seen", "tier": "short", "seed": 0, "jailed": True, "passive": False}
results: list[tuple[str, bool, str]] = []


def check(name: str):
    def wrap(fn):
        fn.check = name
        return fn
    return wrap


def base_tree(root: Path) -> Path:
    """The worker's engine checkout as a two-commit tree (base, empty change): the shape every job gets."""
    tree = root / "base"
    shutil.copytree(engine.repo(), tree, ignore=shutil.ignore_patterns(".venv", ".git", "__pycache__", "*.pyc"))
    grader._commit(tree, "base", init=True)
    grader._commit(tree, "change")
    return tree


def holder(name: str, wb: bool, tmp: Path) -> subprocess.Popen:
    """A process holding 1 GiB of GPU memory: in a workbench container (`wb`) or on the host."""
    py = engine.python()
    if wb:
        argv = container.workbench_argv(name, tmp, [py, "-c", HOLD], {"HOME": str(tmp)}, [tmp], [engine.venv()], py)
    else:
        argv = [py, "-c", HOLD]
    p = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline and "holding" not in (p.stdout.readline() or ""):
        if p.poll() is not None:
            raise RuntimeError(f"the GPU holder exited {p.returncode}")
    return p


@check("jail_gpu")
def jail_gpu(tmp: Path) -> str:
    """A measurement room (no network, read-only root, uid 10001) sees the GPU."""
    argv = container.argv(container.new_name(), tmp, ["nvidia-smi", "-L"], {}, [tmp], [], engine.python(), gpu=True)
    out = subprocess.run(argv, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0 and "GPU 0" in out.stdout, out.stdout + out.stderr
    return out.stdout.strip()


@check("workbench")
def workbench(tmp: Path) -> str:
    """The workbench: the GPU, root, apt and pip over the internet, and no cloud metadata service."""
    script = ("cat /etc/resolv.conf; set -e; nvidia-smi -L; id -u; apt-get update -qq; apt-get install -y -qq --no-install-recommends "
              "python3-pip >/dev/null; pip3 install --quiet --break-system-packages six; python3 -c 'import six'; "
              "echo pip-ok; python3 -c \"import socket; socket.create_connection(('169.254.169.254', 80), timeout=3)\" "
              "&& echo METADATA-REACHABLE || echo metadata-blocked")
    argv = container.workbench_argv(container.new_name(), tmp, ["sh", "-c", script], {"HOME": str(tmp)}, [tmp], [],
                                    engine.python())
    out = subprocess.run(argv, capture_output=True, text=True, timeout=600, stdin=subprocess.DEVNULL)
    text = out.stdout + out.stderr
    assert out.returncode == 0 and "GPU 0" in text and "pip-ok" in text and "metadata-blocked" in text, text[-2000:]
    assert "\n0\n" in f"\n{out.stdout}", "not root inside the workbench (run this check as root, like the worker)"
    return "GPU, root, apt, pip; metadata blocked"


@check("quiet")
def quiet(tmp: Path) -> str:
    """`container.quiet` kills the workbench's GPU holder, freezes the rest, and thaws it after."""
    name = container.new_name()
    p = holder(name, True, tmp)
    try:
        mine = container.pids(name)
        assert any(a["pid"] in mine for a in gpu.compute_apps() or []), "the holder is not on the GPU"
        with container.quiet(name) as killed:
            state = subprocess.run(["docker", "inspect", "-f", "{{.State.Paused}}", name], capture_output=True,
                                   text=True).stdout.strip()
            left = [a for a in gpu.compute_apps() or [] if a["pid"] in mine]
        after = subprocess.run(["docker", "inspect", "-f", "{{.State.Paused}}", name], capture_output=True,
                               text=True).stdout.strip()
        assert killed and not left and state == "true" and after == "false", (killed, left, state, after)
        return f"killed {killed}; paused during, thawed after"
    finally:
        p.kill()
        container.remove(name)


@check("exclusive")
def exclusive(tmp: Path) -> str:
    """A clean measurement is not contaminated (only the engine's processes on the GPU); a foreign one is refused."""
    tree = base_tree(tmp)
    clean = worker.Local().call("measure", {"tree": tree}, MEASURE, tmp / "clean")
    p = holder(None, False, tmp)
    try:
        worker.Local().call("measure", {"tree": tree}, MEASURE, tmp / "busy")
        raise AssertionError("measured with a foreign process on the GPU")
    except serve.GpuBusy as e:
        refused = e.apps
    finally:
        p.kill()
    v = clean["results"][0]
    return f"clean: {v['objective']} {v['value']:.4f} valid={v['valid']}; refused with {refused}"


@check("ncu")
def ncu(tmp: Path) -> str:
    """Nsight Compute inside a measurement room (needs the GPU's performance counters for the jail's user)."""
    r = worker.Local().call("profile", {"tree": base_tree(tmp)}, {"kind": "kernel", "args": {"launch_count": 2}},
                            tmp / "out")
    assert not r.get("refused") and r["returncode"] == 0 and r["head"], r.get("refused") or r["output"][-1500:]
    return r["head"][:300]


@check("hostprof")
def hostprof(tmp: Path) -> str:
    """py-spy inside a measurement room."""
    r = worker.Local().call("profile", {"tree": base_tree(tmp)}, {"kind": "hostprof", "args": {"seconds": 5}},
                            tmp / "out")
    assert not r.get("refused") and r["returncode"] == 0 and r["head"], r.get("refused") or r["output"][-1500:]
    return r["head"][:300]


@check("build")
def build_step(tmp: Path) -> str:
    """A dependency change is built from scratch (index-only network) and the built venv runs torch on the GPU."""
    from lab import build
    tree = tmp / "base"
    shutil.copytree(engine.repo(), tree, ignore=shutil.ignore_patterns(".venv", ".git", "__pycache__", "*.pyc"))
    grader._commit(tree, "base", init=True)
    uv = shutil.which("uv") or str(Path.home() / ".local" / "bin" / "uv")
    subprocess.run([uv, "add", "--no-sync", "six"], cwd=tree, check=True, capture_output=True)
    grader._commit(tree, "change")
    assert build.changed(tree), "the change to pyproject.toml/uv.lock was not seen"
    t0 = time.monotonic()
    py = build.python_for(tree)
    out = subprocess.run([py, "-c", "import six, torch; print(torch.__version__, torch.cuda.is_available())"],
                         capture_output=True, text=True)
    assert out.returncode == 0 and out.stdout.strip().endswith("True"), out.stdout + out.stderr
    return f"built in {time.monotonic() - t0:.0f}s: {py}: torch {out.stdout.strip()}"


CHECKS = [jail_gpu, workbench, quiet, exclusive, ncu, hostprof, build_step]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m lab.gpucheck", description=__doc__.split("\n\n")[0])
    ap.add_argument("--only", help="comma-separated: " + ",".join(c.check for c in CHECKS))
    ap.add_argument("--skip-build", action="store_true")
    a = ap.parse_args(argv)
    want = set(a.only.split(",")) if a.only else {c.check for c in CHECKS} - ({"build"} if a.skip_build else set())
    print(f"as uid {os.getuid()}; GPU: {gpu.query()}", flush=True)
    for c in CHECKS:
        if c.check not in want:
            continue
        tmp = Path(tempfile.mkdtemp(prefix=f"gpucheck-{c.check}-"))
        t0 = time.monotonic()
        try:
            detail, ok = c(tmp), True
        except Exception as e:
            detail, ok = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}", False
        results.append((c.check, ok, detail))
        print(f"{'PASS' if ok else 'FAIL'} {c.check} ({time.monotonic() - t0:.0f}s): {detail}", flush=True)
        shutil.rmtree(tmp, ignore_errors=True)
    print(json.dumps({n: ok for n, ok, _ in results}))
    return sum(not ok for _, ok, _ in results)


if __name__ == "__main__":
    sys.exit(main())
