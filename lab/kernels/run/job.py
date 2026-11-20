"""The kernel search on this VM, start to end: the task's captures, hold, search, freeze, gate, patch. The run folder and
the ledger go to the store after every saved generation, at each stage and at the end; then the VM deletes itself,
also after a failure and before its deadline. `lab kernels run` starts it and sends the secrets on stdin (JSON).

  python -m lab.kernels.run.job --run RUN --task TASK --vm NAME --instance ID --deadline EPOCH  < env.json
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import sys
import threading
import time
import traceback
from pathlib import Path

from lab.kernels import run as runs
from lab.records import ledger, sync
from lab.targets import engine

SAVE_S = 60.0           # between checks for a newly saved generation
MARGIN_S = 600.0        # the end (save, delete) starts this long before the deadline


class Job:
    """One run's state and its store; `save` is safe from any thread."""

    def __init__(self, a, remote, provider):
        self.a, self.remote, self.provider = a, remote, provider
        self.out = runs.folder(a.run)
        self.state = {"run": a.run, "task": a.task, "vm": a.vm, "instance": a.instance, "deadline": a.deadline,
                      "state": "running", "stage": "start", "generation": -1, "updated": time.time()}
        self.lock, self.ending = threading.Lock(), threading.Lock()
        self.ended = threading.Event()

    def save(self, **changes) -> None:
        with self.lock:
            self.state.update(changes, updated=time.time())
            runs.save(self.remote, self.a.run, self.out, self.state)
            sync.push_ledger(ledger.ROOT, self.remote)

    def watch(self) -> None:
        """Save whenever the search saved a new generation, until the end."""
        while not self.ended.wait(SAVE_S):
            g = saved(self.out / "search")
            if g > self.state["generation"]:
                try:
                    self.save(generation=g)
                except Exception:                    # noqa: BLE001  the next check tries again
                    traceback.print_exc()

    def end(self, state: str, reason: str | None = None) -> None:
        """Save the final state, then delete this VM; the delete runs even when the save fails."""
        if not self.ending.acquire(blocking=False):      # the deadline's end and the search's: the first one only
            return
        self.ended.set()
        try:
            self.save(state=state, **({"reason": reason[-2000:]} if reason else {}))
        except Exception:                            # noqa: BLE001
            traceback.print_exc()
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            self.provider.delete_id(self.a.instance)


def saved(search_dir: Path) -> int:
    """The last generation the search saved (pipeline.saved)."""
    from lab.kernels.tasks import pipeline
    return pipeline.saved(search_dir)


def search(task_name: str, out: Path, stage) -> dict:
    """The kernel_search tool's pipeline on this host: the worker's jobs, the hold and the jailed evaluator run here as
    root, the search and its model calls in this process. Returns the result written to result.json."""
    from lab.evaltools import _layer_facts
    from lab.infra.runner import worker
    from lab.kernels.eval.ship import gate_passed
    from lab.kernels.tasks import load, pipeline
    task = load(task_name)
    tree = out / "tree"
    if not tree.is_dir():
        shutil.copytree(engine.repo(), tree, ignore=shutil.ignore_patterns(".venv", ".git", "__pycache__"))
    env = {"LAB_TARGET": os.environ["LAB_TARGET"], "LAB_ENGINE_REPO": str(engine.repo())}
    w = worker.Remote([], str(engine.ENV_ROOT), engine.python(), env, sudo=True)

    def profile(t, a):
        dest = out / "captures" / f"{a['corpus_class']}_{a['concurrency']}"
        shutil.rmtree(dest, ignore_errors=True)
        w.call("profile", {"tree": t}, {"kind": "profile", "args": a, "workbench": None}, dest)
        return dest / "files" if (dest / "files" / "captures").is_dir() else None

    stage("capture")
    caps = pipeline.captures(task, tree, profile, (), out / "captures.json")
    if not caps:
        raise RuntimeError("no capture traffic produced calls to the task's entry point")
    jail = pipeline.remote_env(["sh", "-c"], str(engine.ENV_ROOT), engine.python(), env, sudo=True)
    region = task.name.startswith("region:")
    stage("hold" if region else "search")
    with pipeline.hold(task, caps, out, jail) if region else contextlib.nullcontext() as held:
        start = out / "start.py"
        start.write_text(held["start"] if held else pipeline.start_program(task, tree))
        stage("search")
        found = pipeline.search(task, caps, start, out / "search", ledger.ROOT, jail, device="cuda", held=held,
                                facts_text=pipeline.facts(task, _layer_facts(ledger.ROOT, "kernel_tasks")))
        stage("gate")
        best = pipeline.freeze(task, caps, out / "search" / "best_program.py", out, jail, held, "cuda")
        g = pipeline.gate(task, caps, best, out, jail, held, "cuda")
    passed = gate_passed(g)
    if passed:
        (out / "winner.patch").write_text(pipeline.patch(task, tree, best.read_text(), held))
    result = {"best": found.get("best"), "search_record": found.get("ledger"), "cost": found.get("cost"),
              "gate": {"passed": passed, "metrics": g.get("metrics")}, "patch": "winner.patch" if passed else None}
    (out / "result.json").write_text(json.dumps(result, indent=1, default=str))
    return result


def detach(log: Path) -> bool:
    """Fork into a new session with stdio on `log`; True in the parent."""
    if os.fork():
        return True
    os.setsid()
    fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    null = os.open(os.devnull, os.O_RDONLY)
    os.dup2(null, 0)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m lab.kernels.run.job", description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--vm", required=True)
    ap.add_argument("--instance", required=True, help="this VM's instance id, which it deletes at the end")
    ap.add_argument("--deadline", type=float, required=True, help="epoch seconds by which the VM is deleted")
    a = ap.parse_args(argv)
    os.environ.update(json.loads(sys.stdin.read() or "{}"))       # the secrets: in this process's memory only
    from lab.infra.cloud import providers
    from lab.records.remote import open_remote
    remote, provider = open_remote(), providers.load(os.environ.get("LAB_VM_PROVIDER", "nebius"))
    out = runs.folder(a.run)
    if not out.exists():
        runs.restore(remote, a.run, out)
    out.mkdir(parents=True, exist_ok=True)
    if detach(out / "job.log"):
        print(f"started {a.run}", flush=True)
        return 0
    job = Job(a, remote, provider)
    sync.pull_ledger(ledger.ROOT, remote)
    threading.Thread(target=job.watch, daemon=True).start()
    timer = threading.Timer(max(0.0, a.deadline - MARGIN_S - time.time()), job.end, ("stopped", "deadline reached"))
    timer.daemon = True
    timer.start()
    try:
        search(a.task, out, lambda s: job.save(stage=s))
        job.end("done")
    except BaseException as e:                       # noqa: BLE001  the VM is deleted whatever happened
        traceback.print_exc()
        job.end("failed", f"{type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
