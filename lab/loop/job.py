"""The run on this VM, start to end: `lab run` starts it detached and sends the secrets on stdin (JSON); the loop
saves to the store after every turn and stops the VM at its end, and so does the deadline.

  python -m lab.loop.job --run RUN --task TASK.toml [--n N|loop] [--budget USD] --vm NAME --instance ID
                         --deadline EPOCH  < env.json
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import subprocess
import sys
import threading
import time
import traceback

from lab.loop import loop, record
from lab.loop.pick import pick
from lab.loop.push import push
from lab.loop.turn import CHANGE, KERNEL, turn
from lab.records import ledger, sync
from lab.targets import engine

MARGIN_S = 600.0        # the deadline's stop starts this long before the deadline


def unwired(gap, best, work, budget):
    raise NotImplementedError(f"no way is wired for the {gap.layer} gap")


WAYS = {KERNEL: unwired, CHANGE: unwired}


def parts(provider, instance: str) -> loop.Parts:
    """The loop's parts on this VM: the model's pick, WAYS, pull requests on the engine repo, this VM's stop."""
    from lab.kernels.search.llm import Claude
    return loop.Parts(functools.partial(pick, ask=Claude()), functools.partial(turn, ways=WAYS),
                      lambda c, body, branch, onto: push(c, body, engine.repo(), branch, onto),
                      lambda: provider.stop_id(instance))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m lab.loop.job", description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--task", required=True, help="task spec TOML, relative to this repo")
    ap.add_argument("--n")
    ap.add_argument("--budget", type=float)
    ap.add_argument("--vm", required=True)
    ap.add_argument("--instance", required=True, help="this VM's instance id, which it stops at the end")
    ap.add_argument("--deadline", type=float, required=True, help="epoch seconds by which the VM is stopped")
    a = ap.parse_args(argv)
    os.environ.update(json.loads(sys.stdin.read() or "{}"))       # the secrets: in this process's memory only
    from evals.targets import target
    from lab.budget import gpu_rate
    from lab.infra.cloud import providers
    from lab.infra.runner import worker
    from lab.kernels.run.job import detach
    from lab.records.remote import open_remote
    from lab.task import Task
    remote, provider = open_remote(), providers.load(os.environ.get("LAB_VM_PROVIDER", "nebius"))
    work = engine.ENV_ROOT / "local" / "runs" / "loop" / a.run
    work.mkdir(parents=True, exist_ok=True)
    if detach(work / "job.log"):
        print(f"started {a.run}", flush=True)
        return 0

    def deadline() -> None:
        try:
            record.record(remote, a.run, {"state": "stopped", "reason": "deadline reached"})
        finally:
            provider.stop_id(a.instance)
    timer = threading.Timer(max(0.0, a.deadline - MARGIN_S - time.time()), deadline)
    timer.daemon = True
    timer.start()
    try:
        sync.pull_ledger(ledger.ROOT, remote)
        env = {"LAB_TARGET": os.environ["LAB_TARGET"], "LAB_ENGINE_REPO": str(engine.repo())}
        wk = worker.Remote([], str(engine.ENV_ROOT), engine.python(), env, sudo=True)
        onto = subprocess.run(["git", "-C", str(engine.repo()), "rev-parse", "--abbrev-ref", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
        args = (loop.plan(a.run, a.n, a.budget, gpu_rate()[0]), Task.parse(engine.ENV_ROOT / a.task), target.load(),
                wk, parts(provider, a.instance), remote, work, onto)
    except BaseException as e:                   # noqa: BLE001  the VM is stopped whatever happened
        traceback.print_exc()
        record.finish(remote, a.run, {"state": "failed", "reason": f"{type(e).__name__}: {e}"[-2000:]}, [],
                      lambda: provider.stop_id(a.instance))
        return 0
    try:
        loop.run(*args)
    except BaseException:                        # noqa: BLE001  the loop finished the run and stopped the VM
        traceback.print_exc()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
