"""One run, start to finish: a GPU VM, the target's setup, the session, its feedback, the store. The VM is deleted
when the run ends, however it ends.

  lab run --task TASK.toml --budget USD [--target T]

`--budget` caps the session, and the VM is armed for the GPU time the same dollars rent.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import uuid

from evals.targets import target
from lab.records import ledger


def steps(task: str, budget: float, run_id: str) -> list[list[str]]:
    """The `lab` commands of a run, in order."""
    usd = f"{budget:g}"
    return [["vm", "start", "--budget", usd], ["vm", "setup"], ["setup", "--task", task],
            ["session", "--task", task, "--budget", usd, "--run", run_id],
            ["feedback", str(ledger.ROOT), "--run", run_id], ["store", "push"]]


def absent(budget: float) -> list[str]:
    """What a run needs that is absent here: what a session needs, and a GPU rate for the VM's budget."""
    from lab import session
    from lab.infra.cloud import vm
    from lab.infra.cloud.providers import ProviderError
    out = session.preflight()
    try:
        vm.minutes_for(budget)
    except ProviderError as e:
        out.append(str(e))
    return out


def lab(step: list[str]) -> int:
    print("lab", *step, flush=True)
    return subprocess.run([sys.executable, "-m", "lab", *step]).returncode


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="lab run", description=__doc__.split("\n")[0])
    target.add_argument(ap)
    ap.add_argument("--task", required=True, help="task spec TOML (lab/task.py)")
    ap.add_argument("--budget", type=float, required=True, help="dollars for the run")
    a = ap.parse_args(argv)
    if a.target:
        os.environ["LAB_TARGET"] = a.target
    missing = absent(a.budget)
    if missing:
        print("lab run: nothing was started.\n" + "\n".join(f"- {x}" for x in missing), file=sys.stderr)
        return 2
    os.environ["LAB_WORKER"] = "ssh"
    run_id = f"run-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    code = 0
    try:
        for step in steps(a.task, a.budget, run_id):
            code = lab(step)
            if code:
                break
    finally:
        lab(["vm", "stop"])
    return code


if __name__ == "__main__":
    raise SystemExit(main())
