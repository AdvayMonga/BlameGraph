"""One run: checked here, then started on its own GPU VM, which runs the loop, saves every turn to the store and stops
itself at the end. `lab status` reads the run and deletes its VM.

  lab run [--task TASK.toml] [--n N|loop] [--budget USD]

No `--n` is one turn; `--n loop` runs until `--budget` is spent; `--budget` stops any run early.
"""
from __future__ import annotations

import argparse
import sys
import time
import uuid
from pathlib import Path

from lab.targets import engine

JOB = "lab.loop.job"
TASK = "lab/tasks/single_stream.toml"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="lab run", description=__doc__.split("\n")[0])
    ap.add_argument("--task", default=TASK, help=f"task spec TOML inside this repo (default: {TASK})")
    ap.add_argument("--n", help="turns: a number, or `loop` (until the budget is spent)")
    ap.add_argument("--budget", type=float, help="dollars; the run stops before the turn after they are spent")
    a = ap.parse_args(argv)
    from lab.infra.cloud import vm
    from lab.kernels.run import launch
    from lab.loop import loop, record, status
    from lab.loop.setup import setup
    from lab.task import Task
    run = f"run-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    try:
        loop.plan(run, a.n, a.budget)
        task = Path(a.task).resolve().relative_to(engine.ENV_ROOT).as_posix()
        Task.parse(engine.ENV_ROOT / task)
    except (ValueError, OSError) as e:
        print(f"lab run: nothing was started. {e}", file=sys.stderr)
        return 2
    v = vm.VM()
    problems = setup(v)
    if problems:
        print("lab run: nothing was started.\n" + "\n".join(f"- {p.what}\n  fix: {p.fix}" for p in problems),
              file=sys.stderr)
        return 2
    from lab.records.remote import open_remote
    remote = open_remote()
    last = record.last_run(remote)
    if status.tidy(v, record.status(remote, last) if last else None):
        print(f"deleted {v.name}, the VM of {last}", flush=True)
    if v.provider.get(v.name) is not None:
        print(f"lab run: nothing was started. A VM named {v.name} already exists: `lab status` reads its run, "
              "`lab vm stop` deletes it.", file=sys.stderr)
        print(status.billing(v))
        return 2
    launch.go(v, run, [JOB, "--task", task, *(["--n", a.n] if a.n else []),
                       *(["--budget", f"{a.budget:g}"] if a.budget is not None else [])])
    print(f"{run}: running on {v.name}; `lab status` reads it. The VM stops itself at the end; `lab status` then "
          "deletes it.")
    print(status.billing(v))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
