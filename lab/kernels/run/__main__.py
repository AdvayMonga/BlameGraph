"""A kernel search on its own GPU VM, to its end: the VM saves to the store after every generation and deletes itself
when the search ends. This host only starts it; `lab kernels status` reads it.

  lab kernels run --task TASK          a new run
  lab kernels run --run RUN            the stopped run RUN again, from its last saved generation
"""
from __future__ import annotations

import argparse
import sys
import time
import uuid

from lab.kernels import run as runs
from lab.kernels.run import launch


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="lab kernels run", description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--task", help="the region to search: region:FIRST:LAST")
    g.add_argument("--run", help="a stopped run to go on with")
    a = ap.parse_args(argv)
    missing = launch.missing()
    if missing:
        print("lab kernels run: nothing was started.\n" + "\n".join(f"- {x}" for x in missing), file=sys.stderr)
        return 2
    from lab.infra.cloud import vm
    from lab.records.remote import open_remote
    v = vm.VM()
    if a.run:
        st = runs.state(open_remote(), a.run)
        if st is None or st["state"] == "done":
            print(f"lab kernels run: {a.run} is {'not in the store' if st is None else 'done'}", file=sys.stderr)
            return 2
        run, task = a.run, st["task"]
        v = vm.VM(name=st["vm"])
        record = v.provider.get(v.name)
        if record is not None and record.state == "running":
            print(f"lab kernels run: {v.name} is running {a.run}: `lab kernels status --run {a.run}` reads it",
                  file=sys.stderr)
            return 2
    else:
        from lab.kernels.evolve.run import task_name
        try:
            task_name(a.task)
        except argparse.ArgumentTypeError as e:
            print(f"lab kernels run: {e}", file=sys.stderr)
            return 2
        if v.provider.get(v.name) is not None:
            print(f"lab kernels run: nothing was started. A VM named {v.name} already exists: `lab vm stop` deletes "
                  "it, or set LAB_VM to another name.", file=sys.stderr)
            return 2
        run, task = f"kgen-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}", a.task
    launch.go(v, run, task)
    print(f"{run}: running on {v.name}; `lab kernels status --run {run}` reads it. The VM deletes itself at the end.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
