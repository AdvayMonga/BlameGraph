"""A kernel search run's status, from the store. The VM of a run that ended is deleted; a run whose VM was preempted
(stopped) is started again from its last saved generation.

  lab kernels status [--run RUN]       default: the newest run
"""
from __future__ import annotations

import argparse
import sys
import time

from lab.kernels import run as runs
from lab.kernels.run import launch


def line(st: dict, vm_state: str) -> str:
    age = time.time() - st["updated"]
    left = st["deadline"] - time.time()
    return (f"{st['run']} {st['task']}: {st['state']}, stage {st['stage']}, generation {st['generation']} "
            f"(saved {age / 60:.0f} min ago); VM {st['vm']} {vm_state}"
            + (f", deleted by its deadline in {left / 3600:.1f} h" if st["state"] == "running" and left > 0 else "")
            + (f"\n  {st['reason']}" if st.get("reason") else ""))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="lab kernels status", description=__doc__.split("\n")[0])
    ap.add_argument("--run")
    a = ap.parse_args(argv)
    from lab.infra.cloud import vm
    from lab.records.remote import open_remote
    remote = open_remote()
    run = a.run or runs.last_run(remote)
    st = runs.state(remote, run) if run else None
    if st is None:
        print(f"lab kernels status: {run or 'no run'} is not in the store", file=sys.stderr)
        return 1
    v = vm.VM(name=st["vm"])
    record = v.provider.get(v.name)
    print(line(st, record.state if record else "deleted"))
    if launch.tidy(v, st):
        print(f"  {v.name} deleted: billing ended")
    elif st["state"] == "running" and record is not None and record.state == "stopped":
        missing = launch.missing()
        if missing:
            print("  preempted; not started again here: " + "; ".join(missing), file=sys.stderr)
            return 1
        print("  preempted: starting it again from its last saved generation", flush=True)
        launch.go(v, run, st["task"])
        print(f"  running again on {v.name}")
    elif st["state"] == "running" and record is None:
        print(f"  its VM is gone: `lab kernels run --run {run}` goes on from generation {st['generation']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
