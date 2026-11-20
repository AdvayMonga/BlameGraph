"""A run's status, from the store; the VM of a run that is no longer running is deleted.

  lab status [--run RUN]       default: the newest run
"""
from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

from lab.infra.cloud import vm
from lab.loop import record
from lab.records.remote import Remote


def billing(v: vm.VM) -> str:
    """One line on what is still billed."""
    r = v.provider.get(v.name)
    if r is None:
        return "billing: nothing"
    left = vm.left_s(v)
    return (f"billing: VM {v.name} is {r.state}" + ("" if r.state == "running" else " (its disk)")
            + (f"; deleted by its deadline in {left / 3600:.1f} h" if left else "; `lab vm stop` deletes it"))


def tidy(v: vm.VM, st: dict | None) -> bool:
    """Delete the VM once the run on it no longer runs: it ended, or its VM is not running; True when deleted."""
    r = v.provider.get(v.name)
    if st is None or r is None or (st["state"] == "running" and r.state == "running"):
        return False
    vm.stop(v)
    return True


def status(remote: Remote, v: vm.VM, run: str | None = None) -> list[str]:
    """The run's lines: its state, its report once it has one, and what was done about its VM."""
    run = run or record.last_run(remote)
    st = record.status(remote, run) if run else None
    if st is None:
        return [f"{run or 'no run'} is not in the store"]
    r = v.provider.get(v.name)
    out = [f"{run}: {st['state']}, {st.get('stage', '-')}, turn {st.get('turn', 0)}, {st.get('wins', 0)} win(s), "
           f"${st.get('spent_usd', 0.0):.2f} (saved {(time.time() - st['updated']) / 60:.0f} min ago)"
           + (f"\n  {st['reason']}" if st.get("reason") else "")
           + (f"\n  stopped: {st['stopped']}" if st.get("stopped") else "")]
    if st["state"] == "running" and (r is None or r.state != "running"):
        out.append(f"  its VM is {r.state if r else 'gone'}: the run ended without finishing")
    key = f"{record.PREFIX}/{run}/report.md"
    if remote.exists(key):
        with tempfile.TemporaryDirectory() as d:
            remote.get(key, Path(d) / "r.md")
            out.append((Path(d) / "r.md").read_text().rstrip())
    if tidy(v, st):
        out.append(f"  {v.name} deleted")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="lab status", description=__doc__.split("\n")[0])
    ap.add_argument("--run")
    a = ap.parse_args(argv)
    from lab.records.remote import open_remote
    v = vm.VM()
    print("\n".join(status(open_remote(), v, a.run)))
    print(billing(v))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
