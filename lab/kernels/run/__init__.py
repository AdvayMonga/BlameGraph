"""A kernel search run on one GPU VM: started from here (`lab kernels run`), run there to its end (`job`), read from the
store (`lab kernels status`). The store keeps each run's folder and status after every saved generation:

  runs/kernels/<run>/<n>.tar.zst   snapshot n of the run folder, without what the VM rebuilds (SKIP)
  runs/kernels/<run>/<n>.json      its status: task, vm, instance, deadline, stage, generation, state
"""
from __future__ import annotations

import json
import re
import shutil
import tempfile
from pathlib import Path

from lab.records import sync
from lab.records.remote import Remote
from lab.targets import engine

PREFIX = "runs/kernels"
SKIP = ("tree", "samples", "holder")      # the engine tree, the captured samples, the held weights: rebuilt on the VM
SNAPSHOT = re.compile(r"/(\d+)\.json$")


def folder(run: str) -> Path:
    return engine.ENV_ROOT / "local" / "runs" / "kernels" / run


def latest(remote: Remote, run: str) -> int:
    """The number of the run's last snapshot; 0 when it has none."""
    return max((int(m.group(1)) for k in remote.keys(f"{PREFIX}/{run}/") if (m := SNAPSHOT.search(k))), default=0)


def state(remote: Remote, run: str) -> dict | None:
    """The run's last saved status; None when it has none."""
    n = latest(remote, run)
    if not n:
        return None
    with tempfile.TemporaryDirectory() as d:
        remote.get(f"{PREFIX}/{run}/{n:05d}.json", Path(d) / "s.json")
        return json.loads((Path(d) / "s.json").read_text())


def save(remote: Remote, run: str, src: Path, state: dict) -> int:
    """Snapshot `src` (without SKIP) and `state` as the run's next pair; returns its number."""
    n = latest(remote, run) + 1
    with tempfile.TemporaryDirectory() as d:
        copy = Path(d) / run
        shutil.copytree(src, copy, ignore=lambda at, names: [x for x in names if Path(at) == Path(src) and x in SKIP],
                        symlinks=True)
        packed = Path(d) / "snap.tar.zst"
        sync.pack(copy, packed)
        remote.put(f"{PREFIX}/{run}/{n:05d}.tar.zst", packed)
        (Path(d) / "s.json").write_text(json.dumps(state))
        remote.put(f"{PREFIX}/{run}/{n:05d}.json", Path(d) / "s.json")
    return n


def restore(remote: Remote, run: str, dest: Path) -> bool:
    """The run's last snapshot unpacked as `dest`; False when it has none."""
    n = latest(remote, run)
    if n:
        with tempfile.TemporaryDirectory() as d:
            remote.get(f"{PREFIX}/{run}/{n:05d}.tar.zst", Path(d) / "snap.tar.zst")
            sync.unpack(Path(d) / "snap.tar.zst", dest)
    return bool(n)


def last_run(remote: Remote) -> str | None:
    """The newest run in the store (run ids sort by their start time)."""
    return max({k.split("/")[2] for k in remote.keys(f"{PREFIX}/")}, default=None)
