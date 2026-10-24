"""Shared read-only weights (kernels/shared_weights.py), on the CPU's shared memory: another process attaches to the
holder's tensors without copying, reads the same values, and a write it makes is caught by the checksum."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import subprocess  # noqa: E402

import torch  # noqa: E402

from kernels.shared_weights import Holder, checksum  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CHILD = """
import sys, torch
sys.path.insert(0, {root!r})
from kernels.shared_weights import attach
w = attach({path!r})
print(float(w["a"].sum()), w["a"].data_ptr() != 0)
if {write}:
    w["b"][0, 0] += 1
"""


def run(path, write):
    code = CHILD.format(root=str(ROOT), path=str(path), write=write)
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.split()


def test_a_child_reads_the_holders_weights_and_a_write_is_caught(tmp_path):
    torch.multiprocessing.set_sharing_strategy("file_system")
    h = Holder({"a": torch.arange(12, dtype=torch.float32).view(3, 4), "b": torch.ones(2, 2, dtype=torch.bfloat16)},
               tmp_path / "w.pkl")
    total, ok = run(h.path, False)
    assert float(total) == 66.0 and ok == "True" and h.changed() == []
    run(h.path, True)
    assert h.changed() == ["b"]


def test_the_checksum_sees_order_and_tail_bytes():
    a = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.uint8)
    b = a.clone()
    b[[0, 1]] = b[[1, 0]]
    c = a.clone()
    c[-1] += 1
    assert len({checksum(a), checksum(b), checksum(c)}) == 3


if __name__ == "__main__":
    import inspect
    import tempfile
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(*[Path(tempfile.mkdtemp()) for _ in inspect.signature(fn).parameters]); print("ok", name)
