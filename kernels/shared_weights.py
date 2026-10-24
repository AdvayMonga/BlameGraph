"""Shared read-only weights (decided 2026-10-23): a region as large as a whole decode step reads the whole model
(~30 GB), and each candidate is judged in a fresh process, so reloading per judgment would take minutes. A holder
process keeps the weights on the device and writes handles to a file; each candidate's process attaches to the same
memory without copying (CUDA IPC through torch's tensor sharing; on the CPU, shared memory). CUDA cannot map shared
memory read-only, so the holder checksums every tensor before and after each judgment: a judgment that changed any
weight is void (`Holder.changed`). A jail attaching over CUDA IPC needs the holder's IPC namespace (docker --ipc)."""
from __future__ import annotations

import pickle
from multiprocessing.reduction import ForkingPickler
from pathlib import Path

import torch
from torch.multiprocessing import reductions

reductions.init_reductions()                     # torch's tensor reducers: a reference to shared memory, not its bytes


def checksum(t: torch.Tensor) -> int:
    """A cheap fingerprint of a tensor's bytes, computed where it lives (no copy to the host beyond one number)."""
    b = t.detach().contiguous().view(torch.uint8).flatten()
    n = b.numel() - b.numel() % 8
    words = b[:n].view(torch.int64)
    pos = torch.arange(1, words.numel() + 1, device=t.device, dtype=torch.int64)
    tail = b[n:].to(torch.int64).sum() if n < b.numel() else torch.zeros((), dtype=torch.int64, device=t.device)
    return int(((words * pos).sum() + (words ^ pos).sum() + tail).item())     # order-sensitive, overflow wraps


class Holder:
    """Keeps `tensors` alive and shared; `path` holds what `attach` needs."""

    def __init__(self, tensors: dict[str, torch.Tensor], path: Path, constants: dict | None = None):
        """`constants`: small picklable values shared beside the weights (a region's layer config)."""
        self.tensors = {k: (v if v.is_cuda else v.share_memory_()) for k, v in tensors.items() if v is not None}
        self.sums = {k: checksum(v) for k, v in self.tensors.items()}
        self.path = Path(path)
        nones = {k: None for k, v in tensors.items() if v is None}            # e.g. bf16 experts carry no scale
        self.path.write_bytes(bytes(ForkingPickler.dumps({**self.tensors, **nones, **(constants or {})})))

    def changed(self) -> list[str]:
        """The weights whose bytes differ from when they were shared (written by a judgment)."""
        return [k for k, v in self.tensors.items() if checksum(v) != self.sums[k]]


def attach(path: Path) -> dict[str, torch.Tensor]:
    """The holder's tensors, in this process, aliasing its memory (no copy)."""
    return pickle.loads(Path(path).read_bytes())


CHILD = """
import sys, torch
sys.path.insert(0, {root!r})
from kernels.shared_weights import attach
w = attach({path!r})
a = w["a"]
print("attached", a.device, float(a.float().sum()), a.data_ptr())
a[0] += 1
torch.cuda.synchronize()
"""


def check(jailed: bool = True) -> dict:
    """On a GPU host: a holder shares a weight, a child (in the container jail, host IPC namespace) attaches, reads it
    and writes to it; the holder must see the same values and catch the write. `python -m kernels.shared_weights`."""
    import subprocess
    import sys
    import tempfile
    root = Path(__file__).resolve().parents[1]
    d = Path(tempfile.mkdtemp(prefix="shared-weights-"))
    h = Holder({"a": torch.arange(1024, dtype=torch.float32, device="cuda")}, d / "w.pkl")
    code = CHILD.format(root=str(root), path=str(h.path))
    (d / "child.py").write_text(code)
    if jailed:
        from lab.safety import grader
        p = grader.jailed(d, [sys.executable, str(d / "child.py")], timeout_s=300, read=[root], ipc_host=True)
    else:
        p = subprocess.run([sys.executable, str(d / "child.py")], capture_output=True, text=True, timeout=300)
    return {"returncode": p.returncode, "child": (p.stdout + p.stderr)[-1500:], "expected_sum": float(h.tensors["a"].sum()) - 1,
            "changed": h.changed()}


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps({"plain": check(False), "jailed": check("--plain" not in sys.argv)}, indent=1))
