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

    def __init__(self, tensors: dict[str, torch.Tensor], path: Path):
        self.tensors = {k: (v if v.is_cuda else v.share_memory_()) for k, v in tensors.items()}
        self.sums = {k: checksum(v) for k, v in self.tensors.items()}
        self.path = Path(path)
        self.path.write_bytes(bytes(ForkingPickler.dumps(self.tensors)))

    def changed(self) -> list[str]:
        """The weights whose bytes differ from when they were shared (written by a judgment)."""
        return [k for k, v in self.tensors.items() if checksum(v) != self.sums[k]]


def attach(path: Path) -> dict[str, torch.Tensor]:
    """The holder's tensors, in this process, aliasing its memory (no copy)."""
    return pickle.loads(Path(path).read_bytes())
