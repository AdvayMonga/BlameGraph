"""Shared read-only weights (decided 2026-10-23): a region as large as a whole decode step reads the whole model
(~30 GB), and each candidate is judged in a fresh process, so reloading per judgment would take minutes. A holder
process keeps the weights on the device and writes handles to a file; each candidate's process attaches to the same
memory without copying (CUDA IPC through torch's tensor sharing; on the CPU, shared memory). CUDA cannot map shared
memory read-only, so the holder also writes a pristine CPU copy beside the handles file (never mounted in a jail);
before and after each judgment the evaluator compares every tensor with it on the device (`restore`): a judgment that
changed any weight is void and the weight is copied back. Judgments on one holder run one at a time (`exclusive`). A
jail attaching over CUDA IPC needs the holder's IPC namespace (docker --ipc). Not working yet: on the H200
(2026-10-25) a process outside a jail attaches, writes and is restored, but the jailed child's attach fails in
cuIpcOpenEventHandle (CUDA_ERROR_MAP_FAILED), also with --pid host and the holder's IPC pool file opened to it; `python
-m kernels.shared_weights` reports both. Residual risk: a write undone before
the judgment ends is unseen here (the timer fingerprints inputs between calls, kernels/timing.py, and stage 2's
outputs are recomputed outside the jail on the restored weights, verify.py)."""
from __future__ import annotations

import fcntl
import os
import pickle
from contextlib import contextmanager
from multiprocessing.reduction import ForkingPickler
from pathlib import Path

import torch
from torch.multiprocessing import reductions

reductions.init_reductions()                     # torch's tensor reducers: a reference to shared memory, not its bytes
CHUNK = 1 << 28                                  # bytes compared per copy of the pristine copy to the device
SHM = Path("/dev/shm")
_HELD: dict = {}                                    # holder files made by this process -> their entries
_ATTACHED: set[int] = set()                      # data_ptr of every tensor attached in this process


def _side(path: Path, kind: str) -> Path:
    return Path(path).with_name(f"{Path(path).name}.{kind}")


def _trusted(path: Path) -> Path:
    """`path`, refused under /dev/shm (a jail with the host's IPC namespace writes there) and unless it is owned by
    this user (or root) and writable by no one else: it is unpickled outside the jail."""
    p = Path(path).resolve()
    if p.is_relative_to(SHM.resolve()) or p.is_relative_to(SHM):
        raise ValueError(f"shared weights file {p} is under {SHM}, which a jail can write")
    st = p.stat()
    if st.st_uid not in (os.getuid(), 0) or st.st_mode & 0o022:
        raise ValueError(f"shared weights file {p} is owned by uid {st.st_uid} with mode {oct(st.st_mode & 0o777)}: "
                         "it must be this user's (or root's) and writable by no one else")
    return p


def _write(path: Path, data: bytes) -> None:
    path.write_bytes(data)
    path.chmod(0o644)


class Holder:
    """Keeps `tensors` alive and shared; `path` holds what `attach` needs."""

    def __init__(self, tensors: dict[str, torch.Tensor], path: Path, constants: dict | None = None):
        """`constants`: small picklable values shared beside the weights (a region's layer config and `layer`)."""
        torch.multiprocessing.set_sharing_strategy("file_system")
        self.tensors = {k: (v if v.is_cuda else v.share_memory_()) for k, v in tensors.items() if v is not None}
        self.path = Path(path)
        torch.save({k: v.detach().cpu() for k, v in self.tensors.items()}, _side(self.path, "pristine"))
        _side(self.path, "pristine").chmod(0o644)
        rest = {**{k: None for k, v in tensors.items() if v is None}, **(constants or {})}   # None: bf16 has no scale
        _write(_side(self.path, "constants"), pickle.dumps(rest))
        _write(self.path, bytes(ForkingPickler.dumps({**self.tensors, **rest})))
        _HELD[self.path.resolve()] = {**self.tensors, **rest}

    def changed(self) -> list[str]:
        """The weights whose bytes differ from when they were shared (written by a judgment)."""
        clean = _pristine(self.path)
        return [k for k, v in self.tensors.items() if not _equal(v, clean[k])]


def attach(path: Path) -> dict[str, torch.Tensor]:
    """The holder's tensors, in this process, aliasing its memory (no copy)."""
    torch.multiprocessing.set_sharing_strategy("file_system")
    raw = _trusted(path).read_bytes()
    own = _HELD.get(Path(path).resolve())        # the holder's own process: CUDA IPC cannot reopen its own handles
    held = dict(own) if own is not None else pickle.loads(raw)
    _ATTACHED.update(v.data_ptr() for v in held.values() if isinstance(v, torch.Tensor))
    return held


def is_shared_weight(t: torch.Tensor) -> bool:
    """Whether `t` is a tensor attached from a holder in this process."""
    return t.data_ptr() in _ATTACHED


def constants(path: Path) -> dict:
    """The holder's non-tensor entries (its constants and None weights), read without attaching any tensor."""
    return pickle.loads(_trusted(_side(path, "constants")).read_bytes())


def _pristine(path: Path) -> dict:
    return torch.load(_trusted(_side(path, "pristine")), weights_only=True, mmap=True)


def _equal(t: torch.Tensor, clean: torch.Tensor) -> bool:
    """`t`'s bytes against the pristine copy's, compared on t's device a chunk at a time."""
    a, b = (x.detach().reshape(-1).contiguous().view(torch.uint8) for x in (t, clean))
    return a.numel() == b.numel() and all(torch.equal(a[i:i + CHUNK], b[i:i + CHUNK].to(a.device))
                                          for i in range(0, a.numel(), CHUNK))


def restore(path: Path) -> list[str]:
    """The held tensors whose bytes differ from the pristine copy, each copied back from it."""
    held, clean = attach(path), _pristine(path)
    bad = [k for k, v in held.items() if isinstance(v, torch.Tensor) and not _equal(v, clean[k])]
    for k in bad:
        held[k].copy_(clean[k])                  # into the shared memory: the holder's tensor too
    if any(held[k].is_cuda for k in bad):
        torch.cuda.synchronize()
    return bad


@contextmanager
def exclusive(path: Path):
    """Held around a judgment that attaches the holder: such judgments run one at a time, so a write is blamed on the
    judgment that made it."""
    with open(_side(path, "lock"), "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield


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


def harness(root: Path) -> Path:
    """`root` holding this module alone as `kernels.shared_weights` (it imports only torch): what `check` mounts."""
    import shutil
    (root / "kernels").mkdir(parents=True)
    (root / "kernels" / "__init__.py").write_text("")
    shutil.copyfile(__file__, root / "kernels" / "shared_weights.py")
    return root


def check(jailed: bool = True) -> dict:
    """On a GPU host: a holder shares a weight, a child (in the container jail, host IPC namespace) attaches, reads it
    and writes to it; the holder must see the same values, catch the write and restore it. `python -m
    kernels.shared_weights`."""
    import shutil
    import subprocess
    import sys
    import tempfile
    d = Path(tempfile.mkdtemp(prefix="shared-weights-"))
    root = harness(Path(tempfile.mkdtemp(prefix="shared-weights-harness-")))
    hd = Path(tempfile.mkdtemp(prefix="shared-weights-holder-"))       # outside the jail's writable tree
    h = Holder({"a": torch.arange(1024, dtype=torch.float32, device="cuda")}, hd / "w.pkl")
    code = CHILD.format(root=str(root), path=str(h.path))
    (d / "child.py").write_text(code)
    if jailed:
        from lab.safety import grader
        p = grader.jailed(d, [sys.executable, str(d / "child.py")], timeout_s=300, read=[root, hd], ipc_host=True)
    else:
        p = subprocess.run([sys.executable, str(d / "child.py")], capture_output=True, text=True, timeout=300)
    shutil.rmtree(root, ignore_errors=True)
    return {"returncode": p.returncode, "child": (p.stdout + p.stderr)[-1500:], "expected_sum": float(h.tensors["a"].sum()) - 1,
            "changed": h.changed(), "restored": restore(h.path), "after_restore": h.changed()}


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps({"plain": check(False), "jailed": check("--plain" not in sys.argv)}, indent=1))
