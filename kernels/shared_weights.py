"""Shared read-only weights (decided 2026-10-23): a region as large as a whole decode step reads the whole model
(~30 GB), and each candidate is judged in a fresh process, so reloading per judgment would take minutes. A holder
process keeps the weights on the device and writes handles to a file; each candidate's process attaches to the same
memory without copying (CUDA IPC through torch's tensor sharing; on the CPU, shared memory). CUDA cannot map shared
memory read-only, so the holder also writes each tensor's blake2b hash and a pristine CPU copy beside the handles
file (never mounted in a jail); after each judgment the evaluator hashes every tensor again (`restore`): a judgment
that changed any weight is void and the weight is copied back. A jail attaching over CUDA IPC needs the holder's IPC
namespace (docker --ipc). Residual risk: a write undone before the judgment ends is unseen by the hashes (stage 2's
outputs are still recomputed outside the jail on the restored weights, verify.py), and judgments sharing a holder at
the same time can affect each other."""
from __future__ import annotations

import hashlib
import json
import pickle
from multiprocessing.reduction import ForkingPickler
from pathlib import Path

import torch
from torch.multiprocessing import reductions

reductions.init_reductions()                     # torch's tensor reducers: a reference to shared memory, not its bytes
CHUNK = 1 << 26                                  # bytes hashed per copy to the host


def digest(t: torch.Tensor) -> str:
    """blake2b of a tensor's bytes, copied to the host a chunk at a time."""
    b = t.detach().reshape(-1).contiguous().view(torch.uint8)
    h = hashlib.blake2b(digest_size=32)
    for i in range(0, b.numel(), CHUNK):
        h.update(b[i:i + CHUNK].cpu().numpy())
    return h.hexdigest()


def _side(path: Path, kind: str) -> Path:
    return Path(path).with_name(f"{Path(path).name}.{kind}")


class Holder:
    """Keeps `tensors` alive and shared; `path` holds what `attach` needs."""

    def __init__(self, tensors: dict[str, torch.Tensor], path: Path, constants: dict | None = None):
        """`constants`: small picklable values shared beside the weights (a region's layer config and `layer`)."""
        torch.multiprocessing.set_sharing_strategy("file_system")
        self.tensors = {k: (v if v.is_cuda else v.share_memory_()) for k, v in tensors.items() if v is not None}
        self.path = Path(path)
        self.hashes = {k: digest(v) for k, v in self.tensors.items()}
        _side(self.path, "hashes").write_text(json.dumps(self.hashes))
        torch.save({k: v.detach().cpu() for k, v in self.tensors.items()}, _side(self.path, "pristine"))
        rest = {**{k: None for k, v in tensors.items() if v is None}, **(constants or {})}   # None: bf16 has no scale
        _side(self.path, "constants").write_bytes(pickle.dumps(rest))
        self.path.write_bytes(bytes(ForkingPickler.dumps({**self.tensors, **rest})))

    def changed(self) -> list[str]:
        """The weights whose bytes differ from when they were shared (written by a judgment)."""
        return [k for k, v in self.tensors.items() if digest(v) != self.hashes[k]]


def attach(path: Path) -> dict[str, torch.Tensor]:
    """The holder's tensors, in this process, aliasing its memory (no copy)."""
    torch.multiprocessing.set_sharing_strategy("file_system")
    return pickle.loads(Path(path).read_bytes())


def constants(path: Path) -> dict:
    """The holder's non-tensor entries (its constants and None weights), read without attaching any tensor."""
    return pickle.loads(_side(path, "constants").read_bytes())


def restore(path: Path) -> list[str]:
    """The held tensors whose hash changed since the holder shared them, each copied back from the pristine copy."""
    want = json.loads(_side(path, "hashes").read_text())
    held = attach(path)
    bad = [k for k, h in want.items() if digest(held[k]) != h]
    if bad:
        clean = torch.load(_side(path, "pristine"), weights_only=True, mmap=True)
        for k in bad:
            held[k].copy_(clean[k])              # into the shared memory: the holder's tensor too
        if any(held[k].is_cuda for k in bad):
            torch.cuda.synchronize()
    return bad


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
    h = Holder({"a": torch.arange(1024, dtype=torch.float32, device="cuda")}, d / "w.pkl")
    code = CHILD.format(root=str(root), path=str(h.path))
    (d / "child.py").write_text(code)
    if jailed:
        from lab.safety import grader
        p = grader.jailed(d, [sys.executable, str(d / "child.py")], timeout_s=300, read=[root], ipc_host=True)
    else:
        p = subprocess.run([sys.executable, str(d / "child.py")], capture_output=True, text=True, timeout=300)
    shutil.rmtree(root, ignore_errors=True)
    return {"returncode": p.returncode, "child": (p.stdout + p.stderr)[-1500:], "expected_sum": float(h.tensors["a"].sum()) - 1,
            "changed": h.changed(), "restored": restore(h.path), "after_restore": h.changed()}


if __name__ == "__main__":
    import sys
    print(json.dumps({"plain": check(False), "jailed": check("--plain" not in sys.argv)}, indent=1))
