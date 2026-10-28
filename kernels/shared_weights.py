"""Shared read-only weights (decided 2026-10-23): a region as large as a whole decode step reads the whole model
(~30 GB), and each candidate is judged in a fresh process, so reloading per judgment would take minutes. A holder
process keeps the weights on the device and writes handles to a file; each candidate's process attaches to the same
memory without copying (CUDA IPC through torch's tensor sharing; on the CPU, shared memory). CUDA cannot map shared
memory read-only, so the holder also writes a pristine CPU copy beside the handles file (mounted read-only in a jail
for the copy fallback: weights leak nothing, and integrity comes from `restore` comparing the holder's real memory with
it outside the jail); before and after each judgment the evaluator compares every tensor with it on the device
(`restore`): a judgment that changed any weight is void and the weight is copied back. Judgments on one holder run one at a time (`exclusive`). A
jail attaching over CUDA IPC needs the holder's IPC namespace (docker --ipc) and the holder's IPC semaphore pool:
CUDA keeps it in /dev/shm as `cuda.shm.<uid>.<pid>.<n>` (both in hex) and an importer looks for it under its own uid,
so a holder run as root for a jail of another user (`jail_uid`) links its pools under that uid's name, open to that
user's group (found on the H200, 2026-10-27: "Failed to open/find remote IPC semaphore pool", CUDA_ERROR_MAP_FAILED
from cuIpcOpenEventHandle, until then; the jail's own uid, caps and namespaces were not the cause). A jail can then
write the pool's semaphores, which can break the holder's IPC events but not its weights. `python -m
kernels.shared_weights` reports a plain and a jailed attach. When the attach is refused, `attach(copy_ok=True)` (the evaluator's task
inside the jail) copies the pristine weights to the device instead (a layer's weights are ~0.6 GB, so a single-layer
region is judged this way; a whole step is not): its writes reach only its copy. `restore` never copies: it raises when
it cannot alias the holder's memory. Residual risk: a write undone before
the judgment ends is unseen here (the timer fingerprints inputs between calls, kernels/timing.py, and stage 2's
outputs are recomputed outside the jail on the restored weights, verify.py)."""
from __future__ import annotations

import fcntl
import os
import pickle
import weakref
from contextlib import contextmanager
from multiprocessing.reduction import ForkingPickler
from pathlib import Path

import torch
from torch.multiprocessing import reductions

reductions.init_reductions()                     # torch's tensor reducers: a reference to shared memory, not its bytes
CHUNK = 1 << 28                                  # bytes compared per copy of the pristine copy to the device
SHM = Path("/dev/shm")
DEVICE = "_device"                               # the holder's device, kept in the constants file, never a constant
_HELD: dict = {}                                    # holder files made by this process -> their entries
_COPIED: set[Path] = set()                      # holders attached by copying the pristine file
_ATTACHED: list[weakref.ref] = []               # every tensor attached in this process, while it lives


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

    def __init__(self, tensors: dict[str, torch.Tensor], path: Path, constants: dict | None = None,
                 jail_uid: int | None = None):
        """`constants`: small picklable values shared beside the weights (a region's layer config and `layer`);
        `jail_uid`: the user a jail attaches as, when it is not this process's (a root lab's jail)."""
        torch.multiprocessing.set_sharing_strategy("file_system")
        # a model's Parameters, detached: attached ones requiring grad refused the checker's in-place refill (2026-10-27)
        tensors = {k: (v.detach() if v is not None else None) for k, v in tensors.items()}
        self.tensors = {k: (v if v.is_cuda else v.share_memory_()) for k, v in tensors.items() if v is not None}
        self.path = Path(path)
        torch.save({k: v.detach().cpu() for k, v in self.tensors.items()}, _side(self.path, "pristine"))
        _side(self.path, "pristine").chmod(0o644)
        rest = {**{k: None for k, v in tensors.items() if v is None}, **(constants or {})}   # None: bf16 has no scale
        dev = next(iter(self.tensors.values())).device.type if self.tensors else "cpu"   # a jail renumbers GPUs
        _write(_side(self.path, "constants"), pickle.dumps({**rest, DEVICE: dev}))
        _write(self.path, bytes(ForkingPickler.dumps({**self.tensors, **rest})))
        _HELD[self.path.resolve()] = {**self.tensors, **rest}
        if jail_uid is not None and jail_uid != os.getuid():
            _share_pools(jail_uid)

    def changed(self) -> list[str]:
        """The weights whose bytes differ from when they were shared (written by a judgment)."""
        clean = _pristine(self.path)
        return [k for k, v in self.tensors.items() if not _equal(v, clean[k])]


def _share_pools(uid: int, shm: Path = SHM) -> list[Path]:
    """This process's CUDA IPC semaphore pools linked under `uid`'s name, open to `uid`'s group (as `uid` == gid in
    the jail); the links go when this process exits."""
    import atexit
    me, pid = format(os.getuid(), "x"), format(os.getpid(), "x")
    links = []
    for f in shm.glob(f"cuda.shm.{me}.{pid}.*"):
        link = f.with_name(f"cuda.shm.{format(uid, 'x')}.{pid}.{f.name.rsplit('.', 1)[1]}")
        os.chown(f, -1, uid)
        f.chmod(0o660)
        link.unlink(missing_ok=True)
        os.link(f, link)
        links.append(link)
    atexit.register(lambda: [x.unlink(missing_ok=True) for x in links])
    return links


def attach(path: Path, copy_ok: bool = False) -> dict[str, torch.Tensor]:
    """The holder's tensors, in this process, aliasing its memory (no copy); with `copy_ok`, a copy of the pristine
    weights on the holder's device when CUDA IPC refuses the attach."""
    torch.multiprocessing.set_sharing_strategy("file_system")
    raw = _trusted(path).read_bytes()
    own = _HELD.get(Path(path).resolve())        # the holder's own process: CUDA IPC cannot reopen its own handles
    try:
        held = dict(own) if own is not None else _rebuild(raw)
    except RuntimeError as e:                    # CUDA IPC refused (a jail, 2026-10-25)
        if "cuda" not in str(e).lower():
            raise
        if not copy_ok:
            raise RuntimeError(f"cannot attach the shared weights {path} over CUDA IPC ({e}): checking or restoring "
                               "them needs the holder's own memory") from e
        held = {**constants(path), **{k: v.to(_device(path)) for k, v in _pristine(path).items()}}
        _COPIED.add(Path(path).resolve())
    _ATTACHED[:] = [r for r in _ATTACHED if r() is not None] + \
        [weakref.ref(v) for v in held.values() if isinstance(v, torch.Tensor)]
    return held


def _rebuild(raw: bytes) -> dict:
    """The handles file's entries, the tensors mapped over IPC (torch's reducers)."""
    return pickle.loads(raw)


def attached_by_copy(path: Path) -> bool:
    """Whether this process holds its own copy of `path`'s weights (the IPC attach was refused)."""
    return Path(path).resolve() in _COPIED


def _device(path: Path) -> str:
    """The device the holder's tensors live on, as it wrote it; for an older holder, whether its handles name CUDA."""
    dev = pickle.loads(_trusted(_side(path, "constants")).read_bytes()).get(DEVICE)
    return dev or ("cuda" if b"cuda" in _trusted(path).read_bytes() else "cpu")


def is_shared_weight(t: torch.Tensor) -> bool:
    """Whether `t` shares memory with a live tensor attached from a holder in this process."""
    return any(w is not None and w.data_ptr() == t.data_ptr() for w in (r() for r in _ATTACHED))


def constants(path: Path) -> dict:
    """The holder's non-tensor entries (its constants and None weights), read without attaching any tensor."""
    d = pickle.loads(_trusted(_side(path, "constants")).read_bytes())
    d.pop(DEVICE, None)
    return d


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
    for p in (root, root / "kernels", hd):       # mkdtemp's 0700: the jail's user (not root) could not read them
        p.chmod(0o755)
    for f in (root / "kernels").iterdir():
        f.chmod(0o644)
    from lab.safety import container
    h = Holder({"a": torch.arange(1024, dtype=torch.float32, device="cuda")}, hd / "w.pkl",
               jail_uid=container.user()[0] if jailed else None)
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
