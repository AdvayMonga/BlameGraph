"""The Linux jails (lab/RUNTIME.md). A measurement room: one container per jailed command (NVIDIA runtime for the
GPU), no network at all, read-only root, non-root user, no capabilities; a served engine's port reaches the host
through a socket bridge. The agent's workbench: one container per session with the GPU, the internet and root
inside it (writable system dirs, Docker's default capabilities), paused by `quiet` while anything is measured."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import signal
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

IMAGE = os.environ.get("LAB_JAIL_IMAGE", "lab-jail")
DOCKERFILE = Path(__file__).parent / "jail.Dockerfile"
JAIL_UID = 10001             # the jail's user when the lab runs as root; otherwise the lab's own uid
CLIENT_CORES = 4             # kept out of every jail's cpuset so the measurement clients cannot be starved
SYSTEM_PREFIXES = {Path("/"), Path("/usr"), Path("/usr/local")}   # the image has its own; never mounted over it
BRIDGE = "bridge.sock"


def _docker() -> str | None:
    return shutil.which("docker")


def available() -> bool:
    """Docker answers and the jail image is built."""
    d = _docker()
    return bool(d) and subprocess.run([d, "image", "inspect", IMAGE], capture_output=True).returncode == 0


def build() -> None:
    subprocess.run([_docker(), "build", "-q", "-t", IMAGE, "-f", str(DOCKERFILE), str(DOCKERFILE.parent)], check=True)


def gpus() -> bool:
    return shutil.which("nvidia-smi") is not None


def user() -> tuple[int, int]:
    return (JAIL_UID, JAIL_UID) if os.getuid() == 0 else (os.getuid(), os.getgid())


def give(paths: list[Path]) -> None:
    """Running as root, hand the writable paths to the jail's user so nothing in the jail runs as root."""
    if os.getuid() != 0:
        return
    for root in paths:
        for p in [root, *root.rglob("*")]:
            os.lchown(p, JAIL_UID, JAIL_UID)


def cpuset() -> str | None:
    n = os.cpu_count() or 1
    return f"{CLIENT_CORES}-{n - 1}" if n > 2 * CLIENT_CORES else None


def mounts(writable: list[Path], readonly: list[Path], python: str) -> list[str]:
    """Bind mounts at the same absolute paths, so argv and env mean the same inside; interpreter dirs read-only."""
    interpreter = {Path(sys.base_prefix).resolve(), Path(python).resolve().parent.parent}
    ro = [p for p in [*readonly, *sorted(interpreter)] if p.exists() and p.resolve() not in SYSTEM_PREFIXES]
    if Path("/usr/local/cuda").exists():
        ro.append(Path("/usr/local/cuda").resolve())     # nvcc for JIT-compiling libraries
    args = []
    for p in writable:
        args += ["--mount", f"type=bind,src={p.resolve()},dst={p.resolve()}"]
    for p in dict.fromkeys(q.resolve() for q in ro):
        args += ["--mount", f"type=bind,src={p},dst={p},readonly"]
    return args


def argv(name: str, workdir: Path, command: list[str], env: dict[str, str], writable: list[Path],
         readonly: list[Path], python: str, bridge_port: int | None = None, gpu: bool | None = None) -> list[str]:
    """`docker run` for `command`. With `bridge_port`, the engine's 127.0.0.1:port is also served on the unix socket
    `<first writable tmp>/bridge.sock`, which the host side of `Bridge` forwards to the host's 127.0.0.1:port."""
    uid, gid = user()
    out = [_docker(), "run", "--rm", "--init", "--name", name, "--network", "none", "--read-only",
           "--tmpfs", "/tmp:exec", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
           "--pids-limit", "8192", "--shm-size", "16g", "--user", f"{uid}:{gid}", "--workdir", str(workdir.resolve()),
           "--label", "lab.jail=1"]
    if gpus() if gpu is None else gpu:
        out += ["--gpus", "all"]
    if (cpus := cpuset()):
        out += ["--cpuset-cpus", cpus]
    for k, v in env.items():
        out += ["--env", f"{k}={v}"]
    out += mounts(writable, readonly, python)
    out.append(IMAGE)
    if bridge_port is None:
        return out + command
    sock = writable[-1].resolve() / BRIDGE
    return out + ["sh", "-c", f'socat UNIX-LISTEN:{sock},fork,mode=666 TCP:127.0.0.1:{bridge_port} & exec "$@"',
                  "jail", *command]


def workbench_argv(name: str, workdir: Path, command: list[str], env: dict[str, str], writable: list[Path],
                   readonly: list[Path], python: str) -> list[str]:
    """`docker run -i` for the agent's CLI: stdio stays the SDK's, the default bridge network gives the internet,
    root inside when the lab runs as root (otherwise the lab's uid, so the workspace stays the lab's)."""
    uid, gid = (0, 0) if os.getuid() == 0 else (os.getuid(), os.getgid())
    out = [_docker(), "run", "-i", "--rm", "--init", "--name", name, "--security-opt", "no-new-privileges",
           "--pids-limit", "16384", "--shm-size", "16g", "--user", f"{uid}:{gid}", "--workdir", str(workdir.resolve()),
           "--label", "lab.workbench=1"]
    if gpus():
        out += ["--gpus", "all"]
    if (cpus := cpuset()):
        out += ["--cpuset-cpus", cpus]
    for k, v in env.items():
        out += ["--env", f"{k}={v}"]
    return out + mounts(writable, readonly, python) + [IMAGE, *command]


def bridge_gateway() -> str:
    """The host's address on Docker's default bridge: what a workbench reaches the host at, and nothing outside."""
    out = subprocess.run([_docker(), "network", "inspect", "bridge", "-f", "{{(index .IPAM.Config 0).Gateway}}"],
                         capture_output=True, text=True, check=True)
    return out.stdout.strip()


def running(name: str) -> bool:
    out = subprocess.run([_docker(), "inspect", "-f", "{{.State.Running}}", name], capture_output=True, text=True)
    return out.returncode == 0 and out.stdout.strip() == "true"


def _kill_gpu_holders(name: str) -> list[dict]:
    from lab import gpu
    mine = pids(name)
    hit = [a for a in (gpu.compute_apps() or []) if a["pid"] in mine]
    for a in hit:
        try:
            os.kill(a["pid"], signal.SIGKILL)
        except ProcessLookupError:
            pass
    return hit


@contextmanager
def quiet(name: str | None):
    """The workbench paused for a measurement: processes in it holding the GPU are killed (and listed), the rest
    frozen; thawed after. A no-op when there is no such running container."""
    if not name or _docker() is None or not running(name):
        yield []
        return
    from lab import gpu
    killed = _kill_gpu_holders(name)
    subprocess.run([_docker(), "pause", name], capture_output=True, check=True)
    try:
        killed += _kill_gpu_holders(name)            # anything that took the GPU before the pause landed
        deadline = time.monotonic() + 15
        while killed and time.monotonic() < deadline and any(
                a["pid"] in {k["pid"] for k in killed} for a in gpu.compute_apps() or []):
            time.sleep(0.2)                          # until the driver has released their memory
        yield killed
    finally:
        subprocess.run([_docker(), "unpause", name], capture_output=True)


def new_name() -> str:
    return f"lab-jail-{uuid.uuid4().hex[:12]}"


def pids(name: str) -> set[int]:
    """The container's processes as host pids; empty once it is gone."""
    out = subprocess.run([_docker(), "top", name, "-eo", "pid"], capture_output=True, text=True)
    return {int(x) for x in out.stdout.split() if x.isdigit()} if out.returncode == 0 else set()


def remove(name: str) -> None:
    """Kill and remove the container; a no-op if it is already gone."""
    subprocess.run([_docker(), "rm", "-f", name], capture_output=True)


class Bridge:
    """Host side: 127.0.0.1:port on the host forwarded to the jail's socket. Only host processes can reach it."""

    def __init__(self, port: int, sock: Path):
        self.proc = subprocess.Popen(["socat", f"TCP-LISTEN:{port},bind=127.0.0.1,reuseaddr,fork",
                                      f"UNIX-CONNECT:{sock}"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                     start_new_session=True)

    def stop(self) -> None:
        try:
            os.killpg(self.proc.pid, 9)
        except ProcessLookupError:
            pass
        self.proc.wait()


if __name__ == "__main__":
    if sys.argv[1:] != ["build"]:
        sys.exit("usage: python -m lab.safety.container build")
    build()
