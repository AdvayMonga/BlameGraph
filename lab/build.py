"""The dependency build step (lab/RUNTIME.md). A measurement room is rebuilt from the agent's change alone and runs
offline, so a change to the engine's dependency files (the target's `deps`) is installed from scratch for it: a
fresh venv built by the target's `build` command in a container with no GPU, no secrets and no network but a
CONNECT proxy to the package-index hosts, cached by the dependency files' hash. A tree whose change leaves the dependency
files as they were in its base keeps the engine's venv (built from that base by vm-setup.sh), so the base and an
unchanged candidate run identically. A measured tree is a two-commit repo (base, change; grader.pristine_tree); one
without that history, such as the exported base itself, has no change."""

from __future__ import annotations

import hashlib
import os
import re
import select
import shutil
import socket
import socketserver
import subprocess
import tempfile
import threading
from pathlib import Path

from lab import engine, target
from lab.safety import container

INDEX_HOSTS = ("pypi.org", "files.pythonhosted.org")        # plus every host the dependency files and build name
PROXY_PORT_IN = 3128
LOG_TAIL = 4000
TIMEOUT_S = 3600


class BuildFailed(RuntimeError):
    """The tree's dependencies could not be installed; the message carries the build's output."""


def deps_sha(tree: Path) -> str:
    h = hashlib.sha256()
    for rel in target.load().deps:
        p = Path(tree) / rel
        h.update(rel.encode() + b"\0" + (p.read_bytes() if p.is_file() else b"<absent>") + b"\0")
    return h.hexdigest()


def hosts(tree: Path, build: str) -> set[str]:
    """The index hosts a build may reach: the defaults and every https host its dependency files and command name."""
    text = build + "".join((Path(tree) / r).read_text(errors="replace") for r in target.load().deps
                           if (Path(tree) / r).is_file())
    return set(INDEX_HOSTS) | {m.lower() for m in re.findall(r"https://([A-Za-z0-9.-]+)", text)}


def changed(tree: Path) -> bool:
    """Whether the tree's change touched a dependency file: its working files against its base commit (HEAD~1)."""
    tree = Path(tree)
    if not target.load().deps or not (tree / ".git").exists():
        return False
    for rel in target.load().deps:
        old = subprocess.run(["git", "-C", str(tree), "show", f"HEAD~1:{rel}"], capture_output=True)
        if old.returncode != 0 and b"exists on disk, but not in" not in old.stderr and b"does not exist" not in old.stderr:
            return False                        # no base commit: nothing to compare against
        now = (tree / rel).read_bytes() if (tree / rel).is_file() else None
        if (old.stdout if old.returncode == 0 else None) != now:
            return True
    return False


def python_for(tree: Path) -> str:
    """The interpreter that runs `tree`: the engine's own, or a venv built from the tree's changed dependency files."""
    if not changed(tree):
        return engine.python()
    venv = cache_dir() / deps_sha(tree)
    if not (venv / ".lab-built").exists():
        build(tree, venv)
    return str(venv / "env" / "bin" / "python")


def cache_dir() -> Path:
    from lab import worker
    return worker.cache_dir() / "venvs"


class _Connect(socketserver.BaseRequestHandler):
    """One CONNECT host:443 to an allowed host, then bytes both ways; anything else is refused."""
    allowed: set[str] = set()

    def handle(self):
        head = b""
        while b"\r\n\r\n" not in head and len(head) < 8192:
            chunk = self.request.recv(4096)
            if not chunk:
                return
            head += chunk
        line = head.split(b"\r\n", 1)[0].decode(errors="replace").split()
        host, _, port = (line[1] if len(line) > 1 else "").rpartition(":")
        if len(line) < 2 or line[0] != "CONNECT" or port != "443" or host.lower() not in self.allowed:
            self.request.sendall(b"HTTP/1.1 403 Forbidden\r\ncontent-length: 0\r\n\r\n")
            return
        try:
            up = socket.create_connection((host, 443), timeout=30)
        except OSError:
            self.request.sendall(b"HTTP/1.1 502 Bad Gateway\r\ncontent-length: 0\r\n\r\n")
            return
        self.request.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
        with up:
            pair = (self.request, up)
            while True:
                ready, _, _ = select.select(pair, [], [], 300)
                if not ready:
                    return
                for s in ready:
                    data = s.recv(65536)
                    if not data:
                        return
                    (up if s is self.request else self.request).sendall(data)


class IndexProxy:
    """`with IndexProxy(hosts, sock):` a CONNECT-only proxy on unix socket `sock` that reaches `hosts` and nothing else."""

    def __init__(self, allowed: set[str], sock: Path):
        handler = type("Handler", (_Connect,), {"allowed": {h.lower() for h in allowed}})
        self.server = socketserver.ThreadingUnixStreamServer(str(sock), handler)
        self.server.daemon_threads = True
        os.chmod(sock, 0o666)                       # the build runs as the jail's user

    def __enter__(self):
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()


def build(tree: Path, venv: Path) -> None:
    """The target's `build` in a container: the tree's dependency files only, the engine's interpreter and uv read-only,
    uv's cache and `venv` writable (uv makes the environment at `venv/env`), the index proxy as the only way out.
    A failed build leaves no venv behind."""
    t = target.load()
    if not t.build:
        raise BuildFailed("the dependency files changed, and the target has no `build` command to install them")
    if not container.available():
        raise BuildFailed("a dependency build needs the container jail (docker and the lab-jail image)")
    uv = shutil.which("uv") or str(Path.home() / ".local" / "bin" / "uv")
    interpreter = Path(engine.python()).resolve()
    venv.parent.mkdir(parents=True, exist_ok=True)
    uv_cache = cache_dir().parent / "uv-cache"
    uv_cache.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="build-", dir=cache_dir()))
    src, sock_dir = work / "src", Path(tempfile.mkdtemp(prefix="lab-index-"))   # short: a unix socket path
    sock_dir.chmod(0o711)                           # the jail's user reaches the socket, lists nothing
    src.mkdir()
    for rel in t.deps:
        if (Path(tree) / rel).is_file():
            (src / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(Path(tree) / rel, src / rel)
    shutil.rmtree(venv, ignore_errors=True)         # an interrupted earlier build
    venv.mkdir()
    container.give([src, venv, uv_cache])
    name = container.new_name()
    env = {"UV_PROJECT_ENVIRONMENT": str(venv / "env"), "UV_CACHE_DIR": str(uv_cache), "UV_PYTHON": str(interpreter),
           "UV_PYTHON_DOWNLOADS": "never", "HOME": str(src), "PATH": f"{Path(uv).parent}:/usr/bin:/bin",
           "HTTPS_PROXY": f"http://127.0.0.1:{PROXY_PORT_IN}", "HTTP_PROXY": f"http://127.0.0.1:{PROXY_PORT_IN}"}
    argv = container.argv(name, src, ["sh", "-c", f"socat TCP-LISTEN:{PROXY_PORT_IN},bind=127.0.0.1,fork,reuseaddr "
                                      f"UNIX-CONNECT:{sock_dir}/index.sock & {t.build}"],
                          env, [src, venv, uv_cache, sock_dir], [Path(uv).parent, interpreter.parent.parent], str(interpreter),
                          gpu=False)
    try:
        with IndexProxy(hosts(tree, t.build), sock_dir / "index.sock"):
            p = subprocess.run(argv, capture_output=True, text=True, timeout=TIMEOUT_S)
    except subprocess.TimeoutExpired:
        p = subprocess.CompletedProcess(argv, -9, "", f"timed out after {TIMEOUT_S}s")
    finally:
        container.remove(name)
        shutil.rmtree(work, ignore_errors=True)
        shutil.rmtree(sock_dir, ignore_errors=True)
    if p.returncode != 0 or not (venv / "env" / "bin" / "python").exists():
        shutil.rmtree(venv, ignore_errors=True)
        raise BuildFailed(f"dependency build exited {p.returncode}: {(p.stdout + p.stderr)[-LOG_TAIL:]}")
    (venv / ".lab-built").write_text(deps_sha(tree))
