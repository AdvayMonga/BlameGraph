"""Serve an engine tree the way the agent cannot tamper with: launched from the pristine copy, inside the jail, on a
free localhost port, with the target's env; ready when its health path answers 200; torn down with its whole
process group. `with Served(tree, spec) as s: ... s.url`."""

from __future__ import annotations

import os
import shlex
import signal
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from lab import engine, target
from lab.safety import grader


def free_port() -> int:
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    return port


def argv_for(server: target.Server, port: int) -> list[str]:
    """The launch command with `{port}` filled and `python` meaning the engine's interpreter."""
    words = shlex.split(server.launch.replace("{port}", str(port)))
    if words and words[0] in ("python", "python3"):
        words[0] = engine.python()
    return words


class NotReady(RuntimeError):
    pass


class Served:
    def __init__(self, tree: Path, server: target.Server, log: Path | None = None, port: int | None = None,
                 timeout_s: float | None = None, env_extra: dict | None = None, jailed: bool = True):
        self.tree, self.server = Path(tree), server
        self.port = port or free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.log = Path(log) if log else None
        self.timeout_s = timeout_s if timeout_s is not None else server.startup_timeout_s
        self.env_extra = {**server.env, **(env_extra or {})}
        self.jailed = jailed
        self.proc: subprocess.Popen | None = None
        self.tmp: Path | None = None
        self.started_at: float | None = None
        self.ready_s: float | None = None

    def __enter__(self) -> "Served":
        argv = argv_for(self.server, self.port)
        out = open(self.log, "ab") if self.log else subprocess.DEVNULL
        self.started_at = time.monotonic()
        if self.jailed:
            self.proc, self.tmp = grader.jailed_popen(self.tree, argv, env_extra=self.env_extra, stdout=out, stderr=out)
        else:
            self.proc = subprocess.Popen(argv, cwd=self.tree, env={**os.environ, **self.env_extra}, stdout=out,
                                         stderr=out, start_new_session=True)
        try:
            self._wait_ready()
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def _wait_ready(self) -> None:
        deadline = time.monotonic() + self.timeout_s
        health = self.url + self.server.health
        while True:
            if self.proc.poll() is not None:
                raise NotReady(f"engine exited with {self.proc.returncode} before {self.server.health} answered"
                               + (f"; log: {self.log}" if self.log else ""))
            try:
                with urllib.request.urlopen(health, timeout=5) as r:
                    if r.status == 200:
                        self.ready_s = time.monotonic() - self.started_at
                        return
            except (urllib.error.URLError, OSError, ConnectionError):
                pass
            if time.monotonic() > deadline:
                raise NotReady(f"{health} not ready after {self.timeout_s:.0f}s" + (f"; log: {self.log}" if self.log else ""))
            time.sleep(0.5)

    def __exit__(self, *exc) -> None:
        if self.proc is not None and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
                self.proc.wait(timeout=30)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        if self.tmp:
            shutil.rmtree(self.tmp, ignore_errors=True)
