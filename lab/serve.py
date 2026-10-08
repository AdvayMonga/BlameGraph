"""Serve an engine tree the way the agent cannot tamper with: launched from the pristine copy, inside the jail, on a
free localhost port, with the target's env; ready when its health path answers 200; torn down with its whole
process group. `with Served(tree, spec) as s: ... s.url`. With `artifacts`, the run's passive data lands in that dir:
`device/` (samples for the whole served window), `telemetry/` (files the engine wrote, when the target asks for them)
and `serve.log` (this run's part of the log)."""

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

from lab import engine, gpu, target
from lab.safety import grader

TELEMETRY_SUBDIR = "lab-telemetry"        # inside the served tree, the one place a jailed engine may write


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
                 timeout_s: float | None = None, env_extra: dict | None = None, jailed: bool = True,
                 artifacts: Path | None = None):
        self.tree, self.server = Path(tree), server
        self.port = port or free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.log = Path(log) if log else None
        self.timeout_s = timeout_s if timeout_s is not None else server.startup_timeout_s
        self.env_extra = {**server.env, **(env_extra or {})}
        self.jailed = jailed
        self.artifacts = Path(artifacts) if artifacts else None
        self.telemetry_dir = self.tree.resolve() / TELEMETRY_SUBDIR if self.artifacts and server.telemetry_env else None
        if self.telemetry_dir:
            self.env_extra.update({k: v.replace("{dir}", str(self.telemetry_dir)) for k, v in server.telemetry_env.items()})
        self.sampler: gpu.DeviceSampler | None = None
        self.log_offset = 0
        self.proc: subprocess.Popen | None = None
        self.tmp: Path | None = None
        self.started_at: float | None = None
        self.ready_s: float | None = None

    def __enter__(self) -> "Served":
        argv = argv_for(self.server, self.port)
        self.log_offset = self.log.stat().st_size if self.log and self.log.exists() else 0
        if self.artifacts:
            self.sampler = gpu.DeviceSampler(self.artifacts / "device")
            self.sampler.start()
        if self.telemetry_dir:
            shutil.rmtree(self.telemetry_dir, ignore_errors=True)
            self.telemetry_dir.mkdir(parents=True)
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
        if self.artifacts:
            self._collect()

    def _collect(self) -> None:
        """Device samples closed, telemetry moved out of the tree (empty means none), this run's log copied."""
        if self.sampler is not None:
            self.sampler.stop()
            self.sampler = None
        if self.telemetry_dir and self.telemetry_dir.is_dir():
            if any(self.telemetry_dir.iterdir()):
                shutil.move(self.telemetry_dir, self.artifacts / "telemetry")
            else:
                self.telemetry_dir.rmdir()
        if self.log and self.log.exists():
            with open(self.log, "rb") as f:
                f.seek(self.log_offset)
                (self.artifacts / "serve.log").write_bytes(f.read())
