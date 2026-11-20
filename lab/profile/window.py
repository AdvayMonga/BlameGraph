"""The profiling window inside every Python process of a served engine (attached by hook/sitecustomize.py).
LAB_PROFILE (JSON) gives the window's dir and profiler. In that dir, the file `start` opens the window in each process
that takes part: a GPU process (one that loaded torch and initialized CUDA, or any torch process where there is no CUDA)
runs the profiler, captures and memory readings; the first process (the launch's) runs py-spy over the process tree.
Each writes procs/<pid>/: `started`, then after `stop` its files with window.json last, or error.txt. The file `end`
makes the first process send itself SIGTERM."""
from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

POLL_S = 0.05
ROOT_ENV = "LAB_PROFILE_ROOT"           # the first process's pid, inherited by the processes it starts


def _wait(path: Path) -> None:
    while not path.exists():
        time.sleep(POLL_S)


def _torch_profiler(torch, device: str):
    """A torch profiler that records all threads where torch supports it, and its config."""
    from torch.profiler import ProfilerActivity
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device == "cuda" else [])
    config = {"activities": [a.name for a in acts], "all_threads": True}
    try:
        from torch._C._profiler import _ExperimentalConfig
        exp = _ExperimentalConfig(profile_all_threads=True)
    except (ImportError, TypeError):             # older torch: the thread that starts it only
        return torch.profiler.profile(activities=acts), {**config, "all_threads": False}
    return torch.profiler.profile(activities=acts, experimental_config=exp), config


def _allow_ptrace() -> None:
    """Allow any process to ptrace this one (Linux), for py-spy."""
    if sys.platform.startswith("linux"):
        import ctypes
        ctypes.CDLL(None).prctl(0x59616D61, ctypes.c_ulong(-1), 0, 0, 0)    # PR_SET_PTRACER, PR_SET_PTRACER_ANY


def _gpu(torch) -> bool:
    return torch is not None and (torch.cuda.is_initialized() or not torch.cuda.is_available())


class Window:
    """This process's part of the window: the profiler, captures and memory readings (a GPU process), py-spy (the
    first process)."""

    def __init__(self, cfg: dict, dir: Path, root: bool):
        self.cfg, self.dir, self.root, self.stack = cfg, dir, root, contextlib.ExitStack()
        self.torch = sys.modules.get("torch")
        self.gpu = _gpu(self.torch)
        self.device = ("cuda" if self.torch.cuda.is_available() else "cpu") if self.gpu else None
        self.out = dir / "procs" / str(os.getpid())
        self.info: dict = {"pid": os.getpid(), "root": root, "gpu": self.gpu, "device": self.device,
                           "torch": self.torch.__version__ if self.torch else None, "profiler": {"kind": cfg["profiler"]}}

    @property
    def takes_part(self) -> bool:
        return self.gpu or (self.root and self.cfg["profiler"] == "pyspy")

    def open(self) -> None:
        cfg, torch = self.cfg, self.torch
        if self.root and cfg["profiler"] == "pyspy":
            self._pyspy()
        if not self.gpu:
            return
        from lab.profile import bundle, capture
        entries = [c for c in cfg["capture"] if not c.startswith("region:")]
        if entries:
            self.stack.enter_context(capture.capture(entries, self.dir / "captures", cfg["capture_bytes"]))
        if capture.STEP in cfg["capture"]:
            self.stack.enter_context(capture.step(self.dir / "captures"))
        self.peak_before = bundle.device_peaks(self.device)
        bundle.reset_peaks(self.device)
        self.mem0 = bundle.device_memory(self.device)
        if cfg["profiler"] == "torch":
            self.prof, config = _torch_profiler(torch, self.device)
            self.info["profiler"].update(config)
            self.prof.start()
        elif cfg["profiler"] == "cuda-range" and self.device == "cuda":
            torch.cuda.profiler.start()

    def _pyspy(self) -> None:
        """py-spy records the process tree for its seconds; one stack dump of it halfway."""
        p, pid = self.cfg["pyspy"], str(os.getpid())
        log = open(self.dir / "pyspy-record.log", "w")
        self.rec = subprocess.Popen([p["exe"], "record", "--pid", pid, "--subprocesses", "--duration",
                                     str(int(p["seconds"])), "--rate", str(p["rate"]), "--format", "speedscope",
                                     "--output", str(self.dir / "pyspy.speedscope.json"), "--nonblocking", *p["extra"]],
                                    stdout=log, stderr=log)

        def dump():
            time.sleep(p["seconds"] / 2)
            d = subprocess.run([p["exe"], "dump", "--pid", pid, "--subprocesses", "--nonblocking"],
                               capture_output=True, text=True)
            (self.dir / "pyspy-dump.txt").write_text(d.stdout + d.stderr)
        self.dumper = threading.Thread(target=dump, daemon=True)
        self.dumper.start()

    def close(self) -> None:
        cfg, torch, out = self.cfg, self.torch, self.out
        if self.root and cfg["profiler"] == "pyspy":
            self.rec.wait(cfg["pyspy"]["seconds"] + 60)
            self.dumper.join()
        if not self.gpu:
            return
        from lab.profile import bundle
        if cfg["profiler"] == "torch":
            self.prof.stop()
        elif cfg["profiler"] == "cuda-range" and self.device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.profiler.stop()
        self.stack.close()
        if cfg["profiler"] == "torch":
            self.prof.export_chrome_trace(str(out / "trace.json"))
            bundle.write_json(out / "ops.json", bundle.ops_table(self.prof))
            kernels = bundle.kernels_table(self.prof.events())
            if kernels is not None:
                bundle.write_json(out / "kernels.json", kernels)
        peak, mem1 = bundle.device_peaks(self.device), bundle.device_memory(self.device)
        bundle.write_json(out / "memory.json", {
            "device": self.device, "window_peak": peak,
            "lifetime_peak": None if peak is None else
            {k: max(v, (self.peak_before or {}).get(k, 0)) for k, v in peak.items()},
            "host_rss_peak_lifetime_bytes": mem1["peak_host_rss_bytes"],
            "at_window_start": self.mem0, "at_window_end": mem1})


def _control(cfg: dict, root: bool) -> None:
    d = Path(cfg["dir"])
    w = None
    try:
        _wait(d / "start")
        w = Window(cfg, d, root)
        if w.takes_part:
            w.out.mkdir(parents=True, exist_ok=True)
            w.open()
            (w.out / "started").write_text(str(time.time()))
            _wait(d / "stop")
            w.close()
            (w.out / "window.json").write_text(json.dumps(w.info, indent=1))
        if root:
            _wait(d / "end")
            os.kill(os.getpid(), signal.SIGTERM)
    except BaseException:                        # noqa: BLE001  reported to the lab
        out = d / "procs" / str(os.getpid())
        out.mkdir(parents=True, exist_ok=True)
        (out / "error.txt").write_text(traceback.format_exc())


def attach() -> None:
    """Join the window: called once at the start of each process."""
    cfg = json.loads(os.environ["LAB_PROFILE"])
    root = ROOT_ENV not in os.environ
    if root:
        os.environ[ROOT_ENV] = str(os.getpid())
        for c in cfg["capture"]:                 # the engine's own region capture, from launch on
            from lab.profile.capture import STEP
            if c.startswith("region:") and c != STEP:
                os.environ.update(CAPTURE_REGION=c.removeprefix("region:"),
                                  CAPTURE_REGION_DIR=str(Path(cfg["dir"]) / "captures"))
    _allow_ptrace()
    threading.Thread(target=_control, args=(cfg, root), daemon=True).start()
