"""NVIDIA GPU state and samples through nvidia-smi (or DCGM). Everything here returns None where there is no nvidia-smi."""

from __future__ import annotations

import csv
import functools
import io
import json
import shutil
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

QUERY = ("name", "driver_version", "clocks.sm", "clocks.mem", "clocks.max.sm", "clocks.max.mem",
         "power.limit", "power.draw", "temperature.gpu", "utilization.gpu", "utilization.memory",
         "memory.used", "memory.total")
SAMPLE = ("timestamp", "utilization.gpu", "utilization.memory", "clocks.sm", "clocks.mem",
          "power.draw", "temperature.gpu", "memory.used")
REASONS = ("clocks_event_reasons.active", "clocks_throttle_reasons.active")   # the field's name on newer, older drivers
APP_CLOCKS = ("clocks.applications.graphics", "clocks.applications.memory")   # locked clocks are not queryable here
MISSING = ("[N/A]", "[Not Supported]", "N/A")
# DCGM field ids -> names: profiling counters (not coarse), then power, clocks and throttle reasons.
DCGM_FIELDS = {1002: "sm_active", 1003: "sm_occupancy", 1004: "tensor_active", 1005: "dram_active",
               1009: "pcie_tx_bytes", 1010: "pcie_rx_bytes", 155: "power_w", 100: "sm_clock_mhz",
               101: "mem_clock_mhz", 112: "throttle_reasons"}


def available() -> bool:
    return shutil.which("nvidia-smi") is not None


def _smi(fields: tuple[str, ...]) -> str | None:
    out = subprocess.run(["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits",
                          "-i", "0"], capture_output=True, text=True)
    return out.stdout if out.returncode == 0 else None


@functools.lru_cache(maxsize=None)
def supported(field: str) -> bool:
    """Whether this driver accepts `field`: one unknown field fails the whole query."""
    return available() and _smi((field,)) is not None


def reasons_field() -> str | None:
    return next((f for f in REASONS if supported(f)), None)


def query() -> dict | None:
    """One snapshot of the first GPU: identity, clocks, limits, throttle reasons; None for what the driver lacks."""
    if not available():
        return None
    reasons = reasons_field()
    extra = tuple(f for f in ((reasons,) if reasons else ()) + APP_CLOCKS if supported(f))
    out = _smi(QUERY + extra)
    if out is None:
        return None
    got = {k: (None if v in MISSING else v)
           for k, v in zip(QUERY + extra, [v.strip() for v in out.strip().split(",")])}
    snap = {k: got.get(k) for k in QUERY + APP_CLOCKS}
    snap[REASONS[0]] = got.get(reasons) if reasons else None
    return snap


def add_epoch(raw: str) -> str:
    """nvidia-smi's local-time `timestamp` column, prefixed with `epoch_s` so samples line up with timeline events."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    for i, row in enumerate(r for r in csv.reader(io.StringIO(raw), skipinitialspace=True) if r):
        if i == 0:
            w.writerow(["epoch_s", *row])
            continue
        try:
            epoch = f"{datetime.strptime(row[0], '%Y/%m/%d %H:%M:%S.%f').timestamp():.3f}"
        except ValueError:
            epoch = ""
        w.writerow([epoch, *row])
    return buf.getvalue()


class Sampler:
    """nvidia-smi sampling at `interval_ms`, as a subprocess so it costs the engine nothing; `epoch_s` added on stop."""

    def __init__(self, path: Path, interval_ms: int = 100):
        self.path, self.interval_ms, self._proc = Path(path), interval_ms, None
        self._raw = self.path.with_suffix(".smi.csv")

    def start(self) -> bool:
        if not available():
            return False
        reasons = reasons_field()
        self._file = open(self._raw, "w")
        self._proc = subprocess.Popen(
            ["nvidia-smi", f"--query-gpu={','.join(SAMPLE + ((reasons,) if reasons else ()))}",
             "--format=csv,nounits", "-lms", str(self.interval_ms), "-i", "0"],
            stdout=self._file, stderr=subprocess.DEVNULL)
        return True

    def stop(self) -> None:
        if self._proc is None:
            return
        try:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
        finally:
            self._file.close()
            self._proc = None
        self.path.write_text(add_epoch(self._raw.read_text()))
        self._raw.unlink()


class DeviceSampler(Sampler):
    """Samples for a whole served window into `dir`: `samples.csv` (each tool line prefixed `<epoch_s>,`) and
    `meta.json` (source, fields, coarse). DCGM profiling fields when `dcgmi` is on PATH, else nvidia-smi with
    throttle reasons; `{"available": false}` with neither."""

    def __init__(self, dir: Path, interval_ms: int = 100):
        super().__init__(Path(dir) / "samples.csv", interval_ms)
        self.source = "dcgm" if shutil.which("dcgmi") else "nvidia-smi" if available() else None
        self.fields = (list(DCGM_FIELDS.values()) if self.source == "dcgm"
                       else [*SAMPLE, *([reasons_field()] if reasons_field() else [])] if self.source else [])

    def start(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.source is None:
            return False
        argv = (["dcgmi", "dmon", "-i", "0", "-d", str(self.interval_ms), "-e", ",".join(map(str, DCGM_FIELDS))]
                if self.source == "dcgm" else
                ["nvidia-smi", f"--query-gpu={','.join(self.fields)}", "--format=csv,nounits",
                 "-lms", str(self.interval_ms), "-i", "0"])
        self._file = open(self.path, "w")
        self._proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        self._reader = threading.Thread(target=self._stamp, args=(self._proc, self._file), daemon=True)
        self._reader.start()
        return True

    @staticmethod
    def _stamp(proc, out) -> None:
        for line in proc.stdout:
            out.write(f"{time.time():.3f},{line.rstrip()}\n")

    def stop(self) -> None:
        meta = {"available": self.source is not None}
        if self.source:
            meta.update(source=self.source, coarse=self.source == "nvidia-smi", fields=self.fields,
                        interval_ms=self.interval_ms, line="<epoch_s>,<raw tool output line>")
            if self._proc is not None and self._proc.poll() is not None:
                meta["exited"] = self._proc.returncode       # the tool died mid-window (e.g. no DCGM host engine)
            if self._proc is not None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._proc.kill(); self._proc.wait()
                self._reader.join(timeout=5)            # drain before closing the file it writes
                self._file.close()
                self._proc = None
        (self.path.parent / "meta.json").write_text(json.dumps(meta, indent=1))


def compute_apps() -> list[dict] | None:
    """Every process holding a GPU context (host pids): pid, used MiB, name. None where there is no nvidia-smi."""
    if not available():
        return None
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory,process_name",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True)
    if out.returncode != 0:
        return None
    apps = []
    for line in out.stdout.splitlines():
        pid, mib, name = ([p.strip() for p in line.split(",", 2)] + ["", ""])[:3]
        if pid.isdigit():
            apps.append({"pid": int(pid), "used_mib": int(mib) if mib.isdigit() else None, "name": name})
    return apps


def descendants(pid: int) -> set[int]:
    """`pid` and every process below it, from /proc (Linux); `{pid}` where there is no /proc."""
    children: dict[int, list[int]] = {}
    for stat in Path("/proc").glob("[0-9]*/stat"):
        try:
            ppid = int(stat.read_text().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            continue
        children.setdefault(ppid, []).append(int(stat.parent.name))
    out, todo = set(), [pid]
    while todo:
        p = todo.pop()
        if p not in out:
            out.add(p)
            todo += children.get(p, [])
    return out
