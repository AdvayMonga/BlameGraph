"""NVIDIA GPU state and samples through nvidia-smi. Everything here returns None where there is no nvidia-smi."""

from __future__ import annotations

import csv
import functools
import io
import shutil
import subprocess
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
