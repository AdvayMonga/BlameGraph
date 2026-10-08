"""NVIDIA GPU state and samples through nvidia-smi (or DCGM). Everything here returns None where there is no nvidia-smi."""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
from pathlib import Path

QUERY = ("name", "driver_version", "clocks.sm", "clocks.mem", "clocks.max.sm", "clocks.max.mem",
         "power.limit", "power.draw", "temperature.gpu", "utilization.gpu", "utilization.memory",
         "memory.used", "memory.total")
SAMPLE = ("timestamp", "utilization.gpu", "utilization.memory", "clocks.sm", "clocks.mem",
          "power.draw", "temperature.gpu", "memory.used")
# DCGM field ids -> names: profiling counters (not coarse), then power, clocks and throttle reasons.
DCGM_FIELDS = {1002: "sm_active", 1003: "sm_occupancy", 1004: "tensor_active", 1005: "dram_active",
               1009: "pcie_tx_bytes", 1010: "pcie_rx_bytes", 155: "power_w", 100: "sm_clock_mhz",
               101: "mem_clock_mhz", 112: "throttle_reasons"}


def available() -> bool:
    return shutil.which("nvidia-smi") is not None


def query() -> dict | None:
    """One snapshot of the first GPU: identity, clocks and limits. The clock state that travels with a number."""
    if not available():
        return None
    out = subprocess.run(["nvidia-smi", f"--query-gpu={','.join(QUERY)}", "--format=csv,noheader,nounits",
                          "-i", "0"], capture_output=True, text=True)
    if out.returncode != 0:
        return None
    return dict(zip(QUERY, [v.strip() for v in out.stdout.strip().split(",")]))


class Sampler:
    """nvidia-smi sampling into a CSV at `interval_ms`, as a subprocess so it costs the engine nothing."""

    def __init__(self, path: Path, interval_ms: int = 100):
        self.path, self.interval_ms, self._proc = Path(path), interval_ms, None

    def start(self) -> bool:
        if not available():
            return False
        self._file = open(self.path, "w")
        self._proc = subprocess.Popen(
            ["nvidia-smi", f"--query-gpu={','.join(SAMPLE)}", "--format=csv,nounits",
             "-lms", str(self.interval_ms), "-i", "0"], stdout=self._file, stderr=subprocess.DEVNULL)
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


class DeviceSampler(Sampler):
    """Samples for a whole served window into `dir`: `samples.csv` (each tool line prefixed `<epoch_s>,`) and
    `meta.json` (source, fields, coarse). DCGM profiling fields when `dcgmi` is on PATH, else nvidia-smi with
    throttle reasons; `{"available": false}` with neither."""

    def __init__(self, dir: Path, interval_ms: int = 100):
        super().__init__(Path(dir) / "samples.csv", interval_ms)
        self.source = "dcgm" if shutil.which("dcgmi") else "nvidia-smi" if available() else None
        self.fields = (list(DCGM_FIELDS.values()) if self.source == "dcgm"
                       else [*SAMPLE, "clocks_throttle_reasons.active"] if self.source else [])

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
