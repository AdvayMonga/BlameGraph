"""Hardware peaks the ceilings are computed against: HBM bandwidth and bf16 tensor-core throughput, measured once
per GPU + driver (`python -m validity peaks`, run by lab/vm-setup.sh) into knowledge/peaks/<gpu>-<driver>.json. Spec-sheet
numbers are the fallback, labelled `source: spec`; a ceiling from them says so."""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from lab import target

PEAKS_DIR = target.ENV_ROOT / "knowledge" / "peaks"

# Vendor spec sheets: dense bf16 tensor-core FLOP/s (no sparsity), HBM bytes/s, HBM bytes.
SPEC = {
    "H200": (989e12, 4.8e12, 141e9), "H100": (989e12, 3.35e12, 80e9), "B200": (2250e12, 8e12, 192e9),
    "A100": (312e12, 2.0e12, 80e9), "L40S": (362e12, 0.864e12, 48e9), "A10G": (125e12, 0.6e12, 24e9),
}


@dataclass(frozen=True)
class Peaks:
    gpu: str                 # nvidia-smi's name
    bf16_flops: float        # dense bf16 FLOP/s
    bw_bytes_s: float        # HBM bytes/s
    mem_bytes: float
    source: str              # measured | spec
    driver: str | None = None
    sm_clock_mhz: int | None = None
    measured_at: str | None = None
    method: str | None = None


def slug(gpu: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", gpu.lower()).strip("-")


def spec_for(gpu: str) -> Peaks | None:
    for key, (flops, bw, mem) in SPEC.items():
        if key.lower() in gpu.lower():
            return Peaks(gpu, flops, bw, mem, "spec")
    return None


def path_for(gpu: str, driver: str | None, peaks_dir: Path | None = None) -> Path:
    return (peaks_dir or PEAKS_DIR) / (f"{slug(gpu)}-{slug(driver)}.json" if driver else f"{slug(gpu)}.json")


def load(gpu: str, driver: str | None = None, peaks_dir: Path | None = None) -> Peaks | None:
    """Measured peaks for this GPU and driver; without a driver, the latest measurement of the GPU; else the spec
    sheet; else None. A measurement on another driver is not used."""
    d = peaks_dir or PEAKS_DIR
    if driver:
        p = path_for(gpu, driver, d)
        return Peaks(**json.loads(p.read_text())) if p.exists() else spec_for(gpu)
    found = sorted(d.glob(f"{slug(gpu)}*.json"), key=lambda q: json.loads(q.read_text()).get("measured_at") or "")
    return Peaks(**json.loads(found[-1].read_text())) if found else spec_for(gpu)


def measure(seconds: float = 3.0) -> Peaks:
    """Copy bandwidth and a large bf16 GEMM on the first CUDA device; the best of repeated trials (clocks warm)."""
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device")
    dev = torch.device("cuda:0")
    name = torch.cuda.get_device_name(dev)
    props = torch.cuda.get_device_properties(dev)
    n = 1 << 28                                                 # 512 MiB per bf16 buffer, well past L2
    a, b = torch.empty(n, dtype=torch.bfloat16, device=dev), torch.empty(n, dtype=torch.bfloat16, device=dev)
    best_bw = 0.0
    t_end = time.monotonic() + seconds
    while time.monotonic() < t_end:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        b.copy_(a)
        torch.cuda.synchronize()
        best_bw = max(best_bw, 2 * n * 2 / (time.perf_counter() - t0))   # read + write, 2 bytes each
    del a, b
    k = 8192
    x = torch.randn(k, k, dtype=torch.bfloat16, device=dev)
    y = torch.randn(k, k, dtype=torch.bfloat16, device=dev)
    best_flops = 0.0
    t_end = time.monotonic() + seconds
    while time.monotonic() < t_end:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        x @ y
        torch.cuda.synchronize()
        best_flops = max(best_flops, 2 * k ** 3 / (time.perf_counter() - t0))
    driver = sm = None
    try:
        import subprocess
        out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version,clocks.sm", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.strip().split(",")
        driver, sm = out[0].strip(), int(out[1])
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        pass
    return Peaks(name, best_flops, best_bw, float(props.total_memory), "measured", driver, sm,
                 time.strftime("%Y-%m-%dT%H:%M:%S"), f"best of {seconds}s: 512MiB bf16 copy; {k}^3 bf16 GEMM")


def write(p: Peaks, peaks_dir: Path | None = None) -> Path:
    d = peaks_dir or PEAKS_DIR
    d.mkdir(parents=True, exist_ok=True)
    out = path_for(p.gpu, p.driver, d)
    out.write_text(json.dumps(asdict(p), indent=1) + "\n")
    return out
