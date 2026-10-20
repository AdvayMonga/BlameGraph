"""ncu counters for one candidate, a fact for the search's judge: CudaForge's curated 24 Nsight Compute metrics (its
Table 8; it found them better than the full set, and KernelPro found a raw dump worse than none) plus each kernel's
duration, for the device kernels of one call on the seen shape it is given (jailed runs the smallest and largest).

`python -m kernels.evolve.ncu CONFIG PROGRAM` is the probe ncu wraps (stage 3 of kernels.evolve.jailed): it warms the
candidate up, then makes one call between cudaProfilerStart/Stop, so ncu (`--profile-from-start off`) sees only that."""
from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path

METRICS = (
    "sm__cycles_active.avg", "sm__warps_active.avg.pct_of_peak_sustained_active", "launch__occupancy_limit_blocks",
    "launch__occupancy_limit_registers", "launch__occupancy_limit_shared_mem", "launch__registers_per_thread",
    "sm__inst_executed.sum", "sm__inst_executed_pipe_fp32.avg.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active", "dram__bytes_read.sum", "dram__bytes_write.sum",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed", "dram__bytes.sum.per_second",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed", "l1tex__t_sector_hit_rate.pct",
    "l1tex__throughput.avg.pct_of_peak_sustained_active", "lts__t_sector_hit_rate.pct",
    "lts__throughput.avg.pct_of_peak_sustained_active",
    "smsp__warp_issue_stalled_memory_dependency_per_warp_active.pct",
    "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
    "smsp__warp_issue_stalled_branch_resolving_per_warp_active.pct",
    "smsp__sass_average_branch_targets_threads_uniform.pct",
    "gpu__time_duration.sum",
)


def flags() -> list[str]:
    return ["--profile-from-start", "off", "--target-processes", "all", "--csv", "--page", "raw",
            "--metrics", ",".join(METRICS)]


def parse(stdout: str, top: int = 3) -> dict:
    """ncu's raw CSV (one row per kernel launch, after its own log lines) -> the `top` kernels by duration, each with
    its metrics as numbers where they parse."""
    lines = [x for x in stdout.splitlines() if x.startswith('"')]
    if len(lines) < 2:
        return {"kernels": [], "note": "ncu printed no metrics (no kernel ran inside the profiled call?)"}
    rows = list(csv.DictReader(io.StringIO("\n".join(lines))))
    rows = [r for r in rows if r.get("Kernel Name") and r.get("ID", "").strip().isdigit()]   # drop the units row
    out = []
    for r in rows:
        m = {}
        for k in METRICS:
            v = (r.get(k) or "").replace(",", "")
            try:
                m[k] = float(v)
            except ValueError:
                m[k] = v or None
        out.append({"name": r["Kernel Name"], **m})
    out.sort(key=lambda k: -(k.get("gpu__time_duration.sum") or 0) if isinstance(k.get("gpu__time_duration.sum"),
                                                                                    float) else 0)
    return {"kernels": out[:top], "launches": len(out)}


def probe(cfg: dict, program: Path) -> int:
    import torch
    from kernels.evolve.core import load_program, screen
    from kernels.tasks import load
    refused = screen(program.read_text())
    if refused:
        print("refused: " + "; ".join(refused), file=sys.stderr)
        return 2
    task = load(cfg["task"]).with_ints(cfg.get("ints"))
    fn = task.bind(load_program(program))
    inputs = task.make_inputs(tuple(cfg["seen"][0]), task.dtypes[0], cfg.get("device", "cuda"),
                              torch.Generator().manual_seed(0))
    for _ in range(2):                           # compile and autotune outside the profiled call
        fn(*inputs)
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    fn(*inputs)
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    return 0


if __name__ == "__main__":
    sys.exit(probe(json.loads(Path(sys.argv[1]).read_text()), Path(sys.argv[2])))
