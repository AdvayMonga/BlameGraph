"""Instrumented runs of the agent's pristine tree, one tool per instrument. Each wraps lab.profile's workload inside
the jail with fixed, validated arguments, keeps the raw output as a ledger blob and returns a short raw summary.

  trace     Nsight Systems: timeline of the measured window (.nsys-rep) plus `nsys stats` CSV tables
  kernel    Nsight Compute: kernels whose name matches a regex (.ncu-rep) plus `ncu --import --csv --page raw`
  hostprof  py-spy: the engine process sampled for D seconds under a sustained workload (speedscope) plus one dump
"""

from __future__ import annotations

import re
import shutil
import time
from pathlib import Path

from lab import engine
from lab.agent import ToolSpec

OUT = "_instr"                      # in the pristine tree: writable in the jail, rebuilt on every call
NSYS_REPORTS = "cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum,nvtx_sum"
NCU_SETS = ("basic", "full")
KERNEL_RE = re.compile(r"[\w.*+?|()\[\]{}^$\\:,<>~-]{1,200}")    # regex syntax only: no spaces, quotes, ; & ` or /
SUMMARY_CHARS = 2000
WORKLOAD = {"requests": (8, 1, 256), "prompt_len": (64, 1, 32768), "max_tokens": (32, 1, 4096)}


def _int(args: dict, name: str, default: int, lo: int, hi: int) -> int:
    v = args.get(name, default)
    if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi:
        raise ValueError(f"{name} must be an integer in [{lo}, {hi}]")
    return v


def workload(args: dict) -> list[str]:
    """lab.profile's workload flags from validated integers."""
    return [w for k, (d, lo, hi) in WORKLOAD.items() for w in (f"--{k.replace('_', '-')}", str(_int(args, k, d, lo, hi)))]


def kernel_regex(value) -> str:
    if not isinstance(value, str) or not KERNEL_RE.fullmatch(value) or "$(" in value:
        raise ValueError("kernel_regex must be 1-200 characters: letters, digits and _.*+?|()[]{}^$\\:,<>~-")
    try:
        re.compile(value)
    except re.error as e:
        raise ValueError(f"kernel_regex is not a regex: {e}") from None
    return value


def _schema(**extra) -> dict:
    props = {k: {"type": "integer", "minimum": lo, "maximum": hi, "default": d} for k, (d, lo, hi) in WORKLOAD.items()}
    return {"type": "object", "properties": {**props, **extra}}


def _head(path: Path | None) -> str:
    return path.read_text(errors="replace")[:SUMMARY_CHARS] if path and path.is_file() else ""


class ProfTools:
    """Mixin for lab.tools.Toolbox: needs self._audited, self._pristine, self._record, self._keep."""

    def trace(self, args: dict) -> str:
        wl = workload(args)
        wrap = lambda exe, out: [exe, "profile", "--output", str(out / "trace"), "--force-overwrite", "true",
                                 "--trace", "cuda,nvtx,osrt", "--capture-range", "cudaProfilerApi",
                                 "--capture-range-end", "stop", "--stats", "false"]
        post = lambda exe, out: ([exe, "stats", "--report", NSYS_REPORTS, "--format", "csv", "--output",
                                  str(out / "stats"), str(out / "trace.nsys-rep")], None)
        return self._instrumented("trace", args, "nsys", wrap, wl + ["--profiler", "cuda-range"], post,
                                  "stats_cuda_gpu_kern_sum.csv")

    def kernel(self, args: dict) -> str:
        rx = kernel_regex(args.get("kernel_regex"))
        skip, count = _int(args, "launch_skip", 0, 0, 100000), _int(args, "launch_count", 8, 1, 64)
        kset = args.get("set", "basic")
        if kset not in NCU_SETS:
            raise ValueError(f"set must be one of {NCU_SETS}")
        wl = workload(args)
        wrap = lambda exe, out: [exe, "--export", str(out / "kernel"), "--force-overwrite", "--kernel-name",
                                 f"regex:{rx}", "--launch-skip", str(skip), "--launch-count", str(count), "--set", kset,
                                 "--profile-from-start", "off", "--target-processes", "all"]
        post = lambda exe, out: ([exe, "--import", str(out / "kernel.ncu-rep"), "--csv", "--page", "raw"],
                                 out / "metrics.csv")
        return self._instrumented("kernel", args, "ncu", wrap, wl + ["--profiler", "cuda-range"], post, "metrics.csv")

    def hostprof(self, args: dict) -> str:
        wl = workload(args)
        seconds, rate = _int(args, "seconds", 20, 1, 300), _int(args, "rate", 100, 1, 1000)
        extra = lambda exe: ["--profiler", "pyspy", "--seconds", str(seconds), "--rate", str(rate), "--pyspy-bin", exe]
        return self._instrumented("hostprof", args, "py-spy", lambda exe, out: [], wl, None,
                                  "bundle/*/pyspy-dump.txt", extra)

    def _instrumented(self, tool: str, args: dict, binary: str, wrap, flags: list[str], post, summary: str,
                      extra=lambda exe: []) -> str:
        """`wrap(exe, out)` + lab.profile `flags`, jailed with the instrument readable; then `post`, jailed too
        (the report was written by agent code). Kept as one blob; the head of `summary` is returned."""
        from lab import tools as T
        snap = self._audited(tool, args)
        exe = shutil.which(binary)
        if not exe:
            reason = f"{binary} is not on PATH on this host"
            self._record("note", tool, args, {"verdict": "refused", "reason": reason}, snap)
            return f"{tool} refused: {reason}"
        exe = str(Path(exe).resolve())
        tree = self._pristine()
        out = tree / OUT
        out.mkdir()
        T.stage_harness(tree)
        read = [Path(exe).parent]
        argv = [*wrap(exe, out), engine.python(), "-m", "lab.profile", *flags, *extra(exe), "--out", str(out / "bundle")]
        t0 = time.monotonic()
        proc = T.default_profile_runner(tree, argv, read=read)
        log = proc.stdout + proc.stderr
        commands = [argv]
        if post and proc.returncode == 0:
            pargv, dest = post(exe, out)
            commands.append(pargv)
            p = T.default_profile_runner(tree, pargv, read=read)
            if dest:
                dest.write_text(p.stdout)
            log += p.stderr if dest else p.stdout + p.stderr
            proc = p if p.returncode else proc
        found = sorted(out.glob(summary))
        head = _head(found[-1] if found else None)
        blob, visible = self._keep(out) if any(p.is_file() for p in out.rglob("*")) else ("", "")
        result = {"returncode": proc.returncode, "bundle": blob, "workspace_copy": visible, "instrument": exe,
                  "argv": commands, "seconds": time.monotonic() - t0, "output": log[-3000:]}
        self._record("profile", tool, args, result, snap)
        if proc.returncode != 0 or not head:
            return f"{tool} failed ({proc.returncode}):\n{result['output']}"
        return f"{tool} output at {visible} (in your workspace; left out of your change)\n{head}"

    def prof_specs(self, gpu) -> list[ToolSpec]:
        """`gpu(time)` renders the cost fact for a description."""
        return [
            ToolSpec("trace", "Run the profile workload of your current workspace under Nsight Systems (nsys), "
                     "capturing the measured window only. Stores the .nsys-rep and `nsys stats` CSV tables (GPU "
                     "kernels, CUDA API, memory copies, NVTX) in the ledger; returns the head of the kernel table. "
                     "Refused where nsys is not installed." + gpu("3-6 min (engine start, workload, report export)"),
                     _schema(), self.trace),
            ToolSpec("kernel", "Run the profile workload under Nsight Compute (ncu) on the kernels whose name "
                     "matches `kernel_regex`: skips `launch_skip` matching launches, then profiles `launch_count` "
                     "with metric set `set`. Stores the .ncu-rep and its raw metrics CSV in the ledger; returns the "
                     "head of the CSV. Refused where ncu is not installed."
                     + gpu("5-20 min (every profiled launch is replayed once per metric pass; set=full has the most)"),
                     _schema(kernel_regex={"type": "string", "maxLength": 200},
                             launch_skip={"type": "integer", "minimum": 0, "maximum": 100000, "default": 0},
                             launch_count={"type": "integer", "minimum": 1, "maximum": 64, "default": 8},
                             set={"type": "string", "enum": list(NCU_SETS), "default": "basic"})
                     | {"required": ["kernel_regex"]}, self.kernel),
            ToolSpec("hostprof", "Sample the engine process with py-spy for `seconds` while the profile workload "
                     "repeats (scheduler and backend in process, warmup excluded). Stores a speedscope profile and one "
                     "stack dump taken halfway in the ledger; returns the head of the dump. Refused where py-spy is "
                     "not installed." + gpu("2-4 min plus `seconds`"),
                     _schema(seconds={"type": "integer", "minimum": 1, "maximum": 300, "default": 20},
                             rate={"type": "integer", "minimum": 1, "maximum": 1000, "default": 100}), self.hostprof),
        ]
