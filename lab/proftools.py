"""Instrumented runs of the agent's pristine tree, one tool per instrument. Each wraps lab.profile's workload inside
the jail with fixed, validated arguments, keeps the raw output as a ledger blob and returns a short raw summary.

  trace     Nsight Systems: timeline of the measured window (.nsys-rep) plus `nsys stats` CSV tables
  kernel    Nsight Compute: kernels whose name matches a regex (.ncu-rep) plus `ncu --import --csv --page raw`
  hostprof  py-spy: the engine process sampled for D seconds under a sustained workload (speedscope) plus one dump
"""

from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path

from lab.agent import ToolSpec
from lab.evaltools import _killed

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
    props.update(corpus_class={"type": "string"}, seed={"type": "integer"}, synthetic={"type": "boolean"})
    return {"type": "object", "properties": {**props, **extra}}


def _head(path: Path | None) -> str:
    return path.read_text(errors="replace")[:SUMMARY_CHARS] if path and path.is_file() and not path.is_symlink() else ""


def instrument(kind: str, args: dict) -> dict:
    """One instrument from validated args: its binary, `wrap(exe, out)` around lab.profile, lab.profile `flags`,
    `post(exe, out)` -> (argv, file for its stdout) run after, the `summary` glob, and `extra(exe)` flags."""
    wl = workload(args)
    if kind == "trace":
        return {"binary": "nsys", "flags": wl + ["--profiler", "cuda-range"], "summary": "stats_cuda_gpu_kern_sum.csv",
                "wrap": lambda exe, out: [exe, "profile", "--output", str(out / "trace"), "--force-overwrite", "true",
                                          "--trace", "cuda,nvtx,osrt", "--capture-range", "cudaProfilerApi",
                                          "--capture-range-end", "stop", "--cuda-graph-trace", "node", "--stats", "false"],
                "post": lambda exe, out: ([exe, "stats", "--report", NSYS_REPORTS, "--format", "csv", "--output",
                                           str(out / "stats"), str(out / "trace.nsys-rep")], None),
                "extra": lambda exe: []}
    if kind == "kernel":
        rx = kernel_regex(args.get("kernel_regex"))
        skip, count = _int(args, "launch_skip", 0, 0, 100000), _int(args, "launch_count", 8, 1, 64)
        kset = args.get("set", "basic")
        if kset not in NCU_SETS:
            raise ValueError(f"set must be one of {NCU_SETS}")
        return {"binary": "ncu", "flags": wl + ["--profiler", "cuda-range"], "summary": "metrics.csv",
                "wrap": lambda exe, out: [exe, "--export", str(out / "kernel"), "--force-overwrite", "--kernel-name",
                                          f"regex:{rx}", "--launch-skip", str(skip), "--launch-count", str(count),
                                          "--set", kset, "--profile-from-start", "off", "--target-processes", "all"],
                "post": lambda exe, out: ([exe, "--import", str(out / "kernel.ncu-rep"), "--csv", "--page", "raw"],
                                          out / "metrics.csv"),
                "extra": lambda exe: []}
    if kind == "hostprof":
        seconds, rate = _int(args, "seconds", 20, 1, 300), _int(args, "rate", 100, 1, 1000)
        return {"binary": "py-spy", "flags": wl, "summary": "bundle/*/pyspy-dump.txt", "wrap": lambda exe, out: [],
                "post": None,
                "extra": lambda exe: ["--profiler", "pyspy", "--seconds", str(seconds), "--rate", str(rate),
                                      "--pyspy-bin", exe]}
    raise ValueError(f"unknown instrument {kind!r}")


class ProfTools:
    """Mixin for lab.tools.Toolbox: needs self._audited, self._pristine, self._record, self._keep, self._work,
    self.worker."""

    def trace(self, args: dict) -> str:
        return self._instrumented("trace", args)

    def kernel(self, args: dict) -> str:
        return self._instrumented("kernel", args)

    def hostprof(self, args: dict) -> str:
        return self._instrumented("hostprof", args)

    def _instrumented(self, tool: str, args: dict) -> str:
        """The instrument around lab.profile's workload, then its report step, both jailed on the worker (the report
        was written by agent code). Kept as one blob; the head of its summary is returned."""
        instrument(tool, args)                  # refuse bad arguments before anything is snapshotted or run
        snap = self._audited(tool, args)
        tree = self._pristine()
        out = self._work(tool)
        t0 = time.monotonic()
        try:
            r = self.worker.call("profile", {"tree": tree},
                                 {"kind": tool, "args": args, "workbench": getattr(self.s, "workbench", None)}, out)
            if r.get("refused"):
                self._record("note", tool, args, {"verdict": "refused", "reason": r["refused"]}, snap)
                return f"{tool} refused: {r['refused']}"
            files = out / "files"
            blob, visible = self._keep(files) if files.is_dir() and any(p.is_file() for p in files.rglob("*")) else ("", "")
        finally:
            shutil.rmtree(out, ignore_errors=True)
        result = {"returncode": r["returncode"], "bundle": blob, "workspace_copy": visible, "instrument": r["instrument"],
                  "argv": r["argv"], "seconds": time.monotonic() - t0, "output": r["output"], **_killed(r)}
        self._record("profile", tool, args, result, snap)
        killed = f"\nkilled in your workbench (held the GPU): {json.dumps(r['killed'])}" if r.get("killed") else ""
        if r["returncode"] != 0 or not r["head"]:
            return f"{tool} failed ({r['returncode']}):\n{result['output']}{killed}"
        return f"{tool} output at {visible} (in your workspace; left out of your change)\n{r['head']}{killed}"

    def prof_specs(self, gpu) -> list[ToolSpec]:
        """`gpu(time)` renders the cost fact for a description."""
        return [
            ToolSpec("trace", "Run the profile workload of your current workspace under Nsight Systems (nsys), "
                     "capturing the measured window only. Stores the .nsys-rep and `nsys stats` CSV tables (GPU "
                     "kernels, CUDA API, memory copies, NVTX) in the ledger; returns the head of the kernel table. "
                     "Refused where nsys is not installed." + gpu("about 1 min measured at 16 requests x 64 tokens on an H200 (engine start, workload, report export)"),
                     _schema(), self.trace),
            ToolSpec("kernel", "Run the profile workload under Nsight Compute (ncu) on the kernels whose name "
                     "matches `kernel_regex`: skips `launch_skip` matching launches, then profiles `launch_count` "
                     "with metric set `set`. Stores the .ncu-rep and its raw metrics CSV in the ledger; returns the "
                     "head of the CSV. Refused where ncu is not installed."
                     + gpu("about 1.5 min measured at 4-8 launches on an H200; each profiled launch is replayed once per metric pass, so it grows with launch_count and set=full"),
                     _schema(kernel_regex={"type": "string", "maxLength": 200},
                             launch_skip={"type": "integer", "minimum": 0, "maximum": 100000, "default": 0},
                             launch_count={"type": "integer", "minimum": 1, "maximum": 64, "default": 8},
                             set={"type": "string", "enum": list(NCU_SETS), "default": "basic"})
                     | {"required": ["kernel_regex"]}, self.kernel),
            ToolSpec("hostprof", "Sample the engine process with py-spy for `seconds` while the profile workload "
                     "repeats (scheduler and backend in process, warmup excluded). Stores a speedscope profile and one "
                     "stack dump taken halfway in the ledger; returns the head of the dump. Refused where py-spy is "
                     "not installed." + gpu("about 35 s plus `seconds`, measured on an H200"),
                     _schema(seconds={"type": "integer", "minimum": 1, "maximum": 300, "default": 20},
                             rate={"type": "integer", "minimum": 1, "maximum": 1000, "default": 100}), self.hostprof),
        ]
