"""Instrumented runs of the agent's pristine tree, one tool per instrument, all through one path (`instrument` on the
worker, `run_instrument` here): lab.profile's workload inside the jail, the instrument's own flags passed through as
an argument list, the GPU watched like a served engine, one record shape (verdict ok | failed | refused |
contaminated), the raw output kept as a ledger blob and a short raw summary returned.

  profile   torch.profiler: the bundle (event timeline, chrome trace, op and kernel tables, memory) with layer facts
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

from lab import calibrate, layers, profiles, serve, target
from lab.agent import ToolSpec
from lab.evaltools import _failed, _killed

PROFILE_ARG = " `profile`: run the engine under that regime's launch profile (default the general one)."
OUT = "_instr"                      # in the pristine tree: writable in the jail, rebuilt on every call
NSYS_REPORTS = "cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum,nvtx_sum"
NSYS_FLAGS = ["--trace", "cuda,nvtx,osrt", "--capture-range", "cudaProfilerApi", "--capture-range-end", "stop",
              "--cuda-graph-trace", "node"]
SUMMARY_CHARS = 2000
WORKLOAD = {"requests": (8, 1), "prompt_len": (64, 1), "max_tokens": (32, 1)}   # (default, minimum)


def _int(args: dict, name: str, default: int, lo: int) -> int:
    v = args.get(name, default)
    if isinstance(v, bool) or not isinstance(v, int) or v < lo:
        raise ValueError(f"{name} must be an integer >= {lo}")
    return v


def _flags(args: dict, name: str, default: list[str] | None = None) -> list[str]:
    """An instrument's own flags, passed through as argv items (never through a shell)."""
    v = args.get(name, default if default is not None else [])
    if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
        raise ValueError(f"{name} must be a list of strings")
    return list(v)


def workload(args: dict) -> list[str]:
    """lab.profile's workload flags from validated integers."""
    return [w for k, (d, lo) in WORKLOAD.items() for w in (f"--{k.replace('_', '-')}", str(_int(args, k, d, lo)))]


def kernel_regex(value) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("kernel_regex must be a non-empty string")
    try:
        re.compile(value)
    except re.error as e:
        raise ValueError(f"kernel_regex is not a regex: {e}") from None
    return value


FLAGS = {"type": "array", "items": {"type": "string"}}


def _schema(**extra) -> dict:
    props = {k: {"type": "integer", "minimum": lo, "default": d} for k, (d, lo) in WORKLOAD.items()}
    props.update(corpus_class={"type": "string"}, seed={"type": "integer"}, synthetic={"type": "boolean"})
    return {"type": "object", "properties": {**props, **extra}}


def _head(path: Path | None) -> str:
    return path.read_text(errors="replace")[:SUMMARY_CHARS] if path and path.is_file() and not path.is_symlink() else ""


def instrument(kind: str, args: dict) -> dict:
    """One instrument from validated args: its binary (None: torch, in lab.profile itself), `wrap(exe, out)` around
    lab.profile, lab.profile `flags`, `post(exe, out)` -> (argv, file for its stdout) run after, the `summary` glob,
    `extra(exe)` flags, and `bundle_only`: the blob is lab.profile's bundle rather than the whole output dir."""
    wl = workload(args)
    if kind == "profile":
        capture = [w for f in _flags(args, "capture") for w in ("--capture", f)]
        return {"binary": None, "flags": wl + ["--profiler", "torch", *capture], "summary": None,
                "wrap": lambda exe, out: [], "post": None, "extra": lambda exe: [], "bundle_only": True}
    if kind == "trace":
        flags = _flags(args, "nsys_args", NSYS_FLAGS)
        return {"binary": "nsys", "flags": wl + ["--profiler", "cuda-range"], "summary": "stats_cuda_gpu_kern_sum.csv",
                "wrap": lambda exe, out: [exe, "profile", "--output", str(out / "trace"), "--force-overwrite", "true",
                                          *flags, "--stats", "false"],
                "post": lambda exe, out: ([exe, "stats", "--report", NSYS_REPORTS, "--format", "csv", "--output",
                                           str(out / "stats"), str(out / "trace.nsys-rep")], None),
                "extra": lambda exe: [], "bundle_only": False}
    if kind == "kernel":
        rx = kernel_regex(args["kernel_regex"]) if "kernel_regex" in args else None
        skip, count = _int(args, "launch_skip", 0, 0), _int(args, "launch_count", 8, 1)
        kset = args.get("set", "basic")
        if not isinstance(kset, str) or not kset:
            raise ValueError("set must be a non-empty string")
        flags = _flags(args, "ncu_args")
        return {"binary": "ncu", "flags": wl + ["--profiler", "cuda-range"], "summary": "metrics.csv",
                "wrap": lambda exe, out: [exe, "--export", str(out / "kernel"), "--force-overwrite",
                                          *(["--kernel-name", f"regex:{rx}"] if rx else []), "--launch-skip", str(skip),
                                          "--launch-count", str(count), "--set", kset, "--profile-from-start", "off",
                                          "--target-processes", "all", *flags],
                "post": lambda exe, out: ([exe, "--import", str(out / "kernel.ncu-rep"), "--csv", "--page", "raw"],
                                          out / "metrics.csv"),
                "extra": lambda exe: [], "bundle_only": False}
    if kind == "hostprof":
        seconds, rate = _int(args, "seconds", 20, 1), _int(args, "rate", 100, 1)
        more = [w for f in _flags(args, "pyspy_args") for w in ("--pyspy-arg", f)]
        return {"binary": "py-spy", "flags": wl, "summary": "bundle/*/pyspy-dump.txt", "wrap": lambda exe, out: [],
                "post": None,
                "extra": lambda exe: ["--profiler", "pyspy", "--seconds", str(seconds), "--rate", str(rate),
                                      "--pyspy-bin", exe, *more], "bundle_only": False}
    raise ValueError(f"unknown instrument {kind!r}")


class ProfTools:
    """Mixin for lab.tools.Toolbox: needs self._audited, self._pristine, self._record, self._keep, self._work,
    self.worker."""

    def profile(self, args: dict) -> str:
        return self._instrumented("profile", args)

    def trace(self, args: dict) -> str:
        return self._instrumented("trace", args)

    def kernel(self, args: dict) -> str:
        return self._instrumented("kernel", args)

    def hostprof(self, args: dict) -> str:
        return self._instrumented("hostprof", args)

    def _instrumented(self, tool: str, args: dict) -> str:
        """A profiler tool call: arguments checked before anything is snapshotted, then one run, recorded as `profile`."""
        instrument(tool, args)
        snap = self._audited(tool, args)
        tree = self._pristine()
        why = profiles.refusal(tree, target.load(), args.get("profile"))
        result = {"verdict": "refused", "reason": why} if why else self.run_instrument(tool, tree, args)
        head = result.pop("head", "")
        self._record("profile", tool, args, result, snap)
        killed = f"\nkilled in your workbench (held the GPU): {json.dumps(result['workbench_killed'])}" \
            if result.get("workbench_killed") else ""
        if result["verdict"] in ("refused", "contaminated"):
            return f"{tool} {result['verdict']}: {result['reason']}{killed}"
        if result["verdict"] == "failed":
            return f"{tool} failed ({result['returncode']}):\n{result['output']}{killed}"
        if not head:                             # torch's bundle has no one summary file: its file names
            head = "\n".join(f"  {p.name}" for p in sorted((self.s.workspace.path / result["workspace_copy"]).iterdir()))
        return f"{tool} output at {result['workspace_copy']} (in your workspace; left out of your change)\n{head}{killed}"

    def run_instrument(self, tool: str, tree: Path, args: dict, timeout_s: int | None = None) -> dict:
        """One run of `tool`'s instrument on `tree` on the worker, as the result its record keeps: `verdict` ok |
        failed | refused | contaminated (`reason`, `gpu_processes`, as a bench's), the output as a blob and a
        workspace copy, `calibrated`, and `head` (the summary returned to the agent, not recorded)."""
        out = self._work(tool)
        t0 = time.monotonic()
        job = {"kind": tool, "args": args, "workbench": getattr(self.s, "workbench", None),
               **({"timeout_s": timeout_s} if timeout_s else {})}
        try:
            try:
                r = self.worker.call("profile", {"tree": tree}, job, out)
            except (serve.GpuBusy, serve.Contaminated) as e:
                return {**_failed(e), "seconds": time.monotonic() - t0}
            if r.get("refused"):
                return {"verdict": "refused", "reason": r["refused"], "seconds": time.monotonic() - t0}
            files = out / "files"
            blob, visible = self._keep(files) if files.is_dir() and any(p.is_file() for p in files.rglob("*")) else ("", "")
        finally:
            shutil.rmtree(out, ignore_errors=True)
        ok = r["returncode"] == 0 and bool(blob)
        return {"verdict": "ok" if ok else "failed", "returncode": r["returncode"], "bundle": blob,
                "workspace_copy": visible, "instrument": r.get("instrument"), "argv": r.get("argv"),
                "seconds": time.monotonic() - t0, "output": r["output"], "gpu": r.get("gpu"),
                "calibrated": self._calibrated(tool, args, blob, r.get("gpu")) if blob else None,
                "head": r.get("head", ""), **_killed(r)}

    def _calibrated(self, tool: str, args: dict, blob: str, gpu: dict | None) -> dict | None:
        """The instrument's numbers in context: torch's kernel time with each layer's facts and each kernel task's
        share, ncu's metrics against the GPU's peaks, nsys's kernel and API time against the window, py-spy's samples
        per thread and function; never costs the run it describes."""
        from lab.tools import task_shares
        path = self.s.ledger_root / blob
        try:
            if tool == "profile":
                cal = calibrate.calibrate_kernels(path, gpu)
                if cal is not None:
                    cal["layers"] = layers.from_bundle(path, target.load(), gpu)
                    cal["kernel_tasks"] = task_shares(path)
                return cal
            if tool == "kernel":
                return calibrate.calibrate_ncu(path / "metrics.csv", gpu)
            if tool == "trace":
                return calibrate.calibrate_nsys(path, any("cudaProfilerApi" in f for f in _flags(args, "nsys_args", NSYS_FLAGS)))
            return calibrate.calibrate_pyspy(path)
        except Exception as e:                   # noqa: BLE001
            return {"error": f"calibration failed: {type(e).__name__}: {e}"}

    def prof_specs(self, gpu) -> list[ToolSpec]:
        """`gpu(time)` renders the cost fact for a description."""
        has = bool(target.load().profiles)
        extra = {"profile": {"type": "string"}} if has else {}
        note = PROFILE_ARG if has else ""
        return [
            ToolSpec("profile", "Run the engine in process under the profiler, jailed, on `requests` sampled from a seen "
                     "corpus class (`corpus_class`, default steady_interactive; `seed`) or on synthetic token ids "
                     "(`synthetic`, `prompt_len`); `max_tokens` default 32. Stores the raw bundle (event timeline, chrome "
                     "trace, op and kernel tables, memory, GPU samples, provenance) in the ledger and copies it into your "
                     "workspace under lab/runs/, left out of your change. `capture`: \"module:function\" entry points whose every "
                     "call's argument shapes (and one whole call per shape) go into the bundle's captures/; it "
                     "synchronizes, so that bundle's times are not representative. Returns the bundle path and its file names."
                     + note + gpu("1-7 min measured at 16 requests x 64 tokens on an H200 (7 min when the weights were still loading from disk)"),
                     _schema(capture=FLAGS, **extra), self.profile),
            ToolSpec("trace", "Run the profile workload of your current workspace under Nsight Systems (nsys). "
                     f"`nsys_args`: the `nsys profile` flags, default {' '.join(NSYS_FLAGS)} (the measured window "
                     "only); output and overwrite are set by the tool. Stores the .nsys-rep and `nsys stats` CSV tables "
                     "(GPU kernels, CUDA API, memory copies, NVTX) in the ledger; returns the head of the kernel table. The "
                     "record's `calibrated`: kernel time, launches, memory-copy time, the top kernels and CUDA API calls "
                     "by time, and kernel time's share of the profiled window. "
                     "Refused where nsys is not installed." + note + gpu("about 1 min measured at 16 requests x 64 tokens on an H200 (engine start, workload, report export)"),
                     _schema(nsys_args=FLAGS, **extra), self.trace),
            ToolSpec("kernel", "Run the profile workload under Nsight Compute (ncu) on the kernels whose name "
                     "matches `kernel_regex` (every kernel if omitted): skips `launch_skip` matching launches, then "
                     "profiles `launch_count` with section set `set` (any `ncu --list-sets` name); `ncu_args` are "
                     "appended as further ncu flags. Stores the .ncu-rep and its raw metrics CSV in the ledger; returns "
                     "the head of the CSV. Refused where ncu is not installed."
                     + note + gpu("about 1.5 min measured at 4-8 launches on an H200; each profiled launch is replayed once per metric pass, so it grows with launch_count and set=full"),
                     _schema(**extra, kernel_regex={"type": "string"},
                             launch_skip={"type": "integer", "minimum": 0, "default": 0},
                             launch_count={"type": "integer", "minimum": 1, "default": 8},
                             set={"type": "string", "default": "basic"}, ncu_args=FLAGS), self.kernel),
            ToolSpec("hostprof", "Sample the engine process with py-spy for `seconds` while the profile workload "
                     "repeats (scheduler and backend in process, warmup excluded). Stores a speedscope profile and one "
                     "stack dump taken halfway in the ledger; returns the head of the dump. The record's `calibrated`: "
                     "samples per thread and the functions with the most samples on top of the stack (`self`) and "
                     "anywhere in it (`total`), as shares of all samples. `pyspy_args` are appended "
                     "to `py-spy record`. Refused where py-spy is "
                     "not installed." + note + gpu("about 35 s plus `seconds`, measured on an H200"),
                     _schema(**extra, seconds={"type": "integer", "minimum": 1, "default": 20},
                             rate={"type": "integer", "minimum": 1, "default": 100}, pyspy_args=FLAGS), self.hostprof),
        ]
