"""One candidate kernel, judged in its own process: its program loaded, checked against the task's reference and timed
against the baseline; returns metrics and raw facts. Selection sees only the seen shapes; `gate` judges the held-out."""
from __future__ import annotations

import importlib.util
import json
import math
import sys
import traceback
from pathlib import Path

import torch

from lab.kernels.eval import check_kernel, time_pair
from lab.kernels.eval.check import TOLERANCES, _compare, baseline_envelope    # bound here, before any candidate loads
from lab.kernels.eval.freeze import FREEZE_STAGE
from lab.kernels.tasks import Task, from_config


# What a candidate may not do: stated in the setting, read for by the gaming detector.
NOT_ALLOWED = ("reading the clock, timers or profilers", "CUDA streams, events or graphs of its own",
               "threads, processes or other work outside the timed call", "files, the network or raw memory access",
               "patching or writing into anything it imports", "importing the evaluator or the task's reference",
               "process-wide numerics switches", "frame or module introspection")


def _watched(task: Task) -> dict:
    """What the checker and the timer stand on: the libraries' public functions, the evaluator's and the task's own
    modules (all of their functions), and the tensor and event classes (any attribute, added ones too)."""
    import statistics
    import time
    import lab.kernels.eval.check
    import lab.kernels.tasks
    import lab.kernels.eval.timing
    mods = [torch, torch.cuda, torch.nn.functional, time, math, statistics, lab.kernels.eval.check, lab.kernels.eval.timing,
            lab.kernels.tasks, sys.modules[__name__], sys.modules[task.reference.__module__]]
    if task.error_budget:                        # the baseline sets the tolerance: its module and the stations' too
        base = task.baseline()
        names = sorted({*task.watch, getattr(base, "__module__", "")} - {""})
        mods += [importlib.import_module(n) for n in names]      # imported before the snapshot, never skipped
    out = {}
    for m in mods:
        own = m.__name__.startswith(("lab.kernels.eval", "lab.kernels"))
        for k, v in vars(m).items():
            if (callable(v) or isinstance(v, type(math))) and not k.startswith("__") and (own or not k.startswith("_")):
                out[f"{m.__name__}.{k}"] = id(v)
    for c in (torch.Tensor, torch.cuda.Event, torch.cuda.CUDAGraph):
        out.update({f"{c.__module__}.{c.__qualname__}.{k}": id(v) for k, v in vars(c).items()})
    out.update({f"sys.modules[{m.__name__}]": id(sys.modules.get(m.__name__)) for m in mods})   # a module swapped
    return out


def _patched(before: dict, task: Task) -> list[str]:
    """Names in `before` that now point elsewhere or are gone, and attributes added to the watched classes."""
    now = _watched(task)
    cls = (f"{c.__module__}.{c.__qualname__}." for c in (torch.Tensor, torch.cuda.Event, torch.cuda.CUDAGraph))
    prefixes = tuple(cls)
    return sorted([k for k, v in before.items() if now.get(k) != v] +
                  [k for k in now if k not in before and k.startswith(prefixes)])


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(f"candidate_{abs(hash(str(path)))}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not callable(getattr(mod, "kernel", None)):
        raise AttributeError("the program defines no callable `kernel`")
    return mod


def load_program(path: Path):
    return load_module(path).kernel


def _peaks(device: str):
    if torch.device(device).type != "cuda":
        return None
    from lab.infra.context import peaks
    return peaks.load(torch.cuda.get_device_name(torch.device(device)))


def _roofline(task: Task, shape, dtype, ms: float, hw, inputs: list) -> dict:
    """For the inputs the timer rotated through (averaged): % of the roofline bound, which term binds (memory or
    compute) and the achieved fraction of each peak."""
    if hw is None or not ms:
        return {"pct_of_roofline": None}
    costs = [task.cost(shape, dtype, x) for x in inputs]
    flops, nbytes = (sum(c[i] for c in costs) / len(costs) for i in (0, 1))
    t_mem, t_flop, t = nbytes / hw.bw_bytes_s, flops / hw.bf16_flops, ms / 1e3
    return {"pct_of_roofline": max(t_mem, t_flop) / t * 100, "bound": "memory" if t_mem >= t_flop else "compute",
            "pct_of_peak_bandwidth": t_mem / t * 100, "pct_of_peak_flops": t_flop / t * 100}


# Outputs of the plain check cases and the timed calls, kept for lab/kernels/eval/verify.py, which recomputes the
# reference and compares outside this process (a candidate can reach anything in here). Copied and saved through
# functions bound before any candidate loads.
RECORDS: list = []
_detach, _cpu, _clone, _save = torch.Tensor.detach, torch.Tensor.cpu, torch.Tensor.clone, torch.save


def _keep(kind: str, shape, dtype: str, seed: int, outputs, scale: float | None = None) -> None:
    outs = outputs if isinstance(outputs, (list, tuple)) else [outputs]
    RECORDS.append({"kind": kind, "shape": list(shape), "dtype": dtype, "seed": seed, "scale": scale,
                    "outputs": [_clone(_cpu(_detach(o))) for o in outs if isinstance(o, torch.Tensor)]})


# Value distributions beyond the task's own (KernelBench-Verified's hidden suite: x3, x0.01, negated): a kernel tuned to
# one input distribution, or one that special-cases it, fails here.
SCALES = (3.0, 0.01, -1.0)
PHYSICS_MARGIN_PCT = 105          # measured faster than the bound (beyond timer error) = the kernel skipped work
MEMORY_FACTOR, MEMORY_SLACK = 1.5, 64 * 2**20


def _scaled(make_inputs, k: float):
    """Inputs with every floating tensor times k; 1-byte floats (fp8 weights) are kept, their scales scaled instead."""
    def make(shape, dtype, device, g):
        return tuple(t * k if t.is_floating_point() and t.element_size() > 1 else t
                     for t in make_inputs(shape, dtype, device, g))
    return make


def _peak_bytes(fn, inputs, device) -> int | None:
    if torch.device(device).type != "cuda":
        return None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fn(*inputs)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() - base


def _error(result: dict) -> dict:
    """Max abs and rel error against the reference over the seen-shape cases, per dtype."""
    ev = result["gates"]["correctness_seen"]["evidence"]
    return {"max_abs": ev["max_abs"], "max_rel": ev["max_rel"], "cases": ev["cases"]}


def _shown(ratio: dict) -> bool:
    """A difference counts when the middle half of the paired baseline/candidate ratios lies on one side of 1."""
    return ratio["p25"] > 1 or ratio["p75"] < 1


def judge(task: Task, program: Path, seen: list, heldout: list, weights: dict | None, device: str,
          tolerances: dict | None, stage: int, iters: int = 30, full: bool = False, seeds: int = 3,
          timing: bool = True) -> dict:
    """Stage 1: correctness on the two most frequent seen shapes. Stage 2: correctness on every seen shape and timing;
    with `full` or held-out shapes the full check, and without `timing` nothing after it. FREEZE_STAGE records each
    autotuner's choice per key and scores nothing. Anything the evaluator watches left patched scores 0."""
    task.baseline()                              # the engine's import-time effects happen before the snapshot
    before, patched = _watched(task), []

    def checkpoint() -> bool:
        """Record anything patched since the snapshot; True once anything has been."""
        patched.extend(n for n in _patched(before, task) if n not in patched)
        return bool(patched)
    out = (_freeze(task, program, seen, device, checkpoint) if stage == FREEZE_STAGE
           else _judge(task, program, seen, heldout, weights, device, tolerances, stage, iters, checkpoint, full, seeds,
                       timing))
    if checkpoint():
        return {"metrics": {"combined_score": 0.0, "correct": 0.0, "stage": stage},
                "artifacts": {"error": "patched what the evaluator uses: " + ", ".join(patched[:20])}}
    return out


def _judge(task: Task, program: Path, seen: list, heldout: list, weights: dict | None, device: str,
           tolerances: dict | None, stage: int, iters: int, checkpoint, full: bool = False, seeds: int = 3,
           timing: bool = True) -> dict:
    """The gates and timing; returns {} as soon as `checkpoint` finds a patch (judge reports it)."""
    bfn = task.baseline() if task.error_budget else None                 # bound before the candidate loads
    budget = {"baseline": task.bind(bfn), "budget": task.error_budget} if bfn else {}
    try:
        timed = task.bind(load_program(program))
    except Exception as e:                       # noqa: BLE001  a program that does not load scores 0
        return {"metrics": {"combined_score": 0.0, "correct": 0.0, "stage": stage},
                "artifacts": {"error": f"load failed: {type(e).__name__}: {e}"}}
    if checkpoint():
        return {}

    def cand(*xs):                               # checked after every call the checker makes, before it compares
        out = timed(*xs)
        if checkpoint():
            raise RuntimeError("the candidate patched what the evaluator uses")
        return out
    ref = task.reference
    shapes = seen[:2] if stage == 1 else seen
    full = stage > 1 and (full or bool(heldout))     # the gate: every check; selection: correctness and timing only
    try:
        keep = lambda case, out: _keep("check", case["shape"], case["dtype"], case["seed"], out)    # noqa: E731
        r = check_kernel(ref, cand, task.make_inputs, shapes, heldout if full else [], dtypes=task.dtypes,
                         device=device, seeds=tuple(range(seeds)) if full else (0,), tolerances=tolerances,
                         edge_shapes=(task.edge_shapes(shapes[0]) if task.edge_shapes else None) if full else [],
                         repeats=3 if full else 1, layouts=task.layouts if full else (), record=keep,
                         nonfinite=task.nonfinite, judged=task.judged, **budget)
    except Exception:                            # noqa: BLE001
        return {"metrics": {"combined_score": 0.0, "correct": 0.0, "stage": stage},
                "artifacts": {"error": "checker raised:\n" + traceback.format_exc()[-3000:]}}
    gates = {k: {"passed": g["passed"], "reason": g["reason"], "cases": (g.get("evidence") or {}).get("cases")}
             for k, g in r["gates"].items()}
    if r["passed"] and full:                      # other value distributions, on the two most frequent seen shapes
        for k in (SCALES if task.scales is None else task.scales):
            v = check_kernel(ref, cand, _scaled(task.make_inputs, k), seen[:2], [], dtypes=task.dtypes, device=device,
                             seeds=(7,), tolerances=tolerances, edge_shapes=[], repeats=1, layouts=(),
                             nonfinite=task.nonfinite, judged=task.judged, **budget,
                             record=lambda case, out, k=k: _keep("check", case["shape"], case["dtype"], case["seed"],
                                                                 out, k))
            g = v["gates"]["correctness_seen"]
            gates[f"values_x{k:g}"] = {"passed": g["passed"], "reason": g["reason"]}
            r["passed"] = r["passed"] and g["passed"]
    if not r["passed"] or stage == 1 or not timing:
        return {"metrics": {"combined_score": 0.5 if r["passed"] and stage == 1 else 0.0, "correct": float(r["passed"]),
                            "stage": stage},
                "artifacts": {"gates": json.dumps(gates, indent=1)}}
    base_fn = task.baseline()
    base = task.bind(base_fn) if base_fn else ref
    error = {"candidate": _error(r)}                # a fact, not a gate: lower precision inside a kernel is allowed
    if base_fn and full:
        error["baseline"] = _error(check_kernel(ref, base, task.make_inputs, seen, [], dtypes=task.dtypes,
                                                device=device, seeds=(0, 1, 2), tolerances=tolerances,
                                                edge_shapes=[], repeats=1, layouts=(), nonfinite=task.nonfinite))
    hw, dtype = _peaks(device), task.dtypes[0]
    inputs0 = task.make_inputs(seen[0], dtype, device, torch.Generator().manual_seed(0))
    pc, pb = (_peak_bytes(timed, inputs0, device), _peak_bytes(base, inputs0, device)) if full else (None, None)
    if checkpoint():
        return {}
    if pc is not None and pc > MEMORY_FACTOR * pb + MEMORY_SLACK:
        gates["peak_memory"] = {"passed": False, "reason": f"peak {pc / 2**20:.0f} MiB vs baseline {pb / 2**20:.0f} MiB "
                                f"(> {MEMORY_FACTOR}x + {MEMORY_SLACK // 2**20} MiB)"}
        return {"metrics": {"combined_score": 0.0, "correct": 1.0, "stage": stage},
                "artifacts": {"gates": json.dumps(gates, indent=1)}}
    if pc is not None:
        gates["peak_memory"] = {"passed": True, "reason": f"peak {pc / 2**20:.0f} MiB vs baseline {pb / 2**20:.0f} MiB"}
    try:
        rows, score, raw = _timed(task, timed, base, seen, weights, dtype, device, iters, hw, tolerances)
        held = (_timed(task, timed, base, heldout, weights, dtype, device, iters, hw, tolerances)   # gate only
                if heldout else None)
    except Exception as e:                       # noqa: BLE001  cannot be captured, or wrong output while timed
        gates["timing"] = {"passed": False, "reason": (f"{type(e).__name__}: {e}")[:2000] +
                           (" (timed as a CUDA graph replay: the kernel must capture and stay correct)"
                            if task.graphs else "")}
        return {"metrics": {"combined_score": 0.0, "correct": 1.0, "stage": stage},
                "artifacts": {"gates": json.dumps(gates, indent=1)}}
    over = [x for x in rows + (held[0] if held else []) if (x["pct_of_roofline"] or 0) > PHYSICS_MARGIN_PCT]
    if over:
        gates["physics"] = {"passed": False, "reason": f"{len(over)} shapes measured faster than the roofline bound "
                            f"(max {max(x['pct_of_roofline'] for x in over):.0f}%): work was skipped or not timed"}
        return {"metrics": {"combined_score": 0.0, "correct": 1.0, "stage": stage},
                "artifacts": {"timings": json.dumps(rows, indent=1), "gates": json.dumps(gates, indent=1)}}
    if hw is None:                               # a fact: the alarm above needs this device's measured peaks
        gates["physics"] = {"passed": True, "reason": "not checked: no measured peaks for this device"}
    metrics = {"combined_score": score, "correct": 1.0, "speedup": score, "raw_speedup": raw, "stage": stage,
               "min_pct_of_roofline": min((x["pct_of_roofline"] for x in rows if x["pct_of_roofline"]), default=0.0)}
    artifacts = {"timings": json.dumps(rows, indent=1), "gates": json.dumps(gates, indent=1),
                 "error_vs_reference": json.dumps(error, indent=1), "baseline": task.entry or "reference"}
    if held:                                     # a fact for the researcher; never part of the score
        metrics["heldout_speedup"], metrics["heldout_raw_speedup"] = held[1], held[2]
        artifacts["heldout_timings"] = json.dumps(held[0], indent=1)
    return {"metrics": metrics, "artifacts": artifacts}


def _freeze(task: Task, program: Path, seen: list, device: str, checkpoint) -> dict:
    """{artifacts: {frozen: {name: {keys, table}}}}: each autotuner the program binds at module level, run on every
    seen shape (each dtype, seed 0), with the config it chose for each key it saw."""
    from lab.kernels.eval import freeze
    try:
        mod = load_module(program)
    except Exception as e:                       # noqa: BLE001
        return {"metrics": {"stage": FREEZE_STAGE}, "artifacts": {"error": f"load failed: {type(e).__name__}: {e}"}}
    if checkpoint():
        return {}
    recorded = {n: freeze.record(t) for n, t in freeze.tuners(mod, Path(program).read_text()).items()}
    fn = task.bind(mod.kernel)
    try:
        for s in seen:
            for dt in task.dtypes:
                fn(*task.make_inputs(s, dt, device, torch.Generator().manual_seed(0)))
                if checkpoint():
                    return {}
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize()
        frozen = json.dumps(freeze.tables(recorded))
    except Exception as e:                       # noqa: BLE001  a key or config that is not a literal included
        return {"metrics": {"stage": FREEZE_STAGE},
                "artifacts": {"error": f"{type(e).__name__}: {e}\n" + traceback.format_exc()[-2000:]}}
    return {"metrics": {"stage": FREEZE_STAGE}, "artifacts": {"frozen": frozen}}


def _timed(task, cand, base, shapes, weights, dtype, device, iters, hw, tolerances=None) -> tuple[list, float, float]:
    """Per-shape paired timings against the baseline, and the call-weighted geometric-mean speedup with gains inside
    the noise counted as 1.0 (and without that rule). Raises if a timed call's output differs from the reference."""
    atol, rtol = {**TOLERANCES, **(tolerances or {})}[dtype]
    rows, logs, raw = [], 0.0, 0.0
    total = sum((weights or {}).get(json.dumps(list(s)), 1) for s in shapes)
    for s in shapes:
        p = time_pair(cand, base, task.make_inputs, s, dtype, device, iters=iters,      # unwrapped: no check in
                      graphs=task.graphs)                                              # the window
        for i, out in enumerate(p["outputs_a"]):
            _keep("time", s, str(dtype).removeprefix("torch."), i, out)     # time_pair's buffer i: seed i
        for inputs, out in zip(p["inputs"], p["outputs_a"]):
            ref = task.reference(*[t.clone() for t in inputs])
            b = baseline_envelope(base, inputs, ref) if task.error_budget else None
            d = _compare(ref, task.judged(out) if task.judged else out, atol, rtol, base=b, k=task.error_budget or 0.0)
            if not d["ok"]:
                raise ValueError(f"the timed calls' output differs from the reference at shape {list(s)}: "
                                 f"{d.get('error') or ('max_abs %.3g, max_rel %.3g' % (d['max_abs'], d['max_rel']))}")
        c, b, speed = p["a"], p["b"], p["ratio"]["median"]
        shown = _shown(p["ratio"])
        w = (weights or {}).get(json.dumps(list(s)), 1) / total
        logs += w * math.log(max(speed if shown else 1.0, 1e-9))
        raw += w * math.log(max(speed, 1e-9))
        rows.append({"shape": list(s), "candidate_ms": c["median_ms"], "candidate_iqr_ms": c["iqr_ms"],
                     "baseline_ms": b["median_ms"], "baseline_iqr_ms": b["iqr_ms"], "speedup": speed,
                     "speedup_p25": p["ratio"]["p25"], "speedup_p75": p["ratio"]["p75"], "weight": w,
                     "beyond_noise": shown, "clock": p["clock"],
                     **_roofline(task, s, dtype, c["median_ms"], hw, p["inputs"])})
    return rows, math.exp(logs), math.exp(raw)


def main() -> int:
    """python -m lab.kernels.eval.core CONFIG.json PROGRAM|- STAGE  ->  one JSON object on stdout ("-": the program's
    source on stdin, as a remote evaluation sends it)."""
    cfg, stage = json.loads(Path(sys.argv[1]).read_text()), int(sys.argv[3])
    key = None                                   # read and removed before the candidate loads: its result line's key
    if cfg.get("result_key"):
        key = Path(cfg["result_key"]).read_text()
        Path(cfg["result_key"]).unlink()
    if sys.argv[2] == "-":
        import tempfile
        fd, name = tempfile.mkstemp(suffix=".py", prefix="candidate-")
        with open(fd, "w") as f:
            f.write(sys.stdin.read())
        program = Path(name)
    else:
        program = Path(sys.argv[2])
    task = from_config(cfg)      # captured int args (real sequence lengths) where given
    tol = {getattr(torch, k): tuple(v) for k, v in (cfg.get("tolerances") or {}).items()}
    out = judge(task, program, [tuple(s) for s in cfg["seen"]], [tuple(s) for s in cfg.get("heldout", [])],
                cfg.get("weights"), cfg.get("device", "cpu"), tol or None, stage, cfg.get("iters", 30),
                bool(cfg.get("full")), cfg.get("seeds", 3), not cfg.get("check_only"))
    if cfg.get("record"):
        _save({"records": RECORDS}, cfg["record"])
    sys.stdout.write("\n" + json.dumps({**out, "result_key": key} if key else out) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
