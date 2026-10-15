"""One candidate kernel, judged: load its program file, check it against the task's reference, time it against the
baseline, and return metrics plus raw facts for the next prompt. Selection sees only the seen shapes; the held-out
shapes are judged by `gate` alone. Runs in its own process (see evaluator.py), so a crash, a hung kernel or a patched
timer cannot outlive one evaluation."""
from __future__ import annotations

import importlib.util
import json
import math
import sys
import traceback
from pathlib import Path

import torch

from kernels import check_kernel, time_kernel
from kernels.tasks import Task, load


# What the published exploits used (Sakana's reused buffers, CUDA-L1's extra streams, METR's timer overwrite, KernelBot's
# patched elapsed_time and background launches): a candidate's source may not reach for any of it.
FORBIDDEN = {
    r"\bgc\b": "the garbage collector (finding the reference's tensors)",
    r"\binspect\b|\bsys\._getframe|\bf_back\b|\bf_locals\b": "frame introspection",
    r"\bctypes\b|\bcffi\b": "raw memory access",
    r"\bthreading\b|\bmultiprocessing\b|\bconcurrent\b|\basyncio\b": "background work outside the timed call",
    r"torch\.cuda\.(Stream|stream|Event|set_stream|current_stream)\b|\.elapsed_time\b|\bcuda\.Event\b":
        "streams or timer events",
    r"\btime\.(perf_counter|time|monotonic|sleep)\b|\bimport time\b|\bfrom time\b": "the clock",
    r"\bsys\.modules\b|\bbuiltins\b|__builtins__|\bimportlib\b|\bsetattr\s*\(\s*(torch|kernels|time)\b":
        "patching modules",
    r"\bkernels\.(check|timing|evolve)\b|\bfrom kernels\b|\bimport kernels\b": "the evaluator itself",
    r"\bopen\s*\(|\bos\.(system|popen|fork|exec\w*|spawn\w*|kill|remove|unlink|open|write|dup2?)\b|\bsubprocess\b"
    r"|\bsocket\b|\burllib\b|\brequests\b": "files, processes or the network",
}


def screen(source: str) -> list[str]:
    """Reasons a program is refused before it runs (static: a pattern anywhere in the source)."""
    import re
    return [f"uses {why}: `{m.group(0)}`" for pat, why in FORBIDDEN.items() if (m := re.search(pat, source))]


def load_program(path: Path):
    spec = importlib.util.spec_from_file_location(f"candidate_{abs(hash(str(path)))}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not callable(getattr(mod, "kernel", None)):
        raise AttributeError("the program defines no callable `kernel`")
    return mod.kernel


def _peaks(device: str):
    if torch.device(device).type != "cuda":
        return None
    from validity import peaks
    return peaks.load(torch.cuda.get_device_name(torch.device(device)))


def _pct_of_roofline(task: Task, shape, dtype, ms: float, hw) -> float | None:
    if hw is None or not ms:
        return None
    flops, nbytes = task.cost(shape, dtype)
    bound_s = max(flops / hw.bf16_flops, nbytes / hw.bw_bytes_s)
    return bound_s / (ms / 1e3) * 100


# Value distributions beyond the task's own (KernelBench-Verified's hidden suite: x3, x0.01, negated): a kernel tuned to
# one input distribution, or one that special-cases it, fails here.
SCALES = (3.0, 0.01, -1.0)
PHYSICS_MARGIN_PCT = 105          # measured faster than the bound (beyond timer error) = the kernel skipped work
MEMORY_FACTOR, MEMORY_SLACK = 1.5, 64 * 2**20


def _scaled(make_inputs, k: float):
    def make(shape, dtype, device, g):
        return tuple(t * k if t.is_floating_point() else t for t in make_inputs(shape, dtype, device, g))
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


def _shown(c: dict, b: dict) -> bool:
    """A difference counts when the medians differ by more than the two runs' half-IQRs: the timer's own noise."""
    return abs(b["median_ms"] - c["median_ms"]) > (c["iqr_ms"] + b["iqr_ms"]) / 2


def judge(task: Task, program: Path, seen: list, heldout: list, weights: dict | None, device: str,
          tolerances: dict | None, stage: int, iters: int = 30) -> dict:
    """stage 1: correctness on the two most frequent seen shapes, one seed. stage 2: the full check on every seen shape
    (held-out shapes only when given, which selection never does), then timing against the baseline."""
    refused = screen(Path(program).read_text())
    if refused:
        return {"metrics": {"combined_score": 0.0, "correct": 0.0, "stage": stage},
                "artifacts": {"error": "refused before running: " + "; ".join(refused)}}
    try:
        cand = task.bind(load_program(program))
    except Exception as e:                       # noqa: BLE001  a program that does not load scores 0
        return {"metrics": {"combined_score": 0.0, "correct": 0.0, "stage": stage},
                "artifacts": {"error": f"load failed: {type(e).__name__}: {e}"}}
    ref = task.reference
    shapes = seen[:2] if stage == 1 else seen
    try:
        r = check_kernel(ref, cand, task.make_inputs, shapes, heldout if stage > 1 else [], dtypes=task.dtypes,
                         device=device, seeds=(0,) if stage == 1 else (0, 1, 2), tolerances=tolerances,
                         edge_shapes=task.edge_shapes(shapes[0]) if task.edge_shapes else None,
                         repeats=1 if stage == 1 else 3, layouts=task.layouts)
    except Exception:                            # noqa: BLE001
        return {"metrics": {"combined_score": 0.0, "correct": 0.0, "stage": stage},
                "artifacts": {"error": "checker raised:\n" + traceback.format_exc()[-3000:]}}
    gates = {k: {"passed": g["passed"], "reason": g["reason"]} for k, g in r["gates"].items()}
    if r["passed"] and stage > 1:                 # other value distributions, on the two most frequent seen shapes
        for k in SCALES:
            v = check_kernel(ref, cand, _scaled(task.make_inputs, k), seen[:2], [], dtypes=task.dtypes, device=device,
                             seeds=(7,), tolerances=tolerances, edge_shapes=[], repeats=1, layouts=())
            g = v["gates"]["correctness_seen"]
            gates[f"values_x{k:g}"] = {"passed": g["passed"], "reason": g["reason"]}
            r["passed"] = r["passed"] and g["passed"]
    if not r["passed"] or stage == 1:
        return {"metrics": {"combined_score": 0.0 if not r["passed"] else 0.5, "correct": float(r["passed"]),
                            "stage": stage},
                "artifacts": {"gates": json.dumps(gates, indent=1)}}
    base_fn = task.baseline()
    base = task.bind(base_fn) if base_fn else ref
    hw, dtype, rows, logs, raw = _peaks(device), task.dtypes[0], [], 0.0, 0.0
    inputs0 = task.make_inputs(seen[0], dtype, device, torch.Generator().manual_seed(0))
    pc, pb = _peak_bytes(cand, inputs0, device), _peak_bytes(base, inputs0, device)
    if pc is not None and pc > MEMORY_FACTOR * pb + MEMORY_SLACK:
        gates["peak_memory"] = {"passed": False, "reason": f"peak {pc / 2**20:.0f} MiB vs baseline {pb / 2**20:.0f} MiB "
                                f"(> {MEMORY_FACTOR}x + {MEMORY_SLACK // 2**20} MiB)"}
        return {"metrics": {"combined_score": 0.0, "correct": 1.0, "stage": stage},
                "artifacts": {"gates": json.dumps(gates, indent=1)}}
    if pc is not None:
        gates["peak_memory"] = {"passed": True, "reason": f"peak {pc / 2**20:.0f} MiB vs baseline {pb / 2**20:.0f} MiB"}
    total = sum((weights or {}).get(json.dumps(list(s)), 1) for s in seen)
    for s in seen:
        c = time_kernel(cand, task.make_inputs, s, dtype, device, iters=iters)
        b = time_kernel(base, task.make_inputs, s, dtype, device, iters=iters)
        speed = b["median_ms"] / c["median_ms"] if c["median_ms"] else 0.0
        shown = _shown(c, b)
        w = (weights or {}).get(json.dumps(list(s)), 1) / total
        logs += w * math.log(max(speed if shown else 1.0, 1e-9))      # inside the noise: no change claimed
        raw += w * math.log(max(speed, 1e-9))
        rows.append({"shape": list(s), "candidate_ms": c["median_ms"], "candidate_iqr_ms": c["iqr_ms"],
                     "baseline_ms": b["median_ms"], "baseline_iqr_ms": b["iqr_ms"], "speedup": speed, "weight": w,
                     "beyond_noise": shown, "clock": c.get("clock"),
                     "pct_of_roofline": _pct_of_roofline(task, s, dtype, c["median_ms"], hw)})
    over = [x for x in rows if (x["pct_of_roofline"] or 0) > PHYSICS_MARGIN_PCT]
    if over:
        gates["physics"] = {"passed": False, "reason": f"{len(over)} shapes measured faster than the roofline bound "
                            f"(max {max(x['pct_of_roofline'] for x in over):.0f}%): work was skipped or not timed"}
        return {"metrics": {"combined_score": 0.0, "correct": 1.0, "stage": stage},
                "artifacts": {"timings": json.dumps(rows, indent=1), "gates": json.dumps(gates, indent=1)}}
    score = math.exp(logs)
    return {"metrics": {"combined_score": score, "correct": 1.0, "speedup": score, "raw_speedup": math.exp(raw),
                        "stage": stage,
                        "min_pct_of_roofline": min((x["pct_of_roofline"] for x in rows if x["pct_of_roofline"]),
                                                   default=0.0)},
            "artifacts": {"timings": json.dumps(rows, indent=1), "gates": json.dumps(gates, indent=1),
                          "baseline": task.entry or "reference"}}


def main() -> int:
    """python -m kernels.evolve.core CONFIG.json PROGRAM|- STAGE  ->  one JSON object on stdout ("-": the program's
    source on stdin, as a remote evaluation sends it)."""
    cfg, stage = json.loads(Path(sys.argv[1]).read_text()), int(sys.argv[3])
    if sys.argv[2] == "-":
        import tempfile
        fd, name = tempfile.mkstemp(suffix=".py", prefix="candidate-")
        with open(fd, "w") as f:
            f.write(sys.stdin.read())
        program = Path(name)
    else:
        program = Path(sys.argv[2])
    task = load(cfg["task"])
    tol = {getattr(torch, k): tuple(v) for k, v in (cfg.get("tolerances") or {}).items()}
    out = judge(task, program, [tuple(s) for s in cfg["seen"]], [tuple(s) for s in cfg.get("heldout", [])],
                cfg.get("weights"), cfg.get("device", "cpu"), tol or None, stage, cfg.get("iters", 30))
    sys.stdout.write("\n" + json.dumps(out) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
