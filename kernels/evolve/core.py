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


def judge(task: Task, program: Path, seen: list, heldout: list, weights: dict | None, device: str,
          tolerances: dict | None, stage: int, iters: int = 30) -> dict:
    """stage 1: correctness on the two most frequent seen shapes, one seed. stage 2: the full check on every seen shape
    (held-out shapes only when given, which selection never does), then timing against the baseline."""
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
    if not r["passed"] or stage == 1:
        return {"metrics": {"combined_score": 0.0 if not r["passed"] else 0.5, "correct": float(r["passed"]),
                            "stage": stage},
                "artifacts": {"gates": json.dumps(gates, indent=1)}}
    base_fn = task.baseline()
    base = task.bind(base_fn) if base_fn else ref
    hw, dtype, rows, logs = _peaks(device), task.dtypes[0], [], 0.0
    total = sum((weights or {}).get(json.dumps(list(s)), 1) for s in seen)
    for s in seen:
        c = time_kernel(cand, task.make_inputs, s, dtype, device, iters=iters)
        b = time_kernel(base, task.make_inputs, s, dtype, device, iters=iters)
        speed = b["median_ms"] / c["median_ms"] if c["median_ms"] else 0.0
        w = (weights or {}).get(json.dumps(list(s)), 1) / total
        logs += w * math.log(max(speed, 1e-9))
        rows.append({"shape": list(s), "candidate_ms": c["median_ms"], "candidate_iqr_ms": c["iqr_ms"],
                     "baseline_ms": b["median_ms"], "baseline_iqr_ms": b["iqr_ms"], "speedup": speed, "weight": w,
                     "pct_of_roofline": _pct_of_roofline(task, s, dtype, c["median_ms"], hw)})
    score = math.exp(logs)
    return {"metrics": {"combined_score": score, "correct": 1.0, "speedup": score, "stage": stage,
                        "min_pct_of_roofline": min((x["pct_of_roofline"] for x in rows if x["pct_of_roofline"]),
                                                   default=0.0)},
            "artifacts": {"timings": json.dumps(rows, indent=1), "gates": json.dumps(gates, indent=1),
                          "baseline": task.entry or "reference"}}


def main() -> int:
    """python -m kernels.evolve.core CONFIG.json PROGRAM STAGE  ->  one JSON object on stdout."""
    cfg, program, stage = json.loads(Path(sys.argv[1]).read_text()), Path(sys.argv[2]), int(sys.argv[3])
    task = load(cfg["task"])
    tol = {getattr(torch, k): tuple(v) for k, v in (cfg.get("tolerances") or {}).items()}
    out = judge(task, program, [tuple(s) for s in cfg["seen"]], [tuple(s) for s in cfg.get("heldout", [])],
                cfg.get("weights"), cfg.get("device", "cpu"), tol or None, stage, cfg.get("iters", 30))
    sys.stdout.write("\n" + json.dumps(out) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
