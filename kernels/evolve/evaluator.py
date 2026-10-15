"""OpenEvolve's evaluator: a two-stage cascade, each stage a fresh process running kernels.evolve.core, so no
candidate shares a process with the timer, the reference or another candidate. Reads its config from
$KERNEL_EVAL_CONFIG (written by kernels.evolve.run). Facts only go back to the model: gate reasons, errors, timings."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from openevolve.evaluation_result import EvaluationResult

TIMEOUT_S = float(os.environ.get("KERNEL_EVAL_TIMEOUT_S", "600"))
ROOT = Path(__file__).resolve().parents[2]


def _run(program_path: str, stage: int) -> EvaluationResult:
    cfg = os.environ["KERNEL_EVAL_CONFIG"]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(ROOT), os.environ.get("PYTHONPATH")]))}
    try:
        p = subprocess.run([sys.executable, "-m", "kernels.evolve.core", cfg, program_path, str(stage)],
                           capture_output=True, text=True, timeout=TIMEOUT_S, env=env, cwd=ROOT)
    except subprocess.TimeoutExpired:
        return EvaluationResult({"combined_score": 0.0, "correct": 0.0, "stage": stage},
                                {"error": f"evaluation exceeded {TIMEOUT_S:.0f} s"})
    lines = [x for x in p.stdout.splitlines() if x.startswith("{")]
    if p.returncode != 0 or not lines:
        return EvaluationResult({"combined_score": 0.0, "correct": 0.0, "stage": stage},
                                {"error": f"evaluation process exited {p.returncode}:\n{(p.stderr or p.stdout)[-3000:]}"})
    d = json.loads(lines[-1])
    return EvaluationResult(d["metrics"], d.get("artifacts") or {})


def evaluate_stage1(program_path: str) -> EvaluationResult:
    return _run(program_path, 1)


def evaluate_stage2(program_path: str) -> EvaluationResult:
    return _run(program_path, 2)


def evaluate(program_path: str) -> EvaluationResult:
    r = _run(program_path, 1)
    return r if r.metrics.get("correct", 0) < 1 else _run(program_path, 2)
