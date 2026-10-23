"""OpenEvolve's evaluator: a two-stage cascade, each stage a fresh jailed process (kernels.evolve.jailed running
kernels.evolve.core), so no candidate outlives its evaluation, shares a process with another, or sees this repo. Reads its config from
$KERNEL_EVAL_CONFIG (written by kernels.evolve.run). Facts only go back to the model: gate reasons, errors, timings.

Remote: with $KERNEL_EVAL_SSH (a JSON argv prefix, e.g. ["ssh", "-i", KEY, "lab@host"]) and $KERNEL_EVAL_REMOTE (a
shell prefix run there, e.g. "cd ~/BlameGraph && PATH=... "), each stage runs on the GPU host with the config and the
program's source on stdin, so nothing is copied there first and the model's key stays on this machine. No API key is
ever in a candidate's environment."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

TIMEOUT_S = float(os.environ.get("KERNEL_EVAL_TIMEOUT_S", "600"))
ROOT = Path(__file__).resolve().parents[2]


def seconds(log: Path) -> float:
    """Wall seconds spent judging (the GPU host busy or reserved), summed from the log `judge_stage` appends to."""
    return round(sum(float(x) for x in Path(log).read_text().split()), 1) if Path(log).exists() else 0.0


def judge_stage(program_path: str, stage: int) -> dict:
    """{metrics, artifacts} for one stage of one candidate, judged jailed (here or on the GPU host): no OpenEvolve
    types, so kernels.search uses it too. Its wall time goes to $KERNELS_EVAL_LOG when set."""
    t0 = time.monotonic()
    try:
        return _judge_stage(program_path, stage)
    finally:
        if os.environ.get("KERNELS_EVAL_LOG"):
            with open(os.environ["KERNELS_EVAL_LOG"], "a") as f:
                f.write(f"{time.monotonic() - t0:.2f}\n")


def _judge_stage(program_path: str, stage: int) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.endswith("_API_KEY")}       # candidates never see a key
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT), os.environ.get("PYTHONPATH")]))
    remote = os.environ.get("KERNEL_EVAL_SSH")
    if remote:
        cmd = json.loads(remote) + [os.environ.get("KERNEL_EVAL_REMOTE", "") + f" python -m kernels.evolve.jailed - - {stage}"]
        stdin = json.dumps({"config": json.loads(Path(os.environ["KERNEL_EVAL_CONFIG"]).read_text()),
                            "source": Path(program_path).read_text()})
    else:
        cmd = [sys.executable, "-m", "kernels.evolve.jailed", os.environ["KERNEL_EVAL_CONFIG"], program_path, str(stage)]
        stdin = None
    zero = {"combined_score": 0.0, "correct": 0.0, "stage": stage}
    try:
        limit = float(os.environ.get("KERNEL_EVAL_TIMEOUT_S", TIMEOUT_S))      # read per call: the gate sets more
        p = subprocess.run(cmd, input=stdin, capture_output=True, text=True, timeout=limit, env=env, cwd=ROOT)
    except subprocess.TimeoutExpired:
        return {"metrics": zero, "artifacts": {"error": f"evaluation exceeded {limit:.0f} s"}}
    lines = [x for x in p.stdout.splitlines() if x.startswith("{")]
    if p.returncode != 0 or not lines:
        return {"metrics": zero,
                "artifacts": {"error": f"evaluation process exited {p.returncode}:\n{(p.stderr or p.stdout)[-3000:]}"}}
    d = json.loads(lines[-1])
    return {"metrics": d["metrics"], "artifacts": d.get("artifacts") or {}}


def _run(program_path: str, stage: int):
    from openevolve.evaluation_result import EvaluationResult
    d = judge_stage(program_path, stage)
    return EvaluationResult(d["metrics"], d["artifacts"])


def evaluate_stage1(program_path: str):
    return _run(program_path, 1)


def evaluate_stage2(program_path: str):
    return _run(program_path, 2)


def evaluate(program_path: str):
    r = _run(program_path, 1)
    return r if r.metrics.get("correct", 0) < 1 else _run(program_path, 2)
