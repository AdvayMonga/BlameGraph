"""OpenEvolve's evaluator: two stages, each a fresh jailed process; config from $KERNEL_EVAL_CONFIG.
With $KERNEL_EVAL_SSH (a JSON argv prefix) and $KERNEL_EVAL_REMOTE (a shell prefix) each stage runs on the GPU host."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

TIMEOUT_S = float(os.environ.get("KERNEL_EVAL_TIMEOUT_S", "600"))
ROOT = Path(__file__).resolve().parents[3]


def tail(out: str, n: int = 3000) -> str:
    """The last `n` characters of a process's output, C++ stack frames removed."""
    return "\n".join(x for x in out.splitlines() if not x.startswith("frame #"))[-n:]


def seconds(log: Path) -> float:
    """Wall seconds spent judging, summed from the log `judge_stage` appends to."""
    return round(sum(float(x) for x in Path(log).read_text().split()), 1) if Path(log).exists() else 0.0


def judge_stage(program_path: str, stage: int) -> dict:
    """{metrics, artifacts} for one stage of one candidate, judged jailed; wall time appended to $KERNELS_EVAL_LOG."""
    t0 = time.monotonic()
    try:
        return _judge_stage(program_path, stage)
    finally:
        if os.environ.get("KERNELS_EVAL_LOG"):
            with open(os.environ["KERNELS_EVAL_LOG"], "a") as f:
                f.write(f"{time.monotonic() - t0:.2f}\n")


def _judge_stage(program_path: str, stage: int) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.endswith("_API_KEY")}       # no API keys
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT), os.environ.get("PYTHONPATH")]))
    remote = os.environ.get("KERNEL_EVAL_SSH")
    if remote:
        cmd = json.loads(remote) + [os.environ.get("KERNEL_EVAL_REMOTE", "") + f" python -m lab.kernels.eval.jailed - - {stage}"]
        stdin = json.dumps({"config": json.loads(Path(os.environ["KERNEL_EVAL_CONFIG"]).read_text()),
                            "source": Path(program_path).read_text()})
    else:
        cmd = [sys.executable, "-m", "lab.kernels.eval.jailed", os.environ["KERNEL_EVAL_CONFIG"], program_path, str(stage)]
        stdin = None
    zero = {"combined_score": 0.0, "correct": 0.0, "stage": stage}
    try:
        limit = float(os.environ.get("KERNEL_EVAL_TIMEOUT_S", TIMEOUT_S))      # read per call
        p = subprocess.run(cmd, input=stdin, capture_output=True, text=True, timeout=limit, env=env, cwd=ROOT)
    except subprocess.TimeoutExpired:
        return {"metrics": zero, "artifacts": {"error": f"evaluation exceeded {limit:.0f} s"}}
    lines = [x for x in p.stdout.splitlines() if x.startswith("{")]
    if p.returncode != 0 or not lines:
        return {"metrics": zero,
                "artifacts": {"error": f"evaluation process exited {p.returncode}:\n{tail(p.stderr or p.stdout)}"}}
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
