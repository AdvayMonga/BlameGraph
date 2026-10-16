"""python -m kernels.evolve.jailed CONFIG.json PROGRAM|- STAGE  (or `- - STAGE`: {config, source} as JSON on stdin)  ->  kernels.evolve.core's JSON, judged in the jail.

The candidate runs as agent code (lab/safety/grader.jailed: a container with no network on Linux, srt elsewhere;
LAB_NO_JAIL=1 waives it). The jail sees a fresh tree holding only the candidate and its config, a read-only copy of
the harness core needs (kernels/, validity.peaks, lab.target, the measured peaks), and the engine's source read-only
when the task's baseline lives there; never this repo, which holds the held-out shapes, the ledger and .env."""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from kernels.tasks import load
from lab import engine
from lab.safety import grader

ROOT = Path(__file__).resolve().parents[2]
HARNESS = {"kernels": None, "validity": ("__init__.py", "peaks.py"), "lab": ("__init__.py", "target.py")}
TIMEOUT_S = float(os.environ.get("KERNEL_EVAL_TIMEOUT_S", "600"))


def stage_harness(dest: Path) -> Path:
    """Copy what kernels.evolve.core imports from this environment into `dest`."""
    for pkg, files in HARNESS.items():
        if files is None:
            shutil.copytree(ROOT / pkg, dest / pkg, ignore=shutil.ignore_patterns("__pycache__"))
            continue
        (dest / pkg).mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copyfile(ROOT / pkg / f, dest / pkg / f)
    peaks = ROOT / "knowledge" / "peaks"
    if peaks.is_dir():
        shutil.copytree(peaks, dest / "knowledge" / "peaks")
    return dest


def _engine_src(task) -> Path | None:
    """The source root the task's baseline imports from, as this process resolves it (None for a toy task)."""
    if not task.entry:
        return None
    spec = importlib.util.find_spec(task.entry.partition(":")[0].split(".")[0])
    return Path(next(iter(spec.submodule_search_locations))).parent if spec and spec.submodule_search_locations else None


def judge_jailed(cfg: dict, source: str, stage: int, timeout_s: float = TIMEOUT_S) -> dict:
    """core's result for `source` under `cfg`, from a jailed process; a crash or timeout scores 0 with its output."""
    fail = {"metrics": {"combined_score": 0.0, "correct": 0.0, "stage": stage}}
    with tempfile.TemporaryDirectory(prefix="kernel-eval-") as d:
        tree, harness = Path(d, "tree"), stage_harness(Path(d, "harness"))
        tree.mkdir()
        (tree / "candidate.py").write_text(source)
        (tree / "eval_config.json").write_text(json.dumps(cfg))
        src = _engine_src(load(cfg["task"]))
        path = os.pathsep.join(str(p) for p in (harness, src) if p)
        p = grader.jailed(tree, [engine.python(), "-m", "kernels.evolve.core", str(tree / "eval_config.json"),
                                 str(tree / "candidate.py"), str(stage)],
                          timeout_s=timeout_s, env_extra={"PYTHONPATH": path}, read=[harness] + ([src] if src else []))
    lines = [x for x in p.stdout.splitlines() if x.startswith("{")]
    if p.returncode != 0 or not lines:
        return {**fail, "artifacts": {"error": f"evaluation process exited {p.returncode}:\n"
                                               f"{(p.stderr or p.stdout)[-3000:]}"}}
    return json.loads(lines[-1])


def main() -> int:
    stage = int(sys.argv[3])
    if sys.argv[1] == "-":                       # `- - STAGE`: {"config", "source"} on stdin (a remote evaluation)
        d = json.loads(sys.stdin.read())
        cfg, source = d["config"], d["source"]
    else:
        cfg = json.loads(Path(sys.argv[1]).read_text())
        source = sys.stdin.read() if sys.argv[2] == "-" else Path(sys.argv[2]).read_text()
    sys.stdout.write("\n" + json.dumps(judge_jailed(cfg, source, stage)) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
