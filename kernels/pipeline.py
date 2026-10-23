"""The kernel layer as one call (the central agent's `kernel_search` tool, decided 2026-10-22): capture the task's
calls under real traffic, split them seen / held-out, search, gate the winner on the held-out shapes, and hand the
engine module back as a patch the agent applies itself. It runs on the controller: the search's model calls stay
here, and its evaluator runs where the GPU is (`remote_env` for a worker host, kernels/evolve/evaluator.py's ssh mode).
A blocking call for now; a parallel job on its own GPU host comes later."""
from __future__ import annotations

import difflib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Callable

# The capture traffic (kernels/evolve/validate_gpu.sh's CAPTURE_SPECS): corpus classes at real concurrencies.
CAPTURE_SPECS = (("mixed_1", 256), ("mixed_3", 64), ("long_context", 60))
ROOT = Path(__file__).resolve().parents[1]


def module_file(task, tree: Path) -> Path:
    """The engine module the task's entry point lives in, inside an engine tree."""
    mod = task.entry.partition(":")[0]
    return Path(tree) / "src" / (mod.replace(".", "/") + ".py")


def start_program(task, tree: Path) -> str:
    """The starting program from the engine module in `tree` (the agent's workspace, not a commit)."""
    from kernels.evolve.run import _start
    return _start(task, module_file(task, tree).read_text(), task.entry.partition(":")[2])


def captures(task, tree: Path, profile: Callable[[Path, dict], Path | None]) -> list[Path]:
    """Each capture traffic's `captures/` dir, from `profile(tree, args) -> bundle dir` (the `profile` job)."""
    out = []
    for cls, n in CAPTURE_SPECS:
        bundle = profile(tree, {"capture": [task.entry], "corpus_class": cls, "requests": n})
        if bundle is not None and (Path(bundle) / "captures").is_dir():
            out.append(Path(bundle) / "captures")
    return out


def remote_env(prefix: list[str], env_dir: str, python: str, env: dict) -> dict:
    """The evaluator's ssh mode for a worker host: each judgment runs there, config and source on stdin."""
    exports = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
    return {"KERNEL_EVAL_SSH": json.dumps(prefix),
            "KERNEL_EVAL_REMOTE": f"cd {shlex.quote(env_dir)} && PATH={shlex.quote(str(Path(python).parent))}:$PATH "
                                  f"{exports}"}


def search(task, caps: list[Path], start: Path, out: Path, usd: float, evals: int, ledger_root: Path, env: dict,
           funnel: bool = False) -> dict:
    """`python -m kernels.search` on the captures: its last JSON line (best, cost, out, ledger record id)."""
    argv = [sys.executable, "-m", "kernels.search", "--task", task.name, "--out", str(out), "--usd", str(usd),
            "--evals", str(evals), "--initial", str(start), "--ledger", str(ledger_root),
            *[w for c in caps for w in ("--captures", str(c))], *(["--funnel"] if funnel else [])]
    p = subprocess.run(argv, capture_output=True, text=True, cwd=ROOT, env={**os.environ, **env})
    lines = [x for x in p.stdout.splitlines() if x.startswith("{")]
    if p.returncode != 0 or not lines:
        raise RuntimeError(f"kernels.search exited {p.returncode}: {(p.stderr or p.stdout)[-2000:]}")
    return json.loads(lines[-1])


def gate(task, caps: list[Path], program: Path, out: Path, env: dict) -> dict:
    """The held-out gate on the winner, judged where the evaluator runs (here, or the worker over ssh)."""
    from kernels import shapes as shapes_mod
    from kernels.evolve import evaluator
    from kernels.evolve.run import _ints, _target_tolerances, options_from_target, tolerances
    sets = shapes_mod.split(task, shapes_mod.from_captures(task, caps))
    sets["ints"] = shapes_mod.ints_from_captures(task, caps)
    cfg = {"task": task.name, "seen": [list(s) for s in sets["seen"]], "heldout": [list(s) for s in sets["heldout"]],
           "weights": sets.get("calls"), "device": "cuda", "tolerances": tolerances(_target_tolerances()),
           "ints": _ints(sets, ("seen", "heldout")), "options": options_from_target(task)}
    path = Path(out) / "gate_config.json"
    path.write_text(json.dumps(cfg))
    saved = {k: os.environ.get(k) for k in ("KERNEL_EVAL_CONFIG", *env)}
    os.environ.update({"KERNEL_EVAL_CONFIG": str(path), **env})
    try:
        return evaluator.judge_stage(str(program), 2)
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


def patch(task, tree: Path, program: str) -> str:
    """The winner as a unified diff of the engine module in `tree`, the task's start edits undone."""
    from kernels.evolve.ship import module_source, unstart
    path = module_file(task, tree)
    old = path.read_text()
    new = unstart(task, module_source(program, task.entry.partition(":")[2]))
    rel = path.relative_to(tree).as_posix()
    return "".join(difflib.unified_diff(old.splitlines(True), new.splitlines(True), f"a/{rel}", f"b/{rel}"))
