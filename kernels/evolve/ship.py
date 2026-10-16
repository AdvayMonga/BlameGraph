"""python -m kernels.evolve ship --task T --program BEST.py --gate GATE.json --engine REPO [--base REF] [--budget USD]

A kernel that passed `gate` becomes an engine change: a commit on a new branch of the engine repo whose only change is
the task's module, built with git plumbing so the engine's working tree and index are never touched. With --budget,
the researcher run is started on it (`lab.experiment start`), which the lab then measures end to end (`bench`,
`equiv`, `submit`) with a person's hypothesis and expectations; the kernel loop writes no reasoning of its own."""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path


def _git(repo: Path, *args: str, env: dict | None = None, data: str | None = None) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True, input=data,
                          env={**os.environ, **(env or {})}).stdout.strip()


def module_source(program: str, attr: str) -> str:
    """The engine module inside an evolved program (`run.initial_program`'s markers), with the entry point bound to
    whatever the program's `kernel` names."""
    body = program.split("# EVOLVE-BLOCK-START\n", 1)[-1]
    src, _, tail = body.partition("# EVOLVE-BLOCK-END\n")
    name = re.search(r"^kernel\s*=\s*(\w+)\s*$", tail or src, re.M)
    src = re.sub(r"^kernel\s*=\s*\w+\s*$", "", src, flags=re.M).rstrip("\n") + "\n"
    target = name.group(1) if name else "kernel" if re.search(r"^def kernel\(", src, re.M) else None
    if target and target != attr:                # a full rewrite may define `kernel` itself
        src += f"\n{attr} = {target}\n"
    return src


def gate_passed(gate: dict) -> bool:
    """Every gate passed (correctness, physics, memory, timing/capture) and the held-out check judged > 0 cases."""
    gates = json.loads(gate.get("artifacts", {}).get("gates") or "{}")
    held = gates.get("correctness_heldout") or {}
    return (gate.get("metrics", {}).get("correct") == 1.0 and bool(gates) and
            all(g.get("passed") is True for g in gates.values()) and (held.get("cases") or 0) > 0)


def _module_path(repo: Path, base: str, module: str) -> str:
    rel = module.replace(".", "/") + ".py"
    for p in (f"src/{rel}", rel):
        if subprocess.run(["git", "-C", str(repo), "cat-file", "-e", f"{base}:{p}"], capture_output=True).returncode == 0:
            return p
    raise FileNotFoundError(f"{module} is not in {repo} at {base}")


def ship(task, program: str, gate: dict, repo: Path, base: str = "HEAD", branch: str | None = None) -> dict:
    """The engine branch: {branch, base, commit, path}. Refuses a program without a passing held-out gate."""
    if not gate_passed(gate):
        raise ValueError("the program has no fully passing gate on held-out shapes (`python -m kernels.evolve gate`)")
    module, _, attr = task.entry.partition(":")
    base = _git(repo, "rev-parse", base)
    path = _module_path(repo, base, module)
    blob = _git(repo, "hash-object", "-w", "--stdin", data=module_source(program, attr))
    with tempfile.TemporaryDirectory() as d:
        env = {"GIT_INDEX_FILE": str(Path(d) / "index")}
        _git(repo, "read-tree", base, env=env)
        _git(repo, "update-index", "--cacheinfo", f"100644,{blob},{path}", env=env)
        tree = _git(repo, "write-tree", env=env)
    msg = f"{task.name}: kernel from the layer-1 loop ({task.entry}), passed the held-out gate"
    commit = _git(repo, "commit-tree", tree, "-p", base, "-m", msg)
    branch = branch or f"kernels/{task.name}-{time.strftime('%Y%m%d-%H%M%S')}"
    _git(repo, "branch", branch, commit)
    return {"branch": branch, "base": base, "commit": commit, "path": path}
