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


def unstart(task, src: str) -> str:
    """The task's start edits undone where the program kept them as they were (`run._start`): the engine gets back its
    own spelling, and the commit holds the kernel change only."""
    for old, new in task.start_edits:
        if src.count(new) == 1:
            src = src.replace(new, old)
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


def _overrides_at(repo: Path, base: str, pkg: str) -> list[tuple[str, tuple[str, str]]]:
    """(module path, (first, last)) of every OVERRIDES key in the overrides package's modules at `base`."""
    import ast
    out = []
    for path in _git(repo, "ls-tree", "--name-only", f"{base}:{pkg}").split() if _exists(repo, base, pkg) else []:
        if path.endswith(".py"):
            for node in ast.walk(ast.parse(_git(repo, "show", f"{base}:{pkg}/{path}"))):
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict) and \
                        any(isinstance(t, ast.Name) and t.id == "OVERRIDES" for t in node.targets):
                    out += [(f"{pkg}/{path}", tuple(ast.literal_eval(k))) for k in node.value.keys]
    return out


def _exists(repo: Path, base: str, path: str) -> bool:
    return subprocess.run(["git", "-C", str(repo), "cat-file", "-e", f"{base}:{path}"], capture_output=True).returncode == 0


def _region_files(task, program: str, repo: Path, base: str) -> dict[str, str]:
    """A region's winner as a module of the engine's overrides package (`regions.override_module`), beside the
    stations module; the package's `__init__.py` added when the engine lacks it."""
    from kernels import regions
    _, first, last = task.name.split(":")
    root = _module_path(repo, base, regions.ENGINE).removesuffix(regions.ENGINE.replace(".", "/") + ".py")
    pkg = root + regions.OVERRIDES_PACKAGE.replace(".", "/")
    names = [s.name for s in regions.engine_stations()]
    for path, (a, b) in _overrides_at(repo, base, pkg):
        if names.index(a) <= names.index(last) and names.index(first) <= names.index(b):
            raise ValueError(f"region {first}..{last} overlaps the override {a}..{b} already in {path} at {base}")
    files = {f"{pkg}/{first}_{last}.py": regions.override_module(program, first, last)}
    init = f"{pkg}/__init__.py"
    if not _exists(repo, base, init):
        files[init] = '"""Region overrides from the layer-1 loop (models/stations.py: `overrides`)."""\n'
    return files


def ship(task, program: str, gate: dict, repo: Path, base: str = "HEAD", branch: str | None = None) -> dict:
    """The engine branch: {branch, base, commit, path}. Refuses a program without a passing held-out gate."""
    if not gate_passed(gate):
        raise ValueError("the program has no fully passing gate on held-out shapes (`python -m kernels.evolve gate`)")
    base = _git(repo, "rev-parse", base)
    if task.name.startswith("region:"):
        files = _region_files(task, program, repo, base)
        path = next(p for p in files if not p.endswith("__init__.py"))
    else:
        module, _, attr = task.entry.partition(":")
        path = _module_path(repo, base, module)
        files = {path: unstart(task, module_source(program, attr))}
    with tempfile.TemporaryDirectory() as d:
        env = {"GIT_INDEX_FILE": str(Path(d) / "index")}
        _git(repo, "read-tree", base, env=env)
        for p, src in files.items():
            blob = _git(repo, "hash-object", "-w", "--stdin", data=src)
            _git(repo, "update-index", "--add", "--cacheinfo", f"100644,{blob},{p}", env=env)
        tree = _git(repo, "write-tree", env=env)
    msg = f"{task.name}: kernel from the layer-1 loop ({task.entry}), passed the held-out gate"
    commit = _git(repo, "commit-tree", tree, "-p", base, "-m", msg)
    branch = branch or f"kernels/{task.name}-{time.strftime('%Y%m%d-%H%M%S')}"
    _git(repo, "branch", branch, commit)
    return {"branch": branch, "base": base, "commit": commit, "path": path}
