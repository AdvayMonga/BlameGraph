"""Every package, module and path this repo names as text exists."""
from __future__ import annotations

import ast
import glob
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEXT = (".py", ".sh", ".yml", ".md", ".toml", ".Dockerfile")
MODULE = re.compile(r"(?<![\w./-])(?:lab|evals)(?:\.[a-z_][a-z0-9_]*)+")
PATH = re.compile(r"(?<![\w./-])(?:lab|evals)/[\w./*-]+")
LABELS = {"lab.jail", "lab.workbench"}                               # docker labels
NOT_HERE = ("lab/runs", "lab/kernels/library/ptx/T.txt")             # the agent's workspace; a placeholder in a usage line
SUFFIXES = {"py", "md", "toml", "sh", "json", "jsonl", "yml"}        # `lab.py` is a file name, not a module


def _files() -> list[Path]:
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    return [ROOT / f for f in out if f.endswith(TEXT) and not f.startswith("lab/kernels/library/src/")]


def _is_module(parts: list[str]) -> bool:
    p = ROOT.joinpath(*parts)
    return p.with_suffix(".py").is_file() or (p / "__init__.py").is_file()


def _assigned(path: str, name: str):
    tree = ast.parse((ROOT / path).read_text())
    return next(ast.literal_eval(n.value) for n in tree.body
                if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == name)


def test_every_named_module_exists():
    missing = set()
    for f in _files():
        for m in MODULE.findall(f.read_text()):
            parts = m.split(".")
            if parts[-1] in SUFFIXES or m in LABELS:
                continue
            # the last part may be a function or class of the module before it
            if not (_is_module(parts) or (len(parts) > 2 and _is_module(parts[:-1]))):
                missing.add((str(f.relative_to(ROOT)), m))
    assert not missing, sorted(missing)


def _is_path(p: str) -> bool:
    stem = p.rsplit(".", 1)[0]                                       # `lab/tools.stage_harness`: a module and a name in it
    return any(glob.glob(str(ROOT / q)) for q in (p, p + ".py", stem + ".py"))


def test_every_named_path_exists():
    missing = set()
    for f in _files():
        if f.parent.name == "tests":                                 # fixtures name files of made-up trees
            continue
        for p in PATH.findall(f.read_text()):
            p = p.rstrip(".-")
            if not (_is_path(p) or p.startswith(NOT_HERE) or p.endswith(("heldout.jsonl", ".seed"))):   # not in git
                missing.add((str(f.relative_to(ROOT)), p))
    assert not missing, sorted(missing)


def test_the_jails_copy_lists_name_real_files():
    for pkg, files in _assigned("lab/kernels/eval/jailed.py", "HARNESS").items():
        assert (ROOT / pkg / "__init__.py").is_file(), pkg
        for f in files or ():
            assert (ROOT / pkg / f).is_file(), f"{pkg}/{f}"
    for f in _assigned("lab/tools.py", "HARNESS"):
        assert (ROOT / "lab" / "profile" / f).is_file(), f


def test_shell_scripts_reach_real_paths():
    for f in _files():
        if f.suffix != ".sh":
            continue
        for rel in re.findall(r"\$here/([\w./-]+)", f.read_text()):
            if "local/" not in rel and rel.strip("./"):
                assert (f.parent / rel).exists(), f"{f.relative_to(ROOT)}: $here/{rel}"


def test_the_held_out_split_is_named_where_the_corpus_is():
    held = "evals/regimes/*/heldout.jsonl"
    assert glob.glob(str(ROOT / "evals/regimes/*/seen.jsonl"))
    assert held in (ROOT / ".gitignore").read_text().split()
    assert held in _assigned("lab/infra/sandbox/surfaces.py", "HIDDEN")
    assert f"--include={held}" in (ROOT / "lab/infra/cloud/vm.py").read_text()
