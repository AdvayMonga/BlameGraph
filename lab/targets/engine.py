"""Where the engine under optimization lives, per the target spec: its repo and the Python that runs its code."""

from __future__ import annotations

import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

ENV_ROOT = Path(__file__).resolve().parents[2]


def repo() -> Path:
    """The target's engine repo (LAB_ENGINE_REPO overrides)."""
    from evals.targets import target
    return target.load().engine_repo


_using: list[str] = []


@contextmanager
def using(py: str):
    """`python()` is `py` inside the block."""
    _using.append(py)
    try:
        yield
    finally:
        _using.pop()


def python() -> str:
    """The target's engine interpreter (LAB_ENGINE_PYTHON overrides); this interpreter if that file does not exist."""
    if _using:
        return _using[-1]
    from evals.targets import target
    py = target.load().engine_python
    return py if Path(py).exists() else sys.executable


def venv(py: str | None = None) -> Path:
    """The prefix the jail must let `py` read: its venv dir (bin/python -> two levels up)."""
    return Path(py or python()).parent.parent


def commit(repo: Path, ref: str = "HEAD", clean: bool = False) -> str | None:
    """`ref` as a commit sha in the engine repo; with `clean`, None when the working tree has uncommitted changes."""
    def git(*a):
        return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True, check=True).stdout.strip()
    try:
        if clean and git("status", "--porcelain", "--untracked-files=no"):
            return None
        return git("rev-parse", ref)
    except (OSError, subprocess.CalledProcessError):
        return None
