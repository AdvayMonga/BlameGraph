"""Where the engine under optimization lives, per the target spec (lab/target.py): its repo (the agent's workspace
is exported from it) and the Python that runs its code. The environment never imports the engine itself."""

from __future__ import annotations

import sys
from pathlib import Path

ENV_ROOT = Path(__file__).resolve().parents[1]


def repo() -> Path:
    """The target's engine repo (LAB_ENGINE_REPO overrides)."""
    from lab import target
    return target.load().engine_repo


def python() -> str:
    """The target's engine interpreter (LAB_ENGINE_PYTHON overrides); this interpreter if that file does not exist.
    The engine's venv needs what its test and lint commands import (e.g. `uv sync --extra dev`)."""
    from lab import target
    py = target.load().engine_python
    return py if Path(py).exists() else sys.executable


def venv(py: str | None = None) -> Path:
    """The prefix the jail must let `py` read: its venv dir (bin/python -> two levels up)."""
    return Path(py or python()).parent.parent
