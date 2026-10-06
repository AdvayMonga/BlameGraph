"""Where the engine under optimization lives: its repo (the agent's workspace is exported from it) and the Python that
runs its code. The environment never imports the engine itself; it runs the engine's own interpreter on it."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ENV_ROOT = Path(__file__).resolve().parents[1]


def repo() -> Path:
    """LAB_ENGINE_REPO, else the sibling checkout ../inference-server."""
    return Path(os.environ.get("LAB_ENGINE_REPO") or ENV_ROOT.parent / "inference-server").expanduser().resolve()


def python() -> str:
    """LAB_ENGINE_PYTHON, else the engine repo's .venv/bin/python when it exists, else this interpreter.
    The engine's venv needs its dev extras (pytest, ruff): `uv sync --extra dev` there."""
    if os.environ.get("LAB_ENGINE_PYTHON"):
        return os.environ["LAB_ENGINE_PYTHON"]
    venv_py = repo() / ".venv" / "bin" / "python"
    return str(venv_py) if venv_py.exists() else sys.executable


def venv(py: str | None = None) -> Path:
    """The prefix the jail must let `py` read: its venv dir (bin/python -> two levels up)."""
    return Path(py or python()).parent.parent
