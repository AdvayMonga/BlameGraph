"""The OS jail untrusted code runs in: sandbox-runtime (`srt`) settings and command wrapper."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path


class JailMissing(RuntimeError):
    """No sandbox-runtime on this machine, and LAB_NO_JAIL is not set."""


def available() -> bool:
    return shutil.which("srt") is not None


def backend() -> str:
    """`container` on Linux, `srt` elsewhere; LAB_JAIL overrides."""
    return os.environ.get("LAB_JAIL") or ("container" if sys.platform.startswith("linux") else "srt")


def required() -> bool:
    """False only when LAB_NO_JAIL=1."""
    return os.environ.get("LAB_NO_JAIL") != "1"


def settings(writable: list[Path], venv: Path, domains: list[str],
             readonly: tuple[Path, ...] | list[Path] = (), python: str | None = None,
             local_binding: bool = False) -> dict:
    """srt settings: write only `writable`; read nothing under home but it, `readonly`, `venv` and the interpreter;
    reach only `domains`. `python` is the interpreter run inside; `local_binding` allows listening on localhost."""
    # the base interpreter a venv's python links to
    interpreter = {Path(sys.base_prefix).resolve(), Path(python or sys.executable).resolve().parent.parent}
    return {
        "network": {"allowedDomains": list(domains), "deniedDomains": [],
                    "allowUnixSockets": [], "allowAllUnixSockets": False,
                    "allowLocalBinding": local_binding},
        "filesystem": {
            "denyRead": [str(Path.home())],
            "allowRead": [str(p) for p in [*writable, *readonly, venv, *sorted(interpreter)]],
            "allowWrite": [str(p) for p in writable],
            "denyWrite": [],
        },
        "enableWeakerNestedSandbox": False,
        "enableWeakerNetworkIsolation": False,
        "allowAppleEvents": False,
    }


def wrap(config: dict, settings_path: Path, argv: list[str]) -> list[str]:
    """`argv` under srt with `config` (written to `settings_path`); bare argv when the jail is waived."""
    if not available():
        if required():
            raise JailMissing("sandbox-runtime not installed: npm install -g @anthropic-ai/sandbox-runtime "
                              "(or LAB_NO_JAIL=1 on a box you trust)")
        return argv
    settings_path.write_text(json.dumps(config, indent=2))
    return [shutil.which("srt"), "--settings", str(settings_path), "--", *argv]   # `--`: a `-c` in argv is not srt's
