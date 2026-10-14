"""Launch profiles (decided 2026-10-14): an engine may ship a config per regime in a TOML file of its own tree, named
by the target (`[engine] profiles`): `[general]` plus a table per regime. A measurement serves the general config once
for every regime, then serves each regime whose table is non-empty on its own with `<profile_env>=<regime>`; that
launch's number is the regime's headline. The harness reads only the file, never engine code, to decide the launches."""

from __future__ import annotations

import tomllib
from pathlib import Path

from lab import target


def overrides(tree: Path, t: target.Target) -> dict[str, dict]:
    """{regime: its table} for every non-empty table but [general]; {} when the target names no file or it is absent.
    A file that does not parse raises: the measurement fails rather than silently serving one config."""
    if not t.profiles:
        return {}
    p = Path(tree) / t.profiles
    if not p.is_file():
        return {}
    raw = tomllib.loads(p.read_text())
    return {k: v for k, v in raw.items() if k != "general" and isinstance(v, dict) and v}


def launches(tree: Path, t: target.Target, names: list[str]) -> list[tuple[str | None, list[str]]]:
    """(profile, regimes) per engine launch: (None, every regime) for the general config, then one per regime in
    `names` that has its own table."""
    own = overrides(tree, t)
    return [(None, list(names))] + [(n, [n]) for n in names if n in own]


def refusal(tree: Path, t: target.Target, profile: str | None) -> str | None:
    """Why `profile` cannot be served from `tree`, or None (no profile is the general launch)."""
    if not profile:
        return None
    if not t.profiles:
        return "this target names no profiles file"
    if profile not in overrides(tree, t):
        return f"no [{profile}] table with settings in {t.profiles}"
    return None


def env(t: target.Target, profile: str | None) -> dict[str, str]:
    """The launch env that selects `profile` (nothing for the general config)."""
    return {t.profile_env: profile} if profile else {}
