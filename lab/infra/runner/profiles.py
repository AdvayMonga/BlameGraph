"""Launch profiles: the engine's per-regime configs, read from the TOML file the target names (`[engine] profiles`).
One launch serves `[general]` for every regime, then one launch per regime with a non-empty table of its own."""

from __future__ import annotations

import tomllib
from pathlib import Path

from lab.targets import target


def overrides(tree: Path, t: target.Target) -> dict[str, dict]:
    """{regime: its table} for every non-empty table but [general]; {} when the target names no file or it is absent.
    Raises tomllib.TOMLDecodeError on a file that does not parse."""
    if not t.profiles:
        return {}
    p = Path(tree) / t.profiles
    if not p.is_file():
        return {}
    raw = tomllib.loads(p.read_text())
    return {k: v for k, v in raw.items() if k != "general" and isinstance(v, dict) and v}


def launches(tree: Path, t: target.Target, names: list[str]) -> list[tuple[str | None, list[str]]]:
    """(profile, regimes) per engine launch: (None, every regime), then one per regime in `names` with its own table."""
    own = overrides(tree, t)
    return [(None, list(names))] + [(n, [n]) for n in names if n in own]


def refusal(tree: Path, t: target.Target, profile: str | None) -> str | None:
    """Why `profile` cannot be served from `tree`, or None. Never raises."""
    from evals.loadgen import suite
    if not profile:
        return None
    if not t.profiles:
        return "this target names no profiles file"
    if profile not in suite.REGIMES:
        return f"{profile!r} is not a regime; one of {sorted(suite.REGIMES)}"
    try:
        own = overrides(tree, t)
    except ValueError as e:                       # tomllib.TOMLDecodeError
        return f"{t.profiles} does not parse: {e}"
    if profile not in own:
        return f"no [{profile}] table with settings in {t.profiles}"
    return None


def env(t: target.Target, profile: str | None) -> dict[str, str]:
    """The launch env that selects `profile` (nothing for the general config)."""
    return {t.profile_env: profile} if profile else {}
