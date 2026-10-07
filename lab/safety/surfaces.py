"""What the agent may write. Deny by default: unlisted paths are protected."""

from __future__ import annotations

from fnmatch import fnmatch

from lab import target


def write_surface() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(may add/modify/delete, may add but never change), from the target spec."""
    t = target.load()
    return t.write, t.add_only


# Denied even inside an allowed glob: the evaluator's own code and import-time hooks.
ALWAYS_DENY = (
    "*conftest.py", "*.pth", "*sitecustomize.py", "*usercustomize.py",
    "*ruff.toml", "*pyproject.toml", "*setup.cfg", "*pytest.ini", "*tox.ini",
)
# Removed from the agent's workspace; it can neither read nor recreate them.
HIDDEN = ("corpus/*/heldout.jsonl",)


def may_write(path: str, *, new_file: bool) -> bool:
    """Whether the agent may touch repo-relative `path`."""
    if any(fnmatch(path.lower(), p.lower()) for p in ALWAYS_DENY + HIDDEN):   # macOS ignores case
        return False
    write, add_only = write_surface()
    if any(fnmatch(path, p) for p in write):
        return True
    return new_file and any(fnmatch(path, p) for p in add_only)
