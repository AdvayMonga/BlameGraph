"""What the agent may write. Deny by default: unlisted paths are protected."""

from __future__ import annotations

from fnmatch import fnmatch

from evals.targets import target


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
HIDDEN = ("evals/regimes/*/heldout.jsonl",)


# The dependency tables of a pyproject.toml; the rest of the file is protected.
DEP_TABLES = (("project", "dependencies"), ("project", "optional-dependencies"), ("dependency-groups",),
              ("tool", "uv", "sources"), ("tool", "uv", "index"))


def may_write(path: str, *, new_file: bool) -> bool:
    """Whether the agent may touch repo-relative `path`; the target's dependency files may be changed."""
    if not new_file and path in target.load().deps:
        return True
    if any(fnmatch(path.lower(), p.lower()) for p in ALWAYS_DENY + HIDDEN):   # case-insensitive
        return False
    write, add_only = write_surface()
    if any(fnmatch(path, p) for p in write):
        return True
    return new_file and any(fnmatch(path, p) for p in add_only)


def deps_only(old: str, new: str) -> bool:
    """Whether a pyproject.toml changed nothing but its dependency tables (and still parses)."""
    import tomllib

    def rest(text: str) -> dict:
        d = tomllib.loads(text)
        for keys in DEP_TABLES:
            path = [d]
            for k in keys[:-1]:
                path.append(path[-1].get(k) if isinstance(path[-1], dict) else None)
            if isinstance(path[-1], dict):
                path[-1].pop(keys[-1], None)
                for parent, k, child in reversed(list(zip(path, keys, path[1:]))):   # drop tables it emptied
                    if child == {}:
                        parent.pop(k)
        return d
    try:
        return rest(old) == rest(new)
    except tomllib.TOMLDecodeError:
        return False
