"""A win on the engine repo: the patch kept in the ledger, its branch pushed, a pull request labelled perf opened."""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from lab.loop.types import Candidate, Pushed
from lab.records import ledger

LABEL = "perf"


def sh(argv: list[str], cwd: Path, stdin: str | None = None) -> str:
    return subprocess.run(argv, cwd=cwd, input=stdin, text=True, capture_output=True, check=True).stdout.strip()


def push(c: Candidate, body: str, repo: Path, branch: str, onto: str, root: Path = ledger.ROOT, sh=sh) -> Pushed:
    """`c.patch` committed on a new `branch` from `onto` in a worktree of `repo`, pushed, and a pull request into
    `onto` opened with LABEL; nothing merges it."""
    blob = ledger.put_blob(c.patch.encode(), root, ".patch")
    with tempfile.TemporaryDirectory() as d:
        tree = Path(d) / "tree"
        sh(["git", "worktree", "add", "-b", branch, str(tree), onto], repo)
        try:
            sh(["git", "apply", "--index", "-"], tree, c.patch)
            sh(["git", "commit", "-m", c.note], tree)
            sh(["git", "push", "-u", "origin", branch], tree)
            url = sh(["gh", "pr", "create", "--base", onto, "--head", branch, "--label", LABEL, "--title", c.note,
                      "--body", body], tree)
        finally:
            sh(["git", "worktree", "remove", "--force", str(tree)], repo)
    return Pushed(branch, url.splitlines()[-1] if url else "", blob)
