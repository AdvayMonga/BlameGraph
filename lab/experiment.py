"""Experiments made outside the lab (a person's change on a branch), recorded the way the agent's are: a run whose
workspace is the change, a `hypothesis`, measurements through the lab's own tools with an `expect`, and a
`reflection` on each. Run ids start with `researcher-`, so no agent view shows these records (lab/experiments.py).

  lab start --base REF --change REF --budget USD [--repo PATH]   -> prints the run id
  lab baseline --run ID
  lab hypothesis --run ID --hypothesis TEXT --rationale TEXT --layer LAYER [--abandon-if TEXT]
  lab TOOL --run ID --hypothesis-id H --expect JSON [the tool's own arguments]   (lab/cli.py)
  lab reflect --run ID --record ID --reflection TEXT [--alternative TEXT]

The change's files outside the target's write surface are left out of the measured tree, and the start record lists
them: what is measured is what an agent's change could have been."""
from __future__ import annotations

import subprocess
import time
import uuid
from pathlib import Path

from targets import engine
from lab import experiments
from data.knowledge import ledger
from lab.budget import Budget
from measure.sandbox.surfaces import may_write
from lab.workspace import Workspace

RUNS = engine.ENV_ROOT / "local" / "runs"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout


def start(repo: Path, base: str, change: str, budget_usd: float, root: Path = ledger.ROOT, runs: Path = RUNS) -> str:
    """A run whose workspace is `base` plus the files `change` touched inside the write surface."""
    base, change = _git(repo, "rev-parse", base).strip(), _git(repo, "rev-parse", change).strip()
    run = f"{experiments.RESEARCHER}{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    ws = Workspace(repo, base, runs / run / "workspace", root)
    ws.create()
    in_base = set(_git(repo, "ls-tree", "-r", "--name-only", base).splitlines())
    applied, left_out = [], []
    for line in _git(repo, "diff", "--no-renames", "--name-status", base, change).splitlines():
        status, rel = line.split("\t", 1)
        if not may_write(rel, new_file=rel not in in_base):
            left_out.append(rel)
            continue
        dest = ws.path / rel
        if status == "D":
            dest.unlink(missing_ok=True)
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(subprocess.run(["git", "-C", str(repo), "show", f"{change}:{rel}"],
                                            capture_output=True, check=True).stdout)
        applied.append(rel)
    snap = ws.snapshot()
    if snap.violations:
        raise PermissionError(f"the change fails the workspace audit: {snap.violations}")
    ledger.append({"kind": "note", "run": run, "session": f"{run}-researcher", "tool": "start",
                   "args": {"repo": str(repo), "base": base, "change": change, "budget_usd": budget_usd},
                   "snapshot": snap.id, "snapshot_blob": snap.blob, "patch": snap.patch,
                   "result": {"author": _git(repo, "log", "-1", "--format=%an", change).strip(), "applied": applied,
                              "left_out": left_out},
                   "cost": {"usd": 0.0, "seconds": 0.0}}, root)
    return run


def toolbox(run: str, root: Path = ledger.ROOT, runs: Path = RUNS):
    """The lab's Toolbox bound to a researcher run started by `start`."""
    from lab.session import Session
    from lab.tools import Toolbox
    if not experiments.researcher(run):
        raise ValueError(f"{run!r} is not a researcher run")
    s = next((r for r in ledger.records(root, run=run, kind="note") if r.get("tool") == "start"), None)
    if s is None:
        raise ValueError(f"no run {run!r} on this ledger")
    a = s["args"]
    ws = Workspace(Path(a["repo"]), a["base"], runs / run / "workspace", root)
    return Toolbox(Session(run, f"{run}-researcher", runs / run, ws, Budget.resume(a["budget_usd"], run, root), root))


def call(tb, tool: str, args: dict) -> str:
    spec = next((t for t in tb.specs() if t.name == tool), None)
    if spec is None:
        raise ValueError(f"no lab tool {tool!r}; one of {sorted(t.name for t in tb.specs())}")
    return spec.fn(args)
