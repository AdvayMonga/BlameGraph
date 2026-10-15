"""Experiments made outside the lab (a person's change on a branch), recorded the way the agent's are: a run whose
workspace is the change, a `hypothesis`, measurements through the lab's own tools with an `expect`, and a
`reflection` on each. Run ids start with `researcher-`, so no agent view shows these records (lab/experiments.py).

  python -m lab.experiment start --base REF --change REF --budget USD [--repo PATH]   -> prints the run id
  python -m lab.experiment baseline --run ID
  python -m lab.experiment hypothesis --run ID --hypothesis TEXT --rationale TEXT [--abandon-if TEXT]
  python -m lab.experiment call TOOL --run ID [--args JSON] [--hypothesis-id H --metric M --value V --unit U --why T]
  python -m lab.experiment reflect --run ID --record ID --reflection TEXT [--alternative TEXT]

The change's files outside the target's write surface are left out of the measured tree, and the start record lists
them: what is measured is what an agent's change could have been."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path

from lab import engine, experiments, ledger
from lab.budget import Budget
from lab.safety.surfaces import may_write
from lab.workspace import Workspace

RUNS = engine.ENV_ROOT / "lab" / "runs"


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
                   "result": {"author": _git(repo, "config", "user.name").strip() or None, "applied": applied,
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m lab.experiment", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("start")
    p.add_argument("--repo", type=Path, default=None, help="the engine repo (default: LAB_ENGINE_REPO)")
    p.add_argument("--base", required=True)
    p.add_argument("--change", required=True)
    p.add_argument("--budget", type=float, required=True, help="dollars of GPU time for this run")
    p = sub.add_parser("baseline")
    p.add_argument("--run", required=True)
    p = sub.add_parser("hypothesis")
    p.add_argument("--run", required=True)
    p.add_argument("--hypothesis", required=True)
    p.add_argument("--rationale", required=True)
    p.add_argument("--abandon-if")
    p = sub.add_parser("call")
    p.add_argument("tool")
    p.add_argument("--run", required=True)
    p.add_argument("--args", default="{}", help="the tool's arguments as JSON")
    p.add_argument("--hypothesis-id")
    p.add_argument("--metric")
    p.add_argument("--value", type=float)
    p.add_argument("--unit", choices=experiments.UNITS)
    p.add_argument("--why")
    p = sub.add_parser("reflect")
    p.add_argument("--run", required=True)
    p.add_argument("--record", required=True)
    p.add_argument("--reflection", required=True)
    p.add_argument("--alternative")
    a = ap.parse_args(argv)

    try:
        print(_run(a))
    except (ValueError, PermissionError) as e:      # a refusal is the lab's answer, not a crash
        print(f"refused: {e}", file=sys.stderr)
        return 1
    return 0


def _run(a) -> str:
    if a.cmd == "start":
        return start(a.repo or engine.repo(), a.base, a.change, a.budget, ledger.ROOT, RUNS)
    tb = toolbox(a.run, ledger.ROOT, RUNS)
    if a.cmd == "baseline":
        return json.dumps(tb.baseline(tb.s.workspace.base)["result"])
    if a.cmd == "hypothesis":
        return call(tb, "hypothesis", {k: v for k, v in {"hypothesis": a.hypothesis, "rationale": a.rationale,
                                                          "abandon_if": a.abandon_if}.items() if v})
    if a.cmd == "reflect":
        return call(tb, "reflect", {k: v for k, v in {"record": a.record, "reflection": a.reflection,
                                                       "alternative": a.alternative}.items() if v})
    args = json.loads(a.args)
    if a.hypothesis_id:
        args["hypothesis_id"] = a.hypothesis_id
    if a.metric is not None:
        args["expect"] = {"metric": a.metric, "value": a.value, "unit": a.unit, "why": a.why}
    return call(tb, a.tool, args)


if __name__ == "__main__":
    sys.exit(main())
