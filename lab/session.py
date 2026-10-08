"""The one loop: measure the base once, then while budget remains, start a session with the task, the baseline and the
tools, and record what it did."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from lab import agent, engine, ledger, target
from lab.budget import Budget, BudgetExceeded
from lab.safety import grader
from lab.task import Task
from lab.tools import Toolbox, clean_pristine
from lab.workspace import Workspace

SYSTEM = (Path(__file__).parent / "prompts" / "system.md").read_text()      # a template, filled by system_prompt
MIN_SESSION_USD = 0.25


@dataclass
class Session:
    """What one agent session sees and what its tools need: wired by `run`, read by `Toolbox`."""
    run_id: str
    session_id: str
    run_dir: Path
    workspace: Workspace
    budget: Budget
    ledger_root: Path
    profile_runner: object = None          # None: lab.tools.default_profile_runner
    task: Task | None = None


@dataclass
class RunConfig:
    goal: str
    budget_usd: float
    task: Task | None = None              # the objective and constraints; `goal` is its text when given
    repo: Path = field(default_factory=engine.repo)     # the engine; the agent's workspace is exported from it
    base: str = "HEAD"
    runs_dir: Path = engine.ENV_ROOT / "lab" / "runs"
    ledger_root: Path = ledger.ROOT
    run_id: str | None = None             # resume an earlier run's workspace and spend
    max_sessions: int = 50
    max_turns: int = 200
    session_usd: float = 5.0              # per-session cap handed to the provider
    provider: str | None = None
    baseline: bool = True                 # measure the base commit once per run before the first session
    extra: dict = field(default_factory=dict)


def referee(s: Session) -> dict:
    """What the agent gets from the referee at the start of every session: the integrity verdict on this run so
    far, with evidence, and the harness facts it cannot read off its ledger. Nothing judged, nothing advised."""
    from feedback.report import lab_feedback
    from lab.evaltools import harness_facts
    recs = list(ledger.records(s.ledger_root, run=s.run_id))
    try:
        facts = harness_facts(target.load(), s.ledger_root)
    except Exception as e:                  # a missing corpus or reference is itself a fact
        facts = {"unavailable": f"{type(e).__name__}: {e}"}
    return lab_feedback(recs, s.run_id, str(s.ledger_root), facts)["for_agent"]


def system_prompt(t: target.Target, spec: agent.AgentSpec) -> str:
    """The system prompt from the target spec and the tool specs: what is there and what it costs, nothing advised."""
    from lab.evaltools import HOLDOUT_BUDGET
    from lab.safety.surfaces import ALWAYS_DENY

    def globs(g):
        return ", ".join(f"`{x}`" for x in g) or "nothing"
    return SYSTEM.format(
        target=t.name, model=t.model, engine=t.engine_repo.name, launch=t.engine.launch,
        env=json.dumps(t.engine.env), write=globs(t.write), add_only=globs(t.add_only), deny=globs(ALWAYS_DENY),
        tools="\n".join(f"- `{x.name}`: {x.description}" for x in spec.tools),
        session_usd=spec.max_budget_usd, max_turns=spec.max_turns, timeout_s=spec.timeout_s,
        min_session_usd=MIN_SESSION_USD, holdout=HOLDOUT_BUDGET)


def _baseline(s: Session) -> str:
    b = next(ledger.records(s.ledger_root, run=s.run_id, kind="baseline"), None)
    if b is None:
        return "Baseline: not measured in this run."
    c, r = b["config"], b["result"]
    head = f"Baseline (base commit {c['commit']}, {c['split']} split, {c['tier']} tier, record {b['id']})"
    if r["verdict"] != "ok":
        return f"{head}: not measured; {r['reason']}"
    return f"{head}:\n{json.dumps(r['metrics'])}"


def brief(cfg: RunConfig, s: Session, snapshot_id: str, n: int) -> str:
    last = [r for r in ledger.records(s.ledger_root, run=s.run_id, kind="session")][-1:]
    head = cfg.task.brief() if cfg.task else f"Goal: {cfg.goal}"
    return (f"{head}\n\n"
            f"Run {s.run_id}, session {n}. Budget: ${s.budget.remaining_usd:.2f} of ${s.budget.cap_usd:.2f} left.\n"
            f"Workspace snapshot: {snapshot_id}. Base commit: {cfg.base}.\n\n"
            f"Referee (this run so far):\n{json.dumps(referee(s), default=str)}\n\n"
            f"{_baseline(s)}\n\n"
            f"Last session of this run:\n{json.dumps(last[0]) if last else 'none'}\n\n"
            "Every record of this and earlier runs is readable with the `ledger` tool.")


def _resolve(repo: Path, ref: str) -> str:
    return subprocess.run(["git", "-C", str(repo), "rev-parse", ref], capture_output=True,
                          text=True, check=True).stdout.strip()


def run(cfg: RunConfig, provider: agent.Provider | None = None) -> dict:
    provider = provider or agent.load(cfg.provider)
    base = _resolve(cfg.repo, cfg.base)
    run_id = cfg.run_id or f"run-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    run_dir = Path(cfg.runs_dir) / run_id
    ws = Workspace(cfg.repo, base, run_dir / "workspace", cfg.ledger_root)
    if not ws.path.exists():
        ws.create()
    budget = Budget.resume(cfg.budget_usd, run_id, cfg.ledger_root)
    done = sum(1 for _ in ledger.records(cfg.ledger_root, run=run_id, kind="session"))
    base_files = grader.base_files(cfg.repo, base)
    summary = {"run": run_id, "base": base, "sessions": 0, "stopped": None, "spent_usd": budget.spent_usd}
    if cfg.baseline:
        Toolbox(Session(run_id, f"{run_id}-baseline", run_dir, ws, budget, cfg.ledger_root, task=cfg.task)).baseline(base)

    for n in range(done + 1, done + cfg.max_sessions + 1):
        if budget.remaining_usd < MIN_SESSION_USD:
            summary["stopped"] = "budget"
            break
        s = Session(run_id, f"{run_id}-s{n}", run_dir, ws, budget, cfg.ledger_root, task=cfg.task)
        tools = Toolbox(s)
        snap = ws.snapshot()
        spec = agent.AgentSpec(system="", prompt=brief(cfg, s, snap.id, n), workspace=ws.path,
                               scratch=run_dir / "scratch" / s.session_id, tools=tools.specs(),
                               base_files=base_files, max_turns=cfg.max_turns,
                               max_budget_usd=min(cfg.session_usd, budget.remaining_usd))
        spec.system = system_prompt(target.load(), spec)
        t0 = time.monotonic()
        try:
            reply = provider.run(spec)
        except Exception as e:                  # a provider that raises still gets charged and recorded
            reply = agent.AgentReply(None, spec.max_budget_usd, 0, f"provider raised: {e}", True)
        clean_pristine(run_dir)
        try:
            budget.charge(reply.cost_usd, s.session_id)
            over = None
        except BudgetExceeded as e:             # the provider overshot its cap; recorded, then the run ends
            budget.spent_usd += reply.cost_usd
            over = str(e)
        end = ws.snapshot()
        if end.violations and not tools.violation:      # a Bash write outside the surface, with no tool call after
            tools.violation = "; ".join(end.violations)
        status = (reply.output or {}).get("status") if reply.output else None
        ledger.append({"kind": "session", "run": run_id, "session": s.session_id, "provider": provider.name,
                       "model": spec.model, "snapshot": end.id, "snapshot_blob": end.blob, "patch": end.patch,
                       "cost": {"usd": reply.cost_usd, "estimated": reply.cost_estimated},
                       "turns": reply.turns, "seconds": time.monotonic() - t0,
                       "status": status, "error": reply.error, "violation": tools.violation,
                       "claim": {"note": (reply.output or {}).get("note")}}, cfg.ledger_root)
        summary["sessions"] = n - done
        summary["spent_usd"] = budget.spent_usd
        if tools.violation:
            summary["stopped"] = f"security: {tools.violation}"
            break
        if over:
            summary["stopped"] = over
            break
        if status == "stop":
            summary["stopped"] = "agent"
            break
    else:
        summary["stopped"] = "max_sessions"
    if summary["stopped"] is None:
        summary["stopped"] = "budget"
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m lab.session")
    ap.add_argument("--goal", help="plain goal text (or give --task)")
    ap.add_argument("--task", help="task spec TOML: goal, objective regimes, constraints (lab/task.py)")
    ap.add_argument("--budget", type=float, required=True, help="dollars for the whole run")
    ap.add_argument("--base", default="HEAD")
    ap.add_argument("--run", help="resume this run id")
    ap.add_argument("--max-sessions", type=int, default=50)
    ap.add_argument("--session-usd", type=float, default=5.0)
    ap.add_argument("--provider")
    a = ap.parse_args(argv)
    task = Task.parse(a.task) if a.task else None
    if not task and not a.goal:
        ap.error("give --task or --goal")
    out = run(RunConfig(goal=task.goal if task else a.goal, budget_usd=a.budget, task=task, base=a.base, run_id=a.run,
                        max_sessions=a.max_sessions, session_usd=a.session_usd, provider=a.provider))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
