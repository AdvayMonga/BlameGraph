"""The agent's metered tools. Each runs outside the jail, snapshots the workspace, and writes the ledger whether the agent likes it or not."""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from lab import corpus, engine, ledger, target
from lab.agent import ToolSpec
from lab.budget import Budget, BudgetExceeded, gpu_rate
from lab.evaltools import EvalTools
from lab.proftools import ProfTools
from lab.safety import grader

# Tools that occupy the GPU: their wall time is charged at the venue's rate. The rest cost $0.
GPU_TOOLS = frozenset({"profile", "trace", "kernel", "hostprof", "bench", "equiv", "submit"})


class Toolbox(EvalTools, ProfTools):
    """Bound to one session: knows the workspace, the budget, the ledger root and the run."""

    def __init__(self, session: Any):
        self.s = session
        self.violation: str | None = None

    # -- plumbing ------------------------------------------------------------------------
    def _cost(self, tool: str) -> dict:
        """Wall seconds since the call began; a GPU tool charges them at the venue rate to the run's budget."""
        seconds = time.monotonic() - getattr(self, "_t0", time.monotonic())
        if tool not in GPU_TOOLS:
            return {"usd": 0.0, "seconds": seconds}
        rate, known = gpu_rate()
        usd = Budget.gpu_usd(seconds, rate)
        try:
            self.s.budget.charge(usd, tool)
        except BudgetExceeded:              # the GPU time is already spent; the next GPU tool call is refused
            self.s.budget.spent_usd += usd
        return {"usd": usd, "seconds": seconds, "gpu_seconds": seconds, "usd_per_hour": rate, "rate_known": known}

    def _record(self, kind: str, tool: str, args: dict, result: dict, snapshot, config: dict | None = None) -> dict:
        body = {"kind": kind, "run": self.s.run_id, "session": self.s.session_id, "tool": tool,
                "args": args, "snapshot": snapshot.id, "snapshot_blob": snapshot.blob,
                "patch": snapshot.patch, "result": result, "cost": self._cost(tool)}
        if config:
            body["config"] = config
        return ledger.append(body, self.s.ledger_root)

    def _record_heldout(self, kind: str, tool: str, args: dict, config: dict, metrics: dict, snapshot) -> dict:
        """A held-out result: top-level config + one aggregate per metric, the only shape the ledger accepts."""
        body = {"kind": kind, "run": self.s.run_id, "session": self.s.session_id, "tool": tool, "args": args,
                "snapshot": snapshot.id, "snapshot_blob": snapshot.blob, "patch": snapshot.patch,
                "config": {**config, "split": "heldout"}, "metrics": metrics, "cost": self._cost(tool)}
        return ledger.append(body, self.s.ledger_root)

    def _audited(self, tool: str, args: dict):
        """Snapshot, and stop the run on a surface violation: integrity failures are the one hard rule."""
        self._t0 = time.monotonic()
        if tool in GPU_TOOLS and self.s.budget.remaining_usd <= 0 and gpu_rate()[0] > 0:
            raise BudgetExceeded(f"{tool}: no budget left for GPU time")
        snap = self.s.workspace.snapshot()
        if snap.violations:
            self.violation = "; ".join(snap.violations)
            self._record("note", tool, args, {"violation": snap.violations}, snap)
            raise PermissionError(f"workspace violates the write surface: {self.violation}")
        return snap

    def _pristine(self) -> Path:
        a = self.s.workspace.audit()
        dest = self.s.run_dir / "pristine"
        grader.pristine_tree(self.s.workspace.repo, self.s.workspace.base, self.s.workspace.path, a, dest)
        return dest

    def _keep(self, src: Path) -> tuple[str, str]:
        """`src` into the ledger as a blob, and a copy in the workspace: the ledger is outside the jail, so the agent
        could not read it there. Returns (blob, workspace-relative copy)."""
        blob = ledger.put_blob(src, self.s.ledger_root)
        visible = Path("lab") / "runs" / Path(blob).name
        dest = self.s.workspace.path / visible
        shutil.rmtree(dest, ignore_errors=True)
        shutil.copytree(src, dest)
        return blob, str(visible)

    # -- tools ---------------------------------------------------------------------------
    def test(self, args: dict) -> str:
        snap = self._audited("test", args)
        tree = self._pristine()
        lint = grader.run_lint(tree)
        tests = grader.run_tests(tree) if lint.passed else grader.Run(False, -1, "skipped: lint failed")
        result = {"lint": lint.passed, "tests": tests.passed, "returncode": tests.returncode,
                  "lint_output": lint.output[-2000:], "test_output": tests.output[-4000:],
                  "changed": snap.files + snap.deleted, "scratch_left_out": self.s.workspace.audit().scratch}
        self._record("test", "test", args, result, snap)
        head = "PASS" if lint.passed and tests.passed else "FAIL"
        return f"{head} lint={'ok' if lint.passed else 'fail'} tests={'ok' if tests.passed else 'fail'}\n" \
               f"{lint.output[-1500:]}\n{tests.output[-3000:]}"

    def profile(self, args: dict) -> str:
        snap = self._audited("profile", args)
        tree = self._pristine()
        out = tree / "lab" / "runs"
        harness = stage_harness(tree)
        n = int(args.get("requests", 8))
        argv = [engine.python(), "-m", "lab.profile", "--requests", str(n),
                "--max-tokens", str(args.get("max_tokens", 32)), "--out", str(out),
                *workload_flags(args, harness, n)]
        t0 = time.monotonic()
        proc = (self.s.profile_runner or default_profile_runner)(tree, argv)
        bundles = sorted(out.glob("*")) if out.exists() else []
        blob, visible = self._keep(bundles[-1]) if bundles else ("", "")
        result = {"returncode": proc.returncode, "bundle": blob, "workspace_copy": visible,
                  "seconds": time.monotonic() - t0, "output": (proc.stdout + proc.stderr)[-3000:]}
        self._record("profile", "profile", args, result, snap)
        if proc.returncode != 0 or not blob:
            return f"profile failed ({proc.returncode}):\n{result['output']}"
        return f"bundle at {visible} (in your workspace; left out of your change)\n" + "\n".join(
            f"  {p.name}" for p in sorted((self.s.workspace.path / visible).iterdir()))

    def ledger_tool(self, args: dict) -> str:
        match = {k: v for k, v in args.items() if k in ("kind", "session", "snapshot", "tool") and v}
        rows = list(ledger.records(self.s.ledger_root, **match))
        last = int(args.get("last") or 20)
        return "\n".join(json.dumps(r) for r in rows[-last:]) or "(no records)"

    def budget(self, args: dict) -> str:
        b = self.s.budget
        return json.dumps({"cap_usd": b.cap_usd, "spent_usd": round(b.spent_usd, 4),
                           "remaining_usd": round(b.remaining_usd, 4)})

    def restore(self, args: dict) -> str:
        sid = args["snapshot"]
        self._audited("restore", args)
        self.s.workspace.restore(sid)
        snap = self._audited("restore", args)
        self._record("note", "restore", args, {"restored": sid, "now": snap.id}, snap)
        return f"workspace restored to {sid}"

    def note(self, args: dict) -> str:
        snap = self._audited("note", args)
        self._record("note", "note", args, {"text": args["text"]}, snap)
        return "noted"

    def specs(self) -> list[ToolSpec]:
        obj = {"type": "object", "properties": {}}
        rate, known = gpu_rate()
        free = " Costs no GPU time ($0)."

        def gpu(t: str) -> str:
            return (f" Costs GPU time: ~{t} per call, charged at ${rate:g}/h." if known else
                    f" Costs GPU time: ~{t} per call; no GPU rate is configured on this host, so it is charged $0.")
        return [
            ToolSpec("test", "Lint and run the test suite on a pristine copy of your current workspace, "
                     "inside the referee's jail. Returns pass/fail and the tail of the output." + free, obj, self.test),
            ToolSpec("profile", "Run the engine under the profiler on requests sampled from a seen corpus class (or "
                     "synthetic token ids) and store the raw bundle (event timeline, chrome trace, op and kernel "
                     "tables, memory, provenance) in the ledger." + gpu("2-5 min (engine start plus the workload)"),
                     {"type": "object", "properties": {"corpus_class": {"type": "string"}, "seed": {"type": "integer"},
                                                       "synthetic": {"type": "boolean"}, "requests": {"type": "integer"},
                                                       "prompt_len": {"type": "integer"},
                                                       "max_tokens": {"type": "integer"}}}, self.profile),
            ToolSpec("ledger", "Read past records: every tool call in this and earlier runs, with snapshots and "
                     "results." + free,
                     {"type": "object", "properties": {"kind": {"type": "string"}, "session": {"type": "string"},
                                                       "snapshot": {"type": "string"}, "tool": {"type": "string"},
                                                       "last": {"type": "integer"}}}, self.ledger_tool),
            ToolSpec("budget", "Dollars left in this run. Every record's `cost` holds the measured seconds and "
                     "dollars of that call." + free, obj, self.budget),
            ToolSpec("restore", "Put the workspace back to a snapshot id from the ledger ('base' resets it)." + free,
                     {"type": "object", "required": ["snapshot"], "properties": {"snapshot": {"type": "string"}}}, self.restore),
            ToolSpec("note", "Leave a note for the human running the lab. It is recorded and changes nothing." + free,
                     {"type": "object", "required": ["text"], "properties": {"text": {"type": "string"}}}, self.note),
            ToolSpec("bench", "Serve your current workspace (pristine copy, jailed) and measure it under the load "
                     "regimes on the seen split. Returns one headline number per regime, raw."
                     + gpu("5-10 min per regime at tier short and 15-25 min at tier full, plus ~2 min engine start"),
                     {"type": "object", "properties": {"regimes": {"type": "array", "items": {"type": "string"}},
                                                       "tier": {"type": "string", "enum": ["short", "full"]}}}, self.bench),
            ToolSpec("equiv", "Serve your current workspace and run the correctness gate against the reference "
                     "model. Returns pass, fail or inconclusive with every metric. tier=full is required before submit."
                     + gpu("10 min at tier dev and 30-60 min at tier full"),
                     {"type": "object", "properties": {"tier": {"type": "string", "enum": ["dev", "full"]}}}, self.equiv),
            ToolSpec("submit", "Measure your current workspace on the held-out split at the full tier and compare it "
                     "with the base. Needs a passing full-tier equiv and a seen bench on this snapshot. The only "
                     "thing that can produce a win. Returns one aggregate per regime."
                     + gpu("15-25 min per regime, plus the base commit measured the same way once per run"),
                     {"type": "object", "properties": {"regimes": {"type": "array", "items": {"type": "string"}}}},
                     self.submit),
            *self.prof_specs(gpu),
        ]


HARNESS = ("__init__.py", "profile.py", "bundle.py", "gpu.py")     # all lab.profile imports from the environment
HARNESS_DIR = "_harness"
DEFAULT_CORPUS_CLASS = "steady_interactive"   # targets name no corpus class of their own


def stage_harness(tree: Path) -> Path:
    """Copy just the profiling harness into the pristine tree: the jail must not open the environment repo (it holds
    the ledger and .env), and the agent's own tree has no `lab` package any more."""
    dest = tree / HARNESS_DIR / "lab"
    dest.mkdir(parents=True, exist_ok=True)
    for name in HARNESS:
        shutil.copyfile(engine.ENV_ROOT / "lab" / name, dest / name)
    return dest.parent


def workload_flags(args: dict, harness: Path, n: int) -> list[str]:
    """lab.profile's workload source: synthetic ids if asked, else `n` seen-split corpus requests sampled out here
    (the jail holds no corpus, and the verified loader needs the held-out split too) and staged in the harness."""
    if args.get("synthetic"):
        return ["--synthetic", "--prompt-len", str(args.get("prompt_len", 64))]
    t = target.load()
    w = corpus.sample(args.get("corpus_class") or DEFAULT_CORPUS_CLASS, "seen", int(args.get("seed", 0)), n, t.corpus_dir)
    (harness / "workload.json").write_text(json.dumps(w))
    return ["--workload", str(harness / "workload.json")] + (["--enable-thinking"] if t.chat_kwargs.get("enable_thinking") else [])


def default_profile_runner(tree: Path, argv: list[str], read: list[Path] = ()):
    """lab.profile in the pristine tree as agent code: jailed, weights and `read` read-only, harness first on the path."""
    path = os.pathsep.join([str(tree / HARNESS_DIR), str(tree / "src"), str(tree)])
    return grader.jailed(tree, argv, timeout_s=3600, env_extra={"PYTHONPATH": path}, read=read)


def clean_pristine(run_dir: Path) -> None:
    shutil.rmtree(run_dir / "pristine", ignore_errors=True)
