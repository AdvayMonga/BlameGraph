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
from lab.evaltools import EvalTools, _killed
from lab.proftools import ProfTools
from lab.safety import grader

# Tools that occupy the GPU: their wall time is charged at the venue's rate. The rest cost $0.
GPU_TOOLS = frozenset({"profile", "trace", "kernel", "hostprof", "bench", "equiv", "submit", "baseline"})


def _drop_links(root: Path) -> None:
    """Remove symlinks the jailed run left in its output: copied outside the jail they would read anything."""
    if root.is_symlink():
        raise PermissionError(f"{root.name}: output is a symlink")
    for p in root.rglob("*"):
        if p.is_symlink():
            p.unlink()


class Toolbox(EvalTools, ProfTools):
    """Bound to one session: knows the workspace, the budget, the ledger root and the run."""

    def __init__(self, session: Any):
        from lab import worker
        self.s = session
        self.violation: str | None = None
        self.worker = getattr(session, "worker", None) or worker.load(
            encoder=getattr(session, "encoder", None), profile_runner=getattr(session, "profile_runner", None))

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
        self._sync("down")
        snap = self.s.workspace.snapshot()
        if snap.violations:
            self.violation = "; ".join(snap.violations)
            self._record("note", tool, args, {"violation": snap.violations}, snap)
            raise PermissionError(f"workspace violates the write surface: {self.violation}")
        return snap

    def _sync(self, way: str) -> None:
        """With the agent on a Remote worker: its workspace there is the live one, this one its mirror."""
        rws = getattr(self.s, "remote_workspace", None)
        if rws:
            (self.worker.pull(rws, self.s.workspace.path) if way == "down" else
             self.worker.push(self.s.workspace.path, rws))

    def _pristine(self) -> Path:
        a = self.s.workspace.audit()
        dest = self.s.run_dir / "pristine"
        grader.pristine_tree(self.s.workspace.repo, self.s.workspace.base, self.s.workspace.path, a, dest)
        return dest

    def _keep(self, src: Path) -> tuple[str, str]:
        """`src` into the ledger as a blob, and a copy in the workspace: the ledger is outside the jail, so the agent
        could not read it there. Returns (blob, workspace-relative copy)."""
        _drop_links(src)
        blob = ledger.put_blob(src, self.s.ledger_root)
        visible = Path("lab") / "runs" / Path(blob).name
        dest = self.s.workspace.path / visible
        shutil.rmtree(dest, ignore_errors=True)
        shutil.copytree(src, dest)
        self._sync("up")
        return blob, str(visible)

    # -- tools ---------------------------------------------------------------------------
    def test(self, args: dict) -> str:
        snap = self._audited("test", args)
        out = self._work("test")
        try:
            r = self.worker.call("test", {"tree": self._pristine()}, {}, out)
        finally:
            shutil.rmtree(out, ignore_errors=True)
        lint, tests = grader.Run(**r["lint"]), grader.Run(**r["tests"])
        result = {"lint": lint.passed, "tests": tests.passed, "returncode": tests.returncode,
                  "lint_output": lint.output[-2000:], "test_output": tests.output[-4000:],
                  "changed": snap.files + snap.deleted, "scratch_left_out": self.s.workspace.audit().scratch}
        self._record("test", "test", args, result, snap)
        head = "PASS" if lint.passed and tests.passed else "FAIL"
        return f"{head} lint={'ok' if lint.passed else 'fail'} tests={'ok' if tests.passed else 'fail'}\n" \
               f"{lint.output[-1500:]}\n{tests.output[-3000:]}"

    def profile(self, args: dict) -> str:
        snap = self._audited("profile", args)
        out = self._work("profile")
        t0 = time.monotonic()
        try:
            r = self.worker.call("profile", {"tree": self._pristine()},
                                 {"kind": "profile", "args": args, "workbench": getattr(self.s, "workbench", None)}, out)
            blob, visible = self._keep(out / "files") if (out / "files").is_dir() else ("", "")
        finally:
            shutil.rmtree(out, ignore_errors=True)
        result = {"returncode": r["returncode"], "bundle": blob, "workspace_copy": visible,
                  "seconds": time.monotonic() - t0, "output": r["output"], **_killed(r)}
        self._record("profile", "profile", args, result, snap)
        killed = f"\nkilled in your workbench (held the GPU): {json.dumps(r['killed'])}" if r.get("killed") else ""
        if r["returncode"] != 0 or not blob:
            return f"profile failed ({r['returncode']}):\n{result['output']}{killed}"
        return f"bundle at {visible} (in your workspace; left out of your change)\n" + "\n".join(
            f"  {p.name}" for p in sorted((self.s.workspace.path / visible).iterdir())) + killed

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
        self._sync("up")
        snap = self._audited("restore", args)
        self._record("note", "restore", args, {"restored": sid, "now": snap.id}, snap)
        return f"workspace restored to {sid}"

    def note(self, args: dict) -> str:
        snap = self._audited("note", args)
        self._record("note", "note", args, {"text": args["text"]}, snap)
        return "noted"

    def specs(self) -> list[ToolSpec]:
        """What each tool does, returns and costs; never when or how to use it."""
        from regimes import suite
        from lab.evaltools import HOLDOUT_BUDGET
        obj = {"type": "object", "properties": {}}
        rate, known = gpu_rate()
        free = " Costs no GPU time ($0)."

        from lab.agent import ClaudeAgentSDK
        paused = (" While it runs, the container your shell runs in is paused, and processes in it holding the GPU are "
                  "killed and listed in the result." if ClaudeAgentSDK.containerized() else "")

        def gpu(t: str) -> str:
            return (f" Costs GPU time: ~{t} per call, charged at ${rate:g}/h." if known else
                    f" Costs GPU time: ~{t} per call; no GPU rate is configured on this host, so it is charged $0.") + paused
        short, full = suite.TIERS["short"], suite.TIERS["full"]
        windows = (f"one engine start plus, per regime, a {short['final_s']:.0f} s measurement window at tier short or "
                   f"{full['final_s']:.0f} s at full, plus {short['probe_s']:.0f} s / {full['probe_s']:.0f} s probes for "
                   f"regimes that search for a load")
        return [
            ToolSpec("test", "Lint and run the test suite on a pristine copy of your current workspace, inside the "
                     "referee's jail. Returns PASS or FAIL, lint and test status, and the tail of each output." + free,
                     obj, self.test),
            ToolSpec("profile", "Run the engine in process under the profiler, jailed, on `requests` sampled from a seen "
                     "corpus class (`corpus_class`, default steady_interactive; `seed`) or on synthetic token ids "
                     "(`synthetic`, `prompt_len`); `max_tokens` default 32. Stores the raw bundle (event timeline, chrome "
                     "trace, op and kernel tables, memory, GPU samples, provenance) in the ledger and copies it into your "
                     "workspace under lab/runs/, left out of your change. Returns the bundle path and its file names."
                     + gpu("1-7 min measured at 16 requests x 64 tokens on an H200 (7 min when the weights were still loading from disk)"),
                     {"type": "object", "properties": {"corpus_class": {"type": "string"}, "seed": {"type": "integer"},
                                                       "synthetic": {"type": "boolean"}, "requests": {"type": "integer"},
                                                       "prompt_len": {"type": "integer"},
                                                       "max_tokens": {"type": "integer"}}}, self.profile),
            ToolSpec("ledger", "Read records of this and earlier runs: every tool call, session and baseline, with "
                     "snapshots and results. Filters: `kind`, `session`, `snapshot`, `tool`. Returns the `last` N "
                     "matches (default 20) as JSON lines." + free,
                     {"type": "object", "properties": {"kind": {"type": "string"}, "session": {"type": "string"},
                                                       "snapshot": {"type": "string"}, "tool": {"type": "string"},
                                                       "last": {"type": "integer"}}}, self.ledger_tool),
            ToolSpec("budget", "Returns the run's dollar cap, spent and remaining. Every record's `cost` holds the "
                     "measured seconds and dollars of that call." + free, obj, self.budget),
            ToolSpec("restore", "Put the workspace back to a snapshot id from the ledger ('base' resets it to the base "
                     "commit). Returns the snapshot restored." + free,
                     {"type": "object", "required": ["snapshot"], "properties": {"snapshot": {"type": "string"}}}, self.restore),
            ToolSpec("note", "Leave a note for the human running the lab. It is recorded and changes nothing." + free,
                     {"type": "object", "required": ["text"], "properties": {"text": {"type": "string"}}}, self.note),
            ToolSpec("bench", "Serve a pristine copy of your current workspace (jailed) and measure it under load "
                     f"regimes on the seen split. `regimes`: default {_names(self.bench_defaults())}; 'all'; or "
                     f"any of {_names(suite.REGIMES)}. `tier`: short (default) or full. `seed`: the seen-split workload sample "
                     f"(default {self.seed}). Returns one headline per regime "
                     f"(objective, value, direction, validity), raw. Takes {windows}." + gpu("5-25 min per regime"),
                     {"type": "object", "properties": {"regimes": {"type": "array", "items": {"type": "string"}},
                                                       "tier": {"type": "string", "enum": ["short", "full"]},
                                                       "seed": {"type": "integer"}}}, self.bench),
            ToolSpec("equiv", "Serve your current workspace and run the correctness gate against the reference "
                     "model's outputs at `tier` dev (default) or full. Returns pass, fail or inconclusive with every "
                     "metric. submit requires a passing full-tier equiv on the snapshot. Takes one engine start plus "
                     "generating the tier's task set." + gpu("10 min at tier dev and 30-60 min at tier full"),
                     {"type": "object", "properties": {"tier": {"type": "string", "enum": ["dev", "full"]}}}, self.equiv),
            ToolSpec("submit", "Measure your current workspace on the held-out split at the full tier and compare it "
                     "with the base commit measured the same way. Needs a passing full-tier equiv and a completed "
                     "short-tier seen bench on this snapshot. `regimes`: default those of that bench. The only thing "
                     "that can produce a win. Returns one aggregate per regime (base, new, delta %, noise band %, "
                     "verdict) and, with a task, its score. Takes one engine start plus a held-out full-tier "
                     f"measurement per regime, the base commit's held-out measurement once per run, and up to one of the "
                     f"{HOLDOUT_BUDGET} held-out queries per regime." + gpu("15-25 min per regime"),
                     {"type": "object", "properties": {"regimes": {"type": "array", "items": {"type": "string"}}}},
                     self.submit),
            *self.prof_specs(gpu),
        ]


def _names(names) -> str:
    return ", ".join(f"`{n}`" for n in names)


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
    """lab.profile in the pristine tree as agent code: jailed, with the target's engine env, weights and `read`
    read-only, harness first on the path."""
    path = os.pathsep.join([str(tree / HARNESS_DIR), str(tree / "src"), str(tree)])
    return grader.jailed(tree, argv, timeout_s=3600, env_extra={**target.load().engine.env, "PYTHONPATH": path},
                         read=read)


def clean_pristine(run_dir: Path) -> None:
    shutil.rmtree(run_dir / "pristine", ignore_errors=True)
