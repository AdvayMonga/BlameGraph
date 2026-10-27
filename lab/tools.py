"""The agent's metered tools. Each runs outside the jail, snapshots the workspace, and writes the ledger whether the agent likes it or not."""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from lab import calibrate, corpus, engine, experiments, layers, ledger, profiles, target
from lab.agent import ToolSpec
from lab.budget import Budget, BudgetExceeded, gpu_rate
from lab.evaltools import EvalTools, _killed
from lab.proftools import PROFILE_ARG, ProfTools
from lab.safety import grader

# Tools that occupy the GPU: their wall time is charged at the venue's rate. The rest cost $0.
QUERY_MAX_BYTES = 64_000           # a query's reply: whole rows up to this, so one call cannot flood the context
AFTER_BENCH_TIMEOUT_S = 900          # the after-bench profile: 1-7 min measured; a hang costs at most this
GPU_TOOLS = frozenset({"profile", "trace", "kernel", "hostprof", "bench", "equiv", "submit", "baseline",
                       "kernel_search"})
KERNEL_SEARCH_GPU_MIN = 3.0          # GPU minutes per judged program, for sizing a search to its budget (an estimate)
KERNEL_SEARCH_MIN_EVALS = 8          # fewer judged programs than this is not a search
KERNEL_SEARCH_ROUND_USD = 15.0       # model spend of generation 0 and one more (a 40-idea round measured $9-14 on
                                     # 2026-10-24/25, a 24-idea generation ~$5; a $6 share ended a round before any
                                     # program was timed); further generations run while the budget lasts


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
        self._experiment(body, tool, args, result=result)
        return ledger.append(body, self.s.ledger_root)

    def _record_heldout(self, kind: str, tool: str, args: dict, config: dict, metrics: dict, snapshot) -> dict:
        """A held-out result: top-level config + one aggregate per metric, the only shape the ledger accepts."""
        body = {"kind": kind, "run": self.s.run_id, "session": self.s.session_id, "tool": tool, "args": args,
                "snapshot": snapshot.id, "snapshot_blob": snapshot.blob, "patch": snapshot.patch,
                "config": {**config, "split": "heldout"}, "metrics": metrics, "cost": self._cost(tool)}
        self._experiment(body, tool, args, metrics=metrics)
        return ledger.append(body, self.s.ledger_root)

    def _experiment(self, body: dict, tool: str, args: dict, result: dict | None = None, metrics: dict | None = None):
        """A measured call made with a hypothesis and a prediction carries both and the computed gap."""
        if tool not in experiments.MEASURED or "expect" not in args or "hypothesis_id" not in args:
            return
        if body["kind"] == "note" or (result or {}).get("verdict") == "refused":
            return                                   # nothing was measured, so there is nothing to reflect on
        from lab.evaltools import noise_band_pct
        base = next(ledger.records(self.s.ledger_root, run=self.s.run_id, kind="baseline"), None)
        name = args["expect"].get("metric")
        band = (noise_band_pct(name, target.load(), self.s.ledger_root)
                if tool in ("bench", "submit") and isinstance(name, str) else None)
        body["experiment"] = experiments.outcome(tool, args, result, metrics,
                                                 ((base or {}).get("result") or {}).get("metrics"), band,
                                                 (base or {}).get("result"))

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
        tree = self._pristine()
        why = profiles.refusal(tree, target.load(), args.get("profile"))
        if why:
            self._record("note", "profile", args, {"verdict": "refused", "reason": why}, snap)
            return f"profile refused: {why}"
        t0 = time.monotonic()
        r, blob, visible, cal = self._profile_job(tree, args)
        result = {"returncode": r["returncode"], "bundle": blob, "workspace_copy": visible,
                  "seconds": time.monotonic() - t0, "output": r["output"], "calibrated": cal, **_killed(r)}
        self._record("profile", "profile", args, result, snap)
        killed = f"\nkilled in your workbench (held the GPU): {json.dumps(r['killed'])}" if r.get("killed") else ""
        if r["returncode"] != 0 or not blob:
            return f"profile failed ({r['returncode']}):\n{result['output']}{killed}"
        return f"bundle at {visible} (in your workspace; left out of your change)\n" + "\n".join(
            f"  {p.name}" for p in sorted((self.s.workspace.path / visible).iterdir())) + killed

    def _profile_job(self, tree: Path, args: dict, timeout_s: int | None = None) -> tuple[dict, str, str, dict | None]:
        """One profile job on `tree`: the worker's reply, the bundle's blob and workspace copy, and its calibration
        with each layer's facts."""
        out = self._work("profile")
        job = {"kind": "profile", "args": args, "workbench": getattr(self.s, "workbench", None)}
        try:
            r = self.worker.call("profile", {"tree": tree}, {**job, **({"timeout_s": timeout_s} if timeout_s else {})},
                                 out)
            blob, visible = self._keep(out / "files") if (out / "files").is_dir() else ("", "")
        finally:
            shutil.rmtree(out, ignore_errors=True)
        try:                                     # the blob is the bundle itself (worker copies it to `files`)
            cal = calibrate.calibrate_kernels(self.s.ledger_root / blob, r.get("gpu")) if blob else None
            if cal is not None:                  # each layer's share of the window and its distance from a ceiling
                cal["layers"] = layers.from_bundle(self.s.ledger_root / blob, target.load(), r.get("gpu"))
                cal["kernel_tasks"] = task_shares(self.s.ledger_root / blob)
        except Exception as e:                   # noqa: BLE001  never costs the measurement it describes
            cal = {"error": f"calibration failed: {type(e).__name__}: {e}"}
        return r, blob, visible, cal

    def _after_bench(self, tree: Path) -> dict:
        """The profile a completed bench takes of the same tree right after its measurement: its own launch, the
        general config and `profile`'s default workload (8 requests x 32 tokens), so its layer facts describe that load,
        not the bench's. The layer facts a mechanism is checked against. Never costs the bench."""
        rate = gpu_rate()[0]
        if rate > 0 and self.s.budget.remaining_usd <= 0:
            return {"skipped": "no budget left for GPU time"}
        try:
            r, blob, visible, cal = self._profile_job(tree, {}, timeout_s=AFTER_BENCH_TIMEOUT_S)
        except Exception as e:                   # noqa: BLE001
            return {"error": f"{type(e).__name__}: {e}"}
        out = {"launch": "general", "workload": "profile defaults", "returncode": r["returncode"], "bundle": blob,
               "workspace_copy": visible, "calibrated": cal, **_killed(r)}
        if r.get("gpu_others"):                  # another process shared the GPU: its numbers are not evidence
            out.update(contaminated=True, gpu_others=r["gpu_others"])
        return out if r["returncode"] == 0 and blob else {**out, "output": r["output"][-2000:]}


    def kernel_search(self, args: dict) -> str:
        """The kernel layer in one blocking call (kernels/pipeline.py): capture the task's calls under real traffic,
        search, gate the winner on the held-out shapes; the winner comes back as a patch of the engine module, saved
        in the workspace (left out of your change) for you to apply."""
        from kernels import pipeline
        from kernels.tasks import TASKS, load
        if str(args.get("task")).startswith("region:"):    # a region's entry is the stations tuple, not a function
            raise ValueError("`task`: regions (region:FIRST:LAST) are not wired into kernel_search yet")
        if args.get("task") not in TASKS:
            raise ValueError(f"`task` must be one of {list(TASKS)}")
        usd = args.get("budget_usd")
        if isinstance(usd, bool) or not isinstance(usd, (int, float)) or usd <= 0:
            raise ValueError("`budget_usd` must be a positive number")
        if usd > self.s.budget.remaining_usd:
            raise ValueError(f"`budget_usd` {usd} is more than the run has left (${self.s.budget.remaining_usd:.2f})")
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise ValueError("kernel_search needs ANTHROPIC_API_KEY on this host (its search calls the models)")
        rate = gpu_rate()[0]
        model_usd = max(usd / 2, KERNEL_SEARCH_ROUND_USD) if rate else usd    # the models get two generations at least
        evals = int((usd - model_usd) / (rate * KERNEL_SEARCH_GPU_MIN / 60)) if rate else 30
        if evals < KERNEL_SEARCH_MIN_EVALS:
            least = KERNEL_SEARCH_ROUND_USD + KERNEL_SEARCH_MIN_EVALS * rate * KERNEL_SEARCH_GPU_MIN / 60
            raise ValueError(f"`budget_usd` {usd} leaves ${usd - model_usd:.2f} of GPU time after two generations of model calls "
                             f"(${KERNEL_SEARCH_ROUND_USD:g}): {evals} judged programs at ${rate:g}/GPU-hour; a search needs "
                             f"at least {KERNEL_SEARCH_MIN_EVALS} (~${least:.2f} in all)")
        evals = min(60, evals)
        task = load(args["task"])
        snap = self._audited("kernel_search", args)
        tree = self._pristine()
        out = self.s.run_dir / "kernel_search" / f"{snap.id}-{task.name}-{int(time.time())}"
        out.mkdir(parents=True, exist_ok=True)

        def profile(t, a):
            r, blob, _, _ = self._profile_job(t, a)
            return self.s.ledger_root / blob if r["returncode"] == 0 and blob else None
        from lab import worker
        w = self.worker
        remote = isinstance(w, worker.Remote)
        env = pipeline.remote_env(w.prefix, w.env_dir, w.python, w.env, w.sudo) if remote else {}
        import torch
        device = "cuda" if remote or torch.cuda.is_available() else "cpu"     # judged where the GPU is
        try:
            scored = (*getattr(self.s.task, "regimes", ()), *getattr(self.s.task, "constraints", ())) \
                if getattr(self.s, "task", None) else ()
            caps = pipeline.captures(task, tree, profile, scored, self.s.run_dir / "kernel_search" / "captures.json")
            if not caps:
                raise RuntimeError("no capture traffic produced calls to the task's entry point")
            start = out / "start.py"
            start.write_text(pipeline.start_program(task, tree))
            from lab.evaltools import _layer_facts
            found = pipeline.search(task, caps, start, out / "search", model_usd, evals, self.s.ledger_root, env,
                                    device=device,
                                    facts_text=pipeline.facts(task, _layer_facts(self.s.ledger_root, "kernel_tasks"),
                                                              scored))
            best = out / "search" / "best_program.py"
            g = pipeline.gate(task, caps, best, out, env)
        except Exception as e:                   # noqa: BLE001  a tool never crashes the session
            model_cost = self._charge_models(pipeline.spent(out / "search"))
            result = {"verdict": "error", "reason": f"{type(e).__name__}: {e}"[-2000:], "model_usd": model_cost}
            self._record("kernels", "kernel_search", args, result, snap)
            return f"kernel_search failed: {result['reason']}"
        from kernels.evolve.ship import gate_passed
        passed = gate_passed(g)
        rel = None
        if passed:
            text = pipeline.patch(task, tree, best.read_text())
            dest = self.s.workspace.path / "lab" / "runs" / "kernel_search" / f"{task.name}-{snap.id}.patch"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(text)
            rel = dest.relative_to(self.s.workspace.path).as_posix()
        model_cost = self._charge_models(float((found.get("cost") or {}).get("usd") or 0.0))
        gm = g.get("metrics") or {}
        result = {"verdict": "ok", "metrics": {"speedup": (found.get("best") or {}).get("combined_score"),
                                               "heldout_speedup": gm.get("heldout_speedup")},
                  "best": found.get("best"), "search_record": found.get("ledger"), "evals": evals,
                  "model_usd": model_cost, "gate": {"passed": passed, "metrics": gm,
                                                    "gates": json.loads((g.get("artifacts") or {}).get("gates") or "{}"),
                                                    **({"error": g["artifacts"]["error"][-1500:]}
                                                       if (g.get("artifacts") or {}).get("error") else {})},
                  "patch": rel, "captures": len(caps)}
        self._record("kernels", "kernel_search", args, result, snap)
        return json.dumps({k: result[k] for k in ("metrics", "gate", "patch", "search_record", "model_usd")}, indent=1,
                          default=str)

    def _charge_models(self, usd: float) -> float:
        """Model dollars charged to the run; spent already, so an exhausted budget records them and refuses what's next."""
        try:
            self.s.budget.charge(usd, "kernel_search models")
        except BudgetExceeded:
            self.s.budget.spent_usd += usd
        return usd

    def ledger_tool(self, args: dict) -> str:
        match = {k: v for k, v in args.items() if k in ("kind", "session", "snapshot", "tool") and v}
        rows = [r for r in ledger.records(self.s.ledger_root, **match)
                if experiments.researcher(self.s.run_id) or not experiments.researcher(r.get("run"))]
        last = int(args.get("last") or 20)
        return "\n".join(json.dumps(r) for r in rows[-last:]) or "(no records)"

    def query(self, args: dict) -> str:
        from store import index
        try:
            out = index.query(index.agent_db(self.s.ledger_root), args["sql"])
        except Exception as e:                      # a bad query is the agent's to read, never a crash
            return f"query failed: {type(e).__name__}: {e}"[:2000]
        text = json.dumps(out, default=str)
        while len(text) > QUERY_MAX_BYTES and out["rows"]:      # whole rows only, so the reply stays valid JSON
            out["rows"] = out["rows"][:len(out["rows"]) // 2]
            out["truncated"] = True
            text = json.dumps({**out, "truncated_at_bytes": QUERY_MAX_BYTES}, default=str)
        return text

    def search(self, args: dict) -> str:
        from lab import search
        if not hasattr(self, "_embedder"):         # loaded once per toolbox; a missing model is a fact in each reply
            try:
                self._embedder, self._no_embedder = search.FastEmbed(self.s.ledger_root / "index" / "models"), None
            except Exception as e:                  # noqa: BLE001
                self._embedder, self._no_embedder = None, f"{type(e).__name__}: {e}"[:500]
        out = search.search(self.s.ledger_root, str(args.get("query") or ""), args.get("k") or 10, self._embedder,
                            researcher=experiments.researcher(self.s.run_id))
        if self._no_embedder:
            out["embeddings_unavailable"] = self._no_embedder
        text = json.dumps(out, default=str)
        while len(text) > QUERY_MAX_BYTES and out["results"]:     # whole results only, like `query`
            out["results"] = out["results"][:len(out["results"]) // 2]
            out["truncated"] = True
            text = json.dumps(out, default=str)
        return text

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

    def hypothesis(self, args: dict) -> str:
        experiments.open_batch(self.s.ledger_root, self.s.run_id)
        fields = experiments.new_hypothesis(args)
        rec = ledger.append({"kind": "hypothesis", "run": self.s.run_id, "session": self.s.session_id, **fields},
                            self.s.ledger_root)
        return f"hypothesis {rec['id']}"

    def reflect(self, args: dict) -> str:
        fields = experiments.new_reflection(self.s.ledger_root, self.s.run_id, args)
        rec = ledger.append({"kind": "reflection", "run": self.s.run_id, "session": self.s.session_id, **fields},
                            self.s.ledger_root)
        return f"reflection {rec['id']} on {fields['record']}"

    def _measured(self, spec: ToolSpec) -> ToolSpec:
        """A measured tool: refused without an open hypothesis and a prediction, or while the last result lacks a
        reflection; its reply ends with the prediction, the measurement and the gap."""
        def call(args: dict) -> str:
            experiments.check(self.s.ledger_root, self.s.run_id, spec.name, args)
            before = {r["id"] for r in experiments.measured(self.s.ledger_root, self.s.run_id)}
            text = spec.fn(args)
            last = next((r for r in reversed(experiments.measured(self.s.ledger_root, self.s.run_id))
                         if r["id"] not in before), None)          # this call's record, if it measured anything
            if last is None or last["experiment"].get("hypothesis") != args.get("hypothesis_id"):
                return text
            x = last["experiment"]
            measured = x.get("actual") if "actual_missing" not in x else f"not computed ({x['actual_missing']})"
            how = "".join(f"; {p['metric']} expected {p['value']} ({p['unit']}), measured "
                          + (f"{p['actual']}, gap {p['gap']:+.4g}" if "actual_missing" not in p
                             else f"not computed ({p['actual_missing']})") for p in x.get("mechanism") or [])
            return (f"{text}\n\nExperiment record {last['id']}: expected {x['expect']['metric']} = "
                    f"{x['expect']['value']} ({x['expect']['unit']}), measured {measured}"
                    + (f", gap {x['gap']:+.4g}" if x.get("gap") is not None else "") + how
                    + f". Further measurements of {', '.join(sorted(experiments.involved(last)))} and any new "
                    f"hypothesis are refused until `reflect` records a reflection on {last['id']}.")
        schema = {**spec.schema, "properties": {**spec.schema.get("properties", {}), **experiments.MEASURED_PROPS},
                  "required": sorted(set(spec.schema.get("required", [])) | {"hypothesis_id", "expect"})}
        note = (" Requires `hypothesis_id` (a `hypothesis` record of this run) and `expect` (metric, value, unit, "
                "why; optional `mechanism`: the same for other numbers in the result, each by its dotted path); the "
                "record keeps them with each measured value and gap. Optional `bundle` (further hypothesis ids "
                "measured in the same call) with `independence` (why the changes do not affect each other). Refused "
                "while an earlier measurement of one of these hypotheses has no reflection.")
        return ToolSpec(spec.name, spec.description + note, schema, call)

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

        deps = target.load().deps
        offline = (f" Only your change reaches the engine here: if it changes {' or '.join(deps)}, those are installed "
                   "from scratch for it (package indexes only), and the engine then runs with no network, so anything it "
                   "needs at runtime (draft models, prebuilt kernels, files a library downloads when it is used) has to be "
                   "in your change." if deps else " The engine runs with no network.")

        def gpu(t: str) -> str:
            return (f" Costs GPU time: ~{t} per call, charged at ${rate:g}/h." if known else
                    f" Costs GPU time: ~{t} per call; no GPU rate is configured on this host, so it is charged $0."
                    ) + offline + paused
        short, full = suite.TIERS["short"], suite.TIERS["full"]
        windows = (f"one engine start plus, per regime, a {short['final_s']:.0f} s measurement window at tier short or "
                   f"{full['final_s']:.0f} s at full, plus {short['probe_s']:.0f} s / {full['probe_s']:.0f} s probes for "
                   f"regimes that search for a load")
        tg = target.load()
        prof = (f" Launch profiles: the engine's {tg.profiles} holds [general] and optional tables named after regimes. "
                f"Every regime is measured on the general launch; a regime whose table is non-empty is also measured "
                f"in its own engine start with {tg.profile_env}=<regime>, and that number is its headline (the general "
                f"one is reported beside it). The harness sets only {tg.profile_env}; what a table does is the engine's "
                f"code (as shipped, it writes the table over the launch env, including the target's settings)."
                if tg.profiles else "")
        specs = [
            ToolSpec("test", "Lint and run the test suite on a pristine copy of your current workspace, inside the "
                     "referee's jail. Returns PASS or FAIL, lint and test status, and the tail of each output." + offline
                     + free,
                     obj, self.test),
            ToolSpec("profile", "Run the engine in process under the profiler, jailed, on `requests` sampled from a seen "
                     "corpus class (`corpus_class`, default steady_interactive; `seed`) or on synthetic token ids "
                     "(`synthetic`, `prompt_len`); `max_tokens` default 32. Stores the raw bundle (event timeline, chrome "
                     "trace, op and kernel tables, memory, GPU samples, provenance) in the ledger and copies it into your "
                     "workspace under lab/runs/, left out of your change. `capture`: \"module:function\" entry points whose every "
                     "call's argument shapes (and one whole call per shape) go into the bundle's captures/; it "
                     "synchronizes, so that bundle's times are not representative. Returns the bundle path and its file names."
                     + (PROFILE_ARG if tg.profiles else "")
                     + gpu("1-7 min measured at 16 requests x 64 tokens on an H200 (7 min when the weights were still loading from disk)"),
                     {"type": "object", "properties": {"capture": {"type": "array", "items": {"type": "string"}},
                                                       "corpus_class": {"type": "string"}, "seed": {"type": "integer"},
                                                       "synthetic": {"type": "boolean"}, "requests": {"type": "integer"},
                                                       "prompt_len": {"type": "integer"},
                                                       "max_tokens": {"type": "integer"},
                                                       **({"profile": {"type": "string"}} if tg.profiles else {})}},
                     self.profile),
            ToolSpec("ledger", "Read records of this and earlier runs: every tool call, session and baseline, with "
                     "snapshots and results. Filters: `kind`, `session`, `snapshot`, `tool`. Returns the `last` N "
                     "matches (default 20) as JSON lines." + free,
                     {"type": "object", "properties": {"kind": {"type": "string"}, "session": {"type": "string"},
                                                       "snapshot": {"type": "string"}, "tool": {"type": "string"},
                                                       "last": {"type": "integer"}}}, self.ledger_tool),
            ToolSpec("query", "Run one read-only DuckDB SQL query over every run's data. Tables: records (every ledger "
                     "record; args, config, result, metrics, claim as JSON), regimes (each regime result of a bench or "
                     "baseline), heldout_aggregates (each regime of a submit), noise (run-to-run band per regime), "
                     "client_rows (per-request client rows of every bench and equiv, record_id = records.id), "
                     "engine_rows (the engine's per-request telemetry, trace_id = 'bg-' || client_rows.id), "
                     "device_samples (GPU samples over each served window), kernels and ops (profile bundles). "
                     "`DESCRIBE <table>` lists a table's columns. Returns columns and up to 200 rows as JSON, cut to whole "
                     f"rows under {QUERY_MAX_BYTES // 1000} kB (`truncated` says when rows were dropped)." + free,
                     {"type": "object", "required": ["sql"], "properties": {"sql": {"type": "string"}}}, self.query),
            ToolSpec("search", "Search the experiments of every run on this ledger by text. One document per "
                     "`hypothesis` record: its hypothesis, rationale and abandon_if, and each measurement made under it "
                     "(record id, tool, snapshot, config, GPU, `expect`, measured value, gap, reflection, alternative), "
                     "all as stored. `query` (text), `k` results (default 10, at most 50). Ranked by reciprocal-rank "
                     "fusion of BM25 over the words and cosine similarity of local embeddings (model named in the "
                     "reply). Returns JSON: each result's rank under each order and the document; cut to whole results "
                     f"under {QUERY_MAX_BYTES // 1000} kB." + free,
                     {"type": "object", "required": ["query"], "properties": {"query": {"type": "string"},
                                                                             "k": {"type": "integer"}}}, self.search),
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
                     f"(objective, value, direction, validity), raw. Takes {windows}. At the default seed and tier short, each "
                     "search starts where the run's base measurement of that regime ended; every search stops once its "
                     "bracket is within the regime's noise band (10% where none is measured). Then profiles the same tree once "
                     "with `profile`'s defaults, in its own launch, and keeps it under the result's `profiled` (bundle, "
                     "calibrated layer facts); a failed profile is recorded there and never fails the bench."
                     + prof + gpu("5-25 min per regime, plus the profile's 1-7 min"),
                     {"type": "object", "properties": {"regimes": {"type": "array", "items": {"type": "string"}},
                                                       "tier": {"type": "string", "enum": ["short", "full"]},
                                                       "seed": {"type": "integer"}}}, self.bench),
            ToolSpec("equiv", "Serve your current workspace and run the correctness gate against the reference "
                     "model's outputs at `tier` dev (default) or full. Returns pass, fail or inconclusive with every "
                     "metric. Items are answered in one fixed order and the gate stops at 200, 400, 800, ... items once "
                     "the pooled score ratio's CI is above the line or the net loss is significant (Bonferroni over the "
                     "looks), else after the whole tier; `metrics.sample` says where. `tier` quick generates nothing: the engine scores the dev reference's own answers "
                     "teacher-forced (its sequences.jsonl, a few hundred) and returns the divergence (KL mean, p99 and "
                     "worst sequence, top-1 agreement, reference-token logprob shift) as facts, with no verdict; "
                     "measured cost not yet known. "
                     f"submit requires a passing {tg.submit_tier}-tier equiv on the snapshot"
                     + (", and one for each regime it measures under its own profile (`profile`: that regime's name; "
                        "default the general launch)" if tg.profiles else "") + ". Takes one engine start plus "
                     "generating the tier's task set, 32 requests in flight." + gpu("10 min at tier dev on unmodified vLLM and ~95 min on the example engine (about 17 items/min, measured at 16 requests in flight, H200); the full tier is about 12x the items"),
                     {"type": "object", "properties": {"tier": {"type": "string", "enum": ["quick", "dev", "full"]},
                                                       **({"profile": {"type": "string"}} if tg.profiles else {})}},
                     self.equiv),
            ToolSpec("submit", "Measure your current workspace on the held-out split at the short tier and compare it "
                     f"with the base commit measured the same way. Needs a passing {tg.submit_tier}-tier equiv and a completed "
                     "short-tier seen bench on this snapshot. `regimes`: default those of that bench. The only thing "
                     "that can produce a win. Returns one aggregate per regime (base, new, delta %, noise band % (measured on "
                     "the base engine's general config), "
                     "verdict, and % of the ceiling computed from the base's seen-split request shape) and, with a task, its score. Takes one engine start plus a held-out short-tier "
                     f"measurement per regime, the base commit's held-out measurement once per run, and up to one of the "
                     f"{HOLDOUT_BUDGET} held-out queries per regime." + prof + gpu("5-25 min per regime"),
                     {"type": "object", "properties": {"regimes": {"type": "array", "items": {"type": "string"}}}},
                     self.submit),
            *self.prof_specs(gpu),
            ToolSpec("kernel_search", "Search for a faster version of one engine kernel: `task` (one of the kernel "
                     "tasks, the engine entry point it replaces), `budget_usd` (model calls plus GPU time, charged to "
                     "the run). Captures the task's calls under real seen-split traffic at several concurrencies, splits "
                     "the shapes seen / held-out, searches from your workspace's version of the kernel (kernels.search), "
                     "and gates the winner on the held-out shapes. Returns the winner's seen-shape speedup, the gate's "
                     "verdict and held-out speedup, the `kernels` ledger record of the search, and, if the gate passed, "
                     "the path of a patch of the engine module in your workspace (not applied). Blocks until done. "
                     f"`budget_usd` covers two generations of the search's model calls (${KERNEL_SEARCH_ROUND_USD:g}; "
                     f"later generations run while the model half lasts) plus GPU time at {KERNEL_SEARCH_GPU_MIN:g} min "
                     f"per judged program, {KERNEL_SEARCH_MIN_EVALS} programs at least."
                     + gpu("hours, sized to `budget_usd`"),
                     {"type": "object", "required": ["task", "budget_usd"],
                      "properties": {"task": {"type": "string"}, "budget_usd": {"type": "number"}}},
                     self.kernel_search),
            ToolSpec("hypothesis", "Record a hypothesis: `hypothesis` (the bottleneck and the proposed change), "
                     f"`rationale` (why), `layer` (where the change is: {', '.join(experiments.LAYERS)}), optional "
                     "`abandon_if` (the result that would end it) and `from_record` (the measured record whose "
                     "reflection proposed it). Returns its id, which measured tools take as `hypothesis_id`. Several "
                     "may be open at once; a new one is refused while any measurement of the run has no reflection."
                     + free,
                     {"type": "object", "required": ["hypothesis", "rationale", "layer"],
                      "properties": {"hypothesis": {"type": "string"}, "rationale": {"type": "string"},
                                     "layer": {"type": "string", "enum": list(experiments.LAYERS)},
                                     "abandon_if": {"type": "string"}, "from_record": {"type": "string"}}},
                     self.hypothesis),
            ToolSpec("reflect", "Record a reflection on a measured record: `record` (its id), `reflection` (why the "
                     "result matched or differed from its `expect`), optional `alternative` (another hypothesis). "
                     "A measurement needs one before its hypotheses are measured again, and every measurement of the "
                     "run before a new hypothesis." + free,
                     {"type": "object", "required": ["record", "reflection"],
                      "properties": {"record": {"type": "string"}, "reflection": {"type": "string"},
                                     "alternative": {"type": "string"}}}, self.reflect),
        ]
        return [self._measured(s) if s.name in experiments.MEASURED else s for s in specs]


def _names(names) -> str:
    return ", ".join(f"`{n}`" for n in names)


HARNESS = ("__init__.py", "profile.py", "bundle.py", "gpu.py", "capture.py")     # all lab.profile imports from the environment
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


def task_shares(bundle: Path) -> dict:
    """Each kernel task's share of the profiled window, the kernels it launches and the speedup bound if they took no
    time (kernels/share.py), plus the kernels no task covers; or why there is none."""
    try:
        from kernels import share
        from kernels.tasks import TASKS, load
        tasks = [load(n) for n in TASKS]
    except ImportError as e:                    # the kernel tasks import torch, which the controller may lack
        return {"missing": f"the kernel tasks do not import here: {e}"}
    out = share.shares(bundle, tasks)
    if not out:
        return {"missing": "no kernel table in the bundle"}
    for t in tasks:                              # tasks matching the same device kernels show the same share
        same = sorted(o.name for o in tasks if o is not t and set(o.kernel_names) == set(t.kernel_names) and t.kernel_names)
        if same:
            out["tasks"][t.name]["same_kernels_as"] = same
    return out


def default_profile_runner(tree: Path, argv: list[str], read: list[Path] = (), env: dict | None = None,
                           timeout_s: int = 3600):
    """lab.profile in the pristine tree as agent code: jailed, with the target's engine env (and `env`: a launch
    profile's), weights and `read` read-only, harness first on the path."""
    path = os.pathsep.join([str(tree / HARNESS_DIR), str(tree / "src"), str(tree)])
    return grader.jailed(tree, argv, timeout_s=timeout_s,
                         env_extra={**target.load().engine.env, **(env or {}), "PYTHONPATH": path}, read=read)


def clean_pristine(run_dir: Path) -> None:
    shutil.rmtree(run_dir / "pristine", ignore_errors=True)
