"""The three tools that can make a change count: bench, equiv and submit. Each has the worker (lab/worker.py) serve
the agent's pristine tree outside the jail and measure it with the environment's own instruments; the policy and the
ledger stay here.

  bench    the regimes on the seen split, short tier -> one headline per regime, raw
  equiv    the correctness gate against the target's reference -> pass | fail | inconclusive, with every metric
  submit   the only thing that can produce a win: needs a passing equiv at the target's submit tier and a seen bench for this
           snapshot, measures the held-out split at the short tier, compares with the base commit measured the same
           way, and records one aggregate per metric through the Thresholdout guard (the held-out numbers
           themselves never reach the ledger). Verdict per metric: improved | regressed | within_band | unknown_band.

Each also keeps the run's passive data (lab/artifacts.py): bench and equiv as ledger blobs under `result.artifacts`,
submit only under `<run>/heldout-private/`, outside the ledger and the jail.
`baseline` is not a tool: the run measures its base commit under bench's defaults once, before the first session,
in the same cache submit compares against.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from pathlib import Path

from lab import artifacts, calibrate, ledger, live, profiles, serve, target, worker
from validity.holdout import HoldoutGuard, Refused

HOLDOUT_BUDGET, HOLDOUT_THRESHOLD_PCT, HOLDOUT_SIGMA_PCT = 100, 2.0, 0.5
BENCH_TIER = "short"          # the seen-split tier submit compares against (base is measured at the same tier)


def _regimes(args: dict, default: tuple[str, ...]) -> list[str]:
    from regimes import suite
    want = args.get("regimes") or list(default)
    if want == "all" or want == ["all"]:
        want = list(suite.REGIMES)
    unknown = [n for n in want if n not in suite.REGIMES]
    if unknown:
        raise ValueError(f"unknown regime(s) {unknown}; one of {list(suite.REGIMES)}")
    return list(want)


def run_regimes(url: str, t: target.Target, names: list[str], split: str, tier: str, seed: int,
                rows: list | None = None, hints: dict | None = None, bands: dict | None = None) -> list[dict]:
    """Each regime's result; with `rows`, every measured client row is appended to it, tagged with its regime.
    `hints` / `bands`: where the base's searches ended and each regime's noise band (`suite.Ctx`)."""
    from regimes import suite
    watch = split == "seen"             # the dashboard gets the seen split's progress, never the held-out's
    ctx = suite.Ctx.from_target(t, url, split=split, tier=tier, seed=seed, rows=rows, hints=hints or {},
                                bands=bands or {}, on_rows=_progress if watch else None)
    out = []
    for n in names:
        if watch:
            live.emit("regime", regime=n)
        k = len(rows) if rows is not None else 0
        out.append(suite.REGIMES[n](ctx))
        for r in (rows or [])[k:]:
            r["regime"] = n
    return out


def _progress(rows) -> None:
    """One load call's raw numbers for the dashboard (`lab.live`), computed after the call: never during it."""
    from regimes.runner import _q
    ok = [r for r in rows if r.status == "ok"]
    ttft, tpot = [r.ttft_s for r in ok if r.ttft_s is not None], [r.tpot_s for r in ok if r.tpot_s is not None]
    live.emit("probe", n=len(rows), ok=len(ok), ttft_p50_s=_q(ttft, 0.5), ttft_p99_s=_q(ttft, 0.99),
              tpot_p50_s=_q(tpot, 0.5), tpot_p99_s=_q(tpot, 0.99))


def headline(results: list[dict]) -> dict:
    """{regime: {objective, value, better, valid, invalid_reasons}}: the numbers, nothing judged. A regime also served
    under its own profile reports that launch, with `profile` and the general launch's value and validity."""
    out: dict = {}
    for r in results:
        h = {"objective": r["objective"], "value": r["value"], "better": r["better"], "valid": r["valid"],
             "invalid_reasons": r["invalid_reasons"], **({"concurrency": r["concurrency"]} if r.get("concurrency") else {})}
        if r.get("profile"):
            g = out.get(r["regime"])
            h.update(profile=r["profile"], general={"value": g["value"], "valid": g["valid"]} if g else None)
        out[r["regime"]] = h
    return out


def noise_band_pct(regime: str, t: target.Target | None = None, root: Path = ledger.ROOT) -> float | None:
    """The latest run-to-run band for a regime from the ledger's `noise` records, only if it was measured on this
    target's engine (at any engine commit: the calibration is once per user) and on its current corpus version (a
    record without one counts); a band from another target or corpus version is ignored."""
    from regimes.workload import corpus_version
    version = corpus_version(t.corpus_dir) if t is not None and t.corpus_dir.is_dir() else None
    d = None
    for r in ledger.records(root, kind="noise"):
        x = r["result"]
        if x.get("regime") == regime and (t is None or (x.get("target") == t.name and x.get("server") == "engine"
                                                         and x.get("corpus_version") in (None, version))):
            d = x
    return None if d is None else d.get("band_pct")


LOAD_SEARCHED = ("bursty", "long_prompt_short_output", "short_prompt_long_output", "shared_prefix_multi_turn")


def search_hints(base: dict) -> dict:
    """Where each of the base's valid searches ended (saturated: its concurrency; a load search: its load), for a
    candidate's search to start there."""
    out = {}
    for n, h in base.items():
        if not h.get("valid") or h.get("value") is None:
            continue
        if h.get("concurrency"):
            out[n] = {"concurrency": h["concurrency"]}
        elif n in LOAD_SEARCHED:
            out[n] = {"load": h["value"]}
    return out


def delta_pct(base: dict, new: dict) -> float | None:
    """Percent change of a headline, None when either side is missing or invalid."""
    b, n = base.get("value"), new.get("value")
    if b is None or n is None or not b or not base.get("valid") or not new.get("valid"):
        return None
    return (n - b) / b * 100


def aggregate(base: dict, delta: float | None, better: str, band_pct: float | None) -> dict:
    """One held-out aggregate: base, new (derived from the reported delta), delta %, band %, verdict.
    Direction-aware: improved | regressed | within_band | unknown_band | unknown."""
    b = base.get("value")
    if delta is None or b is None:
        return {"base": b, "new": None, "delta_pct": None, "band_pct": band_pct, "verdict": "unknown"}
    gain = delta if better == "higher" else -delta
    verdict = ("unknown_band" if band_pct is None else "improved" if gain > band_pct
               else "regressed" if gain < -band_pct else "within_band")
    return {"base": b, "new": b * (1 + delta / 100), "delta_pct": delta, "band_pct": band_pct, "verdict": verdict}


def heldout_context(seen_cal: dict, base: dict, delta: float | None, better: str) -> dict:
    """A held-out aggregate's context: the ceiling computed from the base commit's seen-split request shape (nothing
    about the held-out requests goes into it), as % for the base and the new value. The reference is measured on the
    seen split at the short tier, so it is not comparable to a held-out full-tier number."""
    from validity.ceiling import pct_of
    c = seen_cal.get("ceiling")
    b = base.get("value")
    new = b * (1 + delta / 100) if b is not None and delta is not None else None
    return {"ceiling": c, "pct_of_ceiling_base": pct_of(b, c, better), "pct_of_ceiling_new": pct_of(new, c, better),
            "ceiling_from": "the base commit's seen-split request shape" if c is not None
            else (seen_cal.get("ceiling_missing") or "no seen-split calibration of the base in this run"),
            "reference_missing": "the reference is measured on the seen split; this is the held-out split"}


def harness_facts(t: target.Target, ledger_root: Path | None = None) -> dict:
    """What the referee measures against: the parameters the agent cannot read off its ledger. Facts, not advice."""
    from regimes import suite
    from regimes.workload import corpus_version
    facts = {
        "model": t.model, "reference": t.reference.launch if t.reference else None,
        "latency_limits": {"interactive": t.interactive.__dict__, "conversational": t.conversational.__dict__},
        "goodput_attainment": 0.99,
        "tiers": {k: dict(v) for k, v in suite.TIERS.items()},
        "correctness": {"policy": f"pooled task score >= {t.min_score_ratio} of the reference, length >= {t.min_length_ratio}, "
                        f"consistent streams; under the line fails only when the net loss is significant",
                        "tasks": list(t.tasks)},
        "noise_bands_pct": {n: noise_band_pct(n, t, ledger_root or ledger.ROOT) for n in suite.REGIMES},
        "noise_bands_measured_on": "the base commit's engine with its general config, repeated unchanged runs "
                                   "(lab.validate, the latest at any engine commit on this corpus version); submit applies them "
                                   "to every candidate and "
                                   "every launch profile; None where no band was measured for this target",
        "corpus_version": corpus_version(t.corpus_dir) if t.corpus_dir.is_dir() else None,
        "ceilings": _ceiling_facts(t, ledger_root),
        "layers": _layer_facts(ledger_root),
        "kernel_tasks": _layer_facts(ledger_root, "kernel_tasks"),
    }
    try:
        facts["heldout_queries_left"] = holdout_guard(t).state()["budget_left"]
    except Exception:                            # noqa: BLE001  no store here, or unreachable: the fact is absent
        pass
    return facts


def holdout_guard(t: target.Target) -> HoldoutGuard:
    """The one Thresholdout guard of the target's corpus prompts (`Manifest.budget_key`), shared by every run and machine through the store
    (LAB_STORE). Refused where there is no store: a local budget would start over on every machine."""
    from store.remote import NoRemote, open_remote
    try:
        remote = open_remote()
    except NoRemote as e:
        raise Refused(f"no shared held-out budget on this host ({e})") from None
    from lab.corpus import load_manifest
    version = load_manifest(t.corpus_dir).budget_key if t.corpus_dir.is_dir() else "no-corpus"   # synthetic workloads
    return HoldoutGuard(remote, f"heldout/guard/{version}", HOLDOUT_BUDGET, HOLDOUT_THRESHOLD_PCT, HOLDOUT_SIGMA_PCT)


def _ceiling_facts(t: target.Target, ledger_root: Path | None) -> dict:
    """Peaks, streamed bytes and floors on the GPU this run measured on; nothing before its first measurement."""
    g = _last_gpu(ledger_root)
    if g is None:
        return {"missing": "no measurement in this ledger yet, so no GPU to compute ceilings for"}
    try:
        return calibrate.facts(t, g)
    except Exception as e:                       # noqa: BLE001
        return {"missing": f"ceilings not computed: {type(e).__name__}: {e}"}


def _layer_facts(ledger_root: Path | None, key: str = "layers") -> dict:
    """The latest profile's per-layer shares and distances (lab/layers.py), or with `key="kernel_tasks"` each kernel
    task's share of its window, with the record they come from: a `profile` record or a bench's or baseline's
    after-measurement profile (`profiled`)."""
    if not ledger_root or not (Path(ledger_root) / "ledger.jsonl").exists():
        return {"missing": "no profile in this ledger yet"}
    for r in reversed(list(ledger.records(ledger_root))):
        res = r.get("result") or {}
        cal = (res.get("calibrated") if r.get("kind") == "profile" else
               (res.get("profiled") or {}).get("calibrated") if r.get("kind") in ("bench", "baseline") else None) or {}
        if isinstance(cal, dict) and isinstance(cal.get(key), dict) and "missing" not in cal[key]:
            return {"record": r["id"], "run": r.get("run"), "args": r.get("args"), **cal[key]}
    what = "a per-layer breakdown" if key == "layers" else "kernel task shares"
    return {"missing": f"no profile with {what} in this ledger yet (`profile`, and every bench, writes one)"}


def _last_gpu(ledger_root: Path | None) -> dict | None:
    """The GPU the run's measurements last ran on, from the latest bench or baseline record (the controller has none)."""
    if not ledger_root or not (Path(ledger_root) / "ledger.jsonl").exists():
        return None
    for r in reversed(list(ledger.records(ledger_root))):
        g = (r.get("result") or {}).get("gpu") if r.get("kind") == "bench" else None
        if g is None and r.get("kind") == "baseline":
            fp = next(iter(((r.get("result") or {}).get("calibrated") or {}).values()), {}).get("fingerprint") or {}
            g = {"name": fp.get("gpu"), "driver_version": fp.get("driver")} if fp.get("gpu") else None
        if g:
            return g
    return None


def equiv_rows(out: Path) -> list[dict]:
    """The gate's per-item client rows (id, completion_tokens, finish_reason, error), without generated text."""
    p = out.with_suffix(".outputs.jsonl")
    if not p.exists():
        return []
    return [{k: v for k, v in json.loads(x).items() if k != "text"} for x in p.read_text().splitlines() if x.strip()]


def _profiled_reply(prof: dict) -> dict:
    """The bench reply's `profiled`: the record's own shape (so a path read here resolves in the record), with only
    the layer facts of its calibration, or the calibration's error."""
    out = {k: v for k, v in prof.items() if k != "calibrated"}
    cal = prof.get("calibrated")
    if isinstance(cal, dict):
        out["calibrated"] = ({"error": cal["error"]} if "error" in cal else
                             {k: cal.get(k) for k in ("layers", "kernel_tasks") if k == "layers" or k in cal})
    return out


def _failed(e: BaseException) -> dict:
    """The record of a measurement that did not complete; GPU processes that stopped it are kept as facts."""
    if isinstance(e, serve.GpuBusy):
        return {"verdict": "refused", "reason": str(e), "gpu_processes": e.apps}
    if isinstance(e, serve.Contaminated):
        return {"verdict": "contaminated", "reason": str(e), "gpu_processes": e.apps}
    return {"verdict": "error", "reason": _why(e)}


def _killed(r: dict) -> dict:
    """Processes of the agent's workbench killed because they held the GPU when the measurement began (a fact)."""
    return {"workbench_killed": r["killed"]} if r.get("killed") else {}


def _said(tool: str, result: dict) -> str:
    return f"{tool} {'failed' if result['verdict'] == 'error' else result['verdict']}: {result['reason']}"


def _why(e: BaseException) -> str:
    if isinstance(e, serve.NotReady):
        return f"engine did not start: {e}"
    return f"{type(e).__name__}: {e}"[:500]


class EvalTools:
    """Mixin for lab.tools.Toolbox: needs self.s (session), self._audited, self._pristine, self._record,
    self._record_heldout."""

    # -- bench --------------------------------------------------------------------------
    def bench(self, args: dict) -> str:
        snap = self._audited("bench", args)
        names = _regimes(args, self.bench_defaults())
        tier = args.get("tier") or "short"
        seed = args.get("seed", self.seed)
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        tree = self._pristine()
        t0 = time.monotonic()
        work = self._work("bench")
        try:
            base = self._base_cache("seen", tier)
            hints = search_hints(json.loads(base.read_text())) if base.exists() and seed == self.seed else {}
            r = self._measure(tree, work, "serve-bench.log", names=names, split="seen", tier=tier, passive=True,
                              seed=seed, hints=hints, bands=self._bands(names))
        except Exception as e:                      # a tool never crashes the session: the failure is the record
            result = {**_failed(e), "seconds": time.monotonic() - t0,
                      "artifacts": artifacts.store(work, self.s.ledger_root)}
            self._record("bench", "bench", args, result, snap, config={"split": "seen", "tier": tier, "seed": seed})
            return _said("bench", result)
        cal = calibrate.calibrate_results(r["results"], artifacts.read_rows(work), target.load(), r.get("gpu"), tier,
                                          "seen", seed)
        failed = {"profiles_failed": r["profiles_failed"]} if r.get("profiles_failed") else {}
        stored = artifacts.store(work, self.s.ledger_root)
        prof = self._after_bench(tree)
        result = {"verdict": "ok", "metrics": headline(r["results"]), "calibrated": cal, "regimes": r["results"],
                  "tier": tier, "seconds": time.monotonic() - t0, "ready_s": r["ready_s"], "gpu": r.get("gpu"),
                  "artifacts": stored, "profiled": prof, **failed, **_killed(r)}
        self._record("bench", "bench", args, result, snap, config={"split": "seen", "tier": tier, "seed": seed})
        return json.dumps({"snapshot": snap.id, "split": "seen", "tier": tier, "metrics": result["metrics"],
                           "calibrated": cal, "profiled": _profiled_reply(prof), **failed, **_killed(r)}, indent=1)

    # -- equiv --------------------------------------------------------------------------
    def equiv(self, args: dict) -> str:
        snap = self._audited("equiv", args)
        t = target.load()
        tier = args.get("tier") or "dev"
        profile = args.get("profile") or None
        config = {"split": "seen", "tier": tier, **({"profile": profile} if profile else {})}
        ref = t.reference_dir_for("dev" if tier == "quick" else tier)
        if not target.reference_complete(ref):
            self._record("equiv", "equiv", args, {"verdict": "refused", "reason": f"no {tier}-tier reference outputs; "
                         "run `python -m correctness reference` for this target first"}, snap, config=config)
            return "equiv refused: no reference outputs for this target"
        tree = self._pristine()
        why = profiles.refusal(tree, t, profile)
        if why:
            self._record("equiv", "equiv", args, {"verdict": "refused", "reason": why}, snap, config=config)
            return f"equiv refused: {why}"
        out = self.s.run_dir / "equiv" / f"{snap.id}-{tier}{'-' + profile if profile else ''}.json"
        work = self._work("equiv")
        # answers already collected for this snapshot are reused, never re-rolled: the gate is statistical
        prior = Path(tempfile.mkdtemp(prefix="equiv-prior-", dir=self.s.run_dir))
        if out.with_suffix(".outputs.jsonl").exists():
            shutil.copyfile(out.with_suffix(".outputs.jsonl"), prior / "equiv.outputs.jsonl")
        try:
            res = self.worker.call("equiv", {"tree": tree, "reference": ref, "prior": prior},
                                   {"tier": tier, "concurrency": self.equiv_concurrency, "jailed": self.serve_jailed,
                                    "profile": profile, "workbench": getattr(self.s, "workbench", None)}, work)
        except Exception as e:
            result = {**_failed(e), "artifacts": self._equiv_store(work, out, prior)}
            self._record("equiv", "equiv", args, result, snap, config=config)
            return _said("equiv", result)
        passed = {"pass": True, "fail": False}.get(res["verdict"])     # inconclusive and quick's facts are None: neither
        record = {"verdict": res["verdict"], "passed": passed, "reasons": res["reasons"], "gates": res["gates"],
                  "metrics": res["metrics"], "thresholds": res["thresholds"], "tier": tier,
                  "artifacts": self._equiv_store(work, out, prior), **_killed(res)}
        self._record("equiv", "equiv", args, record, snap, config=config)
        return json.dumps({"snapshot": snap.id, "tier": tier, **({"profile": profile} if profile else {}),
                           "verdict": res["verdict"], "reasons": res["reasons"], "metrics": res["metrics"],
                           **_killed(res)}, indent=1, default=str)

    # -- submit -------------------------------------------------------------------------
    def submit(self, args: dict) -> str:
        snap = self._audited("submit", args)
        t = target.load()
        mine = [r for r in ledger.records(self.s.ledger_root, run=self.s.run_id, snapshot=snap.id)]

        need = t.submit_tier

        def full_equiv(profile: str | None) -> dict | None:
            """The latest equiv at the target's submit tier of one profile's launch (None = general) on this snapshot."""
            return next((r for r in reversed(mine) if r["kind"] == "equiv"
                         and (r.get("result") or {}).get("tier") == need
                         and (r.get("config") or {}).get("profile") == profile), None)
        eq = full_equiv(None)
        if not eq or eq["result"].get("verdict") != "pass":
            why = (f"no {need}-tier equiv on this snapshot" if not eq else
                   f"latest {need}-tier equiv is {eq['result'].get('verdict')}")
            self._record("submit", "submit", args, {"verdict": "refused", "reason": f"{why}; run equiv with tier={need} first"}, snap)
            return f"submit refused: {why}"
        benches = [r for r in mine if r["kind"] == "bench" and (r.get("result") or {}).get("verdict") == "ok"
                   and (r.get("config") or {}).get("split") == "seen" and (r.get("config") or {}).get("tier") == BENCH_TIER]
        if not benches:
            why = f"no completed {BENCH_TIER}-tier seen-split bench on this snapshot"
            self._record("submit", "submit", args, {"verdict": "refused", "reason": why}, snap)
            return f"submit refused: {why}"
        seen = benches[-1]["result"]["metrics"]
        try:
            names = [n for n in _regimes(args, tuple(seen)) if n in seen]
            guard = holdout_guard(t)            # before anything is measured: no budget here is a cheap refusal
            tree = self._pristine()
            # every config that will be measured passed the gate on its own: a profile is a different engine
            own = [n for n in names if n in profiles.overrides(tree, t)]
            short = [n for n in own if ((full_equiv(n) or {}).get("result") or {}).get("verdict") != "pass"]
            if short:
                why = f"no passing {need}-tier equiv for profile(s) {short}; run equiv with tier={need} and that profile first"
                self._record("submit", "submit", args, {"verdict": "refused", "reason": why}, snap)
                return f"submit refused: {why}"
            base = self.base_heldout(t, names)
            base_seen = self.base_seen(t, names)
            # held-out passive data stays operator-only: never a blob, never in the record, never inside the jail.
            # A contaminated run raises here, before the holdout guard: it spends no query.
            private = self.s.run_dir / "heldout-private" / f"{snap.id}-{time.strftime('%Y%m%dT%H%M%S')}"
            r = self._measure(tree, private, "serve-submit.log", names=names, split="heldout", tier=BENCH_TIER,
                              passive=True, hints=search_hints(base), bands=self._bands(names))
            # which profile launches failed and how (did not start / the exception's type), never its text
            failed = {p: why.split(":", 1)[0] for p, why in (r.get("profiles_failed") or {}).items()}
            new = headline(r["results"])
            cal_path = self._base_cache("seen", BENCH_TIER).with_suffix(".calibrated.json")
            seen_cal = json.loads(cal_path.read_text()) if cal_path.exists() else {}
            metrics, better = {}, {}
            for n in names:
                held_d, seen_d = delta_pct(base.get(n, {}), new[n]), delta_pct(base_seen.get(n, {}), seen[n])
                # Thresholdout: a held-out delta is reported only through the guard, and only when it disagrees
                # with the seen one. Without a seen delta to compare, nothing held-out is reported at all.
                held_d = guard.query(seen_d, held_d)["value"] if held_d is not None and seen_d is not None else None
                better[n] = new[n].get("better", "higher")
                metrics[n] = {**aggregate(base.get(n, {}), held_d, better[n], noise_band_pct(n, t, self.s.ledger_root)),
                              **heldout_context(seen_cal.get(n) or {}, base.get(n, {}), held_d, better[n])}
        except Refused as e:
            self._record("submit", "submit", args, {"verdict": "refused", "reason": str(e)}, snap)
            return f"submit refused: {e}"
        except Exception as e:
            result = _failed(e)
            self._record("submit", "submit", args, result, snap)
            return _said("submit", result)
        config = {"tier": BENCH_TIER, **({"profiles": own} if own else {}), **({"profiles_failed": failed} if failed else {})}
        rec = self._record_heldout("submit", "submit", args, config, metrics, snap)
        out = {"snapshot": snap.id, "split": "heldout", "tier": BENCH_TIER, "metrics": metrics, "record": rec["id"],
               **({"profiles_failed": failed} if failed else {}), **_killed(r)}
        task = getattr(self.s, "task", None)
        if task is not None:                       # the task's own win condition, applied to these aggregates
            from lab.task import score
            out["score"] = score(metrics, task, better)
        return json.dumps(out, indent=1)

    # -- support ------------------------------------------------------------------------
    bench_default_regimes = ("saturated",)      # the general regime: the corpus mix under the conversational limits

    def bench_defaults(self) -> tuple[str, ...]:
        """The general regime plus, with a task, its objective and constrained regimes (what submit scores)."""
        task = getattr(self.s, "task", None)
        extra = (*task.regimes, *task.constraints) if task else ()
        return tuple(dict.fromkeys((*self.bench_default_regimes, *extra)))
    seed = 0
    equiv_concurrency = 32       # dev tier ~26 min on inference-server #107 at 32 (RUNS.md 2026-10-15); sheds are retried
    serve_jailed = True

    def _bands(self, names: list[str]) -> dict:
        """Each regime's calibrated noise band %, where one exists: the searches resolve to it."""
        t = target.load()
        return {n: b for n in names if (b := noise_band_pct(n, t, self.s.ledger_root)) is not None}

    def _work(self, tool: str) -> Path:
        """A scratch dir outside the jail for one worker job's output; emptied into blobs or removed after."""
        return Path(tempfile.mkdtemp(prefix=f"passive-{tool}-", dir=self.s.run_dir))

    def _measure(self, tree: Path, out: Path, log: str, **args) -> dict:
        """The worker's `measure` job into `out`; the engine's log is appended to `<run>/<log>` either way."""
        try:
            return self.worker.call("measure", {"tree": tree}, {"seed": self.seed, **args, "jailed": self.serve_jailed,
                                                                "workbench": getattr(self.s, "workbench", None)}, out)
        finally:
            src = out / worker.ENGINE_LOG
            if src.exists():
                with open(self.s.run_dir / log, "ab") as f:
                    f.write(src.read_bytes())
                src.unlink()

    def _equiv_store(self, work: Path, out: Path, prior: Path) -> dict:
        """The gate's files to `out` (kept per snapshot for the next equiv), its rows and passive data as blobs."""
        shutil.rmtree(prior, ignore_errors=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        for name, dest in (("equiv.json", out), ("equiv.outputs.jsonl", out.with_suffix(".outputs.jsonl"))):
            if (work / name).exists():
                shutil.move(work / name, dest)
        log = work / worker.ENGINE_LOG
        if log.exists():
            with open(self.s.run_dir / "serve-equiv.log", "ab") as f:
                f.write(log.read_bytes())
            log.unlink()
        artifacts.write_rows(work, equiv_rows(out))
        return artifacts.store(work, self.s.ledger_root)

    def _base_cache(self, split: str, tier: str) -> Path:
        """The base's cached measurement, one per corpus version: a base from another corpus is never compared."""
        from regimes.workload import corpus_version
        t = target.load()
        v = corpus_version(t.corpus_dir) if t.corpus_dir.is_dir() else None
        return self.s.run_dir / (f"base-{split}-{tier}-{v[:12]}.json" if v else f"base-{split}-{tier}.json")

    def _base_measure(self, t: target.Target, names: list[str], split: str, tier: str) -> dict:
        """The base commit measured like a candidate, once per run per split; cached outside the jail."""
        cache = self._base_cache(split, tier)
        have = json.loads(cache.read_text()) if cache.exists() else {}
        missing = [n for n in names if n not in have]
        if missing:
            tree = self.s.run_dir / f"base-tree-{split}"
            from lab.safety import grader
            grader.export(self.s.workspace.repo, self.s.workspace.base, tree)
            work = self._work("base")
            try:                                # a contaminated base raises: never cached, every later submit reads it
                r = self._measure(tree, work, f"serve-base-{split}.log", names=missing, split=split, tier=tier,
                                  passive=split == "seen", bands=self._bands(missing))   # held-out rows stay on the worker
                base_rows = artifacts.read_rows(work) if split == "seen" else []
            finally:
                shutil.rmtree(work, ignore_errors=True)
            have.update(headline(r["results"]))
            cache.write_text(json.dumps(have, indent=1))
            if split == "seen":                 # the base's context travels with the baseline record
                cal = cache.with_suffix(".calibrated.json")
                prior = json.loads(cal.read_text()) if cal.exists() else {}
                prior.update(calibrate.calibrate_results(r["results"], base_rows, t, r.get("gpu"), tier, split,
                                                         self.seed))
                cal.write_text(json.dumps(prior, indent=1))
        return have

    def baseline(self, commit: str) -> dict:
        """The base commit under bench's defaults (seen split, short tier), once per run: a `baseline` record."""
        have = next(ledger.records(self.s.ledger_root, run=self.s.run_id, kind="baseline"), None)
        if have:
            return have
        names = list(self.bench_defaults())
        self._t0 = time.monotonic()
        try:
            seen = self.base_seen(target.load(), names)
            cal = self._base_cache("seen", BENCH_TIER).with_suffix(".calibrated.json")
            calibrated = json.loads(cal.read_text()) if cal.exists() else {}
            result = {"verdict": "ok", "metrics": {n: seen[n] for n in names},
                      "calibrated": {n: calibrated[n] for n in names if n in calibrated}}
        except Exception as e:                      # no GPU or a base that will not serve is itself the fact
            result = _failed(e)
        else:                                       # profiled like every bench, for change_pct mechanism predictions
            from lab.safety import grader
            tree = self.s.run_dir / "base-tree-profile"
            shutil.rmtree(tree, ignore_errors=True)
            try:
                grader.export(self.s.workspace.repo, self.s.workspace.base, tree)
                result["profiled"] = self._after_bench(tree)
            except Exception as e:                  # noqa: BLE001  the baseline record is written regardless
                result["profiled"] = {"error": f"{type(e).__name__}: {e}"}
            finally:
                shutil.rmtree(tree, ignore_errors=True)
        return ledger.append({"kind": "baseline", "run": self.s.run_id, "session": self.s.session_id,
                              "config": {"split": "seen", "tier": BENCH_TIER, "commit": commit},
                              "result": result, "cost": self._cost("baseline")}, self.s.ledger_root)

    def base_heldout(self, t, names):
        return self._base_measure(t, names, "heldout", BENCH_TIER)

    def base_seen(self, t, names):
        return self._base_measure(t, names, "seen", BENCH_TIER)
