"""The three tools that can make a change count: bench, equiv and submit. Each serves the agent's pristine tree
from outside the jail (lab/serve.py), measures it with the environment's own instruments, and writes the ledger.

  bench    the regimes on the seen split, short tier -> one headline per regime, raw
  equiv    the correctness gate against the target's reference -> pass | fail | inconclusive, with every metric
  submit   the only thing that can produce a win: needs a passing full-tier equiv and a seen bench for this
           snapshot, measures the held-out split at the full tier, compares with the base commit measured the same
           way, and records one aggregate per metric through the Thresholdout guard (the held-out numbers
           themselves never reach the ledger). Verdict per metric: improved | regressed | within_band | unknown_band.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from lab import ledger, serve, target
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


def run_regimes(url: str, t: target.Target, names: list[str], split: str, tier: str, seed: int) -> list[dict]:
    from regimes import suite
    ctx = suite.Ctx.from_target(t, url, split=split, tier=tier, seed=seed)
    return [suite.REGIMES[n](ctx) for n in names]


def headline(results: list[dict]) -> dict:
    """{regime: {objective, value, better, valid, invalid_reasons}}: the numbers, nothing judged."""
    return {r["regime"]: {"objective": r["objective"], "value": r["value"], "better": r["better"],
                          "valid": r["valid"], "invalid_reasons": r["invalid_reasons"]} for r in results}


def noise_band_pct(regime: str, t: target.Target | None = None) -> float | None:
    """The measured run-to-run band for a regime from `knowledge/noise/<regime>.json`, only if it was measured on
    this target's engine (a band from another engine or model says nothing about this one)."""
    p = target.noise_dir() / f"{regime}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    if t is not None and (d.get("target") != t.name or d.get("server") != "engine"):
        return None
    return d.get("band_pct")


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
        "noise_bands_pct": {n: noise_band_pct(n) for n in suite.REGIMES},
        "corpus_version": corpus_version(t.corpus_dir) if t.corpus_dir.is_dir() else None,
    }
    if ledger_root and (Path(ledger_root) / "holdout_state.json").exists():
        st = json.loads((Path(ledger_root) / "holdout_state.json").read_text())
        facts["heldout_queries_left"] = st.get("budget_left")
    return facts


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
        t = target.load()
        names = _regimes(args, self.bench_default_regimes)
        tier = args.get("tier") or "short"
        tree = self._pristine()
        t0 = time.monotonic()
        try:
            with serve.Served(tree, t.engine, log=self.s.run_dir / "serve-bench.log",
                              jailed=self.serve_jailed) as srv:
                results = run_regimes(srv.url, t, names, "seen", tier, self.seed)
        except Exception as e:                      # a tool never crashes the session: the failure is the record
            result = {"verdict": "error", "reason": _why(e), "seconds": time.monotonic() - t0}
            self._record("bench", "bench", args, result, snap, config={"split": "seen", "tier": tier})
            return f"bench failed: {result['reason']}"
        result = {"verdict": "ok", "metrics": headline(results), "regimes": results, "tier": tier,
                  "seconds": time.monotonic() - t0, "ready_s": srv.ready_s}
        self._record("bench", "bench", args, result, snap, config={"split": "seen", "tier": tier})
        return json.dumps({"snapshot": snap.id, "split": "seen", "tier": tier, "metrics": result["metrics"]}, indent=1)

    # -- equiv --------------------------------------------------------------------------
    def equiv(self, args: dict) -> str:
        snap = self._audited("equiv", args)
        t = target.load()
        tier = args.get("tier") or "dev"
        ref = t.reference_dir_for(tier)
        if not target.reference_complete(ref):
            self._record("equiv", "equiv", args, {"verdict": "refused", "reason": f"no {tier}-tier reference outputs; "
                         "run `python -m correctness reference` for this target first"}, snap)
            return "equiv refused: no reference outputs for this target"
        from correctness import client, run as crun
        from correctness.gate import Thresholds
        tree = self._pristine()
        out = self.s.run_dir / "equiv" / f"{snap.id}-{tier}.json"
        try:
            with serve.Served(tree, t.engine, log=self.s.run_dir / "serve-equiv.log", jailed=self.serve_jailed) as srv:
                res = crun.candidate(ref, srv.url, t.model, out, self.encoder(t),
                                     Thresholds(min_score_ratio=t.min_score_ratio, min_length_ratio=t.min_length_ratio),
                                     concurrency=self.equiv_concurrency, api=client.Api(t.engine.api, t.chat_kwargs),
                                     config={"tier": tier})
        except Exception as e:
            self._record("equiv", "equiv", args, {"verdict": "error", "reason": _why(e)}, snap, config={"split": "seen", "tier": tier})
            return f"equiv failed: {_why(e)}"
        passed = {"pass": True, "fail": False}.get(res["verdict"])     # inconclusive is None: neither passing nor failing
        record = {"verdict": res["verdict"], "passed": passed, "reasons": res["reasons"], "gates": res["gates"],
                  "metrics": res["metrics"], "thresholds": res["thresholds"], "tier": tier}
        self._record("equiv", "equiv", args, record, snap, config={"split": "seen", "tier": tier})
        return json.dumps({"snapshot": snap.id, "tier": tier, "verdict": res["verdict"], "reasons": res["reasons"],
                           "metrics": res["metrics"]}, indent=1, default=str)

    # -- submit -------------------------------------------------------------------------
    def submit(self, args: dict) -> str:
        snap = self._audited("submit", args)
        t = target.load()
        mine = [r for r in ledger.records(self.s.ledger_root, run=self.s.run_id, snapshot=snap.id)]
        eq = [r for r in mine if r["kind"] == "equiv" and (r.get("result") or {}).get("tier") == "full"]
        if not eq or eq[-1]["result"].get("verdict") != "pass":
            why = ("no full-tier equiv on this snapshot" if not eq else
                   f"latest full-tier equiv is {eq[-1]['result'].get('verdict')}")
            self._record("submit", "submit", args, {"verdict": "refused", "reason": f"{why}; run equiv with tier=full first"}, snap)
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
            base = self.base_heldout(t, names)
            base_seen = self.base_seen(t, names)
            guard = self.holdout_guard()
            tree = self._pristine()
            with serve.Served(tree, t.engine, log=self.s.run_dir / "serve-submit.log", jailed=self.serve_jailed) as srv:
                results = run_regimes(srv.url, t, names, "heldout", "full", self.seed)
            new = headline(results)
            metrics, better = {}, {}
            for n in names:
                held_d, seen_d = delta_pct(base.get(n, {}), new[n]), delta_pct(base_seen.get(n, {}), seen[n])
                # Thresholdout: a held-out delta is reported only through the guard, and only when it disagrees
                # with the seen one. Without a seen delta to compare, nothing held-out is reported at all.
                held_d = guard.query(seen_d, held_d)["value"] if held_d is not None and seen_d is not None else None
                better[n] = new[n].get("better", "higher")
                metrics[n] = aggregate(base.get(n, {}), held_d, better[n], noise_band_pct(n, t))
        except Refused as e:
            self._record("submit", "submit", args, {"verdict": "refused", "reason": str(e)}, snap)
            return f"submit refused: {e}"
        except Exception as e:
            self._record("submit", "submit", args, {"verdict": "error", "reason": _why(e)}, snap)
            return f"submit failed: {_why(e)}"
        rec = self._record_heldout("submit", "submit", args, {"tier": "full"}, metrics, snap)
        out = {"snapshot": snap.id, "split": "heldout", "tier": "full", "metrics": metrics, "record": rec["id"]}
        task = getattr(self.s, "task", None)
        if task is not None:                       # the task's own win condition, applied to these aggregates
            from lab.task import score
            out["score"] = score(metrics, task, better)
        return json.dumps(out, indent=1)

    # -- support ------------------------------------------------------------------------
    bench_default_regimes = ("single_stream", "saturated", "bursty")
    seed = 0
    equiv_concurrency = 16
    serve_jailed = True

    def encoder(self, t: target.Target):
        from correctness.run import HFEncoder
        return getattr(self.s, "encoder", None) or HFEncoder(t.model, t.chat_kwargs)

    def holdout_guard(self) -> HoldoutGuard:
        seed = os.environ.get("LAB_HOLDOUT_SEED") or f"{self.s.ledger_root}"
        return HoldoutGuard(self.s.ledger_root / "holdout_state.json", HOLDOUT_BUDGET, HOLDOUT_THRESHOLD_PCT,
                            HOLDOUT_SIGMA_PCT, seed)

    def _base_measure(self, t: target.Target, names: list[str], split: str, tier: str) -> dict:
        """The base commit measured like a candidate, once per run per split; cached outside the jail."""
        cache = self.s.run_dir / f"base-{split}-{tier}.json"
        have = json.loads(cache.read_text()) if cache.exists() else {}
        missing = [n for n in names if n not in have]
        if missing:
            tree = self.s.run_dir / f"base-tree-{split}"
            from lab.safety import grader
            grader.export(self.s.workspace.repo, self.s.workspace.base, tree)
            with serve.Served(tree, t.engine, log=self.s.run_dir / f"serve-base-{split}.log", jailed=self.serve_jailed) as srv:
                have.update(headline(run_regimes(srv.url, t, missing, split, tier, self.seed)))
            cache.write_text(json.dumps(have, indent=1))
        return have

    def base_heldout(self, t, names):
        return self._base_measure(t, names, "heldout", "full")

    def base_seen(self, t, names):
        return self._base_measure(t, names, "seen", BENCH_TIER)
