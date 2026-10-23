"""python -m kernels.search --task T [--shapes SHAPES.json | --captures DIR ...] --out DIR [--evals N] [--usd X]
                          [--model CODER] [--judge-model JUDGE|none] [--effort E] [--[no-]ncu] [--device cuda]
                          [--initial PROGRAM.py] [--retime K] [--tune-points N]

Our own layer-1 generator (kernels/search/loop.py) on one kernel task, judged by kernels.evolve's evaluator exactly as
`python -m kernels.evolve run` is: the same run dir, selection config (seen shapes only), re-timed promotion and
`kernels` ledger record, so the two generators compare like for like. Model calls cost money; --usd caps them."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Callable

import torch

from kernels.evolve import evaluator
from kernels.evolve.run import (_shapes, _target_tolerances, prepare, record, retime, setting, system_message,
                               tolerances)
from kernels.search import detect, tune
from kernels.search.llm import Claude
from kernels.search.loop import cascade, search
from kernels.search.tree import UCT_C
from kernels.tasks import TASKS, load


TUNE_FACT = ("A program may define, at module level, `CONFIG` (a dict of the launch parameters its kernel uses) and "
             "`TUNABLES` (for each CONFIG key, a list of values). After the search, each of the top re-timed programs "
             "that declares a grid has up to {points} points of it judged, CONFIG rewritten in its source; the best "
             "point overall is re-timed and replaces the winner if its median score is higher.")


def _median(code: str, judge, n: int = 3) -> tuple[float, dict]:
    """The median of n judgments; 0 when any of them scored 0 (a gate failed once: not promoted)."""
    runs = [judge(code)["metrics"] for _ in range(n)]
    scores = sorted(float(m.get("combined_score") or 0.0) for m in runs)
    med = scores[n // 2] if scores[0] > 0 else 0.0
    return med, min(runs, key=lambda m: abs(float(m.get("combined_score") or 0.0) - med))


def _tune(cands: list[tuple[str, dict, float | None]], judge, points: int,
          size_expr: str | None = None, margin: float = 0.0) -> tuple[str, dict, dict]:
    """`cands`: (code, metrics, re-timed median or None), the winner first. Every candidate that declares a grid is
    swept at `points` points (KernelEvolve tunes every candidate; tuning only the winner picks the program before its
    parameters are known); the best point overall, then a per-size-region dispatch over that candidate's grid, replace
    the winner only if their median beats the incumbent's by more than `margin` (the spread of the winner's own
    re-timings: a gain inside it is noise): (code, metrics, a record of the sweeps)."""
    code, metrics, base = cands[0]
    swept = [(i, c) + tune.sweep(c, judge, points)[::2] for i, (c, _, _) in enumerate(cands)
             if points and tune.params(c) is not None]
    if not swept:
        return code, metrics, {}
    rec = {"candidates": [{"index": i, "points": [{k: r[k] for k in ("config", "score")} for r in rows]}
                          for i, _, _, rows in swept], "promoted": False}
    i, src, variant, rows = max(swept, key=lambda x: max((r["score"] for r in x[3]), default=0.0))
    rec["from"] = i
    table = tune.regions(rows) if size_expr else None
    if variant == code and not table:
        return code, metrics, rec
    if base is None:
        base, _ = _median(code, judge)
    best, best_m = code, metrics
    if variant != code:
        med, m = _median(variant, judge)
        rec.update(median=med, base_median=base, margin=margin)
        if med > base + margin:
            best, best_m, base, rec["promoted"] = variant, m, med, True
    if table:
        regional = tune.dispatch(variant, size_expr, table)
        med, m = _median(regional, judge)
        rec["regions"] = {"table": table, "median": med, "promoted": med > base + margin}
        if med > base + margin:
            best, best_m, rec["promoted"] = regional, m, True
    return best, best_m, rec


def _top(trace: Path, retimed: list, code: str | None, metrics: dict | None, best, k: int) -> list:
    """The candidates to tune: the re-timed ones by median (flagged ones never re-timed), else the tree's best."""
    timed = sorted((r for r in retimed if "median" in r), key=lambda r: -r["median"])
    if code is None or not timed:
        return [(best.code, best.metrics, None)]
    codes = {r["child_id"]: r["child_code"] for r in map(json.loads, trace.read_text().splitlines()) if r.get("child_code")}
    return [(code, metrics, timed[0]["median"])] + [(codes[r["id"]], None, r["median"]) for r in timed[1:k]
                                                     if r["id"] in codes]


def _blob(path: Path, root: str | None = None) -> str | None:
    """The run's hypothesis log as a ledger blob: every change the judge named and what became of it."""
    from lab import ledger
    return ledger.put_blob(path, Path(root) if root else ledger.ROOT, ".jsonl") if path.exists() and path.stat().st_size \
        else None


def staged(judge_stage) -> Callable[[str, int], dict]:
    """(code, stage) -> the evaluator's result for that stage, the program written to a temp file."""
    def stage(code: str, n: int) -> dict:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(code)
        try:
            return judge_stage(f.name, n)
        finally:
            Path(f.name).unlink()
    return stage


def _funnel(a, task, review):
    from kernels import archive
    from kernels.search.funnel import Funnel, Models
    from lab import ledger
    if review is None:
        raise SystemExit("--funnel needs a judge (--judge-model)")
    entries = [e for e in archive.load(Path(a.archive)) if e.task == task.name] if a.archive else []
    return Funnel(Models(review, Claude(a.cheap_model, "low"), Claude(a.wide_model, a.effort), Claude(a.model, a.effort)),
                  staged(evaluator.judge_stage), entries, task.name, archive._embedder(ledger.ROOT) if entries else None,
                  Path(a.ptx).read_text() if a.ptx else None, a.per_call, a.keep, a.parallel, a.max_repairs,
                  a.ncu, a.precheck)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="kernels.search")
    ap.add_argument("--task", required=True, choices=TASKS)
    ap.add_argument("--shapes")
    ap.add_argument("--captures", action="append")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ledger", help="the ledger the run's `kernels` record goes to (default: this repo's)")
    ap.add_argument("--equal-captures", action="store_true",
                    help="each --captures dir (one traffic, e.g. a scored regime) weighs the same in the shape counts")
    ap.add_argument("--evals", type=int, default=30, help="programs judged, the root included")
    ap.add_argument("--usd", type=float, default=10.0, help="cap on model spend")
    ap.add_argument("--model", default="claude-opus-5-5", help="the coder")
    ap.add_argument("--judge-model", default="claude-opus-5-5", help="the judge; `none`: the coder alone")
    ap.add_argument("--judge-tools", action=argparse.BooleanOptionalAction, default=True,
                    help="the judge may search the web and query the ledger (its choice when)")
    ap.add_argument("--ncu", action=argparse.BooleanOptionalAction, default=None,
                    help="ncu counters for the judge (default: on with --device cuda)")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--initial")
    ap.add_argument("--timing-shapes", type=int, default=8)
    ap.add_argument("--retime", type=int, default=4, help="top candidates re-judged 3x; these are also the ones tuned")
    ap.add_argument("--detector-model", default="claude-opus-5-5",
                    help="reads each would-be winner for gaming before promotion; `none`: off")
    ap.add_argument("--tune-points", type=int, default=8,
                    help="grid points tried on each re-timed candidate's TUNABLES; 0: none")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--uct-c", type=float, default=UCT_C,
                    help="UCT exploration constant on rewards in [0, 1] (KernelPro's sqrt 2 is on ln speedup)")
    ap.add_argument("--stop-pct", type=float, default=80.0,
                    help="stop once the best is at this %% of its roofline (MANTIS: within 25%% of speed-of-light)")
    ap.add_argument("--patience", type=int, default=16, help="stop after this many judged programs with no new best")
    ap.add_argument("--beam", type=int, default=4, help="changes the judge names per expansion, each written and judged")
    ap.add_argument("--max-repairs", type=int, default=3, help="repairs of a failed child, chained")
    ap.add_argument("--funnel", action="store_true", help="each expansion is one funnel round (kernels/search/funnel.py)")
    ap.add_argument("--wide-model", default="claude-sonnet-5-5", help="the funnel's wide coder")
    ap.add_argument("--cheap-model", default="claude-haiku-5-5", help="merges duplicate hypotheses")
    ap.add_argument("--per-call", type=int, default=20, help="hypotheses per generator call (one call per slice)")
    ap.add_argument("--keep", type=int, default=6, help="programs past the quick speed test to the full exam")
    ap.add_argument("--parallel", type=int, default=4, help="correctness checks run at once")
    ap.add_argument("--archive", help="an exported idea archive (python -m kernels.archive export)")
    ap.add_argument("--ptx", help="the PTX comparison's feature cards (text)")
    ap.add_argument("--precheck", action=argparse.BooleanOptionalAction, default=True,
                    help="the funnel's no-GPU build check and compiler report before stage 1")
    a = ap.parse_args(argv)
    a.ncu = a.device.startswith("cuda") if a.ncu is None else a.ncu
    task = load(a.task)
    out = Path(a.out)
    tol = tolerances(_target_tolerances())
    sets, program = prepare(task, _shapes(task, a), out, a.device, tol, a.timing_shapes, a.initial)
    review = None if a.judge_model == "none" else Claude(a.judge_model, a.effort, tools=a.judge_tools)
    judge = cascade(evaluator.judge_stage, counters=a.ncu)
    system = (system_message(task) + "\n\n" + setting(task, sets, a.device, tol)
              + ("\n\n" + TUNE_FACT.format(points=a.tune_points) if a.tune_points else ""))
    fun = _funnel(a, task, review) if a.funnel else None
    tree, cost = search(program.read_text(), system, Claude(a.model, a.effort), judge, out, a.evals, a.usd, a.seed,
                        review=review, stop_pct=a.stop_pct or None, patience=a.patience or None,
                        beam=a.beam, max_repairs=a.max_repairs, uct_c=a.uct_c, funnel=fun)
    spent: list = []
    exclude = detect.detector(Claude(a.detector_model, a.effort), spent) if a.detector_model != "none" else None
    code, metrics, retimed = (retime(out / "trace.jsonl", a.retime, exclude=exclude) if a.retime
                              else (None, None, []))
    cost["usd"] = round(cost["usd"] + sum(spent), 4)
    cost["model_calls"] += len(spent)
    best = tree.best()
    if code is None and (any("excluded" in r for r in retimed) or (exclude and exclude(best.code, {
            "child_metrics": best.metrics}))):
        best = tree.root()                       # every candidate flagged: nothing is promoted over the engine's kernel
    plain = cascade(evaluator.judge_stage)       # tuning reads scores only: no ncu per grid point
    won = max((r for r in retimed if "median" in r), key=lambda r: r["median"], default=None)
    margin = max(won["scores"]) - min(won["scores"]) if won and code is not None else 0.0
    code, metrics, tuned = _tune(_top(out / "trace.jsonl", retimed, code, metrics, best, a.retime), plain,
                                 a.tune_points, task.size_expr, margin)
    cost["eval_s"] = evaluator.seconds(out / "eval_seconds.log")
    (out / "best_program.py").write_text(code or best.code)
    (out / "best_metrics.json").write_text(json.dumps(metrics or best.metrics, indent=1))
    models = [{"name": a.model, "role": "coder", "effort": a.effort}] + (
        [{"name": a.judge_model, "role": "judge", "effort": a.effort}] if review else [])
    rec = record(task, sets, models, a.evals, a.device,
                 out / "trace.jsonl", out / "best_program.py", metrics or best.metrics, retimed=retimed,
                 tool="kernels.search", cost=cost, root=Path(a.ledger) if a.ledger else None,
                 extra={"tuned": tuned, "hypotheses": _blob(out / "hypotheses.jsonl", a.ledger)})
    print(json.dumps({"best": metrics or best.metrics, "cost": cost, "nodes": len(tree.nodes), "out": str(out),
                      "ledger": rec["id"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
