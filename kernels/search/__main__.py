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
from pathlib import Path

import torch

from kernels.evolve import evaluator
from kernels.evolve.run import _shapes, _target_tolerances, prepare, record, retime, system_message, tolerances
from kernels.search import tune
from kernels.search.llm import Claude
from kernels.search.loop import cascade, search
from kernels.tasks import TASKS, load


TUNE_FACT = ("A program may define, at module level, `CONFIG` (a dict of the launch parameters its kernel uses) and "
             "`TUNABLES` (for each CONFIG key, a list of values). After the search, up to --tune-points points of that "
             "grid are judged on the best program, CONFIG rewritten in its source; the best point is re-timed and "
             "replaces the program if its median score is higher.")


def _median(code: str, judge, n: int = 3) -> tuple[float, dict]:
    runs = [judge(code)["metrics"] for _ in range(n)]
    scores = sorted(float(m.get("combined_score") or 0.0) for m in runs)
    med = scores[n // 2]
    return med, min(runs, key=lambda m: abs(float(m.get("combined_score") or 0.0) - med))


def _tune(code: str, metrics: dict, judge, points: int, size_expr: str | None = None) -> tuple[str, dict, dict]:
    """The winner, its best grid point, or a per-size-region dispatch over grid points, whichever re-times highest
    (each challenger must beat the incumbent's median): (code, metrics, a record of the sweep)."""
    if not points or tune.params(code) is None:
        return code, metrics, {}
    variant, _, rows = tune.sweep(code, judge, points)
    rec = {"points": [{k: r[k] for k in ("config", "score")} for r in rows], "promoted": False}
    best, best_m = code, metrics
    table = tune.regions(rows) if size_expr else None
    if variant == code and not table:
        return best, best_m, rec
    base, _ = _median(code, judge)
    if variant != code:
        med, m = _median(variant, judge)
        rec.update(median=med, base_median=base)
        if med > base:
            best, best_m, base, rec["promoted"] = variant, m, med, True
    if table:
        regional = tune.dispatch(best, size_expr, table)
        med, m = _median(regional, judge)
        rec["regions"] = {"table": table, "median": med, "promoted": med > base}
        if med > base:
            best, best_m, rec["promoted"] = regional, m, True
    return best, best_m, rec


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="kernels.search")
    ap.add_argument("--task", required=True, choices=TASKS)
    ap.add_argument("--shapes")
    ap.add_argument("--captures", action="append")
    ap.add_argument("--out", required=True)
    ap.add_argument("--evals", type=int, default=30, help="programs judged, the root included")
    ap.add_argument("--usd", type=float, default=10.0, help="cap on model spend")
    ap.add_argument("--model", default="claude-opus-5-5", help="the coder")
    ap.add_argument("--judge-model", default="claude-opus-5-5", help="the judge; `none`: the coder alone")
    ap.add_argument("--ncu", action=argparse.BooleanOptionalAction, default=None,
                    help="ncu counters for the judge (default: on with --device cuda)")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--initial")
    ap.add_argument("--timing-shapes", type=int, default=8)
    ap.add_argument("--retime", type=int, default=3)
    ap.add_argument("--tune-points", type=int, default=16, help="grid points tried on the winner's TUNABLES; 0: none")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stop-pct", type=float, default=80.0,
                    help="stop once the best is at this %% of its roofline (MANTIS: within 25%% of speed-of-light)")
    ap.add_argument("--patience", type=int, default=16, help="stop after this many judged programs with no new best")
    ap.add_argument("--hypotheses", type=int, default=16, help="changes the judge names per expansion")
    ap.add_argument("--beam", type=int, default=4, help="of those, programs written and judged per expansion")
    ap.add_argument("--explore", type=int, default=1, help="of the beam, picked at random (unbiased labels)")
    ap.add_argument("--max-repairs", type=int, default=3, help="repairs of a failed child, chained")
    a = ap.parse_args(argv)
    a.ncu = a.device.startswith("cuda") if a.ncu is None else a.ncu
    task = load(a.task)
    out = Path(a.out)
    sets, program = prepare(task, _shapes(task, a), out, a.device, tolerances(_target_tolerances()),
                            a.timing_shapes, a.initial)
    review = None if a.judge_model == "none" else Claude(a.judge_model, a.effort)
    judge = cascade(evaluator.judge_stage, counters=a.ncu)
    system = system_message(task) + ("\n\n" + TUNE_FACT if a.tune_points else "")
    tree, cost = search(program.read_text(), system, Claude(a.model, a.effort), judge, out, a.evals, a.usd, a.seed,
                        review=review, stop_pct=a.stop_pct or None, patience=a.patience or None,
                        n_hypotheses=a.hypotheses, beam=a.beam, explore=a.explore, max_repairs=a.max_repairs)
    code, metrics, retimed = retime(out / "trace.jsonl", a.retime) if a.retime else (None, None, [])
    best = tree.best()
    code, metrics, tuned = _tune(code or best.code, metrics or best.metrics, judge, a.tune_points, task.size_expr)
    (out / "best_program.py").write_text(code or best.code)
    (out / "best_metrics.json").write_text(json.dumps(metrics or best.metrics, indent=1))
    models = [{"name": a.model, "role": "coder", "effort": a.effort}] + (
        [{"name": a.judge_model, "role": "judge", "effort": a.effort}] if review else [])
    rec = record(task, sets, models, a.evals, a.device,
                 out / "trace.jsonl", out / "best_program.py", metrics or best.metrics, retimed=retimed,
                 tool="kernels.search", cost=cost, extra={"tuned": tuned})
    print(json.dumps({"best": metrics or best.metrics, "cost": cost, "nodes": len(tree.nodes), "out": str(out),
                      "ledger": rec["id"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
