"""python -m kernels.search --task T [--shapes SHAPES.json | --captures DIR ...] --out DIR [--evals N] [--usd X]
                          [--samples DIR --shared-weights FILE]   (T = region:FIRST:LAST: its captures, its holder)
                          [--model CODER] [--judge-model JUDGE|none] [--effort E] [--[no-]ncu] [--device cuda]
                          [--initial PROGRAM.py] [--retime K] [--tune-points N] [--generations N --gen-size N
                          --parents N] [--no-types] [--mcts] [--facts FILE] [--archive FILE]

Our own layer-1 generator on one kernel task: by default the genetic search (kernels/search/generations.py: generations
of rounds from fitness-sampled parents); with --mcts one round per MCTS-chosen parent (kernels/search/
loop.py). Judged by kernels.evolve's evaluator exactly as `python -m kernels.evolve run` is: the
same run dir, selection config (seen shapes only), re-timed promotion and `kernels` ledger record, so the generators
compare like for like. Model calls cost money; --usd caps them."""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Callable

import torch

from kernels.evolve import evaluator
from kernels.evolve.run import (_shapes, _target_tolerances, prepare, record, region_args, retime, setting,
                               system_message, task_args, tolerances)
from kernels.search import detect
from kernels.search.llm import Claude
from kernels.search.loop import cascade, search
from kernels.search.tree import UCT_C
from kernels.tasks import load


TUNE_FACT = ("A program may define, at module level, `CONFIG` (a dict of the launch parameters its kernel uses) and "
             "`TUNABLES` (for each CONFIG key, a list of values). Before a program's full exam, up to {points} points "
             "of that grid (its current CONFIG first, the rest a seeded sample) are each loaded with CONFIG rewritten "
             "in its source, checked on the most-called seen shape and timed on every seen shape. The program is "
             "measured with the fastest point written into CONFIG; where different points are fastest in different "
             "size regions, a dispatcher (between `# TUNED-DISPATCH-START` and `# TUNED-DISPATCH-END`) sets CONFIG per "
             "call from the call's size instead, and is removed before the grid is tuned again. Children start from "
             "the tuned program.")


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
    if review is None:
        raise SystemExit("--funnel needs a judge (--judge-model)")
    library = archive.load(Path(a.archive)) if a.archive else []      # every task's ideas: the memory
    second = Claude(a.second_judge_model, a.effort, tools=a.judge_tools) if a.second_judge_model != "none" else None
    mix = (review,) * (a.second_judge_share - 1) + (second,) if second and a.second_judge_share > 1 else ()
    return Funnel(Models(review, mix), staged(evaluator.judge_stage), task.name,
                  Path(a.ptx).read_text() if a.ptx else None, a.per_call, a.parallel, a.ncu, a.precheck,
                  workers=a.workers, tune=bool(a.tune_points), size_expr=task.size_expr, library=library)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="kernels.search")
    task_args(ap)
    ap.add_argument("--shapes")
    ap.add_argument("--captures", action="append")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ledger", help="the ledger the run's `kernels` record goes to (default: this repo's)")
    ap.add_argument("--equal-captures", action="store_true",
                    help="each --captures dir (one traffic, e.g. a scored regime) weighs the same in the shape counts")
    ap.add_argument("--evals", type=int, default=30, help="with --mcts or no judge: programs judged, the root included")
    ap.add_argument("--usd", type=float, default=10.0, help="cap on model spend")
    ap.add_argument("--model", default="claude-opus-5-5", help="the model alone, with --judge-model none")
    ap.add_argument("--judge-model", default="claude-opus-5-5", help="the model making each change; `none`: --model alone")
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
                    help="grid points of a program's TUNABLES timed before each of its measurements; 0: none")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--uct-c", type=float, default=UCT_C,
                    help="UCT exploration constant on rewards in [0, 1] (KernelPro's sqrt 2 is on ln speedup)")
    ap.add_argument("--patience", type=int, default=16,
                    help="with --mcts: stop after this many judged programs with no new best; 0: never")
    ap.add_argument("--mcts", action="store_true",
                    help="one parent per round, chosen by MCTS (the 2026-10-24 A/B's search), instead of generations")
    ap.add_argument("--generations", type=int, default=10, help="generations after generation 0 (the start alone)")
    ap.add_argument("--gen-size", type=int, default=24, help="ideas written per generation, over its parents")
    ap.add_argument("--parents", type=int, default=4, help="parents per generation, drawn by fitness")
    ap.add_argument("--types", action=argparse.BooleanOptionalAction, default=True,
                    help="split each generation's ideas over refine, persist, fresh and crossover (kernels/search/"
                         "generations.py); --no-types: refine only")
    ap.add_argument("--second-judge-model", default="claude-sonnet-5-5",
                    help="a second model making changes, one call in --second-judge-share; `none`: the judge alone")
    ap.add_argument("--second-judge-share", type=int, default=4, help="every nth call goes to the second judge")
    ap.add_argument("--per-call", type=int, default=20, help="changes per expansion with --mcts")
    ap.add_argument("--facts", help="a text of further facts for every prompt (e.g. the kernel's share of the engine's "
                                    "time and the regimes it is scored under), appended to the system message")
    ap.add_argument("--parallel", type=int, default=4, help="correctness checks run at once")
    ap.add_argument("--workers", type=int, default=8, help="the funnel's model calls run at once")
    ap.add_argument("--archive", help="an exported idea archive (python -m kernels.archive export)")
    ap.add_argument("--ptx", help="the PTX comparison's feature cards (text)")
    ap.add_argument("--precheck", action=argparse.BooleanOptionalAction, default=True,
                    help="the funnel's no-GPU build check and compiler report before stage 1")
    a = ap.parse_args(argv)
    a.ncu = a.device.startswith("cuda") if a.ncu is None else a.ncu
    task = load(a.task, **region_args(a))
    out = Path(a.out)
    tol = tolerances(_target_tolerances())
    sets, program = prepare(task, _shapes(task, a), out, a.device, tol, a.timing_shapes, a.initial, region_args(a))
    review = None if a.judge_model == "none" else Claude(a.judge_model, a.effort, tools=a.judge_tools)
    if a.tune_points:                            # the evaluator's tune stage reads its grid size from the config
        cfg = json.loads(Path(os.environ["KERNEL_EVAL_CONFIG"]).read_text())
        Path(os.environ["KERNEL_EVAL_CONFIG"]).write_text(json.dumps({**cfg, "tune_points": a.tune_points}))
    judge = cascade(evaluator.judge_stage, counters=a.ncu, tuned=bool(a.tune_points), size_expr=task.size_expr)
    system = (system_message(task) + "\n\n" + setting(task, sets, a.device, tol)
              + ("\n\n" + TUNE_FACT.format(points=a.tune_points) if a.tune_points else "")
              + ("\n\n" + Path(a.facts).read_text().strip() if a.facts else ""))
    fun = _funnel(a, task, review) if review is not None else None
    if fun is not None and not a.mcts:
        from kernels.search.generations import search as evolve
        tree, cost = evolve(program.read_text(), system, fun, judge, out, a.usd, a.seed, a.gen_size, a.parents,
                            a.generations, types=a.types)
    else:
        tree, cost = search(program.read_text(), system, Claude(a.model, a.effort), judge, out, a.evals, a.usd, a.seed,
                            patience=a.patience or None, uct_c=a.uct_c, funnel=fun)
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
    cost["eval_s"] = evaluator.seconds(out / "eval_seconds.log")
    (out / "best_program.py").write_text(code or best.code)
    (out / "best_metrics.json").write_text(json.dumps(metrics or best.metrics, indent=1))
    models = ([{"name": a.judge_model, "role": "judge", "effort": a.effort}] if review else
              [{"name": a.model, "role": "coder", "effort": a.effort}]) + (
        [{"name": a.second_judge_model, "role": "judge", "effort": a.effort, "share": f"1/{a.second_judge_share}"}]
        if fun is not None and fun.models.judges else [])
    rec = record(task, sets, models, a.evals, a.device,
                 out / "trace.jsonl", out / "best_program.py", metrics or best.metrics, retimed=retimed,
                 tool="kernels.search", cost=cost, root=Path(a.ledger) if a.ledger else None,
                 extra={"hypotheses": _blob(out / "hypotheses.jsonl", a.ledger)})
    print(json.dumps({"best": metrics or best.metrics, "cost": cost, "nodes": len(tree.nodes), "out": str(out),
                      "ledger": rec["id"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
