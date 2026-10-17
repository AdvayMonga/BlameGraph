"""python -m kernels.search --task T [--shapes SHAPES.json | --captures DIR ...] --out DIR [--evals N] [--usd X]
                          [--model CODER] [--judge-model JUDGE|none] [--effort E] [--[no-]ncu] [--device cuda]
                          [--initial PROGRAM.py] [--retime K]

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
from kernels.search.llm import Claude
from kernels.search.loop import cascade, search
from kernels.tasks import TASKS, load


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
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    a.ncu = a.device.startswith("cuda") if a.ncu is None else a.ncu
    task = load(a.task)
    out = Path(a.out)
    sets, program = prepare(task, _shapes(task, a), out, a.device, tolerances(_target_tolerances()),
                            a.timing_shapes, a.initial)
    review = None if a.judge_model == "none" else Claude(a.judge_model, a.effort)
    tree, cost = search(program.read_text(), system_message(task), Claude(a.model, a.effort),
                        cascade(evaluator.judge_stage, counters=a.ncu), out, a.evals, a.usd, a.seed, review=review)
    code, metrics, retimed = retime(out / "trace.jsonl", a.retime) if a.retime else (None, None, [])
    best = tree.best()
    (out / "best_program.py").write_text(code or best.code)
    (out / "best_metrics.json").write_text(json.dumps(metrics or best.metrics, indent=1))
    models = [{"name": a.model, "role": "coder", "effort": a.effort}] + (
        [{"name": a.judge_model, "role": "judge", "effort": a.effort}] if review else [])
    rec = record(task, sets, models, a.evals, a.device,
                 out / "trace.jsonl", out / "best_program.py", metrics or best.metrics, retimed=retimed,
                 tool="kernels.search", cost=cost)
    print(json.dumps({"best": metrics or best.metrics, "cost": cost, "nodes": len(tree.nodes), "out": str(out),
                      "ledger": rec["id"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
