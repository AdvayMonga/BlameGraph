"""lab kernels search --task T [--shapes SHAPES.json | --captures DIR ...] --out DIR [--initial PROGRAM.py]
                          [--samples DIR --shared-weights FILE]   (T = region:FIRST:LAST: its captures, its holder)
                          [--facts FILE] [--device cuda] [--seed N] [--ledger ROOT]

The kernel search on one task and one set of shapes (generate/kernels/search/loop.py), judged by the jailed
evaluator. After it, the top programs are re-timed, the winner is read for gaming, and the run goes to the ledger as
one `kernels` record."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from evals.kernels import evaluator as jail
from generate.kernels.evolve.run import _shapes, _target_tolerances, load_task, prepare, record, region_args, retime, setting, system_message, task_args, tolerances
from generate.kernels.library import block as other_implementations
from generate.kernels.library import competitors
from generate.kernels.search import detect
from generate.kernels.search.diversity import Novelty
from generate.kernels.search.evaluate import Evaluator
from generate.kernels.search.llm import Claude
from generate.kernels.search.loop import search
from generate.kernels.search.program import Config

MODELS = ("claude-sonnet-5-5", "claude-sonnet-5-5", "claude-sonnet-5-5", "claude-opus-5-5")   # a generation's calls cycle
DETECTOR = "claude-opus-5-5"
EFFORT = "high"
TIMING_SHAPES = 8
RETIME = 4                   # top programs re-judged 3x; the highest median is promoted


def _stations(task, sets: dict) -> list[str] | None:
    """A region's station names (from its shapes file, else from the engine here), for the libraries block; None for a
    task."""
    if not task.name.startswith("region:"):
        return None
    if sets.get("stations"):
        return list(sets["stations"])
    from generate.kernels.tasks import regions
    from measure.profile.capture import STEP
    try:
        if task.name == STEP:
            return [s.name for s in regions.engine_stations()]
        first, last = task.name.split(":")[1:3]
        return [s.name for s in regions.span(regions.engine_stations(regions.station_set(first)), first, last)]
    except Exception:                            # noqa: BLE001  no engine here: the block shows what it can, nothing
        return None


def _embedder(root: Path):
    try:
        from data.knowledge.search import FastEmbed
        return FastEmbed(root / "index" / "models")
    except Exception:                            # noqa: BLE001  without fastembed only identical programs are refused
        print("no embedder (fastembed): near-duplicate programs are not refused", file=sys.stderr)
        return None


def main(argv: list[str] | None = None, models: tuple[str, ...] = MODELS, retime_k: int = RETIME,
         detector: str | None = DETECTOR) -> int:
    from data.knowledge import ledger
    ap = argparse.ArgumentParser(prog="lab kernels search")
    task_args(ap)
    ap.add_argument("--shapes")
    ap.add_argument("--captures", action="append")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ledger", help="the ledger the run's `kernels` record goes to (default: this repo's)")
    ap.add_argument("--equal-captures", action="store_true", help="each --captures dir weighs the same in the shape counts")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--initial")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--facts", help="a text of further facts for every prompt, appended to the system message")
    a = ap.parse_args(argv)
    task, out = load_task(a), Path(a.out)
    tol = tolerances(_target_tolerances(task))
    sets, program = prepare(task, _shapes(task, a), out, a.device, tol, TIMING_SHAPES, a.initial, region_args(a))
    start = program.read_text()
    facts = [system_message(task), setting(task, sets, a.device, tol), Path(a.facts).read_text().strip() if a.facts else "",
             other_implementations(task.name, stations=_stations(task, sets))]
    clients = {m: Claude(m, EFFORT, tools=True) for m in set(models)}
    cfg = Config(seed=a.seed)
    population, cost = search(start, competitors.starts(task.name, start), "\n\n".join(f for f in facts if f),
                              [clients[m] for m in models], Evaluator(counters=a.device.startswith("cuda")),
                              Novelty(_embedder(ledger.ROOT)), out, cfg)
    spent: list = []
    exclude = detect.detector(Claude(detector, EFFORT), spent) if detector else None
    code, metrics, retimed = retime(out / "trace.jsonl", retime_k, exclude=exclude)
    cost["usd"] = round(cost["usd"] + sum(spent), 4)
    cost["model_calls"] += len(spent)
    cost["eval_s"] = jail.seconds(out / "eval_seconds.log")
    best = max(population, key=lambda p: (p.score, -p.generation))
    if code is None and (any("excluded" in r for r in retimed) or (exclude and exclude(best.code, {
            "child_metrics": best.metrics}))):
        best = population[0]                     # every candidate flagged: nothing is promoted over the start
    (out / "best_program.py").write_text(code or best.code)
    (out / "best_metrics.json").write_text(json.dumps(metrics or best.metrics, indent=1))
    hyp = out / "hypotheses.jsonl"
    root = Path(a.ledger) if a.ledger else None
    rec = record(task, sets, [{"name": m, "role": "judge", "effort": EFFORT, "share": f"{models.count(m)}/{len(models)}"}
                              for m in dict.fromkeys(models)], cfg.generations, a.device, out / "trace.jsonl",
                 out / "best_program.py", metrics or best.metrics, retimed=retimed, tool="kernels.search", cost=cost,
                 root=root, extra={"hypotheses": ledger.put_blob(hyp, root or ledger.ROOT, ".jsonl")
                                   if hyp.stat().st_size else None})
    print(json.dumps({"best": metrics or best.metrics, "cost": cost, "nodes": len(population), "out": str(out),
                      "ledger": rec["id"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
