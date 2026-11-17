"""lab kernels search --task T [--shapes SHAPES.json | --captures DIR ...] --out DIR [--initial PROGRAM.py]
                          [--samples DIR --shared-weights FILE]   (T = region:FIRST:LAST: its captures, its holder)
                          [--facts FILE] [--device cuda] [--seed N] [--ledger ROOT]

The kernel search on one task (loop.py), then promotion: the top programs take the full check, are read for gaming
and are re-timed. The run is a `kernels` ledger record after every generation (`finished: false`) and at the end,
all under one run id; run again with the same --out, a search that stopped goes on from its last generation."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from lab.kernels.eval import evaluator as jail
from lab.kernels.evolve.run import _shapes, _target_tolerances, load_task, prepare, record, region_args, retime, setting, system_message, task_args, tolerances
from lab.kernels.library import block as other_implementations
from lab.kernels.library import competitors
from lab.kernels.search import detect
from lab.kernels.search.diversity import Novelty, Similar
from lab.kernels.search.evaluate import WINNER, Evaluator
from lab.kernels.search.llm import Claude
from lab.kernels.search.loop import opened, search
from lab.kernels.search.program import Config

NAMER = "claude-opus-5-5"    # names and reviews a generation's changes
WRITER = "claude-sonnet-5-5"  # makes each change
DETECTOR = "claude-opus-5-5"
EFFORT = "high"
TIMING_SHAPES = 8
RETIME = 4                   # top programs re-judged 3x; the highest median is promoted
def _exclude(evaluator: Evaluator, gaming):
    """`exclude` for retime: why a program is passed over at promotion (it fails the full check on 20 seeds, or the
    detector flags it), or None."""
    def check(code: str, row: dict) -> str | None:
        why = evaluator.fails(code, WINNER)
        return f"failed the winner's check: {why}" if why else gaming(code, row) if gaming else None
    return check


def _stations(task, sets: dict) -> list[str] | None:
    """A region's station names (from its shapes file, else from the engine here), for the libraries block; None for a
    task."""
    if not task.name.startswith("region:"):
        return None
    if sets.get("stations"):
        return list(sets["stations"])
    from lab.kernels.tasks import regions
    from lab.profile.capture import STEP
    try:
        eng = regions.load_engine()
        if task.name == STEP:
            return [s.name for s in regions.engine_stations(eng, eng.step)]
        first, last = task.name.split(":")[1:3]
        return [s.name for s in regions.span(regions.engine_stations(eng, eng.which(first)), first, last)]
    except Exception:                            # noqa: BLE001  no engine here: the block shows what it can, nothing
        return None


def _embedder(root: Path):
    try:
        from lab.records.search import FastEmbed
        return FastEmbed(root / "index" / "models")
    except Exception:                            # noqa: BLE001
        print("no embedder (fastembed): close ideas are not matched", file=sys.stderr)
        return None


def _past(root: Path, task: str, embedder) -> list:
    """The task's distinct ideas from earlier runs on the ledger at `root` (the idea archive)."""
    from lab.kernels import archive
    return [e for e in archive.build(root, embedder) if e.task == task]


def main(argv: list[str] | None = None, namer: str = NAMER, writer: str = WRITER, retime_k: int = RETIME,
         detector: str | None = DETECTOR) -> int:
    from lab.records import ledger
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
    root = Path(a.ledger) if a.ledger else ledger.ROOT
    embedder, cfg = _embedder(ledger.ROOT), Config(seed=a.seed)
    evaluator, calls = Evaluator(counters=a.device.startswith("cuda")), opened(out)
    judge = Claude(detector, EFFORT) if detector else None
    gaming = detect.detector(lambda s, u: calls(judge, "detect", None, s, u)[0]) if detector else None
    run = out / "run_id"
    if not run.exists():
        run.write_text(f"kernels-{time.strftime('%Y%m%d-%H%M%S')}")

    def save(code: str, metrics: dict, done: int, finished: bool, retimed: list | None = None) -> dict:
        """The run so far as a `kernels` ledger record."""
        (out / "best_program.py").write_text(code)
        (out / "best_metrics.json").write_text(json.dumps(metrics, indent=1))
        blobs = {k: ledger.put_blob(out / f"{k}.jsonl", root, ".jsonl")
                 if (out / f"{k}.jsonl").exists() and (out / f"{k}.jsonl").stat().st_size else None
                 for k in ("hypotheses", "ideas", "calls", "texts")}
        reflection = out / "reflection.txt"
        cost = {"usd": round(calls.usd, 4), "model_calls": calls.n, "evals": evaluator.evals, "generations": done,
                "eval_s": jail.seconds(out / "eval_seconds.log")}
        return record(task, sets, [{"name": namer, "role": "namer", "effort": EFFORT},
                                   {"name": writer, "role": "writer", "effort": EFFORT}], cfg.generations, a.device,
                      out / "trace.jsonl", out / "best_program.py", metrics, retimed=retimed, tool="kernels.search",
                      cost=cost, root=root, run=run.read_text(),
                      extra={**blobs, "finished": finished,
                             "reflection": reflection.read_text() if reflection.exists() else ""})

    def each(done: int) -> None:
        rows = [json.loads(x) for x in (out / "trace.jsonl").read_text().splitlines()]
        top = max(rows, key=lambda r: (r["child_metrics"].get("correct") == 1.0,
                                       r["child_metrics"].get("combined_score") or 0.0, -r["iteration"]))
        save(top["child_code"], top["child_metrics"], done, False)
    population, ran = search(start, competitors.starts(task.name, start), "\n\n".join(f for f in facts if f),
                           Claude(namer, EFFORT, tools=True), Claude(writer, EFFORT), evaluator, Novelty(),
                           Similar(embedder, _past(root, task.name, embedder)), out, cfg, gaming, calls, each)
    exclude = _exclude(evaluator, gaming)
    code, metrics, retimed = retime(out / "trace.jsonl", retime_k, exclude=exclude)
    best = max(population, key=lambda p: (p.score, -p.generation))
    if code is None and (any("excluded" in r for r in retimed) or exclude(best.code, {"child_metrics": best.metrics})):
        best = population[0]                     # every candidate flagged: nothing is promoted over the start
    rec = save(code or best.code, metrics or best.metrics, ran["generations"], True, retimed)
    cost = rec["cost"]
    calls.close()
    print(json.dumps({"best": metrics or best.metrics, "cost": cost, "nodes": len(population), "out": str(out),
                      "ledger": rec["id"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
