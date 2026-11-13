"""The search: generations of model calls from the best programs so far, every new program judged.

Each generation picks its parents (diversity.pick), makes `ideas` model calls spread over them (context.build), drops
the programs that repeat one already written (diversity.Novelty), judges the rest (evaluate.Evaluator) and adds every
program that took the full exam to the population. It runs every generation whatever they score.

`<out>/trace.jsonl` holds every judged program, `<out>/hypotheses.jsonl` every generation's ideas and their fates."""
from __future__ import annotations

import difflib
import json
import random
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Sequence

from generate.kernels.search import context, diff, diversity
from generate.kernels.search.evaluate import Evaluator, Result
from generate.kernels.search.program import Config, Idea, Program

Model = Callable[[str, str], tuple[str, float]]  # (system, user) -> (text, $)


def new_id() -> str:
    return uuid.uuid4().hex[:8]


def trace_row(p: Program) -> dict:
    return {"iteration": p.generation, "parent_id": p.parent, "child_id": p.id, "child_metrics": p.metrics,
            "child_code": p.code, "artifacts": p.artifacts, "judge": json.dumps(p.plan) if p.plan else "",
            "kind": p.kind}


def idea_row(i: Idea, parent: Program) -> dict:
    row = {"idea": {**i.plan, "model": i.model, "parent_id": i.parent, "idea_id": i.id}, "fate": i.fate}
    if i.child:
        row |= {"child_id": i.child.id, "child_score": i.child.score, "improved": i.child.score > parent.score}
    if i.code:
        row["diff"] = "".join(difflib.unified_diff(parent.code.splitlines(True), i.code.splitlines(True), n=2))[:4000]
    return row


def propose(model: Model, setting: str, parent: Program, population: list[Program], ideas: list[Idea],
            proposed: list[dict], generation: int) -> Idea:
    """One model call: the idea it names and the program its change makes of `parent`."""
    text, usd = model(*context.build(setting, parent, population, ideas, proposed))
    code, why = diff.apply(parent.code, text) if text else (None, "the model returned no text")
    return Idea(new_id(), parent.id, generation, context.read(text), getattr(model, "model", None), usd, code,
                "" if code else f"not applied: {why}")


def search(start: str, seeds: dict[str, str], setting: str, models: Sequence[Model], evaluator: Evaluator,
           novelty: diversity.Novelty, out: Path, cfg: Config = Config()) -> tuple[list[Program], dict]:
    """The population (the start first) and the run's cost {usd, model_calls, evals, generations}. `seeds`: name ->
    program, judged beside the start. `models`: call k of a generation goes to models[k % len(models)]."""
    out.mkdir(parents=True, exist_ok=True)
    rng, population, ideas, ran = random.Random(cfg.seed), [], [], 0
    with open(out / "trace.jsonl", "w") as trace, open(out / "hypotheses.jsonl", "w") as log:
        def keep(code: str, r: Result, generation: int, parent: str | None = None, plan: dict | None = None,
                 kind: str = "") -> Program:
            p = Program(new_id(), parent, code, r.metrics, r.artifacts, generation, plan or {}, kind)
            population.append(p)
            trace.write(json.dumps(trace_row(p)) + "\n")
            trace.flush()
            return p

        first = [start, *seeds.values()]
        for code in first:
            novelty.add(code)
        results = evaluator.evaluate(first)
        root = keep(start, results[0], 0)
        for name, code, r in zip(seeds, first[1:], results[1:]):
            keep(code, r, 0, root.id, {"competitor": name}, "competitor")

        for gen in range(1, cfg.generations + 1):
            parents = diversity.pick(population, cfg.parents, rng)
            if not parents:
                break
            for p in parents:
                evaluator.profile(p)
            by, new = {p.id: p for p in parents}, []
            slots = list(enumerate(parents[k % len(parents)] for k in range(cfg.ideas)))

            def call(slot: tuple[int, Program], seen: list[Idea]) -> Idea:
                k, parent = slot
                return propose(models[k % len(models)], setting, parent, population, ideas,
                               [i.plan for i in seen if i.parent == parent.id], gen)
            with ThreadPoolExecutor(max_workers=cfg.workers) as ex:
                for w in range(0, len(slots), cfg.workers):   # a wave's calls see what the waves before it proposed
                    new += ex.map(lambda slot, seen=list(new): call(slot, seen), slots[w:w + cfg.workers])
            for i in new:
                why = novelty.refuse(i.code, by[i.parent]) if i.code else None
                if why:
                    i.code, i.fate = None, f"not applied: {why}"
            written = [i for i in new if i.code]
            for i, r in zip(written, evaluator.evaluate([i.code for i in written])):
                i.fate = r.fate
                if r.examined:
                    i.child = keep(i.code, r, gen, i.parent, i.plan)
            ideas, ran = ideas + new, gen
            kids = [i.child for i in new if i.child]
            log.write(json.dumps({
                "iteration": gen, "parents": [{"id": p.id, "score": p.score} for p in parents], "named": len(new),
                "written": len(written), "examined": len(kids),
                "improved": sum(c.score > by[c.parent].score for c in kids),
                "usd": round(sum(i.usd for i in new), 4), "ideas": [idea_row(i, by[i.parent]) for i in new]},
                default=str) + "\n")
            log.flush()
    return population, {"usd": round(sum(i.usd for i in ideas), 4), "model_calls": len(ideas),
                        "evals": evaluator.evals, "generations": ran}
