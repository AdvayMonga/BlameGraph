"""The search: each generation names its changes in one call, reviews them in a second, and has a writer make each.

Before the first generation the start and the seeds are judged, and one call names public implementations to port
as further starting programs (context.ports), each made by a writer call and judged beside them.

A generation picks its parents (diversity.pick) and drops the ones that fail the full correctness check
(evaluate.Evaluator.check) or that the gaming detector flags (detect.detector). One naming call reflects on the last generation's ideas and names `ideas` changes
(context.name). The ones whose text is close to another idea's are found (diversity.Similar) and one review call
writes the final list, the same change merged and combinations added (context.review). A writer call makes each
change (context.write). Programs that repeat one already written are dropped (diversity.Novelty), the rest are built
and timed (evaluate.Evaluator.evaluate) and join the population. It runs every generation whatever they score.

`<out>/trace.jsonl` holds every judged program, `<out>/hypotheses.jsonl` every generation's ideas and their fates,
`<out>/ideas.jsonl` one row per idea (plan, prediction, program, outcome, reflection), rewritten after every
generation, `<out>/reflection.txt` the model's closing reflection on the run, and `<out>/calls.jsonl` with
`<out>/texts.jsonl` every model call whole (calls.py)."""
from __future__ import annotations

import difflib
import json
import random
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from lab.kernels.search import context, diff, diversity
from lab.kernels.search.calls import Calls, Model
from lab.kernels.search.evaluate import Evaluator, Result
from lab.kernels.search.program import Config, Idea, Program


def new_id() -> str:
    return uuid.uuid4().hex[:8]


def trace_row(p: Program) -> dict:
    return {"iteration": p.generation, "parent_id": p.parent, "child_id": p.id, "child_metrics": p.metrics,
            "child_code": p.code, "artifacts": p.artifacts, "judge": json.dumps(p.plan) if p.plan else "",
            "kind": p.kind}


def idea_row(i: Idea, parent: Program) -> dict:
    """An idea in its generation's row of hypotheses.jsonl (what the idea archive reads)."""
    row = {"idea": {**i.plan, "model": i.model, "parent_id": i.parent, "idea_id": i.id}, "fate": i.fate}
    if i.child:
        row |= {"child_id": i.child.id, "child_score": i.child.score, "improved": i.child.score > parent.score}
    if i.code:
        row["diff"] = "".join(difflib.unified_diff(parent.code.splitlines(True), i.code.splitlines(True), n=2))[:4000]
    return row


def table_row(i: Idea, score: dict) -> dict:
    """An idea's row of ideas.jsonl: everything about it, its texts by their keys in texts.jsonl."""
    kid = i.child
    return {"idea_id": i.id, "generation": i.generation, "parent_id": i.parent, "parent_score": score.get(i.parent),
            "plan": i.plan, "merged_from": i.merged, "writer": i.model, "calls": i.calls, "program": i.program,
            "fate": i.fate, "child_id": kid.id if kid else None, "child_score": kid.score if kid else None,
            "gain": context.outcome(i, score).get("gain"), "metrics": kid.metrics if kid else None,
            "reflection": i.reflection}


def changes(answer: dict, parents: list[Program]) -> list[dict]:
    """An answer's ideas that change one of `parents`."""
    ids = {p.id for p in parents}
    return [p for p in answer.get("ideas") or [] if isinstance(p, dict) and p.get("parent") in ids]


def idea(plan: dict, generation: int, calls: dict) -> Idea:
    """A named change as an idea: its parent and the named changes it is from, apart from the rest of its plan."""
    return Idea(new_id(), plan["parent"], generation, {k: v for k, v in plan.items() if k not in ("parent", "from")},
                plan["from"] if isinstance(plan.get("from"), list) else [], dict(calls))


def reflected(ideas: list[Idea], answer: dict) -> None:
    """Each reflection of an answer onto the idea it names."""
    by = {i.id: i for i in ideas}
    for r in answer.get("reflections") or []:
        if isinstance(r, dict) and r.get("id") in by:
            by[r["id"]].reflection = str(r.get("reflection") or "")


def flagged(programs: list[Program], detect: Callable, workers: int) -> dict[str, str]:
    """{id: why} for the programs the gaming detector flags; each is read once, and one flagged can no longer be a
    parent."""
    new = [p for p in programs if "gaming" not in p.artifacts]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for p, why in zip(new, ex.map(lambda p: detect(p.code, {"child_metrics": p.metrics}), new)):
            p.artifacts["gaming"] = why or "none"
            if why:
                p.metrics["correct"] = 0.0
    return {p.id: p.artifacts["gaming"] for p in programs if p.artifacts["gaming"] != "none"}


def write(calls: Calls, writer: Model, setting: str, idea: Idea, by: dict[str, Program]) -> None:
    """One writer call: the program its change makes of the idea's parent, or why it could not be applied."""
    parent = by[idea.parent]
    borrowed = [by[b] for b in idea.plan.get("borrows") or [] if isinstance(b, str) and b in by and b != parent.id]
    text, idea.calls["write"] = calls(writer, "write", idea.generation,
                                      *context.write(setting, parent, idea.plan, borrowed))
    idea.model = getattr(writer, "model", None)
    idea.code, why = diff.apply(parent.code, text) if text else (None, "the model returned no text")
    if idea.code:
        idea.program = calls.put(idea.code)
    else:
        idea.fate = f"not applied: {why}"


def search(start: str, seeds: dict[str, str], setting: str, namer: Model, writer: Model, evaluator: Evaluator,
           novelty: diversity.Novelty, similar: diversity.Similar, out: Path,
           cfg: Config = Config(), detect: Callable | None = None) -> tuple[list[Program], dict]:
    """The population (the start first) and the run's cost {usd, model_calls, evals, generations}. `seeds`: name ->
    program, judged beside the start. `namer` names the ports and names and reviews a generation's changes, and
    writes the ports (it can read their sources); `writer` makes each change; `detect(code, row)` says why a picked
    parent games the evaluation, or None."""
    out.mkdir(parents=True, exist_ok=True)
    rng, population, ideas, ran = random.Random(cfg.seed), [], [], 0
    calls = Calls(out)
    with open(out / "trace.jsonl", "w") as trace, open(out / "hypotheses.jsonl", "w") as log:
        def keep(code: str, r: Result, generation: int, parent: str | None = None, plan: dict | None = None,
                 kind: str = "") -> Program:
            p = Program(new_id(), parent, code, r.metrics, r.artifacts, generation, plan or {}, kind)
            population.append(p)
            trace.write(json.dumps(trace_row(p)) + "\n")
            trace.flush()
            return p

        def make(gen: int, parents: list[Program], new: list[Idea], model: Model, spent: float, row: dict,
                 kind: str = "") -> None:
            """A generation's ideas written, the repeats refused, the rest judged; its row of hypotheses.jsonl."""
            by = {p.id: p for p in population}
            with ThreadPoolExecutor(max_workers=cfg.workers) as ex:
                list(ex.map(lambda i: write(calls, model, setting, i, by), new))
            for i in new:
                why = novelty.refuse(i.code) if i.code else None
                if why:
                    i.code, i.fate = None, f"not applied: {why}"
            written = [i for i in new if i.code]
            for i, r in zip(written, evaluator.evaluate([i.code for i in written])):
                i.fate = r.fate
                if r.examined:
                    i.child = keep(i.code, r, gen, i.parent, i.plan, kind)
            similar.add(new)
            ideas.extend(new)
            kids = [i.child for i in new if i.child]
            log.write(json.dumps({
                "iteration": gen, "parents": [{"id": p.id, "score": p.score} for p in parents], **row,
                "written": len(written), "examined": len(kids),
                "improved": sum(c.score > by[c.parent].score for c in kids), "usd": round(calls.usd - spent, 4),
                "ideas": [idea_row(i, by[i.parent]) for i in new]}, default=str) + "\n")
            log.flush()
            table()

        def table() -> None:
            score = {p.id: p.score for p in population}
            (out / "ideas.jsonl").write_text(
                "".join(json.dumps(table_row(i, score), default=str) + "\n" for i in ideas))

        first = [start, *seeds.values()]
        for code in first:
            novelty.add(code)
        results = evaluator.evaluate(first)
        root = keep(start, results[0], 0)
        for name, code, r in zip(seeds, first[1:], results[1:]):
            keep(code, r, 0, root.id, {"competitor": name}, "competitor")
        if cfg.ports:
            spent = calls.usd
            text, nid = calls(namer, "ports", 0, *context.ports(setting, root, list(seeds), cfg.ports))
            plans = changes(context.read(text), [root])[:cfg.ports]
            make(0, [root], [idea(p, 0, {"name": nid}) for p in plans], namer, spent,
                 {"named": len(plans), "calls": {"name": nid}}, "competitor")

        for gen in range(1, cfg.generations + 1):
            picked = diversity.pick(population, cfg.parents, rng)
            if not picked:
                break
            spent, dropped = calls.usd, evaluator.check(picked)
            gaming = flagged([p for p in picked if p.id not in dropped], detect, cfg.workers) if detect else {}
            for i in ideas:
                if i.child and i.child.id in dropped:
                    i.fate = f"failed the full check as a parent: {dropped[i.child.id]}"
                elif i.child and i.child.id in gaming:
                    i.fate = f"as a parent, {gaming[i.child.id]}"
            dropped = {**dropped, **gaming}
            parents = [p for p in picked if p.id not in dropped]
            for p in parents:
                evaluator.profile(p)
            plans, final, nid, rid = [], [], None, None
            if parents:
                text, nid = calls(namer, "name", gen,
                                  *context.name(setting, parents, population, ideas, gen, cfg.ideas))
                answer = context.read(text)
                reflected(ideas, answer)
                plans = changes(answer, parents)[:cfg.ideas]
            if plans:
                text, rid = calls(namer, "review", gen, *context.review(
                    setting, parents, population, ideas, plans, similar.match(plans), cfg.merges))
                final = changes(context.read(text), parents)[:cfg.ideas + cfg.merges] or plans   # unreadable: as named
            make(gen, parents, [idea(p, gen, {"name": nid, "review": rid}) for p in final], writer, spent,
                 {"dropped": dropped, "named": len(plans), "reviewed": len(final), "calls": {"name": nid, "review": rid}})
            ran = gen

        if ran:                                  # the last generation's ideas have no naming call after them
            text, _ = calls(namer, "reflect", ran + 1, *context.reflect(setting, population, ideas, ran + 1))
            answer = context.read(text)
            reflected(ideas, answer)
            (out / "reflection.txt").write_text(str(answer.get("run") or ""))
        table()
    calls.close()
    return population, {"usd": round(calls.usd, 4), "model_calls": calls.n, "evals": evaluator.evals,
                        "generations": ran}
