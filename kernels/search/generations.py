"""The genetic search (decided 2026-10-26): generations inside one call, each a funnel round from several parents.

Generation 0 is the funnel from the start program alone (GEN0 ideas written, GEN0_KEEP to the full exam). Every later
generation samples `parents` programs from the population (the POPULATION best correct, scored programs still open;
the best always among them, the rest drawn without replacement with weight = percentile reward + FLOOR, so a weaker
program keeps a chance: soft fitness-weighted selection), gives each a quota of `gen_size / parents` ideas, and sends
the fastest `keep` of the round to the full exam, which join the tree under their own parent. A generation in which
fewer than SHRINK_BELOW of the examined programs beat their parent makes the next one SHRINK as large (ideas, not
parents). Stops: `generations` done, `evals` programs judged, `usd` spent on models, the GPU cap, the best program at
`stop_pct` of its roofline, or `patience` generations with no new best. Why many small generations: Kevin's 16 x 8
beat 128 x 1 (1.10x against 0.65x) and ShinkaEvolve's archive of 40 under 150 evaluations; why weighted parents
rather than MCTS: every tree of the 2026-10-24/25 runs stayed one level deep.

With `ideas` (decided 2026-10-27, after K-Search; an A/B arm until it wins one) a generation's slots split three
ways: persistence (the idea table's best active ideas refined again, kernels/search/ideas.py), exploration (sub-ideas
the table's edit call added, then new ideas on fitness-drawn parents) and immigrants (IMMIGRANTS ideas naming
whole-program changes on the start program or, every other generation, a random program outside the parents: for
diversity, as FunSearch's and AlphaEvolve's islands reset theirs). `wall_s` caps the search's wall-clock.

Same trace.jsonl and hypotheses.jsonl as kernels/search/loop.py's search, so retime, record and the archive read it."""
from __future__ import annotations

import json
import math
import random
import time
import uuid
from pathlib import Path
from typing import Callable

from kernels.search.funnel import Funnel, generation
from kernels.search.loop import trace_row
from kernels.search.tree import Node, Tree

GEN0, GEN0_KEEP = 40, 12     # the first generation, from the start program alone
POPULATION = 32              # programs parents are drawn from
FLOOR = 0.05                 # added to each percentile reward: the weakest still has a chance
SHRINK, SHRINK_BELOW = 2 / 3, 0.2   # the next generation's size when fewer than this fraction beat their parent
IMMIGRANTS = 4               # with `ideas`: whole-program ideas per generation on one immigrant parent


def parents(tree: Tree, n: int, rng: random.Random) -> list[Node]:
    """The best open program, then up to n - 1 more drawn from the population without replacement by weight."""
    pop = sorted(tree.open(), key=lambda x: (-x.score, x.iteration))[:POPULATION]
    if not pop:
        return []
    out, rest = [pop[0]], pop[1:]
    while rest and len(out) < n:
        w = [tree.reward(x) + FLOOR for x in rest]
        pick = rng.choices(range(len(rest)), weights=w)[0]
        out.append(rest.pop(pick))
    return out


def immigrant(tree: Tree, ps: list[Node], gen: int, rng: random.Random) -> Node:
    """The start program, or on odd generations a random correct program outside this generation's parents."""
    others = [n for n in tree.open() if n.id not in {p.id for p in ps} and n.parent is not None]
    return rng.choice(others) if gen % 2 and others else tree.root()


def search(initial: str, system: str, funnel: Funnel, judge: Callable[[str], dict], out: Path, evals: int,
           usd: float, seed: int = 0, gen_size: int = 24, keep: int = 8, n_parents: int = 4, generations: int = 10,
           stop_pct: float | None = None, patience: int | None = 3, gpu_s: float | None = None,
           gpu_used: Callable[[], float] | None = None, crossover: float = 0.0, ideas: bool = False,
           wall_s: float | None = None) -> tuple[Tree, dict]:
    """Returns the tree and the run's cost {usd, model_calls, evals, generations, stopped_*}. `crossover`: the fraction
    of a generation's ideas that combine one parent with another parent's diff (0: none; an A/B arm)."""
    from kernels.search.ideas import EDIT, Table
    out.mkdir(parents=True, exist_ok=True)
    tree, rng, judged, table = Tree(seed), random.Random(seed), 0, Table()
    funnel.usd_cap = usd if funnel.usd_cap is None else funnel.usd_cap
    funnel.gpu_s_cap, funnel.gpu_used = gpu_s, gpu_used
    funnel.deadline = time.monotonic() + wall_s if wall_s else None

    def out_of_budget() -> bool:
        return bool((gpu_s is not None and gpu_used and gpu_used() >= gpu_s)
                    or (funnel.deadline is not None and time.monotonic() >= funnel.deadline))

    def near() -> bool:
        return bool(stop_pct and (tree.best().pct_of_roofline or 0) >= stop_pct)
    with open(out / "trace.jsonl", "w") as trace, open(out / "hypotheses.jsonl", "w") as hyplog:
        def keep_node(node: Node, cost: float) -> None:
            tree.add(node)
            trace.write(json.dumps(trace_row(node, cost)) + "\n")
            trace.flush()

        def log(row: dict) -> None:
            if ideas:
                table.record(row, row["iteration"])
            hyplog.write(json.dumps(row, default=str) + "\n")
            hyplog.flush()
        r = judge(initial)
        judged += 1
        keep_node(Node(uuid.uuid4().hex[:8], None, initial, r["metrics"], r["artifacts"], 0), 0.0)
        gen, idle, size, best = 0, 0, gen_size, tree.best().score
        while gen <= generations and judged < evals and funnel.spent < usd and not out_of_budget() and not near():
            given, extra, restructure = [], [], set()
            if gen == 0:
                ps, quota, k = [tree.root()], GEN0, GEN0_KEEP
            elif not ideas:
                ps, quota, k = parents(tree, n_parents, rng), max(1, math.ceil(size / n_parents)), keep
            else:                                # persistence, sub-ideas, exploration, an immigrant
                given = table.persistent(tree) + table.take_pending(tree)
                ps = parents(tree, max(1, n_parents - 1), rng)
                imm = immigrant(tree, ps, gen, rng)
                ps, restructure = ps + ([imm] if imm.id not in {p.id for p in ps} else []), {imm.id}
                explore = max(len(ps), size - len(given) - IMMIGRANTS)
                quota, k = max(1, math.ceil(explore / len(ps))), keep
                extra = [tree.nodes[h["parent_id"]] for h in given if h["parent_id"] not in {p.id for p in ps}]
                extra = list({n.id: n for n in extra}.values())
            if not ps:
                break
            kids = generation(funnel, tree, ps, system, gen, keep_node, log, quota=quota, keep=k, rotate=gen > 0,
                              crossover=round(crossover * size) if gen > 0 else 0, spread=gen == 0, given=given,
                              restructure=restructure, extra=extra)
            if ideas and not out_of_budget():    # the table's edit: close ideas, add sub-ideas (K-Search's Prune/Insert)
                from kernels.search.funnel import Spent
                try:
                    done = table.edit(funnel._ask(funnel.models.judge, system + "\n\n" + EDIT, table.view()), tree)
                except Spent:
                    done = []
                (out / "ideas.json").write_text(json.dumps({**table.dump(), "last_edit": done}, indent=1, default=str))
            judged += len(kids)
            improved = sum(c.score > tree.nodes[c.parent].score for c in kids)
            size = max(n_parents, round(gen_size * SHRINK)) if kids and improved / len(kids) < SHRINK_BELOW else gen_size
            idle = 0 if tree.best().score > best else idle + 1
            best = max(best, tree.best().score)
            gen += 1
            if patience and idle >= patience:
                break
    stalled = bool(patience and idle >= patience and not near())
    return tree, {"usd": round(funnel.spent, 4), "model_calls": funnel.calls, "evals": judged, "generations": gen,
                  "stopped_near_roofline": near(), "stopped_no_progress": stalled,
                  "stopped_wall_clock": bool(funnel.deadline is not None and time.monotonic() >= funnel.deadline)}
