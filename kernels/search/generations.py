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

Same trace.jsonl and hypotheses.jsonl as kernels/search/loop.py's search, so retime, record and the archive read it."""
from __future__ import annotations

import json
import math
import random
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


def search(initial: str, system: str, funnel: Funnel, judge: Callable[[str], dict], out: Path, evals: int,
           usd: float, seed: int = 0, gen_size: int = 24, keep: int = 8, n_parents: int = 4, generations: int = 10,
           stop_pct: float | None = None, patience: int | None = 3, gpu_s: float | None = None,
           gpu_used: Callable[[], float] | None = None, crossover: float = 0.0) -> tuple[Tree, dict]:
    """Returns the tree and the run's cost {usd, model_calls, evals, generations, stopped_*}. `crossover`: the fraction
    of a generation's ideas that combine one parent with another parent's diff (0: none; an A/B arm)."""
    out.mkdir(parents=True, exist_ok=True)
    tree, rng, judged = Tree(seed), random.Random(seed), 0
    funnel.usd_cap = usd if funnel.usd_cap is None else funnel.usd_cap
    funnel.gpu_s_cap, funnel.gpu_used = gpu_s, gpu_used

    def gpu_out() -> bool:
        return bool(gpu_s is not None and gpu_used and gpu_used() >= gpu_s)

    def near() -> bool:
        return bool(stop_pct and (tree.best().pct_of_roofline or 0) >= stop_pct)
    with open(out / "trace.jsonl", "w") as trace, open(out / "hypotheses.jsonl", "w") as hyplog:
        def keep_node(node: Node, cost: float) -> None:
            tree.add(node)
            trace.write(json.dumps(trace_row(node, cost)) + "\n")
            trace.flush()

        def log(row: dict) -> None:
            hyplog.write(json.dumps(row, default=str) + "\n")
            hyplog.flush()
        r = judge(initial)
        judged += 1
        keep_node(Node(uuid.uuid4().hex[:8], None, initial, r["metrics"], r["artifacts"], 0), 0.0)
        gen, idle, size, best = 0, 0, gen_size, tree.best().score
        while gen <= generations and judged < evals and funnel.spent < usd and not gpu_out() and not near():
            if gen == 0:
                ps, quota, k = [tree.root()], GEN0, GEN0_KEEP
            else:
                ps, quota, k = parents(tree, n_parents, rng), max(1, math.ceil(size / n_parents)), keep
            if not ps:
                break
            kids = generation(funnel, tree, ps, system, gen, keep_node, log, quota=quota, keep=k, rotate=gen > 0,
                              crossover=round(crossover * size) if gen > 0 else 0, spread=gen == 0)
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
                  "stopped_near_roofline": near(), "stopped_no_progress": stalled}
