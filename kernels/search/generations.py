"""The genetic search (decided 2026-10-26): generations inside one call, each a funnel round from several parents.

Generation 0 is the funnel from the start program alone (GEN0 ideas written, GEN0_KEEP to the full exam). Every later
generation splits its `gen_size` ideas over four idea types (decided 2026-10-29), each with its own judge calls:
  refine     new ideas on `parents` programs drawn from the population (the POPULATION best correct, scored programs
             still open; the best always among them, the rest drawn without replacement with weight = percentile
             reward + FLOOR, so a weaker program keeps a chance: soft fitness-weighted selection)
  persist    take a line's last change further: every child that beat its parent heads a line, which moves to a
             persist child that beats it and closes after LINE_MISSES generations of persist children without one
             (K-Search's persistence); the best-scoring open lines get the slots
  fresh      a different overall structure, half from the start program and half from a random program of the
             population
  crossover  one parent shown another parent's diff, one idea per call
The split starts at TYPES and adapts: TYPE_FLOOR ideas each, the rest by each type's base share times its rate of
children that beat the best score their generation started from (Laplace-smoothed over the run); a type with nothing
to work on (no open line, one parent) gives its ideas to the others. The `keep` full exams are shared out the same
way, so each type is cut on its own fastest. Every judge call gets the memory (kernels/search/memory.py), rebuilt
before each generation. A generation in which fewer than SHRINK_BELOW of the examined programs beat their parent
makes the next one SHRINK as large (ideas, not parents). Stops: `generations` done, `evals` programs judged, `usd`
spent on models, the GPU cap, the wall-clock cap (`wall_s`), or `patience` generations with no new best. Why many
small generations: Kevin's 16 x 8 beat 128 x 1 (1.10x against 0.65x) and ShinkaEvolve's archive of 40 under 150
evaluations; why weighted parents rather than MCTS: every tree of the 2026-10-24/25 runs stayed one level deep.

Same trace.jsonl and hypotheses.jsonl as kernels/search/loop.py's search, so retime, record and the archive read it."""
from __future__ import annotations

import json
import math
import random
import time
import uuid
from pathlib import Path
from typing import Callable

from kernels.search import memory
from kernels.search.funnel import Funnel, Slot, generation
from kernels.search.loop import trace_row
from kernels.search.tree import Node, Tree

GEN0, GEN0_KEEP = 40, 12     # the first generation, from the start program alone
POPULATION = 32              # programs parents are drawn from
FLOOR = 0.05                 # added to each percentile reward: the weakest still has a chance
SHRINK, SHRINK_BELOW = 2 / 3, 0.2   # the next generation's size when fewer than this fraction beat their parent
TYPES = {"refine": 8, "persist": 6, "fresh": 6, "crossover": 4}   # a generation's ideas by type, before adapting
TYPE_FLOOR = 2               # ideas every available type keeps, however it has done
LINE_MISSES = 3              # generations of persist children without a better one before a line closes


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


def shares(total: int, weights: dict[str, float]) -> dict[str, int]:
    """`total` split in proportion to `weights` by largest remainder (ties to the first)."""
    w = sum(weights.values())
    if total <= 0 or w <= 0:
        return {k: 0 for k in weights}
    raw = {k: total * v / w for k, v in weights.items()}
    out = {k: math.floor(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: -(raw[k] - out[k]))[:total - sum(out.values())]:
        out[k] += 1
    return out


def split(size: int, stats: dict[str, list[int]], available: set[str]) -> dict[str, int]:
    """Each available type's ideas: TYPE_FLOOR each (fewer when `size` is small), the rest by base share x the type's
    rate of new bests, (new bests + 1) / (examined + 2)."""
    kinds = [t for t in TYPES if t in available]
    floor = min(TYPE_FLOOR, size // len(kinds))
    rate = {t: (stats[t][1] + 1) / (stats[t][0] + 2) for t in kinds}
    rest = shares(size - floor * len(kinds), {t: max(TYPES[t] - TYPE_FLOOR, 1) * rate[t] for t in kinds})
    return {t: floor + rest[t] for t in kinds}


def slots(tree: Tree, ps: list[Node], q: dict[str, int], lines: dict[str, int], rng: random.Random) -> list[Slot]:
    """A generation's judge calls from its split `q`: refine over the parents, persist over the best open lines (about
    3 ideas each), fresh half from the start and half from a random program, one crossover call per idea."""
    out = [Slot(p, "refine", n) for p, n in zip(ps, shares(q.get("refine", 0), {p.id: 1 for p in ps}).values()) if n]
    heads = sorted((tree.nodes[h] for h in lines), key=lambda x: -x.score)[:max(1, q.get("persist", 0) // 3)]
    out += [Slot(h, "persist", n) for h, n in zip(heads, shares(q.get("persist", 0), {h.id: 1 for h in heads}).values())
            if n]
    fresh, pop = q.get("fresh", 0), [x for x in sorted(tree.open(), key=lambda x: -x.score)[:POPULATION] if x.parent]
    other = fresh // 2 if pop else 0
    out += [s for s in (Slot(tree.root(), "fresh", fresh - other), Slot(rng.choice(pop), "fresh", other) if other
                        else None) if s and s.n]
    out += [Slot(ps[k % len(ps)], "crossover", 1, ps[(k + 1) % len(ps)]) for k in range(q.get("crossover", 0))]
    return out


def advance(tree: Tree, lines: dict[str, int], plan: list[Slot], kids: list[Node]) -> dict[str, int]:
    """The open lines after a generation: a head whose persist child beat it gives way to that child; one that got
    persist slots and no better child counts a miss and closes at LINE_MISSES; every child that beat its parent heads
    a line."""
    lines = dict(lines)
    for s in plan:
        if s.kind != "persist" or s.parent.id not in lines:
            continue
        if any(c.parent == s.parent.id and c.kind == "persist" and c.score > s.parent.score for c in kids) \
                or lines[s.parent.id] + 1 >= LINE_MISSES:
            del lines[s.parent.id]
        else:
            lines[s.parent.id] += 1
    return lines | {c.id: 0 for c in kids if c.score > tree.nodes[c.parent].score}


def search(initial: str, system: str, funnel: Funnel, judge: Callable[[str], dict], out: Path, evals: int,
           usd: float, seed: int = 0, gen_size: int = 24, keep: int = 8, n_parents: int = 4, generations: int = 10,
           patience: int | None = 3, gpu_s: float | None = None,
           gpu_used: Callable[[], float] | None = None, types: bool = True,
           wall_s: float | None = None) -> tuple[Tree, dict]:
    """Returns the tree and the run's cost {usd, model_calls, evals, generations, types, stopped_*}. `types=False`:
    every idea is a refine idea (the search before 2026-10-29; an A/B arm)."""
    out.mkdir(parents=True, exist_ok=True)
    tree, rng, judged = Tree(seed), random.Random(seed), 0
    funnel.usd_cap = usd if funnel.usd_cap is None else funnel.usd_cap
    funnel.gpu_s_cap, funnel.gpu_used = gpu_s, gpu_used
    funnel.deadline = time.monotonic() + wall_s if wall_s else None
    library = funnel.library or funnel.archive
    general, rows = memory.general(library, funnel.task), []
    stats = {t: [0, 0] for t in TYPES}           # per type: children examined, children that beat their generation's start
    lines: dict[str, int] = {}                   # each open line's head -> generations of persist children with no better

    def out_of_budget() -> bool:
        return bool((gpu_s is not None and gpu_used and gpu_used() >= gpu_s)
                    or (funnel.deadline is not None and time.monotonic() >= funnel.deadline))

    with open(out / "trace.jsonl", "w") as trace, open(out / "hypotheses.jsonl", "w") as hyplog:
        def keep_node(node: Node, cost: float) -> None:
            tree.add(node)
            trace.write(json.dumps(trace_row(node, cost)) + "\n")
            trace.flush()

        def log(row: dict) -> None:
            rows.append(row)
            hyplog.write(json.dumps(row, default=str) + "\n")
            hyplog.flush()
        r = judge(initial)
        judged += 1
        keep_node(Node(uuid.uuid4().hex[:8], None, r.get("code", initial), r["metrics"], r["artifacts"], 0), 0.0)
        gen, idle, size, best = 0, 0, gen_size, tree.best().score
        while gen <= generations and judged < evals and funnel.spent < usd and not out_of_budget():
            funnel.memory = (general, memory.task(library, funnel.task, rows))
            extra, keep_by, ps = {}, None, [tree.root()] if gen == 0 else parents(tree, n_parents, rng)
            lines = {h: m for h, m in lines.items() if tree.expandable(tree.nodes[h])}
            if not ps:
                break
            if gen == 0:
                plan, kw = [Slot(ps[0], "start", GEN0)], {"keep": GEN0_KEEP, "rotate": False, "spread": True}
            elif types:
                available = {"refine", "fresh"} | ({"persist"} if lines else set()) | ({"crossover"} if len(ps) > 1
                                                                                      else set())
                q = split(size, stats, available)
                plan, keep_by = slots(tree, ps, q, lines, rng), shares(keep, {t: n for t, n in q.items() if n})
                kw, extra = {"keep": keep}, {"types": q, "keep_by": keep_by, "open_lines": len(lines)}
            else:
                quota = max(1, math.ceil(size / n_parents))
                plan, kw = [Slot(p, "refine", quota) for p in ps], {"keep": keep}
            start = tree.best().score
            kids = generation(funnel, tree, ps, system, gen, keep_node, log, quota=0, slots=plan, keep_by=keep_by,
                              extra=extra, **kw)
            judged += len(kids)
            for c in kids:
                if c.kind in stats:
                    stats[c.kind][0] += 1
                    stats[c.kind][1] += c.score > start
            lines = advance(tree, lines, plan, kids)
            improved = sum(c.score > tree.nodes[c.parent].score for c in kids)
            size = max(n_parents, round(gen_size * SHRINK)) if kids and improved / len(kids) < SHRINK_BELOW else gen_size
            idle = 0 if tree.best().score > best else idle + 1
            best = max(best, tree.best().score)
            gen += 1
            if patience and idle >= patience:
                break
    stalled = bool(patience and idle >= patience)
    return tree, {"usd": round(funnel.spent, 4), "model_calls": funnel.calls, "evals": judged, "generations": gen,
                  "types": {t: {"examined": e, "new_best": b} for t, (e, b) in stats.items()},
                  "stopped_no_progress": stalled,
                  "stopped_wall_clock": bool(funnel.deadline is not None and time.monotonic() >= funnel.deadline)}
