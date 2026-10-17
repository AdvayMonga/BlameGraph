"""Launch parameters chosen by a grid search, not by the model (KernelEvolve, AlphaEvolve's tiling heuristic; also the
2026-10-16 moe_gemm win, mostly tile sizes and stages). A program may define, at module level, `CONFIG` (a dict of the
launch parameters its kernel uses) and `TUNABLES` (for each key, the values to try). The sweep judges points of that
grid like any program, with `CONFIG` rewritten in the source, so the chosen values ship as plain constants and the
engine pays no autotuning at startup."""
from __future__ import annotations

import ast
import itertools
import json
import math
import random
import re
from typing import Callable


def _assign(tree: ast.Module, name: str) -> ast.Assign | None:
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets):
            return n
    return None


def params(code: str) -> tuple[dict, dict] | None:
    """(CONFIG, TUNABLES) when the program declares both as literal dicts whose keys line up, else None."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    c, t = _assign(tree, "CONFIG"), _assign(tree, "TUNABLES")
    if c is None or t is None:
        return None
    try:
        config, tunables = ast.literal_eval(c.value), ast.literal_eval(t.value)
    except ValueError:
        return None
    if not (isinstance(config, dict) and isinstance(tunables, dict) and tunables and set(tunables) <= set(config)
            and all(isinstance(v, list) and v for v in tunables.values())):
        return None
    return config, tunables


def with_config(code: str, config: dict) -> str:
    """The program with its CONFIG assignment replaced by `config`."""
    node = _assign(ast.parse(code), "CONFIG")
    lines = code.splitlines(keepends=True)
    start, end = node.lineno - 1, node.end_lineno
    return "".join(lines[:start]) + f"CONFIG = {config!r}\n" + "".join(lines[end:])


def grid(config: dict, tunables: dict, points: int, seed: int = 0) -> list[dict]:
    """Every combination of TUNABLES over CONFIG, or a seeded sample of `points` of them; the current CONFIG first."""
    keys = sorted(tunables)
    combos = [dict(config, **dict(zip(keys, vals))) for vals in itertools.product(*(tunables[k] for k in keys))]
    combos = [c for c in combos if c != config]
    random.Random(seed).shuffle(combos)
    return [config] + combos[: max(0, points - 1)]


def sweep(code: str, judge: Callable[[str], dict], points: int = 16, seed: int = 0) -> tuple[str, dict, list]:
    """(the best point's code, its result, every point's {config, score}); the program unchanged when it declares no
    grid. Each point is one judgment; the caller re-times the winner before promoting it."""
    p = params(code)
    if p is None:
        return code, {}, []
    rows, best = [], (None, None, -1.0)
    for cfg in grid(*p, points, seed):
        variant = with_config(code, cfg)
        r = judge(variant)
        s = float(r["metrics"].get("combined_score") or 0.0)
        timed = json.loads(r["artifacts"].get("timings") or "[]") if s > 0 else []
        rows.append({"config": cfg, "score": s, "timings": [{k: t.get(k) for k in ("shape", "speedup", "weight")}
                                                            for t in timed]})
        if s > best[2]:
            best = (variant, r, s)
    return best[0], best[1], rows


def regions(rows: list, bins: int = 4) -> list | None:
    """Per size region (quantile bins of the seen shapes' first axis), the grid point with the best call-weighted
    log-speedup there: [(upper bound or None, config)], or None when one point wins everywhere."""
    sizes = sorted({t["shape"][0] for r in rows for t in r["timings"]})
    if len(sizes) < 2:
        return None
    edges = sorted({sizes[min(len(sizes) - 1, (i * len(sizes)) // bins)] for i in range(1, bins)} - {sizes[-1]})
    bounds = [*edges, None]
    table = []
    for i, ub in enumerate(bounds):
        lo = edges[i - 1] if i else None
        inside = lambda n: (lo is None or n > lo) and (ub is None or n <= ub)    # noqa: E731
        def gain(r):
            ts = [t for t in r["timings"] if inside(t["shape"][0]) and (t["speedup"] or 0) > 0]
            return sum((t["weight"] or 1) * math.log(t["speedup"]) for t in ts) if ts else -math.inf
        best = max((r for r in rows if r["timings"]), key=gain, default=None)
        if best is not None and gain(best) > -math.inf:
            table.append((ub, best["config"]))
    if len({json.dumps(c, sort_keys=True) for _, c in table}) < 2:
        return None
    return table


DISPATCH = """
_CONFIG_BY_SIZE = {table!r}
_dispatch_impl = {target}


def _dispatched(*args, **kwargs):
    global CONFIG
    n = {size}
    CONFIG = next(c for ub, c in _CONFIG_BY_SIZE if ub is None or n <= ub)
    return _dispatch_impl(*args, **kwargs)
"""


def dispatch(code: str, size_expr: str, table: list) -> str:
    """The program with its entry wrapped: each call sets CONFIG from `table` by the call's size (`size_expr`, from
    tensor shapes only, so it works inside a CUDA graph capture, where the choice is baked in per captured size)."""
    end = "# EVOLVE-BLOCK-END\n"
    if end in code:
        body, tail = code.split(end, 1)
        m = re.search(r"^kernel\s*=\s*(\w+)\s*$", tail, re.M)
        if m:
            tail = tail[:m.start()] + "kernel = _dispatched" + tail[m.end():]
            return body + DISPATCH.format(table=table, target=m.group(1), size=size_expr) + end + tail
        return body + DISPATCH.format(table=table, target="kernel", size=size_expr) + "kernel = _dispatched\n" + end + tail
    return code + DISPATCH.format(table=table, target="kernel", size=size_expr) + "kernel = _dispatched\n"
