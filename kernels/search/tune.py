"""Launch parameters chosen by a grid search, not by the model (KernelEvolve, AlphaEvolve's tiling heuristic; also the
2026-10-16 moe_gemm win, mostly tile sizes and stages). A program may define, at module level, `CONFIG` (a dict of the
launch parameters its kernel uses) and `TUNABLES` (for each key, the values to try). The sweep judges points of that
grid like any program, with `CONFIG` rewritten in the source, so the chosen values ship as plain constants and the
engine pays no autotuning at startup."""
from __future__ import annotations

import ast
import itertools
import random
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
        rows.append({"config": cfg, "score": s})
        if s > best[2]:
            best = (variant, r, s)
    return best[0], best[1], rows
