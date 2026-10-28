"""Launch parameters chosen by a measured grid search, not guessed by the model (KernelEvolve, AlphaEvolve's tiling
heuristic; also the 2026-10-16 moe_gemm win, mostly tile sizes and stages). A program may define, at module level,
`CONFIG` (a dict of the launch parameters its kernel uses) and `TUNABLES` (for each key, the values to try).

Decided 2026-10-27: every program is tuned before each first measurement of its speed (`tuned`): the quick timing on
the most-called shape, the full exam on every seen shape. The evaluator's tune stage (kernels/evolve/core.py
`TUNE_STAGE`) loads each point with CONFIG rewritten in the source, checks it on the most-called shape and times it;
the program goes on with the fastest point written in, or, after the full exam's sweep, a per-size dispatcher where
different points win in different regions. The chosen values ship as plain constants: the engine pays no autotuning
at startup."""
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


DISPATCH_START, DISPATCH_END = "# TUNED-DISPATCH-START\n", "# TUNED-DISPATCH-END\n"
DISPATCH = DISPATCH_START + """
_CONFIG_BY_SIZE = {table!r}
_dispatch_impl = {target}


def _dispatched(*args, **kwargs):
    global CONFIG
    n = {size}
    CONFIG = next(c for ub, c in _CONFIG_BY_SIZE if ub is None or n <= ub)
    return _dispatch_impl(*args, **kwargs)
""" + DISPATCH_END


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


def strip(code: str) -> str:
    """The program without a dispatcher `dispatch` added (its entry bound back), so its grid can be tuned again."""
    i, j = code.find(DISPATCH_START), code.find(DISPATCH_END)
    if i < 0 or j < i:
        return code
    m = re.search(r"^_dispatch_impl = (\w+)$", code[i:j], re.M)
    out = code[:i] + code[j + len(DISPATCH_END):]
    if m and m.group(1) != "kernel":
        out = re.sub(r"^kernel = _dispatched$", f"kernel = {m.group(1)}", out, count=1, flags=re.M)
        return out if out.endswith("\n") else out + "\n"         # `dispatch` took the binding's line ending
    return re.sub(r"^kernel = _dispatched\n", "", out, count=1, flags=re.M)


def tuned(stage: Callable[[str, int], dict], code: str, quick: bool, size_expr: str | None = None) -> tuple[str, list]:
    """(the program with its grid's fastest point written in, every point's row): `stage(code, n)` runs the evaluator's
    tune stage, 7 on the most-called seen shape (`quick`), else 6 on every seen shape, where a per-size dispatcher over
    `size_expr` replaces one point when different points win in different regions. The program unchanged when it
    declares no grid or no point passed."""
    base = strip(code)
    if params(base) is None:
        return code, []
    r = stage(base, 7 if quick else 6)
    try:
        rows = json.loads((r.get("artifacts") or {}).get("tune") or "[]")
    except ValueError:
        rows = []
    ok = [x for x in rows if (x.get("score") or 0) > 0]
    if not ok:
        return code, rows
    new = with_config(base, max(ok, key=lambda x: x["score"])["config"])
    table = regions(ok) if size_expr and not quick else None
    return (dispatch(new, size_expr, table) if table else new), rows
