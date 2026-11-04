"""A program's Triton autotune frozen into a table before it is gated and shipped (decided 2026-11-04).

`triton.autotune` tunes at run time for every key it has not seen, so a kernel keyed on a size that changes every step
(the batch's rows) tunes again in the middle of serving. The freeze stage (core.FREEZE_STAGE, jailed) runs the program
on every seen shape and records the config each autotuner chose for each key; `rewrite` replaces each autotune with
`_FrozenTune`, which launches the config tuned for the call's key, else for the nearest tuned key (equal non-numeric
values, then the smallest tuned sizes at or above the call's, then the closest), and never tunes. The table is a
literal in the module, so it ships with it, as vLLM's and SGLang's offline-tuned MoE configs do."""
from __future__ import annotations

import ast
import inspect
import json
import re
from typing import Callable

FREEZE_STAGE = 6         # core's stage: every seen shape run once, each autotuner's chosen config recorded


def _frozen_key(arg_names, keys, args, kwargs):
    """What an autotuner keys on, by argument name: its `key` args, then every tensor arg's dtype."""
    named = dict(zip(arg_names, args))
    named.update((k, v) for k, v in kwargs.items() if k in arg_names)
    vals = [named[k] for k in keys if k in named] + [str(v.dtype) for v in named.values() if hasattr(v, "dtype")]
    return [v if v is None or isinstance(v, (bool, int, float, str)) else str(v) for v in vals]


def _frozen_pick(table, key):
    """The launch config tuned for `key`, else for the nearest tuned key."""
    def num(x):
        return isinstance(x, (int, float)) and not isinstance(x, bool)

    def rank(row):
        k = row[0]
        if len(k) != len(key):
            return (2, 1, 0.0)
        pairs = list(zip(k, key))
        same = all(a == b for a, b in pairs if not (num(a) and num(b)))
        above = all(a >= b for a, b in pairs if num(a) and num(b))
        far = sum(abs(a - b) / max(abs(a), abs(b), 1) for a, b in pairs if num(a) and num(b))
        return (0 if same else 1, 0 if above else 1, far)
    return min(table, key=rank)[1]


class _FrozenTune:
    """A Triton kernel launched with the config its autotuner chose for the nearest tuned key; it never tunes."""

    def __init__(self, fn, keys, table):
        self.fn, self.keys, self.table, self.arg_names, self.picked = fn, keys, table, fn.arg_names, {}

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            key = tuple(_frozen_key(self.arg_names, self.keys, args, kwargs))
            if key not in self.picked:           # each key ranked once, as Triton caches its choice per key
                self.picked[key] = _frozen_pick(self.table, list(key))
            return self.fn[grid](*args, **kwargs, **self.picked[key])
        return launch


MARK = "# FROZEN-AUTOTUNE: the configs this program's autotune chose, as a table (no tuning at run time)"
HELPER = "\n\n".join(inspect.getsource(f) for f in (_frozen_key, _frozen_pick, _FrozenTune))


def _is_autotune(node) -> bool:
    return isinstance(node, ast.Call) and ast.unparse(node.func).split(".")[-1] == "autotune"


def sites(source: str) -> list[dict]:
    """Each autotune in `source`: {name, line, form ("decorator" | "call"), text}; `name` is the module-level name it
    binds, None when it binds none (inside a function, a class or an expression), which cannot be frozen."""
    tree = ast.parse(source)
    top, out = set(), []
    for st in tree.body:
        if isinstance(st, ast.FunctionDef):
            for d in st.decorator_list:
                if _is_autotune(d):
                    top.add(id(d))
                    out.append({"name": st.name, "line": st.lineno, "form": "decorator", "text": ast.unparse(d)})
        elif (isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name)
              and isinstance(st.value, ast.Call) and _is_autotune(st.value.func) and len(st.value.args) == 1):
            top.add(id(st.value.func))
            out.append({"name": st.targets[0].id, "line": st.lineno, "form": "call", "text": ast.unparse(st.value.func)})
    out += [{"name": None, "line": n.lineno, "form": "other", "text": ast.unparse(n)}
            for n in ast.walk(tree) if _is_autotune(n) and id(n) not in top]
    return sorted(out, key=lambda s: s["line"])


def introduced(source: str, base: str | None) -> list[str]:
    """The live autotunes in `source` that are not, as written, in `base` (the engine module it replaces): what a
    shipped program would tune while serving that the engine did not."""
    def live(src):
        return [re.sub(r"\s+", "", s["text"]) for s in sites(src)] if src else []
    have = live(base)
    return [s["text"] for s in sites(source) if re.sub(r"\s+", "", s["text"]) not in have]


def config_of(c) -> dict:
    """A Triton Config as the launch keywords it adds (`all_kwargs`: its meta-parameters, num_warps, num_stages, ...)."""
    if getattr(c, "pre_hook", None) is not None:
        raise ValueError("a config with a pre_hook cannot be frozen into a table")
    if hasattr(c, "all_kwargs"):
        return dict(c.all_kwargs())
    return {**c.kwargs, **{k: getattr(c, k) for k in ("num_warps", "num_stages", "num_ctas", "maxnreg")
                           if getattr(c, k, None) is not None}}


def record(tuner) -> dict:
    """Wrap one autotuner's `run` so each call's key and the config it launched with are kept (the returned dict,
    {"keys", "table"}, filled as the program runs)."""
    keys = list(tuner.keys)
    names = list(getattr(tuner, "arg_names", None) or tuner.fn.arg_names)
    rows, run = {}, tuner.run
    out = {"keys": keys, "table": rows}

    def recorded(*args, **kwargs):
        r = run(*args, **kwargs)
        key = _frozen_key(names, keys, args, kwargs)
        rows.setdefault(json.dumps(key), [key, config_of(tuner.best_config)])
        return r
    tuner.run = recorded
    return out


def tuners(module, source: str) -> dict:
    """The program's autotuners by the module-level name each autotune site binds."""
    names = {s["name"] for s in sites(source) if s["name"]}
    return {n: v for n, v in vars(module).items() if n in names and hasattr(v, "configs") and hasattr(v, "fn")
            and callable(getattr(v, "run", None))}


def tables(recorded: dict) -> dict:
    """{name: {keys, table: [[key, config], ...]}} from `record`'s dicts, the autotuners that never ran left out."""
    return {n: {"keys": r["keys"], "table": list(r["table"].values())} for n, r in recorded.items() if r["table"]}


def rewrite(source: str, frozen: dict) -> str:
    """`source` with each autotune named in `frozen` replaced by `_FrozenTune` over its table, the helper written once
    before the first; the rest left as they are."""
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    edits = []                                   # (first line, last line, replacement), 1-based, inclusive
    first = None
    for st in tree.body:
        if isinstance(st, ast.FunctionDef) and st.name in frozen:
            ds = [d for d in st.decorator_list if _is_autotune(d)]
            if not ds:
                continue
            for d in ds:
                edits.append((d.lineno, d.end_lineno, ""))
            edits.append((st.end_lineno + 1, st.end_lineno, "\n\n" + _bind(st.name, st.name, frozen[st.name])))
            first = min(first or st.lineno, min(d.lineno for d in st.decorator_list))
        elif (isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name)
              and st.targets[0].id in frozen and isinstance(st.value, ast.Call) and _is_autotune(st.value.func)):
            fn = ast.get_source_segment(source, st.value.args[0])
            edits.append((st.lineno, st.end_lineno, _bind(st.targets[0].id, fn, frozen[st.targets[0].id])))
            first = min(first or st.lineno, st.lineno)
    if not edits:
        return source
    if "class _FrozenTune" not in source:
        edits.append((first, first - 1, f"{MARK}\n{HELPER}\n\n"))
    for a, b, text in sorted(edits, key=lambda e: (e[0], e[1]), reverse=True):
        lines[a - 1:b] = [text] if text else []
    out = "".join(lines)
    ast.parse(out)
    return out


def _bind(name: str, fn: str, t: dict) -> str:
    rows = "".join(f"    ({k!r}, {c!r}),\n" for k, c in t["table"])
    return f"{name} = _FrozenTune({fn}, {t['keys']!r}, [\n{rows}])\n"


def freeze(source: str, judge: Callable[[str], dict]) -> tuple[str, dict]:
    """(`source` with every autotune that ran on a seen shape frozen, the tables). `judge(source)` is the freeze
    stage's result for it; a source without autotune is returned as it is, never judged. Raises when an autotune binds
    no module-level name or the stage failed."""
    found = sites(source)
    if not found:
        return source, {}
    loose = [s for s in found if s["name"] is None]
    if loose:
        raise ValueError(f"autotune at line {loose[0]['line']} binds no module-level name, so it cannot be frozen: "
                         f"`{loose[0]['text'][:80]}`")
    r = judge(source)
    a = r.get("artifacts") or {}
    if "frozen" not in a:
        raise RuntimeError(f"the freeze stage failed: {a.get('error', json.dumps(r)[:1500])}")
    frozen = json.loads(a["frozen"])
    return rewrite(source, frozen), frozen
