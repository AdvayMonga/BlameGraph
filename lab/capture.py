"""Record what an engine's kernels are actually called with: wrap Python entry points ("module:function") for the
duration of a block and log every call's argument shapes, dtypes and small integer tensors (sequence lengths, block
tables, expert offsets); keep the first call of each shape signature whole when it fits a byte cap, as real inputs.
Synchronizes to read integer tensors, so it is for a capture run, never a timed one."""
from __future__ import annotations

import contextlib
import functools
import importlib
import json
import sys
from pathlib import Path

import torch

SMALL_INT = 4096           # integer tensors up to this many elements are logged by value


def _describe(x) -> dict:
    if isinstance(x, torch.Tensor):
        d = {"shape": list(x.shape), "dtype": str(x.dtype).removeprefix("torch."), "contiguous": x.is_contiguous()}
        if not x.is_floating_point() and x.numel() <= SMALL_INT:
            d["values"] = x.detach().flatten().tolist()
        return d
    if isinstance(x, (int, float, bool, str)) or x is None:
        return {"value": x}
    return {"type": type(x).__name__}


def signature(call: dict) -> str:
    """The shape signature of a call: argument shapes and dtypes, scalars as values."""
    return json.dumps([[a.get("shape"), a.get("dtype"), a.get("value")] for a in call["args"]] +
                      [[k, v.get("shape"), v.get("dtype"), v.get("value")] for k, v in sorted(call["kwargs"].items())])


def _resolve(target: str):
    mod, _, attr = target.partition(":")
    return importlib.import_module(mod), attr


@contextlib.contextmanager
def capture(targets: list[str], out: Path, sample_bytes: int = 64 * 2**20):
    """Wrap each "module:function" (and every loaded module that imported the same object) while the block runs;
    write <out>/calls.jsonl and <out>/samples/<n>.pt (the first call per signature, if its tensors fit)."""
    out = Path(out)
    (out / "samples").mkdir(parents=True, exist_ok=True)
    log = open(out / "calls.jsonl", "a")
    seen: set[str] = set()
    patched: list[tuple[object, str, object]] = []

    def wrap(name, fn):
        @functools.wraps(fn)
        def traced(*args, **kwargs):
            call = {"fn": name, "args": [_describe(a) for a in args],
                    "kwargs": {k: _describe(v) for k, v in kwargs.items()}}
            sig = signature(call)
            if sig not in seen:
                seen.add(sig)
                tensors = [a for a in (*args, *kwargs.values()) if isinstance(a, torch.Tensor)]
                size = sum(t.numel() * t.element_size() for t in tensors)
                if size <= sample_bytes:
                    path = out / "samples" / f"{len(seen)}.pt"
                    torch.save({"args": [a.detach().cpu() if isinstance(a, torch.Tensor) else a for a in args],
                                "kwargs": {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                                           for k, v in kwargs.items()}}, path)
                    call["sample"] = path.name
                else:
                    call["sample_skipped_bytes"] = size
            log.write(json.dumps(call) + "\n")
            return fn(*args, **kwargs)
        return traced

    try:
        for t in targets:
            mod, attr = _resolve(t)
            orig = getattr(mod, attr)
            new = wrap(t, orig)
            for m in list(sys.modules.values()):          # the defining module and every `from x import f` copy
                if m is not None and getattr(m, "__dict__", {}).get(attr) is orig:   # never trips lazy __getattr__
                    setattr(m, attr, new)
                    patched.append((m, attr, orig))
        yield out
    finally:
        for m, attr, orig in patched:
            setattr(m, attr, orig)
        log.close()


def load(out: Path) -> list[dict]:
    p = Path(out) / "calls.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []
