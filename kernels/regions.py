"""Regions (decided 2026-10-22/23): a run of consecutive decode stations of the engine, from one kernel up to the whole
step, as a kernel task whose harness is generated rather than hand-written.

The engine declares its stations, in order (name, what each reads and writes, and a function computing its writes;
data/notes/handoff-engine-stations.md). A region's inputs are what its stations read before any of them writes it;
its outputs are what it writes that a later station reads or that outlives the step (`live`: the residual stream, the
KV cache). The baseline runs the stations in order in the engine's dtypes; the reference runs them on fp32 copies of
the floating inputs, the truth a region's error budget is measured against (2x the baseline's own error). The fusion
bound is the bytes the region must move (its inputs and outputs; intermediates can stay on chip) at peak bandwidth,
beside the bytes its stations move one by one."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from kernels.tasks import Task

ERROR_BUDGET = 2.0           # a region's output may be at most this many times as wrong as the engine's own code


@dataclass(frozen=True)
class Station:
    name: str
    reads: tuple[str, ...]
    writes: tuple[str, ...]
    fn: Callable[[dict], dict]          # fn(ctx) -> {name: tensor} for its writes; never mutates what it reads


def span(stations: tuple[Station, ...], first: str, last: str) -> tuple[Station, ...]:
    names = [s.name for s in stations]
    if first not in names or last not in names or names.index(first) > names.index(last):
        raise ValueError(f"no region {first}..{last} in stations {names}")
    return tuple(stations[names.index(first):names.index(last) + 1])


def interface(stations: tuple[Station, ...], first: str, last: str,
              live: tuple[str, ...] = ()) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(inputs, outputs) of the region: inputs in the order first read; outputs in the order last written."""
    region = span(stations, first, last)
    after = stations[stations.index(region[-1]) + 1:]
    inputs, written = [], []
    for s in region:
        inputs += [r for r in s.reads if r not in written and r not in inputs]
        written = [w for w in written if w not in s.writes] + list(s.writes)
    needed = {r for s in after for r in s.reads} | set(live)
    return tuple(inputs), tuple(w for w in written if w in needed)


def run(region: tuple[Station, ...], ctx: dict, outputs: tuple[str, ...]) -> tuple:
    ctx = dict(ctx)
    for s in region:
        ctx.update(s.fn(ctx))
    return tuple(ctx[o] for o in outputs)


def functions(stations, first, last, live=()) -> tuple[Callable, Callable, tuple[str, ...], tuple[str, ...]]:
    """(baseline, reference, inputs, outputs): both take the inputs as tensors in `inputs` order and return the outputs
    in `outputs` order; the reference computes in fp32 and returns each output in the baseline's dtype."""
    region = span(stations, first, last)
    ins, outs = interface(stations, first, last, live)

    def baseline(*xs):
        return run(region, dict(zip(ins, xs)), outs)

    def reference(*xs):
        up = [x.float() if isinstance(x, torch.Tensor) and x.is_floating_point() else x for x in xs]
        exact = run(region, dict(zip(ins, up)), outs)
        dtypes = [o.dtype for o in baseline(*[x.clone() if isinstance(x, torch.Tensor) else x for x in xs])]
        return tuple(e.to(d) for e, d in zip(exact, dtypes))
    return baseline, reference, ins, outs


def bytes_moved(stations, first, last, ctx: dict, live=()) -> dict:
    """What the region must read and write (`fused_bytes`: its inputs and outputs) beside what its stations read and
    write one by one (`unfused_bytes`, each intermediate written and read again), on the tensors of `ctx`."""
    region = span(stations, first, last)
    ins, outs = interface(stations, first, last, live)
    seen, unfused = dict(ctx), 0
    for s in region:
        got = s.fn(seen)
        unfused += sum(_n(seen.get(r)) for r in s.reads) + sum(_n(v) for v in got.values())
        seen.update(got)
    return {"fused_bytes": sum(_n(ctx.get(i)) for i in ins) + sum(_n(seen[o]) for o in outs),
            "unfused_bytes": unfused}


def _n(x) -> int:
    return x.numel() * x.element_size() if isinstance(x, torch.Tensor) else 0


def task(name: str, stations: tuple[Station, ...], first: str, last: str,
         make_ctx: Callable[[tuple, torch.dtype, str, torch.Generator], dict], shape_of: Callable,
         dtypes: tuple = (torch.bfloat16,), live: tuple[str, ...] = (), watch: tuple[str, ...] = (), **kw) -> Task:
    """The region as a kernel task: inputs from `make_ctx(shape, dtype, device, generator)` (a context of named
    tensors: captured activations, the layer's weights), the stations as the baseline, fp32 as the reference, the
    error budget on, and the fused bytes as the roofline's bytes (no FLOP count: the bound is memory's)."""
    base, ref, ins, _ = functions(stations, first, last, live)

    def make_inputs(shape, dtype, device, g):
        ctx = make_ctx(shape, dtype, device, g)
        return tuple(ctx[i] for i in ins)

    def cost(shape, dtype, inputs=None):
        inputs = inputs or make_inputs(shape, dtype, "cpu", torch.Generator().manual_seed(0))
        return 0.0, bytes_moved(stations, first, last, dict(zip(ins, inputs)), live)["fused_bytes"]
    watched = tuple(sorted({*watch, __name__, *(s.fn.__module__ for s in span(stations, first, last))}))
    return Task(name=name, entry=None, reference=ref, make_inputs=make_inputs, shape_of=shape_of, cost=cost,
                dtypes=dtypes, layouts=(), error_budget=ERROR_BUDGET, baseline_fn=base, watch=watched, **kw)
