"""Regions (decided 2026-10-22/23): a run of consecutive decode stations of the engine, from one kernel up to the whole
step, as a kernel task whose harness is generated rather than hand-written.

The engine declares its stations, in order (name, what each reads and writes, and a function computing its writes;
data/notes/handoff-engine-stations.md). A region's inputs are what its stations read before any of them writes it;
its outputs are what it writes that a later station reads or that outlives the step (`live`: the residual stream, the
KV cache). The baseline runs the stations in order in the engine's dtypes; the reference runs them on fp32 copies of
the floating inputs, the truth a region's error budget is measured against (2x the baseline's own error). The fusion
bound is the bytes the region must move (its inputs and outputs; intermediates can stay on chip) at peak bandwidth,
beside the bytes its stations move one by one. `functional()` gives an in-place writer (kv_append) a copy of the K/V
pool, a copy the engine never makes, so its winner could not install as is: a region must end before kv_append or
start after it (refused otherwise, for now). The checker clones every input, shared weights included, several times
(the reference's fp32 copy, the baseline's, the before-copies), so a whole-step region does not fit in memory yet."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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


def run(region: tuple[Station, ...], ctx: dict, outputs: tuple[str, ...], pinned: dict | None = None) -> tuple:
    """The stations in order; a station named in `pinned` runs that function (ctx -> its writes) instead."""
    ctx, pinned = dict(ctx), pinned or {}
    for s in region:
        ctx.update((pinned.get(s.name) or s.fn)(ctx))
    return tuple(ctx[o] for o in outputs)


ROUTE_WRITES = ("sorted_ids", "expert_ids", "tok", "pair_w")    # the engine's `route` station's writes
DISCRETE = ("sorted_ids", "expert_ids", "tok")                  # its discrete choice: which expert each pair goes to


def pair_weights(logits, sorted_ids, expert_ids, tok, renorm: bool) -> torch.Tensor:
    """Each pair's routing weight in fp32 at the expert it was sent to: the softmax of its token's logits there,
    renormalized over the token's pairs when `renorm` (norm_topk_prob). The layout is the engine's route's: pair ids
    grouped by expert in `sorted_ids`, padded with M to blocks of `block_m`, `expert_ids` each block's expert."""
    M = tok.numel()
    block_m = sorted_ids.numel() // expert_ids.numel()
    slots = torch.nonzero(sorted_ids < M).view(-1)
    expert = torch.empty(M, dtype=torch.long, device=tok.device)
    expert[sorted_ids[slots].long()] = expert_ids[slots // block_m].long()
    t = tok.long()
    w = torch.softmax(logits.float(), dim=-1)[t, expert]
    if renorm:
        w = w / torch.zeros(logits.shape[0], device=w.device).index_add_(0, t, w)[t]
    return w


def functions(stations, first, last, live=(), constants: dict | None = None,
              reference_stations=None) -> tuple[Callable, Callable, tuple[str, ...], tuple[str, ...]]:
    """(baseline, reference, inputs, outputs): both take the inputs as tensors in `inputs` order (the `constants`, e.g.
    a layer config, bound instead) and return the outputs in `outputs` order. The reference computes in fp32 on
    `reference_stations` (the same stations in a form that runs in fp32: an engine's torch versions of its kernels),
    returning each output in the baseline's dtype; without them, on the stations themselves. A region holding the
    `route` station keeps the baseline's routing (fp32 logits may send a near-tied token to another expert) and
    computes each pair's weight in fp32 at the baseline's expert (`pair_weights`)."""
    consts = dict(constants or {})
    region = span(stations, first, last)
    ref_region = span(reference_stations, first, last) if reference_stations is not None else region
    ins, outs = interface(stations, first, last, live)
    ins = tuple(i for i in ins if i not in consts)
    route = next((s for s in region if set(ROUTE_WRITES) <= set(s.writes)), None)   # its reads include `logits`

    def baseline(*xs):
        return run(region, {**consts, **dict(zip(ins, xs))}, outs)

    dtypes: dict = {}                    # the baseline's output dtypes, per input shapes and dtypes

    def reference(*xs):
        up = [x.float() if isinstance(x, torch.Tensor) and x.is_floating_point() else x for x in xs]
        pinned = None
        if route is not None:            # the baseline's discrete routing, from the un-upcast inputs
            clones = [x.clone() if isinstance(x, torch.Tensor) else x for x in xs]
            picks = dict(zip(DISCRETE, run(region[:region.index(route) + 1], {**consts, **dict(zip(ins, clones))},
                                           DISCRETE)))
            pinned = {route.name: lambda c: {**picks, "pair_w": pair_weights(
                c["logits"], *(picks[k] for k in DISCRETE), bool(getattr(c.get("cfg"), "norm_topk_prob", False)))}}
        exact = run(ref_region, {**consts, **dict(zip(ins, up))}, outs, pinned)
        key = tuple((tuple(x.shape), x.dtype) if isinstance(x, torch.Tensor) else x for x in xs)
        if key not in dtypes:
            dtypes[key] = [o.dtype for o in baseline(*[x.clone() if isinstance(x, torch.Tensor) else x for x in xs])]
        return tuple(e.to(d) for e, d in zip(exact, dtypes[key]))
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
    return {"fused_bytes": sum(_n(ctx.get(i)) for i in ins) + sum(_n(seen.get(o)) for o in outs),
            "unfused_bytes": unfused}


def _n(x) -> int:
    return x.numel() * x.element_size() if isinstance(x, torch.Tensor) else 0


def task(name: str, stations: tuple[Station, ...], first: str, last: str,
         make_ctx: Callable[[tuple, torch.dtype, str, torch.Generator], dict], shape_of: Callable,
         dtypes: tuple = (torch.bfloat16,), live: tuple[str, ...] = (), watch: tuple[str, ...] = (),
         constants: dict | None = None, reference_stations=None, **kw) -> Task:
    """The region as a kernel task: inputs from `make_ctx(shape, dtype, device, generator)` (a context of named
    tensors: captured activations, the layer's weights), the stations as the baseline, fp32 as the reference, the
    error budget on, and the fused bytes as the roofline's bytes (no FLOP count: the bound is memory's)."""
    base, ref, ins, _ = functions(stations, first, last, live, constants, reference_stations)

    def make_inputs(shape, dtype, device, g):
        ctx = make_ctx(shape, dtype, device, g)
        return tuple(ctx[i] for i in ins)

    def cost(shape, dtype, inputs=None):
        inputs = inputs or make_inputs(shape, dtype, "cpu", torch.Generator().manual_seed(0))
        return 0.0, bytes_moved(stations, first, last, dict(zip(ins, inputs)), live)["fused_bytes"]
    watched = tuple(sorted({*watch, __name__, *(s.fn.__module__ for s in span(stations, first, last))}))
    return Task(name=name, entry=kw.pop("entry", None), reference=ref, make_inputs=make_inputs, shape_of=shape_of,
                cost=cost, dtypes=dtypes, layouts=(), error_budget=ERROR_BUDGET, baseline_fn=base, watch=watched, **kw)


# -- the engine's stations (inference-server models/stations.py; data/notes/handoff-engine-stations.md) ---------------
ENGINE = "inference_server.models.stations"
OVERRIDES_PACKAGE = "inference_server.models.regions"    # where the engine's `overrides()` reads OVERRIDES from
# The engine modules the stations call into at call time (watched for patches; the engine may export this later).
STATION_MODULES = ("inference_server.models.fused", "inference_server.models.moe_kernel",
                   "inference_server.models.paged_attention_kernel", "inference_server.models.paged_kv_cache",
                   "inference_server.models.qwen3_moe")
POOL = ("k_cache", "v_cache")           # the K/V pool's names when the engine declares no IN_PLACE


def _engine():
    import importlib
    return importlib.import_module(ENGINE)


def _importable(names: tuple[str, ...]) -> tuple[str, ...]:
    """The modules of `names` that exist (core._watched imports every watched module)."""
    import importlib.util
    out = []
    for n in names:
        try:
            out += [n] if importlib.util.find_spec(n) else []
        except (ImportError, ValueError):
            pass
    return tuple(out)


def _pool() -> tuple[str, ...]:
    """The K/V pool's entries: the whole pool, whose block count is the launch's memory budget, not a call's shape."""
    return tuple(getattr(_engine(), "IN_PLACE", POOL))


def engine_stations() -> tuple[Station, ...]:
    """The engine's decode stations (`DECODE`), made pure (`functional(DECODE)`: what writes in place gets a copy)."""
    st = _engine()
    pure = st.functional(st.DECODE) if hasattr(st, "functional") else st.DECODE
    return tuple(Station(s.name, tuple(s.reads), tuple(s.writes), s.fn) for s in pure)


def reference_stations() -> tuple[Station, ...] | None:
    """The engine's stations in a form that runs in fp32 (`reference()`: torch versions of its kernels), or None."""
    st = _engine()
    if not hasattr(st, "reference"):
        return None
    return tuple(Station(s.name, tuple(s.reads), tuple(s.writes), s.fn) for s in st.reference())


def live() -> tuple[str, ...]:
    return tuple(getattr(_engine(), "LIVE", ("h", "k_cache", "v_cache")))


def _shaped(reads: dict) -> list[str]:
    """The reads that give a call its shape: activations with a shape, not weights (the same for every call) and not
    the K/V pool (sized by the launch, not the call)."""
    pool = _pool()
    return [k for k in sorted(reads) if not k.startswith("w.") and k not in pool and "shape" in reads[k]]


def _signature(reads: dict) -> tuple[int, ...]:
    """A captured call's shape: its activations' shapes in name order, flattened."""
    return tuple(d for k in _shaped(reads) for d in reads[k]["shape"])


def _heldout_axes(reads: dict, ins: tuple[str, ...]) -> tuple[int, ...]:
    """Where in the flat signature the held-out axes are: the rows of the region's first activation, and the attention
    block tables' width (the context, in blocks) when the region reads them."""
    at, pos = {}, 0
    for k in _shaped(reads):
        at[k], pos = pos, pos + len(reads[k]["shape"])
    first = next((i for i in ins if i in at and reads[i]["shape"]), None)
    ctx = "attn_block_tables" in ins and len(reads.get("attn_block_tables", {}).get("shape", ())) == 2
    return tuple(([at[first]] if first else []) + ([at["attn_block_tables"] + 1] if ctx else []))


def _consts(entries: dict, constants: dict | None) -> dict:
    """A region's constants, the one source for its task and its start program: a holder's non-tensor entries (`cfg`,
    `layer`, weights that are None, e.g. bf16 experts' scales) and `constants`."""
    return {**{k: v for k, v in entries.items() if not isinstance(v, torch.Tensor)}, **(constants or {})}


def _region(first: str, last: str) -> tuple[Station, ...]:
    """The engine's stations, refusing a region that writes the K/V pool in place (see the module docstring)."""
    stations = engine_stations()
    inplace = {w for s in span(stations, first, last) for w in s.writes} & set(getattr(_engine(), "IN_PLACE", ()))
    if inplace:
        raise ValueError(f"region {first}..{last} writes {sorted(inplace)} in place (kv_append): a region must end "
                         "before kv_append or start after it, for now")
    return stations


def from_engine(first: str, last: str, samples: str | None = None, shared_weights: str | None = None,
                dtypes: tuple = (torch.bfloat16,), constants: dict | None = None) -> Task:
    """The region first..last of the engine's decode stations as a task: inputs from captured samples (`samples`: a
    CAPTURE_REGION dir, its first call per shape kept whole; only the holder's `layer` when it names one) and the
    layer's weights and config attached from a holder (`shared_weights`: a kernels.shared_weights file whose
    non-tensor entries, e.g. `cfg`, are constants); seed 0 is the captured inputs exactly, later seeds perturb the
    activations, never the weights."""
    from kernels.shared_weights import attach
    stations = _region(first, last)
    held = attach(Path(shared_weights), copy_ok=True) if shared_weights else {}   # the jail's IPC may be refused
    consts = _consts(held, constants)
    ins = tuple(i for i in interface(stations, first, last, live())[0] if i not in consts)
    fn, layer = f"region:{first}:{last}", consts.get("layer")
    calls = []
    if samples:
        from lab import capture
        calls = [c for c in capture.load(Path(samples))
                 if c.get("fn") == fn and (layer is None or c.get("layer") == layer)]
    by_shape: dict[tuple, Path] = {}
    for call in calls:
        if call.get("sample"):
            by_shape.setdefault(_signature(call.get("reads") or {}), Path(samples) / "samples" / call["sample"])
    reads = (calls[0].get("reads") or {}) if calls else {}
    missing = [i for i in ins if "shape" not in reads.get(i, {}) or (i.startswith("w.") and i not in held)] \
        if calls else []
    if missing:
        raise ValueError(f"region {first}..{last} reads {missing}, which are neither captured tensors nor constants: "
                         "the layer config `cfg` and the weights come from a holder (`shared_weights`) or `constants`")

    def make_ctx(shape, dtype, device, g):
        got = by_shape.get(tuple(shape))
        if got is None:
            raise ValueError(f"no captured sample of region {first}..{last} at shape {shape}")
        xs = {k: v.to(device) for k, v in torch.load(got, weights_only=True)["reads"].items()
              if isinstance(v, torch.Tensor)}
        if g.initial_seed():                         # seed 0: the captured inputs; others perturbed, per seed
            for k, v in xs.items():
                if v.is_floating_point() and k not in ("cos", "sin", "k_cache", "v_cache"):
                    xs[k] = (v.float() * (1 + 1e-2 * torch.randn(v.shape, generator=g).to(device))).to(v.dtype)
        return {**xs, **{k: v for k, v in held.items() if isinstance(v, torch.Tensor)}}

    def shape_of(call):
        if call.get("fn") != fn or (layer is not None and call.get("layer") != layer):
            return None
        s = _signature(call.get("reads") or {})
        return s if s in by_shape else None          # a shape with no saved sample (over the engine's cap) is skipped
    start = (lambda: start_program(first, last, constants, shared_weights)) if shared_weights else None
    return task(fn, stations, first, last, make_ctx, shape_of, dtypes=dtypes, live=live(),
                watch=(ENGINE, *_importable(STATION_MODULES)), entry=f"{ENGINE}:DECODE", capture_fn=fn,
                seen_default=tuple(by_shape), constants=consts, reference_stations=reference_stations(),
                heldout_axes=_heldout_axes(reads, ins) if calls else (), start_program=start,
                edge_shapes=lambda shape: [])     # inputs come from captures only: no synthetic edge shapes


def start_program(first: str, last: str, constants: dict | None = None, shared_weights: str | None = None) -> str:
    """The starting program of a region: the engine's own stations, called in order, its constants (the holder's,
    as `from_engine` binds them, and `constants`) written in as values."""
    from kernels import shared_weights as sw
    stations = _region(first, last)
    consts = _consts(sw.constants(Path(shared_weights)) if shared_weights else {}, constants)
    ins, outs = interface(stations, first, last, live())
    ins = tuple(i for i in ins if i not in consts)
    names = [s.name for s in span(stations, first, last)]
    bound = "".join(f"_CONST[{k!r}] = _st.{type(v).__name__}(**{vars(v)!r})\n" if hasattr(v, "__dataclass_fields__")
                    else f"_CONST[{k!r}] = {v!r}\n" for k, v in consts.items())
    return (f"# EVOLVE-BLOCK-START\nimport {ENGINE} as _st\n\n"
            f"_NAMES = {names!r}\n_INPUTS = {list(ins)!r}\n_OUTPUTS = {list(outs)!r}\n_CONST = {{}}\n{bound}"
            "_PURE = _st.functional(_st.DECODE) if hasattr(_st, 'functional') else _st.DECODE\n"
            "_STATIONS = [s.fn for s in _PURE if s.name in _NAMES]\n\n\n"
            "def kernel(*xs):\n    ctx = {**_CONST, **dict(zip(_INPUTS, xs))}\n    for fn in _STATIONS:\n"
            "        ctx.update(fn(ctx))\n    return tuple(ctx[o] for o in _OUTPUTS)\n# EVOLVE-BLOCK-END\n")


def override_module(program: str, first: str, last: str) -> str:
    """A winning program as a module of the engine's overrides package: the program (`kernel`, `_INPUTS`, `_OUTPUTS`,
    `_CONST`) plus `OVERRIDES = {(first, last): _override}`, which the engine's `overrides()` reads and `resolve()`
    checks against the stretch's interface. Each call checks that what the engine passes for a baked constant is
    that constant (None stays None, `cfg` equal): per call but trivial next to the kernel (graph replays skip it)."""
    from kernels.evolve.ship import module_source
    return (module_source(program, "kernel").rstrip("\n") + "\n\n\ndef _override(c):\n"
            "    bad = [k for k, v in _CONST.items() if k in c and not (c[k] is None if v is None else\n"
            "                                                           type(c[k]) is type(v) and c[k] == v)]\n"
            "    if bad:\n"
            "        raise ValueError(f'region override {list(OVERRIDES)}: the engine passes {bad} unlike the values '\n"
            "                         f'it was built with ({ {k: _CONST[k] for k in bad} }): rebuild it for this engine')\n"
            "    out = kernel(*(c[k] for k in _INPUTS))\n"
            "    return dict(zip(_OUTPUTS, out if isinstance(out, (tuple, list)) else (out,)))\n\n\n"
            f"OVERRIDES = {{({first!r}, {last!r}): _override}}\n")
