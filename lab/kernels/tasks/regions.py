"""Regions: a run of consecutive stations of the engine as a kernel task with a generated harness. The baseline runs
the stations in the engine's dtypes, the reference on fp32 copies; the engine is the target's `[regions]` (`Engine`)."""
from __future__ import annotations

import importlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import torch

from lab.kernels.tasks import Task

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


def functions(stations, first, last, live=(), constants: dict | None = None, reference_stations=None,
              discrete: tuple[str, ...] = ()) -> tuple[Callable, Callable, tuple[str, ...], tuple[str, ...]]:
    """(baseline, reference, inputs, outputs): both take tensors in `inputs` order and return them in `outputs` order.
    The reference runs `reference_stations` (else the stations) in fp32, in the baseline's dtypes. A station that
    writes `discrete` entries runs there with the baseline's values of them as ctx["pinned"]."""
    consts = dict(constants or {})
    region = span(stations, first, last)
    ref_region = span(reference_stations, first, last) if reference_stations is not None else region
    ins, outs = interface(stations, first, last, live)
    ins = tuple(i for i in ins if i not in consts)
    chooser = next((s for s in region if set(discrete) & set(s.writes)), None)
    picked = tuple(d for d in discrete if chooser and d in chooser.writes)

    def baseline(*xs):
        return run(region, {**consts, **dict(zip(ins, xs))}, outs)

    dtypes: dict = {}                    # the baseline's output dtypes, per input shapes and dtypes

    def reference(*xs):
        up = [x.float() if isinstance(x, torch.Tensor) and x.is_floating_point() else x for x in xs]
        pinned = None
        if chooser is not None:          # the baseline's discrete choice
            clones = [x.clone() if isinstance(x, torch.Tensor) else x for x in xs]
            picks = dict(zip(picked, run(region[:region.index(chooser) + 1], {**consts, **dict(zip(ins, clones))},
                                         picked)))
            ref_chooser = ref_region[region.index(chooser)]
            pinned = {chooser.name: lambda c: ref_chooser.fn({**c, "pinned": picks})}
        exact = run(ref_region, {**consts, **dict(zip(ins, up))}, outs, pinned)
        key = tuple((tuple(x.shape), x.dtype) if isinstance(x, torch.Tensor) else x for x in xs)
        if key not in dtypes:
            dtypes[key] = [o.dtype for o in baseline(*[x.clone() if isinstance(x, torch.Tensor) else x for x in xs])]
        return tuple(e.to(d) for e, d in zip(exact, dtypes[key]))
    return baseline, reference, ins, outs


def bytes_moved(stations, first, last, ctx: dict, live=()) -> dict:
    """{fused_bytes: the region's inputs and outputs; unfused_bytes: every station's reads and writes}, on `ctx`."""
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
         constants: dict | None = None, reference_stations=None, discrete: tuple[str, ...] = (), **kw) -> Task:
    """The region as a kernel task: inputs from `make_ctx(shape, dtype, device, generator)`, a dict of named tensors.
    The stations are the baseline, fp32 the reference; its cost is (0 FLOPs, the fused bytes)."""
    base, ref, ins, _ = functions(stations, first, last, live, constants, reference_stations, discrete)

    def make_inputs(shape, dtype, device, g):
        ctx = make_ctx(shape, dtype, device, g)
        return tuple(ctx[i] for i in ins)

    def cost(shape, dtype, inputs=None):
        inputs = inputs or make_inputs(shape, dtype, "cpu", torch.Generator().manual_seed(0))
        ctx = {**(constants or {}), **dict(zip(ins, inputs))}      # the stations read `cfg` too
        return 0.0, bytes_moved(stations, first, last, ctx, live)["fused_bytes"]
    watched = tuple(sorted({*watch, __name__, *(s.fn.__module__ for s in span(stations, first, last))}))
    return Task(name=name, entry=kw.pop("entry", None), reference=ref, make_inputs=make_inputs, shape_of=shape_of,
                cost=cost, dtypes=dtypes, layouts=(), error_budget=ERROR_BUDGET, baseline_fn=base, watch=watched, **kw)


# -- the engine's stations --
@dataclass(frozen=True)
class StationSet:
    """One of the engine's station sets, as the target declares it."""
    prefix: str = ""                                 # its stations' names start with this
    graphs: bool = False                             # the engine replays its steps in CUDA graphs
    capture_env: dict = field(default_factory=dict)  # what a capture of it sets
    rows: int = 0                                    # the axis of a region's first activation that grows with load
    tables: dict = field(default_factory=dict)       # further entries -> their axis that grows with it
    instead: dict = field(default_factory=dict)      # entries -> the axis held out in place of the rows, when read


@dataclass(frozen=True)
class Engine:
    """The target's `[regions]`: where the engine's stations are and what a region of them needs."""
    stations: str                                    # the module holding the station sets and their helpers
    overrides: str                                   # the package the engine reads a region's replacement from
    watch: tuple[str, ...] = ()                      # the modules the stations call into
    layers: tuple[int, ...] = ()                     # the layers a region is captured at and held for
    heldout_layer: int | None = None                 # its calls are held out
    weights: str = "w."                              # context entries with this prefix are weights
    hold_env: dict = field(default_factory=dict)     # what the hold process sets before the engine loads
    step: str = ""                                   # the set a whole decode step runs
    sets: dict = field(default_factory=dict)         # name -> StationSet

    def module(self):
        return importlib.import_module(self.stations)

    def which(self, first: str) -> str:
        """The name of the station set the station `first` belongs to: the longest matching prefix's."""
        return max((n for n, s in self.sets.items() if first.startswith(s.prefix)), key=lambda n: len(self.sets[n].prefix))

    def as_dict(self) -> dict:
        return asdict(self)


def load_engine(d: dict | None = None) -> Engine:
    """The engine from `d` (an `Engine.as_dict()`, or a target's `[regions]` table), else from the target."""
    if d is None:
        from lab.targets import target
        d = target.load().raw.get("regions")
        if not d:
            raise ValueError("the target declares no [regions]: its engine has no stations to search a region of")
    return Engine(**{**d, "watch": tuple(d.get("watch") or ()), "layers": tuple(d.get("layers") or ()),
                     "sets": {k: v if isinstance(v, StationSet) else StationSet(**v)
                              for k, v in (d.get("sets") or {}).items()}})


def _needs_worker(*_a, **_k):
    raise RuntimeError("this region task is described, not loaded: it loads where its samples and holder are")


def light(name: str, heldout_axes: tuple[int, ...] = (), eng: Engine | None = None) -> Task:
    """A region without its samples or its holder: its name, entry, capture env and held-out axes.
    Its reference, make_inputs and cost raise."""
    from lab.profile.capture import STEP, STEP_ENTRY
    step = name == STEP
    eng = eng or (None if step else load_engine())
    which = eng.which(name.split(":")[1]) if not step else ""
    env = () if step else (("CAPTURE_REGION_LAYERS", ",".join(map(str, eng.layers))),
                           *eng.sets[which].capture_env.items())
    return Task(name=name, entry=STEP_ENTRY if step else f"{eng.stations}:{which}", reference=_needs_worker,
                make_inputs=_needs_worker, shape_of=lambda call: None, cost=_needs_worker, layouts=(),
                heldout_axes=tuple(heldout_axes), error_budget=ERROR_BUDGET, capture_fn=name, capture_env=env)


def _importable(names: tuple[str, ...]) -> tuple[str, ...]:
    """The modules of `names` that import here."""
    import importlib
    out = []
    for n in names:
        try:
            importlib.import_module(n)
            out.append(n)
        except ImportError:
            pass
    return tuple(out)


def engine_stations(eng: Engine, which: str, functional: bool = True) -> tuple[Station, ...]:
    """The engine's stations of set `which`; `functional` gives in-place writers a copy."""
    st = eng.module()
    own = getattr(st, which)
    pure = st.functional(own) if functional and hasattr(st, "functional") else own
    return tuple(Station(s.name, tuple(s.reads), tuple(s.writes), s.fn) for s in pure)


def reference_stations(eng: Engine, which: str) -> tuple[Station, ...] | None:
    """The engine's stations of set `which` in a form that runs in fp32, or None."""
    st = eng.module()
    if not hasattr(st, "reference"):
        return None
    return tuple(Station(s.name, tuple(s.reads), tuple(s.writes), s.fn) for s in st.reference(getattr(st, which)))


def live(eng: Engine) -> tuple[str, ...]:
    """The entries that outlive a layer's stations."""
    return tuple(eng.module().LIVE)


def _in_place(eng: Engine) -> tuple[str, ...]:
    """The entries the engine writes in place."""
    return tuple(getattr(eng.module(), "IN_PLACE", ()))


def _shaped(eng: Engine, reads: dict) -> list[str]:
    """The reads that give a call its shape: activations with a shape, not weights or entries written in place."""
    fixed = _in_place(eng)
    return [k for k in sorted(reads) if not k.startswith(eng.weights) and k not in fixed and "shape" in reads[k]]


def _signature(eng: Engine, reads: dict) -> tuple[int, ...]:
    """A captured call's shape: its activations' shapes in name order, flattened."""
    return tuple(d for k in _shaped(eng, reads) for d in reads[k]["shape"])


def _heldout_axes(eng: Engine, which: str, reads: dict, ins: tuple[str, ...]) -> tuple[int, ...]:
    """Positions of the held-out axes in the flat signature, as the station set declares what grows with load: the
    first activation's `rows` axis and its `tables`; or only an `instead` entry's axis when the region reads one."""
    at, pos, s = {}, 0, eng.sets[which]
    for k in _shaped(eng, reads):
        at[k], pos = pos, pos + len(reads[k]["shape"])
    for name, axis in s.instead.items():
        if name in ins and name in at:
            return (at[name] + axis,)
    first = next((i for i in ins if i in at and len(reads[i]["shape"]) > s.rows), None)
    return tuple(([at[first] + s.rows] if first else [])
                 + [at[n] + axis for n, axis in s.tables.items() if n in ins and n in at and len(reads[n]["shape"]) > axis])


def _consts(entries: dict, constants: dict | None) -> dict:
    """A region's constants: a holder's non-tensor entries and `constants`."""
    return {**{k: v for k, v in entries.items() if not isinstance(v, torch.Tensor)}, **(constants or {})}


def _region(eng: Engine, first: str, last: str) -> tuple[tuple[Station, ...], tuple[str, ...]]:
    """(the engine's stations of the region's set, the names the region writes in place).
    The stations are not made functional when it writes in place; raises if it reads an entry the engine marks
    `OPAQUE` (one that is no tensor or constant, so it cannot be captured)."""
    which = eng.which(first)
    stations = engine_stations(eng, which)
    region = span(stations, first, last)
    inplace = tuple(sorted({w for s in region for w in s.writes} & set(_in_place(eng))))
    opaque = sorted({r for s in region for r in s.reads} & set(getattr(eng.module(), "OPAQUE", ())))
    if opaque:
        raise ValueError(f"region {first}..{last} reads {opaque}, which the engine cannot hand to a region: it must "
                         "start after the station that reads it")
    return (engine_stations(eng, which, functional=False) if inplace else stations), inplace


def _copying(ins: tuple[str, ...], outs: tuple[str, ...], inplace: tuple[str, ...]) -> Callable:
    """A task's `call` that passes fn a copy of the `inplace` inputs; raises unless fn returns those same tensors."""
    at = {n: ins.index(n) for n in inplace if n in ins}

    def call(fn, *xs):
        xs = list(xs)
        mine = {}
        for n, i in at.items():
            xs[i] = mine[n] = xs[i].clone()
        out = fn(*xs)
        got = out if isinstance(out, (tuple, list)) else (out,)
        for n, t in mine.items():
            if n in outs and got[outs.index(n)] is not t:
                raise ValueError(f"{n} must be written in place and returned (the engine passes its own pool)")
        return out
    return call


def from_engine(first: str, last: str, samples: str | None = None, shared_weights: str | None = None,
                dtypes: tuple = (torch.bfloat16,), constants: dict | None = None, engine: dict | None = None) -> Task:
    """The region first..last of the engine's stations as a task: inputs from `samples` (a CAPTURE_REGION dir), weights
    and constants from the `shared_weights` holder. Seed 0 is the captured inputs; later seeds perturb activations.
    `engine`: an `Engine.as_dict()`, where no target is set."""
    from lab.kernels.tasks.shared_weights import attach
    eng = load_engine(engine)
    st = eng.module()
    stations, inplace = _region(eng, first, last)
    held = attach(Path(shared_weights), copy_ok=True) if shared_weights else {}
    consts = _consts(held, constants)
    alive = live(eng)
    ins = tuple(i for i in interface(stations, first, last, alive)[0] if i not in consts)
    fn, which = f"region:{first}:{last}", eng.which(first)
    fixed = {*getattr(st, "STEP", ()), *_in_place(eng)}      # step inputs and the entries written in place
    layers = consts.get("layers") or ([consts["layer"]] if consts.get("layer") is not None else None)
    multi = bool(layers) and len(layers) > 1     # a call's shape starts with its layer
    calls = []
    if samples:
        from lab.profile import capture
        calls = [c for c in capture.load(Path(samples))
                 if c.get("fn") == fn and (layers is None or c.get("layer") in layers)]
    # A read captured as None is a constant None; calls where it is a tensor are left out.
    nones = sorted({i for c in calls for i, d in (c.get("reads") or {}).items() if i in ins and d.get("value", 1) is None})
    if nones:
        calls = [c for c in calls if all((c.get("reads") or {}).get(i, {}).get("value", 1) is None for i in nones)]
        consts = {**consts, **{i: None for i in nones}}
        ins = tuple(i for i in ins if i not in nones)
    def key(call) -> tuple:
        return ((call.get("layer"),) if multi else ()) + _signature(eng, call.get("reads") or {})
    by_shape: dict[tuple, Path] = {}
    for call in calls:
        if call.get("sample"):
            by_shape.setdefault(key(call), Path(samples) / "samples" / call["sample"])
    reads = (calls[0].get("reads") or {}) if calls else {}
    weights = {k.split(".", 1)[1] if k[:1] == "L" and k.split(".", 1)[0][1:].isdigit() else k for k in held}
    missing = [i for i in ins if "shape" not in reads.get(i, {}) or (i.startswith(eng.weights) and i not in weights)] \
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
        if g.initial_seed():                         # seed 0: the captured inputs
            for k, v in xs.items():
                if v.is_floating_point() and k not in fixed:
                    xs[k] = (v.float() * (1 + 1e-2 * torch.randn(v.shape, generator=g).to(device))).to(v.dtype)
        own = f"L{shape[0]}." if multi else None     # that layer's weights ("L<i>.w.*")
        return {**xs, **{(k[len(own):] if own else k): v for k, v in held.items() if isinstance(v, torch.Tensor)
                         and (own is None or k.startswith(own))}}

    def shape_of(call):
        if call.get("fn") != fn or (layers is not None and call.get("layer") not in layers):
            return None
        s = key(call)
        return s if s in by_shape else None          # a shape with no saved sample is skipped
    start = (lambda: start_program(eng, first, last, {**(constants or {}), **{i: None for i in nones}},
                                   shared_weights)) if shared_weights else None
    return task(fn, stations, first, last, make_ctx, shape_of, dtypes=dtypes, live=alive,
                watch=(eng.stations, *_importable(eng.watch)), entry=f"{eng.stations}:{which}", capture_fn=fn,
                seen_default=tuple(by_shape), constants=consts, reference_stations=reference_stations(eng, which),
                discrete=tuple(getattr(st, "DISCRETE", ())),
                heldout_axes=tuple(a + int(multi) for a in _heldout_axes(eng, which, reads, ins)) if calls else (),
                heldout_values=((0, (eng.heldout_layer,)),) if multi and eng.heldout_layer in layers else (),
                start_program=start,
                graphs=eng.sets[which].graphs,
                **({"call": _copying(ins, interface(stations, first, last, alive)[1], inplace)} if inplace else {}),
                edge_shapes=lambda shape: [])     # no synthetic edge shapes


def start_program(eng: Engine, first: str, last: str, constants: dict | None = None,
                  shared_weights: str | None = None) -> str:
    """The region's starting program: the engine's stations called in order, its constants written in as values."""
    from lab.kernels.tasks import shared_weights as sw
    stations, inplace = _region(eng, first, last)
    which = eng.which(first)
    consts = _consts(sw.constants(Path(shared_weights)) if shared_weights else {}, constants)
    ins, outs = interface(stations, first, last, live(eng))
    ins = tuple(i for i in ins if i not in consts)
    names = [s.name for s in span(stations, first, last)]
    bound = "".join(f"_CONST[{k!r}] = _st.{type(v).__name__}(**{vars(v)!r})\n" if hasattr(v, "__dataclass_fields__")
                    else f"_CONST[{k!r}] = {v!r}\n" for k, v in consts.items())
    return (f"# EVOLVE-BLOCK-START\nimport {eng.stations} as _st\n\n"
            f"_NAMES = {names!r}\n_INPUTS = {list(ins)!r}\n_OUTPUTS = {list(outs)!r}\n_CONST = {{}}\n{bound}"
            + (f"_PURE = _st.{which}\n" if inplace else       # writes in place, as the engine runs it
               f"_PURE = _st.functional(_st.{which}) if hasattr(_st, 'functional') else _st.{which}\n")
            + "_STATIONS = [s.fn for s in _PURE if s.name in _NAMES]\n\n\n"
            "def kernel(*xs):\n    ctx = {**_CONST, **dict(zip(_INPUTS, xs))}\n    for fn in _STATIONS:\n"
            "        ctx.update(fn(ctx))\n    return tuple(ctx[o] for o in _OUTPUTS)\n# EVOLVE-BLOCK-END\n")


def override_module(eng: Engine, program: str, first: str, last: str) -> str:
    """A winning program as a module of the engine's overrides package, with `OVERRIDES = {(first, last): _override}`.
    `_override` runs the engine's own stations when a None constant arrives as a tensor; raises on other mismatches."""
    from lab.kernels.eval.ship import module_source
    which = eng.which(first)
    return (module_source(program, "kernel").rstrip("\n") + "\n\n\ndef _own(c):\n"
            f"    import {eng.stations} as _eng\n"
            f"    names = [s.name for s in _eng.{which}]\n"
            f"    ctx = dict(c)\n    for s in _eng.{which}[names.index({first!r}):names.index({last!r}) + 1]:\n"
            "        ctx.update(s.fn(ctx))\n"
            "    return {k: ctx[k] for k in _OUTPUTS}\n\n\ndef _override(c):\n"
            "    consts = globals().get('_CONST', {})\n"
            "    if any(v is None and c.get(k) is not None for k, v in consts.items()\n"
            f"           if k in c and not k.startswith({eng.weights!r})):\n"
            "        return _own(c)\n"
            "    bad = [k for k, v in consts.items() if k in c and not (c[k] is None if v is None else\n"
            "                                                           type(c[k]) is type(v) and c[k] == v)]\n"
            "    if bad:\n"
            "        raise ValueError(f'region override {list(OVERRIDES)}: the engine passes {bad} unlike the values '\n"
            "                         f'it was built with ({ {k: _CONST[k] for k in bad} }): rebuild it for this engine')\n"
            "    out = kernel(*(c[k] for k in _INPUTS))\n"
            "    return dict(zip(_OUTPUTS, out if isinstance(out, (tuple, list)) else (out,)))\n\n\n"
            f"OVERRIDES = {{({first!r}, {last!r}): _override}}\n")
