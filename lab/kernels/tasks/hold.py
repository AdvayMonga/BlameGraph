"""python -m lab.kernels.tasks.hold --task region:FIRST:LAST --samples DIR --out HOLDER.pkl

A region's GPU-host side: loads the engine, shares the region's weights, prints one JSON line describing the task,
and keeps the weights shared until its stdin closes."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

def held(name: str, model) -> tuple[dict, dict]:
    """(tensors, constants) the region `name` reads from the loaded engine `model`."""
    from lab.kernels.tasks import regions, step
    from lab.profile.capture import STEP
    if name == STEP:
        return step.weights(model)
    eng = regions.load_engine()
    st = eng.module()
    tensors = {}
    for i in eng.layers:
        for k, v in st.layer_weights(model, i).items():
            tensors.update({f"L{i}.{k}": v} if v is not None else {k: None})     # a None (bf16 scale) is every layer's
    return tensors, {"cfg": st.layer_config(model), "layers": list(eng.layers)}


def describe(name: str, samples: Path, holder: Path) -> dict:
    """The region as the controller needs it: {holder, samples, start, shapes, stations, set}; `set` is every
    station of the region's set, in order."""
    from lab.kernels.tasks import regions, shapes
    from lab.kernels.evolve.run import initial_program
    from lab.kernels.tasks import load
    from lab.profile.capture import STEP
    task = load(name, samples=str(samples), shared_weights=str(holder))
    caps = sorted(Path(samples).glob("traffic*/*/captures"))
    counts = shapes.from_captures(task, caps, equal=True)
    if not counts:
        raise SystemExit(f"the captures in {samples} hold no sampled call of {name}")
    eng = regions.load_engine()
    whole = regions.engine_stations(eng, eng.step if name == STEP else eng.which(name.split(":")[1]))
    names = [s.name for s in (whole if name == STEP else regions.span(whole, *name.split(":")[1:3]))]
    sets = shapes.split(task, counts)
    sets = {**sets, "ints": shapes.ints_from_captures(task, caps),
            "sources": shapes.sources_from_captures(task, caps), "heldout_axes": list(task.heldout_axes),
            "example_calls": shapes.calls_from_captures(task, caps, sets["seen"]), "stations": names,
            **({} if name == STEP else {"code": regions.code(eng, *name.split(":")[1:3])})}
    return {"holder": str(holder), "samples": str(samples), "start": initial_program(task),
            "shapes": {k: [list(s) for s in v] if k in ("seen", "heldout") else v for k, v in sets.items()},
            "stations": names, "set": [s.name for s in whole]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="lab.kernels.tasks.hold")
    ap.add_argument("--task", required=True)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--out", required=True, help="the holder file to write (its dir is made)")
    a = ap.parse_args(argv)
    from lab.targets import target
    from lab.infra.sandbox import container
    for k, v in target.load().engine.env.items():
        os.environ.setdefault(k, v)
    from lab.kernels.tasks import regions
    from lab.kernels.tasks.shared_weights import Holder
    eng = regions.load_engine()
    os.environ.update(eng.hold_env)
    model = eng.module().load_model()
    tensors, consts = held(a.task, model)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    h = Holder(tensors, out, consts, jail_uid=container.user()[0])
    sys.stdout.write(json.dumps(describe(a.task, Path(a.samples), h.path)) + "\n")
    sys.stdout.flush()
    sys.stdin.read()                             # hold until stdin closes
    return 0


if __name__ == "__main__":
    sys.exit(main())
