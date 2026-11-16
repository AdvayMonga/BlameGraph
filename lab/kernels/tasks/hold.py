"""python -m lab.kernels.tasks.hold --task region:FIRST:LAST --samples DIR --out HOLDER.pkl

A region's GPU-host side: loads the engine, shares the region's weights, prints one JSON line describing the task,
and keeps the weights shared until its stdin closes."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HOLD_ENV = {"CUSTOM_BACKEND_BLOCKS": "16", "CUSTOM_BACKEND_CUDA_GRAPH": "0", "MEGAKERNEL": "0"}   # weights only


def held(name: str, model) -> tuple[dict, dict]:
    """(tensors, constants) the region `name` reads from the loaded engine `model`."""
    from lab.kernels.tasks import regions, step
    from lab.profile.capture import STEP
    if name == STEP:
        return step.weights(model)
    st = regions._engine()
    tensors = {}
    for i in regions.LAYERS:
        for k, v in st.layer_weights(model, i).items():
            tensors.update({f"L{i}.{k}": v} if v is not None else {k: None})     # a None (bf16 scale) is every layer's
    return tensors, {"cfg": st.layer_config(model), "layers": list(regions.LAYERS)}


def describe(name: str, samples: Path, holder: Path) -> dict:
    """The region as the controller needs it: {holder, samples, start, shapes, stations}."""
    from lab.kernels.tasks import regions, shapes
    from lab.kernels.evolve.run import initial_program
    from lab.kernels.tasks import load
    from lab.profile.capture import STEP
    task = load(name, samples=str(samples), shared_weights=str(holder))
    caps = sorted(Path(samples).glob("traffic*/*/captures"))
    counts = shapes.from_captures(task, caps, equal=True)
    if not counts:
        raise SystemExit(f"the captures in {samples} hold no sampled call of {name}")
    names = [s.name for s in regions.engine_stations()] if name == STEP else \
        [s.name for s in regions.span(regions.engine_stations(regions.station_set(name.split(":")[1])),
                                      *name.split(":")[1:3])]
    sets = {**shapes.split(task, counts), "ints": shapes.ints_from_captures(task, caps),
            "sources": shapes.sources_from_captures(task, caps), "heldout_axes": list(task.heldout_axes),
            "stations": names}
    return {"holder": str(holder), "samples": str(samples), "start": initial_program(task),
            "shapes": {k: [list(s) for s in v] if k in ("seen", "heldout") else v for k, v in sets.items()},
            "stations": names}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="lab.kernels.tasks.hold")
    ap.add_argument("--task", required=True)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--out", required=True, help="the holder file to write (its dir is made)")
    a = ap.parse_args(argv)
    from evals.targets import target
    from lab.infra.sandbox import container
    for k, v in target.load().engine.env.items():
        os.environ.setdefault(k, v)
    os.environ.update(HOLD_ENV)
    from inference_server.config import load_settings
    from inference_server.server import build_backend
    from lab.kernels.tasks.shared_weights import Holder
    model = build_backend(load_settings()).model
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
