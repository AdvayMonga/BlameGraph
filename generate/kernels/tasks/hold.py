"""python -m generate.kernels.tasks.hold --task region:FIRST:LAST --samples DIR --out HOLDER.pkl

A region's GPU-host side, for a search whose controller cannot load it (generate/kernels/tasks/pipeline.py): loads the engine as it
serves (its target's env, a token K/V pool, no CUDA graphs), shares the region's weights from this process
(kernels/shared_weights.Holder: the layers of `regions.LAYERS`, each as "L<i>.w.*", for a stretch of stations; every layer's and the head's
for the decode step), then loads the task from its samples (a packed capture dir, kernels/pipeline.pack) and the
holder, and prints one JSON line: the holder file, its shapes split seen / held-out with the held-out axes, the
starting program and the engine's station names. It keeps the weights shared until its stdin closes."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HOLD_ENV = {"CUSTOM_BACKEND_BLOCKS": "16", "CUSTOM_BACKEND_CUDA_GRAPH": "0", "MEGAKERNEL": "0"}   # weights only


def held(name: str, model) -> tuple[dict, dict]:
    """(tensors, constants) the region `name` reads from the loaded engine `model`."""
    from generate.kernels.tasks import regions, step
    from measure.profile.capture import STEP
    if name == STEP:
        return step.weights(model)
    st = regions._engine()
    tensors = {}
    for i in regions.LAYERS:
        for k, v in st.layer_weights(model, i).items():
            tensors.update({f"L{i}.{k}": v} if v is not None else {k: None})     # a None (bf16 scale) is every layer's
    return tensors, {"cfg": st.layer_config(model), "layers": list(regions.LAYERS)}


def describe(name: str, samples: Path, holder: Path) -> dict:
    """The region as the controller needs it: shapes from every traffic's calls in `samples` (each weighing the same),
    split seen / held-out, with the held-out axes and the captured ints; the starting program; the station names."""
    from generate.kernels.tasks import regions, shapes
    from generate.kernels.evolve.run import initial_program
    from generate.kernels.tasks import load
    from measure.profile.capture import STEP
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
            "stations": names}                   # the controller's prompt shows the libraries implementing them
    return {"holder": str(holder), "samples": str(samples), "start": initial_program(task),
            "shapes": {k: [list(s) for s in v] if k in ("seen", "heldout") else v for k, v in sets.items()},
            "stations": names}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="generate.kernels.tasks.hold")
    ap.add_argument("--task", required=True)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--out", required=True, help="the holder file to write (its dir is made)")
    a = ap.parse_args(argv)
    from targets import target
    from measure.sandbox import container
    for k, v in target.load().engine.env.items():
        os.environ.setdefault(k, v)
    os.environ.update(HOLD_ENV)
    from inference_server.config import load_settings
    from inference_server.server import build_backend
    from generate.kernels.tasks.shared_weights import Holder
    model = build_backend(load_settings()).model
    tensors, consts = held(a.task, model)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    h = Holder(tensors, out, consts, jail_uid=container.user()[0])
    sys.stdout.write(json.dumps(describe(a.task, Path(a.samples), h.path)) + "\n")
    sys.stdout.flush()
    sys.stdin.read()                             # shared until the controller closes our stdin
    return 0


if __name__ == "__main__":
    sys.exit(main())
