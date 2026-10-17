"""Correctness checked outside the candidate's process: the jailed judge (kernels.evolve.core) records the outputs of
every plain check case and every timed call; this process, which never loads the candidate, regenerates each input
from its seed, runs the reference itself and compares. A candidate that patches the checker inside its own process
(and restores it before anything there notices) still has its outputs judged here. The record is loaded with
`weights_only=True`: tensors and plain values only, no code."""
from __future__ import annotations

from pathlib import Path


def verify(cfg: dict, path: Path) -> tuple[bool, str]:
    import torch
    from kernels.check import TOLERANCES, _compare, _gen
    from kernels.evolve.core import _scaled
    from kernels.tasks import load
    if not Path(path).exists():
        return False, "the candidate's process left no outputs to verify"
    recs = torch.load(path, weights_only=True).get("records") or []
    if not recs:
        return False, "no outputs were recorded"
    task, device = load(cfg["task"]).with_ints(cfg.get("ints")), cfg.get("device", "cpu")
    tol = {**TOLERANCES, **{getattr(torch, k): tuple(v) for k, v in (cfg.get("tolerances") or {}).items()}}
    for r in recs:
        dt = getattr(torch, r["dtype"])
        make = task.make_inputs if r.get("scale") is None else _scaled(task.make_inputs, r["scale"])
        inputs = make(tuple(r["shape"]), dt, device, _gen(r["seed"]))
        d = _compare(task.reference(*[t.clone() for t in inputs]), [o.to(device) for o in r["outputs"]], *tol[dt])
        if not d["ok"]:
            why = d.get("error") or f"max_abs {d['max_abs']:.3g}, max_rel {d['max_rel']:.3g}"
            return False, f"{r['kind']} output at shape {r['shape']} seed {r['seed']} differs from the reference: {why}"
    return True, f"{len(recs)} recorded outputs match the reference, recomputed outside the candidate's process"
