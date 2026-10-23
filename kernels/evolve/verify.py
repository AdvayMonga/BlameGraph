"""Correctness checked outside the candidate's process: the jailed judge (kernels.evolve.core) records the outputs of
every plain check case and every timed call; this process, which never loads the candidate, regenerates each input
from its seed, runs the reference itself and compares. A candidate that patches the checker inside its own process
(and restores it before anything there notices) still has its outputs judged here. The record is loaded with
`weights_only=True`: tensors and plain values only, no code."""
from __future__ import annotations

import json
from pathlib import Path


MAX_BYTES = 32 * 2**30


def expected(cfg: dict, task, timed: bool) -> set:
    """The records a full stage-2 judgment under `cfg` keeps (kernels.evolve.core.judge): every seen and held-out
    shape x dtype x seed 0-2, the two most-called seen shapes at each value scale (seed 7), and with `timed` the
    four timed buffers of every timed shape. A candidate cannot drop or swap any of them unnoticed."""
    import inspect
    from kernels.evolve.core import SCALES
    from kernels.timing import time_pair
    dts = [str(d).removeprefix("torch.") for d in task.dtypes]
    seen, held = [list(x) for x in cfg["seen"]], [list(x) for x in cfg.get("heldout", [])]
    out = {("check", json.dumps(s), d, sd, None) for s in seen + held for d in dts for sd in (0, 1, 2)}
    out |= {("check", json.dumps(s), d, 7, k) for k in (SCALES if task.scales is None else task.scales)
            for s in seen[:2] for d in dts}
    if timed:
        n = inspect.signature(time_pair).parameters["n_buffers"].default
        out |= {("time", json.dumps(s), dts[0], i, None) for s in seen + held for i in range(n)}
    return out


def verify(cfg: dict, path: Path, timed: bool = False) -> tuple[bool, str]:
    import stat
    import torch
    from kernels.check import TOLERANCES, _compare, _gen, budget_atol
    from kernels.evolve.core import _scaled
    from kernels.tasks import from_config
    path = Path(path)
    st = path.lstat() if path.is_symlink() or path.exists() else None
    if st is None:
        return False, "the candidate's process left no outputs to verify"
    if not stat.S_ISREG(st.st_mode) or st.st_size > MAX_BYTES:    # a link, a FIFO or a huge file: never opened here
        return False, "the outputs file is not a regular file under the size cap"
    recs = torch.load(path, weights_only=True).get("records") or []
    if not recs:
        return False, "no outputs were recorded"
    task, device = from_config(cfg), cfg.get("device", "cpu")
    have = {(r["kind"], json.dumps(list(r["shape"])), r["dtype"], r["seed"], r.get("scale")) for r in recs}
    missing = expected(cfg, task, timed) - have
    if missing:
        k = sorted(missing, key=str)[0]
        return False, f"{len(missing)} expected outputs were not recorded (e.g. {k[0]} at shape {k[1]} seed {k[3]})"
    tol = {**TOLERANCES, **{getattr(torch, k): tuple(v) for k, v in (cfg.get("tolerances") or {}).items()}}
    for r in recs:
        dt = getattr(torch, r["dtype"])
        make = task.make_inputs if r.get("scale") is None else _scaled(task.make_inputs, r["scale"])
        inputs = make(tuple(r["shape"]), dt, device, _gen(r["seed"]))
        ref, (atol, rtol) = task.reference(*[t.clone() for t in inputs]), tol[dt]
        if task.error_budget:                   # the baseline is the engine's own code, never the candidate's
            atol = budget_atol(ref, task.bind(task.baseline())(*[t.clone() for t in inputs]), atol, task.error_budget)
        d = _compare(ref, [o.to(device) for o in r["outputs"]], atol, rtol)
        if not d["ok"]:
            why = d.get("error") or f"max_abs {d['max_abs']:.3g}, max_rel {d['max_rel']:.3g}"
            return False, f"{r['kind']} output at shape {r['shape']} seed {r['seed']} differs from the reference: {why}"
    return True, f"{len(recs)} recorded outputs match the reference, recomputed outside the candidate's process"
