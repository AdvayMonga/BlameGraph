"""Kernel equivalence: a candidate kernel against a reference op on seen, held-out and edge-case inputs."""
from __future__ import annotations

import torch

TOLERANCES = {torch.float32: (1e-5, 1e-5), torch.float16: (1e-3, 1e-3), torch.bfloat16: (1e-2, 1e-2)}  # (atol, rtol)


def _gen(seed):
    return torch.Generator().manual_seed(seed)


def _flat(x):
    return list(x) if isinstance(x, (tuple, list)) else [x]


def _bits(t):
    return t.detach().reshape(-1).contiguous().view(torch.uint8)


def _span(t):
    """Storage byte range; on MPS data_ptr is a buffer handle, so only identical storages compare as overlapping."""
    s = t.untyped_storage()
    return s.data_ptr(), s.data_ptr() + (1 if t.device.type == "mps" else s.nbytes())


def _strided(t):
    """Same values in a non-contiguous view (every other element of a buffer twice as wide)."""
    if t.dim() == 0:
        return t
    return torch.empty((*t.shape[:-1], 2 * t.shape[-1]), dtype=t.dtype, device=t.device)[..., ::2].copy_(t)


def _transposed(t):
    """Same values in a column-major view."""
    return t.transpose(-1, -2).contiguous().transpose(-1, -2) if t.dim() >= 2 else t


def _poison(inputs):
    """Write NaN, +Inf and -Inf into the first floating-point input."""
    v = next(t for t in inputs if t.is_floating_point()).view(-1)
    v[0], v[v.numel() // 2], v[-1] = float("nan"), float("inf"), float("-inf")


def _edge_shapes(shape):
    """Each dim of shape set to 1, plus all power-of-two dims bumped by one."""
    ones = [tuple(1 if j == i else d for j, d in enumerate(shape)) for i in range(len(shape))]
    return ones + [tuple(d + 1 if d > 1 and d & (d - 1) == 0 else d for d in shape)]


def _compare(ref, out, atol, rtol) -> dict:
    """Max abs/rel error over finite positions; non-finite positions (and Inf signs) must match exactly."""
    refs, outs = _flat(ref), _flat(out)
    if len(refs) != len(outs):
        return {"ok": False, "error": f"{len(outs)} outputs, expected {len(refs)}"}
    ok, max_abs, max_rel, nonfinite = True, 0.0, 0.0, 0
    for r, o in zip(refs, outs):
        if not isinstance(o, torch.Tensor) or o.shape != r.shape or o.dtype != r.dtype:
            got = (tuple(o.shape), str(o.dtype)) if isinstance(o, torch.Tensor) else type(o).__name__
            return {"ok": False, "error": f"output {got}, expected {(tuple(r.shape), str(r.dtype))}"}
        dt = torch.float64 if r.dtype == torch.float64 else torch.float32
        r, o = r.to(dt), o.to(dt)
        both = r.isfinite() & o.isfinite()
        nonfinite += int((~both & ~((r.isnan() & o.isnan()) | (r == o))).sum())
        d, mag = (r - o).abs()[both], r.abs()[both]
        if d.numel():
            max_abs = max(max_abs, d.max().item())
            max_rel = max(max_rel, (d / mag.clamp_min(atol)).max().item())
        ok = ok and bool((d <= atol + rtol * mag).all())
    return {"ok": ok and nonfinite == 0, "max_abs": max_abs, "max_rel": max_rel, "nonfinite_mismatch": nonfinite}


def _describe(c) -> str:
    where = f"{c.get('variant', '')} shape {c['shape']} {c['dtype']} seed {c['seed']}".strip()
    if "error" in c:
        return f"{where}: {c['error']}"
    return (f"{where}: max_abs {c['max_abs']:.3g}, max_rel {c['max_rel']:.3g}, non-finite mismatches "
            f"{c['nonfinite_mismatch']} (atol {c['atol']:.0e}, rtol {c['rtol']:.0e})")


def _summary(cases) -> tuple[bool, str, dict]:
    bad = [c for c in cases if not c["ok"]]
    ev = {"cases": len(cases), "failed": len(bad), "failures": bad[:5]}
    for k in ("max_abs", "max_rel"):
        ev[k] = {dt: max((c[k] for c in cases if c["dtype"] == dt and k in c), default=None)
                 for dt in dict.fromkeys(c["dtype"] for c in cases)}
    reason = f"{len(bad)}/{len(cases)} cases failed; first: {_describe(bad[0])}" if bad else f"{len(cases)} cases within tolerance"
    return not bad, reason, ev


def check_kernel(reference, candidate, make_inputs, seen_shapes, heldout_shapes, dtypes=(torch.float32,),
                 device="cpu", seeds=(0, 1, 2), tolerances=None, edge_shapes=None, require_deterministic=False,
                 repeats=3) -> dict:
    """Return {"passed", "reasons", "gates": {name: {passed, reason, evidence}}} for candidate vs reference.
    make_inputs(shape, dtype, device, generator) -> tuple of tensors; generator is a seeded CPU torch.Generator."""
    tol = {**TOLERANCES, **(tolerances or {})}
    mutated, aliased, nondet = [], [], []

    def run(shape, dtype, seed, inputs, variant=None, reps=1):
        """Candidate vs reference (reference gets clones); records mutation, aliasing and repeat mismatches."""
        atol, rtol = tol[dtype]
        case = {"shape": tuple(shape), "dtype": str(dtype).removeprefix("torch."), "seed": seed, "atol": atol, "rtol": rtol}
        if variant:
            case["variant"] = variant
        ref = reference(*[t.clone() for t in inputs])
        before = [t.clone() for t in inputs]
        try:
            out = candidate(*inputs)
            first = [o.clone() if isinstance(o, torch.Tensor) else o for o in _flat(out)]
            again = [_flat(candidate(*inputs)) for _ in range(reps - 1)]
        except Exception as e:
            return {**case, "ok": False, "error": f"raised {type(e).__name__}: {e}"}
        if any(not torch.equal(_bits(a), _bits(b)) for a, b in zip(before, inputs)):
            mutated.append(case)
        spans = [_span(t) for t in inputs]
        if any(a0 < b1 and b0 < a1 for o in _flat(out) if isinstance(o, torch.Tensor)
               for a0, a1 in [_span(o)] for b0, b1 in spans):
            aliased.append(case)
        if any(len(a) != len(first) or not all(torch.equal(_bits(x), _bits(y)) for x, y in zip(a, first)) for a in again):
            nondet.append(case)
        return {**case, **_compare(ref, first, atol, rtol)}

    def cases(shapes, seeds_, reps=1):
        return [run(s, dt, sd, make_inputs(s, dt, device, _gen(sd)), reps=reps)
                for s in shapes for dt in dtypes for sd in seeds_]

    gates = {}
    for name, cs in (("correctness_seen", cases(seen_shapes, seeds, repeats)),
                     ("correctness_heldout", cases(heldout_shapes, seeds))):
        ok, reason, ev = _summary(cs)
        gates[name] = {"passed": ok, "reason": reason, "evidence": ev}

    s0, sd0 = seen_shapes[0], seeds[0]
    edge = cases(edge_shapes or _edge_shapes(s0), seeds[:1])
    memo = []
    for dt in dtypes:
        for name, f in (("strided", _strided), ("transposed", _transposed)):
            edge.append(run(s0, dt, sd0, tuple(f(t) for t in make_inputs(s0, dt, device, _gen(sd0))), name))
        inputs = make_inputs(s0, dt, device, _gen(sd0))
        _poison(inputs)
        edge.append(run(s0, dt, sd0, inputs, "nonfinite"))
        inputs = make_inputs(s0, dt, device, _gen(sd0))
        run(s0, dt, sd0, inputs)
        for t, n in zip(inputs, make_inputs(s0, dt, device, _gen(sd0 + 1000))):
            t.copy_(n)
        memo.append(run(s0, dt, sd0 + 1000, inputs, "refilled_in_place"))
    for name, cs in (("edge_cases", edge), ("memoization", memo)):
        ok, reason, ev = _summary(cs)
        gates[name] = {"passed": ok, "reason": reason, "evidence": ev}

    for name, hits, what in (("inputs_unchanged", mutated, "inputs modified by the call"),
                             ("no_aliasing", aliased, "output shares storage with an input")):
        gates[name] = {"passed": not hits, "evidence": {"cases": hits[:5], "count": len(hits)},
                       "reason": f"{what} in {len(hits)} cases; first: {_describe({**hits[0], 'error': what})}"
                       if hits else "none"}
    n_rep = len(seen_shapes) * len(dtypes) * len(seeds)
    reason = (f"outputs differ across {repeats} calls on identical inputs in {len(nondet)}/{n_rep} cases"
              if nondet else f"bitwise identical across {repeats} calls in {n_rep} cases")
    gates["determinism"] = {"passed": not (nondet and require_deterministic),
                            "reason": reason + ("" if require_deterministic else " (not required)"),
                            "evidence": {"cases": nondet[:5], "count": len(nondet), "required": require_deterministic}}
    reasons = [f"{k}: {g['reason']}" for k, g in gates.items() if g["passed"] is not True]
    return {"passed": all(g["passed"] is True for g in gates.values()), "reasons": reasons, "gates": gates}
