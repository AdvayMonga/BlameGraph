"""Regions (kernels/regions.py): the interface, the generated baseline and fp32 reference, the bytes a region moves,
and the error budget in the checker (a candidate may be at most 2x as wrong as the baseline). CPU, fake stations."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from kernels import check_kernel
from kernels.regions import Station, bytes_moved, functions, interface, task


def _norm(c):
    x = c["x"].float()
    return {"n": (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)).to(c["x"].dtype)}


STATIONS = (
    Station("norm", ("x",), ("n",), _norm),
    Station("mm", ("n", "w"), ("y",), lambda c: {"y": c["n"] @ c["w"]}),
    Station("act", ("y",), ("a",), lambda c: {"a": torch.nn.functional.silu(c["y"])}),
    Station("add", ("x", "a"), ("x",), lambda c: {"x": c["x"] + c["a"]}),
)


def ctx(shape, dtype, device, g):
    t, h = shape
    return {"x": torch.randn(t, h, generator=g).to(dtype), "w": (torch.randn(h, h, generator=g) / h ** 0.5).to(dtype)}


def test_the_interface_is_what_the_region_reads_first_and_what_is_read_after_it():
    assert interface(STATIONS, "norm", "act") == (("x", "w"), ("a",))           # n and y stay inside
    assert interface(STATIONS, "mm", "mm") == (("n", "w"), ("y",))
    assert interface(STATIONS, "norm", "add", live=("x",)) == (("x", "w"), ("x",))


def test_the_reference_is_fp32_returned_in_the_baselines_dtype():
    base, ref, ins, outs = functions(STATIONS, "norm", "add", live=("x",))
    xs = tuple(ctx((4, 64), torch.bfloat16, "cpu", torch.Generator().manual_seed(0))[i] for i in ins)
    b, r = base(*xs), ref(*xs)
    assert b[0].dtype == r[0].dtype == torch.bfloat16 and not torch.equal(b[0], r[0])
    assert (b[0].float() - r[0].float()).abs().max() < 0.1


def test_fusing_moves_fewer_bytes_than_the_stations_one_by_one():
    c = ctx((8, 64), torch.float32, "cpu", torch.Generator().manual_seed(0))
    m = bytes_moved(STATIONS, "norm", "act", c)
    assert m["fused_bytes"] == (8 * 64 + 64 * 64 + 8 * 64) * 4 and m["unfused_bytes"] > m["fused_bytes"]


def test_the_error_budget_is_per_element_twice_the_baselines_own_error_there():
    t = task("toy_region", STATIONS, "norm", "add", ctx, lambda call: None, live=("x",),
             seen_default=((4, 64),))
    base, ref = t.baseline(), t.reference

    def toward(k):                                          # each element moved k times the baseline's error there
        return lambda *a: ((ref(*a)[0].float() + k * (base(*a)[0].float() - ref(*a)[0].float())).to(torch.bfloat16),)
    tol = {torch.bfloat16: (1e-6, 0.0)}                       # tiny dtype tolerance: the budget decides
    common = dict(dtypes=(torch.bfloat16,), tolerances=tol, edge_shapes=[], layouts=(), seeds=(0, 1), repeats=1,
                  baseline=base, budget=t.error_budget)
    assert check_kernel(ref, toward(1.5), t.make_inputs, [(4, 64)], [], **common)["passed"]
    assert not check_kernel(ref, toward(3.0), t.make_inputs, [(4, 64)], [], **common)["passed"]
    assert not check_kernel(ref, toward(1.5), t.make_inputs, [(4, 64)], [],
                            **{**common, "budget": None})["passed"]       # without the budget the dtype's rule holds


def test_one_large_value_loosens_nothing_else():
    """Outlier channels (x1000) make the baseline's largest error large; a candidate that zeroes the rest must fail."""
    def outliers(shape, dtype, device, g):
        c = ctx(shape, dtype, device, g)
        c["x"] = c["x"].float()
        c["x"][:, :8] *= 1000
        c["x"] = c["x"].to(dtype)
        return c
    t = task("toy_region", STATIONS, "norm", "add", outliers, lambda call: None, live=("x",))
    base, ref = t.baseline(), t.reference

    def zero_most(*a):
        o = ref(*a)[0].clone()
        o[:, 8:] = 0
        return (o,)
    r = check_kernel(ref, zero_most, t.make_inputs, [(256, 1024)], [], dtypes=(torch.bfloat16,),
                     tolerances={torch.bfloat16: (1e-6, 0.0)}, edge_shapes=[], layouts=(), seeds=(0, 1, 2), repeats=1,
                     baseline=base, budget=t.error_budget)
    assert not r["passed"]


def test_a_region_is_judged_end_to_end_by_the_evaluator(tmp_path):
    """Stage 2 on the CPU: the stations as the program pass and time against themselves; a program off by far more
    than the budget fails."""
    from kernels.evolve import core
    t = task("toy_region", STATIONS, "norm", "add", ctx, lambda call: None, live=("x",), dtypes=(torch.float32,))
    good = tmp_path / "good.py"
    good.write_text("import torch\n\ndef kernel(x, w):\n    xf = x.float()\n"
                    "    n = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype)\n"
                    "    return (x + torch.nn.functional.silu(n @ w),)\n")
    bad = tmp_path / "bad.py"
    bad.write_text("import torch\n\ndef kernel(x, w):\n    return (x * 1.5,)\n")
    r = core.judge(t, good, [(4, 64)], [], None, "cpu", None, 2, iters=3)
    assert r["metrics"]["correct"] == 1.0, r["artifacts"]
    assert core.judge(t, bad, [(4, 64)], [], None, "cpu", None, 2, iters=3)["metrics"]["correct"] == 0.0


if __name__ == "__main__":
    import inspect
    import tempfile
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(*[Path(tempfile.mkdtemp()) for _ in inspect.signature(fn).parameters]); print("ok", name)
