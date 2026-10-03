"""Kernel equivalence on RMSNorm: a correct candidate passes; each kind of broken or cheating kernel is caught.
Run: python tests/test_kernels.py  (CPU always; correctness tests also on MPS when available)"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from blamegraph.kernels import check_kernel, check_memoization, time_kernel  # noqa: E402

SEEN, HELDOUT = [(8, 64), (16, 128)], [(5, 96), (3, 200)]
DTYPES = (torch.float32, torch.float16, torch.bfloat16)
EPS = 1e-6


def make_inputs(shape, dtype, device, g):
    return (torch.randn(shape, generator=g).to(device, dtype), torch.randn(shape[-1], generator=g).to(device, dtype))


def reference(x, w):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS) * w.float()).to(x.dtype)


def correct(x, w):
    xf = x.float()
    return (xf / torch.sqrt((xf * xf).sum(-1, keepdim=True) / x.shape[-1] + EPS) * w.float()).to(x.dtype)


def wrong(x, w):
    return (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + EPS)).to(x.dtype)


def seen_only(x, w):
    return correct(x, w) if tuple(x.shape) in SEEN else wrong(x, w)


def mutates(x, w):
    out = correct(x, w)
    x.zero_()
    return out


def drops_nan(x, w):
    return torch.nan_to_num(correct(x, w), nan=0.0, posinf=0.0, neginf=0.0)


_ptr_cache = {}


def ptr_memo(x, w):
    key = (x.data_ptr(), tuple(x.shape), x.dtype)
    if key not in _ptr_cache:
        _ptr_cache[key] = correct(x, w)
    return _ptr_cache[key]


_content_cache = {}


def slow(x, w):
    for _ in range(10):
        out = correct(x, w)
    return out


def content_memo(x, w):
    key = (x.data_ptr(), x.sum().item(), w.sum().item())
    if key not in _content_cache:
        _content_cache[key] = slow(x, w)
    return _content_cache[key]


def _check(cand, device="cpu", **kw):
    return check_kernel(reference, cand, make_inputs, SEEN, HELDOUT, DTYPES, device, **kw)


def _failed(v):
    return {k for k, g in v["gates"].items() if g["passed"] is not True}


def test_correct_passes(device="cpu"):
    v = _check(correct, device)
    assert v["passed"], v["reasons"]
    assert v["gates"]["correctness_heldout"]["evidence"]["max_abs"]["float32"] < 1e-5


def test_wrong_fails(device="cpu"):
    v = _check(wrong, device)
    assert {"correctness_seen", "correctness_heldout"} <= _failed(v), v["reasons"]
    assert "max_abs" in v["gates"]["correctness_seen"]["reason"]


def test_shape_specialized_fails_heldout(device="cpu"):
    v = _check(seen_only, device)
    assert v["gates"]["correctness_seen"]["passed"] and not v["gates"]["correctness_heldout"]["passed"], v["reasons"]


def test_mutation_fails(device="cpu"):
    assert _failed(_check(mutates, device)) == {"inputs_unchanged"}


def test_aliasing_fails(device="cpu"):
    v = check_kernel(lambda x, w: x.clone(), lambda x, w: x.view_as(x), make_inputs, SEEN, HELDOUT, DTYPES, device)
    assert _failed(v) == {"no_aliasing"}, v["reasons"]


def test_dropped_nan_fails(device="cpu"):
    v = _check(drops_nan, device)
    assert _failed(v) == {"edge_cases"}, v["reasons"]
    assert "nonfinite" in v["gates"]["edge_cases"]["reason"]


def test_ptr_memoization_fails(device="cpu"):
    _ptr_cache.clear()
    v = _check(ptr_memo, device)
    assert "memoization" in _failed(v), v["reasons"]


def test_nondeterminism_gated_only_when_required(device="cpu"):
    n = [0]

    def alternating(x, w):
        """Every third call is one ulp off."""
        n[0] += 1
        out = correct(x, w)
        return out if n[0] % 3 else torch.nextafter(out, torch.full_like(out, float("inf")))
    v = check_kernel(reference, alternating, make_inputs, SEEN, HELDOUT, (torch.float32,), device,
                     tolerances={torch.float32: (1e-4, 1e-4)})
    assert v["gates"]["determinism"]["passed"] and v["gates"]["determinism"]["evidence"]["count"] > 0
    v = check_kernel(reference, alternating, make_inputs, SEEN, HELDOUT, (torch.float32,), device,
                     tolerances={torch.float32: (1e-4, 1e-4)}, require_deterministic=True)
    assert not v["gates"]["determinism"]["passed"]


def test_timing_sane():
    t = time_kernel(correct, make_inputs, (64, 1024), torch.float32, iters=20, warmup=3)
    assert 0 < t["p25_ms"] <= t["median_ms"] <= t["p75_ms"] < 1000 and t["iqr_ms"] >= 0


def test_content_memoization_flagged_by_timing():
    shape = (256, 2048)
    assert check_memoization(slow, make_inputs, shape, torch.float32)["passed"]
    _content_cache.clear()
    g = check_memoization(content_memo, make_inputs, shape, torch.float32)
    assert not g["passed"], g["reason"]


DEVICE_TESTS = [test_correct_passes, test_wrong_fails, test_shape_specialized_fails_heldout, test_mutation_fails,
                test_aliasing_fails, test_dropped_nan_fails, test_ptr_memoization_fails]

if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
    if torch.backends.mps.is_available():
        for fn in DEVICE_TESTS:
            fn("mps"); print("ok", fn.__name__, "[mps]")
    else:
        print("skip mps tests (not available)")
