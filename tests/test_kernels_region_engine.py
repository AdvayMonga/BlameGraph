"""Regions bound to the engine's stations (kernels/regions.from_engine), with a fake stations module shaped as
data/notes/handoff-engine-stations.md asks: inputs from a CAPTURE_REGION dir, weights attached from a shared holder,
the generated start program judged end to end by the evaluator on the CPU, and `region:` names through the loader."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json  # noqa: E402

import torch  # noqa: E402

FAKE = """
import torch
from dataclasses import dataclass, replace

@dataclass(frozen=True)
class Station:
    name: str
    reads: tuple
    writes: tuple
    fn: object

@dataclass(frozen=True)
class LayerConfig:
    eps: float

LIVE = ("h", "kv")
IN_PLACE = ("kv",)

def _norm(c):
    x = c["h"].float()
    return {"n": (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + c["cfg"].eps)).to(c["h"].dtype)}

def _mm(c):
    w = c["w.mm"] if c["w.mm_scale"] is None else c["w.mm"] * c["w.mm_scale"]      # bf16 experts carry no scale
    return {"y": c["n"] @ w}

def _add(c):
    c["h"].add_(c["a"])                       # in place, as the engine's residual add is
    return {"h": c["h"]}

def _kv(c):
    c["kv"][0] = c["h"].sum()
    return {"kv": c["kv"]}

DECODE = (
    Station("norm", ("h", "cfg"), ("n",), _norm),
    Station("mm", ("n", "w.mm", "w.mm_scale"), ("y",), _mm),
    Station("act", ("y",), ("a",), lambda c: {"a": torch.nn.functional.silu(c["y"])}),
    Station("add", ("h", "a"), ("h",), _add),
    Station("kv", ("h", "kv"), ("kv",), _kv),
)

def functional(stations=DECODE):
    def pure(fn):
        return lambda c: fn({**c, "h": c["h"].clone()})
    return tuple(replace(s, fn=pure(s.fn)) if s.name == "add" else s for s in stations)

def reference():
    return functional(DECODE)
"""


def engine(tmp, monkeypatch):
    from kernels import regions
    (tmp / "fake_stations_mod.py").write_text(FAKE)
    monkeypatch.syspath_prepend(str(tmp))
    monkeypatch.setattr(regions, "ENGINE", "fake_stations_mod")
    return regions


def captures(tmp, calls):
    """A CAPTURE_REGION dir of region:norm:add calls, (rows, hidden, layer, sampled) each."""
    d = tmp / "caps"
    (d / "samples").mkdir(parents=True)
    with open(d / "calls.jsonl", "w") as f:
        for i, (t, h, layer, sampled) in enumerate(calls):
            call = {"fn": "region:norm:add", "layer": layer,
                    "reads": {"h": {"shape": [t, h], "dtype": "float32"}, "cfg": {"value": {"eps": 1e-6}},
                              "w.mm": {"shape": [h, h]}}}
            if sampled:
                torch.save({"reads": {"h": torch.randn(t, h), "cfg": {"eps": 1e-6}}, "writes": {}},
                           d / "samples" / f"{i}.pt")
                call["sample"] = f"{i}.pt"
            else:
                call["sample_skipped_bytes"] = 2**30
            f.write(json.dumps(call) + "\n")
    return d


def held(tmp, regions, layer=0):
    from kernels.shared_weights import Holder
    return Holder({"w.mm": torch.randn(64, 64) / 8, "w.mm_scale": None}, tmp / "w.pkl",
                  {"cfg": regions._engine().LayerConfig(1e-6), "layer": layer})


def test_a_region_of_the_engines_stations_is_judged_end_to_end(tmp_path, monkeypatch):
    from kernels.evolve import core
    regions = engine(tmp_path, monkeypatch)
    h = held(tmp_path, regions)
    caps = captures(tmp_path, [(4, 64, 0, True), (16, 64, 0, True)])
    t = regions.from_engine("norm", "add", str(caps), str(h.path), dtypes=(torch.float32,))
    assert t.name == "region:norm:add" and set(t.seen_default) == {(4, 64), (16, 64)}
    xs = t.make_inputs((4, 64), torch.float32, "cpu", torch.Generator().manual_seed(0))
    assert len(xs) == 2 and xs[1].shape == (64, 64)                  # h, then the weight; its None scale bound
    start = tmp_path / "start.py"
    start.write_text(regions.start_program("norm", "add", shared_weights=str(h.path)))
    assert "_INPUTS = ['h', 'w.mm']" in start.read_text() and "_CONST['w.mm_scale'] = None" in start.read_text()
    assert core.screen(start.read_text()) == []
    r = core.judge(t, start, [(4, 64), (16, 64)], [], None, "cpu", None, 2, iters=3)
    assert r["metrics"]["correct"] == 1.0, r["artifacts"]
    assert h.changed() == []


def test_region_names_go_through_the_task_loader(tmp_path, monkeypatch):
    from kernels.tasks import from_config
    regions = engine(tmp_path, monkeypatch)
    caps = captures(tmp_path, [(4, 64, 0, True)])
    t = from_config({"task": "region:norm:act", "samples": str(caps), "shared_weights": str(held(tmp_path, regions).path)})
    assert t.name == "region:norm:act" and t.error_budget == 2.0 and "fake_stations_mod" in t.watch


def test_a_config_read_with_no_holder_is_refused_clearly(tmp_path, monkeypatch):
    regions = engine(tmp_path, monkeypatch)
    caps = captures(tmp_path, [(4, 64, 0, True)])
    import pytest
    with pytest.raises(ValueError, match=r"\['cfg', 'w.mm', 'w.mm_scale'\].*shared_weights"):
        regions.from_engine("norm", "add", str(caps))


def test_a_region_writing_in_place_is_refused(tmp_path, monkeypatch):
    regions = engine(tmp_path, monkeypatch)
    import pytest
    with pytest.raises(ValueError, match="in place"):
        regions.from_engine("add", "kv")


def test_captured_shapes_split_with_a_heldout_region_from_the_holders_layer_only(tmp_path, monkeypatch):
    from kernels import shapes
    regions = engine(tmp_path, monkeypatch)
    calls = [(4, 64, 0, True)] + [(4, 64, 0, False)] * 4 + [(16, 64, 0, True)] * 3 + [(64, 64, 0, True)]
    calls += [(32, 64, 1, True), (8, 64, 0, False)]          # another layer's; a shape whose sample was over the cap
    caps = captures(tmp_path, calls)
    t = regions.from_engine("norm", "add", str(caps), str(held(tmp_path, regions).path), dtypes=(torch.float32,))
    assert t.heldout_axes == (0,) and set(t.seen_default) == {(4, 64), (16, 64), (64, 64)}
    counts = shapes.from_captures(t, [caps])                 # read by the task's capture fn, not its entry
    assert counts == {(4, 64): 5, (16, 64): 3, (64, 64): 1} and shapes.ints_from_captures(t, [caps]) == {}
    sets = shapes.split(t, counts)
    assert sets["heldout"] == [(64, 64)] and sets["seen"] == [(4, 64), (16, 64)]


def test_seed_zero_is_the_capture_and_later_seeds_perturb_it_differently(tmp_path, monkeypatch):
    from kernels.check import _gen
    regions = engine(tmp_path, monkeypatch)
    caps = captures(tmp_path, [(4, 64, 0, True)])
    t = regions.from_engine("norm", "add", str(caps), str(held(tmp_path, regions).path), dtypes=(torch.float32,))
    saved = torch.load(caps / "samples" / "0.pt", weights_only=True)["reads"]["h"]
    h0, h1, h2 = (t.make_inputs((4, 64), torch.float32, "cpu", _gen(s))[0] for s in (0, 1, 2))
    assert torch.equal(h0, saved) and not torch.equal(h1, saved) and not torch.equal(h2, saved)
    assert not torch.equal(h1, h2) and torch.equal(h1, t.make_inputs((4, 64), torch.float32, "cpu", _gen(1))[0])


def test_the_pools_block_count_is_not_part_of_a_calls_shape(tmp_path, monkeypatch):
    regions = engine(tmp_path, monkeypatch)                                 # the fake's IN_PLACE is ("kv",)
    small = {"h": {"shape": [4, 64]}, "kv": {"shape": [100, 2, 16, 8]}, "attn_block_tables": {"shape": [4, 7]},
             "w.mm": {"shape": [64, 64]}}
    large = {**small, "kv": {"shape": [4000, 2, 16, 8]}}
    assert regions._signature(small) == regions._signature(large) == (4, 7, 4, 64)
    assert regions._heldout_axes(small, ("h", "kv", "attn_block_tables")) == (2, 1)
    monkeypatch.delattr(regions._engine(), "IN_PLACE")                      # no IN_PLACE: the engine's pool names
    assert regions._signature({"h": {"shape": [4, 64]}, "k_cache": {"shape": [9, 2, 16, 8]}}) == (4, 64)
