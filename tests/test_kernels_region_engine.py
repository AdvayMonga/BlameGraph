"""Regions bound to the engine's stations (kernels/regions.from_engine), with a fake stations module shaped as
data/notes/handoff-engine-stations.md asks: inputs from a CAPTURE_REGION dir, weights attached from a shared holder,
the generated start program judged end to end by the evaluator on the CPU, and `region:` names through the loader."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json  # noqa: E402

import torch  # noqa: E402

FAKE = '''
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

LIVE = ("h", "k_cache", "v_cache")

def _norm(c):
    x = c["h"].float()
    return {"n": (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + c["cfg"].eps)).to(c["h"].dtype)}

def _add(c):
    c["h"].add_(c["a"])                       # in place, as the engine's residual add is
    return {"h": c["h"]}

DECODE = (
    Station("norm", ("h", "cfg"), ("n",), _norm),
    Station("mm", ("n", "w.mm"), ("y",), lambda c: {"y": c["n"] @ c["w.mm"]}),
    Station("act", ("y",), ("a",), lambda c: {"a": torch.nn.functional.silu(c["y"])}),
    Station("add", ("h", "a"), ("h",), _add),
)

def functional(stations=DECODE):
    def pure(fn):
        return lambda c: fn({**c, "h": c["h"].clone()})
    return tuple(replace(s, fn=pure(s.fn)) if s.name == "add" else s for s in stations)

def reference():
    return functional(DECODE)
'''


def engine(tmp, monkeypatch):
    from kernels import regions
    (tmp / "fake_stations_mod.py").write_text(FAKE)
    monkeypatch.syspath_prepend(str(tmp))
    monkeypatch.setattr(regions, "ENGINE", "fake_stations_mod")
    return regions


def captures(tmp, shapes):
    d = tmp / "caps"
    (d / "samples").mkdir(parents=True)
    with open(d / "calls.jsonl", "w") as f:
        for i, (t, h) in enumerate(shapes):
            x = torch.randn(t, h)
            torch.save({"reads": {"h": x}, "writes": {}}, d / "samples" / f"{i}.pt")
            f.write(json.dumps({"fn": "region:norm:add", "reads": {"h": {"shape": [t, h], "dtype": "float32"},
                                                                    "w.mm": {"shape": [h, h]}},
                                "sample": f"{i}.pt"}) + "\n")
    return d


def test_a_region_of_the_engines_stations_is_judged_end_to_end(tmp_path, monkeypatch):
    from kernels.evolve import core
    from kernels.shared_weights import Holder
    regions = engine(tmp_path, monkeypatch)
    torch.multiprocessing.set_sharing_strategy("file_system")
    h = Holder({"w.mm": torch.randn(64, 64) / 8}, tmp_path / "w.pkl", {"cfg": regions._engine().LayerConfig(1e-6)})
    caps = captures(tmp_path, [(4, 64), (16, 64)])
    t = regions.from_engine("norm", "add", str(caps), str(h.path), dtypes=(torch.float32,))
    assert t.name == "region:norm:add" and set(t.seen_default) == {((4, 64),), ((16, 64),)}
    xs = t.make_inputs(((4, 64),), torch.float32, "cpu", torch.Generator().manual_seed(0))
    assert len(xs) == 2 and xs[1].shape == (64, 64)                  # h, then the weight
    start = tmp_path / "start.py"
    start.write_text(regions.start_program("norm", "add", {"cfg": regions._engine().LayerConfig(1e-6)}))
    assert core.screen(start.read_text()) == []
    r = core.judge(t, start, [((4, 64),), ((16, 64),)], [], None, "cpu", None, 2, iters=3)
    assert r["metrics"]["correct"] == 1.0, r["artifacts"]
    assert h.changed() == []


def test_region_names_go_through_the_task_loader(tmp_path, monkeypatch):
    from kernels.tasks import from_config
    engine(tmp_path, monkeypatch)
    caps = captures(tmp_path, [(4, 64)])
    t = from_config({"task": "region:norm:act", "samples": str(caps)})
    assert t.name == "region:norm:act" and t.error_budget == 2.0 and "fake_stations_mod" in t.watch
