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
    assert r["metrics"]["combined_score"] > 0, r["artifacts"]          # timed: the roofline's bytes read `cfg` too
    assert t.cost((4, 64), torch.float32)[1] > 0
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


def test_the_stations_callees_are_watched_when_they_import(tmp_path, monkeypatch):
    regions = engine(tmp_path, monkeypatch)
    (tmp_path / "fake_fused_mod.py").write_text("def f(x):\n    return x\n")
    monkeypatch.setattr(regions, "STATION_MODULES", ("fake_fused_mod", "no_such_pkg.models.fused"))
    t = regions.from_engine("norm", "add", shared_weights=str(held(tmp_path, regions).path))
    assert "fake_fused_mod" in t.watch and "fake_stations_mod" in t.watch and "no_such_pkg.models.fused" not in t.watch


def _engine_repo(tmp, monkeypatch):
    """A git repo of a fake engine package holding the fake stations at models/stations.py, with no regions package."""
    import subprocess
    from kernels import regions
    repo = tmp / "engine"
    pkg = repo / "src" / "fake_engine" / "models"
    pkg.mkdir(parents=True)
    (pkg.parent / "__init__.py").write_text("")
    (pkg / "__init__.py").write_text("")
    (pkg / "stations.py").write_text(FAKE)
    git = lambda *a: subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True, check=True).stdout
    git("init", "-q")
    git("add", ".")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "engine")
    monkeypatch.syspath_prepend(str(repo / "src"))
    monkeypatch.setattr(regions, "ENGINE", "fake_engine.models.stations")
    monkeypatch.setattr(regions, "OVERRIDES_PACKAGE", "fake_engine.models.regions")
    for k in ("GIT_COMMITTER_NAME", "GIT_AUTHOR_NAME"):
        monkeypatch.setenv(k, "t")
    for k in ("GIT_COMMITTER_EMAIL", "GIT_AUTHOR_EMAIL"):
        monkeypatch.setenv(k, "t@t")
    return regions, repo, git


def test_a_region_ships_as_an_override_module_never_through_the_entry_path(tmp_path, monkeypatch):
    import pytest
    from kernels import pipeline
    from kernels.evolve import ship as ship_mod
    from kernels.evolve.run import initial_program
    regions, repo, git = _engine_repo(tmp_path, monkeypatch)
    h = held(tmp_path, regions)
    t = regions.from_engine("norm", "add", shared_weights=str(h.path))
    program = initial_program(t)
    assert program == regions.start_program("norm", "add", shared_weights=str(h.path))
    with pytest.raises(ValueError, match="holder"):
        initial_program(regions.from_engine("norm", "add"))
    for f in (pipeline.module_file, lambda t, d: pipeline.start_program(t, d), lambda t, d: pipeline.patch(t, d, "")):
        with pytest.raises(ValueError, match="region"):
            f(t, tmp_path)
    monkeypatch.setattr(ship_mod, "unstart", lambda *a: pytest.fail("the entry path was used for a region"))
    ok = {"metrics": {"correct": 1.0}, "artifacts": {"gates": json.dumps(
        {"correctness_heldout": {"passed": True, "cases": 6}, "physics": {"passed": True}})}}
    out = ship_mod.ship(t, program, ok, repo, branch="kernels/region")
    path = "src/fake_engine/models/regions/norm_add.py"
    assert out["path"] == path
    assert sorted(git("diff", "--name-only", out["base"], "kernels/region").split()) == \
        ["src/fake_engine/models/regions/__init__.py", path]                # stations.py untouched
    src = git("show", f"kernels/region:{path}")
    assert "OVERRIDES = {('norm', 'add'): _override}" in src and "EVOLVE" not in src
    ns: dict = {}
    exec(compile(src, path, "exec"), ns)
    hh = torch.randn(4, 64)
    st = regions._engine()
    c = {"h": hh.clone(), "cfg": st.LayerConfig(1e-6), "w.mm": h.tensors["w.mm"], "w.mm_scale": None, "kv": torch.zeros(2)}
    got = ns["OVERRIDES"][("norm", "add")](c)
    want = t.baseline()(hh.clone(), h.tensors["w.mm"])
    assert list(got) == ["h"] and torch.equal(got["h"], want[0]) and torch.equal(c["h"], hh)   # h not written in place
    import pytest
    for k, v in (("w.mm_scale", torch.ones(1)), ("cfg", st.LayerConfig(1e-5))):    # fp8 scales for bf16; another cfg
        with pytest.raises(ValueError, match=f"passes \\['{k}'\\] unlike"):
            ns["OVERRIDES"][("norm", "add")]({**c, k: v})


def test_a_region_overlapping_an_override_already_shipped_is_refused(tmp_path, monkeypatch):
    import pytest
    from kernels.evolve import ship as ship_mod
    from kernels.evolve.run import initial_program
    regions, repo, git = _engine_repo(tmp_path, monkeypatch)
    h = held(tmp_path, regions)
    ok = {"metrics": {"correct": 1.0}, "artifacts": {"gates": json.dumps(
        {"correctness_heldout": {"passed": True, "cases": 6}, "physics": {"passed": True}})}}

    def ship(first, last, base, branch):
        t = regions.from_engine(first, last, shared_weights=str(h.path))
        return ship_mod.ship(t, initial_program(t), ok, repo, base=base, branch=branch)
    ship("mm", "mm", "HEAD", "a")
    ship("act", "add", "a", "b")                                            # beside it: shipped
    with pytest.raises(ValueError, match=r"norm\.\.act overlaps the override (act\.\.add|mm\.\.mm) already in"):
        ship("norm", "act", "b", "c")


def test_the_search_cli_takes_a_region_with_its_samples_and_holder(tmp_path, monkeypatch):
    import argparse
    import pytest
    from kernels.evolve import run
    from kernels.search import __main__ as cli
    regions = engine(tmp_path, monkeypatch)
    h = held(tmp_path, regions)
    caps = captures(tmp_path, [(4, 64, 0, True)] * 3 + [(16, 64, 0, True)] * 2 + [(64, 64, 0, True)])
    for k in ("KERNEL_EVAL_CONFIG", "KERNELS_EVAL_LOG"):
        monkeypatch.setenv(k, "")
    monkeypatch.setattr(cli, "Claude", lambda *a, **k: None)
    monkeypatch.setattr(cli, "search", lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt))   # stop before it
    out = tmp_path / "out"
    with pytest.raises(KeyboardInterrupt):
        cli.main(["--task", "region:norm:add", "--samples", str(caps), "--shared-weights", str(h.path),
                  "--out", str(out), "--judge-model", "none", "--device", "cpu"])
    cfg = json.loads((out / "eval_config.json").read_text())
    assert cfg["task"] == "region:norm:add" and cfg["samples"] == str(caps.resolve())
    assert cfg["shared_weights"] == str(h.path.resolve()) and cfg["seen"] == [[4, 64], [16, 64]]   # split from samples
    assert "_CONST['w.mm_scale'] = None" in (out / "initial_program.py").read_text()
    gates = []
    monkeypatch.setattr(run, "judge_jailed", lambda cfg, *a, **k: gates.append(cfg) or {"metrics": {}})
    run.main(["gate", "--task", "region:norm:add", "--samples", str(caps), "--shared-weights", str(h.path),
              "--program", str(out / "initial_program.py"), "--device", "cpu"])
    assert gates[0]["heldout"] == [[64, 64]] and gates[0]["samples"] == cfg["samples"]
    assert gates[0]["shared_weights"] == cfg["shared_weights"]
    with pytest.raises(argparse.ArgumentTypeError):
        run.task_name("region:norm")


def test_a_region_task_with_a_remote_evaluator_is_refused(monkeypatch):
    from types import SimpleNamespace
    from kernels.evolve.run import region_args
    monkeypatch.setenv("KERNEL_EVAL_SSH", '["ssh", "host"]')
    try:
        region_args(SimpleNamespace(task="region:norm:add", samples="s", shared_weights="w"))
        raise AssertionError("a remote region search was not refused")
    except SystemExit as e:
        assert "GPU host" in str(e)
