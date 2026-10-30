"""Regions through the kernel layer's pipeline (kernels/pipeline.py, kernels/hold.py) with the controller unable to
load them: captures packed into one samples dir, the hold process's protocol (local and over a stand-in for ssh), its
description of a region on the fake stations, the winner as an override-module patch of the agent's tree, and the
`kernel_search` tool on a region end to end with the capture, hold, search and gate faked (no GPU, no model)."""
from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest_plugins = ["tests.test_lab_proftools"]   # its `tb` fixture: a Toolbox with fake instruments

from kernels import pipeline  # noqa: E402
from tests.test_kernels_region_engine import captures, engine, held  # noqa: E402

EXPECT = {"metric": "speedup", "value": 1.1, "unit": "absolute", "why": "fused"}
FAKE_HOLD = """
import json, sys
a = sys.argv[1:]
print("engine log line")
print(json.dumps({"holder": a[a.index("--out") + 1], "samples": a[a.index("--samples") + 1], "task": a[1]}), flush=True)
sys.stdin.read()
open(a[a.index("--samples") + 1] + "/../released", "w").write("1")
"""


def two_traffics(tmp):
    a, b = tmp / "a", tmp / "b"
    a.mkdir()
    b.mkdir()
    return [captures(a, [(4, 64, 0, True)] * 3 + [(64, 64, 0, True)]), captures(b, [(16, 64, 0, True)])]


def test_captures_pack_into_one_samples_dir_each_traffic_counted_apart(tmp_path, monkeypatch):
    from kernels import shapes
    regions = engine(tmp_path, monkeypatch)
    caps = two_traffics(tmp_path)
    pack = pipeline.pack(caps, tmp_path / "pack")
    t = regions.from_engine("norm", "add", str(pack), str(held(tmp_path, regions).path), dtypes=(torch.float32,))
    assert set(t.seen_default) == {(4, 64), (16, 64), (64, 64)}                # every traffic's samples, renamed
    traffic = sorted(pack.glob("traffic*/*/captures"))
    assert len(traffic) == 2 and shapes.from_captures(t, traffic, equal=True) == {(4, 64): 3, (64, 64): 1, (16, 64): 4}
    x = t.make_inputs((16, 64), torch.float32, "cpu", torch.Generator().manual_seed(0))[0]
    assert torch.equal(x, torch.load(caps[1] / "samples" / "0.pt", weights_only=True)["reads"]["h"])


def test_the_hold_side_describes_a_region_from_its_samples_and_holder(tmp_path, monkeypatch):
    from kernels import hold
    regions = engine(tmp_path, monkeypatch)
    pack = pipeline.pack(two_traffics(tmp_path), tmp_path / "pack")
    d = hold.describe("region:norm:add", pack, held(tmp_path, regions).path)
    assert d["shapes"]["heldout"] == [[64, 64]] and d["shapes"]["heldout_axes"] == [0]
    assert sorted(d["shapes"]["seen"]) == [[4, 64], [16, 64]] and d["stations"] == ["norm", "mm", "act", "add", "kv"]
    assert "_CONST['w.mm_scale'] = None" in d["start"] and json.loads(json.dumps(d)) == d     # one JSON line


@pytest.mark.parametrize("remote", [False, True])
def test_the_hold_process_runs_where_the_evaluator_does_and_ends_with_the_block(tmp_path, monkeypatch, remote):
    (tmp_path / "fake_hold.py").write_text(FAKE_HOLD)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.setattr(pipeline, "HOLD", "fake_hold")
    caps = two_traffics(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    pushed = []

    class Worker:                                # lab.worker.Remote's push and root, over the local disk
        def root(self):
            return str(tmp_path / "worker")

        def push(self, local, there):
            import shutil
            pushed.append(there)
            shutil.copytree(local, there)
    env = {"KERNEL_EVAL_SSH": json.dumps(["sh", "-c"]),
           "KERNEL_EVAL_REMOTE": f'cd ~ && PATH="{Path(sys.executable).parent}:$PATH" PYTHONPATH="{tmp_path}"'} \
        if remote else {}
    from kernels.tasks import load
    task = load("region:norm:add")
    with pipeline.hold(task, caps, out, env, Worker() if remote else None) as d:
        assert d["task"] == "region:norm:add" and Path(d["samples"]).is_dir()
        assert (Path(d["samples"]) / "calls.jsonl").exists() and d["holder"].endswith("/holder/weights.pkl")
        assert not (Path(d["samples"]).parent / "released").exists()
    assert (Path(d["samples"]).parent / "released").exists()                    # stdin closed: it let go
    assert pushed == ([str(tmp_path / "worker" / "lab/runs/remote/kernel_search/out/samples")] if remote else [])


def test_a_region_winner_is_a_patch_adding_its_override_module(tmp_path, monkeypatch):
    regions = engine(tmp_path, monkeypatch)
    monkeypatch.setattr(regions, "OVERRIDES_PACKAGE", "fake_engine.models.regions")
    tree = tmp_path / "tree"
    pkg = tree / "src" / "fake_engine" / "models" / "regions"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    h = held(tmp_path, regions)
    t = regions.from_engine("norm", "add", shared_weights=str(h.path))
    program = regions.start_program("norm", "add", shared_weights=str(h.path))
    names = {"stations": ["norm", "mm", "act", "add", "kv"]}
    diff = pipeline.patch(t, tree, program, names)
    assert diff.startswith("--- /dev/null\n+++ b/src/fake_engine/models/regions/norm_add.py")
    assert "+OVERRIDES = {('norm', 'add'): _override}" in diff and "__init__" not in diff
    (pkg / "act_kv.py").write_text("OVERRIDES = {('act', 'kv'): None}\n")
    with pytest.raises(ValueError, match="overlaps the override act..kv"):
        pipeline.patch(t, tree, program, names)


def test_kernel_search_runs_a_region_and_hands_back_its_override_module(tb, monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("LAB_GPU_USD_PER_HOUR", "3")
    monkeypatch.setattr(pipeline, "captures", lambda t, tree, profile, *a: [tmp_path])
    start = "# EVOLVE-BLOCK-START\nimport torch\n\n\ndef kernel(h):\n    return (h,)\n# EVOLVE-BLOCK-END\n"
    seen = {}

    @contextmanager
    def hold(task, caps, out, env, worker=None):
        seen["hold"] = task.name
        yield {"holder": "/gpu/w.pkl", "samples": "/gpu/s", "start": start, "stations": ["norm", "mm", "act", "add"],
               "shapes": {"seen": [[4, 64]], "heldout": [[64, 64]], "heldout_axes": [0]}}
    monkeypatch.setattr(pipeline, "hold", hold)

    def search(t, cs, s, out, evals, root, env, device="cuda", facts_text=None, held=None):
        seen["search"] = (s.read_text(), held["holder"])
        out.mkdir(parents=True)
        (out / "best_program.py").write_text(s.read_text().replace("(h,)", "(h + 0,)"))
        return {"best": {"combined_score": 1.2, "correct": 1.0}, "cost": {"usd": 1.0}, "ledger": "ev"}
    monkeypatch.setattr(pipeline, "search", search)
    gates = {"correctness_seen": {"passed": True}, "correctness_heldout": {"passed": True, "cases": 2}}
    monkeypatch.setattr(pipeline, "gate", lambda t, caps, best, out, env, held: seen.setdefault("gate", held) and {
        "metrics": {"correct": 1.0, "heldout_speedup": 1.1}, "artifacts": {"gates": json.dumps(gates)}})
    h = next(s for s in tb.specs() if s.name == "hypothesis").fn
    k = h({"hypothesis": "fuse norm..add", "rationale": "r", "layer": "kernels"}).split()[1]
    fn = next(s for s in tb.specs() if s.name == "kernel_search").fn
    out = json.loads(fn({"hypothesis_id": k, "expect": EXPECT, "task": "region:norm:add", "budget_usd": 20}
                        ).split("\n\nExperiment record")[0])
    assert seen["hold"] == "region:norm:add" and seen["search"] == (start, "/gpu/w.pkl") and seen["gate"]["samples"]
    patch = (tb.s.workspace.path / out["patch"]).read_text()
    assert "+++ b/src/inference_server/models/regions/norm_add.py" in patch and "OVERRIDES = {('norm', 'add')" in patch
