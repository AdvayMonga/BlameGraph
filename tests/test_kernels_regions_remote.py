"""Regions through the kernel layer's pipeline (generate/kernels/tasks/pipeline.py, generate/kernels/tasks/hold.py) with the controller unable to
load them: captures packed into one samples dir, the hold process's protocol (local and over a stand-in for ssh), its
description of a region on the fake stations, the winner as an override-module patch of the agent's tree, and the
`kernel_search` tool on a region end to end: once with the capture, hold, search and gate faked, and once for real on
the CPU with a fake engine held by `generate.kernels.tasks.hold` and a fake model answering `python -m generate.kernels.search` (no GPU)."""
from __future__ import annotations

import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

if __name__ == "__main__":          # CI's referee job runs each test file as a script: these need pytest's fixtures
    import pytest
    raise SystemExit(pytest.main(["-q", __file__]))

import pytest  # noqa: E402

torch = pytest.importorskip("torch")
pytest_plugins = ["tests.test_lab_proftools"]   # its `tb` fixture: a Toolbox with fake instruments

from generate.kernels.tasks import pipeline  # noqa: E402
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
    return [captures(a, [(4, 64, 0, True)] * 3 + [(32, 64, 0, True), (64, 64, 0, True)]), captures(b, [(16, 64, 0, True)])]


def test_captures_pack_into_one_samples_dir_each_traffic_counted_apart(tmp_path, monkeypatch):
    from generate.kernels.tasks import shapes
    regions = engine(tmp_path, monkeypatch)
    caps = two_traffics(tmp_path)
    pack = pipeline.pack(caps, tmp_path / "pack")
    t = regions.from_engine("norm", "add", str(pack), str(held(tmp_path, regions).path), dtypes=(torch.float32,))
    assert set(t.seen_default) == {(4, 64), (16, 64), (32, 64), (64, 64)}      # every traffic's samples, renamed
    traffic = sorted(pack.glob("traffic*/*/captures"))
    assert len(traffic) == 2 and shapes.from_captures(t, traffic, equal=True) == {(4, 64): 3, (32, 64): 1, (64, 64): 1,
                                                                                  (16, 64): 5}
    x = t.make_inputs((16, 64), torch.float32, "cpu", torch.Generator().manual_seed(0))[0]
    assert torch.equal(x, torch.load(caps[1] / "samples" / "0.pt", weights_only=True)["reads"]["h"])


def test_the_hold_side_describes_a_region_from_its_samples_and_holder(tmp_path, monkeypatch):
    from generate.kernels.tasks import hold
    regions = engine(tmp_path, monkeypatch)
    pack = pipeline.pack(two_traffics(tmp_path), tmp_path / "pack")
    d = hold.describe("region:norm:add", pack, held(tmp_path, regions).path)
    assert d["shapes"]["heldout"] == [[32, 64]] and d["shapes"]["heldout_axes"] == [0]
    assert sorted(d["shapes"]["seen"]) == [[4, 64], [16, 64], [64, 64]] and d["stations"] == ["norm", "mm", "act", "add"]
    assert d["shapes"]["stations"] == d["stations"]                  # in the shapes file the controller's search reads
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

    class Worker:                                # measure.runner.worker.Remote's push and root, over the local disk
        def root(self):
            return str(tmp_path / "worker")

        def push(self, local, there):
            import shutil
            pushed.append(there)
            shutil.copytree(local, there)
    env = {"KERNEL_EVAL_SSH": json.dumps(["sh", "-c"]),
           "KERNEL_EVAL_REMOTE": f'cd ~ && PATH="{Path(sys.executable).parent}:$PATH" PYTHONPATH="{tmp_path}"'} \
        if remote else {}
    from generate.kernels.tasks import load
    task = load("region:norm:add")
    with pipeline.hold(task, caps, out, env, Worker() if remote else None) as d:
        assert d["task"] == "region:norm:add" and Path(d["samples"]).is_dir()
        assert (Path(d["samples"]) / "calls.jsonl").exists() and d["holder"].endswith("/holder/weights.pkl")
        assert not (Path(d["samples"]).parent / "released").exists()
    assert (Path(d["samples"]).parent / "released").exists()                    # stdin closed: it let go
    assert pushed == ([str(tmp_path / "worker" / "local/runs/remote/kernel_search/out/samples")] if remote else [])


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

    def search(t, cs, s, out, root, env, device="cuda", facts_text=None, held=None):
        seen["search"] = (s.read_text(), held["holder"])
        out.mkdir(parents=True)
        (out / "best_program.py").write_text(s.read_text().replace("(h,)", "(h + 0,)"))
        return {"best": {"combined_score": 1.2, "correct": 1.0}, "cost": {"usd": 1.0}, "ledger": "ev"}
    monkeypatch.setattr(pipeline, "search", search)
    gates = {"correctness_seen": {"passed": True}, "correctness_heldout": {"passed": True, "cases": 2}}
    monkeypatch.setattr(pipeline, "gate", lambda t, caps, best, out, env, held, device: seen.setdefault("gate", held) and {
        "metrics": {"correct": 1.0, "heldout_speedup": 1.1}, "artifacts": {"gates": json.dumps(gates)}})
    h = next(s for s in tb.specs() if s.name == "hypothesis").fn
    k = h({"hypothesis": "fuse norm..add", "rationale": "r", "layer": "kernels"}).split()[1]
    fn = next(s for s in tb.specs() if s.name == "kernel_search").fn
    out = json.loads(fn({"hypothesis_id": k, "expect": EXPECT, "task": "region:norm:add"}
                        ).split("\n\nExperiment record")[0])
    assert seen["hold"] == "region:norm:add" and seen["search"] == (start, "/gpu/w.pkl") and seen["gate"]["samples"]
    patch = (tb.s.workspace.path / out["patch"]).read_text()
    assert "+++ b/src/inference_server/models/regions/norm_add.py" in patch and "OVERRIDES = {('norm', 'add')" in patch


FAKE_ENGINE = {                      # the engine as generate.kernels.tasks.hold loads it: settings, a backend with a model, stations
    "inference_server/__init__.py": "",
    "inference_server/config.py": "def load_settings():\n    return None\n",
    "inference_server/server.py": "import torch\n\nclass _M:\n    w = torch.randn(64, 64, generator=torch.Generator()"
                                  ".manual_seed(0)) / 8\n\nclass _B:\n    model = _M()\n\ndef build_backend(settings):"
                                  "\n    return _B()\n",
    "inference_server/models/__init__.py": "",
    "inference_server/models/regions/__init__.py": "",
}
FAKE_WEIGHTS = """

def layer_weights(model, i):
    return {"w.mm": model.w, "w.mm_scale": None}

def layer_config(model):
    return LayerConfig(1e-6)
"""
# python -m generate.kernels.search runs in its own process: its model is faked where every process starts (sitecustomize).
FAKE_MODEL = """
import json, sys
sys.path.insert(0, {root!r})
try:
    from generate.kernels.search import llm
except Exception:
    llm = None

class Claude:
    def __init__(self, *a, **k):
        pass

    def __call__(self, system, user):
        if "making changes" not in system:                               # the gaming detector
            return json.dumps({{"gaming": False, "evidence": ""}}), 0.01
        head = json.dumps({{"bottleneck": "b", "optimisation_method": "m", "modification_plan": "p", "mechanism": []}})
        if "_K = 1" in user:                                             # the one change exists: nothing new to write
            return head, 0.01
        return head + "\\n<<<<<<< SEARCH\\n_CONST = {{}}\\n=======\\n_CONST = {{}}\\n_K = 1\\n>>>>>>> REPLACE", 0.01

if llm is not None:
    llm.Claude = Claude
"""


def test_kernel_search_runs_a_region_through_hold_search_and_gate_on_the_cpu(tb, monkeypatch, tmp_path):
    """The real pipeline with only the capture and the model faked: `generate.kernels.tasks.hold` loads a fake engine and shares its
    weights, `python -m generate.kernels.search` evolves the region's start program (every call answering one change) with the
    jailed evaluator on the CPU, the held-out gate judges the winner there too, and the override module comes back."""
    from tests.test_kernels_region_engine import FAKE as STATIONS
    from data.knowledge import ledger
    fake = tmp_path / "fake"
    for rel, text in {**FAKE_ENGINE, "inference_server/models/stations.py": STATIONS + FAKE_WEIGHTS,
                      "sitecustomize.py": FAKE_MODEL.format(root=str(pipeline.ROOT))}.items():
        (fake / rel).parent.mkdir(parents=True, exist_ok=True)
        (fake / rel).write_text(text)
    for k in [m for m in sys.modules if m.split(".")[0] == "inference_server"]:
        monkeypatch.delitem(sys.modules, k)                              # the real engine, if a test imported it
    monkeypatch.syspath_prepend(str(fake))
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, [str(fake), os.environ.get("PYTHONPATH")])))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("LAB_GPU_USD_PER_HOUR", "3")
    for k in ("KERNEL_EVAL_CONFIG", "KERNELS_EVAL_LOG", "KERNEL_EVAL_SSH"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(pipeline, "captures", lambda t, tree, profile, *a: two_traffics(tmp_path))
    h = next(s for s in tb.specs() if s.name == "hypothesis").fn
    k = h({"hypothesis": "fuse norm..add", "rationale": "r", "layer": "kernels"}).split()[1]
    fn = next(s for s in tb.specs() if s.name == "kernel_search").fn
    text = fn({"hypothesis_id": k, "expect": EXPECT, "task": "region:norm:add"})
    out = json.loads(text.split("\n\nExperiment record")[0])
    assert out["gate"]["passed"] is True, out
    gates = out["gate"]["gates"]
    assert gates["correctness_seen"]["passed"] and gates["correctness_heldout"]["passed"], gates
    assert gates["correctness_heldout"]["cases"] > 0                     # the held-out shape (64 rows) was judged
    assert out["metrics"]["speedup"] > 0 and out["metrics"]["heldout_speedup"] > 0
    patch = (tb.s.workspace.path / out["patch"]).read_text()
    assert "+++ b/src/inference_server/models/regions/norm_add.py" in patch and "OVERRIDES = {('norm', 'add')" in patch
    rec = next(r for r in ledger.records(tb.s.ledger_root, kind="kernels") if "candidates" in r["result"])   # the search's
    assert rec["config"]["task"] == "region:norm:add"
    assert any(c["parent"] and c["metrics"].get("correct") == 1.0 for c in rec["result"]["candidates"])  # the change, judged
    run = next(d for d in (tb.s.run_dir / "kernel_search").iterdir() if d.is_dir() and "region" in d.name)
    assert (run / "holder" / "weights.pkl").exists() and (run / "samples" / "calls.jsonl").exists()
    assert out["model_usd"] > 0
