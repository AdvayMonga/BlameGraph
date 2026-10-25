"""The `kernel_search` tool (lab/tools.py) and the pipeline it runs (kernels/pipeline.py): the start program and the
patch from the agent's engine tree, the ssh environment for a worker host, the tool's refusals, and one call end to end
with the capture, search and gate faked (no GPU, no model)."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")             # the kernel tasks import it
pytest_plugins = ["tests.test_lab_proftools"]   # its `tb` fixture: a Toolbox with fake instruments

from kernels import pipeline  # noqa: E402
from kernels.tasks import load  # noqa: E402
from lab import ledger  # noqa: E402

EXPECT = {"metric": "speedup", "value": 1.1, "unit": "absolute", "why": "fewer launches"}
MODULE = "import torch\n\n\ndef fn(x):\n    return x * 2\n"


def task():
    return dataclasses.replace(load("silu_mul"), entry="pkg.mod:fn", start_edits=())


def test_the_start_program_and_the_patch_come_from_the_workspace_tree(tmp_path):
    (tmp_path / "src" / "pkg").mkdir(parents=True)
    (tmp_path / "src" / "pkg" / "mod.py").write_text(MODULE)
    start = pipeline.start_program(task(), tmp_path)
    assert "EVOLVE-BLOCK-START" in start and start.rstrip().endswith("kernel = fn")
    diff = pipeline.patch(task(), tmp_path, start.replace("x * 2", "x + x"))
    assert diff.startswith("--- a/src/pkg/mod.py") and "-    return x * 2" in diff and "+    return x + x" in diff
    assert pipeline.patch(task(), tmp_path, start) == ""


def test_a_worker_host_gets_the_evaluators_ssh_mode_quoted_as_the_worker_quotes():
    import os
    import subprocess
    env = pipeline.remote_env(["ssh", "lab@h"], "~", "$HOME/inf/.venv/bin/python", {"X": "$HOME/a"})
    assert json.loads(env["KERNEL_EVAL_SSH"]) == ["ssh", "lab@h"]
    line = env["KERNEL_EVAL_REMOTE"] + " sh -c 'printf \"%s|%s|%s\" \"$X\" \"$(pwd)\" \"$PATH\"'"
    x, pwd, path = subprocess.run(["sh", "-c", line], capture_output=True, text=True, check=True).stdout.split("|")
    home = os.path.realpath(os.environ["HOME"])
    assert x == os.environ["HOME"] + "/a" and os.path.realpath(pwd) == home      # ~ and $HOME expand there
    assert path.startswith(os.environ["HOME"] + "/inf/.venv/bin:")
    assert "sudo -n" in pipeline.remote_env(["ssh", "h"], "~", "/p/python", {}, sudo=True)["KERNEL_EVAL_REMOTE"]
    with pytest.raises(ValueError, match="unsafe"):
        pipeline.remote_env([], "~", "/p", {"X": 'a"b'})


def test_every_capture_regime_is_a_real_regime():
    from regimes import suite
    assert set(pipeline.REGIME_CAPTURE) <= set(suite.REGIMES)


def spec(tb):
    return next(s for s in tb.specs() if s.name == "kernel_search")


def test_refused_off_the_kernels_layer_and_beyond_the_budget(tb, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("LAB_GPU_USD_PER_HOUR", "3")
    h = next(s for s in tb.specs() if s.name == "hypothesis").fn
    rt = h({"hypothesis": "h", "rationale": "r", "layer": "runtime"}).split()[1]
    with pytest.raises(ValueError, match="kernels layer only"):
        spec(tb).fn({"hypothesis_id": rt, "expect": EXPECT, "task": "silu_mul", "budget_usd": 1})
    k = h({"hypothesis": "k", "rationale": "r", "layer": "kernels"}).split()[1]
    for bad, why in (({"task": "nope", "budget_usd": 1}, "task"), ({"task": "silu_mul", "budget_usd": 0}, "budget_usd"),
                     ({"task": "silu_mul", "budget_usd": 10_000}, "more than the run has left"),
                     ({"task": "silu_mul", "budget_usd": 0.5}, "a search needs at least")):
        with pytest.raises(ValueError, match=why):
            spec(tb).fn({"hypothesis_id": k, "expect": EXPECT, **bad})


def test_one_call_records_the_search_and_saves_a_patch_when_the_gate_passes(tb, monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("LAB_GPU_USD_PER_HOUR", "3")
    from kernels.tasks import load as real_load
    import kernels.tasks
    engine_task = dataclasses.replace(task(), entry="inference_server.engine:speed")     # the fixture repo's module
    monkeypatch.setattr(kernels.tasks, "load", lambda n: engine_task if n == "silu_mul" else real_load(n))
    caps = tmp_path / "captures"
    caps.mkdir()
    monkeypatch.setattr(pipeline, "captures", lambda t, tree, profile, *a: [caps])

    def search(t, cs, start, out, usd, evals, root, env, funnel=True, device="cuda"):
        out.mkdir(parents=True)
        (out / "best_program.py").write_text(start.read_text().replace("return 1", "return 2"))
        return {"best": {"combined_score": 1.3, "correct": 1.0}, "cost": {"usd": 2.5}, "ledger": "ev-search"}
    monkeypatch.setattr(pipeline, "search", search)
    gates = {"correctness_seen": {"passed": True}, "correctness_heldout": {"passed": True, "cases": 4}}
    monkeypatch.setattr(pipeline, "gate", lambda *a: {"metrics": {"correct": 1.0, "heldout_speedup": 1.2},
                                                        "artifacts": {"gates": json.dumps(gates)}})
    h = next(s for s in tb.specs() if s.name == "hypothesis").fn
    k = h({"hypothesis": "fuse", "rationale": "r", "layer": "kernels"}).split()[1]
    spent = tb.s.budget.spent_usd
    out = json.loads(spec(tb).fn({"hypothesis_id": k, "expect": EXPECT, "task": "silu_mul", "budget_usd": 20}
                                 ).split("\n\nExperiment record")[0])
    assert out["gate"]["passed"] and out["metrics"] == {"speedup": 1.3, "heldout_speedup": 1.2}
    patch = (tb.s.workspace.path / out["patch"]).read_text()
    assert "+    return 2" in patch and tb.s.budget.spent_usd >= spent + 2.5
    rec = list(ledger.records(tb.s.ledger_root, kind="kernels"))[-1]
    assert rec["tool"] == "kernel_search" and rec["experiment"]["actual"] == 1.3 and rec["result"]["search_record"]


def test_captures_follow_the_scored_regimes_weigh_each_the_same_and_are_reused(tmp_path, monkeypatch):
    from kernels import shapes
    assert pipeline.specs(("saturated", "bursty", "nope")) == (("mixed_1", 256), ("mixed_3", 64))
    assert pipeline.specs() == pipeline.CAPTURE_SPECS
    tree = tmp_path / "tree"
    (tree / "src" / "pkg").mkdir(parents=True)
    (tree / "src" / "pkg" / "mod.py").write_text(MODULE)
    (tree / "src" / "pkg" / "other.py").write_text("x = 1\n")
    made = []

    def profile(t, a):
        made.append(a["corpus_class"])
        b = tmp_path / f"bundle-{len(made)}"
        (b / "captures").mkdir(parents=True)
        return b
    cache = tmp_path / "captures.json"
    first = pipeline.captures(task(), tree, profile, ("saturated",), cache)
    (tree / "src" / "pkg" / "mod.py").write_text(MODULE + "# the kernel itself changed\n")
    assert pipeline.captures(task(), tree, profile, ("saturated",), cache) == first and made == ["mixed_1"]
    (tree / "src" / "pkg" / "other.py").write_text("x = 2\n")                       # upstream code changed
    assert pipeline.captures(task(), tree, profile, ("saturated",), cache) != first and len(made) == 2
    calls = {"a": [(8,)] * 9, "b": [(64,)] * 1}
    monkeypatch.setattr(shapes.capture, "load", lambda d: [{"fn": "pkg.mod:fn", "s": s} for s in calls[Path(d).name]])
    t = dataclasses.replace(task(), shape_of=lambda call: call["s"])
    assert shapes.from_captures(t, [Path("a"), Path("b")]) == {(8,): 9, (64,): 1}
    assert shapes.from_captures(t, [Path("a"), Path("b")], equal=True) == {(8,): 9, (64,): 9}
    asked = []
    pipeline.captures(dataclasses.replace(task(), name="t8", capture_env=(("KV", "fp8"),)), tree,
                      lambda t, a: asked.append(a), ("saturated",))
    assert asked[0]["capture_task"] == "t8"                       # the worker applies the task's capture env


def test_a_search_that_dies_still_charges_what_it_spent(tb, monkeypatch, tmp_path):
    from kernels.tasks import load as real_load
    import kernels.tasks
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("LAB_GPU_USD_PER_HOUR", "3")
    engine_task = dataclasses.replace(task(), entry="inference_server.engine:speed")
    monkeypatch.setattr(kernels.tasks, "load", lambda n: engine_task if n == "silu_mul" else real_load(n))
    monkeypatch.setattr(pipeline, "captures", lambda *a: [tmp_path])

    def dies(t, cs, start, out, *a, **k):
        out.mkdir(parents=True)
        (out / "trace.jsonl").write_text(json.dumps({"cost_usd": 1.25}) + "\n" + json.dumps({"cost_usd": 0.5}) + "\n")
        raise RuntimeError("kernels.search exited 1")
    monkeypatch.setattr(pipeline, "search", dies)
    h = next(s for s in tb.specs() if s.name == "hypothesis").fn
    k = h({"hypothesis": "fuse", "rationale": "r", "layer": "kernels"}).split()[1]
    spent = tb.s.budget.spent_usd
    out = spec(tb).fn({"hypothesis_id": k, "expect": EXPECT, "task": "silu_mul", "budget_usd": 20})
    assert out.startswith("kernel_search failed") and tb.s.budget.spent_usd >= spent + 1.75
    assert list(ledger.records(tb.s.ledger_root, kind="kernels"))[-1]["result"]["model_usd"] == 1.75
