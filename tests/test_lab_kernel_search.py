"""The `kernel_search` tool (lab/tools.py) and its pipeline: captures and their reuse, the worker's ssh env, the tool's refusals, and what a search that dies is charged."""
from __future__ import annotations

import contextlib
import dataclasses
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")             # the kernel tasks import it
pytest_plugins = ["tests.test_lab_proftools"]   # its `tb` fixture: a Toolbox with fake instruments

from lab.kernels.tasks import pipeline  # noqa: E402
from lab.kernels.tasks import load  # noqa: E402
from lab.records import ledger  # noqa: E402

EXPECT = {"metric": "speedup", "value": 1.1, "unit": "absolute", "why": "fewer launches"}
MODULE = "import torch\n\n\ndef fn(x):\n    return x * 2\n"


def task():
    return dataclasses.replace(load("toy_rmsnorm"), entry="pkg.mod:fn")


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
    from evals.loadgen import suite
    assert set(pipeline.REGIME_CAPTURE) <= set(suite.REGIMES)


def spec(tb):
    return next(s for s in tb.specs() if s.name == "kernel_search")


def test_refused_off_the_kernels_layer_and_for_a_task_that_is_no_region(tb, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    h = next(s for s in tb.specs() if s.name == "hypothesis").fn
    rt = h({"hypothesis": "h", "rationale": "r", "layer": "runtime"}).split()[1]
    with pytest.raises(ValueError, match="kernels layer only"):
        spec(tb).fn({"hypothesis_id": rt, "expect": EXPECT, "task": "region:norm:add"})
    k = h({"hypothesis": "k", "rationale": "r", "layer": "kernels"}).split()[1]
    for bad in ("toy_rmsnorm", "region:norm", "kb:1:1"):
        with pytest.raises(ValueError, match="region:START:END"):
            spec(tb).fn({"hypothesis_id": k, "expect": EXPECT, "task": bad})


def test_captures_follow_the_scored_regimes_weigh_each_the_same_and_are_reused(tmp_path, monkeypatch):
    from lab.kernels.tasks import shapes
    assert pipeline.specs(("saturated", "bursty", "nope")) == (("mixed_1", 256), ("mixed_3", 64))
    assert pipeline.specs() == pipeline.CAPTURE_SPECS
    tree = tmp_path / "tree"
    (tree / "src" / "pkg").mkdir(parents=True)
    (tree / "src" / "pkg" / "mod.py").write_text(MODULE)
    (tree / "src" / "pkg" / "other.py").write_text("x = 1\n")
    made = []

    def profile(t, a):
        made.append(a["corpus_class"])
        assert (a["requests"], a["concurrency"], a["max_tokens"]) == (768, 256, pipeline.CAPTURE_TOKENS)   # steady
        b = tmp_path / f"bundle-{len(made)}"
        (b / "captures").mkdir(parents=True)
        return b
    cache = tmp_path / "captures.json"
    first = pipeline.captures(task(), tree, profile, ("saturated",), cache)
    assert pipeline.captures(task(), tree, profile, ("saturated",), cache) == first and made == ["mixed_1"]
    (tree / "src" / "pkg" / "other.py").write_text("x = 2\n")                       # upstream code changed
    assert pipeline.captures(task(), tree, profile, ("saturated",), cache) != first and len(made) == 2
    calls = {"a": [(8,)] * 9, "b": [(64,)] * 1}
    monkeypatch.setattr(shapes.capture, "load", lambda d: [{"fn": "pkg.mod:fn", "s": s} for s in calls[Path(d).name]])
    t = dataclasses.replace(task(), shape_of=lambda call: call["s"])
    assert shapes.from_captures(t, [Path("a"), Path("b")]) == {(8,): 9, (64,): 1}
    assert shapes.from_captures(t, [Path("a"), Path("b")], equal=True) == {(8,): 9, (64,): 9}
    k1 = pipeline.tree_key(task(), tree, ("t", "mixed_1", 256, sorted(pipeline.capture_env(task()).items())))
    t8 = dataclasses.replace(task(), capture_env=(("KV", "fp8"),))
    assert k1 != pipeline.tree_key(t8, tree, ("t", "mixed_1", 256, sorted(pipeline.capture_env(t8).items())))
    asked = []
    pipeline.captures(dataclasses.replace(task(), name="t8", capture_env=(("KV", "fp8"),)), tree,
                      lambda t, a: asked.append(a), ("saturated",))
    assert asked[0]["capture_task"] == "t8"                       # the worker applies the task's capture env
    assert asked[0]["capture_only"] is True                       # no profiler trace; the samples stay on the worker


def test_a_search_that_dies_still_charges_what_it_spent(tb, monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("LAB_GPU_USD_PER_HOUR", "3")
    monkeypatch.setattr(pipeline, "captures", lambda *a: [tmp_path])
    monkeypatch.setattr(pipeline, "hold", lambda *a: contextlib.nullcontext({"start": "x = 1\n"}))

    def dies(t, cs, start, out, *a, **k):
        out.mkdir(parents=True)
        (out / "calls.jsonl").write_text(json.dumps({"usd": 1.25}) + "\n" + json.dumps({"usd": 0.5}) + "\n")
        raise RuntimeError("lab.kernels.search exited 1")
    monkeypatch.setattr(pipeline, "search", dies)
    h = next(s for s in tb.specs() if s.name == "hypothesis").fn
    k = h({"hypothesis": "fuse", "rationale": "r", "layer": "kernels"}).split()[1]
    spent = tb.s.budget.spent_usd
    out = spec(tb).fn({"hypothesis_id": k, "expect": EXPECT, "task": "region:norm:add"})
    assert out.startswith("kernel_search failed") and tb.s.budget.spent_usd >= spent + 1.75
    assert list(ledger.records(tb.s.ledger_root, kind="kernels"))[-1]["result"]["model_usd"] == 1.75
