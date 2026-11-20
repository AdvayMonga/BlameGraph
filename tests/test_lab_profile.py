"""lab.profile.profile serves an engine whose every Python process joins the window, opens it from outside, and writes
the bundle."""

from __future__ import annotations

import dataclasses
import json
import os
import stat
import sys
from pathlib import Path

import pytest

from evals.targets import target
from lab.profile import bundle, profile

ROOT = Path(__file__).resolve().parents[1]
FAKE = ROOT / "tests" / "fake_engine.py"
PYSPY = '''#!/usr/bin/env python3
import sys
a = sys.argv[1:]
if a[0] == "record":
    open(a[a.index("--output") + 1], "w").write("{}")
else:
    print("Thread 1 (active): MainThread")
'''
SPAWN = '''import multiprocessing as mp, os, time
def model():                                    # the engine's model process: the only one with torch
    import torch
    torch.ones(4).sum()
    open(os.environ["CHILD_PID"], "w").write(str(os.getpid()))
    time.sleep(3600)
'''


@pytest.fixture
def served(tmp_path, monkeypatch):
    """The fake engine as the target's engine; `launch(top, main)` adds code at its top and before it serves."""
    t = target.load()

    def launch(top: str = "", main: str = "") -> None:
        script = tmp_path / "engine.py"
        script.write_text(f"import runpy, sys\n{top}\nif __name__ == '__main__':\n    {main or 'pass'}\n"
                          f"    runpy.run_path({str(FAKE)!r}, run_name='__main__')\n")
        eng = dataclasses.replace(t.engine, launch=f"python {script} --port {{port}} --token-s 0", env={},
                                  stats="/scheduler/stats", telemetry_env={}, startup_timeout_s=30)
        monkeypatch.setattr(target, "load", lambda *a: dataclasses.replace(t, engine=eng))
    monkeypatch.setattr(profile.engine, "python", lambda: sys.executable)
    launch()
    return launch


def chats(n, max_tokens=4):
    return [profile.Request(f"r{i}", [{"role": "user", "content": f"hi {i}"}], max_tokens, ignore_eos=True)
            for i in range(n)]


def run(tmp_path, window, **kw):
    return profile.run(ROOT, window, chats(2), tmp_path / "bundles", source={"kind": "test"}, jailed=False,
                       harness=ROOT, **kw)


def fake_pyspy(tmp_path) -> str:
    exe = tmp_path / "py-spy"
    exe.write_text(PYSPY)
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    return str(exe)


def test_a_window_runs_the_requests_and_the_bundle_holds_its_counters_log_and_provenance(tmp_path, served):
    out = run(tmp_path, chats(3), profiler="pyspy", pyspy=(fake_pyspy(tmp_path), 1, 100, []))
    meta = json.loads((out / "meta.json").read_text())
    assert meta["requests"] == 3 and meta["max_tokens"] == 4 and meta["failed"] == 0 and meta["rounds"] >= 1
    assert meta["outcomes"]["tokens_out"]["min"] == 4 and meta["warmup"]["requests"] == 2
    [proc] = meta["profiler"]["processes"]                                       # the launch's process, no torch
    assert meta["profiler"]["kind"] == "pyspy" and proc["root"] and not proc["gpu"] and meta["window"]["wall_s"] >= 1
    assert meta["workload"] == {"kind": "test"} and meta["engine"]["ready_s"] > 0
    stats = json.loads((out / "stats.json").read_text())
    assert stats["window_start"]["completed"] == 2 and stats["window_delta"]["completed"] >= 3     # warmup before it
    assert "MainThread" in (out / "pyspy-dump.txt").read_text() and (out / "pyspy.speedscope.json").exists()
    assert (out / "serve.log").exists() and not {"start", "stop", "end"} & {p.name for p in out.iterdir()}


def test_a_window_that_fails_in_the_engine_is_raised(tmp_path, served):
    with pytest.raises(RuntimeError, match="window failed in engine process"):
        run(tmp_path, chats(1), profiler="pyspy", pyspy=(str(tmp_path / "no-py-spy"), 1, 100, []))
    assert "Traceback" in next((tmp_path / "bundles").glob("*/procs/*/error.txt")).read_text()


def test_the_model_process_the_engine_spawns_is_profiled_and_ends_with_the_engine(tmp_path, served, monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setenv("CHILD_PID", str(tmp_path / "child.pid"))
    served(SPAWN, "mp.get_context('spawn').Process(target=model, daemon=True).start()")
    out = run(tmp_path, chats(2))
    child = int((tmp_path / "child.pid").read_text())
    [proc] = json.loads((out / "meta.json").read_text())["profiler"]["processes"]
    assert proc["pid"] == child and proc["gpu"] and not proc["root"] and proc["profiler"]["activities"][0] == "CPU"
    own = out / "procs" / str(child)
    assert {"trace.json", "ops.json", "memory.json", "window.json"} <= {p.name for p in own.iterdir()}
    assert "top" in json.loads((out / "ops.json").read_text())                 # merged at the bundle's root
    with pytest.raises(ProcessLookupError):
        os.kill(child, 0)                                                      # stopped with the engine's group


def test_captures_record_the_windows_calls(tmp_path, served):
    pytest.importorskip("torch")
    from lab.profile import capture
    served("import torch")
    out = run(tmp_path, chats(2), profiler="pyspy", pyspy=(fake_pyspy(tmp_path), 1, 100, []), capture=["time:sleep"])
    calls = capture.load(out / "captures")
    assert calls and {c["fn"] for c in calls} == {"time:sleep"}


def test_a_region_capture_is_the_engines_own_from_launch_on(tmp_path, served):
    pytest.importorskip("torch")
    served("import os; print('REGION', os.environ.get('CAPTURE_REGION'), os.environ.get('CAPTURE_REGION_DIR'), flush=True)")
    out = run(tmp_path, chats(1), profiler="pyspy", pyspy=(fake_pyspy(tmp_path), 1, 100, []),
              capture=["region:attn:moe"])
    assert f"REGION attn:moe {out / 'captures'}" in (out / "serve.log").read_text()


def test_processes_kernel_tables_merge_by_name():
    a = {"kernels": [{"name": "gemm", "count": 2, "total_us": 10.0, "mean_us": 5.0, "max_us": 6.0}]}
    b = {"kernels": [{"name": "gemm", "count": 1, "total_us": 8.0, "mean_us": 8.0, "max_us": 8.0},
                     {"name": "norm", "count": 4, "total_us": 4.0, "mean_us": 1.0, "max_us": 1.5}]}
    m = bundle.merge_kernels([a, b])
    assert m["kernels"][0] == {"name": "gemm", "count": 3, "total_us": 18.0, "max_us": 8.0, "mean_us": 6.0}
    assert m["total_us"] == 22.0 and m["launches"] == 7 and bundle.merge_kernels([]) is None


def test_synthetic_requests_are_token_ids_and_the_warmup_is_the_next_seeds():
    window, warm, source = profile.workload(3, 8, seed=4, prompt_len=5)
    assert [len(r.prompt_ids) for r in window] == [5] * 3 and len(warm) == 3 and window[0].ignore_eos
    assert window[0].prompt_ids != warm[0].prompt_ids and source == {"kind": "synthetic", "prompt_len": 5, "seed": 4}
    assert profile.workload(3, 8, seed=4, prompt_len=5)[0][0].prompt_ids == window[0].prompt_ids


def test_the_counter_delta_is_every_number_both_ends_hold():
    start = {"a": 1, "b": "x", "c": {"d": 2, "e": True}, "f": 3}
    end = {"a": 4, "b": "y", "c": {"d": 7, "e": False}, "g": 1}
    assert profile.delta(start, end) == {"a": 3, "c": {"d": 5}}
