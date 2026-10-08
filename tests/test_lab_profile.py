"""lab.profile writes a complete bundle whose trace carries the engine's phase ranges."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("lab.profile", reason="needs the engine and its deps: run with the engine's python", exc_type=ImportError)
from inference_server.config import Settings  # noqa: E402
from lab import bundle, gpu, profile  # noqa: E402


@pytest.mark.asyncio
async def test_bundle_is_complete(tmp_path, stub_backend_cls):
    prompts = profile.synthetic_prompts(4, 8)
    settings = Settings(max_batch_size=4)
    out = await profile.run(stub_backend_cls(), settings, prompts, max_tokens=3, out=tmp_path, warmup=1)
    names = {p.name for p in out.iterdir()}
    assert set(bundle.FILES) <= names
    assert ("gpu.csv" in names) == gpu.available()

    meta = json.loads((out / "meta.json").read_text())
    assert meta["requests"] == 4 and meta["max_tokens"] == 3 and meta["failed"] == 0
    assert meta["workload_hash"] == bundle.workload_hash(prompts, 3)
    assert meta["device"] == "cpu" and meta["window"]["wall_s"] > 0 and meta["torch"]
    assert meta["settings"]["max_batch_size"] == 4

    trace = json.loads((out / "trace.json").read_text())
    names_in_trace = {e.get("name", "") for e in trace["traceEvents"]}
    assert any(n.startswith("decode step=") for n in names_in_trace)
    assert any(n.startswith("prefill step=") for n in names_in_trace)

    events = [json.loads(l) for l in (out / "events.jsonl").read_text().splitlines()]
    windows = [e["state"] for e in events if e["kind"] == "profile_window"]
    assert windows == ["begin", "end"]
    finished = [e for e in events if e["kind"] == "finish"]
    assert len(finished) == 5          # 1 warmup + 4 profiled
    stats = json.loads((out / "stats.json").read_text())
    assert stats["total_completed"] == 5 and stats["timeline"]["events_dropped"] == 0


def test_synthetic_prompts_are_deterministic():
    assert profile.synthetic_prompts(2, 5) == profile.synthetic_prompts(2, 5)
    assert profile.synthetic_prompts(2, 5) != profile.synthetic_prompts(2, 5, seed=1)


@pytest.mark.asyncio
async def test_cuda_range_window_writes_no_torch_trace(tmp_path, stub_backend_cls):
    out = await profile.run(stub_backend_cls(), Settings(max_batch_size=2), profile.synthetic_prompts(2, 4), max_tokens=2,
                            out=tmp_path, profiler="cuda-range")
    assert not (out / "trace.json").exists()
    assert json.loads((out / "meta.json").read_text())["profiler"] == "cuda-range"


FAKE_PYSPY = r'''#!/usr/bin/env python3
import json, sys
a = sys.argv[1:]
open(LOG, "a").write(json.dumps(a) + "\n")
if a[0] == "record":
    open(a[a.index("--output") + 1], "w").write("{}")
else:
    print("Thread 1 (active): MainThread")
'''


@pytest.mark.asyncio
async def test_pyspy_samples_this_process_under_a_repeated_workload(tmp_path, stub_backend_cls):
    import os
    exe = tmp_path / "py-spy"
    exe.write_text(FAKE_PYSPY.replace("LOG, ", f"{str(tmp_path / 'calls.log')!r}, "))
    exe.chmod(0o755)
    out = await profile.run(stub_backend_cls(), Settings(max_batch_size=2), profile.synthetic_prompts(2, 4), max_tokens=2,
                            out=tmp_path / "b", profiler="pyspy", pyspy=(str(exe), 0.5, 50))
    rec, dump = [json.loads(line) for line in (tmp_path / "calls.log").read_text().splitlines()]
    assert rec[:3] == ["record", "--pid", str(os.getpid())] and "--nonblocking" in rec and "speedscope" in rec
    assert dump == ["dump", "--pid", str(os.getpid()), "--nonblocking"]
    assert (out / "pyspy.speedscope.json").exists() and "MainThread" in (out / "pyspy-dump.txt").read_text()
    meta = json.loads((out / "meta.json").read_text())
    assert meta["profiler"] == "pyspy" and meta["rounds"] >= 1 and meta["failed"] == 0
