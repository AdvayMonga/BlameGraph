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
    assert meta["outcomes"]["full"] == 4 and meta["outcomes"]["tokens_out"]["total"] == 12
    assert meta["warmup"]["requests"] == 1 and meta["profiler_on"] is True
    assert meta["profiler"]["activities"] == ["CPU"] and "all_threads" in meta["profiler"]
    assert set(meta["gpu"]) == {"window_start", "window_end"}
    assert meta["timeline"]["events_dropped"] == 0 and meta["timeline"]["events_written"] > 0
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
    assert stats["window_start"]["total_completed"] == 1 and stats["window_end"]["total_completed"] == 5
    assert stats["window_delta"]["total_completed"] == 4 and stats["window_delta"]["total_admitted"] == 4

    ops = json.loads((out / "ops.json").read_text())
    assert ops["top"] and {"name", "count", "self_cpu_us", "self_device_us"} <= set(ops["top"][0])
    assert not (out / "kernels.json").exists()          # no CUDA activity on CPU
    mem = json.loads((out / "memory.json").read_text())
    assert mem["window_peak"] is None and mem["host_rss_peak_lifetime_bytes"] > 0


@pytest.mark.asyncio
async def test_warmup_defaults_to_the_window_and_peaks_reset_after_it(tmp_path, stub_backend_cls, monkeypatch):
    calls = []
    peaks = iter([{"allocated_bytes": 900, "reserved_bytes": 1000}, {"allocated_bytes": 500, "reserved_bytes": 1200}])
    monkeypatch.setattr(bundle, "device_peaks", lambda d: calls.append("peak") or next(peaks))
    monkeypatch.setattr(bundle, "reset_peaks", lambda d: calls.append("reset"))
    prompts = profile.synthetic_prompts(3, 8)
    out = await profile.run(stub_backend_cls(), Settings(max_batch_size=4), prompts, max_tokens=2, out=tmp_path)
    assert calls == ["peak", "reset", "peak"]
    mem = json.loads((out / "memory.json").read_text())
    assert mem["window_peak"] == {"allocated_bytes": 500, "reserved_bytes": 1200}
    assert mem["lifetime_peak"] == {"allocated_bytes": 900, "reserved_bytes": 1200}
    meta = json.loads((out / "meta.json").read_text())
    assert meta["warmup"]["requests"] == 3
    events = [json.loads(l) for l in (out / "events.jsonl").read_text().splitlines()]
    begin = next(i for i, e in enumerate(events) if e["kind"] == "profile_window")
    assert sum(e["kind"] == "finish" for e in events[:begin]) == 3


def test_warmup_prompts_match_lengths_not_contents():
    prompts = [[5] * 3, [7] * 9]
    w = profile.warmup_prompts(prompts, 3)
    assert [len(p) for p in w] == [3, 9, 3] and not any(p in prompts for p in w)


def test_outcomes_count_empty_as_failed_and_short_separately():
    o = profile.outcomes([[1, 2, 3], [], [1], RuntimeError("x")], max_tokens=3)
    assert (o["errors"], o["empty"], o["short"], o["full"]) == (1, 1, 1, 1)
    assert o["tokens_out"] == {"total": 4, "min": 0, "mean": 4 / 3, "max": 3}


def test_stats_delta_counts_only_the_window():
    s0 = {"total_completed": 2, "decode_steps": 10, "active_size": 1, "wave_sizes": {1: 2}}
    s1 = {"total_completed": 6, "decode_steps": 25, "active_size": 0, "wave_sizes": {1: 2, 4: 1}}
    assert profile.stats_delta(s0, s1) == {"total_completed": 4, "decode_steps": 15, "wave_sizes": {4: 1}}


def test_kernels_table_aggregates_cuda_events_by_name():
    from types import SimpleNamespace as NS
    from torch.autograd import DeviceType

    def ev(name, us, dev=DeviceType.CUDA):
        return NS(name=name, device_type=dev, time_range=NS(elapsed_us=lambda: us))
    assert bundle.kernels_table([ev("aten::mm", 50, DeviceType.CPU)]) is None
    k = bundle.kernels_table([ev("gemm", 10), ev("gemm", 30), ev("softmax", 5), ev("aten::mm", 50, DeviceType.CPU)])
    assert k["launches"] == 3 and k["total_us"] == 45
    assert k["kernels"][0] == {"name": "gemm", "count": 2, "total_us": 40, "mean_us": 20, "max_us": 30}


def test_main_profiles_a_presampled_corpus_workload(tmp_path, stub_backend_cls, monkeypatch):
    import inference_server.tokenizer as tokmod

    class Tok:
        def __init__(self, *a): pass
        def encode_messages(self, msgs, thinking=True):
            return [len(m["content"]) for m in msgs] + [int(thinking)]
    monkeypatch.setattr(tokmod, "Tokenizer", Tok)
    monkeypatch.setattr(profile, "build_backend", lambda s: (stub_backend_cls(), None))
    w = {"source": {"kind": "corpus", "class": "steady_interactive", "split": "seen", "seed": 0,
                    "corpus_version": "v", "lines": [0, 1]},
         "messages": [[{"role": "user", "content": "hi"}], [{"role": "user", "content": "hello"}]]}
    (tmp_path / "w.json").write_text(json.dumps(w))
    assert profile.main(["--workload", str(tmp_path / "w.json"), "--max-tokens", "2", "--out", str(tmp_path / "o")]) == 0
    meta = json.loads(next((tmp_path / "o").glob("*/meta.json")).read_text())
    assert meta["workload"] == {**w["source"], "thinking": False} and meta["requests"] == 2
    assert meta["workload_hash"] == bundle.workload_hash([[2, 0], [5, 0]], 2)


def test_main_needs_an_explicit_workload_source():
    with pytest.raises(SystemExit):
        profile.main([])


def test_synthetic_prompts_are_deterministic():
    assert profile.synthetic_prompts(2, 5) == profile.synthetic_prompts(2, 5)
    assert profile.synthetic_prompts(2, 5) != profile.synthetic_prompts(2, 5, seed=1)


@pytest.mark.asyncio
async def test_cuda_range_window_writes_no_torch_trace(tmp_path, stub_backend_cls):
    out = await profile.run(stub_backend_cls(), Settings(max_batch_size=2), profile.synthetic_prompts(2, 4), max_tokens=2,
                            out=tmp_path, profiler="cuda-range")
    assert not (out / "trace.json").exists()
    assert json.loads((out / "meta.json").read_text())["profiler"]["kind"] == "cuda-range"


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
    assert meta["profiler"]["kind"] == "pyspy" and meta["rounds"] >= 1 and meta["failed"] == 0
