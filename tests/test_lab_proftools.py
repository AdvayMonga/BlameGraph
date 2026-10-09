"""trace, kernel and hostprof against fake nsys, ncu and py-spy on PATH; and every tool's cost accounting."""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess

import pytest

from lab import ledger, proftools
from lab import tools as labtools
from lab.budget import Budget, BudgetExceeded
from lab.session import Session
from lab.tools import GPU_TOOLS, Toolbox
from lab.workspace import Workspace
from tests.lab_fixtures import make_repo

FAKE = r'''#!/usr/bin/env python3
"""Fake Nsight / py-spy: every call is appended to LOG (the jail wipes the env); reports are dummies where asked."""
import json, os, sys
name, a = os.path.basename(sys.argv[0]), sys.argv[1:]
open(LOG, "a").write(json.dumps([name] + a) + "\n")
val = lambda flag: a[a.index(flag) + 1]
if name == "nsys" and a[0] == "profile":
    open(val("--output") + ".nsys-rep", "w").write("rep")
elif name == "nsys" and a[0] == "stats":
    open(val("--output") + "_cuda_gpu_kern_sum.csv", "w").write("Time (%),Total Time (ns),Instances,Name\n91.0,1000,4,gemm_kernel\n")
elif name == "ncu" and "--import" in a:
    print('"ID","Kernel Name","gpu__time_duration.sum"\n"0","gemm_kernel","1234"')
elif name == "ncu":
    open(val("--export") + ".ncu-rep", "w").write("rep")
'''
INSTRUMENTS = ("nsys", "ncu", "py-spy")


@pytest.fixture
def tb(tmp_path, monkeypatch):
    monkeypatch.setenv("LAB_NO_JAIL", "1")
    monkeypatch.setenv("LAB_GPU_USD_PER_HOUR", "3600")          # $1 per second: the charge equals the seconds
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in INSTRUMENTS:
        (bin_dir / name).write_text(FAKE.replace("LOG, ", f"{str(tmp_path / 'calls.log')!r}, "))
        (bin_dir / name).chmod(0o755 | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    repo = make_repo(tmp_path)
    ws = Workspace(repo, "HEAD", tmp_path / "ws", tmp_path / "ledger"); ws.create()
    run_dir = tmp_path / "run"; run_dir.mkdir()
    s = Session("r1", "r1-s1", run_dir, ws, Budget(100.0), tmp_path / "ledger")
    return Toolbox(s)


def calls(tb):
    log = tb.s.run_dir.parent / "calls.log"
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


def last(tb, **match):
    return list(ledger.records(tb.s.ledger_root, **match))[-1]


def after(argv, flag):
    return argv[argv.index(flag) + 1]


def assert_charged(tb, rec):
    c = rec["cost"]
    assert c["rate_known"] is True and c["usd_per_hour"] == 3600.0
    assert c["usd"] == pytest.approx(c["seconds"]) and c["usd"] > 0 and c["gpu_seconds"] == c["seconds"]
    assert tb.s.budget.spent_usd == pytest.approx(sum(r["cost"]["usd"] for r in ledger.records(tb.s.ledger_root)))


def test_trace_wraps_the_workload_in_nsys_then_exports_stats(tb):
    out = tb.trace({"requests": 2})
    assert "gemm_kernel" in out and out.startswith("trace output at lab/runs/")
    prof, stats = calls(tb)
    assert prof[:2] == ["nsys", "profile"] and after(prof, "--capture-range") == "cudaProfilerApi"
    assert after(prof, "--cuda-graph-trace") == "node"
    assert after(prof, "-m") == "lab.profile" and after(prof, "--profiler") == "cuda-range"
    assert after(prof, "--requests") == "2" and after(prof, "--max-tokens") == "32"
    assert stats[:2] == ["nsys", "stats"] and after(stats, "--report") == proftools.NSYS_REPORTS
    assert stats[-1].endswith("_instr/trace.nsys-rep")
    rec = last(tb, kind="profile")
    assert rec["tool"] == "trace" and rec["result"]["returncode"] == 0
    blob = tb.s.ledger_root / rec["result"]["bundle"]
    assert (blob / "trace.nsys-rep").exists() and (blob / "stats_cuda_gpu_kern_sum.csv").exists()
    assert (tb.s.workspace.path / rec["result"]["workspace_copy"] / "trace.nsys-rep").exists()
    assert_charged(tb, rec)


def test_kernel_targets_the_regex_with_limits_and_keeps_raw_metrics(tb):
    out = tb.kernel({"kernel_regex": "gemm|fused_moe.*", "launch_count": 4, "launch_skip": 10, "set": "full"})
    assert "gpu__time_duration.sum" in out
    prof, imp = calls(tb)
    assert prof[0] == "ncu" and after(prof, "--kernel-name") == "regex:gemm|fused_moe.*"
    assert after(prof, "--launch-count") == "4" and after(prof, "--launch-skip") == "10" and after(prof, "--set") == "full"
    assert after(prof, "--profile-from-start") == "off" and after(prof, "--profiler") == "cuda-range"
    assert imp[0] == "ncu" and after(imp, "--import").endswith("kernel.ncu-rep") and after(imp, "--page") == "raw"
    rec = last(tb, kind="profile")
    blob = tb.s.ledger_root / rec["result"]["bundle"]
    assert (blob / "kernel.ncu-rep").exists() and "gemm_kernel" in (blob / "metrics.csv").read_text()
    assert_charged(tb, rec)


def test_hostprof_runs_the_workload_with_py_spy_readable_in_the_jail(tb, monkeypatch):
    seen = {}

    def runner(tree, argv, read=()):            # stands in for lab.profile, which needs the engine
        seen.update(argv=argv, read=read)
        b = tree / proftools.OUT / "bundle" / "b1"
        b.mkdir(parents=True)
        (b / "pyspy-dump.txt").write_text("Thread 1 (active): MainThread\n  step (scheduler.py:10)\n")
        (b / "pyspy.speedscope.json").write_text("{}")
        return subprocess.CompletedProcess(argv, 0, "", "")
    monkeypatch.setattr(labtools, "default_profile_runner", runner)
    out = tb.hostprof({"seconds": 5, "rate": 250})
    assert "MainThread" in out
    argv, exe = seen["argv"], shutil.which("py-spy")
    assert argv[1:3] == ["-m", "lab.profile"] and after(argv, "--profiler") == "pyspy"
    assert after(argv, "--seconds") == "5" and after(argv, "--rate") == "250"
    assert after(argv, "--pyspy-bin") == os.path.realpath(exe)
    assert os.path.dirname(os.path.realpath(exe)) in map(str, seen["read"])
    rec = last(tb, kind="profile")
    assert rec["tool"] == "hostprof" and (tb.s.ledger_root / rec["result"]["bundle"] / "bundle/b1/pyspy-dump.txt").exists()
    assert_charged(tb, rec)


@pytest.mark.parametrize("tool,args", [("trace", {}), ("kernel", {"kernel_regex": "gemm"}), ("hostprof", {})])
def test_refused_cleanly_without_the_instrument(tb, monkeypatch, tool, args):
    real = shutil.which
    monkeypatch.setattr(proftools.shutil, "which", lambda n, *a, **k: None if n in INSTRUMENTS else real(n, *a, **k))
    out = getattr(tb, tool)(args)
    assert out.startswith(f"{tool} refused:") and "not on PATH" in out
    rec = last(tb, kind="note")
    assert rec["tool"] == tool and rec["result"]["verdict"] == "refused" and calls(tb) == []


@pytest.mark.parametrize("bad", ["gemm; rm -rf /", "$(id)", "`id`", "a b", "x\nid", "a'b", 'a"b', "a&b", "../x",
                                 "", "a" * 201, "(", None, 7, ["gemm"]])
def test_kernel_regex_rejects_injection(tb, bad):
    with pytest.raises(ValueError):
        tb.kernel({"kernel_regex": bad})
    assert calls(tb) == [] and not list(ledger.records(tb.s.ledger_root))


@pytest.mark.parametrize("tool,args", [
    ("kernel", {"kernel_regex": "gemm", "set": "full --replay-mode application"}),
    ("kernel", {"kernel_regex": "gemm", "launch_count": 0}),
    ("kernel", {"kernel_regex": "gemm", "launch_count": "4; id"}),
    ("trace", {"requests": True}), ("trace", {"requests": "8 --out /"}), ("trace", {"max_tokens": 10**6}),
    ("hostprof", {"seconds": 301}), ("hostprof", {"rate": -1}),
])
def test_arguments_are_validated_before_anything_runs(tb, tool, args):
    with pytest.raises(ValueError):
        getattr(tb, tool)(args)
    assert calls(tb) == []


def test_cost_free_tools_record_wall_time_and_zero(tb):
    tb.note({"text": "hi"})
    tb.restore({"snapshot": "base"})
    for rec in ledger.records(tb.s.ledger_root):
        assert rec["cost"]["usd"] == 0.0 and rec["cost"]["seconds"] >= 0 and "rate_known" not in rec["cost"]
    assert tb.s.budget.spent_usd == 0.0
    assert {"test", "ledger", "budget", "restore", "note"}.isdisjoint(GPU_TOOLS)


def test_an_unknown_rate_is_recorded_as_unknown_and_charged_zero(tb, monkeypatch):
    monkeypatch.delenv("LAB_GPU_USD_PER_HOUR")
    monkeypatch.setattr(labtools, "default_profile_runner",
                        lambda tree, argv, read=(): subprocess.CompletedProcess(argv, 1, "", "no engine"))
    tb.profile({})
    c = last(tb, kind="profile")["cost"]
    assert c["usd"] == 0.0 and c["rate_known"] is False and c["seconds"] > 0
    assert any("no GPU rate is configured" in t.description for t in tb.specs() if t.name == "profile")


def test_gpu_time_is_charged_to_the_budget_and_stops_at_the_cap(tb, monkeypatch):
    monkeypatch.setattr(labtools, "default_profile_runner",
                        lambda tree, argv, read=(): subprocess.CompletedProcess(argv, 1, "", "no engine"))
    tb.profile({})
    spent = tb.s.budget.spent_usd
    assert spent > 0 and Budget.resume(100.0, "r1", tb.s.ledger_root).spent_usd == pytest.approx(spent)
    tb.s.budget.spent_usd = tb.s.budget.cap_usd
    with pytest.raises(BudgetExceeded):
        tb.trace({})
    assert calls(tb) == []
    specs = {t.name: t.description for t in tb.specs()}
    assert all("Costs GPU time" in specs[n] and "$3600/h" in specs[n] for n in GPU_TOOLS if n in specs)
    assert all("Costs no GPU time ($0)" in specs[n] for n in specs if n not in GPU_TOOLS)


def test_every_instrument_passes_a_workload_source_to_lab_profile(tb):
    """lab.profile refuses to run without one of --corpus-class/--workload/--prompts/--synthetic."""
    tb.trace({"requests": 2, "synthetic": True})
    prof = calls(tb)[0]
    assert "--synthetic" in prof and after(prof, "--prompt-len") == "64"
    tb.kernel({"kernel_regex": "gemm", "requests": 2})          # default: the seen corpus, staged in the harness
    ncu = [c for c in calls(tb) if c[0] == "ncu"][0]
    wl = after(ncu, "--workload")
    assert wl.endswith("_harness/workload.json")


def test_symlinks_left_by_the_jailed_run_are_not_copied_out(tb, monkeypatch):
    secret = tb.s.run_dir / "heldout-private" / "rows.jsonl"
    secret.parent.mkdir()
    secret.write_text("held-out")

    def runner(tree, argv, read=()):
        b = __import__("pathlib").Path(after(argv, "--out")) / "b1"
        b.mkdir(parents=True)
        (b / "meta.json").write_text("{}")
        (b / "leak").symlink_to(secret)
        (b / "dir").symlink_to(secret.parent, target_is_directory=True)
        return subprocess.CompletedProcess(argv, 0, "", "")
    monkeypatch.setattr(labtools, "default_profile_runner", runner)
    tb.profile({"synthetic": True})
    copy = tb.s.workspace.path / last(tb, kind="profile")["result"]["workspace_copy"]
    assert sorted(p.name for p in copy.iterdir()) == ["meta.json"]
