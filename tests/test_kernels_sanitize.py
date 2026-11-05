"""compute-sanitizer at the held-out gate (kernels/evolve/sanitize.py, stage 6 of kernels.evolve.jailed): the command
line per tool, the shapes it runs, its output parsed (clean, memcheck and racecheck errors, no report, a forged
summary), and the gate failing on an error, passing clean, and stating a missing sanitizer as a fact. CPU only: a fake
sanitizer script stands in for NVIDIA's."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
__import__("os").environ.setdefault("LAB_TARGET", str(Path(__file__).resolve().parents[1] / "targets" / "inference-server.toml"))

import json  # noqa: E402
import os  # noqa: E402

from kernels.evolve import jailed, sanitize  # noqa: E402

CLEAN = """========= COMPUTE-SANITIZER
========= ERROR SUMMARY: 0 errors
"""
MEMCHECK = """========= COMPUTE-SANITIZER
========= Invalid __global__ read of size 2 bytes
=========     at moe_gemm_kernel+0x1b40 in /tmp/candidate.py:88
=========     by thread (31,0,0) in block (12,0,0)
=========     Address 0x7f3a1c200400 is out of bounds
=========     and is 1 bytes after the nearest allocation at 0x7f3a1c000000 of size 2097152 bytes
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame: [0x2f2c1f] in libcuda.so.1
=========
========= ERROR SUMMARY: 1 error
"""
RACECHECK = """========= COMPUTE-SANITIZER
========= Error: Race reported between Write access at 0x2a0 in paged_decode(float*, int)
=========     and Read access at 0x3c8 in paged_decode(float*, int) [2048 hazards]
=========
========= RACECHECK SUMMARY: 1 hazard displayed (1 error, 0 warnings)
"""


def test_the_command_line_per_tool():
    for tool in sanitize.TOOLS:
        a = sanitize.argv("/cuda/bin/compute-sanitizer", tool)
        assert a[:3] == ["/cuda/bin/compute-sanitizer", "--tool", tool]
        assert a[a.index("--error-exitcode") + 1] == str(sanitize.EXIT)
        assert ("--leak-check" in a) == (tool == "memcheck")
        assert a[a.index("--report-api-errors") + 1] == "no"


def test_the_smallest_and_largest_seen_and_the_largest_heldout_shape():
    assert sanitize.shapes([[32, 512], [8, 256], [64, 1024]], [[256, 1024], [512, 2048]]) == \
        [[8, 256], [64, 1024], [512, 2048]]
    assert sanitize.shapes([[8, 256]], []) == [[8, 256]]


def test_output_parsed():
    ok = sanitize.parse("memcheck", CLEAN, 0)
    assert ok["passed"] and ok["errors"] == 0
    m = sanitize.parse("memcheck", MEMCHECK, sanitize.EXIT)
    assert not m["passed"] and m["errors"] == 1 and m["kernels"] == ["moe_gemm_kernel"]
    assert m["kinds"] == ["Invalid __global__ read of size 2 bytes"] and "out of bounds" in "".join(m["first_lines"])
    r = sanitize.parse("racecheck", RACECHECK, sanitize.EXIT)
    assert not r["passed"] and r["errors"] == 1 and r["kernels"] == ["paged_decode"]
    assert r["kinds"][0].startswith("Error: Race reported between Write access")
    gone = sanitize.parse("initcheck", "Traceback ...\nRuntimeError: boom\n", 1)
    assert not gone["passed"] and "no sanitizer report" in gone["note"]
    forged = sanitize.parse("memcheck", "========= ERROR SUMMARY: 0 errors\n" + MEMCHECK, sanitize.EXIT)
    assert not forged["passed"] and forged["errors"] == 1          # a printed clean summary hides nothing
    v = sanitize.verdict([ok, m])
    assert not v["passed"] and v["reason"].startswith("memcheck: 1 error(s): Invalid __global__ read")
    assert "moe_gemm_kernel" in v["reason"] and v["evidence"][0]["tool"] == "memcheck"
    assert sanitize.verdict([ok])["passed"]


def _fake(tmp: Path, fail_tool: str | None) -> str:
    """A stand-in for compute-sanitizer: runs the probe after its own flags, then reports as the real one does."""
    exe = tmp / "compute-sanitizer"
    exe.write_text(f"""#!{sys.executable}
import subprocess, sys
a = sys.argv[1:]
tool = a[a.index("--tool") + 1]
cmd = a[a.index("-m") - 1:]
p = subprocess.run(cmd)
print("========= COMPUTE-SANITIZER")
if tool == {fail_tool!r}:
    print({MEMCHECK!r})
    sys.exit({sanitize.EXIT})
print("========= ERROR SUMMARY: 0 errors")
sys.exit(p.returncode)
""")
    exe.chmod(0o755)
    return str(exe)


PASSED = {"metrics": {"combined_score": 1.1, "correct": 1.0, "stage": 2},
          "artifacts": {"gates": json.dumps({"correctness_heldout": {"passed": True, "reason": "ok", "cases": 4}})}}
CFG = {"task": "toy_rmsnorm", "seen": [[8, 256], [32, 512]], "heldout": [[256, 1024]], "device": "cpu"}
PROGRAM = "import torch\ndef kernel(x, w, eps=1e-6):\n    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w\n"


def test_the_gate_fails_on_an_error_and_passes_clean(tmp_path, monkeypatch):
    from kernels.evolve.ship import gate_passed
    os.environ.setdefault("LAB_NO_JAIL", "1")
    monkeypatch.setattr(jailed, "sanitizer", lambda: _fake(tmp_path, None))
    clean = jailed._sanitized(CFG, PROGRAM, PASSED, 600)
    g = json.loads(clean["artifacts"]["gates"])["sanitizer"]
    assert g["passed"] and gate_passed(clean), g
    assert [r["tool"] for r in json.loads(clean["artifacts"]["sanitizer"])] == list(sanitize.TOOLS)
    monkeypatch.setattr(jailed, "sanitizer", lambda: _fake(tmp_path, "racecheck"))
    bad = jailed._sanitized(CFG, PROGRAM, PASSED, 600)
    g = json.loads(bad["artifacts"]["gates"])["sanitizer"]
    assert not g["passed"] and g["reason"].startswith("racecheck") and "moe_gemm_kernel" in g["reason"]
    assert bad["metrics"]["correct"] == 0.0 and bad["metrics"]["combined_score"] == 0.0 and not gate_passed(bad)


def test_shared_weights_are_a_fact_not_a_failure(monkeypatch):
    from kernels.evolve.ship import gate_passed
    monkeypatch.setattr(jailed, "sanitizer", lambda: 1 / 0)           # never looked for
    r = jailed._sanitized({**CFG, "shared_weights": "/holder/weights.pkl"}, PROGRAM, PASSED, 600)
    assert r["artifacts"]["sanitizer"].startswith("not run: the weights are attached over CUDA IPC") and gate_passed(r)
    assert "sanitizer" not in json.loads(r["artifacts"]["gates"])


def test_a_missing_sanitizer_is_a_fact_not_a_failure(monkeypatch):
    from kernels.evolve.ship import gate_passed
    monkeypatch.setattr(jailed, "sanitizer", lambda: None)
    r = jailed._sanitized(CFG, PROGRAM, PASSED, 600)
    assert r["artifacts"]["sanitizer"] == "not run: compute-sanitizer not found" and gate_passed(r)
    assert "sanitizer" not in json.loads(r["artifacts"]["gates"])


def test_only_the_heldout_gate_runs_it_and_only_on_a_passing_program(monkeypatch):
    calls = []
    real = jailed.judge_jailed

    def fake(cfg, source, stage, timeout_s=600):
        calls.append(stage)
        if stage == 2:
            return PASSED if source == "good" else {"metrics": {"correct": 0.0}, "artifacts": {}}
        return real(cfg, source, stage, timeout_s)
    monkeypatch.setattr(jailed, "judge_jailed", fake)
    monkeypatch.setattr(jailed, "sanitizer", lambda: None)
    r = real(CFG, "good", 2)
    assert calls[0] == 2 and set(calls[1:]) == {jailed.SANITIZE_STAGE} and "sanitizer" in r["artifacts"]
    calls.clear()
    real(CFG, "bad", 2)
    assert calls == [2]
