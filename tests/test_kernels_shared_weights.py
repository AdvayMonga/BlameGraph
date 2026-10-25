"""Shared read-only weights (kernels/shared_weights.py), on the CPU's shared memory: another process attaches to the
holder's tensors without copying, reads the same values, and a write it makes is caught by the hash; the jailed
evaluator (kernels/evolve/jailed.py) voids a judgment that wrote one and restores it."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json  # noqa: E402
import subprocess  # noqa: E402

import torch  # noqa: E402

from kernels.shared_weights import Holder, attach, digest, harness, restore  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CHILD = """
import sys, torch
sys.path.insert(0, {root!r})
from kernels.shared_weights import attach
w = attach({path!r})
print(float(w["a"].sum()), w["a"].data_ptr() != 0)
if {write}:
    w["b"][0, 0] += 1
"""


def run(path, write, root=ROOT):
    code = CHILD.format(root=str(root), path=str(path), write=write)
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.split()


def holder(tmp_path):
    return Holder({"a": torch.arange(12, dtype=torch.float32).view(3, 4), "b": torch.ones(2, 2, dtype=torch.bfloat16)},
                  tmp_path / "w.pkl")


def test_a_child_reads_the_holders_weights_and_a_write_is_caught_and_restored(tmp_path):
    h = holder(tmp_path)
    total, ok = run(h.path, False)
    assert float(total) == 66.0 and ok == "True" and h.changed() == []
    run(h.path, True)
    assert h.changed() == ["b"]
    assert restore(h.path) == ["b"] and h.changed() == [] and float(h.tensors["b"][0, 0]) == 1.0


def test_the_digest_sees_order_tail_bytes_and_one_bit():
    a = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.uint8)
    b, c, d = a.clone(), a.clone(), a.clone()
    b[[0, 1]] = b[[1, 0]]
    c[-1] += 1
    d[4] ^= 1
    assert len({digest(a), digest(b), digest(c), digest(d)}) == 4 and digest(a) == digest(a.clone())


def test_the_check_mounts_only_this_module(tmp_path):
    root = harness(tmp_path / "h")
    assert sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()) == \
        ["kernels/__init__.py", "kernels/shared_weights.py"]
    assert run(holder(tmp_path).path, False, root)[0] == "66.0"


def fake_jail(monkeypatch, calls, result, write=None):
    """grader.jailed replaced: records its kwargs and the config it was given, optionally writes a shared weight (the
    candidate), and prints the keyed result line."""
    from kernels.evolve import jailed

    def fake(tree, argv, **kw):
        calls.append({**kw, "cfg": json.loads((tree / "eval_config.json").read_text())})
        if write:
            attach(write)["a"][0, 0] += 1
        key = (tree / ".result_key").read_text()
        return subprocess.CompletedProcess(argv, 0, json.dumps({**result, "result_key": key}) + "\n", "")
    monkeypatch.setattr(jailed.grader, "jailed", fake)
    return jailed


def test_a_judgment_keeps_its_call_weights_and_mounts_the_holder(tmp_path, monkeypatch):
    h = holder(tmp_path)
    calls = []
    jailed = fake_jail(monkeypatch, calls, {"metrics": {"combined_score": 0.0, "correct": 0.0, "stage": 1},
                                            "artifacts": {}, "passed": True, "reason": "built"})
    cfg = {"task": "toy_rmsnorm", "seen": [[4, 64], [8, 64]], "weights": {"[4, 64]": 3, "[8, 64]": 1},
           "shared_weights": str(h.path)}
    for stage in (1, jailed.QUICK_STAGE, jailed.PRECHECK_STAGE):
        jailed.judge_jailed(cfg, "def kernel(x, w): return x\n", stage)
    one, quick, pre = calls
    assert one["cfg"]["weights"] == cfg["weights"] and quick["cfg"]["weights"] == {"[4, 64]": 3}
    assert one["ipc_host"] and quick["ipc_host"] and "ipc_host" not in pre      # no GPU at the precheck
    assert h.path in one["read"]


def test_a_judgment_that_writes_a_shared_weight_is_void_and_the_weight_restored(tmp_path, monkeypatch):
    h = holder(tmp_path)
    jailed = fake_jail(monkeypatch, [], {"metrics": {"combined_score": 1.5, "correct": 1.0, "stage": 2},
                                         "artifacts": {}}, write=h.path)
    monkeypatch.setattr(jailed, "verify", lambda *a, **k: (_ for _ in ()).throw(AssertionError("verify ran")))
    r = jailed.judge_jailed({"task": "toy_rmsnorm", "seen": [[4, 64]], "shared_weights": str(h.path)},
                            "def kernel(x, w): return x\n", 2)
    assert r["metrics"]["combined_score"] == 0.0 and r["metrics"]["correct"] == 0.0
    assert "['a']" in r["artifacts"]["shared_weights_changed"]
    assert h.changed() == [] and float(h.tensors["a"][0, 0]) == 0.0


if __name__ == "__main__":
    import inspect
    import tempfile
    for name, fn in list(globals().items()):
        if name.startswith("test_") and "monkeypatch" not in inspect.signature(fn).parameters:
            fn(*[Path(tempfile.mkdtemp()) for _ in inspect.signature(fn).parameters]); print("ok", name)
