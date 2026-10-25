"""Shared read-only weights (kernels/shared_weights.py), on the CPU's shared memory: another process attaches to the
holder's tensors without copying, reads the same values, and a write it makes is caught against the pristine copy;
the jailed evaluator (kernels/evolve/jailed.py) voids a judgment that wrote one, restores it on every exit path, and
runs judgments on one holder one at a time."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json  # noqa: E402
import subprocess  # noqa: E402

import torch  # noqa: E402


from kernels.shared_weights import Holder, attach, constants, harness, restore  # noqa: E402

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


def test_a_one_bit_write_is_caught():
    from kernels.shared_weights import _equal
    a = torch.arange(9, dtype=torch.uint8)
    b = a.clone()
    b[4] ^= 1
    assert _equal(a, a.clone()) and not _equal(a, b)


def test_files_others_can_write_or_under_dev_shm_are_never_unpickled(tmp_path):
    import pytest
    from kernels.shared_weights import _trusted
    h = holder(tmp_path)
    for f in (h.path, h.path.with_name("w.pkl.constants"), h.path.with_name("w.pkl.pristine")):
        f.chmod(0o666)
        with pytest.raises(ValueError, match="writable by no one else"):
            attach(h.path) if f == h.path else constants(h.path) if f.name.endswith("constants") else restore(h.path)
        f.chmod(0o644)
    assert constants(h.path) == {} and restore(h.path) == []
    with pytest.raises(ValueError, match="/dev/shm"):
        _trusted(Path("/dev/shm/w.pkl"))


def test_poison_never_writes_a_shared_weight(tmp_path):
    from kernels.check import _poison
    w, x = attach(holder(tmp_path).path)["a"], torch.ones(3)
    _poison((w, x))
    assert torch.equal(w, torch.arange(12, dtype=torch.float32).view(3, 4)) and x.isnan().any()


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
        if result is None:
            raise RuntimeError("the jail died")
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


def test_a_judgment_that_dies_after_writing_still_restores_the_weight(tmp_path, monkeypatch):
    import pytest
    h = holder(tmp_path)
    jailed = fake_jail(monkeypatch, [], None, write=h.path)
    with pytest.raises(RuntimeError, match="died"):
        jailed.judge_jailed({"task": "toy_rmsnorm", "seen": [[4, 64]], "shared_weights": str(h.path)}, "x = 1\n", 1)
    assert h.changed() == []


def test_an_ncu_run_that_writes_a_weight_keeps_the_evidence(tmp_path, monkeypatch):
    h = holder(tmp_path)
    jailed = fake_jail(monkeypatch, [], {"metrics": {"stage": 3}, "artifacts": {}}, write=h.path)
    monkeypatch.setattr(jailed.shutil, "which", lambda name: "/bin/sh")
    r = jailed.judge_jailed({"task": "toy_rmsnorm", "seen": [[4, 64]], "shared_weights": str(h.path)}, "x = 1\n",
                            jailed.NCU_STAGE)
    assert "wrote shared weights ['a']" in json.loads(r["artifacts"]["ncu"])["shapes"][0]["error"]


def test_judgments_on_one_holder_run_one_at_a_time(tmp_path, monkeypatch):
    import threading
    import time
    h = holder(tmp_path)
    spans = []
    jailed = fake_jail(monkeypatch, [], {"metrics": {"combined_score": 0.0, "correct": 0.0, "stage": 1},
                                         "artifacts": {}})
    inner = jailed.grader.jailed

    def slow(tree, argv, **kw):
        t0 = time.monotonic()
        time.sleep(0.3)
        spans.append((t0, time.monotonic()))
        return inner(tree, argv, **kw)
    monkeypatch.setattr(jailed.grader, "jailed", slow)
    cfg = {"task": "toy_rmsnorm", "seen": [[4, 64]], "shared_weights": str(h.path)}
    ts = [threading.Thread(target=jailed.judge_jailed, args=(cfg, f"x = {i}\n", 1)) for i in range(2)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    (a0, a1), (b0, b1) = sorted(spans)
    assert len(spans) == 2 and b0 >= a1


def test_a_holder_under_a_dir_the_jail_writes_is_refused(tmp_path, monkeypatch):
    import pytest
    from kernels.evolve import jailed
    monkeypatch.setattr(jailed, "EVAL_DIRS", tmp_path)
    h = holder(tmp_path)
    with pytest.raises(ValueError, match="which a jail writes"):
        jailed.judge_jailed({"task": "toy_rmsnorm", "seen": [[4, 64]], "shared_weights": str(h.path)}, "x = 1\n", 1)


def test_a_dir_a_program_made_unwritable_is_still_removed(tmp_path):
    from kernels.evolve import jailed
    d = tmp_path / "torch_extensions" / "ext"
    (d / "keep").mkdir(parents=True)
    (d / "keep" / "answers").write_text("42")
    (d / "keep").chmod(0o500)
    d.chmod(0o500)
    jailed._keep_builds_only(tmp_path)
    assert list(d.iterdir()) == []

def test_the_holders_own_process_restores_without_reopening_its_handles(tmp_path, monkeypatch):
    from kernels import shared_weights as sw
    h = sw.Holder({"w": torch.ones(4)}, tmp_path / "w.pkl")
    monkeypatch.setattr(sw.pickle, "loads", lambda b: (_ for _ in ()).throw(AssertionError("reopened its own handles")))
    h.tensors["w"].add_(1)
    assert sw.restore(h.path) == ["w"] and torch.equal(h.tensors["w"], torch.ones(4))


if __name__ == "__main__":
    import importlib.util
    import inspect
    import tempfile
    have_pytest = importlib.util.find_spec("pytest") is not None
    for name, fn in list(globals().items()):
        if name.startswith("test_") and "monkeypatch" not in inspect.signature(fn).parameters:
            if not have_pytest and "pytest." in inspect.getsource(fn):
                print("skip", name, "(needs pytest)")
                continue
            fn(*[Path(tempfile.mkdtemp()) for _ in inspect.signature(fn).parameters]); print("ok", name)
