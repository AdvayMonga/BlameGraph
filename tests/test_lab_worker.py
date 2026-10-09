"""lab.worker: tree hashes, and the Remote transport end to end through a local shell (the same CLI ssh would run)."""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

import pytest

from lab import engine, serve, target, worker
from lab.safety import grader
from tests.lab_fixtures import make_repo
from tests.test_lab_evaltools import _target

MEASURE = {"names": ["single_stream"], "split": "seen", "tier": "short", "seed": 0, "jailed": False, "passive": False}


@pytest.fixture
def remote(tmp_path, monkeypatch):
    """A Remote whose 'host' is this machine, with its own cache; returns (remote, a pristine tree, calls made)."""
    monkeypatch.setenv("LAB_NO_JAIL", "1")
    repo = make_repo(tmp_path)
    spec = _target(tmp_path, repo, tmp_path / "ref")
    monkeypatch.setenv("LAB_TARGET", str(spec))
    monkeypatch.setenv("LAB_ENGINE_REPO", str(repo))
    target._cache.clear()
    tree = tmp_path / "tree"
    grader.export(repo, "HEAD", tree)
    r = worker.Remote([], str(engine.ENV_ROOT), sys.executable, {"LAB_WORKER_CACHE": str(tmp_path / "cache")})
    calls = []
    run = r._run
    r._run = lambda command, stdin=b"": calls.append(command.split()[0]) or run(command, stdin)
    yield r, tree, calls
    target._cache.clear()


def test_tree_sha_covers_content_and_exec_bit_and_refuses_links(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "f").write_text("x")
    first = worker.tree_sha(tmp_path / "a")
    (tmp_path / "a" / "f").chmod(0o755)
    second = worker.tree_sha(tmp_path / "a")
    (tmp_path / "a" / "f").write_text("y")
    assert len({first, second, worker.tree_sha(tmp_path / "a")}) == 3
    (tmp_path / "a" / "l").symlink_to("/etc/passwd")
    with pytest.raises(worker.WorkerError):
        worker.tree_sha(tmp_path / "a")


def test_remote_runs_the_job_on_the_shipped_tree_and_ships_it_once(remote, tmp_path):
    r, tree, calls = remote
    res = r.call("test", {"tree": tree}, {}, tmp_path / "out1")
    assert res["lint"]["passed"] and res["tests"]["passed"], res
    assert calls == ["has", "put", "run"]
    calls.clear()
    r.call("test", {"tree": tree}, {}, tmp_path / "out2")
    assert calls == ["has", "run"]                              # cached by hash: not shipped again


def test_remote_carries_gpu_busy_with_its_facts(remote, tmp_path, monkeypatch):
    r, tree, _ = remote
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    smi = bin_dir / "nvidia-smi"
    smi.write_text(f"#!{sys.executable}\nimport sys\n"
                   "if any(a.startswith('--query-compute-apps') for a in sys.argv): print('4242, 9000, python')\n")
    smi.chmod(smi.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    with pytest.raises(serve.GpuBusy) as e:
        r.call("measure", {"tree": tree}, MEASURE, tmp_path / "out")
    assert e.value.apps == [{"pid": 4242, "used_mib": 9000, "name": "python"}]


def test_remote_carries_an_engine_that_did_not_start(remote, tmp_path, monkeypatch):
    r, tree, _ = remote
    spec = Path(os.environ["LAB_TARGET"])
    spec.write_text(spec.read_text().replace('launch = "python ', 'launch = "python -c \'import sys; sys.exit(3)\' #', 1))
    with pytest.raises(serve.NotReady):
        r.call("measure", {"tree": tree}, MEASURE, tmp_path / "out")
    assert (tmp_path / "out" / worker.ENGINE_LOG).exists()      # what was written still comes back


def test_a_tree_changed_after_it_was_put_is_refused(remote, tmp_path):
    r, tree, _ = remote
    r.call("test", {"tree": tree}, {}, tmp_path / "out1")
    cached = tmp_path / "cache" / worker.tree_sha(tree)
    (cached / "src" / "inference_server" / "engine.py").write_text("def speed():\n    return 100\n")
    with pytest.raises(worker.WorkerError, match="changed after it was put"):
        r.call("test", {"tree": tree}, {}, tmp_path / "out2")


def test_put_refuses_a_tree_that_does_not_hash_to_its_name(remote):
    r, tree, _ = remote
    with pytest.raises(worker.WorkerError, match="received tree hashes to"):
        r._run(f"put {'0' * 64}", worker._tar(tree))


def test_local_by_default_and_nothing_else_but_ssh(monkeypatch):
    monkeypatch.delenv("LAB_WORKER", raising=False)
    assert isinstance(worker.load(), worker.Local)
    monkeypatch.setenv("LAB_WORKER", "gpu-box")
    with pytest.raises(ValueError):
        worker.load()
