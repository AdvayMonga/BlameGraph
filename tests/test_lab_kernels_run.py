"""lab kernels run / status and the job on the VM, with fakes of the VM, the provider and the search: snapshots in the
store, saves per generation, the VM stopped at every end and deleted from here, secrets on stdin only, the service
account's grants."""

from __future__ import annotations

import io
import json
import types

import pytest

from lab.infra.cloud import vm as labvm
from lab.kernels import run as runs
from lab.kernels.run import job, launch, status
from lab.records.remote import DirRemote


def _folder(tmp_path):
    out = tmp_path / "local" / "kgen-1"
    (out / "search").mkdir(parents=True)
    (out / "search" / "trace.jsonl").write_text("{}\n")
    for skip in runs.SKIP:
        (out / skip).mkdir()
        (out / skip / "big").write_text("x")
    (out / "search" / "tree").mkdir()                  # a SKIP name below the top level is kept
    return out


def test_snapshots_are_numbered_and_restore_the_last_without_what_the_vm_rebuilds(tmp_path):
    remote, out = DirRemote(tmp_path / "store"), _folder(tmp_path)
    assert runs.latest(remote, "kgen-1") == 0 and runs.state(remote, "kgen-1") is None
    assert runs.save(remote, "kgen-1", out, {"generation": 0}) == 1
    (out / "search" / "trace.jsonl").write_text("{}\n{}\n")
    assert runs.save(remote, "kgen-1", out, {"generation": 1}) == 2
    assert runs.state(remote, "kgen-1") == {"generation": 1} and runs.last_run(remote) == "kgen-1"
    back = tmp_path / "back" / "kgen-1"
    assert runs.restore(remote, "kgen-1", back)
    assert (back / "search" / "trace.jsonl").read_text() == "{}\n{}\n" and (back / "search" / "tree").is_dir()
    assert not any((back / s).exists() for s in runs.SKIP)
    assert not runs.restore(remote, "kgen-2", tmp_path / "none")


class FakeProvider:
    def __init__(self):
        self.stopped = []

    def stop_id(self, instance):
        self.stopped.append(instance)


def _job(tmp_path, monkeypatch, remote=None):
    remote = remote or DirRemote(tmp_path / "store")
    out = tmp_path / "runs" / "kgen-1"
    out.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(runs, "folder", lambda run: tmp_path / "runs" / run)
    monkeypatch.setattr(job.sync, "push_ledger", lambda root, r: 0)
    monkeypatch.setattr(job.sync, "pull_ledger", lambda root, r: 0)
    a = types.SimpleNamespace(run="kgen-1", task="region:a:b", vm="lab-gpu", instance="computeinstance-1",
                              deadline=4e9)
    return job.Job(a, remote, FakeProvider()), remote, out


def test_the_job_saves_when_the_search_saves_a_generation(tmp_path, monkeypatch):
    j, remote, out = _job(tmp_path, monkeypatch)
    monkeypatch.setattr(job, "SAVE_S", 0.01)
    (out / "search").mkdir()
    (out / "search" / "state.json").write_text(json.dumps({"ran": 3}))
    import threading
    monkeypatch.setattr(job, "saved", lambda d: json.loads((d / "state.json").read_text())["ran"])
    t = threading.Thread(target=j.watch)
    t.start()
    import time
    deadline = time.time() + 5
    while runs.latest(remote, "kgen-1") == 0 and time.time() < deadline:
        time.sleep(0.01)
    j.ended.set()
    t.join()
    assert runs.state(remote, "kgen-1")["generation"] == 3


def test_the_end_saves_then_stops_the_vm_even_when_the_save_fails(tmp_path, monkeypatch):
    j, remote, out = _job(tmp_path, monkeypatch)
    j.end("done")
    j.end("failed", "late")                             # once only
    assert runs.state(remote, "kgen-1")["state"] == "done" and j.provider.stopped == ["computeinstance-1"]

    class Broken(DirRemote):
        def put(self, key, path):
            raise OSError("store down")
    j, _, _ = _job(tmp_path / "b", monkeypatch, Broken(tmp_path / "b" / "store"))
    j.end("failed", "boom")
    assert j.provider.stopped == ["computeinstance-1"]


def _main(tmp_path, monkeypatch, search):
    remote, provider = DirRemote(tmp_path / "store"), FakeProvider()
    monkeypatch.setattr(runs, "folder", lambda run: tmp_path / "runs" / run)
    monkeypatch.setattr(job.sync, "push_ledger", lambda root, r: 0)
    monkeypatch.setattr(job.sync, "pull_ledger", lambda root, r: 0)
    monkeypatch.setattr("lab.records.remote.open_remote", lambda: remote)
    monkeypatch.setattr("lab.infra.cloud.providers.load", lambda name: provider)
    monkeypatch.setattr(job, "detach", lambda log: False)
    monkeypatch.setattr(job, "search", search)
    monkeypatch.setattr(job, "SAVE_S", 3600.0)
    monkeypatch.setattr(job, "saved", lambda d: -1)
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({"ANTHROPIC_API_KEY": "sk-test"})))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    argv = ["--run", "kgen-1", "--task", "region:a:b", "--vm", "lab-gpu", "--instance", "ci-1", "--deadline", "4e9"]
    return remote, provider, argv


def test_the_job_ends_done_or_failed_and_stops_its_vm_either_way(tmp_path, monkeypatch):
    import os
    seen = {}

    def ok(task, out, stage):
        seen["key"] = os.environ.get("ANTHROPIC_API_KEY")
        stage("search")
        return {}
    remote, provider, argv = _main(tmp_path, monkeypatch, ok)
    assert job.main(argv) == 0
    assert seen["key"] == "sk-test" and "sk-test" not in " ".join(argv)
    assert runs.state(remote, "kgen-1")["state"] == "done" and provider.stopped == ["ci-1"]

    def boom(task, out, stage):
        raise RuntimeError("no capture traffic")
    remote, provider, argv = _main(tmp_path / "f", monkeypatch, boom)
    job.main(argv)
    st = runs.state(remote, "kgen-1")
    assert st["state"] == "failed" and "no capture traffic" in st["reason"] and provider.stopped == ["ci-1"]


def test_a_run_on_a_new_vm_starts_from_its_last_snapshot(tmp_path, monkeypatch):
    def search(task, out, stage):
        assert (out / "search" / "trace.jsonl").read_text() == "{}\n"
        return {}
    remote, provider, argv = _main(tmp_path, monkeypatch, search)
    runs.save(remote, "kgen-1", _folder(tmp_path / "old"), {"state": "running"})
    job.main(argv)
    assert runs.state(remote, "kgen-1")["state"] == "done"


class FakeNebius:
    default_user = "lab"

    def __init__(self):
        self.calls, self.state = [], "absent"

    def account(self):
        self.calls.append("account")
        return "sa-1"

    def allow_stop(self, name):
        self.calls.append(("allow_stop", name))

    def instance_id(self, name):
        return "ci-9"

    def get(self, name):
        return None if self.state == "absent" else labvm.Record(name, self.state, "203.0.113.9", "t")


def _launch(monkeypatch, setup_code=0, reply="started kgen-1\n"):
    p, did = FakeNebius(), []
    v = labvm.VM(name="lab-gpu", provider=p)

    def start(v_, spot=True, account=None):
        did.append(("start", account))
        p.state = "running"
        return p.get(v_.name)
    monkeypatch.setattr(labvm, "start", start)
    monkeypatch.setattr(labvm, "wait_ssh", lambda v_, r: None)
    monkeypatch.setattr(labvm, "left_s", lambda v_: 0.0 if not any(d == "arm" for d in did) else 43000.0)
    monkeypatch.setattr(labvm, "arm", lambda v_, r, m: did.append("arm"))
    monkeypatch.setattr(labvm, "setup", lambda v_: did.append("setup") or setup_code)
    monkeypatch.setattr(labvm, "stop", lambda v_, keep_disk=False: did.append("stop"))

    def ssh(argv, input, text, capture_output, timeout):
        did.append(("ssh", argv[-1], json.loads(input)))
        return types.SimpleNamespace(returncode=0, stdout=reply, stderr="")
    monkeypatch.setattr(launch.subprocess, "run", ssh)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("LAB_STORE", "s3://bucket")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws-secret")
    monkeypatch.setenv("LAB_TARGET", "evals/targets/inference-server.toml")
    return v, p, did


def test_launch_creates_the_vm_with_the_account_and_sends_secrets_on_stdin_only(monkeypatch):
    v, p, did = _launch(monkeypatch)
    launch.go(v, "kgen-1", ["lab.kernels.run.job", "--task", "region:a:b"])
    assert did[0] == ("start", "sa-1") and did[1:3] == ["arm", "setup"] and ("allow_stop", "lab-gpu") in p.calls
    _, cmd, sent = did[-1]
    assert "python -m lab.kernels.run.job --run kgen-1 --task region:a:b --vm lab-gpu --instance ci-9 --deadline" in cmd
    assert sent["ANTHROPIC_API_KEY"] == "sk-test" and sent["AWS_SECRET_ACCESS_KEY"] == "aws-secret"
    assert sent["LAB_TARGET"] == "evals/targets/inference-server.toml"
    assert "sk-test" not in cmd and "aws-secret" not in cmd and "stop" not in did


def test_launch_deletes_the_vm_when_setup_or_the_start_of_the_job_fails(monkeypatch):
    v, p, did = _launch(monkeypatch, setup_code=1)
    with pytest.raises(RuntimeError, match="setup` failed"):
        launch.go(v, "kgen-1", ["lab.kernels.run.job", "--task", "region:a:b"])
    assert did[-1] == "stop" and not any(isinstance(d, tuple) and d[0] == "ssh" for d in did)
    v, p, did = _launch(monkeypatch, reply="Traceback ...")
    with pytest.raises(RuntimeError, match="did not start"):
        launch.go(v, "kgen-1", ["lab.kernels.run.job", "--task", "region:a:b"])
    assert did[-1] == "stop"


def test_missing_names_what_this_host_lacks(monkeypatch):
    for k in ("ANTHROPIC_API_KEY", "LAB_STORE", "LAB_TARGET"):
        monkeypatch.delenv(k, raising=False)
    assert len(launch.missing()) == 3


def _status(tmp_path, monkeypatch, state, vm_state):
    remote = DirRemote(tmp_path / "store")
    out = tmp_path / "f"
    out.mkdir(parents=True)
    runs.save(remote, "kgen-1", out, {"run": "kgen-1", "task": "region:a:b", "vm": "lab-gpu", "state": state,
                                      "stage": "search", "generation": 4, "updated": 0.0, "deadline": 4e9})
    monkeypatch.setattr("lab.records.remote.open_remote", lambda: remote)
    p = FakeNebius()
    p.state = vm_state
    monkeypatch.setattr(labvm, "VM", lambda name: types.SimpleNamespace(name=name, provider=p))
    went = []
    monkeypatch.setattr(launch, "go", lambda v, run, job: went.append((v.name, run, job[-1])))
    monkeypatch.setattr(labvm, "stop", lambda v, keep_disk=False: went.append(("deleted", v.name)))
    monkeypatch.setattr(launch, "missing", lambda: [])
    return went


def test_status_starts_a_preempted_run_again_deletes_an_ended_runs_vm_and_leaves_others(tmp_path, monkeypatch, capsys):
    went = _status(tmp_path, monkeypatch, "running", "stopped")
    assert status.main([]) == 0 and went == [("lab-gpu", "kgen-1", "region:a:b")]
    went = _status(tmp_path / "d", monkeypatch, "done", "absent")
    assert status.main(["--run", "kgen-1"]) == 0 and went == []
    for ended in ("done", "failed", "stopped"):
        went = _status(tmp_path / ended, monkeypatch, ended, "stopped")
        assert status.main([]) == 0 and went == [("deleted", "lab-gpu")]
    went = _status(tmp_path / "g", monkeypatch, "running", "absent")
    status.main([])
    assert went == [] and "goes on from generation 4" in capsys.readouterr().out


def _nebius(monkeypatch, groups=(), accounts=(), permits=(), instances=()):
    from lab.infra.cloud.providers import nebius
    seen = []

    def cli(self, *args):
        seen.append(args)
        items = {("iam", "group", "list"): groups, ("iam", "service-account", "list"): accounts,
                 ("iam", "access-permit", "list"): permits, ("compute", "instance", "list"): instances}
        if args[:3] in items:
            return json.dumps({"items": list(items[args[:3]])})
        if args[:3] in (("iam", "service-account", "create"), ("iam", "group", "create")):
            return json.dumps({"metadata": {"id": f"new-{args[1]}"}})
        if args[:3] == ("vpc", "subnet", "list"):
            return json.dumps({"items": [{"metadata": {"id": "sub1"}}]})
        return "{}"
    monkeypatch.setattr(nebius.Nebius, "_cli", cli)
    return nebius.Nebius(), seen


def _meta(id_, name):
    return {"metadata": {"id": id_, "name": name}}


def test_the_account_and_its_group_are_made_once(monkeypatch):
    p, seen = _nebius(monkeypatch)
    assert p.account() == "new-service-account"
    assert ("iam", "group-membership", "create", "--parent-id", "new-group", "--member-id", "new-service-account") in seen
    p, seen = _nebius(monkeypatch, groups=[_meta("g1", "lab-vm")], accounts=[_meta("sa1", "lab-vm")])
    assert p.account() == "sa1" and not any(a[2] == "create" for a in seen)


def test_the_group_powers_only_the_new_vm_and_permits_on_gone_vms_are_removed(monkeypatch):
    permits = [{"metadata": {"id": "p-old"}, "spec": {"resource_id": "ci-gone"}},
               {"metadata": {"id": "p-live"}, "spec": {"resource_id": "ci-1"}}]
    p, seen = _nebius(monkeypatch, groups=[_meta("g1", "lab-vm")], permits=permits,
                      instances=[_meta("ci-1", "other"), _meta("ci-2", "lab-gpu")])
    p.allow_stop("lab-gpu")
    assert ("iam", "access-permit", "delete", "p-old") in seen and ("iam", "access-permit", "delete", "p-live") not in seen
    assert ("iam", "access-permit", "create", "--parent-id", "g1", "--resource-id", "ci-2", "--role", "compute.instance-power-operator") in seen


def test_a_vm_is_created_as_the_account_and_stops_itself_by_id(tmp_path, monkeypatch):
    key = tmp_path / "k.pub"
    key.write_text("ssh-ed25519 AAA")
    p, seen = _nebius(monkeypatch)
    p.create("n1", str(key), True, "sa1")
    create = next(a for a in seen if a[:3] == ("compute", "instance", "create"))
    assert create[create.index("--service-account-id") + 1] == "sa1"
    p.stop_id("ci-7")
    assert seen[-1] == ("compute", "instance", "stop", "ci-7")
