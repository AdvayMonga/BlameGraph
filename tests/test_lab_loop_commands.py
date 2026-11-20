"""setup, lab status and the job on the VM, with fakes of the provider, the store and the loop."""
from __future__ import annotations

import io
import json
import types

from lab.infra.cloud import vm as labvm
from lab.infra.cloud.providers import ProviderError
from lab.loop import job, loop, record, status
from lab.loop.setup import setup
from lab.records.remote import DirRemote

ENV = {"LAB_TARGET": "evals/targets/inference-server.toml", "ANTHROPIC_API_KEY": "sk", "GH_TOKEN": "gh"}


class FakeProvider:
    default_user = "lab"

    def __init__(self, state=None, login=True, account=True):
        self.state, self.login, self.ok, self.stopped, self.asked = state, login, account, [], []

    def get(self, name):
        if not self.login:
            raise ProviderError("not logged in")
        return None if self.state is None else labvm.Record(name, self.state, "203.0.113.9", "t")

    def account(self):
        self.asked.append("account")
        if not self.ok:
            raise ProviderError("permission denied")
        return "sa-1"

    def stop_id(self, instance):
        self.stopped.append(instance)


def _vm(p):
    return labvm.VM(name="lab-gpu", provider=p)


def test_setup_finds_nothing_on_a_ready_machine_and_names_each_missing_thing_with_its_fix(tmp_path):
    env = {**ENV, "LAB_STORE": str(tmp_path / "store")}
    p = FakeProvider()
    assert setup(_vm(p), env) == () and p.asked == ["account"]
    found = setup(_vm(FakeProvider(account=False)), {"LAB_TARGET": "/etc/target.toml", "LAB_STORE": str(tmp_path)})
    assert [x.what.split(":")[0] for x in found] == [
        "LAB_TARGET names no target spec inside this repo", "no model key", "no GitHub token",
        "no service account for the VM to stop itself as"]
    assert all(x.fix for x in found)


def test_setup_does_not_ask_for_the_account_when_the_login_fails(tmp_path, monkeypatch):
    monkeypatch.delenv("LAB_STORE", raising=False)
    p = FakeProvider(login=False)
    found = setup(_vm(p), ENV)
    assert [x.what.split(":")[0] for x in found] == ["no store", "the cloud cannot be asked"] and p.asked == []


def _status(tmp_path, monkeypatch, state, vm_state):
    remote, p, deleted = DirRemote(tmp_path / "store"), FakeProvider(vm_state), []
    record.record(remote, "run-1", {"state": state, "stage": "turn", "turn": 2, "wins": 1, "spent_usd": 12.5},
                  tmp_path / "ledger")
    monkeypatch.setattr(labvm, "stop", lambda v, keep_disk=False: deleted.append(v.name) or setattr(p, "state", None))
    monkeypatch.setattr(labvm, "left_s", lambda v: 0.0)
    return remote, _vm(p), deleted


def test_status_reads_the_newest_run_and_leaves_a_running_vm(tmp_path, monkeypatch):
    remote, v, deleted = _status(tmp_path, monkeypatch, "running", "running")
    lines = status.status(remote, v)
    assert lines[0].startswith("run-1: running, turn, turn 2, 1 win(s), $12.50") and deleted == []
    assert status.billing(v) == "billing: VM lab-gpu is running; `lab vm stop` deletes it"
    assert status.status(remote, v, "run-9") == ["run-9 is not in the store"]


def test_status_deletes_the_vm_of_a_run_that_ended_or_lost_it_and_prints_the_report(tmp_path, monkeypatch):
    remote, v, deleted = _status(tmp_path, monkeypatch, "done", "stopped")
    record._put(remote, f"{record.PREFIX}/run-1/report.md", "# run-1\n\nturn 1: win\n")
    lines = status.status(remote, v)
    assert "turn 1: win" in lines[1] and lines[-1] == "  lab-gpu deleted" and deleted == ["lab-gpu"]
    assert status.billing(v) == "billing: nothing"
    remote, v, deleted = _status(tmp_path / "b", monkeypatch, "running", "stopped")
    lines = status.status(remote, v)
    assert "its VM is stopped: the run ended without finishing" in lines[1] and deleted == ["lab-gpu"]
    remote, v, deleted = _status(tmp_path / "c", monkeypatch, "failed", None)
    assert len(status.status(remote, v)) == 1 and deleted == []


def _job(tmp_path, monkeypatch, run):
    remote, p = DirRemote(tmp_path / "store"), FakeProvider()
    monkeypatch.setattr("lab.records.remote.open_remote", lambda: remote)
    monkeypatch.setattr("lab.infra.cloud.providers.load", lambda name: p)
    monkeypatch.setattr("lab.kernels.run.job.detach", lambda log: False)
    monkeypatch.setattr(job.engine, "ENV_ROOT", tmp_path)
    monkeypatch.setattr(job.engine, "repo", lambda: tmp_path)
    monkeypatch.setattr(job.engine, "python", lambda: "python")
    monkeypatch.setattr(job.sync, "pull_ledger", lambda root, r: 0)
    monkeypatch.setattr(job.subprocess, "run", lambda *a, **k: types.SimpleNamespace(stdout="main\n"))
    monkeypatch.setattr(job, "parts", lambda provider, instance: "parts")
    monkeypatch.setattr("lab.budget.gpu_rate", lambda: (3.0, True))
    monkeypatch.setattr("evals.targets.target.load", lambda: "target")
    monkeypatch.setattr("lab.task.Task.parse", lambda path: f"task {path.name}")
    monkeypatch.setattr(loop, "run", run)
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({"LAB_TARGET": "t.toml", "ANTHROPIC_API_KEY": "sk-test"})))
    monkeypatch.setenv("LAB_TARGET", "t.toml")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    return remote, p, ["--run", "run-1", "--task", "lab/tasks/t.toml", "--vm", "lab-gpu", "--instance", "ci-1",
                       "--deadline", "4e9"]


def test_the_job_hands_the_loop_its_plan_the_task_and_the_engines_branch(tmp_path, monkeypatch):
    import os
    seen = {}

    def run(plan, task, t, wk, parts, remote, work, onto):
        seen.update(plan=plan, task=task, onto=onto, work=work, key=os.environ.get("ANTHROPIC_API_KEY"))
    remote, p, argv = _job(tmp_path, monkeypatch, run)
    assert job.main([*argv, "--n", "loop", "--budget", "40"]) == 0
    assert seen["plan"] == loop.Plan("run-1", None, 40.0, 3.0) and seen["task"] == "task t.toml"
    assert seen["onto"] == "main" and seen["work"] == tmp_path / "local" / "runs" / "loop" / "run-1"
    assert seen["key"] == "sk-test" and p.stopped == []          # the loop's own finish stops the VM


def test_a_job_that_cannot_start_its_loop_records_the_failure_and_stops_the_vm(tmp_path, monkeypatch):
    remote, p, argv = _job(tmp_path, monkeypatch, lambda *a: None)
    assert job.main([*argv, "--n", "loop"]) == 0                 # --n loop without a budget
    st = record.status(remote, "run-1")
    assert st["state"] == "failed" and "--n loop needs --budget" in st["reason"] and p.stopped == ["ci-1"]


def test_no_way_is_wired():
    from lab.budget import Budget
    from lab.loop.types import Candidate, Gap
    import pytest
    with pytest.raises(NotImplementedError, match="no way is wired for the kernels gap"):
        job.WAYS["kernels"](Gap("kernels", 1, None, 1), Candidate(job.engine.ENV_ROOT), job.engine.ENV_ROOT, Budget(1))
