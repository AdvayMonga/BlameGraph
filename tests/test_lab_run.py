"""lab run: nothing started while something is missing or a VM exists, an ended run's VM deleted first, the job's
command and the billing line."""
from __future__ import annotations

import types

import pytest

from lab import run
from lab.infra.cloud import vm as labvm
from lab.kernels.run import launch
from lab.loop import record
from lab.loop import setup as S
from lab.records.remote import DirRemote


class FakeProvider:
    default_user = "lab"

    def __init__(self, state=None):
        self.state = state

    def get(self, name):
        return None if self.state is None else labvm.Record(name, self.state, "203.0.113.9", "t")

    def account(self):
        return "sa-1"


@pytest.fixture
def host(tmp_path, monkeypatch):
    h = types.SimpleNamespace(provider=FakeProvider(), remote=DirRemote(tmp_path / "store"), went=[], deleted=[],
                              problems=())
    real = labvm.VM
    monkeypatch.setattr(labvm, "VM", lambda: real(name="lab-gpu", provider=h.provider))
    monkeypatch.setattr(labvm, "left_s", lambda v: 3600.0)
    monkeypatch.setattr(S, "setup", lambda v: h.problems)
    monkeypatch.setattr("lab.records.remote.open_remote", lambda: h.remote)

    def go(v, run_id, job):
        h.went.append((run_id, job))
        h.provider.state = "running"

    def stop(v, keep_disk=False):
        h.deleted.append(v.name)
        h.provider.state = None
    monkeypatch.setattr(launch, "go", go)
    monkeypatch.setattr(labvm, "stop", stop)
    return h


def test_no_flags_start_one_turn_of_the_default_task(host, capsys):
    assert run.main([]) == 0
    (run_id, job), = host.went
    assert run_id.startswith("run-") and job == ["lab.loop.job", "--task", "lab/tasks/single_stream.toml"]
    out = capsys.readouterr().out
    assert f"{run_id}: running on lab-gpu" in out
    assert out.strip().splitlines()[-1] == "billing: VM lab-gpu is running; deleted by its deadline in 1.0 h"


def test_the_turns_and_the_budget_go_to_the_job(host):
    assert run.main(["--n", "loop", "--budget", "40"]) == 0 and run.main(["--n", "3"]) == 2    # the VM now exists
    assert host.went[0][1][3:] == ["--n", "loop", "--budget", "40"] and len(host.went) == 1


def test_nothing_is_started_on_bad_flags_a_problem_or_an_existing_vm(host, capsys):
    assert run.main(["--n", "loop"]) == 2 and "--n loop needs --budget" in capsys.readouterr().err
    assert run.main(["--task", "/etc/hosts"]) == 2
    host.problems = (S.Problem("no model key", "set ANTHROPIC_API_KEY in .env"),)
    assert run.main([]) == 2
    assert "- no model key\n  fix: set ANTHROPIC_API_KEY in .env" in capsys.readouterr().err
    host.problems, host.provider.state = (), "running"
    assert run.main([]) == 2
    io = capsys.readouterr()
    assert "lab-gpu already exists" in io.err and io.out.startswith("billing: VM lab-gpu is running")
    assert host.went == [] and host.deleted == []


def test_the_vm_of_an_ended_run_is_deleted_before_the_next_starts(host, capsys):
    record.record(host.remote, "run-0", {"state": "done"}, host.remote.root / "none")
    host.provider.state = "stopped"
    assert run.main([]) == 0
    assert host.deleted == ["lab-gpu"] and len(host.went) == 1 and "the VM of run-0" in capsys.readouterr().out
