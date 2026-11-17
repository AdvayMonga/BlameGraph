"""lab.infra.cloud.vm against a fake `nebius` CLI, ssh and rsync on PATH: the lifecycle it drives, not the cloud."""

from __future__ import annotations

import json
import os
import shutil
import stat

import pytest

from lab.infra.cloud import vm as labvm
from lab.infra.cloud.providers import ProviderError, nebius
from test_lab_nebius import FAKE as FAKE_NEBIUS

FAKE_TOOL = r'''#!/usr/bin/env python3
import json, os, sys
open(os.environ["FAKE_LOG"], "a").write(json.dumps([os.path.basename(sys.argv[0])] + sys.argv[1:]) + "\n")
# FAKE_EXIT applies to the user's command only (the one that cd's into the repo), not to ssh probes
sys.exit(int(os.environ.get("FAKE_EXIT", "0")) if any(a.startswith("cd ") for a in sys.argv) else 0)
'''


@pytest.fixture
def fake(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in (("nebius", FAKE_NEBIUS), ("ssh", FAKE_TOOL), ("rsync", FAKE_TOOL)):
        p = bin_dir / name
        p.write_text(body)
        p.chmod(p.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_STATE", str(tmp_path / "state.json"))
    monkeypatch.setenv("FAKE_LOG", str(tmp_path / "calls.log"))
    monkeypatch.delenv("LAB_VM_PROVIDER", raising=False)
    monkeypatch.delenv("LAB_VM_SPOT", raising=False)
    (tmp_path / "key.pub").write_text("ssh-ed25519 AAAA lab\n")
    monkeypatch.setenv("LAB_VM_KEYFILE", str(tmp_path / "key.pub"))
    monkeypatch.setattr(labvm, "KNOWN_HOSTS_DIR", tmp_path / "kh")
    monkeypatch.setattr(labvm, "POLL_S", 0.0)
    monkeypatch.setattr(labvm, "_spawn_watchdog", _fake_spawn)     # the watchdog: recorded, never spawned
    monkeypatch.setattr(labvm, "SSH_TIMEOUT_S", 0.05)
    return tmp_path


SPAWNED: list = []


def _fake_spawn(argv, log):
    SPAWNED.append(argv)
    return 4242


def calls(tmp_path):
    log = tmp_path / "calls.log"
    return [json.loads(l) for l in log.read_text().splitlines()] if log.exists() else []


def test_start_creates_an_absent_vm(fake):
    vm = labvm.VM(name="t1")
    record = labvm.start(vm)
    assert record.state == "running" and record.ip == "203.0.113.9"
    create = next(c for c in calls(fake) if c[:3] == ["compute", "instance", "create"])
    assert create[create.index("--resources-preset") + 1] == vm.provider.type


def kinds(tmp_path):
    return [c[2] for c in calls(tmp_path) if c[:2] == ["compute", "instance"] and c[2] != "list"]


def test_stop_deletes_the_vm_and_its_known_hosts(fake):
    vm = labvm.VM(name="t2")
    labvm.start(vm)
    vm.known_hosts.parent.mkdir(parents=True, exist_ok=True)
    vm.known_hosts.write_text("x")
    labvm.stop(vm)
    assert vm.provider.get("t2") is None and not vm.known_hosts.exists()
    assert kinds(fake) == ["create", "delete"]


def test_stop_keeping_the_disk_stops_and_start_resumes(fake):
    vm = labvm.VM(name="t2k")
    labvm.start(vm)
    labvm.stop(vm, keep_disk=True)
    labvm.stop(vm, keep_disk=True)                     # already stopped: nothing more is asked
    assert vm.provider.get("t2k").state == "stopped"
    labvm.start(vm)
    assert kinds(fake) == ["create", "stop", "start"]


def test_stop_deletes_a_stopped_vm_and_is_a_no_op_when_absent(fake):
    labvm.stop(labvm.VM(name="t3"))
    vm = labvm.VM(name="t3b")
    labvm.start(vm)
    labvm.stop(vm, keep_disk=True)
    labvm.stop(vm)
    labvm.stop(vm)
    assert kinds(fake) == ["create", "stop", "delete"]


def test_cli_stop_deletes_unless_keep_disk(fake, monkeypatch, capsys):
    monkeypatch.setenv("LAB_VM", "t3c")
    labvm.start(labvm.VM())
    assert labvm.main(["stop", "--keep-disk"]) == 0 and "disk is kept" in capsys.readouterr().out
    assert labvm.VM().provider.get("t3c").state == "stopped"
    assert labvm.main(["stop"]) == 0 and "deleted" in capsys.readouterr().out
    assert labvm.VM().provider.get("t3c") is None


def test_run_pushes_runs_in_repo_dir_and_fetches(fake, monkeypatch):
    monkeypatch.setenv("LAB_ENGINE_REPO", str(labvm.engine.ENV_ROOT))     # its sha ships as LAB_GIT_SHA
    vm = labvm.VM(name="t4", user="u", remote_dir="~/repo")
    local = fake / "out"
    assert labvm.run(vm, "python -m lab.profile.profile", fetch_dir="lab/runs", local=local) == 0
    log = calls(fake)
    rsyncs = [c for c in log if c[0] == "rsync"]
    assert rsyncs[0][-1] == "u@203.0.113.9:~/repo/" and "--filter=:- .gitignore" in rsyncs[0]          # the engine
    assert rsyncs[1][-1] == "u@203.0.113.9:~/BlameGraph/" and "--filter=:- .gitignore" in rsyncs[1]    # this environment
    assert rsyncs[2][-2] == "u@203.0.113.9:~/repo/lab/runs/"
    sshs = [c for c in log if c[0] == "ssh"]
    cmd = [c for c in sshs if c[-1].startswith("cd ")][-1][-1]
    assert cmd.startswith("cd ~/repo && LAB_GIT_SHA=")
    assert cmd.endswith('PATH="$HOME/BlameGraph/bin:$HOME/repo/.venv/bin:$PATH" timeout --signal=TERM --kill-after=60 180m python -m lab.profile.profile')
    assert any(a.startswith("UserKnownHostsFile=") and a.endswith("t4.known_hosts") for a in sshs[-1])
    # the VM stays running afterwards with the auto-stop armed
    assert vm.provider.get("t4").state == "running" and labvm._watchdog_file(vm).exists()
    assert any("shutdown -h +180" in c[-1] for c in sshs)
    assert SPAWNED and SPAWNED[-1][-3:-1] == ["watchdog", "--deadline"]


def test_run_in_the_environment_dir(fake, monkeypatch):
    from lab.records import ledger
    monkeypatch.setattr(ledger, "ROOT", fake / "ledger")
    monkeypatch.delenv("LAB_STORE", raising=False)
    vm = labvm.VM(name="t4e", user="u", remote_dir="~/repo", env_dir="~/env")
    assert labvm.run(vm, "lab regimes run all --url http://localhost:8000", fetch_dir="out", local=fake / "o",
                     where="env") == 0
    log = calls(fake)
    cmd = [c for c in log if c[0] == "ssh" and c[-1].startswith("cd ")][-1][-1]
    assert cmd.startswith("cd ~/env && ") and 'PATH="$HOME/env/bin:$HOME/repo/.venv/bin:$PATH" timeout --signal=TERM --kill-after=60 180m lab regimes' in cmd
    rsyncs = [c for c in log if c[0] == "rsync"]
    assert "--include=evals/regimes/*/heldout.jsonl" in rsyncs[1]                     # held-out traces travel
    assert [r[-2] for r in rsyncs[-3:]] == ["u@203.0.113.9:~/env/out/", "u@203.0.113.9:~/env/local/ledger/",
                                            "u@203.0.113.9:~/env/local/runs/"]      # a lab run is always brought home
    assert "--exclude=workspace/" in rsyncs[-1]


def test_collect_merges_the_vms_ledger(fake, monkeypatch):
    from lab.records import ledger
    monkeypatch.setattr(ledger, "ROOT", fake / "ledger")
    monkeypatch.delenv("LAB_STORE", raising=False)
    remote = fake / "vm-ledger"
    rec = ledger.append({"kind": "note", "run": "on-vm"}, remote)
    def fetch(vm, record, path, local, base=None, *extra):
        if path == "local/ledger":
            shutil.copytree(remote, local, dirs_exist_ok=True)
    monkeypatch.setattr(labvm, "fetch", fetch)
    labvm.collect(labvm.VM(name="t4c", user="u"), None)
    assert [r["id"] for r in ledger.records(fake / "ledger")] == [rec["id"]]


def test_run_leaves_the_vm_running_with_the_watchdog_armed(fake):
    vm = labvm.VM(name="t4k", user="u", remote_dir="~/repo")
    assert labvm.run(vm, "true", minutes=45) == 0
    assert vm.provider.get("t4k").state == "running"
    pid, deadline = labvm._watchdog_file(vm).read_text().split()
    assert pid == "4242" and 0 < float(deadline) - labvm.time.time() <= 45 * 60 + 1
    assert any("shutdown -h +45" in c[-1] for c in calls(fake) if c[0] == "ssh")
    assert "45m true" in [c for c in calls(fake) if c[0] == "ssh" and c[-1].startswith("cd ")][-1][-1]


def test_watchdog_deletes_a_running_vm_at_its_deadline(fake, monkeypatch):
    vm = labvm.VM(name="t4w")
    labvm.start(vm)
    assert vm.provider.get("t4w").state == "running"
    labvm.KNOWN_HOSTS_DIR.mkdir(parents=True, exist_ok=True)
    labvm._watchdog_file(vm).write_text(f"{os.getpid()} 0\n")      # the armed file names the watchdog itself
    monkeypatch.setenv("LAB_VM", "t4w")
    assert labvm.watchdog(labvm.time.time() - 1) == 0               # must not SIGTERM itself on the way
    assert vm.provider.get("t4w") is None and not labvm._watchdog_file(vm).exists()


def test_run_deletes_the_vm_when_asked(fake):
    vm = labvm.VM(name="t4d", user="u", remote_dir="~/repo")
    assert labvm.run(vm, "true", keep=False) == 0
    assert vm.provider.get("t4d") is None and not labvm._watchdog_file(vm).exists()     # stop disarms it


def test_run_returns_the_remote_exit_code_and_still_fetches(fake, monkeypatch):
    monkeypatch.setenv("FAKE_EXIT", "3")
    assert labvm.run(labvm.VM(name="t5"), "false", fetch_dir="lab/runs", local=fake / "out") == 3
    assert any(c[0] == "rsync" and c[-2].endswith("lab/runs/") for c in calls(fake))


def test_login_defaults_to_the_provider(fake, monkeypatch):
    assert labvm.VM().login == "lab"
    monkeypatch.setenv("LAB_VM_USER", "me")
    assert labvm.VM().login == "me"


def test_provider_reads_env_at_construction(fake, monkeypatch):
    monkeypatch.setenv("LAB_VM_TYPE", "2gpu-32vcpu-400gb")
    assert nebius.Nebius().type == "2gpu-32vcpu-400gb"


def test_cli_status_types_and_separator(fake, capsys, monkeypatch):
    assert labvm.main(["status"]) == 0
    assert "absent" in capsys.readouterr().out
    assert labvm.main(["types"]) == 0
    assert "gpu-h200-sxm: 1gpu-16vcpu-200gb" in capsys.readouterr().out
    seen = {}
    monkeypatch.setattr(labvm, "run", lambda vm, command, **kw: seen.setdefault("cmd", command) and 0)
    labvm.main(["run", "--", "pytest", "--", "tests"])
    assert seen["cmd"] == "pytest -- tests"


def test_missing_cli_is_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(ProviderError, match="not installed"):
        nebius.Nebius().get("t7")


def test_unknown_provider(monkeypatch):
    monkeypatch.setenv("LAB_VM_PROVIDER", "nimbus")
    with pytest.raises(ProviderError, match="unknown provider"):
        labvm.VM()


def test_ssh_uses_the_private_half_of_the_keyfile(fake, monkeypatch):
    monkeypatch.setenv("LAB_VM_KEYFILE", "~/.ssh/lab_ed25519.pub")
    opts = labvm.VM().ssh_opts
    assert opts[:2] == ["-i", os.path.expanduser("~/.ssh/lab_ed25519")]


def test_rsync_quotes_a_keyfile_path_with_spaces(fake, monkeypatch):
    import shlex
    key = fake / "A B" / "lab key.pub"
    key.parent.mkdir()
    key.write_text("ssh-ed25519 AAAA lab\n")
    monkeypatch.setenv("LAB_VM_KEYFILE", str(key))
    labvm.run(labvm.VM(name="t5"), "true")
    e = next(c for c in calls(fake) if c[0] == "rsync")[3]
    assert shlex.split(e)[1:3] == ["-i", str(fake / "A B" / "lab key")]


def test_run_reports_a_vm_stopped_under_the_command(fake, monkeypatch, capsys):
    vm = labvm.VM(name="t9")
    real_ssh = labvm.ssh

    def preempting_ssh(vm_, record, command, **kw):
        if command.startswith("cd "):                  # the provider takes the VM back mid-command
            vm_.provider.stop(vm_.name)
            return labvm.subprocess.CompletedProcess([], 255)
        return real_ssh(vm_, record, command, **kw)
    monkeypatch.setattr(labvm, "ssh", preempting_ssh)
    assert labvm.run(vm, "true", fetch_dir="lab/runs", local=fake / "out") == 255
    assert "stopped under the command" in capsys.readouterr().err
    assert not any(c[0] == "rsync" and c[-2].endswith("lab/runs/") for c in calls(fake))    # no fetch from a stopped VM
    assert not labvm._watchdog_file(vm).exists()
    assert vm.provider.get("t9").state == "stopped"                  # its disk is kept for the restart


def test_a_vm_that_stopped_under_a_session_is_started_and_set_up_again_under_the_deadline_it_had(fake, monkeypatch):
    monkeypatch.setattr(labvm, "LOST_WAIT_S", 0.0)
    vm = labvm.VM(name="t10")
    labvm.arm(vm, labvm.start(vm), 60)
    assert labvm.restart(vm) is False                                # it is running: the session failed otherwise
    vm.provider.stop(vm.name)
    assert labvm.restart(vm) is True and vm.provider.get("t10").state == "running"
    assert any(c[0] == "ssh" and "vm-setup.sh" in c[-1] for c in calls(fake))
    assert float(labvm._watchdog_file(vm).read_text().split()[1]) <= labvm.time.time() + 60 * 60
    vm.provider.stop(vm.name)
    labvm._watchdog_file(vm).write_text("4242 1\n")                  # its deadline has passed
    assert labvm.restart(vm) is False and vm.provider.get("t10").state == "stopped"


def _nebius(monkeypatch, instances, spot):
    from lab.infra.cloud.providers import nebius
    monkeypatch.setenv("LAB_VM_SPOT", "1" if spot else "0")
    seen = []

    def cli(self, *args):
        seen.append(args)
        if args[:3] == ("compute", "instance", "list"):
            return json.dumps({"items": instances})
        if args[:3] == ("vpc", "subnet", "list"):
            return json.dumps({"items": [{"metadata": {"id": "sub1"}}]})
        return "{}"
    monkeypatch.setattr(nebius.Nebius, "_cli", cli)
    return nebius.Nebius(), seen


def test_nebius_spot_creates_a_preemptible_vm_that_stops(tmp_path, monkeypatch):
    key = tmp_path / "k.pub"
    key.write_text("ssh-ed25519 AAA")
    for spot in (True, False):
        p, seen = _nebius(monkeypatch, [], spot)
        p.create("n1", str(key))
        create = next(a for a in seen if a[:3] == ("compute", "instance", "create"))
        flags = ("--follows-spot-price", "--preemptible-on-preemption", "stop", "--recovery-policy", "fail")
        assert (create[-5:] == flags) is spot and ("--follows-spot-price" in create) is spot   # Nebius: "fail" only


def test_start_refuses_the_other_pricing_running_or_not(monkeypatch):
    for state in ("STOPPED", "RUNNING"):
        inst = {"metadata": {"name": "n2", "id": "i2"}, "spec": {"preemptible": {"on_preemption": "stop"}},
                "status": {"state": state}}
        p, seen = _nebius(monkeypatch, [inst], spot=False)
        assert p.get("n2").detail == f"{state.lower()} spot"
        with pytest.raises(ProviderError, match="created spot"):
            labvm.start(labvm.VM(name="n2", provider=p))
        assert not any(a[:3] == ("compute", "instance", "start") for a in seen)


def test_spot_flag_refuses_other_spellings(monkeypatch):
    from lab.infra.cloud.providers import nebius
    monkeypatch.setenv("LAB_VM_SPOT", "true")
    with pytest.raises(ProviderError, match="use 1 or 0"):
        nebius.Nebius()


def test_a_dropped_session_on_a_running_vm_still_fetches_and_deletes(fake, monkeypatch):
    monkeypatch.setenv("FAKE_EXIT", "255")              # ssh dropped, but the VM never stopped
    monkeypatch.setattr(labvm, "LOST_WAIT_S", 0.0)
    assert labvm.run(labvm.VM(name="t10"), "true", fetch_dir="lab/runs", local=fake / "out", keep=False) == 255
    assert any(c[0] == "rsync" and c[-2].endswith("lab/runs/") for c in calls(fake))
    assert kinds(fake) == ["create", "delete"]


def test_a_provider_error_after_a_dropped_session_still_deletes(fake, monkeypatch):
    vm = labvm.VM(name="t11")
    real_get, n = type(vm.provider).get, {"calls": 0}

    def flaky_get(self, name):
        n["calls"] += 1
        if n["calls"] == 3:                             # the one call _lost makes
            raise ProviderError("api hiccup")
        return real_get(self, name)
    monkeypatch.setenv("FAKE_EXIT", "255")
    monkeypatch.setattr(type(vm.provider), "get", flaky_get)
    assert labvm.run(vm, "true", keep=False) == 255
    assert real_get(vm.provider, "t11") is None
