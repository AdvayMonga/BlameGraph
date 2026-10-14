"""The container jail: the docker argv everywhere; isolation and the port bridge where docker and the image exist."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import textwrap
import time
import urllib.request
from pathlib import Path

import pytest

from lab.safety import container, grader, jail
from tests.lab_fixtures import make_repo


@pytest.fixture
def docker_named(monkeypatch):
    monkeypatch.setattr(container, "_docker", lambda: "docker")


def test_argv_has_no_network_no_caps_and_a_non_root_user(docker_named, tmp_path, monkeypatch):
    monkeypatch.setattr(container.os, "getuid", lambda: 0)
    a = container.argv("n", tmp_path, ["true"], {"K": "v"}, [tmp_path], [], sys.executable)
    s = " ".join(a)
    for flag in ("--network none", "--read-only", "--cap-drop ALL", "--security-opt no-new-privileges",
                 f"--user {container.JAIL_UID}:{container.JAIL_UID}", "--env K=v", "--rm"):
        assert flag in s
    assert a[-1] == "true" and a[-2] == container.IMAGE


def test_system_prefixes_are_never_mounted_over_the_image(tmp_path):
    m = " ".join(container.mounts([tmp_path], [Path("/usr"), tmp_path / "ro"], "/usr/bin/python3"))
    assert "dst=/usr," not in m and "dst=/usr " not in m and not m.endswith("dst=/usr")


def test_readonly_mounts_are_readonly(tmp_path):
    (tmp_path / "ro").mkdir()
    m = container.mounts([tmp_path / "w"], [tmp_path / "ro"], "/usr/bin/python3")
    ro = [x for x in m if f"dst={(tmp_path / 'ro').resolve()}" in x]
    assert ro and ro[0].endswith(",readonly")


def test_bridge_wraps_the_command_and_keeps_it_last(docker_named, tmp_path):
    a = container.argv("n", tmp_path, ["python", "-m", "srv"], {}, [tmp_path / "t"], [], sys.executable,
                       bridge_port=8123)
    assert a[-3:] == ["python", "-m", "srv"] and "TCP:127.0.0.1:8123" in a[a.index(container.IMAGE) + 3]


def test_linux_defaults_to_the_container_jail(monkeypatch):
    monkeypatch.delenv("LAB_JAIL", raising=False)
    monkeypatch.setattr(jail.sys, "platform", "linux")
    assert jail.backend() == "container"
    monkeypatch.setattr(jail.sys, "platform", "darwin")
    assert jail.backend() == "srt"


live = pytest.mark.skipif(not (jail.backend() == "container" and container.available()),
                          reason="needs the container jail: Linux, docker and the lab-jail image")

PROBE = textwrap.dedent("""
    import os, socket, sys
    out = {}
    out["uid"] = os.getuid()
    out["caps"] = [l.split()[1] for l in open("/proc/self/status") if l.startswith("CapEff")][0]
    open("written_by_jail", "w").write("x")
    def tries(f):
        try:
            f(); return "yes"
        except Exception:
            return "no"
    out["write_root"] = tries(lambda: open("/etc/x", "w"))
    out["read_secret"] = tries(lambda: open(sys.argv[1]).read())
    out["net"] = tries(lambda: socket.create_connection(("1.1.1.1", 53), timeout=2))
    out["host_port"] = tries(lambda: socket.create_connection(("127.0.0.1", int(sys.argv[2])), timeout=2))
    print(out)
""")


@live
def test_jail_isolates_files_network_and_privileges(tmp_path, monkeypatch):
    monkeypatch.delenv("LAB_NO_JAIL", raising=False)
    tree = tmp_path / "t"
    grader.export(make_repo(tmp_path), "HEAD", tree)
    (tree / "probe.py").write_text(PROBE)
    secret = tmp_path / "secret"
    secret.write_text("key")
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen()
    proc = grader.jailed(tree, ["python3", "probe.py", str(secret), str(srv.getsockname()[1])], timeout_s=60)
    srv.close()
    assert proc.returncode == 0, proc.stderr
    out = eval(proc.stdout.strip().splitlines()[-1])
    assert out["uid"] != 0 and int(out["caps"], 16) == 0
    assert (tree / "written_by_jail").exists()
    assert out["write_root"] == "no" and out["read_secret"] == "no"
    assert out["net"] == "no" and out["host_port"] == "no"


@live
def test_timeout_removes_the_container(tmp_path, monkeypatch):
    monkeypatch.delenv("LAB_NO_JAIL", raising=False)
    tree = tmp_path / "t"
    grader.export(make_repo(tmp_path), "HEAD", tree)
    proc = grader.jailed(tree, ["sleep", "60"], timeout_s=3)
    assert proc.returncode == -9
    left = subprocess.run(["docker", "ps", "-q", "--filter", "label=lab.jail=1"], capture_output=True, text=True)
    assert left.stdout.strip() == ""


@live
def test_served_port_reaches_the_host_and_nothing_else(tmp_path, monkeypatch):
    from lab import serve
    monkeypatch.delenv("LAB_NO_JAIL", raising=False)
    tree = tmp_path / "t"
    grader.export(make_repo(tmp_path), "HEAD", tree)
    (tree / "index.html").write_text("ok")
    port = serve.free_port()
    proc, tmp, stop, pids = grader.jailed_popen(tree, ["python3", "-m", "http.server", str(port), "--bind", "127.0.0.1"],
                                          port=port)
    try:
        body, deadline = None, time.monotonic() + 30
        while body is None and time.monotonic() < deadline:
            try:
                body = urllib.request.urlopen(f"http://127.0.0.1:{port}/index.html", timeout=2).read()
            except OSError:
                time.sleep(0.3)
        assert body == b"ok" and pids()                 # the jail reports its processes as host pids
        (tree / "probe.py").write_text(PROBE)       # another jail cannot reach the served engine
        other = grader.jailed(tree, ["python3", "probe.py", "/nonexistent", str(port)], timeout_s=60)
        assert eval(other.stdout.strip().splitlines()[-1])["host_port"] == "no"
    finally:
        os.killpg(proc.pid, 15)
        proc.wait(timeout=30)
        stop()
    left = subprocess.run(["docker", "ps", "-q", "--filter", "label=lab.jail=1"], capture_output=True, text=True)
    assert left.stdout.strip() == ""
    with pytest.raises(OSError):
        socket.create_connection(("127.0.0.1", port), timeout=2)


def test_the_workbench_has_network_gpu_and_root_inside(docker_named, tmp_path, monkeypatch):
    monkeypatch.setattr(container.os, "getuid", lambda: 0)
    monkeypatch.setattr(container, "gpus", lambda: True)
    a = container.workbench_argv("wb", tmp_path, ["claude"], {"K": "v"}, [tmp_path], [], sys.executable)
    s = " ".join(a)
    assert "--user 0:0" in s and "--gpus all" in s and "-i" in a and a[-1] == "claude"
    for absent in ("--network", "--read-only", "--cap-drop"):
        assert absent not in s


def test_quiet_kills_the_workbenchs_gpu_holders_then_pauses_and_thaws(monkeypatch):
    from lab import gpu
    calls, killed = [], []
    monkeypatch.setattr(container, "_docker", lambda: "docker")
    monkeypatch.setattr(container, "running", lambda n: True)
    monkeypatch.setattr(container, "pids", lambda n: {101, 102})
    apps = [[{"pid": 101, "used_mib": 9000, "name": "python"}, {"pid": 999, "used_mib": 1, "name": "other"}]]
    monkeypatch.setattr(gpu, "compute_apps", lambda: apps[0])
    monkeypatch.setattr(container.os, "kill", lambda pid, sig: killed.append(pid) or apps.__setitem__(0, [apps[0][1]]))
    monkeypatch.setattr(container.subprocess, "run", lambda argv, **k: calls.append(argv[1:]) or
                        subprocess.CompletedProcess(argv, 0, "false", ""))
    states = lambda: [c for c in calls if c[0] in ("pause", "unpause")]     # noqa: E731
    with container.quiet("wb") as got:
        assert states() == [["pause", "wb"]]
    assert killed == [101] and [a["pid"] for a in got] == [101]          # 999 is not the workbench's
    assert states() == [["pause", "wb"], ["unpause", "wb"]]


def test_an_already_paused_workbench_is_left_paused(monkeypatch):
    calls = []
    monkeypatch.setattr(container, "_docker", lambda: "docker")
    monkeypatch.setattr(container, "running", lambda n: True)
    monkeypatch.setattr(container.subprocess, "run", lambda argv, **k: calls.append(argv[1]) or
                        subprocess.CompletedProcess(argv, 0, "true", ""))
    with container.paused("wb"):
        with container.paused("wb"):
            pass
    assert "pause" not in calls and "unpause" not in calls


def test_quiet_is_a_no_op_without_a_running_workbench(monkeypatch):
    monkeypatch.setattr(container, "_docker", lambda: "docker")
    monkeypatch.setattr(container, "running", lambda n: False)
    with container.quiet("wb") as got:
        assert got == []
    with container.quiet(None) as got:
        assert got == []


@live
def test_a_paused_workbench_is_frozen_and_thawed_after(tmp_path):
    name = container.new_name()
    subprocess.run(["docker", "run", "-d", "--rm", "--name", name, container.IMAGE, "sleep", "60"], check=True,
                   capture_output=True)
    state = lambda: subprocess.run(["docker", "inspect", "-f", "{{.State.Paused}}", name], capture_output=True,  # noqa: E731
                                   text=True).stdout.strip()
    try:
        with container.quiet(name):
            assert state() == "true"
        assert state() == "false"
    finally:
        container.remove(name)


@live
def test_the_workbench_is_root_with_writable_system_dirs_and_reaches_the_host_gateway(tmp_path):
    gw = container.bridge_gateway()
    srv = socket.socket()
    srv.bind((gw, 0))
    srv.listen()
    port = srv.getsockname()[1]
    probe = ("import os, socket; open('/usr/local/x', 'w').write('y'); "
             f"socket.create_connection(('{gw}', {port}), timeout=5); print(os.getuid())")
    argv = container.workbench_argv(container.new_name(), tmp_path, ["python3", "-c", probe], {}, [tmp_path], [],
                                    sys.executable)
    out = subprocess.run(argv, capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL)
    srv.close()
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == ("0" if os.getuid() == 0 else str(os.getuid()))
