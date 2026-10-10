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
