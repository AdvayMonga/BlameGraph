"""lab.build: when a tree gets its own venv, the cache, the index-only proxy, and (Linux + docker + internet) a build
room that reaches the package index through the proxy and nothing else."""

from __future__ import annotations

import dataclasses
import socket
import threading
from pathlib import Path

import pytest

from lab import build, engine, target
from lab.safety import container, jail


@pytest.fixture
def deps_target(tmp_path, monkeypatch):
    """The committed target with its engine repo here, holding one dependency file."""
    repo = tmp_path / "engine"
    repo.mkdir()
    (repo / "pyproject.toml").write_text('[project]\nname = "e"\ndependencies = ["a"]\n')
    t = dataclasses.replace(target.load(), engine_repo=repo, deps=("pyproject.toml",))
    monkeypatch.setattr(target, "load", lambda path=None: t)
    monkeypatch.setenv("LAB_WORKER_CACHE", str(tmp_path / "cache"))
    return t, repo


def pristine(tmp_path, repo: Path, change: dict[str, str]) -> Path:
    """A two-commit tree like grader.pristine_tree's: base, then `change`."""
    from lab.safety import grader
    from tests.lab_fixtures import make_repo
    tree = tmp_path / "tree"
    grader.export(make_repo(tmp_path), "HEAD", tree)
    (tree / "pyproject.toml").write_text((repo / "pyproject.toml").read_text())
    grader._commit(tree, "base", init=True)
    for rel, text in change.items():
        (tree / rel).write_text(text)
    grader._commit(tree, "change")
    return tree


def test_unchanged_dependencies_keep_the_engines_venv_and_changed_ones_build_once(deps_target, tmp_path, monkeypatch):
    t, repo = deps_target
    tree = pristine(tmp_path, repo, {"src/inference_server/engine.py": "def speed():\n    return 2\n"})
    assert not build.changed(tree) and build.python_for(tree) == engine.python()
    flat = tmp_path / "flat"                                      # an exported base: no history, no change
    flat.mkdir()
    (flat / "pyproject.toml").write_text("anything")
    assert not build.changed(flat)
    built = []

    def fake(tree, venv):
        built.append(venv)
        (venv / "env" / "bin").mkdir(parents=True)
        (venv / ".lab-built").write_text("")
    monkeypatch.setattr(build, "build", fake)
    (tree / "pyproject.toml").write_text('[project]\nname = "e"\ndependencies = ["a", "b"]\n')
    assert build.changed(tree)
    py = build.python_for(tree)
    assert py == str(built[0] / "env" / "bin" / "python") and build.python_for(tree) == py and len(built) == 1


def test_the_allowed_hosts_are_the_index_defaults_plus_what_the_files_and_build_name(deps_target, tmp_path):
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "pyproject.toml").write_text('[[tool.uv.index]]\nurl = "https://download.pytorch.org/whl/cu130"\n')
    got = build.hosts(tree, "uv sync --index-url https://mirror.example.org/simple")
    assert got == {"pypi.org", "files.pythonhosted.org", "download.pytorch.org", "mirror.example.org"}


def proxy_request(sock_path: Path, line: bytes) -> bytes:
    c = socket.socket(socket.AF_UNIX)
    c.connect(str(sock_path))
    c.sendall(line)
    out = c.recv(4096)
    return c, out


def test_the_index_proxy_tunnels_to_allowed_hosts_only(tmp_path, monkeypatch):
    echo = socket.socket()
    echo.bind(("127.0.0.1", 0))
    echo.listen()

    def serve():
        conn, _ = echo.accept()
        conn.sendall(conn.recv(100).upper())
        conn.close()
    threading.Thread(target=serve, daemon=True).start()
    real = socket.create_connection
    monkeypatch.setattr(build.socket, "create_connection",
                        lambda addr, timeout=None: real(echo.getsockname(), timeout=timeout))   # "pypi.org:443" here
    sock = Path("/tmp") / f"lab-test-{id(tmp_path)}.sock"
    with build.IndexProxy({"pypi.org"}, sock):
        c, head = proxy_request(sock, b"CONNECT pypi.org:443 HTTP/1.1\r\nHost: pypi.org\r\n\r\n")
        assert head.startswith(b"HTTP/1.1 200")
        c.sendall(b"hello")
        assert c.recv(100) == b"HELLO"
        for line in (b"CONNECT example.com:443 HTTP/1.1\r\n\r\n", b"CONNECT pypi.org:80 HTTP/1.1\r\n\r\n",
                     b"GET http://pypi.org/ HTTP/1.1\r\n\r\n"):
            assert proxy_request(sock, line)[1].startswith(b"HTTP/1.1 403")
    sock.unlink(missing_ok=True)


def _online() -> bool:
    try:
        socket.create_connection(("pypi.org", 443), timeout=5).close()
        return True
    except OSError:
        return False


PROBE = r'''import os, socket, urllib.request, urllib.error
def get(url):
    try:
        return urllib.request.urlopen(url, timeout=20).status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception as e:                       # a refused tunnel is a URLError carrying the proxy's 403
        return "403" if "403" in str(e) else type(e).__name__
direct = "no"
try:
    socket.create_connection(("1.1.1.1", 443), timeout=5); direct = "yes"
except OSError:
    pass
print("index", get("https://pypi.org/simple/pip/"), "other", get("https://example.com/"), "direct", direct)
'''


@pytest.mark.skipif(not (jail.backend() == "container" and container.available()), reason="needs the container jail")
@pytest.mark.skipif(not _online(), reason="needs the internet")
def test_a_build_room_reaches_the_index_through_the_proxy_and_nothing_else(deps_target, tmp_path, monkeypatch):
    t, repo = deps_target
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "pyproject.toml").write_text('[project]\nname = "e"\ndependencies = ["b"]\n')
    (tree / "probe.py").write_text(PROBE)
    out = tmp_path / "probe.out"
    out.touch()
    cmd = (f'python3 -c "$(cat {tree}/probe.py)" > "$(dirname "$UV_PROJECT_ENVIRONMENT")/probe.out" 2>&1; '
           'mkdir -p "$UV_PROJECT_ENVIRONMENT/bin" && ln -s /usr/bin/python3 "$UV_PROJECT_ENVIRONMENT/bin/python"')
    monkeypatch.setattr(target, "load", lambda path=None: dataclasses.replace(t, build=cmd))
    real = container.argv
    monkeypatch.setattr(container, "argv", lambda *a, **k: real(*a[:5], [*a[5], tree], *a[6:], **k))  # probe readable
    venv = build.cache_dir() / "probe"
    build.build(tree, venv)
    said = (venv / "probe.out").read_text()
    assert said.split() == ["index", "200", "other", "403", "direct", "no"], said
