"""lab.agent's CLI wrapper: the model credential never enters the jail; the CLI is pointed at the API proxy."""

from __future__ import annotations

from pathlib import Path

import pytest

from lab import agent
from lab.safety import container


def spec(tmp_path: Path) -> agent.AgentSpec:
    (tmp_path / "ws").mkdir()
    return agent.AgentSpec(system="", prompt="", workspace=tmp_path / "ws", scratch=tmp_path / "scratch", tools=[],
                           workbench="lab-workbench-r1-s1")


@pytest.mark.parametrize("containerized", [True, False])
def test_the_wrapper_holds_no_credential_and_points_at_the_proxy(tmp_path, monkeypatch, containerized):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret-key")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "oat-secret-token")
    monkeypatch.setenv("LAB_NO_JAIL", "1")
    monkeypatch.setattr(agent.ClaudeAgentSDK, "containerized", staticmethod(lambda: containerized))
    monkeypatch.setattr(container, "_docker", lambda: "docker")
    monkeypatch.setattr(agent.jail, "available", lambda: False)
    s = spec(tmp_path)
    s.scratch.mkdir()
    script = agent.ClaudeAgentSDK()._wrapper(s, tmp_path / "bin" / "claude", "http://172.17.0.1:4321").read_text()
    assert "sk-secret" not in script and "oat-secret" not in script
    assert "ANTHROPIC_BASE_URL=http://172.17.0.1:4321" in script and f"ANTHROPIC_API_KEY={agent.apiproxy.PLACEHOLDER}" in script
    if containerized:
        assert "docker run -i" in script and "--name lab-workbench-r1-s1" in script
        assert "--env CLAUDE_CODE_ENTRYPOINT" in script            # the SDK's value, passed through by name
    else:
        assert script.splitlines()[1].startswith("exec env -i")


def test_a_remote_workbench_is_launched_over_ssh_with_the_proxy_tunnelled_and_args_intact(tmp_path, monkeypatch):
    import shlex
    import subprocess

    class FakeRemote:
        prefix = ["ssh", "-i", "key", "root@203.0.113.5"]

        def workbench(self, name, session, workspace, version):
            return {"argv": ["docker", "run", "-i", "--name", name, "lab-jail", "sh", "-c", 'exec "$0" "$@"', "/c/claude"],
                    "sock": f"/tmp/lab-proxy-{session}/proxy.sock"}

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret-key")
    import sys
    import types
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", types.SimpleNamespace(__version__="9.9.9"))
    s = spec(tmp_path)
    s.scratch.mkdir()
    s.remote, s.remote_workspace = FakeRemote(), "/root/BlameGraph/lab/runs/remote/r1/workspace"
    script = agent.ClaudeAgentSDK()._wrapper(s, Path("/unused"), "http://127.0.0.1:5555")
    text = script.read_text()
    assert "sk-secret" not in text
    assert "-R /tmp/lab-proxy-scratch/proxy.sock:127.0.0.1:5555 root@203.0.113.5" in text
    fake = tmp_path / "bin" / "ssh"                            # an ssh that prints the remote command it was given
    fake.parent.mkdir()
    fake.write_text('#!/bin/sh\nfor a; do last="$a"; done\nprintf "%s\\n" "$last"\n')
    fake.chmod(0o755)
    out = subprocess.run(["bash", str(script), "--output-format", "stream-json", "it's \"quoted\" $HOME"],
                         capture_output=True, text=True, env={"PATH": f"{fake.parent}:/usr/bin:/bin"})
    first, launch = out.stdout.splitlines()
    assert first == "rm -f /tmp/lab-proxy-scratch/proxy.sock"           # a stale socket from an earlier launch goes
    remote_argv = shlex.split(launch)
    assert remote_argv == ["docker", "run", "-i", "--name", "lab-workbench-r1-s1", "lab-jail", "sh", "-c",
                           'exec "$0" "$@"', "/c/claude", "--output-format", "stream-json", "it's \"quoted\" $HOME"]
