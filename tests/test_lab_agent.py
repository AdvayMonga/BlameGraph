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
