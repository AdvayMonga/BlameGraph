"""The OpenAI-compatible provider against a scripted fake API: tools, write guard, finish, cost, URL and price checks."""
from __future__ import annotations

import json

import pytest

from lab import agent
from lab.agent import AgentSpec, ToolSpec
from lab import agent_openai
from lab.agent_openai import OpenAICompatible, base_url, prices

URL = "https://api.example.com/v1"


def call(name, args, i=0):
    return {"id": f"c{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


class FakeAPI:
    """Returns one scripted assistant turn per request and keeps every request body."""

    def __init__(self, turns, usage=None):
        self.turns, self.bodies = list(turns), []
        self.usage = usage if usage is not None else {"prompt_tokens": 1000, "completion_tokens": 100}

    def __call__(self, url, key, body, timeout):
        assert url == URL + "/chat/completions" and key == "k"
        self.bodies.append(json.loads(json.dumps(body)))
        msg = self.turns.pop(0)
        if isinstance(msg, list):
            msg = {"role": "assistant", "content": None, "tool_calls": msg}
        return {"choices": [{"message": msg}], **({"usage": self.usage} if self.usage else {})}


@pytest.fixture
def spec(tmp_path, monkeypatch):
    monkeypatch.setenv("LAB_NO_JAIL", "1")
    ws = tmp_path / "ws"
    (ws / "src/inference_server").mkdir(parents=True)
    (ws / "src/inference_server/engine.py").write_text("BATCH = 8\n")
    (ws / "README.md").write_text("readme\n")
    seen = []
    ping = ToolSpec("ping", "echo", {"type": "object", "properties": {"x": {"type": "string"}}},
                    lambda a: seen.append(a) or f"pong {a.get('x')}")
    s = AgentSpec(system="sys", prompt="go", workspace=ws, scratch=tmp_path / "scratch", tools=[ping],
                  base_files=frozenset({"src/inference_server/engine.py", "README.md"}), max_budget_usd=1.0)
    s.seen = seen
    return s


def provider(api):
    return OpenAICompatible(url=URL, api_key="k", price=(2.0, 8.0), post=api)


def test_a_session_uses_the_tools_and_finishes(spec):
    api = FakeAPI([
        [call("Read", {"file_path": "src/inference_server/engine.py"}, 0),
         call("Edit", {"file_path": "src/inference_server/engine.py", "old_string": "8", "new_string": "16"}, 1)],
        [call("Write", {"file_path": "README.md", "content": "x"}, 2),
         call("Read", {"file_path": "../outside.txt"}, 3),
         call("Bash", {"command": "cat src/inference_server/engine.py"}, 4),
         call("Grep", {"pattern": "BATCH", "glob": "src/*"}, 5),
         call("ping", {"x": "a"}, 6)],
        {"role": "assistant", "content": "done"},                        # no tool call: told how a session ends
        [call("finish", {"status": "stop", "note": "batch 16"}, 7)],
    ])
    reply = provider(api).run(spec)
    assert reply.output == {"status": "stop", "note": "batch 16"} and reply.error is None and reply.turns == 4
    assert reply.cost_usd == pytest.approx(4 * (1000 * 2.0 + 100 * 8.0) / 1e6) and not reply.cost_estimated
    assert (spec.workspace / "src/inference_server/engine.py").read_text() == "BATCH = 16\n"
    assert (spec.workspace / "README.md").read_text() == "readme\n"           # outside the write surface
    assert spec.seen == [{"x": "a"}]
    results = {m["tool_call_id"]: m["content"] for m in api.bodies[2]["messages"] if m["role"] == "tool"}
    assert results["c0"] == "1\tBATCH = 8"
    assert results["c2"].startswith("Write refused") and results["c3"].startswith("Read refused")
    assert results["c4"] == "exit 0\nBATCH = 16\n"
    assert results["c5"] == "src/inference_server/engine.py:1:BATCH = 16" and results["c6"] == "pong a"
    assert api.bodies[3]["messages"][-1] == {"role": "user", "content": "A session ends only through the `finish` tool."}
    names = [t["function"]["name"] for t in api.bodies[0]["tools"]]
    assert names == [*(n for n in agent.BUILTIN_TOOLS if n in agent_openai.BUILTIN), "finish", "ping"]


def test_a_builtin_the_spec_leaves_out_is_refused(spec):
    spec.builtin = ["Read"]
    api = FakeAPI([[call("Bash", {"command": "true"})], [call("finish", {"status": "continue", "note": None})]])
    assert provider(api).run(spec).output == {"status": "continue", "note": None}
    assert api.bodies[1]["messages"][-1]["content"] == "Bash refused: no such tool"


def test_the_session_cap_ends_the_session(spec):
    api = FakeAPI([[call("ping", {"x": "a"})]] * 3, usage={"prompt_tokens": 300_000, "completion_tokens": 0})
    reply = provider(api).run(spec)
    assert reply.output is None and "budget" in reply.error and reply.turns == 2
    assert reply.cost_usd == pytest.approx(1.2)


def test_a_response_without_usage_is_charged_at_the_cap(spec):
    api = FakeAPI([[call("ping", {})]], usage={})
    reply = provider(api).run(spec)
    assert reply.cost_usd == 1.0 and reply.cost_estimated and reply.output is None


def test_a_failing_api_is_a_reply_not_a_crash(spec):
    def broken(url, key, body, timeout):
        raise RuntimeError("401 from the model API")
    reply = provider(broken).run(spec)
    assert reply.output is None and "401" in reply.error and reply.cost_usd == 0.0


@pytest.mark.parametrize("url", ["http://api.openai.com/v1", "https://localhost:4000", "https://127.0.0.1/v1",
                                 "https://10.0.0.5/v1", "https://[::1]/v1", "https://gw.internal/v1", "not a url"])
def test_only_public_https_apis(url):
    with pytest.raises(ValueError):
        base_url(url)


def test_the_defaults_and_required_settings(monkeypatch):
    assert base_url("") == "https://api.openai.com/v1"
    assert base_url("https://generativelanguage.googleapis.com/v1beta/openai/") == \
        "https://generativelanguage.googleapis.com/v1beta/openai"
    assert prices("1.25, 10") == (1.25, 10.0)
    for bad in ("", "1", "a,b", "-1,2"):
        with pytest.raises(ValueError):
            prices(bad)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("LAB_AGENT_PROVIDER", "openai")
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        agent.load()
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.delenv("LAB_MODEL_PRICE", raising=False)
    with pytest.raises(ValueError, match="LAB_MODEL_PRICE"):
        agent.load()
    monkeypatch.setenv("LAB_MODEL_PRICE", "1,2")
    assert agent.load().name == "openai"


def test_grep_never_follows_a_link_out_of_the_workspace(tmp_path):
    from lab.agent_openai import Workbench
    ws, secret = tmp_path / "ws", tmp_path / ".env"
    ws.mkdir()
    secret.write_text("OPENAI_API_KEY=sk-secret\n")
    (ws / "a.py").write_text("x = 1\n")
    (ws / "leak").symlink_to(secret)
    (ws / "dir").symlink_to(tmp_path, target_is_directory=True)
    out = Workbench(ws, set()).Grep(".")
    assert "sk-secret" not in out and "a.py:1:x = 1" in out
