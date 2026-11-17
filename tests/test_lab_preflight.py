"""What `lab session` checks before it starts anything, and the .env the CLI loads."""
import os

os.environ.setdefault("LAB_TARGET", "evals/targets/inference-server.toml")

from lab import agent, cli, session


def test_env_file_fills_only_unset_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("KEPT", "mine")
    monkeypatch.delenv("FILLED", raising=False)
    (tmp_path / ".env").write_text("# a comment\nFILLED='from file'\n\nKEPT=theirs\nnot a pair\n")
    cli.load_env(tmp_path / ".env")
    assert os.environ["FILLED"] == "from file" and os.environ["KEPT"] == "mine" and "not a pair" not in os.environ
    monkeypatch.delenv("FILLED")
    cli.load_env(tmp_path / "absent")                                    # no file: nothing happens


def test_preflight_names_everything_absent(tmp_path, monkeypatch):
    for k in ("ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN", "LAB_STORE", "OPENAI_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(agent.importlib.util, "find_spec", lambda name: None)
    absent = session.preflight("claude")
    assert len(absent) == 3 and "Claude Agent SDK is not installed" in absent[0]
    assert "ANTHROPIC_API_KEY" in absent[1] and "LAB_STORE" in absent[2]
    assert session.preflight("openai") == ["OPENAI_API_KEY is not set", absent[2]]
    assert session.main(["--goal", "g", "--budget", "1"]) == 2             # nothing is started
    monkeypatch.setattr(agent.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k"); monkeypatch.setenv("LAB_STORE", str(tmp_path))
    assert session.preflight("claude") == []
