"""The dashboard's progress file and its view: a run writes what it is doing, the view reads only files, and the held-out
split's load numbers never reach the file."""
from __future__ import annotations

import json

import pytest

from lab import dashboard, evaltools, ledger, live, session, target
from lab.agent import AgentReply
from regimes import suite
from regimes.client import Row
from tests.test_lab_runtime import ScriptedProvider

pytest_plugins = ["tests.test_lab_runtime"]     # its `cfg` fixture: a run on a tiny repo


@pytest.fixture(autouse=True)
def no_live(monkeypatch):
    monkeypatch.setattr(live, "path", None)     # restored after each test: session.run sets it


def lines(p):
    return [json.loads(x) for x in p.read_text().splitlines()]


def test_emit_writes_only_while_a_run_is_going(tmp_path):
    live.emit("agent", kind="text")
    assert not list(tmp_path.iterdir())
    live.path = tmp_path / "live.jsonl"
    live.emit("agent", kind="text", detail="hi")
    (e,) = lines(live.path)
    assert e["event"] == "agent" and e["detail"] == "hi" and e["ts"] > 0


def test_a_run_writes_progress_and_the_view_reads_it(cfg):
    seen = []

    def script(n, ws):
        yield ("call", "test")
        live.emit("agent", kind="tool", name="bench", detail="{}", spent_usd=0.25)
        seen.append(dashboard.state(None, cfg.ledger_root, cfg.runs_dir))
        yield ("reply", AgentReply({"status": "stop", "note": None}, 0.5, 3, None))
    out = session.run(cfg, ScriptedProvider(script))

    mid = seen[0]
    assert mid["run"] == out["run"] and mid["status"] == "running"
    assert mid["now_doing"]["agent"]["what"] == "running bench" and mid["now_doing"]["jobs"] == []
    assert mid["session"]["n"] == 1 and not mid["session"]["ended"] and mid["budget"]["model_usd"] == 0.25

    end = dashboard.state(out["run"], cfg.ledger_root, cfg.runs_dir)
    assert end["status"] == "ended: agent" and end["session"]["ended"]
    assert end["budget"] == {"cap_usd": 2.0, "model_usd": 0.5, "gpu_usd": 0.0}     # the ledger's figure, not the live one
    assert [m["kind"] for m in end["measurements"]] == ["test"] and end["measurements"][0]["tests"] is True
    evs = lines(cfg.runs_dir / out["run"] / "live.jsonl")
    assert [e["event"] for e in evs if e["event"] != "agent"] == ["run", "session", "job", "job", "run_end"]


def test_load_progress_is_written_for_the_seen_split_only(tmp_path, monkeypatch):
    def fake(ctx):
        ctx._keep([Row("a", "ok", 0.0, 0.0, ttft_s=0.1, tpot_s=0.01), Row("b", "error", 0.0, 0.0)], 0.0)
        return {}
    monkeypatch.setitem(suite.REGIMES, "fake", fake)
    live.path = tmp_path / "live.jsonl"
    evaltools.run_regimes("http://127.0.0.1:1", target.load(), ["fake"], "heldout", "short", 0)
    assert not live.path.exists()
    evaltools.run_regimes("http://127.0.0.1:1", target.load(), ["fake"], "seen", "short", 0)
    regime, probe = lines(live.path)
    assert regime["regime"] == "fake"
    assert probe["n"] == 2 and probe["ok"] == 1 and probe["ttft_p99_s"] == 0.1 and probe["tpot_p50_s"] == 0.01
    now = dashboard._now([regime, probe])
    assert now["load"]["regime"] == "fake" and len(now["load"]["probes"]) == 1


def test_runs_are_listed_newest_first_and_an_empty_ledger_is_fine(tmp_path):
    assert dashboard.state(None, tmp_path / "ledger", tmp_path / "runs")["runs"] == []
    for run in ("old", "new"):
        ledger.append({"kind": "note", "run": run, "result": "x"}, tmp_path / "ledger")
    (tmp_path / "runs" / "newest").mkdir(parents=True)
    (tmp_path / "runs" / "newest" / "live.jsonl").write_text("")
    ids = [r["id"] for r in dashboard.state(None, tmp_path / "ledger", tmp_path / "runs")["runs"]]
    assert ids[0] == "newest" and set(ids) == {"old", "new", "newest"}
