"""Researcher experiments: a change on a branch measured through the lab's own tools, with the same hypothesis,
expect and reflection rules as the agent's, and hidden from every agent view. Fake instruments from the proftools
tests."""
from __future__ import annotations

import json
import subprocess

import pytest

from lab import experiment, experiments, ledger
pytest_plugins = ["tests.test_lab_proftools"]   # its `tb` fixture: an agent Toolbox with fake instruments

EXPECT = {"metric": "kernel_ms", "value": 1.0, "unit": "absolute", "why": "half the launches"}


def commit(repo, files, branch="change"):
    """`files` committed on `branch` off HEAD; HEAD stays where it was (the agent's workspace is exported from it)."""
    git = ["git", "-C", str(repo), "-c", "user.name=Res Earcher", "-c", "user.email=r@r", "-c", "commit.gpgsign=false"]
    subprocess.run(git + ["checkout", "-q", "-b", branch], check=True)
    for rel, text in files.items():
        (repo / rel).write_text(text)
    subprocess.run(git + ["add", "-A"], check=True)
    subprocess.run(git + ["commit", "-q", "-m", "change"], check=True)
    subprocess.run(git + ["checkout", "-q", "-"], check=True)


@pytest.fixture
def run(tb, tmp_path):
    repo = tb.s.workspace.repo
    commit(repo, {"src/inference_server/engine.py": "def speed():\n    return 2\n",
                  "tests/test_engine.py": "def test_speed():\n    pass\n", "README.md": "# changed\n"})
    return experiment.start(repo, "HEAD", "change", 10.0, tb.s.ledger_root, tmp_path / "runs")


def test_start_takes_the_change_inside_the_write_surface_and_lists_the_rest(tb, tmp_path, run):
    assert experiments.researcher(run)
    rec = next(ledger.records(tb.s.ledger_root, run=run, tool="start"))
    assert rec["result"]["applied"] == ["src/inference_server/engine.py"]
    assert rec["result"]["left_out"] == ["README.md", "tests/test_engine.py"]
    rb = experiment.toolbox(run, tb.s.ledger_root, tmp_path / "runs")
    assert "return 2" in (rb.s.workspace.path / "src/inference_server/engine.py").read_text()
    assert (rb.s.workspace.path / "README.md").read_text() == "# tiny\n"
    assert rb.s.budget.cap_usd == 10.0 and rec["snapshot"] != "base" and rec["result"]["author"] == "Res Earcher"


def test_a_researcher_measurement_follows_the_agents_rules(tb, tmp_path, run):
    rb = experiment.toolbox(run, tb.s.ledger_root, tmp_path / "runs")
    with pytest.raises(ValueError, match="hypothesis_id"):
        experiment.call(rb, "trace", {"expect": EXPECT})
    h = experiment.call(rb, "hypothesis", {"hypothesis": "launch-bound decode", "rationale": "4 launches per step"}).split()[1]
    out = experiment.call(rb, "trace", {"hypothesis_id": h, "expect": EXPECT})
    rec = list(ledger.records(tb.s.ledger_root, run=run, tool="trace"))[-1]
    assert rec["experiment"]["hypothesis"] == h and f"Experiment record {rec['id']}" in out
    assert rec["cost"]["usd"] > 0 and rb.s.budget.spent_usd == pytest.approx(rec["cost"]["usd"])
    with pytest.raises(ValueError, match="no reflection"):
        experiment.call(rb, "trace", {"hypothesis_id": h, "expect": EXPECT})
    experiment.call(rb, "reflect", {"record": rec["id"], "reflection": "trace reports no scalar kernel_ms"})
    assert experiments.pending(tb.s.ledger_root, run) is None


def test_no_agent_view_shows_a_researcher_run(tb, tmp_path, run):
    from store import index
    rb = experiment.toolbox(run, tb.s.ledger_root, tmp_path / "runs")
    experiment.call(rb, "hypothesis", {"hypothesis": "a person's belief", "rationale": "r"})
    tb.note({"text": "the agent's own"})
    seen = experiment.call(tb, "ledger", {"last": 100})
    assert "the agent's own" in seen and run not in seen and "a person's belief" not in seen
    assert "a person's belief" in experiment.call(rb, "ledger", {"last": 100})
    sql = "SELECT count(*) FROM records WHERE run LIKE 'researcher-%'"
    assert json.loads(experiment.call(tb, "query", {"sql": sql}))["rows"] == [[0]]
    db = tb.s.ledger_root / "index" / "all.duckdb"
    index.build(tb.s.ledger_root, db)
    assert index.query(db, sql)["rows"][0][0] == 2          # the researcher's index has the start and the hypothesis


def test_the_cli_runs_and_prints_refusals(tb, tmp_path, monkeypatch, capsys):
    repo = tb.s.workspace.repo
    commit(repo, {"src/inference_server/engine.py": "def speed():\n    return 3\n"}, "other")
    monkeypatch.setattr(ledger, "ROOT", tb.s.ledger_root)
    monkeypatch.setattr(experiment, "RUNS", tmp_path / "runs")
    assert experiment.main(["start", "--repo", str(repo), "--base", "HEAD", "--change", "other", "--budget", "5"]) == 0
    run = capsys.readouterr().out.strip()
    assert experiment.main(["hypothesis", "--run", run, "--hypothesis", "h", "--rationale", "r"]) == 0
    h = capsys.readouterr().out.split()[1]
    assert experiment.main(["call", "trace", "--run", run, "--metric", "kernel_ms"]) == 1
    assert "refused" in capsys.readouterr().err
    assert experiment.main(["call", "trace", "--run", run, "--hypothesis-id", h, "--metric", "kernel_ms",
                            "--value", "1", "--unit", "absolute", "--why", "w"]) == 0
    assert experiment.main(["call", "trace", "--run", "run-agent", "--args", "{}"]) == 1
