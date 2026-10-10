"""bench, equiv and submit end to end: the agent's tiny engine repo launches tests/fake_engine.py as its server."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from lab import ledger, serve, target
from lab.budget import Budget
from lab.session import Session
from lab.tools import Toolbox
from lab.workspace import Workspace
from regimes import suite

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lab_fixtures import make_repo  # noqa: E402

FAKE = Path(__file__).resolve().parent / "fake_engine.py"


class Enc:
    def encode(self, text):
        return [hash(w) % 1000 + 1 for w in text.split()]

    def chat_ids(self, messages):
        return self.encode(" ".join(m["content"] for m in messages))


def _target(tmp_path: Path, repo: Path, ref_dir: Path) -> Path:
    t = tmp_path / "fake.toml"
    t.write_text(f'''
[model]
name = "fake"
chat_kwargs = {{ }}
[engine]
repo = "{repo}"
python = "{sys.executable}"
api = "vllm"
launch = "python {FAKE} --port {{port}} --slots 4 --token-s 0.002 --out-tokens 6"
health = "/health"
startup_timeout_s = 30
write = ["src/inference_server/*"]
add_only = ["tests/test_*.py"]
test = "python -m pytest -q -p no:cacheprovider"
lint = "python -m ruff check ."
[engine.telemetry]
env = {{ TELEMETRY_DIR = "{{dir}}" }}
[reference]
api = "vllm"
launch = "python {FAKE} --port {{port}}"
dir = "{ref_dir}"
[limits]
interactive = {{ ttft_s = 0.25, tpot_s = 0.05 }}
conversational = {{ ttft_s = 0.25, tpot_s = 0.05 }}
[corpus]
dir = "{tmp_path / 'no-corpus'}"
''')
    return t


def _reference(url: str, out: Path, n: int = 40):
    from correctness import run as crun
    items = [{"id": f"mc-{k}", "task": "mmlu_pro", "gold": "C", "max_tokens": 6,
              "messages": [{"role": "user", "content": f"question {k} pick one"}]} for k in range(n)]
    crun.reference(url, "fake", out, items, Enc(), top_k=3, div_n=8, concurrency=4)


@pytest.fixture
def lab(tmp_path, monkeypatch):
    monkeypatch.setenv("LAB_NO_JAIL", "1")
    repo = make_repo(tmp_path)
    # a reference from the same fake engine, dev and full tiers
    import socket
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    proc = subprocess.Popen([sys.executable, str(FAKE), "--port", str(port)])
    try:
        import time
        import urllib.request
        for _ in range(100):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1); break
            except OSError:
                time.sleep(0.1)
        _reference(f"http://127.0.0.1:{port}", tmp_path / "ref")
        _reference(f"http://127.0.0.1:{port}", tmp_path / "ref-full")
    finally:
        proc.kill()
    spec = _target(tmp_path, repo, tmp_path / "ref")
    monkeypatch.setenv("LAB_TARGET", str(spec))
    monkeypatch.setenv("LAB_ENGINE_REPO", str(repo))
    monkeypatch.setenv("LAB_ENGINE_PYTHON", sys.executable)
    target._cache.clear()
    ws = Workspace(repo, "HEAD", tmp_path / "ws", tmp_path / "ledger"); ws.create()
    run_dir = tmp_path / "run"; run_dir.mkdir()
    s = Session("r1", "r1-s1", run_dir, ws, Budget.resume(5.0, "r1", tmp_path / "ledger"), tmp_path / "ledger")
    s.encoder = Enc()
    tb = Toolbox(s)
    tb.serve_jailed = False
    tb.bench_default_regimes = ("single_stream",)
    from regimes import runner
    saved, think, lag = dict(suite.TIERS["short"]), suite.THINK_S, runner.MAX_CLIENT_LAG_S
    runner.MAX_CLIENT_LAG_S = 0.05      # wiring tests on a shared laptop; client precision has its own test
    suite.TIERS["short"].update(probe_s=1.0, final_s=1.5)
    suite.TIERS["full"].update(probe_s=1.0, final_s=1.5)
    suite.THINK_S = 0.3
    yield tb, s
    suite.TIERS["short"].update(saved); suite.TIERS["full"].update(probe_s=60.0, final_s=600.0); suite.THINK_S = think
    runner.MAX_CLIENT_LAG_S = lag
    target._cache.clear()


def test_bench_serves_and_measures(lab):
    tb, s = lab
    out = json.loads(tb.bench({"regimes": ["single_stream"]}))
    assert out["split"] == "seen" and out["metrics"]["single_stream"]["valid"]
    assert out["metrics"]["single_stream"]["value"] is not None
    rec = next(ledger.records(s.ledger_root, kind="bench"))
    assert rec["config"] == {"split": "seen", "tier": "short"} and rec["result"]["verdict"] == "ok"
    assert rec["result"]["ready_s"] is not None and (s.run_dir / "serve-bench.log").exists()


def test_bench_keeps_passive_data_as_blobs(lab):
    from lab import artifacts, gpu
    tb, s = lab
    tb.bench({"regimes": ["single_stream"]})
    rec = next(ledger.records(s.ledger_root, kind="bench"))
    a = rec["result"]["artifacts"]
    assert set(a) == {"device", "telemetry", "client_rows", "serve_log"}
    assert all((s.ledger_root / p).exists() for p in a.values())
    back = artifacts.load(rec, s.ledger_root)
    assert back["device"]["available"] == gpu.available()
    rows = back["client_rows"]
    assert len(rows) == rec["result"]["regimes"][0]["summary"]["n"] and {r["regime"] for r in rows} == {"single_stream"}
    assert "text" not in rows[0]
    joined = artifacts.joined(rec, s.ledger_root)
    assert all(j["engine"] and j["engine"]["tokens_out"] == 6 for j in joined)      # every request has its engine row
    assert not (s.run_dir / "pristine" / serve.TELEMETRY_SUBDIR).exists()          # moved out of the served tree
    assert not list(s.run_dir.glob("passive-*"))                                     # scratch emptied into blobs
    contract = json.loads((s.ledger_root / a["telemetry"] / "contract.json").read_text())
    assert contract["valid"] is False and contract["errors"]      # checked on collection; the fake's rows don't conform


def test_equiv_judges_against_the_reference(lab):
    tb, s = lab
    out = json.loads(tb.equiv({}))
    assert out["verdict"] == "pass", out["reasons"]
    assert out["metrics"]["accuracy"]["unanswered"] == 0 and "kl_mean" in out["metrics"]["divergence"]
    rec = next(ledger.records(s.ledger_root, kind="equiv"))
    assert rec["result"]["passed"] is True and rec["result"]["tier"] == "dev"
    from lab import artifacts
    rows = artifacts.load(rec, s.ledger_root)["client_rows"]
    assert len(rows) == 40 and all("text" not in r for r in rows)
    from correctness.run import TRACE_PREFIX
    joined = artifacts.joined(rec, s.ledger_root, prefix=TRACE_PREFIX)
    assert all(j["engine"] for j in joined)                    # every gate request has its engine row (X-Trace-Id)


def test_a_second_equiv_on_the_same_snapshot_reuses_its_answers(lab):
    tb, s = lab
    tb.equiv({})
    answers = next((s.run_dir / "equiv").glob("*-dev.outputs.jsonl"))
    marked = "".join(json.dumps({**json.loads(x), "kept": True}) + "\n" for x in answers.read_text().splitlines())
    answers.write_text(marked)
    tb.equiv({})
    assert answers.read_text() == marked                        # nothing asked again: the gate is never re-rolled
    assert not list(s.run_dir.glob("equiv-prior-*")) and not list(s.run_dir.glob("passive-*"))


def test_submit_keeps_heldout_passive_data_out_of_the_ledger(lab):
    tb, s = lab
    tb.equiv({"tier": "full"}); tb.bench({"regimes": ["single_stream"]})
    before = {p.name for p in (s.ledger_root / "blobs").iterdir()}
    json.loads(tb.submit({}))
    rec = next(r for r in ledger.records(s.ledger_root, kind="submit") if r.get("config", {}).get("split") == "heldout")
    assert rec.keys() - ledger.WRITER_FIELDS <= ledger.HELDOUT_FIELDS and "artifacts" not in json.dumps(rec)
    assert {p.name for p in (s.ledger_root / "blobs").iterdir()} == before          # no held-out blob of any kind
    private = list((s.run_dir / "heldout-private").iterdir())
    assert len(private) == 1 and (private[0] / "client_rows.jsonl").read_text().strip()
    assert (private[0] / "telemetry" / "requests.jsonl").exists() and (private[0] / "device" / "meta.json").exists()
    assert not (s.run_dir / "pristine" / serve.TELEMETRY_SUBDIR).exists()


def test_submit_needs_full_equiv_and_a_bench_then_records_one_aggregate_per_metric(lab):
    tb, s = lab
    assert tb.submit({}).startswith("submit refused: no full-tier equiv")
    assert json.loads(tb.equiv({"tier": "full"}))["verdict"] == "pass"
    assert tb.submit({}).startswith("submit refused: no completed short-tier seen-split bench")
    tb.bench({"regimes": ["single_stream"], "tier": "full"})            # the wrong tier does not count
    assert tb.submit({}).startswith("submit refused: no completed short-tier seen-split bench")
    tb.bench({"regimes": ["single_stream"]})
    out = json.loads(tb.submit({}))
    assert out["split"] == "heldout" and set(out["metrics"]) == {"single_stream"}
    m = out["metrics"]["single_stream"]
    assert set(m) == {"base", "new", "delta_pct", "band_pct", "verdict"} and m["verdict"] == "unknown_band"
    rec = next(r for r in ledger.records(s.ledger_root, kind="submit") if r.get("config", {}).get("split") == "heldout")
    assert "result" not in rec and rec["metrics"] == out["metrics"]        # the ledger's held-out shape, nothing else
    state = json.loads((s.ledger_root / "holdout_state.json").read_text())
    assert state["n_queries"] == 1 and (s.run_dir / "base-heldout-full.json").exists()


def test_inconclusive_equiv_is_neither_passing_nor_failing(lab, monkeypatch):
    from correctness import run as crun
    tb, s = lab
    real = crun.candidate

    def inconclusive(*a, **kw):
        res = real(*a, **kw)
        res["verdict"], res["passed"] = "inconclusive", False
        return res
    monkeypatch.setattr(crun, "candidate", inconclusive)
    assert json.loads(tb.equiv({}))["verdict"] == "inconclusive"
    rec = list(ledger.records(s.ledger_root, kind="equiv"))[-1]
    assert rec["result"]["passed"] is None                              # lab_verdict counts neither pass nor fail
    from feedback.lab_verdict import passed
    assert passed(rec) is None


def test_submit_reports_no_heldout_delta_without_a_seen_delta(lab, monkeypatch, tmp_path):
    from lab import evaltools
    tb, s = lab
    tb.equiv({"tier": "full"}); tb.bench({"regimes": ["single_stream"]})
    monkeypatch.setattr(evaltools.EvalTools, "base_seen", lambda self, t, names: {})   # no seen baseline at all
    out = json.loads(tb.submit({}))
    m = out["metrics"]["single_stream"]
    assert m["delta_pct"] is None and m["new"] is None and m["verdict"] == "unknown"     # nothing held-out leaks
    # a noise band from another target is ignored
    nd = tmp_path / "noise2"; nd.mkdir()
    (nd / "single_stream.json").write_text(json.dumps({"band_pct": 1.0, "target": "other", "server": "engine"}))
    monkeypatch.setattr(target, "NOISE_DIR", nd)
    assert evaltools.noise_band_pct("single_stream", target.load()) is None
    (nd / "single_stream.json").write_text(json.dumps({"band_pct": 1.0, "target": target.load().name, "server": "engine"}))
    assert evaltools.noise_band_pct("single_stream", target.load()) == 1.0


def test_submit_scores_the_task(lab, tmp_path):
    from lab.task import Task
    tb, s = lab
    t = tmp_path / "task.toml"
    t.write_text('[task]\ngoal = "g"\n[objective]\nregimes = ["single_stream"]\n')
    s.task = Task.parse(t)
    tb.equiv({"tier": "full"}); tb.bench({"regimes": ["single_stream"]})
    out = json.loads(tb.submit({}))
    sc = out["score"]
    assert set(sc["gains_pct"]) == {"single_stream"} and sc["win"] is False       # no noise band yet: unknown_band
    assert sc["verdicts"]["single_stream"] == "unknown_band" and sc["missing_regimes"] == []


def test_tools_refuse_cleanly_when_the_engine_does_not_start(lab, monkeypatch):
    tb, s = lab
    t = target.load()
    broken = target.Server("python -c 'import sys; sys.exit(3)'", {}, "/health", "vllm", 5.0)
    monkeypatch.setattr(target, "load", lambda path=None: t.__class__(**{**t.__dict__, "engine": broken}))
    assert tb.bench({}).startswith("bench failed: engine did not start")
    rec = list(ledger.records(s.ledger_root, kind="bench"))[-1]
    assert rec["result"]["verdict"] == "error"
    assert rec["result"]["artifacts"]["telemetry"] is None and rec["result"]["artifacts"]["serve_log"]   # no section: skipped


def test_the_workbench_is_quieted_for_a_measurement_and_what_was_killed_is_a_fact(lab, monkeypatch):
    from contextlib import contextmanager
    from lab.safety import container
    tb, s = lab
    s.workbench = "lab-workbench-r1-s1"
    asked = []

    @contextmanager
    def quiet(name):
        asked.append(name)
        yield [{"pid": 4242, "used_mib": 9000, "name": "python"}]
    monkeypatch.setattr(container, "quiet", quiet)
    out = json.loads(tb.bench({"regimes": ["single_stream"]}))
    assert asked == ["lab-workbench-r1-s1"] and out["workbench_killed"][0]["pid"] == 4242
    rec = list(ledger.records(s.ledger_root, kind="bench"))[-1]["result"]
    assert rec["verdict"] == "ok" and rec["workbench_killed"][0]["pid"] == 4242


AGENT_PROC = {"pid": 4242, "used_mib": 9000, "name": "python"}


def test_nothing_is_measured_while_another_process_holds_the_gpu(lab, monkeypatch):
    from lab import gpu
    tb, s = lab
    monkeypatch.setattr(gpu, "compute_apps", lambda: [AGENT_PROC])
    out = tb.bench({"regimes": ["single_stream"]})
    assert out.startswith("bench refused: the GPU was in use") and "pid 4242" in out
    rec = list(ledger.records(s.ledger_root, kind="bench"))[-1]["result"]
    assert rec["verdict"] == "refused" and rec["gpu_processes"] == [AGENT_PROC]
    assert not (s.run_dir / "serve-bench.log").exists()                 # no engine was launched


def test_a_measurement_shared_with_another_gpu_process_is_contaminated_and_never_cached(lab, monkeypatch):
    from lab import gpu
    tb, s = lab
    calls = []
    monkeypatch.setattr(serve, "POLL_S", 0.05)
    monkeypatch.setattr(gpu, "compute_apps", lambda: calls.append(1) or ([] if len(calls) == 1 else [AGENT_PROC]))
    out = tb.bench({"regimes": ["single_stream"]})
    assert out.startswith("bench contaminated: another process held the GPU")
    rec = list(ledger.records(s.ledger_root, kind="bench"))[-1]["result"]
    assert rec["verdict"] == "contaminated" and rec["gpu_processes"] == [AGENT_PROC] and "metrics" not in rec
    calls.clear()
    with pytest.raises(serve.Contaminated):
        tb.base_seen(target.load(), ["single_stream"])
    assert not list(s.run_dir.glob("base-*.json"))


def test_the_engines_own_gpu_processes_are_not_foreign(lab, monkeypatch):
    import time
    from lab import gpu
    tb, s = lab
    srv = serve.Served(tb._pristine(), target.load().engine, jailed=False)
    monkeypatch.setattr(serve, "POLL_S", 0.05)
    monkeypatch.setattr(gpu, "compute_apps", lambda: [] if srv.proc is None else
                        [{"pid": srv.proc.pid, "used_mib": 1, "name": "engine"}])
    with srv:
        time.sleep(0.3)
        srv.exclusive()
    assert srv.foreign == {}


class TwoSessions:
    """Continues once, then stops; keeps each session's spec."""
    name = "scripted"

    def __init__(self):
        self.specs = []

    def run(self, spec):
        from lab.agent import AgentReply
        self.specs.append(spec)
        return AgentReply({"status": "continue" if len(self.specs) == 1 else "stop", "note": None}, 0.1, 1, None)


def test_baseline_is_measured_once_per_run_and_briefed_with_one_ledger_record(lab, monkeypatch, tmp_path):
    from lab import evaltools, session
    tb, s = lab
    monkeypatch.setattr(evaltools.EvalTools, "bench_default_regimes", ("single_stream",))
    monkeypatch.setattr(evaltools.EvalTools, "serve_jailed", False)
    calls = []
    real = evaltools.run_regimes
    monkeypatch.setattr(evaltools, "run_regimes", lambda *a, **kw: calls.append(a[2:4]) or real(*a, **kw))
    cfg = session.RunConfig(goal="g", budget_usd=2.0, repo=s.workspace.repo, runs_dir=tmp_path / "runs",
                            ledger_root=tmp_path / "led", run_id="rb", max_sessions=2)
    p = TwoSessions()
    assert session.run(cfg, p)["sessions"] == 2
    session.run(cfg, TwoSessions())                       # resumed: still the same baseline
    assert calls == [(["single_stream"], "seen")]
    base = list(ledger.records(cfg.ledger_root, kind="baseline"))
    assert len(base) == 1 and base[0]["result"]["verdict"] == "ok"
    assert base[0]["cost"]["seconds"] > 0 and "rate_known" in base[0]["cost"]
    assert base[0]["config"] == {"split": "seen", "tier": "short", "commit": session._resolve(cfg.repo, "HEAD")}
    assert (tmp_path / "runs" / "rb" / "base-seen-short.json").exists()      # the cache submit compares against
    first, second = (sp.prompt for sp in p.specs)
    for prompt in (first, second):
        line = prompt.split("Baseline (", 1)[1].split("\n")[1]
        assert json.loads(line)["single_stream"]["value"] == base[0]["result"]["metrics"]["single_stream"]["value"]
    assert "Last session of this run:\nnone" in first
    last = json.loads(second.split("Last session of this run:\n", 1)[1].split("\n")[0])
    assert last["kind"] == "session" and last["session"] == "rb-s1"
    assert second.count('"id": "ev-') == 1                # one ledger record, nothing else inlined


def test_a_base_that_will_not_serve_is_briefed_as_a_fact(lab, monkeypatch, tmp_path):
    from lab import session
    tb, s = lab
    t = target.load()
    broken = target.Server("python -c 'import sys; sys.exit(3)'", {}, "/health", "vllm", 5.0)
    monkeypatch.setattr(target, "load", lambda path=None: t.__class__(**{**t.__dict__, "engine": broken}))
    cfg = session.RunConfig(goal="g", budget_usd=2.0, repo=s.workspace.repo, runs_dir=tmp_path / "runs",
                            ledger_root=tmp_path / "led", run_id="rx", max_sessions=1)
    p = TwoSessions()
    session.run(cfg, p)
    assert "not measured; engine did not start" in p.specs[0].prompt
    assert next(ledger.records(cfg.ledger_root, kind="baseline"))["result"]["verdict"] == "error"


def test_bench_defaults_follow_the_task():
    from types import SimpleNamespace
    from lab.evaltools import EvalTools
    from lab.task import Task
    tb = EvalTools()
    tb.s = SimpleNamespace(task=None)
    assert tb.bench_defaults() == EvalTools.bench_default_regimes
    tb.s.task = Task("g", ("long_prompt_short_output",), constraints={"single_stream": {"max_regression_pct": 5},
                                                                     "long_prompt_short_output": {"max_regression_pct": 1}})
    assert tb.bench_defaults() == ("long_prompt_short_output", "single_stream")


def test_with_the_agent_on_a_remote_worker_its_workspace_is_pulled_before_every_audit_and_pushed_after_a_restore(lab):
    tb, s = lab
    moves = []
    local = tb.worker

    class Mirror:
        def call(self, *a, **k):
            return local.call(*a, **k)

        def pull(self, remote, path):
            moves.append(("pull", remote))

        def push(self, path, remote):
            moves.append(("push", remote))
    tb.worker, s.remote_workspace = Mirror(), "/far/ws"
    tb.test({})
    assert moves == [("pull", "/far/ws")]
    moves.clear()
    tb.restore({"snapshot": "base"})
    assert ("push", "/far/ws") in moves and moves[0] == ("pull", "/far/ws")
