"""Load regimes against a fake engine with a known capacity. Run: python tests/test_regimes.py"""
from __future__ import annotations

import json
import socket
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from canaries.cheat_proxy import serve  # noqa: E402
from fake_engine import make_server, spawn  # noqa: E402
from regimes import runner, suite, workload as W  # noqa: E402
from regimes.client import Request, send, wall_clock  # noqa: E402
from regimes.runner import Limits, find_goodput, run_open, summarize  # noqa: E402

LOOSE = Limits(ttft_s=0.3, tpot_s=0.05)


def _url(srv):
    return f"http://127.0.0.1:{srv.server_address[1]}"


def _free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close()
    return p


def test_client_times_and_classifies():
    srv = make_server(slots=2, token_s=0.005, out_tokens=10)
    clock = wall_clock()
    row = send(_url(srv), "m", Request("a", [{"role": "user", "content": "hi"}], 64), clock(), clock)
    assert row.status == "ok" and row.n_tokens == 10 and row.text.startswith("t0 ")
    assert 0 < row.ttft_s < row.e2e_s and 0.003 < row.tpot_s < 0.03, row
    full = make_server(slots=1, token_s=0.001, queue_limit=0)
    assert send(_url(full), "m", Request("b", [{"role": "user", "content": "hi"}], 8), clock(), clock).status == "error"
    dead = f"http://127.0.0.1:{_free_port()}"
    assert send(dead, "m", Request("c", [{"role": "user", "content": "hi"}], 8), clock(), clock).status == "crash"
    srv.shutdown(); full.shutdown()


def test_open_loop_keeps_its_schedule():
    proc, url = spawn(slots=64, token_s=0.001, out_tokens=5)
    try:
        reqs = [Request(f"r{k}", [{"role": "user", "content": "hi"}], 8, at_s=k * 0.02) for k in range(50)]
        s = summarize(run_open(url, "m", reqs), LOOSE)
        assert s["n"] == 50 and s["ok"] == 50 and s["valid"] and s["meets_limits"], s
        assert s["client_lag_p99_s"] < 0.01
    finally:
        proc.kill()


def test_goodput_search_finds_the_capacity():
    proc, url = spawn(slots=4, token_s=0.005, out_tokens=20)       # 0.1 s per request x 4 slots = 40 req/s
    try:
        pool = W.synthetic(50)
        probe = lambda rate: summarize(run_open(url, "m", W.poisson(pool, rate, 2.0, seed=1)), LOOSE, 2.0)
        res = find_goodput(probe, 5.0)
        assert res["goodput"] is not None and 15 <= res["goodput"] <= 45, \
            [(p["load"], p["attainment"], p["invalid_reasons"]) for p in res["probes"]]
    finally:
        proc.kill()


def test_overload_explicit_rejection_is_valid_silent_drop_is_not():
    srv = make_server(slots=2, token_s=0.005, out_tokens=10, queue_limit=3)
    reqs = W.poisson(W.synthetic(20), 200.0, 1.0, seed=2)
    honest = summarize(run_open(_url(srv), "m", reqs), LOOSE, 1.0)
    assert honest["error"] > 0 and honest["valid"], honest
    roomy = make_server(slots=64, token_s=0.001, out_tokens=5)
    px = serve(_url(roomy), 0, "drop")
    cheat = summarize(run_open(_url(px), "m", W.poisson(W.synthetic(20), 50.0, 1.0, seed=3)), LOOSE, 1.0)
    assert cheat["silent_drop"] > 0 and not cheat["valid"], cheat
    srv.shutdown(); roomy.shutdown(); px.shutdown()


def test_workload_builders():
    rec = {"id": "c-0", "session_id": "s", "turn_index": 2, "max_tokens": 64, "prompt": "x",
           "messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
                        {"role": "user", "content": "c"}, {"role": "assistant", "content": "d"},
                        {"role": "user", "content": "e"}]}
    conv = W.conversations([rec])[0]
    assert [len(t["messages"]) for t in conv] == [1, 3, 5] and all(t["messages"][-1]["role"] == "user" for t in conv)
    reqs = W.sessions([rec], rate=10.0, duration_s=1.0, seed=0, think_s=2.0, system="SYS")
    assert reqs and reqs[0].messages[0] == {"role": "system", "content": "SYS"}
    s0 = [r for r in reqs if r.session_id == reqs[0].session_id]
    assert [round(r.at_s - s0[0].at_s, 6) for r in s0] == [0.0, 2.0, 4.0]
    trace = [{"id": f"t{k}", "prompt": "p", "arrival_s": 100.0 + k, "max_tokens": 8} for k in range(10)]
    assert [r.at_s for r in W.replay(trace, 2.0)][:3] == [0.0, 0.5, 1.0]
    lo, hi = W.peak_window(trace + [{"id": "b", "prompt": "p", "arrival_s": 500.0 + k / 10, "max_tokens": 8} for k in range(30)])
    assert lo < 500 < hi
    from lab.corpus import CorpusError, TraceRequest, WorkloadClass, build_manifest, write_trace
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        cls = {"spike": WorkloadClass("spike", "", 2000.0, None, 1.0, "spike/seen.jsonl", "spike/heldout.jsonl")}
        for split in ("seen", "heldout"):
            write_trace(d / "spike" / f"{split}.jsonl",
                        [TraceRequest(float(k), f"s{k}", 0, f"p{k}", 8, build_prompt_tokens=3) for k in range(10)])
        m = build_manifest(cls, d)
        (d / "manifest.json").write_text(json.dumps(m.to_dict(), indent=2))
        recs = W.corpus(d, "spike")
        assert len(recs) == 10 and recs[0]["id"] == "spike-seen-0" and recs[0]["build_prompt_tokens"] == 3
        assert W.corpus_version(d) == m.corpus_version and W.corpus_classes(d) == ["spike"]
        (d / "spike" / "seen.jsonl").write_text("{}\n")             # a changed trace refuses, it is a new version
        try:
            W.corpus(d, "spike"); raise AssertionError("tampered trace loaded")
        except CorpusError:
            pass


def test_every_regime_runs_on_a_fake_engine():
    proc, url = spawn(slots=2, token_s=0.005, out_tokens=10)       # ~40 req/s capacity
    saved, think, lag = dict(suite.TIERS["short"]), suite.THINK_S, runner.MAX_CLIENT_LAG_S
    suite.TIERS["short"].update(probe_s=1.0, final_s=2.0)
    suite.THINK_S = 0.3
    runner.MAX_CLIENT_LAG_S = 0.05      # wiring test; client precision is test_open_loop_keeps_its_schedule's job
    try:
        tight = Limits(ttft_s=0.25, tpot_s=0.05)                  # 1 s probes only see overload against a tight TTFT
        ctx = suite.Ctx(url, "m", min_requests=10, interactive=tight, conversational=tight)
        for name, fn in suite.REGIMES.items():
            res = fn(ctx, max_concurrency=8) if name == "saturated" else fn(ctx)
            assert res["regime"] == name and res["valid"], (name, res["invalid_reasons"])
            assert res["value"] is not None, (name, res)
    finally:
        suite.TIERS["short"].update(saved)
        suite.THINK_S, runner.MAX_CLIENT_LAG_S = think, lag
        proc.kill()


def test_cold_start_measures_launch_to_correct_token():
    port = _free_port()
    cmd = f"{sys.executable} {Path(__file__).resolve().parent / 'fake_engine.py'} --port {port} --delay 0.5"
    res = suite.cold_start(f"http://127.0.0.1:{port}", "m", cmd, timeout_s=30, poll_s=0.05)
    assert res["valid"] and 0.5 <= res["value"] < 10, res
    assert res["warm"]["first_correct_token_s"] is not None
    bad = suite.cold_start(f"http://127.0.0.1:{port}", "m", cmd, expect="43", timeout_s=3, poll_s=0.05, warm=False)
    assert not bad["valid"] and "expected '43'" in bad["cold"]["detail"], bad


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
