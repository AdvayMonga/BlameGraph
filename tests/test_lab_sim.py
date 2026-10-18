"""The scheduler simulator: the engine's real scheduler on a fake clock. Skips unless the engine imports (run with
the engine's python, its src on sys.path: tests/conftest.py)."""
from __future__ import annotations

import pytest

pytest.importorskip("inference_server.scheduler", reason="needs the engine and its deps", exc_type=ImportError)

from regimes.runner import CONVERSATIONAL, summarize  # noqa: E402
from sim.scheduler import Clock, CostModel, Settings, installed, simulate  # noqa: E402


def req(k, at=0.0, prompt=100, out=20, max_tokens=None, session=None):
    return {"id": f"r{k}", "arrival_s": at, "build_prompt_tokens": prompt, "max_tokens": max_tokens or 2 * out,
            "expected_output_tokens": out, "session_id": session or f"s{k}", "turn_index": 0}


def test_one_request_alone_costs_exactly_the_model():
    c = CostModel()
    r = simulate([req(0, prompt=1000, out=50)], Settings(max_batch_size=4), c)
    [row] = r.rows
    assert row.status == "ok" and row.n_tokens == 50 and row.finish_reason == "stop"
    assert row.ttft_s == pytest.approx(c.prefill_s(1000))
    assert row.tpot_s == pytest.approx(c.decode_s(1, 1000 + 25), rel=0.02)      # KV grows a token a step
    assert r.prefill_waves == 1 and r.decode_steps == 50                          # 49 tokens + the EOS step


def test_the_batch_limit_queues_the_rest():
    r = simulate([req(k) for k in range(8)], Settings(max_batch_size=4))
    ttft = sorted(x.ttft_s for x in r.rows)
    assert ttft[:4] == [pytest.approx(ttft[0])] * 4 and all(t > ttft[0] + 0.3 for t in ttft[4:])
    assert all(x.n_tokens == 20 for x in r.rows)


def test_a_full_queue_is_an_explicit_rejection_and_the_scheduler_is_the_engines():
    from inference_server.scheduler import ContinuousBatchScheduler
    r = simulate([req(k) for k in range(8)], Settings(max_batch_size=1, max_queue_size=2))
    assert sorted(x.status for x in r.rows) == ["error"] * 6 + ["ok"] * 2
    assert {x.http_status for x in r.rows if x.status == "error"} == {429}
    assert ContinuousBatchScheduler.step                                          # the seam the driver calls


def test_kv_budget_holds_admission_and_arrivals_keep_their_own_time():
    # 4 blocks of 16: one request of 100 + 20 tokens needs 8 blocks and can never fit (400); a smaller one fits
    r = simulate([req(0, prompt=100, out=20), req(1, at=0.5, prompt=20, out=4)], Settings(kv_blocks=4, kv_block_size=16))
    by = {x.id: x for x in r.rows}
    assert by["r0"].status == "error" and by["r0"].http_status == 400                # too large for the cache
    assert by["r1"].status == "ok" and by["r1"].scheduled_s == 0.5
    # arrivals during a long step still count from their own arrival time
    r = simulate([req(0, prompt=20000, out=2), req(1, at=0.1, prompt=10, out=2)], Settings(max_batch_size=1))
    by = {x.id: x for x in r.rows}
    assert by["r1"].ttft_s > by["r0"].ttft_s - 0.1 - 1e-6                             # it waited the whole prefill


def test_settings_from_the_engines_env_and_a_profile_shaped_run():
    s = Settings.from_env({"MAX_BATCH_SIZE": "32", "SCHEDULING_POLICY": "fcfs", "MAX_QUEUE_WAIT_S": "2.5", "OTHER": "x"})
    assert (s.max_batch_size, s.scheduling_policy, s.max_queue_wait_s) == (32, "fcfs", 2.5)
    recs = [req(k, at=0.01 * k, out=30) for k in range(200)]
    fair = summarize(simulate(recs, Settings(max_batch_size=16)).rows, CONVERSATIONAL)
    fcfs = summarize(simulate(recs, Settings(max_batch_size=16, scheduling_policy="fcfs")).rows, CONVERSATIONAL)
    assert fair["n"] == fcfs["n"] == 200 and fair["ttft_p99_s"] > 0 and fcfs["ttft_p99_s"] > 0


def test_max_queue_wait_sheds_what_cannot_be_served_in_time():
    recs = [req(k, out=400) for k in range(40)]
    r = simulate(recs, Settings(max_batch_size=2, max_queue_wait_s=1.0))
    assert sum(x.status == "error" for x in r.rows) >= 30 and sum(x.status == "ok" for x in r.rows) >= 2


def test_the_clock_is_installed_only_while_simulating():
    from inference_server import scheduler as mod
    import time
    assert mod.time is time
    with installed(Clock()):
        assert mod.time is not time
    assert mod.time is time
