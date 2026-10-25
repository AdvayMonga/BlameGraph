"""Fixed-length requests: a record with a measured reference length is sent at that length with ignore_eos, and an
answer that ends before it is truncated."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fake_engine import make_server  # noqa: E402

from regimes import workload as W  # noqa: E402
from regimes.client import Request, send, wall_clock  # noqa: E402

REC = {"id": "r", "prompt": "hi", "max_tokens": 2048, "session_id": "s", "turn_index": 0}


def test_a_measured_record_is_sent_at_its_length_or_the_regimes():
    assert W.to_request(REC).max_tokens == 2048 and not W.to_request(REC).ignore_eos
    fixed = {**REC, "expected_output_tokens": 37}
    assert (W.to_request(fixed).max_tokens, W.to_request(fixed).ignore_eos) == (37, True)
    assert (W.to_request(fixed, max_tokens=64).max_tokens, W.to_request(fixed, max_tokens=64).ignore_eos) == (64, True)


def test_a_fixed_length_answer_runs_to_max_tokens_and_an_early_stop_is_truncated():
    srv, plain = make_server(slots=2, token_s=0.0, out_tokens=5), make_server(token_s=0.0, out_tokens=5, ignore_eos=False)
    url = f"http://127.0.0.1:{srv.server_port}"
    clock = wall_clock()
    try:
        row = send(url, "m", Request("a", [{"role": "user", "content": "hi"}], 12, ignore_eos=True), clock(), clock)
        assert (row.status, row.n_tokens, row.finish_reason) == ("ok", 12, "length")
        row = send(url, "m", Request("b", [{"role": "user", "content": "hi"}], 12), clock(), clock)
        assert (row.status, row.n_tokens) == ("ok", 5)                       # natural stop: today's behaviour
        row = send(f"http://127.0.0.1:{plain.server_port}", "m",
                   Request("c", [{"role": "user", "content": "hi"}], 12, ignore_eos=True), clock(), clock)
        assert row.status == "truncated" and "before max_tokens" in row.detail   # the server ignored ignore_eos
    finally:
        srv.shutdown()
        plain.shutdown()
