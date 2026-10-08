"""lab.trace_contract: conforming trace dirs pass (JSONL, SQLite, the engine's own writer); bad clocks, ids and units fail."""

from __future__ import annotations

import json
import sqlite3
import time

import pytest

from lab import trace_contract as tc

NOW = 1_790_000_000.0


def _row(**kw) -> dict:
    base = dict(schema_version=1, trace_id="bg-1", arrival_ts=NOW - 60, terminal_state="ok", prompt_tokens=10,
                tokens_out=5, total_s=1.0, ttft_s=0.2, tpot_s=0.2, queue_wait_s=0.05, prefill_s=0.1,
                session_id="s", cache_hit_tokens=None, prefill_mode="batched")
    return {**base, **kw}


def _meta(n, **kw) -> dict:
    return {"schema_version": 1, "rows_written": n, "rows_dropped": 0, "closed": 1, **kw}


def _jsonl(tmp_path, rows, meta="auto", name="run"):
    (tmp_path / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    if meta is not None:
        (tmp_path / f"{name}.meta.json").write_text(json.dumps(_meta(len(rows)) if meta == "auto" else meta))
    return tmp_path


def _errors(row) -> list[str]:
    return tc.check_row(row, now=NOW)


def test_conforming_jsonl_dir_passes(tmp_path):
    rows = [_row(), _row(trace_id="bg-2", tokens_out=1, tpot_s=None), _row(trace_id="bg-3", terminal_state="rejected_429",
            tokens_out=0, ttft_s=None, tpot_s=None, total_s=0.0, queue_wait_s=None, prefill_s=None)]
    rep = tc.validate(_jsonl(tmp_path, rows), now=NOW)
    assert rep.valid, rep.errors
    assert len(rep.rows) == 3 and rep.rows_dropped == 0


def test_conforming_sqlite_in_the_engine_layout_passes(tmp_path):
    row = _row()
    conn = sqlite3.connect(tmp_path / "run-x.sqlite")
    conn.execute(f"CREATE TABLE requests ({', '.join(row)})")
    conn.execute(f"INSERT INTO requests VALUES ({', '.join('?' * len(row))})", list(row.values()))
    conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value)")
    conn.executemany("INSERT INTO meta VALUES (?, ?)", list(_meta(1, rows_dropped=2, run_id="run-x").items()))
    conn.commit()
    conn.close()
    rep = tc.validate(tmp_path, now=NOW)
    assert rep.valid, rep.errors
    assert rep.rows_dropped == 2 and rep.rows[0]["trace_id"] == "bg-1"


@pytest.mark.parametrize("row,needle", [
    ({k: v for k, v in _row().items() if k != "trace_id"}, "missing trace_id"),
    (_row(trace_id=""), "trace_id is empty"),
    (_row(trace_id=None), "trace_id is null"),
    (_row(arrival_ts=123_456.7), "not epoch seconds"),                 # perf_counter / uptime
    (_row(arrival_ts=NOW * 1000), "epoch ms or ns"),                  # epoch milliseconds
    (_row(ttft_s=200.0), "ttft_s 200.0 > total_s"),                   # ttft in ms, total in s
    (_row(queue_wait_s=50.0), "queue_wait_s + prefill_s"),            # one span in ms
    (_row(total_s=-0.5), "outside [0"),                               # wall minus monotonic
    (_row(total_s=1_000_000.0, ttft_s=200.0, tpot_s=199_950.0), "outside [0"),   # everything in ms... and a day long
    (_row(tpot_s=0.05), "tpot_s 0.05 != (total_s - ttft_s)"),         # a different TPOT definition
    (_row(tokens_out=3, ttft_s=None, tpot_s=None), "ttft_s must be set exactly"),
    (_row(tokens_out=0, tpot_s=None), "ttft_s must be set exactly"),
    (_row(tokens_out=1), "set with tokens_out 1"),
    (_row(terminal_state="done"), "terminal_state 'done'"),
    (_row(schema_version=2), "schema_version 2"),
    (_row(prompt_tokens="10"), "prompt_tokens is str"),
    (_row(tokens_out=True), "tokens_out is bool"),
    (_row(cache_hit_tokens=11), "cache_hit_tokens 11"),
    (_row(kv_free_frac=1.5), "kv_free_frac 1.5"),
])
def test_bad_rows_are_rejected_with_a_reason(row, needle):
    errs = _errors(row)
    assert any(needle in e for e in errs), errs


def test_a_bad_row_fails_the_dir_and_names_the_file_and_trace(tmp_path):
    rep = tc.validate(_jsonl(tmp_path, [_row(), _row(trace_id="bg-9", arrival_ts=5.0)]), now=NOW)
    assert not rep.valid and len(rep.errors) == 1
    assert rep.errors[0].startswith("run.jsonl row 1 ('bg-9')")


def test_drops_must_be_reported(tmp_path):
    assert "no meta" in " ".join(tc.validate(_jsonl(tmp_path, [_row()], meta=None), now=NOW).errors)


def test_meta_missing_a_count_or_disagreeing_with_the_rows_fails(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    no_drops = {k: v for k, v in _meta(1).items() if k != "rows_dropped"}
    assert "meta missing rows_dropped" in " ".join(tc.validate(_jsonl(a, [_row()], meta=no_drops), now=NOW).errors)
    assert "rows_written 3 but 1 rows" in " ".join(tc.validate(_jsonl(b, [_row()], meta=_meta(3)), now=NOW).errors)


def test_open_file_may_lag_its_count(tmp_path):
    """closed=0: the engine was killed before final counts; the row count is not held against it."""
    assert tc.validate(_jsonl(tmp_path, [_row()], meta=_meta(0, closed=0)), now=NOW).valid


def test_empty_dir_fails_and_cli_exit_codes(tmp_path, capsys):
    assert tc.main([str(tmp_path)]) == 1
    _jsonl(tmp_path, [_row(arrival_ts=time.time() - 5)])
    assert tc.main([str(tmp_path)]) == 0
    assert capsys.readouterr().out.strip().endswith("ok: 1 rows in 1 files, 0 dropped, 0 errors")


def test_inference_server_rows_conform(tmp_path):
    """The engine's own writer, through its own scheduler-facing API, produces a conforming dir."""
    telemetry = pytest.importorskip("inference_server.telemetry", reason="needs the engine", exc_type=ImportError)
    if not hasattr(telemetry, "SCHEMA_VERSION"):
        pytest.skip("engine predates the trace contract (inference-server#102)")
    store = telemetry.RowStore(tmp_path)
    t0 = time.perf_counter()
    for i, (state, n) in enumerate([("ok", 5), ("ok", 1), ("ok", 0), ("rejected_429", 0), ("cancelled", 2)]):
        rec = telemetry.RequestRecord(
            trace_id=f"bg-{i}", session_id="s", turn_index=0, arrival_ts=time.time(), pending_depth=0,
            active_size=0, max_batch_size=8, kv_free_blocks=10, kv_free_frac=0.5, concurrent_sessions=0,
            prompt_tokens=10, replica_age_s=1.0)
        admitted = state != "rejected_429"
        rec.finish(state, arrival_mono=t0, enqueue_ts=t0 + 0.01, admit_ts=t0 + 0.02 if admitted else 0.0,
                   first_token_ts=t0 + 0.1 if admitted else 0.0, end_ts=t0 + 0.5, tokens_out=n,
                   cache_hit_tokens=4 if admitted else None, decode_width_steps=n, decode_width_sum=n,
                   decode_width_max=1, prefill_mode="batched" if admitted else None)
        store.put(rec)
    store.close()
    rep = tc.validate(tmp_path)
    assert rep.valid, rep.errors
    assert len(rep.rows) == 5 and list(rep.files) == [store.path.name]
