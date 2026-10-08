"""lab.artifacts: blobs round-trip, and client rows join engine rows by trace id (SQLite and JSONL)."""
from __future__ import annotations

import json
import sqlite3

from lab import artifacts


def test_store_load_and_join(tmp_path):
    work, root = tmp_path / "work", tmp_path / "ledger"
    (work / "device").mkdir(parents=True)
    (work / "device" / "meta.json").write_text(json.dumps({"available": False}))
    (work / "telemetry").mkdir()
    con = sqlite3.connect(work / "telemetry" / "run1.sqlite")
    con.execute("CREATE TABLE requests (trace_id TEXT, queue_wait_s REAL)")
    con.execute("INSERT INTO requests VALUES ('bg-a', 0.5)")
    con.commit(); con.close()
    (work / "telemetry" / "events.jsonl").write_text('{"event": "step"}\n{"trace_id": "bg-b", "tokens_out": 3}\n')
    (work / "serve.log").write_text("ready\n")
    artifacts.write_rows(work, [{"id": "a", "ttft_s": 0.1}, {"id": "b"}, {"id": "c"}])
    a = artifacts.store(work, root)
    assert not work.exists() and all((root / p).exists() for p in a.values())
    rec = {"result": {"artifacts": a}}
    back = artifacts.load(rec, root)
    assert back["device"] == {"available": False} and back["serve_log"] == "ready\n" and back["device_samples"] is None
    j = {r["id"]: r["engine"] for r in artifacts.joined(rec, root)}
    assert j == {"a": {"trace_id": "bg-a", "queue_wait_s": 0.5}, "b": {"trace_id": "bg-b", "tokens_out": 3}, "c": None}


def test_absent_telemetry_is_none(tmp_path):
    work = tmp_path / "work"
    artifacts.write_rows(work, [])
    a = artifacts.store(work, tmp_path / "ledger")
    assert a["telemetry"] is None and a["device"] is None and artifacts.joined({"result": {"artifacts": a}}, tmp_path / "ledger") == []
