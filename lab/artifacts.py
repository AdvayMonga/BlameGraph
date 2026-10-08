"""Passive data from one served run (device samples, engine telemetry, client rows, serve log): stored as ledger blobs
referenced from a bench/equiv record's `result.artifacts`, and loaded back with client rows joined to engine rows."""

from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path

from lab import ledger

ROWS = "client_rows.jsonl"


def write_rows(work: Path, rows: list[dict]) -> None:
    """The client's per-request rows (no generated text) into `work`, one JSON object per line."""
    Path(work).mkdir(parents=True, exist_ok=True)
    (Path(work) / ROWS).write_text("".join(json.dumps(r) + "\n" for r in rows))


def store(work: Path, root: Path) -> dict:
    """`work`'s passive data as ledger blobs, {device, telemetry, client_rows, serve_log} (None when absent); `work`
    is removed."""
    def blob(name: str, suffix: str = "") -> str | None:
        p = Path(work) / name
        return ledger.put_blob(p, root, suffix) if p.exists() else None
    tel = Path(work) / "telemetry"
    if tel.is_dir() and any(tel.iterdir()):       # the request-trace contract check travels with the data (lab/TRACE.md)
        from lab.trace_contract import validate
        rep = validate(tel)
        (tel / "contract.json").write_text(json.dumps({"valid": rep.valid, "errors": rep.errors[:50],
                                                       "rows": len(rep.rows), "files": rep.files}, default=str))
    out = {"device": blob("device"), "telemetry": blob("telemetry"), "client_rows": blob(ROWS, ".jsonl"),
           "serve_log": blob("serve.log", ".log")}
    shutil.rmtree(work, ignore_errors=True)
    return out


def telemetry_rows(d: Path | None) -> list[dict]:
    """Every engine row with a trace_id: the `requests` table of each *.sqlite, and *.jsonl lines that carry one."""
    if d is None:
        return []
    rows = []
    for p in sorted(Path(d).glob("*.sqlite")):          # top level only: the timeline lives in timeline/
        con = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        try:
            rows += [dict(r) for r in con.execute("SELECT * FROM requests")]
        except sqlite3.OperationalError:          # no requests table: not per-request rows
            pass
        finally:
            con.close()
    for p in sorted(Path(d).glob("*.jsonl")):
        rows += [r for r in map(json.loads, p.read_text().splitlines()) if isinstance(r, dict) and "trace_id" in r]
    return rows


def load(record: dict, root: Path) -> dict:
    """A record's artifacts back: device meta, telemetry dir (or None), client rows, serve log text."""
    a = (record.get("result") or {}).get("artifacts") or {}

    def path(k: str) -> Path | None:
        return Path(root) / a[k] if a.get(k) else None
    dev, rows, log = path("device"), path("client_rows"), path("serve_log")
    return {"device": json.loads((dev / "meta.json").read_text()) if dev else None,
            "device_samples": dev / "samples.csv" if dev and (dev / "samples.csv").exists() else None,
            "telemetry": path("telemetry"),
            "client_rows": [json.loads(x) for x in rows.read_text().splitlines()] if rows else [],
            "serve_log": log.read_text(errors="replace") if log else None}


def joined(record: dict, root: Path, prefix: str = "bg") -> list[dict]:
    """Each client row with `engine`: the telemetry row whose trace_id is `<prefix>-<row id>` (the client's
    X-Trace-Id), or None."""
    a = load(record, root)
    by_trace = {r["trace_id"]: r for r in telemetry_rows(a["telemetry"])}
    return [{**r, "engine": by_trace.get(f"{prefix}-{r['id']}")} for r in a["client_rows"]]
