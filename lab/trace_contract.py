"""The request-trace contract (lab/TRACE.md): load an engine's trace directory and check every row and file against it."""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

SCHEMA_VERSION = 1
TERMINAL_STATES = frozenset({"ok", "rejected_429", "rejected_400", "expired", "preempted", "error",
                             "cancelled", "aborted"})
EPOCH_MIN = 946_684_800.0           # 2000-01-01: anything smaller is a monotonic reading, not epoch seconds
MAX_DURATION_S = 86_400.0           # a request longer than a day is a unit error (ms or ns)
TPOT_RTOL = 1e-3

REQUIRED = {"schema_version": int, "trace_id": str, "arrival_ts": float, "terminal_state": str,
            "prompt_tokens": int, "tokens_out": int, "total_s": float, "ttft_s": float, "tpot_s": float}
NULLABLE = {"ttft_s", "tpot_s"}
OPTIONAL = {"session_id": str, "config_id": str, "prefill_mode": str, "turn_index": int,
            "queue_wait_s": float, "prefill_s": float, "decode_s": float, "replica_age_s": float,
            "pending_depth": int, "active_size": int, "max_batch_size": int, "concurrent_sessions": int,
            "kv_free_blocks": int, "kv_free_frac": float, "cache_hit_tokens": int, "decode_steps": int,
            "decode_batch_width_max": int, "preempted": int, "decode_batch_width_mean": float}
DURATIONS = ("total_s", "ttft_s", "tpot_s", "queue_wait_s", "prefill_s", "decode_s")
META_KEYS = ("schema_version", "rows_written", "rows_dropped", "closed")


@dataclass
class Report:
    rows: list[dict] = field(default_factory=list)
    files: dict[str, dict] = field(default_factory=dict)     # file name -> its meta
    errors: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.errors

    @property
    def rows_dropped(self) -> int:
        return sum(int(m.get("rows_dropped") or 0) for m in self.files.values())


def _typed(v, t) -> bool:
    if isinstance(v, bool):
        return False
    return isinstance(v, (int, float)) and math.isfinite(v) if t is float else isinstance(v, t)


def check_row(row: dict, now: float | None = None) -> list[str]:
    """Every way `row` breaks the contract, as plain sentences; empty when it conforms."""
    now = time.time() if now is None else now
    errs = [f"missing {k}" for k in REQUIRED if k not in row]
    for k, t in {**REQUIRED, **OPTIONAL}.items():
        v = row.get(k)
        if v is None:
            if k in REQUIRED and k in row and k not in NULLABLE:
                errs.append(f"{k} is null")
        elif not _typed(v, t):
            errs.append(f"{k} is {type(v).__name__}, expected {t.__name__}")
    if errs:
        return errs
    if row["schema_version"] != SCHEMA_VERSION:
        errs.append(f"schema_version {row['schema_version']}, expected {SCHEMA_VERSION}")
    if not row["trace_id"]:
        errs.append("trace_id is empty")
    if row["terminal_state"] not in TERMINAL_STATES:
        errs.append(f"terminal_state {row['terminal_state']!r} not one of {sorted(TERMINAL_STATES)}")
    ts = row["arrival_ts"]
    if ts < EPOCH_MIN:
        errs.append(f"arrival_ts {ts} is not epoch seconds (a monotonic clock?)")
    elif ts > now + MAX_DURATION_S:
        errs.append(f"arrival_ts {ts} is in the future: epoch ms or ns, not seconds?")
    for k in DURATIONS:
        v = row.get(k)
        if v is not None and not 0 <= v <= MAX_DURATION_S:
            errs.append(f"{k} {v} outside [0, {MAX_DURATION_S:.0f}] s (mixed clocks or units?)")
    n, ttft, tpot, total = row["tokens_out"], row["ttft_s"], row["tpot_s"], row["total_s"]
    for k in ("prompt_tokens", "tokens_out"):
        if row[k] < 0:
            errs.append(f"{k} is negative")
    if (n >= 1) != (ttft is not None):
        errs.append(f"ttft_s must be set exactly when tokens_out >= 1 (tokens_out {n}, ttft_s {ttft})")
    if ttft is not None and ttft > total * (1 + 1e-9):
        errs.append(f"ttft_s {ttft} > total_s {total} (mixed units?)")
    if n >= 2 and ttft is not None:
        want = (total - ttft) / (n - 1)
        if tpot is None or abs(tpot - want) > TPOT_RTOL * abs(want) + 1e-9:
            errs.append(f"tpot_s {tpot} != (total_s - ttft_s) / (tokens_out - 1) = {want:.6g}")
    elif tpot is not None:
        errs.append(f"tpot_s {tpot} set with tokens_out {n} (< 2)")
    qw, pf = row.get("queue_wait_s"), row.get("prefill_s")
    if ttft is not None and qw is not None and pf is not None and qw + pf > ttft * (1 + 1e-6) + 1e-9:
        errs.append(f"queue_wait_s + prefill_s = {qw + pf:.6g} > ttft_s {ttft} (mixed units?)")
    frac = row.get("kv_free_frac")
    if frac is not None and not 0 <= frac <= 1:
        errs.append(f"kv_free_frac {frac} outside [0, 1]")
    hit = row.get("cache_hit_tokens")
    if hit is not None and not 0 <= hit <= row["prompt_tokens"]:
        errs.append(f"cache_hit_tokens {hit} outside [0, prompt_tokens {row['prompt_tokens']}]")
    return errs


def _read_sqlite(path: Path) -> tuple[list[dict], dict]:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        rows = [dict(r) for r in conn.execute("SELECT * FROM requests")] if "requests" in tables else None
        meta = dict(tuple(r) for r in conn.execute("SELECT key, value FROM meta")) if "meta" in tables else None
    finally:
        conn.close()
    return rows, meta


def _read_jsonl(path: Path) -> tuple[list[dict], dict | None]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    meta_path = path.with_name(path.stem + ".meta.json")
    return rows, (json.loads(meta_path.read_text()) if meta_path.exists() else None)


def validate(directory: str | Path, now: float | None = None) -> Report:
    """Load every trace file in `directory` and check its meta and each row."""
    d, rep = Path(directory), Report()
    paths = sorted(d.glob("*.sqlite")) + sorted(d.glob("*.jsonl"))
    if not paths:
        rep.errors.append(f"{d}: no *.sqlite or *.jsonl trace files")
    for p in paths:
        try:
            rows, meta = _read_sqlite(p) if p.suffix == ".sqlite" else _read_jsonl(p)
        except (sqlite3.Error, json.JSONDecodeError, UnicodeDecodeError) as e:
            rep.errors.append(f"{p.name}: unreadable ({e})")
            continue
        if rows is None:
            rep.errors.append(f"{p.name}: no requests table")
            continue
        if meta is None:
            rep.errors.append(f"{p.name}: no meta (drop counts must be written, never silent)")
            meta = {}
        for k in META_KEYS:
            if k in meta and not _typed(meta[k], int):
                rep.errors.append(f"{p.name}: meta {k} is {type(meta[k]).__name__}, expected int")
            elif k not in meta and meta:
                rep.errors.append(f"{p.name}: meta missing {k}")
        if meta.get("schema_version") not in (None, SCHEMA_VERSION):
            rep.errors.append(f"{p.name}: meta schema_version {meta['schema_version']}, expected {SCHEMA_VERSION}")
        if meta.get("closed") == 1 and meta.get("rows_written") != len(rows):
            rep.errors.append(f"{p.name}: meta rows_written {meta.get('rows_written')} but {len(rows)} rows")
        rep.files[p.name] = meta
        for i, row in enumerate(rows):
            rep.errors += [f"{p.name} row {i} ({row.get('trace_id')!r}): {e}" for e in check_row(row, now)]
        rep.rows += rows
    return rep


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Check an engine's request-trace directory against lab/TRACE.md")
    ap.add_argument("dir")
    rep = validate(ap.parse_args(argv).dir)
    for e in rep.errors:
        print(e)
    print(f"{'ok' if rep.valid else 'INVALID'}: {len(rep.rows)} rows in {len(rep.files)} files, "
          f"{rep.rows_dropped} dropped, {len(rep.errors)} errors")
    return 0 if rep.valid else 1


if __name__ == "__main__":
    sys.exit(main())
