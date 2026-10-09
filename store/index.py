"""SQL over every run: DuckDB tables rebuilt from the ledger and its blobs. Derived and disposable; the ledger and
blobs are the data, so a new table or column is a rebuild, never a migration.

  records             one row per ledger record (args, config, result, metrics, claim as JSON)
  regimes             one row per regime result of a bench or baseline
  heldout_aggregates  one row per regime of a submit (the guarded aggregates the ledger holds)
  noise               one row per noise record: the run-to-run band per regime
  client_rows         per-request client rows of every bench and equiv (record_id joins records.id)
  engine_rows         the engine's per-request telemetry rows (trace_id = 'bg-' || client_rows.id)
  device_samples      GPU samples over each served window
  kernels, ops        profile bundles' CUDA kernel and operator tables
  heldout_rows        submit's per-request held-out rows: researcher index only, never the agent's

Two files under <ledger>/index/: agent.duckdb (everything but heldout_rows) and all.duckdb (researcher).
"""

from __future__ import annotations

import csv
import io
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Iterable, Iterator

import duckdb

from lab import artifacts, ledger

QUERY_TIMEOUT_S = 60
MAX_ROWS = 200


def _rows_file(rows: Iterable[dict], path: Path) -> int:
    n = 0
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
            n += 1
    return n


def _table(con, name: str, rows: Iterable[dict], tmp: Path) -> int:
    """`rows` as table `name`, columns inferred over every row; an empty table still exists."""
    path = tmp / f"{name}.jsonl"
    n = _rows_file(rows, path)
    if n:
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM read_json(?, format='newline_delimited', "
                    f"sample_size=-1, maximum_object_size=1073741824)", [str(path)])
    else:
        con.execute(f"CREATE TABLE {name} (record_id VARCHAR)")
    return n


def _j(v) -> str | None:
    return None if v is None else json.dumps(v, default=str)


def _records(recs: list[dict]) -> Iterator[dict]:
    for r in recs:
        res = r.get("result") if isinstance(r.get("result"), dict) else {}
        cost = r.get("cost") or {}
        yield {"id": r["id"], "at": r.get("at"), "kind": r["kind"], "run": r.get("run"), "session": r.get("session"),
               "tool": r.get("tool"), "snapshot": r.get("snapshot"), "patch": r.get("patch"),
               "verdict": res.get("verdict"), "split": (r.get("config") or {}).get("split"),
               "tier": (r.get("config") or {}).get("tier") or res.get("tier"),
               "cost_usd": cost.get("usd"), "seconds": cost.get("seconds"), "gpu_seconds": cost.get("gpu_seconds"),
               "args": _j(r.get("args")), "config": _j(r.get("config")), "result": _j(r.get("result")),
               "metrics": _j(r.get("metrics")), "claim": _j(r.get("claim"))}


def _key(r: dict) -> dict:
    return {"record_id": r["id"], "record_kind": r["kind"], "run": r.get("run"), "snapshot": r.get("snapshot")}


def _regimes(recs: list[dict]) -> Iterator[dict]:
    for r in recs:
        res = r.get("result") or {}
        if r["kind"] not in ("bench", "baseline") or res.get("verdict") != "ok":
            continue
        full = {x.get("regime"): x for x in res.get("regimes") or []}
        for name, h in (res.get("metrics") or {}).items():
            yield {**_key(r), "split": (r.get("config") or {}).get("split"), "tier": (r.get("config") or {}).get("tier"),
                   "regime": name, "objective": h.get("objective"), "value": h.get("value"), "better": h.get("better"),
                   "valid": h.get("valid"), "invalid_reasons": _j(h.get("invalid_reasons")), "detail": _j(full.get(name))}


def _heldout(recs: list[dict]) -> Iterator[dict]:
    for r in recs:
        if r["kind"] == "submit":
            for name, m in (r.get("metrics") or {}).items():
                yield {**_key(r), "regime": name, **m}


def _noise(recs: list[dict]) -> Iterator[dict]:
    for r in recs:
        if r["kind"] == "noise":
            d = r.get("result") or {}
            yield {"record_id": r["id"], **{k: v for k, v in d.items() if k != "values"}, "values": _j(d.get("values"))}


def _blob(root: Path, rel: str | None, missing: list) -> Path | None:
    if not rel:
        return None
    p = Path(root) / rel
    if not p.exists():
        missing.append(rel)
        return None
    return p


def _passive(root: Path, recs: list[dict], missing: list) -> tuple[Iterator, Iterator, Iterator]:
    with_art = [r for r in recs if ((r.get("result") or {}).get("artifacts") if isinstance(r.get("result"), dict) else None)]

    def client():
        for r in with_art:
            p = _blob(root, r["result"]["artifacts"].get("client_rows"), missing)
            for line in p.read_text().splitlines() if p else []:
                yield {**_key(r), **json.loads(line)}

    def engine():
        for r in with_art:
            for row in artifacts.telemetry_rows(_blob(root, r["result"]["artifacts"].get("telemetry"), missing)):
                yield {**_key(r), **row}

    def device():
        for r in with_art:
            d = _blob(root, r["result"]["artifacts"].get("device"), missing)
            if not d or not (d / "samples.csv").exists():
                continue
            meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
            text = (d / "samples.csv").read_text(errors="replace")
            if meta.get("source") == "nvidia-smi":          # a CSV with a header: one column per field
                for row in csv.DictReader(io.StringIO(text), skipinitialspace=True):
                    yield {**_key(r), "source": "nvidia-smi", **row}
            else:                                            # DCGM lines stay raw, prefixed with epoch seconds
                for line in text.splitlines():
                    t, _, rest = line.partition(",")
                    yield {**_key(r), "source": meta.get("source"), "epoch_s": t, "line": rest}
    return client(), engine(), device()


def _profiles(root: Path, recs: list[dict], missing: list) -> tuple[Iterator, Iterator]:
    bundles = [(r, _blob(root, (r.get("result") or {}).get("bundle"), missing))
               for r in recs if r["kind"] == "profile" and isinstance(r.get("result"), dict)]

    def kernels():
        for r, b in bundles:
            if b and (b / "kernels.json").exists():
                for row in json.loads((b / "kernels.json").read_text()).get("kernels") or []:
                    yield {**_key(r), **row}

    def ops():
        for r, b in bundles:
            if b and (b / "ops.json").exists():
                for row in json.loads((b / "ops.json").read_text()).get("top") or []:
                    yield {**_key(r), **row}
    return kernels(), ops()


def _heldout_rows(runs: Path) -> Iterator[dict]:
    for p in sorted(Path(runs).glob("*/heldout-private/*/client_rows.jsonl")):
        for line in p.read_text().splitlines():
            yield {"run": p.parents[2].name, "serve": p.parent.name, **json.loads(line)}


def build(root: Path, db: Path, heldout_runs: Path | None = None) -> dict:
    """Rebuild `db` from the ledger at `root`; with `heldout_runs`, add heldout_rows from its run dirs. Returns the
    row count per table and the blobs that were not on disk (pull them to index their rows)."""
    recs = list(ledger.records(root))
    missing: list[str] = []
    db = Path(db)
    db.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=db.parent) as d:
        tmp = Path(d)
        con = duckdb.connect(str(tmp / "new.duckdb"))
        client, engine, device = _passive(root, recs, missing)
        kernels, ops = _profiles(root, recs, missing)
        tables = {"records": _records(recs), "regimes": _regimes(recs), "heldout_aggregates": _heldout(recs),
                  "noise": _noise(recs), "client_rows": client, "engine_rows": engine, "device_samples": device,
                  "kernels": kernels, "ops": ops}
        if heldout_runs is not None:
            tables["heldout_rows"] = _heldout_rows(heldout_runs)
        counts = {name: _table(con, name, rows, tmp) for name, rows in tables.items()}
        con.close()
        os.replace(tmp / "new.duckdb", db)
    return {"tables": counts, "missing_blobs": sorted(set(missing))}


def agent_db(root: Path) -> Path:
    """The agent's index, rebuilt whenever the ledger has grown since the last build."""
    db, stamp = Path(root) / "index" / "agent.duckdb", Path(root) / "index" / "agent.stamp"
    f = Path(root) / "ledger.jsonl"
    now = str(f.stat().st_size if f.exists() else 0)
    if not db.exists() or not stamp.exists() or stamp.read_text() != now:
        build(root, db)
        stamp.write_text(now)
    return db


def query(db: Path, sql: str, limit: int = MAX_ROWS) -> dict:
    """Run `sql` read-only against `db` with no file or network access; at most `limit` rows."""
    con = duckdb.connect(":memory:")
    timer = threading.Timer(QUERY_TIMEOUT_S, con.interrupt)
    try:
        con.execute("ATTACH '" + str(db).replace("'", "''") + "' AS lab (READ_ONLY)")
        con.execute("USE lab")
        con.execute("SET memory_limit = '4GB'")
        con.execute("SET enable_external_access = false")
        con.execute("SET lock_configuration = true")
        timer.start()
        cur = con.execute(sql)
        cols = [c[0] for c in cur.description] if cur.description else []
        rows = cur.fetchmany(limit + 1)
        return {"columns": cols, "rows": [list(r) for r in rows[:limit]], "truncated": len(rows) > limit}
    finally:
        timer.cancel()
        con.close()
