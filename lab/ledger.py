"""ledger: append-only record of every tool call (format: lab/LEDGER.md)."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

SCHEMA = 1
ROOT = Path(__file__).resolve().parents[1] / "lab" / "ledger"
KINDS = {"test", "equiv", "bench", "profile", "submit", "session", "note", "baseline", "noise", "hypothesis",
         "reflection", "kernels"}
LEGACY_KINDS = {"finding"}             # in older ledgers: read and merged, never written
WRITER_FIELDS = {"id", "at", "schema"}
# A held-out record is one aggregate per metric and nothing else: these fields, scalar values.
HELDOUT_FIELDS = {"kind", "config", "metrics", "run", "session", "snapshot", "snapshot_blob", "patch",
                  "tool", "args", "cost", "claim", "experiment"}
HELDOUT_METRIC_KEYS = {"base", "new", "delta_pct", "band_pct", "verdict",
                       "ceiling", "pct_of_ceiling_base", "pct_of_ceiling_new", "ceiling_from", "reference_missing"}
EXPERIMENT_FIELDS = {"hypothesis", "expect", "actual", "actual_missing", "gap", "gap_pct_of_expected", "noise_band_pct",
                     "bundle", "independence"}
SCALAR = (int, float, str, bool, type(None))


class LedgerError(ValueError):
    pass


def _file(root: Path) -> Path:
    return Path(root) / "ledger.jsonl"


@contextlib.contextmanager
def _lock(root: Path):
    """One lock file per ledger; parallel sessions share it. Held across whole read-then-write steps."""
    Path(root).mkdir(parents=True, exist_ok=True)
    with (Path(root) / "ledger.lock").open("a") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def _mentions_heldout(obj: Any) -> bool:
    """A `"split": "heldout"` anywhere in a nested value, so held-out data can't slip in under another field."""
    if isinstance(obj, dict):
        return obj.get("split") == "heldout" or any(_mentions_heldout(v) for v in obj.values())
    if isinstance(obj, list):
        return any(_mentions_heldout(v) for v in obj)
    return False


def _check(record: dict, kinds: set[str] = KINDS) -> None:
    """Integrity rules only: known kind, writer-owned fields, no held-out detail beyond one aggregate per metric."""
    if record.get("kind") not in kinds:
        raise LedgerError(f"unknown kind {record.get('kind')!r}; one of {sorted(kinds)}")
    if WRITER_FIELDS & record.keys():
        raise LedgerError(f"{sorted(WRITER_FIELDS & record.keys())} are set by the writer")
    config = record.get("config")
    if config is not None and not isinstance(config, dict):
        raise LedgerError("config must be an object")
    # `claim` is untrusted text that may name the split; everything else marks the record held-out
    if not _mentions_heldout({k: v for k, v in record.items() if k != "claim"}):
        return
    if (config or {}).get("split") != "heldout":
        raise LedgerError("held-out data must be declared with config.split == 'heldout' and written as "
                          "top-level config + metrics, not nested under another field")
    extra = record.keys() - HELDOUT_FIELDS
    if extra:
        raise LedgerError(f"held-out records may only carry {sorted(HELDOUT_FIELDS)}; got {sorted(extra)}")
    metrics = record.get("metrics") or {}
    if not isinstance(metrics, dict):
        raise LedgerError("metrics must be an object")
    exp = record.get("experiment")
    if exp is not None:                      # the agent's prediction and one number derived from an aggregate
        if not isinstance(exp, dict) or exp.keys() - EXPERIMENT_FIELDS:
            raise LedgerError(f"a held-out experiment may only carry {sorted(EXPERIMENT_FIELDS)}")
        flat = {k: v for k, v in exp.items() if k not in ("expect", "bundle")}
        if not all(isinstance(h, str) for h in exp.get("bundle") or []) \
                or not all(isinstance(v, SCALAR) for v in flat.values()) or not isinstance(exp.get("expect", {}), dict) \
                or not all(isinstance(v, SCALAR) for v in exp.get("expect", {}).values()):
            raise LedgerError("a held-out experiment carries scalar values only")
    for name, m in metrics.items():
        if (not isinstance(m, dict) or not m.keys() <= HELDOUT_METRIC_KEYS
                or not all(isinstance(v, SCALAR) for v in m.values())):
            raise LedgerError(f"held-out metric {name!r} may only carry scalar "
                              f"{sorted(HELDOUT_METRIC_KEYS)}")


def _stamp(record: dict) -> dict:
    now = datetime.now(timezone.utc)
    return {"id": f"ev-{now:%Y%m%d}-{uuid.uuid4().hex[:12]}",
            "at": now.isoformat(timespec="seconds"), "schema": SCHEMA, **record}


def _write(stored: dict, root: Path) -> None:
    """One line, one os.write, so a reader never sees half a record. Caller holds the lock."""
    try:
        line = json.dumps(stored) + "\n"
    except TypeError as e:
        raise LedgerError(f"record is not JSON (lossless means no coercion): {e}")
    fd = os.open(_file(root), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(fd, line.encode())
    finally:
        os.close(fd)


def append(record: dict, root: Path = ROOT) -> dict:
    """Validate, stamp id/at/schema, append one line; return the stored record."""
    _check(record)
    stored = _stamp(record)
    with _lock(root):
        _write(stored, root)
    return stored


def _hash(src: Path | bytes) -> str:
    """Content hash; directories hash each file's path and sha with length prefixes, so layouts cannot collide."""
    h = hashlib.sha256()
    if isinstance(src, bytes):
        h.update(b"bytes\0" + src)
    elif Path(src).is_dir():
        h.update(b"dir\0")
        for p in sorted(Path(src).rglob("*")):
            if p.is_file():
                name = str(p.relative_to(src)).encode()
                h.update(f"{len(name)}\0".encode() + name + hashlib.sha256(p.read_bytes()).digest())
    else:
        h.update(b"file\0" + Path(src).read_bytes())
    return h.hexdigest()


def put_blob(src: Path | bytes, root: Path = ROOT, suffix: str = "") -> str:
    """Store a file, directory or bytes by content hash; return its path relative to `root`."""
    rel = f"blobs/{_hash(src)[:24]}{suffix}"
    dest = Path(root) / rel
    with _lock(root):
        if dest.exists():
            return rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + f".tmp-{os.getpid()}")
        if isinstance(src, bytes):
            tmp.write_bytes(src)
        elif Path(src).is_dir():
            shutil.copytree(src, tmp)
        else:
            shutil.copyfile(src, tmp)
        os.replace(tmp, dest)          # a blob exists whole or not at all
    return rel


def records(root: Path = ROOT, **match: Any) -> Iterator[dict]:
    """Every record in write order, optionally only those whose top-level fields equal `match`.

    A torn final line (a writer killed mid-append) is skipped with a warning; a bad line anywhere
    else is corruption and raises.
    """
    path = _file(root)
    if not path.exists():
        return
    lines = path.read_text().split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    for i, line in enumerate(lines):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                print(f"ledger: skipping torn final line in {path}", file=sys.stderr)
                return
            raise LedgerError(f"{path}:{i + 1} is not JSON")
        if all(r.get(k) == v for k, v in match.items()):
            yield r


def merge(src: Path, root: Path = ROOT) -> int:
    """Records and blobs of another ledger dir (one fetched from a VM) that `root` lacks, in their order; returns
    the records added. Record ids and blob names are unique, so merging twice adds nothing. Every new record passes
    the same integrity check as an appended one (it comes from the VM, where agent code runs), or nothing is merged."""
    with _lock(root):
        have = {r["id"] for r in records(root)}
        new = [r for r in records(src) if r["id"] not in have]
        for r in new:
            if WRITER_FIELDS - r.keys():
                raise LedgerError(f"record {r.get('id')!r} lacks {sorted(WRITER_FIELDS - r.keys())}")
            try:
                _check({k: v for k, v in r.items() if k not in WRITER_FIELDS}, KINDS | LEGACY_KINDS)
            except LedgerError as e:
                raise LedgerError(f"record {r['id']} refused, nothing merged: {e}") from None
        for p in sorted((Path(src) / "blobs").iterdir()) if (Path(src) / "blobs").is_dir() else []:
            dest = Path(root) / "blobs" / p.name
            if dest.exists() or ".tmp-" in p.name:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(dest.name + f".tmp-{os.getpid()}")
            shutil.copytree(p, tmp) if p.is_dir() else shutil.copyfile(p, tmp)
            os.replace(tmp, dest)
        for r in new:                      # blobs first: a record never names a blob that is not there yet
            _write(r, root)
    return len(new)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m lab.ledger")
    ap.add_argument("--root", type=Path, default=ROOT)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ls = sub.add_parser("list", help="print records as JSONL")
    for k in ("kind", "session", "snapshot"):
        ls.add_argument(f"--{k}")
    sub.add_parser("show", help="pretty-print one record").add_argument("id")
    sub.add_parser("merge", help="add another ledger dir's new records and blobs").add_argument("src", type=Path)
    a = ap.parse_args(argv)

    if a.cmd == "list":
        match = {k: getattr(a, k) for k in ("kind", "session", "snapshot") if getattr(a, k)}
        for r in records(a.root, **match):
            print(json.dumps(r))
    elif a.cmd == "show":
        hit = next(records(a.root, id=a.id), None)
        if hit is None:
            print(f"no record {a.id}", file=sys.stderr)
            return 1
        print(json.dumps(hit, indent=1))
    else:
        print(f"merged {merge(a.src, a.root)} records")
    return 0


if __name__ == "__main__":
    sys.exit(main())
