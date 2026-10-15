"""python -m store push | pull | index | sql | archive (the remote is LAB_STORE; see store/sync.py for its layout)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from lab import ledger
from lab.engine import ENV_ROOT
from store import index, sync
from store.remote import open_remote

RUNS, CORPUS = ENV_ROOT / "lab" / "runs", ENV_ROOT / "corpus"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m store")
    ap.add_argument("--root", type=Path, default=ledger.ROOT, help="the ledger dir")
    ap.add_argument("--remote", help="s3://<bucket> or a directory (default LAB_STORE)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("push", help="ledger, blobs, held-out passive data and held-out traces")
    pl = sub.add_parser("pull", help="the ledger")
    pl.add_argument("--blobs", action="store_true", help="also every blob missing locally")
    pl.add_argument("--corpus", action="store_true", help="also the held-out traces of the corpus in corpus/")
    sub.add_parser("index", help="rebuild the researcher's index, held-out rows included (index/all.duckdb)")
    sq = sub.add_parser("sql", help="query an index")
    sq.add_argument("sql")
    sq.add_argument("--agent", action="store_true", help="the agent's index instead of the researcher's")
    ar = sub.add_parser("archive", help="keep files or directories as compressed objects named by content hash")
    ar.add_argument("paths", nargs="+", type=Path)
    ar.add_argument("--heldout", action="store_true", help="held-out material: under heldout/")
    a = ap.parse_args(argv)

    if a.cmd == "index":
        print(json.dumps(index.build(a.root, a.root / "index" / "all.duckdb", heldout_runs=RUNS), indent=1))
        return 0
    if a.cmd == "sql":
        db = index.agent_db(a.root) if a.agent else a.root / "index" / "all.duckdb"
        if not db.exists():
            print("no index yet: run `python -m store index`", file=sys.stderr)
            return 1
        out = index.query(db, a.sql, limit=10_000)
        print("\t".join(out["columns"]))
        for row in out["rows"]:
            print("\t".join(str(v) for v in row))
        return 0
    remote = open_remote(a.remote)
    if a.cmd == "push":
        print(json.dumps(sync.push(a.root, remote, runs=RUNS, corpus=CORPUS)))
    elif a.cmd == "pull":
        print(json.dumps(sync.pull(a.root, remote, blobs=a.blobs, corpus=CORPUS if a.corpus else None)))
    else:
        for p in a.paths:
            print(sync.archive(p, remote, heldout=a.heldout))
    return 0


if __name__ == "__main__":
    sys.exit(main())
