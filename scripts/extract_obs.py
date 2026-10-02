"""Run the Haiku observation extractor. Usage: extract_obs.py [--pilot N] [--runs K]"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from blamegraph.extract import extract_runs, load_cache, load_env  # noqa: E402
from blamegraph.traces import iter_runs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", type=int, default=None, help="only extract this many steps")
    ap.add_argument("--runs", type=int, default=None, help="only the first K runs")
    ap.add_argument("--concurrency", type=int, default=8)
    a = ap.parse_args()
    load_env()
    runs = list(iter_runs(limit=a.runs))
    usage = asyncio.run(extract_runs(runs, concurrency=a.concurrency, limit_steps=a.pilot))
    print(json.dumps(usage))
    if a.pilot:
        cache = load_cache()
        for (rid, i), d in list(cache.items())[-min(a.pilot, 12):]:
            print(rid, i, json.dumps(d["data"])[:220])


if __name__ == "__main__":
    main()
