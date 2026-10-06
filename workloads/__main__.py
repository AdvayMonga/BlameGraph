"""python -m workloads fetch BENCH[,BENCH...]|all [--tokenizer NAME]   (into data/workloads/)
python -m workloads list"""
from __future__ import annotations

import argparse
import importlib
import json

from . import BENCHES, DATA, fetch


def main():
    ap = argparse.ArgumentParser(prog="workloads", description="fetch frontier benchmark request data")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch"); f.add_argument("benches", help=f"comma-separated, or 'all': {', '.join(BENCHES)}")
    f.add_argument("--tokenizer", help="re-count build_prompt_tokens with this HF tokenizer (chat template applied)")
    sub.add_parser("list")
    a = ap.parse_args()
    if a.cmd == "list":
        man = DATA / "manifest.json"
        for name in BENCHES:
            doc = importlib.import_module(f"workloads.{name}").__doc__.strip().splitlines()[0]
            print(f"{name:24} {doc}")
        if man.exists():
            print("\nfetched:", json.dumps({k: {"n": v["n"], "fetched_at": v["fetched_at"]} for k, v in json.loads(man.read_text()).items()}, indent=1))
        return
    names = list(BENCHES) if a.benches == "all" else a.benches.split(",")
    unknown = [n for n in names if n not in BENCHES]
    if unknown:
        ap.error(f"unknown bench(es): {', '.join(unknown)}")
    for name in names:
        out = fetch(importlib.import_module(f"workloads.{name}"), a.tokenizer)
        print(f"{name}: {sum(1 for _ in open(out))} records -> {out}")


if __name__ == "__main__":
    main()
