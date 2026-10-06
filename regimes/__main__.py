"""python -m regimes run REGIME[,REGIME...]|all --url URL [--corpus DIR] [--split seen|heldout] [--tier short|full]
                       [--seed N] [--tokenizer NAME] [--out FILE]
python -m regimes cold-start --url URL --cmd "launch command" [--expect 42] [--no-warm] [--out FILE]"""
from __future__ import annotations

import argparse
import json
import sys
import time

from . import suite
from .workload import corpus_version

MODEL = "Qwen/Qwen3-30B-A3B"


def main():
    ap = argparse.ArgumentParser(prog="regimes", description="measure a live server under the eight regimes")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("regimes", help=f"comma-separated, or 'all': {', '.join(suite.REGIMES)}")
    r.add_argument("--url", required=True); r.add_argument("--model", default=MODEL)
    r.add_argument("--corpus", help="inference-server corpus/ dir (default: synthetic prompts)")
    r.add_argument("--split", choices=("seen", "heldout"), default="seen")
    r.add_argument("--tier", choices=tuple(suite.TIERS), default="short")
    r.add_argument("--seed", type=int, default=0); r.add_argument("--timeout", type=float, default=600.0)
    r.add_argument("--tokenizer", help="re-count output tokens with this HF tokenizer (default: server usage)")
    r.add_argument("--out")
    c = sub.add_parser("cold-start")
    c.add_argument("--url", required=True); c.add_argument("--model", default=MODEL); c.add_argument("--cmd", required=True)
    c.add_argument("--expect", default="42"); c.add_argument("--timeout", type=float, default=1800.0)
    c.add_argument("--no-warm", action="store_true"); c.add_argument("--log-dir"); c.add_argument("--out")
    a = ap.parse_args()

    if a.cmd == "cold-start":
        res = suite.cold_start(a.url, a.model, a.cmd, a.expect, a.timeout, warm=not a.no_warm, log_dir=a.log_dir)
        report = {"model": a.model, "url": a.url, "at": time.strftime("%Y-%m-%dT%H:%M:%S"), "results": [res]}
    else:
        names = list(suite.REGIMES) if a.regimes == "all" else a.regimes.split(",")
        unknown = [n for n in names if n not in suite.REGIMES]
        if unknown:
            ap.error(f"unknown regime(s) {unknown}; one of {list(suite.REGIMES)}")
        count = None
        if a.tokenizer:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(a.tokenizer)
            count = lambda s: tok.encode(s, add_special_tokens=False)
        ctx = suite.Ctx(a.url, a.model, a.corpus, a.split, a.tier, a.seed, a.timeout, count)
        results = []
        for n in names:
            res = suite.REGIMES[n](ctx)
            results.append(res)
            print(f"{n}: {res['objective']} = {res['value']}" + ("" if res["valid"] else f"  INVALID: {res['invalid_reasons']}"),
                  file=sys.stderr)
        report = {"model": a.model, "url": a.url, "tier": a.tier, "split": a.split, "seed": a.seed,
                  "corpus_version": corpus_version(a.corpus), "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                  "results": results}
    text = json.dumps(report, indent=1, default=str)
    if a.out:
        open(a.out, "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
