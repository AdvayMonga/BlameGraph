"""python -m regimes run REGIME[,REGIME...]|all --url URL [--split seen|heldout] [--tier short|full] [--seed N]
                       [--corpus DIR] [--tokenizer] [--out FILE] [--target T]
python -m regimes cold-start --url URL --cmd "launch command" [--expect 42] [--no-warm] [--out FILE] [--target T]
The model, chat kwargs, latency limits and corpus come from the target spec ($LAB_TARGET)."""
from __future__ import annotations

import argparse
import json
import sys
import time

from lab import target

from . import suite
from .workload import corpus_version


def main():
    ap = argparse.ArgumentParser(prog="regimes", description="measure a live server under the eight regimes")
    target.add_argument(ap)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("regimes", help=f"comma-separated, or 'all': {', '.join(suite.REGIMES)}")
    r.add_argument("--url", required=True)
    r.add_argument("--corpus", help="corpus dir (default: the target's; synthetic prompts if it does not exist)")
    r.add_argument("--split", choices=("seen", "heldout"), default="seen")
    r.add_argument("--tier", choices=tuple(suite.TIERS), default="short")
    r.add_argument("--seed", type=int, default=0); r.add_argument("--timeout", type=float, default=600.0)
    r.add_argument("--tokenizer", help="re-count output tokens with this HF tokenizer (default: server usage)")
    r.add_argument("--out")
    c = sub.add_parser("cold-start")
    c.add_argument("--url", required=True); c.add_argument("--cmd", required=True)
    c.add_argument("--expect", default="42"); c.add_argument("--timeout", type=float, default=1800.0)
    c.add_argument("--no-warm", action="store_true"); c.add_argument("--log-dir"); c.add_argument("--out")
    a = ap.parse_args()
    t = target.load(a.target)

    if a.cmd == "cold-start":
        res = suite.cold_start(a.url, t.model, a.cmd, a.expect, a.timeout, warm=not a.no_warm, log_dir=a.log_dir)
        report = {"model": t.model, "url": a.url, "at": time.strftime("%Y-%m-%dT%H:%M:%S"), "results": [res]}
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
        ctx = suite.Ctx.from_target(t, a.url, split=a.split, tier=a.tier, seed=a.seed, timeout=a.timeout, count_tokens=count)
        if a.corpus:
            ctx.corpus = a.corpus
        results = []
        for n in names:
            res = suite.REGIMES[n](ctx)
            results.append(res)
            print(f"{n}: {res['objective']} = {res['value']}" + ("" if res["valid"] else f"  INVALID: {res['invalid_reasons']}"),
                  file=sys.stderr)
        report = {"model": t.model, "target": t.name, "url": a.url, "tier": a.tier, "split": a.split, "seed": a.seed,
                  "corpus_version": corpus_version(ctx.corpus), "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                  "results": results}
    text = json.dumps(report, indent=1, default=str)
    if a.out:
        open(a.out, "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
