"""python -m correctness reference --url URL --out DIR [--tier dev|full] [--tasks a,b] [--target T]
python -m correctness candidate --url URL --out FILE.json [--ref DIR] [--api vllm|sglang|none] [--target T]
python -m correctness verdict A.json [B.json ...] [--target T]   (re-judge saved results, no server)
The model, chat kwargs, tasks, reference dir, logprobs API and policy come from the target spec ($LAB_TARGET)."""
from __future__ import annotations

import argparse
import json

from lab import target


def main():
    ap = argparse.ArgumentParser(prog="correctness", description="correctness gate against live servers")
    target.add_argument(ap)
    sub = ap.add_subparsers(dest="step", required=True)
    r = sub.add_parser("reference"); r.add_argument("--url", required=True); r.add_argument("--out", required=True)
    r.add_argument("--tasks", help="comma-separated (default: the target's)")
    r.add_argument("--tier", choices=("dev", "full"), default="dev", help="dev: ~1.6k items; full: ~18.6k, for final tests")
    r.add_argument("--div-n", type=int, default=400); r.add_argument("--concurrency", type=int, default=16)
    c = sub.add_parser("candidate"); c.add_argument("--url", required=True); c.add_argument("--out", required=True)
    c.add_argument("--ref", help="reference outputs dir (default: the target's reference.dir)")
    c.add_argument("--api", choices=("vllm", "sglang", "none"), help="logprobs API (default: the target's engine.api)")
    c.add_argument("--concurrency", type=int, default=16)
    v = sub.add_parser("verdict"); v.add_argument("results", nargs="+")
    a = ap.parse_args()
    t = target.load(a.target)
    from correctness import client, run, tasks
    from correctness.gate import Thresholds
    th = Thresholds(min_score_ratio=t.min_score_ratio, min_length_ratio=t.min_length_ratio)
    enc = lambda: run.HFEncoder(t.model, t.chat_kwargs)
    if a.step == "reference":
        api = client.Api.from_target(t.reference, t) if t.reference else client.Api("vllm", t.chat_kwargs)
        names = tuple(a.tasks.split(",")) if a.tasks else t.tasks
        meta = run.reference(a.url, t.model, a.out, tasks.load(names, tier=a.tier), enc(), div_n=a.div_n,
                             concurrency=a.concurrency, api=api)
        print(json.dumps(meta, indent=1))
    elif a.step == "candidate":
        ref = a.ref or (str(t.reference_dir) if t.reference_dir else None)
        if not ref:
            ap.error("no reference dir: pass --ref or set reference.dir in the target")
        api = client.Api(a.api or t.engine.api, t.chat_kwargs)
        res = run.candidate(ref, a.url, t.model, a.out, enc(), th, concurrency=a.concurrency, api=api)
        print(res["verdict"].upper(), *res["reasons"], sep="\n  ")
    else:
        for path in a.results:
            res = run.verdict_file(path, th)
            print(path, res["verdict"].upper(), *res["reasons"], sep="\n  ")


if __name__ == "__main__":
    main()
