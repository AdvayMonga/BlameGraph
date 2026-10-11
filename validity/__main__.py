"""python -m validity repeat --url URL [--n 20] [--api chat|completions] [--target T]
python -m validity peaks [--seconds 3] [--out DIR]   (measures this GPU's bandwidth and bf16 peaks; needs torch+CUDA)"""
from __future__ import annotations

import argparse
import json

from lab import target
from validity.repeat_audit import PROMPTS, repeat_audit


def main():
    ap = argparse.ArgumentParser(prog="validity", description="validity audits against a live server")
    target.add_argument(ap)
    sub = ap.add_subparsers(dest="kind", required=True)
    q = sub.add_parser("repeat"); q.add_argument("--url", required=True)
    q.add_argument("--n", type=int, default=20); q.add_argument("--api", choices=("chat", "completions"), default="chat")
    pk = sub.add_parser("peaks"); pk.add_argument("--seconds", type=float, default=3.0); pk.add_argument("--out")
    a = ap.parse_args()
    if a.kind == "peaks":
        from pathlib import Path
        from validity import peaks
        p = peaks.measure(a.seconds)
        print(peaks.write(p, Path(a.out) if a.out else None), json.dumps(p.__dict__), sep="\n")
        return
    t = target.load(a.target)
    res = repeat_audit(a.url, t.model, PROMPTS[:a.n], api=a.api, chat_kwargs=t.chat_kwargs)
    print("PASS" if res["passed"] else "FAIL", *res["reasons"], json.dumps(res["evidence"]), sep="\n  ")


if __name__ == "__main__":
    main()
