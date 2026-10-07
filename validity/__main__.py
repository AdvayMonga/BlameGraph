"""python -m validity repeat --url URL [--n 20] [--api chat|completions] [--target T]"""
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
    a = ap.parse_args()
    t = target.load(a.target)
    res = repeat_audit(a.url, t.model, PROMPTS[:a.n], api=a.api, chat_kwargs=t.chat_kwargs)
    print("PASS" if res["passed"] else "FAIL", *res["reasons"], json.dumps(res["evidence"]), sep="\n  ")


if __name__ == "__main__":
    main()
