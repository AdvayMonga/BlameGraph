"""python -m validity repeat --url URL [--model M] [--n 20] [--api chat|completions]"""
from __future__ import annotations

import argparse
import json

from validity.repeat_audit import PROMPTS, repeat_audit

MODEL = "Qwen/Qwen3-30B-A3B"


def main():
    ap = argparse.ArgumentParser(prog="validity", description="validity audits against a live server")
    sub = ap.add_subparsers(dest="kind", required=True)
    q = sub.add_parser("repeat"); q.add_argument("--url", required=True); q.add_argument("--model", default=MODEL)
    q.add_argument("--n", type=int, default=20); q.add_argument("--api", choices=("chat", "completions"), default="chat")
    a = ap.parse_args()
    res = repeat_audit(a.url, a.model, PROMPTS[:a.n], api=a.api)
    print("PASS" if res["passed"] else "FAIL", *res["reasons"], json.dumps(res["evidence"]), sep="\n  ")


if __name__ == "__main__":
    main()
