"""python -m correctness reference --url URL --out DIR [--tasks mmlu_pro,math,code,needle]
python -m correctness candidate --ref DIR --url URL --out FILE.json [--thresholds FILE]
python -m correctness calibrate --good A.json B.json --bad C.json --out thresholds.json"""
from __future__ import annotations

import argparse
import json

MODEL = "Qwen/Qwen3-30B-A3B"


def main():
    ap = argparse.ArgumentParser(prog="correctness", description="correctness gate against live servers")
    sub = ap.add_subparsers(dest="step", required=True)
    r = sub.add_parser("reference"); r.add_argument("--url", required=True); r.add_argument("--out", required=True)
    r.add_argument("--model", default=MODEL); r.add_argument("--tasks", default="mmlu_pro,math,code,needle")
    r.add_argument("--div-n", type=int, default=400); r.add_argument("--concurrency", type=int, default=16)
    c = sub.add_parser("candidate"); c.add_argument("--ref", required=True); c.add_argument("--url", required=True)
    c.add_argument("--out", required=True); c.add_argument("--model", default=MODEL); c.add_argument("--thresholds")
    c.add_argument("--concurrency", type=int, default=16)
    k = sub.add_parser("calibrate"); k.add_argument("--good", nargs="+", required=True); k.add_argument("--bad", nargs="+", required=True)
    k.add_argument("--out", required=True)
    a = ap.parse_args()
    from correctness import run, tasks
    from correctness.gate import Thresholds
    if a.step == "reference":
        print(json.dumps(run.reference(a.url, a.model, a.out, tasks.load(tuple(a.tasks.split(","))), run.HFEncoder(a.model),
                                       div_n=a.div_n, concurrency=a.concurrency), indent=1))
    elif a.step == "candidate":
        th = Thresholds(**json.loads(open(a.thresholds).read())) if a.thresholds else None
        res = run.candidate(a.ref, a.url, a.model, a.out, run.HFEncoder(a.model), th, concurrency=a.concurrency)
        print("PASS" if res["passed"] else "FAIL", *res["reasons"], sep="\n  ")
    else:
        print(json.dumps(run.calibrate_files(a.good, a.bad, a.out).__dict__, indent=1))


if __name__ == "__main__":
    main()
