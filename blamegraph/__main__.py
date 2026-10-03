"""python -m blamegraph feedback PATH [--scenario A] [--run RUN_ID] [--agent-text]
python -m blamegraph equiv reference --url URL --out DIR [--tasks mmlu_pro,math,code,needle]
python -m blamegraph equiv candidate --ref DIR --url URL --out FILE.json [--thresholds FILE]
python -m blamegraph equiv calibrate --good A.json B.json --bad C.json --out thresholds.json"""
from __future__ import annotations

import argparse
import json

from .feedback import feedback, render_for_agent

MODEL = "Qwen/Qwen3-30B-A3B"


def main():
    ap = argparse.ArgumentParser(prog="blamegraph")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("feedback", help="feedback for one finished session (run dir, session .jsonl, or lab ledger)")
    p.add_argument("path")
    p.add_argument("--scenario", default=None)
    p.add_argument("--run", default=None, help="lab ledger: the run to report (default: the latest)")
    p.add_argument("--agent-text", action="store_true", help="print only the agent-facing half as plain text")
    e = sub.add_parser("equiv", help="correctness gate against live servers").add_subparsers(dest="step", required=True)
    r = e.add_parser("reference"); r.add_argument("--url", required=True); r.add_argument("--out", required=True)
    r.add_argument("--model", default=MODEL); r.add_argument("--tasks", default="mmlu_pro,math,code,needle")
    r.add_argument("--div-n", type=int, default=400); r.add_argument("--concurrency", type=int, default=16)
    c = e.add_parser("candidate"); c.add_argument("--ref", required=True); c.add_argument("--url", required=True)
    c.add_argument("--out", required=True); c.add_argument("--model", default=MODEL); c.add_argument("--thresholds")
    c.add_argument("--concurrency", type=int, default=16)
    k = e.add_parser("calibrate"); k.add_argument("--good", nargs="+", required=True); k.add_argument("--bad", nargs="+", required=True)
    k.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "feedback":
        fb = feedback(a.path, a.scenario, a.run)
        print(render_for_agent(fb) if a.agent_text else json.dumps(fb, indent=1, default=str))
        return
    from .equivalence import run, tasks
    from .equivalence.gate import Thresholds
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
