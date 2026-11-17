"""lab correctness reference --out DIR [--tasks a,b] [--target T]   (the model in transformers, on the worker's GPU)
lab correctness candidate --url URL --out FILE.json [--ref DIR] [--api vllm|sglang|none] [--target T]
lab correctness verdict A.json [B.json ...] [--target T]   (re-judge saved results, no server)"""
from __future__ import annotations

import argparse
import os

from evals.targets import target


def main():
    ap = argparse.ArgumentParser(prog="lab correctness", description="correctness gate against live servers")
    target.add_argument(ap)
    sub = ap.add_subparsers(dest="step", required=True)
    r = sub.add_parser("reference"); r.add_argument("--out", required=True)
    r.add_argument("--tasks", help="comma-separated (default: the target's)")
    c = sub.add_parser("candidate"); c.add_argument("--url", required=True); c.add_argument("--out", required=True)
    c.add_argument("--ref", help="reference outputs dir (default: the target's reference.dir)")
    c.add_argument("--api", choices=("vllm", "sglang", "none"), help="logprobs API (default: the target's engine.api)")
    c.add_argument("--concurrency", type=int, default=128)
    v = sub.add_parser("verdict"); v.add_argument("results", nargs="+")
    a = ap.parse_args()
    if a.target:
        os.environ["LAB_TARGET"] = a.target        # the worker's jobs read the target from the environment
    t = target.load()
    from lab.gate.correctness import client, run
    from lab.gate.correctness.gate import Thresholds
    th = Thresholds(max_kl=t.max_kl, min_length_ratio=t.min_length_ratio)
    enc = lambda: run.HFEncoder(t.model, t.chat_kwargs)
    if a.step == "reference":
        from pathlib import Path
        from lab.gate.correctness import hf
        from lab.infra.runner import worker
        hf.build(Path(a.out), tuple(a.tasks.split(",")) if a.tasks else t.tasks, worker.load())
        print((Path(a.out) / "meta.json").read_text())
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
