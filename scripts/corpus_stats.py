"""One row per run: outcome + cheap code-derived process counters. Writes data/derived/runs.csv."""
from __future__ import annotations

import csv
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from blamegraph.traces import iter_runs  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "data" / "derived"
OUT.mkdir(parents=True, exist_ok=True)

EVAL_RE = re.compile(r"evaluate\.py")
QUICK_RE = re.compile(r"evaluate\.py[^\n|;&]*--quick")
TIMER_RE = re.compile(r"timer\.sh")


def hms_to_min(s: str | None) -> float | None:
    if not s:
        return None
    h, m, sec = s.split(":")
    return int(h) * 60 + int(m) + int(sec) / 60


def row_for(run) -> dict:
    steps = run.steps()
    out_sizes = sorted(len(s.output) for s in steps) or [0]
    cmds = [s.cmd for s in steps]
    frameworks = Counter()
    for t in cmds:
        for fw in ("vllm", "sglang", "tgi", "tensorrt", "trtllm"):
            if fw in t.lower():
                frameworks[fw] += 1
    metric, reason = run.primary_metric
    qc = (run.metrics or {}).get("quality_check") or {}
    ds = (qc.get("datasets") or {}).get("mmlu_pro") or {}
    return dict(
        run_id=run.run_id, agent=run.agent, scenario=run.scenario, harness=run.harness,
        time_min=hms_to_min(run.meta.get("time_taken")),
        flagged=int(run.flagged), has_metrics=int(run.metrics is not None),
        gate_pass=int(run.gate_passed), scored=int(run.scored), speedup=run.speedup,
        mmlu_ratio=ds.get("ratio"), primary=metric if metric > 1e-6 else None, primary_reason=reason,
        n_events=len(run.events), n_tool_calls=len(steps),
        n_text=sum(1 for _ in run.assistant_text()),
        n_thinking=sum(1 for e in run.events if e.type == "thinking"),
        eval_mentions=sum(1 for t in cmds if EVAL_RE.search(t)), quick_mentions=sum(1 for t in cmds if QUICK_RE.search(t)),
        timer_calls=sum(1 for t in cmds if TIMER_RE.search(t)),
        fw_vllm=frameworks["vllm"], fw_sglang=frameworks["sglang"], fw_tgi=frameworks["tgi"],
        fw_trt=frameworks["tensorrt"] + frameworks["trtllm"],
        out_p50=out_sizes[len(out_sizes) // 2], out_p90=out_sizes[int(len(out_sizes) * 0.9)],
        out_max=out_sizes[-1], out_total=sum(out_sizes),
        approx_tokens=sum(len(repr(e.raw)) for e in run.events) // 4,
    )


def main():
    rows = [row_for(r) for r in iter_runs()]
    with open(OUT / "runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows)} rows to {OUT / 'runs.csv'}")


if __name__ == "__main__":
    main()
