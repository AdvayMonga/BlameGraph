"""Build System-One (Laya) training sets from the cached model labels. No human labels involved.

Task A (extraction gate): state = trimmed tool output (<= ~900 tokens), question = "Does this output contain a
benchmark result the agent observed?", label = Haiku's is_benchmark_result (8,876 items).
Task B (judge): state = judge window (truncated), question = the judge question, label = Sonnet's answer.
Splits are by run_id so no run leaks across train/val/test.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from blamegraph.experiment import build_log  # noqa: E402
from blamegraph.extract import candidates, load_cache, obs_by_run  # noqa: E402
from blamegraph.judge import QUESTIONS, build_instances  # noqa: E402
from blamegraph.judge import load_cache as load_judgments  # noqa: E402
from blamegraph.traces import iter_runs  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "data" / "laya"
MAX_CHARS = 3600   # ~900 tokens; Laya checkpoints read 512-1024 tokens


def split_of(run_id: str) -> str:
    h = int(hashlib.sha1(run_id.encode()).hexdigest(), 16) % 10
    return "test" if h < 2 else ("val" if h < 3 else "train")


def shrink(text: str, cap: int = MAX_CHARS) -> str:
    """Keep the metric-bearing middle if too long: head + tail."""
    if len(text) <= cap:
        return text
    return text[: cap // 2] + "\n...\n" + text[-cap // 2:]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ex = load_cache()
    writers = {f"{t}_{s}": open(OUT / f"{t}_{s}.jsonl", "w") for t in ("extract", "judge") for s in ("train", "val", "test")}
    counts = {}
    runs = list(iter_runs())
    llm = obs_by_run()
    for run in runs:
        s = split_of(run.run_id)
        # Task A
        for step_i, text in candidates(run):
            d = ex.get((run.run_id, step_i))
            if d is None:
                continue
            item = {"id": f"{run.run_id}:{step_i}", "state": shrink(text),
                    "question": "Does this tool output contain a benchmark result (latency, throughput or failure rate) that the agent actually observed, as opposed to logs, code, plans or configuration?",
                    "type": "noul", "label": bool(d["data"].get("is_benchmark_result")), "source": "haiku-4-5"}
            writers[f"extract_{s}"].write(json.dumps(item) + "\n"); counts[f"extract_{s}"] = counts.get(f"extract_{s}", 0) + 1
    jd = load_judgments()
    by_run: dict[str, list] = {}
    for (rid, q, anchor), d in jd.items():
        by_run.setdefault(rid, []).append((q, anchor, d["data"]))
    for run in runs:
        if run.run_id not in by_run:
            continue
        s = split_of(run.run_id)
        inst = {(i.question, i.anchor): i.window for i in build_instances(run, build_log(run, llm_obs=llm.get(run.run_id)))}
        for q, anchor, data in by_run[run.run_id]:
            if data["answer"] == "not_applicable" or (q, anchor) not in inst:
                continue
            item = {"id": f"{run.run_id}:{q}:{anchor}", "state": shrink(inst[(q, anchor)]), "question": QUESTIONS[q], "type": "noul",
                    "label": data["answer"] == "yes", "confidence": data.get("confidence"), "source": "sonnet-5-5", "question_id": q}
            writers[f"judge_{s}"].write(json.dumps(item) + "\n"); counts[f"judge_{s}"] = counts.get(f"judge_{s}", 0) + 1
    for w in writers.values():
        w.close()
    print(json.dumps(counts, indent=1))


if __name__ == "__main__":
    main()
