"""Judge pilot. Usage: judge_pilot.py --per-agent 3 --budget 8 [--dry]"""
from __future__ import annotations

import argparse
import asyncio
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from blamegraph.experiment import build_log  # noqa: E402
from blamegraph.extract import obs_by_run  # noqa: E402
from blamegraph.judge import QUESTIONS, build_instances, judge_instances, load_cache  # noqa: E402
from blamegraph.traces import iter_runs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-agent", type=int, default=3, help="runs per agent (first N by id, spread over scenarios)")
    ap.add_argument("--budget", type=float, default=8.0)
    ap.add_argument("--dry", action="store_true", help="only build instances and estimate size")
    a = ap.parse_args()
    llm = obs_by_run()
    picked = collections.defaultdict(list)
    for run in iter_runs():
        taken = picked[run.agent]
        # spread the first picks over scenarios; beyond 4 per agent take everything
        if len(taken) < a.per_agent and (len(taken) >= 4 or run.scenario not in {r.scenario for r in taken}):
            taken.append(run)
    runs = [r for v in picked.values() for r in v]
    instances = []
    for run in runs:
        instances.extend(build_instances(run, build_log(run, llm_obs=llm.get(run.run_id))))
    byq = collections.Counter(i.question for i in instances)
    chars = sum(len(i.window) for i in instances)
    print(f"{len(runs)} runs -> {len(instances)} instances {dict(byq)}; ~{chars//4} input tokens; est ${chars/4/1e6*2 + len(instances)*400/1e6*10:.2f}")
    if a.dry:
        for i in instances[:2]:
            print("\n=====", i.run_id, i.question, i.anchor); print(i.window[:1500])
        return
    usage = asyncio.run(judge_instances(instances, budget_usd=a.budget))
    print(json.dumps(usage))
    cache = load_cache()
    print("\nanswers by question:")
    for q in QUESTIONS:
        ans = collections.Counter(d["data"]["answer"] for k, d in cache.items() if k[1] == q)
        print(f"  {q:24s} {dict(ans)}")


if __name__ == "__main__":
    main()
