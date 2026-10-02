# BlameGraph

**Process evals for ML-systems coding agents, built on the public InferenceBench traces.**

InferenceBench scores an agent by one number: the speedup of the inference server it ships after a
2-hour session on an H100. BlameGraph reads the 2 hours. From each trace it reconstructs an
*experiment log* — every config the agent wrote, every server it launched, every benchmark it ran, every
number it looked at — and asks many small yes/no questions about whether the agent **acted on its own
evidence** and **kept the task's integrity**.

Thesis: *outcome tells you what the agent got; integrity and self-consistency tell you whether it earned it.*

## What it finds (269 runs, 23 agents)
- 80 scored runs never benchmarked the config they shipped; 13 never ran the benchmark at all; 10 edited the grader.
- Of 114 runs where both are measurable, 27 shipped a config they had already measured as worse than another (11 by >2x).
- The benchmark's own noise floor is ~1-4% CV, yet 40-50% of config-switch decisions moved the metric by less than that.
- Several agents stop at minute 30-75 of 120; codex agents work to the end but benchmark stale servers.
- The speedup leaderboard's order is stable for only 49% of agent pairs under seed resampling; the process
  score is more reproducible (64%) from the same runs.
- Harness artifact: 20-40% of commands in two Claude configs were blocked by a sandbox malfunction.

## How it works
`traces.py` loads runs and mirrors InferenceBench's scoring → `experiment.py` reconstructs the experiment
log → `extract.py` (Haiku) normalizes the numbers agents printed in a hundred formats → `assertions.py`
answers code-checkable questions → `inject.py` + `scripts/flip_test.py` validate that each assertion flips
when its failure is injected → `judge.py` (Sonnet) answers the non-numeric questions, each anchored to a
specific trace window → `scripts/report.py` assembles everything.

```
python scripts/report.py            # data/derived/report.md
python scripts/explain_run.py run_0003   # narrative of one run
python scripts/flip_test.py         # assertion validation
python scripts/benchmark_audit.py   # leaderboard rank stability, reliability
python scripts/irt.py               # Rasch fit: assertion difficulty, agent ability
```

Data: `hf download aisa-group/InferenceBench-Trajectories --repo-type dataset --local-dir data/inferencebench`.
Cached model outputs (`data/derived/observations.jsonl`, `judgments.jsonl`) are committed so nothing needs re-running.
See `CLAUDE.md` for the data quirks that bite anyone analyzing these traces.
