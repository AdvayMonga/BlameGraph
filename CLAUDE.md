# BlameGraph

Assertion-based *process* evals for ML-systems coding agents, built on the public InferenceBench
trajectories. Goal: score how an agent experimented (many small code/judge-checked assertions), not
just its final speedup. See the parent `../CLAUDE.md` for coding guidelines.

Thesis (2026-10-01): score only **integrity** (grader edits, unmeasured submissions, impossible metrics)
and **self-consistency** (acted on its own evidence: read results, shipped the best config it saw,
benchmarked the live server). **Methodology** assertions (baseline-first, one-factor-at-a-time,
search-space coverage) are reported descriptively, never scored — they penalize legitimate approaches.

## Layout
- `data/inferencebench/` — HF `aisa-group/InferenceBench-Trajectories` (269 runs, 23 agents × 4 scenarios × 3 seeds). Not committed.
- `data/derived/` — CSVs produced by scripts.
- `blamegraph/traces.py` — `Run`/`Event`/`Step` loader. `Run.steps()` pairs tool calls with results (FIFO within an assistant turn). Mirrors InferenceBench scoring (`primary_metric`, integrity floor, `gate_passed`, `scored`) and estimates `speedup` from backed-out baseline constants.
- `blamegraph/experiment.py` — `build_log(run)` reconstructs the experiment log: `start_server.sh` versions (file tools, heredocs, codex cumulative diffs, or passive full reads), server restarts (command or startup banner), `evaluate.py` launches (quick/full, stale-config, killed, observed), metric observations seen by the agent, timer anchors, `eval_script_modified`.
- `blamegraph/assertions.py` — code-checkable assertions (`ASSERTIONS` registry, `evaluate(run, log)`); each returns True/False/None(n/a).
- `blamegraph/exploration.py` — knobs varied across measured configs vs. the search baseline's 11-knob vLLM space; one-factor-at-a-time rate.
- `blamegraph/inject.py` — failure injectors (`INJECTORS`) that turn a real run into one with a known process failure.
- `scripts/corpus_stats.py` — raw per-run counters → `data/derived/runs.csv`.
- `scripts/experiment_stats.py` — experiment-log metrics → `data/derived/experiment.csv` + summaries.
- `scripts/free_pass.py` — all assertions over the corpus → `data/derived/assertions.csv`; per-agent rates, discrimination, seed consistency, outcome relationship.
- `scripts/flip_test.py` — inject each failure into every applicable run and report which assertions flip (validates the assertion set).

## Data quirks (handled in the loader; don't re-derive)
No timestamps (use `timer.sh` outputs). Old codex traces append the cumulative `git diff` to tool
results; new codex (gpt-5.5) traces omit file-write tool calls entirely; all codex traces keep only the
first line of multi-line commands. claude-opus-4-6/sonnet-4-6 runs have 20–40% of commands blocked by a
sandbox malfunction (`Exit code 126`, even on `date`/`echo`). Baseline numbers are not in the dataset;
`BASELINE_METRIC` is backed out from the README leaderboard.

## Status
Free (code-only) layer done and validated by flip tests. Next (needs `ANTHROPIC_API_KEY`): Haiku
extractor for observed metrics (~$10), then judgment assertions for self-consistency (~$50, pilot first).
