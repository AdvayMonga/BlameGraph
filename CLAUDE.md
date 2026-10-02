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
- `blamegraph/extract.py` — Haiku observation extractor (`claude-haiku-4-5`, structured JSON); cache `data/derived/observations.jsonl` (8,876 steps, $15.61 — never re-run blindly; `extract_runs` only pays for uncached steps). Key from `.env` (`ANTHROPIC_API_KEY`, optional `ANTHROPIC_WORKSPACE_ID`).
- `blamegraph/noise.py` — `clean_observations` (comparable measurements: standard invocation, full, failure-free, plausible), within-config noise floor, decision margins.
- `blamegraph/audit.py` — claims audit: numbers in the final report vs numbers in tool outputs the agent saw.
- `blamegraph/blame.py` — found × kept × executed decomposition and loss events (model-free blame graph).
- `blamegraph/oracle.py` — pooled config→metric landscape across runs; nearest-neighbour "what others measured near your shipped config".
- `scripts/report.py` — everything above → `data/derived/report.md` + `report_runs.csv`, incl. the v1 BG score (integrity + self-consistency, bootstrap CIs) vs the speedup leaderboard.
- `blamegraph/judge.py` — judge layer (`claude-sonnet-5-5`, effort low, JSON schema): 7 anchored yes/no questions (`QUESTIONS`), windows built from the experiment log; cache `data/derived/judgments.jsonl` (user-approved $8 cap; full corpus cost ~$3).
- `scripts/judge_pilot.py` / `scripts/judge_analysis.py` — run the judge (`--dry` to size) and summarize answers vs code assertions.
- `scripts/explain_run.py RUN_ID` — narrative of one run's experiment log + assertions + blame events (use this to audit any claim).
- `scripts/irt.py` — Rasch fit: assertion difficulty/discrimination, agent ability with bootstrap CIs.
- `scripts/benchmark_audit.py` — speedup-leaderboard rank stability under seed resampling vs BG score; reliability; harness normalization.
- `scripts/early_warning.py` — negative result: process at minute T does not predict scoring.
- `scripts/figures.py` — `data/derived/fig_*.png` used by the report.
- `scripts/extract_obs.py` — run the extractor (`--pilot N` first).
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
Code layer + Haiku extractor done; `scripts/report.py` is the single entry point for results. Comparability
rule: only standard, full, failure-free in-run evals are compared with anything (54% of launches are
non-standard). Open: judge layer for non-numeric self-consistency (needs budget approval), SGLang knob
parsing, oracle is sparse (configs rarely repeat across runs — needs a surrogate), D/C in-run metrics are
approximations of the official geomeans.
