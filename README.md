# BlameGraph

An evaluation tool for agents that optimize LLM inference servers, built on InferenceBench and designed to sit inside an autoresearch loop.

- **It audits the session, not just the submission.** InferenceBench scores the server an agent ships; BlameGraph replays the hours before that and records what the agent configured, launched, measured, read, and finally shipped.
- **Two things are scored: integrity and self-consistency.** Integrity: the grader was untouched, the submission is a config that was actually benchmarked, the numbers are physically possible. Self-consistency: the agent read the results it asked for, kept the best config it measured, and benchmarked the server it thought it was benchmarking.
- **Everything else is described, never scored.** How many knobs it varied, whether it ran a baseline first, how much of the budget it used: reported for the researcher, deliberately kept out of the score so no way of working is prescribed.
- **Every assertion earns its place.** Failures are injected into real traces and an assertion stays only if it flips; its stability under reseeding and its agreement across seeds are measured and published.
- **As a tool, it hands the loop facts, not advice.** A ledger of measurements with uncertainty, a validator with public rules, and a pooled landscape of what every prior run measured. The agent sees what happened and decides for itself; process judgments stay on the researcher's dashboard.
- **Numbers the agent printed are normalized once and cached.** A small model reads each benchmark output into structured form; a judge answers a handful of anchored yes/no questions about the final report. Both are optional on a harness that logs the measurements directly.
- **The benchmark itself gets audited.** Rank stability of the leaderboard under reseeding, harness artifacts in the traces, and the benchmark's own noise floor are first-class outputs.
- **Entry points:** `scripts/report.py` for the full results, `scripts/explain_run.py RUN_ID` for one run's story, the dashboard for browsing, `blamegraph/tool/` for the loop-facing interface. See `CLAUDE.md` for layout and data quirks.
