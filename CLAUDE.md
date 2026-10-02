# BlameGraph

Feedback tool for autoresearch loops that optimize inference (servers now, kernels next). The user is building the
loop itself (measurement + context retrieval) in a separate repo; BlameGraph only *judges finished sessions*.
See the parent `../CLAUDE.md` for coding guidelines.

Hard rule from the user: the loop may receive **facts only, never heuristics or advice**. Integrity is a verdict
(valid/invalid + reasons); self-consistency and methodology are researcher diagnostics, never fed to the agent.

## Layout
- `blamegraph/feedback.py` — `feedback(path)` → `{for_agent: {integrity, facts}, for_researcher: {assertions, blame}}`; `python -m blamegraph feedback PATH [--agent-text]`. Accepts an InferenceBench run dir or a session ledger.
- `blamegraph/session.py` — the session format (append-only JSONL events: session_start, config, launch, measure, submission; grader hashes). The user's loop will get an adapter that writes/converts to this.
- `blamegraph/validate.py` — integrity rules → `Verdict`. Live mode (hash files on disk) or recorded mode (hashes from the submission event).
- `blamegraph/adapter.py` — session → `ExperimentLog`, so assertions/blame run on loop sessions.
- `blamegraph/traces.py`, `experiment.py` — InferenceBench trace loader and experiment-log reconstruction (configs, launches, evals, observations, stale/warm/standard flags). Regex-only number parsing.
- `blamegraph/assertions.py` — code-checkable assertions; `inject.py` — failure injectors; `blame.py` — found×kept×executed; `noise.py` — comparable observations + noise floor; `audit.py` — claims audit; `exploration.py` — knobs varied.
- `tests/test_feedback.py` (synthetic sessions, validator flips) and `tests/test_flip.py` (injection on real traces; skips without data).
- `data/` is gitignored and local only: `data/inferencebench/` (public traces, `hf download aisa-group/InferenceBench-Trajectories --repo-type dataset --local-dir data/inferencebench`), `data/derived/` (cached Haiku/Sonnet outputs from the research phase — the only copy), `data/laya/`.

## History
`research-v1` tag = full research snapshot: InferenceBench harness copy with hooks, Haiku extractor, Sonnet judge,
dashboard, Laya experiments, report/IRT/benchmark-audit scripts, loop tool (ledger/landscape/context). Restore from
there rather than re-deriving. Trace quirks (codex truncation, sandbox exit-126, no timestamps) are handled in
`traces.py`/`experiment.py`.

## Next
Kernel/server equivalence checks (`blamegraph/equivalence/`), adapter for the user's loop logs once that repo is ready.
