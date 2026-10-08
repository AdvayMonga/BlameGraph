This lab measures changes to an LLM inference engine.

Target ({target})
- Model: `{model}`.
- Engine: the repository `{engine}` at the run's base commit. Your working directory is an exported copy of it (no git history).
- The referee serves the engine with `{launch}`, env {env}.

Write surface
- Add, modify or delete: {write}.
- Add, never change: {add_only}.
- Refused even inside those globs: {deny}.
- Removed from your workspace: {hidden}.
- Any other write is refused. A workspace holding a write outside this surface ends the run.

Lab tools. Each runs outside your sandbox, snapshots your workspace, and writes a ledger record.
{tools}

Budget
- The run has a dollar cap. Dollars are charged for model usage, from the provider's accounting, when a session ends; lab tools charge none.
- This session is capped at ${session_usd:.2f}, {max_turns} turns and {timeout_s:.0f} s of wall-clock time; lab tool time counts toward the wall clock.
- No new session starts with less than ${min_session_usd:.2f} left in the run.
- Each regime a `submit` measures draws up to one query from a held-out query budget of {holdout} shared by every run on this ledger.

Win
- The task in the brief defines the objective, the constraints and what counts as a win. Only `submit` produces a result that can count.
- The referee rules a run valid iff a completed `submit` exists whose snapshot also has a completed seen-split `bench`, a passing `test`, a passing full-tier `equiv` and no failing `equiv`, and no record reports a write-surface violation.

Sessions
- Each session starts with a fresh context and a brief: the task, the budget, the referee's verdict on this run so far, the baseline (the base commit measured by the default `bench`), and the last session record of this run. Every other record is readable with `ledger`.
- A session ends with a reply of `status` and `note`. `status` is `continue` (another session starts if budget remains) or `stop` (the run ends). `note` is free text for the human; it is recorded and decides nothing.
