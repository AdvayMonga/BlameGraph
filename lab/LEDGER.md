# The ledger: the environment's public interface

Any loop that writes these records gets a verdict and facts back (`python -m feedback <ledger dir>`); the lab's
tools are one writer. Append-only JSONL, one record per line, `lab/ledger/ledger.jsonl`; `lab/ledger.py` validates
on write. Every record: `id`, `at`, `schema` (set by the writer), `kind`, `run`, `session`, `snapshot`
(content hash of the agent's workspace), `snapshot_blob`, `patch`, `cost: {usd}`. Tool records add `tool`, `args`,
`result`. The agent's own words live only under `claim` and never decide anything.

| kind | written by | `result` (or shape) |
|---|---|---|
| `session` | the loop | provider, model, turns, seconds, status (`stop`/`continue`), error, `violation`, `claim.note` |
| `test` | `test` | `lint`, `tests` (bools), returncode, output tails, `changed`, `scratch_left_out` |
| `profile` | `profile` | `bundle` (ledger blob of the profiler output), `workspace_copy`, seconds |
| `bench` | `bench` | `config: {split: seen, tier}`; `verdict: ok`, `metrics: {regime: {objective, value, better, valid, invalid_reasons}}`, `regimes` (full per-regime results), `ready_s` |
| `equiv` | `equiv` | `config: {split: seen, tier}`; `verdict: pass\|fail\|inconclusive`, `passed`, `reasons`, `gates`, `metrics` (accuracy with CI, length, consistency, divergence, per-task flips, unanswered), `thresholds` |
| `submit` | `submit` | **held-out shape**: no `result`; top-level `config: {split: heldout, tier: full}` and `metrics: {regime: {base, new, delta_pct, band_pct, verdict}}`, scalars only. The ledger refuses anything else that mentions the held-out split. |
| `baseline` | the loop, once per run before the first session | `config: {split: seen, tier: short, commit}`; `verdict: ok` with `metrics` shaped like `bench` (the base commit under bench's default regimes, from the cache submit compares against), or `verdict: error`, `reason` |
| `finding` | `lab.ledger seed` | a `knowledge/` entry, whole, as `claim`; `source`, `content_sha` |
| `note` | `note`, `restore`, refusals | free text, or `{verdict: refused, reason}` |

A tool that could not run writes `{verdict: refused | error, reason}` and the agent sees that text; it never
crashes the session. The integrity verdict (`feedback/lab_verdict.py`) reads only these records: a run is valid
iff a completed `submit` exists whose snapshot also has a completed seen-split `bench`, a passing `test`, a
passing `equiv` (tier full) and no failing `equiv`, and no record reports a write-surface violation.
