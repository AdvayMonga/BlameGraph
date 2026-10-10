# The ledger: the environment's public interface

Any loop that writes these records gets a verdict and facts back (`python -m feedback <ledger dir>`); the lab's
tools are one writer. Append-only JSONL, one record per line, `lab/ledger/ledger.jsonl`; `lab/ledger.py` validates
on write. Every record: `id`, `at`, `schema` (set by the writer), `kind`, `run`, `session`, `snapshot`
(content hash of the agent's workspace), `snapshot_blob`, `patch`, `cost: {usd}`. Tool records' `cost` also has
`seconds` (wall time of the call); GPU tools add `gpu_seconds`, `usd_per_hour` and `rate_known`. Tool records add `tool`, `args`,
`result`. The agent's own words live only under `claim` and never decide anything.

| kind | written by | `result` (or shape) |
|---|---|---|
| `session` | the loop | provider, model, turns, seconds, status (`stop`/`continue`), error, `violation`, `claim.note` |
| `test` | `test` | `lint`, `tests` (bools), returncode, output tails, `changed`, `scratch_left_out` |
| `profile` | `profile` | `bundle` (ledger blob of the profiler output), `workspace_copy`, seconds |
| `profile` | `trace`, `kernel`, `hostprof` | `bundle` (blob of the instrument's output dir), `workspace_copy`, `instrument` (resolved binary), `argv` (every command run), returncode, seconds, output tail |
| `bench` | `bench` | `config: {split: seen, tier}`; `verdict: ok`, `metrics: {regime: {objective, value, better, valid, invalid_reasons}}`, `regimes` (full per-regime results), `ready_s`, `artifacts` |
| `equiv` | `equiv` | `config: {split: seen, tier}`; `verdict: pass\|fail\|inconclusive`, `passed`, `reasons`, `gates`, `metrics` (accuracy with CI, length, consistency, divergence, per-task flips, unanswered), `thresholds`, `artifacts` |
| `submit` | `submit` | **held-out shape**: no `result`; top-level `config: {split: heldout, tier: full}` and `metrics: {regime: {base, new, delta_pct, band_pct, verdict}}`, scalars only. The ledger refuses anything else that mentions the held-out split. |
| `baseline` | the loop, once per run before the first session | `config: {split: seen, tier: short, commit}`; `verdict: ok` with `metrics` shaped like `bench` (the base commit under bench's default regimes: the task's objective and constrained ones, from the cache submit compares against), or `verdict: error`, `reason` |
| `finding` | `lab.ledger seed` | a `knowledge/` entry, whole, as `claim`; `source`, `content_sha` |
| `note` | `note`, `restore`, refusals | free text, or `{verdict: refused, reason}` |

A tool that could not run writes `{verdict: refused | error, reason}` and the agent sees that text; it never
crashes the session. `bench`, `equiv`, `submit` and `baseline` also write `refused` when a process already held
the GPU (no engine launched) and `contaminated` when a process other than the served engine held it during the
measurement; both carry `gpu_processes: [{pid, used_mib, name}]`. A contaminated run is never evidence and never
cached as the base. A measurement that killed processes in the agent's workbench (they held the GPU) lists them
as `workbench_killed: [{pid, used_mib, name}]` (bench, equiv, profile records; submit's reply). A `session`'s
`cost.usd` is what the API proxy priced, and `cost.tokens` its token counts. The integrity verdict (`feedback/lab_verdict.py`) reads only these records: a run is valid
iff a completed `submit` exists whose snapshot also has a completed seen-split `bench`, a passing `test`, a
passing `equiv` (tier full) and no failing `equiv`, and no record reports a write-surface violation.

## Passive data (`artifacts`)

Every bench and equiv (including one that errored after serving) keeps the cheap data from the same served run as
ledger blobs, referenced from `result.artifacts: {device, telemetry, client_rows, serve_log}` (blob paths under the
ledger root, `telemetry` None when the target has no `[engine.telemetry]` or the engine wrote nothing). The tool's
reply to the agent never includes them. `lab/artifacts.py` loads them back (`load`) and joins client rows to engine
rows by trace id (`joined`: client `X-Trace-Id` is `bg-<row id>`).

- `device/`: `meta.json` (`available`; when true `source` dcgm|nvidia-smi, `coarse` (nvidia-smi utilization is
  coarse, DCGM profiling fields are not), `fields`, `interval_ms`, `exited` if the sampler died) and `samples.csv`,
  each raw tool line prefixed with epoch seconds. Covers the whole served window, launch to teardown.
- `telemetry/`: whatever the engine wrote into `{dir}` (inference-server: `<run>.sqlite` with a `requests` table,
  `events.jsonl`).
- `client_rows`: JSONL, one per measured request, without generated text. Bench: the regimes' rows with `regime`,
  `call` (which run_open/run_closed call), `call_started_at` (epoch) and run-clock times; warmups are not kept.
  Equiv: one per gate item (`id`, `completion_tokens`, `finish_reason`, `error`).
- `serve_log`: this run's part of the serve log.

**Submit keeps none of this in the ledger.** Its held-out record stays in the strict held-out shape above, with no
field pointing at passive data, and no blob is written for it. The same data for the held-out serve goes to
`<run dir>/heldout-private/<snapshot>-<time>/` (same layout, `client_rows.jsonl`), outside the ledger and outside
every jail: operator-only. The base-commit measurements submit makes keep no passive data.
