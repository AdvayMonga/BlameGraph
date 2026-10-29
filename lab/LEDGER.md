# The ledger: the environment's public interface

Any loop that writes these records gets a verdict and facts back (`python -m feedback <ledger dir>`); the lab's
tools are one writer. Append-only JSONL, one record per line, `lab/ledger/ledger.jsonl`; `lab/ledger.py` validates
on write. Every record: `id`, `at`, `schema` (set by the writer), `kind`, `run`, `session`, `snapshot`
(content hash of the agent's workspace), `snapshot_blob`, `patch`, `cost: {usd}`. Tool records' `cost` also has
`seconds` (wall time of the call); GPU tools add `gpu_seconds`, `usd_per_hour` and `rate_known`. Tool records add `tool`, `args`,
`result`. The agent's own words live only under `claim` and never decide anything.

| kind | written by | `result` (or shape) |
|---|---|---|
| `session` | the loop | provider, model, turns, seconds, status (`stop`/`continue`), error, `violation`, `claim.note`, optional `closing: {records, reflected, usd, estimated, tokens, turns, error}` or `{records, skipped: budget}` (the reflection-only call after a session that left `records` unreflected; `cost.usd` includes its `usd`; older records name one `record`) |
| `test` | `test` | `lint`, `tests` (bools), returncode, output tails, `changed`, `scratch_left_out` |
| `profile` | `profile`, `trace`, `kernel`, `hostprof` | one shape for every instrument (`proftools.run_instrument`): `verdict: ok \| failed \| refused \| contaminated` (`reason`, and `gpu_processes` as a bench's), `bundle` (blob: `profile`'s bundle itself, the others' whole output dir), `workspace_copy`, `instrument` (resolved binary; None for torch), `argv` (every command run), returncode, seconds, output tail, `gpu`, `calibrated` (`profile`: kernel time, `layers`, `kernel_tasks`; `kernel`: ncu against the peaks; `trace`: nsys kernel, memory-copy and CUDA API time, top kernels, kernel time's share of the window; `hostprof`: py-spy samples per thread, top functions by `self` and `total` share), `workbench_killed` |
| `bench` | `bench` | `config: {split: seen, tier, seed}`; `verdict: ok`, `metrics: {regime: {objective, value, better, valid, invalid_reasons}}`, `calibrated: {regime: {value, ceiling, pct_of_ceiling, bound, ceiling_basis, peaks_source, shape, reference, pct_of_reference, fingerprint}}` (a missing ceiling or reference comes with `ceiling_missing` / `reference_missing` saying why), `regimes` (full per-regime results, each with `profile`), `ready_s`, `profiles_ready_s`, `profiles_failed` (profile: reason, when a profile launch failed; its regimes are invalid with that reason), `gpu`, `artifacts` (plus `profiles` when a regime had its own launch), `profiled` (only with `args.profile: true`: the same tree profiled right after the measurement, in its own launch: `launch: general` and `workload: profile defaults` (8 requests x 32 tokens, so its facts describe that load, not the regimes'), then the `profile` record's result (`verdict`, `bundle`, `calibrated` with its `layers`, ...; `output` only when not ok); or `error` when it raised and `skipped` when no budget was left, neither of which fails the bench; capped at 15 min). A regime served under its own launch profile (`lab/profiles.py`) adds `profile` and `general: {value, valid}` to its metric |
| `equiv` | `equiv` | `config: {split: seen, tier: quick\|dev\|full, profile?}` (`profile` absent = the general launch); `verdict: pass\|fail\|inconclusive` (tier `quick`: `facts`, `passed: null`, `metrics` = the divergence summary only), `passed`, `reasons`, `gates`, `metrics` (accuracy with CI, length, per-task flips (consistency and divergence only in records before 2026-10-28), unanswered, sample: {items, of, looks}: how many of the tier's items, in their fixed order, were answered before the gate stopped), `thresholds`, `artifacts` |
| `submit` | `submit` | **held-out shape**: no `result`; top-level `config: {split: heldout, tier: short, profiles?}` (the regimes measured under their own launch profile) and `metrics: {regime: {base, new, delta_pct, band_pct, verdict}}`, scalars only. The ledger refuses anything else that mentions the held-out split. |
| `baseline` | the loop, once per run before the first session | `config: {split: seen, tier: min, commit}`; `verdict: ok` with `metrics` and `calibrated` shaped like `bench`'s (the base commit under bench's default regimes: saturated plus the task's objective and constrained ones, from the cache submit compares against), plus `profiled` as in `bench` (always), or `verdict: error`, `reason` |
| `noise` | `lab.validate` | `config: {split: seen, tier}`; `result`: regime, band_pct (at least `lab.validate.RESOLUTION_PCT` for a searched regime), values, runs, model, target, server, corpus_version, measured_at, engine_commit (None when the engine tree had uncommitted changes); `submit`, experiment records and the brief use the latest band for the target, at any engine commit, on the current corpus version (a record without one counts) |
| `note` | `note`, `restore`, refusals | free text, or `{verdict: refused, reason}` |
| `note` (`tool: start`) | `python -m lab.experiment start` | a researcher run: `args: {repo, base, change, budget_usd}`, `result: {author, applied, left_out}` |
| `hypothesis` | `hypothesis` | the agent's text: `hypothesis` (bottleneck and proposed change), `rationale`, `layer` (kernels, model_execution, runtime or serving; older records have none), optional `abandon_if`, `from_record` |
| `reflection` | `reflect` | the agent's text on one measured record: `record`, `reflection` (why it matched or differed from `expect`), optional `alternative` |
| `kernels` | `python -m kernels.evolve run` | one kernel-loop run (run id `kernels-…`): `config: {split: seen, task, entry, iterations, device, seen_shapes, selected_from, models}`; `result: {best` (the best program's metrics), `candidates: [{id, parent, iteration, island, metrics}]` (every evaluated candidate, seen shapes only), `retimed: [{id, selected_score, scores, median}]` (the top candidates re-judged 3x; the highest median is `best`), for `kernels.search` runs also `tuned` (`candidates`: each tuned program's index among the re-timed ones and its grid points; `from`: the one whose best point was re-timed; its `median` vs the winner's `base_median`, `promoted`, any `regions` dispatch), `hypotheses` (blob: every expansion's named changes, and each one's child, score, whether it beat its parent and whether it was a new best; runs before 2026-10-20 also carry ROI and screener scores and a random-pick flag, and `screen` names the screener), `trace` (blob: OpenEvolve's evolution trace with code, diff and prompt per candidate), `best_program` (blob)`}`. Held-out shapes are never in it: `gate` judges them and its output stays in the run dir. The `kernel_search` tool writes a second `kernels` record (`tool: kernel_search`): `result: {verdict, metrics: {speedup, heldout_speedup}, best, search_record` (the search's own record), `evals, model_usd, gate: {passed, metrics, gates}, patch` (its path in the workspace when the gate passed), `captures}` or `{verdict: error, reason}`, with the experiment block |

Every measured record (`bench`, `equiv`, `submit`, `profile`, including `trace`/`kernel`/`hostprof`) made by the
agent carries `experiment: {hypothesis, expect: {metric, value, unit: absolute|change_pct, why}, actual, gap,
gap_pct_of_expected, noise_band_pct}`, or `actual_missing` saying why the lab could not compute the measured
counterpart (`lab/experiments.py`). The profilers' results have no `metrics`, so their `expect.metric` is a dotted
path into `result` like a mechanism's (e.g. `calibrated.kernel_share_of_window`; a list's row is named by its
`name` or `function`, e.g. `calibrated.kernels.<kernel>.total_s`), and their `change_pct` has a baseline only for a
`profile` at its default workload and launch (the baseline record's `profiled`). An `expect` may carry `mechanism`, a list of further predictions on other numbers
in the same result (`metric` = a dotted path into `result`, e.g. `calibrated.layers.runtime.gpu_idle_share`;
`change_pct` against the same path in the run's `baseline` record); the record keeps them as `experiment.mechanism`,
each with its `actual` and `gap` or `actual_missing`. `submit` takes none: its held-out record is aggregates only. An `expect` and each mechanism prediction take only their own keys
(`metric`, `value`, `unit`, `why`; `mechanism`; a mechanism's `hypothesis`): the measured ones are the lab's. Runs whose id starts `researcher-` are
experiments a person made outside the lab (`lab/experiment.py`), with the same records; the agent's `ledger` tool and
SQL index leave them out.

A tool that could not run writes `{verdict: refused | error, reason}` and the agent sees that text; it never
crashes the session. `bench`, `equiv`, `submit`, `baseline` and every profiler also write `refused` when a process already held
the GPU (no engine launched) and `contaminated` when a process other than the served engine held it during the
measurement; both carry `gpu_processes: [{pid, used_mib, name}]`. A contaminated run is never evidence and never
cached as the base. A measurement that killed processes in the agent's workbench (they held the GPU) lists them
as `workbench_killed: [{pid, used_mib, name}]` (bench, equiv, profile records; submit's reply). A `session`'s
`cost.usd` is what the API proxy priced, and `cost.tokens` its token counts. The integrity verdict (`feedback/lab_verdict.py`) reads only these records: a run is valid
iff a completed `submit` exists whose snapshot also has a completed seen-split `bench`, a passing `test`, a
passing `equiv` and no failing `equiv`, and no record reports a write-surface violation.

## Passive data (`artifacts`)

Every bench and equiv (including one that errored after serving) keeps the cheap data from the same served run as
ledger blobs, referenced from `result.artifacts: {device, telemetry, client_rows, serve_log}` (blob paths under the
ledger root, `telemetry` None when the target has no `[engine.telemetry]` or the engine wrote nothing). The tool's
reply to the agent never includes them. `lab/artifacts.py` loads them back (`load`) and joins client rows to engine
rows by trace id (`joined`: client `X-Trace-Id` is `bg-<row id>`).

- `device/`: `meta.json` (`available`; when true `source` dcgm|nvidia-smi, `coarse` (nvidia-smi utilization is
  coarse, DCGM profiling fields are not), `fields`, `interval_ms`, `exited` if the sampler died) and `samples.csv`,
  each raw tool line prefixed with epoch seconds. Covers the whole served window, launch to teardown.
  `gpu.read_samples` reads either source as rows (the index's `device_samples`): `epoch_s`, then one number per
  field under DCGM's name where both sample it (`power_w`, `sm_clock_mhz`, `mem_clock_mhz`, `throttle_reasons`),
  nvidia-smi's others with dots as underscores, N/A as null.
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
