# Request traces: what an engine reports per request

The engine's own account of each request: one row per request, keyed by the trace id the client sent. The
environment's client rows (`regimes/client.py`) are still the measurement, and the referee never scores these
rows. They exist so the researcher can join "what the client saw" with "where the engine spent the time". An
engine that writes nothing still gets judged. Checked by `lab/trace_contract.py`
(`python -m lab.trace_contract DIR`).

## How an engine exposes it

The target names an environment variable in `[engine.telemetry] dir_env` (for inference-server, `TELEMETRY_DIR`).
The lab sets that variable to a fresh, empty directory for each launch. When the engine exits, the lab collects
every file in it. The directory holds only trace files, one or more (one per engine process), in either format:

- **SQLite** `*.sqlite`: a table `requests` with one column per field below, and a table `meta (key TEXT PRIMARY
  KEY, value)`.
- **JSONL** `*.jsonl`: one JSON object per row, plus `<stem>.meta.json`, one object with the meta keys.

The meta keys are all required:

| key | type | meaning |
|---|---|---|
| `schema_version` | int | the contract version the file follows (now `1`) |
| `rows_written` | int | rows in this file |
| `rows_dropped` | int | rows the engine produced but could not write (queue full, write error, after shutdown) |
| `closed` | int 0/1 | 1 once the engine shut down cleanly, so the counts are final |

Drops are never silent. A file without `rows_dropped` fails the check. A closed file whose `rows_written` differs
from its row count also fails.

## Clocks and units

- One absolute time per row: `arrival_ts`, **epoch seconds** (wall clock, float). Epoch milliseconds or
  nanoseconds and monotonic readings (seconds since boot or since the process started) are rejected.
- Every field ending in `_s` is a **duration in seconds**, measured on a monotonic clock. A duration is never the
  difference of two wall-clock readings, and never of a wall reading and a monotonic one.
- **Arrival** is when the engine received the HTTP request, before tokenizing. The wall-clock `arrival_ts` and the
  monotonic origin of the durations must be one instant: read both clocks together, or back-date one from the other.
- Token counts use the engine's own tokenizer. The client recounts with the reference tokenizer, so the two may
  differ.

## Required fields

| field | type | rule |
|---|---|---|
| `schema_version` | int | `1` |
| `trace_id` | str | the client's `X-Trace-Id`, verbatim; non-empty. A retry may reuse it, so duplicates are allowed |
| `arrival_ts` | float | epoch seconds at arrival |
| `terminal_state` | str | one of `ok`, `rejected_429`, `rejected_400`, `expired`, `preempted`, `error`, `cancelled` (the client went away), `aborted` (the engine shut down with the request unfinished) |
| `prompt_tokens` | int | ≥ 0 |
| `tokens_out` | int | ≥ 0, real tokens only (no EOS) |
| `total_s` | float | arrival to the request's end (last token, rejection or abort), ≥ 0 |
| `ttft_s` | float or null | arrival to the first real token. Set exactly when `tokens_out ≥ 1`, and ≤ `total_s` |
| `tpot_s` | float or null | `(total_s − ttft_s) / (tokens_out − 1)` when `tokens_out ≥ 2`, else null. The environment's definition, checked to 0.1% |

## Optional fields

Typed if present. Null means "not measured". Any other column is allowed and ignored.

| field | type | meaning |
|---|---|---|
| `session_id`, `config_id`, `prefill_mode` | str | client session; applied config; how the prompt was prefilled |
| `turn_index` | int | position in the session |
| `queue_wait_s`, `prefill_s`, `decode_s` | float | spans. `queue_wait_s + prefill_s ≤ ttft_s` |
| `replica_age_s` | float | seconds since the engine started serving |
| `pending_depth`, `active_size`, `max_batch_size`, `concurrent_sessions` | int | load at arrival |
| `kv_free_blocks` | int | KV-cache blocks free at arrival |
| `kv_free_frac` | float | the same as a fraction in [0, 1] |
| `cache_hit_tokens` | int | prompt tokens served from a prefix cache, ≤ `prompt_tokens`. Null when unknown, never a stand-in 0 |
| `decode_steps`, `decode_batch_width_max`, `preempted` | int | forward passes after prefill; widest batch it decoded in; preemptions |
| `decode_batch_width_mean` | float | mean batch width over its own decode steps |

## inference-server

`TELEMETRY_DIR` makes the engine write `<run_id>.sqlite`, in the SQLite layout above with the same column names
(`src/inference_server/telemetry.py`, schema 1 since inference-server#102), so no adapter is needed. Its
`TIMELINE_DIR` event log (`events.jsonl`: `ts` epoch seconds, `step`, `kind`, `trace_id` on request events) is
engine-specific and outside this contract. It joins to the rows on `trace_id`.

## vLLM (mapping only, not implemented)

vLLM's Prometheus metrics (`vllm:time_to_first_token_seconds`, `vllm:e2e_request_latency_seconds`, ...) are
histograms, which can't be split back into rows. Its OpenTelemetry request spans (`--otlp-traces-endpoint`) are
per request and could map like this. The attribute names below are from the `gen_ai.*` semantic conventions; check
them against the vLLM version in use.

- `trace_id` ← the request id. The client also sends `X-Request-Id` (honoured with `--enable-request-id-headers`),
  or the span's W3C trace id when the client sends `traceparent`.
- `arrival_ts` ← the span start time (ns since epoch, divided by 1e9).
- `ttft_s` ← `gen_ai.latency.time_to_first_token`. `total_s` ← `gen_ai.latency.e2e`.
- `queue_wait_s` / `prefill_s` / `decode_s` ← `gen_ai.latency.time_in_queue` / `time_in_model_prefill` /
  `time_in_model_decode`.
- `prompt_tokens` / `tokens_out` ← `gen_ai.usage.prompt_tokens` / `completion_tokens`. `tpot_s` is derived.
- `terminal_state` ← `finish_reason` (`stop`/`length` → `ok`, `abort` → `cancelled`).
- Requests rejected before scheduling emit no span. Their rows are missing, so `rows_dropped` can't count them.
- vLLM measures from its own engine arrival, which can come after tokenizing.

An OTLP collector's file exporter writes the spans as JSON. A small adapter would turn them into the JSONL
layout above.
