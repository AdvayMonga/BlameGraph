# State

What works now. Rewritten in place, never appended: when something changes, edit its line. History is in git,
`DECISIONS.md` and `RUNS.md`. As of 2026-10-15 (main at `10eae2c`).

## The four layers (goal: `DECISIONS.md` 2026-10-14)
| | 1. Kernels | 2. Model execution | 3. Runtime | 4. Serving |
|---|---|---|---|---|
| In the example engine | `models/paged_attention_kernel.py`, MoE grouped GEMM, norm/rope, sampling | `models/qwen3_moe.py`, `backends/custom_torch_backend.py`, `launch_table.py`, `quant.py` | `scheduler.py`, `scheduling_policy.py`, `kv_cache/` | `server.py`, `openai_shim.py`, tokenizer, startup, `control_plane/` |
| Data | have: real call shapes and inputs captured from runs (`lab.profile --capture`, `kernels/shapes.py` splits them), validated on GPU; long-context shapes thin (~30 seen requests) | partial: correctness task sets + corpus; missing a small fixed batch for logit checks and a list of step configurations | partial: corpus (public, repeats prompts); the load generator (separate work) replaces it | have: same workloads + the cold-start command |
| Simulator | not needed (the real kernel runs in seconds; the roofline is the bound) | partial: analytic step floors (`validity/ceiling.py`); missing a measured step-time model | missing: scheduler simulator (real `scheduler.py` on a fake clock and backend, step times from layer 2's model) | not needed |
| Hypothesis generator | have: OpenEvolve replication (`kernels/evolve/`), validated on GPU (`RUNS.md` 2026-10-14/15); one task with real shapes (`paged_decode`) | missing (hypothesis-driven + knob search) | missing (hypothesis-driven, simulator-screened) | not needed: the general agent |
| Measurement tools | have: `time_kernel`, ncu `kernel`, nsys `trace`, % of bandwidth peak | have: `profile`, `trace`, kernel share of time, step floors | have: `bench` client rows, engine telemetry, timeline, py-spy | have: `bench` cold_start/overload, serve logs, telemetry |
| Evaluator | have: cascade judge + held-out gate (`kernels/evolve/core.py`), hardened 2026-10-15; candidates judged in the jail (`kernels/evolve/jailed.py`), untested on a GPU | missing: fast step-time + logit check (no server) | partial: `bench` + `equiv`; no noise bands measured for this engine yet | have: `bench` regimes |

Across layers, built: the end-to-end judge, `search` over every run's experiments (BM25 + local embeddings,
`lab/search.py`; the store's ledger holds no experiment records yet), calibration (`lab/calibrate.py`), experiment records
(`lab/experiments.py`; a closing reflection call per session; a person's experiments via `lab/experiment.py`, hidden
from the agent), the workbench/measurement sandbox, the GPU VM tooling.

## Validated on a real GPU (evidence in `RUNS.md`)
- Correctness gate, dev and full tier (2026-10-04, 2026-10-05).
- Kernel checker and timer (2026-10-04).
- Profiler tools `profile`, `trace`, `kernel`, `hostprof` (2026-10-08).
- The agent loop end to end, one session, `LAB_WORKER=ssh` (2026-10-10).
- Jails, workbench, quiet, exclusivity, dependency build, ncu: `lab.gpucheck` (2026-10-13).
- Ceilings below every measurement (2026-10-13).
- Launch profiles (#33): two engine starts back to back, the profiled launch measured apart (2026-10-15).
- `python -m regimes run` single_stream, saturated, long_prompt against the engine (2026-10-15).

## Built, not yet run on a GPU
- `python -m lab.validate` (reference, good/bad launches, noise bands): tested on fakes only.
- `python -m regimes run all` against the engine and against vLLM (three regimes ran 2026-10-15; the other four not).
- A real `python -m lab.session --task ...` run.
- `validity/tier_agreement.py`: needs real short/full runs.
- Remote worker (`LAB_WORKER=ssh`): used in the one-session dry run only.

## Where the engine stands (2026-10-16, inference-server main after #109)
Engine vs plain vLLM (2026-10-13 for vLLM): single_stream TPOT p99 9.6 ms vs 4.1 ms, saturated 1.07-1.65 vs 5.8 req/s
(concurrency 64 straddles the 2 s TTFT line: a prefill wave still blocks decode), long prompt 6.25 vs 68 req/s; every
regime valid; the dev-tier gate passes (1,597/1,597). Decode step 9.7 ms at batch 1, 14 at 8, 19 at 16, 43 at 128.
Still launch-bound; the next structural item is prefill/decode overlap (mixed batching), then fp8 once bandwidth-bound.
fp8 experts exist, off (`RUNS.md` 2026-10-15/16).

## In flight (branches not on main)
| Branch | Worktree | What | Status |
|---|---|---|---|
| `lab/profiles` | `BlameGraph-wt/profiles` | Launch profiles: a config per regime | Not merged |
| `target/batch-size` | `BlameGraph-wt/batch` | The engine's own batch size, not a pinned 32 | One commit, not merged |
| `gpu/session` | `BlameGraph-wt/gpu` | Merges `lab/profiles` + `target/batch-size`: the tree pushed to the VM for the 2026-10-15 session | Disposable; not for merging |
| `docs/gpu-1016` | `BlameGraph-wt/docs1016` | The 2026-10-16 session's RUNS.md/STATE.md entries | PR open |

## Open problems
- BF16 puzzle: plain BF16 matched the reference on Nebius at concurrency 128 but diverged on Verda at 32
  (`RUNS.md` 2026-10-05). Don't treat a BF16 rerun as a noise sample until resolved.
- `equiv` at tier dev: ~26 min on inference-server #107 at concurrency 32, but 12/1,585 items were 429-rejected
  (inconclusive). 429s are now retried for ~6 min (was ~1 min); not yet rerun on a GPU. Submit takes a dev-tier equiv
  for this engine (`submit_tier`); equiv tier `quick` (teacher-forced divergence only) is unmeasured on a GPU.
- The engine's radix prefix cache evicts by scanning the whole tree per leaf (~100% of scheduler time in long-context
  prefill): a layer-3 engine bug (`RUNS.md` 2026-10-14/15).
- `session.py --run` re-resolves `base` from HEAD rather than the run's original base; resuming after the engine moved
  would misaudit.
- A host without `LAB_STORE` (a VM `--env` session) cannot submit (the holdout guard is in the store).
- The store index is rebuilt in full when the ledger grows; make it incremental when that gets slow. DCGM device
  samples are indexed as raw lines.
- Engine branches `integ/gpu-session-5` and `engine/gpu-validated` (2026-10-06 report) were judged by the old gate:
  re-judge with `python -m correctness verdict` on their saved results before deciding anything.
- The corpus's long_prompt_short_output window repeats each prompt ~5x, which rewards prefix caching.

## Next
- GPU (Nebius), built and CPU-tested, awaiting measurement in this order, one at a time: inference-server #110
  (`PREFILL_MODE=mixed`: bounded multi-row prefill between decode steps, for saturated at concurrency 64-128),
  #111 (`ASYNC_DECODE=1`: step n launched on n-1's device tokens, host work overlapped), then fp8 experts at wide
  batch (`CUSTOM_BACKEND_QUANT=fp8`, already in main), then a Triton `moe_align` + fused MoE epilogue built live
  (the remaining ~25 routing launches per layer). The slowest prefill waves (a 6-prompt 8.5k-token wave at 1.5 s)
  and long-prompt cache hits need a profile before anything is built for them.
- GPU, needs spend approval: resolve the BF16 puzzle (dev tier at concurrency 32 and 128 on one VM, ~20 min, ~$1.50);
  first real `python -m regimes run all` against the engine and against vLLM.
- GPU, with approval and the auto-stop armed: `python -m lab.validate --target targets/inference-server.toml`
  (~3-4 h; serves unjailed, so not blocked), then a real `python -m lab.session --task ...` run.
- Layer 1: the improvement list in `data/notes/kernel-generator-improvements.md` (local). Engine PR
  inference-server #106 (the evolved kernel) is merged, its `equiv` not run; the jailed evaluator needs a GPU run.
- Agent regime: undecided items in `DECISIONS.md` 2026-10-14.

## Store
Bucket `bg-data` (`.env`: `LAB_STORE`, `LAB_STORE_ENDPOINT`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`) holds the
ledger, its blobs, the corpus's held-out traces + seed, the holdout guard, and archives of `data/derived`,
`data/equiv*`, `data/h200-run-*`, `data/engine-h200-*`, `data/agent-traces` (2026-10-14).
