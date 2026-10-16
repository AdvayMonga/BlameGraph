# State

What works now. Rewritten in place, never appended: when something changes, edit its line. History is in git,
`DECISIONS.md` and `RUNS.md`. As of 2026-10-15 (main at `10eae2c`).

## The four layers (goal: `DECISIONS.md` 2026-10-14)
| | 1. Kernels | 2. Model execution | 3. Runtime | 4. Serving |
|---|---|---|---|---|
| In the example engine | `models/paged_attention_kernel.py`, MoE grouped GEMM, norm/rope, sampling | `models/qwen3_moe.py`, `backends/custom_torch_backend.py`, `launch_table.py`, `quant.py` | `scheduler.py`, `scheduling_policy.py`, `kv_cache/` | `server.py`, `openai_shim.py`, tokenizer, startup, `control_plane/` |
| Data | have: real call shapes and their int args (real sequence lengths) captured from runs (`lab.profile --capture`, `kernels/shapes.py`), used for checking and timing; each task's share of a profiled window (`kernels.evolve share`). Captures validated on GPU; real-input timing and share not yet run on one. One task (`paged_decode`); long-context shapes thin (~30 seen requests) | partial: correctness task sets + corpus; missing a small fixed batch for logit checks and a list of step configurations | partial: corpus (public, repeats prompts); the load generator (separate work) replaces it | have: same workloads + the cold-start command |
| Simulator | not needed (the real kernel runs in seconds; the roofline is the bound) | partial: analytic step floors (`validity/ceiling.py`); missing a measured step-time model | missing: scheduler simulator (real `scheduler.py` on a fake clock and backend, step times from layer 2's model) | not needed |
| Hypothesis generator | have: OpenEvolve replication (`kernels/evolve/`), validated on GPU (`RUNS.md` 2026-10-14/15); facts-only prompts; every candidate in a `kernels` ledger record; top candidates re-timed before promotion. Prompts, records and re-timing not yet run on a GPU | missing (hypothesis-driven + knob search) | missing (hypothesis-driven, simulator-screened) | not needed: the general agent |
| Measurement tools | have: `time_kernel`, `time_pair` (paired with the baseline; CUDA-graph replay for graph-replayed tasks, untested on a GPU), ncu `kernel`, nsys `trace`, % of bandwidth peak | have: `profile`, `trace`, kernel share of time, step floors | have: `bench` client rows, engine telemetry, timeline, py-spy | have: `bench` cold_start/overload, serve logs, telemetry |
| Evaluator | have: cascade judge + held-out gate (correctness and held-out speed), jailed, hardened and reviewed 2026-10-15 (`kernels/evolve/core.py`, `jailed.py`; exploit canaries in `tests/test_kernels_evolve.py`); `ship` turns a gated kernel into an engine branch for the end-to-end judge. None of #46-#54 run on a GPU yet | missing: fast step-time + logit check (no server) | partial: `bench` + `equiv`; no noise bands measured for this engine yet | have: `bench` regimes |

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

## Where the engine stands (2026-10-15, inference-server #107, not merged)
Engine vs plain vLLM (2026-10-13 for vLLM): single_stream TPOT p99 15.6 ms vs 4.1 ms, saturated 0.37 vs 5.8 req/s,
long prompt 2.66 vs 68 req/s; every regime valid. The decode step is launch-bound (~2,300 kernels per step after
fusion; the CUTLASS grouped GEMM 3.5 ms of 13.3); prefill attention is torch SDPA. The gate on this state is
inconclusive (429s at concurrency 32); fp8 experts and a Triton MoE GEMM exist, off by default (`RUNS.md` 2026-10-15).

## In flight (branches not on main)
| Branch | Worktree | What | Status |
|---|---|---|---|
| `lab/profiles` | `BlameGraph-wt/profiles` | Launch profiles: a config per regime | Not merged |
| `target/batch-size` | `BlameGraph-wt/batch` | The engine's own batch size, not a pinned 32 | One commit, not merged |
| `gpu/session` | `BlameGraph-wt/gpu` | Merges `lab/profiles` + `target/batch-size`: the tree pushed to the VM for the 2026-10-15 session | Disposable; not for merging |
| `docs/gpu-1015` | `BlameGraph-wt/docs1015` | This session's RUNS.md/STATE.md entries | PR open |

## Open problems
- BF16 puzzle: plain BF16 matched the reference on Nebius at concurrency 128 but diverged on Verda at 32
  (`RUNS.md` 2026-10-05). Don't treat a BF16 rerun as a noise sample until resolved.
- `equiv` at tier dev: ~26 min on inference-server #107 at concurrency 32, but 12/1,585 items were 429-rejected
  (inconclusive). 429s are now retried for ~6 min (was ~1 min); not yet rerun on a GPU. Submit takes a dev-tier equiv
  for this engine (`submit_tier`); equiv tier `quick` (teacher-forced divergence only) is unmeasured on a GPU.
- inference-server #107's Triton MoE grouped GEMM asserts inside the decode CUDA graph under load (exact in eager mode).
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
- GPU (Nebius): the gate on inference-server #107 at concurrency 16; debug its MoE kernel in the decode graph (3.5 ms
  of the 13 ms step); then the remaining launch overhead (per-layer graph-replay copies, routing gather/scatter) and
  fp8 once the step is bandwidth-bound. Merge #107 with the evolved decode kernel (#106).
- GPU, needs spend approval: resolve the BF16 puzzle (dev tier at concurrency 32 and 128 on one VM, ~20 min, ~$1.50);
  first real `python -m regimes run all` against the engine and against vLLM.
- GPU, with approval and the auto-stop armed: `python -m lab.validate --target targets/inference-server.toml`
  (~3-4 h; serves unjailed, so not blocked), then a real `python -m lab.session --task ...` run.
- Layer 1, end to end (2026-10-15: off-GPU work done, #46-#54): one GPU run to validate the jailed evaluator,
  graph and paired timing, real inputs and the step-time share; one OpenEvolve run under them; `gate` then `ship`
  the winner and measure it through `lab.experiment` (`bench`, `equiv`, `submit`). Engine PR inference-server #106
  (the evolved kernel) is merged, its `equiv` not run. Generator improvements after that:
  `data/notes/kernel-generator-improvements.md` (local).
- Agent regime: undecided items in `DECISIONS.md` 2026-10-14.

## Store
Bucket `bg-data` (`.env`: `LAB_STORE`, `LAB_STORE_ENDPOINT`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`) holds the
ledger, its blobs, the corpus's held-out traces + seed, the holdout guard, and archives of `data/derived`,
`data/equiv*`, `data/h200-run-*`, `data/engine-h200-*`, `data/agent-traces` (2026-10-14).
