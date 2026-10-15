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

Across layers, built: the end-to-end judge, calibration (`lab/calibrate.py`), experiment records
(`lab/experiments.py`; a closing reflection call per session; a person's experiments via `lab/experiment.py`, hidden
from the agent), the workbench/measurement sandbox, the GPU VM tooling.

## Validated on a real GPU (evidence in `RUNS.md`)
- Correctness gate, dev and full tier (2026-10-04, 2026-10-05).
- Kernel checker and timer (2026-10-04).
- Profiler tools `profile`, `trace`, `kernel`, `hostprof` (2026-10-08).
- The agent loop end to end, one session, `LAB_WORKER=ssh` (2026-10-10).
- Jails, workbench, quiet, exclusivity, dependency build, ncu: `lab.gpucheck` (2026-10-13).
- Ceilings below every measurement (2026-10-13).

## Built, not yet run on a GPU
- `python -m lab.validate` (reference, good/bad launches, noise bands): tested on fakes only.
- `python -m regimes run all` against the engine and against vLLM.
- A real `python -m lab.session --task ...` run.
- `validity/tier_agreement.py`: needs real short/full runs.
- Remote worker (`LAB_WORKER=ssh`): used in the one-session dry run only.

## Where the engine stands (2026-10-13)
Engine vs plain vLLM: single_stream TPOT p99 21 ms vs 4.1 ms (7% vs 37% of ceiling), saturated 0.097 vs 5.8 req/s
(0.5% vs 19%), long prompt invalid (111 requests without a response) vs 68 req/s (94%).

## In flight (branches not on main)
| Branch | Worktree | What | Status |
|---|---|---|---|
| `lab/profiles` | `BlameGraph-wt/profiles` | Launch profiles: a config per regime | Not merged |
| `target/batch-size` | `BlameGraph-wt/batch` | The engine's own batch size, not a pinned 32 | One commit, not merged |
| `gpu/session` | `BlameGraph-wt/gpu` | Merges `lab/profiles` + `target/batch-size` | Active 2026-10-15; purpose not recorded |

## Open problems
- BF16 puzzle: plain BF16 matched the reference on Nebius at concurrency 128 but diverged on Verda at 32
  (`RUNS.md` 2026-10-05). Don't treat a BF16 rerun as a noise sample until resolved.
- `equiv` at tier dev takes > 46 min on this engine (vLLM ~10 min); the tool descriptions' "10 min at tier dev" is wrong.
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
