# Decisions

One entry per decision, newest first: what was decided, why (as recorded at the time), what it replaced. Append-only;
a reversed decision gets a new entry that names the old one. Change any of these only with the user.

## 2026-10-15 — State, decisions and runs kept in three files
`STATE.md` (what works now, rewritten in place), `DECISIONS.md` (this file, append-only), `RUNS.md` (every GPU run,
append-only). `CLAUDE.md` keeps instructions and layout. Why: `CLAUDE.md` had grown to 330 lines mixing all four, so
the current state was hard to find and explain.

## 2026-10-14 — Holdout guard: one budget per corpus version, in the store (#35)
The guard's state lives in the store (`heldout/guard/<version>/`), shared by every run and machine. Replaces a guard
per ledger, which gave a VM `--env` session a fresh budget (state not pushed, seed = the ledger path). Consequence: a
host without `LAB_STORE` cannot submit.

## 2026-10-14 — Launch profiles (branch `lab/profiles`, not on main)
One engine, a config per regime in a TOML file of the engine's tree, each profile gated on its own (`lab/profiles.py`).

## 2026-10-14 — Central goal: per-layer machinery
Make the loop rely less on the raw model and more on per-layer machinery. Four layers (kernels, model execution,
runtime, serving), each with data, a simulator, a hypothesis generator, measurement tools and an evaluator, all
answering to one end-to-end judge (`equiv` + `submit` + the held-out guard). The agent chooses the layer, given each
layer's share of time × distance from the ceiling as a fact. Layer 1 first. Status per cell: `STATE.md`.

## 2026-10-14 — Agent regime planned (not built)
Long-horizon agent traffic as a ninth regime scored on goodput; text from Exgentic agent-llm-traces v2, timing and
tool gaps from FreeInference. Undecided: 32k cap vs 128k with YaRN, Exgentic v2's missing license vs v1, the per-turn
latency limit.

## 2026-10-11 — No measured number without the agent's reasoning
Measured tools require an open `hypothesis` record and an `expect` {metric, value, unit, why}; the record keeps both
with the computed gap; the next measured call is refused until a `reflection` on the last one is filed
(`lab/experiments.py`).

## 2026-10-11 — Every measured number with its context
`% of ceiling` and `% of reference`, shown only under a matching config fingerprint, else the field says why it is
missing (`lab/calibrate.py`). The performance reference is the correctness reference minus its determinism switches,
which are slow by design.

## 2026-10-09 — No knowledge base of beliefs
Neither hand-written nor the agent's: a belief bounded by its evidence is a measurement row; anything more is a
generalization. The findings in `knowledge/` were removed (tag `archive/knowledge-base`); measurement stays
(`knowledge/peaks/`, `knowledge/reference/`). The agent reads raw data only, through `ledger` and `query`.

## 2026-10-09 — Data store; held-out traces out of git
R2 (or a directory) with write-once keys, DuckDB index rebuilt from the ledger and blobs (`store/`). Held-out traces
and the corpus seed are gitignored and live in the store from corpus version `131149…`; the previous version's
held-out split was public in git. VMs never hold store keys.

## 2026-10-08 — On Linux the jail is a container with the NVIDIA runtime
`lab/safety/container.py`, `lab/RUNTIME.md`. Why: srt mounts a fresh minimal `/dev` with no device passthrough, so
nothing in an srt jail sees the GPU, and each srt jail on Linux has its own network namespace, so a jailed server is
unreachable (found in the 2026-10-08 run, `RUNS.md`). srt stays elsewhere.

## 2026-10-08 — Controller / worker split
Controller on the laptop (ledger, referee, secrets), worker on the GPU VM (jails, clients, profilers) behind a job
interface; no separate measurement VM. Design: `lab/RUNTIME.md`.

## 2026-10-06 — Facts only to the loop ("option c")
The loop receives facts only, never heuristics or advice. At session start: the integrity verdict with its evidence,
the raw ledger, and only the facts it cannot derive from the ledger (`lab/evaltools.harness_facts`). Nothing derived
from the records goes to the agent; that is `for_researcher.activity`.

## 2026-10-06 — Engine and lab in separate repos; target specs
inference-server holds only the engine and its stack (pre-split tree: tag `archive/pre-split` there); everything that
runs, measures or judges it lives here. The engine is described by a target spec (`targets/<name>.toml`); nothing
defaults to the user's engine or to Qwen3.

## 2026-10-04 — Correctness: MLPerf's accuracy rule
Pooled task score ≥ 99% of the BF16 reference's, output length ≥ 90%, consistency; KL and per-task flips reported as
facts, not gated. Replaces the KL/flip-rate thresholds derived in the 2026-10-04 calibration.

## 2026-10-04 — InferenceBench harness code removed from history
Research snapshot kept as tag `research-v1` (Haiku extractor, Sonnet judge, dashboard, Laya experiments,
report/IRT/benchmark-audit scripts, loop tool). Restore from there rather than re-deriving.

## 2026-10-02 — Eval conventions
Target: Qwen3-30B-A3B (MoE) on one H200 (Verda; the user has no H100 access), **thinking off**. Engine under optimization: the user's own engine (inference-server). Reference for correctness: unmodified vLLM, BF16, `VLLM_BATCH_INVARIANT=1`. No method specifications to the agent (Bitter Lesson):
the benchmark defines objective + correctness + validity only, checked end to end (no internal-invariant checks).
- **Regimes (8):** single stream, saturated, bursty, long prompt/short output, short prompt/long output, shared
  prefix multi-turn, overload, cold start. All always run; the task picks the objective (one, several, or all, with
  optional "don't get worse than X" limits). Each regime is a seeded recipe; workloads sampled fresh per run;
  seen / held-out (query budget) / sealed splits; corpus versioned.
- **Metrics (client-side, per-request rows):** TTFT from scheduled send to first *real* token; TPOT =
  (E2E − TTFT)/(n − 1); tokens re-counted with the reference tokenizer; goodput = max rate with ≥99% of requests
  meeting both TTFT and TPOT limits (p99); failures; peak memory as a limit, not an objective; cost = goodput per
  GPU-second from process start; joules/token. Latency limits (MLPerf Llama-3.1-8B, checked at p99):
  conversational TTFT ≤ 2000 ms / TPOT ≤ 100 ms (saturated, bursty, long prompt, long output, shared prefix,
  overload); interactive TTFT ≤ 500 ms / TPOT ≤ 30 ms (single stream). Check at calibration whether the
  long-prompt regime's prompt lengths fit 2000 ms TTFT on the unmodified baseline.
- **Load:** open-loop Poisson for bursty/overload/limit-finding; closed-loop doubling concurrency sweep for
  saturated; concurrency 1 for single stream; warmup outside the window; full tier ≥600 s per regime; short tier
  validated by rank correlation with the full tier; client must be shown not to be the bottleneck.
- **Overload:** explicit error = handled but not goodput; truncation, silent drop, or crash = invalid.
- **Cold start:** process launch → first correct token; weights on disk allowed, compile caches empty; warm-cache
  figure reported alongside.
- **Correctness gate (quantization allowed, MLPerf-style: purely mathematical, original weights, no retraining/
  distillation/pruning):** MLPerf's accuracy rule (decided 2026-10-04): (1) pooled task score ≥ 99% of the
  BF16 reference's, over every item of MMLU-Pro, MATH, a unit-tested code set and long-context retrieval; (2) output
  length ≥ 90% of reference; (3) consistency (first token, single EOS, token count match the text). Teacher-forced
  KL divergence and per-task paired flips (McNemar) are reported as facts, not gated. Tiers: `dev` (~1.6k items) while
  iterating, where the 99% line is within noise (~±0.7%); `full` (~18.6k, ~±0.2%) for final tests. Reference runs in batch-invariant mode
  (`VLLM_BATCH_INVARIANT=1` / SGLang `--enable-deterministic-inference`); the agent's server need not be deterministic.
- **Validity:** noise bands per regime from repeated unchanged runs; clocks and engine/driver versions recorded;
  comparisons refused across hardware, versions, or corpus versions; MLPerf-style repeated-request audit (same
  request must not get >10% faster) as an anti-caching canary.
- **Canaries:** honest regressions (slower kernel, KV leak, fewer admissions, slower startup) and adversarial
  cheats (cached answers, early EOS, silent drops, pre-clock work, benchmark-pattern detection); each must be flagged.
