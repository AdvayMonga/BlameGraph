# BlameGraph

The referee side of the eval for autoresearch loops that optimize inference: correctness gate, adversarial canaries,
integrity verdict and session feedback. The loop, its engine and its measurement live in the user's repo
github.com/AdvayMonga/inference-server (custom engine, `lab/` environment, design in its `ENVIRONMENT.md`); don't
duplicate what is there. See the parent `../CLAUDE.md` for coding guidelines.

Hard rule from the user: the loop may receive **facts only, never heuristics or advice**. Integrity is a verdict
(valid/invalid + reasons); self-consistency and methodology are researcher diagnostics, never fed to the agent.

## Layout
- `blamegraph/feedback.py` — `feedback(path)` → `{for_agent: {integrity, facts}, for_researcher: {assertions, blame}}`; `python -m blamegraph feedback PATH [--agent-text]`. Accepts an InferenceBench run dir or a session ledger.
- `blamegraph/session.py` — the session format (append-only JSONL events: session_start, config, launch, measure, submission; grader hashes). The user's loop will get an adapter that writes/converts to this.
- `blamegraph/validate.py` — integrity rules → `Verdict`. Live mode (hash files on disk) or recorded mode (hashes from the submission event).
- `blamegraph/adapter.py` — session → `ExperimentLog`, so assertions/blame run on loop sessions.
- `blamegraph/traces.py`, `experiment.py` — InferenceBench trace loader and experiment-log reconstruction (configs, launches, evals, observations, stale/warm/standard flags). Regex-only number parsing.
- `blamegraph/assertions.py` — code-checkable assertions; `inject.py` — failure injectors; `blame.py` — found×kept×executed; `noise.py` — comparable observations + noise floor; `audit.py` — claims audit; `exploration.py` — knobs varied.
- `blamegraph/equivalence/` — correctness gate. `divergence.py` (teacher-forced KL / top-1 agreement / ref-token
  logprob shift, top-k or full logits), `flips.py` (paired flip test, exact one-sided McNemar), `checks.py` (length
  ratio, consistency), `gate.py` (`evaluate` → verdict with per-check gates; `calibrate` from known-good/known-bad
  candidates, refuses if inseparable; `to_ledger_record` → inference-server lab ledger `equiv` record),
  `scoring.py` (MMLU-Pro letters, MATH `\boxed{}`), `tasks.py` (loaders from the Hub into `data/tasks/`: MMLU-Pro
  490 stratified, MATH-500, MBPP sanitized 427 scored by running unit tests in a limited subprocess, synthetic
  needle-in-haystack 180 at ~4k/12k/24k tokens), `client.py` (vLLM-style `generate`, `generate_stream`,
  `score_tokens` via `prompt_logprobs`), `run.py` (`reference` once per model/hardware, `candidate` per change,
  `calibrate_files`). CLI: `python -m blamegraph equiv reference|candidate|calibrate`.
- `blamegraph/canaries.py` — cheat proxy in front of any OpenAI-compatible server (`truncate`, `early_eos`,
  `fake_first`, `drop`, `cache`, `inflate_usage`); `python -m blamegraph.canaries --upstream URL --cheat NAME`.
- `tests/test_feedback.py`, `tests/test_equivalence.py`, `tests/test_equiv_run.py` (full reference→calibrate→verdict
  flow on fake servers), `tests/test_canaries.py` (all synthetic, no GPU) and
  `tests/test_flip.py` (injection on real traces; skips without data). Run each with `python tests/<file>.py`.
- `data/` is gitignored and local only: `data/inferencebench/` (public traces, `hf download aisa-group/InferenceBench-Trajectories --repo-type dataset --local-dir data/inferencebench`), `data/derived/` (cached Haiku/Sonnet outputs from the research phase — the only copy), `data/laya/`.

## History
`research-v1` tag = full research snapshot: InferenceBench harness copy with hooks, Haiku extractor, Sonnet judge,
dashboard, Laya experiments, report/IRT/benchmark-audit scripts, loop tool (ledger/landscape/context). Restore from
there rather than re-deriving. Trace quirks (codex truncation, sandbox exit-126, no timestamps) are handled in
`traces.py`/`experiment.py`.

## Decided eval conventions (2026-10-02; change only with the user)
Target: Qwen3-30B-A3B (MoE) on one H100, **thinking off**. Engine under optimization: the user's own engine (inference-server). Reference for correctness: unmodified vLLM, BF16, `VLLM_BATCH_INVARIANT=1`. No method specifications to the agent (Bitter Lesson):
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
  distillation/pruning):** (1) primary: KL divergence vs the BF16 reference, teacher-forced on reference text over a
  large varied set, threshold calibrated so FP8/INT8 pass and a deliberately degraded quant fails; (2) paired
  flip test (McNemar) on mid-difficulty non-thinking tasks (MMLU-Pro, MATH-500, a unit-tested code set,
  long-context retrieval) — drop any task the model aces; (3) output length ≥90% of reference; (4) consistency
  (first token, single EOS, token count match the text). Reference runs in batch-invariant mode
  (`VLLM_BATCH_INVARIANT=1` / SGLang `--enable-deterministic-inference`); the agent's server need not be deterministic.
- **Validity:** noise bands per regime from repeated unchanged runs; clocks and engine/driver versions recorded;
  comparisons refused across hardware, versions, or corpus versions; MLPerf-style repeated-request audit (same
  request must not get >10% faster) as an anti-caching canary.
- **Canaries:** honest regressions (slower kernel, KV leak, fewer admissions, slower startup) and adversarial
  cheats (cached answers, early EOS, silent drops, pre-clock work, benchmark-pattern detection); each must be flagged.

## Overlap with inference-server (checked 2026-10-02)
Exists there, don't rebuild: client-side open-loop measurement (`scripts/bench/replay_trace.py`, `bench_serving.py`),
per-request rows, noise bands + significance, comparison refusal (`research/compare.py`), process-start accounting,
corpus with seen/held-out + versioning (cold_start, steady_interactive, long_context, spike), sandbox/grader audit,
lab ledger (`lab/ledger.py`, kinds test/equiv/bench/profile/submit/finding). Absent there (built here): output
equivalence beyond fixed-prompt parity, quantization quality gate, flip tests, KL, adversarial canaries.
Requirement on their side: the OpenAI shim returns `logprobs: None`; the divergence check needs `prompt_logprobs`
(contract in `equivalence/client.py`). Regimes missing from their corpus (short/long output, overload, shared prefix)
belong in their corpus builder, not here.

## Next
Calibration on the H100: `equiv reference` against unmodified vLLM (BF16, VLLM_BATCH_INVARIANT=1); `equiv candidate`
for FP8 and INT8 (good) and a degraded quant (bad); `equiv calibrate`. Needs GPU spend approval. Engine-side work
(prompt_logprobs in the shim, honest canaries, missing corpus regimes) goes to inference-server on a branch.
