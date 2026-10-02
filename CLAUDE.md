# BlameGraph

Feedback tool for autoresearch loops that optimize inference (servers now, kernels next). The user is building the
loop itself (measurement + context retrieval) in a separate repo; BlameGraph only *judges finished sessions*.
See the parent `../CLAUDE.md` for coding guidelines.

Hard rule from the user: the loop may receive **facts only, never heuristics or advice**. Integrity is a verdict
(valid/invalid + reasons); self-consistency and methodology are researcher diagnostics, never fed to the agent.

## Layout
- `blamegraph/feedback.py` — `feedback(path)` → `{for_agent: {integrity, facts}, for_researcher: {assertions, blame}}`; `python -m blamegraph feedback PATH [--agent-text]`. Accepts an InferenceBench run dir or a session ledger.
- `blamegraph/session.py` — the session format (append-only JSONL events: session_start, config, launch, measure, submission; grader hashes). The user's loop will get an adapter that writes/converts to this.
- `blamegraph/validate.py` — integrity rules → `Verdict`. Live mode (hash files on disk) or recorded mode (hashes from the submission event).
- `blamegraph/adapter.py` — session → `ExperimentLog`, so assertions/blame run on loop sessions.
- `blamegraph/traces.py`, `experiment.py` — InferenceBench trace loader and experiment-log reconstruction (configs, launches, evals, observations, stale/warm/standard flags). Regex-only number parsing.
- `blamegraph/assertions.py` — code-checkable assertions; `inject.py` — failure injectors; `blame.py` — found×kept×executed; `noise.py` — comparable observations + noise floor; `audit.py` — claims audit; `exploration.py` — knobs varied.
- `tests/test_feedback.py` (synthetic sessions, validator flips) and `tests/test_flip.py` (injection on real traces; skips without data).
- `data/` is gitignored and local only: `data/inferencebench/` (public traces, `hf download aisa-group/InferenceBench-Trajectories --repo-type dataset --local-dir data/inferencebench`), `data/derived/` (cached Haiku/Sonnet outputs from the research phase — the only copy), `data/laya/`.

## History
`research-v1` tag = full research snapshot: InferenceBench harness copy with hooks, Haiku extractor, Sonnet judge,
dashboard, Laya experiments, report/IRT/benchmark-audit scripts, loop tool (ledger/landscape/context). Restore from
there rather than re-deriving. Trace quirks (codex truncation, sandbox exit-126, no timestamps) are handled in
`traces.py`/`experiment.py`.

## Decided eval conventions (2026-10-02; change only with the user)
Target: Qwen3-30B-A3B (MoE) on one H100, **thinking off**. No method specifications to the agent (Bitter Lesson):
the benchmark defines objective + correctness + validity only, checked end to end (no internal-invariant checks).
- **Regimes (8):** single stream, saturated, bursty, long prompt/short output, short prompt/long output, shared
  prefix multi-turn, overload, cold start. All always run; the task picks the objective (one, several, or all, with
  optional "don't get worse than X" limits). Each regime is a seeded recipe; workloads sampled fresh per run;
  seen / held-out (query budget) / sealed splits; corpus versioned.
- **Metrics (client-side, per-request rows):** TTFT from scheduled send to first *real* token; TPOT =
  (E2E − TTFT)/(n − 1); tokens re-counted with the reference tokenizer; goodput = max rate with ≥99% of requests
  meeting both TTFT and TPOT limits (p99); failures; peak memory as a limit, not an objective; cost = goodput per
  GPU-second from process start; joules/token. Latency limits: not yet fixed (borrow MLPerf Llama-3.1-8B
  2000/100 ms conversational, 500/30 ms interactive, or set from the baseline).
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

## Next
Corpus generator and correctness gate (both GPU-free to build), then load client/metrics/validity, canaries;
adapter for the user's loop logs once that repo is ready. Avoid duplicating measurement the user's loop already has.
