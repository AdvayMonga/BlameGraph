# BlameGraph

The environment for autoresearch loops that optimize LLM inference: the lab (agent loop, tools, ledger, sandbox,
GPU VMs), the workload corpus and knowledge base, load regimes, the correctness gate, canaries, and the referee
(integrity verdict, session feedback). Design in `ENVIRONMENT.md`. The engine under optimization is a separate repo,
github.com/AdvayMonga/inference-server (engine + stack only since the 2026-10-06 split); the lab finds it via
`LAB_ENGINE_REPO` (default `../inference-server`) and runs it with its own `.venv/bin/python`. See the parent
`../CLAUDE.md` for coding guidelines.

Hard rule from the user: the loop may receive **facts only, never heuristics or advice**. Integrity is a verdict
(valid/invalid + reasons); self-consistency and methodology are researcher diagnostics, never fed to the agent.

## Layout
Top-level packages, run from the repo root (no install step; `pyproject.toml` lists optional extras and holds the
pytest/ruff config). Each folder with commands has its own `python -m`.
- `lab/` — moved from inference-server 2026-10-06 (`lab/README.md`). `session.py` (the loop over a dollar budget),
  `agent.py` (the one place a model is called; the agent's jailed shell gets the engine's python on PATH),
  `tools.py` (metered tools: test, profile, ledger, budget, restore, note; bench/equiv/submit still
  refuse until wired to `regimes`/`correctness`), `workspace.py` (exported engine copy + snapshots), `ledger.py`
  (append-only JSONL, `lab/ledger/` gitignored), `budget.py`, `engine.py` (where the engine repo and its python
  are), `safety/` (write surfaces, srt jail, grader: lint/tests run with the engine's python in the jail),
  `profile.py` + `bundle.py` + `gpu.py` (profiler harness, staged into the pristine tree under `_harness/` so the
  jail never opens this repo, which holds `.env` and the ledger), `canary.py` (honest regressions patched into
  the engine's process), `vm.py` + `providers/` + `vm-setup.sh` (one GPU VM on Verda/Nebius/Crusoe; pushes both
  trees, engine to `LAB_VM_DIR`, this repo to `LAB_VM_ENV_DIR`; `run --env` runs here), `corpus.py` +
  `chat_template.py` (corpus loader, template fingerprint). Tests: `tests/test_lab_*.py` (pytest); the ones
  that drive engine code skip unless the engine and its deps import (run them with the engine's python).
- `corpus/` — frozen workload traces (BurstGPT timing, WildChat text), seen/heldout, hashed, `manifest.json`;
  `build_corpus.py` / `fetch_traces.py` rebuild it; any change is a new `corpus_version`.
- `knowledge/` — measured findings (one JSON each) + `evidence/`; seeded into the ledger as `finding` records.
- `feedback/` — what each side gets after a session. `report.py`: `feedback(path)` → `{for_agent: {integrity, facts}, for_researcher: {assertions, blame}}`; `python -m feedback PATH [--run RUN_ID] [--agent-text]`. Accepts an InferenceBench run dir, a BlameGraph session ledger, or an inference-server lab ledger (detected from its first line). Integrity carries `evidence` per violated rule (rule, plain detail, concrete refs: hashes, measurement ids, offending values and limits).
  - `verdict.py` — integrity rules → `Verdict(valid, reasons, evidence, facts)`. Live mode (hash files on disk) or recorded mode (hashes from the submission event). Physical limits (`unphysical()`) are shown to the agent on purpose: they are hardware facts, not detection tricks.
  - `lab_verdict.py` — inference-server lab ledger → integrity + facts directly (snapshots stand in for configs; no ExperimentLog, so `for_researcher` is empty for lab ledgers). Assumed result shapes are in its docstring; their bench/equiv/submit tools don't write results yet. Submitted snapshot must have a completed seen-split bench, tests passed, and equiv with no failing record (retrying a statistical gate until it passes would let bad changes through).
  - `claim_facts.py` — claim-vs-evidence facts from ledger `claim` text (regex, no LLM): unrecorded numbers, contradicted speedups, verification words with no matching record, unknown record ids. Facts only; never part of the verdict (their lab/README rule).
- `logs/` — reading sessions. `session.py`: the session format (append-only JSONL events: session_start, config, launch, measure, submission; grader hashes). `inferencebench.py` (trace loader) and `reconstruct.py` (experiment-log reconstruction: configs, launches, evals, observations, stale/warm/standard flags; regex-only number parsing). `to_experiment.py`: session → `ExperimentLog`, so assertions/blame run on loop sessions.
- `diagnostics/` — researcher-only, never fed to the agent. `assertions.py` (code-checkable assertions), `inject.py` (failure injectors), `blame.py` (found×kept×executed), `noise.py` (comparable observations + noise floor), `claims_audit.py` (claims audit on InferenceBench traces), `exploration.py` (knobs varied).
- `correctness/` — correctness gate. `divergence.py` (teacher-forced KL / top-1 agreement / ref-token
  logprob shift, top-k or full logits), `flips.py` (paired flip test, exact one-sided McNemar), `checks.py` (length
  ratio, consistency), `gate.py` (`evaluate` → verdict; gates = pooled accuracy ≥ 99% of reference, length,
  consistency; divergence and per-task flips are reported facts; `to_ledger_record` → lab ledger `equiv` record),
  `scoring.py` (MMLU-Pro letters, MATH `\boxed{}`), `tasks.py` (loaders from the Hub into `data/tasks/`; tier `dev`
  ~1.6k = MMLU-Pro 490 stratified, MATH-500, MBPP sanitized 427, needle 180; tier `full` ~18.6k = MMLU-Pro 12,032,
  MATH 5,000 (gold = last `\boxed{}` of the solution), MBPP full 974, needle 600; code scored by running unit tests
  in a limited subprocess; needle = synthetic retrieval at ~4k/12k/24k tokens), `client.py` (vLLM-style `generate`,
  `generate_stream`, `score_tokens` via `prompt_logprobs`), `run.py` (`reference` once per model/hardware/tier,
  `candidate` per change, `verdict_file`). CLI: `python -m correctness reference [--tier dev|full]|candidate|verdict`
  (`verdict` re-judges saved results under the policy, no server).
- `canaries/cheat_proxy.py` — cheat proxy in front of any OpenAI-compatible server (`truncate`, `early_eos`,
  `fake_first`, `drop`, `cache` (streamed and not), `inflate_usage`); `python -m canaries --upstream URL --cheat NAME`.
- `validity/` — `roofline.py` (analytical decode ceiling, no GPU; to be generalized into physical limits), `repeat_audit.py`: repeated-request audit (MLPerf TEST04-style), judges decode time per token, not TTFT, so prefix caching passes and replayed answers fail; `python -m validity repeat --url URL`. `tier_agreement.py`: `tier_agreement(changes)`: Spearman, Kendall tau-b, bootstrap CI, out-of-band disagreements; short tier usable if n ≥ 8, rho ≥ 0.8, CI low ≥ 0.5. `holdout.py`: `HoldoutGuard` (Thresholdout query budget, state persisted atomically under a file lock, deterministic from seed) and `SealedSplit` (hash manifest, one logged unseal per corpus version). Keep sigma small relative to the threshold.
- `kernels/` — kernel equivalence: `check_kernel` (seen + held-out shapes, per-dtype tolerances, edge cases incl. non-contiguous and NaN/Inf, inputs unchanged, no aliasing, memoization, determinism) and `time_kernel` / `check_memoization` (sync, rotated buffers, L2 flush on CUDA). Validated on CPU, MPS and CUDA (H200, 2026-10-04): all tests pass; the timer resolves ~20 µs kernels at ~9% IQR, 0.2 ms at 1.5%. `make_inputs` gets a seeded CPU generator (generate on CPU, then `.to(device)`).
- `regimes/` — load generation: drives a live OpenAI-compatible server the eight ways it gets used, client-side.
  `client.py` (one streamed chat request: TTFT from *scheduled* send to first real token, TPOT = (E2E − TTFT)/(n − 1),
  status ok / error (explicit 4xx or 503 rejection) / silent_drop / truncated / crash (other 5xx, no response,
  stream died); a one-token answer has no TPOT and never meets the limits; corpus headers X-Session-Id,
  X-Turn-Index, X-Trace-Id), `runner.py` (open loop: schedule dealt to 4 client processes on one shared start, each
  request on a thread started 5 ms early, so p99 send lag stays ~1–3 ms; closed loop; `summarize`: p50/p99, failures,
  attainment, valid only if no silent drop/truncation/crash and client lag p99 ≤ 10 ms; `find_goodput`: double then
  bisect on ≥99% meeting both limits), `workload.py` (reads `corpus/` through `lab.corpus`'s verified loader — hashes checked, a changed trace
  refuses — plus Poisson, trace replay with speed-up and peak-window crop, multi-turn
  conversations rebuilt from stored histories under one shared system prompt; synthetic prompts when no corpus),
  `suite.py` (the 8 regimes; probes ≥ `min_requests` (100) and tier length, capped at 4× the tier length, then a
  confirmation at `final_s` backing off 15% up to 5 times; the load search stops below start/16; closed-loop
  schedules are generated lazily; shared-prefix runs score only the steady-state window; cold start launches a command with
  fresh VLLM/inductor/Triton cache dirs, polls until "17 + 25" is answered "42", then relaunches for the warm figure).
  Tiers: short (30 s probes, 60 s final), full (60 s probes, 600 s final). Probes must last well beyond the TTFT limit
  or a short overload goes unseen; limits are settable per `Ctx` (MLPerf by default). CLI:
  `python -m regimes run REGIME[,..]|all --url URL [--corpus DIR] [--split] [--tier] [--tokenizer] [--out]`,
  `python -m regimes cold-start --url URL --cmd "..."`. Closed-loop runs can't detect a client bottleneck by lag.
- `tests/test_*.py` — all synthetic, no GPU, each runnable with `python3 tests/<file>.py`, named after the module they cover; `test_correctness_run.py` is the full reference→candidate→verdict flow on fake servers; `test_kernels.py` also runs on MPS; `test_inject.py` injects failures into real traces and skips without data; `test_regimes.py` drives `tests/fake_engine.py` (fixed decode slots, run out of process) through every regime (~2.5 min).
- `data/` is gitignored and local only: `data/inferencebench/` (public traces, `hf download aisa-group/InferenceBench-Trajectories --repo-type dataset --local-dir data/inferencebench`), `data/derived/` (cached Haiku/Sonnet outputs from the research phase — the only copy), `data/laya/`.

## History
`research-v1` tag = research snapshot (InferenceBench harness code removed from history 2026-10-04): Haiku extractor, Sonnet judge,
dashboard, Laya experiments, report/IRT/benchmark-audit scripts, loop tool (ledger/landscape/context). Restore from
there rather than re-deriving. Trace quirks (codex truncation, sandbox exit-126, no timestamps) are handled in
`traces.py`/`experiment.py`.

## Decided eval conventions (2026-10-02; change only with the user)
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

## The engine repo (inference-server, after the 2026-10-06 split)
Holds only the engine and its stack: `src/` (engine, control plane), engine tests, `monitoring/`, and engine tools
(`scripts/bench/tune_triton_launch.py`, `scripts/gpu_tests/checks.py`, `scripts/tools/smoke_custom.py`). The
combined pre-split tree is tag `archive/pre-split` there. Its shim serves `prompt_logprobs` (contract in
`correctness/client.py`). Everything that runs, measures or judges it lives here; the user builds the agent's tools
(e.g. a knowledge tool) themselves, so don't add agent tools without asking. Corpus classes for the regimes the
corpus lacks (short/long output, overload, shared prefix) belong in `corpus/build_corpus.py`; the regimes synthesize
them from existing records meanwhile.

## Full-tier recalibration (H200, Nebius, 2026-10-05)
Policy confirmed, nothing tuned. 18,606 items, vLLM 0.30.0, concurrency 128, 3 h 40 min, ~$16.50. Local only:
`data/h200-run-20261005/REPORT.md`, `data/equiv-full/`. Pooled score ratio: plain BF16 1.0000, FP8 (DeepGEMM off)
1.0009, INT8 `nytopop/Qwen3-30B-A3B.w8a8` 0.9993, FP8 (DeepGEMM on, nvcc 13.0) 1.0034 — all PASS; GPTQ-Int4 0.9869
FAIL (MMLU-Pro 0.9807, p = 3e-5). FP8/INT8 each lose and gain ~800 answers in balance: that churn is the realistic
noise. Consistency clean on every run.
Open: plain BF16 matched the batch-invariant reference exactly here (18,602/18,606 identical, KL 0.0), but on Verda at
concurrency 32 it diverged on 466/490 MMLU-Pro answers (median 13% into the answer). Logs show both BF16 servers plain
(non-invariant compile path, requests served by the BF16 server); the reference reproduced across both days and
clouds (691/691 same-prompt answers identical). Differences: venue, driver 580.126 vs 580.173, concurrency 32 vs 128.
Don't treat a BF16 rerun as a noise sample until resolved.

## Calibration, dev tier (H200, Verda, 2026-10-04)
Ran via inference-server's `lab/vm.py`: vLLM 0.30.0, torch 2.13 + CUDA 13.0, clocks locked 1980 MHz, 83 min, $6.75.
Report and outputs are local only: `data/h200-run-20261004/REPORT.md`, `data/equiv/` (ref, candidates, `thresholds.json`).
Dev tier. Candidates: plain BF16 (no batch-invariant mode), official FP8 (served with `VLLM_USE_DEEP_GEMM=0`: the
image's nvcc 12.8 is too old for DeepGEMM), official GPTQ-Int4. KL: BF16 3.7e-7, FP8 3.0e-3, Int4 1.4e-2 (clean
separation). Per task, accuracy can't separate at this size: plain BF16 alone flips 8.2% of MMLU-Pro answers and
loses 1.9% there; no task reaches significance. Pooled score ratio (the adopted MLPerf rule): BF16 1.0008, FP8
1.0072, Int4 0.9847 (fails 0.99; McNemar p 0.049). The KL/flip-rate thresholds derived that day were superseded by
the MLPerf policy.
INT8 (`JunHowie/Qwen3-30B-A3B-GPTQ-Int8`) was skipped: vLLM 0.30 can't load act-order GPTQ MoE experts. Every
consistency result is void: the client counted vLLM's role chunk as a fake first token (fixed on
`fix/calibration-findings`). Repeat audit passes on the honest server and fails the `cache` canary.

## Next
GPU (needs spend approval): resolve the BF16 puzzle — plain BF16 on the dev tier at concurrency 32 and 128 on one
VM (~20 min, ~$1.50).
Their side: the user builds the agent's tools (not this repo; don't push tools there). In their lab: wire `equiv` to
`python -m correctness candidate` once it can launch a candidate engine; `tiers.py` needs real short/full runs; missing
corpus regimes are their call. README stays a few one-line feature bullets, not a dev doc.
