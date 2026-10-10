# BlameGraph

The environment for autoresearch loops that optimize LLM inference: the lab (agent loop, tools, ledger, sandbox,
GPU VMs), the workload corpus and knowledge base, load regimes, the correctness gate, canaries, and the referee
(integrity verdict, session feedback). Design in `ENVIRONMENT.md`. The engine under optimization is a separate repo,
github.com/AdvayMonga/inference-server (engine + stack only since the 2026-10-06 split); the lab finds it via
`LAB_ENGINE_REPO` (default `../inference-server`) and runs it with its own `.venv/bin/python`. See the parent
`../CLAUDE.md` for coding guidelines.

Hard rule from the user: the loop may receive **facts only, never heuristics or advice**. Decided 2026-10-06
("option c"): at the start of every session the agent gets the integrity verdict with its evidence, the raw ledger
(the `ledger` tool, unfiltered) and only the facts it cannot derive from that ledger (`lab/evaltools.harness_facts`:
the limits, tiers, policy, noise bands, corpus version the referee measures against). Nothing derived from the
records (counts, best measured, claims vs evidence) goes to the agent; that is `for_researcher.activity`.
The task (`lab/task.py`, `--task task.toml`: goal, objective regimes, constraints) is stated in full in the brief,
and a submit returns its `score` under that task: the win condition is known, never inferred. The system prompt is
generated per session (`session.system_prompt`: template `lab/prompts/system.md` filled from the target spec and the
ToolSpec descriptions, which state what each tool does, returns and costs, never when to use it). The brief is the
task, budget, referee, a BASELINE (the base commit under bench's defaults = the task's objective + constrained regimes, measured once per run, `baseline` ledger
kind; `RunConfig.baseline=False` skips it) and only the run's last `session` record.

## Layout
Top-level packages, run from the repo root (no install step; `pyproject.toml` lists optional extras and holds the
pytest/ruff config). Each folder with commands has its own `python -m`.
- `targets/<name>.toml` + `lab/target.py` — **the target spec** (since 2026-10-06): model + chat kwargs, engine
  (repo, python, launch command with `{port}`, health path, env, logprobs `api` vllm|sglang|none, write surface,
  add-only globs, test and lint commands, optional `[engine.telemetry] env` with `{dir}`), reference server + its outputs dir, MLPerf latency limits, correctness
  tasks and policy, corpus dir. Selected by `LAB_TARGET` or `--target`; `target.load()` raises `NoTarget` otherwise.
  Nothing defaults to the user's engine or to Qwen3; `targets/inference-server.toml` is the committed example and
  what `tests/conftest.py` points at. Surfaces, grader commands, `lab/engine.py`, and the `correctness`,
  `regimes`, `validity` CLIs all read it.
- `lab/` — moved from inference-server 2026-10-06 (`lab/README.md`). `session.py` (the loop over a dollar budget),
  `agent.py` (the one place a model is called; the agent's jailed shell gets the engine's python on PATH;
  `LAB_AGENT_PROVIDER` claude = Agent SDK CLI, openai = `agent_openai.py`: any OpenAI-compatible API with its own
  key, https public hosts only, `LAB_MODEL_PRICE` required, loop outside the jail, Bash jailed with no network; on
  Linux the Claude CLI runs in a per-session workbench container: GPU, internet, root inside, engine venv read-only;
  its model calls go through `apiproxy.py`, which injects the key, prices usage and enforces the session cap; the
  workbench is paused by `container.quiet` around measure/equiv/profile jobs, GPU holders killed and recorded),
  `tools.py` (metered tools: test, profile, ledger, budget, restore, note; every record's `cost` has wall seconds, and
  GPU tools charge them at `LAB_GPU_USD_PER_HOUR` / the target's `[cost] gpu_usd_per_hour` to the run's budget) +
  `proftools.py` (trace = nsys, kernel = ncu, hostprof = py-spy, each wrapping lab.profile's workload in the jail
  with validated args) + `evaltools.py` (bench → `regimes` on
  the seen split; equiv → `correctness` candidate vs the target's reference; submit → held-out full tier vs the
  base commit, one aggregate per regime through the Thresholdout guard; nothing is measured while another process holds
  the GPU (`refused`), and a process other than the engine on the GPU mid-run makes it `contaminated` (never evidence,
  never the cached base; `serve.Served.exclusive()`); all three serve the pristine tree via
  `serve.py`, jailed (on Linux, port bridged out of a network-less container); every serve also keeps its passive data, `artifacts.py`: device
  samples via `gpu.DeviceSampler` (DCGM if `dcgmi`, else nvidia-smi), engine telemetry files from `{dir}`, all
  client rows, the serve log; bench/equiv as ledger blobs under `result.artifacts`, submit only in
  `<run>/heldout-private/`, never in the ledger; `artifacts.joined` joins client rows to engine rows by trace id),
  `workspace.py` (exported engine copy + snapshots), `ledger.py`
  (append-only JSONL, `lab/ledger/` gitignored), `budget.py`, `engine.py` (where the engine repo and its python
  are), `safety/` (write surfaces, grader: lint/tests run with the engine's python in the jail; the jail is `container.py` on
  Linux (one container per jailed command, NVIDIA runtime, `--network none`, socat bridge for a served port; tested in
  Docker-in-Docker, GPU untested) and srt elsewhere, `LAB_JAIL` overrides),
  `profile.py` (`--profiler torch|cuda-range|pyspy`) + `bundle.py` + `gpu.py` (profiler harness, staged into the pristine tree under `_harness/` so the
  jail never opens this repo, which holds `.env` and the ledger; the tool samples its seen-split corpus workload out here, `corpus.sample`, and stages it as `_harness/workload.json`; window-only memory peaks and scheduler counters, warmup sized to the window, op/kernel tables), `canary.py` (honest regressions patched into
  the engine's process), `vm.py` + `providers/` + `vm-setup.sh` (one GPU VM on Verda/Nebius/Crusoe; pushes both
  trees, engine to `LAB_VM_DIR`, this repo to `LAB_VM_ENV_DIR`; `run --env` runs here; `run` stops the VM when done
  unless `--keep`, and every session is capped by `LAB_VM_MAX_MINUTES` via `timeout` + a local watchdog + in-VM
  shutdown, because a stalled session once billed 12 h for 3.5 h of work), `corpus.py` +
  `chat_template.py` (corpus loader, template fingerprint). Tests: `tests/test_lab_*.py` (pytest); the ones
  that drive engine code skip unless the engine and its deps import (run them with the engine's python).
- `lab/validate.py` — `python -m lab.validate --target T`: serves the target's reference and writes its outputs, judges
  each `[validation]` good/bad launch (good must pass, bad must fail: the gate separates on *this* model and GPU), and
  measures each regime's run-to-run band from repeated reference runs into `knowledge/noise/<regime>.json` (which
  `submit` reads). Exit 0 only if both hold. Needs a GPU for a real target; tested on fakes. `lab/LEDGER.md` is the
  record-format spec: the environment's public interface.
- `lab/RUNTIME.md` — design (2026-10-08): controller (laptop: ledger, referee, secrets) and worker (GPU VM: jails,
  clients, profilers) behind a job interface; trust zones and rules; why srt cannot jail GPU code on Linux.
- `lab/build.py` — the dependency build step: a tree whose change touched the target's `deps` (pyproject: dependency
  tables only, checked by the audit) gets a venv built by the target's `build` command in a no-GPU, `--network none`
  container whose only way out is a CONNECT proxy to the package-index hosts; cached by the files' hash; jobs run
  under `engine.using(build.python_for(tree))`.
- `lab/worker.py` — the job interface, built for the measured jobs: `test`, `measure`, `equiv`, `profile` take input
  trees + JSON args, write an out dir, return JSON; the tools call `self.worker`. `LAB_WORKER=local` (default, in
  process) or `ssh` (the `lab.vm` VM): inputs by content hash, re-hashed on the worker before every job; GpuBusy,
  Contaminated, NotReady, ValueError cross intact. Remote is tested through a local shell, not yet over SSH. With
  `ssh` the agent works on the VM too: its workspace mirrored by rsync (pulled before every audit), its workbench
  built there (`workbench`), its CLI's stdio over `ssh -R`, the API proxy tunnelled to a unix socket in the workbench.
- `lab/TRACE.md` + `lab/trace_contract.py` — the request-trace contract: per-request engine rows (trace_id = the
  client's `X-Trace-Id`, `arrival_ts` epoch s, `*_s` monotonic durations, TPOT = (E2E − TTFT)/(n − 1)) in a dir of
  `*.sqlite` (`requests` + `meta`) or `*.jsonl` + `<stem>.meta.json`, with schema version and drop counts;
  `python -m lab.trace_contract DIR`. Diagnostics only, never scored. inference-server's `TELEMETRY_DIR` conforms
  natively (schema 1, inference-server#102); the target names its env vars in `[engine.telemetry] env` (`{dir}`); serve collects
  the files after every bench/equiv/submit and the contract check's report is stored beside them. vLLM OTel span mapping is in the spec, not implemented.
- `corpus/` — frozen workload traces (BurstGPT timing, WildChat text), seen/heldout, hashed, `manifest.json`;
  `build_corpus.py` / `fetch_traces.py` rebuild it; any change is a new `corpus_version`.
- `knowledge/` — measured findings (one JSON each) + `evidence/`; seeded into the ledger as `finding` records.
- `feedback/` — what each side gets after a session. `report.py`: `feedback(path)` → `{for_agent: {integrity, facts}, for_researcher: {assertions, blame}}` (lab ledgers: `lab_feedback`, `for_agent.facts = {harness}` only, `for_researcher.activity` holds the derived facts); `python -m feedback PATH [--run RUN_ID] [--agent-text]`. Accepts an InferenceBench run dir, a BlameGraph session ledger, or an inference-server lab ledger (detected from its first line). Integrity carries `evidence` per violated rule (rule, plain detail, concrete refs: hashes, measurement ids, offending values and limits).
  - `verdict.py` — integrity rules → `Verdict(valid, reasons, evidence, facts)`. Live mode (hash files on disk) or recorded mode (hashes from the submission event). Physical limits (`unphysical()`) are shown to the agent on purpose: they are hardware facts, not detection tricks.
  - `lab_verdict.py` — lab ledger → integrity + derived activity facts (snapshots stand in for configs; no ExperimentLog, so assertions/blame are None for lab ledgers). Record shapes are in `lab/LEDGER.md`; `gain()` reads submit aggregates (improved / regressed / within_band). Submitted snapshot must have a completed short-tier seen bench, tests passed, a passing equiv and no failing equiv (inconclusive is neither; retrying a statistical gate until it passes would let bad changes through).
  - `claim_facts.py` — claim-vs-evidence facts from ledger `claim` text (regex, no LLM): unrecorded numbers, contradicted speedups, verification words with no matching record, unknown record ids. Facts only; never part of the verdict (their lab/README rule).
- `logs/` — reading sessions. `session.py`: the session format (append-only JSONL events: session_start, config, launch, measure, submission; grader hashes). `inferencebench.py` (trace loader) and `reconstruct.py` (experiment-log reconstruction: configs, launches, evals, observations, stale/warm/standard flags; regex-only number parsing). `to_experiment.py`: session → `ExperimentLog`, so assertions/blame run on loop sessions.
- `diagnostics/` — researcher-only, never fed to the agent. `assertions.py` (code-checkable assertions), `inject.py` (failure injectors), `blame.py` (found×kept×executed), `noise.py` (comparable observations + noise floor), `claims_audit.py` (claims audit on InferenceBench traces), `exploration.py` (knobs varied).
- `correctness/` — correctness gate. `divergence.py` (teacher-forced KL / top-1 agreement / ref-token
  logprob shift, top-k or full logits), `flips.py` (paired flip test, exact one-sided McNemar), `checks.py` (length
  ratio, consistency), `gate.py` (`evaluate` → three-valued verdict pass|fail|inconclusive; gates = pooled
  accuracy ≥ 99% of reference, length, consistency; a ratio under the line FAILS only if the net loss is
  significant (McNemar p < alpha 0.01), else INCONCLUSIVE, because a non-deterministic engine churns ~1 point at
  the dev tier; unanswered items (shed/errored after retries) → inconclusive; 95% CI on the ratio is a fact;
  divergence and per-task flips are facts; `to_ledger_record` → lab ledger `equiv` record),
  `scoring.py` (MMLU-Pro letters, MATH `\boxed{}`), `tasks.py` (loaders from the Hub into `data/tasks/`; tier `dev`
  ~1.6k = MMLU-Pro 490 stratified, MATH-500, MBPP sanitized 427, needle 180; tier `full` ~18.6k = MMLU-Pro 12,032,
  MATH 5,000 (gold = last `\boxed{}` of the solution), MBPP full 974, needle 600; code scored by running unit tests
  in a limited subprocess; needle = synthetic retrieval at ~4k/12k/24k tokens), `client.py` (`Api(name, chat_kwargs)`: `generate`,
  `generate_stream`, `score_tokens` with adapters vllm (`prompt_logprobs`), sglang (native `/generate`
  `input_token_logprobs`), none (raises `NoLogprobs`; divergence reported as not run)), `run.py` (`reference` once
  per model/hardware/tier, `candidate` per change; 429/503 retried with backoff, items still failing are
  "unanswered" and never scored as wrong; scoring runs at a quarter of the generation concurrency; `verdict_file`). CLI: `python -m correctness reference [--tier dev|full]|candidate|verdict`
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
  `python -m regimes run REGIME[,..]|all --url URL [--corpus DIR | --workload FILE] [--split] [--tier] [--tokenizer] [--out]`
  (`--workload`: a `workloads/` file is one pool for every regime; bursty is refused, no arrival times),
  `python -m regimes cold-start --url URL --cmd "..."`. Closed-loop runs can't detect a client bottleneck by lag.
- `workloads/` — frontier benchmark request data, researcher-only (never reaches the agent: nothing in `lab/` reads
  it), for manual runs: overfitting checks on data the agent never saw, the corpus's missing shapes, engine vs
  vLLM/SGLang on identical requests. One module per benchmark, docstring = source, sizes, output limit, MLPerf
  latency limits: `mlperf_llama3_1_8b` (CNN/DM 13,368), `mlperf_llama2_70b` (OpenOrca 24,576), `mlperf_mixtral_8x7b`
  (15k; GSM8K as 6-turn few-shot), `mlperf_deepseek_r1` (4,388 reasoning), `mlperf_gpt_oss_120b` (perf 6,396 + acc
  4,395), `sharegpt` (58,659 conversations). `python -m workloads fetch all|NAME [--tokenizer]` →
  `data/workloads/<name>.jsonl` + `manifest.json` (source URL, md5, sha256, limits, MLPerf commit). Fetched on demand,
  no token: MLPerf files from the public `inference.mlcommons-storage.org` bucket (`metadata/<name>.uri` + `.md5`),
  ShareGPT from the Hub. Not re-hosted (GPQA asks not to be; ShareGPT's license is unclear). Chat templates are
  stripped. Llama-3.1-405B (404 on the bucket) and edge-agentic (tool-call transcripts) are not imported.
- `tests/test_*.py` — all synthetic, no GPU, each runnable with `python3 tests/<file>.py`, named after the module they cover; `test_correctness_run.py` is the full reference→candidate→verdict flow on fake servers; `test_kernels.py` also runs on MPS; `test_inject.py` injects failures into real traces and skips without data; `test_regimes.py` drives `tests/fake_engine.py` (fixed decode slots, run out of process) through every regime (~2.5 min).
- `data/` is gitignored and local only: `data/inferencebench/` (public traces, `hf download aisa-group/InferenceBench-Trajectories --repo-type dataset --local-dir data/inferencebench`), `data/derived/` (cached Haiku/Sonnet outputs from the research phase — the only copy), `data/laya/`, `data/tasks/` (correctness task sets), `data/equiv*/` and `data/h200-run-*/` (GPU run outputs and reports), `data/archive/research-loop/runs/` (the removed research loop's run panels and temp worktrees, Sept 2026), `data/notes/` (the user's notes). Live lab state is `lab/ledger/` and `lab/runs/`, also gitignored. Findings with evidence go in `knowledge/` (committed), never in `data/`.

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

## Profiler tools on the H200 (Nebius, 2026-10-08)
Unjailed, all four work on the real engine (16 requests × 64 tokens): `profile` ~7 min (first call: weights loading from disk), `trace` 44 s, `kernel` 90-99 s,
`hostprof` 64 s (30 s sampling, 2,981 samples ≈ 100 Hz × 30 s, scheduler thread in the dump); ~$0.87 of GPU time.
`launch_count` honoured; ncu durations are ~1.8× nsys's on a tiny kernel (base clocks), DRAM figures physical.
Fixed from the run: the tools now pass the target's engine env (they had loaded the engine's default model);
nsys traces CUDA-graph nodes (it had missed every decode kernel: 46,720 vs torch's 860,365 launches);
`kernels.json` drops record_function annotations (they had doubled its total). srt always mounts a fresh minimal
`/dev` and has no device passthrough, so nothing in an srt jail sees the GPU (`No CUDA GPUs are available`), and
each srt jail on Linux has its own network namespace, so a jailed server is unreachable. Decided 2026-10-08: on
Linux the jail is a container with the NVIDIA runtime (`lab/safety/container.py`, `lab/RUNTIME.md`); its GPU
check (`vm-setup.sh` prints `jail: gpu ok`) has not run yet. The agent's CLI still runs under srt, which on Ubuntu
24.04 needs `kernel.apparmor_restrict_unprivileged_userns=0` (in `vm-setup.sh`).

## First agent dry run (Nebius H200, 2026-10-10)
`LAB_WORKER=ssh`, laptop controller, `claude-fable-5-1`, one session ($2.53 as the proxy priced it, 24 turns, 465 s;
~1 h 20 min of GPU for the whole run). Validated on the real GPU: the GPU in the network-less jail, metadata blocked,
base and candidate served jailed through the port bridge, `profile`, `trace` (after mounting nsys's install root),
`test` (after caching the tests' gpt2 tokenizer), the remote workbench over `ssh -R`, the proxy's pricing. The agent
fused RMSNorm/RoPE in Triton and added a batch-1 decode graph: seen-split short-tier `single_stream` tpot p99 21.4 ->
13.1 ms; not submitted, correctness not run (equiv needs `data/equiv/ref`, absent in a worktree). A base dev-tier equiv
on this engine took > 46 min for 774 of 1,597 items at concurrency 16 (vLLM: ~10 min): the tool descriptions' "10 min
at tier dev" is wrong for this engine. Workbench processes cannot change clocks or power limits on that platform.

## Next
GPU (needs spend approval): resolve the BF16 puzzle (plain BF16 on the dev tier at concurrency 32 and 128 on one VM,
~20 min, ~$1.50); first real run of `python -m regimes run all` against the engine and against vLLM.
GPU, with approval (and the auto-stop armed): `python -m lab.validate --target targets/inference-server.toml`
(reference, FP8/INT8 good, Int4 bad, noise bands for all 8 regimes; ~3-4 h; serves unjailed, so not blocked), then
a first `python -m lab.session --task ...` dry run with a tiny budget, which waits on the rest of `lab/RUNTIME.md`
(the controller/worker split; the agent's CLI still runs under srt, which has no GPU on Linux). Both engine branches from the 2026-10-06 report (`integ/gpu-session-5`,
`engine/gpu-validated`) were judged by the old gate: re-judge with `python -m correctness verdict` on their saved
results before deciding anything. lab `session.py --run` re-resolves
`base` from HEAD rather than the run's original base (pre-existing; resuming after the engine moved would misaudit);
`validity/tier_agreement.py` needs real short/full runs; `validity/roofline.py` wants generalizing to the H200 + MoE
target. The agent's tool set (e.g. a knowledge tool) is the user's design: ask first. README stays a few one-line
feature bullets, not a dev doc.
