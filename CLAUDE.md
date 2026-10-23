# BlameGraph

The environment for autoresearch loops that optimize LLM inference: the lab (agent loop, tools, ledger, sandbox,
GPU VMs), the workload corpus, the data store, load regimes, the correctness gate, canaries, and the referee
(integrity verdict, session feedback). Design in `ENVIRONMENT.md`. The engine under optimization is a separate repo,
github.com/AdvayMonga/inference-server (engine + stack only since the 2026-10-06 split); the lab finds it via
`LAB_ENGINE_REPO` (default `../inference-server`) and runs it with its own `.venv/bin/python`. See the parent
`../CLAUDE.md` for coding guidelines.

Hard rule from the user: the loop may receive **facts only, never heuristics or advice**. Decided 2026-10-06
("option c"): at the start of every session the agent gets the integrity verdict with its evidence, the raw ledger
(the `ledger` tool, unfiltered) and only the facts it cannot derive from that ledger (`lab/evaltools.harness_facts`:
the limits, tiers, policy, noise bands, corpus version the referee measures against). Nothing derived from the
records (counts, best measured, claims vs evidence) goes to the agent; that is `for_researcher.activity`.
Decided 2026-10-09: no knowledge base of beliefs, hand-written or the agent's (a belief bounded by its evidence is a
measurement row; anything more is a generalization). The findings in `knowledge/` were removed (tag
`archive/knowledge-base`); what stays there is measurement (`knowledge/peaks/`, `knowledge/reference/`). The
agent reads raw data only, through `ledger`, `query` (read-only SQL over every run, `store/index.py`) and `search`
(text search over every run's experiments, `lab/search.py`).
The task (`lab/task.py`, `--task task.toml`: goal, objective regimes, constraints) is stated in full in the brief,
and a submit returns its `score` under that task: the win condition is known, never inferred. The agent's objective
(its system prompt) is generated per session (`session.system_prompt`: template `lab/prompts/objective.md`, a role line
and facts only: the engine and its files by layer (target `[layers]`), how results are scored (each regime's load and
report from `regimes.suite.FACTS`, the limits, tiers, verdicts, correctness policy), the environment, write surface,
tools (ToolSpec descriptions: what each does, returns and costs, never when to use it), budget, sessions). Everything
task-specific is in the per-session brief (goal, optional `context` of specs and facts, objective, constraints, budget
left, referee, baseline, hypotheses). Both are saved per session to `<run>/sessions/<id>/{objective,brief}.md`. The brief is the
task, budget, referee, a BASELINE (the base commit under bench's defaults = the task's objective + constrained regimes, measured once per run, `baseline` ledger
kind; `RunConfig.baseline=False` skips it) and only the run's last `session` record.

## Central goal (decided 2026-10-14)
Make the loop rely less on the raw model and more on per-layer machinery. The engine is looked at as four layers;
each needs data, a simulator, a hypothesis generator, measurement tools and an evaluator, and all of them answer to one
end-to-end judge (`equiv` + `submit` + the held-out guard). The agent chooses which layer to work on; it is given each
layer's share of time × its distance from the ceiling as a fact (`lab/layers.py`, in every profile record and the
brief). Layer 1 first.

## Layout
Top-level packages, run from the repo root (no install step; `pyproject.toml` lists optional extras and holds the
pytest/ruff config). Each folder with commands has its own `python -m`.
- `lab/profiles.py` — launch profiles (decided 2026-10-14): one engine, a config per regime in a TOML file of its own
  tree (`[general]` + a table per regime, on the agent's write surface). bench, submit and the base measurement serve
  the general config once for every regime, then each regime whose table is non-empty in its own launch with
  `<profile_env>=<regime>`; that number is the headline, the general one beside it (`metrics.<regime>.general`).
  Each profile is gated on its own: equiv takes `profile`, submit needs a passing submit-tier equiv per profile it
  measures and records them in `config.profiles`, and the verdict checks the same. The harness reads only the file.
  A profile launch that fails makes its regimes invalid with the reason (`profiles_failed`) and the rest is kept; a
  busy GPU, contamination or the general launch failing still fail the measurement. The profiler tools take
  `profile` too. Noise bands stay the base engine's general config (the latest per target, any engine commit), stated in the harness facts.
- `targets/<name>.toml` + `lab/target.py` — **the target spec** (since 2026-10-06): model + chat kwargs, engine
  (repo, python, launch command with `{port}`, health path, env, logprobs `api` vllm|sglang|none, write surface,
  add-only globs, test and lint commands, optional `[engine.telemetry] env` with `{dir}`, optional `profiles` +
  `profile_env`: launch profiles, `lab/profiles.py`), reference server + its outputs dir, MLPerf latency limits, correctness
  tasks and policy (`submit_tier`: the equiv tier submit requires, `dev` for the example engine until full fits), corpus dir. Selected by `LAB_TARGET` or `--target`; `target.load()` raises `NoTarget` otherwise.
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
  `experiments.py` (decided 2026-10-11: no measured number without the agent's reasoning and no reasoning without
  the number; measured tools require an open `hypothesis` record and an `expect` {metric, value, unit, why, optional
  `mechanism`: the same on other numbers of the result by dotted path, e.g. a profile's layer facts}, the
  record keeps both with the computed gap, and the next measured call is refused until a `reflection` on the last
  one is filed; the brief lists the run's hypotheses with their measurements and reflections; a session that ends
  with its last measurement unreflected is followed by one call (`session.closing`: $0.50, 4 turns, tools `reflect`
  and `ledger` only, prompt = that record's facts), kept under the session record's `closing`; run ids starting
  `researcher-` are a person's, and no agent view (`ledger` tool, `agent.duckdb`) shows their records) +
  `experiment.py` (`python -m lab.experiment start|baseline|hypothesis|call|reflect`: a change made outside the lab,
  e.g. on a branch, recorded like the agent's: a `researcher-` run whose workspace is base + the change's files inside
  the write surface (the rest listed as `left_out` in the start record), measured through the lab's own Toolbox
  under the same hypothesis/expect/reflection rules) +
  `calibrate.py` (every measured number with its context, decided 2026-10-11: `% of ceiling` from `validity/ceiling`
  on measured peaks, `% of reference` from `knowledge/reference/<target>/<regime>.json` written by `lab.validate`
  with `perf_reference` (the correctness reference minus its determinism switches, which are slow by design),
  and for kernels their share of the window; a ceiling or reference is shown only under a matching config
  fingerprint (model, dtypes, max_model_len, tier, split, corpus version, GPU), else the field says why it is
  missing; bench, baseline, profile and kernel records carry a `calibrated` block, the brief shows the baseline's;
  submit's held-out aggregates carry the ceiling from the base's seen-split shape and % of it, never anything computed
  from held-out requests; harness facts carry peaks and floors for the GPU the ledger last measured on),
  `import_runs.py` (a hand-made run dir → `bench`/`equiv` records under a `manual-` run, numbers only),
  `tools.py` (metered tools: test, profile, ledger, query, search, budget, restore, note; every record's `cost` has wall seconds, and
  GPU tools charge them at `LAB_GPU_USD_PER_HOUR` / the target's `[cost] gpu_usd_per_hour` to the run's budget) +
  `proftools.py` (trace = nsys, kernel = ncu, hostprof = py-spy, each wrapping lab.profile's workload in the jail
  with validated args) + `evaltools.py` (bench → `regimes` on
  the seen split, default regimes saturated + the task's, each search started where the run's base measurement ended and
  resolved to the regime's noise band; equiv → `correctness` candidate vs the target's reference, 32 requests in flight (scoring at 4),
  items in one fixed order and stopped at 200, 400, 800, ... once accuracy is clear of the line; submit → held-out short tier vs the
  base commit, one aggregate per regime through the Thresholdout guard; nothing is measured while another process holds
  the GPU (`refused`), and a process other than the engine on the GPU mid-run makes it `contaminated` (never evidence,
  never the cached base; `serve.Served.exclusive()`); all three serve the pristine tree via
  `serve.py`, jailed (on Linux, port bridged out of a network-less container); every serve also keeps its passive data, `artifacts.py`: device
  samples via `gpu.DeviceSampler` (DCGM if `dcgmi`, else nvidia-smi), engine telemetry files from `{dir}`, all
  client rows, the serve log; bench/equiv as ledger blobs under `result.artifacts`, submit only in
  `<run>/heldout-private/`, never in the ledger; `artifacts.joined` joins client rows to engine rows by trace id),
  `workspace.py` (exported engine copy + snapshots), `ledger.py`
  (append-only JSONL, `lab/ledger/` gitignored; `merge` adds another ledger dir's new records and blobs), `budget.py`, `engine.py` (where the engine repo and its python
  are), `safety/` (write surfaces, grader: lint/tests run with the engine's python in the jail; the jail is `container.py` on
  Linux (one container per jailed command, NVIDIA runtime, `--network none`, socat bridge for a served port; tested in
  Docker-in-Docker, GPU untested) and srt elsewhere, `LAB_JAIL` overrides),
  `profile.py` (`--profiler torch|cuda-range|pyspy`) + `bundle.py` + `gpu.py` (profiler harness, staged into the pristine tree under `_harness/` so the
  jail never opens this repo, which holds `.env` and the ledger; the tool samples its seen-split corpus workload out here, `corpus.sample`, and stages it as `_harness/workload.json`; window-only memory peaks and scheduler counters, warmup sized to the window, op/kernel tables), `canary.py` (honest regressions patched into
  the engine's process), `vm.py` + `providers/` + `vm-setup.sh` (one GPU VM on Verda/Nebius/Crusoe; pushes both
  trees, engine to `LAB_VM_DIR`, this repo to `LAB_VM_ENV_DIR`; `run --env` runs here; `run` stops the VM when done
  unless `--keep`; a `--env` run is always brought home first, `collect`: the VM's ledger merged, its run dirs minus
  rebuildable trees, then `store push` when `LAB_STORE` is set; held-out traces are pushed explicitly since git
  lacks them), and every session is capped by `LAB_VM_MAX_MINUTES` via `timeout` + a local watchdog + in-VM
  shutdown, because a stalled session once billed 12 h for 3.5 h of work), `corpus.py` +
  `chat_template.py` (corpus loader, template fingerprint). Tests: `tests/test_lab_*.py` (pytest); the ones
  that drive engine code skip unless the engine and its deps import (run them with the engine's python).
- `lab/rehearsal.py` — `python -m lab.rehearsal setup DIR`: the whole loop on a laptop, no GPU. A practice engine
  (`lab/rehearsal_server.py` copied into a git repo: fixed decode slots and time per token, one-token words in the
  Qwen3 tokenizer), its target spec, a task, a local store, and reference answers to a synthetic multiple-choice set;
  the docstring lists the validate and session commands. Run from its own checkout so the ledger is separate. The
  practice decoder refolds its state over the whole output every step (the reference keeps it): a real, legitimate
  fix exists; with only a sleep to cut, the agent refused ("gaming the benchmark"). First rehearsal 2026-10-17:
  one Sonnet 5.5 session, $0.19, 18 turns, valid, submit improved -85.9% (band 0.9%).
- `lab/validate.py` — `python -m lab.validate --target T`: serves the target's reference and writes its outputs, judges
  each `[validation]` good/bad launch (good must pass, bad must fail: the gate separates on *this* model and GPU), and
  measures each regime's run-to-run band from repeated runs of the engine into `noise` ledger records tagged with
  its commit (which `submit` reads for runs on that base commit only). Exit 0 only if both hold. Needs a GPU for a real target; tested on fakes. `lab/LEDGER.md` is the
  record-format spec: the environment's public interface.
- `lab/RUNTIME.md` — design (2026-10-08): controller (laptop: ledger, referee, secrets) and worker (GPU VM: jails,
  clients, profilers) behind a job interface; trust zones and rules; why srt cannot jail GPU code on Linux.
- `lab/build.py` — the dependency build step: a tree whose change touched the target's `deps` (pyproject: dependency
  tables only, checked by the audit) gets a venv built by the target's `build` command in a no-GPU, `--network none`
  container whose only way out is a CONNECT proxy to the package-index hosts; cached by the files' hash; jobs run
  under `engine.using(build.python_for(tree))`.
- `lab/gpucheck.py` — `python -m lab.gpucheck` on a GPU VM (as root, like the worker): what the jails need from a real
  GPU host, PASS/FAIL each: jail GPU, workbench (GPU, root, apt/pip, metadata blocked), quiet, exclusivity, ncu,
  hostprof, the dependency build.
- `lab/worker.py` — the job interface, built for the measured jobs: `test`, `measure`, `equiv`, `profile` take input
  trees + JSON args, write an out dir, return JSON; the tools call `self.worker`. `LAB_WORKER=local` (default, in
  process) or `ssh` (the `lab.vm` VM): inputs by content hash, re-hashed on the worker before every job; GpuBusy,
  Contaminated, NotReady, ValueError cross intact. Remote is tested through a local shell, and ran over real SSH in the 2026-10-10 dry run and the 2026-10-16
  kernel run. With
  `ssh` the agent works on the VM too: its workspace mirrored by rsync (pulled before every audit), its workbench
  built there (`workbench`), its CLI's stdio over `ssh -R`, the API proxy tunnelled to a unix socket in the workbench.
- `lab/TRACE.md` + `lab/trace_contract.py` — the request-trace contract: per-request engine rows (trace_id = the
  client's `X-Trace-Id`, `arrival_ts` epoch s, `*_s` monotonic durations, TPOT = (E2E − TTFT)/(n − 1)) in a dir of
  `*.sqlite` (`requests` + `meta`) or `*.jsonl` + `<stem>.meta.json`, with schema version and drop counts;
  `python -m lab.trace_contract DIR`. Diagnostics only, never scored. inference-server's `TELEMETRY_DIR` conforms
  natively (schema 1, inference-server#102); the target names its env vars in `[engine.telemetry] env` (`{dir}`); serve collects
  the files after every bench/equiv/submit and the contract check's report is stored beside them. vLLM OTel span mapping is in the spec, not implemented.
- `corpus/` — frozen workload traces (BurstGPT timing, WildChat text), seen/heldout, hashed, `manifest.json`;
  `build_corpus.py` / `fetch_traces.py` rebuild it; any change is a new `corpus_version`. Since 2026-10-09 (version
  `131149…`) the held-out traces and the build seed (`corpus/.seed`, which reproduces them) are gitignored and live in
  the store (`python -m store pull --corpus`); the previous version's held-out split was public in git.
- `lab/search.py` — the `search` tool (2026-10-15): one document per `hypothesis` record across every run, with its
  measurements (expect, actual, gap, GPU, config) and reflections, ranked by reciprocal-rank fusion of BM25 (English
  stopwords dropped) and cosine similarity of local `BAAI/bge-base-en-v1.5` embeddings (fastembed 0.9.0, ONNX on CPU,
  weights in `<ledger>/index/models`, vectors cached by content hash in `<ledger>/index/embeddings/`); researcher runs
  hidden from the agent; without fastembed it ranks by BM25 and says why. `LAB_TEST_REAL_EMBEDDINGS=1` runs the
  real-model test. It finds paraphrases ("tokens generated slowly" -> per-token latency) but not domain synonyms
  (fp8 -> quantization).
- `store/` — the data store (2026-10-09; bucket `bg-data`, credentials in `.env`: `LAB_STORE`, `LAB_STORE_ENDPOINT`,
  `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`). `remote.py` (`LAB_STORE` = `s3://<bucket>` with `LAB_STORE_ENDPOINT` for
  Cloudflare R2, or a directory; write-once keys), `sync.py` (`push`/`pull`: ledger as byte-range segments, blobs,
  submit's held-out passive data, held-out traces + seed, `archive` for
  anything else; zstd, `heldout/` prefix for held-out material), `index.py` (DuckDB tables rebuilt from the ledger and
  blobs: records, regimes, heldout_aggregates, noise, client_rows, engine_rows, device_samples, kernels, ops;
  `all.duckdb` adds heldout_rows for the researcher; the agent's `agent.duckdb` never has them nor researcher runs'
  records, and is queried with
  external access off and configuration locked). Controller-only: VMs never hold store keys (`lab/RUNTIME.md`).
  CLI: `python -m store push|pull [--blobs] [--corpus]|index|sql "..." [--agent]|archive PATH.. [--heldout]`.
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
  per model/hardware/tier, `candidate` per change, `quick` (the candidate scores the dev reference's sequences
  teacher-forced, nothing generated: divergence facts, no verdict; the equiv tool's tier `quick`); 429/503 retried with backoff, items still failing are
  "unanswered" and never scored as wrong; scoring runs at a quarter of the generation concurrency; `verdict_file`). CLI: `python -m correctness reference [--tier dev|full]|candidate|verdict`
  (`verdict` re-judges saved results under the policy, no server).
- `canaries/cheat_proxy.py` — cheat proxy in front of any OpenAI-compatible server (`truncate`, `early_eos`,
  `fake_first`, `drop`, `cache` (streamed and not), `inflate_usage`); `python -m canaries --upstream URL --cheat NAME`.
- `validity/` — `ceiling.py` (physical bounds from the model's HF config and the GPU's peaks: active/total params,
  bytes streamed per step with MoE experts-touched, decode and prefill floors, the largest batch under a TPOT limit,
  per-objective regime ceilings: TPOT = decode step floor, TTFT = prefill floor at the p99 prompt, goodput =
  1 / (prefill compute + output tokens at the decode ceiling, batch capped by TPOT and by KV fitting in HBM); no
  bound for speedups, cold start or sessions/s; assumptions (one token per step, uniform routing, declared dtypes)
  are in each `basis`, so a value above 100% means one of them does not hold; `pct_of`, `ceiling_gain` = Amdahl) + `peaks.py` (`python -m validity peaks`: HBM
  bandwidth and bf16 GEMM measured once per GPU+driver into `knowledge/peaks/<gpu>.json`, run by `vm-setup.sh`; spec
  sheet as the labelled fallback), `repeat_audit.py`: repeated-request audit (MLPerf TEST04-style), judges decode time per token, not TTFT, so prefix caching passes and replayed answers fail; `python -m validity repeat --url URL`. `tier_agreement.py`: `tier_agreement(changes)`: Spearman, Kendall tau-b, bootstrap CI, out-of-band disagreements; short tier usable if n ≥ 8, rho ≥ 0.8, CI low ≥ 0.5. `holdout.py`: `HoldoutGuard` (Thresholdout query budget; state n is a write-once object in the store, a query writes n + 1 and retries on a lost race, so one budget per corpus version is shared by every run and machine; seed drawn once with state 0, kept only in the store) and `SealedSplit` (hash manifest, one logged unseal per corpus version). Keep sigma small relative to the threshold.
- `kernels/` — kernel equivalence: `check_kernel` (seen + held-out shapes, per-dtype tolerances, edge cases incl. non-contiguous and NaN/Inf, inputs unchanged, no aliasing, memoization, determinism) and `time_kernel` / `check_memoization` (sync, rotated buffers, L2 flush on CUDA). Validated on CPU, MPS and CUDA (H200, 2026-10-04): all tests pass; the timer resolves ~20 µs kernels at ~9% IQR, 0.2 ms at 1.5%. `make_inputs` gets a seeded CPU generator; large tensors are made on the device by a generator seeded from it
(`kernels.tasks.randn`: generating them on the CPU took seconds per check case at real shapes).
  `kernels/tasks/` (a kernel task: the engine entry point it replaces = the baseline, a reference, an input builder,
  how to read a shape from a captured call, the held-out axes, the layouts its callers pass, bytes and FLOPs per call
  for the roofline; `toy_rmsnorm` for CPU, `paged_decode` = the example engine's `paged_decode_attention` (block tables as wide as its graphed decode passes them, from the target's CONTEXT_WINDOW: `Task.options`, `tasks.from_config`), `moe_gemm` = its MoE grouped
  GEMM `moe_kernel.moe_gemm`, both projections, real routing from captures, bf16 weights; `moe_gemm_fp8` = the same with fp8 e4m3 weights and per-channel
  scales, as the engine runs it by default since inference-server #122; `moe_gemm_scatter_fp8` = its down projection
  `moe_kernel.moe_gemm_scatter` (weighted rows added into an fp32 output); each reads only its own calls by weight dtype; `rmsnorm`, `add_rmsnorm`, `rmsnorm_rope`,
  `silu_mul`, `topk_softmax` = its fused kernels in `models/fused.py`, references from its own eager paths, the start
  that module with its kernels module imported directly (`Task.start_edits`: the screen refuses its importlib/getattr); prefill
  attention has no task: `sdpa_prefill` takes the backend's paged context, not tensors),
  `kernels/shapes.py` (shapes and call counts from captures; held-out = whole regions above the 80th percentile on the
  task's axes, never random samples; `ints_from_captures` keeps each shape's captured int args, e.g. real sequence
  lengths, which `Task.with_ints` uses for checking and timing in place of synthetic ones), `kernels/evolve/` (the layer-1 generator, a replication of OpenEvolve 0.4.0 =
  open AlphaEvolve: `python -m kernels.evolve run|gate|split`; the starting program is the engine's kernel module;
  each run appends one `kernels` ledger record (every candidate's seen-shape metrics and lineage, OpenEvolve's
  evolution trace and the best program as blobs; `lab/LEDGER.md`);
  prompts carry facts only (`kernels/evolve/prompts/` overrides OpenEvolve's advice; `run.system_message` states how a
  program is judged, from the evaluator's own constants);
  a two-stage cascade, each stage a fresh jailed process (`kernels.evolve.jailed`: the jail gets the candidate, its
  config and a copied harness, never this repo; `core.judge`: stage 1 = two shapes, one seed; stage 2 = the full
  check on seen shapes + paired timing against the baseline (`kernels.time_pair`: same inputs, random order per
  pair; a task whose callers replay it in CUDA graphs, `Task.graphs`, e.g. `paged_decode`, is captured and timed as a
  graph replay behind a GPU spin so its launch is not timed, and a kernel that cannot be captured scores 0; the
  timed calls' outputs are checked against the reference too, so a kernel that does less only when timed scores 0; every
  plain check case's and timed call's outputs are verified again outside the jail by a process that never loads the
  candidate, `kernels/evolve/verify.py`, gate `verified_outside`, which requires every record the config implies:
  each seen and held-out shape x dtype x seed, each value scale, each timed buffer; the result is the one stdout line
  carrying a per-judgment key that core reads and removes before the candidate loads; between judgments only torch's
  build files survive in a program's dir, and the gate has its own), score = call-weighted geometric-mean speedup, a shape counting 1.0 when the middle half of its paired ratios
  includes 1; at the end the top candidates are re-judged 3x and the highest median is promoted, `run --retime`); a program may build CUDA C++ at import with
  `torch.utils.cpp_extension.load_inline` (nvcc from the host's toolkit, mounted read-only into the jail; the build is
  cached per program source and toolkit across its stages, in one fixed dir per program, `jailed._eval_dir`; C++ only as
  literals, `core._cpp_built_at_run_time`); a candidate is
  refused if its source, its C++ included, reaches for an exploit route or writes into anything it imported, under any alias
  (`core.screen`), and scores 0 if any function the checker or timer uses is patched, checked after loading and
  after every call the checker makes (`core._watched`; the timer binds its own at import); the exploit canaries in
  `tests/test_kernels_evolve.py` must all score 0; selection never sees held-out shapes, `gate` judges them (correctness, and their speedup as a fact); `validate_gpu.sh` = the GPU validation (shapes captured under real corpus traffic at several concurrencies, `CAPTURE_SPECS`; never synthetic prompts, whose batches have uniform lengths); tolerances from the target's
  `[kernels]`). `kernels/search/` (extra `kernel-search`; our own layer-1 generator, replacing OpenEvolve, which stays as the A/B baseline: `python -m
  kernels.search`; a tree of refinement chains, parents chosen by MCTS (UCT with C = 0.6 on percentile rewards and max backup,
  KernelPro's progressive widening and virtual-child rule), a node closed after 3 judged failed children or 3 edits
  that never reached the GPU, the model shown only the
  parent program, its measurements and its lineage, SEARCH/REPLACE answers, duplicates refused before any GPU time,
  with a judge (default) each step is two calls,
  the judge (Opus 5.5) naming as many changes as the beam is wide (`--beam`, 4) as JSON, every one written and judged
  (each and its fate in `hypotheses.jsonl`), from the facts (and, its choice when, Anthropic's web search
  and read-only SQL over the agent's view of the ledger index, `--judge-tools`; the harness pastes no research), with
  up to 2 better programs from other branches as diffs (CodeEvolve's inspirations), including CudaForge's 24 ncu counters
  (`kernels/evolve/ncu.py`, stage 3 of the jailed evaluator; none collected yet: every profile so far failed with
  ERR_NVGPUCTRPERM, and a run stops asking after the first refusal), and the coder writing it; a failed child is repaired in
  correction mode, chained up to 3 deep; launch parameters a program declares as `CONFIG` + `TUNABLES` are chosen afterwards by a grid
  search on each of the top 4 re-timed programs (8 points each, no ncu), written back as constants, the best point
  promoted only if its re-timed median beats the winner's
  (`kernels/search/tune.py`), and when different points win in different size regions a dispatcher picks CONFIG per
  call from the call's size (`Task.size_expr`, tensor shapes only, so CUDA graphs bake it per captured size), promoted
  on the same rule; the search stops once the best program is at `--stop-pct` (80) of its call-weighted
  roofline bound, a figure the prompt states, or after `--patience` (16) judged programs with no new best (MANTIS);
  each timed shape carries its bound type (memory or compute) and the achieved % of peak bandwidth and of peak FLOPs,
  and ncu runs on the smallest and the largest seen shape; before promotion a model (`--detector-model`, Opus 5.5) reads each would-be winner for gaming and a flagged one is
  passed over, recorded with its evidence (`kernels/search/detect.py`, MANTIS; never changes a score);
  every call priced (OpenEvolve's too, via a usage log);
  `python -m kernels.search.ab --arms ...` runs OpenEvolve and kernels.search (and variants differing in one flag:
  `search-no-tools`, `search-c1.41`) on one task with the same shapes, start, evaluator,
  the same post-search promotion for every arm (`ab.SAME_PROMOTION`: top 3 re-timed, no tuning, no detector), single model
  and judged-program budget, gates each winner on held-out shapes, and writes `ab.json`; `kernels/METHOD.md` maps
  every component to its published source and number, or says why ours; same evaluator, run dir, re-timed promotion and `kernels` record as `kernels.evolve run`). `kernels/evolve/ship.py` (`python -m kernels.evolve ship`: a program with a passing held-out gate becomes a
  commit on a new engine branch, built with git plumbing so the engine's tree is untouched; `--budget` starts the
  `lab.experiment` researcher run on it, measured end to end by `bench`/`equiv`/`submit`). `kernels/share.py` (`python -m kernels.evolve share --profile BUNDLE [--profile ...] [--gate TASK=GATE.json]`: tasks
  ranked by share x (1 - % of roofline) across profiles, each task's share of a profiled engine
  window from the device kernels it launches, `Task.kernel_names`, the speedup bound if they took no time, and the
  kernels no task covers, by share: the facts for choosing the next task). `lab/capture.py` + `lab.profile --capture MODULE:FUNCTION` (the `profile` tool's `capture`): every
  call's argument shapes and small int tensors, the first call per shape kept whole as real inputs.
- `lab/layers.py` — each layer's share of a profiled window and its distance from a ceiling, as facts (2026-10-18):
  kernels = CUDA kernel time / window; model execution = prefill + decode phase wall time beyond kernel time;
  runtime = admit + evict phases, with the GPU idle share, batch fill and admissions blocked; serving = window time
  outside scheduler steps. Distance: the mean decode step against `validity.ceiling`'s step floor at the window's
  batch and context; kernels' % of roofline from gate results when given; layers 3 and 4 say they have no single
  ceiling. Every `profile` record carries it under `calibrated.layers`; the brief's harness facts show the latest
  (`layers`). `python -m lab.layers BUNDLE`. The timeline covers the whole run, so only events inside the window
  count; nested phases (prefill_mixed inside prefill_chunk) are counted once.
- `sim/` — the scheduler simulator (layer 3, 2026-10-18): `sim/scheduler.py` drives the engine's real
  `ContinuousBatchScheduler` one `step()` at a time (inference-server #120) on a fake clock (`Clock`, installed as the
  scheduler module's `time`) with `SimBackend`, which runs no model and charges a `CostModel` (prefill and decode
  step times, answer lengths; defaults are unfitted stand-ins, phase 2 fits them from a GPU run's step telemetry) and
  reserves KV blocks worst-case like the custom backend. Real: the scheduler, its policy, admission, KV arithmetic,
  request timestamps, TTFT/TPOT as the regimes define them; a 10-minute class simulates in under a second.
  `simulate(records, Settings, CostModel)` → regimes-shaped rows for `runner.summarize`; `Settings.from_env` takes the
  engine's scheduler env vars, so a profile applies as is. `python -m sim run CLASS [--synthetic] [--env JSON]
  [--cost FILE]`, with the engine's python. Not yet validated against a real bench (the bar: ranks changes like bench
  does, Spearman ≥ 0.8); not an agent tool yet (the user's design).
- `regimes/synth.py` — synthetic traffic (2026-10-18): generated fresh per run from the corpus's seen split and then
  gone, never written, the seed regenerates it (`Ctx.synthetic`, `regimes run --synthetic`). Whole real sessions in
  shuffled passes (structure, turns, think time, max_tokens, regime), the real gap sequence repeated with a seeded
  jitter (bursts stay where they were), text cut from real messages of the same role at sentence ends and
  recombined to length: 0 of ~2,000 generated prompts per class equal a real one; KS on prompt tokens ≤ 0.02 and
  per-10 s arrival counts within a few % of the real class (`python -m regimes.synth compare`). Not yet compared
  against the real corpus on a GPU; until then bench/submit stay on the public corpus.
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
  `python -m regimes run REGIME[,..]|all --url URL [--corpus DIR | --workload FILE] [--synthetic] [--split] [--tier] [--tokenizer] [--out]`
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
- `tests/test_*.py` — all synthetic, no GPU, each runnable with `python3 tests/<file>.py`, named after the module they cover; `test_correctness_run.py` is the full reference→candidate→verdict flow on fake servers; `test_kernels.py` also runs on MPS; `test_inject.py` injects failures into real traces and skips without data; `test_regimes.py` drives `tests/fake_engine.py` (fixed decode slots, run out of process) through every regime (~4 min in CI, where PRs run it only if they touch `regimes/`, `canaries/`, `corpus/`, `lab/corpus.py`, `lab/target.py`, its fake engine or the workflows; main pushes and the weekly run always do).
- `data/` is gitignored and local only: `data/inferencebench/` (public traces, `hf download aisa-group/InferenceBench-Trajectories --repo-type dataset --local-dir data/inferencebench`), `data/derived/` (cached Haiku/Sonnet outputs from the research phase — the only copy), `data/tasks/` (correctness task sets), `data/equiv*/` and `data/h200-run-*/` (GPU run outputs and reports), `data/archive/research-loop/runs/` (the removed research loop's run panels and temp worktrees, Sept 2026), `data/notes/` (the user's notes). Live lab state is `lab/ledger/` and `lab/runs/`, also gitignored, kept in the store by `python -m store push`.

## History
`research-v1` tag = research snapshot (InferenceBench harness code removed from history 2026-10-04): Haiku extractor, Sonnet judge,
dashboard, Laya experiments, report/IRT/benchmark-audit scripts, loop tool (ledger/landscape/context). Restore from
there rather than re-deriving. Trace quirks (codex truncation, sandbox exit-126, no timestamps) are handled in
`traces.py`/`experiment.py`.

## The engine repo (inference-server, after the 2026-10-06 split)
Holds only the engine and its stack: `src/` (engine, control plane), engine tests, `monitoring/`, and engine tools
(`scripts/bench/tune_triton_launch.py`, `scripts/gpu_tests/checks.py`, `scripts/tools/smoke_custom.py`). The
combined pre-split tree is tag `archive/pre-split` there. Its shim serves `prompt_logprobs` (contract in
`correctness/client.py`). Everything that runs, measures or judges it lives here; the user builds the agent's tools
themselves, so don't add agent tools without asking. Corpus classes for the regimes the
corpus lacks (short/long output, overload, shared prefix) belong in `corpus/build_corpus.py`; the regimes synthesize
them from existing records meanwhile.

## Decisions and runs
- `DECISIONS.md` — append-only, one dated entry per decision (what, why, what it replaced), newest first. The eval
  conventions (2026-10-02) are there; change any decision only with the user.
- GPU runs go into the ledger, not a file (decided 2026-10-18). A run made through the lab's tools records itself;
  a run made by hand (`python -m regimes run --out DIR`, `python -m correctness candidate --out DIR/equiv.json`) is
  imported with `python -m lab.import_runs --run manual-<date> --commit SHA [--env K=V ...] DIR...` (`bench` / `equiv`
  records, numbers only, visible to the agent), then `python -m store push`. `RUNS.md` is the frozen history of the
  runs before that (2026-10-04 to 2026-10-18); nothing is added to it.

There is no status file: what works now is the code, the merged PRs and the ledger (`python -m store sql`).

@DECISIONS.md

The agent's tool set is the user's design: ask first. README stays a few one-line feature bullets, not a dev doc.
