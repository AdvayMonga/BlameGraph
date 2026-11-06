# BlameGraph

The environment for autoresearch loops that optimize LLM inference: the lab (agent loop, tools, ledger, sandbox,
GPU VMs), the workload corpus, the data store, load regimes, the correctness gate, canaries, and the referee
(integrity verdict, session feedback). The engine under optimization is a separate repo,
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
`archive/knowledge-base`); what stays there is measurement (`data/calibration/peaks/`, `data/calibration/reference/`). The
agent reads raw data only, through `ledger`, `query` (read-only SQL over every run, `data/knowledge/index.py`) and `search`
(text search over every run's experiments, `data/knowledge/search.py`).
The task (`lab/task.py`, `--task task.toml`: goal, objective regimes, constraints) is stated in full in the brief,
and a submit returns its `score` under that task: the win condition is known, never inferred. The agent's objective
(its system prompt) is generated per session (`session.system_prompt`: template `lab/prompts/objective.md`, a role line
and facts only: the engine and its files by layer (target `[layers]`), how results are scored (each regime's load and
report from `measure.loadgen.suite.FACTS`, the limits, tiers, verdicts, correctness policy), the environment, write surface,
tools (ToolSpec descriptions: what each does, returns and costs, never when to use it), budget, sessions). Everything
task-specific is in the per-session brief (goal, optional `context` of specs and facts, objective, constraints, budget
left, referee, baseline, hypotheses). Both are saved per session to `<run>/sessions/<id>/{objective,brief}.md`. The brief is the
task, budget, referee, a BASELINE (the base commit under bench's defaults = the task's objective + constrained regimes, measured once per run, `baseline` ledger
kind; `RunConfig.baseline=False` skips it) and only the run's last `session` record.

## Running the loop
One command: `lab <command> [args]` (`bin/lab`, which runs `python -m lab` with this repo's `.venv` when it has one,
else `python3`; put `bin/` on PATH or link `bin/lab` into it). `lab` lists every command, `lab <command> -h` its
arguments; the dispatch table is `lab/cli.py`. The agent's tools are the same commands with the same names and
arguments, run by a person on a researcher run: `lab bench --run RUN --hypothesis-id H --expect JSON`. On a GPU VM,
`lab vm run` puts the environment's `bin/` on PATH ahead of the engine's venv, so `lab` runs with the engine's python.
1. Once per target and GPU: `lab store pull --corpus` (held-out traces and seed); the reference's answers (`lab
   correctness reference --url URL --out local/equiv/ref`); `lab validate --target T` (good launches pass, bad ones
   fail, the noise bands, the performance reference). `lab vm setup` measures the GPU's peaks (`lab peaks`).
2. A GPU: `lab vm start`, then `lab vm setup` on a new VM; `LAB_WORKER=ssh` runs the measured jobs there.
3. A run: `lab session --task task.toml --budget USD` (or `lab vm run --env -- lab session ...`); `lab dashboard`
   shows it live. On a laptop with no GPU: `lab rehearsal setup DIR` and the commands it prints.
4. After it: `lab feedback local/ledger --run RUN` (the verdict and the researcher's facts), `lab store push`, and
   `lab vm stop` (a VM is never left running).
5. A person's change, under the agent's rules: `lab start --base REF --change REF --budget USD` prints the run id;
   `lab baseline --run RUN`; `lab hypothesis --run RUN --hypothesis ... --rationale ... --layer L`; a measured tool
   (`lab bench|equiv|submit|profile|trace|kernel|hostprof|kernel-search --run RUN --hypothesis-id H --expect JSON`);
   `lab reflect --run RUN --record ID --reflection ...` before the next.
The rules every run keeps: facts only to the agent (above); no measured number without an open hypothesis and an
`expect`, and none after it until a reflection on the last; a merge is a seen-split `bench` and a passing `equiv`;
`submit` reads the held-out split through the guard's query budget; nothing is measured while another process holds
the GPU. Runs go in the ledger (`lab import-runs` for one made by hand), then `lab store push`.

## Central goal (decided 2026-10-14)
Make the loop rely less on the raw model and more on per-layer machinery. The engine is looked at as four layers;
each needs data, a simulator, a hypothesis generator, measurement tools and an evaluator, and all of them answer to one
end-to-end judge (`equiv` + `submit` + the held-out guard). The agent chooses which layer to work on; it is given each
layer's share of time × its distance from the ceiling as a fact (`measure/context/layers.py`, in every profile record and the
brief). Layer 1 first.

## Layout
Top-level packages, run from the repo root (no install step; `pyproject.toml` lists optional extras and holds the
pytest/ruff config). Every command is `lab <command>` (`bin/lab`, below).
Six top-level folders (since 2026-11-03): `lab/` (the loop), `targets/`, `measure/` (what touches hardware and
load), `evals/` (what judges), `generate/` (what proposes changes; the kernel layer), `data/` (corpus,
workloads, calibration, knowledge); `tests/` stays flat; `local/` (gitignored) holds local data, the ledger and runs.

### `lab/` — the loop: session, agent, tools, experiments, rehearsal
- `lab/` — moved from inference-server 2026-10-06. `session.py` (the loop over a dollar budget),
  `agent.py` (the one place a model is called; the agent's jailed shell gets the engine's python on PATH;
  `LAB_AGENT_PROVIDER` claude = Agent SDK CLI, openai = `agent_openai.py`: any OpenAI-compatible API with its own
  key, https public hosts only, `LAB_MODEL_PRICE` required, loop outside the jail, Bash jailed with no network; on
  Linux the Claude CLI runs in a per-session workbench container: GPU, internet, root inside, engine venv read-only;
  its model calls go through `apiproxy.py`, which injects the key, prices usage and enforces the session cap; the
  workbench is paused by `container.quiet` around measure/equiv/profile jobs, GPU holders killed and recorded),
  `experiments.py` (decided 2026-10-11: no measured number without the agent's reasoning and no reasoning without
  the number; measured tools require an open `hypothesis` record and an `expect` {metric, value, unit, why, optional
  `mechanism`: the same on other numbers of the result by dotted path, e.g. a profile's layer facts}, the
  record keeps both with the computed gap; a hypothesis names its layer, several may be open at once and a call may
  measure several as a `bundle` with its `independence` stated; a hypothesis is measured again only after a
  `reflection` on its last measurement, and a new hypothesis only once every measurement has one (batches); the brief lists the run's hypotheses with their measurements and reflections; a session that ends
  with measurements unreflected is followed by one call (`session.closing`: $0.50, 4 turns, tools `reflect`
  and `ledger` only, prompt = that record's facts), kept under the session record's `closing`; run ids starting
  `researcher-` are a person's, and no agent view (`ledger` tool, `agent.duckdb`) shows their records) +
  `cli.py` (`lab <command>`: every command's module, and the agent's tools as commands on a researcher run, each
  tool's JSON schema as flags) + `experiment.py` (`lab start`, `lab baseline`, then the tools: a change made outside the lab,
  e.g. on a branch, recorded like the agent's: a `researcher-` run whose workspace is base + the change's files inside
  the write surface (the rest listed as `left_out` in the start record), measured through the lab's own Toolbox
  under the same hypothesis/expect/reflection rules) +
  `calibrate.py` (every measured number with its context, decided 2026-10-11: `% of ceiling` from `measure/context/ceiling`
  on measured peaks, `% of reference` from `data/calibration/reference/<target>/<regime>.json` written by `evals.agent.validate`
  with `perf_reference` (the correctness reference minus its determinism switches, which are slow by design),
  and for kernels their share of the window; a ceiling or reference is shown only under a matching config
  fingerprint (model, dtypes, max_model_len, tier, split, corpus version, GPU), else the field says why it is
  missing; bench, baseline and every profiler's records (profile, kernel, trace, hostprof) carry a `calibrated` block, the brief shows the baseline's;
  submit's held-out aggregates carry the ceiling from the base's seen-split shape and % of it, never anything computed
  from held-out requests; harness facts carry peaks and floors for the GPU the ledger last measured on),
  `import_runs.py` (a hand-made run dir → `bench`/`equiv` records under a `manual-` run, numbers only),
  `tools.py` (metered tools: test, profile, ledger, query, search, budget, restore, note, `kernel_search` (decided
  2026-10-22; a blocking call: `generate/kernels/tasks/pipeline.py` captures the task's calls under each scored regime's traffic
  (`REGIME_CAPTURE`; each regime weighted equally, `shapes.from_captures(equal=True)`; reused while the engine outside
  the task's module, the traffic and the spec are unchanged, `tree_key`), runs
  `generate.kernels.search` on the controller with its evaluator on the worker over ssh, gates the winner on held-out shapes,
  and saves a patch of the engine module in the workspace for the agent to apply; kernels-layer hypotheses only;
  `budget` (USD) split between model calls and GPU time); every record's `cost` has wall seconds, and
  GPU tools charge them at `LAB_GPU_USD_PER_HOUR` / the target's `[cost] gpu_usd_per_hour` to the run's budget) +
  `proftools.py` (profile = torch.profiler, trace = nsys, kernel = ncu, hostprof = py-spy, all one path: measure.profile.profile's
  workload in the jail with validated args, the GPU watched like a served engine (`serve.Watch`: refused when busy,
  contaminated when shared), one record shape) + `evaltools.py` (bench → `regimes` on
  the seen split at tier `min` (`BENCH_TIER`), default regimes saturated + the task's, each search started where the run's base measurement ended and
  resolved to the regime's noise band, then, only with `profile: true`, the same tree profiled once in its own launch (`profiled`: the layer facts
  a mechanism prediction is measured against; the baseline always); equiv → `correctness` candidate vs the target's reference, 32 requests in flight,
  MLPerf's accuracy and length gates only, items in one fixed order and stopped at 200, 400, 800, ... once accuracy is clear of the line; submit → held-out `min` tier vs the
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
  unless `--keep`; Nebius `LAB_VM_SPOT=1` = a preemptible VM at the spot price, stopped on preemption; a `--env` run is always brought home first, `collect`: the VM's ledger merged, its run dirs minus
  rebuildable trees, then `store push` when `LAB_STORE` is set; held-out traces are pushed explicitly since git
  lacks them), and every session is capped by `LAB_VM_MAX_MINUTES` via `timeout` + a local watchdog + in-VM
  shutdown, because a stalled session once billed 12 h for 3.5 h of work), `corpus.py` +
  `chat_template.py` (corpus loader, template fingerprint). Tests: `tests/test_lab_*.py` (pytest); the ones
  that drive engine code skip unless the engine and its deps import (run them with the engine's python).
- `lab/rehearsal.py` — `lab rehearsal setup DIR`: the whole loop on a laptop, no GPU. A practice engine
  (`lab/rehearsal_server.py` copied into a git repo: fixed decode slots and time per token, one-token words in the
  Qwen3 tokenizer), its target spec, a task, a local store, and reference answers to a synthetic multiple-choice set;
  the docstring lists the validate and session commands. Run from its own checkout so the ledger is separate. The
  practice decoder refolds its state over the whole output every step (the reference keeps it): a real, legitimate
  fix exists; with only a sleep to cut, the agent refused ("gaming the benchmark"). First rehearsal 2026-10-17:
  one Sonnet 5.5 session, $0.19, 18 turns, valid, submit improved -85.9% (band 0.9%).

### `targets/` — the target spec
- `targets/<name>.toml` + `targets/target.py` — **the target spec** (since 2026-10-06): model + chat kwargs, engine
  (repo, python, launch command with `{port}`, health path, env, logprobs `api` vllm|sglang|none, write surface,
  add-only globs, test and lint commands, optional `[engine.telemetry] env` with `{dir}`, optional `profiles` +
  `profile_env`: launch profiles, `measure/runner/profiles.py`), reference server + its outputs dir, MLPerf latency limits, correctness
  tasks and policy (`submit_tier`: the equiv tier submit requires, `dev` for the example engine until full fits), corpus dir. Selected by `LAB_TARGET` or `--target`; `target.load()` raises `NoTarget` otherwise.
  Nothing defaults to the user's engine or to Qwen3; `targets/inference-server.toml` is the committed example and
  what `tests/conftest.py` points at. Surfaces, grader commands, `targets/engine.py`, and the `correctness`,
  `regimes`, `validity` CLIs all read it.

### `measure/` — the physical side: runner, cloud, sandbox, profile, loadgen, sim, context
- `measure/runner/profiles.py` — launch profiles (decided 2026-10-14): one engine, a config per regime in a TOML file of its own
  tree (`[general]` + a table per regime, on the agent's write surface). bench, submit and the base measurement serve
  the general config once for every regime, then each regime whose table is non-empty in its own launch with
  `<profile_env>=<regime>`; that number is the headline, the general one beside it (`metrics.<regime>.general`).
  Each profile is gated on its own: equiv takes `profile`, submit needs a passing submit-tier equiv per profile it
  measures and records them in `config.profiles`, and the verdict checks the same. The harness reads only the file.
  A profile launch that fails makes its regimes invalid with the reason (`profiles_failed`) and the rest is kept; a
  busy GPU, contamination or the general launch failing still fail the measurement. The profiler tools take
  `profile` too. Noise bands stay the base engine's general config (the latest per target, any engine commit, same corpus version), stated in the harness facts.
- `measure/sandbox/build.py` — the dependency build step: a tree whose change touched the target's `deps` (pyproject: dependency
  tables only, checked by the audit) gets a venv built by the target's `build` command in a no-GPU, `--network none`
  container whose only way out is a CONNECT proxy to the package-index hosts; cached by the files' hash; jobs run
  under `engine.using(build.python_for(tree))`.
- `measure/cloud/gpucheck.py` — `lab gpucheck` on a GPU VM (as root, like the worker): what the jails need from a real
  GPU host, PASS/FAIL each: jail GPU, workbench (GPU, root, apt/pip, metadata blocked), quiet, exclusivity, ncu,
  compute-sanitizer, hostprof, the dependency build.
- `measure/runner/worker.py` — the job interface, built for the measured jobs: `test`, `measure`, `equiv`, `profile` take input
  trees + JSON args, write an out dir, return JSON; the tools call `self.worker`. `LAB_WORKER=local` (default, in
  process) or `ssh` (the `measure.cloud.vm` VM): inputs by content hash, re-hashed on the worker before every job; GpuBusy,
  Contaminated, NotReady, ValueError cross intact. Remote is tested through a local shell, and ran over real SSH in the 2026-10-10 dry run and the 2026-10-16
  kernel run. With
  `ssh` the agent works on the VM too: its workspace mirrored by rsync (pulled before every audit), its workbench
  built there (`workbench`), its CLI's stdio over `ssh -R`, the API proxy tunnelled to a unix socket in the workbench.
- `measure/runner/trace_contract.py` — the request-trace contract: per-request engine rows (trace_id = the
  client's `X-Trace-Id`, `arrival_ts` epoch s, `*_s` monotonic durations, TPOT = (E2E − TTFT)/(n − 1)) in a dir of
  `*.sqlite` (`requests` + `meta`) or `*.jsonl` + `<stem>.meta.json`, with schema version and drop counts;
  `lab trace-contract DIR`. Diagnostics only, never scored. inference-server's `TELEMETRY_DIR` conforms
  natively (schema 1, inference-server#102); the target names its env vars in `[engine.telemetry] env` (`{dir}`); serve collects
  the files after every bench/equiv/submit and the contract check's report is stored beside them.
- `measure/context/{ceiling,peaks}.py`, `evals/agent/{repeat_audit,holdout}.py`, `evals/researcher/tier_agreement.py` (were `validity/`) — `ceiling.py` (physical bounds from the model's HF config and the GPU's peaks: active/total params,
  bytes streamed per step with MoE experts-touched, decode and prefill floors, the largest batch under a TPOT limit,
  per-objective regime ceilings: TPOT = decode step floor, TTFT = prefill floor at the p99 prompt, goodput =
  1 / (prefill compute + output tokens at the decode ceiling, batch capped by TPOT and by KV fitting in HBM); no
  bound for speedups, cold start or sessions/s; assumptions (one token per step, uniform routing, declared dtypes)
  are in each `basis`, so a value above 100% means one of them does not hold; `pct_of`, `ceiling_gain` = Amdahl) + `peaks.py` (`lab peaks`: HBM
  bandwidth and bf16 GEMM measured once per GPU+driver into `data/calibration/peaks/<gpu>.json`, run by `vm-setup.sh`; spec
  sheet as the labelled fallback), `repeat_audit.py`: repeated-request audit (MLPerf TEST04-style), judges decode time per token, not TTFT, so prefix caching passes and replayed answers fail; `lab repeat-audit --url URL`. `tier_agreement.py`: `tier_agreement(changes)`: Spearman, Kendall tau-b, bootstrap CI, out-of-band disagreements; short tier usable if n ≥ 8, rho ≥ 0.8, CI low ≥ 0.5. `holdout.py`: `HoldoutGuard` (Thresholdout query budget; state n is a write-once object in the store, a query writes n + 1 and retries on a lost race, so one budget per corpus version is shared by every run and machine; seed drawn once with state 0, kept only in the store) and `SealedSplit` (hash manifest, one logged unseal per corpus version). Keep sigma small relative to the threshold.
- `measure/context/layers.py` — each layer's share of a profiled window and its distance from a ceiling, as facts (2026-10-18):
  kernels = CUDA kernel time / window; model execution = prefill + decode phase wall time beyond kernel time;
  runtime = admit + evict phases, with the GPU idle share, batch fill and admissions blocked; serving = window time
  outside scheduler steps. Distance: the mean decode step against `measure.context.ceiling`'s step floor at the window's
  batch and context; kernels' % of roofline from gate results when given; layers 3 and 4 say they have no single
  ceiling. Every `profile` record (and every bench's and baseline's `profiled`) carries it under
  `calibrated.layers`, and each kernel task's share of the window under `calibrated.kernel_tasks`
  (`lab.tools.task_shares`, `generate/kernels/tasks/share.py`; why it is missing where the tasks don't import); the brief's harness
  facts show the latest of each (`layers`, `kernel_tasks`). `lab layers BUNDLE`. The timeline covers the whole run, so only events inside the window
  count; nested phases (prefill_mixed inside prefill_chunk) are counted once.
- `measure/sim/` — the scheduler simulator (layer 3, 2026-10-18): `measure/sim/scheduler.py` drives the engine's real
  `ContinuousBatchScheduler` one `step()` at a time (inference-server #120) on a fake clock (`Clock`, installed as the
  scheduler module's `time`) with `SimBackend`, which runs no model and charges a `CostModel` (prefill and decode
  step times, answer lengths; defaults are unfitted stand-ins, phase 2 fits them from a GPU run's step telemetry) and
  reserves KV blocks worst-case like the custom backend. Real: the scheduler, its policy, admission, KV arithmetic,
  request timestamps, TTFT/TPOT as the regimes define them; a 10-minute class simulates in under a second.
  `simulate(records, Settings, CostModel)` → regimes-shaped rows for `runner.summarize`; `Settings.from_env` takes the
  engine's scheduler env vars, so a profile applies as is. `lab sim run CLASS [--synthetic] [--env JSON]
  [--cost FILE]`, with the engine's python. Not yet validated against a real bench (the bar: ranks changes like bench
  does, Spearman ≥ 0.8); not an agent tool yet (the user's design).
- `measure/loadgen/synth.py` — synthetic traffic (2026-10-18): generated fresh per run from the corpus's seen split and then
  gone, never written, the seed regenerates it (`Ctx.synthetic`, `regimes run --synthetic`). Whole real sessions in
  shuffled passes (structure, turns, think time, max_tokens, regime), the real gap sequence repeated with a seeded
  jitter (bursts stay where they were), text cut from real messages of the same role at sentence ends and
  recombined to length: 0 of ~2,000 generated prompts per class equal a real one; KS on prompt tokens ≤ 0.02 and
  per-10 s arrival counts within a few % of the real class (`lab synth compare`). Not yet compared
  against the real corpus on a GPU; until then bench/submit stay on the public corpus.
- `measure/loadgen/arrivals.py` + `arrivals.json` — request timing fitted from real conversation traffic (2026-11-01):
  BurstGPT_3's conversation log cut into sittings (a session's gap over 30 min ends one; the user returns as a resumed
  sitting with that many turns of history), fitted on the seen weeks: counts of (first turn, turns) and gap quantiles
  by transition index (buckets at 5, 10, 20, 40). `Model.schedule(rate, duration_s, seed)`: sittings start as
  Poisson(rate) from -horizon (the p99.99 span of the model's own sittings, 23,049 s; a pre-roll sitting whose shape
  cannot reach t=0 is skipped undrawn) so users are mid-conversation at t=0; load rises by raising the sitting
  rate, which is what the data does (turns and gaps per sitting are the same at every hour of day, `seen_by_hour`).
  `lab arrivals fit|compare`: the fit carries the source's hash; compare puts the model beside the
  held-out weeks (sitting-start CV and KS against exponential, request-count dispersion at 1/10/60 s per busy
  window, KS real vs model for turns, first turn and gaps), numbers only. `workload.realistic` turns a schedule into
  requests: each sitting draws a conversation of the pool with turns enough for its indices (else one of the longest,
  the turns past its end not sent), turn j resends the history to the j-th user message at the reference's length for
  it, every sitting salted. Regimes on it (2026-11-01): `realistic` (the corpus mix, the task's objective; headline
  good req/s, `sittings_per_s` beside it), `long_prompt_short_output`, `shared_prefix_multi_turn` (sittings/s) and
  `overload` (2x the sitting rate); `short_prompt_long_output` stays Poisson (single-turn by nature) and `bursty` the
  real spike replay. `offline` = every record of every class at once, closed loop, output tokens/s, no limits
  (MLPerf's Offline). `saturated` also reports `curve`: each probed level's output tokens/s beside 1 / median TPOT.
  Every request is salted ("[k] " before the conversation's opening message; `Cycle`, `Once`, `realistic`), so no two
  share a prefix except a conversation's turns and the shared system prompt (MLPerf's rule). Tests set
  `suite.ARRIVALS` to `tests/fake_engine.fast_arrivals()`.
- `measure/loadgen/` — load generation: drives a live OpenAI-compatible server the ten ways it gets used, client-side.
  `client.py` (one streamed chat request: TTFT from *scheduled* send to first real token, TPOT = (E2E − TTFT)/(n − 1),
  status ok / error (explicit 4xx or 503 rejection) / silent_drop / truncated / crash (other 5xx, no response,
  stream died); a one-token answer has no TPOT and never meets the limits; corpus headers X-Session-Id,
  X-Turn-Index, X-Trace-Id; a record with `expected_output_tokens` is sent at that length, or the regime's
  `max_tokens`, with `ignore_eos` and `skip_special_tokens: false`, and one that ends earlier is `truncated`:
  `workload.to_request`), `runner.py` (open loop: schedule dealt to 4 client processes on one shared start, each
  process one event loop (`client.asend`: body encoded and connection opened 100 ms early, only the write at the send
  time; a thread per request queued a burst's sends, 14 ms p99 lag at bursty 16x), so p99 send lag stays ~0.5–1.5 ms
  even for 800 arrivals in 50 ms; closed loop: a thread per client; `summarize`: p50/p99, failures,
  attainment, valid only if no silent drop/truncation/crash and client lag p99 ≤ 10 ms; `find_goodput`: double then
  bisect on ≥99% meeting both limits, up to 40 probes, reporting the lowest failing load `fails_at` and why it stopped,
  `stopped`; saturated sweeps to 4,096 concurrency; bursty = the spike trace at N x its requests over its own span,
  `workload.scaled`, decided 2026-10-27; `regimes run --tol` sets the search's resolution; a record reused in one
  schedule is salted, `to_request(salt=)`; closed loop can split over processes, `--client-procs`, invalid when a client
  process ran above 80% CPU; `--client-cpus` pins the client), `workload.py` (reads `data/corpus/` through `data.corpus`'s verified loader — hashes checked, a changed trace
  refuses — plus Poisson, trace replay with speed-up and peak-window crop, multi-turn
  conversations rebuilt from stored histories under one shared system prompt; synthetic prompts when no corpus),
  `suite.py` (the 10 regimes; probes ≥ `min_requests` (100) and tier length, capped at 4× the tier length, then a
  confirmation at `final_s` backing off by the search's resolution up to 5 times, a failed confirmation counting as
  a failing load; the load search stops below start/16; saturated doubles concurrency, then bisects between the last
  passing and the lowest failing level to the band (else 10%); closed-loop schedules are generated lazily; open-loop
  probes (and overload's) score only requests sent under steady load: arrivals start one request's time at the limits
  (`_settle_s`: TTFT limit + p90 length x TPOT limit, 104 s for a 1,024-token answer) before the scored window and run
  as long after it, decided 2026-10-29; before the first regime measured on a launch, `_warmup` serves the corpus mix closed
  loop at `WARM_CONCURRENCY` (256) for a probe's length in an order of its own, unscored, because an engine's first load
  at a size runs slow (2026-10-28: the first saturated level after a launch read up to 40% low); single_stream runs its
  final length and at least `min_requests` requests, never past `MAX_STRETCH` x the final (`run_closed(min_requests=,
  max_s=)`: a p99 of 14 requests was the slowest of 14); shared-prefix runs also add their ramp; cold start launches a command with
  fresh VLLM/inductor/Triton cache dirs, polls until "17 + 25" is answered "42", then relaunches for the warm figure).
  Tiers: min (15 s probes, 30 s final; bursty cropped to its peak window like short), short (30 s probes, 60 s final),
  full (60 s probes, 600 s final). Probes must last well beyond the TTFT limit
  or a short overload goes unseen; limits are settable per `Ctx` (MLPerf by default). CLI:
  `lab regimes run REGIME[,..]|all --url URL [--corpus DIR | --workload FILE] [--synthetic] [--split] [--tier] [--tokenizer] [--out]`
  (`--workload`: a `data/workloads/` file is one pool for every regime; bursty is refused, no arrival times),
  `lab regimes cold-start --url URL --cmd "..."`. Closed-loop runs can't detect a client bottleneck by lag.
- `measure/loadgen/clientcheck.py` — `lab clientcheck`: the client's own TPOT error at each closed-loop concurrency
  and open-loop rate and burst size (`BURST`: N arrivals within 50 ms), against a fixed-schedule server that logs its
  true send times.

### `evals/` — the judges: the agent's gates, the researcher's views, benches, the kernel evaluator
- `evals/researcher/dashboard.py` + `dashboard.html` + `evals/researcher/live.py` — the researcher's live view (2026-10-23): `lab dashboard
  [--port 8080]`, on 127.0.0.1, refreshed every 2 s. Reads only the ledger and each run's
  `<run>/live.jsonl`, which `session.run` has the lab append to (`live.emit`): run start/end with the cap, each session,
  the agent's text and tool calls with the API proxy's spend so far (both providers), worker job start/end, and on the
  seen split each regime and each load call's raw numbers, computed after the call (never the held-out split's).
  Never touches the engine or the GPU, and no agent view reads it. Live load numbers need `LAB_WORKER=local`; a VM
  worker's are not pulled back (not built).
- `evals/agent/validate.py` — `lab validate --target T`: serves the target's reference and writes its outputs, judges
  each `[validation]` good/bad launch (good must pass, bad must fail: the gate separates on *this* model and GPU), and
  measures each regime's run-to-run band from repeated runs of the engine into `noise` ledger records tagged with
  its commit (which `submit` reads for runs on that base commit only), its load searches resolved to
  `RESOLUTION_PCT` (5%), below which a searched regime's band is never reported. Exit 0 only if both hold. Needs a GPU for a real target; tested on fakes. `data/knowledge/LEDGER.md` is the
  record-format spec: the environment's public interface.
- `evals/researcher/feedback/` — what each side gets after a session of a lab ledger. `report.py`: `feedback(path, run_id)` / `lab_feedback` → `{for_agent: {integrity, facts: {harness}}, for_researcher: {activity, process}}` (`for_researcher.activity` holds the derived facts, `for_researcher.process` the process evals); `lab feedback PATH [--run RUN_ID] [--agent-text]`. Integrity carries `evidence` per violated rule (rule, plain detail, concrete refs: hashes, measurement ids, offending values and limits).
  - `lab_verdict.py` — lab ledger → integrity + derived activity facts. Record shapes are in `data/knowledge/LEDGER.md`; `gain()` reads submit aggregates (improved / regressed / within_band). Submitted snapshot must have a completed short-tier seen bench, tests passed, a passing equiv and no failing equiv (inconclusive is neither; retrying a statistical gate until it passes would let bad changes through).
  - `claim_facts.py` — claim-vs-evidence facts from ledger `claim` text (regex, no LLM): unrecorded numbers, contradicted speedups, verification words with no matching record, unknown record ids. Facts only; never part of the verdict.
- `evals/researcher/lab_process.py` (was `diagnostics/`) — researcher-only, never fed to the agent. `lab_process.py` (lab ledgers: whether each
  prediction's direction held, outcome and mechanism, in four cases; bundles never measured apart; numbers in
  reflections that match no record; in `for_researcher.process`).
- `evals/agent/correctness/` — correctness gate. `divergence.py` (teacher-forced KL / top-1 agreement / ref-token
  logprob shift, top-k or full logits), `flips.py` (paired flip test, exact one-sided McNemar), `checks.py` (length
  ratio, consistency), `gate.py` (`evaluate` → three-valued verdict pass|fail|inconclusive; gates = pooled
  accuracy ≥ 99% of reference, length, and consistency only when a caller passes it (the candidate run does not,
  nor does it score divergence: that is `quick`'s, decided 2026-10-28); a ratio under the line FAILS only if the net loss is
  significant (McNemar p < alpha 0.01), else INCONCLUSIVE, because a non-deterministic engine churns ~1 point at
  the dev tier; unanswered items (shed/errored after retries) → inconclusive; 95% CI on the ratio is a fact;
  divergence and per-task flips are facts; `to_ledger_record` → lab ledger `equiv` record),
  `scoring.py` (MMLU-Pro letters, MATH `\boxed{}`), `tasks.py` (loaders from the Hub into `local/tasks/`; tier `dev`
  ~1.6k = MMLU-Pro 490 stratified, MATH-500, MBPP sanitized 427, needle 180; tier `full` ~18.6k = MMLU-Pro 12,032,
  MATH 5,000 (gold = last `\boxed{}` of the solution), MBPP full 974, needle 600; code scored by running unit tests
  in a limited subprocess; needle = synthetic retrieval at ~4k/12k/24k tokens), `client.py` (`Api(name, chat_kwargs)`: `generate`,
  `generate_stream`, `score_tokens` with adapters vllm (`prompt_logprobs`), sglang (native `/generate`
  `input_token_logprobs`), none (raises `NoLogprobs`; divergence reported as not run)), `run.py` (`reference` once
  per model/hardware/tier, `candidate` per change, `quick` (the candidate scores the dev reference's sequences
  teacher-forced, nothing generated: divergence facts, no verdict; the equiv tool's tier `quick`); 429/503 retried with backoff, items still failing are
  "unanswered" and never scored as wrong; scoring runs at a quarter of the generation concurrency; `verdict_file`). CLI: `lab correctness reference [--tier dev|full]|candidate|verdict`
  (`verdict` re-judges saved results under the policy, no server).
- `evals/researcher/canaries/cheat_proxy.py` — cheat proxy in front of any OpenAI-compatible server (`truncate`, `early_eos`,
  `fake_first`, `drop`, `cache` (streamed and not), `inflate_usage`); `lab cheat-proxy --upstream URL --cheat NAME`.
- `evals/benches/compare/` — the engine comparison (decided 2026-10-27): `engines.toml` (ours, vLLM 0.32.0, SGLang 0.5.22,
  TensorRT-LLM 1.4.0rc1 in their official containers; bf16 and fp8 configurations; shared limits; server cores 0-7,
  client 8-15), `run.py` (`lab compare run --out DIR` on the GPU VM: the client check, then per configuration a
  launch check, the regimes at `--tol 0.02`, equiv, cold start; repeats rotate the order; each stage one file, skipped
  when present), `report.py` (`lab compare report DIR [--gpu-usd-per-hour P]`: per regime the winner and whether
  the lead is clear of every spread and of the runner-up's bracket; the saturated curve with $/M output tokens),
  `vm-setup.sh` (images and checkpoints; its header lists the session's steps), `profile.py` (`lab compare profile
  --out DIR`: each engine under `nsys launch` with the comparison's settings, at fixed load points
  (`POINTS`: single stream, saturated mix at 256, 1,024-token answers at 512, long prompts at 8 req/s), a 10 s nsys
  window once steady, then 10 s of py-spy over the server's processes; `--plain`: no profiler, the client's numbers
  only, the engine's env from the caller's, for A/B runs of engine trees), `gap.py` (`lab compare gap DIR`: per
  point each engine's client numbers, GPU busy share, kernel launches, top kernels, top Python frames). First run
  2026-10-28 (spot, `local/compare-20261028/`): vLLM bf16/fp8 and SGLang bf16 measured; ours hung at saturated
  (inference-server#137, a lock-order deadlock); long-output/long-prompt values were 30 s probes shorter than one
  request, and bursty stopped on client lag: both fixed 2026-10-29.

### `generate/` — generators: the kernel search, the OpenEvolve arm, tasks, the library of other implementations
- `evals/kernels/` (the evaluator: check, timing, core, jailed, verify, ncu, precheck, ship) + `generate/kernels/` (were `kernels/`) — kernel equivalence: `check_kernel` (seen + held-out shapes, per-dtype tolerances, edge cases incl. non-contiguous and NaN/Inf, inputs unchanged, no aliasing, memoization, determinism) and `time_kernel` / `check_memoization` (sync, rotated buffers, L2 flush on CUDA). Validated on CPU, MPS and CUDA (H200, 2026-10-04): all tests pass; the timer resolves ~20 µs kernels at ~9% IQR, 0.2 ms at 1.5%. `make_inputs` gets a seeded CPU generator; large tensors are made on the device by a generator seeded from it
(`generate.kernels.tasks.randn`: generating them on the CPU took seconds per check case at real shapes).
  `generate/kernels/tasks/` (a kernel task: the engine entry point it replaces = the baseline, a reference, an input builder,
  how to read a shape from a captured call, the held-out axes, the layouts its callers pass, bytes and FLOPs per call
  for the roofline; `toy_rmsnorm` for CPU, `paged_decode` = the example engine's `paged_decode_attention` (block tables as wide as its graphed decode passes them, from the target's CONTEXT_WINDOW: `Task.options`, `tasks.from_config`), `paged_decode_fp8` = the same on fp8 KV pools (KV_CACHE_DTYPE=fp8, unit scale;
  its captures run with that env, `Task.capture_env`), `moe_gemm` = its MoE grouped
  GEMM `moe_kernel.moe_gemm`, both projections, real routing from captures, bf16 weights; `moe_gemm_fp8` = the same with fp8 e4m3 weights and per-channel
  scales, as the engine runs it by default since inference-server #122; `moe_gemm_scatter_fp8` = its down projection
  `moe_kernel.moe_gemm_scatter` (weighted rows added into an fp32 output); each reads only its own calls by weight dtype; `rmsnorm`, `add_rmsnorm`, `rmsnorm_rope`,
  `silu_mul`, `topk_softmax` = its fused kernels in `models/fused.py`, references from its own eager paths, the start
  that module with its kernels module imported directly (`Task.start_edits`: the screen refuses its importlib/getattr); prefill
  attention has no task: `sdpa_prefill` takes the backend's paged context, not tensors),
  `generate/kernels/tasks/shapes.py` (shapes and call counts from captures; held-out = a whole band of each held-out axis, above its 60th and up to its 80th percentile by calls, never
  the axis's largest value and never random samples (decided 2026-11-02); `ints_from_captures` keeps each shape's captured int args, e.g. real sequence
  lengths, which `Task.with_ints` uses for checking and timing in place of synthetic ones), `generate/kernels/evolve/` (the layer-1 generator, a replication of OpenEvolve 0.4.0 =
  open AlphaEvolve: `lab kernels evolve run|gate|split`; the starting program is the engine's kernel module;
  each run appends one `kernels` ledger record (every candidate's seen-shape metrics and lineage, OpenEvolve's
  evolution trace and the best program as blobs; `data/knowledge/LEDGER.md`);
  prompts carry facts only (`generate/kernels/evolve/prompts/` overrides OpenEvolve's advice; `run.setting` states the seen
  shapes with calls, score weight and capture traffic (`shapes.sources_from_captures`), the GPU, peaks, toolchain,
  launch settings and tolerances, for both generators; `run.system_message` states how a
  program is judged, from the evaluator's own constants);
  a precheck with no GPU attached (`jailed.PRECHECK_STAGE`, `evals/kernels/precheck.py`: the screen, the
  CUDA C++ build for the GPU's architecture from the config's `arch`, each built kernel's registers, stack, shared and
  local memory from cuobjdump as `compile_report`; only a failed build drops a program, spills are facts, an import
  that needs a device is inconclusive; Triton kernels build on first call, so not here), then a two-stage cascade, each stage a fresh jailed process (`evals.kernels.jailed`: the jail gets the candidate, its
  config and a copied harness, never this repo; `core.judge`: stage 1 = two shapes, one seed; stage 2 = correctness
  on every seen shape (one seed, plus memoization, non-finite inputs, inputs unchanged, aliasing) + paired timing against
  the baseline, and only in the gate (config `full`, or held-out shapes given) the full check: three seeds x 3 repeats, edge shapes and
  layouts, value scales, the baseline's error, peak memory (decided 2026-11-01) (`kernels.time_pair`: same inputs, random order per
  pair; a task whose callers replay it in CUDA graphs, `Task.graphs`, e.g. `paged_decode`, is captured and timed as a
  graph replay behind a GPU spin so its launch is not timed, and a kernel that cannot be captured scores 0; the
  timed calls' outputs are checked against the reference too, so a kernel that does less only when timed scores 0; every
  plain check case's and timed call's outputs are verified again outside the jail by a process that never loads the
  candidate, `evals/kernels/verify.py`, gate `verified_outside`, which requires every record the config implies:
  each seen shape x dtype at seed 0 (the gate, `full`: each seen and held-out shape x dtype x seed 0-2 and each value scale),
  each timed buffer; the result is the one stdout line
  carrying a per-judgment key that core reads and removes before the candidate loads; between judgments only torch's
  build files survive in a program's dir, and the gate has its own), score = call-weighted geometric-mean speedup, a shape counting 1.0 when the middle half of its paired ratios
  includes 1; at the end the top candidates are re-judged 3x and the highest median is promoted, `run --retime`); a program may build CUDA C++ at import with
  `torch.utils.cpp_extension.load_inline` (nvcc from the host's toolkit, mounted read-only into the jail; the build is
  cached per program source and toolkit across its stages, in one fixed dir per program, `jailed._eval_dir`; C++ only as
  literals, `core._cpp_built_at_run_time`); a candidate is
  refused if its source, its C++ included, reaches for an exploit route (the operator registry among them:
  `torch.library`, `TORCH_LIBRARY`, dispatch modes, which could change what the baseline's own ops do) or writes into anything it imported, under any alias
  (`core.screen`), and scores 0 if any function the checker or timer uses is patched, checked after loading and
  after every call the checker makes (`core._watched`; the timer binds its own at import); the exploit canaries in
  `tests/test_kernels_evolve.py` must all score 0; selection never sees held-out shapes, `gate` judges them (correctness, and their speedup as a fact), then runs a program that passed under `compute-sanitizer` (`evals/kernels/sanitize.py`, jailed stage 6, decided 2026-11-04: memcheck, racecheck, initcheck, synccheck, each in its own jail, one call on the smallest and largest seen and the largest held-out shape, caching allocator off, host API errors not reported (Triton's loader makes them on the engine's own kernels); any device-side error fails gate `sanitizer` with the report as evidence; a host without it, or a region whose weights are attached over IPC from the holder, records "not run" as a fact); `validate_gpu.sh` = the GPU validation (shapes captured under real corpus traffic at several concurrencies, `CAPTURE_SPECS`; never synthetic prompts, whose batches have uniform lengths); tolerances from the target's
  `[kernels]`). `generate/kernels/search/` (extra `kernel-search`; our own layer-1 generator, replacing OpenEvolve, which stays as the A/B baseline: `lab kernels
  search`; by default the genetic search (`generate/kernels/search/generations.py`, decided 2026-10-26): generation 0 is
  one round from the start program (40 changes; competitors' ported kernels, `generate.kernels.library.competitors.starts`, judged beside
  it and joined to the population), then exactly `--generations` (10) generations (decided 2026-10-30:
  nothing stops it sooner), each a round whose `--gen-size` (24) changes are split over
  four idea types (decided 2026-10-29;
  `--no-types` = refine only, arm `search-untyped`): refine (new ideas on `--parents` (8; was 4 until 2026-11-03) programs drawn from the
  population: the 32 best open programs and the fastest open program on each seen shape (`elites`, MAP-Elites with the
  shape as the niche, decided 2026-11-03), the best always, the rest by percentile reward + 0.05 (an elite's as the
  best's) without replacement),
  persist (a line's last change taken further; every child that beat its parent heads a line, which moves to a persist
  child that beats it and closes after 3 generations without one), fresh (a different overall structure, half from the
  start, half from a random program) and crossover (a parent shown another parent's full code), 8/6/6/4 adapted each
  generation by each type's Laplace-smoothed rate of new bests (floor 2), after a new best found by the last generation
  takes half of the ideas as refine ideas (`NEW_BEST`, decided 2026-11-03); every correct program takes the full exam,
  children under their own parent with their type (`Node.kind`); every call gets
  the memory (`generate/kernels/search/memory.py`, cached blocks: ideas that beat their parent on other tasks with their best gain
  and bound type, then every idea tried on this task, earlier runs from `--archive` and this run so far, with measured
  gains; archived ideas naming a program of their own run left out); with `--mcts` instead a tree of refinement chains, one parent per round chosen by MCTS (UCT with
  C = 0.6 on percentile rewards and max backup,
  KernelPro's progressive widening and virtual-child rule), a node closed after 3 judged failed children or 3 edits
  that never reached the GPU; in every mode the model is shown the
  parent program, its measurements, its lineage, the starting program and up to 2 better programs from other branches,
  all as full code (CodeEvolve's inspirations; decided 2026-10-30), one call per change answering with its hypothesis as
  JSON and the change as SEARCH/REPLACE blocks (`generate/kernels/search/funnel.py`; each and its fate in `hypotheses.jsonl`),
  duplicates refused before any GPU time, from the facts (and, its choice when, Anthropic's web search
  and read-only SQL over the agent's view of the ledger index, `--judge-tools`; the harness pastes no research),
  including CudaForge's 24 ncu counters
  (`evals/kernels/ncu.py`, stage 3 of the jailed evaluator; collected in the jail since 2026-10-27, on a Nebius H200
  whose driver had them open: the earlier refusals, ERR_NVGPUCTRPERM, were on hosts with them closed, which
  `vm-setup.sh` never opened because it grepped the module option's name, not the driver's `RmProfilingAdminOnly`; a
  run stops asking after the first refusal); the harness tunes nothing (decided 2026-11-03, its grid search removed): a
  program tunes itself at run time if it does (e.g. `triton.autotune`), and the system message states that the checks
  and warm-up calls run before each shape's timed window; before the held-out gate (`generate.kernels.evolve gate`, and
  `kernel_search` via `pipeline.freeze`) the winner's autotune is frozen (decided 2026-11-04, `evals/kernels/freeze.py`:
  jailed stage 7 runs it on every seen shape and records each autotuner's choice per key; each module-level
  `triton.autotune` becomes `_FrozenTune` over that table, the call's key or the nearest tuned one: equal non-numeric
  values, then the smallest sizes at or above, then the closest), and the frozen program is what the gate judges and
  ship commits; ship and the patch refuse an autotune the engine module did not already have as written; the prompt states the best program's % of its call-weighted roofline bound as a fact, and with
  `--mcts` the search stops after `--patience` (16) judged programs with no new best (MANTIS; no roofline stop since
  2026-10-27);
  each timed shape carries its bound type (memory or compute) and the achieved % of peak bandwidth and of peak FLOPs,
  and ncu runs on the smallest and the largest seen shape; before promotion a model (`--detector-model`, Opus 5.5) reads each would-be winner for gaming and a flagged one is
  passed over, recorded with its evidence (`generate/kernels/search/detect.py`, MANTIS; never changes a score);
  every call priced (OpenEvolve's too, via a usage log);
  `lab kernels ab --arms ...` runs OpenEvolve and kernels.search (and variants differing in one flag:
  `search-no-tools`, `search-mcts`, `search-c1.41`, `search-gen20`, `search-untyped`) on one task with the same shapes, start, evaluator,
  the same post-search promotion for every arm (`ab.SAME_PROMOTION`: top 3 re-timed, no tuning, no detector), single model
  and judged-program budget, gates each winner on held-out shapes, and writes `ab.json`; `generate/kernels/search/METHOD.md` maps
  every component to its published source and number, or says why ours; same evaluator, run dir, re-timed promotion and `kernels` record as `generate.kernels.evolve run`). `evals/kernels/ship.py` (`lab kernels evolve ship`: a program with a passing held-out gate becomes a
  commit on a new engine branch, built with git plumbing so the engine's tree is untouched; `--budget` starts the
  `lab start` researcher run on it, measured end to end by `bench`/`equiv`/`submit`). `generate/kernels/tasks/share.py` (`lab kernels evolve share --profile BUNDLE [--profile ...] [--gate TASK=GATE.json]`: tasks
  ranked by share x (1 - % of roofline) across profiles, each task's share of a profiled engine
  window from the device kernels it launches, `Task.kernel_names`, the speedup bound if they took no time, and the
  kernels no task covers, by share: the facts for choosing the next task). `measure/profile/capture.py` + `measure.profile.profile --capture MODULE:FUNCTION` (the `profile` tool's `capture`): every
  call's argument shapes and small int tensors, the first call per shape kept whole as real inputs.
- `generate/kernels/tasks/regions.py` — regions (decided 2026-10-22/23): any run of consecutive decode stations of the engine as a
  kernel task with a generated harness (since 2026-11-02 prefill stations too: a region whose first station starts
  "p_" binds the engine's `PREFILL`, `station_set`; its held-out axis is the tokens of the packed wave; a read the
  engine passes as None in the captured calls, e.g. a packed wave's `tokens`, is a constant None and the calls where it
  is a tensor are left out; a shipped override runs the engine's own stations on such a call): `Station(name, reads, writes, fn)`; `interface` (inputs read before written,
  outputs read after the region or `live`), the stations as the baseline (`Task.baseline_fn`), an fp32 run as the
  reference, `bytes_moved` (fused vs one-by-one), and the error budget: `Task.error_budget` (2) lets each element differ from the reference by
  that many times the baseline's own largest error among reference values of that element's size (the same power of two; the farthest of 3 calls, `check.baseline_envelope`, so a
  non-deterministic baseline's spread is its error, and its error may land anywhere, `check._by_size`), on top of the dtype's tolerance, and a NaN/Inf mismatch is within it
  where the baseline has one there too (`check._compare(base=, k=)`), in the
  checker, the timed outputs and `verify.py`; the baseline is bound before the candidate loads and its modules (`Task.watch`) are watched for
  patches. The engine's stations (`models/stations.py`) and region capture are merged in the engine (main 0a00b0c); the
  tests use fake stations. `from_engine(first, last, samples, shared_weights)` binds a region to the engine's
  `inference_server.models.stations.DECODE` (as `functional(DECODE)`, which copies what kv_append writes in place, so a
  region including an `IN_PLACE` writer is refused): inputs from a CAPTURE_REGION dir's samples of the holder's `layer`
  (seed 0 exact, later seeds perturbed), the layer's weights attached from a shared holder, its non-tensor entries
  (`cfg`, None weights) the constants that `start_program` writes in too; the flat shape signature holds out the first
  activation's rows (and the attention block tables' width); shapes are read from calls named `Task.capture_fn`
  (`region:F:L`); `load("region:FIRST:LAST")` and `from_config` (keys `samples`, `shared_weights`) take it, and so do
  `generate.kernels.search` and `generate.kernels.evolve run|gate|split|ship` (`--task region:F:L --samples DIR --shared-weights FILE`;
  the samples are the captures unless `--captures` is given; both keys go into `eval_config.json` and the gate's
  config). `load` with neither key gives `regions.light`: name, entry, capture env (`CAPTURE_REGION_LAYERS` =
  `regions.LAYER` 0, `MEGAKERNEL=0`) and held-out axes, no engine import: what the controller knows. kernel_search takes
  `region:F:L` end to end (`generate/kernels/tasks/pipeline.py`): captured with `measure.profile.profile --capture region:F:L` (sets the engine's
  CAPTURE_REGION/_DIR before the backend is built) under the scored regimes' traffic, `pipeline.pack` merges the
  captures into one samples dir (each traffic's calls kept apart under traffic<i>/profile-<source>/captures, so each
  weighs the same), pushed to a remote worker, and `pipeline.hold` runs `python -m generate.kernels.tasks.hold` where the evaluator
  runs (over the evaluator's ssh env, or locally with the engine's python): it loads the engine (target env, 16 K/V
  blocks, no graphs, no megakernel), shares the region's weights (`Holder(jail_uid=)`), describes the region (shapes
  split with `heldout_axes`, ints, sources, the start program, the engine's station names) on one JSON line, and holds
  until its stdin closes. The search runs on `--shapes --initial --samples --shared-weights` (paths on the worker;
  with KERNEL_EVAL_SSH `run.load_task` describes the region by `regions.light` from the shapes file, never loading it
  on the controller), the gate on the described shapes, and the winner comes back as a patch adding the override
  module (`ship.region_files`, git-free, overlap checked against the overrides in the agent's tree). The jail mounts the samples and the holder
  file with its `.pristine` and `.constants` sidecars read-only and shares the host's IPC
  namespace (root lab only, never at the precheck); afterwards every /dev/shm entry of the jail's uid is removed.
  `generate/kernels/tasks/shared_weights.py`: a holder shares tensors by reference (CUDA IPC, or shared memory on the CPU) and writes
  a pristine copy beside the file (files unpickled outside the jail must be the lab's, not group/other-writable, not
  under /dev/shm or a jail-writable dir); judgments on one holder hold its lock, `restore` runs before and (in a
  `finally`) after each one, and a judgment that wrote a weight is void and the weight copied back. `time_pair`
  fingerprints every input between calls (outside the window): a call that changes an input fails the timing. On the
  H200 (2026-10-27) a jail attaches over CUDA IPC once the holder (`Holder(jail_uid=)`, a root lab's jail user) links
  its IPC semaphore pools in /dev/shm under the jail user's name (`cuda.shm.<uid>.<pid>.<n>`, hex; an importer looks
  them up under its own uid), open to that user's group; where an attach is still refused, `attach(copy_ok=True)`
  (only `from_engine`, the evaluator's task) copies the pristine weights to the holder's device, kept in
  `.constants` (`attached_by_copy`; fine for one layer's ~0.6 GB, not for a whole step); `restore` never copies and
  raises when it cannot alias the holder's memory. A region's winner ships as a module of the engine's overrides
  package (`OVERRIDES_PACKAGE`, `override_module`, `ship._region_files`; refused when it overlaps an override already in
  the base tree); each call checks the engine passes the constants it was built with (None scales, `cfg`); the fp32
  reference keeps the baseline's experts (`run(..., pinned)`) and computes each pair's weight in fp32 at them
  (`pair_weights`: softmax at the pair's expert, renormalized when `norm_topk_prob`); the modules the
  stations call (`STATION_MODULES`) are watched; the K/V pool's shape is not part of a shape. First run on the H200
  2026-10-27 (engine branch engine/region-capture): router..down of layer 0 captured under mixed_1 and mixed_3, held,
  split and gated jailed with the weights attached over IPC: the start program passed every gate, 1.0x seen, 1.001x
  held-out. Not yet run through kernel_search or over ssh on a GPU.
- `generate/kernels/tasks/step.py` — the whole decode step as one region, `region:embed:argmax` (`measure.profile.capture.STEP`): the engine's
  megakernel (`megakernel.decode_step`) as a kernel task. Captured by `measure.profile.capture.step` (`measure.profile.profile --capture
  region:embed:argmax`: wraps decode_step; the first call per (rows, block-table width) kept whole up to 256 MiB:
  tokens, pos, bt renumbered to the blocks its rows touch, every layer's K/V of those blocks stacked [layers, blocks,
  ...], and logits, next tokens and those blocks after). Inputs (tokens, pos, bt, k_cache, v_cache), outputs (logits,
  K/V; `next` is checked as a greatest logit of its row and dropped, `step.derived`: an integer fits no budget and a
  bf16 step's argmax flips where the fp32 reference's does not): pure, so baseline and programs copy the K/V per call. The weights are not inputs: `Task.call` passes the
  holder's dict (`step.weights`: every layer's `<i>.w.*`, embed, final norm, lm_head, inv_freq; cfg, n_layers) after
  the inputs, so the checker never clones them; attached over IPC only (no copy fallback). Baseline: the engine's
  decode_step on a Model namespace built from the held weights; reference: the engine's `stations.reference()` in
  fp32 over every layer, error budget 2x; bytes bound: every non-expert weight + top-k experts per layer + lm_head + the
  K/V. Start program: decode_step.cu as a raw literal (its globaltimer stamp edited to 0, `START_EDITS`) built with
  load_inline under the engine's build flags. Shapes (rows, bt width), both held-out axes. No NaN/Inf inputs
  (`Task.nonfinite` False, decided 2026-11-04): a row whose router logits are all NaN has no top-8 (the megakernel
  picked expert E and read past the expert stack, the gate's crash of 2026-11-03, until inference-server's fix). A
  winner does not ship: the engine's `STEP` hook (a module of `models/regions/` defining `STEP(tokens, pos, bt,
  k_pools, v_pools, weights)`) writes the pools in place while the task's K/V is pure, so `ship.region_files` refuses
  it and kernel_search reports `patch_refused`. Tested on the CPU with a fake megakernel and the engine's stations; on
  the H200 (lab-gpu-18, 2026-11-04, steady_interactive 1 and 8) the start program failed correctness on 3 of 6 cases
  (max_abs 0.5) while the determinism check found the megakernel's own outputs differ across identical calls; with
  the spread in the budget (lab-gpu-19, 2026-11-05) still 1 of 6 cases and memoization (a `next` flip) failed, hence
  the budget by size and the `next` rule (decided 2026-11-06); not re-run yet.
- `generate/kernels/library/ptx.py` — the PTX comparison's cards (decided 2026-10-22, after Standard Kernel): each implementation's
  kernels counted from its PTX (or `cuobjdump -sass` output, for libraries that ship no PTX): instruction classes,
  widest vector access, declared registers, static shared memory, and the Hopper features in use (wgmma, TMA,
  setmaxnreg, mbarrier, cp.async, mma, ldmatrix), with a measured time beside it when given; fastest first.
  `lab kernels ptx card --name N PTX ... [--timing N=MS] --out F`; the cards live in
  `generate/kernels/library/ptx/<task>.txt`, built by `lab kernels library cards` (only moe_gemm_fp8's exist yet).
- `generate/kernels/library/` — other implementations of every kernel task in one place (decided 2026-10-31): `index.toml`
  names 2-3 per task from vLLM v0.32.0, SGLang v0.5.22, FlashInfer v0.7.2.post1 and TensorRT-LLM v1.2.1 (upstream
  paths with line ranges; whole files kept verbatim in `src/`, `lab kernels library fetch` re-downloads them),
  and `block(task)` is the part of every search prompt (the system message, cached) that shows each one's source, URL
  and PTX/SASS card where built, and the starting program's card; no timings. Each entry also names the engine stations
  it implements (`stations`), so a region's prompt gets every entry for its stations (`for_stations`; the hold side
  writes the region's station names into its shapes file), FlashInfer's batch prefill included for `p_attention`. `lab kernels library cards --task T
  --out generate/kernels/library/ptx/T.txt [--dump DIR]` on the GPU host: each entry's card from a `generate.kernels.library.competitors` dump or from
  its package's .so (`sass` in the index; SGLang's JIT kernels and Triton kernels without a dump have none). Regions
  have no entries.
- `generate/kernels/library/competitors.py` — competitors timed on our shapes (`lab kernels competitors`, libraries installed) and
  `starts(task, program)`: each task's vLLM v0.32.0 port as a starting program from source vendored in
  `generate/kernels/library/vllm_sources.py` (no library needed; rmsnorm, add_rmsnorm, silu_mul, topk_softmax as CUDA C++ built at import,
  moe_gemm(_fp8), moe_gemm_scatter_fp8 and paged_decode(_fp8) as Triton), routed to the engine's kernel where unsupported.
- `generate/kernels/search/funnel.py` — one round (decided 2026-10-22; one call per change and every correct program measured
  since 2026-10-30, replacing the funnel's tiers: named ideas, merge, past-failure check, wide coder, repairs, quick
  timing, strong rewrite), from one parent (`expand`, an MCTS expansion: `--per-call` changes) or several
  (`generation`: one `Slot` per parent and idea type with its count; a refine slot's calls read the slices in turn;
  persist, fresh and crossover calls read the measurements). The slices: measurements, winners elsewhere in the tree,
  compiler report (the PTX cards moved into every call's system message with `generate/kernels/library`); a slice with no facts is skipped; the type's request ends the prompt (`loop.MODES`).
  Calls cycle through the mix (`--judge-model` Sonnet 5.5, `--second-judge-model` Opus 5.5 one call in `--second-judge-share` 4; decided 2026-11-03). Then edits that
  do not apply, repeat a program, or change their parent within cosine `NOVEL` (0.95) of a sibling's change (decided 2026-11-03)
  (fastembed on the changed lines, `near_duplicates`; ShinkaEvolve) are dropped, the no-GPU precheck (`--precheck`, default on) drops what fails to
  build and keeps the compiler's report on the program (the compiler slice of its children), stage 1 in `--parallel`
  jails, and every correct program takes the full exam and joins the tree. A parent gets its ncu
  counters (stage 3) when it is first chosen. Model calls run in waves of `--workers` (8), each call shown the changes earlier waves of the round proposed and
  asked for a different one (`funnel.proposed`, decided 2026-11-01), the round's first call alone so the rest read
  the prompt cache it writes (`generate/kernels/search/llm.py`'s `CACHE_SPLIT`: the stable part of a prompt is
  sent as cached blocks); GPU timings run one at a time. Each hypothesis may predict ncu counter changes (`mechanism`),
  measured against the parent's. Every idea's fate goes to `hypotheses.jsonl` (each idea with its `parent_id`; the
  row's `parents` and `improved`), which the archive reads. `--facts FILE` adds facts to every prompt; the
  `kernel_search` tool passes the scored regimes and the task's share of the latest profiled window (`pipeline.facts`).
- `generate/kernels/archive.py` — the idea archive (decided 2026-10-22): every distinct kernel idea from the `kernels` records'
  hypothesis logs and traces, with each outcome (run, parent and its score, child and its score, improved, stage
  reached, error); an idea within cosine 0.92 of an earlier one on the same task is folded into it (identical texts only
  without fastembed); `matches` (similarity, else BM25), `retry` (never beat its parent, tried only on parents other
  than the task's current best), `facts` (what the past-failure check sees); each outcome keeps the bound type of
  the child's seen shapes; `generate/kernels/search/memory.py` turns it into the judge's memory tables. `lab kernels archive export --out F` on the
  controller (the GPU host reads the file), `query --archive F TEXT`. Measurements only.
- `generate/kernels/tasks/kernelbench.py` — KernelBench as kernel tasks, researcher-only (2026-10-28): `lab kernels kernelbench
  fetch|list|score|run`; the repo at a pinned commit into `local/kernelbench/<commit>/` (problems + the benchmark's own
  eval package, sha256 manifest); `kb:LEVEL:ID` tasks (`generate.kernels.tasks.load`; the eval config carries the problem's
  source into the jail): inputs from its get_inputs on the device, weights under a fixed seed, fp32, reference and
  baseline its eager forward, its tolerance (Task.tolerances), the benchmark's contract (no layouts, scalings, NaN/Inf
  (Task.nonfinite) or edge shapes, no roofline), a held-out size; start program = problem + `ModelNew(Model)` + `kernel`.
  `score`: the benchmark's `eval_kernel_against_ref` jailed (its package imported without the LLM clients), both
  tolerances, static checker, torch.compile time; `run`: problems through `generate.kernels.search.ab` arms then `score`,
  fast_p per arm in `kernelbench.json`. Level 3 problems 48-49 need einops.
- `generate/kernels/tasks/flashinfer_bench.py` — FlashInfer-Bench as kernel tasks, researcher-only (2026-11-02): `lab kernels flashinfer-bench
  fetch [--definitions]|list|solution|score|run`; the HF dataset `flashinfer-ai/flashinfer-trace`
  at a pinned revision into `local/flashinfer_bench/<rev>/` laid out as its TraceSet (sha256 manifest; safetensors only
  for the definitions named). `fib:NAME` tasks: a shape is a workload (its variable axes + an ordinal), recorded tensors
  from its safetensors, the rest drawn as the benchmark draws them, each workload's scalars as 0-d CPU tensors handed
  to a program as numbers; reference, baseline and start program = the definition's `run`; the checker at half the
  benchmark's atol/rtol (which implies its rule); held out = `shapes.split`'s band over every variable axis; the eval
  config's `samples` is the dataset dir, mounted in the jail. Refused: no workloads, sampling, low-bit MoE, DSA (other
  evaluators), random scalars. `score`: the benchmark's own evaluator (flashinfer-bench at a pinned commit, checked)
  jailed on a TraceSet of the definition, the program as a Solution and the FlashInfer baseline: correct, the
  leaderboard's score (mean baseline/ours, 0 on any failure), speedup over the reference; `solution`: the Solution JSON
  the leaderboard takes. 95 of 190 definitions at the pinned revision are taken. Never run on a GPU yet.

### `data/` — corpus, workloads, calibration, knowledge (ledger, store, search)
- `data/corpus/` — frozen workload traces (BurstGPT timing, WildChat text), seen/heldout, hashed, `manifest.json`;
  `build_corpus.py` / `fetch_traces.py` rebuild it; any change is a new `corpus_version`. `lengths.py` (`lab corpus
  lengths --url URL`, decided 2026-10-25) writes each record's `expected_output_tokens` (and a multi-turn
  record's per-turn `expected_turn_output_tokens`) from a served reference, both splits, resuming from a log in
  `local/`; a new version whose manifest `prompts_version` stays the held-out guard's key (`Manifest.budget_key`); the
  provenance test rebuilds and applies them; run 2026-10-27 against the target's reference (vLLM 0.32.0 bf16, batch
  invariant, H200): version `7bfae7f7…`, 16,929 answers, p50 525-1,072 tokens per class, ~1% at the 2,048 cap; 300
  sampled jobs re-measured on the reference 2026-10-28 all came back the same length. The run's base cache is per corpus version. Since 2026-10-09 (version
  `131149…`) the held-out traces and the build seed (`data/corpus/.seed`, which reproduces them) are gitignored and live in
  the store (`lab store pull --corpus`); the previous version's held-out split was public in git.
- `data/knowledge/search.py` — the `search` tool (2026-10-15): one document per `hypothesis` record across every run, with its
  measurements (expect, actual, gap, GPU, config) and reflections, ranked by reciprocal-rank fusion of BM25 (English
  stopwords dropped) and cosine similarity of local `BAAI/bge-base-en-v1.5` embeddings (fastembed 0.9.0, ONNX on CPU,
  weights in `<ledger>/index/models`, vectors cached by content hash in `<ledger>/index/embeddings/`); researcher runs
  hidden from the agent; without fastembed it ranks by BM25 and says why. `LAB_TEST_REAL_EMBEDDINGS=1` runs the
  real-model test. It finds paraphrases ("tokens generated slowly" -> per-token latency) but not domain synonyms
  (fp8 -> quantization).
- `data/knowledge/` (was `store/`) — the data store (2026-10-09; bucket `bg-data`, credentials in `.env`: `LAB_STORE`, `LAB_STORE_ENDPOINT`,
  `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`). `remote.py` (`LAB_STORE` = `s3://<bucket>` with `LAB_STORE_ENDPOINT` for
  Cloudflare R2, or a directory; write-once keys), `sync.py` (`push`/`pull`: ledger as byte-range segments, blobs
  (once stored, a blob's trace.json is deleted locally, with its copies in `lab/runs/*/workspace/lab/runs/`),
  submit's held-out passive data, held-out traces + seed, `archive` for
  anything else; zstd, `heldout/` prefix for held-out material), `index.py` (DuckDB tables rebuilt from the ledger and
  blobs: records, regimes, heldout_aggregates, noise, client_rows, engine_rows, device_samples, kernels, ops;
  `all.duckdb` adds heldout_rows for the researcher; the agent's `agent.duckdb` never has them nor researcher runs'
  records, and is queried with
  external access off and configuration locked). Controller-only: VMs never hold store keys.
  CLI: `lab store push|pull [--blobs] [--corpus]|index|sql "..." [--agent]|archive PATH.. [--heldout]`.
- `data/workloads/` — frontier benchmark request data, researcher-only (never reaches the agent: nothing in `lab/` reads
  it), for manual runs: overfitting checks on data the agent never saw, the corpus's missing shapes, engine vs
  vLLM/SGLang on identical requests. One module per benchmark, docstring = source, sizes, output limit, MLPerf
  latency limits: `mlperf_llama3_1_8b` (CNN/DM 13,368), `mlperf_llama2_70b` (OpenOrca 24,576), `mlperf_mixtral_8x7b`
  (15k; GSM8K as 6-turn few-shot), `mlperf_deepseek_r1` (4,388 reasoning), `mlperf_gpt_oss_120b` (perf 6,396 + acc
  4,395), `sharegpt` (58,659 conversations). `lab workloads fetch all|NAME [--tokenizer]` →
  `local/workloads/<name>.jsonl` + `manifest.json` (source URL, md5, sha256, limits, MLPerf commit). Fetched on demand,
  no token: MLPerf files from the public `inference.mlcommons-storage.org` bucket (`metadata/<name>.uri` + `.md5`),
  ShareGPT from the Hub. Not re-hosted (GPQA asks not to be; ShareGPT's license is unclear). Chat templates are
  stripped. Llama-3.1-405B (404 on the bucket) and edge-agentic (tool-call transcripts) are not imported.
- `tests/test_*.py` — all synthetic, no GPU, each runnable with `python3 tests/<file>.py`, named after the module they cover; `test_correctness_run.py` is the full reference→candidate→verdict flow on fake servers; `test_kernels.py` also runs on MPS; `test_regimes.py` drives `tests/fake_engine.py` (fixed decode slots, run out of process) through every regime (~4 min in CI, where PRs run it only if they touch `measure/loadgen/`, `evals/researcher/canaries/`, `data/corpus/`, `data/corpus/__init__.py`, `targets/target.py`, its fake engine or the workflows; main pushes and the weekly run always do).
- `local/` is gitignored and local only: `local/tasks/` (correctness task sets), `local/equiv*/` and `local/h200-run-*/` (GPU run outputs and reports), `local/archive/research-loop/runs/` (the removed research loop's run panels and temp worktrees, Sept 2026), `local/notes/` (the user's notes). Live lab state is `lab/ledger/` and `lab/runs/`, also gitignored, kept in the store by `lab store push`.

## The engine repo (inference-server, after the 2026-10-06 split)
Holds only the engine and its stack: `src/` (engine, control plane), engine tests, `monitoring/`, and engine tools
(`scripts/bench/tune_triton_launch.py`, `scripts/gpu_tests/checks.py`, `scripts/tools/smoke_custom.py`). The
combined pre-split tree is tag `archive/pre-split` there. Its shim serves `prompt_logprobs` (contract in
`evals/agent/correctness/client.py`). Everything that runs, measures or judges it lives here; the user builds the agent's tools
themselves, so don't add agent tools without asking. Corpus classes for the regimes the
corpus lacks (short/long output, overload, shared prefix) belong in `data/corpus/build_corpus.py`; the regimes synthesize
them from existing records meanwhile.

## Decisions and runs
- `~/Desktop/BlameGraph-notes/DECISIONS.md` — private, not in git (decided 2026-11-04): append-only, one dated entry
  per decision (what, why, what it replaced), newest first. The eval conventions (2026-10-02) are there; change any
  decision only with the user. A new decision is added there, never to a file in this repo.
- GPU runs go into the ledger, not a file (decided 2026-10-18). A run made through the lab's tools records itself;
  a run made by hand (`lab regimes run --out DIR`, `lab correctness candidate --out DIR/equiv.json`) is
  imported with `lab import-runs --run manual-<date> --commit SHA [--env K=V ...] DIR...` (`bench` / `equiv`
  records, numbers only, visible to the agent), then `lab store push`. The runs before that
  (2026-10-04 to 2026-10-18) are in `RUNS.md` at tag `archive/docs`, with the old design docs (`ENVIRONMENT.md`,
  `lab/README.md`, `lab/RUNTIME.md`, `lab/TRACE.md`).

There is no status file: what works now is the code, the merged PRs and the ledger (`lab store sql`).

@~/Desktop/BlameGraph-notes/DECISIONS.md

The agent's tool set is the user's design: ask first. README stays a few one-line feature bullets, not a dev doc.
