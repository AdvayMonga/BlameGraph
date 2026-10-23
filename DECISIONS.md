# Decisions

One entry per decision, newest first: what was decided, why (as recorded at the time), what it replaced. Append-only;
a reversed decision gets a new entry that names the old one. Change any of these only with the user.



## 2026-10-23 — The funnel: volume through measured tiers; an idea archive; facts in slices
Each expansion becomes one round of a funnel: several judge calls, each shown one slice of the facts (the
measurements; similar archived ideas and their outcomes; measured winners elsewhere in the tree; the compiler's report;
a PTX comparison of other implementations) naming ~20 hypotheses each; same-idea clusters merged; ideas like archived
failures kept only when the judge names what differs now; a cheaper wide coder for all, correctness in parallel, the
strong coder repairing failures and rewriting survivors, a one-shape quick timing, the fastest 6 to the full exam.
Every distinct idea and its outcomes are kept in an archive (measurements only). Why (the user): generate far more and
narrow in tiers, but every cheap tier a measurement (a model ranking idea text did no better than random, 2026-10-20);
keep good ideas that lost in one context; one prompt carrying everything dilutes attention. Kernel ideas predict their
mechanism (ncu counter changes). Multiple rounds stay. It replaces the beam only if it wins the A/B (arm
`search-funnel`).
## 2026-10-23 — Regions replace hand-written kernel tasks; a region's error budget is measured
The kernel layer's unit of work becomes a region: any run of consecutive decode operations (stations), from one kernel
to the whole step, with its harness generated from the engine's declared stations (inputs, the stations as baseline,
an fp32 run as reference). Why (the user): a task was only facts plus a test harness, and hand-drawn task boundaries
made fusion across kernels unreachable; the megakernel's 4.1 vs 5.6 ms at batch 1 came from fusion alone. Today's 10
tasks become the first single-station regions once the engine has stations. A region's output may be at most 2x as
wrong as the engine's own code against the fp32 run (k = 2 over k = 1, leaving room for lower precision inside),
never stricter than the dtype's tolerance; a winner above the baseline's error gets the quick equiv; MLPerf's rule
stays the end-to-end judge. Captures run under the regimes the task scores (equal weight per regime, call-weighted
within) and are reused until the engine outside the region, the launch settings or the traffic change.

## 2026-10-23 — A live dashboard that reads files; no Prometheus/Grafana
The engine's Prometheus + Grafana stack is removed (inference-server #129; its `/metrics` endpoint stays). Instead
`lab/dashboard.py` shows a run live from the ledger and a progress file the lab appends to (`lab/live.py`). Why (the
user): anything that scrapes the engine during a measurement can move the numbers; the lab already keeps every
request's client and engine rows, the step timeline and device samples, so the stack only added a live view, which
the lab's own files give for free. Researcher-only; human-in-the-loop intervention is out of scope.

## 2026-10-22 — Every bench is followed by a profile of the same tree
A completed `bench` (and the run's baseline) profiles the measured tree once, right after the measurement and in its
own launch, with the `profile` tool's defaults, and keeps it as `profiled` with its layer facts. Why (the user): an
experiment is checked against how it was meant to work, so each measurement needs fresh profiling, and the layer and
region facts stay current without a separate call. Not during the timed window, which profiling would slow; not in the
same launch either, because the profiler runs the engine in process on its own workload: the general config and the
profile tool's default 8 requests x 32 tokens, so its facts describe that load, not the regimes'. Costs 1-7 GPU
minutes per bench, capped at 15; another process on the GPU marks it contaminated.

## 2026-10-22 — Hypotheses in batches, each tagged with its layer; bundles allowed
Several hypotheses may be open at once, each naming its layer (kernels, model_execution, runtime, serving), and a
measured call may measure several together (`bundle`) when the agent states why they do not affect each other. A
hypothesis is measured again only after a reflection on its last measurement, and a new hypothesis is filed only once
every measurement has one: a batch is reflected on before the next. Why (the user): the kernel loop's many-ideas
funnel is the shape for the whole loop; each hypothesis goes to its layer's machinery; separate measurements are
preferred and bundles obscure causality, so the bundle and its stated independence are on the record, and its
mechanism predictions check each part. Amends 2026-10-11 (the next measured call waited for the last one's reflection).

## 2026-10-22 — A hypothesis predicts its mechanism, not only its outcome
`expect` may carry `mechanism`: predictions on other numbers of the same result (e.g. the GPU idle share in a
profile's layer facts), each kept with its measured value and gap. Why (the user): an improvement should be checked
against how it was meant to work, not taken blindly; outcome and mechanism together say whether a change worked as
intended, improved for another reason, or moved its mechanism without moving the outcome. The records are measurements,
so they fit the no-beliefs rule (2026-10-09), and they are the labels process evals and a later trained model need.
Extends 2026-10-11.

## 2026-10-20 — The judge names 4 changes and all 4 are measured: no screener, no merge hypothesis
Reverses part of "The layer-1 loop copies the frontier per component" (2026-10-18): the 16 hypotheses ranked by a
screener (Haiku 4.5; MANTIS ROI and random as arms), the random slot for a learned screener's labels, and the merge
hypothesis are removed. Why (the user, on the review's numbers): ranked picks beat their parent 60% of the time and
random ones 69% (81 vs 29 measured, 2026-10-18/19); 12 of every 16 named changes were thrown away; the merge reached
a new best 2 times in 17. Inspirations (other branches' best programs shown to the judge) stay.

## 2026-10-18 — Kernel candidates may be CUDA C++; the top 4 are tuned
A layer-1 program may build CUDA C++ at import (`torch.utils.cpp_extension.load_inline`) beside Triton, the model's
choice; the build is cached per program source. Why: every frontier kernel system with the strongest numbers writes
CUDA (CudaForge, KernelPro, MANTIS), Triton cannot reach warp specialization, TMA or wgmma, and the engine's own
decode megakernel is CUDA. Rust was considered and not taken: no GPU kernel toolchain in use. Accepted risk: C++ in
the evaluator's process can do what ctypes can (banned in Python); correctness stays guarded by outputs verified in a
separate process; faked timing is guarded by the C++ screen (Python C-API, dynamic loading, syscalls, threads,
streams, graphs, timer events), the roofline physics alarm and the gaming detector. Tuning moves from the winner
only to each of the top 4 re-timed programs (8 grid points each; KernelEvolve tunes every candidate). The judge and
coder stay on the metered API (the user, 2026-10-18).

## 2026-10-18 — Short gates: bench, equiv and submit sized by significance
The user asked for the least data that keeps each gate statistically meaningful. Bench defaults to the general regime
(saturated) plus the task's objective and constrained regimes when a task names them (was single_stream, saturated,
bursty without a task). Load searches start where the run's base measurement of that regime ended and stop once the
bracket is within the regime's noise band (10% where none is measured; was always 10%, from a cold start). Submit
measures the held-out split at the short tier with the same searches (was full: 600 s finals, and a band calibrated at
short applied to full-tier numbers). Equiv answers items in one fixed order and looks at 200, 400, 800, ... of them,
stopping once the pooled ratio's CI is above the line or the net loss is significant (both Bonferroni-corrected over
the looks); otherwise it runs the tier's whole set and the rule is unchanged. The order is the same for every
candidate, so a rerun cannot draw a new sample. The Thresholdout guard stays. Replaces, in part, the eval conventions'
full-tier submit (2026-10-02).

## 2026-10-18 — Noise bands: one calibration per user, any engine commit
Reverses "Noise bands belong to one engine commit" (same day): the latest band measured for the target counts at
any engine commit. Why: the user wants calibration as a one-time setup per user, not a re-run after every engine
change. `lab.validate` still records the commit a band was measured at.

## 2026-10-18 — The layer-1 loop copies the frontier per component (kernels/METHOD.md)
Every component copies the published method with the strongest numbers, or is ours with a reason and an A/B arm
(the user, 2026-10-17). Adopted: MCTS parents (KernelPro), 16 hypotheses ranked then a beam of 4 measured (MANTIS,
Kevin), chained repairs (CudaForge, KernelPro), MANTIS's stop rule, two-shape ncu facts and per-shape bound type, a
gaming detector at promotion (MANTIS; its library-only filter not copied: a faster library call is a real engine
gain). Ours, measured by the A/B: a System-1 screener over hypotheses, Haiku 4.5 now (Jev and OpenAI's Decisions API
are hosted-only and cannot be fine-tuned), a fine-tuned local classifier once one random pick per beam has labelled a
few hundred; the judge may use web search and read-only ledger SQL, its own choice when, with no research pasted by
the harness (the facts-only line: the environment hands over facts; what the model chooses to read is its own, as in
the agent's workbench); a merge hypothesis competing for the beam.
## 2026-10-18 — The per-layer facts the agent chooses a layer by
From each `profile` bundle (`lab/layers.py`): the four layers' shares of the profiled window (kernel time; prefill +
decode phase time beyond it; admit + evict; time outside steps) and a distance where one is defined (the decode
step against its floor; kernels against their roofline when gated). Layers 3 and 4 get the facts a change there
moves (GPU idle share, batch fill, admissions blocked) and say they have no single ceiling, rather than a made-up
one. Why: 2026-10-14 decided the agent picks the layer and is given these numbers, never a recommendation.

## 2026-10-18 — A scheduler simulator for layer 3: the real scheduler on a fake clock
`sim/scheduler.py` runs the engine's own scheduler code one step at a time (a `step()` seam, inference-server
#120) with a fake clock and a backend that charges a cost model instead of running the model. Why: a scheduling
change costs 30+ GPU minutes to try for real; this tries it in seconds on a CPU, on the regimes' own traffic. It
screens; `bench`/`submit` judge. Its bar before the agent gets it as a tool (the user: a `simulate` tool, interface
theirs): it must rank changes the way the real bench does (Spearman ≥ 0.8 over several known changes), not match
milliseconds. The cost model is fitted from step telemetry of the first GPU run (phase 2); until then its defaults
are stand-ins and every report says so.

## 2026-10-18 — Synthetic traffic is generated per run, never kept
`regimes/synth.py` generates each run's traffic fresh from the public corpus's seen split (whole real sessions,
the real timing pattern jittered, real text recombined) and keeps nothing: the seed in the run's config regenerates
it. Why (the user): no second corpus to maintain or leak; the public data stays the credible source and is what the
traffic is synthesized from. Interchangeable with the corpus for bench/submit once a GPU run shows the two agree
within the noise band; until then it is a source for experiments and the simulator.

## 2026-10-18 — Noise bands belong to one engine commit
`lab.validate` records the engine commit a band was measured at (None for a tree with uncommitted changes), and
`submit`, experiment records and the brief's harness facts use a band only at the run's base commit; otherwise it is
None and submit says `unknown_band`. Why: a changed engine can be more or less jittery, and a band from an older
engine was being applied silently. Consequence: after the engine moves, re-run `lab.validate`'s noise step
(minutes per regime on the GPU) before submits on the new base mean anything.

## 2026-10-17 — Our own layer-1 generator; OpenEvolve becomes the baseline
`kernels/search/` replaces the `openevolve` package as the generator, taking from what the published systems learned:
refinement chains in a tree (Kevin, STARK, KernelPro, Kernel Forge), a strong-model judge reading curated facts beside a
coder (CudaForge, KForge, Astra), the model writing structure while a grid search picks tile sizes (KernelEvolve,
AlphaEvolve; also our 2026-10-16 moe_gemm win, mostly tile sizes and stages), stop rules near speed-of-light (Hari),
duplicate refusal before GPU time (ShinkaEvolve). It is proved against OpenEvolve on the same tasks, evaluator and
budget. Choices: Opus 5.5 for both judge and coder (a weak coder cost CudaForge 0.8x; model spend is small next to GPU
time); the judge's text is kept per candidate in the run's trace blob, not inline in the ledger record; the judge sees
a curated set of ncu counters, starting from CudaForge's 24 (a view of facts, not advice). The evaluator, tasks,
gate, ship and records are unchanged.
## 2026-10-17 — The end-to-end judge runs once several shipped kernels stack
Shipped kernels are stacked on one engine branch (each `kernels.evolve ship --base` the previous shipped branch; the
first is `kernels/moe_gemm-20261016-154523`), and `bench` + `equiv` + `submit` run once, on the stack, when its gains
can add up past the regimes' noise bands. Why: one kernel's gain (moe_gemm 1.09x on 13.7% of the decode window, ≤ ~1.2%
end to end) sits inside the bands, so judging it alone costs hours of GPU for a within-band verdict. Each kernel still
passes its own held-out gate before it ships; a failing stack is bisected with `equiv`.

## 2026-10-18 — Manual measurements enter the ledger as numbers, visible to the agent
A run measured by hand (`python -m regimes run`, `python -m correctness candidate` on a branch) is imported as a
`bench` / `equiv` record under a `manual-<date>` run (`python -m lab.import_runs`): the engine commit and launch env
as its config, the numbers as its result, no text. The agent sees it like any measurement (`ledger`, `query`,
`search` index the numbers; there is no hypothesis to search). Why: four days of engine sessions had their results
only in `RUNS.md` prose and `data/` files, where nothing compounds; the 2026-10-15 rule hid a person's runs because
their *reasoning* would reach the agent — numbers alone carry no judgment. `RUNS.md` is frozen as the history of
the runs before this; new runs are not added to it (the ledger is the record, `python -m store sql` the view). The
researcher's reasoning stays out of the ledger. Amends 2026-10-15 for manual runs and 2026-10-16 (`RUNS.md`).

## 2026-10-16 — No status file
`STATE.md` is removed; the record is `DECISIONS.md` and `RUNS.md`, both append-only. Why: a rewritten-in-place
status file went stale between sessions (it still listed merged engine PRs as open). Replaces the three-file entry of
2026-10-15.

## 2026-10-15 — Kernel-level precision: no global switches, precision inside a kernel allowed
A candidate kernel may not change process-wide numerics (e.g. `torch.backends.cuda.matmul.allow_tf32 = True`): the
source screen refuses it (`kernels/evolve/core.screen`). Lower precision inside the kernel (e.g. FP8 math) is allowed
when the output passes the task's tolerance; its max error against the reference, beside the baseline's on the same
inputs, is reported as a fact (`error_vs_reference`), and `equiv` stays the final judge. Why: a global switch also
changes the reference and the baseline the evaluator compares against, and every other layer of the engine, so it is
an exploit of the evaluator, not an optimization; quantization inside a kernel is what the 2026-10-04 rule allows.

## 2026-10-15 — Submit takes a dev-tier equiv for the example engine; a quick equiv tier
Equiv's time is the engine generating answers: ~95 min at tier dev on inference-server, so the full tier (~11.5M
output tokens) would be ~8-17 h. To get a first working run, the target names the tier submit requires
(`[correctness] submit_tier`, default full; `dev` in `targets/inference-server.toml`): ±0.7% precision instead of
±0.2%. Back to full once the engine is fast. Equiv sends 32 requests at once (was 16, #43). Equiv's new tier `quick`
scores the reference's answers teacher-forced (no generation) and returns divergence facts with no verdict, so a
change can be checked in minutes while iterating. Why: the user wants one working end-to-end product before
improvements.

## 2026-10-15 — A `search` tool over experiments, with embeddings from day one
The agent gets `search`: every run's hypotheses with their measurements and reflections, ranked by BM25 fused with
local embeddings (`lab/search.py`). Why: the user wanted the better-recall version from the start rather than
keyword-only. The model runs locally (no key, nothing sent out) and is pinned, so rankings are reproducible.
Kernel-archive entries and research claims are not in it.

## 2026-10-15 — Every experiment keeps its reasoning, a person's ones included, hidden from the agent
A session that ends with its last measurement unreflected gets one short reflection-only call (tools `reflect` and
`ledger`, that record's facts as the prompt). Changes made outside the lab (e.g. inference-server #106, #108) are
recorded with `python -m lab.experiment` as `researcher-` runs: hypothesis, expect, the lab's own measurement, the
gap and a reflection. No agent view shows those runs. Why: every experiment should carry its hypothesis, rationale,
prediction and post-analysis; a person's text reaching the agent would be human judgment (2026-10-09). The kernel
loop's candidates still carry no reasoning (offered, not chosen).

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
layer's share of time × distance from the ceiling as a fact. Layer 1 first.

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
