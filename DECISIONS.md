# Decisions

One entry per decision, newest first: what was decided, why (as recorded at the time), what it replaced. Append-only;
a reversed decision gets a new entry that names the old one. Change any of these only with the user.

## 2026-10-30 — The kernel search: 10 generations always, one call per change, every correct program measured
Three changes to `kernels.search`'s genetic search:
1. It runs exactly 10 generations after generation 0, and nothing stops it sooner. Removed: patience, shrinking a
   generation that improved little, the wall-clock cap, the GPU-seconds cap, the evals count, and the model budget
   (`--usd`; the user: there should be no budget flag).
2. Every program that builds and passes correctness takes the full exam and joins the population. Removed: the quick
   timing on one shape, racing at the cutoff, the 8 kept per generation, and the cut per idea type.
3. One call makes each change: it states the hypothesis and writes the SEARCH/REPLACE edit in one answer, as
   OpenEvolve's calls do. The call is shown the parent, the starting program and up to 2 better programs from other
   branches, all as full code, plus its slice of facts and the memory tables.

Added the same day (the user, from the list of published methods we had not used):
- the PTX comparison's cards are read by task name from `knowledge/ptx/<task>.txt` (no `--ptx`), without competitor
  timings (no target numbers in prompts);
- a program whose change from the starting program is within cosine 0.95 of one already written is refused before
  any GPU time (local embeddings of the changed lines; ShinkaEvolve's novelty rejection);
- competitors' kernels ported from source (`kernels.competitors.starts`) are judged at generation 0 beside the start
  and join the population.

Removed with change 3, as orphans: the named-ideas step, merging duplicate ideas (`--cheap-model`), the past-failure
check, the wide coder (`--wide-model`), repairs (`--max-repairs`), the strong coder's rewrite of kept programs, and
arm `search-gen20` (it varied `--keep`).

Why (the user, comparing with OpenEvolve on moe_gemm_fp8): OpenEvolve fully measured 130 programs from 140 calls in
2.5 h; our typed run fully measured 34 of about 100 correct programs from 206 calls, and stopped at the wall-clock cap
after 5 generations. Our calls never saw the starting program, only diffs against it, and the coder never saw the
judge's reasoning. "No cap" (the user) means no target number for the task, never a cut in the generation count.

Replaces, in part:
- 2026-10-23 (the funnel's tiers)
- 2026-10-26 (patience 3, the shrink rule, the fastest 8 to the full exam)
- 2026-10-27 (the wall-clock cap; the strong coder writing only the kept programs)

## 2026-10-29 — Four idea types per generation; the judge remembers every measured idea
Each generation after the first splits its 24 ideas over four types, each with its own judge calls: refine (new ideas on
the fitness-drawn parents, as before), persist (a line's last change taken further; every child that beat its parent
heads a line, which moves to a persist child that beats it and closes after 3 generations without one), fresh (a
different overall structure, half from the start program and half from a random program) and crossover (one parent
shown another's diff). The split starts at 8/6/6/4 and adapts each generation to each type's measured rate of new
bests (Laplace-smoothed, at least 2 each); each type gets its share of the 8 full exams, so a type is judged on its own
fastest, and every idea is tagged with its type. Every judge call also gets the memory as two cached prompt blocks: the
general table (ideas that beat their parent on other kernel tasks, best measured gain, the task and its bound type)
and the task table (every idea tried on this task in earlier runs and so far in this run, with how far it got and its
measured gains); it replaces the archive slice, which showed only the ideas most like one parent's to one call in five.
Why (the user): a search should explore new ideas, push the ones that show an improvement and combine branches in
every generation, and, like an expert who remembers thousands of experiments, know what was already tried rather than
start from square one and be deduplicated afterwards. The tables are measurements, not lessons (the no-beliefs rule,
2026-10-09); the model draws its own conclusions inside each call. The 2026-10-27 persistence arm bundled persistence
with an idea table, immigrants and an extra judge call, so its loss did not measure persistence alone; this costs no
extra model calls. Arm `search-untyped` keeps the search before this (refine only); `--crossover`/`search-crossover`
are replaced by the crossover type. Found the same day: no search so far passed `--archive`, so none read past ideas.

## 2026-10-28 — Everything from the InferenceBench era is removed
`logs/`, the InferenceBench diagnostics (`assertions`, `blame`, `claims_audit`, `exploration`, `inject`, `noise`),
`feedback/verdict.py`, their tests, and the `research-v1` tag are deleted; `python -m feedback` reads lab ledgers
only. Why (the user): none of it is used by the lab, and it is not BlameGraph's. The one helper the lab used
(`claims_audit._matches`) moved into `feedback/claim_facts.py`. Replaces the 2026-10-04 snapshot tag.

## 2026-10-28 — Minimal gates: a merge is bench + equiv; bench at a `min` tier; equiv is MLPerf's rule only
A change is merged on a seen-split `bench` and a passing dev-tier `equiv`; `submit` stays as built but is not
required (the agent's referee still asks for one). `bench` runs at a new tier `min` (15 s probes, 30 s final, still
at least 100 requests per probe), which submit, the base cache and `lab.validate`'s noise bands and performance
reference use too, and profiles the tree afterwards only when asked (`profile: true`; the run's baseline still
profiles). `equiv`'s candidate run gates on accuracy (>= 99% of the reference, pooled) and length (>= 90%) only: no
consistency probe and no teacher-forced divergence pass (that stays the `quick` tier's). The dev tier (1,597 items,
early stop at 200, 400, 800, ...) is kept: fewer items would leave a real 1% loss inconclusive. Why (the user): do
only as much as a merge needs for now; with one session at a time and no noise bands measured, a second held-out
bench adds little. Replaces, in part, 2026-10-18 (short gates: short-tier bench and submit), 2026-10-22 (every bench
followed by a profile) and the consistency gate of 2026-10-04. Consequence: the performance references in
`knowledge/reference/` were measured at `short`, so a `min` bench shows no `% of reference` (the field says why)
until `lab.validate` measures them again.

## 2026-10-27 — The engine comparison: one GPU, one client, matched precision, a win only beyond noise
`compare/` runs ours, vLLM 0.32.0, SGLang 0.5.22 and TensorRT-LLM 1.4.0rc1 (each project's newest release; official
containers) one at a time on one H200, at bf16 (weights, activations, KV) and fp8 (W8A8 + fp8 KV; the others load
Qwen's FP8 checkpoint, ours quantizes the experts of the bf16 one), each with its own documented settings at the same
limits (32k context, 1,024 sequences, 90% of HBM), prefix caching at each engine's default (on). The server gets cores
0-7, the client 8-15. Every configuration passes a launch check (answers 42, a fixed-length answer runs to its
length), runs the regimes on the seen split at a 2% search resolution, equiv against the same bf16 reference, and
cold start with emptied page cache and compile caches; repeats rotate the order. A regime's winner is clear only when
its lead exceeds every contender's run-to-run spread and, for a load search, the runner-up failed at a load the
winner passed (`compare/report.py`); otherwise the report says why not. Three changes to the measurement itself:
a corpus record reused within one schedule is salted ("[k] " before its first message), because a pool of 60
long prompts at 80 req/s let prefix caching carry the long-prompt result; the closed-loop client splits over
processes and a closed-loop run is invalid when a client process ran above 80% CPU (closed loop has no send schedule
to lag behind, so this is its bottleneck signal); `python -m regimes.clientcheck` measures the client's own TPOT error
against a server that logs its true send times, run first on the comparison's VM. Why (the user): a fair eval against
vLLM, SGLang and TensorRT-LLM with every other variable controlled, and a winner on every regime; the 10-19 comparison
ran vLLM in bf16 against our fp8.

## 2026-10-27 — No ties at the search's edge: bursty scales its request count, searches run to a break point
Bursty's load multiple now means that many times the trace's requests over the trace's own span (`W.scaled`: each
arrival joined by factor - 1 more, spread over the gap to the next, drawn from the trace), not the same requests
replayed faster. The load searches run until a probe fails (40 probes, was 10; saturated up to 4,096 concurrency, was
512), and every goodput result carries where the search ended: the lowest failing load (`fails_at`) and why it
stopped (`stopped`: resolved, invalid, below_min, probe_budget; saturated: max_concurrency). `regimes run --tol` sets
the search's resolution for an engine comparison. Why (the user: never accept a tie unless a physical ceiling is hit):
the short tier's spike window holds 158 requests, so at 256x they all arrived within 0.7 s and any engine that could
batch 158 passed at every speed; 512x was voided by the client's own send lag (32-51 ms), not by an engine. Long
prompt (80) and shared prefix (0.81) were real break points that both engines reported alike because the search
stops at a 10% bracket (80 passed, 86.7 failed); the bracket now shows when two results are within it. Replaces the
time-compressed replay (2026-10-02 eval conventions' bursty).
## 2026-10-27 — Every program is tuned before it is measured
`kernels.search` grid-searches a program's declared launch parameters (`CONFIG` + `TUNABLES`, 8 points) before each
first measurement of its speed: on the most-called shape before the quick timing, on every seen shape before the full
exam (with a per-size dispatcher where different points win in different regions), and for the root program. Each
point is loaded with CONFIG rewritten, checked on one shape and timed in the evaluator's own jailed stage; the program
is measured and kept with the fastest point written in, so its children start from it. Re-timing and the gate
measure the tuned program as it is; every point's score is shown to the model as a fact. Why (the user: the search space must be unrestricted, nothing that is a lever for
speed fixed): programs were ranked at the tile sizes the model guessed, and tuning ran only on the top 4 after the
search (off in the 2026-10-27 race), while OpenEvolve's winner autotuned inside its own program; structure goes to
the model, numeric knobs to a measured grid. Replaces post-search tuning (2026-10-18, "the top 4 are tuned"); the A/B
promotion (`ab.SAME_PROMOTION`) no longer sets tuning, which is now part of the search.

## 2026-10-27 — No roofline stop: a search runs to its budget
`kernels.search` no longer stops when the best program reaches a share of its roofline (`--stop-pct`, 80 by
default, removed); it stops at its generations, budget, wall-clock or patience. The % of roofline stays a fact in
every prompt and record. Why (the user): within a session the search should go as high as it can; the fixed
generations already bound it, and the % of roofline is what tells the central agent whether another round on the same
kernel is worth it. In the 2026-10-27 race on moe_gemm_fp8 the rule ended the generations arm at 83% (1.194x seen,
1.391x held-out, 33 min) and the ideas arm at 81%, while OpenEvolve, with no such rule, ran its 2.5 h to 87% (1.250x,
1.478x held-out). Replaces MANTIS's stop rule (2026-10-18, METHOD.md row 13); patience stays.

## 2026-10-27 — Searches capped by wall-clock; K-Search's persistence tested and not adopted
`--wall-s` caps a kernel search by wall-clock. Why: the 2,700-s `--gpu-s` cap stopped the first real generations run
(2026-10-27) after 2 of 10 generations because it sums parallel jobs and the no-GPU precheck. Tested the same day and
removed: an `--ideas` arm after K-Search (Berkeley Sky Lab, arXiv 2602.19128), which kept an idea table so the 4 best
ideas by measured gain were refined again each generation (an idea closed after 3 generations without a better
measurement), with a judge call closing ideas and adding sub-ideas and 4 immigrant ideas of whole-program changes per
generation; and K-Search itself as an arm on our evaluator. On moe_gemm_fp8 (same shapes, start, evaluator, every role
on Opus 5.5, 2.5 h cap): generations 1.196x seen / 1.391x held-out for $6.95; ideas 1.162x / 1.376x for $14.70;
K-Search 1.158x / 1.267x for $29.92; OpenEvolve 1.250x / 1.478x for $78.67 (both of ours stopped at the roofline rule,
removed the same day). The user dropped both arms; one run per arm, one task.

## 2026-10-27 — The beam is removed
The beam (each MCTS expansion: the judge names 4 changes, the coder writes all 4, each judged in full, a failed child
repaired in correction mode up to 3 deep) and its A/B arm `search-beam`, `--no-funnel` and `--beam` are deleted;
`kernels.search` with a judge always runs funnel rounds, and without one the coder alone proposes children (tests).
Why (the user): it is no longer the main path and nothing measures it any more; the funnel beat it on 2026-10-24
(moe_gemm_fp8 1.161x to 1.026x seen, 1.284x to 1.096x held-out). Its last code is on main before this change (git
history), its records stay in the ledger. Revisit only to rerun that A/B. Replaces "the beam becomes an arm"
(2026-10-25).

## 2026-10-27 — The funnel's model calls run in parallel; the strong coder writes only the programs kept
Four changes to every funnel round, none to what a round decides. (1) Model calls (judge calls, merges, drafts,
repairs, rewrites) run 8 at a time (`--workers`), the first of each batch alone so the rest read its prompt cache;
GPU timings stay one at a time. (2) The strong coder (Opus) rewrites only the programs that pass the quick timing's
cut, not every correct draft, and a rewrite replaces its draft only when it passes stage 1 and times faster; the
drafts stay the wide coder's (Sonnet), repairs the strong coder's. (3) Prompts are cached: the task's setting and a
parent's program and measurements are sent as cached blocks before the per-call part (`llm.CACHE_SPLIT`). (4) ncu
counters are taken when a program is first chosen as a parent, not at its full exam: they feed only its children's
prompts, so a child's predicted counter changes are measured only when its parent had them. Why (the user, after the
first real generations run sat ~1 h in generation 0): the checks took ~10 of its 57 minutes; the rest was ~80 serial
Opus calls, as the frontier systems avoid by sampling in parallel (AlphaEvolve, OpenEvolve, ShinkaEvolve). Replaces
the strong coder rewriting every survivor (2026-10-23) and ncu at every full exam (2026-10-17).


## 2026-10-27 — Every profiler's prediction is measured: numbers for trace and hostprof
`trace` and `hostprof` records get a `calibrated` block like `profile`'s and `kernel`'s (nsys: kernel, memory-copy and
CUDA API time, top kernels, kernel time's share of the profiled window; py-spy: samples per thread, top functions by
self and total share), and a profiler's `expect.metric` is a dotted path into its result, as a mechanism's already
was. Why (the user, choosing this over making `expect` optional for them): every measured tool requires an `expect`,
but the main prediction was looked up only in `result.metrics`, which no profiler record has, so a prediction on
`profile`, `trace`, `kernel` or `hostprof` never had a measured value; trace and hostprof had no numbers to predict.
Amends 2026-10-11.

## 2026-10-26 — The kernel search runs generations: funnel rounds from fitness-drawn parents
`kernels.search` becomes a genetic search inside one call (`kernels/search/generations.py`): generation 0 is the funnel
from the start program (40 ideas, the fastest 12 to the full exam); each later generation draws 4 parents from the
population (the 32 best open programs; the best always, the rest by percentile reward + 0.05, without replacement),
gives each 6 ideas (one judge call on one slice of facts, the slices rotated across the parents), and sends the
fastest 8 of the 24 to the full exam, which join the tree under their own parent; up to 10 generations, stopping at the
roofline or after 3 generations with no new best; a generation where under 1/5 of the examined beat their parent makes
the next 2/3 as large. Hypothesis calls go 3 Opus : 1 Sonnet; the hypothesis stays free text (bottleneck, idea, how to
implement it, predicted counter changes). The tiers are unchanged (precheck with one repair, dedup, two-shape
correctness, one-shape quick timing, full exam); programs within 5% of the quick cutoff are timed once more and ranked
by their mean, and every generation logs the Spearman between the quick timing and the full exam. Every prompt also
gets the bigger picture as facts: the regimes the engine is scored under and the kernel's share of the latest profiled
window. Why (the user, after the 2026-10-26 research): the funnel had become one wide round per expansion, and every
tree of the 2026-10-24/25 runs stayed one level deep (root + 6 children; round 2 always ran out of model budget before
timing anything); the research says many small generations beat one wide one (Kevin's 16 x 8 at 1.10x against 128 x 1
at 0.65x; ShinkaEvolve's archive of 40 under 150 evaluations), soft fitness-weighted parents keep diversity, dedup
before GPU time is the largest diversity gain, and a cheap tier that is not a measurement does not help. The MCTS round
stays as arm `search-mcts`, the beam as `search-beam`; `search-gen20` (20 x 12 kept) and `search-crossover` (10% of a
generation's ideas from a judge call shown another parent's diff) are the arms to run. ~$50 of model calls and ~2 GPU
hours per 10-generation call, about twice the 2026-10-25 round. Replaces one funnel round per MCTS expansion
(2026-10-25) as the default.
## 2026-10-25 — Fixed answer lengths: each request runs to the reference's own answer length
Once the corpus carries each record's `expected_output_tokens` (the reference's greedy answer length, measured by
`python -m corpus.lengths` on both splits: a new corpus version), every load request is sent with `ignore_eos` at that
length (the regime's own `max_tokens` where it names one) and `skip_special_tokens: false`; an answer that ends
earlier is a truncation, so the run is invalid. inference-server takes both fields since #134. Why (the user, after
looking at InferenceX, which fixes lengths the same way): a change that shifted answer lengths moved the speed
numbers without the engine getting faster or slower, which added to the noise; with fixed lengths a base and a
candidate do the same work. The length is the reference's own answer (the user's choice over BurstGPT's recorded
ChatGPT lengths or one cap for all), so the work is what this model really does; special tokens stay in the text so
the client's re-count stays exact. Natural stopping stays judged by `equiv` (length ratio, consistency). Once the
lengths are filled, the noise bands and the performance reference are measured again (`lab.validate`); a band now
counts only on the corpus version it was measured on, so an old band is not applied to fixed-length runs, and a run's
cached base is kept per corpus version. The lengths change no prompt, so the held-out guard keeps its budget: it is
keyed to the version before lengths (`prompts_version`), not to the new one. Shared-prefix turns each get their own
measured length; the `profile` tool's requests also run to their max_tokens. Replaces natural-length answers under
a 2,048 cap.

## 2026-10-25 — The funnel is kernels.search's default; the beam becomes an arm
`kernels.search` runs one funnel round per expansion by default (`--no-funnel`: the beam, A/B arm `search-beam`);
the `kernel_search` tool gets it through `kernels/pipeline.py`. Why: it won the A/B the 2026-10-23 decision set as
its bar. On moe_gemm_fp8 (same shapes, start, evaluator and 2,700 GPU-s cap, one run per arm) it reached 1.161x on
seen shapes and 1.284x held-out against the beam's 1.026x and 1.096x, using 1,188 GPU-s against 1,701 and $14.25
against $12.32; both passed the held-out gate. The user called it the clear winner; a second task or seed has not been
run. Replaces the beam as the default (2026-10-20).

## 2026-10-23 — The agent calls the kernel layer as a tool, blocking for now, and gets a patch
`kernel_search` (task, budget_usd, with a kernels-layer hypothesis and its expect) runs the kernel layer end to end:
capture under real traffic, search from the agent's own version of the kernel, the held-out gate, and a patch of the
engine module saved in the workspace for the agent to apply and judge with its own bench/equiv/submit. Why (the user):
the agent decides which kernel to work on, given each task's share of time as a fact; the agent applying the patch
keeps it in charge and avoids clashing with its own edits. A blocking call on the run's GPU for now; a parallel job on
its own host later. The search runs on the controller (the model's key stays there) with its evaluator on the worker.

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
