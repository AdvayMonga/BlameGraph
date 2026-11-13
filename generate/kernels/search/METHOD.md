# The kernel search's method

One loop (`loop.py`) over three parts, each entered through one function. Every choice below is either copied from
the published method named, or ours with its reason. A change to any of them is built on a branch and merges only
when it wins `lab kernels ab` or a like-for-like run against this loop.

| Part | What it does | Source | Where |
|---|---|---|---|
| Loop | 10 generations of 24 model calls over 8 parents; one call names one change and writes it as SEARCH/REPLACE; every program that passes the checks is timed and joins the population; nothing stops it early | OpenEvolve / AlphaEvolve: one call per change, every program evaluated; Kevin: 16 x 8 serial refinement beat 128 x 1 | `loop.py`, `program.Config` |
| Context | Every call reads: the task, how it is judged, the seen shapes and the GPU as facts; other libraries' implementations; every idea of the run with its measured gain; the parent with its timings, gates, ncu counters and compiler report; the start program; the 2 best programs off the parent's lineage as full code. No advice | CudaForge: stateless calls, curated counters; CodeEvolve: 2-3 inspirations | `context.build` |
| Diversity: parents | The best usable program, then a draw without replacement weighted by percentile score, from the best 32 | ShinkaEvolve: fitness-weighted parents from an archive | `diversity.pick` |
| Diversity: repeats | A program identical to one already written (comments and whitespace aside) is refused; so is one whose change is within 0.95 cosine of a sibling's; calls on one parent are shown what earlier calls proposed for it this generation | ShinkaEvolve: novelty rejection | `diversity.Novelty`, `context.siblings` |
| Evals | Build check and correctness in parallel jails, then paired timing one program at a time; ncu counters when a program first becomes a parent | CUDA-L1: paired randomized timing | `evaluate.Evaluator`, `evals/kernels/` |
| Models | Sonnet 5.5, with Opus 5.5 every fourth call; each may search the web and query the ledger | CudaForge: a strong model for each change | `__main__.MODELS`, `llm.py` |
| Promotion | The top 4 re-timed 3x, the highest median wins; a model reads each would-be winner for gaming | MANTIS: game detector | `evolve/run.retime`, `detect.py` |
| Generalization | Held-out shape regions, gated after the search, never in selection | AlphaEvolve: held-out split | `tasks/shapes.py`, `gate` |
