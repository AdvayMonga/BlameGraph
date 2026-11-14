# The kernel search's method

One loop (`loop.py`) over three parts, each entered through one function. Every choice below is either copied from
the published method named, or ours with its reason. A change to any of them is built on a branch and merges only
when it wins `lab kernels ab` or a like-for-like run against this loop.

| Part | What it does | Source | Where |
|---|---|---|---|
| Loop | 6 generations over 8 parents. Each: one naming call names 24 changes, one review call writes the final list, one writer call per change writes it as SEARCH/REPLACE; every program that passes the checks is timed and joins the population; nothing stops it early | Ours: on three runs (2026-11-03/04) 45-65% of calls that each named and wrote one change were refused as repeats; one call that names the whole list does not repeat itself. Every program evaluated: OpenEvolve / AlphaEvolve | `loop.py`, `program.Config` |
| Context: naming and review | The task, how it is judged, the seen shapes and the GPU as facts; other libraries' implementations; every distinct idea of earlier runs on the task with what was measured (200 rows); every idea of this run with its measured gain; every parent with its timings, gates, ncu counters, compiler report, lineage and children; the start program. No advice | CudaForge: stateless calls, curated counters | `context.name`, `context.review` |
| Context: writer | The task facts, the one parent with its measurements, the change, and the programs the change takes from as full code | Ours: the writer needs the program, not the search | `context.write` |
| Diversity: parents | The best usable program, then a draw without replacement weighted by percentile score, from the best 32 | ShinkaEvolve: fitness-weighted parents from an archive | `diversity.pick` |
| Diversity: ideas | Named changes whose text is within 0.92 cosine of another in the list, of an idea of this run, or of an idea of an earlier run are shown to the review call with what was measured; it writes the same change once and adds up to 4 changes that combine named ones | Ours: embeddings find the close pairs, a model decides whether they are the same change and what combines | `diversity.Similar`, `context.review` |
| Diversity: repeats | A program identical to one already written (comments and whitespace aside) is refused; so is one whose change is within 0.95 cosine of a sibling's | ShinkaEvolve: novelty rejection | `diversity.Novelty` |
| Evals | Build check and correctness in parallel jails, then paired timing one program at a time; ncu counters when a program first becomes a parent | CUDA-L1: paired randomized timing | `evaluate.Evaluator`, `evals/kernels/` |
| Models | Opus 5.5 names and reviews (it may search the web and query the ledger); Sonnet 5.5 writes | CudaForge: a strong model for each change | `__main__.NAMER`, `__main__.WRITER`, `llm.py` |
| Predictions | Every change is named with a predicted speedup and counter changes; the next naming call reflects on each against what was measured, and one call after the last generation reflects on its ideas | Ours: the central loop's expect and reflect rule, kept as data for a later model | `context.name`, `context.reflect` |
| Records | One row per idea (plan, prediction, program, fate, gain, reflection) and every model call whole (prompt parts and answer, each text stored once), as blobs of the run's `kernels` record | Ours: every row rebuilds as prompt, answer, measured result | `loop.table_row`, `calls.py` |
| Promotion | The top 4 re-timed 3x, the highest median wins; a model reads each would-be winner for gaming | MANTIS: game detector | `evolve/run.retime`, `detect.py` |
| Generalization | Held-out shape regions, gated after the search, never in selection | AlphaEvolve: held-out split | `tasks/shapes.py`, `gate` |
