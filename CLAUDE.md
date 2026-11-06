# BlameGraph

The environment for autoresearch loops that optimize LLM inference: an agent loop with its tools, sandbox and GPU
VMs; the data it reads; the load and correctness measurements; the referee; and the kernel generator. The engine under
optimization is a separate repo, github.com/AdvayMonga/inference-server, found via `LAB_ENGINE_REPO` (default
`../inference-server`) and run with its own `.venv/bin/python`. Coding guidelines: `../CLAUDE.md`.

## Rules
- **Facts only to the agent, never heuristics or advice.** Each session it gets its objective (generated from
  `lab/prompts/objective.md`), the brief (task, budget, referee verdict with evidence, baseline, hypotheses), the raw
  ledger, and the facts it cannot derive from it (`lab/evaltools.harness_facts`). Anything derived from the records
  is for the researcher (`lab feedback`), never the agent. No knowledge base of beliefs, hand-written or the agent's.
- **The agent's tool set is the user's design.** Ask before adding or changing a tool.
- **No measured number without its reasoning:** a measured tool needs an open hypothesis and an `expect`; the next
  measurement of it waits for a reflection on the last (`lab/experiments.py`).
- **The GPU:** nothing is measured while another process holds it; a VM is never left running (`lab vm stop`).
- **Engine changes** (performance work) belong in inference-server; everything that runs, measures or judges it is here.
- **Decisions** live only in `~/Desktop/BlameGraph-notes/DECISIONS.md` (private, not in git). Read the entries that
  cover what you are changing; change a decision only with the user; add new ones there, never to this repo.
- **Runs go in the ledger**, never a file: tool runs record themselves; a hand-made run is imported (`lab import-runs
  --run manual-<date> --commit SHA DIR...`), then `lab store push`. There is no status file: the code, merged PRs and
  `lab store sql` are the state.
- `README.md` stays a few one-line feature bullets.

## Running the loop
One command: `lab <command> [args]` (`pip install -e .` puts it on PATH; else `bin/lab`, this repo's `.venv` python
if present, else `python3`). `lab` lists every command, `lab <command> -h` its arguments; the table is `lab/cli.py`, where a new module
with a CLI gets its entry. The agent's tools are the same commands by the same names, run by a person on a researcher
run. Jailed and remote internals keep `python -m`.
1. Once per target and GPU: `lab store pull --corpus`; the reference's answers (`lab correctness reference --url URL
   --out local/equiv/ref`); `lab validate --target T` (launch checks, noise bands, performance reference).
2. A GPU: `lab vm start`, `lab vm setup` (also measures the GPU's peaks); `LAB_WORKER=ssh` runs measured jobs there.
3. A run: `lab session --task task.toml --budget USD` (or `lab vm run --env -- lab session ...`); `lab dashboard`
   shows it live. No GPU: `lab rehearsal setup DIR` and the commands it prints.
4. After it: `lab feedback local/ledger --run RUN`, `lab store push`, `lab vm stop`.
5. A person's change under the agent's rules: `lab start --base REF --change REF --budget USD` (prints the run id),
   `lab baseline --run RUN`, `lab hypothesis --run RUN ...`, a measured tool (`lab bench|equiv|submit|profile|trace|
   kernel|hostprof|kernel-search --run RUN --hypothesis-id H --expect JSON`), then `lab reflect --run RUN ...`.
A merge of an engine change is a seen-split `bench` and a passing `equiv`; `submit` reads the held-out split through
the guard's query budget.

## Layout
- `lab/` — the loop: `session`, `agent` (+ `agent_openai`, `apiproxy`), `tools`/`evaltools`/`proftools`,
  `experiments` (hypothesis/expect/reflection rules), `experiment` (a person's run), `task`, `workspace`, `budget`,
  `rehearsal`, `cli`, `prompts/`.
- `targets/` — the target spec (`<name>.toml`, `target.py`, selected by `LAB_TARGET`) and `engine.py`.
- `measure/` — `runner/` (worker, serve, launch profiles, passive artifacts, trace contract), `cloud/` (vm, providers,
  gpucheck), `sandbox/` (jail, container, write surfaces, grader, dependency build), `profile/` (profilers, capture),
  `loadgen/` (the regimes: client, runner, suite, workload, synth, arrivals, clientcheck), `sim/` (scheduler
  simulator), `context/` (calibration, layer facts, ceiling, peaks).
- `evals/` — `agent/` (correctness gate, holdout guard, repeat audit, validate), `researcher/` (feedback and verdict,
  process evals, canaries, tier agreement, dashboard), `benches/compare/` (engine comparison), `kernels/` (the kernel
  evaluator: check, timing, core, jailed, verify, ncu, precheck, sanitize, freeze, ship).
- `generate/kernels/` — `search/` (the kernel search; `METHOD.md` maps each part to its source), `evolve/` (OpenEvolve
  arm), `tasks/` (kernel tasks, regions, shapes, pipeline, KernelBench, FlashInfer-Bench), `library/` (other
  implementations and PTX cards), `archive.py` (idea archive).
- `data/` — `corpus/` (frozen traces; held-out split in the store), `workloads/` (frontier benchmark loaders,
  researcher-only), `calibration/` (peaks, performance references), `knowledge/` (ledger, store, index, search;
  `LEDGER.md` is the record format, the environment's public interface).
- `local/` — gitignored: the ledger (`local/ledger`), run outputs (`local/runs`), task sets, downloads, notes.
- `tests/` — pytest, all synthetic, no GPU. Old design docs and `RUNS.md` are at tag `archive/docs`.

## Checks
`ruff check .`; `LAB_NO_JAIL=1 pytest -q` (tests that drive engine code skip unless its deps import); every
`tests/test_*.py` other than `test_lab_*` also runs as a script (`python tests/test_x.py`), as CI does. CI runs
`test_regimes.py` (~4 min) only when a PR touches `measure/loadgen/`, the canaries, the corpus, `targets/target.py`
or the workflows. Merge only on green CI.
