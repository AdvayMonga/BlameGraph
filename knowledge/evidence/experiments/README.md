# experiments/ — the experiment ledger

One JSON file per experiment (the research loop's ledger, 2026-09; that loop and its `premerge_check.py` were
removed on 2026-10-02, inference-server tag `archive/research-loop`). Kept as the evidence behind the findings that
cite them. `ranker-bakeoff/` is the pre-registered hypothesis-ranking bake-off behind `kb-20260926-5dea54d0`.

| field | meaning |
|---|---|
| `hypothesis_id` | the hypothesis under test, with its prediction written before the run |
| `arms` | `baseline` and `treatment`, each with the engine SHA and the `runs/` panel ids it produced |
| `gates` | validity → sanity → significance → correctness → cost, judged in that order |
| `verdict` | `confirmed`, `rejected`, `noise` (effect inside the variance budget), `invalid` (harness never exercised the change) |
| `delta` | per-metric before / after / pct on the predicted metric |
| `regression_test` | for `correctness_fix` experiments: the test that fails at the base SHA and passes at the treatment |
| `no_behaviour_change` | for engine diffs that claim nothing (dead code, rename, comment): one line on why, vouching for the treatment SHA only. Written by `loop no-claim --why "..."` |
| `source` | `loop` (run through the procedure) or `reconstructed` (back-filled from the tuning log) |
| `cost_usd` | GPU spend, declared |

`exp-backfill-*` records were reconstructed from `benchmarks/tuning_log.md` and session notes
before the loop existed. They are honest about that: `source: "reconstructed"`, and no `runs/`
panels behind them.

Raw panels (`runs/*.json`) are gitignored — large and regenerable. What is kept is the record
of what was measured, at which SHA, and what the gates said.
