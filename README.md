# BlameGraph

The environment for autoresearch loops that optimize LLM inference. Point it at your inference system.

- **`lab/`** — the optimization system: the agent loop and its tools, the kernel search, the correctness gate, the ledger, the sandbox and cloud GPUs
- **`evals/`** — the benchmarking suite: the real-trace corpus, the load regimes, the engine comparison, and the evals of the optimizing agent

`lab` uses `evals`; `evals` never uses `lab`. Everything runs through one command, `lab <command>`.
