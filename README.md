# BlameGraph

The environment for autoresearch loops that optimize LLM inference. The engine lives in [inference-server](https://github.com/AdvayMonga/inference-server).

- **One command:** `lab <command>` runs everything, and the agent's own tools by the same names
- **Lab:** the agent loop, its tools, ledger, sandbox and cloud GPUs
- **Workloads:** real-trace corpus
- **Data store:** every run's data kept in R2 and queryable with SQL
- **Integrity verdict:** did the agent earn its result
- **Session feedback:** facts for the agent, diagnostics for the researcher
- **Live dashboard:** what a run is doing, spending and measuring, read from files only
- **Load regimes:** drives a server the ten ways it gets used, real conversation timing, and finds its goodput
- **Correctness gate:** optimized outputs still match the original model
- **Canaries:** planted cheats that must get caught
- **Validity checks:** held-out budget, cache audit, short-vs-full test agreement
- **Kernel search:** an LLM-driven search that writes faster GPU kernels for the engine, judged by the kernel checker
- **Kernel checker:** rewritten kernels compute the same thing, on shapes the search never saw
- **Engine comparison:** the engine against vLLM, SGLang and TensorRT-LLM on one GPU at matched precision

## Install

```sh
git clone https://github.com/AdvayMonga/BlameGraph && cd BlameGraph
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"    # extras: store, correctness, kernel-search, agents, ...
export LAB_TARGET=targets/inference-server.toml                  # the engine to optimize
```

`pip install -e .` puts `lab` on your PATH; without installing, use `bin/lab`.

## Usage

```sh
lab                                   # every command
lab regimes run all --url http://localhost:8000   # load-test a running server
lab correctness candidate --url http://localhost:8000 --out equiv.json
lab session --task task.toml --budget 50          # an optimization run
lab dashboard                                     # watch it live
lab rehearsal setup /tmp/rehearsal                # the whole loop on a laptop, no GPU
```

`lab <command> -h` shows a command's arguments.
