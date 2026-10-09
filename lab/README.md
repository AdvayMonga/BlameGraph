# lab

The environment that measures, and later grades, changes to the engine. Design in
`ENVIRONMENT.md` at the repo root.

## The target

Everything about *what* is being optimized lives in one TOML file, `targets/<name>.toml`, selected with
`LAB_TARGET` (or `--target`): the model and its chat kwargs, the engine repo, interpreter, launch command, health
path, env, logprobs API, write surface and test/lint commands, the reference server, the latency limits, the
correctness tasks and policy, and the corpus. Nothing defaults; without a target every command stops and lists
`targets/`. `targets/inference-server.toml` is the committed example. To point the environment at another engine,
copy it, change the engine section (and the reference if the model changes), and run the validation command.

## Contract with the engine

- The engine is a separate repo (github.com/AdvayMonga/inference-server). The lab finds it through
  `LAB_ENGINE_REPO` (default `../inference-server`) and runs its code, tests and lint with
  `LAB_ENGINE_PYTHON` (default that repo's `.venv/bin/python` when it exists, else this interpreter; the
  engine's venv needs its dev extras, `uv sync --extra dev`, for lint and tests). Nothing here imports the
  engine except the profiler harness and canaries, which run in the engine's process: run `python -m
  lab.profile` and `python -m lab.canary` with the engine's interpreter, from this directory.
- The target's write surface is the only thing the agent may write (for inference-server: `src/inference_server/`
  plus new `tests/test_*.py`).
- Black-box tools (bench, equiv, submit) serve the agent's pristine copy with the target's launch command, inside
  the jail (`lab/serve.py`: free localhost port, the target's env, ready when `health` answers 200, torn down with
  its process group), and talk to it over its OpenAI-compatible HTTP API.
- White-box tools (profile) build the backend and scheduler in process, because torch.profiler
  has to live in the process it traces. Profiling never gates anything.
- Engine instrumentation the lab reads: `TIMELINE_DIR` turns on the event timeline
  (`src/inference_server/timeline.py`), `TELEMETRY_DIR` the per-request rows. The target names them in
  `[engine.telemetry] env` with `{dir}`; serve points them at `lab-telemetry/` in the served tree and collects what
  appeared there after teardown. Optional: without the section nothing is collected. The rows follow the
  engine-agnostic request-trace contract in `lab/TRACE.md` (a dir of `*.sqlite` or `*.jsonl` + meta, keyed by
  `X-Trace-Id`); check a dir with `python -m lab.trace_contract DIR`.

## Tools

    python -m lab.profile --backend custom-mps --corpus-class steady_interactive --requests 8 --max-tokens 32 --out lab/runs

The engine configuration comes from the same env vars the server reads (`MAX_BATCH_SIZE`,
`PREFILL_MODE`, ...), so a profile measures what is served; `--backend` and `--model` override
`BACKEND` and `MODEL_NAME`. The workload is explicit, one of: `--corpus-class NAME` (`--split`, `--seed`,
`--requests`: sampled through the verified loader, chat-templated with the engine's tokenizer, thinking off
unless `--enable-thinking`), `--workload FILE` (the same, pre-sampled by `lab.corpus.sample`; the `profile`
tool uses it because the jail holds no corpus), `--prompts FILE` or `--synthetic` (random ids of `--prompt-len`).
Output length is `--max-tokens` for every request. Before the window, `--warmup` requests (default: as many as
the window) at the window's prompt lengths with random ids are fired together, so compiles and graph captures for
the window's batch sizes happen outside it. Writes one bundle directory per run, raw files only:

    events.jsonl   engine event timeline: scheduler decisions and phases, per step
    trace.json     torch.profiler chrome trace; phase ranges carry the step id
    ops.json       prof.key_averages() by self device then self CPU time: top 200 and totals (us)
    kernels.json   CUDA kernels by name: count, total, mean, max (us); only when there was CUDA activity
    memory.json    window peak (reset at window start) and lifetime peak (CUDA), allocator snapshots at window
                   start and end, peak host RSS (lifetime)
    gpu.csv        nvidia-smi samples at 100 ms with an `epoch_s` column, throttle reasons when the driver has them (CUDA hosts only)
    stats.json     the scheduler's counters at window start and end, and their difference
    meta.json      git sha, torch, device, engine settings, workload source and hash, window, warmup, profiler config
                   (`profiler_on`: every number was taken under the profiler), GPU clock state and throttle reasons at
                   window start and end, timeline events written/dropped, per-request outcomes and token counts

`--profiler` picks what wraps the measured window: `torch` (default, `trace.json`), `cuda-range`
(cudaProfilerStart/Stop around it, no torch trace, for an outer `nsys --capture-range=cudaProfilerApi` or
`ncu --profile-from-start off`), or `pyspy` (`--pyspy-bin`, `--seconds`, `--rate`: the workload repeats for
`--seconds` while py-spy samples this process without stopping it; adds `pyspy.speedscope.json`, `pyspy-dump.txt`
taken halfway, `pyspy-record.log`). For `pyspy` the process lets its own children ptrace it
(`PR_SET_PTRACER_ANY`), since Yama's default only lets ancestors.

## Ledger

`lab/ledger.py`: one JSON line per tool call in `lab/ledger/ledger.jsonl`, the raw knowledge base.
Not committed for now. Tools write it through `ledger.append`; the agent reads it and never writes it
(the jail's write surface is `src/inference_server/` only).

    python -m lab.ledger seed                      # import knowledge/*.json as `finding` records
    python -m lab.ledger list --kind bench         # JSONL; also --session, --snapshot
    python -m lab.ledger show ev-20261002-7f3a91

A record: `kind` (test, equiv, bench, profile, submit, finding), `session`, `snapshot`,
`parent_snapshot`, `base`, `change`, `config` (everything the number is a fact about), `metrics`,
`gates`, `raw`, `cost`, and `claim`, which holds the agent's own hypothesis and note: untrusted, never
used for a verdict. The writer stamps `id`, `at` and `schema`. Heavy raw files (diffs, bundles) go
through `ledger.put_blob` and are referenced by their content-hashed path under `blobs/`.
A held-out record (`config.split == "heldout"`) carries no `raw` and one aggregate per metric
(`base`, `new`, `delta_pct`, `band_pct`, `verdict`); the writer refuses anything finer. A `"split":
"heldout"` anywhere else in a record (outside `claim`) is refused, so held-out data can't hide under
`result`; tools write held-out results with `Toolbox._record_heldout`.

## Runtime

    python -m lab.session --task task.toml --budget 20      # or --goal "free text" with no objective
    # task.toml: [task] goal; [objective] regimes = [...], combine = "min"|"mean"; [constraints] <regime> = { max_regression_pct }

One loop: once per run, before the first session, the base commit is measured under `bench`'s default regimes
(seen split, short tier; the same cache `submit` compares against) and recorded as a `baseline` record
(`RunConfig.baseline=False` skips it). Then, while dollars remain, a fresh agent session gets a system prompt
generated from the target spec and the tool specs (`lab/prompts/system.md` is its template: target, write surface,
each tool with what it does, returns and costs, budget mechanics, how a session ends), the brief, and the tools,
and is free. Its shell and file tools run inside the srt jail on an
exported copy of the engine (no git history, held-out data removed); it may write
`src/inference_server/` and add `tests/test_*.py`, nothing else. The lab's own tools run out here
with the referee's rights, snapshot the workspace and write the ledger on every call:

| tool | does |
|---|---|
| `test` | lint and the fast suite on a pristine two-commit copy of the workspace, jailed |
| `profile` | `lab.profile` on the pristine copy, jailed, on `args.corpus_class` (default `steady_interactive`, seen split, `args.seed`) or `args.synthetic`; the bundle goes into the ledger as a blob |
| `trace` | `nsys profile` around `lab.profile --profiler cuda-range` (window only), then `nsys stats` (kernel, CUDA API, memcpy, NVTX tables as CSV), both jailed; `.nsys-rep` + CSVs as a blob |
| `kernel` | `ncu --kernel-name regex:<kernel_regex> --launch-skip N --launch-count N --set basic\|full` around the same workload, then `ncu --import --csv --page raw`; `.ncu-rep` + `metrics.csv` as a blob |
| `hostprof` | `lab.profile --profiler pyspy`: py-spy record (speedscope) for `seconds` under the repeated workload plus one `py-spy dump`; the bundle as a blob |
| `ledger` | read records (this run and earlier ones) |
| `budget` | dollars left |
| `restore` | workspace back to a snapshot id (`base` resets) |
| `note` | a note for the human; recorded, changes nothing |
| `bench` | serve the pristine copy and run the load regimes (`regimes/`) on the seen split, short tier; one headline per regime, raw (`args.regimes`, `args.tier`) |
| `equiv` | serve it and run the correctness gate (`correctness/`) against the target's reference outputs; pass, fail or inconclusive with every metric; `tier=full` uses `<reference.dir>-full` |
| `submit` | the only thing that can produce a win. Needs a passing full-tier equiv and a seen bench on this snapshot. Measures the held-out split at the full tier, measures the base commit the same way (once per run, cached outside the jail), and records one aggregate per regime (base, new, delta %, noise band %, verdict improved/regressed/within_band/unknown_band) through the Thresholdout guard (`validity/holdout.py`, state in `lab/ledger/holdout_state.json`, seed `LAB_HOLDOUT_SEED`): the held-out numbers themselves never reach the ledger. Noise bands come from `knowledge/noise/<regime>.json` when measured |

`trace`, `kernel` and `hostprof` take integers in fixed ranges (`requests` 1-256, `prompt_len` 1-32768,
`max_tokens` 1-4096, `launch_count` 1-64, `launch_skip` 0-100000, `seconds` 1-300, `rate` 1-1000), `set` from
basic/full, and a `kernel_regex` of at most 200 regex characters (no spaces, quotes, `;`, `&`, backticks, `/` or
`$(`); nothing else reaches the command line, which is an argv list, never a shell. The instrument is resolved on the
referee's PATH (refused, and recorded as a refusal, when absent) and its directory is added to the jail's read list.
The workload is lab.profile's in-process scheduler and backend, so these tools see the engine's kernels and host
code but not the HTTP front end.

### Cost

Every tool record's `cost` has `seconds`, the wall time of the call. Tools that occupy the GPU also charge those
seconds at the venue's rate to the run's budget (`budget.charge`), so `budget` and the next session's cap reflect
them, and record `gpu_seconds`, `usd_per_hour` and `rate_known`. The rate is `LAB_GPU_USD_PER_HOUR`, else the
target's `[cost] gpu_usd_per_hour`, else 0 with `rate_known: false` (no provider module knows a price offline; Verda's
`python -m lab.vm status` shows the live `$/h`). A GPU tool is refused once the budget is spent and a rate is set.
Only tool time is charged: the VM also bills while the model thinks between calls, which the budget does not see.

| tool | GPU | charged | typical wall time per call (estimates until measured on the H200) |
|---|---|---|---|
| `test` | no | $0 | CPU lint and suite |
| `ledger`, `budget` | no | $0 (pure reads, write no record) | milliseconds |
| `restore`, `note` | no | $0 | under a second |
| `profile` | yes | seconds × rate | 1-7 min measured (16 requests × 64 tokens, H200); 7 min when the weights were still loading from disk |
| `trace` | yes | seconds × rate | ~1 min measured (same workload): engine start, workload, report export, `nsys stats` |
| `kernel` | yes | seconds × rate | ~1.5 min measured at 4-8 launches; each launch is replayed once per metric pass, so it grows with `launch_count` and `full` |
| `hostprof` | yes | seconds × rate | ~35 s plus `seconds` (measured) |
| `bench` | yes | seconds × rate | 5-10 min per regime at tier short, 15-25 at full, plus ~2 min engine start |
| `equiv` | yes | seconds × rate | ~10 min at tier dev, 30-60 at full |
| `submit` | yes | seconds × rate | 15-25 min per regime (held-out, full tier), plus the base commit measured once per run |

The same estimates are in each tool's description, stated as facts about cost, with the configured rate.

Every session opens with the task in full, the referee's verdict on the run so far (integrity, with the evidence
behind each broken rule) and the harness facts the agent cannot read off its ledger (latency limits, tiers,
correctness policy, noise bands, corpus version, held-out queries left), then the baseline's headline numbers (or
why the base could not be measured) and only the last `session` record of the run. Nothing in it is
derived from the agent's own records and nothing is advice; the `ledger` tool has every record, unfiltered.

A session ends when the agent says `stop`, its per-session cap is spent, or it times out. The
run ends on budget, on `stop`, or on a write-surface violation (the one hard rule). Model cost
comes from the provider's own accounting; `LAB_MODEL` picks the model, `LAB_AGENT_PROVIDER` the
provider (only `claude` today, through the Agent SDK CLI; `lab/agent.py` is the seam for others).
Agent code runs jailed. On Linux the jail is a container (`lab/safety/container.py`: NVIDIA runtime, no
network, read-only root, non-root user, no capabilities; a served engine's port reaches the host through a socket
bridge; image from `python -m lab.safety.container build`, which `vm-setup.sh` runs). Elsewhere it is
sandbox-runtime (`npm install -g @anthropic-ai/sandbox-runtime`); `LAB_JAIL` picks one. `LAB_NO_JAIL=1` waives
the jail for tests and a box you trust. The agent's own CLI still runs under srt (`lab/RUNTIME.md`).

## Canaries

    python -m lab.canary slow_decode,kv_leak --port 8000

Serves the engine with deliberate regressions applied from outside (`lab/canary.py`
monkeypatches the backend and scheduler in the server process; the engine tree never changes,
so the player cannot edit a canary away). The eval harness must flag every one; a canary nothing
catches is a blind spot. `slow_decode` (+3 ms per step), `slow_start` (+20 s after load),
`kv_leak` (reservations never freed), `admit_fewer` (one fewer row per admission pass, config
unchanged in stats).

## GPU arm

One persistent VM, started and stopped by hand, on the cloud `LAB_VM_PROVIDER` names:
`verda` (default; 1x H200, REST API), `nebius` (1x H200, `nebius` CLI) or `crusoe` (1x A100 PCIe, `crusoe` CLI). Same
commands either way:

    python -m lab.vm status
    python -m lab.vm types                  # what the account can rent right now
    python -m lab.vm start                  # create (first time) or start, wait for ssh
    python -m lab.vm setup                  # lab/vm-setup.sh: venv, CUDA torch, Nsight, counter and clock checks
    python -m lab.vm run --env --fetch lab/runs -- \
        env BACKEND=custom-cuda python -m lab.profile --corpus-class steady_interactive --requests 8      # --env: run in this repo's tree
    python -m lab.vm run -- python scripts/gpu_tests/checks.py         # default: run in the engine's tree
    python -m lab.vm stop                   # ends all billing (Verda: deletes the VM and its disk)

`run` rsyncs the working tree minus `.git` and everything `.gitignore` excludes (what is on disk
here is what gets measured, secrets and weights stay home), runs the command in the repo dir with
the engine's `.venv/bin` first on PATH, and brings `--fetch` (relative to the dir it ran in) back under
`lab/runs/`. Both trees are pushed: the engine to `LAB_VM_DIR`, this repo to `LAB_VM_ENV_DIR`.
Do not run `uv sync` on the box: it would put CPU torch back.

**A VM is never left running by accident.** `run` stops the VM when its command ends (`--keep` leaves it up;
on Verda, where stop deletes the disk and a rebuilt VM re-downloads the weights, `run` keeps it by default and
`--stop` forces the delete).
Every session has a wall-clock cap, `LAB_VM_MAX_MINUTES` (default 180, `--minutes` per call): the command runs under
`timeout`, a detached local watchdog calls the provider's stop at the deadline (`start` and `--keep` arm it; each `run`
re-arms it; `stop` disarms it), and the VM schedules its own `shutdown -h` as a last resort. Powering off from inside
does not end billing on Nebius, which is why the watchdog goes through the provider. A stopped VM still bills its
disk; `stop` on Verda deletes the VM and disk outright. Cut an experiment short rather than extend the cap.

Credentials never live in the repo. Verda: `VERDA_CLIENT_ID` and `VERDA_CLIENT_SECRET` (console >
Keys > Cloud API credentials) in your shell. Nebius: `nebius profile create`. Crusoe: `crusoe config init`.

Nebius: `stop` is a real stop (boot disk kept, GPU billing ends) and `start` resumes it. The VM is
created with a `lab` user from cloud-init (Nebius refuses root). Platform `gpu-h200-sxm`, preset
`1gpu-16vcpu-200gb` (`LAB_VM_PLATFORM`, `LAB_VM_TYPE`), image family `ubuntu24.04-cuda13.0`
(`LAB_VM_IMAGE`), the project's first subnet unless `LAB_VM_SUBNET` is set, `LAB_VM_PROJECT` for a
project other than the profile's.

Verda's `stop` deletes the instance and its OS volume: Verda's hibernate hides the instance from
its own API so it cannot be restored, and its shutdown keeps billing the GPU. Each session therefore
starts on a fresh VM; `setup` takes about a minute plus weight downloads. The image ships nvcc 12.8,
which vLLM's DeepGEMM FP8 path refuses (needs 12.9+; serve with `VLLM_USE_DEEP_GEMM=0`).

| env | verda default | crusoe default | |
|---|---|---|---|
| `LAB_VM_PROVIDER` | `verda` | | |
| `LAB_VM` | `lab-gpu` | `lab-gpu` | VM name (Verda: hostname) |
| `LAB_VM_TYPE` | `1H200.141S.44V` | `a100-80gb.1x` | `types` lists what is rentable |
| `LAB_VM_LOCATION` | `FIN-02` | `us-east1-a` | |
| `LAB_VM_IMAGE` | `ubuntu-24.04-cuda-12.8-open-docker` | `ubuntu22.04-nvidia-slurm:latest` | an image with the NVIDIA driver |
| `LAB_VM_DISK_GB` | `400` | fixed by type | OS volume size (Verda) |
| `LAB_VM_USER` | `root` | `ubuntu` | |
| `LAB_VM_KEYFILE` | `~/.ssh/lab_ed25519.pub` | same | must have no passphrase (ssh runs in BatchMode); Verda: uploaded on first use |
| `LAB_VM_DIR` | `~/inference-server` | same | where the engine tree lands; its `.venv` is on PATH for every command |
| `LAB_VM_ENV_DIR` | `~/BlameGraph` | same | where this environment's tree lands; `run --env` runs there |
