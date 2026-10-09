# Where the lab runs

Design, 2026-10-08. Decided with the user: the lab splits into a controller and a worker; no separate
measurement VM (cost); NVIDIA only for now; the jail on Linux is a container with the NVIDIA runtime (option A
below). Built: the container jail (`lab/safety/container.py`), GPU exclusivity during measurements, and the
measured jobs behind the worker interface (`lab/worker.py`). Not built: the agent on the worker (its container, the
API proxy) and the agent-side jobs. Open decisions are marked **Decide**.

## Why

Today every part of the lab runs on one machine and serves the engine on that machine's localhost. On a
Linux GPU VM no jailed GPU tool can work, for two reasons:

- **No GPU in the jail.** srt mounts a fresh minimal `/dev` and has no device passthrough: jailed code gets
  `No CUDA GPUs are available` (H200, Nebius, 2026-10-08; see `fix/profiler-gpu`'s CLAUDE.md).
- **No way in from outside.** On Linux srt runs every jail in its own network namespace (`bwrap
  --unshare-net`); `allowLocalBinding` only means something on macOS. A server inside a jail answers only
  inside that jail, so `serve.py`'s health check and the measurement clients outside it are refused.
  Measured in Docker (Linux, srt from npm, 2026-10-08): same jail serve + request → 200; host → jailed
  server, other jail → jailed server, jail → host server: all refused.

What srt does well on Linux, from the same test: its own PID namespace (a jail sees only its processes and
cannot signal others), and everything in a jail dies when the jail's command exits.

## Trust zones

| Zone | Where | Holds | Agent's code can reach it |
|---|---|---|---|
| Controller | laptop (later an always-on box) | session loop, ledger, referee, holdout guard, budget, snapshot store, every secret | no |
| Worker | GPU VM, host side, outside every jail | job runner, measurement clients, profilers, GPU state | only through the engine's HTTP port |
| Jails | GPU VM, inside containers | agent's shell and workspace; each served engine | yes, it is the agent's |

Rules:

1. **No secrets on the VM.** No cloud keys, no model API key, no HF token (weights are staged on a read-only
   disk). Model calls go through a proxy on the controller that adds the key.
2. **Only the controller writes the ledger.** The worker returns results, raw files and hashes; the controller
   records them.
3. **What is measured is what was audited.** The controller pulls each snapshot's changed files, stores them,
   audits them, and sends that exact content back for `test`/`bench`/`equiv`/`submit`/`profile`. The worker
   rebuilds the tree from base + those files and returns its hash; a mismatch refuses the job.
4. **Nothing of the agent runs during a measurement.** Before every measured job the worker freezes the agent's
   jail (cgroup freezer) and kills any of its processes holding a GPU context; it thaws after. The tool result
   states what was killed (a fact).
5. **Measurement code is out of reach.** The clients run on the host, outside every container, from a tree the
   jails cannot read; their CPU cores are reserved (cpuset) so the engine cannot starve them.
6. **One VM per run, destroyed after** (**Decide**; recommended). Nothing from one run can touch the next one's numbers. Setup time is the
   cost; a prebuilt image with the venv plus a reattached read-only weights disk cuts it.
7. **Network.** VM egress: the controller tunnel only. Agent jail: the API proxy only. Engine jail: nothing; its
   port is reachable from the host alone.

Left open by not having a separate measurement VM: the clients and the agent's code share a kernel, so a
kernel or container escape defeats rules 4 and 5. A measurement VM in the same private network closes it later.

## The jail on Linux

Needs: the GPU (devices and driver libraries); a port the host can reach and nothing else; write access to the
workspace and a tmp only; weights read-only; an unprivileged user; its own PID namespace; CPU, memory and
process limits; freeze and kill as a unit.

Options considered; **A was chosen** (2026-10-08):

- **A. Containers with the NVIDIA runtime.** Docker or Podman plus NVIDIA Container Toolkit:
  `--gpus`, an `--internal` network per jail (host reaches the engine's port, no egress), read-only root,
  non-root user, all capabilities dropped, `no-new-privileges`, seccomp default, cgroup limits, `pause` as the
  freezer. Standard and well-trodden (SWE-bench and OpenHands sandbox this way), and the same unit a scheduler
  runs at scale (a pod). Harder later: gVisor (`runsc`, GPU via nvproxy) as a drop-in for a stronger boundary.
- **B. bwrap called directly.** Bind the `/dev/nvidia*` devices and driver libraries in, share the host network,
  and restrict egress per uid with nftables. Lighter, but we would own every sharp edge srt hides.
- **C. The VM as the boundary.** One VM per jail. Strongest, and ruled out for cost.

macOS stays as today (srt; Seatbelt shares the network stack) for development and the fake-engine tests.

As built: one container per jailed command (`lab-jail` image: Ubuntu 24.04 with a compiler, Python and socat;
the engine's venv, interpreter, weights and `/usr/local/cuda` bind-mounted read-only at their host paths).
`--network none` rather than an internal network: the engine keeps binding 127.0.0.1, a socat inside the
container serves that port on a unix socket in the jail's tmp, and a host socat serves the socket on the host's
127.0.0.1. Nothing in any jail can reach it; the clients pay one forwarding hop per connection, the same for base
and candidate. Run as root (Verda), the jail's user is uid 10001 and the writable paths are handed to it; the
first `CLIENT_CORES` cores are kept out of every jail. Tested in Docker-in-Docker on Ubuntu 24.04 (no network,
no capabilities, non-root, host files invisible, port reaches the host and no other jail, nothing left after
timeout or teardown); `--gpus` and the profilers inside it are untested until a GPU run.

Profilers under A: nsys runs inside the engine's container; ncu needs GPU performance counters, which
the single-use VM can open to non-admins (`NVreg_RestrictProfilingToAdminUsers=0`) since nothing else runs on it;
py-spy attaches from the host (root) into the container's process.

## A tool call, end to end (Claude provider)

1. The agent's CLI runs in the agent container on the VM. Its stdin/stdout reach the Agent SDK on the
   controller over SSH, so the lab's tools stay an in-process MCP server on the controller, as today.
2. The agent edits and runs commands in its container (built-in tools; they never leave the VM).
3. It calls a lab tool. The controller asks the worker for the snapshot (changed files), stores and audits
   them, and writes the ledger record's snapshot.
4. For a measured tool: the controller sends the job with the snapshot's files and hash. The worker freezes
   the agent container, rebuilds and checks the tree, serves it in an engine container, runs the clients from
   the host, tears the engine down, thaws, and returns results, raw files and hashes.
5. The controller applies the policy (submit's preconditions, the holdout guard, noise bands), writes the
   ledger, and answers the agent.

The model API path: CLI → `ANTHROPIC_BASE_URL` on the agent network → a host forwarder → SSH tunnel →
controller proxy (adds the key, meters tokens) → api.anthropic.com. Metering at the proxy makes the budget a
measured fact instead of the CLI's report.

## The worker job interface

A short fixed list; each job takes a snapshot (files + hash), the target spec and arguments, and returns result
JSON, raw files with hashes, and hardware facts (device, driver, clocks).

| Job | Does |
|---|---|
| `agent` | start the agent container on the workspace and stream the CLI's stdio (Claude) |
| `exec` | one command in the agent container (providers with their own tool loop) |
| `snapshot` | the workspace's changed files since base |
| `restore` | put the workspace at a snapshot the controller sends |
| `test` | lint and the engine's suite on the rebuilt tree, jailed |
| `measure` | serve the rebuilt tree and run regimes or the correctness gate (`bench`, `equiv`, `submit`) |
| `profile` | `profile`, `trace`, `kernel`, `hostprof` |
| `facts` | GPU, driver, clocks, versions |

Transport now: SSH from the controller, one worker. Transport later: a queue. The job list does not change.

Built (2026-10-09, `lab/worker.py`): `test`, `measure` (bench, submit and the base share it), `equiv` and `profile`
(all four instruments). The tools keep the policy, the ledger and the budget and call `self.worker`; `LAB_WORKER`
picks `local` (default: in process, as before) or `ssh` (the VM `lab.vm` names, started and set up beforehand).
Over SSH each input tree goes by content hash, at most once per worker (`has`, `put`), and the worker re-hashes a
fresh copy right before every job, refusing a mismatch. The job's out dir comes back as a tar; GpuBusy,
Contaminated, NotReady and ValueError are raised again on the controller. equiv ships the snapshot's earlier
answers with the job, so calling it again never re-rolls the gate. Tested through a local shell with the same CLI;
not yet over SSH to a VM. `agent`, `exec`, `snapshot`, `restore` and `facts` come with the agent container.

## What moves where

- Controller: `session`, `agent` (SDK side), the tool front ends in `tools`/`evaltools`/`proftools` (policy,
  ledger writes), `ledger`, `budget`, `task`, `feedback/`, the holdout guard, `workspace`'s snapshot store,
  `grader.audit`, `vm` and `providers/`.
- Worker: a new job runner; `serve`, `regimes/`, `correctness/` (clients), `profile`/`bundle`/`gpu`,
  `grader.run_tests`, `canary`, the container jail.

## Leftover processes, today

On Linux, srt already isolates them (own PID and network namespace; killed when the jail exits). During a
session they could still share the GPU with the engine being measured. Since 2026-10-09 `serve.py` refuses to
launch an engine while any process holds the GPU (`GpuBusy`) and polls the GPU's processes every second while it
serves; anything outside the engine's own processes marks the measurement `contaminated` (`Served.exclusive()`):
never evidence for submit, never cached as the base, and caught before the holdout guard spends a query. The
facts (pid, MiB, name) go to the agent. It refuses rather than kills because the lab cannot yet tell the agent's
processes apart; rule 4's freeze-and-kill comes with the agent container. Not covered: CPU contention, and a GPU
burst shorter than the poll interval.

## At scale (not built)

Many single-use workers under a scheduler (SkyPilot, Kubernetes or Ray) instead of `vm.py`; the controller as
an always-on service holding the one holdout guard per corpus version; corpus, tasks and weights in object
storage fetched by hash; the target spec describing multi-GPU and multi-node layouts; ledgers indexed in a
database, still append-only.

## Order

1. ~~Decide the jail~~ (A) and build it. Done, minus the GPU check.
2. Without a GPU: ~~the job interface with a local transport, the snapshot round trip and rule 3~~ (done for the
   measured jobs); next the agent container with its freeze, and the API proxy.
3. With a GPU (spend approval): `--gpus` and the profilers inside the container, then a first `lab.session`
   dry run with a tiny budget.
