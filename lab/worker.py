"""Where the GPU work runs (lab/RUNTIME.md). A job takes input trees and JSON arguments, writes files into an output
dir and returns JSON; the tools keep the policy, the ledger and the budget. `Local` runs a job in this process.
`Remote` ships each input by content hash, runs `python -m lab.worker run JOB` on the worker host over SSH and brings
the output dir back; the worker re-hashes every tree right before using it and refuses one that does not match, so
what is measured is what the controller audited. GpuBusy, Contaminated, NotReady and ValueError cross intact. measure, equiv and profile take `workbench`, the agent's
container: it is paused for the job, and processes in it holding the GPU are killed and listed under `killed`.
Every job runs the tree with the interpreter `build.python_for` gives it: the engine's, or a venv built from the
tree's own dependency files.

  test     lint + the engine's suite on `tree`                      -> {lint, tests}
  measure  serve `tree`, run regimes; passive data into out          -> {results, ready_s}
  equiv    serve `tree`, the correctness gate against `reference`   -> the gate's result; outputs in out
  profile  lab.profile under torch, nsys, ncu or py-spy on `tree`   -> {returncode, output, ...}; files in out

    python -m lab.worker has SHA...  |  put SHA < tar  |  run JOB < json > tar      (the worker host's side)
    python -m lab.worker root  |  mkdir PATH  |  workbench < json > json  |  rm NAME    (the agent's side)

When the agent runs on the worker host too (Remote), its workspace lives there: `push` and `pull` mirror it with
rsync (pulled files are untrusted: links stay links and the audit refuses them), and `workbench` builds that host's
`docker run` for the CLI, with the Linux CLI of the controller's Agent SDK version.
"""

from __future__ import annotations

import dataclasses
from contextlib import contextmanager
import hashlib
import io
import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

from lab import engine, live, serve, target
from lab.safety import container

RESULT = "_result.json"
ENGINE_LOG = "engine.log"          # the served engine's log, in every job's out dir


class WorkerError(RuntimeError):
    """The worker could not run the job (transport, hash mismatch, or an unexpected failure there)."""


def tree_sha(root: Path) -> str:
    """Content hash of a directory: every regular file's path, exec bit and bytes; refuses symlinks."""
    h = hashlib.sha256()
    for p in sorted(Path(root).rglob("*")):
        rel = p.relative_to(root).as_posix()
        if p.is_symlink():
            raise WorkerError(f"symlink in a shipped tree: {rel}")
        if p.is_file():
            h.update(f"{rel}\0{os.access(p, os.X_OK):d}\0{p.stat().st_size}\0".encode())
            h.update(p.read_bytes())
    return h.hexdigest()


# -- jobs (run where the GPU is) ---------------------------------------------------------------------------------

def job_test(inputs: dict, out: Path, args: dict, hooks: dict) -> dict:
    from lab import build
    from lab.safety import grader
    tree = inputs["tree"]
    try:
        py = build.python_for(tree)
    except build.BuildFailed as e:              # a failed test run, recorded like any other
        skipped = grader.Run(False, -1, str(e))
        return {"lint": dataclasses.asdict(skipped), "tests": dataclasses.asdict(skipped)}
    with engine.using(py):
        lint = grader.run_lint(tree)
        tests = grader.run_tests(tree) if lint.passed else grader.Run(False, -1, "skipped: lint failed")
    return {"lint": dataclasses.asdict(lint), "tests": dataclasses.asdict(tests)}


def job_measure(inputs: dict, out: Path, args: dict, hooks: dict) -> dict:
    """`args`: names, split, tier, seed, jailed, passive (keep device/telemetry/client rows in `out`), hints and bands
    (`run_regimes`). One engine
    launch per profile (lab/profiles.py), general first; every result and row carries its `profile` (None = general),
    and a profile's device/telemetry/serve log land in `out/profiles/<regime>/`. A profile launch that fails makes
    its regimes invalid with the reason and the measurement goes on; the general launch failing, a busy GPU or a
    contaminated measurement still fail it whole."""
    from lab import artifacts, build, profiles
    from lab.evaltools import _why, run_regimes
    t = target.load()
    rows = [] if args["passive"] else None
    results, ready, failed = [], {}, {}
    try:
        with engine.using(build.python_for(inputs["tree"])), container.quiet(args.get("workbench")) as killed:
            for prof, names in profiles.launches(inputs["tree"], t, args["names"]):
                where = None
                if args["passive"]:
                    where = out if prof is None else out / "profiles" / prof
                    where.mkdir(parents=True, exist_ok=True)
                k = len(rows) if rows is not None else 0
                try:
                    with serve.Served(inputs["tree"], t.engine, log=out / ENGINE_LOG, jailed=args["jailed"],
                                      artifacts=where, env_extra=profiles.env(t, prof)) as srv:
                        got = run_regimes(srv.url, t, names, args["split"], args["tier"], args["seed"], rows,
                                          args.get("hints"), args.get("bands"))
                        srv.exclusive()
                    ready[prof] = srv.ready_s
                except (serve.GpuBusy, serve.Contaminated):
                    raise
                except Exception as e:
                    if prof is None:
                        raise
                    failed[prof] = _why(e)
                    general = {r["regime"]: r for r in results}
                    got = [{"regime": n, "objective": general[n]["objective"], "better": general[n]["better"],
                            "value": None, "valid": False,
                            "invalid_reasons": [f"profile launch failed: {failed[prof]}"]} for n in names]
                for r in got + (rows or [])[k:]:
                    r["profile"] = prof
                results += got
        from lab import gpu
        return {"results": results, "ready_s": ready[None], "profiles_ready_s": {p: s for p, s in ready.items() if p},
                "profiles_failed": failed, "gpu": gpu.query(), "killed": killed}
    finally:
        if rows is not None:
            artifacts.write_rows(out, rows)


def job_equiv(inputs: dict, out: Path, args: dict, hooks: dict) -> dict:
    """`args`: tier, concurrency, jailed, profile (None = general). The gate's outputs land in `out` as equiv.json and
    equiv.outputs.jsonl; answers already in `prior`'s equiv.outputs.jsonl are reused, not asked again."""
    from correctness import client, run as crun
    from correctness.gate import Thresholds
    from lab import build, profiles
    t = target.load()
    if (inputs["prior"] / "equiv.outputs.jsonl").exists():
        shutil.copyfile(inputs["prior"] / "equiv.outputs.jsonl", out / "equiv.outputs.jsonl")
    enc = hooks.get("encoder") or crun.HFEncoder(t.model, t.chat_kwargs)
    with engine.using(build.python_for(inputs["tree"])), container.quiet(args.get("workbench")) as killed, \
            serve.Served(inputs["tree"], t.engine, log=out / ENGINE_LOG, jailed=args["jailed"], artifacts=out,
                         env_extra=profiles.env(t, args.get("profile"))) as srv:
        api = client.Api(t.engine.api, t.chat_kwargs)
        res = (crun.quick(inputs["reference"], srv.url, t.model, out / "equiv.json", api=api) if args["tier"] == "quick"
               else crun.candidate(inputs["reference"], srv.url, t.model, out / "equiv.json", enc,
                                   Thresholds(min_score_ratio=t.min_score_ratio, min_length_ratio=t.min_length_ratio),
                                   concurrency=args["concurrency"], api=api, config={"tier": args["tier"]}))
        srv.exclusive()
    return {**res, "killed": killed}


def job_profile(inputs: dict, out: Path, args: dict, hooks: dict) -> dict:
    """`args`: kind (profile | trace | kernel | hostprof), the tool's args, workbench. Files go to `out/files`."""
    from lab import build
    try:
        py = build.python_for(inputs["tree"])
    except build.BuildFailed as e:              # a failed profile run, recorded like any other
        return {"returncode": -1, "output": str(e)[-3000:], "instrument": None, "argv": [], "head": "", "killed": []}
    with engine.using(py), container.quiet(args.get("workbench")) as killed:
        return {**_profile(inputs, out, args, hooks), "killed": killed}


def _profile(inputs: dict, out: Path, args: dict, hooks: dict) -> dict:
    from lab import profiles, proftools, tools as T
    tree, kind, a = inputs["tree"], args["kind"], args["args"]
    prof = {"env": profiles.env(target.load(), a["profile"])} if a.get("profile") else {}
    if a.get("capture"):                         # calls replayed from CUDA graphs never reach the capture's wrappers
        prof["env"] = {**prof.get("env", {}), **((target.load().raw.get("kernels") or {}).get("capture_env") or {})}
    if a.get("capture_task"):                    # what the task's calls need to occur at all (e.g. fp8 KV pools)
        from kernels.tasks import load as load_task
        prof["env"] = {**prof.get("env", {}), **dict(load_task(a["capture_task"]).capture_env)}
    files = out / "files"
    harness = T.stage_harness(tree)
    if kind == "profile":
        runs = tree / "lab" / "runs"
        n = int(a.get("requests", 8))
        argv = [engine.python(), "-m", "lab.profile", "--requests", str(n), "--max-tokens", str(a.get("max_tokens", 32)),
                "--out", str(runs), *T.workload_flags(a, harness, n),
                *[w for f in a.get("capture") or [] for w in ("--capture", str(f))]]
        from lab import gpu
        before = gpu.compute_apps()                  # anything on the GPU before or after the run shares it
        limit = {"timeout_s": args["timeout_s"]} if args.get("timeout_s") else {}
        proc = (hooks.get("profile_runner") or T.default_profile_runner)(tree, argv, **prof, **limit)
        others = [*(before or []), *(gpu.compute_apps() or [])]
        bundles = sorted(runs.glob("*")) if runs.exists() else []
        if bundles:
            T._drop_links(bundles[-1])
            shutil.copytree(bundles[-1], files)
        return {"returncode": proc.returncode, "output": (proc.stdout + proc.stderr)[-3000:], "gpu": gpu.query(),
                "gpu_others": others}
    ins = proftools.instrument(kind, a)
    exe = shutil.which(ins["binary"])
    if not exe:
        return {"refused": f"{ins['binary']} is not on PATH on this host"}
    exe = str(Path(exe).resolve())
    work = tree / proftools.OUT
    work.mkdir()
    read = [Path(exe).parent, Path(exe).parent.parent]     # the install root: nsys's importer lives beside its binary dir
    if Path("/opt/nvidia").is_dir():
        read.append(Path("/opt/nvidia"))                    # where launchers like CUDA's bin/ncu find the real install
    src = T.workload_flags(a, harness, proftools._int(a, "requests", *proftools.WORKLOAD["requests"]))
    argv = [*ins["wrap"](exe, work), engine.python(), "-m", "lab.profile", *ins["flags"], *src, *ins["extra"](exe),
            "--out", str(work / "bundle")]
    proc = T.default_profile_runner(tree, argv, read=read, **prof)
    log, commands = proc.stdout + proc.stderr, [argv]
    T._drop_links(work)
    if ins["post"] and proc.returncode == 0:
        pargv, dest = ins["post"](exe, work)
        commands.append(pargv)
        p = T.default_profile_runner(tree, pargv, read=read, **prof)
        if dest:
            dest.write_text(p.stdout)
        log += p.stderr if dest else p.stdout + p.stderr
        proc = p if p.returncode else proc
        T._drop_links(work)
    found = sorted(work.glob(ins["summary"]))
    if any(p.is_file() for p in work.rglob("*")):
        shutil.copytree(work, files)
    from lab import gpu
    return {"returncode": proc.returncode, "instrument": exe, "argv": commands, "output": log[-3000:],
            "head": proftools._head(found[-1] if found else None), "gpu": gpu.query()}


JOBS = {"test": job_test, "measure": job_measure, "equiv": job_equiv, "profile": job_profile}

# Failures the tools turn into records; anything else is a WorkerError on the controller.
_ERRORS = {"GpuBusy": lambda e: serve.GpuBusy(e["apps"]), "Contaminated": lambda e: serve.Contaminated(e["apps"]),
           "NotReady": lambda e: serve.NotReady(e["message"]), "ValueError": lambda e: ValueError(e["message"])}


def _error(e: BaseException) -> dict:
    return {"type": type(e).__name__, "message": str(e), "apps": getattr(e, "apps", None)}


def _raise(err: dict | None) -> None:
    if err:
        make = _ERRORS.get(err["type"])
        raise make(err) if make else WorkerError(f"{err['type']}: {err['message']}")


# -- transports ---------------------------------------------------------------------------------------------------

class Local:
    """Jobs in this process, on the controller's own paths. `hooks`: in-process stand-ins (an encoder, a profile
    runner) that tests and fakes provide; a remote worker has none."""

    def __init__(self, **hooks):
        self.hooks = {k: v for k, v in hooks.items() if v is not None}

    def call(self, job: str, inputs: dict[str, Path], args: dict, out: Path) -> dict:
        out.mkdir(parents=True, exist_ok=True)
        live.emit("job", job=job, state="start")
        try:
            return JOBS[job](inputs, out, args, self.hooks)
        finally:
            live.emit("job", job=job, state="end")


def _tar(root: Path) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        t.add(root, arcname=".")
    return buf.getvalue()


def _untar(data: bytes, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as t:
        t.extractall(dest, filter="data")


class Remote:
    """Jobs on another host. `prefix` is the argv that runs one shell command there (ssh ... user@host); empty runs
    it here through sh, which the tests use. `env_dir` is this repo on that host, `python` its interpreter."""

    def __init__(self, prefix: list[str], env_dir: str, python: str, env: dict[str, str], rsh: str | None = None,
                 sudo: bool = False):
        """`sudo`: the login user is not root, so the worker's side runs as root (the workbench is root inside, the
        measurement rooms drop to their own uid), keeping the login user's HOME (weights, venv, uv)."""
        self.prefix, self.env_dir, self.python, self.env, self.rsh, self.sudo = prefix, env_dir, python, env, rsh, sudo
        self._root: str | None = None

    def _where(self, path: str) -> str:
        return f"{self.prefix[-1]}:{path}" if self.prefix else path

    def _rsync(self, src: str, dst: str) -> None:
        """By content, not size and mtime: an edit within the same second must not be missed."""
        p = subprocess.run(["rsync", "-a", "--checksum", "--delete", *(["-e", self.rsh] if self.rsh else []),
                            *(["--rsync-path", "sudo -n rsync"] if self.sudo else []), src, dst],
                           capture_output=True, text=True)
        if p.returncode != 0:
            raise WorkerError(f"rsync {src} -> {dst}: {p.stderr.strip()[-1000:]}")

    def root(self) -> str:
        """This repo's absolute path on the worker host."""
        if self._root is None:
            self._root = self._run("root").decode().strip()
        return self._root

    def workspace_path(self, run_id: str) -> str:
        return f"{self.root()}/lab/runs/remote/{run_id}/workspace"

    def push(self, local: Path, remote: str) -> None:
        self._run(f"mkdir {shlex.quote(remote)}")
        self._rsync(f"{local}/", self._where(f"{remote}/"))

    def pull(self, remote: str, local: Path) -> None:
        Path(local).mkdir(parents=True, exist_ok=True)
        self._rsync(self._where(f"{remote}/"), f"{local}/")

    def workbench(self, name: str, session: str, workspace: str, sdk_version: str) -> dict:
        """{argv, sock}: that host's `docker run` for the CLI, and where the proxy's tunnel must land."""
        req = {"name": name, "session": session, "workspace": workspace, "sdk_version": sdk_version}
        return json.loads(self._run("workbench", json.dumps(req).encode()))

    @contextmanager
    def paused(self, name: str | None):
        """The workbench on the worker host frozen for the block (as `container.paused` there)."""
        did = bool(name) and self._run(f"pause {shlex.quote(name)}").decode().strip() == "paused"
        try:
            yield
        finally:
            if did:
                self._run(f"unpause {shlex.quote(name)}")

    def remove_workbench(self, name: str | None) -> None:
        if name:
            self._run(f"rm {shlex.quote(name)}")

    def _run(self, command: str, stdin: bytes = b"") -> bytes:
        for v in self.env.values():
            if any(c in v for c in '"`\\'):
                raise WorkerError(f"unsafe value in the worker's env: {v!r}")
        exports = " ".join(f'{k}="{v}"' for k, v in self.env.items())
        root = 'sudo -n env HOME="$HOME" PATH="$PATH" ' if self.sudo else ""
        line = f"cd {self.env_dir} && {root}{exports} {self.python} -m lab.worker {command}"
        argv = [*self.prefix, line] if self.prefix else ["sh", "-c", line]
        p = subprocess.run(argv, input=stdin, capture_output=True)
        if p.returncode != 0:
            raise WorkerError(f"worker `{command}` exited {p.returncode}: {p.stderr.decode(errors='replace')[-2000:]}")
        return p.stdout

    def call(self, job: str, inputs: dict[str, Path], args: dict, out: Path) -> dict:
        shas = {k: tree_sha(p) for k, p in inputs.items()}
        have = set(self._run("has " + " ".join(shas.values())).decode().split()) if shas else set()
        for k, p in inputs.items():
            if shas[k] not in have:
                self._run(f"put {shas[k]}", _tar(p))
        live.emit("job", job=job, state="start")
        try:
            _untar(self._run(f"run {job}", json.dumps({"inputs": shas, "args": args}).encode()), out)
        finally:
            live.emit("job", job=job, state="end")
        reply = json.loads((out / RESULT).read_text())
        (out / RESULT).unlink()
        _raise(reply.get("error"))
        return reply["result"]


def load(**hooks):
    """`LAB_WORKER`: local (default) or ssh, the VM `lab.vm` names (started and set up beforehand)."""
    kind = os.environ.get("LAB_WORKER", "local")
    if kind == "local":
        return Local(**hooks)
    if kind != "ssh":
        raise ValueError(f"LAB_WORKER is local or ssh, not {kind!r}")
    from lab import vm
    v = vm.VM()
    record = v.provider.get(v.name)
    if record is None or record.state != "running":
        raise WorkerError(f"{v.name} is not running: start it with `python -m lab.vm start` and set it up first")
    spec = Path(os.environ.get("LAB_TARGET", "")).resolve()
    if not spec.is_relative_to(engine.ENV_ROOT):
        raise WorkerError(f"LAB_TARGET must be inside this repo to be found on the worker: {spec}")
    home = lambda p: p.replace("~", "$HOME", 1)       # noqa: E731 - expanded by the worker's shell
    return Remote(["ssh", *v.ssh_opts, vm.ssh_target(v, record)], v.env_dir, home(v.remote_dir) + "/.venv/bin/python",
                  {"LAB_TARGET": spec.relative_to(engine.ENV_ROOT).as_posix(), "LAB_ENGINE_REPO": home(v.remote_dir)},
                  rsh=f"ssh {shlex.join(v.ssh_opts)}", sudo=v.login != "root")


# -- the worker host's side ---------------------------------------------------------------------------------------

def cache_dir() -> Path:
    return Path(os.environ.get("LAB_WORKER_CACHE") or engine.ENV_ROOT / "lab" / "runs" / "worker-cache")


def _put(sha: str, data: bytes) -> None:
    dest = cache_dir() / sha
    if dest.exists():
        return
    part = Path(tempfile.mkdtemp(prefix=f"{sha[:12]}-", dir=cache_dir()))
    _untar(data, part)
    got = tree_sha(part)
    if got != sha:
        shutil.rmtree(part)
        raise WorkerError(f"received tree hashes to {got}, not {sha}")
    part.rename(dest)


def _run_job(job: str, request: dict) -> bytes:
    """Fresh copies of the inputs (jobs write into them), each re-hashed right before use; out dir as a tar."""
    work = Path(tempfile.mkdtemp(prefix=f"job-{job}-"))
    out = work / "out"
    out.mkdir()
    try:
        try:
            inputs = {}
            for name, sha in request["inputs"].items():
                src = cache_dir() / sha
                if not src.is_dir():
                    raise WorkerError(f"input {name} ({sha}) was never put")
                inputs[name] = work / name
                shutil.copytree(src, inputs[name])
                if tree_sha(inputs[name]) != sha:
                    raise WorkerError(f"input {name} changed after it was put")
            reply = {"result": JOBS[job](inputs, out, request["args"], {})}
        except Exception as e:                  # the controller re-raises it; the tar still carries what was written
            reply = {"error": _error(e)}
        (out / RESULT).write_text(json.dumps(reply, default=str))
        return _tar(out)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def agent_cli(version: str) -> Path:
    """The Agent SDK's bundled CLI for this host at `version`, installed once into the cache."""
    d = cache_dir() / f"cli-{version}"
    exe = d / "claude_agent_sdk" / "_bundled" / "claude"
    if not exe.exists():
        uv = shutil.which("uv") or str(Path.home() / ".local" / "bin" / "uv")
        subprocess.run([uv, "pip", "install", "--quiet", "--no-deps", "--target", str(d), "--python", sys.executable,
                        f"claude-agent-sdk=={version}"], check=True, stdout=sys.stderr)
    return exe


def _workbench(req: dict) -> dict:
    from lab import agent
    sock_dir = Path(f"/tmp/lab-proxy-{req['session']}")
    sock_dir.mkdir(mode=0o700, exist_ok=True)
    if os.environ.get("SUDO_UID"):                  # sshd binds the tunnel's socket here as the login user
        os.chown(sock_dir, int(os.environ["SUDO_UID"]), int(os.environ["SUDO_GID"]))
    (sock_dir / "proxy.sock").unlink(missing_ok=True)
    scratch = engine.ENV_ROOT / "lab" / "runs" / "remote" / "scratch" / req["session"]
    argv = agent.workbench_command(req["name"], Path(req["workspace"]), scratch, agent_cli(req["sdk_version"]),
                                   sock_dir=sock_dir)
    return {"argv": argv, "sock": str(sock_dir / "proxy.sock")}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    cache_dir().mkdir(parents=True, exist_ok=True)
    if argv[:1] == ["has"]:
        print(" ".join(s for s in argv[1:] if (cache_dir() / s).is_dir()))
    elif argv[:1] == ["put"] and len(argv) == 2:
        _put(argv[1], sys.stdin.buffer.read())
    elif argv[:1] == ["run"] and len(argv) == 2 and argv[1] in JOBS:
        sys.stdout.buffer.write(_run_job(argv[1], json.loads(sys.stdin.buffer.read())))
    elif argv == ["root"]:
        print(engine.ENV_ROOT)
    elif argv[:1] == ["mkdir"] and len(argv) == 2:
        Path(argv[1]).mkdir(parents=True, exist_ok=True)
    elif argv == ["workbench"]:
        print(json.dumps(_workbench(json.loads(sys.stdin.buffer.read()))))
    elif argv[:1] == ["rm"] and len(argv) == 2:
        container.remove(argv[1])
    elif argv[:1] == ["pause"] and len(argv) == 2:        # prints "paused" only if this call froze it
        print("paused" if container.pause(argv[1]) else "")
    elif argv[:1] == ["unpause"] and len(argv) == 2:
        container.unpause(argv[1])
    else:
        print(f"usage: python -m lab.worker has SHA... | put SHA | run {{{','.join(JOBS)}}} | root | mkdir PATH | "
              "workbench | rm NAME | pause NAME | unpause NAME", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
