"""Grader: checks the agent's workspace from outside the jail, trusting nothing inside it."""

from __future__ import annotations

import difflib
import hashlib
import io
import os
import shutil
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path

from lab import engine, gpu, target
from lab.safety import container, jail
from lab.safety.surfaces import ALWAYS_DENY, HIDDEN, deps_only, may_write

IGNORED = ("*/__pycache__/*", "__pycache__/*", "*.pyc", ".pytest_cache/*", "*/.pytest_cache/*",
           ".ruff_cache/*", "*/.ruff_cache/*", "*.egg-info/*", ".DS_Store", "*/.DS_Store",
           "lab/runs/*")          # profile bundles the tool copies in for the agent to read; often > MAX_FILE_BYTES
MAX_FILE_BYTES = 5_000_000
HF_HUB = Path.home() / ".cache" / "huggingface" / "hub"   # weights only; the token beside it stays unreadable


@dataclass
class Audit:
    added: list[str] = field(default_factory=list)
    modified: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    scratch: list[str] = field(default_factory=list)     # added outside the surface: left out, not a violation
    violations: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations

    @property
    def changed(self) -> list[str]:
        return sorted(self.added + self.modified + self.deleted)


@dataclass
class Run:
    passed: bool
    returncode: int
    output: str


def export(repo: Path, base: str, dest: Path) -> None:
    """The tree at `base` into `dest`, with no .git and no hidden files."""
    tar = subprocess.run(["git", "-C", str(repo), "archive", "--format=tar", base],
                         capture_output=True, check=True).stdout
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(tar)) as t:
        t.extractall(dest, filter="data")
    for p in list(dest.rglob("*")):
        rel = p.relative_to(dest).as_posix()
        if p.is_file() and any(fnmatch(rel, h) for h in HIDDEN):
            p.unlink()


def _base_blobs(repo: Path, base: str) -> dict[str, str]:
    out = subprocess.run(["git", "-C", str(repo), "ls-tree", "-r", "-z", base],
                         capture_output=True, check=True, text=True).stdout
    blobs = {}
    for entry in filter(None, out.split("\0")):
        meta, path = entry.split("\t", 1)
        blobs[path] = meta.split()[2]
    return blobs


def _blob_sha(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def base_files(repo: Path, base: str) -> frozenset[str]:
    return frozenset(_base_blobs(repo, base))


def audit(repo: Path, base: str, workspace: Path) -> Audit:
    """Every difference between `workspace` and `base`, and which ones the agent may not make.

    Integrity failures are violations: symlinks, hidden files, evaluator config, and any change or
    deletion of a protected file. A new file outside the surface is scratch: it never reaches the
    pristine tree, and running the engine in the workspace is allowed to leave it behind.
    """
    blobs = _base_blobs(repo, base)
    a, seen = Audit(), set()
    for root, dirs, files in os.walk(workspace):
        for name in dirs + files:
            p = Path(root) / name
            rel = p.relative_to(workspace).as_posix()
            if not p.is_symlink() and any(fnmatch(rel, g) for g in IGNORED):
                continue
            if _scratch(rel, blobs) and (p.is_symlink() or not (p.is_dir() or p.is_file())
                                         or (p.is_file() and p.stat().st_size > MAX_FILE_BYTES)):
                a.scratch.append(rel)               # never reaches the pristine tree: a venv's links, big wheels
                continue
            if p.is_symlink():
                a.violations.append(f"symlink: {rel}")
            elif p.is_dir() or any(fnmatch(rel, g) for g in IGNORED):
                continue
            elif not p.is_file():
                a.violations.append(f"not a regular file: {rel}")
            elif p.stat().st_size > MAX_FILE_BYTES:
                a.violations.append(f"over {MAX_FILE_BYTES} bytes: {rel}")
            else:
                seen.add(rel)
                if any(fnmatch(rel.lower(), h.lower()) for h in HIDDEN):
                    a.violations.append(f"recreated hidden file: {rel}")
                elif rel not in blobs:
                    a.added.append(rel)
                elif _blob_sha(p) != blobs[rel]:
                    a.modified.append(rel)
    a.deleted = sorted(p for p in blobs if p not in seen and not any(fnmatch(p, h) for h in HIDDEN))
    for rel in list(a.added):
        if not may_write(rel, new_file=True):
            if any(fnmatch(rel.lower(), p.lower()) for p in ALWAYS_DENY):
                a.violations.append(f"may not add: {rel}")
            else:
                a.added.remove(rel)
                a.scratch.append(rel)
    for rel in a.modified + a.deleted:
        if not may_write(rel, new_file=False):
            a.violations.append(f"may not change: {rel}")
    deps = target.load().deps
    for rel in a.deleted:
        if rel in deps:
            a.violations.append(f"may not delete a dependency file: {rel}")
    for rel in a.modified:
        if rel in deps and rel.endswith(".toml"):
            old = subprocess.run(["git", "-C", str(repo), "show", f"{base}:{rel}"], capture_output=True, text=True).stdout
            if not deps_only(old, (workspace / rel).read_text(errors="replace")):
                a.violations.append(f"may change only the dependency tables of {rel}")
    return a


def _scratch(rel: str, blobs: dict) -> bool:
    """Outside what the agent may write and not a base file: left out of every tree, whatever it is."""
    return (rel not in blobs and not may_write(rel, new_file=True)
            and not any(fnmatch(rel.lower(), p.lower()) for p in ALWAYS_DENY + HIDDEN))


def pristine_tree(repo: Path, base: str, workspace: Path, result: Audit, dest: Path) -> None:
    """A fresh export of `base` plus only the audited changes, as a two-commit repo (base, change)."""
    if not result.ok:
        raise ValueError("refusing to build from a workspace that failed its audit")
    if dest.exists():
        shutil.rmtree(dest)
    export(repo, base, dest)
    _commit(dest, "base", init=True)
    for rel in result.added + result.modified:
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(workspace / rel, dest / rel)
    for rel in result.deleted:
        (dest / rel).unlink()
    _commit(dest, "change")


def _commit(tree: Path, message: str, init: bool = False) -> None:
    git = ["git", "-C", str(tree), "-c", "user.name=lab", "-c", "user.email=lab@localhost",
           "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null"]
    if init:
        subprocess.run(["git", "-C", str(tree), "init", "-q"], check=True)
    subprocess.run(git + ["add", "-A", "--force"], check=True)
    subprocess.run(git + ["commit", "-q", "--allow-empty", "--no-verify", "-m", message], check=True)


def diff(repo: Path, base: str, tree: Path, result: Audit) -> str:
    """Unified diff of the audited changes, from `base` to `tree`."""
    out = []
    for rel in result.changed:
        old = b"" if rel in result.added else subprocess.run(
            ["git", "-C", str(repo), "show", f"{base}:{rel}"], capture_output=True, check=True).stdout
        new = b"" if rel in result.deleted else (tree / rel).read_bytes()
        if b"\0" in old or b"\0" in new:
            out.append(f"Binary file {rel} changed\n")
            continue
        out += difflib.unified_diff(old.decode(errors="replace").splitlines(keepends=True),
                                    new.decode(errors="replace").splitlines(keepends=True),
                                    f"a/{rel}", f"b/{rel}")
    return "".join(out)


def _argv(command: str) -> list[str]:
    """A target command, `python` meaning the engine's interpreter."""
    import shlex
    words = shlex.split(command)
    if words and words[0] in ("python", "python3"):
        words[0] = engine.python()
    return words


def _engine_path() -> str:
    return str(Path(engine.python()).parent) + os.pathsep + os.environ["PATH"]


def run_lint(tree: Path) -> Run:
    """The target's lint command, with the engine's python first on PATH; static, so it needs no jail."""
    proc = subprocess.run(_argv(target.load().lint), cwd=tree, capture_output=True, text=True,
                          env={**os.environ, "PATH": _engine_path()})
    return Run(proc.returncode == 0, proc.returncode, (proc.stdout + proc.stderr)[-5000:])


def _jail_argv(tree: Path, argv: list[str], env_extra: dict | None, domains, local_binding: bool = False,
               tag: str = "", read: list[Path] = (), port: int | None = None
               ) -> tuple[list[str], dict, Path, str | None]:
    """The wrapped argv, env and private tmp for running `argv` in `tree` as agent code: wiped env, jail around the
    tree and the tmp, weights read-only. The last item names the container when the container jail is used."""
    tmp = Path(tempfile.mkdtemp(prefix="jail-")).resolve()   # short: srt's sockets live here
    env = {"PATH": _engine_path(), "HOME": str(Path.home()), "TMPDIR": str(tmp),
           "PYTHONPATH": str(tree / "src") + os.pathsep + str(tree),
           "PYTHONDONTWRITEBYTECODE": "1",
           "HF_HOME": str(tmp / "hf-home"), "HF_HUB_CACHE": str(HF_HUB), "HF_HUB_OFFLINE": "1",
           **(env_extra or {})}
    readonly = ([HF_HUB] if HF_HUB.exists() else []) + list(read)
    if jail.backend() == "container":
        if not container.available():
            if jail.required():
                raise jail.JailMissing(f"no container jail: needs docker and the {container.IMAGE} image "
                                       "(python -m lab.safety.container build), or LAB_NO_JAIL=1 on a box you trust")
            return argv, env, tmp, None
        if domains:
            raise ValueError("the container jail has no network")
        name = container.new_name()
        container.give([tree.resolve(), tmp])
        wrapped = container.argv(name, tree, argv, {**env, "HOME": str(tmp)}, [tree.resolve(), tmp],
                                 [engine.venv(), *readonly], engine.python(), bridge_port=port)
        cli_env = {k: os.environ[k] for k in ("PATH", "HOME", "DOCKER_HOST", "DOCKER_CONFIG") if k in os.environ}
        return wrapped, cli_env, tmp, name
    config = jail.settings([tree.resolve(), tmp], engine.venv(), list(domains), readonly=readonly,
                           python=engine.python(), local_binding=local_binding)
    # srt sets TMPDIR to a dir that does not exist inside bwrap (nsys and ncu then refuse to start): restate ours.
    return jail.wrap(config, tree.with_suffix(f"{tag}.srt.json"), ["env", f"TMPDIR={tmp}", *argv]), env, tmp, None


def jailed_popen(tree: Path, argv: list[str], *, env_extra: dict | None = None, stdout=None, stderr=None,
                 local_binding: bool = True, port: int | None = None):
    """A long-running jailed process (a served engine) in its own session, reachable on the host's 127.0.0.1:`port`.
    Returns (proc, tmp dir to remove, stop, pids): `stop` removes what lives outside proc's process group; `pids()`
    is every host pid the jail runs."""
    wrapped, env, tmp, name = _jail_argv(tree, argv, env_extra, (), local_binding, tag=".serve", port=port)
    proc = subprocess.Popen(wrapped, cwd=tree, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
    if name is None:
        return proc, tmp, lambda: None, lambda: gpu.descendants(proc.pid)
    bridge = container.Bridge(port, tmp / container.BRIDGE) if port else None

    def stop() -> None:
        container.remove(name)
        if bridge:
            bridge.stop()
    return proc, tmp, stop, lambda: container.pids(name)


def jailed(tree: Path, argv: list[str], *, timeout_s: float, env_extra: dict | None = None,
           domains: list[str] = (), read: list[Path] = ()) -> subprocess.CompletedProcess:
    """Run `argv` in `tree` as agent code: wiped env, jail around the tree and a private tmp, weights and `read`
    read-only."""
    wrapped, env, tmp, name = _jail_argv(tree, argv, env_extra, domains, read=read)
    try:
        return subprocess.run(wrapped, cwd=tree, env=env, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as e:     # a hung suite is a failed run, recorded like any other
        out = (e.stdout or b"")
        return subprocess.CompletedProcess(argv, -9, out.decode(errors="replace") if isinstance(out, bytes) else out,
                                           f"timed out after {timeout_s:.0f}s")
    finally:
        if name:
            container.remove(name)
        shutil.rmtree(tmp, ignore_errors=True)


def run_tests(tree: Path, timeout_s: float = 1800) -> Run:
    """The fast suite on a pristine tree, inside the jail: its code is the agent's."""
    proc = jailed(tree, _argv(target.load().test), timeout_s=timeout_s)
    return Run(proc.returncode == 0, proc.returncode, (proc.stdout + proc.stderr)[-5000:])
