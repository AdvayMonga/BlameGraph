"""The `kernel_search` tool's pipeline, run on the controller: capture the task's calls, split them seen / held-out,
search, gate the winner on the held-out shapes, return it as a patch. A region is held and judged where the GPU is."""
from __future__ import annotations

import contextlib
import difflib
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Callable

# Capture traffic as (corpus class, concurrency): the default, and each scored regime's.
CAPTURE_SPECS = (("mixed_1", 256), ("mixed_3", 64), ("long_context", 60))
REGIME_CAPTURE = {"single_stream": ("steady_interactive", 1), "saturated": ("mixed_1", 256), "bursty": ("mixed_3", 64),
                  "long_prompt_short_output": ("long_context", 60), "short_prompt_long_output": ("steady_interactive", 64),
                  "shared_prefix_multi_turn": ("mixed_3", 64), "overload": ("mixed_1", 256)}


CAPTURE_TOKENS = 256          # each capture request's answer length


def steady(cls: str, n: int) -> tuple[int, int]:
    """(requests, in flight) for a capture of corpus class `cls` at load `n`: 3x `n` requests at `n` in flight; a class
    with fewer seen requests runs them all at a third as many in flight."""
    from evals import regimes as corpus
    have = len(corpus.load_trace(cls, "seen")[1])
    total = min(3 * n, have)
    return total, max(1, min(n, total // 3))


def specs(regimes: tuple[str, ...] = ()) -> tuple[tuple[str, int], ...]:
    """The capture traffic for the task's scored regimes (each once), else the default."""
    got = tuple(dict.fromkeys(REGIME_CAPTURE[r] for r in regimes if r in REGIME_CAPTURE))
    return got or CAPTURE_SPECS


def capture_env(task) -> dict:
    """The launch env a capture of `task` runs under: the target's [kernels] capture_env, then the task's own."""
    try:
        from evals.targets import target
        env = dict((target.load().raw.get("kernels") or {}).get("capture_env") or {})
    except Exception:                            # noqa: BLE001  no target: the task's own only
        env = {}
    return {**env, **dict(task.capture_env)}


def tree_key(task, tree: Path, spec: tuple) -> str:
    """What a capture depends on: every engine file but the task's own module, and the traffic. Equal keys reuse."""
    import hashlib
    h = hashlib.sha256(json.dumps(spec).encode())
    skip = None if task.name.startswith("region:") else module_file(task, tree)     # a region's code is the engine's
    for f in sorted(p for p in Path(tree).rglob("*") if p.is_file() and p != skip
                    and not {"_harness", "runs", "__pycache__", ".git"} & set(p.relative_to(tree).parts)):
        h.update(f.relative_to(tree).as_posix().encode() + b"\0" + f.read_bytes())
    return h.hexdigest()
ROOT = Path(__file__).resolve().parents[3]
GATE_TIMEOUT_S = 3600        # seconds for a gate judgment: every seen and held-out shape
HOLD = "lab.kernels.tasks.hold"        # the module a region's hold process runs


def module_file(task, tree: Path) -> Path:
    """The engine module the task's entry point lives in, inside an engine tree."""
    if task.name.startswith("region:"):              # its entry is the stations tuple; a winner ships as an override
        raise ValueError(f"{task.name}: regions have no engine module to start from or patch (`lab.kernels.evolve ship` "
                         "writes a winner into the engine's overrides package)")
    mod = task.entry.partition(":")[0]
    return Path(tree) / "src" / (mod.replace(".", "/") + ".py")


def start_program(task, tree: Path) -> str:
    """The starting program from the engine module in `tree` (the agent's workspace, not a commit)."""
    from lab.kernels.evolve.run import _start
    return _start(task, module_file(task, tree).read_text(), task.entry.partition(":")[2])


def captures(task, tree: Path, profile: Callable[[Path, dict], Path | None], regimes: tuple[str, ...] = (),
             cache: Path | None = None) -> list[Path]:
    """Each capture traffic's `captures/` dir, from `profile(tree, args) -> bundle dir`; `cache` (a JSON map) reuses
    a bundle with an equal `tree_key`."""
    known = json.loads(cache.read_text()) if cache and cache.exists() else {}
    out = []
    env = capture_env(task)
    for cls, n in specs(regimes):
        key = tree_key(task, tree, (task.name, cls, n, sorted(env.items())))
        total, lanes = steady(cls, n)
        bundle = Path(known[key]) if key in known and Path(known[key]).is_dir() else \
            profile(tree, {"capture": [task.capture_fn or task.entry], "corpus_class": cls, "requests": total,
                           "concurrency": lanes, "max_tokens": CAPTURE_TOKENS,
                           **({"capture_task": task.name} if task.capture_env else {})})
        if bundle is not None and (Path(bundle) / "captures").is_dir():
            out.append(Path(bundle) / "captures")
            known[key] = str(bundle)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(known, indent=1))
    return out


def pack(caps: list[Path], dest: Path) -> Path:
    """One samples dir of a region's captures: every call in calls.jsonl, its samples under samples/ renamed per
    capture, and each capture's calls again in traffic<i>/profile-<source>/captures/."""
    from lab.kernels.tasks import shapes
    from lab.profile import capture
    shutil.rmtree(dest, ignore_errors=True)
    (dest / "samples").mkdir(parents=True)
    every = []
    for i, cap in enumerate(caps):
        calls = capture.load(cap)
        for c in calls:
            if c.get("sample"):
                name = f"{i}-{c['sample']}"
                try:
                    os.link(Path(cap) / "samples" / c["sample"], dest / "samples" / name)
                except OSError:                  # another file system
                    shutil.copyfile(Path(cap) / "samples" / c["sample"], dest / "samples" / name)
                c["sample"] = name
        own = dest / f"traffic{i}" / f"profile-{shapes.source(cap)}" / "captures"
        own.mkdir(parents=True)
        (own / "calls.jsonl").write_text("".join(json.dumps(c) + "\n" for c in calls))
        every += calls
    (dest / "calls.jsonl").write_text("".join(json.dumps(c) + "\n" for c in every))
    return dest


@contextlib.contextmanager
def hold(task, caps: list[Path], out: Path, env: dict, worker=None):
    """Hold a region where the evaluator runs, for the block: pack its captures (pushed to `worker`), start the hold
    process there and yield its description: {holder, samples (paths there), shapes, start, stations}."""
    local = pack(caps, Path(out) / "samples")
    there = f"{worker.root()}/local/runs/remote/kernel_search/{Path(out).name}" if worker else str(Path(out).resolve())
    samples = f"{there}/samples"
    if worker:
        worker.push(local, samples)
    args = ["-m", HOLD, "--task", task.name, "--samples", samples, "--out", f"{there}/holder/weights.pkl"]
    if "KERNEL_EVAL_SSH" in env:
        argv = json.loads(env["KERNEL_EVAL_SSH"]) + [f"{env['KERNEL_EVAL_REMOTE']} python {shlex.join(args)}"]
    else:
        from lab.targets import engine
        argv = [engine.python(), *args]
    log = open(Path(out) / "hold.log", "w")
    p = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, cwd=ROOT,
                         env={**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(ROOT),
                                                                                        os.environ.get("PYTHONPATH")]))})
    try:
        line = next((x for x in p.stdout if x.startswith("{")), None)
        if line is None:
            p.wait()
            raise RuntimeError(f"lab.kernels.tasks.hold exited {p.returncode}: {(Path(out) / 'hold.log').read_text()[-2000:]}")
        yield json.loads(line)
    finally:
        p.stdin.close()
        try:
            p.wait(timeout=120)
        except subprocess.TimeoutExpired:
            p.kill()
        log.close()


def remote_env(prefix: list[str], env_dir: str, python: str, env: dict, sudo: bool = False) -> dict:
    """The evaluator's ssh-mode env for a worker host, quoted as `lab.infra.runner.worker.Remote._run` quotes: paths
    raw (`~` and `$HOME` expand there), values in double quotes."""
    for v in env.values():
        if any(c in v for c in '"`\\'):
            raise ValueError(f"unsafe value in the worker's env: {v!r}")
    exports = " ".join(f'{k}="{v}"' for k, v in env.items())
    root = 'sudo -n env HOME="$HOME" PATH="$PATH" ' if sudo else ""
    return {"KERNEL_EVAL_SSH": json.dumps(prefix),
            "KERNEL_EVAL_REMOTE": f"cd {env_dir} && {root}PATH=\"{Path(python).parent}:$PATH\" {exports}"}


def spent(out: Path) -> float:
    """Model dollars a search has spent: its model calls' (calls.jsonl)."""
    f, usd = Path(out) / "calls.jsonl", 0.0
    for line in f.read_text().splitlines() if f.exists() else []:
        try:
            usd += float(json.loads(line).get("usd") or 0.0)
        except ValueError:
            continue
    return round(usd, 4)


def facts(task, kernel_tasks: dict | None, regimes: tuple[str, ...] = ()) -> str:
    """Facts for every prompt of the search: the scored regimes and, from the latest profile (`kernel_tasks`), the
    task's share of the profiled window and the window's speedup if its kernels took no time."""
    out = ["# The engine this kernel runs in"]
    if regimes:
        out.append(f"Scored regimes of the task the engine is optimized for: {', '.join(regimes)}.")
    t = ((kernel_tasks or {}).get("tasks") or {}).get(task.name)
    if t and t.get("share_of_window") is not None:
        s = t["share_of_window"]
        free = t.get("speedup_if_free")
        out.append(f"In the latest profile of the engine (record {kernel_tasks.get('record')}), this kernel's launches "
                   f"took {100 * s:.1f}% of the window ({', '.join(t.get('kernels') or [])}); the window would be "
                   + (f"{free:.3f}x faster if they took no time" if free else "no faster: the share is 1") + ".")
    else:
        out.append("No profile of the engine with this kernel's share of its time is recorded yet.")
    return "\n".join(out)


def search(task, caps: list[Path], start: Path, out: Path, ledger_root: Path, env: dict,
           device: str = "cuda", facts_text: str | None = None, held: dict | None = None) -> dict:
    """The kernel search on the captures (a region: on its shapes, samples and holder, `hold`'s), judged under `env`:
    {best, cost, out, ledger}. A search that stopped in `out` goes on from its last generation."""
    from lab.kernels.search.__main__ import run
    out.mkdir(parents=True, exist_ok=True)
    if facts_text:
        (out / "facts.txt").write_text(facts_text)
    if held:
        (out / "shapes.json").write_text(json.dumps(held["shapes"]))
        given = ["--shapes", str(out / "shapes.json"), "--samples", held["samples"], "--shared-weights", held["holder"]]
    else:
        given = [w for c in caps for w in ("--captures", str(c))]
    argv = ["--task", task.name, "--out", str(out), "--initial", str(start), "--ledger", str(ledger_root),
            "--equal-captures", "--device", device, *(["--facts", str(out / "facts.txt")] if facts_text else []), *given]
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        return run(argv)
    except SystemExit as e:                      # the search's own refusal of its arguments
        raise RuntimeError(f"lab.kernels.search: {e}") from None
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


def freeze(task, caps: list[Path], program: Path, out: Path, env: dict, held: dict | None, device: str) -> Path:
    """The winner with its autotune frozen into a table by key (lab/kernels/eval/freeze.py); `program` itself when
    nothing in it autotunes."""
    from lab.kernels.eval import freeze as frozen_by
    src = Path(program).read_text()
    new, _ = frozen_by.freeze(src, lambda s: _judged(task, caps, program, out, env, held, frozen_by.FREEZE_STAGE, device))
    if new == src:
        return Path(program)
    dest = Path(out) / "frozen_program.py"
    dest.write_text(new)
    return dest


def gate(task, caps: list[Path], program: Path, out: Path, env: dict, held: dict | None, device: str) -> dict:
    """The held-out gate on the winner, judged where the evaluator runs, on `device`."""
    return _judged(task, caps, program, out, env, held, 2, device)


def _judged(task, caps: list[Path], program: Path, out: Path, env: dict, held: dict | None, stage: int,
            device: str) -> dict:
    """`stage` of `program` under the gate's config (every seen and held-out shape) on `device`."""
    from lab.kernels.tasks import shapes as shapes_mod
    from lab.kernels.eval import evaluator
    from lab.kernels.evolve.run import _ints, _target_tolerances, options_from_target, tolerances
    if held:
        sets = {**held["shapes"], "seen": [tuple(s) for s in held["shapes"]["seen"]],
                "heldout": [tuple(s) for s in held["shapes"]["heldout"]]}
    else:
        sets = shapes_mod.split(task, shapes_mod.from_captures(task, caps, equal=True))
        sets["ints"] = shapes_mod.ints_from_captures(task, caps)
    cfg = {"task": task.name, "seen": [list(s) for s in sets["seen"]], "heldout": [list(s) for s in sets["heldout"]],
           "weights": sets.get("calls"), "device": device, "tolerances": tolerances(_target_tolerances()),
           "ints": _ints(sets, ("seen", "heldout")), "options": options_from_target(task), "full": True,
           **({"samples": held["samples"], "shared_weights": held["holder"], "engine": _engine()} if held else {})}
    path = Path(out) / "gate_config.json"
    path.write_text(json.dumps(cfg))
    env = {**env, "KERNEL_EVAL_TIMEOUT_S": str(GATE_TIMEOUT_S)}            # every seen and held-out shape: longer
    if "KERNEL_EVAL_REMOTE" in env:                                         # the worker's jail reads it too
        env["KERNEL_EVAL_REMOTE"] += f' KERNEL_EVAL_TIMEOUT_S="{GATE_TIMEOUT_S}"'
    saved = {k: os.environ.get(k) for k in ("KERNEL_EVAL_CONFIG", *env)}
    os.environ.update({"KERNEL_EVAL_CONFIG": str(path), **env})
    try:
        return evaluator.judge_stage(str(program), stage)
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


def _engine() -> dict:
    """The target's engine, for a region judged where no target is set."""
    from lab.kernels.tasks import regions
    return regions.load_engine().as_dict()


def patch(task, tree: Path, program: str, held: dict | None = None) -> str:
    """The winner as a unified diff of the engine module in `tree`, the task's start edits undone; a region's as its
    override module, checked for overlap with the overrides in `tree`."""
    from lab.kernels.eval.ship import module_source, no_live_autotune, override_keys, region_files, unstart
    if task.name.startswith("region:"):
        from lab.kernels.tasks import regions
        rel_pkg = regions.load_engine().overrides.replace(".", "/")
        pkg = next((p for p in (Path(tree) / "src" / rel_pkg, Path(tree) / rel_pkg) if p.parent.is_dir()),
                   Path(tree) / "src" / rel_pkg)
        rel = pkg.relative_to(tree).as_posix()
        have = [(f"{rel}/{f.name}", f.read_text()) for f in sorted(pkg.glob("*.py"))] if pkg.is_dir() else []
        files = region_files(task, program, rel, [k for f, src in have for k in override_keys(f, src)],
                             held["set"], (pkg / "__init__.py").exists())
        return "".join("".join(difflib.unified_diff([], src.splitlines(True), "/dev/null", f"b/{f}"))
                       for f, src in files.items())
    path = module_file(task, tree)
    old = path.read_text()
    new = no_live_autotune(unstart(task, module_source(program, task.entry.partition(":")[2])), old)
    rel = path.relative_to(tree).as_posix()
    return "".join(difflib.unified_diff(old.splitlines(True), new.splitlines(True), f"a/{rel}", f"b/{rel}"))
