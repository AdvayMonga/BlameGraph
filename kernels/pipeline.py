"""The kernel layer as one call (the central agent's `kernel_search` tool, decided 2026-10-22): capture the task's
calls under real traffic, split them seen / held-out, search, gate the winner on the held-out shapes, and hand the
engine module back as a patch the agent applies itself. It runs on the controller: the search's model calls stay
here, and its evaluator runs where the GPU is (`remote_env` for a worker host, kernels/evolve/evaluator.py's ssh mode).
A region (`region:FIRST:LAST`, or the whole step) loads only where the GPU is: its captures are packed and sent there,
a hold process there shares its weights and describes it (`hold`, kernels/hold.py), and the controller searches and
gates it from that description. A blocking call for now; a parallel job on its own GPU host comes later."""
from __future__ import annotations

import contextlib
import difflib
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable

# The capture traffic (kernels/evolve/validate_gpu.sh's CAPTURE_SPECS): corpus classes at real concurrencies, for a
# run without a task; with one, each scored regime's traffic from REGIME_CAPTURE (its corpus class, regimes/suite.py,
# at a concurrency like the regime's load: the profile job sends that many requests at once).
CAPTURE_SPECS = (("mixed_1", 256), ("mixed_3", 64), ("long_context", 60))
REGIME_CAPTURE = {"single_stream": ("steady_interactive", 1), "saturated": ("mixed_1", 256), "bursty": ("mixed_3", 64),
                  "long_prompt_short_output": ("long_context", 60), "short_prompt_long_output": ("steady_interactive", 64),
                  "shared_prefix_multi_turn": ("mixed_3", 64), "overload": ("mixed_1", 256)}


def specs(regimes: tuple[str, ...] = ()) -> tuple[tuple[str, int], ...]:
    """The capture traffic for the task's scored regimes (each once), else the default."""
    got = tuple(dict.fromkeys(REGIME_CAPTURE[r] for r in regimes if r in REGIME_CAPTURE))
    return got or CAPTURE_SPECS


def capture_env(task) -> dict:
    """The launch env a capture of `task` runs under (the target's [kernels] capture_env, then the task's own): part
    of what a capture depends on."""
    try:
        from lab import target
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
ROOT = Path(__file__).resolve().parents[1]
GATE_TIMEOUT_S = 3600        # as `kernels.evolve gate`: the held-out shapes on top of the seen ones
HOLD = "kernels.hold"        # the module a region's hold process runs


def module_file(task, tree: Path) -> Path:
    """The engine module the task's entry point lives in, inside an engine tree."""
    if task.name.startswith("region:"):              # its entry is the stations tuple; a winner ships as an override
        raise ValueError(f"{task.name}: regions have no engine module to start from or patch (`kernels.evolve ship` "
                         "writes a winner into the engine's overrides package)")
    mod = task.entry.partition(":")[0]
    return Path(tree) / "src" / (mod.replace(".", "/") + ".py")


def start_program(task, tree: Path) -> str:
    """The starting program from the engine module in `tree` (the agent's workspace, not a commit)."""
    from kernels.evolve.run import _start
    return _start(task, module_file(task, tree).read_text(), task.entry.partition(":")[2])


def captures(task, tree: Path, profile: Callable[[Path, dict], Path | None], regimes: tuple[str, ...] = (),
             cache: Path | None = None) -> list[Path]:
    """Each capture traffic's `captures/` dir, from `profile(tree, args) -> bundle dir` (the `profile` job); one
    already made for the same engine outside the task's module and the same traffic is reused (`cache`: a JSON map)."""
    known = json.loads(cache.read_text()) if cache and cache.exists() else {}
    out = []
    env = capture_env(task)
    for cls, n in specs(regimes):
        key = tree_key(task, tree, (task.name, cls, n, sorted(env.items())))
        bundle = Path(known[key]) if key in known and Path(known[key]).is_dir() else \
            profile(tree, {"capture": [task.capture_fn or task.entry], "corpus_class": cls, "requests": n,
                           **({"capture_task": task.name} if task.capture_env else {})})
        if bundle is not None and (Path(bundle) / "captures").is_dir():
            out.append(Path(bundle) / "captures")
            known[key] = str(bundle)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(known, indent=1))
    return out


def pack(caps: list[Path], dest: Path) -> Path:
    """One samples dir of a region's captures, what its task loads (`samples`): every call in calls.jsonl, its samples
    under samples/ renamed per capture, and each capture's calls again in traffic<i>/profile-<source>/captures/, so the
    shapes are counted per traffic (each weighing the same) and keep their source."""
    from kernels import shapes
    from lab import capture
    (dest / "samples").mkdir(parents=True, exist_ok=True)
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
    """A region held where the evaluator runs, for the block: its captures packed (and pushed to `worker`, a
    lab.worker.Remote), `python -m kernels.hold` started there, and its description yielded: {holder, samples (paths
    there), shapes, start, stations}. The hold process exits when its stdin closes, here or over ssh."""
    local = pack(caps, Path(out) / "samples")
    there = f"{worker.root()}/lab/runs/remote/kernel_search/{Path(out).name}" if worker else str(Path(out).resolve())
    samples = f"{there}/samples"
    if worker:
        worker.push(local, samples)
    args = ["-m", HOLD, "--task", task.name, "--samples", samples, "--out", f"{there}/holder/weights.pkl"]
    if "KERNEL_EVAL_SSH" in env:
        argv = json.loads(env["KERNEL_EVAL_SSH"]) + [f"{env['KERNEL_EVAL_REMOTE']} python {shlex.join(args)}"]
    else:
        from lab import engine
        argv = [engine.python(), *args]
    log = open(Path(out) / "hold.log", "w")
    p = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, cwd=ROOT,
                         env={**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(ROOT),
                                                                                        os.environ.get("PYTHONPATH")]))})
    try:
        line = next((x for x in p.stdout if x.startswith("{")), None)
        if line is None:
            p.wait()
            raise RuntimeError(f"kernels.hold exited {p.returncode}: {(Path(out) / 'hold.log').read_text()[-2000:]}")
        yield json.loads(line)
    finally:
        p.stdin.close()
        try:
            p.wait(timeout=120)
        except subprocess.TimeoutExpired:
            p.kill()
        log.close()


def remote_env(prefix: list[str], env_dir: str, python: str, env: dict, sudo: bool = False) -> dict:
    """The evaluator's ssh mode for a worker host: each judgment runs there, config and source on stdin. Quoted as
    `lab.worker.Remote._run` quotes (paths raw, so `~` and `$HOME` expand there; values in double quotes)."""
    for v in env.values():
        if any(c in v for c in '"`\\'):
            raise ValueError(f"unsafe value in the worker's env: {v!r}")
    exports = " ".join(f'{k}="{v}"' for k, v in env.items())
    root = 'sudo -n env HOME="$HOME" PATH="$PATH" ' if sudo else ""
    return {"KERNEL_EVAL_SSH": json.dumps(prefix),
            "KERNEL_EVAL_REMOTE": f"cd {env_dir} && {root}PATH=\"{Path(python).parent}:$PATH\" {exports}"}


def spent(out: Path) -> float:
    """Model dollars a search spent so far, from its logs (a search that died never printed its cost)."""
    usd = 0.0
    for name, key in (("trace.jsonl", "cost_usd"), ("hypotheses.jsonl", "usd")):
        f = Path(out) / name
        for line in f.read_text().splitlines() if f.exists() else []:
            try:
                v = json.loads(line).get(key)
            except ValueError:
                continue
            usd += float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else 0.0
    return round(usd, 4)


def facts(task, kernel_tasks: dict | None, regimes: tuple[str, ...] = ()) -> str:
    """The bigger picture as facts for every prompt of the search: the regimes the kernel's engine is scored under,
    and from the latest profile (`kernel_tasks`, lab/evaltools.harness_facts) the task's share of the engine's
    profiled window and the window's speedup if its kernels took no time; or that no profile has them."""
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


def search(task, caps: list[Path], start: Path, out: Path, evals: int, ledger_root: Path, env: dict,
           device: str = "cuda", facts_text: str | None = None, held: dict | None = None) -> dict:
    """`python -m kernels.search` on the captures (a region: on its shapes, samples and holder, `hold`'s): its last
    JSON line (best, cost, out, ledger record id)."""
    out.mkdir(parents=True, exist_ok=True)
    if facts_text:
        (out / "facts.txt").write_text(facts_text)
    if held:
        (out / "shapes.json").write_text(json.dumps(held["shapes"]))
        given = ["--shapes", str(out / "shapes.json"), "--samples", held["samples"], "--shared-weights", held["holder"]]
    else:
        given = [w for c in caps for w in ("--captures", str(c))]
    argv = [sys.executable, "-m", "kernels.search", "--task", task.name, "--out", str(out),
            "--evals", str(evals), "--initial", str(start), "--ledger", str(ledger_root), "--equal-captures",
            "--device", device, *(["--facts", str(out / "facts.txt")] if facts_text else []), *given]
    p = subprocess.run(argv, capture_output=True, text=True, cwd=ROOT, env={**os.environ, **env})
    lines = [x for x in p.stdout.splitlines() if x.startswith("{")]
    if p.returncode != 0 or not lines:
        raise RuntimeError(f"kernels.search exited {p.returncode}: {(p.stderr or p.stdout)[-2000:]}")
    return json.loads(lines[-1])


def gate(task, caps: list[Path], program: Path, out: Path, env: dict, held: dict | None = None) -> dict:
    """The held-out gate on the winner, judged where the evaluator runs (here, or the worker over ssh); a region on
    the shapes, samples and holder `hold` described."""
    from kernels import shapes as shapes_mod
    from kernels.evolve import evaluator
    from kernels.evolve.run import _ints, _target_tolerances, options_from_target, tolerances
    if held:
        sets = {**held["shapes"], "seen": [tuple(s) for s in held["shapes"]["seen"]],
                "heldout": [tuple(s) for s in held["shapes"]["heldout"]]}
    else:
        sets = shapes_mod.split(task, shapes_mod.from_captures(task, caps, equal=True))
        sets["ints"] = shapes_mod.ints_from_captures(task, caps)
    cfg = {"task": task.name, "seen": [list(s) for s in sets["seen"]], "heldout": [list(s) for s in sets["heldout"]],
           "weights": sets.get("calls"), "device": "cuda", "tolerances": tolerances(_target_tolerances()),
           "ints": _ints(sets, ("seen", "heldout")), "options": options_from_target(task),
           **({"samples": held["samples"], "shared_weights": held["holder"]} if held else {})}
    path = Path(out) / "gate_config.json"
    path.write_text(json.dumps(cfg))
    env = {**env, "KERNEL_EVAL_TIMEOUT_S": str(GATE_TIMEOUT_S)}            # every seen and held-out shape: longer
    if "KERNEL_EVAL_REMOTE" in env:                                         # the worker's jail reads it too
        env["KERNEL_EVAL_REMOTE"] += f' KERNEL_EVAL_TIMEOUT_S="{GATE_TIMEOUT_S}"'
    saved = {k: os.environ.get(k) for k in ("KERNEL_EVAL_CONFIG", *env)}
    os.environ.update({"KERNEL_EVAL_CONFIG": str(path), **env})
    try:
        return evaluator.judge_stage(str(program), 2)
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


def patch(task, tree: Path, program: str, held: dict | None = None) -> str:
    """The winner as a unified diff of the engine module in `tree`, the task's start edits undone; a region's as the
    override module `kernels.evolve ship` would commit (the overlap checked against the overrides in `tree`, with the
    station names `hold` read from the engine)."""
    from kernels.evolve.ship import module_source, override_keys, region_files, unstart
    if task.name.startswith("region:"):
        from kernels import regions
        pkg = Path(tree) / "src" / regions.OVERRIDES_PACKAGE.replace(".", "/")
        rel = pkg.relative_to(tree).as_posix()
        have = [(f"{rel}/{f.name}", f.read_text()) for f in sorted(pkg.glob("*.py"))] if pkg.is_dir() else []
        files = region_files(task, program, rel, [k for f, src in have for k in override_keys(f, src)],
                             held["stations"], (pkg / "__init__.py").exists())
        return "".join("".join(difflib.unified_diff([], src.splitlines(True), "/dev/null", f"b/{f}"))
                       for f, src in files.items())
    path = module_file(task, tree)
    old = path.read_text()
    new = unstart(task, module_source(program, task.entry.partition(":")[2]))
    rel = path.relative_to(tree).as_posix()
    return "".join(difflib.unified_diff(old.splitlines(True), new.splitlines(True), f"a/{rel}", f"b/{rel}"))
