"""The kernel layer as one call (the central agent's `kernel_search` tool, decided 2026-10-22): capture the task's
calls under real traffic, split them seen / held-out, search, gate the winner on the held-out shapes, and hand the
engine module back as a patch the agent applies itself. It runs on the controller: the search's model calls stay
here, and its evaluator runs where the GPU is (`remote_env` for a worker host, kernels/evolve/evaluator.py's ssh mode).
A blocking call for now; a parallel job on its own GPU host comes later."""
from __future__ import annotations

import difflib
import json
import os
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
    h, skip = hashlib.sha256(json.dumps(spec).encode()), module_file(task, tree)
    for f in sorted(p for p in Path(tree).rglob("*") if p.is_file() and p != skip
                    and not {"_harness", "runs", "__pycache__", ".git"} & set(p.relative_to(tree).parts)):
        h.update(f.relative_to(tree).as_posix().encode() + b"\0" + f.read_bytes())
    return h.hexdigest()
ROOT = Path(__file__).resolve().parents[1]
GATE_TIMEOUT_S = 3600        # as `kernels.evolve gate`: the held-out shapes on top of the seen ones


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
            profile(tree, {"capture": [task.entry], "corpus_class": cls, "requests": n,
                           **({"capture_task": task.name} if task.capture_env else {})})
        if bundle is not None and (Path(bundle) / "captures").is_dir():
            out.append(Path(bundle) / "captures")
            known[key] = str(bundle)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(known, indent=1))
    return out


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


def search(task, caps: list[Path], start: Path, out: Path, usd: float, evals: int, ledger_root: Path, env: dict,
           funnel: bool = True, device: str = "cuda", facts_text: str | None = None) -> dict:
    """`python -m kernels.search` on the captures: its last JSON line (best, cost, out, ledger record id)."""
    out.mkdir(parents=True, exist_ok=True)
    if facts_text:
        (out / "facts.txt").write_text(facts_text)
    argv = [sys.executable, "-m", "kernels.search", "--task", task.name, "--out", str(out), "--usd", str(usd),
            "--evals", str(evals), "--initial", str(start), "--ledger", str(ledger_root), "--equal-captures",
            "--device", device, *(["--facts", str(out / "facts.txt")] if facts_text else []),
            *[w for c in caps for w in ("--captures", str(c))], *([] if funnel else ["--no-funnel"])]
    p = subprocess.run(argv, capture_output=True, text=True, cwd=ROOT, env={**os.environ, **env})
    lines = [x for x in p.stdout.splitlines() if x.startswith("{")]
    if p.returncode != 0 or not lines:
        raise RuntimeError(f"kernels.search exited {p.returncode}: {(p.stderr or p.stdout)[-2000:]}")
    return json.loads(lines[-1])


def gate(task, caps: list[Path], program: Path, out: Path, env: dict) -> dict:
    """The held-out gate on the winner, judged where the evaluator runs (here, or the worker over ssh)."""
    from kernels import shapes as shapes_mod
    from kernels.evolve import evaluator
    from kernels.evolve.run import _ints, _target_tolerances, options_from_target, tolerances
    sets = shapes_mod.split(task, shapes_mod.from_captures(task, caps, equal=True))
    sets["ints"] = shapes_mod.ints_from_captures(task, caps)
    cfg = {"task": task.name, "seen": [list(s) for s in sets["seen"]], "heldout": [list(s) for s in sets["heldout"]],
           "weights": sets.get("calls"), "device": "cuda", "tolerances": tolerances(_target_tolerances()),
           "ints": _ints(sets, ("seen", "heldout")), "options": options_from_target(task)}
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


def patch(task, tree: Path, program: str) -> str:
    """The winner as a unified diff of the engine module in `tree`, the task's start edits undone."""
    from kernels.evolve.ship import module_source, unstart
    path = module_file(task, tree)
    old = path.read_text()
    new = unstart(task, module_source(program, task.entry.partition(":")[2]))
    rel = path.relative_to(tree).as_posix()
    return "".join(difflib.unified_diff(old.splitlines(True), new.splitlines(True), f"a/{rel}", f"b/{rel}"))
