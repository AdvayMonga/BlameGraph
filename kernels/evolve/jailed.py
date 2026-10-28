"""python -m kernels.evolve.jailed CONFIG.json PROGRAM|- STAGE (1, 2, or 3: ncu counters)  (or `- - STAGE`: {config, source} as JSON on stdin)  ->  kernels.evolve.core's JSON, judged in the jail.

The candidate runs as agent code (lab/safety/grader.jailed: a container with no network on Linux, srt elsewhere;
LAB_NO_JAIL=1 waives it). The jail sees a fresh tree holding only the candidate and its config, a read-only copy of
the harness core needs (kernels/, validity.peaks, lab.target, the measured peaks), and the engine's source read-only
when the task's baseline lives there; never this repo, which holds the held-out shapes, the ledger and .env."""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import os
import secrets
import shutil
import sys
import tempfile
from pathlib import Path

from kernels import shared_weights
from kernels.evolve import ncu
from kernels.evolve.verify import verify
from kernels.tasks import load
from lab import engine
from lab.safety import container, grader

ROOT = Path(__file__).resolve().parents[2]
HARNESS = {"kernels": None, "validity": ("__init__.py", "peaks.py"),
           "lab": ("__init__.py", "target.py", "capture.py")}   # capture: a region reads its samples (regions.from_engine)
TIMEOUT_S = float(os.environ.get("KERNEL_EVAL_TIMEOUT_S", "600"))
NCU_STAGE = 3                    # not a judging stage: ncu counters for a program that passed stage 2 (kernels/evolve/ncu.py)
QUICK_STAGE = 4                  # stage 2 on the most-called seen shape only: the funnel's quick speed test
PRECHECK_STAGE = 5               # no GPU attached: the screen, the CUDA C++ build, the compiler's report (precheck.py)
EVAL_DIRS = Path(os.environ.get("KERNEL_EVAL_DIRS") or Path(tempfile.gettempdir()) / "kernel-eval")
CUDA = Path("/usr/local/cuda")
MEMORY = os.environ.get("KERNEL_EVAL_MEMORY", "32g")   # host RAM a judgment may use: a program cannot starve the host


def cuda_home() -> Path | None:
    """The toolkit for the CUDA torch was built with (an extension linked against another runtime does not load:
    Verda's image is CUDA 12.8 beside torch cu130), else /usr/local/cuda; None without nvcc."""
    import torch
    v = torch.version.cuda
    for p in ([Path(f"/usr/local/cuda-{v}")] if v else []) + [CUDA]:
        if (p / "bin" / "nvcc").exists():
            return p.resolve()
    return None


BUILD_FILES = ("build.ninja", ".ninja_deps", ".ninja_log", "main.cpp", "cuda.cu", "lock")


def _open(p: Path) -> None:
    """Give the owner full access to dir `p` again (a program may have taken it away to keep its files)."""
    if p.is_dir() and not p.is_symlink():
        p.chmod(0o700)


def _rm(p: Path) -> None:
    """Remove `p` (a link is unlinked, never followed), every dir under it opened first."""
    if p.is_symlink() or not p.is_dir():
        p.unlink()
        return
    _open(p)
    for root, dirs, _ in os.walk(p):             # top-down: a dir is listed after its parent opened it
        for x in dirs:
            _open(Path(root) / x)
    shutil.rmtree(p)


def _keep_builds_only(tree: Path) -> None:
    """After a judgment: in each extension's build dir keep torch's build files and its .so, nothing else, so no other
    file a program wrote outlives its judgment (e.g. answers kept for the next one)."""
    ext = tree / "torch_extensions"
    if not ext.is_dir() or ext.is_symlink():
        return
    _open(ext)
    for d in ext.iterdir():
        if not d.is_dir() or d.is_symlink():
            d.unlink()
            continue
        _open(d)
        for f in d.iterdir():
            keep = not f.is_symlink() and f.is_file() and (f.name in BUILD_FILES or f.suffix == ".o"
                                                            or f.name == f"{d.name}.so")
            if not keep:
                _rm(f)


def _result(stdout: str, key: str) -> tuple[dict | None, str]:
    """The one result line carrying this judgment's key (core reads the key and removes it before the candidate loads,
    so a line the candidate prints cannot carry it), or None and why."""
    found = []
    for x in stdout.splitlines():
        if x.startswith("{"):
            try:
                d = json.loads(x)
            except json.JSONDecodeError:
                continue
            if isinstance(d, dict) and d.get("result_key") == key:
                found.append(d)
    if len(found) != 1:
        return None, f"{len(found)} result lines carry this judgment's key (one expected)"
    found[0].pop("result_key")
    return found[0], ""


def _eval_dir(source: str, toolkit: Path | None, purpose: str = "") -> tuple[Path, Path]:
    """(the dir, its tree) where this program is judged, the same path every time: torch reuses a C++ build only at
    the absolute path it was built in, so a program's build serves all its stages and re-timings. Keyed by the source
    and the toolkit; everything in the tree but the build dir is removed (links unlinked, never followed) before each
    judgment. A program cannot keep answers there: file reads and writes are refused by the screen."""
    d = EVAL_DIRS / hashlib.sha256(f"{toolkit}\0{purpose}\0{source}".encode()).hexdigest()[:24]
    tree = d / "tree"
    tree.mkdir(parents=True, exist_ok=True)
    _open(tree)
    for x in tree.iterdir():
        if x.name == "torch_extensions" and x.is_dir() and not x.is_symlink():
            continue
        _rm(x)
    shutil.rmtree(d / "harness", ignore_errors=True)
    return d, tree


def stage_harness(dest: Path) -> Path:
    """Copy what kernels.evolve.core imports from this environment into `dest`."""
    for pkg, files in HARNESS.items():
        if files is None:
            shutil.copytree(ROOT / pkg, dest / pkg, ignore=shutil.ignore_patterns("__pycache__"))
            continue
        (dest / pkg).mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copyfile(ROOT / pkg / f, dest / pkg / f)
    peaks = ROOT / "knowledge" / "peaks"
    if peaks.is_dir():
        shutil.copytree(peaks, dest / "knowledge" / "peaks")
    return dest


def _engine_src(task) -> Path | None:
    """The source root the task's baseline imports from, as this process resolves it (None for a toy task)."""
    if not task.entry:
        return None
    spec = importlib.util.find_spec(task.entry.partition(":")[0].split(".")[0])
    return Path(next(iter(spec.submodule_search_locations))).parent if spec and spec.submodule_search_locations else None


def _refuse_writable(holder: Path) -> None:
    """A holder file is unpickled outside the jail: never one under a dir a jail writes (eval trees, jail tmp dirs)."""
    p = holder.resolve()
    for d in [EVAL_DIRS.resolve(), *Path(tempfile.gettempdir()).resolve().glob("jail-*")]:
        if p.is_relative_to(d):
            raise ValueError(f"shared weights file {p} is under {d}, which a jail writes")


def judge_jailed(cfg: dict, source: str, stage: int, timeout_s: float = TIMEOUT_S) -> dict:
    """core's result for `source` under `cfg`, from a jailed process; a crash or timeout scores 0 with its output.
    Stage 3 (ncu) profiles the smallest and the largest seen shape (by their first axis), each in its own jail:
    a batch of 1 and a batch of 512 rarely share a bottleneck."""
    if stage == NCU_STAGE and not cfg.get("_ncu_one"):
        seen = sorted(cfg["seen"], key=lambda s: s[0])
        out = []
        for shape in [seen[0]] + ([seen[-1]] if seen[-1] != seen[0] else []):
            a = judge_jailed({**cfg, "seen": [shape], "_ncu_one": True}, source, stage, timeout_s)["artifacts"]
            if "ncu_missing" in a:
                return {"metrics": {"stage": stage}, "artifacts": a}
            why = a.get("ncu_error") or a.get("shared_weights_changed") or a.get("error", "")    # a void's evidence too
            out.append({"shape": shape, **(json.loads(a["ncu"]) if "ncu" in a else {"error": why[-500:]})})
        return {"metrics": {"stage": stage}, "artifacts": {"ncu": json.dumps({"shapes": out})}}
    if stage == QUICK_STAGE:                     # selection orders seen shapes by calls: the first is the most called
        one = json.dumps(list(cfg["seen"][0]))
        r = judge_jailed({**cfg, "seen": cfg["seen"][:1], "weights": {one: (cfg.get("weights") or {}).get(one, 1)},
                          "ints": {k: v for k, v in (cfg.get("ints") or {}).items() if k == one}}, source, 2, timeout_s)
        return {**r, "metrics": {**r["metrics"], "stage": QUICK_STAGE}}
    fail = {"metrics": {"combined_score": 0.0, "correct": 0.0, "stage": stage}}
    if cfg.get("shared_weights"):
        _refuse_writable(Path(cfg["shared_weights"]))
    home = cuda_home()
    d, tree = _eval_dir(source, home, "gate" if cfg.get("heldout") else "")   # the gate never shares selection's dir
    key = secrets.token_hex(16)
    try:
        harness = stage_harness(d / "harness")
        (tree / "candidate.py").write_text(source)
        (tree / ".result_key").write_text(key)
        cfg = {**cfg, "result_key": str(tree / ".result_key")}
        if stage == 2:                           # outputs to verify outside the jail
            cfg = {**cfg, "record": str(tree / "outputs.pt")}
        (tree / "eval_config.json").write_text(json.dumps(cfg))
        src = _engine_src(load(cfg["task"]))
        path = os.pathsep.join(str(p) for p in (harness, src) if p)
        read = [harness] + ([src] if src else [])
        gpu = None
        if stage == PRECHECK_STAGE:              # no device: built for the GPU's architecture, from the config
            argv = [engine.python(), "-m", "kernels.evolve.precheck", str(tree / "eval_config.json"),
                    str(tree / "candidate.py")]
            gpu = False
        elif stage == NCU_STAGE:                 # ncu around the probe; its install root mounted read-only
            exe = shutil.which("ncu")
            if not exe:
                return {"metrics": {"stage": stage}, "artifacts": {"ncu_missing": "ncu is not installed here"}}
            exe = str(Path(exe).resolve())
            read += [Path(exe).parent, Path(exe).parent.parent] + ([Path("/opt/nvidia")] if Path("/opt/nvidia").exists()
                                                                   else [])
            argv = [exe, *ncu.flags(), engine.python(), "-m", "kernels.evolve.ncu", str(tree / "eval_config.json"),
                    str(tree / "candidate.py")]
        else:
            argv = [engine.python(), "-m", "kernels.evolve.core", str(tree / "eval_config.json"),
                    str(tree / "candidate.py"), str(stage)]
        env = {"PYTHONPATH": path, "TORCH_EXTENSIONS_DIR": str(tree / "torch_extensions"),
               **({"CUDA_HOME": str(home)} if home else {})}
        if cfg.get("arch"):                      # every stage builds for the same architecture: one build serves all
            env["TORCH_CUDA_ARCH_LIST"] = cfg["arch"]
        if stage == PRECHECK_STAGE:
            env["CUDA_VISIBLE_DEVICES"] = ""
        read += [home] if home else []
        read += [Path(cfg["samples"])] if cfg.get("samples") else []                  # a region's captures
        read += [shared_weights._side(Path(cfg["shared_weights"]), k) if k else Path(cfg["shared_weights"])
                 for k in ("", "pristine", "constants")] if cfg.get("shared_weights") else []   # handles + sidecars
        ipc = bool(cfg.get("shared_weights")) and stage != PRECHECK_STAGE     # CUDA IPC to the holder; no GPU there
        sw, changed = cfg.get("shared_weights"), []
        with shared_weights.exclusive(sw) if ipc else contextlib.nullcontext():   # one judgment per holder at a time
            if ipc:
                shared_weights.restore(sw)       # a write left by a judgment killed before its restore
            try:
                p = grader.jailed(tree, argv, timeout_s=timeout_s, env_extra=env, read=read, memory=MEMORY,
                                  **({"gpu": gpu} if gpu is not None else {}), **({"ipc_host": True} if ipc else {}))
            finally:
                if ipc:
                    container.clear_shm()
                    changed = shared_weights.restore(sw)
        _keep_builds_only(tree)                  # the precheck's build is what stage 1 then imports
        if changed:                              # restored before verify reads them
            return {**fail, "artifacts": {"shared_weights_changed": f"the judgment wrote shared weights {changed}; "
                                                                    "void, and they were restored"}}
        if stage == PRECHECK_STAGE:
            return _precheck_result(p, stage, key)
        r, why = _result(p.stdout, key) if stage != NCU_STAGE else (None, "")
        if stage == 2 and p.returncode == 0 and r is not None:
            if r["metrics"].get("correct") == 1.0:
                ok, why = verify(cfg, tree / "outputs.pt", timed=r["metrics"].get("combined_score", 0) > 0)
                gates = {**json.loads(r["artifacts"].get("gates") or "{}"),
                         "verified_outside": {"passed": ok, "reason": why}}
                r["artifacts"]["gates"] = json.dumps(gates, indent=1)
                if not ok:
                    r["metrics"] = {**r["metrics"], "combined_score": 0.0, "correct": 0.0}
            return r
    finally:
        shutil.rmtree(d / "harness", ignore_errors=True)
    if stage == NCU_STAGE:
        if p.returncode != 0:
            return {"metrics": {"stage": stage}, "artifacts": {"ncu_error": (p.stderr or p.stdout)[-2000:]}}
        return {"metrics": {"stage": stage}, "artifacts": {"ncu": json.dumps(ncu.parse(p.stdout))}}
    if p.returncode != 0 or r is None:
        return {**fail, "artifacts": {"error": f"evaluation process exited {p.returncode}"
                                               f"{'; ' + why if p.returncode == 0 else ''}:\n"
                                               f"{(p.stderr or p.stdout)[-3000:]}"}}
    return r


def _precheck_result(p, stage: int, key: str) -> dict:
    """The precheck's keyed line as a stage result: passed (correct 1.0), failed (0.0) or inconclusive (no `correct`),
    with the compiler's report as `compile_report`. A line without the key (printed by the candidate) is ignored."""
    d, why = _result(p.stdout, key) if p.returncode == 0 else (None, "")
    if d is None:
        return {"metrics": {"stage": stage}, "artifacts": {"precheck": f"exited {p.returncode}{'; ' + why if why else ''}"
                                                          f": {(p.stderr or p.stdout)[-1500:]}"}}
    a = {"precheck": d.get("reason", "")}
    if d.get("compile_report") is not None:
        a["compile_report"] = json.dumps(d["compile_report"])
    m = {"stage": stage, **({"correct": 1.0 if d["passed"] else 0.0} if d.get("passed") is not None else {})}
    return {"metrics": m, "artifacts": a}


def main() -> int:
    stage = int(sys.argv[3])
    if sys.argv[1] == "-":                       # `- - STAGE`: {"config", "source"} on stdin (a remote evaluation)
        d = json.loads(sys.stdin.read())
        cfg, source = d["config"], d["source"]
    else:
        cfg = json.loads(Path(sys.argv[1]).read_text())
        source = sys.stdin.read() if sys.argv[2] == "-" else Path(sys.argv[2]).read_text()
    sys.stdout.write("\n" + json.dumps(judge_jailed(cfg, source, stage)) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
