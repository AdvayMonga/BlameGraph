"""python -m kernels.evolve.jailed CONFIG.json PROGRAM|- STAGE (1, 2, or 3: ncu counters)  (or `- - STAGE`: {config, source} as JSON on stdin)  ->  kernels.evolve.core's JSON, judged in the jail.

The candidate runs as agent code (lab/safety/grader.jailed: a container with no network on Linux, srt elsewhere;
LAB_NO_JAIL=1 waives it). The jail sees a fresh tree holding only the candidate and its config, a read-only copy of
the harness core needs (kernels/, validity.peaks, lab.target, the measured peaks), and the engine's source read-only
when the task's baseline lives there; never this repo, which holds the held-out shapes, the ledger and .env."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from kernels.evolve import ncu
from kernels.evolve.verify import verify
from kernels.tasks import load
from lab import engine
from lab.safety import grader

ROOT = Path(__file__).resolve().parents[2]
HARNESS = {"kernels": None, "validity": ("__init__.py", "peaks.py"), "lab": ("__init__.py", "target.py")}
TIMEOUT_S = float(os.environ.get("KERNEL_EVAL_TIMEOUT_S", "600"))
NCU_STAGE = 3                    # not a judging stage: ncu counters for a program that passed stage 2 (kernels/evolve/ncu.py)
EXT_CACHE = Path(os.environ.get("KERNEL_EXT_CACHE") or Path(tempfile.gettempdir()) / "kernel-ext")
EXT_MAX_BYTES = 512 * 2**20      # a build dir larger than this is not kept
CUDA = Path("/usr/local/cuda")


def _builds(source: str, tree: Path) -> tuple[Path, Path]:
    """(the jail's TORCH_EXTENSIONS_DIR inside its tree, this source's cache): a program's C++ builds once for all its
    stages and re-timings. Keyed by the source, so a program only ever reads what an identical program built."""
    ext, cache = tree / "torch_extensions", EXT_CACHE / hashlib.sha256(source.encode()).hexdigest()[:24]
    if cache.is_dir():
        shutil.copytree(cache, ext, symlinks=True)
    return ext, cache


def _keep_builds(ext: Path, cache: Path) -> None:
    """Keep the build dir the jail wrote (links copied as links, never followed on this side), once per source."""
    if cache.exists() or not ext.is_dir():
        return
    files = [f for f in ext.rglob("*") if f.is_file() and not f.is_symlink()]
    if not files or sum(f.stat().st_size for f in files) > EXT_MAX_BYTES:
        return
    tmp = cache.with_name(cache.name + f".{os.getpid()}.tmp")
    shutil.copytree(ext, tmp, symlinks=True)
    tmp.rename(cache)


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
            out.append({"shape": shape, **(json.loads(a["ncu"]) if "ncu" in a else
                                           {"error": a.get("ncu_error", "")[-500:]})})
        return {"metrics": {"stage": stage}, "artifacts": {"ncu": json.dumps({"shapes": out})}}
    fail = {"metrics": {"combined_score": 0.0, "correct": 0.0, "stage": stage}}
    with tempfile.TemporaryDirectory(prefix="kernel-eval-") as d:
        tree, harness = Path(d, "tree"), stage_harness(Path(d, "harness"))
        tree.mkdir()
        (tree / "candidate.py").write_text(source)
        if stage == 2:                           # outputs to verify outside the jail
            cfg = {**cfg, "record": str(tree / "outputs.pt")}
        (tree / "eval_config.json").write_text(json.dumps(cfg))
        ext, cache = _builds(source, tree)
        src = _engine_src(load(cfg["task"]))
        path = os.pathsep.join(str(p) for p in (harness, src) if p)
        read = [harness] + ([src] if src else [])
        if stage == NCU_STAGE:                   # ncu around the probe; its install root mounted read-only
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
        env = {"PYTHONPATH": path, "TORCH_EXTENSIONS_DIR": str(ext),
               **({"CUDA_HOME": str(CUDA.resolve())} if CUDA.exists() else {})}   # mounted at its resolved path
        p = grader.jailed(tree, argv, timeout_s=timeout_s, env_extra=env, read=read)
        _keep_builds(ext, cache)
        lines = [x for x in p.stdout.splitlines() if x.startswith("{")]
        if stage == 2 and p.returncode == 0 and lines:
            r = json.loads(lines[-1])
            if r["metrics"].get("correct") == 1.0:
                ok, why = verify(cfg, tree / "outputs.pt")
                gates = {**json.loads(r["artifacts"].get("gates") or "{}"),
                         "verified_outside": {"passed": ok, "reason": why}}
                r["artifacts"]["gates"] = json.dumps(gates, indent=1)
                if not ok:
                    r["metrics"] = {**r["metrics"], "combined_score": 0.0, "correct": 0.0}
            return r
    if stage == NCU_STAGE:
        if p.returncode != 0:
            return {"metrics": {"stage": stage}, "artifacts": {"ncu_error": (p.stderr or p.stdout)[-2000:]}}
        return {"metrics": {"stage": stage}, "artifacts": {"ncu": json.dumps(ncu.parse(p.stdout))}}
    lines = [x for x in p.stdout.splitlines() if x.startswith("{")]
    if p.returncode != 0 or not lines:
        return {**fail, "artifacts": {"error": f"evaluation process exited {p.returncode}:\n"
                                               f"{(p.stderr or p.stdout)[-3000:]}"}}
    return json.loads(lines[-1])


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
