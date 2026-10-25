"""Where the decode megakernel stops beating the separate kernels (decided 2026-10-23: a measured routing table, by batch
size and context length, instead of MEGAKERNEL_MAX_ROWS). One `lab.profile` run per path, prompt length and batch
size, its requests decoding together: the engine's timeline records each scheduler step's wall time and its batch
(`step` events); only steps at the run's own batch size count (a run's tail, as requests finish, has a few).
Each run has the target's launch env (as served: its CONTEXT_WINDOW sizes the decode graphs, and a row past it decodes
eagerly), then the path's: the megakernel forced for every batch it takes (MEGAKERNEL_MAX_ROWS=16), or off
(MEGAKERNEL=0). Run on the GPU host from this repo with the engine's venv:

  python -m kernels.crossover --out DIR [--batches 1,2,4,8] [--prompt-lens 512,4096,16384] [--max-tokens 256]

DIR gets each run's bundle and `crossover.json`: per prompt length and batch, each path's median step and the faster."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

PATHS = {"megakernel": {"MEGAKERNEL": "1", "MEGAKERNEL_MAX_ROWS": "16"}, "kernels": {"MEGAKERNEL": "0"}}


def steps(bundle: Path) -> dict[int, list[float]]:
    """Decode step seconds by batch size, from a bundle's timeline: steps with no prefill and some rows decoding."""
    out: dict[int, list[float]] = {}
    for f in Path(bundle).rglob("events.jsonl"):
        for line in f.read_text().splitlines():
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if e.get("kind") == "step" and (e.get("active") or 0) > 0 and not e.get("prefilling"):
                out.setdefault(int(e["active"]), []).append(float(e.get("dur_s") or 0.0))
    return out


def table(runs: dict[tuple[str, int, int], dict[int, list[float]]]) -> dict:
    """{prompt_len: {batch: {megakernel_ms, kernels_ms, steps, faster}}} from each run's steps at its own batch."""
    out: dict = {}
    for (path, plen, b), by in runs.items():
        xs = by.get(b) or []
        if not xs:
            continue
        cell = out.setdefault(str(plen), {}).setdefault(str(b), {})
        cell[f"{path}_ms"] = round(statistics.median(xs) * 1e3, 4)
        cell[f"{path}_steps"] = len(xs)
    for cells in out.values():
        for c in cells.values():
            if "megakernel_ms" in c and "kernels_ms" in c:
                c["faster"] = "megakernel" if c["megakernel_ms"] < c["kernels_ms"] else "kernels"
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="kernels.crossover")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batches", default="1,2,4,8")
    ap.add_argument("--prompt-lens", default="512,4096,16384")
    ap.add_argument("--max-tokens", type=int, default=256)
    a = ap.parse_args(argv)
    out = Path(a.out)
    from lab import target
    served = target.load().engine.env
    runs = {}
    for plen in (int(x) for x in a.prompt_lens.split(",")):
        for b in (int(x) for x in a.batches.split(",")):
            for path, env in PATHS.items():
                d = out / f"{path}-{plen}-b{b}"
                if not list(d.rglob("events.jsonl")):
                    p = subprocess.run([sys.executable, "-m", "lab.profile", "--synthetic", "--prompt-len", str(plen),
                                        "--requests", str(b), "--max-tokens", str(a.max_tokens), "--warmup", "0",
                                        "--profiler", "cuda-range", "--out", str(d)],
                                       env={**os.environ, **served, **env}, capture_output=True, text=True)
                    (out / f"{path}-{plen}-b{b}.log").write_text(p.stdout + p.stderr)
                    if p.returncode != 0:
                        print(f"{path} {plen} b{b}: exited {p.returncode}: {p.stderr[-500:]}", file=sys.stderr)
                        continue
                runs[(path, plen, b)] = steps(d)
    t = table(runs)
    (out / "crossover.json").write_text(json.dumps(t, indent=1))
    print(json.dumps(t))
    return 0


if __name__ == "__main__":
    sys.exit(main())
