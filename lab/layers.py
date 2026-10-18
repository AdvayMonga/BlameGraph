"""Each layer's share of a profiled window and its distance from a ceiling, as facts (decided 2026-10-14: the agent
chooses the layer; it is given these, never a recommendation). From one `profile` bundle (lab/bundle.py: meta.json's
window, kernels.json, the engine's timeline in events.jsonl, stats.json):

  kernels          CUDA kernel time / window                                   (layer 1)
  model_execution  prefill + decode phase wall time beyond kernel time / window: launches, framework, sampling  (2)
  runtime          admit + evict phase time / window, with the GPU idle share, batch fill, admissions blocked      (3)
  serving          window time outside scheduler steps / window: HTTP, tokenizer, waiting for arrivals            (4)

Distance: layer 2 = the mean decode step against the step floor (validity.ceiling) at the window's mean batch and
context; layer 1 = the kernels' call-weighted % of their roofline bounds when gate results are given
(kernels.share.weighted_pct), else missing and why; layers 3 and 4 have no single ceiling, so the facts beside them
are what a change there moves. Every number names its basis; a missing one says why.

  python -m lab.layers BUNDLE [--target T]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

OUTER_PHASES = ("admit", "prefill_chunk", "evict", "decode")      # the step's own phases; others nest inside these


def _events(bundle: Path, window: dict) -> list[dict]:
    """The timeline events of the profiled window (the timeline covers the whole run, warmup included)."""
    p = bundle / "events.jsonl"
    if not p.exists():
        return []
    lo, hi = float(window["start"]), float(window["end"])
    out = []
    for line in p.read_text().splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if lo <= float(e.get("ts", 0)) <= hi + 1e-6:
            out.append(e)
    return out


def from_bundle(bundle: str | Path, t=None, gpu: dict | None = None, gates: dict | None = None,
                tasks: list | None = None) -> dict | None:
    """The four layers' facts for one bundle; None when it has no window."""
    bundle = Path(bundle)
    try:
        meta = json.loads((bundle / "meta.json").read_text())
    except (OSError, ValueError):
        return None
    window = meta.get("window") or {}
    w = window.get("wall_s")
    if not w:
        return None
    events = _events(bundle, window)
    phases: dict[str, float] = {}
    for e in events:
        if e.get("kind") == "phase" and e.get("name") in OUTER_PHASES:
            phases[e["name"]] = phases.get(e["name"], 0.0) + float(e.get("dur_s") or 0.0)
    steps = [e for e in events if e.get("kind") == "step"]
    step_s = sum(float(e.get("dur_s") or 0.0) for e in steps)
    decoding = [e for e in steps if (e.get("active") or 0) > 0]
    batch_mean = sum(e["active"] for e in decoding) / len(decoding) if decoding else None
    pending_idle = sum(1 for e in steps if (e.get("pending") or 0) > 0 and not e.get("active") and not e.get("prefilling"))
    try:
        kern = json.loads((bundle / "kernels.json").read_text())
        kernel_s, launches = float(kern.get("total_us") or 0.0) / 1e6, kern.get("launches")
    except (OSError, ValueError):
        kernel_s, launches = None, None
    try:
        delta = (json.loads((bundle / "stats.json").read_text()).get("window_delta") or {})
    except (OSError, ValueError):
        delta = {}
    prefill_s, decode_s = phases.get("prefill_chunk", 0.0), phases.get("decode", 0.0)
    admin_s = phases.get("admit", 0.0) + phases.get("evict", 0.0)
    n_decode = delta.get("decode_steps") or len(decoding) or 0
    prompts = [e.get("prompt_tokens") for e in events if e.get("kind") == "enqueue" and e.get("prompt_tokens")]
    if not prompts and (meta.get("workload") or {}).get("prompt_len"):
        prompts = [meta["workload"]["prompt_len"]]
    max_tokens = meta.get("max_tokens") or 0
    context = (sum(prompts) / len(prompts) + max_tokens / 2) if prompts else None
    gpu = gpu or (meta.get("gpu") or {}).get("window_start") or meta.get("gpu")

    def share(x):
        return None if x is None else x / w

    out = {
        "window_s": w, "scheduler_steps": len(steps), "step_share": share(step_s),
        "basis": "shares are of the profiled window's wall time; phases from the engine's timeline, kernel time from "
                 "the profiler (overlapping kernels counted once each, not merged)",
        "kernels": {"share": share(kernel_s), "kernel_s": kernel_s, "launches": launches,
                    "basis": "CUDA kernel time (kernels.json) / window"},
        "model_execution": {"share": share(max(0.0, prefill_s + decode_s - (kernel_s or 0.0))) if kernel_s is not None else None,
                            "prefill_phase_s": prefill_s, "decode_phase_s": decode_s, "decode_steps": n_decode,
                            "decode_step_mean_s": decode_s / n_decode if n_decode and decode_s else None,
                            "basis": "prefill + decode phase wall time beyond kernel time: launches, framework, sampling"},
        "runtime": {"share": share(admin_s), "admit_evict_s": admin_s,
                    "gpu_idle_share": None if kernel_s is None else max(0.0, 1 - kernel_s / w),
                    "batch_mean": batch_mean,
                    "batch_fill": batch_mean / meta["settings"]["max_batch_size"]
                    if batch_mean and (meta.get("settings") or {}).get("max_batch_size") else None,
                    "steps_idle_with_pending": pending_idle, "kv_admit_blocked": delta.get("kv_admit_blocked"),
                    "admitted": delta.get("total_admitted"), "completed": delta.get("total_completed"),
                    "preempted": delta.get("total_preempted"), "expired": delta.get("total_expired"),
                    "basis": "admit + evict phase time / window; the GPU idle share and batch fill are what admission "
                             "and batching leave on the table"},
        "serving": {"share": share(max(0.0, w - step_s)), "outside_steps_s": max(0.0, w - step_s),
                    "basis": "window time outside scheduler steps: HTTP, tokenizer, waiting for arrivals (idle included)"},
    }
    out["kernels"].update(_kernel_distance(gates, tasks))
    out["model_execution"].update(_decode_distance(t, gpu, batch_mean, context, out["model_execution"]["decode_step_mean_s"]))
    out["runtime"]["ceiling_missing"] = "no single runtime ceiling; the facts here are what a scheduler change moves"
    out["serving"]["ceiling_missing"] = "measured per bench: cold_start against its floor, client TTFT minus engine first token"
    return out


def _kernel_distance(gates: dict | None, tasks: list | None) -> dict:
    if not gates:
        return {"pct_of_ceiling": None, "ceiling_missing": "no kernel gate results given (kernels.share.rank joins them)"}
    per = {name: _weighted_pct(g) for name, g in gates.items()}
    known = {k: v for k, v in per.items() if v is not None}
    return {"pct_of_ceiling": sum(known.values()) / len(known) if known else None, "pct_of_roofline_by_task": per,
            "ceiling_basis": "mean over tasks of the call-weighted % of each kernel's roofline bound"}


def _weighted_pct(gate: dict) -> float | None:
    """Call-weighted % of the roofline bound over the shapes a kernel `gate` result timed (as kernels.share.weighted_pct;
    repeated here because the kernels package imports torch, which the lab does not require)."""
    rows = [r for k in ("timings", "heldout_timings") for r in json.loads((gate.get("artifacts") or {}).get(k) or "[]")
            if r.get("pct_of_roofline") is not None]
    w = sum(r.get("weight") or 1 for r in rows)
    return sum((r.get("weight") or 1) * r["pct_of_roofline"] for r in rows) / w if rows else None


def _decode_distance(t, gpu: dict | None, batch: float | None, context: float | None, measured: float | None) -> dict:
    if measured is None:
        return {"pct_of_ceiling": None, "ceiling_missing": "no decode steps in the window"}
    if t is None:
        return {"pct_of_ceiling": None, "ceiling_missing": "no target: the model's spec gives the floor"}
    from lab import calibrate
    from validity import ceiling, peaks
    spec = calibrate.model_spec(t)
    hw = peaks.load(gpu["name"], gpu.get("driver_version")) if gpu and gpu.get("name") else None
    if spec is None or hw is None or context is None or not batch:
        why = ("no model spec" if spec is None else "no measured or spec-sheet peaks for this GPU" if hw is None
               else "no prompt lengths in the window" if context is None else "no decode batch in the window")
        return {"pct_of_ceiling": None, "ceiling_missing": why}
    wd, kd = calibrate.dtypes(t)
    f = ceiling.decode_step_floor(spec, hw, max(1, round(batch)), context, wd, kd)
    return {"pct_of_ceiling": ceiling.pct_of(measured, f["floor_s"], "lower"), "ceiling": f["floor_s"],
            "ceiling_bound": f["bound"],
            "ceiling_basis": f"decode step floor at batch {max(1, round(batch))}, mean context {context:.0f} tokens "
                             f"({wd} weights, {kd} KV), peaks {hw.source}"}


def main(argv: list[str] | None = None) -> int:
    from lab import target
    ap = argparse.ArgumentParser(prog="python -m lab.layers", description=__doc__.split("\n\n")[0])
    target.add_argument(ap)
    ap.add_argument("bundle", type=Path)
    a = ap.parse_args(argv)
    try:
        t = target.load(a.target) if getattr(a, "target", None) else target.load()
    except Exception:                        # noqa: BLE001  shares need no target; only the decode floor does
        t = None
    out = from_bundle(a.bundle, t)
    if out is None:
        print(f"{a.bundle}: no profiled window (meta.json)", file=sys.stderr)
        return 1
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
