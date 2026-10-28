"""Every measured number with its context: % of the physical ceiling (validity/ceiling.py on measured peaks), % of
the reference engine measured the same way, and for parts of a step their share of the whole and the Amdahl gain.
A ceiling or a reference is only shown when its config fingerprint matches the measurement's; otherwise the field
says why it is missing. Facts for the agent, never a judgement."""
from __future__ import annotations

import csv
import json
import re
import statistics
from pathlib import Path

from lab import target
from regimes.suite import NO_PRICE, usd_per_m
from validity import ceiling, peaks
from validity.ceiling import ModelSpec, Shape

REFERENCE_DIR = target.ENV_ROOT / "knowledge" / "reference"
FINGERPRINT_KEYS = ("model", "w_dtype", "kv_dtype", "max_model_len", "tier", "split", "corpus_version", "gpu")
DETERMINISM_ENV = ("VLLM_BATCH_INVARIANT",)
DETERMINISM_FLAGS = ("--enable-deterministic-inference",)


# -- model config -----------------------------------------------------------------------
def model_config(model: str) -> dict | None:
    """The model's HF config.json from the local hub cache, else the Hub; None offline without a cache."""
    from lab.safety.grader import HF_HUB
    cached = sorted((HF_HUB / f"models--{model.replace('/', '--')}" / "snapshots").glob("*/config.json"))
    if cached:
        return json.loads(cached[-1].read_text())
    try:
        from huggingface_hub import hf_hub_download
        return json.loads(Path(hf_hub_download(model, "config.json")).read_text())
    except Exception:
        return None


def model_spec(t: target.Target) -> ModelSpec | None:
    cfg = model_config(t.model)
    return ModelSpec.from_hf_config(cfg, t.model) if cfg else None


def dtypes(t: target.Target) -> tuple[str, str]:
    """The weight and KV dtypes the target declares (`[model] dtype`, `kv_dtype`; bf16 when unset). The lab cannot see
    what the agent's engine actually runs, so a quantized engine is measured against these declared ceilings."""
    m = t.raw.get("model") or {}
    w = ceiling.dtype(m.get("dtype", "bf16"))
    return w, ceiling.dtype(m.get("kv_dtype", w))


def perf_reference(t: target.Target) -> target.Server | None:
    """The reference server as a performance comparison: the correctness reference without its determinism switches
    (batch-invariant vLLM, deterministic SGLang), which are slower by design."""
    import dataclasses
    if t.reference is None:
        return None
    env = {k: v for k, v in t.reference.env.items() if k not in DETERMINISM_ENV}
    launch = " ".join(w for w in t.reference.launch.split() if w not in DETERMINISM_FLAGS)
    return dataclasses.replace(t.reference, env=env, launch=launch)


# -- fingerprint --------------------------------------------------------------------------
def fingerprint(t: target.Target, gpu: dict | None, tier: str, split: str, seed: int | None = None) -> dict:
    from regimes.workload import corpus_version
    w, kv = dtypes(t)
    return {"model": t.model, "w_dtype": w, "kv_dtype": kv, "max_model_len": t.max_model_len, "tier": tier,
            "split": split, "corpus_version": corpus_version(t.corpus_dir) if t.corpus_dir.is_dir() else None,
            "gpu": (gpu or {}).get("name"), "driver": (gpu or {}).get("driver_version"),
            "sm_clock_mhz": (gpu or {}).get("clocks.sm"), "seed": seed}


def same_config(a: dict, b: dict) -> list[str]:
    """The fingerprint keys that differ (empty = comparable). Driver, clocks and seed are reported, not required:
    the seed picks a sample of the same split, it does not change what is measured."""
    return [k for k in FINGERPRINT_KEYS if a.get(k) != b.get(k)]


# -- shapes from client rows --------------------------------------------------------------
def shape_from_rows(rows: list[dict]) -> Shape | None:
    ok = [r for r in rows if r.get("status") == "ok" and r.get("e2e_s") is not None]
    if not ok:
        return None
    prompts = [r["prompt_tokens"] for r in ok if r.get("prompt_tokens")]
    outs = [r.get("n_tokens") or 0 for r in ok]
    start = min(r["scheduled_s"] for r in ok)
    span = max(r["scheduled_s"] + r["e2e_s"] for r in ok) - start
    conc = sum(r["e2e_s"] for r in ok) / span if span > 0 else 1.0
    if not prompts:
        return None
    p99 = sorted(prompts)[min(len(prompts) - 1, int(0.99 * len(prompts)))]
    reported = [r["prompt_tokens"] - r["cached_tokens"] for r in ok
                if r.get("prompt_tokens") and r.get("cached_tokens") is not None]
    uncached = statistics.fmean(reported) if len(reported) == len(ok) else None
    return Shape(statistics.fmean(prompts), float(p99), statistics.fmean(outs), max(1.0, conc), uncached)


# -- reference numbers -----------------------------------------------------------------------
def reference_path(t: target.Target, regime: str) -> Path:
    return REFERENCE_DIR / t.name / f"{regime}.json"


def write_reference(t: target.Target, regime: str, result: dict, fp: dict, rows: list[dict] | None = None) -> Path:
    """One regime of the reference server, measured by the same clients, stored with its fingerprint."""
    p = reference_path(t, regime)
    p.parent.mkdir(parents=True, exist_ok=True)
    srv = perf_reference(t)
    d = {"target": t.name, "server": "reference", "launch": srv.launch if srv else None, "env": srv.env if srv else None,
         "regime": regime, "objective": result["objective"],
         "value": result["value"] if result.get("valid") else None, "better": result["better"],
         "valid": result.get("valid"), "summary": result.get("summary"), "fingerprint": fp}
    sh = shape_from_rows(rows) if rows else None
    if sh:
        d["shape"] = sh.__dict__
    p.write_text(json.dumps(d, indent=1, default=str) + "\n")
    return p


def reference_for(t: target.Target, regime: str, fp: dict) -> tuple[float | None, str | None]:
    """(value, why_missing): the reference's objective for this regime, only under the same fingerprint."""
    p = reference_path(t, regime)
    if not p.exists():
        return None, "no reference measurement of this regime on record"
    d = json.loads(p.read_text())
    diff = same_config(d.get("fingerprint") or {}, fp)
    if diff:
        return None, f"reference measured under a different config: {', '.join(diff)}"
    return d.get("value"), None if d.get("value") is not None else "reference measurement was invalid"


# -- regime results ----------------------------------------------------------------------------
def calibrate_results(results: list[dict], rows: list[dict], t: target.Target, gpu: dict | None, tier: str,
                      split: str, seed: int | None = None) -> dict:
    """{regime: {value, ceiling, pct_of_ceiling, reference, pct_of_reference, ...}} for a bench's results. Never
    raises: a calibration that fails says why instead of costing the measurement it describes."""
    try:
        return _calibrate_results(results, rows, t, gpu, tier, split, seed)
    except Exception as e:                       # noqa: BLE001
        return {"error": f"calibration failed: {type(e).__name__}: {e}"}


def cost(r: dict, value: float | None, usd_per_hour: float | None) -> dict:
    """A regime's $ facts on one GPU at its price: per 1M good output tokens and per good request (per session for
    shared_prefix_multi_turn), from the measured summary; else `cost_missing` says why."""
    s = r.get("summary") or {}
    if usd_per_hour is None:
        return {"cost_missing": NO_PRICE}
    if value is None or not s:
        return {"cost_missing": "no valid measurement"}
    out = {"gpu_usd_per_hour": usd_per_hour, "usd_per_m_output_tokens": usd_per_m(usd_per_hour, s.get("good_output_tok_per_s")),
           "usd_per_request": usd_per_hour / 3600 / s["good_req_per_s"] if s.get("good_req_per_s") else None}
    if r["regime"] == "shared_prefix_multi_turn":
        out["usd_per_session"] = usd_per_hour / 3600 / value
    return out


def _calibrate_results(results, rows, t, gpu, tier, split, seed):
    from lab.budget import gpu_rate
    rate, known = gpu_rate(t)
    price = rate if known else None
    fp = fingerprint(t, gpu, tier, split, seed)
    hw = peaks.load(fp["gpu"], fp.get("driver")) if fp["gpu"] else None
    spec = model_spec(t)
    w, kv = dtypes(t)
    out = {}
    for r in results:
        name, obj = r["regime"], r["objective"]
        value = r["value"] if r.get("valid") else None
        row = {"objective": obj, "value": value, "better": r["better"], "fingerprint": fp}
        if value is None:
            row["value_missing"] = r.get("value_missing") or ("the measurement was invalid: "
                                                               + "; ".join(r.get("invalid_reasons") or []))
        row.update(cost(r, value, price))
        shape = shape_from_rows([x for x in rows if x.get("regime") == name and x.get("profile") == r.get("profile")])
        if hw is None:
            row["ceiling"], row["ceiling_missing"] = None, "no measured or spec-sheet peaks for this GPU"
        elif spec is None:
            row["ceiling"], row["ceiling_missing"] = None, "model config unavailable"
        elif shape is None:
            row["ceiling"], row["ceiling_missing"] = None, "no completed requests with prompt token counts"
        else:
            limits = {"single_stream": t.interactive, "batch": t.batch}.get(name, t.conversational).__dict__
            c = ceiling.regime_ceiling(obj, spec, hw, shape, limits, w, kv, price)
            if c is None and obj == "usd_per_m_output_tokens" and price is None:
                row["ceiling"], row["ceiling_missing"] = None, NO_PRICE
            elif c is None:
                row["ceiling"], row["ceiling_missing"] = None, f"no physical bound for {obj}"
            else:
                row.update(ceiling=c["ceiling"], pct_of_ceiling=ceiling.pct_of(value, c["ceiling"], r["better"]),
                           bound=c["bound"], ceiling_basis=c["basis"], peaks_source=hw.source,
                           shape=shape.__dict__)
        ref, why = reference_for(t, name, fp)
        row["reference"] = ref
        if why:
            row["reference_missing"] = why
        else:
            row["pct_of_reference"] = ceiling.pct_of(value, ref, r["better"])
        out[name] = row
    return out


# -- kernels and steps from a profile bundle -----------------------------------------------------
def calibrate_kernels(bundle: Path, gpu: dict | None, top: int = 30) -> dict | None:
    """From a torch-profiler bundle: each kernel's share of the profiled window (the Amdahl denominator)."""
    kj, meta = bundle / "kernels.json", bundle / "meta.json"
    if not kj.exists() or not meta.exists():
        return None
    window_s = (json.loads(meta.read_text()).get("window") or {}).get("wall_s")
    if not window_s:
        return None
    rows = [{"name": k["name"], "count": k["count"], "total_us": k["total_us"],
             "share_of_window": k["total_us"] / (window_s * 1e6)} for k in json.loads(kj.read_text())["kernels"][:top]]
    return {"window_s": window_s, "kernel_share_total": sum(r["share_of_window"] for r in rows), "kernels": rows,
            "gpu": (gpu or {}).get("name")}


def calibrate_ncu(path: Path, gpu: dict | None) -> dict | None:
    """From `ncu --csv --page raw`: each profiled launch's achieved DRAM rate as % of the measured bandwidth peak."""
    if not path.exists():
        return None
    rows = list(csv.DictReader(path.open()))
    if len(rows) < 2:
        return None
    units, data = rows[0], rows[1:]
    hw = peaks.load(gpu["name"], gpu.get("driver_version")) if gpu and gpu.get("name") else None
    col = "dram__bytes.sum.per_second"
    unit = re.fullmatch(r"([KMGTP]?)byte/s(?:econd)?", units.get(col, "").strip())
    scale = {"": 1, "K": 1e3, "M": 1e6, "G": 1e9, "T": 1e12, "P": 1e15}[unit.group(1)] if unit else None
    out = []
    for r in data:
        v = _num(r.get(col))
        bw = v * scale if v is not None and scale is not None else None
        dur = _num(r.get("gpu__time_duration.sum"))
        out.append({"name": r.get("Kernel Name"), "duration": dur,
                    "duration_unit": units.get("gpu__time_duration.sum"), "dram_bytes_s": bw,
                    "pct_of_bandwidth_peak": bw / hw.bw_bytes_s * 100 if bw and hw else None})
    return {"launches": out, "peaks_source": hw.source if hw else None,
            "ceiling_missing": None if hw else "no measured or spec-sheet peaks for this GPU"}


def _window_s(out: Path) -> float | None:
    """The profiled window of the lab.profile bundle inside an instrument's output dir."""
    metas = sorted(out.glob("bundle/*/meta.json"))
    return (json.loads(metas[-1].read_text()).get("window") or {}).get("wall_s") if metas else None


NSYS_COUNTS = ("Instances", "Num Calls", "Count", "Calls")   # the count column's name differs by report and version


def _nsys_rows(path: Path, name: str, top: int) -> tuple[float, int, list[dict]]:
    """An `nsys stats` CSV: total seconds, total count, and its `top` rows by time."""
    if not path.exists():
        return 0.0, 0, []
    rows = [{"name": r.get(name), "total_s": (_num(r.get("Total Time (ns)")) or 0) / 1e9,
             "count": int(next((_num(r[c]) for c in NSYS_COUNTS if r.get(c) not in (None, "")), None) or 0)}
            for r in csv.DictReader(path.open())]
    rows.sort(key=lambda r: -r["total_s"])
    return sum(r["total_s"] for r in rows), sum(r["count"] for r in rows), rows[:top]


def calibrate_nsys(out: Path, ranged: bool, top: int = 15) -> dict | None:
    """From a `trace` output dir (`nsys stats` CSVs): GPU kernel time, launches and memory-copy time, each kernel's
    and CUDA API call's total, and their share of lab.profile's window when the trace covered only that window."""
    kern = out / "stats_cuda_gpu_kern_sum.csv"
    if not kern.exists():
        return None
    kernel_s, launches, kernels = _nsys_rows(kern, "Name", top)
    copy_s, copies, _ = _nsys_rows(out / "stats_cuda_gpu_mem_time_sum.csv", "Operation", 0)
    _, _, api = _nsys_rows(out / "stats_cuda_api_sum.csv", "Name", 10)
    window_s = _window_s(out) if ranged else None
    for k in kernels:
        k["share_of_kernel_time"] = k["total_s"] / kernel_s if kernel_s else None
        k["share_of_window"] = k["total_s"] / window_s if window_s else None
    return {"window_s": window_s, "kernel_s": kernel_s, "launches": launches,
            "kernel_share_of_window": kernel_s / window_s if window_s else None,
            "memcpy_s": copy_s, "memcpy_count": copies, "kernels": kernels, "cuda_api": api,
            "window_missing": None if window_s else ("the trace was not limited to the profiled window (no "
                                                     "--capture-range cudaProfilerApi)" if not ranged else
                                                     "no lab.profile bundle with a window in the output"),
            "basis": "nsys stats over the capture range; kernel times are summed per kernel, so kernels overlapping "
                     "on several streams count once each and the share can exceed the GPU's busy time"}


def calibrate_pyspy(out: Path, top: int = 15) -> dict | None:
    """From a `hostprof` output dir (py-spy's speedscope file): the samples per thread, and the functions holding the
    most samples, at the top of the stack (self) and anywhere in it (total), as shares of all samples."""
    found = sorted(out.glob("bundle/*/pyspy.speedscope.json"))
    if not found:
        return None
    doc = json.loads(found[-1].read_text())
    frames = (doc.get("shared") or {}).get("frames") or []
    threads, own, total, n = [], {}, {}, 0
    for p in doc.get("profiles") or []:
        weights = p.get("weights") or [1] * len(p.get("samples") or [])
        count = 0
        for stack, w in zip(p.get("samples") or [], weights):
            count += w
            if stack:                                # root first, leaf last (speedscope's order)
                own[stack[-1]] = own.get(stack[-1], 0) + w
            for i in set(stack):
                total[i] = total.get(i, 0) + w
        threads.append({"name": p.get("name"), "samples": count})
        n += count
    if not n:
        return None

    def ranked(counts: dict) -> list[dict]:
        best = sorted(counts.items(), key=lambda kv: -kv[1])[:top]
        return [{"function": frames[i].get("name"), "file": frames[i].get("file"), "line": frames[i].get("line"),
                 "share": c / n} for i, c in best if i < len(frames)]
    for t in threads:
        t["share"] = t["samples"] / n
    return {"samples": n, "window_s": _window_s(out), "threads": sorted(threads, key=lambda t: -t["samples"]),
            "self": ranked(own), "total": ranked(total),
            "basis": "share of all py-spy samples of the engine process (idle threads are not sampled unless "
                     "pyspy_args has --idle); self = the function on top of the stack, total = anywhere in it"}


def _num(s) -> float | None:
    """An ncu CSV cell as a number: thousands separators dropped, 'n/a' and blanks as None."""
    try:
        return float(str(s).replace(",", "")) if s not in (None, "") else None
    except ValueError:
        return None


def facts(t: target.Target, gpu: dict | None) -> dict:
    """The ceiling machinery's inputs, for the brief: peaks, the model's streamed bytes, and the single-stream floors."""
    spec, (w, kv) = model_spec(t), dtypes(t)
    hw = peaks.load(gpu["name"], gpu.get("driver_version")) if gpu and gpu.get("name") else None
    out: dict = {"peaks": hw.__dict__ if hw else None, "model": None}
    if spec:
        out["model"] = {"active_params": spec.active_params, "total_params": spec.total_params,
                        "weight_bytes_per_decode_step_batch1": spec.weight_bytes(1, w),
                        "kv_bytes_per_token": spec.kv_bytes_per_token(kv), "w_dtype": w, "kv_dtype": kv}
    if spec and hw:
        out["floors_s"] = {"decode_step_batch1_ctx1k": ceiling.decode_step_floor(spec, hw, 1, 1024, w, kv)["floor_s"],
                           "prefill_1k_prompt": ceiling.prefill_floor(spec, hw, 1024, w, kv)["floor_s"]}
    return out
