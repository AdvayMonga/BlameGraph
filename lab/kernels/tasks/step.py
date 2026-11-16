"""The whole decode step (`region:embed:argmax`) as a kernel task; its baseline is the engine's megakernel.

Inputs: tokens [B], pos [B], block table [B, cols], K/V [layers, blocks, kv heads, 16, head dim]; the holder's weights
follow them as a dict `w`. Outputs: logits [B, V], next tokens [B], and the K/V after the step as new tensors."""
from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch

from lab.kernels.tasks.regions import ERROR_BUDGET
from lab.kernels.tasks import Task, randn
from lab.profile.capture import STEP, STEP_ENTRY

INPUTS = ("tokens", "pos", "bt", "k_cache", "v_cache")
OUTPUTS = ("logits", "k_cache", "v_cache")      # compared; `next` is checked against the logits and dropped
ROW = ("w.input_norm", "w.qkv", "w.q_norm", "w.k_norm", "w.o_proj", "w.post_norm", "w.router", "w.gate_up",
       "w.gate_up_scale", "w.down", "w.down_scale")      # the megakernel's per-layer pointer table, then K and V
EXPERTS = ("w.gate_up", "w.down")        # the reference widens these per expert used, never whole
MODULE = STEP_ENTRY.partition(":")[0]
STATIONS = "inference_server.models.stations"
START_EDITS = (("__device__ __forceinline__ unsigned long long gtimer() { unsigned long long t; asm volatile(\"mov.u64 "
                "%0, %%globaltimer;\" : \"=l\"(t)); return t; }",
                "__device__ __forceinline__ unsigned long long gtimer() { return 0ull; }"),)   # zero the debug timer


def weights(model) -> tuple[dict, dict]:
    """(tensors, constants) a holder keeps for the step.
    Tensors: each layer's weights as `<i>.w.*`, embed, final_norm, lm_head, inv_freq; constants: cfg, n_layers."""
    st = importlib.import_module(STATIONS)
    m = model.model
    t = {f"{i}.{k}": v for i in range(len(m.layers)) for k, v in st.layer_weights(model, i).items()}
    t.update(embed=m.embed_tokens.weight, final_norm=m.norm.weight, lm_head=model.lm_head.weight,
             inv_freq=m.rope.inv_freq.to(torch.float32).contiguous())
    return t, {"cfg": st.layer_config(model), "n_layers": len(m.layers)}


def _table(w: dict, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    rows = [[w[f"{i}.{n}"].data_ptr() if w[f"{i}.{n}"] is not None else 0 for n in ROW] + [k[i].data_ptr(),
                                                                                         v[i].data_ptr()]
            for i in range(w["n_layers"])]
    return torch.tensor(rows, dtype=torch.int64, device=k.device)


def baseline(tokens, pos, bt, k_cache, v_cache, w):
    """The engine's megakernel on copies of the K/V; returns (logits, next, k, v)."""
    mk = importlib.import_module(MODULE)
    k, v = k_cache.clone(), v_cache.clone()
    kv8 = k.dtype == torch.float8_e4m3fn
    ext = mk._ext(kv8, mk.TMA)
    dev = k.device
    mdl = SimpleNamespace(table=_table(w, k, v), embed=w["embed"], final_norm=w["final_norm"], lm_head=w["lm_head"],
                          inv_freq=w["inv_freq"], scale=w["cfg"].scale, fp8=w["0.w.gate_up_scale"] is not None,
                          kv8=kv8, f32=torch.empty(ext.scratch_f32_elems(), dtype=torch.float32, device=dev),
                          bf16=torch.empty(ext.scratch_bf16_elems(), dtype=torch.bfloat16, device=dev),
                          i32=torch.empty(ext.scratch_i32_elems(), dtype=torch.int32, device=dev))
    logits, nxt = mk.decode_step(mdl, tokens, pos, bt)
    return logits, nxt, k, v


def reference(w: dict):
    """The step in fp32 through the engine's stations in torch, every layer in turn; outputs in the baseline's dtypes."""
    def run(tokens, pos, bt, k_cache, v_cache):
        st = importlib.import_module(STATIONS)
        from inference_server.models.paged_kv_cache import step_inputs
        cfg, stations = w["cfg"], st.reference()
        h = w["embed"][tokens.long()].float()[:, None, :]
        f = pos[:, None, None].float() * w["inv_freq"].float()[None, None, :]
        cos, sin = torch.cat([f, f], -1).cos(), torch.cat([f, f], -1).sin()
        ctx = {**step_inputs(bt, pos, k_cache.shape[3]), "cos": cos, "sin": sin, "cfg": cfg}
        ks, vs = [], []
        for i in range(w["n_layers"]):
            lw = {k: w[f"{i}.{k}"] for k in ROW}
            lw = {k: x.float() if isinstance(x, torch.Tensor) and k not in EXPERTS else x for k, x in lw.items()}
            c = {**ctx, **lw, "h": h, "k_cache": k_cache[i].float(), "v_cache": v_cache[i].float()}
            for s in stations:
                c.update(s.fn(c))
            h = c["h"]
            ks.append(c["k_cache"])
            vs.append(c["v_cache"])
        x = h[:, 0]
        n = w["final_norm"].float() * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + cfg.eps))
        logits = n @ w["lm_head"].float().T
        return (logits.to(torch.bfloat16), torch.stack(ks).to(k_cache.dtype),
                torch.stack(vs).to(v_cache.dtype))
    return run


def _cost(w: dict):
    """A cost function giving (0, bytes): every non-expert weight, each layer's top-k experts, and the K/V given."""
    def size(x):
        return x.numel() * x.element_size() if isinstance(x, torch.Tensor) else 0
    cfg = w["cfg"]
    fixed = sum(size(x) for k, x in w.items() if k.rsplit(".", 1)[-1] not in ("gate_up", "down", "embed"))
    experts = sum(size(w[f"{i}.{k}"]) for i in range(w["n_layers"]) for k in EXPERTS)
    least = fixed + experts * min(cfg.top_k, cfg.num_experts) // cfg.num_experts

    def cost(shape, dtype, inputs=None):
        return 0.0, least + sum(size(x) for x in (inputs or ())[3:5])
    return cost


def judged(out):
    """(logits, k, v) of a program's (logits, next, k, v); raises unless `next` is a greatest logit of its row."""
    logits, nxt, k, v = out
    if nxt.shape != logits.shape[:1] or bool(((nxt < 0) | (nxt >= logits.shape[-1])).any()):
        raise ValueError("`next` is not one token id per row")
    top, best = logits.gather(-1, nxt.long()[:, None])[:, 0], logits.amax(-1)
    if not bool(((top == best) | (top.isnan() & best.isnan())).all()):
        raise ValueError("`next` is not a greatest logit of its row")
    return logits, k, v


def _baseline(*xs):
    """The baseline's compared outputs; its `next` is not checked."""
    logits, _, k, v = baseline(*xs)
    return logits, k, v


def task(samples: str | None = None, shared_weights: str | None = None) -> Task:
    """The step as a task: inputs from `samples` (a step capture dir), weights from the `shared_weights` holder.
    Seed 0 is the capture; later seeds perturb the K/V."""
    from lab.kernels.tasks.shared_weights import attach
    from lab.profile import capture
    w = attach(Path(shared_weights)) if shared_weights else {}
    calls = [c for c in capture.load(Path(samples)) if c.get("fn") == STEP] if samples else []
    by_shape: dict[tuple, Path] = {}
    for c in calls:
        if c.get("sample"):
            by_shape.setdefault(_shape(c), Path(samples) / "samples" / c["sample"])

    def make_inputs(shape, dtype, device, g):
        got = by_shape.get(tuple(shape))
        if got is None:
            raise ValueError(f"no captured sample of the decode step at shape {shape}")
        r = torch.load(got, weights_only=True)["reads"]
        xs = [r[n].to(device) for n in INPUTS]
        if g.initial_seed():                     # seed 0: the captured step
            xs[3:5] = [(x.float() * (1 + 1e-2 * randn(x.shape, g, device, torch.float32))).to(x.dtype) for x in xs[3:5]]
        return tuple(xs)

    def shape_of(call):
        return _shape(call) if call.get("fn") == STEP and _shape(call) in by_shape else None
    ok = bool(w)
    return Task(name=STEP, entry=STEP_ENTRY, reference=reference(w) if ok else _missing, make_inputs=make_inputs,
                shape_of=shape_of, cost=_cost(w) if ok else _missing, call=lambda fn, *xs: fn(*xs, dict(w)), judged=judged,
                heldout_axes=(0, 1), edge_shapes=lambda shape: [], layouts=(), nonfinite=False, error_budget=ERROR_BUDGET,
                baseline_fn=_baseline, watch=_watch(), capture_fn=STEP, seen_default=tuple(by_shape),
                start_program=(lambda: start_program()) if ok else None)


def _missing(*_a, **_k):
    raise ValueError("the decode step's reference and cost need its weights: a holder (`shared_weights`)")


def _shape(call: dict) -> tuple[int, int]:
    """A step's shape: (rows, block table width)."""
    r = call.get("reads") or {}
    return r["tokens"]["shape"][0], r["bt"]["shape"][1]


def _watch() -> tuple[str, ...]:
    from lab.kernels.tasks.regions import _importable
    return tuple(sorted({__name__, MODULE, *_importable((STATIONS, "inference_server.models.paged_kv_cache"))}))


def start_program() -> str:
    """The starting program: the engine's megakernel source as a literal, built with load_inline under its flags."""
    mk = importlib.import_module(MODULE)
    d = Path(importlib.util.find_spec(MODULE).origin).parent
    src = (d / "decode_step.cu").read_text()
    if (d / "common.cuh").exists():              # inline the shared header
        src = src.replace('#include "common.cuh"', (d / "common.cuh").read_text().replace("#pragma once", ""), 1)
    for old, new in START_EDITS:
        if src.count(old) != 1:
            raise ValueError(f"{STEP}: a start edit no longer matches decode_step.cu: {old[:60]!r}")
        src = src.replace(old, new)
    if "'''" in src or src.endswith("\\"):
        raise ValueError(f"{STEP}: decode_step.cu cannot be written as a raw string literal")
    bps = 1 if mk.TMA else mk.BPS
    flags = ["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo", f"-DMEGA_TMA={int(mk.TMA)}",
             f"-DMEGA_BPS={bps}", f"-DMEGA_NST={mk.NST}", f"-DMEGA_CHUNK={mk.CHUNK}", f"-DMEGA_PRIME={int(mk.PRIME)}",
             f"-DMEGA_THREADS={mk.THREADS}", f"-DMEGA_RING_KB={mk.RING_KB}", f"-DMEGA_NOMATH={int(mk.NOMATH)}"]
    return f'''# EVOLVE-BLOCK-START
import torch
from torch.utils.cpp_extension import load_inline

CUDA_SRC = r\'\'\'{src}\'\'\'
FLAGS = {flags!r}
TMA = {bool(mk.TMA)!r}
ROW = {ROW!r}
_EXT = {{}}


def _ext(kv8):
    if kv8 not in _EXT:
        _EXT[kv8] = load_inline(name=f"step_kv8{{int(kv8)}}", cpp_sources="", cuda_sources=CUDA_SRC,
                                extra_cuda_cflags=FLAGS + [f"-DMEGA_KV8={{int(kv8)}}"])
    return _EXT[kv8]


def kernel(tokens, pos, bt, k_cache, v_cache, w):
    k, v = k_cache.clone(), v_cache.clone()
    kv8 = k.dtype == torch.float8_e4m3fn
    e, dev = _ext(kv8), k.device
    rows = [[w[f"{{i}}.{{n}}"].data_ptr() if w[f"{{i}}.{{n}}"] is not None else 0 for n in ROW]
            + [k[i].data_ptr(), v[i].data_ptr()] for i in range(w["n_layers"])]
    table = torch.tensor(rows, dtype=torch.int64, device=dev)
    f32 = torch.empty(e.scratch_f32_elems(), dtype=torch.float32, device=dev)
    bf16 = torch.empty(e.scratch_bf16_elems(), dtype=torch.bfloat16, device=dev)
    i32 = torch.empty(e.scratch_i32_elems(), dtype=torch.int32, device=dev)
    times = torch.empty(0, dtype=torch.int64, device=dev)
    logits, nxt = e.decode_step(tokens.contiguous(), pos.contiguous(), bt.contiguous(), table, w["embed"],
                                w["final_norm"], w["lm_head"], w["inv_freq"], f32, bf16, i32, float(w["cfg"].scale),
                                w["0.w.gate_up_scale"] is not None, times, 0, kv8, TMA)
    return logits, nxt, k, v
# EVOLVE-BLOCK-END
'''
