"""Other implementations of our kernel tasks, timed on our own shapes beside the engine's kernel, and their compiled
code kept for the PTX comparison's cards (kernels/ptx.py). Run on the GPU host in a venv that has the competitors'
libraries (vLLM, FlashInfer) beside this repo and the engine's source:

  uv venv ~/comp && uv pip install --python ~/comp/bin/python vllm flashinfer-python ninja   (FlashInfer JIT-builds;
  ~/comp/bin on PATH)
  PYTHONPATH=~/BlameGraph:~/inference-server/src ~/comp/bin/python -m kernels.competitors \\
      --task rmsnorm --shapes shapes-rmsnorm.json --out DIR

Each competitor is first checked against the task's reference on every shape, inputs from the shapes file's captured
ints where it has them (a wrong one is reported, never timed), then timed in pairs with the engine's kernel on the same
inputs (kernels.time_pair; as CUDA graph replays for a task the engine replays in graphs). DIR gets `timings.json`
(per shape, and our time over each competitor's call-weighted, `fastest`) and one text file of compiled code per
implementation: Triton kernels' PTX from Triton's cache by kernel name, a library's SASS from `cuobjdump -sass` of its
extension, filtered to the kernels that ran. `--start` also writes `start-<competitor>.py`, the search's starting
program built on that competitor's kernel where it is source (vLLM's fused_moe), for `kernels.search --initial`; the
score stays relative to the engine's kernel."""
from __future__ import annotations

import argparse
import functools
import importlib
import inspect
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable

import torch

from kernels.check import TOLERANCES, _compare

ADAPTERS: dict[str, dict[str, Callable]] = {}       # task -> {competitor: make(inputs...) -> outputs}
SASS_NAMES: dict[str, str] = {}                     # competitor -> regex of its device kernels' names
TRITON: set[str] = set()                            # competitors whose kernels are Triton: their PTX, not SASS
CONFIGS: dict[str, dict] = {}                       # competitor -> the launch config it chose (a fact in timings.json)


def adapter(task: str, name: str, sass: str, triton: bool = False):
    def reg(fn):
        ADAPTERS.setdefault(task, {})[name] = fn
        SASS_NAMES[name] = sass
        if triton:
            TRITON.add(name)
        return fn
    return reg


EPS = 1e-6


def _vllm(*mods: str):
    """vLLM's custom ops register when their extension modules are imported (`vllm._C`; from 0.32 on
    `vllm._C_stable_libtorch`, the ops still under torch.ops._C)."""
    for m in mods:
        _module(m)


def _module(m: str):
    import importlib
    try:
        return importlib.import_module(m)
    except ImportError:
        return importlib.import_module(m + "_stable_libtorch")


@adapter("rmsnorm", "vllm.rms_norm", r"rms_norm_kernel")
def _vllm_rms_norm(x, w):
    _vllm("vllm._C")
    out = torch.empty_like(x)
    torch.ops._C.rms_norm(out, x, w, EPS)
    return out


@adapter("add_rmsnorm", "vllm.fused_add_rms_norm", r"fused_add_rms_norm_kernel")
def _vllm_fused_add_rms_norm(h, a, w):
    _vllm("vllm._C")
    res, x = h.clone(), a.clone()                   # in place: res becomes h + a, x its norm
    torch.ops._C.fused_add_rms_norm(x, res, w, EPS)
    return res, x


@adapter("silu_mul", "vllm.silu_and_mul", r"act_and_mul_kernel")
def _vllm_silu_and_mul(x):
    _vllm("vllm._C")
    out = torch.empty((*x.shape[:-1], x.shape[-1] // 2), dtype=x.dtype, device=x.device)
    torch.ops._C.silu_and_mul(out, x)
    return out


@adapter("topk_softmax", "vllm.topk_softmax", r"topkGating|moeSoftmax|topk")
def _vllm_topk_softmax(logits):
    from kernels.tasks._fused import TOP_K
    _vllm("vllm._moe_C")
    m = logits.shape[0]
    w = torch.empty((m, TOP_K), dtype=torch.float32, device=logits.device)
    ids = torch.empty((m, TOP_K), dtype=torch.int32, device=logits.device)
    tok = torch.empty((m, TOP_K), dtype=torch.int32, device=logits.device)
    try:                                         # vLLM 0.32: renormalize, bias, is_padding
        torch.ops._moe_C.topk_softmax(w, ids, tok, logits, True, None, None)
    except (RuntimeError, TypeError):
        try:
            torch.ops._moe_C.topk_softmax(w, ids, tok, logits, True)
        except (RuntimeError, TypeError):
            torch.ops._moe_C.topk_softmax(w, ids, tok, logits)
            w = w / w.sum(dim=-1, keepdim=True)
    return w, ids.long()


@adapter("paged_decode", "flashinfer.batch_decode", r"BatchDecode|batch_decode|SingleDecode")
def _flashinfer_decode(q, k_pool, v_pool, block_tables, seq_lens):
    """FlashInfer's paged decode on our pool (its HND layout: [pages, kv heads, page size, head dim])."""
    import flashinfer
    n, hq, d = q.shape
    hkv, bs = k_pool.shape[1], k_pool.shape[2]
    pages = (seq_lens + bs - 1) // bs
    indptr = torch.zeros(n + 1, dtype=torch.int32, device=q.device)
    indptr[1:] = torch.cumsum(pages, 0)
    indices = torch.cat([block_tables[i, :int(pages[i])] for i in range(n)]).to(torch.int32)
    last = (seq_lens - (pages - 1) * bs).to(torch.int32)
    key = (block_tables.data_ptr(), seq_lens.data_ptr(), n, hq, hkv, d, bs)    # planned once per input set: the plan
    if key not in _FI:                                                          # is host work, not the kernel
        wrap = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            torch.empty(128 * 2**20, dtype=torch.uint8, device=q.device), kv_layout="HND")
        wrap.plan(indptr, indices, last, hq, hkv, d, bs, q_data_type=q.dtype, kv_data_type=k_pool.dtype)
        _FI[key] = wrap
    return _FI[key].run(q, (k_pool, v_pool))


_FI: dict = {}


def _vllm_routing(a, tok, sorted_ids, expert_ids, E: int, block_m: int):
    """Our routing in vLLM's layout. vLLM reads pair p's row at p // top_k, so when every token has the same number of
    pairs (real routing does) pairs are renumbered token by token and the rows are read in place; otherwise top_k = 1
    on rows gathered per call (`gather`). Returns (vLLM's pair of each of ours, top_k, gather, its sorted ids at its
    own block size, expert per block, padded length as a 1-element tensor)."""
    from kernels.tasks.moe_gemm import _block_m, align
    dev, T = a.device, a.shape[0]
    M = tok.numel() or T
    bm = _block_m(sorted_ids, expert_ids)
    rows = expert_ids.long().repeat_interleave(bm)
    ids = sorted_ids[: rows.numel()].long()
    real = (ids < M) & (rows < E)
    pair_e = torch.empty(M, dtype=torch.long, device=dev)
    pair_e[ids[real]] = rows[real]
    top_k, gather, p = 1, False, torch.arange(M, device=dev)
    if tok.numel():
        counts = torch.bincount(tok.long(), minlength=T)
        k = int(counts[0])
        if k > 0 and bool((counts == k).all()):
            top_k = k
            p = torch.empty(M, dtype=torch.long, device=dev)
            p[torch.argsort(tok.long() * M + torch.arange(M, device=dev))] = torch.arange(M, device=dev)
        else:
            gather = True
    pe = torch.empty(M, dtype=torch.long, device=dev)
    pe[p] = pair_e
    order = torch.argsort(pe, stable=True)
    s_ids, e_ids = align(pe[order], E, block_m)
    v_ids = torch.where(s_ids < M, order[s_ids.long().clamp_max(M - 1)].to(torch.int32), s_ids)
    padded = ((e_ids < E).sum() * block_m).to(torch.int32).reshape(1)
    return p, top_k, gather, v_ids.contiguous(), e_ids.contiguous(), padded


_MOE: dict = {}


def _vllm_fused_moe(name: str, w8a8: bool):
    """vLLM's fused_moe Triton kernel at the config vLLM picks for this GPU and expert shape: fp8 weights with bf16
    activations (`int8_w8a16`'s path, which upcasts fp8 the same way: our engine's decode numerics), or W8A8 (its
    activations quantized per token by vLLM's own op inside the call). The routing is translated once per input set,
    outside the timed call (layout, as FlashInfer's plan). Called with timed=True it returns vLLM's own output order."""
    def call(a, tok, w, w_scale, sorted_ids, expert_ids, timed=False):
        import triton.language as tl
        fm = importlib.import_module("vllm.model_executor.layers.fused_moe.fused_moe")
        E, N, K = w.shape
        key = (tok.data_ptr(), sorted_ids.data_ptr(), expert_ids.data_ptr(), tuple(a.shape), tuple(w.shape), w8a8)
        if key not in _MOE:
            k0 = int(torch.bincount(tok.long()).max()) if tok.numel() else 1
            cfg = dict(fm.try_get_optimal_moe_config(tuple(w.shape), (E, K, N // 2), k0,
                                                     "fp8_w8a8" if w8a8 else None, a.shape[0]))
            _MOE[key] = (cfg, *_vllm_routing(a, tok, sorted_ids, expert_ids, E, cfg["BLOCK_SIZE_M"]))
            CONFIGS[name] = cfg
        cfg, p, top_k, gather, v_ids, e_ids, padded = _MOE[key]
        A, A_scale = (a[tok.long()] if gather else a), None
        if w8a8:
            from vllm import _custom_ops as ops
            A, A_scale = ops.scaled_fp8_quant(A, None, use_per_token_if_dynamic=True)
        C = torch.empty((A.shape[0], top_k, N), dtype=a.dtype, device=a.device)
        fm.invoke_fused_moe_triton_kernel(A, w, C, A_scale, w_scale, None, v_ids, e_ids, padded, False, top_k, cfg,
                                          tl.bfloat16, w8a8, False, not w8a8, False, w8a8, None)
        return C if timed else C.view(-1, N)[p]
    return call


adapter("moe_gemm_fp8", "vllm.fused_moe", r"fused_moe_kernel", triton=True)(_vllm_fused_moe("vllm.fused_moe", False))
adapter("moe_gemm_fp8", "vllm.fused_moe_w8a8", r"fused_moe_kernel", triton=True)(
    _vllm_fused_moe("vllm.fused_moe_w8a8", True))


# -- starting the search from a competitor: its kernel ported into the engine module ------------------------------
# Exact edits to vLLM 0.32's fused_moe_kernel so it reads our routing: rows through `tok` (vLLM reads pair // top_k),
# blocks past the last expert end (ours carry E, vLLM's a padded length it loads), our block size. Each must apply once.
FUSED_MOE_EDITS = (
    ("    a_ptr,\n    b_ptr,", "    a_ptr,\n    tok_ptr,\n    b_ptr,"),
    ("    USE_TD: tl.constexpr = False,\n):",
     "    HAS_TOK: tl.constexpr = False,\n    NUM_EXPERTS: tl.constexpr = 0,\n    USE_TD: tl.constexpr = False,\n):"),
    ("    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)\n",
     "    if NUM_EXPERTS > 0:\n        num_tokens_post_padded = EM\n    else:\n"
     "        num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)\n"),
    ("    token_mask = offs_token < num_valid_tokens\n",
     "    token_mask = offs_token < num_valid_tokens\n    if HAS_TOK:\n"
     "        offs_row = tl.load(tok_ptr + offs_token, mask=token_mask, other=0).to(tl.int64) * top_k\n"
     "    else:\n        offs_row = offs_token\n"),
    ("    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)\n",
     "    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)\n    if NUM_EXPERTS > 0:\n"
     "        if off_experts >= NUM_EXPERTS:\n            return\n"),
    ("offs_token[:, None] // top_k * stride_am", "offs_row[:, None] // top_k * stride_am"),
    ("offs_token[None, :] // top_k * stride_am", "offs_row[None, :] // top_k * stride_am"),
)

FUSED_MOE_WRAPPER = '''

_moe_gemm_engine = moe_gemm


def moe_gemm(a, tok, w, sorted_ids, expert_ids, M, block_m=16, split_k=None, w_scale=None, a_scale=None, tile=None):
    """fp8 weights with bf16 rows on vLLM's kernel at its tuned config for the token count (block rows: ours); the
    engine's kernel for the other paths."""
    if w.dtype != torch.float8_e4m3fn or a_scale is not None or tile is not None:
        return _moe_gemm_engine(a, tok, w, sorted_ids, expert_ids, M, block_m, split_k, w_scale, a_scale, tile)
    E, N, K = w.shape
    cfg = _VLLM_CONFIGS[min(_VLLM_CONFIGS, key=lambda t: abs(t - a.shape[0]))]
    C = torch.empty(M, N, dtype=a.dtype, device=a.device)
    EM = sorted_ids.numel()
    grid = (triton.cdiv(EM, block_m) * triton.cdiv(N, cfg["BLOCK_SIZE_N"]),)
    fused_moe_kernel[grid](
        a_ptr=a, tok_ptr=tok if tok is not None else a, b_ptr=w, c_ptr=C, b_bias_ptr=None, a_scale_ptr=None,
        b_scale_ptr=w_scale, topk_weights_ptr=None, sorted_token_ids_ptr=sorted_ids, expert_ids_ptr=expert_ids,
        num_tokens_post_padded_ptr=None, N=N, K=K, EM=EM, num_valid_tokens=M, stride_am=a.stride(0),
        stride_ak=a.stride(1), stride_be=w.stride(0), stride_bk=w.stride(2), stride_bn=w.stride(1),
        stride_cm=C.stride(0), stride_cn=C.stride(1), stride_asm=0, stride_ask=0, stride_bse=w_scale.stride(0),
        stride_bsk=0, stride_bsn=w_scale.stride(1), stride_bbe=0, stride_bbn=0, group_n=0, group_k=0,
        naive_block_assignment=False, BLOCK_SIZE_M=block_m, BLOCK_SIZE_N=cfg["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=cfg["BLOCK_SIZE_K"], GROUP_SIZE_M=cfg["GROUP_SIZE_M"], SPLIT_K=1, MUL_ROUTED_WEIGHT=False,
        top_k=1, compute_type=tl.bfloat16, use_fp8_w8a8=False, use_int8_w8a8=False, use_int8_w8a16=True,
        per_channel_quant=False, HAS_BIAS=False, SWAP_AB=False, HAS_TOK=tok is not None, NUM_EXPERTS=E,
        num_warps=cfg.get("num_warps", 4), num_stages=cfg.get("num_stages", 3))
    return C
'''


def port_fused_moe(program: str, kernel_src: str, zeros_src: str, configs: dict, version: str) -> str:
    """The starting program `program` (the engine module) with vLLM's fused_moe kernel ported in and `moe_gemm`
    routed to it for the decode path's fp8 weights; `configs` = vLLM's tuned configs {tokens: config}."""
    for old, new in FUSED_MOE_EDITS:
        if kernel_src.count(old) != 1:
            raise ValueError(f"vLLM's fused_moe_kernel changed: an edit no longer applies once: {old[:60]!r}")
        kernel_src = kernel_src.replace(old, new)
    block = (f"\n\n# vLLM {version} fused_moe_kernel (Apache-2.0), ported: rows through tok, our block layout.\n"
             f"_VLLM_CONFIGS = {json.dumps({int(k): v for k, v in configs.items()})}\n\n\n"
             f"{zeros_src}\n\n{kernel_src}{FUSED_MOE_WRAPPER}")
    end = "\n# EVOLVE-BLOCK-END\n"
    if program.count(end) != 1:
        raise ValueError("the starting program has no single EVOLVE-BLOCK-END")
    return program.replace(end, block + end)


def start_program(task, name: str) -> str | None:
    """The starting program built on competitor `name`'s kernel, or None when it has no port (a library's compiled
    code cannot be evolved as source)."""
    if name != "vllm.fused_moe" or task.name != "moe_gemm_fp8":
        return None
    import vllm
    from kernels.evolve.run import initial_program
    fm = importlib.import_module("vllm.model_executor.layers.fused_moe.fused_moe")
    src = inspect.getsource(fm.fused_moe_kernel.fn)
    zeros = inspect.getsource(fm.write_zeros_to_output.fn)
    E, N, K = 128, 1536, 2048                                    # the target's gate_up projection
    configs = fm.get_moe_configs(E, N // 2, None) or {1: fm.get_default_config(1, E, N // 2, K, 8, None)}
    jit = lambda t: t if t.lstrip().startswith("@") else "@triton.jit\n" + t      # noqa: E731  getsource keeps it
    return port_fused_moe(initial_program(task), jit(src), jit(zeros), configs, vllm.__version__)


def check(task, fn, shapes, device) -> str | None:
    """Why `fn` is not a correct implementation of the task on these shapes, or None."""
    from kernels.check import _gen
    from kernels.evolve.run import _target_tolerances, tolerances
    dt = task.dtypes[0]
    tol = tolerances(_target_tolerances())                      # what the gate judges by: the target's, else the checker's
    atol, rtol = tuple(tol.get(str(dt).removeprefix("torch."), TOLERANCES[dt]))
    for s in shapes:
        _MOE.clear()                             # a translated routing is keyed by address: never reused across inputs
        xs = task.make_inputs(tuple(s), dt, device, _gen(0))
        try:
            d = _compare(task.reference(*[x.clone() for x in xs]), fn(*[x.clone() for x in xs]), atol, rtol)
        except Exception as e:                   # noqa: BLE001  a signature that differs in this library version
            return f"raised {type(e).__name__}: {str(e)[:300]}"
        if not d["ok"]:
            why = d.get("error") or (f"max_abs {d['max_abs']:.3g}, max_rel {d['max_rel']:.3g}, "
                                     f"non-finite mismatches {d['nonfinite_mismatch']} (atol {atol}, rtol {rtol})")
            return f"differs from the reference at shape {list(s)}: {why}"
    return None


def _tool(name: str) -> str:
    return shutil.which(name) or str(Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")) / "bin" / name)


def launched(fn, inputs) -> set[str]:
    """The device kernels one call of `fn` launches, by the names the profiler gives (demangled)."""
    from torch.profiler import ProfilerActivity, profile
    fn(*[x.clone() for x in inputs])                          # built and warm before the profiled call
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn(*[x.clone() for x in inputs])
        torch.cuda.synchronize()
    return {e.key for e in prof.key_averages() if e.device_type.name == "CUDA" and not e.key.startswith("Memcpy")}


def sass(module_file: str, pattern: str, names: set[str] | None = None) -> str:
    """The SASS of the functions in a library's extension whose names match `pattern`; with `names` (from `launched`)
    only the instantiations that ran, matched on their demangled names (a library holds every template's copy)."""
    p = subprocess.run([_tool("cuobjdump"), "-sass", module_file], capture_output=True, text=True)
    blocks = [b for b in re.split(r"(?=\n\s+Function : )", p.stdout) if re.search(r"Function : \S*(" + pattern + r")", b)]
    if not names or not blocks:
        return "".join(blocks)
    mangled = [re.search(r"Function : (\S+)", b).group(1) for b in blocks]
    d = subprocess.run([_tool("cu++filt")], input="\n".join(mangled), capture_output=True, text=True).stdout.splitlines()
    keys = {re.sub(r"\s+", "", n) for n in names}
    return "".join(b for b, n in zip(blocks, d) if re.sub(r"\s+", "", n) in keys
                   or any(re.sub(r"\s+", "", n).startswith(k.split("(")[0]) for k in keys))


def ptx(cache: Path, pattern: str) -> str:
    """The PTX of the Triton kernels in `cache` whose names match `pattern` (Triton names each file after its kernel),
    so each implementation's card holds its own kernels only."""
    return "".join(f.read_text() for f in sorted(cache.rglob("*.ptx")) if re.search(pattern, f.stem))


def weighted(per: list[dict]) -> float:
    """Our time over the competitor's, call-weighted geometric mean over the shapes (above 1: the competitor is
    faster)."""
    w = sum(r["calls"] for r in per)
    return math.exp(sum(r["calls"] * math.log(r["ours_over_competitor"]) for r in per) / w) if w else float("nan")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="kernels.competitors")
    ap.add_argument("--task", required=True)
    ap.add_argument("--shapes", required=True, help="a shapes file (kernels.shapes.write): its seen shapes are timed")
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--start", action="store_true", help="write start-<competitor>.py: the search's starting program "
                                                           "built on each competitor that has a port")
    a = ap.parse_args(argv)
    cache = Path(tempfile.mkdtemp(prefix="triton-cache-"))
    os.environ["TRITON_CACHE_DIR"] = str(cache)                    # the engine's Triton kernels' PTX lands here
    from kernels import shapes as shapes_mod
    from kernels.tasks import load, options_from_target
    from kernels.timing import time_pair
    task = load(a.task)
    task = task.with_options(options_from_target(task)) if task.options else task
    sets = shapes_mod.read(a.shapes)
    task = task.with_ints(sets.get("ints"))                        # real routing and lengths where captured
    seen = [tuple(s) for s in sets["seen"]]
    calls = sets.get("calls") or {}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ours = task.bind(task.baseline())
    rows = {"task": a.task, "shapes": [list(s) for s in seen], "graphs": task.graphs, "competitors": {}}
    for name, fn in (ADAPTERS.get(a.task) or {}).items():
        why = check(task, fn, seen, "cuda")
        if why:
            rows["competitors"][name] = {"not_timed": why}
            continue
        timed = functools.partial(fn, timed=True) if "timed" in inspect.signature(fn).parameters else fn
        per = []
        for s in seen:
            _MOE.clear()                         # the 4 input sets of one timing stay alive, so their addresses differ
            p = time_pair(timed, ours, task.make_inputs, s, task.dtypes[0], "cuda", iters=a.iters, graphs=task.graphs)
            per.append({"shape": list(s), "calls": calls.get(json.dumps(list(s)), 1), "competitor_ms": p["a"]["median_ms"],
                        "ours_ms": p["b"]["median_ms"], "ours_over_competitor": p["ratio"]["median"]})
        rows["competitors"][name] = {"per_shape": per, "ours_over_competitor": weighted(per)}
        if name in CONFIGS:
            rows["competitors"][name]["config"] = CONFIGS[name]
        if name in TRITON:
            (out / f"{name}.ptx").write_text(ptx(cache, SASS_NAMES[name]))
            continue
        mod = {"vllm": "vllm._moe_C" if "topk" in name else "vllm._C", "flashinfer": "flashinfer"}[name.split(".")[0]]
        try:
            from kernels.check import _gen
            xs = task.make_inputs(seen[0], task.dtypes[0], "cuda", _gen(0))
            text = sass(_module(mod).__file__, SASS_NAMES[name], launched(fn, xs))
        except Exception as e:                   # noqa: BLE001
            text = ""
            rows["competitors"][name]["sass_missing"] = f"{type(e).__name__}: {e}"
        (out / f"{name}.sass").write_text(text)
    if a.start:
        for name in rows["competitors"]:
            prog = start_program(task, name)
            if prog:
                (out / f"start-{name}.py").write_text(prog)
                rows["competitors"][name]["start"] = f"start-{name}.py"
    (out / "ours.ptx").write_text(ptx(cache, "|".join(task.kernel_names) or "."))
    timed_ones = {k: v["ours_over_competitor"] for k, v in rows["competitors"].items() if "per_shape" in v}
    best = max(timed_ones, key=timed_ones.get, default=None)
    rows["fastest"] = best if best and timed_ones[best] > 1 else "ours"
    (out / "timings.json").write_text(json.dumps(rows, indent=1))
    print(json.dumps({"task": a.task, "fastest": rows["fastest"],
                      "competitors": {k: (round(v["ours_over_competitor"], 3) if "per_shape" in v else v["not_timed"][:80])
                                      for k, v in rows["competitors"].items()}, "out": str(out)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
