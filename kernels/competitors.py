"""Other implementations of our kernel tasks, timed on our own shapes beside the engine's kernel, and their compiled
code kept for the PTX comparison's cards (kernels/ptx.py). Run on the GPU host in a venv that has the competitors'
libraries (vLLM, FlashInfer) beside this repo and the engine's source:

  uv venv ~/comp && uv pip install --python ~/comp/bin/python vllm flashinfer-python ninja   (FlashInfer JIT-builds;
  ~/comp/bin on PATH)
  PYTHONPATH=~/BlameGraph:~/inference-server/src ~/comp/bin/python -m kernels.competitors \\
      --task rmsnorm --shapes shapes-rmsnorm.json --out DIR

Each competitor is first checked against the task's reference on every shape, inputs from the shapes file's captured
ints where it has them (a wrong one is reported, never timed), then timed in pairs with the engine's kernel on the same
inputs (kernels.time_pair; as CUDA graph replays for a task the engine replays in graphs), the outputs of its timed
calls checked too. DIR gets `timings.json` (per shape, our time over each competitor's per pair: median, p25, p75; their
call-weighted summaries; `fastest`: a competitor only when its weighted p25 is above 1, else "within_noise" or
"ours") and one text file of compiled code per implementation: Triton kernels' PTX from Triton's cache by kernel name
(a competitor's: the files new since its first call), a library's SASS from `cuobjdump -sass` of its extension, the
running GPU's code filtered to the kernels that ran. `--start` also writes `start-<competitor>.py` for each timed
competitor whose kernel is source (vLLM's fused_moe), the search's starting program for `kernels.search --initial`; the
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
CONFIGS: dict[str, dict] = {}                       # competitor -> the launch config it chose for the current shape
UNTIMED: dict[str, Callable] = {}                   # competitor -> (its timed call's outputs, *inputs) -> the checked form
IN_PLACE: set[str] = set()                          # competitors whose timed call writes its outputs into its inputs
CONFIG_DTYPE = {"vllm.fused_moe": None, "vllm.fused_moe_w8a8": "fp8_w8a8"}    # vLLM's config table key for each


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
def _vllm_fused_add_rms_norm(h, a, w, timed=False):
    _vllm("vllm._C")
    if timed:                                    # its own copies, made once per input set (outside the timed calls):
        key = (h.data_ptr(), a.data_ptr())       # in place on the shared buffers would change what ours then reads
        if key not in _SCRATCH:
            _SCRATCH[key] = (h.clone(), a.clone())
        res, x = _SCRATCH[key]
    else:
        res, x = h.clone(), a.clone()            # in place: res becomes h + a, x its norm
    torch.ops._C.fused_add_rms_norm(x, res, w, EPS)
    return res, x


_SCRATCH: dict = {}


IN_PLACE.add("vllm.fused_add_rms_norm")


@adapter("silu_mul", "vllm.silu_and_mul", r"act_and_mul_kernel")
def _vllm_silu_and_mul(x):
    _vllm("vllm._C")
    out = torch.empty((*x.shape[:-1], x.shape[-1] // 2), dtype=x.dtype, device=x.device)
    torch.ops._C.silu_and_mul(out, x)
    return out


@adapter("topk_softmax", "vllm.topk_softmax", r"topkGating|moeSoftmax|topk")
def _vllm_topk_softmax(logits, timed=False):
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
    return (w, ids) if timed else (w, ids.long())         # timed: vLLM's int32 ids, no conversion


UNTIMED["vllm.topk_softmax"] = lambda out, *_: (out[0], out[1].long())


@adapter("paged_decode", "flashinfer.batch_decode", r"BatchDecode|batch_decode|SingleDecode")
def _flashinfer_decode(q, k_pool, v_pool, block_tables, seq_lens):
    """FlashInfer's paged decode on our pool (its HND layout: [pages, kv heads, page size, head dim]). The page lists
    and the plan are host work, done once per input set (as an engine plans once per step), so a call is `run` alone
    and captures in a CUDA graph."""
    import flashinfer
    n, hq, d = q.shape
    hkv, bs = k_pool.shape[1], k_pool.shape[2]
    key = (block_tables.data_ptr(), seq_lens.data_ptr(), n, hq, hkv, d, bs)
    if key not in _FI:
        pages = (seq_lens + bs - 1) // bs
        indptr = torch.zeros(n + 1, dtype=torch.int32, device=q.device)
        indptr[1:] = torch.cumsum(pages, 0)
        indices = torch.cat([block_tables[i, :int(pages[i])] for i in range(n)]).to(torch.int32)
        last = (seq_lens - (pages - 1) * bs).to(torch.int32)
        wrap = flashinfer.BatchDecodeWithPagedKVCacheWrapper(_fi_workspace(q.device), kv_layout="HND")
        wrap.plan(indptr, indices, last, hq, hkv, d, bs, q_data_type=q.dtype, kv_data_type=k_pool.dtype)
        _FI[key] = wrap
    return _FI[key].run(q, (k_pool, v_pool))


@functools.lru_cache(maxsize=None)
def _fi_workspace(device) -> torch.Tensor:
    """One float workspace for every plan: split-KV scratch used within a run (each plan's data is in its wrapper)."""
    return torch.empty(128 * 2**20, dtype=torch.uint8, device=device)


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


def _reset():
    """Forget what was prepared per input set (vLLM's translated routing, FlashInfer's plans) and the configs chosen:
    keyed by address, so never reused across inputs."""
    _MOE.clear()
    _FI.clear()
    _SCRATCH.clear()
    CONFIGS.clear()


def _vllm_fused_moe(name: str, w8a8: bool):
    """vLLM's fused_moe Triton kernel at the config vLLM picks for this GPU and expert shape: fp8 weights with bf16
    activations (`int8_w8a16`'s path, which upcasts fp8 the same way: our engine's decode numerics), or W8A8 (its
    activations quantized per token by vLLM's own op inside the call). The fp8-weight path's configs come from vLLM's
    table under dtype key None (its bf16 table): vLLM has no tuned table for fp8 weights with bf16 activations at this
    shape (`CONFIG_DTYPE`, recorded as `vllm_config_dtype`). The routing is translated once per input set, outside the
    timed call (layout, as FlashInfer's plan). Called with timed=True it returns vLLM's own output order (`UNTIMED`
    maps it back)."""
    def key(a, tok, w, sorted_ids, expert_ids):
        return tok.data_ptr(), sorted_ids.data_ptr(), expert_ids.data_ptr(), tuple(a.shape), tuple(w.shape), w8a8

    def call(a, tok, w, w_scale, sorted_ids, expert_ids, timed=False):
        import triton.language as tl
        fm = importlib.import_module("vllm.model_executor.layers.fused_moe.fused_moe")
        E, N, K = w.shape
        k = key(a, tok, w, sorted_ids, expert_ids)
        if k not in _MOE:
            k0 = int(torch.bincount(tok.long()).max()) if tok.numel() else 1
            cfg = dict(fm.try_get_optimal_moe_config(tuple(w.shape), (E, K, N // 2), k0, CONFIG_DTYPE[name],
                                                     a.shape[0]))
            _MOE[k] = (cfg, *_vllm_routing(a, tok, sorted_ids, expert_ids, E, cfg["BLOCK_SIZE_M"]))
            CONFIGS[name] = cfg
        cfg, p, top_k, gather, v_ids, e_ids, padded = _MOE[k]
        A, A_scale = (a[tok.long()] if gather else a), None
        if w8a8:
            from vllm import _custom_ops as ops
            A, A_scale = ops.scaled_fp8_quant(A, None, use_per_token_if_dynamic=True)
        C = torch.empty((A.shape[0], top_k, N), dtype=a.dtype, device=a.device)
        fm.invoke_fused_moe_triton_kernel(A, w, C, A_scale, w_scale, None, v_ids, e_ids, padded, False, top_k, cfg,
                                          tl.bfloat16, w8a8, False, not w8a8, False, w8a8, None)
        return C if timed else C.view(-1, N)[p]

    UNTIMED[name] = lambda C, a, tok, w, w_scale, s, e: C.view(-1, w.shape[1])[_MOE[key(a, tok, w, s, e)][1]]
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
    EM = expert_ids.numel() * block_m
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
             f"_VLLM_CONFIGS = {repr({int(k): v for k, v in configs.items()})}\n\n\n"
             f"{zeros_src}\n\n{kernel_src}{FUSED_MOE_WRAPPER}")
    end = "\n# EVOLVE-BLOCK-END\n"
    if program.count(end) != 1:
        raise ValueError("the starting program has no single EVOLVE-BLOCK-END")
    return program.replace(end, block + end)


def vllm_configs(fm, shapes, dtype: str | None) -> dict:
    """vLLM's tuned configs {tokens: config} for the expert shape of `shapes` ((M, K, N, E, T, ft) tuples), else
    get_default_config at each token count T they have."""
    dims = {(K, N, E) for M, K, N, E, T, ft in shapes}
    if len(dims) != 1:
        raise ValueError(f"one expert shape expected, the shapes have {sorted(dims)}")
    (K, N, E), = dims
    return fm.get_moe_configs(E, N // 2, dtype) or {
        T: fm.get_default_config(T, E, N // 2, K, max(1, M // T), dtype) for M, K, N, E, T, ft in shapes}


def start_program(task, name: str, shapes) -> str | None:
    """The starting program built on competitor `name`'s kernel at vLLM's configs for `shapes`, or None when it has
    no port (a library's compiled code cannot be evolved as source)."""
    if name != "vllm.fused_moe" or task.name != "moe_gemm_fp8":
        return None
    import vllm
    from kernels.evolve.run import initial_program
    fm = importlib.import_module("vllm.model_executor.layers.fused_moe.fused_moe")
    src = inspect.getsource(fm.fused_moe_kernel.fn)
    zeros = inspect.getsource(fm.write_zeros_to_output.fn)
    configs = vllm_configs(fm, shapes, CONFIG_DTYPE[name])
    jit = lambda t: t if t.lstrip().startswith("@") else "@triton.jit\n" + t      # noqa: E731  getsource keeps it
    return port_fused_moe(initial_program(task), jit(src), jit(zeros), configs, vllm.__version__)


def _tolerance(task) -> tuple[float, float]:
    """(atol, rtol) the gate judges by: the target's, else the checker's."""
    from kernels.evolve.run import _target_tolerances, tolerances
    dt = task.dtypes[0]
    return tuple(tolerances(_target_tolerances()).get(str(dt).removeprefix("torch."), TOLERANCES[dt]))


def _why(d: dict, atol: float, rtol: float) -> str:
    return d.get("error") or (f"max_abs {d['max_abs']:.3g}, max_rel {d['max_rel']:.3g}, "
                              f"non-finite mismatches {d['nonfinite_mismatch']} (atol {atol}, rtol {rtol})")


def check(task, fn, shapes, device) -> str | None:
    """Why `fn` is not a correct implementation of the task on these shapes, or None."""
    from kernels.check import _gen
    atol, rtol = _tolerance(task)
    for s in shapes:
        _reset()
        xs = task.make_inputs(tuple(s), task.dtypes[0], device, _gen(0))
        try:
            d = _compare(task.reference(*[x.clone() for x in xs]), fn(*[x.clone() for x in xs]), atol, rtol)
        except Exception as e:                   # noqa: BLE001  a signature that differs in this library version
            return f"raised {type(e).__name__}: {str(e)[:300]}"
        if not d["ok"]:
            return f"differs from the reference at shape {list(s)}: {_why(d, atol, rtol)}"
    return None


def timed_wrong(task, name: str, timed, p: dict) -> str | None:
    """Why competitor `name`'s outputs of its timed calls (time_pair's `outputs_a`) differ from the reference on the
    same input sets, or None. An in-place one's inputs already hold its outputs, so its timed path runs once more on a
    copy of each set."""
    atol, rtol = _tolerance(task)
    for i, (xs, out) in enumerate(zip(p["inputs"], p["outputs_a"])):
        if name in IN_PLACE:
            xs = [x.clone() for x in xs]
            out = timed(*[x.clone() for x in xs])
        out = UNTIMED.get(name, lambda o, *_: o)(out, *xs)
        d = _compare(task.reference(*[x.clone() for x in xs]), out, atol, rtol)
        if not d["ok"]:
            return f"its timed outputs differ from the reference on input set {i}: {_why(d, atol, rtol)}"
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


def _signature(name: str) -> str:
    """A demangled name's qualified function name with its template args, whitespace removed: the parameter list (the
    first "(" outside template args) and any return type before it (cu++filt keeps `std::enable_if<...>::type`, the
    profiler does not) dropped."""
    s, depth, start = name.replace("(anonymous namespace)", "{anon}"), 0, 0
    for i, c in enumerate(s):
        depth += (c == "<") - (c == ">")
        if c == " " and depth == 0:
            start = i + 1                        # a top-level space ends the return type
        if c == "(" and depth == 0:
            return re.sub(r"\s+", "", s[start:i])
    return re.sub(r"\s+", "", s[start:])


def sass(module_file: str, pattern: str, names: set[str] | None = None, arch: str | None = None) -> str:
    """The SASS of the functions in a library's extension whose names match `pattern`, only `arch`'s code when given
    (e.g. sm_90, which also takes sm_90a); with `names` (from `launched`) only the instantiations that ran, matched on
    their demangled names (a library holds every template's copy). Raises if cuobjdump fails."""
    p = subprocess.run([_tool("cuobjdump"), "-sass", module_file], capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"cuobjdump exited {p.returncode} on {module_file}: {p.stderr.strip()[:300]}")
    code = p.stdout
    if arch:                                     # cuobjdump heads each architecture's code with "code for sm_XY"
        code = "".join(c for c in re.split(r"(?=\n\s*code for sm_)", code) if re.match(rf"\s*code for {arch}[a-z]?\s", c))
    blocks = [b for b in re.split(r"(?=\n\s+Function : )", code) if re.search(r"Function : \S*(" + pattern + r")", b)]
    if not names or not blocks:
        return "".join(blocks)
    mangled = [re.search(r"Function : (\S+)", b).group(1) for b in blocks]
    d = subprocess.run([_tool("cu++filt")], input="\n".join(mangled), capture_output=True, text=True).stdout.splitlines()
    keys = {re.sub(r"\s+", "", n) for n in names}
    sigs = {_signature(n) for n in names}
    ran = "".join(b for b, n in zip(blocks, d) if re.sub(r"\s+", "", n) in keys or _signature(n) in sigs)
    return ran or ("// the instantiations that ran were not told apart: every match of the pattern\n" + "".join(blocks))


def _flashinfer_jit_dir() -> Path:
    """Where FlashInfer keeps its JIT-built extensions (`flashinfer.jit.env.FLASHINFER_JIT_DIR`, else its default)."""
    try:
        from flashinfer.jit import env
        return Path(env.FLASHINFER_JIT_DIR)
    except (ImportError, AttributeError):
        return Path.home() / ".cache" / "flashinfer"


def library_sass(name: str, names: set[str], arch: str | None) -> str:
    """Competitor `name`'s SASS from its library's extension (FlashInfer's: every JIT-built .so); raises with the
    reason when none holds a matching function."""
    if name.startswith("flashinfer."):
        files = sorted(str(f) for f in _flashinfer_jit_dir().rglob("*.so"))
        if not files:
            raise FileNotFoundError(f"no JIT-built .so under {_flashinfer_jit_dir()}")
    else:
        files = [_module("vllm._moe_C" if "topk" in name else "vllm._C").__file__]
    texts, errors = [], []
    for f in files:
        try:
            texts.append(sass(f, SASS_NAMES[name], names, arch))
        except RuntimeError as e:
            errors.append(str(e))
    if not "".join(texts):
        raise RuntimeError(f"no {arch or 'any'} function matching {SASS_NAMES[name]!r} that ran in {len(files)} "
                           f"file(s)" + (f": {'; '.join(errors)[:500]}" if errors else ""))
    return "".join(texts)


def ptx(cache: Path, pattern: str, before: set | frozenset = frozenset()) -> str:
    """The PTX of the Triton kernels in `cache` whose names match `pattern` (Triton names each file after its kernel)
    and that are not in `before` (the files present before its first call), so each implementation's card holds its
    own kernels only, also when two share a kernel."""
    return "".join(f.read_text() for f in sorted(cache.rglob("*.ptx")) if f not in before and re.search(pattern, f.stem))


RATIOS = ("ours_over_competitor", "ratio_p25", "ratio_p75")     # per pair: median, p25, p75 of our time over theirs


def weighted(per: list[dict], key: str = "ours_over_competitor") -> float:
    """A ratio's call-weighted geometric mean over the shapes (above 1: the competitor is faster)."""
    w = sum(r["calls"] for r in per)
    return math.exp(sum(r["calls"] * math.log(r[key]) for r in per) / w) if w else float("nan")


def fastest(competitors: dict) -> str:
    """The timed competitor whose call-weighted p25 ratio is above 1 (faster clear of the pairs' spread; of several,
    the highest weighted median), else "within_noise" when one's weighted p25..p75 spans 1, else "ours"."""
    timed = {k: v for k, v in competitors.items() if "per_shape" in v}
    clear = [k for k, v in timed.items() if v["ratio_p25"] > 1]
    if clear:
        return max(clear, key=lambda k: timed[k]["ours_over_competitor"])
    return "within_noise" if any(v["ratio_p75"] > 1 for v in timed.values()) else "ours"


def time_competitor(task, name: str, fn, ours, shapes, calls: dict, iters: int, device="cuda") -> list[dict] | str:
    """Each shape's paired timing of competitor `name` against ours (the config it chose there, if any), or why it is
    not timed: its timed outputs differ from the reference (a kernel doing other work when timed) or timing raised."""
    from kernels.timing import time_pair
    timed = functools.partial(fn, timed=True) if "timed" in inspect.signature(fn).parameters else fn
    per = []
    try:
        for s in shapes:
            _reset()                             # the 4 input sets of one timing stay alive, so their addresses differ
            p = time_pair(timed, ours, task.make_inputs, s, task.dtypes[0], device, iters=iters, graphs=task.graphs)
            why = timed_wrong(task, name, timed, p)
            if why:
                return f"at shape {list(s)}: {why}"
            r = p["ratio"]
            per.append({"shape": list(s), "calls": calls.get(json.dumps(list(s)), 1),
                        "competitor_ms": p["a"]["median_ms"], "ours_ms": p["b"]["median_ms"],
                        "ours_over_competitor": r["median"], "ratio_p25": r["p25"], "ratio_p75": r["p75"],
                        **({"config": CONFIGS[name]} if name in CONFIGS else {})})
    except Exception as e:                       # noqa: BLE001  one competitor's failure keeps the others' results
        return f"timing raised {type(e).__name__}: {str(e)[:300]}"
    return per


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="kernels.competitors")
    ap.add_argument("--task", required=True)
    ap.add_argument("--shapes", required=True, help="a shapes file (kernels.shapes.write): its seen shapes are timed")
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--start", action="store_true", help="write start-<competitor>.py: the search's starting program "
                                                           "built on each timed competitor that has a port")
    a = ap.parse_args(argv)
    cache = Path(tempfile.mkdtemp(prefix="triton-cache-"))
    os.environ["TRITON_CACHE_DIR"] = str(cache)                    # the engine's Triton kernels' PTX lands here
    from kernels import shapes as shapes_mod
    from kernels.check import _gen
    from kernels.tasks import load, options_from_target
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
    arch = "sm_%d%d" % torch.cuda.get_device_capability()          # the running GPU's code only, in SASS cards
    for name, fn in (ADAPTERS.get(a.task) or {}).items():
        before = set(cache.rglob("*.ptx"))       # its card: the Triton kernels it compiles, not an earlier one's
        per = check(task, fn, seen, "cuda") or time_competitor(task, name, fn, ours, seen, calls, a.iters)
        if isinstance(per, str):
            rows["competitors"][name] = {"not_timed": per}
            continue
        r = rows["competitors"][name] = {"per_shape": per, **{k: weighted(per, k) for k in RATIOS}}
        if name in CONFIG_DTYPE:
            r["vllm_config_dtype"] = str(CONFIG_DTYPE[name])
        if name in TRITON:
            (out / f"{name}.ptx").write_text(ptx(cache, SASS_NAMES[name], before))
            continue
        try:
            _reset()
            text = library_sass(name, launched(fn, task.make_inputs(seen[0], task.dtypes[0], "cuda", _gen(0))), arch)
        except Exception as e:                   # noqa: BLE001
            text = ""
            r["sass_missing"] = f"{type(e).__name__}: {e}"
        (out / f"{name}.sass").write_text(text)
    if a.start:
        for name in [k for k, v in rows["competitors"].items() if "per_shape" in v]:
            prog = start_program(task, name, seen)
            if prog:
                (out / f"start-{name}.py").write_text(prog)
                rows["competitors"][name]["start"] = f"start-{name}.py"
    (out / "ours.ptx").write_text(ptx(cache, "|".join(task.kernel_names) or "."))
    rows["fastest"] = fastest(rows["competitors"])
    (out / "timings.json").write_text(json.dumps(rows, indent=1))
    print(json.dumps({"task": a.task, "fastest": rows["fastest"],
                      "competitors": {k: (round(v["ours_over_competitor"], 3) if "per_shape" in v else v["not_timed"][:80])
                                      for k, v in rows["competitors"].items()}, "out": str(out)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
