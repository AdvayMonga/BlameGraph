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
competitor that has a port (vLLM's fused_moe as Triton source, vLLM's topk_softmax as CUDA C++ built at import), the
search's starting program for `kernels.search --initial`; the score stays relative to the engine's kernel."""
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
    return _before_end(program, block)


def _before_end(program: str, block: str) -> str:
    """`program` with `block` inserted before its EVOLVE-BLOCK-END marker, so the port is open to the search."""
    end = "\n# EVOLVE-BLOCK-END\n"
    if program.count(end) != 1:
        raise ValueError("the starting program has no single EVOLVE-BLOCK-END")
    return program.replace(end, block + end)


# vLLM's topk_softmax CUDA kernel, made self-contained for `torch.utils.cpp_extension.load_inline` in the engine's fused
# module: only system and torch headers, one C++ entry point, nothing of vLLM's binding layer.
TOPK_CUDA = r'''
/*
 * Adapted from vLLM's csrc/moe/topk_softmax_kernels.cu (commit 58759182c, 2026-10-25), itself adapted from
 * https://github.com/NVIDIA/TensorRT-LLM/blob/v0.7.1/cpp/tensorrt_llm/kernels/mixtureOfExperts/moe_kernels.cu
 * Copyright (c) 2024, The vLLM team.
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Kept from vLLM: the warp top-k gating softmax (topkGating: softmax, max and arg-max fused, a row shared by the lanes
 * of one warp, no shared memory), its launch constants, the PDL launch, bf16/fp16/fp32 logits, renormalisation,
 * ties to the lowest expert index. Dropped: every expert count but 128 (vLLM instantiates 1 to 512), the cub softmax +
 * top-k pair for expert counts that are not a power of two, sigmoid scoring, bias, finished and padding rows,
 * expert-parallel ranges, source rows, the torch stable bindings and ROCm. Changed: a row whose softmax is NaN (a NaN or +Inf logit) gets NaN weights and expert
 * E - 1, as the engine's kernel gives (vLLM clamps such rows to zeros), and the indices are written as int64, the
 * engine's dtype, so no cast follows the kernel.
 */
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include <cstdint>
#include <type_traits>

namespace {

constexpr int WARP_SIZE = 32;

template <typename T, int N, int Alignment = sizeof(T) * N>
struct alignas(Alignment) AlignedArray {
    T data[N];
};

template <typename T>
__device__ __forceinline__ float toFloat(T value) {
    if constexpr (std::is_same_v<T, float>) {
        return value;
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        return __bfloat162float(value);
    } else {
        return __half2float(value);
    }
}

/*
  A Top-K gating softmax written to exploit when the number of experts in the MoE layers
  are a small power of 2. This allows us to cleanly share the rows among the threads in
  a single warp and eliminate communication between warps (so no need to use shared mem).

  It fuses the softmax, max and argmax into a single kernel.

  Limitations:
  1) This implementation is optimized for when the number of experts is a small power of 2.
  2) This implementation assumes k is small, but will work for any k.
*/
template <int VPT, int NUM_EXPERTS, int WARPS_PER_CTA, int BYTES_PER_LDG, typename InputType>
__launch_bounds__(WARPS_PER_CTA * WARP_SIZE) __global__
    void topkGating(const InputType* input, float* output, const int num_rows, int64_t* indices, const int k,
                    const bool renormalize)
{
    static_assert(std::is_same_v<InputType, float> || std::is_same_v<InputType, __nv_bfloat16> ||
                      std::is_same_v<InputType, __half>,
                  "InputType must be float, __nv_bfloat16, or __half");

    // We begin by enforcing compile time assertions and setting up compile time constants.
    static_assert(BYTES_PER_LDG == (BYTES_PER_LDG & -BYTES_PER_LDG), "BYTES_PER_LDG must be power of 2");
    static_assert(BYTES_PER_LDG <= 16, "BYTES_PER_LDG must be leq 16");

    // Number of bytes each thread pulls in per load
    static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(InputType);
    static constexpr int ELTS_PER_ROW = NUM_EXPERTS;
    static constexpr int THREADS_PER_ROW = ELTS_PER_ROW / VPT;
    static constexpr int LDG_PER_THREAD = VPT / ELTS_PER_LDG;

    if constexpr (std::is_same_v<InputType, __nv_bfloat16> || std::is_same_v<InputType, __half>) {
        static_assert(ELTS_PER_LDG == 1 || ELTS_PER_LDG % 2 == 0,
            "ELTS_PER_LDG must be 1 or even for 16-bit conversion");
    }

    // Restrictions based on previous section.
    static_assert(VPT % ELTS_PER_LDG == 0, "The elements per thread must be a multiple of the elements per ldg");
    static_assert(WARP_SIZE % THREADS_PER_ROW == 0, "The threads per row must cleanly divide the threads per warp");
    static_assert(THREADS_PER_ROW == (THREADS_PER_ROW & -THREADS_PER_ROW), "THREADS_PER_ROW must be power of 2");
    static_assert(THREADS_PER_ROW <= WARP_SIZE, "THREADS_PER_ROW can be at most warp size");

    // We have NUM_EXPERTS elements per row. We specialize for small #experts
    static constexpr int ELTS_PER_WARP = WARP_SIZE * VPT;
    static constexpr int ROWS_PER_WARP = ELTS_PER_WARP / ELTS_PER_ROW;
    static constexpr int ROWS_PER_CTA = WARPS_PER_CTA * ROWS_PER_WARP;

    // Restrictions for previous section.
    static_assert(ELTS_PER_WARP % ELTS_PER_ROW == 0, "The elts per row must cleanly divide the total elt per warp");

    // ===================== From this point, we finally start computing run-time variables. ========================

    // Compute CTA and warp rows. We pack multiple rows into a single warp, and a block contains WARPS_PER_CTA warps.
    // This, each block processes a chunk of rows. We start by computing the start row for each block.
    const int cta_base_row = blockIdx.x * ROWS_PER_CTA;

    // Now, using the base row per thread block, we compute the base row per warp.
    const int warp_base_row = cta_base_row + threadIdx.y * ROWS_PER_WARP;

    // The threads in a warp are split into sub-groups that will work on a row.
    // We compute row offset for each thread sub-group
    const int thread_row_in_warp = threadIdx.x / THREADS_PER_ROW;
    const int thread_row = warp_base_row + thread_row_in_warp;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    // PDL permits early launch, but logits/metadata still depend on the producer.
    cudaGridDependencySynchronize();
    cudaTriggerProgrammaticLaunchCompletion();
#endif

    // Threads with indices out of bounds should early exit here.
    if (thread_row >= num_rows)
    {
        return;
    }

    // We finally start setting up the read pointers for each thread. First, each thread jumps to the start of the
    // row it will read.
    const InputType* thread_row_ptr = input + thread_row * ELTS_PER_ROW;

    // Now, we compute the group each thread belong to in order to determine the first column to start loads.
    const int thread_group_idx = threadIdx.x % THREADS_PER_ROW;
    const int first_elt_read_by_thread = thread_group_idx * ELTS_PER_LDG;
    const InputType* thread_read_ptr = thread_row_ptr + first_elt_read_by_thread;

    // Finally, we pull in the data from global mem
    float row_chunk[VPT];

    // Dispatch different input types loading, BF16/FP16 convert to float
    if constexpr (std::is_same_v<InputType, float>) {
        using VecType = AlignedArray<float, ELTS_PER_LDG>;
        VecType* row_chunk_vec_ptr = reinterpret_cast<VecType*>(&row_chunk);
        const VecType* vec_thread_read_ptr = reinterpret_cast<const VecType*>(thread_read_ptr);
#pragma unroll
        for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
            row_chunk_vec_ptr[ii] = vec_thread_read_ptr[ii * THREADS_PER_ROW];
        }
    } else if constexpr (std::is_same_v<InputType, __nv_bfloat16>) {
        if constexpr (ELTS_PER_LDG >= 2) {
            using VecType = AlignedArray<__nv_bfloat16, ELTS_PER_LDG>;
            float2* row_chunk_f2 = reinterpret_cast<float2*>(row_chunk);
            const VecType* vec_thread_read_ptr = reinterpret_cast<const VecType*>(thread_read_ptr);
#pragma unroll
            for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
                VecType vec = vec_thread_read_ptr[ii * THREADS_PER_ROW];
                int base_idx_f2 = ii * ELTS_PER_LDG / 2;
#pragma unroll
                for (int jj = 0; jj < ELTS_PER_LDG / 2; ++jj) {
                    row_chunk_f2[base_idx_f2 + jj] = __bfloat1622float2(
                        *reinterpret_cast<const __nv_bfloat162*>(vec.data + jj * 2)
                    );
                }
            }
        } else { // ELTS_PER_LDG == 1
#pragma unroll
            for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
                const __nv_bfloat16* scalar_ptr = thread_read_ptr + ii * THREADS_PER_ROW;
                row_chunk[ii] = __bfloat162float(*scalar_ptr);
            }
        }
    } else if constexpr (std::is_same_v<InputType, __half>) {
        if constexpr (ELTS_PER_LDG >= 2) {
            using VecType = AlignedArray<__half, ELTS_PER_LDG>;
            float2* row_chunk_f2 = reinterpret_cast<float2*>(row_chunk);
            const VecType* vec_thread_read_ptr = reinterpret_cast<const VecType*>(thread_read_ptr);
#pragma unroll
            for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
                VecType vec = vec_thread_read_ptr[ii * THREADS_PER_ROW];
                int base_idx_f2 = ii * ELTS_PER_LDG / 2;
#pragma unroll
                for (int jj = 0; jj < ELTS_PER_LDG / 2; ++jj) {
                    row_chunk_f2[base_idx_f2 + jj] = __half22float2(
                        *reinterpret_cast<const __half2*>(vec.data + jj * 2)
                    );
                }
            }
        } else { // ELTS_PER_LDG == 1
#pragma unroll
            for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
                const __half* scalar_ptr = thread_read_ptr + ii * THREADS_PER_ROW;
                row_chunk[ii] = __half2float(*scalar_ptr);
            }
        }
    }

    // First, we perform a max reduce within the thread.
    float thread_max = row_chunk[0];
#pragma unroll
    for (int ii = 1; ii < VPT; ++ii) {
        thread_max = max(thread_max, row_chunk[ii]);
    }

    // Now, we find the max within the thread group and distribute among the threads. We use a butterfly reduce.
#pragma unroll
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2)
    {
        thread_max = max(thread_max, __shfl_xor_sync(uint32_t(-1), thread_max, mask, THREADS_PER_ROW));
    }

    // From this point, thread max in all the threads have the max within the row.
    // Now, we subtract the max from each element in the thread and take the exp. We also compute the thread local sum.
    float row_sum = 0;
#pragma unroll
    for (int ii = 0; ii < VPT; ++ii)
    {
        row_chunk[ii] = expf(row_chunk[ii] - thread_max);
        row_sum += row_chunk[ii];
    }

    // Now, we perform the sum reduce within each thread group. Similar to the max reduce, we use a bufferfly pattern.
#pragma unroll
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2)
    {
        row_sum += __shfl_xor_sync(uint32_t(-1), row_sum, mask, THREADS_PER_ROW);
    }

    // A NaN or +Inf logit (or a row of -Inf) makes the sum NaN in every lane of the row: the engine's kernel gives such
    // a row NaN weights and expert NUM_EXPERTS - 1 for every k, and so does this one (vLLM writes zeros instead).
    if (isnan(row_sum))
    {
        if (thread_group_idx == 0)
        {
            for (int k_idx = 0; k_idx < k; ++k_idx)
            {
                output[k * thread_row + k_idx] = CUDART_NAN_F;
                indices[k * thread_row + k_idx] = NUM_EXPERTS - 1;
            }
        }
        return;
    }

    // From this point, all threads have the max and the sum for their rows in the thread_max and thread_sum variables
    // respectively. Finally, we can scale the rows for the softmax. Technically, for top-k gating we don't need to
    // compute the entire softmax row. We can likely look at the maxes and only compute for the top-k values in the row.
    // However, this kernel will likely not be a bottle neck and it seems better to closer match torch and find the
    // argmax after computing the softmax.
    const float reciprocal_row_sum = 1.f / row_sum;

#pragma unroll
    for (int ii = 0; ii < VPT; ++ii)
    {
        row_chunk[ii] = row_chunk[ii] * reciprocal_row_sum;
    }

    // Now, row_chunk contains the softmax of the row chunk. Now, I want to find the topk elements in each row, along
    // with the max index.
    int start_col = first_elt_read_by_thread;
    static constexpr int COLS_PER_GROUP_LDG = ELTS_PER_LDG * THREADS_PER_ROW;

    float selected_sum = 0.f;
    for (int k_idx = 0; k_idx < k; ++k_idx)
    {
        // First, each thread does the local argmax
        float max_val = row_chunk[0];
        int expert = start_col;
#pragma unroll
        for (int ldg = 0, col = start_col; ldg < LDG_PER_THREAD; ++ldg, col += COLS_PER_GROUP_LDG)
        {
#pragma unroll
            for (int ii = 0; ii < ELTS_PER_LDG; ++ii)
            {
                float val = row_chunk[ldg * ELTS_PER_LDG + ii];

                // No check on the experts here since columns with the smallest index are processed first and only
                // updated if > (not >=)
                if (val > max_val)
                {
                    max_val = val;
                    expert = col + ii;
                }
            }
        }

        // Now, we perform the argmax reduce. We use the butterfly pattern so threads reach consensus about the max.
        // This will be useful for K > 1 so that the threads can agree on "who" had the max value. That thread can
        // then blank out their max with -inf and the warp can run more iterations...
#pragma unroll
        for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2)
        {
            float other_max = __shfl_xor_sync(uint32_t(-1), max_val, mask, THREADS_PER_ROW);
            int other_expert = __shfl_xor_sync(uint32_t(-1), expert, mask, THREADS_PER_ROW);

            // We want lower indices to "win" in every thread so we break ties this way
            if (other_max > max_val || (other_max == max_val && other_expert < expert))
            {
                max_val = other_max;
                expert = other_expert;
            }
        }

        // Write the max for this k iteration to global memory.
        if (thread_group_idx == 0)
        {
            // The lead thread from each sub-group will write out the final results to global memory. (This will be a
            // single) thread per row of the input/output matrices.
            const int idx = k * thread_row + k_idx;
            output[idx] = max_val;
            indices[idx] = expert;
            if (renormalize) {
                selected_sum += max_val;
            }
        }

        // Finally, we clear the value in the thread with the current max if there is another iteration to run.
        if (k_idx + 1 < k)
        {
            const int ldg_group_for_expert = expert / COLS_PER_GROUP_LDG;
            const int thread_to_clear_in_group = (expert / ELTS_PER_LDG) % THREADS_PER_ROW;

            // Only the thread in the group which produced the max will reset the "winning" value to -inf.
            if (thread_group_idx == thread_to_clear_in_group)
            {
                const int offset_for_expert = expert % ELTS_PER_LDG;
                // Safe to set to any negative value since row_chunk values must be between 0 and 1.
                row_chunk[ldg_group_for_expert * ELTS_PER_LDG + offset_for_expert] = -10000.f;
            }
        }
    }

    // Apply renormalization to final weights.
    if (renormalize && thread_group_idx == 0) {
        const float scale = 1.f / (selected_sum > 0.f ? selected_sum : 1.f);
        for (int k_idx = 0; k_idx < k; ++k_idx) {
            const int idx = k * thread_row + k_idx;
            output[idx] = output[idx] * scale;
        }
    }
}

// Constructs some constants needed to partition the work across threads at compile time.
template <int EXPERTS, int BYTES_PER_LDG, typename InputType>
struct TopkConstants
{
    static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(InputType);
    static_assert(EXPERTS / (ELTS_PER_LDG * WARP_SIZE) == 0 || EXPERTS % (ELTS_PER_LDG * WARP_SIZE) == 0, "");
    static constexpr int VECs_PER_THREAD = EXPERTS / (ELTS_PER_LDG * WARP_SIZE) > 1 ? EXPERTS / (ELTS_PER_LDG * WARP_SIZE) : 1;
    static constexpr int VPT = VECs_PER_THREAD * ELTS_PER_LDG;
    static constexpr int THREADS_PER_ROW = EXPERTS / VPT;
    static constexpr int ROWS_PER_WARP = WARP_SIZE / THREADS_PER_ROW;
};

template <int EXPERTS, typename InputType>
void launchTopkGating(const InputType* input, float* output, int64_t* indices, const int num_rows, const int k,
                      const bool renormalize, cudaStream_t stream)
{
    static constexpr int WARPS_PER_TB = 4;
    static constexpr int MAX_BYTES_PER_LDG = 16;
    static constexpr int BYTES_PER_LDG = int(sizeof(InputType)) * EXPERTS < MAX_BYTES_PER_LDG
                                             ? int(sizeof(InputType)) * EXPERTS : MAX_BYTES_PER_LDG;
    using Constants = TopkConstants<EXPERTS, BYTES_PER_LDG, InputType>;
    static constexpr int VPT = Constants::VPT;
    static constexpr int ROWS_PER_WARP = Constants::ROWS_PER_WARP;
    const int num_warps = (num_rows + ROWS_PER_WARP - 1) / ROWS_PER_WARP;
    const int num_blocks = (num_warps + WARPS_PER_TB - 1) / WARPS_PER_TB;

    // Programmatic dependent launch on the current stream, as vLLM launches it: the kernel may start before the
    // producer finishes and waits for its data inside (cudaGridDependencySynchronize above).
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(num_blocks);
    config.blockDim = dim3(WARP_SIZE, WARPS_PER_TB);
    config.stream = stream;
    cudaLaunchAttribute attr{};
    attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr.val.programmaticStreamSerializationAllowed = 1;
    config.attrs = &attr;
    config.numAttrs = 1;
    auto err = cudaLaunchKernelEx(&config, topkGating<VPT, EXPERTS, WARPS_PER_TB, BYTES_PER_LDG, InputType>,
                                  input, output, num_rows, indices, k, renormalize);
    TORCH_CHECK(err == cudaSuccess, "topkGating launch failed: ", cudaGetErrorString(err));
}

template <typename InputType>
void topkGatingForExperts(const InputType* input, float* output, int64_t* indices, const int num_rows,
                          const int num_experts, const int k, const bool renormalize, cudaStream_t stream)
{
    // Only Qwen3-30B-A3B's 128 experts are instantiated (vLLM builds 1 to 512): one template, a shorter build.
    TORCH_CHECK(num_experts == 128, "topk_softmax_cuda: ", num_experts, " experts; only 128 is built");
    launchTopkGating<128>(input, output, indices, num_rows, k, renormalize, stream);
}

} // namespace

// logits [M, E] (bf16, fp16 or fp32, contiguous) -> weights fp32 [M, k], ids int64 [M, k], both preallocated.
void topk_softmax_cuda(torch::Tensor logits, torch::Tensor weights, torch::Tensor ids, bool renorm)
{
    TORCH_CHECK(logits.is_cuda() && logits.dim() == 2 && logits.is_contiguous(), "logits: a contiguous [M, E] CUDA tensor");
    TORCH_CHECK(weights.scalar_type() == at::kFloat && weights.is_contiguous() && weights.dim() == 2,
                "weights: a contiguous fp32 [M, k] tensor");
    TORCH_CHECK(ids.scalar_type() == at::kLong && ids.is_contiguous() && ids.sizes() == weights.sizes(),
                "ids: a contiguous int64 tensor of the weights' shape");
    const int M = logits.size(0), E = logits.size(1), k = weights.size(1);
    TORCH_CHECK(weights.size(0) == M && k <= E, "weights: [M, k] with k <= E");
    if (M == 0) return;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    float* w = weights.data_ptr<float>();
    int64_t* i = ids.data_ptr<int64_t>();
    switch (logits.scalar_type()) {
        case at::kBFloat16:
            topkGatingForExperts(reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr()), w, i, M, E, k, renorm, stream);
            break;
        case at::kHalf:
            topkGatingForExperts(reinterpret_cast<const __half*>(logits.data_ptr()), w, i, M, E, k, renorm, stream);
            break;
        case at::kFloat:
            topkGatingForExperts(logits.data_ptr<float>(), w, i, M, E, k, renorm, stream);
            break;
        default:
            TORCH_CHECK(false, "topk_softmax_cuda: logits must be bf16, fp16 or fp32, got ", logits.scalar_type());
    }
}
'''

TOPK_WRAPPER = f'''

# vLLM's topk_softmax (csrc/moe/topk_softmax_kernels.cu, Apache-2.0, from TensorRT-LLM) as CUDA C++ built when this
# module is imported; its header comment says what was kept, dropped and changed. The build dir and the architecture
# are the evaluator's ($TORCH_EXTENSIONS_DIR, $TORCH_CUDA_ARCH_LIST).
from torch.utils.cpp_extension import load_inline

_TOPK_CPP = "void topk_softmax_cuda(torch::Tensor logits, torch::Tensor weights, torch::Tensor ids, bool renorm);"
_TOPK_CUDA = r"""{TOPK_CUDA}"""
_topk_ext = load_inline(name="vllm_topk_softmax", cpp_sources=[_TOPK_CPP], cuda_sources=[_TOPK_CUDA],
                        functions=["topk_softmax_cuda"], extra_cuda_cflags=["-O3"], verbose=False)

_topk_softmax_engine = topk_softmax


def topk_softmax(logits: torch.Tensor, k: int, renorm: bool, fp32: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """vLLM's warp top-k gating softmax in one launch for 128 experts and k = 8 on CUDA; the engine's kernel
    otherwise."""
    M, E = logits.shape
    if not logits.is_cuda or E != 128 or k != 8:
        return _topk_softmax_engine(logits, k, renorm, fp32)
    logits = logits.contiguous()
    w = torch.empty(M, k, dtype=torch.float32, device=logits.device)
    ids = torch.empty(M, k, dtype=torch.int64, device=logits.device)
    _topk_ext.topk_softmax_cuda(logits, w, ids, renorm)
    return (w if fp32 else w.to(logits.dtype)), ids
'''


def port_topk_softmax(program: str) -> str:
    """The starting program `program` (the engine's fused module) with vLLM's topk_softmax kernel built at import and
    `topk_softmax` routed to it."""
    return _before_end(program, TOPK_WRAPPER)


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
    from kernels.evolve.run import initial_program
    if name == "vllm.topk_softmax" and task.name == "topk_softmax":
        return port_topk_softmax(initial_program(task))
    if name != "vllm.fused_moe" or task.name != "moe_gemm_fp8":
        return None
    import vllm
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
