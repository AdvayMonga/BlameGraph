"""Competitors' kernels as source for the starting programs in kernels/competitors.py (`starts`), so a search can start
from them with no competitor library installed: vLLM v0.32.0's Triton kernels vendored verbatim (fused_moe_kernel, the
decode attention's grouped stage 1 and stage 2) with its tuned fused_moe table for the H200, and its CUDA C++ norm and
activation kernels made self-contained for `torch.utils.cpp_extension.load_inline` (each string's header says what was
kept, dropped and changed). Each wrapper is appended to the engine module and routes the engine's entry point to the
competitor's kernel for the dtypes and layouts it supports, to the engine's own kernel otherwise. Apache-2.0 notices
are kept in every string."""

VERSION = "0.32.0"            # github.com/vllm-project/vllm tag every source here is taken from

FUSED_MOE_KERNEL = r'''# Vendored from vLLM v0.32.0 vllm/model_executor/layers/fused_moe/fused_moe.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
@triton.jit
def fused_moe_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    b_bias_ptr,
    a_scale_ptr,
    b_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_asm,
    stride_ask,
    stride_bse,
    stride_bsk,
    stride_bsn,
    stride_bbe,  # bias expert stride
    stride_bbn,  # bias N stride
    # Block size for block-wise quantization
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    naive_block_assignment: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    use_fp8_w8a8: tl.constexpr,
    use_int8_w8a8: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    per_channel_quant: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SWAP_AB: tl.constexpr,
    # Tensor-descriptor path for the A gather and B load in the K-loop.
    USE_TD: tl.constexpr = False,
):
    """Implements the fused computation for a Mixture of Experts (MOE) using
    token and expert matrices.

    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K), where '*' can
        be any shape representing batches and K is the feature dimension of
        each token.
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is
        the number of experts, K is the input feature dimension, and N is
        the output feature dimension.
    - C: The output cache tensor with shape (M, topk, N), where M is the
        total number of tokens post padding, topk is the number of times
        each token is repeated, and N is the output feature dimension.
    - sorted_token_ids: A tensor containing the sorted indices of tokens,
        repeated topk times and arranged by the expert index they are
        assigned to.
    - expert_ids: A tensor containing the indices of the expert for each
        block. It determines which expert matrix from B should be used for
        each block in A.
    - naive_block_assignment: A boolean flag indicating whether to use naive
        token wise block assignment. If True, each block corresponds to a
        single token.
    This kernel performs the multiplication of a token by its corresponding
    expert matrix as determined by `expert_ids`. The sorting of
    `sorted_token_ids` by expert index and padding ensures divisibility by
    BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix
    multiplication across different blocks processed by the same expert.
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    offs = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    if not naive_block_assignment:
        offs_token_id = pid_m * BLOCK_SIZE_M + offs
        offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    else:
        offs_token = tl.where(
            offs == 0,
            pid_m,  # first element = pid_m
            num_valid_tokens,  # remaining elements = constant
        )
    # Cast to int64 to prevent overflow in stride*offset products
    # (e.g. stride_cm * offs_token can exceed int32 for large token counts)
    offs_token = offs_token.to(tl.int64)

    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts == -1:
        # -----------------------------------------------------------
        # Write back zeros to the output when the expert is not
        # in the current expert parallel rank.
        write_zeros_to_output(
            c_ptr,
            stride_cm,
            stride_cn,
            pid_n,
            N,
            offs_token,
            token_mask,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            compute_type,
        )
        return

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    # TD gather and the SWAP_AB accumulator layout are mutually exclusive.
    tl.static_assert(not (USE_TD and SWAP_AB))
    if USE_TD:
        # ``tt.descriptor_gather`` requires block_shape[0] == 1 and i32 idx.
        m_td = num_valid_tokens // top_k
        a_desc = tl.make_tensor_descriptor(
            base=a_ptr,
            shape=(m_td, K),
            strides=(stride_am, stride_ak),
            block_shape=(1, BLOCK_SIZE_K),
        )
        b_desc = tl.make_tensor_descriptor(
            base=b_ptr + off_experts * stride_be,
            shape=(N, K),
            strides=(stride_bn, stride_bk),
            block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_K),
        )
        gather_idx = (offs_token // top_k).to(tl.int32)
    elif SWAP_AB:
        a_ptrs = a_ptr + (
            offs_k[:, None] * stride_ak + offs_token[None, :] // top_k * stride_am
        )
        b_ptrs = (
            b_ptr
            + off_experts * stride_be
            + (offs_bn[:, None] * stride_bn + offs_k[None, :] * stride_bk)
        )
    else:
        a_ptrs = a_ptr + (
            offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
        )
        b_ptrs = (
            b_ptr
            + off_experts * stride_be
            + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
        )
    if use_int8_w8a16:
        b_scale_ptrs = (
            b_scale_ptr + off_experts * stride_bse + offs_bn[None, :] * stride_bsn
        )
        b_scale = tl.load(b_scale_ptrs)

    if use_fp8_w8a8 or use_int8_w8a8:
        # block-wise
        if group_k > 0 and group_n > 0:
            a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
            offs_bsn = offs_bn // group_n
            b_scale_ptrs = (
                b_scale_ptr + off_experts * stride_bse + offs_bsn * stride_bsn
            )
        # channel-wise
        elif per_channel_quant:
            b_scale_ptrs = (
                b_scale_ptr + off_experts * stride_bse + offs_bn[None, :] * stride_bsn
            )
            b_scale = tl.load(b_scale_ptrs)
            # Load per-token scale for activations
            a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
            a_scale = tl.load(a_scale_ptrs, mask=token_mask, other=0.0)[:, None]
        # tensor-wise
        else:
            a_scale = tl.load(a_scale_ptr)
            b_scale = tl.load(b_scale_ptr + off_experts)
    if HAS_BIAS:
        # bias shape: [num_experts, N]
        bias_ptrs = b_bias_ptr + off_experts * stride_bbe + offs_bn * stride_bbn
        bias = tl.load(bias_ptrs, mask=(offs_bn < N), other=0.0)
    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    if SWAP_AB:
        accumulator = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
    else:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the
        # K dimension.
        if USE_TD:
            a = a_desc.gather(gather_idx, k * BLOCK_SIZE_K)
            b = b_desc.load([pid_n * BLOCK_SIZE_N, k * BLOCK_SIZE_K]).T
        elif SWAP_AB:
            a_mask = (offs_k[:, None] < K - k * BLOCK_SIZE_K) & token_mask[None, :]
            b_mask = offs_k[None, :] < K - k * BLOCK_SIZE_K
            a = tl.load(a_ptrs, mask=a_mask, other=0.0)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        else:
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                other=0.0,
            )
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        if use_int8_w8a16:
            accumulator = tl.dot(a, b.to(compute_type), acc=accumulator)
        elif use_fp8_w8a8 or use_int8_w8a8:
            if group_k > 0 and group_n > 0:
                k_start = k * BLOCK_SIZE_K
                offs_ks = k_start // group_k
                a_scale = tl.load(
                    a_scale_ptrs + offs_ks * stride_ask, mask=token_mask, other=0.0
                )
                b_scale = tl.load(b_scale_ptrs + offs_ks * stride_bsk)
                if SWAP_AB:
                    accumulator += tl.dot(b, a) * b_scale[:, None] * a_scale[None, :]
                else:
                    accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]
            else:
                if use_fp8_w8a8:
                    # acc used to enable fp8_fast_accum
                    if SWAP_AB:
                        accumulator = tl.dot(b, a, acc=accumulator)
                    else:
                        accumulator = tl.dot(a, b, acc=accumulator)
                else:
                    accumulator += tl.dot(a, b)
        else:
            accumulator += tl.dot(a, b)
        if not USE_TD:
            # Advance the ptrs to the next K block.
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

    if SWAP_AB:
        accumulator = tl.trans(accumulator, (1, 0))

    # Dequantization for supported quantization schemes:
    #   - int8_w8a16
    #   - fp8_w8a8
    #   - int8_w8a8
    # Accumulator and scalings are in float32 to preserve numerical accuracy.
    if use_int8_w8a16:
        accumulator = accumulator * b_scale
    elif (use_fp8_w8a8 or use_int8_w8a8) and not (group_k > 0 and group_n > 0):
        accumulator = accumulator * a_scale * b_scale

    # Bias addition:
    # Bias must be applied after dequantization:
    #   - Since bias is typically not quantized
    #   - Bias should not be scaled by quantization factors
    if HAS_BIAS:
        accumulator += bias[None, :]

    # Router (MoE) weight multiplication:
    # This multiplication MUST be performed in float32 before any precision
    # conversion to ensure numerical stability, which is especially critical
    # on ROCm platforms.
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(
            topk_weights_ptr + offs_token,
            mask=token_mask,
            other=0,
        )
        accumulator *= moe_weight[:, None]

    # Final precision conversion:
    # Cast once at the end to the desired compute/output dtype.
    accumulator = accumulator.to(compute_type)

    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)
'''

WRITE_ZEROS = r'''# Vendored from vLLM v0.32.0 vllm/model_executor/layers/fused_moe/fused_moe.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
@triton.jit
def write_zeros_to_output(
    c_ptr,
    stride_cm,
    stride_cn,
    pid_n,
    N,
    offs_token,
    token_mask,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    compute_type,
):
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)
'''

# vLLM's tuned table for E=128, N=768 on the H200 (configs/E=128,N=768,device_name=NVIDIA_H200.json): {tokens: config}
FUSED_MOE_CONFIGS = {
    1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
    2: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
    4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 2},
    8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
    16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 2},
    24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
    32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 2},
    48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 3},
    64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 2},
    96: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 2},
    128: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 2},
    256: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 2},
    512: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
    1024: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 16, 'num_warps': 8, 'num_stages': 4},
    1536: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 4},
    2048: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 4},
    3072: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 4},
    4096: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 16, 'num_warps': 8, 'num_stages': 4},
}

DECODE_ATTENTION = r'''# Vendored from vLLM v0.32.0 vllm/v1/attention/ops/triton_decode_attention.py: tanh, _fwd_grouped_kernel_stage1,
# _fwd_kernel_stage2, unchanged. Its header follows.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/sgl-project/sglang/blob/9f635ea50de920aa507f486daafba26a5b837574/python/sglang/srt/layers/attention/triton_ops/decode_attention.py
# which was originally adapted from
# https://github.com/ModelTC/lightllm/blob/96353e868a840db4d103138caf15ed9dbea8c186/lightllm/models/deepseek2/triton_kernel/gqa_flash_decoding_stage1.py
# https://github.com/ModelTC/lightllm/blob/96353e868a840db4d103138caf15ed9dbea8c186/lightllm/models/deepseek2/triton_kernel/gqa_flash_decoding_stage2.py

# Changes:
# - Add support for page size >= 1.

# Copyright 2025 vLLM Team
# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


@triton.jit
def tanh(x):
    # Tanh is just a scaled sigmoid
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def _fwd_grouped_kernel_stage1(
    Q,
    K_Buffer,
    V_Buffer,
    sm_scale,
    Req_to_tokens,
    B_Seqlen,
    Att_Out,
    stride_req_to_tokens_b,
    stride_qbs,
    stride_qh,
    stride_buf_kpbs,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vpbs,
    stride_buf_vbs,
    stride_buf_vh,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    k_scale,
    v_scale,
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    logit_cap: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
    IS_MLA: tl.constexpr = False,
):
    cur_batch = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    cur_kv_head = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)
    split_kv_id = tl.program_id(2)

    VALID_BLOCK_H: tl.constexpr = BLOCK_H if kv_group_num > BLOCK_H else kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < q_head_num)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_req_idx = cur_batch

    offs_q = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_d[None, :]
    q = tl.load(
        Q + offs_q,
        mask=(mask_h[:, None]) & (mask_d[None, :]),
        other=0.0,
        cache_modifier=".ca",
    )

    if BLOCK_DPE > 0:
        offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
        mask_dpe = offs_dpe < Lk
        off_qpe = (
            cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_dpe[None, :]
        )
        qpe = tl.load(
            Q + off_qpe,
            mask=(mask_h[:, None]) & (mask_dpe[None, :]),
            other=0.0,
            cache_modifier=".ca",
        )

    kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        base_offs_k = cur_kv_head * stride_buf_kh + offs_d[:, None]
        base_offs_v = cur_kv_head * stride_buf_vh + offs_dv[None, :]
        if BLOCK_DPE > 0:
            base_offs_kpe = cur_kv_head * stride_buf_kh + offs_dpe[:, None]

        ks = tl.load(k_scale)
        vs = tl.load(v_scale)
        for start_n in tl.range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            kv_page_number = tl.load(
                Req_to_tokens
                + stride_req_to_tokens_b * cur_batch_req_idx
                + offs_n // PAGE_SIZE,
                mask=offs_n < split_kv_end,
                other=0,
                cache_modifier=".ca",
            ).to(tl.int64)  # page_number * page stride overflows int32
            kv_off_k = (
                kv_page_number * stride_buf_kpbs + (offs_n % PAGE_SIZE) * stride_buf_kbs
            )

            # explicitly facilitate overlapping load/compute
            offs_buf_k = kv_off_k[None, :] + base_offs_k
            k = tl.load(
                K_Buffer + offs_buf_k,
                mask=(offs_n[None, :] < split_kv_end) & (mask_d[:, None]),
                other=0.0,
                cache_modifier=".cg",
            )

            if k.dtype.is_fp8():
                k = (k.to(tl.float32) * ks).to(q.dtype)
            qk = tl.dot(q, k.to(q.dtype))
            if BLOCK_DPE > 0:
                offs_buf_kpe = kv_off_k[None, :] + base_offs_kpe
                kpe = tl.load(
                    K_Buffer + offs_buf_kpe,
                    mask=(offs_n[None, :] < split_kv_end) & (mask_dpe[:, None]),
                    other=0.0,
                    cache_modifier=".cg",
                )
                if kpe.dtype.is_fp8():
                    kpe = (kpe.to(tl.float32) * ks).to(qpe.dtype)
                qk += tl.dot(qpe, kpe.to(qpe.dtype))
            qk *= sm_scale

            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            qk = tl.where(
                mask_h[:, None] & (offs_n[None, :] < split_kv_end), qk, float("-inf")
            )

            if not IS_MLA:
                kv_off_v = (
                    kv_page_number * stride_buf_vpbs
                    + (offs_n % PAGE_SIZE) * stride_buf_vbs
                )
                offs_buf_v = kv_off_v[:, None] + base_offs_v
                v = tl.load(
                    V_Buffer + offs_buf_v,
                    mask=(offs_n[:, None] < split_kv_end) & (mask_dv[None, :]),
                    other=0.0,
                )
                if v.dtype.is_fp8():
                    v = (v.to(tl.float32) * vs).to(q.dtype)
            else:
                # MLA uses a single c_kv.
                # loading the same c_kv to interpret it as v is not necessary.
                # transpose the existing c_kv (aka k) for the dot product.
                v = tl.trans(k)

            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            acc *= re_scale[:, None]
            acc += tl.dot(p.to(v.dtype), v)

            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

        offs_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head[:, None] * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_dv[None, :]
        )

        tl.store(
            Att_Out + offs_mid_o,
            acc / e_sum[:, None],
            mask=(mask_h[:, None]) & (mask_dv[None, :]),
        )

        offs_mid_o_1 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
            + Lv
        )

        tl.store(
            Att_Out + offs_mid_o_1,
            e_max + tl.log(e_sum),
            mask=mask_h,
        )


@triton.jit
def _fwd_kernel_stage2(
    Mid_O,
    o,
    lse,
    B_Seqlen,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_obs,
    stride_oh,
    stride_lse_bs,
    NUM_KV_SPLITS: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    Lv: tl.constexpr,
    OUTPUT_FP16: tl.constexpr = 0,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)

    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv

    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_logic = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + Lv

    for split_kv_id in range(0, NUM_KV_SPLITS):
        kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
        split_kv_start = kv_len_per_split * split_kv_id
        split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

        if split_kv_end > split_kv_start:
            tv = tl.load(
                Mid_O + offs_v + split_kv_id * stride_mid_os, mask=mask_d, other=0.0
            )
            tlogic = tl.load(Mid_O + offs_logic + split_kv_id * stride_mid_os)
            n_e_max = tl.maximum(tlogic, e_max)

            old_scale = tl.exp(e_max - n_e_max)
            acc *= old_scale
            exp_logic = tl.exp(tlogic - n_e_max)
            acc += exp_logic * tv

            e_sum = e_sum * old_scale + exp_logic
            e_max = n_e_max

    result = acc / e_sum
    if OUTPUT_FP16:
        result = result.to(tl.float16)
    tl.store(
        o + cur_batch * stride_obs + cur_head * stride_oh + offs_d,
        result,
        mask=mask_d,
    )
    lse_val = e_max + tl.log(e_sum)
    tl.store(
        lse + cur_batch * stride_lse_bs + cur_head,
        lse_val,
    )
'''

_LICENSE = r''' * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
'''

# What every norm and activation port includes: system and torch headers only, vLLM's cub sum op.
_PRELUDE = r'''#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cub/cub.cuh>
#include <algorithm>
#include <cstdint>

#if CUB_VERSION >= 200800
  #include <cuda/std/functional>
using CubAddOp = cuda::std::plus<>;
#else
using CubAddOp = cub::Sum;
#endif
'''

RMS_NORM_CUDA = r'''/*
 * Adapted from vLLM v0.32.0 csrc/layernorm_kernels.cu (rms_norm_kernel), with csrc/cub_helpers.h and the aligned
 * path of csrc/quantization/vectorization_utils.cuh (vectorize_read_with_alignment).
''' + _LICENSE + r''' *
 * Kept from vLLM: one block per row; the sum of squares read in 16-byte vectors and reduced by cub::BlockReduce over
 * at most 1024 threads; x * rsqrt(mean + eps) * w in fp32 with one rounding; the block size (hidden / 8 threads, at
 * most 1024 under 256 rows, else 256). Dropped: every dtype but bf16, vector widths other than 8 and the read's
 * unaligned prefix and tail (rows that are not 16-byte vectors go to the engine's kernel), 3-D and 4-D inputs, 2-D
 * weights, no weight, the batch-invariant launch, the torch stable bindings and ROCm. Changed: bf16 conversions are
 * CUDA intrinsics (torch's extension build turns implicit bf16 conversions off).
 */
''' + _PRELUDE + r'''
namespace {

constexpr int VEC_SIZE = 8;

template <typename T, int N>
struct __align__(N * sizeof(T)) vec_n_t {
  T val[N];
};

__global__ void rms_norm_kernel(
    __nv_bfloat16* __restrict__ out,           // [..., hidden_size]
    const __nv_bfloat16* __restrict__ input,   // [..., hidden_size]
    const int64_t input_stride,                // input.stride(-2)
    const __nv_bfloat16* __restrict__ weight,  // [hidden_size]
    const float epsilon, const int num_tokens, const int hidden_size) {
  using vec_t = vec_n_t<__nv_bfloat16, VEC_SIZE>;
  __shared__ float s_variance;
  float variance = 0.0f;
  const __nv_bfloat16* input_row = input + blockIdx.x * input_stride;
  auto* v_in = reinterpret_cast<const vec_t*>(input_row);

  for (int i = threadIdx.x; i < hidden_size / VEC_SIZE; i += blockDim.x) {
    vec_t tmp = v_in[i];
#pragma unroll
    for (int j = 0; j < VEC_SIZE; ++j) {
      float x = __bfloat162float(tmp.val[j]);
      variance += x * x;
    }
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  variance = BlockReduce(reduceStore).Reduce(variance, CubAddOp{}, blockDim.x);

  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  __nv_bfloat16* out_row = out + blockIdx.x * hidden_size;
  auto* v_w = reinterpret_cast<const vec_t*>(weight);
  auto* v_out = reinterpret_cast<vec_t*>(out_row);
  for (int i = threadIdx.x; i < hidden_size / VEC_SIZE; i += blockDim.x) {
    vec_t dst;
    vec_t src1 = v_in[i];
    vec_t src2 = v_w[i];
#pragma unroll
    for (int j = 0; j < VEC_SIZE; j++) {
      float x = __bfloat162float(src1.val[j]);
      float w = __bfloat162float(src2.val[j]);
      dst.val[j] = __float2bfloat16(x * s_variance * w);
    }
    v_out[i] = dst;
  }
}

}  // namespace

// out [rows, hidden] contiguous, input [rows, hidden] with unit last stride, weight [hidden]; all bf16, 16-byte rows.
void rms_norm_cuda(torch::Tensor out, torch::Tensor input, torch::Tensor weight, double epsilon) {
  TORCH_CHECK(input.is_cuda() && input.dim() == 2 && input.stride(1) == 1, "input: a [rows, hidden] CUDA tensor");
  TORCH_CHECK(out.is_contiguous() && out.sizes() == input.sizes() && weight.is_contiguous(), "out, weight: contiguous");
  const int hidden_size = input.size(1);
  const int num_tokens = input.size(0);
  TORCH_CHECK(hidden_size % VEC_SIZE == 0 && input.stride(0) % VEC_SIZE == 0, "rows of 16-byte vectors only");
  if (num_tokens == 0) return;
  // For large num_tokens, use smaller blocks to increase SM concurrency.
  const int max_block_size = (num_tokens < 256) ? 1024 : 256;
  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size / VEC_SIZE, max_block_size));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  rms_norm_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
      input.stride(0), reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()), static_cast<float>(epsilon),
      num_tokens, hidden_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
'''

FUSED_ADD_RMS_NORM_CUDA = r'''/*
 * Adapted from vLLM v0.32.0 csrc/layernorm_kernels.cu (fused_add_rms_norm_kernel, its vectorized specialization),
 * with _f16Vec from csrc/type_convert.cuh and csrc/cub_helpers.h.
''' + _LICENSE + r''' *
 * Kept from vLLM: one block per row; the sum h + a in bf16x2 adds over 16-byte vectors (_f16Vec, width 8), stored and
 * read back for the second pass; the sum of squares reduced by cub::BlockReduce; x * rsqrt(mean + eps) * w in fp32 with
 * one rounding; the block size (min(hidden, 1024) under 256 rows, else min(hidden, 256)). Dropped: every dtype but
 * bf16, the generic width-0 kernel (rows that are not 16-byte vectors go to the engine's kernel), no weight, row
 * strides, the batch-invariant launch, the torch stable bindings and ROCm. Changed: vLLM writes in place (the residual
 * becomes the sum, the input its norm); this writes both to new tensors and reads h and a only, as the engine's
 * add_rmsnorm returns (h + a, norm). bf16 arithmetic is spelled with CUDA intrinsics (__hadd2) as torch's extension
 * build turns the bf16 operators off.
 */
''' + _PRELUDE + r'''
namespace {

/* Vector POD struct to generate vectorized and packed BF16 ops. Alignment to 16 bytes is required to use 128-bit
   global memory ops. */
template <int width>
struct alignas(16) _f16Vec {
  static_assert(width > 0 && width % 2 == 0 && (width & (width - 1)) == 0, "width: an even power of 2");
  __nv_bfloat16 data[width];

  __device__ _f16Vec& operator+=(const _f16Vec<width>& other) {
#pragma unroll
    for (int i = 0; i < width; i += 2) {
      __nv_bfloat162 temp = __hadd2(__halves2bfloat162(data[i], data[i + 1]),
                                    __halves2bfloat162(other.data[i], other.data[i + 1]));
      data[i] = temp.x;
      data[i + 1] = temp.y;
    }
    return *this;
  }

  __device__ float sum_squares() const {
    float result = 0.0f;
#pragma unroll
    for (int i = 0; i < width; i += 2) {
      float2 z = __bfloat1622float2(__halves2bfloat162(data[i], data[i + 1]));
      result += z.x * z.x + z.y * z.y;
    }
    return result;
  }
};

template <int width>
__global__ void fused_add_rms_norm_kernel(
    __nv_bfloat16* __restrict__ out,             // [..., hidden_size]: rmsnorm(input + residual)
    __nv_bfloat16* __restrict__ residual_out,    // [..., hidden_size]: input + residual
    const __nv_bfloat16* __restrict__ input,     // [..., hidden_size]
    const __nv_bfloat16* __restrict__ residual,  // [..., hidden_size]
    const __nv_bfloat16* __restrict__ weight,    // [hidden_size]
    const float epsilon, const int num_tokens, const int hidden_size) {
  const int vec_hidden_size = hidden_size / width;
  __shared__ float s_variance;
  float variance = 0.0f;
  auto* __restrict__ input_v = reinterpret_cast<const _f16Vec<width>*>(input);
  auto* __restrict__ residual_v = reinterpret_cast<const _f16Vec<width>*>(residual);
  auto* __restrict__ residual_out_v = reinterpret_cast<_f16Vec<width>*>(residual_out);
  auto* __restrict__ out_v = reinterpret_cast<_f16Vec<width>*>(out);
  auto* __restrict__ weight_v = reinterpret_cast<const _f16Vec<width>*>(weight);

  for (int idx = threadIdx.x; idx < vec_hidden_size; idx += blockDim.x) {
    int64_t id = static_cast<int64_t>(blockIdx.x) * vec_hidden_size + idx;
    _f16Vec<width> temp = input_v[id];
    temp += residual_v[id];
    variance += temp.sum_squares();
    residual_out_v[id] = temp;
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  variance = BlockReduce(reduceStore).Reduce(variance, CubAddOp{}, blockDim.x);

  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < vec_hidden_size; idx += blockDim.x) {
    int64_t id = static_cast<int64_t>(blockIdx.x) * vec_hidden_size + idx;
    _f16Vec<width> res = residual_out_v[id];
    _f16Vec<width> w = weight_v[idx];
    _f16Vec<width> o;
#pragma unroll
    for (int j = 0; j < width; ++j) {
      float x = __bfloat162float(res.data[j]);
      float wf = __bfloat162float(w.data[j]);
      o.data[j] = __float2bfloat16(x * s_variance * wf);
    }
    out_v[id] = o;
  }
}

}  // namespace

// out, residual_out, input, residual [rows, hidden] contiguous bf16, weight [hidden]: 16-byte aligned rows.
void fused_add_rms_norm_cuda(torch::Tensor out, torch::Tensor residual_out, torch::Tensor input,
                             torch::Tensor residual, torch::Tensor weight, double epsilon) {
  for (const auto& t : {out, residual_out, input, residual, weight}) {
    TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.scalar_type() == at::kBFloat16, "contiguous bf16 CUDA tensors");
  }
  TORCH_CHECK(input.dim() == 2 && residual.sizes() == input.sizes(), "input, residual: [rows, hidden]");
  constexpr int vector_width = 8;
  const int hidden_size = input.size(1);
  const int num_tokens = input.size(0);
  TORCH_CHECK(hidden_size % vector_width == 0, "rows of 16-byte vectors only");
  if (num_tokens == 0) return;
  /* This kernel is memory-latency bound in many scenarios. When num_tokens is large, a smaller block size allows for
     increased block occupancy on CUs and better latency hiding on global mem ops. */
  const int max_block_size = (num_tokens < 256) ? 1024 : 256;
  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, max_block_size));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  fused_add_rms_norm_kernel<vector_width><<<grid, block, 0, stream>>>(
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), reinterpret_cast<__nv_bfloat16*>(residual_out.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()), static_cast<float>(epsilon), num_tokens,
      hidden_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
'''

SILU_AND_MUL_CUDA = r'''/*
 * Adapted from vLLM v0.32.0 csrc/activation_kernels.cu (act_and_mul_kernel with silu_kernel, silu_and_mul's launch)
 * and csrc/cuda_vec_utils.cuh (PackedVec, ld128, st128, cast_to_float2, cast_to_packed).
''' + _LICENSE + r''' *
 * Kept from vLLM: one block per row; the gate and up halves read with 128-bit __ldg loads and the output written with
 * 128-bit stores (PackedVec of 4 bf16x2); per pair, SiLU of the gate in fp32 rounded to bf16, then times the up value
 * in fp32 and rounded again (packed_compute, act first, alpha 1, beta 0, no clamp); the block size min(d / 8, 1024).
 * Dropped: every dtype but bf16, the scalar path for d not a multiple of 8 (the engine's kernel takes those), the
 * 256-bit SM100 path, the clamp, the other activations, the torch stable bindings and ROCm.
 */
''' + _PRELUDE + r'''
namespace {

// PackedVec<__nv_bfloat16, false>: 16 bytes, 4 bf16x2.
struct alignas(16) PackedVec {
  static constexpr int NUM_ELTS = 4;
  __nv_bfloat162 elts[NUM_ELTS];
};

// 128-bit load / store via __ldg (read-only cache hint).
__device__ __forceinline__ void ld128(PackedVec& val, const PackedVec* ptr) {
  *reinterpret_cast<int4*>(&val) = __ldg(reinterpret_cast<const int4*>(ptr));
}

__device__ __forceinline__ void st128(PackedVec& val, PackedVec* ptr) {
  *reinterpret_cast<int4*>(ptr) = *reinterpret_cast<int4*>(&val);
}

__device__ __forceinline__ __nv_bfloat162 packed_silu_kernel(const __nv_bfloat162& val, const float alpha) {
  // x * sigmoid(alpha * x)
  float2 fval = __bfloat1622float2(val);
  fval.x = fval.x / (1.0f + expf(-fval.x * alpha));
  fval.y = fval.y / (1.0f + expf(-fval.y * alpha));
  return __float22bfloat162_rn(fval);
}

__device__ __forceinline__ __nv_bfloat162 packed_compute(const __nv_bfloat162& x, const __nv_bfloat162& y,
                                                         const float alpha, const float beta) {
  // act_first: gate is the activated half -> alpha applies to gate; beta is added to up (the non-activated half).
  float2 u = __bfloat1622float2(y);
  float2 activated = __bfloat1622float2(packed_silu_kernel(x, alpha));
  activated.x *= u.x + beta;
  activated.y *= u.y + beta;
  return __float22bfloat162_rn(activated);
}

__global__ void act_and_mul_kernel(
    __nv_bfloat16* __restrict__ out,          // [..., d]
    const __nv_bfloat16* __restrict__ input,  // [..., 2, d]
    const int d, const float alpha, const float beta) {
  const int64_t token_idx = blockIdx.x;
  const __nv_bfloat16* x_ptr = input + token_idx * 2 * d;
  const __nv_bfloat16* y_ptr = x_ptr + d;
  __nv_bfloat16* out_ptr = out + token_idx * d;

  const PackedVec* x_vec = reinterpret_cast<const PackedVec*>(x_ptr);
  const PackedVec* y_vec = reinterpret_cast<const PackedVec*>(y_ptr);
  PackedVec* out_vec = reinterpret_cast<PackedVec*>(out_ptr);
  const int num_vecs = d / 2 / PackedVec::NUM_ELTS;

  for (int i = threadIdx.x; i < num_vecs; i += blockDim.x) {
    PackedVec x, y;
    ld128(x, &x_vec[i]);
    ld128(y, &y_vec[i]);
#pragma unroll
    for (int j = 0; j < PackedVec::NUM_ELTS; j++) {
      x.elts[j] = packed_compute(x.elts[j], y.elts[j], alpha, beta);
    }
    st128(x, &out_vec[i]);
  }
}

}  // namespace

// out [rows, d] and input [rows, 2d], contiguous bf16, 16-byte aligned, d a multiple of 8.
void silu_and_mul_cuda(torch::Tensor out, torch::Tensor input) {
  TORCH_CHECK(input.is_cuda() && input.is_contiguous() && out.is_contiguous() && input.dim() == 2, "contiguous 2-D");
  TORCH_CHECK(input.scalar_type() == at::kBFloat16 && out.scalar_type() == at::kBFloat16, "bf16 only");
  const int d = input.size(1) / 2;
  const int64_t num_tokens = input.size(0);
  TORCH_CHECK(d % 8 == 0 && out.size(0) == num_tokens && out.size(1) == d, "out [rows, d] with d a multiple of 8");
  if (num_tokens == 0) return;
  dim3 grid(num_tokens);
  dim3 block(std::min(d / 8, 1024));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  act_and_mul_kernel<<<grid, block, 0, stream>>>(reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
                                                 reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()), d, 1.0f,
                                                 0.0f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
'''

RMS_NORM_WRAPPER = f'''

# vLLM's rms_norm (csrc/layernorm_kernels.cu, Apache-2.0) as CUDA C++ built when this module is imported; its header
# comment says what was kept, dropped and changed. The build dir and the architecture are the evaluator's.
from torch.utils.cpp_extension import load_inline

_RMS_CPP = "void rms_norm_cuda(torch::Tensor out, torch::Tensor input, torch::Tensor weight, double epsilon);"
_RMS_CUDA = r"""{RMS_NORM_CUDA}"""
_rms_ext = load_inline(name="vllm_rms_norm", cpp_sources=[_RMS_CPP], cuda_sources=[_RMS_CUDA],
                       functions=["rms_norm_cuda"], extra_cuda_cflags=["-O3"], verbose=False)

_rmsnorm_engine = rmsnorm


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """vLLM's rms_norm kernel for bf16 rows of 16-byte vectors on CUDA; the engine's kernel otherwise."""
    D = x.shape[-1]
    x2 = x.reshape(-1, D)
    if not (x2.is_cuda and x2.dtype == weight.dtype == torch.bfloat16 and D % 8 == 0 and x2.stride(-1) == 1
            and x2.stride(0) % 8 == 0 and x2.data_ptr() % 16 == 0 and weight.is_contiguous()
            and weight.data_ptr() % 16 == 0):
        return _rmsnorm_engine(x, weight, eps)
    y = torch.empty(x2.shape, dtype=x2.dtype, device=x2.device)
    _rms_ext.rms_norm_cuda(y, x2, weight, eps)
    return y.view(x.shape)
'''

FUSED_ADD_RMS_NORM_WRAPPER = f'''

# vLLM's fused_add_rms_norm (csrc/layernorm_kernels.cu, Apache-2.0) as CUDA C++ built when this module is imported; its
# header comment says what was kept, dropped and changed. The build dir and the architecture are the evaluator's.
from torch.utils.cpp_extension import load_inline

_ADD_RMS_CPP = ("void fused_add_rms_norm_cuda(torch::Tensor out, torch::Tensor residual_out, torch::Tensor input, "
                "torch::Tensor residual, torch::Tensor weight, double epsilon);")
_ADD_RMS_CUDA = r"""{FUSED_ADD_RMS_NORM_CUDA}"""
_add_rms_ext = load_inline(name="vllm_fused_add_rms_norm", cpp_sources=[_ADD_RMS_CPP], cuda_sources=[_ADD_RMS_CUDA],
                           functions=["fused_add_rms_norm_cuda"], extra_cuda_cflags=["-O3"], verbose=False)

_add_rmsnorm_engine = add_rmsnorm


def add_rmsnorm(h: torch.Tensor, a: torch.Tensor, weight: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """(h + a, rmsnorm(h + a)) by vLLM's fused_add_rms_norm kernel for bf16 rows of 16-byte vectors on CUDA; the
    engine's kernel otherwise."""
    D = h.shape[-1]
    if not (h.is_cuda and h.dtype == a.dtype == weight.dtype == torch.bfloat16 and D % 8 == 0
            and weight.is_contiguous()):
        return _add_rmsnorm_engine(h, a, weight, eps)
    h2, a2 = h.reshape(-1, D).contiguous(), a.reshape(-1, D).contiguous()
    if h2.data_ptr() % 16 or a2.data_ptr() % 16 or weight.data_ptr() % 16:
        return _add_rmsnorm_engine(h, a, weight, eps)
    ho, y = torch.empty_like(h2), torch.empty_like(h2)
    _add_rms_ext.fused_add_rms_norm_cuda(y, ho, h2, a2, weight, eps)
    return ho.view(h.shape), y.view(h.shape)
'''

SILU_AND_MUL_WRAPPER = f'''

# vLLM's silu_and_mul (csrc/activation_kernels.cu, Apache-2.0) as CUDA C++ built when this module is imported; its
# header comment says what was kept, dropped and changed. The build dir and the architecture are the evaluator's.
from torch.utils.cpp_extension import load_inline

_SILU_CPP = "void silu_and_mul_cuda(torch::Tensor out, torch::Tensor input);"
_SILU_CUDA = r"""{SILU_AND_MUL_CUDA}"""
_silu_ext = load_inline(name="vllm_silu_and_mul", cpp_sources=[_SILU_CPP], cuda_sources=[_SILU_CUDA],
                        functions=["silu_and_mul_cuda"], extra_cuda_cflags=["-O3"], verbose=False)

_silu_mul_engine = silu_mul


def silu_mul(x: torch.Tensor) -> torch.Tensor:
    """x [M, 2I] -> silu(x[:, :I]) * x[:, I:] by vLLM's act_and_mul kernel for bf16 with I a multiple of 8 on CUDA;
    the engine's kernel otherwise (fp32 inputs among them)."""
    M, two_i = x.shape
    if not (x.is_cuda and x.dtype == torch.bfloat16 and two_i % 16 == 0):
        return _silu_mul_engine(x)
    x = x.contiguous()
    if x.data_ptr() % 16:
        return _silu_mul_engine(x)
    y = torch.empty(M, two_i // 2, dtype=torch.bfloat16, device=x.device)
    _silu_ext.silu_and_mul_cuda(y, x)
    return y
'''

FUSED_MOE_SCATTER_WRAPPER = '''

_moe_gemm_scatter_engine = moe_gemm_scatter


def moe_gemm_scatter(a, w, sorted_ids, expert_ids, M, tok, pair_w, out, block_m=16, w_scale=None, a_scale=None,
                     tile=None):
    """vLLM's down projection for fp8 weights with bf16 rows: its kernel at its tuned config for the token count (block
    rows: ours), each pair's row times its routing weight (MUL_ROUTED_WEIGHT) written in bf16, then added into its
    token's fp32 row (vLLM sums a token's rows with moe_sum); the engine's kernel for the other paths."""
    if w.dtype != torch.float8_e4m3fn or a.dtype != torch.bfloat16 or a_scale is not None or tile is not None:
        return _moe_gemm_scatter_engine(a, w, sorted_ids, expert_ids, M, tok, pair_w, out, block_m, w_scale, a_scale,
                                        tile)
    E, N, K = w.shape
    cfg = _VLLM_CONFIGS[min(_VLLM_CONFIGS, key=lambda t: abs(t - out.shape[0]))]
    C = torch.empty(M, N, dtype=a.dtype, device=a.device)
    EM = expert_ids.numel() * block_m
    grid = (triton.cdiv(EM, block_m) * triton.cdiv(N, cfg["BLOCK_SIZE_N"]),)
    fused_moe_kernel[grid](
        a_ptr=a, tok_ptr=a, b_ptr=w, c_ptr=C, b_bias_ptr=None, a_scale_ptr=None, b_scale_ptr=w_scale,
        topk_weights_ptr=pair_w, sorted_token_ids_ptr=sorted_ids, expert_ids_ptr=expert_ids,
        num_tokens_post_padded_ptr=None, N=N, K=K, EM=EM, num_valid_tokens=M, stride_am=a.stride(0),
        stride_ak=a.stride(1), stride_be=w.stride(0), stride_bk=w.stride(2), stride_bn=w.stride(1),
        stride_cm=C.stride(0), stride_cn=C.stride(1), stride_asm=0, stride_ask=0, stride_bse=w_scale.stride(0),
        stride_bsk=0, stride_bsn=w_scale.stride(1), stride_bbe=0, stride_bbn=0, group_n=0, group_k=0,
        naive_block_assignment=False, BLOCK_SIZE_M=block_m, BLOCK_SIZE_N=cfg["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=cfg["BLOCK_SIZE_K"], GROUP_SIZE_M=cfg["GROUP_SIZE_M"], SPLIT_K=1, MUL_ROUTED_WEIGHT=True,
        top_k=1, compute_type=tl.bfloat16, use_fp8_w8a8=False, use_int8_w8a8=False, use_int8_w8a16=True,
        per_channel_quant=False, HAS_BIAS=False, SWAP_AB=False, HAS_TOK=False, NUM_EXPERTS=E,
        num_warps=cfg.get("num_warps", 4), num_stages=cfg.get("num_stages", 3))
    out.index_add_(0, tok, C.float())
'''

DECODE_WRAPPER = '''

_paged_decode_attention_engine = paged_decode_attention
_UNIT_SCALE: dict = {}


def paged_decode_attention(q, k_pool, v_pool, block_tables, seq_lens, scale=1.0):
    """vLLM's Triton decode attention for GQA: its grouped stage 1 over KV splits and stage 2 merging them, reading our
    pools [blocks, kv heads, block_size, D] through their strides and our block tables as its page table; q bf16 or
    fp16, pools in q's dtype or fp8 e4m3 at a unit scale, on CUDA. The engine's kernel otherwise. The split count is
    vLLM's rule (triton_mla: next power of 2 of max length / 512, at most 2 per SM) on the longest length the block
    table holds, as the true one is on the device."""
    N, Hq, D = q.shape
    Hkv, bs = k_pool.shape[1], k_pool.shape[2]
    if (not q.is_cuda or q.dtype not in (torch.bfloat16, torch.float16) or Hq == Hkv or Hq % Hkv
            or k_pool.dtype not in (q.dtype, torch.float8_e4m3fn) or v_pool.dtype != k_pool.dtype
            or k_pool.stride(3) != 1 or v_pool.stride(3) != 1):
        return _paged_decode_attention_engine(q, k_pool, v_pool, block_tables, seq_lens, scale)
    q = q.contiguous()
    sms = torch.cuda.get_device_properties(q.device).multi_processor_count
    splits = min(triton.next_power_of_2(max(1, block_tables.shape[1] * bs // 512)), 2 * sms)
    dv = triton.next_power_of_2(D)
    att = torch.empty(N, Hq, splits, D + 1, dtype=torch.float32, device=q.device)
    o = torch.empty(N, Hq, D, dtype=q.dtype, device=q.device)
    lse = torch.empty(N, Hq, dtype=torch.float32, device=q.device)
    one = _UNIT_SCALE.get(q.device)
    if one is None:
        one = _UNIT_SCALE[q.device] = torch.ones((), dtype=torch.float32, device=q.device)
    group = Hq // Hkv
    _fwd_grouped_kernel_stage1[(N, triton.cdiv(Hq, min(16, group)), splits)](
        q, k_pool, v_pool, scale, block_tables, seq_lens, att, block_tables.stride(0), q.stride(0), q.stride(1),
        k_pool.stride(0), k_pool.stride(2), k_pool.stride(1), v_pool.stride(0), v_pool.stride(2), v_pool.stride(1),
        att.stride(0), att.stride(1), att.stride(2), one, one, kv_group_num=group, q_head_num=Hq, BLOCK_DMODEL=dv,
        BLOCK_DPE=0, BLOCK_DV=dv, BLOCK_N=32, BLOCK_H=16, NUM_KV_SPLITS=splits, PAGE_SIZE=bs, logit_cap=0.0,
        num_warps=4, num_stages=2, Lk=D, Lv=D, IS_MLA=False)
    _fwd_kernel_stage2[(N, Hq)](
        att, o, lse, seq_lens, att.stride(0), att.stride(1), att.stride(2), o.stride(0), o.stride(1), lse.stride(0),
        NUM_KV_SPLITS=splits, BLOCK_DV=dv, Lv=D, num_warps=4, num_stages=2)
    return o
'''
