# Runs

Every GPU run: one row in the table (newest first), details below. Append-only. Reports and outputs under `data/` are
local and archived in the store (`python -m store pull`).

| Date | Venue | What | GPU cost | Result | Evidence |
|---|---|---|---|---|---|
| 2026-10-16 | Nebius H200 (`lab-gpu-2`, the engine session's VM) | Layer-1 loop validated on a GPU (#46-#58), then OpenEvolve on `moe_gemm` through `ship` | ~$9.3 GPU (~2 h 5 min of kernel steps) + ~$1 model (estimate) | Every new evaluator piece runs; engine kernels pass every gate. Evolved `moe_gemm` 1.087x seen / 1.094x held-out, all gates pass; shipped as an engine branch, end-to-end judge not run | laptop `lab/runs/kernels-validate*/`, `lab/runs/kernels-moe/`, ledger `ev-20261016-9969664d6bb0`, inference-server `kernels/moe_gemm-20261016-154523` |
| 2026-10-15 | Verda H200 | Manual engine serving fixes (inference-server #107), #33 profiles check | ~$24 (4 h 25 min) | single_stream TPOT p99 21 -> 15.6 ms, saturated 0.097 -> 0.37 req/s, long_prompt invalid -> 2.66 req/s; gate inconclusive at concurrency 32 | `data/engine-h200-20261015/`, below |
| 2026-10-14/15 | Nebius H200 | Layer-1 kernel loop on `paged_decode` (branch `feat/kernel-loop`) | not recorded | GQA-packed decode kernel 3.88x on 196 seen shapes, passes held-out; end to end single_stream TPOT p99 21.6 -> 15.4 ms, saturated 0.100 -> 0.158 req/s (short tier, one run each); equiv not run | store `archive/kernels-2026-10-14-622240de7cc1b4aa.tar.zst` (gate results, best program, shapes), engine PR inference-server #106 |
| 2026-10-13 | Nebius H200 | Ceilings vs measurements, vLLM and the engine | not recorded | Every floor below every measurement; engine at 0.5-7% of ceiling, vLLM 19-94% | `knowledge/reference/inference-server/` |
| 2026-10-13 | Nebius H200 (x3) | `python -m lab.gpucheck` | not recorded | Every item passes after three fixes | below |
| 2026-10-10 | Nebius H200 | First agent dry run, one session | $2.53 model + ~1 h 20 min GPU | Loop works end to end on the GPU; single_stream TPOT p99 21.4 -> 13.1 ms, not submitted | below |
| 2026-10-08 | Nebius H200 | Profiler tools on the real engine | ~$0.87 | All four tools work unjailed; three fixes | below |
| 2026-10-05 | Nebius H200 | Full-tier correctness recalibration | ~$16.50 | Policy confirmed; FP8/INT8 pass, Int4 fails | `data/h200-run-20261005/REPORT.md`, `data/equiv-full/` |
| 2026-10-04 | Verda H200 | Dev-tier correctness calibration | $6.75 | Gate separates Int4; consistency results void | `data/h200-run-20261004/REPORT.md`, `data/equiv/` |

## 2026-10-16 — Layer-1 loop on a GPU: validation, then `moe_gemm` through `ship` (Nebius H200)
Run by the engine session on its VM (`lab-gpu-2`), BlameGraph main `a303313` then `0420b37`; engine = branch
`engine/gpu-session-2` (`eb7b683` + #110/#111 behind flags; `moe_kernel.py` and `paged_attention_kernel.py` as on
`eb7b683`). The kernel steps had the GPU to themselves; auto-stop armed throughout.

Phase A (`paged_decode`, 91 min, ~$6.8): capture + split = 143 seen / 27 held-out shapes. `gate` on the engine's own
kernel (#106's evolved one) and on `tests/fixtures/paged_decode_evolved.py` (the same kernel): every gate passes
(429 seen and 81 held-out cases, edge, memoization, values ×3/×0.01/−1, peak memory), no patch or timing false
positive, timing as CUDA-graph replays (`clock: cuda_graph`), paired ratios with ~1 µs IQR on 28 µs calls, 16-43%
of the roofline bound, speedup 1.0 and held-out speedup 1.0 as expected. Step-time share (32 requests × 64 tokens,
graphs on, 4.64 s window): kernels 48.9% of the window; `moe_gemm` 13.7% (`speedup_if_free` 1.159); `paged_decode`
read 0 because #106 renamed its kernels (fixed, #59); top unmatched: the CUTLASS grouped GEMM on the prefill path
6.2%, cuDNN SDPA 5.2%. Each full `gate` took ~45 min with the GPU near idle.

Phase B (`moe_gemm`): the first gate stopped after 17 min (~$1.3): py-spy put the time in generating the expert
weights (128×1536×2048) on the CPU for every check case, and the split held out 0 shapes (most calls at the largest
M, so the call-weighted cut equalled the max). Both fixed (#60: large inputs made on the device, roofline from the
timed inputs, a distinct-value cut). Redo: 44 seen / 10 held-out shapes; the full gate in < 1 min; the engine's own
kernel passes every gate (132 / 30 cases) at 37-78% of its roofline bound. OpenEvolve from the laptop (controller),
candidates judged on the VM over SSH: 20 iterations, 9.5 min, ~15 s per evaluation, Haiku 4.5 (0.8) + Opus 5.5
(0.2), 19 model calls (~192k prompt + ~14k response tokens, ~$1 at list prices: an estimate; the `kernels` record has
no cost, nothing priced it). Best (after re-timing the top 3): 1.0908 on the seen shapes. Its change: BLOCK_N/BLOCK_K
128/64 → 64/128, num_stages 3 → 4, `EVEN_K`/`EVEN_N` constexpr flags dropping the masks, `evict_first` on the
weight loads. `gate`: every gate passes; seen 1.087x (43/44 shapes beyond noise, 0.99-1.16x, one 1.42x), held-out
1.094x (10/10 beyond noise, 1.06-1.16x, 80-99.6% of roofline); error against the reference equal to the baseline's.
`ship`: inference-server branch `kernels/moe_gemm-20261016-154523` (`59e6808` on `eb7b683`, `moe_kernel.py` only,
+15/−6), pushed, no PR. Not run: the end-to-end judge (`lab.experiment` → `bench`, `equiv`, `submit`); the user
stopped here (Amdahl: ≤ ~1.2% end to end, likely within the noise bands). Kernel steps after Phase A ≈ 16 min, ~$1.2.

## 2026-10-15 — Engine serving fixes by hand (Verda H200, inference-server PR #107)
Short tier, seen split, bf16, `python -m regimes run` against the engine served with the target's env. Baseline of
#107's first commits (KV from 90% of raw free memory, batch 128, fair policy): single_stream 18.9 ms, saturated
0.32 req/s, long_prompt invalid again, this time 1,444 HTTP 500s = CUDA OOM (no cap on a batched prefill wave's
tokens). Fixed with `MAX_PREFILL_TOKENS` (32768, rows x longest prompt per wave) and KV sized after a measured
max-prefill peak: long_prompt valid at 1.56 req/s. fp8 experts (`torch._scaled_grouped_mm`, per-channel weights,
per-token activations): correct, no change in any regime. Profile (`lab.profile --profiler torch`, batch 1): 288,759
kernel launches in 64 decode steps (~4,500 per step), the CUTLASS grouped GEMM 3.5 ms of the ~20 ms step, the rest
1-3 us elementwise/copy/reduce ops: launch-bound, not bandwidth-bound. Microbenchmark: the paged prefill Triton
kernel at ~5 TFLOP/s vs torch SDPA ~600 (56x at 1k tokens, 136x at 16k; 1.8 s per layer at 32k). SDPA prefill
(prefix gathered from blocks + suffix, one flash call per row): long_prompt 2.13 req/s. Fused Triton kernels
(RMSNorm, add+RMSNorm, RMSNorm+RoPE, SiLU*up, softmax top-k): launches halved, batch-1 step 24 -> 13.3 ms,
single_stream 15.6 ms, saturated 0.37, long_prompt 2.66 req/s. A Triton MoE grouped GEMM (reads only active experts)
is exact in eager mode but asserts inside the decode CUDA graph under load: off by default. Gate (dev tier, conc 32)
on the fused+SDPA engine: inconclusive, 12/1,585 unanswered (429s), ratio 0.9951 (CI 0.982-1.008) on the rest.
#33 profiles on the real GPU: two engine starts back to back (12.7 s, 13.2 s), profiled single_stream (batch 1,
graph bucket 1) 15.8 vs 19.3 ms general, per-profile passive data kept. Lessons: an idle ssh session to the VM dies
after ~1 h (a watcher missed a finished run; an hour of GPU idle), `lab.vm setup` needs `LAB_TARGET`, Verda deletes
the disk on stop (use Nebius). Engine tag `loop-baseline` = the engine before all of this.

## 2026-10-14/15 — Layer-1 kernel loop (Nebius H200, branch `feat/kernel-loop`)
`kernels/evolve` on `paged_decode`: capture (steady_interactive, 64 requests, graphs off: 27,216 calls, all contiguous;
batch 1-9, contexts 2.7k-3.4k) → 196 seen / 119 held-out shapes. The engine's own kernel passes every gate against the
reference (588 seen + 357 held-out cases) and runs at 1.7-4.8% of the roofline (launch/latency bound). OpenEvolve,
4 iterations (Haiku 4.5 + Opus 5.5, controller on the laptop, candidates judged on the VM over SSH): best = a GQA-packed
split-K decode kernel (one program per KV head serving its 8 query heads, each K/V tile loaded once, `tl.dot`), 3.88x
call-weighted over all 196 seen shapes (2.7-4.3x, up to 19% of roofline), passing every gate incl. held-out. One of
four iterations failed stage 1 and was dropped. End to end (2026-10-15, short tier, one run each): single_stream TPOT
p99 21.6 -> 15.4 ms, saturated goodput 0.100 -> 0.158 req/s; engine PR inference-server #106 (draft; equiv not run).
Found on the way: the long_context capture crawls (~5 calls/s) because the engine's radix prefix cache evicts by
scanning the whole tree per leaf, once per block needed (`paged_kv_cache._evict_lru_leaf` via `reclaim`: ~100% of
scheduler time in long-context prefill, py-spy): a layer-3 bug in the engine. long_context's seen split holds ~30
requests, calls inside a CUDA graph recording cannot be read, Claude rejects OpenEvolve's temperature
(`kernels/evolve/llm.py`).

## 2026-10-13 — Ceilings checked (Nebius H200, driver 580.173)
Peaks 4.14 TB/s, 794 TFLOP/s bf16: every floor below every measurement once goodput charges only the prefill the
server reports as uncached (vLLM's prefix cache made a prefill-inclusive bound read 147-471%). Plain vLLM:
single_stream TPOT p99 4.1 ms (37% of ceiling), saturated 5.8 req/s (19%), long prompt 68 req/s (94%); the engine:
21 ms (7%), 0.097 req/s (0.5%), long prompt invalid (111 requests without a response). Stored in
`knowledge/reference/inference-server/`. Finding for the corpus: long_prompt_short_output's window repeats each
prompt ~5x (23,185 requests over 4,794 prompts), which rewards prefix caching.

## 2026-10-13 — `lab.gpucheck` (Nebius H200, three runs, fixes between them)
Passed every item: jail GPU, quiet, exclusivity, hostprof; workbench (after blocking only 169.254.169.254: the /16 took
DNS with it); the dependency build (after allowing pypi.nvidia.com); ncu (after linking the real /opt/nvidia install).
The jail's user can read GPU performance counters on that image.

## 2026-10-10 — First agent dry run (Nebius H200)
`LAB_WORKER=ssh`, laptop controller, `claude-fable-5-1`, one session ($2.53 as the proxy priced it, 24 turns, 465 s;
~1 h 20 min of GPU for the whole run). Validated on the real GPU: the GPU in the network-less jail, metadata blocked,
base and candidate served jailed through the port bridge, `profile`, `trace` (after mounting nsys's install root),
`test` (after caching the tests' gpt2 tokenizer), the remote workbench over `ssh -R`, the proxy's pricing. The agent
fused RMSNorm/RoPE in Triton and added a batch-1 decode graph: seen-split short-tier `single_stream` tpot p99 21.4 ->
13.1 ms; not submitted, correctness not run (equiv needs `data/equiv/ref`, absent in a worktree). A base dev-tier equiv
on this engine took > 46 min for 774 of 1,597 items at concurrency 16 (vLLM: ~10 min): the tool descriptions' "10 min
at tier dev" is wrong for this engine. Workbench processes cannot change clocks or power limits on that platform.

## 2026-10-08 — Profiler tools on the H200 (Nebius)
Unjailed, all four work on the real engine (16 requests × 64 tokens): `profile` ~7 min (first call: weights loading from disk), `trace` 44 s, `kernel` 90-99 s,
`hostprof` 64 s (30 s sampling, 2,981 samples ≈ 100 Hz × 30 s, scheduler thread in the dump); ~$0.87 of GPU time.
`launch_count` honoured; ncu durations are ~1.8× nsys's on a tiny kernel (base clocks), DRAM figures physical.
Fixed from the run: the tools now pass the target's engine env (they had loaded the engine's default model);
nsys traces CUDA-graph nodes (it had missed every decode kernel: 46,720 vs torch's 860,365 launches);
`kernels.json` drops record_function annotations (they had doubled its total). srt always mounts a fresh minimal
`/dev` and has no device passthrough, so nothing in an srt jail sees the GPU (`No CUDA GPUs are available`), and
each srt jail on Linux has its own network namespace, so a jailed server is unreachable (→ DECISIONS 2026-10-08).
The agent's CLI still runs under srt, which on Ubuntu 24.04 needs `kernel.apparmor_restrict_unprivileged_userns=0`
(in `vm-setup.sh`).

## 2026-10-05 — Full-tier recalibration (Nebius H200)
Policy confirmed, nothing tuned. 18,606 items, vLLM 0.30.0, concurrency 128, 3 h 40 min, ~$16.50. Local only:
`data/h200-run-20261005/REPORT.md`, `data/equiv-full/`. Pooled score ratio: plain BF16 1.0000, FP8 (DeepGEMM off)
1.0009, INT8 `nytopop/Qwen3-30B-A3B.w8a8` 0.9993, FP8 (DeepGEMM on, nvcc 13.0) 1.0034 — all PASS; GPTQ-Int4 0.9869
FAIL (MMLU-Pro 0.9807, p = 3e-5). FP8/INT8 each lose and gain ~800 answers in balance: that churn is the realistic
noise. Consistency clean on every run.
Open: plain BF16 matched the batch-invariant reference exactly here (18,602/18,606 identical, KL 0.0), but
on Verda at concurrency 32 it diverged on 466/490 MMLU-Pro answers (median 13% into the answer). Logs show both BF16
servers plain (non-invariant compile path, requests served by the BF16 server); the reference reproduced across both
days and clouds (691/691 same-prompt answers identical). Differences: venue, driver 580.126 vs 580.173, concurrency
32 vs 128. Don't treat a BF16 rerun as a noise sample until resolved.

## 2026-10-04 — Calibration, dev tier (Verda H200)
Ran via inference-server's `lab/vm.py`: vLLM 0.30.0, torch 2.13 + CUDA 13.0, clocks locked 1980 MHz, 83 min, $6.75.
Report and outputs are local only: `data/h200-run-20261004/REPORT.md`, `data/equiv/` (ref, candidates, `thresholds.json`).
Dev tier. Candidates: plain BF16 (no batch-invariant mode), official FP8 (served with `VLLM_USE_DEEP_GEMM=0`: the
image's nvcc 12.8 is too old for DeepGEMM), official GPTQ-Int4. KL: BF16 3.7e-7, FP8 3.0e-3, Int4 1.4e-2 (clean
separation). Per task, accuracy can't separate at this size: plain BF16 alone flips 8.2% of MMLU-Pro answers and
loses 1.9% there; no task reaches significance. Pooled score ratio (the adopted MLPerf rule): BF16 1.0008, FP8
1.0072, Int4 0.9847 (fails 0.99; McNemar p 0.049). The KL/flip-rate thresholds derived that day were superseded by
the MLPerf policy.
INT8 (`JunHowie/Qwen3-30B-A3B-GPTQ-Int8`) was skipped: vLLM 0.30 can't load act-order GPTQ MoE experts. Every
consistency result is void: the client counted vLLM's role chunk as a fake first token (fixed on
`fix/calibration-findings`). Repeat audit passes on the honest server and fails the `cache` canary.
Same day, on an H200 (whether this run's VM is not recorded): the kernel checker and timer, all tests pass; the timer
resolves ~20 µs kernels at ~9% IQR, 0.2 ms at 1.5%.
