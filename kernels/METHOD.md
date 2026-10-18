# The layer-1 loop's method, component by component

Rule (the user, 2026-10-17): every part copies the published method with the strongest numbers, or is our own
design with the reason it should do better. "Ours" rows are A/B arms or noted as unmeasured. Sources were read in full
unless marked [abs]. Compiled 2026-10-18.

| # | Component | Published (number) | Ours | Where |
|---|---|---|---|---|
| 1 | Which kernel | Amdahl ranking: GEAK, share of GPU time x achievable speedup [abs]; Ansor's task scheduler beats round-robin (figure) | Copied: share x (1 - % of roofline) per task, across profiles (`ceiling_gain`) | `kernels/share.py` |
| 2 | Diagnosis facts | Curated counters: CudaForge 24 metrics 1.77x vs all 1.41x; KernelPro tools 4.00x vs raw ncu 1.77x vs none 3.35x; MANTIS SOL report | Copied: CudaForge's 24 on the smallest and largest seen shape; per shape, bound type and % of each peak; facts only | `kernels/evolve/ncu.py`, `core._roofline` |
| 3 | Hypotheses + triage | MANTIS: nominate several, rank by ROI (most damaging ablation for weaker models); KernelBand bandit 1.57x vs model's choice 0.97x | Copied: 16 per expansion with the judge's speedup and risk estimates, MANTIS's ROI | `search/loop.py` (`hypotheses`, `roi`) |
| 4 | System-1 screener | GPU Forecasters (code, not hypotheses): propose 4x, measure the top quarter, 3/6 faster, 1 tied, 2 slower | Ours: Haiku 4.5 scores hypotheses before any code; arms `roi`, `random`; a random slot per beam for unbiased labels; a fine-tuned local model when labels allow | `search/screen.py` |
| 5 | Candidates per step | Kevin 4x16 beam 1.56x vs best@16 1.41x; METR still rising at 300 attempts | Copied: beam 4 (3 ranked + 1 random) | `--beam`, `--explore` |
| 6 | Parent selection | KernelPro MCTS (UCT C = sqrt 2, widening alpha = 0.5) 4.60x vs greedy 3.65x, p = 0.004 | Copied, with KernelPro's virtual-child expansion rule (widen only when a new child's UCT, the node's mean as prior, beats the best child's); ours: the reward is a node's percentile among every judged program (score over the start clipped at 2x saturated in the 2026-10-18 A/B: every child 1.0, so UCT only counted visits). Both 2026-10-18 A/Bs stayed at depth 2 at 30 programs; on a synthetic landscape C decides depth more than the rule (C = sqrt 2: 3, 0.5: 3, 0.2: ~5), so C is an arm, `search-c0.5` | `search/tree.py`, `--uct-c` |
| 7 | Crossover | CodeEvolve: 2-3 inspirations needed to beat AlphaEvolve (circle packing; none on kernels) | Copied (2 inspirations) + ours: a merge hypothesis competing for the beam, its expected speedup the other branch's score over this program's (over the start's it was 8x and took a slot in every beam of the 2026-10-18 A/B) | `loop.inspirations`, `merge_hypothesis` |
| 8 | Context | CudaForge stateless; GEAK retrieval +; KernelPro memory n.s.; web search never ablated; OpenEvolve shows top + diverse programs and previous attempts | Copied stateless per call; ours: the judge sees every hypothesis measured anywhere in the tree with its outcome (it re-proposed the same ~20 changes at every expansion of the 2026-10-18 A/B), and may search the web and query the ledger (its choice), arm `search-no-tools` | `loop.history`, `search/llm.py` |
| 9 | Roles | CudaForge: strong judge and strong coder (2.11x; a weak coder 0.79x) | Copied: Opus 5.5 both | `--model`, `--judge-model` |
| 10 | Repair | CudaForge correction mode 57.6% -> 97.6% correct; KernelPro: terminal after 3 failed repairs | Copied: correction mode, chained 3 deep | `--max-repairs` |
| 11 | Launch parameters | KernelEvolve: structure from the model, configs from autotune (no head-to-head published) | Ours: grid search over declared TUNABLES on each of the top 4 re-timed programs (8 points each; tuning only the winner chose the program before its parameters were known), the best point overall re-timed against the winner, written back as constants (no startup autotune); per-size-region dispatch | `search/tune.py`, `__main__._tune` |
| 12 | Timing | CUDA-L1 paired randomized runs; nothing published for graph replay | Ours: paired, randomized order, CUDA-graph replay behind a GPU spin, timed outputs checked, re-timed promotion | `kernels/timing.py` |
| 13 | Stopping | MANTIS: within 25% of SOL or 16 attempts without progress: 96% of the gain for 33% fewer tokens | Copied (80% of roofline, patience 16) | `--stop-pct`, `--patience` |
| 14 | Exploit defences | MANTIS game detector (2.85x -> 5.34x without it); KernelGuard rules 3.45% -> 0.37% | Copied detector at promotion; ours: jail, source screen, patch checks, outputs re-verified outside the candidate's process, physics alarm, canaries | `search/detect.py`, `kernels/evolve/` |
| 15 | Generalization | AlphaEvolve random half split of shapes; KernelBench-Verified hidden values | Ours: held-out whole shape regions + value scalings, gated, never in selection | `kernels/shapes.py`, `gate` |
| 16 | Language | CudaForge, KernelPro (CUDA + CuTe, 1.23x over hand-tuned Triton on MoE), MANTIS: CUDA C++ | Copied: CUDA C++ built with `load_inline`, beside Triton, the model's choice; the build cached per program source; its C++ screened like the Python | `jailed._builds`, `core.FORBIDDEN` |

Unmeasured choices, in order of what the A/B should settle first: the screener (4), judge tools (8), the merge
hypothesis (7), grid search vs model-chosen parameters (11). Arms: `python -m kernels.search.ab --arms
openevolve,kernels.search,search-roi,search-random,search-no-tools,search-c0.5`.
