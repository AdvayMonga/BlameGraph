"""MLPerf Inference DeepSeek-R1: 4,388 reasoning prompts from MMLU-Pro (2,410), AIME 1983-2024 (932), MATH-500 (499),
LiveCodeBench code_generation_lite (349) and GPQA (198); inputs <= 3,136 tokens, output up to 20,000 (long thinking).
Server limits TTFT 2000 ms / TPOT 80 ms; interactive 1500 ms / 15 ms (p99). Accuracy there: exact match / pass rate
>= 99% of the FP8 reference.
"""
from __future__ import annotations

from . import MLPERF_STORAGE, download, record

NAME = "mlperf_deepseek_r1"
SOURCE = f"{MLPERF_STORAGE}/deepseek_r1/datasets/mlperf_deepseek_r1_dataset_4388_fp8_eval.pkl"
MD5 = "994a30e4787831b8c8f6e44da7de67fe"
MAX_TOKENS = 20_000
LIMITS = {"server": {"ttft_s": 2.0, "tpot_s": 0.080}, "interactive": {"ttft_s": 1.5, "tpot_s": 0.015}}
TOKENIZER = "deepseek-ai/DeepSeek-R1"


def records() -> list[dict]:
    import pandas as pd
    d = pd.read_pickle(download(SOURCE, MD5))
    return [record(f"{NAME}-{r.dataset}-{k}", MAX_TOKENS, r.tok_input_len, r.dataset, prompt=r.text_input)
            for k, r in enumerate(d.itertuples())]
