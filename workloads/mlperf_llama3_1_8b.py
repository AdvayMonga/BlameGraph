"""MLPerf Inference Llama-3.1-8B: CNN/DailyMail 3.0.0 validation, 13,368 news articles to summarize in 128 tokens.
Short output, prompts up to 8k tokens. Server limits TTFT 2000 ms / TPOT 100 ms; interactive 500 ms / 30 ms (p99).
Accuracy there: ROUGE >= 99% of BF16 reference, generated tokens >= 90%.
"""
from __future__ import annotations

import json

from . import MLPERF_STORAGE, download, record

NAME = "mlperf_llama3_1_8b"
SOURCE = f"{MLPERF_STORAGE}/llama3.1_8b/datasets/cnn_eval.json"
MD5 = "c41c613c17794d117899fdcdeeadbe3c"
MAX_TOKENS = 128
LIMITS = {"server": {"ttft_s": 2.0, "tpot_s": 0.100}, "interactive": {"ttft_s": 0.5, "tpot_s": 0.030}}
TOKENIZER = "meta-llama/Llama-3.1-8B-Instruct"
TEMPLATE = ("Summarize the following news article in 128 tokens. Please output the summary only, without any other "
            "text.\n\nArticle:\n{input}\n\nSummary:")


def records() -> list[dict]:
    rows = json.loads(download(SOURCE, MD5).read_text())
    return [record(f"{NAME}-{k}", MAX_TOKENS, len(r["tok_input"]), "cnn_dailymail",
                   prompt=TEMPLATE.format(input=r["input"])) for k, r in enumerate(rows)]
