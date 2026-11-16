"""MLPerf Inference gpt-oss-120b (v6.0): `perf:pubmed_summarization` and `acc:<dataset>` sets, told apart by `subset`.
The harmony-formatted prompt is split back into system (developer) and user messages.
"""
from __future__ import annotations

import re

from . import MLPERF_STORAGE, download, record

NAME = "mlperf_gpt_oss_120b"
SOURCE = f"{MLPERF_STORAGE}/gpt-oss_data/perf/perf_eval_ref.parquet"
ACC_SOURCE = f"{MLPERF_STORAGE}/gpt-oss_data/acc/acc_eval_ref.parquet"
MD5 = "e4cd6cef6dd975f3e50c85b3279b358b"
ACC_MD5 = "29d35424fc8d31461e73a7766446480e"
MAX_TOKENS = 10_240
ACC_MAX_TOKENS = 32_768
LIMITS = {}
TOKENIZER = "openai/gpt-oss-120b"
HARMONY = re.compile(r"<\|start\|>(\w+)<\|message\|>(.*?)<\|end\|>", re.S)
ROLE = {"developer": "system", "user": "user"}


def records() -> list[dict]:
    import pandas as pd
    out = []
    for k, r in enumerate(pd.read_parquet(download(SOURCE, MD5)).itertuples()):
        msgs = [{"role": ROLE[role], "content": text} for role, text in HARMONY.findall(r.text_input) if role in ROLE]
        out.append(record(f"{NAME}-perf-{k}", MAX_TOKENS, r.num_tokens, f"perf:{r.dataset}", messages=msgs))
    for k, r in enumerate(pd.read_parquet(download(ACC_SOURCE, ACC_MD5)).itertuples()):
        msgs = [{"role": m["role"], "content": m["content"]} for m in r.original_messages]
        out.append(record(f"{NAME}-acc-{k}", ACC_MAX_TOKENS, r.num_tokens, f"acc:{r.dataset}", messages=msgs))
    return out
