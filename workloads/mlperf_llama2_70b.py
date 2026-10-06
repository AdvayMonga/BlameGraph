"""MLPerf Inference Llama-2-70B: OpenOrca GPT-4 subset, 24,576 instruction prompts sampled equally from its
CoT / NIV / FLAN / T0 origins, English only, input and reference output < 1024 tokens each. Output up to 1024 tokens.
Server limits TTFT 2000 ms / TPOT 200 ms; interactive 450 ms / 40 ms (p99). Accuracy there: ROUGE >= 99% of FP16.
"""
from __future__ import annotations

from . import MLPERF_STORAGE, download, read_gz_or_plain, record

NAME = "mlperf_llama2_70b"
SOURCE = f"{MLPERF_STORAGE}/open_orca/open_orca_gpt4_tokenized_llama.sampled_24576.pkl.gz"
MD5 = "c247a315b9328365f66d2c2585b35cd4"
MAX_TOKENS = 1024
LIMITS = {"server": {"ttft_s": 2.0, "tpot_s": 0.200}, "interactive": {"ttft_s": 0.45, "tpot_s": 0.040}}
TOKENIZER = "meta-llama/Llama-2-70b-chat-hf"


def records() -> list[dict]:
    import pandas as pd
    with read_gz_or_plain(download(SOURCE, MD5)) as f:
        d = pd.read_pickle(f)
    out = []
    for r in d.itertuples():
        msgs = ([{"role": "system", "content": r.system_prompt}] if r.system_prompt else []) + \
               [{"role": "user", "content": r.question}]
        out.append(record(f"{NAME}-{r.id}", MAX_TOKENS, r.tok_input_length, r.origin, messages=msgs))
    return out
