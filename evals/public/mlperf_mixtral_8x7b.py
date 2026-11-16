"""MLPerf Inference Mixtral-8x7B: 15,000 prompts, 5k each from OpenOrca, GSM8K and MBXP; output up to 1024 tokens.
Server limits TTFT 2000 ms / TPOT 200 ms (p99). MBXP's assistant prefill and stop sequence are dropped.
"""
from __future__ import annotations

import re

from . import MLPERF_STORAGE, download, record

NAME = "mlperf_mixtral_8x7b"
SOURCE = f"{MLPERF_STORAGE}/mixtral_8x7b/09292024_mixtral_15k_mintoken2_v1.pkl"
MD5 = "ded6c711288c9bbca02929855557b8c1"
MAX_TOKENS = 1024
LIMITS = {"server": {"ttft_s": 2.0, "tpot_s": 0.200}}
TOKENIZER = "mistralai/Mixtral-8x7B-Instruct-v0.1"
TURN = re.compile(r"\[INST\]\s*(.*?)\s*\[/INST\]\s*(.*?)\s*(?=\[INST\]|$)", re.S)   # (user, assistant) pairs


def records() -> list[dict]:
    import pandas as pd
    d = pd.read_pickle(download(SOURCE, MD5))
    out = []
    for r in d.itertuples():
        turns = TURN.findall(r.input)
        rid = f"{NAME}-{r.dataset}-{r.id}"
        if len(turns) == 1:
            out.append(record(rid, MAX_TOKENS, r.tok_input_len, r.dataset, prompt=turns[0][0]))
        else:
            msgs = [m for u, a in turns for m in ({"role": "user", "content": u}, {"role": "assistant", "content": a})]
            out.append(record(rid, MAX_TOKENS, r.tok_input_len, r.dataset, messages=msgs[:-1]))
    return out
