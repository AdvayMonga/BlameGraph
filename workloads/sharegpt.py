"""ShareGPT (Vicuna unfiltered, V3 cleaned split): the ~90k real user conversations behind most engine benchmark
numbers (vLLM / SGLang `bench_serving`, InferenceMAX-style sweeps). Not an MLPerf benchmark: no latency limits or
accuracy target. Conversations that start with the user and alternate cleanly are kept whole, so they also serve the
shared-prefix multi-turn regime; single-exchange ones become plain prompts. Output up to 1024 tokens (`bench_serving`
instead replays each reply's length). Token counts need `--tokenizer`; the file carries none.
"""
from __future__ import annotations

import json

from . import DATA, record

NAME = "sharegpt"
REPO, FILE = "anon8231489123/ShareGPT_Vicuna_unfiltered", "ShareGPT_V3_unfiltered_cleaned_split.json"
SOURCE = f"https://huggingface.co/datasets/{REPO}/resolve/main/{FILE}"
MAX_TOKENS = 1024
LIMITS = {}
TOKENIZER = None
ROLE = {"human": "user", "gpt": "assistant"}


def records() -> list[dict]:
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(REPO, FILE, repo_type="dataset", local_dir=str(DATA / "raw" / "sharegpt"))
    out = []
    for conv in json.loads(open(path, encoding="utf-8").read()):
        msgs = [{"role": ROLE.get(t.get("from"), "other"), "content": t.get("value") or ""} for t in conv.get("conversations", [])]
        if not msgs or msgs[0]["role"] != "user" or any(m["role"] == "other" for m in msgs) \
                or any(a["role"] == b["role"] for a, b in zip(msgs, msgs[1:])) or not all(m["content"].strip() for m in msgs):
            continue
        n_user = sum(m["role"] == "user" for m in msgs)
        if n_user == 1:
            out.append(record(f"{NAME}-{conv['id']}", MAX_TOKENS, None, "single", prompt=msgs[0]["content"]))
        else:
            if msgs[-1]["role"] == "assistant":
                msgs = msgs[:-1]
            out.append(record(f"{NAME}-{conv['id']}", MAX_TOKENS, None, "multi", messages=msgs))
    return out
