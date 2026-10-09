"""Frontier benchmark workloads: the request data public inference benchmarks are scored on, one module per benchmark,
fetched from their public sources into data/workloads/ (gitignored) as trace-shaped JSONL the regimes consume.

Record: {"id", "prompt" | "messages", "max_tokens", "build_prompt_tokens", "subset"}. Prompts are the benchmark's raw
text with its model-specific chat template stripped, so any chat server can serve them; `build_prompt_tokens` is the
source benchmark's own tokenizer count unless re-counted with `--tokenizer`. The manifest records source URL, checksum,
item count, output limit and the benchmark's latency limits, so comparisons can refuse mismatched corpora.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import time
import urllib.request
from pathlib import Path
from types import ModuleType

DATA = Path(__file__).resolve().parents[1] / "data" / "workloads"
MLPERF_STORAGE = "https://inference.mlcommons-storage.org"
MLPERF_COMMIT = "3fbc329939999c13d0a7b5e67fb2092287e06047"       # mlcommons/inference master, 2026-09-02
BENCHES = ("mlperf_llama3_1_8b", "mlperf_llama2_70b", "mlperf_mixtral_8x7b", "mlperf_deepseek_r1",
           "mlperf_gpt_oss_120b", "sharegpt")


def download(url: str, md5: str | None = None) -> Path:
    """Fetch `url` into data/workloads/raw/ once; verify md5 when the source publishes one. Returns the local path."""
    dest = DATA / "raw" / url.rsplit("/", 1)[1]
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        tmp.rename(dest)
    if md5 and _digest(dest, "md5") != md5:
        raise ValueError(f"{dest.name}: md5 {_digest(dest, 'md5')} != published {md5}")
    return dest


def read_gz_or_plain(path: Path):
    return gzip.open(path, "rb") if path.suffix == ".gz" else open(path, "rb")


def record(id: str, max_tokens: int, tokens: int | None, subset: str, prompt: str | None = None,
           messages: list[dict] | None = None) -> dict:
    r = {"id": id, "max_tokens": max_tokens, "subset": subset}
    if messages is not None:
        r["messages"] = messages
    else:
        r["prompt"] = prompt
    if tokens is not None:
        r["build_prompt_tokens"] = int(tokens)
    return r


def fetch(bench: ModuleType, tokenizer: str | None = None) -> Path:
    """Build a benchmark's records, write data/workloads/<name>.jsonl and its manifest entry."""
    recs = bench.records()
    if tokenizer:
        _recount(recs, tokenizer)
    DATA.mkdir(parents=True, exist_ok=True)
    out = DATA / f"{bench.NAME}.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs))
    man_path = DATA / "manifest.json"
    man = json.loads(man_path.read_text()) if man_path.exists() else {}
    man[bench.NAME] = {"file": out.name, "n": len(recs), "sha256": _digest(out, "sha256"), "source": bench.SOURCE,
                       "source_md5": getattr(bench, "MD5", None), "max_tokens": bench.MAX_TOKENS,
                       "limits": bench.LIMITS, "tokenizer": tokenizer or bench.TOKENIZER,
                       "mlperf_commit": MLPERF_COMMIT if bench.NAME.startswith("mlperf") else None,
                       "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    man_path.write_text(json.dumps(man, indent=1))
    return out


def _recount(recs: list[dict], tokenizer: str):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer)
    for r in recs:
        msgs = r.get("messages") or [{"role": "user", "content": r["prompt"]}]
        r["build_prompt_tokens"] = len(tok.apply_chat_template(msgs, add_generation_prompt=True))


def _digest(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()
