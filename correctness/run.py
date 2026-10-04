"""Run the correctness gate against live servers.

  reference   once per model/hardware: the unmodified model (vLLM, BF16, VLLM_BATCH_INVARIANT=1) answers every task
              item and scores its own outputs token by token. Saved to a directory.
  candidate   per change: the optimized server answers the same items and scores the reference's tokens; the gate
              compares. Writes the verdict and an inference-server lab-ledger `equiv` record.
  calibrate   thresholds from candidate results of known-good (e.g. FP8, INT8) and known-bad (deliberately degraded)
              quantizations.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Protocol

from . import client
from .checks import consistency, length_ratio
from .divergence import Position, compare_positions, divergence_summary
from .flips import flip_test
from .gate import Thresholds, calibrate, evaluate, to_ledger_record


class Encoder(Protocol):
    def chat_ids(self, messages: list[dict]) -> list[int]: ...
    def encode(self, text: str) -> list[int]: ...


class HFEncoder:
    """Reference tokenizer with the chat template applied, thinking off."""
    def __init__(self, name: str = "Qwen/Qwen3-30B-A3B"):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name)

    def chat_ids(self, messages):
        out = self.tok.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False, tokenize=True)
        return list(out["input_ids"] if hasattr(out, "keys") else out)

    def encode(self, text):
        return self.tok.encode(text, add_special_tokens=False)


def _jsonl_write(path: Path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _jsonl_read(path: Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()] if Path(path).exists() else []


def collect(url: str, model: str, items: list[dict], out: Path, concurrency: int = 16) -> dict[str, dict]:
    """Greedy answers for every item; resumable (items already in `out` are skipped)."""
    done = {r["id"]: r for r in _jsonl_read(out)}
    todo = [it for it in items if it["id"] not in done]

    def one(it):
        try:
            g = client.generate(url, model, it["messages"], it.get("max_tokens", 2048))
        except Exception as e:                      # a failed request is a recorded fact, not a crash
            g = {"text": "", "completion_tokens": None, "finish_reason": None, "error": str(e)[:200]}
        return {"id": it["id"], **g}

    with open(out, "a") as f, ThreadPoolExecutor(concurrency) as ex:
        for r in ex.map(one, todo):
            f.write(json.dumps(r) + "\n"); f.flush(); done[r["id"]] = r
    return done


def _positions(rows) -> list[Position]:
    return [Position(token=p[0], logprob=p[1], top={int(k): v for k, v in p[2].items()}) for p in rows]


def reference(url: str, model: str, out_dir: str | Path, items: list[dict], encoder: Encoder, top_k: int = 20,
              div_n: int = 400, concurrency: int = 16) -> dict:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    _jsonl_write(out / "items.jsonl", items)
    outputs = collect(url, model, items, out / "outputs.jsonl", concurrency)
    # divergence set: prompt + the reference's own answer, for a spread of items across tasks
    pick = [it for k, it in enumerate(items) if k % max(1, len(items) // div_n) == 0][:div_n]
    seqs = []
    for it in pick:
        text = outputs[it["id"]].get("text") or ""
        if text:
            p = encoder.chat_ids(it["messages"])
            seqs.append({"id": it["id"], "ids": p + encoder.encode(text), "start": len(p)})

    def score(s):
        pos = client.score_tokens(url, model, s["ids"], s["start"], top_k)
        return {**s, "ref": [[q.token, q.logprob, {str(k): v for k, v in q.top.items()}] for q in pos]}

    with ThreadPoolExecutor(concurrency) as ex:
        _jsonl_write(out / "sequences.jsonl", list(ex.map(score, seqs)))
    meta = {"model": model, "url": url, "top_k": top_k, "n_items": len(items), "n_sequences": len(seqs)}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def candidate(ref_dir: str | Path, url: str, model: str, out_path: str | Path, encoder: Encoder,
              thresholds: Thresholds | None = None, probe_n: int = 20, concurrency: int = 16,
              score_item: Callable | None = None, config: dict | None = None) -> dict:
    from .tasks import score_item as default_score
    score_item = score_item or default_score
    ref = Path(ref_dir)
    items = _jsonl_read(ref / "items.jsonl")
    ref_out = {r["id"]: r for r in _jsonl_read(ref / "outputs.jsonl")}
    meta = json.loads((ref / "meta.json").read_text())
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    cand_out = collect(url, model, items, out_path.with_suffix(".outputs.jsonl"), concurrency)
    # flips, per task
    by_task: dict[str, tuple[list, list]] = {}
    for it in items:
        r, c = ref_out.get(it["id"]), cand_out.get(it["id"])
        if r is None or c is None:
            continue
        a, b = by_task.setdefault(it["task"], ([], []))
        a.append(score_item(it, r.get("text") or "")); b.append(score_item(it, c.get("text") or ""))
    flips = {t: flip_test(a, b) for t, (a, b) in by_task.items()}
    # length, counted with the reference tokenizer on both sides
    ids = [it["id"] for it in items if it["id"] in ref_out and it["id"] in cand_out]
    length = length_ratio([len(encoder.encode(ref_out[i].get("text") or "")) for i in ids],
                          [len(encoder.encode(cand_out[i].get("text") or "")) for i in ids])
    # divergence on the reference's own tokens
    seqs = _jsonl_read(ref / "sequences.jsonl")

    def comp(s):
        return compare_positions(_positions(s["ref"]), client.score_tokens(url, model, s["ids"], s["start"], meta["top_k"]))

    with ThreadPoolExecutor(concurrency) as ex:
        div = divergence_summary(list(ex.map(comp, seqs)))
    # consistency, on a streamed probe of the first items
    probe = []
    for it in items[:probe_n]:
        s = client.generate_stream(url, model, it["messages"], it.get("max_tokens", 2048))
        probe.append({**s, "reported_tokens": cand_out.get(it["id"], {}).get("completion_tokens")
                      if (cand_out.get(it["id"], {}).get("text") == s["text"]) else None})
    cons = consistency(probe, encoder.encode)
    result = evaluate(div, flips, length, cons, thresholds)
    record = to_ledger_record(result, {"split": "seen", "model": model, **(config or {})}, base=meta["model"])
    out_path.write_text(json.dumps({"result": result, "ledger_record": record}, indent=1, default=str))
    return result


def verdict_file(result_path: str | Path, thresholds: Thresholds) -> dict:
    """Re-judge a saved candidate result under new thresholds, from its recorded metrics (no server needed)."""
    m = json.loads(Path(result_path).read_text())["result"]["metrics"]
    return evaluate(m.get("divergence"), {k[6:]: v for k, v in m.items() if k.startswith("flips:")},
                    m.get("length"), m.get("consistency"), thresholds)


def calibrate_files(good: list[str], bad: list[str], out: str) -> Thresholds:
    load = lambda p: json.loads(Path(p).read_text())["result"]["metrics"]
    def item(m):
        return {"divergence": m.get("divergence"), "flips": {k[6:]: v for k, v in m.items() if k.startswith("flips:")}}
    th = calibrate([item(load(p)) for p in good], [item(load(p)) for p in bad],
                   label=f"good={[Path(p).stem for p in good]} bad={[Path(p).stem for p in bad]}")
    Path(out).write_text(json.dumps(asdict(th), indent=1))
    return th
