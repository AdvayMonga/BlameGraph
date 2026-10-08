"""Run the correctness gate against live servers.

  reference   once per model/hardware: the unmodified model (vLLM, BF16, VLLM_BATCH_INVARIANT=1) answers every task
              item and scores its own outputs token by token. Saved to a directory.
  candidate   per change: the optimized server answers the same items and scores the reference's tokens; the gate
              compares. Writes the verdict and an inference-server lab-ledger `equiv` record.
  verdict     re-judge saved candidate results under a policy, without a server.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Protocol

from . import client
from .checks import consistency, length_ratio
from .divergence import Position, compare_positions, divergence_summary
from .flips import flip_test
from .gate import Thresholds, evaluate, to_ledger_record


class Encoder(Protocol):
    def chat_ids(self, messages: list[dict]) -> list[int]: ...
    def encode(self, text: str) -> list[int]: ...


class HFEncoder:
    """Reference tokenizer with the chat template applied under the target's chat kwargs."""
    def __init__(self, name: str, chat_kwargs: dict | None = None):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name)
        self.chat_kwargs = dict(chat_kwargs or {})

    def chat_ids(self, messages):
        out = self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, **self.chat_kwargs)
        return list(out["input_ids"] if hasattr(out, "keys") else out)

    def encode(self, text):
        return self.tok.encode(text, add_special_tokens=False)


def _jsonl_write(path: Path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _jsonl_read(path: Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()] if Path(path).exists() else []


TRACE_PREFIX = "eq"          # X-Trace-Id of a gate request: eq-<item id>, so equiv rows join engine telemetry
RETRIES = 6                 # a 429/503 (the server shedding under the gate's load) is retried with backoff


def _retrying(fn, *args, **kw):
    """fn(*args) with exponential backoff on HTTP 429/503: being shed is not a wrong answer. Other errors propagate."""
    import time
    import urllib.error
    for attempt in range(RETRIES):
        try:
            return fn(*args, **kw)
        except urllib.error.HTTPError as e:
            if e.code not in (429, 503) or attempt == RETRIES - 1:
                raise
            time.sleep(min(60.0, 2.0 ** attempt))


def collect(url: str, model: str, items: list[dict], out: Path, concurrency: int = 16,
            api: client.Api = client.DEFAULT) -> dict[str, dict]:
    """Greedy answers for every item; resumable (answered items in `out` are skipped, errored ones retried).
    An item that still fails after retries is recorded with `error`: it is unanswered, never a wrong answer."""
    done = {r["id"]: r for r in _jsonl_read(out) if not r.get("error")}
    todo = [it for it in items if it["id"] not in done]

    def one(it):
        try:
            g = _retrying(client.generate, url, model, it["messages"], it.get("max_tokens", 2048), api,
                          trace_id=f"{TRACE_PREFIX}-{it['id']}")
        except Exception as e:                      # a failed request is a recorded fact, not a crash
            g = {"text": "", "completion_tokens": None, "finish_reason": None, "error": f"{type(e).__name__}: {e}"[:200]}
        return {"id": it["id"], **g}

    with open(out, "a") as f, ThreadPoolExecutor(concurrency) as ex:
        for r in ex.map(one, todo):
            f.write(json.dumps(r) + "\n"); f.flush(); done[r["id"]] = r
    return done


def _positions(rows) -> list[Position]:
    return [Position(token=p[0], logprob=p[1], top={int(k): v for k, v in p[2].items()}) for p in rows]


def reference(url: str, model: str, out_dir: str | Path, items: list[dict], encoder: Encoder, top_k: int = 20,
              div_n: int = 400, concurrency: int = 16, api: client.Api = client.DEFAULT) -> dict:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    _jsonl_write(out / "items.jsonl", items)
    outputs = collect(url, model, items, out / "outputs.jsonl", concurrency, api)
    unanswered = sorted(i for i, r in outputs.items() if r.get("error"))
    if unanswered:
        raise RuntimeError(f"the reference left {len(unanswered)} item(s) unanswered (e.g. {unanswered[0]}: "
                           f"{outputs[unanswered[0]]['error']}); a reference must answer everything")
    # divergence set: prompt + the reference's own answer, for a spread of items across tasks
    pick = [it for k, it in enumerate(items) if k % max(1, len(items) // div_n) == 0][:div_n]
    seqs = []
    for it in pick:
        text = outputs[it["id"]].get("text") or ""
        if text:
            p = encoder.chat_ids(it["messages"])
            seqs.append({"id": it["id"], "ids": p + encoder.encode(text), "start": len(p)})

    def score(s):
        pos = _retrying(client.score_tokens, url, model, s["ids"], s["start"], top_k, api)
        return {**s, "ref": [[q.token, q.logprob, {str(k): v for k, v in q.top.items()}] for q in pos]}

    with ThreadPoolExecutor(_score_concurrency(concurrency)) as ex:
        _jsonl_write(out / "sequences.jsonl", list(ex.map(score, seqs)))
    meta = {"model": model, "url": url, "top_k": top_k, "n_items": len(items), "n_sequences": len(seqs)}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def _score_concurrency(concurrency: int) -> int:
    """Teacher-forced scoring holds a [prompt, vocab] logits tensor per request: a quarter of the generation
    concurrency, so a 32k prompt x 32 in flight does not OOM an engine that generation alone would not."""
    return max(1, concurrency // 4)


def candidate(ref_dir: str | Path, url: str, model: str, out_path: str | Path, encoder: Encoder,
              thresholds: Thresholds | None = None, probe_n: int = 20, concurrency: int = 16,
              score_item: Callable | None = None, config: dict | None = None,
              api: client.Api = client.DEFAULT) -> dict:
    from .tasks import score_item as default_score
    score_item = score_item or default_score
    ref = Path(ref_dir)
    items = _jsonl_read(ref / "items.jsonl")
    ref_out = {r["id"]: r for r in _jsonl_read(ref / "outputs.jsonl")}
    meta = json.loads((ref / "meta.json").read_text())
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    cand_out = collect(url, model, items, out_path.with_suffix(".outputs.jsonl"), concurrency, api)
    answered = {i: r for i, r in cand_out.items() if not r.get("error")}
    unanswered = {i: r["error"] for i, r in cand_out.items() if r.get("error")}
    # flips, per task, over the items the candidate answered: an unanswered item is neither right nor wrong
    by_task: dict[str, tuple[list, list]] = {}
    for it in items:
        r, c = ref_out.get(it["id"]), answered.get(it["id"])
        if r is None or c is None:
            continue
        a, b = by_task.setdefault(it["task"], ([], []))
        a.append(score_item(it, r.get("text") or "")); b.append(score_item(it, c.get("text") or ""))
    flips = {t: flip_test(a, b) for t, (a, b) in by_task.items()}
    # length, counted with the reference tokenizer on both sides
    ids = [it["id"] for it in items if it["id"] in ref_out and it["id"] in answered]
    length = length_ratio([len(encoder.encode(ref_out[i].get("text") or "")) for i in ids],
                          [len(encoder.encode(answered[i].get("text") or "")) for i in ids])
    # divergence on the reference's own tokens (a fact, not a gate); skipped when the engine exposes no logprobs
    seqs = _jsonl_read(ref / "sequences.jsonl")
    unscored: list[str] = []

    def comp(s):
        try:
            return compare_positions(_positions(s["ref"]),
                                     _retrying(client.score_tokens, url, model, s["ids"], s["start"], meta["top_k"], api))
        except client.NoLogprobs:
            raise
        except Exception as e:
            unscored.append(f"{s['id']}: {type(e).__name__}: {e}"[:200])
            return None

    try:
        with ThreadPoolExecutor(_score_concurrency(concurrency)) as ex:
            scored = [c for c in ex.map(comp, seqs) if c is not None]
        div = divergence_summary(scored) if scored else None
        if div is not None:
            div["sequences_unscored"] = len(unscored)
    except client.NoLogprobs as e:
        div, unscored = None, [str(e)]
    # consistency, on a streamed probe of the first items; a shed or errored probe is counted, not fatal
    probe, probe_errors = [], []
    for it in items[:probe_n]:
        try:
            s = _retrying(client.generate_stream, url, model, it["messages"], it.get("max_tokens", 2048), api)
        except Exception as e:
            probe_errors.append(f"{it['id']}: {type(e).__name__}: {e}"[:200])
            continue
        probe.append({**s, "reported_tokens": answered.get(it["id"], {}).get("completion_tokens")
                      if (answered.get(it["id"], {}).get("text") == s["text"]) else None})
    cons = consistency(probe, encoder.encode) if probe else None
    result = evaluate(div, flips, length, cons, thresholds, unanswered=len(unanswered))
    if probe_errors:
        result["metrics"]["consistency_probe_errors"] = probe_errors[:5]
    result["metrics"]["unanswered"] = {"n": len(unanswered), "of": len(items),
                                       "examples": dict(list(unanswered.items())[:5])}
    if unscored:
        result["metrics"]["divergence_unscored"] = unscored[:5]
    record = to_ledger_record(result, {"split": "seen", "model": model, **(config or {})}, base=meta["model"])
    out_path.write_text(json.dumps({"result": result, "ledger_record": record}, indent=1, default=str))
    return result


def verdict_file(result_path: str | Path, thresholds: Thresholds) -> dict:
    """Re-judge a saved candidate result under new thresholds, from its recorded metrics (no server needed)."""
    m = json.loads(Path(result_path).read_text())["result"]["metrics"]
    return evaluate(m.get("divergence"), {k[6:]: v for k, v in m.items() if k.startswith("flips:")},
                    m.get("length"), m.get("consistency"), thresholds)
