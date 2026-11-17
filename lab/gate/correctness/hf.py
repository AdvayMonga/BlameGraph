"""The reference: the model's published weights run greedy in transformers. Writes the files `candidate`
reads: items.jsonl, outputs.jsonl, sequences.jsonl, meta.json."""
from __future__ import annotations

import json
import shutil
import tempfile
import time
from pathlib import Path

from .run import HFEncoder, _jsonl_read, _jsonl_write, sequences

ROWS = 64            # most prompts in one generate call
TOKENS = 160_000     # most prompt + new tokens in one generate call


def batches(todo: list[tuple[dict, list[int]]]) -> list[list[tuple[dict, list[int]]]]:
    """(item, prompt ids) pairs grouped by max_tokens, shortest prompts first, each group within ROWS and TOKENS."""
    out: list[list] = []
    for it, ids in sorted(todo, key=lambda p: (p[0].get("max_tokens", 2048), len(p[1]))):
        new = it.get("max_tokens", 2048)
        last = out[-1] if out else None
        if (last is None or last[0][0].get("max_tokens", 2048) != new or len(last) == ROWS
                or (len(last) + 1) * (len(ids) + new) > TOKENS):
            out.append([])
        out[-1].append((it, ids))
    return out


def build(ref_dir: Path, tasks: tuple[str, ...], wk, seconds: float = 600.0) -> None:
    """The reference in `ref_dir` completed by `reference` jobs on worker `wk`, each continuing the last."""
    ref_dir.mkdir(parents=True, exist_ok=True)
    done = lambda: len(_jsonl_read(ref_dir / "outputs.jsonl"))     # noqa: E731
    while not (ref_dir / "meta.json").exists():
        before = done()
        with tempfile.TemporaryDirectory() as d:
            reply = wk.call("reference", {"prior": ref_dir}, {"tasks": list(tasks), "seconds": seconds}, Path(d))
            shutil.copytree(d, ref_dir, dirs_exist_ok=True)
        if not reply["complete"] and done() == before:
            raise RuntimeError(f"the reference job answered nothing new ({before} answers in {ref_dir})")


def reference(model: str, out_dir: str | Path, items: list[dict], chat_kwargs: dict | None = None, top_k: int = 20,
              div_n: int = 400, seconds: float | None = None) -> dict | None:
    """Greedy answers for every item (resumable), then the reference's logprobs on its own answers. With `seconds`,
    generation stops after the batch that passes it and None is returned while items are left."""
    import torch
    import transformers
    from transformers import AutoModelForCausalLM
    from transformers.utils import cached_file
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    _jsonl_write(out / "items.jsonl", items)
    enc = HFEncoder(model, chat_kwargs)
    tok = enc.tok
    lm = AutoModelForCausalLM.from_pretrained(model, torch_dtype="auto").cuda().eval()
    eos = lm.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    pad = tok.pad_token_id if tok.pad_token_id is not None else min(eos)
    outputs = {r["id"]: r for r in _jsonl_read(out / "outputs.jsonl")}
    todo = [(it, enc.chat_ids(it["messages"])) for it in items if it["id"] not in outputs]
    t0 = time.monotonic()
    with open(out / "outputs.jsonl", "a") as f, torch.inference_mode():
        for batch in batches(todo):
            if seconds is not None and time.monotonic() - t0 > seconds:
                return None
            width, new = max(len(ids) for _, ids in batch), batch[0][0].get("max_tokens", 2048)
            x = torch.full((len(batch), width), pad, dtype=torch.long)
            mask = torch.zeros_like(x)
            for k, (_, ids) in enumerate(batch):
                x[k, width - len(ids):] = torch.tensor(ids)
                mask[k, width - len(ids):] = 1
            y = lm.generate(input_ids=x.cuda(), attention_mask=mask.cuda(), max_new_tokens=new, do_sample=False,
                            temperature=None, top_p=None, top_k=None, pad_token_id=pad)
            for (it, _), row in zip(batch, y[:, width:].tolist()):
                stop = next((i for i, t in enumerate(row) if t in eos), None)
                kept = row if stop is None else row[:stop]
                r = {"id": it["id"], "text": tok.decode(kept, skip_special_tokens=True),
                     "completion_tokens": len(kept) + (stop is not None),
                     "finish_reason": "length" if stop is None else "stop"}
                f.write(json.dumps(r) + "\n"); f.flush(); outputs[r["id"]] = r
    rows = []
    with torch.inference_mode():
        for s in sequences(items, outputs, enc, div_n):
            ids, start = s["ids"], s["start"]
            logp = torch.log_softmax(lm(torch.tensor([ids]).cuda()).logits[0, start - 1:-1].float(), -1)
            top = logp.topk(top_k)
            ref = []
            for j, t in enumerate(ids[start:]):
                alts = dict(zip(top.indices[j].tolist(), top.values[j].tolist()))
                alts[t] = logp[j, t].item()
                best = sorted(alts.items(), key=lambda kv: -kv[1])[:top_k]
                ref.append([t, alts[t], {str(k): v for k, v in best}])
            rows.append({**s, "ref": ref})
    _jsonl_write(out / "sequences.jsonl", rows)
    meta = {"model": model, "source": "transformers", "transformers": transformers.__version__,
            "torch": torch.__version__, "revision": Path(cached_file(model, "config.json")).parent.name, "dtype": str(lm.dtype),
            "top_k": top_k, "n_items": len(items), "n_sequences": len(rows)}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta
