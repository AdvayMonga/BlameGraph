"""Convert data/laya/*_train.jsonl into the tokenized item cache the official MPS fine-tune script consumes.
Usage: laya_make_items.py --model-dir ./laya_base --task extract [--task judge] --out ./train_items.pt
Soft targets: label with the teacher's confidence when available, else 0.95/0.05."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "laya"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", required=True, help="path to laya_finetune_typed_decisions_mps.py")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--task", action="append", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-items", type=int, default=None)
    a = ap.parse_args()
    sys.path.insert(0, str(Path(a.script).parent))
    import importlib
    ft = importlib.import_module(Path(a.script).stem)
    ft.prepare_model(a.model_dir)
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    cfg = json.loads((Path(a.model_dir) / "rl_agent_config.json").read_text())
    cfg = {**cfg, "max_len": cfg.get("max_len", 1024), "head_max_len": cfg.get("head_max_len", 256)}
    items, skipped = [], 0
    for task in a.task:
        for line in (DATA / f"{task}_{a.split}.jsonl").read_text().splitlines():
            d = json.loads(line)
            conf = d.get("confidence")
            p_true = (conf if d["label"] else 1 - conf) if isinstance(conf, (int, float)) and 0.5 <= conf <= 1 else (0.95 if d["label"] else 0.05)
            q = {"type": "noul", "instructions": d["question"]}
            gold = {"probabilities": {"true": p_true, "false": 1 - p_true}}
            it = ft.build_training_item(tok, cfg, d["state"], q, gold)
            if it is None:
                skipped += 1
            else:
                items.append(it)
            if a.max_items and len(items) >= a.max_items:
                break
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(items, out)
    # sidecar meta matching the script's cache key so it treats the file as a valid cache
    meta = {"model_dir": str(Path(a.model_dir).resolve()), "max_len": cfg["max_len"], "head_max_len": cfg["head_max_len"],
            "dataset": ft.DATASET_ID, "split": "train"}
    out.with_name(out.name + ".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"saved {len(items)} items to {out} (skipped {skipped}); mean seq len {sum(len(i['ids']) for i in items)/max(1,len(items)):.0f}")


if __name__ == "__main__":
    main()
