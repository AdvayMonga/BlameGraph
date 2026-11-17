"""Every model call of a search, kept whole: `<out>/calls.jsonl` has one row per call, its prompt parts and answer as
keys into `<out>/texts.jsonl`, which stores each distinct text once and every program a writer made. Both files are
appended to: a search that continues in `<out>` counts on from the calls already there."""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Callable

from lab.kernels.search.llm import CACHE_SPLIT

Model = Callable[[str, str], tuple[str, float]]  # (system, user) -> (text, $)


def lines(path: Path) -> list[str]:
    """A file's whole lines; a line cut short by a stop is dropped from the file."""
    if not path.exists():
        return []
    text = path.read_text()
    whole = text[:text.rfind("\n") + 1]
    if whole != text:
        path.write_text(whole)
    return whole.splitlines()


class Calls:
    def __init__(self, out: Path):
        had = [json.loads(x) for x in lines(out / "calls.jsonl")]
        self.seen = {json.loads(x)["key"] for x in lines(out / "texts.jsonl")}
        self.n, self.usd, self.lock = len(had), sum(r["usd"] for r in had), threading.Lock()
        self.calls, self.texts = open(out / "calls.jsonl", "a"), open(out / "texts.jsonl", "a")

    def put(self, text: str) -> str:
        """The key `text` is stored under."""
        key = hashlib.sha256(text.encode()).hexdigest()[:16]
        with self.lock:
            if key not in self.seen:
                self.seen.add(key)
                self.texts.write(json.dumps({"key": key, "text": text}) + "\n")
                self.texts.flush()
        return key

    def __call__(self, model: Model, role: str, generation: int, system: str, user: str) -> tuple[str, str]:
        """(the answer, the call's id)."""
        text, usd = model(system, user)
        parts = {k: [self.put(t) for t in v.split(CACHE_SPLIT) if t] for k, v in (("system", system), ("user", user))}
        answer = self.put(text or "")
        with self.lock:
            self.n, self.usd = self.n + 1, self.usd + usd
            row = {"id": f"c{self.n}", "role": role, "generation": generation, "model": getattr(model, "model", None),
                   **parts, "answer": answer, "usd": usd,
                   "stop": getattr(getattr(model, "last", None), "stop", None)}
            self.calls.write(json.dumps(row) + "\n")
            self.calls.flush()
        return text, row["id"]

    def close(self) -> None:
        self.calls.close()
        self.texts.close()
