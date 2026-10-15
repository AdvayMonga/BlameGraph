"""Held-out access control: a Thresholdout query budget (Dwork et al. 2015, reusable holdout) kept in a write-once
store, and a sealed split that can be unsealed once per corpus version. Stdlib only."""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import secrets
import tempfile
import time
from pathlib import Path


class Refused(RuntimeError):
    """Held-out budget exhausted, or a sealed split cannot be unsealed."""


def _atomic_write(path: Path, data: dict) -> None:
    """Write JSON to a temp file in the same directory, then rename it over path."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, indent=1)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def _laplace(rng: random.Random, scale: float) -> float:
    return rng.expovariate(1 / scale) - rng.expovariate(1 / scale)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


STATE = re.compile(r"/state-(\d{6})\.json$")


def _step(s: dict, seen_value: float, heldout_value: float, threshold: float, sigma: float) -> tuple[dict, dict]:
    """One Thresholdout query on state `s`: (the answer, the next state). The seen value if it agrees with held-out
    within the noisy threshold, else held-out + Laplace (spends budget). Pure: the noise is drawn from the state's
    seed and counters."""
    if s["budget_left"] <= 0:
        raise Refused(f"held-out budget of {s['params']['budget']} exhausted after {s['n_queries']} queries")
    seed = s["seed"]
    spent = s["params"]["budget"] - s["budget_left"]
    noisy_t = threshold + _laplace(random.Random(f"{seed}:t{spent}"), 2 * sigma)   # refreshed per spend
    rng = random.Random(f"{seed}:q{s['n_queries']}")
    used = abs(seen_value - heldout_value) > noisy_t + _laplace(rng, 4 * sigma)
    out = {"value": heldout_value + _laplace(rng, sigma) if used else seen_value,
           "used_heldout": used, "budget_left": s["budget_left"] - used}
    n = s["n_queries"] + 1
    return out, {**s, "n_queries": n, "budget_left": out["budget_left"],
                 "log": s["log"] + [{"t": time.time(), "i": n, **out}]}             # inputs omitted


class HoldoutGuard:
    """Thresholdout for one scalar metric per query, its state in a write-once store (store/remote.py: `keys`,
    `get`, and a `put` that refuses an existing key). State n is `<prefix>/state-<n>.json`; a query writes n + 1 and,
    when another writer took n + 1 first, re-reads and tries again, so every run and machine sharing the store shares
    one budget and loses no query. The seed is drawn once, with state 0, and lives only in the store."""

    RACES = 100

    def __init__(self, remote, prefix: str, budget: int, threshold: float, sigma: float, seed=None):
        self.remote, self.prefix = remote, prefix.rstrip("/")
        self.threshold, self.sigma = threshold, sigma
        params = {"budget": budget, "threshold": threshold, "sigma": sigma}
        if not self._keys():          # a lost race here is another writer creating the same guard
            self._put(0, {"params": params, "seed": secrets.token_hex(16) if seed is None else str(seed),
                          "budget_left": budget, "n_queries": 0, "log": []})
        if self._latest()[1]["params"] != params:
            raise ValueError(f"{self.prefix} was created with different parameters")

    def _keys(self) -> list[str]:
        return sorted(k for k in self.remote.keys(self.prefix + "/") if STATE.search(k))

    def _latest(self) -> tuple[int, dict]:
        key = self._keys()[-1]
        with tempfile.TemporaryDirectory() as d:
            self.remote.get(key, Path(d) / "s.json")
            return int(STATE.search(key)[1]), json.loads((Path(d) / "s.json").read_text())

    def _put(self, n: int, state: dict) -> bool:
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "s.json").write_text(json.dumps(state, indent=1))
            return self.remote.put(f"{self.prefix}/state-{n:06d}.json", Path(d) / "s.json")

    def state(self) -> dict:
        """The latest state without its seed."""
        return {k: v for k, v in self._latest()[1].items() if k != "seed"}

    def query(self, seen_value: float, heldout_value: float) -> dict:
        for _ in range(self.RACES):
            n, s = self._latest()
            out, nxt = _step(s, seen_value, heldout_value, self.threshold, self.sigma)
            if self._put(n + 1, nxt):
                return out
        raise RuntimeError(f"{self.prefix}: lost the race for the next state {self.RACES} times")


class SealedSplit:
    """Sealed files recorded by sha256 in a JSON manifest; unsealing is logged and allowed once per corpus version."""

    def __init__(self, manifest_path):
        self.path = Path(manifest_path)
        self.manifest = json.loads(self.path.read_text())
        self.log_path = self.path.with_name(self.path.name + ".unseal.jsonl")

    @classmethod
    def seal(cls, manifest_path, files, corpus_version: str) -> "SealedSplit":
        """Hash files and write the manifest; paths are stored relative to the manifest's directory."""
        path = Path(manifest_path)
        hashes = {os.path.relpath(Path(f).resolve(), path.parent.resolve()): _sha256(Path(f)) for f in files}
        _atomic_write(path, {"corpus_version": corpus_version, "files": hashes})
        return cls(path)

    def verify(self) -> list[str]:
        """Manifest paths that are missing or changed; empty means intact."""
        return [rel for rel, h in self.manifest["files"].items()
                if not (self.path.parent / rel).is_file() or _sha256(self.path.parent / rel) != h]

    def unseal(self, reason: str, who: str) -> list[Path]:
        """Log the unseal and return the file paths; refuses a second unseal of the same version or changed files."""
        version = self.manifest["corpus_version"]
        log = self.log_path.read_text().splitlines() if self.log_path.exists() else []
        if any(json.loads(ln)["corpus_version"] == version for ln in log):
            raise Refused(f"corpus version {version!r} was already unsealed")
        if bad := self.verify():
            raise Refused(f"sealed files changed or missing: {bad}")
        with self.log_path.open("a") as f:
            f.write(json.dumps({"t": time.time(), "corpus_version": version, "reason": reason, "who": who}) + "\n")
            f.flush(); os.fsync(f.fileno())
        return [self.path.parent / rel for rel in self.manifest["files"]]
