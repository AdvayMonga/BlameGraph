"""Held-out access control: a Thresholdout query budget (Dwork et al. 2015, reusable holdout) and a sealed split
that can be unsealed once per corpus version. Stdlib only; state lives in JSON files written atomically."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import random
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


class HoldoutGuard:
    """Thresholdout for one scalar metric per query; budget, count and query log persist in state_path."""

    def __init__(self, state_path, budget: int, threshold: float, sigma: float, seed):
        self.path = Path(state_path)
        self.threshold, self.sigma, self.seed = threshold, sigma, seed
        params = {"budget": budget, "threshold": threshold, "sigma": sigma,
                  "seed_sha256": hashlib.sha256(str(seed).encode()).hexdigest()}
        if not self.path.exists():
            _atomic_write(self.path, {"params": params, "budget_left": budget, "n_queries": 0, "log": []})
        elif json.loads(self.path.read_text())["params"] != params:
            raise ValueError(f"{self.path} was created with different parameters")

    def state(self) -> dict:
        return json.loads(self.path.read_text())

    def query(self, seen_value: float, heldout_value: float) -> dict:
        """Seen value if it agrees with held-out within the noisy threshold, else held-out + Laplace (spends budget).
        Read-modify-write under an exclusive lock, so parallel sessions sharing the state file never lose a query."""
        with open(self.path.with_name(self.path.name + ".lock"), "a") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            try:
                return self._query(seen_value, heldout_value)
            finally:
                fcntl.flock(lf, fcntl.LOCK_UN)

    def _query(self, seen_value: float, heldout_value: float) -> dict:
        s = self.state()
        if s["budget_left"] <= 0:
            raise Refused(f"held-out budget of {s['params']['budget']} exhausted after {s['n_queries']} queries")
        spent = s["params"]["budget"] - s["budget_left"]
        noisy_t = self.threshold + _laplace(random.Random(f"{self.seed}:t{spent}"), 2 * self.sigma)  # refreshed per spend
        rng = random.Random(f"{self.seed}:q{s['n_queries']}")
        used = abs(seen_value - heldout_value) > noisy_t + _laplace(rng, 4 * self.sigma)
        out = {"value": heldout_value + _laplace(rng, self.sigma) if used else seen_value,
               "used_heldout": used, "budget_left": s["budget_left"] - used}
        s["n_queries"] += 1
        s["budget_left"] = out["budget_left"]
        s["log"].append({"t": time.time(), "i": s["n_queries"], **out})  # inputs omitted
        _atomic_write(self.path, s)
        return out


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
