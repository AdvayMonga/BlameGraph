"""The target spec: one TOML file naming the model, the engine (repo, python, launch, write surface, test and lint
commands, logprobs API), the reference, the latency limits and the corpus. Selected by LAB_TARGET; never defaulted."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

ENV_ROOT = Path(__file__).resolve().parents[1]
TARGETS_DIR = ENV_ROOT / "targets"


class NoTarget(RuntimeError):
    pass


@dataclass(frozen=True)
class Limits:
    ttft_s: float
    tpot_s: float


@dataclass(frozen=True)
class Server:
    """How to launch something that serves the model: a command with `{port}`, its env, its health path."""
    launch: str
    env: dict = field(default_factory=dict)
    health: str = "/health"
    api: str = "vllm"
    startup_timeout_s: float = 1800.0


@dataclass(frozen=True)
class Target:
    path: Path
    model: str
    chat_kwargs: dict
    max_model_len: int | None
    engine_repo: Path
    engine_python: str
    engine: Server
    write: tuple[str, ...]
    add_only: tuple[str, ...]
    test: str
    lint: str
    reference: Server | None
    reference_dir: Path | None
    interactive: Limits
    conversational: Limits
    tasks: tuple[str, ...]
    min_score_ratio: float
    min_length_ratio: float
    corpus_dir: Path
    raw: dict

    @property
    def name(self) -> str:
        return self.path.stem


def _server(d: dict | None) -> Server | None:
    if not d:
        return None
    return Server(d["launch"], dict(d.get("env") or {}), d.get("health", "/health"), d.get("api", "vllm"),
                  float(d.get("startup_timeout_s", 1800)))


def parse(path: Path) -> Target:
    path = Path(path).resolve()
    raw = tomllib.loads(path.read_text())
    m, e, r = raw["model"], raw["engine"], raw.get("reference")
    repo = Path(os.environ.get("LAB_ENGINE_REPO") or (ENV_ROOT / e["repo"])).expanduser().resolve()
    py = os.environ.get("LAB_ENGINE_PYTHON") or str((repo / e.get("python", ".venv/bin/python")))
    lim = raw.get("limits", {})
    c = raw.get("correctness", {})
    return Target(
        path=path, model=m["name"], chat_kwargs=dict(m.get("chat_kwargs") or {}), max_model_len=m.get("max_model_len"),
        engine_repo=repo, engine_python=py, engine=_server(e),
        write=tuple(e.get("write") or ()), add_only=tuple(e.get("add_only") or ()),
        test=e["test"], lint=e["lint"],
        reference=_server(r), reference_dir=(ENV_ROOT / r["dir"]) if r and r.get("dir") else None,
        interactive=Limits(**lim.get("interactive", {"ttft_s": 0.5, "tpot_s": 0.030})),
        conversational=Limits(**lim.get("conversational", {"ttft_s": 2.0, "tpot_s": 0.100})),
        tasks=tuple(c.get("tasks") or ("mmlu_pro", "math", "code", "needle")),
        min_score_ratio=float(c.get("min_score_ratio", 0.99)), min_length_ratio=float(c.get("min_length_ratio", 0.90)),
        corpus_dir=ENV_ROOT / raw.get("corpus", {}).get("dir", "corpus"), raw=raw)


_cache: dict[str, Target] = {}


def load(path: str | Path | None = None) -> Target:
    """`path`, else LAB_TARGET. No target is an error that lists what targets/ holds; nothing is assumed."""
    p = path or os.environ.get("LAB_TARGET")
    if not p:
        have = sorted(x.name for x in TARGETS_DIR.glob("*.toml"))
        raise NoTarget("no target: set LAB_TARGET=targets/<name>.toml (or pass --target). "
                       + (f"Available: {', '.join(have)}" if have else "targets/ is empty"))
    p = Path(p).expanduser()
    if not p.is_absolute():
        p = (Path.cwd() / p) if (Path.cwd() / p).exists() else (ENV_ROOT / p)
    key = str(p.resolve())
    if key not in _cache:
        _cache[key] = parse(p)
    return _cache[key]


def add_argument(parser) -> None:
    """`--target PATH` on a CLI, defaulting to LAB_TARGET."""
    parser.add_argument("--target", default=None, help="target spec (default: $LAB_TARGET)")
