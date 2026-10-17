"""A rehearsal of the whole loop on a laptop, no GPU: a practice engine (a small OpenAI-compatible server the agent
can edit and measure), a target spec for it, a task, and the reference answers equiv compares against.

  python -m lab.rehearsal setup DIR
    DIR/engine/       the practice engine: a git repo (src/inference_server/server.py, tests, ruff config)
    DIR/reference.py  an unmodified copy of its server: the reference, and validation's known-good/bad launches
    DIR/ref, ref-full the reference's answers to a synthetic multiple-choice set (what equiv compares against)
    DIR/target.toml, DIR/task.toml, DIR/store/ (a local store: the held-out guard, never the real one)

Then, from this repo (the ledger and run dirs are this checkout's own):
  export LAB_TARGET=DIR/target.toml LAB_STORE=DIR/store LAB_ENGINE_REPO=DIR/engine
  python -m lab.validate --regimes single_stream --noise-runs 3 --tier dev      # noise bands, the gate on good/bad
  python -m lab.session --task DIR/task.toml --budget 2                         # the loop, with the real model

The model's tokenizer counts tokens on both sides of every check, so the server streams words that are one token
each in it (checked at setup)."""
from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from lab import engine, rehearsal_server

MODEL = "Qwen/Qwen3-30B-A3B"       # its tokenizer only: nothing loads the weights
SERVER = Path(__file__).with_name("rehearsal_server.py")
TESTS = '''from inference_server.server import ANSWER, WORDS, answer


def test_multiple_choice_is_answered_c():
    chunks, n = answer([{"role": "user", "content": "q pick one"}], 8)
    assert chunks == [ANSWER] and n == 6


def test_other_prompts_stream_one_word_per_token():
    chunks, n = answer([{"role": "user", "content": "hello"}], 5)
    assert n == 5 and chunks == WORDS[:5]
'''


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=rehearsal", "-c", "user.email=rehearsal@lab",
                    "-c", "commit.gpgsign=false", *args], check=True, capture_output=True)


def make_engine(dest: Path) -> Path:
    """The practice engine as a one-commit git repo."""
    files = {"src/inference_server/__init__.py": "", "src/inference_server/server.py": SERVER.read_text(),
             "tests/test_server.py": TESTS, "tests/conftest.py": "import sys\nsys.path.insert(0, 'src')\n",
             "pyproject.toml": "[tool.ruff]\nline-length = 120\n[tool.ruff.lint]\nselect = ['F']\n",
             "README.md": "# practice engine\nA small OpenAI-compatible server for rehearsing the lab's loop.\n",
             ".gitignore": "__pycache__/\n*.pyc\n"}
    for rel, text in files.items():
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        (dest / rel).write_text(text)
    _git(dest, "init", "-q")
    _git(dest, "add", "-A")
    _git(dest, "commit", "-q", "-m", "practice engine")
    return dest


def _serve(script: Path, *extra: str):
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    proc = subprocess.Popen([sys.executable, str(script), "--port", str(port), *extra])
    for _ in range(200):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
            return proc, f"http://127.0.0.1:{port}"
        except OSError:
            time.sleep(0.05)
    proc.kill()
    raise RuntimeError(f"{script} did not start")


def items(n: int) -> list[dict]:
    """A synthetic multiple-choice set the practice engine answers: gold C, scored like MMLU-Pro."""
    return [{"id": f"mc-{k}", "task": "mmlu_pro", "gold": "C", "max_tokens": 8,
             "messages": [{"role": "user", "content": f"question {k}: pick one of A, B, C or D"}]} for k in range(n)]


def make_reference(script: Path, out: Path, n: int, encoder) -> dict:
    from correctness import run as crun
    proc, url = _serve(script)
    try:
        return crun.reference(url, MODEL, out, items(n), encoder, top_k=5, div_n=min(n, 60), concurrency=8)
    finally:
        proc.kill()


def target_toml(d: Path, python: str) -> str:
    ref = d / "reference.py"
    return f'''# The rehearsal target (python -m lab.rehearsal): the practice engine in {d / "engine"}.
[model]
name = "{MODEL}"                             # its tokenizer counts tokens; nothing loads the weights
chat_kwargs = {{ enable_thinking = false }}

[engine]
repo = "{d / "engine"}"
python = "{python}"
api = "vllm"
launch = "python src/inference_server/server.py --port {{port}}"
health = "/health"
startup_timeout_s = 60
write = ["src/inference_server/*"]
add_only = ["tests/test_*.py"]
test = "python -m pytest -q -p no:cacheprovider -c pyproject.toml"   # its own config: pytest stops looking upward
lint = "python -m ruff check ."

[reference]
api = "vllm"
launch = "{python} {ref} --port {{port}}"
dir = "{d / "ref"}"

[limits]                                     # MLPerf Llama-3.1-8B latency limits, checked at p99
interactive = {{ ttft_s = 0.5, tpot_s = 0.030 }}
conversational = {{ ttft_s = 2.0, tpot_s = 0.100 }}

[correctness]
tasks = ["mmlu_pro"]
min_score_ratio = 0.99
min_length_ratio = 0.90
submit_tier = "dev"

[corpus]
dir = "{engine.ENV_ROOT / "corpus"}"

[validation]                                 # the gate must pass the unmodified server and fail a wrong one
good = {{ same = "{python} {ref} --port {{port}}" }}
bad = {{ wrong = "{python} {ref} --port {{port}} --wrong-every 3" }}
'''


TASK = '''[task]
goal = "cut single_stream p99 time per output token"
[objective]
regimes = ["single_stream"]
'''


def setup(d: Path, python: str = sys.executable, n_dev: int = 200, n_full: int = 400, encoder=None) -> dict:
    from correctness import run as crun
    d = d.resolve()
    if (d / "engine").exists():
        raise SystemExit(f"{d / 'engine'} exists: set up a fresh directory")
    d.mkdir(parents=True, exist_ok=True)
    enc = encoder or crun.HFEncoder(MODEL, {"enable_thinking": False})
    one = [len(enc.encode(w)) for w in rehearsal_server.WORDS]
    if set(one) != {1}:
        raise SystemExit(f"the server's words are not one token each in {MODEL}: {one}")
    make_engine(d / "engine")
    (d / "reference.py").write_text(SERVER.read_text())
    (d / "store").mkdir(exist_ok=True)
    (d / "target.toml").write_text(target_toml(d, python))
    (d / "task.toml").write_text(TASK)
    refs = {"dev": make_reference(d / "reference.py", d / "ref", n_dev, enc),
            "full": make_reference(d / "reference.py", d / "ref-full", n_full, enc)}
    return {"dir": str(d), "engine": str(d / "engine"), "target": str(d / "target.toml"), "task": str(d / "task.toml"),
            "store": str(d / "store"), "references": refs}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m lab.rehearsal", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("setup")
    p.add_argument("dir", type=Path)
    a = ap.parse_args(argv)
    print(json.dumps(setup(a.dir), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
