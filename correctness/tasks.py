"""Task sets for the flip test: loaders, prompts (non-thinking), and the code and long-context scorers.

Every item: {"id", "task", "messages", "gold", "max_tokens"}. Files are fetched from the Hub into data/tasks/
(gitignored). Scorers only need to be consistent: reference and candidate are scored by the same function, and only
answers that differ between them carry information.
"""
from __future__ import annotations

import json
import os
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data" / "tasks"
LETTERS = "ABCDEFGHIJ"


def _hub(repo: str, filename: str) -> Path:
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo, filename, repo_type="dataset", local_dir=str(DATA / repo.split("/")[-1])))


# ---------------------------------------------------------------- MMLU-Pro (multiple choice, A-J)
def mmlu_pro(n: int | None = 500, seed: int = 0) -> list[dict]:
    """n questions stratified by category (n // categories each); None = the whole test set."""
    import pandas as pd
    d = pd.read_parquet(_hub("TIGER-Lab/MMLU-Pro", "data/test-00000-of-00001.parquet"))
    if n is not None:
        per_cat = max(1, n // d.category.nunique())
        d = pd.concat([g.sample(min(len(g), per_cat), random_state=seed) for _, g in d.groupby("category")])
    items = []
    for _, r in d.sort_values("question_id").iterrows():
        opts = "\n".join(f"({LETTERS[i]}) {o}" for i, o in enumerate(r.options))
        msg = (f"The following is a multiple choice question about {r.category}. Think step by step, then finish "
               f"your answer with \"The answer is (X)\" where X is the correct letter.\n\nQuestion: {r.question}\n{opts}")
        items.append({"id": f"mmlu_pro-{r.question_id}", "task": "mmlu_pro", "messages": [{"role": "user", "content": msg}],
                      "gold": r.answer, "max_tokens": 1024})
    return items


# ---------------------------------------------------------------- MATH-500 (final answer in \boxed{})
def math500(n: int | None = None, seed: int = 0) -> list[dict]:
    rows = [json.loads(l) for l in _hub("HuggingFaceH4/MATH-500", "test.jsonl").read_text().splitlines() if l.strip()]
    if n:
        rows = random.Random(seed).sample(rows, min(n, len(rows)))
    return [{"id": f"math-{r['unique_id']}", "task": "math", "gold": r["answer"], "max_tokens": 2048,
             "messages": [{"role": "user", "content": r["problem"] + "\n\nPlease reason step by step, and put your final "
                                                                       "answer within \\boxed{}."}]} for r in rows]


_MATH_SUBJECTS = ("algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory",
                  "prealgebra", "precalculus")


def math_full() -> list[dict]:
    """The full MATH test set (5,000); gold = the last \\boxed{} of the reference solution."""
    import pandas as pd
    from .scoring import extract_math
    items = []
    for sub in _MATH_SUBJECTS:
        d = pd.read_parquet(_hub("EleutherAI/hendrycks_math", f"{sub}/test-00000-of-00001.parquet"))
        for k, r in enumerate(d.itertuples()):
            items.append({"id": f"math-{sub}-{k}", "task": "math", "gold": extract_math(r.solution), "max_tokens": 2048,
                          "messages": [{"role": "user", "content": r.problem + "\n\nPlease reason step by step, and put "
                                                                              "your final answer within \\boxed{}."}]})
    return items


# ---------------------------------------------------------------- MBPP (code, scored by unit tests)
def mbpp(seed: int = 0, full: bool = False) -> list[dict]:
    """Sanitized MBPP (427) or the full set (974), all splits."""
    import pandas as pd
    cfg = "full" if full else "sanitized"
    parts = [pd.read_parquet(_hub("google-research-datasets/mbpp", f"{cfg}/{s}-00000-of-00001.parquet"))
             for s in ("test", "validation", "train", "prompt")]
    d = pd.concat(parts).drop_duplicates("task_id").sort_values("task_id")
    items = []
    for _, r in d.iterrows():
        tests = list(r.test_list)
        prompt = r.text if full else r.prompt
        imports = [l for l in (r.test_setup_code or "").splitlines() if l.strip()] if full else list(r.test_imports)
        msg = (f"You are an expert Python programmer. {prompt}\nYour code should pass these tests:\n\n"
               + "\n".join(tests) + "\n\nReturn only the code in a single ```python block.")
        items.append({"id": f"mbpp-{r.task_id}", "task": "code", "messages": [{"role": "user", "content": msg}],
                      "gold": {"imports": imports, "tests": tests}, "max_tokens": 1024})
    return items


def extract_code(text: str) -> str:
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.S)
    return max(blocks, key=len) if blocks else text


def _limits():  # best effort; RLIMIT_AS is unavailable on macOS
    import resource
    for res, val in ((resource.RLIMIT_CPU, 10), (getattr(resource, "RLIMIT_AS", None), 2 << 30)):
        if res is not None:
            try:
                resource.setrlimit(res, (val, val))
            except (ValueError, OSError):
                pass


def run_code_tests(text: str, gold: dict, timeout: float = 10.0) -> bool:
    """Run model-written code plus its asserts in an isolated interpreter (temp dir, no env, CPU/memory limits).
    This executes model output; run it on a disposable machine."""
    program = "\n".join(gold.get("imports") or []) + "\n" + extract_code(text) + "\n\n" + "\n".join(gold["tests"]) + "\n"
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "prog.py"; f.write_text(program)
        try:
            r = subprocess.run([sys.executable, "-I", str(f)], cwd=d, env={"PATH": os.environ.get("PATH", "")},
                               capture_output=True, timeout=timeout, preexec_fn=_limits if os.name == "posix" else None)
        except subprocess.TimeoutExpired:
            return False
    return r.returncode == 0


# ---------------------------------------------------------------- long-context retrieval (seeded needle-in-haystack)
_SUBJECTS = ["The river", "A quiet town", "The old library", "Every winter", "The market", "Her garden", "The harbor",
             "Our neighbor", "The train station", "That mountain", "The bakery", "A small school"]
_PREDICATES = ["changed slowly over the years", "was busier than usual", "smelled of rain and cedar",
               "had a long and unremarkable history", "was painted a pale shade of blue", "closed early on Sundays",
               "was remembered fondly by visitors", "stood near the edge of the valley", "rarely made the news",
               "was known for its patient staff"]


def needle(lengths_chars: tuple[int, ...] = (16_000, 48_000, 96_000), per_length: int = 60, n_needles: int = 4,
           seed: int = 0) -> list[dict]:
    """Synthetic retrieval: filler prose with several 'magic number' needles; ask for one key's number.
    Character lengths ~ 4k / 12k / 24k tokens. Distractor needles make it a retrieval task, not a lookup."""
    rng = random.Random(seed)
    items = []
    for L in lengths_chars:
        for k in range(per_length):
            keys = rng.sample([f"{a}-{b}" for a in ("amber", "cobalt", "crimson", "ivory", "jade", "onyx", "scarlet", "teal")
                               for b in ("falcon", "harbor", "lantern", "meadow", "orchid", "summit", "willow")], n_needles)
            nums = {key: str(rng.randint(1_000_000, 9_999_999)) for key in keys}
            filler = []
            while sum(len(s) for s in filler) < L:
                filler.append(f"{rng.choice(_SUBJECTS)} {rng.choice(_PREDICATES)}.")
            for key in keys:
                filler.insert(rng.randrange(len(filler)), f"The special magic number for {key} is: {nums[key]}.")
            target = rng.choice(keys)
            msg = (" ".join(filler) + f"\n\nWhat is the special magic number for {target} mentioned in the text above? "
                   "Answer with the number only.")
            items.append({"id": f"needle-{L}-{k}", "task": "needle", "messages": [{"role": "user", "content": msg}],
                          "gold": nums[target], "max_tokens": 32})
    return items


def score_item(item: dict, text: str) -> bool:
    from .scoring import score
    t = item["task"]
    if t == "code":
        return run_code_tests(text, item["gold"])
    if t == "needle":
        return item["gold"] in re.findall(r"\d{7}", text)
    return score(t, text, item["gold"])


def load(tasks: tuple[str, ...] = ("mmlu_pro", "math", "code", "needle"), seed: int = 0, tier: str = "dev") -> list[dict]:
    """dev: MMLU-Pro 490 stratified, MATH-500, MBPP sanitized 427, needle 180 (~1.6k, for iteration).
    full: MMLU-Pro 12,032, MATH 5,000, MBPP 974, needle 600 (~18.6k, for final tests)."""
    full = tier == "full"
    makers = {"mmlu_pro": lambda: mmlu_pro(n=None if full else 500, seed=seed),
              "math": lambda: math_full() if full else math500(seed=seed),
              "code": lambda: mbpp(seed=seed, full=full),
              "needle": lambda: needle(per_length=200 if full else 60, seed=seed)}
    return [it for t in tasks for it in makers[t]()]
