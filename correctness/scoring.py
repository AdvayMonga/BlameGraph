"""Answer extraction and scoring for the flip-test task sets (non-thinking outputs).

Implemented: multiple choice (MMLU-Pro, letters A-J) and math final answers (MATH-500 style \\boxed{...}).
Code (unit-tested) and long-context retrieval scorers come next.
"""
from __future__ import annotations

import re

_CHOICE_PATTERNS = [
    r"answer is \(?([A-J])\)?",          # MMLU-Pro's own prompt format: "The answer is (X)"
    r"[Aa]nswer\s*[:：]\s*\(?([A-J])\)?",
    r"^\s*\(?([A-J])\)?[.)]?\s*$",       # a reply that is only the letter
]


def extract_choice(text: str) -> str | None:
    for pat in _CHOICE_PATTERNS:
        m = re.findall(pat, text, flags=re.M)
        if m:
            return m[-1]
    m = re.findall(r"\b([A-J])\b", text)   # fallback: the last standalone letter
    return m[-1] if m else None


def _last_boxed(text: str) -> str | None:
    k = text.rfind("\\boxed")
    if k == -1:
        return None
    i = text.find("{", k)
    if i == -1:
        return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j]
    return None


def normalize_math(ans: str) -> str:
    s = ans.strip().strip("$").replace("\\!", "").replace("\\left", "").replace("\\right", "")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\(d|t)frac", r"\\frac", s)
    s = re.sub(r"\s+", "", s).rstrip(".")
    s = re.sub(r"^([a-z]|\\[a-z]+)=", "", s)            # "x=5" -> "5"
    if re.fullmatch(r"-?\d+\.0+", s):
        s = s.split(".")[0]
    return s


def extract_math(text: str) -> str | None:
    b = _last_boxed(text)
    if b is not None:
        return normalize_math(b)
    m = re.findall(r"answer is\s*\$?([^\n$]+)", text)
    return normalize_math(m[-1]) if m else None


def math_equal(a: str | None, b: str | None) -> bool:
    if a is None or b is None:
        return False
    if a == b:
        return True
    try:
        return abs(float(a) - float(b)) <= 1e-6 * max(1.0, abs(float(b)))
    except ValueError:
        return False


def score(task: str, text: str, gold: str) -> bool:
    if task == "mmlu_pro":
        return extract_choice(text) == gold.strip().upper()
    if task == "math":
        return math_equal(extract_math(text), normalize_math(gold))
    raise ValueError(f"no scorer for task {task!r}")
