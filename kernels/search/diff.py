"""SEARCH/REPLACE blocks: parse a model's response and apply it to a program, all-or-nothing."""
from __future__ import annotations

import re

BLOCK = re.compile(r"<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE", re.S)


def apply(code: str, response: str) -> tuple[str | None, str]:
    """(the new program, "") or (None, why): every block's SEARCH text must occur exactly once."""
    blocks = BLOCK.findall(response)
    if not blocks:
        return None, "no SEARCH/REPLACE block in the response"
    for i, (old, new) in enumerate(blocks, 1):
        n = code.count(old)
        if n != 1:
            return None, f"block {i}: its SEARCH text occurs {n} times in the program (must be exactly once)"
        code = code.replace(old, new, 1)
    return code, ""
