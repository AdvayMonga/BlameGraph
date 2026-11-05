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
            at = _loose(code, old)
            if at is None:
                return None, f"block {i}: its SEARCH text occurs {n} times in the program (must be exactly once)"
            lines = code.splitlines(True)
            code = "".join(lines[:at[0]]) + new + ("\n" if at[1] < len(lines) or code.endswith("\n") else "") \
                + "".join(lines[at[1]:])
            continue
        code = code.replace(old, new, 1)
    return code, ""


def _loose(code: str, old: str) -> tuple[int, int] | None:
    """The line span where `old` occurs exactly once when trailing whitespace is ignored on every line, else None
    (OpenEvolve's applier tolerates it too)."""
    want = [x.rstrip() for x in old.split("\n")]
    have = [x.rstrip() for x in code.splitlines()]
    hits = [k for k in range(len(have) - len(want) + 1) if have[k:k + len(want)] == want]
    return (hits[0], hits[0] + len(want)) if len(hits) == 1 else None
