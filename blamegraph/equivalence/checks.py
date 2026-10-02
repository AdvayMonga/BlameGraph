"""Output-level checks: no truncation, and the reported tokens match the returned text."""
from __future__ import annotations

from typing import Callable


def length_ratio(ref_lengths: list[int], cand_lengths: list[int]) -> dict:
    """Candidate output length relative to the reference over the same requests (MLPerf requires >= 90%)."""
    if len(ref_lengths) != len(cand_lengths):
        raise ValueError("length lists must be paired request by request")
    tr, tc = sum(ref_lengths), sum(cand_lengths)
    short = sum(1 for r, c in zip(ref_lengths, cand_lengths) if r and c < 0.5 * r)
    return {"ratio": tc / tr if tr else float("nan"), "n": len(ref_lengths), "requests_under_half_length": short}


def consistency(responses: list[dict], tokenize: Callable[[str], list], eos_text: tuple[str, ...] = ()) -> dict:
    """Each response: {"text": full text, "first_token_text": text of the first streamed token chunk,
    "reported_tokens": token count the server reported}. Counts are re-derived with the reference tokenizer."""
    bad_first = bad_count = after_eos = 0
    for r in responses:
        text = r.get("text") or ""
        ft = r.get("first_token_text")
        if ft is not None and (not ft or not text.startswith(ft)):
            bad_first += 1
        rep = r.get("reported_tokens")
        if rep is not None and rep != len(tokenize(text)):
            bad_count += 1
        for e in eos_text:
            k = text.find(e)
            if k != -1 and text[k + len(e):].strip():
                after_eos += 1
                break
    n = len(responses)
    return {"n": n, "first_token_mismatch": bad_first, "token_count_mismatch": bad_count, "text_after_eos": after_eos,
            "consistent": n > 0 and bad_first == bad_count == after_eos == 0}
