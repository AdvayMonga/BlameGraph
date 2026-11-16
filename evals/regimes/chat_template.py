"""Fingerprint of the chat template the tokenizer applies to a corpus prompt, checked against the server's token count.
transformers is imported lazily and every failure yields None."""

from __future__ import annotations

import hashlib
from typing import Any

# The `enable_thinking` the shim's chat route passes to Tokenizer.encode_messages.
SHIM_ENABLE_THINKING = False

_CACHE: dict[str, Any] = {}


def _tokenizer(model_name: str):
    """AutoTokenizer, cached per process. None when transformers or the model is unavailable."""
    if model_name not in _CACHE:
        try:
            from transformers import AutoTokenizer
            _CACHE[model_name] = AutoTokenizer.from_pretrained(model_name)
        except Exception:                # noqa: BLE001 — no tokenizer is a state, not a crash
            _CACHE[model_name] = None
    return _CACHE[model_name]


def rendered_len(model_name: str, prompt: str, *,
                 enable_thinking: bool = SHIM_ENABLE_THINKING) -> int | None:
    """Token count the shim's chat route would encode `prompt` to, or None if unknowable."""
    tk = _tokenizer(model_name)
    if tk is None or not getattr(tk, "chat_template", None):
        return None
    try:
        return len(tk.apply_chat_template(
            [{"role": "user", "content": prompt}], return_dict=True, return_tensors=None,
            add_generation_prompt=True, enable_thinking=enable_thinking)["input_ids"])
    except Exception:                    # noqa: BLE001
        return None


def fingerprint(model_name: str, *,
                enable_thinking: bool = SHIM_ENABLE_THINKING) -> dict[str, Any] | None:
    """Identify the template that will be applied; None when it cannot be read.
    `probe_tokens` is the rendered length of a fixed one-word prompt."""
    tk = _tokenizer(model_name)
    template = getattr(tk, "chat_template", None) if tk is not None else None
    if not isinstance(template, str):
        return None
    return {
        "tokenizer": model_name,
        "chat_template_sha256": hashlib.sha256(template.encode()).hexdigest(),
        "enable_thinking": enable_thinking,
        "probe_tokens": rendered_len(model_name, "probe", enable_thinking=enable_thinking),
        "verified": None,
    }


def verify(fp: dict[str, Any] | None, prompt: str, server_prompt_tokens: int | None,
           *, enable_thinking: bool = SHIM_ENABLE_THINKING) -> dict[str, Any] | None:
    """Set `verified` by re-rendering `prompt` and comparing with the server's count; None when either is unknown."""
    if fp is None or server_prompt_tokens is None:
        return fp
    mine = rendered_len(fp["tokenizer"], prompt, enable_thinking=enable_thinking)
    if mine is None:
        return fp
    fp["verified"] = mine == server_prompt_tokens
    if not fp["verified"]:
        fp["verify_detail"] = (f"client rendered {mine} prompt tokens, server encoded "
                               f"{server_prompt_tokens}")
    return fp
