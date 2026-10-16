"""OpenEvolve's OpenAI-compatible client, minus the parameters current Claude models refuse over Anthropic's
OpenAI-compatible endpoint (a temperature, which OpenEvolve always sends). Named as `init_client` in the config."""
from __future__ import annotations

from openevolve.llm.openai import OpenAILLM


class ClaudeCompat(OpenAILLM):
    async def _call_api(self, params):
        params = {k: v for k, v in params.items() if k not in ("temperature", "top_p", "seed")}
        return await super()._call_api(params)


def init_client(model_cfg):
    return ClaudeCompat(model_cfg)
