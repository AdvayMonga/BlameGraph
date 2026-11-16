"""OpenEvolve's OpenAI-compatible client without temperature, top_p and seed; named as `init_client` in the config.
Each call's cost is appended to $KERNELS_USAGE_LOG."""
from __future__ import annotations

import json
import os

from openevolve.llm.openai import OpenAILLM


class ClaudeCompat(OpenAILLM):
    async def _call_api(self, params):
        params = {k: v for k, v in params.items() if k not in ("temperature", "top_p", "seed")}
        out = await super()._call_api(params)
        u, log = getattr(self, "last_usage", None), os.environ.get("KERNELS_USAGE_LOG")
        if u and log:
            from lab.apiproxy import cost_usd
            usd = cost_usd(u["model"], {"input_tokens": u["prompt_tokens"], "output_tokens": u["completion_tokens"]})
            with open(log, "a") as f:
                f.write(json.dumps({"model": u["model"], "usd": usd}) + "\n")
        return out



def init_client(model_cfg):
    return ClaudeCompat(model_cfg)
