"""OpenEvolve's OpenAI-compatible client, minus the parameters current Claude models refuse over Anthropic's
OpenAI-compatible endpoint (a temperature, which OpenEvolve always sends). Named as `init_client` in the config.
Each call's usage is priced at lab.apiproxy's rates and appended to $KERNELS_USAGE_LOG (OpenEvolve calls models from
worker processes, so a file, not a counter), so a run's model cost is measured, as kernels.search's is."""
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


def usage(log) -> dict:
    """The run's model cost from the usage log: {usd, model_calls}."""
    rows = [json.loads(x) for x in open(log).read().splitlines() if x.strip()] if os.path.exists(log) else []
    return {"usd": round(sum(r["usd"] for r in rows), 4), "model_calls": len(rows)}


def init_client(model_cfg):
    return ClaudeCompat(model_cfg)
