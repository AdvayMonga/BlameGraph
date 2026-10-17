"""The one place kernels.search calls a model: Claude through the Anthropic SDK, streamed (a diff can be long), each
call priced from its usage at lab.apiproxy's rates. Opus/Sonnet 5.5 think adaptively (always on) at an explicit
effort, with server-side refusal fallbacks; Haiku 4.5 takes neither."""
from __future__ import annotations

from lab.apiproxy import cost_usd

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class Claude:
    def __init__(self, model: str = "claude-opus-5-5", effort: str = "high", max_tokens: int = 64000):
        import anthropic
        self.client, self.model, self.effort, self.max_tokens = anthropic.Anthropic(), model, effort, max_tokens

    def __call__(self, system: str, user: str) -> tuple[str, float]:
        """(the response's text, its cost in $); a refused request returns no text."""
        kw = {} if self.model.startswith("claude-haiku") else {
            "output_config": {"effort": self.effort}, "betas": [FALLBACK_BETA], "fallbacks": "default"}
        with self.client.beta.messages.stream(model=self.model, max_tokens=self.max_tokens, system=system,
                                              messages=[{"role": "user", "content": user}], **kw) as s:
            m = s.get_final_message()
        usd = cost_usd(m.model, m.usage.model_dump())
        if m.stop_reason == "refusal":
            return "", usd
        return "".join(b.text for b in m.content if b.type == "text"), usd
