"""The one place kernels.search calls a model: Claude through the Anthropic SDK, streamed (a diff can be long), each
call priced from its usage at lab.apiproxy's rates (web searches at $10 per 1,000). Opus/Sonnet 5.5 think adaptively
(always on) at an explicit effort, with server-side refusal fallbacks; Haiku 4.5 takes neither.

With `tools=True` the model may use two tools, its own choice when: Anthropic's web search (server-side), and
`query`, read-only SQL over the agent's view of the ledger index (store/index.py: every run's records, kernel-loop
runs and their candidates included; never held-out rows or a person's runs). The harness pastes no research or
advice into a prompt; what the model reads it chose to read."""
from __future__ import annotations

import json
from pathlib import Path

from lab.apiproxy import cost_usd, price

FALLBACK_BETA = "server-side-fallback-2026-07-01"
WEB_SEARCH_USD = 0.01
QUERY_MAX_BYTES = 20_000
RETRIES = 3                                      # a turn whose stream dies (a read timeout, a dropped connection) is re-sent
TOOLS = [
    {"type": "web_search_20260209", "name": "web_search", "max_uses": 5},
    {"name": "query", "strict": True,
     "description": "Run one read-only DuckDB SQL query over every run recorded in this lab's ledger. Tables: records "
                    "(every ledger record: id, kind, run, tool, args, config, result, metrics as JSON; kind = 'kernels' "
                    "rows are kernel-loop runs, their result holding each candidate's lineage and seen-shape metrics), "
                    "kernels and ops (profile bundles' device kernels and operators), regimes, noise. `DESCRIBE "
                    "<table>` lists columns. Returns columns and up to 200 rows as JSON, cut under 20 kB.",
     "input_schema": {"type": "object", "properties": {"sql": {"type": "string"}}, "required": ["sql"],
                      "additionalProperties": False}},
]


def run_query(sql: str, root: Path | None = None) -> str:
    from lab import ledger
    try:
        from store import index                  # duckdb: missing on a host without the store extra
        out = index.query(index.agent_db(root or ledger.ROOT), sql)
    except Exception as e:                       # noqa: BLE001  a bad query is the model's to read
        return f"query failed: {type(e).__name__}: {e}"[:2000]
    text = json.dumps(out, default=str)
    while len(text) > QUERY_MAX_BYTES and out["rows"]:
        out["rows"], out["truncated"] = out["rows"][: len(out["rows"]) // 2], True
        text = json.dumps(out, default=str)
    return text


def _transient(e: Exception) -> bool:
    """A failure a resend can cure: the connection or the stream died, or the API was overloaded. The SDK retries these
    when a request starts, not once a stream is under way."""
    import anthropic
    if isinstance(e, (anthropic.APIConnectionError, anthropic.InternalServerError, anthropic.RateLimitError,
                      TimeoutError, ConnectionError)):
        return True
    return type(e).__name__ in ("ReadTimeout", "ReadError", "RemoteProtocolError", "WriteTimeout", "PoolTimeout")


class Claude:
    def __init__(self, model: str = "claude-opus-5-5", effort: str = "high", max_tokens: int = 64000,
                 tools: bool = False, max_turns: int = 8, ledger_root: Path | None = None):
        import anthropic
        self.client, self.model, self.effort, self.max_tokens = anthropic.Anthropic(), model, effort, max_tokens
        self.tools, self.max_turns, self.ledger_root = tools, max_turns, ledger_root

    def _usd(self, m) -> float:
        u = m.usage.model_dump()
        web = ((u.get("server_tool_use") or {}).get("web_search_requests") or 0) * WEB_SEARCH_USD
        return cost_usd(m.model if price(m.model) else self.model, u) + web      # a dated id prices as its model

    def _turn(self, system, messages, kw):
        """One turn, re-sent up to RETRIES times when it dies in transit (a dead stream's partial usage is not priced:
        the API reports usage only with the final message)."""
        for attempt in range(RETRIES + 1):
            try:
                return _stream(self.client, model=self.model, max_tokens=self.max_tokens, system=system,
                               messages=messages, **kw)
            except Exception as e:               # noqa: BLE001
                if attempt == RETRIES or not _transient(e):
                    raise
                import time
                time.sleep(5 * 2 ** attempt)

    def __call__(self, system: str, user: str) -> tuple[str, float]:
        """(the final response's text, the cost of every turn in $); a refused request returns no text."""
        kw = {} if self.model.startswith("claude-haiku") else {
            "output_config": {"effort": self.effort}, "betas": [FALLBACK_BETA], "fallbacks": "default"}
        if self.tools:
            kw["tools"] = TOOLS
        messages, usd = [{"role": "user", "content": user}], 0.0
        for _ in range(self.max_turns):
            m = self._turn(system, messages, kw)
            usd += self._usd(m)
            if m.stop_reason == "refusal":
                return "", usd
            if m.stop_reason not in ("tool_use", "pause_turn"):
                break
            messages.append({"role": "assistant", "content": m.content})
            if m.stop_reason == "tool_use":
                results = [{"type": "tool_result", "tool_use_id": b.id,
                            "content": run_query(str((b.input or {}).get("sql") or ""), self.ledger_root)}
                           for b in m.content if b.type == "tool_use" and b.name == "query"]
                messages.append({"role": "user", "content": results or [{"type": "text", "text": "no tool ran"}]})
        return "".join(b.text for b in m.content if b.type == "text"), usd


def _stream(client, **kw):
    with client.beta.messages.stream(**kw) as s:
        return s.get_final_message()
