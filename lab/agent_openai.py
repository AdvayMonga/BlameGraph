"""A provider for any OpenAI-compatible chat-completions API (OpenAI, Gemini, DeepSeek, Mistral, OpenRouter, ...),
called directly with that provider's key. Selected by LAB_AGENT_PROVIDER=openai; LAB_MODEL names the model,
LAB_OPENAI_BASE_URL the API (https, public host; default OpenAI's), OPENAI_API_KEY the key, LAB_MODEL_PRICE
"in,out" its $ per million tokens. The loop runs here, outside the jail: the key never enters it, file tools are
confined to the workspace and pass the write guard, and Bash runs jailed with no network."""

from __future__ import annotations

import fnmatch
import ipaddress
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

from lab.agent import OUTPUT_SCHEMA, AgentReply, AgentSpec
from lab.safety import grader
from lab.safety.hooks import allowed

BASE_URL = "https://api.openai.com/v1"
MAX_TOOL_TEXT = 30_000                  # chars of one tool result the model sees (the tail is kept)
MAX_MATCHES = 200
BASH_TIMEOUT_S = 600
RETRY_STATUS = (429, 500, 502, 503, 504)


def base_url(url: str | None = None) -> str:
    """The API's base URL: https and a public host only, so nothing is routed through this machine or its network."""
    url = (url if url is not None else os.environ.get("LAB_OPENAI_BASE_URL", "")).strip() or BASE_URL
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    if parts.scheme != "https" or not host:
        raise ValueError(f"LAB_OPENAI_BASE_URL must be an https URL: {url!r}")
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        ip = None
    if host == "localhost" or host.endswith((".localhost", ".local", ".internal")) or (ip and not ip.is_global):
        raise ValueError(f"LAB_OPENAI_BASE_URL must be a public host, not {host!r}")
    return url.rstrip("/")


def prices(raw: str | None = None) -> tuple[float, float]:
    """LAB_MODEL_PRICE "in,out" in $ per million tokens. Required: without it the budget cannot be enforced."""
    raw = raw if raw is not None else os.environ.get("LAB_MODEL_PRICE", "")
    try:
        p_in, p_out = (float(x) for x in raw.split(","))
    except ValueError:
        raise ValueError(f'LAB_MODEL_PRICE must be "in,out" in $ per million tokens, got {raw!r}') from None
    if p_in < 0 or p_out < 0:
        raise ValueError(f"LAB_MODEL_PRICE must not be negative: {raw!r}")
    return p_in, p_out


def _http_post(url: str, key: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code not in RETRY_STATUS or attempt == 4:
                raise RuntimeError(f"{e.code} from the model API: {e.read()[:500]!r}") from None
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def _fn(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": properties, "required": required, "additionalProperties": False}}}


_S, _I, _B = {"type": "string"}, {"type": "integer"}, {"type": "boolean"}
BUILTIN = {
    "Read": _fn("Read", "Read a file in the workspace; returns numbered lines.",
                {"file_path": _S, "offset": _I, "limit": _I}, ["file_path"]),
    "Write": _fn("Write", "Create or overwrite a file in the workspace. Writes outside the write surface are refused.",
                 {"file_path": _S, "content": _S}, ["file_path", "content"]),
    "Edit": _fn("Edit", "Replace old_string with new_string in a workspace file; old_string must be unique unless "
                "replace_all. Writes outside the write surface are refused.",
                {"file_path": _S, "old_string": _S, "new_string": _S, "replace_all": _B},
                ["file_path", "old_string", "new_string"]),
    "Glob": _fn("Glob", "Workspace paths matching a glob pattern (e.g. `src/**/*.py`).", {"pattern": _S}, ["pattern"]),
    "Grep": _fn("Grep", "Lines matching a regex in workspace files; `glob` filters paths. Returns path:line:text.",
                {"pattern": _S, "glob": _S}, ["pattern"]),
    "Bash": _fn("Bash", f"Run a shell command in the workspace, inside the sandbox, with no network; returns exit "
                f"code and output tail. Timeout in seconds, at most {BASH_TIMEOUT_S}.",
                {"command": _S, "timeout_s": _I}, ["command"]),
}
FINISH = _fn("finish", "End this session with `status` (`continue` or `stop`) and `note`.",
             OUTPUT_SCHEMA["properties"], OUTPUT_SCHEMA["required"])


class Workbench:
    """The built-in tools, confined to the workspace; writes go through the same guard as the Claude provider."""

    def __init__(self, workspace: Path, base_files: frozenset[str]):
        self.ws, self.base_files = workspace.resolve(), base_files

    def _path(self, p: str) -> Path:
        target = (self.ws / p).resolve()
        if not target.is_relative_to(self.ws):
            raise ValueError(f"{p} is outside the workspace")
        return target

    def _files(self):
        for p in sorted(self.ws.rglob("*")):
            if p.is_file() and ".git" not in p.relative_to(self.ws).parts:
                yield p

    def Read(self, file_path: str, offset: int = 1, limit: int = 2000) -> str:
        lines = self._path(file_path).read_text(errors="replace").splitlines()
        start = max(offset, 1)
        return "\n".join(f"{i}\t{line}" for i, line in enumerate(lines[start - 1:start - 1 + limit], start))

    def _write(self, file_path: str, text: str) -> None:
        if not allowed(self.ws, self.base_files, file_path):
            raise PermissionError(f"{file_path} is outside what the agent may write")
        p = self._path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)

    def Write(self, file_path: str, content: str) -> str:
        self._write(file_path, content)
        return f"wrote {file_path}"

    def Edit(self, file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
        text = self._path(file_path).read_text()
        n = text.count(old_string)
        if n == 0 or (n > 1 and not replace_all):
            raise ValueError(f"old_string occurs {n} times in {file_path}")
        self._write(file_path, text.replace(old_string, new_string, -1 if replace_all else 1))
        return f"edited {file_path} ({n if replace_all else 1} replacement{'s' if replace_all and n > 1 else ''})"

    def Glob(self, pattern: str) -> str:
        hits = [p.relative_to(self.ws).as_posix() for p in self.ws.glob(pattern)
                if p.resolve().is_relative_to(self.ws)]
        return "\n".join(sorted(hits)[:MAX_MATCHES * 5]) or "no matches"

    def Grep(self, pattern: str, glob: str | None = None) -> str:
        rx, out = re.compile(pattern), []
        for p in self._files():
            rel = p.relative_to(self.ws).as_posix()
            if glob and not fnmatch.fnmatch(rel, glob):
                continue
            for i, line in enumerate(p.read_text(errors="replace").splitlines(), 1):
                if rx.search(line):
                    out.append(f"{rel}:{i}:{line}")
                    if len(out) >= MAX_MATCHES:
                        return "\n".join(out + [f"(stopped at {MAX_MATCHES} matches)"])
        return "\n".join(out) or "no matches"

    def Bash(self, command: str, timeout_s: int = 120) -> str:
        proc = grader.jailed(self.ws, ["bash", "-c", command], timeout_s=min(max(timeout_s, 1), BASH_TIMEOUT_S))
        return f"exit {proc.returncode}\n{proc.stdout}{proc.stderr}"


class OpenAICompatible:
    name = "openai"

    def __init__(self, url: str | None = None, api_key: str | None = None, price: tuple[float, float] | None = None,
                 post: Callable[[str, str, dict, float], dict] = _http_post):
        """Refuses to start without a valid URL, a key and prices."""
        self.url = base_url(url) + "/chat/completions"
        self.key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")
        if not self.key:
            raise ValueError("OPENAI_API_KEY is not set")
        self.price = price or prices()
        self.post = post

    def _cost(self, usage: dict | None) -> float | None:
        if not usage or "prompt_tokens" not in usage:
            return None
        return (usage["prompt_tokens"] * self.price[0] + usage.get("completion_tokens", 0) * self.price[1]) / 1e6

    def run(self, spec: AgentSpec) -> AgentReply:
        """Never raises. A response without usage is charged at the session cap."""
        bench = Workbench(spec.workspace, spec.base_files)
        lab = {t.name: t for t in spec.tools}
        builtin = {n: BUILTIN[n] for n in spec.builtin if n in BUILTIN}
        tools = [*builtin.values(), FINISH, *({"type": "function", "function": {
            "name": t.name, "description": t.description, "parameters": t.schema}} for t in spec.tools)]
        messages: list[dict[str, Any]] = [{"role": "system", "content": spec.system},
                                          {"role": "user", "content": spec.prompt}]
        cost, turns, deadline = 0.0, 0, time.monotonic() + spec.timeout_s
        try:
            while turns < spec.max_turns:
                if time.monotonic() > deadline:
                    return AgentReply(None, cost, turns, f"timed out after {spec.timeout_s:.0f}s")
                out = self.post(self.url, self.key, {"model": spec.model, "messages": messages, "tools": tools},
                                max(deadline - time.monotonic(), 1.0))
                turns += 1
                step = self._cost(out.get("usage"))
                if step is None:
                    return AgentReply(None, spec.max_budget_usd, turns, "the model API returned no usage", True)
                cost += step
                msg = out["choices"][0]["message"]
                calls = msg.get("tool_calls") or []
                messages.append({"role": "assistant", "content": msg.get("content"),
                                 **({"tool_calls": calls} if calls else {})})
                if not calls:
                    messages.append({"role": "user", "content": "A session ends only through the `finish` tool."})
                for c in calls:
                    name = c["function"]["name"]
                    try:
                        args = json.loads(c["function"].get("arguments") or "{}")
                    except json.JSONDecodeError as e:
                        args, text = None, f"{name} refused: arguments are not JSON: {e}"
                    if args is not None and name == "finish":
                        if args.get("status") in ("continue", "stop"):
                            return AgentReply({"status": args["status"], "note": args.get("note")}, cost, turns, None)
                        text = "finish refused: status must be `continue` or `stop`"
                    elif args is not None:
                        text = self._call(bench, builtin, lab, name, args)
                    messages.append({"role": "tool", "tool_call_id": c["id"], "content": text[-MAX_TOOL_TEXT:]})
                if cost >= spec.max_budget_usd:
                    return AgentReply(None, cost, turns, f"session budget ${spec.max_budget_usd:.2f} spent")
            return AgentReply(None, cost, turns, f"max turns ({spec.max_turns}) reached")
        except Exception as e:
            return AgentReply(None, cost, turns, f"provider failed: {type(e).__name__}: {e}")

    @staticmethod
    def _call(bench: Workbench, builtin: dict, lab: dict, name: str, args: dict) -> str:
        """A refusal is text the agent sees, never a crash."""
        try:
            if name in lab:
                return lab[name].fn(args)
            if name in builtin:
                return getattr(bench, name)(**args)
            return f"{name} refused: no such tool"
        except Exception as e:
            return f"{name} refused: {e}"
