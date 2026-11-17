"""The one place a model is called: a Provider runs one session with the lab's tools."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol
from urllib.parse import urlsplit

from lab import apiproxy
from lab import live
from lab.targets import engine
from lab.infra.sandbox import container, jail
from lab.infra.sandbox.grader import HF_HUB
from lab.hooks import WRITE_TOOLS, write_guard

BUILTIN_TOOLS = ["Read", "Grep", "Glob", "Edit", "Write", "Bash", "WebSearch", "WebFetch"]
PASS_ENV = ("CLAUDE_CODE_ENTRYPOINT", "CLAUDE_AGENT_SDK_VERSION")       # what the SDK sets for its CLI
# Output tokens the CLI may ask for per request.
MAX_OUTPUT_TOKENS = 16_000
SYSTEM_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
# What a session returns when it ends; `stop` ends the run.
OUTPUT_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["status", "note"],
                 "properties": {"status": {"type": "string", "enum": ["continue", "stop"]},
                                "note": {"type": ["string", "null"]}}}


@dataclass(frozen=True)
class ToolSpec:
    """A lab tool: runs in the harness process, outside the jail."""
    name: str
    description: str
    schema: dict[str, Any]
    fn: Callable[[dict[str, Any]], str]    # returns the text the agent sees


@dataclass
class AgentSpec:
    system: str
    prompt: str
    workspace: Path
    scratch: Path                           # outside the workspace: the CLI's home, jail settings
    tools: list[ToolSpec]
    base_files: frozenset[str] = frozenset()   # for the write hook: new means not in the base tree
    builtin: list[str] = field(default_factory=lambda: list(BUILTIN_TOOLS))
    max_turns: int = 200
    max_budget_usd: float = 5.0
    timeout_s: float = 3600.0
    model: str = field(default_factory=lambda: os.environ.get("LAB_MODEL", "claude-fable-5-1"))
    workbench: str | None = None            # the container name of the agent's workbench (Linux)
    remote: Any = None                      # a lab.infra.runner.worker.Remote: the workbench runs on that host
    remote_workspace: str | None = None     # the workspace's path there


@dataclass
class AgentReply:
    output: dict[str, Any] | None
    cost_usd: float
    turns: int
    error: str | None
    cost_estimated: bool = False     # no accounting came back; charged at the session cap
    usage: dict | None = None        # tokens as the API proxy counted them


class Provider(Protocol):
    name: str

    def run(self, spec: AgentSpec) -> AgentReply: ...


def load(name: str | None = None) -> Provider:
    name = name or os.environ.get("LAB_AGENT_PROVIDER", "claude")
    if name == "claude":
        return ClaudeAgentSDK()
    if name == "openai":
        from lab.agent_openai import OpenAICompatible
        return OpenAICompatible()
    raise ValueError(f"unknown agent provider {name!r}; LAB_AGENT_PROVIDER is claude or openai")


def missing(name: str | None = None) -> list[str]:
    """What the provider needs that this machine lacks, one line each."""
    name = name or os.environ.get("LAB_AGENT_PROVIDER", "claude")
    if name != "claude":
        try:
            load(name)
        except ValueError as e:
            return [str(e)]
        return []
    out = []
    if importlib.util.find_spec("claude_agent_sdk") is None:
        out.append("the Claude Agent SDK is not installed: pip install -e '.[agents]'")
    try:
        apiproxy.credential()
    except RuntimeError as e:
        out.append(str(e))
    return out


PROXY_PORT_IN = 4000              # the tunnelled proxy's port inside a remote workbench


def _scratch(scratch: Path) -> tuple[Path, Path]:
    home, tmp = scratch / "home", scratch / "tmp"
    home.mkdir(parents=True, exist_ok=True)
    tmp.mkdir(exist_ok=True)
    return home, tmp


def _weights() -> list[Path]:
    return [HF_HUB] if HF_HUB.exists() else []      # model weights, read-only


def cli_env(home: Path, tmp: Path, ws: Path, base_url: str) -> dict[str, str]:
    """The CLI's whole environment inside its jail: the proxy's URL and a placeholder key, no credential."""
    return {"HOME": str(home), "CLAUDE_CONFIG_DIR": str(home), "TMPDIR": str(tmp), "PYTHONPATH": str(ws / "src"),
            "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_CACHE": str(HF_HUB), "HF_HUB_OFFLINE": "1",
            "HF_HOME": str(tmp / "hf-home"), "ANTHROPIC_BASE_URL": base_url,
            "ANTHROPIC_API_KEY": apiproxy.PLACEHOLDER, "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(MAX_OUTPUT_TOKENS)}


def workbench_command(name: str, ws: Path, scratch: Path, cli: Path, base_url: str | None = None,
                      sock_dir: Path | None = None) -> list[str]:
    """`docker run` argv for the CLI's workbench on this host.
    With `sock_dir`, the proxy is the unix socket `proxy.sock` there, served inside on 127.0.0.1:PROXY_PORT_IN."""
    home, tmp = _scratch(scratch)
    ws = Path(ws).resolve()
    env = cli_env(home, tmp, ws, base_url or f"http://127.0.0.1:{PROXY_PORT_IN}")
    env["PATH"] = f"{Path(engine.python()).parent}:{SYSTEM_PATH}"      # the agent's `python` is the engine's
    command = [str(cli)] if sock_dir is None else [
        "sh", "-c", f'socat TCP-LISTEN:{PROXY_PORT_IN},bind=127.0.0.1,fork,reuseaddr UNIX-CONNECT:{sock_dir}/proxy.sock '
        '& exec "$0" "$@"', str(cli)]
    writable = [home.resolve(), tmp.resolve(), ws] + ([Path(sock_dir)] if sock_dir else [])
    return container.workbench_argv(name, ws, command, env, writable, [engine.venv(), *_weights(), cli.parent],
                                    engine.python())


def _live(msg, spent_usd: float) -> None:
    """Send what the agent just said or called to the dashboard."""
    from claude_agent_sdk import AssistantMessage, TextBlock, ThinkingBlock, ToolResultBlock, ToolUseBlock, UserMessage
    if isinstance(msg, AssistantMessage):
        for b in msg.content:
            if isinstance(b, ToolUseBlock):
                live.emit("agent", kind="tool", name=b.name.removeprefix("mcp__lab__"),
                          detail=json.dumps(b.input)[:300], spent_usd=spent_usd)
            elif isinstance(b, (TextBlock, ThinkingBlock)):
                live.emit("agent", kind="text", detail=(b.text if isinstance(b, TextBlock) else b.thinking)[:600],
                          spent_usd=spent_usd)
    elif isinstance(msg, UserMessage) and isinstance(msg.content, list) and any(
            isinstance(b, ToolResultBlock) for b in msg.content):
        live.emit("agent", kind="result", spent_usd=spent_usd)


class ClaudeAgentSDK:
    """Claude through the Agent SDK CLI, jailed; model calls go through the lab's API proxy, lab tools run here."""

    name = "claude"
    remote = True                       # can work on a Remote worker's host (its CLI over ssh)

    @staticmethod
    def containerized() -> bool:
        return jail.backend() == "container" and container.available()

    def _wrapper(self, spec: AgentSpec, cli: Path, base_url: str) -> Path:
        script = spec.scratch / "cli.sh"
        if spec.remote is not None:            # the worker host builds its own workbench
            import claude_agent_sdk
            w = spec.remote.workbench(spec.workbench, spec.scratch.name, spec.remote_workspace,
                                      claude_agent_sdk.__version__)
            *ssh, target = spec.remote.prefix
            # remove a socket left by an earlier launch; -n: do not read the SDK's stdin
            line = (f'args=$(printf " %q" "$@")\ncmd={shlex.quote(shlex.join(w["argv"]))}\n'
                    f'{shlex.join(ssh)} -n -T {target} {shlex.quote("rm -f " + shlex.quote(w["sock"]))} || exit 1\n'
                    f'exec {shlex.join(ssh)} -T -o ExitOnForwardFailure=yes -o StreamLocalBindUnlink=yes '
                    f'-R {w["sock"]}:127.0.0.1:{urlsplit(base_url).port} {target} "$cmd$args"')
        elif self.containerized():
            argv = workbench_command(spec.workbench, spec.workspace, spec.scratch, cli, base_url=base_url)
            at = argv.index(container.IMAGE)
            argv[at:at] = [w for k in PASS_ENV for w in ("--env", k)]     # the SDK's values, passed through
            line = f'exec {shlex.join(argv)} "$@"'
        else:
            home, tmp = _scratch(spec.scratch)
            ws = spec.workspace.resolve()
            tmp_root = "/private/tmp" if sys.platform == "darwin" else "/tmp"
            cli_tmp = Path(f"{tmp_root}/claude-{os.getuid()}")    # the CLI's own per-project scratch
            cli_tmp.mkdir(parents=True, exist_ok=True)
            config = jail.settings([home.resolve(), tmp.resolve(), ws, cli_tmp.resolve()], engine.venv(),
                                   [base_url.removeprefix("http://")], readonly=_weights(), python=engine.python(),
                                   local_binding=True)     # lets srt on macOS reach 127.0.0.1
            argv = jail.wrap(config, spec.scratch / "srt.json", [str(cli)])
            keep = " ".join(f'"{k}=${k}"' for k in PASS_ENV)
            sets = " ".join(f"{k}={shlex.quote(v)}" for k, v in cli_env(home, tmp, ws, base_url).items())
            line = f'exec env -i {keep} "PATH={Path(engine.python()).parent}:$PATH" {sets} {shlex.join(argv)} "$@"'
        script.write_text(f"#!/bin/bash\n{line}\n")
        script.chmod(0o700)
        return script

    async def _run(self, spec: AgentSpec, proxy: apiproxy.Proxy) -> AgentReply:
        import claude_agent_sdk
        from claude_agent_sdk import (ClaudeAgentOptions, HookMatcher, ResultMessage,
                                      create_sdk_mcp_server, query, tool)
        base_url = proxy.url

        # The workbench is paused during every lab tool call.
        if spec.remote is not None:
            frozen = lambda: spec.remote.paused(spec.workbench)                                  # noqa: E731
        else:
            frozen = lambda: container.paused(spec.workbench if self.containerized() else None)  # noqa: E731

        def adapt(t: ToolSpec):
            async def call(args: dict[str, Any]) -> dict[str, Any]:
                try:
                    def run():
                        with frozen():
                            return t.fn(args)
                    text = await asyncio.to_thread(run)          # off the event loop
                except Exception as e:          # the agent sees the refusal
                    text = f"{t.name} refused: {e}"
                return {"content": [{"type": "text", "text": text}]}
            return tool(t.name, t.description, t.schema)(call)

        server = create_sdk_mcp_server("lab", tools=[adapt(t) for t in spec.tools])
        cli = Path(claude_agent_sdk.__file__).parent / "_bundled" / "claude"
        options = ClaudeAgentOptions(
            model=spec.model, system_prompt=spec.system,
            tools=spec.builtin, allowed_tools=spec.builtin + [f"mcp__lab__{t.name}" for t in spec.tools],
            mcp_servers={"lab": server}, permission_mode="dontAsk", setting_sources=[],
            cwd=str(spec.workspace), max_turns=spec.max_turns, max_budget_usd=spec.max_budget_usd,
            output_format={"type": "json_schema", "schema": OUTPUT_SCHEMA},
            cli_path=str(self._wrapper(spec, cli, base_url)),
            hooks={"PreToolUse": [HookMatcher(matcher=WRITE_TOOLS,
                                              hooks=[write_guard(Path(spec.remote_workspace or spec.workspace),
                                                                 spec.base_files)])]},
        )
        reply = AgentReply(None, 0.0, 0, "no result message")
        async for msg in query(prompt=spec.prompt, options=options):
            _live(msg, proxy.spent_usd)
            if isinstance(msg, ResultMessage):
                reply = AgentReply(msg.structured_output if not msg.is_error else None,
                                   float(msg.total_cost_usd or 0.0), msg.num_turns,
                                   None if not msg.is_error else msg.subtype)
        return reply

    def run(self, spec: AgentSpec) -> AgentReply:
        """Run one session; never raises. The spend is what the API proxy priced."""
        spec.scratch.mkdir(parents=True, exist_ok=True)
        try:
            host = container.bridge_gateway() if spec.remote is None and self.containerized() else "127.0.0.1"
            proxy = apiproxy.Proxy(spec.max_budget_usd, host)
        except Exception as e:                      # nothing was sent: nothing spent
            return AgentReply(None, 0.0, 0, f"provider failed: {type(e).__name__}: {e}")
        with proxy:
            try:
                reply = asyncio.run(asyncio.wait_for(self._run(spec, proxy), spec.timeout_s))
            except TimeoutError:
                reply = AgentReply(None, 0.0, 0, f"timed out after {spec.timeout_s:.0f}s")
            except Exception as e:
                reply = AgentReply(None, 0.0, 0, f"provider failed: {type(e).__name__}: {e}")
            finally:                                    # remove the CLI's container
                if spec.remote is not None:
                    spec.remote.remove_workbench(spec.workbench)
                elif spec.workbench and self.containerized():
                    container.remove(spec.workbench)
        reply.cost_usd, reply.cost_estimated, reply.usage = proxy.spent_usd, False, dict(proxy.usage)
        return reply
