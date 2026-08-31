from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from seniorcare_agents.observability import flow_event


class MCPToolError(RuntimeError):
    pass


class MCPToolGateway:
    """Discover and cache LangChain tools from the remote Streamable HTTP MCP server."""

    def __init__(
        self, server_url: str, read_max_attempts: int = 3, retry_base_seconds: float = 0.25
    ):
        self.server_url = server_url.rstrip("/")
        self.read_max_attempts = max(1, read_max_attempts)
        self.retry_base_seconds = max(0.0, retry_base_seconds)
        self.client = MultiServerMCPClient(
            {"seniorcare": {"url": self.server_url, "transport": "streamable_http"}}
        )
        self._tools: dict[str, BaseTool] | None = None
        self._discovery_lock = asyncio.Lock()

    async def get_tools(self, allowed: set[str] | frozenset[str] | None = None) -> list[BaseTool]:
        if self._tools is None:
            async with self._discovery_lock:
                if self._tools is None:
                    try:
                        discovered = await self.client.get_tools(server_name="seniorcare")
                    except Exception as exc:
                        raise MCPToolError(
                            f"Unable to discover MCP tools at {self.server_url}"
                        ) from exc
                    self._tools = {tool.name: tool for tool in discovered}
        if allowed is None:
            return list(self._tools.values())
        missing = allowed.difference(self._tools)
        if missing:
            raise MCPToolError(f"MCP server is missing required tools: {sorted(missing)}")
        return [self._tools[name] for name in sorted(allowed)]

    async def call(self, tool_name: str, *, _retry_safe: bool = False, **arguments: Any) -> Any:
        tool = (await self.get_tools({tool_name}))[0]
        started = time.perf_counter()
        flow_event("mcp_tool", tool_name, "input", arguments, selected_tool=tool_name)
        attempts = self.read_max_attempts if _retry_safe else 1
        for attempt in range(1, attempts + 1):
            try:
                result = await tool.ainvoke(arguments)
                break
            except Exception as exc:
                flow_event(
                    "mcp_tool",
                    tool_name,
                    "error",
                    {"attempt": attempt, "maxAttempts": attempts, "error": str(exc)},
                    selected_tool=tool_name,
                    duration_ms=(time.perf_counter() - started) * 1000,
                )
                if attempt == attempts:
                    raise MCPToolError(
                        f"MCP tool {tool_name!r} failed after {attempts} attempt(s)"
                    ) from exc
                await asyncio.sleep(self.retry_base_seconds * (2 ** (attempt - 1)))
        decoded = self._decode_text_blocks(result)
        flow_event(
            "mcp_tool",
            tool_name,
            "output",
            decoded,
            selected_tool=tool_name,
            duration_ms=(time.perf_counter() - started) * 1000,
        )
        return decoded

    @staticmethod
    def _decode_text_blocks(result: Any) -> Any:
        """Decode JSON carried in one or more LangChain MCP text blocks."""
        if not isinstance(result, list) or not result:
            return result
        if not all(
            isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
            for block in result
        ):
            return result

        decoded: list[Any] = []
        try:
            for block in result:
                decoded.append(json.loads(block["text"]))
        except (KeyError, TypeError, json.JSONDecodeError):
            return result

        # Preserve the original JSON type. A one-record list must remain a
        # list because list_* MCP tools rely on that stable return contract.
        if len(decoded) == 1:
            return decoded[0]
        combined: list[Any] = []
        for value in decoded:
            combined.extend(value if isinstance(value, list) else [value])
        return combined

    async def list_tool_names(self) -> list[str]:
        return sorted(tool.name for tool in await self.get_tools())

    def clear_cache(self) -> None:
        self._tools = None
