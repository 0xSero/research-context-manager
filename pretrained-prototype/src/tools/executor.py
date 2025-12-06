"""Tool executor for running tools."""

import asyncio
import json
import traceback
from typing import Any

from providers.base import ToolCall, ToolResult
from tools.registry import ToolRegistry, get_registry


class ToolExecutor:
    """Executes tools and handles results."""

    def __init__(self, registry: ToolRegistry | None = None):
        self.registry = registry or get_registry()
        self._mcp_client = None  # Will be set if MCP is enabled

    def set_mcp_client(self, client) -> None:
        """Set the MCP client for executing MCP tools."""
        self._mcp_client = client

    async def execute(self, tool_call: ToolCall) -> ToolResult:
        """Execute a single tool call."""
        tool = self.registry.get(tool_call.name)

        if tool is None:
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                content=f"Error: Tool '{tool_call.name}' not found",
                is_error=True,
            )

        try:
            # Record usage
            self.registry.record_usage(tool_call.name)

            # Execute based on source
            if tool.source == "mcp" and self._mcp_client:
                result = await self._execute_mcp_tool(tool, tool_call.arguments)
            elif tool.handler:
                result = await tool.handler(**tool_call.arguments)
            else:
                return ToolResult(
                    tool_call_id=tool_call.id,
                    name=tool_call.name,
                    content=f"Error: Tool '{tool_call.name}' has no handler",
                    is_error=True,
                )

            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                content=result if isinstance(result, str) else json.dumps(result),
                is_error=False,
            )

        except Exception as e:
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                content=f"Error executing {tool_call.name}: {str(e)}\n{traceback.format_exc()}",
                is_error=True,
            )

    async def execute_many(self, tool_calls: list[ToolCall]) -> list[ToolResult]:
        """Execute multiple tool calls concurrently."""
        tasks = [self.execute(tc) for tc in tool_calls]
        return await asyncio.gather(*tasks)

    async def _execute_mcp_tool(
        self,
        tool,
        arguments: dict[str, Any],
    ) -> str:
        """Execute a tool via MCP."""
        if not self._mcp_client:
            raise RuntimeError("MCP client not configured")

        # Call the MCP server
        result = await self._mcp_client.call_tool(
            server_name=tool.mcp_server,
            tool_name=tool.name,
            arguments=arguments,
        )
        return result


# Global executor
_executor: ToolExecutor | None = None


def get_executor() -> ToolExecutor:
    """Get the global tool executor."""
    global _executor
    if _executor is None:
        _executor = ToolExecutor()
    return _executor
