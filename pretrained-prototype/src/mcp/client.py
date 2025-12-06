"""MCP Client for connecting to MCP servers."""

import asyncio
import json
import subprocess
from dataclasses import dataclass, field
from typing import Any
from pathlib import Path

from tools.registry import Tool, ToolRegistry


@dataclass
class MCPServer:
    """Configuration for an MCP server."""

    name: str
    command: str  # Command to start the server
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    enabled: bool = True


@dataclass
class MCPConnection:
    """Active connection to an MCP server."""

    server: MCPServer
    process: subprocess.Popen | None = None
    tools: list[Tool] = field(default_factory=list)


class MCPClient:
    """Client for managing MCP server connections."""

    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self._connections: dict[str, MCPConnection] = {}
        self._servers: dict[str, MCPServer] = {}

    def load_config(self, config_path: str | Path) -> None:
        """Load MCP configuration from JSON file."""
        path = Path(config_path)
        if not path.exists():
            return

        with open(path) as f:
            config = json.load(f)

        for name, server_config in config.get("mcpServers", {}).items():
            server = MCPServer(
                name=name,
                command=server_config.get("command", ""),
                args=server_config.get("args", []),
                env=server_config.get("env", {}),
                enabled=server_config.get("enabled", True),
            )
            self._servers[name] = server

    async def connect_all(self) -> None:
        """Connect to all configured MCP servers."""
        for name, server in self._servers.items():
            if server.enabled:
                await self.connect(name)

    async def connect(self, server_name: str) -> bool:
        """Connect to a specific MCP server."""
        if server_name not in self._servers:
            return False

        server = self._servers[server_name]

        try:
            # Start the MCP server process
            # Note: This is a simplified implementation
            # Real MCP uses stdio JSON-RPC communication
            process = subprocess.Popen(
                [server.command] + server.args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env={**dict(subprocess.os.environ), **server.env},
            )

            connection = MCPConnection(server=server, process=process)
            self._connections[server_name] = connection

            # Get available tools from the server
            tools = await self._list_tools(server_name)
            connection.tools = tools

            # Register tools with the registry
            for tool in tools:
                self.registry.register(tool)

            return True

        except Exception as e:
            print(f"Failed to connect to MCP server {server_name}: {e}")
            return False

    async def disconnect(self, server_name: str) -> None:
        """Disconnect from an MCP server."""
        if server_name not in self._connections:
            return

        conn = self._connections[server_name]

        # Unregister tools
        for tool in conn.tools:
            self.registry.unregister(tool.name)

        # Kill process
        if conn.process:
            conn.process.terminate()
            try:
                conn.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                conn.process.kill()

        del self._connections[server_name]

    async def disconnect_all(self) -> None:
        """Disconnect from all MCP servers."""
        for name in list(self._connections.keys()):
            await self.disconnect(name)

    async def _list_tools(self, server_name: str) -> list[Tool]:
        """List tools available from an MCP server."""
        # This is a simplified implementation
        # Real MCP would use JSON-RPC to query the server

        # For now, return a placeholder
        # In production, this would:
        # 1. Send {"jsonrpc": "2.0", "method": "tools/list", "id": 1}
        # 2. Parse response and create Tool objects

        return []

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> str:
        """Call a tool on an MCP server."""
        if server_name not in self._connections:
            raise RuntimeError(f"Not connected to MCP server: {server_name}")

        conn = self._connections[server_name]

        # Send JSON-RPC request
        request = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments,
            },
            "id": 1,
        }

        # Write to stdin
        if conn.process and conn.process.stdin:
            conn.process.stdin.write((json.dumps(request) + "\n").encode())
            conn.process.stdin.flush()

            # Read response from stdout
            if conn.process.stdout:
                response_line = conn.process.stdout.readline()
                response = json.loads(response_line)

                if "result" in response:
                    return json.dumps(response["result"])
                elif "error" in response:
                    raise RuntimeError(f"MCP error: {response['error']}")

        raise RuntimeError("Failed to communicate with MCP server")

    def get_server_tools(self, server_name: str) -> list[Tool]:
        """Get tools for a specific server."""
        if server_name in self._connections:
            return self._connections[server_name].tools
        return []

    def get_all_mcp_tools(self) -> list[Tool]:
        """Get all tools from all connected MCP servers."""
        tools = []
        for conn in self._connections.values():
            tools.extend(conn.tools)
        return tools

    def list_servers(self) -> list[str]:
        """List configured server names."""
        return list(self._servers.keys())

    def list_connected_servers(self) -> list[str]:
        """List connected server names."""
        return list(self._connections.keys())
