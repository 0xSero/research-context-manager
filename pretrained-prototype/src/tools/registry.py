"""Tool registry for managing available tools."""

from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable
import json

from providers.base import ToolDefinition


@dataclass
class Tool:
    """A registered tool."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    handler: Callable[..., Awaitable[str]] | None = None
    category: str = "general"
    source: str = "local"  # local, mcp, plugin
    mcp_server: str | None = None  # MCP server name if from MCP
    examples: list[dict[str, Any]] = field(default_factory=list)
    usage_count: int = 0
    last_used: str | None = None

    def to_definition(self) -> ToolDefinition:
        """Convert to provider-agnostic ToolDefinition."""
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )

    def to_summary(self) -> str:
        """Get a short summary for Model A context."""
        params_str = ", ".join(self.parameters.get("properties", {}).keys())
        return f"{self.name}({params_str}): {self.description[:100]}"


class ToolRegistry:
    """Registry of all available tools."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}
        self._categories: dict[str, list[str]] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool

        # Update category index
        if tool.category not in self._categories:
            self._categories[tool.category] = []
        if tool.name not in self._categories[tool.category]:
            self._categories[tool.category].append(tool.name)

    def unregister(self, name: str) -> None:
        """Unregister a tool."""
        if name in self._tools:
            tool = self._tools[name]
            if tool.category in self._categories:
                self._categories[tool.category].remove(name)
            del self._tools[name]

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def get_all(self) -> list[Tool]:
        """Get all registered tools."""
        return list(self._tools.values())

    def get_by_category(self, category: str) -> list[Tool]:
        """Get tools by category."""
        names = self._categories.get(category, [])
        return [self._tools[n] for n in names if n in self._tools]

    def get_by_names(self, names: list[str]) -> list[Tool]:
        """Get specific tools by name."""
        return [self._tools[n] for n in names if n in self._tools]

    def get_definitions(self, names: list[str] | None = None) -> list[ToolDefinition]:
        """Get ToolDefinitions for specified tools (or all if None)."""
        if names is None:
            tools = self.get_all()
        else:
            tools = self.get_by_names(names)
        return [t.to_definition() for t in tools]

    def get_summary_list(self) -> str:
        """Get a summary of all tools for Model A context."""
        lines = []
        for category, names in sorted(self._categories.items()):
            lines.append(f"\n## {category.upper()}")
            for name in names:
                tool = self._tools.get(name)
                if tool:
                    lines.append(f"  - {tool.to_summary()}")
        return "\n".join(lines)

    def search(self, query: str) -> list[Tool]:
        """Search tools by name or description."""
        query_lower = query.lower()
        matches = []
        for tool in self._tools.values():
            if (
                query_lower in tool.name.lower()
                or query_lower in tool.description.lower()
            ):
                matches.append(tool)
        return matches

    def record_usage(self, name: str) -> None:
        """Record that a tool was used."""
        if name in self._tools:
            from datetime import datetime

            self._tools[name].usage_count += 1
            self._tools[name].last_used = datetime.utcnow().isoformat()

    def get_stats(self) -> dict[str, Any]:
        """Get registry statistics."""
        return {
            "total_tools": len(self._tools),
            "categories": list(self._categories.keys()),
            "tools_per_category": {
                cat: len(names) for cat, names in self._categories.items()
            },
            "most_used": sorted(
                self._tools.values(),
                key=lambda t: t.usage_count,
                reverse=True,
            )[:5],
        }

    def export_for_context(self, tool_names: list[str]) -> str:
        """Export tool definitions as context for Model B."""
        tools = self.get_by_names(tool_names)
        if not tools:
            return "No tools selected."

        output = ["# Available Tools\n"]
        for tool in tools:
            output.append(f"## {tool.name}")
            output.append(f"{tool.description}\n")
            output.append("**Parameters:**")
            output.append(f"```json\n{json.dumps(tool.parameters, indent=2)}\n```")
            if tool.examples:
                output.append("\n**Examples:**")
                for ex in tool.examples[:2]:  # Limit examples
                    output.append(f"```json\n{json.dumps(ex, indent=2)}\n```")
            output.append("")

        return "\n".join(output)


# Global registry instance
_registry: ToolRegistry | None = None


def get_registry() -> ToolRegistry:
    """Get the global tool registry."""
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry


def register_tool(
    name: str,
    description: str,
    parameters: dict[str, Any],
    handler: Callable[..., Awaitable[str]] | None = None,
    category: str = "general",
    examples: list[dict[str, Any]] | None = None,
) -> Tool:
    """Register a tool in the global registry."""
    tool = Tool(
        name=name,
        description=description,
        parameters=parameters,
        handler=handler,
        category=category,
        source="local",
        examples=examples or [],
    )
    get_registry().register(tool)
    return tool
