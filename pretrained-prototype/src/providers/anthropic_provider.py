"""Anthropic (Claude) provider implementation."""

import os
from typing import AsyncIterator

from anthropic import AsyncAnthropic

from providers.base import (
    LLMProvider,
    LLMResponse,
    Message,
    ToolCall,
    ToolDefinition,
)


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 8192,
        temperature: float = 0.7,
    ):
        super().__init__(model, api_key, base_url, max_tokens, temperature)
        self.client = AsyncAnthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
            base_url=base_url,
        )

    def _convert_messages(self, messages: list[Message]) -> list[dict]:
        """Convert messages to Anthropic format."""
        converted = []
        for msg in messages:
            if msg.role == "system":
                continue  # System handled separately in Anthropic

            content = msg.content
            if msg.role == "assistant" and msg.tool_calls:
                # Anthropic uses content blocks for tool use
                content = []
                if msg.content:
                    content.append({"type": "text", "text": msg.content})
                for tc in msg.tool_calls:
                    content.append({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.arguments,
                    })

            if msg.role == "tool":
                converted.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.tool_call_id,
                        "content": msg.content,
                    }],
                })
            else:
                converted.append({
                    "role": msg.role,
                    "content": content,
                })

        return converted

    def _convert_tools(self, tools: list[ToolDefinition]) -> list[dict]:
        """Convert tools to Anthropic format."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters,
            }
            for tool in tools
        ]

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        system: str | None = None,
    ) -> LLMResponse:
        """Generate a completion."""
        # Extract system message if not provided
        if system is None:
            for msg in messages:
                if msg.role == "system":
                    system = msg.content
                    break

        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": self._convert_messages(messages),
        }

        if system:
            kwargs["system"] = system

        if tools:
            kwargs["tools"] = self._convert_tools(tools)

        response = await self.client.messages.create(**kwargs)

        # Parse response
        content = ""
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=block.input)
                )

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=response.stop_reason or "stop",
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        )

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        system: str | None = None,
    ) -> AsyncIterator[str]:
        """Stream a completion."""
        if system is None:
            for msg in messages:
                if msg.role == "system":
                    system = msg.content
                    break

        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": self._convert_messages(messages),
        }

        if system:
            kwargs["system"] = system

        if tools:
            kwargs["tools"] = self._convert_tools(tools)

        async with self.client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield text

    def count_tokens(self, text: str) -> int:
        """Count tokens in text (approximate for Anthropic)."""
        # Anthropic doesn't expose tokenizer, use approximation
        # ~4 chars per token is a reasonable estimate for Claude
        return len(text) // 4
