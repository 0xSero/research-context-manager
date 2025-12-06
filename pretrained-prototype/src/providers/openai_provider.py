"""OpenAI provider implementation."""

import json
import os
from typing import AsyncIterator

import tiktoken
from openai import AsyncOpenAI

from providers.base import (
    LLMProvider,
    LLMResponse,
    Message,
    ToolCall,
    ToolDefinition,
)


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 8192,
        temperature: float = 0.7,
    ):
        super().__init__(model, api_key, base_url, max_tokens, temperature)
        self.client = AsyncOpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url,
        )
        # Try to get tokenizer for this model
        try:
            self.tokenizer = tiktoken.encoding_for_model(model)
        except KeyError:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def _convert_messages(self, messages: list[Message]) -> list[dict]:
        """Convert messages to OpenAI format."""
        converted = []
        for msg in messages:
            d = {"role": msg.role, "content": msg.content}

            if msg.role == "assistant" and msg.tool_calls:
                d["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in msg.tool_calls
                ]

            if msg.role == "tool":
                d["tool_call_id"] = msg.tool_call_id
                d["name"] = msg.name

            converted.append(d)

        return converted

    def _convert_tools(self, tools: list[ToolDefinition]) -> list[dict]:
        """Convert tools to OpenAI format."""
        return [tool.to_dict() for tool in tools]

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        system: str | None = None,
    ) -> LLMResponse:
        """Generate a completion."""
        converted_messages = self._convert_messages(messages)

        # Add system message at the beginning if provided
        if system and (not converted_messages or converted_messages[0]["role"] != "system"):
            converted_messages.insert(0, {"role": "system", "content": system})

        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": converted_messages,
        }

        if tools:
            kwargs["tools"] = self._convert_tools(tools)

        response = await self.client.chat.completions.create(**kwargs)

        message = response.choices[0].message

        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments),
                    )
                )

        return LLMResponse(
            content=message.content or "",
            tool_calls=tool_calls,
            finish_reason=response.choices[0].finish_reason or "stop",
            usage={
                "input_tokens": response.usage.prompt_tokens if response.usage else 0,
                "output_tokens": response.usage.completion_tokens if response.usage else 0,
            },
        )

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        system: str | None = None,
    ) -> AsyncIterator[str]:
        """Stream a completion."""
        converted_messages = self._convert_messages(messages)

        if system and (not converted_messages or converted_messages[0]["role"] != "system"):
            converted_messages.insert(0, {"role": "system", "content": system})

        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": converted_messages,
            "stream": True,
        }

        if tools:
            kwargs["tools"] = self._convert_tools(tools)

        stream = await self.client.chat.completions.create(**kwargs)

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenizer.encode(text))
