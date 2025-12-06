"""Factory for creating LLM providers."""

from typing import Literal

from cortex.config import ModelConfig
from providers.base import LLMProvider
from providers.anthropic_provider import AnthropicProvider
from providers.openai_provider import OpenAIProvider


# OpenAI-compatible providers (use OpenAI SDK with different base_url)
OPENAI_COMPATIBLE_PROVIDERS = {
    "ollama": "http://localhost:11434/v1",
    "together": "https://api.together.xyz/v1",
    "groq": "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "fireworks": "https://api.fireworks.ai/inference/v1",
}


def create_provider(config: ModelConfig) -> LLMProvider:
    """Create an LLM provider from config."""
    provider_type = config.provider

    if provider_type == "anthropic":
        return AnthropicProvider(
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
        )

    if provider_type == "openai":
        return OpenAIProvider(
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
        )

    if provider_type in OPENAI_COMPATIBLE_PROVIDERS:
        base_url = config.base_url or OPENAI_COMPATIBLE_PROVIDERS[provider_type]
        return OpenAIProvider(
            model=config.model,
            api_key=config.api_key,
            base_url=base_url,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
        )

    if provider_type == "custom":
        if not config.base_url:
            raise ValueError("Custom provider requires base_url")
        # Assume OpenAI-compatible for custom
        return OpenAIProvider(
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
        )

    raise ValueError(f"Unknown provider type: {provider_type}")


def list_providers() -> list[str]:
    """List available provider types."""
    return ["anthropic", "openai"] + list(OPENAI_COMPATIBLE_PROVIDERS.keys()) + ["custom"]
