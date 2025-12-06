"""Tests for configuration."""

import pytest
from cortex.config import CortexConfig, ModelConfig, Neo4jConfig


def test_default_config():
    """Test default configuration."""
    config = CortexConfig()
    assert config.model_a.provider == "anthropic"
    assert config.model_b.provider == "anthropic"
    assert config.debug is False


def test_custom_config():
    """Test custom configuration."""
    config = CortexConfig(
        model_a=ModelConfig(provider="openai", model="gpt-4o"),
        model_b=ModelConfig(provider="anthropic", model="claude-sonnet-4-20250514"),
        debug=True,
    )
    assert config.model_a.provider == "openai"
    assert config.model_a.model == "gpt-4o"
    assert config.model_b.provider == "anthropic"
    assert config.debug is True


def test_neo4j_config():
    """Test Neo4j configuration."""
    config = CortexConfig(
        neo4j=Neo4jConfig(
            uri="bolt://custom:7687",
            user="admin",
            password="secret",
        )
    )
    assert config.neo4j.uri == "bolt://custom:7687"
    assert config.neo4j.user == "admin"
