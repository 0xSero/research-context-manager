"""Configuration for Cortex router."""

from typing import Literal
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class ModelConfig(BaseModel):
    """Configuration for a single model (A or B)."""

    provider: Literal["anthropic", "openai", "ollama", "together", "groq", "custom"] = "anthropic"
    model: str = "claude-sonnet-4-20250514"
    api_key: str | None = None
    base_url: str | None = None
    max_tokens: int = 8192
    temperature: float = 0.7


class Neo4jConfig(BaseModel):
    """Neo4j connection configuration."""

    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "password"
    database: str = "neo4j"


class ContextConfig(BaseModel):
    """Context management configuration."""

    # How much conversation history Model A sees
    model_a_context_window: int = 50000

    # Maximum context to inject into Model B
    model_b_max_context: int = 100000

    # Number of memory nodes to consider
    max_memory_nodes: int = 50

    # Number of tools to show Model A
    max_tools_to_consider: int = 100

    # Compression threshold - when to trigger compaction
    compaction_threshold: int = 40000


class PruningConfig(BaseModel):
    """Background graph pruning configuration."""

    enabled: bool = True
    interval_seconds: int = 60
    max_nodes: int = 10000
    relevance_threshold: float = 0.3
    age_decay_factor: float = 0.95


class CortexConfig(BaseSettings):
    """Main Cortex configuration."""

    # Model configurations
    model_a: ModelConfig = Field(default_factory=ModelConfig)
    model_b: ModelConfig = Field(default_factory=ModelConfig)

    # Neo4j configuration
    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)

    # Context management
    context: ContextConfig = Field(default_factory=ContextConfig)

    # Pruning configuration
    pruning: PruningConfig = Field(default_factory=PruningConfig)

    # System prompt path
    system_prompt_path: str | None = None

    # MCP servers configuration path
    mcp_config_path: str | None = None

    # Debug mode
    debug: bool = False

    class Config:
        env_prefix = "CORTEX_"
        env_nested_delimiter = "__"


# Default configuration for quick start
DEFAULT_CONFIG = CortexConfig(
    model_a=ModelConfig(
        provider="anthropic",
        model="claude-sonnet-4-20250514",
        temperature=0.3,  # Lower temp for context decisions
    ),
    model_b=ModelConfig(
        provider="anthropic",
        model="claude-sonnet-4-20250514",
        temperature=0.7,  # Normal temp for creative work
    ),
)
