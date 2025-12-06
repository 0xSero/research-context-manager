# Cortex Pretrained Prototype

A working prototype of the Cortex auxiliary context manager using pretrained models. This validates the architecture before training a custom 1.5B model.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INPUT                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MODEL A (Context Manager)                     │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Sees:                                                       │ │
│  │   • All available tools (MCP servers, local tools)          │ │
│  │   • Last 50k tokens of conversation                         │ │
│  │   • Neo4j memory graph                                      │ │
│  │   • System prompt                                           │ │
│  │                                                             │ │
│  │ Decides:                                                    │ │
│  │   • Which tools Model B needs                               │ │
│  │   • What memories to inject                                 │ │
│  │   • Additional context/instructions                         │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                    (Curated Context)
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     MODEL B (Worker)                             │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Sees ONLY:                                                  │ │
│  │   • Selected tools (not all!)                               │ │
│  │   • Relevant memories                                       │ │
│  │   • Curated conversation context                            │ │
│  │   • User's current message                                  │ │
│  │                                                             │ │
│  │ Does:                                                       │ │
│  │   • Actual work (coding, reasoning, tool use)               │ │
│  │   • Responds directly to user                               │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        USER OUTPUT                               │
└─────────────────────────────────────────────────────────────────┘

      ┌──────────────────────────────────────────────────────┐
      │            BACKGROUND: Graph Pruning                  │
      │  Model A continuously prunes the memory graph:        │
      │    • Decay relevance scores over time                 │
      │    • Remove low-relevance nodes                       │
      │    • Maintain graph size limits                       │
      └──────────────────────────────────────────────────────┘
```

## Key Benefits

1. **Model B never sees all tools** - Only what it needs, keeping context clean
2. **Memory is managed externally** - Neo4j graph, not in context
3. **Automatic compaction** - Conversation state can be compressed and continued
4. **Provider agnostic** - Use any LLM for Model A or Model B

## Installation

```bash
# Clone and enter the directory
cd pretrained-prototype

# Install dependencies
pip install -e .

# Or with dev dependencies
pip install -e ".[dev]"
```

## Requirements

- Python 3.11+
- Neo4j (optional, for memory features)
- API keys for your chosen providers

## Quick Start

### 1. Set API Keys

```bash
export ANTHROPIC_API_KEY="your-key"
# or
export OPENAI_API_KEY="your-key"
```

### 2. Start Neo4j (Optional)

```bash
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:5
```

### 3. Run Cortex

```bash
# Simple mode (no config file)
python -m cortex.cli

# With configuration
python -m cortex.cli --config config/example.json

# With different models
python -m cortex.cli \
  --provider-a anthropic \
  --model-a claude-sonnet-4-20250514 \
  --provider-b openai \
  --model-b gpt-4o

# With debug output
python -m cortex.cli --debug
```

## Configuration

### Model Configuration

Both Model A and Model B support these providers:
- `anthropic` - Claude models
- `openai` - GPT models
- `ollama` - Local models via Ollama
- `together` - Together AI
- `groq` - Groq
- `openrouter` - OpenRouter
- `custom` - Any OpenAI-compatible API

### Example Configurations

**Fast & Cheap (Local)**:
```json
{
  "model_a": {
    "provider": "ollama",
    "model": "llama3.2:3b"
  },
  "model_b": {
    "provider": "ollama",
    "model": "llama3.2:8b"
  }
}
```

**Balanced (Cloud)**:
```json
{
  "model_a": {
    "provider": "anthropic",
    "model": "claude-3-5-haiku-20241022"
  },
  "model_b": {
    "provider": "anthropic",
    "model": "claude-sonnet-4-20250514"
  }
}
```

**Maximum Quality**:
```json
{
  "model_a": {
    "provider": "anthropic",
    "model": "claude-sonnet-4-20250514"
  },
  "model_b": {
    "provider": "anthropic",
    "model": "claude-opus-4-20250514"
  }
}
```

## MCP Server Support

Configure MCP servers in `config/mcp.json`:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/home"],
      "enabled": true
    }
  }
}
```

## Commands

In the interactive CLI:
- `/stats` - Show statistics
- `/compact` - Compact conversation history
- `/help` - Show help
- `exit` - Quit

## Project Structure

```
pretrained-prototype/
├── src/
│   ├── cortex/
│   │   ├── __init__.py
│   │   ├── config.py      # Configuration classes
│   │   ├── router.py      # Main router (Model A/B orchestration)
│   │   └── cli.py         # CLI interface
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── base.py        # Base provider interface
│   │   ├── anthropic_provider.py
│   │   ├── openai_provider.py
│   │   └── factory.py     # Provider factory
│   ├── memory/
│   │   ├── __init__.py
│   │   └── graph.py       # Neo4j memory graph
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── registry.py    # Tool registry
│   │   └── executor.py    # Tool execution
│   └── mcp/
│       ├── __init__.py
│       └── client.py      # MCP client
├── config/
│   ├── example.json       # Example configuration
│   ├── mcp.json          # MCP server config
│   └── system_prompt.md  # Default system prompt
├── tests/
└── pyproject.toml
```

## How It Works

### 1. User sends a message

### 2. Model A analyzes the message
- Sees all available tools
- Sees recent conversation history (50k tokens)
- Queries the memory graph
- Decides what Model B needs

### 3. Model A returns a decision
```json
{
  "selected_tools": ["web_search", "code_executor"],
  "memory_query": "user project preferences",
  "additional_context": "User prefers TypeScript",
  "priority": "medium",
  "reasoning": "User is asking about code, needs these tools"
}
```

### 4. Context is built for Model B
- Only selected tools are included
- Relevant memories are injected
- Clean, focused context

### 5. Model B responds
- Does the actual work
- Uses provided tools
- Response goes directly to user

### 6. Background processes
- Memories are extracted and stored
- Graph is pruned periodically

## Next Steps

After validating this prototype:
1. Collect training data from Model A decisions
2. Train a custom 1.5B model to replace Model A
3. The trained model runs 24/7 alongside Model B
4. Reduced latency and cost compared to using full LLM for context management

## License

TBD
