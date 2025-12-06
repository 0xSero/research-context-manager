# Cortex: Auxiliary Context Management Model
## Phase 1 Research - Executive Summary

**Project Codename**: Cortex
**Target Size**: 1.5B Parameters
**Hardware**: 8x RTX 3090 (192GB VRAM total) + 512GB DDR4
**Purpose**: Auxiliary "cognitive peripheral" for managing memory, tools, and context for larger LLMs

---

## Core Concept

Train a specialized small language model (1.5B params) to run 24/7 alongside a larger LLM, acting as its:
1. **Memory Manager** - Graph database of memories with semantic retrieval
2. **Tool Router** - Intelligent tool selection and context injection
3. **Context Compactor** - Conversation summarization and state continuation

This approach is inspired by:
- **Hierarchical Reasoning Models (HRM)** - Controller/Worker architecture
- **MemGPT** - Virtual context management with tiered memory
- **RouteLLM** - Cost-effective query routing via small models

---

## Key Research Findings

### 1. Similar Systems Exist (Partially)
| System | What It Does | Gap |
|--------|-------------|-----|
| **HRM** (27M params) | Hierarchical reasoning with Controller/Worker modules | Not designed for memory/tool management |
| **MemGPT/Letta** | Virtual context management | Relies on large LLM, not specialized small model |
| **RouteLLM** | Route queries to appropriate models | Only routing, no memory/compaction |
| **TinyAgent** | On-device tool calling | Limited to tool selection, no memory |

**Conclusion**: No existing system combines all three functions (memory + tools + compaction) in a single small specialized model. This is a novel contribution.

### 2. Architecture Is Feasible
- **Inheritune** demonstrated training 1.5B models on 1B tokens in <12 hours on a single A6000
- **TinyLlama** architecture (Llama 2 based, GQA, RoPE, SwiGLU) is well-proven
- **HRM's Controller/Worker** architecture provides a proven hierarchical approach
- 8x 3090s can handle 1.5B model training via FSDP/DeepSpeed ZeRO-3

### 3. Training Data Is Available
- **Berkeley Function Calling Leaderboard (BFCL)** datasets for tool calling
- **ToolACE** provides 26,507 diverse APIs with synthetic dialog data
- **GraphRAG** techniques provide graph memory patterns
- **Self-Instruct/Evol-Instruct** enable synthetic data generation at scale

### 4. Hardware Is Sufficient
- 8x RTX 3090 = 192GB VRAM (24GB each, but NVLink not available)
- 512GB DDR4 enables CPU offloading for larger batch sizes
- DeepSpeed ZeRO-3 with CPU offload can train 1.5B model comfortably
- Estimated training time: 2-4 weeks for full training run

---

## Proposed Architecture: "Cortex"

```
┌─────────────────────────────────────────────────────────┐
│                     CORTEX (1.5B)                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────────┐  ┌─────────────────┐              │
│  │  META-CONTROLLER │  │   TASK HEADS    │              │
│  │   (Slow/Abstract)│  │                 │              │
│  │                  │  │ ┌─────────────┐ │              │
│  │  - Decides which │  │ │ Memory Head │ │              │
│  │    head to use   │  │ ├─────────────┤ │              │
│  │  - Manages state │  │ │ Tool Head   │ │              │
│  │  - Priority      │  │ ├─────────────┤ │              │
│  │    arbitration   │  │ │ Compact Head│ │              │
│  │                  │  │ └─────────────┘ │              │
│  └────────┬─────────┘  └────────┬────────┘              │
│           │                     │                       │
│           └─────────┬───────────┘                       │
│                     ▼                                   │
│  ┌─────────────────────────────────────────────────┐   │
│  │            SHARED TRANSFORMER BACKBONE           │   │
│  │         (24 layers, 2048 hidden, GQA)           │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   MAIN LLM CONTEXT    │
              │   (Claude, GPT, etc.) │
              └───────────────────────┘
```

---

## Three Core Capabilities

### 1. Memory Management (Graph-Based)
- Maintain a knowledge graph of conversation entities, facts, and relationships
- Semantic retrieval based on current context
- Output: Relevant memory nodes to inject into main LLM context

### 2. Tool Routing
- Analyze incoming prompts to determine required tools
- Score and rank tools by relevance
- Output: Tool definitions + usage examples to inject into context

### 3. Context Compaction
- Summarize and compress conversation history
- Maintain semantic fidelity while reducing token count
- Output: Compressed state that can bootstrap new conversation

---

## Next Steps

1. **Phase 2: Debate** - Validate architecture decisions, compare alternatives
2. **Phase 3: Prototype** - Build minimal viable architecture, test on subset
3. **Phase 4: Training** - Full-scale training with curated dataset

---

## Document Index

| Document | Description |
|----------|-------------|
| [02-EXISTING-SYSTEMS.md](./02-EXISTING-SYSTEMS.md) | Deep dive on HRM, MemGPT, RouteLLM, etc. |
| [03-ARCHITECTURE-OPTIONS.md](./03-ARCHITECTURE-OPTIONS.md) | Transformer vs Mamba vs Hybrid analysis |
| [04-TRAINING-DATA.md](./04-TRAINING-DATA.md) | Data sources, synthetic generation |
| [05-HARDWARE-ANALYSIS.md](./05-HARDWARE-ANALYSIS.md) | 8x 3090 training considerations |
| [06-TRAINING-METHODOLOGY.md](./06-TRAINING-METHODOLOGY.md) | Training algorithms and schedules |
| [07-SOURCES.md](./07-SOURCES.md) | Complete bibliography with links |
