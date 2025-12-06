# Cortex: Auxiliary Context Management Model

A specialized 1.5B parameter language model designed to run alongside larger LLMs, acting as their cognitive peripheral for memory management, tool routing, and context compaction.

## Overview

Cortex is designed to solve a fundamental problem with modern LLM systems: **context pollution**. When an LLM handles memory, tools, and conversation management directly, these auxiliary tasks consume valuable context tokens and can degrade performance on the primary task.

Cortex runs 24/7 alongside your main LLM, handling:

1. **Memory Management** - Graph database of memories with semantic retrieval
2. **Tool Routing** - Intelligent tool selection and context injection
3. **Context Compaction** - Conversation summarization and state continuation

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | 8x RTX 3090 (192GB VRAM) | 8x A100 40GB |
| RAM | 256GB DDR4 | 512GB DDR4 |
| Storage | 2TB NVMe SSD | 4TB NVMe SSD |

## Project Status

**Current Phase**: Phase 1 - Research and Documentation

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1: Preparation | In Progress | Deep research and documentation |
| Phase 2: Debate | Pending | Architecture validation and alternatives |
| Phase 3: Prototype | Pending | Minimal viable implementation |
| Phase 4: Training | Pending | Full-scale model training |

## Research Inspiration

This project draws from several cutting-edge research areas:

- **[Hierarchical Reasoning Models (HRM)](https://arxiv.org/abs/2506.21734)** - Controller/Worker architecture
- **[MemGPT](https://arxiv.org/abs/2310.08560)** - Virtual context management
- **[RouteLLM](https://github.com/lm-sys/RouteLLM)** - Cost-effective query routing
- **[TinyAgent](https://github.com/SalesforceAIResearch/xLAM)** - On-device tool calling

## Documentation

### Phase 1 Research Documents

| Document | Description |
|----------|-------------|
| [Executive Summary](./docs/phase1-research/01-EXECUTIVE-SUMMARY.md) | High-level project overview |
| [Existing Systems](./docs/phase1-research/02-EXISTING-SYSTEMS.md) | Analysis of HRM, MemGPT, RouteLLM, etc. |
| [Architecture Options](./docs/phase1-research/03-ARCHITECTURE-OPTIONS.md) | Transformer vs Mamba vs Hybrid |
| [Training Data](./docs/phase1-research/04-TRAINING-DATA.md) | Data sources and synthetic generation |
| [Hardware Analysis](./docs/phase1-research/05-HARDWARE-ANALYSIS.md) | 8x 3090 training considerations |
| [Training Methodology](./docs/phase1-research/06-TRAINING-METHODOLOGY.md) | Training algorithms and schedules |
| [Sources](./docs/phase1-research/07-SOURCES.md) | Complete bibliography with links |

## Proposed Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     CORTEX (1.5B)                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────────┐  ┌─────────────────┐              │
│  │  META-CONTROLLER │  │   TASK HEADS    │              │
│  │   (Slow/Abstract)│  │ ┌─────────────┐ │              │
│  │                  │  │ │ Memory Head │ │              │
│  │  - Task routing  │  │ │ Tool Head   │ │              │
│  │  - Priority      │  │ │ Compact Head│ │              │
│  └────────┬─────────┘  └───────┬───────┘              │
│           └─────────┬───────────┘                       │
│                     ▼                                   │
│  ┌─────────────────────────────────────────────────┐   │
│  │            SHARED TRANSFORMER BACKBONE           │   │
│  │         (24-28 layers, 2048 hidden, GQA)        │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   MAIN LLM CONTEXT    │
              │   (Claude, GPT, etc.) │
              └───────────────────────┘
```

## Key Research Findings

1. **Novel Contribution**: No existing system combines memory + tools + compaction in a single small specialized model

2. **Architecture Feasibility**: 1.5B models can be trained on consumer hardware in reasonable timeframes
   - Inheritune: 1.5B model in <12 hours on single A6000
   - TinyLlama: Proven architecture at 1.1B scale

3. **Training Data Available**:
   - Berkeley Function Calling Leaderboard for tool routing
   - ToolACE with 26,507 diverse APIs
   - Self-Instruct/Evol-Instruct for synthetic generation

4. **Hardware Sufficient**: 8x RTX 3090 can train 1.5B model via DeepSpeed ZeRO-3
   - Estimated time: 2-4 weeks for prototype, 2-3 months for full training
   - Cost: ~$500-900 electricity vs $70,000+ cloud equivalent

## Training Plan Summary

| Phase | Duration | Tokens | Focus |
|-------|----------|--------|-------|
| Phase 1: Distillation | 2-3 weeks | 20B | Knowledge transfer from larger model |
| Phase 2: Task Training | 3-4 weeks | 30B | Memory, Tool, Compaction tasks |
| Phase 3: Multi-Task | 1-2 weeks | 10B | Combined training |
| Phase 4: Alignment | 1 week | 5B | DPO/preference optimization |

## Next Steps

1. **Phase 2: Debate** - Validate architecture decisions with ablation studies
2. **Phase 3: Prototype** - Build minimal viable model, test on subset
3. **Phase 4: Training** - Full-scale training with curated dataset

## Contributing

This project is in early research phase. Contributions welcome in:
- Architecture proposals
- Training data curation
- Benchmark design
- Hardware optimization

## License

TBD

## Acknowledgments

This project builds on research from:
- Sapient Intelligence (HRM)
- UC Berkeley (MemGPT, Gorilla, RouteLLM)
- Microsoft (LLMLingua, GraphRAG)
- HuggingFace (SmolLM, training infrastructure)
- Salesforce (xLAM, ToolACE)
