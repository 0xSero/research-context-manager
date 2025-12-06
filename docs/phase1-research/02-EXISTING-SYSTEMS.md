# Existing Systems Analysis

## Overview

This document provides a comprehensive analysis of existing systems that partially address the goals of the Cortex project. While no single system combines all three capabilities (memory management, tool routing, context compaction), several provide valuable architectural insights.

---

## 1. Hierarchical Reasoning Model (HRM)

**Source**: [arxiv.org/abs/2506.21734](https://arxiv.org/abs/2506.21734)
**GitHub**: [github.com/sapientinc/HRM](https://github.com/sapientinc/HRM)
**Released**: July 2025

### Architecture
HRM introduces a brain-inspired hierarchical architecture with two interdependent recurrent modules:

1. **Controller Module (Slow/Abstract)**
   - Responsible for high-level planning
   - Operates at longer timescales
   - Generates abstract goals and strategies

2. **Worker Module (Fast/Detailed)**
   - Handles rapid, detailed computations
   - Executes low-level operations
   - Reports results back to Controller

### Key Innovations
- **No CoT Required**: Achieves complex reasoning without Chain-of-Thought data
- **Minimal Parameters**: Only 27M parameters yet outperforms GPT-o3-mini, DeepSeek-R1
- **One-Step Gradient**: Uses one-step gradient approximation instead of BPTT
- **Q-Learning Training**: Reinforcement learning approach without replay buffers

### Relevance to Cortex
- **Highly Relevant**: The Controller/Worker hierarchy maps directly to our Meta-Controller/Task-Heads design
- **Adaptation Needed**: HRM focuses on reasoning tasks; we need to adapt for memory/tool/compaction
- **Training Insight**: Their Q-learning approach could inform our reward design

### Results
- 5% on ARC-AGI-2 (outperforming much larger models)
- Near-perfect on complex Sudoku and maze-finding
- 97% accuracy on climate forecasting (S2S)

---

## 2. MemGPT / Letta Framework

**Paper**: [arxiv.org/abs/2310.08560](https://arxiv.org/abs/2310.08560)
**Framework**: [letta.com](https://www.letta.com/)
**Released**: October 2023, Updated September 2024

### Architecture
MemGPT treats the LLM as an operating system, with virtual memory management:

```
┌─────────────────────────────────────────┐
│            MAIN CONTEXT                 │
│  ┌─────────────────────────────────┐   │
│  │ System Instructions              │   │
│  │ Working Context (scratchpad)     │   │
│  │ FIFO Queue (recent messages)     │   │
│  └─────────────────────────────────┘   │
├─────────────────────────────────────────┤
│           EXTERNAL CONTEXT              │
│  ┌─────────────────────────────────┐   │
│  │ Archival Storage (long-term)     │   │
│  │ Recall Storage (episodic)        │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

### Key Components
1. **Queue Manager**: Handles incoming messages and context overflow
2. **Function Executor**: Interprets LLM outputs as function calls
3. **Memory Functions**: `archival_memory_insert`, `archival_memory_search`, etc.
4. **Heartbeat Mechanism**: LLM can request immediate follow-up inference

### Relevance to Cortex
- **Highly Relevant**: Core inspiration for our memory management approach
- **Key Difference**: MemGPT uses the main LLM for memory decisions; we use a specialized small model
- **Advantage of Cortex**: Offloads memory management to avoid polluting main LLM context

### Limitations
- Requires the main LLM to understand and execute memory operations
- Memory operations consume main LLM tokens
- No specialized optimization for memory retrieval

---

## 3. RouteLLM

**GitHub**: [github.com/lm-sys/RouteLLM](https://github.com/lm-sys/RouteLLM)
**Released**: 2024

### Purpose
Route queries between strong (expensive) and weak (cheap) models based on query complexity.

### Architecture Options
1. **Similarity-Weighted Ranking**: Compute embeddings, find similar queries, use their routing decisions
2. **Matrix Factorization**: Learn latent factors for queries and models
3. **BERT Classifier**: Fine-tuned classifier for routing decisions
4. **Causal LLM**: Use a small LLM to predict routing

### Results
- Up to **85% cost reduction** while maintaining 95% of GPT-4 performance
- 60% cost reduction with <1% performance drop (BEST-Route from Microsoft)

### Relevance to Cortex
- **Relevant for Tool Routing**: Similar approach for selecting which tools to inject
- **Architecture Insight**: Shows small models can make effective routing decisions
- **Different Scope**: RouteLLM routes between models; we route between tools/contexts

---

## 4. TinyAgent

**Context**: On-device tool calling with small models (1-7B parameters)

### Key Finding
> "Even small fine-tuned models (1–7B parameters) can learn to perform API function calls on-device, achieving performance comparable to GPT-4 Turbo for routing queries to tools."

### Approach
- Fine-tune small models specifically for function calling
- Use differentiable controllers instead of LLM-based routing
- Achieve 3x cost reduction with maintained accuracy

### Relevance to Cortex
- **Highly Relevant**: Validates our approach of specialized small models for tool routing
- **Technical Insight**: Differentiable controllers can outperform LLM-based selection
- **Size Confirmation**: 1-7B parameter range is sufficient for complex tool selection

---

## 5. Cognitive Workspace

**Paper**: [arxiv.org/abs/2508.13171](https://arxiv.org/html/2508.13171v1)
**Concept**: Active memory management for LLMs

### Problem Statement
Traditional approaches (RAG, context extension) operate through **passive retrieval** rather than **active cognitive engagement**.

### Proposed Solution
- Emulate human cognitive mechanisms for external memory use
- Metacognitive control over memory operations
- Active workspace that the model consciously manages

### Three Dominant Approaches Analyzed
1. **Hardware-optimized** (Flash Attention, MInference): Fast but static patterns
2. **Memory-augmented** (MemGPT, Hierarchical Memory Transformer): Persistent but lacks metacognitive control
3. **RAG variants** (Self-RAG, CRAG, Adaptive RAG): Access to knowledge but passive

### Relevance to Cortex
- **Philosophical Alignment**: Our model should actively manage context, not passively retrieve
- **Design Insight**: Need metacognitive signals about when/what to retrieve
- **Gap Identified**: Current systems lack the "active engagement" we aim to provide

---

## 6. Context Compression Systems

### KVzip (November 2025)
**Source**: Seoul National University
- Compresses conversation memory 3-4x while maintaining accuracy
- Doubles response speed
- Supports up to 170,000 tokens

### Acon (October 2025)
**Paper**: [arxiv.org/abs/2510.00615](https://arxiv.org/html/2510.00615v1)
- Agent Context Compression
- Reduces peak tokens by 26-54%
- Enables distillation into smaller models (95% accuracy retention)

### LLMLingua (Microsoft)
**Blog**: [microsoft.com/research/blog/llmlingua](https://www.microsoft.com/en-us/research/blog/llmlingua-innovating-llm-efficiency-with-prompt-compression/)
- Up to 20x compression for prompts
- LongLLMLingua variant for long-context scenarios
- Preserves ICL and reasoning capabilities

### Factory.ai Incremental Summarization
- Rolling summary of conversation state
- Summarize only dropped spans, merge into persistent summary
- Different compression thresholds for different task types

### Relevance to Cortex
- **Core Capability**: These techniques directly inform our compaction head design
- **Implementation Options**: Can use learned compression vs. rule-based
- **Quality Metrics**: Need to preserve "important" information during compression

---

## 7. GraphRAG and Knowledge Graph Integration

### Microsoft GraphRAG
**GitHub**: Open-sourced 2024
- Uses community detection to partition knowledge graphs
- Enables queries requiring understanding across entire datasets
- Combines GNN reasoning with LLM generation

### QA-GNN
- Combines GNN reasoning with language model scoring
- State-of-the-art on commonsense and medical QA
- Interpretable reasoning paths

### GNN-RAG
- Graph Neural Retrieval for LLM reasoning
- Retrieves subgraphs relevant to queries
- Fuses graph structure with text generation

### Relevance to Cortex
- **Memory Head Design**: Our memory management should leverage graph structures
- **Retrieval Strategy**: Use GNN-like retrieval over stored memories
- **Entity Linking**: Need to maintain entity relationships across conversations

---

## Gap Analysis: Why Cortex Is Novel

| Capability | HRM | MemGPT | RouteLLM | TinyAgent | GraphRAG | Cortex |
|------------|-----|--------|----------|-----------|----------|--------|
| Hierarchical reasoning | ✓ | ✗ | ✗ | ✗ | ✗ | ✓ |
| Memory management | ✗ | ✓ | ✗ | ✗ | ✓ | ✓ |
| Tool routing | ✗ | ✗ | ✓ | ✓ | ✗ | ✓ |
| Context compaction | ✗ | Partial | ✗ | ✗ | ✗ | ✓ |
| Small specialized model | ✓ (27M) | ✗ | ✓ | ✓ | ✗ | ✓ (1.5B) |
| Runs alongside main LLM | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| Graph-based memory | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ |

**Conclusion**: Cortex uniquely combines all these capabilities into a single specialized model designed to run as a persistent auxiliary to a main LLM.

---

## References

1. [Hierarchical Reasoning Model - arXiv](https://arxiv.org/abs/2506.21734)
2. [HRM GitHub Repository](https://github.com/sapientinc/HRM)
3. [MemGPT Paper - arXiv](https://arxiv.org/abs/2310.08560)
4. [Letta Framework](https://www.letta.com/blog/memory-blocks)
5. [RouteLLM GitHub](https://github.com/lm-sys/RouteLLM)
6. [BEST-Route Microsoft](https://github.com/microsoft/best-route-llm)
7. [Cognitive Workspace - arXiv](https://arxiv.org/html/2508.13171v1)
8. [Acon Context Compression](https://arxiv.org/html/2510.00615v1)
9. [LLMLingua - Microsoft Research](https://www.microsoft.com/en-us/research/blog/llmlingua-innovating-llm-efficiency-with-prompt-compression/)
10. [GraphRAG Overview](https://www.datacamp.com/blog/knowledge-graphs-and-llms)
11. [Awesome Graph-LLM Collection](https://github.com/XiaoxinHe/Awesome-Graph-LLM)
