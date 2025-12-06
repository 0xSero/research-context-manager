# Architecture Options Analysis

## Overview

This document analyzes different architectural choices for the Cortex model, comparing transformer variants, state space models, and hybrid approaches. Given our target of 1.5B parameters and specialized multi-task requirements, architecture selection is critical.

---

## 1. Base Architecture Comparison

### Option A: Dense Transformer (Recommended)

**Examples**: TinyLlama, SmolLM, Qwen-1.5B

```
┌────────────────────────────────────────────┐
│              CORTEX-1.5B                   │
├────────────────────────────────────────────┤
│  Layers: 24-28                             │
│  Hidden: 2048                              │
│  Heads: 16 (with GQA: 4 KV heads)          │
│  Vocab: 32,000-50,000                      │
│  Context: 8,192 tokens                     │
│  Activation: SwiGLU                        │
│  Position: RoPE                            │
│  Norm: RMSNorm (pre-norm)                  │
└────────────────────────────────────────────┘
```

**Pros**:
- Well-understood, proven architecture
- Extensive tooling (FSDP, DeepSpeed, FlashAttention)
- Easy to distill from larger models
- Good for multi-task learning

**Cons**:
- Quadratic attention complexity (manageable at 8K context)
- Higher memory per token than SSMs

**Why Recommended**: SmolLM and TinyLlama demonstrate that this architecture works well at 1.5B scale. The tooling ecosystem is mature, and multi-task learning is well-studied.

---

### Option B: Mamba / State Space Model

**Examples**: Mamba-1.4B, Mamba-2

```
┌────────────────────────────────────────────┐
│              CORTEX-MAMBA                  │
├────────────────────────────────────────────┤
│  Layers: 48                                │
│  Hidden: 2048                              │
│  State Size: 16                            │
│  Expand: 2                                 │
│  Context: Unlimited (linear)               │
│  Activation: SiLU                          │
└────────────────────────────────────────────┘
```

**Pros**:
- Linear time complexity (O(n) vs O(n²))
- 5x faster inference than transformers
- Better for very long sequences
- Unlimited context window in theory

**Cons**:
- **Struggles with copying/retrieval tasks** (critical for our use case)
- Requires 100x more data to learn copying vs transformers
- Less mature tooling
- Harder to fine-tune for structured outputs

**Why Not Recommended**: Research shows Mamba struggles with retrieving and copying input information - a critical capability for our memory and tool routing tasks.

> "A small Transformer quickly learns to perfectly repeat the input string, while a Mamba model of a similar size fails. Even when increasing the hidden state size, Mamba requires 100x more data." - [Harvard Kempner Institute](https://kempnerinstitute.harvard.edu/research/deeper-learning/repeat-after-me-transformers-are-better-than-state-space-models-at-copying/)

---

### Option C: Hybrid Transformer-Mamba (Jamba-style)

```
┌────────────────────────────────────────────┐
│              CORTEX-HYBRID                 │
├────────────────────────────────────────────┤
│  Transformer Layers: 12 (for retrieval)    │
│  Mamba Layers: 12 (for sequence modeling)  │
│  Interleaved: T-M-T-M-T-M...              │
└────────────────────────────────────────────┘
```

**Pros**:
- Best of both worlds in theory
- Transformer layers handle retrieval
- Mamba layers handle long-range dependencies

**Cons**:
- Complex to train and optimize
- Less proven at small scales
- May not gain efficiency benefits at 1.5B scale

**Recommendation**: Consider for Phase 3 experimentation if pure transformer underperforms.

---

## 2. Multi-Task Architecture Design

### HRM-Inspired Controller/Worker

Based on the Hierarchical Reasoning Model architecture:

```
┌─────────────────────────────────────────────────────────┐
│                      CORTEX                             │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │           META-CONTROLLER (Slow Module)          │   │
│  │  - 4-6 dedicated transformer layers              │   │
│  │  - Larger hidden state (recurrent)               │   │
│  │  - Decides: which task head? what priority?      │   │
│  │  - Updates at lower frequency                    │   │
│  └─────────────────────────────────────────────────┘   │
│                         │                               │
│            ┌────────────┼────────────┐                 │
│            ▼            ▼            ▼                 │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐      │
│  │ MEMORY HEAD │ │  TOOL HEAD  │ │COMPACT HEAD │      │
│  │  (Worker 1) │ │  (Worker 2) │ │  (Worker 3) │      │
│  │             │ │             │ │             │      │
│  │ Graph query │ │ Tool select │ │ Summarize   │      │
│  │ Entity link │ │ Param fill  │ │ Prioritize  │      │
│  │ Relevance   │ │ Context gen │ │ Compress    │      │
│  └─────────────┘ └─────────────┘ └─────────────┘      │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │       SHARED BACKBONE (Fast Module)              │   │
│  │  - 18-20 transformer layers                      │   │
│  │  - Grouped-Query Attention                       │   │
│  │  - FlashAttention-2 optimized                    │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Multi-Task Head Design

Each task head consists of:

1. **Task-Specific Projection Layer** (2048 → 512)
2. **Task-Specific Transformer Block** (2 layers)
3. **Task-Specific Output Head**

```python
# Pseudo-architecture
class CortexModel:
    def __init__(self):
        self.backbone = TransformerBackbone(layers=20, hidden=2048)
        self.controller = ControllerModule(layers=4, hidden=2048)

        self.memory_head = TaskHead(
            input_dim=2048,
            output_types=["entity_ids", "relevance_scores", "graph_ops"]
        )
        self.tool_head = TaskHead(
            input_dim=2048,
            output_types=["tool_ids", "param_schema", "context_snippet"]
        )
        self.compact_head = TaskHead(
            input_dim=2048,
            output_types=["summary", "importance_mask", "state_vector"]
        )

    def forward(self, x, task_type=None):
        # Backbone encoding
        hidden = self.backbone(x)

        # Controller decides task (or use explicit task_type)
        if task_type is None:
            task_type, confidence = self.controller(hidden)

        # Route to appropriate head
        if task_type == "memory":
            return self.memory_head(hidden)
        elif task_type == "tool":
            return self.tool_head(hidden)
        elif task_type == "compact":
            return self.compact_head(hidden)
```

---

## 3. Specific Architecture Recommendations

### Primary Recommendation: SmolLM-Style Architecture

Based on HuggingFace's SmolLM architecture, optimized for our use case:

| Component | Value | Rationale |
|-----------|-------|-----------|
| **Total Parameters** | 1.5B | Target specification |
| **Layers** | 28 | Depth over width (SmolLM finding) |
| **Hidden Dimension** | 2048 | Standard for 1.5B scale |
| **Attention Heads** | 16 | Standard ratio |
| **KV Heads (GQA)** | 4 | 4:1 ratio reduces KV cache |
| **Intermediate (FFN)** | 8192 | 4x hidden (SwiGLU) |
| **Vocabulary** | 49,152 | Match SmolLM tokenizer or use custom |
| **Max Position** | 8,192 | Sufficient for our context management |
| **Positional Encoding** | RoPE | State-of-the-art for position generalization |
| **Normalization** | RMSNorm (Pre-LN) | Stable training |
| **Activation** | SwiGLU | Better than GELU at this scale |

### Task Head Specifications

| Head | Parameters | Output Format |
|------|------------|---------------|
| **Controller** | ~50M | Task routing logits, priority scores |
| **Memory Head** | ~100M | Entity IDs, relevance scores, graph operations |
| **Tool Head** | ~100M | Tool IDs, parameter schemas, context snippets |
| **Compact Head** | ~100M | Summary text, importance mask, state embedding |
| **Shared Backbone** | ~1.15B | Hidden representations |

---

## 4. Input/Output Specifications

### Input Format

```
<|cortex|>
<|task|>{task_type}</|task|>
<|context|>
{current_conversation_context}
</|context|>
<|memory_state|>
{serialized_graph_state}
</|memory_state|>
<|available_tools|>
{tool_definitions}
</|available_tools|>
<|query|>
{what_to_do}
</|query|>
```

### Output Formats

**Memory Head Output:**
```json
{
  "action": "retrieve|store|update|delete",
  "entities": [
    {"id": "e123", "relevance": 0.95, "content": "..."}
  ],
  "graph_ops": [
    {"op": "add_edge", "from": "e123", "to": "e456", "type": "related_to"}
  ],
  "inject_context": "Relevant memories: ..."
}
```

**Tool Head Output:**
```json
{
  "selected_tools": ["web_search", "code_executor"],
  "rationale": "User wants to find and run code",
  "context_injection": "Available tools:\n1. web_search(query)...",
  "parameter_hints": {
    "web_search": {"query": "python sorting algorithms"}
  }
}
```

**Compact Head Output:**
```json
{
  "summary": "Compressed conversation summary...",
  "key_facts": ["User prefers Python", "Working on sorting"],
  "state_vector": [0.1, 0.5, ...],  // For continuity
  "importance_mask": [1, 1, 0, 0, 1, ...]  // Which turns to keep
}
```

---

## 5. Alternative Architectures Considered

### Mixture of Experts (MoE)

**Concept**: Use sparse MoE with each expert specializing in one task.

```
┌─────────────────────────────────────────────┐
│  Router → Expert 1 (Memory)                 │
│         → Expert 2 (Tools)                  │
│         → Expert 3 (Compact)                │
│         → Expert 4 (General)                │
└─────────────────────────────────────────────┘
```

**Pros**: More parameters with same compute
**Cons**: Complex routing, training instability at small scale
**Verdict**: Consider if initial architecture underperforms

### Encoder-Decoder

**Concept**: Use encoder for context understanding, decoder for output.

**Pros**: Good for summarization (compact head)
**Cons**: Less natural for other tasks, larger memory footprint
**Verdict**: Decoder-only is more versatile for our use case

### Retrieval-Augmented (RETRO-style)

**Concept**: Built-in retrieval mechanism for memory

**Pros**: Efficient memory access
**Cons**: Complex training, requires retrieval infrastructure
**Verdict**: Implement retrieval at application level, not architecture level

---

## 6. Implementation Notes

### Grouped-Query Attention (GQA)

Essential for memory efficiency. With 16 attention heads and 4 KV heads:
- 4x reduction in KV cache memory
- Minimal quality degradation
- Enables longer batch processing on 3090s

### FlashAttention-2 Integration

Required for efficient training on 3090s:
- Reduces memory from O(N²) to O(N)
- 2-4x speedup in attention computation
- Native support in PyTorch 2.0+

### Rotary Position Embeddings (RoPE)

Benefits:
- Relative position encoding
- Better length generalization
- No learned position embeddings to manage

---

## 7. Recommended Final Architecture

```
CORTEX-1.5B Configuration:
├── Embedding Layer (49,152 × 2048)
├── Transformer Blocks × 28
│   ├── RMSNorm (Pre-LN)
│   ├── Self-Attention (GQA: 16 heads, 4 KV heads)
│   │   └── RoPE positional encoding
│   ├── RMSNorm
│   └── SwiGLU FFN (2048 → 8192 → 2048)
├── Task Router (4 specialized layers)
│   └── Outputs: task_logits, routing_weights
├── Task Heads × 3
│   ├── Memory Head (2 layers + output projection)
│   ├── Tool Head (2 layers + output projection)
│   └── Compact Head (2 layers + output projection)
└── Shared Output Projection (2048 → vocab)
```

**Total Parameters**: ~1.5B
**Training Tokens Target**: 50B-100B (mixture of tasks)
**Context Length**: 8,192 tokens

---

## References

1. [SmolLM Training Playbook](https://gist.github.com/jph00/3c97a2c6c5075c4e7b98faae634b033a)
2. [TinyLlama Architecture](https://www.digitalocean.com/community/tutorials/tinyllama)
3. [Mamba Visual Guide](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mamba-and-state)
4. [Mamba vs Transformer Tradeoffs](https://goombalab.github.io/blog/2025/tradeoffs/)
5. [GQA Paper - Grouped Query Attention](https://arxiv.org/abs/2305.13245)
6. [Multi-Task Learning Survey](https://arxiv.org/abs/2009.09796)
7. [HRM Architecture](https://arxiv.org/abs/2506.21734)
