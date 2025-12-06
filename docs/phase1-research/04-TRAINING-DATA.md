# Training Data Analysis

## Overview

This document analyzes available training data sources for the Cortex model, covering each of its three core capabilities: memory management, tool routing, and context compaction. We also discuss synthetic data generation strategies.

---

## 1. Data Requirements Summary

| Capability | Primary Data Need | Estimated Tokens |
|------------|------------------|------------------|
| **Memory Management** | Graph-structured conversations, entity extraction | 15-20B tokens |
| **Tool Routing** | Function calling datasets, API schemas | 15-20B tokens |
| **Context Compaction** | Summarization, conversation compression | 10-15B tokens |
| **Base Language** | General language understanding | 20-30B tokens |
| **Total** | Mixed multi-task dataset | **60-80B tokens** |

---

## 2. Tool Routing Datasets

### 2.1 Berkeley Function Calling Leaderboard (BFCL)

**Source**: [huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard)

**Description**: Industry-standard benchmark for LLM function calling capabilities.

**Categories**:
| Category | Count | Description |
|----------|-------|-------------|
| Simple | 258 | Single function, single argument |
| Multiple | 1,037 | Multiple functions to choose from |
| Parallel | 16 | Multiple functions called in parallel |
| Parallel Multiple | 24 | Complex parallel scenarios |
| Irrelevance Detection | 875 | Recognizing when no function applies |

**Relevance**: Directly applicable for training the Tool Head to:
- Select appropriate tools from a list
- Recognize when no tool is needed
- Handle complex multi-tool scenarios

### 2.2 ToolACE Dataset

**Paper**: [arxiv.org/abs/2409.00920](https://arxiv.org/html/2409.00920v1) (ICLR 2025)

**Description**: Large-scale synthetic dataset for tool learning.

**Key Stats**:
- **26,507 diverse APIs** across multiple domains
- **Dual-layer verification** (rule-based + model-based)
- **Complexity evaluator** ensures dialog diversity

**Generation Method** (Tool Self-Evolution Synthesis):
1. **Speciation**: Generate diverse tool categories
2. **Adaptation**: Evolve tools with various data types/constraints
3. **Evolution**: Create increasingly complex tool chains

**Data Format**:
```json
{
  "conversation": [
    {"role": "user", "content": "Find flights to Paris"},
    {"role": "assistant", "function_call": {
      "name": "search_flights",
      "arguments": {"destination": "Paris", "date": "2024-03-15"}
    }},
    {"role": "function", "content": "{\"flights\": [...]}"},
    {"role": "assistant", "content": "I found 5 flights to Paris..."}
  ],
  "tools": [...]
}
```

### 2.3 xLAM Datasets (Salesforce)

**Source**: [github.com/SalesforceAIResearch/xLAM](https://github.com/SalesforceAIResearch/xLAM)

**Components**:
- **xLAM-1b-fc-r**: 1B parameter function calling model
- **APIGen-MT**: Multi-turn agentic data generation pipeline

**Features**:
- High-quality synthetic agentic data
- Multi-turn conversation support
- Tool chaining scenarios

### 2.4 Gorilla Dataset

**Source**: [github.com/ShishirPatil/gorilla](https://github.com/ShishirPatil/gorilla)

**Description**: Training data for Gorilla LLM, specialized in API calls.

**Includes**:
- 16,000+ API documentation entries
- Paired (query, API call) examples
- Multiple API providers (HuggingFace, TensorFlow, PyTorch)

---

## 3. Memory/Graph Datasets

### 3.1 GraphQA Datasets

**Knowledge Graph QA Datasets**:
- **Freebase QA**: Entity-centric questions over Freebase
- **WebQuestionsSP**: Complex multi-hop questions
- **ComplexWebQuestions**: Multi-constraint queries

**Application**: Train the Memory Head to:
- Retrieve relevant entities
- Navigate graph relationships
- Answer queries over structured data

### 3.2 Conversational Entity Tracking

**MultiWOZ Dataset**:
- Multi-domain task-oriented dialogs
- Entity slot tracking across turns
- State updates throughout conversation

**Application**: Train memory state management across conversation turns.

### 3.3 Synthetic Graph Memory Data

We will need to **generate synthetic data** for graph memory operations:

```python
# Example synthetic data generation
def generate_memory_training_example():
    return {
        "conversation_context": "User discussed their project deadline...",
        "current_query": "What was the deadline mentioned?",
        "memory_graph": {
            "nodes": [
                {"id": "e1", "type": "project", "name": "Website Redesign"},
                {"id": "e2", "type": "date", "value": "March 15"},
                {"id": "e3", "type": "person", "name": "John"}
            ],
            "edges": [
                {"from": "e1", "to": "e2", "type": "deadline"},
                {"from": "e3", "to": "e1", "type": "owner"}
            ]
        },
        "expected_output": {
            "retrieved_entities": ["e1", "e2"],
            "answer": "The deadline for the Website Redesign project is March 15"
        }
    }
```

---

## 4. Context Compaction Datasets

### 4.1 Summarization Datasets

| Dataset | Size | Type | Source |
|---------|------|------|--------|
| **CNN/DailyMail** | 300K | News summarization | [HuggingFace](https://huggingface.co/datasets/cnn_dailymail) |
| **XSum** | 227K | Extreme summarization | [HuggingFace](https://huggingface.co/datasets/xsum) |
| **SAMSum** | 16K | Dialog summarization | [HuggingFace](https://huggingface.co/datasets/samsum) |
| **DialogSum** | 13K | Dialog summarization | [HuggingFace](https://huggingface.co/datasets/knkarthick/dialogsum) |

### 4.2 Conversation Compression Data

**Synthetic Generation Strategy**:
1. Take long conversations from ShareGPT or similar
2. Use GPT-4/Claude to create high-quality compressions
3. Validate compression quality (information preservation)
4. Create (long_conv, compressed_conv) pairs

**Format**:
```json
{
  "original_conversation": [
    {"role": "user", "content": "...long message 1..."},
    {"role": "assistant", "content": "...long response 1..."},
    // ... many turns
  ],
  "compression_ratio": 0.3,
  "compressed_state": {
    "summary": "User is building a Python web scraper...",
    "key_facts": [
      "Using BeautifulSoup library",
      "Target site: example.com",
      "Wants to extract product prices"
    ],
    "user_preferences": ["Prefers concise code", "Python 3.10+"],
    "current_task": "Debugging rate limiting issue"
  },
  "continuation_prompt": "Continuing conversation about web scraper debugging..."
}
```

### 4.3 Multi-Turn Datasets for Continuity

**ShareGPT Dataset**:
- Large collection of ChatGPT conversations
- Natural multi-turn dialogs
- Diverse topics

**LMSYS Chat Dataset**:
- Real user conversations
- Multiple model outputs
- Quality ratings

---

## 5. Base Language Data

### 5.1 SlimPajama

**Source**: [HuggingFace](https://huggingface.co/datasets/cerebras/SlimPajama-627B)

**Size**: 627B tokens (deduplicated RedPajama)

**Composition**:
| Source | Percentage |
|--------|------------|
| CommonCrawl | 52% |
| C4 | 27% |
| GitHub | 5% |
| Books | 4% |
| ArXiv | 4% |
| Wikipedia | 4% |
| StackExchange | 4% |

**Usage**: Subset for general language understanding pretraining.

### 5.2 StarCoder Data

**Source**: The Stack

**Purpose**: Code understanding for tool parameter generation.

---

## 6. Synthetic Data Generation Pipeline

### 6.1 Self-Instruct Approach

**Source**: [huggingface.co/blog/davanstrien/self-instruct](https://huggingface.co/blog/davanstrien/self-instruct)

**Process**:
1. Create seed examples (50-100 high-quality examples per task)
2. Use large LLM to generate similar examples
3. Filter for quality and diversity
4. Iterate with evolved complexity

### 6.2 Evol-Instruct for Complexity

**Source**: Microsoft WizardLM approach

**Evolution Types**:
1. **In-Depth**: Make instructions more detailed
2. **In-Breadth**: Generate diverse new instructions
3. **Elimination**: Remove failed/low-quality examples

**Example Evolution Chain**:
```
Seed: "Call the weather API to get today's forecast"
↓ Evolve (depth)
"Call the weather API to get a 5-day forecast for multiple cities, then format the results as a comparison table"
↓ Evolve (breadth)
"Use the flight search API to find the cheapest flights, then call the weather API for destination forecasts"
```

### 6.3 Task-Specific Generation

**Memory Task Generation**:
```python
def generate_memory_task(complexity="medium"):
    """Generate synthetic memory management training data."""
    scenarios = [
        "extract_entities",      # Extract entities from conversation
        "update_relationships",  # Update graph relationships
        "query_memory",          # Answer questions from memory
        "prune_irrelevant",      # Remove outdated information
        "merge_duplicates"       # Combine duplicate entities
    ]
    # ... generate based on scenario and complexity
```

**Tool Routing Generation**:
```python
def generate_tool_routing_task():
    """Generate tool selection training data."""
    tool_pool = load_tool_definitions()  # From ToolACE/BFCL
    query = generate_user_query()
    relevant_tools = select_relevant_subset(tool_pool, query)
    return {
        "query": query,
        "available_tools": random.sample(tool_pool, 10),
        "selected_tools": relevant_tools,
        "rationale": generate_rationale(query, relevant_tools)
    }
```

**Compaction Generation**:
```python
def generate_compaction_task():
    """Generate context compression training data."""
    conversation = sample_long_conversation()
    compressed = call_gpt4_for_compression(conversation)
    validate_information_preservation(conversation, compressed)
    return {
        "original": conversation,
        "compressed": compressed,
        "compression_ratio": len(compressed) / len(conversation)
    }
```

---

## 7. Data Quality Assurance

### 7.1 Dual-Layer Verification (ToolACE approach)

**Layer 1: Rule-Based Checks**
- Syntax validation for function calls
- Argument type checking
- Required parameter presence
- Format consistency

**Layer 2: Model-Based Checks**
- Semantic correctness verification
- Response quality scoring
- Diversity metrics
- Complexity validation

### 7.2 Quality Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Syntax Accuracy** | >99% | Automated parsing |
| **Semantic Accuracy** | >95% | GPT-4 evaluation |
| **Diversity Score** | >0.8 | Embedding clustering |
| **Complexity Distribution** | Uniform | Complexity evaluator |
| **Information Preservation** | >90% | Q&A validation |

---

## 8. Data Mixing Strategy

### 8.1 Multi-Task Mixing Ratios

```
Training Data Distribution:
├── 25% Tool Routing Data
│   ├── 50% ToolACE synthetic
│   ├── 30% BFCL + Gorilla
│   └── 20% Generated evolve-instruct
├── 25% Memory Management Data
│   ├── 40% Synthetic graph operations
│   ├── 40% Entity tracking (MultiWOZ, etc.)
│   └── 20% GraphQA adaptation
├── 20% Context Compaction Data
│   ├── 50% Dialog summarization
│   ├── 50% Synthetic compression pairs
├── 25% Base Language Data
│   └── SlimPajama subset
└── 5% Multi-Task Combined
    └── Examples requiring multiple capabilities
```

### 8.2 Curriculum Learning

**Phase 1 (Single Task)**:
- Train each capability separately
- Establish baseline performance

**Phase 2 (Task Mixing)**:
- Gradually introduce task mixing
- Start with 80% single-task, 20% mixed

**Phase 3 (Full Multi-Task)**:
- Equal distribution across all tasks
- Include complex multi-capability examples

---

## 9. Estimated Data Acquisition Effort

| Source | Availability | Effort | Tokens |
|--------|--------------|--------|--------|
| BFCL | Public | Low | 500K |
| ToolACE | Public | Low | 5B |
| SlimPajama | Public | Low | 20B (subset) |
| SAMSum/DialogSum | Public | Low | 100M |
| Synthetic Memory | Generate | High | 10B |
| Synthetic Compaction | Generate | High | 5B |
| Synthetic Multi-Task | Generate | Medium | 5B |
| **Total** | | | **~45B tokens** |

---

## 10. References

1. [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html)
2. [ToolACE Paper](https://arxiv.org/html/2409.00920v1)
3. [xLAM Repository](https://github.com/SalesforceAIResearch/xLAM)
4. [Gorilla LLM](https://github.com/ShishirPatil/gorilla)
5. [Self-Instruct Tutorial](https://huggingface.co/blog/davanstrien/self-instruct)
6. [Evol-Instruct (WizardLM)](https://arxiv.org/abs/2304.12244)
7. [SlimPajama Dataset](https://huggingface.co/datasets/cerebras/SlimPajama-627B)
8. [Awesome LLM Synthetic Data](https://github.com/wasiahmad/Awesome-LLM-Synthetic-Data)
9. [NVIDIA Nemotron Synthetic Data](https://blogs.nvidia.com/blog/nemotron-4-synthetic-data-generation-llm-training/)
