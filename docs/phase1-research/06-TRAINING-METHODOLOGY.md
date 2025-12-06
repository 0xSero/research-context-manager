# Training Methodology

## Overview

This document outlines the training methodology for Cortex, including pretraining strategies, multi-task learning approaches, and alignment techniques.

---

## 1. Training Philosophy

### Core Principles

1. **Efficiency First**: Use knowledge distillation and efficient pretraining techniques
2. **Multi-Task from Start**: Train on all three tasks simultaneously
3. **Curriculum Learning**: Gradually increase task complexity
4. **Continuous Validation**: Evaluate on held-out task-specific benchmarks

### Training Phases

```
Phase 1: Base Capability Building
    └── Knowledge distillation + General pretraining
    └── Duration: 2-3 weeks

Phase 2: Task-Specific Training
    └── Memory, Tool, Compaction task training
    └── Duration: 2-3 weeks

Phase 3: Multi-Task Integration
    └── Combined training with task mixing
    └── Duration: 1-2 weeks

Phase 4: Alignment & Refinement
    └── RLHF/DPO for output quality
    └── Duration: 1 week
```

---

## 2. Phase 1: Knowledge Distillation

### 2.1 Why Distillation?

Based on research findings:

> "Distilling step-by-step enables a 770M parameter T5 model to outperform the few-shot prompted 540B PaLM model using only 80% of examples, demonstrating a >700x model size reduction." - [Google Research](https://research.google/blog/distilling-step-by-step-outperforming-larger-language-models-with-less-training-data-and-smaller-model-sizes/)

**Benefits**:
- Faster training convergence
- Better sample efficiency
- Inherits reasoning patterns from teacher

### 2.2 Teacher Model Selection

| Teacher Option | Pros | Cons |
|----------------|------|------|
| **GPT-4** | Best quality | Expensive API costs, black-box |
| **Claude 3.5** | High quality, good reasoning | API costs |
| **Llama 3.1 70B** | Open weights, self-hostable | Large compute for inference |
| **Qwen2.5 72B** | Open, strong on tasks | Large compute |

**Recommendation**: Use **Llama 3.1 70B** or **Qwen2.5 72B** for open-weights distillation.

### 2.3 Distillation Approaches

#### Method 1: Standard KD (Output Matching)

```python
def distillation_loss(student_logits, teacher_logits, labels, temperature=2.0, alpha=0.5):
    """Standard knowledge distillation loss."""
    # Soft target loss (KL divergence)
    soft_targets = F.softmax(teacher_logits / temperature, dim=-1)
    soft_student = F.log_softmax(student_logits / temperature, dim=-1)
    soft_loss = F.kl_div(soft_student, soft_targets, reduction='batchmean') * (temperature ** 2)

    # Hard target loss (cross-entropy)
    hard_loss = F.cross_entropy(student_logits, labels)

    return alpha * soft_loss + (1 - alpha) * hard_loss
```

#### Method 2: MiniLLM (Forward KL Optimization)

**Source**: [arxiv.org/abs/2306.08543](https://arxiv.org/abs/2306.08543)

MiniLLM optimizes forward KL divergence instead of reverse KL, which is more suitable for language model distillation.

```python
def minillm_loss(student_logits, teacher_logits):
    """MiniLLM forward KL loss."""
    teacher_probs = F.softmax(teacher_logits, dim=-1)
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    return F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
```

#### Method 3: Step-by-Step Distillation

**Source**: [Google Research](https://research.google/blog/distilling-step-by-step-outperforming-larger-language-models-with-less-training-data-and-smaller-model-sizes/)

Extract reasoning steps (rationales) from teacher alongside answers:

```python
# Teacher generates: (rationale, answer) pairs
teacher_output = teacher.generate(
    prompt,
    generate_rationale=True
)

# Student learns to produce both
student_loss = (
    rationale_loss(student_rationale, teacher_rationale) +
    answer_loss(student_answer, teacher_answer)
)
```

### 2.4 Distillation Data Generation

```python
def generate_distillation_data(teacher_model, prompts):
    """Generate teacher outputs for distillation."""
    data = []
    for prompt in prompts:
        # Generate with high temperature for diversity
        outputs = teacher_model.generate(
            prompt,
            num_return_sequences=4,
            temperature=0.7,
            do_sample=True
        )

        # Also get logits for soft targets
        with torch.no_grad():
            logits = teacher_model(prompt).logits

        data.append({
            "prompt": prompt,
            "teacher_outputs": outputs,
            "teacher_logits": logits
        })
    return data
```

---

## 3. Phase 2: Task-Specific Training

### 3.1 Memory Task Training

**Objective**: Learn graph memory operations

**Training Signal Types**:
1. **Entity Extraction**: Identify entities in conversation
2. **Relationship Prediction**: Predict edges between entities
3. **Relevance Scoring**: Score memory items by query relevance
4. **Graph Operations**: Generate correct add/update/delete operations

**Loss Function**:
```python
def memory_task_loss(predictions, labels):
    # Entity extraction loss
    entity_loss = F.cross_entropy(
        predictions['entity_logits'],
        labels['entities']
    )

    # Relationship prediction loss
    relation_loss = F.binary_cross_entropy_with_logits(
        predictions['relation_logits'],
        labels['relations']
    )

    # Relevance scoring loss (regression)
    relevance_loss = F.mse_loss(
        predictions['relevance_scores'],
        labels['relevance']
    )

    return entity_loss + relation_loss + 0.5 * relevance_loss
```

### 3.2 Tool Routing Training

**Objective**: Select and configure appropriate tools

**Training Signal Types**:
1. **Tool Selection**: Multi-label classification over tool set
2. **Relevance Detection**: Binary - are any tools relevant?
3. **Parameter Generation**: Generate correct function arguments
4. **Context Generation**: Generate helpful context for main LLM

**Loss Function**:
```python
def tool_task_loss(predictions, labels):
    # Tool selection (multi-label)
    selection_loss = F.binary_cross_entropy_with_logits(
        predictions['tool_scores'],
        labels['selected_tools']
    )

    # Relevance detection (binary)
    relevance_loss = F.binary_cross_entropy_with_logits(
        predictions['any_relevant'],
        labels['tools_relevant']
    )

    # Parameter generation (language modeling)
    param_loss = F.cross_entropy(
        predictions['param_logits'].view(-1, vocab_size),
        labels['parameters'].view(-1)
    )

    return selection_loss + relevance_loss + param_loss
```

### 3.3 Compaction Training

**Objective**: Compress conversations while preserving information

**Training Signal Types**:
1. **Summarization**: Generate compressed summaries
2. **Key Fact Extraction**: Identify critical information
3. **Importance Masking**: Score turns by importance
4. **Continuation Coherence**: Compressed state enables good continuation

**Loss Function**:
```python
def compaction_task_loss(predictions, labels):
    # Summary generation loss
    summary_loss = F.cross_entropy(
        predictions['summary_logits'].view(-1, vocab_size),
        labels['summary'].view(-1)
    )

    # Key fact extraction
    fact_loss = F.cross_entropy(
        predictions['fact_logits'].view(-1, vocab_size),
        labels['key_facts'].view(-1)
    )

    # Importance mask (binary per turn)
    importance_loss = F.binary_cross_entropy_with_logits(
        predictions['importance_scores'],
        labels['importance_mask']
    )

    # Information preservation reward (evaluated separately)

    return summary_loss + fact_loss + 0.3 * importance_loss
```

---

## 4. Phase 3: Multi-Task Training

### 4.1 Task Mixing Strategy

**Option A: Round-Robin**
```python
def round_robin_sampler(memory_data, tool_data, compact_data):
    """Alternate between tasks each batch."""
    iterators = [
        iter(memory_data),
        iter(tool_data),
        iter(compact_data)
    ]
    while True:
        for it in iterators:
            yield next(it)
```

**Option B: Proportional Sampling**
```python
def proportional_sampler(datasets, proportions=[0.33, 0.33, 0.34]):
    """Sample from tasks according to proportions."""
    while True:
        task_idx = np.random.choice(len(datasets), p=proportions)
        yield next(datasets[task_idx])
```

**Option C: Gradient-Based Balancing**
```python
def gradient_balanced_sampler(datasets, model):
    """Dynamically adjust based on gradient magnitudes."""
    # Track gradient norms per task
    grad_norms = [1.0, 1.0, 1.0]

    for step in range(max_steps):
        # Sample inversely proportional to gradient norm
        probs = [1/g for g in grad_norms]
        probs = [p/sum(probs) for p in probs]

        task_idx = np.random.choice(len(datasets), p=probs)
        batch = next(datasets[task_idx])

        loss = compute_loss(model, batch, task_idx)
        loss.backward()

        # Update gradient norm for this task
        grad_norms[task_idx] = compute_grad_norm(model)

        optimizer.step()
```

### 4.2 Multi-Task Loss Weighting

**Uncertainty Weighting** (Kendall et al.):
```python
class MultiTaskLoss(nn.Module):
    def __init__(self, num_tasks=3):
        super().__init__()
        # Learnable log variance per task
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses):
        weighted_losses = []
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            weighted_losses.append(precision * loss + self.log_vars[i])
        return sum(weighted_losses)
```

### 4.3 Preventing Catastrophic Forgetting

**Elastic Weight Consolidation (EWC)**:
```python
def ewc_loss(model, fisher_info, optimal_params, lambda_=1000):
    """Penalize changes to important weights."""
    ewc_penalty = 0
    for name, param in model.named_parameters():
        if name in fisher_info:
            ewc_penalty += (fisher_info[name] * (param - optimal_params[name]) ** 2).sum()
    return lambda_ * ewc_penalty
```

**Progressive Training**:
- Start with task-specific heads frozen
- Gradually unfreeze from output to input layers
- Use lower learning rate for backbone

---

## 5. Phase 4: Alignment

### 5.1 Direct Preference Optimization (DPO)

**Why DPO over RLHF**: Simpler, no reward model needed, stable training.

**Source**: [DPO Paper](https://arxiv.org/abs/2305.18290)

```python
def dpo_loss(model, ref_model, chosen, rejected, beta=0.1):
    """Direct Preference Optimization loss."""
    # Get log probs from current model
    chosen_logps = get_log_probs(model, chosen)
    rejected_logps = get_log_probs(model, rejected)

    # Get log probs from reference model
    with torch.no_grad():
        ref_chosen_logps = get_log_probs(ref_model, chosen)
        ref_rejected_logps = get_log_probs(ref_model, rejected)

    # DPO objective
    chosen_rewards = beta * (chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (rejected_logps - ref_rejected_logps)

    return -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
```

### 5.2 Preference Data Generation

For each task, generate preference pairs:

**Memory Task Preferences**:
```json
{
  "prompt": "Retrieve memories about the user's project deadlines",
  "chosen": {
    "retrieved": ["Project X deadline: March 15", "Project Y deadline: April 1"],
    "quality": "relevant, complete"
  },
  "rejected": {
    "retrieved": ["User likes coffee", "Weather is sunny"],
    "quality": "irrelevant"
  }
}
```

**Tool Task Preferences**:
```json
{
  "prompt": "User wants to search for Python documentation",
  "chosen": {
    "tools": ["web_search"],
    "params": {"query": "Python documentation official"},
    "quality": "correct tool and params"
  },
  "rejected": {
    "tools": ["calculator", "weather"],
    "params": {},
    "quality": "wrong tools"
  }
}
```

---

## 6. Training Hyperparameters

### 6.1 Recommended Configuration

| Parameter | Phase 1 | Phase 2 | Phase 3 | Phase 4 |
|-----------|---------|---------|---------|---------|
| **Learning Rate** | 3e-4 | 1e-4 | 5e-5 | 1e-6 |
| **Warmup Steps** | 2000 | 1000 | 500 | 100 |
| **Batch Size** | 128 | 128 | 64 | 32 |
| **Sequence Length** | 2048 | 4096 | 4096 | 4096 |
| **Weight Decay** | 0.1 | 0.1 | 0.01 | 0.01 |
| **Gradient Clip** | 1.0 | 1.0 | 1.0 | 0.5 |
| **LR Schedule** | Cosine | Cosine | Linear | Linear |
| **Dropout** | 0.1 | 0.1 | 0.05 | 0.0 |

### 6.2 Optimizer Configuration

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    betas=(0.9, 0.95),
    weight_decay=0.1,
    eps=1e-8
)

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=2000,
    num_training_steps=500000
)
```

---

## 7. Evaluation Strategy

### 7.1 Task-Specific Benchmarks

**Memory Task**:
- Entity extraction F1
- Relationship accuracy
- Retrieval relevance (MRR, NDCG)

**Tool Task**:
- Tool selection accuracy
- Irrelevance detection accuracy
- Parameter generation BLEU/exact match
- BFCL benchmark scores

**Compaction Task**:
- ROUGE scores vs reference summaries
- Information preservation (QA accuracy)
- Compression ratio achieved
- Continuation coherence (perplexity)

### 7.2 Integrated Evaluation

```python
def evaluate_cortex(model, test_suite):
    """Full evaluation across all capabilities."""
    results = {}

    # Memory evaluation
    results['memory'] = {
        'entity_f1': evaluate_entity_extraction(model, test_suite.memory),
        'retrieval_mrr': evaluate_retrieval(model, test_suite.memory),
    }

    # Tool evaluation
    results['tool'] = {
        'selection_acc': evaluate_tool_selection(model, test_suite.tool),
        'irrelevance_acc': evaluate_irrelevance_detection(model, test_suite.tool),
        'bfcl_score': evaluate_bfcl(model),
    }

    # Compaction evaluation
    results['compact'] = {
        'rouge_l': evaluate_summarization(model, test_suite.compact),
        'info_preservation': evaluate_info_preservation(model, test_suite.compact),
    }

    # Integrated evaluation
    results['integrated'] = evaluate_full_pipeline(model, test_suite.integrated)

    return results
```

---

## 8. Training Schedule

### Week-by-Week Plan

| Week | Phase | Focus | Checkpoints |
|------|-------|-------|-------------|
| 1-2 | 1 | Knowledge distillation | distill-10B |
| 3 | 1 | Continue distillation | distill-20B |
| 4-5 | 2 | Memory task training | memory-v1 |
| 6-7 | 2 | Tool task training | tool-v1 |
| 8 | 2 | Compaction training | compact-v1 |
| 9-10 | 3 | Multi-task integration | multi-v1 |
| 11 | 4 | DPO alignment | aligned-v1 |
| 12 | - | Evaluation & iteration | final-v1 |

---

## 9. References

1. [MiniLLM Distillation](https://arxiv.org/abs/2306.08543)
2. [Distilling Step-by-Step](https://research.google/blog/distilling-step-by-step-outperforming-larger-language-models-with-less-training-data-and-smaller-model-sizes/)
3. [MINIPLM - ICLR 2025](https://arxiv.org/abs/2410.17215)
4. [Multi-Task Learning Survey](https://arxiv.org/abs/2009.09796)
5. [DPO Paper](https://arxiv.org/abs/2305.18290)
6. [Uncertainty Weighting for MTL](https://arxiv.org/abs/1705.07115)
7. [EWC for Continual Learning](https://arxiv.org/abs/1612.00796)
8. [HRM Training Approach](https://arxiv.org/abs/2506.21734)
