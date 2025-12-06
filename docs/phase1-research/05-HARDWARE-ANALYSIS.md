# Hardware Analysis: 8x RTX 3090 Training Setup

## Overview

This document analyzes the training capabilities of your hardware setup (8x RTX 3090 + 512GB DDR4) for training a 1.5B parameter model.

---

## 1. Hardware Specifications

### GPU Configuration

| Spec | Value |
|------|-------|
| **GPU Model** | NVIDIA RTX 3090 |
| **GPU Count** | 8 |
| **VRAM per GPU** | 24 GB GDDR6X |
| **Total VRAM** | 192 GB |
| **Memory Bandwidth** | 936 GB/s per GPU |
| **FP16 Performance** | ~35.6 TFLOPS per GPU |
| **Total FP16** | ~284 TFLOPS |
| **Interconnect** | PCIe 4.0 (no NVLink) |

### System Memory

| Spec | Value |
|------|-------|
| **RAM** | 512 GB DDR4 |
| **Purpose** | CPU offloading, data loading |

### Key Limitation: No NVLink

The RTX 3090 consumer cards lack NVLink for direct GPU-to-GPU communication. All inter-GPU communication goes through PCIe, which is significantly slower:
- **NVLink**: 600 GB/s bidirectional
- **PCIe 4.0 x16**: 32 GB/s per direction

**Implication**: Gradient synchronization will be slower than datacenter setups. Must use efficient sharding strategies.

---

## 2. Memory Requirements

### 1.5B Parameter Model

**Memory Components (FP16 Training)**:

| Component | Calculation | Memory |
|-----------|-------------|--------|
| **Model Parameters** | 1.5B × 2 bytes | 3.0 GB |
| **Gradients** | 1.5B × 2 bytes | 3.0 GB |
| **Optimizer State (AdamW)** | 1.5B × 8 bytes | 12.0 GB |
| **Activations** (batch=1, seq=2048) | ~2 GB per layer × 28 | ~56 GB |
| **Total (Single GPU)** | | **~74 GB** |

**Problem**: 74 GB far exceeds 24 GB per GPU.

**Solution**: Distributed training with sharding.

---

## 3. Distributed Training Strategies

### 3.1 DeepSpeed ZeRO-3 (Recommended)

**Source**: [huggingface.co/docs/accelerate/concept_guides/fsdp_and_deepspeed](https://huggingface.co/docs/accelerate/en/concept_guides/fsdp_and_deepspeed)

ZeRO Stage 3 shards:
- Model parameters across GPUs
- Gradients across GPUs
- Optimizer states across GPUs

**Memory per GPU with ZeRO-3**:

| Component | Calculation | Memory per GPU |
|-----------|-------------|----------------|
| **Parameters** | 3.0 GB ÷ 8 | 0.375 GB |
| **Gradients** | 3.0 GB ÷ 8 | 0.375 GB |
| **Optimizer** | 12.0 GB ÷ 8 | 1.5 GB |
| **Activations** | (with checkpointing) | ~4-6 GB |
| **KV Cache** | (variable) | ~2-4 GB |
| **Working Memory** | | ~2-3 GB |
| **Total** | | **~10-15 GB** |

**Result**: Comfortably fits in 24 GB per GPU with room for larger batches.

### 3.2 DeepSpeed Configuration

```json
{
    "bf16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "none"
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": true
    },
    "gradient_accumulation_steps": 8,
    "gradient_clipping": 1.0,
    "steps_per_print": 100,
    "train_micro_batch_size_per_gpu": 2,
    "wall_clock_breakdown": false
}
```

### 3.3 PyTorch FSDP Alternative

**Source**: [pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/)

FSDP (Fully Sharded Data Parallel) is PyTorch's native implementation, similar to ZeRO-3.

**Advantages**:
- Native PyTorch integration
- Simpler configuration
- Good community support

**Configuration**:
```python
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    CPUOffload,
)

model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    cpu_offload=CPUOffload(offload_params=True),
    mixed_precision=MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    ),
)
```

### 3.4 Hybrid Sharding for PCIe Systems

Since you don't have NVLink, consider **ZeRO++** with hierarchical partitioning:

```json
{
    "zero_optimization": {
        "stage": 3,
        "zero_hpz_partition_size": 8,
        "zero_quantized_weights": true,
        "zero_quantized_gradients": true
    }
}
```

This reduces inter-GPU communication by keeping some redundancy within the node.

---

## 4. Training Speed Estimation

### 4.1 Theoretical Throughput

**Assumptions**:
- 1.5B parameters
- 8x RTX 3090
- 2048 token sequence length
- Batch size: 16 (2 per GPU × 8 GPUs)
- BF16/FP16 mixed precision

**Calculation**:
```
FLOPs per token = 6 × parameters = 6 × 1.5B = 9 TFLOPs
FLOPs per step = 9 TFLOPs × 2048 tokens × 16 batch = 294 PFLOPs

Hardware capacity = 284 TFLOPS (all 8 GPUs)
Efficiency estimate = 40% (PCIe bottleneck, synchronization)
Effective TFLOPS = 284 × 0.4 = 113.6 TFLOPS

Time per step = 294 PFLOPs ÷ 113.6 TFLOPS = ~2.6 seconds
Tokens per second = 16 × 2048 ÷ 2.6 = ~12,600 tokens/sec
```

### 4.2 Realistic Estimates

Based on similar setups (TinyLlama, SmolLM training logs):

| Metric | Conservative | Optimistic |
|--------|--------------|------------|
| **Tokens/second** | 8,000 | 15,000 |
| **Tokens/hour** | 28.8M | 54M |
| **Tokens/day** | 691M | 1.3B |
| **Time for 50B tokens** | 72 days | 38 days |
| **Time for 100B tokens** | 145 days | 77 days |

### 4.3 Optimization Strategies

**To maximize throughput**:

1. **Gradient Checkpointing**: Trade compute for memory
   ```python
   model.gradient_checkpointing_enable()
   ```

2. **Flash Attention 2**: Faster attention computation
   ```python
   from flash_attn import flash_attn_func
   ```

3. **Larger Batch Sizes**: Increase with gradient accumulation
   ```
   effective_batch = micro_batch × grad_accum × num_gpus
   effective_batch = 2 × 8 × 8 = 128
   ```

4. **Data Loading Optimization**: Prevent CPU bottleneck
   ```python
   DataLoader(
       dataset,
       num_workers=8,
       pin_memory=True,
       prefetch_factor=4
   )
   ```

---

## 5. Memory Optimization Techniques

### 5.1 Activation Checkpointing

Instead of storing all activations, recompute during backward pass:

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False}
)
```

**Trade-off**: ~33% more compute, ~60% less activation memory.

### 5.2 CPU Offloading

With 512GB DDR4, you can offload optimizer states:

```json
{
    "zero_optimization": {
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        }
    }
}
```

**Trade-off**: ~10-20% slower, but allows larger models/batches.

### 5.3 Mixed Precision Training

Use BF16 if supported, otherwise FP16 with loss scaling:

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    bf16=True,  # or fp16=True
    bf16_full_eval=True
)
```

---

## 6. Recommended Training Configuration

### 6.1 Hardware Setup

```bash
# Environment setup
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_P2P_DISABLE=1  # Disable P2P for PCIe systems
export NCCL_IB_DISABLE=1   # No InfiniBand
```

### 6.2 Training Script Configuration

```python
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer

# Accelerate config for 8x 3090
accelerator = Accelerator(
    mixed_precision="bf16",
    gradient_accumulation_steps=8,
)

training_args = TrainingArguments(
    output_dir="./cortex-1.5b",

    # Batch configuration
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,  # Effective batch = 128

    # Precision
    bf16=True,

    # Optimization
    learning_rate=3e-4,
    warmup_steps=2000,
    max_steps=500000,
    lr_scheduler_type="cosine",

    # Memory optimization
    gradient_checkpointing=True,

    # Logging
    logging_steps=10,
    save_steps=1000,
    eval_steps=1000,

    # DeepSpeed
    deepspeed="./ds_config.json",
)
```

### 6.3 Launch Command

```bash
accelerate launch \
    --num_processes 8 \
    --num_machines 1 \
    --mixed_precision bf16 \
    --use_deepspeed \
    --deepspeed_config_file ds_config.json \
    train.py
```

---

## 7. Power and Cooling Considerations

### Power Requirements

| Component | Power |
|-----------|-------|
| RTX 3090 × 8 | 350W × 8 = 2,800W |
| CPU + System | ~300W |
| **Total** | **~3,100W** |

**Requirements**:
- 2x 1600W+ PSUs (for redundancy and capacity)
- 30A 240V circuit or 2x 20A 120V circuits
- UPS recommended for training stability

### Cooling

At 2.8kW GPU heat output:
- Adequate case airflow or open-air mining frame
- Room A/C to handle ~10,000 BTU/hr
- Monitor GPU temperatures (target <80°C sustained)

---

## 8. Training Timeline Estimates

### Scenario 1: Minimal Training (Quick Prototype)

| Phase | Tokens | Time |
|-------|--------|------|
| Base pretraining | 10B | 8-14 days |
| Task fine-tuning | 5B | 4-7 days |
| **Total** | **15B** | **12-21 days** |

### Scenario 2: Full Training (Production Quality)

| Phase | Tokens | Time |
|-------|--------|------|
| Base pretraining | 50B | 38-72 days |
| Task fine-tuning | 20B | 15-29 days |
| Alignment | 5B | 4-7 days |
| **Total** | **75B** | **57-108 days** |

### Scenario 3: Knowledge Distillation (Faster)

| Phase | Tokens | Time |
|-------|--------|------|
| Distillation from larger model | 20B | 15-29 days |
| Task fine-tuning | 10B | 8-14 days |
| **Total** | **30B** | **23-43 days** |

---

## 9. Cost Analysis

### Electricity

```
Power consumption: 3.1 kW
Running time: 60 days (average scenario)
Hours: 1,440 hours

kWh = 3.1 × 1,440 = 4,464 kWh
Cost at $0.12/kWh = $535
Cost at $0.20/kWh = $893
```

### Comparison to Cloud

| Option | Cost for 60 days |
|--------|------------------|
| **Your Setup** | $500-900 (electricity only) |
| **8x A100 (Lambda)** | ~$70,000 |
| **8x H100 (Lambda)** | ~$140,000 |

**Conclusion**: Your hardware setup is extremely cost-effective for this project.

---

## 10. Monitoring and Reliability

### GPU Monitoring

```bash
# Real-time monitoring
watch -n 1 nvidia-smi

# Log temperatures and utilization
nvidia-smi --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,memory.used --format=csv -l 60 >> gpu_log.csv
```

### Training Monitoring

```python
# Weights & Biases integration
import wandb

wandb.init(project="cortex-1.5b")
trainer = Trainer(
    ...
    report_to="wandb"
)
```

### Checkpointing Strategy

- Save checkpoints every 1,000 steps
- Keep last 5 checkpoints
- Save to SSD with backup to NAS/cloud

---

## 11. Summary

Your 8x RTX 3090 setup is **well-suited** for training a 1.5B parameter model:

| Aspect | Assessment |
|--------|------------|
| **VRAM** | Sufficient with ZeRO-3/FSDP |
| **RAM** | Excellent for offloading |
| **Speed** | 2-4 weeks for prototype, 2-3 months for full |
| **Cost** | Highly economical vs cloud |
| **Reliability** | Good with proper monitoring |

**Recommendation**: Proceed with DeepSpeed ZeRO-3, gradient checkpointing, and CPU offloading for optimizer states.

---

## 12. References

1. [DeepSpeed ZeRO](https://huggingface.co/blog/zero-deepspeed-fairscale)
2. [FSDP vs DeepSpeed](https://huggingface.co/docs/accelerate/en/concept_guides/fsdp_and_deepspeed)
3. [PyTorch FSDP](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/)
4. [Distributed Training Primer](https://sumanthrh.com/post/distributed-and-efficient-finetuning/)
5. [Training 70B with DeepSpeed](https://medium.com/@zaiinn440/multi-gpu-training-of-70b-llm-with-deepspeed-and-fsdp-qlora-cb738a2a2229)
