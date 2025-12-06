# Complete Bibliography and Sources

## Overview

This document provides all sources referenced in the Phase 1 research documentation, organized by topic. All links verified as of December 2025.

---

## 1. Hierarchical Reasoning Models

### Primary Papers
- **Hierarchical Reasoning Model (HRM)** - [arXiv:2506.21734](https://arxiv.org/abs/2506.21734)
  - Brain-inspired architecture with Controller/Worker modules
  - 27M parameters achieving complex reasoning
  - Released July 2025

- **HRM Official Repository** - [github.com/sapientinc/HRM](https://github.com/sapientinc/HRM)
  - Open-source implementation
  - Training code and weights available

### Analysis Articles
- [Hierarchical Reasoning Models: Thinking in Layers](https://www.apolo.us/blog-posts/hierarchical-reasoning-models-thinking-in-layers) - Apolo AI
- [The Era of Hierarchical Reasoning Models](https://aipapersacademy.com/hierarchical-reasoning-model/) - AI Papers Academy
- [Beyond Chain-of-Thought: A Look at HRM](https://bdtechtalks.substack.com/p/beyond-chain-of-thought-a-look-at) - BD Tech Talks
- [HRM: The Key to AGI?](https://www.analyticsvidhya.com/blog/2025/09/hierarchical-reasoning-model/) - Analytics Vidhya

---

## 2. Memory Management Systems

### MemGPT / Letta
- **MemGPT Paper** - [arXiv:2310.08560](https://arxiv.org/abs/2310.08560)
  - Virtual context management for LLMs
  - Tiered memory architecture
  - October 2023

- **Letta Framework** - [letta.com](https://www.letta.com/)
  - Production implementation of MemGPT
  - Memory blocks abstraction

- **Memory Blocks Blog** - [letta.com/blog/memory-blocks](https://www.letta.com/blog/memory-blocks)
  - Technical deep-dive on context management

### Cognitive Workspace
- **Cognitive Workspace Paper** - [arXiv:2508.13171](https://arxiv.org/html/2508.13171v1)
  - Active memory management paradigm
  - Beyond traditional RAG approaches

### Context Engineering
- [Context Engineering: Optimizing LLM Memory](https://medium.com/@kuldeep.paul08/context-engineering-optimizing-llm-memory-for-production-ai-agents-6a7c9165a431) - Medium
- [Cutting Through the Noise: Smarter Context Management](https://blog.jetbrains.com/research/2025/12/efficient-context-management/) - JetBrains Research
- [Memory in LLMs: How Machines Learn to Remember](https://medium.com/data-and-beyond/memory-in-llms-how-machines-learn-to-remember-86cceaf42e13) - Medium

---

## 3. Tool Routing and Function Calling

### Benchmarks
- **Berkeley Function Calling Leaderboard** - [gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html](https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html)
  - Industry standard for tool calling evaluation
  - BFCL V3 (multi-turn) and V4 (agentic)

- **BFCL Dataset** - [huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard)

### Routing Research
- **RouteLLM** - [github.com/lm-sys/RouteLLM](https://github.com/lm-sys/RouteLLM)
  - LLM routing framework
  - Up to 85% cost reduction

- **BEST-Route** (Microsoft) - [github.com/microsoft/best-route-llm](https://github.com/microsoft/best-route-llm)
  - ICML 2025 paper
  - 60% cost reduction with <1% performance drop

- **LLM Routing Strategies** - [aws.amazon.com/blogs/machine-learning/multi-llm-routing-strategies](https://aws.amazon.com/blogs/machine-learning/multi-llm-routing-strategies-for-generative-ai-applications-on-aws/)

### Tool Learning
- **ToolACE Paper** - [arXiv:2409.00920](https://arxiv.org/html/2409.00920v1)
  - ICLR 2025
  - 26,507 diverse APIs dataset
  - Self-evolution synthesis

- **xLAM (Salesforce)** - [github.com/SalesforceAIResearch/xLAM](https://github.com/SalesforceAIResearch/xLAM)
  - Large Action Models
  - APIGen-MT for multi-turn data

- **Gorilla LLM** - [github.com/ShishirPatil/gorilla](https://github.com/ShishirPatil/gorilla)
  - Training LLMs for function calls

- [Optimizing Tool Selection for LLM Workflows](https://viksit.substack.com/p/optimizing-tool-selection-for-llm) - Substack

---

## 4. Context Compression

### Research Papers
- **Acon: Agent Context Compression** - [arXiv:2510.00615](https://arxiv.org/html/2510.00615v1)
  - 26-54% memory reduction
  - Distillation to smaller models

- **LLMLingua** (Microsoft) - [microsoft.com/research/blog/llmlingua](https://www.microsoft.com/en-us/research/blog/llmlingua-innovating-llm-efficiency-with-prompt-compression/)
  - Up to 20x prompt compression
  - LongLLMLingua for long contexts

- **Pretraining Context Compressor** - [aclanthology.org/2025.acl-long.1394.pdf](https://aclanthology.org/2025.acl-long.1394.pdf)

### Industry Articles
- [How We Extended LLM Conversations by 10x](https://dev.to/amitksingh1490/how-we-extended-llm-conversations-by-10x-with-intelligent-context-compaction-4h0a) - DEV Community
- [KVzip: AI Tech Compresses LLM Memory 3-4x](https://techxplore.com/news/2025-11-ai-tech-compress-llm-chatbot.html) - TechXplore
- [LLM Chat History Summarization Guide 2025](https://mem0.ai/blog/llm-chat-history-summarization-guide-2025) - Mem0
- [Compressing Context](https://factory.ai/news/compressing-context) - Factory.ai

---

## 5. Knowledge Graphs and LLMs

### Surveys and Papers
- **LLM+Graph Workshop** - [VLDB 2025](https://www.vldb.org/2025/Workshops/VLDB-Workshops-2025/LLM+Graph/LLMGraph-1.pdf)
- **Unifying LLMs and Knowledge Graphs** - [openproceedings.org/2025/conf/edbt/paper-T4.pdf](https://www.openproceedings.org/2025/conf/edbt/paper-T4.pdf)
- **LLM-empowered Knowledge Graph Construction** - [arXiv:2510.20345](https://arxiv.org/html/2510.20345v1)

### Resources
- **Awesome Graph-LLM** - [github.com/XiaoxinHe/Awesome-Graph-LLM](https://github.com/XiaoxinHe/Awesome-Graph-LLM)
  - Curated collection of Graph+LLM papers

- [From LLMs to Knowledge Graphs in 2025](https://medium.com/@claudiubranzan/from-llms-to-knowledge-graphs-building-production-ready-graph-systems-in-2025-2b4aff1ec99a) - Medium
- [Enhancing LLMs with Knowledge Graphs](https://www.datacamp.com/blog/knowledge-graphs-and-llms) - DataCamp

---

## 6. Small Language Model Training

### Architecture References
- **TinyLlama** - [github.com/jzhang38/TinyLlama](https://github.com/jzhang38/TinyLlama)
  - 1.1B parameters trained on 3T tokens
  - Architecture and training recipe

- **SmolLM Training Playbook** - [gist.github.com/jph00/3c97a2c6c5075c4e7b98faae634b033a](https://gist.github.com/jph00/3c97a2c6c5075c4e7b98faae634b033a)
  - HuggingFace's training methodology
  - Depth over width principle

- **Inheritune** - [arXiv:2404.08634](https://arxiv.org/html/2404.08634v1)
  - 1.5B model from 1B tokens
  - <12 hours on single A6000

- **MINIPLM** - [arXiv:2410.17215](https://arxiv.org/pdf/2410.17215)
  - ICLR 2025
  - Offline knowledge distillation

### Surveys
- **Survey of Small Language Models** - [arXiv:2410.20011](https://arxiv.org/html/2410.20011v1)
- **Small Language Models Can Still Pack a Punch** - [arXiv:2501.05465](https://arxiv.org/html/2501.05465v1)

---

## 7. Mamba and State Space Models

### Core Papers
- **Mamba: Linear-Time Sequence Modeling** - [OpenReview](https://openreview.net/forum?id=tEYskw1VY2)
- **Mamba-360 Survey** - [arXiv:2404.16112](https://arxiv.org/html/2404.16112v1)

### Analysis
- [Visual Guide to Mamba and SSMs](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mamba-and-state) - Maarten Grootendorst
- [Mamba Explained](https://thegradient.pub/mamba-explained/) - The Gradient
- [SSM vs Transformer Tradeoffs](https://goombalab.github.io/blog/2025/tradeoffs/) - Goomba Lab
- [Transformers Better at Copying](https://kempnerinstitute.harvard.edu/research/deeper-learning/repeat-after-me-transformers-are-better-than-state-space-models-at-copying/) - Harvard Kempner Institute

---

## 8. Distributed Training

### DeepSpeed
- [DeepSpeed ZeRO Overview](https://huggingface.co/blog/zero-deepspeed-fairscale) - HuggingFace
- [FSDP vs DeepSpeed](https://huggingface.co/docs/accelerate/en/concept_guides/fsdp_and_deepspeed) - HuggingFace Accelerate
- [Multi-GPU Training with DeepSpeed](https://medium.com/@zaiinn440/multi-gpu-training-of-70b-llm-with-deepspeed-and-fsdp-qlora-cb738a2a2229) - Medium

### PyTorch FSDP
- [PyTorch FSDP Introduction](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/) - PyTorch Blog
- [FSDP Tutorial](https://huggingface.co/blog/pytorch-fsdp) - HuggingFace

### General
- [Distributed Training Primer](https://sumanthrh.com/post/distributed-and-efficient-finetuning/) - Sumanth's Blog
- [Benchmarking Multi-GPU Training](https://medium.com/@savyasachi.thati/benchmarking-advanced-multi-gpu-training-strategies-20c9675003db) - Medium

---

## 9. Knowledge Distillation

### Papers
- **MiniLLM** - [arXiv:2306.08543](https://arxiv.org/abs/2306.08543)
  - White-box LLM distillation

- **Distilling Step-by-Step** - [research.google/blog/distilling-step-by-step](https://research.google/blog/distilling-step-by-step-outperforming-larger-language-models-with-less-training-data-and-smaller-model-sizes/)
  - 770M outperforming 540B PaLM
  - Rationale extraction

- **Survey on KD for LLMs** - [ACM TIST](https://dl.acm.org/doi/10.1145/3699518)

### Resources
- **Awesome KD for LLMs** - [github.com/Tebmer/Awesome-Knowledge-Distillation-of-LLMs](https://github.com/Tebmer/Awesome-Knowledge-Distillation-of-LLMs)
- [LLM Distillation Guide](https://snorkel.ai/blog/llm-distillation-demystified-a-complete-guide/) - Snorkel AI
- [Knowledge Distillation Deep Dive](https://zilliz.com/learn/knowledge-distillation-from-large-language-models-deep-dive) - Zilliz

---

## 10. Synthetic Data Generation

### Methods
- **Self-Instruct** - [huggingface.co/blog/davanstrien/self-instruct](https://huggingface.co/blog/davanstrien/self-instruct)
- **Evol-Instruct (WizardLM)** - [arXiv:2304.12244](https://arxiv.org/abs/2304.12244)

### Platforms
- **NVIDIA Nemotron-4** - [blogs.nvidia.com/blog/nemotron-4-synthetic-data-generation-llm-training](https://blogs.nvidia.com/blog/nemotron-4-synthetic-data-generation-llm-training/)
- **InstructLab (IBM)** - [redhat.com/blog/instructlab-synthetic-data-generation](https://www.redhat.com/en/blog/how-instructlabs-synthetic-data-generation-enhances-llms)

### Resources
- **Awesome LLM Synthetic Data** - [github.com/wasiahmad/Awesome-LLM-Synthetic-Data](https://github.com/wasiahmad/Awesome-LLM-Synthetic-Data)
- [Synthetic Data Survey](https://arxiv.org/html/2406.15126v1) - arXiv
- [Synthetic Data Definitive Guide](https://www.confident-ai.com/blog/the-definitive-guide-to-synthetic-data-generation-using-llms) - Confident AI

---

## 11. Multi-Task Learning

### Surveys
- **MTL with DNNs Survey** - [arXiv:2009.09796](https://arxiv.org/abs/2009.09796)
- **MTL Overview** - [ruder.io/multi-task](https://www.ruder.io/multi-task/)

### Resources
- **Awesome Multi-Task Learning** - [github.com/thuml/awesome-multi-task-learning](https://github.com/thuml/awesome-multi-task-learning)
- [Multi-Task Learning Guide](https://www.v7labs.com/blog/multi-task-learning-guide) - V7 Labs

---

## 12. Datasets

### Function Calling
- [BFCL Dataset](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard) - HuggingFace
- [ToolACE](https://openreview.net/forum?id=8EB8k6DdCU) - ICLR 2025

### Pretraining
- [SlimPajama-627B](https://huggingface.co/datasets/cerebras/SlimPajama-627B) - HuggingFace

### Summarization
- [SAMSum](https://huggingface.co/datasets/samsum) - Dialog Summarization
- [DialogSum](https://huggingface.co/datasets/knkarthick/dialogsum) - Dialog Summarization

### Datasets Collection
- [LLM Datasets Repository](https://github.com/mlabonne/llm-datasets) - mlabonne

---

## 13. Additional Resources

### Model Context Protocol
- [MCP + Cognee](https://www.cognee.ai/blog/deep-dives/model-context-protocol-cognee-llm-memory-made-simple) - Cognee
- [Beyond the Bubble: Context-Aware Memory 2025](https://www.tribe.ai/applied-ai/beyond-the-bubble-how-context-aware-memory-systems-are-changing-the-game-in-2025) - Tribe AI

### LLM Compression
- **Awesome LLM Compression** - [github.com/HuangOwen/Awesome-LLM-Compression](https://github.com/HuangOwen/Awesome-LLM-Compression)
- [LLMs Can Invent Their Own Compression](https://www.rajan.sh/llm-compression) - Rajan Agarwal

---

## Citation Format

When referencing these sources in academic work:

```bibtex
@article{hrm2025,
  title={Hierarchical Reasoning Model},
  author={Sapient Intelligence},
  journal={arXiv preprint arXiv:2506.21734},
  year={2025}
}

@inproceedings{memgpt2023,
  title={MemGPT: Towards LLMs as Operating Systems},
  author={Packer, Charles et al.},
  booktitle={arXiv preprint arXiv:2310.08560},
  year={2023}
}

@inproceedings{toolace2025,
  title={ToolACE: Winning the Points of LLM Function Calling},
  author={Various},
  booktitle={ICLR 2025},
  year={2025}
}
```
