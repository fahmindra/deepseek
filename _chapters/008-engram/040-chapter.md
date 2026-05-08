---
slug: ch-engram-results
title: "8.4 Engram-27B: Pre-training Results and Analysis"
layout: chapter
---

## Large-Scale Pre-training

Guided by the U-shaped allocation law, Engram-27B was trained and compared against a strictly iso-parameter and iso-FLOPs MoE baseline.

### Knowledge Retrieval Gains

As expected for a memory module:
- **MMLU: +3.4**
- **CMMLU: +4.0**
- **MMLU-Pro: +1.8**

### Surprising General Reasoning Gains

The more striking results come from domains where memory capacity was not intuitively the bottleneck:
- **BBH: +5.0** (general reasoning)
- **ARC-Challenge: +3.7** (science reasoning)
- **DROP: +3.3** (reading comprehension)
- **HumanEval: +3.0** (code generation)
- **MATH: +2.4** (mathematical reasoning)
- **GSM8K: +2.2** (grade-school math)

### Long-Context Performance

By delegating local dependencies to lookups, Engram frees up attention capacity for global context:
- **Multi-Query NIAH:** 97.0 vs. 84.2 (baseline)
- **Variable Tracking:** 89.0 vs. 77.0 (baseline)

The LongPPL and RULER benchmarks both show substantial improvements.

## Mechanistic Analysis

Using LogitLens and CKA (Centered Kernel Alignment), the analysis reveals **why** Engram helps beyond knowledge retrieval:

1. **Relieves Early Layers:** Engram offloads static knowledge reconstruction from the backbone's early layers
2. **Increases Effective Depth:** By freeing layers from static pattern matching, more sequential depth becomes available for complex reasoning
3. **Frees Attention Capacity:** Attention heads can focus on global context rather than local dependencies, explaining the long-context gains

## System Efficiency

Engram's deterministic addressing enables **runtime prefetching** from host memory. Since retrieval indices depend solely on input tokens (not hidden states), the system can asynchronously fetch embeddings before the module is reached:

- Offloading a **100B-parameter table** to host memory incurs **<3% overhead**
- Training uses All-to-All communication to gather active rows across sharded GPUs
- The Zipfian distribution of N-gram patterns enables a **Multi-Level Cache Hierarchy** (GPU HBM → Host DRAM → NVMe SSD)

This decoupling of compute and memory means Engram effectively **bypasses GPU HBM constraints**, enabling aggressive parameter expansion without proportional hardware cost.
