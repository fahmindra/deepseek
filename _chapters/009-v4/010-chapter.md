---
slug: ch-v4-architecture
title: "9.1 Architectural Unification: From V3 to the V4 Frontier"
layout: chapter
---

DeepSeek-V4 represents the structural unification of multiple independent research vectors. Two models are released:

| Model | Total Parameters | Activated Parameters | Context |
|-------|-----------------|---------------------|---------|
| **DeepSeek-V4-Pro** | 1.6T | 49B | 1M tokens |
| **DeepSeek-V4-Flash** | 284B | 13B | 1M tokens |

Both are pre-trained on **more than 32 trillion diverse, high-quality tokens**, followed by comprehensive post-training.

## The Evolutionary Lineage

V4 inherits from its predecessors but aggressively refactors key components:

**From V3:** MoE framework and MTP, but replaces initial dense FFN layers entirely with **Hash-routing MoE layers** and shifts expert affinity activation from Sigmoid to **Sqrt(Softplus(·))**.

**From R1:** Extended "thinking" budget (formalized as Think High and Think Max modes), but replaces mixed RL training with **On-Policy Distillation (OPD)** to fuse domain experts into a unified model.

**From V3.2:** Inherits DSA, but fundamentally overhauls attention with the **hybrid CSA/HCA architecture**.

**From mHC and Engram:** Adopts both papers directly — mHC for stable residual connections and Engram for conditional memory.

## The Efficiency Leap vs. V3.2

At the 1-million-token mark, V4-Pro achieves dramatic efficiency gains:

- **Single-token inference FLOPs:** Only **27%** of V3.2
- **KV cache memory:** Only **10%** of V3.2
- **V4-Flash is even more aggressive:** 10% of FLOPs and 7% of KV cache

Compared to a standard BF16 GQA8 (grouped query attention with 8 heads) baseline, V4's KV cache is approximately **2%** of the size — a 50× reduction.

## Novel Infrastructure

- **Muon optimizer** with hybrid Newton-Schulz iterations for weight orthogonalization
- **FP4 (MXFP4) quantization** during training with FP32 master weights → FP4 storage → FP8 computation
- **Anticipatory Routing** to decouple backbone and routing network updates
- **SwiGLU clamping** to cap gate activations and prevent explosion
- **DSec sandbox:** Rust-based microVM platform for secure RL rollouts
