---
slug: ch-mhc-infrastructure
title: "7.3 Efficient Infrastructure: Kernel Fusion, Recomputing, and DualPipe"
layout: chapter
---

While mHC solves the numerical stability problem, system-level efficiency requires careful engineering. The paper addresses three infrastructure challenges.

## Memory Access Analysis

The per-token memory access overhead for standard HC is proportional to the expansion rate n:

| Method | Read (elements) | Write (elements) |
|--------|----------------|------------------|
| Standard Residual | 2C | C |
| Hyper-Connections | (5n+1)C + n²+2n | (3n+1)C + n²+2n |

For n=4, this represents approximately a **20× increase in I/O** compared to standard residual connections.

## Kernel Fusion

To mitigate the memory access bottleneck, mHC employs **fused kernels** implemented with TileLang — a domain-specific language for writing high-performance GPU kernels. Fusion combines multiple operations (RMSNorm, linear projections, matrix mixing) into single GPU kernel launches, drastically reducing the round-trips to global memory.

## Selective Recomputing

Rather than storing all intermediate activations for backpropagation, mHC uses **selective recomputing** (gradient checkpointing) to trade computation for memory. Key activations are discarded during the forward pass and recomputed during backpropagation, reducing the GPU memory footprint.

## Overlapping Communication in DualPipe

DualPipe is DeepSeek's training pipeline scheduler (first introduced in V3) that overlaps computation and communication to maximize GPU utilization. For mHC, the widened residual stream (n×C dimensions) requires proportional communication during pipeline parallelism.

mHC extends DualPipe to:
- Carefully schedule communication of the expanded residual stream
- Overlap communication with computation to hide latency
- Keep communication bubbles minimal despite the n-fold increase in data transfer

## Runtime Overhead

The combined optimizations result in only a **6.7% additional time overhead** when using expansion rate n=4, making mHC practical for large-scale deployment despite the architectural complexity.
