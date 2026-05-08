---
slug: ch-dsa-evaluation
title: "6.3 Parity Evaluation and Inference Cost Reduction"
layout: chapter
---

## Standard Benchmark Performance

Evaluated in September 2025, DeepSeek-V3.2-Exp shows similar performance to DeepSeek-V3.1-Terminus on a diverse suite of benchmarks. Despite the introduction of sparse attention, **there is no substantial performance degradation on either short- or long-context tasks**.

## Human Preference (ChatbotArena)

Both V3.1-Terminus and V3.2-Exp share an identical post-training strategy, and their Elo scores evaluated on November 10, 2025, are closely matched. This confirms the sparse attention mechanism does not degrade user-facing quality.

## Long Context Evaluation

Independent long-context evaluations using previously unseen test sets confirm DSA's effectiveness:

- **AA-LCR benchmark:** V3.2-Exp scores **4 points higher** than V3.1-Terminus in reasoning mode
- **Fiction.liveBench:** V3.2-Exp consistently outperforms V3.1-Terminus across multiple metrics

This evidence indicates that the base checkpoint with DSA not only preserves but potentially improves long-context capabilities.

## Inference Cost Reduction

The core advantage is captured in the complexity reduction:

- **Before (dense attention):** O(L²)
- **After (DSA):** O(L·k) where k ≪ L

While the lightning indexer still has O(L²) complexity, it requires **far less computation** than the full MLA in DeepSeek-V3.1-Terminus. Combined with optimized implementation:

- **Prefilling:** Costs scale linearly with position for long sequences, compared to quadratic for dense attention
- **Decoding:** Significant speedup across all sequence positions

For **short-sequence prefilling**, a specially implemented masked MHA mode simulates DSA for higher efficiency. These costs are estimated from benchmarking the actual service deployed on **H800 GPUs** at a rental price of $2 USD per GPU hour.

## Key Insight

DSA proves that sparse attention can be introduced through continued pre-training with negligible quality loss, while fundamentally changing the computational complexity profile. The two-stage training — dense warm-up for the indexer followed by sparse adaptation of the full model — provides a practical recipe for converting any dense model to sparse attention.
