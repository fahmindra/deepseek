---
slug: ch09-summary
title: "2.9 Summary"
layout: chapter
---

*   Autoregressive generation, where each new token is appended to the input, results in the re-processing of the entire sequence at every step in a naive implementation.
*   This repeated computation leads to a quadratic (O(n²)) complexity problem, which makes generating long sequences of text computationally impractical.
*   The Key-Value (KV) Cache optimizes inference by storing the Key and Value matrices of past tokens, avoiding redundant calculations and transforming the process into a linear-time (O(n)) operation.
*   While the KV Cache dramatically accelerates computation, it introduces a severe memory bottleneck, as its size grows proportionally with sequence length, number of layers, and attention heads.
*   Multi-Query Attention (MQA) drastically reduces the KV cache's memory footprint by forcing all attention heads to share a single Key and Value projection, but this significantly degrades model performance by preventing head specialization.
*   Grouped-Query Attention (GQA) provides a tunable compromise by having groups of attention heads share Key and Value projections, allowing a balance between memory savings and model expressivity.
*   Architectures like MQA and GQA fundamentally operate by reducing the number of unique Key/Value pairs, establishing an inherent trade-off between memory efficiency and the expressive power of the model.