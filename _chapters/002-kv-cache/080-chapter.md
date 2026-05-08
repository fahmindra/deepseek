---
slug: ch08-the-performance-vs-memory-trade-off
title: "2.8 The performance vs. memory trade-off"
layout: chapter
---

We have now explored the first generation of solutions to the KV Cache memory crisis: Multi-Query Attention (MQA) and Grouped-Query Attention (GQA). Both techniques offer a significant reduction in the memory footprint of the KV cache, making it possible to run larger models with longer context lengths on existing hardware.

However, they both operate on the same fundamental principle: they save memory by reducing the number of unique Key and Value projections.

*   **MQA** is the most extreme case, collapsing n unique K/V heads down to just one.
*   **GQA** offers a more moderate compromise, collapsing n heads into g groups.

While effective, this is ultimately a trade-off. We are sacrificing the expressive power that comes from having fully independent, specialized attention heads in order to save memory. GQA allows us to choose a more palatable point on the performance-vs-memory curve, but it doesn't change the curve itself. We are still forced to choose between maximum performance (MHA) and maximum memory efficiency (MQA), or a compromise in between (GQA).

This unresolved tension is what makes the DeepSeek architecture so innovative. The developers asked a different question: instead of reducing the number of heads, can we make the information within each head more compact? Can we compress the Key and Value matrices themselves?

This shift in thinking from **reducing heads** to **compressing information** is the conceptual leap that leads directly to Multi-Head Latent Attention, which we’ll discuss in chapter 3. It represents a fundamentally new approach to solving the KV Cache bottleneck, one that aims to preserve the full expressive power of MHA while still achieving dramatic memory savings.