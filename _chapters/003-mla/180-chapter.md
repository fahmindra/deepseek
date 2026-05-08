---
slug: ch18-summary
title: "3.18 Summary"
layout: chapter
---

*   **Multi-Head Latent Attention (MLA)** breaks the performance-memory trade-off by caching a compressed latent representation of Keys and Values, which is then decompressed on-the-fly to reconstruct unique, full-sized matrices for each head during the attention calculation.
*   Applying positional information by adding it to the initial token embeddings pollutes their semantic meaning before they are processed by the attention mechanism.
*   **Rotary Positional Encoding (RoPE)** solves this by rotating the Query and Key vectors directly within the attention block, which injects positional information later in the process and preserves the vectors' original learned magnitudes.
*   The efficiency of MLA depends on a mathematical "**absorption trick**" that requires position-agnostic weights, which is fundamentally incompatible with the position-dependent transformations of standard RoPE.
*   The **Decoupled RoPE architecture** resolves this conflict by separating attention into two parallel streams: a content path that uses a cache-efficient MLA, and a position path that handles RoPE. The final attention scores are the **sum of the outputs from both paths.**