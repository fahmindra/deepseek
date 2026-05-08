---
slug: ch-engram-intro
title: "8.1 Two Axes of Sparsity: Computation and Memory"
layout: chapter
---

While Mixture-of-Experts (MoE) scales capacity via conditional computation, Transformers lack a native primitive for knowledge lookup. Current LLMs are forced to **simulate retrieval through computation** — resolving common multi-token entities by consuming multiple early layers of attention and feed-forward networks. This process amounts to an expensive runtime reconstruction of what is essentially a static lookup table.

## The Linguistic Duality

Language modeling entails two qualitatively different sub-tasks:

1. **Compositional Reasoning:** Demands deep, dynamic computation through attention and feed-forward layers
2. **Knowledge Retrieval:** Named entities, formulaic patterns, local dependencies — static, highly stereotyped, naturally represented as computationally inexpensive lookups

The effectiveness of classical N-gram models in capturing local dependencies implies these regularities should be retrievable through O(1) lookup rather than O(L²) attention computation.

## Two Complementary Axes of Sparsity

Engram introduces **conditional memory** as a complementary sparsity axis to MoE's conditional computation:

| Axis | Mechanism | What it does |
|------|-----------|--------------|
| **Conditional Computation** (MoE) | Sparse expert routing | Activates a subset of experts per token |
| **Conditional Memory** (Engram) | Sparse lookup operations | Retrieves static embeddings for fixed knowledge |

In MoE, inactive experts don't consume compute (free parameter scaling). In Engram, **unretrieved embeddings don't consume compute** — enabling massive memory scaling without proportional FLOPs increase.

## Modernizing N-gram Embeddings

Engram revisits classic N-gram embeddings with four modern adaptations:

1. **Tokenizer Compression:** Maps semantically equivalent tokens to canonical IDs (23% vocabulary reduction)
2. **Multi-Head Hashing:** Uses K distinct hash heads per N-gram order to mitigate collisions
3. **Context-Aware Gating:** Dynamic modulation of retrieved embeddings based on current hidden state
4. **Multi-Branch Integration:** Designed for mHC architectures with branch-specific gating
