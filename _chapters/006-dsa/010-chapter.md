---
slug: ch-dsa-prototype
title: "6.1 DSA Prototype: Lightning Indexer and Fine-Grained Token Selection"
layout: chapter
---

DeepSeek Sparse Attention (DSA) is the key architectural innovation introduced in DeepSeek-V3.2. The model uses exactly the same architecture as DeepSeek-V3.2-Exp, and compared to DeepSeek-V3.1-Terminus, **the only architectural modification is the introduction of DSA through continued training**.

The core insight is simple but powerful: standard dense attention has O(L²) complexity, which becomes a severe bottleneck at long sequence lengths. DSA reduces this to O(L·k) where k ≪ L is the number of selected tokens.

## Lightning Indexer

The indexer computes an **index score** I_{t,s} between a query token h_t and a preceding token h_s:

```
I_{t,s} = Σ_{j=1}^{H^I} w_{t,j}^I · ReLU(q_{t,j}^I · k_s^I)
```

Where:
- H^I is the number of indexer heads (kept small for efficiency)
- q_{t,j}^I and w_{t,j}^I are derived from query token h_t
- k_s^I is derived from the preceding token h_s
- **ReLU** is chosen as the activation function for throughput optimization

The lightning indexer has a small number of heads and can be implemented in **FP8**, making its computational overhead remarkably low.

## Fine-Grained Token Selection

Given the index scores {I_{t,s}} for each query token, the selection mechanism retrieves only the key-value entries corresponding to the **top-k** index scores:

```
u_t = Attn(h_t, {c_s | I_{t,s} ∈ Top-k(I_{t,:})})
```

Where c_s represents the key-value entry (latent vector in MLA) for token s. The attention output u_t is computed by applying the standard attention mechanism between the query token and the sparsely selected entries only.

## Instantiation Under MLA

For continued training from DeepSeek-V3.1-Terminus, DSA is instantiated based on MLA. At the kernel level, each key-value entry must be shared across multiple queries for computational efficiency, so DSA is implemented under the **MQA (Multi-Query Attention) mode of MLA** — where each latent vector (the key-value entry of MLA) is shared across all query heads of the query token.
