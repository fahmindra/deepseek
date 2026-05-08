---
slug: ch-dsa-training
title: "6.2 DSA Continued Pre-Training: Warm-up and Sparse Stages"
layout: chapter
---

Starting from a base checkpoint of DeepSeek-V3.1-Terminus (whose context length has been extended to 128K), continued pre-training creates DeepSeek-V3.2 through two distinct stages.

## Stage 1: Dense Warm-up

The warm-up stage initializes the lightning indexer while keeping dense attention and **freezing all model parameters except for the lightning indexer**.

To align the indexer outputs with the main attention distribution: for the t-th query token, the main attention scores are aggregated by summing across all attention heads, then L1-normalized along the sequence dimension to produce a target distribution p_{t,:}.

The training objective uses **KL-divergence loss**:

```
ℒ^I = Σ_t D_KL(p_{t,:} ‖ Softmax(I_{t,:}))
```

**Training details:**
- Learning rate: 10⁻³
- Duration: 1000 steps
- Each step: 16 sequences of 128K tokens
- Total tokens: **2.1 billion**

## Stage 2: Sparse Training

Following indexer warm-up, the fine-grained token selection mechanism is introduced and **all model parameters** are optimized to adapt to the sparse pattern.

The indexer continues to align with the main attention distribution, but considers **only the selected token set** S_t:

```
ℒ^I = Σ_t D_KL(p_{t,S_t} ‖ Softmax(I_{t,S_t}))
```

A critical implementation detail: **the indexer input is detached from the computational graph** for separate optimization. The training signal for the indexer comes only from ℒ^I, while the main model is optimized according only to the language modeling loss.

**Training details:**
- Learning rate: 7.3×10⁻⁶
- Duration: 15,000 steps
- Each step: 480 sequences of 128K tokens
- Tokens selected per query: **2048** (k=2048)
- Total tokens: **943.7 billion**

This two-stage approach — warm-up for the indexer followed by full sparse adaptation — proves remarkably effective, with the resulting model showing parity performance on both short and long-context tasks while dramatically reducing attention complexity.
