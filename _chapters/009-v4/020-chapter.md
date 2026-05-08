---
slug: ch-v4-hybrid-attention
title: "9.2 Hybrid Attention: CSA and HCA Interleaved Across Layers"
layout: chapter
---

The hallmark of V4 is its solution to the KV cache bottleneck. Rather than using a single attention mechanism, V4 runs **two distinct attention mechanisms in parallel, interleaved across layers**.

## Compressed Sparse Attention (CSA)

CSA inherits the sparse-selection idea from DSA (V3.2) but runs it over **blocks that are already 4× shorter** than the original sequence.

**Three components:**

1. **Compressor:** Every 4 tokens are collapsed into one compressed KV entry using softmax-gated pooling with a learned positional bias
2. **Lightning Indexer:** Runs in **FP4** precision, using ReLU-scored multi-head dot products to pick top-k compressed blocks per query. The indexer's search space shrinks by 4× compared to DSA
3. **Sliding Window Branch:** Handles the most recent uncompressed tokens for recency

The result: sparse attention over a compressed sequence, where k selected entries represent 4× the context compared to standard DSA.

## Heavily Compressed Attention (HCA)

HCA takes an opposite approach:

1. **Heavy Compressor:** Compresses KV entries by **128×** — every 128 tokens become one compressed entry
2. **Dense Attention:** Since the compressed sequence is so short, every query attends densely to every compressed block (no sparse selection needed)
3. **Sliding Window Branch:** Same as CSA, for recent tokens

## Interleaving Strategy

In V4-Pro's **61-layer stack**:
- Layers 0-1: HCA (initial broad context sweep)
- Layers 2-60: Alternate CSA and HCA
- MTP block at the end: Sliding-window only

This interleaving ensures the model captures both fine-grained token dependencies (via CSA) and document-level semantic sweeps (via HCA).

**Storage precision:**
- Most KV entries: **FP8**
- RoPE dimensions only: **BF16**
- CSA lightning indexer: **FP4**

These storage choices compound with the compression ratios to produce the extraordinary 2% KV cache figure.
