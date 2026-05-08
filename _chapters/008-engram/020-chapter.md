---
slug: ch-engram-architecture
title: "8.2 Engram Architecture: Retrieval and Fusion"
layout: chapter
---

The Engram module processes each position in two functional phases: retrieval and fusion.

## Phase 1: Sparse Retrieval via Hashed N-grams

### Tokenizer Compression

Standard subword tokenizers assign disjoint IDs to semantically equivalent terms (e.g., "Apple" vs. "␣apple"). Engram implements a vocabulary projection function P: V → V' that collapses raw token IDs into canonical identifiers based on NFKC normalization, lowercasing, and other equivalence rules. This achieves a **23% reduction** in effective vocabulary size for a 128K tokenizer.

### Multi-Head Hashing

Directly parameterizing the combinatorial space of all possible N-grams is intractable. Engram uses K distinct hash heads per N-gram order n, each mapping compressed context to an index within an embedding table E_{n,k} of prime size M_{n,k} via a deterministic function φ_{n,k}:

```
z_{t,n,k} = φ_{n,k}(g_{t,n})
e_{t,n,k} = E_{n,k}[z_{t,n,k}]
```

Where φ_{n,k} is a lightweight multiplicative-XOR hash. All retrieved embeddings are concatenated to form the final memory vector:

```
e_t = ‖_{n=2}^{N} ‖_{k=1}^{K} e_{t,n,k}
```

The embedding tables are massive (up to billions of parameters) but — critically — only a constant number of slots are retrieved per token, so **scaling table size does not increase per-token FLOPs**.

## Phase 2: Context-Aware Gating

The retrieved embeddings e_t are static priors that lack contextual adaptability. Engram uses the current hidden state h_t (which has aggregated global context via preceding attention layers) as a dynamic query:

```
k_t = W_K · e_t      (Key projection)
v_t = W_V · e_t      (Value projection)
α_t = σ(RMSNorm(h_t)⊤ · RMSNorm(k_t) / √d)
```

The gated output: ṽ_t = α_t · v_t

If the retrieved memory contradicts the current context, the gate α_t tends toward zero, **effectively suppressing noise** from hash collisions or polysemous entries.

### Depthwise Causal Convolution

To expand the receptive field and enhance non-linearity, a short depthwise causal convolution (kernel size 4, dilation set to max N-gram order, SiLU activation) processes the sequence of gated values:

```
Y = SiLU(Conv1D(RMSNorm(Ṽ))) + Ṽ
```

The output is integrated via residual connection: H^(ℓ) ← H^(ℓ) + Y, followed by standard Attention and MoE layers.
