---
slug: ch15-quantifying-the-gains-the-final-cache-memory-comparison
title: "3.15 Quantifying the gains: The final cache memory comparison"
layout: chapter
---

The multi-step, decoupled process of the fused MLA-RoPE architecture is a masterpiece of engineering. But what does it achieve in practice? Let's quantify the two main benefits we set out to achieve: a dramatic reduction in cache size and the preservation of the model's expressive performance.

In the previous chapter, we established the formulas for the memory cost of MHA, MQA, and GQA. Now, we can define the final formula for our advanced MLA with Decoupled RoPE.

The crucial insight is that at each step, we only need to store two things in memory:

*   **The Latent KV Cache**: This stores the compressed content information for all past tokens. Its dimension per token is $d\_latent$.
*   **The Decoupled Key Cache**: This stores the rotated positional information for all past tokens. Its dimension per token is $d\_rope$.

Therefore, the new formula for the size of the cache is:

![Equation: Cache_Size = layers * batch * sequence_len * (d_latent + d_rope) * 2](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/03__image057.png)

Let's break this down:

*   $l$ (layers), $b$ (batch size), $s$ (sequence length), and 2 (bytes per parameter) are the same as before.
*   The $(d\_latent + d\_rope)$ term replaces the $g * h$ from GQA or the $n * h$ from MHA.
*   The `* 2` factor that previously accounted for storing two separate matrices (one for Keys and one for Values) is gone. In this new architecture, we are caching specialized, combined representations ($d\_latent$ for content and $d\_rope$ for the positional key), not separate $K$ and $V$ matrices.

Let's plug in the numbers for a model at the scale of DeepSeek-V2, where $n=128$ and $h=128$:

*   **MHA Cache Dimension:** $2 * 128 * 128 = 32,768$
*   **MLA+RoPE Cache Dimension:** In the DeepSeek paper, $d\_latent$ is $4 * h = 512$, and $d\_rope$ is $h / 2 = 64$. The total is $512 + 64 = 576$.

The final reduction in the size of the cached data is: $32,768 / 576 \approx 57$ times.

This is an incredible result. By decoupling content and position and using a compressed latent cache, the MLA+RoPE architecture reduces the memory footprint of the KV cache by nearly two orders of magnitude compared to standard MHA. The theoretical 400 GB cache we calculated collapses to a much more manageable **~7 GB**.