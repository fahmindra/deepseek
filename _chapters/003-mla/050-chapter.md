---
slug: ch05-quantifying-the-gains
title: "3.5 Quantifying the gains"
layout: chapter
---

The multi-step process of MLA—compressing, caching, and decompressing—is a brilliant piece of engineering. But what does it achieve in practice? Let's quantify the two main benefits: the reduction in cache size and the preservation of performance.

### 3.5.1 The new KV cache formula: A 64x reduction

In chapter 2, we established the formula for the size of a standard MHA KV cache:

![Equation: Size = 2 * (n * h) * layers * batch * length * 2](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/03__image025.png)

The critical term was $2 * (n * h)$, representing the two separate caches (Keys and Values), each with a dimension equal to the number of heads ($n$) times the head dimension ($h$).

With MLA, this term is completely replaced. We no longer store two caches, and the dimension we store is not $n * h$, but the much smaller latent dimension, $d\_latent$.

The new formula becomes:

![Equation: Size = d_latent * layers * batch * length * 2](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/03__image026.png)

Let's plug in the numbers for a model at the scale of DeepSeek-V3:

*   $n$ (heads) = 128
*   $h$ (head dimension) = 128
*   $n * h$ (total dimension) = 16,384
*   $d\_latent$ (latent dimension, as used in DeepSeek's paper) ≈ 512

Let's compare the core factors:

*   **MHA Cache Dimension:** $2 * (n * h) = 2 * 16,384 = 32,768$
*   **MLA Cache Dimension:** $d\_latent = 512$

The reduction in the size of the cached data is: $32,768 / 512 = 64$ times.

This is an incredible achievement. By caching a single, compressed latent matrix, MLA reduces the memory footprint of the KV cache by a factor of 64. The theoretical 400 GB cache size we calculated for a DeepSeek-scale model collapses to a much more manageable **~7 GB**.

### 3.5.2 Preserving performance: Why head diversity is maintained

More importantly, MLA achieves this massive memory reduction without the performance degradation seen in MQA and GQA. Why?

The answer lies in the up-projection matrices, $Wuk$ and $Wuv$. In the DeepSeek architecture, these up-projection matrices are themselves multi-headed. This means that there is a unique, learned $Wuk$ and $Wuv$ for every single attention head.

When we decompress the shared latent cache $cKV$, each head uses its own special up-projector. This means that:

*   Head 1 reconstructs $K1$ and $V1$.
*   Head 2 reconstructs a different $K2$ and $V2$.
*   ...and so on for all 128 heads.

At the moment of the attention calculation, every head is working with its own unique Key and Value matrices, just like in standard MHA. We have not shared any content across the heads. We have preserved the full diversity and expressive power of the multi-head design. This is how MLA achieves the best of both worlds.