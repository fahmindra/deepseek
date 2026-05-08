---
slug: ch13-the-incompatibility-problem-why-standard-mla-and-rope-dont
title: "3.13 The incompatibility problem: Why standard MLA and RoPE don’t work together"
layout: chapter
---

To see exactly why standard MLA and RoPE don't work together, we must look at the mathematics of the "absorption trick" that makes MLA so efficient.

As we established, the efficiency of MLA comes from a clever mathematical rearrangement of the attention score formula. The original formula $Q * K^T$ was transformed into:

![Equation: Attention Scores = X * (Wq * Wuk^T) * (X * Wdkv)^T](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/03__image047.png)

This trick works because the two weight matrices, $Wq$ and $Wuk$, are adjacent to each other. Their product, $(Wq * Wuk^T)$, can be pre-computed as a single, fixed matrix. This allows the model to avoid caching the full Key matrix and instead only cache the much smaller latent matrix, $cKV = X * Wdkv$. This absorption is only possible because the weight matrices are **position-agnostic**; their values do not change based on a token's position.

Now, let's recall how RoPE works. It injects positional information by applying a rotation function, `RoPE()`, directly to the Query and Key vectors after they have been projected.

```python
Q_rotated = RoPE(Q, position)
K_rotated = RoPE(K, position)
```

Crucially, the `RoPE()` function is **position-dependent.** The rotation applied to the vector for a token at position 5 is different from the rotation applied to a token at position 10.

Herein lies the problem. If we try to combine these two techniques naively, the `RoPE()` function ends up sitting directly between the two matrices we need to absorb. The attention score calculation would look something like this:

![Equation: Scores = RoPE(X * Wq, pos) * RoPE((X * Wdkv * Wuk)^T, pos)](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/03__image047.png)

The position-dependent `RoPE()` function acts as a barrier, preventing $Wq$ and $Wuk$ from being multiplied together. The absorption trick is broken.

This has bad consequences for inference efficiency. Without the absorption trick, we would be forced to re-compute the full, rotated Key matrix for every token in the entire history at every single generation step. This would completely negate the speed benefits of caching and defeat the entire purpose of MLA.

This incompatibility between the position-dependent nature of RoPE and the position-agnostic requirement of the MLA cache is the final puzzle we must solve. The DeepSeek team's solution to this problem is a clever and elegant modification to the architecture itself, which we will now build from scratch.