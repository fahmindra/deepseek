---
slug: ch04-the-new-inference-loop-with-mla
title: "3.4 The new inference loop with MLA"
layout: chapter
---

Now that we understand the mathematical magic behind MLA, let's put it all together and define the new, highly efficient workflow for generating a single new token. This process brilliantly leverages the cached latent matrix to avoid re-calculating the entire history.

We will trace the journey of a new token, "`bright,`" as it enters the system, assuming the model has already processed "`The next day is`" and has their latent representations stored in the $cKV$ cache.

### 3.4.1 What happens when a new token arrives?

The entire process is designed to be incredibly efficient, performing the absolute minimum number of computations necessary. When a new token like "`bright`" arrives, the model follows these steps:

**Step 1: Compute the new projections**

The very first step is to compute the three essential vectors for our new token. We take the input embedding for "`bright,`" which we'll call $X\_bright$, and perform three parallel multiplications.

##### Figure 3.6 Computing the Query vector for the new token.
![Figure 3.6 Computing the Query vector for the new token.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/03__image020.png)

First, we compute the **Query Vector** for "`bright`". This is a standard projection and is always calculated fresh for the current token.

```python
Query_bright = X_bright * Wq
```

Next, and most importantly, we compute the token's contribution to our new, compressed cache.

##### Figure 3.7 Computing the new Latent Vector cKV for "bright."
![Figure 3.7 Computing the new Latent Vector cKV for "bright."](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/03__image021.png)

We multiply the input embedding $X\_bright$ by the **KV Down-Projector** ($Wdkv$). This produces a single, small, compressed **Latent Vector for Token Bright.** This unit vector contains all the necessary Key and Value information for this token in a compressed format.

These are the only new projections we need to perform. We do *not* compute the full Key and Value vectors directly from the input. Instead, we will use our cache.

### 3.4.2 Caching the latent vector: The only thing we store

Now that we have the new `Latent Vector for Token Bright`, the next step is to update our cache. In the previous inference step (for the input "The next day is"), we would have already computed and stored the latent vectors for those tokens in our $cKV$ cache.

Our task now is to simply append our new latent vector to this existing cache.

##### Figure 3.8 Updating the Latent KV Cache. The newly computed latent vector for "bright" is appended to the existing cache for the previous tokens.
![Figure 3.8 Updating the Latent KV Cache. The newly computed latent vector for "bright" is appended to the existing cache for the previous tokens.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/03__image022.png)

As shown in figure 3.8, our original $cKV$ cache had a shape of (4, 4), containing the compressed history. By appending the new (1, 4) latent vector, we now have an **Updated latent matrix** with a shape of (5, 4).

This single, compact matrix is the **only piece of historical information** we store in memory between steps. We do not need a separate cache for Keys and a separate cache for Values. This is the source of MLA's memory efficiency.

### 3.4.3 Decompressing the cache and calculating attention

With our updated cache and the fresh Query vector for "`bright`" we now have everything we need to perform the attention calculation.

First, we must "decompress" our latent cache back into the full-dimensional Key and Value matrices. This is done on the fly, just for this single calculation.

##### Figure 3.9 Decompressing the updated latent cache into the full Key and Value matrices using the up-projection matrices.
![Figure 3.9 Decompressing the updated latent cache into the full Key and Value matrices using the up-projection matrices.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/03__image023.png)

As shown in figure 3.9, we perform two parallel multiplications:

1.  The `Updated cKV Cache` (shape 5, 4) is multiplied by the `Key Up-Projector` ($Wuk$) to produce the full Key Matrix $K$ (shape 5, 4).
2.  The same `Updated cKV Cache` (shape 5, 4) is multiplied by the `Value Up-Projector` ($Wuv$) to produce the full Value Matrix $V$ (shape 5, 4).

Now, we have everything a standard attention mechanism needs: a fresh Query vector for our current token, and the full Key and Value matrices for the entire history, faithfully reconstructed from our compact cache.

The final steps proceed just as we've seen before.

##### Figure 3.10 The final attention calculation using the fresh Query and the decompressed Key and Value matrices.
![Figure 3.10 The final attention calculation using the fresh Query and the decompressed Key and Value matrices.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/03__image024.png)

1.  **Calculate Attention Weights**: We multiply the `Query Vector for “bright”` by the transpose of the full, updated `Key Matrix K`. The result, after scaling and softmax, is the `Attention Weights for token “bright”`. This is the only row we need because, as we established in chapter 2, the prediction for the next token depends only on the context vector of the last token in the sequence.
2.  **Compute Context Vector**: We multiply these `Attention Weights` by the full, updated `Value Matrix V` to produce the single, `final Context Vector for token "bright"`.

This single context vector is then passed through the rest of the Transformer's layers to predict the next token. We have successfully generated our output while only ever storing the small, latent $cKV$ matrix in our cache.