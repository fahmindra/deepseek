---
slug: ch03-the-mathematical-magic-how-the-latent-matrix-helps
title: "3.3 The mathematical magic: How the latent matrix helps"
layout: chapter
---

We have seen the new architecture of MLA, which introduces a latent $cKV$ matrix. At first glance, this seems to add complexity. How does introducing an extra matrix and two extra projection steps ($Wdkv$ and $Wuk/Wuv$) actually lead to a more efficient system?

The answer lies in a property of matrix multiplication that the DeepSeek team exploited, a technique we'll call the "absorption trick." By cleverly rearranging the attention formula, we can show that the latent matrix is the only piece of historical information we need to cache, allowing us to have both a small cache and fully unique, expressive heads. To understand this, we need to write out the full sequence of calculations mathematically.

### 3.3.1 A Step-by-Step Derivation of Q, K, and V in MLA

Let's formalize the steps we saw in the architectural diagram. We will use $X$ to represent our input embedding matrix.

##### Figure 3.3 A step-by-step mathematical derivation of the Query, Key, and Value matrices in the MLA architecture.
![Figure 3.3 A step-by-step mathematical derivation of the Query, Key, and Value matrices in the MLA architecture.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/03__image003.png)

Following the steps in figure 3.3:

1.  **The Query Matrix** ($Q$): The query calculation remains standard. It is a direct projection of the input embeddings.
    ![Equation: Q = X * Wq](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/03__image004.png)

2.  **The Latent KV Matrix** ($cKV$): This is the first new step. The input embeddings are projected down into the compressed latent space. This $cKV$ matrix is what we will eventually cache.
    ![Equation: cKV = X * Wdkv](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/03__image005.png)

3.  **The Key Matrix** ($K$): The final Key matrix is no longer a direct projection of $X$. Instead, it's an up-projection of the latent matrix $cKV$.
    ![Equation: K = cKV * Wuk](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/03__image006.png)
    *   If we substitute the definition of $cKV$ from step 2, we can see the full transformation from the original input:
    ![Equation: K = (X * Wdkv) * Wuk](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/03__image007.png)

4.  **The Value Matrix** ($V$): Similarly, the final Value matrix is also an up-projection of the same latent matrix $cKV$.
    ![Equation: V = cKV * Wuv](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/03__image008.png)
    *   And again, substituting the definition of $cKV$:
    ![Equation: V = (X * Wdkv) * Wuv](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/03__image009.png)

So far, it still looks like we've just added extra steps. The key to understanding the efficiency gain comes when we plug these new definitions for $K$ and $V$ into the main attention score formula. This is where the magic begins.

### 3.3.2 The absorption trick: How attention scores are calculated

Now that we have the new mathematical definitions for our Query, Key, and Value matrices, let's see what happens when we plug them into the core formula for attention scores:

![Equation: Scores = Q * K^T](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/03__image010.png)

We start by substituting our new definitions for $Q$ and $K$:

![Equation: Scores = (X * Wq) * ((X * Wdkv) * Wuk)^T](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/03__image011.png)

Using the rule of matrix transposition $(AB)^T = B^TA^T$, we can expand the transposed term:

![Equation: Scores = (X * Wq) * (Wuk^T * (X * Wdkv)^T)](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/03__image012.png)

And applying the rule again:

![Equation](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/03__image013.png)

This equation looks complicated, but it reveals something incredible. Notice that two of our learned weight matrices ($Wq$, $Wuk^T$) are all grouped together in the middle. Because matrix multiplication is associative, we can re-group the terms.

##### Figure 3.4 The mathematical rearrangement of the attention score calculation in MLA.
![Figure 3.4 The mathematical rearrangement of the attention score calculation in MLA.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/03__image014.png)

As shown in figure 3.4, we can rearrange the equation as follows:

![Equation: Scores = X * (Wq * Wuk^T) * (X * Wdkv)^T](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/03__image015.png)

Let's analyze the two main components of this rearranged formula:

1.  This is the product of two of our learned weight matrices. Since $Wq$ and $Wuk$ are **fixed during inference** (they are learned during pre-training), this entire product is also a fixed matrix. We can pre-compute it once and reuse it. It doesn't need to be calculated at every step.
2.  This term should look familiar. $X * Wdkv$ is exactly the definition of our **latent KV matrix,** **cKV**.

This is the "absorption trick." The up-projection matrix for the Keys, $Wuk$, has been mathematically rearranged to be next to the query projection matrix, $Wq$. Because both $Wq$ and $Wuk$ are fixed, learned matrices from pre-training, their product $(Wq * Wuk^T)$ is also just a single, fixed matrix that we don't need to re-calculate during inference.

This simplifies the process. To get our attention scores, the original formula $Q * K^T$ now effectively becomes:

![Equation: Absorbed_Query * cKV_Cache^T](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/03__image016.png)

This is a profound change. We started with a formula that required a full Query matrix and a full Key matrix. After this mathematical rearrangement, we see that the calculation only requires two things:

1.  The original input $X$.
2.  The latent matrix $cKV = X * Wdkv$.

Crucially, as we generate new tokens and our input sequence grows, the **matrix $X$ representing this growing history also expands.** Consequently, the latent matrix $cKV$ that summarizes this history must also expand by appending the new **token's latent vector.** This proves that the $cKV$ matrix is the only piece of historical information that needs to be updated and stored to compute the attention scores. We have successfully reformulated the attention score calculation to depend only on this single, compact, cached matrix, eliminating the need to cache the full $K$ and $V$ matrices entirely.

### 3.3.3 The final step: Calculating the context vector

We have successfully reformulated the attention score calculation to rely on a single, cached latent matrix, $cKV$. Now, let's see if this efficiency holds for the final step: creating the `Context Vector Matrix.`

The standard formula is:

![Equation: Context = Attention_Weights * V](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/03__image017.png)

We know that `Attention Weights` are derived from the attention scores ($Q * K^T$), which we just simplified. And we have our new definition for the `Value` matrix:

![Equation: V = (X * Wdkv) * Wuv](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/03__image009.png)

Let's plug everything together.

##### Figure 3.5 The mathematical rearrangement for calculating the final output in MLA.
![Figure 3.5 The mathematical rearrangement for calculating the final output in MLA.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/03__image018.png)

As figure 3.5 illustrates, the full calculation for the context vector, before it's passed to the final output projection layer ($Wo$), looks like this:

![Equation: Output = (Weights * (cKV * Wuv)) * Wo](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/03__image019.png)

Again, we can recognize the term $(X * Wdkv)$. This is our **cached latent matrix.**

And just like before, we can use the associative property of matrix multiplication. The $Wuv$ matrix can be "absorbed" by the final output projection layer, $Wo$. The product $(Wuv * Wo)$ is just another single, fixed matrix that is learned during pre-training.

This confirms our central thesis: **we only need to cache the latent matrix.**

Let's summarize the beauty of this workflow:

1.  When a new token arrives, we compute its contribution to the $cKV$ cache.
2.  We use the full, updated $cKV$ cache to help compute the attention scores.
3.  We also use that same, updated $cKV$ cache to help compute the final context vector.

By introducing this intermediate latent space, the DeepSeek team designed a system where a single, compact cached matrix can be efficiently used for both the key and value sides of the attention calculation. This is how they achieved the best of both worlds: a dramatically smaller memory footprint for the cache, without sacrificing the performance benefit of having unique, expressive projections for every attention head.