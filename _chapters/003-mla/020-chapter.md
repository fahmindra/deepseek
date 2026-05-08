---
slug: ch02-the-mla-architecture-a-visual-walkthrough
title: "3.2 The MLA architecture: A visual walkthrough"
layout: chapter
---

To achieve this "compress for storage, decompress for use" strategy, Multi-Head Latent Attention introduces a new set of projection layers and modifies the data flow within the attention block. Instead of projecting the input directly into Keys and Values, it first projects the input into a compressed latent space.

Let's examine the complete workflow, as illustrated in figure 3.2.

##### Figure 3.2 The full architectural data flow of Multi-Head Latent Attention (MLA).
![Figure 3.2 The full architectural data flow of Multi-Head Latent Attention (MLA).](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/03__image002.png)

This diagram looks complex at first, but it's a logical extension of what we already know. Let's break it down into its two main paths: the query path and the key/value path.

### 3.2.1 The query path (unchanged)

The query path in this simplified version of MLA remains the same as in standard Multi-Head Attention. The goal is to create a full-sized query matrix that can ask "questions" of the entire context.

*   The input embedding matrix $X$ (shape 4, 8) is multiplied by the $Wq$ weight matrix (shape 8, 4).
*   This produces the final `Query` matrix $Q$ (shape 4, 4), ready for the attention calculation. In this example, the head dimension (`d_head`) is 4.

*(Note: More advanced versions of MLA also compress the query, but for understanding the core KV cache innovation, we can consider the query path to be standard).*

### 3.2.2 The key/value path (the innovation)

This is where the magic happens. Instead of two separate projections for keys and values, there is now a two-step process involving a new, intermediate latent matrix.

**Step 1: Down-projection to the latent space**

The input embedding matrix $X$ is first multiplied by a new, learnable weight matrix called the **KV Down-Projector** ($Wdkv$).

*   $X$ (shape 4, 8) is multiplied by $Wdkv$ (shape 8, 4).
*   This produces a single, smaller, compressed matrix called the **Latent KV Matrix** ($cKV$), with a shape of (4, 4).

> **Note**
>
> In our example, the **latent dimension** (4) happens to be the same as the number of tokens. This is purely a coincidence for the sake of simple diagrams. In practice, these dimensions are completely independent and much different. For a model like **DeepSeek-V3**, the **model dimension** is **7168** and the **latent dimension** is **512**.

So, the KV Down-Projector ($Wdkv$) would have a shape of (7168, 512), compressing a large 7168-dimensional vector into a much smaller 512-dimensional one. The numbers in our example are kept small only to build intuition.

This $cKV$ matrix is the **only thing getting cached** during inference. Notice two things immediately:

*   We are only caching **one matrix**, not two.
*   The dimension of this matrix (4 in this example, but 576 in the real DeepSeek model) is **much smaller than the full dimension of the Keys and Values combined.**

**Step 2: Up-projection from the latent space**

Now that we have our compressed $cKV$ matrix, we can reconstruct the full Key and Value matrices from it whenever we need them for the attention calculation. This is done with two new **Up-Projection** matrices:

*   The $cKV$ matrix (shape 4, 4) is multiplied by the **Key Up-Projector** ($Wuk$ (shape 4, 4)) to produce the final Key matrix $K$ (shape 4, 4).
*   The $cKV$ matrix (shape 4, 4) is also multiplied by the **Value Up-Projector** ($Wuv$ (shape 4, 4)) to produce the final `Value` matrix $V$ (shape 4, 4).

After this two-step process, we have our three required matrices: $Q$ (from the standard path), and $K$ and $V$ (reconstructed from the latent matrix). From this point forward, the rest of the attention calculation—computing scores, applying softmax, and calculating the context matrix—proceeds exactly as it does in standard Multi-Head Attention.

The efficiency of this method comes from a two-step process. First, we store the historical Key and Value information in a single, compressed matrix to save memory. Second, we decompress this matrix on the fly to reconstruct the full, unique Key and Value matrices for each head right when they are needed for calculation. This raises a key question: how can adding projection and reconstruction steps actually make the process more efficient? The solution is found in a specific application of matrix multiplication that we'll refer to as the "absorption trick."