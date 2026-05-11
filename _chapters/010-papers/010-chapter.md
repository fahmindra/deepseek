---
slug: ch-10-1
title: "DeepSeek-v4 beyond basics"
layout: chapter
---

![Preview image](https://miro.medium.com/v2/resize:fit:700/1*5FVlk2ExIIM9lbe86GhNsg.png)

# DeepSeek-v4 beyond basics: A Practical Guide to mHC, CSA, HCA, and Muon
## Step-by-step walkthrough [Intuition & Math], explaining why & how

**Oleh:** [James Koh, PhD](https://medium.com/@byjameskoh) | **Publikasi:** [MITB For All](https://medium.com/mitb-for-all)
**Tanggal:** May 9, 2026 (Updated: May 9, 2026) | **Waktu baca:** ~15 min read

---

### Contents & Background

In this article, we will cover the following:

*   3.1 Manifold-Constrained Hyper-Connections (mHC)
*   3.1.1 Hyper-Connections (HC)
    3.1.2 Manifold-Constrained Hyper-Connections (mHC)
    3.1.3 mHC in DeepSeek-v4
*   3.2 Compressed Sparse Attention (CSA)
*   3.2.1 DeepSeek Sparse Attention (DSA)
    3.2.2 CSA in DeepSeek-v4
*   3.3 Heavily Compressed Attention (HCA)
*   3.4 Muon optimizer

This is a continuation from [Deepseek-V4 and before — getting the foundations right](https://medium.com/mitb-for-all/deepseek-v4-and-before-getting-the-foundations-right-d45ae9a07724), which covered the following:

*   Why DeepSeek?
*   References
*   1. Fundamentals you should not skip
*   1.1 Byte Pair Encoding
    1.2 Embeddings
    1.3 Considering context
    1.4 Positional Encodings
    1.5 Getting outputs
*   2. Building blocks
*   2.1 Attention
    2.2 Multi-heads Latent Attention (MLA)
    2.3 Mixture-of-Experts (MoE)
    2.4 Multi-Token Prediction (MTP)

In this article, we will focus on the new additions. I strongly recommend that if you're new to this, to first ensure that you familiarize yourself with the basics in the previous article before continuing here.

### References

By DeepSeek

*   DeepSeek-V3.2 (2025) *Pushing the Frontier of Open Large Language Models* https://arxiv.org/pdf/2512.02556
*   DeepSeek-v4 (2026) *Towards Highly Efficient Million-Token Context Intelligence* https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf
*   mHC: Manifold-Constrained Hyper-Connections by Xie et al. (2026) https://arxiv.org/pdf/2512.24880

Others

*   Hyper Connections by Zhu et al. (ByteDance) (2025) https://openreview.net/pdf?id=9FqARW7dwB

---

### 3.1 Manifold-Constrained Hyper-Connections (mHC)

#### 3.1.1 Hyper-Connections (HC)

The authors of the HC paper in 2025 stated that in the widely used residual connections (or skip connections as we commonly call it), there are "*trade-offs between gradient vanishing and representation collapse*".

Here, the authors talk about 'Pre-Norm' and 'Post-Norm'. In Pre-Norm, the input x is normalized before each residual block, ie. x + F(Norm(x)), while in Post-Norm, it is Norm(x + F(x)).

We know that residual connections were introduced in 2016 to mitigate the vanishing gradient problem. That is the Pre-Norm approach, which potentially suffers from representation collapse, where "*hidden features in deeper layers become highly similar, diminishing the contribution of additional layers*". On the other hand, the Post-Norm approach is still prone to the vanishing gradient problem.

Hence, instead of adding an identity mapping with either pre- or post-Norm, the authors of HC propose learnable depth-connections and width-connections.

![Figure 2 of Hyper Connections paper](https://miro.medium.com/v2/resize:fit:700/1*_w3cTb-4ZoIWigi-_KYGSA.png)
*Screenshot of Figure 2 of Hyper Connections paper by Zhu et al. (2025) https://arxiv.org/pdf/2104.09864. Under (b), the hyper-connections enable lateral information exchange and vertical integration of features across depths.*

Some people might feel a little uncomfortable seeing math like this.

![Equations 1 and 2](https://miro.medium.com/v2/resize:fit:700/1*WHikiFwmKNv88wpPHTyaVA.png)
*Screenshot of equations (1) and (2) of Section 2.1 of Hyper Connections paper by Zhu et al. (2025).*

However, once you expand the matrix out, using n=2 (it's a small number, and also corresponds to the Figure 2), it's not too bad. *α* and *β* are scalars, **h**₁ has shape (d, 1), as clearly stated in the first line of section 2.1 of https://arxiv.org/pdf/2104.09864.

![Equation expanded](https://miro.medium.com/v2/resize:fit:700/1*LAdHMa4ZdO_dH76wGyZUkg.png)
*Equation expanded and typed by author, based on equation (1), and the first paragraph of section 2.1 (for **H**), of Hyper Connections paper.*

Therefore, the output of HC, as given by equation (2), works out to be

![Equation 2 expanded](https://miro.medium.com/v2/resize:fit:700/1*LTDSsIvtnY8fI6l9fIFldQ.png)
*Equation expanded and typed by author, based on equation (2) of Hyper Connections paper.*

This equation is just Figure 2(b) of the paper, written in math. Basically, what we have is this: take a weighted combination of **h**₁ and **h**₂, pass it through network layer calligraphic *T*, weight that output with *β*, and then add that to some other weighted combinations of **h**₁ and **h**₂.

![Figure 2(b)](https://miro.medium.com/v2/resize:fit:700/1*c81GmbBrEOmqZQqBo5LSOQ.png)
*Screenshot of Figure 2(b) of Hyper Connections paper by Zhu et al. (2025) https://arxiv.org/pdf/2104.09864.*

The superscript to **h** is dropped, but **h**⁰ is just the initial input while **h**ᵏ is the input to the (k+1) layer, where **h**₁⁰ and **h**₂⁰ are just replicas of **h**⁰.

This 'layer' itself *could* correspond to a block that includes attention and MoE. The left of the bottom figure shows where the original skip connections exist within multiple transformer blocks, while the right of the bottom figure shows HC in its place. Here, the However, note that the blocks in DeepSeek-v4 is NOT the simply the transformer block in DeepSeek-v3 with the residual connections replaced with the above.

![Figure 8](https://miro.medium.com/v2/resize:fit:700/1*RDByA_6DGcR84FX2-x2MSA.png)
*Screenshot of Figure 8 of Hyper Connections paper https://arxiv.org/pdf/2104.09864.*

#### 3.1.2 Manifold-Constrained Hyper-Connections (mHC)

Now, we're ready to jump into mHC proper.

In DeepSeek-v4 section 2.2, the authors wrote that using HC, "*training will frequently exhibit numerical instability when stacking multiple layers*".

Let's focus on the cited paper where mHC was first introduced (Xie et al. 2026). It is also a work by the DeepSeek organization. mHC was presented by Xie et al. as a means to mitigate the "*risks of instability*" as training scale increases.

![Figure 1 of mHC paper](https://miro.medium.com/v2/resize:fit:700/1*j4pSkX5KkRTu3FWqYFnZuQ.png)
*Screenshot of Figure 1 of mHC paper by Xie et al. (2026). Unlike the unconstrained HC (b), mHC (c) focuses on optimizing the residual connection space by projecting the matrices onto a constrained manifold.*

mHC expands the residual stream into multiple residual channels. Before each Transformer block, the channels are mixed into a d-dimensional input, and after, the output is written back into the expanded residual state. Hence, flexibility is increased, without increasing the hidden dimension.

What's with the "Manifold-Constrained"? Well, it constrains the residual mixing matrix onto a specific manifold, as described by Equation (6) below, which almost looks mythical.

![Equation 6 of mHC paper](https://miro.medium.com/v2/resize:fit:700/1*oXRNUapWH-x8aroSZwgJcA.png)
*Screenshot of equation (6) of mHC paper by Xie et al. (2026). https://arxiv.org/pdf/2512.24880*

Let's break it down. The calligraphic *H*ₗʳᵉˢ with superscript ʳᵉˢ and subscript ₗ is the residual hyper-connection matrix at layer l, which the authors stated "*represents a learnable mapping that mixes features within the residual stream*" (section 1). *H*ₗʳᵉˢ is itself mathematically obtained from equation (5) of that paper.

We start with the simplest part, just the matrix alone. The component below means that the matrix of residuals has shape (n, n), where all values are real and non-negative.

![Equation component](https://miro.medium.com/v2/resize:fit:700/1*2MOf12lPoNlIZtHtsI9v0Q.png)
*Equation component typed by author*

Since Xie et al. (2026) stated that they implemented mHC with n=4 (section 4.3 of the paper), I will plug in this value directly. The first constraint of the mHC equation (*H*ₗʳᵉˢ**1**ₙ = **1**ₙ) dictates that each row of *H*ₗʳᵉˢ sums to 1, while the second constraint (**1**ₙᵀ*H*ₗʳᵉˢ = **1**ₙᵀ) dictates that each column sums to 1.

![Equation constraints](https://miro.medium.com/v2/resize:fit:700/1*0Wb4V6F8wIvMj7Eo8EXevw.png)
*Equation components typed by author, for 2nd and 3rd constraints of equation (6)*

Collectively, all three constraints would mean that every stream distributes its weight fully, and every stream receives a total weight of 1. Values cannot be arbitrarily large, and every component *H*ₗʳᵉˢ would inevitably be bounded between 0 and 1. And that is it. You now understand what it means by restricting *H*ₗʳᵉˢ to be a "*doubly stochastic matrix*".

#### 3.1.3 mHC in DeepSeek-v4

Now, linking the above 3.3.1 (explaining HC) and 3.3.2 (explaining mHC) back to the DeepSeek-v4 paper.

Recall that in DeepSeek-v3, we have ordinary residual connections in each transformer block. (Figure 2 of https://arxiv.org/abs/2412.19437). In DeepSeek-v4, the authors recap (in section 2.2) that HC expands the residual stream from *d* to *n*ₕ꜀×*d*, and that they used mHC to constrain the residual mapping matrix *B*ₗ to "*the manifold of doubly stochastic matrices*" *M*.

![Equation 2 DeepSeek-v4](https://miro.medium.com/v2/resize:fit:700/1*q83WXjEebIkcRMZXLO2dwA.png)
*Screenshot of equation (2) of DeepSeek-v4 paper. https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf*

There's differences in the nomenclatures between those used by Zhu et al. (2025) and the DeepSeek-v4 authors, summarized in the table below.

![Nomenclatures table](https://miro.medium.com/v2/resize:fit:700/1*LLr-nX0zrblqOeVs4E8LNQ.png)
*Equations matched by author, comparing difference in nomenclatures*

Meanwhile, *H*ₗʳᵉˢ in mHC of Xie et al. (2026) is represented as *M* in DeepSeek-v4.

Note that in DeepSeek-v4, *Aₗ*, *Bₗ* and *Cₗ* are actually not in boldface, but I find it a good habit to show matrices in bold. Also, *Aₗ*, *Bₗ* and *Cₗ* are dynamically generated, ie. they dependent on the input as well as other learnable parameters as given in equations (3) to (7) of the DeepSeek-v4 paper.

The model set-up details are specified in 4.2.1 of the DeepSeek-v4 paper. For V4-Pro, there are 61 Transformer layers, with hidden dimension d = 7168 and nₕ꜀ = 4.

---

### 3.2 Compressed Sparse Attention (CSA)

In the above section 3.1 of this article, we looked at the mHC, which is mainly about improving information flow and training stability for large context. In this section and the next, we focus on the aspects to "*substantially reduce the computational cost of attention in long-text scenarios*."

#### 3.2.1 DeepSeek Sparse Attention (DSA)

The predecessor to CSA is DSA, introduced in DeepSeek-v3.2 (2025). DSA contains a "Lightning Indexer" which scores how relevant prior tokens are to the current query (see equation 1 of [section 2.1](https://arxiv.org/pdf/2512.02556)). Along with the 'Top-k Selector' (which the authors termed a "fine-grained token selection mechanism"), it selects the top-k key-value entries before applying attention.

![Figure 2 DeepSeek-v3.2](https://miro.medium.com/v2/resize:fit:700/1*4P3Atgw8GYwUwN5IrLjf_Q.png)
*Screenshot of Figure 2 of DeepSeek-v3.2 paper https://arxiv.org/pdf/2512.02556. Attention architecture where DSA is instantiated under MLA.*

Based on my understanding (from section 2.1 of DeepSeek-v3.2), this "token selection mechanism" is really just a fanciful name for simply retrieving entries corresponding to top-k scores, which is straightforward conceptually as well as in terms of implementation.

For a quick comparison of this DSA differ from ordinary multi-head latent attention, I've include the screenshot of the MLA used in DeepSeek-v3. For more foundational knowledge of how DeepSeek-v3 works, refer to this [guide](https://medium.com/mitb-for-all/deepseek-v4-and-before-getting-the-foundations-right-d45ae9a07724) by me.

![Figure 2 DeepSeek-v3](https://miro.medium.com/v2/resize:fit:700/1*Du8IIGUgZmbkAMsRSSK20A.png)
*Screenshot from the bottom-right portion of Figure 2 of Deepseek-v3 (2024) https://arxiv.org/pdf/2501.12948.*

Notice that moving up from the input hidden state **h**ₜ, the left side behaves in a similar manner where we have the latent cₜ𐞥 (the superscript should be Q, but unable to represent in-text), eventually leading to [qₜˏᵢꟲ; qₜˏᵢᴿ]. cₜ𐞥 is the shared compressed latent query representation for token t, while qₜˏᵢꟲ is the expanded content-query vector used by attention head i, while qₜˏᵢᴿ is its RoPE counterpart.

The right side is what differs. Both MLA and DSA still have the latent cₜᴷⱽ which is a compact cache that will allow subsequent reconstruction into kₜᶜ and vₜᶜ. But, in MLA of DeepSeek-v3, we have [qₜˏᵢꟲ; qₜᴿ] with [kₜˏᵢꟲ; kₜᴿ] and **v**ₜˏᵢꟲ (across the multiple heads i) entering a multi-head attention for all past tokens. On the other hand, in DSA of DeepSeek-v3.2, the queries [qₜˏᵢꟲ; qₜᴿ] enter a multi-query attention just a subset of [**c**ₜᴷⱽ; kₜᴿ] (corresponding to only the tokens selected by the 'Top-k Selector') and not all tokens. From **c**ₜᴷⱽ, the **k** and **v** can be reconstructed, and in multi-query attention, the latent KV is shared across multiple queries for computational efficiency.

#### 3.2.2 CSA in DeepSeek-v4

We can think of CSA as DSA applied to compressed KV blocks (rather than individual KV entries), with a local sliding-window branch added back for fine details.

![Figure 3 DeepSeek-v4](https://miro.medium.com/v2/resize:fit:700/1*2cMZjebqt_WCZXuhS5Qp_w.png)
*Screenshot of Figure 3 of DeepSeek-v4 paper, showing the core architecture of CSA.*

To start off, consider H (a sequence of input hidden states, with sequence length n and hidden size d), represented by the bottom-left of the above figure. Within the Token-Level compressor, we have the following.

Cᵃ and Cᵇ are two token-level latent KV-entry streams produced by projecting the hidden sequence H through two different learned matrices Wᵃᴷⱽ and Wᵇᴷⱽ (equation 9), respectively, while Zᵃ and Zᵇ provide the logits (equation 11). Each *m* set of these gets compressed into one KV entry (equation 12). Recall that we had **c**ₜᴷⱽ in MLA (DeepSeek-v3) and DSA (DeepSeek-v3.2). Here, we have two streams to allow information from overlapping windows to be combined into Cᶜᵒᵐᵖ, to reduce boundary artifacts and mitigate problems of possible 'bad splits'.

![Equations 9 to 12](https://miro.medium.com/v2/resize:fit:700/1*nEM5gq95HczIfI33zCpYSg.png)
*Screenshot of equations (9) to (12) of [DeepSeek-v4 paper](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf).*

⊙ in equation (12) denotes the Hadamard product (element-wise product of each matrix entry), while the softmax operation is performed along the row dimension comprising elements from both Zᵃ and Zᵇ. The added components Bᵃ and Bᵇ are learnable position biases.

CSA includes a component called Lightning Indexer (dotted box in Figure 3) for sparse selection (see equations 13 to 17 of section 2.3.1 of DeepSeek-v4).

As stated in 4.2.1 of the DeepSeek-v4 paper, for V4-Pro, the compression rate 𝑚 = 4, indexer query heads 𝑛ᴵₕ = 64, the indexer head dimension 𝑐ᴵ = 128, and the attention top-k is set to 1024. This means that for 1 million tokens, there would still be 250,000 Cᶜᵒᵐᵖ, and with sparse selection we keep to top 1024.

![Equations 13 to 17](https://miro.medium.com/v2/resize:fit:700/1*Hievxq2LIhT1T1kmEeTQEw.png)
*Screenshot of equations (13) to (17) of [DeepSeek-v4 paper](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf).*

Wᴰ𐞥 and Wᴵᵁ𐞥 (again, 𐞥 should be superscript Q which I am unable to present in-text) are learnable matrices, of shapes (d, d꜀) and (d꜀, cᴵnᴵₕ) respectively, which projects the input hidden state **h**ₜ into **q**ₜᴵ. (see equations 13 and 14). The use of Wᴰ𐞥 with Wᴵᵁ𐞥 is analogous to low-rank factorization, making the projection operation computationally cheaper. **q**ₜᴵ is the indexer query used for sparse selection, containing concatenated information from across indexer heads. (An indexer head is different from an attention head, but the idea remains to have multiple 'viewpoints', in this case for scoring to subsequently determine selection).

Here comes the tricky part that is not explicitly written out mathematically. How is Kₛᴵᶜᵒᵐᵖ (on the RHS of equation 16) even obtained in the first place?

It is written that "*CSA performs the same compression operation used for Cᶜᵒᵐᵖ to get compressed indexer keys Kᴵᶜᵒᵐᵖ*" (equations 9 to 12, likely on different learned projection matrices).

Meanwhile, **w**ₜᴵ corresponds to the weights for the indexer heads (equation 15), and the index score Iₜₛ is obtained (equation 16) from query token t and a preceding compressed block s. In equation 17, we get Cₜˢᵖʳˢᶜᵒᵐᵖ which is a subset of compressed KV entries (refer back to Figure 3 of DeepSeek-v4, on the arrows pointing into, and out of, the Top-k Selector) for those whose score Iₜₛ is within the top-k.

Finally, Cₜˢᵖʳˢᶜᵒᵐᵖ "*serves as both attention key and value*", and enters Multi-Query Attention (MQA), with **q**ₜ as the query (equation 19 of DeepSeek-v4 paper).

![Comparison MLA vs CSA](https://miro.medium.com/v2/resize:fit:700/1*5FVlk2ExIIM9lbe86GhNsg.png)
*Figure drawn by author, on the final inputs entering the multi-head/multi-query attention, taking reference to Figure 2 of Deepseek-v3, for (a), Figure 3 of DeepSeek-v4 paper, for (b).*

Having a side-by-side comparison between MLA (of v3) and CSA (of v4), we see that the concatenation of {sliding window KV entries} and {selected compressed KV entries from top-k} in CSA serves the role of the attention memory that, in MLA, is represented by {[**k**ₜˏᵢᶜ ; **k**ₜˏᵢᴿ]} and {**v**ₜˏᵢᶜ}. The sliding-window in DeepSeek-v4 is the most-recent 128 original uncompressed tokens.

---

### 3.3 Heavily Compressed Attention (HCA)

The good (?) news is that we can jump straight-in without referring to another paper. This is unlike mHC, where we had to start with HC (3.1.1), nor like CSA, where we had to start with DSA (3.2.1).

The idea of HCA is to give the model a global view of the long context without too much computational costs, by performing attention over all the heavily compressed sequences. It "*compresses the KV cache in a heavier manner, but does not employ sparse attention*". In a way, HCA is like a more extreme counterpart of CSA, where it selects only one of the two cost-saving aspects, and take it to the extreme.

![Figure 4 DeepSeek-v4 HCA](https://miro.medium.com/v2/resize:fit:700/1*ZJabnJVBu7IBgxOpcfbc4g.png)
*Screenshot of Figure 4 of DeepSeek-v4 paper, showing the core architecture of HCA.*

As stated in section 2.3.2, HCA employs a larger compression rate while not performing overlapped compression. The math (equations 20 to 23) is therefore very similar to equations (9) to (12).

In DeepSeek-v4, the compression rate m' is 128 for HCA (instead of 4 in CSA). Therefore, suppose there are 1 million context tokens, HCA attends to 1000000/128 ~= 7,800 compressed KV entries Cᵢᶜᵒᵐᵖ (note that i refers to block index and not attention head), without any top-k selection. And unlike CSA which has two KV-entry streams with overlapping windows, there is only one stream here (boundary precision is less central here, since HCA is intended to provide a coarser global memory.).

Fine local details can also be handled by the sliding-window branch (containing uncompressed KV entries) which is concatenated to the heavily compressed (hence coarse) blocks.

![Equations 20 to 26 HCA](https://miro.medium.com/v2/resize:fit:700/1*12Z8nZFyVJMjUdOm5_r5ew.png)
*Screenshot of equations (20) to (26) of [DeepSeek-v4 paper](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf), stitched together for compact presentation. Note that the key/value args within equation (26) has been written in a relaxed notation, because apart from Cᶜᵒᵐᵖ there should also be the non-compressed sliding-window entries, as mentioned in page 13 of the paper.*

Like in CSA, attention queries are then obtained by low-rank projections (equation 24 and 25), and thereafter shared key-value multi-query attention is performed (equation 26), where multiple query heads attend to the same shared KV entries for efficiency.

As stated in 4.2.1 of the DeepSeek-v4 paper, CSA and HCA are used in an interleaved manner, and for both, the number of query heads 𝑛ₕ = 128, head dimension 𝑐 = 512, and the dimension of each intermediate attention output is 1024.

**o**ₜˏᵢ computed from equation (26) is the core attention output of head i at token t. If we concatenate all head outputs, the dimensions will be 128 × 512 = 65536 which is large. Hence it is split into g=16 groups, with each projected into a 1024-dimensional intermediate output oₜˏᵢᴳ′. Finally, [oₜˏ₁ᴳ′ ; oₜˏ₂ᴳ′ ; … ; oₜˏ₁₆ᴳ′] are projected to the final attention output.

The DeepSeek-v4 authors also noted (in section 2.3.3) that before the core attention operation (MQA), RMSNorm is performed "*on each head of the queries and the only head of the compressed KV entries*", and for each query and KV entry vector, they "*apply RoPE to its last 64 dimensions*".

---

### 3.4 Muon optimizer

The muon optimizer has been introduced back in 2024, in an [online post](https://kellerjordan.github.io/posts/muon/) by Jordan, although it is not a formal academic paper or conference proceeding. (Then again, DeepSeek-v4 is also released as a technical report rather than academic paper or conference).

Now, you can simply run the Muon optimizer from [Pytorch](https://docs.pytorch.org/docs/2.11/generated/torch.optim.Muon.html).

The name comes from **M**oment**U**m **O**rthogonalized by **N**ewton-Schulz. Was not expecting that, and with this, I won't be picturing Muon as an unstable subatomic particle in my mind now.

Muon optimizes neural network by "*taking the updates generated by SGD-momentum, and then applying a Newton-Schulz (NS) iteration … before applying them to the parameters*". Implementation of NS is just a couple of lines of code.

![Muon Jordan post](https://miro.medium.com/v2/resize:fit:700/1*b77nfhLJhYaw8o1qB3WJVQ.png)
*Screenshot from Jordan (2024) https://kellerjordan.github.io/posts/muon/*

Jordan (2024) clarified that Muon differs from Orthogonal-SGDM, in that it "*moves the momentum to before the orthogonalization which performs better empirically*", and "*uses a NS iteration instead of SVD*" for efficiency. Earlier works were cited, along with alternatives for orthogonalizing matrix, a proof that NS iteration works (basically by transforming its singular values toward 1, thereby moving the matrix toward its orthogonal factor UVᵀ, without computing the SVD), and empirical observations. But we'll not go down the rabbit hole, and instead move on to understand how it is applied.

> *For a deeper explanation of SVD, refer to [Singular Value Decomposition — Explained with code](https://medium.com/mitb-for-all/singular-value-decomposition-explained-step-by-step-with-code-4ee86b12a021). For concrete numbers on how gradient descent works, refer to my [worked example on updating weights by gradient descent](https://medium.com/mitb-for-all/a-worked-example-on-updating-weights-by-gradient-descent-9a12181ed2b9) (which also has thousands of reads).*

Here's how Muon works at the high level. It looks very palatable when you put it side-by-side with something you already know, like SGD with momentum.

![Equations for SGD and Muon](https://miro.medium.com/v2/resize:fit:700/1*_QY5qENGxcAuDqOazooKEg.png)
*Equations for SGD without and with momentum, and Muon (simplified at the high level, with details encapsulated within the NS operator). W refers to the trainable parameters ie. network weights, while M refers to the momentum.*

In DeepSeek-v4, the authors stated that they implement hybrid NS with 10 iterations over two distinct stages (with specific combinations of coefficients *a*, *b* and *c*), for driving rapid convergence and then stabilizing the singular values. The pseudocode for implementation is as follows:

![Algorithm 1 Muon](https://miro.medium.com/v2/resize:fit:700/1*J6rgzpBOrWlwnX7k6ScINA.png)
*Screenshot of Algorithm 1 of [DeepSeek-v4 paper](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf).*

The '*Nesterov trick*' in line 5 refers to computing gradients at the look-ahead rather than current parameters. The '*Rescale update RMS*' in line 6 is as such because for a semi-orthogonal matrix in ℝⁿˣᵐ, its entries have RMS equal to 1/sqrt(max(n,m)). (A matrix is semi-orthogonal if its columns or rows are orthonormal. The RMS can be derived in a couple of lines.)

The DeepSeek-v4 authors wrote that [AdamW](https://arxiv.org/pdf/1711.05101) (Adam with decoupled weight decay) optimizer was used for the embedding module, prediction head, mHC and weights of RMSNorm modules, while all others are updated with Muon.

![Adam and AdamW](https://miro.medium.com/v2/resize:fit:700/1*Kn3F3CpTgH4i5YaZ_9VlXA.png)
*Equations for Adam and AdamW, by author. Note that it is common for sources to refer to the first moment as m and second moment as v. However, I will use M and S instead, to be consistent with the above, M for 'momentum' (or first moment) and S for 'second moment' (helpful to remember squared too).*

---

### 3.5 Infrastructures and Post-Training

There are also many other clever engineering aspects that makes DeepSeek-v4 possible, and we won't be able to replicate anything similar at home. There's another 29 pages of contents for sections 3, 4 and 5 of the DeepSeek-v4 paper, and this is where I draw the line.

Looking at the author list in Appendix A.1 of the paper, there are 270 authors listed, just under 'Research & Engineering'.

![Author count computation](https://miro.medium.com/v2/resize:fit:700/1*AFBiUDqvBvSxPuJppG_nXg.png)
*Computation of number of authors under 'Research & Engineering'.*

No one in this world can convince me that this work would not have been possible if not for each and every one of the 270. But still, it's clear that DeepSeek-v4 is the product of many brilliant minds working together. I don't plan to touch things like "*Expert Parallelism that fuses communication and computation*" or "*CUDA-based mega-kernel*" or "*translating TileLang's integer expressions into Z3's QF_NIA*". There's simply too many other things to learn and to work on.

### Conclusion

A year from now, these would probably be 'old stuff' in our fast-moving industry. However, I'm pretty confident that you'll need to understand the concepts covered here as well as in the [previous](https://medium.com/mitb-for-all/deepseek-v4-and-before-getting-the-foundations-right-d45ae9a07724) article (on fundamentals) in order to keep up with the new developments.

This is the sort of guide I wish I had at the beginning, and now, I share them with you.

*Disclaimer: All opinions and interpretations are that of the writer, and not of MITB. I declare that I have full rights to use the contents published here, and nothing is plagiarized. I declare that this article is written by me and not with any generative AI tool such as ChatGPT. I declare that no data privacy policy is breached, and that any data associated with the contents here are obtained legitimately to the best of my knowledge. I agree not to make any changes without first seeking the editors' approval. Any violations may lead to this article being retracted from the publication.*
