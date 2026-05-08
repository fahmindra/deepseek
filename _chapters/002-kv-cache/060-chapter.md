---
slug: ch06-the-memory-first-approach-multi-query-attention-mqa
title: "2.6 The memory-first approach: Multi-Query Attention (MQA)"
layout: chapter
---

What is the simplest, most direct thing one could do to solve the KV Cache memory problem? Multi-Query Attention (MQA) answers this question with a radical proposal: what if all the attention heads simply share the same Key and Value matrices?

### 2.6.1 The core idea: Sharing a single key and value

To understand MQA's innovation, we must first recall how standard Multi-Head Attention (MHA) works. In a standard Transformer, within each layer, each attention head acts as an independent expert. This means it has its own unique, learned weight matrices for Keys (Wk) and Values (Wv), distinct from all other heads in that layer. This can be summarized as: in vanilla MHA, each head has distinct Wk and Wv.

![Figure 2.23 Standard Multi-Head Attention (MHA). Each of the four attention heads has its own distinct Key and Value weight matrices, indicated by the different colors. This allows each head to specialize and learn different patterns.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/02__image024.png)

As shown in figure 2.23, if we have four attention heads, the Key weight matrix is effectively split into four unique parts: Wk1, Wk2, Wk3, and Wk4. The same is true for the Value weight matrix. Because these weights are initialized randomly and trained independently, each head learns to project the input embeddings into a different representational space. Q1 is different from Q2, V1 is different from V2, and so on. This diversity is the source of MHA's power; it allows the model to capture multiple perspectives simultaneously.

However, this is also the source of its memory problem. To enable fast inference, we must cache the full Key and Value matrices for every single head. Multi-Query Attention takes a direct and aggressive approach to solve this. It proposes a simple change: while each head still gets its own unique Query projection (allowing each head to "ask" a different question), all heads are forced to share one single, common set of Key and Value projections.

![Figure 2.24 Multi-Query Attention (MQA). All four heads still have unique Query projections, but they now share a single, common Key and Value projection, indicated by the uniform light blue and yellow colors.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/02__image025.png)

Look closely at the difference in figure 2.24. The Key weight matrices for all heads (Wk1 through Wk4) are now identical, and the same is true for the Value weight matrices. This means that when the input embedding is projected, the resulting K1, K2, K3, and K4 matrices are all exact copies of each other. The same applies to the Value matrices.

The implication for caching is immediate and profound. Instead of needing to store four separate Key matrices and four separate Value matrices in our cache, we only need to store one Key matrix and one Value matrix. During inference, each of the four query heads will simply attend to this single, shared set of keys and values.

This simple architectural tweak is the core idea behind Multi-Query Attention. It prioritizes memory savings above all else. In the next sections, we will explore the dramatic impact this has on the KV Cache formula and the inevitable trade-off it creates for model performance.

### 2.6.2 The impact on the KV cache formula

The architectural change from MHA to MQA has a dramatic and direct impact on the size of the KV Cache. Let's revisit the formula we established in Section 2.5.1:

![Equation](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/02__image026.png)

The key variable here is n, the number of attention heads. In MHA, because every head has its own unique Key and Value matrices, the total memory required scales linearly with the number of heads.

In Multi-Query Attention, since all heads share the same single Key and Value pair, we no longer need to store n different versions. We only need to store one. The formula becomes:

![Equation](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/02__image027.png)

In this revised formula, the n term is effectively replaced with 1, where n is the total number of attention heads, eliminating the linear scaling with the number of heads. This reduces the size of the KV Cache by a factor of n. The impact of this reduction is staggering for large models:

GPT-3 (175B): This model has 96 attention heads (n=96). Using MQA would reduce its KV Cache size by a factor of 96, from 4.5 GB down to a mere 48 MB.

DeepSeek-V3 (671B): This model has 128 attention heads (n=128). MQA would reduce its theoretical KV Cache size by a factor of 128, from ~400 GB down to just over 3 GB.

This is an incredible reduction in memory footprint, and it directly translates to faster inference (as we will see in the code) because less data needs to be loaded from memory at each step. So, if MQA is so effective at solving the memory problem, why doesn't every model use it?

### 2.6.3 The performance trade-off: Loss of expressivity

The dramatic memory savings of MQA seem almost too good to be true, and in a way, they are. This efficiency comes at a significant cost: a degradation in the model's performance and its ability to understand complex language. To understand this trade-off, we must revisit the fundamental reason we use multi-head attention in the first place.

Let's consider the following ambiguous sentence:

> *"The artist painted the portrait of a woman with a brush."*

This sentence has at least two possible interpretations:

1.  Interpretation A (Instrument): The artist used a brush to paint the portrait. (painted with a brush)
2.  Interpretation B (Attribute): The portrait is of a woman who is holding a brush. (woman with a brush)

A sophisticated language model needs to be able to understand and disentangle both of these potential relationships simultaneously. This is precisely what Multi-Head Attention (MHA) was designed to do.

**How MHA Handles Ambiguity?**

In a standard MHA block, each attention head is an independent "expert analyst" with its own set of learned weights (Wk and Wv). This independence allows them to specialize. During training, the model might learn the following:

*   **Head 1** could specialize in syntactic relationships. Its learned weights might cause its Key and Value vectors to focus on verb-instrument pairings. When its Query for the token "painted" looks at the Key for "brush," it would calculate a very high attention score, effectively capturing the meaning: "The action of painting was done using a brush."
*   **Head 2**, on the other hand, could specialize in semantic or descriptive relationships. Its unique weights might cause its Key and Value vectors to focus on noun-attribute pairings. When its Query for "woman" looks at the Key for "brush", it might calculate a high score, capturing the alternative meaning: "The woman in the portrait is associated with a brush."

Because K1 is different from K2, and V1 is different from V2, the model can process both interpretations in parallel. The final context vectors contain a rich, blended understanding of all the different relationships detected by all the heads. This is the source of MHA's expressive power.

**How MQA Loses This Power?**

Now, let's consider what happens in Multi-Query Attention. MQA forces all heads to share a single, common Key and Value matrix. K1 is now identical to K2, and V1 is identical to V2.

This creates a critical problem. The single, shared Key matrix can no longer specialize. It must try to be a jack-of-all-trades, encoding a generic representation of the sentence's information. It cannot be an expert at understanding two different kinds of relationships at the same time. For example, it struggles to precisely capture both that the brush is a tool for painting and that it is an object held by the woman.

When Head 1 (the syntax expert) and Head 2 (the semantics expert) both send out their unique queries, they are both looking at the exact same, generic set of keys. The shared Key for "brush" cannot effectively signal both "I am an instrument for painting" and "I am an object held by a woman" at the same time. One of these nuances will likely be weakened or lost entirely.

This is the fundamental drawback of MQA:

> By forcing all heads to share the same Key and Value representations, MQA severely restricts their ability to specialize. The model loses a significant amount of its capacity to capture diverse and subtle relationships within the text, leading to a degradation in overall performance.

While MQA is a brilliant solution for the memory problem, it achieves this by fundamentally compromising the core strength of the multi-head design. This is why it's often seen as a "memory-first" approach. This significant performance trade-off is what led researchers to seek a more balanced middle ground, which we will explore later with Grouped-Query Attention. But first, let's see how we can implement this memory-saving MQA architecture in code.

### 2.6.4 Implementing an MQA layer from scratch

Implementing Multi-Query Attention in PyTorch is straightforward. The core logic of the attention calculation remains the same; the only change is in how the Key and Value projections are handled. Instead of creating `num_heads` different projections, we create only one and then repeat it for all heads.

The following code defines a MultiQueryAttention module. Pay close attention to the `__init__` method, where the architectural difference is most apparent.

##### Listing 2.3 Implementing an MQA layer from scratch

```python
import torch
import torch.nn as nn
 
class MultiQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        assert d_model % num_heads == 0, \
            "d_model must be divisible by num_heads"
 
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model / num_heads
 
        self.W_q = nn.Linear(d_model, d_model)
        # The Key and Value projections are now single, shared linear layers. 
        # They project down to the dimension of a single head (d_head) 
        # because only one projection is being created, not num_heads. 
        # This is the core architectural change that saves KV cache memory.
        self.W_k = nn.Linear(d_model, self.d_head)
        self.W_v = nn.Linear(d_model, self.d_head)
        self.W_o = nn.Linear(d_model, d_model)
 
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(
        torch.ones(1, 1, 1024, 1024), diagonal=1))
 
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
 
        # The Query projection remains the same as in standard Multi-Head Attention.
        # It projects to the full model dimension, which is then split among the heads. 
        # This allows each head to "ask" a unique question.
        q = self.W_q(x).view(batch_size, seq_len, self.num_heads, 
        self.d_head).transpose(1, 2)
 
        k = self.W_k(x).view(batch_size, seq_len, 1, 
        self.d_head).transpose(1, 2)
        v = self.W_v(x).view(batch_size, seq_len, 1, 
        self.d_head).transpose(1, 2)
 
        # The single Key and Value tensors are "repeated" or broadcast to match 
        # the number of query heads. This is how all heads are made to share 
        # the same K and V information without creating expensive data copies in memory. 
        # It's the implementation of the "sharing" mechanism.
        k = k.repeat(1, self.num_heads, 1, 1)
        v = v.repeat(1, self.num_heads, 1, 1)
 
        attn_scores = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)
 
        attn_scores = attn_scores.masked_fill(
        self.mask[:,:,:seq_len,:seq_len] == 0, float('-inf'))
 
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
 
        context_vector = (attn_weights @ v).transpose(1, 2) \
        .contiguous().view(batch_size, seq_len, self.d_model)
 
        output = self.W_o(context_vector)
        return output
```

Let's break down the key differences between this MultiQueryAttention module and a standard MultiHeadAttention module:

*   **Key and Value Projections:** In a standard MHA, the output dimension of `W_k` and `W_v` would be `d_model`. Here, they project down to `d_head`, the size of a single head. This is because we are only creating one projection, not `num_heads` projections that are later split.
*   **Repeating K and V:** The magic of MQA happens in the forward pass. After computing the single Key and Value projections, we use the `.repeat()` method. This doesn't actually copy data in memory in the same way as a full matrix would; instead, it creates a "view" of the data where the same Key and Value tensors are presented to each of the `num_heads` Query heads. This is how "sharing" is implemented efficiently.
*   **Efficiency Gain:** The primary gain comes from the reduced size of the Key and Value caches. In an MHA implementation, we would need to cache tensors of shape `(batch_size, num_heads, seq_len, d_head)` for both Keys and Values. In MQA, we only need to cache tensors of shape `(batch_size, 1, seq_len, d_head)`, drastically reducing the memory footprint.

With this implementation, we have a functional attention layer that aggressively optimizes for memory, albeit at the cost of the performance trade-offs we've discussed. This sets the stage perfectly for exploring a more balanced solution.