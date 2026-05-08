---
slug: ch07-the-middle-ground-grouped-query-attention-gqa
title: "2.7 The middle ground: Grouped-Query Attention (GQA)"
layout: chapter
---

This trade-off sacrificing model expressivity for memory efficiency is not ideal. It led researchers to seek a more balanced approach, a technique that could offer substantial memory savings without completely dismantling the power of the multi-head design. This solution is Grouped-Query Attention (GQA).

GQA provides a pragmatic compromise between the high expressivity of MHA and the significant memory efficiency of MQA. It lies somewhere in the middle, offering a tunable knob to balance these competing priorities.

### 2.7.1 The core idea: Sharing keys and values within groups

The core idea of Grouped-Query Attention is simple but effective: instead of forcing all attention heads to share the same Key and Value matrices, what if we create groups of attention heads and only share the Keys and Values within those groups?

Let's visualize what this means. In our four-head example, instead of treating all four heads as one single unit (like in MQA), we can partition them into two groups.

![Figure 2.25 Grouped-Query Attention (GQA). The four attention heads are divided into two groups. Within Group 1 (light blue/light yellow), Head 1 and Head 2 share the same Key and Value projections. Within Group 2 (dark blue/dark yellow), Head 3 and Head 4 share a different, distinct set of Key and Value projections.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/02__image028.png)

As figure 2.25 illustrates, we've created a hybrid model:

**Within Group 1:** Head 1 and Head 2 share the same Wk and Wv matrices. Their resulting K1 and K2 are identical, and V1 and V2 are identical.

**Within Group 2:** Similarly, Head 3 and Head 4 share their own set of Wk and Wv matrices, making K3 identical to K4, and V3 identical to V4.

**Between Groups:** Crucially, the K/V pair for Group 1 is different from the K/V pair for Group 2. The light blue K1/K2 matrices are distinct from the dark blue K3/K4 matrices.

This grouping strategy elegantly solves the main drawback of MQA. We are no longer forcing all heads to look at the same information. Now, Head 1 (in Group 1) and Head 3 (in Group 2) have different Key and Value matrices, allowing them to specialize and capture different perspectives, just like in standard MHA. We have reintroduced diversity into the system.

At the same time, we are still saving a significant amount of memory compared to MHA. Instead of caching four unique Key matrices, we only need to cache two: one for Group 1 and one for Group 2. GQA provides a middle ground, allowing us to find a sweet spot between model performance and memory cost.

### 2.7.2 The tunable knob: Balancing memory and performance

The introduction of groups in GQA provides a powerful "tunable knob" to balance the trade-off between memory efficiency and model expressivity. The number of groups, which we'll call g, directly controls this balance.

Let's revisit the KV Cache size formula.

*   For **MHA**, the size scaled with n (the total number of attention heads).
*   For **MQA**, the size scaled with 1 (a single shared K/V pair).
*   For **GQA**, the size now scales with g (the number of unique groups).

The formula becomes:

$$ Size_{GQA} = l * b * g * h * s * 2 * 2 $$

This gives us a spectrum of possibilities:

*   If we set the number of groups equal to the number of heads (g = n), GQA becomes identical to MHA. We have maximum performance and maximum memory usage.
*   If we set the number of groups to one (g = 1), GQA becomes identical to MQA. We have maximum memory savings and the lowest performance.
*   By choosing a value for g between 1 and n, we can find a practical middle ground.

For example, a model like Llama 3 8B has 32 total attention heads. Instead of the extremes of MHA (32 unique K/V pairs) or MQA (1 unique K/V pair), it uses GQA with 8 groups. This means that every 4 query heads share a single key/value head.

This reduces the KV cache size by a factor of 4 (from 32 to 8), offering a significant memory saving while retaining much more of the expressive power than MQA would. This balanced approach has made GQA a very popular choice in modern, open-source LLMs. It offers a practical way to manage the KV cache bottleneck without a crippling hit to the model's performance.

However, it is still fundamentally a compromise. We are trading some amount of model expressivity for a reduction in memory. While GQA is a clever and effective optimization, it doesn't solve the core tension between performance and memory; it just allows us to choose a better point on the trade-off curve.

This led the DeepSeek team to ask a different, more profound question: can we fundamentally change the nature of this trade-off? Is it possible to keep the full expressive power of having unique projections for every head (like in MHA) while also achieving significant memory reduction?

The answer to that question is yes, and the solution is Multi-Head Latent Attention. But before we start into that groundbreaking technique, let's solidify our understanding of GQA by implementing it from scratch.

### 2.7.3 Implementing a GQA layer from scratch

Implementing Grouped-Query Attention is a natural extension of our MQA code. The key difference is that instead of having one shared projection for Keys and Values, we now have `num_groups` of them. We then ensure that the query heads within each group attend to the corresponding key/value group.

The following listing implements a GroupedQueryAttention module. The key variable to watch is `num_groups`, which acts as the "tunable knob.” It directly controls the number of Key and Value projections, allowing us to balance memory savings and model performance.

##### Listing 2.4 Implementing a GQA layer from scratch

```python
import torch
import torch.nn as nn
 
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_groups, 
    dropout=0.0, max_seq_len: int = 0):
        super().__init__()
        assert d_model % num_heads == 0, \
            "d_model must be divisible by num_heads"
         assert num_heads % num_groups == 0, \
             "num_heads must be divisible by num_groups"
 
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.d_head = d_model / num_heads
 
        self.W_q = nn.Linear(d_model, d_model)
        # Instead of creating a single projection (d_head) like in MQA, 
        # we create num_groups projections. This parameter acts as a "tunable knob": 
        # if num_groups is 1, this is MQA; if num_groups equals num_heads, this becomes standard MHA.
        self.W_k = nn.Linear(d_model, self.num_groups * self.d_head)
        self.W_v = nn.Linear(d_model, self.num_groups * self.d_head)
        self.W_o = nn.Linear(d_model, d_model)
 
        self.dropout = nn.Dropout(dropout)
        # Optional causal mask pre-allocation logic...
        self._register_mask_buffer(max_seq_len)
 
    def forward(self, x):
        B, T, _ = x.shape
        
        q = self.W_q(x).view(B, T, self.num_heads, 
        self.d_head).transpose(1, 2)

        # The input is projected and reshaped into num_groups distinct Key and Value groups.
        k = self.W_k(x).view(B, T, self.num_groups, 
        self.d_head).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.num_groups, 
        self.d_head).transpose(1, 2)
 
        heads_per_group = self.num_heads / self.num_groups
        
        # repeat_interleave broadcasts the K/V groups to the query heads. 
        # Each of the num_groups of Keys and Values is shared across heads_per_group queries. 
        # For example, if there are 8 query heads and 2 K/V groups, the first K/V group is shared 
        # by the first 4 query heads, and the second group is shared by the last 4 query heads.
        k = k.repeat_interleave(heads_per_group, dim=1)
        v = v.repeat_interleave(heads_per_group, dim=1)
 
        # ... rest of attention calculation ...
        attn_scores = (q @ k.transpose(-2, -1)) * (self.d_head**-0.5)
        causal_mask = self._get_causal_mask(T, x.device)
        attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = (attn_weights @ v).transpose(1, 2).contiguous() \
        .view(B, T, self.d_model)
        
        return self.W_o(context)
 
    # Helper methods for mask management
    def _register_mask_buffer(self, max_seq_len):
        if max_seq_len > 0:
            mask = torch.triu(torch.ones(1, 1, max_seq_len, max_seq_len,
            dtype=torch.bool), diagonal=1)
            self.register_buffer("causal_mask", mask, persistent=False)
        else:
            self.causal_mask = None
 
    def _get_causal_mask(self, seq_len, device):
        if self.causal_mask is not None and \
        self.causal_mask.size(-1) >= seq_len:
            return self.causal_mask[:, :, :seq_len, :seq_len]
        return torch.triu(torch.ones(1, 1, seq_len, seq_len, 
        dtype=torch.bool, device=device), diagonal=1)
```

This implementation provides the "tunable knob" we discussed. By simply changing the `num_groups` argument, we can move seamlessly along the spectrum from MQA-like behavior (`num_groups=1`) to MHA-like behavior (`num_groups=num_heads`).