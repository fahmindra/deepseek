---
slug: ch06-building-an-mla-module-from-scratch
title: "3.6 Building an MLA module from scratch"
layout: chapter
---

Now that we have a solid grasp of the theory and the mathematical workflow of MLA, we are ready to implement it in code. We will build a PyTorch module that encapsulates the entire logic: the down-projection, the up-projection, and the final attention calculation.

For this implementation, we will focus on the core MLA mechanism itself. We will handle the caching logic outside of this module in our main inference loop, and we will address positional encodings further in this chapter.

##### Listing 3.1 Building MLA From Scratch
```python
import torch
import torch.nn as nn
 
class MultiHeadLatentAttention(nn.Module):
    def __init__(self, d_model, num_heads, d_latent, dropout=0.0):
        super().__init__()
        assert d_model % num_heads == 0, \
            "d_model must be divisible by num_heads"
 
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads # Corrected integer division
        self.d_latent = d_latent
 
        self.W_q = nn.Linear(d_model, d_model)
        self.W_dkv = nn.Linear(d_model, d_latent) # The compression layer
 
        self.W_uk = nn.Linear(d_latent, d_model) # The decompression layer for Keys
        self.W_uv = nn.Linear(d_latent, d_model) # The decompression layer for Values
 
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(
             1, 1, 1024, 1024), diagonal=1).bool())
 
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
 
        # 1. Query Path
        # The Query projection remains standard, projecting to the full model dimension before being split into heads.
        q = self.W_q(x).view(
                 batch_size, seq_len, self.num_heads, self.d_head
             ).transpose(1, 2)
 
        # 2. Key/Value Path (The MLA Innovation)
        # Down-project to the latent space (Compression)
        # This compressed c_kv is the only value that gets cached during autoregressive inference.
        c_kv = self.W_dkv(x)
 
        # Up-project from latent space to get full K and V (Decompression)
        # These values are used for the attention calculation but are not cached.
        k = self.W_uk(c_kv).view(
                 batch_size, seq_len, self.num_heads, self.d_head
             ).transpose(1, 2)
        v = self.W_uv(c_kv).view(
                 batch_size, seq_len, self.num_heads, self.d_head
             ).transpose(1, 2)
 
        # 3. Standard Attention Calculation
        attn_scores = (q @ k.transpose(-2, -1)) / \
                 (self.d_head ** 0.5)
 
        attn_scores = attn_scores.masked_fill(
                 self.mask[:, :, :seq_len, :seq_len], float('-inf'))
 
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
 
        context_vector = (attn_weights @ v).transpose(1, 2) \
                 .contiguous().view(batch_size, seq_len, self.d_model)
 
        # 4. Final Output Projection
        output = self.W_o(context_vector)
        return output
```

While Multi-Head Latent Attention optimizes the Transformer's ability to process information efficiently, the attention module is still fundamentally "time-blind." It has no inherent sense of word order, treating sentences as a "bag of words." For LLMs to truly understand context and generate coherent text, they need to know the position of each token. This leads us to the next crucial step on our path: positional awareness.