---
slug: ch16-building-mla-decoupled-rope-from-scratch
title: "3.16 Building MLA + decoupled RoPE from scratch"
layout: chapter
---

We have now covered all the theoretical and mathematical foundations of Multi-Head Latent Attention and Decoupled Rotary Position Embedding. The final step is to translate this knowledge into a functional PyTorch module. The complete, executable code for this chapter can be found in the book's official GitHub repository. We highly recommend having it open to follow along with the implementation:

[https://github.com/VizuaraAI/DeepSeek-From-Scratch/tree/main/ch03](https://github.com/VizuaraAI/DeepSeek-From-Scratch/tree/main/ch03)

Our implementation will construct the `DeepSeekAttention` module, which contains the entire decoupled architecture. Before we write the code, let's look at a complete visual blueprint of the module's forward pass in figure 3.36.

##### Figure 3.36 A detailed flowchart of the DeepSeekAttention module. The input tensor x is processed through two parallel streams: the blue Content Path, which uses pure MLA for cache efficiency, and the purple Position Path, which applies RoPE. The two cached tensors, c_kv and k_r, are highlighted with key icons. The outputs are combined to produce the final context vector.
![Figure 3.36 A detailed flowchart of the DeepSeekAttention module. The input tensor x is processed through two parallel streams: the blue Content Path, which uses pure MLA for cache efficiency, and the purple Position Path, which applies RoPE. The two cached tensors, c_kv and k_r, are highlighted with key icons. The outputs are combined to produce the final context vector.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/03__image058.png)

This diagram is a direct map of the code we are about to build. It clearly shows the two parallel pathways:

1.  **Path A: The Content Path**: This stream is responsible for understanding the semantic content of the tokens. It uses the pure MLA architecture, creating the compressed latent tensor `c_kv` which is the only content-based value we need to cache. It also produces the content-based Query (`q_c`), Key (`k_c`), and Value (`v_c`) matrices.
2.  **Path B: The Position Path**: This stream is dedicated to understanding the relative positions of tokens. It creates specialized Key (`k_rope`) and Query (`q_rope`) vectors, which are then infused with positional information by the RoPE module. The resulting rotated key, `k_r`, is the second value we cache.

The attention scores from these two paths are added together. The resulting attention weights are then applied to the content-only Value (`v_c`) to produce the final context vector.

Breaking down a complex architecture like this into smaller, manageable pieces is a key engineering skill. We will follow this principle by constructing our `DeepSeekAttention` module in stages, starting with the self-contained RoPE helper class.

**Part 1: The RotaryPositionalEncoding Helper Class**

Before we can build the main attention module, we need a way to implement RoPE. It's a self-contained concept, making it the perfect candidate for our first code snippet. We will present the `RotaryPositionalEncoding` class in its entirety because it's a single, cohesive unit.

##### Listing 3.2 The RotaryPositionalEncoding Helper Module
```python
import torch
import torch.nn as nn
 
class RotaryPositionalEncoding(nn.Module):
    def __init__(self, d_head, max_seq_len=2048):
        super().__init__()
        # Precomputes the base frequencies (theta) for the rotations, one for each pair of dimensions in the head.
        theta = 1.0 / (10000 ** (
        torch.arange(0, d_head, 2).float()
        / d_head))
        self.register_buffer('theta', theta)
        
        # Precomputes the full rotational matrices (freqs_cis) for all positions up to max_seq_len using the complex number representation cos(theta) + i*sin(theta).
        positions = torch.arange(max_seq_len).unsqueeze(1)
        freqs = positions * self.theta.unsqueeze(0)
        
        self.register_buffer('freqs_cis', 
        torch.polar(torch.ones_like(freqs),
        freqs))
 
    def forward(self, x):
        seq_len = x.shape[2]
        # Reshapes the input vector by pairing adjacent dimensions and casting them as complex numbers, ready for rotation.
        x_complex = x.float().reshape(*x.shape[:-1], -1, 2)
        x_complex = torch.view_as_complex(x_complex)
        
        freqs_cis = self.freqs_cis[:seq_len, :].unsqueeze(0).unsqueeze(0)
        
        # Applies the rotation by performing element-wise multiplication in the complex plane. This is the core mathematical operation of RoPE.
        x_rotated = x_complex * freqs_cis
        
        x_rotated = torch.view_as_real(x_rotated)
        x_rotated = x_rotated.flatten(3)
        return x_rotated.type_as(x)
```

**Step 2: Initializing the Main Attention Class**

With our `RotaryPositionalEncoding` module ready, we can now define the main `DeepSeekAttention` class. The `__init__` method is responsible for creating all the learnable weight matrices (`nn.Linear` layers) and instantiating our RoPE helper.

Notice how the weight matrices are grouped into two distinct sections in the code, mirroring our decoupled architecture. One set is for the content path, which uses the standard MLA projections. The other set is for the position path, which creates the specialized Query and Key vectors that will be infused with positional information via RoPE.

##### Listing 3.3 Initializing the DeepSeekAttention Module
```python
class DeepSeekAttention(nn.Module):
    def __init__(self, d_model, num_heads, d_latent, d_rope, dropout=0.0):
        super().__init__()
        # ... (asserts and initializations) ...
        self.d_rope = d_rope
        self.num_heads = num_heads
        self.d_head = d_model // num_heads # Assuming d_model is divisible by num_heads
        self.d_model = d_model

        # --- Content Path (Pure MLA) ---
        # The set of projection matrices for the content path, which uses the standard MLA architecture to create a cache-friendly representation of token semantics.
        self.W_q_content = nn.Linear(d_model, d_model)
        self.W_dkv_content = nn.Linear(d_model, d_latent)
        self.W_uk_content = nn.Linear(d_latent, d_model)
        self.W_uv_content = nn.Linear(d_latent, d_model)
        
        # --- Position Path (RoPE Applied) ---
        # The set of projection matrices for the separate position path. These create specialized Query and Key vectors designed to be encoded with positional information.
        self.W_k_pos = nn.Linear(d_model, d_rope * num_heads)
        self.W_q_pos = nn.Linear(d_model, d_rope * num_heads)
        
        # Instantiates the RoPE helper module, which will be used to apply the positional rotations.
        self.rope = RotaryPositionalEncoding(d_rope)
        
        # --- Final Output Projection ---
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        # ... (mask initialization) ...
        self.register_buffer('mask', torch.triu(torch.ones(1, 1, 1024, 1024), diagonal=1).bool())
```

**Step 3: The Forward Pass - Combining Content and Position**

The forward method handles the entire decoupled attention calculation. It takes the input tensor `x` and passes it through both the content and position paths simultaneously to compute two separate attention score matrices. These are then added together to form the final scores, which are used to generate the context vector.

This implementation is a direct translation of the architecture we derived. It keeps the cache-friendly "absorption trick" alive for the content path while cleanly integrating the position-dependent rotations in a separate, parallel stream.

##### Listing 3.4 The forward Method of DeepSeekAttention
```python
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
 
        # --- Content Path Calculation ---
        q_c = self.W_q_content(x).view(
                 batch_size, seq_len, self.num_heads, self.d_head
             ).transpose(1, 2)
        c_kv = self.W_dkv_content(x)
        k_c = self.W_uk_content(c_kv).view(
                 batch_size, seq_len, self.num_heads, self.d_head
             ).transpose(1, 2)
        v_c = self.W_uv_content(c_kv).view(
                 batch_size, seq_len, self.num_heads, self.d_head
             ).transpose(1, 2)
        
        # --- Position Path Calculation ---
        q_r_unrotated = self.W_q_pos(x).view(
                 batch_size, seq_len, self.num_heads, self.d_rope
             ).transpose(1, 2)
        k_r_unrotated = self.W_k_pos(x).view(
                 batch_size, seq_len, self.num_heads, self.d_rope
             ).transpose(1, 2)
 
        # Applies Rotary Positional Encoding to the unrotated positional Query and Key vectors. This is where positional information is injected.
        q_r = self.rope(q_r_unrotated)
        k_r = self.rope(k_r_unrotated)
        
        # --- Combining Paths for Final Attention Score ---
        content_scores = (q_c @ k_c.transpose(-2, -1)) / \
                 (self.d_head ** 0.5)
        position_scores = (q_r @ k_r.transpose(-2, -1)) / \
                 (self.d_rope ** 0.5)
        
        # The final attention score is the sum of the content-based scores and the position-based scores, elegantly combining the two parallel paths.
        attn_scores = content_scores + position_scores
        
        # --- Final Steps (Masking, Softmax, Output) ---
        attn_scores = attn_scores.masked_fill(
                 self.mask[:, :, :seq_len, :seq_len], float('-inf'))
            
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context_vector = (attn_weights @ v_c).transpose(1, 2) \
                 .contiguous().view(batch_size, seq_len, self.d_model)
            
        output = self.W_o(context_vector)
        return output
```

With this final snippet, we have constructed a complete, from-scratch implementation of the `DeepSeekAttention` module. This single class encapsulates a series of advanced techniques:

*   **A memory-efficient content pathway** using Multi-Head Latent Attention.
*   **A separate, specialized position pathway** for handling relative positions.
*   **Clean integration of Rotary Positional Encoding** where it is most effective.
*   **A decoupled fusion** of these two pathways through simple addition.

This architecture masterfully resolves the inherent conflict between the position-agnostic caching of MLA and the position-dependent nature of RoPE. By trading a slight increase in computational cost for a massive reduction in memory bandwidth requirements, this design achieves the best of both worlds, enabling efficient and powerful long-context inference.