---
slug: ch05-building-a-complete-deepseek-moe-language-model-from-scratch
title: "4.5 Building a complete DeepSeek-MoE language model from scratch"
layout: chapter
---

We have explored the core mechanics of Mixture-of-Experts, the challenge of load balancing, and the advanced solutions pioneered by DeepSeek. Now, we will integrate all these concepts into a single, functional PyTorch module. We will build the DeepSeekMoE class step-by-step, starting with its initialization. The complete code for this chapter is available in the book's GitHub repository, and we encourage you to follow along: [https://github.com/VizuaraAI/DeepSeek-From-Scratch/tree/main/ch04](https://github.com/VizuaraAI/DeepSeek-From-Scratch/tree/main/ch04).

Let's visualize the entire forward pass. Figure 4.27 provides a complete roadmap of the DeepSeekMoE module we are about to build, showing how it processes an input tensor $x$.

![Figure 4.27: The complete forward pass of the DeepSeekMoE module, showing the three parallel data paths: the dense shared expert path, the sparse routed expert path with dynamic load balancing, and the residual connection.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/04__image033.png)

As the figure illustrates, our module's logic is built around three parallel paths that are combined at the end:

1.  **The Shared Path**: Every input token is processed by a small set of "generalist" shared experts. This dense path ensures that common, foundational knowledge is consistently applied.
2.  **The Routed Path**: This sparse path is the core of the MoE mechanism. It involves a sequence of steps: calculating expert scores (logits), applying the dynamic bias for load balancing, selecting the top-$k$ experts, and finally processing tokens only through their assigned specialists.
3.  **The Residual Connection**: A direct path for the original input $x$, which is crucial for stable training and preserving information from the previous layer.

These three outputs are then summed together—$x + \text{shared\_out} + \text{routed\_out}$—to produce the final result. With this blueprint in mind, let's begin by defining the smallest building block of our system.

**Step 1: Defining the Expert FFN**

In the DeepSeek architecture, this is a two-layer MLP that performs the same "expansion-contraction" sequence we saw in dense models, though typically with a smaller hidden dimension. The code below defines this `ExpertFFN`, which includes a GELU activation function.

##### Listing 4.1 The ExpertFFN Module
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

def _gelu(x: torch.Tensor) -> torch.Tensor:
    # Slightly faster GELU (approx)
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) *
    (x + 0.044715 * torch.pow(x, 3))))
 
class ExpertFFN(nn.Module):
    """
    A 2-layer MLP expert. Hidden dim is usually smaller than a dense FFN
    (e.g., 0.25 × d_model in DeepSeek-V3).
    """
    def __init__(self, d_model: int, hidden: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(_gelu(self.fc1(x))))
```

At its heart, each "expert" in an MoE layer is simply a standard Feed-Forward Network (FFN). In the DeepSeek architecture, this is a two-layer MLP that performs the same "expansion-contraction" sequence we saw in dense models, though typically with a smaller hidden dimension. The code above defines this `ExpertFFN`, which includes a GELU activation function.

**Step 2: Initializing the MoE Layer**

The `__init__` method sets up all the components of our MoE layer. This includes creating the `ModuleLists` for our two distinct groups of experts: a small set of **shared experts** that every token will see, and a much larger pool of **routed experts** that will be used sparsely.

Crucially, it also sets up the parameters for the routing mechanism. This includes the `centroids` (the learnable weight matrix for the router) and the `bias`, a non-trainable buffer that will be dynamically updated to enforce load balancing without an auxiliary loss.

##### Listing 4.2 The DeepSeekMoE `__init__` Method
```python
class DeepSeekMoE(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_routed_exp: int,
        n_shared_exp: int = 1,
        top_k: int = 8,
        routed_hidden: int = 2_048,
        shared_hidden: Optional[int] = None,
        bias_lr: float = 0.01,
        fp16_router: bool = False,
    ):
        super().__init__()
        assert top_k <= n_routed_exp, "k must be <= number of routed experts"
 
        self.d_model = d_model
        self.n_routed = n_routed_exp
        self.n_shared = n_shared_exp
        self.top_k = top_k
        self.bias_lr = bias_lr
        self.fp16_router = fp16_router
 
        # A) Creates the large pool of specialized, sparsely-activated experts.
        self.routed = nn.ModuleList(
            [ExpertFFN(d_model, routed_hidden) for _ in range(n_routed_exp)]
        )
 
        # B) Creates the small set of generalist experts, activated for every token.
        hidden_shared = shared_hidden or routed_hidden
        self.shared = nn.ModuleList(
            [ExpertFFN(d_model, hidden_shared) for _ in range(n_shared_exp)]
        )
 
        # C) The learnable experts centroid (router). 
        # The router calculates expert routes by measuring the similarity (dot product) 
        # between each token’s representation and each expert’s centroid vector.
        self.register_parameter("centroids",
            nn.Parameter(torch.empty(n_routed_exp, d_model)))
        nn.init.normal_(self.centroids, std=d_model ** -0.5)
 
        # D) The non-trainable bias for auxiliary-loss-free load balancing.
        self.register_buffer("bias", torch.zeros(n_routed_exp))
```

**Step 3: The Forward Pass - The Shared Expert Path**

The forward method begins by processing the input $x$ through the shared experts. As we discussed in the theory, this is a dense operation: every token in the batch is passed through every expert in the `self.shared` list.

This path ensures that common, foundational knowledge (like grammar or basic facts) is handled by a consistent set of parameters, solving the "knowledge redundancy" problem. The output from these experts forms a base result, which will then be refined by the specialized routed experts.

##### Listing 4.3 The Shared Expert Path in the `forward` Method
```python
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        x_flat = x.reshape(-1, D)  # [N, D] with N=B*S
 
        # 1) shared path
        shared_out = torch.zeros_like(x_flat)
        for exp in self.shared:
            shared_out += exp(x_flat)
 
        # ... (routed path comes next)
```

In this section of the code, we first reshape the input tensor for easier processing. We then initialize an output tensor `shared_out` and iterate through our list of shared experts, accumulating their results. Note that for simplicity and clarity, our implementation processes the shared experts sequentially in a for loop. In a production-level framework, these expert computations would be performed in parallel to maximize throughput. Every token is processed by every shared expert, creating a dense output that captures the generalist knowledge required for the entire batch.

**Step 4: The Forward Pass - The Routed Expert Path**

This is where the sparse part of the MoE architecture comes into play. After the shared experts have processed the tokens, the routing mechanism takes over to select a small, specialized subset of "routed" experts for each token. This section of the forward pass handles the entire routing and dispatching logic.

The process involves four key stages:

1.  **Calculate Router Logits**: A linear layer (the "router") calculates a raw score for each expert for every token.
2.  **Apply Dynamic Bias**: The auxiliary-loss-free bias term is added to these scores to dynamically encourage a balanced load.
3.  **Top-K Gating**: A `topk` operation selects the best experts, and a softmax function converts their scores into normalized weights.
4.  **Dispatch and Combine**: Each token is sent to its chosen experts, and their outputs are combined in a weighted sum.

##### Listing 4.4 The Routed Expert Path and Final Combination
```python
        # ... (shared path from previous listing)
 
        # 2) router logits in (optional) mixed precision
        # A) The router calculates raw scores for each expert.
        use_autocast = self.fp16_router and x.is_cuda
        device_type = "cuda" if x.is_cuda else x.device.type
        with torch.autocast(device_type=device_type, enabled=use_autocast):
            logits = F.linear(x_flat, self.centroids)
            
            # B) The dynamic bias is added to the scores to enforce load balancing.
            logits = logits + self.bias.to(logits.dtype)
 
        # C) Sparsity in action: top-k selects the experts and softmax computes their weights.
        topk_logits, topk_idx = torch.topk(logits, self.top_k, dim=-1)
        gate = F.softmax(topk_logits, dim=-1, dtype=x.dtype)
 
        # 3) dispatch per expert (correct indexing)
        routed_out = torch.zeros_like(x_flat)
        for i in range(self.n_routed):
            # D) Efficiently identifies all tokens that should be routed to the current expert `i`.
            mask = (topk_idx == i)
            row_idx, which_k = mask.nonzero(as_tuple=True)
            if row_idx.numel() == 0:
                continue
            
            exp_in = x_flat.index_select(0, row_idx)
            out = self.routed[i](exp_in)
            w = gate[row_idx, which_k].unsqueeze(-1)
            
            # E) The weighted outputs from the expert are added back to their original token positions.
            routed_out.index_add_(0, row_idx, out * w)
 
        routed_out = routed_out.view(B, S, D)
        
        # F) The final output is the sum of the residual, shared, and routed paths.
        return x + shared_out.view(B, S, D) + routed_out
```

The logic here is a highly effective implementation of sparse dispatch. Instead of sending tokens one by one, we process all tokens destined for a single expert in a batch. The loop iterates through each expert `i`. Inside the loop, `(topk_idx == i)` creates a boolean mask to identify which tokens have selected expert `i`. The `nonzero()` function gives us the indices of these tokens (`row_idx`). We then select these tokens, pass them through the expert, weight their outputs using the gate values, and use `index_add_` to add the results back to the correct positions in the `routed_out` tensor.

Finally, the original input `x`, the output from the shared experts `shared_out`, and the output from the routed experts `routed_out` are summed together to produce the final output of the MoE layer.

**Step 5: The Auxiliary-Loss-Free Bias Update**

We have now implemented a complete DeepSeekMoE forward pass. The final component is the dynamic adjustment mechanism that ensures our experts remain balanced over time. This is handled by the `update_bias` method, which is called once per training step.

This function operates `@torch.no_grad()`, meaning its operations do not contribute to the model's gradients. Its sole purpose is to calculate the current expert load, determine which experts are over- or under-utilized, and apply a small adjustment to the `self.bias` buffer. This adjustment will then influence the routing decisions in the next forward pass, creating the self-correcting feedback loop we discussed in the theory.

##### Listing 4.5 The `update_bias` Method for Load Balancing
```python
    @torch.no_grad()
    def update_bias(self, x: torch.Tensor):
        # Call once per optimizer step on the same tokens seen by forward.
        # Uses the SAME logits (including current bias) to estimate loads.
        
        N = x.shape[0] * x.shape[1]
        
        # A) Recalculates the logits using the current bias to accurately measure the load.
        logits = F.linear(x.reshape(-1,
            self.d_model), self.centroids) + self.bias
        
        _, idx = torch.topk(logits, self.top_k, dim=-1)
 
        # B) Efficiently counts how many tokens were routed to each expert using bincount.
        counts = torch.bincount(idx.flatten(),
            minlength=self.n_routed).float()
        
        avg = counts.sum() / max(1, self.n_routed)
 
        # Smooth, bounded update; avoids large jumps from sign()
        # C) Calculates the load violation; a positive value indicates an under-loaded expert.
        violation = (avg - counts) / (avg + 1e-6)
 
        # D) Updates the bias term to influence the next training step's routing decisions.
        self.bias.add_(self.bias_lr * torch.tanh(violation))
```

The logic is a direct implementation of the theory:

1.  It recalculates the router logits for the batch, including the current bias.
2.  It performs a `topk` selection to find the chosen experts for each token.
3.  `torch.bincount` efficiently counts how many tokens were routed to each expert.
4.  It calculates the "load violation" for each expert—a positive value for under-loaded experts and a negative one for over-loaded experts.
5.  Finally, it updates `self.bias` by adding a small, scaled value of this violation. The `torch.tanh` function is used to smooth the update and prevent extreme jumps, ensuring a stable adjustment process.

This concludes our from-scratch implementation of the DeepSeekMoE layer. By breaking the code into these four distinct parts, we have seen how each theoretical concept—shared experts, sparse routing, and dynamic balancing—is translated into a functional and efficient PyTorch module.