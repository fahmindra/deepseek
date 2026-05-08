---
slug: ch04-implementing-a-causal-multi-token-prediction-module-from
title: "5.4 Implementing a causal multi-token prediction module from scratch"
layout: chapter
---

We have now explored the theory behind Multi-Token Prediction, from its core advantages to the specifics of DeepSeek’s advanced causal architecture. While a full, large-scale implementation is deeply integrated into a complex training framework, we can solidify our understanding by building a functional, self-contained version of this mechanism in PyTorch.

This hands-on approach will translate the diagrams and equations we've just studied into concrete code. We will build the entire MTP system in stages:

1.  **The MTP Module**: The core component that processes one step in the causal chain.
2.  **The Main Model**: A wrapper class that integrates the main Transformer trunk and the chain of MTP modules.
3.  **The Forward Pass & Loss**: The complete logic for sequential prediction and the combined loss calculation.

Let's start with the most important new component: the MTP module itself. This class is a direct implementation of the logic shown in Figures 5.9 through 5.13. It takes the hidden state from the previous step and the embedding of the next token, and produces a refined hidden state and a prediction.

We begin by defining two key components. First, we'll implement RMSNorm, the specific normalization layer used throughout the DeepSeek architecture. This is a foundational utility that ensures numerical stability during training. You can find all imports and helper classes in the chapter's official GitHub repository.

##### Listing 5.1 Implementing a Multi-Token Prediction Module from Scratch

```python
class RMSNorm(nn.Module):
    """
    Implements Root Mean Square Layer Normalization.
    """
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calculate the inverse square root of the mean of squares
        norm_x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        # Apply the learnable weight
        return self.weight * norm_x
```

Next, we will build the core of the MTP architecture: the DeepSeekMTPModule. It contains a single, dedicated Transformer block and the necessary projection layers. Its purpose is to take the hidden state from the previous step and the embedding of the next token, and produce a refined hidden state for the next step in the causal chain.

##### Listing 5.2 The Causal MTP Module

```python
class DeepSeekMTPModule(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
 
        self.projection_matrix = nn.Linear(2 * d_model, d_model, bias=False)  # A
 
        self.transformer_block = nn.TransformerEncoderLayer(  # B
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
 
        self.norm_hidden = RMSNorm(d_model) # C
        self.norm_embed = RMSNorm(d_model)
 
    def forward(self, h_prev: torch.Tensor, future_token_embeds: torch.Tensor) -> torch.Tensor:
        h_normed = self.norm_hidden(h_prev)
        embed_normed = self.norm_embed(future_token_embeds)
 
        concatenated = torch.cat([h_normed, embed_normed], dim=-1) # D
        h_prime = self.projection_matrix(concatenated)
 
        h_output = self.transformer_block(h_prime) # E
 
        return h_output
```
*   **A**: The projection matrix M_k, which maps the concatenated 2D vector back to the model's dimension D.
*   **B**: A standard, dedicated Transformer block (TRM_k) for this MTP depth.
*   **C**: Separate RMSNorm layers for the previous hidden state and the future token embedding, as specified in the official formula.
*   **D**: The two normalized inputs are concatenated along the feature dimension.
*   **E**: The projected vector is processed by the Transformer block to produce the new, refined hidden state.

Now that we have the DeepSeekMTPModule building block, we can assemble the full model. The following listing shows the initialization of the DeepSeekV3WithMTP class.

Pay close attention to how the different components are organized. The model contains not just one, but a list of modules_m, one for each prediction depth. It also defines the shared embedding and output layers that will be used by both the main Transformer trunk and all the MTP modules, which is a key aspect of the architecture's efficiency.

##### Listing 5.3 Initializing the Full MTP Model Architecture

```python
class DeepSeekV3WithMTP(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        nhead: int,
        num_mtp_heads: int,     # D (number of MTP depths) # A
        dim_feedforward: int,
        dropout: float = 0.0,
        mtp_loss_weight: float = 0.1
    ):
        super().__init__()
        # ... (store parameters) ...
 
        # Shared components used across the model
        self.shared_embed = nn.Embedding(vocab_size, d_model) # B
        self.shared_lm_head = nn.Linear(d_model, vocab_size, bias=False) # C
 
        # Main transformer backbone (Shared Transformer Trunk)
        self.blocks = nn.ModuleList([ # D
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                dropout=dropout, activation='gelu', batch_first=True,
                norm_first=True
            )
            for _ in range(num_layers)
        ])
        self.norm_f = RMSNorm(d_model)
 
        # Weight tying between embedding and output head
        self.shared_lm_head.weight = self.shared_embed.weight
 
        # The chain of MTP modules
        self.mtp_modules = nn.ModuleList([ # E
            DeepSeekMTPModule(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_mtp_heads)
        ])
    # ... (forward method comes next) ...
```
*   **A**: The parameter num_mtp_heads corresponds to D, the number of future tokens to predict.
*   **B**: The single, shared token embedding layer.
*   **C**: The single, shared output head (un-embedding layer) that projects hidden states to logits.
*   **D**: The main stack of Transformer blocks, referred to as the "Shared Transformer Trunk."
*   **E**: A list of DeepSeekMTPModule instances, creating the sequential chain for MTP.

With the model structure initialized, we can now implement the forward method. This is where the entire sequential MTP process comes to life. The logic follows a clear sequence:

1.  The input tokens are first passed through the main Transformer trunk to produce the initial hidden state, main_h.
2.  The model then enters a loop that iterates through each MTP module in the chain.
3.  Inside the loop, for each prediction depth k, it uses the hidden state from the previous step (prev_h) and the ground-truth embedding of the k-th future token to generate the new hidden state, curr_h.
4.  The logits for the k-th future token are computed from curr_h.
5.  Finally, the total loss is calculated as the sum of the main next-token prediction loss and the weighted average of all the MTP losses.

##### Listing 5.4 The MTP Forward Pass and Combined Loss Calculation

```python
# ... (inside the DeepSeekV3WithMTP class) ...
 
    def forward(self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None):
        B, S = input_ids.shape
        # ... (other setup) ...
 
        # --- Main model forward pass ---
        x = self.get_embedding(input_ids)
        # ... (pass through self.blocks) ...
        h_main = self.norm_f(x)
        logits_main = self.get_output_logits(h_main)
 
        all_logits = [logits_main]
        h_prev = h_main # A
 
        # --- MTP chain: Sequential prediction ---
        for depth_k in range(1, self.num_mtp_heads + 1):
            L = S - depth_k # B
            if L <= 0: break
 
            h_prev_sliced = h_prev[:, :L, :] # C
 
            future_token_ids = input_ids[:, depth_k:depth_k + L]
            future_token_embeds = self.get_embedding(future_token_ids) # D
 
            h_curr = self.mtp_modules[depth_k - 1](h_prev_sliced, future_token_embeds) # E
            
            logits_k = self.get_output_logits(h_curr)
            all_logits.append(logits_k)
 
            h_prev = h_curr # F
 
        # --- Loss computation ---
        loss = None
        if targets is not None:
            # ... (loss calculation logic) ...
            total_loss = # ... Main model loss ...
 
            for k, logits_k in enumerate(all_logits[1:], start=1):
                # ... (calculate loss for MTP depth k) ...
                mtp_loss_sum += loss_mtp_k
            
            # Final loss: L = L_main + (λ/D) * Σ(L_MTP^k)
            if self.num_mtp_heads > 0 and mtp_loss_sum > 0:
                mtp_loss_weighted = (self.mtp_loss_weight / self.num_mtp_heads) * mtp_loss_sum # G
                total_loss += mtp_loss_weighted
            
            loss = total_loss
 
        return {"logits_all": all_logits, "loss": loss}
```
*   **A**: The output of the main trunk serves as the initial hidden state for the MTP chain.
*   **B**: The sequence length L shrinks at each depth k, as there are fewer future tokens to predict.
*   **C**: Slices the previous hidden state to match the current sequence length.
*   **D**: Gathers the ground-truth embeddings for the future tokens at the current depth k.
*   **E**: The core MTP step: the k-1 module is called to produce the new hidden state.
*   **F**: The causal link: the output hidden state becomes the input for the next iteration.
*   **G**: Implements the final loss formula, averaging the MTP losses and scaling by the weight λ.

##### Listing 5.5 Verifying the Causal MTP Implementation

```python
def verify_deepseek_v3_mtp():
    # --- Model configuration ---
    vocab_size = 1000
    d_model = 128
    num_layers = 6
    nhead = 8
    num_mtp_heads = 3  # D=3 (predict next 3 tokens) # A
    dim_feedforward = 512
    mtp_loss_weight = 0.1
 
    model = DeepSeekV3WithMTP(
        vocab_size=vocab_size, d_model=d_model, num_layers=num_layers,
        nhead=nhead, num_mtp_heads=num_mtp_heads,
        dim_feedforward=dim_feedforward, mtp_loss_weight=mtp_loss_weight
    )
    print(f"Model created with {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")
 
    # --- Test data ---
    batch_size = 2
    seq_len = 20
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len)) # B
 
    # --- Forward pass and output verification ---
    outputs = model(input_ids, targets=input_ids)
    all_logits = outputs['logits_all']
    loss = outputs['loss']
 
    print("\nLogits shapes:")
    for i, logits in enumerate(all_logits): # C
        pred_type = "Main" if i == 0 else f"MTP k={i}"
        print(f"  {pred_type:10}: {list(logits.shape)}")
 
    print(f"\nTotal loss: {loss.item():.4f}") # D
```
*   **A**: We set the number of MTP heads (prediction depth) to 3.
*   **B**: A dummy batch of input data with a sequence length of 20.
*   **C**: Iterates through the list of output logits to print the shape of each one.
*   **D**: Prints the final, combined loss value.

Running this script produces the following output. Notice how the sequence length (the middle dimension) of the logit tensors decreases by one at each step, confirming our causal chain is correctly implemented.

```
Model created with 2.01M params
 
Logits shapes:
  Main      : [2, 20, 1000]
  MTP k=1   : [2, 19, 1000]
  MTP k=2   : [2, 18, 1000]
  MTP k=3   : [2, 17, 1000]
 
Total loss: 109.1470
```

With this, we have successfully implemented the core logic of DeepSeek’s Multi-Token Prediction architecture. We've seen how a causal chain of dedicated Transformer blocks can be used to predict a horizon of future tokens, providing a richer training signal and a powerful mechanism for better planning and forecasting.