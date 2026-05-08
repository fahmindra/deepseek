---
slug: ch03-the-deepseek-mtp-architecture-a-visual-and-mathematical
title: "5.3 The DeepSeek MTP architecture: A visual and mathematical walkthrough"
layout: chapter
---

While the original MTP paper from Meta proved the concept's effectiveness, it did so by predicting multiple future tokens using independent output heads. This means the prediction for the second future token was made without any information from the prediction for the first.

DeepSeek recognized a key opportunity for improvement. Their implementation is designed to sequentially predict additional tokens and keep the complete causal chain at each prediction depth. This means the prediction for future token t + 2 is informed by the prediction for token t + 1, creating a more coherent and powerful forecasting mechanism. Let's break down their architecture step-by-step, starting with the initial input to the entire MTP process.

### 5.3.1 The starting point: The shared transformer trunk

The Multi-Token Prediction process does not start from the raw input tokens. Instead, it begins *after* the main body of the Transformer has already processed the input sequence.

An input sequence (e.g., "Artificial Intelligence is") is first passed through what the original MTP paper calls the "Shared Transformer Trunk." This is simply the standard stack of Transformer blocks that we are already familiar with (e.g., 61 blocks in DeepSeek-V3).

**Figure 5.7 The initial step of the Multi-Token Prediction process. Input tokens are passed through the main "Shared Transformer Trunk," which consists of multiple Transformer blocks. The output is the initial matrix of hidden states (denoted as h⁰ or z), which serves as the starting point for the MTP modules.**
![Figure 5.7 The initial step of the Multi-Token Prediction process. Input tokens are passed through the main "Shared Transformer Trunk," which consists of multiple Transformer blocks. The output is the initial matrix of hidden states (denoted as h⁰ or z), which serves as the starting point for the MTP modules.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/05__image007.png)

As shown in figure 5.7, the output of this main trunk is a matrix of hidden states. Let's define this term precisely:

> **What is a Hidden State?**
>
> A hidden state is another name for the **context vector** that is output by a Transformer block. It is a rich, contextualized representation of an input token after it has been processed and has gathered information from its neighbors via the attention mechanism.

The output of the final Transformer block in the shared trunk is a matrix of these hidden states, one for each input token. For the purposes of MTP, we will call this initial matrix Hidden State 0, or h⁰, following the notation in the DeepSeek paper.

This h⁰ matrix is the starting point for the entire MTP process. It can be thought of as a stack of hidden state vectors, one for each token in the input sequence. For each token, its corresponding h⁰ vector will be fed into a chain of MTP modules to predict its future.

### 5.3.2 The MTP modules: A sequential chain of prediction

Instead of a single output head that predicts one token, the DeepSeek architecture uses a series of MTP Modules, one for each future token we want to predict. If we want to predict 3 future tokens (a prediction depth of D=3), we will have 3 MTP modules chained together.

The key innovation here is not just that these modules are dependent, but how that dependency is structured. Unlike approaches with independent prediction heads, DeepSeek's modules form a causal chain. The refined hidden state from one module becomes the input for the next, allowing the model to sequentially refine its predictions at each future step. This architecture of cascaded latent refinement is what makes the forecasting mechanism so powerful and coherent.

**Figure 5.8 The sequential architecture of the DeepSeek MTP modules. The hidden state from one module is passed as an input to the next, forming a causal chain.**
![Figure 5.8 The sequential architecture of the DeepSeek MTP modules. The hidden state from one module is passed as an input to the next, forming a causal chain.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/05__image008.png)

This diagram is the key to understanding DeepSeek's innovation. Let's trace the journey of a single input token's h⁰ vector as it enters this chain. We will focus on the operations inside a single MTP module, for example, the first one, which we'll call Head 1 (or k=1 for the prediction depth).

Each MTP head is a sophisticated piece of machinery designed to perform two jobs:

1.  Predict one future token.
2.  Generate a new, refined hidden state to pass to the next head in the chain.

Let's look inside a single head to see how it achieves this.

**Figure 5.9 The internal operations of a single MTP head.**
![Figure 5.9 The internal operations of a single MTP head.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/05__image009.png)

The operations within each head (k) can be broken down into the following steps:

**Step 1: Gathering Inputs**

Each head k takes two distinct inputs:

*   The Hidden State (hᵏ⁻¹) from the *previous* head. For Head 1, this is the initial h⁰ from the main Transformer trunk.
*   The Input Embedding (Emb(tᵢ₊ₖ)) for the future token it is trying to predict. During training, this is the ground-truth embedding of that future token. For Head 1 (predicting token t+1), it uses the embedding of token t+1.

**Step 2: Merging and Projecting**

These two input vectors are first passed through a separate RMS Norm layer and then concatenated to form a Merged Embedding. This merged vector now contains both the contextual information from the previous step and the semantic information of the next token to be predicted.

This merged vector, which now has twice the normal dimension (2d), is then passed through a Linear Projection Layer (denoted by Mₖ in the paper) to project it back down to the model's standard dimension (d). The output of this step is the Transformer Input.

This entire process is captured by **Equation 21** from the DeepSeek V3 paper.

**Equation 5.1**
$$h_{i}'^{k} = M_k[RMSNorm(h_i^{k-1}) ; RMSNorm(Emb(t_{i+k}))]$$

This formula precisely describes the process: the new input [h'] for the Transformer block inside the MTP module is created by projecting (Mₖ) the concatenation ([;]) of the normalized previous hidden state and the normalized future token embedding.

**Step 3: The MTP Transformer Block**

The projected vector [h'] now serves as the input to a single, dedicated Transformer Block (TRMₖ) within the MTP module.

**Figure 5.10 The Transformer block within a single MTP module. This block takes the merged and projected vector (combining the previous hidden state and the next token's embedding) as input. It then performs a full Transformer computation to produce a new, refined hidden state for the next step in the causal chain.**
![Figure 5.10 The Transformer block within a single MTP module. This block takes the merged and projected vector (combining the previous hidden state and the next token's embedding) as input. It then performs a full Transformer computation to produce a new, refined hidden state for the next step in the causal chain.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/05__image010.png)

This is a crucial step. It's not a simple linear transformation; it's a full, deep computation involving multi-head attention and a feed-forward network. This allows the model to perform complex reasoning about the combination of the previous hidden state and the next token's embedding, effectively asking, "Given my understanding so far (hᵢᵏ⁻¹), and knowing that the next word is tᵢ₊ₖ, what is my new, updated understanding?"

The output of this Transformer block is the new, refined hidden state, hᵢᵏ. This process is described by Equation 22 from the paper.

**Figure 5.11 Equation 22 from the DeepSeek V3 paper.**
![Figure 5.11 Equation 22 from the DeepSeek V3 paper.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/05__image011.png)

**Step 4: Generating Outputs**

This new hidden state, hᵢᵏ, now serves two purposes, completing the loop:

1.  It is passed as an input to the next MTP Module. The h¹ from Head 1 becomes the input hidden state for Head 2. The h² from Head 2 becomes the input for Head 3, and so on. This is the causal link that makes DeepSeek's MTP implementation sequential and powerful. Crucially, passing the entire refined hidden state provides a rich, contextualized summary of the sequence so far—far more information than a single predicted token ID could. This is the key advantage of DeepSeek's causal MTP, as it allows each subsequent module to make its prediction based on a much deeper understanding of the evolving context.
2.  It is also passed to a Shared Un-Embedding Matrix. This is the same final output/logit layer used by the main model and all other MTP heads. This layer projects the hidden state into the full vocabulary space to produce the logits for the k-th future token.

**Figure 5.12 The final prediction steps within an MTP module. The new hidden state generated by the MTP Transformer block is passed to a "Shared Un-Embedding Matrix." This projects the hidden state into the vocabulary space to produce a logits vector, from which the final token for that prediction step is selected.**
![Figure 5.12 The final prediction steps within an MTP module. The new hidden state generated by the MTP Transformer block is passed to a "Shared Un-Embedding Matrix." This projects the hidden state into the vocabulary space to produce a logits vector, from which the final token for that prediction step is selected.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/05__image012.png)

This process is described by Equation 23 from the paper.

**Figure 5.13 Equation 23 from the DeepSeek V3 paper.**
![Figure 5.13 Equation 23 from the DeepSeek V3 paper.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/05__image013.png)

This formula states that the probability distribution P for the k-th future token is generated by passing the hidden state from the k-th MTP Module through the output head (LitHead).

### 5.3.3 The final loss calculation

This entire sequential process is performed for each of the D prediction depths. At the end, for a single input token tᵢ, we will have D different predicted tokens. During training, we compare these D predictions to the D actual ground-truth tokens from the input data.

**Figure 5.14 The total loss for a single input token is the sum of the individual cross-entropy losses for each of the predicted future tokens.**
![Figure 5.14 The total loss for a single input token is the sum of the individual cross-entropy losses for each of the predicted future tokens.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/05__image014.png)

As shown in figure 5.14, the total loss is simply the sum of the individual cross-entropy losses for each prediction depth. This means the model receives a rich, multi-faceted gradient signal, pushing it to become better at not just immediate next-token prediction, but also at longer-term forecasting.