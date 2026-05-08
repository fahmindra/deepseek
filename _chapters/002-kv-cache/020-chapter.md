---
slug: ch02-the-core-task-predicting-the-next-token
title: "2.2 The core task: Predicting the next token"
layout: chapter
---

Now we know that the LLM performs a full computational pass for each new token. Let's peel back the layers of the architecture to understand what that computation entails. Let's focus on the heart of the Transformer block: the Multi-Head Attention mechanism. This is where the model figures out the relationships between tokens.

The diagram below provides a high-level map of this journey. It shows how our example, "The next day is bright," flows through the key components we will deconstruct in the coming sections. Keep this image in mind as it represents the entire process we are about to build.

![Figure 2.3 A high-level overview of the Transformer block architecture. This diagram illustrates the complete data flow from the initial input tokens ("The next day is bright") through embedding, multi-head attention, and feed-forward layers, ultimately resulting in the logits used for next-token prediction.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/02__image003.png)

### 2.2.1 From input embeddings to context vectors: A mathematical walkthrough

The diagram in Figure 2.3 showed us the major components, but to understand the computations we might be repeating during inference, we need to zoom in on the most critical component: the Multi-Head Attention block. This is where the model calculates the relationships between tokens and creates the enriched "context vectors" that form the basis of its understanding.

Let's trace the path of our input sequence, "The next day is," through a single attention calculation to see exactly what's happening under the hood.

**Step 1: Projecting Inputs into Query, Key, and Value**

After tokenization and embedding, our input is represented as a matrix, which we'll call X. For this walkthrough, we will use small, simplified dimensions to make the mathematics and diagrams easy to follow. Let's assume our matrix X has a shape of (4, 8), representing our four tokens, each with an 8-dimensional embedding. In a real model, this embedding dimension would be much larger, for example, 5120 in DeepSeek-V2 and 7168 in DeepSeek-V3, but the underlying mathematical principles remain identical.

The very first step within the attention block is to project this input matrix into three distinct representations: the Query (Q), Key (K), and Value (V) matrices. This is done by multiplying X with three separate, learnable weight matrices: Wq (for Query), Wk (for Key), and Wv (for Value).

![Figure 2.4 The input embedding matrix X is projected into three new matrices: Query, Key, and Value. Each projection is a matrix multiplication with a unique, learned weight matrix.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/02__image004.png)

As the diagram illustrates:

*   The input X (shape 4x8) is multiplied by Wq (shape 8x4) to produce the Query matrix (shape 4x4).
*   The input X (shape 4x8) is multiplied by Wk (shape 8x4) to produce the Key matrix (shape 4x4).
*   The input X (shape 4x8) is multiplied by Wv (shape 8x4) to produce the Value matrix (shape 4x4).

These three new matrices represent our input tokens in different roles. The Query matrix represents what each token is "looking for," while the Key and Value matrices represent what each token "offers" as context.

**Step 2: Calculating Attention Scores**

Next, the model needs to determine how relevant each token is to every other token. This is done by calculating attention scores. We take the Query matrix and perform a matrix multiplication with the transpose of the Key matrix (Q * K.T).

![Figure 2.5 The dot product between the Query and transposed Key matrices produces the Attention Scores matrix. Each element in this matrix represents the relevance of one token to another.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/02__image005.png)

The resulting Attention Scores matrix (shape 4x4) quantifies the relationship between every pair of tokens. For example, the value in the fourth row and second column would represent how much attention the token "is" should pay to the token "next."

**Step 3: From Scores to Context Vectors**

These raw scores are then processed further. They are scaled (to stabilize training), and a causal mask is applied to ensure tokens can only gather context from previous tokens in the sequence, preventing it from "cheating" by looking ahead at tokens it is not supposed to know about yet. This mask effectively zeroes out the upper triangle of the attention scores matrix. Finally, a softmax function is applied to convert the scores into attention weights, a set of probabilities that sum to 1 for each row.

These final attention weights are then multiplied by the Value matrix.

![Figure 2.6 The Attention Weights are multiplied by the Value matrix to produce the final Context Matrix. Each row is now a context-aware representation of the original token.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/02__image006.png)

This final multiplication produces the Context Matrix (shape 4x4). Each row in this matrix is a new, enriched vector for each of our input tokens. The context vector for "is," for example, is now a weighted sum of all the Value vectors that came before it, containing rich information about the entire preceding sequence.

**Step 4: Scaling to Multi-Head Attention**

The process we've just described above is for a single attention calculation. However, a model might need to track different kinds of relationships simultaneously; for example, syntactic dependencies (like subject-verb agreement) and semantic relationships (like word meanings). A single attention calculation might struggle to capture this diversity.

This is where multi-head attention comes in. Instead of one large set of projection matrices (Wq, Wk, Wv), the model uses multiple, smaller, independent sets - one for each "head."

![Figure 2.7 Parallel Projections in Multi-Head Attention. The input embedding X is projected into separate Query, Key, and Value matrices for each attention head.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/02__image007.png)

As shown in figure 2.7, if our model has two heads, the initial (4, 8) input embedding is not projected into three (4, 4) matrices. Instead, it is projected in parallel into six smaller (4, 2) matrices: Q1, K1, V1 for Head 1, and Q2, K2, V2 for Head 2.

Next, each head computes its own attention scores independently and in parallel. Head 1 calculates the relevance between its queries (Q1) and keys (K1), while Head 2 does the same with Q2 and K2.

![Figure 2.8 Each attention head computes its own unique attention scores matrix in parallel.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/02__image008.png)

This ability to analyze the input from multiple perspectives simultaneously is the core of multi-head attention's power. By having its own unique projection matrices, each head learns to view the input in a different representational subspace. This allows for specialization: Head 1 might learn to focus on grammatical structure, while Head 2 might focus on semantic meaning, all from the same input sequence.

After each head has independently calculated its raw attention scores, these scores represent a measure of raw, unnormalized relevance. For Head 1, its (4, 4) Attention Scores matrix tells it how strongly each of its queries connects with each of its keys. However, these raw scores are not yet in a usable format for creating a weighted average. To make them useful, each head processes its score matrix through a series of transformations, as shown in figure 2.9.

![Figure 2.9 The attention scores for each head are independently processed to create final attention weights.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/02__image009.png)

Just like in the single-head case, each head's raw attention scores are then scaled, masked, and converted into a probability distribution of attention weights via a softmax function.

With the final, normalized attention weights for each head, the model can now create its enriched output. This is achieved by multiplying each head's Attention Weights matrix with its corresponding Value matrix.

![Figure 2.10 Each head produces its own context matrix, representing its unique, context-aware view of the input sequence.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/02__image010.png)

At the end of this step, we have two separate context matrices: Head 1 Context Matrix and Head 2 Context Matrix, both of shape (4, 2). Each matrix is a different contextualized representation of the original input. The final step in the multi-head attention block is to unify these parallel streams of information back into a single representation that the next layer can use. This is done in two stages.

First, the individual context matrices from all the heads are concatenated together along their last dimension (column-wise).

![Figure 2.11 The context matrices from all heads are concatenated to form a single, richer matrix, which is then passed through a final projection layer.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/02__image011.png)

As shown in figure 2.11, concatenating the Head 1 Context Matrix (shape 4, 2) and the Head 2 Context Matrix (shape 4, 2) results in a combined matrix of shape (4, 4). This new matrix now contains the insights from both heads.

Second, this concatenated matrix is passed through one final linear layer, often called the output projection layer. This layer has its own learnable weights and is responsible for mixing the information from the different heads and projecting it back into the main model's expected dimension, producing the final Context Matrix (shape 4, 4) that exits the multi-head attention block.

This parallel, multi-faceted, and finally unified approach is what gives the Transformer architecture its expressive power. The Context Matrix it produces is what gets passed to the subsequent layers to eventually produce the logits for our next-token prediction.

### 2.2.2 From context vectors to logits

We saw how the attention mechanism processes our input embeddings and produces an enriched Context Matrix. For our input "The next day is," this is a (4, 4) matrix where each row is a new, context-aware vector for each token.

It is the culmination of the model's effort to understand the relationships between words in the sequence. Now, this matrix gets passed to the subsequent layers in the Transformer block to eventually produce the logits for our next-token prediction.

**Step 1: The Feed-Forward Network**

The Context Matrix first passes through a Feed-Forward Network (FFN) within the Transformer block. Unlike the attention mechanism, which looks across all tokens, the FFN processes each token's context vector independently. It typically consists of two linear layers with a non-linear activation function in between. This step allows the model to perform more complex calculations on each token's contextualized representation. Crucially, the FFN is designed to output a matrix of the exact same shape as its input, preserving the dimensions of our Context Matrix.

**Step 2: Iterating Through the Transformer Blocks**

The output from the FFN doesn't immediately exit the model. It first goes through the final components of the current Transformer block, which include another Layer Normalization and a residual connection that adds the block's input to its output. This entire process—attention, feed-forward network, normalization, and residual connections—constitutes one full Transformer block. The resulting matrix is then fed as the input to the next Transformer block. This cycle repeats for every block in the model's architecture (e.g., 12 times for GPT-2 small).

**Step 3: The Final Projection to Logits**

After the sequence has been processed by the very last Transformer block in the stack, the resulting matrix of context vectors goes through one final Layer Normalization. This normalized matrix is then passed to the Final Output Layer, which is where the model makes its prediction.

The Output Layer performs the crucial step of projecting our context vectors into the vast space of the model's vocabulary. Let's define a key term: logits.

> **Definition**
>
> What are Logits? A logit is a raw, unnormalized score. For any given position in a sequence, the model produces a logit for every single word in its vocabulary. A higher logit score for a particular word indicates a higher likelihood that that word is the correct next token.

The Output Layer is a simple linear layer whose job is to take the final Context Matrix and transform it into the Logits Matrix.

![Figure 2.12 The journey from the final Context Matrix to the Logits Matrix. The Output Layer projects each context-aware vector into a long vector of scores, one for each word in the vocabulary.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/02__image012.png)

As shown in figure 2.12, the entire Context Matrix is processed. Each of its rows is transformed into a very long vector of logits:

*   **Input:** The final Context Matrix from the last Transformer block, with shape (4, 4).
*   **Transformation:** The Output Layer projects each of the 4 rows into a vector of size 50,257 (the vocabulary size of GPT-2).
*   **Output:** The final Logits Matrix with shape (4, 50257).

Each row in this massive matrix represents a complete prediction. The first row contains the model's scores for what token should follow "The," the second row for what token should follow "next," and so on.

Now that we have this matrix of raw scores, how does the model make a final, single-token decision? This next step contains the most important insight for optimizing the entire inference process.

### 2.2.3 The key insight: Why only the last row matters

We now have our Logits Matrix, a massive tensor of shape (4, 50257) containing predictions for every position in our input sequence. However, our goal is very specific: we only want to predict the single token that comes after our complete input, "The next day is."

**This means we can discard almost all of the information in the Logits Matrix.**

*   The first row (predictions for what comes after "The") is irrelevant.
*   The second row (predictions for what comes after "next") is irrelevant.
*   The third row (predictions for what comes after "day") is irrelevant.

The only row we care about is the last one, the logits vector corresponding to the token "is." This single vector holds the key to our next token. This is the crucial insight: since we only ever use the last row to make our prediction, re-computing the logits for all the earlier rows at every single step is incredibly wasteful. This observation is the fundamental motivation for optimizations like the KV cache and its derivatives, MQA and GQA.

![Figure 2.13 The final prediction step. The logits vector for the last token is converted into a probability distribution, and the token with the highest probability is selected as the output.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/02__image013.png)

To select the next token for our input we:

*   **Isolate the Final Logits Vector:** We select only the last row from the Logits Matrix. This gives us a vector of shape (1, 50257) containing the model's unnormalized confidence scores for every possible word in its vocabulary.
*   **Apply the Softmax Function:** These raw logits are converted into probabilities using the softmax function. This function transforms the entire vector into a probability distribution, where each value is between 0 and 1, and all values sum to 1. The output is a vector of probabilities, as shown in the figure with values like 0.002, 0.006, etc.
*   **Select the Most Likely Token:** We now simply find the index of the highest probability in this distribution (an argmax operation). This index corresponds to a specific token in the model's vocabulary. As the diagram shows, if the highest probability points to the token "bright," then that becomes the model's final, generated output for this step.

This entire, multi-step path from raw text to a single predicted token is performed for every token the model generates. And this gives us a key insight: after all the complex, context-building work, the final prediction depends only on the context vector of the last token. It's crucial to remember that this last context vector is significant because, thanks to the self-attention mechanism, it already contains a weighted summary of information from all previous tokens in the sequence.

This should make you suspicious. If we are constantly feeding a growing sequence back into the model, are we doing a lot of unnecessary, repeated work in the attention block itself? As we'll prove mathematically in the next section, the answer is a resounding yes.