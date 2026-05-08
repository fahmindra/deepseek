---
slug: ch03-the-problem-of-redundant-computations
title: "2.3 The problem of redundant computations"
layout: chapter
---

So far, we've established two critical facts about LLM inference:

1.  The model generates text one token at a time in an autoregressive loop, feeding its own output back as input.
2.  To predict the single next token, the model only requires the context vector of the last token in the current sequence.

Now, let's connect these two ideas. If the model is constantly re-processing a growing sequence of tokens, and it only needs the information from the very last one to make its next decision, it seems that a huge number of computations might be performed unnecessarily. Intuitively, it feels like we might be performing the same calculations again and again.

I'm going to show you that during inference, we are indeed repeating many computations. We will then see what we can do to avoid these repetitions, which will lead us directly to the concept of the KV Cache.

### 2.3.1 Intuition: Are we calculating the same thing over and over?

Let's move from intuition to concrete mathematical proof. We’ll first show the redundancy intuitively by tracing the data at each step, and then we will quantify its performance impact by analyzing the computational complexity. Let's revisit the autoregressive loop from figure 2.2, but this time, let's focus on the data being passed into the model at each step.

Imagine we start with the prompt "The next day."

**Inference Step 1:**

*   **Input:** "The next day"
*   **Process:** The three tokens go through the entire LLM pipeline.
*   **Output:** "is"

**Inference Step 2:**

The new token is appended.

*   **Input:** "The next day is"
*   **Process:** These four tokens go through the entire LLM pipeline.
*   **Output:** "bright"

**Inference Step 3:**

The new token is appended again.

*   **Input:** "The next day is bright"
*   **Process:** These five tokens go through the entire LLM pipeline.
*   **Output:** "and"

Notice the pattern. In Step 2, we are re-processing the tokens "The," "next," and "day." We already processed them in Step 1. In Step 3, we are re-processing "The," "next," "day," and "is," all of which have been processed in previous steps. It seems we are passing the same tokens through the entire architecture again and again, just to add one new token at the end.

This process feels inefficient. It's like re-reading the first nine chapters of a book every time you want to read a new chapter. If reading each chapter takes a fixed amount of time, then reading up to chapter n requires 1 + 2 + ... + n units of work—scaling quadratically, O(n²).

The main drawback of these repeated computations is their explosive cost: for every additional token, the GPU must re-process and store increasingly large amounts of data, making both time and memory requirements grow rapidly with sequence length.

This intuitive sense of repetition is, in fact, correct. Let’s take a hands-on example and prove mathematically that we are repeating the exact same calculations within the attention mechanism at every single step of inference.

### 2.3.2 A mathematical proof: Visualizing repeated calculations

Our intuition suggests we are performing redundant work. Now, let's prove it by walking through the attention mechanism for two consecutive inference steps. We will see with our own eyes that we are calculating the exact same values multiple times.

**Step A: Inference at Time T=4 (Input: "The next day is")**

First, let's consider the state of our model at the moment it has processed the input "The next day is" and is about to predict the next token. This is an input sequence with 4 tokens. As shown in figure 2.14, this sequence is about to be processed by a single attention head.

![Figure 2.14 The full attention calculation for the input sequence "The next day is."](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/02__image014.png)

Let's trace the data flow in figure 2.14 step-by-step:

1. **Input Embedding (X):** We start on the far left with our input embedding matrix X, which has a shape of (4, 8) for our four tokens.
2. **Projection:** This X matrix is multiplied by the fixed, pre-trained weight matrices Wq, Wk, and Wv. This projection creates the Query (Q), Key (K), and Value (V) matrices, which in this example all have the shape (4, 4).
3. **Attention Scores:** Next, the model calculates the raw relevance between tokens. The Query matrix is multiplied by the transpose of the Key matrix (Q * K.T), resulting in the (4, 4) Attention Scores matrix.
4. **Attention Weights:** This raw score matrix is then processed (scaled and passed through a causal softmax) to produce the final (4, 4) Attention Weights matrix. The grayed-out parts represent future positions that have been masked to zero.
5. **Context Matrix:** Finally, the Attention Weights are multiplied by the Value matrix to produce the (4, 4) Context Matrix.

This Context Matrix then travels through the rest of the Transformer architecture. As we established, only the last row of this matrix—the context vector for "is"—is used to generate the final logits and predict the next token. Let's say the model correctly predicts the token "bright."

Now, we move to the next step in the autoregressive loop, and this is where the inefficiency becomes glaringly obvious.

**Step B: Inference at Time T=5 (Input: "The next day is bright")**

Following the autoregressive loop, the newly predicted token, "bright," is appended to our sequence. The new input for the model is now "The next day is bright," a sequence of 5 tokens. This new, longer sequence is fed back into the exact same attention mechanism, with the exact same learned weight matrices (Wq, Wk, Wv).

![Figure 2.15 The full attention calculation for this new, 5-token input.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/02__image015.png)

At first glance, this looks like a completely new calculation. But let's compare it closely to the computation we just performed in figure 2.15.

The first four rows of our new input matrix X (shape 5, 8) are identical to the entire input matrix from the previous step. Since the weight matrices (Wq, Wk, Wv) are fixed during inference, this means the first four rows of our new Query, Key, and Value matrices (all now shape 5, 4) are identical to the entire Query, Key, and Value matrices from the previous step.

This redundancy cascades directly into the attention score calculation. The score at any position (j, i) is the dot product of the j-th query vector and the i-th key vector. Since the first four query vectors and the first four key vectors are identical to the previous step, the entire top-left (4, 4) sub-block of our new (5, 5) Attention Scores matrix is also identical to the entire score matrix we calculated just a moment ago.

We are performing a massive amount of redundant computation. We are re-calculating the projections and the attention scores for the entire history of the sequence at every single step. This is a huge waste of computational resources. And the most inefficient part? As we've established, after all this redundant work, the only piece of information we are actually going to use to predict the next token is the context vector derived from the new token, "bright."

We are re-calculating an entire history of interactions just to compute one new row, and then we throw away most of the old row's final outputs anyway. This is the root problem that inference optimization techniques must solve.

### 2.3.3 The performance impact: From quadratic to linear complexity

The redundant calculations we've identified are not just theoretically inefficient; they have a severe impact on performance, especially as the input sequence grows longer. This impact is best understood through the lens of computational complexity.

> **Note**
>
> It is important to clarify that this discussion is strictly about the inference stage.

Without any optimization, the core of the attention mechanism is quadratic in nature. For each layer and for each attention head, the number of calculations needed to compute the attention scores scales quadratically with the length of the input sequence (n), often denoted as O(n²). While the total number of computations is multiplied by constants like the number of layers (L) and heads (H), it's the quadratic relationship with the sequence length n that dominates performance.

Why quadratic? Think about the Attention Scores matrix.

*   For an input of 4 tokens, we compute a 4x4 matrix (16 scores).
*   For an input of 5 tokens, we compute a 5x5 matrix (25 scores).
*   For an input of 1,000 tokens, we would compute a 1,000 x 1,000 matrix (1,000,000 scores).

At each step of autoregressive generation, we are re-calculating this entire n x n matrix, performing O(n²) work repeatedly. As the sequence length n grows, the number of computations explodes. This quadratic complexity is the primary reason why unoptimized inference for long sequences is computationally expensive and extremely slow. Each new token becomes progressively slower to generate than the last because the model has to do more and more historical re-calculation.

![Figure 2.16 A plot comparing the quadratic (O(n²)) growth of computations in uncached autoregressive inference versus the ideal linear (O(n)) growth.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/02__image016.png)

The goal of inference optimization is to transform this quadratic process into a linear one (O(n)). In a linear complexity scenario, the amount of computation required to generate a new token grows linearly with the length of the sequence, not quadratically. This means that while generating a new token for a long sequence still requires more computation than for a short one, the increase is far more manageable. For instance, doubling the context length would roughly double the work for the next token, rather than quadrupling it, preventing the explosive growth of the unoptimized approach.

This is precisely what caching allows us to achieve. By storing the results of past computations instead of repeating them, we can break free from the quadratic trap. As we saw visually, the only new computations we need to perform for a new token are related to that token itself. The computations for all previous tokens can be retrieved from memory.

This shift from quadratic to linear complexity is the fundamental reason why caching is not just an optimization but a foundational requirement for making LLMs with large context windows practical. It explains the dramatic speedups we will demonstrate in code:

*   **Without Caching (Quadratic):** Generating the 100th token is much slower than generating the 10th token.
*   **With Caching (Linear):** Generating the 100th token takes roughly the same amount of time as generating the 10th token.

Having established the dire need for a solution, we are now ready to build it.