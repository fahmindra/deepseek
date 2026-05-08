---
slug: ch04-the-solution-caching-for-efficiency
title: "2.4 The solution: Caching for efficiency"
layout: chapter
---

The solution to this problem is both elegant and intuitive: if we are repeatedly calculating the same values, why not just calculate them once and store them for future use? This is the core principle of caching.

By storing the results of past computations, we can avoid re-doing the work for tokens we've already seen. This allows us to break free from the quadratic trap and achieve a much more efficient, linear computation time. This powerful optimization is known as the Key-Value Cache, or KV Cache.

### 2.4.1 What to cache? A step-by-step derivation

To understand what we need to cache, we must start with our end goal and work backward. As we established in section 2.2.3, our entire objective at each inference step is to produce the single context vector for the most recent token.

Let's take our example where the model has just processed "The next day is" and generated "bright." The new input sequence is now 5 tokens long. To predict the next token, we only need the context vector for "bright."

The values shown in the box are from previous steps. Since these values remain unchanged across steps, we cache them. The same principle applies to the subsequent images in this section.

![Figure 2.17 Out of the entire context matrix, only the final row, corresponding to the newest token, is required to predict the next token in the sequence.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/02__image017.png)

So, our immediate goal is to compute this one vector. Let's backtrack to figure out the minimal set of computations required to produce it.

**How is the Context Vector for "bright" Calculated?**

From our previous exploration of the attention mechanism, we know that the context vector is the result of multiplying the attention weights by the Value matrix. Since we only need the context vector for "bright," we only need to perform the calculation for that specific row.

![Figure 2.18 The context vector for "bright" is calculated by multiplying the attention weights for "bright" with the full Value matrix.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/02__image018.png)

As figure 2.18 shows, to get our target vector, we need two components:

*   **The Attention Weights for "bright":** This is a single row vector (shape 1x5) that tells us how much "bright" should attend to every token in the sequence (including itself).
*   **The full Value matrix:** This is a matrix of shape (5x4) that contains the "content" representation for every token in the sequence.

Let's continue backtracking. How do we get these two components?

**How are the Attention Weights for "bright" Calculated?**

The attention weights are simply the softmax-normalized attention scores. So, to get the weights for "bright," we first need the raw attention scores for "bright." These scores are calculated by taking the dot product of the Query vector with the transpose of the full Key matrix.

![Figure 2.19 The attention weights for "bright" are derived from its Query vector and the Key vectors of all tokens in the sequence.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/02__image019.png)

To get the attention weights, we fundamentally need:

*   The **Query vector for our new token.**
*   The **full Key matrix**, which contains the Key vectors for all five tokens in the sequence.

So, our list of requirements has now grown. To get the single context vector for "bright," we need:

*   The **Query vector for "bright" (q_bright).**
*   The full Key matrix (K).
*   The full Value matrix (V).

This finally brings us to the source: how are these Query, Key, and Value vectors created?

The Query, Key, and Value vectors are all created by projecting the input embeddings through the learned weight matrices Wq, Wk, and Wv. Here is where we can separate what is new from what is old.

When the new token "bright" arrives, we must compute its specific Query, Key, and Value vectors. This is done via three simple matrix multiplications using its input embedding, X_bright.

![Figure 2.20 The three essential computations for a new token. The input embedding for "bright" is projected to create its unique Query, Key, and Value vectors.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/02__image020.png)

As shown in figure 2.20, these are the only new projections required:

*   We compute the **Query vector** for “bright.”
*   We compute the **Key vector** for “bright.”
*   We compute the **Value vector** for “bright.”

Now we can finally answer the central question: what about the Key and Value vectors for all the previous tokens ("The", "next", "day", "is")?

In the inefficient, uncached approach we saw in section 2.3, we would have re-calculated them from scratch. But now we understand that this is wasteful. These previous Key and Value vectors have already been computed in prior inference steps. They don't change.

This is precisely where the concept of caching comes into play. Instead of re-computing them, we can simply store the Key and Value matrices from the previous step in memory. This stored data is the Key-Value (KV) Cache. This leads us to the final, efficient workflow and answers why we only cache Keys and Values, but not Queries:

1.  **Compute for the New Token:** When the token "bright" arrives, we perform the three essential multiplications to get its Query, Key, and Value vectors.
2.  **Assemble the Full Key & Value Matrices:**
    *   We retrieve the Key matrix for "The next day is" (a 4x4 matrix) from our KV Cache.
    *   We append the new Key vector for "bright" to it, creating the full 5x4 Key matrix.
    *   We do the exact same thing for the Value matrix, retrieving the old Value matrix from the Value Cache and appending the new Value vector.
3.  **Calculate Attention:** We use the new **Query vector** for "bright" and the full, updated **Key and Value matrices** to perform the attention calculation and get the single context vector we need.
4.  **Update the Cache:** We update our cache by storing the new, larger 5x4 Key and Value matrices, ready for the next token.

This is the essence of the Key-Value Cache. We only perform the expensive projection calculations for the single new token at each step. All historical information is preserved in the cache. We don't need to cache the Query vectors because, as we've established, we only ever use the query for the current token, which must always be computed fresh. This simple but powerful technique of caching the Keys and Values is what transforms the attention calculation from a cripplingly slow quadratic operation into an efficient linear one.

### 2.4.2 The new inference loop with KV caching

With our understanding of what to cache, we can now define a new, highly efficient workflow for autoregressive generation. At each step, instead of re-calculating the entire history, we leverage our stored Key and Value matrices.

Let's summarize the new process when generating a token:

1.  **Receive New Token:** The model receives the embedding for the single, new token.
2.  **Compute New Projections:** Perform the three essential matrix multiplications to get the Query, Key, and Value vectors for only this new token.
3.  **Retrieve from Cache:** Load the existing Key and Value matrices for all previous tokens from the KV Cache.
4.  **Append to Cache:** Append the new Key and Value vectors to the cached matrices to form the full, updated Key and Value matrices for the entire sequence.
5.  **Calculate Attention:** Multiply the new Query vector with the transpose of the full, updated Key matrix to produce the raw attention scores, which are then converted into attention weights.
6.  **Compute Context Vector:** Multiply the attention weights by the full, updated Value matrix to get the single context vector for the new token.
7.  **Predict Next Token:** Pass this context vector through the rest of the model's layers to predict the next token.
8.  **Update Cache:** Save the updated Key and Value matrices back into the KV Cache for the next iteration.

This loop avoids the massive redundancy of the naive approach. The heavy lifting of matrix multiplications for past tokens is done only once and their results are simply reused. It's important to remember that this caching happens independently at each level of the architecture: the KV cache is maintained on a per-layer and per-head basis. This means each Transformer layer in the model keeps its own distinct set of Key and Value caches for each of its attention heads, ensuring that the specialized context learned at each layer is preserved.

### 2.4.3 Demonstrating the speedup of KV caching

The theoretical benefit of moving from a quadratic to a linear process is clear, but the real-world impact is even more striking. We can demonstrate this with a simple test using the pre-trained GPT-2 model from Hugging Face, which allows us to enable or disable the KV cache with a simple flag (`use_cache`).

The following code will generate 100 new tokens from a prompt, first with the KV cache enabled, and then with it disabled, timing both processes.

##### Listing 2.2 Demonstrating the Speedup of KV Caching

```python
prompt = "The next day is bright"
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs.input_ids
attention_mask = inputs.attention_mask
 
# Timing without KV cache
start_time_without_cache = time.time()
# Generation is performed with the KV cache explicitly disabled. 
# This forces the model to re-calculate the Key and Value matrices for the entire sequence 
# for every single token it generates, which is computationally expensive.
output_without_cache = model.generate(input_ids,
    max_new_tokens=100,
    use_cache=False,
    attention_mask=attention_mask)
end_time_without_cache = time.time()
duration_without_cache = end_time_without_cache - start_time_without_cache
print(f"Time without KV Cache: {duration_without_cache:.4f} seconds")
 
# Timing with KV cache
start_time_with_cache = time.time()
# We enable the KV cache by setting use_cache=True. 
# Now, the model only computes the projections for the newest token and reuses the cached 
# Keys and Values for all previous tokens, resulting in a dramatic speedup.
output_with_cache = model.generate(input_ids,
    max_new_tokens=100,
    use_cache=True,
    attention_mask=attention_mask)
end_time_with_cache = time.time()
duration_with_cache = end_time_with_cache - start_time_with_cache
print(f"Time with KV Cache: {duration_with_cache:.4f} seconds")
 
# Calculate and print the speedup
speedup = duration_without_cache / duration_with_cache
print(f"\nKV Cache Speedup: {speedup:.2f}x")
```

Running this code on a standard machine reveals the efficiency of this technique.

```
Time without KV Cache: 30.9818 seconds
Time with KV Cache: 6.1630 seconds
 
KV Cache Speedup: 5.03x
```

The results are unambiguous. By simply enabling the KV cache, we achieve a more than 5x speedup in generating 100 tokens. For very large models and much longer sequences, this speedup factor can be even more dramatic, often reaching 6x or more. This is the incredible advantage of the KV cache: it makes real-time, interactive generation feasible by eliminating the crippling cost of repeated computation.

However, this incredible speed comes at a price. Caching isn't free. Storing these Key and Value matrices in memory introduces its own significant challenge, the "dark side" of the KV Cache.