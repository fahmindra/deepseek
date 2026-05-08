---
slug: ch05-the-dark-side-of-the-kv-cache-the-memory-cost
title: "2.5 The dark side of the KV cache: The memory cost"
layout: chapter
---

Now we have seen the incredible speedup the KV Cache provides. By eliminating redundant computations, it makes interactive, long-sequence generation possible. However, this efficiency comes at a steep, non-negotiable price: memory.

This isn't just a matter of storage capacity; the inference process becomes memory-bandwidth bound. At every single generation step, the massive Key and Value matrices for all previous tokens must be read from the GPU's main memory (HBM) into its faster compute cores. This constant data movement becomes the new performance bottleneck, which is why modern GPUs designed for AI prioritize higher-capacity HBM and greater memory bandwidth, often more so than raw computational power (FLOPS).

Caching, by its very nature, is a trade-off. We are trading memory space to save computation time. While we avoid re-calculating the Key and Value matrices, we must now store them in the GPU's memory. For large models with long context windows, this memory footprint can become the new primary bottleneck.

### 2.5.1 The KV cache formula: Deconstructing the size

We can precisely calculate the memory required for the KV cache using a straightforward formula. Figure 2.21 breaks down each component of this calculation.

![Figure 2.21 The formula for calculating the KV cache size and its application to well-known models.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/02__image021.png)

Let's walk through the formula:

![Equation](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/02__image022.png)

*   **l (layers):** The total number of Transformer blocks in the model. We need a separate cache for each layer.
*   **u (batch size):** The number of sequences we process in parallel.
*   **n (heads):** The number of attention heads per layer.
*   **b (head size):** The dimension of each attention head's Key and Value vectors.
*   **z (sequence length):** The number of tokens in the context. This is a critical factor.
*   **First *2:** We need to cache two matrices: one for Keys and one for Values.
*   **Second *2:** This represents the number of bytes per parameter. A standard 16-bit floating-point number (like float16 or bfloat16) takes up 2 bytes of memory.

The formula makes the trade-off explicit. Every time we want to increase the model's context length (z), or use a larger model with more layers (l) and heads (n), the memory required for the KV cache grows proportionally.

The examples in the figure highlight the real-world impact:

*   The original GPT-2 (128M) model required a relatively modest 36 MB for its KV cache.
*   GPT-3, a much larger and more powerful model, requires a staggering 4.5 GB of memory for the same purpose—over 100 times more!

### 2.5.2 The scaling problem in practice

This exponential growth in memory consumption is a fundamental challenge in scaling Transformer models. As models grow larger and support longer context windows, the KV cache often becomes the primary limiting factor for deployment.

![Figure 2.22 A comparison of model parameter count versus KV cache size for different GPT-3 model variants. The dashed line shows that as models become larger, their memory requirements for the KV cache grow at a similar, steep rate.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/02__image023.png)

As figure 2.22 illustrates, there is a strong correlation between the size of a model and the size of its KV cache. This memory burden restricts the number of sequences we can process in a single batch and puts a hard cap on the maximum context length a model can support on a given piece of hardware.

Let's consider two modern examples based on our notes:

For a large 30B parameter model with 48 layers, 7168 total head dimensions (*n*), and a context length of 1024, the KV cache for a batch size of 128 would be a massive 180 GB. This exceeds the memory capacity of even the most powerful modern GPUs.

For a model with the architectural scale of DeepSeek-V3 (61 layers, 128 heads of size 128) and a huge context length of 100,000 tokens, the KV cache for a single sequence would be approximately 400 GB.

This is the ugly side of the KV cache. It speeds things up, but it takes up an enormous amount of space. This memory pressure is the direct reason why API providers like OpenAI charge significantly more for models with larger context windows; the hardware cost to support that memory is substantial.

This bottleneck is what motivated researchers to find a better way. How can we get the speed benefits of caching without paying such a high price in memory? This question leads us to the first generation of architectural solutions: Multi-Query and Grouped-Query Attention.