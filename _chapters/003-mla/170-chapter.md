---
slug: ch17-the-payoff-an-empirical-head-to-head-comparison
title: "3.17 The Payoff: An Empirical Head-to-Head Comparison"
layout: chapter
---

Having explored the theoretical and mathematical underpinnings of Multi-Head Latent Attention and its fusion with Decoupled RoPE, it's time to put our knowledge to the test. Theory is one thing, but empirical results provide the final verdict. The central promise of MLA is to break the painful trade-off between the high performance of MHA and the memory efficiency of its leaner cousins, MQA and GQA. Does it truly deliver the best of both worlds?

To answer this, we conducted a head-to-head comparison by building and training four models from scratch. Each model uses a different attention mechanism, but all other hyperparameters from model size and dataset to the number of training iterations are held constant to ensure a fair and direct comparison. This experiment will allow us to move beyond theory and quantify the real-world impact of DeepSeek's architectural innovations.

The code used to run this experiment and generate the following results is available for you to explore and replicate in the book's GitHub repository.

Bonus Code Link:
[https://github.com/VizuaraAI/DeepSeek-From-Scratch/tree/main/ch03/02-bonus-code](https://github.com/VizuaraAI/DeepSeek-From-Scratch/tree/main/ch03/02-bonus-code)

To properly evaluate the trade-offs, we will train and benchmark four distinct attention architectures:

1.  **Multi-Head Attention (MHA):** Our baseline for maximum performance. Each head has its own unique Query, Key, and Value projections, foregoing the highest efficiency at the cost of a large KV cache.
2.  **Multi-Query Attention (MQA):** The first efficiency-focused variant. All heads share a single Key and Value projection, drastically reducing the KV cache size but potentially limiting the model's learning capacity.
3.  **Grouped-Query Attention (GQA):** A compromise between MHA and MQA. Query heads are divided into groups, with each group sharing a common Key and Value projection. This offers a middle ground in both performance and efficiency.
4.  **Multi-Head Latent Attention (MLA):** The DeepSeek innovation. As we've seen, it uses a compressed latent cache and decoupled RoPE to aim for the performance of MHA with the memory footprint closer to MQA/GQA.

Before we begin the training process, it's crucial to confirm that our models are of a comparable size. A significant difference in the total number of learnable parameters would make for an unfair comparison. The table below shows the parameter counts for each of our four model variants, demonstrating that they are all within a very similar range.

##### Table 3.1 Parameters of model variants

| Attention Type | Total Parameters |
| :--- | :--- |
| Multi-Head Attention (MHA) | 17,653,248 |
| Grouped-Query Attention (GQA) | 16,965,120 |
| Multi-Query Attention (MQA) | 17,260,032 |
| Multi-Head-Latent Attention (MLA) | 17,391,104 |

The total number of learnable parameters is nearly identical across all four architectures, ensuring a fair comparison of their performance and efficiency. The minor differences come from the unique projection matrices required by each attention mechanism.

Our first objective is to answer the most critical question: does MLA's memory-saving design compromise its learning performance? A lower validation loss indicates better generalization to unseen data, which is our primary measure of a model's effectiveness. The plot in figure 3.37 tracks the validation loss for all four models over 5,000 training iterations.

##### Figure 3.37 A comparison of the validation loss curves for the four attention mechanisms. The MHA and MLA models achieve a consistently lower final loss than their MQA and GQA counterparts, indicating superior learning. Both models were trained for 5,000 iterations.
![Figure 3.37 A comparison of the validation loss curves for the four attention mechanisms. The MHA and MLA models achieve a consistently lower final loss than their MQA and GQA counterparts, indicating superior learning. Both models were trained for 5,000 iterations.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/03__image059.png)

The learning curves reveal an edge. While all four models successfully learn to process the TinyStories dataset, their performance diverges. The MHA and MLA curves track each other closely, achieving the lowest validation loss and demonstrating the strongest learning capability. MQA, as expected, lags slightly behind due to its reduced expressivity, with GQA occupying the middle ground. This plot provides our first piece of powerful evidence: the MLA architecture does not sacrifice performance. It learns just as effectively as the full, resource-intensive MHA model, successfully preserving the expressive power of having unique, multi-headed projections.

##### Table 3.2 Comparison of attention types

| Attention Type | Best Validation Loss | KV Cache Memory Size (MB) | Best Validation Perplexity |
| :--- | :--- | :--- | :--- |
| MHA | 2.3721 | 48.00 | 10.72 |
| MQA | 2.4080 | 6.00 | 11.11 |
| GQA | 2.3776 | 24.00 | 10.78 |
| MLA | 2.3089 | 12.00 | 10.06 |

The results in table 3.2 are definitive. While the validation perplexity confirms what we saw in the learning curves—that MLA achieves the best performance (10.06), closely followed by MHA (10.72)—the KV Cache size column reveals the other half of the story. MLA dramatically reduces the memory footprint by 4x compared to MHA (12.00 MB vs. 48.00 MB), validating the core motivation behind the latent cache architecture.

This head-to-head experiment provides a clear verdict. By decoupling the content and position pathways and leveraging a compressed latent cache, the DeepSeek architecture successfully breaks the performance-memory trade-off. The result is an attention mechanism that is not only powerful but also remarkably efficient, providing a solid foundation for the scalable and intelligent models we aim to build. We will turn to the other key component of the Transformer layer in chapter 4. We will explore how Mixture-of-Experts (MoE) replaces the standard Feed-Forward Network to unlock another dimension of efficient scaling and we will also see DeepSeek Innovations.