---
slug: ch06-the-payoff-an-empirical-head-to-head-comparison
title: "4.6 The payoff: An empirical head-to-head comparison"
layout: chapter
---

Having explored the theoretical underpinnings of both the traditional Mixture-of-Experts architecture and the advanced solutions pioneered by DeepSeek, it's time to put our knowledge to the test. Theory is one thing, but empirical results provide the final verdict. To that end, we conducted a head-to-head comparison by building and training two models from scratch: a baseline "Standard MoE" using a conventional load balancing loss, and our innovative "DeepSeek-MoE" implementing shared experts and auxiliary-loss-free dynamic balancing.

Our goal is to answer a simple question: Do these architectural innovations actually lead to a better and more efficient model? The complete code used to run this experiment and generate the following results is available for you to explore and replicate in the book's GitHub repository.

Bonus Code Link: [https://github.com/VizuaraAI/DeepSeek-From-Scratch/tree/main/ch04/02-bonus-code](https://github.com/VizuaraAI/DeepSeek-From-Scratch/tree/main/ch04/02-bonus-code)

Let's begin our analysis by looking at the most important metric: the validation loss. This tells us how well each model is generalizing to new, unseen data throughout the training process. The plot in Figure 4.28 tracks the validation loss for both models over 5,000 training iterations.

![Figure 4.28: A comparison of the validation loss curves for the Standard MoE and DeepSeek-MoE models. Despite having a similar number of parameters, the DeepSeek-MoE architecture consistently achieves a lower loss, indicating superior learning. Both models were trained for 5,000 iterations.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/04__image034.png)

As the learning curves illustrate, both models successfully learn to process the TinyStories dataset. However, a clear trend emerges. From the very early stages of training, the DeepSeek-MoE model consistently achieves a lower validation loss than its standard counterpart. While the difference isn't dramatic, a result of the relatively small model size and the less diverse nature of the training data, the advantage is consistent and widens slightly over time. This is our first piece of evidence that the DeepSeek architecture enables more effective learning.

While the loss curve gives us a great overview of learning performance, a detailed metrics table allows us to quantify the "total payoff" by looking at performance and computational efficiency side-by-side. Table 4.1 provides a comprehensive summary of the training run for both models, which were configured to have a nearly identical number of learnable parameters for a fair comparison.

##### Table 4.1 Comparison of training run

| Model | Parameters (M) | Training Time (min) | Throughput (iter/min) | Best Val Loss |
| :--- | :--- | :--- | :--- | :--- |
| **STANDARD\_MOE** | 101.30 | 14.29 | 350.0 | 1.9854 |
| **DEEPSEEK\_MOE** | 101.28 | 11.67 | 428.6 | 1.9451 |

Not only did the **DeepSeek-MoE** model achieve a lower final validation loss (1.9451 vs. 1.9854), but it was also significantly faster. It completed the training run in just 11.67 minutes, demonstrating a throughput of 428.6 iterations per minute, a 22% increase over the standard MoE's 350.0 iter/min. This proves that the DeepSeek architecture is not just more effective at learning, but it is also more computationally efficient.

The key difference between the two models lies in their approach to load balancing. The standard MoE uses an auxiliary loss to encourage balance over the entire training run, while the DeepSeek-MoE uses a dynamic bias to enforce it on a more immediate, per-batch basis.

Figure 4.29 visualizes the inherent challenge that the standard MoE faces. The bar chart shows how many tokens from a single validation batch were routed to each of the 22 experts in its first layer.

![Figure 4.29: Expert selection frequency for the Standard-MoE model in a sample batch. The uneven distribution highlights the problem of imbalanced routing.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/04__image035.png)

The chart clearly shows a significant load imbalance. Some experts, like #7 and #17, have become "hotspots," processing a disproportionately high number of tokens. Conversely, others, like #10 and #16, are underutilized. While the auxiliary loss aims to even this out over thousands of iterations, it struggles to prevent these imbalances from occurring within individual batches. This inefficiency means that some experts become computational bottlenecks while the knowledge capacity of others is wasted.

Now, Figure 4.30 shows the expert utilization for the same validation batch, but this time processed by our DeepSeek model.

![Figure 4.30: Expert selection frequency for the DeepSeek-MoE model in a sample batch. The distribution is remarkably uniform, demonstrating the effectiveness of the dynamic bias mechanism.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/04__image036.png)

The auxiliary-loss-free dynamic bias mechanism has done its job perfectly, resulting in a remarkably uniform load distribution. The self-correcting system, which penalizes overloaded experts and rewards underloaded ones in real-time, ensures that the workload is spread evenly across all available specialists. This is the tangible proof of the architecture's efficiency: no single expert becomes a bottleneck, and the full parallel processing power of the committee is leveraged effectively.

Our head-to-head experiment provides a clear verdict. The architectural innovations of the DeepSeek-MoE model, specifically its combination of powerful shared generalists and the elegant auxiliary-loss-free mechanism for balancing its specialists, are not merely theoretical. They deliver tangible benefits in both performance and efficiency. By decoupling the task of load balancing from the primary learning objective, the model learns better and faster. This is the core principle that allows DeepSeek to scale intelligence so efficiently, providing a powerful blueprint for the future of large-scale language models.

In the next chapter, we will dive into further performance and memory optimizations, exploring advanced techniques like Multi-Head Latent Attention (MLA) and FP8 quantization that push the boundaries of large language model efficiency.