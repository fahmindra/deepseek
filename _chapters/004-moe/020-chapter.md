---
slug: ch02-the-mechanics-of-moe-a-hands-on-mathematical-walkthrough
title: "4.2 The mechanics of MoE: A hands-on mathematical walkthrough"
layout: chapter
---

Now that we have the core intuition behind Mixture of Experts, it's time to open the black box and see how it is actually implemented. We will trace the journey of a batch of tokens step-by-step, from the input that enters the MoE block to the final output it produces.

To make the mathematics as clear as possible, we will switch from the previous conceptual example to a new, standard input sequence that is easier to visualize in matrix form. For the remainder of this walkthrough, we will use the input "The next day is." Our goal is to understand how the model uses sparsity and routing to efficiently combine the knowledge of multiple experts for this sequence.

### 4.2.1 The goal: Combining multiple expert outputs into one

Let's start by defining our setup. We have an **input matrix** of shape (4, 8), representing four tokens ("`The`", "`next`", "`day`", "`is`"), each with an 8-dimensional embedding. This matrix is the output from the preceding attention layer in the Transformer block.

Let's assume for our simplified example that our MoE layer contains three separate expert networks (E1, E2, E3), though in a real model this number would be much larger. As we know, each expert is a full Feed-Forward Network. If we were to pass our input matrix through each of these experts, we would get three separate output matrices.

![Figure 4.5: The initial challenge of MoE. The input matrix is passed through each of the three expert networks in parallel, resulting in three separate expert output matrices.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/04__image005.png)

As Figure 4.5 shows, this leaves us with a challenge. We start with one (4, 8) input matrix, but we end up with three (4, 8) output matrices. The Transformer architecture, however, expects only a single matrix to pass to the next layer.

Our entire task in this section is to figure out: **How do we intelligently combine these three expert outputs into a single, final output matrix of the same (4, 8) shape?** The answer lies in two key concepts we introduced intuitively back in **Section 4.1**: sparsity and routing.

### 4.2.2 Sparsity in action: Top-K selection for load balancing

The first step in processing tokens with a Mixture of Experts (MoE) model is selecting which experts to use for each token before merging their outputs. As we've discussed, the core efficiency of MoE comes from sparsity: not every token is processed by every expert.

We enforce this by making a simple but powerful decision: for each token, we will only select the top-$k$ most relevant experts. The value of $k$ is a hyperparameter we choose. For our example, we will set $k=2$. This means that out of our three available experts, only two will be activated for any given token.

![Figure 4.6: The principle of sparsity or load balancing. For each token, we decide to route it to only a subset (k=2) of the available experts.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/04__image006.png)

As illustrated in Figure 4.6, when the token "The" is processed, it will only be sent to two of the three experts (in this example, E1 and E2 are selected, while E3 is ignored). This act of selective activation is what prevents the model from having to perform the full computation of all experts for every token.

This immediately raises the next logical question: how does the model decide which two experts to select? And once selected, how does it know how much importance, or **weight**, to give to each of them? This is the job of the routing mechanism.

### 4.2.3 The routing mechanism: From input to expert scores

To decide which experts are best suited for each token, the model uses a small, learnable neural network called a **router**. The router's job is to look at the input tokens and produce a score for each expert, for each token.

This is implemented as a simple linear layer. We take our `Input Matrix` and multiply it by a learnable `Routing Matrix`.

![Figure 4.7: The routing mechanism. The input matrix is multiplied by a learned routing matrix to produce an expert selector matrix, which contains a raw score for each expert for each token.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/04__image007.png)

Let's break down the dimensions shown in Figure 4.7:

*   **Input Matrix**: Shape (4, 8) - four tokens, each with an 8-dimensional embedding.
*   **Routing Matrix**: Shape (8, 3) - the input dimension must match the input matrix (8), and the output dimension is the number of experts we have (3). This is a learned weight matrix.
*   **Expert Selector Matrix**: Shape (4, 3) - the result of the multiplication.

This Expert Selector Matrix (also often called the logits matrix) is the crucial output of the router. Let's interpret its structure:

*   Each **row** corresponds to one of our input tokens (`"The", "next", "day", "is"`).
*   Each **column** corresponds to one of our experts (`E1, E2, E3`).
*   The value at each position (`row, column`) is a raw, unnormalized score indicating how suitable that expert is for that token.

For example, the value in the first row, second column is the score for routing the token "`The`" to `Expert 2`. Now that we have these scores, we can use them to implement our top-$k$ selection.

### 4.2.4 From scores to weights: Top-K selection and softmax normalization

We now have our `Expert Selector Matrix`, which contains the raw scores. Our next task is to use these scores to achieve two goals:

1.  Select only the top 2 experts for each token.
2.  Convert the scores for those selected experts into a set of weights that sum to 1.

This is a three-step process.

**Step A: Select the Top-K Experts**

First, for each row (each token), we identify the $k=2$ experts with the highest scores. The scores for all other experts are discarded.

![Figure 4.8: The top-k selection process. For each row, only the two highest scores are kept, and the rest are masked out.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/04__image008.png)

As shown in Figure 4.8, for the first token, the scores `5` and `4` (for E2 and E3) are the highest, so the score 1 (for E1) is masked out. This is repeated for every token, fulfilling our sparsity requirement.

**Step B: Masking with Negative Infinity**

To mathematically eliminate the non-selected experts, we replace their scores with negative infinity. This might seem like a strange choice, but it's a clever trick that works perfectly with the softmax function in the next step.

![Figure 4.9: The masked scores are replaced with negative infinity in preparation for the softmax function.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/04__image009.png)

**Step C: Applying Softmax**

Finally, we apply the **softmax function** to each row of this new matrix. The softmax function has two properties that are perfect for our needs:

1.  It converts any real numbers into a probability distribution where all values are between 0 and 1, and the sum of the values in each row is exactly 1.
2.  The exponential function $e^x$ for a very large negative number (like negative infinity) is effectively zero.

As Figure 4.10 shows, applying softmax to our matrix does two things:

*   The scores for the non-selected experts (which were negative infinity) become **zero**.
*   The scores for the top-2 selected experts are converted into probabilities that **map to 1.**

![Figure 4.10: The softmax function converts the scores into a final expert selector weight matrix.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/04__image010.png)

The final output is our **Expert Selector Weight Matrix**. This matrix now contains all the information we need to combine the expert outputs. For the first token, "The," it tells us: "Ignore Expert 1, and give 70% of the final output's weight to Expert 2 and 30% to Expert 4." We have now answered both of our original questions: which experts to use, and what weight to give them.

### 4.2.5 The final output: Creating the weighted sum of expert outputs

For each token, we will take the outputs from its selected experts, multiply them by their assigned weights, and add them together. Let's walk through this for our first token, "`The.`"

![Figure 4.11: Calculating the final output for a single token. The weights from the selector matrix are used to create a weighted sum of the corresponding expert outputs.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/04__image011.png)

As illustrated in Figure 4.11, the calculation for the token "`The`" proceeds as follows:

1.  **Look up the Weights**: We look at the first row of our `Expert Selector Weight Matrix`. It tells us to use Expert 2 with a weight of `0.6` and Expert 3 with a weight of `0.4.` Expert 1 has a weight of `0` and will be ignored.
2.  **Look up the Expert Outputs:** We look at the first row (corresponding to the token "The") of the `Expert Output 2` and `Expert Output 3` matrices.
3.  **Perform the Weighted Sum:** We perform the following calculation:

```
(0.6 * OutputVector_The_from_E2) + (0.4 * OutputVector_The_from_E3)
```

The result is a single (1, 8) vector, which is the final, context-aware output for the token "The."

This same process is performed in parallel for every token in our input sequence.

*   For "`next`" we would take `0.9 * Output_next_E1 + 0.1 * Output_next_E4.`
*   For "`day`" we would take `0.4 * Output_day_E2 + 0.6 * Output_day_E4.`
*   And so on.

When we stack all of these resulting vectors together, we get our final output matrix.

![Figure 4.12: The complete MoE process for a batch of tokens. The expert selector weight matrix guides the weighted summation of the expert outputs to produce a single, final output matrix of the same shape as the input.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/04__image012.png)

As shown in Figure 4.12, the final output is a single matrix with the same shape as our original input matrix (4, 8). We have successfully replaced the single dense FFN with a sparse mixture of experts. By using the Expert Selector Weight Matrix to perform a weighted sum of only the top-$k$ expert outputs for each token, our new mechanism avoids the expensive computation for all non-selected experts, making it far more computationally efficient than a dense FFN.