---
slug: ch10-attempt-3-the-attention-is-all-you-need-breakthrough--
title: "3.10 Attempt #3: The "Attention Is All You Need" breakthrough - sinusoidal positional encodings"
layout: chapter
---

So far we have seen that integer encodings failed due to their large, unbounded values, and binary encodings, while solving the magnitude problem, introduced undesirable discontinuities. This led us to a key insight: we need a method that can represent positions using the multi-frequency oscillating patterns of binary encoding, but with smooth, continuous values.

This is precisely the problem that the authors of the seminal "Attention Is All You Need" paper solved with the introduction of **Sinusoidal Positional Encodings**. This elegant technique uses the sine and cosine functions to create smooth, continuous waves that are far better suited for training deep neural networks.

### 3.10.1 From discrete jumps to smooth waves: Introducing sine and cosine

The core idea of sinusoidal positional encodings is to replace the jumpy 0s and 1s of binary encoding with continuous values that flow on a spectrum from -1 to 1. This is achieved using a now-famous formula that, while looking intimidating at first, is built on the same principles we've already discovered.

##### Figure 3.17 Sinusoidal Positional Encoding. The binary values are replaced with continuous values derived from sine and cosine functions, which are then added to the token embedding.
![Figure 3.17 Sinusoidal Positional Encoding. The binary values are replaced with continuous values derived from sine and cosine functions, which are then added to the token embedding.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/03__image033.png)

The positional encoding $PE$ for any given token depends on two variables, just as before:

*   `pos`: The absolute position of the token in the sequence.
*   `i`: The index of the dimension within the embedding vector.

The formula is defined in two parts:

*   For **even indices** ($2i$), the value is calculated using a `sin` function.
*   For **odd indices** ($2i + 1$), the value is calculated using a `cos` function.

![Equation: PE(pos, 2i) = sin(pos / 10000^(2i/d_model))](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/03__image034.png)

![Equation: PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/03__image035.png)

Let's deconstruct this. Don't be intimidated by math. At its heart, this formula is designed to do exactly what binary encoding did: create a set of waves that oscillate at different frequencies. The term $1 / 10000^{(2i / d\_model)}$ is simply the **frequency** of the wave.

Notice that the index $i$ is in the denominator. This means:

*   For **low indices** (small $i$), the frequency is high, and the wave oscillates very fast.
*   For **high indices** (large $i$), the frequency is low, and the wave oscillates very slowly.

This is the exact same property we discovered with binary encoding! Sinusoidal encodings preserve the crucial intuition of using fast-changing signals for fine-grained local information and slow-changing signals for coarse-grained global information.

##### Figure 3.18 Visualization of sinusoidal positional encodings for a GPT-2-sized model. The plots show the encoding values for different indices (i) across 1024 positions.
![Figure 3.18 Visualization of sinusoidal positional encodings for a GPT-2-sized model. The plots show the encoding values for different indices (i) across 1024 positions.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/03__image036.png)

As figure 3.18 illustrates, the resulting curves are exactly what we wanted:

*   **They have the same multi-frequency property as binary encodings:** The wave for index=1 oscillates much faster than the wave for index=150.
*   **They are smooth and continuous:** The jumpy, discrete steps have been replaced with smooth sine and cosine waves, which are much better for the gradient-based optimization used to train LLMs.

This formula is a brilliant piece of engineering. It solves the discontinuity problem of binary encodings while preserving their most important feature. However, the use of both sine and cosine together unlocks an even deeper, more powerful property.

### 3.10.2 The power of rotation: Encoding relative positions

The sinusoidal formula works, but why the specific choice of pairing `sin` for even indices and `cos` for odd indices? This isn't an arbitrary decision. This specific pairing endows the positional encodings with a remarkable property: **the positional encoding for any position can be represented as a linear rotation of the encoding for any other position.**

This is a crucial property. It means there is a simple, consistent mathematical relationship between the encodings of different positions. If the model can learn this simple relationship, it can generalize its understanding of positions far more effectively than if it had to memorize each position's encoding independently.

To understand how this works, let's visualize it. In the sinusoidal positional encoding, we can think of each pair of dimensions ($2i$ and $2i+1$) as the coordinates of a 2D vector.

*   Let's define the x-coordinate as the value at the even index ($2i$), which is $\sin(\theta)$.
*   Let's define the y-coordinate as the value at the odd index ($2i+1$), which is $\cos(\theta)$.

Here, the angle $\theta$ is determined by two factors: the token's position in the sequence and the index of the dimension pair. The formula is:

![Equation: theta = position / 10000^(2i/d)](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/03__image037.png)

where $d$ represents the absolute position (or index) of the token in the input sequence.

##### Figure 3.19 A pair of sinusoidal positional encoding values can be viewed as the coordinates of a 2D vector v1 on a unit circle, defined by the angle θ.
![Figure 3.19 A pair of sinusoidal positional encoding values can be viewed as the coordinates of a 2D vector v1 on a unit circle, defined by the angle θ.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/03__image038.png)

As shown in figure 3.19, for a given position $p$ and an index pair $i$, the positional encoding values form a vector $v1$ on a circle. Now, here is the puzzle: how can we find the positional encoding for a new position that is **k steps away**, $p + k$, using our existing encoding for position $p$?

The answer is remarkably elegant. It turns out that the positional encoding vector for position $p + k$ is simply the original vector for position $p$ **rotated by a fixed angle.**

##### Figure 3.20 The positional encoding vector v2 for a future position p + k is a simple rotation of the original vector v1 for position p.
![Figure 3.20 The positional encoding vector v2 for a future position p + k is a simple rotation of the original vector v1 for position p.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/03__image039.png)

As illustrated in figure 3.20, if we want to find the positional encoding for a position that is $k$ steps away, we just need to rotate our original vector $v1$ by an angle $\theta_1 = k * \omega_i$. This is a direct consequence of the trigonometric identities $\sin(A+B)$ and $\cos(A+B)$.

This means that relative positional relationships are encoded as simple rotations. The model doesn't need to learn complex, arbitrary patterns. It can learn that to understand the relationship between a token at position $p$ and a token at position $p+2$, it just needs to apply a consistent "two-step rotation."

This is the genius of using the `sin` and `cos` pair.

*   It provides the smooth, multi-frequency signals we need.
*   It embeds a simple, linear relationship between positions, making it much easier for the Transformer to learn and generalize positional patterns.

This powerful technique, introduced in the original Transformer paper, became the gold standard for positional encodings for years. However, it still has one subtle, lingering flaw.

### 3.10.3 The remaining flaw: Still polluting the token embeddings

Sinusoidal positional encodings are a brilliant solution. They are continuous, they capture multi-frequency positional signals, and they encode relative positions as simple rotations. They seem to solve all of our problems. So why did the field move on to something else like RoPE?

The answer lies in how these positional encodings are integrated into the model.

If we look back at the overall architecture (Figure 3.11), we see that the positional embeddings are **directly added** to the token embeddings before they enter the first Transformer block.

##### Figure 3.21 The fundamental issue with sinusoidal positional encodings. The positional information is added directly to the token's semantic embedding, potentially altering or "polluting" its original meaning.
![Figure 3.21 The fundamental issue with sinusoidal positional encodings. The positional information is added directly to the token's semantic embedding, potentially altering or "polluting" its original meaning.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/03__image040.png)

As figure 3.21 highlights, even though the magnitude of the sinusoidal values is small (between -1 and 1), the very act of addition mixes the positional signal with the semantic signal. The vector that represents the meaning of the word "dog" is fundamentally altered before the model even begins to process it. This is the **semantic pollution** problem, or more formally representation entanglement.

It is worth noting that while this is a theoretical issue, it is not a catastrophic one. Transformers are remarkably powerful; their deep stacks of layers can learn to separate and re-interpret the mixed token and positional information. Indeed, models trained with additive sinusoidal encodings have achieved excellent performance for years. However, the core insight remained: could we find a cleaner method that avoids this entanglement from the start?

Ideally, we want the semantic information from the token embeddings to be preserved as cleanly as possible. This led researchers to ask a critical question: must we inject positional information at the very beginning?

The attention mechanism is the part of the model where the relationships between tokens and thus their relative positions truly matter. The influence of one token on another is quantified by the interaction between their Query and Key vectors.

##### Figure 3.22 The attention score calculation. The influence of one token on another is quantified by the dot product of their respective Query and Key vectors. This core interaction is where relative position truly matters.
![Figure 3.22 The attention score calculation. The influence of one token on another is quantified by the dot product of their respective Query and Key vectors. This core interaction is where relative position truly matters.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/03__image041.png)

This insight, prompted by the diagram in figure 3.22, is the final step in our journey. The diagram reminds us that the entire measure of relevance between tokens boils down to the interaction between their Query and Key vectors. This begs a critical question: if this is where positional relationships are quantified, why are we modifying the token embeddings at the very beginning?

This leads to a new, cleaner approach:

1.  **Don't touch the initial token embeddings.** Let them carry pure semantic information into the Transformer blocks.
2.  **Inject positional information later,** directly into the Query and Key vectors right before the attention scores are calculated.

But how should we inject this information? If we simply add a positional vector to the Query and Key vectors, we might still change their magnitude and alter their learned representations. This leads to the final, elegant idea.

##### Figure 3.23 The core ideas of RoPE. We can avoid polluting the original vectors by rotating them instead of adding to them, and we can apply this rotation directly to the Query and Key vectors where positional information is most relevant.
![Figure 3.23 The core ideas of RoPE. We can avoid polluting the original vectors by rotating them instead of adding to them, and we can apply this rotation directly to the Query and Key vectors where positional information is most relevant.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/03__image042.png)

As figure 3.23 suggests, what if we combine the best ideas we've discovered?

*   What if we inject the positional information directly into the **Query and Key vectors**?
*   And what if, instead of adding a vector, we simply **rotate** them, preserving their original magnitude and learned semantic direction?

These two ideas together form the foundation of **Rotary Positional Encoding (RoPE)**, the state-of-the-art technique that we will now explore in detail.