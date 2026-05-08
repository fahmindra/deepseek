---
slug: ch09-attempt-2-a-step-forward---binary-positional-encodings
title: "3.9 Attempt #2: A step forward - Binary positional encodings"
layout: chapter
---

The failure of integer positional encodings taught us a valuable lesson: the magnitude of our positional values matters. We need a system that can represent unique positions without using large, unbounded numbers that would overwhelm the semantic token embeddings.

This leads us to a natural next thought: what if we use a numerical system where the values are inherently constrained? This is the core idea behind **Binary Positional Encoding.**

### 3.9.1 Solving the magnitude problem with binary representation

Instead of using the raw integer of a position, we can represent it using its binary form. The binary system uses only two digits, 0 and 1, which perfectly solves our magnitude problem.

Let's consider a token at position 200. If our model's embedding dimension is 8, we can represent this position with an 8-bit binary number:

*   **Position 200 (Integer):** `200`
*   **Position 200 (8-bit Binary):** `11001000`

We can now create a positional embedding vector directly from this binary representation: `[1, 1, 0, 0, 1, 0, 0, 0]`.

##### Figure 3.13 Binary Positional Encoding. The position number is converted to its binary representation, resulting in a vector of 0s and 1s that can be safely added to the token embedding without overwhelming its semantic information.
![Figure 3.13 Binary Positional Encoding. The position number is converted to its binary representation, resulting in a vector of 0s and 1s that can be safely added to the token embedding without overwhelming its semantic information.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/03__image029.png)

As shown in figure 3.13, this approach is a significant improvement. The positional values are now all 0s or 1s, which are on a similar order of magnitude as the small floating-point numbers in the token embedding. When we add them together, the semantic information is preserved, not polluted. We have successfully solved the primary issue with integer encodings.

This seems like a great solution. It provides a unique, fixed-length, and bounded representation for every position. However, by looking closer at the structure of these binary numbers, we can uncover a much deeper, more interesting pattern that will eventually lead us to the state-of-the-art.

### 3.9.2 Uncovering a deeper pattern: Oscillation frequencies

Binary positional encoding solves the magnitude problem, but it also opens the door to a new way of thinking about positions. Let's analyze the structure of these binary numbers more closely. Within each binary representation, we can think of each digit as occupying an **index**, from the **Least Significant Bit (LSB)** on the right to the **Most Significant Bit (MSB)** on the left.

##### Figure 3.14 The oscillating frequency of bits in binary positional encodings. Lower indices (like the LSB) oscillate rapidly across positions, while higher indices (like the MSB) oscillate slowly.
![Figure 3.14 The oscillating frequency of bits in binary positional encodings. Lower indices (like the LSB) oscillate rapidly across positions, while higher indices (like the MSB) oscillate slowly.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/03__image030.png)

As figure 3.14 illustrates, when we look at a sequence of consecutive positions (from 64 to 75), a clear pattern emerges in how the bits at different indices change:

*   **Index 1 (LSB):** Look at the rightmost column. The values flip at every single position: `0, 1, 0, 1, 0, 1...` This bit is **oscillating** very rapidly.
*   **Index 2:** The next bit flips every two positions: `0, 0, 1, 1, 0, 0...` It oscillates, but at half the frequency of the LSB.
*   **Index 3:** This bit flips every four positions: `0, 0, 0, 0, 1, 1...` Its frequency is even lower.
*   **Index 8 (MSB):** The leftmost bit doesn't change at all in this range. It oscillates the slowest, flipping only every 128 positions.

This leads us to a profound observation: **lower indices oscillate fast between positions, while higher indices oscillate slow between positions.** We can visualize this relationship between the index position and its oscillation frequency directly.

##### Figure 3.15 The oscillation frequency for each bit index in an 8-bit binary representation. The frequency is highest for the lowest index and decreases exponentially for higher indices.
![Figure 3.15 The oscillation frequency for each bit index in an 8-bit binary representation. The frequency is highest for the lowest index and decreases exponentially for higher indices.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/03__image031.png)

This pattern can be seen even more clearly when we plot the value of each index across a longer range of positions.

##### Figure 3.16 The oscillation patterns of each bit index across 128 positions. Index 1 (top) oscillates very quickly, while Index 8 (bottom) oscillates very slowly.
![Figure 3.16 The oscillation patterns of each bit index across 128 positions. Index 1 (top) oscillates very quickly, while Index 8 (bottom) oscillates very slowly.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/03__image032.png)

This visualization, shown in figure 3.16, is incredibly important. It reveals that binary encoding doesn't just give each position a unique ID; it decomposes the position into a set of waves, or oscillating signals, each with a different frequency.

1.  **The fast-changing, low indices** (like Index 1 and 2) provide **fine-grained information.** They are very good at distinguishing between adjacent positions (e.g., position 64 vs. 65).
2.  **The slow-changing, high indices** (like Index 7 and 8) provide **coarse-grained information.** They are good at telling the model that two distant positions (e.g., 5 and 100) are far apart, as their high-index bits will be different.

This discovery that we can represent position using a combination of fast and slow oscillating signals is the key intuition that led to the development of sinusoidal and, eventually, rotary positional encodings. However, binary encoding still has one final, critical flaw.

### 3.9.3 The new problem: The issue with discontinuous jumps

Binary positional encoding has brought us tantalizingly close to a perfect solution. It provides a unique, bounded representation for each position and even decomposes this information into a set of meaningful, multi-frequency signals. So, what's the problem?

The issue lies in the nature of the oscillations themselves. Look again at the plot in figure 3.16. The transition from a 0 to a 1 for any given index is instantaneous and abrupt. These are **discrete jumps**, creating a series of "square waves" rather than smooth curves.

These "jumpy" and discontinuous values are **not good for optimization**. Deep learning models, including Transformers, are trained using gradient-based optimization. This process works best when the functions involved are smooth and continuous. The sharp, step-like changes from 0 to 1 in binary encoding create a "bumpy" landscape for the optimization algorithm to navigate. It becomes difficult for the model to learn the subtle relationships between positions when the underlying positional signal is so abrupt.

Ideally, what we want is a system that keeps the brilliant multi-frequency oscillation pattern we discovered in binary encoding but makes the transitions smooth. We want a graph that has the same properties as figure 3.16, but with smooth, continuous waves instead of sharp, square steps. This immediately suggests an intuition: if we want smooth, oscillating waves, what are the most famous mathematical functions that produce them? **Sine and cosine.**

This exact line of reasoning is what led the authors of the original "Attention Is All You Need" paper to develop the next major innovation in positional encodings. By replacing the discrete, jumpy signals of binary encoding with the smooth, continuous waves of sinusoidal functions, they created a method that is both mathematically elegant and far more effective for training deep neural networks.