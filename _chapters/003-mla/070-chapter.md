---
slug: ch07-the-problem-of-order
title: "3.7 The problem of order"
layout: chapter
---

The attention mechanism, for all its power, has a fundamental limitation: it has no built-in sense of sequence order. By default, it treats a sentence as a "bag of words," where the position of a word doesn't matter. To the raw attention mechanism, the sentences "**the dog chased the cat**" and "**cat the chased dog the**" look identical. This is a huge problem, as the order of words is critical to the meaning of a sentence.

Let's consider an example:

> "The dog chased another dog."

This sentence contains two instances of the word "dog." Semantically, the token embedding for the first "dog" (the chaser) and the second "dog" (the one being chased) are identical. If we feed these token embeddings directly into the Transformer blocks without any positional information, the model has no way to distinguish between them.

As shown in figure 3.11, the input to the Transformer block for both tokens would be exactly the same. Consequently, the output context vectors produced by the attention mechanism would also be identical. The model would be blind to the fact that one dog is the subject and the other is the object. It cannot understand the true context of the sentence.

##### Figure 3.11 Without positional information, the Transformer processes identical tokens in different positions in the exact same way, leading to identical context vectors and a loss of crucial contextual understanding.
![Figure 3.11 Without positional information, the Transformer processes identical tokens in different positions in the exact same way, leading to identical context vectors and a loss of crucial contextual understanding.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/03__image027.png)

To solve this, we must find a way to tell the model where each token is located in the sequence. We need to create a distinction between the first "dog" at position 2 and the second "dog" at position 5. The mechanism for doing this is called **positional encoding**: we create a vector that represents a token's position and add it to the token's semantic embedding.

The challenge is to inject this positional information in a way that is both useful for the model and does not destroy the valuable semantic information contained in the original token embeddings. As we will see, this is a surprisingly difficult needle to thread. Let's start by exploring the most straightforward, intuitive solution one might think of first: simply using the integer position numbers themselves.