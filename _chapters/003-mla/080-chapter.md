---
slug: ch08-attempt-1-the-naive-approach---integer-positional-encodings
title: "3.8 Attempt #1: The naive approach - integer positional encodings"
layout: chapter
---

Now that we've established the critical need for positional information, let's think from first principles. If you were tasked with designing a system to encode a token's position, what is the simplest, most direct method you could imagine? Most likely, you would suggest using the position number itself.

### 3.8.1 The simple idea: Using position numbers directly

This is the core idea behind Integer Positional Encoding. We simply take the integer index of a token in the sequence and use that value to create its positional embedding.

Let's return to our example: "**The dog chased another dog.**"

*   The first "dog" is at position 2.
*   The second "dog" is at position 5.

To create a positional embedding vector that we can add to the token embedding, we need a vector of the same dimension. If our token embeddings have a dimension of 8, we would simply create a vector where every element is the position number.

*   **Positional Embedding for the first "dog" (position 2):** `[2, 2, 2, 2, 2, 2, 2, 2]`
*   **Positional Embedding for the second "dog" (position 5):** `[5, 5, 5, 5, 5, 5, 5, 5]`

When we add these vectors to the identical token embeddings for "dog", the resulting input embeddings fed into the Transformer will be different.

```python
InputEmbedding_dog1 = TokenEmbedding_dog + [2, 2, 2, ...]
InputEmbedding_dog2 = TokenEmbedding_dog + [5, 5, 5, ...]
```

Problem solved, right? We are now successfully providing the model with a way to distinguish between the two dogs. The transformer will produce different context vectors for them, and it can learn the different roles they play in the sentence.

However, this simple solution introduces a new, and much more severe, problem.

### 3.8.2 The major flaw: Polluting semantic embeddings with large magnitudes

The main problem with Integer Positional Encoding lies in the **magnitude** of the values. Token embeddings, which capture the semantic meaning of words, are typically initialized as small values clustered around zero. This is a deliberate choice in deep learning that helps with training stability. A token embedding for "dog" might look something like `[0.1, 0.3, 0.2, 0.8, ...]`, where the values are small.

Now, consider the positional embeddings. While the values are small for the first few tokens (2, 5, etc.), what happens in a model with a large context window? A token at position 200 would have a positional embedding of `[200, 200, 200, ...]`. A token at position 1000 would have `[1000, 1000, 1000, ...]`.

When we add these two vectors together, the large integer values of the positional encoding completely overwhelm the small, nuanced values of the token embedding.

##### Figure 3.12 The problem with Integer Positional Encoding. The large magnitude of the positional values completely dominates the small, semantic values of the token encoding, making it very difficult for the model to separate the two sources of information.
![Figure 3.12 The problem with Integer Positional Encoding. The large magnitude of the positional values completely dominates the small, semantic values of the token encoding, making it very difficult for the model to separate the two sources of information.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/03__image028.png)

As figure 3.12 illustrates, the sum of the two vectors will be dominated by the positional information. The subtle semantic information, the very core meaning of the word "dog," is effectively lost in the noise. **We are polluting the semantics.**

This defeats the purpose of having rich token embeddings in the first place. We want the model to learn from the meaning of words, but this naive approach forces the positional information to drown it out. The model would struggle to learn that the token at position 200 and the token at position 500 are related if they both happen to be the word "dog."

Ideally, we need a method that satisfies two criteria:

1.  It must provide a unique encoding for each position.
2.  The values of the encoding must be constrained to a small range (e.g., between 0 and 1) so they don't overpower the token embeddings.

This line of thinking leads directly to our second attempt: using a numerical system that is inherently constrained.