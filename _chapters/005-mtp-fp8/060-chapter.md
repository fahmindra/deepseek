---
slug: ch06-summary
title: "5.6 Summary"
layout: chapter
---

*   Multi-Token Prediction (MTP) enhances data efficiency by training models to predict a horizon of future tokens simultaneously, providing richer gradient signals for long-range coherence.
*   The MTP training objective implicitly assigns higher weight to "choice points" or critical pattern-shifting tokens in a sequence, thereby developing better planning capabilities in the model.
*   DeepSeek's causal MTP implementation sequentially refines a hidden state, where the prediction for one token informs the next, creating a powerful forecasting mechanism.
*   Quantization is essential for large-scale training, reducing memory usage and computational cost by converting high-precision parameters to low-precision formats like FP8.
*   DeepSeek’s FP8 framework uses a mixed precision strategy, performing core computations in low precision while storing sensitive components, such as master weights and optimizer states, in high precision to ensure numerical stability.
*   **Fine-Grained Quantization:** To mitigate precision loss from outliers, DeepSeek calculates separate scaling factors for small blocks of activations and weights, preserving fidelity at a local level.
*   **Increasing Accumulation Precision:** DeepSeek promotes intermediate results from low-precision Tensor Cores to high-precision CUDA Cores, where accumulation occurs in full FP32 precision to prevent underflow errors.
*   **Mantissa over Exponents:** DeepSeek uniformly uses the higher-precision E4M3 format for both forward and backward passes, enabled by fine-grained quantization, which effectively manages dynamic range.
*   **Online Quantization:** DeepSeek calculates scaling factors in real-time based on the current batch's data, ensuring accurate scaling and avoiding the instability associated with historical data.