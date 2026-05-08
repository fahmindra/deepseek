---
slug: ch-mhc-experiments
title: "7.4 Experiments: Stability, Scaling, and Performance Gains"
layout: chapter
---

## Experimental Setup

mHC was evaluated on language model pretraining at three scales:
- **3B parameters**
- **9B parameters**
- **27B parameters**

All models used identical training pipelines and optimization hyperparameters. The key comparison is between standard residual connections, unconstrained HC, and mHC.

## Training Stability

The most striking result is stability. At the 27B scale:

- **HC (unconstrained):** Exhibits a loss surge around step 12K, coinciding with gradient norm explosion. The composite residual mapping ∏H_{L-i}^res shows extreme Amax Gain Magnitudes peaking at 3000, confirming signal explosion.

- **mHC:** Trains smoothly throughout, with well-behaved gradient norms and no loss anomalies. The doubly stochastic constraint on every residual mapping matrix ensures non-expansive signal propagation at every layer.

## Performance: Absolute Loss Gap

Compared to standard residual connections (measured as the difference in validation loss):

- HC: -0.027 absolute loss improvement (but unstable)
- mHC: Similar or better loss improvement, **with guaranteed stability**

The key result: mHC achieves the performance benefits of HC (expanded residual stream, richer connectivity) while maintaining the training stability of standard residual connections.

## Scaling Behavior

As model size increases from 3B to 9B to 27B:
- The performance advantage of mHC over standard residuals **persists and grows**
- HC's instability becomes **more severe** at larger scales
- mHC's training overhead remains at **6-7%**, negligible for large-scale models

## Key Insight

mHC proves that the identity mapping property — the core insight from ResNet in 2015 — remains critical for modern architectures. By constraining the expanded residual stream to the Birkhoff polytope via the Sinkhorn-Knopp algorithm, mHC enables richer connectivity patterns without sacrificing the training stability that made residual connections foundational to deep learning.
