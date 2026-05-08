---
slug: ch-v4-muon-mhc
title: "9.3 Stabilizing Trillion-Parameter Training: mHC and Muon"
layout: chapter
---

Scaling to 1.6 trillion parameters across 61 Transformer layers introduces profound instability in signal propagation and optimization. V4 addresses this with two advanced mathematical substrates.

## Manifold-Constrained Hyper-Connections (mHC)

V4 fully adopts mHC from the earlier paper. The state update is governed by:

```
X_{l+1} = B_l · X_l + C_l · F_l(A_l · X_l)
```

Where:
- X_l: Residual state before the l-th layer
- F_l: Non-linear transformation (MoE layer)
- A_l, C_l: Input/output projections (non-negative constrained)
- **B_l: Residual mapping — projected onto doubly stochastic manifold via Sinkhorn-Knopp**

By constraining B_l to the Birkhoff polytope, its spectral norm is bounded by 1, guaranteeing the residual transformation remains strictly non-expansive across all 61 layers. This completely averts signal explosion.

## The Muon Optimizer

V4 adopts Muon — an optimizer that orthogonalizes weight matrices using hybrid Newton-Schulz iterations:

```
M_k = a·M_{k-1} + b·(M_{k-1}·M^T_{k-1})·M_{k-1} + c·(M_{k-1}·M^T_{k-1})²·M_{k-1}
```

Where a, b, c are predefined coefficients:
- Early iterations: Drive rapid convergence
- Final iterations: Stabilize singular values precisely at 1

This orthogonalization ensures optimal gradient flow through massive MoE layers without the stagnation typical of simpler first-order methods like AdamW.

**Why Muon matters:** In MoE architectures, different experts receive vastly different amounts of training signal. Muon's orthogonalization prevents any single expert's weight matrix from dominating or stagnating, enabling all experts to contribute effectively.

## Complementary Stabilization

mHC and Muon work at different levels:
- **mHC** stabilizes signal propagation between layers (the residual stream)
- **Muon** stabilizes optimization within layers (the weight matrices)

Together they enable stable training at the trillion-parameter scale — a regime where standard residual connections with AdamW would exhibit catastrophic instability.
