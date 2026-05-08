---
slug: ch-mhc-method
title: "7.2 mHC: Projecting onto the Birkhoff Polytope"
layout: chapter
---

Manifold-Constrained Hyper-Connections (mHC) solves HC's instability by constraining the residual mapping H_l^res onto a specific mathematical manifold — the **Birkhoff polytope** of doubly stochastic matrices.

## The Doubly Stochastic Constraint

A matrix H_l^res is doubly stochastic if:
- All entries are non-negative
- Each row sums to 1
- Each column sums to 1

Formally, mHC constrains H_l^res to the manifold M^res:

```
P_M^res(H_l^res) := {H_l^res ∈ ℝ^{n×n} | H_l^res·1_n = 1_n, 1_n⊤·H_l^res = 1_n⊤, H_l^res ≥ 0}
```

When n=1, this degenerates to the scalar 1 — recovering the **original identity mapping**.

## Three Rigorous Theoretical Properties

**1. Norm Preservation:** The spectral norm of a doubly stochastic matrix is bounded by 1 (‖H_l^res‖₂ ≤ 1). This means the learnable mapping is **non-expansive**, effectively mitigating the gradient explosion problem.

**2. Compositional Closure:** The set of doubly stochastic matrices is **closed under matrix multiplication**. The composite residual mapping across multiple layers, ∏H_{L-i}^res, remains doubly stochastic, preserving stability throughout the entire depth.

**3. Birkhoff Polytope Geometry:** The manifold M^res forms the convex hull of permutation matrices. The residual mapping acts as a **convex combination of permutations**, functioning as a robust feature fusion mechanism that monotonically increases information mixing across streams.

## The Sinkhorn-Knopp Algorithm

To enforce these constraints during training, mHC uses the **Sinkhorn-Knopp algorithm** — an entropic projection method that iteratively normalizes rows and columns until the matrix converges to a doubly stochastic form.

The process:
1. Start with raw parameters (unconstrained)
2. Ensure non-negativity (e.g., via ReLU or exp)
3. Alternate row and column normalization: divide each row by its sum, then each column by its sum, repeating until convergence
4. The result is a valid doubly stochastic matrix on the Birkhoff polytope

Additional constraints:
- **Input mappings H_l^pre and output mappings H_l^post** have non-negativity constraints enforced to prevent signal cancellation from composing positive and negative coefficients
