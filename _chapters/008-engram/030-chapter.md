---
slug: ch-engram-allocation
title: "8.3 The Sparsity Allocation Problem and U-Shaped Scaling Law"
layout: chapter
---

## The Sparsity Allocation Problem

Given a fixed total parameter budget, how should capacity be distributed between MoE experts (computation) and Engram memory (lookup)?

Key metrics:
- **P_tot**: Total trainable parameters
- **P_act**: Activated parameters per token (determines FLOPs)
- **P_sparse** = P_tot - P_act: "Free" parameter budget (unselected experts + unretrieved embeddings)

The allocation ratio ρ ∈ [0,1] determines what fraction of the inactive-parameter budget goes to MoE:

```
P_MoE(sparse) = ρ · P_sparse
P_Engram = (1 - ρ) · P_sparse
```

Where:
- ρ = 1: Pure MoE (all inactive parameters are routed experts)
- ρ < 1: Reallocate freed parameters to Engram embedding slots

## Experimental Protocol

Evaluated at two compute regimes, maintaining constant sparsity ratio P_tot/P_act ≈ 10:

| Compute Budget | P_tot | P_act | Pure MoE (ρ=1) |
|---------------|-------|-------|-----------------|
| 2×10²⁰ FLOPs | 5.7B | 568M | 106 experts |
| 6×10²⁰ FLOPs | 9.9B | 993M | 99 experts |

All runs use identical training pipelines and hyperparameters.

## The U-Shaped Law

Validation loss exhibits a **consistent U-shaped relationship** with allocation ratio ρ:

- **MoE-dominated (ρ → 100%):** Loss degrades — model lacks dedicated memory for static patterns, forced to reconstruct them through computation
- **Engram-dominated (ρ → 0%):** Loss degrades — model loses conditional computation capacity for dynamic reasoning
- **Optimum:** ρ ≈ 75-80% — roughly 20-25% of sparse budget allocated to Engram memory

**Key result:** In the 10B regime, validation loss improves from 1.7248 (pure MoE) to 1.7109 at the optimum (Δ = 0.0139). The optimum location is **stable across compute scales**, suggesting a robust architectural preference.

## Infinite Memory Scaling

With a fixed MoE backbone (P_tot ≈ 3B, P_act = 568M), scaling Engram slots from 2.58×10⁵ to 1.0×10⁷ (up to ≈13B additional parameters) shows a **strict power-law improvement** in validation loss — larger memory continues to pay off without additional FLOPs.
