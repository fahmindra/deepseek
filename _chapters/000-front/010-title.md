---
slug: book
title: "Build DeepSeek from Scratch"
abstract: "A comprehensive hands-on guide covering every major DeepSeek innovation — from KV Cache and Multi-Head Latent Attention to Mixture-of-Experts, Multi-Token Prediction, FP8 Quantization, Sparse Attention, Manifold-Constrained Hyper-Connections, Conditional Memory (Engram), and the million-token DeepSeek-V4 architecture."
---

This book guides you through building the key components of the DeepSeek language model family from scratch. Based on research papers from DeepSeek-AI spanning V2 through V4, each part combines theory with practical PyTorch implementations.

DeepSeek represents a turning point in open-source AI. By the end of this book, you will not only understand what makes each DeepSeek generation unique but also how to implement those innovations yourself.

## What you'll build

- **Part 1: Introduction** — Why DeepSeek matters and a high-level roadmap
- **Part 2: KV Cache** — Solving the inference bottleneck, from MHA to MQA and GQA
- **Part 3: Multi-Head Latent Attention** — DeepSeek's breakthrough 64× memory reduction
- **Part 4: Mixture-of-Experts** — Scaling intelligence efficiently with sparse expert routing
- **Part 5: MTP & FP8 Quantization** — Advanced training objectives and numerical precision
- **Part 6: DeepSeek Sparse Attention (DSA)** — O(L·k) complexity via lightning indexing
- **Part 7: mHC** — Manifold-Constrained Hyper-Connections for stable deep networks
- **Part 8: Engram** — Conditional memory as a second axis of sparsity
- **Part 9: DeepSeek-V4** — Hybrid CSA/HCA attention for million-token contexts

## Paper Sources

- DeepSeek-V2 / V3 / R1 papers (2024-2025)
- DeepSeek-V3.2: Pushing the Frontier of Open LLMs (Dec 2025)
- mHC: Manifold-Constrained Hyper-Connections (Dec 2025)
- Engram: Conditional Memory via Scalable Lookup (Jan 2026)
- DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence (Apr 2026)
