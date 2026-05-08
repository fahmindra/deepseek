---
slug: ch-v4-benchmarks
title: "9.5 Benchmarks: Competitive Reasoning, Dominant Agent Performance"
layout: chapter
---

## Knowledge and Reasoning

DeepSeek-V4-Pro-Max (highest reasoning effort):

- **SimpleQA-Verified:** 57.9% — leading open-source, establishing a new frontier for parametric world knowledge
- **Codeforces Elo:** 3206 — on par with GPT-5.4-xHigh
- **MRCR (1M token retrieval):** 83.5 MMR, outperforming Gemini-3.1-Pro (76.3) with negligible degradation across the full context window
- **MRCR 8-needle accuracy:** Stays above 0.82 through 256K tokens, holds at 0.59 at 1M

## Agent Benchmarks (V4-Pro-Max)

Where V4 separates from the field:

| Benchmark | V4-Pro-Max | Best Competitor |
|-----------|-----------|-----------------|
| Terminal Bench 2.0 | 67.9 | Gemini-3.1-Pro (68.5), GPT-5.4-xHigh (75.1) |
| SWE Verified | 80.6 | Opus-4.6-Max (80.8), Gemini-3.1-Pro (80.6) |
| MCPAtlas Public | 73.6 | Opus-4.6-Max (73.8) |
| Toolathlon | 51.8 | K2.6 (50.0) |

On the internal R&D coding benchmark (30 curated tasks across PyTorch, CUDA, Rust, C++):
- **V4-Pro-Max: 67% pass rate**
- Sonnet 4.5: 47%
- Opus 4.5: 70%

In an internal survey of 85 DeepSeek developers using V4-Pro as their daily driver, **52% said it was ready to replace their current primary coding model** and 39% leaned toward yes.

## The Real Innovation

The benchmark numbers are competitive but not always state-of-the-art. The true innovation is that V4 achieves these results **at practical cost for long-context, long-running agent workloads** — the scenarios where previous models would break due to KV cache overflow, context budget exhaustion, or tool-call degradation.

By reducing single-token inference FLOPs to 27% of V3.2 and KV cache to 10%, V4 makes million-token contexts economically viable for the first time.
