---
slug: ch-v4-post-training
title: "9.4 Post-Training: On-Policy Distillation and Agent-Specific Design"
layout: chapter
---

## On-Policy Distillation (OPD)

Moving away from the mixed RL methods of R1, V4 consolidates knowledge from independent domain experts into a single student model:

```
L_OPD(θ) = Σ_{i=1}^{N} w_i · D_KL(π_θ ‖ π_{E_i})
```

Where:
- π_θ: Unified student policy
- π_{E_i}: Policy of the i-th domain-specific teacher expert
- w_i: Static weight for that expert's importance

By using **full-vocabulary logit distillation** rather than token-level KL estimates, V4 achieves highly stable gradient estimates, ensuring faithful transfer of reasoning capabilities across math, code, and general domains.

## Interleaved Thinking Across Tool Calls

A critical agent-specific innovation: V4 preserves reasoning content **across user message boundaries** when the conversation contains tool calls. Key behaviors:

- **With tools:** Reasoning history is retained across all rounds, including user turns — enabling coherent cumulative chain-of-thought over long-horizon agent tasks
- **Without tools (conversational):** Reasoning is flushed at each new user message to keep context concise

This contrasts with V3.2, which discarded reasoning traces whenever a new user message arrived, forcing agents to reconstruct state for multi-turn workflows.

## Tool-Call Schema with Dedicated Tokens

V4 introduces the `|DSML|` special token and an XML-based tool-call format:

- Separates string parameters from structured (JSON) parameters
- Reduces escaping failures compared to JSON-in-string tool calls
- Eliminates a class of parsing errors around numbers and booleans

## Three Reasoning Modes

All instruct models support:
1. **Non-think:** Fast, no chain of thought
2. **Think High:** Explicit reasoning in `<think>` blocks
3. **Think Max:** Maximum reasoning effort with dedicated system prompt; requires ≥384K context window

Recommended parameters across all modes: `temperature=1.0, top_p=1.0`
