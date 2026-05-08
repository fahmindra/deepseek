<!-- ---
slug: ch-10-3
title: "Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models"
layout: chapter
---

# Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models

## Abstract

While Mixture-of-Experts (MoE) scales capacity via conditional computation, Transformers lack a native primitive for knowledge lookup, forcing them to inefficiently simulate retrieval through computation. To address this, we introduce conditional memory as a complementary sparsity axis, instantiated via **Engram**, a module that modernizes classic $N$-gram embedding for $\mathcal{O}(1)$ lookup. By formulating the *Sparsity Allocation* problem, we uncover a U-shaped scaling law that optimizes the trade-off between neural computation (MoE) and static memory (Engram). Guided by this law, we scale Engram to 27B parameters, achieving superior performance over a strictly iso-parameter and iso-FLOPs MoE baseline. Most notably, while the memory module is expected to aid knowledge retrieval (e.g., MMLU $+3.4$; CMMLU $+4.0$), we observe even larger gains in general reasoning (e.g., BBH $+5.0$; ARC-Challenge $+3.7$) and code/math domains (HumanEval $+3.0$; MATH $+2.4$). Mechanistic analyses reveal that Engram relieves the backbone’s early layers from static reconstruction, effectively deepening the network for complex reasoning. Furthermore, by delegating local dependencies to lookups, it frees up attention capacity for global context, substantially boosting long-context retrieval (e.g., Multi-Query NIAH: $84.2 \to 97.0$). Finally, Engram establishes infrastructure-aware efficiency: its deterministic addressing enables runtime prefetching from host memory, incurring negligible overhead. We envision conditional memory as an indispensable modeling primitive for next-generation sparse models. Code available at: [https://github.com/deepseek-ai/Engram](https://github.com/deepseek-ai/Engram)

---

## 1 Introduction

Sparsity is a recurring design principle for intelligent systems, spanning from biological neural circuits to modern Large Language Models (LLMs). Currently, this principle is primarily realized through Mixture-of-Experts (MoE), which scales capacity via conditional computation. Owing to its ability to drastically increase model size without proportional increases in compute, MoE has become the de facto standard for frontier models.

Despite the success of this conditional computation paradigm, the intrinsic heterogeneity of linguistic signals suggests significant room for *structural optimization*. Specifically, language modeling entails two qualitatively different sub-tasks: compositional reasoning and knowledge retrieval. While the former demands deep, dynamic computation, a substantial portion of text—such as named entities and formulaic patterns—is local, static, and highly stereotyped. The effectiveness of classical $N$-gram models in capturing such local dependencies implies that these regularities are naturally represented as computationally inexpensive lookups. Since standard Transformers lack a native knowledge lookup primitive, current LLMs are forced to **simulate retrieval through computation**. For instance, resolving a common multi-token entity requires consuming multiple early layers of attention and feed-forward networks (see Table 3). This process essentially amounts to an expensive runtime reconstruction of a static lookup table, wasting valuable sequential depth on trivial operations that could otherwise be allocated to higher-level reasoning.

To align model architecture with this linguistic duality, we advocate for a complementary axis of sparsity: conditional memory. Whereas conditional computation sparsely activates parameters to process dynamic logic, conditional memory relies on sparse lookup operations to retrieve static embeddings for fixed knowledge. As a preliminary exploration of this paradigm, we revisit $N$-gram embeddings as a canonical instantiation: local context serves as a key to index a massive embedding table via constant-time $\mathcal{O}(1)$ lookups. Our investigation reveals that, perhaps surprisingly, this static retrieval mechanism can serve as an ideal complement to modern MoE architecture—but only if it is properly designed. In this paper, we propose **Engram**, a conditional memory module grounded in the classic $N$-gram structure but equipped with modern adaptations such as tokenizer compression, multi-head hashing, contextualized gating, and multi-branch integration (detailed in Section 2).

To quantify the synergy between these two primitives, we formulate the *Sparsity Allocation* problem: given a fixed total parameter budget, how should capacity be distributed between MoE experts and Engram memory? Our experiments uncover a distinct U-shaped scaling law, revealing that even simple lookup mechanisms, when treated as a first-class modeling primitive, act as essential complements to neural computation. Guided by this allocation law, we scale Engram to a 27B-parameter model. Compared to a strictly iso-parameter and iso-FLOPs MoE baseline, Engram-27B achieves superior efficiency across diverse domains. Crucially, the gains are not limited to knowledge-intensive tasks (e.g., MMLU: $+3.4$; CMMLU: $+4.0$; MMLU-Pro: $+1.8$), where memory capacity is intuitively beneficial; we observe even more significant improvements in general reasoning (e.g., BBH: $+5.0$; ARC-Challenge: $+3.7$; DROP: $+3.3$) and code/math domains (e.g., HumanEval: $+3.0$; MATH: $+2.4$; GSM8K: $+2.2$).

Mechanistic analysis via LogitLens and CKA reveals the source of these gains: Engram relieves the backbone from reconstructing static knowledge in early layers, thereby increasing effective depth available for complex reasoning. Furthermore, by delegating local dependencies to lookups, Engram frees up attention capacity to focus on global context, enabling exceptional performance in long-context scenarios—substantially outperforming baselines on LongPPL and RULER (e.g., Multi-Query NIAH: $97.0$ vs. $84.2$; Variable Tracking: $89.0$ vs. $77.0$).

Finally, we establish infrastructure-aware efficiency as a first-class principle. Unlike MoE’s dynamic routing, Engram employs deterministic IDs to enable runtime prefetching, overlapping communication with computation. Empirical results show that offloading a 100B-parameter table to host memory incurs negligible overhead ($<3\%$). This demonstrates that Engram effectively bypasses GPU memory constraints, facilitating aggressive parameter expansion.

![The Engram Architecture](https://arxiv.org/html/2512.02556v1/x1.png)
*Figure 1: **The Engram Architecture.** The module augments the backbone by retrieving static $N$-gram memory and fusing it with dynamic hidden states via context-aware gating. This module is applied only to specific layers to decouple memory from compute, leaving the standard input embedding and un-embedding module intact.*

---

## 2 Architecture

### 2.1 Overview
As shown in Figure 1, Engram is a conditional memory module designed to augment the Transformer backbone by structurally separating static pattern storage from dynamic computation. Formally, given an input sequence $X=(x_1, \dots, x_T)$ and hidden states $\mathbf{H}^{(\ell)} \in \mathbb{R}^{T \times d}$ at layer $\ell$, the module processes each position $t$ in two functional phases: **retrieval and fusion**. First, as detailed in Section 2.2, we extract and compress suffix $N$-grams to deterministically retrieve static embedding vectors via hashing. Subsequently, in Section 2.3, these retrieved embeddings are dynamically modulated by the current hidden state and refined via a lightweight convolution. Finally, we discuss the integration with multi-branch architectures in Section 2.4 and the system-level design in Section 2.5.

### 2.2 Sparse Retrieval via Hashed $N$-grams
The first phase maps local contexts to static memory entries, involving tokenizer compression and retrieving embeddings via deterministic hashing.

##### Tokenizer Compression
While $N$-gram models typically operate directly on tokenizer outputs, standard subword tokenizers prioritize *lossless reconstruction*, often assigning disjoint IDs to semantically equivalent terms (e.g., `Apple` vs. ` apple`). To maximize semantic density, we implement a vocabulary projection layer. Specifically, we pre-compute a surjective function $\mathcal{P}: V \to V'$ that collapses raw token IDs into canonical identifiers based on normalized textual equivalence (using NFKC, lowercasing, etc.). In practice, this process achieves a 23% reduction in the effective vocabulary size for a 128k tokenizer (see Appendix C). Formally, for a token at position $t$, we map its raw ID $x_t$ to a canonical ID $x'_t = \mathcal{P}(x_t)$ to form the suffix $N$-gram $g_{t,n} = (x'_{t-n+1}, \dots, x'_t)$.

##### Multi-Head Hashing
Directly parameterizing the combinatorial space of all possible $N$-grams is intractable. Following Trito (2017), we adopt a hashing-based approach. To mitigate collisions, we employ $K$ distinct hash heads for each $N$-gram order $n$. Each head $k$ maps the compressed context to an index within an embedding table $\mathbf{E}_{n,k}$ (of prime size $M_{n,k}$) via a deterministic function $\varphi_{n,k}$:

$$z_{t,n,k} \triangleq \varphi_{n,k}(g_{t,n}), \quad \mathbf{e}_{t,n,k} = \mathbf{E}_{n,k}[z_{t,n,k}] \quad (1)$$

In practice, $\varphi_{n,k}$ is implemented as a lightweight multiplicative-XOR hash. We construct the final memory vector $\mathbf{e}_t \in \mathbb{R}^{d_{\text{mem}}}$ by concatenating all retrieved embeddings:

$$\mathbf{e}_t \triangleq \mathop{\|}_{n=2}^{N} \mathop{\|}_{k=1}^{K} \mathbf{e}_{t,n,k} \quad (2)$$

### 2.3 Context-aware Gating
The retrieved embeddings $\mathbf{e}_t$ serve as context-independent priors. Being static, however, they inherently lack contextual adaptability and may suffer from noise due to hash collisions or polysemy. To enhance expressivity and resolve this ambiguity, we employ a context-aware gating mechanism inspired by Attention. Specifically, we utilize the current hidden state $\mathbf{h}_t$—which has aggregated global context via preceding attention layers—as a dynamic Query, while the retrieved memory $\mathbf{e}_t$ serves as the source for both Key and Value projections:

$$\mathbf{k}_t = \mathbf{W}_K \mathbf{e}_t, \quad \mathbf{v}_t = \mathbf{W}_V \mathbf{e}_t \quad (3)$$

where $\mathbf{W}_K, \mathbf{W}_V$ are learnable projection matrices. To ensure gradient stability, we apply RMSNorm to the Query and Key before computing the scalar gate $\alpha_t \in (0, 1)$:

$$\alpha_t = \sigma \left( \frac{\text{RMSNorm}(\mathbf{h}_t)^\top \text{RMSNorm}(\mathbf{k}_t)}{\sqrt{d}} \right) \quad (4)$$

The gated output is defined as $\tilde{\mathbf{v}}_t = \alpha_t \cdot \mathbf{v}_t$. This design enforces semantic alignment: if the retrieved memory $\mathbf{e}_t$ contradicts the current context $\mathbf{h}_t$, the gate $\alpha_t$ tends toward zero, effectively suppressing the noise.

Finally, to expand the receptive field and enhance the model’s non-linearity, we introduce a short, depthwise causal convolution. Let $\tilde{\mathbf{V}} \in \mathbb{R}^{T \times d}$ denote the sequence of gated values. Using a kernel size $w$ (set to 4), dilation $\delta$ (set to the max $N$-gram order) and SiLU activation, the final output $\mathbf{Y}$ is computed as:

$$\mathbf{Y} = \text{SiLU} (\text{Conv1D}(\text{RMSNorm}(\tilde{\mathbf{V}}))) + \tilde{\mathbf{V}} \quad (5)$$

The Engram module is integrated into the backbone via a residual connection: $\mathbf{H}^{(\ell)} \leftarrow \mathbf{H}^{(\ell)} + \mathbf{Y}$, followed by the standard Attention and MoE. Crucially, Engram is not applied to every layer; its specific placement is governed by the system-level latency constraints detailed in Section 2.5.

![System implementation of Engram](https://arxiv.org/html/2512.02556v1/x2.png)
*Figure 2: **System implementation of Engram.** (a) Training Phase: The massive embedding tables are sharded across available GPUs. An All-to-All communication primitive is employed to retrieve active embedding rows across devices. (b) Inference Phase: Engram tables are offloaded to host memory. By exploiting the deterministic retrieval logic, the host asynchronously prefetches and transfers embeddings, overlapping communication with the on-device computation of preceding Transformer blocks.*

### 2.4 Integration with Multi-branch Architecture
In this work, rather than standard single-stream connections, we adopt the advanced multi-branch architecture as our default backbone. A defining characteristic of this architecture is the expansion of the residual stream into $M$ parallel branches, where information flow is modulated by learnable connection weights.

Although the Engram module is inherently topology-agnostic, adapting it to this multi-branch framework necessitates structural optimization to balance efficiency and expressivity. Specifically, we implement a parameter-sharing strategy: a single sparse embedding table and a Value projection matrix $\mathbf{W}_V$ are shared across all $M$ branches, whereas $M$ distinct Key projection matrices $\{\mathbf{W}_K^{(m)}\}_{m=1}^M$ are employed to enable branch-specific gating behaviors. For the $m$-th branch with hidden state $\mathbf{h}_t^{(m)}$, the branch-specific gating signal is computed as:

$$\alpha_t^{(m)} = \sigma \left( \frac{\text{RMSNorm}(\mathbf{h}_t^{(m)})^\top \text{RMSNorm}(\mathbf{W}_K^{(m)} \mathbf{e}_t)}{\sqrt{d}} \right) \quad (6)$$

The retrieved memory is then modulated by these independent gates applied to the shared value vector: $\mathbf{u}_t^{(m)} = \alpha_t^{(m)} \cdot (\mathbf{W}_V \mathbf{e}_t)$. This design allows the linear projections (one $\mathbf{W}_V$ and $M$ distinct $\mathbf{W}_K^{(m)}$) to be fused into a single dense FP8 matrix multiplication. Unless otherwise stated, all experiments utilize this integration with Manifold-Constrained Hyper-Connections ($M=4$).

### 2.5 System Efficiency: Decoupling Compute and Memory
Scaling memory-augmented models is often constrained by the limited capacity of GPU high-bandwidth memory (HBM). However, the deterministic retrieval mechanism of Engram naturally supports the decoupling of parameter storage from computational resources. Unlike MoE, which relies on runtime hidden states for dynamic routing, Engram’s retrieval indices depend solely on the input token sequence.

During training, to accommodate large-scale embedding tables, we employ standard model parallelism by sharding the tables across available GPUs. During inference, this deterministic nature enables a prefetch-and-overlap strategy. Since memory indices are known prior to the forward pass, the system can asynchronously retrieve embeddings from abundant host memory via PCIe. To effectively mask communication latency, the Engram module is placed at specific layers within the backbone, leveraging the computation of preceding layers as a buffer.

Furthermore, natural language $N$-grams inherently follow a Zipfian distribution, where a small fraction of patterns accounts for the vast majority of memory accesses. This statistical property motivates a Multi-Level Cache Hierarchy: frequently accessed embeddings can be cached in faster storage tiers (e.g., GPU HBM or Host DRAM), while the long tail of rare patterns resides in slower, high-capacity media (e.g., NVMe SSD).

![Sparsity allocation and Engram scaling](https://arxiv.org/html/2512.02556v1/x3.png)
*Figure 3: **Sparsity allocation and Engram scaling.** Left: Validation loss across allocation ratios $\rho$. Two compute budgets are shown ($2\text{e}20$ and $6\text{e}20$ FLOPs). Both regimes exhibit a U-shape, with hybrid allocation surpassing Pure MoE. Right: Scaling behavior in the infinite-memory regime. Validation loss exhibits a log-linear trend with respect to the number of embeddings.*

---

## 3 Scaling Laws and Sparsity Allocation

### 3.1 Optimal Allocation Ratio Between MoE and Engram

##### Compute-matched formulation.
We analyze the trade-off using three parameter metrics:
*   $P_{\mathrm{tot}}$: total trainable parameters, excluding vocabulary embedding and LM head.
*   $P_{\mathrm{act}}$: activated parameters per token. This quantity determines the training cost (FLOPs).
*   $P_{\mathrm{sparse}} \triangleq P_{\mathrm{tot}} - P_{\mathrm{act}}$: the *inactive* parameters, representing the “free” parameter budget.

##### Allocation ratio.
We define the allocation ratio $\rho \in [0,1]$ as the fraction of the inactive-parameter budget assigned to MoE expert capacity:
$$P_{\mathrm{MoE}}^{(\mathrm{sparse})} = \rho \, P_{\mathrm{sparse}}, \qquad P_{\mathrm{Engram}} = (1 - \rho) \, P_{\mathrm{sparse}} \quad (7)$$

##### Results and Analysis.
Figure 3 (left) reveals a consistent U-shaped relationship between validation loss and the allocation ratio $\rho$. Reallocating roughly $20\% \text{--} 25\%$ of the sparse parameter budget to Engram yields the best performance. Quantitatively, in the 10B regime, validation loss improves from $1.7248$ (at $\rho=100\%$) to $1.7109$ near the optimum of $\rho \approx 80\%$.

### 3.2 Engram under Infinite Memory Regime
We sweep the number of slots $M$ from $2.58 \times 10^5$ to $1.0 \times 10^7$ (adding up to $\approx 13$ billion parameters). Figure 3 (right) demonstrates that scaling the number of memory slots yields a clear and consistent improvement. The curve follows a strict power law, indicating that Engram provides a predictable scaling knob.

---

## 4 Large Scale Pre-training

### 4.1 Experimental Setup
All models are pre-trained on 262 billion tokens with a 128k tokenizer.
*   **Dense-4B**: 4.1B total parameters, standard dense FFN.
*   **MoE-27B**: 26.7B total parameters, 72 routed + 2 shared experts (top-6).
*   **Engram-27B**: 26.7B total parameters, 55 routed experts, 5.7B-parameter Engram memory ($\rho = 74.3\%$).
*   **Engram-40B**: 39.5B total parameters, 18.5B-parameter Engram memory.

**Table 1: Pre-training performance comparison between dense, MoE, and Engram models.**

| Benchmark (Metric) | # Shots | Dense-4B | MoE-27B | Engram-27B | Engram-40B |
| :--- | :---: | :---: | :---: | :---: | :---: |
| # Total Params | | 4.1B | 26.7B | 26.7B | 39.5B |
| # Activated (w/o embed) | | 3.8B | 3.8B | 3.8B | 3.8B |
| # Trained Tokens | | 262B | 262B | 262B | 262B |
| # Experts (shared + routed, top-k) | | - | 2+72 (top-6) | 2+55 (top-6) | 2+55 (top-6) |
| # Engram Params | | - | - | 5.7B | 18.5B |
| **Language Modeling** | | | | | |
| Pile (loss) | - | 2.091 | 1.960 | **1.950** | 1.942 |
| Validation Set (loss) | - | 1.768 | 1.634 | **1.622** | 1.610 |
| **Knowledge & Reasoning** | | | | | |
| MMLU (Acc.) | 5-shot | 48.6 | 57.4 | **60.4** | 60.6 |
| MMLU-Redux (Acc.) | 5-shot | 50.7 | 60.6 | **64.0** | 64.5 |
| MMLU-Pro (Acc.) | 5-shot | 21.1 | 28.3 | **30.1** | 31.3 |
| CMMLU (Acc.) | 5-shot | 47.9 | 57.9 | **61.9** | 63.4 |
| C-Eval (Acc.) | 5-shot | 46.9 | 58.0 | **62.7** | 63.3 |
| AGIEval (Acc.) | 0-shot | 29.1 | 38.6 | **41.8** | 45.9 |
| ARC-Easy (Acc.) | 25-shot | 76.8 | 86.5 | **89.0** | 90.1 |
| ARC-Challenge (Acc.) | 25-shot | 59.3 | 70.1 | **73.8** | 76.4 |
| TriviaQA (EM) | 5-shot | 33.0 | 48.8 | **50.7** | 51.8 |
| CCPM (Acc.) | 0-shot | 72.2 | 79.6 | **87.1** | 87.7 |
| BBH (EM) | 3-shot | 42.8 | 50.9 | **55.9** | 57.5 |
| **Reading Comprehension** | | | | | |
| DROP (F1) | 1-shot | 41.6 | 55.7 | **59.0** | 60.7 |
| RACE-Middle (Acc.) | 5-shot | 72.4 | 80.9 | **82.8** | 83.3 |
| RACE-High (Acc.) | 5-shot | 66.0 | 75.4 | **78.2** | 79.2 |
| **Code & Math** | | | | | |
| HumanEval (Pass@1) | 0-shot | 26.8 | 37.8 | **40.8** | 38.4 |
| GSM8K (EM) | 8-shot | 35.5 | 58.4 | **60.6** | 62.6 |
| MATH (EM) | 4-shot | 15.2 | 28.3 | **30.7** | 30.6 |

---

## 5 Long Context Training

**Table 2: Long-context performance comparison (32k).**

| Model | LongPPL Book | LongPPL Paper | RULER MK | RULER MQ | RULER VT | RULER QA |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| MoE-27B (50k, 1.63) | 4.38 | 2.91 | 88.0 | 84.2 | 77.0 | 34.5 |
| Engram-27B (46k, 1.63) | 4.19 | 2.84 | 89.0 | **97.0** | 87.2 | 37.5 |
| Engram-27B (50k, 1.62) | **4.14** | **2.82** | **89.3** | **97.0** | **89.0** | **40.5** |

![Analysis of representational alignment and convergence speed](https://arxiv.org/html/2512.02556v1/x4.png)
*Figure 4: **Analysis of representational alignment and convergence speed.** (a) Layer-wise KL Divergence via LogitLens. The consistently lower divergence in early layers indicates that Engram accelerates prediction convergence. (b-c) Similarity heatmap computed by CKA. The distinct upward shift of the high-similarity diagonal demonstrates that Engram’s shallow layers are functionally equivalent to deeper layers of the MoE model.*

**Table 3: Entity resolution example reproduced from Ghandeharioun et al. (2024).**

| Layer | Latent State Translation | Explanation |
| :--- | :--- | :--- |
| 1-2 | : Country in the United Kingdom | **Wales** |
| 4 | : Title held by female sovereigns... | **Princess of Wales (unspecific)** |
| 6 | : Diana, Princess of Wales (1961-1997)... | **Diana, Princess of Wales** |

---

## 6 Analysis

### 6.1 Functional Depth
Engram reduces computational steps required for feature composition. As shown in Figure 4, Engram-27B exhibits systematically smaller KL divergence in early layers, reaching prediction readiness earlier. CKA analysis shows layer 5 of Engram aligns with layer 12 of MoE.

### 6.2 Structural Ablation
![Architecture ablation results](https://arxiv.org/html/2512.02556v1/x5.png)
*Figure 5: **Architecture ablation results.** (1) Layer Sensitivity: Layer 2 injection is optimal. (2) Component Ablation: multi-branch integration, tokenizer compression, and context-aware gating are critical.*

### 6.3 Sensitivity Analysis
Suppressing Engram output during inference causes a 71% drop in TriviaQA (factual knowledge) but only a 7% drop in C3 (reading comprehension).

![Retained performance under Engram ablation](https://arxiv.org/html/2512.02556v1/x6.png)
*Figure 6: **Retained performance under Engram ablation.** Factual knowledge relies heavily on the Engram module, whereas reading comprehension is largely preserved by the backbone.*

### 6.4 System Efficiency
**Table 4: End-to-end Inference Throughput.**

| Base Model | Configuration | Throughput (tok/s) |
| :--- | :--- | :---: |
| 4B-Dense | Baseline | 9,031.62 |
| 4B-Dense | + 100B Engram (CPU Offload) | 8,858.28 |
| 8B-Dense | Baseline | 6,315.52 |
| 8B-Dense | + 100B Engram (CPU Offload) | 6,140.02 |

### 6.5 Case Study: Gating Visualization
![Visualization of the gating mechanism of Engram](https://arxiv.org/html/2512.02556v1/x7.png)
*Figure 7: **Visualization of the gating mechanism of Engram.** The heatmap intensity corresponds to the magnitude of the gating scalar $\alpha_t \in [0,1]$. Stronger red indicates recognition of a static pattern ending at token $t$ (e.g., "Alexander the Great", "Milky Way").*

---

## 7 Related Work
We explore $N$-gram modeling, embedding scaling (FastText, SCONE, SuperBPE), Mixture-of-Experts (DeepSeek-MoE, GLaM), and memory networks (PKM, RETRO). Engram differs by enabling communication-computation overlap and being a first-class modeling primitive in an iso-FLOPs framework.

---

## 8 Conclusion
Engram introduces conditional memory as a scalable axis of sparsity. A hybrid MoE-Engram allocation yields a U-shaped scaling law, improving general reasoning and long-context capabilities. Its infrastructure-aware design allows offloading to host memory with $<3\%$ overhead, paving the way for next-generation sparse models.

---

## Appendix A: Detailed Model Architecture

**Table 5: Detailed model architecture information and training hyper parameters.**

| Parameter | Dense-4B | MoE-27B | Engram-27B | Engram-40B |
| :--- | :---: | :---: | :---: | :---: |
| Total Params | 4.1B | 26.7B | 26.7B | 39.5B |
| Active Params | 3.8B | 3.8B | 3.8B | 3.8B |
| Layers | 30 | 30 | 30 | 30 |
| Dimension | 2560 | 2560 | 2560 | 2560 |
| Routed Experts | - | 72 | 55 | 55 |
| Engram Layer | - | - | [2,15] | [2,15] |
| Engram $N$-gram | - | - | [2,3] | [2,3] |
| Engram Vocab | - | - | 2,262,400 | 7,239,680 |

---

## Appendix B: Full Benchmark Curves

![Last 10k pre-training benchmark curve](https://arxiv.org/html/2512.02556v1/x8.png)
*Figure 8: Last 10k pre-training benchmark curve.*

---

## Appendix C: Case Study of Tokenizer Compression

**Table 6: Top-5 merged tokens by Tokenizer Compression.**

| Rank | Merge Count | Normalized Token | Original Tokens |
| :---: | :---: | :---: | :--- |
| 1 | 163 | '␣' | '\t', '\n', '\r', '␣', '␣␣', '\n\n', ... |
| 2 | 54 | 'a' | 'A', 'a', '␣a', '␣A', 'á', 'ä', 'ã', ... |
| 3 | 40 | 'o' | 'O', 'o', '␣o', '␣O', 'ó', 'ö', 'ô', ... |
| 4 | 35 | 'e' | 'E', 'e', '␣e', '␣E', 'é', 'è', '␣é', ... |
| 5 | 30 | 'i' | 'I', 'i', '␣I', '␣i', 'í', 'ì', 'î', ... | -->