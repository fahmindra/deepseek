---
slug: ch02-the-key-innovations-we-will-build
title: "1.2 The key innovations we will build"
layout: chapter
---

Now, let’s lay out the architectural roadmap for building DeepSeek from scratch, identifying the key innovations that differentiate DeepSeek from a standard Transformer-based language model. Understanding these specific components is crucial, as they are targeted solutions to fundamental bottlenecks in scaling language models. Each innovation from Multi-Head Latent Attention to Mixture-of-Experts addresses a distinct challenge related to computational complexity, memory bandwidth, or parameter scaling. By deconstructing and implementing these systems, we gain a first-principles understanding of the engineering trade-offs involved in modern LLM design. Our approach in this book will be to treat each innovation as a case study, first analyzing the limitations of the standard approach and then building its advanced replacement from the ground up

**1.2.1 Architecture**

DeepSeek's architecture builds upon the well-established Transformer foundation that powers models like GPT-3 and ChatGPT. However, it introduces significant innovations to overcome key performance bottlenecks. To understand what makes DeepSeek unique, we must first look at the standard Transformer building block.

The core of most modern LLMs is a stack of identical layers. Each layer consists of two primary sub-components: a multi-head self-attention mechanism, which allows the model to weigh the importance of different tokens in the input, and a feed-forward neural network, which processes the information further. Figure 1.4 shows a detailed view of this standard architecture..

### **Figure 1.3 A detailed view of a standard Transformer block, the foundational architecture used in models like LLaMA and the GPT series. It is composed of a multi-head attention block and a feed-forward network (NN).**

![](https://drek4537l1klr.cloudfront.net/dandekar/v-1/Figures/01__image003.png)

DeepSeek's key architectural innovation lies in replacing both of these standard sub-components with more efficient and powerful alternatives. As illustrated in Figure 1.4, the standard multi-head attention is replaced by **Multi-Head Latent Attention (MLA)**, and the feed-forward network is replaced by a **DeepSeek-Mixture-of-Experts (MoE)** structure.

### **Figure 1.4 A simplified view of the DeepSeek model architecture. It modifies the standard Transformer by replacing the core components with Multi-Head Latent Attention (MLA) and a Mixture-of-Experts (MoE) layer. This design also utilizes RMS Norm (Root Mean Square Normalization) and a specialized Decoupled RoPE (Rotary Position Embedding).**

![](https://drek4537l1klr.cloudfront.net/dandekar/v-1/Figures/01__image004.png)

These two architectural changes, MLA and DeepSeek-MoE, are targeted solutions to major challenges in scaling LLMs.

On top of these architectural changes, DeepSeek introduces advanced training and efficiency techniques. A new training objective called Multi-Token Prediction (MTP) improves learning and inference speed, while FP8 quantization (an 8-bit floating-point format) addresses computational efficiency and resource utilization.

Together, these four innovations MLA, MoE, MTP, and FP8 quantization form the pillars of DeepSeek’s technical advancement. Each one is a targeted solution to a different fundamental challenge in scaling language models:quantization for inference) to push efficiency to the limit..

Each of these addresses a different problem area:

- MLA tackles the *speed and memory bottleneck* in attention for long sequences.
- MoE tackles the scaling and model capacity issue
- MTP improves *learning and inference speed* by predicting more than one token at a time
- FP8 quantization addresses the computational efficiency and resource utilization problem.

**1.2.2 Training**

Beyond the core architecture, DeepSeek also innovates in how the model is trained and refined. The training pipeline is carefully designed to make large-scale training as efficient as possible. For example, DeepSeek employs an optimized scheduling strategy (internally nicknamed DualPipe) that overlaps different training tasks to keep hardware utilization high. In practice, this means data loading, preprocessing, and neural network computations are coordinated so that the GPU is never idle. When one batch is being processed by the model, the next batch is being prepared in parallel.

### **Figure 1.5 An illustration of the DualPipe training pipeline on a single device. By overlapping the forward pass (the initial blocks), backward pass (the hatched blocks), and combined computations, this scheduling strategy minimizes GPU idle time and maximizes hardware utilization during large-scale training.**

![](https://drek4537l1klr.cloudfront.net/dandekar/v-1/Figures/01__image005.png)

Figure 1.5 shows a timeline of Device 1 running tasks in a dual-pipe pipeline. The process begins with the forward pass, followed by the backward pass. Crucially, the pipeline then enters a steady state where the forward pass of a new batch is performed concurrently with the backward pass of the previous one (represented by the cross-hatched blocks). This overlapping ensures the device remains fully utilized throughout the training process.

**1.2.3 Post-training**

The base model, which was trained by DeepSeek, was called DeepSeek-v3. DeepSeek-v3 went through several post-training steps, ultimately resulting in DeepSeek-R1. These steps are shown in figure 1.6.

### **Figure 1.6 The multi-step post-training pipeline used to create DeepSeek-R1 from the DeepSeek-V3 base model. This process involves a combination of reinforcement learning (Pure RL), data generation (Rejection sampling), and fine-tuning to instill advanced reasoning capabilities.**

![](https://drek4537l1klr.cloudfront.net/dandekar/v-1/Figures/01__image006.png)

Below is a very simplified explanation of the five steps involved in DeepSeek R1 post-training:

- **Step 1 (Foundation):** Start with a lightly fine-tuned base (DeepSeek-V3) using a relatively small “cold-start” dataset.
- **Step 2 (Pure RL):** Implement reinforcement learning algorithms to allow the model to explore and develop reasoning patterns through trial-and-error learning. This unsupervised approach enabled the model to discover effective problem-solving strategies without explicit human guidance.
- **Step 3 (Self-Labeling):** Introduce rejection sampling techniques. The model generated multiple candidate responses and selected the highest-quality outputs to create its own synthetic training data.
- **Step 4 (Blending Data):** Merge this synthetic data with supervised examples to balance quality with domain breadth.
- **Step 5 (Final RL):** The training concluded with a comprehensive reinforcement learning phase using diverse prompt distributions. This final optimization step enhanced the model's robustness and generalization across various task categories and input formats.

Figure 1.7 is taken from the DeepSeek-R1 paper, which was released in January 2025. Here we can see that DeepSeek-R1 is on par with or performs better than OpenAI reasoning models on several benchmarks that were tested.

### **Figure 1.7 Benchmark performance of DeepSeek-R1 against other leading models (as of January 2025).**

![](https://drek4537l1klr.cloudfront.net/dandekar/v-1/Figures/01__image007.png)

Another important post-training technique is knowledge distillation and model compression. The idea is to take the large “teacher” model (the full DeepSeek) and compress its knowledge into one or several smaller, more practical “student” models. This has been shown in figure 1.8.

### **Figure 1.8 The concept of knowledge distillation. A large, powerful "teacher" model (like DeepSeek-R1) is used to generate training data to teach a much smaller, more efficient "student" model, transferring its capabilities without the high computational cost.**

![](https://drek4537l1klr.cloudfront.net/dandekar/v-1/Figures/01__image008.png)

The DeepSeek R1 model was based on DeepSeek v3, which had around 671 billion parameters. When the DeepSeek paper was released, the team also released distilled models that were as low as 1.5 billion parameters in size. In particular, DeepSeek open-sourced 1.5B, 7B, 8B, 14B, 32B, and 70B checkpoints based on Qwen2.5 and Llama3 series to the community. These smaller models are highly performant and efficient.

In summary, DeepSeek’s roadmap consists of innovations at multiple levels:

- Novel architectural components (MLA and MoE)
- A smarter training objective (MTP) with cutting-edge precision techniques (FP8)
- An efficient large-scale training pipeline (overlapping computations and communications, etc.)
- Post-training (RL-based reasoning skills and model distillation).

In the coming chapters, we will build each of these components step by step and demonstrate how they come together into a cohesive *mini-DeepSeek* model.