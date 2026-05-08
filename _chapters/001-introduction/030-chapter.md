---
slug: ch03-book-structure-and-scope
title: "1.3 Book structure and scope"
layout: chapter
---

We have organized the book into a clear, four-stage roadmap. This structure is designed to be progressive, with each stage building directly upon the knowledge and code from the previous one. We will begin with the essential building blocks of modern LLM inference, move through the core architectural innovations of DeepSeek, and then explore the advanced training and post-training techniques that give the model its power.

Figure 1.9 provides a high-level overview of this entire process. It visualizes the four distinct stages and lists the key technical concepts we will implement within each one. Think of this as the master blueprint for our project and the table of contents for your learning journey.

### **Figure 1.9 The four-stage roadmap for building a mini-DeepSeek model in this book. We will progress from foundational concepts (Stage 1) and core architecture (Stage 2) to advanced training (Stage 3) and post-training techniques (Stage 4), implementing each key innovation along the way.**

![](https://drek4537l1klr.cloudfront.net/dandekar/v-1/Figures/01__image009.png)

Stages 1 and 2 are related to the architectural innovations in DeepSeek. Stage 3 is related to the training pipeline, and Stage 4 is related to the post-training pipeline.

In stage 1, we will understand what is meant by the Key-Value Cache (KV Cache) and why it is the foundational building block for ultimately understanding multi-head latent attention (MLA), which is one of the key innovations in the DeepSeek architecture.

In stage 2, we will look at multi-head latent attention (MLA) and mixture of experts (MoE). We will visually understand how MLA and MoE work and also code them in practice.

In Stage 3, we will implement the DeepSeek training pipeline. In this stage, we will learn about:

1. Multi-token prediction (MTP)
2. FP8 quantization
3. Dual pipe parallelism

Finally, in stage 4, we will look at post-training techniques implemented by DeepSeek:

1. Supervised Fine-tuning
2. Reinforcement Learning (RL)
3. Model distillation