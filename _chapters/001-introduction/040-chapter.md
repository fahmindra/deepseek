---
slug: ch04-what-this-book-will-teach-you-and-what-it-wont
title: "1.4 What this book will teach you and what it won’t"
layout: chapter
---

This book is designed as a hands-on journey through the architectural innovations that make DeepSeek possible. We believe that the best way to understand these complex systems is to build them yourself.

What you will learn in this book spans both theoretical understanding and practical implementation.

- You will discover how Multi-Head Latent Attention (MLA) dramatically reduces memory requirements while maintaining model quality, implementing the mechanism that allows DeepSeek to run efficiently on hardware that would struggle with traditional transformers.
- You will master the intricacies of Mixture-of-Experts (MoE) architectures, understanding how to route different tokens to specialized sub-networks and balance computational load across experts.
- Through Multi-Token Prediction (MTP), you'll see how predicting multiple future tokens simultaneously can accelerate both training and inference.
- And with FP8 quantization, you'll learn to compress model weights and activations to just eight bits while preserving the model's capabilities.
- You will also learn about how DeepSeek pre-trained their model.
- Finally, you will learn the details regarding the post-training techniques that DeepSeek implemented, which include Reinforcement Learning (RL) and Distillation.

What this book won't do is reproduce DeepSeek's proprietary training data or attempt to replicate their exact model weights. We will not dive into the massive-scale distributed training infrastructure required to train models with hundreds of billions of parameters—that would require resources beyond what most readers have access to. We also will not cover production deployment concerns like serving models to millions of users or implementing safety filters and content moderation systems.

Instead, we focus on clarity and comprehension. Every concept is introduced with minimal assumptions about prior knowledge, building up from first principles. For example, when we implement MLA, we will start with standard attention, understand its limitations, and then gradually transform it into the latent version. This approach means that you will not only know how to implement these techniques but also understand them deeply enough to modify and improve upon them.