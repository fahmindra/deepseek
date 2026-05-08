---
slug: ch05-what-you-will-need-to-follow-along
title: "1.5 What you will need to follow along"
layout: chapter
---

Following along with this book requires a foundation in machine learning concepts, but not expertise. If you are comfortable with Python and have worked through introductory deep learning materials, you are ready to begin. Specifically, you should understand how neural networks learn through backpropagation, be familiar with the basic operations in PyTorch or a similar framework, and have some exposure to the transformer architecture, even if you haven't implemented one yourself.

In terms of hardware, we have designed all the implementations to be accessible. While training large language models typically requires massive computational resources, we will work with scaled-down versions that capture the essential ideas while remaining tractable on consumer hardware.

A laptop with a decent CPU can run most of the examples, though they will train slowly. A single consumer GPU with 8-12GB of VRAM will make the experience much smoother, allowing you to experiment more freely and see results faster. For the more ambitious experiments, particularly when working with the MoE architectures, having 24-48GB of VRAM opens up additional possibilities, though it's not required.

We will provide complete environment specifications for each chapter, ensuring you can reproduce our results exactly. We will include configurations for Google Colab and similar platforms, making it possible to follow along without any local setup. While DeepSeek was trained on trillions of tokens, we will use smaller datasets that still allow us to observe the key phenomena while keeping training times reasonable.

Let us build something amazing together!