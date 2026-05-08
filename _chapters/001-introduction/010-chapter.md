---
slug: ch01-why-deepseek-a-turning-point-in-open-source-ai
title: "1.1 Why DeepSeek? A turning point in open-source AI"
layout: chapter
---

With so many language models out there, you might wonder: why focus on DeepSeek? What makes this model so special that we decided to write an entire book about building it? The short answer is that DeepSeek represents a turning point in open-source AI, demonstrating for the first time that an openly available model can rival the performance of the best proprietary models.

Figure 1.1 shows a simple interaction with the DeepSeek chat interface.

### **Figure 1.1 A simple interaction with the DeepSeek chat interface.**

![](https://drek4537l1klr.cloudfront.net/dandekar/v-1/Figures/01__image001.png)

Let us step back for a moment and consider the state of AI before DeepSeek arrived. In the early 2020s, large language models were dominated by a few tech giants and research labs with vast resources. OpenAI’s GPT series, Google’s models like PaLM, and similar closed-source models led the field in capability, but they were (and remain) proprietary, often accessible only via paid APIs. The open-source community had successes like BERT and smaller GPT-style models, but there was a gap in performance. Open models tended to lag a generation behind closed models. Then, around 2023-2024, we saw a shift: Meta released LLaMA and later LLaMA-2 openly to researchers, and other organizations started emphasizing open science in AI.

DeepSeek emerged in this context, but it took openness to a new level, both in terms of releasing weights freely to the public and in terms of pushing technical boundaries. DeepSeek was founded in 2023 (as an AI lab based in China, led by researcher Liang Wenfeng), and within a short span, it made waves by open-sourcing extremely large LLMs with performance on par with the very best.

The significance of DeepSeek became clear with the release of its first major model, often referred to as DeepSeek-R1. A screenshot of the paper that introduced DeepSeek-R1 is shown in figure 1.2 (https://arxiv.org/pdf/2501.12948).

### **Figure 1.2 The title and abstract of the DeepSeek-R1 research paper.**

![](https://drek4537l1klr.cloudfront.net/dandekar/v-1/Figures/01__image002.png)

This model immediately stunned the AI community. Despite being openly available, R1 demonstrated an intelligence level that rivaled top-tier models from giants like OpenAI and Google, effectively narrowing the gap between open-source and closed-source AI to the smallest it had ever been. At the time of its release in early 2025, it was on par with or outperformed leading models like OpenAI's o1-1217 across a range of demanding reasoning benchmarks, including mathematical problem-solving (AIME 2024) and competitive coding (Codeforces).

This achievement effectively narrowed the gap between open-source and closed-source AI to the smallest it had ever been.

Another shocking announcement was that DeepSeek was trained at a fraction of the cost of leading OpenAI models. For us as learners and builders, DeepSeek-R1 is a perfect case study because its success comes from technical breakthroughs that we can understand. This raises several key questions that we will answer throughout this book:

- How did DeepSeek-R1 achieve state-of-the-art results with a fraction of the training cost?
- What was so novel in the DeepSeek-R1 architecture?
- What was so novel in the DeepSeek-R1 pre-training and post-training?

You will learn answers to all the above questions in the subsequent chapters. DeepSeek’s team openly published many of their methods, and we will leverage those insights in this book. By reproducing key elements of DeepSeek, we get to explore state-of-the-art techniques in practice. This includes the topics we previewed earlier: new forms of attention, new training objectives, massive model scaling strategies, and novel ways to compress models.

From a historical perspective, we can say that DeepSeek marked the moment when open-source AI truly went head-to-head with the tech giants and held its own. By replicating parts of DeepSeek, we effectively retrace the steps of some of the most advanced AI research of the day. This is immensely valuable if you aspire to work in AI research.

Finally, there is a philosophical reason: DeepSeek embodies the spirit of democratization of AI. Knowledge that was once confined is now shared. In writing this book, we align with that spirit. The authors of DeepSeek published technical reports detailing their methods, and we take it a step further by translating those into an approachable tutorial.

When you build something yourself, you own that knowledge in a way that reading a paper or using an API can’t provide. The hope is that by empowering more people to understand and build advanced models, we accelerate innovation and broaden the base of who can contribute to AI. Today it’s DeepSeek. Tomorrow, perhaps it will be you inventing the next big idea after having learned from this experience.