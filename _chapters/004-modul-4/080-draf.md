---
slug: modul-4-8
title: modul-4-8
---
### Penutup
### Modul 4 Training LLM: From-Scratch, Fine-Tuning, atau Continued Pretraining?
---

![Ikon Modul 4](https://lms.sdmdigital.id/pluginfile.php/635385/mod_page/content/11/_6c05842c-b6bc-4934-893f-691bd4692fb6-removebg-preview.png)
**Gambar: Ikon Modul 4**

Ringkasan poin utama adalah sebagai berikut:

*   Proses training LLM memerlukan investasi yang signifikan untuk memperoleh data maupun menyediakan *compute* yang diperlukan.
*   *Scaling laws* menunjukkan bahwa ada fungsi tertentu yang bisa digunakan untuk melakukan estimasi peningkatan performa berdasarkan peningkatan resource. Selain itu, Scaling laws juga dapat digunakan untuk menentukan ukuran model dan jumlah data yang tepat untuk memaksimalkan compute dengan suatu target performa tertentu.
*   Tahap melatih LLM terbagi dua, yaitu *pretraining* dan *post training*. Pretraining bertujuan mempelajari struktur dan konsep bahasa, sementara post training bertujuan mempertajam kemampuan *base model* hasil pretraining dalam berbagai aspek seperti kemampuan mengikuti instruksi dan keselarasannya dengan preferensi manusia.
*   Distilasi merupakan salah satu cara untuk melakukan kompresi model, dengan cara menggunakan informasi yang berasal dari suatu model yang disebut *teacher* untuk membantu melatih model tujuan yang biasanya disebut *student*.
*   LLM bisa diadaptasikan agar memiliki perilaku tertentu menggunakan instruksi dan konteks, melakukan *fine tuning*, maupun melakukan *continued pretraining*.
*   Pertimbangan yang matang perlu dipikirkan sebelum membuang *resource* untuk melakukan sesuatu yang tidak perlu dalam menentukan konfigurasi LLM. Mulai dari memahami tujuan dan menurunkan kriteria dan evaluasi, lalu mulailah dari pilihan paling pragmatis dulu, yaitu menggunakan LLM yang tersedia, secara bertahap dinilai apakah perlu berpindah ke langkah adaptasi selanjutnya atau tidak.

> **Refleksi Akhir**
>
> *   Digiers telah mempelajari berbagai macam hal mengenai melatih LLM. Mulai dari kebutuhan data dan compute, scaling law, tahapan melatih LLM seperti pretraining dan post training, salah satu metode kompresi LLM yang dilakukan dalam proses latih yaitu knowledge distillation, hingga berbagai macam pendekatan dalam mengembangkan LLM serta bagaimana cara memilih pendekatan mana yang harus dilakukan.
> *   Setelah lebih memahami mengenai berbagai langkah pengembangan LLM, mungkin anda bertanya-tanya use case seperti apa yang cukup unik menjadi justifikasi melatih sebuah model dari nol. Dalam industri sendiri terdapat berbagai macam use case yang menarik. Salah satunya adalah sebuah [LLM yang dikembangkan oleh seorang research scientist bernama Eugene Yan](https://eugeneyan.com/writing/semantic-ids/) yang merupakan hibrida dengan sebuah recommendation system. Coba pelajari
> *   Bagaimana cara kerjanya?

---

### Daftar Pustaka

1.  Wei, J., et al. (2023). A Survey of Large Language Models. arXiv:2303.18223. [https://arxiv.org/abs/2303.18223](https://arxiv.org/abs/2303.18223)
2.  Dynomight. (n.d.). First Principles of AI Scaling. [https://dynomight.net/scaling/](https://dynomight.net/scaling/)
3.  OpenAI. (n.d.). What are tokens and how to count them. OpenAI Help Center. [https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them](https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them)
4.  Zheng, L., et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. arXiv:2306.05685. [https://arxiv.org/abs/2306.05685](https://arxiv.org/abs/2306.05685)
5.  Villalobos, P., et al. (2024). Position: Will we run out of data? Limits of LLM scaling based on human-generated data. In Proceedings of Machine Learning Research, Vol. 235. [https://proceedings.mlr.press/v235/villalobos24a.html](https://proceedings.mlr.press/v235/villalobos24a.html)
6.  Wei, J., et al. (2022). Emergent Abilities of Large Language Models. arXiv:2206.07682. [https://arxiv.org/abs/2206.07682](https://arxiv.org/abs/2206.07682)
7.  Kaplan, J., et al. (2020). Scaling Laws for Neural Language Models. arXiv:2001.08361. [https://arxiv.org/abs/2001.08361](https://arxiv.org/abs/2001.08361)
8.  Hoffmann, J., et al. (2022). Training Compute-Optimal Large Language Models. arXiv:2203.15556. [https://arxiv.org/abs/2203.15556](https://arxiv.org/abs/2203.15556)
9.  Nakkiran, P., et al. (2024). Reconciling Kaplan and Chinchilla Scaling Laws. arXiv:2406.12907. [https://arxiv.org/abs/2406.12907](https://arxiv.org/abs/2406.12907)
10. Touvron, H., et al. (2023). LLaMA: Open and Efficient Foundation Language Models. arXiv:2302.13971. [https://arxiv.org/abs/2302.13971](https://arxiv.org/abs/2302.13971)
11. Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the Knowledge in a Neural Network. arXiv:1503.02531. [https://arxiv.org/abs/1503.02531](https://arxiv.org/abs/1503.02531)
12. Vincent, J. (2023, December 6). Google’s Gemini isn’t the generative AI model we expected. TechCrunch. [https://techcrunch.com/2023/12/06/googles-gemini-isnt-the-generative-ai-model-we-expected](https://techcrunch.com/2023/12/06/googles-gemini-isnt-the-generative-ai-model-we-expected)
13. DeepSeek AI. (n.d.). DeepSeek-R1-Distill-Qwen-7B. Hugging Face. [https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)
14. LMSYS Org. (2023, March 30). Vicuna: An open-source chatbot impressing GPT-4 with 90% ChatGPT quality. [https://lmsys.org/blog/2023-03-30-vicuna/](https://lmsys.org/blog/2023-03-30-vicuna/)
15. Stanford CRFM. (2023, March 13). Alpaca: A strong, replicable instruction-following model. [https://crfm.stanford.edu/2023/03/13/alpaca.html](https://crfm.stanford.edu/2023/03/13/alpaca.html)
16. Yan, Ziyou. (Sep 2025). Training an LLM-RecSys Hybrid for Steerable Recs with Semantic IDs. eugeneyan.com. [https://eugeneyan.com/writing/semantic-ids/](https://eugeneyan.com/writing/semantic-ids/).