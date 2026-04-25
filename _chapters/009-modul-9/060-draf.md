---
slug: modul-9-6
title: modul-9-6
---
Berikut adalah hasil perapian konten ke dalam format Markdown sesuai dengan hirarki dan instruksi Anda:

# Analisis Konseptual: Evaluasi dan Perencanaan Arsitektur LLM

---

**Tujuan Praktikum:**
Peserta mampu menganalisis dan menjelaskan strategi adaptasi Large Language Model (LLM) berdasarkan kebutuhan penggunaan, mencakup perbedaan pendekatan Supervised Fine-Tuning (SFT), Continual Pretraining (CPT), Retrieval-Augmented Generation (RAG), serta teknik efisiensi seperti LoRA dan Model Merging.

**Deskripsi Singkat Aktivitas:**
Aktivitas ini meminta Anda bertindak sebagai **AI Engineer** yang mengevaluasi beberapa skenario penerapan LLM di lingkungan perusahaan. Anda akan menganalisis kebutuhan bisnis, mengidentifikasi risiko teknis seperti *Catastrophic Forgetting*, serta memilih arsitektur dan strategi adaptasi model yang paling tepat dan efisien.

---

## Langkah-Langkah Praktikum

### 1. Siapkan Dokumen Jawaban
*   Buat sebuah file teks (misalnya menggunakan Notepad, Google Docs, atau editor Markdown) untuk menuliskan jawaban Anda.
*   Beri judul file: **“Unjuk Kerja Modul – Strategi Adaptasi LLM – [Nama Anda]”**.
*   Pastikan seluruh jawaban ditulis dalam satu dokumen yang terstruktur.

### 2. Analisis Strategi Adaptasi Model (Konsep: SFT vs CPT vs RAG)

**Skenario:**
Sebuah perusahaan ingin menggunakan LLM untuk menjawab pertanyaan berbasis 30.000 dokumen internal (kebijakan, laporan, dan kontrak). Engineer junior mengusulkan untuk mengubah seluruh dokumen menjadi format tanya–jawab dan melakukan **Supervised Fine-Tuning (SFT)**.

**Pertanyaan:**
*   **a.** Tuliskan jawaban singkat (1–2 poin) yang menjelaskan mengapa pendekatan SFT kurang efektif untuk tujuan penguasaan pengetahuan dokumen secara luas.
*   **b.** Tuliskan jawaban singkat (1–2 poin) mengenai konsekuensi teknis dan biaya jika perusahaan memilih **Continual Pretraining (CPT)**.
*   **c.** Sebutkan satu strategi alternatif non-training yang lebih efisien dan jelaskan alasannya secara singkat.

### 3. Analisis Risiko Adaptasi Model (Konsep: Catastrophic Forgetting)

**Skenario:**
Sebuah model LLM dilatih ulang sepenuhnya menggunakan data domain medis. Setelah proses pelatihan, model menjadi sangat baik dalam menjawab pertanyaan medis, namun kehilangan kemampuan menghasilkan teks umum yang koheren.

**Pertanyaan:**
*   **a.** Identifikasi fenomena teknis yang terjadi pada model tersebut.
*   **b.** Jelaskan penyebabnya secara singkat menggunakan konsep **distribution shift**.
*   **c.** Usulkan satu strategi mitigasi berbasis **data mixing** atau **replay**.

### 4. Analisis Efisiensi Training (Konsep: Adapter-Based / LoRA)

**Skenario:**
Anda hanya memiliki satu GPU consumer-grade dengan VRAM terbatas, namun diminta melakukan adaptasi domain pada model LLM berukuran besar.

**Pertanyaan:**
*   **a.** Jelaskan mengapa fine-tuning penuh (full CPT) tidak realistis dilakukan pada kondisi perangkat keras tersebut.
*   **b.** Jelaskan secara konseptual bagaimana **LoRA** memungkinkan proses adaptasi tetap berjalan.
*   **c.** Apa yang terjadi pada bobot asli model selama proses training berbasis LoRA?

### 5. Analisis Integrasi Kemampuan Model (Konsep: Model Merging)

**Skenario:**
Tersedia dua model open-source berbasis arsitektur yang sama:
*   Model A unggul dalam logika dan matematika.
*   Model B unggul dalam bahasa Indonesia.

**Pertanyaan:**
Jelaskan secara konseptual bagaimana **Model Merging** dapat digunakan untuk menggabungkan kedua kemampuan tersebut tanpa melakukan proses training ulang.

---

## Tools & Resource yang Dibutuhkan

*   **Akses Materi Modul 2.2**, khususnya:
    *   Tabel perbandingan SFT vs CPT (Subtopik 2.2.1)
    *   Diagram alur Continual Pretraining (CPT) (Subtopik 2.2.4)
    *   Konsep Adapter-Based Training (LoRA) (Subtopik 2.2.5)
*   **Aplikasi Pengolah Kata**: Microsoft Word, Google Docs, atau Text Editor (Notepad / VS Code) untuk menyusun jawaban analisis.
*   **Koneksi Internet (Opsional)**: Digunakan untuk mencari referensi tambahan, seperti spesifikasi GPU NVIDIA RTX 4090 atau informasi teknis model Llama-3, jika diperlukan sebagai justifikasi teknis dalam jawaban.

---

## Contoh Output atau Hasil Akhir
Satu file dokumen (`.md`, `.txt`, atau `.pdf`) berisi jawaban terstruktur sesuai nomor langkah dan sub-pertanyaan (misal: 2a, 2b, 2c, 3a, 3b). Jawaban harus ringkas, teknis, dan konsisten dengan konsep yang dipelajari.

---

## Kriteria Keberhasilan / Penilaian Mandiri
*   Mampu membedakan kapan menggunakan SFT, CPT, atau RAG.
*   Mampu menjelaskan penyebab dan mitigasi Catastrophic Forgetting.
*   Mampu menjelaskan prinsip efisiensi LoRA secara konseptual.
*   Mampu memahami konsep dasar Model Merging.

---

## Petunjuk Troubleshooting

*   **Skenario 1 (SFT vs CPT):** Rujuk **Subtopik 2.2.1**. Ingat prinsip utama: **SFT untuk perilaku, CPT untuk pengetahuan**. Evaluasi apakah tujuan skenario membutuhkan perubahan perilaku model atau penambahan pengetahuan faktual.
*   **Skenario 2 (Catastrophic Forgetting):** Lihat **Subtopik 2.2.3**. Fokus pada konsep **Distribution Shift**—model yang dilatih ulang penuh pada domain baru berisiko kehilangan kemampuan lama.
*   **Skenario 3 (LoRA vs Full CPT):** Rujuk **Subtopik 2.2.2 & 2.2.5**. Bandingkan kebutuhan VRAM full training (bobot + gradien + optimizer) dengan LoRA yang hanya menambah <1% parameter.
*   **Skenario 4 (Model Merging):** Rujuk **Subtopik 2.2.7**. Ingat bahwa parameter model adalah vektor yang dapat digabungkan secara matematis tanpa training ulang.

---

**Durasi Praktikum Disarankan:** 60 menit - 90 menit

**Aksi Lanjutan atau Refleksi (Opsional):**
Setelah menyelesaikan self practice ini, refleksikan kembali pilihan strategi adaptasi LLM yang telah dianalisis. Bayangkan Anda terlibat dalam pengambilan keputusan teknis pada sebuah tim pengembang AI:

Menurut Anda, dalam kondisi seperti apa **adaptasi berbasis training** (SFT atau CPT) menjadi relevan, dan kapan pendekatan **tanpa atau minim training** (seperti RAG atau LoRA) lebih tepat digunakan? Jelaskan pertimbangan Anda secara singkat dengan menyoroti aspek **tujuan penggunaan, keterbatasan sumber daya, dan risiko teknis**.

[Link Hands-On: Implementasi Strategi Adaptasi Large Language Model](https://github.com/Modul-LLM/Module-09-2.5-Intermediate)

---

# Penutup
## Modul 5 Arsitektur, Implementasi, dan Evaluasi Sistem RAG

Melalui modul ini, Anda sudah belajar bagaimana meningkatkan kapabilitas LLM menjadi sistem cerdas yang memiliki akses ke basis pengetahuan eksternal yang dinamis dan terverifikasi. Anda sudah mengetahui bahwa membangun sistem RAG (*Retrieval-Augmented Generation*) tidak sekadar menghubungkan database dengan LLM, tetapi memerlukan perancangan pipeline data yang hati-hati, strategi pencarian yang tepat, dan metode evaluasi yang objektif.

### Dalam modul ini, kita telah mempelajari mengenai:
*   **Konsep dasar RAG** sebagai solusi mengatasi keterbatasan pengetahuan internal LLM dan meminimalisir halusinasi dengan menyuntikkan konteks faktual secara dinamis.
*   **Mekanisme Retrieval** yang mencakup perbedaan antara pencarian berbasis kata kunci (Sparse/BM25) dan pencarian berbasis makna (Dense/Vector Search), serta pendekatan Hybrid untuk hasil terbaik.
*   **Proses Ingestion**, mulai dari parsing dokumen berbagai format, normalisasi data, hingga strategi chunking (Fixed-length, Semantic, Hierarchical) yang sangat mempengaruhi kualitas informasi yang disimpan.
*   **Penyimpanan dan Pengindeksan** data ke dalam Vector Database.
*   **Metodologi Evaluasi RAG** menggunakan metrik terukur seperti Context Recall, Context Precision, Faithfulness, dan Answer Relevance, seringkali dengan bantuan LLM-as-a-judge.
*   **Teknik Optimasi dan Troubleshooting** seperti penggunaan Re-ranking, Query Expansion, hingga pemanfaatan metadata untuk meningkatkan akurasi dan efisiensi sistem.

### Dengan menyelesaikan materi ini, Anda telah berhasil mencapai tujuan pembelajaran berikut:
*   Anda kini sudah mampu merancang arsitektur sistem RAG end-to-end yang efektif dan efisien.
*   Anda kini sudah mengerti bagaimana memilih strategi chunking dan retrieval yang paling sesuai dengan karakteristik dokumen dan kebutuhan pengguna.
*   Anda kini mampu mengimplementasikan Vector Database dan memahami trade-off antara berbagai metode indexing untuk performa sistem.
*   Anda mampu melakukan evaluasi objektif terhadap sistem RAG untuk mendiagnosa apakah masalah terletak pada proses pencarian dokumen (retrieval) atau pada kemampuan menjawab (generation).
*   Anda memiliki bekal teknis untuk melakukan debugging dan optimasi sistem ketika menghadapi masalah nyata seperti latency tinggi atau jawaban yang tidak relevan.

> **Refleksi Akhir**
>
> Pada titik ini, Anda mungkin menyadari bahwa kualitas jawaban sebuah sistem RAG sangat bergantung pada prinsip "Garbage In, Garbage Out". Sehebat apapun LLM yang digunakan, jika proses ingestion buruk atau strategi retrieval tidak tepat, hasil akhirnya akan mengecewakan.
>
> Kunci keberhasilan dalam membangun aplikasi RAG terletak pada detail pemrosesan data. Sebagai seorang AI Engineer, tugas Anda bukan hanya menyusun prompt tapi juga merancang cara untuk menghubungkan pertanyaan ini dengan sebuah sumber pengetahuan. Kita tidak perlu sistem yang hanya cepat menjawab, tetapi sistem yang jawabannya dapat dipercaya, dapat dilacak sumbernya, dan relevan dengan kebutuhan pengguna.

---

## Daftar Referensi

1.  Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W.-t., Rocktäschel, T., Riedel, S., & Kiela, D. *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. arXiv:2005.11401 (2020). [https://arxiv.org/abs/2005.11401](https://arxiv.org/abs/2005.11401)
2.  LangChain. *LangChain Documentation*. [https://docs.langchain.com/](https://docs.langchain.com/)
3.  Confident AI. *RAG Evaluation Metrics: Answer Relevancy, Faithfulness, and More*. [https://www.confident-ai.com/blog/rag-evaluation-metrics-answer-relevancy-faithfulness-and-more](https://www.confident-ai.com/blog/rag-evaluation-metrics-answer-relevancy-faithfulness-and-more)
4.  Galileo. *RAG Performance Optimization*. [https://galileo.ai/blog/rag-performance-optimization](https://galileo.ai/blog/rag-performance-optimization)
5.  Milvus. *How to Enhance Your RAG*. [https://milvus.io/docs/how_to_enhance_your_rag.md](https://milvus.io/docs/how_to_enhance_your_rag.md)
6.  Databricks. *Quality Data Pipeline for RAG*. [https://docs.databricks.com/aws/en/generative-ai/tutorials/ai-cookbook/quality-data-pipeline-rag](https://docs.databricks.com/aws/en/generative-ai/tutorials/ai-cookbook/quality-data-pipeline-rag)
7.  Pinecone. *Vector Database Guide*. [https://www.pinecone.io/learn/vector-database/](https://www.pinecone.io/learn/vector-database/)
8.  Pinecone. *A Developer’s Guide to ANN Algorithms*. [https://www.pinecone.io/learn/a-developers-guide-to-ann-algorithms/](https://www.pinecone.io/learn/a-developers-guide-to-ann-algorithms/)
9.  Elastic. *Understanding Approximate Nearest Neighbor (ANN)*. [https://www.elastic.co/blog/understanding-ann](https://www.elastic.co/blog/understanding-ann)
10. Zilliz. *Mastering BM25: A Deep Dive into the Algorithm and Application in Milvus*. [https://zilliz.com/learn/mastering-bm25-a-deep-dive-into-the-algorithm-and-application-in-milvus](https://zilliz.com/learn/mastering-bm25-a-deep-dive-into-the-algorithm-and-application-in-milvus)