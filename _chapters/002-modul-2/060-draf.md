---
slug: modul-2-6
title: modul-2-6
---
### 2.6 Evolusi Arsitektur Transformer
---

Pada Subtopik 2.5, kita telah membedah "cetak biru" orisinal dari Transformer (Gambar 1 dan 9). Arsitektur penuh ini, yang terdiri dari Encoder Stack dan Decoder Stack, dirancang khusus untuk tugas sequence-to-sequence (seq2seq) seperti translasi. Namun, tidak lama setelah paper "Attention Is All You Need" dirilis, para peneliti menyadari bahwa "cetak biru" penuh ini tidak selalu dibutuhkan. Dua bagian (Encoder dan Decoder) tersebut ternyata bisa "berdiri sendiri" dan sangat kuat untuk tugas-tugas yang berbeda.

Apa yang terjadi jika kita hanya mengambil Encoder Stack-nya saja? Apa yang terjadi jika kita hanya mengambil Decoder Stack-nya saja?

Pertanyaan ini melahirkan dua objektif pre-training utama yang mendefinisikan arsitektur model:

1.  **Causal Language Modeling (CLM):**
    *   Ini adalah objektif yang kita bahas di Modul 3.1. Model dilatih untuk menebak kata berikutnya (Next-Token Prediction).
    *   Karena model hanya boleh melihat ke masa lalu, ini secara alami cocok dengan arsitektur Decoder-Only (yang memiliki Masked Self-Attention).
2.  **Masked Language Modeling (MLM):**
    *   Ini adalah objektif yang berbeda. Alih-alih menebak kata terakhir, kita mengambil kalimat, menyembunyikan (masking) beberapa kata di tengah-tengah (misal: "Kucing [MASK] di atas tikar"), dan melatih model untuk menebak kata yang hilang itu.
    *   Untuk melakukan ini, model harus bisa "melihat dua arah" (konteks sebelum dan sesudah [MASK]). Ini secara alami cocok dengan arsitektur Encoder-Only (yang tidak memiliki masking masa depan).

Dua pendekatan pelatihan (CLM vs. MLM) inilah yang secara efektif melahirkan tiga "keluarga" besar arsitektur Transformer yang mendominasi seluruh lanskap LLM modern. Memahami perbedaan ketiganya adalah kunci untuk memilih model yang tepat untuk pekerjaan yang tepat. Di subtopik ini, kita akan memetakan tiga keluarga tersebut.

---

#### 2.6.1 Arsitektur Encoder-Only (Keluarga BERT)

Keluarga pertama adalah model yang membuang Decoder sepenuhnya dan hanya menggunakan Encoder Stack. Model paling terkenal di keluarga ini adalah BERT (Bidirectional Encoder Representations from Transformers).

![BERT dari Google](https://lms.sdmdigital.id/pluginfile.php/635366/mod_page/content/12/Googles-BERT.png)

**Gambar 10. BERT dari Google**

**Apa itu Encoder-Only?**
Arsitektur Encoder-Only adalah model Transformer yang hanya terdiri dari tumpukan Encoder (sisi kiri Diagram 2.1). Model ini dirancang untuk "membaca" seluruh teks input secara bidirectional (dua arah) sekaligus untuk membangun pemahaman kontekstual yang mendalam. Fokus Utama dari arsitektur ini adalah Tugas Natural Language Understanding (NLU).

**Bagaimana Cara Kerjanya?**
Karena hanya menggunakan Encoder, model ini tidak memiliki masking "masa depan". Saat memproses kalimat "Kucing itu mengejar ekornya", token "mengejar" bisa melihat "kucing" (masa lalu) dan "ekornya" (masa depan) secara bersamaan.

**Analogi**
Model ini bekerja seperti seorang detektif atau analis yang membaca sebuah paragraf bukti. Ia tidak mencoba menulis paragraf baru; ia membaca seluruh paragraf (bolak-balik) untuk mendapatkan pemahaman penuh sebelum menjawab pertanyaan (misal, "Apa sentimen paragraf ini?" atau "Sebutkan nama orang yang ada di paragraf ini?").

**Tabel 5. Contoh Model Encoder-Only beserta Tugas dan Cara Kerjanya**

| Model | Contoh Tugas | Cara Kerja |
| :--- | :--- | :--- |
| **BERT** | Klasifikasi Sentimen | Input: "Film ini sangat bagus!" -> Output: "Positif" |
| **RoBERTa** | Named Entity Recognition (NER) | Input: "Tim Cook bekerja di Apple" -> Output: "[Tim Cook-PERSON]... [Apple-ORG]" |
| **DistilBERT** | Ekstraksi Jawaban (QA) | Input: [Konteks], [Pertanyaan] -> Output: [Span jawaban di konteks] |

**Kekuatan model ini adalah pemahaman bidirectional-nya yang mendalam. Kelemahannya: ia bukan model generatif. Ia sangat buruk dalam tugas "melanjutkan kalimat" (generasi autoregressive).**

---

#### 2.6.2 Arsitektur Decoder-Only (Keluarga GPT)

Keluarga kedua, dan yang paling dominan saat ini, adalah model yang membuang Encoder sepenuhnya dan hanya menggunakan Decoder Stack. Model paling terkenal di keluarga ini adalah GPT (Generative Pre-trained Transformer) dan semua penerusnya (Llama, Qwen, Claude, dll.).

![QWEN dari Alibaba](https://lms.sdmdigital.id/pluginfile.php/635366/mod_page/content/12/large_alibaba_qwen.jpg)

**Gambar 13. QWEN dari Alibaba**

**Apa itu Arsitektur Decoder-Only?**
Arsitektur Decoder-Only adalah model Transformer yang hanya terdiri dari tumpukan Decoder (sisi kanan Diagram 2.1). Model ini secara fundamental bersifat autoregressive (generatif, satu arah) karena ia menggunakan Masked Multi-Head Self-Attention. Fokus utama arsitektur ini adalah Tugas Natural Language Generation (NLG) dan zero-shot prompting.

**Bagaimana Cara Kerjanya?**
Model ini hanya menggunakan Masked Self-Attention (Sub-layer 1 di Blok Decoder). Ia tidak memiliki Sub-layer 2 (Encoder-Decoder Attention) karena tidak ada Encoder untuk "dilihat".

**Analogi**
Model ini bekerja seperti seorang novelis atau storyteller. Ia hanya bisa melihat apa yang sudah ia tulis (konteks masa lalu) untuk memutuskan apa kata berikutnya yang akan ia tulis. Ia tidak bisa melihat "masa depan" dari kalimat yang sedang ia tulis.

**Contoh Tugas Arsitektur Decoder-Only:**
*   **Generasi Teks:** Input: "Pada suatu hari, hiduplah..." -> Output: "...seekor naga di gunung..."
*   **Chatbot:** Input: "Apa ibu kota Indonesia?" -> Output: "Ibu kota Indonesia adalah Jakarta."
*   **Zero-Shot Prompting:** Input: "Terjemahkan ke Prancis: 'Saya suka keju'" -> Output: "J'aime le fromage."

Inilah arsitektur yang mendasari ChatGPT, Gemini, Claude, Qwen, Grok, dan sebagian besar LLM generatif modern.

---

#### 2.6.3 Arsitektur Encoder-Decoder (Keluarga T5 / BART)

Keluarga ketiga adalah "kaum tradisionalis" yang menggunakan arsitektur penuh seperti pada Diagram 2.1, yang memiliki Encoder dan Decoder. Model modern yang terkenal di keluarga ini adalah T5 (Text-to-Text Transfer Transformer) dan BART.

![BART](https://lms.sdmdigital.id/pluginfile.php/635366/mod_page/content/12/imagesBART.png)

**Gambar 14. BART**

**Apa itu Arsitektur Encoder-Decoder?**
Arsitektur Encoder-Decoder adalah model Transformer penuh (Diagram 2.1) yang menggunakan Encoder untuk "memahami" input dan Decoder untuk "menghasilkan" output. Model ini memetakan satu sekuens input ke satu sekuens output. Fokus Utamanya adalah Tugas Sequence-to-Sequence (Seq2Seq) murni.

**Bagaimana Cara Kerjanya?**
Ini adalah best of both worlds untuk tugas spesifik. Encoder (seperti BERT) mendapatkan pemahaman bidirectional penuh dari input. Decoder (seperti GPT) kemudian menggunakan pemahaman itu (via cross-attention) untuk menghasilkan output autoregressive.

**Analogi**
Model ini bekerja seperti seorang penerjemah profesional. Ia tidak menerjemahkan kata per kata. Ia pertama-tama membaca seluruh paragraf (Encoder), memahaminya sepenuhnya, lalu (Decoder) mulai menulis terjemahan dari awal, sambil terus-menerus "melihat kembali" ke paragraf asli untuk memastikan akurasi.

**Contoh Tugas Arsitektur Encoder-Decoder:**
*   **Translasi Mesin (Machine Translation):** Input: "The cat sat on the mat" -> Output: "Kucing itu duduk di atas tikar"
*   **Peringkasan Teks (Summarization):** Input: [Artikel panjang 1000 kata] -> Output: [Ringkasan 50 kata]

**Tabel 6. Perbandingan Ringkas Arsitektur Transformer Modern**

| Keluarga | Arsitektur | Analogi | Contoh Model | Tugas Utama |
| :--- | :--- | :--- | :--- | :--- |
| **Encoder-Only** | Tumpukan Encoder | Detektif / Analis | BERT, RoBERTa | NLU, Klasifikasi, NER |
| **Decoder-Only** | Tumpukan Decoder | Novelis / Storyteller | GPT, Llama, Qwen | NLG, Chatbot, Prompting |
| **Encoder-Decoder** | Penuh (Encoder + Decoder) | Penerjemah Profesional | T5, BART, Transformer Asli | Seq2Seq, Translasi, Peringkasan |

Tabel 6 ini secara ringkas memetakan lanskap arsitektur modern. Memahami perbedaan fundamental ini adalah keputusan arsitektural pertama dan paling penting yang akan Anda buat sebagai seorang AI Engineer saat memilih foundation model untuk sebuah proyek.

---

### Mini Quiz

*Silakan akses Mini Quiz melalui platform LMS:*
[Multiple Choice Quiz - Subtopik 2.6](https://lms.sdmdigital.id/h5p/embed.php?url=https%3A%2F%2Flms.sdmdigital.id%2Fpluginfile.php%2F635366%2Fmod_page%2Fcontent%2F12%2Fmultiple-choice-212.p)

---

> **Refleksi Singkat**
>
> Bayangkan Anda ditugaskan membangun dua sistem AI:
> 1. Sistem untuk menganalisis dan memberi label sentimen (Positif/Negatif) pada ribuan ulasan klien.
> 2. Sistem chatbot internal untuk menjawab pertanyaan umum karyawan.
>
> Berdasarkan tiga "keluarga" arsitektur yang telah kita bahas, model mana (Encoder-Only, Decoder-Only, atau Encoder-Decoder) yang akan Anda pilih sebagai fondasi untuk masing-masing tugas, dan mengapa?