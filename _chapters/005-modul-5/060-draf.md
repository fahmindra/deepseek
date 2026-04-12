---
slug: modul-5-6
title: modul-5-6
---
# Desain Pipeline Training LLM End-to-End

---

**Tujuan Praktikum**: Peserta mampu merancang arsitektur pelatihan yang efisien (menggunakan ZeRO & Mixed Precision), menyusun pipeline pembersihan data dan tokenisasi yang tepat, serta merancang format data untuk kemampuan khusus (Tool Use & Reasoning).

**Deskripsi Singkat Aktivitas**: Anda berperan sebagai AI Architect di sebuah perusahaan teknologi Indonesia. Perusahaan ingin membangun LLM *from scratch* atau melakukan *heavy retraining* untuk domain spesifik Bahasa Indonesia dan Matematika. Tugas Anda adalah menyusun dokumen teknis untuk menyanggah proposal tim yang tidak efisien dan memberikan solusi arsitektur yang optimal.

## Langkah-Langkah Praktikum

1.  **Siapkan Dokumen Jawaban**:
    *   Buat file teks/dokumen baru: "Laporan Praktikum Modul 2.1 - [Nama Anda]".

2.  **Analisis Skenario 1: Efisiensi Infrastruktur (Konsep Distributed Training)**
    *   **Skenario:** Tim infrastruktur melaporkan bahwa memori GPU selalu *Out of Memory* (OOM) saat mencoba melatih model 7B parameter, padahal mereka sudah menggunakan Data Parallelism (DP) standar. Log menunjukkan bahwa setiap GPU menyimpan salinan penuh dari optimizer states, gradients, dan parameter.
    *   **Pertanyaan:**
        *   a. Jelaskan mengapa konfigurasi DP standar menyebabkan redundansi memori yang tinggi. Solusi apa yang Anda tawarkan menggunakan konsep **ZeRO (Zero Redundancy Optimizer)**? Jelaskan perbedaan spesifik antara ZeRO-1, ZeRO-2, dan ZeRO-3.
        *   b. Selain ZeRO, tim ingin mempercepat waktu training. Jelaskan bagaimana **Mixed Precision Training** (menggunakan BFloat16) dapat membantu dari segi penghematan memori dan kecepatan komputasi dibandingkan FP32.

3.  **Analisis Skenario 2: Persiapan Data & Tokenizer (Konsep Data Engineering)**
    *   **Skenario:** Tim Data mengumpulkan 1TB teks mentah dari internet (Common Crawl) yang penuh duplikasi dan kode HTML sampah. Mereka juga bersikeras menggunakan tokenizer bawaan LLAMA (yang dilatih pada Bahasa Inggris) untuk model Bahasa Indonesia ini.
    *   **Pertanyaan:**
        *   a. Rancang pipeline pembersihan data menggunakan konsep **Heuristics** dan **Deduplication**. Jelaskan bagaimana metode **MinHash** bekerja untuk mendeteksi *fuzzy deduplication* (dokumen mirip tapi tak sama).
        *   b. Jelaskan risiko menggunakan tokenizer Inggris untuk Bahasa Indonesia menggunakan konsep **Fertility Score**. Mengapa melatih tokenizer baru atau melakukan *vocabulary expansion* sangat disarankan demi efisiensi training dan inference?

4.  **Analisis Skenario 3: Strategi Pretraining (Konsep Packing & Scheduler)**
    *   **Skenario:** Saat training berjalan, utilisasi GPU naik turun drastis karena panjang dokumen yang sangat beragam. Tim menggunakan padding berlebihan untuk menyamakan panjang sequence.
    *   **Pertanyaan:**
        *   a. Usulkan penerapan teknik **Sequence Packing**. Jelaskan mekanisme **Document Masking** pada Attention Map agar dokumen yang digabung dalam satu sequence tidak saling "mengintip" atau bercampur konteksnya.
        *   b. Tim menggunakan Cosine Decay standar, namun khawatir jika dataset perlu ditambah di tengah jalan, training harus diulang. Strategi scheduler alternatif apa (seperti **WSD**) yang Anda sarankan dan mengapa?

5.  **Analisis Skenario 4: Post-Training (Tool Use & Reasoning)**
    *   **Skenario:** Manajemen ingin model mampu: (1) Menghitung matematika dengan runut, dan (2) Menggunakan kalkulator eksternal. Tim data bingung menentukan format data latihnya.
    *   **Pertanyaan:**
        *   a. **Reasoning:** Buatlah satu contoh format data latih menggunakan *Special Token* (misal `<think>`) yang memaksa model melakukan *Chain of Thought* sebelum memberikan jawaban akhir.
        *   b. **Tool Use:** Buatlah contoh format data (JSON schema) untuk definisi fungsi kalkulator, dan bagaimana bentuk respon model (tool call) saat diminta menghitung "25 dikali 48".

## Tools & Resource yang Dibutuhkan

*   **Akses Materi Modul 2.1**: Khususnya diagram ZeRO Stages, konsep MinHash/LSH, dan format dataset instruksi.
*   **Aplikasi Pengolah Kata/Code Editor**: Microsoft Word atau VS Code (untuk menyusun snippet JSON/XML).

## Contoh Output atau Hasil Akhir

Satu dokumen laporan (format .pdf atau .docx) yang berisi jawaban terstruktur beserta snippet format data.
*   **Format**: Setiap jawaban harus diberi nomor yang sesuai (misal: "Analisis Skenario 1: ...").
*   **Isi**: Penjelasan teknis mengenai Sharding (ZeRO), diagram/penjelasan logika MinHash, dan blok kode valid untuk JSON Tool Use.

## Kriteria Keberhasilan

*   Mampu menjelaskan secara teknis bagaimana ZeRO memecah state optimizer dan parameter untuk menghemat VRAM.
*   Mampu mengidentifikasi inefisiensi Tokenizer (Fertility) pada lintas bahasa.
*   Memahami logika Document Masking dalam Packed Sequence untuk mencegah kontaminasi konteks.
*   Mampu menuliskan format data latih yang valid untuk SFT kemampuan Reasoning dan Tool Use.

## Petunjuk Troubleshooting

Gunakan panduan ini jika Anda mengalami kebuntuan dalam menganalisis skenario:
*   **Bingung bedanya ZeRO-2 dan ZeRO-3?**
    Bayangkan komponen model sebagai tiga lapis: Optimizer States, Gradients, dan Parameters. ZeRO-1 hanya memecah Optimizer. ZeRO-2 memecah Optimizer + Gradient. ZeRO-3 memecah ketiganya. Mana yang paling hemat VRAM?
*   **Kesulitan menjelaskan Tokenizer Fertility?**
    Bayangkan kata "Menyusuri". Tokenizer Inggris mungkin memecahnya jadi 4 bagian (Men-yu-sur-i), sedangkan Tokenizer Indonesia hanya 1 atau 2. Token yang lebih banyak = komputasi lebih berat.
*   **Ragu soal Document Masking?**
    Dalam Attention Matrix, token di dokumen B tidak boleh "melihat" token di dokumen A meskipun mereka ada dalam satu batch yang sama. Masking adalah "penutup mata" model.
*   **Bingung format Tool Use?**
    Jangan tulis kode Python untuk kalkulatornya. Tuliskan *schema* JSON yang memberi tahu model bahwa "ada alat bernama kalkulator" dan bagaimana model harus menulis teks (JSON output) untuk memanggil alat tersebut.

**Durasi Praktik Disarankan: 60 – 90 Menit.**

### Aksi Lanjutan atau Refleksi (Opsional)
Setelah merancang pipeline ini, renungkan: Jika budget komputasi dipotong 50%, komponen mana dari desain Anda yang bisa dikorbankan? Apakah Training from Scratch masih layak, atau lebih baik beralih ke Continued Pretraining (CPT) dari model yang sudah ada?

---
Klik [HANDS-ON: Petunjuk Setup](https://github.com/Modul-LLM/Module-05-2.1-Intermediate) untuk membuka sumber.

# Penutup Modul 1 LLM Training: Dari Data Preparation hingga Pre-training dan Post-training

---

Pada titik ini anda telah mempelajari seberapa besar usaha yang perlu anda lakukan untuk melatih sebuah LLM dari nol. Tidak hanya perlu melatih sampai selesai, anda juga perlu memikirkan bagaimana caranya mendapatkan compute yang cukup untuk melakukan berbagai macam eksperimen untuk menemukan konfigurasi yang tepat.

Dalam modul ini, kita telah mempelajari berbagai macam teknis keperluan untuk melatih LLM:
*   **Mempersiapkan proses training yang efisien.** Terutama apabila anda memiliki akses ke banyak GPU.
*   **Mempersiapkan data untuk training.** Mulai dari menggunakan data yang sudah ada untuk memaksimalkan efisiensi usaha, lalu berfokus mengumpulkan data yang spesifik pada kebutuhan saja.
*   **Melakukan training.** Anda sudah mempelajari bagaimana pentingnya memiliki evaluasi yang jelas terlebih dahulu, lalu memahami bahwa perlu banyak eksperimen untuk mencari konfigurasi yang tepat untuk melakukan training, serta langkah yang dilakukan untuk pretraining maupun post training.

Dengan menyelesaikan materi ini, Anda telah berhasil mencapai tujuan pembelajaran berikut:
*   Anda kini sudah memahami metode apa saja yang bisa diterapkan untuk meningkatkan efisiensi training, serta bagaimana praktek penerapannya.
*   Anda sudah memahami bagaimana persiapan data untuk training dilakukan.
*   Anda sudah memahami teknis melakukan training, baik pretraining, maupun post training yang juga dilengkapi dengan melatih berbagai kemampuan seperti reasoning dan penggunaan tool.

> **Refleksi Akhir**
>
> Umumnya dilakukan terlebih dahulu tahap post-training yang supervised sebelum RL. Tapi Deepseek-R1 langsung melakukan RL untuk melatih kemampuan reasoning, tidak melewati tahap SFT terlebih dahulu. Coba pahami mengapa ini bisa bekerja dan buat analisis, apakah hal yang sama bisa dilakukan pada banyak hal lainnya, atau ini unik untuk reasoning saja?
>
> Untungnya, mengembangkan LLM tidak selalu perlu melatih dari nol. Pada modul-modul berikutnya, anda akan mempelajari cara yang lebih realistis untuk mendapatkan model yang sesuai dengan kebutuhan spesifik pengembangan aplikasi yang anda inginkan. Modul 2.2 dan 2.3 akan mempelajari dua pendekatan yang lebih realistis yang berfokus pada mengadaptasikan model yang sudah dilatih sebelumnya.

## Daftar Pustaka

1. HuggingFaceTB. The Smol Training Playbook – Hugging Face Space. 2025. [https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook)
2. nanotron. The Ultra-Scale Playbook – Hugging Face Space. 2025. [https://huggingface.co/spaces/nanotron/ultrascale-playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook)
3. Taori, R., Gulrajani, I., Zhang, T., Dubois, Y., Li, X., Guestrin, C., & Hashimoto, T. B. Alpaca: A Strong, Replicable Instruction-Following Model. 2023. [https://crfm.stanford.edu/2023/03/13/alpaca.html](https://crfm.stanford.edu/2023/03/13/alpaca.html)
4. Wang, Y., Kordi, Y., Mishra, S., Liu, A., Smith, N. A., Khashabi, D., & Hajishirzi, H. Self-Instruct: Aligning Language Models with Self-Generated Instructions. arXiv:2212.10560 (2022). [https://arxiv.org/abs/2212.10560](https://arxiv.org/abs/2212.10560)
5. OpenAI. Synthetic Data Generation (sdg1 example). [https://cookbook.openai.com/examples/sdg1](https://cookbook.openai.com/examples/sdg1)
6. IBM Research. Synthetic training data for LLMs. 2024. [https://research.ibm.com/blog/LLM-generated-data](https://research.ibm.com/blog/LLM-generated-data)
7. Nadas, M., Diosan, L., Tomescu, A. Synthetic Data Generation Using Large Language Models: Advances in Text and Code. arXiv:2503.14023 (2025). [https://arxiv.org/abs/2503.14023](https://arxiv.org/abs/2503.14023)