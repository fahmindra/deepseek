---
slug: modul-6-8
title: modul-6-8
---
# Analisis Strategi: Continued Pretraining (CPT), Efisiensi, dan Model Merging

---

**Tujuan Praktikum**: Peserta mampu menganalisis skenario bisnis untuk memilih strategi adaptasi model yang tepat (SFT vs CPT vs RAG), merancang strategi mitigasi Catastrophic Forgetting, dan mengevaluasi penggunaan teknik efisiensi seperti LoRA dan Model Merging.

**Deskripsi Singkat Aktivitas**: Anda bertindak sebagai Lead AI Engineer. Perusahaan Anda ingin mengadopsi LLM untuk berbagai kebutuhan spesifik. Tim manajemen dan engineer junior mengajukan beberapa proposal proyek. Tugas Anda adalah mengevaluasi proposal tersebut, menunjukkan risiko fatalnya (seperti forgetting), dan mengusulkan arsitektur solusi yang lebih tepat dan efisien.

---

## Langkah-Langkah Praktikum

### 1. Siapkan Dokumen Jawaban
*   Buat file teks/dokumen baru: "Unjuk Kerja Modul 2.2 - [Nama Anda]".

### 2. Analisis Skenario 1: Pengetahuan vs. Perilaku (Konsep 2.2.1 & 2.2.2)
**Skenario**: Tim Legal ingin melatih model Llama-3 agar hafal dan paham 50.000 dokumen kontrak internal perusahaan (Total 20GB data). Engineer junior mengusulkan untuk mengubah dokumen tersebut menjadi format tanya-jawab (Q&A) dan melakukan Supervised Fine-Tuning (SFT).

**Pertanyaan**:
*   **a.** Mengapa pendekatan SFT mungkin tidak efektif untuk tujuan "menghafal/memahami" seluruh isi 50.000 dokumen tersebut secara mendalam? (Hubungkan dengan cara kerja loss SFT).
*   **b.** Jika Anda menyarankan CPT, apa konsekuensi biaya (Compute/VRAM) yang harus disiapkan manajemen dibandingkan dengan proposal SFT awal?
*   **c.** Sebagai alternatif ketiga yang paling hemat biaya dan akurat untuk retrieval fakta, strategi non-training apa yang akan Anda sarankan?

### 3. Analisis Skenario 2: Mitigasi Risiko (Konsep 2.2.3 & 2.2.6)
**Skenario**: Anda memutuskan melakukan CPT pada model GPT-2 menggunakan 100% data jurnal kedokteran. Setelah pelatihan selesai, model sangat jago menjawab istilah medis, tapi saat dites dengan prompt "Tuliskan puisi tentang hujan", model mengeluarkan output yang kacau dan tidak gramatikal.

**Pertanyaan**:
*   **a.** Apa nama fenomena teknis yang terjadi pada model tersebut? Jelaskan penyebabnya menggunakan konsep **distribusi data**.
*   **b.** Rancang strategi **Data Mixing (Replay)** yang konkret untuk pelatihan ulang. Berapa persentase data medis vs data umum yang akan Anda gunakan, dan teknik apa (Concatenation atau Interleaving) yang Anda pilih jika total datanya mencapai 2 Terabyte?

### 4. Analisis Skenario 3: Efisiensi Arsitektur (Konsep 2.2.5)
**Skenario**: Anda hanya memiliki akses ke satu GPU consumer-grade (NVIDIA RTX 4090, 24GB VRAM). Anda perlu melakukan adaptasi domain pada model Llama-3-8B (yang ukuran file bobotnya saja sudah sekitar 16GB).

**Pertanyaan**:
*   **a.** Mengapa **Full CPT** mustahil dilakukan di perangkat keras ini?
*   **b.** Jelaskan bagaimana teknik **Adapter-Based (LoRA)** memungkinkan Anda melatih model ini di GPU tersebut. Apa yang terjadi pada bobot model asli selama proses ini?

### 5. Analisis Skenario 4: Model Merging (Konsep 2.2.7)
**Skenario**: Komunitas open-source telah merilis dua model hebat berbasis Mistral-7B: satu jago Matematika (Math-Mistral) dan satu jago Bahasa Indonesia (Indo-Mistral). Anda ingin membuat model yang jago Matematika DALAM Bahasa Indonesia.

**Pertanyaan**:
*   Jelaskan secara konseptual bagaimana **Model Merging** (seperti teknik TIES atau SLERP) bisa menggabungkan kedua kemampuan ini **tanpa** melakukan pelatihan ulang (training) sama sekali.

---

## Tools & Resource yang Dibutuhkan

*   **Akses Materi Modul 2.2**: Khususnya tabel perbandingan SFT vs CPT (2.2.1), diagram alur CPT (2.2.4), dan konsep LoRA (2.2.5).
*   **Aplikasi Pengolah Kata**: Microsoft Word, Google Docs, atau Text Editor (Notepad/VS Code) untuk menyusun jawaban analisis.
*   **Koneksi Internet**: (Opsional) Untuk mencari referensi tambahan mengenai spesifikasi GPU RTX 4090 atau model Llama-3 jika diperlukan untuk justifikasi teknis.

---

## Contoh Output atau Hasil Akhir

Satu dokumen laporan (format .pdf atau .docx) yang berisi jawaban terstruktur untuk keempat skenario.
*   **Format**: Setiap jawaban harus diberi nomor yang sesuai dengan langkah (misal: "Analisis Skenario 1: ...").
*   **Isi**: Jawaban harus berupa argumen teknis yang padat (bukan esai panjang), menggunakan terminologi yang tepat (misal: *Distribution Shift, Masked Loss, Rank Decomposition*).

---

## Kriteria Keberhasilan

*   Mampu membedakan kapan harus menggunakan SFT, CPT, atau RAG berdasarkan kebutuhan "Pengetahuan vs Perilaku".
*   Mampu mengidentifikasi penyebab *Catastrophic Forgetting* dan merancang strategi *Replay* yang tepat.
*   Mampu menjelaskan mekanisme penghematan memori pada LoRA.
*   Mampu menjelaskan konsep dasar penggabungan bobot pada Model Merging.

---

## Petunjuk Troubleshooting

Gunakan panduan ini jika Anda mengalami kebuntuan dalam menganalisis skenario:
*   **Bingung membedakan efektivitas SFT vs CPT di Skenario 1?**
    Lihat kembali Subtopik 2.2.1. Ingat mantra kuncinya: "SFT untuk Perilaku, CPT untuk Pengetahuan". Apakah "menghafal dokumen" itu perilaku atau pengetahuan? Pikirkan juga tentang Masked Loss, apakah model belajar dari prompt di SFT?
*   **Kesulitan menjelaskan penyebab output kacau di Skenario 2?**
    Buka kembali Subtopik 2.2.3 (Catastrophic Forgetting). Fokus pada konsep "Pergeseran Distribusi" (Distribution Shift). Bayangkan model dipaksa pindah dari "Kota Umum" ke "Kota Medis" secara total.
*   **Ragu mengapa Full CPT mustahil di Skenario 3?**
    Lakukan hitungan kasar VRAM (seperti di Subtopik 2.2.2). Model 8B parameter 16GB (FP16). Untuk training, Anda butuh memuat gradien + optimizer states yang bisa memakan memori 3-4x lipat dari ukuran model. Apakah 24GB cukup? Lalu bandingkan dengan LoRA (2.2.5) yang hanya menambah <1% parameter.
*   **Bingung konsep Model Merging di Skenario 4?**
    Bayangkan parameter model sebagai vektor matematika (Subtopik 2.2.7). Ingat bahwa kita bisa melakukan operasi aritmatika A+B pada vektor tersebut tanpa perlu menjalankan proses backpropagation (training).

---

**Durasi Praktik Disarankan: 60 – 90 Menit.**

---

## Aksi Lanjutan atau Refleksi (Opsional)

Setelah menyelesaikan analisis ini, coba posisikan diri Anda sebagai CTO (Chief Technology Officer). Jika Anda harus mempresentasikan strategi AI perusahaan kepada CEO yang awam teknis: Bagaimana Anda akan menjelaskan bahwa "melatih model agar pintar" (CPT) itu jauh lebih mahal dan berisiko daripada "melatih model agar nurut" (SFT), dan mengapa strategi "Jalan Tengah" (seperti RAG atau LoRA) seringkali merupakan investasi terbaik?

---
*Klik [HANDS-ON: Advanced Training Techniques (CPT & LoRA)](https://github.com/Modul-LLM/Module-06-2.2-Intermediate) untuk membuka sumber.*

---

# Penutup: Modul 2 Continued Pretraining

Di modul ini, kita telah beralih dari sekadar menggunakan model (seperti di level Fundamental) menjadi membedah dan memodifikasi pengetahuannya. Kita belajar bahwa mengubah **pengetahuan** model adalah tantangan yang jauh lebih besar daripada sekadar mengubah **perilaku**.

Berikut adalah intisari dari perjalanan kita di Modul 2:
*   **Strategi Adaptasi (2.1 - 2.2):** Kita membedakan **SFT** (melatih perilaku/keterampilan dengan data instruksi) dan **CPT** (menambah pengetahuan/domain dengan teks mentah). Kita menyadari bahwa CPT adalah "palu godam" yang mahal (Data & Compute tinggi) dan harus digunakan dengan hati-hati dibandingkan alternatif seperti RAG.
*   **Risiko Fatal (2.3):** Kita mengidentifikasi musuh terbesar CPT: **Catastrophic Forgetting**. Kita melihat bagaimana membanjiri model dengan data baru dapat "menimpa" pengetahuan lama, merusak kemampuan bahasa, logika, dan *alignment*.
*   **Teknik Mitigasi (2.4 - 2.6):** Kita membongkar kotak peralatan engineering untuk CPT yang aman:
    *   Menggunakan **Low Learning Rate** dan **Warmup** untuk pendaratan bobot yang mulus.
    *   Menerapkan strategi **Data Mixing (Replay)** —baik *Exact* maupun *Generative*— untuk menjaga ingatan model.
    *   Menggunakan teknik **Interleaving** untuk menangani dataset skala Terabyte secara efisien.
*   **Efisiensi Arsitektur (2.5):** Kita membandingkan **Full CPT** (mahal) dengan **PEFT/LoRA** (hemat). Kita melihat bagaimana LoRA memungkinkan adaptasi domain pada GPU konsumen dengan membekukan otak utama model.
*   **Inovasi Tanpa Pelatihan (2.7):** Terakhir, kita mengeksplorasi **Model Merging**, sebuah teknik "sihir matematika" yang memungkinkan kita menggabungkan kemampuan dua model (misal: Coding + Bahasa Indonesia) hanya dengan menjumlahkan vektor bobotnya, tanpa pelatihan tambahan.

Dengan menyelesaikan materi dalam modul ini, Anda telah berhasil mencapai tujuan pembelajaran berikut:
1.  Anda kini dapat **membedakan secara kritis** antara *Continued Pretraining* (untuk penambahan pengetahuan) dan *Supervised Fine-Tuning* (untuk penyesuaian perilaku), serta menghitung *trade-off* biaya komputasi yang masif di antara keduanya.
2.  Anda mampu **menganalisis risiko fundamental** *Catastrophic Forgetting* dan memahami bagaimana pergeseran distribusi data dapat secara tidak sengaja "menimpa" kemampuan model sebelumnya.
3.  Anda dapat **menerapkan strategi mitigasi** teknis yang tepat, khususnya *Data Mixing (Replay)* dan pengaturan *Learning Rate* yang konservatif, untuk menjaga stabilitas pengetahuan model.
4.  Anda dapat **memilih varian arsitektur pelatihan** yang paling efisien sesuai sumber daya, membandingkan *Full CPT* yang mahal dengan pendekatan *Adapter-based* (LoRA) yang hemat memori.
5.  Anda mampu **menjelaskan konsep Model Merging** sebagai alternatif inovatif untuk menggabungkan kemampuan dua model berbeda tanpa perlu melakukan pelatihan ulang.

---

> **Refleksi Akhir:**
>
> Sebagai penutup, mari refleksikan posisi Anda sebagai AI Engineer. Anda sekarang memiliki kemampuan teknis untuk melakukan Continued Pretraining. Anda tahu cara mencampur data, mengatur learning rate, dan menggunakan LoRA. Namun, kebijaksanaan seorang engineer bukan terletak pada kemampuan menggunakan alat, melainkan pada keputusan kapan menggunakannya.
>
> Jika besok atasan Anda meminta: "Ayo kita latih ulang Llama-3 agar paham semua dokumen PDF perusahaan kita," apakah Anda akan langsung menjalankan skrip CPT? Atau apakah Anda akan berhenti sejenak, mengingat risiko Catastrophic Forgetting dan biaya komputasi yang ekstrem, lalu mengusulkan solusi RAG atau SFT terlebih dahulu?
>
> Pengetahuan di modul ini adalah "rem" dan "gas" Anda. Gunakanlah dengan bijak untuk membangun sistem AI yang tidak hanya pintar, tetapi juga efisien dan robust.

---

## Referensi

1. Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A. A., ... & Hadsell, R. (2017). Overcoming catastrophic forgetting in neural networks. Proceedings of the national academy of sciences, 114(13), 3521-3526. [https://www.pnas.org/doi/full/10.1073/pnas.1611835114](https://www.pnas.org/doi/full/10.1073/pnas.1611835114)
2. Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., ... & Chen, W. (2022). Lora: Low-rank adaptation of large language models. ICLR, 1(2), 3. [https://arxiv.org/pdf/2106.09685v1/1000](https://arxiv.org/pdf/2106.09685v1/1000)
3. Ilharco, G., Ribeiro, M. T., Wortsman, M., Gururangan, S., Schmidt, L., Hajishirzi, H., & Farhadi, A. (2022). Editing models with task arithmetic. arXiv preprint arXiv:2212.04089. [https://arxiv.org/pdf/2212.04089](https://arxiv.org/pdf/2212.04089)
4. Gururangan, S., Marasović, A., Swayamdipta, S., Lo, K., Beltagy, I., Downey, D., & Smith, N. A. (2020). Don't stop pretraining: Adapt language models to domains and tasks. arXiv preprint arXiv:2004.10964. [https://arxiv.org/pdf/2004.10964](https://arxiv.org/pdf/2004.10964)
5. Yadav, P., Tam, D., Choshen, L., Raffel, C. A., & Bansal, M. (2023). Ties-merging: Resolving interference when merging models. Advances in Neural Information Processing Systems, 36, 7093-7115. [https://proceedings.neurips.cc/paper_files/paper/2023/file/1644c9af28ab7916874f6fd6228a9bcf-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2023/file/1644c9af28ab7916874f6fd6228a9bcf-Paper-Conference.pdf)