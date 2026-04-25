---
slug: modul-6-2
title: modul-6-2
---
# 2.2 Kebutuhan Data dan Compute dari CPT dan SFT

---

Di Subtopik 2.1, kita telah menetapkan perbedaan fundamental: SFT adalah untuk **Perilaku/Keterampilan**, sementara CPT adalah untuk **Pengetahuan/Domain**. Kita juga telah melihat di kode (2.1.4) betapa berbedanya setup data (`SFTTrainer` vs. `DataCollatorForLanguageModeling`).

Sekarang, kita harus menjawab pertanyaan *engineering* yang paling penting: **Berapa biayanya?**

Seorang AI Engineer tidak hanya memilih pendekatan berdasarkan teori, tetapi juga berdasarkan *resource* yang tersedia. Pilihan antara SFT dan CPT adalah salah satu *trade-off* terbesar dalam hal biaya **Data** (persiapan) dan **Compute** (GPU). Di subtopik ini, kita akan membedah secara detail mengapa CPT secara eksponensial lebih mahal daripada SFT.

---

## 2.2.1 Perbandingan Kebutuhan Data: Kualitas vs. Kuantitas

Perbedaan biaya pertama dan paling jelas terletak pada dataset yang dibutuhkan.

### 1. Data untuk Supervised Fine-Tuning (SFT)
*   **Fokus:** Kualitas di atas Kuantitas.
*   **Tipe:** Data berlabel, terstruktur, dalam format instruksi (misal, `{"prompt": "...", "response": "..."}`).
*   **Ukuran:** Relatif sangat kecil. Dataset SFT yang efektif bisa berkisar dari beberapa ribu hingga puluhan ribu contoh berkualitas tinggi.
*   **Mengapa?** Karena kita tidak mengajari model pengetahuan baru. Kita hanya mengajarinya *format* atau *pola perilaku*. Model (seperti Llama 3) sudah tahu apa itu "Ibu kota Indonesia". Kita hanya butuh beberapa ratus contoh untuk mengajarinya format jawaban: "Jika ditanya 'Apa ibu kota...', *jawablah* 'Ibu kota... adalah Jakarta.". Ini adalah masalah *alignment*, bukan *knowledge*.

### 2. Data untuk Continued Pretraining (CPT)
*   **Fokus:** Kuantitas di atas Kualitas (meskipun kualitas tetap penting).
*   **Tipe:** Teks mentah (*raw text*), tidak terstruktur.
*   **Ukuran:** Masif. Agar CPT efektif dan dapat "menggeser" pengetahuan internal model (yang dilatih pada triliunan token), kita membutuhkan dataset baru yang sangat besar, biasanya dalam orde puluhan Gigabyte (GB) hingga Terabyte (TB).
*   **Mengapa?** Karena kita mengajari model pengetahuan baru. Agar model "menghafal" fakta baru (misal, nama obat atau peristiwa tahun 2024), ia harus melihat fakta dan pola tersebut *berulang kali* dalam konteks yang berbeda, sama seperti saat ia dilatih (*pre-training*). Memberi CPT hanya 1000 contoh (seperti SFT) tidak akan ada gunanya; itu akan hilang seperti setetes air di lautan *pre-training* aslinya.

---

## 2.2.2 Perbandingan Kebutuhan Compute (FLOPS & VRAM)

Ini adalah biaya yang sebenarnya. "Compute" dapat kita pecah menjadi dua:
1.  **FLOPs (Floating Point Operations):** Total jumlah kalkulasi yang dibutuhkan. Ini menentukan **waktu (durasi)** dan **biaya listrik** training.
2.  **VRAM (Memori GPU):** Total memori yang dibutuhkan untuk menampung model dan data. Ini menentukan **tipe GPU** yang Anda butuhkan (misal, 24GB VRAM atau 80GB VRAM).

### 1. Biaya Compute SFT
*   **FLOPs (Rendah):** Karena datasetnya kecil (misal, 10.000 contoh) dan kita hanya melatih loss pada token respons (ingat Masked Loss di 2.1.4), jumlah total kalkulasi yang dibutuhkan relatif **sangat rendah**. Proses fine-tuning seringkali selesai dalam hitungan menit atau beberapa jam.
*   **VRAM (Tinggi, tapi... Fleksibel):**
    *   Secara default (Full Fine-Tuning), SFT tetap harus memuat **seluruh parameter model** (misal, Llama 3 8B) ke VRAM, yang mana sangat besar.
    *   **NAMUN**, SFT sangat cocok dikombinasikan dengan teknik **PEFT (Parameter-Efficient Fine-Tuning)** seperti **LoRA** (yang akan kita bahas di modul mendatang). LoRA "membekukan" model asli dan hanya melatih "adapter" kecil (mungkin hanya 1-2% dari total parameter). Ini secara drastis **menurunkan kebutuhan VRAM**, memungkinkan SFT dilakukan pada GPU level konsumen (consumer-grade).

### 2. Biaya Compute CPT
*   **FLOPs (Sangat Tinggi):** Ini adalah "pembunuh" anggaran. Seperti yang kita bahas di 2.1.2, CPT melatih model pada **setiap token** dalam dataset masif (puluhan GB). Jumlah total kalkulasi (FLOPs) yang dibutuhkan **sangat-sangat tinggi**, seringkali 100x atau 1000x lipat lebih besar daripada SFT. Ini membutuhkan waktu berhari-hari atau berminggu-minggu, bahkan di cluster GPU yang kuat.
*   **VRAM (Sangat Tinggi):**
    *   CPT (secara *default*) adalah *Full-Parameter Training*. Anda harus memuat seluruh model *dan gradient optimizer* untuk *setiap* parameter.
    *   Ini membutuhkan VRAM yang sangat besar (misal, GPU A100/H100 80GB) dan **seringkali** memerlukan **Distributed Training** (Modul 1 Intermediate) di banyak GPU sekaligus.
    *   Meskipun CPT *bisa* dilakukan dengan PEFT/LoRA (kita akan bahas di 2.2.5), CPT *full-parameter* seringkali lebih disukai untuk injeksi pengetahuan yang mendalam, yang menjadikannya sangat mahal.

---

## 2.2.3 Perbandingan dan Peringatan

Tabel 2 di bawah ini merangkum trade-off biaya secara ringkas.

**Tabel 2. Perbandingan Biaya Kebutuhan CPT vs. SFT**

| Kebutuhan | SFT (Perilaku) | CPT (Pengetahuan) |
| :--- | :--- | :--- |
| **Tipe Data** | Berlabel (Instruksi) | Mentah (Teks) |
| **Ukuran Data** | Kecil (Kualitas) | Masif (Kuantitas) |
| **Biaya Data Prep** | Tinggi (Butuh pelabelan manusia/GPT-4) | Rendah (Hanya perlu cleaning teks) |
| **Biaya Compute (FLOPs)** | Rendah (Menit/Jam) | Ekstrem (Hari/Minggu) |
| **Biaya VRAM** | Fleksibel (Bisa rendah dengan LoRA) | Sangat Tinggi (Butuh A100/H100) |

> **Peringatan (Alasan CPT Bukan Pendekatan Tepat):**
>
> Sebelum memutuskan melakukan CPT, seorang AI Engineer harus bertanya:
> 1. "Apakah saya benar-benar butuh CPT (menambah pengetahuan), atau saya hanya butuh SFT (mengubah perilaku)?"
> 2. "Apakah RAG (Retrieval-Augmented Generation) bisa menyelesaikan masalah saya dengan lebih murah daripada CPT?"
>
> Jika Anda hanya memiliki 10GB data domain dan 1 GPU 24GB, CPT *full-parameter* secara praktis **mustahil** dilakukan. Anda harus mempertimbangkan SFT (jika datanya bisa dilabel) atau RAG. CPT adalah pilihan terakhir yang sangat mahal.

---

## Mini Quiz

[Konten Interaktif: Multiple Choice Quiz]
*(Akses kuis melalui platform LMS: https://lms.sdmdigital.id/h5p/embed.php?url=https%3A%2F%2Flms.sdmdigital.id%2Fpluginfile.php%2F635428%2Fmod_page%2Fcontent%2F13%2Fmultiple-choice-185.h5p)*

---

> **Refleksi Singkat**
> 
> Kita telah melihat bahwa CPT memiliki biaya compute (FLOPs dan VRAM) yang ekstrem (Tabel 2), sementara SFT jauh lebih murah dan RAG (disebutkan di 2.2.1) bahkan lebih murah lagi.
>
> Sebagai AI Engineer, Anda diminta membuat chatbot untuk layanan hukum. Klien Anda (sebuah firma hukum) memberi Anda 50GB data yurisprudensi (kasus-kasus hukum) dan budget GPU yang terbatas.
>
> Mengapa Anda akan menghindari CPT sebagai langkah pertama, dan apa (SFT atau RAG) yang akan Anda usulkan terlebih dahulu untuk menghemat biaya?