---
slug: modul-4-1
title: modul-4-1
---
## 4.1 Memahami Kebutuhan Data dan Compute dalam Proses Pelatihan LLM

Sebutan **Large** dalam LLM berasal dari ukuran parameternya yang besar, melebihi model bahasa sebelumnya. Jumlah parameter yang besar ini memunculkan kemampuan unik dari LLM setelah dilatih, yaitu kemampuan mengikuti instruksi dan melakukan *in-context learning* yang sebelumnya tidak pernah teramati pada model yang berukuran lebih kecil. Kemampuan ini disebut sebagai *emergent ability*, dan menjadi justifikasi keperluan istilah baru untuk menyebut keluarga model bahasa ini. Menurut prinsip *Machine Learning*, umumnya sebuah model dengan jumlah parameter yang besar memerlukan data dengan jumlah yang sangat banyak untuk bisa dilatih secara efektif. Sebagai konsekuensi dari jumlah data yang banyak ini ditambah dengan skala model yang sangat besar, diperlukan kapasitas komputasi yang cukup besar dari segi *memory* maupun *compute* untuk bisa menangani beban dari proses melatih sebuah LLM.

Sehingga ada tiga variabel yang penting untuk dipikirkan ketika ingin mengembangkan sebuah LLM. Ketiga variabel ini tidak berdiri sendiri, melainkan terikat erat dalam hukum matematis:

1.  **N - Number of Parameters (Ukuran Model):** Seberapa besar kapasitas memori/kecerdasan model.
2.  **D - Dataset Size (Jumlah Data):** Seberapa banyak token (kata) yang dipelajari.
3.  **C - Compute Budget (Komputasi):** Seberapa besar daya GPU dan waktu yang tersedia untuk pelatihan.

### 4.1.1 Hubungan Ketiganya: The Scaling Laws

Mengapa kita tidak membuat model sebesar mungkin saja? Jawabannya ada pada *Scaling Laws* (Hukum Penyekalaan). Riset dari Kaplan (2020) dan Hoffmann (2022) menemukan rumus empiris untuk mengoptimalkan hubungan ini. Intinya: **Performa model (Loss) dapat diprediksi.**

*   Jika Anda meningkatkan Parameter (N), Anda **wajib** meningkatkan Data (D) secara proporsional.
*   Jika Anda hanya membesarkan model tapi datanya kurang, komputasi (C) Anda akan terbuang sia-sia (*Compute Inefficient*).
*   Sebaliknya, jika model terlalu kecil untuk data yang melimpah, model tidak akan mampu menyerap semua ilmu tersebut (*Underfitting*).

*Scaling Laws* memberikan "Resep Koki" bagi Engineer untuk menentukan: "Dengan budget GPU sekian (Compute), berapa ukuran model dan jumlah data yang paling optimal agar hasilnya maksimal?"

Umumnya proses melatih sebuah LLM dibagi menjadi dua, pertama dilakukan *self-supervised pretraining* seperti umumnya dilakukan pada model bahasa modern, dilanjutkan dengan proses yang lebih spesifik yang umumnya disebut sebagai *post-training*. Kedua tahap ini memerlukan data yang berbeda untuk tujuan *training* yang berbeda. Seperti apa data yang dibutuhkan, baik dari segi skala maupun bentuknya?

---

### 4.1.2 Keperluan Data untuk Melatih LLM

Untuk memahami keperluan data LLM, kita bisa merujuk kepada *paper* dari model populer seperti GPT-3, LLaMa, instructGPT, Gemma, dan berbagai model *open-source* maupun *closed-source* lainnya. Sebuah *paper* survey berjudul [“A Survey on Large Language Models”](https://arxiv.org/abs/2303.18223) yang diterbitkan pada 11 Maret 2025 memberikan ringkasan mengenai berbagai macam detail dari banyak model LLM yang telah dilatih, termasuk jumlah data yang digunakan untuk melatihnya. Paper ini sendiri membagi datanya berdasarkan setiap tahapan dalam proses melatih LLM yang akan dijelaskan dalam bagian lain pada modul ini.

---

### 4.1.3 Kebutuhan data untuk pretraining

Sebelum masuk ke angka-angka raksasa, kita perlu menyamakan persepsi tentang dua istilah fundamental yang sering tertukar: Token dan Parameter.

**1. Konsep Dasar: Token vs. Parameter**

Bayangkan kita sedang melatih seorang siswa (AI) di sebuah perpustakaan.

*   **Token (Data / Bahan Belajar):**
    Ini adalah "kata-kata" atau potongan teks yang dibaca oleh model. Token adalah input.
    *   *Analogi:* Jumlah buku di perpustakaan yang harus dibaca siswa.
    *   *Fakta:* 1 kata kira kira sekitar 0.75 token. Jadi, 100 kata kira-kira setara dengan 75 token.
*   **Parameter (Kapasitas Model / Otak):**
    Ini adalah variabel internal (bobot sinapsis) di dalam jaringan saraf tiruan yang menyimpan pengetahuan hasil belajar.
    *   *Analogi:* Jumlah sel otak (neuron) yang dimiliki siswa untuk mengingat dan memproses informasi dari buku tersebut.
    *   *Hubungan:* Semakin besar parameter, semakin "cerdas" potensinya, tetapi ia butuh "buku" (token) yang jauh lebih banyak agar otaknya terisi penuh.

**2. Seberapa Besar 1 Triliun Token Itu?**

Angka "1 Triliun" (1T) sering terdengar abstrak. Mari kita bandingkan dengan Wikipedia Bahasa Inggris, yang merupakan salah satu sumber pengetahuan terbesar manusia.

*   Seluruh Wikipedia Bahasa Inggris berisi sekitar 4 Miliar (4B) kata.
*   Jika dikonversi ke token, itu sekitar ~5 Miliar token.

Artinya: **1 Triliun Token berkisar antara 200x kali lipat seluruh isi Wikipedia Bahasa Inggris.**

Untuk melatih model seperti LLaMA-65B (1.4T token), model tersebut harus "membaca" setara dengan **280 kali khatam membaca seluruh Wikipedia.** Inilah mengapa kita menyebutnya *Large Language Model*—skala datanya melampaui kemampuan baca manusia seumur hidup.

**3. Evolusi Kebutuhan Data (Tabel Perbandingan)**

Dulu, peneliti mengira membesarkan "Otak" (Parameter) adalah segalanya. Namun, tren bergeser. Model modern (seperti LLaMA dan Qwen) membuktikan bahwa model dengan "Otak Sedang" tapi "Belajar Sangat Banyak" (Data melimpah) justru lebih unggul.

Tabel berikut menunjukkan pergeseran tren tersebut:

**Tabel 1. Kebutuhan data untuk pretraining beberapa LLM serta jumlah parameternya**

| Nama Model | Jumlah Parameter (Kapasitas Otak) | Jumlah Data (Bahan Belajar) | Rasio (Token per Param) |
| :--- | :--- | :--- | :--- |
| GPT-3 (2020) | 175 Miliar | 300 Miliar Token | ~1.7x (Undertrained) |
| GLM (2022) | 130 Miliar | 400 Miliar Token | ~3x |
| LLaMA-1 (2023) | 65 Miliar | 1.4 Triliun Token | ~21x (Optimal) |
| QWEN / Llama-3 | 14 Miliar | 3 Triliun Token | ~200x (Overtrained) |

Dari mana angka-angka di atas berasal? Mengapa LLaMA menggunakan 1.4 Triliun token, bukan sembarang angka?
Para engineer mengacu pada **"Chinchilla Scaling Laws"** (Hoffmann et al., 2022). Penelitian ini menemukan "Resep Emas" untuk efisiensi pelatihan:

> **Ratio $\approx$ 1:20**
> Untuk setiap 1 Parameter model, kita idealnya membutuhkan sekitar 20 Token data pelatihan.

**Analisis Tabel di atas:**
*   **GPT-3 (Rasio 1:1.7):** Menurut standar modern, GPT-3 sangat "kurang bacaan" (undertrained). Kapasitas otaknya besar, tapi bahannya kurang.
*   **LLaMA-65B (Rasio 1:21):** Ini mengikuti hukum Chinchilla dengan sempurna. Inilah sebabnya LLaMA-65B bisa mengalahkan GPT-3 meskipun ukuran modelnya (parameternya) 3x lebih kecil.
*   **Model Terbaru (Qwen/Llama-3):** Mereka sengaja melanggar batas ini (Overtrained) dengan memberikan data jauh lebih banyak (Rasio 1:200+). Tujuannya agar modelnya kecil (hemat RAM saat dijalankan di laptop/HP) tapi sangat cerdas.

Berikut adalah ilustrasi perbandingannya:

![Perbandingan Parameter dan Token](https://lms.sdmdigital.id/pluginfile.php/635386/mod_page/content/26/image31.png?time=1766635223790)
**Gambar 1. Ilustrasi perbandingan jumlah parameter dan token data latih yang digunakan oleh 4 model di atas**

Satuan token ini menggunakan token dengan tingkatan *subword* yang telah dijelaskan pada modul 2. Dapat dilihat bahwa data yang dibutuhkan mencapai skala milyar bahkan triliun token. Dengan menggunakan rasio kasar konversi 1 token sebagai ¾ kata, kita bisa mendapatkan estimasi jumlah kata yang diperlukan dalam dataset latih LLM sebagai berikut:

**Tabel 2. Estimasi jumlah kata dalam dataset latih dari beberapa LLM**

| Nama Model | Estimasi jumlah kata dalam dataset |
| :--- | :--- |
| GPT-3 | 225B |
| GLM | 300B |
| LLAMA | 1.05T |
| QWEN | 2.25T |

Untuk memberikan perbandingan relatif agar bisa lebih memahami skala dari dataset latih di atas, mari bandingkan dengan sesuatu yang sering ditemui, yaitu sebuah novel. Umumnya, sebuah novel terdiri dari sekitar 70.000 – 120.000 kata. Artinya, untuk melatih GPT-3, dibutuhkan dataset dengan jumlah kata yang ekivalen dengan sekitar 1.8 – 3.2 Juta novel.

---

### 4.1.4 Sumber Data Utama untuk Pretraining LLM : Internet

Internet telah menjadi media penyimpanan dan penyajian informasi yang sepanjang zaman menjadi semakin mudah diakses oleh semua orang. Internet menawarkan beberapa kelebihan yang menjadikannya menarik untuk menjadi sumber utama mendapatkan data untuk melatih LLM:

*   Internet banyak digunakan oleh manusia dan banyak berisi konten yang dibuat oleh manusia untuk dibaca oleh manusia lainnya, sehingga secara alami memiliki keragaman yang tinggi yang dapat memberikan pengetahuan yang lengkap untuk memodelkan bahasa manusia.
*   Internet memiliki skala yang sangat besar. Setiap tahun, tingkat pertambahan jumlah data di Internet terus meningkat, bahkan diproyeksikan mencapai skala 181 Petabytes pada akhir 2025.
*   Data di internet banyak yang tersedia secara publik dan bisa secara legal dikumpulkan secara otomatis menggunakan *web crawler*.
*   Banyak layanan publik / *open access* (Contoh: Internet Archive (archive.org): Menyimpan snapshot website sejak 1996) yang secara berkala melakukan pengarsipan konten dari situs tertentu, sehingga mudah diakses tanpa perlu melakukan *crawling* sendiri.
*   Selain konten yang berada di internet, banyak juga pengarsipan yang melakukan digitalisasi konten dalam bentuk analog seperti buku cetak, atau media analog lainnya yang disimpan dan dapat diakses melalui internet.

Berbagai macam kelebihan ini: **skala**, **keragaman**, dan **kemudahan pengaksesan** menjadikan internet sebagai salah satu sumber utama data latih LLM. Berikut merupakan beberapa dataset dari internet yang populer digunakan beserta skalanya:

**Tabel 3. Beberapa sumber data yang diambil dari internet**

| Dataset | Sumber | Tipe konten | Ukuran |
| :--- | :--- | :--- | :--- |
| **Common Crawl** | Hasil melakukan Web Crawling berkala oleh organisasi CommonCrawl.org | Halaman website dalam bentuk HTML | Mencapai skala Petabyte. |
| **C4** | Hasil pemrosesan lanjut dari Common Crawl | Data teks yang sudah dibersihkan | Sekitar 800GB. |
| **Wikipedia** | Berasal dari dump artikel wikipedia yang diarsip secara berkala | Artikel dalam bentuk teks | Untuk data bahasa inggris, ukuran ~100GB. |
| **Github** | Berasal dari arsip repository publik di Github | Kode pemrograman | ~6TB |
| **Stackoverflow** | Berasal dari arsip publik pada Archive.org | Pasangan pertanyaan dan jawaban + snippet kode | Berkisar puluhan hingga ratusan GB |
| **BookCorpus** | Hasil crawling dari website SmashWords | Buku dengan berbagai genre dalam format teks | 3-5 GB, sekitar 985M kata. |

---

### 4.1.5 Data Terlabel untuk Post Training

LLM juga memerlukan proses *training* tambahan yang membutuhkan data yang dilabeli oleh manusia untuk mengajarkan berbagai macam kemampuan serta perilaku spesifik.

Bentuk data yang dibutuhkan umumnya berupa interaksi dalam bahasa natural:
*   Untuk melatih LLM agar lebih baik mengikuti instruksi, diberikan pasangan instruksi-respon yang ideal.
*   Untuk melatih LLM menjawab pertanyaan, diberikan pasangan pertanyaan dan jawaban yang ideal.
*   Untuk melatih LLM melakukan kemampuan tertentu lainnya seperti kemampuan matematika atau *coding*, diberikan pasangan *problem-solution*.

Selain itu, diperlukan data yang merefleksikan preferensi pengguna (seperti *helpfulness* dan menghindari jawaban yang *harmful* atau *toxic*).

**Inter-Annotator Agreement (IAA)**
Salah satu tantangan utama adalah menjaga konsistensi penilaian antar manusia. Tanpa pengukuran yang jelas, label yang diberikan bisa jadi bias subjektif. Kita menggunakan metrik statistik:

**1. Cohen's Kappa ($\kappa$)**
*   **Kapan dipakai:** Jika Anda memiliki 2 Anotator yang menilai item yang sama.
*   **Cara kerja:** Menghitung kesepakatan sekaligus mengoreksi faktor kebetulan (*chance agreement*).

![Cohen's Kappa Formula](https://lms.sdmdigital.id/pluginfile.php/635386/mod_page/content/26/Screenshot%202025-12-27%20at%2021.25.25.png)
**Gambar: Formula Cohen's Kappa. Dimana $p_o$ adalah kesepakatan yang terobservasi, dan $p_e$ adalah kesepakatan hipotesis karena faktor kebetulan.**

**2. Fleiss' Kappa**
*   **Kapan dipakai:** Jika Anda memiliki 3 Anotator atau lebih.

> **Panduan Skor Kappa (Rule of Thumb):**
> *   **< 0.20:** Buruk (Poor).
> *   **0.21 - 0.60:** Sedang (Moderate).
> *   **0.61 - 0.80:** Bagus (Substantial). Standard minimum dataset LLM berkualitas.
> *   **0.81 - 1.00:** Sangat Bagus (Perfect).

Dalam beberapa tahun terakhir, muncul pendekatan untuk menghasilkan dataset secara sintetis menggunakan model lain (LLM-as-a-judge).

**Tabel 4. Detail mengenai beberapa dataset yang umum digunakan untuk post-training**

| Dataset | Keterangan | Skala |
| :--- | :--- | :--- |
| Alpaca | Pasangan instruksi dan jawabannya, di-generate menggunakan LLM | 52.000 pasangan |
| OpenAssistant | Percakapan dengan percabangan berbagai skenario | 66.487 percakapan |
| P3 | Berbagai macam task dalam NLP | 170 dataset + ~2000 template |
| HH-RLHF | Pesan user, 2 respon model, dan preferensi respon | 169K sampel |

**Catatan Mengenai Dataset Open Source:**
Penting untuk memastikan bahwa data mencerminkan kondisi dimana model akan digunakan. Dataset hukum AS berbeda konteks dengan hukum Indonesia. Namun, LLM memiliki kapabilitas *zero-shot* dan *in-context learning* yang baik untuk membantu proses adaptasi ini.

---

### Mini Quiz
*(Silakan akses quiz interaktif pada platform LMS)*

---

> ### Refleksi Singkat
> Meningkatkan skala jumlah parameter dan data terbukti bisa terus mencapai tingkatan performa yang lebih tinggi. Akan tetapi pada titik tertentu, akan semakin sulit mendapatkan data karena faktor "data exhaustion" (sumber publik sudah habis digunakan) atau meningkatnya konten **AI-generated** di internet yang menurunkan kualitas data latih.
>
> Coba cari tahu dan pahami masalah ini. Kira-kira apa langkah selanjutnya untuk terus meningkatkan performa? Berapa banyak lagi data yang dibutuhkan hingga kita melihat ada **diminishing returns**? Apakah LLM sekarang sudah menggunakan data yang ada secara semaksimal mungkin?