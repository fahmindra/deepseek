---
slug: ch-11-12
title: "DeepSeek-VL2"
layout: chapter
---
## Bab: DeepSeek-VL2: Model Visi-Bahasa Berbasis *Mixture-of-Experts* untuk Pemahaman Multimodal Tingkat Lanjut

Selamat datang di materi kuliah mengenai Model Visi-Bahasa atau *Vision-Language Models* (VLMs). Dalam beberapa tahun terakhir, VLMs telah muncul sebagai kekuatan transformatif dalam Kecerdasan Buatan (AI). Model-model ini memperluas kapabilitas luar biasa dari Model Bahasa Besar atau *Large Language Models* (LLMs) untuk memproses informasi visual dan tekstual secara mulus. Saudara-mahasiswa harus memahami bahwa kemajuan ini secara drastis memperluas potensi sistem AI untuk menangani aplikasi dunia nyata yang kompleks yang memerlukan pemahaman multimodal.

Dalam diktat ini, kita akan membedah secara mendalam **DeepSeek-VL2**, sebuah seri baru dari model visi-bahasa sumber terbuka yang memanfaatkan arsitektur **Campuran Para Ahli** (*Mixture-of-Experts* / MoE). Model ini mencapai peningkatan substansial dalam performa dan efisiensi dibandingkan pendahulunya, DeepSeek-VL. Kemajuan ini berpusat pada tiga aspek kunci: (1) strategi pengodean visi beresolusi tinggi yang dinamis, (2) arsitektur model bahasa yang dioptimalkan untuk efisiensi pelatihan dan inferensi, dan (3) pipa konstruksi data visi-bahasa yang diperhalus untuk mendorong kapabilitas baru seperti *visual grounding* (penambatan visual) yang presisi.

---

## 1. Analisis Kinerja Awal

Sebelum kita masuk ke detail teknis, mari kita perhatikan tren efisiensi model melalui **Gambar 1**.

**Deskripsi Gambar 1: Performa Rata-rata vs Parameter yang Diaktifkan**
Gambar ini menyajikan grafik garis yang mengilustrasikan perbandingan antara performa model (Sumbu-Y) dan jumlah parameter yang diaktifkan dalam satuan miliar (Sumbu-X).
*   **Sumbu-X (Horizontal):** Merepresentasikan *Activated Parameters* dari 0 hingga 10 Miliar.
*   **Sumbu-Y (Vertikal):** Merepresentasikan *Average Performance* (Akurasi rata-rata dari berbagai tolok ukur seperti MMBench, MMMU, MathVista, dll.) dalam rentang 48 hingga 72.
*   **Kurva dan Titik Data:** 
    *   Garis putus-putus merah dengan simbol bintang melambangkan keluarga **DeepSeek-VL2** (Tiny, Small, dan standar).
    *   Garis hijau melambangkan keluarga **InternVL2**.
    *   Garis oranye melambangkan keluarga **Qwen2-VL**.
    *   Titik biru tunggal melambangkan model **Phi-3.5-Vision**.
*   **Tren Grafik:** Terlihat bahwa kurva DeepSeek-VL2 berada di posisi paling atas kiri. Ini menunjukkan bahwa dengan jumlah parameter aktif yang sama (misalnya sekitar 4.5B), DeepSeek-VL2 mencapai skor rata-rata yang lebih tinggi (mendekati 70) dibandingkan kompetitornya.
*   **Kesimpulan Akademis:** DeepSeek-VL2 membuktikan keunggulan arsitektur MoE-nya. Model ini mampu memberikan "kecerdasan" yang setara atau lebih tinggi dibandingkan model *dense* (padat) namun dengan beban komputasi (parameter aktif) yang jauh lebih rendah.

---

## 2. Arsitektur Model

Arsitektur DeepSeek-VL2 terdiri dari tiga modul inti yang terintegrasi secara sistematis: (1) pengode visi (*vision encoder*), (2) adaptor visi-bahasa, dan (3) model bahasa berbasis MoE. Saudara dapat melihat gambaran umumnya pada **Gambar 2**.

**Deskripsi Gambar 2: Ikhtisar DeepSeek-VL2**
Gambar ini menunjukkan diagram alir data dalam model.
*   Di bagian bawah, terdapat gambar masukan yang diproses melalui **Dynamic Tiling** (Pengubinan Dinamis).
*   Data visual kemudian masuk ke **Vision Encoder** dan diteruskan ke **VL Adaptor**.
*   Secara paralel, instruksi teks ("*Describe this image in detail*") masuk ke sistem.
*   Semua informasi (token gambar dan token teks) digabungkan dan dimasukkan ke dalam blok **DeepSeek-MoE**.
*   Keluaran akhirnya adalah prediksi teks yang menjelaskan gambar tersebut.

### 2.1. Strategi Pengubinan Dinamis (*Dynamic Tiling Strategy*)

Masalah utama pada pendahulu model ini adalah batasan resolusi tetap (seperti 1024x1024). DeepSeek-VL2 memperkenalkan pengubinan dinamis untuk memproses gambar beresolusi tinggi dengan rasio aspek yang bervariasi secara efisien.

Algoritma ini bekerja dengan membagi gambar besar menjadi ubin-ubin (*tiles*). Kami menggunakan pengode visi **SigLIP-SO400M-384** yang beroperasi pada resolusi dasar 384x384. Untuk mengakomodasi berbagai rasio aspek, kami mendefinisikan set resolusi kandidat $C_R$:

$$ C_R = \{ (m \cdot 384, n \cdot 384) \mid m \in \mathbb{N}, n \in \mathbb{N}, 1 \le m, n, m \cdot n \le 9 \} $$

**Penjelasan Variabel Rumus:**
*   $m, n$: Bilangan bulat positif yang mewakili pengali jumlah ubin pada dimensi lebar dan tinggi.
*   $384$: Resolusi dasar per ubin dalam piksel.
*   $m \cdot n \le 9$: Batasan total ubin maksimal yang diizinkan adalah 9 ubin untuk menjaga efisiensi jumlah token.
*   $C_R$: Himpunan semua kemungkinan konfigurasi resolusi yang bisa dipilih model.

Sistem akan memilih konfigurasi $(m_i \cdot 384, n_i \cdot 384)$ yang meminimalkan area pengisi (*padding*) saat gambar diubah ukurannya. Selain ubin lokal, kami juga menyertakan satu ubin *thumbnail* global yang mencakup keseluruhan gambar. Jadi, total ubin yang diproses adalah $(1 + m_i \times n_i)$. Setiap ubin menghasilkan $27 \times 27 = 729$ penyisipan visual (*visual embeddings*).

### 2.2. Adaptor Visi-Bahasa (*Vision-Language Adaptor*)

Setelah ubin diproses, model menggunakan operasi *pixel shuffle* 2x2. Operasi ini mengompresi token visual dari 27x27 menjadi 14x14 (196 token) untuk setiap ubin. 

Langkah-langkah penyusunan sequence visual adalah sebagai berikut:
1.  Ubin *thumbnail* global (210 token setelah ditambah token baris baru).
2.  Token pemisah pandangan (`<view_separator>`).
3.  Ubin-ubin lokal yang disusun dalam kisi 2D.
4.  Penambahan token baris baru (`<tile_newline>`) di setiap akhir baris ubin untuk mempertahankan pemahaman struktur spasial.

Proses ini diilustrasikan secara visual pada **Gambar 3**.

**Deskripsi Gambar 3: Ilustrasi Strategi Pengubinan Dinamis**
Gambar ini menunjukkan proses transformasi dari gambar asli menjadi urutan token.
*   Gambar asli dipotong-potong menjadi *Local Tiles*.
*   Masing-masing ubin dikodekan dan digabungkan (*Encode & Merge*).
*   Data visual "diratakan" (*Flatten*) menjadi satu garis sekuensial.
*   Ditunjukkan posisi token khusus seperti **sep** (pemisah) dan **\n** (baris baru) di antara blok-blok token gambar.
*   **Kesimpulan Akademis:** Strategi ini memungkinkan LLM untuk memahami posisi relatif antar bagian gambar seolah-olah ia sedang membaca teks dengan baris-baris baru.

### 2.3. LLM DeepSeekMoE

Komponen bahasa didasarkan pada arsitektur **DeepSeekMoE**. Inovasi krusial di sini adalah penggunaan mekanisme **Atensi Laten Multi-kepala** (*Multi-head Latent Attention* / MLA).

MLA meningkatkan efisiensi inferensi dengan mengompresi *cache* Kunci-Nilai (*Key-Value cache*) menjadi vektor laten yang lebih kecil. Hal ini memungkinkan *throughput* (laju keluaran) yang lebih tinggi tanpa mengorbankan kualitas atensi model terhadap konteks yang panjang. Selama pelatihan MoE, digunakan istilah bias global untuk menyeimbangkan beban antar pakar (*experts*) secara efektif.

DeepSeek-VL2 tersedia dalam tiga varian ukuran yang dirinci pada **Tabel 1**.

| Spesifikasi | DeepSeek-VL2-Tiny | DeepSeek-VL2-Small | DeepSeek-VL2 |
| :--- | :---: | :---: | :---: |
| Ukuran Kosakata | 129,280 | 102,400 | 129,280 |
| Dimensi Penyisipan | 1,280 | 2,048 | 2,560 |
| Jumlah Lapisan | 12 | 27 | 30 |
| Atensi | Multi-Head | MLA (rank=512) | MLA (rank=512) |
| Pakar Terurut (*Routed*) | 64 | 64 | 72 |
| Pakar Bersama (*Shared*) | 2 | 2 | 2 |
| Parameter Aktif | 1.0B | 2.8B | 4.5B |

*Tabel 1: Spesifikasi Arsitektur DeepSeek-VL2.*

---

## 3. Konstruksi Data

Kualitas model AI sangat ditentukan oleh data pelatihannya. DeepSeek-VL2 menggunakan tiga tahap konstruksi data:

1.  **Penyelarasan Visi-Bahasa (*VL Alignment*):** Fokus pada melatih adaptor MLP menggunakan 1,2 juta sampel dari ShareGPT4V.
2.  **Prapelatihan Visi-Bahasa (*VL Pretraining*):** Menggabungkan 70% data visi-bahasa dengan 30% data teks saja. Komponen datanya meliputi:
    *   Data gambar-teks berselang-seling (*interleaved*).
    *   Data keterangan gambar (*captioning*) yang diperbaiki kualitasnya menggunakan DeepSeek Chat.
    *   Data OCR (Optical Character Recognition) berskala besar (12 juta RenderedText).
    *   Data pemahaman dokumen, tabel, dan bagan (*chart*).
3.  **Pelatihan Halus Terselia (*Supervised Fine-Tuning* / SFT):** Menggunakan data instruksi berkualitas tinggi, termasuk penalaran matematis dan logika.

---

## 4. Metodologi Pelatihan

Pelatihan dilakukan dalam tiga tahap sekuensial:
1.  **Tahap 1:** Melatih adaptor visi-bahasa sementara pengode visi dan LLM dibekukan (*frozen*).
2.  **Tahap 2:** Membuka kunci LLM dan adaptor untuk prapelatihan bersama.
3.  **Tahap 3:** Pelatihan halus pada semua parameter untuk meningkatkan kapabilitas instruksi dan percakapan.

Detail hiperparameter pelatihan dapat dilihat pada **Tabel 2**.

| Hiperparameter | Tahap 1 | Tahap 2 | Tahap 3 |
| :--- | :---: | :---: | :---: |
| *Learning Rate* | $5.4 \times 10^{-4}$ | $5.4 \times 10^{-4}$ | $3.0 \times 10^{-5}$ |
| *Optimizer* | AdamW | AdamW | AdamW |
| *Batch Size* | 256 | 2304 | 64 |
| *Sequence Length* | 4096 | 4096 | 4096 |

*Tabel 2: Konfigurasi Hiperparameter (Contoh untuk varian Tiny).*

---

## 5. Evaluasi

### 5.1. Performa Multimodal

DeepSeek-VL2 dievaluasi pada berbagai tolok ukur akademik. Hasilnya pada tugas terkait OCR (pembacaan teks pada gambar) ditunjukkan pada **Tabel 3**.

| Model | Params (Aktif) | DocVQA | ChartQA | TextVQA | OCRBench |
| :--- | :---: | :---: | :---: | :---: | :---: |
| GPT-4V | - | 87.2 | 78.1 | 78.0 | 645 |
| InternVL2-2B | 2.2B | 86.9 | 76.2 | 73.4 | 784 |
| **DeepSeek-VL2-Tiny** | **1.0B** | **88.9** | **81.0** | **80.7** | **809** |
| **DeepSeek-VL2** | **4.5B** | **93.3** | **86.0** | **84.2** | **811** |

*Tabel 3: Perbandingan pada tolok ukur OCR.*

**Analisis Akademik:** Perhatikan bahwa DeepSeek-VL2-Tiny dengan hanya 1 Miliar parameter aktif mampu melampaui InternVL2-2B dan bahkan bersaing dengan model tertutup yang jauh lebih besar dalam tugas DocVQA dan ChartQA. Ini menunjukkan bahwa strategi pengubinan dinamis sangat efektif untuk menangkap detail teks kecil pada dokumen.

### 5.2. Studi Kualitatif

Model ini menunjukkan kemampuan luar biasa dalam memahami nuansa budaya dan humor. Saudara dapat melihat contohnya pada **Gambar 5**.

**Deskripsi Gambar 5: Kapabilitas Pemahaman Meme**
Gambar ini menampilkan beberapa meme kompleks. 
*   **Contoh 1:** Meme tentang mahasiswa PhD yang "bekerja di pantai" namun pikirannya tetap dipenuhi beban riset (diwakili oleh tumpukan kertas virtual). Model mampu menjelaskan kontras antara lingkungan santai dan beban mental yang dialami subjek.
*   **Contoh 2:** Meme tentang anak kecil yang dilarang memakan kue namun tetap memakannya. Model mengidentifikasi tema "ketidakpatuhan" dan "humor dalam kontras".
*   **Kesimpulan Akademis:** Model tidak hanya mengenali objek, tetapi juga mampu melakukan penalaran tingkat tinggi mengenai konteks sosial dan emosional yang terkandung dalam sebuah gambar.

DeepSeek-VL2 juga memperkenalkan kapabilitas **Visual Grounding** (Gambar 8), di mana model dapat memberikan koordinat kotak pembatas (*bounding box*) untuk objek tertentu yang diminta oleh pengguna, baik dalam adegan alami maupun antarmuka perangkat lunak (*GUI perception*).

---

## 6. Kesimpulan, Limitasi, dan Pekerjaan Masa Depan

### 6.1. Kesimpulan
DeepSeek-VL2 menandai lompatan besar dalam pemodelan Visi-Bahasa berskala besar. Melalui penggabungan arsitektur MoE yang efisien, mekanisme MLA yang hemat memori, dan strategi pengubinan dinamis, model ini mencapai performa *state-of-the-art* dengan biaya komputasi yang relatif rendah. Model ini sangat unggul dalam pemahaman visual tingkat halus, OCR, dan analisis dokumen.

### 6.2. Limitasi (*Limitations*)
Walaupun DeepSeek-VL2 menunjukkan kapabilitas yang kuat, terdapat beberapa batasan yang perlu diperhatikan:
*   **Jendela Konteks Gambar:** Saat ini, jendela konteks model hanya memungkinkan beberapa gambar per sesi obrolan.
*   **Ketahanan terhadap Objek Tak Terlihat:** Seperti VLMs lainnya, model terkadang kesulitan menghadapi objek yang sangat jarang atau gambar yang sangat kabur.
*   **Penalaran Logis:** Meskipun unggul dalam persepsi, penguatan pada aspek penalaran murni (*reasoning*) masih terus dikembangkan.

### 6.3. Pekerjaan Masa Depan (*Future Work*)
Arah riset selanjutnya akan difokuskan pada:
1.  Memperluas jendela konteks untuk mendukung interaksi multi-gambar yang lebih kaya.
2.  Meningkatkan ketahanan model terhadap distorsi gambar dan variasi domain yang ekstrem.
3.  Memperkuat kapabilitas penalaran logis yang lebih dalam agar model tidak hanya "melihat" tapi juga "berpikir" secara lebih komprehensif mengenai hubungan kausalitas dalam data visual.

***

*(Tamat)*
