---
slug: ch-11-11
title: "Janus"
layout: chapter
---
## Bab: Janus - Mendekopel Pengodean Visual untuk Pemahaman dan Generasi Multimodal yang Terpadu

Dalam beberapa tahun terakhir, model besar multimodal telah mengalami kemajuan yang signifikan baik dalam domain pemahaman (*understanding*) maupun generasi (*generation*). Saudara-mahasiswa harus memahami bahwa dalam bidang pemahaman multimodal, para peneliti umumnya mengikuti desain **LLaVA**, yang menggunakan pengode visi (*vision encoder*) sebagai jembatan untuk memungkinkan Model Bahasa Besar (LLMs) memahami gambar. Sementara itu, dalam bidang generasi visual, pendekatan berbasis difusi telah meraih kesuksesan besar. Baru-baru ini, beberapa penelitian mulai mengeksplorasi metode autoregresif untuk generasi visi, mencapai performa yang sebanding dengan model difusi.

Untuk membangun model multimodal yang lebih kuat dan bersifat umum (*generalist*), para peneliti berusaha menggabungkan tugas pemahaman dan generasi multimodal. Sebagai contoh, beberapa studi mencoba menghubungkan model pemahaman multimodal dengan model difusi yang telah dilatih sebelumnya (seperti pada model **Emu**). Namun, secara ketat, pendekatan ini tidak dapat dianggap sebagai model yang benar-benar terpadu karena fungsionalitas generasi visual ditangani oleh model difusi eksternal, sementara LLM multimodal itu sendiri tidak memiliki kemampuan untuk menghasilkan gambar secara langsung.

Pendekatan lain menggunakan satu *Transformer* tunggal untuk menyatukan tugas pemahaman dan generasi multimodal. Hal ini meningkatkan kepatuhan instruksi (*instruction-following*) untuk generasi visual, membuka potensi kemampuan darurat (*emergent abilities*), dan mengurangi redundansi model. Metode tersebut biasanya menggunakan satu pengode visi tunggal untuk memproses masukan bagi kedua tugas tersebut. Namun, terdapat masalah fundamental: representasi yang dibutuhkan oleh pemahaman multimodal dan generasi visual sangat berbeda.

1.  **Pemahaman Multimodal:** Bertujuan mengekstrak informasi semantik tingkat tinggi (misalnya kategori objek atau atribut visual). Representasinya cenderung berfokus pada fitur semantik berdimensi tinggi.
2.  **Generasi Visual:** Fokus pada menghasilkan detail lokal dan menjaga konsistensi global gambar. Representasinya memerlukan pengodean berdimensi rendah yang mampu mengekspresikan struktur spasial halus dan detail tekstur.

Menyatukan representasi dari dua tugas ini dalam ruang yang sama akan menyebabkan konflik dan kompromi (*trade-offs*). Akibatnya, model terpadu yang ada saat ini sering kali mengorbankan performa pemahaman multimodal. Untuk mengatasi masalah ini, kami memperkenalkan **Janus**, sebuah kerangka kerja autoregresif yang mendekopel (*decouple*) pengodean visual menjadi jalur terpisah, namun tetap memanfaatkan satu arsitektur *Transformer* yang terpadu untuk pemrosesan.

---

**Deskripsi Gambar 1: Hasil Pemahaman Multimodal dan Generasi Visi dari Janus**

Gambar 1 merupakan representasi grafis dan visual dari kemampuan model Janus.
*   **Gambar 1 (a) Benchmark Performance:** Menyajikan grafik radar (*spider chart*) yang membandingkan Janus (garis ungu tebal) dengan model-model lain seperti MobileVLM, LLaVA-Phi, InstructBLIP, dan Show-o.
    *   **Sumbu Radar:** Mewakili berbagai tolok ukur (*benchmarks*) yaitu POPE, MME-Perception, MMBench, MM-Vet, SEED-Bench, GQA, MMMU, dan VQAv2.
    *   **Tren Data:** Terlihat jelas bahwa garis ungu Janus menempati area paling luar pada hampir semua sumbu, menunjukkan bahwa Janus melampaui model terpadu sebelumnya dan bahkan beberapa model spesifik-tugas dalam pemahaman multimodal.
*   **Gambar 1 (b) Visual Generation Results:** Menampilkan kisi (*grid*) gambar-gambar hasil generasi model Janus. Terlihat berbagai objek seperti mobil futuristik, burung hantu kristal, potret gaya lukisan klasik, hingga wajah manusia yang sangat detail.
*   **Kesimpulan Akademis:** Janus tidak hanya unggul dalam memahami gambar secara semantik, tetapi juga memiliki kemampuan generatif yang sangat kuat dengan resolusi gambar 384 × 384, membuktikan efektivitas dari strategi dekopel pengodean.

---

## 2. Pekerjaan Terkait (*Related Work*)

### 2.1. Generasi Visual
Generasi visual adalah bidang yang berkembang pesat yang menggabungkan konsep dari pemrosesan bahasa alami dengan kemajuan arsitektur *Transformer*. Model autoregresif memanfaatkan *Transformer* untuk memprediksi urutan token visual diskret (*codebook IDs*). Selain itu, model prediksi bermasker (*masked prediction*) menggunakan metode pemaskeran gaya **BERT** untuk meningkatkan efisiensi sintesis. Di sisi lain, model difusi kontinu menunjukkan kemampuan impresif dengan mendekati generasi melalui lensa probabilitas.

### 2.2. Pemahaman Multimodal
Model Bahasa Besar Multimodal (MLLMs) mengintegrasikan teks dan gambar. Dengan memanfaatkan LLM yang dilatih sebelumnya, MLLM menunjukkan kemampuan kuat untuk memproses informasi multimodal. Beberapa metode memperluas MLLM dengan model difusi untuk memfasilitasi pembuatan gambar, namun performanya sering kali inferior dibandingkan menggunakan model difusi secara mandiri karena ketergantungan pada alat eksternal.

### 2.3. Pemahaman dan Generasi Multimodal Terpadu
Model terpadu dianggap kuat untuk memfasilitasi penalaran dan generasi yang mulus di berbagai modalitas. Pendekatan tradisional biasanya menggunakan representasi visual tunggal untuk kedua tugas (seperti pada model **Chameleon** yang menggunakan *VQ Tokenizer*). Namun, hal ini menyebabkan hasil yang suboptimal karena pengode visi harus menghadapi kompromi antara kebutuhan pemahaman dan generasi. Janus secara eksplisit mendekopel representasi ini untuk mengenali bahwa tugas yang berbeda memerlukan tingkat informasi yang berbeda pula.

---

## 3. Janus: Kerangka Kerja Multimodal yang Sederhana, Terpadu, dan Fleksibel

### 3.1. Arsitektur

---

**Deskripsi Gambar 2: Arsitektur Janus**

Gambar 2 mengilustrasikan diagram alir arsitektur Janus yang dibagi menjadi dua bagian utama: **Understanding** (Kiri) dan **Image Generation** (Kanan).
*   **Blok Tengah:** Adalah *Auto-Regressive Transformer* yang menjadi otak terpadu bagi kedua proses.
*   **Jalur Pemahaman (Kiri):** Citra ($X_v$) masuk melalui **Und. Encoder** (Pengode Pemahaman), kemudian melalui adaptor sebelum masuk ke *Transformer*. Instruksi bahasa ($X_q$) masuk melalui *Text Tokenizer*. Keluarannya adalah *Language Response* ($X_a$) melalui *Text De-Tokenizer*.
*   **Jalur Generasi (Kanan):** Gambar ($X_v$) diproses oleh **Gen. Encoder** (Pengode Generasi). Instruksi bahasa ($X_q$) masuk melalui tokenizer. *Transformer* memprediksi token gambar yang kemudian diproses oleh *Image Decoder* untuk menghasilkan *Generated Image* ($X_v$).
*   **Kesimpulan Akademis:** Diagram ini mempertegas konsep utama Janus: penggunaan dua pengode visual yang berbeda (dikodekan dengan warna biru dan kuning) untuk melayani satu *Transformer* yang sama, menghindari konflik representasi.

---

Arsitektur Janus menerapkan metode pengodean independen:
1.  **Pemahaman Multimodal:** Menggunakan pengode **SigLIP** untuk mengekstrak fitur semantik berdimensi tinggi. Fitur ini diratakan (*flattened*) dari kisi 2-D menjadi urutan 1-D, lalu dipetakan ke ruang input LLM menggunakan adaptor pemahaman.
2.  **Generasi Visual:** Menggunakan **VQ Tokenizer** untuk mengonversi gambar menjadi ID diskret. Adaptor generasi kemudian memetakan *codebook embeddings* dari ID tersebut ke ruang input LLM.

Model ini sepenuhnya mematuhi kerangka kerja autoregresif tanpa memerlukan masker atensi (*attention masks*) yang dirancang khusus.

### 3.2. Prosedur Pelatihan

Pelatihan Janus dibagi menjadi tiga tahap:

**Tahap I: Melatih Adaptor dan Kepala Gambar (*Image Head*).**
Tujuan utamanya adalah menciptakan koneksi konseptual antara elemen visual dan linguistik dalam ruang penyisipan (*embedding space*). Pengode visual dan LLM dibekukan (*frozen*), hanya parameter dalam adaptor pemahaman, adaptor generasi, dan kepala gambar yang diperbarui.

**Tahap II: Prapelatihan Terpadu (*Unified Pretraining*).**
Pada tahap ini, dilakukan prapelatihan dengan korpus multimodal. LLM tidak lagi dibekukan. Model belajar dari data teks murni, data pemahaman multimodal, dan data generasi visual. Pelatihan generasi visual awal menggunakan *ImageNet-1k* untuk membantu model memahami dependensi piksel dasar, diikuti oleh data teks-ke-gambar umum.

**Tahap III: Pelatihan Halus Terselia (*Supervised Fine-Tuning* / SFT).**
Model dilatih halus dengan data penyelarasan instruksi untuk meningkatkan kemampuan dialog dan kepatuhan instruksi. Semua parameter kecuali pengode generasi dilatih halus. Digunakan campuran data dialog teks murni, pemahaman multimodal, dan generasi visual.

### 3.3. Objektif Pelatihan

Janus adalah model autoregresif, dan kami menggunakan fungsi kerugian entropi-silang (*cross-entropy loss*) selama pelatihan:

$$ \mathcal{L} = - \sum_{i=1}^{T} \log P_{\theta}(x_i | x_{<i}) \quad (1) $$

**Penjelasan Variabel Rumus (1):**
*   $\mathcal{L}$: Total Kerugian (*Loss*) yang ingin diminimalkan.
*   $T$: Panjang total urutan token.
*   $P_{\theta}$: Probabilitas kondisional yang dimodelkan oleh bobot jaringan $\theta$ dari Janus.
*   $x_i$: Token saat ini yang sedang diprediksi.
*   $x_{<i}$: Semua token sebelumnya dalam urutan (konteks).
*   **Logika:** Rumus ini mengharuskan model untuk memaksimalkan kemungkinan prediksi token berikutnya yang benar berdasarkan semua informasi sebelumnya.

---

**Deskripsi Gambar 3: Prosedur Pelatihan Tiga Tahap Janus**

Gambar 3 menunjukkan tiga kotak (Stage I, II, dan III) dengan simbol api (*flame*) dan kepingan salju (*snowflake*).
*   **Stage I:** Simbol salju pada *Und. Encoder*, *Gen. Encoder*, dan *LLM* (berarti dibekukan). Simbol api pada *Und. Adaptor*, *Gen. Adaptor*, dan *Image Head* (berarti dilatih).
*   **Stage II:** Simbol api muncul pada *LLM*, menunjukkan model mulai belajar secara mendalam dari data multimodal.
*   **Stage III:** Hampir semua bagian memiliki simbol api kecuali *Gen. Encoder* yang tetap dibekukan untuk menjaga integritas pengodean visual diskret.
*   **Kesimpulan:** Strategi ini memungkinkan pemanasan bertahap dari komponen-komponen model, memastikan kestabilan konvergensi.

---

### 3.4. Inferensi

Untuk generasi gambar, Janus memanfaatkan teknik *Classifier-Free Guidance* (CFG). Untuk setiap token, logit $l_g$ dihitung sebagai:

$$ l_g = l_u + s(l_c - l_u) $$

**Penjelasan Variabel Rumus (Konteks Inferensi):**
*   $l_g$: Logit akhir yang digunakan untuk pengambilan sampel.
*   $l_c$: Logit kondisional (berdasarkan instruksi teks).
*   $l_u$: Logit unikondisional (biasanya menggunakan token *padding* sebagai kondisi).
*   $s$: Skala panduan (*guidance scale*). Nilai *default* yang digunakan adalah 5.
*   **Logika:** Rumus ini memperkuat perbedaan antara prediksi dengan teks panduan dan prediksi tanpa panduan, sehingga menghasilkan gambar yang lebih sesuai dengan instruksi.

---

## 4. Eksperimen

### 4.1. Detail Implementasi
Janus menggunakan **DeepSeek-LLM (1.3B)** sebagai model bahasa dasar dengan panjang urutan maksimum 4096. Pengode visi untuk pemahaman adalah *SigLIP-Large-Patch16-384*. Adaptor menggunakan MLP dua lapisan. Seluruh proses pelatihan memakan waktu 7 hari pada kluster berisi 16 simpul, masing-masing dengan 8 GPU Nvidia A100.

**Tabel 1: Hiperparameter Detail Janus**

| Hiperparameter | Stage 1 | Stage 2 | Stage 3 |
| :--- | :---: | :---: | :---: |
| *Learning rate* | $1.0 \times 10^{-3}$ | $1 \times 10^{-4}$ | $2.0 \times 10^{-5}$ |
| *LR scheduler* | *Cosine* | *Constant* | *Constant* |
| *Weight decay* | 0.0 | 0.0 | 0.1 |
| *Batch size* | 256 | 512 | 256 |
| *Data Ratio (Und:Text:Gen)* | 1:0:1 | 2:3:5 | 7:3:10 |

---

### 4.4. Perbandingan dengan Model Mutakhir (*State-of-the-arts*)

**Tabel 2: Perbandingan pada Benchmark Pemahaman Multimodal**

| Model | Params | POPE | MME-P | MMB | SEED | MMMU |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| LLaVA-v1.5 | 7B | 85.9 | 1510.7 | 64.3 | 58.6 | 35.4 |
| Chameleon | 7B | - | - | - | - | 22.4 |
| Show-o | 1.3B | 73.8 | 948.4 | - | - | 25.1 |
| **Janus (Ours)** | **1.3B** | **87.0** | **1338.0** | **69.4** | **63.7** | **30.5** |

**Analisis Akademik:** Terlihat bahwa Janus dengan hanya 1,3 Miliar parameter mampu mengalahkan model LLaVA-v1.5 yang memiliki 7 Miliar parameter pada tolok ukur POPE (87,0 vs 85,9) dan MMBench (69,4 vs 64,3). Hal ini sangat impresif karena membuktikan bahwa efisiensi arsitektur dekopel jauh lebih penting daripada sekadar menambah jumlah parameter.

**Tabel 3: Evaluasi Generasi Teks-ke-Gambar pada GenEval**

| Metode | # Params | Single Obj | Two Obj | Counting | Overall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| SDXL | 2.6B | 0.98 | 0.74 | 0.39 | 0.55 |
| DALL-E 2 | 6.5B | 0.94 | 0.66 | 0.49 | 0.52 |
| Show-o | 1.3B | 0.95 | 0.52 | 0.49 | 0.53 |
| **Janus (Ours)** | **1.3B** | **0.97** | **0.68** | **0.30** | **0.61** |

**Analisis:** Janus mencapai akurasi keseluruhan (*Overall*) sebesar 0,61, melampaui SDXL (0,55) dan model terpadu Show-o (0,53), menunjukkan kemampuan kepatuhan instruksi yang superior.

---

### 4.5. Studi Ablasi (*Ablation Studies*)

Kami melakukan studi ablasi untuk memvalidasi pentingnya dekopel pengodean visual.

**Tabel 5: Dampak Dekopel Pengodean Visual**

| Exp ID | Visual Encoder | Task | POPE | MMB | MMMU | COCO-FID |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| A | VQ Tokenizer | Und+Gen | 60.1 | 35.0 | 24.7 | 8.72 |
| B | Semantic Tokenizer | Und+Gen | 82.4 | 52.7 | 26.6 | 7.11 |
| **D** | **SigLIP + VQ (Janus)** | **Und+Gen** | **87.0** | **69.4** | **30.5** | **8.53** |

**Kesimpulan Akademis Tabel 5:** Eksperimen A menggunakan satu pengode (VQ) untuk kedua tugas dan hasilnya sangat buruk pada pemahaman (POPE hanya 60.1). Eksperimen D (Janus) yang mendekopel pengodean memberikan lompatan performa yang masif pada tugas pemahaman tanpa mengorbankan kualitas generasi secara signifikan.

---

**Deskripsi Gambar 5: Hasil Kualitatif Pemahaman Multimodal pada Meme Humor**

Gambar 5 menunjukkan perbandingan antara Janus, Chameleon-7B, dan Show-o dalam menjelaskan dua meme.
*   **Meme 1 (Drake Meme):** Drake menolak "Turing Award" dan menerima "Nobel Prize in Physics".
    *   **Analisis Janus:** Menjelaskan dengan tepat bahwa ini adalah sindiran lucu tentang pemenang Nobel Fisika baru-baru ini yang berhubungan dengan AI (Hint: Geoffrey Hinton).
    *   **Kelemahan Model Lain:** Chameleon dan Show-o gagal menangkap konteks spesifik pemenang Nobel Fisika dan AI.
*   **Meme 2 (Doge Kuat vs Lemah):** Membandingkan "Decoupling Visual Encoding" (Doge Berotot) dengan "Single Visual Encoder" (Doge Kecil).
    *   **Analisis Janus:** Menjelaskan bahwa tipe pertama lebih kuat dan mampu menangani tugas kompleks, sementara tipe kedua lebih sederhana dan kurang bertenaga.
*   **Kesimpulan:** Janus memiliki pemahaman tingkat halus (*fine-grained*) dan kemampuan penalaran semantik yang jauh lebih baik dalam memahami konten gambar yang kompleks dan bernuansa budaya/humor.

---

## 5. Kesimpulan, Limitasi, dan Pekerjaan Masa Depan

### Kesimpulan
Dalam makalah ini, kami memperkenalkan **Janus**, sebuah model pemahaman dan generasi multimodal yang sederhana, terpadu, dan dapat diperluas. Ide inti dari Janus adalah untuk mendekopel pengodean visual untuk pemahaman dan generasi, yang dapat meredakan konflik yang timbul dari tuntutan berbeda yang ditempatkan pada pengode visual oleh kedua tugas tersebut. Eksperimen ekstensif telah menunjukkan efektivitas dan performa terkemuka dari Janus. Perlu dicatat bahwa Janus bersifat fleksibel dan mudah untuk diperluas ke modalitas input lainnya di masa depan.

### Limitasi (*Limitation*)
Meskipun menunjukkan performa hebat, Janus masih memiliki batasan:
1.  **Resolusi Gambar:** Saat ini masih terbatas pada resolusi 384 × 384, yang mungkin kehilangan beberapa detail mikro pada dokumen teks yang sangat padat.
2.  **Akumulasi Kesalahan:** Sebagai model autoregresif, terdapat potensi akumulasi kesalahan selama proses generasi visual yang panjang, meskipun CFG telah membantu memitigasi hal ini.

### Pekerjaan Masa Depan (*Future Work*)
1.  **Dukungan Modalitas Tambahan:** Mengintegrasikan pengode untuk awan titik 3D (*3D point cloud*), data taktil, dan sinyal EEG.
2.  **Resolusi Dinamis:** Menggunakan teknik resolusi tinggi dinamis untuk memungkinkan model menskala ke resolusi gambar apa pun tanpa interpolasi posisi.
3.  **Efisiensi Komputasi:** Mengeksplorasi metode kompresi token visual yang lebih agresif (seperti operasi *pixel shuffle*) untuk menghemat biaya komputasi tanpa mengurangi kualitas representasi.

*(Tamat)*
