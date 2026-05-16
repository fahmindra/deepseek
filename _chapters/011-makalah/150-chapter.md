---
slug: ch-11-15
title: "Janus-Pro"
layout: chapter
---
## Janus-Pro - Penyatuan Pemahaman dan Pembangkitan Multimodal dengan Penskalaan Data dan Model

Dalam bab ini, kita akan mempelajari **Janus-Pro**, sebuah versi tingkat lanjut dari model pendahulunya, yaitu Janus. Secara spesifik, Janus-Pro menginkorporasikan tiga pilar utama:
1. **Strategi pelatihan yang dioptimalkan** (*an optimized training strategy*).
2. **Ekspansi data pelatihan** (*expanded training data*).
3. **Penskalaan ke ukuran model yang lebih besar** (*scaling to larger model size*).

Dengan perbaikan-perbaikan fundamental ini, Janus-Pro mencapai kemajuan yang sangat signifikan baik dalam **pemahaman multimodal** (*multimodal understanding*) maupun kapabilitas **pembangkitan teks-ke-gambar berbasis instruksi** (*text-to-image instruction-following capabilities*), sembari secara bersamaan meningkatkan stabilitas dari pembangkitan teks-ke-gambar tersebut. Diharapkan karya ilmiah ini akan menginspirasi eksplorasi lebih lanjut di ranah kecerdasan buatan multimodal. Kode dan model telah tersedia untuk publik.

---

## 1. Pendahuluan

Sebelum kita menyelam ke dalam arsitektur, mari kita analisis performa makro dari model ini melalui visualisasi grafis. Perhatikan dengan saksama **Gambar 1**.

### Analisis Akademis: Gambar 1 - Hasil Pemahaman Multimodal dan Pembangkitan Visual dari Janus-Pro

Gambar 1 terdiri dari dua sub-grafik yang menyajikan perbandingan komprehensif antara Janus-Pro dan model-model mutakhir lainnya.

**Gambar 1 (a) - Rata-rata performa pada empat tolok ukur pemahaman multimodal (*Average performance on four multimodal understanding benchmarks*)**
*   **Sumbu X (Horizontal):** Merepresentasikan **Parameter LLM dalam Miliar** (*LLM Parameters (Billions)*), berskala dari 0 hingga sekitar 12 Miliar. Ini menunjukkan kapasitas ukuran model.
*   **Sumbu Y (Vertikal):** Merepresentasikan **Performa Rata-rata** (*Average Performance*) dalam skala 46 hingga 64. Nilai ini didapat dengan merata-ratakan akurasi dari empat tolok ukur: POPE, MME-Perception (skornya dibagi 20 agar berskala 0-100), GQA, dan MMMU.
*   **Titik Data dan Kurva:**
    *   Terdapat garis putus-putus berwarna merah (*red dashed line*) yang menghubungkan keluarga Janus (Unified Model).
    *   Terdapat garis putus-putus berwarna hijau muda yang menghubungkan keluarga LLaVA (Understanding Only).
    *   Terdapat garis putus-putus berwarna ungu yang menghubungkan keluarga Show-o (Unified Model).
    *   **Bintang Merah:** Titik dengan lambang bintang merah adalah model kita, **Janus-Pro-1B** (di kisaran 2 miliar parameter) dan **Janus-Pro-7B** (di kisaran 7 miliar parameter).
*   **Kesimpulan:** Grafik ini secara elegan mendemonstrasikan bahwa Janus-Pro-7B (Bintang merah tertinggi) berada pada puncak performa pemahaman (Y-axis tertinggi) mengungguli model-model khusus pemahaman seperti LLaVA-v1.5-7B, apalagi model terpadu (*unified*) lainnya seperti Show-o. Ini membuktikan efisiensi parameter dari arsitektur Janus-Pro.

**Gambar 1 (b) - Performa pada tolok ukur pematuhan-instruksi untuk pembangkitan teks-ke-gambar (*Performance on instruction-following benchmarks for text-to-image generation*)**
*   **Sumbu Y (Vertikal):** Menunjukkan **Akurasi (%)** (*Accuracy (%)*) dari 0 hingga 100.
*   **Sumbu X (Horizontal):** Menampilkan dua metrik evaluasi utama: **GenEval** dan **DPG-Bench**.
*   **Diagram Batang (*Bar Chart*):** Setiap kelompok batang membandingkan model yang berbeda.
    *   Kuning pastel: `SDXL` (Stable Diffusion XL) dan `SDv1.5`.
    *   Krem: `PixArt-alpha` dan `DALL-E 3`.
    *   Biru muda: `SD3-Medium` dan `Emu3-Gen`.
    *   Biru tua/ungu: `Janus` (versi lama) dan **`Janus-Pro-7B`**.
*   **Kesimpulan:** Pada GenEval, Janus-Pro-7B (batang biru paling kanan) mencetak akurasi 80.0%, melibas semua model difusi raksasa termasuk DALL-E 3 (74.0%) dan SD3-Medium (67.0%). Pola dominasi absolut yang sama terlihat pada DPG-Bench di mana Janus-Pro-7B menyentuh angka 84.2%. Ini menegaskan superioritas model *autoregressive* Janus-Pro dalam memahami *prompt* (perintah teks) yang padat dan mengubahnya menjadi gambar yang akurat.

---

### Analisis Kualitatif: Gambar 2 - Perbandingan Pembangkitan Teks-ke-Gambar

Untuk memahami dampak peningkatan ini pada hasil visual, mari kita perhatikan **Gambar 2**.

*   **Deskripsi Visual:** Gambar ini menyajikan perbandingan *side-by-side* (berdampingan) antara hasil *render* gambar dari model **Janus** (lama) versus **Janus-Pro-7B** (baru) untuk lima *prompt* teks yang berbeda. Resolusi gambar keluaran adalah $384 \times 384$.
*   *Prompt 1 ("The face of a beautiful girl"):* Wajah yang dihasilkan Janus-Pro tampak jauh lebih fotorealistik dengan tekstur kulit dan pencahayaan yang lebih baik.
*   *Prompt 2 ("A steaming cup of coffee on a wooden table"):* Asap pada Janus-Pro terlihat lebih natural dan material kayunya lebih tajam.
*   *Prompt 3 ("A glass of red wine on a reflective surface"):* Pantulan (*reflection*) pada gelas dan cairan pada Janus-Pro diproses secara lebih akurat secara fisika optik.
*   *Prompt 4 ("A minimalist photo of an orange tangerine..."):* Janus-Pro berhasil meletakkan jeruk di atas sutra merah sesuai instruksi, sementara versi lama terlihat kurang proporsional.
*   *Prompt 5 ("A clear image of a blackboard... 'Hello' written precisely..."):* Ini adalah tugas krusial yaitu tipografi/teks di dalam gambar. Janus gagal menuliskan "Hello" (menghasilkan coretan tak bermakna), sedangkan Janus-Pro berhasil menulis kata "Hello" dengan sangat rapi menggunakan kapur putih.
*   *Prompt 6 ("Capture a close-up shot of a vibrant sunflower..."):* Detail kelopak dan lebah pada Janus-Pro jauh lebih tajam dan kaya.
*   **Kesimpulan:** Visualisasi ini membuktikan bahwa Janus-Pro memberikan hasil (*outputs*) yang jauh lebih stabil untuk *prompt* pendek, kualitas visual yang melonjak tajam, detail yang lebih kaya, dan yang paling penting, kapabilitas untuk merender teks sederhana (seperti kata "Hello") secara tipografis dengan tepat.

***

Kemajuan terbaru dalam model terpadu untuk pemahaman dan pembangkitan multimodal (*unified multimodal understanding and generation models*) telah menunjukkan progres yang sangat signifikan [30, 40, 45, 46, 48, 50, 54, 55]. Pendekatan-pendekatan ini telah terbukti mampu meningkatkan kapabilitas pematuhan-instruksi (*instruction-following capabilities*) dalam tugas-tugas pembangkitan visual, sembari menekan redundansi model. 

Namun, mari kita cermati masalah fundamentalnya: sebagian besar dari metode-metode ini menggunakan **penyandi visual yang sama** (*the same visual encoder*) untuk memproses input baik untuk tugas pemahaman maupun pembangkitan multimodal. Karena representasi (*representations*) atau fitur vektor yang dibutuhkan untuk kedua jenis tugas ini pada dasarnya berbeda secara sifat, pemaksaan penggunaan satu *encoder* ini sering kali berujung pada performa yang suboptimal pada fungsi pemahaman multimodal.

Untuk mengatasi paradoks ini, arsitektur **Janus** [46] sebelumnya mengusulkan konsep **pemisahan penyandian visual** (*decoupling visual encoding*). Pemisahan ini meredakan konflik representasional antara pemahaman dan pembangkitan, sehingga memungkinkan pencapaian performa yang sangat baik pada kedua front tugas tersebut.

Sebagai model pionir, Janus awalnya hanya divalidasi pada skala parameter 1 Miliar (1B). Sayangnya, karena keterbatasan volume data pelatihan dan kapasitas (*capacity*) model yang relatif kecil, Janus versi awal masih memperlihatkan beberapa kelemahan nyata, seperti:
1.  Performa yang kurang optimal saat diminta membangkitkan gambar dari *prompt* (instruksi) yang sangat pendek.
2.  Kualitas pembangkitan teks-ke-gambar (*text-to-image generation*) yang sering tidak stabil.

Dalam karya ilmiah ini, kami memperkenalkan **Janus-Pro**, versi pengembangan dari Janus yang menginkorporasikan perbaikan pada tiga dimensi esensial: **strategi pelatihan (*training strategies*)**, **volume data (*data*)**, dan **ukuran model (*model size*)**. Seri Janus-Pro mencakup dua varian ukuran: 1B dan 7B, yang secara empiris mendemonstrasikan skalabilitas yang sangat baik dari metode penyandian/pendekodean visual yang terpisah tersebut.

Kami mengevaluasi Janus-Pro pada berbagai tolok ukur (*benchmarks*), dan hasilnya secara konsisten menyingkap kapabilitas pemahaman multimodal yang superior serta peningkatan dramatis pada performa pematuhan-instruksi untuk teks-ke-gambar. Secara spesifik:
*   **Janus-Pro-7B** meraih skor impresif sebesar **79.2** pada tolok ukur pemahaman multimodal **MMBench** [29], menaklukkan model-model multimodal terpadu berstatus *state-of-the-art* lainnya seperti Janus asli [46] (69.4), TokenFlow [34] (68.9), dan MetaMorph [42] (75.2).
*   Lebih jauh lagi, pada papan peringkat pematuhan-instruksi teks-ke-gambar **GenEval** [14], Janus-Pro-7B mencetak skor **0.80**, jauh meninggalkan Janus [46] (0.61), DALL-E 3 (0.67), dan Stable Diffusion 3 Medium [11] (0.74).

---

## 2. Metodologi (*Method*)

### 2.1 Arsitektur (*Architecture*)

Mari kita bedah anatomi dari Janus-Pro melalui **Gambar 3**.

#### Analisis Akademis: Gambar 3 - Arsitektur Janus-Pro

*   **Deskripsi Visual:** Gambar ini merepresentasikan diagram blok sistem yang menguraikan aliran data melalui komponen-komponen *neural network*. Blok utama di tengah adalah **Auto-Regressive Transformer**. Aliran dipisahkan menjadi dua bagian vertikal:
    *   **Kiri (Pemahaman / *Understanding*):** Memproses gambar menjadi teks. Terdapat balok biru muda `Und. Encoder` (Penyandi Pemahaman) yang mengonversi `Image: X_v` (Gambar Input) menjadi vektor, yang digabungkan dengan `Text Tokenizer` (pemroses teks `Language Instruct: X_q`). Hasil perhitungannya keluar ke atas melalui `Text De-Tokenizer` menghasilkan `Language Response: X_a` (Teks Jawaban).
    *   **Kanan (Pembangkitan / *Image Generation*):** Mengubah teks menjadi gambar. Terdapat balok kuning muda `Text Tokenizer` (`Language Instruct: X_q`) yang dimasukkan ke *Transformer*. Hasilnya masuk ke balok kuning lain `Gen. Encoder` untuk dipadukan dengan `Image: X_v`. Hasil akhir dari atas memancar keluar dari `Image Decoder` menghasilkan gambar akhir `Generated Image: X_v`.
*   **Kesimpulan:** Inti dari desain ini divisualisasikan dengan sangat jelas: terdapat **Dua Jalur Input Visual yang Berbeda** (ditandai dengan warna biru dan kuning di bagian bawah). Penyandian visual secara tegas "dipisahkan" (*decoupled*) antara tugas membaca gambar (pemahaman) dan tugas menggambar (pembangkitan).

Arsitektur Janus-Pro secara esensial identik dengan Janus [46]. Prinsip rancangan utamanya adalah untuk memisahkan penyandian visual antara pemahaman dan pembangkitan multimodal. Kami menerapkan metode penyandian independen untuk mengonversi input mentah menjadi fitur (*features*), yang selanjutnya akan diproses oleh sebuah *transformer autoregressive* yang terpadu (*unified*).

1.  **Untuk Pemahaman Multimodal:** Kami menggunakan penyandi **SigLIP** [53] untuk mengekstraksi fitur semantik berdimensi tinggi dari citra. Fitur-fitur ini, yang awalnya berbentuk *grid* 2-Dimensi (2-D), diratakan (*flattened*) menjadi sebuah sekuens (urutan) 1-Dimensi (1-D). Kemudian, sebuah **adaptor pemahaman (*understanding adaptor*)** bertugas memetakan (*mapping*) fitur-fitur citra tersebut agar masuk ke dalam ruang input (dimensi matematis) yang dipahami oleh LLM.
2.  **Untuk Tugas Pembangkitan Visual:** Kami memanfaatkan **VQ tokenizer** (Penyandi Kuantisasi Vektor) dari [38] untuk mengonversi citra fisik menjadi identitas diskrit (*discrete IDs*). Setelah sekuens ID tersebut diratakan menjadi 1-D, kami memakai sebuah **adaptor pembangkitan (*generation adaptor*)** untuk memetakan penyematan buku-kode (*codebook embeddings*) yang berkorespondensi dengan setiap ID agar selaras dengan ruang input LLM.

Sekuens-sekuens fitur ini kemudian digabungkan (*concatenated*) untuk membentuk sebuah **sekuens fitur multimodal tunggal**, yang akhirnya diumpankan ke dalam LLM untuk pemrosesan lebih lanjut. 

Selain kepala prediksi (*prediction head*) bawaan dari LLM (untuk teks), kami juga mendeploy sebuah kepala prediksi yang diinisialisasi secara acak untuk memprediksi token citra dalam tugas pembangkitan visual. Keseluruhan model ini tunduk patuh pada kerangka kerja *autoregressive* murni (memprediksi token satu per satu dari kiri ke kanan).

---

### 2.2 Strategi Pelatihan yang Dioptimalkan (*Optimized Training Strategy*)

Versi terdahulu (Janus) menggunakan proses pelatihan tiga tahap (*three-stage training process*):
*   **Tahap I (Stage I):** Fokus melatih adaptor dan *image head* (kepala gambar).
*   **Tahap II (Stage II):** Menangani prapelatihan terpadu (*unified pretraining*). Di sini, semua komponen diperbarui parameternya KECUALI *understanding encoder* dan *generation encoder*.
*   **Tahap III (Stage III):** Pelatihan Halus Terselia (*Supervised Fine-Tuning* / SFT). Tahap ini membuka kunci (*unlocking*) parameter *understanding encoder* agar bisa ikut dilatih.

Namun, strategi lama ini menyisakan beberapa kecacatan sistemik. Pada Tahap II, Janus awalnya membagi pelatihan untuk kemampuan teks-ke-gambar menjadi dua sub-bagian (mengikuti metode PixArt [4]). Sub-bagian pertama melatih model menggunakan data **ImageNet** [9], yang sekadar menggunakan nama-nama kategori objek sebagai *prompt* (misal: "kucing", "mobil") dengan tujuan primitif sekadar memodelkan ketergantungan piksel gambar. Sub-bagian kedua barulah menggunakan data teks-ke-gambar normal (deskripsi kalimat). Secara kuantitatif, Janus menghamburkan 66,67% dari total langkah pelatihan Tahap II-nya hanya untuk memproses data ImageNet tersebut. Eksperimen lanjutan kami membuktikan bahwa strategi ini sangat memboroskan komputasi secara sia-sia (*suboptimal and inefficient*).

Untuk memberantas masalah inefisiensi tersebut, kami mendesain **dua modifikasi radikal**:

1.  **Pelatihan yang Lebih Panjang di Tahap I (*Longer Training in Stage I*):** Kami menambah jumlah langkah komputasi pada Tahap I, membiarkannya "memakan" dataset ImageNet secara ekstensif. Temuan kami sangat mengejutkan: bahkan dengan kondisi parameter LLM inti yang **dibekukan (*fixed*)**, model sudah mampu mempelajari ketergantungan piksel secara efektif dan membangkitkan gambar berdasar nama kategori yang masuk akal.
2.  **Pelatihan yang Terfokus di Tahap II (*Focused Training in Stage II*):** Konsekuensi dari poin pertama, di Tahap II, kami **membuang sama sekali** data ImageNet. Tahap II kini 100% menggunakan data teks-ke-gambar dengan deskripsi yang padat (*dense descriptions*). Perombakan ini membuat Tahap II jauh lebih "fokus" mencerna semantik visual tingkat tinggi, yang meledakkan efisiensi pelatihan dan melambungkan performa akhir (*overall performance*).

Selain itu, pada Tahap III (SFT), kami mengatur ulang rasio pencampuran data. Proporsi antara **data multimodal, data teks murni, dan data teks-ke-gambar** diubah dari `7:3:10` menjadi `5:1:4`. Dengan mengurangi sedikit porsi data teks-ke-gambar, kami secara empiris mendapati bahwa model tetap menjaga kualitas pembangkitan visual yang kuat, sembari mengalami perbaikan masif dalam hal pemahaman analitis multimodal.

---

### 2.3 Penskalaan Data (*Data Scaling*)

Kami secara drastis mendongkrak skala (jumlah) data pelatihan untuk Janus-Pro, baik di sektor pemahaman maupun pembangkitan.

*   **Pemahaman Multimodal (*Multimodal Understanding*):** Untuk data prapelatihan Tahap II, kami merujuk pada pangkalan data **DeepSeek-VL2** [49] dan menginjeksi sekitar **90 juta sampel tambahan**. Ini mencakup:
    *   Dataset Keterangan Gambar (*image caption*) seperti YFCC [31].
    *   Data analisis tabel, grafik, dan dokumen (seperti Docmatix [20]). 
    *   Untuk Tahap III (SFT), kami juga meminjam dataset tambahan dari DeepSeek-VL2, mencakup data pemahaman *MEME* komedi, percakapan bahasa Mandarin, dan data simulasi dialog. Suplemen data ini secara masif meregangkan spektrum pengetahuan model untuk merespons percakapan manusiawi.
*   **Pembangkitan Visual (*Visual Generation*):** Kami menginspeksi bahwa data dunia nyata (foto internet) yang dipakai di versi Janus lama sangatlah kotor (*noisy*). Ini menyebabkan output teks-ke-gambar sering kali tidak estetis atau ambyar secara visual. Solusinya? Untuk Janus-Pro, kami mensintesis **72 juta sampel gambar estetis buatan komputer (*synthetic aesthetic data*)**. Kami menyeimbangkan rasio data *real* dan sintetis menjadi **1:1** selama tahap prapelatihan terpadu. *Prompt* untuk data sintetis ini diambil secara publik, merujuk pada [43]. 
    *Fakta Akademis:* Eksperimen membuktikan bahwa model bahasa mengalami *convergence* (pembelajaran optimal) secara lebih cepat saat disuapi data sintetis yang rapi. Hasil gambar akhirnya bukan cuma stabil secara struktur, melainkan melonjak secara estetika.

---

### 2.4 Penskalaan Model (*Model Scaling*)

Janus versi lama telah memvalidasi betapa saktinya konsep "pemisahan penyandian visual" menggunakan model kecil 1.5B parameter. Pada Janus-Pro, kami melakukan penskalaan komputasi hingga menyentuh skala **7 Miliar (7B) parameter LLM**. Hiperparameter detail dari 1.5B maupun 7B bisa Anda periksa di Tabel 1. 

Temuan terpenting dari penskalaan ini adalah: ketika kami menggunakan LLM berukuran raksasa, kurva *loss* (tingkat kesalahan komputasi matematika) merosot dan berkonvergensi dengan kecepatan yang jauh lebih buas dibandingkan model kecil, baik untuk tugas pemahaman maupun pembangkitan. Ini menahbiskan bahwa arsitektur ini memiliki daya skalabilitas tingkat atas (*strong scalability*).

### Tabel 1: Konfigurasi Arsitektural untuk Janus-Pro
*(Tabel ini menguraikan spesifikasi rangka mesin neural dari kedua varian)*

| Parameter | Janus-Pro-1B | Janus-Pro-7B |
| :--- | :---: | :---: |
| **Vocabulary size** (Kapasitas Kosakata) | 100K | 100K |
| **Embedding size** (Ukuran Vektor Sematan) | 2048 | 4096 |
| **Context Window** (Batas Panjang Memori) | 4096 | 4096 |
| **#Attention heads** (Kepala Atensi) | 16 | 32 |
| **#Layers** (Kedalaman Jaringan) | 24 | 30 |

### Tabel 2: Detail Hiperparameter untuk Pelatihan Janus-Pro
*(Tabel ini menjabarkan rasio cairan data dan metrik pembelajaran, sebuah peta jalan penting untuk mereplikasi hasil riset ini).*

| Hiperparameter | Tahap 1 (1B) | Tahap 2 (1B) | Tahap 3 (1B) | Tahap 1 (7B) | Tahap 2 (7B) | Tahap 3 (7B) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Learning rate** | $1.0 \times 10^{-3}$ | $1.0 \times 10^{-4}$ | $4.0 \times 10^{-5}$ | $1.0 \times 10^{-3}$ | $1.0 \times 10^{-4}$ | $4.0 \times 10^{-5}$ |
| **LR scheduler** | Constant | Constant | Constant | Constant | Constant | Constant |
| **Weight decay** | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| **Gradient clip** | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| **Optimizer** | AdamW ($\beta_1=0.9, \beta_2=0.95$) | AdamW | AdamW | AdamW | AdamW | AdamW |
| **Warm-up steps** | 600 | 5000 | 0 | 600 | 5000 | 0 |
| **Training steps** | 20K | 360K | 80K | 20K | 360K | 40K |
| **Batch size** | 256 | 512 | 128 | 256 | 512 | 128 |
| **Data Ratio** | 1:0:3 | 2:3:5 | 5:1:4 | 1:0:3 | 2:3:5 | 5:1:4 |

*(Catatan Dosen: Perhatikan baris Data Ratio. Pada Tahap 1, model murni belajar dari data gambar (ratio 1:0:3). Di Tahap 2 dan 3, rasio data teks (tengah) dan pemahaman multimodal (kiri) mulai disuntikkan secara presisi).*

---

## 3. Eksperimen (*Experiments*)

### 3.1 Detail Implementasi (*Implementation Details*)

Sebagai "otak bahasa" sentral (*base language model*), kami menggunakan **DeepSeek-LLM** versi 1.5B dan 7B [3] yang batas memori bawaannya berada di 4096 token. 
*   Sebagai "Mata Pemahaman" (*vision encoder untuk understanding*), kami memilih model **SigLIP-Large-Patch16-384** [53]. 
*   Sebagai "Tangan Peng
