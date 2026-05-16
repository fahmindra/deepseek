---
slug: ch-11-28
title: "DeepSeek-V4"
layout: chapter
---
## DeepSeek-V4: Menuju Intelijensi Konteks Sejuta Token yang Sangat Efisien

Munculnya model penalaran (*reasoning models*) telah menetapkan paradigma baru dalam penskalaan waktu-uji (*test-time scaling*), yang mendorong peningkatan performa substansial bagi Model Bahasa Besar atau *Large Language Models* (LLMs). Namun, paradigma penskalaan ini secara fundamental dibatasi oleh kompleksitas komputasi kuadratik dari mekanisme atensi (*attention*) konvensional. Kompleksitas ini menciptakan hambatan besar bagi konteks yang sangat panjang dan proses penalaran yang mendalam.

Secara bersamaan, munculnya skenario tugas jangka panjang (*long-horizon tasks*)—mulai dari alur kerja agen cerdas yang kompleks hingga analisis lintas dokumen masif—membuat dukungan efisien untuk konteks ultra-panjang menjadi sangat krusial bagi kemajuan masa depan. Bab ini akan membedah **DeepSeek-V4**, sebuah seri model bahasa berbasis *Mixture-of-Experts* (MoE) yang dirancang untuk memecahkan hambatan efisiensi tersebut. Seri ini mencakup **DeepSeek-V4-Pro** dengan 1,6 Triliun parameter (49 Miliar parameter aktif) dan **DeepSeek-V4-Flash** dengan 284 Miliar parameter (13 Miliar parameter aktif), di mana keduanya mendukung panjang konteks hingga satu juta token.

Melalui inovasi arsitektur seperti *hybrid attention* yang menggabungkan *Compressed Sparse Attention* (CSA) dan *Heavily Compressed Attention* (HCA), serta penggunaan pengoptimal *Muon*, DeepSeek-V4 mencapai lompatan drastis dalam efisiensi komputasi. Sebagai contoh, dalam pengaturan konteks satu juta token, DeepSeek-V4-Pro hanya membutuhkan 27% FLOPs inferensi per token dan 10% ukuran *KV cache* dibandingkan dengan pendahulunya, DeepSeek-V3.2.

---

**Deskripsi Gambar 1:**
Gambar 1 terdiri dari dua bagian utama: sebuah grafik batang di sisi kiri dan dua grafik garis di sisi kanan.

*   **Gambar 1 (Kiri) - Performa Tolok Ukur:**
    *   **Sumbu-X:** Menampilkan kategori evaluasi seperti *Knowledge & Reasoning* (SimpleQA, HLE, Apex Shortlist, Codeforces, SWE Verified) dan *Agentic Capabilities* (Terminal Bench 2.0, Toolathlon).
    *   **Sumbu-Y:** Akurasi atau skor Pass@1 (%).
    *   **Legenda:** Warna biru gelap (DeepSeek-V4-Pro-Max), abu-abu (Claude-Opus-4.6-Max), abu-abu muda (GPT-5.4-xHigh), dan biru muda (Gemini-3.1-Pro-High).
    *   **Tren:** DeepSeek-V4-Pro-Max menunjukkan performa yang dominan, terutama pada tugas penalaran tingkat tinggi seperti *Codeforces* (skor 3206) dan *SimpleQA Verified* (skor 57,9), melampaui model-model tertutup papan atas.

*   **Gambar 1 (Kanan Atas) - FLOPs Inferensi per Token:**
    *   **Sumbu-X:** Posisi Token dalam ribuan (0 hingga 1024K).
    *   **Sumbu-Y:** Single-Token Inference FLOPs (T).
    *   **Kurva:** Garis putus-putus abu-abu (DeepSeek-V3.2), garis biru solid (DeepSeek-V4-Pro), dan garis biru muda (DeepSeek-V4-Flash).
    *   **Tren:** Saat konteks memanjang, biaya komputasi V3.2 naik secara tajam. Sebaliknya, kurva V4-Pro dan V4-Flash tetap jauh lebih rendah. Terdapat label yang menunjukkan bahwa V4-Pro memiliki biaya **3.7x lebih rendah** dan V4-Flash **9.8x lebih rendah** dibandingkan V3.2 pada titik 1 juta token.

*   **Gambar 1 (Kanan Bawah) - Ukuran KV Cache Terakumulasi:**
    *   **Sumbu-X:** Panjang Runtutan dalam ribuan (0 hingga 1024K).
    *   **Sumbu-Y:** Ukuran KV Cache Terakumulasi (GB).
    *   **Tren:** Mirip dengan grafik FLOPs, penggunaan memori KV Cache untuk seri V4 jauh lebih efisien. Label menunjukkan penghematan sebesar **9.5x lebih kecil** untuk V4-Pro dan **13.7x lebih kecil** untuk V4-Flash dibandingkan V3.2.

---

## 2. Arsitektur

Secara keseluruhan, DeepSeek-V4 mempertahankan arsitektur *Transformer* dan modul *Multi-Token Prediction* (MTP) dari DeepSeek-V3, namun memperkenalkan peningkatan kunci untuk memperkuat stabilitas dan efisiensi.

### 2.1. Desain yang Diwarisi dari DeepSeek-V3

Model ini tetap mengadopsi paradigma **DeepSeekMoE** untuk Jaringan Saraf Maju (*Feed-Forward Networks* / FFNs). Inovasi kecil dilakukan pada fungsi aktivasi untuk menghitung skor afinitas, yang diubah dari fungsi *Sigmoid* menjadi:
$$ \text{Sqrt}(\text{Softplus}(\cdot)) $$
Hal ini bertujuan untuk menstabilkan distribusi skor tanpa mengorbankan dinamika pemilihan pakar (*expert routing*). Untuk penyeimbangan beban (*load balancing*), digunakan strategi *auxiliary-loss-free* yang ditambah dengan *sequence-wise balance loss* guna mencegah ketidakseimbangan ekstrem dalam satu rangkaian data.

### 2.2. Manifold-Constrained Hyper-Connections (mHC)

DeepSeek-V4 mengintegrasikan **Manifold-Constrained Hyper-Connections (mHC)** untuk memperkuat koneksi residual konvensional antar blok Transformer. Ide dasarnya adalah membatasi pemetaan residual ke dalam manifol (*manifold*) tertentu untuk meningkatkan stabilitas propagasi sinyal.

Dalam *Hyper-Connections* (HC) standar, lebar aliran residual diperluas dari $\mathbb{R}^d$ ke $\mathbb{R}^{n_{hc} \times d}$, di mana $d$ adalah dimensi tersembunyi dan $n_{hc}$ adalah faktor ekspansi. Jika $X_l = [x_{l,1}; \dots; x_{l,n_{hc}}]^T \in \mathbb{R}^{n_{hc} \times d}$ adalah status residual sebelum lapisan ke-$l$, maka pembaruan status residual dirumuskan sebagai:

$$ X_{l+1} = B_l X_l + C_l \mathcal{F}_l(A_l X_l) \quad (1) $$

**Penjelasan Variabel Rumus (1):**
*   $X_{l+1}$: Status residual untuk lapisan berikutnya.
*   $A_l \in \mathbb{R}^{1 \times n_{hc}}$: Pemetaan *input* untuk menggabungkan status residual yang diperluas menjadi dimensi-$d$ sebelum masuk ke lapisan $\mathcal{F}_l$.
*   $B_l \in \mathbb{R}^{n_{hc} \times n_{hc}}$: Transformasi residual yang menentukan bagaimana informasi dari langkah sebelumnya dipertahankan.
*   $C_l \in \mathbb{R}^{n_{hc} \times 1}$: Pemetaan *output* yang mendistribusikan kembali hasil fungsi lapisan ke dalam aliran residual yang diperluas.
*   $\mathcal{F}_l$: Fungsi lapisan (misalnya, lapisan MoE atau Atensi).

Inovasi utama dalam **mHC** adalah membatasi matriks pemetaan residual $B_l$ agar berada dalam manifol **matriks stokastik ganda** (Birkhoff polytope) $\mathcal{M}$:

$$ B_l \in \mathcal{M} := \{ M \in \mathbb{R}^{n \times n} \mid M \mathbf{1}_n = \mathbf{1}_n, \mathbf{1}_n^T M = \mathbf{1}_n^T, M \ge 0 \} \quad (2) $$

Batasan ini memastikan bahwa norma spektral $\|B_l\|_2$ dibatasi oleh 1, sehingga transformasi residual bersifat non-ekspansif. Hal ini secara drastis meningkatkan stabilitas numerik selama fase pelatihan maju (*forward*) maupun mundur (*backpropagation*).

Untuk implementasinya, digunakan algoritma **Sinkhorn-Knopp**:
$$ M^{(t)} = \mathcal{T}_r(\mathcal{T}_c(M^{(t-1)})) \quad (8) $$
Di mana $\mathcal{T}_r$ dan $\mathcal{T}_c$ adalah normalisasi baris dan kolom. Iterasi dilakukan sebanyak $t_{max} = 20$ kali untuk konvergensi.

---

**Deskripsi Gambar 2:**
Gambar ini menunjukkan diagram arsitektur blok Transformer DeepSeek-V4 secara vertikal.
*   Paling bawah adalah *Input Tokens* yang masuk ke lapisan *Embedding*.
*   Aliran data dibagi menjadi dua jalur utama dalam setiap blok: jalur *Residual* (kiri) dan jalur komputasi (kanan).
*   Di dalam satu blok, terdapat komponen **CSA / HCA** untuk mekanisme atensi dan **DeepSeekMoE** untuk lapisan *Feed-Forward*.
*   Di antara blok-blok tersebut, terdapat unit **Residual Mixing**, **Pre-Block Mixing**, dan **Post-Block Mixing** yang merepresentasikan implementasi mHC.
*   Aliran berakhir di *Prediction Head* yang menghasilkan *LM Loss* (kerugian model bahasa), dan modul *MTP* yang menghasilkan *MTP Loss*.

---

### 2.3. Hybrid Attention dengan CSA dan HCA

Untuk mengatasi hambatan komputasi pada konteks ultra-panjang, DeepSeek-V4 merancang dua arsitektur atensi yang efisien:

#### 2.3.1. Compressed Sparse Attention (CSA)

CSA mengompresi jumlah entri *Key-Value* (KV) sebanyak $1/m$ kali, lalu menerapkan strategi *DeepSeek Sparse Attention* (DSA).

**Entri Key-Value Terkompresi:**
Jika $H \in \mathbb{R}^{n \times d}$ adalah urutan status tersembunyi masukan, CSA menghitung dua seri entri KV, $C^a, C^b$ dan bobot kompresi $Z^a, Z^b$. Setiap $m$ entri KV dikompresi menjadi satu entri berdasarkan bobot tersebut:

$$ [s^a_{mi:m(i+1)-1}; s^b_{m(i-1):mi-1}] = \text{Softmax}_{row}([Z^a_{mi:m(i+1)-1} + B^a; Z^b_{m(i-1):mi-1} + B^b]) \quad (11) $$

$$ C_i^{Comp} = \sum_{j=mi}^{m(i+1)-1} s_j^a \odot C_j^a + \sum_{j=m(i-1)}^{mi-1} s_j^b \odot C_j^b \quad (12) $$

**Penjelasan Variabel Rumus (11 & 12):**
*   $m$: Faktor kompresi (misal: 4).
*   $Z$: Bobot kompresi yang dipelajari.
*   $B$: Bias posisional yang dapat dipelajari (*learnable positional biases*).
*   $\odot$: Produk Hadamard (perkalian elemen-per-elemen).
*   $C_i^{Comp}$: Entri KV yang telah dikompresi, mewakili informasi dari rentang token tertentu.

**Lightning Indexer untuk Seleksi Jarang (*Sparse Selection*):**
Untuk memilih entri KV mana yang akan diperhatikan oleh kueri (*query*), digunakan *Lightning Indexer* dengan proyeksi peringkat rendah:
$$ I_{t,s} = \sum_{h=1}^{n_h^I} w_{t,h}^I \cdot \text{ReLU}(q_{t,h}^I \cdot K_s^{IComp}) \quad (16) $$

**Penjelasan Variabel Rumus (16):**
*   $I_{t,s}$: Skor indeks antara token kueri $t$ dan blok terkompresi $s$.
*   $n_h^I$: Jumlah kepala (*heads*) pada indeks.
*   $q_{t,h}^I$ dan $K_s^{IComp}$: Kueri indeks dan kunci indeks terkompresi.

---

**Deskripsi Gambar 3:**
Gambar ini mengilustrasikan alur kerja arsitektur CSA.
*   Status tersembunyi token KV masuk ke *Token-Level Compressor*.
*   Hasilnya berupa *Compressed KV Entries* yang kemudian masuk ke *Top-k Selector*.
*   Di sisi lain, *Query Token* masuk ke *Lightning Indexer* untuk menghasilkan *Index Scores*.
*   *Top-k Selector* memilih entri KV terkompresi yang paling relevan berdasarkan skor tersebut.
*   Output akhirnya adalah *Shared Key-Value Multi-Query Attention* yang menggabungkan entri terpilih dengan *Sliding Window KV Entries* (untuk ketergantungan lokal).

---

#### 2.3.2. Heavily Compressed Attention (HCA)

HCA melakukan kompresi yang lebih agresif (faktor $m' \gg m$) namun tetap menggunakan atensi padat (*dense attention*). Alurnya mirip dengan CSA tetapi tanpa mekanisme seleksi jarang, sehingga setiap kueri memperhatikan semua entri yang telah dikompresi secara masif tersebut.

---

**Deskripsi Gambar 4:**
Gambar 4 menunjukkan arsitektur HCA yang lebih sederhana dibandingkan CSA. Tidak ada *Lightning Indexer* atau *Top-k Selector*. Status tersembunyi KV langsung dikompresi oleh *Token-Level Compressor* menjadi *Heavily Compressed KV Entries*, yang kemudian langsung diproses melalui mekanisme *Shared Key-Value Multi-Query Attention*.

---

### 2.4. Pengoptimal Muon (*Muon Optimizer*)

DeepSeek-V4 menggunakan pengoptimal **Muon** untuk mayoritas modul karena konvergensi yang lebih cepat dan stabilitas yang lebih baik. Algoritma intinya melibatkan ortogonalisasi matriks pembaruan menggunakan iterasi **Newton-Schulz**:

$$ M_k = aM_{k-1} + b(M_{k-1}M_{k-1}^T)M_{k-1} + c(M_{k-1}M_{k-1}^T)^2M_{k-1} \quad (28) $$

**Penjelasan Variabel Rumus (28):**
*   $M_k$: Matriks pembaruan pada iterasi ke-$k$.
*   $a, b, c$: Koefisien yang disesuaikan dalam dua tahap (tahap awal untuk konvergensi cepat, tahap akhir untuk stabilitas pada nilai singular 1).

---

## 3. Infrastruktur Umum

### 3.1. Overlap Komunikasi-Komputasi pada Paralelisme Pakar

Dalam arsitektur MoE, komunikasi antar-simpul (*inter-node*) sering menjadi hambatan. DeepSeek-V4 mengusulkan skema *Expert Parallelism* (EP) berbutir halus (*fine-grained*) yang menyatukan komunikasi dan komputasi ke dalam satu *pipelined kernel*.

---

**Deskripsi Gambar 5:**
Gambar ini membandingkan tiga solusi penanganan MoE:
*   **(a) Naive Solution:** Langkah *Dispatch* (komunikasi), *L1* (komputasi), *Act* (aktivasi), *L2* (komputasi), dan *Combine* (komunikasi) dilakukan secara sekuensial.
*   **(b) Comet:** Melakukan tumpang tindih (*overlap*) antara *Dispatch* dengan *L1*, dan *L2* dengan *Combine*. Memberikan percepatan teoritis 1.42x.
*   **(c) Ours (DeepSeek-V4):** Membagi pakar menjadi gelombang-gelombang (*waves*). Komputasi gelombang saat ini, pengiriman token untuk gelombang berikutnya, dan pengiriman hasil pakar yang sudah selesai dilakukan secara bersamaan. Memberikan percepatan teoritis **1.92x**.

---

### 3.2. Pelatihan Sadar-Kuantisasi FP4 (*FP4 Quantization-Aware Training*)

Untuk efisiensi inferensi, DeepSeek-V4 menerapkan kuantisasi **FP4 (MXFP4)** pada bobot pakar MoE dan jalur *Query-Key* (QK) pada indeks CSA. Selama pelatihan pasca-prapelatihan (*post-training*), bobot master FP32 dikuantisasi ke FP4, lalu didekuantisasi kembali ke FP8 untuk komputasi. Proses ini memastikan model beradaptasi dengan degradasi presisi sejak tahap pelatihan.

---

## 4. Prapelatihan (*Pre-Training*)

### 4.1. Konstruksi Data

Korpus prapelatihan terdiri dari lebih dari **32 Triliun token**. Fokus utama diberikan pada:
*   Penyaringan konten web yang dibuat secara otomatis (*auto-generated*) untuk mencegah keruntuhan model.
*   Peningkatan porsi data matematika dan pemrograman.
*   Kurasi dokumen panjang seperti makalah ilmiah dan laporan teknis.
*   Data multibahasa yang mencakup pengetahuan *long-tail* dari berbagai budaya.

### 4.2. Pengaturan Model dan Evaluasi

Tabel di bawah merangkum konfigurasi dua model utama:

| Spesifikasi | DeepSeek-V4-Flash | DeepSeek-V4-Pro |
| :--- | :---: | :---: |
| Jumlah Lapisan Transformer | 43 | 61 |
| Dimensi Tersembunyi ($d$) | 4096 | 7168 |
| Total Parameter | 284B | 1,6T |
| Parameter Aktif (per token) | 13B | 49B |
| Faktor Kompresi CSA ($m$) | 4 | 4 |
| Faktor Kompresi HCA ($m'$) | 128 | 128 |

**Tabel 1: Hasil Evaluasi Model Dasar (*Base Models*):**
Evaluasi menunjukkan bahwa DeepSeek-V4-Pro-Base mendominasi hampir semua kategori dibandingkan DeepSeek-V3.2. Sebagai contoh, pada *MMLU*, V4-Pro meraih skor **90,1** dibandingkan V3.2 yang meraih 87,8. Pada konteks panjang (*LongBench-V2*), V4-Pro unggul jauh dengan skor **51,5** (V3.2 hanya 40,2).

---

## 5. Pelatihan Pasca-Prapelatihan (*Post-Training*)

### 5.1. Alur Kerja Post-Training

Siklus ini melibatkan dua tahap:
1.  **Pelatihan Spesialis:** Melatih model pakar domain (matematika, kode, instruksi) secara independen menggunakan *Supervised Fine-Tuning* (SFT) dan *Reinforcement Learning* (RL) dengan algoritma **GRPO**.
2.  **On-Policy Distillation (OPD):** Menggabungkan kemampuan para pakar ke dalam satu model tunggal melalui distilasi distribusi *output* dari banyak model guru ke model siswa.

Fungsi objektif OPD dirumuskan sebagai:
$$ \mathcal{L}_{OPD}(\theta) = \sum_{i=1}^N w_i \cdot D_{KL}(\pi_\theta \| \pi_{E_i}) \quad (29) $$

**Penjelasan Variabel Rumus (29):**
*   $\mathcal{L}_{OPD}$: Kerugian distilasi.
*   $w_i$: Bobot yang diberikan untuk setiap pakar domain $i$.
*   $D_{KL}$: Divergensi Kullback-Leibler, mengukur perbedaan antara distribusi probabilitas model siswa ($\pi_\theta$) dan model pakar guru ($\pi_{E_i}$).

### 5.2. Upaya Penalaran (*Reasoning Efforts*)

DeepSeek-V4 mendukung tiga mode penalaran:
*   **Non-think:** Cepat, intuitif, untuk tugas harian rutin. Format: `</think> Ringkasan`.
*   **Think High:** Analisis logis sadar, lebih akurat namun lebih lambat. Format: `<think> token pemikiran </think> Ringkasan`.
*   **Think Max:** Penalaran maksimal tanpa jalan pintas untuk mengeksplorasi batas kemampuan model. Menggunakan instruksi sistem khusus yang mewajibkan model melakukan uji-tekan (*stress-testing*) terhadap logikanya sendiri.

---

**Deskripsi Gambar 9:**
Gambar ini menunjukkan performa seri DeepSeek-V4 pada tugas **MRCR 8-needle** (uji pencarian fakta di dalam tumpukan konteks yang sangat panjang).
*   **Sumbu-X:** Jumlah token input (8K hingga 1024K).
*   **Sumbu-Y:** Rata-rata MMR (*Mean Reciprocal Rank*).
*   **Kurva:** Bintang biru (DeepSeek-V4-Pro-Max) dan segitiga biru (DeepSeek-V4-Flash-Max).
*   **Tren:** Hingga jendela konteks 128K, kedua model menunjukkan performa yang sangat stabil (mendekati skor 0,90). Setelah 128K, terjadi degradasi performa secara bertahap, namun pada 1024K (1 juta token), V4-Pro-Max masih mempertahankan skor yang kuat di atas 0,59, jauh lebih baik daripada model pesaing dalam skenario dunia nyata.

---

**Deskripsi Gambar 10:**
Gambar ini menyajikan dua grafik yang menunjukkan hubungan antara performa dan jumlah token pemikiran yang dihasilkan.
*   **Grafik Kiri (HLE):** Menunjukkan bahwa semakin banyak model "berpikir" (sumbu-X: *Total Tokens*), semakin tinggi skor akurasinya (sumbu-Y). Kurva DeepSeek-V4-Pro (biru tua) berada di atas V4-Flash dan V3.2, mencapai titik puncak pada mode "Max".
*   **Grafik Kanan (TerminalBench 2.0):** Menunjukkan tren serupa untuk kemampuan agen cerdas. Mode "Max" pada V4-Pro mencapai skor tertinggi (di atas 65%), membuktikan bahwa penskalaan waktu-uji (*test-time scaling*) melalui generasi token penalaran meningkatkan keberhasilan penyelesaian tugas secara signifikan.

---

## 6. Kesimpulan, Limitasi, dan Pekerjaan Masa Depan

### 6.1. Kesimpulan
DeepSeek-V4 telah membuktikan bahwa hambatan efisiensi dalam pemrosesan konteks sejuta token dapat dipecahkan melalui inovasi arsitektur *hybrid attention* (CSA dan HCA) serta optimisasi infrastruktur yang mendalam. Model DeepSeek-V4-Pro-Max saat ini mendefinisikan standar baru untuk model sumber terbuka, dengan kemampuan penalaran dan pengetahuan yang mampu bersaing ketat dengan model tertutup paling mutakhir seperti GPT-4 dan Gemini 3.1 Pro.

### 6.2. Limitasi
Meskipun arsitektur ini sangat efektif, penggunaan komponen yang kompleks dan baru secara preliminari membuat arsitektur keseluruhan menjadi relatif rumit. Selain itu, meskipun teknik seperti *Anticipatory Routing* dan *SwiGLU Clamping* terbukti efektif meredam ketidakstabilan pelatihan, prinsip fundamental di baliknya belum sepenuhnya dipahami secara teoritis. Model ini juga masih rentan terhadap kesalahan kecil atau misinterpretasi terhadap perintah yang sangat ambigu.

### 6.3. Pekerjaan Masa Depan
Iterasi masa depan akan difokuskan pada:
1.  **Penyederhanaan Arsitektur:** Melakukan investigasi prinsipil untuk menyaring desain yang paling esensial agar lebih elegan tanpa mengorbankan performa.
2.  **Sparsitas Model:** Mengeksplorasi dimensi baru seperti modul *embedding* yang lebih jarang (*sparse*) untuk meningkatkan efisiensi memori.
3.  **Konteks Ultra-Panjang:** Terus meneliti arsitektur latensi rendah agar penyebaran dan interaksi pada konteks jutaan token menjadi lebih responsif dan praktis untuk penggunaan harian.

*(Tamat)*
