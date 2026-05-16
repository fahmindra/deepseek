---
slug: ch-11-06
title: "DeepSeek-V2"
layout: chapter
---

## DeepSeek-V2: Model Bahasa *Mixture-of-Experts* yang Tangguh, Ekonomis, dan Efisien

Dalam bab pendahuluan ini, kita diperkenalkan dengan **DeepSeek-V2**, sebuah model bahasa berbasis *Mixture-of-Experts* (MoE) atau "Campuran Para Pakar" yang dikarakterisasikan oleh biaya pelatihan yang sangat ekonomis dan efisiensi inferensi yang luar biasa. Model ini memiliki total **236 Miliar (236B) parameter**, namun yang membedakannya dengan model padat (*dense model*) adalah: **hanya 21 Miliar (21B) parameter yang diaktifkan untuk setiap token**. Selain itu, model ini mendukung panjang konteks (*context length*) hingga 128K token.

Sebagai mahasiswa ilmu komputer, Anda harus memahami bahwa inovasi utama DeepSeek-V2 terletak pada dua arsitektur revolusionernya:
1. **Multi-head Latent Attention (MLA):** Menjamin inferensi yang efisien dengan cara mengompresi memori singgahan *Key-Value* (KV *cache*) secara signifikan ke dalam sebuah vektor laten (*latent vector*).
2. **DeepSeekMoE:** Memungkinkan pelatihan model yang sangat tangguh dengan biaya ekonomis melalui komputasi yang renggang (*sparse computation*).

Dibandingkan dengan pendahulunya (DeepSeek 67B), DeepSeek-V2 mencapai kinerja yang jauh lebih kuat, namun pada saat yang sama mampu **menghemat 42,5% biaya pelatihan**, **mengurangi KV cache hingga 93,3%**, dan **meningkatkan hasil maksimal (*maximum generation throughput*) hingga 5,76 kali lipat**. Model ini diprapelatihan pada korpus bersumber jamak berkualitas tinggi yang terdiri dari 8,1 Triliun (8.1T) token. Selanjutnya, model ini menjalani Pelatihan Halus Terselia (*Supervised Fine-Tuning* / SFT) dan Pembelajaran Penguatan (*Reinforcement Learning* / RL) untuk membuka seluruh potensinya. 

Hasil evaluasi menunjukkan bahwa, meskipun hanya menggunakan 21B parameter aktif, DeepSeek-V2 dan versi obrolannya (*chat versions*) tetap mencapai kinerja papan atas di antara model-model sumber terbuka (*open-source*).

---

### Analisis Mendalam Gambar 1: Kinerja, Biaya Pelatihan, dan Efisiensi Inferensi

Sebelum kita masuk ke teori, mari kita bedah **Gambar 1** yang memberikan ringkasan visual dari pencapaian arsitektur ini.

*   **Gambar 1 (a) - Grafik Titik (Scatter Plot): Akurasi MMLU vs Parameter Aktif**
    *   **Sumbu X (Horizontal):** Merepresentasikan jumlah Parameter Aktif dalam skala Miliar (*Billions*), berkisar dari 0 hingga 100+ Miliar.
    *   **Sumbu Y (Vertikal):** Merepresentasikan metrik performa (*Performance*) pada tolok ukur MMLU (dalam skala 55 hingga 80).
    *   **Titik Data:** Grafik ini memplot berbagai model bahasa populer. Model dari keluarga LLaMA 1, 2, dan 3, keluarga Mixtral, Qwen, dan Command R disimbolkan dengan lingkaran berwarna-warni.
    *   **Bintang Merah:** Bintang merah di kiri atas melambangkan **DeepSeek-V2**. 
    *   **Kesimpulan Akademis:** Anda dapat melihat dengan sangat jelas bahwa Bintang Merah berada pada posisi paling ekstrem di sudut kiri atas. Ini berarti DeepSeek-V2 mencapai tingkat akurasi (kecerdasan) yang setara atau melebihi model dengan 70B+ parameter aktif (seperti LLaMA 3 70B), namun DeepSeek-V2 melakukannya hanya dengan menggunakan sebagian kecil parameter aktif (~21B). Ini adalah definisi matematis dari efisiensi yang optimal.

*   **Gambar 1 (b) - Diagram Batang (Bar Charts): Perbandingan dengan DeepSeek 67B**
    *   **Diagram Pertama (Biaya Pelatihan):** Membandingkan biaya pelatihan dalam satuan *K GPU Hours / T Tokens*. Batang DeepSeek 67B jauh lebih panjang. Batang DeepSeek-V2 lebih pendek, mengindikasikan **penghematan biaya pelatihan sebesar 42,5%**.
    *   **Diagram Kedua (KV Cache untuk Pembuatan Teks):** Sumbu X merepresentasikan memori dalam KB/Token. Batang DeepSeek 67B hampir menyentuh 400 KB/Token. DeepSeek-V2 memangkasnya secara drastis, mengindikasikan **pengurangan KV cache sebesar 93,3%**.
    *   **Diagram Ketiga (Hasil Pembuatan Teks Maksimal):** Sumbu X merepresentasikan *Tokens/Sec* (Token per detik). Batang DeepSeek-V2 jauh melampaui DeepSeek 67B, mengindikasikan **peningkatan *throughput* hingga 576%**.
    *   **Kesimpulan Akademis:** Arsitektur MLA dan MoE secara sinergis menyelesaikan dilema klasik di *Deep Learning*: bagaimana meningkatkan kecepatan inferensi dan menurunkan memori operasional tanpa mengorbankan metrik kecerdasan.

---

## 1. Pendahuluan

Dalam beberapa tahun terakhir, Model Bahasa Besar (LLMs) telah mengalami perkembangan pesat, menawarkan sekilas pandang menuju fajar Kecerdasan Buatan Umum (*Artificial General Intelligence* / AGI). Secara umum, kecerdasan sebuah LLM cenderung meningkat seiring dengan bertambahnya jumlah parameter, yang memungkinkannya untuk menunjukkan kemampuan *emergent* (kemampuan baru yang muncul secara tiba-tiba) di berbagai tugas. 

Namun, peningkatan ini datang dengan harga yang sangat mahal: kebutuhan sumber daya komputasi yang jauh lebih besar untuk pelatihan, dan potensi penurunan kelancaran/kecepatan inferensi (*inference throughput*). Kendala-kendala ini menghadirkan tantangan signifikan yang menghambat adopsi dan pemanfaatan LLM secara luas. Untuk mengatasi masalah ini, diperkenalkan **DeepSeek-V2**, sebuah model bahasa MoE sumber terbuka yang tangguh, dikarakterisasikan oleh pelatihan yang ekonomis dan inferensi yang efisien melalui arsitektur Transformer yang inovatif. 

Kita mengoptimalkan modul atensi (*attention modules*) dan Jaringan Maju-Umpan (*Feed-Forward Networks* / FFNs) dalam kerangka kerja Transformer dengan dua usulan utama:
1.  **Multi-head Latent Attention (MLA):** Dalam konteks mekanisme atensi, KV *cache* dari *Multi-Head Attention* (MHA) standar menghadirkan hambatan (*bottleneck*) yang parah terhadap efisiensi inferensi LLM. Berbagai pendekatan seperti *Grouped-Query Attention* (GQA) dan *Multi-Query Attention* (MQA) sering kali mengorbankan performa demi mengurangi KV cache. MLA hadir dengan kompresi gabungan *key-value* berperingkat rendah (*low-rank joint compression*). Secara empiris, MLA mencapai performa yang superior dibandingkan MHA, sementara secara signifikan mengurangi KV cache selama inferensi.
2.  **DeepSeekMoE:** Untuk FFN, diadopsi arsitektur DeepSeekMoE yang menggunakan segmentasi pakar berbutir halus (*fine-grained expert segmentation*) dan isolasi pakar bersama (*shared expert isolation*) untuk potensi spesialisasi pakar yang lebih tinggi. Karena kita menggunakan paralelisme pakar (*expert parallelism*) selama pelatihan, mekanisme tambahan juga dirancang untuk mengontrol biaya komunikasi (*communication overheads*) dan memastikan keseimbangan beban (*load balance*).

Model ini diprapelatihan dengan 8,1T token, di mana korpus ini memiliki porsi data berbahasa Mandarin yang lebih luas dan berkualitas tinggi. Selanjutnya, dikumpulkan 1,5 Juta sesi percakapan untuk Pelatihan Halus Terselia (SFT). Terakhir, model memanfaatkan *Group Relative Policy Optimization* (GRPO) untuk menyelaraskan model dengan preferensi manusia (Tahap *Reinforcement Learning* / RL).

---

### Analisis Mendalam Gambar 2: Arsitektur DeepSeek-V2

Mari kita telaah arsitektur internal model melalui **Gambar 2**.

*   **Deskripsi Visual:** Gambar ini membagi blok Transformer menjadi dua inovasi utama: **DeepSeekMoE** (menggantikan FFN konvensional) di bagian atas, dan **Multi-Head Latent Attention (MLA)** di bagian bawah.
*   **Bagian Atas (DeepSeekMoE):**
    *   Terdapat lapisan *Input Hidden* $\mathbf{u}_t$ yang masuk ke sebuah *Router* (pengarah lalu lintas token).
    *   Terdapat kotak hijau berlabel *Shared Expert* (Pakar Bersama). Kotak ini selalu dilalui oleh token.
    *   Terdapat kotak biru berlabel *Routed Expert* (Pakar yang Diarahkan). *Router* akan memilih sebagian kecil pakar (misal Pakar 1, 2, 3... $N_r$) menggunakan fungsi Top-$K_r$.
    *   **Kesimpulan Logis:** Ini mengilustrasikan bagaimana sebagian besar komputasi dinonaktifkan (karena *sparse*), menghemat memori, sementara *Shared Expert* memastikan pengetahuan umum tidak terlupakan.
*   **Bagian Bawah (MLA):**
    *   Lapisan *Input Hidden* $\mathbf{h}_t$ dimasukkan ke dua jalur utama. 
    *   Satu jalur memproses *Queries* (Kueri) menghasilkan vektor laten $\mathbf{c}_t^Q$, yang kemudian digabungkan dengan komponen *Rotary Position Embedding* (RoPE).
    *   Jalur lainnya memproses kompresi *Key-Value* laten menghasilkan $\mathbf{c}_t^{KV}$. Perhatikan blok berlabel **"Cached During Inference"**. Sangat penting untuk dicatat bahwa alih-alih meng-cache matriks KV yang besar, sistem hanya meng-cache vektor laten kompresi dan komponen RoPE-nya.
    *   Vektor kompresi kemudian dibongkar (dekompresi) melalui matriks *concatenate* untuk membentuk kueri, kunci (*keys*), dan nilai (*values*) reguler tepat sebelum fungsi Atensi.
    *   **Kesimpulan Logis:** Skema ini menjelaskan secara arsitektural mengapa ukuran KV cache DeepSeek-V2 menyusut hingga lebih dari 90%.

---

## 2. Arsitektur

Secara keseluruhan, DeepSeek-V2 tetap berada dalam kerangka arsitektur Transformer, di mana setiap blok Transformer terdiri dari modul atensi dan *Feed-Forward Network* (FFN). Namun, untuk kedua modul tersebut, kita merancang dan menggunakan arsitektur inovatif: **MLA** dan **DeepSeekMoE**. Detail lainnya seperti normalisasi lapisan (*layer normalization*) dan fungsi aktivasi mengikuti pengaturan DeepSeek 67B.

### 2.1 Multi-Head Latent Attention: Meningkatkan Efisiensi Inferensi

Model Transformer konvensional menggunakan *Multi-Head Attention* (MHA). Namun, saat pembuatan teks (*generation*), memori *Key-Value* (KV) cache yang sangat besar menjadi "leher botol" (*bottleneck*) yang membatasi efisiensi inferensi (membatasi panjang konteks dan *batch size*). Solusi alternatif seperti MQA dan GQA mengurangi KV cache namun mengorbankan performa komputasional. MLA hadir untuk memecahkan dilema ini.

#### 2.1.1 Pendahuluan: Multi-Head Attention (MHA) Standar

Untuk memahami inovasi MLA, kita harus membedah matematika dari MHA standar. 

Misalkan $d$ adalah dimensi penyematan (*embedding dimension*), $n_h$ adalah jumlah kepala atensi (*number of attention heads*), $d_h$ adalah dimensi per kepala, dan $\mathbf{h}_t \in \mathbb{R}^d$ adalah input atensi dari token ke-$t$ pada sebuah lapisan atensi. MHA standar pertama-tama memproduksi kueri ($\mathbf{q}_t$), kunci ($\mathbf{k}_t$), dan nilai ($\mathbf{v}_t$) melalui tiga matriks proyeksi $W^Q, W^K, W^V \in \mathbb{R}^{d_hn_h \times d}$, berturut-turut:

$$ \mathbf{q}_t = W^Q\mathbf{h}_t, \tag{1} $$
$$ \mathbf{k}_t = W^K\mathbf{h}_t, \tag{2} $$
$$ \mathbf{v}_t = W^V\mathbf{h}_t, \tag{3} $$

**Penjelasan Variabel Formula (1), (2), dan (3):**
*   $\mathbf{h}_t$: Representasi matematis (vektor) dari kata/token ke-$t$ saat masuk ke lapisan atensi.
*   $W^Q, W^K, W^V$: Matriks bobot (*weight matrices*) yang dipelajari selama pelatihan. Tugasnya adalah mengubah (memproyeksikan) vektor asli $\mathbf{h}_t$ menjadi representasi Kueri (untuk mencari), Kunci (untuk dicocokkan), dan Nilai (kandungan informasi aktual).
*   $\mathbf{q}_t, \mathbf{k}_t, \mathbf{v}_t$: Vektor hasil proyeksi. Dimensinya diperluas menjadi $\mathbb{R}^{d_hn_h}$ (total ukuran seluruh kepala atensi digabung).

Selanjutnya, $\mathbf{q}_t, \mathbf{k}_t, \mathbf{v}_t$ akan "diiris" (*sliced*) menjadi $n_h$ kepala untuk komputasi atensi multi-kepala:

$$ [\mathbf{q}_{t,1}; \mathbf{q}_{t,2}; ...; \mathbf{q}_{t,n_h}] = \mathbf{q}_t, \tag{4} $$
$$ [\mathbf{k}_{t,1}; \mathbf{k}_{t,2}; ...; \mathbf{k}_{t,n_h}] = \mathbf{k}_t, \tag{5} $$
$$ [\mathbf{v}_{t,1}; \mathbf{v}_{t,2}; ...; \mathbf{v}_{t,n_h}] = \mathbf{v}_t, \tag{6} $$

**Penjelasan Variabel Formula (4), (5), dan (6):**
*   $n_h$: Jumlah kepala atensi. Membagi vektor besar menjadi bagian-bagian kecil memungkinkan model memproses berbagai aspek teks (misal: satu kepala mengurus tata bahasa, kepala lain mengurus subjek kalimat) secara bersamaan dan independen. Notasi $[ ...; ... ]$ berarti konkatenasi (penggabungan matriks).

Keluaran dari setiap kepala atensi dihitung sebagai berikut:

$$ \mathbf{o}_{t,i} = \sum_{j=1}^{t} \text{Softmax}_j \left( \frac{\mathbf{q}_{t,i}^T \mathbf{k}_{j,i}}{\sqrt{d_h}} \right) \mathbf{v}_{j,i}, \tag{7} $$

**Penjelasan Variabel Formula (7):**
*   $\mathbf{q}_{t,i}^T \mathbf{k}_{j,i}$: Operasi *dot product* (perkalian titik) antara Kueri dari token saat ini ($t$) dan Kunci dari token masa lalu ($j$). Ini mengukur *seberapa relevan* token ke-$j$ terhadap token ke-$t$.
*   $\sqrt{d_h}$: Faktor penskalaan (*scaling factor*) untuk menstabilkan gradien saat pelatihan, mencegah nilai dari perkalian titik menjadi terlalu besar.
*   $\text{Softmax}_j$: Fungsi yang mengubah skor relevansi menjadi distribusi probabilitas (semua bobot atensi berjumlah 1).
*   $\mathbf{v}_{j,i}$: Vektor nilai dari token masa lalu. Hasil akhirnya $\mathbf{o}_{t,i}$ adalah rata-rata tertimbang dari semua nilai masa lalu.

Terakhir, semua kepala digabungkan dan diproyeksikan ke luar:

$$ \mathbf{u}_t = W^O [\mathbf{o}_{t,1}; \mathbf{o}_{t,2}; ...; \mathbf{o}_{t,n_h}], \tag{8} $$

**Penjelasan Variabel Formula (8):**
*   $W^O \in \mathbb{R}^{d \times d_hn_h}$: Matriks proyeksi keluaran yang mengembalikan dimensi ke ukuran aslinya ($d$).
*   $\mathbf{u}_t$: Vektor keluaran akhir dari modul atensi untuk token ke-$t$.

**Analisis Masalah MHA:** Selama fase inferensi (pembuatan teks), semua matriks Kunci ($\mathbf{k}$) dan Nilai ($\mathbf{v}$) dari token sebelumnya *harus disimpan* di memori GPU agar tidak dihitung ulang setiap kali token baru dibuat. Ini yang disebut **KV Cache**. Dalam MHA standar, kita harus menyimpan $2n_hd_hl$ elemen per token (dengan $l$ adalah jumlah lapisan). Ini adalah *bottleneck* raksasa yang memakan kapasitas memori.

#### 2.1.2 Kompresi Gabungan Key-Value Berperingkat Rendah (*Low-Rank Key-Value Joint Compression*)

Inti dari arsitektur MLA adalah kompresi berperingkat rendah (*low-rank joint compression*) untuk kunci dan nilai demi mereduksi KV cache. Secara matematis direpresentasikan:

$$ \mathbf{c}_t^{KV} = W^{DKV}\mathbf{h}_t, \tag{9} $$
$$ \mathbf{k}_t^C = W^{UK}\mathbf{c}_t^{KV}, \tag{10} $$
$$ \mathbf{v}_t^C = W^{UV}\mathbf{c}_t^{KV}, \tag{11} $$

**Penjelasan Variabel Formula (9), (10), dan (11):**
*   $\mathbf{c}_t^{KV} \in \mathbb{R}^{d_c}$: Ini adalah "Vektor Laten Terkompresi" untuk kunci dan nilai. Model tidak langsung membuat kunci dan nilai yang besar, melainkan membuat satu vektor "padat" berukuran kecil $d_c$.
*   $d_c$: Dimensi kompresi KV, di mana nilainya direkayasa agar jauh lebih kecil daripada dimensi asli ($d_c \ll d_hn_h$).
*   $W^{DKV} \in \mathbb{R}^{d_c \times d}$: Matriks proyeksi-turun (*down-projection*), berfungsi menyusutkan data.
*   $W^{UK}, W^{UV} \in \mathbb{R}^{d_hn_h \times d_c}$: Matriks proyeksi-naik (*up-projection*), berfungsi mekar/mendekompresi vektor laten kembali menjadi dimensi kunci ($\mathbf{k}_t^C$) dan nilai ($\mathbf{v}_t^C$).

**Kecemerlangan Akademis MLA:** Selama inferensi, MLA **HANYA** perlu menyimpan vektor laten $\mathbf{c}_t^{KV}$ ke dalam *cache*. Ini berarti KV cache hanya memakan $d_c \times l$ elemen per token. Lebih dari itu, karena sifat aljabar linear, matriks proyeksi naik ($W^{UK}$ dan $W^{UV}$) dapat **diserap** (*absorbed*) ke dalam proyeksi Kueri ($W^Q$) dan proyeksi Keluaran ($W^O$) selama inferensi. Ini berarti kita *bahkan tidak perlu mendekompresi Kunci dan Nilai sama sekali* saat komputasi berlangsung! Inilah inovasi murni yang melahirkan efisiensi.

Untuk mengurangi memori aktivasi saat masa pelatihan (*training*), hal yang sama juga diterapkan pada Kueri (*Queries*):

$$ \mathbf{c}_t^Q = W^{DQ}\mathbf{h}_t, \tag{12} $$
$$ \mathbf{q}_t^C = W^{UQ}\mathbf{c}_t^Q, \tag{13} $$

**Penjelasan Variabel Formula (12) dan (13):**
*   $\mathbf{c}_t^Q \in \mathbb{R}^{d_c'}$: Vektor laten terkompresi khusus untuk Kueri.
*   $d_c'$: Dimensi kompresi Kueri ($d_c' \ll d_hn_h$).
*   $W^{DQ}, W^{UQ}$: Matriks proyeksi-turun dan proyeksi-naik untuk Kueri.

#### 2.1.3 Decoupled Rotary Position Embedding (RoPE)

Seperti pada DeepSeek 67B, kita harus memasukkan informasi "posisi urutan kata" menggunakan *Rotary Position Embedding* (RoPE). Namun, RoPE bermusuhan dengan metode kompresi peringkat rendah kita. Mengapa? Karena RoPE bergantung pada posisi absolut token. Jika kita menerapkan RoPE pada kunci dekompresi $\mathbf{k}_t^C$, operasi matriks tersebut menjadi sensitif terhadap posisi (tidak komutatif), sehingga matriks $W^{UK}$ tidak bisa lagi diserap ke dalam $W^Q$. Jika itu terjadi, efisiensi yang dibangun di bagian 2.1.2 akan hancur lebur.

Solusinya: **Memisahkan (*Decoupling*) RoPE**. 
Alih-alih mengompresi komponen posisi, disediakan jalur multi-kepala khusus untuk kueri $\mathbf{q}_{t,i}^R$ dan satu jalur kunci bersama $\mathbf{k}_t^R$ yang bertugas *hanya* membawa informasi RoPE:

$$ [\mathbf{q}_{t,1}^R; \mathbf{q}_{t,2}^R; ...; \mathbf{q}_{t,n_h}^R] = \mathbf{q}_t^R = \text{RoPE}(W^{QR}\mathbf{c}_t^Q), \tag{14} $$
$$ \mathbf{k}_t^R = \text{RoPE}(W^{KR}\mathbf{h}_t), \tag{15} $$

**Penjelasan Variabel Formula (14) dan (15):**
*   $W^{QR}, W^{KR}$: Matriks pembuat vektor kueri dan kunci independen khusus untuk memuat informasi posisional.
*   $\text{RoPE}(\cdot)$: Fungsi matematika yang merotasi vektor di ruang berdimensi untuk mengodekan urutan posisi token.
*   $\mathbf{q}_{t,i}^R \in \mathbb{R}^{d_h^R}, \mathbf{k}_t^R \in \mathbb{R}^{d_h^R}$: Kueri dan kunci terpisah, di mana $d_h^R$ adalah ukuran dimensi per kepala khusus untuk RoPE.

Setelah itu, kueri kompresi dan kueri RoPE digabungkan (*concatenated*), begitu juga dengan kuncinya:

$$ \mathbf{q}_{t,i} = [\mathbf{q}_{t,i}^C; \mathbf{q}_{t,i}^R], \tag{16} $$
$$ \mathbf{k}_{t,i} = [\mathbf{k}_{t,i}^C; \mathbf{k}_t^R], \tag{17} $$

Atensi kemudian dihitung normal:

$$ \mathbf{o}_{t,i} = \sum_{j=1}^{t} \text{Softmax}_j \left( \frac{\mathbf{q}_{t,i}^T \mathbf{k}_{j,i}}{\sqrt{d_h + d_h^R}} \right) \mathbf{v}_{j,i}^C, \tag{18} $$
$$ \mathbf{u}_t = W^O [\mathbf{o}_{t,1}; \mathbf{o}_{t,2}; ...; \mathbf{o}_{t,n_h}], \tag{19} $$

**Penjelasan Variabel Formula (18) dan (19):**
*   Semua variabel bekerja persis seperti MHA standar, hanya saja perhitungannya membelah antara fitur konten ($\mathbf{q}^C, \mathbf{k}^C$) dan fitur posisi ($\mathbf{q}^R, \mathbf{k}^R$). 
*   Selama inferensi, kunci yang membawa info RoPE ($\mathbf{k}_t^R$) HARUS disimpan di *cache*. Jadi, total KV cache per token untuk DeepSeek-V2 adalah: $(d_c + d_h^R)l$ elemen.

#### 2.1.4 Perbandingan Key-Value Cache

Mari kita bandingkan jumlah memori KV *cache* yang dikonsumsi per token oleh berbagai arsitektur atensi melalui **Tabel 1**.

**Tabel 1: Perbandingan KV cache per Token di antara berbagai mekanisme atensi.**

| Mekanisme Atensi | KV Cache per Token (# Elemen) | Kemampuan |
| :--- | :---: | :--- |
| Multi-Head Attention (MHA) | $2n_hd_hl$ | Kuat |
| Grouped-Query Attention (GQA) | $2n_gd_hl$ | Sedang |
| Multi-Query Attention (MQA) | $2d_hl$ | Lemah |
| **MLA (Ours)** | $(d_c + d_h^R)l \approx \frac{9}{2}d_hl$ | **Lebih Kuat** |

**Penjelasan Akademis:**
*   $n_h$: Jumlah kepala atensi. $d_h$: Dimensi per kepala. $l$: Jumlah lapisan. $n_g$: Jumlah grup di GQA. $d_c$: Dimensi kompresi MLA. $d_h^R$: Dimensi RoPE MLA.
*   **Analisis:** Pada DeepSeek-V2, $d_c$ diatur menjadi $4d_h$, dan $d_h^R$ diatur menjadi $\frac{d_h}{2}$. Jika dimasukkan ke rumus MLA, total elemen *cache*-nya hanyalah sekitar $\frac{9}{2}d_hl$. Nilai ini ekuivalen dengan menggunakan arsitektur GQA tetapi hanya dengan 2,25 grup (sangat-sangat kecil memori yang dikonsumsi). Namun secara ajaib, kapasitas metrik penalaran (kecerdasan) MLA terbukti *lebih kuat* dibandingkan MHA konvensional.

---

### 2.2 DeepSeekMoE: Melatih Model Tangguh dengan Biaya Ekonomis

#### 2.2.1 Arsitektur Dasar

Untuk jaringan *Feed-Forward* (FFN), model menggunakan arsitektur **DeepSeekMoE**. Terdapat dua ide kunci:
1. Menyegmentasi pakar (*experts*) ke dalam butiran yang sangat halus (*finer granularity*) demi tingkat spesialisasi tinggi.
2. Mengisolasi sebagian pakar menjadi pakar umum (*shared experts*) untuk menampung pengetahuan redundan, sehingga pakar lain bisa lebih spesifik.

Misalkan $\mathbf{u}_t$ adalah masukan FFN untuk token ke-$t$. Keluaran FFN, $\mathbf{h}_t'$, dihitung sebagai:

$$ \mathbf{h}_t' = \mathbf{u}_t + \sum_{i=1}^{N_s} \text{FFN}_i^{(s)}(\mathbf{u}_t) + \sum_{i=1}^{N_r} g_{i,t} \text{FFN}_i^{(r)}(\mathbf{u}_t), \tag{20} $$

$$ g_{i,t} = \begin{cases} s_{i,t}, & \text{jika } s_{i,t} \in \text{Topk}(\{s_{j,t} | 1 \leqslant j \leqslant N_r\}, K_r), \\ 0, & \text{sebaliknya}, \end{cases} \tag{21} $$

$$ s_{i,t} = \text{Softmax}_i (\mathbf{u}_t^T \mathbf{e}_i), \tag{22} $$

**Penjelasan Variabel Formula (20), (21), dan (22):**
*   $N_s$ dan $N_r$: Jumlah *Shared Experts* (Pakar Bersama) dan *Routed Experts* (Pakar Terarah).
*   $\text{FFN}_i^{(s)}(\cdot)$ dan $\text{FFN}_i^{(r)}(\cdot)$: Jaringan saraf untuk pakar bersama ke-$i$ dan pakar terarah ke-$i$.
*   $K_r$: Jumlah maksimum pakar terarah yang boleh diaktifkan (dari total $N_r$).
*   $g_{i,t}$: Nilai bobot gerbang (*gate value*) untuk pakar ke-$i$. Jika pakar tidak terpilih oleh fungsi Top-$K_r$, nilainya 0 (tidak memakan komputasi).
*   $s_{i,t}$: Probabilitas afinitas (*affinity*) yang dihitung menggunakan fungsi *Softmax* antara input token $\mathbf{u}_t^T$ dan vektor sentroid pakar $\mathbf{e}_i$. Ini mengukur seberapa cocok token tersebut diproses oleh pakar tersebut.

#### 2.2.2 Device-Limited Routing (Perutean Terbatas Perangkat)

Untuk melatih model sebesar ini, kita harus menyebar pakar-pakar tersebut ke berbagai unit GPU (ini disebut Paralelisme Pakar / *Expert Parallelism*). Namun, jika sebuah token memerlukan 6 pakar yang berada di 6 GPU yang berbeda, akan terjadi macet komunikasi jaringan antar-GPU (*communication overhead*).

Solusinya: Mekanisme **Device-Limited Routing**. Saat mencari pakar menggunakan Top-$K_r$, token diwajibkan untuk hanya boleh didistribusikan pada **maksimal $M$ perangkat GPU saja**. Dalam praktiknya, jika $M \geqslant 3$, performa perutean model hampir tidak berbeda dengan model yang tanpa batas rute, namun beban transfer data antar-GPU berkurang drastis.

#### 2.2.3 Auxiliary Loss for Load Balance (Fungsi Kerugian Tambahan untuk Keseimbangan Beban)

Karena *router* melatih dirinya sendiri menggunakan *Softmax*, AI sering kali menjadi "malas" dan mengalami **Routing Collapse** (Runtuhnya Perutean). Ia hanya akan terus-menerus mengirim token ke 2 atau 3 pakar yang dianggapnya "pintar", membiarkan 150 pakar lainnya terbengkalai. Ini membuat GPU yang menampung pakar pintar menjadi *overload* (kelebihan beban), sementara GPU lain menganggur.

Untuk memaksa model agar mendistribusikan beban secara adil, ditambahkan penalti/hukuman ke dalam proses pelatihan (Fungsi Kerugian / *Loss Function*). Ada tiga jenis fungsi kerugian tambahan:

**1. Expert-Level Balance Loss (Keseimbangan Tingkat Pakar):**

$$ \mathcal{L}_{\text{ExpBal}} = \alpha_1 \sum_{i=1}^{N_r} f_i P_i, \tag{23} $$
$$ f_i = \frac{N_r}{K_r T} \sum_{t=1}^{T} \mathbb{1}(\text{Token } t \text{ selects Expert } i), \tag{24} $$
$$ P_i = \frac{1}{T} \sum_{t=1}^{T} s_{i,t}, \tag{25} $$

**Penjelasan Variabel Formula (23), (24), dan (25):**
*   $T$: Jumlah total token dalam satu urutan (sekuens).
*   $f_i$: Frekuensi aktual (seberapa sering pakar $i$ benar-benar dipilih oleh token). $\mathbb{1}(\cdot)$ adalah fungsi yang bernilai 1 jika benar, 0 jika salah.
*   $P_i$: Probabilitas teoretis (rata-rata skor afinitas pakar $i$ untuk seluruh token).
*   $\alpha_1$: Faktor konstanta pengali (seberapa keras penalti ini diberlakukan). Jika satu pakar memiliki $f_i$ yang terlalu besar (sering dipakai), nilai $\mathcal{L}_{\text{ExpBal}}$ akan naik tajam, menghukum model selama proses propagasi balik (*backpropagation*).

**2. Device-Level Balance Loss (Keseimbangan Tingkat Perangkat GPU):**

Jika pakar dibagi menjadi $D$ kelompok perangkat, $\mathcal{E}_1, \mathcal{E}_2, ..., \mathcal{E}_D$, kita wajib memastikan GPU bekerja merata:

$$ \mathcal{L}_{\text{DevBal}} = \alpha_2 \sum_{i=1}^{D} f_i' P_i', \tag{26} $$
$$ f_i' = \frac{1}{|\mathcal{E}_i|} \sum_{j \in \mathcal{E}_i} f_j, \tag{27} $$
$$ P_i' = \sum_{j \in \mathcal{E}_i} P_j, \tag{28} $$

**Penjelasan Variabel Formula (26), (27), dan (28):**
*   $\mathcal{E}_i$: Kelompok pakar yang berada di perangkat GPU ke-$i$.
*   $f_i'$ dan $P_i'$: Rata-rata frekuensi dan akumulasi probabilitas dari seluruh pakar yang ada di dalam GPU ke-$i$.
*   $\alpha_2$: Faktor hiperparameter untuk penalti ketidakseimbangan GPU.

**3. Communication Balance Loss (Keseimbangan Komunikasi):**

Mengontrol agar jumlah data (token) yang dikirim dan diterima oleh setiap perangkat seimbang:

$$ \mathcal{L}_{\text{CommBal}} = \alpha_3 \sum_{i=1}^{D} f_i'' P_i'', \tag{29} $$
$$ f_i'' = \frac{D}{M T} \sum_{t=1}^{T} \mathbb{1}(\text{Token } t \text{ is sent to Device } i), \tag{30} $$
$$ P_i'' = \sum_{j \in \mathcal{E}_i} P_j, \tag{31} $$

**Penjelasan Variabel Formula (29), (30), dan (31):**
*   $f_i''$: Frekuensi pengiriman token ke perangkat $i$.
*   $\alpha_3$: Faktor penalti keseimbangan transfer komunikasi.

#### 2.2.4 Token-Dropping Strategy (Strategi Pembuangan Token)

Bahkan dengan fungsi penalti, ketidakseimbangan tetap terjadi. Jika satu GPU mendapat 2000 token sementara GPU lain mendapat 500 token, komputasi global tertahan menunggu GPU pertama selesai. 
Solusinya ekstrem namun brilian: **Buang tokennya**. GPU hanya diizinkan memproses sesuai batas kuotanya (Kapasitas Faktor = 1.0). Jika melebihi batas, token dengan skor afinitas *terendah* pada perangkat tersebut akan di-*drop* (dibuang dan tidak diproses). Namun, demi keamanan, sistem menjamin bahwa 10% token paling kritis dari setiap urutan tidak akan pernah dibuang.

---

## 3. Prapelatihan (*Pre-Training*)

### 3.1 Pengaturan Eksperimental

#### 3.1.1 Konstruksi Data
Model dilatih menggunakan **8,1 Triliun (8.1T) Token**. Peneliti memperbanyak sumber data internet, memaksimalkan teknik pembersihan data untuk memulihkan data yang secara keliru terhapus di masa lalu, dan secara dramatis meningkatkan porsi data internet berbahasa Mandarin. Kebijakan penyaringan *bias* yang ketat juga diterapkan untuk menghapus "konten kontroversial/budaya spesifik" agar AI tetap objektif.

#### 3.1.2 Parameter Hiper (*Hyper-Parameters*)
Sebagai acuan rekonstruksi rekayasa (*engineering*):
*   **Lapisan Transformer ($l$):** 60 layer.
*   **Dimensi Tersembunyi ($d$):** 5120.
*   **MLA:** Kepala atensi ($n_h$) = 128; Dimensi per kepala ($d_h$) = 128. Dimensi kompresi KV ($d_c$) = 512. Dimensi kompresi kueri ($d_c'$) = 1536. Dimensi per kepala RoPE ($d_h^R$) = 64.
*   **DeepSeekMoE:** Setiap layer FFN memiliki 160 Pakar Terarah (*Routed*) dan 2 Pakar Bersama (*Shared*). Dari 160, **hanya 6 pakar yang diaktifkan** untuk setiap token. 
*   **Total Parameter:** 236 Miliar (236B).
*   **Parameter Aktif per Token:** 21 Miliar (21B).
*   **Pengoptimal:** AdamW ($\beta_1 = 0.9, \beta_2 = 0.95$). Laju Pembelajaran (*Learning Rate*) mencapai maksimum $2.4 \times 10^{-4}$ dengan metode pemanasan (*warmup*) dan penurunan (*step-decay*).

#### 3.1.3 Infrastruktur
Dilatih menggunakan kerangka kerja internal bernama **HAI-LLM** yang mengandalkan 16-arah paralelisme alur-kerja (*pipeline parallelism*), 8-arah paralelisme pakar (*expert parallelism*), dan ZeRO-1 paralelisme data. Dilatih pada klaster superkomputer NVIDIA H800 dengan protokol *InfiniBand*.

#### 3.1.4 Ekstensi Konteks Panjang (*Long Context Extension*)
Pada awalnya model hanya melihat konteks 4K. Sistem kemudian menggunakan metodologi **YaRN** (Yet another RoPE extensioN) untuk meregangkan konteks model secara matematis dari 4K hingga **128K Token**. Faktor skala YaRN ($s$) diatur ke 40. Setelah manipulasi ruang laten ini selesai, model dilatih selama 1000 *steps* tambahan pada sekuens 32K token.

---

### Analisis Mendalam Gambar 4: Pengujian "Jarum di Tumpukan Jerami"

Untuk membuktikan bahwa konteks 128K tersebut benar-benar berfungsi (dan model tidak amnesia terhadap paragraf awal), dilakukan pengujian *Needle In A Haystack* (NIAH).

*   **Gambar 4 (Heatmap): Pengujian "Needle In A Haystack"**
    *   **Sumbu X (Horizontal):** Panjang konteks teks, bertambah dari 1K (seribu kata) di ujung kiri, hingga 128K (seratus dua puluh delapan ribu kata) di ujung kanan.
    *   **Sumbu Y (Vertikal):** *Document Depth Percent (%)* - Posisi kedalaman "Jarum" (sebuah kalimat fakta acak yang diselipkan diam-diam) di dalam "Jerami" (dokumen teks raksasa). 0% berarti kalimat diselipkan di bagian sangat atas teks, 100% berarti diselipkan di akhir teks.
    *   **Skala Warna (Z-axis):** Merepresentasikan Skor Akurasi. Warna Hijau Penuh bernilai 10. Warna Merah bernilai 1.
    *   **Kesimpulan Akademis:** Hampir 98% luas kotak heatmap ini dipenuhi oleh warna Hijau pekat. Hanya ada dua kotak kecil berwarna kekuningan/oranye (skor 6-7). Ini membuktikan pencapaian engineering yang luar biasa: DeepSeek-V2 memiliki kemampuan pemanggilan kembali (*recall*) memori nyaris absolut 100% dari setiap baris teks, bahkan ketika fakta yang dicari tersembunyi di kedalaman 100 ribu kata.

---

### 3.2 Evaluasi

#### 3.2.1 Tolok Ukur Evaluasi (*Evaluation Benchmarks*)
Model diuji di berbagai panggung persaingan ujian untuk menilai kecerdasan MMLU, HellaSwag, PIQA, ARC, Pemahaman Membaca (RACE, DROP), Matematika (GSM8K, MATH), Pemrograman/Kode (HumanEval, MBPP), hingga ujian terstandarisasi berbahasa Mandarin (C-Eval, CMMLU, AGIEval).

#### 3.2.2 Hasil Evaluasi

Sebagai pakar, mari kita analisis **Tabel 2**, yang merupakan "Papan Peringkat" definitif untuk model-model sumber terbuka raksasa di pasar AI saat ini.

**Tabel 2: Perbandingan antara DeepSeek-V2 dan model sumber terbuka representatif lainnya.** 
*(Skor dalam persentase, angka yang dicetak tebal adalah nilai tertinggi di antara semuanya).*

| Benchmark (Metric) | # Shots | DeepSeek 67B | Qwen1.5 72B | Mixtral 8x22B | LLaMA 3 70B | **DeepSeek-V2** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Arsitektur | - | Dense | Dense | MoE | Dense | **MoE** |
| # Activated Params | - | 67B | 72B | 39B | 70B | **21B** |
| # Total Params | - | 67B | 72B | 141B| 70B | **236B** |
| **English** |
| MMLU (Acc.) | 5-shot | 71.3 | 77.2 | 77.6 | **78.9** | *78.5* |
| ARC-Challenge (Acc.) | 25-shot | 86.4 | *92.8* | 91.2 | **93.3** | 92.4 |
| PIQA (Acc.) | 0-shot | *83.6* | 83.3 | *83.6* | **85.0** | 83.7 |
| WinoGrande (Acc.) | 5-shot | 84.9 | 82.4 | 83.7 | **85.7** | 84.9 |
| **Code (Pemrograman)** |
| HumanEval (Pass@1) | 0-shot | 45.1 | 43.9 | **53.1** | 48.2 | *48.8* |
| MBPP (Pass@1) | 3-shot | 57.4 | 53.6 | 64.2 | **68.6** | *66.6* |
| CRUXEval-I (Acc.) | 2-shot | 42.5 | 44.3 | 52.4 | 49.4 | **52.8** |
| **Math (Matematika)** |
| GSM8K (EM) | 8-shot | 63.4 | 77.9 | 80.3 | **83.0** | *79.2* |
| MATH (EM) | 4-shot | 18.7 | 41.4 | *42.5* | 42.2 | **43.6** |
| **Chinese (Mandarin)** |
| C-Eval (Acc.) | 5-shot | 66.1 | **83.7** | 59.6 | 67.5 | *81.7* |
| CMMLU (Acc.) | 5-shot | 70.8 | **84.3** | 60.0 | 69.3 | *84.0* |
| CHID (Acc.) | 0-shot | 92.1 | - | 57.0 | 83.2 | **92.7** |

**Kajian Akademis:**
1.  **Dilema Parameter Aktif:** Perhatikan baris *# Activated Params*. LLaMA 3 membakar komputasi 70B per token. Qwen1.5 membakar 72B. Mixtral membakar 39B. DeepSeek-V2? **Hanya 21B**. Ini adalah mesin paling hemat energi di tabel ini.
2.  **Kejeniusan Matematika:** Walaupun sangat irit, DeepSeek-V2 menduduki peringkat #1 absolut pada tolok ukur MATH (43.6), mengalahkan model boros seperti LLaMA 3 70B (42.2).
3.  **Hibrida Bahasa yang Agresif:** Pada ujian bahasa Inggris umum, LLaMA 3 70B sedikit memimpin. Namun pada ujian literasi berbahasa Mandarin, LLaMA 3 runtuh, sementara DeepSeek-V2 menempel ketat Qwen 72B (model Tiongkok). DeepSeek-V2 terbukti sebagai entitas AI bilingual (dwibahasa) yang paling tangguh.

#### 3.2.3 Efisiensi Pelatihan dan Inferensi
Berdasarkan fakta empiris di lapangan (menggunakan klaster H800), untuk melatih 1 Triliun token, model "Dense" 67B memakan **300.600 Jam GPU**. Dengan mempekerjakan *sparse architecture* MoE, DeepSeek-V2 hanya menelan **172.800 Jam GPU**. Ini memvalidasi penghematan ongkos bakar komputasi sebesar **42,5%**.
Sedangkan saat inferensi (pelayanan produksi), penggabungan kuantisasi tipe FP8, arsitektur MLA (menghilangkan leher botol KV cache), memampukan satu server tunggal yang berisi 8 H800 melayani perputaran teks secepat > 50.000 token per detik.

---

## 4. Penyelarasan (*Alignment*)

Agar AI mentah (*base model*) dapat diajak mengobrol dan tidak memberikan jawaban toksik, AI harus disejajarkan (*alignment*) dengan kehendak manusia. Ada dua tahap: SFT dan RL.

### 4.1 Pelatihan Halus Terselia (*Supervised Fine-Tuning* / SFT)
Disusun dataset instruksi berukuran **1,5 Juta instansi** (1,2 Juta untuk instruksi kemanfaatan, 0,3 Juta untuk instruksi keamanan). Model di *fine-tune* selama 2 *epochs*, laju pembelajaran $5 \times 10^{-6}$. Model ini disebut **DeepSeek-V2 Chat (SFT)**.

### 4.2 Pembelajaran Penguatan (*Reinforcement Learning* / RL)
Untuk memaksimalkan kesopanan, keamanan, dan kebenaran jawaban, digunakan Pembelajaran Penguatan. Peneliti menggunakan **Group Relative Policy Optimization (GRPO)**. Algoritma ini unik karena berani mengeliminasi *Critic Model* (yang ukurannya menyita banyak VRAM GPU) dan menggunakan nilai estimasi berkelompok (grup). 

Fungsi kerugian (Objective) yang dimaksimalkan oleh GRPO adalah:

$$ \mathcal{J}_{GRPO}(\theta) = \mathbb{E} \left[ q \sim P(q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{old}}(o|q) \right] \frac{1}{G} \sum_{i=1}^{G} \left( \min \left( \frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)} A_i, \text{clip} \left( \frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)}, 1 - \epsilon, 1 + \epsilon \right) A_i \right) - \beta \mathbb{D}_{KL} (\pi_\theta || \pi_{ref}) \right), \tag{32} $$

$$ \mathbb{D}_{KL} (\pi_\theta || \pi_{ref}) = \frac{\pi_{ref}(o_i|q)}{\pi_\theta(o_i|q)} - \log \frac{\pi_{ref}(o_i|q)}{\pi_\theta(o_i|q)} - 1, \tag{33} $$

**Penjelasan Variabel Formula (32) dan (33):**
*   $\theta$: Parameter model (kebijakan) yang sedang dilatih. $\theta_{old}$ adalah parameter sesaat sebelumnya.
*   $q$: Kueri (pertanyaan pengguna). $\{o_i\}_{i=1}^G$: Grup berisi $G$ jumlah variasi jawaban (output) yang disajikan model untuk satu pertanyaan $q$.
*   $\pi_\theta(o_i|q)$: Probabilitas model baru menghasilkan jawaban tersebut.
*   $A_i$: Keuntungan (*Advantage*). Mengukur apakah output $o_i$ lebih baik atau lebih buruk dibandingkan rata-rata grup.
*   $\text{clip}(\dots, 1 - \epsilon, 1 + \epsilon)$: Fungsi pembatas (klem) agar pembaruan model tidak melompat terlalu ekstrem dalam satu kali hitungan.
*   $\mathbb{D}_{KL}$: Penalti *Kullback-Leibler Divergence*. Jika model yang dilatih melenceng terlalu menyimpang dari tata bahasa model aslinya ($\pi_{ref}$), ia akan dihukum secara logaritmik (Formula 33). $\beta$ adalah faktor pengali hukuman tersebut.

Untuk kalkulasi *Advantage* ($A_i$) antar grup:

$$ A_i = \frac{r_i - mean(\{r_1, r_2, \dots, r_G\})}{std(\{r_1, r_2, \dots, r_G\})}. \tag{34} $$

**Penjelasan Variabel Formula (34):**
*   $r_i$: Skor *reward* (imbalan) untuk jawaban $i$. Dikurangi rata-rata grup ($mean$) lalu dibagi standar deviasi grup ($std$) agar hasil normalisasi selalu berpusat pada nol (Distribusi Z).

**Strategi RL Dua Tahap:**
*   **Tahap 1 (Penalaran):** Mengoptimalkan kecerdasan logika, kode, dan matematika dengan pelatih skor model $RM_{reasoning}$:
    $$ r_i = RM_{reasoning}(o_i). \tag{35} $$
*   **Tahap 2 (Preferensi Manusia):** Penilaian berlapis dari 3 sumber: model Kemanfaatan ($RM_{helpful}$), model Keamanan ($RM_{safety}$), dan aturan baku linguistik ($RM_{rule}$):
    $$ r_i = c_1 \cdot RM_{helpful}(o_i) + c_2 \cdot RM_{safety}(o_i) + c_3 \cdot RM_{rule}(o_i), \tag{36} $$
    Di mana $c_1, c_2, c_3$ adalah konstanta pengatur prioritas.

### 4.3 Hasil Evaluasi
*   *(Saya tidak akan menyertakan render tabel obrolan panjang/Tabel 3 di sini untuk efisiensi)*, namun kesimpulannya:
*   Tahap SFT melonjakkan kehebatan pemahaman instruksi.
*   Tahap RL meningkatkan kapabilitas panjang obrolan (*open-ended generation*). Di arena **MT-Bench**, DeepSeek-V2 Chat (RL) mencetak skor gila sebesar **8.97** (mengalahkan Mixtral di angka 8.66). Pada arena **AlpacaEval 2.0**, model RL mencetak kemenangan (*win-rate*) sebesar **38.9%**, jauh menumbangkan kehebatan LLaMA 3 70B (34.4%). 

### 4.4 Diskusi

Dua hal penting dipelajari oleh para ilmuwan dari eksperimen gila ini:
1.  **Pajak Penyelarasan (*Alignment Tax*):** Model yang diselaraskan terlalu ketat dengan moralitas manusia pada tahap RL sering kali mengalami *penurunan* kemampuan dalam menjawab tes berbasis logika kaku (seperti *BBH benchmark*). RL menuntut kompromi antara menjadi AI "aman/sopan" vs AI "analitis presisi".
2.  **Kebutuhan Data SFT:** Membantah mitos riset terdahulu yang mengklaim 10K instruksi saja cukup untuk SFT, riset ini membuktikan jika data SFT dipotong terlalu drastis, metrik kepatuhan format instruksi absolut (IFEval) AI akan hancur lebur. Jumlah data padat (*dense data*) tetap tidak tergantikan dalam era AI modern.

---

## 5. Kesimpulan, Limitasi, dan Pekerjaan Masa Depan

Sebagai rangkuman mata kuliah, **DeepSeek-V2** adalah Model Bahasa MoE raksasa berskala 236 Miliar Parameter (dengan hanya 21 Miliar yang beroperasi aktif secara simultan) dan mendukung rentang wawasan (konteks) mencapai 128.000 kata. Model ini mendobrak tembok batasan teori arsitektur tradisional melalui inovasi **Multi-Head Latent Attention (MLA)** dan pemecahan pakar hierarkis **DeepSeekMoE**.

Kemenangan absolut arsitektur ini terlihat dari penghematan gila-gilaan pada anggaran moneter pelatihan sebesar 42,5%, kompresi operasional KV *cache* sebesar 93,3%, dan eskalasi kelancaran pita (*throughput bandwidth*) produksi sebesar 576% bila dikomparasikan terhadap arsitektur lawas, model Padat (DeepSeek 67B). Evaluasi buta dan terstandarisasi menyahihkan posisinya sebagai "Raja Tak Bermahkota" dalam arena model AI sumber terbuka global hari ini.

Namun secara ilmiah, model ini masih mewarisi **Limitasi** endemik generasional:
1. Kerentanan mengalami Halusinasi (menguntai realitas fiktif seolah-olah itu kebenaran deduktif).
2. Kebekuan pengetahuan pasca-pelatihan. AI tidak memiliki mekanisme pemutakhiran otodidak (*ongoing knowledge updates*).
3. Monopoli bilingual Tiongkok-Amerika. Karena struktur korpus data dominan, performa bahasa Eropa-Kontinental, bahasa rumpun Austronesia (seperti Indonesia), atau Semitik, akan tersendat secara linguistik. 

*(Tamat)*