---
slug: bab-1-11
title: "Attention dan MoE"
layout: chapter
---

# BAB 1: Alternatif Mekanisme Perhatian dan Model Campuran Pakar (*Attention Alternatives and Mixtures of Experts*)

## 1.1 Pendahuluan dan Latar Belakang Masalah
Seiring dengan evolusi Model Bahasa Besar (*Large Language Models* / LLMs), ukuran jendela konteks (*context window*) yang dapat diproses oleh model telah mengalami peningkatan yang eksponensial. Jika kita mengamati tren dari tahun 2018 hingga 2025, ukuran konteks berevolusi dari sekadar ribuan token (seperti GPT-1 dan GPT-2) menjadi jutaan token. 

Berdasarkan data observasi empiris pada **Grafik Evolusi Ukuran Jendela Konteks LLM (2018-2025)**:
*   Sumbu X merepresentasikan waktu perilisan model (2018 hingga 2025).
*   Sumbu Y merepresentasikan ukuran jendela konteks dalam skala logaritmik (dari 1.000 hingga 10.000.000 token).
*   Titik-titik data menunjukkan bahwa model awal seperti GPT-3 berada di kisaran ribuan. Namun, di tahun 2024-2025, model dari Anthropic (Claude 2.1/3), Google (Gemini 1.5 Pro), dan Meta (Llama 4 Scout) telah menembus batas 1 juta hingga 2 juta token (*First 1M+ Context Window*).

**Masalah Fundamental:** Biaya komputasi untuk mekanisme perhatian (*attention*) meningkat secara kuadratik seiring dengan panjangnya teks konteks. 
Hal ini diilustrasikan dengan sangat jelas pada **Grafik Profiling Waktu Komputasi**:
*   Sumbu X adalah panjang sekuens/konteks (0 hingga 15.000 token).
*   Sumbu Y adalah waktu eksekusi dalam milidetik (ms).
*   **Area Biru (Feed-forward):** Menunjukkan pertumbuhan waktu yang linier. Lapisan ini memproses setiap token secara independen, sehingga biayanya sebanding dengan jumlah token.
*   **Area Oranye (Attention):** Menunjukkan pertumbuhan waktu yang melengkung tajam ke atas (kuadratik / $O(N^2)$). Pada sekuens pendek, waktu eksekusinya setara dengan *Feed-forward*. Namun, pada panjang sekuens 15.000, komputasi *Attention* mendominasi lebih dari 75% dari total waktu eksekusi (mencapai 600 ms).

Pertanyaan mendasar dalam bab ini adalah: **Bagaimana kita mengendalikan biaya komputasi yang membengkak ini?**

---

## 1.2 Perangkat Dasar (*The Basic Toolkit*) untuk Mengatasi Biaya Perhatian

Sebelum merombak arsitektur, insinyur AI menggunakan dua pendekatan dasar: modifikasi atensi lokal-global dan optimisasi rekayasa sistem.

### 1.2.1 Perhatian Jarang (*Sparse Attention*)
Alih-alih memaksa setiap token untuk "memperhatikan" seluruh token sebelumnya (yang menciptakan matriks padat/$dense$), kita dapat membatasi jangkauan pandangan token.
Pada visualisasi matriks *Sparse Transformer*:
*   **Gambar (a) Transformer Standar:** Menunjukkan matriks segitiga bawah yang padat (seluruh kotak di bawah diagonal berwarna biru tua), merepresentasikan komputasi $O(N^2)$ penuh.
*   **Gambar (b) Sparse Transformer (Strided):** Menunjukkan pola pita diagonal berwarna biru tua dengan titik-titik biru yang melompat-lompat secara periodik. Model hanya melihat token terdekat secara lokal, ditambah beberapa token masa lalu dengan interval tertentu.
*   **Gambar (c) Sparse Transformer (Fixed):** Menggabungkan perhatian lokal berdekatan dengan blok memori global di lokasi tertentu.

Konsep ini diimplementasikan dalam bentuk **Hibrida Atensi Lokal dan Global**. Misalnya, lapisan-lapisan di dalam model dikonfigurasi secara berselang-seling: sebagian besar lapisan hanya menggunakan *Local Attention* (misalnya melihat 4096 token terdekat), sementara lapisan terakhir dikhususkan sebagai *Global Attention* untuk membaca seluruh konteks.

### 1.2.2 Rekayasa Sistem (*Systems Engineering*): FlashAttention
Pendekatan kedua tidak mengubah matematika dari *attention*, melainkan mengoptimalkan pergerakan data di dalam memori GPU (SRAM vs HBM).
Berdasarkan **Grafik Kecepatan Forward + Backward Attention (A100 80GB)**:
*   Sumbu X: Panjang sekuens (512 hingga 16k).
*   Sumbu Y: Kecepatan komputasi (TFLOPs/s) - semakin tinggi semakin baik.
*   Implementasi standar PyTorch (Batang biru) dan xformers (Batang hijau) mengalami kegagalan memori (*Out Of Memory* / OOM) pada panjang 8k dan 16k.
*   Inovasi **FlashAttention-2** (Batang ungu) mampu mempertahankan kecepatan tinggi yang stabil (di kisaran 162 hingga 176 TFLOPs/s) bahkan hingga sekuens 16.000 token, menjadikannya standar mutlak (*basic toolkit*) saat ini.

Namun, jika kita menginginkan lonjakan efisiensi yang lebih radikal, kita harus mengeksplorasi matematika yang sepenuhnya berbeda.

---

# BAB 2: Perhatian Linier (*Linear Attention*) dan Varian Hibrida

## 2.1 Konsep Matematis Perhatian Linier
Pertimbangkan operasi perhatian konvensional. Kita memiliki matriks Query ($Q \in \mathbb{R}^{n \times d_k}$), Key ($K \in \mathbb{R}^{n \times d_k}$), dan Value ($V \in \mathbb{R}^{n \times d_v}$).
Rumus standarnya adalah:
$$Attn(Q, K, V) = \rho(QK^\top)V$$
Di mana $\rho$ adalah fungsi aktivasi *Softmax*. Komputasi ini bersifat kuadratik karena operasi $(QK^\top)$ menghasilkan matriks perantara berukuran $n \times n$, sehingga biayanya adalah $O(n^2 d_k)$.

**Solusi Perhatian Linier:** Bisakah kita mempercepat ini? Ya, dengan syarat fungsi $\rho$ dihapus (dianggap sebagai matriks identitas atau fungsi kernel linier). Jika $\rho$ dihilangkan, kita dapat memanfaatkan sifat asosiatif perkalian matriks:
$$(QK^\top)V = Q(K^\top V)$$
Perubahan urutan kurung ini terlihat sederhana namun berimplikasi sangat masif. Kita mengalikan $(K^\top V)$ terlebih dahulu yang menghasilkan matriks kecil berukuran $d_k \times d_v$. Kemudian $Q$ dikalikan dengan hasil tersebut.
Kompleksitas komputasi turun drastis dari $O(n^2 d_k + n^2 d_v)$ menjadi **$O(2n d_v d_k)$**. Waktu komputasi kini berjalan secara linier terhadap panjang sekuens ($n$).

## 2.2 Bentuk Rekuren dari Perhatian Linier (Dualitas dengan RNN)
Lebih jauh lagi, persamaan $Q(K^\top V)$ memiliki properti menakjubkan: ia setara dengan Jaringan Saraf Rekuren (*Recurrent Neural Network* / RNN).

Secara matematis, keadaan tersembunyi (*hidden state*) $S_t$ pada langkah waktu ke-$t$ dapat ditulis sebagai:
$$S_t = S_{t-1} + k_t v_t^\top \quad \text{dan} \quad y_t = q_t^\top S_t$$
*Di mana:*
*   $S_{t-1}$ adalah akumulasi memori dari masa lalu.
*   $k_t v_t^\top$ adalah informasi baru dari token saat ini yang ditambahkan ke memori.
*   $y_t$ adalah output prediktif dari perbandingan antara $q_t$ (Query saat ini) dengan memori $S_t$.

**Keuntungan Dualitas:** Sifat ini memungkinkan kita untuk melakukan proses *training* secara efisien di GPU menggunakan bentuk matriks kuadratik yang paralel, namun mengeksekusi *inference* (teks generasi) secara super efisien menggunakan bentuk rekuren (serial) yang berjalan dalam waktu linier. *(Catatan: Jika state $S_{t-1}$ dikalikan dengan faktor peluruhan $\gamma$, kita mendapatkan arsitektur RetNet).*

## 2.3 Model Hibrida Linier-Atensi Modern

Meskipun murni linier sangat cepat, kualitasnya sering kali tertinggal dari Transformer murni. Solusinya adalah mencampurkan keduanya (*Hybrids*).

### 2.3.1 Minimax M1
Model Minimax M1 (dan versi teksnya) menggunakan rasio hibrida 7-banding-1 (7 lapisan *linear attention*, 1 lapisan *full attention*).
*   Pada **Grafik Akurasi (AIME, LiveCodeBench, dll)**, Minimax M1 (batang oranye pekat) menunjukkan performa yang mampu bersaing erat (atau melampaui) model-model raksasa berbobot terbuka (*open-weight*) seperti DeepSeek-R1 dan Qwen3.
*   Pada **Grafik FLOPs vs Generation Length**, kurva Minimax M1 (merah) tumbuh secara **linier** (garis lurus landai), sangat kontras dengan kurva DeepSeek-R1 dan Qwen3 yang melengkung tajam ke atas secara kuadratik seiring bertambahnya panjang generasi dari 32K hingga 128K token.
*   **Arsitektur:** Menggunakan `Lightning Attention` untuk lapisan linier, yang diapit oleh `RMSNorm`, dan diselingi dengan `Softmax Attention` pada lapisan tertentu, diakhiri dengan modul *Mixture of Experts* (MoE).

### 2.3.2 Mamba-2 dan Gated Delta Net (GDN)
Kita dapat memodifikasi persamaan linier rekuren di atas agar lebih ekspresif dengan menambahkan bobot per-posisi (mekanisme *Gating*). Ingat: "Gating is good!"
Persamaan dasar linier atensi:
$$S_t = S_{t-1} + k_t v_t^\top \quad \text{dan} \quad y_t = q_t^\top S_t$$
Dimodifikasi pada **Mamba-2** menjadi:
$$S_t = \gamma_t S_{t-1} + k_t v_t^\top \quad \text{dan} \quad y_t = q_t^\top S_t + v_t^\top D \quad \text{dengan} \quad \gamma_t = f(x_t)$$
Penambahan $\gamma_t$ (sebagai fungsi input $x_t$) memungkinkan model untuk secara dinamis melupakan (*forget*) masa lalu.

Konsep ini digeneralisasi lebih ekstrim pada **Gated Delta Net (GDN)**. GDN secara selektif membatasi input dan secara paksa "menghapus" status memori tertentu.
$$S_t = \gamma_t(I - \beta_t k_t k_t^\top)S_{t-1} + \beta_t k_t v_t^\top \quad \text{dan} \quad y_t = q_t^\top S_t \quad \text{dengan} \quad \gamma_t = f(x_t), \beta_t = f(x_t)$$
*Penjelasan Komponen:*
*   $\beta_t$ bertindak sebagai gerbang. Jika $\beta_t = 0$, tidak ada operasi input yang terjadi (*no input operation*).
*   Istilah $(I - \beta_t k_t k_t^\top)$ berfungsi untuk **menghapus** (*erase*) informasi apa pun di dalam memori ($S_{t-1}$) yang mengarah ke arah vektor kunci ($k_t$) saat ini. Mekanisme ini mirip dengan ide pemrograman bobot cepat (*fast weight programming*).

**Penerapan Empiris:**
*   **Nemotron 3 (NVIDIA):** Menggunakan hibrida Mamba (rasio sekitar 3:1). Grafik komparasinya menunjukkan Nemotron 3 mengungguli Qwen3 dan GPT-OSS dalam *benchmark* Chat, Math, dan akurasi, sambil memberikan *Throughput* (kecepatan token/detik) yang jauh lebih tinggi.
*   **Qwen 3.5 / Qwen Next:** Menggunakan hibrida 3 GDN berbanding 1 Attention. Pada grafik *Decode Throughput vs Context Length*, Qwen3-Next mampu berjalan hampir 11 kali lipat lebih cepat daripada arsitektur Qwen3-32B konvensional saat menangani konteks 128K token.

---

# BAB 3: Adaptasi Perhatian Jarang Secara Presisi (DSA)

Jika kita tidak ingin menggunakan model hibrida rekuren/linier, alternatif lainnya adalah mengembangkan mekanisme *Sparse Attention* murni yang sangat cerdas. DeepSeek mempelopori ini dengan **DeepSeek Sparse Attention (DSA)** yang diterapkan pada arsitektur v3.2 dan GLM5.

Alih-alih menyebar pandangan atensi secara statis/kaku, DSA (DSA Prototype) menggunakan dua komponen utama: **Lightning Indexer** dan **Mekanisme Seleksi Token Presisi-Tinggi (*Fine-Grained*)**.

**Metodologi Matematis DSA:**
Indexer ini menghitung skor ketertarikan (indeks) $I_{t,s}$ antara token *query* ($h_t \in \mathbb{R}^d$) dan token-token masa lalu ($h_s \in \mathbb{R}^d$):
$$I_{t,s} = \sum_{j=1}^{H^I} w_{t,j}^I \cdot \text{ReLU} (q_{t,j}^{I} \cdot k_s^I) \quad \dots \dots (1)$$

*Penjelasan Variabel:*
*   $H^I$ adalah jumlah *indexer heads* (kepala indeks pencari).
*   $q_{t,j}^I \in \mathbb{R}^{d^I}$ dan $w_{t,j}^I \in \mathbb{R}$ diturunkan dari token *query* $h_t$.
*   $k_s^I \in \mathbb{R}^{d^I}$ diturunkan dari token riwayat/sebelumnya $h_s$.
*   Fungsi aktivasi ReLU dipilih murni karena pertimbangan *throughput* sistem. Karena komponen ini kecil, ia dieksekusi dalam format komputasi FP8 sehingga efisiensi matematisnya luar biasa cepat.

Setelah sekumpulan skor $\{I_{t,s}\}$ dieksekusi untuk setiap query, mekanisme seleksi mengambil sejumlah entri memori (Key-Value $\{c_s\}$) yang secara spesifik memiliki skor indeks tertinggi (*top-k*). Output akhir dihitung dengan melakukan Atensi standar HANYA pada token yang terpilih tersebut:
$$u_t = \text{Attn}(h_t, \{c_s | I_{t,s} \in \text{Top-k}(I_t; \cdot) \}) \quad \dots \dots (2)$$

**Hasil Empiris DSA:**
*   Pada sekumpulan grafik diagram batang performa (AIME 2025, HMMT, HLE, Codeforces, SWE, dll), DeepSeek-V3.2 (biru tua) dan GLM-5 (biru cerah) terbukti setara atau bahkan mengungguli model padat raksasa seperti Claude 3.5 Sonnet, Gemini 3 Pro, dan GPT-4o.
*   Pada **Grafik Biaya Prefilling dan Decoding** (Sumbu Y = Biaya per juta token, Sumbu X = Panjang token 0 hingga 128K), DeepSeek-V3.2 (tanpa DSA) memperlihatkan biaya yang meroket secara kuadratik hingga mencapai $0.75 pada Prefill dan $2.4 pada Decode. Sebaliknya, varian **Terminus (dengan DSA)** (garis oranye) tetap mendatar (*flat*) di angka $0.15 hingga ujung skala 128K. 
*   **Catatan Penting:** DSA dapat diadaptasikan secara *'post hoc'*. Artinya, model dapat dipra-latih (*pretrained*) menggunakan konteks pendek yang padat secara normal, lalu *Lightning Indexer* disuntikkan dan dilatih (*warm-up*) belakangan. (Rujukan: Tabel 6 RULER Benchmark menunjukkan GLM-4.7-Flash + DSA warmup mencapai akurasi tinggi tanpa perlu melatih ulang model dasar dari nol).

---

# BAB 4: Arsitektur Model Campuran Pakar (*Mixture of Experts* / MoE)

## 4.1 Definisi dan Konsep Dasar MoE
Pergeseran paradigma paling berpengaruh dalam LLM open-source modern (seperti Mixtral, Qwen, DeepSeek, Grok) adalah penggunaan *Mixture of Experts* (MoE). Rumor di komunitas akademik dan industri kuat mensugestikan bahwa GPT-4 sendiri adalah arsitektur MoE (beranggotakan 8 atau 16 pakar).

**Apa itu MoE?**
Mari bedah secara visual perbandingan antara Model Padat (*Dense Model*) dan Model Jarang (*Sparse Model* / MoE).
*   **Dense Model:** Input $X$ masuk ke lapisan *Self-Attention*, dijumlahkan/dinormalisasi, lalu seluruh data diteruskan ke sebuah lapisan **Feed-Forward Network (FFN)** yang berukuran raksasa.
*   **Sparse Model (MoE):** Lapisan FFN raksasa tersebut dibongkar dan diganti dengan beberapa FFN yang lebih kecil namun berjumlah banyak (disebut "Pakar" / *Experts*). Sebelum data masuk ke FFN, data diproses oleh **Router (Lapisan Penyeleksi)**. Router ini mengevaluasi token (misalnya token "The" dan "Dog") dan mengirimkan masing-masing token HANYA ke satu atau dua FFN pakar spesifik (misalnya FFN 1 dan FFN 2), sementara pakar lainnya (FFN 3 dan FFN 4) mati atau tidak diaktifkan (*dormant*) untuk perhitungan token tersebut.

Konsep esensialnya adalah: **Anda dapat melipatgandakan ukuran jumlah parameter model secara keseluruhan (menambah jumlah pakar) TANPA meningkatkan jumlah FLOPs (kalkulasi komputasi aktif) yang tereksekusi pada setiap langkah inferensi.**

## 4.2 Mengapa MoE Menjadi Sangat Populer?

1.  **Hukum Skala yang Superior (FLOP yang Sama, Parameter Lebih Banyak, Hasil Lebih Baik):**
    Berdasarkan plot logaritmik "Sparse Model Parameters" versus "Test Loss" (Fedus et al., 2022), titik data bergeser dari model dasar T5 (garis ungu) menuju varian Switch-Base 16e, 32e, 64e, 128e, hingga 256e (pakar). Semakin banyak pakar yang ditambahkan (sehingga jumlah parameter melonjak hingga $10^{10}$), *Test Loss* (Perplexity) secara konsisten terjun bebas ke angka yang semakin kecil.
2.  **Jauh Lebih Cepat untuk Dilatih:**
    Pada grafik kurva kerugian berbanding waktu komputasi, model Switch-Base 128e mampu menembus target batas metrik *Neg Log Perplexity* (-1.55) nyaris **7x lebih cepat** (secara iterasi/waktu) dibandingkan model Dense T5 konvensional berukuran setara FLOP. Data dari paper OLMoE juga mendemonstrasikan bahwa melatih model MoE aktif 1.3B (Total 6.9B parameter) mencapai akurasi HellaSwag yang sama dengan model dense konvensional dalam waktu yang **~2x hingga 3x lebih cepat**.
3.  **Sangat Kompetitif secara Performa:**
    Grafik sebar (*scatter plot*) "Performance (MMLU)" versus "Activated Parameters" menunjukkan bahwa model MoE modern (DeepSeek-V2, Mixtral 8x22B, Qwen1.5 72B) berkerumun di kuadran efisiensi tinggi (kiri atas). Mereka mencapai tingkat MMLU di kisaran 75-80 dengan menggunakan parameter aktif yang jauh lebih sedikit dibandingkan model Dense keluarga LLaMA 3 70B atau Command R+.
4.  **Skalabilitas Paralelisasi pada Perangkat:**
    Arsitektur MoE sangat ramah terhadap sistem kluster superkomputer (misalnya ribuan GPU). Karena blok FFN pakar berdiri sendiri, arsitektur ini menerapkan prinsip **Model-Parallel MoE** (Expert Parallelism). Output *Encoder* dipecah (sharded), lalu disebarkan melalui jaringan komunikasi *All-to-All Dispatch* ke Perangkat 1 hingga Perangkat E yang menampung spesifik FFN pakar. Setelah selesai diproses, data dikumpulkan kembali melalui kombinasi *All-to-All Combine*.

---

# BAB 5: Konfigurasi dan Routing Pakar (Expert Routing)

Dalam merancang MoE, teknisi AI harus mengeksplorasi beberapa variabel desain: Fungsi *Routing* (bagaimana cara memilih pakar), Ukuran Pakar, dan Objektif/Tujuan Pelatihan.

## 5.1 Overview Fungsi Routing
Mayoritas algoritma *routing* pada akhirnya mengerucut pada prinsip "pilih K terbaik" (*choose top k*). Visualisasi kotak matriks probabilitas antara Tokens (T1, T2, T3) dan Pakar (E1 hingga E5) memperlihatkan tiga paradigma utama:
*   **Token memilih Pakar (*Token chooses expert*):** Model melihat token tunggal T1, lalu memeriksa probabilitas kelima pakar, dan memilih pakar dengan probabilitas tertinggi (misal E1 dan E5 terpilih untuk T1 karena nilainya tinggi 3.13 dan 2.81). Ini adalah standar industri.
*   **Pakar memilih Token (*Expert chooses token*):** Setiap pakar E1 mencari ke seluruh sekuens konteks dan mengambil paksa token yang paling relevan untuknya (metode ini rentan, karena beberapa token mungkin terbuang/tidak diproses sama sekali).
*   **Routing Global via Optimisasi (*Global routing via optimization*):** Menggunakan penyelesaian matematis (seperti *linear assignment*) untuk memetakan distribusi tugas secara rata sempurna secara komprehensif, namun sayangnya terlalu lambat untuk dijalankan secara langsung saat pelatihan model. *(Digunakan dalam eksperimen Clark 2022).*

Selain itu, terdapat metode **RL (Reinforcement Learning)** untuk melatih pembuat rute. Meskipun metode ini secara matematis merupakan "solusi yang paling tepat" dan pernah diuji (Bengio 2013), variansi gradien yang sangat fluktuatif dan kompleksitas penerapannya membuatnya **tidak digunakan secara umum di era modern**.
Sedangkan metode **Hashing** (mengarahkan rute berdasarkan nilai hash/enkripsi kriptografi dari token tanpa probabilitas neural sama sekali) sering digunakan sebagai titik dasar referensi (*baseline*).

## 5.2 Mekanika Matematis Top-K Routing

Metode yang menjadi de-facto standar (digunakan oleh Switch Transformer, Mixtral, Grok, Qwen, DeepSeek) adalah fungsi klasifikasi logistik **Top-K**.

**Definisi Matematis Detail (Berdasarkan Dai et al., 2024):**
Output agregasi representasi tersembunyi ($h_t^l$) untuk token ke-$t$ di lapisan ke-$l$ diformulasikan sebagai:
$$h_t^l = \sum_{i=1}^N \left( g_{i,t} \text{FFN}_i (u_t^l) \right) + u_t^l$$

*Penjelasan Variabel Utama:*
*   $N$ adalah total jumlah pakar yang tersedia.
*   $u_t^l$ adalah representasi token input yang masuk ke lapisan.
*   $\text{FFN}_i$ adalah operasi *feed-forward* dari pakar spesifik ke-$i$.
*   $g_{i,t}$ adalah nilai aktivasi (pembobot) gerbang (*Gating value*). Ia mengontrol seberapa kuat sinyal dari pakar ke-$i$ diikutsertakan.

Nilai dari $g_{i,t}$ ditentukan oleh sebuah mekanisme pemilihan tajam:
$$g_{i,t} = \begin{cases} 
s_{i,t}, & s_{i,t} \in \text{Topk}(\{s_{j,t} | 1 \leqslant j \leqslant N\}, K), \\
0, & \text{otherwise} 
\end{cases}$$
Skor afinitas mentah $s_{i,t}$ dikalkulasi melalui regresi logistik linier dan diloloskan ke fungsi probabilitas *Softmax*:
$$s_{i,t} = \text{Softmax}_i \left( u_t^{l \top} e_i^l \right)$$
*(Di mana $e_i^l$ adalah vektor parameter centroids router yang dapat dipelajari).*

**Variasi Modifikasi Penting:**
*   **DeepSeek (V1-2) / Grok / Qwen:** Menerapkan fungsi *Softmax* terlebih dahulu pada semua skor, baru kemudian memfilter Top-K dan mengabaikan probabilitas sisanya menjadi 0.
*   **Mixtral / DBRX / DeepSeek v3:** Mengambil Top-K nilai afinitas mentah terlebih dahulu, lalu menerapkan fungsi *Softmax* **hanya** pada himpunan K yang terpilih tersebut. Hal ini mengubah gradien backpropagasi.

## 5.3 Inovasi Arsitektur MoE China (Pakar Berbagi & Segmentasi Halus)

Para teknisi di DeepSeek dan Qwen (mengadopsi dari kerangka DeepSpeed) memperkenalkan dua modifikasi radikal pada struktur fisik pakar yang terbukti sangat efektif:

1.  **Segmentasi Pakar Lebih Halus (*Fine-grained Expert Segmentation*):** Daripada memiliki 8 pakar berukuran raksasa, mereka memecahnya menjadi misalnya 64 pakar berukuran kecil, dan memilih K yang lebih besar secara proporsional. Ini memberikan fleksibilitas komputasi.
2.  **Isolasi Pakar Terbagi (*Shared Expert Isolation*):** Mereka mencadangkan beberapa blok pakar (misal 1 atau 2 blok) untuk selalu menyala (*always on*) dan mengevaluasi setiap token yang lewat tanpa syarat, terlepas dari hasil putusan *Router*. Pakar abadi ini (berwarna hijau pada diagram) dikhususkan untuk mempelajari "pengetahuan fundamental/gramatikal" umum, sementara pakar khusus (*Routed Expert* - warna biru) menangani pengetahuan domain spesifik.

**Data Ablasi Empiris (Dari Makalah DeepSeek V1):**
Pada diagram batang pengujian berbagai konfigurasi MoE terhadap model padat berukuran 0.2B aktif:
*   Batang biru muda (0 pakar bersama + 2 pakar khusus) menjadi patokan (*baseline*).
*   Batang oranye/coklat (*1 shared expert + 7 routed experts dengan fine-grained segmentation*) secara meyakinkan melampaui semua metrik (HellaSwag, PIQA, ARC, TriviaQA, NaturalQuestions) hingga mencapai indeks performa normalisasi 1.0. Ini membuktikan kombinasi memecah pakar dan menanamkan pakar permanen meningkatkan kecerdasan model secara drastis. Berlawanan dengan ini, makalah **OLMoE (Barat)** tidak menemukan peningkatan berarti dari *shared experts*, melainkan hanya mengandalkan *fine-grained experts* murni.

### Tabel Konfigurasi Routing MoE Modern (Standar Industri)

| Model | Routed (Khusus) | Active (K) | Shared (Bersama) | Fine-grained ratio |
| :--- | :--- | :--- | :--- | :--- |
| GShard | 2048 | 2 | 0 | |
| Switch Transformer | 64 | 1 | 0 | |
| ST-MOE | 64 | 2 | 0 | |
| Mixtral | 8 | 2 | 0 | |
| DBRX | 16 | 4 | 0 | |
| Grok | 8 | 2 | 0 | |
| DeepSeek v1 | 64 | 6 | 2 | 1/4 |
| Qwen 1.5 | 60 | 4 | 4 | 1/8 |
| DeepSeek v3 | 256 | 8 | 1 | 1/14 |
| OlMoE | 64 | 8 | 0 | 1/8 |
| MiniMax | 32 | 2 | 0 | ~1/4 |
| Llama 4 (maverick) | 128 | 1 | 1 | 1/2 |

---

# BAB 6: Dinamika, Kestabilan, dan Sistem Pelatihan MoE

## 6.1 Tantangan Pelatihan: Memaksa Distribusi Tugas (Load Balancing)
Tantangan terbesar dalam melatih MoE adalah efisiensi operasional jaringan. Jika dibiarkan bebas, *Router Neural Network* cenderung menjadi "malas" dan secara terus-menerus mengarahkan seluruh miliaran token hanya ke 1 atau 2 pakar terfavorit. Jika hal ini terjadi (disebut *expert collapse* / keruntuhan pakar), sisa pakar lainnya akan mati kelaparan, dan GPU yang menampungnya akan menganggur menunggu instruksi (*bottleneck* keras).

Untuk mencegah ini, kita tidak menggunakan RL, melainkan menambahkan fungsi **Heuristic Balancing Losses (Auxiliary Loss)** pada persamaan kesalahan (Loss Function).

**Formulasi Switch Transformer (Fedus et al., 2022):**
Untuk setiap lapisan Switch, fungsi penalti ditambahkan selama komputasi:
$$\text{loss} = \alpha \cdot N \cdot \sum_{i=1}^N f_i \cdot P_i \quad \dots \dots (4)$$

Di mana variabel penyeimbang dihitung sebagai:
*   $f_i$ adalah pecahan/fraksi statistik dari kumpulan token $\mathcal{B}$ di dalam *batch* yang sebenarnya dialokasikan ke pakar ke-$i$.
    $$f_i = \frac{1}{T} \sum_{x \in \mathcal{B}} \mathbf{1} \{\text{argmax } p(x) = i\} \quad \dots \dots (5)$$
*   $P_i$ adalah fraksi akumulatif dari *probabilitas sinyal aktivasi mentah* router yang didelegasikan untuk pakar $i$.
    $$P_i = \frac{1}{T} \sum_{x \in \mathcal{B}} p_i(x) \quad \dots \dots (6)$$

**Logika Kerja:** Karena komponen diskrit fungsi batas (argmax di $f_i$) tidak dapat di-*backpropagate* (tidak memiliki turunan), kita menyandarkan turunan perambatannya pada variabel probabilitas kontinyu $P_i$.
Turunan kerugian (*derivative*) terhadap $p_i(x)$ adalah: $\frac{\alpha N}{T^2} \sum \mathbf{1}_{\text{argmax } p(x)=i}$.
Artinya: **Semakin sering sebuah pakar digunakan, sistem secara eksponensial akan mendorong penalti (downweighting) gradien ke bawah pada router tersebut, memaksa router untuk mencari dan menghidupkan pakar-pakar lain yang terbengkalai.**

**DeepSeek V3 dan Inovasi "Auxiliary-Loss Free":**
Menghitung kerugian auxiliary di atas menimbulkan *overhead* komputasi. DeepSeek V3 bereksperimen dengan mengubah paradigma ini. Mereka memperkenalkan bias per-pakar ($b_i$) yang direvisi secara *online* selama penjalaran model.
$$g'_{i,t} = \begin{cases} 
s_{i,t}, & s_{i,t} + b_i \in \text{Topk}(\{s_{j,t} + b_j | 1 \leqslant j \leqslant N_r\}, K_r), \\
0, & \text{otherwise} 
\end{cases}$$
Semakin lapar sebuah pakar, bias $b_i$ nya dinaikkan secara artifisial, meningkatkan probabilitas dia memenangkan filter Top-K tanpa membebani kalkulasi *backpropagation* gradien kerugian. (Meskipun demikian, mereka masih menyimpan kerugian auxiliary berbasis urutan/sekuens untuk mencegah ketidakseimbangan mikroskopik).

**Apa yang terjadi jika Load Balancing Loss dihilangkan?**
Grafik eksperimen dari OLMoE membuktikan kehancurannya. Tanpa *Load Balancing Loss* (garis biru pada plot performa), *Validation loss* tidak akan pernah turun secara konvergen dan tertahan di langit-langit (kegagalan belajar). Lebih parah lagi, plot penugasan pakar (*Expert assignment*) menunjukkan bahwa Pakar 2 dan Pakar 6 mengambil 100% kapasitas beban trafik (meng-hijack seluruh model), sedangkan 6 pakar sisanya menyentuh rasio 0%.

## 6.2 Stabilitas Matematis Softmax Router
Sebuah isu tersembunyi yang krusial pada model MoE disoroti oleh Zoph (2022). Terdapat instabilitas luar biasa yang membuat metrik *training loss* hancur seketika saat dilatih.
Akar persoalan: Fungsi eksponensial di dalam perhitungan probabilitas Softmax. 
Apabila model menggunakan akurasi presisi komputasi rendah `bfloat16`, dan terdapat 1 logit dengan angka raksasa $128.5$ berdampingan dengan logit bernilai $128$.
Karena `bfloat16` memotong desimal, komputer mengubah perhitungan matematis normal (di mana seharusnya $128.5$ menang $36\%$ lebih besar secara probabilitas) menjadi hasil di mana eksponensial membuang perbedaan nilai tersebut dan secara salah menyamakan seluruh probabilitas menjadi sama persis secara prematur. Kesalahan pembulatan (*roundoff error*) fatal ini menghancurkan rute neural.

**Solusi Teknisi:** 
1.  Mengisolasi modul penyeleksi pakar (Router) dan memaksanya untuk SELALU beroperasi pada level matematika presisi tinggi murni komputasi bilangan titik kambang **Float 32**, biarpun sisa 99% bagian model lain berjalan pada resolusi Bfloat16.
2.  Menambahkan peredam kestabilan yang disebut **Aux Z-loss** pada fungsi router untuk menekan logit agar angkanya tidak beranjak meledak menjadi raksasa:
    $$L_z(x) = \frac{1}{B} \sum_{i=1}^B \left( \log \sum_{j=1}^N e^{x_j^{(i)}} \right)^2 \quad \dots \dots (5)$$

## 6.3 Paralelisme Perangkat dan Isu Stokhastik (Faktor Keacakan)
Karena MoE harus didistribusikan antar node superkomputer (misalnya melalui arsitektur pustaka MegaBlocks yang menyusun Matriks Perkalian Jarang /*Sparse MMs*), terjadi satu perilaku stokhastik yang tidak dapat dihindari: **Penjatuhan Token di Level Batch (*Token Dropping*)**.
Setiap pakar perangkat keras memiliki batas kuota muatan memori (*Capacity Factor*). 
Sistem mengeksekusi algoritma: (1) Routing -> (2) Permutasi -> (3) Komputasi -> (4) Un-Permutasi.
Jika sebuah *batch* dikirimkan ke mesin dan secara acak mayoritas token pada iterasi tersebut memilih Pakar 2, kapasitas buffer Pakar 2 akan penuh. Akibatnya, token yang datang terlambat akan **dijatuhkan/dihapus (*dropped*)** dan dibuang tanpa melalui *feed-forward* sama sekali.
Artinya, determinisme hilangnya kalimat: token yang dikirimkan oleh pengguna/pelanggan Anda pada antarmuka, mungkin tidak terproses bukan karena salah modelnya, namun sekadar karena pada mikrodetik tersebut ia tidak sengaja digabung pada proses memori *batch* GPU yang sama dengan kueri panjang pihak pengguna lain yang memicu kemacetan router!

---

# BAB 7: Lanjutan MoE (Daur Ulang dan Arsitektur DeepSeek)

## 7.1 Upcycling (Pendaurulangan Model Dense Menjadi MoE)
Haruskah kita membuang jutaan dolar GPU untuk melatih MoE dari titik 0? Tidak. Komunitas menginisiasi ide cerdas bernama **Upcycling**.
Kita dapat mengambil model dasar *Pre-trained Dense LM* yang matang, mengkopi-paste (mendaur ulang) secara identik bobot/angka FFN tunggalnya ke dalam $N$ blok replika tiruan sebagai Pakar yang identik, me-reset bobot router acak, dan melatihnya (melalui proses konvergensi cepat).
*Metrik membuktikan:* Model *Upcycled* (titik jingga) menghabiskan biaya komputasi yang jauh lebih rendah daripada melatih ulang secara *Dense* dengan ukuran matriks yang sama, sembari mengunci akurasi validasi superior.

**Tabel Konfirmasi Sukses Upcycling MiniCPM:**
MiniCPM (TopK=2, 8 pakar, ~4B active params) didaur ulang dari basis model 2.4B Dense (520 miliar token latih awal).

| Model | C-Eval | CMMLU | MMLU | HumanEval | MBPP | GSM8K | MATH | BBH |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| MiniCPM-2.4B (Dense) | 51.13 | 51.07 | 53.46 | 50.00 | 47.31 | 53.83 | 10.24 | 36.87 |
| **MiniCPM-MoE (13.6B)** | **58.11** | **58.80** | **58.90** | **56.71** | **51.05** | **61.56** | **10.52** | **39.22** |

*(Pendekatan serupa sukses diaplikasikan ke Qwen1.5-MoE-A2.7B yang diinisialisasi dari model kompak Qwen 1.8B).*

## 7.2 Anatomi Revolusi DeepSeek MoE (V1 hingga V3)
Untuk merangkum seluruh materi pembelajaran, kita membedah evolusi model DeepSeek.

### Fase 1: DeepSeek MoE V1 (16B Total, 2.8B Aktif)
Menggunakan standar yang diusung makalah mereka: Menggabungkan 2 pakar abadi/bersama (*Shared Experts*) dipadu dengan 64 pakar halus (memanggil 4 aktif sekaligus). Masih mempergunakan teknik probabilitas *Top-K Routing* klasik beserta kerugian penyeimbang (*Aux-loss balancing*).

### Fase 2: DeepSeek MoE V2 (236B Total, 21B Aktif)
Eksperimen berskala raksasa mengundang masalah komunikasi bandwidth jaringan GPU (NVLink). 
*Solusi inovatif V2:* Selain menambah pakar halus menjadi 160 (diaktifkan 10), mereka memperkenalkan **Top-M Device Routing**. Sebelum memilih pakar, mereka menyeleksi perangkat/node GPU mana (Max $M$ node) yang akan dikirim, guna meredam beban kabel antar-perangkat. Terdapat rumusan matematika baru: $L_{CommBal} = \alpha_3 \sum f_i'' P_i''$ (objektif komputasi keseimbangan komunikasi trafik antar mesin jaringan di tingkat makro).

### Fase 3: DeepSeek MoE V3 (671B Total, 37B Aktif)
Perbaikan pamungkas pada MoE mutakhir:
1.  **Arsitektur Router:** 1 Shared + 258 Fine-grained (8 aktif). Menggunakan gabungan fungsi aktivasi pelik *Sigmoid+Softmax* sebelum ekstraksi Top-K/Top-M. 
2.  **Multi-Head Latent Attention (MLA):** Menghancurkan *Bottleneck* KV Cache sistem memori saat membaca data panjang. Menyatukan Q, K, dan V ke dalam turunan vektor latensi kompresi kecil tunggal ($c_t^{KV}$). Karena RoPE (Penyematan Rotary) bergantung pada urutan absolut yang merusak matriks kompaksi laten ini, sistem merekonstruksi persamaan RoPE dengan mencopot beberapa dimensi kunci spesifik untuk dieksekusi secara otonom di luar sirkuit latensi. 
3.  **Multi-Token Prediction (MTP):** DeepSeek V3 mendirikan modul transformator independen berukuran ringan di ekor lapisan atasnya. Alih-alih menerka hanya 1 kata/token di masa depan ($t_{i+1}$), mereka secara hiperaktif berupaya memprediksi secara sekuens lintasan beberapa kata ke depan sekaligus ($t_2, t_3, t_4, t_5$ bersamaan) menggunakan mekanisme proyeksi linear beruntai dengan rumus:
    $$h_{i}^{'k} = M_k [\text{RMSNorm}(h_i^{k-1}); \text{RMSNorm}(\text{Emb}(t_{i+k}))]$$
    Hal ini menjadi pendorong utama dari fitur *Speculative Decoding* guna menghasilkan respons jawaban kepada pengguna dalam laju waktu yang spektakuler.

---
*(Bab Selesai - Mahasiswa diinstruksikan untuk memahami kalkulasi komputasi linier, dinamika Load-balancing MoE, serta implikasi sistem dari mekanisme seleksi diskrit).*