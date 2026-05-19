---
slug: bab-1-8
title: "Arsitektur Attention"
layout: chapter
---

## Segala Hal Tentang Arsitektur dan Hiperparameter yang (Mungkin) Belum Anda Ketahui

### 3.1. Pengantar: Bagaimana Kita Memahami Arsitektur?
Dalam mempelajari arsitektur *Large Language Model* (LLM), kita semua mungkin berharap hidup di dunia di mana kita hanya perlu mengetahui konsep teoritis yang sederhana seperti dimensi VC (Vapnik–Chervonenkis). Namun, pada kenyataannya, desain ruang arsitektur LLM modern sangatlah luas dan kompleks. 

Pendekatan terbaik untuk mempelajari arsitektur adalah dengan melatih model Anda sendiri dan mencoba berbagai variasi. Namun, karena keterbatasan komputasi dan waktu untuk mengeksplorasi seluruh ruang desain, pendekatan terbaik kedua adalah **belajar dari pengalaman orang lain**. Modul ini akan mensurvei berbagai model bahasa *dense* (padat/non-MoE) yang dirilis oleh berbagai institusi, mengamati kesamaan di antara mereka, dan memahami variasi arsitektur serta proses pelatihannya.

Hanya dalam kurun waktu 2024-2025 (bahkan sebelum menghitung model awal tahun 2025 secara penuh), terdapat lebih dari 19 perilisan model *dense* baru (seperti Falcon-2, Llama-3, Qwen-2, Gemma, Nemotron, dll.) dengan berbagai penyesuaian arsitektur minor. Jika kita meninjau lintasan sejarah pengembangan LLM, kita melihat sebuah tren:
1. **Era Eksperimentasi:** Dari rilisnya makalah asli Transformer hingga era GPT-3, banyak eksperimen dilakukan tanpa adanya satu standar emas yang disepakati.
2. **Era Konsensus LLaMA:** Ketika LLaMA-2 dirilis, arsitekturnya terbukti sangat baik sehingga hampir semua pihak mulai melatih model "keturunan LLaMA" dengan variasi yang sangat minor.
3. **Era Inovasi Baru (2024-2025):** Belakangan ini, mulai muncul kembali tren modifikasi arsitektur yang berfokus pada dua hal utama: **kestabilan pelatihan (stability)** dan **kapasitas konteks yang lebih panjang (long-context dependence)**.

Oleh karena itu, modul ini akan membahas blok-blok bangunan dasar arsitektur (seperti normalisasi, aktivasi, dan penyematan posisi), dilanjutkan dengan hiperparameter tingkat rendah (seperti dimensi, jumlah *head*, ukuran kosakata), dan diakhiri dengan trik-trik kestabilan dan efisiensi inferensi.

---

### 3.2. Penempatan Normalisasi: *Pre-Norm* vs *Post-Norm*

Satu hal yang hampir disepakati oleh seluruh praktisi *Machine Learning* pada tahun 2024 adalah letak penempatan *Layer Normalization* (LayerNorm). Pada arsitektur Transformer asli (Vaswani et al.), LayerNorm diletakkan di dalam jalur aliran residual (*residual stream*). Pendekatan ini dikenal sebagai **Post-Norm** atau *Residual Norm*.

Namun, hampir semua LLM modern memindahkan LayerNorm ke luar aliran residual, tepat sebelum komputasi utama dilakukan (sebelum *Multi-Head Attention* dan *Feed-Forward Network*). Pendekatan ini disebut **Pre-Norm**. 

> **Catatan Tambahan Anekdotal:** Sebagian besar model menggunakan Pre-Norm. Namun, ada satu pengecualian yang cukup lucu di masa lalu, yaitu model OPT-350M. Secara umum, seri model OPT memang memiliki banyak masalah selama pelatihannya, dan entah mengapa hanya versi 350M dari seri tersebut yang menggunakan Post-Norm di aliran residualnya.

#### Mengapa *Pre-Norm* Lebih Disukai?
Secara historis, motivasi awal memodifikasi penempatan normalisasi adalah untuk menghilangkan kebutuhan *warm-up* pada *learning rate*. Namun, praktisi dengan cepat menyadari bahwa manfaat sesungguhnya terletak pada **stabilitas dan kemampuan model untuk dilatih lebih dalam (lebih banyak *layer*)**.

Aturan praktis dalam mendesain arsitektur adalah: *"Jaga agar aliran residual Anda tetap bersih"*. 
Jika kita melihat diagram komputasi **Pre-Norm Transformer**:
1. Input $x_l$ masuk dari bawah dan menjalar lurus ke atas hingga menjadi output tanpa hambatan. Ini adalah **aliran residual utama**.
2. Sebelum masuk ke blok *Multi-Head Attention* atau *FFN*, $x_l$ diduplikasi, masuk ke *LayerNorm*, diproses, lalu ditambahkan kembali ke aliran utama.
3. Karena aliran utama tidak dimodifikasi oleh normalisasi secara langsung, rambatan gradien (*gradient propagation*) saat *backward pass* dapat mengalir lurus ke bawah tanpa atenuasi (pelemahan). 

Secara matematis, perbandingan persamaannya adalah:
*   **Post-LN Transformer:** $x_{l+1}^{post} = \text{LayerNorm}(x_l^{post} + \text{SubLayer}(x_l^{post}))$
*   **Pre-LN Transformer:** $x_{l+1}^{pre} = x_l^{pre} + \text{SubLayer}(\text{LayerNorm}(x_l^{pre}))$

**Bukti Empiris dari Visualisasi Data (Xiong 2020 & Salazar 2019):**
*   **Grafik Ekspektasi Gradien (Gradient Attenuation):** Pada grafik bar (Xiong 2020), sumbu X menunjukkan lapisan (*layer*) ke-1 hingga 6, dan sumbu Y adalah Ekspektasi Gradien. Pada saat inisialisasi, model Post-LN (warna oranye) menunjukkan gradien yang meledak di lapisan atas (lapisan 6 mendekati nilai 1.5) dan mengecil tajam di lapisan bawah (mendekati 0). Sebaliknya, model Pre-LN (warna biru) menjaga ekspektasi gradien yang stabil dan merata di semua lapisan pada angka 0.2. Ini membuktikan Pre-LN mencegah pelemahan gradien.
*   **Grafik Paku Gradien (Gradient Spikes):** Pada grafik garis (Salazar 2019), sumbu X adalah iterasi dan sumbu Y adalah norma global gradien (*Global norm* dalam skala logaritma). Kurva Post-LN (biru putus-putus) menunjukkan paku-paku gradien (lonjakan tajam) yang sangat ekstrem, mencapai skala $2.5$ hingga $3.0$. Sementara itu, Pre-LN (merah putus-putus) sangat stabil dan mendatar di bawah angka $0.0$.
*   **Kinerja Pelatihan:** Grafik *Validation Loss* menunjukkan Pre-LN (merah) turun lebih cepat dan stabil dibandingkan Post-LN.

#### Variasi Terbaru: *Double Norm* / *Non-Residual Post-Norm*
Mengingat aturan utamanya adalah "jangan letakkan LayerNorm di dalam aliran residual", maka meletakkan normalisasi *setelah* komputasi (namun tetap di luar aliran residual) secara logika sama baiknya dengan Pre-Norm. Beberapa model terbaru seperti Grok, Gemma 2, dan Olmo 2 menerapkan **Non-Residual Post-Norm** atau menempatkan dua normalisasi sekaligus (sebelum dan sesudah blok *Attention/FFN*, di luar panah aliran utama). Khusus untuk Olmo 2, mereka sepenuhnya murni menggunakan non-residual post-norm. Prinsip mendasarnya tetap sama: biarkan sinyal utama lewat tanpa gangguan.

---

### 3.3. Jenis Normalisasi: *LayerNorm* vs *RMSNorm*

Setelah mengetahui *di mana* meletakkan normalisasi, pertanyaan selanjutnya adalah *jenis* normalisasi apa yang digunakan. 

Pada Transformer orisinal, model menggunakan **LayerNorm**. LayerNorm menormalisasi rata-rata (*mean*) dan varians di seluruh dimensi model ($d_{model}$), ditambah dengan parameter yang dapat dipelajari (bias dan skala).
Persamaan LayerNorm:
$$y = \frac{x - \mathbb{E}[x]}{\sqrt{\text{Var}[x] + \epsilon}} * \gamma + \beta$$
Model terkemuka yang menggunakan ini: GPT-1/2/3, OPT, GPT-J, BLOOM.

Namun, mayoritas LLM modern beralih ke **RMSNorm** (*Root Mean Square Normalization*). RMSNorm menghilangkan penghitungan pengurangan rata-rata (*mean*) dan menghilangkan suku bias ($\beta$).
Persamaan RMSNorm:
$$y = \frac{x}{\sqrt{\frac{1}{d}\sum x_i^2 + \epsilon}} * \gamma = \frac{x}{\sqrt{||x||_2^2 + \epsilon}} * \gamma$$
Model terkemuka yang menggunakan ini: LLaMA-family, PaLM, Chinchilla, T5.

#### Mengapa RMSNorm Lebih Dipilih?
Penjelasan modernnya sederhana: RMSNorm lebih cepat dan memiliki performa yang sama baiknya. Ia memiliki operasi yang lebih sedikit (tanpa kalkulasi rata-rata) dan parameter yang lebih sedikit (tanpa parameter bias untuk disimpan).

Namun, ada pertanyaan kritis: *Apakah penjelasan ini masuk akal secara sistem?*
Berdasarkan profil komputasi perangkat keras (Ivanov et al. 2023), kontraksi tensor (perkalian matriks) mengambil **99.80% dari total FLOPs**, sedangkan normalisasi statistik hanya **0.17%** dan operasi *element-wise* hanya **0.03%**. Memangkas operasi mean dan bias tampaknya sangat tidak signifikan jika dilihat dari persentase FLOPs.

**Pelajaran Penting: FLOPs Bukanlah Waktu Proses (*Runtime*)!**
Hal yang krusial dalam perangkat keras modern adalah **pergerakan data (*data movement*)**. Diagram eksekusi sistem menunjukkan bahwa meskipun LayerNorm memiliki ukuran FLOPs yang kecil (misal 29M FLOPs dibanding *Attention* yang 43G FLOPs), nilai **Intensitas Aritmetika** (rasio FLOP-terhadap-memori) LayerNorm sangatlah buruk (hanya 3.5, dibandingkan *Attention* sebesar 153). 

Kita harus menjaga GPU tetap "panas" dengan operasi matematika yang intens (intensitas aritmetika tinggi). Kita tidak ingin membuang waktu GPU hanya untuk memindahkan sejumlah kecil data bolak-balik antara memori lambat dan memori cepat. Membuang perhitungan rata-rata dan bias pada normalisasi mengurangi pergerakan memori ini, sehingga meskipun pengurangan FLOPs-nya kecil (0.17%), pengurangan waktu proses (*runtime*) secara keseluruhan bisa mencapai **25%** untuk beban kerja tertentu. 

Validasi empiris dari Narang et al. 2020 membuktikan hal ini. Pada model Transformer 223M, varian Vanilla menghasilkan 3.50 *step/s*, sementara RMSNorm menghasilkan **3.68 *step/s***. Menariknya, selain mendapat keuntungan komputasi secara gratis, performanya (seperti skor BLEU dan penurunan *loss*) ternyata sedikit lebih baik.

#### Menghilangkan Suku Bias Secara Umum
Prinsip efisiensi pergerakan memori ini berlaku lebih luas. Sebagian besar implementasi LLM modern **menghapus suku bias (konstanta)** dari seluruh lapisan *Linear* di dalam model. 
*   **Asli:** $\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$
*   **Modern:** $\text{FFN}(x) = \sigma(xW_1)W_2$

Alasannya sama dengan RMSNorm: penghematan memori (meningkatkan intensitas aritmetika) dan menambah stabilitas optimisasi tanpa mengorbankan daya ekspresi model.

---

### 3.4. Fungsi Aktivasi: Era *Gated Linear Units* (GLU)

Model membutuhkan fungsi aktivasi non-linear. Terdapat "kebun binatang" fungsi aktivasi: ReLU, GeLU, Swish, ELU, GLU, GeGLU, ReGLU, SeLU, SwiGLU, LiGLU. Apa bedanya dan apakah hal ini penting?

*   **ReLU:** $\text{FFN}(x) = \max(0, xW_1)W_2$. Digunakan oleh Transformer orisinal, T5 awal, Chinchilla.
*   **GeLU (Gaussian Error Linear Unit):** $\text{FFN}(x) = \text{GELU}(xW_1)W_2$. Di mana $\text{GELU}(x) := x\Phi(x)$. Perbedaannya secara visual dengan ReLU hanyalah adanya sedikit kurva ("lembah" atau *divot* kecil) di bawah nol yang memodifikasi gradien di sekitar titik nol. Digunakan oleh seri GPT awal, GPT-J, BLOOM. Model performa tinggi dapat dilatih murni dengan ini.

Namun, inovasi yang saat ini mendominasi secara absolut adalah **Gated Activations (*GLU)**.
Aturan heuristik dalam desain arsitektur menyatakan bahwa *gating* (penggerbangan) biasanya sangat membantu. GLU memodifikasi bagian pertama dari layer *Feed-Forward*.

Bukannya langsung memasukkan hasil linier ke ReLU, kita menambahkan suku matriks linier kedua (secara *element-wise*) untuk "menggerbangi" output tersebut.
Transformasinya dari ReLU biasa menjadi ReGLU:
$$\max(0, xW_1) \longrightarrow \max(0, xW_1) \otimes (xV)$$
Persamaan lengkap FFN dengan **ReGLU**:
$$\text{FFN}_{\text{ReGLU}}(x) = (\max(0, xW_1) \otimes xV) W_2$$
Perhatikan bahwa kita sekarang memperkenalkan **parameter matriks ekstra ($V$)**.

Variasi GLU lainnya hanyalah mengganti fungsi aktivasi utamanya sebelum di-*gate*:
*   **GeGLU:** $\text{FFN}_{\text{GEGLU}}(x, W, V, W_2) = (\text{GELU}(xW) \otimes xV)W_2$ (Digunakan oleh T5 v1.1, Phi3, Gemma 2/3/4).
*   **SwiGLU:** $\text{FFN}_{\text{SwiGLU}}(x, W, V, W_2) = (\text{Swish}_\beta(xW) \otimes xV)W_2$, di mana $\text{Swish}(x) = x * \text{sigmoid}(x)$. (Digunakan oleh mayoritas LLM pasca 2023: seri LLaMA, Mistral, OLMo).

**Catatan Rasio Parameter (Faktor 2/3):**
Karena model ber-gerbang (Gated) memiliki tiga matriks penbobot ($W, V, W_2$) dibandingkan dengan FFN standar yang hanya memiliki dua ($W_1, W_2$), jumlah parameter akan membesar jika dimensinya sama. Untuk menyelaraskan jumlah parameter agar komparasinya adil dengan FFN biasa, dimensi *feed-forward* ($d_{ff}$) pada model ber-gerbang biasanya **dikecilkan sebesar faktor 2/3**.

Apakah unit linier ber-gerbang ini terbukti berhasil? Ya, secara sangat konsisten. Shazeer (2020) menguji ini dengan sangat baik (melakukan koreksi penyetaraan parameter 2/3) dan menemukan bahwa varian GLU, GeGLU, dan SwiGLU memberikan metrik akurasi, *perplexity*, dan konvergensi *loss* yang secara meyakinkan lebih baik dibandingkan ReLU atau GeLU biasa. Hampir mustahil menemukan arsitektur LLM modern saat ini yang tidak menggunakan varian GLU (kecuali beberapa *outlier* aneh seperti Nemotron 340B yang menggunakan *Squared ReLU*).

---

### 3.5. Arsitektur Komputasi: Lapisan Serial vs Paralel

Blok transformer normal bersifat **serial** – model menghitung *Attention*, melakukan residu dan norma, baru kemudian menghitung FFN (MLP), dilakukan secara berurutan.

Apakah kita bisa memparalelkan blok tersebut demi optimisasi perangkat keras?
Beberapa model (awalnya dipopulerkan oleh GPT-J, dan kemudian PaLM serta GPT-NeoX) mengusulkan **Lapisan Paralel (Parallel Layers)**:
$$y = x + \text{MLP}(\text{LayerNorm}(x)) + \text{Attention}(\text{LayerNorm}(x))$$
Dalam bentuk paralel, matriks perkalian input MLP dan *Attention* dapat digabungkan (*fused*), dan LayerNorm dapat digunakan bersama. Pada pengujian awal, ini diklaim memberikan kecepatan pelatihan sekitar 15% lebih cepat tanpa degradasi kualitas yang berarti pada model skala besar (misal PaLM 540B). Model Cohere Command juga awalnya memakai struktur ini.

Namun, tren ini **telah ditinggalkan** secara luas dalam 2 tahun terakhir. Mayoritas arsitektur kembali menggunakan lapisan **Serial**. Mengapa? Karena optimisasi sistem untuk bentuk serial (seperti *FlashAttention* dan kernel *fused* modern) telah menjadi sangat efisien, sehingga keuntungan sistem komputasi dari lapisan paralel tidak lagi sepadan dengan sedikit kehilangan kapasitas representasional (daya ekspresi) dari menghitung *Attention* dan FFN secara terpisah dan bertumpuk. Secara logika representasi, menjalankan paralel sama saja dengan kehilangan separuh "kedalaman" proses logika layer tersebut.

---

### 3.6. *Positional Embeddings*: Masuknya Era RoPE

*Attention* pada dasarnya kebal terhadap posisi (*positionally independent*); ia hanya melakukan produk titik (*inner product*). Kita harus menyuntikkan informasi urutan kata. 

Riwayat pendekatan penyematan posisi:
1.  **Sine Embeddings (Transformer Asli):** Menambahkan nilai trigonometri absolut. $\text{Embed}(x, i) = v_x + PE_{pos}$. 
2.  **Absolute Embeddings (GPT-1/2/3):** Menambahkan vektor parameter posisi mutlak yang dipelajari ke dalam vektor token. $\text{Embed}(x, i) = v_x + u_i$.
3.  **Relative Embeddings (T5, Chinchilla):** Tidak ditambahkan ke token, tetapi menambahkan nilai skalar bias ke dalam perhitungan matriks atensi itu sendiri. $e_{ij} = \frac{x_i W^Q (x_j W^K + a_{ij}^K)^T}{\sqrt{d_z}}$.

Pendekatan absolut dianggap kurang optimal karena model seharusnya tidak peduli apakah kata "sebuah" dan "apel" berada di awal atau di akhir kalimat, melainkan hanya peduli pada **jarak relatif** di antara keduanya. Relative Embedding menyelesaikan ini, namun ia bukan sebuah metrik "produk titik" matematis murni.

Solusi modern (dipakai oleh hampir 99% LLM pasca 2024 seperti LLaMA, PaLM, dll.) adalah **Rotary Position Embeddings (RoPE)**.

#### Konsep Matematika dan Logika RoPE
Tujuan RoPE adalah merumuskan fungsi penyematan $f(x, i)$ (di mana $x$ adalah fitur dan $i$ adalah posisi absolut), sedemikian rupa sehingga produk titik antara dua token hanya bergantung pada jarak relatifnya ($i - j$):
$$\langle f(x, i), f(y, j) \rangle = g(x, y, i - j)$$

Pendekatan absolut dan sinus gagal mencapai identitas ini secara murni karena memunculkan suku silang (*cross-terms*) yang mengandung informasi absolut.

**Solusi RoPE:** Kita ingin penyematan produk titik menjadi invarian terhadap pergeseran posisi. Secara geometris, kita tahu bahwa operasi produk titik invarian (nilainya tetap) terhadap operasi **rotasi** spasial yang sama.
Berdasarkan visualisasi konsep geometri:
1. Bayangkan kita punya dua vektor kata independen dari posisi: "we" dan "know".
2. Misalkan kata "we" berada di posisi indeks 0 dan "know" di posisi 1. Kita memutar vektor "we" sebesar rotasi 0 derajat, dan vektor "know" berputar sebesar jarak 1 derajat posisi. Produk titik (sudut di antara keduanya) adalah 1 unit.
3. Sekarang, jika kalimatnya digeser menjadi "Of course we know". Kata "we" kini di posisi 2, putar vektornya 2 derajat. Kata "know" di posisi 3, putar vektornya 3 derajat. 
4. Sudut relatif di antara "we" dan "know" tetaplah $3 - 2 = 1$ unit rotasi! Nilai produk titiknya persis sama tanpa model harus mengetahui posisi absolutnya.

Karena model berdimensi tinggi ($D$-dimensi), bukan 2 dimensi, strategi RoPE adalah membagi vektor berdimensi $D$ menjadi pasangan-pasangan 2 dimensi ($d/2$ pasang). Setiap pasang sumbu 2D akan dirotasi dengan frekuensi sudut $\theta$ yang berbeda-beda.
Matriks Rotasi RoPE $\boldsymbol{R}_{\Theta, m}^d$ memuat elemen sinus dan kosinus pada diagonal utamanya:
$$
\boldsymbol{R}_{\Theta, m}^d = 
\begin{pmatrix}
\cos m\theta_1 & -\sin m\theta_1 & 0 & 0 & \cdots & 0 & 0 \\
\sin m\theta_1 & \cos m\theta_1 & 0 & 0 & \cdots & 0 & 0 \\
0 & 0 & \cos m\theta_2 & -\sin m\theta_2 & \cdots & 0 & 0 \\
0 & 0 & \sin m\theta_2 & \cos m\theta_2 & \cdots & 0 & 0 \\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
0 & 0 & 0 & 0 & \cdots & \cos m\theta_{d/2} & -\sin m\theta_{d/2} \\
0 & 0 & 0 & 0 & \cdots & \sin m\theta_{d/2} & \cos m\theta_{d/2} 
\end{pmatrix}
$$
Di mana $m$ adalah indeks posisi token. RoPE **dikalikan** secara perkalian matriks *sparse* ke vektor Query ($Q$) dan Key ($K$), BUKAN ditambahkan. Ini memastikan tidak ada *cross-terms* absolut. Penerapannya harus dilakukan pada *setiap operasi atensi* untuk terus menegakkan invariansi posisi tersebut.

---

### 3.7. Analisis Hiperparameter Konsensus (dan Pengecualiannya)

Sebagai praktisi, ketika mulai merancang jaringan, Anda akan dihadapkan pada nilai-nilai parameter numerik spesifik. Beberapa hiperparameter ini ternyata telah mencapai titik "konsensus" industri.

#### 1. Rasio Dimensi *Feed-Forward* terhadap Dimensi Model ($d_{ff} / d_{model}$)
Secara tradisional, dimensi jaringan *feed-forward* diperbesar dibandingkan aliran residual utama. Seberapa besar?
*   **Aturan Jempol Klasik:** $d_{ff} = 4 \times d_{model}$.
*   **Untuk Varian GLU:** Karena aturan penyamaan parameter 2/3, rasionya dikalikan $2/3$, menjadi $\approx 2.66 \times d_{model}$ (contoh: Qwen 14B = 2.67, DeepSeek 67B = 2.68). LLaMA secara sewenang-wenang memilih menggeser ini sedikit menjadi rasio **3.5**.
*   **Pengecualian T5 (Eksperimen Radikal Google):** Model T5 (11B) menetapkan $d_{model} = 1024$ dan $d_{ff} = 65,536$. Ini adalah pengali yang luar biasa ekstrem sebesar **64 kali lipat**. Argumen Google bersifat *systems-oriented*: perkalian matriks raksasa jauh lebih memaksimalkan pemanfaatan perangkat keras (TPU) mereka. Namun lucunya, pada rilis T5 v1.1, Google kembali ke rasio konservatif 2.5. Ini membuktikan pengali ekstrem mungkin efisien secara sistem tapi suboptimal secara kualitas arsitektur.
*   **Zona Optimal (Kaplan et al. 2020):** Grafik uji coba *Scaling Laws* menunjukkan kurva berbentuk lembah (U-shape) pada rasio Feed-Forward (sumbu X) terhadap kenaikan *loss* (sumbu Y). Terdapat area cekungan stabil dari rasio 1 hingga 10, di mana kerugian (*loss*) sangat minim (mendekati 0%). Jika rasio dinaikkan di atas 10 hingga 100, kerugian (*loss*) model akan meningkat secara eksponensial. Jadi rasio standar 2.6 hingga 4 sangat berada di zona aman.

#### 2. Jumlah *Head* dan Dimensi *Head*
Aturan umumnya adalah membagi secara merata: $\text{Head\_Dim} \times \text{Num\_Heads} = \text{Model\_Dim}$. 
Misal GPT-3: 96 heads $\times$ 128 head dim = 12288 model dim (Rasio 1). Hampir semua model mengikuti rasio bernilai 1 ini secara ketat (kecuali variasi PaLM dan T5 yang rasionya sedikit berbeda). Parameter ini sangat memaafkan (*forgiving*), asalkan seimbang.

#### 3. Rasio Aspek (Kedalaman / *Depth* vs Lebar / *Width*)
Apakah model sebaiknya memiliki banyak lapisan (dalam) atau dimensi vektor yang besar (lebar)? Sebagian besar model menunjukkan konsistensi yang sangat mengejutkan.
Tabel Rasio $d_{model} / n_{layer}$:
*   BLOOM: 205
*   PaLM (540B): 156
*   GPT-3 / OPT / Mistral / Qwen: **128**
*   LLaMA / LLaMA 2: 102
*   Gemma 3: 87

Sebagian besar model mencari "Titik Manis" (Sweet spot) di **rentang 100 hingga 200**. Alasan utamanya didikte oleh batasan sistem (*systems concerns*).
Berdasarkan visualisasi keterbatasan oleh Tay et al. 2021: Sebuah model berbaris yang terdistribusi ke beberapa GPU (GPU 0, 1, 2, 3) menunjukkan alur *forward* dan *backward* pass. Model yang *terlalu dalam* (lapisan yang tak terhingga) akan memaksa kita menggunakan **Pipeline Parallelism**. Pemrosesan paralel lapisan berantai menuntut GPU berikutnya menunggu GPU sebelumnya selesai, menciptakan *latency* dan hambatan besar (*bubbles*). Sebaliknya, memperlebar dimensi model memungkinkan **Tensor Parallelism**, di mana perhitungan perkalian matriks tunggal dapat dengan mudah disebar dan dihitung bersamaan oleh puluhan ribu GPU secara harmonis. Oleh karena itu, kita mempertahankan model agar tidak kelewat dalam.

#### 4. Ukuran Kosakata (*Vocabulary Size*)
*   **Model Monolingual (Terutama Inggris):** Cukup sekitar 30k - 50k token (Misal: LLaMA asli 32,000, GPT-2 50,257).
*   **Model Multilingual / Sistem Produksi Raksasa:** Memerlukan rentang 100k - 250k token (Misal: PaLM 256,000, GPT-4 100,276, Gemma 4 262,144). Karena varian bahasa dan karakter yang ditangani lebih kaya. Semakin besar ukuran model, semakin besar kosakata yang mampu didukungnya secara kapasitas parametrik.

#### 5. Regularisasi (Dropout & Weight Decay)
Dalam kelas *Machine Learning 101*, kita diajarkan menggunakan regularisasi untuk mencegah *overfitting* (menghafal data pelatihan). Namun, saat melatih LLM, kita melatih triliunan token menggunakan optimisasi SGD jalan tunggal (*single-pass*). *Overfitting* secara teoritis hampir mustahil terjadi karena model tidak pernah melihat data yang sama dua kali. 
Lalu, apakah kita butuh regularisasi?
*   **Dropout:** Banyak model lama menggunakan Dropout 0.1, tetapi model baru (seperti T5 v1.1, PaLM, LLaMA) menghapusnya sama sekali (0.0). Dropout tidak bersahabat dengan optimisasi sistem dan dianggap usang.
*   **Weight Decay (Penyusutan Bobot):** Mengejutkannya, banyak model modern (seperti LLaMA, GPT-3, Qwen) masih mempertahankan angka *weight decay* sebesar 0.1. Mengapa?
Mengacu pada makalah Andriushchenko et al. 2023, fenomena ini dianalisis secara visual:
    *   *Grafik Validation vs Training Loss:* Tidak ada penyimpangan dari garis diagonal (x = y), membuktikan model sama sekali tidak sedang mengatasi masalah *overfitting*.
    *   *Grafik Training Loss vs Iterations:* Kurva ini mengamati nilai weight decay ($\lambda_{WD}$) disandingkan dengan penurunan skala *learning rate* (*cosine LR decay* vs *Constant LR*). Ditemukan bahwa penerapan *weight decay* kuat (kurva warna biru dan hijau putus-putus pada bagian bawah grafik) berinteraksi sangat baik secara sinergis dengan *learning rate scheduler*. *Weight decay* perlahan mengecilkan bobot, yang memungkinkan *learning rate* lebih tinggi, sehingga optimisasi mencapai titik konvergensi minimum yang jauh lebih baik.
Kesimpulannya: *Weight decay* pada LLM BUKAN sebagai alat pengontrol *overfitting*, melainkan sebagai **intervensi dinamika optimisasi**.

---

### 3.8. Trik Stabilitas: Mengendalikan *Softmax* yang Ganas

Salah satu alasan terbesar memodifikasi arsitektur saat ini adalah **kestabilan pelatihan**. Praktisi sangat takut mendapati kurva kerugian pelatihan yang awalnya mulus tiba-tiba "meledak" dengan paku-paku vertikal tak terhingga yang merusak seluruh bobot (divisualisasikan dengan kurva biru *Loss* dan *Gradient L2 Norm* yang melompat-lompat berantakan secara tiba-tiba). Jika Anda telah membakar jutaan dolar waktu GPU dan model hancur di tengah jalan tanpa bisa dipulihkan, itu adalah bencana mutlak.

Akar masalah ketidakstabilan hampir selalu berasal dari operasi **Softmax**, yang melibatkan eksponensial (rawan meledak menjadi nilai *infinity*) dan pembagian (rawan meledak jika pembaginya mendekati nol). Karena arsitektur memiliki dua letak Softmax (di Output/Probabilitas, dan di perhitungan Attention), dua strategi besar digunakan:

#### 1. Stabilitas Output Softmax: *Z-Loss*
Dalam penghitungan probabilitas silang log (*log prob*), rumus Softmax dijabarkan sebagai:
$$ \log(P(x)) = \log\left(\frac{e^{U_r(x)}}{Z(x)}\right) = U_r(x) - \log(Z(x)) $$
Di mana $Z(x) = \sum_{r'} e^{U_{r'}(x)}$ adalah normalisator (pembagi).
Jika $Z(x)$ menjadi sangat besar, perhitungannya tidak stabil. Karena Softmax memiliki sifat over-parameterisasi (menambahkan konstanta tidak mengubah probabilitas akhir), kita dapat "memaksa" nilai $Z(x)$ agar mendekati nilai 1 (yang berarti $\log(Z)$ mendekati 0) dengan menambahkan pinalti regularisasi artifisial, yang dikenal sebagai trik **Z-loss**.
$$ L_{total} = \sum_i \left[ \log(P(x_i)) - \alpha(\log(Z(x_i)))^2 \right] $$
Digunakan awalnya oleh Devlin (2014) dan diterapkan di PaLM, Baichuan 2, DCLM, dan OLMo.

#### 2. Stabilitas Atensi Softmax: *QK-Norm*
Untuk mencegah masukan logits ke Softmax Atensi membesar tak berhingga, triknya sangat sederhana namun brutal: **Lemparkan LayerNorm ke sana!**
Model yang menggunakan *QK-Norm* akan melakukan normalisasi Layer/RMS langsung pada Query dan Key *sesaat sebelum* operasi produk titik (perkalian matriks), memastikan nilainya berskala konstan di kisaran angka 1. (Diterapkan oleh DCLM, OLMo, Gemma, Qwen3).

#### 3. Capping Eksplisit (Logit Soft-Capping)
Ini adalah trik yang lebih keras, banyak digunakan oleh Google (seri Gemma). 
Bukannya berharap QK-Norm menyeimbangkan nilai, pendekatan ini membatasi nilai maksimum logits secara mutlak menggunakan kurva *Hyperbolic Tangent* ($\tanh$).
$$\text{logits} \leftarrow \text{soft\_cap} * \tanh(\text{logits} / \text{soft\_cap})$$
Batas *soft_cap* biasanya diset pada 50.0 untuk atensi dan 30.0 untuk layer final. Meskipun teknik ini memastikan logits tidak akan pernah "meledak", tabel komparasi Nvidia menunjukkan hal ini bisa sedikit mendegradasi performa murni model dibandingkan dengan QK-Norm biasa. Ini adalah pertukaran antara kestabilan mutlak versus sedikit performa.

---

### 3.9. Varian *Attention Head*: Efisiensi Inferensi

Setelah pelatihan selesai, model harus diterapkan ke pengguna (*serving*). Di sinilah masalah baru muncul: kelambatan akibat pembacaan memori yang intens pada saat generasi teks (satu per satu kata/token). 

**Masalah KV Cache dan Intensitas Aritmetika Inferensi:**
Selama pra-pelatihan, komputasi dihitung seluruhnya secara matriks bersamaan (Intensitas Aritmetika tinggi). Namun saat generasi inferensi (Autoregresif), setiap langkah baru memaksa model membaca semua Query dan Key dari kata-kata sebelumnya yang tersimpan di dalam **KV Cache**.
Secara teoritis:
* Total operasi matematika tetap sama.
* Total **Akses Memori** membengkak menjadi proporsional terhadap: Panjang Sekuens ($n$) $\times$ Dimensi Model Kuadrat ($d^2$).
Intensitas Aritmetikanya anjlok karena rumus $O\left( (\frac{n}{d} + \frac{1}{b})^{-1} \right)$. Semakin panjang sekuens $n$, batas memorinya semakin menjadi leher botol (bottleneck). Kecepatan GPU menjadi tertahan menunggu antrian transfer data lambat.

#### Solusi 1: Multi-Query Attention (MQA)
MQA memotong drastis lalu lintas data ini dengan cara memelihara *banyak Query*, tetapi membagikan **satu Key dan satu Value tunggal** (dimensi $1$) untuk seluruh *heads* yang ada. KV Cache menyusut secara dramatis. Masalahnya: pengorbanan daya ekspresi terlalu tinggi, menyebabkan penurunan metrik performa model (seperti sedikit kejatuhan pada nilai *Perplexity* dalam tabel Shazeer 2019).

#### Solusi 2: Grouped-Query Attention (GQA)
Ini adalah "Jalan Tengah" yang brilian dan kini digunakan secara hampir universal. Daripada memangkasnya menjadi satu Key/Value tunggal, kita mengelompokkan beberapa Queries ke dalam sejumlah kecil *group* Keys/Values (misalnya 1 Key untuk setiap 4 Query). 
Grafik dari Ainslie 2023 menunjukkan superioritas metode ini:
*   **Grafik Atas (Kinerja vs Waktu):** Sumbu X adalah Waktu per sampel inferensi, sumbu Y adalah Kinerja (Akurasi). MQA berada di pojok kiri bawah (Sangat cepat tapi performa rendah). MHA (*Multi-Head* asli) berada di kanan atas (Performa tinggi tapi sangat lambat). GQA berada berdampingan dengan MHA secara vertikal (Performa sama tingginya) namun posisinya tergeser ke kiri selevel dengan MQA (Sangat Cepat!).
*   **Grafik Bawah (Waktu vs Grup GQA):** Menunjukkan bahwa perpindahan dari grup utuh (MHA/64) mundur sedikit ke GQA/16 langsung memangkas waktu proses (*Time per sample*) secara vertikal tajam ke dasar. 

GQA adalah kenop (*knob*) sederhana pengontrol rasio efisiensi tanpa menghancurkan kecerdasan model.

---

### 3.10. Atensi Jendela Geser (*Sliding Window / Sparse Attention*)

Menghitung atensi di mana seluruh token di dalam sekuens melihat token lain secara keseluruhan adalah operasi kuadratik ($O(N^2)$). Memasukkan ratusan ribu token dokumen panjang akan menghabiskan memori komputasi secara seketika. 

Untuk menangani ketergantungan konteks super panjang (*long-context*), standar industri terbaru adalah memadukan Atensi Penuh (Global) dengan Atensi Jendela Geser (Lokal/Sparse).
Visualisasi dari Child et al. 2019 menunjukkan matriks atensi di mana pola *Sliding Window* hanya mengizinkan model melihat beberapa langkah diagonal ke belakang, sementara pola *strided/fixed* di selang-selingkan untuk jangkauan global.

**Implementasi Modern (Cohere Command A, LLaMA 4, Gemma 3/4):**
Pendekatannya adalah melakukan *interleave* (berselang-seling). Misalnya:
*   Transformer Block 1, 2, 3: Menggunakan **Sliding Window Attention (SWA)**. Mereka memproses informasi jarak pendek. Di sini kita menggunakan *RoPE Positional Embedding* untuk menegaskan struktur gramatikal jarak dekat.
*   Transformer Block 4: Menggunakan **Full Attention**. Menggabungkan pemahaman global dari blok lokal di bawahnya. Beberapa arsitektur (seperti Command A) secara sengaja membuang RoPE (*No Positional Embeddings* / NoPE) pada lapisan Full Attention ini agar murni berfokus pada pengambilan informasi semantik tas-kata (*bag-of-words*) melintasi jarak sangat jauh di dalam dokumen.
*   *Catatan Khusus:* Qwen 3.5 memiliki pendekatan serupa, tetapi lapisan jarak pendek (murahnya) diganti dengan arsitektur eksperimental eksotis *State-Space Model* (SSM) seperti Gated DeltaNet, bukan SWA biasa. 

Pendekatan hibrida (SWA + Full) sejauh ini adalah formula pemenang untuk mencapai model berbasis konteks-panjang (*long-context*) yang efisien namun cerdas.

---

### Rangkuman Penutup
Walaupun di luaran sana seolah-olah terjadi kompetisi ratusan model LLM yang berujung pada kebingungan pemilihan desain, kenyataannya pada level fundamen, **industri telah menemui pola arsitektur konsensus:**
1. Anda wajib memindahkan normalisasi ke pra-komputasi (*Pre-Norm*), atau di luar aliran residual.
2. Gunakan *RMSNorm* dan hapus fitur "Bias" dari Linear layers untuk kepuasan aritmetika sistem perangkat keras.
3. Gunakan aktivasi berbentuk gerbang: *SwiGLU* atau *GeGLU* dengan penyesuaian dimensi 2/3.
4. Gunakan penyematan posisi rotasional *RoPE* murni untuk atensi relasional.
5. Pertahankan hiperparameter di dalam rasio wajar (*FF ratio* 2.6 - 4, *aspect ratio* kedalaman:lebar pada ~100).
6. Lindungi diri Anda dari kegagalan latih yang meledak dengan intervensi *Softmax* seperti *Z-loss* atau *QK-Norm*.
7. Hemat inferensi mahal dengan menerapkan *GQA*, dan campurkan dengan *Sliding Window Attention* untuk melahap ratusan halaman teks panjang. 

Memahami dasar di balik perubahan-perubahan kecil namun berdampak masif ini akan menuntun Anda menjauhi sekadar "menebak parameter" menjadi perancang arsitektur model *Machine Learning* tingkat lanjut yang peka terhadap keseimbangan antara teori kecerdasan matematis dan realitas sistem komputasi perangkat keras.