---
slug: bab-1-9
title: "Attention & MoE"
layout: chapter
---

## Bab 4: Alternatif Atensi dan *Mixture of Experts* (MoE)

### 4.1. Pengantar: Ide-Ide Arsitektur Tingkat Lanjut
Pada bab sebelumnya, kita telah membahas arsitektur dasar Transformer dan bagaimana kita dapat memodifikasi berbagai komponen utamanya (seperti normalisasi, aktivasi, dan *positional embeddings*) untuk menghasilkan *Large Language Model* (LLM) modern. Bab ini akan membawa kita pada konsep-konsep yang jauh lebih mendalam dan kompleks yang dibangun di atas fondasi Transformer standar.

Terdapat dua fokus utama dalam bab ini:
1.  **Alternatif Atensi (*Attention Alternatives*):** Bagaimana cara merancang arsitektur yang mampu menangani konteks yang jauh lebih panjang. Kita akan mengeksplorasi perubahan arsitektur yang memungkinkan ketergantungan waktu yang linier (*linear time dependence*) terhadap panjang sekuens, alih-alih ketergantungan kuadratik yang membebani memori.
2.  **Mixture of Experts (MoE):** Memodifikasi bagian *Multi-Layer Perceptron* (MLP) atau *Feed-Forward Network* (FFN) pada Transformer. MoE memberikan pemanfaatan perangkat keras (*hardware utilization*) yang jauh lebih baik dengan meningkatkan jumlah parameter model secara signifikan tanpa harus mengorbankan biaya komputasi (FLOPs) pada setiap proses komputasinya.

---

### 4.2. Alternatif Atensi untuk Konteks Panjang

Sudah menjadi rahasia umum di industri saat ini bahwa pengguna menginginkan ukuran jendela konteks (*context window*) yang semakin panjang. Entah itu untuk memasukkan puluhan dokumen, buku, atau membuat agen AI yang beroperasi dalam waktu lama, model dituntut untuk memiliki lebih banyak pengetahuan di dalam konteksnya.

**Deskripsi Visual (Evolusi Ukuran Jendela Konteks 2018-2025):** 
Pada grafik sebar (*scatter plot*) di slide, sumbu X mewakili tanggal rilis model, sedangkan sumbu Y mewakili ukuran jendela konteks dalam skala logaritmik (dari 1.000 hingga 10.000.000 token). Terlihat jelas adanya "perlombaan" dari berbagai vendor LLM terkemuka (Anthropic, Google, Meta, OpenAI). Model awal seperti GPT-1 dan GPT-2 berada di kisaran 1.000 token. Namun memasuki tahun 2024-2025, model seperti Gemini 1.5 Pro dan Llama 4 telah menembus batas 1 juta hingga 10 juta token.

#### 4.2.1. Biaya Atensi dan Batasan Peralatan Dasar
Pertanyaannya: Bagaimana kita mengontrol biaya komputasi yang membengkak akibat konteks yang sangat panjang ini?

**Deskripsi Visual (Rasio Biaya Atensi vs Feed-Forward):**
Sebuah grafik area bertumpuk (*stacked area chart*) menunjukkan panjang sekuens pada sumbu X (0 hingga 15.000) dan metrik waktu/biaya komputasi (ms) pada sumbu Y. Area berwarna biru mewakili biaya komponen *Feed-Forward*, yang tumbuh secara linier seiring panjang sekuens. Namun, area berwarna oranye mewakili biaya *Attention*. Karena atensi mengharuskan setiap token berinteraksi dengan setiap token lainnya (koneksi *all-to-all*), kurvanya tumbuh secara **kuadratik**. Pada sekuens pendek, *feed-forward* mendominasi biaya. Namun seiring panjangnya sekuens, biaya atensi meledak dan menjadi masalah utama.

**Peralatan Dasar (*Basic Toolkit*):**
1.  **Menggabungkan Atensi Lokal dan Global:** Seperti yang dibahas pada bab sebelumnya, kita bisa menekan biaya dengan menggunakan *Sliding Window Attention* (atensi pita diagonal terbatas) pada sebagian besar lapisan, dan hanya melakukan atensi global penuh pada lapisan tertentu.
2.  **Rekayasa Sistem (*FlashAttention*):** Jangan pernah meremehkan faktor konstanta. Pendekatan seperti *FlashAttention* tidak mengubah notasi Big-O (tetap kuadratik), namun menata ulang operasi atensi sedemikian rupa untuk meminimalkan transfer memori (SRAM ke HBM). 
    *Deskripsi Visual (Kecepatan Forward + Backward A100):* Grafik batang menunjukkan bahwa *FlashAttention-2* (warna ungu) memberikan kecepatan teraflops per detik (TFLOPs/s) yang dua hingga tiga kali lipat lebih tinggi dibandingkan PyTorch dasar (biru) di berbagai panjang sekuens. Bahkan ketika PyTorch dasar kehabisan memori (*Out Of Memory* / OOM) pada 16k token, *FlashAttention* tetap berjalan dengan kecepatan tinggi.

Meskipun trik di atas luar biasa, jika kita ingin memproses 5 hingga 10 juta token, trik ini tidaklah cukup. Kita membutuhkan perolehan efisiensi yang lebih radikal, yakni **Atensi Waktu Linier (*Linear Time Attention*)**. 

#### 4.2.2. Atensi Linear dan Sifat Asosiatif
Untuk memahami metode linier yang saat ini terbukti berhasil dalam skala besar, kita hanya perlu memahami satu ide dasar matematika: **Sifat Asosiatif Perkalian Matrix**.

Mari kita tinjau operasi atensi standar. Misalkan $Q \in \mathbb{R}^{n \times d_k}$, $K \in \mathbb{R}^{n \times d_k}$, dan $V \in \mathbb{R}^{n \times d_v}$ (di mana $n$ adalah panjang sekuens):
$$ \text{Attn}(Q, K, V) = \rho(QK^\top)V $$
*Catatan: $\rho$ mewakili operasi Softmax yang menormalisasi baris matriks.*

Operasi ini bersifat kuadratik terhadap $n$ karena kita menghitung matriks $(QK^\top)$ yang berukuran $n \times n$, sehingga biayanya sebanding dengan $n^2d_k$.

Namun, **bagaimana jika kita mengabaikan operasi Softmax ($\rho$) untuk sejenak?**
Jika $\rho$ dianggap sebagai identitas (dihilangkan), maka berdasarkan sifat asosiatif perkalian matriks, kita bisa memindahkan tanda kurungnya:
$$ (QK^\top)V = Q(K^\top V) $$

Langkah ini tampak sangat sederhana (bahkan terkesan naif), namun berdampak krusial!
*   **Sebelumnya:** Kita mengalikan $Q$ dan $K^\top$ terlebih dahulu (biaya $n^2 d_k$), menghasilkan matriks $(n \times n)$, lalu dikali $V$ (biaya $n^2 d_v$).
*   **Sekarang:** Kita mengalikan $K^\top$ dan $V$ terlebih dahulu. Matriks $K^\top$ berukuran $(d_k \times n)$ dikali matriks $V$ $(n \times d_v)$, menghasilkan matriks padat berukuran $(d_k \times d_v)$. Komputasinya hanya bergantung pada ukuran $n \cdot d_v \cdot d_k$.

Karena dimensi *head* ($d_k, d_v$) biasanya kecil (ratusan) sedangkan $n$ (panjang konteks) bisa mencapai jutaan, kita berhasil mengubah kompleksitas waktu model dari ketergantungan **kuadratik ($n^2$) menjadi linier ($n$)**.

#### 4.2.3. Bentuk Rekuren dari Atensi Linear (Dualitas)
Keunggulan Atensi Linear tidak berhenti di sana. Operasi perkalian di sisi kanan ($K^\top V$) sebenarnya persis sama dengan arsitektur *Recurrent Neural Network* (RNN).
Bukannya menghitung seluruh matriks sekaligus (bentuk *dense*), kita bisa menghitungnya secara inkremental seiring kita bergerak dari kiri ke kanan melintasi konteks:
$$ S_t = S_{t-1} + k_t v_t^\top \quad \text{dan} \quad y_t = q_t^\top S_t $$
*   $S_t$ adalah *State* (status/keadaan) berukuran matriks tetap yang terus diperbarui (diakumulasikan).
*   $k_t v_t^\top$ adalah matriks *outer product* dari token saat ini.

**Dualitas yang Ajaib:** Atensi Linear memiliki sifat dualitas yang sempurna.
*   **Saat Pelatihan (*Training*):** Anda bisa menggunakan bentuk perkalian matriks paralel $(Q(K^\top V))$ yang sangat cepat dieksekusi oleh GPU.
*   **Saat Inferensi (*Inference*):** Anda bisa menggunakan bentuk rekuren (RNN). Anda tidak perlu memelihara KV-Cache raksasa; cukup simpan matriks *state* tetap $S_t$ dan perbarui di setiap langkah. Inferensi menjadi sangat efisien dan penggunaan memori tetap konstan (O(1) terhadap panjang teks yang dihasilkan).

> **Catatan Tambahan Anekdotal:** Apakah Atensi Linear murni (tanpa Softmax) bekerja dengan baik? Ya! Model **Minimax M1** (model *open-source* kuat dari Tiongkok) menggunakan rasio hibrida 7-banding-1 (7 lapisan Atensi Linear diselingi 1 lapisan Atensi Softmax penuh). Kinerjanya terbukti sebanding dengan model top seperti OpenAI o3 atau DeepSeek-R1, sekaligus menunjukkan penskalaan komputasi yang hampir murni linier terhadap panjang konteks.

#### 4.2.4. Dari Atensi Linear menuju Mamba-2
Tentu saja, menyimpan dan mengakumulasikan *state* terus-menerus secara statis agak kaku. Terinspirasi dari LSTM klasik, kita tahu bahwa model membutuhkan kemampuan untuk **memilih kapan harus meneruskan informasi dan kapan harus melupakannya**.

Peneliti (Gu & Dao) mengembangkan keluarga model *State Space* yang dikenal luas sebagai **Mamba** (terutama Mamba-2). Jika kita melihat mekanisme dasarnya, Mamba-2 sebenarnya adalah bentuk perluasan (*elaborasi*) dari Atensi Linear yang disisipi fungsi *gating* (penggerbangan).
Persamaan Mamba-2:
$$ S_t = \color{red}{\gamma_t} S_{t-1} + k_t v_t^\top \quad \text{dan} \quad y_t = q_t^\top S_t + \color{red}{v_t^\top D} $$
*dengan* $\color{red}{\gamma_t = f(x_t)}$

*   **Gerbang $\gamma_t$:** Sebuah gerbang yang mengatur seberapa banyak *state* masa lalu ($S_{t-1}$) yang diteruskan ke masa depan. Penting: gerbang ini *tidak bergantung pada state sebelumnya* (non-stateful), melainkan hanya bergantung pada input saat ini ($x_t$). Hal ini mempertahankan keunggulan perhitungan secara paralel.
*   **Koneksi Residual $D$:** Parameter $D$ mengatur agar keluaran saat ini tidak hanya berdasarkan *state* historis, tetapi juga secara langsung merepresentasikan input nilai (*value*) token saat ini.

Model seperti **Nemotron-3** dari Nvidia memanfaatkan hibrida Mamba-2 dan Atensi Softmax (dengan rasio sekitar 3:1). Evaluasi menunjukkan bahwa hibrida ini mencapai akurasi setara model tradisional namun memberikan *throughput* inferensi yang jauh lebih tinggi.

#### 4.2.5. *Gated Delta Net* (GDN) dan Mekanisme Kelupaan
Mengapa berhenti pada satu gerbang? Kita bisa memperumit bentuk rekuens ini dengan menambahkan gerbang-gerbang baru yang selektif. Inilah yang mendasari arsitektur **Gated Delta Net (GDN)**.

Persamaan Gated Delta Net:
$$ S_t = \gamma_t (\color{blue}{I - \beta_t k_t k_t^\top}) S_{t-1} + \color{blue}{\beta_t} k_t v_t^\top $$
$$ y_t = q_t^\top S_t \quad \text{dengan} \quad \gamma_t = f(x_t), \color{blue}{\beta_t = f(x_t)} $$

*   **Gerbang $\beta_t$ (*No-op Gate*):** Bertindak sangat mirip dengan *input gate* pada LSTM. Jika $\beta_t = 0$, model sama sekali tidak memasukkan informasi dari kata/token saat ini.
*   **Proyektor Penghapus $(I - \beta_t k_t k_t^\top)$:** Mekanisme ini tidak sekadar mengurangi nilai *state* lama, melainkan secara aktif **menghapus (memproyeksikan keluar)** arah informasi apapun dari matriks *state* historis yang searah dengan *key* ($k_t$) saat ini. Intuisi: sebelum kita menulis informasi baru ke memori, hapus dulu data usang yang berasosiasi dengan kunci pencarian yang sama.

> **Catatan Tambahan Anekdotal:** Menariknya, rumus pembaruan ini (proyektor identitas minus *outer product*) ditemukan secara independen di berbagai sub-bidang AI lainnya, seperti pemrograman bobot cepat (*fast weight programming*) dan *test-time training* dari optimisasi *least squares*. Hal ini membuktikan keampuhan matematis dari pendekatan ini.

Banyak model *open-source* teratas saat ini, seperti **Qwen 3.5** dan **Qwen Next**, menggunakan hibrida *Gated Delta Net* dan Atensi (rasio 3:1), yang secara masif meningkatkan *throughput* pelacakan dekode pada konteks yang sangat panjang (hingga 128K token) tanpa penurunan akurasi.

#### 4.2.6. Analisis Kinerja Arsitektur Hibrida
Jika metode ini begitu efisien, mengapa tidak menggunakan 100% arsitektur rekuren/linier? Jawabannya adalah pengorbanan daya ekspresi (*expressive power*). Hubungan *all-to-all* dari Atensi Softmax sangatlah superior secara pemahaman komprehensif.

**Deskripsi Visual (Analisis Hibrida ByteDance & UC Santa Cruz):** 
Grafik garis menunjukkan kinerja evaluasi (seperti *Recall* dan Q&A) pada sumbu Y, sementara sumbu X merepresentasikan rasio hibridisasi (dari rasio Atensi Penuh di ujung kiri, semakin banyak lapisan RNN linier ke arah kanan, hingga 100% RNN murni di ujung kanan). Kurva kinerja (kuning, oranye, biru) untuk arsitektur teratas seperti Mamba dan DeltaNet tetap mendatar/sejajar di rasio awal, membuktikan tidak ada penurunan performa. Namun, begitu mencapai proporsi RNN yang dominan di ujung kanan grafik, kinerja semua model **turun anjlok secara tajam**. Kesimpulannya: Anda *wajib* menyisakan sebagian Atensi Softmax reguler untuk mengikat pemahaman konteks secara utuh.

#### 4.2.7. Alternatif Kuadratik Efisien: *DeepSeek Sparse Attention* (DSA)
Di sisi spektrum yang lain, kita tidak perlu membuang Atensi Softmax sama sekali. Ide alternatifnya adalah: jangan menghitung atensi untuk *seluruh* token masa lalu, melainkan seleksi secara pintar token-token historis mana saja yang krusial. Ini yang dilakukan dalam arsitektur **DeepSeek Sparse Attention (DSA)** yang diimplementasikan pada DeepSeek V3.2 dan GLM5.

DSA terdiri dari dua tahap:
1.  **Lightning Indexer (Indeksator Super Ringan):** Alih-alih menghitung matriks probabilitas yang masif, model memiliki *head* indeks kecil yang menghitung skor indeksasi $I_{t,s}$ antara Query dan Key historis, kemudian memasukannya ke fungsi aktivasi ReLU sederhana.
    $$ I_{t,s} = \sum_{j=1}^{H^I} w_{t,j}^I \cdot \text{ReLU}(q_{t,j}^I \cdot k_s^I) $$
2.  **Top-K Selection & Full Attention:** Model mengurutkan skor tersebut dan HANYA mengambil nilai $K$-teratas (kumpulan token paling relevan). Kemudian, barulah operasi atensi normal (kuadratik) dilakukan HANYA pada *subset* token yang sangat sedikit ini.
    $$ u_t = \text{Attn}(h_t, \{c_s \mid I_{t,s} \in \text{Top-K}(I_{t,:}) \}) $$

**Keuntungan DSA:**
Meskipun operasi akhirnya tetap kuadratik, ia beroperasi pada himpunan (*set*) $K$ yang sangat kecil, sehingga "faktor konstanta" menjadi sangat rendah. Selain itu, Anda tidak perlu melatih model DSA ini dari awal (*from scratch*). Pendekatan tipikalnya adalah melatih model Transformer *dense* normal untuk konteks pendek (misal 8K), lalu pada tahap *post-training/extension*, indeksator ringan ini "ditempelkan" untuk mengadaptasi model melahap konteks 128K tanpa melatih ulang fitur-fitur dasarnya. Evaluasi menunjukkan model DSA mampu menyamai performa komprehensif model *closed-source* (seperti Claude 4.5 Sonnet) dengan biaya inferensi prakomputasi (*prefilling*) dan dekode yang merangkak jauh lebih lambat (efisien) dibandingkan versi Transformer dasar.

---

### 4.3. *Mixture of Experts* (MoE)

Sekarang kita berpindah dari blok *Attention* menuju blok *Feed-Forward Network* (MLP). Memahami *Mixture of Experts* (MoE) sangat krusial karena hampir **semua model bahasa mutakhir pada skala tertentu hari ini adalah MoE**.
MoE secara konseptual tidak mengubah fundamental bahasa; Anda dapat membayangkannya sekadar sebagai: **"Bagaimana jika seseorang memberi saya MLP yang jauh lebih efisien?"**

#### 4.3.1. Konsep Dasar MoE
**Deskripsi Visual (Model Dense vs Sparse Model):**
*   **Model Dense (Kiri):** Memiliki blok Atensi yang diteruskan ke satu blok lapisan besar FFN tunggal (*Feed Forward Network*). Setiap token melewati FFN yang sama.
*   **Model Sparse (Kanan):** FFN besar raksasa tersebut dibongkar dan diganti dengan beberapa FFN independen yang lebih kecil (misalnya FFN 1, 2, 3, 4). Ini disebut sebagai para **Pakar (*Experts*)**. Di hadapan para pakar tersebut, ditambahkan sebuah lapisan pemilih atau **Router**. Router bertugas mengarahkan setiap token masukan ke pakar tertentu saja. Misalnya kata "The" dilempar ke FFN 2, dan kata "Dog" dilempar ke FFN 1.

**Sihir Utama MoE:**
Dengan memotong FFN menjadi banyak pakar, kita **meningkatkan jumlah total parameter model** secara dramatis. Tetapi, karena pada *forward pass* dan *backward pass* sebuah token HANYA mengaktifkan satu (atau beberapa) pakar saja, **biaya komputasi aktual (FLOPs) yang kita bayarkan tetap sama seperti FFN kecil**. Inilah mengapa model ini disebut model "Jarang" (*Sparse*).

#### 4.3.2. Mengapa MoE Menjadi Sangat Populer?
Awalnya, MoE dilihat secara parameter-sentris: Anda ingin punya model raksasa (parameter triliunan) tanpa harus membeli komputer dengan FLOPs tak berhingga.

Bukti empiris dari kertas kerja *Switch Transformer* (Google, Fedus et al. 2022) sangat mencengangkan:
1.  **Performa Inferensi Superior (FLOPs Sama, Parameter Lebih Banyak = Lebih Baik):** Grafik *Test Loss* menunjukkan bahwa dengan batasan FLOPs komputasi yang konstan, memperbanyak jumlah pakar (dari model *dense* T5 biasa menuju 16, 64, hingga 256 pakar) secara monoton dan tajam menurunkan *Loss*. Ini seperti mendapat performa tambahan secara gratis.
2.  **Kecepatan Pelatihan (*Faster to Train*):** Jika kita plot grafik *Neg Log Perplexity* terhadap waktu pelatihan, model *sparse* dengan 128 pakar (kurva biru) mencapai titik konvergensi (kecerdasan) **7x lipat lebih cepat** dalam skala waktu komputasi nyata dibandingkan dengan model dasar *dense* (kurva ungu) pada anggaran perangkat keras yang setara. Data dari OlMoE (2024) juga menegaskan model MoE membutuhkan FLOPs atau token sekitar 3x lebih sedikit untuk menyamai kinerja model *dense*.
3.  **Kinerja Tolok Ukur yang Kompetitif:** Pada grafik evaluasi (MMLU vs Activated Parameters), model-model MoE (Mixtral 8x22B, Llama 4 Scout, Grok-1) selalu menempati garis performa tertinggi (pojok kiri atas grafis), menghancurkan efisiensi dari model *dense* komparatif di kelas parameternya.

#### 4.3.3. Akses Ekstra untuk Paralelisasi Sistem
Karena FFN telah terpecah menjadi kotak-kotal pakar diskrit, kita mendapatkan bonus dari segi arsitektur perangkat keras: **Expert Parallelism**.
Anda tidak hanya bisa memotong model berdasarkan lapisan (*Model Parallelism*) atau membagi data (*Data Parallelism*), tetapi Anda dapat mendistribusikan para Pakar ini ke GPU yang berbeda-beda. Router akan menghitung destinasi token, dan jaringan sistem akan melemparkan aktivasi token tersebut ke node komputasi yang memegang "Pakar" spesifik tersebut, dieksekusi bersamaan, lalu hasilnya dilempar kembali.

#### 4.3.4. Lanskap Historis Model MoE (Timur dan Barat)
> **Catatan Tambahan Anekdotal:** Adaptasi MoE terasa berjalan lambat. Meski Google telah meriset ini secara intens pada 2022 (*Switch Transformer*, *GShard*), butuh waktu hingga tahun 2024 agar komunitas AI menyadari potensinya. Di belahan Barat, Mixtral, DBRX, dan Grok memelopori hal ini, dilanjutkan baru-baru ini oleh OpenAI o3 dan Llama 4 (model *Maverick* dan *Scout*). Namun, inovasi *open-source* paling beringas mengenai teknik MoE belakangan dipimpin oleh komunitas Tiongkok, khususnya dari iterasi seri **DeepSeek** (v1, v2, v3, R1) dan **Qwen** yang berani meracik arsitektur dan melakukan ablasi parameter tingkat rendah.

---

### 4.4. Fungsi Perutean (*Routing Function*)

MoE bergantung sepenuhnya pada kemampuan Router memilah masukan. Apa saja variasi pendekatannya?

#### 4.4.1. Jenis Penugasan Perutean
Ada tiga pendekatan teoretis (divisualisasikan dalam bentuk matriks probabilitas antara Daftar Token dan Daftar Pakar):
1.  **Token Chooses Expert (Pilihan Token - Top-K):** Setiap token, satu per satu, menginspeksi probabilitas dirinya terhadap seluruh pakar, dan memutuskan untuk masuk ke pakar dengan skor teratas (misalnya Top-1 atau Top-2).
2.  **Expert Chooses Token (Pilihan Pakar):** Setiap pakar melihat barisan seluruh token di dalam *batch*, lalu memilih sekian jumlah kapasitas token yang paling ia inginkan untuk diproses.
3.  **Global Routing:** Algoritma sentral menyelesaikan optimisasi *linear assignment problem* secara presisi layaknya masalah transportasi, untuk mengalokasikan token ke pakar secara seimbang demi efisiensi global absolut (Misal: algoritma BASE routing oleh Clark 2022).

Dalam praktiknya, hampir 100% LLM menggunakan **Token Chooses Expert (*Top-K Routing*)**. Ablasi (pengujian pelepasan fitur) menunjukkan bahwa metrik MMLU dan HellaSwag pada metode Pilihan Token (*TC*, kurva merah muda) jauh melampaui Pilihan Pakar (*EC*, kurva biru muda). Pendekatan global (Linear Assignment), meskipun secara teoretis sempurna, sangat mematikan efisiensi waktu kalkulasi pada skala GPU terdistribusi sehingga tidak dipakai di tingkat produksi.

#### 4.4.2. Algoritma Alternatif (*RL* dan *Hashing*)
Bisakah kita menggunakan metode lain selain Regresi Logistik Top-K?
*   **Hashing (Perutean Acak Dasar):** Secara mengejutkan, Anda tidak perlu jaringan saraf yang dipelajari sama sekali! Jika Anda murni membagi teks berdasarkan fungsi *Hash* dan melemparnya acak secara deterministik ke dalam pakar, ia tetap memberikan keuntungan performa dibandingkan model *dense*, meski tak setajam fungsi perutean yang dipelajari. Ini adalah *baseline* paling dasar.
*   **Reinforcement Learning (RL):** Karena Router mengambil keputusan diskrit ("pilih pintu A atau pintu B"), hal ini ideal untuk ditangani dengan kerangka masalah *Bandit* atau teknik RL seperti REINFORCE (Bengio 2013). RL memang memberikan hasil (*Validation Loss* menurun), tetapi kompleksitas teknis RL dan tingginya varians gradien membuatnya tidak sepadan jika dibandingkan dengan trik *deep learning* biasa (*gradient masking*) yang jauh lebih praktis.

#### 4.4.3. Matematika Rinci Perutean *Top-K*
Hampir semua makalah (Mixtral, DBRX, DeepSeek, Qwen) menggunakan turunan persamaan *Top-K* ini.

Setiap lapisan token direpresentasikan sebagai $u_t^l$. Keluaran dari lapisan MoE adalah $h_t^l$:
$$ h_t^l = \sum_{i=1}^N \left( g_{i,t} \cdot \text{FFN}_i (u_t^l) \right) + u_t^l $$

Di mana $g_{i,t}$ adalah gerbang probabilitas (skor) yang menentukan keterlibatan pakar. Ia didefinisikan sebagai:
$$
g_{i,t} = 
\begin{cases} 
s_{i,t}, & s_{i,t} \in \text{Top-K}(\{s_{j,t} \mid 1 \leqslant j \leqslant N\}, K) \\ 
0, & \text{lainnya}
\end{cases}
$$
Di mana $s_{i,t}$ (skor regresi logistik) biasanya dihitung menggunakan fungsi Softmax atas produk titik antara input token dan matriks bobot perutean pakar ($e_i$):
$$ s_{i,t} = \text{Softmax}_i \left( u_t^{l^\top} e_i^l \right) $$

Jika token tidak masuk ke dalam urutan K teratas (*Top-K*), maka nilai gerbangnya adalah $0$, yang berarti FFN tersebut sepenuhnya diabaikan dan tidak perlu dikomputasi, memberikan penghematan FLOPs raksasa.

#### 4.4.4. Inovasi DeepSeek: *Fine-grained* & *Shared Experts*
Hingga tahun 2023, metode *Top-K* klasik di atas adalah puncaknya. Lalu DeepSeek-MoE (V1-V2) menemukan trik arsitektural yang kini disalin oleh semua pihak (Qwen, Llama 4, dll).
Alih-alih memiliki, misalnya, 16 Pakar raksasa dan merutekan Top-2, DeepSeek memecah pakar tersebut menjadi ukuran yang lebih kecil dan halus (*fine-grained*), misalnya 64 pakar kecil, lalu merutekan ke Top-8.

Lebih inovatif lagi, mereka memisahkan beberapa FFN kecil tersebut dan mematenkannya sebagai **Shared Experts (Pakar Bersama)**.
*   **Routed Expert:** Harus dilewati melalui probabilitas *Router Top-K*.
*   **Shared Expert:** Mengelak dari *Router* sepenuhnya. Ia selalu AKTIF (*Always On*) untuk semua token tanpa mempedulikan identitas token tersebut.

**Alasan Konseptual:** Tanpa *Shared Expert*, model cenderung membuang-buang parameter *Routed Expert* hanya untuk mengingat aturan gramatikal, *stop-words*, atau pengetahuan linguistik komunal yang dipakai pada setiap kata. Dengan menugaskan pakar terdedikasi (*shared*) untuk memikul tugas dasar linguistik ini, pakar-pakar *routed* dibebaskan sepenuhnya untuk menjadi spesialis tingkat lanjut murni. Grafik ablasi DeepSeek menunjukkan bahwa memecah pakar (*fine-grained*) dipadukan dengan isolasi *Shared Expert* memberikan dorongan drastis pada tes logika inferensi seperti HellaSwag, PIQA, hingga logika matematika (MATH).

---

### 4.5. Pelatihan MoE dan Manajemen Kestabilan

Sekarang kita memasuki ranah keajaiban nyata dari *deep learning*. Pelatihan MoE seharusnya sangat sulit secara teori, tetapi bekerja dengan kokoh dalam praktik berkat serangkaian trik heuristik.

#### 4.5.1. Tantangan Pelatihan MoE
Kita wajib menggunakan model jarang (*sparse*) pada saat pelatihan agar pelatihan berjalan efisien (tidak menghidupkan semua pakar pada satu token). Namun terdapat *bug* matematis bawaan pada metode *Top-K*:
1.  **Ketidakmampuan Penurunan Gradien (*Non-differentiability*):** Keputusan pemotongan "di atas ambang K" dan "menjadi 0" adalah fungsi tangga (*step function*) yang tidak bisa diturunkan. Backpropagasi standar akan macet di sini.
2.  **Fenomena "Yang Kaya Semakin Kaya" (*Expert Collapse* / *Starvation*):** Inisialisasi bobot adalah acak. Di awal, *Router* akan secara naif merutekan *Top-K* ke satu pakar acak. Pakar tersebut dilatih dengan SGD, sehingga bobotnya semakin menyesuaikan (*align*) dengan token-token. Akibatnya, pada iterasi berikutnya, probabilitas regresi logistik pakar tersebut menjadi *semakin tinggi*, menarik lebih banyak token lagi. Ini adalah lingkaran setan (Siklus Umpan Balik Positif). Jika dibiarkan dengan SGD telanjang, hanya 1 atau 2 pakar dari ke-64 pakar yang akan menangani 100% token, sementara puluhan pakar lainnya mati kelaparan (*starvation*) dengan bobot bernilai 0. 

#### 4.5.2. Solusi: *Stochastic Jitter* vs *Balancing Losses*
Pendekatan awal (Shazeer 2017) mencoba menyuntikkan gangguan (*noise/jitter*) acak berbasis Gaussian ke dalam regresi logistik *Router* sebagai pemecah kebekuan (*tie-breaker*). Idenya agar model "terpaksa mengeksplorasi" pakar lain. Pendekatan ini sempat populer, namun ablasi Google pada 2022 membuktikan gangguan stokastik ini menyebabkan ahli (*experts*) menjadi terlalu ringkih (*brittle*) dan bahkan menurunkan kualitas akhir.

Praktik standar emas (konsensus mutlak) saat ini adalah menambahkan sebuah fungsi penalti buatan yang dikenal sebagai **Heuristic Load Balancing Loss (Kerugian Penyeimbang Beban)**.

#### 4.5.3. Matematika *Load Balancing Loss* (Switch Transformer)
Bagaimana cara kerja penyeimbang ini? Di luar perhitungan *loss* generatif pemodelan bahasa standar, kita menambahkan "Pajak Penyeimbang" (Auxiliary Loss) selama proses *training*.
Formula dari *Switch Transformer* (Fedus et al. 2022):
$$ \text{loss} = \alpha \cdot N \cdot \sum_{i=1}^N f_i \cdot P_i $$
*   $N$: Total jumlah pakar.
*   $f_i$: *Fraksi Aktual* dari jumlah token di dalam satu *batch* yang mendarat di pakar $i$.
    $$ f_i = \frac{1}{T} \sum_{x \in B} \mathbb{1}\{\text{argmax } p(x) = i\} $$
*   $P_i$: *Massa Probabilitas Rata-Rata* yang dialokasikan *Router* untuk pakar $i$.
    $$ P_i = \frac{1}{T} \sum_{x \in B} p_i(x) $$

Rumus ini secara fundamental tidak masuk akal jika dianalisis statis, tetapi akan **terlihat kejeniusannya jika diturunkan secara kalkulus (*derivative*)**. Turunan parsial dari kehilangan (*loss*) ini sehubungan dengan $p_i(x)$ berbanding lurus dengan parameter fraksionalnya. 
Dengan kata lain, dalam ruang gradien, **"Pajak" yang dikenakan berbanding lurus dengan kepopuleran pakar**. Semakin banyak token yang mampir ke sebuah pakar ($f_i$ membesar), maka *gradien negatif* (pinalti) yang dikirim kembali ke probabilitas perutean ($P_i$) akan semakin tajam, menekan pakar raksasa tersebut ke bawah agar mau mengizinkan pakar minoritas ikut terpilih. 

> Jika kita melepaskan penyeimbang ini (seperti pada studi OlMoE), grafik pelacakan alokasi (*Expert assignment*) seketika hancur: garis penugasan menunjukkan 90% token hanya dilempar ke Pakar 0 (Kuning) dan Pakar 6 (Merah Muda). Sebaliknya, dengan ditambahkan *Load Balancing Loss*, distribusi penguraian terlihat saling silang merata ke 8 pakar layaknya tumpukan mie yang harmonis.

#### 4.5.4. Evolusi Penyeimbang pada DeepSeek (v1, v2, v3)
DeepSeek membawa pinalti penyeimbang ke tahap yang radikal. Karena pelatihan LLM bukanlah sekadar teori algoritma ML tetapi juga rekayasa infrastruktur yang harus efisien.
*   **DeepSeek v1 & v2 (Per-Device Balancing):** Memastikan token merata per pakar belumlah cukup. Jika pakar A dan B berada di Server 1, sedangkan pakar C dan D berada di Server 2, dan perutean secara tidak seimbang merutekan sebagian besar ke Pakar A dan B, maka Server 1 akan mati-matian bekerja 100% utilitas sementara GPU Server 2 *idle* (pengangguran menunggu sinkronisasi). Maka mereka menambahkan persamaan *Device Balancing Loss* ($\mathcal{L}_{\text{DevBal}} = \alpha_2 \sum_{i=1}^D f_i' P_i'$) untuk meminalti agregasi deviasi beban berdasarkan lokasi peladen perangkat (*device hardware node*). DeepSeek v2 bahkan menambah *Communication Balancing Loss* ganda (untuk keluar masuknya komunikasi antar perangkat).
*   **DeepSeek v3 (Aux-Loss-Free Balancing):** Mengandalkan persamaan gradien pajak penyeimbang *loss* ternyata membebani pelatihan dan terkadang mencemari optimalisasi konvergensi bahasa utama. V3 meniadakan *Aux-Loss* tersebut secara dominan, beralih pada trik modifikasi regresi secara *Online Learning*. Mereka menambahkan variabel bias per-pakar $\color{blue}{b_i}$ langsung pada *Router*:
    $$ g_{i,t}' = \begin{cases} s_{i,t}, & s_{i,t} + \color{blue}{b_i} \in \text{Top-K}(\dots) \\ 0, & \text{lainnya} \end{cases} $$
    Bias ini memodulasi pemilihan *Router* pada antrean memori. Bias pakar sepi akan didongkrak secara artifisial, dan pakar penuh akan direm, mengikat stabilitas perangkat murni dengan aturan manipulasi *hyperparameter* waktu-berjalan alih-alih merusak propagasi kerugian (*loss*) *backpropagation* utama. 

#### 4.5.5. Isu Kestabilan *Softmax* Perutean dan *Z-Loss*
Di mana ada *Softmax*, di situ ada ancaman kestabilan eksponensial.
Pada FFN biasa kita tidak memiliki Softmax, tetapi di perute MoE (untuk mendapatkan $s_{i,t}$), Softmax kembali hadir.
*   **Solusi Presisi:** Makalah dasar Google (Zoph 2022) menetapkan bahwa komputasi indeksasi regresi *Router* INI SAJA HARUS dikalkulasi pada rentang presisi tinggi murni (*Float32*), meskipun bagian model *Transformer* sisanya menggunakan presisi ringan bfloat16/fp8, untuk mencegah pembulatan eksponensial *overflow* yang menghancurkan perutean.
*   **Solusi Z-Loss Tambahan:** Sama seperti Output probabilitas ujung, regresi *Router* sering diinjeksi dengan penalti *Z-Loss* artifisial $L_z(x) = \frac{1}{B}\sum \left( \log \sum e^{x} \right)^2$ (mencegah denominasi term Softmax lari dari konstanta 1).

#### 4.5.6. *Fine-Tuning* dan Bahaya *Overfitting*
Model MoE jarang adalah mimpi buruk untuk dilatih ulang / dikustomisasi skala kecil (*fine-tuning*). Graf pelatihannya memprihatinkan; *Validation Loss* terpisah menjauh ratusan kali lipat layaknya gunting yang membuka (*Massive Generalization Gap*), menandakan model *overfitting* buta. Mengingat kapasitas parameternya berlipat ganda, puluhan miliar parameter pakar ini "menghafal" data *fine-tune* yang terlalu sedikit dalam hitungan menit.

**Bagaimana menyelesaikannya?**
1.  **Solusi Arsitektural (Zoph et al):** Bekukan (jangan modifikasi) seluruh parameter di dalam MoE layer MLP selama tahap *fine-tune*. Update saja parameter *Attention* atau MLP *Dense* jika ada.
2.  **Solusi Brutal Pembelajaran Mendalam / The Bitter Lesson (DeepSeek):** Daripada pusing memikirkan isolasi parameter yang rapuh, cukup perbanjiri data *Fine-Tune*! DeepSeek v3 membersihkan himpunan data kurasi instruksional (*Supervised Fine-Tuning* / SFT) hingga 1.4 JUTA sampel berkualitas (Matematika, Koding, Penalaran). Pada volume set masif ini, MoE menembus kekangan *overfit* dan melanjutkan generalisasi skala parameternya hingga titik optimal. 

#### 4.5.7. *Upcycling*: Daur Ulang Model *Dense*
Ini adalah praktik trivia sejarah yang brilian dari masa transisi awal MoE (2023). Jika melatih MoE dari awal sangat mahal dan pelik, dapatkah kita menggunakan pondasi *checkpoint pretrained* model *dense* (Non-MoE) yang stabil, dan menyalinnya menjadi MoE?

Teknik ini disebut **Upcycling** (Daur Ulang Ke Atas).
Misalnya Anda memiliki model dasar *dense*. Pada layer FFN-nya, cukup salin-tempel matriks FFN tersebut sebanyak 8 kali ke samping! Lalu, instal sebuah matriks regresi *Router Top-K* yang diinisialisasi secara acak. Kemudian operasikan mesin pelatihannya. Kerangka acak stokastik dari router secara dinamis akan mendorong bobot masing-masing FFN kloningan (*copies*) ini untuk bermutasi menjauh satu sama lain, mengkhususkan diri ke arah klaster data independen seiring SGD berlangsung.
*   **MiniCPM:** Model 2.4B Dense di-*upcycle* menjadi 13.6B MoE (Top-2, 8 pakar). Peningkatan mutlak terjadi pada MMLU dan kemampuan bahasa hanya dengan membakar ~520 Miliar Token komputasi *fine-tune*.
*   **Qwen MoE Pertama:** Meng-*upcycle* model Qwen 1.8B menjadi MoE berkapasitas total 14.3B (Top-4, 60 Routed + 4 Shared). Performa model raksasa ini sangat dominan pada saat perilisannya, menandai rekor pertama kesuksesan mutlak komersialisasi *upcycling* oleh pengembang Tiongkok.

*Catatan: Saat ini, strategi upcycling telah usang. Mengingat infrastruktur MoE sudah paten, vendor merancang "Pencarian Heroik" (*Hero Runs*) dan melatih cluster MoE secara langsung dari nol (from scratch).*

---

### 4.6. Aspek Sistem dan Arsitektur Paralelisasi MoE

Arsitektur model bahasa tak dapat dipisahkan dari rekayasa sistemnya. Terdapat tiga sumbu paralelisasi umum: *Data Parallelism*, *Tensor/Model Parallelism*, dan *Pipeline Parallelism*.

#### 4.6.1. *Expert Parallelism*
MoE menciptakan matriks *layout* sumbu ke-empat: **Expert Parallelism**.
Anda dapat menempatkan masing-masing *expert* (yang berupa blok FFN) ke unit cip pengolah GPU independen. Saat perhitungan *forward-pass*, arsitektur komputasi modern seperti library **MegaBlocks** merombak algoritma aljabar matriks konvensional menjadi **Block Sparse Matrix Multiplication**. Mereka mengonsolidasikan paket-paket token kecil tak berimbang dari ratusan mesin, menjejalkan lalu meratakannya di dalam jalur perkalian matriks blok tanpa celah memori, memaksimalkan inti vektor komputasi (*Tensor Cores*).

#### 4.6.2. Hambatan Komunikasi (*Communication Overhead*)
Saat Anda memecah pakar ke ribuan peladen/mesin jaringan terpisah, Anda merusak performa *Intra-cluster Networking*. *Router* terpaksa menembakkan larik panjang vektor *hidden-activation* sebesar $\mathbb{R}^{d_{model}}$ ke melintasi kabel jaringan antar GPU (yang mana *latency*-nya lebih lambat dari bus data internal cip). Ini disebut biaya kompensasi komunikasi (*Communication Overhead*).

Nvidia melalui arsitekturnya (Nemotron 3) maupun gagasan *Latent MoE*, memperkenalkan konsep pra-kompresi. Alih-alih mengirim representasi vektor berdimensi raksasa untuk diproses Pakar Lintas Mesin, router terlebih dahulu "menurunkan proyeksi" (*Latent down-proj*) vektor tersebut ke dalam dimensi kompresi *bottleneck*. Dimensi kecil inilah yang ditransmisikan menembus kabel jaringan (*All-to-All Dispatch*). Sehabis diproses pakar jarak jauh, data kembali diunggah dan "diproyeksikan naik" (*Latent up-proj*). Trik ini meruntuhkan hambatan transfer pita lebar jaringan.

#### 4.6.3. Trivia: Pengguguran Token (*Token Dropping*) pada Inferensi
Berdasarkan memori kolektif infrastruktur awal MoE, perancangan fungsi *router* memliki kapasitor batasan kapasitas statis atau batas pakar maksimal (*expert capacity factor*).
Apa yang terjadi jika suatu pakar terlalu disukai (*over-popular*) dan antreannya mendadak membengkak?
Dalam kerangka lawas, antrean berlebih akan dieksekusi secara **Token Dropping (Pengguguran Token)**. Model secara "bisu" menghapus token tersebut, membuang nilainya, memasukkan keluaran nol (0) kembali ke residu dan berpura-pura tidak terjadi apa-apa.
Mengerikannya, kalkulasi ini diproses pada tingkat keranjang *batch*. Artinya, token inferensi *Anda* dapat tergugurkan masuk ke dalam antrean eliminasi hanya karena pertanyaan/masukan *(queries)* dari **pengguna antah berantah lain** di mesin peladen yang membebani pakar bahasa Spanyol atau matematika secara agresif! Permasalahan stokastisitas antrean ini (*dropped routing due to other people's queries*) adalah mimpi buruk arsitektur *serving* awan (*cloud*).
(Kabar baiknya: Pustaka *library* terdepan masa kini seperti Megablocks memodifikasi logika *sparse matrix* dinamis (*Drop-less Architecture*), dan masalah ini secara faktual kini menjadi kenangan buruk masa lalu).

---

### 4.7. Inovasi Tambahan pada Skala Raksasa: DeepSeek MoE v3 (MLA & MTP)

Sebagai konklusi puncak modul MoE ini, kita membedah evolusi model V3 raksasa (671B parameter total, tapi hanya aktif menggunakan **37B parameter** saat inferensi). Di samping manajemen *routing Top-K Softmax* dan regulasi heuristiknya yang telah kita pelajari di atas, DeepSeek v3 mengandalkan dua sihir terakhir untuk optimalisasi ekstrem:

**1. Multi-Head Latent Attention (MLA): Kompresi Atensi KV-Cache**
Bahkan dengan MoE, memori KV-Cache (menumpuk semua kunci dan nilai historis saat melahirkan teks secara *autoregressive*) masih mencekik RAM GPU. Ide MLA adalah: kita me-rutekan fitur *hidden* menuju sumbu vektor "Laten" $\color{blue}{c_t}$ yang sangat padat/terkompresi (dimensi rendah). 
Lalu dari ruang vektor laten $c_t$ ini sajalah perumusan Kunci (*Key*, $k_t^C$) dan Nilai (*Value*, $v_t^C$) diekstraksi ke atas/direkonstruksi pada saat detik komputasi atensi diperlukan. 
Selama tahap generatif melintasi waktu historis, **KV Cache murni tidak perlu lagi disimpan** di memori; HANYALAH representasi $c_t^{KV}$ laten mungil kompresi yang dipertahankan. Ini adalah pangkasan monumental pada kebutuhan memori VRAM GPU produksi. 
*(Terdapat gesekan teknis matematis antara proyeksi Latent MLA dengan rotasi metrik posisi RoPE yang telah kita pelajari di Bab 3, yang diselesaikan dengan mengisolasi kompartemen kunci minor non-laten demi dirotasi metrik RoPE).*

**2. Multi-Token Prediction (MTP): Penembak Runduk Prediksi Ganda**
Saat GPU menyelesaikan komputasi satu blok utuh yang besar untuk menebak 1 kata ke depan, ia seringkali memiliki waktu *idle* atau kelebihan kapabilitas memori *bandwidth*. Modul tambahan MTP (berupa lapisan transformer dangkal/ringan di gerbang ekor model) dirancang untuk memprediksi probabilitas secara serempak/paralel ke **2 hingga 3 kata ekstra ke masa depan**, sekaligus di dalam satu siklus iterasi pelintasan komputasi model (tanpa interupsi iterasi). 
Hal ini berfungsi ganda: (a) memberi gradien stabil pemahaman probabilitas semantik kontekstual jangka depan untuk pelatihan *(training)*, dan (b) secara literal bertindak sebagai *Speculative Decoder* internal gratis, yang menyodorkan draf multi-kata secepat kilat untuk dikonfirmasi di siklus dekode berikutnya, meroketkan kecepatan perolehan respon inferensi seketika ke pengguna.

---

### 4.8. Rangkuman Utama
Buku teks arsitektur level atas harus membukakan kacamata teknis tentang hal ini: 
1.  Batas Kuadratik Atensi Murni bukanlah jalan buntu; **Atensi Berbasis Linear (Dual matrix/RNN State Space)** dan **Indeksator Atensi Jarang (DSA)** menawarkan jembatan pelintasan pemrosesan konteks hingga jutaan token dengan efisiensi kecepatan dan akurasi tinggi.
2.  Inovasi **Mixture of Experts (MoE)** bukanlah sekadar gimik modifikasi, melainkan strategi kelangsungan hukum ekonomi komputasi. MoE menginjeksi kemampuan spesialisasi kecerdasan dan memperbesar kapasitas parameter model tanpa sedikitpun mengubah beban pemborosan energi listrik dan alokasi unit perhitungan FLOPs. 
3.  Desain yang terkesan tak bisa diturunkan secara kalkulus (seperti diskritisasi regresi *Top-K Router*) terbukti **bisa ditaklukkan sepenuhnya** oleh *deep learning* heuristik (melalui implementasi arsitektur perutean *Shared Expert* berkolaborasi dengan injeksi *Auxiliary Balancing Losses* pembasmi penumpukan monopoli beban distribusi/ *expert collapse*). 
4.  Lautan data empiris mengkonfirmasi: masa depan LLM di ranah terbuka/tertutup saat ini mutlak berada di bawah trajektori arsitektur berbasis MoE!