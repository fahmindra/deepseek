---
slug: ch-11-02
title: "DeepSeekMoE"
layout: chapter
---

## DeepSeekMoE - Menuju Spesialisasi Pakar Puncak dalam Model Bahasa *Mixture-of-Experts*
Di era Model Bahasa Besar (*Large Language Models*/LLMs) saat ini, *Mixture-of-Experts* (MoE) atau "Campuran Para Pakar" merupakan arsitektur yang sangat menjanjikan untuk mengelola dan menekan biaya komputasi saat kita menskalakan (memperbesar) parameter model. Namun, sebagai mahasiswa Kecerdasan Buatan, Anda harus menyadari bahwa arsitektur MoE konvensional seperti GShard—yang hanya mengaktifkan top-$K$ dari total $N$ pakar—menghadapi tantangan besar dalam memastikan **spesialisasi pakar (*expert specialization*)**. Spesialisasi pakar berarti setiap "pakar" (sub-jaringan) memperoleh pengetahuan yang terfokus dan tidak tumpang tindih.

Sebagai respons terhadap masalah ini, bab ini akan membedah arsitektur **DeepSeekMoE** yang dirancang menuju spesialisasi pakar tingkat puncak. Arsitektur ini melibatkan dua strategi utama:
1.  **Segmentasi Pakar Berbutir Halus (*Fine-Grained Expert Segmentation*):** Memecah pakar menjadi $mN$ pakar yang lebih kecil dan mengaktifkan $mK$ di antaranya, memungkinkan kombinasi pakar aktif yang jauh lebih fleksibel.
2.  **Isolasi Pakar Bersama (*Shared Expert Isolation*):** Mengisolasi $K_s$ pakar sebagai pakar yang selalu aktif (dibagikan) untuk menangkap pengetahuan umum, sehingga mengurangi redundansi pada pakar yang dirutekan (*routed experts*).

Kita akan melihat pembuktian empiris mulai dari skala 2B (Miliar) parameter, di mana DeepSeekMoE 2B menyamai performa GShard 2.9B (yang memiliki 1,5x lipat parameter pakar dan komputasi) dan mendekati batas atas teoretis model padat (*dense*). Selanjutnya, kita akan menskalakannya menjadi 16B parameter yang mampu menyamai LLaMA2 7B hanya dengan 40% komputasi. Terakhir, kita akan melihat upaya awal penskalaan raksasa ke 145B parameter yang menyamai DeepSeek 67B hanya dengan 28,5% (bahkan 18,2%) komputasi.

---

## 1. Pendahuluan

Riset dan praktik terkini telah membuktikan secara empiris bahwa, dengan ketersediaan data pelatihan yang memadai, menskalakan model bahasa dengan menambah jumlah parameter dan anggaran komputasi akan menghasilkan model yang secara luar biasa lebih cerdas (Brown et al., 2020; Hoffmann et al., 2022). Namun, kita harus mengakui bahwa upaya menskalakan model ke ukuran yang sangat ekstrem akan selalu diiringi oleh **biaya komputasi yang sangat mahal**.

Mempertimbangkan biaya yang sangat besar ini, arsitektur *Mixture-of-Experts* (MoE) (Jacobs et al., 1991; Shazeer et al., 2017) muncul sebagai solusi populer. MoE memungkinkan penskalaan jumlah parameter sekaligus menahan biaya komputasi pada tingkat yang wajar. Aplikasi MoE dalam arsitektur Transformers baru-baru ini telah membuahkan hasil yang sukses pada ukuran model yang substansial.

Meskipun menjanjikan, arsitektur MoE yang ada saat ini berpotensi menderita dua masalah fatal yang membatasi "spesialisasi pakar":
1.  **Hibriditas Pengetahuan (*Knowledge Hybridity*):** Praktik MoE konvensional sering menggunakan jumlah pakar yang sedikit (misalnya 8 atau 16). Akibatnya, token (kata/potongan kata) yang diarahkan ke pakar tertentu kemungkinan besar mencakup topik yang sangat beragam. Pakar tersebut terpaksa mencoba menampung pengetahuan yang sangat berbeda-beda ke dalam parameternya, yang pada akhirnya sulit untuk dimanfaatkan secara simultan.
2.  **Redundansi Pengetahuan (*Knowledge Redundancy*):** Token yang diarahkan ke pakar yang berbeda mungkin memerlukan pengetahuan dasar/umum yang sama. Akibatnya, beberapa pakar secara bersamaan mempelajari "pengetahuan yang sama" di dalam parameter mereka masing-masing, yang menyebabkan pemborosan atau redundansi parameter.

Masalah-masalah inilah yang menghalangi arsitektur MoE untuk mencapai performa batas atas teoretisnya. 

Untuk mengatasi masalah tersebut, diperkenalkan **DeepSeekMoE**. Mari kita analisis inovasi arsitekturnya. Sambil mempertahankan jumlah parameter yang sama, pakar disegmentasi (dipecah) ke dalam ukuran yang lebih kecil (*Fine-Grained Expert Segmentation*). Ini memungkinkan pengetahuan dipelajari secara lebih presisi. Selain itu, pakar khusus (*Shared Experts*) diisolasi agar selalu aktif untuk menangkap pengetahuan umum (*Shared Expert Isolation*), sehingga pakar lainnya tidak perlu mempelajari hal berulang.

Untuk memahami kehebatan arsitektur ini, mari kita analisis **Gambar 1**.

**Analisis Akademik Gambar 1: Perbandingan Performa di Open LLM Leaderboard**
*   **Sumbu X (Horizontal):** Merepresentasikan jumlah parameter yang *diaktifkan* (Activated Parameters) dalam satuan Miliar (Billions). Semakin ke kanan, semakin berat komputasi saat inferensi.
*   **Sumbu Y (Vertikal):** Merepresentasikan performa rata-rata (Average Performance) pada berbagai pengujian di *Open LLM Leaderboard*. Semakin tinggi, semakin cerdas modelnya.
*   **Titik Data:** Terdapat berbagai model *open-source* seperti LLaMA2 7B (biru tua), Falcon 7B, OPT, Pythia, dll. Garis putus-putus merah (*red dashed line*) adalah garis regresi linear yang ditarik dari seluruh model standar. Garis ini menunjukkan tren umum: *untuk mendapatkan performa lebih tinggi, Anda butuh parameter aktif yang lebih besar*.
*   **Bintang Merah (DeepSeekMoE 16B):** Perhatikan letak bintang merah. Ia berada jauh tinggi di atas garis tren linear.
*   **Kesimpulan:** Gambar ini membuktikan secara empiris bahwa DeepSeekMoE 16B secara konsisten mengungguli model-model lain dengan parameter aktif yang serupa. Yang paling menakjubkan, model ini mencapai performa yang setara dengan LLaMA2 7B, padahal LLaMA2 7B memiliki parameter aktif sekitar **2,5 kali lipat lebih besar**.

---

## 2. Konsep Dasar: *Mixture-of-Experts* untuk Transformers

Sebelum masuk ke inovasi, kita harus memahami dasar matematis MoE konvensional pada Transformers. Model bahasa Transformer standar dibangun dengan menumpuk $L$ lapisan blok Transformer. Setiap blok didefinisikan sebagai berikut:

$$ \mathbf{u}_{1:T}^l = \text{Self-Att}(\mathbf{h}_{1:T}^{l-1}) + \mathbf{h}_{1:T}^{l-1} \tag{1} $$

$$ \mathbf{h}_t^l = \text{FFN}(\mathbf{u}_t^l) + \mathbf{u}_t^l \tag{2} $$

**Penjelasan Variabel Persamaan (1) dan (2):**
*   $T$: Panjang urutan token (teks input).
*   $\text{Self-Att}(\cdot)$: Modul *Self-Attention* (Perhatian Mandiri) yang memproses konteks kalimat.
*   $\text{FFN}(\cdot)$: *Feed-Forward Network* (Jaringan Maju-Umpan), tempat mayoritas pengetahuan memorisasi disimpan.
*   $\mathbf{h}_{1:T}^{l-1}$: Vektor representasi tersembunyi (*hidden states*) dari lapisan sebelumnya ($l-1$).
*   $\mathbf{u}_{1:T}^l \in \mathbb{R}^{T \times d}$: Vektor representasi setelah melewati modul *Self-Attention* di lapisan ke-$l$. $d$ adalah dimensi tersembunyi (*hidden dimension*).
*   $\mathbf{h}_t^l \in \mathbb{R}^d$: Vektor *output* akhir untuk token ke-$t$ pada lapisan ke-$l$.

Praktik MoE mengubah arsitektur ini dengan **mengganti FFN standar dengan lapisan MoE** pada interval tertentu. Lapisan MoE terdiri dari banyak "pakar" (di mana setiap pakar adalah sebuah FFN independen). Mekanisme *router* (pengarah) akan menugaskan setiap token ke 1 atau 2 pakar saja.

Jika FFN ke-$l$ diganti dengan lapisan MoE, maka output $\mathbf{h}_t^l$ dihitung dengan rumusan berikut:

$$ \mathbf{h}_t^l = \sum_{i=1}^N \left( g_{i,t} \text{FFN}_i(\mathbf{u}_t^l) \right) + \mathbf{u}_t^l \tag{3} $$

$$ g_{i,t} = \begin{cases} s_{i,t}, & s_{i,t} \in \text{Topk}(\{s_{j,t} | 1 \leqslant j \leqslant N\}, K), \\ 0, & \text{sebaliknya,} \end{cases} \tag{4} $$

$$ s_{i,t} = \text{Softmax}_i \left( \mathbf{u}_t^{lT} \mathbf{e}_i^l \right), \tag{5} $$

**Penjelasan Variabel Persamaan (3), (4), dan (5):**
*   $N$: Total jumlah pakar dalam satu lapisan MoE.
*   $\text{FFN}_i(\cdot)$: Jaringan FFN untuk pakar ke-$i$.
*   $g_{i,t}$: Nilai gerbang (*gate value*) untuk pakar ke-$i$. Jika pakar tidak dipilih, nilainya 0.
*   $s_{i,t}$: Skor kedekatan/afinitas (*affinity score*) antara token ke-$t$ dengan pakar ke-$i$, dihitung menggunakan fungsi *Softmax*.
*   $\text{Topk}(\cdot, K)$: Fungsi yang menyeleksi $K$ skor afinitas tertinggi. Ini memaksakan *sparsity* (ketersebaran); token hanya dihitung di $K$ pakar, menghemat komputasi.
*   $\mathbf{e}_i^l$: Vektor *centroid* (titik pusat rujukan) dari pakar ke-$i$ di lapisan ke-$l$, yang dipelajari selama pelatihan.

---

## 3. Arsitektur DeepSeekMoE

Berdasarkan fondasi MoE di atas, DeepSeekMoE diperkenalkan untuk mengeksploitasi potensi penuh dari spesialisasi pakar. Mari kita bedah visualisasi dan kerangka matematisnya melalui **Gambar 2**.

**Analisis Akademik Gambar 2: Ilustrasi Evolusi Arsitektur DeepSeekMoE**
*   **Subgambar (a) Rute Top-2 Konvensional:** Menampilkan lapisan MoE standar. FFN dipecah menjadi pakar besar berjumlah $N$. Sebuah modul *Router* mengarahkan token (Input Hidden) ke tepat 2 pakar ($K=2$).
*   **Subgambar (b) Segmentasi Pakar Berbutir Halus:** Parameter dari $N$ pakar besar dipecah (disegmentasi) menjadi lebih banyak pakar yang bentuknya lebih kecil (misal $2N$). Agar beban komputasi stabil, Router kini mengarahkan token ke lebih banyak pakar ($K=4$).
*   **Subgambar (c) DeepSeekMoE (Penambahan Isolasi Pakar Bersama):** Pakar yang telah disegmentasi kini dibagi fungsinya. Ada pakar bergaris hijau ("Shared Expert") yang *selalu aktif* untuk semua token. Sisanya adalah pakar bergaris biru ("Routed Expert") yang dipilih melalui Router ($K=3$). Total komputasi tetap sama dengan (a).

### 3.1. Segmentasi Pakar Berbutir Halus (*Fine-Grained Expert Segmentation*)

Dalam mengejar spesialisasi, kita menjaga total parameter dan biaya komputasi tetap konstan, namun kita menyegmentasi pakar menjadi butiran yang lebih halus. Caranya: kita membagi setiap pakar FFN menjadi $m$ pakar yang lebih kecil dengan cara mengecilkan dimensi tersembunyi (*hidden dimension*) FFN menjadi $\frac{1}{m}$ dari ukuran aslinya. 

Karena ukuran setiap pakar menjadi $\frac{1}{m}$, maka untuk mempertahankan biaya komputasi yang sama, kita harus **meningkatkan jumlah pakar yang diaktifkan sebanyak $m$ kali lipat**.

Rumus output MoE (merujuk ke pembaruan dari Persamaan 3-5) kini berubah menjadi:

$$ \mathbf{h}_t^l = \sum_{i=1}^{mN} \left( g_{i,t} \text{FFN}_i(\mathbf{u}_t^l) \right) + \mathbf{u}_t^l, \tag{6} $$

$$ g_{i,t} = \begin{cases} s_{i,t}, & s_{i,t} \in \text{Topk}(\{s_{j,t} | 1 \leqslant j \leqslant mN\}, mK), \\ 0, & \text{sebaliknya,} \end{cases} \tag{7} $$

$$ s_{i,t} = \text{Softmax}_i \left( \mathbf{u}_t^{lT} \mathbf{e}_i^l \right), \tag{8} $$

**Penjelasan Modifikasi Variabel (Persamaan 6-8):**
*   $mN$: Jumlah total pakar yang baru, di mana $N$ adalah jumlah awal, dan $m$ adalah faktor pemecahan (segmentasi).
*   $mK$: Jumlah pakar yang baru yang akan diaktifkan untuk setiap token. 
*   **Dampak Kombinatorial:** Pemecahan ini secara matematis meningkatkan fleksibilitas rute. Jika $N=16$ dengan routing Top-2, kemungkinan kombinasi pakar adalah $\binom{16}{2} = 120$. Jika kita memecah setiap pakar menjadi 4 ($m=4$), maka total pakar menjadi $64$ dan kita memilih $8$. Kemungkinan kombinasinya meledak menjadi $\binom{64}{8} = 4.426.165.368$. Fleksibilitas kombinatorial ini memastikan setiap pakar bisa sangat spesifik dalam satu jenis pengetahuan.

### 3.2. Isolasi Pakar Bersama (*Shared Expert Isolation*)

Bahkan dengan segmentasi halus, berbagai token dari topik berbeda mungkin tetap membutuhkan *pengetahuan umum/tata bahasa* yang sama. Jika dibiarkan, pakar-pakar yang tersebar akan secara mubazir mempelajari pengetahuan umum tersebut berulang kali di parameter mereka sendiri.

Solusinya adalah mengisolasi sejumlah $K_s$ pakar untuk difungsikan sebagai **Pakar Bersama (Shared Experts)**. Apapun hasil dari modul *Router*, setiap token **pasti** akan diproses oleh pakar bersama ini. Untuk menjaga komputasi tetap konstan, jumlah pakar yang dirutekan (*routed experts*) yang dipilih oleh Router akan dikurangi sebanyak $K_s$.

Formulasi matematis lengkap dari lapisan DeepSeekMoE kini menjadi:

$$ \mathbf{h}_t^l = \sum_{i=1}^{K_s} \text{FFN}_i(\mathbf{u}_t^l) + \sum_{i=K_s+1}^{mN} \left( g_{i,t} \text{FFN}_i(\mathbf{u}_t^l) \right) + \mathbf{u}_t^l, \tag{9} $$

$$ g_{i,t} = \begin{cases} s_{i,t}, & s_{i,t} \in \text{Topk}(\{s_{j,t} | K_s + 1 \leqslant j \leqslant mN\}, mK - K_s), \\ 0, & \text{sebaliknya,} \end{cases} \tag{10} $$

$$ s_{i,t} = \text{Softmax}_i \left( \mathbf{u}_t^{lT} \mathbf{e}_i^l \right). \tag{11} $$

**Penjelasan Variabel (Persamaan 9-11):**
*   Suku Pertama pada (9): Ini adalah komputasi dari $K_s$ pakar bersama yang dieksekusi tanpa syarat gerbang (*gate*).
*   Suku Kedua pada (9): Ini adalah komputasi dari pakar rute biasa.
*   $\text{Topk}(\dots, mK - K_s)$: Fungsi Top-K sekarang hanya mencari $(mK - K_s)$ pakar dari himpunan $(mN - K_s)$ pakar rute, karena $K_s$ kuota telah dipakai oleh pakar bersama.

### 3.3. Pertimbangan Keseimbangan Beban (*Load Balance Consideration*)

Strategi rute yang dipelajari otomatis rentan terhadap **ketidakseimbangan beban**, yaitu:
1.  **Runtuhnya Rute (*Routing Collapse*):** Model hanya memilih segelintir pakar secara terus-menerus, membiarkan pakar lain tidak terlatih (*mati*).
2.  **Kemacetan Komputasi (*Computation Bottleneck*):** Jika pakar disebar ke beberapa GPU (Perangkat), penumpukan token pada satu pakar akan menyebabkan GPU lain *idle* menunggu.

**Fungsi Kerugian Keseimbangan Tingkat Pakar (*Expert-Level Balance Loss*):**
Untuk mencegah runtuhnya rute, kita menambahkan penalti pada fungsi kerugian (*loss function*) saat pelatihan:

$$ \mathcal{L}_{\text{ExpBal}} = \alpha_1 \sum_{i=1}^{N'} f_i P_i, \tag{12} $$

$$ f_i = \frac{N'}{K' T} \sum_{t=1}^T \mathbb{1}(\text{Token } t \text{ memilih Pakar } i), \tag{13} $$

$$ P_i = \frac{1}{T} \sum_{t=1}^T s_{i,t}, \tag{14} $$

**Penjelasan Variabel (Persamaan 12-14):**
*   $N' = mN - K_s$: Total pakar rute.
*   $K' = mK - K_s$: Total pakar rute yang diaktifkan per token.
*   $T$: Jumlah total token dalam satu *batch*.
*   $f_i$: Frekuensi aktual (fraksi) seberapa sering pakar $i$ dipilih. $\mathbb{1}(\cdot)$ adalah fungsi indikator bernilai 1 jika benar, 0 jika salah.
*   $P_i$: Probabilitas agregat pakar $i$ menerima token berdasarkan nilai *Softmax* dari Router.
*   $\alpha_1$: Faktor hiperparameter untuk mengontrol seberapa kuat penalti ini ditegakkan.

**Fungsi Kerugian Keseimbangan Tingkat Perangkat (*Device-Level Balance Loss*):**
Jika kita menggunakan banyak GPU, kita mempartisi seluruh pakar rute ke dalam $D$ grup/perangkat $\{\mathcal{E}_1, \mathcal{E}_2, \dots, \mathcal{E}_D\}$. Kita hanya peduli agar setiap **perangkat GPU** mendapat beban komputasi yang merata, tanpa memaksakan keseimbangan ketat pada setiap pakar individual di dalam GPU tersebut.

$$ \mathcal{L}_{\text{DevBal}} = \alpha_2 \sum_{i=1}^D f_i' P_i', \tag{15} $$

$$ f_i' = \frac{1}{|\mathcal{E}_i|} \sum_{j \in \mathcal{E}_i} f_j, \tag{16} $$

$$ P_i' = \sum_{j \in \mathcal{E}_i} P_j, \tag{17} $$

**Penjelasan Variabel (Persamaan 15-17):**
*   $D$: Jumlah grup/perangkat keras (GPU).
*   $\mathcal{E}_i$: Himpunan pakar yang dialokasikan ke perangkat ke-$i$.
*   $f_i'$ dan $P_i'$: Rata-rata frekuensi dan total probabilitas untuk seluruh pakar di dalam satu perangkat GPU ke-$i$.
*   $\alpha_2$: Faktor hiperparameter untuk tingkat perangkat. Kita biasanya membuat $\alpha_2 > \alpha_1$ agar model lebih peduli pada pemerataan beban antar GPU untuk efisiensi sistem.

---

## 4. Eksperimen Validasi

### 4.1. Setup Eksperimental

**4.1.1. Data Pelatihan dan Tokenisasi:**
Model divalidasi menggunakan subset 100B (Miliar) token dari korpus multilingual DeepSeek (didominasi teks Inggris, Mandarin, matematika, dan kode). Tokenisasi menggunakan *Byte Pair Encoding* (BPE) dengan kosakata berukuran 8K (diperbesar nantinya untuk model raksasa).

**4.1.2. Infrastruktur:**
Menggunakan *framework* kustom HAI-LLM yang mengintegrasikan Paralelisme Tensor, ZeRO Data Parallelism, PipeDream Pipeline Parallelism, dan *Expert Parallelism*. Eksekusi dilakukan di kluster NVIDIA A100 dan H800 dengan interkoneksi NVLink dan InfiniBand.

**4.1.3. Hiperparameter:**
*   **Pengaturan Model Validasi:** 9 Lapisan Transformer, Dimensi tersembunyi 1280, 10 Kepala Perhatian (*Attention Heads*). Semua FFN diubah menjadi MoE. Total pakar = 16x ukuran FFN standar. Pakar yang aktif (termasuk *shared* dan *routed*) = 2x ukuran FFN standar. Ini menghasilkan model dengan **Total Parameter 2B** dan **Parameter Aktif 0.3B**.
*   **Pengaturan Pelatihan:** Pengoptimal AdamW ($\beta_1=0.9, \beta_2=0.95$, *weight decay*=$0.1$). Jadwal *learning rate* (LR): naik konstan 2K langkah pertama, lalu turun (dikali 0.316) pada 80% dan 90% waktu pelatihan. LR maks = $1.08 \times 10^{-3}$. *Batch size* 2K (4M token/batch). Total 25.000 langkah. Tanpa *dropout*.

**Tolok Ukur Evaluasi:** Menggunakan *Pile (Cross-entropy Loss), HellaSwag, PIQA, ARC (Accuracy), RACE, HumanEval, MBPP (Pass@1), TriviaQA, NaturalQuestions (Exactly Matching/EM).*

### 4.2. Hasil Evaluasi

Sebagai pembanding (*Baselines*), DeepSeekMoE diuji melawan arsitektur MoE terkenal lainnya dengan skala parameter 2B yang identik:
*   **Dense:** Transformer standar murni (0.2B parameter).
*   **Hash Layer:** Top-1 *hash routing* statis.
*   **Switch Transformer:** Top-1 *learnable routing*.
*   **GShard:** Top-2 *learnable routing* (0.3B parameter aktif).

| Metric | # Shot | Dense | Hash Layer | Switch | GShard | **DeepSeekMoE** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| # Total Params | N/A | 0.2B | 2.0B | 2.0B | 2.0B | 2.0B |
| # Activated Params | N/A | 0.2B | 0.2B | 0.2B | 0.3B | 0.3B |
| FLOPs per 2K Tokens | N/A | 2.9T | 2.9T | 2.9T | 4.3T | 4.3T |
| # Training Tokens | N/A | 100B | 100B | 100B | 100B | 100B |
| **Pile (Loss)** | N/A | 2.060 | 1.932 | 1.881 | 1.867 | **1.808** |
| **HellaSwag (Acc.)** | 0-shot | 38.8 | 46.2 | 49.1 | 50.5 | **54.8** |
| **PIQA (Acc.)** | 0-shot | 66.8 | 68.4 | 70.5 | 70.6 | **72.3** |
| **ARC-easy (Acc.)** | 0-shot | 41.0 | 45.3 | 45.9 | 43.9 | **49.4** |
| **ARC-challenge (Acc.)** | 0-shot | 26.0 | 28.2 | 30.2 | 31.6 | **34.3** |
| **RACE-middle (Acc.)** | 5-shot | 38.8 | 38.8 | 43.6 | 42.1 | **44.0** |
| **RACE-high (Acc.)** | 5-shot | 29.0 | 30.0 | 30.9 | 30.4 | **31.7** |
| **HumanEval (Pass@1)** | 0-shot | 0.0 | 1.2 | 2.4 | 3.7 | **4.9** |
| **MBPP (Pass@1)** | 3-shot | 0.2 | 0.6 | 0.4 | 0.2 | **2.2** |
| **TriviaQA (EM)** | 5-shot | 4.9 | 6.5 | 8.9 | 10.2 | **16.6** |
| **NaturalQuestions (EM)** | 5-shot | 1.4 | 1.4 | 2.5 | 3.2 | **5.7** |

*Tabel 1: Hasil validasi. Huruf tebal menunjukkan yang terbaik.*

**Kesimpulan:** Dengan jumlah total parameter dan parameter aktif yang sama, DeepSeekMoE menunjukkan **keunggulan absolut yang mutlak** atas GShard di semua jenis metrik pengukuran. 

### 4.3. DeepSeekMoE Mendekati Batas Atas Model MoE

Pertanyaan mendasar di ilmu arsitektur MoE: Seberapa efisien model ini dibanding model padat (*Dense*) penuh? Untuk mengujinya, DeepSeekMoE dikomparasi melawan GShard yang dibengkakkan komputasinya (GShardx1.5) dan Dense Transformer raksasa (Densex16).

| Metric | # Shot | GShard$\times$1.5 | Dense$\times$16 | **DeepSeekMoE** |
| :--- | :--- | :---: | :---: | :---: |
| Relative Expert Size | N/A | 1.5 | 1 | 0.25 |
| # Experts | N/A | 0 + 16 | 16 + 0 | 1 + 63 |
| # Activated Experts | N/A | 0 + 2 | 16 + 0 | 1 + 7 |
| # Total Expert Params | N/A | 2.83B | 1.89B | 1.89B |
| # Activated Expert Params | N/A | 0.35B | 1.89B | 0.24B |
| FLOPs per 2K Tokens | N/A | 5.8T | 24.6T | 4.3T |
| # Training Tokens | N/A | 100B | 100B | 100B |
| **Pile (Loss)** | N/A | 1.808 | 1.806 | **1.808** |
| **HellaSwag (Acc.)** | 0-shot | 54.4 | 55.1 | **54.8** |
| **PIQA (Acc.)** | 0-shot | 71.1 | 71.9 | **72.3** |
| **ARC-easy (Acc.)** | 0-shot | 47.3 | 51.9 | **49.4** |
| **ARC-challenge (Acc.)** | 0-shot | 34.1 | 33.8 | **34.3** |

*Tabel 2 (Dipotong untuk relevansi metodologi): Perbandingan skala model.*

**Kesimpulan Penting:** 
1.  DeepSeekMoE (Parameter aktif 0.24B, FLOPs 4.3T) mampu mengalahkan/menyamai GShardx1.5 yang memakan biaya komputasi jauh lebih besar (Parameter aktif 0.35B, FLOPs 5.8T).
2.  DeepSeekMoE menyamai Densex16 (FLOPs 24.6T). Densex16 bertindak sebagai *upper bound* teoretis (batas kejeniusan maksimal) dari kapasitas parameter tersebut karena *seluruh* jaringannya dieksekusi tanpa *sparse routing*. DeepSeekMoE mencapai titik jenius ini hanya dengan sebagian kecil komputasi.

### 4.4. Studi Ablasi (*Ablation Studies*)

Untuk membuktikan secara ilmiah bahwa inovasi (1) Segmentasi Halus dan (2) Isolasi Bersama adalah sumber kekuatan model ini, sebuah studi ablasi dipertunjukkan pada **Gambar 3**.

**Analisis Akademik Gambar 3: Studi Ablasi Komponen Arsitektur**
*   **Sumbu X (Horizontal):** Berbagai dataset pengujian linguistik dan logika (HellaSwag, PIQA, ARC, dll).
*   **Sumbu Y (Vertikal):** Performa Normalisasi (Maksimal 1.0/1.2).
*   **Batang Diagram:** Setiap tugas memiliki 4 warna batang.
    *   *Biru:* GShard Baseline (tanpa inovasi).
    *   *Kuning:* GShard + Isolasi Pakar Bersama. (Terlihat batang kuning konsisten lebih tinggi dari biru).
    *   *Hijau:* + Segmentasi berbutir halus (dibagi 2, jadi 31 routed expert). (Batang hijau lebih tinggi dari kuning).
    *   *Merah Bata:* Konfigurasi penuh DeepSeekMoE (dibagi 4, jadi 63 routed expert). (Batang tertinggi di semua tugas).
*   **Kesimpulan:** Penambahan setiap inovasi arsitektur memberikan kontribusi linear dan inkremental terhadap kecerdasan model. Segmentasi pakar yang *semakin halus* (membelah pakar dari 31 ke 63) secara konsisten meroketkan performa.

### 4.5. Analisis pada Spesialisasi Pakar (*Analysis on Expert Specialization*)

Mengapa DeepSeekMoE bekerja sangat baik? Jawabannya terletak pada "Spesialisasi Pakar" yang sejati, dibuktikan oleh **Gambar 4, 5, dan 6**.

**Analisis Akademik Gambar 4: Ketahanan terhadap Pemotongan Pakar**
*   **Sumbu X:** Rasio pakar rute yang dinonaktifkan secara paksa saat inferensi (0, 1/16, 2/16, dst).
*   **Sumbu Y:** *Pile Loss* (semakin tinggi loss, semakin model rusak/bodoh).
*   **Kurva:** GShardx1.5 (garis biru dengan simbol x) dan DeepSeekMoE (garis kuning dengan simbol bulat).
*   **Fenomena:** Saat pakar mulai "dimatikan", *Loss* pada kurva DeepSeekMoE meroket tajam ke atas jauh melampaui kurva GShard.
*   **Kesimpulan Ilmiah:** Mengapa model menjadi cepat bodoh saat pakar dimatikan? Ini justru membuktikan **redudansi yang sangat rendah**. Dalam DeepSeekMoE, setiap pakar menguasai pengetahuan unik (sangat terspesialisasi). Jika satu pakar hilang, pengetahuan itu musnah. Sebaliknya, GShard lebih kebal karena banyak pakarnya yang menduplikasi (mengulang-ulang) pengetahuan yang sama, menandakan efisiensi parameter yang buruk.

**Analisis Akademik Gambar 5: Kurva Akuisisi Pengetahuan Efisien**
*   **Sumbu X:** Jumlah pakar rute yang diaktifkan (3 hingga 7 pakar).
*   **Sumbu Y:** *Pile Loss*.
*   **Kurva:** Garis putus-putus biru horizontal adalah batas performa GShard. Garis kuning lengkung ke bawah adalah DeepSeekMoE.
*   **Fenomena:** DeepSeekMoE dengan hanya 4 pakar aktif (dari batas 7) sudah menyilang ke bawah menembus garis GShard!
*   **Kesimpulan:** Berkat segmentasi fleksibel, model ini merakit kepingan pengetahuan dengan akurasi yang mematikan. Hanya dengan sedikit kepingan aktif, ia sudah bisa mengalahkan arsitektur lama.

**Analisis Akademik Gambar 6: Pembuktian Komputasi Separuh**
*   **Sumbu X & Y:** Dataset Evaluasi dan Performa (Loss/Akurasi).
*   **Diagram Batang:** Membandingkan GShard "Penuh" vs DeepSeekMoE yang dilatih dari awal hanya menggunakan **SEPARUH** aktivasi pakar.
*   **Kesimpulan:** Bahkan dengan otak komputasional yang "dilumpuhkan" setengahnya, DeepSeekMoE masih mendominasi GShard. Ini membuktikan bahwa proporsi *parameter efektif* di dalam pakar yang aktif jauh lebih tinggi.

---

## 5. Penskalaan Menjadi DeepSeekMoE 16B

Berbekal kesuksesan arsitektur skala kecil, model diperbesar menjadi **16 Miliar (16B) total parameter** dan dilatih pada **2 Triliun (2T) Token**.

### 5.1. Setup Eksperimental
*   **Kosakata (*Vocab*):** 100.000 token.
*   **Arsitektur 16B:** 28 Lapisan Transformer, dimensi 2048, 16 *Attention heads*. Lapisan pertama menggunakan FFN standar (keseimbangan rute awal butuh adaptasi), lapisan 2-28 diubah menjadi MoE.
*   **Konfigurasi MoE:** 2 Pakar Bersama + 64 Pakar Rute. Setiap token dirutekan ke 2 Bersama + 6 Rute. Parameter total ~16.4B, Parameter aktif ~2.8B per token.
*   **Pelatihan:** *Batch size* 4.5K (18M token per batch), sekuens maksimum 4K, 106.449 langkah komputasi logis. *Pipeline parallelism* diaktifkan tanpa menghilangkan token. Faktor pakar level (*expert-level balance*) dibuat sangat rendah (0.001) agar tak mengganggu spesialisasi.
*   **Tolok Ukur Evaluasi:** Ditambah dengan GSM8K dan MATH (Logika Matematika), MMLU, WinoGrande, dan Evaluasi Bahasa Mandarin (CLUEWSC, CEval, CMMLU, CHID).

### 5.2. Hasil Evaluasi

Dibandingkan dengan LLaMA2 7B (yang memakan FLOPs jauh lebih boros karena *dense*):

| Metric | # Shot | LLaMA2 7B (Dense) | **DeepSeekMoE 16B** |
| :--- | :--- | :---: | :---: |
| # Total Params | N/A | 6.7B | 16.4B |
| # Activated Params | N/A | 6.7B | **2.8B** |
| FLOPs per 4K Tokens | N/A | 187.9T | **74.4T** |
| **HellaSwag (Acc.)** | 0-shot | 75.6 | **77.1** |
| **PIQA (Acc.)** | 0-shot | 78.0 | **80.2** |
| **ARC-challenge (Acc.)** | 0-shot | 49.0 | **49.8** |
| **GSM8K (EM)** | 8-shot | 15.5 | **18.8** |
| **MATH (EM)** | 4-shot | 2.6 | **4.3** |
| **HumanEval (Pass@1)** | 0-shot | 14.6 | **26.8** |
| **MBPP (Pass@1)** | 3-shot | 21.8 | **39.2** |

*Tabel 4 (Dipotong untuk relevansi): Komparasi LLaMA2 7B vs DeepSeekMoE 16B.*

**Pembahasan Akademis:** 
1.  **Efisiensi Komputasional Esktrim:** DeepSeekMoE 16B mengkonsumsi hanya **39.6% (~40%)** komputasi FLOPs dibanding LLaMA2 7B. Ini menjadikannya model yang ramah VRAM dan efisien secara daya, memungkinkannya di-deploy pada GPU kelas-menengah (seperti 40GB memori) tanpa *quantization*. Kecepatan inferensinya 2,5 kali lebih cepat dari model 7B standar.
2.  **Keunggulan Sintetis:** Pada pengujian nalar matematis (MATH 4.3 vs 2.6) dan pembentukan kode (HumanEval 26.8 vs 14.6), DeepSeekMoE mendemonstrasikan keunggulan kognitif yang sangat curam melawan arsitektur *dense*.

---

## 6. Penyelarasan (*Alignment*) untuk DeepSeekMoE 16B

Langkah selanjutnya adalah menyelaraskan (men-*tune*) model agar bisa menjadi agen percakapan (*Chatbot*). Terdapat skeptisisme di literatur sebelumnya bahwa model MoE susah untuk di-*fine-tune*. Untuk mematahkannya, kita mengimplementasikan **Pelatihan Halus Terselia (Supervised Fine-Tuning/SFT)**.

### 6.1. Setup Eksperimental
*   **Data Pelatihan:** 1.4 Juta contoh perintah-respon berkualitas tinggi, mencakup matematika, *coding*, ringkasan teks, logika deduktif, dan lain-lain (dwibahasa: Inggris & Mandarin).
*   **Hiperparameter:** Batch size 1024, dilatih selama 8 epoch dengan optimizer AdamW. *Learning rate* diatur konstan tanpa penjadwalan di angka $10^{-5}$. Panjang sequence hingga batas 4K dijejalkan (*packing*) sepadat mungkin.

### 6.2. Evaluasi SFT

Dibandingkan dengan model yang telah melewati SFT dari LLaMA2 dan model padat pendahulunya.

| Metric | # Shot | LLaMA2 SFT 7B | DeepSeek Chat 7B | **DeepSeekMoE Chat 16B** |
| :--- | :--- | :---: | :---: | :---: |
| # Total Params | N/A | 6.7B | 6.9B | 16.4B |
| # Activated Params | N/A | 6.7B | 6.9B | **2.8B** |
| **HellaSwag (Acc.)** | 0-shot | 67.9 | 71.0 | **72.2** |
| **HumanEval (Pass@1)** | 0-shot | 35.4 | 45.1 | **45.7** |
| **MBPP (Pass@1)** | 3-shot | 27.8 | 39.0 | **46.2** |
| **TriviaQA (EM)** | 5-shot | 60.1 | 59.5 | **63.3** |
| **CLUEWSC (EM)** | 5-shot | 48.4 | 66.2 | **68.2** |

*Tabel 5 (Dipotong untuk relevansi): Komparasi Chatbot.*

**Analisis Evaluasi Chat:** 
Hasilnya membuktikan arsitektur ini sangat ramah terhadap *Alignment*. Bahkan sebagai model obrolan interaktif, dengan hanya 40% dari total operasi komputasi (FLOPs), DeepSeekMoE Chat 16B sukses mendepak LLaMA2 SFT 7B pada tugas *Coding* dan pemahaman bahasa, dan menunjukkan keunggulan dwibahasa yang ekuivalen dengan DeepSeek Chat 7B yang berlatih secara *dense*.

---

## 7. Pekerjaan Berlangsung: DeepSeekMoE 145B (*Ongoing*)

Terdorong oleh fenomena tersebut, para ilmuwan melangkah pada fase rekayasa skala mega: **Model Berparameter 145 Miliar (145B)**. 

### 7.1. Setup Eksperimental

Ini adalah studi percontohan awal (*initial study*), sehingga data pelatihan dibatasi pada 245B token. 
*   **Arsitektur 145B:** 62 Lapisan Transformer, dimensi 4096, 32 *Attention heads*. Lapisan FFN diganti MoE. 
*   **Konfigurasi MoE Raksasa:** **4 Pakar Bersama** dan **128 Pakar Rute**. Setiap token dirutekan menuju 4 Bersama + 12 Rute (Total aktif = 16 pakar per token). 
*   **Parameter:** Total parameter = 144.6B. Parameter aktif = **22.2B**.
*   **Pengaturan Distrikusi Pelatihan:** Batch size 4.5K. Menggunakan strategi paralelisme ekstrem: setiap *layer*, semua pakar rute disebar melintasi 4 GPU (kombinasi *expert parallelism* + *data parallelism*). Faktor *Device-level balance* diatur tinggi (0.05) agar mencegah GPU mengalami diam tunggu (*idle bottleneck*), dan penalti runtuh rute dijaga rendah (0.003).

### 7.2. Evaluasi Skala Mega

Sebagai studi perbandingan kelas berat, model ini diadu dengan DeepSeek 67B *Dense* (Sang Raksasa) dan arsitektur kuno GShard 137B. Bahkan diuji juga versi "Otak Separuh" dari model ini (Hanya 6 pakar rute aktif / 12.2B parameter aktif).

| Metric | DeepSeek 67B (Dense) | GShard 137B | **DeepSeekMoE 145B** | **DeepSeekMoE 142B (Setengah Aktif)** |
| :--- | :---: | :---: | :---: | :---: |
| # Total Params | 67.4B | 136.5B | 144.6B | 142.3B |
| # Activated Params | 67.4B | 21.6B | 22.2B | **12.2B** |
| FLOPs per 4K Tokens | 2057.5T | 572.7T | 585.6T | **374.6T** |
| **HellaSwag (Acc.)** | 74.8 | 72.0 | **75.8** | 74.9 |
| **PIQA (Acc.)** | 79.8 | 77.6 | **80.7** | 80.2 |
| **HumanEval (Pass@1)**| 23.8 | 17.7 | 19.5 | 23.2 |
| **TriviaQA (EM)** | 57.2 | 52.5 | **61.1** | 59.8 |
| **WinoGrande (Acc.)** | 70.7 | 67.6 | **71.9** | 70.8 |

*Tabel 6 (Dipotong untuk relevansi): Komparasi skala 140B.*

**Signifikansi Sains:** 
1.  **DeepSeekMoE 145B menghancurkan GShard 137B**, membuktikan validitas tak tergoyahkan bahwa Segmentasi dan Isolasi Pakar mampu membuahkan hasil pada skala model monster.
2.  Model 145B menyamai performa DeepSeek 67B (model padat yang solid) tapi hanya mengkonsumsi **28,5% komputasi**!
3.  Fenomena Puncak: Varian "Setengah Aktif" (Hanya mengaktifkan 12.2B parameter), hanya tertinggal tipis, padahal beroperasi pada **18,2% tingkat komputasi** dari versi 67B. Ini adalah pendobrak batas ketahanan ekonomis dalam penciptaan AI generasi berikutnya.

---

## 8. Tinjauan Pustaka Terkait (*Related Work*)

Teknik *Mixture of Experts* (MoE) pertama kali dicetuskan pada awal dekade 90-an (Jacobs et al., 1991) untuk menangani data berbasis wilayah pakar independen. Pada 2017, Shazeer membawanya ke dalam ranah *Language Modeling* melalui arsitektur LSTM besar. Semenjak Transformer mendominasi NLP, banyak yang menduplikasi FFN menjadi lapisan MoE. 

**GShard** dan **Switch Transformer** adalah pionir yang pertama menskalakan MoE menjadi model raksasa menggunakan strategi *top-2* atau *top-1 learnable routing*. *Hash Layer* dan *StableMoE* kemudian menggunakan sistem rute statis agar jaringan tidak goyah selama masa latih (*training instability*). Secara umum, semua model MoE sebelumnya terpaku pada arsitektur konvensional yang menyisakan ruang kerugian besar dalam mewujudkan "Spesialisasi Pakar". Oleh karena itu, arsitektur yang kita bahas saat ini dirancang sebagai inovator pemecah batas konvensionalitas tersebut hingga tingkat paling ultim.

---

## 9. Kesimpulan

Dalam bab ini, kita telah membedah secara fundamental arsitektur **DeepSeekMoE** yang diformulasikan demi mencapai spesialisasi pakar absolut pada LLM. Melalui dua terobosan pilar—**Segmentasi Pakar Berbutir Halus** dan **Isolasi Pakar Bersama**—DeepSeekMoE membuktikan mampu menghasilkan performa kognitif model yang secara signifikan mengungguli arsitektur-arsitektur MoE yang sudah beredar di dunia akademis.

Pembuktian ini disusun secara bertahap. Mulai dari skala sederhana 2B parameter, kita melihat secara empiris bahwa DeepSeekMoE telah menyentuh *upper bound* teoretis dari arsitektur MoE. Bergerak pada penskalaan praktis 16B total parameter, model divalidasi sanggup sejajar dengan LLaMA2 7B dan DeepSeek 7B *Dense*, tetapi dengan keuntungan operasional biaya inferensi hanya 40%. Sifat adaptabilitas dan plastisitasnya juga divalidasi melalui penyelarasan *Supervised Fine-Tuning* (SFT) yang menghasilkan model *chat* superior. 

Secara konklusif, eksplorasi menuju model raksasa 145B memantapkan hukum skalabilitas model ini. Menyentuh ambang kecerdasan DeepSeek 67B *Dense* menggunakan sekadar ~28% konsumsi sumber daya komputasi merupakan revolusi yang memberikan kontribusi tak ternilai bagi keberlangsungan penelitian akademis maupun ekspansi efisiensi industri Kecerdasan Buatan di masa mendatang.