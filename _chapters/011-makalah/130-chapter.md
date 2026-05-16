---
slug: ch-11-13
title: "DeepSeek-V3"
layout: chapter
---
## Laporan Teknis DeepSeek-V3

DeepSeek-V3 adalah model bahasa berbasis **Campuran Pakar** (*Mixture-of-Experts* / MoE) yang sangat kuat dengan total **671 Miliar (671B)** parameter, di mana **37 Miliar (37B)** parameter diaktifkan untuk setiap token. Untuk mencapai inferensi yang efisien dan pelatihan yang hemat biaya, DeepSeek-V3 mengadopsi arsitektur **Atensi Laten Multi-kepala** (*Multi-head Latent Attention* / MLA) dan **DeepSeekMoE**, yang telah divalidasi secara mendalam pada DeepSeek-V2. 

Inovasi utama dalam model ini mencakup strategi **bebas kerugian tambahan** (*auxiliary-loss-free*) untuk penyeimbangan beban (*load balancing*) dan penetapan tujuan pelatihan **prediksi multi-token** (*multi-token prediction* / MTP) untuk performa yang lebih kuat. DeepSeek-V3 diprapelatih pada **14,8 triliun** token yang beragam dan berkualitas tinggi, diikuti oleh tahap **Pelatihan Halus Terselia** (*Supervised Fine-Tuning* / SFT) dan **Pembelajaran Penguatan** (*Reinforcement Learning* / RL). Evaluasi komprehensif menunjukkan bahwa DeepSeek-V3 mengungguli model sumber terbuka lainnya dan mencapai performa yang sebanding dengan model tertutup terkemuka. Yang luar biasa, model ini hanya membutuhkan **2,788 juta jam GPU H800** untuk pelatihan penuhnya dengan proses yang sangat stabil tanpa lonjakan kerugian (*loss spikes*) yang tidak dapat dipulihkan.

---

## 1. Pendahuluan

Dalam beberapa tahun terakhir, Model Bahasa Besar (LLM) telah mengalami iterasi cepat, memperkecil kesenjangan menuju Kecerdasan Umum Buatan (*Artificial General Intelligence* / AGI). DeepSeek-V3 hadir untuk mendorong batasan model sumber terbuka dengan skala 671B parameter.

DeepSeek-V3 dibangun dengan perspektif berwawasan ke depan, mengutamakan performa kuat sekaligus efisiensi ekonomi. Secara arsitektur, ia tetap menggunakan MLA untuk inferensi efisien dan DeepSeekMoE untuk pelatihan hemat biaya. Namun, DeepSeek-V3 memperkenalkan dua strategi tambahan:
1.  **Strategi Bebas Kerugian Tambahan (*Auxiliary-loss-free*):** Meminimalkan dampak negatif pada performa model yang biasanya timbul akibat upaya memaksakan keseimbangan beban antar pakar.
2.  **Tujuan Prediksi Multi-Token (MTP):** Meningkatkan sinyal pelatihan dan efisiensi data.

Untuk efisiensi komputasi, model ini mendukung pelatihan presisi campuran **FP8** dan optimisasi kerangka kerja pelatihan yang komprehensif. Algoritma **DualPipe** dirancang untuk paralelisme saluran pipa (*pipeline parallelism*) yang efisien, menyembunyikan sebagian besar komunikasi melalui tumpang tindih (*overlap*) komputasi-komunikasi.

---

**Deskripsi Gambar 1: Performa Tolok Ukur DeepSeek-V3 dan Kompetitornya**

Gambar ini menyajikan grafik batang yang membandingkan performa DeepSeek-V3 terhadap model-model ternama lainnya di berbagai metrik evaluasi.
*   **Sumbu-X (Horizontal):** Menampilkan berbagai tolok ukur seperti MMLU-Pro (Pengetahuan Umum), GPQA-Diamond (Sains tingkat PhD), MATH 500 (Matematika), AIME 2024 (Olimpiade Matematika), Codeforces (Pemrograman), dan SWE-bench Verified (Rekayasa Perangkat Lunak).
*   **Sumbu-Y (Vertikal):** Menunjukkan skor akurasi atau persentil dalam rentang 0 hingga 100%.
*   **Representasi Data:** DeepSeek-V3 (biru tua) dibandingkan dengan DeepSeek-V2.5 (biru muda), Qwen2.5-72B-Inst (abu-abu tua), Llama-3.1-405B-Inst (abu-abu muda), GPT-4o-0513 (emas), dan Claude-3.5-Sonnet-1022 (jingga).
*   **Tren Data:** DeepSeek-V3 menunjukkan performa dominan, terutama pada **MATH 500 (90,2%)** dan **Codeforces (persentil 51,6)**. Pada sebagian besar kategori, ia melampaui Llama-3.1-405B dan berdiri sejajar atau bahkan melampaui GPT-4o dan Claude-3.5-Sonnet pada tugas-tugas penalaran teknis.
*   **Kesimpulan Akademis:** DeepSeek-V3 membuktikan bahwa arsitektur MoE yang efisien dapat menandingi model-model *dense* yang jauh lebih besar dan model-model tertutup yang paling canggih saat ini.

---

## 2. Arsitektur

### 2.1. Arsitektur Dasar

DeepSeek-V3 tetap berada dalam kerangka kerja **Transformer**.

#### 2.1.1. Atensi Laten Multi-Kepala (*Multi-Head Latent Attention* / MLA)

MLA bertujuan mengurangi beban *cache* Kunci-Nilai (*Key-Value* / KV) selama inferensi melalui kompresi gabungan peringkat rendah. Misalkan $d$ adalah dimensi penyisipan (*embedding*), $n_h$ jumlah kepala atensi, $d_h$ dimensi per kepala, dan $\mathbf{h}_t$ input atensi untuk token ke-$t$:

$$ \mathbf{c}_t^{KV} = W^{DKV} \mathbf{h}_t \quad (1) $$
$$ [\mathbf{k}_{t,1}^C; \mathbf{k}_{t,2}^C; \dots; \mathbf{k}_{t,n_h}^C] = \mathbf{k}_t^C = W^{UK} \mathbf{c}_t^{KV} \quad (2) $$
$$ \mathbf{k}_t^R = \text{RoPE}(W^{KR} \mathbf{h}_t) \quad (3) $$
$$ \mathbf{k}_{t,i} = [\mathbf{k}_{t,i}^C; \mathbf{k}_t^R] \quad (4) $$
$$ [\mathbf{v}_{t,1}^C; \mathbf{v}_{t,2}^C; \dots; \mathbf{v}_{t,n_h}^C] = \mathbf{v}_t^C = W^{UV} \mathbf{c}_t^{KV} \quad (5) $$

**Penjelasan Variabel Rumus (1-5):**
*   $\mathbf{c}_t^{KV}$: Vektor laten terkompresi untuk kunci dan nilai.
*   $W^{DKV}$: Matriks proyeksi bawah (*down-projection*) untuk mengompresi input menjadi dimensi $d_c$.
*   $W^{UK}$ dan $W^{UV}$: Matriks proyeksi atas (*up-projection*) untuk memulihkan kunci dan nilai dari ruang laten.
*   $\mathbf{k}_t^R$: Kunci yang membawa **Penyisipan Posisi Rotasi** (*Rotary Positional Embedding* / RoPE) untuk menangkap informasi posisi.
*   $\mathbf{k}_{t,i}$ dan $\mathbf{v}_{t,i}$: Kunci dan nilai akhir yang digunakan dalam perhitungan atensi.
*   **Hubungan Logis:** Dengan hanya menyimpan $\mathbf{c}_t^{KV}$ dan $\mathbf{k}_t^R$ dalam *cache*, model secara drastis mengurangi penggunaan memori tanpa mengorbankan akurasi atensi.

#### 2.1.2. DeepSeekMoE dengan Penyeimbangan Beban Bebas Kerugian Tambahan

DeepSeekMoE menggunakan pakar yang lebih halus (*finer-grained*) dan mengisolasi beberapa pakar sebagai pakar bersama (*shared experts*). Output FFN $\mathbf{h}'_t$ dihitung sebagai:

$$ \mathbf{h}'_t = \mathbf{u}_t + \sum_{i=1}^{N_s} \text{FFN}_i^{(s)}(\mathbf{u}_t) + \sum_{i=1}^{N_r} g_{i,t} \text{FFN}_i^{(r)}(\mathbf{u}_t) \quad (12) $$

**Penjelasan Variabel Rumus (12):**
*   $\mathbf{u}_t$: Input ke lapisan FFN.
*   $N_s$: Jumlah pakar bersama yang selalu aktif.
*   $N_r$: Jumlah pakar yang dirutekan (*routed experts*).
*   $g_{i,t}$: Nilai gerbang (*gating value*) untuk pakar ke-$i$.

**Inovasi Load Balancing:**
Strategi tradisional menggunakan penalti kerugian tambahan (*auxiliary loss*) yang bisa merusak performa jika terlalu besar. DeepSeek-V3 memperkenalkan istilah bias $b_i$ untuk setiap pakar guna menentukan perutean *top-K*:

$$ g'_{i,t} = \begin{cases} s_{i,t}, & s_{i,t} + b_i \in \text{Topk}(\{s_{j,t} + b_j \mid 1 \le j \le N_r\}, K_r) \\ 0, & \text{sebaliknya} \end{cases} \quad (16) $$

**Penjelasan Variabel Rumus (16):**
*   $s_{i,t}$: Skor afinitas token terhadap pakar $i$.
*   $b_i$: Bias yang disesuaikan secara dinamis selama pelatihan. Jika pakar kelebihan beban (*overloaded*), $b_i$ dikurangi; jika kekurangan beban (*underloaded*), $b_i$ ditambah.
*   **Kesimpulan:** Hal ini menjaga beban pakar tetap seimbang secara alami tanpa mengganggu fungsi objektif utama model.

---

**Deskripsi Gambar 2: Ilustrasi Arsitektur Dasar DeepSeek-V3**

Diagram ini menunjukkan struktur blok Transformer DeepSeek-V3.
*   **Sisi Kiri (Transformer Block):** Menunjukkan aliran data standar: RMSNorm $\rightarrow$ Atensi $\rightarrow$ RMSNorm $\rightarrow$ Feed-Forward Network (FFN).
*   **Sisi Kanan Atas (DeepSeekMoE):** Memperlihatkan input $\mathbf{u}_t$ masuk ke modul "Router". Router memilih pakar dari "Routed Expert" (kotak biru bernomor 1 hingga $N_r$) melalui mekanisme "Top-$K_r$". Terdapat juga "Shared Expert" (kotak hijau) yang selalu menerima input. Output akhirnya adalah gabungan ($\mathbf{h}'_t$).
*   **Sisi Kanan Bawah (MLA):** Menunjukkan proses kompresi input $\mathbf{h}_t$ menjadi "Latent $\mathbf{c}_t^{KV}$". Vektor ini kemudian diproyeksikan kembali menjadi kunci dan nilai. Bagian yang diarsir biru menunjukkan data yang disimpan dalam *cache* selama inferensi.

---

### 2.2. Prediksi Multi-Token (*Multi-Token Prediction* / MTP)

MTP memperluas cakupan prediksi model ke beberapa token masa depan di setiap posisi, bukan hanya token berikutnya.

**Modul MTP:**
DeepSeek-V3 menggunakan $D$ modul sekuensial. Untuk kedalaman prediksi ke-$k$, representasi token ke-$i$ pada kedalaman $k-1$ ($\mathbf{h}_i^{k-1}$) digabungkan dengan penyisipan token ke-$(i+k)$ melalui proyeksi linear:

$$ \mathbf{h}_i'^k = M_k [\text{RMSNorm}(\mathbf{h}_i^{k-1}); \text{RMSNorm}(\text{Emb}(t_{i+k}))] \quad (21) $$

**Penjelasan Variabel Rumus (21):**
*   $M_k$: Matriks proyeksi untuk kedalaman $k$.
*   $\text{Emb}(t_{i+k})$: Penyisipan token masa depan yang akan diprediksi.
*   **Fungsi:** MTP memaksa model untuk merencanakan representasinya demi prediksi masa depan yang lebih baik, sehingga meningkatkan efisiensi data.

---

## 3. Infrastruktur

### 3.1. Kluster Komputasi

DeepSeek-V3 dilatih pada kluster yang dilengkapi dengan **2048 NVIDIA H800 GPU**. Setiap simpul (*node*) berisi 8 GPU yang terhubung dengan NVLink dan NVSwitch, sementara antar simpul dihubungkan via InfiniBand (IB).

### 3.2. Kerangka Kerja Pelatihan (*Training Framework*)

Pelatihan didukung oleh kerangka kerja **HAI-LLM**. Strategi paralelisme yang digunakan meliputi:
*   **Paralelisme Saluran Pipa (PP) 16-arah.**
*   **Paralelisme Pakar (EP) 64-arah** yang mencakup 8 simpul.
*   **Paralelisme Data (DP) ZeRO-1.**

#### 3.2.1. DualPipe dan Tumpang Tindih Komputasi-Komunikasi

Masalah utama pada EP lintas simpul adalah rasio komputasi-ke-komunikasi yang tidak efisien (1:1). **DualPipe** mengatasi ini dengan membagi setiap bongkahan (*chunk*) menjadi empat komponen: atensi, *dispatch* (pengiriman) all-to-all, MLP, dan *combine* (penggabungan) all-to-all. 

---

**Deskripsi Gambar 4: Strategi Tumpang Tindih Chunks Forward dan Backward**

Gambar ini adalah diagram linier waktu yang menunjukkan bagaimana berbagai tugas komputasi dan komunikasi berjalan bersamaan.
*   **Kotak Oranye:** Menunjukkan fase *Forward* (maju).
*   **Kotak Hijau/Biru:** Menunjukkan fase *Backward* (mundur) untuk input dan bobot.
*   **Kotak Ungu:** Komunikasi PP.
*   **Elemen Kunci:** Diagram menunjukkan bahwa saat GPU sedang melakukan komputasi MLP atau Atensi (atas), ia secara bersamaan melakukan pengiriman data (*Dispatch*) atau penggabungan (*Combine*) melalui jaringan (bawah).
*   **Kesimpulan:** Dengan mengatur jadwal ini secara cermat, hambatan komunikasi jaringan hampir sepenuhnya "tersembunyi" di balik waktu komputasi.

---

### 3.3. Pelatihan FP8

DeepSeek-V3 mempelopori kerangka kerja pelatihan presisi campuran FP8 pada model skala sangat besar. 

#### 3.3.2. Peningkatan Presisi dari Kuantisasi dan Perkalian

Karena rentang dinamis FP8 yang terbatas, DeepSeek-V3 menggunakan strategi **kuantisasi berbutir halus** (*fine-grained quantization*):
1.  **Untuk Aktivasi:** Elemen dikelompokkan dan diskalakan per ubin (*tile*) 1x128.
2.  **Untuk Bobot:** Elemen dikelompokkan per blok 128x128.

---

**Deskripsi Gambar 7: Metode Kuantisasi dan Akumulasi Presisi Tinggi**

Gambar ini terdiri dari dua bagian (a) dan (b).
*   **(a) Kuantisasi Berbutir Halus:** Menunjukkan bagaimana matriks "Input" dan "Weight" dibagi menjadi kotak-kotak kecil (ubin/blok). Setiap kotak memiliki "Scaling Factor"-nya sendiri untuk menjaga akurasi saat dikonversi ke FP8.
*   **(b) Peningkatan Presisi Akumulasi:** Menunjukkan bahwa hasil perkalian FP8 pada *Tensor Core* dikumpulkan sementara dalam bit terbatas, namun setiap interval $N_C=128$ elemen, hasilnya dipindahkan ke *CUDA Core* untuk akumulasi presisi penuh **FP32**.
*   **Kesimpulan:** Strategi ini memungkinkan model mendapatkan kecepatan FP8 tanpa kehilangan akurasi yang biasanya menyebabkan ketidakstabilan pelatihan.

---

## 4. Prapelatihan (*Pre-Training*)

### 4.1. Konstruksi Data

Korpus prapelatihan terdiri dari 14,8T token. Dibandingkan dengan DeepSeek-V2, porsi sampel matematika dan pemrograman ditingkatkan secara signifikan. Digunakan strategi **Isi-di-Tengah** (*Fill-in-Middle* / FIM) dengan struktur:
`<|fim_begin|> f_pre <|fim_hole|> f_suf <|fim_end|> f_middle <|eos_token|>`

### 4.2. Hiperparameter

*   **Lapisan Transformer:** 61 lapisan.
*   **Dimensi Tersembunyi:** 7168.
*   **Kepala Atensi:** 128 kepala.
*   **Optimizer:** AdamW dengan $\beta_1 = 0.9, \beta_2 = 0.95$, *weight decay* 0.1.
*   **Learning Rate:** Meningkat linear hingga $2.2 \times 10^{-4}$ dalam 2K langkah pertama, tetap konstan hingga 10T token, lalu meluruh secara kosinus.

---

**Deskripsi Gambar 8: Uji Tekanan "Needle In A Haystack" 128K**

Gambar ini adalah peta panas (*heatmap*) evaluasi kemampuan model menemukan informasi spesifik dalam konteks panjang (128 ribu token).
*   **Sumbu-X (Horizontal):** Panjang konteks dari 2K hingga 128K token.
*   **Sumbu-Y (Vertikal):** Kedalaman posisi informasi ("jarum") dalam dokumen dari 0% (awal) hingga 100% (akhir).
*   **Warna:** Warna hijau solid mendominasi seluruh bagan, menunjukkan skor sempurna (akurasi 100%) dalam pengambilan informasi.
*   **Kesimpulan:** DeepSeek-V3 memiliki memori jangka panjang yang sangat kuat dan stabil hingga batas maksimal jendela konteksnya.

---

### 4.4. Evaluasi

Evaluasi menunjukkan keunggulan DeepSeek-V3-Base.

**Tabel 3: Perbandingan DeepSeek-V3-Base dengan Model Sumber Terbuka Lainnya**

| Tolok Ukur (Metrik) | DeepSeek-V2 | Qwen2.5-72B | Llama-3.1-405B | DeepSeek-V3 |
| :--- | :---: | :---: | :---: | :---: |
| **Pakar Aktif** | 21B | 72B | 405B | **37B** |
| **MMLU (Akurasi)** | 78.4 | 85.0 | 84.4 | **87.1** |
| **MMLU-Pro (Akurasi)** | 51.4 | 58.3 | 52.8 | **64.4** |
| **HumanEval (Coding)** | 43.3 | 53.0 | 54.9 | **65.2** |
| **GSM8K (Math)** | 81.6 | 88.3 | 83.5 | **89.3** |
| **MATH (Math)** | 43.4 | 54.4 | 49.0 | **61.6** |

**Analisis Akademik:** DeepSeek-V3-Base secara komprehensif mengungguli Llama-3.1-405B yang jauh lebih besar dan Qwen2.5-72B. Hal ini membuktikan efisiensi luar biasa dari arsitektur MoE DeepSeek, di mana ia hanya mengaktifkan 37B parameter namun menghasilkan kecerdasan yang melampaui model *dense* dengan 405B parameter aktif.

---

## 5. Pascapelatihan (*Post-Training*)

### 5.1. Pelatihan Halus Terselia (*Supervised Fine-Tuning* / SFT)

Data SFT mencakup 1,5 juta instansi. Untuk tugas penalaran, DeepSeek-V3 mendistilasi kemampuan dari model **DeepSeek-R1**. Proses ini melibatkan pembuatan respons yang menyertakan pola refleksi dan verifikasi ala R1, namun tetap menjaga gaya bahasa yang ringkas dan efektif.

### 5.2. Pembelajaran Penguatan (*Reinforcement Learning* / RL)

DeepSeek-V3 menggunakan algoritma **Optimisasi Kebijakan Relatif Grup** (*Group Relative Policy Optimization* / GRPO).

#### 5.2.1. Model Imbalan (*Reward Model*)

Dua jenis model imbalan digunakan:
1.  **RM berbasis aturan:** Digunakan untuk matematika (format kotak jawaban) dan pemrograman (verifikasi kompiler). Ini tahan terhadap manipulasi imbalan (*reward hacking*).
2.  **RM berbasis model:** Digunakan untuk pertanyaan bentuk bebas (seperti penulisan kreatif). Model dilatih untuk memberikan umpan balik berdasarkan rantai pemikiran (*chain-of-thought*).

---

**Tabel 6: Perbandingan DeepSeek-V3 Chat dengan Model Chat Terkemuka**

| Tolok Ukur | Llama-3.1-405B-I | Claude-3.5-S | GPT-4o-0513 | DeepSeek-V3 |
| :--- | :---: | :---: | :---: | :---: |
| **MMLU (EM)** | 88.6 | 88.3 | 87.2 | **88.5** |
| **GPQA-Diamond** | 51.1 | 65.0 | 49.9 | **59.1** |
| **MATH-500 (EM)** | 73.8 | 78.3 | 74.6 | **90.2** |
| **AIME 2024 (Pass@1)**| 23.3 | 16.0 | 9.3 | **39.2** |
| **Codeforces (Perc.)** | 25.3 | 20.3 | 23.6 | **51.6** |

**Analisis Akademik:** Hasil pada **MATH-500 (90,2%)** dan **AIME 2024 (39,2%)** sangat mencolok. DeepSeek-V3 menghancurkan rekor model-model non-penalaran tradisional, bahkan melampaui GPT-4o dengan margin yang sangat lebar. Hal ini menunjukkan keberhasilan distilasi pengetahuan dari DeepSeek-R1 ke dalam model V3.

---

## 6. Kesimpulan, Limitasi, dan Pekerjaan Masa Depan

### Kesimpulan
DeepSeek-V3 adalah model MoE skala raksasa yang membuktikan bahwa performa tingkat dunia bisa dicapai dengan biaya pelatihan yang jauh lebih ekonomis melalui inovasi arsitektur (MLA, MoE) dan infrastruktur (FP8, DualPipe). Model ini berdiri sebagai model sumber terbuka terkuat saat ini, khususnya dalam domain teknis seperti matematika dan pemrograman.

### Limitasi
1.  **Ukuran Unit Deployment:** Unit penyebaran minimum masih cukup besar, sehingga memberikan beban bagi tim dengan sumber daya terbatas.
2.  **Potensi Peningkatan Kecepatan:** Meskipun sudah 2x lebih cepat dari V2, masih ada ruang untuk optimisasi lebih lanjut dengan perangkat keras masa depan.

### Pekerjaan Masa Depan
1.  **Dukungan Konteks Tak Terhingga:** Menyempurnakan arsitektur untuk mendukung panjang konteks yang lebih besar lagi secara efisien.
2.  **Penskalaan Data:** Terus mengiterasi kualitas data pelatihan dengan sinyal tambahan.
3.  **Kemampuan Berpikir Mendalam:** Mengintegrasikan kemampuan penalaran yang lebih dalam melalui perluasan panjang dan kedalaman rantai pemikiran.

***

*(Tamat)*
