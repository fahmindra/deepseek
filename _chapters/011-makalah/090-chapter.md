---
slug: ch-11-09
title: "ESFT"
layout: chapter
---

# Expert-Specialized Fine-Tuning: Biarkan Ahli Berfokus pada Bidangnya: Pelatihan Halus Spesialisasi-Pakar untuk Model Bahasa Besar Berarsitektur Renggang

## Abstrak

Pelatihan Halus Efisien Parameter (*Parameter-Efficient Fine-Tuning* / **PEFT**) sangat krusial untuk menyesuaikan Model Bahasa Besar (*Large Language Models* / **LLMs**) dengan sumber daya komputasi yang terbatas. Meskipun telah ada berbagai metode PEFT untuk LLM dengan arsitektur padat (*dense-architecture*), PEFT untuk LLM dengan arsitektur renggang (*sparse-architecture*) masih sangat kurang dieksplorasi.

Dalam karya ilmiah ini, kita akan mempelajari metode PEFT untuk LLM dengan arsitektur *Mixture-of-Experts* (**MoE**). Kontribusi utama dari karya ini mencakup tiga hal:
1.  **Investigasi Derajat Dispersi (*Dispersion Degree*):** Kami menginvestigasi derajat dispersi dari para "pakar" (*experts*) yang diaktifkan dalam tugas-tugas penyesuaian khusus. Ditemukan bahwa distribusi rute (*routing distribution*) untuk tugas tertentu cenderung **sangat terkonsentrasi** pada pakar tertentu, sedangkan distribusi pakar yang aktif bervariasi secara signifikan di berbagai tugas yang berbeda.
2.  **Pelatihan Halus Spesialisasi-Pakar (*Expert-Specialized Fine-Tuning* / ESFT):** Kami mengusulkan metode ESFT yang hanya melatih (*tunes*) pakar-pakar yang paling relevan dengan tugas turunan (*downstream tasks*), sementara pakar dan modul lainnya dibekukan (*freezing*). Hasil eksperimen mendemonstrasikan bahwa metode kami tidak hanya meningkatkan efisiensi pelatihan, tetapi juga **menyamai atau bahkan melampaui** performa dari pelatihan halus parameter penuh (*full-parameter fine-tuning* / FFT).
3.  **Analisis Arsitektur MoE pada ESFT:** Kami menganalisis lebih lanjut dampak arsitektur MoE pada pelatihan halus spesialisasi-pakar ini. Kami menemukan bahwa model MoE dengan pakar berbutir lebih halus (*finer-grained experts*) lebih menguntungkan dalam memilih kombinasi pakar yang paling relevan dengan tugas turunan, yang pada akhirnya meningkatkan efisiensi sekaligus efektivitas pelatihan.

---

## 1. Pendahuluan

Seiring dengan skala parameter Model Bahasa Besar (LLMs) yang terus membengkak (Meta, 2024; Mistral, 2024a; DeepSeek, 2024; Qwen, 2024), metode pelatihan halus efisien parameter (*parameter-efficient fine-tuning* / **PEFT**) (Han et al., 2024) menjadi semakin penting dalam mengadaptasi LLM prapelatihan (*pre-trained*) ke tugas penyesuaian khusus. Namun, karya-karya yang ada tentang PEFT seperti Adaptasi Peringkat Rendah (*Low-Rank Adaptation* / **LoRA**) dan P-Tuning (Hu et al., 2021; Liu et al., 2021) utamanya hanya berfokus pada LLM berarsitektur padat (*dense-architecture*), sedangkan penelitian pada LLM berarsitektur renggang (*sparse-architecture*) masih sangat tidak memadai.

Dalam karya ini, kami berfokus pada eksplorasi teknik PEFT di dalam LLM berarsitektur *Mixture-of-Experts* (MoE) (Mistral, 2024b; Databricks, 2024), yang akan diperkenalkan secara matematis pada Sub-bab 3.1. 

**Apa bedanya dengan model Padat (*Dense*)?**
Berbeda dengan model padat di mana *semua tugas* ditangani oleh matriks parameter yang persis sama, dalam arsitektur MoE, tugas yang berbeda diproses oleh pakar teraktivasi yang *berbeda* (Lepikhin et al., 2021; Fedus et al., 2021). Observasi empiris mengindikasikan bahwa **spesialisasi tugas (*task specialization*)** dalam sistem pakar ini adalah kunci dari kinerja LLM MoE (Dai et al., 2024).

Kami mengilustrasikan lebih lanjut spesialisasi tersebut di Sub-bab 3.2, di mana terbukti bahwa pakar yang diaktifkan oleh data dari tugas yang sama bersifat terkonsentrasi, sementara untuk tugas yang berbeda, pakarnya bervariasi secara signifikan. Ini mengisyaratkan bahwa model MoE menggunakan kombinasi pakar yang terspesialisasi untuk menangani tugas yang berbeda-beda.

Termotivasi oleh fenomena tersebut, kami mengusulkan **Pelatihan Halus Spesialisasi-Pakar (*Expert-Specialized Fine-Tuning* / ESFT)**, seperti yang diilustrasikan di Sub-bab 3.3. ESFT hanya melatih pakar yang memiliki tingkat kedekatan (*affinity*) tertinggi terhadap suatu tugas, sembari membekukan parameter dari pakar dan modul lainnya.

Keuntungan utama dari ESFT terletak pada dua aspek fundamental:
1.  **Mempertahankan Spesialisasi Pakar (*Maintaining Expert Specialization*):** ESFT mencegah penurunan spesialisasi yang sering terjadi pada pelatihan halus parameter penuh (*full-parameter fine-tuning* / FFT). Pada FFT, pakar yang *tidak mahir* dalam tugas tersebut juga ikut memperbarui parameternya, yang merusak struktur pengetahuan aslinya. Hasil eksperimen pada Sub-bab 5.1 menunjukkan bahwa ESFT dapat mencapai kinerja yang sejajar atau bahkan superior pada tugas turunan dibandingkan dengan FFT, dan secara bersamaan mempertahankan kinerja yang lebih baik pada tugas-tugas umum.
2.  **Menghemat Sumber Daya Komputasi (*Saving Computation Resources*):** ESFT hanya melatih parameter dari pakar yang dipilih. Hal ini secara efektif mengurangi kebutuhan ruang penyimpanan hingga **90%** dan waktu pelatihan hingga **30%** dibandingkan dengan FFT, seperti yang ditunjukkan di Sub-bab 5.2.

Selain itu, kami menelaah lebih dalam ke dalam mekanisme kerja metode ESFT. Kami menganalisis proses pemilihan pakar pada Sub-bab 6.1 dan mendemonstrasikan bagaimana ESFT memanfaatkan pakar yang dispesialisasikan secara efektif, karena hanya dengan memilih 5-15% pakar, ia mampu mencapai performa yang menjanjikan pada berbagai tugas yang berbeda. 

Kami juga menginvestigasi efisiensi ESFT di bawah berbagai kendala komputasional (Sub-bab 6.2), menampilkan kemampuannya untuk memanfaatkan sumber daya pelatihan secara lebih efisien dibandingkan dengan metode PEFT lain seperti LoRA. Studi kami di Sub-bab 6.3 menganalisis efek parameter yang dibagikan (*shared parameters*) dan yang tidak dibagikan (*non-shared parameters*) terhadap performa terspesialisasi maupun performa umum, yang menunjukkan bahwa memprioritaskan pelatihan selektif pada parameter yang tidak dibagikan adalah yang terbaik. Melalui studi ablasi (Sub-bab 6.4), kami menyoroti pentingnya skor relevansi pakar yang kami usulkan dan efektivitas arsitektur segmentasi pakar berbutir halus (*fine-grained*).

---

## 2. Pekerjaan Terkait (*Related Work*)

### 2.1 Pelatihan Halus Efisien Parameter (PEFT) untuk LLM Berarsitektur Padat

Tujuan dari PEFT (Han et al., 2024) adalah untuk menyesuaikan LLM secara efisien bagi tugas-tugas turunan, di mana studi yang ada utamanya hanya berfokus pada LLM berarsitektur padat. Metode PEFT untuk model padat umumnya dapat dikategorikan menjadi tiga pendekatan:
1.  **Menambahkan parameter baru (*Adding new parameters*):** Metode jenis ini membekukan parameter model yang ada dan melatih model pada sejumlah kecil parameter yang baru ditambahkan. *Adapter* (Houlsby et al., 2019; Pfeiffer et al., 2020; He et al., 2021; Wang et al., 2022) dan *Soft Prompt* (Li and Liang, 2021; Liu et al., 2021; Zhang et al., 2023b; Lester et al., 2021) adalah dua perwakilan tipikal dari kategori ini.
2.  **Memilih parameter yang ada (*Selecting existing parameters*):** Metode tipe ini melatih sebagian kecil dari parameter yang sudah ada, sementara sebagian besar parameter lainnya tetap dibekukan. Berdasarkan apakah ruang parameter yang dilatih tersebut bersambungan (*continuous*) atau tidak, metode ini umumnya dibagi menjadi pelatihan terstruktur (Guo et al., 2020; Gheini et al., 2021; He et al., 2023; Vucetic et al., 2022) dan pelatihan tidak terstruktur (Liao et al., 2023; Ansell et al., 2021; Sung et al., 2021; Xu et al., 2021).
3.  **Menerapkan adaptasi peringkat rendah (*Applying low-rank adaptation*):** LoRA (Hu et al., 2021; Fomenko et al., 2024) adalah metode PEFT yang sangat populer, yang mendekomposisi matriks bobot asal menjadi komponen berperingkat rendah (*low-rank components*). Karya-karya selanjutnya (Zhang et al., 2023a; Ding et al., 2023; Lin et al., 2024; Liu et al., 2023) telah memperkenalkan banyak perbaikan pada metode LoRA asli.

Namun, studi mengenai PEFT pada **model renggang (*sparse models*)** masih sangat langka. Dalam karya ini, kami menyeleksi dan melatih sebagian pakar berdasarkan kedekatannya (*affinity*) dengan tugas turunan, yang merupakan dimensi seleksi unik yang eksklusif hanya pada arsitektur MoE.

### 2.2 LLM MoE Berbutir Kasar dan Berbutir Halus (*Coarse- and Fine-grained MoE LLMs*)

Dibandingkan dengan LLM Padat (misal, seri LLaMA, Meta, 2023b,a), LLM MoE (misal, seri Mixtral, Mistral, 2024a,b) dapat memperbesar ukuran model (total parameter) sembari menghemat biaya pelatihan dan inferensi. 

Berdasarkan granularitas (ukuran butiran) dari pakar, model MoE skala besar yang ada saat ini secara umum dapat dibagi menjadi dua kategori: LLM berbutir kasar (*coarse-grained*) dan LLM berbutir halus (*fine-grained*). 
Sebagian besar LLM MoE yang ada (Lepikhin et al., 2021; Fedus et al., 2021; Roller et al., 2021; Dai et al., 2022; Shen et al., 2024) memiliki pakar **berbutir kasar** di mana jumlah total pakar sangat terbatas. Sebagai contoh, 2 dari 8 pakar diaktifkan untuk seri Mixtral MoE (Mistral, 2024a,b) dan Grok-V1 (XAI, 2024). Akibatnya, seorang pakar tunggal harus mempelajari pola rumit dari berbagai tugas domain yang berbeda secara bersamaan (ini menyebabkan kebingungan kognitif bagi pakar tersebut).

Untuk mengatasi masalah ini, DeepSeek MoE (Dai et al., 2024) telah memperkenalkan **segmentasi pakar berbutir halus**. Dalam DeepSeek-V2 (DeepSeek, 2024), terdapat sebanyak 162 pakar, dengan 8 pakar aktif (atau 8 dari 66 pakar yang diaktifkan untuk versi DeepSeek-V2-Lite). Pembagian pakar secara halus memastikan tingkat spesialisasi yang tinggi di antara para pakar. Selain itu, sistem pakar yang terspesialisasi ini memungkinkan pemilihan pakar yang *paling relevan* dengan suatu tugas demi efisiensi pelatihan yang maksimal.

---

## 3. Metode (*Methods*)

### 3.1 Pendahuluan: *Mixture-of-Experts* untuk Transformers

Arsitektur *Mixture-of-Experts* (MoE) untuk Transformers mengganti *Feed-Forward Networks* (FFNs) standar dengan lapisan-lapisan MoE. Setiap lapisan MoE terdiri dari *banyak pakar*, di mana secara struktural, setiap pakar identik dengan satu jaringan FFN. 

Token-token dari kalimat didistribusikan dan diproses hanya oleh *sebagian kecil* dari pakar (subset pakar) yang paling relevan. Keputusan ini didasarkan pada **skor kedekatan (*affinity scores*)**, yang memastikan efisiensi komputasional di dalam lapisan MoE. Status representasi tersembunyi (*hidden state*) keluaran, yakni $\mathbf{h}_t^l$, untuk token ke-$t$ pada lapisan MoE ke-$l$ dihitung sebagai berikut:

$$ \mathbf{h}_t^l = \sum_{i=1}^N \left( g_{i,t}\text{FFN}_i^n(\mathbf{u}_t^l) \right) + \mathbf{u}_t^l, \tag{1} $$

$$ g_{i,t} = \begin{cases} s_{i,t}, & s_{i,t} \in \text{TopK}(\{s_{j,t} | 1 \leqslant j \leqslant N\}, K), \\ 0, & \text{sebaliknya,} \end{cases} \tag{2} $$

$$ s_{i,t} = \text{Softmax}_i \left( \mathbf{u}_t^{l \top} \mathbf{e}_i^l \right), \tag{3} $$

**Penjelasan Mendetail Variabel dan Rumus:**
*   $\mathbf{h}_t^l$: Adalah vektor keluaran (*output*) untuk token ke-$t$ di lapisan ke-$l$.
*   $N$: Adalah jumlah *total* pakar yang ada di lapisan tersebut.
*   $\text{FFN}_i^n(\cdot)$: Adalah fungsi komputasi untuk pakar ke-$i$ (yang berupa jaringan *Feed-Forward*).
*   $\mathbf{u}_t^l$: Adalah vektor input (masukan) dari token ke-$t$ di lapisan ke-$l$ yang akan masuk ke jaringan pakar. Ditambahkan kembali di akhir (seperti pada Rumus 1) sebagai *residual connection* (koneksi sisa) untuk menstabilkan gradien saat pelatihan.
*   $g_{i,t}$: Adalah **nilai gerbang (*gate value*)** untuk pakar ke-$i$. Nilai ini menentukan seberapa besar kontribusi (bobot) keluaran pakar ke-$i$ terhadap hasil akhir $\mathbf{h}_t^l$. Jika pakar tersebut *tidak* terpilih oleh fungsi $\text{TopK}$, maka $g_{i,t} = 0$, artinya pakar tersebut dilewati begitu saja (menghemat komputasi).
*   $s_{i,t}$: Adalah **afinitas token-ke-pakar (*token-to-expert affinity*)** atau probabilitas bahwa token tersebut cocok diproses oleh pakar $i$.
*   $\text{TopK}(\cdot, K)$: Adalah operasi himpunan yang menyeleksi $K$ buah skor afinitas tertinggi ($s$) dari seluruh $N$ pakar.
*   $\text{Softmax}_i(\cdot)$: Adalah fungsi yang menormalisasi skor menjadi probabilitas antara 0 dan 1.
*   $\mathbf{e}_i^l$: Adalah vektor pusat (*centroid*) dari pakar ke-$i$ di lapisan ke-$l$. Perkalian titik (*dot product*) $\mathbf{u}_t^{l \top} \mathbf{e}_i^l$ adalah inti dari penilaian "seberapa mirip/cocok" vektor token masukan dengan karakteristik dari pakar tersebut.

Baru-baru ini, **DeepSeekMoE** (Dai et al., 2024) mengusulkan penyempurnaan pada arsitektur MoE melalui beberapa teknik:
1.  **Segmentasi berbutir halus (*Fine-grained segmentation*):** Memecah setiap pakar (misal FFN besar) menjadi beberapa pakar yang lebih kecil, namun tetap mempertahankan persentase/fraksi pakar yang sama untuk memproses setiap token. Hal ini memungkinkan pakar berspesialisasi pada jenis pengetahuan yang berbeda sambil mempertahankan biaya komputasi yang konstan.
2.  **Isolasi pakar bersama (*Shared expert isolation*):** Memanfaatkan "pakar bersama" (*shared experts*) yang secara permanen memproses **semua token** untuk menangkap pengetahuan umum (*common knowledge*). Ini akan mengurangi redudansi memori dan meningkatkan efisiensi. 

Keluaran dari lapisan MoE di arsitektur **DeepSeekMoE** adalah:

$$ \mathbf{h}_t^l = \sum_{i=1}^{K_s} \text{FFN}_i^s(\mathbf{u}_t^l) + \sum_{i=1}^N \left( g_{i,t}\text{FFN}_i^n(\mathbf{u}_t^l) \right) + \mathbf{u}_t^l, \tag{4} $$

$$ g_{i,t} = \begin{cases} s_{i,t}, & s_{i,t} \in \text{TopK}(\{s_{j,t} | 1 \leqslant j \leqslant N\}, K - K_s), \\ 0, & \text{sebaliknya,} \end{cases} \tag{5} $$

**Penjelasan Tambahan untuk DeepSeekMoE:**
*   $K_s$: Adalah jumlah *Shared Experts* (Pakar Bersama).
*   $\text{FFN}_i^s$: Menunjukkan *Shared Experts* (yang dihitung untuk *semua* token tanpa melalui gerbang pemilihan/selalu dijumlahkan di awal Rumus 4).
*   $\text{FFN}_i^n$: Menunjukkan *Non-shared experts* (Pakar yang harus dipilih/dirutekan melalui *Router*).
*   Pada Rumus 5, fungsi $\text{TopK}$ kini hanya memilih $(K - K_s)$ pakar, karena $K_s$ kuota telah dialokasikan secara paksa kepada pakar bersama. Ini memastikan bahwa meskipun arsitektur pakarnya dipecah menjadi sangat kecil ($m$ buah yang lebih kecil), komputasi keseluruhannya tetap setara dengan arsitektur berbutir kasar.

---

### Analisis Akademis: Gambar 1 - Perbandingan Metode *Fine-Tuning*

Mari kita telaah **Gambar 1** secara komprehensif sebagai representasi visual dari inovasi makalah ini.

*   **Deskripsi Visual:** Gambar ini terdiri dari tiga panel yang membandingkan arsitektur blok Transformer selama masa pelatihan (*training task*). Elemen-elemen yang dilatih (*Trainable Modules*) ditandai dengan warna **hijau kekuningan**, sementara elemen yang dibekukan (*Frozen Modules*) berwarna **biru muda**.
*   **Panel Kiri: Full-Parameter Fine-Tuning (FFT):**
    *   Dalam metode tradisional ini, *seluruh blok* arsitektur dilatih. Kotak *Feed-Forward Layer* dan *Attention & Norm* berwarna hijau. Ini adalah metode "tumbuk rata" yang sangat boros memori dan merusak spesialisasi bawaan model karena semua matriks diperbarui.
*   **Panel Tengah: Low Rank Adaptation (LoRA):**
    *   Matriks bawaan (*Pretrained Weights*) dibekukan (warna biru). Sebagai gantinya, *bypass matrix* berskala kecil, yaitu **LoRA - A** dan **LoRA - B** ditambahkan dan dilatih (warna hijau). Ini efisien dalam hal parameter, tetapi penambahannya berlaku *secara global* pada setiap token tanpa memedulikan hierarki pakar di dalam MoE.
*   **Panel Kanan: Expert-Specialized Fine-Tuning (ESFT) - (Metode Usulan):**
    *   Arsitekturnya adalah *Mixture-of-Experts* (perhatikan blok *Experts* 1, 2, 3... $N_e$ di dalam *Feed-Forward Layer*).
    *   *Perbedaan Krusial:* Modul *Attention & Norm*, *Router*, dan mayoritas Pakar **dibekukan** (berwarna biru). Hanya Pakar spesifik (contohnya Pakar nomor 2, yang berwarna hijau) yang dilatih. Pemilihan Pakar nomor 2 ini diatur oleh komponen *Top-K* berdasarkan probabilitas afinitas tertinggi.
*   **Kesimpulan Akademis:** Gambar 1 ini merangkum secara elegan bahwa ESFT adalah metode bedah saraf presisi tingkat tinggi (*precision neurosurgery*) untuk LLM. ESFT hanya memperbarui sinapsis (Pakar) yang relevan dengan tugas spesifik, menekan biaya komputasi, dan secara simultan melestarikan pengetahuan murni (Pakar lain) yang tidak berelasi dengan tugas tersebut.

---

### 3.2 Menyelidiki Spesialisasi Pakar Spesifik Tugas (*Task-Specific Expert Specialization*) dalam Model MoE

Meskipun kesuksesan LLM MoE sangat signifikan, pemahaman jernih tentang mekanisme kerja di baliknya tetaplah kabur. Kami melakukan eksperimen penyelidikan (*probing experiments*) untuk memahami bagaimana pakar yang *tidak dibagikan* (pakar perutean) dimanfaatkan di berbagai tugas yang berbeda. 

Eksperimen ini mengungkap fenomena **Spesialisasi Pakar** dalam model MoE pada dua aspek:

**1. Rute Pakar Terkonsentrasi pada Tugas yang Sama (*Expert Routing is Concentrated in the Same Task*)**
Kami menginvestigasi distribusi nilai gerbang yang dinormalisasi (*normalized gate values*). Kami melihat total nilai gerbang yang ditugaskan ke setiap pakar, dibagi dengan total dari seluruh pakar. 

*Analisis Akademis Gambar 2 (Top Expert distribution for specific tasks):*
*   **Sumbu X:** Menunjukkan peringkat Pakar (dari 1 hingga 64).
*   **Sumbu Y:** Menunjukkan Nilai Gerbang Rata-Rata Normalisasi (*Normalized Average Gate Value*).
*   **Garis Kurva:** Berbagai warna mewakili tugas yang berbeda (Niat/Intent, Ringkasan/Summary, Hukum/Law, Terjemahan/Translation, Matematika/Math, Kode/Code).
*   **Kesimpulan:** Grafik menunjukkan kurva eksponensial menurun yang sangat curam (*steep decay*). Hanya *segelintir kecil pakar* (sekitar 1 hingga 8 pakar pertama di sumbu X) yang menyerap porsi terbesar (mayoritas) nilai gerbang. Ini membuktikan bahwa alokasi pakar **sangat terkonsentrasi** untuk tugas tertentu. Pakar yang berada di sisi kanan sumbu X hampir tidak tersentuh (nilainya mendekati 0).

**2. Pakar Aktif Bervariasi Secara Signifikan Antar Tugas (*Active Experts Vary Significantly across Tasks*)**
Kami menyelidiki distribusi gabungan (*joint distribution*) pakar lintas tugas.

*Analisis Akademis Gambar 3 (Average number of shared Top-6 routed experts across tasks):*
*   **Deskripsi Visual:** Ini adalah *Heatmap* (Peta Panas) berupa matriks persegi. Angka di dalam kotak menunjukkan rata-rata jumlah pakar (dari total 6 pakar rute per lapisan) yang **bertumpang tindih (*overlap*)** antara dua tugas yang saling disilangkan.
*   **Sumbu X dan Y:** Merepresentasikan jenis-jenis tugas (Intent, Summary, Law, Translation, Math, Code).
*   **Warna dan Angka:** Kotak diagonal utama berwarna biru sangat gelap dengan angka mendekati 6 (misalnya, Math bersilangan dengan Math adalah 5.81). Kotak di luar diagonal (bersilangan antar tugas berbeda) berwarna biru sangat pudar hingga putih, dengan angka mendekati 0 (misalnya, Math bersilangan dengan Law hanya 0.58).
*   **Kesimpulan Akademis:** Nilai tinggi di diagonal membuktikan bahwa tugas yang sama secara konsisten memanggil himpunan pakar yang sama. Namun, nilai yang nyaris 0 di luar diagonal (*off-diagonal*) membuktikan bahwa **tugas yang berbeda menggunakan set pakar yang sepenuhnya berbeda secara radikal**. Kognisi spasial (Math) tidak menggunakan neuron yang sama dengan kemampuan linguistik (Translation).

### 3.3 Pelatihan Halus Spesialisasi-Pakar (*Expert-Specialized Fine-Tuning* / ESFT)

Sistem pakar yang sangat terspesialisasi ini menunjukkan bahwa pakar yang berbeda dapat dioptimalkan untuk tugas yang spesifik. Terinspirasi oleh hal ini, kami mengusulkan **ESFT** untuk penyesuaian (*customization*) LLM MoE.

ESFT bekerja dengan **secara selektif melatih (*fine-tuning*) hanya pakar yang paling relevan** untuk tugas-tugas turunan demi efisiensi komputasi dan mempertahankan spesialisasi pakar.

**Langkah-Langkah Metode ESFT:**

**1. Pengambilan Sampel Data (*Data Sampling*)**
Kami secara acak mengambil himpunan bagian ($D_s$) sampel data (input $x_i$, label $y_i$) dari kumpulan data pelatihan $D$. Secara empiris, ukuran himpunan bagian sebanyak 32 sampel gabungan, dengan panjang tetap $L = 4096$, sudah sangat tangguh (robust) untuk mendeteksi mana pakar yang relevan untuk suatu tugas.

**2. Skor Relevansi Pakar (*Expert Relevance Score*)**
Kami mengusulkan dua metode matematis untuk mengkalkulasi relevansi seorang pakar terhadap suatu tugas berdasarkan afinitasnya terhadap token-token sampel.

*   **Metode A: Average Gate Score (ESFT-Gate)**
    Metode ini mengkalkulasi rata-rata skor gerbang (*affinity/gate score*) dari pakar $e_i$ terhadap seluruh token di data sampel $D_s$. Formulasinya:

    $$ g_i^l = \frac{1}{N_s} \sum_{j=1}^{N_s} \frac{1}{L_j} \sum_{k=1}^{L_j} g_{i,k}^l, \tag{6} $$

    **Penjelasan Variabel Rumus 6:**
    *   $g_i^l$: Rata-rata skor gerbang untuk pakar $i$ di lapisan $l$.
    *   $N_s$: Total jumlah sampel urutan data.
    *   $L_j$: Panjang urutan (*length of input sequence*) $x_j$ pada sampel data.
    *   $g_{i,k}^l$: Skor gerbang absolut yang diberikan oleh *Router* kepada pakar $i$ untuk memproses token ke-$k$.
    *   **Logika:** Semakin tinggi rata-rata bobot (kepercayaan *Router* pada pakar tersebut) di seluruh teks, semakin "relevan" pakar itu.

*   **Metode B: Token Selection Ratio (ESFT-Token)**
    Skor ini mengkalkulasi rasio (persentase) token di mana pakar $e_i$ berhasil di- *select* (terpilih masuk ke dalam fungsi Top-K). Formulasinya:

    $$ r_i^l = \frac{1}{N_s} \sum_{j=1}^{N_s} \frac{1}{L_j} \sum_{k=1}^{L_j} \frac{\mathbb{1} (g_{i,k}^l > 0)}{K}, \tag{7} $$

    **Penjelasan Variabel Rumus 7:**
    *   $r_i^l$: Rasio pemilihan token untuk pakar $i$ di lapisan $l$.
    *   $\mathbb{1} (g_{i,k}^l > 0)$: Adalah **Fungsi Indikator**. Nilainya 1 jika skor gerbang positif (artinya pakar TERPILIH untuk memproses token tersebut), dan 0 jika tidak terpilih.
    *   $K$: Adalah kuota maksimum jumlah pakar yang boleh diaktifkan per token.
    *   **Logika:** Metode ini tidak memedulikan seberapa besar bobot gerbangnya, yang penting adalah: "Apakah pakar ini dipanggil bekerja atau tidak?". Jika ia sering dipanggil, ia sangat relevan.

**3. Seleksi Pakar dan *Fine-tuning***
Untuk setiap lapisan MoE ke-$l$, kami memilih subset pakar ($E_s^l$) untuk dilatih, berdasarkan skor relevansi di atas. Kami mendefinisikan batas ambang (threshold) $p \in (0, 1]$ sebagai hiperparameter yang mengontrol proporsi kumulatif dari total skor relevansi.

Untuk setiap lapisan $l$, kami menyeleksi urutan pakar dengan skor teratas, membentuk himpunan $E_s^l$, di mana jumlah kumulatif skor relevansinya melebih ambang batas $p$:

$$ \sum_{i \in E_s^l} R_i^l \geqslant p, \tag{8} $$

**Penjelasan Variabel Rumus 8:**
*   $R_i^l$: Adalah skor relevansi pakar $i$ di lapisan $l$ (bisa berupa $r_i^l$ atau $g_i^l$ sesuai metode yang dipilih).
*   $p$: Ambang batas persentase (misal $p = 0.2$, berarti kita terus mengambil pakar berperingkat teratas hingga total pangsa kerjanya menyentuh 20%).
*   **Proses Pelatihan (*Training*):** Selama inferensi, token bisa masuk ke pakar mana pun. **NAMUN selama pelatihan (*fine-tuning*)**, HANYA parameter matriks dari pakar-pakar yang masuk dalam himpunan seleksi $E_s^l$ ini saja yang diperbarui dengan *Gradient Descent*. Pakar lainnya beserta modul atensi akan dibekukan 100%.

---

## 4. Pengaturan Eksperimen (*Experiment Setup*)

### 4.1 Evaluasi Utama

Metode ESFT dievaluasi pada dua skenario penyesuaian LLM yang umum:
1.  Meningkatkan kemampuan spesifik pada **domain yang sudah dikuasai model secara layak** (Peningkatan Model / *Model Enhancement*).
2.  Mengadaptasi model ke **tugas terspesialisasi yang sempit dan asing** (Adaptasi Model / *Model Adaptation*).

**1. Tugas untuk Peningkatan Model (Math & Code):**
*   **Matematika (Math):** Dilatih menggunakan MetaMathQA dan dievaluasi pada **GSM8K** dan **MATH**.
*   **Kode (Code):** Dilatih pada subset Python dari evol-codealpaca dan dievaluasi pada **HumanEval** dan **MBPP**.

**2. Tugas untuk Adaptasi Model (Specialized Tasks):**
Diuji dengan *few-shot examples* dan diukur akurasinya menggunakan ekstraksi GPT-4 (kecuali JSON *exact match*). Tugasnya meliputi:
*   **Niat (*Intent Recognition*):** Mengubah instruksi teks menjadi format JSON (BDCI-21 Smart HCI NLU).
*   **Ringkasan (*Text Summarization*):** Meringkas transkrip layanan pelanggan.
*   **Hukum (*Legal Judgment Prediction*):** Memprediksi vonis hukum berdasarkan "deskripsi kasus".
*   **Terjemahan (*Low-resource Translation*):** Menerjemahkan bahasa minoritas Cherokee ke Bahasa Inggris (ChrEn dataset).

### 4.2 Evaluasi Kemampuan Umum (*General Ability*)

Sebuah PEFT tidak boleh merusak memori umum AI. Oleh karena itu, *benchmark* linguistik umum (MMLU, TriviaQA, HellaSwag, ARC, IFEval, CEval, CLUEWSC) dieksekusi pasca-pelatihan.

### 4.3 Model Backbone dan Pengaturan Pelatihan

*   **Model:** Arsitektur yang digunakan adalah **DeepSeek-V2-Lite** (menggunakan 66 pakar per lapisan *transformer*), menjadikannya kandidat optimal untuk spesialisasi pakar tingkat tinggi ini.
*   *Baseline* perbandingan: FFT (parameter penuh) dan LoRA.
*   *Hardware:* Klaster HFAI dengan NVIDIA A100 PCIe GPU.
*   Ambangan batas ESFT: $p = 0.1$ untuk ESFT-Gate, dan $p = 0.2$ untuk ESFT-Token.

---

## 5. Hasil (*Results*)

### 5.1 Hasil Kinerja Tolok Ukur (*Benchmark Performance Results*)

Silakan perhatikan rangkuman performa pada tabel makro (Tabel 1 dan Tabel 2, data referensinya diinterpretasikan secara akademis berikut ini).

*   **Evaluasi Kemampuan Kustomisasi (Sub-tugas Spesialis):**
    ESFT **mengungguli LoRA secara signifikan** dan sangat kompetitif dengan FFT yang mahal. ESFT-Token dan ESFT-Gate mencapai skor nyaris tertinggi pada tugas Peningkatan Model (Matematika dan Kode). Hebatnya, pada adaptasi tugas Spesialisasi Sempit (Intent, Hukum, Terjemahan), rata-rata metrik ESFT-Gate adalah **50.2**, yang bersaing ketat dengan FFT yang melatih miliaran parameter (**51.0**), dan membantai LoRA (**44.9**).
    *Kesimpulan Akademis:* Mengidentifikasi pakar spesifik jaringan saraf sangat efisien untuk adaptasi otak AI terhadap tugas yang sangat tajam (*narrow domains*).

*   **Evaluasi Kemampuan Umum (Pencegahan Amnesia):**
    ESFT secara konsisten menunjukkan penurunan kinerja bahasa umum yang **paling kecil** (lebih kebal terhadap *Catastrophic Forgetting*) dibandingkan dengan FFT dan LoRA. Rata-rata skor generalisasi ESFT-Token adalah **61.5**, menghancurkan FFT (**58.8**) dan LoRA (**59.1**). 
    *Kesimpulan Akademis:* Metode FFT merusak seluruh sinapsis jaringan, sementara ESFT menyelamatkan neuron-neuron "pengetahuan umum" dengan cara membekukannya secara fisik.

### 5.2 Hasil Efisiensi Komputasional (*Computational Efficiency Results*)

*Analisis Akademis Gambar 5 (Batang Pelatihan dan Garis Penyimpanan):*
*   **Waktu Pelatihan (*Training Time* - Batang Biru):** Rata-rata waktu untuk ESFT adalah sekitar 19,8 hingga 20,9 menit. Ini jauh lebih singkat daripada FFT (28,5 menit) dan berada di level kecepatan yang mirip dengan LoRA (16,5 menit).
*   **Kapasitas Penyimpanan (*Storage Space* - Garis Hijau):** Ini adalah keunggulan absolut. File bobot matriks pelatihan untuk ESFT hanya memakan ukuran **2,57 GB** (ESFT-Token) dan 3,20 GB (ESFT-Gate). Sebaliknya, FFT menuntut beban VRAM super raksasa sebesar **28,6 GB**. Meskipun LoRA sangat hemat ukuran, kualitas penalarannya (seperti dijelaskan di 5.1) kalah jauh dari ESFT.

---

## 6. Analisis Mendalam (*Analysis*)

Pada sub-bab kritis ini, kami mendemonstrasikan bagaimana ESFT secara mekanis memanfaatkan sistem pakarnya.

### 6.1 ESFT Memanfaatkan Pakar Spesialisasi Secara Efektif

*Analisis Akademis Gambar 4 (Number of experts trained in ESFT across layers and tasks):*
*   **Deskripsi Visual:** Ini adalah peta kotak yang merepresentasikan 26 lapisan arsitektur AI (sumbu X) dan beragam tugas spesifik (sumbu Y). Angka di dalam kotak menunjukkan jumlah pakar (dari total 66 pakar per lapisan) yang **terpilih untuk dilatih**.
*   **Kesimpulan:**
    1. Jumlah rata-rata pakar yang dilatih hanya berkisar antara **2 hingga 15 dari 66 pakar**, mengindikasikan penghematan parameter tertatih hingga 75%-95% lebih sedikit daripada FFT.
    2. Pada lapisan-lapisan tengah, jumlah pakar yang dilatih semakin sedikit (angka berwarna gelap), menunjukkan bahwa distribusi keahlian (*expert knowledge*) AI ternyata jauh lebih terpusat secara agresif di "lapisan tengah" otak jaringannya.

### 6.2 ESFT Memanfaatkan Sumber Daya Pelatihan Secara Lebih Efisien

*Analisis Akademis Gambar 6 (Perbandingan Tiga Metode Pengaturan Efisiensi):*
*   **Deskripsi Visual:** Grafik komparasi pada Domain Matematika. Sumbu X adalah rasio rata-rata jumlah pakar/peringkat (ukuran kapasitas parameter yang diizinkan untuk di-*training*). Sumbu Y adalah tingkat skor kemampuan spesialisasi (atas) dan kemampuan umum (bawah).
*   **Kesimpulan:** ESFT-Token (Garis Biru) dan ESFT-Gate (Garis Merah) mencapai puncak kecerdasan dan kemudian mendatar dengan laju yang sangat gesit. Grafik mengonfirmasi bahwa setelah nilai $p = 0.5$, peningkatan proporsi latihan tidak lagi signifikan. ESFT **selalu berada secara konsisten** di atas kurva performa LoRA pada tingkat pembatasan parameter (VRAM) mana pun.

### 6.3 Secara Selektif Melatih Parameter yang Tidak Dibagikan adalah Kunci ESFT

Sistem MoE memiliki *Shared Experts* (Pakar yang memproses setiap token secara wajib). Haruskah kita juga melepaskan pembekuan parameter pada *Shared Experts* ini saat melakukan ESFT?

Tabel 3 menguji ragam ablasi ini. Hasil yang konklusif dan terbukti adalah:
1.  Performa spesialisasi meningkat jika parameter spesifik yang dilatih diperbesar ukurannya.
2.  **Performa Kemampuan Umum (Kecerdasan General AI) AKAN MEROSOT secara linear terhadap besarnya jumlah *Shared Parameters* (parameter umum) yang dilatih ulang.** Jika AI dipaksa melatih ulang *shared experts*-nya untuk belajar Terjemahan Bahasa Cherokee, ia akan mengalami kebingungan tata bahasa umum.
3.  Oleh karena itu, aturan terkuat dari ESFT adalah: **Latih pakar spesialis yang paling relevan (Non-shared) dan BEKUKAN seluruh pakar umum (Shared).**

### 6.4 Analisis Modul Kunci di ESFT

Apakah algoritma penilaian ESFT kita berfungsi? Mengapa tidak memilih pakar secara acak (*random experts*)?

Tabel 4 membuktikan bahwa **mengganti pakar berafinitas tinggi (Pilihan ESFT) dengan pakar acak, menghancurkan performa model di seluruh tugas secara signifikan** (Rata-rata drop -2.8 hingga -4.4 poin absolut). Hal ini mengesahkan validitas saintifik dari metodologi pemilihan *Token Selection Ratio* (ESFT-Token) dan *Average Gate Score* (ESFT-Gate).

Lebih jauh, pada Gambar 7 dibuktikan bahwa **Granularitas (Ukuran Serpihan) Pakar itu sangat penting**. DeepSeek menggunakan segmentasi halus. Jika dalam pengujian kita memaksa pakar-pakar halus itu digabungkan (*grouped*) menjadi satu entitas bongkahan besar (seperti arsitektur LLaMA lama), performa metode ESFT merosot jauh lebih parah daripada metode FFT. ESFT adalah instrumen bedah mikroskopis yang efisiensinya mutlak bergantung pada seberapa rinci otak LLM dipecah-pecah ke dalam spesialisasi yang sempit.

---

## 7. Kesimpulan

Dalam karya monumental ini, kita telah menyelidiki secara komprehensif lanskap metode pelatihan-halus efisien parameter (*parameter-efficient fine-tuning* / PEFT) pada Model Bahasa Besar (LLMs) dengan arsitektur perutean renggang (*sparse / Mixture of Experts*). 

Kita pertama kali mengamati bahwa secara biologis (pada *neural network*), tugas-tugas dari domain linguistik yang berbeda didistribusikan pada **kombinasi pakar yang berbeda secara mutlak**. Menunggangi fenomena arsitektural ini, metode **Expert-Specialized Fine-Tuning (ESFT)** diusulkan, yang menggunakan metrik *average gate score* dan *token selection ratio* untuk menemukan "Pakar Spesifik" mana yang patut dilatih ulang. 

Hasil eksperimen yang presisi membuktikan bahwa metode ESFT sukses **memangkas beban biaya komputasional waktu pelatihan dan VRAM GPU secara brutal**, namun di saat yang sama mampu sejajar atau bahkan secara paradoks melampaui kemampuan model yang disiksa dengan pelatihan mahal parameter penuh (FFT), sekaligus melestarikan kewarasan logika *general ability* AI. Analisis mendalam juga menegaskan bahwa metode ini berhasil karena ia memperkuat sifat spesialisasi pakar laten di dalam jeroan arsitektur *Mixture of Experts*.

*(Demikian pemaparan detail mengenai inovasi ESFT. Mahasiswa dipersilakan merenungkan signifikansi algoritma ini dalam revolusi efisiensi komputasi AI).*
