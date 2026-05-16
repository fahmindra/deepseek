---
slug: ch-11-01
title: "DeepSeek LLM"
layout: chapter
---

## DeepSeek LLM: Menskalakan Model Bahasa Sumber Terbuka dengan Pendekatan Jangka Panjang (Longtermism)

Perkembangan pesat Model Bahasa Besar atau *Large Language Models* (LLMs) sumber terbuka (*open-source*) telah mencapai titik yang sungguh luar biasa dalam beberapa tahun terakhir. Di dalam literatur terdahulu, hukum penskalaan (*scaling laws*)—yakni prinsip matematis yang memandu bagaimana kinerja model meningkat seiring dengan penambahan ukuran model dan data—sering kali menyajikan kesimpulan yang bervariasi. Ketidakkonsistenan ini menciptakan "awan gelap" atau ketidakpastian dalam upaya menskalakan LLMs. 

Sebagai mahasiswa Kecerdasan Buatan, Anda harus memahami bahwa untuk menembus batas-batas kemampuan AI, kita harus mendalami studi tentang *scaling laws*. Bab ini akan membedah secara mendalam temuan empiris yang memfasilitasi penskalaan model dalam dua konfigurasi sumber terbuka yang sangat populer: 7B (7 Miliar parameter) dan 67B (67 Miliar parameter). Dengan dipandu oleh *scaling laws* ini, kita akan membahas arsitektur dan pengembangan **DeepSeek LLM**, sebuah proyek yang didedikasikan untuk memajukan LLM sumber terbuka dengan perspektif jangka panjang. 

Untuk mendukung fase Prapelatihan (*Pre-training*), DeepSeek menggunakan korpus yang terdiri dari 2 triliun *tokens* yang terus berkembang. Selain itu, kita akan mempelajari penerapan Pelatihan Halus Terselia (*Supervised Fine-Tuning* / SFT) dan Optimisasi Preferensi Langsung (*Direct Preference Optimization* / DPO) pada model dasar (*Base models*) untuk menciptakan model percakapan (*Chat models*). Hasil evaluasi ekstensif akan menunjukkan bahwa DeepSeek LLM 67B mampu melampaui arsitektur terkemuka seperti LLaMA-2 70B pada berbagai tolok ukur, khususnya di domain pemrograman (*code*), matematika, dan penalaran (*reasoning*).

---

## 1. Konteks dan Motivasi

Selama beberapa tahun terakhir, LLMs yang berbasis pada arsitektur Transformers *decoder-only* telah menjadi landasan utama dan jalur menuju *Artificial General Intelligence* (AGI). Dengan memprediksi kata berikutnya (*next-word prediction*) dalam teks yang kontinu, LLMs menjalani prapelatihan swaselia (*self-supervised pre-training*) pada kumpulan data masif. Hal ini memungkinkan model untuk menguasai berbagai kemampuan, seperti kreasi sastra, peringkasan teks, hingga penyelesaian kode program. Perkembangan lanjutan seperti *supervised fine-tuning* dan *reward modeling* memungkinkan LLMs untuk lebih baik dalam mengikuti niat dan instruksi manusia. Hal inilah yang membekali AI dengan kapabilitas percakapan yang lebih fleksibel.

Gelombang inovasi ini dipicu oleh produk-produk tertutup (*closed products*) seperti ChatGPT (OpenAI, 2022), Claude (Anthropic, 2023), dan Bard (Google, 2023). Produk-produk ini dikembangkan dengan sumber daya komputasi yang masif dan biaya anotasi yang fantastis. Tentu saja, hal ini meningkatkan ekspektasi komunitas terhadap LLMs sumber terbuka. Seri LLaMA muncul sebagai pelopor, mengonsolidasikan arsitektur efisien dan stabil, dari 7B hingga 70B parameter, dan menjadi tolok ukur (*de facto benchmark*) bagi model *open-source*.

Namun, komunitas sumber terbuka sering kali hanya berfokus pada pelatihan model berukuran tetap (seperti 7B, 13B, 34B, dan 70B) dengan kualitas tinggi, dan kerap **mengabaikan eksplorasi penelitian ke dalam *scaling laws*** LLM. Padahal, riset tentang *scaling laws* sangat krusial karena model saat ini masih berada pada tahap awal pengembangan AGI. Studi awal menunjukkan variasi kesimpulan terkait penskalaan model dan data ketika anggaran komputasi dinaikkan, dan sering kali kurang membahas penyesuaian hiperparameter (*hyperparameters*). 

Dalam bab ini, kita menginvestigasi perilaku penskalaan model bahasa dan mengaplikasikan temuan tersebut pada model 7B dan 67B. Secara spesifik, kita akan menguji *scaling laws* dari **ukuran *batch* (*batch size*)** dan **tingkat pembelajaran (*learning rate*)**, serta trennya terhadap ukuran model. Melalui studi komprehensif ini, kita berhasil mengungkap strategi alokasi optimal untuk penskalaan model/data. Temuan penting lainnya adalah: *scaling laws* yang diturunkan dari dataset yang berbeda menunjukkan perbedaan yang signifikan. Kualitas dataset sangat memengaruhi perilaku penskalaan.

Di bawah panduan *scaling laws* ini, model dibangun dari awal. Di tingkat arsitektur, pendekatan ini mengikuti LLaMA, tetapi mengganti penjadwal *learning rate* kosinus dengan penjadwal multi-langkah (*multi-step*) guna mempertahankan performa sekaligus memfasilitasi pelatihan berkelanjutan (*continual training*).

---

## 2. Prapelatihan (*Pre-Training*)

Tahap prapelatihan adalah fondasi dari LLMs. Di sinilah model menelan miliaran kata untuk mempelajari tata bahasa, fakta dasar, dan logika dunia.

### 2.1. Data

Objektif utama dari pengumpulan data adalah untuk secara komprehensif meningkatkan kekayaan dan keragaman dataset. Pendekatan ini diorganisasikan ke dalam tiga tahap esensial: **Deduplikasi (*deduplication*)**, **Penyaringan (*filtering*)**, dan **Pencampuran ulang (*remixing*)**. Deduplikasi dan pencampuran memastikan keragaman representasi, sedangkan penyaringan meningkatkan kepadatan informasi.

Strategi deduplikasi yang agresif diadopsi dengan memperluas cakupan deduplikasi. Analisis mengungkapkan bahwa mendeduplikasi seluruh hasil *Common Crawl* secara global menghasilkan penghapusan instansi ganda yang jauh lebih tinggi dibandingkan deduplikasi hanya pada satu bagian (*dump*) tunggal. 

Tabel berikut mengilustrasikan rasio deduplikasi di berbagai *dumps* Common Crawl:

| Dumps Used | 1 | 2 | 6 | 12 | 16 | 22 | 41 | 91 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Deduplication Rate (%)** | 22.2 | 46.7 | 55.7 | 69.9 | 75.7 | 76.3 | 81.6 | 89.8 |

*Tabel 1: Rasio deduplikasi untuk berbagai Common Crawl dumps.*
Dapat dilihat bahwa melakukan deduplikasi pada 91 *dumps* mampu menghilangkan empat kali lipat lebih banyak dokumen duplikat dibandingkan metode *dump* tunggal.

Pada tahap **penyaringan**, fokus diberikan pada pengembangan kriteria kuat untuk penilaian kualitas dokumen, yang mencakup evaluasi linguistik dan semantik. Pada fase **pencampuran ulang (*remixing*)**, pendekatan disesuaikan untuk mengatasi ketidakseimbangan data, berfokus pada peningkatkan kehadiran domain yang kurang terwakili (*underrepresented domains*).

Untuk sistem tokenisasi (*tokenizer*), diimplementasikan algoritma **Byte-level Byte-Pair Encoding (BBPE)**. Pra-tokenisasi digunakan untuk mencegah penggabungan token dari kategori karakter yang berbeda (seperti garis baru, tanda baca, dan simbol *Chinese-Japanese-Korean* / CJK). Angka-angka juga dipisah menjadi digit individual. Menggunakan pengalaman empiris, kosakata (*vocabulary*) konvensional ditetapkan sebanyak 100.000 token, dilatih pada korpus multibahasa berukuran ~24 GB. Kosakata ini kemudian ditambah dengan 15 token khusus, sehingga berukuran 100.015. Untuk efisiensi komputasi *hardware* (terutama GPU) dan cadangan masa depan, ukuran *vocabulary* model akhirnya di-*padding* menjadi **102.400**.

### 2.2. Arsitektur

Secara mikro, desain DeepSeek LLM sangat mengikuti desain LLaMA. Arsitektur ini mengadopsi struktur **Pre-Norm** menggunakan fungsi **RMSNorm** (*Root Mean Square Normalization*), serta menggunakan fungsi aktivasi **SwiGLU** untuk *Feed-Forward Network* (FFN) dengan dimensi lapisan menengah sebesar $\frac{8}{3} d_{model}$. Model ini juga menginkorporasikan **Rotary Embedding (RoPE)** untuk pengodean posisional. Untuk menekan biaya inferensi, model 67B menggunakan **Grouped-Query Attention (GQA)** alih-alih *Multi-Head Attention* (MHA) tradisional.

Tabel di bawah ini memaparkan spesifikasi teknis model:

| Params | $n_{layers}$ | $d_{model}$ | $n_{heads}$ | $n_{kv\_heads}$ | Context Length | Sequence Batch Size | Learning Rate | Tokens |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **7B** | 30 | 4096 | 32 | 32 | 4096 | 2304 | 4.2e-4 | 2.0T |
| **67B** | 95 | 8192 | 64 | 8 | 4096 | 4608 | 3.2e-4 | 2.0T |

*Tabel 2: Spesifikasi detail keluarga model DeepSeek LLM.* 

Hal yang esensial untuk diperhatikan secara makro adalah perbedaan jumlah lapisan jaringan ($n_{layers}$). Model 7B memiliki 30 lapisan, sedangkan 67B memiliki 95 lapisan. Mengapa demikian? Berbeda dengan mayoritas karya lain yang menggunakan GQA di mana mereka biasanya memperlebar FFN untuk menambah jumlah parameter, arsitektur ini **memperluas parameter model 67B pada kedalaman jaringannya (*network depth*)**. Penambahan lapisan ini dirancang untuk mencapai performa yang lebih superior sekaligus memfasilitasi partisi *pipeline* model agar pelatihan dan inferensi dapat dioptimalkan.

### 2.3. Hiperparameter (*Hyperparameters*)

Bobot DeepSeek LLM diinisialisasi dengan standar deviasi sebesar 0.006 dan dilatih menggunakan pengoptimal **AdamW**, dengan hiperparameter berikut: $\beta_1 = 0.9$, $\beta_2 = 0.95$, dan *weight decay* (penurunan bobot) = $0.1$. Pemotongan gradien (*gradient clipping*) diatur pada nilai $1.0$.

Satu inovasi krusial di sini adalah **penggunaan penjadwal *learning rate* multi-langkah (*multi-step learning rate scheduler*)**, bukan penjadwal kosinus (*cosine scheduler*) yang biasa dipakai. Secara spesifik: tingkat pembelajaran naik hingga maksimum setelah 2000 *warmup steps*, kemudian secara perlahan menurun (*decay*) menjadi 31,6% dari nilai maksimalnya setelah memproses 80% dari total *tokens* pelatihan. Tingkat ini akan diturunkan lagi hingga 10% dari nilai maksimal setelah 90% *tokens* diproses.

Untuk lebih memahami dampak penjadwal *learning rate* ini, mari kita analisis **Gambar 1**.

**Deskripsi Gambar 1:** 
Gambar ini menyajikan grafik garis yang mengilustrasikan kurva kerugian pelatihan (*Training Loss curves*) pada sebuah model kecil (1,6 Miliar parameter yang dilatih dengan 100 Miliar token).
*   **Sumbu-X (Horizontal):** Mempresentasikan *Processed Tokens* (Miliar), yaitu seberapa banyak data yang telah dilihat oleh model dari 0 hingga 100 Miliar.
*   **Sumbu-Y (Vertikal):** Mempresentasikan *Training Loss*, yaitu tingkat kesalahan model dalam memprediksi kata berikutnya, bernilai dari sekitar 2.2 hingga 3.0 (semakin rendah semakin baik).
*   **Gambar 1 (a) Multi-step vs. cosine learning rate decay:** Terdapat dua kurva. Kurva jingga merepresentasikan metode *cosine scheduler*, sedangkan kurva biru merepresentasikan *multi-step scheduler* dengan proporsi (80% + 10% + 10%). Terlihat bahwa meskipun pada pertengahan pelatihan kurva biru (multi-step) memiliki *loss* yang sedikit lebih tinggi dibanding kurva jingga (cosine), pada akhir masa pelatihan (mendekati 100 Miliar token), *loss* dari kurva biru terjun tajam dan akhirnya setara atau identik dengan hasil akhir dari kurva jingga.
*   **Gambar 1 (b) Different proportions of multi-step stages:** Terdapat tiga kurva biru (80+10+10), jingga (70+15+15), dan hijau (60+20+20). Ketiganya menunjukkan penurunan *loss* tajam sesuai dengan titik *decay* masing-masing. Kurva hijau (60+20+20) memberikan performa yang sedikit lebih baik di akhir, namun untuk menjaga keseimbangan bagi kemampuan meneruskan pelatihan di masa mendatang (*continual training*), proporsi 80%+10%+10% (kurva biru) dipilih sebagai *default*.

**Kesimpulan Akademik dari Gambar 1:** Meskipun terdapat sedikit perbedaan tren kerugian di tengah jalan, performa final *multi-step scheduler* pada dasarnya konsisten dengan *cosine scheduler*. Keuntungan absolut dari *multi-step scheduler* adalah memungkinkan penggunaan ulang (*reuse*) fase pertama untuk eksperimen *continual training* saat menskalakan komputasi tanpa mengubah arsitektur.

### 2.4. Infrastruktur (*Infrastructures*)

Pelatihan model besar tidak akan mungkin terjadi tanpa *framework* komputasi terdistribusi. Digunakan *framework* berbobot ringan yang sangat efisien bernama **HAI-LLM**. *Framework* ini mengintegrasikan berbagai paralelisme (*parallelisms*): *Data parallelism, tensor parallelism, sequence parallelism*, dan *1F1B pipeline parallelism*.

Teknik **FlashAttention-2** dimanfaatkan untuk memaksimalkan utilisasi *hardware*. Algoritma **ZeRO-1** digunakan untuk mempartisi status pengoptimal (*optimizer states*) ke berbagai *data parallel ranks*. Upaya juga dikerahkan untuk *overlapping* (tumpang tindih) antara komputasi dan komunikasi data (jaringan) agar *waiting overhead* menjadi minimal. 

Untuk stabilitas pelatihan, model dilatih menggunakan presisi **bf16** (*bfloat16*), tetapi akumulasi gradien tetap dihitung dalam presisi **fp32** (*float32*). *In-place cross-entropy* diterapkan untuk menghemat memori GPU secara drastis: konversi *bf16 logits* menjadi *fp32* tidak dilakukan terlebih dahulu di *High Bandwidth Memory* (HBM), melainkan dikonversi dan dikalkulasi secara *on the fly* di dalam kernel CUDA untuk menimpa memori asal.

Pembuatan pos pemeriksaan (*checkpointing*) bobot model dan *optimizer states* disimpan setiap 5 menit secara asinkron. Evaluasi (*generation tasks*) memanfaatkan pustaka **vLLM**, sedangkan *non-generative tasks* menggunakan teknik *continuous batching* untuk mengurangi pemborosan komputasi (*token padding*).

---

## 3. Hukum Penskalaan (*Scaling Laws*)

Riset tentang *scaling laws* sudah ada sebelum era LLM modern. Secara klasik, *scaling laws* menyatakan bahwa performa model dapat diprediksi secara presisi dengan peningkatan tiga komponen utama: 
1.  **Anggaran Komputasi / *Compute budget* ($C$)**
2.  **Skala Model / *Model scale* ($N$)**
3.  **Skala Data / *Data scale* ($D$)**

Apabila skala model ($N$) direpresentasikan oleh jumlah parameter model, dan skala data ($D$) diwakili oleh jumlah *tokens*, maka Komputasi ($C$) dapat diaproksimasi sebagai $C \approx 6ND$. Pertanyaan riset paling krusial bagi peneliti AI adalah: **Bagaimana cara mengoptimalkan alokasi antara skala model dan skala data saat kita secara konstan meningkatkan anggaran komputasi $C$?**

Literatur sebelumnya (misalnya Kaplan et al. dari OpenAI, dan Hoffmann et al. dengan model Chinchilla dari DeepMind) menyajikan kesimpulan yang sangat berbeda mengenai formula alokasi yang optimal, menimbulkan keraguan akan generalisasi dari hukum ini. Apalagi, studi masa lalu jarang merinci hiperparameter (seperti *batch size* dan *learning rate*), meninggalkan pertanyaan apakah mereka menguji model-model tersebut pada konfigurasi optimal.

Untuk memastikannya, riset ini meninjau kembali *scaling laws* dari hiperparameter terlebih dahulu.

### 3.1. Hukum Penskalaan untuk Hiperparameter

Secara empiris diamati bahwa sebagian besar hiperparameter (seperti $\beta$ pada AdamW, *weight decay*, dl.) tidak memerlukan perubahan saat memperbesar anggaran komputasi. Namun, dua variabel yang memiliki dampak paling masif, yakni **Ukuran Batch (*Batch Size*)** dan **Tingkat Pembelajaran (*Learning Rate*)**, harus dianalisis ulang.

Langkah pertama adalah melakukan optimisasi *Grid Search* secara ekstensif pada komputasi skala kecil (anggaran komputasi $1\times10^{17}$ FLOPs) untuk berbagai rentang *batch size* dan *learning rate*.

**Deskripsi Gambar 2:**
Gambar 2 mengilustrasikan diagram kontur *heatmap* (peta panas) dari *Training loss* sehubungan dengan ukuran batch dan learning rate.
*   **Sumbu-X (Horizontal):** Menyatakan *Learning Rate*, dari skala kecil ($2^{-10.5}$) hingga skala besar ($2^{-7.75}$ atau $2^{-8.5}$).
*   **Sumbu-Y (Vertikal):** Menyatakan *Batch Size (Tokens)* dalam logaritma basis 2, contohnya $2^{15}$ hingga $2^{23.5}$ token.
*   **Warna dalam sel (Heatmap):** Warna hijau hingga kuning merepresentasikan kerugian (*loss*) terendah (terbaik), sedangkan biru pekat merepresentasikan kerugian tertinggi (terburuk). Angka di dalam kotak adalah nilai *loss* konkret.
*   **Gambar 2 (a) 1e17 FLOPs (Model 177M FLOPs/token):** Menunjukkan ruang parameter (hijau) cukup luas di mana *generalization error* (kesalahan prediksi) tetap stabil meskipun *batch size* dan *learning rate* diubah-ubah dalam zona tersebut.
*   **Gambar 2 (b) 1e20 FLOPs (Model 2.94B FLOPs/token):** Model berskala lebih besar. Sama halnya, area hijau juga memanjang membentuk sabuk miring, menandakan ruang yang lebar untuk parameter *near-optimal*. Bintang merah di tengah grafik melambangkan titik hiperparameter terpas (fitted) yang tepat berada di area paling optimal.

**Kesimpulan dari Gambar 2:** Performa *near-optimal* dapat dicapai dalam ruang parameter yang relatif luas. 

Selanjutnya, menggunakan pendekatan *multi-step scheduler*, banyak eksperimen dilakukan. Mengingat redundansi di dalam ruang parameter, nilai yang menghasilkan kesalahan generalisasi tidak melebihi 0,25% dari *loss minimum* dianggap sebagai parameter *near-optimal*. Berdasarkan hal ini, data yang diperoleh diformulasikan ke dalam hukum pangkat (*power law*) terhadap anggaran komputasi $C$.

Rumus empiris final untuk *Batch Size* Optimal ($B_{opt}$) dan *Learning Rate* Optimal ($\eta_{opt}$) adalah:

$$ \eta_{opt} = 0.3118 \cdot C^{-0.1250} $$
$$ B_{opt} = 0.2920 \cdot C^{0.3271} $$

**Penjelasan Variabel Rumus (1):**
*   $C$: Anggaran Komputasi (*Compute Budget*) dalam hitungan FLOPs.
*   $\eta_{opt}$: *Learning Rate* (tingkat pembelajaran) yang paling optimal. Korelasinya negatif pangkat $-0.1250$, yang secara logika matematika berarti: **semakin besar anggaran komputasi (model dan data membesar), maka *learning rate* optimal harus dibuat semakin kecil**.
*   $B_{opt}$: *Batch Size* optimal dalam jumlah token. Korelasinya positif pangkat $0.3271$, yang berarti: **semakin besar anggaran komputasi, ukuran batch optimal secara bertahap harus ditingkatkan**.

**Deskripsi Gambar 3:**
Gambar ini menunjukkan Kurva Penskalaan (*Scaling Curves*) dari Batch Size dan Learning Rate.
*   **Sumbu-X:** Adalah Anggaran Komputasi *Non-Embedding* FLOPs secara logaritmik ($10^{16}$ hingga $10^{24}$).
*   **Lingkaran Abu-abu:** Merepresentasikan data dari eksperimen-eksperimen ukuran kecil yang error-nya kurang dari 0.25% batas minimum.
*   **Garis Dotted (Putih-Abu):** Merepresentasikan kurva regresi hukum pangkat (*power law*) yang ditarik (difit) dari titik-titik kecil tersebut untuk memprediksi arah bagi komputasi yang jauh lebih besar.
*   **Bintang Biru:** Mempresentasikan model aktual DeepSeek LLM 7B dan 67B.
*   **Gambar 3 (a) Batch size scaling curve:** Sumbu-Y adalah Optimal Batch Size. Menunjukkan bahwa seiring membesarnya komputasi (ke kanan), batch size harus dinaikkan. Titik bintang biru (DeepSeek 7B dan 67B) secara sempurna berada tepat di lintasan garis *power law* hasil prediksi model kecil.
*   **Gambar 3 (b) Learning rate scaling curve:** Sumbu-Y adalah Optimal Learning Rate. Menunjukkan kurva menurun di mana model semakin besar (ke kanan) memerlukan learning rate yang lebih kecil. Bintang biru juga berada di atas garis *power law*.

Penting untuk dicatat bahwa *optimal batch size* juga bergantung pada faktor alokasi rasio model/data, bukan hanya pada generalisasi *loss* semata. Topik dinamika pelatihan ini memerlukan kajian lebih lanjut.

### 3.2. Mengestimasi Penskalaan Model dan Data Optimal

Setelah menurunkan rumus hiperparameter, tahap berikutnya adalah mencari strategi alokasi optimal model/data. Secara konseptual, kita ingin mencari pangkat penskalaan model ($a$) dan pangkat penskalaan data ($b$) sedemikian sehingga ukuran model optimal berbanding lurus $N_{opt} \propto C^a$ dan data optimal $D_{opt} \propto C^b$.

Pada literatur sebelumnya, komputasi (FLOPs) dihitung dengan mengalikan parameter dan token dengan angka pendekatan 6. 
*   Pendekatan OpenAI menggunakan parameter non-embedding ($N_1$).
*   Pendekatan DeepMind/Chinchilla menggunakan parameter lengkap dengan kosakata ($N_2$).

Masalahnya, rumus $6N_1$ tidak menghitung beban komputasional dari operasi mekanisme *Attention*, sedangkan rumus $6N_2$ menghitung komputasi layer keluaran (kosakata / *vocabulary*) yang nyatanya tidak banyak menambah kapasitas intelektual dari model bahasa. Oleh karena itu, kedua pendekatan *lama* ini memberikan **kesalahan aproksimasi (pembulatan)** yang sangat signifikan, khususnya pada model berukuran kecil.

Maka, untuk keakuratan saintifik, diperkenalkan representasi skala model baru: **Non-embedding FLOPs/token ($M$)**. Metrik $M$ ini secara akurat memperhitungkan operasi aritmatika komputasi *Attention*, namun secara cerdas mengabaikan kalkulasi *vocabulary*.

Perbandingan rumus 6$N_1$, 6$N_2$, dan formula $M$ yang baru dijabarkan secara detail di bawah ini:

$$ 6N_1 = 72 \cdot n_{layer} \cdot d_{model}^2 $$
$$ 6N_2 = 72 \cdot n_{layer} \cdot d_{model}^2 + 6 \cdot n_{vocab} \cdot d_{model} $$
$$ M = 72 \cdot n_{layer} \cdot d_{model}^2 + 12 \cdot n_{layer} \cdot d_{model} \cdot l_{seq} $$

**Penjelasan Variabel Rumus (2):**
*   $n_{layer}$: Jumlah lapisan (layer) Transformer dalam jaringan.
*   $d_{model}$: Dimensi representasi vektor/lebar dimensi model (*hidden size*).
*   $n_{vocab}$: Ukuran daftar kosakata (*vocabulary size*).
*   $l_{seq}$: Panjang runtutan data/konteks maksimum per *input* (*sequence length*).

**Tabel 3** memvalidasi perbedaan metrik-metrik tersebut:

| $n_{layers}$ | $d_{model}$ | $n_{vocab}$ | $l_{seq}$ | $N_1$ (Non-emb Params) | $N_2$ (Total Params) | $M$ (Non-emb FLOPs) | $\frac{6N_1}{M}$ | $\frac{6N_2}{M}$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 8 | 512 | 102400 | 4096 | 25.2M | 77.6M | 352M | 0.43 | 1.32 |
| 12 | 768 | 102400 | 4096 | 84.9M | 164M | 963M | 0.53 | 1.02 |
| 24 | 1024 | 102400 | 4096 | 302M | 407M | 3.02B | 0.60 | 0.81 |
| 24 | 2048 | 102400 | 4096 | 1.21B | 1.42B | 9.66B | 0.75 | 0.88 |
| 32 | 4096 | 102400 | 4096 | 6.44B | 6.86B | 45.1B | 0.85 | 0.91 |
| 40 | 5120 | 102400 | 4096 | 12.6B | 13.1B | 85.6B | 0.88 | 0.92 |
| 80 | 8192 | 102400 | 4096 | 64.4B | 65.3B | 419B | 0.92 | 0.94 |

*Tabel 3: Perbedaan representasi skala model. Menggunakan $6N_1$ terlalu kecil (rasio < 1), sedangkan $6N_2$ bisa sangat bengkak pada model kecil (rasio 1.32 untuk 8 layer).*

Setelah mengadopsi $M$ sebagai metrik ukuran model baru, objektif penelitian diformulasikan ulang secara matematis: 
> "Diberikan anggaran komputasi $C = M \cdot D$, temukan skala model optimal $M_{opt}$ dan skala data optimal $D_{opt}$ yang meminimalkan kerugian generalisasi (Loss) dari model."

$$ M_{opt}(C), D_{opt}(C) = \text{argmin}_{M, D \text{ s.t. } C=MD} \ L(M, D) $$

**Penjelasan Variabel Rumus (3):**
*   $\text{argmin}$: Fungsi matematis untuk menemukan nilai argumen masukan (dalam hal ini $M$ dan $D$) yang membuat fungsi biaya $L$ mencapai titik minimumnya.
*   $\text{s.t. } C = MD$: Kondisi *Subject To* (tunduk pada), yang berarti total biaya komputasi tetap harus sama dengan jumlah FLOPs/token ($M$) dikali jumlah data/token ($D$).
*   $L(M, D)$: Fungsi Kerugian (*Loss function*) dari model yang bergantung pada ukuran model dan data.

Pendekatan *IsoFLOP profile* digunakan untuk menekan biaya eksperimental.

**Deskripsi Gambar 4:** 
Gambar ini menjabarkan Kurva IsoFLOP dan Alokasi Model/Data Optimal. Metrik utamanya adalah bit-per-byte pada *validation set* (mirip *loss*, semakin rendah semakin baik).
*   **Gambar 4 (a) IsoFLOP curve:** Sumbu-X adalah *Non-Embedding FLOPs/Token (M)*. Sumbu-Y adalah kinerja *Bits-per-Byte*. Terdapat banyak titik dengan berbagai warna (masing-masing warna merepresentasikan satu level komputasi $C$ mulai dari 1e17 hingga 3e20). Tiap kumpulan warna membentuk formasi "U" atau parabola melengkung. Titik lembah atau titik terendah dari kurva parabola tersebut adalah titik paling optimal bagi komputasi tersebut.
*   **Gambar 4 (b) Optimal model scaling:** Sumbu-X adalah Training FLOPs ($C$). Sumbu-Y adalah FLOPs/token model ($M$). Lingkaran abu-abu memplot titik-titik "lembah" dari grafik 4(a). Garis putus-putus (*dotted*) biru mem-fitting titik abu-abu. Area kotak biru dengan bintang menunjuk proyeksi ke arah model masa depan (seperti DeepSeek 67B). Menunjukkan bahwa penambahan komputasi harus secara teratur memperbesar *size* model.
*   **Gambar 4 (c) Optimal data scaling:** Sumbu-X adalah Training FLOPs ($C$). Sumbu-Y adalah Jumlah Tokens data ($D$). Memiliki tren logaritmik linear yang serupa, di mana penambahan dana komputasi juga harus digunakan untuk memperbanyak token pelatihan secara simetris.

Berdasarkan *fitting* pada titik lembah (titik optimal) di kurva IsoFLOP, rumus matematis final untuk skala optimal diturunkan:

$$ M_{opt} = M_{base} \cdot C^a, \quad M_{base} = 0.1715, \quad a = 0.5243 $$
$$ D_{opt} = D_{base} \cdot C^b, \quad D_{base} = 5.8316, \quad b = 0.4757 $$

**Penjelasan Variabel Rumus (4):**
*   $M_{opt}$ dan $D_{opt}$: Besaran model dan data yang paling optimal.
*   $M_{base}$ dan $D_{base}$: Konstanta matematis pergeseran (intersep) dari regresi model.
*   $a = 0.5243$: Pangkat (*exponent*) proporsi untuk model.
*   $b = 0.4757$: Pangkat (*exponent*) proporsi untuk data.
*   Penting: Perhatikan bahwa nilai $a$ (0.5243) lebih besar daripada $b$ (0.4757). Ini mematahkan paradigma lama. Artinya, seiring Anda memperbesar komputasi, Anda harus mengalokasikan **sedikit lebih banyak uang/komputasi untuk menambah *size parameter model* ketimbang menjejalkan lebih banyak data**.

**Deskripsi Gambar 5:** 
Berjudul *Performance scaling curve*, menampilkan titik komputasi (Sumbu X) vs *Loss* Bits-per-Byte (Sumbu Y). Titik abu-abu adalah uji skala kecil. Garis *dotted* adalah perpanjangan matematis. Bintang biru melambangkan hasil riil pelatihan DeepSeek 7B (bintang kiri) dan 67B (bintang kanan). Bintang biru ini jatuh dengan sangat mulus dan *perfect* bersinggungan di atas garis putus-putus. Ini membuktikan secara empiris bahwa hukum skala dari model kecil ini sangat akurat memprediksi kecerdasan model masa depan (model besar), sehingga penelitian ini valid memprediksi komputasi yang 1000x lebih raksasa.

### 3.3. Hukum Penskalaan dengan Dataset Berbeda (*Scaling Laws with Different Data*)

Dalam prosesnya, kualitas data terus-menerus ditingkatkan. Observasi yang sangat menarik ditemukan ketika menganalisis ulang konstanta $a$ dan $b$ menggunakan dataset yang berbeda secara kualitas.

| Approach | Coeff. $a$ where $N_{opt}(M_{opt}) \propto C^a$ | Coeff. $b$ where $D_{opt} \propto C^b$ |
| :--- | :---: | :---: |
| OpenAI (OpenWebText2) | 0.73 | 0.27 |
| Chinchilla (MassiveText) | 0.49 | 0.51 |
| **Ours (Early Data)** | **0.450** | **0.550** |
| **Ours (Current Data)** | **0.524** | **0.476** |
| **Ours (OpenWebText2)** | **0.578** | **0.422** |

*Tabel 4: Koefisien skala model dan skala data bervariasi bergantung pada distribusi dan kualitas data.*

Berdasarkan Tabel 4 di atas, ketika kualitas data ditingkatkan (dari *Early Data* ke *Current Data*, hingga ke *OpenWebText2* yang di-kurasi dengan ketat), koefisien $a$ merangkak naik (dari 0.450 menjadi 0.578), sementara koefisien $b$ mengecil.
**Kesimpulan Akademis Krusial:** Kualitas data (*data quality*) sangat mendikte perilaku hukum skala. **Semakin tinggi kualitas data, semakin besar peningkatan anggaran komputasi harus dialokasikan ke parameter model (bukan jumlah data).** Intuisi di balik hal ini adalah: data berkualitas tinggi memiliki kejelasan logis (*logical clarity*) dan kesulitan prediksi yang lebih rendah setelah dilatih. Jadi, daripada memaksa model terus membaca data bersih yang sama, lebih menguntungkan secara inteligensia jika "otak" model (*size* model) tersebut diperbesar.

---

## 4. Penyelarasan (*Alignment*)

Penyelarasan atau *Alignment* adalah tahap di mana model *mentah* (*Base model*) diajari norma dan format interaksi agar dapat menjadi asisten percakapan (seperti *Chatbot*) yang *Helpful* (berguna) dan *Harmless* (tidak berbahaya/aman).

DeepSeek mengumpulkan sekitar 1,5 juta instansi data instruksi dalam bahasa Inggris dan Mandarin. Distribusi datanya:
*   Data berguna (1,2 Juta): Tugas bahasa umum (31,2%), masalah matematis (46,6%), dan penyelesaian kode program (22,2%).
*   Data aman (300 Ribu): Topik yang sensitif untuk memastikan *harmlessness*.

Saluran pipa (*pipeline*) penyelarasan melibatkan dua langkah:

**1. Pelatihan Halus Terselia (*Supervised Fine-Tuning* / SFT):** 
Model 7B di-*fine-tune* selama 4 epoch, namun 67B hanya selama 2 epoch akibat isu *overfitting* yang cepat pada model besar.
Tingkat pembelajaran diatur pada 1e-5 (7B) dan 5e-6 (67B). Mahasiswa harus mengamati fenomena paradoks yang dialami: Seiring ditambahkannya porsi data matematika di tahap SFT, kemampuan *benchmark* meningkat, namun rasio **repetisi tak berujung (*endless repetition*)** dari jawaban chatbot juga ikut melonjak tajam (sering kali terjebak dalam *loop* teks). Hal ini dikarenakan data *math SFT* memiliki pola penalaran yang persis sama dan repetitif. Solusinya: model di-*fine-tune* dengan skema dua tahap (mengeksklusi data matematika pada tahap akhir) dan algoritma DPO.

**2. Optimisasi Preferensi Langsung (*Direct Preference Optimization* / DPO):**
Menggantikan algoritma lawas yang rumit *Reinforcement Learning from Human Feedback* (RLHF), DPO memformulasikan ulang masalah *reward modeling* dan *policy optimization* menjadi satu formulasi probabilitas sederhana. DPO melatih model dari data "preferensi" (jawaban A lebih baik dari B, atau jawaban A aman dan jawaban B toksik). Model dilatih 1 epoch, learning rate 5e-6, *batch size* 512, menggunakan penjadwal *cosine*. DPO secara terbukti melambungkan kemampuan bercakap-cakap tak terbatas (*open-ended generation skill*), serta mengurangi repetisi, walau dampaknya pada pengujian *benchmark* akademis standar tidak terlalu bergeser secara signifikan.

---

## 5. Evaluasi (*Evaluation*)

### 5.1. Evaluasi Tolok Ukur Publik (*Public Benchmark Evaluation*)

Model dievaluasi di puluhan *dataset* berbasis bahasa Inggris dan Mandarin. Sebagai pengantar akademis, format pengujian terbagi menjadi:
*   **Perplexity-based evaluation:** Model harus "memilih" opsi (A/B/C/D) dengan mengkalkulasi probabilitas (*perplexity*) dari setiap opsi, dan opsi ber-probabilitas tertinggi dianggap sebagai jawaban model (Digunakan pada MMLU, ARC, C-Eval).
*   **Generation-based evaluation:** Model diminta menghasilkan baris teks baru secara rakus (*greedy decoding*), lalu sistem secara otomatis melakukan ekstraksi jawaban teks tersebut untuk mencocokkan dengan kunci (Digunakan pada GSM8K, MATH, HumanEval).

#### 5.1.1. Model Dasar (*Base Model*)

**Tabel 5** menampilkan hasil utama pada *benchmark* evaluasi:

| Language | Benchmark | Test-shots | LLaMA2 7B | DeepSeek 7B | LLaMA2 70B | DeepSeek 67B |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **English** | HellaSwag | 0-shot | 75.6 | 75.4 | **84.0** | **84.0** |
| | PIQA | 0-shot | 78.0 | 79.2 | 82.0 | **83.6** |
| | WinoGrande | 0-shot | 69.6 | 70.5 | **80.4** | 79.8 |
| | RACE-Middle | 5-shot | 60.7 | 63.2 | **70.1** | 69.9 |
| | RACE-High | 5-shot | 45.8 | 46.5 | **54.3** | 50.7 |
| | TriviaQA | 5-shot | 63.8 | 59.7 | **79.5** | 78.9 |
| | NaturalQuestions | 5-shot | 25.5 | 22.2 | 36.1 | **36.6** |
| | MMLU | 5-shot | 45.8 | 48.2 | 69.0 | **71.3** |
| | ARC-Easy | 0-shot | 69.1 | 67.9 | 76.5 | **76.9** |
| | ARC-Challenge | 0-shot | 49.0 | 48.1 | **59.5** | 59.0 |
| | OpenBookQA | 0-shot | 57.4 | 55.8 | **60.4** | 60.2 |
| | DROP | 1-shot | 39.8 | 41.0 | **69.2** | 67.9 |
| | MATH | 4-shot | 2.5 | 6.0 | 13.5 | **18.7** |
| | GSM8K | 8-shot | 15.5 | 17.4 | 58.4 | **63.4** |
| | HumanEval | 0-shot | 14.6 | 26.2 | 28.7 | **42.7** |
| | MBPP | 3-shot | 21.8 | 39.0 | 45.6 | **57.4** |
| | BBH | 3-shot | 38.5 | 39.5 | 62.9 | **68.7** |
| | AGIEval | 0-shot | 22.8 | 26.4 | 37.2 | **41.3** |
| | Pile-test | - | 0.741 | 0.725 | 0.649 | **0.642** |
| **Chinese** | CLUEWSC | 5-shot | 64.0 | 73.1 | 76.5 | **81.0** |
| | CHID | 0-shot | 37.9 | 89.3 | 55.5 | **92.1** |
| | C-Eval | 5-shot | 33.9 | 45.0 | 51.4 | **66.1** |
| | CMMLU | 5-shot | 32.6 | 47.2 | 53.1 | **70.8** |
| | CMath | 3-shot | 25.1 | 34.5 | 53.9 | **63.0** |
| | C3 | 0-shot | 47.4 | 65.4 | 61.7 | **75.3** |
| | CCPM | 0-shot | 60.7 | 76.9 | 66.2 | **88.5** |

*Tabel 5: Hasil Evaluasi Utama (Angka tebal adalah performa terbaik). LLaMA2 difokuskan mayoritas Inggris, sementara DeepSeek pada dwibahasa (Inggris & Mandarin).*

**Analisis Akademik:** 
Meskipun dilatih pada dua bahasa, performa bahasa Inggris DeepSeek (7B maupun 67B) terbukti sebanding dengan LLaMA2 (yang mendedikasikan 2T Token-nya khusus bahasa Inggris). Khususnya, DeepSeek 67B mendominasi LLaMA2 70B dengan margin sangat jauh pada ranah matematika (MATH, GSM8K), penyelesaian kode (HumanEval, MBPP), dan penalaran kritis (BBH).
Menariknya, keuntungan margin pada level model skala 67B jauh lebih drastis ketimbang margin keuntungan pada level 7B. Ini menunjukkan **pengaruh "konflik bahasa" (*language conflict*) yang jauh lebih mengusik otak model berukuran kecil** yang tidak punya kapasitas penyimpanan pengetahuan raksasa. Selain itu, walau LLaMA2 tak diajari bahasa Mandarin secara khusus, ia rupanya lumayan baik pada *CMath*, mengindikasikan bahwa kemampuan fundamental seperti penalaran matematis ternyata dapat **secara ajaib ditransfer lintas bahasa (*cross-lingual transfer*)**.

#### 5.1.2. Model Obrolan (*Chat Model*)

**Tabel 6** merepresentasikan performa antara model Base (tanpa Alignment) dengan Chat (Setelah SFT dan DPO):

| Language | Benchmark | DeepSeek 7B Base | DeepSeek 7B Chat | DeepSeek 67B Base | DeepSeek 67B Chat |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **English** | HellaSwag | 75.4 | 68.5 | **84.0** | 75.7 |
| | PIQA | 79.2 | 77.6 | **83.6** | 82.6 |
| | WinoGrande | 70.5 | 66.9 | **79.8** | 76.0 |
| | RACE-Middle | 63.2 | 65.2 | 69.9 | **70.9** |
| | RACE-High | 46.5 | 50.8 | 50.7 | **56.0** |
| | TriviaQA | 59.7 | 57.9 | 78.9 | **81.5** |
| | NaturalQuestions| 22.2 | 32.5 | 36.6 | **47.0** |
| | MMLU | 48.2 | 49.4 | **71.3** | 71.1 |
| | ARC-Easy | 67.9 | 71.0 | 76.9 | **81.6** |
| | ARC-Challenge | 48.1 | 49.4 | 59.0 | **64.1** |
| | GSM8K | 17.4 | 63.0 | 63.4 | **84.1** |
| | MATH | 6.0 | 15.8 | 18.7 | **32.6** |
| | HumanEval | 26.2 | 48.2 | 42.7 | **73.8** |
| | MBPP | 39.0 | 35.2 | 57.4 | **61.4** |
| | DROP | 41.0 | 49.1 | 67.9 | **71.9** |
| | OpenBookQA | 55.8 | 54.8 | 60.2 | **63.2** |
| | BBH | 39.5 | 42.3 | 68.7 | **71.7** |
| | AGIEval | 26.4 | 19.3 | 41.3 | **46.4** |
| **Chinese** | CLUEWSC | 73.1 | 71.9 | **81.0** | 60.0 |
| | CHID | 89.3 | 64.9 | **92.1** | 72.6 |
| | C-Eval | 45.0 | 47.0 | **66.1** | 65.2 |
| | CMMLU | 47.2 | 49.7 | **70.8** | 67.8 |
| | CMath | 34.5 | 68.4 | 63.0 | **80.3** |
| | C3 | 65.4 | 66.4 | 75.3 | **77.0** |
| | CCPM | 76.9 | 76.5 | **88.5** | 84.9 |

*Tabel 6: Perbandingan antara base dan chat models.*

Ada beberapa fenomena kunci untuk dipelajari dari hasil *fine-tuning* di atas:
*   **Tugas Turun (*Performance Drop Tasks*):** Nilai HellaSwag dan beberapa tugas bahasa dasar terus-menerus turun setelah proses SFT (Contohnya HellaSwag Base 84.0 terjun menjadi Chat 75.7). Secara logis, HellaSwag adalah uji pelengkapan kalimat kosong (*cloze test*), dan bahasa *mentah* (*pure language models*) jelas jauh lebih alami melengkapi kalimat karena memang itulah hakekat Prapelatihannya, ketimbang asisten Chat yang cenderung reaktif dan komprehensif membalas.
*   **Peningkatan Pesat di Matematika & Pemrograman:** Sebaliknya, nilai GSM8K (soal matematika sekolah) dan HumanEval meroket pesat setelah SFT. Model *Base* sebenarnya menderita "underfitted" karena tidak paham format tanya jawab soal cerita, setelah diajari dengan SFT, pengetahuan laten matematika yang terkubur meledak ke permukaan. 
*   Penurunan fluktuatif di pengetahuan faktual (MMLU/TriviaQA) bukanlah indikasi model melupakan pengetahuan tersebut, melainkan sekadar perbedaan tata cara mengekstraksi metrik secara otomatis.

### 5.2. Evaluasi Jawaban Terbuka (*Open-Ended Evaluation*)

Di dunia nyata, metrik ujian statis kurang menggambarkan keahlian model saat diajak bercakap-cakap berbalas-balasan (*multi-turn*) oleh manusia (*Open-ended*).

#### 5.2.1. Evaluasi Terbuka Tiongkok (*Chinese Open-Ended Evaluation*)

Menggunakan instrumen *AlignBench* yang diverifikasi ulang dengan GPT-4.

| Model | Overall (Total) | Reasoning (Avg) | Math | Logi. | Language (Avg) | Fund. | Chi. | Open. | Writ. | Role. | Pro. |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| gpt-4-1106-preview | 8.01 | 7.73 | 7.80 | 7.66 | 8.29 | 7.99 | 7.33 | 8.61 | 8.67 | 8.47 | 8.65 |
| gpt-4-0613 | 7.53 | 7.47 | 7.56 | 7.37 | 7.59 | 7.81 | 6.93 | 7.42 | 7.93 | 7.51 | 7.94 |
| **DeepSeek-67B-Chat-DPO*** | **6.69** | **5.77** | 6.13 | 5.41 | **7.60** | 7.29 | 7.47 | 7.82 | 7.51 | 7.83 | 7.71 |
| **DeepSeek-67B-Chat*** | **6.43** | **5.75** | 5.71 | 5.79 | **7.11** | 7.12 | 6.52 | 7.58 | 7.20 | 6.91 | 7.37 |
| chatglm-turbo | 6.24 | 5.00 | 4.74 | 5.26 | 7.49 | 6.82 | 7.17 | 8.16 | 7.77 | 7.76 | 7.24 |
| erniebot-3.5 | 6.14 | 5.15 | 5.03 | 5.27 | 7.13 | 6.62 | 7.60 | 7.26 | 7.56 | 6.83 | 6.90 |
| gpt-3.5-turbo-0613 | 6.08 | 5.35 | 5.68 | 5.02 | 6.82 | 6.71 | 5.81 | 7.29 | 7.03 | 7.28 | 6.77 |

*Tabel 7: Papan Peringkat AlignBench. Hanya menampilkan 7 besar teratas.*

DeepSeek-67B-Chat berhasil menaklukkan dan melampaui ChatGPT (GPT-3.5, skor 6.08) dengan skor impresif (6.69), menempatkannya secara mutlak pada jajaran elit langsung membayangi dewa bahasa GPT-4. Kita juga secara jelas menyaksikan bahwa fase DPO telah mengangkat rata-rata nilai secara universal.

#### 5.2.2. Evaluasi Terbuka Bahasa Inggris (*English Open-Ended Evaluation*)

Menggunakan alat *MT-Bench*. Pengujian ini sangat penting untuk menilai kesadaran obrolan multiturn panjang (*multi-turn open-ended generation ability*).

| Model | STEM | Humanities | Reasoning | Coding | Math | Extraction | Roleplay | Writing | Average |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| GPT-4-1106-preview | 9.90 | 9.95 | 8.10 | 9.05 | 7.95 | 9.90 | 9.50 | 9.70 | 9.26 |
| GPT-3.5-turbo-0613 | 9.55 | 9.95 | 6.20 | 7.05 | 7.05 | 9.00 | 8.65 | 9.65 | 8.39 |
| LLaMA-2-Chat 70B | 8.93 | 9.63 | 5.80 | 3.15 | 3.30 | 7.25 | 7.50 | 9.30 | 6.86 |
| TÜLU 2+DPO 70B | 9.00 | 9.90 | 7.00 | 4.70 | 4.65 | 9.35 | 9.25 | 9.25 | 7.89 |
| **DeepSeek LLM 67B Chat** | 9.60 | 9.70 | 8.00 | 7.35 | 6.25 | 8.40 | 8.20 | 9.30 | 8.35 |
| **DeepSeek LLM 67B Chat DPO**| **9.70** | **9.80** | **9.05** | **6.75** | **6.65** | **9.30** | **9.10** | **9.75** | **8.76** |

*Tabel 8: MT-Bench Evaluation (hanya ekstraksi model krusial).*

DeepSeek-67B Chat mencapai skor rata-rata menakjubkan 8.35, berdiri tegak di level GPT-3.5. Menakjubkannya lagi, pasca-DPO skor melonjak lagi menjadi **8.76**, yang mana menghempaskan model elit raksasa seperti TÜLU 2 70B dan hanya tunduk pada supremasi sang raja, GPT-4 (9.26). 

### 5.3. Evaluasi Kasus Data Tak Terlihat (*Held-Out Evaluation*)

Dalam dunia akademik AI, "Kecurangan Akademik" sering terjadi di mana model tanpa sadar sudah "membaca" kertas ujian (*dataset test*) saat ia menelan miliaran teks internet di tahap prapelatihan. Hal ini dinamakan "Data Contamination" atau "Benchmark Overfitting".
Untuk menghindari bias ini, diuji 3 dataset mutlak "*Held-Out*" (ujian yang baru saja diterbitkan ke internet baru-baru ini saat riset dilakukan, sehingga *mustahil* model curang menghafal kuncinya):
1.  **LeetCode:** Mengujikan masalah programing dari soal Olimpiade mingguan LeetCode (Juli 2023 - November 2023).
2.  **Hungarian National High-School Exam:** Soal matematika nasional Hungaria untuk level Grok-1.
3.  **Instruction Following Evaluation (IFEval):** Rilis Google bulan November 2023 untuk mengecek tingkat kepatuhan model dalam mematuhi batasan *prompt* super kompleks.

| Model | LeetCode | Hungarian Exam | IFEval |
| :--- | :---: | :---: | :---: |
| GPT-4 | 48.4 | 68 | 79.3 |
| ChatGLM3 6B | 2.4 | 32 | 29.7 |
| DeepSeek LLM 7B Chat | 4.7 | 28.5 | 41.2 |
| Baichuan2-Chat 13B | 1.6 | 19.5 | 44.5 |
| Yi-Chat 34B | 7.9 | 39 | 48.4 |
| Qwen 72B Chat | 12.7 | 52 | 50.8 |
| **DeepSeek LLM 67B Chat**| **17.5** | **58** | **55.5** |

*Tabel 9: Held-out Dataset Evaluation.*

**Temuan Utama bagi Ilmuwan AI:** Terdapat divergensi kemampuan (*capability gap*) yang tajam antara model kecil dan model raksasa. Model kecil seperti ChatGLM3 sering memanipulasi metrik konvensional yang lawas sehingga nampak sama pintarnya dengan 67B. Namun ketika diuji dengan soal *Held-Out* super berat yang tak pernah mereka lihat, nilai ChatGLM3 jatuh menjadi sekadar 2.4 (Leetcode), jauh terpuruk dibandingkan arsitektur 67B (17.5). Intinya: total kapasitas parameter komputasional adalah kunci nyata pembentuk *raw intelligence*.

### 5.4. Evaluasi Keselamatan (*Safety Evaluation*)

Premis hakiki pembentukan AGI adalah ia **harus mematuhi nilai kemanusiaan dan bersifat aman (friendly) bagi keberadaan manusia**. Upaya menjaga AI agar tak menjadi "racun" disertakan mulai dari prapelatihan, SFT, dan DPO. 

Taksonomi klasifikasi konten keselamatan yang dibangun oleh 20 pakar diuraikan di tabel ini:

| Category | Subcategory | #Safety Answers / #Total Cases |
| :--- | :--- | :---: |
| Diskriminasi dan Prasangka | Etnis, Ras, Agama, Geografis, Gender, Umur, Pekerjaan | 486 / 500 |
| Pelanggaran Hak Hukum Orang Lain | Kesehatan mental/fisik, properti, reputasi, privasi | 473 / 500 |
| Rahasia Dagang dan Hak Kekayaan Intelektual | Pelanggaran IP, tindakan monopolistik, rahasia dagang | 281 / 300 |
| Perilaku Kriminal & Ilegal | Aliran Sesat (Cults), Pornografi, Perjudian, Narkoba, Kekerasan | 290 / 300 |
| Isu Keamanan Lainnya | Halusinasi, sensivitas waktu, topik sangat berisiko | 767 / 800 |

*Tabel 10: Taksonomi sistematis untuk evaluasi keselamatan.*

Seluruh kategori melewati 94%+ persentase keamanan karena tim melakukan anotasi "Menolak Secara Aman", yang melindungi masyarakat dari anjuran melukai diri atau provokasi kejahatan. Model juga dievaluasi di basis independen "Do-Not-Answer" (Kumpulan *prompt* terlarang). 

| Model | Do-Not-Answer Score |
| :--- | :---: |
| LLAMA-2-7B-Chat | 99.4 |
| Claude | 98.3 |
| **DeepSeek-67B-Chat**| **97.8** |
| ChatGPT | 97.7 |
| GPT-4 | 96.5 |

*Tabel 11: Skor Do-Not-Answer.* 
Skor tinggi menandakan arsitektur memiliki imunitas toksisitas yang lebih memukau daripada GPT-4, dengan aman menangkis perintah jahat untuk merakit senjata atau menyebarkan kebencian.

### 5.5. Diskusi 

Bagian ini menyajikan wawasan empiris mendalam seputar teknik-teknik pembentukan LLM:

**Pelatihan Halus Bertahap (*Staged Fine-Tuning*):** 
Seperti disinggung di atas, melatih *math* dan *code* terlalu lama merusak keluwesan ngobrol (meningkatnya tingkat *repetisi*). Solusinya adalah model 7B dilatih di SFT tahap-1 dengan seluruh data gabungan, kemudian "dicampurkan" atau di *fine-tune* murni dengan data percakapan dialog rileks pada SFT tahap-2. 

| Model | HumanEval | GSM8K | Repetition | IFEval |
| :--- | :---: | :---: | :---: | :---: |
| DeepSeek LLM 7B Chat Stage1 | 48.2 | 63.9 | 0.020 | 38.0 |
| DeepSeek LLM 7B Chat Stage2 | 48.2 | 63.0 | 0.014 | 41.2 |

*Tabel 12: Pengurangan drastis kecacatan Repetisi lewat Stage-2.*
Bisa dilihat di Tabel 12, performa *Code* (48.2) tidak hilang sama sekali, tapi rasio Repetisi menurun dari 0.020 menjadi 0.014. Tahap-2 tidak menyakiti kapabilitas teknis namun drastis memperbaiki patologi kebingungan bahasa.

**Pertanyaan Pilihan Ganda (*Multi-Choice Question* / MC):**
Pertanyaannya: Apakah menyuntikkan 20 Juta data latihan Pilihan Ganda A/B/C/D dalam SFT bagus?

| Model | MMLU | C-Eval | CMMLU | TriviaQA | ChineseQA |
| :--- | :---: | :---: | :---: | :---: | :---: |
| DeepSeek LLM 7B Chat | 49.4 | 47.0 | 49.7 | 57.9 | 75.0 |
| DeepSeek LLM 7B Chat + MC | 60.9 | 71.3 | 73.8 | 57.9 | 74.4 |

*Tabel 13: Dampak Injeksi Pilihan Ganda (MC).*
Data MC sangat mendongkrak tolok ukur ujian pilihan ganda (MMLU terbang dari 49 ke 60, C-Eval dari 47 ke 71). Akan tetapi, nilainya **mandek alias tidak memberikan eskalasi sedikitpun** pada uji kreatif generatif seperti TriviaQA. Konklusi krusial: Memasukkan data MC dapat menjadi *overfitting* murni demi "mendongkrak rating" secara semu (*benchmark decoration*), namun sesungguhnya itu tidak memberikan kecerdasan substansial. Maka data Pilihan Ganda dieksklusi secara total pada model final!

**Data Instruksi dalam Prapelatihan:**
Meskipun mencampurkan sedikit data instruksi di akhir prapelatihan dapat mempercepat laju perbaikan metrik, potensial akhirnya terbukti identik dengan memisahkan data tersebut utuh hanya pada fase SFT. Sehingga instruksi di prapelatihan dianggap tak esensial.

**Sistem Penuntun (*System Prompt*):**
*Prompt* sistem yang terstruktur ("*You are DeepSeek Chat, a helpful, respectful...*") dieksekusi secara unik oleh masing-masing ukuran arsitektur.

```markdown
System prompt: You are DeepSeek Chat, a helpful, respectful and honest AI assistant developed by DeepSeek. The knowledge cut-off date for your training data is up to May 2023. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
```

*Analisis Kode Prompt:* Prompt panjang di atas digunakan sebagai instruksi inti awal (*prefix*) di dalam sistem. Kode prompt ini mengikat entitas identitas AI ("You are DeepSeek"), memberikan *boundary* pengetahuan ("May 2023"), dan mensyaratkan protokol keamanan anti-SARA. 

| Model | MT Bench |
| :--- | :---: |
| DeepSeek LLM 7B Chat | 7.15 |
| DeepSeek LLM 7B Chat + System Prompt | 7.11 |
| DeepSeek LLM 67B Chat | 8.35 |
| DeepSeek LLM 67B Chat + System Prompt | 8.58 |

*Tabel 14: Efek ajaib dari System Prompt pada model kecil vs masif.*
Fenomena mengagumkan: **Model kecil (7B)** mengalami **penurunan performa (degradasi)** saat dipasangi teks instruksi yang panjang (dari 7.15 turun jadi 7.11). Sebaliknya, **model raksasa (67B)** yang dipasangi instruksi yang persis sama, tingkat kecerdasannya **melonjak drastis** (dari 8.35 menembus 8.58). Penjelasan rasionalnya: Kapasitas "otak" model kecil terlalu kebingungan menyerap batasan logis (*constraints*) kompleks dari *prompt*, berbeda dengan otak model raksasa yang menangkap sempurna nuansa instruksi dan memanifestasikannya lewat respons superior yang elegan.

---

## 6. Kesimpulan, Limitasi, dan Pekerjaan Masa Depan (*Conclusion, Limitation, and Future Work*)

Di dalam bab komprehensif ini, Anda telah menyelami perancangan **DeepSeek LLMs**, korpus pelatihan 2 triliun token yang revolusioner. Sebagai cendekiawan AI, poin sentral dari bab ini meliputi demonstrasi seleksi empiris hiperparameter dan pembuktian koreksi ilmiah atas **Hukum Penskalaan (*Scaling laws*)** di mana ditemukan formula alokasi model/data yang jauh lebih optimum, sekaligus menguak misteri bahwa **kualitas data akan mendikte bentuk lekukan *scaling laws* itu sendiri**. Model telah terhindar sepenuhnya dari manipulasi nilai rapor *benchmark* semu.

Namun demikian, karya ini mewarisi **keterbatasan (limitasi)** lumrah yang masih menghantui dunia LLM modern. Di antaranya, inabilitas memutakhirkan memori faktual usai Prapelatihan dikunci, kecenderungan menenun fakta fiktif (Halusinasi/ *Hallucinations*), serta dominasi mutlak bahasa Mandarin dan Inggris yang membikin kepiawaian di literatur lintas bahasa lain menjadi defisit.

Pada babak eksplorasi inovasi selanjutnya (*Future Work*), lanskap ini akan digiring menuju:
1.  Penciptaan **Mixture-of-Experts (MoE)**: Merancang "otak" model yang berongga (*sparse model*) namun bertransformasi merilis efisiensi ledakan kekuatan komputasi (*dense model performance*).
2.  Meracik kecerdasan logika Pemrograman (*Code Intelligence*) pada prapelatihan yang melampaui imajinasi hari ini.
3.  Memasukkan intervensi *Reinforcement Learning* (Pembelajaran Penguatan) guna menggebrak kebuntuan plafon kaca kapabilitas logika penalaran rumit (*Complex Reasoning*).

*(Tamat)*