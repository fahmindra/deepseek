---
slug: modul-4-2
title: modul-4-2
---

## 4.2 Scaling Laws: Menyeimbangkan Jumlah Data dan Ukuran Model untuk Memaksimalkan Efisiensi Compute
---

Sejarah telah membuktikan menambahkan jumlah data dan parameter untuk melatih sebuah model *machine learning* dalam batasan tertentu menghasilkan peningkatan performa sebuah model, bahkan bisa memunculkan kemampuan baru seperti [emergent abilities](https://arxiv.org/abs/2206.07682) sebuah LLM yang jumlah parameternya melebihi jumlah tertentu. Banyak penemuan yang berasal dari meningkatkan jumlah parameter, data, serta sebagai konsekuensinya skala *compute* yang digunakan. Meskipun begitu, kita tidak bisa sembarangan meningkatkan jumlah parameter dan/atau menambahkan data untuk melatih model mengingat konstrain seperti budget *compute* dan waktu yang terbatas, sehingga diperlukan metoda yang lebih disiplin, terukur dan terarah dalam merencanakan peningkatan skala ini. Perlu diketahui lebih jelas **bagaimana performa meningkat seiring dengan meningkatnya skala**. Dengan kata lain, diperlukan fungsi yang bisa memberikan estimasi peningkatan performa yang bisa didapatkan dengan meningkatkan skala dari suatu variabel, baik itu data, parameter maupun *compute* (yang dibarengi meningkatkan skala data dan parameter untuk memaksimalkannya).

Berbagai usaha dilakukan oleh komunitas riset untuk memahami hubungan antara tingkatan performa dengan skala dari sebuah model. Pada dasarnya, riset mengenai hubungan ini, yang umumnya disebut sebagai **scaling law**, mencoba menjawab beberapa pertanyaan ini:

*   Berapa *compute*, data dan jumlah parameter yang diperlukan untuk mencapai tingkat performa tertentu?
*   Berapa jumlah data yang optimal untuk melatih model dengan jumlah parameter tertentu? Berapa lama model ini perlu dilatih hingga optimal?
*   Jika diberikan *compute budget* tambahan, bagaimana caranya meningkatkan jumlah data dan parameter model agar bisa dimanfaatkan secara optimal?

### 4.2.1 Memahami Scaling Law: Definisi dan Sejarah Penelitiannya

Pada bagian ini, kita akan membahas selayang pandang mengenai **scaling law** pada konteks LLM berbasis arsitektur *transformer*. Umumnya, scaling laws berusaha mencari hubungan jumlah parameter model (N), jumlah data dalam token (D) untuk memaksimalkan performa model dengan *constraint* berupa *compute* atau kapasitas komputasi (C) yang biasa dihitung dalam satuan FLOPS (Floating Point Operations atau total operasi yang dilakukan untuk melakukan proses *training*). Mari terlebih dahulu pahami apa yang dimaksud dengan *compute budget* dan bagaimana caranya menentukan *compute* yang dibutuhkan. Hubungan ini diasumsikan memiliki bentuk *power law*. Jika sebuah variabel y memiliki hubungan *power law* dengan variabel x, maka hubungan ini dapat diekspresikan dengan rumus:

$$y = k * x^\alpha$$

Dimana $k$ adalah sebuah koefisien dan $\alpha$ adalah koefisien pangkat yang mendeskripsikan hubungan tersebut.

### 4.2.2 Intermezzo: Intuisi Singkat untuk Memahami FLOP dan Cara Menentukan Kebutuhan Compute

Satuan ini mungkin terdengar begitu asing dan sulit digambarkan, tetapi sebenarnya cukup intuitif untuk dipahami. Umumnya dalam dunia *scientific computing*, kemampuan suatu akselerator seperti GPU untuk melakukan komputasi dihitung menggunakan satuan FLOPS (Floating-point Operations per Second) atau jumlah operasi yang dia bisa lakukan per detik. Sebagai pengembang model, kita ingin memanfaatkan akselerator untuk menyelesaikan proses pengembangan model. Namun yang menjadi masalah adalah seringkali budget yang kita miliki terbatas, sehingga perlu dipastikan bahwa budget yang kita miliki cukup untuk memenuhi kebutuhan komputasi.

Satuan yang digunakan untuk menyebut kebutuhan compute adalah FLOPs (jumlah total operasi yang dilakukan), yang bisa dihitung secara analitik terutama karena berbagai macam model *machine learning* dan algoritma *training* memiliki jumlah operasi yang sudah diketahui. Informasi ini bisa memberikan kita gambaran:

*   Berapa banyak GPU / Data center yang perlu kita sewa, dan selama berapa lama?
*   Berapa lama proses training ini akan berjalan pada GPU yang kita miliki

Perlu diingat semua ini masih di luar kebutuhan lain, seperti kebutuhan *memory* yang menjadi constraint lainnya dalam menentukan kebutuhan *compute*.

Misalkan kita memperkirakan total kebutuhan komputasi untuk proses training sebesar $5 \times 10^{23}$ FLOPs (misalnya untuk melatih model berukuran 7 miliar parameter pada sekitar 12 miliar token). Kita memiliki 4 perangkat GPU, masing-masing dengan kemampuan 104.8 TFLOPS (setara dengan $1.048 \times 10^{14}$ FLOPS).

Jika seluruh kapasitas tersebut dapat dimanfaatkan sepenuhnya secara paralel, maka total throughput (total FLOPS) kita adalah:

$$4 \times 1.048 \times 10^{14} = 4.192 \times 10^{14} \text{ FLOPS}$$

Kemudian, waktu untuk menyelesaikan training bisa dihitung dengan:

$$\frac{5 \times 10^{23} \text{ FLOPs}}{4.192 \times 10^{14} \text{ FLOPs/detik}} \approx 1.19 \times 10^9 \text{ detik} \approx 13,800 \text{ hari} \approx 37.8 \text{ tahun}$$

Meskipun terdengar tidak realistis, estimasi ini masih terlalu optimis. Pada praktiknya, total FLOPS yang mungkin dicapai tidak akan sempurna 4 kali lipat jika menggunakan GPU. Figur efisiensi FLOPS ini umumnya berkisar 30-60% untuk model dengan skala besar, dikarenakan beberapa *overhead* (waktu ekstra) yang harus dihabiskan untuk setiap GPU berkomunikasi satu sama lain. Sehingga pada praktiknya bisa lebih lama lagi. Karena itu, untuk menurunkan waktu training hingga durasi yang masuk akal (misalnya 1-4 minggu), skala hardware yang dibutuhkan harus jauh lebih besar, bisa mencapai ratusan hingga ribuan GPU. 

Sebagai perbandingan, [hasil benchmark yang dipublikasikan](https://aiconference.com/wp-content/uploads/2024/09/Sandeep-Krishnamurthy-Engineering-Director-databricks-Pre-Training-LLM-Lessons-from-Trenches-.pdf) menunjukkan bahwa model berukuran 7B parameter dapat selesai dilatih menggunakan 140B token dalam waktu sekitar 6–7 hari menggunakan sekitar 64 GPU kelas tinggi dengan kemampuan $3.12 \times 10^{14}$ FLOPS. Maka dari itu, menggunakan hanya 4 GPU akan menghasilkan waktu training yang jauh di luar batas realistis. Selain itu pada praktiknya model dengan ukuran besar memerlukan banyak GPU hanya agar kebutuhan *memory* untuk semua parameter, aktivasi, *state* optimizer dan nilai lainnya yang dihitung saat **training** bisa terpenuhi. Pada praktiknya, skala *training* yang dilakukan pada industri maupun riset membutuhkan puluhan, ratusan, hingga ribuan GPU. Berikut adalah informasi mengenai beberapa model yang dilatih yang melaporkan detail *training* nya:

**Tabel 5. Perbandingan durasi training beberapa model beserta konfigurasinya**

| Model | Jumlah parameter | Jumlah data | GPU / Infrastruktur | Waktu training |
| :--- | :--- | :--- | :--- | :--- |
| **Gopher** (DeepMind) | ~280B parameter ([Scribd](https://www.scribd.com/document/920014288/CL3410-Language-Models-and-Agents-Scalling-Laws-Corpora-for-Pre-Training?utm_source=chatgpt.com)) | ~300B ([Scribd](https://www.scribd.com/document/920014288/CL3410-Language-Models-and-Agents-Scalling-Laws-Corpora-for-Pre-Training?utm_source=chatgpt.com)) | 4096 chip TPU-v3 ([deepsense.ai](https://deepsense.ai/wp-content/uploads/2023/03/2112.11446.pdf?utm_source=chatgpt.com)) | ~920 jam (~38 hari) ([deepsense.ai](https://deepsense.ai/wp-content/uploads/2023/03/2112.11446.pdf?utm_source=chatgpt.com)) |
| **GPT-3** (OpenAI) | ~175B parameter ([Tomorrow Desk](https://tomorrowdesk.com/info/gpt-3?utm_source=chatgpt.com)) | ~300B ([ownyourai.com](https://www.ownyourai.com/training-compute-optimal-large-language-models/?utm_source=chatgpt.com)) | ~1.024 GPU A100 (diperkirakan) ([Scribd](https://www.scribd.com/document/920014282/CL3410-Language-Models-and-Agents-Tokenization-Byte-Latent-Transformer?utm_source=chatgpt.com)) | Sekitar satu bulan ([NVIDIA Developer](https://developer.nvidia.com/blog/scaling-language-model-training-to-a-trillion-parameters-using-megatron/?utm_source=chatgpt.com)) |
| **Megatron-Turing NLG 530B** | ~530B parameter ([ownyourai.com](https://www.ownyourai.com/training-compute-optimal-large-language-models/?utm_source=chatgpt.com)) | ~270–339B ([ownyourai.com](https://www.ownyourai.com/training-compute-optimal-large-language-models/?utm_source=chatgpt.com)) | 560 DGX A100 servers (8×A100 each) ([GeeksforGeeks](https://www.geeksforgeeks.org/nlp/megatron-turing-natural-language-generation-nlg-530b/?utm_source=chatgpt.com)) | ~40–50 hari pada cluster besar ([Reddit](https://www.reddit.com/r/mlscaling/comments/q5wsww?utm_source=chatgpt.com)) |

### 4.2.3 Scaling Laws for Neural Language Models: Kaplan Scaling Laws

*Kaplan Scaling Laws* secara singkat menyatakan bahwa untuk mengoptimalkan komputasi, lebih baik meningkatkan ukuran model (jumlah parameter) dibandingkan menambah data latih untuk mencapai suatu tingkatan performa tertentu.

Riset ini menemukan bahwa performa proses *training* LLM (yang diukur sebagai *test loss* yang bisa dicapai dalam proses training, digunakan notasi L) hanya meningkat seiring dengan meningkatnya beberapa faktor:
*   Jumlah parameter model (N)
*   Jumlah data yang digunakan (D)
*   Jumlah *compute* yang tersedia (C)

Riset ini melakukan *training* pada model dengan arsitektur yang sama dengan skala yang berbeda dari segi data, parameter maupun *compute*, dan menemukan bahwa nilai loss yang bisa dicapai menunjukkan hubungan berikut:

**Formula 1**
$$
\begin{aligned}
L(C) &= \left(\frac{3.1 \times 10^8}{C}\right)^{0.05} \\
L(D) &= \left(\frac{5.4 \times 10^{13}}{D}\right)^{0.095} \\
L(N) &= \left(\frac{8.8 \times 10^{13}}{N}\right)^{0.076}
\end{aligned}
$$
**Caption: Formula 1. Persamaan-persamaan yang menunjukkan hubungan compute, jumlah data dan jumlah parameter dengan tingkat loss yang bisa dicapai dalam proses training menurut Kaplan Scaling Law**

Walaupun terkesan rumit, arti dari ketiga rumus tersebut cukup simpel. Nilai *loss* yang bisa dicapai dalam proses training memiliki hubungan terbalik dengan tiga variabel tersebut. Artinya adalah meningkatkan ketiga variabel tersebut dapat menurunkan *loss*. Satu temuan yang penting untuk diingat adalah hubungan ini hanya berlaku apabila rasio antara jumlah data dan jumlah parameter memenuhi rasio tertentu saja.

![Gambar 2](https://lms.sdmdigital.id/pluginfile.php/635387/mod_page/content/20/image50.jpg?time=1765183531084)
**Gambar 2. Temuan utama dalam Kaplan Scaling Law yang menunjukkan tren penurunan loss dengan meningkatkan setiap variabel data, parameter dan compute**

Riset ini juga menemukan efek meningkatkan D dan N secara bersamaan terhadap nilai L:

![Gambar 3](https://lms.sdmdigital.id/pluginfile.php/635387/mod_page/content/20/image46.jpg?time=1765183590367)
**Gambar 3. Efek meningkatkan D dan N terhadap Loss, perhatikan bahwa meningkatkan jumlah parameter memberikan penurunan loss yang lebih besar dibanding menambahkan data hingga titik tertentu**

Grafik ini menunjukkan bahwa model yang lebih besar bisa mencapai **loss** yang lebih rendah untuk jumlah data yang sama sehingga lebih *sample-efficient*. Keperluan penambahan data seiring bertambahnya parameter memiliki hubungan:

$$D \underset{\sim}{\propto} N^{0.74}$$

Terakhir, riset ini menyajikan bagaimana variabel perlu ditingkatkan seiring meningkatnya *compute*:

$$N \propto C_{min}^{0.73}, \quad B \propto C_{min}^{0.24}, \quad S \propto C_{min}^{0.03}$$

Namun, riset Kaplan memiliki beberapa catatan pengganjal seperti tidak menghitung *embeddings* sebagai parameter dan menggunakan model yang belum selesai dilatih (*early stopping*).

### 4.2.4 Training Compute-Optimal Large Language Models: Chincilla Scaling Laws

Temuan utama dari riset ini adalah untuk **meningkatkan performa, baik data dan jumlah parameter memiliki prioritas yang sama untuk ditingkatkan**. Riset ini menunjukkan model *Chincilla* 70B yang dilatih dengan 1.4T token mampu mengalahkan *Gopher* (280B, 300B token) dan GPT-3 (175B, 300B token).

#### Pendekatan 1: Ukuran Model Tetap, Variasikan Jumlah Data (Token)
Eksperimen ini menemukan bahwa untuk *compute budget* yang digunakan melatih *Gopher* (280B), performa optimal sebenarnya bisa dicapai oleh model 67B yang dilatih dengan 1.5T token.

![Gambar 4](https://lms.sdmdigital.id/pluginfile.php/635387/mod_page/content/20/image2.jpg?time=1765183798457)
**Gambar 4. Hasil dari pendekatan pertama, dimana nilai loss yang didapatkan digunakan untuk melakukan estimasi kombinasi jumlah parameter (tengah) dan jumlah data (kanan) yang optimal untuk compute budget tertentu.**

Hubungan optimal: $N_{opt} \propto C^{0.5}; \quad D_{opt} \propto C^{0.5}$

#### Pendekatan 2: Nilai FLOPs yang Sama, Variasikan Ukuran Model

![Gambar 5](https://lms.sdmdigital.id/pluginfile.php/635387/mod_page/content/20/image16.jpg?time=1765183845778)
**Gambar 5. Hasil Pendekatan 2**

Ditemukan bahwa N dan D memiliki prioritas hampir sama: $N_{opt} \propto C^{0.49}; \quad D_{opt} \propto C^{0.51}$

#### Pendekatan Ketiga: Fitting Model Prediktor Loss Function

**Formula 2**
$$\hat{L}(N, D) \triangleq E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$$
**Caption: Formula 2. Fungsi yang digunakan untuk memodelkan hubungan loss dengan N (jumlah parameter) dan D (jumlah data dalam token)**

Hasil pemodelan akhir menunjukkan hubungan: $N_{opt} \propto C^{0.46}; \quad D_{opt} \propto C^{0.54}$

**Tabel 6. Jumlah Parameter dan Data yang optimal (Estimasi Chincilla)**

| FLOPs | Parameter | Jumlah data (token) | Rasio token / parameter |
| :--- | :--- | :--- | :--- |
| 1.92e+19 | 400M | 8B | 20 |
| 1.21e+20 | 1B | 20.2B | 20.2 |
| 1.23e+22 | 10B | 205.1B | 20.51 |
| 5.76e+23 | 67B | 1.5T | 22.38 |
| 3.85e+24 | 175B | 3.7T | 21.14 |

Berdasarkan hasil ini, diperoleh rasio dimana untuk setiap 1 parameter, dibutuhkan sekitar **20 token** untuk melatihnya.

#### Menggabungkan Temuan Kaplan dan Hoffman (Chincilla)
Penelitian Pearce & Song ([“Reconciling Kaplan and Chinchilla Scaling Laws”](https://arxiv.org/abs/2406.12907)) menunjukkan bahwa temuan keduanya konsisten jika skenario eksperimen disamakan (terutama terkait perhitungan *embeddings*).

![Gambar 6](https://lms.sdmdigital.id/pluginfile.php/635387/mod_page/content/20/image27.png)
**Gambar 6. Temuan dari penelitian Pearce & Song yang menemukan alasan dari perbedaan hasil yang ditemukan pada penelitian Kaplan dan Hoffman**

### 4.2.5 Beyond Chincilla-Optimal: Sudut Pandang Inference

Temuan utama: model yang optimal menurut Chincilla (untuk training) mungkin masih terlalu boros saat *inference*. Untuk efisiensi *production*, disarankan melatih model yang lebih kecil menggunakan data yang jauh lebih banyak dari rasio Chincilla.

**Formula 3 (Estimasi Compute Training):** $C_{train} = 6 * N * D_{train}$
**Formula 4 (Estimasi Compute Inference):** $C_{inference} = 2 * N * D_{inf}$

Prinsipnya: Untuk performa mirip tapi lebih efisien di *production*, gunakan model parameter lebih kecil dan latih dengan data lebih banyak. Contoh: LLaMa 6.7B dilatih dengan 1T token (jauh melampaui rasio 1:20).

### Mini Quiz
*(Silakan akses kuis interaktif pada platform LMS)*

> **Refleksi Singkat**
> **Scaling laws** memberikan prinsip dasar untuk merencanakan kebutuhan *compute*, data dan parameter. Bagaimana pelaku industri memanfaatkan prinsip ini? Apakah prinsip ini masih berlaku untuk *fine tuning*? Coba cari tahu penggunaan praktikalnya di jurnal ilmiah atau blog post industri!