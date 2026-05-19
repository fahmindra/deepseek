---
slug: ch-12-07
title: "Hyper-Connections"
layout: chapter
---

## Abstrak

Dalam kajian ini, para peneliti mempresentasikan **hyper-connections**, sebuah metode yang sederhana namun sangat efektif yang dapat berfungsi sebagai alternatif pengganti koneksi residual (*residual connections*). Pendekatan ini secara spesifik mengatasi kelemahan-kelemahan umum yang sering diobservasi pada berbagai varian koneksi residual, seperti efek jungkat-jungkit (*seesaw effect*) antara masalah gradien yang menghilang (*gradient vanishing*) dan keruntuhan representasi (*representation collapse*).

Secara teoretis, koneksi hiper (hyper-connections) memungkinkan jaringan saraf untuk menyesuaikan (mengatur) kekuatan koneksi antara fitur-fitur pada tingkat kedalaman (*depths*) yang berbeda dan secara dinamis mengatur ulang lapisan-lapisan (*rearrange layers*). Para peneliti melakukan berbagai eksperimen yang berfokus pada pra-pelatihan model bahasa besar (*pre-training of large language models* / LLMs), termasuk model yang padat (*dense*) maupun model yang jarang (*sparse*). 

Dalam eksperimen tersebut, koneksi hiper menunjukkan peningkatan performa yang sangat signifikan dibandingkan dengan koneksi residual konvensional. Eksperimen tambahan yang dilakukan pada tugas-tugas visi komputer (*vision tasks*) juga mendemonstrasikan peningkatan yang serupa. Para peneliti mengantisipasi bahwa metode ini akan dapat diaplikasikan secara luas dan memberikan manfaat yang besar di berbagai spektrum permasalahan Kecerdasan Buatan (AI).

---

## 1. Pendahuluan (*Introduction*)

Pembelajaran mendalam (*Deep learning*) telah meraih kesuksesan yang sangat luar biasa di berbagai domain, di mana **koneksi residual (*residual connections*)** (He et al., 2016) telah menjadi instrumen yang sangat penting dalam arsitektur jaringan saraf kontemporer, termasuk pada arsitektur Transformers dan CNNs (*Convolutional Neural Networks*). 

Koneksi residual sangat membantu memitigasi masalah gradien yang menghilang (*gradient vanishing*), sehingga memungkinkan pelatihan jaringan yang sangat dalam menjadi efektif. Namun, sangat penting untuk menyadari bahwa koneksi residual bukanlah solusi yang sempurna tanpa cela; mereka masih memiliki keterbatasan-keterbatasan yang hingga kini belum terselesaikan.

Dua varian utama dari koneksi residual, yaitu **Pre-Norm** dan **Post-Norm**, masing-masing membuat kompromi (*trade-offs*) yang berbeda antara masalah *gradient vanishing* dan *representation collapse* (keruntuhan representasi).
*   **Pre-Norm** mengaplikasikan operasi normalisasi pada masukan (*input*) sebelum masuk ke setiap blok residual. Hal ini secara efektif mengatasi masalah gradien yang menghilang. Namun, metode ini juga dapat mengarah pada masalah keruntuhan pada representasi yang dalam (Liu et al., 2020), di mana fitur-fitur tersembunyi (*hidden features*) di lapisan-lapisan yang lebih dalam menjadi sangat mirip satu sama lain. Akibatnya, kontribusi dari penambahan lapisan-lapisan baru menjadi semakin berkurang seiring dengan bertambahnya jumlah lapisan.
*   Sebagai kontras, **Post-Norm** mengaplikasikan normalisasi *setelah* keluaran (*output*) dari setiap blok residual, sehingga mengurangi pengaruh dari suatu status tersembunyi (*hidden state*) terhadap lapisan-lapisan berikutnya. Pendekatan ini dapat meringankan masalah keruntuhan representasi, tetapi sayangnya, hal ini kembali memperkenalkan masalah gradien yang menghilang.

Masalah gradien yang menghilang dan keruntuhan representasi ini bagaikan dua ujung dari sebuah papan jungkat-jungkit (*seesaw*), di mana kedua varian (Pre-Norm dan Post-Norm) membuat kompromi masing-masing di antara kedua masalah tersebut. Isu utamanya adalah bahwa koneksi residual (baik varian Pre-Norm maupun Post-Norm) **telah menetapkan secara kaku (predefine) kekuatan koneksi antara keluaran dan masukan di dalam sebuah lapisan**.

---
### Analisis Gambar 1: Performa OLMoE-1B-7B dengan Hyper-Connections

*(Deskripsi Gambar: Terdapat empat grafik yang menunjukkan perbandingan performa antara model dasar OLMoE-1B-7B (kurva biru) dan model yang menggunakan Hyper-Connections, yakni OLMoE-1B-7B-DHC×4 (kurva merah).*

*   **Grafik (1) - Training Loss (Sumbu X: Tokens (Billions) 100-500, Sumbu Y: Loss 2.275-2.400):** Menunjukkan kurva kerugian pelatihan yang telah diperhalus menggunakan *Exponential Moving Average* (0.99 EMA). Kurva merah (metode usulan) menukik jauh lebih tajam ke bawah dan secara konsisten berada di bawah kurva biru. Terdapat teks anotasi "$\times1.8$ 0.027" di dekat ujung kurva, yang mengindikasikan bahwa metode usulan konvergen 1.8 kali lebih cepat.
*   **Grafik (2) - C4 en val. loss (Sumbu X: Tokens 100-500, Sumbu Y: Loss 2.65-2.85):** Menunjukkan kerugian pada data validasi C4 bahasa Inggris. Sekali lagi, kurva merah berada di bawah kurva biru dengan selisih yang signifikan (anotasi "$\times1.8$ 0.028").
*   **Grafik (3) - HellaSwag Acc. (Sumbu X: Tokens 100-500, Sumbu Y: Accuracy (%) 57.5-70.0):** Menunjukkan akurasi pada tolok ukur HellaSwag. Kurva merah (DHC$\times4$) mendaki lebih tinggi dan stabil di atas kurva biru (Baseline) di sepanjang fase pelatihan.
*   **Grafik (4) - ARC Challenge Acc. (Sumbu X: Tokens 100-500, Sumbu Y: Accuracy (%) 32.5-47.5):** Menunjukkan akurasi pada ARC-Challenge. Meskipun berfluktuasi (zigzag), tren umum kurva merah secara meyakinkan berada di atas kurva biru.

**Kesimpulan Akademis Gambar 1:** Rangkaian grafik ini memberikan bukti empiris yang kuat. Model yang dilengkapi dengan *Dynamic Hyper-Connections* (DHC$\times4$) tidak hanya belajar (konvergen) secara drastis lebih cepat (1.8 kali lipat) dibandingkan baseline, tetapi juga mencapai tingkat kesalahan (Loss) yang secara absolut lebih rendah pada 500 Miliar token pelatihan. Keunggulan teknis ini langsung berdampak positif dan signifikan pada kemampuan nyata model dalam menyelesaikan tugas pemahaman bahasa (HellaSwag dan ARC-Challenge).
---

Didorong oleh berbagai keterbatasan dari koneksi residual tersebut, sebuah pertanyaan penting muncul: *Dapatkah jaringan saraf secara otonom (mandiri) mempelajari kekuatan koneksi yang optimal untuk meningkatkan performa?* 

Untuk menjawab tantangan ini, para peneliti mengusulkan **Hyper-Connections (HC)**, yang mengarah pada peningkatan performa yang signifikan dengan penambahan komputasi dan parameter yang dapat diabaikan (*negligible*). Peneliti akan menunjukkan bahwa baik varian Post-Norm maupun Pre-Norm sebenarnya dapat diekspresikan sebagai bentuk-bentuk spesifik yang tidak dapat dilatih (*non-trainable*) dari hyper-connections (akan dibahas di Subbab 3.1).

Ide inti dari hyper-connections (HC) adalah untuk mengusulkan representasi **koneksi-kedalaman (*depth-connections*)** dan **koneksi-lebar (*width-connections*)** yang dapat dipelajari (*learnable*), seperti yang digambarkan pada Gambar 2(b).

Koneksi-koneksi ini secara fleksibel mengintegrasikan berbagai fitur secara vertikal melintasi berbagai kedalaman (*depths*), sangat berbeda dengan koneksi residual konvensional yang kaku seperti pada Gambar 2(a). *Depth-connections* dapat dianggap sebagai koneksi residual yang digeneralisasi (*generalized residual connections*), yang memberikan pembobotan pada koneksi antara masukan dan keluaran dari setiap lapisan.

Untuk memungkinkan jaringan memodelkan berbagai *depth-connections* yang berbeda secara simultan, jaringan memperluas (*expand*) masukannya menjadi $n$ salinan (copies), di mana setiap salinan memiliki *depth connection*-nya sendiri, seperti yang ditunjukkan pada Gambar 2(b). Desain arsitektur ini memungkinkan banyak vektor tersembunyi (*multiple hidden vectors*) untuk mencadangkan (menyimpan) berbagai pola ganda yang menghubungkan lapisan-lapisan sebelumnya (dibahas di Subbab 4.5).

Lebih jauh lagi, dibangun **width-connections** (koneksi lebar) di antara ke-$n$ vektor tersembunyi tersebut, yang memungkinkan pertukaran informasi di antara vektor-vektor tersembunyi yang berada di dalam lapisan yang sama (Gambar 2(b)).

Para peneliti berargumen bahwa memiliki $n$ (> 1) *hidden states* adalah suatu keharusan. Seperti yang dianalisis pada Lampiran F (tidak disertakan), efek jungkat-jungkit (seesaw) akan tetap ada jika kita hanya menggunakan $n = 1$, dan eksperimen membuktikan bahwa hal tersebut tidak meningkatkan performa (lihat Gambar 5 nantinya). Sebaliknya, ketika $n > 1$, hyper-connections tidak hanya dapat belajar untuk menyesuaikan kekuatan dari nilai residual, tetapi juga dapat **mengatur ulang urutan lapisan (*rearrange layers*)**, baik secara sekuensial (berurutan) maupun secara paralel, seperti yang akan didiskusikan di Subbab 3.2.

Untuk lebih meningkatkan fleksibilitas, diperkenalkan **Dynamic Hyper-Connections (DHC)**, yang memungkinkan jaringan untuk secara dinamis menyesuaikan bobot koneksinya berdasarkan masukan (*input*). Patut dicatat, meskipun HC tampaknya memperlebar ukuran jaringan sebesar $n$ kali lipat, **parameter tambahan dan biaya komputasi yang dibutuhkan hampir dapat diabaikan (*negligible*)**, seperti yang dianalisis di Lampiran B (tidak disertakan). Arsitektur Transformer yang menggunakan HC diilustrasikan pada Gambar 17.

Riset ini, yang pada dasarnya berpusat pada pra-pelatihan model bahasa besar (*Large Language Models* / LLMs), juga diperluas keberlakuannya ke tugas-tugas generasi dan klasifikasi visual. Menggunakan metode Pre-Norm sebagai garis dasar (*baseline*), penelitian ini mendemonstrasikan manfaat signifikan dari hyper-connections, mencakup pengujian pada model padat (*dense models*) berukuran 1B (Miliar) dan 7B parameter, serta model 7B *Mixture-of-Experts* (MoE), seperti yang dirinci pada Subbab 4.

Manfaat ini sangat menonjol pada model OLMoE (Muennighoff et al., 2024) yang disajikan pada Gambar 1. Model yang memanfaatkan DHC konvergen **1.8 kali lebih cepat** dan menunjukkan peningkatan sebesar **6 poin** pada dataset ARC-Challenge dibandingkan dengan *baseline* yang sama-sama dilatih dengan 500 Miliar (500 B) token.

---
### Analisis Gambar 2: Perbandingan Residual Connections vs Hyper-Connections

*(Deskripsi Gambar: Diagram arsitektur struktural blok jaringan)*
*   **(a) Residual connections:** Masukan ($h$) bercabang dua. Satu jalur langsung ke bawah (*skip connection*), jalur lain masuk ke dalam kotak komputasi "layer". Keluaran dari "layer" kemudian **dijumlahkan** dengan jalur *skip connection* menggunakan simbol operator $+$. Arsitekturnya sangat linier dan statis.
*   **(b) Hyper-connections (HC):** Arsitekturnya jauh lebih kompleks. Input dipecah menjadi beberapa *hidden vectors* (dilambangkan kotak biru, hijau). Vektor-vektor ini mengalami persilangan pertukaran informasi secara lateral/horizontal (Width-connections dengan parameter bobot $\alpha$). Kemudian, keluaran dari "layer" dan vektor-vektor asli disilangkan kembali secara vertikal (Depth-connections dengan parameter bobot $\beta$). Semuanya adalah operasi penjumlahan berbobot.
*   **(c) Depth-connections:** Menyoroti bagian vertikal dari HC. Melakukan penjumlahan berbobot (skalar $\beta$) antara keluaran dari blok "layer" dengan vektor tersembunyi aslinya.
*   **(d) Width-connections:** Menyoroti bagian horizontal dari HC. Menunjukkan bagaimana informasi dari berbagai vektor tersembunyi di dalam satu lapisan yang sama dicampur menggunakan bobot skalar $\alpha$.
*   **Legenda:** Kotak berwarna merepresentasikan skalar atau bobot dari hyper-connections. Kotak panjang merepresentasikan *hidden vectors*. Kotak putih "layer" merepresentasikan operasi atensi (Attention) atau Feed-Forward Network (FFN). Tanda $\oplus$ adalah operasi penjumlahan.

**Kesimpulan Akademis Gambar 2:** Gambar ini dengan brilian mengilustrasikan transisi dari topologi jaringan yang "kaku" (Residual) menuju topologi yang "mengalir dan fleksibel" (Hyper-connections). Dengan ekspansi ke $n=2$ vektor tersembunyi, arsitektur HC mengizinkan jaringan saraf untuk tidak hanya melewatkan informasi dari bawah ke atas, tetapi juga melakukan *mixing* (pencampuran) informasi di berbagai ruang dimensi secara terukur melalui bobot $\alpha$ dan $\beta$ yang dapat dipelajari secara dinamis.
---

---
### Analisis Gambar 3: Analisis Similaritas Kosinus (Cosine Similarity)

*(Deskripsi Gambar: Grafik garis yang memplot Similaritas Kosinus terhadap Indeks Lapisan)*
*   **Sumbu X:** Layer Index $i$ (Indeks kedalaman lapisan jaringan dari 0 hingga 30).
*   **Sumbu Y:** $\cos(h_0^i, h_0^{i+1})$ yang berarti nilai Similaritas Kosinus (tingkat kemiripan) antara vektor masukan pada lapisan saat ini ($i$) dengan lapisan sebelumnya ($i+1$). Nilai berkisar antara 0.0 hingga 1.0. Nilai 1.0 berarti representasinya identik atau kolaps (*collapse*).
*   **Kurva Merah (Pre-Norm):** Menunjukkan fluktuasi di awal, tetapi seiring bertambahnya kedalaman lapisan (indeks > 10), kurvanya memaku di atas angka 0.8 bahkan menyentuh 1.0. Ini adalah bukti visual yang nyata dari **Representation Collapse** (keruntuhan representasi). Lapisan-lapisan dalam pada arsitektur Pre-Norm berhenti menghasilkan fitur baru dan hanya mengulang-ulang fitur yang sama.
*   **Kurva Biru (Hyper-Connection):** Menunjukkan nilai median kemiripan (garis tebal) dan rentang persentil ke-5 hingga ke-95 (area bayangan biru). Rata-rata kemiripannya jauh lebih rendah, berfluktuasi secara sehat di kisaran 0.6 hingga 0.8 di seluruh kedalaman lapisan. Area bayangannya juga lebar.

**Kesimpulan Akademis Gambar 3:** Analisis visualisasi ini adalah justifikasi utama mengapa Hyper-Connections jauh lebih superior dari Pre-Norm. Hyper-Connections secara aktif mencegah keruntuhan representasi. Fakta bahwa nilai kemiripannya lebih rendah dan memiliki jangkauan (variansi) yang luas membuktikan bahwa setiap lapisan di dalam arsitektur HC dipaksa untuk bekerja keras mengekstrak fitur-fitur baru dan berbeda, sehingga **memaksimalkan dampak dan kegunaan dari setiap penambahan lapisan jaringan**.
---

Bukti-bukti kuat yang disajikan dalam pendahuluan ini secara nyata mendemonstrasikan generalitas dari prinsip hyper-connections, dan kita (para akademisi) sangat mengantisipasi penerapan teknisnya di berbagai macam tantangan AI lainnya.

---

## 2. Metode (*Method*)

### 2.1 Static Hyper-Connections (Koneksi Hiper Statis)

Mari kita pertimbangkan sebuah vektor tersembunyi (hidden vector) $\mathbf{h}^{k-1} \in \mathbb{R}^d$ (atau $\mathbf{h}^{k-1} \in \mathbb{R}^{d \times 1}$) sebagai masukan (*input*) menuju lapisan ke-$k$, dengan masukan absolut paling awal ke dalam seluruh jaringan adalah $\mathbf{h}^0$. 

Pada mulanya, vektor $\mathbf{h}^0 \in \mathbb{R}^d$ direplikasi (disalin) sebanyak $n$ kali untuk membentuk matriks **hyper hidden awal** $\mathbf{H}^0$:

$$ \mathbf{H}^0 = \begin{pmatrix} \mathbf{h}^0 & \mathbf{h}^0 & \dots & \mathbf{h}^0 \end{pmatrix}^\top \in \mathbb{R}^{n \times d} $$

Di sini, $n$ adalah **tingkat ekspansi (*expansion rate*)**. 

Untuk lapisan ke-$k$, masukannya terdiri dari matriks hyper hidden yang dihasilkan dari lapisan sebelumnya, yaitu $\mathbf{H}^{k-1} = \begin{pmatrix} \mathbf{h}_1^{k-1} & \mathbf{h}_2^{k-1} & \dots & \mathbf{h}_n^{k-1} \end{pmatrix}^\top \in \mathbb{R}^{n \times d}$. 

Pada akhirnya, kita akan menjumlahkan (*sum*) matriks hyper hidden terakhir secara berbaris (row-wise) untuk memperoleh vektor tersembunyi final yang dibutuhkan, yang kemudian akan dilewatkan melalui sebuah proyektor akhir untuk menghasilkan *output* akhir dari jaringan (misalnya, masuk ke lapisan normalisasi dan lapisan *unembedding* pada arsitektur Transformers). 

Untuk menyederhanakan notasi matematis pada analisis-analisis selanjutnya, kita akan mengabaikan indeks lapisan ($k$) dan cukup mendonasikan matriks hyper-hidden sebagai $\mathbf{H} = \begin{pmatrix} \mathbf{h}_1 & \mathbf{h}_2 & \dots & \mathbf{h}_n \end{pmatrix}^\top$.

**Hyper-connections (HC)** dapat direpresentasikan secara matematis oleh sebuah matriks, $\mathcal{HC}$, di mana setiap elemen di dalam matriks tersebut mendefinisikan bobot koneksinya. Matriks tersebut distrukturkan sebagai berikut:

$$ \mathcal{HC} = \begin{pmatrix} \mathbf{0}_{1 \times 1} & \mathbf{B} \\ \mathbf{A}_m & \mathbf{A}_r \end{pmatrix} = \begin{pmatrix} 0 & \beta_1 & \beta_2 & \dots & \beta_n \\ \alpha_{1,0} & \alpha_{1,1} & \alpha_{1,2} & \dots & \alpha_{1,n} \\ \alpha_{2,0} & \alpha_{2,1} & \alpha_{2,2} & \dots & \alpha_{2,n} \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ \alpha_{n,0} & \alpha_{n,1} & \alpha_{n,2} & \dots & \alpha_{n,n} \end{pmatrix} \in \mathbb{R}^{(n+1) \times (n+1)} \quad \text{(1)} $$

**Penjelasan Rumus (1):**
*   $\mathcal{HC}$ adalah matriks blok (block matrix) berukuran $(n+1) \times (n+1)$. Matriks ini menyimpan seluruh parameter bobot skalar dari koneksi hiper.
*   $\mathbf{0}_{1 \times 1}$ adalah skalar nol di kiri atas.
*   $\mathbf{B}$ adalah vektor baris yang berisi bobot $\{\beta_1, \dots, \beta_n\}$. Ini adalah bobot untuk **Depth-connections** (mengatur seberapa besar informasi residual yang dipertahankan).
*   $\mathbf{A}_m$ adalah vektor kolom yang berisi bobot $\{\alpha_{1,0}, \dots, \alpha_{n,0}\}$.
*   $\mathbf{A}_r$ adalah sub-matriks persegi berukuran $n \times n$ yang berisi bobot $\alpha_{i,j}$. Gabungan dari $\mathbf{A}_m$ dan $\mathbf{A}_r$ ini bertanggung jawab atas **Width-connections** (mengatur pertukaran silang antar vektor tersembunyi).

Mari kita tinjau sebuah lapisan (layer) jaringan $\mathcal{T}$. Lapisan ini mengintegrasikan lapisan-lapisan *self-attention* dan jaringan *feed-forward* konvensional di dalam sebuah transformer. Keluaran (output) dari blok HC, yang dilambangkan dengan $\hat{\mathbf{H}}$, dapat diformulasikan secara sederhana sebagai berikut:

$$ \hat{\mathbf{H}} = \mathcal{HC}(\mathcal{T}, \mathbf{H}) = \mathbf{B}^\top \mathcal{T}(\mathbf{H}^\top \mathbf{A}_m)^\top + \mathbf{A}_r^\top \mathbf{H} \quad \text{(2)} $$

**Penjelasan Rumus (2):**
*   $\mathbf{H}^\top \mathbf{A}_m$: Pertama, kita melakukan penjumlahan berbobot pada matriks input $\mathbf{H}$ menggunakan vektor bobot $\mathbf{A}_m$. Hasilnya adalah vektor tunggal yang menjadi masukan murni bagi lapisan $\mathcal{T}$.
*   $\mathcal{T}(\dots)$: Fungsi lapisan jaringan saraf asli (misal: layer Self-Attention).
*   $\mathbf{B}^\top \mathcal{T}(\dots)^\top$: Keluaran dari layer $\mathcal{T}$ kemudian dikalikan dengan bobot *depth-connection* $\mathbf{B}$ untuk menentukan seberapa besar hasil komputasi layer ini yang akan disuntikkan kembali ke dalam masing-masing aliran *hidden vector*.
*   $\mathbf{A}_r^\top \mathbf{H}$: Ini adalah operasi *width-connection* murni. Memperkalikan matriks input $\mathbf{H}$ dengan sub-matriks $\mathbf{A}_r$ untuk melakukan pencampuran (mixing) silang antar aliran $n$ vektor sebelum dijumlahkan dengan hasil dari $\mathcal{T}$.

Kita menggunakan $\mathbf{A}_m$ sebagai bobot untuk melakukan penjumlahan berbobot (*weighted sum*) pada masukan $\mathbf{H} = \begin{pmatrix} \mathbf{h}_1 & \mathbf{h}_2 & \dots & \mathbf{h}_n \end{pmatrix}^\top$ untuk mendapatkan vektor input tunggal $\mathbf{h}_0$ bagi lapisan $\mathcal{T}$ saat ini, yang rumusnya diberikan oleh:

$$ \mathbf{h}_0^\top = \mathbf{A}_m^\top \mathbf{H} \quad \text{(3)} $$

Sementara itu, sub-matriks $\mathbf{A}_r$ digunakan untuk menghubungkan $\mathbf{H}$ dan memetakannya menjadi matriks hyper hidden perantara $\mathbf{H}'$, seperti yang ditunjukkan di bawah ini:

$$ \mathbf{H}' = \mathbf{A}_r^\top \mathbf{H} \quad \text{(4)} $$

Selanjutnya, keluaran finalnya dirumuskan menjadi:

$$ \hat{\mathbf{H}} = \mathbf{B}^\top (\mathcal{T}\mathbf{h}_0)^\top + \mathbf{H}' \quad \text{(5)} $$

Secara arsitektur, bagian **depth-connections (koneksi kedalaman)** dapat dipisahkan dan diekspresikan sebagai matriks berikut (seperti yang ditunjukkan pada Gambar 2(a)):

$$ DC = \begin{pmatrix} \mathbf{B} \\ \text{diag}(\mathbf{A}_r) \end{pmatrix} = \begin{pmatrix} \beta_1 & \beta_2 & \dots & \beta_n \\ \alpha_{1,1} & \alpha_{2,2} & \dots & \alpha_{n,n} \end{pmatrix} \in \mathbb{R}^{2 \times n} \quad \text{(6)} $$

**Penjelasan Rumus (6):**
*   Matriks $DC$ (Depth Connection) merangkum bobot residual vertikal. 
*   Baris pertama $\mathbf{B}$ merepresentasikan bobot pengali untuk *keluaran* dari layer $\mathcal{T}$ saat ini. 
*   Baris terakhir $\text{diag}(\mathbf{A}_r)$ merepresentasikan bobot pengali untuk *masukan (residual path)*. Notasi $\text{diag}(\dots)$ berarti kita hanya mengambil elemen diagonal utama dari matriks $\mathbf{A}_r$ lalu meratakannya menjadi vektor.

Sementara itu, bagian matriks **width-connections (koneksi lebar)** dapat didefinisikan sebagai berikut (sesuai Gambar 2(b)):

$$ \mathcal{WC} = \begin{pmatrix} \mathbf{A}_m & \mathbf{A}_r \end{pmatrix} \in \mathbb{R}^{n \times (n+1)} \quad \text{(7)} $$

Algoritma teknis yang menggunakan *static hyper-connections* ini dapat dilihat di *Algorithm 1* (di lampiran makalah, tidak disertakan di sini).

---

### 2.2 Dynamic Hyper-Connections (Koneksi Hiper Dinamis)

Pendekatan statis di atas menggunakan matriks bobot yang tetap setelah pelatihan selesai. Namun, untuk membuatnya lebih superior, entri-entri (elemen bobot) di dalam matriks $\mathcal{HC}$ dapat dibuat bergantung secara **dinamis** terhadap vektor masukan $\mathbf{H}$ yang sedang diproses. Representasi matriks dari *dynamic hyper-connections* (DHC) didefinisikan sebagai berikut:

$$ \mathcal{HC}(\mathbf{H}) = \begin{pmatrix} \mathbf{0}_{1 \times 1} & \mathbf{B}(\mathbf{H}) \\ \mathbf{A}_m(\mathbf{H}) & \mathbf{A}_r(\mathbf{H}) \end{pmatrix} \quad \text{(8)} $$

Sama halnya dengan versi statis, jika diberikan sebuah layer $\mathcal{T}$ dan masukan $\mathbf{H}$, kita memperoleh keluaran dari DHC sebagai berikut:

$$ \hat{\mathbf{H}} = \mathcal{HC}(\mathbf{H})(\mathcal{T}, \mathbf{H}) \quad \text{(9)} $$

Dalam praktiknya untuk implementasi pemrograman, para peneliti menggabungkan matriks dinamis dan statis secara bersamaan untuk menghasilkan bentuk akhir dari DHC. 
Parameter-parameter dinamis ini didapatkan melalui sebuah transformasi linier dari input aslinya. Untuk menstabilkan proses pelatihan, peneliti memperkenalkan operasi normalisasi sebelum transformasi linier tersebut, dan mengaplikasikan fungsi aktivasi $\tanh$ (Hyperbolic Tangent) setelahnya, dan kemudian menskalakannya (*scaling*) menggunakan sebuah faktor inisialisasi yang nilainya sangat kecil dan dapat dipelajari. 

Persamaan-persamaan matematis berikut menjabarkan secara rinci bagaimana matriks parameter dinamis tersebut dihitung pada saat *runtime*:

$$ \overline{\mathbf{H}} = \text{norm}(\mathbf{H}) \quad \text{(10)} $$
$$ \mathbf{B}(\mathbf{H}) = s_\beta \circ \tanh(\overline{\mathbf{H}} \mathbf{W}_\beta)^\top + \mathbf{B} \in \mathbb{R}^{1 \times n} \quad \text{(11)} $$
$$ \mathbf{A}_m(\mathbf{H}) = s_\alpha \circ \tanh(\overline{\mathbf{H}} \mathbf{W}_m) + \mathbf{A}_m \in \mathbb{R}^{n \times 1} \quad \text{(12)} $$
$$ \mathbf{A}_r(\mathbf{H}) = s_\alpha \circ \tanh(\overline{\mathbf{H}} \mathbf{W}_r) + \mathbf{A}_r \in \mathbb{R}^{n \times n} \quad \text{(13)} $$

**Eksplorasi Mendalam Rumus-Rumus DHC (10-13):**
*   **Persamaan (10):** Matriks input hyper $\mathbf{H}$ dinormalisasi terlebih dahulu menjadi $\overline{\mathbf{H}}$ untuk menjaga stabilitas numerik.
*   **Persamaan (11), (12), (13):** Memiliki struktur formula yang persis identik. Mari kita bedah salah satunya, misalnya Persamaan (11) untuk menghitung bobot *depth* dinamis $\mathbf{B}(\mathbf{H})$.
    *   $\overline{\mathbf{H}} \mathbf{W}_\beta$: Input yang sudah dinormalisasi dikalikan secara matriks dengan matriks bobot yang dapat dilatih $\mathbf{W}_\beta$. Ini adalah transformasi linier untuk "meraba" pola input.
    *   $\tanh(\dots)$: Hasil transformasi linier dimasukkan ke fungsi aktivasi non-linier Tangen Hiperbolik, yang akan memampatkan nilainya ke rentang keras $[-1, 1]$.
    *   $s_\beta \circ \dots$: Hasil aktivasi kemudian dikalikan secara elemen-demi-elemen (Hadamard product, disimbolkan dengan $\circ$) dengan parameter skalar $s_\beta$ yang nilainya dinisialisasi sangat kecil. Gunanya agar adaptasi dinamis ini tidak merusak kestabilan jaringan di awal masa pelatihan.
    *   $+ \mathbf{B}$: Terakhir, nilai adaptasi dinamis ini dijumlahkan/ditambahkan dengan parameter statis asli $\mathbf{B}$ yang bersifat konstan. Jadi, parameter akhirnya adalah perpaduan antara memori statis jangka panjang + refleks dinamis jangka pendek terhadap masukan kata yang sedang diproses!

Eksperimen para peneliti di Subbab 4 mendemonstrasikan bahwa *dynamic hyper-connections* secara telak mengungguli *static hyper-connections* dalam tugas-tugas pemodelan bahasa. Implementasi menggunakan *framework* PyTorch untuk kedua varian (statis dan dinamis) telah dirinci pada Algoritma 2 dan 3 di lampiran makalah aslinya.

---

### 2.3 Inisialisasi (*Initialization*)

Untuk memastikan bahwa pada titik mula (Langkah 0/Epoch 0) hiper-koneksi bertindak persis sama ekuivalen dengan arsitektur **Pre-Norm residual connections** konvensional, para peneliti mengadopsi strategi inisialisasi yang spesifik. 

Matriks parameter dinamis $\mathbf{W}_\beta$, $\mathbf{W}_m$, dan $\mathbf{W}_r$ pada Persamaan 11, 12, dan 13 diinisialisasi sepenuhnya ke nilai $0$ (nol mutlak). Sementara itu, matriks statis dasar diinisialisasi dengan struktur sebagai berikut:

$$ \begin{pmatrix} \mathbf{0}_{1 \times 1} & \mathbf{B}^k \\ \mathbf{A}_m^k & \mathbf{A}_r^k \end{pmatrix} = \begin{pmatrix} \mathbf{0}_{1 \times 1} & \mathbf{1}_{1 \times n} \\ \mathbf{e}_{k \bmod n} & \mathbf{e}_{n \times n} \end{pmatrix} \quad \text{(14)} $$

**Penjelasan Rumus (14):**
*   $k$: Indeks lapisan saat ini.
*   $\mathbf{1}_{1 \times n}$: Matriks yang isinya murni angka 1 semua. Artinya di awal *training*, matriks $\mathbf{B}$ (pengali untuk output Layer $\mathcal{T}$) langsung menerima dan menyalurkan 100% informasi keluaran layer.
*   $\mathbf{e}_{n \times n}$: Matriks Identitas berukuran $n \times n$ (Diagonalnya 1, sisanya 0). Artinya di awal, matriks $\mathbf{A}_r$ tidak melakukan pencampuran silang antar aliran vektor (100% passthrough).
*   $\mathbf{e}_{k \bmod n}$: Adalah trik inisialisasi. Tanda $\bmod$ adalah operasi modulo (sisa bagi). Ini menugaskan agar bobot masukan $\mathbf{A}_m$ secara bergantian mengambil data dari indeks kolom/aliran yang berbeda tergantung pada indeks layernya. Ini memastikan keragaman asupan informasi di awal pelatihan.

---

## 3. Mengapa Menggunakan Hyper-Connections? (*Why Hyper-Connections*)

Pada bagian ini, para peneliti menjelaskan landasan konseptual/rasional di balik penciptaan hyper-connections. Mereka mengeksplorasi bagaimana varian dari koneksi residual (Pre-Norm dan Post-Norm) secara fundamental dapat dipandang sebagai bentuk *hyper-connections* khusus yang **kaku dan tidak dapat dilatih (*non-trainable*)**. 

Selanjutnya, mereka memperkenalkan konsep **dualitas sekuensial-paralel (*sequential-parallel duality*)**, yang mendemonstrasikan bagaimana hyper-connections dapat secara dinamis mengoptimalkan susunan arsitektur lapisan-lapisan demi meningkatkan kinerja jaringan saraf secara drastis. Analisis visual dari fenomena hiper-koneksi melalui pandangan terlipat (unfolded view) didiskusikan di Subbab 4.5.

### 3.1 Koneksi Residual sebagai Hyper-Connections yang Tidak Dapat Dilatih

Dalam pembuktian teoretisnya, struktur residual konvensional Pre-Norm dan Post-Norm ternyata dapat direpresentasikan ke dalam format matriks hyper-connections dengan tingkat ekspansi vektor $n = 1$:

$$ \mathcal{HC}_{\text{PreNorm}} = \begin{pmatrix} 0 & 1 \\ 1 & 1 \end{pmatrix} \quad \text{(15)} $$

$$ \mathcal{HC}_{\text{PostNorm}} = \begin{pmatrix} 0 & \frac{1}{\sqrt{\sigma_i^2 + \sigma_o^2 + 2\sigma_{io}}} \\ 1 & \frac{1}{\sqrt{\sigma_i^2 + \sigma_o^2 + 2\sigma_{io}}} \end{pmatrix} \quad \text{(16)} $$

**Penjelasan Rumus (15) & (16):**
*   $\sigma_i$ dan $\sigma_o$ merepresentasikan nilai standar deviasi dari masukan dan keluaran dari lapisan jaringan saraf tersebut, sedangkan $\sigma_{io}$ adalah nilai kovariansi antara keduanya.
*   Perhatikan bentuk **Matriks Pre-Norm (Persamaan 15):** Ini adalah matriks berukuran $2 \times 2$. Bagian segitiga kanan bawah semuanya diisi dengan konstanta mati angka $1$, dan sisanya adalah angka $0$ murni. Tidak ada yang bisa diubah oleh proses *gradient descent*. 
*   Perhatikan bentuk **Matriks Post-Norm (Persamaan 16):** Di sini, bobot koneksinya bukanlah konstanta, melainkan 100% didikte (bergantung) pada perhitungan variansi data statistika yang sedang lewat di dalam layer. 

Oleh karena itu, matriks hyper-connection untuk kedua arsitektur Transformer tradisional ini secara absolut adalah **non-trainable (tidak dapat dipelajari/dilatih)**. 

Di dalam riset fundamental ini, para peneliti mengusulkan Hyper-connections sejati yang berukuran $(n+1) \times (n+1)$, di mana seluruh bobot di dalamnya **adalah parameter murni yang dapat dilatih/diperbarui (*trainable*)** atau bahkan dapat diprediksi secara dinamis berdasarkan sinyal masukan datanya. Derivasi (penurunan rumus) lengkap dari bukti matematis ini disediakan di Lampiran G (tidak disertakan).

---

### 3.2 Dualitas Sekuensial-Paralel (*Sequential-Parallel Duality*)

Diberikan serangkaian modul-modul jaringan saraf tiruan (lapisan), kita sebagai arsitek AI memiliki pilihan mendasar untuk menyusun mereka secara **sekuensial (berurutan seperti gerbong kereta)** atau secara **paralel (berdampingan seperti jalur tol)**. 

Keajaiban dari arsitektur hyper-connections adalah kemampuannya menawarkan suatu mekanisme pendekatan pembelajaran otomatis yang dapat **mempelajari dan menyusun ulang (rearrange)** susunan lapisan-lapisan tersebut menjadi suatu konfigurasi arsitektur baru yang merupakan peleburan (blending) dari susunan sekuensial dan paralel sekaligus.

---
### Analisis Gambar 4: Susunan Sekuensial dan Paralel dari Hyper-Connections dengan n = 2

*(Deskripsi Gambar: Terdapat dua diagram kotak-kotak layer jaringan)*
*   **(a) Sequential Arrangement:** Ini adalah skema di mana matriks DHC kebetulan belajar menjadi identitas lurus. Layer 1 beroperasi penuh. Hasilnya dilempar menjadi masukan (input) penuh bagi Layer 2 di atasnya. Jalur *skip connection* tetap lurus melaluinya.
*   **(b) Parallel Arrangement:** Ini adalah skema di mana matriks DHC memecah jalur logikanya. Masukan utama dikirim ke Layer 1 DAN Layer 2 secara bersamaan. Kedua Layer tersebut melakukan kalkulasi secara independen tanpa saling menunggu. Pada akhirnya, keluaran dari Layer 1 dan Layer 2 digabungkan kembali di tahap atas menggunakan operasi penjumlahan (+).
*   **Kesimpulan Akademis Gambar 4:** Matriks DHC adalah sebuah saklar kereta api dimensi tinggi. Ia bisa menjadi mode seri, bisa menjadi mode paralel. Karena parameter matriks ini dilatih dengan *Gradient Descent*, maka matriks DHC akan selalu mencari topologi "jalur kereta" yang menghasilkan kerugian terendah (kalkulasi optimal) pada data yang ia kerjakan.
---

Tanpa kehilangan nilai keumumannya, mari kita tetapkan tingkat ekspansi $n = 2$. Jika hyper-connections mempelajari bobotnya sedemikian rupa sehingga matriks bobotnya menjadi seperti matriks di bawah ini, maka secara arsitektur, lapisan-lapisan jaringan saraf secara efektif akan tersusun **secara sekuensial (berurutan)**:

$$ \mathcal{HC} = \begin{pmatrix} 0 & 1 & 1 \\ 1 & 1 & 0 \\ 0 & 0 & 1 \end{pmatrix} \quad \text{(17)} $$

Dalam kasus matriks identitas degeneratif ini, koneksi *depth* merosot wujudnya menjadi sekadar koneksi residual konvensional kuno, seperti yang diilustrasikan pada Gambar 4 (a).

Namun, bagaimana jika hyper-connections untuk lapisan bernomor indeks **ganjil** (odd) dan bernomor indeks **genap** (even) mempelajari matriks yang berbeda? Jika matriks yang dipelajari menjadi seperti di bawah ini, maka lapisan-lapisan jaringan saraf akan **tersusun secara paralel setiap dua lapisan berturut-turut**, persis menyerupai susunan blok transformer paralel (*parallel transformer blocks*) pada paper Wang (2021), seperti yang diilustrasikan pada Gambar 4 (b). Penurunan matematis lengkap dari teori ini ada di Lampiran H.

$$ \mathcal{HC}_{\text{odd}} = \begin{pmatrix} 0 & 1 & 0 \\ 1 & 1 & 1 \\ 1 & 1 & 1 \end{pmatrix} \quad \text{(18)} $$

$$ \mathcal{HC}_{\text{even}} = \begin{pmatrix} 0 & 0 & 1 \\ 0 & 1 & 0 \\ 1 & 0 & 1 \end{pmatrix} \quad \text{(19)} $$

Oleh karena itu, membiarkan mesin mempelajari wujud matriks hyper-connection dalam berbagai bentuk bebas akan dapat menciptakan sebuah pengaturan lapisan arsitektur abstrak yang **jauh melampaui kemampuan konfigurasi sekuensial atau paralel tradisional**. Ia menciptakan struktur "campuran-halus" (*soft-mixture*) atau bahkan menciptakan "pengaturan dinamis" *(dynamic arrangement)*. 

Untuk tipe *static hyper-connections*, susunan lapisan arsitektur di dalam jaringan akan dipatenkan (ditetapkan) setelah proses pelatihan selesai. Sebagai kontras, tipe **dynamic hyper-connections** akan mengizinkan susunan topologi lapisannya untuk terus *berubah wujud beradaptasi secara dinamis secara seketika* bergantung kepada token (kata) spesifik apa yang saat itu sedang ia baca. Sebuah perilaku kognisi mesin yang sangat *mind-blowing*.

---

## 4. Hasil Eksperimen (*Results*)

---
### Analisis Gambar 5: Perbandingan Kurva Kerugian Pelatihan untuk Ekspansi (n) yang Berbeda

*(Deskripsi Gambar: Terdapat dua grafik kurva garis yang berdampingan)*
*   **Grafik Kiri (DHC):** Menunjukkan kurva *Training Loss* untuk varian DHC (Dinamis). Sumbu X adalah miliar token (100 hingga 500). Sumbu Y adalah Loss (2.40 hingga 2.60). Terdapat 5 kurva.
    *   Kurva Merah: OLMo-1B-baseline (Koneksi konvensional lama). Terlihat paling atas (paling jelek nilainya).
    *   Kurva Hijau Tua (DHCx1): Tumpang tindih dekat merah.
    *   Kurva Hijau Muda (DHCx2): Menjauh ke bawah.
    *   Kurva Ungu (DHCx4) dan Kurva Oranye (DHCx8): Berada paling dasar (performa terbaik, Loss paling kecil).
*   **Grafik Kanan (DHC W/O tanh):** Menunjukkan kurva DHC yang dieksperimenkan tanpa menggunakan aktivasi *tanh*. Secara visual bentuk penurunannya persis sama: DHCx4 dan DHCx8 merajai di dasar paling bawah.
*   **Kesimpulan Akademis Gambar 5:** Grafik ini menjustifikasi bahwa nilai $n$ (tingkat ekspansi hidden vector) sangat penting. Nilai $n=1$ nyaris tidak berguna. Namun begitu kita pecah menjadi 4 jalur ($n=4$), jaringan mendapat kebebasan komputasional yang tinggi untuk memformulasikan pola data (seperti ditunjukkan di turunnya garis ungu). Membengkakkan $n$ hingga angka 8 juga bermanfaat, namun *margin of returns* (tingkat keuntungannya) mulai mengecil dan menipis dibandingkan dengan loncatan dari 2 ke 4.
---

### Tabel 1: Studi Ablasi mengenai Tingkat Ekspansi ($n$) pada Pelatihan 500 Miliar Token

| Metode | V2 Eval Loss $\downarrow$ | V2 Eval PPL $\downarrow$ | V3 Eval Loss $\downarrow$ | V3 Eval PPL $\downarrow$ | Rata-rata Akurasi Hilir (Down Stream) $\uparrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| OLMo-1B (Baseline) | 2.811 | 18.023 | 2.544 | 14.229 | 62.5% |
| **Tanpa $\tanh$ (W/O tanh):** | | | | | |
| OLMo-1B-DHC$\times 1$ | 2.822 | 18.270 | 2.556 | 14.428 | 62.3% |
| OLMo-1B-DHC$\times 2$ | 2.792 | 17.663 | 2.537 | 14.033 | 63.8% |
| OLMo-1B-DHC$\times 4$ | 2.779 | 17.451 | 2.516 | 13.844 | **64.4%** |
| OLMo-1B-DHC$\times 8$ | **2.777** | **17.425** | **2.514** | **13.819** | 63.8% |
| **Dengan $\tanh$ (Varian Normal):** | | | | | |
| OLMo-1B-DHC$\times 1$ | 2.819 | 18.125 | 2.556 | 14.418 | 62.3% |
| OLMo-1B-DHC$\times 2$ | 2.802 | 17.950 | 2.534 | 14.114 | 63.0% |
| OLMo-1B-DHC$\times 4$ | 2.781 | 17.509 | 2.514 | 13.826 | **63.8%** |
| OLMo-1B-DHC$\times 8$ | **2.778** | **17.445** | 2.516 | 13.843 | 62.8% |

*(Catatan Dosen: Simbol $\downarrow$ berarti angka semakin kecil semakin baik. $\uparrow$ berarti semakin besar semakin baik. PPL adalah Perplexity, indikator seberapa "bingung" model membaca teks. Tabel ini memberikan konfirmasi statistik atas visualisasi Gambar 5. Menariknya, pada pengujian empiris evaluasi hilir (HellaSwag, ARC, dll), konfigurasi arsitektur **DHCx4 tanpa fungsi $\tanh$** keluar sebagai juara absolut dengan skor Akurasi tertinggi rata-rata **64.4%**).*

Para peneliti pada prinsipnya melakukan serangkaian percobaan yang difokuskan pada tahap pra-pelatihan (*pre-training*) LLM. Eksperimen mencakup model berarsitektur Padat (*Dense*) maupun arsitektur *Mixture-of-Experts* (MoE) (Shazeer et al., 2017), dan juga diperluas hingga ke wilayah tugas-tugas generasi dan klasifikasi visual (Computer Vision). Namun karena batasan halaman, eksperimen visi komputer direlegasikan ke Lampiran E.

**Pengaturan Eksperimen (*Experiment Settings*):**
Para peneliti menggunakan metodologi standar (experimental setup) yang dirancang oleh arsitektur dasar **OLMo** (Groeneveld et al., 2024) untuk pengujian model Padat, serta metodologi OLMoE (Muennighoff et al., 2024) untuk model MoE. 
*   **Untuk model Padat (*Dense models*):** Dataset pelatihan yang dijejalkan adalah `dolmap-v1.5-sample`. Studi ablasi (pembongkaran parameter) dikerjakan pada spesifikasi kelas model berukuran 1 Miliar parameter (1B), sementara efektivitas puncak metode dinilai pada kelas kompetisi berat model berukuran 7 Miliar parameter (7B). 
*   **Untuk model MoE:** Pelatihan dikerjakan pada fondasi model `OLMoE-1B-7B` (versi dengan maupun tanpa hyper-connections), mengunyah dataset `OLMOE-MIX`. Model MoE ini secara arsitektur bertotal 7B parameter, namun hanya **mengaktifkan 1.3B parameter** pada setiap kali komputasi forward-pass. 
*   Seluruh eksperimen besar ini dilatih tanpa ampun pada asupan **500 Miliar Token**.

**Implementasi Teknis (*Implementation*):**
Peneliti mematenkan konfigurasi bawaan pelatihannya dan hanya mensubstitusi/mengganti sambungan blok residual usangnya dengan modul hyper-connections (HC). 
Satu trik matematis vital dilakukan di sini: Karena aliran vektor *hyper hidden* di blok Transformer paling akhir pada ujungnya dijumlahkan (*summed*) secara total, peneliti harus memproteksi skala distribusi angka statistiknya. Mereka harus menjamin bahwa deviasi standar (standard deviation / `std`) dari tensor output (sebelum disuntikkan ke layer *Layernorm final* dan blok *Unembedding*) nilainya tetap jinak dan konsisten sama dengan tensor aslinya (sebelum diganti HC). 
Saat inisialisasi di awal waktu (Epoch 0), mereka melakukan normalisasi skalar `std` pada setiap bobot keluaran matriks di semua lapisan (termasuk pada lapisan linier kedua dari struktur *feedforward network/FFN* dan matriks proyektor keluaran dari arsitektur mekanik Atensi) dengan jalan **mengalikannya dengan faktor pembagi $\sqrt{n}$**. (Di mana $n$ merepresentasikan nilai *expansion rate* / jumlah cabang kopian vektor).
Trik ini sangat krusial agar *backpropagation* di masa inkubasi pelatihan awal tidak meledak (*Gradient Exploding*). Parameter matriks ekstra serta beban irisan komputasi (*overhead*) matematika tambahan yang disisipkan oleh instalasi arsitektur Hyper-connections sangat minim hingga statusnya dianggap *negligible* (diabaikan secara statistik), rujuk Tabel 7 dan 8.

**Metrik Penilaian (*Metrics*):**
Berpedoman taat pada konstitusi metodologi OLMo, mereka menyajikan laporan evaluasi formal pada besaran nilai kebingungan (*Perplexity* / PPL) dan Kurva Loss pada *Validation Sets* kategori V2 dan V3, diiringi pencatatan kalkulasi performa uji buta *Zero-shot* pada arena tes hilir/downstream. 

---

### 4.1 Studi Ablasi (*Ablation Study*)

Pada studi ablasi ini, konfigurasi asali (default method) yang dijadikan rujukan adalah **dynamic hyperconnections (DHC)** dipatok pada ukuran rentang ekspansi $n = 4$ diserta fungsional aktivasi *tanh*. Varian eksperimental Statis murni dilambangkan embel-embel sufiks -SHC.

Berangkat pada Tabel 1 dan kurva visual Gambar 5. Terdapat pola konsensus saintifik tajam di sini: Apabila kita hanya menahan *expansion rate* pada titik rasio $n = 1$, kinerja DHC tersungkur kalah menyedihkan bertekuk lutut di bawah standar *baseline*.
Akan tetapi, manakala arsitek membuka keran saluran memori ekspansi $n > 1$, DHC bangun dan merajalela membantai skor kinerja *baseline*. Kinerja optimal *sweet spot* ditemukan di titik ordinat $n = 4$. Usaha memompa rakus nilai ekspansi menuju horizon $n = 8$ sekadar menyumbangkan tambahan amunisi manfaat yang teramat marjinal (minimal).

*Fakta mengejutkan:* Varian modifikasi `OLMo-1B-DHCx8 W/O tanh` (tanpa pembatasan fungsi tanh) menelurkan nilai keunggulan fantastis mutlak mencabik-cabik evaluasi validasi V2 dan V3, terukur menorehkan pemangkasan defisit loss V2 sebesar **0.034** dan loss V3 sebesar **0.029** mendahului model konvensional baseline. Fenomena memukau lainnya: kurva derajat gradien penurunan (*decline rate*) di masa *training* bagi instalasi DHC ($n \ge 2$) menukik menuruni kawah eksekusi lebih cepat/curam (steeper) dibanding garis kurva *baseline*. Sebagai bonus final, asupan aliran gradien arsitektur DHC lebih tegar dan kokoh mengeliminasi anomali ledakan gradien *Spikes* secara tuntas.

**Koneksi Hiper Statis vs Dinamis (Static vs Dynamic).** 
Mari kita singgah pada Tabel 2 untuk mengeksekusi komparasi ablasi *head-to-head* antara SHC melawan DHC. Kesimpulan sakleknya: *Seluruh* keturunan dinasti Hyper-connection (HC) secara kejam mempermalukan arsitektur Residual (baseline). Pada rasio penyetelan $n=2$, unjuk gigi peningkatan metrik SHC maupun DHC bernilai setara/mirip. Tetapi tunggu dulu, ketika keran mesin diputar ke rasio $n=4$, kognisi *Dynamic* (DHC) secara eksponensial melebarkan jurang superioritas mendepak varian *Static* (SHC). Hal ini karena DHC secara lincah mampu menata ulang alur matriks sirkuit kabel otaknya secara *real-time* bergantung kata (token) mematikan apa yang sedang dilemparkan masuk ke badannya.

**Signifikansi Matriks B dan $\mathcal{WC}$.** 
(Lihat Tabel 3 - Tidak ditampilkan detail). Tim AI ByteDance menguak tabir rahasia: Melakukan pembekuan paksa (tidak di-training/beku) terhadap komponen jalur komputasional matriks $\mathcal{WC}$ (Width Connections) akan mencetuskan bencana katastropik runtuhnya akal jaringan. *V2 loss* meledak naik parah di angka **0.021** (merugikan). Hal sebaliknya tidak terjadi jika komponen Matriks B (Depth/Residual factor) yang dibekukan secara eksperimental (efek keburukannya nyaris tidak kentara). Fakta uji klinis ini mendaulat keberadaan nilai parameter matriks Lebar $\mathcal{WC}$ (Pencampuran aliran ruang vektor antar sel horizontal) serta vektor matriks Baris $\mathbf{B}$ merupakan urat nadi vital yang haram hukumnya untuk dibiarkan mati/statis (un-trainable) di sistem ini.

---

### 4.2 Perbandingan dengan Karya Terkait (*Comparison with Related Works*)

Para peneliti merasa berkewajiban merakit dan mengujicobakan arsitektur saingan di era mutakhir ini, secara spesifik: sistem **Altup** (Baykal et al., 2024) dan arsitektur ganda **ResiDual** (Xie et al., 2023) menggunakan kerangka ekosistem eksperimental model bahasa OLMo.

1.  **Arsitektur Altup:** Dimotivasi dari obsesi akademik menekan laju kalori komputasi seminim mungkin (low computation cost) namun dengan ambisi egois melebarkan matriks vektor (widen hidden dimension). Altup mendiskon biaya ini dengan trik "jalan pintas" melontarkan sebagian irisan vektor saja menembus masuk diolah ke rahim balok Transformer.
2.  **Arsitektur ResiDual:** Karya brilian ini digadang-gadang sukses mengawinkan *Pre-Norm* dan *Post-Norm* layaknya dua aliran pusaran air membelah lautan kembar (two-stream style). 

Kedua rival raksasa akademis ini pada hakikat asalnya memang murni memperlebar ukuran perut (hidden size) matriks dimensi sebanyak parameter $n$ kali kelipatan berongkos memori pelit (negligible), di mana ResiDual secara statis dipaten membelah $n=2$. Memastikan asas pengadilan yang fair, tim ByteDance mematok HC miliknya juga murni pada kelas pembagian $n=2$. 

Kabar buruk bagi para pesaingnya: Walau Altup dan ResiDual memukau menyalip lari di etape minggu-minggu masa pra-pelatihan perdana awal, nafas intelijensinya lekas tersengal dan pelan tapi pasti dihempaskan dan dilampaui kinerjanya dibungkam telak oleh rejim model OLMo *Baseline*, terbukti tersurat telanjang lewat bukti forensik numerik di Tabel 4 serta persilangan grafik sadis kurva *loss* Gambar 15 (Apendiks).

---

### 4.3 Eksperimen Pada Model Skala Besar (7B Models)

Para peneliti mengevaluasi keefektifan koneksi-hiper pada model 7B. Mereka melatih sebuah model menggunakan DHC dengan tingkat ekspansi 4, yang diberi nama **OLMo-7B-DHC$\times 4$**. 

Berdasarkan Tabel 5, OLMo-7B-DHC$\times 4$ secara meyakinkan dan signifikan mengalahkan model OLMo-7B asli (baseline) di semua metrik rata-rata. Pada evaluasi set data V2, OLMo-7B-DHC$\times 4$ menorehkan peningkatan kualitas (reduksi error) yang masif: turunnya nilai Loss sebesar **0.022** dan PPL sebesar **0.293**. Selanjutnya, rata-rata skor pada tolok ukur pengujian performa pemahaman hilir (downstream benchmark) melompat menjadi **0.710**, membungkam baseline (0.701).

---
### Analisis Gambar 6: Performa Unggul Model 7B-DHC$\times 4$

*(Deskripsi Gambar: Terdapat empat grafik yang membandingkan kurva pelatihan model 7B)*
*   **Grafik (1) - Training Loss & (2) C4-en Loss:** Menunjukkan kurva kerugian saat latihan. Kurva merah (OLMo-7B baseline) memiliki ciri khas buruk: dipenuhi dengan lonjakan atau duri-duri tajam (*spikes*) secara terus-menerus. Kurva biru (OLMo-7B-DHCx4) meluncur stabil, mulus, dan sama sekali tanpa lonjakan *error*. Kurva biru juga selalu berada di bawah (lebih pintar) dari merah.
*   **Grafik (3) - HellaSwag Acc & (4) SciQ Acc:** Menunjukkan akurasi tes uji logika di akhir. Kurva biru stabil merangkak naik dan secara visual menciptakan jarak aman (margin) di atas kurva merah *baseline*.

**Kesimpulan Akademis Gambar 6:** Pada skala 7 Miliar parameter, arsitektur dasar Transformer (Pre-Norm) konvensional mulai menunjukkan gejala penyakit parah: ketidakstabilan numerik yang dibuktikan dengan frekuensi kemunculan *spikes* (ledakan gradien) yang tidak wajar. Penemuan yang brilian di sini adalah: Modul DHCx4 tidak hanya membuat model jauh lebih cerdas (Akurasi naik dan Loss turun secara persisten tanpa menunjukkan tanda-tanda kelelahan komputasi (*diminishing returns*) di token 400B+), tetapi yang paling penting: DHC bertindak layaknya *Shock Absorber* (peredam kejut) algoritmik murni yang **membasmi kuman *spikes* (ketidakstabilan) dari keseluruhan siklus pelatihan**, menjamin keamanan biaya *training* tanpa takut terhenti karena ledakan loss yang mematikan program.
---

### 4.4 Eksperimen Pada Model MoE (*Mixture-of-Experts*)

Para peneliti menguji keampuhan *hyper-connections* pada keluarga model MoE (*Mixture-of-Experts*). Mereka melatih ulang pondasi kokoh `OLMoE-1B-7B` sebagai kerangka acuan dan melepaskan agen AI canggih miliknya yang dicangkok dengan implementasi **Dynamic Hyper-Connections (DHC) ekspansi $n = 4$**. Proses operasi cangkok ini membabat habis urat saraf koneksi residual konvensionalnya yang sudah lapuk. 

Fakta mencengangkan hadir di layar monitor (Tabel 5 & 6, dan Gambar 9). Sistem koneksi hiper (DHC) meluluhlantakkan arsitektur residual klasik pada nyaris seluruh matriks parameter penentu (*metrics*). 

Kecerdasan absolut usulan ini tergambar gamblang pada metrik-metrik inti di mana temuan metode ini **hanya mengonsumsi biaya asupan makanan "SETENGAH" (50%) porsi data training** untuk mencapai kesetaraan prestasi otak kognitif dari batas limit yang dicapai susah payah oleh agen AI *Baseline*! Tabel 6 merangkum data kemenangan telak DHC pada pengujian akal sehat: penambahan akurasi sebanyak **6 poin murni** mempecundangi rekor ARC-Challenge dan mencakar tambahan 1.2 poin di papan MMLU.

---

### 4.5 Analisis Visualisasi (*Visualization Analysis*)

Pada bab pemungkas ini, para pakar AI secara manual mengoperasi sirkuit memori arsitektur yang terlatih (membedah *learned hyper-connection weights*) untuk meneliti dan memvisualisasikan bagaimana limbah representasi *output* pada lapisan sebelumnya mengkontaminasi dan menyumbang asupan (*contribution*) kepada susunan sel lapisan berikutnya di lantai atas.
Sirkuit ini digambar secara forensik mengubah hiper koneksi dimensi sempit menjadi koneksi padat menembus dimensi matriks kepadatan lapisan *(dense connections cross layers)*. Konsep matematis dari visualisasi matriks ini mengizinkan tim ilmuwan melihat kontribusi absolut bobot layer-$j$ terhadap layer-$k$. (Lihat Persamaan matematis no 20).

Tim ahli memonitor *checkpoint* pada umur pelatihan `500B` token. Disodorkan baris kalimat uji tes rambang ke depan mesin (*forward random validation text*) guna menarik sampel detak jantung matriks parameter *dynamic hyper-connection weights* seketika. Pola peta matriks yang dicitrakan tergambar pada kanvas spektakuler **Gambar 7**.

---
### Analisis Gambar 7: Peta Korelasi Panas (Heatmap) Matriks Visualisasi Hyper-Connections

*(Deskripsi Gambar: Terdapat sebaris grafik heatmap segitiga matriks ukuran 32x32)*
*   **Sumbu X dan Y:** Menunjukkan indeks lapisan (Layer 0 hingga 32). Warna merepresentasikan korelasi bobot kontribusi (Merah berarti kuat/positif $\sim 1.0$, Biru berarti negatif $\sim -1.0$).
*   **Kotak "Hyper-Connection" (Kiri Pertama):** Memiliki bentuk **pola huruf V terbalik atau Lambda ($\Lambda$)**. Pada bagian diagonalnya (koneksi antar lapisan yang bersebelahan) warnanya sangat merah pekat. Lebih unik lagi, lapisan dasar/bawah (Layer 0, 1, 2) terlihat diakses terus-menerus oleh lapisan yang letaknya sangat tinggi (Layer 20-30), terlihat dari sisi kiri matriks yang membentuk kolom vertikal memerah. Terlihat juga pola zigzag bergerigi (*jagged*).
*   **Kotak "Post-Norm" (Kedua):** Bentuknya memudar ke arah luar (garis merah hanya kuat di tengah diagonal). Lapisan atas melupakan lapisan bawah.
*   **Kotak "Pre-Norm" (Ketiga):** Warnanya nyaris merah 100% mengisi penuh separuh segitiga bawah. Semua layer bawah disuntik mati secara merata ke layer atas (Penyebab *Representation Collapse*).

**Kesimpulan Akademis Forensik Gambar 7:**
Ini adalah detektif algoritmik. Kita bisa membaca perilaku belajar mesin:
1.  **Pola Huruf Lambda ($\Lambda$):** Model Hyper-Connections terbukti mengombinasikan sifat *Pre-Norm* (sisi vertikal merah di kiri) di mana layer dasar terus "diingat" dan diserap, dan sifat *Post-Norm* (merah pudar memanjang di area diagonal) di mana layer atas fokus pada layer pendahulu di dekatnya. Mesin pintar ini secara otonom mengawinkan (*free mixture*) dua arsitektur yang sebelumnya dirancang mati oleh tangan manusia!
2.  **Kematian Word Embedding:** Kolom paling awal (Word Embedding murni) diserap kuat oleh *semua* layer kecuali layer paling puncak (yang dipakai memprediksi kata). Layer puncak belajar secara otonom mematikan/menendang data *Word Embedding* karena terbukti data mentah merusak intuisi keakuratan menebak kata di penghujung eksekusi algoritma (*layer 32*).
3.  **Lahirnya Sirkuit Transformer Paralel Otonom:** Lihat pola gigi gergaji kecil (*jagged*) di grafik. Di titik tertentu (misal layer 11 dan 12), koneksi silangnya mendekati 0. Artinya, model ini dengan sadar diri *membongkar* sistem sekuensial "tunggu-menunggu" dan merakit ulang struktur otaknya menjadi **Blok Transformer Paralel**! Otak mesin ini mendesain jalan arsitekturnya sendiri.
---

---

## 5. Pekerjaan Terkait (*Related Work*)

Kajian riset ini diposisikan dengan menunggangi pundak pionir-pionir sebelumnya. Transformers (Vaswani et al., 2017) merevolusi wajah pengolahan NLP dan juga meluas dominan ke Visi Komputer. Jantung fundamental dari *framework* ini tertopang di atas punggung *residual connections* sebagai pengalir nafas keabadian kelancaran model komputasi arsitektur dimensi ekstrem super dalam. Inovasi canggih hiper-koneksi (*hyper-connections*) karya tim ini disodorkan sebagai pengganti takdir dari struktur usang *residual connection*, berpotensi mengukir jaminan stabilisasi ritme durasi *training* secara simultan dalam lanskap kecerdasan pemrosesan bahasa maupun rekayasa genetika *Computer Vision*.

Terkait perdebatan klasik abadi polemik sindrom hilangnya detak gradien (*gradient vanishing*) (Bengio et al., 1994) yang dikutuk bersamaan dengan hancur kolapsnya nilai parameter bobot otak (*representation collapse*) (Liu et al., 2020) telah dikupas tuntas sepanjang satu dekade lalu oleh raksasa pendiri AI. Arsitektur lama yang menggabungkan manipulasi *Normalization* (Ba et al., 2016) dan manipulasi tambal-sulam *Residual* (He et al., 2016) bermutasi menghasilkan sub-sekte varian kompromistis *Pre-Norm* dan *Post-Norm* yang memikul mandat dan fokus pengerjaan masalah terpisah di ranahnya masing-masing.

Kendati peradaban AI dijejali sekumpulan metode tambal sulam di atas, dilema absolut "*trade-off* berdarah" (pertukaran ongkos) dari teka-teki memecahkan masalah keseimbangan *gradient vanishing* dengan *representation collapse* dalam kegelapan jaring labirin model bahasa ini tetap bermukim menjadi tantangan horor raksasa bagi saintis dunia. Bersandar di fondasi pemikiran ini, tim peneliti mengorbitkan sebuah mazhab inovatif arsitektur mesin cerdas super canggih: Di mana kecerdasan mekanis Jaringan Saraf mendelegasikan kedaulatan dirinya secara Otonom (*autonomously*) untuk menemukan rumusan porsi ramuan kalkulasi matematika terhebat ("optimal strength of connections") guna menyempurnakan harmoni kestabilan energi lintasan *gradient* sembari melipatgandakan bobot kebrilianan formasi kualitas *representation*.

---

## 6. Kesimpulan (*Conclusion*)

Di penghujung literatur ilmiah ini, para ilmuwan telah secara utuh membongkar-pasang serta mendemonstrasikan kelahiran **hyper-connections** sebagai mahakarya rekayasa alternatif yang membinasakan takdir limitasi kerangka arsitektur *residual connections* warisan era usang *transformers*. Analisis anatomi komprehensif menguak tabir fenomena mempesona: Hyper-connections sama sekali bukan hanya menambal kebocoran kelumpuhan limitasi memori residual; namun lebih mengerikan lagi, model cerdas mekanik perintis ini diizinkan menyetel dan "Membongkar-Pasang Transformasi Struktur Organ Intinya Sendiri Secara Dinamis" (*dynamic adjustments in network architecture*). Evaluasi hasil laboratorium dan uji publik dunia nyata (*Experimental results*) mensahihkan janji gilang-gemilang janji masa depan hiperkonektivitas ini memanen kedigdayaan dalam menumbangkan ragam tugas esoterik terpelik, mulai dari penjejalan ribuan buku saat pra-pelatihan Model Bahasa Raksasa (LLMs), berinkarnasi menulis *image generation* presisi, hingga sukses dalam ranah klasifikasi mutlak dimensi citra (*image classification*).

---
*(Catatan Dosen: Pembahasan materi utama dari makalah "Hyper-Connections" ini selesai di sini. Pahami secara mendalam transisi paradigma dari "Static/Fixed Residual Connections" menuju "Dynamic Matrix Mixing (Hyper-Connections)". Ujian akan mencakup pemahaman teoretis mengenai fenomena Seesaw Effect dan analisis keruntuhan representasi (Representation Collapse) beserta bukti mitigasinya pada metrik Cosine Similarity. Bagian Referensi dan Apendiks dari makalah ini diabaikan sesuai panduan studi).*