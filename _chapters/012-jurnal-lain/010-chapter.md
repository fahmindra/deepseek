---
slug: ch-12-01
title: "Attention Is All You Need"
layout: chapter
---

## Abstrak

Model transduksi sekuens (pemetaan suatu urutan masukan menjadi urutan keluaran) yang paling dominan saat ini didasarkan pada arsitektur Jaringan Saraf Berulang (*Recurrent Neural Networks* / RNN) atau Jaringan Saraf Konvolusional (*Convolutional Neural Networks* / CNN) yang kompleks, yang mencakup sebuah *encoder* (penyandi) dan *decoder* (pemuasandi). Model-model dengan performa terbaik juga menghubungkan *encoder* dan *decoder* tersebut melalui sebuah mekanisme perhatian (*attention mechanism*). 

Dalam kajian ini, para peneliti mengusulkan sebuah arsitektur jaringan baru yang sangat sederhana, yang diberi nama **Transformer**. Arsitektur ini didasarkan **sepenuhnya** pada mekanisme perhatian, sehingga meniadakan kebutuhan akan perulangan (*recurrence*) maupun konvolusi sama sekali. 

Eksperimen yang dilakukan pada dua tugas penerjemahan mesin (*machine translation*) menunjukkan bahwa model ini memiliki kualitas yang jauh lebih superior, sekaligus jauh lebih mudah untuk diparalelisasi, dan membutuhkan waktu pelatihan yang secara signifikan lebih singkat. Model ini berhasil mencapai skor **28.4 BLEU** pada tugas penerjemahan Bahasa Inggris-ke-Jerman WMT 2014, mengungguli hasil-hasil terbaik yang ada sebelumnya, termasuk model *ensemble* (gabungan beberapa model), dengan selisih lebih dari 2 BLEU. 

Pada tugas penerjemahan Bahasa Inggris-ke-Prancis WMT 2014, model ini menetapkan standar pencapaian baru (state-of-the-art) untuk model tunggal dengan skor BLEU sebesar **41.8** setelah dilatih hanya selama 3.5 hari menggunakan delapan unit GPU. Biaya pelatihan ini hanyalah sebagian kecil dari biaya pelatihan model-model terbaik dari literatur yang ada. Lebih lanjut, para peneliti menunjukkan bahwa Transformer mampu melakukan generalisasi dengan sangat baik pada tugas-tugas lain, terbukti dari penerapannya yang sukses pada *English constituency parsing* (penguraian tata bahasa konstituen bahasa Inggris), baik pada kondisi data pelatihan yang besar maupun terbatas.

---

## 1. Pendahuluan (*Introduction*)

Jaringan Saraf Berulang (*Recurrent Neural Networks* / RNN), khususnya jaringan *Long Short-Term Memory* (LSTM) dan *Gated Recurrent Unit* (GRU), telah lama dikukuhkan sebagai pendekatan *state-of-the-art* (paling mutakhir) dalam pemodelan sekuens dan masalah transduksi, seperti pemodelan bahasa (*language modeling*) dan penerjemahan mesin. Sejak saat itu, berbagai upaya terus dilakukan untuk mendobrak batas kemampuan model bahasa berulang dan arsitektur *encoder-decoder*.

**Mengapa RNN memiliki keterbatasan fundamental?**
Model berulang secara tipikal memfaktorkan komputasi di sepanjang posisi simbol dari urutan masukan (*input*) dan keluaran (*output*). Karena posisi-posisi tersebut harus diselaraskan dengan langkah waktu (*time steps*) komputasi, model ini akan menghasilkan sekumpulan status tersembunyi (*hidden states*) $h_t$, yang dihitung sebagai fungsi dari status tersembunyi sebelumnya $h_{t-1}$ dan masukan pada posisi $t$. 

Sifat bawaan yang sangat *sekuensial* (berurutan) ini **mencegah adanya paralelisasi** dalam satu contoh pelatihan (*training example*). Hal ini menjadi masalah kritis terutama ketika berhadapan dengan urutan data yang panjang (seperti dokumen yang panjang), karena keterbatasan memori menghalangi pengelompokan (*batching*) lintas contoh. Meskipun beberapa penelitian terbaru berhasil mencapai peningkatan efisiensi komputasi yang signifikan melalui trik faktorisasi dan komputasi kondisional (yang juga meningkatkan performa model), **kendala fundamental dari komputasi sekuensial tersebut tetap tidak terpecahkan.**

Di sisi lain, **mekanisme perhatian (*attention mechanisms*)** telah menjadi bagian integral dari model pemodelan sekuens dan transduksi yang andal pada berbagai tugas. Mekanisme ini memungkinkan pemodelan ketergantungan (dependensi) tanpa memperdulikan seberapa jauh jarak elemen tersebut di dalam urutan masukan maupun keluaran. Namun, dalam hampir semua kasus, mekanisme perhatian tersebut selalu digunakan **bersamaan dengan jaringan berulang (RNN)**.

Dalam kajian ini, para penulis mengusulkan **Transformer**, sebuah arsitektur model yang sama sekali menghindari penggunaan perulangan (*recurrence*), dan sebagai gantinya hanya mengandalkan mekanisme perhatian untuk menarik dependensi global antara masukan dan keluaran. Model Transformer memungkinkan paralelisasi yang jauh lebih besar dan mampu mencapai kualitas terjemahan tingkat mutakhir (SOTA) hanya dengan dilatih selama dua belas jam menggunakan delapan GPU P100.

---

## 2. Latar Belakang (*Background*)

Tujuan untuk mengurangi komputasi sekuensial juga menjadi dasar dari berbagai model lain seperti *Extended Neural GPU*, *ByteNet*, dan *ConvS2S*. Semua model ini menggunakan **Jaringan Saraf Konvolusional (CNN)** sebagai blok pembangun dasarnya. Mereka menghitung representasi tersembunyi (*hidden representations*) secara paralel untuk semua posisi masukan dan keluaran. 

Namun, pada model-model berbasis CNN ini, jumlah operasi yang dibutuhkan untuk menghubungkan sinyal dari dua posisi masukan atau keluaran sembarang akan **tumbuh/meningkat seiring dengan jarak antar posisi tersebut**—tumbuh secara linear pada ConvS2S dan secara logaritmik pada ByteNet. Pertumbuhan operasi ini membuat model lebih sulit untuk mempelajari dependensi atau hubungan antara posisi yang saling berjauhan. Pada Transformer, masalah ini direduksi menjadi jumlah operasi yang **konstan**, meskipun ada harga yang harus dibayar berupa berkurangnya resolusi efektif karena model merata-ratakan posisi yang diberi bobot perhatian. Namun, efek negatif ini berhasil diatasi dengan menggunakan **Perhatian Multi-Kepala (*Multi-Head Attention*)** yang akan kita bahas pada Subbab 3.2.

**Perhatian-Mandiri (*Self-Attention*)**, yang terkadang disebut *intra-attention*, adalah sebuah mekanisme perhatian yang menghubungkan posisi-posisi yang berbeda dari sebuah urutan yang sama (satu kalimat yang sama) untuk menghitung representasi dari urutan tersebut. *Self-attention* telah sukses digunakan dalam berbagai tugas termasuk pemahaman bacaan (*reading comprehension*), peringkasan ekstraktif/abstraktif, keterikatan tekstual (*textual entailment*), dan pembelajaran representasi kalimat yang tidak bergantung pada tugas tertentu.

Jaringan memori ujung-ke-ujung (*End-to-end memory networks*) didasarkan pada mekanisme perhatian yang berulang, alih-alih perulangan yang diselaraskan dengan urutan, dan terbukti berkinerja baik pada tugas tanya-jawab bahasa sederhana dan pemodelan bahasa.

Namun, sejauh pengetahuan para penulis, **Transformer adalah model transduksi pertama yang mengandalkan HANYA pada *self-attention*** untuk menghitung representasi masukan dan keluarannya, tanpa menggunakan RNN atau CNN.

---

## 3. Arsitektur Model (*Model Architecture*)

Hampir semua model transduksi sekuens neural yang kompetitif memiliki struktur **Encoder-Decoder**. 
- ***Encoder*** (Penyandi) memetakan urutan masukan dari representasi simbol, misal $(x_1, ..., x_n)$, menjadi urutan representasi kontinu $z = (z_1, ..., z_n)$. 
- ***Decoder*** (Pemuasandi), berdasarkan $z$ yang diberikan, kemudian akan menghasilkan urutan keluaran simbol $(y_1, ..., y_m)$ elemen demi elemen. 

Pada setiap langkah, model bersifat **auto-regresif**, artinya ia akan menggunakan (mengonsumsi) simbol-simbol yang telah dihasilkan sebelumnya sebagai tambahan masukan saat menghasilkan simbol berikutnya.

Arsitektur Transformer mengikuti struktur keseluruhan ini, menggunakan kombinasi *self-attention* yang ditumpuk (*stacked*) dan lapisan yang terhubung penuh (FC / *fully connected layers*) untuk masing-masing *encoder* dan *decoder*.

---

### Analisis Mendalam Gambar 1: Arsitektur Model Transformer

*(Catatan Dosen: Teks asli merujuk pada Figure 1. Mari kita bedah secara komprehensif apa yang direpresentasikan oleh diagram arsitektur tersebut, karena ini adalah 'jantung' dari makalah ini).*

**Deskripsi Gambar 1 (The Transformer - model architecture):**
Gambar 1 mengilustrasikan diagram blok arsitektur Transformer secara utuh. Diagram ini terbagi menjadi dua bagian besar: sisi kiri (blok abu-abu) adalah **Encoder**, dan sisi kanan (blok abu-abu) adalah **Decoder**.

**Pada Sisi Encoder (Kiri):**
1.  **Inputs (Masukan):** Data berwujud token yang masuk ke sistem.
2.  **Input Embedding (Penyematan Masukan):** Kotak merah muda di mana token diubah menjadi vektor berdimensi tinggi.
3.  **Positional Encoding (Pengkodean Posisional):** Disimbolkan dengan ikon gelombang (sinus/kosinus) di dalam lingkaran. Vektor ini ditambahkan ke Input Embedding untuk memberi tahu model urutan posisi kata.
4.  **Tumpukan Nx (Blok Encoder):** Kotak abu-abu berlabel $N \times$ (yang berarti blok ini diulang sebanyak $N=6$ kali). Di dalamnya terdapat:
    *   **Multi-Head Attention (oranye):** Menghitung keterkaitan setiap kata dengan kata lainnya dalam kalimat masukan.
    *   **Add & Norm (kuning):** Melakukan koneksi residual (penjumlahan masukan awal dengan keluaran sub-lapisan) dilanjutkan dengan normalisasi lapisan (*Layer Normalization*).
    *   **Feed Forward (biru):** Jaringan saraf konvensional yang dioperasikan pada tiap posisi secara terpisah.
    *   Terdapat panah melengkung (*skip connections*) di sekitar blok oranye dan biru, merepresentasikan *residual connection*.

**Pada Sisi Decoder (Kanan):**
1.  **Outputs (shifted right):** Masukan untuk decoder adalah keluaran (target) yang telah dihasilkan sejauh ini, digeser ke kanan satu posisi agar tidak melihat "masa depan" dari prediksi.
2.  **Output Embedding & Positional Encoding:** Serupa dengan bagian encoder.
3.  **Tumpukan Nx (Blok Decoder):** Diulang sebanyak $N=6$ kali. Di dalamnya terdapat TIGA sub-lapisan:
    *   **Masked Multi-Head Attention:** Modifikasi *attention* untuk mencegah posisi saat ini memperhatikan (*attend*) posisi di masa depan (yang belum ditebak).
    *   **Multi-Head Attention (Encoder-Decoder Attention):** Mengambil kunci (Key) dan nilai (Value) dari *Encoder* terakhir, dan kueri (Query) dari keluaran *Masked Attention Decoder*.
    *   **Feed Forward:** Sama seperti pada encoder.
    *   Masing-masing dari ketiga sub-lapisan ini juga diapit oleh **Add & Norm** dan koneksi residual.
4.  **Linear & Softmax:** Di puncak arsitektur, vektor keluaran dilewatkan ke lapisan Linear dan fungsi Softmax untuk menghasilkan **Output Probabilities** (probabilitas kata berikutnya).

---

### 3.1 Tumpukan *Encoder* dan *Decoder* (*Encoder and Decoder Stacks*)

**Encoder:**
*Encoder* tersusun dari sebuah tumpukan yang berisi $N = 6$ lapisan yang identik. Setiap lapisan memiliki dua sub-lapisan. 
1. Sub-lapisan pertama adalah mekanisme *self-attention* multi-kepala (*multi-head self-attention*).
2. Sub-lapisan kedua adalah jaringan *feed-forward* sederhana yang terhubung penuh berdasarkan posisi (*position-wise fully connected feed-forward network*).

Para peneliti menerapkan **koneksi residual** (koneksi lewatan/ *skip connection*) di sekeliling setiap sub-lapisan, yang kemudian diikuti dengan **normalisasi lapisan (*layer normalization*)**. 
Secara matematis, keluaran dari setiap sub-lapisan didefinisikan sebagai:
$$\text{LayerNorm}(x + \text{Sublayer}(x))$$
Di mana $\text{Sublayer}(x)$ adalah fungsi yang diimplementasikan oleh sub-lapisan itu sendiri. Untuk memfasilitasi koneksi residual ini, sangat penting bahwa seluruh sub-lapisan di dalam model, termasuk lapisan *embedding*, menghasilkan keluaran dengan dimensi yang seragam, yaitu $d_{model} = 512$.

**Decoder:**
*Decoder* juga tersusun dari tumpukan $N = 6$ lapisan identik. Selain dua sub-lapisan yang ada pada setiap lapisan *encoder*, *decoder* menyisipkan sub-lapisan ketiga. Sub-lapisan ketiga ini melakukan *multi-head attention* terhadap representasi keluaran dari tumpukan *encoder*. 

Sama halnya dengan *encoder*, koneksi residual diterapkan di sekeliling tiap sub-lapisan yang diikuti oleh normalisasi lapisan. Para peneliti juga memodifikasi sub-lapisan *self-attention* di dalam *decoder* untuk **mencegah agar suatu posisi tidak memperhatikan posisi-posisi setelahnya (masa depan)**. Pemenggalan (masking) ini, dikombinasikan dengan fakta bahwa *output embedding* digeser satu posisi (*offset by one position*), memastikan bahwa prediksi untuk elemen pada posisi $i$ hanya boleh bergantung pada keluaran yang sudah diketahui pada posisi-posisi sebelum $i$ (yakni kurang dari $i$).

---

### 3.2 Perhatian (*Attention*)

Secara konseptual, sebuah fungsi perhatian (*attention*) dapat digambarkan sebagai pemetaan dari sebuah kueri (*query*) dan sekumpulan pasangan kunci-nilai (*key-value pairs*) menuju sebuah keluaran (*output*), di mana kueri, kunci, nilai, dan keluaran semuanya berwujud vektor. Keluaran akhir dihitung menggunakan penjumlahan berbobot (*weighted sum*) dari vektor nilai (Values), di mana bobot (*weight*) untuk masing-masing nilai dihitung oleh fungsi kompatibilitas antara kueri dan kunci yang berkorespondensi.

---

#### Analisis Mendalam Gambar 2: Scaled Dot-Product Attention & Multi-Head Attention

*(Catatan Dosen: Mari kita pahami mekanisme komputasi internal dari perhatian ini merujuk pada Figure 2 di makalah asli).*

**Bagian Kiri (Scaled Dot-Product Attention):**
Ini adalah inti dari perhitungan nilai perhatian. Terdapat tiga vektor yang masuk: **Q (Query)**, **K (Key)**, dan **V (Value)**.
1.  **MatMul (Perkalian Matriks):** Kueri $Q$ dikalikan dengan transposisi dari Kunci $K$ ($K^T$). Ini menghasilkan skor kedekatan (kompatibilitas) antar token.
2.  **Scale (Penskalaan):** Skor tersebut dibagi dengan nilai skalar $\sqrt{d_k}$. Ini untuk mencegah gradien menjadi terlalu kecil.
3.  **Mask (opt.):** (Opsional, hanya digunakan di Decoder). Nilai-nilai masa depan diubah menjadi $-\infty$ sehingga diabaikan.
4.  **Softmax:** Menerapkan fungsi aktivasi *Softmax* untuk menormalkan skor menjadi probabilitas (rentang 0-1 dan totalnya 1).
5.  **MatMul:** Probabilitas ini dikalikan dengan vektor Nilai $V$ untuk menghasilkan representasi akhir (mana kata/konsep yang harus "diperhatikan" lebih kuat).

**Bagian Kanan (Multi-Head Attention):**
Alih-alih melakukan satu operasi perhatian besar, masukan V, K, dan Q diproyeksikan (menggunakan lapisan **Linear**) ke dimensi yang lebih kecil secara berulang kali (sebanyak $h$ kepala).
1.  Setiap "kepala" (head) melakukan **Scaled Dot-Product Attention** secara terpisah di dimensi yang lebih kecil ini. Hal ini memungkinkan model untuk menangkap berbagai jenis relasi (misal: satu kepala mengurus relasi subjek-predikat, kepala lain mengurus penunjuk *pronoun*).
2.  Keluaran dari semua kepala kemudian **digabungkan (Concat)** secara sejajar.
3.  Hasil gabungan tersebut diproyeksikan lagi menggunakan lapisan **Linear** terakhir untuk mendapatkan keluaran akhir.

---

#### 3.2.1 Perhatian Produk-Titik Terskalakan (*Scaled Dot-Product Attention*)

Para peneliti menyebut mekanisme perhatian spesifik mereka sebagai "Scaled Dot-Product Attention". Masukannya terdiri atas *queries* dan *keys* dengan dimensi $d_k$, serta *values* dengan dimensi $d_v$. 

Pertama, dihitung produk-titik (*dot product*) dari suatu *query* dengan seluruh *keys*. Kemudian setiap hasil dibagi dengan $\sqrt{d_k}$, dan fungsi *softmax* diterapkan untuk mendapatkan rentang bobot (probabilitas) untuk diaplikasikan pada *values*.

Dalam praktiknya, komputasi fungsi perhatian ini dilakukan pada sekumpulan *queries* secara simultan dengan menggabungkannya menjadi matriks $Q$. *Keys* dan *values* juga dikelompokkan ke dalam matriks $K$ dan $V$. Matriks keluaran dihitung berdasarkan rumus berikut:

$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$

**Penjelasan Rumus (1):**
*   **$Q$ (Kueri / Query):** Matriks yang merepresentasikan elemen yang sedang mencari kecocokan (apa yang sedang difokuskan saat ini).
*   **$K$ (Kunci / Key):** Matriks yang merepresentasikan "label" atau "identitas" dari elemen-elemen lain dalam sekuens. Operasi $QK^T$ menghasilkan produk-titik untuk mengecek seberapa cocok (mirip) $Q$ dengan setiap $K$.
*   **$d_k$ (Dimensi Key/Query):** Merupakan dimensi dari vektor kunci dan kueri.
*   **$\sqrt{d_k}$ (Faktor Penskalaan):** Akar kuadrat dari $d_k$. *Mengapa dibutuhkan?* Jika $d_k$ sangat besar, nilai dot-product ($QK^T$) bisa membengkak drastis. Saat nilai masuk ke rentang yang terlalu ekstrem, fungsi *softmax* akan masuk ke area di mana gradiennya (*gradient*) mendekati nol (sangat kecil), sehingga mempersulit proses pembaruan parameter saat pelatihan (*vanishing gradient*). Oleh karena itu, nilainya "diskalakan" (dibagi) dengan $\sqrt{d_k}$.
*   **$\text{softmax}$:** Fungsi yang menormalkan skor kecocokan agar semuanya bernilai antara 0 dan 1, dan total seluruh skor dalam satu baris adalah 1.
*   **$V$ (Nilai / Value):** Matriks yang menyimpan isi informasi yang sebenarnya. Matriks $V$ akan dikalikan dengan bobot *softmax* tadi. Jadi, elemen dengan kecocokan tinggi (bobot mendekati 1) akan disumbangkan lebih banyak ke keluaran akhir, sedangkan yang tidak cocok (mendekati 0) akan diabaikan.

Dua fungsi perhatian yang paling umum digunakan sebelumnya adalah *additive attention* dan *dot-product (multiplicative) attention*. *Dot-product attention* identik dengan algoritma di atas, kecuali absennya faktor penskalaan $\frac{1}{\sqrt{d_k}}$. Walaupun secara teori keduanya memiliki kompleksitas yang sama, komputasi *dot-product attention* jauh lebih cepat dan hemat ruang memori dalam praktik, karena dapat diimplementasikan menggunakan kode optimasi perkalian matriks standar.

*(Catatan kaki makalah menjelaskan matematisnya: jika komponen q dan k adalah variabel acak independen dengan mean 0 dan variansi 1, maka dot product-nya memiliki mean 0 dan variansi $d_k$. Penskalaan digunakan untuk menetralkan pembengkakan variansi ini).*

#### 3.2.2 Perhatian Multi-Kepala (*Multi-Head Attention*)

Alih-alih hanya menjalankan fungsi perhatian tunggal (*single attention*) dengan dimensi *keys, values,* dan *queries* sebesar $d_{model}$, peneliti menemukan bahwa memproyeksikan secara linear vektor-vektor tersebut sebanyak $h$ kali (menggunakan proyeksi bobot linear berbeda yang dipelajari mesin) ke dimensi yang lebih kecil ($d_k, d_k,$ dan $d_v$) memberikan hasil yang jauh lebih baik. 

Pada setiap versi *queries, keys,* dan *values* yang telah diproyeksikan ini, fungsi perhatian (Scaled Dot-Product) dilakukan secara paralel (bersamaan), sehingga menghasilkan keluaran berdimensi $d_v$. Nilai-nilai ini kemudian dirangkaikan (*concatenated*) dan diproyeksikan sekali lagi untuk menghasilkan nilai akhir.

*Multi-head attention* memungkinkan model untuk menaruh perhatian (attend) pada informasi dari subruang representasi yang berbeda (*different representation subspaces*) secara bersamaan pada posisi yang berbeda-beda. Jika hanya ada satu kepala perhatian (*single attention head*), proses merata-ratakan nilai akan mematikan kemampuan pemisahan relasi ini.

$$ \text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O $$
$$ \text{where } \text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V) $$

**Penjelasan Rumus & Variabel:**
*   **$h$:** Jumlah kepala (heads).
*   **$\text{head}_i$:** Perhitungan *Scaled Dot-Product Attention* untuk kepala ke-$i$.
*   **$W_i^Q \in \mathbb{R}^{d_{model} \times d_k}$:** Matriks parameter bobot yang dipelajari untuk memproyeksikan masukan matriks $Q$ (Kueri) pada kepala ke-$i$ menuju dimensi $d_k$.
*   **$W_i^K \in \mathbb{R}^{d_{model} \times d_k}$:** Matriks parameter bobot untuk memproyeksikan $K$ (Kunci) pada kepala ke-$i$ menuju dimensi $d_k$.
*   **$W_i^V \in \mathbb{R}^{d_{model} \times d_v}$:** Matriks parameter bobot untuk memproyeksikan $V$ (Nilai) pada kepala ke-$i$ menuju dimensi $d_v$.
*   **$W^O \in \mathbb{R}^{hd_v \times d_{model}}$:** Matriks parameter bobot untuk memproyeksikan kembali hasil gabungan seluruh kepala (yang memiliki dimensi total $h \times d_v$) agar kembali ke dimensi asli $d_{model}$.
*   **$\text{Concat}$:** Operasi penggabungan (merangkaikan) vektor keluaran dari semua kepala secara horizontal.

Dalam makalah ini, penulis menggunakan $h = 8$ lapisan perhatian paralel (kepala). Untuk setiap kepala, digunakan dimensi $d_k = d_v = d_{model}/h = 512/8 = 64$. Mengingat dimensi setiap kepala diperkecil, total beban komputasinya tetap setara dengan *single-head attention* dengan dimensi penuh.

#### 3.2.3 Penerapan Perhatian dalam Model Ini

Transformer menggunakan *multi-head attention* dalam tiga cara berbeda:
1.  **Pada lapisan "Encoder-Decoder Attention"**: *Queries* (Q) berasal dari lapisan *decoder* sebelumnya, sementara *memory keys* (K) dan *values* (V) bersumber dari representasi keluaran *encoder*. Hal ini memungkinkan **setiap posisi di decoder untuk memperhatikan seluruh posisi pada urutan masukan**. Pola ini meniru mekanisme perhatian *encoder-decoder* pada model Sequence-to-Sequence (Seq2Seq) klasik.
2.  **Pada *Self-Attention* Encoder**: *Encoder* memiliki lapisan *self-attention*. Di sini, seluruh komponen K, V, dan Q berasal dari tempat yang sama, yaitu dari keluaran lapisan *encoder* yang sebelumnya. Setiap posisi pada *encoder* dapat menaruh perhatian pada semua posisi lain dalam lapisan *encoder* sebelumnya (menangkap konteks keseluruhan kalimat).
3.  **Pada *Self-Attention* Decoder**: Serupa dengan *encoder*, lapisan *self-attention* di *decoder* memungkinkan setiap posisi di *decoder* memperhatikan semua posisi di *decoder* **hanya hingga posisi itu sendiri** (termasuk dirinya). Kita harus mencegah aliran informasi bergeser ke kiri (melihat ke masa depan) di dalam *decoder* untuk mempertahankan sifat *auto-regressive*. Hal ini diimplementasikan di dalam *scaled dot-product attention* dengan **menutupi (masking out)** dengan cara mengatur nilainya menjadi $-\infty$ (negatif tak terhingga) pada semua entri masukan menuju *softmax* yang berhubungan dengan koneksi-koneksi ilegal (masa depan). Nilai $-\infty$ jika dimasukkan ke *softmax* akan menjadi 0 (sama sekali tidak diberi perhatian).

---

### 3.3 Jaringan Jalin-Maju Berbasis Posisi (*Position-wise Feed-Forward Networks*)

Selain sub-lapisan perhatian, setiap lapisan (*layer*) pada *encoder* maupun *decoder* memuat sebuah jaringan saraf *feed-forward* terhubung-penuh (*fully connected*) yang diaplikasikan ke tiap-tiap posisi secara terpisah dan identik. Jaringan ini tersusun atas dua transformasi linear dengan fungsi aktivasi **ReLU (Rectified Linear Unit)** di antaranya.

$$ \text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2 \quad \quad \text{(2)} $$

**Penjelasan Rumus (2) & Variabel:**
*   **$x$:** Representasi masukan pada sebuah posisi tertentu (keluaran dari lapisan *attention* sebelumnya).
*   **$W_1$ dan $W_2$:** Matriks bobot transformasi linear pertama dan kedua.
*   **$b_1$ dan $b_2$:** Vektor *bias* (bias) dari transformasi pertama dan kedua.
*   **$\max(0, \dots)$:** Ini adalah representasi matematis dari fungsi aktivasi ReLU. Jika hasil di dalam kurung negatif, hasilnya 0. Jika positif, nilainya dipertahankan.

Walaupun transformasi linear (nilai matriks bobotnya) sama untuk semua posisi berlainan di dalam satu lapisan, mereka menggunakan parameter yang **berbeda** dari satu lapisan ke lapisan lainnya. Cara lain mendeskripsikan blok ini adalah sebagai dua operasi konvolusi dengan ukuran *kernel* sebesar 1. 

Dimensinya adalah sebagai berikut: Dimensi masukan dan keluaran adalah $d_{model} = 512$, dan lapisan tersembunyi di dalamnya (*inner-layer*) diekspansi (*di-upscale*) ke dimensi $d_{ff} = 2048$.

---

### 3.4 Penyematan dan *Softmax* (*Embeddings and Softmax*)

Serupa dengan model transduksi sekuens pada umumnya, Transformer menggunakan vektor penyematan yang dipelajari (*learned embeddings*) untuk mengubah token masukan dan token keluaran menjadi vektor berdimensi $d_{model}$. Model ini juga menggunakan transformasi linear standar yang dipelajari serta fungsi *softmax* untuk mengonversi representasi keluaran *decoder* menjadi probabilitas token berikutnya (*predicted next-token probabilities*). 

Dalam arsitektur model ini, para penulis **berbagi matriks bobot yang sama** (*share the same weight matrix*) antara kedua lapisan penyematan (baik masukan maupun keluaran) dan transformasi linear pra-softmax (sebelum masuk *softmax*). Selain itu, pada tahap *embedding*, bobot tersebut dikalikan dengan konstanta $\sqrt{d_{model}}$ untuk menjaga agar skala bobotnya stabil setelah inisialisasi.

---

### 3.5 Pengkodean Posisional (*Positional Encoding*)

Karena model Transformer secara absolut **tidak mengandung perulangan (RNN) dan tidak mengandung konvolusi (CNN)**, maka untuk membuat model ini mengerti struktur "urutan" dari sekuens, kita harus secara artifisial menyuntikkan informasi terkait posisi relatif atau posisi absolut dari token-token di dalam sekuens tersebut. 

Untuk mengatasi hal ini, penulis menambahkan *"positional encodings"* (pengkodean posisi) pada input awal dari *embeddings* di bagian paling bawah tumpukan *encoder* dan *decoder*. Vektor pengkodean posisional ini memiliki dimensi $d_{model}$ yang **sama** persis dengan vektor *embeddings*, sehingga keduanya bisa dijumlahkan (*summed*) secara langsung. Ada banyak opsi fungsi untuk ini, baik yang dipelajari mesin (learned) maupun fiks (fixed).

Dalam kajian ini, penulis menggunakan fungsi sinus dan kosinus dengan frekuensi gelombang yang berbeda-beda:

$$ PE_{(pos, 2i)} = \sin(pos / 10000^{2i / d_{model}}) $$
$$ PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i / d_{model}}) $$

**Penjelasan Rumus & Variabel:**
*   **$PE$:** Positional Encoding (Nilai skalar yang akan ditambahkan ke vektor embedding).
*   **$pos$:** Posisi absolut token (kata) di dalam kalimat masukan (contoh: kata pertama pos=0, kata kedua pos=1, dst).
*   **$i$:** Indeks dimensi dari vektor (dari $0$ hingga $d_{model}/2 - 1$).
*   **$d_{model}$:** Dimensi representasi total (yaitu 512).
*   **$2i$ (genap) dan $2i+1$ (ganjil):** Rumus ini mengatur bahwa pada indeks dimensi genap, digunakan fungsi sinus ($\sin$), sedangkan pada indeks ganjil, digunakan fungsi kosinus ($\cos$).

Dengan demikian, setiap dimensi dari vektor pengkodean posisi berhubungan dengan satu gelombang sinusoid. Panjang gelombang (*wavelengths*) membentuk progresi geometrik dari $2\pi$ hingga $10000 \cdot 2\pi$. 
Penulis memilih fungsi spesifik ini karena mereka berhipotesis bahwa rumus sinus/kosinus ini akan memudahkan model dalam belajar menaruh perhatian (*attend*) dengan mengandalkan posisi relatif (*relative positions*). Sebab, untuk jarak (offset) tetap $k$ berapapun nilainya, nilai $PE_{pos+k}$ dapat direpresentasikan sebagai sebuah fungsi linear dari $PE_{pos}$.

Sebagai catatan tambahan, penulis juga mengeksperimenkan penggunaan *learned positional embeddings* (vektor posisi yang dipelajari secara acak/stokastik selama *training*). Mereka menemukan bahwa kedua metode menghasilkan luaran yang hampir identik (bisa dilihat nanti di Tabel 3 baris E). Pada akhirnya, versi sinusoidal dipilih karena secara teoritis memungkinkan model untuk bisa melakukan ekstrapolasi ke panjang urutan kalimat yang lebih panjang dibandingkan yang pernah ditemui selama masa pelatihan.

---

## 4. Mengapa Menggunakan Perhatian-Mandiri (*Why Self-Attention*)

Pada bagian ini, mari kita bandingkan berbagai aspek *self-attention* berbanding terbalik dengan lapisan *recurrent* (RNN) dan *convolutional* (CNN) yang sering digunakan secara konvensional untuk memetakan urutan variabel $(x_1, ..., x_n)$ menjadi urutan lainnya $(z_1, ..., z_n)$ yang sama panjangnya, di mana $x_i, z_i \in \mathbb{R}^d$. 

Penulis memotivasi penggunaan arsitektur berbasis murni *self-attention* dengan merujuk pada tiga kriteria desiderata (kebutuhan ideal) utama:
1.  **Kompleksitas Komputasi Total Per Lapisan.**
2.  **Kuantitas Komputasi yang Dapat Diparalelisasi.** (Diukur berdasarkan minimum operasi sekuensial berurut yang diperlukan).
3.  **Panjang Jalur (Path Length) Antara Ketergantungan Jarak Jauh.** Mengapa ini penting? Tantangan utama pemrosesan bahasa adalah menghubungkan kata yang muncul di awal dokumen dengan kata di akhir dokumen. Semakin pendek jalur operasional (maju-mundur) antar posisi, semakin mudah jaringan belajar menangkap pola (*learning long-range dependencies*).

Tabel berikut meringkas perbandingan tersebut:

| Jenis Lapisan (*Layer Type*) | Kompleksitas per Lapisan | Operasi Sekuensial | Panjang Jalur Maksimum |
| :--- | :---: | :---: | :---: |
| Self-Attention | $O(n^2 \cdot d)$ | $O(1)$ | $O(1)$ |
| Recurrent (RNN) | $O(n \cdot d^2)$ | $O(n)$ | $O(n)$ |
| Convolutional (CNN) | $O(k \cdot n \cdot d^2)$ | $O(1)$ | $O(\log_k(n))$ |
| Self-Attention (restricted) | $O(r \cdot n \cdot d)$ | $O(1)$ | $O(n/r)$ |

*(Keterangan Tabel: $n$ adalah panjang sekuens/kalimat, $d$ adalah dimensi representasi, $k$ adalah ukuran kernel konvolusi, dan $r$ adalah batasan ukuran lingkungan/neighborhood pada restricted self-attention).*

**Analisis Konklusi Akademis dari Tabel 1:**
1.  **Self-Attention vs RNN:** Lapisan *self-attention* menghubungkan HAMPIR seluruh posisi satu sama lain dalam jumlah operasi sekuensial yang konstan sebesar $O(1)$, berbeda dengan RNN yang membutuhkan rantai berantai sebanyak langkah $O(n)$. Terkait perhitungan numerik, *self-attention* lebih cepat dari arsitektur *recurrent* selama panjang sekuens $n$ lebih kecil dibandingkan dimensi representasinya $d$. Hal ini biasanya terpenuhi karena arsitektur modern (seperti representasi *word-piece* atau *byte-pair*) menggunakan $d$ yang cukup besar (misal 512, 1024).
2.  **Self-Attention Terbatas (Restricted):** Untuk mengatasi komputasi berat pada urutan yang teramat sangat panjang (*very long sequences*), Transformer bisa dimodifikasi. Model hanya akan melihat sekelilingnya sejauh ukuran ketetapan $r$. Ini akan membuat jalur maksimum melemah menjadi $O(n/r)$, yang direncanakan akan dieksplorasi dalam penelitian masa depan para penulis.
3.  **Self-Attention vs CNN:** CNN dengan ukuran *kernel* $k < n$ tidak secara langsung menyambungkan jarak pandang semua input-output secara seketika. Jika *kernel* CNN *contiguous* (bersambung), dibutuhkan tumpukan lapisan bertingkat sebanyak $O(n/k)$. Jika *dilated CNN*, membutuhkan $O(\log_k(n))$. Semakin tebal lapisannya, semakin sulit ia menyebarkan informasi. Belum lagi dari segi *cost* (biaya), CNN secara general lebih mahal dibanding RNN dengan skala pengali $k$. Menggunakan trik konvolusi terpisah (*separable convolutions*) dapat menekan biaya, tetapi bagaimanapun *self-attention* jauh lebih elegan.

*Sebagai manfaat tambahan, self-attention juga terbukti menghasilkan model yang jauh lebih mudah untuk diinterpretasikan (interpretable). Meneliti distribusi atensi membuktikan bahwa setiap "kepala" atensi mempelajari tugas sintaksis dan struktur gramatikal semantik secara independen.*

---

## 5. Pelatihan (*Training*)

Bagian ini membeberkan rezim pelatihan yang digunakan.

### 5.1 Data Pelatihan dan Pembagian Kelompok (*Training Data and Batching*)
*   **English-German (EN-DE):** Model dilatih di atas set data terstandardisasi WMT 2014 berisi **4,5 juta pasang kalimat**. Kalimat dienkode menggunakan algoritma *byte-pair encoding* (BPE) yang mencakup kosakata gabungan target-sumber sebesar **37.000 token**.
*   **English-French (EN-FR):** Model menggunakan pangkalan data WMT 2014 berukuran super raksasa dengan **36 juta kalimat**, kemudian memecahnya menjadi kosakata terpotong (word-piece) sebesar **32.000 token**.
*   **Batching:** Pasangan kalimat dikelompokkan berdasarkan kemiripan panjang sekuens untuk optimalisasi memori. Setiap blok *batch* setidaknya mengandung estimasi kasar 25.000 token sumber (*source tokens*) dan 25.000 token target.

### 5.2 Perangkat Keras dan Jadwal (*Hardware and Schedule*)
*   Mesin yang digunakan: Sebuah server bertenaga **8 keping NVIDIA P100 GPUs**.
*   **Base Model (Model Dasar):** Di bawah parameter standar yang ada di *paper*, 1 langkah (*step*) dieksekusi dalam 0,4 detik. Total waktu pelatihannya adalah 100.000 langkah (memakan waktu **12 jam**).
*   **Big Model (Model Besar):** (Dideskripsikan pada Tabel 3). Model ini berukuran raksasa, langkah latihannya 1,0 detik per *step*, dilatih sepanjang 300.000 langkah (total memakan **3.5 hari** tanpa henti).

### 5.3 Pengoptimal (*Optimizer*)
Peneliti menggunakan pengoptimal **Adam** (Algoritma Adaptive Moment Estimation) dengan konfigurasi hiper-parameter unik: 
$\beta_1 = 0.9$, $\beta_2 = 0.98$, dan $\epsilon = 10^{-9}$.

Selain itu, angka kecepatan pembelajaran (Learning Rate / LR) tidak dibuat statis. Ia divariasikan selama masa komputasi dengan rumus perhitungan berikut:

$$ \text{lrate} = d_{model}^{-0.5} \cdot \min(\text{step\_num}^{-0.5}, \, \text{step\_num} \cdot \text{warmup\_steps}^{-1.5}) \quad \quad \text{(3)} $$

**Penjelasan Rumus (3) & Variabel:**
*   **$lrate$:** *Learning rate* (angka tingkat pembelajaran) pada saat itu.
*   **$d_{model}^{-0.5}$:** Dimensi arsitektur Transformer dipangkatkan minus setengah (atau $1/\sqrt{d_{model}}$). Digunakan sebagai skala basis awal.
*   **$\text{step\_num}$:** Counter perulangan iterasi saat ini.
*   **$\text{warmup\_steps}$:** Konstanta statis yang menentukan fase "pemanasan". Nilai yang digunakan penulis adalah $4000$.
*   **Logika di balik fungsi $\min(...)$:** Formula matematika di atas ekuivalen dengan mengatakan bahwa pada periode awal pelatihan ($0$ hingga $4000$ langkah), nilai *learning rate* akan **meningkat secara linear dan agresif** (*warm-up*). Setelah melewati batas waktu langkah pemanasan ($>4000$), *learning rate* akan **menyusut lambat (decay)** yang proporsional dengan balikan dari akar kuadrat dari jumlah iterasi langkah saat itu ($1/\sqrt{step}$). Tujuannya agar pada mulanya model menjelajah mencari posisi optimal secara radikal, namun melambat dengan saksama di akhir untuk konvergensi ke optimum global.

### 5.4 Regularisasi (*Regularization*)
Untuk mencegah *overfitting* (terlalu menghafal data dan gagal pada data baru), sistem memakai tiga teknik stabilisasi:
1.  **Dropout Residual (Residual Dropout):** Peneliti melempar pengacakan nonaktif (Dropout) kepada keluaran dari masing-masing sub-lapisan (sub-layer), persis SEBELUM diakumulasikan ke masukan dari blok tambahan *add* sub-layer. Serta mem-fungsikan Dropout saat menjumlah vektor *embedding* dan vektor Posisi Posisional baik di blok tumpuk *encoder* dan *decoder*. Nilai kemungkinan putus/drop-out untuk "Base Model" adalah $P_{drop} = 0.1$.
2.  **Pelembutan Label (Label Smoothing):** Sistem memaksa agar hasil akhir dari *Softmax* memiliki toleransi probabilitas kejut (*uncertainty*) $\epsilon_{ls} = 0.1$. Ini mencederai kebingungan (Perplexity) secara angka di atas kertas, tetapi secara dramatis memperbaiki akurasi linguistik serta menyumbang poin penambahan tinggi pada tes kalkulasi BLEU (Bilingual Evaluation Understudy).

---

## 6. Hasil Utama (*Results*)

### 6.1 Penerjemahan Mesin (*Machine Translation*)

Pada tes terberat linguistik global, yakni **WMT 2014 English-to-German (EN-DE)**, arsitektur "Transformer (big)" secara spektakuler melibas dan menundukkan seluruh arsitektur mutakhir (termasuk jaringan raksasa *ensembles* yang memakai gabungan model). Transformer menang telak dengan jarak margin margin fantastis di atas **2.0 BLEU**. Pencapaian itu berhasil membukukan Rekor Dunia/SOTA terbaru dalam poin BLEU sebesar **28.4**. Model ini dikalkulasi hanya dengan 8 GPU P100 dalam 3,5 hari. Hebatnya lagi, versi kecilnya (Base Model) juga masih melampaui keandalan seluruh model jurnal akademik lalu-lalu dengan durasi serta biaya pecahan super hemat.

Pada gugus **WMT 2014 English-to-French (EN-FR)**, konfigurasi *big model* tembus 41.0 BLEU, mencetak dominasi penuh melampaui gabungan arsitektur model-tunggal di literatur masa lalu dengan konsumsi *training cost* sangat kerdil (tidak sampai $\frac{1}{4}$ harga operasional jaringan *state-of-the-art* kuno).

Untuk meminimalkan variansi (ketidakstabilan), mereka menerapkan teknis rata-rata (*averaging*). Untuk basis-model, rataan diambil terhadap 5 *checkpoints* (tanda simpan) beruntun dalam rentang masa 10 menitan. Pada model raksasa (*big model*), dirata-rata hingga 20 pos-simpan (*checkpoints*). Parameter pencarian (Beam Search) dipatok sebesar dimensi $4$ dan besaran pinalti (*Length Penalty*) sebesar $\alpha = 0.6$. 

Tabel berikut merangkum kalkulasi biaya, komparasi kekuatan performa, serta hitung-hitungan ekspektasi FLOPs (*Floating Point Operations*) operasional GPU:

**Tabel 2: Hasil Kinerja dan Biaya Pelatihan pada Model English-to-German dan English-to-French WMT 2014**

| Model | BLEU (EN-DE) | BLEU (EN-FR) | Training Cost (FLOPs) EN-DE | Training Cost (FLOPs) EN-FR |
| :--- | :---: | :---: | :---: | :---: |
| ByteNet | 23.75 | - | - | - |
| Deep-Att + PosUnk | - | 39.2 | - | $1.0 \cdot 10^{20}$ |
| GNMT + RL | 24.6 | 39.92 | $2.3 \cdot 10^{19}$ | $1.4 \cdot 10^{20}$ |
| ConvS2S | 25.16 | 40.46 | $9.6 \cdot 10^{18}$ | $1.5 \cdot 10^{20}$ |
| MoE | 26.03 | 40.56 | $2.0 \cdot 10^{19}$ | $1.2 \cdot 10^{20}$ |
| Deep-Att + PosUnk Ensemble | - | 40.4 | - | $8.0 \cdot 10^{20}$ |
| GNMT + RL Ensemble | 26.30 | 41.16 | $1.8 \cdot 10^{20}$ | $1.1 \cdot 10^{21}$ |
| ConvS2S Ensemble | 26.36 | **41.29** | $7.7 \cdot 10^{19}$ | $1.2 \cdot 10^{21}$ |
| **Transformer (base model)** | 27.3 | 38.1 | $\mathbf{3.3 \cdot 10^{18}}$ | - |
| **Transformer (big)** | **28.4** | **41.8** | $2.3 \cdot 10^{19}$ | - |

*(Catatan Dosen: Amati bagaimana Transformer 'big' mencapai BLEU tertinggi (28.4 dan 41.8) namun dengan FLOPs yang jauh lebih kecil dibandingkan model arsitektur Ensembel GNMT atau ConvS2S, membuktikan tingginya efisiensi).*

### 6.2 Variasi-Variasi Parameter Model (*Model Variations*)

Untuk melihat seberapa vital sebuah organ dalam struktur arsitektur ini bekerja, penulis membongkar-pasang parameter Transformer *(Ablation Study)* untuk penerjemahan EN-DE. 

**Tabel 3: Eksperimen Variasi pada Arsitektur Transformer (Berdasarkan model dasar "Base")**

| | $N$ | $d_{model}$ | $d_{ff}$ | $h$ | $d_k$ | $d_v$ | $P_{drop}$ | $\epsilon_{ls}$ | train steps | PPL (dev) | BLEU (dev) | params $\times 10^6$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| base | 6 | 512 | 2048 | 8 | 64 | 64 | 0.1 | 0.1 | 100K | 4.92 | 25.8 | 65 |
| (A) | | | | 1 | 512 | 512 | | | | 5.29 | 24.9 | |
| | | | | 4 | 128 | 128 | | | | 5.00 | 25.5 | |
| | | | | 16 | 32 | 32 | | | | 4.91 | 25.8 | |
| | | | | 32 | 16 | 16 | | | | 5.01 | 25.4 | |
| (B) | | | | | 16 | | | | | 5.16 | 25.1 | 58 |
| | | | | | 32 | | | | | 5.01 | 25.4 | 60 |
| (C) | 2 | | | | | | | | | 6.11 | 23.7 | 36 |
| | 4 | | | | | | | | | 5.19 | 25.3 | 50 |
| | 8 | | | | | | | | | 4.88 | 25.5 | 80 |
| | | 256 | | | 32 | 32 | | | | 5.75 | 24.5 | 28 |
| | | 1024 | | | 128 | 128 | | | | 4.66 | 26.0 | 168 |
| | | | 1024 | | | | | | | 5.12 | 25.4 | 53 |
| | | | 4096 | | | | | | | 4.75 | 26.2 | 90 |
| (D) | | | | | | | 0.0 | | | 5.77 | 24.6 | |
| | | | | | | | 0.2 | | | 4.95 | 25.5 | |
| | | | | | | | | 0.0 | | 4.67 | 25.3 | |
| | | | | | | | | 0.2 | | 5.47 | 25.7 | |
| (E) | \multicolumn{9}{l|}{positional embedding instead of sinusoids} | 4.92 | 25.7 | |
| big | 6 | 1024 | 4096 | 16 | | | 0.3 | | 300K | **4.33** | **26.4** | 213 |

**Kesimpulan dari Ablasi di Tabel 3:**
*   **Baris (A):** Mengevaluasi jumlah "kepala perhatian" ($h$) dengan dimensi total kunci dan nilai yang dikunci mati (konstan). Jika hanya memakai sistem satu kepala ($h=1$), terjemahan cacat -0,9 poin BLEU dibanding versi delapan kepala terbaik. Jika sistem dikepalai terlalu banyak (sampai 32), kualitas terjemahannya justru merosot lagi (jatuh kembali ke 25,4).
*   **Baris (B):** Mengamati dimensi vektor kunci pencari ($d_k$). Saat ukuran mengecil terlalu kecil, angka parameter berkinerja buruk (jelek). Ini membuktikan kalkulasi kecocokan fungsi *dot-product* tidak semudah itu apabila vektor sangat terkompresi.
*   **Baris (C) & (D):** Ukuran yang lebih gemuk dan lapisan tumpukan yang lebih banyak tentu meningkatkan performa kecerdasan artifisial, namun harus disiasati parameter penghindar ingatan buta *Drop-out* guna menghindar dari fenomena gagal generik *Over-fitting*.
*   **Baris (E):** Membandingkan gelombang Sinus/Cosinus statis absolut terhadap *embeddings* posisi belajar dinamis. Terlihat keduanya menghadirkan titik stabil yang presisi identik satu sama lain.

### 6.3 Penguraian Konstituen Bahasa Inggris (*English Constituency Parsing*)

Untuk mengevaluasi apakan model jenius ini hanya ahli untuk penerjemahan saja, peneliti menantang sang Transformer pada arena di luar teritorinya: *English constituency parsing* (Pemecahan Konstituen Gramatikal). Domain ini punya rintangan tersendiri; panjang tulisan keluaran *output* jauh lebih bengkak dan memanjang, serta mengusung batasan struktur hirarki sintaks yang ketat. 

Peneliti memodifikasi tipis Transformer (berdimensi tebal menjadi 4 lapis dan dimensi $d_{model}=1024$) dengan menggunakan arsip koleksi The Wall Street Journal (WSJ) milik Penn Treebank sebesar sekitar **40 Ribu kalimat pelatihan**. Model ini pun dicemplungkan untuk belajar metode komputasi Semi-Supervisi di *BerkleyParser* bervolume 17 Juta kalimat. 

**Tabel 4: Kinerja Transformer pada Parsing (Nilai dievaluasi di Bagian 23 WSJ F1)**

| Parser | Training | WSJ 23 F1 |
| :--- | :--- | :---: |
| Vinyals & Kaiser el al. (2014) | WSJ only, discriminative | 88.3 |
| Petrov et al. (2006) | WSJ only, discriminative | 90.4 |
| Zhu et al. (2013) | WSJ only, discriminative | 90.4 |
| Dyer et al. (2016) | WSJ only, discriminative | 91.7 |
| **Transformer (4 layers)** | **WSJ only, discriminative** | **91.3** |
| Zhu et al. (2013) | semi-supervised | 91.3 |
| Huang & Harper (2009) | semi-supervised | 91.3 |
| McClosky et al. (2006) | semi-supervised | 92.1 |
| Vinyals & Kaiser el al. (2014)| semi-supervised | 92.1 |
| **Transformer (4 layers)** | **semi-supervised** | **92.7** |
| Luong et al. (2015) | multi-task | 93.0 |
| Dyer et al. (2016) | generative | 93.3 |

**Kesimpulan dari Hasil Tabel 4:**
Ternyata, **tanpa perlu di-tuning (*Fine-Tuning*) secara khusus untuk tugas ini**, arsitektur dasar Transformer berkinerja luar biasa gemilang (F1 score = 91.3 pada WSJ only dan 92.7 pada semi-supervised). Model ini mampu menggilas mayoritas model pakar tata bahasa yang pernah dipublikasikan sebelumnya, terkecuali *Recurrent Neural Network Grammar*. Bahkan ketika ditantang untuk diuji latih eksklusif dalam lingkungan kelangkaan data kecil 40.000 kalimat, sang Transformer berhasil memencundangi program populer seperti Berkeley-Parser.

---

## 7. Kesimpulan (*Conclusion*)

Di dalam literatur kajian monumental ini, tim riset mendemonstrasikan **Transformer**, perintis dan nenek moyang model konversi urutan pertama di planet bumi yang fondasinya direkatkan SECARA MUTLAK pada kerangka perhatian jaringan saraf (*attention mechanism*). Ia membuang jauh-jauh peradaban kuno *encoder-decoder* berbekal memori perulangan, lalu merajut evolusi barunya di atas pilar agung *multi-headed self-attention*.

Untuk sektor keilmuan penerjemahan bahasa antarbangsa, keperkasaan mesin generasi Transformer terbukti secara absolut mampu mereduksi durasi dan biaya pelatihan sampai titik ekstrem tercepat ketimbang keluarga CNN dan RNN. Transformer merebut dominasi tahta *State-of-the-art* dalam WMT 2014 EN-DE maupun EN-FR, menumbangkan seluruh mahakarya terdahulu termasuk gabungan kekuatan *Ensemble*.

Visi masa depan yang antusias segera disiapkan oleh para ilmuwan dari fenomena lahirnya perhatian arsitektural ini (*attention-based models*), dan bermaksud menggiring fusi penerapannya demi resolusi dimensi sensorik yang lebih agung seperti gambar (citra), instrumen audio, dan video; membuat generasi beruntun (sekuensial) bisa menjadi kian sirna digantikan paralelisasi serentak.

*Seluruh repositori dan kode sumber kecerdasan program rilis di:* `https://github.com/tensorflow/tensor2tensor`

---
*(Catatan Dosen: Mahasiswa sekalian, ini adalah batas akhir dari teks utama makalah inti. Bagian referensi dan lampiran yang berisi demonstrasi visualisasi atensi diabaikan sesuai panduan silabus, karena konsep terpenting dari arsitektur matematika di makalah ini sudah terjabarkan secara sempurna di atas. Pastikan Anda menguasai setiap rumus $Q, K, V$ serta arsitektur encoder-decoder yang baru saja kita diskusikan sebelum menghadapi ujian!)*