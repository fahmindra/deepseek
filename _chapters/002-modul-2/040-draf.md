---
slug: modul-2-4
title: modul-2-4
---
### 2.4 Mekanisme Attention, Perbandingannya dengan Pendahulunya serta Evolusinya
---

Pada Subtopik 2.3, kita telah berhasil melakukan pra-proses krusial. Kita kini memiliki metode untuk mengubah teks mentah apa pun menjadi serangkaian vektor yang padat makna (hasil dari Token Embedding + Positional Encoding). Kita sekarang memiliki input yang siap: **[Vektor_1, Vektor_2, Vektor_3, ...].**

Pertanyaan fundamental berikutnya adalah: Arsitektur neural network apa yang akan memproses sekuens vektor ini? Bahasa, pada intinya, bersifat sekuensial. Makna sebuah kata sangat bergantung pada konteks yang mendahuluinya. Selama bertahun-tahun, sejak awal era deep learning untuk NLP, arsitektur pilihan untuk masalah ini adalah Recurrent Neural Networks (RNNs) dan varian canggihnya, LSTM serta GRU.

RNN dirancang untuk meniru cara manusia membaca: satu token pada satu waktu.

> **Analogi Sederhana RNN:**
> Bayangkan RNN sebagai seorang pembaca yang memproses kalimat kata demi kata. Saat ia membaca kata ke-$t$ (misal, "makan"), ia menggabungkan Input Vektor ("makan") dengan sebuah "Memori" (disebut Hidden State) dari kata sebelumnya ($t-1$, misal, "saya"). Ia kemudian menghasilkan output dan memperbarui "Memori" $H_t$ tersebut untuk diberikan ke kata berikutnya ($t+1$, misal, "nasi").

Pendekatan sekuensial ini tampak sangat logis. Namun, arsitektur rekuren ini memiliki satu kelemahan arsitektural yang fundamental: masalah ketergantungan jangka panjang (*long-term dependency*). Bayangkan kalimat ini: "Pria yang mengenakan kemeja biru dan kacamata hitam, yang baru saja masuk ke ruangan di seberang aula, ... [beberapa kalimat lagi]... adalah ayah saya."

Saat model tiba di kata "adalah", ia harus menghubungkannya kembali ke "Pria" di awal kalimat untuk memahami subjeknya. Bagi RNN, "memori" yang dibawa dari satu kata ke kata berikutnya sering kali sudah "tercucian" atau "terlupakan" (*vanishing gradient problem*) setelah melewati begitu banyak langkah waktu.

Keterbatasan memori inilah yang memicu revolusi dalam desain arsitektur. Alih-alih mengandalkan satu vektor "memori" yang rapuh, para peneliti bertanya: "Bagaimana jika model, pada setiap langkah, dapat 'melihat kembali' (*look back*) ke semua kata sebelumnya dan memutuskan mana yang penting?" Inilah ide inti dari **mekanisme attention**.

Di subtopik ini, kita akan membedah arsitektur rekuren (RNN/LSTM) untuk memahami masalahnya, lalu memperkenalkan attention sebagai solusinya, dan menelusuri evolusinya menjadi self-attention yang menjadi inti dari Transformer.

---

#### 2.4.1 Keterbatasan Arsitektur Rekuren (RNN, LSTM, GRU)

Arsitektur neural pertama yang dirancang khusus untuk data sekuensial adalah **Recurrent Neural Network (RNN).**

> **Apa Sebenarnya itu RNN?**
> RNN adalah sebuah kelas neural network di mana koneksi antar node membentuk sebuah siklus (loop), yang memungkinkannya untuk memelihara "memori" atau state internal $(H_t)$ dari input sebelumnya.

![Gambar 7. Arsitektur RNN dan Visualisasi Proses dalam RNN](https://lms.sdmdigital.id/pluginfile.php/635364/mod_page/content/12/img22.png)
**Gambar 7. Arsitektur RNN dan Visualisasi Proses dalam RNN**

Diagram sebelah kiri menunjukkan konsep arsitektur RNN: sebuah "Sel" (A) yang terisolasi. Ia menerima Input (seperti token $X_t$) dari atas, menghasilkan Output ($Y_t$) ke bawah, dan memiliki siklus (loop) internal (H). Loop horizontal ini melambangkan "Memori" (Hidden State) yang disimpan oleh sel tersebut untuk digunakan pada langkah berikutnya.

Diagram sebelah kanan adalah visualisasi praktik dari diagram kiri saat "dibuka" (*unrolled*) untuk memproses sebuah sekuens. Ini menunjukkan bahwa "Sel A" yang sama digunakan berulang kali. Alur panah horizontal (dari $H_{t-1}$ ke $H_t$) adalah "memori" yang mengalir dari satu langkah waktu ke langkah waktu berikutnya. Ini adalah visualisasi eksplisit dari loop yang digambarkan di diagram kiri. Aliran memori sekuensial inilah yang merupakan sumber kekuatan (konteks) sekaligus kelemahan (*vanishing gradient*) dari RNN.

Alur memori horizontal yang kita lihat di Diagram kanan (dari $H_{t-1}$ ke $H_t$) adalah inti dari cara RNN 'mengingat'. Namun, rantai ketergantungan yang panjang inilah yang secara ironis menjadi masalah komputasi terbesar selama proses pelatihan model. Saat model mencoba belajar dari kesalahannya (melalui backpropagation), sinyal pembelajaran harus mengalir 'mundur' dari token terakhir hingga token pertama melalui rantai yang sama. Proses 'mengalir mundur' melintasi banyak langkah waktu inilah yang mengarah ke masalah matematis fundamental.

**Masalah Fundamental: Vanishing/Exploding Gradients**

Kekuatan RNN (memori) juga merupakan kelemahannya. Saat melatih model (backpropagation), gradien (sinyal pembelajaran) harus mengalir "mundur" melalui setiap langkah waktu (misal, dari $H_{10}$ ke $H_9$ ke $H_8$ ke... ke $H_1$). Proses ini melibatkan perkalian berulang dari matriks weight yang sama:

*   Jika nilai-nilai dalam matriks itu (rata-rata) < 1, gradien akan menyusut secara eksponensial menuju nol (*Vanishing Gradients*). Model di $H_{10}$ tidak bisa "belajar" apapun dari $H_1$.
*   Jika nilai-nilai (rata-rata) > 1, gradien akan meledak secara eksponensial (*Exploding Gradients*), membuat pelatihan tidak stabil.

**Solusi Sementara dari Masalah ini? LSTM dan GRU**

Untuk mengatasi masalah vanishing gradient, arsitektur yang lebih canggih diperkenalkan:

1.  **LSTM (Long Short-Term Memory):** Ini adalah RNN yang jauh lebih kompleks. Alih-alih hanya memiliki satu hidden state, LSTM memelihara "jalur memori" terpisah yang disebut Cell State $C_t$.
2.  **Gated Mechanism:** LSTM menggunakan tiga "gerbang" (Gates) yang cerdas:
    *   **Forget Gate:** Memutuskan informasi apa yang harus dibuang dari memori (Cell State).
    *   **Input Gate:** Memutuskan informasi baru apa yang harus disimpan ke memori.
    *   **Output Gate:** Memutuskan apa yang harus dikeluarkan dari memori sebagai hidden state berikutnya $(H_t)$.
3.  **GRU (Gated Recurrent Unit):** Varian yang lebih sederhana dari LSTM yang menggabungkan Forget dan Input Gate, sehingga secara komputasi sedikit lebih ringan.

Meskipun LSTM dan GRU jauh lebih baik dalam menangani ketergantungan jangka panjang daripada RNN sederhana, mereka tetap memiliki dua kelemahan bawaan:
*   **Sekuensial:** Mereka harus memproses kata demi kata. Ini lambat dan tidak dapat diparalelkan. Anda tidak bisa menghitung $H_{10}$ sebelum Anda selesai menghitung $H_9$.
*   **Bottleneck:** Seluruh "pengetahuan" tentang kalimat (sekaya apa pun) masih harus dikompres menjadi satu vektor state ($H_t$ dan $C_t$).

Solusi dari kelemahan ini lah yang akan kita bahas di bagian selanjutnya.

---

#### 2.4.2 Ide Inti: Mekanisme Attention

Masalah bottleneck pada RNN/LSTM memicu ide baru ([Bahdanau et al., 2014](https://arxiv.org/pdf/1409.0473)) dalam konteks translasi mesin.

**Analogi: Ujian "Tutup Buku" vs "Buka Buku"**
*   **RNN/LSTM bekerja seperti mengerjakan ujian "Tutup Buku".** Model encoder membaca seluruh kalimat (misal Bahasa Inggris), mengompresnya menjadi satu vektor "memori" (rangkuman), lalu decoder harus menghasilkan terjemahan (Bahasa Indonesia) hanya berdasarkan rangkuman tunggal tersebut.
*   **Attention bekerja seperti ujian "Buka Buku".** Saat decoder hendak menghasilkan satu kata (misal, "cat"), ia diizinkan untuk "melihat kembali" ke seluruh kalimat input Bhs. Inggris dan memutuskan kata mana yang paling relevan untuk membantunya saat itu juga.

**Bagaimana Attention Bekerja (Konsep Q, K, V)?**

Secara konseptual, attention adalah mekanisme untuk menghitung "representasi kontekstual" dari sebuah query berdasarkan sekumpulan key-value pairs.
*   **Query (Q):** Vektor yang merepresentasikan "Apa yang saya cari saat ini?"
*   **Keys (K):** Sekumpulan vektor yang merepresentasikan "Label/Indeks dari semua informasi yang saya miliki".
*   **Values (V):** Sekumpulan vektor yang merepresentasikan "Informasi/Konten aktual yang saya miliki". (Dalam arsitektur awal, seringkali $V=K$).

Mekanisme yang menjadi standar emas (dan digunakan di dalam Transformer) untuk melakukan ini disebut **Scaled Dot-Product Attention**. Resep ini dapat dipecah menjadi tiga langkah utama:

**Proses (Scaled Dot-Product Attention):**

1.  **Hitung Skor (Score):** Kita membandingkan Query (Q) dengan setiap Key (K) menggunakan dot product untuk menghasilkan skor relevansi:
    $$Score = Q \cdot K^T$$
2.  **Normalisasi (Softmax):** Kita mengubah skor mentah menjadi distribusi probabilitas yang jumlahnya 1.0 menggunakan fungsi Softmax.
    $$Weights = Softmax\left(\frac{Score}{\sqrt{d_k}}\right)$$
    *(Dibagi $\sqrt{d_k}$ untuk stabilitas pelatihan, di mana $d_k$ adalah dimensi Key).*
3.  **Ambil Nilai (Weighted Sum):** Kita mengalikan probabilitas (Weights) ini dengan Values (V) yang sesuai.
    $$Output = Weights \cdot V$$
    Ini menghitung "rata-rata tertimbang" dari Values berdasarkan skor Weights.

![Gambar 8. Ilustrasi QKV](https://lms.sdmdigital.id/pluginfile.php/635364/mod_page/content/12/img23.png)
**Gambar 8. Ilustrasi QKV**

Gambar 8 mengilustrasikan alur komputasi dari Scaled Dot-Product Attention:
*   **Alur 1 (Perhitungan Bobot/Skor):** Input Q dan K dikalikan (Dot Product), kemudian di-Scale (dibagi $\sqrt{d_k}$) dan dimasukkan ke Softmax untuk menghasilkan Weights.
*   **Alur 2 (Konten):** Input Values (V) menunggu.
*   **Penyatuan (Weighted Sum):** Weights dan Values (V) dikalikan. Value yang relevan mendapat bobot tinggi, yang tidak relevan mendekati nol. Hasilnya adalah satu Output vektor yang kaya akan konteks.

Lompatan jenius yang sebenarnya datang dari paper "[Attention Is All You Need](https://arxiv.org/html/1706.03762v7)":
1. Bagaimana jika sebuah kalimat menggunakan attention pada **dirinya sendiri** (Self-Attention)?
2. Bagaimana jika kita melakukannya berkali-kali secara paralel (Multi-Head)?

---

#### 2.4.3 Evolusi: Self-Attention, Multi-Head, dan FlashAttention

Lompatan jenius berikutnya terjadi di paper "Attention Is All You Need" (Vaswani et al., 2017), yang menjadi dasar arsitektur Transformer.

**Self-Attention**
"Bagaimana jika Q, K, dan V semuanya berasal dari sekuens yang sama?"

> **Apa itu Self-Attention?**
> Self-Attention adalah sebuah mekanisme attention di mana sekuens input (misal, satu kalimat) memperhatikan dirinya sendiri. Setiap token dalam kalimat menghasilkan Q, K, dan V-nya sendiri, lalu menghitung skor attention terhadap setiap token lain dalam kalimat yang sama.

*   **Fungsi:** Memungkinkan model memahami konteks internal.
*   **Contoh:** Dalam kalimat "Kucing itu mengejar ekornya", self-attention memungkinkan token "nya" menyadari bahwa "Kucing" adalah token paling relevan, sehingga ia memahami siapa pemilik ekor tersebut.

**Multi-Head Attention (MHA)**
Daripada melakukan satu attention berdimensi besar, MHA membagi vektor menjadi beberapa bagian kecil ("kepala").

> **Apa itu Multi-Head Attention?**
> MHA adalah mekanisme yang menjalankan beberapa attention (misal, 8 atau 12 "kepala") secara paralel. Setiap head menghitung attention secara independen, lalu hasilnya digabungkan (*concatenate*).

*   **Analogi:** Seperti meminta 8 "ahli" melihat kalimat yang sama. Head 1 mungkin fokus pada Subjek-Predikat, Head 2 pada preposisi, Head 3 pada kata ganti. Ini menangkap berbagai relasi sintaksis dan semantik secara bersamaan.

**Evolusi Komputasi: FlashAttention**
Attention memiliki masalah komputasi kuadratik $(O(N^2))$. Untuk sekuens sepanjang $N=10.000$, model harus menghitung 100 juta skor, yang sangat boros memori.

**FlashAttention** adalah implementasi kernel GPU yang sangat efisien. Ia tidak mengubah arsitektur, melainkan cara komputasinya menggunakan teknik *tiling* (membagi matriks menjadi blok kecil) dan *recomputation*. Hasilnya adalah attention yang jauh lebih cepat dan hemat memori, memungkinkan pelatihan LLM pada sekuens yang sangat panjang.

---

### Mini Quiz

*Silakan akses Mini Quiz melalui platform LMS:*
[Multiple Choice Quiz - Subtopik 2.4](https://lms.sdmdigital.id/h5p/embed.php?url=https%3A%2F%2Flms.sdmdigital.id%2Fpluginfile.php%2F635364%2Fmod_page%2Fcontent%2F12%2Fmultiple-choice-207.h5p)

---

> **Refleksi Singkat**
> Kita telah membahas bahwa RNN/LSTM memiliki "bottleneck memori" karena mereka mencoba mengompres seluruh makna kalimat masa lalu ke dalam satu vektor state tunggal. Bayangkan analogi "Ujian Buka Buku". Jika decoder (yang menulis terjemahan) adalah "siswa" dan encoder (kalimat asli) adalah "buku teks", bagaimana mekanisme attention (Q, K, V) memungkinkan siswa untuk "mencari" jawaban di buku secara efisien, alih-alih hanya mengandalkan ingatan (satu vektor state tunggal) tersebut?