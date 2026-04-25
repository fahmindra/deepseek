---
slug: modul-2-5
title: modul-2-5
---

### 2.5 Arsitektur Transformer dan Komponen Penyusun Sebuah LLM
---

Pada Subtopik 2.4, kita mengakhiri dengan sebuah pertanyaan radikal: "Bagaimana jika kita membuang arsitektur rekuren (RNN/LSTM) sepenuhnya, dan membangun model yang hanya bergantung pada attention?" Inilah pertanyaan yang dijawab oleh paper dari Vaswani et al. (2017): "Attention Is All You Need". Mereka mengusulkan sebuah arsitektur baru yang non-sekuensial (paralel) yang disebut Transformer.

![Gambar 9. Arsitektur Transformer](https://lms.sdmdigital.id/pluginfile.php/635365/mod_page/content/10/image.png)

**Gambar 9. Arsitektur Transformer**

Arsitektur inilah yang menjadi "cetak biru" (blueprint) untuk hampir setiap LLM modern yang kita gunakan saat ini, mulai dari BERT hingga GPT dan Llama. Di subtopik inilah kita akhirnya akan "membongkar" Gambar 1 (yang kita perkenalkan di awal modul) dan memahami bagaimana semua bagian yang telah kita pelajari (Tokenisasi, Embedding, Q, K, V, dan MHA) bekerja sama dalam satu sistem yang koheren. Kita akan membedah arsitektur ini dari level tertinggi hingga ke komponen-komponen inti yang menyusunnya.

---

#### 2.5.1 Gambaran Arsitektur Penuh (Encoder-Decoder Stack)

Arsitektur Transformer orisinal (seperti yang ditunjukkan pada Gambar 1 dan Gambar 9) adalah model sequence-to-sequence (seq2seq) yang dirancang untuk tugas seperti translasi mesin (misal, mengubah Bahasa Inggris ke Bahasa Indonesia). Arsitektur ini terdiri dari dua tumpukan (stacks) komponen utama:

1.  **Encoder Stack (Sisi Kiri):**
    *   **Tugas:** "Membaca" dan "memahami" (meng-encode) seluruh kalimat input secara holistik.
    *   **Analogi:** Ini adalah seorang analis atau pembaca ahli. Ia membaca seluruh kalimat input (misal: "the cat sat on the mat") sekaligus, dan membangun representasi kontekstual yang kaya untuk setiap token (misal, ia tahu "mat" terkait dengan "sat on").
    *   **Input:** Vektor Embedding + Positional Encoding (dari 2.3).
    *   **Output:** Satu set vektor kontekstual (K dan V) yang kaya makna, yang akan digunakan oleh Decoder.
2.  **Decoder Stack (Sisi Kanan):**
    *   **Tugas:** "Menulis" atau "menghasilkan" (meng-decode) kalimat output token demi token (secara autoregressive).
    *   **Analogi:** Ini adalah seorang penulis yang cerdas. Saat ia hendak menulis kata ke-3 (misal, "duduk"), ia melakukan dua hal: (1) Melihat apa yang sudah ia tulis (misal, "kucing itu"), dan (2) "Melihat" (via attention) ke seluruh catatan analis (Encoder) untuk mendapatkan konteks input.

Di bagian akhir (paling atas Decoder), terdapat Linear Layer dan Softmax Function (yang akan kita bahas di Modul 3) yang mengambil vektor output akhir dari Decoder dan mengubahnya menjadi distribusi probabilitas (seperti Tabel 1) untuk memilih kata berikutnya.

Sekarang, mari kita bedah apa yang ada di dalam satu blok Encoder dan satu blok Decoder.

---

#### 2.5.2 Bedah Komponen: Blok Encoder

"Encoder Stack" (Sisi Kiri Gambar 9) sebenarnya adalah tumpukan dari Blok Encoder yang identik. Setiap blok memiliki dua sub-lapisan (sub-layers) utama.

![Gambar 10. Blok Encoder](https://lms.sdmdigital.id/pluginfile.php/635365/mod_page/content/10/image%20%281%29.png)

**Gambar 10. Blok Encoder dari Arsitektur Transformer**

1.  **Sub-layer 1: Multi-Head Self-Attention (MHA)**
    Ini adalah implementasi langsung dari Self-Attention yang kita pelajari di 2.4.3.
    *   **Fungsi:** Ini adalah lapisan di mana setiap token di kalimat input "melihat" dan "berdiskusi" dengan setiap token lain di kalimat yang sama. Di sinilah model belajar hubungan kontekstual internal (misal, menghubungkan "nya" dengan "kucing" dalam "kucing mengejar ekornya").
    *   **Input:** Vektor dari lapisan sebelumnya (atau dari Embedding di lapisan pertama).
    *   **Output:** Vektor baru yang kaya konteks (setelah semua "diskusi").

2.  **Sub-layer 2: Position-wise Feed-Forward Network (FFN)**
    *   **Fungsi:** Ini adalah neural network standar (terdiri dari dua Linear layers dengan aktivasi ReLU) yang memproses setiap vektor token secara independen.
    *   **Analogi:** Jika MHA adalah "rapat tim" (di mana semua token berdiskusi), FFN adalah "waktu kerja individu" (di mana setiap token "memikirkan" dan memproses hasil rapat tersebut secara pribadi).

**"Lem" Penghubung: Add & Norm (Residual Connections)**
Perhatikan bahwa setelah setiap sub-layer (MHA dan FFN), terdapat langkah yang disebut "Add & Norm". Ini adalah Residual Connection yang diikuti oleh Layer Normalization.

> **Apa itu Residual Connection (Add)?**
> Ini adalah "jalur pintas". Kita mengambil input sebelum masuk ke MHA/FFN (kita sebut X) dan menambahkannya (X + SubLayer(X)) ke output dari MHA/FFN. Tujuan: Ini sangat krusial untuk melatih deep network. Ini mencegah masalah vanishing gradient dengan memastikan bahwa sinyal asli (X) selalu memiliki "jalur cepat" untuk terus maju, bahkan jika lapisan MHA/FFN gagal belajar apa pun.

---

#### 2.5.3 Bedah Komponen: Blok Decoder

Blok Decoder (Sisi Kanan Gambar 1 dan Gambar 9) sedikit lebih kompleks karena ia harus melakukan dua tugas: (1) memperhatikan apa yang sudah ditulisnya, dan (2) memperhatikan input dari Encoder. Oleh karena itu, ia memiliki tiga sub-lapisan.

![Gambar 11. Blok Decoder](https://lms.sdmdigital.id/pluginfile.php/635365/mod_page/content/10/image%20%282%29.png)

**Gambar 11. Blok Decoder dari Arsitektur Transformer**

1.  **Sub-layer 1: Masked Multi-Head Self-Attention**
    Ini mirip dengan Self-Attention di Encoder, tetapi dengan satu perbedaan krusial: Masking (Penutupan).
    *   **Fungsi:** Decoder bersifat autoregressive, artinya ia menghasilkan kata demi kata. Saat ia sedang memprediksi kata ke-3, ia tidak boleh diizinkan untuk "melihat" atau "mencontek" kata ke-4 atau ke-5 (kata-kata dari masa depan).
    *   **Masking** adalah proses di mana kita secara paksa mengatur skor attention untuk semua token di masa depan menjadi $-\infty$ (minus tak terhingga) sebelum Softmax. Ini memastikan bahwa saat memprediksi $Y_{t'}$, model hanya bisa attention ke $Y_{1'}, \dots, Y_{t-1}$.

2.  **Sub-layer 2: Multi-Head Encoder-Decoder Attention**
    *   **Fungsi:** Inilah implementasi sebenarnya dari analogi "Ujian Buka Buku" (2.4.2). Di sinilah Decoder (penulis) "melihat" ke catatan Encoder (analis).
    *   **Q, K, V:**
        *   **Query (Q):** Berasal dari output Sub-layer 1 (Masked MHA). Ini adalah pertanyaan "Mengingat apa yang sudah saya tulis, apa yang perlu saya cari dari input?".
        *   **Keys (K) & Values (V):** Berasal dari output akhir Encoder Stack. Ini adalah "catatan" dari kalimat input.
    *   Lapisan ini memungkinkan Decoder untuk menyelaraskan (align) outputnya dengan input yang relevan (misal, saat menerjemahkan "cat", ia akan attention kuat ke token "kucing" di Encoder).

3.  **Sub-layer 3: Position-wise Feed-Forward Network (FFN)**
    Sama seperti di Encoder. Ini adalah lapisan "pemrosesan" akhir sebelum output blok ini diteruskan ke blok Decoder berikutnya (atau ke Softmax di akhir).

Semua sub-lapisan ini juga dihubungkan dengan **Add & Norm.**

---

### Mini Quiz

*Silakan akses Mini Quiz melalui platform LMS:*
[Multiple Choice Quiz - Subtopik 2.5](https://lms.sdmdigital.id/h5p/embed.php?url=https%3A%2F%2Flms.sdmdigital.id%2Fpluginfile.php%2F635365%2Fmod_page%2Fcontent%2F10%2Fmultiple-choice-209.h5p)

---

> **Refleksi Singkat**
> Kita telah membedah arsitektur Transformer penuh (Gambar 9) dan melihat bahwa ia memiliki dua bagian: Encoder Stack (kiri) dan Decoder Stack (kanan). Pikirkan tentang tugas translasi (misal, Inggris ke Indonesia). Mengapa model ini membutuhkan kedua tumpukan tersebut? Apa fungsi spesifik dari Encoder yang tidak bisa dilakukan oleh Decoder, dan sebaliknya?