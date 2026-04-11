---
slug: modul-3-1
title: modul-3-1
---
### 3.1 Arsitektur LLM Modern (Autoregressive Decoder-Only) dan Evolusinya
---

Di Modul 2, kita telah membedah arsitektur Transformer penuh dan mengidentifikasi tiga keluarganya. Kita melihat arsitektur Encoder-Only (BERT) sebagai "Analis" yang hebat dalam pemahaman (NLU), dan arsitektur Encoder-Decoder (T5) sebagai "Penerjemah" yang hebat dalam tugas Seq2Seq yaitu pemetaan satu sekuens input ke satu sekuens output, seperti translasi.

Namun, arsitektur AI generatif saat ini mulai dari ChatGPT, Llama, Claude, hingga Grok sepenuhnya didominasi oleh keluarga **Decoder-Only.** Hal ini menimbulkan sebuah pertanyaan penting: Mengapa sebuah arsitektur yang tampak lebih sederhana di mana hanya menggunakan komponen Decoder dari arsitektur penuh dan memiliki batasan *unidirectional* (hanya bisa melihat masa lalu), karena *masking* justru mengungguli arsitektur Encoder-Decoder yang lebih kompleks?

Di subtopik ini, kita akan membedah anatomi arsitektur Decoder-Only ini, memahami mengapa kelemahannya justru menjadi **kekuatan terbesarnya** untuk pelatihan skala besar dan fleksibilitas tugas.

#### 3.1.1 Anatomi Arsitektur Decoder-Only (GPT-Style)

Arsitektur Decoder-Only, sering juga disebut arsitektur "GPT-style" (karena dipopulerkan oleh model seri GPT, atau Generative Pre-trained Transformer, dari OpenAI), pada dasarnya adalah Tumpukan Decoder dari Transformer original, namun dengan satu modifikasi krusial.

Mari kita ingat kembali komponen Blok Decoder orisinal dari Modul 2. Ia memiliki tiga sub-lapisan:
1. Masked Multi-Head Self-Attention (Melihat ke output-nya sendiri di masa lalu)
2. Encoder-Decoder Cross-Attention (Melihat ke output Encoder)
3. Feed-Forward Network (FFN).

Dalam arsitektur Decoder-Only, tidak ada Encoder. Konsekuensinya, Sub-lapisan 2 (Cross-Attention) dihapus seluruhnya, seperti yang diilustrasikan Gambar 1.

![Perbandingan Blok Decoder Asli vs. Blok Decoder-Only](https://lms.sdmdigital.id/pluginfile.php/635374/mod_page/content/19/image27.jpg)
**Gambar 1. Perbandingan Blok Decoder Asli vs. Blok Decoder-Only**

Arsitektur Decoder-Only adalah sebuah tumpukan (*stack*) Blok Decoder yang dimodifikasi, di mana setiap blok hanya terdiri dari dua sub-lapisan utama:
1. Masked Multi-Head Self-Attention
2. Position-wise Feed-Forward Network (FFN).

Tentu saja, lapisan Add & Norm (koneksi residual) tetap digunakan setelah setiap sub-lapisan, dan model menerima Token Embedding + Positional Encoding sebagai input awal. Seperti yang dibahas di Modul 2.5.2, Add & Norm adalah "lem" krusial dalam arsitektur Transformer.

*   **Add (Residual Connection):** Ini adalah "jalur pintas" yang menjumlahkan input sebelum lapisan MHA/FFN dengan output setelah lapisan MHA/FFN. Ini sangat penting untuk mencegah *vanishing gradient* dan memungkinkan pelatihan model yang sangat dalam (banyak lapis).
*   **Norm (Layer Normalization):** Ini adalah teknik yang menstabilkan proses pelatihan dengan menormalisasi (menyesuaikan skala) nilai-nilai vektor yang mengalir di antara lapisan.

Gambar 2 di bawah ini menunjukkan arsitektur Blok Decoder-Only yang disederhanakan ini secara lengkap dan alur datanya murni ***autoregressive***.

![Arsitektur Blok Decoder-Only (GPT-Style)](https://lms.sdmdigital.id/pluginfile.php/635374/mod_page/content/19/image53.jpg)
**Gambar 2. Arsitektur Blok Decoder-Only (GPT-Style)**

> **Apa itu Autoregressive?**
> *Autoregressive* (atau "regresi pada diri sendiri") adalah sebuah properti model di mana prediksi output saat ini (kata *t*) bergantung pada output dari langkah-langkah sebelumnya (kata *t*-1, *t*-2, ...). Sederhananya, ini adalah model yang "menulis kata demi kata", di mana setiap kata baru yang ditulisnya ditambahkan ke input untuk membantunya memutuskan kata berikutnya.

**Cara Kerjanya mirip seperti ini:**
1. Model mengambil input (misal: "Ibu kota Indonesia adalah") yang sudah di-embedding. Input tersebut melewati tumpukan Blok Decoder-Only (diulang sebanyak N lapis).
2. Output dari blok terakhir (yang tugasnya mengubah vektor final menjadi skor probabilitas) menghasilkan probabilitas untuk kata berikutnya (misal: "Jakarta").
3. Di langkah berikutnya, token "Jakarta" ditambahkan ke sekuens input, dan seluruh proses diulang untuk memprediksi kata setelah "Jakarta".

Proses *autoregressive* yang diuraikan di atas, yaitu di mana model memprediksi satu token, menambahkannya ke input, dan mengulang prosesnya—bukan hanya mekanisme menghasilkan kata berikutnya saja, tetapi secara fundamental juga adalah objektif pelatihannya. Arsitektur ini ternyata sangat cocok untuk metode pelatihan yang paling sederhana namun paling skalabel yaitu *Self-Supervised Learning*.

#### 3.1.2 Keunggulan: Pelatihan Self-Supervised dan In-Context Learning

> **Apa itu Self-Supervised Learning?**
> *Self-Supervised Learning* adalah sebuah metode pelatihan di mana label (jawaban) secara otomatis dihasilkan dari data itu sendiri, tanpa memerlukan anotasi manusia. Dalam konteks LLM (CLM), kita cukup mengambil teks mentah ("Kucing duduk di tikar"), menyembunyikan satu kata yang perlu diisi kata terakhir ("tikar"), dan meminta model menebaknya. Teks itu sendiri adalah data sekaligus "kunci jawaban". Inilah yang membuatnya sangat skalabel: kita bisa menggunakan triliunan halaman teks dari internet tanpa biaya pelabelan manual.

**1. Pelatihan *Self-Supervised* yang Sangat Efisien**
Arsitektur ini dilatih pada satu tugas yang sangat sederhana namun kuat yaitu prediksi Token Berikutnya (*Next-Token Prediction*). Ini adalah tugas yang 100% *self-supervised*.

*   **Analogi:** Bayangkan Anda memiliki seluruh koleksi buku di internet. Anda tidak perlu memberi label apa pun. Anda cukup mengambil sebuah kalimat, misal: "Kucing itu duduk di atas...", menyembunyikan kata terakhir "tikar", dan melatih model untuk menebak "tikar" tersebut.
*   Model Encoder-Decoder (T5) membutuhkan data berpasangan (misal: [Inggris -> Indonesia]). Model Encoder-Only (BERT) membutuhkan tugas yang lebih kompleks seperti Masked Language Model, atau MLM. Berbeda dengan Causal Language Model, atau CLM yang hanya menebak kata di akhir, MLM menyembunyikan kata di tengah kalimat, misal: "Kucing [MASK] di atas tikar" dan memaksa model untuk menggunakan konteks dari kedua arah (kiri dan kanan) untuk menebak kata yang hilang itu.
*   Model Decoder-Only hanya membutuhkan teks mentah. Ini memungkinkan para peneliti AI Generatif untuk melatihnya pada triliunan kata dari internet tanpa memerlukan biaya yang mahal. Pendekatan inilah yang secara harfiah melahirkan nama 'Generative Pre-training' (GP) dalam akronim GPT.

**2. Kemampuan Emergent: In-Context Learning (ICL)**
Sebuah perilaku yang tidak terduga (disebut *emergent ability* karena kemampuan ini "muncul" begitu saja, padahal model tidak pernah dilatih secara eksplisit untuk melakukannya), muncul dari pelatihan skala besar ini. Kemampuan ini dikenal sebagai In-Context Learning (ICL). Model Decoder-Only belajar untuk mengenali pola dalam prompt (input) dan menggunakannya sebagai "instruksi" atau "contoh" untuk menyelesaikan tugas, tanpa perlu melatih ulang modelnya.

> **Contoh untuk In-Context Learning / Few-Shot:**
> Anda dapat memberi model prompt berikut, padahal ia belum pernah dilatih khusus untuk tugas ini:
> Terjemahkan ke Prancis:
> kucing -> chat
> anjing -> chien
> keju -> ???
>
> Karena dilatih untuk "melanjutkan pola", model akan mengenali pola [Indonesia -> Prancis] dan menghasilkan: **fromage**.

Kemampuan ICL inilah yang membuat arsitektur Decoder-Only begitu fleksibel. Kita dapat "memprogram" model hanya dengan kata-kata (*prompting*), memintanya bertindak sebagai penerjemah, penulis kode, penyair, atau chatbot, hanya dengan mengubah konteks input.

#### 3.1.3 Evolusi: Hukum Skala (Scaling Laws)

Evolusi arsitektur Decoder-Only dari GPT-1 ke GPT-4 termasuk model lain seperti Llama, sebenarnya **bukanlah perubahan desain arsitektur** yang radikal. Komponen intinya tetap sama. Evolusinya didorong oleh penemuan ***Scaling Laws***.

> **Apa itu Scaling Laws?**
> *Scaling Laws* adalah temuan empiris yang menunjukkan bahwa kinerja model dapat diprediksi dengan sangat akurat hanya dengan melihat loss (kesalahan model) selama pelatihan. Kinerja ini terbukti meningkat secara log-linear dengan meningkatkan tiga elemen penting berikut:
> 1. **Ukuran Model (Jumlah Parameter):** Misal, dari 1.5 Miliar (GPT-2) ke 175 Miliar (GPT-3).
> 2. **Ukuran Data Latih:** Misal, dari 40GB (GPT-2) ke 45TB (GPT-3).
> 3. **Jumlah Komputasi atau Floating Point Operations (FLOPs):** ini adalah ukuran total "kerja" komputasi untuk melatih model.

**Apa itu Peningkatan Log-Linear?**
Peningkatan log-linear berarti kinerja model meningkat secara stabil dan bisa diprediksi.
**Analoginya** seperti skala Richter untuk gempa bumi: setiap peningkatan 1 poin berarti kekuatan gempa naik 10 kali lipat. Begitu pula dalam Scaling Laws—jika kita meningkatkan compute 10x lipat (misal dari 1 GPU-bulan ke 10 GPU-bulan), loss model akan turun dengan pola yang dapat diprediksi.

![Ilustrasi Grafik Scaling Laws](https://lms.sdmdigital.id/pluginfile.php/635374/mod_page/content/19/image39.jpg)
**Gambar 3. Ilustrasi Grafik Scaling Laws**

Gambar 3 secara visual mengilustrasikan *Scaling Laws* (Hukum Skala):
*   **Sumbu X (horizontal):** merepresentasikan Sumber Daya Komputasi (seperti jumlah GPU, ukuran data, atau jumlah parameter model) pada skala logaritmik. Bergerak ke kanan berarti peningkatan 10x, 100x, dst.
*   **Sumbu Y (vertikal):** merepresentasikan Kesalahan Model (Loss). Semakin rendah maka akan semakin "pintar" modelnya.
*   Garis lurus yang menurun adalah intinya. Ini menunjukkan bahwa kinerja model meningkat dan dapat diprediksi seiring dengan peningkatan sumber daya (skala).

Singkatnya, grafik ini menunjukkan bahwa jika Anda bersedia meningkatkan 10x lipat biaya komputasi, maka Anda dijamin akan mendapatkan model yang secara proporsional lebih baik (*loss*-nya turun secara drastis), yang mengubah pembangunan LLM dari "seni" menjadi "engineering".

> **Mengapa Ini Sangat Penting?**
> Sebelum *Scaling Laws*, membangun model AI yang lebih baik terasa seperti "seni". Tim peneliti mencoba berbagai arsitektur baru, hyperparameter baru, dan berharap mendapatkan hasil yang lebih baik. Penemuan ini memberikan "resep" yang dapat diprediksi: "Jika kita mengalokasikan *budget* komputasi X dan data sebesar Y TB, kita dapat memprediksi bahwa model Z miliar parameter akan mencapai tingkat loss A." Ini adalah "lampu hijau" bagi perusahaan untuk menginvestasikan miliaran dolar. Mereka sekarang tahu bahwa jika mereka melipatgandakan anggaran, bisa dipastikan akan mendapatkan model yang secara proporsional lebih baik.

Inilah yang memicu "perlombaan" skala besar yang kita lihat hari ini, di mana model-model terbesar yang dikenal sebagai Large Language Models atau LLMs, dilatih dengan biaya ratusan juta dolar. Singkatnya, evolusi arsitektur Decoder-Only bukanlah tentang menemukan desain blok yang baru, melainkan tentang menyadari bahwa desain blok yang sudah ada bekerja sangat baik jika ukurannya diperbesar secara masif.

#### Mini Quiz
*(Silakan akses kuis interaktif pada platform LMS untuk menguji pemahaman Anda)*

> **Refleksi Singkat**
>
> Kita telah melihat bahwa arsitektur Encoder-Only (BERT) sangat baik dalam "memahami" (NLU) dan arsitektur Decoder-Only (GPT) sangat baik dalam "menulis" (NLG).
>
> Mengingat arsitektur Decoder-Only dilatih hanya untuk "menebak kata berikutnya", mengapa ia bisa menjadi begitu fleksibel dan menunjukkan kemampuan In-Context Learning (ICL) untuk melakukan tugas-tugas yang tampaknya belum pernah dilatih (seperti menerjemahkan atau menjawab pertanyaan)?