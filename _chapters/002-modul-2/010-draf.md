---
slug: modul-2-1
title: modul-2-1
---
### 2.1 Memahami Masalah Pemodelan Bahasa Secara Statistik
---

Subtopik ini meletakkan fondasi matematis untuk seluruh modul. Sebelum kita dapat membangun arsitektur yang kompleks, kita harus terlebih dahulu mendefinisikan secara presisi *masalah* apa yang sebenarnya kita coba selesaikan. Fokus kita adalah memandang bahasa bukan sebagai seperangkat aturan tata bahasa yang kaku, tetapi sebagai fenomena statistik yang dapat dimodelkan.

![Gambar 2. Bahasa Pemodelan Statistik](https://lms.sdmdigital.id/pluginfile.php/635361/mod_page/content/25/Gambar2.jpg)
**Gambar 2. Bahasa Pemodelan Statistik**

Untuk mencapai ini, kita harus melakukan transisi fundamental. Kita beralih dari pendekatan linguistik tradisional (yang mungkin berfokus pada *preskripsi* tata bahasa, atau *parsing trees* yang kaku) ke pendekatan *empiris* dan *probabilistik*. Dalam paradigma baru ini, sebuah kalimat tidak lagi hanya "benar" atau "salah" secara *gramatikal*.

Sebaliknya, setiap kalimat memiliki tingkat "kemungkinan" atau "kewajaran" (*plausibility*) yang dapat diukur, yang kita estimasi dari data. Tugas kita sebagai *engineer* adalah membangun sebuah sistem yang dapat menghitung probabilitas ini.

Pertanyaannya menjadi: Bagaimana kita secara formal dan matematis merumuskan "probabilitas dari sebuah sekuens teks"? Langkah pertama dalam perjalanan komputasi ini adalah menetapkan definisi yang solid. Kita harus mendefinisikan secara tepat apa yang kita maksud dengan "model" dalam konteks "model bahasa".

---

#### 2.1.1 Definisi Formal: Model Bahasa (Language Model)

Secara fundamental, sebuah model bahasa (*language model* atau LM) adalah sebuah distribusi probabilitas atas sekuens-sekuens token (misalnya, kata atau karakter) dalam sebuah bahasa. Token adalah satu unit dari vocabulary sebuah model yang dihasilkan melalui proses tokenisasi, dan dapat berupa sebuah kata, potongan sub-kata (*subword piece*), karakter, atau simbol khusus yang diproses oleh LM.

Misalkan kita memiliki sebuah vokabulari $\mathcal{V}$, yaitu himpunan semua token unik yang dikenali oleh model. Sebuah sekuens $W$ didefinisikan sebagai $W = (w_{1'}, w_{2'}, \dots, w_n)$, di mana setiap $w_i \in \mathcal{V}$. Tugas dari sebuah model bahasa adalah untuk menetapkan sebuah probabilitas $P(W)$ ke seluruh sekuens $W$.

Probabilitas $P(W)$ ini merepresentasikan "kemungkinan" atau "kewajaran" dari sekuens tersebut. Sebuah LM yang baik akan memberikan probabilitas yang lebih tinggi untuk sekuens yang "masuk akal" (secara sintaksis dan semantik) dan probabilitas yang sangat rendah untuk sekuens yang tidak masuk akal.

> **Contoh Kasus:**
> Sebuah LM yang dilatih dengan baik pada korpus Bahasa Indonesia akan menghasilkan estimasi probabilitas sebagai berikut:
> $P(\{\text{"mahasiswa mengerjakan tugas akhir"}\}) \gg P(\{\text{"tugas akhir mengerjakan mahasiswa"}\})$
> Meskipun kedua kalimat tersebut menggunakan kata-kata yang valid, model statistik menangkap fakta bahwa urutan pada kalimat pertama jauh lebih sering terjadi dan lebih logis dalam bahasa.

Untuk menghitung probabilitas gabungan (*joint probability*) $P(W)$ dari sebuah sekuens, kita dapat menggunakan aturan rantai (*chain rule*) dari probabilitas. Aturan ini menguraikan probabilitas gabungan menjadi produk dari probabilitas kondisional:

$$P(W) = P(w_1, w_2, \dots, w_n) = P(w_1) \times P(w_2 \mid w_1) \times P(w_3 \mid w_1, w_2) \times \dots \times P(w_n \mid w_1, \dots, w_{n-1})$$

dimana $P(w_1|w_0)$ (atau $P(w_1)$) adalah probabilitas kata pertama (seringkali dikondisikan pada token BOS atau Beginning-of-Sequence).

Dekomposisi Aturan Rantai pada persamaan di atas ini lebih dari sekadar penyederhanaan matematis; ia adalah sebuah wawasan komputasional yang fundamental. Ia mengubah masalah yang tampak sangat sulit (mengestimasi probabilitas dari sebuah sekuens $W$ yang panjangnya $n$ secara holistik) menjadi serangkaian sub-masalah yang berurutan dan lebih mudah dikelola.

Alih-alih mencoba membangun satu model raksasa yang menerima $n$ kata dan menghasilkan satu nilai probabilitas $P(W)$, aturan rantai memberitahu bahwa kita dapat (dan seharusnya) membangun model yang hanya berfokus pada satu tugas: **mengestimasi probabilitas dari satu token berikutnya, dengan dikondisikan pada histori token sebelumnya**. Jika kita dapat membangun model yang andal untuk tugas $P(w_n \mid w_1, \dots, w_{n-1})$ ini, kita dapat menghitung probabilitas sekuens penuh $P(W)$ hanya dengan mengalikan output model tersebut secara berurutan. Pendekatan inilah yang melahirkan paradigma pelatihan dominan untuk model bahasa modern.

---

#### 2.1.2 Tugas Utama: Prediksi Kata Berikutnya (Next-Token Prediction)

Formulasi aturan rantai di atas secara alami mengubah masalah pemodelan bahasa menjadi tugas yang lebih mudah dikelola: **prediksi kata berikutnya**. Perhatikan komponen $P(w_n \mid w_1, \dots, w_{n-1})$. Ini adalah probabilitas token $w_i$ (token target) dengan syarat (diberikan) semua token sebelumnya (konteks $w_1, \dots, w_{n-1}$).

Sebagian besar Large Language Models modern (terutama arsitektur decoder-only seperti GPT) dilatih secara eksplisit untuk menyelesaikan tugas ini. Mereka adalah model probabilistik autoregresif yang jika diberikan sebuah konteks maka akan menghasilkan sebuah distribusi probabilitas atas seluruh vokabulari $\mathcal{V}$ untuk token berikutnya.

**Contoh Kasus:**
Misalkan model diberikan konteks: $W = (\text{"Siswa", "sedang", "belajar", "di"})$. Tugas model adalah menghitung $P(w_i \mid \{\text{"Siswa sedang belajar di"}\})$ untuk setiap $w_i$ di $\mathcal{V}$. Model yang baik akan menghasilkan distribusi probabilitas seperti yang diilustrasikan pada Tabel 1.

**Tabel 1. Contoh Distribusi Probabilitas Prediksi Token Berikutnya**

| Token Target ($w_n$) | Probabilitas ($P(w_n \mid \text{"Siswa sedang belajar di"})$) | Keterangan |
| :--- | :--- | :--- |
| "kelas" | 0.35 | Sangat relevan, Probabilitas Tinggi |
| "perpustakaan" | 0.25 | Relevan, Probabilitas tinggi |
| "laboratorium" | 0.15 | Relevan, Probabilitas sedang |
| "rumah" | 0.1 | Relevan, Probabilitas sedang |
| ... | ... | Tidak Relevan, Token lain dengan probabilitas rendah |
| "langit" | 0.000001 | Tidak relevan, Probabilitas sangat rendah |
| "mangga" | 0.0000005 | Tidak relevan, Probabilitas sangat rendah |
| ... | ... | Tidak relevan, Token lain dengan probabilitas yang sangat rendah |
| **Total keseluruhan $\mathcal{V}$** | $\sum P(w_i) = 1.0$ | **Probabilitas harus berjumlah 1** |

Tabel 1 mengilustrasikan properti fundamental dari output sebuah model bahasa. Dapat diamati bahwa model tidak sekadar 'memilih' satu kata terbaik. Sebaliknya, ia menghasilkan sebuah **distribusi probabilitas penuh** (seringkali melalui fungsi softmax, yang akan dibahas di Modul 3) atas keseluruhan vokabulari $\mathcal{V}$. Dengan melatih model untuk secara akurat memprediksi token berikutnya pada korpus teks yang sangat besar, model tersebut secara implisit "dipaksa" untuk mempelajari sintaksis, semantik, dan bahkan pengetahuan faktual tentang dunia untuk meminimalkan kesalahannya (loss function).

*Catatan: Perhatikan bahwa model masih memiliki kemungkinan untuk memilih perpustakaan daripada kelas walaupun kelas memiliki probabilitas yang lebih tinggi.*

Kemampuan untuk menghasilkan distribusi probabilitas yang akurat seperti pada Tabel 1 adalah tujuan akhir kita. Namun, formulasi $P(w_i \mid w_1, \dots, w_{n-1})$ menyembunyikan sebuah tantangan komputasi yang luar biasa besar, yang berakar pada sifat bawaan bahasa itu sendiri.

Masalahnya terletak pada **konteks kondisional** $(w_1, \dots, w_{n-1})$. Secara teori, untuk membuat prediksi yang sempurna pada kata ke-i, model harus mempertimbangkan seluruh histori yang mendahuluinya. Dalam bahasa nyata, histori ini bisa jadi sangat panjang dan memiliki variasi yang hampir tak terbatas. Apa yang terjadi ketika model menghadapi sebuah sekuens konteks yang valid secara linguistik, tetapi tidak pernah muncul (atau sangat jarang muncul) dalam miliaran kalimat data latihnya? Bagaimana model harus mengestimasi probabilitas untuk sesuatu yang belum pernah dilihatnya?

Di sinilah kita berhadapan langsung dengan dua rintangan kembar yang telah mendefinisikan bidang pemodelan bahasa statistik selama beberapa dekade: skala kombinatorial bahasa dan konsekuensi logisnya, yaitu kelangkaan data.

---

#### 2.1.3 Tantangan Fundamental: Sparsitas Data dan Skala Kombinatorial

Masalah pemodelan bahasa, terutama ketika didekati menggunakan aturan rantai penuh $P(w_i \mid w_1, \dots, w_{i-1})$, menghadapi dua tantangan komputasi dan statistik yang sangat besar yaitu sebagai berikut:

*   **Ledakan Kombinatorial (Combinatorial Explosion):** Jumlah kemungkinan sekuens kata tumbuh secara eksponensial seiring dengan panjang sekuens $n$ dan ukuran vokabulari. Jika $|\mathcal{V}| = 50,000$ (standar umum), jumlah sekuens dengan panjang 10 adalah $50,000^{10}$, sebuah angka yang sangat besar.
*   **Sparsitas Data (Data Sparsity):** Sparsitas Data adalah sebuah fenomena dimana sebagian besar sekuens kata yang valid secara teoritis tidak pernah muncul (atau sangat jarang muncul) dalam data latih (corpus), tidak peduli seberapa besar korpus tersebut.

Akibat sparsitas data, estimasi berbasis hitungan langsung (sekadar "menghitung kemunculan") sering memberi probabilitas nol untuk urutan yang tak pernah terlihat, sehingga model gagal menggeneralisasi. Sebagai contoh, untuk mengestimasi $P(\text{"Transformer"} \mid \text{"Arsitektur neural modern yang disebut"})$ maka, kita mungkin tidak akan pernah menemukan konteks "Arsitektur neural modern yang disebut" di data latih kita, dan jika kita hanya mengandalkan perhitungan frekuensi, probabilitas untuk konteks yang tidak terlihat (*unseen contexts*) akan menjadi nol, yang merupakan estimasi yang buruk.

![Gambar 3. Visualisasi Ruang Sekuens Teoretis vs. Data Korpus Terobservasi](https://lms.sdmdigital.id/pluginfile.php/635361/mod_page/content/25/image%20%283%29.png)
**Gambar 3. Visualisasi Ruang Sekuens Teoretis vs. Data Korpus Terobservasi**

Gambar 3 mengilustrasikan masalah dimana ruang dari semua kemungkinan kalimat sangat besar, sementara korpus data latih kita (sekalipun miliaran dokumen) hanya mencakup sebagian kecil dari ruang tersebut.

Tantangan sparsitas inilah yang memotivasi evolusi pemodelan bahasa. Kita memerlukan model yang dapat menggeneralisasi dari data yang terlihat ke data yang tidak terlihat. Model harus belajar bahwa konteks $\text{"Siswa belajar di"}$ dan $\text{"Mahasiswa belajar di"}$ memiliki kemiripan semantik, dan oleh karena itu, prediksi kata berikutnya harus serupa.

Kita telah menetapkan bahwa tantangan sparsitas dan skala kombinatorial (2.1.3) adalah rintangan fundamental. Menghadapi masalah yang begitu sulit, sebuah pertanyaan logis akan muncul: Mengapa kita mengerahkan begitu banyak upaya komputasi dan penelitian hanya untuk mengestimasi $P(W)$?

Jawabannya adalah karena pemodelan bahasa merupakan masalah hulu (*upstream problem*) yang sangat fundamental dalam NLP dan memiliki implikasi langsung ke banyak aplikasi nyata. Keberhasilan dalam mengaproksimasi $P(W)$ secara langsung membuka kemampuan untuk melakukan berbagai tugas hilir (*downstream tasks*) yang sangat bernilai. Kemampuan untuk menilai "kewajaran" sebuah kalimat secara statistik sangat esensial untuk hampir semua hal yang kita asosiasikan dengan "pemahaman" bahasa. Oleh karena itu, sebelum kita menyelam ke cara memecahkan masalah ini (dimulai dengan n-gram di 2.2), subtopik penutup ini akan memetakan implikasi dan aplikasi praktis dari apa yang kita dapatkan jika kita berhasil.

---

#### 2.1.4 Implikasi dan Aplikasi Pemodelan Bahasa

Kemampuan untuk memodelkan $P(W)$ secara akurat memiliki implikasi yang sangat luas di luar sekadar "menebak kata berikutnya". Model bahasa yang kuat berfungsi sebagai fondasi untuk berbagai tugas Natural Language Processing.

**Tabel 2. Aplikasi Inti dari Model Bahasa Statistik**

| Aplikasi | Deskripsi Cara Kerja |
| :--- | :--- |
| Teks Generasi (Autoregresif) | Menghasilkan teks baru dengan mengambil sampel (sampling) berulang kali dari distribusi $P(W)$ |
| Peringkasan Teks | Model Encoder-Decoder (Subtopik 2.5) dilatih untuk menghasilkan sekuens ringkasan yang memaksimalkan probabilitas $P(\{\text{Ringkasan}\})$ |
| Translasi Mesin | Model dilatih untuk memaksimalkan probabilitas sekuens terjemahan (misal, Bahasa Indonesia) yang dikondisikan pada sekuens input (misal, Bahasa Inggris), yaitu $P(W)$ |
| Pengenalan Ucapan (Voice Recognition) | Dari sinyal audio, sistem ASR menghasilkan beberapa hipotesis transkrip. LM kemudian digunakan untuk "menilai" transkrip mana yang paling mungkin (memiliki $P(W)$ tertinggi) secara linguistik. |
| Koreksi Ejaan | Model dapat mengevaluasi $P(\{\text{"Saya pergi ke pasar"}\}) > P(\{\text{"Saya prgi ke pasar"}\})$ dan menyarankan koreksi yang memiliki probabilitas lebih tinggi. |

Kita telah mendedikasikan Subtopik 2.1 untuk menetapkan apa masalah pemodelan bahasa (mengestimasi $P(W)$) dan mengapa ia penting (aplikasi di Tabel 2). Kita juga telah mengidentifikasi rintangan utamanya: sparsitas data (Gambar 3), yang membuat estimasi $P(w_n \mid w_1, \dots, w_{n-1})$ untuk histori penuh menjadi mustahil secara komputasi.

Di subtopik berikutnya, kita beralih ke "bagaimana" kita secara praktis mengaproksimasi solusi $P(W)$. Untuk mengatasi masalah sparsitas, kita harus membuat sebuah asumsi penyederhanaan (*simplifying assumption*).

Pendekatan statistik klasik pertama adalah dengan mengadopsi asumsi Markov: alih-alih melihat seluruh histori, kita berasumsi probabilitas sebuah kata hanya bergantung pada $n-1$ kata sebelumnya. Asumsi inilah yang melahirkan model n-gram, fondasi statistik yang akan kita bedah di subtopik berikutnya.

---

### Mini Quiz

*Silakan akses Mini Quiz melalui tautan berikut:*
[Multiple Choice Quiz - Subtopik 2.1](https://lms.sdmdigital.id/h5p/embed.php?url=https%3A%2F%2Flms.sdmdigital.id%2Fpluginfile.php%2F635361%2Fmod_page%2Fcontent%2F25%2Fmultiple-choice-197.h5p)

---

> **Refleksi Singkat**
> *Dalam Subtopik 2.1, kita telah mendefinisikan masalah $P(W)$. Bayangkan sebuah pendekatan di mana kita mengestimasi probabilitas sebuah sekuens hanya berdasarkan frekuensi kemunculan eksplisitnya di dalam korpus data latih. Jika sebuah sekuens yang valid secara semantik (misal: "astronom mengamati supernova") kebetulan tidak pernah muncul (memiliki 0 kemunculan) dalam data latih kita, berapa nilai probabilitas yang akan diberikan oleh pendekatan tersebut? Fenomena apa yang diilustrasikan oleh Gambar 3 yang menjelaskan masalah fundamental ini?*