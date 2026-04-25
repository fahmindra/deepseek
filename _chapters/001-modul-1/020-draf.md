---
slug: modul-1-2
title: modul-1-2
---

### 1.2. Konsep Matematika Penting untuk Memahami LLM
---

Sebelum masuk ke bagian modul yang lebih dalam membahas mengenai cara kerja dan komponen pembangun sebuah LLM, mari terlebih dahulu mengulang kembali beberapa konsep matematika yang relevan untuk dipahami kembali sebelum mempelajari cara kerja LLM dan komponen dibaliknya.

### **1.2.1 Matriks dan Tensor**

Sebuah matriks merupakan susunan angka dalam kolom dan baris yang membentuk sebuah persegi panjang. Sebuah matriks memiliki 2 dimensi yang mewakili baris dan kolomnya. Matriks biasanya digunakan sebagai representasi yang compact untuk mendeskripsikan hubungan antara dua index pada dimensi yang berbeda dalam sebuah operasi, yang biasanya digunakan untuk mendeskripsikan sebuah transformasi linear yang dilakukan dengan melakukan operasi matriks. Sebagai contoh, operasi linear untuk melakukan transformasi input dengan dimensi 3 dan output dengan dimensi 2 bisa direpresentasikan sebagai berikut:

![Representasi operasi linear matriks](https://lms.sdmdigital.id/pluginfile.php/635350/mod_page/content/15/image21.png)
**Gambar 12. Representasi operasi linear menggunakan matriks.**

Notasi ∈ℝM⨉N artinya suatu matriks merupakan anggota dari himpunan matriks bilangan real dengan ukuran M⨉N.

b₁₁ merepresentasikan hubungan antara a₁ dengan y₁, b₁₂ merepresentasikan hubungan antara a₂ dengan y₁, dan seterusnya. Perkalian matriks *B* merepresentasikan suatu transformasi linear dari R₃→R₂: setiap keluaran adalah kombinasi linear dari komponen input. Secara teori, setiap transformasi linear dapat direpresentasikan sebagai perkalian matriks, dengan ukuran matriks yang harus sesuai (jumlah baris sama dengan dimensi output, dan jumlah kolom sama dengan dimensi input). Penambahan bias *C* akan menggeser hasil transformasi tersebut. Secara geometris, hal ini mengubah transformasi linear menjadi transformasi *affine* (linear dan translasi): *y=Bx+C*. Tanpa bias, semua pemetaan “melewati” titik asal (origin). Dengan bias, hasil dapat digeser sehingga tidak harus melalui origin. Hal ini penting pada banyak model (misalnya jaringan saraf atau SVM), agar *hyperplane* atau garis keputusan tidak dipaksa melewati origin.

Sebuah matriks dengan dimensi A x B bisa difaktorisasi untuk merepresentasikannya sebagai perkalian dua matriks dengan dimensi A x R dan R x B. Dimensi R bisa memiliki angka berapapun. Dekomposisi matriks memiliki banyak kegunaan, seperti mendapatkan insight penting mengenai struktur matriks (PCA, SVD) atau merepresentasikan sebuah matriks yang besar sebagai perkalian dua matriks kecil dengan memilih nilai R yang lebih kecil dari A dan B. Merepresentasikan sebuah matriks besar sebagai perkalian dua matriks yang lebih kecil ini disebut sebagai low-rank decomposition. Contohnya, sebuah matriks M kita representasikan sebagai perkalian dari dua matriks lain yaitu matriks U dan matriks V.

![Ilustrasi low-rank decomposition](https://lms.sdmdigital.id/pluginfile.php/635350/mod_page/content/15/image26.png)
**Gambar 13. Ilustrasi *low-rank decomposition*.**

Dengan memfaktorkan matriks besar tersebut menjadi perkalian dua matriks yang berukuran lebih kecil, komputasi menjadi lebih sederhana dan kebutuhan memori untuk penyimpanan juga berkurang.

![Rasio kompresi matriks](https://lms.sdmdigital.id/pluginfile.php/635350/mod_page/content/15/image3.png)
**Gambar 14. Dengan memilih R yang cukup kecil di bawah A dan B, rasio kompresi yang cukup tinggi bisa dicapai.**

Matriks dapat digeneralisasi untuk mendeskripsikan hubungan antara berapapun jumlah indeks sebagai **Tensor**. Matriks sendiri bisa disebut sebagai sebuah **Tensor** dua dimensi. Operasi **Tensor** merupakan konsep dasar yang membangun berbagai macam algoritma *Deep Learning*.

---

### **1.2.2 Vektor**

Secara geometris, vektor adalah besaran yang memiliki nilai dan arah di dalam suatu ruang. Vektor dapat dipandang sebagai deretan angka yang masing-masing merepresentasikan posisi pada sumbu tertentu. Berikut adalah contoh vektor dengan 3 komponen:

![Contoh vektor 3 komponen](https://lms.sdmdigital.id/pluginfile.php/635350/mod_page/content/15/image19.png)

Dalam aljabar linier, sebuah vektor umumnya direpresentasikan sebagai matriks kolom. Vektor di atas dapat direpresentasikan sebagai berikut:

![Representasi vektor matriks kolom](https://lms.sdmdigital.id/pluginfile.php/635350/mod_page/content/15/image11.png)

Dalam konteks Machine Learning, vektor berfungsi sebagai representasi numerik dari suatu entitas baik yang bersifat konkret seperti kata, gambar, atau wajah, maupun yang bersifat abstrak seperti makna atau hubungan antar konsep. Melalui representasi ini, mesin dapat memahami dan memanipulasi hubungan antar entitas secara geometris, sehingga kedekatan makna atau kemiripan konsep dapat diukur sebagai jarak antar titik dalam ruang.

Operasi vektor yang paling relevan dengan *Machine Learning* adalah operasi linear yang sudah dijelaskan di bagian matriks, dan mengukur kedekatan antara dua vektor. Dalam Machine Learning, kedekatan antara dua vektor biasanya diukur menggunakan *L1 Distance, L2 Distance, dan Cosine Similarity*.

![Rumus L1, L2, dan Cosine Similarity](https://lms.sdmdigital.id/pluginfile.php/635350/mod_page/content/15/image27.png)
**Gambar 15. Ukuran kedekatan vektor yang biasanya digunakan dalam *Machine Learning*.**

**L1 dan L2 Distance**

Secara fisis, bila setiap vektor dibayangkan sebagai sebuah panah yang menunjuk ke suatu arah dalam ruang, L1 dan L2 *Distance* menghitung jarak antara dua titik yang ditunjuk oleh masing-masing vektor.

**Cosine Similarity**

*Cosine similarity* mengukur sudut antara dua vektor untuk mengukur seberapa mirip arah dari dua vektor. Berbeda dengan L1 dan L2 Distance, *Cosine Similarity* memiliki sebaran nilai yang terbatas, antara 0 dan 1 dan tidak terlalu memperdulikan nilai dari masing-masing vektor, hanya kemiripan arahnya saja.

Masing masing ukuran kedekatan bisa divisualkan sebagai berikut:

![Visualisasi L1, L2, dan Cosine Similarity](https://lms.sdmdigital.id/pluginfile.php/635350/mod_page/content/15/gambar%2016.jpg)
**Gambar 16. Ilustrasi kedekatan vektor. L2 mengukur jarak langsung antara dua titik yang ditunjuk vektor, L1 mengukur total jarak horizontal dan vertikal antara dua vektor, *Cosine similarity* mengukur kemiripan antara dua vektor berdasarkan cosinus dari sudut di antara keduanya (orange).**

---

### **1.2.3 Distribusi Peluang dan Proses Sampling**

Distribusi peluang, yang pada umumnya dinotasikan dengan notasi matematis P, adalah sebuah fungsi yang menggambarkan bagaimana kemungkinan setiap hasil dari suatu percobaan acak tersebar. Dalam sebuah distribusi peluang, semua hasil yang mungkin sudah diketahui lebih dahulu, baik berupa rentang nilai untuk distribusi kontinyu maupun himpunan nilai diskrit untuk distribusi diskrit. Distribusi ini menunjukkan seberapa besar peluang masing-masing hasil untuk muncul. Sebagai contoh, ketika melempar sebuah dadu ideal, terdapat enam kemungkinan hasil yang setara, dan setiap sisi memiliki peluang sebesar ⅙ untuk muncul setelah dadu dilempar. Dengan demikian, distribusi peluang memberikan cara untuk memahami dan memodelkan ketidakpastian dari suatu proses acak. Apabila dijumlahkan, peluang semua kemungkinan akan memiliki total 1, yang artinya semua kemungkinan yang terjadi sudah masuk di dalam distribusi peluang.

Untuk mendapatkan suatu nilai dari distribusi peluang, dilakukan proses yang disebut *sampling*. Proses sampling dapat dipandang sebagai tindakan mengambil contoh acak dari suatu distribusi. Setiap hasil sampling mewakili satu kemungkinan realisasi dari proses acak tersebut. Jika *sampling* dilakukan berkali-kali, kumpulan hasilnya akan membentuk pola yang secara statistik menyerupai bentuk distribusi aslinya. Dengan kata lain, semakin banyak sampel yang diambil, semakin jelas distribusi peluang yang mendasarinya akan tampak.

Secara intuitif, proses ini dapat diibaratkan seperti mengambil bola dari sebuah kantong yang berisi bola dengan berbagai warna dalam proporsi tertentu. Setiap kali kita mengambil satu bola, kita sedang melakukan satu kali sampling. Jika kita mengulanginya banyak kali, frekuensi kemunculan setiap warna akan mendekati perbandingan aslinya di dalam kantong.

![Ilustrasi proses sampling bola](https://lms.sdmdigital.id/pluginfile.php/635350/mod_page/content/15/gambar%2017.jpg)
**Gambar 17. Ilustrasi proses *sampling* (mengambil bola) dari sebuah kotak yang sudah diketahui peluang setiap jenis warna yang bisa diambil dari kotak tersebut.**

Sampling bisa dilakukan *with replacement* dimana hasil *sampling* dimasukkan kembali ke ruang sampel yang menyebabkan distribusi peluang selalu tetap atau *without replacement* dimana hasil *sampling* tidak dimasukkan kembali ke ruang sampel, sehingga setiap pengambilan sampel akan mengubah distribusi peluangnya.

Konsep sampling sangat penting dalam *machine learning*, karena banyak model yang menghasilkan prediksi sebagai sebuah distribusi peluang atas berbagai kemungkinan. Proses *sampling* dari distribusi inilah yang nantinya menentukan hasil konkret yang dihasilkan model, seperti memilih satu kata dari sekian banyak kata yang mungkin muncul dalam sebuah kalimat.

---

### **Mini Quiz**

*Silakan akses quiz interaktif melalui tautan di bawah ini:*

1. [Quiz 1 - Operasi Matriks](https://app.lumi.education/api/v1/run/UeqTWd/embed)
2. [Quiz 2 - Kedekatan Vektor](https://app.lumi.education/api/v1/run/mqF0bf/embed)
3. [Quiz 3 - Distribusi dan Sampling](https://app.lumi.education/api/v1/run/Gmxeam/embed)

---

> ***Refleksi Singkat***
> 
> Dari semua konsep di atas, coba hubungkan dengan apa yang kalian telah pelajari dalam *Machine Learning* dan *Deep Learning*. Dimanakah setiap konsep di atas digunakan?