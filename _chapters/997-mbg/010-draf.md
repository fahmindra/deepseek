---
slug: mbg-1
title: mbg-1
---

# Bab 1: Mesin Angka yang Bisa Menebak

Pernahkah kamu membayangkan bagaimana sebuah robot bisa membedakan mana bunga dan mana daun? Padahal, robot tidak punya mata seperti kita. Ternyata, rahasianya adalah **angka-angka**.

Di bab ini, kita akan belajar bagaimana "otak" robot bekerja. Ingat satu aturan penting ini: **Robot hanya mengerti angka.** Robot tidak tahu apa itu "merah", "harum", atau "hijau". Ia hanya tahu angka seperti 1, 10, atau 0,5.

## 1. Rahasia Bahasa Robot: Semuanya Adalah Angka
Bayangkan kamu punya mesin ajaib. Kamu tidak bisa memasukkan benda ke dalamnya, kamu hanya bisa memasukkan kertas berisi angka. 

Jika kamu ingin robot mengenali sebuah **Daun** atau **Bunga Matahari**, kamu harus mengubah ciri-ciri benda itu menjadi angka. Contohnya:
*   **Warna:** Kita beri angka untuk warna merah, hijau, dan biru (disebut RGB).
*   **Ukuran:** Kita ukur berapa besar benda itu dalam mililiter.

Sekarang, kita punya 4 angka untuk diberikan kepada robot.

| Ciri-ciri | Daun (Angka) | Bunga Matahari (Angka) |
| :--- | :--- | :--- |
| Merah | 32 | 245 |
| Hijau | 107 | 190 |
| Biru | 56 | 20 |
| Ukuran | 11.2 | 45.5 |

## 2. Bagaimana Robot Mengambil Keputusan?
Robot punya dua cara untuk memberitahu kita apa yang ia lihat:
1.  **Cara Satu Tombol:** Robot mengeluarkan satu angka. Jika angkanya positif (+), itu Daun. Jika negatif (-), itu Bunga.
2.  **Cara Dua Tombol:** Robot mengeluarkan dua angka. Angka pertama untuk "Daun", angka kedua untuk "Bunga". Mana angka yang **lebih besar**, itulah jawaban si robot.

Kita akan menggunakan **Cara Dua Tombol** karena ini yang paling sering digunakan oleh robot-robot pintar zaman sekarang.

## 3. Di Dalam Otak Robot (Neural Network)
Bayangkan di dalam kotak robot ada kabel-kabel yang menghubungkan lingkaran-lingkaran. Kita menyebut lingkaran ini **Neuron** (seperti sel saraf di otak kita).

### Bagian-bagian Otak Robot:
*   **Neuron (Lingkaran):** Tempat menyimpan angka.
*   **Weights/Bobot (Garis Berwarna):** Ini adalah "tingkat kepentingan". Setiap garis punya angka yang akan mengalikan angka dari neuron sebelumnya.
*   **Layer (Lapisan):** Kelompok lingkaran. Ada lapisan **Masuk** (Input), lapisan **Tengah** (Sembunyi), dan lapisan **Keluar** (Output).

### Cara Menghitung (Forward Pass)
Ini adalah saat robot mulai "berpikir". Caranya sangat mudah, hanya **Kali** dan **Tambah**:
1.  Ambil angka dari depan.
2.  **Kalikan** dengan angka yang ada di garis (Bobot).
3.  **Tambahkan** semua hasilnya ke lingkaran berikutnya.

**Contoh:**
Jika angka warna merah adalah 32 dan angka di garis adalah 0.10, maka robot menghitung: $32 \times 0.10 = 3.2$. Ia melakukan ini untuk semua angka dan menjumlahkannya. 

Jika di akhir perjalanan, angka di tombol "Daun" lebih besar daripada tombol "Bunga", maka robot akan berteriak: **"Ini adalah Daun!"**

## 4. Robot Bisa Jadi Apa Saja!
Hal yang paling keren adalah robot ini tidak tahu kalau ia sedang menebak daun. Jika kamu mengganti angkanya dengan data cuaca (seperti kelembapan atau awan), robot yang sama bisa menebak: **"Hari ini akan Hujan!"** atau **"Hari ini akan Cerah!"**.

Robot hanya mesin hitung. Kitalah yang menentukan apa arti angka-angka itu.

## 5. Bagaimana Cara Robot Belajar? (Training)
Awalnya, robot itu sangat bodoh. Saat pertama kali dinyalakan, angka di garis-garisnya (Bobot) masih acak-acakan. Ia mungkin akan menebak bunga padahal itu jelas-jelas daun.

Lalu, bagaimana ia jadi pintar? Caranya disebut **Latihan (Training)**.

### Langkah-langkah Robot Belajar:
1.  **Menebak:** Robot mencoba menebak menggunakan angka acak.
2.  **Melihat Kesalahan (Loss):** Kita beri tahu robot jawaban yang benar. "Hei Robot, jawaban yang benar harusnya Daun (1.0) dan Bunga (0.0). Tapi kamu malah menjawab Daun (0.6) dan Bunga (0.4)!"
3.  **Menghitung Selisih:** Kita hitung seberapa jauh kesalahan robot. Selisih ini disebut **Loss**. Kita ingin nilai Loss ini sekecil mungkin (sampai nol).
4.  **Memperbaiki Diri (Gradient Descent):** Robot akan mengubah sedikit demi sedikit angka di garis-garisnya (Bobot). 
    *   "Oh, kalau angka ini aku naikkan sedikit, jawabanku jadi lebih benar."
    *   "Oh, kalau angka itu aku kurangi, jawabanku jadi makin salah."
5.  **Ulangi Terus:** Robot melakukan ini ribuan kali sampai ia tidak pernah salah lagi.

## 6. Info Tambahan untuk Calon Profesor Cilik
Ada beberapa hal rahasia yang dilakukan ilmuwan agar robot lebih hebat:
*   **Pintu Ajaib (Activation Layer):** Kadang robot mengubah angka negatif menjadi nol agar otaknya tidak pusing. Ini disebut **ReLU**.
*   **Angka Bonus (Bias):** Selain angka di garis, setiap lingkaran punya angka bonus kecil untuk membantu penghitungan agar lebih akurat.
*   **Merapikan Jawaban (Softmax):** Di akhir, robot mengubah jawabannya menjadi persen. Contohnya: "Aku 99% yakin ini Daun, dan cuma 1% yakin ini Bunga."

---

**Kesimpulan Bab 1:**
Robot tidak melihat dunia dengan mata, tapi dengan angka. Ia belajar dengan cara mencoba, salah, lalu memperbaiki angka-angka di dalam "otaknya" sampai ia menjadi ahli penebak yang hebat!

**Siap untuk bab selanjutnya? Kita akan melihat bagaimana robot belajar membaca kata-kata!**