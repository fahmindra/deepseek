---
slug: mbg-8
title: mbg-8 
---
# Bab 8: Merapikan Barisan Angka (Layer Normalization)

Di dalam otak robot, ada jutaan angka yang bertebaran. Kadang-kadang, angka-angka ini sangat berantakan. Ada angka yang sangat besar (seperti 1.000) dan ada angka yang sangat kecil (seperti 0,001). 

Jika angka-angkanya terlalu berbeda jauh, robot akan kesulitan menghitung dan menjadi "pusing". Agar robot tetap tenang dan pintar, kita butuh asisten perapi angka yang disebut **Layer Normalization** (Normalisasi Lapisan).

---

## 1. Apa Itu Normalisasi?

Bayangkan kamu memiliki sekelompok teman dengan tinggi badan yang sangat berbeda. Ada yang setinggi raksasa dan ada yang seukuran semut. Tugas **Layer Normalization** adalah merapikan mereka sehingga semuanya memiliki ukuran yang "standar" dan mudah diatur.

Cara merapikannya menggunakan dua rumus matematika sederhana:
1.  **Mencari Rata-rata (Mean):** Kita sebut saja **M**.
2.  **Mencari Selisih Standar (Standard Deviation):** Kita sebut saja **D**.

Setiap angka asli (kita sebut **x**) akan diubah menjadi angka baru dengan rumus:
**Angka Baru = (x - M) / D**

Dengan rumus ini, angka yang tadinya berantakan akan menjadi lebih rapi dan stabil. Ini sangat membantu robot saat ia memiliki lapisan otak yang sangat dalam agar ia tidak bingung di tengah jalan.

---

## 2. Masalah: Bagaimana Jika Informasi Penting Hilang?

Ada satu kekhawatiran: jika kita merapikan semua angka menjadi standar yang sama, apakah kita akan membuang informasi yang penting? 

Misalnya, jika kita merapikan tinggi semua orang menjadi sama, kita jadi tidak tahu lagi siapa yang sebenarnya paling tinggi, bukan?

Untuk mengatasi hal ini, para ilmuwan menambahkan dua "tombol pengatur" tambahan di dalam Layer Normalization:
1.  **Skala (Scale):** Angka dikalikan lagi dengan angka tertentu.
2.  **Geseran (Bias):** Angka ditambah dengan angka tertentu.

Hebatnya, robot akan **belajar sendiri** berapa angka yang pas untuk tombol Skala dan Bias ini. Jadi, robot bisa merapikan angka-angkanya tapi tetap menyimpan informasi yang ia anggap penting.

---

## 3. Bentuk Blok Layer Normalization

Jika digambarkan, kotak Layer Normalization terlihat seperti ini:

![Layer Normalization](https://towardsdatascience.com/wp-content/uploads/2024/10/1145xBeyaNCU3mblFQyi2_A.png)
*Gambar: Proses merapikan angka dengan Skala dan Bias (Karya Rohit Patel)*

Di dalam kotak ini, setiap angka diproses satu per satu. Ini mirip dengan "Pintu Ajaib" (Activation Layer/ReLU) yang kita bahas di Bab 1, bedanya di sini ada tombol Skala dan Bias yang bisa dilatih agar robot makin pintar.

---

## 4. Cara Menghitung "Selisih Standar" (Standard Deviation)

Mungkin kamu penasaran, bagaimana cara menghitung **D** atau Selisih Standar tadi? Ini adalah cara untuk melihat seberapa jauh angka-angka kita tersebar. Jika semua angkanya sama, maka selisihnya adalah nol.

Begini langkah-langkahnya (misalnya kita punya sekelompok angka):
1.  Cari **Rata-rata** dari semua angka itu.
2.  Kurangi setiap angka dengan rata-rata tadi.
3.  **Kuadratkan** hasilnya (kalikan dengan dirinya sendiri, supaya semua angka jadi positif).
4.  Jumlahkan semua hasil kuadrat itu.
5.  Bagi dengan jumlah total angka yang ada.
6.  Terakhir, cari **Akar Kuadratnya** ($\sqrt{x}$).

Memang terdengar agak panjang, tapi bagi robot, ini adalah hitungan yang sangat cepat!

---

## 5. Catatan untuk Calon Profesor

Mungkin nanti kamu akan mendengar istilah lain bernama *Batch Normalization*. Namun, di dalam buku ini kita tidak membahasnya. 

Kenapa? Karena *Batch Normalization* hanyalah trik tambahan untuk mempercepat latihan robot, tapi tidak mengubah rahasia utama tentang bagaimana robot memahami bahasa. Untuk membuat robot LLM (seperti ChatGPT), **Layer Normalization** jauh lebih penting!

---

### Ringkasan Bab 8:
*   **Layer Normalization** bertugas merapikan angka agar tidak terlalu besar atau terlalu kecil.
*   Caranya adalah dengan mengurangi **Rata-rata** dan membagi dengan **Selisih Standar**.
*   Ada tombol **Skala** dan **Bias** agar robot tidak kehilangan informasi penting.
*   Proses ini membuat "otak" robot yang sangat dalam tetap stabil dan mudah diajari.

**Sekarang angka-angka robot sudah rapi! Di bab selanjutnya, kita akan belajar trik lucu: bagaimana robot belajar dengan cara "pura-pura lupa"! Penasaran? Yuk, lanjut!**