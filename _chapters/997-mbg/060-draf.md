---
slug: mbg-6
title: mbg-6 
---
# Bab 6: Sang Hakim Angka (Softmax)

Di bab-bab sebelumnya, kita sudah melihat bagaimana robot menghitung banyak hal untuk menebak sebuah kata atau benda. Tapi, ada satu masalah besar: **Bagaimana robot menentukan seberapa yakin dia dengan pilihannya?**

Bayangkan robot punya banyak tombol di depannya, dan setiap tombol mengeluarkan angka. Ada yang mengeluarkan angka 5, ada yang 10, bahkan ada yang 1000. Bagaimana cara kita (dan si robot) memahami angka-angka ini? Mari kita pelajari tugas si "Hakim Angka" yang bernama **Softmax**.

---

## 1. Masalah "Berapa Angka yang Cukup?"

Ingat saat kita melatih robot untuk membedakan daun dan bunga? Kita ingin robot memberikan angka tinggi untuk pilihan yang benar. Tapi, seberapa tinggi?

*   Apakah angka 5 sudah cukup?
*   Atau harus 10?
*   Atau 10 juta?

Secara teori, semakin tinggi angkanya, semakin bagus. Tapi kalau kita menyuruh robot mengejar angka sampai "tak terhingga", matematika di dalam otak robot akan rusak. Robot akan pusing karena angkanya terlalu besar untuk dihitung. Kita butuh cara agar angka-angka ini tetap masuk akal.

---

## 2. Masalah "Kelebihan Pintar"

Bayangkan robot sedang belajar. Di satu waktu, dia memberikan skor:
*   **Daun:** 5
*   **Bunga:** 1

Ini jawaban yang benar (Daun lebih besar). Tapi di waktu lain, setelah belajar lebih keras, dia memberikan skor:
*   **Daun:** 0
*   **Bunga:** 1

Nah, di kasus kedua, robot malah salah! Padahal angkanya kecil. Kita butuh cara untuk mengubah semua angka keluarannya menjadi rentang yang pasti, misalnya antara **0 sampai 1**. 

---

## 3. Mesin Pengecil Angka (Fungsi Logistik)

Dalam matematika, ada yang namanya **Fungsi**. Bayangkan fungsi itu seperti sebuah mesin: kamu masukkan satu angka, lalu mesin itu akan mengeluarkan angka lain berdasarkan aturan tertentu.

Salah satu mesin yang keren bernama **Fungsi Logistik**. 
*   Apa pun angka yang kamu masukkan (mau itu -1000 atau +1.000.000), mesin ini akan mengubahnya menjadi angka di antara **0 dan 1**.
*   Hebatnya, mesin ini tidak mengubah urutan. Angka yang awalnya paling besar akan tetap jadi yang paling besar setelah dikecilkan.

Sekarang, robot bisa memberikan jawaban seperti: "Skor Daun adalah 0.8 dan Bunga adalah 0.2". Ini jauh lebih mudah dipahami, bukan?

---

## 4. Kenapa Tidak Pilih yang Terbesar Saja?

Mungkin kamu bertanya: *"Kenapa kita tidak langsung saja ambil angka yang paling besar?"*

Mari kita lihat contoh kata **"Humpty Dumpty"**. 
Bayangkan robot sedang menebak huruf setelah `d-u-m-p-t`.
*   Pilihan A: **y** (Skornya sangat tinggi)
*   Pilihan B: **i** (Skornya sedikit di bawah 'y')

Jika robot selalu memilih yang paling besar (y), dia akan selalu menulis "Humpty Dumpty". Tapi terkadang, bahasa itu fleksibel. Jika robot melakukan kesalahan kecil di awal, misalnya salah menulis `d-u-u`, dia mungkin akan tersesat. 

Namun, jika kita tahu bahwa huruf **"m"** sebenarnya punya skor yang hampir sama besarnya dengan **"u"**, robot bisa punya pilihan lain. Kita tidak ingin robot menjadi kaku. Kita ingin robot tahu tentang **Peluang (Kemungkinan)**.

---

## 5. Mengenal Softmax: Sang Pembagi Peluang

**Softmax** adalah versi lebih canggih dari mesin pengecil angka tadi. Jika mesin logistik hanya mengolah satu angka, Softmax mengolah **semua angka** di lapisan terakhir secara bersamaan.

Inilah kehebatan Softmax:
1.  **Mengubah ke 0 sampai 1:** Semua angka hasil hitungan robot diubah menjadi angka kecil antara 0 dan 1.
2.  **Totalnya Harus Satu:** Ini yang paling penting! Softmax memastikan jika semua angka pilihan dijumlahkan, hasilnya harus **tepat 1 (atau 100%)**.

### Contoh Kerja Softmax:
Misalnya robot punya 3 pilihan kata berikutnya:
*   "Kucing": skor aslinya 10
*   "Anjing": skor aslinya 8
*   "Meja": skor aslinya 1

Setelah melewati Hakim Softmax, angkanya berubah menjadi:
*   "Kucing": **0.70 (70% peluang)**
*   "Anjing": **0.25 (25% peluang)**
*   "Meja": **0.05 (5% peluang)**
*   *(Jika dijumlahkan: 0.70 + 0.25 + 0.05 = 1)*

Sekarang robot bisa berkata: *"Aku 70% yakin jawabannya Kucing, tapi ada kemungkinan 25% itu Anjing."*

---

## 6. Kenapa Softmax Penting?

Hampir semua robot bahasa (LLM) menggunakan Softmax di bagian paling akhir otaknya. 
*   **Untuk Belajar:** Memudahkan ilmuwan menghitung seberapa jauh kesalahan robot (karena targetnya jelas, yaitu angka 1 untuk jawaban benar).
*   **Untuk Kreativitas:** Memungkinkan robot untuk kadang-kadang memilih jawaban pilihan kedua atau ketiga agar hasil tulisannya tidak membosankan dan lebih terasa seperti manusia.

---

### Ringkasan Bab 6:
*   Robot butuh cara agar angka hasil hitungannya tidak meledak menjadi terlalu besar.
*   **Fungsi** adalah mesin matematika yang mengubah satu angka menjadi angka lain.
*   **Softmax** adalah hakim yang mengubah semua skor robot menjadi **Peluang**.
*   Dengan Softmax, semua pilihan robot ada di antara **0 dan 1**, dan jika semuanya dijumlahkan, hasilnya adalah **1**.

**Sekarang robot kita sudah bisa menentukan pilihan dengan bijak. Di bab selanjutnya, kita akan melihat trik agar robot tidak mudah lelah atau bingung saat belajarnya semakin sulit!**