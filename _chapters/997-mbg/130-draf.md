---
slug: mbg-13
title: mbg-13 
---
# Bab 13: Lampiran — Tas Alat Sang Ilmuwan AI

Selamat! Kamu sudah sampai di bagian terakhir buku ini. Di bab-bab sebelumnya, kita sering mendengar istilah-istilah matematika yang terdengar keren. Nah, di bab lampiran ini, kita akan membuka "Tas Alat" para ilmuwan untuk melihat bagaimana cara kerja alat-alat tersebut secara lebih mendalam. 

Jangan takut dengan namanya yang sulit, karena sebenarnya ini hanyalah permainan angka yang seru!

---

## 1. Perkalian Matriks (Matrix Multiplication)

Di Bab 1 dan 3, kita belajar tentang **Vektor** (daftar angka) dan **Matriks** (tabel angka). Matriks punya dua ukuran: jumlah baris (ke samping) dan jumlah kolom (ke bawah).

### Apa itu Perkalian Matriks?
Perkalian matriks sebenarnya adalah cara cepat untuk melakukan banyak hitungan sekaligus. Bayangkan kamu punya data masukan (Vektor) dan ingin mengalikannya dengan semua "tombol pengatur" (Weights) di otak robot.

Perhatikan gambar "Titik-titik Ajaib" ini:

![Perkalian Matriks](https://towardsdatascience.com/wp-content/uploads/2024/10/0woel5Da5Z22EmiGx.png)
*Gambar: Cara angka-angka bertemu dan dikalikan (Karya Rohit Patel)*

Tanda titik di gambar tersebut artinya **Dikali**. 

### Contoh di Otak Robot
Ingat neuron biru dan oranye di Bab 1? Jika kita menuliskan semua berat garisnya dalam sebuah tabel (Matriks **W**) dan angka masukannya dalam sebuah daftar (Vektor **x**), robot bisa menghitung semuanya dalam satu kedipan mata:

![Matriks dalam Robot](https://towardsdatascience.com/wp-content/uploads/2024/10/0yn1TPuxw_QqnD93k.png)
*Gambar: Masukan dikali Tabel Berat = Hasil Pikiran Robot (Karya Rohit Patel)*

Jadi, **W dikali x** adalah cara robot menghasilkan lapisan pikiran baru. Para ilmuwan sangat menyukai ini karena daripada menghitung satu per satu, mereka bisa langsung menghitung satu tabel sekaligus!

---

## 2. Simpangan Baku (Standard Deviation)

Kita menggunakan alat ini di Bab 8 tentang **Layer Normalization**. Simpangan Baku adalah cara statistik untuk mengukur seberapa "berantakan" atau "tersebar" angka-angka kita.

*   Jika semua angka hampir sama (misal: 5, 5, 6, 5), maka simpangan bakunya **kecil (mendekati nol)**.
*   Jika angkanya sangat jauh berbeda (misal: 1, 100, 50, 200), maka simpangan bakunya **besar**.

### Resep Memasak Simpangan Baku:
Untuk mencari angka ini, robot mengikuti resep 6 langkah:
1.  **Cari Rata-rata:** Jumlahkan semua angka, lalu bagi dengan banyaknya angka.
2.  **Kurangi:** Setiap angka dikurangi dengan hasil rata-rata tadi.
3.  **Kuadratkan:** Hasil pengurangan tadi dikalikan dengan dirinya sendiri (supaya tidak ada angka negatif).
4.  **Jumlahkan:** Tambahkan semua hasil kuadrat tadi.
5.  **Bagi:** Bagi hasil jumlahnya dengan banyaknya angka.
6.  **Akar Kuadrat:** Cari akar kuadratnya (seperti mencari angka berapa yang jika dikali dirinya sendiri menghasilkan angka tersebut).

Hasil akhirnya adalah satu angka yang memberi tahu robot seberapa "heboh" angka-angka di dalam otaknya.

---

## 3. Kode Posisi Gelombang (Positional Encoding)

Di Bab 10, kita belajar tentang memberi label nomor urutan. Ada cara canggih untuk melakukan ini tanpa harus menyuruh robot belajar sendiri, yaitu menggunakan rumus **Positional Encoding**.

### Kenapa Tidak Pakai Angka Biasa (1, 2, 3)?
Jika kita pakai angka 1, 2, 3, dan seterusnya:
*   Untuk kalimat yang sangat panjang, angkanya bisa jadi jutaan. Ini membuat robot pusing karena angkanya terlalu besar.
*   Jika kita kecilkan (misal dibagi total kata), angkanya akan berubah-ubah terus tergantung panjang kalimatnya. Robot jadi bingung karena "nomor antrean 1" tidak selalu punya nilai yang sama.

### Solusi: Gelombang Sinus dan Kosinus
Para ilmuwan menggunakan gelombang (seperti gelombang suara atau laut). Mereka membuat 10 gelombang yang berbeda-beda kecepatannya. 

Rumusnya terlihat menyeramkan: $s_i(p) = \sin(p/10000^{(i/d)})$. Tapi artinya sederhana:
*   Kita membuat pola naik-turun yang unik untuk setiap posisi.
*   Karena kita pakai 10 gelombang sekaligus, setiap posisi akan punya "sidik jari" gelombang yang tidak akan kembar dengan posisi lainnya.
*   Angkanya akan selalu berada di antara **-1 dan 1**, jadi tidak akan pernah meledak menjadi terlalu besar.

Ilmuwan menggunakan campuran gelombang **Sinus** (naik-turun biasa) dan **Kosinus** (gelombang yang mulainya agak beda sedikit) untuk membuat kode posisi yang super akurat.

---

## Kesimpulan Akhir
Sekarang kamu sudah tahu rahasia terdalam dari arsitektur **Transformer**! Mulai dari tambah-tambahan sederhana, cara robot memberikan perhatian, sampai rumus gelombang yang rumit. 

Semua ini adalah blok-blok bangunan yang membuat AI menjadi sangat pintar. Jika kamu tertarik untuk mencoba membuatnya sendiri, ada banyak kode di internet (seperti *nanoGPT*) yang menggunakan rumus-rumus tepat seperti yang baru saja kita pelajari.

**Terima kasih sudah membaca dan selamat menjadi penjelajah dunia masa depan!**