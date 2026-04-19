---
slug: mbg-7
title: mbg-7 
---
# Bab 7: Jembatan Pengingat (Residual Connections)

Seiring berjalannya petualangan kita, cara kita menggambarkan otak robot mulai berubah. Kita tidak lagi menggambar ratusan lingkaran dan garis satu per satu karena itu akan membuat buku ini sangat penuh! Sekarang, kita menggunakan **Kotak** atau **Blok** untuk mewakili kumpulan proses yang rumit.

Dalam bab ini, kita akan mempelajari sebuah ide jenius yang disebut **Residual Connection** (Koneksi Residu) atau bisa kita sebut sebagai "Jembatan Pengingat".

---

## 1. Menggambar Ulang Otak Robot

Perhatikan gambar di bawah ini. Ini adalah cara ilmuwan menggambarkan bagaimana informasi mengalir di dalam otak robot yang canggih:

![Koneksi Residu](https://towardsdatascience.com/wp-content/uploads/2024/10/1270MXDfslVtvmBjShHL2hQ.png)
*Gambar: Sebuah Koneksi Residu (Karya Rohit Patel)*

Di gambar ini, ada kotak bernama **Input** (Masukan) dan **Output** (Keluaran). Meskipun bentuknya kotak, ingatlah bahwa di dalamnya tetap berisi ribuan angka atau neuron seperti yang kita pelajari di bab-bab awal.

---

## 2. Apa yang Terjadi di Gambar Itu?

Mari kita telusuri jalannya angka-angka tersebut:
1.  **Jalan Utama:** Angka-angka dari **Input** masuk ke dalam **Blok Self-Attention** untuk diproses dan dicari mana kata yang penting.
2.  **Jalan Pintas:** Lihat garis yang melengkung di atas? Itu adalah jalan pintas! Angka-angka dari **Input** juga dikirim langsung ke depan tanpa diubah-ubah.
3.  **Pertemuan (Simbol +):** Di ujung jalan, hasil dari Blok Self-Attention **ditambahkan** dengan angka asli dari Input tadi.

Jadi, hasilnya adalah: **Hasil Olahan + Informasi Asli.**

---

## 3. Aturan Main: Ukuran Harus Sama

Ada satu syarat penting agar jembatan ini bisa bekerja. Karena kita sedang melakukan **tambah-tambahan (+)**, maka ukuran kotak "Hasil Olahan" dan kotak "Informasi Asli" harus **sama persis**.

Bayangkan jika kamu punya dua kotak kelereng:
*   Kotak pertama punya 10 lubang.
*   Kotak kedua punya 10 lubang.
*   Kamu bisa menggabungkan isinya dengan mudah karena lubangnya pas.

Jika ukurannya berbeda (misalnya yang satu punya 10 angka dan yang satu punya 5 angka), robot akan bingung dan tidak bisa menjumlahkannya. Untungnya, para ilmuwan bisa mengatur agar Blok Self-Attention selalu mengeluarkan jumlah angka yang pas dengan inputnya.

---

## 4. Kenapa Robot Butuh Jalan Pintas Ini?

Kamu mungkin bertanya: *"Kenapa kita repot-repot menambahkan kembali angka yang asli? Bukankah kita ingin hasil yang baru?"*

Ternyata, ada alasan yang sangat penting:

### Masalah "Pesan yang Hilang"
Semakin pintar seorang robot, semakin banyak lapisan (layer) di otaknya. Bayangkan ada 100 lapisan! Saat robot belajar, informasi harus melewati 100 lapisan ini. 
Kadang-kadang, di tengah jalan, informasi aslinya bisa menjadi kacau, hilang, atau terlupakan karena terlalu banyak dikalikan dan ditambah. Ini membuat robot sangat sulit untuk dilatih (belajar).

### Solusi: Jembatan Pengingat
Dengan adanya **Residual Connection**, robot selalu punya "salinan asli" dari apa yang ia baca sebelumnya. 
*   Jika di dalam Blok Self-Attention informasinya menjadi agak membingungkan, robot masih punya informasi asli dari jalan pintas tadi.
*   Ini membantu robot untuk tetap stabil dan lebih mudah belajar meskipun otaknya sangat tebal dan berlapis-lapis (yang disebut sebagai *Deep Networks*).

---

## 5. Kesimpulan Bab 7

Residual Connection adalah seperti jembatan yang membawa informasi asli untuk digabungkan dengan hasil olahan robot. 
*   **Caranya:** Menjumlahkan Input langsung ke Output.
*   **Syaratnya:** Ukuran angka harus sama.
*   **Gunanya:** Agar robot tidak "lupa" atau "bingung" saat belajar di otak yang sangat dalam dan rumit.

**Sekarang kita tahu cara menjaga ingatan robot tetap kuat. Di bab selanjutnya, kita akan belajar cara merapikan angka-angka robot agar tidak berantakan!**