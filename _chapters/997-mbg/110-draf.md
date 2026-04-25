---
slug: mbg-11
title: mbg-11
---
# Bab 11: Merakit Robot GPT yang Hebat

Selama sepuluh bab terakhir, kita sudah mempelajari semua "onderdil" atau potongan Lego untuk membuat robot pintar. Sekarang, saatnya kita menyatukan semua potongan itu untuk membangun sebuah mahakarya bernama **Arsitektur GPT**.

GPT adalah singkatan dari *Generative Pre-trained Transformer*. Ini adalah teknologi yang digunakan oleh robot-robot pintar paling terkenal di dunia saat ini. Jika kamu sudah mengikuti bab-bab sebelumnya, bagian ini akan terasa sangat mudah!

---

## 1. Peta Besar Arsitektur GPT

Mari kita lihat gambar "peta" atau denah rumah robot GPT di bawah ini. Bayangkan ini adalah sebuah pabrik besar tempat angka-angka diproses:

![Arsitektur GPT](https://towardsdatascience.com/wp-content/uploads/2024/10/0U9mQKCWiyakNVwxU.png)
*Gambar: Denah besar robot GPT (Karya Rohit Patel)*

Mari kita telusuri jalannya dari bawah ke atas:

1.  **Input (Masukan):** Kalimat yang kamu ketik masuk ke sini.
2.  **Word Embedding:** Kata-kata diubah menjadi kode angka (Vektor).
3.  **Positional Embedding:** Urutan kata diberi label nomor antrean.
4.  **Simbol Tambah (+):** Kedua kode angka tadi (Kata + Posisi) dijumlahkan menjadi satu. Ingat, supaya bisa ditambah, kedua daftar angka ini harus **sama panjangnya**!
5.  **GPT Transformer Block:** Ini adalah "Stasiun Utama" tempat semua pemikiran rumit terjadi. Biasanya ada banyak stasiun seperti ini yang ditumpuk-tumpuk.
6.  **Linear & Softmax:** Di bagian paling atas, robot menggunakan "Hakim Angka" untuk memilih satu kata terbaik sebagai jawabannya.

---

## 2. Di Dalam "Stasiun Utama" (GPT Transformer Block)

Jika kita membedah kotak **GPT Transformer Block**, inilah yang ada di dalamnya:

![Isi GPT Block](https://towardsdatascience.com/wp-content/uploads/2024/10/1Mq4hBZcKPL9GALPSSQG9Xg.png)
*Gambar: Apa yang terjadi di dalam stasiun utama (Karya Rohit Patel)*

Di dalam stasiun ini, terjadi siklus kerja yang sangat rapi:
*   **Multi-head Attention:** Pertama, robot menggunakan "banyak mata" untuk melihat kata mana yang paling penting.
*   **Add & Norm:** Hasilnya digabungkan dengan "Jembatan Pengingat" (Add) agar tidak lupa informasi asli, lalu dirapikan oleh "Stasiun Perapi" (Norm).
*   **Feed-forward:** Robot menggunakan otak matematikanya untuk berpikir lebih dalam tentang informasi yang sudah didapat.
*   **Add & Norm (Lagi):** Sebelum keluar dari stasiun, hasilnya sekali lagi dilewatkan ke jembatan pengingat dan dirapikan kembali agar benar-benar siap.

Satu blok ini disebut "Transformer" karena tugasnya adalah mengubah (*transform*) angka-angka masukan menjadi angka-angka jawaban yang jauh lebih pintar.

---

## 3. Mengingat Kembali Perjalanan Kita

Mari kita lihat betapa hebatnya kamu! Kamu sudah mempelajari semua ini:

1.  **Robot adalah Matematika:** Kita tahu robot mengambil angka, melakukan kali dan tambah, lalu mengeluarkan angka lagi. Ia punya "tombol pengatur" (Weights) yang bisa dilatih.
2.  **Angka Punya Arti:** Kita bisa memberikan arti pada angka-angka itu, seperti "ini bunga" atau "ini kata selanjutnya".
3.  **Kotak-kotak Ajaib:** Kita bisa merantai banyak proses menjadi satu dan menyebutnya sebagai "Blok". Setiap blok punya tugas khusus.
4.  **Jenis-jenis Blok:** Kita sudah belajar tentang Blok Perhatian (Attention), Blok Perapi (Normalization), dan Jembatan Pengingat (Residual).
5.  **GPT adalah Susunan Spesial:** GPT sebenarnya hanyalah sebuah **susunan khusus** dari blok-blok yang sudah kita pelajari tadi.

Para ilmuwan di perusahaan-perusahaan besar terus memperbaiki susunan ini agar robotnya makin pintar, tapi rahasia dasarnya tetap sama dengan yang kamu pelajari sekarang.

---

## 4. Rahasia Nama: Sang Penjawab (Decoder)

Ada satu rahasia kecil lagi. Di dalam jurnal ilmiah pertama yang memperkenalkan teknologi ini, susunan blok seperti GPT ini sebenarnya disebut sebagai **"Decoder"** (Sang Penjawab). 

Kenapa disebut *Decoder*? Karena tugasnya adalah "memecahkan kode" informasi yang masuk untuk menghasilkan jawaban yang bisa dimengerti manusia. 

Di bab terakhir nanti, kita akan melihat "kakak" dari GPT, yaitu arsitektur **Transformer** yang asli, yang memiliki satu bagian tambahan lagi selain *Decoder*.

---

### Ringkasan Bab 11:
*   **Arsitektur GPT** adalah gabungan dari semua blok yang sudah kita pelajari: Embedding, Posisi, Attention, dan Normalization.
*   Dua jenis Embedding (Kata dan Posisi) harus dijumlahkan sebelum masuk ke blok utama.
*   **GPT Transformer Block** adalah tempat kerja utama yang berisi banyak mata (Attention) dan proses berpikir (Feed-forward).
*   GPT sebenarnya adalah sebuah **Decoder**, bagian dari robot yang bertugas memberikan jawaban.

**Hebat! Kamu sudah memahami rahasia teknologi paling canggih saat ini. Tinggal satu langkah lagi untuk menyelesaikan petualangan ini!**