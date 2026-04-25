---
slug: mbg-12
title: mbg-12 
---
# Bab 12: Sang Maha Guru — Arsitektur Transformer

Inilah bab terakhir sekaligus puncak dari semua yang kita pelajari! Kita akan mengenal **Transformer**. Ini bukan robot yang bisa berubah jadi mobil, tapi ini adalah penemuan paling hebat di dunia AI saat ini. 

Transformer membuat robot tidak hanya jadi lebih akurat, tapi juga lebih cepat belajar. Bayangkan sebuah sekolah yang awalnya hanya punya satu guru, sekarang punya ribuan asisten yang bekerja sangat efisien. Inilah yang mendasari robot GPT yang kita bahas sebelumnya.

---

## 1. Masalah Besar: Menerjemahkan Bahasa

Robot GPT yang kita pelajari di Bab 11 sangat jago menebak kata selanjutnya. Tapi bagaimana jika tugasnya adalah **menerjemahkan**? 

Misalnya, kita punya kalimat bahasa Jerman: **"Wo wohnst du?"** dan kita ingin robot mengubahnya ke bahasa Inggris: **"Where do you live?"**.

Dulu, ilmuwan mencoba cara sederhana: menggabungkan kalimat Jerman dan Inggris jadi satu, lalu dipisahkan dengan tanda pembatas. Tapi cara ini punya banyak kelemahan:
1.  **Ingatannya Terbatas:** Jika kalimat Jermannya terlalu panjang, robot bisa "lupa" kata-kata di awal karena jendelanya (Context Length) penuh.
2.  **Bingung Dua Bahasa:** Robot harus belajar dua bahasa sekaligus dalam satu waktu dan harus tahu kapan dia harus mulai menerjemahkan.
3.  **Boros Tenaga:** Robot harus memproses ulang seluruh kalimat Jerman berkali-kali untuk setiap kata Inggris yang dia hasilkan.

---

## 2. Solusi Jenius: Si Pendengar (Encoder) dan Si Pembicara (Decoder)

Untuk mengatasi masalah itu, Transformer diciptakan dengan dua bagian utama yang bekerja sama seperti sebuah tim:

*   **Encoder (Si Pendengar/Penerjemah):** Tugasnya hanya satu, yaitu membaca kalimat bahasa Jerman tadi dan mengubahnya menjadi sebuah "Catatan Rahasia" berisi angka-angka.
*   **Decoder (Si Pembicara/Penjawab):** Tugasnya adalah menulis kalimat bahasa Inggris. Tapi, saat dia menulis, dia tidak hanya melihat kata yang sudah dia tulis, dia juga **terus-menerus mengintip** "Catatan Rahasia" dari si Encoder tadi.

Jadi, meskipun kalimatnya panjang, si Pembicara tidak akan lupa apa maksud kalimat aslinya karena dia punya catatan lengkap dari si Pendengar.

---

## 3. Membedah Denah Asli Transformer

Mari kita lihat denah asli yang dibuat oleh para ilmuwan saat pertama kali menciptakan Transformer. Jangan bingung dengan garis-garisnya, kita akan bedah satu per satu!

![Denah Transformer](https://towardsdatascience.com/wp-content/uploads/2024/10/1uPwJ24na4D_ECHBv1Lg5Fg.png)
*Gambar: Arsitektur Transformer (Karya Vaswani dkk., 2017)*

### Bagian-Bagian Penting:

#### A. Kotak Berpikir Mendalam (Feed Forward)
Di setiap lantai bangunan robot ini, ada kotak bernama **Feed Forward**. Ini adalah tempat robot melakukan matematika yang sangat dalam.
*   Isinya adalah dua lapisan garis-garis (Linear Layers).
*   Setiap kata diproses secara **mandiri**. Artinya, kata ke-1 tidak boleh "menyontek" hasil hitungan kata ke-2 di dalam kotak ini. Kenapa? Supaya robot tidak bisa curang dengan melihat jawaban di masa depan saat dia sedang latihan.

#### B. Perhatian Silang (Cross-Attention)
Lihat di sisi kanan (Decoder), ada panah yang datang dari sisi kiri (Encoder). Inilah yang disebut **Cross-Attention**.
*   Ingat Q (Pertanyaan), K (Kunci), dan V (Nilai) dari Bab 5?
*   Di sini, **Pertanyaan (Q)** datang dari apa yang sedang ditulis si Pembicara (Decoder).
*   Tapi **Kunci (K)** dan **Nilai (V)** diambil dari catatan si Pendengar (Encoder).
*   Ini seperti kamu sedang menulis tugas (Decoder), tapi matamu terus melihat ke buku pelajaran (Encoder) untuk mencari jawaban yang benar.

#### C. Tumpukan Lantai (Nx)
Kamu melihat simbol **Nx** di samping kotak besar? Itu artinya "Lantai ini diulang sebanyak N kali". 
*   Biasanya ilmuwan menumpuk 6, 12, bahkan ratusan lantai yang sama ke atas. 
*   Semakin banyak tumpukannya, robot akan semakin pintar karena dia punya banyak lapisan untuk berpikir.

**Catatan Penting:** Robot menjalankan si Pendengar (Encoder) sampai selesai satu kali saja untuk membuat "Catatan Rahasia". Setelah catatan itu jadi, barulah catatan itu diberikan ke **setiap lantai** di sisi Pembicara (Decoder).

#### D. Jembatan Perapi (Add & Norm)
Setiap kali selesai melakukan satu proses (seperti Attention atau Feed Forward), robot selalu melewati **Add & Norm**. Ini adalah gabungan dari "Jembatan Pengingat" (Bab 7) dan "Stasiun Perapi Angka" (Bab 8). Tujuannya agar angka-angka robot tetap rapi dan tidak ada informasi yang hilang saat naik ke lantai berikutnya.

---

## 4. Kamu Sudah Berhasil!

Coba bayangkan: kita memulai buku ini dengan belajar **tambah-tambahan (+) dan kali-kalian (x)** yang sangat sederhana. Dan sekarang, kamu sudah tahu apa arti setiap kotak, setiap garis, dan setiap kata di dalam arsitektur robot paling canggih di dunia!

Secara teori, dengan semua pengetahuan yang kamu punya dari Bab 1 sampai Bab 12 ini, kamu sudah bisa mulai membangun robot pintarmu sendiri dari nol. Kamu tahu:
*   Bagaimana angka masuk dan keluar.
*   Bagaimana robot belajar dari kesalahannya.
*   Bagaimana robot memberikan perhatian pada kata-kata penting.
*   Bagaimana semua bagian itu disusun menjadi sebuah mahakarya bernama **Transformer**.

---

### Ringkasan Bab 12:
*   **Transformer** membagi tugas menjadi dua bagian: **Encoder** (memahami bahasa asal) dan **Decoder** (menulis bahasa tujuan).
*   **Cross-Attention** adalah rahasia bagaimana si Pembicara bisa selalu ingat apa yang dikatakan si Pendengar.
*   **Nx** berarti kita menumpuk banyak lapisan otak agar robot makin jenius.
*   Semua teknologi hebat ini, pada dasarnya, hanyalah jutaan operasi **tambah dan kali** yang diatur dengan sangat cerdik.

**Selamat! Kamu sekarang sudah resmi menjadi "Pakar Robot Cilik". Teruslah belajar, karena masa depan dunia AI ada di tangan anak-anak hebat seperti kamu!**