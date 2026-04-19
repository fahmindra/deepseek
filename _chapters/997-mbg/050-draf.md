---
slug: mbg-5
title: mbg-5
---

# Bab 5: Ilmu "Memberi Perhatian" (Self-Attention)

Pernahkah kamu berada di dalam kelas yang sangat ramai, tapi kamu tetap bisa mendengar suara gurumu dengan jelas? Itu karena otakmu melakukan hal ajaib: **Memberikan Perhatian**. Kamu memilih untuk mendengarkan guru dan mengabaikan suara teman yang berisik.

Nah, robot juga perlu melakukan hal yang sama agar bisa memahami kalimat manusia yang panjang. Teknologi ini disebut **Self-Attention** (Perhatian pada Diri Sendiri). Mari kita pelajari bagaimana robot melakukannya!

---

## 1. Kenapa Robot Butuh "Perhatian"?

Sejauh ini, kita belajar bahwa robot menghubungkan angka dari satu lapisan ke lapisan berikutnya seperti barisan tentara yang kaku. Masalahnya, dalam bahasa manusia, kata yang penting bisa berada di mana saja.

Coba perhatikan kalimat misteri ini:
> "Damian memiliki seorang anak rahasia, seorang **anak perempuan**, dan di dalam surat wasiatnya ia menulis bahwa semua harta bendanya akan diberikan kepada ____."

Kata apa yang cocok untuk mengisi titik-titik itu? Jawabannya adalah **"dia"** (atau dalam bahasa Inggris *"her"*). 

Bagaimana kita tahu? Karena ada kata **"anak perempuan"** di bagian awal kalimat. Meskipun kata itu sudah lewat jauh di depan, robot harus bisa "menengok ke belakang" dan memberikan perhatian khusus pada kata tersebut agar tidak salah menebak (misalnya menebak "dia laki-laki").

---

## 2. Masalah dengan Sistem "Kabel Kaku"

Sistem robot yang lama punya berat garis (Bobot) yang tetap berdasarkan **posisi**. Misalnya, ia selalu menganggap kata ke-1 itu penting. 

Tapi, bagaimana jika kata "anak perempuan" tadi pindah posisi? 
*   "Anak rahasianya adalah seorang **anak perempuan**..."
*   "Seorang **anak perempuan** adalah anak rahasianya..."

Robot butuh cara agar ia bisa fokus pada **makna** kata itu, di mana pun kata itu berada, bukan hanya karena ia ada di urutan tertentu.

---

## 3. Cara Kerja Self-Attention: Menggunakan "Tingkat Kepentingan"

Bayangkan setiap kata dalam kalimat memiliki nilai kepentingan. Robot akan menghitung hasil akhir dengan cara menjumlahkan semua kata, tapi setiap kata dikalikan dulu dengan "angka kepentingan".

Misalnya untuk kalimat "Humpty Dumpty sat":
*   Hasil = (Kepentingan 1 $\times$ Humpty) + (Kepentingan 2 $\times$ Dumpty) + (Kepentingan 3 $\times$ sat)

Pertanyaannya: **Bagaimana robot menentukan siapa yang paling penting?**

---

## 4. Tiga Peran Ajaib: Kunci, Pertanyaan, dan Nilai

Untuk menentukan siapa yang penting, setiap kata di dalam otak robot diberi tiga tugas atau peran (yang masing-masing adalah daftar angka/vektor):

1.  **Pertanyaan (Query - Q):** "Apa yang sedang aku cari?" (Biasanya ini adalah kata terakhir yang sedang diproses).
2.  **Kunci (Key - K):** "Apa yang aku miliki?" (Setiap kata menawarkan informasi yang ia punya).
3.  **Nilai (Value - V):** "Pesan apa yang akan aku berikan jika aku terpilih?"

### Proses Mencocokkan:
Robot akan mengambil **Pertanyaan (Q)** dari kata terakhir dan mencocokkannya dengan **Kunci (K)** dari semua kata yang sudah muncul. 

*   Jika **Pertanyaan** dan **Kunci** sangat cocok (seperti kunci pas masuk ke lubangnya), maka robot akan memberikan nilai kepentingan yang tinggi.
*   Jika tidak cocok, kepentingannya kecil.

Setelah tahu siapa yang penting, robot akan mengambil **Nilai (V)** dari kata-kata tersebut dan menjumlahkannya. 

Agar total kepentingannya tidak berlebihan, robot menggunakan fungsi **Softmax** (yang sudah kita pelajari) supaya jika semua angka kepentingan dijumlahkan, hasilnya pas 1 (atau 100%).

---

## 5. Blok Perhatian (Self-Attention Block)

Semua proses rumit di atas (Q, K, dan V) dibungkus dalam sebuah kotak yang disebut **Self-Attention Block**. 

Kotak ini punya tiga alat pengali rahasia:
*   **$W_q$**: Alat untuk membuat "Pertanyaan".
*   **$W_k$**: Alat untuk membuat "Kunci".
*   **$W_v$**: Alat untuk membuat "Nilai".

Robot hanya perlu memasukkan daftar angka kata-kata (Embeddings), dan kotak ini akan mengeluarkan satu daftar angka baru yang sudah mengandung "perhatian" terhadap kata-kata penting.

---

## 6. Perhatian Silang (Cross-Attention)

Ada juga jenis perhatian lain bernama **Cross-Attention**. Bedanya, "Pertanyaan" datang dari satu kalimat (misalnya bahasa Indonesia), tapi "Kunci" dan "Nilai" datang dari kalimat lain (misalnya bahasa Inggris). Ini sangat berguna untuk robot yang tugasnya menerjemahkan bahasa!

---

## 7. Masalah Urutan: Siapa yang Di Depan?

Ada satu hal lucu tentang Self-Attention: ia terlalu fokus pada makna sampai-sampai ia **lupa urutan**. 

Bagi Self-Attention, kalimat "Kucing makan ikan" dan "Ikan makan kucing" terlihat hampir sama karena kata-katanya sama. Padahal, kita tahu urutan itu sangat penting!

Oleh karena itu, sebelum masuk ke kotak perhatian ini, robot biasanya diberi "bumbu tambahan" yang disebut **Positional Encoding** (Kode Posisi) agar robot tahu kata mana yang ada di depan dan mana yang di belakang. Kita akan pelajari ini di bab selanjutnya.

---

### Ringkasan Bab 5:
*   **Self-Attention** adalah cara robot fokus pada kata yang paling penting dalam sebuah kalimat.
*   Robot menggunakan **Pertanyaan (Query)**, **Kunci (Key)**, dan **Nilai (Value)** untuk menghitung tingkat kepentingan.
*   Ini membantu robot memahami hubungan antar kata meskipun letaknya berjauhan.
*   Sistem ini dibungkus dalam sebuah **Block** yang bisa digunakan berulang kali.

**Sekarang robot sudah bisa "fokus". Selanjutnya, kita akan melihat bumbu-bumbu kecil lainnya yang membuat otak robot tetap stabil dan tidak cepat pusing!**