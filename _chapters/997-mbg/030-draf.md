---
slug: mbg-3
title: mbg-3 
---

# Bab 3: Rahasia Besar—Mengapa Robot Sekarang Sangat Pintar?

Menulis kalimat "Humpty Dumpty sat on a wall" huruf demi huruf adalah permulaan yang bagus. Tapi, robot pintar zaman sekarang (seperti ChatGPT atau Llama) bisa melakukan hal yang jauh lebih hebat dari itu! Mereka bisa menulis esai, membuat kode komputer, bahkan bercanda.

Apa yang membedakan robot sederhana yang kita buat di Bab 2 dengan robot super pintar masa kini? Ada beberapa penemuan hebat yang menjadi "bumbu rahasia" mereka. Mari kita pelajari bumbu rahasia yang pertama: **Embeddings**.

---

## 1. Masalah dengan Angka Sembarangan

Di bab sebelumnya, kita memberikan angka sembarangan untuk huruf (A=1, B=2, dan seterusnya). Tapi, apakah benar angka "1" adalah cara terbaik untuk menggambarkan huruf "A"? Belum tentu.

Bagaimana jika ada angka lain yang lebih bagus? Angka yang bisa membuat robot belajar lebih cepat dan lebih sedikit melakukan kesalahan?

---

## 2. Trik Jenius: Membiarkan Robot Memilih Angkanya Sendiri

Para ilmuwan punya ide yang sangat cerdik. Ingat bagaimana robot belajar dengan cara mengubah-ubah angka di garis (Bobot) sampai jawabannya benar? Ilmuwan berpikir: **"Kenapa kita tidak biarkan robot mengubah angka hurufnya juga?"**

Begini cara kerjanya:
1. Kita mulai dengan memberikan angka asal untuk huruf (misal A=1, B=2).
2. Robot mencoba menebak huruf berikutnya.
3. Kita lihat seberapa besar kesalahannya (**Loss**).
4. Selain memperbaiki angka di garis penghubung (Bobot), robot juga **memperbaiki angka untuk huruf itu sendiri**.

Misalnya, robot merasa angka "1" untuk huruf "A" kurang cocok. Setelah mencoba berkali-kali, robot mengubahnya menjadi "1,2". Ternyata, dengan angka "1,2", si robot lebih jarang salah! 

Proses ini disebut **Embedding**. Jadi, *Embedding* adalah angka-angka khusus untuk mewakili sesuatu (huruf atau kata) yang ditemukan robot sendiri melalui latihan terus-menerus.

---

## 3. Kenapa Satu Angka Saja Tidak Cukup?

Dulu kita memberikan satu angka untuk satu huruf. Tapi, bahasa manusia itu sangat kaya dan rumit. Satu angka saja tidak cukup untuk menggambarkan sebuah huruf atau kata.

Ingat contoh daun dan bunga di Bab 1? Kita menggunakan **4 angka** (Merah, Hijau, Biru, dan Ukuran) untuk mengenali satu benda. Semakin banyak angka yang kita gunakan untuk menggambarkan sesuatu, robot akan semakin paham.

Jika untuk mengenali daun saja butuh 4 angka, maka untuk mengenali bahasa manusia yang sangat luas, kita butuh banyak angka untuk setiap hurufnya!

---

## 4. Mengenal "Vektor" (Daftar Angka Berurutan)

Karena satu angka tidak cukup, kita memberikan "sekelompok angka" untuk setiap huruf. Kelompok angka yang berurutan ini disebut **Vektor**.

Bayangkan Vektor seperti sebuah alamat atau kartu identitas. 
*   Huruf **"h"** mungkin punya Vektor: `[0.12, -0.5, 0.88, 0.23, ...]`
*   Isinya bisa 10 angka, 100 angka, bahkan lebih!

**Penting:** Urutan angka dalam Vektor tidak boleh tertukar. Jika kamu menukar angka pertama dengan angka kedua, maka maknanya akan berubah total. Ini sama seperti warna RGB; jika angka Merah dan Hijau tertukar, warnanya jadi berbeda, bukan?

---

## 5. Bagaimana Memasukkan Vektor ke Otak Robot?

Mungkin kamu bingung: *"Jika satu huruf punya 10 angka, bagaimana cara memasukkannya ke dalam robot?"*

Caranya sangat mudah. Kita hanya perlu memperluas "pintu masuk" (Input Layer) si robot.
*   Jika kita ingin memasukkan 12 huruf (seperti "humpty dumpt")...
*   Dan setiap huruf diwakili oleh 10 angka (Vektor panjangnya 10)...
*   Maka di pintu masuk robot, kita sediakan **120 lingkaran (Neuron)**.

Tinggal kita jejerkan saja semua angkanya, dan robot siap menghitung!

---

## 6. Gudang Angka (Embedding Matrix)

Agar robot tidak bingung, semua daftar angka (Vektor) untuk setiap huruf disimpan dalam sebuah tabel besar yang disebut **Embedding Matrix**.

Bayangkan ini seperti **Gudang Rahasia Angka**:
*   Ada kolom untuk huruf "a".
*   Ada kolom untuk huruf "b".
*   Dan seterusnya.

Setiap kali robot melihat huruf "a", ia akan pergi ke gudang ini, melihat kolom "a", dan mengambil daftar angka (Vektor) yang ada di sana untuk mulai menghitung.

**Satu Aturan Penting:** Semua Vektor untuk setiap huruf harus punya **panjang yang sama**. Jika huruf "a" punya 10 angka, maka huruf "z" dan tanda "spasi" juga harus punya 10 angka. Kenapa? Supaya pintu masuk robot yang sudah kita buat (misalnya 120 lingkaran tadi) selalu pas dan tidak kekecilan atau kebesaran.

---

## Kesimpulan Bab 3:
*   **Embeddings** adalah cara robot menciptakan angka-angkanya sendiri agar ia bisa belajar dengan lebih baik.
*   **Vektor** adalah sekelompok angka yang mewakili satu huruf atau kata.
*   **Embedding Matrix** adalah tabel tempat robot menyimpan semua daftar angka tersebut.

Dengan menggunakan banyak angka (Vektor) untuk setiap kata, robot jadi punya "perasaan" yang lebih dalam terhadap bahasa. Inilah langkah pertama yang membuat robot menjadi sangat pintar seperti sekarang!

**Di bab selanjutnya, kita akan melihat bagaimana robot memotong-motong kata agar lebih mudah dimengerti. Siap?**