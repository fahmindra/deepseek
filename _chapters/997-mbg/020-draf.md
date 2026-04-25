---
slug: mbg-2
title: mbg-2
---

# Bab 2: Mengajari Robot Cara Berbicara

Di bab sebelumnya, kita sudah tahu kalau robot bisa membedakan bunga dan daun menggunakan angka. Sekarang, pertanyaannya: **Bagaimana robot bisa menulis cerita, menjawab pertanyaan, atau bahkan membuat puisi?**

Apakah robot mengerti kata-kata? Jawabannya: **Tidak.** Robot tetap hanya bermain dengan angka. Mari kita lihat rahasia di baliknya!

## 1. Rahasia Kode Huruf
Karena robot hanya suka angka, kita harus mengubah huruf menjadi angka. Bayangkan kita punya "Kode Rahasia":
*   A = 1
*   B = 2
*   C = 3
*   ...dan seterusnya sampai Z = 26.
*   Spasi = 27
*   Titik (.) = 28

Sekarang, jika kita ingin menulis kata **"ADA"**, kita memberitahu robot angka: **1, 4, 1**. Mudah, bukan?

## 2. Permainan Tebak Huruf Selanjutnya
Bagaimana AI menulis kalimat? Ternyata, AI bekerja seperti permainan **"Tebak Kelanjutannya"**.

Bayangkan kamu sedang menulis kalimat: **"Cicak-cicak di dindin..."**
Kamu pasti otomatis ingin menjawab huruf: **"g"**.

Nah, robot melakukan hal yang sama! Kita memberi robot beberapa huruf, lalu tugas robot adalah menebak **satu huruf berikutnya**.

### Contoh: Humpty Dumpty
Jika kita memasukkan kata: `H u m p t y   D u m p t`
Robot akan menghitung menggunakan matematika di dalam otaknya, lalu ia akan menekan tombol huruf yang menurutnya paling cocok, yaitu: **"y"**.

Sekarang kita punya kata lengkap: **"Humpty Dumpty"**.

## 3. Menjadi Pabrik Cerita (Generative AI)
Setelah robot berhasil menebak huruf **"y"**, apa yang terjadi selanjutnya?
1. Robot mengambil huruf **"y"** hasil tebakannya tadi.
2. Huruf itu ditempelkan di belakang kalimat sebelumnya.
3. Sekarang robot punya kalimat baru: `Humpty Dumpty `.
4. Robot menebak lagi huruf berikutnya... mungkin sebuah **spasi**, lalu huruf **"s"**, lalu **"a"**, lalu **"t"**.

Lama-kelamaan, dari satu huruf ke satu huruf, robot bisa menulis: **"Humpty Dumpty sat on a wall"** (Humpty Dumpty duduk di atas dinding). 

Inilah yang kita sebut sebagai **Generative AI**—AI yang bisa menciptakan sesuatu!

## 4. "Kotak Ingatan" yang Terbatas (Context Length)
Robot punya satu masalah: **Ingatannya tidak selamanya.** 

Bayangkan robot punya sebuah jendela kecil yang hanya bisa melihat 12 huruf saja. Jendela ini disebut **Context Length** (Panjang Konteks). 

Jika robot sedang menebak huruf ke-13, ia harus membuang huruf ke-1 agar muat di jendelanya. 
*   **Awalnya:** `H u m p t y   D u m p t` (12 huruf) -> Robot tebak **"y"**.
*   **Selanjutnya:** Huruf `H` dibuang, lalu `u m p t y   D u m p t y` (12 huruf) masuk ke jendela -> Robot tebak **" " (spasi)**.

Robot zaman sekarang punya "jendela" yang sangat besar. Mereka bisa mengingat ribuan kata sekaligus, jadi mereka tidak mudah lupa apa yang sedang mereka bicarakan.

## 5. Kenapa Tombol Keluar dan Tombol Masuk Berbeda?
Mungkin kamu bertanya: *"Kenapa saat memasukkan huruf kita pakai 1 angka (A=1), tapi saat robot menjawab, dia punya 26 tombol (A sampai Z)?"*

Bayangkan seperti ini:
*   **Saat Masuk:** Kita memberikan robot sebuah kunci kecil yang ada nomornya.
*   **Saat Keluar:** Robot berdiri di depan sebuah papan besar yang punya 26 lampu. Lampu yang menyala paling terang adalah jawaban robot.

Kenapa begitu? Karena dengan memiliki 26 lampu, robot bisa menunjukkan pilihan-pilihannya. Misalnya, setelah kata "Makan ", lampu huruf **"N"** (untuk Nasi) dan lampu huruf **"R"** (untuk Roti) mungkin sama-sama menyala redup, tapi lampu **"N"** sedikit lebih terang. Ini membantu robot memilih jawaban yang paling masuk akal!

---

### Kamus Robot Bab ini:
*   **Predict (Tebak):** Tugas utama robot untuk mencari huruf atau kata berikutnya.
*   **Recursive (Berulang):** Cara robot menulis cerita dengan menempelkan hasil tebakannya terus-menerus.
*   **Context Length (Panjang Konteks):** Seberapa banyak kata yang bisa diingat robot dalam satu waktu.

---
**Di bab berikutnya, kita akan belajar bahwa ternyata angka 1, 2, 3 untuk huruf itu terlalu sederhana bagi robot. Robot punya cara yang lebih hebat lagi untuk mengenali kata! Penasaran? Yuk, lanjut!**