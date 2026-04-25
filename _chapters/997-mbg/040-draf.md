---
slug: mbg-4
title: mbg-4
---
# Bab 4: Rahasia Potongan Puzzle (Tokenizers)

Di bab sebelumnya, kita belajar bahwa robot mengubah huruf menjadi daftar angka yang disebut **Vektor**. Tapi, ada satu masalah: kalau robot belajar huruf demi huruf (seperti h-u-k-u), ia butuh waktu sangat lama untuk mengerti arti sebuah kata. 

Nah, di bab ini kita akan belajar cara yang lebih pintar, yaitu memotong-motong kata!

---

## 1. Apa itu "Token"?

Dalam dunia robot pintar, ada satu kata keren yang harus kamu tahu: **Token**. 

**Token** adalah potongan terkecil dari bahasa yang dimengerti oleh robot. 
*   Dulu, kita menggunakan **huruf** sebagai token (H, u, k, u).
*   Sekarang, kita bisa menggunakan **kata** sebagai token (Buku).
*   Atau, kita bisa menggunakan **potongan kata** (Bu-ku).

Bayangkan Token seperti balok Lego. Kamu bisa membangun istana (kalimat) dari balok-balok kecil (huruf) atau dari balok yang lebih besar (kata).

---

## 2. Kenapa Tidak Memakai "Kata" Saja?

Mungkin kamu berpikir: *"Kenapa tidak satu kata utuh saja yang dijadikan satu angka? Kan lebih gampang!"*

Memang kelihatannya gampang, tapi ada dua masalah besar:

1.  **Tombol yang Terlalu Banyak:** Ada lebih dari 180.000 kata dalam bahasa Inggris (dan banyak sekali dalam bahasa Indonesia). Jika kita membuat satu tombol untuk setiap kata, robot kita akan punya **180.000 tombol** di pintu keluarnya! Wah, itu keyboard yang sangat raksasa dan membuat robot pusing.
2.  **Robot Jadi Bingung dengan Kata Mirip:** Bayangkan ada kata **"Kucing"** dan **"Kucing-kucing"**. Jika kita menganggap keduanya adalah hal yang benar-benar berbeda, robot tidak akan tahu kalau keduanya sebenarnya sama-sama hewan berbulu yang mengeong. Robot harus belajar dari nol lagi untuk setiap kata baru.

---

## 3. Rahasia Subword: Cara Tengah yang Jenius

Para ilmuwan menemukan jalan tengah yang sangat pintar bernama **Subword Tokenization**. Caranya adalah memotong kata yang panjang menjadi potongan-potongan kecil yang bermakna.

**Contohnya:**
Kata **"Cats"** (Kucing-kucing dalam bahasa Inggris) dipotong menjadi dua token: 
1.  **"Cat"** 
2.  **"s"** (tanda kalau jumlahnya banyak)

Dengan cara ini, robot jadi lebih pintar karena:
*   Ia cukup belajar arti "Cat". 
*   Ia belajar bahwa kalau ada tambahan "s" di belakang, artinya bendanya jadi banyak.
*   Jumlah "tombol" atau kosakata (Vocabulary) yang harus diingat robot jadi jauh lebih sedikit (hanya puluhan ribu, bukan ratusan ribu).

---

## 4. Mengenal Si "Tukang Potong" (Tokenizer)

Robot punya asisten khusus bernama **Tokenizer**. Tugasnya adalah mengubah kalimat manusia menjadi daftar angka yang siap dimasukkan ke dalam mesin.

Mari kita lihat tugas si Tokenizer saat melihat kalimat **"Humpty Dumpt"**:

1.  **Memotong (Splitting):** Tokenizer memotong-motong kalimat itu menjadi token. (Misalnya: `h`, `u`, `m`, `p`, `t`, `y`, ` `, `D`, `u`, `m`, `p`, `t`).
2.  **Memberi Nomor ID:** Tokenizer melihat "Kamus Rahasia" miliknya. 
    *   Huruf `h` mungkin ada di rak nomor 8.
    *   Huruf `u` ada di rak nomor 21.
    *   Dan seterusnya.
3.  **Mengambil Vektor:** Setelah tahu nomor raknya (ID), robot pergi ke **Embedding Matrix** (Gudang Angka yang kita pelajari di Bab 3) untuk mengambil daftar angka (Vektor) yang ada di rak tersebut.

**Ingat:** Nomor ID (seperti 8 atau 21) hanyalah nomor rak. Robot tidak menghitung angka 8 itu, tapi ia mengambil **isi** di dalam rak nomor 8 tersebut (yaitu Vektor berisi banyak angka) untuk mulai bekerja.

---

## 5. Bentuk Baru Otak Robot Kita

Sekarang, otak robot kita sudah lebih canggih! Bentuknya kira-kira seperti ini:
*   **Input:** Kalimat masuk.
*   **Tokenizer:** Memotong kalimat jadi Token.
*   **Embedding Matrix:** Mengubah Token jadi daftar angka (Vektor).
*   **Neural Network:** Menghitung dengan kali dan tambah yang sangat banyak.
*   **Output:** Menebak Token berikutnya!

---

## 6. Persiapan Menjadi Ilmuwan Hebat

Di bab-bab selanjutnya, kita akan mempelajari teknologi yang lebih ajaib lagi—yang membuat robot bisa memahami perasaan dan maksud kalimatmu. Tapi, sebelum lanjut, kamu perlu tahu bahwa ada beberapa "Alat Ajaib" matematika yang digunakan para ilmuwan. Jangan takut, ini hanya nama-namanya saja:

1.  **Matriks:** Tabel berisi angka-angka (seperti kotak cokelat).
2.  **Fungsi:** Seperti mesin pembuat jus (masukkan buah, keluar jus).
3.  **Pangkat:** Mengalikan angka dengan dirinya sendiri berkali-kali (seperti $2 \times 2 \times 2$).
4.  **Statistik (Mean & Varians):** Cara menghitung rata-rata dan melihat seberapa berantakan angka-angka kita.

(Penjelasan alat-alat ini ada di bagian lampiran buku ini kalau kamu ingin mengintipnya!)

---

### Kesimpulan Bab 4:
*   **Token** adalah potongan bahasa (bisa huruf, kata, atau potongan kata).
*   **Subword Tokenizer** memotong kata agar robot tidak perlu punya terlalu banyak tombol dan bisa memahami kata-kata yang mirip.
*   **Tokenizer** adalah asisten yang mengubah kata menjadi nomor rak (ID) agar robot bisa mengambil daftar angka (Vektor) yang tepat.

**Sudah siap melihat bagaimana robot bisa memberikan "Perhatian" pada kata-kata yang penting? Yuk, kita lanjut ke bab tentang "Self-Attention"!**