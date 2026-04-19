---
slug: mbg-10
title: mbg-10 
---
# Bab 10: Label Nomor Antrean (Positional Embedding)

Di bab-bab sebelumnya, kita sudah membuat robot yang sangat hebat. Robot kita bisa mengubah kata menjadi angka, punya banyak mata untuk fokus (Multi-head Attention), dan bisa merapikan angkanya sendiri. 

Tapi, ada satu masalah lucu yang dimiliki robot kita: **Dia tidak tahu urutan kata!**

Bagi robot yang kita buat sejauh ini, kalimat **"Kucing mengejar tikus"** dan **"Tikus mengejar kucing"** terlihat sama persis. Kenapa? Karena isinya sama-sama ada kata kucing, mengejar, dan tikus. Robot tidak tahu kalau urutan itu sangat penting. Mari kita beri robot kita "label nomor antrean" agar dia tidak bingung lagi!

---

## 1. Kenapa Urutan itu Penting?

Bayangkan kamu sedang mengantre es krim. 
*   Jika kamu di urutan **ke-1**, kamu dapat es krim duluan.
*   Jika kamu di urutan **ke-10**, kamu harus menunggu lama.

Dalam bahasa, urutan juga menentukan arti. Robot butuh cara untuk membedakan mana kata yang muncul di awal, di tengah, dan di akhir. Jika tidak, robot akan membaca sebuah buku seperti tumpukan kata yang dikocok di dalam gelas.

---

## 2. Mengenal Embedding Posisi (Positional Embedding)

Di Bab 3, kita belajar tentang **Word Embedding**, yaitu daftar angka (Vektor) yang mewakili arti sebuah kata. Nah, sekarang kita akan membuat hal yang serupa untuk posisi, yang disebut **Positional Embedding**.

Caranya sangat sederhana:
1.  Kita buat "Kamus Rahasia" khusus untuk angka posisi.
2.  Posisi **ke-1** punya daftar angka (Vektor) sendiri.
3.  Posisi **ke-2** punya daftar angka sendiri.
4.  Posisi **ke-3** punya daftar angka sendiri, dan seterusnya.

Jadi, robot tidak hanya melihat "Apa kata ini?", tapi juga "Di urutan ke berapa kata ini muncul?".

---

## 3. Tabel Angka Posisi (Positional Matrix)

Sama seperti Word Embedding, daftar angka untuk posisi ini disimpan dalam sebuah tabel besar (Matriks). 

*   **Kolom 1:** Berisi daftar angka untuk posisi pertama.
*   **Kolom 2:** Berisi daftar angka untuk posisi kedua.
*   **Panjang Vektor:** Daftar angka untuk posisi ini harus **sama panjangnya** dengan daftar angka untuk kata. 

Kenapa panjangnya harus sama? Karena nanti robot akan **menjumlahkan** keduanya!

---

## 4. Cara Robot Menggabungkan Kata dan Posisi

Bayangkan kata **"Kucing"** memiliki kode angka: `[1, 2, 3]`.
Lalu, karena "Kucing" ada di urutan **pertama**, ia mengambil kode posisi ke-1: `[0.1, 0.1, 0.1]`.

Robot akan menggabungkannya seperti ini:
`[1, 2, 3]` + `[0.1, 0.1, 0.1]` = **`[1.1, 2.1, 3.1]`**

Sekarang, angka `1.1, 2.1, 3.1` ini mengandung dua informasi sekaligus:
1.  Informasi bahwa ini adalah seekor **Kucing**.
2.  Informasi bahwa kucing ini berada di **Urutan Pertama**.

Dengan cara ini, robot bisa membedakan "Kucing" yang ada di depan kalimat dengan "Kucing" yang ada di belakang kalimat.

---

## 5. Embedding vs. Enkoding (Apa Bedanya?)

Mungkin kamu akan mendengar dua istilah yang mirip: **Positional Embedding** dan **Positional Encoding**. 

*   **Positional Encoding:** Ini adalah cara lama. Ilmuwan menggunakan rumus matematika yang rumit (seperti gelombang naik turun) untuk menentukan angka posisinya. (Penjelasan lengkapnya ada di bagian lampiran buku ini).
*   **Positional Embedding:** Ini adalah cara yang paling banyak digunakan robot pintar zaman sekarang. Alih-alih pakai rumus matematika yang kaku, kita membiarkan robot **belajar sendiri** angka-angka apa yang paling cocok untuk setiap posisi saat ia berlatih. 

Karena robot zaman sekarang sangat kuat, mereka lebih suka belajar sendiri angkanya daripada diberi rumus yang sudah jadi.

---

## 6. Kesimpulan Bab 10

Tanpa nomor antrean, robot kita akan buta terhadap urutan. Dengan memberikan **Positional Embedding**, setiap kata sekarang punya "stiker" yang memberi tahu robot di mana posisi kata itu berada.

Sekarang robot kita sudah punya:
*   Arti kata (Word Embedding).
*   Urutan kata (Positional Embedding).
*   Perhatian yang tajam (Multi-head Attention).
*   Kerapian (Layer Normalization).

---

### Ringkasan Bab 10:
*   **Positional Embedding** adalah daftar angka yang mewakili urutan kata (1, 2, 3, dst).
*   Setiap posisi memiliki **Vektor** (daftar angka) yang unik.
*   Vektor posisi dijumlahkan dengan vektor kata agar robot tahu arti dan urutan sekaligus.
*   Robot zaman sekarang lebih suka mempelajari angka posisinya sendiri daripada menggunakan rumus matematika kaku.

**Wah, robot kita sudah hampir jadi! Di bab selanjutnya, kita akan melihat bagaimana semua bagian ini digabungkan menjadi satu "Rumah Robot" yang utuh!**