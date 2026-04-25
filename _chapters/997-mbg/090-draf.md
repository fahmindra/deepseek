---
slug: mbg-9
title: mbg-9 
---
# Bab 9: Banyak Mata untuk Memahami (Multi-head Attention)

Kita sudah belajar tentang **Self-Attention** di Bab 5, yaitu cara robot memberikan perhatian pada kata-kata yang penting. Namun, ilmuwan menemukan bahwa satu "mata" saja tidak cukup untuk memahami kalimat yang rumit. 

Oleh karena itu, lahirlah **Multi-head Attention** (Perhatian Banyak Kepala). Ini adalah bagian paling penting dalam otak robot canggih saat ini. Mari kita bedah bagaimana cara kerjanya!

---

## 1. Satu Kepala vs. Banyak Kepala

Bayangkan kamu sedang membaca buku cerita. 
*   Mata pertama mungkin fokus pada **siapa** tokohnya.
*   Mata kedua fokus pada **kapan** kejadiannya.
*   Mata ketiga fokus pada **perasaan** tokohnya.

Jika kamu hanya punya satu mata, mungkin kamu hanya bisa fokus pada satu hal dalam satu waktu. Tapi robot canggih punya banyak "kepala" perhatian yang bekerja secara **paralel** (bersamaan).

---

## 2. Cara Kerja Multi-head Attention

Ide utamanya sangat sederhana: robot menjalankan beberapa proses *Attention* sekaligus pada saat yang sama menggunakan data yang sama.

Perhatikan gambar di bawah ini:

![Multi-head attention](https://towardsdatascience.com/wp-content/uploads/2024/10/1BmZd8SIDEQu7r5_R7y54kQ.png)
*Gambar: Bagaimana banyak kepala bekerja bersama (Karya Rohit Patel)*

### Langkah-langkahnya:
1.  **Input yang Sama:** Semua kepala perhatian mengambil angka yang sama dari kata-kata yang masuk.
2.  **Kaca Mata Berbeda (Linear Layers):** Sebelum mulai menghitung, setiap kepala punya "kaca mata" khusus. Di gambar, ada panah dari $v1$ ke $v1h1$. Di setiap panah itu sebenarnya ada **Linear Layer** (sebuah tabel angka pengali). Tugasnya adalah mengubah angka asli menjadi angka baru yang khusus untuk kepala tersebut.
3.  **Proses Berbarengan:** Setiap kepala menghitung *Query* (Q), *Key* (K), dan *Value* (V) versinya sendiri-sendiri.
4.  **Menggabungkan Hasil (Concatenation):** Setelah semua kepala selesai menghitung, hasilnya tidak dibuang, melainkan **ditempelkan menjadi satu**.
    *   Misal: Kepala 1 menghasilkan angka `[1, 2]`.
    *   Kepala 2 menghasilkan angka `[3, 4]`.
    *   Maka hasil akhirnya adalah `[1, 2, 3, 4]`.

Ini seperti sebuah tim ahli yang memberikan laporan pendek masing-masing, lalu laporan-laporan itu dijilid menjadi satu buku besar.

---

## 3. Rahasia di Balik Panah (Linear Transformation)

Di dalam diagram, kamu melihat panah-panah kecil. Kamu harus tahu bahwa setiap panah itu adalah sebuah proses matematika. 

Robot mengambil nilai K, Q, dan V yang sudah dibuat, lalu memberikan "sentuhan tambahan" (transformasi linear) untuk setiap kepala. Inilah yang membuat setiap kepala bisa melihat hal yang berbeda meskipun mereka membaca kalimat yang sama. 

---

## 4. Pertanyaan Misterius bagi Para Ilmuwan

Ada satu hal yang menarik dan sedikit mengejutkan bagi para ilmuwan (termasuk penulis aslinya!). 

Kenapa kita harus menggunakan satu mesin pembuat K, Q, dan V yang sama lalu memberinya lapisan tambahan untuk setiap kepala? Kenapa tidak kita buatkan saja mesin pembuat K, Q, dan V yang benar-benar berbeda untuk setiap kepala dari awal?

Sampai sekarang, cara yang kita pelajari di bab ini adalah cara yang paling banyak digunakan karena terbukti sangat manjur, meskipun para ilmuwan masih terus bertanya-tanya dan bereksperimen untuk menemukan cara yang lebih baik lagi. Ini menunjukkan bahwa dunia AI masih penuh dengan misteri yang bisa kamu pecahkan suatu hari nanti!

---

## 5. Kenapa Ini Sangat Hebat?

Dengan memiliki banyak "kepala" (biasanya robot punya 8, 12, atau bahkan lebih), robot bisa memahami hubungan antar kata dengan sangat dalam:
*   Kepala 1 memahami hubungan subjek dan kata kerja.
*   Kepala 2 memahami hubungan kata ganti (seperti "dia") dengan namanya.
*   Kepala 3 memahami suasana kalimat (sedih atau senang).

Karena semua kepala ini bekerja bersamaan, robot jadi sangat cepat dan sangat cerdas dalam memahami maksud manusia.

---

### Ringkasan Bab 9:
*   **Multi-head Attention** adalah menjalankan banyak proses perhatian secara bersamaan.
*   Setiap kepala menggunakan **Linear Layer** (kaca mata khusus) untuk melihat informasi yang berbeda.
*   **Concatenate** adalah cara robot menempelkan hasil dari semua kepala menjadi satu daftar angka yang panjang.
*   Teknologi inilah yang menjadi "jantung" utama dari model AI yang kita kenal sekarang (seperti Transformer).

**Sekarang robot kita punya banyak mata yang sangat teliti! Di bab selanjutnya, kita akan belajar bagaimana robot mengingat posisi kata agar tidak tertukar-tukar!**