---
slug: tpa-10-t
title: Peluang & Persamaan & Pertidaksamaan
---
# Pembahasan Lengkap Materi Peluang, Persamaan, dan Pertidaksamaan

Assalamualaikum warahmatullahi wabarakatuh. Selamat pagi, Teman-teman semuanya. Apa kabarnya hari ini? Semoga dalam keadaan sehat, ya.

Pagi ini kita bertemu lagi untuk membahas materi **Peluang**. Lalu, ada juga materi **Persamaan dan Pertidaksamaan**. Jadi, insyaallah kita akan bahas dua materi itu hari ini. Teman-teman sudah pada mengerjakan soal *pre-test*-nya belum? Sebelum kita mulai nih, sudah mencoba mengerjakan soal PR-nya? Udah, ya.

Oke, kalau teman-teman sudah mengerjakan soal PR-nya, berarti setidaknya sudah mulai pahamlah ya arah materi kita hari ini mau ke mana. Jadi kalau kita lihat, memang untuk materi PR, jika sudah paham itu bagus. Kalau belum paham, itu juga satu langkah yang baik. Artinya, *next*-nya teman-teman sudah tahu harus mempelajari bagian yang mana.

Kalau teman-teman mau lihat hasil pengerjaan PR-nya, untuk soal nomor satu masih setengah-setengah hasilnya. Soal nomor satu itu tentang rumus dasar dari peluang sebenarnya. Kalau yang nomor dua, sudah pada benar. Jadi, sebenarnya hasil PR-nya lumayan baik ya, Teman-teman. Kalau untuk persamaan dan pertidaksamaan, apalagi tuh, sebagian besar sudah pada benar. Mungkin kesulitannya ada di bagian tertentu, tapi semuanya sudah lumayan. Nanti sambil jalan kita bahas soal persamaan dan pertidaksamaannya juga.

### Konsep Dasar Peluang

Sebelum kita bahas soal PR-nya, mungkin kita balik lagi dulu ke materi dasar. Kan peluang ini mungkin tidak terlalu terpakai ya di kehidupan sehari-hari. Jadi kalau teman-teman agak lupa atau tidak terlalu familiar, itu maklum banget. Kita coba mengenal lebih jauh lagi tentang peluang.

Peluang itu kalau di rumusnya ditulis $P(A)$, yaitu peluang terjadinya kejadian A. Rumus dasarnya adalah:

$$P(A) = \frac{n(A)}{n(S)}$$

Di mana $n(A)$ adalah kejadian yang mau kita cari, dibagi dengan $n(S)$ yaitu banyak kejadian keseluruhannya atau semesta.

Nilai dari peluang itu *range*-nya dari 0 sampai 1. Jadi kalau peluangnya 0, berarti tidak mungkin terjadi. Tapi kalau peluangnya 1, sudah pasti terjadi.

Misalnya, kalau teman-teman tahu mata dadu itu ada enam. Kalau ditanya banyak kemungkinan mata dadu yang muncul adalah dari 1 sampai 6. Kita tahu mata dadunya memang hanya ada enam. Jadi kalau peluang munculnya mata dadu 1 sampai 6, berarti $n(A)$-nya ada 6, $n(S)$-nya ada 6 juga. $6/6 = 1$. Jadi kalau kita lempar satu mata dadu, sudah pasti yang muncul antara 1 sampai 6. Itu sudah pasti terjadi.

Tapi kalau ditanya peluang munculnya mata dadu 7, sedangkan dadunya cuma ada 6 mata dadu, jadi tidak mungkin ada mata dadu 7. Maka $0/6 = 0$, tidak mungkin terjadi.

### Frekuensi Harapan

Berkaitan dengan peluang, ada juga yang namanya **Frekuensi Harapan (FH)**. Konsepnya adalah semakin banyak percobaan, maka frekuensi harapan kejadian itu akan terjadi akan semakin besar. Jadi makin sering kita mencoba, kemungkinan berhasilnya makin besar.

Kalau ditanya frekuensi harapan, Teman-teman tinggal kalikan peluang kejadian itu terjadi dikalikan dengan $n$-nya (banyak percobaannya). Rumusnya:

$$FH = n \times P(A)$$

Di mana $n$ adalah banyak percobaan dan $P(A)$ adalah peluang kejadian tersebut.

### Hubungan Antar Dua Kejadian: Saling Lepas dan Saling Bebas

Sekarang kita coba mengenal kejadian-kejadian yang berkaitan dengan peluang. Tadi ingat ya, $P(A) = n(A) / n(S)$. Kadang dalam menentukan $n(A)$-nya, Teman-teman membutuhkan pengetahuan tambahan karena kejadian yang ditanya di soal bisa jadi ada dua kejadian, misalnya ada $n(A)$ kecil dan $n(B)$ kecil. Kita harus paham dulu ini kejadian saling lepas atau saling bebas.

#### 1. Kejadian Saling Lepas (Atau / Ditambah)
Kejadian saling lepas itu apabila terdapat dua buah kejadian A dan B, di mana tidak ada elemen pada kejadian A yang sama dengan elemen yang terdapat pada kejadian B. Peluang salah satu A atau B mungkin terjadi. Rumusnya adalah $n(A \cup B)$.

Contoh simpelnya: Asumsi awalnya adalah saya akan membawa payung ketika hujan. Ditanya peluang kejadian kita membawa payung **atau** tidak membawa payung. Kejadiannya hanya bisa terjadi salah satu. Kalau kita bawa payung berarti hujan. Kalau kita tidak bawa payung berarti tidak hujan. Jadi, hanya ada salah satu kejadian yang bisa terjadi. Itu disebut kejadian saling lepas.

Contoh soal: Sebuah kotak berisi bola merah dan putih. Diambil dua bola sekaligus. Peluang mendapatkan dua bola berwarna sama?
Bisa jadi dua-duanya merah **atau** dua-duanya putih. Kalau kita ambil dua-duanya merah, berarti kejadian dua-duanya putih tidak terjadi, dan sebaliknya.

**Kata kunci:** Teman-teman bisa melogikakan dengan kata **"ATAU"**. Hubungannya adalah **DITAMBAH**.

#### 2. Kejadian Saling Bebas (Dan / Dikali)
Dua buah kejadian A dan B dikatakan saling bebas jika munculnya kejadian A tidak mempengaruhi kejadian B. Peluang kejadian A **dan** B terjadi bersama-sama adalah $n(A) \times n(B)$.

Contoh: Kita mengambil bola satu per satu. Kita akan mengambil dua bola. Berarti kita mengalami dua kejadian: pengambilan bola pertama **dan** pengambilan bola kedua.

**Kata kunci:** Teman-teman bisa melogikakan dengan kata **"DAN"**. Hubungannya adalah **DIKALI**.

Jadi, kalau "atau" itu ditambah, kalau "dan" itu dikali.

### Permutasi dan Kombinasi

Selain kejadian saling lepas dan bebas, kita kadang membutuhkan materi permutasi atau kombinasi.

#### Permutasi (Memperhatikan Urutan)
Permutasi digunakan ketika teman-teman diminta untuk memilih dengan memperhatikan susunan atau urutan.

Contoh: Ada lima orang, mau dipilih dua orang untuk menjadi **Ketua dan Wakil**.
Jika yang terpilih A dan B, maka urutan siapa Ketua dan siapa Wakil itu berbeda. A jadi Ketua dan B jadi Wakil beda dengan B jadi Ketua dan A jadi Wakil. Karena urutan diperhatikan ($AB \neq BA$), kita pakai permutasi.

Rumus Permutasi:
$$_nP_x = \frac{n!}{(n-x)!}$$

#### Kombinasi (Tidak Memperhatikan Urutan)
Kombinasi digunakan ketika teman-teman diminta untuk memilih saja tanpa memperhatikan urutan.

Contoh: Ada lima orang, diminta memilih dua orang menjadi **anggota**.
Tidak ada peran spesifik. Jika yang terpilih A dan B, itu sama saja dengan B dan A ($AB = BA$).

Rumus Kombinasi:
$$_nC_x = \frac{n!}{(n-x)! \cdot x!}$$

#### Menyusun Semua Objek & Permutasi Unsur Sama
*   **Menyusun Semua:** Kalau teman-teman diminta menyusun semua objek yang ada, tinggal difaktorialkan saja. Contoh: Menyusun 7 orang, maka $7!$. Menyusun 3 huruf ABC, maka $3! = 3 \times 2 \times 1 = 6$.
*   **Permutasi Unsur Sama:** Kalau ada unsur yang sama, rumusnya adalah $n!$ dibagi dengan faktorial dari jumlah unsur yang sama.
    Contoh: Menyusun huruf "ACC". Jumlah huruf 3, huruf C ada 2.
    $$\frac{3!}{2!} = 3$$

Contoh lain: Kata "MATEMATIKA".
Total huruf 10. M ada 2, A ada 3, T ada 2.
$$\frac{10!}{2! \cdot 3! \cdot 2!}$$

---

### Pembahasan Soal Pre-Test & Latihan

Sekarang kita coba bahas soal *pre-test*-nya. Pertidaksamaan nanti saja, kita fokus ke peluang dulu.

**Soal 1: Frekuensi Harapan**
*Dari 60 kali pelemparan dadu, maka frekuensi harapan munculnya mata dadu faktor dari 6 adalah...*

Frekuensi harapan = Banyak percobaan $\times P(A)$.
Kita cari $P(A)$ dulu. $P(A) = n(A) / n(S)$.
Kejadian A (faktor dari 6): Angka yang bisa membagi habis 6 adalah 1, 2, 3, 6. Ada 4 angka. Jadi $n(A) = 4$.
$n(S)$ (mata dadu) = 6.
$P(A) = 4/6$.
Frekuensi Harapan = $60 \times (4/6) = 10 \times 4 = 40$.

**Soal 2: Permutasi Jabatan**
*Dari tujuh orang calon akan dipilih tiga orang untuk jabatan ketua, sekre, dan bendahara.*

Karena ada jabatan (role), urutan diperhatikan. Pakai Permutasi.
$_7P_3 = \frac{7!}{(7-3)!} = \frac{7!}{4!} = 7 \times 6 \times 5 = 210$.
Bisa juga pakai "Kotak Kemungkinan": 7 (Ketua) $\times$ 6 (Sekre) $\times$ 5 (Bendahara) = 210.

**Soal 3: Pengambilan Bola (Saling Lepas)**
*Sebuah kotak berisi 6 bola merah dan 6 bola putih. Diambil dua bola sekaligus. Peluang mendapatkan dua bola berwarna sama?*

Total bola = 12. Diambil 2.
Dua bola warna sama artinya: (2 Merah) **ATAU** (2 Putih).
Pakai kombinasi karena urutan bola tidak penting, dan ditambah karena "atau".
$P(A) = \frac{_6C_2 + _6C_2}{_{12}C_2}$.
$_6C_2 = 15$. $_{12}C_2 = 66$.
$P(A) = \frac{15 + 15}{66} = \frac{30}{66}$.
Disederhanakan bagi 6 = **5/11**.

**Soal 4: Memilih Soal Ujian**
*Tokyo diminta mengerjakan 8 soal dari 18 soal. Nomor 1 sampai 5 harus dikerjakan. Banyak pilihan soal?*

Total soal yang harus dipilih = 8.
Soal wajib (1-5) = 5 soal. Ini sudah pasti terpilih ($_5C_5 = 1$).
Sisa soal yang harus dipilih = $8 - 5 = 3$ soal lagi.
Sisa stok soal = $18 - 5 = 13$ soal.
Jadi, kita memilih 3 soal dari 13 soal sisa.
$_{13}C_3 = \frac{13 \times 12 \times 11}{3 \times 2 \times 1} = 286$ pilihan.

**Soal 5: Dua Dadu (Jumlah 9 atau 10)**
*Dua buah dadu dilempar. Peluang muncul jumlah mata dadu 9 atau 10?*

$n(S)$ dua dadu = $6 \times 6 = 36$.
Kejadian jumlah 9: (3,6), (4,5), (5,4), (6,3) $\rightarrow$ Ada 4.
Kejadian jumlah 10: (4,6), (5,5), (6,4) $\rightarrow$ Ada 3.
Karena "atau", kita jumlahkan: $4 + 3 = 7$.
Peluang = **7/36**.

*Catatan:* Kenapa tidak pakai kombinasi?
Ada pertanyaan, "Kenapa nggak pakai kombinasi Kak?". Kombinasi itu dipakai kalau kita memilih semuanya. Tapi kalau cuma pilih satu atau kasus dadu ini, kita pakai ruang sampel saja. Kalau kasus bola tadi, misal mau diambil satu bola saja, sebenarnya itu $_6C_1$, tapi karena hasilnya sama dengan 6, kita langsung tulis 6 saja.

---

### Materi Persamaan dan Pertidaksamaan

Kita lanjut bahas Persamaan dan Pertidaksamaan.

**Soal 6: Persamaan Linear**
*$-102x + 17y = 136$. Maka $66x - 11y = ...$*

Tips: Sederhanakan dulu persamaannya. Lihat angka terkecil ($17y$). Coba bagi semua dengan 17.
$-102x/17 + 17y/17 = 136/17$
$-6x + y = 8$.
Soal minta $66x - 11y$.
Dari $-6x$ supaya jadi $66x$, dikali $-11$.
Cek: $(-6x + y) \times -11 = 66x - 11y$.
Ruas kanan juga dikali: $8 \times -11 = -88$.
Jadi jawabannya **-88**.

**Soal 7: Himpunan Nilai (Pertidaksamaan)**
*Jika $x > 0$ dan $x \leq 5$, $y \geq -4$ dan $y < 4$. Manakah yang tidak termasuk himpunan nilai $x \cdot y$?*

$x$ (bilangan asli) = {1, 2, 3, 4, 5}.
$y$ = {-4, -3, -2, -1, 0, 1, 2, 3} (4 tidak ikut karena $<$).
Untuk mencari range $x \cdot y$, gunakan angka terkecil dan terbesar (ekstrem).
$x_{kecil} \cdot y_{kecil} = 1 \cdot (-4) = -4$
$x_{kecil} \cdot y_{besar} = 1 \cdot 3 = 3$
$x_{besar} \cdot y_{kecil} = 5 \cdot (-4) = -20$
$x_{besar} \cdot y_{besar} = 5 \cdot 3 = 15$
Nilai paling kecil adalah -20, paling besar adalah 15.
Jadi range-nya: $-20 \leq xy \leq 15$.
Cari di pilihan ganda angka yang di luar range ini. Misalnya 25, itu tidak termasuk.

**Soal 9: Akar Persamaan Kuadrat**
*Salah satu akar persamaan $x^2 + (a+1)x + (3a+2) = 0$ adalah 5. Akar lainnya adalah...*

Kalau salah satu akarnya 5, berarti $x = 5$. Masukkan ke persamaan.
$5^2 + (a+1)5 + (3a+2) = 0$
$25 + 5a + 5 + 3a + 2 = 0$
$32 + 8a = 0 \rightarrow 8a = -32 \rightarrow a = -4$.
Masukkan $a = -4$ ke persamaan awal:
$x^2 + (-3)x - 10 = 0$
Faktorkan: $(x-5)(x+2) = 0$.
Akar-akarnya $x=5$ dan **$x=-2$**.

*Tambahan Materi:*
Rumus jumlah dan hasil kali akar:
$x_1 + x_2 = -b/a$
$x_1 \cdot x_2 = c/a$

**Soal 10: Konsep Dasar Pertidaksamaan Pecahan**
Untuk pertidaksamaan pecahan, ada aturan penting:
1.  **Tidak boleh kali silang** jika ada variabel di penyebut dan pembilang.
2.  **Penyebut tidak boleh sama dengan 0**.
3.  Pindahkan semua ke satu ruas, samakan penyebut.
4.  Buat **garis bilangan** dengan mencari pembuat nol.
5.  **Tanda pertidaksamaan berubah** jika dikali/dibagi bilangan negatif.
6.  Cek tanda (+/-) di garis bilangan. Gunakan koefisien pangkat tertinggi atau uji titik. Pangkat ganjil tanda berubah, pangkat genap tanda tetap.

---

### Pembahasan Soal Post-Test

Kita ulas sedikit beberapa soal dari *post-test* yang tadi dikerjakan.

**Soal 1 Post-Test: Susunan Huruf BIOLA**
*Banyak susunan huruf BIOLA sehingga tidak ada dua huruf hidup berurutan.*
Huruf hidup (I, O, A) = 3. Huruf mati (B, L) = 2.
Agar tidak berurutan, huruf hidup harus diselang-seling huruf mati.
Pola: Hidup - Mati - Hidup - Mati - Hidup.
Posisi Hidup: 3 faktorial ($3 \times 2 \times 1$).
Posisi Mati: 2 faktorial ($2 \times 1$).
Total = $3! \times 2! = 6 \times 2 = 12$.

**Soal 2 Post-Test: Memilih Buku**
*20 buku (10 biru, 10 lain). Ambil 17 buku (seluruh biru + sisanya).*
Ambil 10 biru dari 10 biru = $_{10}C_{10} = 1$.
Sisa butuh 7 buku lagi dari 10 buku non-biru = $_{10}C_7$.
$_{10}C_7 = \frac{10 \times 9 \times 8}{3 \times 2 \times 1} = 120$.
Total cara = $1 \times 120 = 120$.

**Soal 3 Post-Test: Kartu Remi (Inklusi-Eksklusi)**
*Peluang terambil kartu Merah ATAU bernomor 13 dari 2 set kartu (104 lembar).*
$P(A \cup B) = P(A) + P(B) - P(A \cap B)$.
Total kartu ($nS$) = 104.
Kartu Merah ($nA$): 26 kartu (karena 2 set).
Kartu Nomor 13 ($nB$): Ada 4 warna $\times$ 2 set = 8 kartu.
Irisan (Merah DAN Nomor 13): Ada 2 kartu.
Hitungan: $(26 + 8 - 2) / 104 = 32 / 104$.
Sederhanakan bagi 4 = **8/26**.
*Hati-hati, jangan sampai menghitung ganda (double counting).*

**Soal 4 Post-Test: Permutasi Unsur Sama (ONE PIECE)**
Kata "ONE PIECE". Total 8 huruf.
Huruf sama: E ada 3. Sisanya 1.
Rumus: $8! / 3! = 6.720$.

**Soal 5 Post-Test: Ketua, Wakil, Anggota**
10 orang (6 Pria, 4 Wanita). Pilih 2 (Ketua, Wakil) dan 3 Anggota.
Ketua & Wakil (Perhatikan urutan) = $_{10}P_2$.
Sisa orang = 8. Pilih 3 Anggota (Tanpa urutan) = $_8C_3$.
Total = $_{10}P_2 \times _8C_3 = 90 \times 56 = 5.040$.

**Soal 7 Post-Test: Jumlah Akar Bulat Positif**
Persamaan $x^2 - 6x + q = 0$. Akar-akar bulat positif. Jumlah semua nilai $q$ yang mungkin?
$x_1 + x_2 = -b/a = -(-6)/1 = 6$.
Cari pasangan bilangan bulat positif yang jumlahnya 6:
(1, 5), (2, 4), (3, 3).
Nilai $q = x_1 \cdot x_2$.
Kemungkinan $q$:
$1 \times 5 = 5$
$2 \times 4 = 8$
$3 \times 3 = 9$
Jumlah semua nilai $q = 5 + 8 + 9 = 22$.

**Soal 8 Post-Test: Pertidaksamaan Sederhana**
$3x / (2-x) < 3$.
Pindah ruas: $3x/(2-x) - 3 < 0$.
Samakan penyebut, sederhanakan pembilang.
Dapat pembuat nol: $x=1$ dan $x=2$.
Cek garis bilangan. Hasilnya $x < 1$ atau $x > 2$. (Perhatikan syarat penyebut tidak boleh 0).

**Soal 10 Post-Test: Aljabar Kreatif**
$X = 667 \times 665 - 666^2 + 1$.
Ubah ke bentuk $(666+1)(666-1) - 666^2 + 1$.
$(666^2 - 1) - 666^2 + 1 = 0$. Jadi $X = 0$.
$Y = 120^2 - 121 \times 119$.
$120^2 - (120+1)(120-1) = 120^2 - (120^2 - 1) = 1$. Jadi $Y = 1$.
Kesimpulan: **X < Y**.

***

Oke, Teman-teman. Itu tadi pembahasan soal *pre-test* dan *post-test* kita. Semoga bisa dipahami, ya. Untuk pertidaksamaan, kuncinya jangan kali silang sembarangan dan hati-hati dengan tanda negatif. Untuk peluang, perhatikan kata kunci "dan" atau "atau", serta urutan (permutasi vs kombinasi).

Terima kasih banyak teman-teman yang sudah *join*. Jaga selalu semangat belajarnya. Semoga selalu sehat. *Sorry* kalau ada kesalahan dan kekurangan, ya.

Wassalamualaikum warahmatullahi wabarakatuh. Selamat melanjutkan *weekend*, Teman-teman. Makasih!