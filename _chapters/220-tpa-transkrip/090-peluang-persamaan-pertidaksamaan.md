---
slug: tpa-9-t
title: Peluang & Persamaan & Pertidaksamaan
---
# Pembahasan Lengkap: Peluang, Persamaan, dan Pertidaksamaan (Persiapan Tubel)

**Assalamualaikum warahmatullahi wabarakatuh.**

Selamat pagi teman-teman semuanya. Apa kabarnya hari ini? Semoga dalam keadaan sehat ya. Ketemu lagi kita hari ini dalam suasana yang agak beda karena sedang puasa. Semoga tetap semangat ya, *Guys*.

Pada kesempatan kali ini, kita akan membahas materi tentang **Peluang, Persamaan, dan Pertidaksamaan**. Sebelum kita mulai masuk ke materi, saya ingin mengecek apakah teman-teman sudah mengerjakan soal PR-nya? Dari hasil pengerjaan PR, terlihat yang agak susah sepertinya di soal nomor 3 tentang peluang, dan nomor 4. Selain itu, soal persamaan dan pertidaksamaan juga lumayan menantang, kecuali soal nomor 9 dan 10.

Oleh karena itu, kita akan mengerjakan soal sambil membahas materinya sedikit demi sedikit. Mari kita mulai.

## Mengingat Kembali Materi Peluang

Mungkin kita mulai dari materi peluang dulu. Peluang ini pernah tidak muncul sama sekali di tahun tertentu, tapi sepertinya di tahun lalu muncul kembali. Makanya, dari tim persiapan Tubel menambahkan materi peluang untuk kita pelajari lagi. Karena materi ini mungkin tidak sering kita pakai di kehidupan sehari-hari, sangat wajar jika teman-teman lupa rumusnya. Mari kita ingat kembali bersama-sama.

### Rumus Dasar dan Frekuensi Harapan
Rumus dasar peluang kejadian A ($P(A)$) adalah:
$$P(A) = \frac{n(A)}{n(S)}$$
Dimana:
*   $n(A)$ adalah banyak kejadian yang ditanya di soal.
*   $n(S)$ adalah banyak kejadian semesta secara keseluruhan.

Peluang memiliki *range* antara 0 sampai 1. Jika 0 berarti tidak mungkin terjadi sama sekali, dan jika 1 berarti sudah pasti terjadi.

Selain itu, dikenal juga istilah **Frekuensi Harapan**. Seperti yang teman-teman kerjakan di soal *pretest*, peluang harapan adalah peluang dikali dengan banyaknya percobaan ($n$). Logikanya, semakin sering kita mencoba, harapannya kejadian itu akan semakin mungkin terjadi.
Contohnya, jika peluangnya setengah ($1/2$) dan dilakukan satu kali, frekuensi harapannya tetap setengah. Tapi jika dilakukan dua kali, maka $2 \times 1/2$ hasilnya frekuensi harapannya menjadi 1.

### Kejadian Saling Lepas vs Saling Bebas
Dalam menentukan $n(A)$, ada dua jenis kejadian yang sering keluar:

**1. Kejadian Saling Lepas (Menggunakan Logika "ATAU")**
Dua kejadian A dan B disebut saling lepas jika tidak ada elemen pada kejadian A yang sama dengan elemen pada kejadian B. Peluang salah satu A atau B mungkin terjadi adalah dengan menjumlahkannya.
*   **Kata kunci:** "Atau".
*   **Rumus:** $P(A) + P(B)$ atau $n(A) + n(B)$.
*   **Contoh:** Peluang hari hujan *atau* saya membawa payung. Asumsinya, jika hujan bawa payung, jika tidak hujan tidak bawa payung. Kejadiannya hanya salah satu, tidak mungkin berbarengan. Jadi, kalau saling lepas itu **ditambah**.

**2. Kejadian Saling Bebas (Menggunakan Logika "DAN")**
Dikatakan saling bebas jika munculnya kejadian A tidak mempengaruhi kejadian B. Peluang kejadian A dan B terjadi bersama-sama.
*   **Kata kunci:** "Dan".
*   **Rumus:** $P(A) \times P(B)$.
*   **Contoh:** Memilih satu laki-laki di kelas A *dan* satu laki-laki di kelas B. Apa yang terjadi di kelas A tidak mempengaruhi kelas B. Jadi, kalau saling bebas itu **dikali**.

**Catatan:** Jangan terpaku menghafal "kalau bola ditambah, kalau koin dikali". Itu tergantung kasus soalnya. Ingat saja: kalau logikanya "atau" (saling lepas) berarti ditambah, kalau logikanya "dan" (saling bebas) berarti dikali.

### Permutasi dan Kombinasi
Selanjutnya, kita mengenal Permutasi dan Kombinasi.

**1. Permutasi (Memperhatikan Urutan)**
Permutasi digunakan ketika kita memilih dengan memperhatikan urutan.
*   **Contoh:** Dari 5 orang (A, B, C, D, E) dipilih 2 orang untuk jadi Ketua dan Wakil.
    *   Jika terpilih A dan B: A Ketua, B Wakil.
    *   Jika dibalik: B Ketua, A Wakil.
    *   Dalam permutasi, AB tidak sama dengan BA ($AB \neq BA$). Dua kejadian ini dihitung berbeda.
*   **Rumus:**
    $$nPr = \frac{n!}{(n-r)!}$$

**2. Kombinasi (Tidak Memperhatikan Urutan)**
Dalam kombinasi, urutan tidak penting.
*   **Contoh:** Dari 5 orang dipilih 2 untuk jadi anggota tim.
    *   Terpilih A dan B. Mau A dulu atau B dulu yang dipilih, hasilnya sama saja, mereka berdua jadi tim.
    *   Dalam kombinasi, AB sama dengan BA ($AB = BA$). Hanya dihitung satu kejadian.
*   **Rumus:**
    $$nCr = \frac{n!}{(n-r)! \cdot r!}$$

**Catatan Tambahan:**
*   Jika diminta **memilih lalu menyusun**, gunakan Permutasi.
*   Jika hanya diminta **memilih**, gunakan Kombinasi.
*   Jika hanya diminta **menyusun** (semua unsur dipakai), gunakan Faktorial ($n!$). Contoh: Menyusun huruf ABC (3 huruf), maka $3! = 6$ cara.

**Permutasi dengan Unsur yang Sama**
Bagaimana jika kita menyusun huruf "ABB"? Apakah ada 3 atau 2 B? Jika ada unsur yang sama, rumusnya adalah:
$$\frac{n!}{k! \cdot m! \dots}$$
(Dibagi dengan faktorial dari jumlah unsur yang sama).
Contoh: Menyusun ABB. Jumlah huruf 3, huruf B ada 2. Maka: $3! / 2! = 3$ cara.

---

## Pembahasan Soal Latihan (Peluang)

Mari kita terapkan materi di atas ke dalam soal.

**Soal 1: Frekuensi Harapan Dadu**
*Soal:* Dari 60 kali pelemparan dadu, frekuensi harapan munculnya mata dadu faktor dari 6 adalah...
*Bahasan:*
Faktor dari 6 adalah angka yang bisa membagi habis 6, yaitu: 1, 2, 3, dan 6. Ada 4 angka.
$n$ (percobaan) = 60.
$n(A)$ (kejadian faktor 6) = 4.
$n(S)$ (total mata dadu) = 6.
Peluang = $4/6$.
Frekuensi Harapan = $60 \times (4/6) = 40$.

**Soal 2: Susunan Jabatan**
*Soal:* Dari 7 orang calon akan dipilih 3 orang untuk jabatan Ketua, Sekretaris, dan Bendahara.
*Bahasan:*
Karena ada jabatan (Ketua, Sekre, Bendahara), maka urutan diperhatikan (siapa jadi apa). Gunakan Permutasi.
$7P3 = \frac{7!}{(7-3)!} = \frac{7!}{4!} = 7 \times 6 \times 5 = 210$.
Bisa juga pakai "Kotak Kemungkinan":
Kotak 1 (Ketua): 7 pilihan.
Kotak 2 (Sekre): 6 pilihan (satu sudah jadi ketua).
Kotak 3 (Bendahara): 5 pilihan.
Total: $7 \times 6 \times 5 = 210$.

**Soal 3: Pengambilan Bola (Saling Lepas)**
*Soal:* Kotak berisi 6 merah dan 6 putih. Diambil 2 bola sekaligus. Peluang mendapatkan dua bola berwarna sama.
*Bahasan:*
Dua bola warna sama artinya: (2 Merah) **ATAU** (2 Putih). Karena "atau", ini kejadian saling lepas (ditambah).
Total bola = 12. Diambil 2.
$n(S) = 12C2 = 66$.
Kejadian 2 Merah ($n(A)$) = Dari 6 merah ambil 2 = $6C2 = 15$.
Kejadian 2 Putih ($n(B)$) = Dari 6 putih ambil 2 = $6C2 = 15$.
Peluang = $(15 + 15) / 66 = 30/66$. Disederhanakan dibagi 6 menjadi **5/11**.

**Soal 4: Memilih Soal Ujian**
*Soal:* Mengerjakan 8 dari 18 soal. Nomor 1 sampai 5 harus dikerjakan.
*Bahasan:*
Total soal 18. Harus pilih 8.
Syarat: 1-5 wajib.
Artinya, 5 soal pertama sudah pasti terpilih (1 cara: $5C5$).
Sisa soal yang harus dipilih: Kita butuh 8, sudah ada 5, berarti kurang 3 soal lagi.
Sisa pilihan soal: Total 18, dikurang 5 yang wajib, sisa 13 soal.
Jadi, kita memilih 3 soal dari 13 sisa soal.
Rumus: $1 \times 13C3$.
$13C3 = \frac{13 \times 12 \times 11}{3 \times 2 \times 1} = 286$ cara.

**Soal 5: Dadu Berjumlah 9 atau 10**
*Soal:* Dua dadu dilempar. Peluang jumlah mata dadu 9 atau 10.
*Bahasan:*
Ruang sampel 2 dadu = 36.
Jumlah 9: (3,6), (6,3), (4,5), (5,4) $\rightarrow$ Ada 4.
Jumlah 10: (4,6), (6,4), (5,5) $\rightarrow$ Ada 3. (Ingat, 5,5 hanya dihitung satu kali).
Karena "atau", maka dijumlah: $4 + 3 = 7$.
Peluang = **7/36**.

---

## Persamaan dan Pertidaksamaan

Mari kita lanjut ke persamaan linear dan kuadrat serta pertidaksamaan.

**Soal 6: Persamaan Linear**
*Soal:* Diketahui $-102x + 17y = 136$. Maka nilai $66x - 11y = \dots$
*Bahasan:*
Sederhanakan persamaan yang diketahui. Bagi kedua ruas dengan 17.
$-6x + y = 8$.
Lihat yang ditanya: $66x - 11y$.
Jika persamaan tadi dikali dengan -11, maka:
$(-6x)(-11) + (y)(-11) = 8(-11)$
$66x - 11y = -88$.

**Soal 7: Range Nilai Perkalian (x kali y)**
*Soal:* $0 < x \le 5$ dan $-4 \le y < 4$. Manakah yang *tidak* termasuk himpunan nilai $xy$?
*Bahasan:*
Gunakan nilai terkecil dan terbesar dari range.
$x$: (mendekati 0) sampai 5.
$y$: -4 sampai (mendekati 4).
Cari kemungkinan ekstrem perkalian:
1.  Min $x$ dikali Min $y$: (anggap $>0$) $\times -4 \approx$ mendekati 0.
2.  Min $x$ dikali Max $y$: (anggap $>0$) $\times 4 \approx$ mendekati 0.
3.  Max $x$ dikali Min $y$: $5 \times -4 = -20$.
4.  Max $x$ dikali Max $y$: $5 \times 3$ (karena $y < 4$, ambil batas integer terdekat atau limitnya) $\approx 15$.
Range $xy$ adalah dari **-20 sampai kurang dari 20**.
Dari pilihan jawaban, angka **25** jelas tidak termasuk karena batas maksimumnya sekitar 20 (atau 15 jika bilangan bulat).

**Soal 9: Akar Persamaan Kuadrat**
*Soal:* Salah satu akar $x^2 + (a+1)x + (3a+2) = 0$ adalah 5. Akar lainnya adalah?
*Bahasan:*
Karena 5 adalah akar, substitusi $x = 5$ ke persamaan.
$5^2 + (a+1)5 + 3a+2 = 0$
$25 + 5a + 5 + 3a + 2 = 0$
$8a + 32 = 0 \rightarrow 8a = -32 \rightarrow a = -4$.
Masukkan $a = -4$ ke persamaan awal:
$x^2 + (-4+1)x + (3(-4)+2) = 0$
$x^2 - 3x - 10 = 0$.
Faktorkan: $(x - 5)(x + 2) = 0$.
Akarnya adalah $x = 5$ dan **$x = -2$**.

### Pertidaksamaan Pecahan
Ada aturan main yang harus diingat untuk pertidaksamaan pecahan:
1.  **Tidak boleh kali silang.** Pindahkan semua ke satu ruas agar ruas kanan = 0.
2.  **Penyebut tidak boleh nol.**
3.  **Buat Garis Bilangan.** Cari pembuat nol pembilang dan penyebut.
4.  **Uji Tanda.** Tentukan daerah positif/negatif.
5.  **Tanda Pertidaksamaan.** Tanda berubah jika dikali/dibagi bilangan negatif.

**Soal 10: Pertidaksamaan Pecahan**
*Soal:* $\frac{x+2}{x} \le \frac{x+3}{x-2}$
*Bahasan:*
Pindahkan ke kiri:
$\frac{x+2}{x} - \frac{x+3}{x-2} \le 0$
Samakan penyebut:
$\frac{(x+2)(x-2) - x(x+3)}{x(x-2)} \le 0$
$\frac{(x^2 - 4) - (x^2 + 3x)}{x(x-2)} \le 0$
$\frac{-3x - 4}{x(x-2)} \le 0$

Cari pembuat nol:
Pembilang: $-3x - 4 = 0 \rightarrow x = -4/3$.
Penyebut: $x = 0$ dan $x = 2$.
Garis bilangan: $-4/3$, $0$, $2$.

Cek tanda paling kanan (lihat koefisien $x$):
Pembilang: $-3$ (negatif).
Penyebut: $1 \times 1 = 1$ (positif).
Negatif dibagi positif = Negatif.
Maka paling kanan **Negatif**. Karena pangkat ganjil semua, selang-seling:
Positif | Negatif | Positif | Negatif.
Yang diminta $\le 0$ (Negatif).
Daerah: $x$ antara $-4/3$ dan $0$, atau $x > 2$.
Perhatikan syarat penyebut: $x \neq 0$ dan $x \neq 2$. Jadi tanda di 0 dan 2 tidak pakai "sama dengan".
Himpunan penyelesaian: **$-4/3 \le x < 0$ atau $x > 2$**.

---

## Pembahasan Post-Test (Soal Tambahan)

Berikut pembahasan singkat dari sesi Post-Test:

1.  **Susunan Kata "BIOLA" (Syarat Huruf Hidup)**
    *   Tidak boleh dua huruf hidup berurutan. Huruf hidup (I, O, A), Mati (B, L).
    *   Pola: Hidup-Mati-Hidup-Mati-Hidup.
    *   Tempat Hidup: $3! = 6$. Tempat Mati: $2! = 2$.
    *   Total: $6 \times 2 = 12$ cara.

2.  **Mengambil Buku dengan Syarat Warna**
    *   20 buku (10 biru, 10 lain). Ambil 17. Syarat: Semua biru terambil.
    *   Ambil 10 biru (dari 10) = $10C10 = 1$.
    *   Sisa ambil 7 lagi dari 10 buku non-biru = $10C7 = 120$.
    *   Total: $1 \times 120 = 120$ cara.

3.  **Kartu (Merah atau Nomor 13)**
    *   Total 104 kartu (2 set).
    *   Merah: $13 \times 2 = 26$.
    *   Nomor 13: Ada 4 warna x 2 set = 8 kartu.
    *   Irisan (Merah & No 13): Ada 2 kartu.
    *   Rumus gabungan: $(26 + 8 - 2) / 104 = 32/104 = 8/26$.

4.  **Susunan Kata "ONE PIECE"**
    *   Total 8 huruf. Huruf sama: E (3 huruf).
    *   Permutasi unsur sama: $8! / 3! = 6.720$.

5.  **Tim Kerja (Ketua, Wakil, Anggota)**
    *   10 orang. Pilih Ketua & Wakil (Permutasi) dan 3 Anggota (Kombinasi).
    *   Pilih Ketua & Wakil dari 10: $10P2 = 90$.
    *   Sisa 8 orang, pilih 3 Anggota: $8C3 = 56$.
    *   Total: $90 \times 56 = 5.040$.

6.  **Sistem Persamaan**
    *   $3x + 5y = 27$ dan $2x + 5y = 23$.
    *   Kurangkan langsung: $x = 4$.
    *   Substitusi: $2(4) + 5y = 23 \rightarrow 5y = 15 \rightarrow y = 3$.
    *   Jawaban: $x=4, y=3$.

7.  **Akar Bulat Positif Persamaan Kuadrat**
    *   $x^2 - 6x + q = 0$.
    *   $x_1 + x_2 = -b/a = 6$.
    *   $x_1 \cdot x_2 = q$.
    *   Pasangan bilangan bulat positif yang jumlahnya 6: (1,5) dan (2,4) dan (3,3).
    *   Nilai $q$ yang mungkin (hasil kali):
        *   $1 \times 5 = 5$
        *   $2 \times 4 = 8$
        *   $3 \times 3 = 9$
    *   Jumlah semua nilai $q$: $5 + 8 + 9 = 22$.

8.  **Pertidaksamaan Rasional**
    *   $\frac{3x}{2-x} < 3$
    *   Pindah ruas: $\frac{3x}{2-x} - 3 < 0$.
    *   Samakan penyebut: $\frac{3x - 3(2-x)}{2-x} < 0 \rightarrow \frac{6x - 6}{2-x} < 0$.
    *   Pembuat nol: $x=1$ dan $x=2$.
    *   Uji tanda (Koefisien atas positif, bawah negatif $\rightarrow$ Kanan Negatif).
    *   Garis bilangan: (Negatif) 1 (Positif) 2 (Negatif).
    *   Diminta $<0$ (Negatif). Maka $x < 1$ atau $x > 2$.

9.  **Logika Pertidaksamaan**
    *   $A > B > C$ dan $X > Y > Z$.
    *   Hubungan: $A > C > Y > Z$. Dan $X > B$.
    *   Dari pilihan, yang pasti benar adalah $A > Z$ (karena A paling atas, Z paling bawah).

10. **Aljabar Selisih Kuadrat**
    *   $x = 666 \times 666 \times 665 - 666^2 + 1$.
        *   Gunakan manipulasi aljabar atau lihat polanya. Ternyata hasilnya 0.
    *   $y = 120^2 - 121 \times 119$.
        *   $121 \times 119 = (120+1)(120-1) = 120^2 - 1$.
        *   $y = 120^2 - (120^2 - 1) = 1$.
    *   Kesimpulan: $x < y$.

---

Demikian pembahasan materi dan soal untuk hari ini. Terima kasih banyak teman-teman sudah bergabung. Semoga semangat belajarnya selalu bertahan dan bertambah. Mohon maaf jika ada kekurangan.

**Wassalamualaikum warahmatullahi wabarakatuh.**