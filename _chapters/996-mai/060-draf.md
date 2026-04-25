---
slug: mai-6
title: mai-6
---

# Bab 6: Konsep Matematika pada Aljabar Linear

## 6.1 Aljabar Linear

Aljabar merupakan cabang matematika Aljabar linear merupakan topik yang sangat penting dari matematika Aljabar yang banyak digunakan dalam berbagai dasar ilmu keteknikan, dan juga diperdalam bahkan diperluas lagi dalam berbagai mata kuliah; komputasi numerik, fenomena perpindahan, aliran fluida, perancangan struktur, rekayasa reaksi kimia, pemodelan, program linear, teknik informatika dan masih banyak yang lainnya. Selain itu, Penerapan aljabar linear di dalam ilmu komputer diantaranya; penggunaan aplikasi matlab, *image processing* dalam proses pengenalan wajah, *game* komputer dan kecerdasan buatan (AI). Dalam buku ini akan dibahas secara singkat beberapa topik Aljabar linear diantaranya mengenai; pengenalan matriks, operasi matriks; persamaan linear, sistem linear; determinan, kofaktor, hubungan bilangan skalar, determinan matriks; pengenalan vektor, vektor dalam sistem koordinat, penjumlahan dan pengurangan vektor, perkalian skalar dengan vektor, kombinasi linear, panjang vektor, *dot product*, *cross product*; *eigen value* dan *eigen vector*; transformasi linear.

## 6.2 Matriks

### 6.2.1 Pengenalan Matriks

Bentuk matriks pertama kali diperkenalkan oleh ahli matematika dari Inggris bernama James Sylvester tahun 1850 dengan istilah *Oblong Arrangement of Terms*. Banyak para ahli yang menyebut matriks berupa susunan berupa data sains dalam bentuk baris dan kolom membentuk persegi panjang diantaranya menurut Anton & Rorres (2014). *Information in science, business, and mathematics is often organized into rows and columns to form rectangular arrays called “matrices”* (plural of “matrix”). Matriks merupakan bilangan-bilangan yang berbentuk *array* persegi panjang. Matriks yang merupakan susunan informasi numerik, sangat diperlukan untuk mengembangkan program komputer dalam memecahkan sistem persamaan, karena komputer sangat cocok untuk memanipulasi susunan informasi numerik. Matriks dalam penerapannya bukan hanya sebatas notasi untuk menyelesaikan sistem persamaan, melainkan dapat dipandang sebagai objek matematika tersendiri. Topik matematika yang terkait dengan hal tersebut yaitu Aljabar Linear. Pada matriks, bilangan dalam *array* disebut *entries*. Gunakan huruf kapital untuk menunjukkan matriks dan huruf kecil untuk menunjukkan *entries*. Lambang *entries* yang terdapat pada baris ditulis dengan i sedangkan *entries* pada kolom dilambangkan j. Ukuran matriks berdasarkan banyaknya jumlah baris m (horizontal) dan jumlah kolom n (vertical). Matriks yang hanya mempunyai satu baris disebut matriks baris (*row matrix* atau *row vektor*). Sedangkan matriks yang hanya mempunyai satu kolom disebut matriks kolom (*column matrix* atau *column vektor*). Misal, matrik A dengan *entries* $a_{ij}$ atau bisa ditulis $(A)_{ij}$ berukuran $2 \times 3$, angka 2 menunjukkan banyaknya jumlah baris, dan angka 3 menunjukkan banyaknya jumlah kolom, sehingga matriks A bisa ditulis:

$$A = \begin{bmatrix} a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \end{bmatrix} \tag{6.1}$$

*Entries* $a_{11}$ berarti menunjukkan baris ke-1 kolom ke-1 pada matriks A, $a_{21}$ berarti menunjukkan baris ke-2 kolom ke-1 pada matriks A. Jika Matriks A mempunyai jumlah baris sama dengan jumlah kolom ($m = n$) disebut matrik persegi berorde n, dengan diagonal utama matriks A tersebut $a_{11}, a_{22}, \dots, a_{nn}$.

$$A = \begin{bmatrix} a_{11} & \dots & a_{1n} \\ \vdots & \ddots & \vdots \\ a_{n1} & \dots & a_{nn} \end{bmatrix} \tag{6.2}$$

Jika matriks persegi dengan diagonal utamanya mempunyai nilai = 1, maka disebut dengan matriks Identitas (I).

$$I = \begin{bmatrix} 1 & \dots & 0 \\ \vdots & \ddots & \vdots \\ 0 & \dots & 1 \end{bmatrix} \tag{6.3}$$

### 6.2.2 Operasi Matriks

Operasi aritmatika pada matriks dapat berupa, penjumlahan, pengurangan dan perkalian. Berikut beberapa hal yang berkaitan dengan operasi pada matriks.

**1. Dua matriks didefinisikan sama, ketika ukurannya sama dan *entries* yang bersesuaian sama.**

Contoh:
$$A = \begin{bmatrix} 1 & 2 \\ 3 & x \end{bmatrix} \quad B = \begin{bmatrix} 1 & 2 \\ 3 & 5 \end{bmatrix}$$
Matriks A berukuran $2 \times 2$ = Matriks B berukuran $2 \times 2$. Sehingga $x = 5$.

$$C = \begin{bmatrix} 1 & 1 & 2 \\ 2 & 4 & 5 \end{bmatrix}$$
Matriks A berukuran $2 \times 2 \neq$ Matriks C berukuran $2 \times 3$.

**2. Jika Matriks A dan Matriks B berukuran sama, maka operasi aritmatika penjumlahan dapat dilakukan dengan menjumlahkan *entries* yang bersesuaian pada Matriks A dan Matriks B. Selain itu, operasi aritmatika pengurangan dapat dilakukan dengan mengurangkan *entries* yang bersesuaian pada matriks A dan Matriks B. Tetapi kalau matriks A dan matriks B berukuran beda, operasi aritmatika penjumlahan dan pengurangan pada matriks A dan matriks B tidak bisa dilakukan.**

Contoh:
$$A = \begin{bmatrix} 1 & 1 \\ 2 & 3 \\ 4 & 5 \end{bmatrix} \quad B = \begin{bmatrix} 2 & 3 \\ 4 & 5 \\ 2 & 3 \end{bmatrix} \quad C = \begin{bmatrix} 2 & 4 \\ 3 & 1 \end{bmatrix}$$

$$A + B = \begin{bmatrix} 1 & 1 \\ 2 & 3 \\ 4 & 5 \end{bmatrix} + \begin{bmatrix} 2 & 3 \\ 4 & 5 \\ 2 & 3 \end{bmatrix} = \begin{bmatrix} 3 & 4 \\ 6 & 8 \\ 6 & 8 \end{bmatrix}$$

$$B - A = \begin{bmatrix} 2 & 3 \\ 4 & 5 \\ 2 & 3 \end{bmatrix} - \begin{bmatrix} 1 & 1 \\ 2 & 3 \\ 4 & 5 \end{bmatrix} = \begin{bmatrix} 1 & 2 \\ 2 & 2 \\ -2 & -2 \end{bmatrix}$$

$A + C, B + C, A - C, B - C$ *undefined* dikarenakan matriks C berbeda ukuran dengan matriks A dan matriks B.

**3. Jika matriks A dikalikan dengan bilangan skalar c, maka setiap *entries* pada matriks A dikalikan dengan skalar c.**

Contoh:
$$A = \begin{bmatrix} 1 & 1 \\ 2 & 3 \\ 4 & 5 \end{bmatrix} \quad B = \begin{bmatrix} 2 & 3 \\ 4 & 5 \\ 2 & 3 \end{bmatrix} \quad C = \begin{bmatrix} 2 & 4 \\ 3 & 1 \end{bmatrix}$$

$$2A = 2 \times \begin{bmatrix} 1 & 1 \\ 2 & 3 \\ 4 & 5 \end{bmatrix} = \begin{bmatrix} 2 & 2 \\ 4 & 6 \\ 8 & 10 \end{bmatrix}$$

$$-3B = -3 \times \begin{bmatrix} 2 & 3 \\ 4 & 5 \\ 2 & 3 \end{bmatrix} = \begin{bmatrix} -6 & -9 \\ -12 & -15 \\ -6 & -9 \end{bmatrix}$$

$$5C = 5 \times \begin{bmatrix} 2 & 4 \\ 3 & 1 \end{bmatrix} = \begin{bmatrix} 10 & 20 \\ 15 & 5 \end{bmatrix}$$

**4. Jika matriks A berukuran $m \times n$, maka *transpose* matriks A ($A^T$) berukuran $n \times m$ yang dihasilkan dengan menukarkan baris dengan kolom.**

Contoh:
$$A = \begin{bmatrix} 1 & 1 \\ 2 & 3 \\ 4 & 5 \end{bmatrix} \quad A^T = \begin{bmatrix} 1 & 2 & 4 \\ 1 & 3 & 5 \end{bmatrix}$$

**5. Jika dan hanya jika matriks A merupakan matriks persegi, maka *trace* matrik A dinotasikan $tr(A)$ dan didefinisikan sebagai jumlah *entries* diagonal utama matriks A.**

Contoh:
$$A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 4 & 2 \\ 1 & 3 & 5 \end{bmatrix} \quad tr(A) = 1 + 4 + 5 = 10$$

**6. Jika Matriks A berukuran $m \times r$ dan matriks B berukuran $r \times n$, maka hasil kali matriks A dengan matriks B yaitu matriks AB akan berukuran $m \times n$. Ketika mencari *entries* baris i serta kolom j pada matriks AB, pilihlah baris i dari matriks A dan kolom j matriks B, lalu kalikan *entries-entries* yang bersesuaian, kemudian jumlahkan untuk mendapatkan hasilnya.**

Contoh:
Jika Matriks A berukuran $2 \times 3$ dan Matriks B berukuran $3 \times 4$, maka Matriks AB berukuran $2 \times 4$.

$$A = \begin{bmatrix} 1 & 2 & 4 \\ 2 & 6 & 0 \end{bmatrix} \quad B = \begin{bmatrix} 4 & 1 & 4 & 3 \\ 0 & -1 & 3 & 1 \\ 2 & 7 & 5 & 2 \end{bmatrix}$$

$$A \times B = \begin{bmatrix} 1 & 2 & 4 \\ 2 & 6 & 0 \end{bmatrix} \begin{bmatrix} 4 & 1 & 4 & 3 \\ 0 & -1 & 3 & 1 \\ 2 & 7 & 5 & 2 \end{bmatrix}$$

$\sum (Entries \text{ baris ke-1} \times Entries \text{ kolom ke-1}) = 1 \times 4 + 2 \times 0 + 4 \times 2 = 12$
$\sum (Entries \text{ baris ke-1} \times Entries \text{ kolom ke-2}) = 1 \times 1 + 2 \times -1 + 4 \times 7 = 27$
$\sum (Entries \text{ baris ke-1} \times Entries \text{ kolom ke-3}) = 1 \times 4 + 2 \times 3 + 4 \times 5 = 30$
$\sum (Entries \text{ baris ke-1} \times Entries \text{ kolom ke-4}) = 1 \times 3 + 2 \times 1 + 4 \times 2 = 13$
$\sum (Entries \text{ baris ke-2} \times Entries \text{ kolom ke-1}) = 2 \times 4 + 6 \times 0 + 0 \times 2 = 8$
$\sum (Entries \text{ baris ke-2} \times Entries \text{ kolom ke-2}) = 2 \times 1 + 6 \times -1 + 0 \times 7 = -4$
$\sum (Entries \text{ baris ke-2} \times Entries \text{ kolom ke-3}) = 2 \times 4 + 6 \times 3 + 0 \times 5 = 26$
$\sum (Entries \text{ baris ke-2} \times Entries \text{ kolom ke-4}) = 2 \times 3 + 6 \times 1 + 0 \times 2 = 12$

$$A \times B = \begin{bmatrix} 12 & 27 & 30 & 13 \\ 8 & -4 & 26 & 12 \end{bmatrix}$$

**7. Asumsikan bahwa ukuran matriks sedemikian rupa, sehingga operasi-operasi matrik berikut dapat dilakukan.**

| No | Operasi | Sifat |
| :--- | :--- | :--- |
| a. | $A + B = B + A$ | Sifat Komutatif Penjumlahan |
| b. | $A + (B + C) = (A + B) + C$ | Sifat Asosiatif Penjumlahan |
| c. | $A(BC) = (AB)C$ | Sifat Asosiatif Perkalian |
| d. | $A(B + C) = AB + AC$ | Distribusi sebelah kiri |
| e. | $(B + C)A = BA + CA$ | Distribusi sebelah kanan |
| f. | $a(B \pm C) = aB \pm aC$ | |
| g. | $(a \pm b)C = aC \pm bC$ | |
| h. | $a(bC) = (ab)C$ | |
| i. | $a(BC) = (aB)C = B(aC)$ | |

## 6.3 Sistem Persamaan Linear

### 6.3.1 Pengenalan Persamaan Linear

Biasanya matrik muncul berupa tabel data numerik dari suatu pengamatan atau dalam bentuk konteks matematika. Misalnya, informasi mengenai sistem persamaan berikut yang diwujudkan dalam bentuk matriks.

Sistem Persamaan:
$x - y = 1$
$2x + y = 6$

Bentuk Matriks:
$$\begin{bmatrix} 1 & -1 & 1 \\ 2 & 1 & 6 \end{bmatrix}$$

Solusi sistem persamaan di atas dapat diperoleh dengan melakukan operasi yang sesuai pada matriks tersebut. Sebagai mana diketahui bahwa dalam dua dimensi sebuah garis pada sistem koordinat $x-y$ dapat direpresentasikan oleh sebuah persamaan dalam bentuk:
$ax + by = c$ yakni $a, b \neq 0$

Persamaan dalam bentuk di atas merupakan contoh persamaan linear dengan variable $x$ dan $y$. Sedangkan untuk tiga dimensi sebuah garis pada sistem koordinat $xyz$ dapat direpresentasikan oleh sebuah persamaan dalam bentuk:
$ax + by + cz = d$ yakni $a, b, c \neq 0$

Persamaan dalam bentuk di atas merupakan contoh persamaan linear dengan variable $x, y,$ dan $z$. Secara umum bentuk persamaaan linear untuk $n$ variabel $x_1, x_2, x_3, \dots, x_n$ dapat ditulis sebagai berikut:

$$a_1x_1 + a_2x_2 + a_3x_3 + \dots + a_nx_n = b \tag{6.4}$$

dengan $a_1, a_2, a_3, \dots, a_n$ dan $b$ konstan, serta $a_1, a_2, a_3, \dots, a_n \neq 0$.
dengan $b = 0$, persamaan (6.4) menjadi:

$$a_1x_1 + a_2x_2 + a_3x_3 + \dots + a_nx_n = 0 \tag{6.5}$$

Persamaan (6.5) disebut persamaan linear homogen dengan variabel $x_1, x_2, x_3, \dots, x_n$.

Contoh Persamaan Linear:
a. $3y = 7 - x$
b. $-3x_3 + x_4 = 2x_2 - x_1$
c. $6z + 2 = 2y - x$

Contoh Persamaan Non Linear:
a. $x + 3y^2 - 4 = 0$
b. $\sin x + y = 0$
c. $3x + 2y = 5 + xy$

### 6.3.2 Sistem Persamaan Linear

Himpunan persamaan linear berhingga disebut sistem persamaan linear atau sistem linear, dan variabelnya disebut *unknowns* (yang tidak diketahui). Contoh sistem persamaan linear dengan *unknowns* $x$ dan $y$:
$5x + y - 3 = 0$
$2x - y - 4 = 0$

Contoh sistem persamaan linear dengan *unknowns* $x_1, x_2,$ dan $x_3$:
$3x_3 + 1 = x_2 - 4x_1$
$3x_1 + x_2 + 9x_3 + 4 = 0$

Secara umum bentuk sistem linear untuk $m$ persamaan dan $n$ variabel yang tidak diketahui (*unknowns*) $x_1, x_2, x_3, \dots, x_n$ dapat ditulis sebagai berikut:

$$a_{11}x_1 + a_{12}x_2 + a_{13}x_3 + \dots + a_{1n}x_n = b_1 \tag{6.6}$$
$$a_{21}x_1 + a_{22}x_2 + a_{23}x_3 + \dots + a_{2n}x_n = b_2$$
$$a_{31}x_1 + a_{32}x_2 + a_{33}x_3 + \dots + a_{3n}x_n = b_3$$
$$\vdots$$
$$a_{m1}x_1 + a_{m2}x_2 + a_{m3}x_3 + \dots + a_{mn}x_n = b_m$$

Koefisien $a_{ij}$ dari variabel yang tidak diketahui (*unknowns*) menunjukkan lokasinya pada sistem persamaan linear tersebut. Misal, $a_{12}$ menunjukkan persamaan ke-1 dan mengalikannya dengan *unknowns* $x_2$. Solusi sistem linear n variabel yang tidak diketahui (*unknowns*) $x_1, x_2, x_3, \dots, x_n$ yaitu barisan bilangan $s_1, s_2, s_3, \dots, s_n$ hasil substitusi dan dapat ditulis $(s_1, s_2, s_3, \dots, s_n)$.

$x_1 = s_1 \quad x_2 = s_2 \quad x_3 = s_3 \quad x_n = s_n$

Sistem linear dua variabel yang tidak diketahui (*unknowns*) berhubungan dengan perpotongan garis. Misal, untuk sistem linear berikut:
$a_1x + b_1y = c_1 \tag{6.7}$
$a_2x + b_2y = c_2$

dengan grafik persamaannya adalah garis pada bidang $xy$. Setiap solusi $(x,y)$ dari sistem linear dengan dua variabel yang tidak diketahui (*unknowns*) merupakan titik potong garis persamaan tersebut. Ada tiga kemungkinan yang akan terjadi, yaitu:

1. Garis-garis yang berbeda sejajar, tidak ada perpotongan garis, akibatnya tidak ada solusi.
   Contoh:
   $x + y = 4 \quad 3x + 3y = 6$
2. Garis-garis berpotongan hanya satu titik, akibatnya mempunyai tepat satu solusi.
   Contoh:
   $x - y = 1 \quad 2x + y = 6$
3. Garis-garis yang berhimpitan (terdapat banyak sekali titik berpotongan atau titik-titik pada garis yang sama), akibatnya mempunyai banyak solusi.
   Contoh:
   $4x - 2y = 1 \quad 16x - 8y = 4$

Secara umum sistem linear dikatakan konsisten, jika sistem tersebut mempunyai paling sedikit satu solusi, dan tidak konsisten jika sistem tersebut tidak mempunyai solusi. Sistem linear konsisten untuk sistem linear yang terdiri dari dua persamaan dengan dua variabel yang tidak diketahui (*unknowns*). Gambar 6.1 akan memperjelas hal tersebut. Hal yang sama berlaku juga untuk tiga persamaan dengan tiga variabel yang tidak diketahui (*unknowns*).

$a_{11}x + a_{12}y + a_{13}z = d_1 \tag{6.8}$
$a_{21}x + a_{22}y + a_{23}z = d_2$
$a_{31}x + a_{32}y + a_{33}z = d_3$

Ketika grafik persamaan berbentuk bidang pada koordinat $xyz$. Setiap solusi $(x,y,z)$ dari sistem linear dengan tiga variabel yang tidak diketahui (*unknowns*) merupakan titik-titik yang bersesuaian dengan ketiga bidang yang berpotongan. Ada tiga kemungkinan, yaitu tidak ada solusi, tepat satu solusi dan banyak solusi. Gambar 6.2 akan memperjelas hal tersebut.

*(Gambar 6.1: Sistem Linear dengan Dua Persamaan Linear)*
*(Gambar 6.2: Sistem Linear dengan Tiga Persamaan Linear)*
*Sumber Anton & Rorres, 11th Edition*

Operasi Aritmatika perkalian matriks memiliki peranan penting dalam penerapan mencari solusi sistem linear. Pada saat jumlah persamaan linear dan jumlah variabel yang tidak diketahui (*unknowns*) makin banyak maka mencari solusi sistem linear menjadi lebih kompleks. Supaya perhitungan mencari solusi lebih mudah, maka perlu menyederhanakan notasi dan menstandarkan prosedurnya. Di sini digunakan matriks untuk mendandakan sistem linear. Istilah matriks biasanya digunakan dalam matematika untuk menunjukkan susunan angka yang berbentuk persegi panjang. Secara umum bentuk sistem linear untuk m persamaan linear dan n variabel yang tidak diketahui (*unknowns*) $x_1, x_2, x_3, \dots, x_n$ sebagai berikut:

$a_{11}x_1 + a_{12}x_2 + a_{13}x_3 + \dots + a_{1n}x_n = b_1$
$a_{21}x_1 + a_{22}x_2 + a_{23}x_3 + \dots + a_{2n}x_n = b_2$
$\vdots$
$a_{m1}x_1 + a_{m2}x_2 + a_{m3}x_3 + \dots + a_{mn}x_n = b_m$

dapat ditulis dalam bentuk ***augmented matrix*** menjadi:

$$\begin{bmatrix} a_{11} & a_{12} & a_{13} \dots a_{1n} & b_1 \\ a_{21} & a_{22} & a_{23} \dots a_{2n} & b_2 \\ \vdots & \vdots & \vdots & \vdots \\ a_{m1} & a_{m2} & a_{m3} \dots a_{mn} & b_m \end{bmatrix} \tag{6.9}$$

Selain itu, dengan menggunakan definisi bahwa dua matrik dikatakan sama, jika dan hanya jika ukurannya sama dan *entries-entries* yang bersesuaian sama. Sehingga sistem linear untuk m persamaan liniear dan n variabel yang tidak diketahui (*unknowns*) dapat ditulis menjadi persamaan matriks kolom ($m \times 1$).

$$\begin{bmatrix} a_{11}x_1 + a_{12}x_2 + a_{13}x_3 \dots a_{1n}x_n \\ a_{21}x_1 + a_{22}x_2 + a_{23}x_3 \dots a_{2n}x_n \\ \vdots \\ a_{m1}x_1 + a_{m2}x_2 + a_{m3}x_3 \dots a_{mn}x_n \end{bmatrix} = \begin{bmatrix} b_1 \\ b_2 \\ \vdots \\ b_m \end{bmatrix} \tag{6.10}$$
*Matrik berukuran (size) $m \times n$ $\times$ Matrik (size) $m \times 1$*

$$\begin{bmatrix} a_{11} & a_{12} & a_{13} & \dots & a_{1n} \\ a_{21} & a_{22} & a_{23} & \dots & a_{2n} \\ \vdots & \vdots & \vdots & & \vdots \\ a_{m1} & a_{m2} & a_{m3} & \dots & a_{mn} \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix} = \begin{bmatrix} b_1 \\ b_2 \\ \vdots \\ b_m \end{bmatrix} \tag{6.11}$$
*Matriks berukuran $m \times n$ $n \times 1$ $m \times 1$*

Sehingga bisa ditulis $Ax = b$, dengan matriks A sebagai koefisien matriks pada sistem linear tersebut. ***Augmented matrix*** dapat ditulis dengan menggabungkan *entries* matriks A dengan b, dan disimpan di kolom terakhir, menjadi seperti di bawah ini.

$$\begin{bmatrix} a_{11} & a_{12} & a_{13} \dots a_{1n} & b_1 \\ a_{21} & a_{22} & a_{23} \dots a_{2n} & b_2 \\ \vdots & \vdots & \vdots & \vdots \\ a_{m1} & a_{m2} & a_{m3} \dots a_{mn} & b_m \end{bmatrix} \tag{6.12}$$

Metode dasar untuk menyelesaikan sistem linear yaitu dengan melakukan operasi aljabar dengan tidak mengubah himpunan solusi untuk menghasilkan sistem yang makin sederhana dan konsisten. Sistem linear dalam bentuk *augmented matrix* di atas dapat diselesaikan dengan menggunakan operasi baris dasar (*elementary row operations*) sebagai berikut:

* Kalikan baris dengan konstanta tak nol
* Tukarkan dua baris
* Tambahkan baris yang telah dikalikan dengan konstanta dengan baris lainnya.

Berikut contoh menyelesaikan sistem linear dengan tiga persamaan dan tiga variabel yang tidak diketahui menggunakan bentuk *augmented matrix* dan operasi baris dasar (*elementary row operations*).

| Sistem Linear | Operasi Baris Dasar | Augmented Matrix |
| :--- | :--- | :--- |
| $x + y + 2z = 9$ | | $\begin{bmatrix} 1 & 1 & 2 & 9 \\ 2 & 4 & -3 & 1 \\ 3 & 6 & -5 & 0 \end{bmatrix}$ |
| $2x + 4y - 3z = 1$ | | |
| $3x + 6y - 5z = 0$ | | |
| | -2 x baris ke-1 + baris ke-2 | |
| $x + y + 2z = 9$ | -3 x baris ke-1 + baris ke-3 | $\begin{bmatrix} 1 & 1 & 2 & 9 \\ 0 & 2 & -7 & -17 \\ 3 & 6 & -5 & 0 \end{bmatrix}$ |
| $2y - 7z = -17$ | | |
| $3x + 6y - 5z = 0$ | | |
| $x + y + 2z = 9$ | 1/2 x baris ke-2 | $\begin{bmatrix} 1 & 1 & 2 & 9 \\ 0 & 1 & -7/2 & -17/2 \\ 0 & 3 & -11 & -27 \end{bmatrix}$ |
| $y - 7/2z = -17/2$ | | |
| $3y - 11z = -27$ | | |
| $x + y + 2z = 9$ | -3 x baris ke-2 + baris ke-3 | $\begin{bmatrix} 1 & 1 & 2 & 9 \\ 0 & 1 & -7/2 & -17/2 \\ 0 & 0 & -1/2 & -3/2 \end{bmatrix}$ |
| $y - 7/2z = -17/2$ | | |
| $-1/2z = -3/2$ | | |
| $x + y + 2z = 9$ | -2 x baris ke-3 | $\begin{bmatrix} 1 & 1 & 2 & 9 \\ 0 & 1 & -7/2 & -17/2 \\ 0 & 0 & 1 & 3 \end{bmatrix}$ |
| $y - 7/2z = -17/2$ | | |
| $z = 3$ | | |
| $x + y + 2z = 9$ | -1 x baris ke-2 + baris ke-1 | $\begin{bmatrix} 1 & 0 & 11/2 & 35/2 \\ 0 & 1 & -7/2 & -17/2 \\ 0 & 0 & 1 & 3 \end{bmatrix}$ |
| $y - 7/2z = -17/2$ | | |
| $z = 3$ | | |
| $x + 11/2z = 35/2$ | -11/2 x baris ke-3 + baris ke-1 | $\begin{bmatrix} 1 & 0 & 0 & 1 \\ 0 & 1 & 0 & 2 \\ 0 & 0 & 1 & 3 \end{bmatrix}$ |
| $y - 7/2z = -17/2$ | 7/2 x baris ke-3 + baris ke-2 | |
| $z = 3$ | | |

Jadi, solusinya $x = 1, y = 2,$ dan $z = 3$ untuk sistem linear:
$x + y + 2z = 9$
$2x + 4y - 3z = 1$
$3x + 6y - 5z = 0$

## 6.4 Determinan

### 6.4.1 Menghitung Determinan

Sistem linear tertentu dapat dicari penyelesaiannya yaitu menggunakan konsep determinan, yaitu persamaan linear tersebut dibuat bentuk matriks terlebih dahulu. Misal, jika matrik A dibentuk dari dua persamaan linear dengan ukuran $2 \times 2$ seperti di bawah ini:
$$A = \begin{bmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{bmatrix}$$
determinan matriks A dapat ditulis:
$$\det(A) = \det \begin{bmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{bmatrix} = \begin{vmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{vmatrix} = a_{11}a_{22} - a_{12}a_{21} \tag{6.13}$$

Matriks A tersebut mempunyai invers ($A^{-1}$) jika dan hanya jika $\det(A) = a_{11}a_{22} - a_{12}a_{21} \neq 0$, sehingga invers dari matriks A ($A^{-1}$) dapat dicari menggunakan rumus berikut:
$$A^{-1} = \frac{1}{a_{11}a_{22} - a_{12}a_{21}} \begin{bmatrix} a_{22} & -a_{12} \\ -a_{21} & a_{11} \end{bmatrix} \tag{6.14}$$

Rumus tersebut bisa diuji kevalidannya, ketika matriks A dikalikan dengan invers matriks A ($A^{-1}$) akan menghasilkan matriks Identitas (I).
$$A \cdot A^{-1} = I = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} \tag{6.15}$$

Jika matriks A dan matriks B mempunyai invers, maka invers dari matriks AB dapat ditulis:
$$(AB)^{-1} = B^{-1}A^{-1} \tag{6.16}$$

Jika matriks A dan matriks B merupakan matriks persegi berukuran sama, maka berlaku:
$$\det(AB) = \det(A) \det(B) \tag{6.17}$$

### 6.4.2 Kofaktor

Jika matriks A merupakan matriks persegi, maka minor *entries* $a_{ij}$ dinotasikan $M_{ij}$, dan didefinisikan sebagai determinan sub matriks tersisa, setelah baris ke-i dan kolom ke-j dihapus dari matriks A. Nilai $(-1)^{i+j}M_{ij}$ ditulis sebagai $C_{ij}$ dan disebut kofaktor dengan *entries* $a_{ij}$.

Contoh:
$$A = \begin{bmatrix} 3 & 1 & -4 \\ 2 & 5 & 6 \\ 1 & 4 & 8 \end{bmatrix}$$
Minor *entries* $a_{11}$ ditulis $M_{11}$, baris ke-1 dan kolom ke-1 pada matriks A dicoret, sehingga determinan sub matriksnya menjadi:
$$M_{11} = \begin{vmatrix} 5 & 6 \\ 4 & 8 \end{vmatrix} = 40 - 24 = 16$$
Kofaktor dengan *entries* $a_{11}$ dapat ditulis $C_{11}$ yaitu:
$C_{11} = (-1)^{1+1}M_{11} = (-1)^2 \cdot 16 = 1 \cdot 16 = 16$

### 6.4.3 Hubungan Bilangan Skalar dan Determinan Matriks

Beberapa kemungkinan hubungan $\det(A)$ dengan $\det(B)$ yang akan terjadi untuk matriks A, matriks B dan matriks C yang berukuran $n \times n$ dengan $k$ bilangan sembarang skalar, antara lain:
1. $\det(kA) = k^n \det(A)$
2. $\det(A + B) \neq \det(A) + \det(B)$
3. $\det(C) = \det(A) + \det(B)$
4. $\det(AB) = \det(A) \det(B)$
5. $\det(A^{-1}) = \frac{1}{\det(A)}$
6. *Transpose* dari matrik kofaktor disebut *adjoint* dari matrik A dan ditulis $adj(A)$.
7. Hubungan invers matriks A dengan *adjoint* matrik A:
   $$A^{-1} = \frac{1}{\det(A)} adj(A)$$

## 6.5 Ruang Vektor

### 6.5.1 Pengenalan Vektor

Besaran menurut para ahli fisika dan insinyur dibedakan menjadi dua, antara lain besaran skalar dan besaran vektor. Besaran Skalar yaitu besaran yang hanya memiliki nilai numerik saja, sedangkan besaran vektor yaitu besaran yang memiliki nilai dan deskripsi arah secara lengkap. Contoh yang termasuk besaran skalar: suhu, panjang, waktu, kecepatan. Contoh besaran vektor kelajuan, gaya, perpindahan, tekanan, medan magnet. Walaupun besaran skalar dan vektor berasal dari keteknikan dan fisika, namun demikian dalam buku ini dibahas supaya dapat membangun struktur matematika yang bisa diterapkan pada bidang komputer, pemrograman dan *Artificial Intelligence* (AI) atau kecerdasan buatan. Pokok materi utama pada aljabar linear yaitu kecerdasan dan vektor. Matrik sudah dijelaskan pada sub bab sebelumnya. Pada sub bab selanjutnya akan diulas mengenai sifat-sifat dasar vektor pada dua dimensi (*2-space*), tiga dimensi (*3-space*) dan $R^n$ dimensi. Hal yang perlu diperhatikan pada vektor; panjang panah menentukan nilai atau besarnya vektor; ekor anak panah disebut titik awal (*initial point*); ujung anak panah disebut titik akhir (*terminal point*). Pada buku ini vektor simbol vektor ditulis dengan huruf kecil tebal misal, **a, b, c, v, w, x**. Sedangkan skalar ditulis dengan huruf kecil miring misal, $a, b, c, v, w, x$. Pada saat kita akan menunjukkan bahwa pada suatu vektor **v** mempunyai titik awal A dan titik akhir B maka dapat dinotasikan:
$$\mathbf{v} = \vec{AB} \tag{6.18}$$

Vektor dengan panjang dan arah yang sama disebut dengan vektor yang *equivalent*. Misal, vektor $\mathbf{v = u = w}$ seperti pada Gambar 6.3 berikut yang menunjukkan mengenai geometri vektor dengan dibuat menggunakan Aplikasi Geogebra.

*(Gambar 6.3 Geometri Vektor)*
*Sumber Penulis*

### 6.5.2 Vektor dalam Sistem Koordinat

Perhitungan vektor berdimensi dua, tiga dan berdimensi n ($R^n$) akan lebih mudah dipahami dan dicari solusinya dengan sistem koordinat. Misal, jika vektor **v** yang *initial point*nya pada titik *origin* (0,0) berada pada dimensi dua $\mathbf{v} = (v_1, v_2)$ dengan komponen $v_1, v_2$ dan dimensi tiga $\mathbf{v} = (v_1, v_2, v_3)$ dengan komponen $v_1, v_2, v_3$ dapat dilihat dalam sistem koordinat berikut pada Gambar 6.4.

*(Gambar 6.4 Vektor dalam Sistem Koordinat)*
*Sumber Anton & Rorres, 11th Edition*

Dapat ditulis bentuk vektor yang dibatasi koma pada dimensi $R^n$ dengan notasi:
$$\mathbf{v} = (v_1, v_2, \dots, v_n) \tag{6.19}$$

Ketika bentuk vektor tersebut tidak dibatasi koma, maka dapat ditulis:
*Vektor baris* $\mathbf{v} = \begin{bmatrix} v_1 & v_2 & \dots & v_n \end{bmatrix} \tag{6.20}$
*Vektor kolom* $\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix} \tag{6.21}$

Jika suatu vektor titik awal (*initial point*) tidak pada titik *origin*, maka dapat ditulis $\vec{P_1P_2}$, yaitu untuk vektor dimensi dua (*2-space*) titik awal $P_1(x_1, y_1)$ dan titik akhir $P_2(x_2, y_2)$. Sedangkan jika vektor tersebut berdimensi tiga (*3-space*) titik awal $P_1(x_1, y_1, z_1)$ dan titik akhir $P_2(x_2, y_2, z_2)$. Sehingga komponen $\vec{P_1P_2}$ dapat ditulis sebagai berikut:

$\vec{P_1P_2} = (x_2 - x_1, y_2 - y_1)$ *komponen vektor 2-space*
$\vec{P_1P_2} = (x_2 - x_1, y_2 - y_1, z_2 - z_1)$ *komponen vektor 3-space*

### 6.5.3 Penjumlahan dan Pengurangan Vektor

Operasi penjumlahan dua buah vektor berdimensi 2 dan berdimensi 3 mengikuti aturan jajaran genjang (Gambar 6.5a) dan aturan segitiga (Gambar 6.5b). Sifat komutatif berlaku pada penjumlahan dua buah vektor, hal tersebut dapat direkonstruksi dengan aturan jajaran genjang dan aturan segitiga (Gambar 6.5c).
$$\mathbf{v + w = w + v} \tag{6.22}$$

*(Gambar 6.5 Penjumlahan Vektor. Sumber Penulis)*

Pada aritmatika, operasi pengurangan dalam bentuk operasi penjumlah dapat ditulis $a - b = a + (-b)$, hal tersebut berlaku juga pada operasi pengurangan pada vektor. Misal, suatu vektor **v** merupakan vektor negatif dan ditulis **-v**, dimana panjangnya sama dengan vektor **v** namun arahnya yang berlawanan (Gambar 6.6a), operasi pengurangannya direkonstruksi dengan aturan jajaran genjang (Gambar 6.6b atau 6.6c) dan dapat ditulis:
$$\mathbf{w - v = w + (-v)} \tag{6.23}$$

*(Gambar 6.6 Pengurangan Vektor. Sumber Penulis)*

Jika $\mathbf{v} (v_1, v_2, \dots, v_n)$ dan $\mathbf{w} (w_1, w_2, \dots, w_n)$ merupakan vektor pada dimensi n ($R^n$), maka dapat didefinisikan penjumlahan dan pengurangan vektor sebagai berikut:
$$\mathbf{w + v = v + w} = (v_1+w_1, v_2+w_2, \dots, v_n+w_n) \tag{6.24}$$
$$\mathbf{w - v = w + (-v)} = (w_1-v_1, w_2-v_2, \dots, w_n-v_n) \tag{6.25}$$

### 6.5.4 Perkalian Skalar dengan Vektor

Pada waktu akan mengubah panjang vektor dan arah, maka harus dikalikan bilangan skalar. Jika **v** merupakan vektor tak nol di ruang dimensi dua ($R^2$) dan dimensi tiga ($R^3$), sedangkan $k$ bilangan skalar tak nol, maka dapat didefinisikan hasil kali skalar (*scalar product*) $k$ dengan **v** menjadi vektor yang panjangnya $|k|$ dikali **v** dengan arah sama dengan vektor **v** jika $k$ positif, dan berlawanan arah dengan vektor **v** jika $k$ negatif. Selain itu, jika $k = 0$ atau $\mathbf{v} = 0$, maka $\mathbf{kv} = 0$. Lebih jelasnya dapat dilihat pada Gambar 6.7 berikut:

*(Gambar 6.7 Perkalian Skalar dengan Vektor. Sumber Penulis)*

Jika $\mathbf{v} (v_1, v_2, \dots, v_n)$, merupakan vektor pada dimensi n ($R^n$) dan k bilangan skalar, maka kita dapat mendefinisikan $\mathbf{kv} = (kv_1, kv_2, \dots, kv_n)$.

### 6.5.5 Kombinasi Linear

Operasi aritmatika pada vektor penjumlahan, pengurangan dan perkalian skalar sering digunakan secara bersamaan (*combination*) membentuk vektor baru. Jika $k_1, k_2, \dots, k_n$ bilangan skalar yang disebut koefisien dan $\mathbf{w}$ vektor kombinasi linear dengan dimensi $R^n$ dari vektors, $\mathbf{v_1, v_2, v_3, \dots, v_n}$ dalam dimensi $R^n$. Maka dapat ditulis bentuk kombinasi linear sebagai berikut:
$$\mathbf{w} = k_1\mathbf{v_1} + k_2\mathbf{v_2} + \dots + k_n\mathbf{v_n} \tag{6.26}$$

### 6.5.6 Panjang Vektor, Dot Product dan Cross Product

Panjang vektor (*norm*) ditulis $\|\mathbf{v}\|$. Vektor $\mathbf{v} (v_1, v_2)$ di $R^2$ dan $\mathbf{v} (v_1, v_2, v_3)$ di $R^3$ dengan *initial point* di titik *origin* (0,0) panjang vektor $\|\mathbf{v}\|$ dapat ditentukan dengan menggunakan teorema Pythagoras seperti pada Gambar 6.8a ($R^2$) dan Gambar 6.8b ($R^3$).

Sehingga dapat ditulis $\|\mathbf{v}\| = \sqrt{v_1^2 + v_2^2}$ dan $\|\mathbf{v}\| = \sqrt{v_1^2 + v_2^2 + v_3^2}$. Secara umum, jika vektor $\mathbf{v} (v_1, v_2, \dots, v_n)$ dalam dimensi $R^n$ maka panjang vektor (*norm*) $\|\mathbf{v}\|$ didefinisikan dengan rumus berikut:
$$\|\mathbf{v}\| = \sqrt{v_1^2 + v_2^2 + \dots + v_n^2} \tag{6.27}$$

*(Gambar 6.8 Panjang Vektor dengan Pythagoras)*
*Sumber Anton & Rorres, 11th Edition*

Jika vektor **u** dan **v** bukan vektor nol, berada dalam dimensi $R^2$ dan $R^3$, serta $\theta$ merupakan sudut diantara vektor **u** dengan vektor **v**, maka hasil kali titik (*Dot Product/Euclidean Inner Product*) vektor **u** dengan vektor **v** yang ditulis **u.v** dan didefinisikan:
$$\mathbf{u.v} = \|\mathbf{u}\| \|\mathbf{v}\| \cos \theta \tag{6.28}$$

Sehingga secara umum jika vektor $\mathbf{u} (u_1, u_2, \dots, u_n)$ dan vektor $\mathbf{v} (v_1, v_2, \dots, v_n)$ pada dimensi $R^n$. Hasil kali titik $\mathbf{u.v}$ didefinisikan
$$\mathbf{u.v} = u_1v_1 + u_2v_2 + \dots + u_nv_n \tag{6.29}$$

Hasil kali titik (*Dot product*) dua buah vektor akan menghasilkan skalar sedangkan hasil kali silang (*cross product*) menghasilkan vektor. Pada buku ini untuk *cross product* penulis batasi sampai ruang vektor dimensi 3 ($R^3$). Jika vektor $\mathbf{u} (u_1, u_2, u_3)$ dan vektor $\mathbf{v} (v_1, v_2, v_3)$ maka hasil kali silang vektor **u** dengan vektor **v** ditulis $\mathbf{u \times v}$ dan didefinisikan:
$$\mathbf{u \times v} = (u_2v_3 - u_3v_2, u_3v_1 - u_1v_3, u_1v_2 - u_2v_1) \tag{6.30}$$

dalam bentuk determinan:
$$\mathbf{u \times v} = \left( \begin{vmatrix} u_2 & u_3 \\ v_2 & v_3 \end{vmatrix}, -\begin{vmatrix} u_1 & u_3 \\ v_1 & v_3 \end{vmatrix}, \begin{vmatrix} u_1 & u_2 \\ v_1 & v_2 \end{vmatrix} \right) \tag{6.31}$$

## 6.6 Nilai Eigen (Eigenvalue) dan Vektor Eigen (Eigenvector)

Istilah Eigen diperkenalkan sekitar tahun 1900 di Jerman yaitu ketika mempelajari gerak rotasi, mengklasifikasikan berbagai jenis permukaan dan menjelaskan solusi persamaan differensial tertentu. Penerapan mengenai eigen ini pada matriks dan transformasi matrik saat bermanfaat untuk beberapa bidang ilmu pengetahuan, diantaranya: komputer grafis dan mekanika kuantum. Kata eigen berarti: khas, milik, kharakteristik, individu. Selanjutnya, Misal A merupakan matriks berukuran $n \times n$, kemudian **x** merupakan vektor tak nol pada dimensi $R^n$ dapat disebut vektor eigen (*eigenvektor*) dari matrik A (atau operator matrik $T_A$), jika **Ax** adalah perkalian skalar dengan vektor **x**.
$$\mathbf{Ax} = \lambda \mathbf{x} \tag{6.32}$$

yaitu untuk skalar $\lambda$ disebut nilai eigen (*eigenvalue*) dari matrik A ($T_A$), sehingga **x** merupakan vektor eigen yang sesuai dengan nilai eigen ($\lambda$). Besaran dan arah bergantung pada tanda dan nilai eigen $\lambda$ yang bersesuaian dengan vektor eigen (**x**), pembalikan arah terjadi jika nilai eigen $\lambda$ negatif.

*(Gambar 6.9 Nilai Eigen & Vektor Eigen. Sumber Penulis)*

Contoh: Diketahui matrik A dengan vektor eigen **x** sebagai berikut:
$$A = \begin{bmatrix} 3 & 0 \\ 8 & -1 \end{bmatrix} \quad \mathbf{x} = \begin{bmatrix} 1 \\ 2 \end{bmatrix}$$
Perkalian matriks A dengan vektor **x** menghasilkan nilai eigen $\lambda$ yang dikalikan dengan vektor eigen **x**.
$$\mathbf{Ax} = \lambda \mathbf{x}$$
$$\begin{bmatrix} 3 & 0 \\ 8 & -1 \end{bmatrix} \begin{bmatrix} 1 \\ 2 \end{bmatrix} = \begin{bmatrix} 3 \\ 6 \end{bmatrix} = 3 \begin{bmatrix} 1 \\ 2 \end{bmatrix}$$
Jadi nilai eigennya $\lambda = 3$

*(Gambar 6.10 Grafik Nilai Eigen & Vektor Eigen. Sumber Penulis)*

## 6.7 Transformasi Linear

Transformasi matriks merupakan fungsi-fungsi khusus yang berasal dari perkalian matriks yang mendasar untuk mempelajari transformasi linear. Penerapan transformasi matriks sangat berguna pada bidang teknik, fisika ataupun sosial.

Sebagaimana diketahui bahwa fungsi merupakan suatu aturan yang mengasosiasikan setiap elemen pada himpunan A (*domain*) satu dan hanya satu-satunya elemen himpunan B (*codomain*). Jika f merupakan sebuah fungsi dengan *domain* $R^n$ dan *codomain* $R^m$, lalu disebut fungsi f adalah transformasi $R^n$ menjadi $R^m$ atau fungsi tersebut memetakan $R^n$ ke $R^m$, hal tersebut dapat ditulis:
$$f : R^n \to R^m \tag{6.33}$$

Berdasarkan konsep fungsi di atas, dapat ditulis transformasi matriks (atau matriks operator jika $m = n$) sebagai berikut:
$$T_A : R^n \to R^m \tag{6.34}$$

$T_A$ menunjukkan terjadinya transformasi hasil dari perkalian vektor di $R^n$ menjadi $R^m$ dengan matriks A yang berukuran $m \times n$ menjadi pemetaan yang berbentuk:
$$T_A(\mathbf{x}) = A\mathbf{x} \tag{6.35}$$

*(Gambar 6.11 Transformasi Matrik A di $R^n$ ke $R^m$)*
*Sumber Anton & Rorres, 11th Edition*

Transformasi nol $T_0$ terjadi hasil dari perkalian vektor di $R^n$ menjadi $R^m$ dengan matrik 0 yang berukuran $m \times n$ sebagai berikut
$$T_0(\mathbf{x}) = 0\mathbf{x} = \mathbf{0} \tag{6.36}$$

Transformasi identitas $T_I$ atau disebut juga operator identitas, terjadi hasil dari perkalian vektor di $R^n$ menjadi $R^m$ ($m = n$), dengan matrik Identitas I yang berukuran $n \times n$ sebagai berikut:
$$T_I(\mathbf{x}) = I\mathbf{x} = \mathbf{x} \tag{6.37}$$

Transformasi matriks $T : R^n \to R^m$ jika dan hanya jika untuk semua vektor **u** dan **v** serta skalar $k$, berlaku hubungan sebagai berikut:
1. $T(\mathbf{u + v}) = T(\mathbf{u}) + T(\mathbf{v})$ *(Sifat Aditif)*
2. $T(k\mathbf{u}) = k T(\mathbf{u})$ *(Sifat Homogenitas)*

Sifat aditif dan sifat homogenitas tersebut disebut kondisi linear, transformasi yang memenuhi kondisi tersebut disebut transformasi linear. Setiap transformasi matriks dari $R^n$ ke $R^m$ merupakan transformasi linear begitu juga sebaliknya, setiap transformasi linear dari $R^n$ ke $R^m$ merupakan transformasi matriks. Sifat aditif dan sifat homogenitas tersebut dapat ditulis merupakan titik awal ketika kita akan mendefinisikan trasformasi linear umum. Misal, Transformasi linear $T : V \to W$ dapat digunakan dalam kombinasi untuk menunjukkan bahwa jika $\mathbf{v_1, v_2, \dots, v_r}$ adalah vektor-vektor di V dan $k_1, k_2, \dots, k_r$ beberapa skalar sehingga dapat ditulis:
$$T(k_1\mathbf{v_1} + k_2\mathbf{v_2} + \dots + k_r\mathbf{v_r}) = k_1T(\mathbf{v_1}) + k_2T(\mathbf{v_2}) + \dots + k_rT(\mathbf{v_r}) \tag{6.38}$$

***