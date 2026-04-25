---
slug: mai-7
title: mai-7
---

# Bab 7: Memahami Kompleksitas Komputasi: Konsep, Model Deterministik dan Non-Deterministik, Serta Aplikasi Praktis

## 7.1 Kompleksitas Komputasi

Kompleksitas komputasi adalah salah satu bidang yang paling penting dalam ilmu komputer, yang berfokus pada mempelajari seberapa efisien suatu algoritma dapat memecahkan masalah tertentu. Kompleksitas ini dapat diukur dari berbagai aspek, seperti waktu yang dibutuhkan (kompleksitas waktu) dan ruang memori yang diperlukan (kompleksitas ruang).

Kompleksitas waktu mengukur jumlah langkah yang diperlukan oleh sebuah algoritma untuk menyelesaikan masalah sebagai fungsi dari ukuran inputnya. Misalnya, jika dimiliki algoritma dengan kompleksitas waktu $O(n)$, ini berarti bahwa waktu yang dibutuhkan oleh algoritma tersebut berbanding lurus dengan ukuran input $n$. Contoh lain adalah algoritma dengan kompleksitas waktu $O(n^2)$, yang menunjukkan bahwa waktu yang diperlukan meningkat secara kuadrat seiring dengan peningkatan ukuran input.

Kompleksitas ruang mengukur jumlah memori yang diperlukan oleh sebuah algoritma selama eksekusinya sebagai fungsi dari ukuran input. Misalnya, algoritma dengan kompleksitas ruang $O(1)$ hanya memerlukan jumlah memori yang tetap, terlepas dari ukuran inputnya. Sebaliknya, algoritma dengan kompleksitas ruang $O(n)$ memerlukan memori yang berbanding lurus dengan ukuran input $n$.

Pemahaman yang mendalam tentang kompleksitas komputasi sangat penting untuk merancang algoritma yang efisien dan optimal. Algoritma yang efisien tidak hanya mempercepat proses komputasi, tetapi juga mengurangi penggunaan sumber daya, yang sangat penting dalam lingkungan dengan sumber daya terbatas. Misalnya, dalam aplikasi *real-time* atau sistem dengan daya komputasi rendah, algoritma yang efisien dapat membuat perbedaan antara kegagalan dan keberhasilan.

Selain itu, pemahaman tentang kompleksitas komputasi membantu dalam memilih algoritma yang tepat untuk masalah tertentu. Dalam beberapa kasus, algoritma sederhana dengan kompleksitas waktu yang lebih tinggi mungkin lebih cocok dibandingkan algoritma yang lebih kompleks tetapi lebih cepat, tergantung pada konteks dan kebutuhan spesifik.

Sebagai contoh penerapan, pertimbangkan masalah pencarian elemen dalam sebuah *array*. Algoritma pencarian linier memiliki kompleksitas waktu $O(n)$, yang berarti waktu yang dibutuhkan untuk mencari elemen dalam *array* berukuran $n$ adalah sebanding dengan $n$. Di sisi lain, algoritma pencarian biner memiliki kompleksitas waktu $O(\log n)$, yang jauh lebih efisien untuk *array* yang sudah terurut.

Dalam dunia nyata, berbagai masalah komputasi seperti penjadwalan, optimasi jaringan, analisis data besar, dan pengenalan pola sangat bergantung pada algoritma yang efisien. Oleh karena itu, studi tentang kompleksitas komputasi bukan hanya bidang teoretis, tetapi juga memiliki aplikasi praktis yang luas. Dengan demikian, memahami dan mengukur kompleksitas komputasi adalah langkah fundamental dalam pengembangan teknologi informasi dan ilmu komputer, membantu kita untuk terus meningkatkan efisiensi dan kinerja sistem komputasi.

## 7.2 Mesin Turing Deterministik (DTM)

**Komponen Utama:**

1.  **Unit Kontrol:** Menyimpan sejumlah keadaan terbatas.
2.  **Unit Memori:** Pita tak terbatas dibagi menjadi sel-sel yang menyimpan simbol.
3.  **Kepala Baca/Tulis:** Membaca, menulis, dan bergerak di sepanjang pita.

**Langkah Operasi DTM:**

1.  Membaca simbol di sel yang dipindai.
2.  Menulis simbol baru di sel tersebut.
3.  Memindahkan kepala pita (kiri atau kanan).
4.  Mengubah keadaan kontrol berdasarkan keadaan dan simbol yang dibaca.

**Definisi Formal:**

*   $Q$ : Set keadaan.
*   $q_0$ : Keadaan awal.
*   $F$ : Set keadaan penerimaan.
*   $\Sigma$ : Set simbol input.
*   $\Gamma$ : Set simbol pita (termasuk simbol kosong $B$).
*   $\Delta$ : Fungsi transisi, $\delta: (Q - F) \times \Gamma \to Q \times \Gamma \times \{L, R\}$.

**Konfigurasi:**

Konfigurasi $\alpha$ dari $MT \ M$ adalah elemen $(q, x, y)$ dari $Q \times \Gamma^* \times \Gamma^*$, dengan $q$ adalah keadaan saat ini, $x$ dan $y$ adalah simbol di pita, dan kepala pita memindai simbol pertama dari $y$.

**Fungsi Transisi:**

1.  **Kasus 1:** $\delta(q1, s1) = (q2, s2, L)$
    Jika $x1 = \lambda$, maka $s3 = B$; jika tidak, $x1 = x2s3$.
    $(q1, x1, y1) \vdash (q2, \ell(x2), r(s3s2y2))$.
2.  **Kasus 2:** $\delta(q1, s1) = (q2, s2, R)$
    $(q1, x1, y1) \vdash (q2, \ell(x1s2), r(y2))$.
3.  **Kasus 3:** $\delta(q1, s1)$ tidak terdefinisi.

**Penerimaan String:**

MT M menerima string $w$ jika urutan konfigurasi dari $\alpha 0$ ke $\alpha n$ dengan $\alpha 0 = (q_0, \lambda, w)$ dan keadaan akhir $\alpha n = (q, x, y)$ dengan $q \in F$.

**Contoh:**

Mesin M menerima string dalam $L = \{a^i b^j : 0 \le i \le j\}$.
Keadaan: $Q = \{q_0, q_1, q_2, q_3, q_4, q_5\}$

1.  Keadaan Awal: $q_0$
2.  Keadaan Penerimaan: $q_5$
3.  Simbol Input: $\Sigma = \{a, b\}$
4.  Simbol Pita: $\Gamma = \{a, b, c, B\}$

**Fungsi Transisi $\delta$:**

| $\delta$ | a | b | c | B |
| :--- | :--- | :--- | :--- | :--- |
| **q0** | (q1, c, R) | (q4, B, R) | (q0, B, R) | (q0, B, R) |
| **q1** | (q1, a, R) | (q2, b, R) | | |
| **q2** | (q3, c, L) | (q2, c, R) | (q2, B, R) | |
| **q3** | (q3, a, L) | (q3, b, L) | (q3, c, L) | (q0, B, R) |
| **q4** | (q4, a, R) | (q4, b, R) | (q5, B, R) | |

**Proses:**

Input "abaa":
$q_0 abaa \vdash cq_1 baa \vdash cbq_2 aa \vdash cq_3 bcaa \vdash q_3 cbcaa \vdash q_3 Bcbcaa \vdash q_0 cbcaa \vdash q_0 bcaa \vdash q_4 caa \vdash q_4 aa \vdash q_4 B \vdash q_5$.

DTM menerima string dalam $w$ jika, dan hanya jika, $w$ dalam $L$.

## 7.3 Mesin Turing Nondeterministik (NTM)

Mesin Turing deterministik (DTM) hanya memiliki satu langkah yang dapat diambil dari setiap konfigurasi, menghasilkan satu konfigurasi berikutnya. Sebaliknya, Mesin Turing nondeterministik (NTM) bisa memiliki beberapa langkah dari suatu konfigurasi, menghasilkan beberapa konfigurasi berikutnya. Sebuah NTM M didefinisikan oleh keadaan Q, keadaan awal q0, keadaan penerimaan F, simbol input $\Sigma$, simbol pita $\Gamma$ (termasuk simbol kosong B), dan relasi transisi $\Delta$. Relasi transisi $\Delta$ adalah himpunan bagian dari $(Q - F) \times \Gamma \times Q \times \Gamma \times \{L, R\}$, yang memungkinkan berbagai langkah potensial. Konfigurasi berikut dari sebuah NTM pada input $w$ diwakili oleh pohon perhitungan di mana setiap simpul adalah konfigurasi dan anak-anaknya adalah konfigurasi berikutnya. Sebuah NTM berhenti pada $w$ jika ada jalur terbatas dalam pohon perhitungannya, dan menerima $w$ jika jalur ini berakhir pada keadaan penerimaan. Bahasa yang diterima oleh sebuah NTM adalah himpunan string yang diterima oleh NTM tersebut.

NTM M dapat menghitung sebuah fungsi jika ada jalur perhitungan yang menghasilkan nilai tertentu dan setiap jalur penerimaan menghasilkan nilai yang sama. Teorema Church-Turing menyatakan bahwa DTM cukup kuat untuk mensimulasikan NTM, meskipun membutuhkan lebih banyak sumber daya. Teorema menunjukkan bahwa semua bahasa yang diterima oleh NTM adalah rekursif *enumerable*, dan semua fungsi yang dihitung oleh NTM adalah rekursif parsial. Simulasi NTM oleh DTM melibatkan penggunaan pita tambahan untuk menyimpan informasi simulasi dan menghapus sisa dari tahap sebelumnya. DTM kemudian menyalin input, menghasilkan string dalam urutan leksikografis, dan mensimulasikan langkah-langkah nondeterministik berdasarkan simbol yang dihasilkan. Algoritma nondeterministik sering dijelaskan sebagai menebak-dan-memverifikasi, ketika langkah-langkah nondeterministik menebak jalur perhitungan, dan subrutin deterministik memverifikasi jalur tersebut.

## 7.4 NP

Kelas kompleksitas $NP$ mencakup masalah yang dapat diverifikasi dalam waktu polinomial oleh sebuah mesin Turing deterministik jika diberikan sebuah "saksi" atau "sertifikat". Secara formal, sebuah bahasa $A$ berada dalam $NP$ jika dan hanya jika ada sebuah bahasa $B$ dalam $P$ dan sebuah fungsi polinomial $p$ sehingga untuk setiap *instance* $x$:

$$x \in A (\exists y, |y| \le p(|x|)) \langle x, y \rangle \in B \tag{7.1}$$

**Bukti Theorem 2.1**

$(\Leftarrow)$ Misalkan $A = L(M)$ di mana $M$ adalah mesin Turing nondeterministik dengan batas waktu polinomial $p(n)$. Kita mendefinisikan sebuah mesin Turing deterministik $M^*$ yang menerima pasangan $\langle x, y \rangle$. $M^*$ mensimulasikan $M$ pada input $x$ menggunakan langkah-langkah yang ditentukan oleh $y$, dan menerima jika simulasi menerima dalam waktu polinomial. Maka, $x \in L(M)$ jika dan hanya jika ada $y$ sedemikian sehingga $\langle x, y \rangle \in L(M^*)$.

$(\Rightarrow)$ Jika $A$ dapat direpresentasikan oleh set $B$ dan fungsi $p$ seperti dalam (2.1), maka kita bisa mendesain mesin Turing nondeterministik $M$ untuk $A$ yang pertama membuat $p(|x|)$ gerakan nondeterministik untuk menulis *string* $y$, dan kemudian memverifikasi bahwa $\langle x, y \rangle \in B$ menggunakan mesin Turing deterministik untuk $B$.

**Contoh-Contoh Masalah dalam NP**

1.  **SATISFIABILITY (SAT):**
    *   Definisi: Diberikan sebuah formula Boolean $\phi$, tentukan apakah ada penugasan nilai variabel yang membuat $\phi$ bernilai benar.
    *   Verifikasi: Untuk setiap formula Boolean $\phi$ dengan panjang $n$ dan penugasan $\tau$ pada variabel-variabelnya, kita bisa memeriksa deterministik dalam waktu $O(n^2)$ apakah $\tau$ memuaskan $\phi$. Oleh karena itu, SAT berada dalam $NP$ karena kita bisa menebak penugasan $\tau$ dan memeriksanya dalam waktu polinomial.

2.  **HAMILTONIAN CIRCUIT (HC):**
    *   Definisi: Diberikan sebuah graf $G$, tentukan apakah ada sebuah siklus yang melewati setiap simpul tepat satu kali.
    *   Verifikasi: Kita bisa menebak urutan simpul dan memverifikasi apakah urutan tersebut membentuk siklus Hamiltonian dalam waktu polinomial. Verifikasi mencakup memeriksa apakah urutan tersebut adalah permutasi dari semua simpul dan apakah setiap pasangan simpul yang bertetangga dalam urutan tersebut terhubung oleh sebuah tepi dalam graf.

3.  **INTEGER PROGRAMMING (IP):**
    *   Definisi: Diberikan sebuah matriks bilangan bulat $A$ dan sebuah vektor $b$, tentukan apakah ada sebuah vektor bilangan bulat $x$ sehingga $Ax \ge b$.
    *   Verifikasi: Kita bisa menebak sebuah solusi bilangan bulat $x$ dan memverifikasi ketidaksamaan $Ax \ge b$ dalam waktu polinomial. Namun, kita juga harus memastikan bahwa jika masalah memiliki solusi, maka ada solusi yang panjangnya dibatasi secara polinomial oleh ukuran input.

**Lema Penting untuk IP**
a.  Lema 2.5: Jika $B$ adalah submatriks persegi dari $A$, maka $|\det B| \le (aq)^q$, di mana $a$ nilai absolut maksimum elemen dalam $A$ dan $q$ adalah ordo dari $B$.
b.  Lema 2.6: Jika *rank* dari $A$ adalah $r < m$, maka ada vektor nonzero $z$ sehingga $Az = 0$ dan nilai absolut dari setiap komponen $z$ tidak melebihi $(aq)^q$.
c.  Lema 2.7: Jika $Ax \ge b$ memiliki solusi bilangan bulat, maka ada solusi bilangan bulat $x'$ yang elemen-elemennya adalah bilangan bulat dengan panjang paling banyak $2(aq)^{2q+2}$.

Lema-lema ini menunjukkan bahwa solusi bilangan bulat untuk IP memiliki panjang yang dibatasi secara polinomial oleh ukuran input. Hal ini memastikan bahwa masalah IP berada dalam NP karena bisa ditebak solusi bilangan bulat yang dibatasi dan memverifikasi ketidaksamaan dalam waktu polinomial.

## 7.5 Teorema Cook

Konsep *reducibility* pertama kali dikembangkan dalam teori rekursi. *Reducibility* ($\le r$) adalah relasi biner pada bahasa yang memenuhi sifat refleksivitas dan transitivitas, sehingga membentuk urutan parsial pada kelas semua bahasa. *Reducibility many-one* dalam waktu polinomial ($\le Pm$) diperkenalkan untuk mengukur kesulitan masalah dalam kelas kompleksitas tertentu. Jika $A \le Pm B$, artinya ada fungsi komputabel $f$ yang memetakan setiap elemen dalam $A$ ke $B$ dalam waktu polinomial. *Reducibility* ini memenuhi sifat refleksivitas dan transitivitas.

Himpunan kompleksitas seperti P, NP, dan PSPACE tertutup di bawah $\le Pm$. Artinya, jika $A \le Pm B$ dan $B \in C$, maka $A$ juga termasuk dalam $C$. Sebuah himpunan $B$ disebut $\le Pm$-hard untuk kelas $C$ jika setiap himpunan dalam $C$ dapat direduksi ke $B$, dan disebut $\le Pm$-complete jika $B$ juga termasuk dalam $C$. Himpunan NP-complete adalah masalah paling sulit dalam NP yang tidak termasuk dalam P jika dan hanya jika $P \neq NP$.

Masalah *halting* terbatas waktu (BHP) adalah masalah pertama yang terbukti NP-complete. BHP menentukan apakah mesin Turing nondeterministik (NTM) menerima masukan dalam sejumlah langkah tertentu. Teorema Cook menyatakan bahwa masalah *satisfiability* (SAT) adalah NP-complete. Buktinya menunjukkan bahwa setiap masalah $A$ dalam NP dapat direduksi ke SAT dalam waktu polinomial. Dengan kata lain, jika SAT dapat diselesaikan dalam waktu polinomial, maka setiap masalah dalam NP juga dapat diselesaikan dalam waktu polinomial, yang berarti $P = NP$.

Proses pembuktian Teorema Cook melibatkan konstruksi formula Boolean Fx dari variabel yang mewakili konfigurasi NTM. Fx dibangun sedemikian rupa sehingga Fx satisfiable jika dan hanya jika NTM menerima input dalam waktu polinomial. Pembuktian ini menunjukkan bahwa SAT adalah masalah yang sangat penting dan sulit dalam teori kompleksitas. Keberadaan algoritma polinomial untuk SAT akan menyelesaikan semua masalah dalam NP dengan waktu polinomial.

Untuk mempermudah pembuktian masalah NP-complete lainnya, SAT dapat diubah menjadi bentuk normal konjungtif (CNF) atau 3-CNF, yang terdiri dari klausa dengan tiga literal. 3-SAT, yaitu SAT dalam bentuk 3-CNF, juga terbukti NP-complete, yang mempermudah reduksi dari masalah lain ke 3-SAT.

## 7.6 Masalah NP-Lengkap Lainnya

Konsep NP-kelengkapan sangat penting dengan ribuan masalah NP-lengkap yang ada dalam berbagai bidang seperti ilmu komputer, matematika diskrit, dan penelitian operasi. Secara teoritis, semua masalah ini bisa dibuktikan NP-lengkap dengan mereduksi SAT ke masalah tersebut. Namun, secara praktis lebih mudah membuktikan masalah lain yang memiliki struktur serupa.

Dalam bagian ini, dipelajari beberapa masalah NP-lengkap terkenal:

1.  **Vertex Cover (VC):** Diberikan graf $G = (V,E)$ dan bilangan bulat $K \ge 0$, tentukan apakah $G$ memiliki *vertex cover* dengan ukuran paling banyak $K$.
    *   Teorema 2.14: VC adalah NP-lengkap.
    *   Bukti: VC berada dalam NP dan dapat dibuktikan NP-lengkap dengan mereduksi 3-SAT ke VC.

2.  **Clique:** Diberikan graf $G = (V,E)$ dan bilangan bulat $K \ge 0$, tentukan apakah $G$ memiliki klik dengan ukuran $K$.
    **Independent Set (IS):** Diberikan graf $G = (V,E)$ dan bilangan bulat $K \ge 0$, tentukan apakah $G$ memiliki himpunan independen dengan ukuran $K$.
    *   Teorema 2.15: Clique dan IS adalah NP-lengkap.
    *   Bukti: IS adalah NP-lengkap karena dapat direduksi dari VC, dan Clique adalah NP-lengkap karena dapat direduksi dari IS.

3.  **Hamiltonian Circuit (HC):**
    *   Teorema 2.16: HC adalah NP-lengkap.
    *   Bukti: HC adalah NP-lengkap dengan mereduksi 3-SAT ke HC.

4.  **Integer Programming (IP):**
    *   Teorema 2.17: IP adalah NP-lengkap.
    *   Bukti: IP adalah NP-lengkap dengan mereduksi 3-SAT ke IP.

5.  **Subset Sum (SS):** Diberikan daftar bilangan bulat $L = (a1, a2, \dots, an)$ dan bilangan bulat $S$, tentukan apakah ada *subset* $I \subseteq \{1, 2, \dots, n\}$ sehingga $\sum_{i \in I} a_i = S$.
    *   Teorema 2.18: SS adalah NP-lengkap.
    *   Bukti: SS adalah NP-lengkap dengan mereduksi 3-SAT ke SS.

Bukti dan penjelasan masing-masing teorema melibatkan reduksi dari masalah NP-lengkap lain seperti 3-SAT dan menunjukkan bahwa keputusan untuk masalah-masalah tersebut dapat dilakukan dalam waktu polinomial.

## 7.7 Polinomial-Time Turing Reducibility

Bagian ini membahas reduksi Turing dalam waktu polinomial, bentuk reduksi yang lebih lemah dibandingkan dengan reduksi *polinomial-time many-one*. Reduksi Turing dalam waktu polinomial juga mempertahankan keanggotaan dalam kelas $P$, dan dapat diterapkan pada masalah pencarian (fungsi).

Secara intuitif, sebuah masalah $A$ dikatakan *Turing-reducible* ke masalah $B$ jika ada algoritma $M$ untuk $A$ yang dapat mengajukan beberapa pertanyaan mengenai keanggotaan di set $B$ selama komputasi. Jika total waktu yang digunakan oleh $M$ pada *input* $x$, tanpa waktu *querying*, dibatasi oleh polinomial $p(|x|)$, dan panjang setiap *query* yang diajukan juga dibatasi oleh $p(|x|)$, maka $A$ dikatakan *Turing-reducible* dalam waktu polinomial ke $B$ dan dilambangkan dengan $A \le_p^T B$.

Contoh-contoh yang menggambarkan konsep ini adalah:

1.  Untuk setiap set $A, A \le_p^T A - complement$. Ini dilakukan dengan mengajukan *oracle* $A$ apakah input $x$ ada dalam $A$ atau tidak, lalu membalikkan jawabannya. Ini menunjukkan bahwa reduksi $\le_p^T$ bisa lebih lemah dibandingkan dengan reduksi $\le_p^M$ jika $NP \neq coNP$.
2.  Untuk masalah NP-complete CLIQUE, kita mendefinisikan variasi masalahnya sebagai EXACT-CLIQUE, yang mengharuskan menentukan apakah apakah ukuran maksimum klik dari graf G adalah K. Meskipun tidak jelas apakah EXACT-CLIQUE termasuk NP, kita menunjukkan bahwa CLIQUE dan EXACT-CLIQUE saling *Turing-reducible* dalam waktu polinomial, sehingga keduanya termasuk dalam $P$ atau tidak termasuk $P$.
3.  MAX-CLIQUE adalah *Turing-reducible* dalam waktu polinomial ke CLIQUE dan sebaliknya. Algoritma yang dijelaskan untuk MAX-CLIQUE menunjukkan bagaimana masalah ini dapat diselesaikan menggunakan CLIQUE dengan menggunakan melakukan *query* terhadap berbagai ukuran $K$.

Bagian ini juga mendefinisikan mesin Turing *oracle* sebagai model formal untuk reduksi Turing dalam waktu polinomial. Mesin Turing *oracle* memiliki *tape query* tambahan dan dua status tambahan untuk menangani *query* ke fungsi atau set *oracle*. Proses komputasi dilakukan dengan memasukkan *string query* ke *tape query*, dan *oracle* menggantikan string tersebut dengan hasilnya.

Definisi reduksi Turing dalam waktu polinomial adalah sebagai berikut:

1.  Sebuah fungsi parsial $f$ Turing-reducible ke fungsi total $g$ jika ada mesin Turing *oracle* $M$ yang menghitung fungsi $f$ menggunakan *oracle* $g$.
2.  Sebuah set $A$ Turing-reducible ke set $B$ jika karakteristik fungsi $\chi A$ Turing-reducible ke $\chi B$.

Proposisi-proposisi yang dibahas menunjukkan bahwa reduksi $\le_p^T$ bersifat refleksif dan transitif, serta lebih lemah dibandingkan reduksi $\le_p^M$. Selain itu, kelas $P$ dan $PSPACE$ tertutup di bawah $\le_p^T$, dan $NP = coNP$ jika dan hanya jika $NP$ tertutup di bawah $\le_p^T$.

Terakhir, bagian ini mencatat bahwa kompleksitas ruang mesin Turing *oracle* lebih sulit didefinisikan dibandingkan kompleksitas waktu, terutama dalam hal penggunaan *tape query*. Pengukuran kompleksitas ruang diatur dengan cara alternatif di mana *tape query* dianggap sebagai *tape* yang hanya dapat ditulis dan tidak dihitung dalam pengukuran ruang.

## 7.8 Masalah Optimasi NP-Complete

Masalah optimasi kombinatorial sering kali merupakan masalah pencarian NP-hard. Untuk membuktikan *NP-hardness*, kita menunjukkan bahwa masalah keputusan terkait adalah $\le P$ m-complete untuk NP, dan kemudian membuktikan bahwa pencarian solusi optimal untuk masalah tersebut adalah $\le P$ *T-equivalent* dengan masalah keputusan terkait. Namun, dalam banyak aplikasi, solusi mendekati optimal sudah memadai. *NP-hardness* dari masalah optimasi tidak selalu berarti bahwa aproksimasi masalah tersebut juga *NP-hard*. Kami menunjukkan bahwa untuk beberapa masalah optimasi *NP-complete*, versi aproksimasinya juga *NP-hard*, sementara untuk masalah lainnya, aproksimasi waktu-*polynomial* mungkin dicapai.

Diperkenalkan kerangka umum untuk menangani masalah aproksimasi. Biasanya, masalah optimasi $\Pi$ memiliki struktur umum yaitu untuk setiap instance input $x$, terdapat sejumlah solusi $y$ untuk $x$. Tujuan dari masalah $\Pi$ adalah menemukan solusi $y$ sehingga nilai $v(y)$ dimaksimalkan (atau diminimalkan). Sebagai contoh, masalah MAX-CLIQUE cocok dengan kerangka ini yaitu input adalah graf G, solusi adalah klik C dalam G, nilai $v(C)$ adalah jumlah *vertex* dalam C, dan masalahnya adalah menemukan klik terbesar dalam graf G.

Untuk masalah maksimalisasi $\Pi$ dengan rasio aproksimasi $r > 1$, versi aproksimasi dari $\Pi$ didefinisikan sebagai $r$-APPROX-$\Pi$, yaitu, menemukan solusi $y$ dengan $v(y) = \frac{v^*(x)}{r}$, dengan $v^*(x)$ adalah nilai maksimum dari semua solusi untuk $x$. Untuk masalah minimisasi $\Pi$, versi aproksimasinya dengan rasio $r$ adalah menemukan solusi $y$ dengan $v(y) \le r \cdot v^*(x)$.

Kemudian dipertimbangkan masalah optimasi terkenal: *Traveling Salesman Problem* (TSP), yang meminta tur terpendek dari $n$ kota pada peta. Versi keputusan dari TSP adalah menentukan apakah ada tur (*Hamiltonian circuit*) dengan biaya total kurang dari atau sama dengan $K$. TSP terbukti *NP-complete* dengan reduksi sederhana dari HC (*Hamiltonian Circuit*) ke TSP. Selain itu, masalah pencarian TSP juga *NP-hard*, dan aproksimasi untuk TSP juga *NP-hard* untuk setiap rasio $r > 1$.

TSP yang memenuhi ketentuan ketidaksamaan segitiga (*triangle inequality*) juga *NP-complete*. Namun, untuk versi aproksimasi TSP yang memenuhi ketidaksamaan segitiga, algoritma aproksimasi polinomial ada untuk semua rasio $r \ge \frac{3}{2}$. Apakah rasio ini bisa ditingkatkan masih merupakan pertanyaan terbuka. Versi Euclidean TSP pada bidang dua dimensi juga memiliki algoritma aproksimasi polinomial untuk semua rasio $r > 1$.

Selanjutnya, diperkenalkan hasil bahwa aproksimasi $\frac{3}{2}$ APPROX-TSP pada graf yang memenuhi ketidaksamaan segitiga dapat diselesaikan dalam waktu polinomial. Digunakan teknik yang melibatkan pohon merentang minimum (*Minimum Spanning Tree*, MST) dan pencocokan minimum (*Minimum Matching*) untuk mencapai hasil ini.

Terakhir, dikaji masalah *knapsack*, yang merupakan contoh masalah dengan FPTAS (*Fully Polynomial-Time Approximation Scheme*). Untuk masalah *knapsack*, algoritma aproksimasi FPTAS dapat memberikan solusi mendekati optimal dalam waktu polinomial terhadap ukuran input $n$ dan parameter $W$.

1.  **Kompleksitas Komputasi**
    *   Kompleksitas Waktu dan Kompleksitas Ruang adalah metrik utama untuk mengevaluasi efisiensi algoritma. Kompleksitas waktu mengukur berapa lama algoritma bekerja sebagai fungsi dari ukuran input, sedangkan kompleksitas ruang mengukur seberapa banyak memori yang digunakan.
    *   Algoritma yang efisien penting untuk aplikasi nyata di mana sumber daya terbatas, seperti dalam sistem *real-time*.

2.  **Mesin Turing Deterministik (DTM)**
    *   DTM memiliki unit kontrol, unit memori, dan kepala baca/tulis yang dapat bergerak di sepanjang pita memori. Operasinya meliputi membaca, menulis, dan memindahkan kepala pita.
    *   Definisi Formal dan Fungsi Transisi DTM menyediakan cara untuk formal dan sistematik dalam menentukan hasil dari algoritma berdasarkan konfigurasi dan transisi.

3.  **Mesin Turing Nondeterministik (NTM)**
    *   Berbeda dengan DTM, NTM dapat memiliki beberapa langkah transisi dari satu konfigurasi, memungkinkan lebih banyak kemungkinan konfigurasi berikutnya.
    *   Teorema Church-Turing menunjukkan bahwa DTM dapat mensimulasikan NTM dengan lebih banyak sumber daya, mengarah pada pemahaman bahwa NTM dan DTM memiliki kekuatan komputasi yang setara dalam hal fungsi yang dapat dihitung.

4.  **Kompleksitas Kelas NP**
    *   NP (*Nondeterministic Polynomial time*) mencakup masalah yang solusinya dapat diverifikasi dalam waktu polinomial. Masalah ini dapat memiliki sertifikat verifikasi yang memudahkan proses verifikasi solusi.
    *   Masalah NP-Complete seperti SAT, HC, IP, dan SS merupakan masalah yang paling sulit dalam NP. Teorema Cook mengindikasikan bahwa setiap masalah dalam NP dapat direduksi ke SAT dalam waktu polinomial.
    *   Teorema menunjukkan bahwa masalah-masalah ini, jika dapat diselesaikan dalam waktu polinomial, berarti P = NP.

5.  **Reduksi Turing dalam Waktu Polinomial**
    *   Reduksi Turing dalam waktu polinomial ($\le_p^T$) mengukur kekuatan relatif masalah berdasarkan kemampuan mereka untuk saling mengajukan pertanyaan tentang keanggotaan dalam waktu polinomial.
    *   Contoh termasuk pengurangan antara CLIQUE dan MAX-CLIQUE serta masalah TSP. Reduksi ini membantu dalam memahami hubungan antar masalah dalam hal kompleksitas mereka.

6.  **Masalah Optimasi NP-Complete**
    *   Masalah optimasi sering kali adalah NP-hard, dan membuktikan NP-hardness melibatkan pengurangan dari masalah keputusan terkait.
    *   Contoh masalah optimasi termasuk *Traveling Salesman Problem* (TSP) dan Knapsack, ketika TSP memiliki solusi aproksimasi dengan rasio tertentu, sementara Knapsack dapat diselesaikan dengan algoritma FPTAS.

***