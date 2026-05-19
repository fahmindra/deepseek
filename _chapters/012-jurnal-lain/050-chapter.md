---
slug: ch-12-05
title: "RoPE"
layout: chapter
---

## Abstrak (*Abstract*)

Pengkodean posisi (*Position encoding*) akhir-akhir ini telah terbukti sangat efektif di dalam arsitektur Transformer. Pengkodean ini memberikan supervisi (panduan) yang sangat berharga untuk memodelkan ketergantungan (*dependency modeling*) antara elemen-elemen yang berada pada posisi berbeda di dalam suatu urutan data (sekuens). 

Di dalam makalah ini, para peneliti pertama-tama menginvestigasi berbagai metode yang ada untuk mengintegrasikan informasi posisional ke dalam proses pembelajaran dari model bahasa berbasis Transformer. Kemudian, mereka mengusulkan sebuah metode baru yang diberi nama **Rotary Position Embedding (RoPE)** untuk memanfaatkan informasi posisional secara efektif. 

Secara spesifik, RoPE yang diusulkan ini bekerja dengan cara **mengkodekan posisi absolut menggunakan sebuah matriks rotasi**, dan pada saat yang bersamaan, secara langsung **menggabungkan ketergantungan posisi relatif secara eksplisit ke dalam formulasi *self-attention***. 

Patut dicatat, RoPE memungkinkan beberapa properti (sifat) yang sangat bernilai, di antaranya:
1.  Fleksibilitas terhadap panjang urutan (sekuens) teks.
2.  Adanya peluruhan (*decaying*) pada ketergantungan antar-token seiring dengan meningkatnya jarak relatif antar token tersebut.
3.  Kemampuan untuk melengkapi arsitektur *linear self-attention* dengan pengkodean posisi relatif. 

Sebagai penutup, para peneliti mengevaluasi arsitektur Transformer yang telah ditingkatkan kinerjanya dengan *rotary position embedding* ini—yang selanjutnya disebut sebagai **RoFormer**—pada berbagai dataset tolok ukur (*benchmark*) klasifikasi teks panjang. 

Eksperimen yang dilakukan menunjukkan bahwa RoFormer secara konsisten berhasil mengungguli arsitektur-arsitektur alternatif lainnya. Lebih jauh lagi, para peneliti memberikan analisis teoretis untuk menjelaskan beberapa hasil eksperimen tersebut. RoFormer saat ini telah diintegrasikan ke dalam ekosistem Huggingface, yang dapat diakses di: `https://huggingface.co/docs/transformers/model_doc/roformer`.

---

## 1. Pendahuluan (*Introduction*)

Urutan sekuensial (susunan berurutan) dari kata-kata memiliki nilai yang sangat krusial bagi pemahaman bahasa alami (*natural language understanding*). 
*   Model yang berbasis Jaringan Saraf Berulang (*Recurrent Neural Networks* / RNNs) mengkodekan urutan token-token dengan cara menghitung representasi status tersembunyi (*hidden state*) secara rekursif di sepanjang dimensi waktu. 
*   Di sisi lain, model yang berbasis Jaringan Saraf Konvolusional (*Convolutional Neural Networks* / CNNs) pada umumnya dianggap tidak menyadari posisi (*position-agnostic*), namun penelitian terbaru telah menunjukkan bahwa operasi *padding* (penambahan nilai nol di ujung data) yang umum digunakan ternyata dapat secara implisit mempelajari informasi posisi. 

Baru-baru ini, Model Bahasa Pra-latih (*Pre-trained Language Models* / PLMs), yang dibangun di atas arsitektur Transformer, telah berhasil mencapai performa paling mutakhir (*state-of-the-art*) di berbagai tugas pemrosesan bahasa alami (NLP). Pencapaian ini mencakup pembelajaran representasi konteks, penerjemahan mesin, dan pemodelan bahasa, dan masih banyak lagi. 

Tidak seperti model berbasis RNN dan CNN, PLMs memanfaatkan mekanisme **Perhatian Mandiri (*Self-Attention*)** untuk secara semantik menangkap representasi kontekstual dari suatu korpus yang diberikan. Sebagai konsekuensinya, PLMs mencapai peningkatan yang sangat signifikan dalam hal paralelisasi komputasi dibandingkan RNN, sekaligus meningkatkan kemampuan pemodelan hubungan (relasi) intra-token yang lebih panjang jika dibandingkan dengan CNN (dengan asumsi CNN lapisan tunggal).

Patut dicatat bahwa arsitektur *self-attention* murni dari PLMs saat ini telah terbukti bersifat **tidak menyadari posisi (*position-agnostic*)**. Menindaklanjuti klaim ini, berbagai pendekatan telah diusulkan untuk menyandikan (meng-*encode*) informasi posisi ke dalam proses pembelajaran. 

**Kubu Pengkodean Posisi Absolut:**
Di satu sisi, ada pendekatan yang membuat pengkodean posisi absolut melalui sebuah fungsi yang sudah didefinisikan sebelumnya (seperti fungsi Sinus/Kosinus pada Transformer asli), yang kemudian ditambahkan (dijumlahkan) ke representasi kontekstual. Pendekatan lain menggunakan pengkodean posisi absolut yang dapat dipelajari mesin (dilatih/*trainable*). 

**Kubu Pengkodean Posisi Relatif:**
Di sisi lain, penelitian-penelitian sebelumnya lebih berfokus pada pengkodean posisi relatif (*relative position encoding*), yang secara tipikal mengkodekan informasi posisi relatif langsung ke dalam mekanisme perhatian (*attention mechanism*) itu sendiri. 

Sebagai tambahan dari pendekatan-pendekatan tersebut, ada penulis yang mengusulkan untuk memodelkan ketergantungan pengkodean posisi dari sudut pandang *Neural ODE* (Persamaan Diferensial Biasa), serta ada juga penulis yang mengusulkan untuk memodelkan informasi posisi di dalam ruang bilangan kompleks (*complex space*). Terlepas dari keefektifan pendekatan-pendekatan tersebut, mereka umumnya bekerja dengan cara **menambahkan (menjumlahkan)** informasi posisi ke dalam representasi konteks. Metode penjumlahan ini sayangnya membuat metode-metode tersebut menjadi **tidak cocok (tidak *suitable*) untuk arsitektur *linear self-attention***.

Di dalam makalah ini, penulis memperkenalkan metode baru, yaitu **Rotary Position Embedding (RoPE)**, untuk memanfaatkan informasi posisional di dalam proses pembelajaran PLMS. Secara spesifik, RoPE mengkodekan posisi absolut menggunakan **matriks rotasi** dan secara bersamaan memadukan ketergantungan posisi relatif eksplisit ke dalam perumusan *self-attention*. 

Perlu dicatat bahwa RoPE yang diusulkan ini lebih diprioritaskan ketimbang metode-metode yang ada karena memiliki properti yang berharga, termasuk: fleksibilitas panjang sekuens, ketergantungan antar-token yang meluruh (*decaying*) seiring dengan bertambahnya jarak relatif, dan kemampuannya untuk diterapkan pada *linear self-attention* bersama dengan pengkodean posisi relatif. Hasil eksperimen pada berbagai dataset *benchmark* klasifikasi teks panjang menunjukkan bahwa Transformer yang diperkuat dengan *rotary position embedding*, dinamakan **RoFormer**, mampu memberikan performa yang lebih baik dibandingkan alternatif *baseline*, sehingga mendemonstrasikan kemanjuran dari RoPE.

Secara ringkas, kontribusi dari penelitian ini mencakup tiga hal berikut:
1.  Peneliti menginvestigasi pendekatan yang sudah ada terhadap pengkodean posisi relatif dan menemukan bahwa sebagian besar pendekatan tersebut dibangun berdasarkan gagasan mengenai dekomposisi dari "penambahan/penjumlahan pengkodean posisi" ke dalam representasi konteks. Peneliti kemudian memperkenalkan metode baru, **Rotary Position Embedding (RoPE)**, untuk memanfaatkan informasi posisional di dalam proses pembelajaran PLMS. Ide kuncinya adalah untuk mengkodekan posisi relatif dengan cara **mengalikan representasi konteks dengan sebuah matriks rotasi** yang memiliki interpretasi teoretis yang sangat jelas.
2.  Peneliti mempelajari sifat-sifat (properti) dari RoPE dan menunjukkan bahwa nilai atensinya akan meluruh (*decays*) ketika jarak relatif antar kata meningkat, yang mana ini adalah sifat yang sangat diinginkan untuk pengkodean bahasa alami. Peneliti juga berargumen bahwa pendekatan-pendekatan berbasis pengkodean posisi relatif sebelumnya tidaklah kompatibel (tidak dapat digunakan) bersama arsitektur *linear self-attention*.
3.  Peneliti mengevaluasi RoFormer pada berbagai dataset *benchmark* teks panjang. Eksperimen menunjukkan bahwa model ini secara konsisten mencapai performa yang lebih baik dibandingkan alternatifnya. Eksperimen dengan model bahasa pra-latih juga tersedia di GitHub: `https://github.com/ZhuiyiTechnology/roformer`.

Sisa dari makalah ini disusun sebagai berikut: Peneliti membangun deskripsi formal mengenai masalah pengkodean posisi dalam arsitektur *self-attention* dan meninjau ulang karya-karya sebelumnya di Bagian (2). Kemudian, peneliti mendeskripsikan *rotary position encoding* (RoPE) dan mempelajari propertinya di Bagian (3). Laporan eksperimen ada di Bagian (4). Terakhir, kesimpulan makalah ini ditarik pada Bagian (5).

---

## 2. Latar Belakang dan Pekerjaan Terkait (*Background and Related Work*)

### 2.1 Konsep Awal (*Preliminary*)

Mari kita definisikan $\mathbb{S}_N = \{w_i\}_{i=1}^N$ sebagai sebuah urutan (sekuens) yang terdiri dari $N$ token masukan, di mana $w_i$ adalah elemen ke-$i$. Vektor *word embedding* (penyematan kata) yang berkorespondensi untuk urutan $\mathbb{S}_N$ tersebut dilambangkan sebagai $\mathbb{E}_N = \{\boldsymbol{x}_i\}_{i=1}^N$, di mana $\boldsymbol{x}_i \in \mathbb{R}^d$ adalah vektor *word embedding* berdimensi-$d$ dari token $w_i$ **tanpa menyertakan informasi posisi**. 

Mekanisme *self-attention* pertama-tama akan menggabungkan informasi posisi ke dalam *word embeddings* tersebut, lalu mentransformasikannya menjadi representasi kueri (*queries*), kunci (*keys*), dan nilai (*values*).

$$ \boldsymbol{q}_m = f_q(\boldsymbol{x}_m, m) $$
$$ \boldsymbol{k}_n = f_k(\boldsymbol{x}_n, n) \quad \quad \text{(1)} $$
$$ \boldsymbol{v}_n = f_v(\boldsymbol{x}_n, n), $$

**Penjelasan Rumus (1):**
*   $\boldsymbol{x}_m$ dan $\boldsymbol{x}_n$: Vektor *word embedding* (makna kata) mentah untuk token pada posisi ke-$m$ dan ke-$n$.
*   $m$ dan $n$: Indeks posisi numerik absolut dari token tersebut di dalam kalimat (misal: kata ke-1, kata ke-5).
*   $f_q, f_k, f_v$: Fungsi matematis yang bertugas menyuntikkan (menggabungkan) informasi posisi $m$ atau $n$ ke dalam vektor kata $\boldsymbol{x}$, lalu memproyeksikannya menjadi matriks **Query ($\boldsymbol{q}$)**, **Key ($\boldsymbol{k}$)**, dan **Value ($\boldsymbol{v}$)**.
*   $\boldsymbol{q}_m$: Vektor *Query* untuk kata di posisi ke-$m$. Ini mewakili "apa yang sedang dicari" oleh kata tersebut.
*   $\boldsymbol{k}_n$: Vektor *Key* untuk kata di posisi ke-$n$. Ini mewakili "identitas" dari kata tersebut untuk dicocokkan dengan kueri.
*   $\boldsymbol{v}_n$: Vektor *Value* untuk kata di posisi ke-$n$. Ini adalah informasi aktual yang akan diambil jika kueri dan kunci cocok.

Nilai kueri dan kunci ($\boldsymbol{q}_m$ dan $\boldsymbol{k}_n$) tersebut kemudian digunakan untuk menghitung bobot perhatian (*attention weights*), sementara keluaran akhir (*output*) dihitung sebagai penjumlahan berbobot (*weighted sum*) terhadap nilai representasi (value).

$$ a_{m,n} = \frac{\exp\left(\frac{\boldsymbol{q}_m^\top \boldsymbol{k}_n}{\sqrt{d}}\right)}{\sum_{j=1}^N \exp\left(\frac{\boldsymbol{q}_m^\top \boldsymbol{k}_j}{\sqrt{d}}\right)} $$
$$ \boldsymbol{o}_m = \sum_{n=1}^N a_{m,n} \boldsymbol{v}_n \quad \quad \text{(2)} $$

**Penjelasan Rumus (2):**
*   **$\boldsymbol{q}_m^\top \boldsymbol{k}_n$**: Ini adalah operasi produk titik (*dot product*) atau *inner product* antara Kueri di posisi $m$ dan Kunci di posisi $n$. Hasilnya adalah sebuah angka skalar yang merepresentasikan seberapa kuat "keterkaitan/kecocokan" antara kata di posisi $m$ dengan kata di posisi $n$.
*   **$\sqrt{d}$**: Faktor penskalaan (*scaling factor*) menggunakan akar kuadrat dari dimensi vektor. Gunanya untuk mencegah nilai *dot product* menjadi terlalu besar yang bisa merusak gradien saat pelatihan.
*   **$\exp(\dots)$**: Fungsi eksponensial (bagian dari proses *Softmax*). Mengubah nilai kecocokan menjadi angka positif yang melonjak tajam.
*   **Pembagi $\sum_{j=1}^N \dots$**: Ini adalah bagian normalisasi dari *Softmax*. Memastikan bahwa total semua bobot atensi untuk satu kueri $m$ terhadap semua kata lain $j$ jika dijumlahkan akan bernilai persis 1.
*   **$a_{m,n}$**: Bobot perhatian (*attention weight*). Ini adalah nilai probabilitas (0 hingga 1) yang menentukan seberapa besar kata ke-$m$ harus "memperhatikan" kata ke-$n$.
*   **$\boldsymbol{o}_m$**: Keluaran (*Output*) akhir dari mekanisme *self-attention* untuk posisi ke-$m$.
*   **$\sum_{n=1}^N a_{m,n} \boldsymbol{v}_n$**: Penjumlahan vektor *Value* ($\boldsymbol{v}_n$) yang dikalikan dengan bobot perhatiannya ($a_{m,n}$). Semakin tinggi kecocokannya, semakin banyak informasi dari kata $n$ yang diserap ke dalam keluaran kata $m$.

Pendekatan-pendekatan yang ada saat ini untuk pengkodean posisi berbasis transformer berfokus utama pada bagaimana memilih sebuah fungsi yang cocok untuk membentuk Persamaan (1) di atas.

---

### 2.2 Pengkodean Posisi Absolut (*Absolute position embedding*)

Pilihan yang paling tipikal (klasik) untuk mendefinisikan Persamaan (1) adalah:

$$ f_{t:t\in\{q,k,v\}}(\boldsymbol{x}_i, i) := \boldsymbol{W}_{t:t\in\{q,k,v\}} (\boldsymbol{x}_i + \boldsymbol{p}_i), \quad \quad \text{(3)} $$

**Penjelasan Rumus (3):**
*   $t$: Melambangkan jenis representasi yang sedang dihitung, yaitu kueri ($q$), kunci ($k$), atau nilai ($v$).
*   $\boldsymbol{p}_i \in \mathbb{R}^d$: Vektor posisional (berdimensi $d$) yang hanya bergantung pada posisi absolut indeks $i$ dari token $\boldsymbol{x}_i$. 
*   **$(\boldsymbol{x}_i + \boldsymbol{p}_i)$**: Ini adalah ciri khas metode absolut tradisional (seperti pada BERT atau Transformer asli). Vektor makna kata $\boldsymbol{x}_i$ murni **ditambahkan/dijumlahkan** secara matematis dengan vektor posisi $\boldsymbol{p}_i$.
*   $\boldsymbol{W}_t$: Matriks bobot linear yang bertugas memproyeksikan gabungan kata+posisi tersebut menjadi vektor Query, Key, atau Value yang siap digunakan.

Penelitian terdahulu menggunakan sekumpulan vektor yang dapat dipelajari (*trainable vectors*) $\boldsymbol{p}_i \in \{\boldsymbol{p}_t\}_{t=1}^L$, di mana $L$ adalah panjang maksimal urutan (sekuens). Sementara itu, Vaswani et al. (arsitek Transformer asli) mengusulkan untuk **membangkitkan (men-generate)** $\boldsymbol{p}_i$ secara otomatis menggunakan fungsi sinusoidal (gelombang trigonometri):

$$ \begin{cases} p_{i, 2t} = \sin(k / 10000^{2t/d}) \\ p_{i, 2t+1} = \cos(k / 10000^{2t/d}) \end{cases} \quad \quad \text{(4)} $$

**Penjelasan Rumus (4):**
*   $\boldsymbol{p}_i$: Vektor posisi untuk indeks kata ke-$i$. Vektor ini memiliki dimensi sepanjang $d$ (misal 512 dimensi).
*   $p_{i, 2t}$: Menunjuk pada indeks dimensi genap ($0, 2, 4, \dots$) dari vektor $\boldsymbol{p}_i$. Dimensi genap diisi menggunakan fungsi Sinus ($\sin$).
*   $p_{i, 2t+1}$: Menunjuk pada indeks dimensi ganjil ($1, 3, 5, \dots$) dari vektor $\boldsymbol{p}_i$. Dimensi ganjil diisi menggunakan fungsi Kosinus ($\cos$).
*   $k$: Indeks posisi absolut kata (misal, $k=1$ untuk kata pertama, $k=2$ untuk kata kedua). *(Catatan Dosen: Teks asli menggunakan variabel $k$, meski di konteks ini maksudnya sama dengan $i$ posisi).*
*   $10000^{2t/d}$: Faktor frekuensi gelombang. Semakin tinggi dimensi $t$, gelombangnya akan semakin panjang. Ini membantu model membedakan pola posisi dari skala jarak dekat hingga jarak jauh.

Di bagian selanjutnya, para peneliti akan menunjukkan bahwa **RoPE (usulan mereka) secara esensial berkaitan erat dengan intuisi fungsi sinusoidal ini**. Namun, alih-alih menjumlahkan (menambahkan) posisi tersebut ke dalam representasi konteks, RoPE mengusulkan untuk menggabungkan informasi posisi relatif melalui proses **perkalian (multiplying)** dengan fungsi sinusoidal tersebut (merotasi vektornya).

---

### 2.3 Pengkodean Posisi Relatif (*Relative position embedding*)

Para peneliti dari Shaw et al. mengaplikasikan modifikasi pada Persamaan (1) sebagai berikut:

$$ f_q(\boldsymbol{x}_m) := \boldsymbol{W}_q \boldsymbol{x}_m $$
$$ f_k(\boldsymbol{x}_n, n) := \boldsymbol{W}_k (\boldsymbol{x}_n + \tilde{\boldsymbol{p}}_r^k) \quad \quad \text{(5)} $$
$$ f_v(\boldsymbol{x}_n, n) := \boldsymbol{W}_v (\boldsymbol{x}_n + \tilde{\boldsymbol{p}}_r^v) $$

**Penjelasan Rumus (5):**
*   $\tilde{\boldsymbol{p}}_r^k, \tilde{\boldsymbol{p}}_r^v \in \mathbb{R}^d$: Ini adalah *embedding* (vektor) posisi relatif yang **dilatih (trainable)**. Vektor ini tidak peduli kata $n$ ada di indeks absolut berapa, melainkan hanya peduli **seberapa jauh jaraknya ($r$)** terhadap kata $m$ yang sedang bertanya (kueri).
*   $r = \text{clip}(m - n, r_{\min}, r_{\max})$: Melambangkan jarak relatif antara posisi $m$ dan $n$. Jarak ini di-*clip* (dipotong) pada batas minimum dan maksimum tertentu. Hipotesis mereka adalah bahwa informasi posisi relatif yang presisi tidak lagi berguna setelah melewati jarak tertentu.
*   Perhatikan bahwa $f_q$ di sini tidak lagi ditambahkan vektor posisi apapun. Hanya Kunci ($f_k$) dan Nilai ($f_v$) yang disuntikkan jarak relatif $\tilde{\boldsymbol{p}}_r$ ini.

Dengan tetap mempertahankan bentuk murni dari Persamaan (3), penulis lain (Dai et al., arsitek Transformer-XL) mengusulkan untuk mendekonstruksi (mengurai) operasi produk titik $\boldsymbol{q}_m^\top \boldsymbol{k}_n$ dari Persamaan (2) menjadi empat suku (term) terpisah:

$$ \boldsymbol{q}_m^\top \boldsymbol{k}_n = \boldsymbol{x}_m^\top \boldsymbol{W}_q^\top \boldsymbol{W}_k \boldsymbol{x}_n + \boldsymbol{x}_m^\top \boldsymbol{W}_q^\top \boldsymbol{W}_k \boldsymbol{p}_n + \boldsymbol{p}_m^\top \boldsymbol{W}_q^\top \boldsymbol{W}_k \boldsymbol{x}_n + \boldsymbol{p}_m^\top \boldsymbol{W}_q^\top \boldsymbol{W}_k \boldsymbol{p}_n \quad \quad \text{(6)} $$

Ide kuncinya kemudian adalah: mengganti *embedding* posisi absolut ($\boldsymbol{p}_n$) di suku terkait dengan "kembaran" relasinya yang dikodekan secara sinusoidal, yaitu $\tilde{\boldsymbol{p}}_{m-n}$. Sementara itu, posisi absolut kueri ($\boldsymbol{p}_m$) pada suku ketiga dan keempat diubah menjadi dua vektor laten $\boldsymbol{u}$ dan $\boldsymbol{v}$ yang dapat dilatih dan sama sekali independen dari posisi kuerinya. Lebih jauh, matriks bobot Kunci dipisahkan menjadi $\boldsymbol{W}_k$ (untuk konten kata $\boldsymbol{x}_n$) dan $\widehat{\boldsymbol{W}}_k$ (untuk vektor posisi $\boldsymbol{p}_n$). Hasil akhirnya menjadi:

$$ \boldsymbol{q}_m^\top \boldsymbol{k}_n = \boldsymbol{x}_m^\top \boldsymbol{W}_q^\top \boldsymbol{W}_k \boldsymbol{x}_n + \boldsymbol{x}_m^\top \boldsymbol{W}_q^\top \widehat{\boldsymbol{W}}_k \tilde{\boldsymbol{p}}_{m-n} + \boldsymbol{u}^\top \boldsymbol{W}_q^\top \boldsymbol{W}_k \boldsymbol{x}_n + \boldsymbol{v}^\top \boldsymbol{W}_q^\top \widehat{\boldsymbol{W}}_k \tilde{\boldsymbol{p}}_{m-n} \quad \quad \text{(7)} $$

Patut dicatat bahwa pada metode ini, informasi posisi sama sekali dihapus dari vektor Nilai (Value) dengan menetapkan $f_v(\boldsymbol{x}_j) := \boldsymbol{W}_v \boldsymbol{x}_j$. Pekerjaan-pekerjaan selanjutnya (seperti arsitektur T5 oleh Raffel et al. dan DeBERTa oleh He et al.) mengikuti pengaturan ini dengan hanya mengkodekan informasi posisi relatif *ke dalam bobot atensinya saja* (yakni skor *dot product*), dan tidak ke *Value*. 

Namun, T5 (Raffel et al.) menyederhanakan formulasi ekstrem di atas dengan mengubah Persamaan (6) menjadi sekadar:

$$ \boldsymbol{q}_m^\top \boldsymbol{k}_n = \boldsymbol{x}_m^\top \boldsymbol{W}_q^\top \boldsymbol{W}_k \boldsymbol{x}_n + b_{i,j} \quad \quad \text{(8)} $$

**Penjelasan Rumus (8):**
*   $\boldsymbol{x}_m^\top \boldsymbol{W}_q^\top \boldsymbol{W}_k \boldsymbol{x}_n$: Ini murni *dot product* antara makna kata (Query dan Key) **tanpa ada posisi sama sekali**.
*   $b_{i,j}$: Sebuah nilai *bias* (skalar) tunggal yang dapat dilatih, yang semata-mata merepresentasikan jarak antara posisi $i$ dan $j$. T5 percaya bahwa menghitung interaksi rumit antara posisi dan konten itu tidak perlu; cukup tambahkan satu angka bias jarak saja di skor akhirnya.

Sementara itu, He et al. berpendapat bahwa posisi relatif dari dua token hanya bisa dimodelkan secara penuh jika kita mempertahankan **dua suku tengah** dari Persamaan (6). Sebagai konsekuensinya, posisi absolut $\boldsymbol{p}_m$ dan $\boldsymbol{p}_n$ di dalam rumus itu secara sederhana langsung diganti seluruhnya dengan posisi relatif $\tilde{\boldsymbol{p}}_{m-n}$:

$$ \boldsymbol{q}_m^\top \boldsymbol{k}_n = \boldsymbol{x}_m^\top \boldsymbol{W}_q^\top \boldsymbol{W}_k \boldsymbol{x}_n + \boldsymbol{x}_m^\top \boldsymbol{W}_q^\top \boldsymbol{W}_k \tilde{\boldsymbol{p}}_{m-n} + \tilde{\boldsymbol{p}}_{m-n}^\top \boldsymbol{W}_q^\top \boldsymbol{W}_k \boldsymbol{x}_n \quad \quad \text{(10)} $$

Perbandingan dari keempat varian pengkodean posisi relatif ini telah menunjukkan bahwa varian yang mirip dengan Persamaan (10) adalah yang paling efisien di antara yang lain. 

Secara garis besar, semua pendekatan sebelumnya berusaha memodifikasi Persamaan (6) yang didasarkan pada logika dekomposisi dari "penjumlahan" di Persamaan (3). Artinya, **mereka umumnya menambahkan (add) informasi posisi ke representasi konteks**. 

Sangat berbeda dengan itu semua, pendekatan usulan makalah ini bertujuan untuk menurunkan pengkodean posisi relatif secara langsung dari mekanisme fungsi *inner product* di Persamaan (1) dengan beberapa aturan/batasan. Di bagian selanjutnya, para peneliti akan membuktikan bahwa pendekatan yang mereka turunkan ini **jauh lebih dapat diinterpretasikan secara geometris** karena ia memasukkan informasi posisi relatif melalui **rotasi (perputaran)** dari representasi konteks.

---

## 3. Pendekatan yang Diusulkan (*Proposed Approach*)

Pada bagian ini, kita akan mendiskusikan *Rotary Position Embedding* (RoPE) yang diusulkan. Kita akan merumuskan masalah pengkodean posisi relatif secara matematis terlebih dahulu, kemudian menurunkan formulasi RoPE, dan terakhir menginvestigasi propertinya.

### 3.1 Formulasi (*Formulation*)

Pemodelan bahasa berbasis Transformer pada umumnya memanfaatkan informasi posisi masing-masing token secara individual melalui mekanisme *self-attention*. Seperti yang terlihat pada Persamaan (2), nilai *dot product* $\boldsymbol{q}_m^\top \boldsymbol{k}_n$ adalah jalur utama yang memungkinkan penyampaian pengetahuan (informasi kecocokan) antara dua token di posisi yang berbeda ($m$ dan $n$). 

Agar kita bisa mengintegrasikan informasi **posisi relatif**, kita harus mengharuskan agar nilai *inner product* (produk dalam/ *dot product*) dari Kueri $\boldsymbol{q}_m$ dan Kunci $\boldsymbol{k}_n$ tersebut dirumuskan murni sebagai sebuah fungsi tunggal $g$. Fungsi $g$ ini *hanya* boleh menerima variabel masukan (input) berupa: *word embedding* kueri ($\boldsymbol{x}_m$), *word embedding* kunci ($\boldsymbol{x}_n$), dan **jarak posisi relatif mereka ($m - n$)**. 

Dengan kata lain, kita berharap bahwa nilai *inner product* tersebut menyandikan posisi *hanya dalam wujud hubungan relatifnya saja*. Bentuk fungsinya adalah sebagai berikut:

$$ \langle f_q(\boldsymbol{x}_m, m), f_k(\boldsymbol{x}_n, n) \rangle = g(\boldsymbol{x}_m, \boldsymbol{x}_n, m - n). \quad \quad \text{(11)} $$

**Penjelasan Mendalam Rumus (11):**
*   $\langle \dots, \dots \rangle$: Merupakan notasi matematis untuk *inner product* (atau *dot product*). Ini ekuivalen dengan $\boldsymbol{q}_m^\top \boldsymbol{k}_n$.
*   $f_q(\boldsymbol{x}_m, m)$ dan $f_k(\boldsymbol{x}_n, n)$: Adalah fungsi Kueri (yang menyuntikkan posisi absolut $m$) dan fungsi Kunci (yang menyuntikkan posisi absolut $n$).
*   $g(\boldsymbol{x}_m, \boldsymbol{x}_n, m - n)$: Adalah fungsi target (impian) kita. Kita ingin memastikan bahwa apapun proses injeksi posisi absolut yang dilakukan di fungsi $f_q$ dan $f_k$, ketika hasil vektor keduanya dikalikan (*inner product*), hasilnya harus setara dengan sebuah fungsi $g$ yang **SAMA SEKALI TIDAK TAHU letak absolut $m$ maupun $n$**. Fungsi $g$ ini hanya tahu kata apa $\boldsymbol{x}_m$ itu, kata apa $\boldsymbol{x}_n$ itu, dan **seberapa jauh selisih jarak mereka ($m - n$)**.

Tujuan akhir dari makalah ini adalah **menemukan mekanisme pengkodean apa (mencari bentuk nyata dari fungsi $f_q$ dan $f_k$)** yang bisa membuat persamaan/hubungan di atas bernilai valid secara matematis.

---

### 3.2 Rotary Position Embedding (RoPE)

#### 3.2.1 Kasus 2D (Ruang Dua Dimensi)

Untuk memudahkan pemahaman, mari kita mulai dengan skenario (kasus) yang sangat sederhana di mana dimensi vektor kata kita hanyalah $d = 2$ (sebuah koordinat bidang datar X dan Y). Di bawah pengaturan ini, peneliti memanfaatkan properti geometris dari vektor di bidang 2D dan bentuk bilangan kompleksnya untuk membuktikan (detail pembuktian matematis diturunkan di Bagian 3.4.1 nanti) bahwa **solusi mutlak** untuk rumusan di Persamaan (11) adalah:

$$ f_q(\boldsymbol{x}_m, m) = (\boldsymbol{W}_q \boldsymbol{x}_m) e^{im\theta} $$
$$ f_k(\boldsymbol{x}_n, n) = (\boldsymbol{W}_k \boldsymbol{x}_n) e^{in\theta} \quad \quad \text{(12)} $$
$$ g(\boldsymbol{x}_m, \boldsymbol{x}_n, m - n) = \text{Re}\left[ (\boldsymbol{W}_q \boldsymbol{x}_m)(\boldsymbol{W}_k \boldsymbol{x}_n)^* e^{i(m-n)\theta} \right] $$

**Penjelasan Mendalam Rumus (12) berdasarkan Bilangan Kompleks Euler:**
*   Ini adalah titik kejeniusan dari RoPE. Karena dimensinya 2 (Dua nilai: x dan y), sebuah vektor kata bisa dianggap sebagai **Satu buah Bilangan Kompleks** (di mana komponen pertama adalah bilangan riil (Real), dan komponen kedua adalah bilangan imajiner (Imajiner)).
*   $(\boldsymbol{W}_q \boldsymbol{x}_m)$: Ini adalah vektor Kueri standar tanpa posisi. Ia memiliki panjang (magnitudo) dan sudut awal tertentu di bidang 2D.
*   **$e^{im\theta}$**: Ini adalah **operator rotasi Euler**. Dikalikan dengan fungsi eksponensial kompleks ini TIDAK akan mengubah panjang dari vektor, melainkan **hanya memutar (merotasi) sudut vektor tersebut**.
*   $m\theta$: Seberapa besar vektor Kueri itu diputar? Ia diputar sebesar indeks posisi absolutnya ($m$) dikalikan dengan sebuah konstanta sudut tertentu ($\theta$). Kata di posisi 1 diputar $1\theta$, kata di posisi 2 diputar $2\theta$, dan seterusnya.
*   *Inner product* dari dua bilangan kompleks menghasilkan fungsi $g$. Fungsi ini menggunakan $\text{Re}[\dots]$ (ekstraksi nilai Riil) dari perkalian bilangan kompleks Kueri dan Konjugat Kunci (*conjugate*, ditandai dengan tanda bintang $*$).
*   Kunci Keberhasilan: Karena hukum eksponen, ketika Kueri (diputar sebesar $m\theta$) dikalikan dengan Kunci (diputar sebesar $n\theta$), sudut hasil perkaliannya secara otomatis menjadi operasi pengurangan: **$e^{i(m-n)\theta}$**. *Voila!* Informasi posisi absolut $m$ dan $n$ musnah, dan yang tersisa di dalam formula hanyalah selisih posisi relatifnya ($m-n$).

Kita dapat menulis ulang bentuk $f_{\{q,k\}}$ di atas menggunakan format **Matriks Perkalian (Multiplication Matrix)** alih-alih eksponen kompleks:

$$ f_{\{q,k\}}(\boldsymbol{x}_m, m) = \begin{pmatrix} \cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta \end{pmatrix} \begin{pmatrix} W_{\{q,k\}}^{(11)} & W_{\{q,k\}}^{(12)} \\ W_{\{q,k\}}^{(21)} & W_{\{q,k\}}^{(22)} \end{pmatrix} \begin{pmatrix} x_m^{(1)} \\ x_m^{(2)} \end{pmatrix} \quad \quad \text{(13)} $$

**Penjelasan Rumus Matriks (13):**
*   $(x_m^{(1)}, x_m^{(2)})$ adalah vektor representasi kata $\boldsymbol{x}_m$ di sumbu $X$ dan $Y$ (karena kasus ini 2D).
*   Matriks $W$ di tengah adalah transformasi *dense layer* konvensional Transformer yang melahirkan vektor kueri/kunci standar.
*   **Matriks $2 \times 2$ Paling Kiri:** Ini adalah **Matriks Rotasi 2D** standar dalam aljabar linear. Memperkalikan matriks berisi $\cos$ dan $\sin$ ini ke sebuah vektor berakibat: vektor tersebut akan **berputar** sebesar sudut $m\theta$ derajat berlawanan arah jarum jam, tanpa mengubah panjang garis vektornya sama sekali.

Secara matematis, menginkorporasikan (*incorporating*) *embedding* posisi relatif melalui metode ini sangatlah sederhana: **Cukup putar (*rotate*) vektor *word embedding* yang sudah melewati proses linear (affine-transformed) tersebut sebesar kelipatan sudut berdasarkan indeks posisinya**. Inilah esensi dan intuisi fundamental di balik penamaan "*Rotary Position Embedding*".

---

#### 3.2.2 Bentuk Umum (*General form*)

Untuk menggeneralisasikan hasil temuan kita di ruang 2D menuju vektor berdimensi tinggi $\boldsymbol{x}_i \in \mathbb{R}^d$ (di mana $d$ bernilai genap, seperti 512 atau 1024), kita **membagi ruang berdimensi $d$ tersebut menjadi potongan-potongan subruang (*sub-spaces*) berdimensi 2 (2D) sebanyak $d/2$ buah**. 

Berdasarkan linearitas dari *inner product*, kita bisa merotasi setiap pasang subruang 2D tersebut secara individual, mengubah fungsi $f_{\{q,k\}}$ menjadi matriks rotasi raksasa:

$$ f_{\{q,k\}}(\boldsymbol{x}_m, m) = \boldsymbol{R}_{\Theta,m}^d \boldsymbol{W}_{\{q,k\}} \boldsymbol{x}_m \quad \quad \text{(14)} $$

Di mana matriks rotasinya adalah sebuah matriks blok-diagonal berukuran $d \times d$:

$$ \boldsymbol{R}_{\Theta,m}^d = \begin{pmatrix} 
\cos m\theta_1 & -\sin m\theta_1 & 0 & 0 & \cdots & 0 & 0 \\
\sin m\theta_1 & \cos m\theta_1 & 0 & 0 & \cdots & 0 & 0 \\
0 & 0 & \cos m\theta_2 & -\sin m\theta_2 & \cdots & 0 & 0 \\
0 & 0 & \sin m\theta_2 & \cos m\theta_2 & \cdots & 0 & 0 \\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
0 & 0 & 0 & 0 & \cdots & \cos m\theta_{d/2} & -\sin m\theta_{d/2} \\
0 & 0 & 0 & 0 & \cdots & \sin m\theta_{d/2} & \cos m\theta_{d/2} 
\end{pmatrix} \quad \quad \text{(15)} $$

**Penjelasan Ekstensif Rumus (15) dan (16):**
*   $\boldsymbol{R}_{\Theta,m}^d$: Adalah "Matriks Rotasi" yang berukuran raksasa ($d \times d$). 
*   Matriks ini sangat kosong (banyak nilai 0). Nilai bukan 0 hanya berada di sepanjang area diagonal utamanya. Setiap blok kecil $2 \times 2$ di diagonal tersebut adalah matriks rotasi 2D persis seperti di Rumus (13).
*   Artinya, vektor Kueri (yang memiliki misal 512 dimensi) dipecah menjadi 256 pasang dimensi (dimensi 1-2, dimensi 3-4, dimensi 5-6, dst). Setiap pasang dimensi ini "diputar" pada bidang 2D mereka masing-masing.
*   **Sudut Putarannya Berbeda-beda ($\theta_i$):** Perhatikan bahwa blok pertama menggunakan sudut $\theta_1$, blok kedua menggunakan $\theta_2$, dan blok terakhir $\theta_{d/2}$. Parameter sudut ini telah didefinisikan sebelumnya (secara *hardcoded*) dengan mengikuti progresi geometri (sama seperti fungsi posisi asli dari Transformer): $\Theta = \{\theta_i = 10000^{-2(i-1)/d}, i \in [1, 2, ..., d/2]\}$.
*   Semakin jauh/dalam dimensinya, kecepatan putaran sudutnya ($\theta_i$) akan semakin lambat.

Jika kita aplikasikan Matriks Rotasi RoPE raksasa ini ke operasi persamaan *Self-Attention* (Persamaan 2), kita mendapatkan:

$$ \boldsymbol{q}_m^\top \boldsymbol{k}_n = (\boldsymbol{R}_{\Theta,m}^d \boldsymbol{W}_q \boldsymbol{x}_m)^\top (\boldsymbol{R}_{\Theta,n}^d \boldsymbol{W}_k \boldsymbol{x}_n) = \boldsymbol{x}_m^\top \boldsymbol{W}_q^\top \boldsymbol{R}_{\Theta,n-m}^d \boldsymbol{W}_k \boldsymbol{x}_n \quad \quad \text{(16)} $$

**Kenapa ini sangat efisien dan brilian secara matematis?**
Perhatikan $\boldsymbol{R}_{\Theta,n-m}^d$. Sifat matematis dari Matriks Rotasi Orthogonal memastikan bahwa jika sebuah vektor diputar sejauh $m$ derajat, lalu dikalikan (*dot product*) dengan vektor lain yang diputar sejauh $n$ derajat, hasil skalar akhirnya **PERSIS SAMA** dengan mengalikan vektor murni (tanpa putaran) dengan vektor lain yang hanya diputar sejauh selisihnya, yaitu **$(n-m)$**. 

Jadi, RoPE berhasil memasukkan posisi absolut secara teknis (merotasi $m$ dan merotasi $n$), namun di dalam "perut komputasi atensi-nya", operasi matematika murni tersebut secara ajaib meleburkan informasi absolut tersebut dan hanya menyisakan perhitungan skor atensi berdasarkan **jarak relatif** $(n-m)$ saja. Ini mengawinkan metode Absolut dan Relatif secara sempurna.

Sangat berbeda dengan sifat "aditif" (penambahan/penjumlahan) yang diadopsi oleh penelitian-penelitian sebelumnya, pendekatan RoPE bersifat **multiplikatif (perkalian)**. Selain itu, tidak perlu ada lagi penambahan matriks vektor atau *bias* aneh di dalam hitungan atensi.

---
### Analisis Gambar 1: Implementasi Rotary Position Embedding (RoPE)

*(Deskripsi Naratif untuk "Figure 1: Implementation of Rotary Position Embedding (RoPE)")*

Gambar 1 merupakan diagram konseptual yang sangat jelas mengilustrasikan bagaimana RoPE diterapkan pada arsitektur Transformer.
*   **Blok di Dalam Garis Putus-putus Atas (Proses Rotasi 2D):**
    *   Terdapat representasi vektor kueri/kunci yang telah dipecah menjadi pasang 2D (ditandai dengan kotak hijau $\boldsymbol{x}_1$ dan abu-abu $\boldsymbol{x}_2$).
    *   Diagram sumbu X dan Y melukiskan operasi geometris: Vektor asal $(\boldsymbol{x}_1, \boldsymbol{x}_2)$ berada pada posisi awal di bidang datar.
    *   Faktor `m` (Posisi Absolut Kata) dan `θ1` (Konstanta Frekuensi) dikalikan menjadi `mθ1`.
    *   Vektor awal $(\boldsymbol{x}_1, \boldsymbol{x}_2)$ kemudian **diputar secara melingkar** (melalui garis lengkung merah) sebesar sudut `mθ1` sehingga mendarat di posisi yang baru, menjadi vektor $(\boldsymbol{x}'_1, \boldsymbol{x}'_2)$. Proses ini memvalidasi esensi nama "Rotary".
*   **Blok Bawah (Proses Keseluruhan Dimensi - General Form):**
    *   Di sisi kiri, terdapat matriks kueri/kunci asli dari sebuah kata. Dimensinya sangat panjang dan dipilah-pilah menjadi pasangan warna-warni (merah, hijau, biru, ungu). Sepasang warna mewakili satu porsi subruang 2D.
    *   Terdapat label baris nomor 1, 2, 3, 4, 5, 6 yang merepresentasikan urutan Posisi Absolut (Posisi 1, Posisi 2, dst).
    *   Panah hitam tebal (arah ke kanan) menunjukkan operasi Perkalian Matriks Rotasi.
    *   Di sisi kanan, kita melihat matriks hasil akhir ("*Position Encoded Query / Key*"). Kotak-kotak dimensinya memudar atau bergeser secara linear, merepresentasikan bahwa setiap posisi telah dirotasi pada *angle* (sudut) yang berbeda-beda. Blok kata pada posisi ke-6 diputar lebih drastis/banyak kali dibandingkan blok kata pada posisi ke-1.

---

### 3.3 Sifat-Sifat dari RoPE (*Properties of RoPE*)

**Peluruhan Jangka Panjang (*Long-term decay*):**
Mengikuti rumusan dari Vaswani (Transformer), sudut didefinisikan sebagai $\theta_i = 10000^{-2i/d}$. Secara matematis dapat dibuktikan (detail bukti ada di Bagian 3.4.3 di paper asli) bahwa pengaturan ini menghasilkan properti *long-term decay* (peluruhan berjangka panjang). Artinya, nilai *inner-product* (skor atensi) akan secara alami **meluruh (mengecil/menurun)** ketika jarak posisi relatif $(m-n)$ di antara kedua kata tersebut semakin membesar. 

Sifat ini sangat krusial dan secara persis sejalan dengan intuisi pemrosesan bahasa: Sebuah pasang kata (token) yang jaraknya berjauhan di dalam sebuah dokumen panjang seharusnya memiliki interaksi atau koneksi semantik yang jauh lebih lemah dibandingkan kata-kata yang saling berdekatan. RoPE memberikan sifat atensi ini "secara gratis" lewat sifat gelombang rotasinya.

**RoPE dapat dikombinasikan dengan Perhatian Linear (*RoPE with linear attention*):**
Mekanisme *self-attention* standar (Vanilla Transformer) mengharuskan model menghitung *dot product* antara Kueri dan Kunci untuk *setiap* pasangan token secara individual. Ini melahirkan beban komputasi kuadratik atau $\mathcal{O}(N^2)$ yang mana sangat memberatkan RAM jika teksnya panjang. 

Ada arsitektur Transformer alternatif yang bernama **Linear Attention** (seperti *Performer*). *Linear Attention* memecah eksponensial di *Softmax* dan memanfaatkan fungsi *kernel* (non-negatif) untuk mengubah beban komputasi dari eksponensial matriks raksasa menjadi perkalian linear biasa $\mathcal{O}(N)$. 

Metode pengkodean posisi alternatif yang mengandalkan PENJUMLAHAN *bias* pada *dot product* (seperti T5 atau DeBERTa di Persamaan 6-10) **tidak bisa** digunakan pada *Linear Attention*. Ini karena saat *dot product* dihilangkan dari rumusnya, *bias* posisional penjumlahan itu tidak tahu harus diletakkan di mana. 
Berbeda total dengan RoPE: Karena RoPE menginjeksikan posisi ke **dalam vektor secara independen melalui rotasi multiplikatif**, vektor-vektor yang telah dirotasi ini masih bisa langsung dimasukkan ke dalam arsitektur fungsi non-negatif *Linear Attention* tanpa merusak rumus matematisnya. Hal ini membuat RoPE menjadi pengkodean posisi yang sangat universal (dibuktikan di Persamaan 19 di *paper* asli).

---

## 4. Eksperimen dan Evaluasi (*Experiments and Evaluation*)

Para peneliti mengevaluasi metode RoFormer yang diusulkan pada berbagai tugas NLP. Mereka melakukan validasi pada tugas penerjemahan mesin, membandingkan pra-pelatihan RoPE dengan BERT, melakukan tugas klasifikasi akhir (GLUE benchmark), serta membuktikan kompatibilitas RoPE pada arsitektur *Performer* bertipe memori linear. Selain data bahasa Inggris, mereka juga melakukan eksperimen komprehensif pada dataset berbahasa Mandarin. Eksperimen dijalankan pada dua server *cloud* yang disokong dengan GPU 4x V100.

### 4.1 Penerjemahan Mesin (*Machine Translation*)

Mendemonstrasikan kinerja pada tugas penerjemahan *sequence-to-sequence*.

**4.1.1 Pengaturan Eksperimen**
Menggunakan dataset standar WMT 2014 English-German yang berisi 4,5 juta pasang kalimat. Pembanding utamanya adalah model dasar Transformer konvensional (Vaswani et al.).

**4.1.2 Detail Implementasi**
Modifikasi pada lapisan *self-attention* dilakukan pada model *baseline* untuk menggantikan fungsi *sinusoidal additive* dengan RoPE (multiplikatif). Tokenisasi menggunakan Byte Pair Encoding (BPE) dengan kosa kata gabungan sebesar 37k. Optimalisasi memakai Adam optimizer. Metrik evaluasi menggunakan BLEU score.

#### Tabel 1: Hasil Uji Coba WMT 2014 English-to-German (Penerjemahan)
| Model | BLEU |
| :--- | :---: |
| Transformer-base (Vaswani et al. 2017) | 27.3 |
| **RoFormer (Usulan)** | **27.5** |

*Kesimpulan: Secara langsung mengganti metode pengkodean posisi Transformer dasar dengan RoPE memberikan skor BLEU yang lebih tinggi (0.2 poin peningkat).*

---

### 4.2 Pemodelan Bahasa Pra-Pelatihan (*Pre-training Language Modeling*)

Eksperimen kedua ini bertujuan untuk memvalidasi kinerja RoPE dalam mempelajari representasi bahasa tingkat dalam (kontekstual). Peneliti mengganti pengkodean posisi absolut sinusoidal asli pada arsitektur **BERT** dengan RoPE yang diusulkan.

**Pengaturan:**
Menggunakan *BookCorpus* dan *Wikipedia Corpus* berbahasa Inggris untuk pra-pelatihan. Evaluasi diukur menggunakan nilai kerugian *Masked Language-Modeling* (MLM Loss). Arsitektur pembanding adalah model standar `bert-base-uncased`. Pelatihan dilakukan selama 100 ribu langkah (*100k steps*) dengan panjang urutan kata 512.

#### Analisis Gambar 3 (Sebelah Kiri): Kurva Evaluasi Pra-Pelatihan
*(Deskripsi Naratif Gambar 3 Kiri: "training loss for BERT and RoFormer")*
*   **Sumbu X:** Train Steps (K) - Langkah pelatihan dalam ribuan iterasi (dari 0 ke 250k).
*   **Sumbu Y:** MLM Loss - Tingkat kesalahan prediksi bahasa (berkisar dari atas 10 turun mendekati 2).
*   **Kurva Data:** Terdapat dua garis hiperbola lengkung yang turun curam di awal lalu melandai (*plateau*). Garis oranye merepresentasikan BERT konvensional, garis biru merepresentasikan RoFormer.
*   **Kesimpulan Akademis:** Kurva biru (RoFormer) selalu melesat dan **berada lebih rendah (jatuh lebih dalam dan lebih cepat)** di bawah kurva oranye (BERT) sepanjang siklus *training*. Hal ini secara klinis membuktikan bahwa integrasi matriks rotasi (RoPE) membuat arsitektur belajar mengenali struktur bahasa jauh lebih cepat (mempercepat *konvergensi* jaringan saraf).

---

### 4.3 Penyesuaian Halus (*Fine-Tuning*) pada Tugas GLUE

Sesuai standar NLP, model yang telah dilatih mandiri pada data Wikipedia tersebut kini ditugaskan (di *fine-tune*) pada 6 tugas kompetisi pemahaman bahasa standar yang tergabung dalam kompilasi dataset **GLUE**. Metrik yang digunakan adalah F1-score dan Akurasi.

#### Tabel 2: Perbandingan RoFormer dan BERT (Fine-tuning pada GLUE)

| Model | MRPC | SST-2 | QNLI | STS-B | QQP | MNLI (m/mm) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| BERT (Devlin et al. 2019) | 88.9 | **93.5** | **90.5** | 85.8 | 71.2 | **84.6/83.4** |
| **RoFormer** | **89.5** | 90.7 | 88.0 | **87.0** | **86.4** | 80.2/79.8 |

*Kesimpulan Analisis Tabel 2: Hasilnya kompetitif. RoFormer berhasil mengalahkan BERT murni secara signifikan pada 3 dari 6 dataset tersebut (MRPC, STS-B, dan QQP). Hal ini membuktikan bahwa intuisi teoretis RoPE tervalidasi dapat diterapkan dengan sukses untuk tugas transfer pengetahuan hilir.*

---

### 4.4 Pengaplikasian RoPE pada Performer (*Linear Attention*)

Seperti yang telah didiskusikan secara teoritis pada subbab 3.3, RoPE adalah satu-satunya metode posisi relatif modern yang dapat diintegrasikan dengan mulus ke dalam arsitektur "Perhatian Linear" (*Linear Attention*) demi mengurangi beban kuadratik menjadi skalar linear. Peneliti membuktikan teori tersebut dengan memasukkan RoPE ke dalam arsitektur **Performer** 12 lapis, kemudian mengujinya pada dataset karakter Wikipedia (Enwik8).

#### Analisis Gambar 3 (Sebelah Kanan): Kurva Performer dengan RoPE
*(Deskripsi Naratif Gambar 3 Kanan: "training loss for PerFormer with and without RoPE")*
*   **Sumbu X:** Train Steps (K) - Langkah pelatihan (dari 0 ke 100k).
*   **Sumbu Y:** LM Loss - Tingkat kesalahan model bahasa (dari 3.0 ke 1.5).
*   **Kurva Data:** Garis Oranye merepresentasikan model Performer tanpa posisi (*w/o RoPE*). Garis Biru merepresentasikan model Performer yang disuntikkan vektor posisi rotasi (*w/ RoPE*).
*   **Kesimpulan Akademis:** Kurva biru jauh lebih curam, menembus jauh ke bawah secara drastis dibandingkan garis oranye. Menyubstitusi RoPE ke dalam Performer memicu akselerasi konvergensi dan tingkat eror yang secara absolut lebih rendah dalam jumlah waktu latihan yang persis sama. Keberhasilan matematis ini mengukuhkan RoPE sebagai komponen vital bagi efisiensi sistem NLP yang panjang (dimana memori kuadratik ditiadakan melalui Performer).

---

### 4.5 Evaluasi pada Data Bahasa Mandarin (Chinese Data)

Untuk memvalidasi kehebatan RoFormer dalam menangani *long texts* (teks yang sangat panjang, melampaui batasan keras 512 karakter milik arsitektur posisi absolut asli BERT), mereka juga bereksperimen menggunakan teks berbahasa Mandarin. RoFormer dilatih pada korpus 34GB teks Wikipedia, Berita, dan Forum. Karena kemampuan unik sifat rotasinya, RoFormer dilatih lintas tahap: pertama pada ukuran kalimat 512 karakter, lalu **dibengkakkan skalanya secara radikal ke 1536 karakter maksimal per kalimat**. 

Untuk menguji performa pemahaman dokumen sangat panjang, RoFormer diikutsertakan pada tugas pencocokan kemiripan dokumen kasus Hukum (Dataset CAIL2019-SCM) yang masing-masing putusan hukumnya rata-rata memanjang lebih dari 512 huruf. 

#### Tabel 5: Hasil Eksperimen pada Pencocokan Dokumen Legal Mandarin (CAIL2019-SCM)

| Model | Evaluasi Validasi | Uji Test Set |
| :--- | :---: | :---: |
| BERT-512 | 64.13% | 67.77% |
| WoBERT-512 | 64.07% | 68.10% |
| **RoFormer-512** | **64.13%** | **68.29%** |
| **RoFormer-1024** | **66.07%** | **69.79%** |

*(Catatan Analisis: Pada batas teks standar (512 kata), RoFormer mencatat F1 Score yang tipis memimpin. Namun, magis sesungguhnya terjadi ketika model dipaksa membaca kalimat yang ditarik super panjang (RoFormer-1024). Di sini, ia melibas dominasi seluruh model baseline absolut dengan peningkatan performa lompatan tajam ke 69.79%. Ini menegaskan bahwa kemampuan model memfaktorkan relasi pembacaan jarak jauh (berkat sifat matematis eksponensial sudut Euler) bekerja tanpa distorsi meskipun urutan kalimat diregangkan hingga panjang maksimal, sesuatu yang haram bagi BERT).*

---

### 4.5.5 Batasan dari Makalah Ini (*Limitations of the work*)
Terlepas dari penemuan ini, para periset dengan jujur mengakui beberapa kekurangan analisis dari makalah mereka yang perlu riset lanjutan:
1. Meskipun mereka secara matematis sangat teliti memformat posisi relatif sebagai manipulasi rotasi matriks ruang 2D, mereka **masih belum dapat menyusun teori matematis yang memuaskan** mengapa model RoPE bisa hội konvergen/paham data bahasa secara instan (lebih cepat dari arsitektur *baseline* lama).
2. Walaupun mereka berhasil mengonfirmasi secara teoritis properti "peluruhan jarak panjang" (*long-term decay*), dan secara empiris membuktikan model ini luar biasa di dokumen teks berhalaman-halaman (melebihi *peer models* lainnya), sayangnya para periset **belum sepenuhnya menemukan alasan kognitif yang memadai** untuk menjawab korelasi fenomena di atas (mengapa hal ini bekerja sangat baik dalam mendeteksi makna kata berjauhan).

## 5. Kesimpulan (*Conclusions*)

Di dalam karya penelitian fundamental ini, telah diusulkan sebuah metode baru penyematan (pengkodean) posisi untuk matriks jaringan saraf yang mampu menggabungkan secara eksplisit kebergantungan informasi "posisi relatif" langsung masuk merasuk ke dalam darah perhitungan arsitektur atensi *(self-attention)*. 

Analisis teoritis radikal mereka membuktikan bahwa posisi spasial relatif dari suatu kata di kalimat secara alami—dan paling logis—**harus diformulasikan menggunakan operasi produk vektor matematika rotasi**. Posisi absolut dikodekan dengan elegan lewat sebuah "Matriks Rotasi", di mana perbedaan (relasi) posisi secara implisit termanifestasi sebagai selisih sudut putar rotasi di kalkulasi akhirnya. 

Secara sistematis, peneliti mendemonstrasikan kelenturan dan keunggulan absolut dari properti metode diusulkan (RoPE) ini jika dipasangkan pada anatomi Transformer (termasuk arsitektur canggih pemecah kompleksitas komputasional seperti arsitektur *Linear Performer*). Akhir kata, seluruh demonstrasi klinis pada pangkalan data korpus teks Bahasa Inggris serta Mandarin menyajikan bukti empiris yang padat: RoPE memicu model menelan data pelatihan (*converge*) secara dipercepat, sekaligus memecahkan limitasi atap kemampuan algoritma di era komputasi masa lalu tatkala menghadapi teks dokumen bacaan yang membentang sangat panjang.

---
*(Catatan Dosen: Sampai di sini batas akhir dari bedah teknis makalah "RoFormer: Enhanced Transformer with Rotary Position Embedding". Bagian Referensi dan Apendiks dari dokumen asli diabaikan sesuai panduan studi. Pelajari setiap landasan logis peralihan dari "posisi aditif sinus" (Vaswani) menjadi "posisi matriks putar multiplikatif" (RoPE). Karena dari persamaan matematis nomor (16) di ataslah, otak-otak LLMs modern yang berukuran ratusan miliar parameter saat ini bisa dilatih membaca buku yang berpanjang ratusan halaman tanpa tersesat!)*