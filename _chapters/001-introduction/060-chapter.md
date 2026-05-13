---
slug: bab-1-6
title: "Jia Bin Huang DeepSeek V4"
layout: chapter
---

---

# Evolusi dan Pemodelan Matematika di Balik Efisiensi *Attention* pada Model Bahasa Besar: Dari DeepSeek V3 hingga DeepSeek V4

Pada akhir September 2025, DeepSeek merilis sebuah model eksperimental (DeepSeek V3/R1) yang memperkenalkan varian mekanisme *attention* baru bernama **DeepSeek Sparse Attention (DSA/DSR)**. Inovasi ini mampu mempertahankan performa *State-of-the-Art* (SOTA) sekaligus memangkas biaya komputasi secara drastis, yang berujung pada penurunan harga API hingga 50%. Tak lama berselang, DeepSeek merilis penerusnya, **DeepSeek V4**, yang membawa efisiensi konteks panjang (*long-context efficiency*) ke tingkat yang jauh lebih ekstrem melalui arsitektur kompresi baru.

Untuk memahami bagaimana model mutakhir ini mencapai efisiensi yang luar biasa—dan secara radikal memangkas memori komputasi—kita harus membedah mekanisme ini secara berurutan dari prinsip-prinsip dasarnya (*first principles*).

## 1. Fondasi Pemrosesan Bahasa: Tokenisasi dan *Embedding*
Pada tingkat abstraksi tertinggi, Model Bahasa Besar (LLM) memproses masukan (*input*) untuk menghasilkan respons keluaran (*output*). Langkah paling fundamental adalah memecah teks ke dalam unit diskrit yang disebut **token**. Setiap token dipetakan pada sebuah ID unik di dalam kamus (*tokenization*).

Namun, ID token secara inheren tidak menyimpan makna semantik. Oleh karena itu, model memetakan setiap token ke dalam ruang vektor berdimensi $d$ kontinu, yang dikenal sebagai ***token embedding***. Di dalam ruang dimensi $d$ ini (sekitar 7.000 dimensi pada arsitektur DeepSeek awal), kata-kata dengan makna serupa memiliki jarak vektor yang saling berdekatan. Akan tetapi, representasi *embedding* ini masih berdiri sendiri dan belum memiliki informasi kontekstual mengenai urutan kata di sekitarnya. Di sinilah mekanisme *Attention* berperan.

## 2. Mekanisme *Self-Attention* Dasar
*Attention* memungkinkan model menentukan seberapa besar kontribusi setiap token dalam mengumpulkan informasi kontekstual dari seluruh sekuens. Hubungan antartoken diukur di dalam sebuah ruang laten yang direpresentasikan oleh matriks *Query* ($W_Q$), *Key* ($W_K$), dan *Value* ($W_V$).

Matriks *Query* berukuran $d \times d_k$, di mana $d$ adalah ukuran *embedding* dan $d_k$ adalah dimensi dari vektor *Query*. Vektor *Query* ($q$) dan vektor *Key* ($k$) untuk setiap token dihitung melalui perkalian matriks dengan representasi *embedding* ($x$):
- Vektor *Query* ($1 \times d_k$): $q = x \cdot W_Q$
- Vektor *Key* ($1 \times d_k$): $k = x \cdot W_K$

Relevansi antara satu token dengan token lainnya dikuantifikasi menggunakan perkalian titik (*dot product*) antara vektor *Query* dan *Key*. Karena LLM memproduksi teks secara **autoregresif** (satu token pada satu waktu), matriks penopeng (*masking matrix* $M$) diaplikasikan agar sebuah token hanya dapat "memperhatikan" dirinya sendiri dan token-token yang mendahuluinya.

Mengingat nilai *dot product* dapat berupa bilangan positif maupun negatif, fungsi **Softmax** diaplikasikan untuk mengubah skor mentah ini menjadi distribusi probabilitas yang ternormalisasi, menghasilkan **skor *attention***. Skor ini menjadi pembobot (*weight*) untuk mengekstraksi informasi dari vektor *Value* ($v = x \cdot W_V$).

Hasil akhirnya adalah agregasi jumlah berbobot (*weighted sum*) dari vektor *Value*. Mengingat dimensi vektor *Value* lebih kecil dari dimensi asal $d$, matriks keluaran ($W_O$) digunakan untuk memproyeksikan kembali vektor ini ke dimensi $d$. Hasilnya, yang disebut representasi residual ($\Delta x$), ditambahkan ke *token embedding* asli untuk memperbarui informasi kontekstualnya. 

Secara ringkas dalam notasi matriks, fungsi *attention* dinyatakan sebagai:
$$ \text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V $$
Keluaran akhir model adalah perkalian antara matriks agregasi *Attention* ($O$) dan matriks $W_O$.

## 3. Skalabilitas *Attention* dan Masalah *Key-Value (KV) Caching*
Satu set komputasi (satu *head*) tidak cukup untuk menangkap kerumitan hubungan linguistik. Oleh karena itu, diperkenalkan ***Multi-Head Attention* (MHA)**, di mana matriks $Q, K, V$ dipecah ke dalam sejumlah $h$ *heads* (misalnya $h=128$). Komputasi dilakukan paralel, lalu keluarannya ($O_1, O_2, \dots, O_h$) dikonkatenasi sebelum dikalikan dengan $W_O$. 

Pada fase dekode (penghasilan teks), vektor *Key* dan *Value* dari token-token sebelumnya tidak pernah berubah. Menghitung ulangnya untuk setiap token baru adalah pemborosan komputasi ekstrem. Solusinya adalah menyimpannya di dalam memori yang disebut ***Key-Value (KV) Caching***. (Vektor *Query* masa lalu tidak perlu disimpan karena telah ditutupi oleh *masking*).

Namun, KV *Cache* memakan memori yang tumbuh linier seiring bertambahnya panjang sekuens. 
Total memori KV Cache = **$2 \times \text{dimensi KV} \times \text{presisi (bytes)} \times \text{jumlah heads} \times \text{jumlah layer} \times \text{panjang sekuens}$**.
Sebagai contoh, pada arsitektur dengan presisi *Float16* (2 *bytes*), 128 angka per vektor, 128 *heads*, 61 *layers*, dan panjang konteks 32K token, memori yang dibutuhkan mencapai **131 GB**. Skala ini menjadi masalah utama dalam LLM.

## 4. Evolusi Kompresi Dimensi *Head*: MQA, GQA, dan MLA
Langkah pertama mengatasi limitasi memori adalah merekayasa jumlah *heads*.
1.  ***Multi-Query Attention* (MQA)**: Memproduksi hanya 1 set vektor *Key* dan *Value* yang dibagi pakai (*shared*) untuk seluruh *Query heads*. Memori turun hingga 128 kali lipat (dari 4 MB/token menjadi 31 kB/token), namun performa model menurun drastis karena hilangnya kapasitas representasi hubungan yang kompleks.
2.  ***Grouped-Query Attention* (GQA)**: Kompromi modern di mana 128 *heads* dibagi ke dalam $n_g$ grup (misal 16 grup). GQA digunakan oleh Llama, Qwen, dan Gemma untuk menyeimbangkan efisiensi memori dan daya ekspresif model.

DeepSeek (pada V3/R1) melangkah lebih jauh dengan ***Multi-Head Latent Attention* (MLA)**. Dari perspektif aljabar linier, GQA sebenarnya adalah faktorisasi peringkat-rendah (*low-rank factorization*). MLA mengembangkan hal ini dengan menjadikan matriks proyeksinya dapat dilatih (*trainable*). 
Representasi *embedding* diturunkan (*down-projection*) oleh matriks $W_{KV-down}$ menjadi vektor laten ringkas $c_{KV}$ (berdimensi $d_c$, misal 576). Vektor laten ini kemudian diproyeksikan ke atas (*up-projection*) oleh matriks $W_{K-up}$ dan $W_{V-up}$ untuk menghasilkan *Key* dan *Value* unik per *head*. MLA mengurangi memori hingga 57 kali lipat dibanding MHA standar, namun terbukti meningkatkan performa.

**Trik Absorpsi Inferensi:** Keunggulan MLA terletak pada komputasi saat inferensi. Karena operasi *dot product* mengikuti sifat asosiatif matriks, proses *up-projection* matriks $W_{Q-up}$ dan $W_{K-up}$ (serta $W_{V-up}$ yang dilebur ke dalam $W_O$) dapat diserap (*absorbed*) menjadi satu matriks pra-komputasi. Ini berarti model tidak perlu mendekonstruksi vektor laten menjadi *Key* dan *Value* secara eksplisit selama inferensi.

**Konflik dengan RoPE:** MLA memiliki masalah dengan *Rotary Positional Embedding* (RoPE)—mekanisme pengkodean posisi melalui rotasi vektor geometris ($\theta$). Matriks rotasi RoPE menyela ekuasi tepat di antara matriks kompresi dan proyeksi, menghancurkan sifat asosiatif sehingga matriks tidak bisa lagi diserap secara efisien. DeepSeek menyelesaikan ini melalui **Decoupled RoPE (Partial RoPE)**, di mana matriks $Q$ dan $K$ untuk RoPE dipisahkan dari kompresi laten, lalu dikonkatenasi sebelum komputasi *attention*.

## 5. Kompresi Berbasis Sekuens: Inovasi DeepSeek V4
Metode di atas berfokus mengompresi "lebar" arsitektur (*heads*). Pada rilis **DeepSeek V4**, fokus dialihkan pada kompresi "panjang" arsitektur (*sequence length*). Memori KV tidak hanya ditekan dari sisi *head*, melainkan dari sisi panjang teks melalui ***Token-Level Compressor***.

Pada tahap ini, representasi laten diproyeksikan ke sebuah entri dimensi $c$ (bernilai 512 pada V4). Sejumlah token yang berdekatan (misalnya grup berisi 4 token) dikompresi menjadi satu entri KV tunggal. Namun, melakukan rata-rata (*averaging*) sederhana akan menghilangkan informasi penting dari token spesifik. 

Sebagai gantinya, V4 menggunakan **Pembobotan Berbasis Data Terhadap Dimensi (*Data-dependent Per-dimension Weighting*)**. 
Model menghitung skor *scalar* dari *dot product* representasi *hidden state* dengan matriks bobot yang dipelajari ($W_z$). Skor ini masuk ke fungsi *Softmax* untuk mendapatkan bobot spesifik **pada setiap dimensi individu** dalam vektor KV berukuran 512 tersebut. Hal ini memberikan model kendali *fine-grained* mengenai fitur matematis apa saja yang harus dipertahankan.

Untuk mencegah terciptanya batasan antar-grup secara artifisial, V4 mengaplikasikan **Jendela Tumpang Tindih (*Overlapping Groups*)**. Sebuah representasi token dibagikan melintasi beberapa jendela kompresi yang saling tumpang tindih. Skema ini memastikan transisi informasi yang mulus (*smoother transition*) tanpa kehilangan konteks lokal.

## 6. Efisiensi Parameter Melalui Proyeksi Peringkat-Rendah (*Low-Rank Projection*)
Karena entri KV kini sangat padat (dibagikan ke seluruh *heads* ala MQA namun dengan ukuran 512-dimensi per *head*), dimensi proyektor untuk *Query* ($W_Q$) menjadi sangat raksasa. 
Untuk menyelesaikannya, DeepSeek V4 menerapkan proyektor *Low-Rank*:
1.  **Untuk Query:** Menggunakan dekomposisi $W_{Q-down}$ dan $W_{Q-up}$, menekan jumlah parameter hanya menjadi 15,8% dari total semula.
2.  **Untuk Output (*Grouped Output Projection*):** Dimensi *output* dibagi menjadi $g$ grup, dipetakan turun oleh $W_{O-down}$ ke vektor $n_g$, dan dipulihkan oleh $W_{O-up}$. Parameter berkurang hingga tersisa 14,3% saja.

## 7. Penyeleksian Jarang: *Lightning Indexer* dan *DeepSeek Sparse Attention* (DSA)
Bahkan setelah dikompresi, komputasi *attention* terhadap sekuens jutaan token masih membebani *throughput* (token-per-detik). DeepSeek mengatasi ini dengan ***Lightning Indexer*** (inti dari mekanisme **DSA**).
*Lightning Indexer* bekerja layaknya mekanisme prakiraan cepat: ia mengevaluasi relevansi seluruh token historis dan secara *sparse* (jarang) hanya mengekstraksi sebagian kecil token historis yang paling relevan untuk diikutsertakan ke dalam komputasi matematis *attention* yang berat.

*Indexer* ini menggunakan komputasi mirip MHA (menggunakan *Query* individual namun *Key* yang dibagikan). Hasil *dot product*-nya melewati fungsi aktivasi **ReLU** untuk membuang nilai korelasi negatif. Komputasi ini dibuat sangat efisien melalui operasi **Kuantisasi 8-bit**.

**Mengatasi Limitasi Kuantisasi:** Kuantisasi secara naif rentan memotong nilai varians ekstrem (lonjakan/*spikes*). Pendekatan ortogonal acak terlalu berat komputasinya. DeepSeek merespons ini dengan menggunakan metode ***Fast Walsh-Hadamard Transform***. Transformasi ini menyebarkan nilai-nilai ekstrem secara deterministik dan seragam ke seluruh koordinat vektor matriks, memungkinkan kuantisasi 8-bit mempertahankan informasi krusial. Lebih jauh, karena sifat transformasinya, proses ini tidak membutuhkan komputasi matriks padat (*dense matrix*), melainkan diselesaikan murni lewat operasi penjumlahan dan pengurangan teroptimisasi pada GPU.

**Pelatihan Indexer:** Pelatihan modul ini dibagi dua fase:
1.  ***Dense Warm-up***: Lapisan utama dibekukan. Objektifnya adalah menyelaraskan distribusi atensi *indexer* agar identik dengan atensi *head* utama (menggunakan Normalisasi L1).
2.  ***Fine-grained Token Selection***: Grafik komputasi (*computational graph*) diputus (*detached*). *Indexer* dilatih secara independen menurut fungsi kerugiannya (*loss*)-nya sendiri untuk menyeleksi token yang tepat, sementara arsitektur utama dioptimasi murni pada fungsi kerugian pemodelan bahasa (*language modeling loss*).

## 8. Arsitektur Komprehensif DeepSeek V4: Integrasi CSA dan HCA
Seluruh komponen di atas dilebur ke dalam rilis DeepSeek V4 yang menetapkan struktur hibrida yang sangat terspesialisasi:

1.  ***Compressed Sparse Attention* (CSA)**: Perpaduan antara *Token-Level Compressor* (rasio 4:1 dengan jendela tumpang tindih) ditambah seleksi *sparse* (dari *Lightning Indexer*/DSA). CSA mengekstraksi detail halus lokal.
2.  ***Heavily Compressed Attention* (HCA)**: Kompresi ekstrem dengan rasio **128:1** (128 token dirangkum jadi 1 KV entri). Karena ukurannya sangat kecil, HCA tidak memerlukan *Lightning Indexer*. HCA dijalankan secara *dense* (padat) untuk menangkap konteks ringkasan global dari jarak yang sangat panjang.

**Jadwal Arsitektur (Layering Schedule):**
Untuk menyeimbangkan akurasi analitis dan efisiensi memori, DeepSeek V4 menggunakan jadwal yang bergiliran:
- **Layer Awal:** Murni menggunakan HCA. Model membangun ringkasan global murah dari rentang konteks raksasa sebelum melakukan pemrosesan detail.
- **Layer Tengah:** Berganti-ganti secara iteratif antara HCA (pengumpulan konteks global) dan CSA (interaksi lokal resolusi tinggi dengan *sparse selection*).
- **Layer Terakhir:** Diakhiri dengan mekanisme *Full Attention* tradisional (tanpa kompresi) untuk memulihkan secara presisi informasi yang mungkin kabur (*blurred*) akibat kompresi di layer sebelumnya, demi menghasilkan *output logits* yang akurat.

Untuk mencegah instabilitas matematis (*exploding logits*) dari rentang variabel yang fluktuatif ini, diterapkan normalisasi khusus pada representasi *Query* dan KV terkompresi. Sistem juga mengimplementasikan representasi laten khusus yang dapat dilatih, yang disebut ***Learnable Attention Sink Logits***, yang bertugas meredam dan menyerap "massa *attention* berlebih", menjamin berjalannya pelatihan yang sangat stabil.

### Kesimpulan
Melalui fusi *Multi-Head Latent Attention*, *Token-Level Sequence Compression*, Transformasi Walsh-Hadamard pada sistem kuantisasi *Indexer*, serta eksekusi hibrida berlapis antara CSA dan HCA, DeepSeek telah berhasil mendobrak salah satu tembok terbesar (*bottleneck*) dalam ilmu komputer modern: mengeksekusi konteks jutaan token dengan penghematan waktu hingga 3 kali lipat dan penghematan memori masif, tanpa mengorbankan setitik pun daya nalar arsitektur AI generatif.