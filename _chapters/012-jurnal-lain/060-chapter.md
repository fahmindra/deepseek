---
slug: ch-12-06
title: "MTP"
layout: chapter
---

## Abstrak (*Abstract*)

Model bahasa besar seperti GPT dan Llama pada dasarnya dilatih dengan menggunakan kerugian prediksi token-berikutnya (*next-token prediction loss*). Dalam kajian ini, para peneliti menyarankan bahwa melatih model bahasa untuk memprediksi **beberapa (multiple) token masa depan secara sekaligus** akan menghasilkan efisiensi sampel (*sample efficiency*) yang lebih tinggi.

Secara lebih spesifik, pada setiap posisi di dalam korpus pelatihan, model diminta untuk memprediksi $n$ token berikutnya dengan menggunakan $n$ buah "kepala keluaran" (*output heads*) yang independen. Semua kepala ini beroperasi di atas satu "batang model" (*model trunk*) yang digunakan bersama. 

Dengan mempertimbangkan prediksi multi-token ini sebagai sebuah tugas pelatihan tambahan (*auxiliary training task*), para peneliti mengukur peningkatan kemampuan model pada tugas-tugas hilir (*downstream capabilities*) **tanpa adanya biaya tambahan (overhead) pada waktu pelatihan**, baik untuk model bahasa pemrograman (kode) maupun bahasa alami. 

Metode ini terbukti semakin berguna untuk ukuran model yang lebih besar, dan tetap mempertahankan daya tariknya ketika model dilatih untuk beberapa siklus (*multiple epochs*). Keuntungan atau peningkatan performa ini secara khusus sangat menonjol pada tolok ukur pengujian generatif (*generative benchmarks*) seperti penulisan kode, di mana model-model yang menggunakan metode ini secara konsisten mengungguli model-model dasar (*baselines*) yang kuat dengan selisih beberapa poin persentase. 

Sebagai contoh, model dengan 13 Miliar (13B) parameter mampu memecahkan **12% lebih banyak** masalah pada dataset HumanEval dan **17% lebih banyak** pada dataset MBPP dibandingkan dengan model prediksi token-berikutnya yang seukuran. 

Eksperimen yang dilakukan pada tugas-tugas algoritmik berskala kecil mendemonstrasikan bahwa prediksi multi-token sangat menguntungkan untuk pengembangan "kepala induksi" (*induction heads*) dan kemampuan penalaran algoritmik. Sebagai keuntungan tambahan, model yang dilatih dengan prediksi 4-token mampu berjalan hingga **3 kali lebih cepat saat inferensi** (proses menghasilkan teks), bahkan ketika menggunakan ukuran kelompok (*batch sizes*) yang besar.

---

## 1. Pendahuluan (*Introduction*)

Umat manusia telah memadatkan berbagai upaya jenius, penemuan yang mengejutkan, dan karya yang indah ke dalam bentuk teks. Model Bahasa Besar (*Large Language Models* / LLMs) yang dilatih menggunakan semua korpus (kumpulan teks) ini mampu mengekstraksi sejumlah besar pengetahuan dunia yang mengesankan, sekaligus mengembangkan kemampuan penalaran dasar. 

Hal tersebut dicapai dengan mengimplementasikan sebuah tugas pembelajaran tak terselia (*unsupervised learning task*) yang sederhana, namun sangat kuat: **prediksi token-berikutnya (*next-token prediction*)**. Meskipun baru-baru ini terjadi gelombang pencapaian yang mengesankan (seperti yang ditunjukkan oleh OpenAI, 2023), prediksi token-berikutnya tetaplah merupakan cara yang **tidak efisien** untuk memperoleh bahasa, pengetahuan dunia, dan kemampuan penalaran.

Secara lebih tepat, metode "pemaksaan guru" (*teacher forcing*) yang dibarengi dengan prediksi token-berikutnya cenderung membuat model hanya terpaku pada **pola-pola lokal (jarak dekat)** dan sering kali mengabaikan keputusan-keputusan yang "sulit" atau yang membutuhkan pemahaman konteks jangka panjang. 

Akibatnya, sebuah fakta yang tak terbantahkan adalah bahwa prediktor token-berikutnya paling mutakhir (*state-of-the-art*) masih **membutuhkan data dalam orde magnitudo yang jauh lebih banyak** dibandingkan dengan anak manusia untuk mencapai tingkat kelancaran bahasa yang sama (Frank, 2023).

### Analisis Gambar 1: Gambaran Umum Prediksi Multi-Token

*(Catatan Dosen: Perhatikan baik-baik Gambar 1 pada makalah. Ini adalah inti dari seluruh arsitektur yang kita bahas).*

**Deskripsi Bagian Atas (Arsitektur Model):**
Gambar bagian atas menunjukkan sebuah diagram arsitektur jaringan saraf. 
*   Di bagian bawah, terdapat deretan kotak kuning bernomor 1, 2, 3, 4 yang diberi label **"Inputs"**. Ini adalah token-token (kata-kata) yang sedang dibaca oleh model dari sebuah teks.
*   Teks input ini dimasukkan ke dalam blok besar berwarna biru gelap yang berlabel **"Shared"**. Ini adalah *model trunk* (batang utama model), yang berisi mayoritas lapisan Transformer. Blok ini merepresentasikan pemahaman memori dan konteks dari teks yang dibaca.
*   Di atas blok "Shared", arsitektur ini memecah diri menjadi **4 cabang hijau**, yang masing-masing berlabel **Head 1, Head 2, Head 3,** dan **Head 4**. Ini adalah kepala keluaran independen.
*   Perhatikan apa yang dihasilkan (target) dari masing-masing kepala:
    *   **Head 1** bertugas memprediksi token berikutnya secara langsung (token ke-2, 3, 4, 5).
    *   **Head 2** melompati satu token, dan memprediksi token masa depan yang kedua (token ke-3, 4, 5, 6).
    *   **Head 3** memprediksi token masa depan yang ketiga (token ke-4, 5, 6, 7).
    *   **Head 4** memprediksi token masa depan yang keempat (token ke-5, 6, 7, 8).
*   *Catatan penting di kanan atas:* Saat proses *Inference* (penggunaan model di dunia nyata untuk *chatting*), model bisa saja membuang (discard) Head 2, 3, dan 4 dan hanya menggunakan Head 1 seperti LLM biasa. Atau, secara opsional, Head 2, 3, dan 4 ini bisa digunakan untuk mempercepat generasi teks hingga 3 kali lipat (menggunakan teknik *speculative decoding*).

**Deskripsi Bagian Bawah (Grafik Performa):**
Bagian bawah menampilkan sebuah grafik batang horizontal (*bar chart*).
*   **Sumbu X:** Merepresentasikan peningkatan absolut (dalam poin persentase) dari metrik `pass@1` (tingkat kelulusan percobaan pertama) pada dataset pemrograman MBPP. Angkanya berkisar dari -1.7 hingga +4.5. Garis vertikal putus-putus di angka 0 merepresentasikan "Baseline" (performa model prediksi 1-token konvensional).
*   **Sumbu Y:** Menunjukkan Ukuran Model (*Model size*), mulai dari yang terkecil 0.3B (Miliar parameter) hingga yang terbesar 13B.
*   **Bilah Batang (Bar):** Terdapat bilah batang untuk setiap ukuran model. Batang berwarna oranye (mengarah ke kiri/negatif) menunjukkan performa yang lebih buruk dari baseline. Batang berwarna hijau (mengarah ke kanan/positif) menunjukkan performa yang lebih baik. Garis hitam horizontal di ujung batang merepresentasikan interval kepercayaan (*confidence interval*) 90% yang dihitung menggunakan teknik *bootstrapping*.
*   **Kesimpulan Akademis Grafik:** Untuk model yang sangat kecil (0.3B dan 0.6B), prediksi 4-token justru **merugikan** performa (nilai negatif). Namun, saat ukuran model diperbesar (mulai dari 1.3B hingga 13B), terjadi fenomena **pembalikan (reversal)**. Prediksi 4-token menjadi sangat menguntungkan. Pada model 13B, metode ini memberikan peningkatan performa yang sangat signifikan, menambah tingkat kelulusan sekitar +4.5 poin persentase di atas *baseline*. Ini membuktikan bahwa metode ini baru akan bersinar ketika model memiliki kapasitas memori (parameter) yang cukup besar untuk menyerap tugas ganda.

---

Dalam studi ini, para peneliti berargumen bahwa melatih LLM untuk memprediksi **beberapa (multiple) token sekaligus** akan mendorong model-model ini menuju tingkat efisiensi sampel yang lebih baik. 

Seperti yang telah diantisipasi pada Gambar 1, prediksi multi-token menginstruksikan LLM untuk memprediksi $n$ token masa depan dari setiap posisi di dalam korpus pelatihan. Proses prediksi ini dilakukan sekaligus pada waktu yang bersamaan dan secara paralel (Qi et al., 2020).

**Kontribusi Penelitian (*Contributions*)**
Meskipun prediksi multi-token telah dipelajari dalam literatur sebelumnya (Qi et al., 2020), penelitian ini menawarkan kontribusi-kontribusi baru sebagai berikut:

1.  Peneliti mengusulkan sebuah arsitektur prediksi multi-token yang sederhana **tanpa adanya biaya tambahan (overhead) pada waktu pelatihan maupun penggunaan memori** (akan dibahas di Bagian 2).
2.  Peneliti memberikan bukti eksperimental bahwa paradigma pelatihan ini bermanfaat pada skala besar (*at scale*), di mana model dengan parameter hingga 13B mampu menyelesaikan masalah penulisan kode **sekitar 15% lebih banyak secara rata-rata** (Bagian 3).
3.  Prediksi multi-token memungkinkan dilakukannya teknik **dekoding berspekulasi-mandiri (*self-speculative decoding*)**, yang membuat model mampu berjalan hingga 3 kali lebih cepat saat waktu inferensi melintasi berbagai ukuran kelompok (*batch-sizes*) (Bagian 3.2).

Meskipun bebas biaya komputasi tambahan dan sangat sederhana, prediksi multi-token adalah sebuah modifikasi yang efektif untuk melatih model transformer yang lebih kuat dan lebih cepat. Para peneliti berharap bahwa karya ini akan memicu ketertarikan pada eksplorasi fungsi kerugian tambahan (*auxiliary losses*) yang inovatif untuk LLM—jauh melampaui sekadar prediksi token-berikutnya—dengan tujuan untuk meningkatkan performa, koherensi, dan kemampuan penalaran dari model-model yang menakjubkan ini.

---

## 2. Metode (*Method*)

Pemodelan bahasa standar belajar dari korpus teks berukuran besar $x_1, \dots x_T$ dengan cara mengimplementasikan tugas prediksi token-berikutnya. Secara formal, tujuan pembelajaran objektifnya adalah untuk meminimalkan fungsi **kerugian lintas-entropi (*cross-entropy loss*)**.

$$ L_1 = - \sum_t \log P_\theta(x_{t+1} \mid x_{t:1}), \quad \quad \text{(1)} $$

**Eksplorasi Mendalam Rumus (1): Fungsi Kerugian Token-Berikutnya Konvensional**
*   **$L_1$**: Melambangkan total nilai kerugian (*loss*) untuk prediksi 1-token ke depan. Tujuan pelatihan adalah meminimalkan nilai $L_1$ ini sekecil mungkin.
*   **$\sum_t$**: Adalah simbol penjumlahan (Sigma). Ini berarti kita menghitung nilai kerugian pada *setiap* posisi waktu $t$ di seluruh dokumen teks yang sedang dibaca.
*   **$P_\theta$**: Adalah model bahasa besar kita (misal: Llama atau GPT) yang dikendalikan oleh miliaran parameter bobot yang disimbolkan dengan $\theta$.
*   **$x_{t:1}$**: Merepresentasikan sejarah token masa lalu (*history of past tokens*). Ini dibaca sebagai: "Semua kata yang telah dilihat model dari awal kalimat (indeks 1) hingga posisi saat ini (indeks $t$)". Urutannya adalah $x_t, \dots, x_1$.
*   **$x_{t+1}$**: Adalah token target masa depan (satu kata persis di depan posisi saat ini).
*   **$\log P_\theta(x_{t+1} \mid x_{t:1})$**: Ini adalah probabilitas logaritmik bersyarat. Model sedang ditanya: "Diberikan seluruh teks sejarah ($x_{t:1}$), seberapa yakin Anda (berapa probabilitasnya) bahwa kata selanjutnya adalah $x_{t+1}$?". 
*   **Tanda Minus ($-$)**: Ditambahkan karena nilai probabilitas berkisar dari 0 hingga 1, yang mana nilai logaritmanya adalah negatif. Tanda minus menjadikannya nilai positif (*Loss*), di mana semakin tinggi probabilitas yang ditebak model, semakin kecil nilai *Loss*-nya.

Dalam karya ini, para peneliti menggeneralisasi pendekatan di atas dengan mengimplementasikan tugas prediksi multi-token. Pada setiap posisi di korpus pelatihan, model diinstruksikan untuk memprediksi **$n$ buah token masa depan sekaligus**. Hal ini secara matematis ditranslasikan ke dalam fungsi kerugian lintas-entropi baru:

$$ L_n = - \sum_t \log P_\theta(x_{t+n:t+1} \mid x_{t:1}). \quad \quad \text{(2)} $$

**Eksplorasi Mendalam Rumus (2): Fungsi Kerugian Multi-Token**
*   **$L_n$**: Kerugian untuk memprediksi $n$-token ke depan secara serentak.
*   Perbedaan utama ada pada **$x_{t+n:t+1}$**. Alih-alih hanya satu kata $x_{t+1}$, model kini diukur kemampuannya untuk menebak seluruh blok kata di masa depan, mulai dari $x_{t+1}$ berlanjut hingga $x_{t+n}$.

Untuk membuat komputasi ini dapat dikelola (*tractable*), diasumsikan bahwa model bahasa besar $P_\theta$ menggunakan **batang utama yang digunakan bersama (*shared trunk*)** untuk menghasilkan sebuah representasi laten (pemahaman abstrak) $z_{t:1}$ dari konteks $x_{t:1}$ yang diobservasi. 

Representasi laten $z_{t:1}$ ini kemudian **diumpankan ke dalam $n$ buah kepala (*heads*) independen** untuk memprediksi secara paralel setiap $n$ token masa depan (seperti yang diilustrasikan pada Gambar 1 tadi). Asumsi arsitektur ini mengarah pada faktorisasi (pemecahan) fungsi kerugian lintas-entropi prediksi multi-token sebagai berikut:

$$ L_n = - \sum_t \log P_\theta(x_{t+n:t+1} \mid z_{t:1}) \cdot P_\theta(z_{t:1} \mid x_{t:1}) $$
$$ = - \sum_t \sum_{i=1}^n \log P_\theta(x_{t+i} \mid z_{t:1}) \cdot P_\theta(z_{t:1} \mid x_{t:1}). $$

**Penjelasan Matematis Faktorisasi di atas:**
Karena kepala-kepala keluaran (*output heads*) tersebut memprediksi kata-kata masa depan secara paralel dan **saling independen satu sama lain** berdasarkan representasi laten yang sama ($z_{t:1}$), maka probabilitas gabungan dari seluruh blok kata masa depan ($x_{t+n:t+1}$) dapat dipecah menjadi perkalian probabilitas prediksi masing-masing kata secara individual. Dalam ranah logaritma ($\log$), operasi perkalian probabilitas ini berubah menjadi operasi **penjumlahan ($\sum_{i=1}^n$)**. 

Dalam praktiknya, arsitektur ini terdiri dari:
1.  Sebuah batang transformer bersama (*shared transformer trunk*) $f_s$ yang memproduksi representasi tersembunyi (*hidden representation*) $z_{t:1}$ dari konteks yang diobservasi $x_{t:1}$.
2.  $n$ buah kepala keluaran independen yang diimplementasikan dalam bentuk lapisan-lapisan transformer (transformer layers) $f_{h_i}$.
3.  Sebuah matriks *unembedding* (matriks yang mengubah vektor laten kembali menjadi probabilitas kata/kosakata) $f_u$ yang digunakan secara bersama. 

Oleh karena itu, untuk memprediksi $n$ token masa depan, perhitungan model adalah:

$$ P_\theta(x_{t+i} \mid x_{t:1}) = \text{softmax}(f_u(f_{h_i}(f_s(x_{t:1})))), $$

untuk $i = 1, \dots n$. Di mana, secara khusus, $P_\theta(x_{t+1} \mid x_{t:1})$ (di mana $i=1$) adalah kepala prediksi token-berikutnya konvensional kita. 

### Implementasi Hemat-Memori (*Memory-efficient implementation*)

Satu tantangan raksasa dalam melatih prediktor multi-token adalah **bagaimana cara mengurangi utilisasi memori GPU mereka**. 

Untuk memahami mengapa ini menjadi masalah besar, ingatlah bahwa dalam LLM modern, ukuran kosakata (*vocabulary size*) $V$ jauh, jauh lebih besar daripada dimensi $d$ dari representasi laten. (Sebagai contoh, memori untuk menebak 1 kata dari 50.000 kemungkinan kata membutuhkan ruang penyimpan matriks yang besar). Oleh karena itu, vektor *logit* (nilai probabilitas sebelum softmax) menjadi sumbatan (*bottleneck*) penggunaan memori GPU. 

Implementasi naif (polos/standar) dari prediktor multi-token yang mewujudkan (materialize) *seluruh* logit dan gradiennya untuk semua $n$ kepala secara serentak—yang memiliki bentuk matriks raksasa $(n, V)$—akan secara parah membatasi ukuran kelompok (*batch-size*) yang diizinkan dan rata-rata utilisasi memori GPU. Sistem akan dengan cepat kehabisan VRAM (Out of Memory/OOM).

Karena alasan ini, di dalam arsitektur mereka, para peneliti mengusulkan untuk **mengadaptasi urutan operasi komputasi maju (*forward*) dan mundur (*backward*) secara hati-hati**.

### Analisis Gambar 2: Urutan Maju/Mundur pada Model Prediksi Multi-Token

*(Catatan Dosen: Mari kita bedah Gambar 2 untuk memahami trik rekayasa *software* yang brilian ini).*

**Deskripsi Gambar Kiri (Diagram Arsitektur Forward/Backward):**
*   Terdapat balok **"Shared"** berwarna biru gelap di bawah (Model Trunk). Panah panah menembus ke atas menuju dua cabang, yaitu **Head 1** dan **Head 2** (model ini $n=2$).
*   Di atas masing-masing kepala, terdapat fungsi kalkulasi **Loss 1** dan **Loss 2**.
*   **Panah Hijau ke Kanan (Forward):** Melambangkan aliran data saat menebak kata. Data mengalir dari *Shared*, masuk ke *Head 1*, dihitung *Loss 1*-nya. Lalu data kembali mengalir dari *Shared*, masuk ke *Head 2*, dihitung *Loss 2*-nya.
*   **Panah Merah ke Kiri (Backward):** Melambangkan proses pembaruan parameter (*backpropagation/gradient descent*).
*   **Kesimpulan Arsitektur:** Ini adalah triknya. Alih-alih menjalankan Head 1 dan Head 2 secara serentak (paralel), sistem ini menjalankan mereka **secara sekuensial (bergantian)**.

**Deskripsi Gambar Kanan (Potongan Kode Semu / Pseudocode):**
*   Kode mendemonstrasikan perulangan `for i in range(n):`. 
*   Di dalam perulangan, model menghitung probabilitas `p = model.heads[i](d)`.
*   Kemudian segera menghitung nilai kerugian dan langsung mengeksekusi perhitungan mundur `loss(p, y[i]).backward()`.
*   Gradien (nilai koreksi) dikumpulkan perlahan-lahan di dalam batang utama (`z.backward(...)`).
*   **Efek Keajaiban:** Karena dilakukan di dalam *loop* untuk setiap kepala secara bergantian, maka segera setelah *Head 1* selesai dihitung *Loss*-nya dan gradiennya dikirim ke bawah, **memori raksasa yang dipakai untuk menyimpan probabilitas matriks $(1, V)$ milik Head 1 tersebut langsung dihapus (freed) dari VRAM GPU**. Barulah sistem lanjut memproses Head 2.

Secara spesifik, setelah proses operan maju (*forward pass*) melewati batang utama $f_s$, kita **secara sekuensial (bergantian)** menghitung operan maju dan mundur dari setiap kepala keluaran independen $f_i$, mengakumulasikan gradien-gradien tersebut di batang utama. Meskipun proses ini menciptakan *logits* (dan gradiennya) untuk kepala keluaran $f_i$, memori dari variabel-variabel ini **langsung dibebaskan (freed)** sebelum melanjutkan ke kepala keluaran berikutnya $f_{i+1}$. 

Dengan trik ini, satu-satunya variabel berukuran besar yang membutuhkan penyimpanan memori jangka panjang (*long-term storage*) hanyalah gradien dari batang utama berdimensi $d$, yaitu $\partial L_n / \partial f_s$. 

Kesimpulannya, metode ini berhasil mereduksi puncak utilisasi memori GPU secara drastis dari **$\mathcal{O}(nV + d)$** turun drastis menjadi hanya **$\mathcal{O}(V + d)$**, tanpa adanya biaya pinalti terhadap kecepatan waktu tempuh komputasi (*runtime*). Puncak memori yang dibutuhkan sama persis dengan melatih model prediksi 1-token konvensional!

### Inferensi (*Inference*)

Selama waktu inferensi (saat model digunakan oleh pengguna), cara paling dasar untuk menggunakan arsitektur ini adalah **prediksi autoregresif token-berikutnya konvensional** (menebak satu kata demi satu kata) murni hanya menggunakan kepala prediksi token-berikutnya $P_\theta(x_{t+1} \mid x_{t:1})$. Semua kepala prediksi masa depan lainnya (Head 2, 3, 4) **dibuang (discarded)** dan tidak dipakai sama sekali. Dengan cara ini, tidak ada biaya komputasi tambahan.

Namun, alih-alih dibuang, kepala keluaran tambahan tersebut ternyata bisa **dimanfaatkan untuk mempercepat proses dekoding (pembuatan teks)** dari kepala prediksi token-berikutnya dengan menggunakan metode **dekoding berspekulasi-mandiri (*self-speculative decoding*)** seperti dekoding paralel berbasis blok (Stern et al., 2018). 

*Speculative decoding* adalah sebuah varian dari teknik di mana model mencoba "menebak" atau "berspekulasi" beberapa kata sekaligus di masa depan. Biasanya teknik ini membutuhkan model draft tambahan yang lebih kecil untuk menebak, lalu model besar memverifikasi tebakan tersebut (Leviathan et al., 2023). Namun berkat arsitektur multi-token ini, model bisa bertindak sebagai *drafter* (pembuat draf spekulasi) untuk dirinya sendiri tanpa memerlukan draf model tambahan. Teknik ini sangat mirip dengan percepatan *speculative decoding* pada arsitektur Medusa dengan atensi pohon (Cai et al., 2024).

---

## 3. Eksperimen pada Data Nyata (*Experiments on real data*)

Para peneliti mendemonstrasikan kemanjuran fungsi kerugian prediksi multi-token ini melalui tujuh eksperimen berskala besar. 
*   **Bagian 3.1** menunjukkan bagaimana prediksi multi-token menjadi semakin berguna seiring dengan perbesaran ukuran model. 
*   **Bagian 3.2** menunjukkan bagaimana kepala prediksi tambahan dapat mempercepat proses inferensi hingga $3\times$ lipat dengan *speculative decoding*. 
*   **Bagian 3.3** mendemonstrasikan bahwa prediksi multi-token mendorong pembelajaran pola jangka panjang, sebuah fakta yang paling nyata terlihat dalam kasus ekstrem tokenisasi tingkat-byte (*byte-level tokenization*). 
*   **Bagian 3.4** menunjukkan bahwa prediktor 4-token memberikan keuntungan yang sangat kuat bila disandingkan dengan *tokenizer* berukuran kosa kata 32k. 
*   **Bagian 3.5** mengilustrasikan bahwa manfaat ini bertahan bahkan untuk sesi pelatihan yang dijalankan berulang-ulang (*multiple epochs*). 
*   **Bagian 3.6** memamerkan kayanya representasi memori yang didapatkan oleh model dari *pretraining* multi-token ini ketika model tersebut kemudian di-*finetune* (disesuaikan) secara khusus untuk kompetisi pemrograman (dataset CodeContests). 
*   **Bagian 3.7** menunjukkan bahwa manfaat ini juga menular kepada model-model bahasa alami (untuk teks manusia), meningkatkan metrik evaluasi generatif seperti peringkasan (*summarization*), tanpa mengalami regresi/penurunan performa yang signifikan pada tolok ukur *benchmark* pilihan ganda.

Untuk memastikan perbandingan yang 100% adil antara prediktor token-berikutnya (1-token) dengan prediktor $n$-token, semua eksperimen yang dilakukan di bawah ini **selalu membandingkan model-model yang memiliki jumlah parameter total yang persis sama**. Artinya, ketika peneliti menambahkan lapisan transformer untuk membangun $n-1$ kepala prediksi masa depan, mereka dengan adil "mencabut" $n-1$ buah lapisan transformer dari dalam batang model utama (*shared model trunk*).

### 3.1 Manfaat berskala sejalan dengan ukuran model (*Benefits scale with model size*)

Untuk mempelajari fenomena ini, peneliti melatih model dalam enam ukuran yang berbeda (berkisar antara 300 Juta hingga 13 Miliar parameter) dan dilatih dari awal (*from scratch*) menggunakan sekurang-kurangnya 91 Miliar token berupa kode pemrograman. Hasil evaluasi disajikan pada Gambar 3.

### Analisis Gambar 3: Hasil Model Prediksi n-token pada MBPP Berdasarkan Ukuran Model

*(Deskripsi Naratif Gambar 3: "Results of n-token prediction models on MBPP by model size")*
*   Gambar ini berisi sekumpulan grafik batang (*bar charts*) yang disusun secara terpisah untuk 3 metrik yang berbeda: **Pass@1** (sebelah kanan atas), **Pass@10** (tengah kanan), dan **Pass@100** (bawah kanan). 
*   Sumbu X (horizontal) pada semua grafik menunjukkan ukuran model yang diuji: `0.3B, 0.6B, 1.3B, 3B, 6.7B, 13B`.
*   Sumbu Y (vertikal) di setiap grafik batang adalah rentang selisih skor performa melawan *baseline* (angka negatif hingga positif).
*   **Grafik Pass@1 (Tingkat kelulusan percobaan pertama):**
    *   Pada ukuran 0.3B dan 0.6B, batang berarah turun (negatif), menandakan model n-token kalah dengan baseline. Nilainya berkisar di angka -1.0 hingga -2.0.
    *   Mulai ukuran 1.3B hingga 13B, batang berbalik arah naik ke atas (positif). Semakin besar model, semakin tinggi batangnya.
    *   Puncaknya pada model 13B, batangnya menembus peningkatan yang signifikan yaitu **+5.0** poin (atau sekitar 17% lebih banyak soal terpecahkan).
*   **Pola yang Konsisten pada Pass@10 dan Pass@100:**
    *   Pola pembalikan yang sama persis terjadi ketika model diberikan kelonggaran 10 kali tebakan (Pass@10) atau 100 kali tebakan (Pass@100).
    *   Pada Pass@100, peningkatan performanya jauh lebih liar. Pada model 13B, model n-token menyumbangkan kenaikan performa hingga mendekati **+7.5** poin absolut.
*   **Kesimpulan Akademis Keseluruhan Gambar 3:** Pengujian ini menggunakan dataset MBPP (Kumpulan soal dasar Python) dan HumanEval (Soal algoritma kompleks). Eksperimen ini menjadi bukti pamungkas bahwa **hanya dengan modal anggaran komputasi pelatihan yang SAMA PERSIS dan dataset yang SAMA PERSIS**, kita bisa memeras dan memeras jauh lebih banyak kecerdasan dan performa dari dalam sebuah model bahasa besar, asalkan kita menggunakan fungsi kerugian prediksi multi-token. 

Peneliti meyakini bahwa fakta di mana "kebermanfaatan ini hanya terlihat pada skala besar" adalah alasan utama mengapa pendekatan prediksi multi-token ini selama ini sebagian besar **diabaikan** oleh para akademisi lain sebagai sebuah kerugian fungsi (loss) pelatihan yang menjanjikan. (Riset akademis skala kecil di masa lalu mungkin mencobanya pada model kecil dan langsung menyerah karena hasilnya malah memburuk).

### 3.2 Inferensi yang Lebih Cepat (*Faster inference*)

Para peneliti mengimplementasikan dekoding berspekulasi-mandiri murni (*greedy self-speculative decoding*) dengan ukuran kelompok (*batch sizes*) yang heterogen menggunakan pustaka *xFormers*. Mereka mengukur kecepatan dekoding (kecepatan menghasilkan teks) dari model prediktor 4-token terbaik mereka yang berkapasitas 7B parameter, saat model ditugaskan melengkapi *prompt* dari dataset pengujian kode dan bahasa alami yang belum pernah dilihat selama masa pelatihan.

Mereka mengamati adanya percepatan kecepatan dekoding hingga **3,0 kali lipat lebih cepat pada teks kode** (dengan rata-rata 2,5 token tambahan yang diterima oleh verifikator dari 3 tebakan spekulasi token masa depan yang diajukan oleh kepala model). Sedangkan pada pemrosesan teks biasa, kecepatannya meningkat hingga **2,7 kali lipat**.

Pada pengujian ekstrem model yang beroperasi di tingkat Byte (di mana model memprediksi 8-byte huruf ke depan), akselerasi kecepatan inferensinya mencapai **6,4 kali lipat**. Fakta ini membuktikan bahwa Pra-pelatihan dengan multi-token tidak hanya memperkuat kepintaran logika model, namun kepala prediksi tambahan (head 2, 3, 4) dilatih menjadi tebakan bayangan yang sangat akurat, jauh melampaui kemampuan model standar jika di *fine-tune* dangkal. Hal ini mengizinkan model untuk membuang kunci gembok kecepatan inferensi sepenuhnya.

---

### Analisis Tabel 1: Prediksi Multi-Token Meningkatkan Performa dan Membuka Jalur Pelatihan Tingkat-Byte yang Efisien

*(Catatan Dosen: Mari kita baca metrik performa komparatif yang disajikan di Tabel 1).*

| Data Pelatihan | Kosakata | $n$ | MBPP (@1, 10, 100) | HumanEval (@1, 10, 100) | APPS/Intro (@1, 10, 100) |
| :--- | :--- | :---: | :--- | :--- | :--- |
| **313B byte** (0.5 epochs) | *bytes* (huruf) | 1 | 19.3 / 42.4 / 64.7 | 18.1 / 28.2 / 47.8 | 0.1 / 0.5 / 2.4 |
| | | 8 | **32.3** / **50.0** / **69.6** | **21.8** / **34.1** / **57.9** | **1.2** / **5.7** / **14.0** |
| | | 16 | 28.6 / 47.1 / 68.0 | 20.4 / 32.7 / 54.3 | 1.0 / 5.0 / 12.9 |
| | | 32 | 23.0 / 40.7 / 60.3 | 17.2 / 30.2 / 49.7 | 0.6 / 2.8 / 8.8 |
| **200B token** (0.8 epochs) | 32k token | 1 | 30.0 / 53.8 / 73.7 | 22.8 / 36.4 / 62.0 | 2.8 / 7.8 / 17.4 |
| | | 2 | 30.3 / 55.1 / 76.2 | 22.2 / 38.5 / 62.6 | 2.1 / 9.0 / 21.7 |
| | | 4 | **33.8** / **55.9** / **76.9** | **24.0** / **40.1** / **66.1** | 1.6 / 7.1 / 19.9 |
| | | 6 | 31.9 / 53.9 / 73.1 | 20.6 / 38.4 / 63.9 | **3.5** / **10.8** / **22.7** |
| | | 8 | 30.7 / 52.2 / 73.4 | 20.0 / 36.6 / 59.6 | 3.5 / 10.4 / 22.1 |
| **1T token** (4 epochs) | 32k token | 1 | 40.7 / 65.4 / 83.4 | 31.7 / 57.6 / 83.0 | **5.4** / **17.8** / **34.1** |
| | | 4 | **43.1** / 65.9 / 83.7 | 31.6 / 57.3 / **86.2** | 4.3 / 15.6 / 33.7 |

**Eksplorasi Data Akademis Tabel 1:**
1.  **Pengujian Tingkat Byte Ekstrem (Baris 313B bytes):** Pada pemodelan berbasis byte (di mana mesin menebak satu huruf demi satu huruf secara harfiah, bukan per kata), performa prediksi 1-huruf (n=1) sangat menyedihkan (MBPP 19.3%). Namun lihat lonjakannya ketika diubah menjadi menebak 8-huruf sekaligus (n=8)! Performanya meledak menjadi **32.3%** di MBPP pass@1 (melonjak +13 poin).
2.  **Mencari Titik Optimal $n$ (Baris 200B token):** Pada sistem tokenisasi standar berukuran 32k kata, peneliti melakukan ablasi (uji coba pembongkaran parameter) untuk $n = 1, 2, 4, 6, 8$.
    *   Ternyata, model dengan $n=4$ (memprediksi 4 kata ke depan sekaligus) menjadi **titik manis (sweet spot)** dan secara konsisten mengalahkan semua model lain di sekujur papan ujian (HumanEval dan MBPP). Peningkatan pada MBPP pass@1 adalah +3.8% poin, dan pada HumanEval pass@1 adalah +1.2%.
    *   Menariknya, pada soal algoritma APPS/Intro, model $n=6$ justru memimpin kemenangan (+0.7%). Ini menandakan bahwa ukuran jendela prediksi ideal (optimal window size) sangat bergantung pada karakteristik kerumitan distribusi data masukan.
3.  **Daya Tahan Pelatihan Multi-Epoch (Baris 1T token):** Bahkan ketika dilatih dengan brutal hingga ukuran 1 Triliun token melintasi 4 siklus berulang (epochs), prediksi 4-token ($n=4$) tetap mempertahankan dominasinya (menang +2.4% pada MBPP pass@1).

---

### 3.6 Penyesuaian Halus (*Finetuning*) Prediktor Multi-Token

Peneliti juga mengevaluasi apakah model pra-latih yang otaknya dibentuk oleh kerugian prediksi multi-token ini akan mempertahankan superioritasnya jika kelak ditugaskan untuk melakukan *finetuning* (pelatihan penyesuaian khusus tugas hilir). Mereka mengujinya dengan mem-finetune model 7B parameter pada dataset khusus kompetisi algoritma (CodeContests).

### Analisis Gambar 4: Perbandingan Performa Finetuning pada CodeContests

*(Deskripsi Naratif Gambar 4: "Comparison of finetuning performance on CodeContests")*
*   **Sumbu X (Logaritmik):** Nilai $k$ untuk probabilitas kelulusan (mulai dari *pass@1* hingga *pass@1000* tebakan).
*   **Sumbu Y:** Akurasi *pass@k* dalam presentase (%) (dari 0.2 hingga melampaui 10.0).
*   **Kurva Data:**
    *   Garis oranye lurus pudar: $n=1, n'=1$. Ini adalah "Baseline". Model dipra-latih memprediksi 1 kata, dan di-*finetune* memprediksi 1 kata.
    *   Garis hitam putus-putus: $n=4, n'=1$. Ini adalah skenario di mana model dipra-latih menggunakan multi-token (4 kata) lalu "kepala-kepala masa depan" dicopot/dilucuti, dan di-*finetune* kembali menggunakan kerugian konvensional (1 kata) pada target.
    *   Garis hijau lurus: $n=4, n'=4$. Model dipertahankan arsitektur multi-token-nya secara utuh hingga tahap *finetuning*.
*   **Kesimpulan Akademis Gambar 4:** Menariknya, metode finetuning garis putus-putus hitam (Pra-latih multi-token $\rightarrow$ Finetune token-tunggal) muncul sebagai metodologi **terbaik dan paling superior**. Pendekatan hibrida ini rupanya mewariskan wawasan memori holistik dan pondasi kognisi mendalam dari model multi-token, namun memanfaatkan kefokusan dari model konvensional untuk tugas hilir.

---

### 3.7 Prediksi Multi-Token pada Bahasa Alami (Teks Manusia)

Untuk mengevaluasi teknik ini pada bahasa manusia (bukan kode pemrograman), peneliti melatih model 7B pada 200 Miliar token teks natural. Evaluasi dilakukan pada 6 metrik *benchmark* pemahaman teks Pilihan Ganda standar (seperti ARC, COPA, PIQA).

### Analisis Gambar 5: Pelatihan Multi-Token pada Tugas Pilihan Ganda

*(Deskripsi Naratif Gambar 5: "Multi-token training with 7B models doesn't improve performance on choice tasks")*
*   **Sumbu X:** Training step (Iterasi pelatihan) dari 5000 hingga 25000.
*   **Sumbu Y:** Akurasi Rata-rata dari 6 pengujian NLP konvensional (40.0 hingga 52.5).
*   **Kurva Data:** Tiga garis berlomba ke atas. $n=1$ (oranye lurus), $n=2$ (garis titik-titik hitam), dan $n=4$ (garis hijau titik kecil).
*   **Kesimpulan Mengecewakan (Namun dapat dipahami):** Pada grafik ini, terlihat bahwa model multi-token gagal memberikan nilai tambah pada ujian "Pilihan Ganda" ini. Garis $n=2$ menempel dengan *baseline*, sementara garis $n=4$ justru merosot secara signifikan (degradasi performa).

Namun, para peneliti sangat menekankan argumen akademis ini: Mereka **TIDAK PERCAYA** bahwa ujian pilihan ganda (yang semata berbasis penghitungan *likelihood/probabilitas*) merupakan arena pengujian yang pas dan valid untuk menilai "Kecerdasan Generatif" (kemampuan merangkai paragraf) yang sesungguhnya dari sebuah Model Bahasa. Pilihan ganda adalah tes probabilitas yang statis.

Sebagai gantinya, mereka menantang model pada arena tugas **Peringkasan Abstrak (*Abstractive Summarization*)** dan **Matematika Bahasa Alami (*GSM8K*)**, karena disinilah koherensi pembuatan teks bebas diukur. 

### Analisis Gambar 6: Performa Peringkasan Abstrak Teks (ROUGE-L)

*(Deskripsi Naratif Gambar 6: "Performance on abstractive text summarization")*
*   **Sumbu X:** Terbagi menjadi dua kategori ukuran set pelatihan: 200 (Miliar token) dan 500 (Miliar token).
*   **Sumbu Y:** Nilai rata-rata F1 Score metrik ROUGE-L (menghitung seberapa mirip urutan kata ringkasan AI dengan ringkasan buatan pakar Manusia).
*   **Bilah/Batang:** Batang Oranye Pudar ($n=1$ / Baseline), Batang Biru Gelap ($n=2$), dan Batang Hijau Pudar ($n=4$).
*   **Kesimpulan Akademis Gambar 6:** Pada tugas yang menuntut kemampuan merangkai cerita dan meringkas teks ini, terlihat dominasi mutlak dari arsitektur multi-token. Batang biru dan hijau **secara tajam memimpin** dan mengangkangi batang oranye (baseline konvensional) baik pada pelatihan 200B maupun 500B token. Arsitektur multi-token secara bawaan memaksa model untuk memandang kalimat dari "mata burung" (perspektif jangka panjang), sehingga intisari peringkasan jauh lebih solid.

---

## 4. Ablasi pada Data Sintetis (*Ablations on synthetic data*)

Apa sebenarnya mekanisme gaib yang mendorong lompatan performa luar biasa pada model prediksi multi-token ini? 

Untuk menguak rahasia ini, para peneliti merancang dua eksperimen mainan (*toy experiments*) dengan tingkat presisi kontrol yang ketat:
1.  **Kemampuan Induksi (*Induction capability*)** pada model ukuran mini.
2.  **Penalaran Algoritmik (*Algorithmic reasoning*)**.

### 4.1 Kemampuan Induksi (*Induction capability*)

*Induction* (Induksi) menggambarkan sebuah pola berpikir logika di mana sistem melengkapi kepingan pola berdasarkan kejadian lampaunya (Olsson et al., 2022). Secara sederhana: Jika di awal dokumen tertulis frasa "Maling Kundang", maka ketika di paragraf selanjutnya mesin mendeteksi kata "Maling", mesin harus secara induktif meramal bahwa kata berikutnya hampir pasti adalah "Kundang".

Peneliti merancang sebuah ujian di mana nama-nama karakter di dalam buku dongeng anak-anak disensor dan diganti secara total menggunakan sepasang token (2-kata) fiktif yang dibangkitkan secara acak. Tugas model adalah menebak "kata kedua" dari nama aneh tersebut saat karakter tersebut muncul untuk kedua kalinya di cerita. 

### Analisis Gambar 7: Kemampuan Induksi pada Model Prediksi n-token

*(Deskripsi Naratif Gambar 7: "Induction capability of n-token prediction models")*
*   **Sumbu X (Logaritmik):** Ukuran Parameter model non-embedding dari super kerdil 1 juta (1M) hingga lumayan besar 1000M (1B).
*   **Sumbu Y:** "Induction success" (Tingkat kesuksesan induksi). Rentang 0.0 hingga 0.5. (Tingkat keberhasilan model menebak identitas nama belakang fiktif).
*   **Kurva Data:** Garis Oranye dengan silang merah ($n=1$ / Baseline konvensional). Garis Biru Putus-putus dengan silang hitam ($n=2$ / Prediksi ganda).
*   **Kesimpulan Investigasi Kognitif:** Ini adalah penemuan yang mencengangkan. Perhatikan pada model berukuran sangat kecil (1M hingga 30M parameter). Garis oranye konvensional tertelungkup mati di dasar mendekati angka nol (model gagal total memiliki nalar memori induksi). Namun, secara ajaib, garis biru putus-putus ($n=2$) **meloncat drastis** membangun kesadaran induktif hingga angka 0.3! 
*   **Argumen Peneliti:** Melatih model dengan memaksa ia menebak banyak kata ke depan secara langsung memaksa jaringan saraf model untuk mentransfer dan mendistribusikan "peta informasi antar-posisi urutan kata" secara prematur ke dalam anatomi pelatihannya. Inilah cikal bakal terbentuknya apa yang disebut sirkuit **kepala induksi (induction heads)**. 

### 4.2 Penalaran Algoritmik (*Algorithmic reasoning*)

Tugas penalaran algoritmik memungkinkan kita untuk mengukur bentuk penalaran dalam-konteks (*in-context reasoning*) yang tingkatnya jauh lebih rumit daripada induksi kata sederhana. Peneliti melatih dan mengevaluasi model pada tugas menyelesaikan **aritmatika polinomial**. 
Model diberikan tugas penjumlahan, perkalian, negasi, dan komposisi dari fungsi persamaan polinomial yang panjang. Operasi-operasi digenerasi secara acak dengan rentang kesulitan ($m$ jumlah operasi) dari 1 hingga 5.

Model diuji pada rentang *In-domain* (soal diulang-ulang dengan panjang 1-5 operasi) dan rentang *Out-of-domain* (model ditodong dengan soal panjang ekstrem dengan jumlah operasi 6 hingga 10, yang tak pernah dilihatnya saat masa pelatihan sekolahnya). Model sengaja diciptakan dengan postur kecil (hanya 30M dan 100M parameter) untuk mensimulasikan kepayahan LLM raksasa yang tidak mungkin menghafal (memorize) seluruh internet.

### Analisis Gambar 8: Akurasi Tugas Aritmatika Polinomial Terhadap Variasi Kesulitan

*(Deskripsi Naratif Gambar 8: "Accuracy on a polynomial arithmetic task with varying number of operations per expression")*
*   **Sumbu X:** Jumlah operasi aritmatika per soal (1, 2, 3... 5) adalah *in-domain*. (6, 7... 10) adalah zona asing *out-of-domain*. Semakin ke kanan, soal semakin absurd panjangnya.
*   **Sumbu Y:** Tingkat Akurasi mesin (%).
*   **Kurva Data:** Garis Oranye bersilang merah ($n=1$), Garis Hitam putus ($n=2$), Garis Biru dengan silang hijau ($n=4$). Terdapat sebaran *scatter plot* untuk dua sesi lari (run) independen.
*   **Kesimpulan Sains Eksak:** Mari amati jurang perbedaan yang terjadi. Pada soal *in-domain* (1-5 operasi), model multi-token secara meyakinkan melampaui kemampuan model standar (batas akurasinya diangkat ke atas). Namun, kehebatan sesungguhnya ada pada wilayah *out-of-domain* (6-10 operasi). Garis $n=2$ dan $n=4$ memiliki umur panjang yang lebih tinggi dan landai. Pelatihan multi-token membuktikan eksistensinya dalam **membangun daya generalisasi logika yang kuat ke zona-zona persoalan yang tidak dikenal**.

---

## 5. Mengapa Ini Bisa Berhasil? Beberapa Spekulasi Teoretis (*Why does it work? Some speculation*)

Mengapa model bahasa tiba-tiba menjadi sangat ahli dalam *coding* (pemrograman) atau tugas penalaran ketika kita hanya memaksa mereka menebak 4 kata sekaligus?

Intuisi yang dibangun oleh para peneliti pada bab ini adalah: Prediksi multi-token sesungguhnya **memitigasi (memperbaiki) celah perbedaan distribusi (distributional discrepancy)** antara metodologi pemaksaan-guru (*teacher forcing*) saat fase pelatihan, dengan proses generasi autoregresif yang berjalan buta tanpa guru pada saat model dilepas di fase inferensi. 

Mereka membedah pandangan ini melalui dua argumen filosofis dan matematis: bobot implisit pada titik-titik persimpangan krusial, dan dekomposisi berbasis teori informasi.

### 5.1 Peninjauan ke Depan (*Lookahead*) Memperkuat Titik Pilihan Krusial (*Choice Points*)

Di dalam sebuah teks atau kode program, tidak semua kata (token) itu sama penting kepentingannya. 
*   Beberapa token bersifat remeh-temeh ("inconsequential"), misalnya variasi gaya bahasa yang maknanya sama saja, atau huruf sambung `\n` dan spasi di Python.
*   Namun, beberapa token lainnya secara mutlak adalah **Titik Pilihan (*Choice points*)**. Titik-titik pilihan ini memegang kendali semantik atas struktur dan nasib alur teks selanjutnya, dan salah tebak di titik ini akan menyebabkan seluruh logika kode pemrograman ancur (derailing).

Pemodelan bahasa tradisional (tebak 1 kata konvensional) menggunakan metode *Teacher-Forcing* (memaksa model menebak kata berdasarkan kunci jawaban 1 kata persis di depan matanya). Model menjadi sangat manja dan rabun jauh. Model lebih mementingkan memprediksi secara baik dalam rentang waktu yang *sangat pendek*, dan abai akan ancaman malapetaka jika ia merusak struktur paragraf secara besar. 

### Analisis Gambar 9: Prediksi Multi-Token Memberikan Bobot Implisit Tinggi pada Token Krusial

*(Deskripsi Naratif Gambar 9: "Multi-token prediction loss assigns higher implicit weights to consequential tokens")*
*   **Ilustrasi:** Menampilkan jejeran bulatan token masa lalu "1 $\rightarrow$ 2 $\rightarrow$ 3 $\rightarrow$ 4 $\rightarrow$ 5" (berwarna hijau). Pada posisi 5, terjadi titik cabang krusial yang menentukan apakah teks akan menuju A atau B. Dalam gambar ini nasib teks harus menuju titik persimpangan "A $\rightarrow$ B".
*   **Mekanisme Pemaksaan:** Dalam arsitektur prediksi 3-token (ditunjukkan pada baris atas), model dipaksa menebak 3 kata serentak:
    *   Saat model berada di kata "3", ia dituntut menebak: "4", "5", "**A**".
    *   Saat model berada di kata "4", ia dituntut menebak: "5", "**A**", "B".
    *   Saat model berada di kata "5", ia dituntut menebak: "**A**", "B", "C".
*   **Rahasia Matematika Terungkap:** Perhatikan bagaimana arsitektur 3-token ini memaksa kata/token kunci "**A**" untuk ditebak dan di-evaluasi *error*-nya secara bertubi-tubi berulang kali sebanyak 3 kali kesempatan pada posisi yang berbeda-beda sebelum kata "A" itu benar-benar tiba! Kata "A" dipaksa dievaluasi sejak posisi "3", "4", dan "5". 
*   **Kesimpulan:** Karena konsekuensi dari masa depan yang sulit (seperti kode logika *if/else*) itu harus diprediksi dari jauh-jauh hari, titik-titik transisi yang menentukan nasib teks ini *secara gaib* menyedot bobot perhatian (*implicit weight*) secara asimetris jauh lebih besar dan menyita perhitungan Loss arsitekturnya secara masif dibandingkan model rabun jauh 1-kata. Algoritma dipaksa menjadi seorang pemain Catur yang memprediksi 4 langkah ke depan.

### 5.2 Argumen Berbasis Teori Informasi (*Information-theoretic argument*)

Ini adalah inti pembuktian matematisnya. 
*   Misalkan **$X$** adalah token kata yang akan datang (pertama).
*   Misalkan **$Y$** adalah token kata masa depan setelahnya (kedua).
*   Proses memprediksi $X$ di model standar disimbolkan sebagai Entropi $H(X)$. 
*   Prediksi $2$-token secara paralel bertujuan menghitung total Entropi masa depan: $H(X) + H(Y)$.

Secara Hukum Matematika Teori Informasi yang kaku, para peneliti mendekomposisi dan memecah dua nilai entropi absolut tersebut menjadi rumus persamaan terpisah:

$$ H(X) = H(X \mid Y) + I(X; Y), $$
$$ H(X) + H(Y) = H(X \mid Y) + 2I(X; Y) + H(Y \mid X). $$

*Penjelasan Rumus:*
*   $H(X \mid Y)$ adalah ketidakpastian memprediksi X jika kita sudah tahu masa depan Y.
*   $I(X; Y)$ adalah *Mutual Information* (Informasi Mutual), yakni seberapa krusial/mengikat relasi kepastian yang dibagi bersama antara keberadaan kata $X$ dengan kata $Y$.
*   Bandingkan baris rumus pertama dan kedua. Pada model konvensional, porsi nilai relasi *Mutual Information* $I(X;Y)$ hanyalah $1$. 
*   Namun ajaibnya, ketika model dituntut menebak dua token sekaligus (baris kedua rumus), suku (term) matematika dari *Mutual Information* **$I(X;Y)$ meledak secara matematis terkalikan dengan faktor angka 2 ($2I(X; Y)$)**. 

Ini adalah bukti absolut secara saintifik. Prediktor multi-token secara aljabar menuntut mesin jaringan saraf untuk mengasah akurasi tembakan prediksi-prediksi mereka pada token-token krusial $X$ yang *relavan untuk nasib struktur sisa kalimat panjang di masa depannya*!

---

## 7. Kesimpulan (*Conclusion*)

Di dalam makalah literatur fundamental ini, para pakar mengusulkan bahwa "Prediksi Multi-Token" (memaksa AI membaca dan meramal rentetan huruf/kata masa depan sekaligus) adalah sebuah titik loncat metodologis radikal yang mendobrak stagnasi dogma "prediksi-token-berikutnya" warisan kuno LLM generatif. 

Rangkaian eksperimen brutal (membedah model raksasa skala 7 Miliar parameter dan merambah hingga 1 Triliun ukuran data corpus token) memancarkan bukti kebenaran bahwa modifikasi ini secara asimetris menjadi semakin ampuh untuk model berukuran gigantik, melukiskan keunggulan performa absolut pada tugas pemrograman kompleks. 

Para peneliti menyimpulkan fenomena metafisik ini ke dalam satu pernyataan: Metode ini menghantam mati cacat perbedaan distribusi pembelajaran yang selama ini memisahkan simulasi "Teacher-Forcing" di meja latihan lab dan "Generasi Auto-Regressive" liar di dunia nyata. Lebih spektakulernya lagi, jika dirangkai dengan dekoding spekulatif, sang asisten penulis virtual bertransformasi mencetak respons barisan kalimat hingga 3 KALI LIPAT lebih cepat.

Di agenda riset masa depan, tim ini akan menyusun formula untuk membongkar teka-teki penentuan nilai $n$ prediksi secara otomatis. 

---
*(Catatan Dosen: Materi kita berakhir di sini. Pahami secara fundamental bagaimana modifikasi pada fungsi Loss dan teknik manipulasi Backward Pass bisa mengelabui memori kuadratik matriks, dan sadari bahwa dengan matematika Teori Informasi sederhana (hukum entropi mutual), sebuah model bisa dipaksa menjadi perencana strategi jangka panjang yang memecahkan kode pemrograman dengan lebih cerdas. Bagian Referensi dan Lampiran tidak disertakan sesuai instruksi. Selamat belajar).*