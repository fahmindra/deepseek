---
slug: ch-12-02
title: "GPT-2"
layout: chapter
---

## Abstrak (*Abstract*)

Tugas-tugas Pemrosesan Bahasa Alami (*Natural Language Processing* / NLP) seperti menjawab pertanyaan (*question answering*), penerjemahan mesin (*machine translation*), pemahaman bacaan (*reading comprehension*), dan peringkasan (*summarization*), secara tipikal didekati menggunakan metode Pembelajaran Terselia (*Supervised Learning*) pada dataset yang spesifik untuk tugas-tugas tersebut. 

Namun, dalam kajian ini, para peneliti mendemonstrasikan bahwa model bahasa (*language models*) mulai mampu mempelajari tugas-tugas ini **tanpa adanya pengawasan eksplisit (tanpa label khusus)** ketika dilatih pada sebuah dataset baru yang berisi jutaan halaman web, yang diberi nama **WebText**. 

Sebagai contoh, ketika model dikondisikan (diberikan konteks) berupa sebuah dokumen yang diikuti dengan pertanyaan, jawaban yang dihasilkan oleh model bahasa ini mencapai skor 55 F1 pada dataset CoQA. Skor ini setara atau bahkan melampaui performa dari 3 dari 4 sistem *baseline* (sistem acuan dasar) tanpa menggunakan satupun dari 127.000+ contoh data pelatihan yang ada pada CoQA. 

Kapasitas model bahasa merupakan faktor esensial bagi kesuksesan transfer tugas tembakan-nol (*zero-shot task transfer*). Terbukti bahwa peningkatan kapasitas model akan meningkatkan performa secara log-linear di berbagai tugas. Model terbesar yang diusulkan oleh peneliti, yang dinamakan **GPT-2**, adalah sebuah model Transformer dengan **1,5 Miliar (1.5B) parameter** yang mencapai hasil *state-of-the-art* (SOTA/paling mutakhir) pada 7 dari 8 dataset pengujian pemodelan bahasa dalam skenario *zero-shot*, meskipun model tersebut masih mengalami *underfitting* (belum sepenuhnya menghafal) terhadap dataset WebText. 

Sampel teks yang dihasilkan dari model ini merefleksikan peningkatan tersebut dan berisi paragraf-paragraf teks yang koheren. Temuan-temuan ini mengindikasikan jalur yang sangat menjanjikan dalam membangun sistem pemrosesan bahasa yang belajar melakukan berbagai tugas murni dari demonstrasi yang terjadi secara alami di dalam teks.

---

## 1. Pendahuluan (*Introduction*)

Sistem Pembelajaran Mesin (*Machine Learning* / ML) saat ini sangat unggul (secara ekspektasi) pada tugas-tugas yang memang secara spesifik dilatihkan kepada mereka. Keunggulan ini dicapai berkat kombinasi dari dataset berukuran besar, model berkapasitas tinggi, dan pembelajaran terselia (*supervised learning*). Namun, sistem-sistem ini sebenarnya sangat rapuh (*brittle*) dan sensitif terhadap perubahan sekecil apa pun pada distribusi data dan spesifikasi tugas. 

Oleh karena itu, sistem AI saat ini lebih tepat dikarakterisasi sebagai **"pakar yang sempit" (*narrow experts*)** daripada "generalis yang kompeten" (*competent generalists*). Tujuan ke depan dari bidang ini adalah bergerak menuju sistem yang lebih umum (general) yang dapat melakukan banyak tugas—yang pada akhirnya tidak lagi membutuhkan pembuatan dan pelabelan dataset pelatihan secara manual untuk setiap tugasnya.

Pendekatan dominan saat ini dalam menciptakan sistem ML adalah:
1. Mengumpulkan dataset berisi contoh pelatihan yang mendemonstrasikan perilaku benar untuk suatu tugas tertentu.
2. Melatih sistem untuk meniru perilaku tersebut.
3. Menguji performanya pada contoh data yang ditahan (*held-out examples*) yang bersifat independen dan terdistribusi secara identik (*Independent and Identically Distributed* / IID).

Pendekatan ini sangat berhasil untuk menciptakan *narrow experts*. Namun, perilaku yang sering kali tidak menentu (*erratic*) dari model pembuat takarir gambar (*captioning models*), sistem pemahaman bacaan, dan pengklasifikasi gambar ketika dihadapkan pada masukan yang beragam dan bervariasi, menyoroti beberapa kelemahan fatal dari pendekatan ini.

Peneliti mencurigai bahwa prevalensi (kebiasaan umum) melakukan **pelatihan tugas-tunggal (*single task training*) pada dataset berdomain-tunggal (*single domain datasets*)** adalah kontributor utama terhadap kurangnya kemampuan generalisasi pada sistem saat ini. Kemajuan menuju sistem yang tangguh (*robust*) menggunakan arsitektur saat ini kemungkinan besar membutuhkan pelatihan dan pengukuran performa pada rentang domain dan tugas yang sangat luas. Baru-baru ini, beberapa tolok ukur (*benchmarks*) telah diusulkan seperti GLUE dan decaNLP untuk mulai mempelajari hal ini.

Pembelajaran Multitugas (*Multitask learning*) adalah kerangka kerja yang menjanjikan untuk meningkatkan performa general ini. Akan tetapi, pelatihan multitugas dalam bidang NLP masih berada pada tahap awal (*nascent*). Pekerjaan sebelumnya hanya melaporkan peningkatan performa yang moderat, dan dua upaya paling ambisius hingga saat makalah ini ditulis hanya melatih pada total 10 dan 17 pasangan `(dataset, objective)` saja.

Dari sudut pandang Pembelajaran Meta (*Meta-learning*), setiap pasangan `(dataset, objective)` sebenarnya hanyalah **satu buah contoh pelatihan tunggal** yang diambil dari seluruh distribusi kemungkinan dataset dan objektif yang ada. Sistem ML saat ini membutuhkan ratusan hingga ribuan contoh untuk menginduksi fungsi yang dapat menggeneralisasi dengan baik. Hal ini mengisyaratkan bahwa pelatihan multitugas mungkin membutuhkan pasangan pelatihan fungsional dalam jumlah yang sama besarnya untuk dapat mewujudkan potensinya. Akan sangat sulit untuk terus menskalakan pembuatan dataset dan desain objektif secara manual (*brute force*) dengan teknik saat ini. Inilah yang memotivasi peneliti untuk mengeksplorasi pengaturan (setup) tambahan guna melakukan pembelajaran multitugas.

Sistem dengan performa terbaik saat ini untuk tugas bahasa menggunakan kombinasi antara pra-pelatihan (*pre-training*) dan penyesuaian halus terselia (*supervised fine-tuning*). Pendekatan ini memiliki sejarah panjang dengan tren yang bergerak menuju bentuk transfer yang lebih fleksibel:
* Pertama, representasi vektor kata (*word vectors*) dipelajari lalu digunakan sebagai masukan untuk arsitektur spesifik-tugas.
* Kemudian, representasi kontekstual dari jaringan saraf berulang (RNN) yang ditransfer.
* Terakhir, penelitian terbaru (merujuk pada GPT-1 dan BERT) menyarankan bahwa arsitektur spesifik-tugas tidak lagi diperlukan, dan mentransfer banyak blok *self-attention* saja sudah cukup.

Meskipun demikian, metode-metode tersebut **masih tetap membutuhkan pelatihan terselia** untuk dapat mengeksekusi suatu tugas. Di sisi lain, ketika hanya ada sedikit atau bahkan tidak ada data terselia, penelitian lain telah mendemonstrasikan potensi model bahasa untuk melakukan tugas spesifik, seperti penalaran akal sehat (*commonsense reasoning*) dan analisis sentimen.

Dalam makalah ini, peneliti menghubungkan dua garis besar penelitian tersebut dan melanjutkan tren metode transfer yang lebih umum. Peneliti mendemonstrasikan bahwa model bahasa dapat melakukan tugas-tugas turunan (*down-stream tasks*) dalam pengaturan **tembakan-nol (*zero-shot setting*)** — yakni murni tanpa modifikasi arsitektur maupun pembaruan parameter sama sekali. 

---

## 2. Pendekatan (*Approach*)

Inti dari pendekatan ini adalah pemodelan bahasa (*language modeling*). Pemodelan bahasa biasanya dibingkai sebagai estimasi distribusi probabilitas tak terselia (*unsupervised distribution estimation*) dari sekumpulan contoh $(x_1, x_2, ..., x_n)$ yang masing-masing terdiri dari urutan simbol dengan panjang yang bervariasi $(s_1, s_2, ..., s_n)$. 

Karena bahasa secara alami memiliki keterurutan sekuensial, sangat umum bagi kita untuk memfaktorkan probabilitas gabungan (*joint probabilities*) atas simbol-simbol tersebut sebagai produk dari probabilitas bersyarat (*conditional probabilities*):

$$ p(x) = \prod_{i=1}^{n} p(s_i|s_1, ..., s_{i-1}) \quad \quad \text{(1)} $$

> **Penjelasan Ekstensif Rumus (1):**
> *   $p(x)$ merepresentasikan probabilitas gabungan dari sebuah kalimat atau urutan teks penuh $x$.
> *   $\prod_{i=1}^{n}$ adalah notasi matematika untuk produk (perkalian beruntun) dari elemen ke-$i=1$ hingga ke-$n$.
> *   $p(s_i|s_1, ..., s_{i-1})$ adalah probabilitas bersyarat. Model memprediksi probabilitas munculnya simbol (kata/token) ke-$i$, yakni $s_i$, *dengan syarat* (diberikan konteks) seluruh simbol yang telah muncul sebelumnya dari indeks $1$ hingga $i-1$.
> *   *Kesimpulan Akademis:* Pendekatan pemfaktoran ini memungkinkan sistem untuk melakukan estimasi maupun pengambilan sampel (*sampling*) dari fungsi $p(x)$ secara *tractable* (dapat diselesaikan secara komputasi). Sistem ini bersifat *auto-regressive* (memprediksi masa depan berdasarkan masa lalu langkah demi langkah).

Dalam beberapa tahun terakhir, telah terjadi peningkatan signifikan dalam daya ekspresi model yang mampu menghitung probabilitas bersyarat ini, seperti arsitektur *self-attention* yaitu **Transformer**.

Belajar melakukan **satu tugas tunggal** dapat diekspresikan dalam kerangka probabilistik sebagai upaya mengestimasi distribusi bersyarat $p(\text{output}|\text{input})$. Namun, karena sebuah sistem yang umum harus mampu melakukan **banyak tugas yang berbeda**, bahkan pada input yang sama persis, model seharusnya tidak hanya dikondisikan pada *input*, tetapi juga pada *tugas* yang harus dilakukan. Artinya, model harus memodelkan persamaan:

$$ p(\text{output}|\text{input}, \text{task}) $$

Konsep pengkondisian tugas (*task conditioning*) ini sering diimplementasikan pada tingkat arsitektural, seperti menyediakan modul *encoder* dan *decoder* khusus tugas tertentu, atau pada tingkat algoritmik seperti kerangka kerja optimasi *inner and outer loop* pada model MAML. 

Namun, seperti yang dicontohkan dalam literatur McCann et al. (2018), **bahasa itu sendiri menyediakan cara yang sangat fleksibel untuk menspesifikasikan tugas, masukan, dan keluaran, semuanya sebagai sebuah urutan simbol teks**. 

Sebagai contoh (perhatikan format *prompting* berikut):
* Contoh pelatihan untuk **penerjemahan** dapat ditulis sebagai sekuens teks: 
  ```text
  (translate to french, english text, french text)
  ```
* Contoh pelatihan untuk **pemahaman bacaan** dapat ditulis sebagai: 
  ```text
  (answer the question, document, question, answer)
  ```

Literatur McCann et al. (2018) telah mendemonstrasikan bahwa adalah mungkin untuk melatih model tunggal (MQAN) untuk menyimpulkan dan melakukan berbagai macam tugas menggunakan format jenis ini.

Kekuatan utama dari **pemodelan bahasa (*language modeling*)** adalah kemampuannya, secara prinsip, untuk mempelajari tugas-tugas tersebut **tanpa memerlukan pengawasan eksplisit** mengenai simbol mana yang menjadi luaran prediksi (seperti yang dituntut dalam *supervised learning* biasa). Karena objektif pembelajaran terselia pada dasarnya sama dengan objektif pembelajaran tak terselia—hanya saja dievaluasi pada subkumpulan urutan teks—titik minimum global dari fungsi objektif tak terselia adalah juga titik minimum global dari fungsi objektif terselia. 

Masalah utamanya kemudian berubah menjadi: *apakah dalam praktiknya kita mampu mengoptimalkan objektif tak terselia tersebut hingga mencapai konvergensi (titik temu solusi)?* Eksperimen awal peneliti mengkonfirmasi bahwa model bahasa yang cukup besar *mampu* melakukan pembelajaran multitugas dalam pengaturan *toy-ish* (skala kecil/eksperimental) ini, tetapi pembelajarannya **jauh lebih lambat** dibandingkan pendekatan yang diselia secara eksplisit.

Meskipun konsep *learning from dialogue* (belajar dari dialog) adalah hal yang menarik, para peneliti merasa itu terlalu membatasi. Internet mengandung jutaan informasi yang tersedia secara pasif tanpa memerlukan komunikasi interaktif. Spekulasi utama peneliti adalah: **Sebuah model bahasa dengan kapasitas komputasi yang sangat besar akan mulai belajar untuk menyimpulkan (infer) dan melakukan tugas-tugas yang didemonstrasikan di dalam urutan bahasa alami (di internet) semata-mata agar ia dapat memprediksi kata berikutnya dengan lebih akurat.** 

Jika sebuah model bahasa mampu melakukan hal ini, maka secara efektif, model tersebut sedang melakukan **pembelajaran multitugas tak terselia (*unsupervised multitask learning*)**. Hipotesis inilah yang diuji oleh peneliti dengan menganalisis performa model bahasa dalam pengaturan *zero-shot* pada berbagai ragam tugas.

### Tabel 1. Demonstrasi Alami dalam Dataset WebText
*(Tabel berikut menampilkan contoh-contoh nyata bagaimana demonstrasi terjemahan muncul secara acak dan alami di dalam teks di internet tanpa ada yang sengaja melabelinya sebagai tugas terjemahan).*

| Contoh demonstrasi penerjemahan Bahasa Inggris ke Prancis & Prancis ke Inggris yang ditemukan alami di WebText |
| :--- |
| "I'm not the cleverest man in the world, but like they say in French: **Je ne suis pas un imbecile [I'm not a fool]**." |
| In a now-deleted post from Aug. 16, Soheil Eid, Tory candidate in the riding of Joliette, wrote in French: "**Mentez mentez, il en restera toujours quelque chose**," which translates as, "**Lie lie and something will always remain**." |
| "I hate the word '**perfume**,'" Burr says. 'It's somewhat better in French: '**parfum**.'" |
| If listened carefully at 29:55, a conversation can be heard between two guys in French: "**-Comment on fait pour aller de l'autre coté? -Quel autre coté?**", which means "**- How do you get to the other side? - What side?**". |
| If this sounds like a bit of a stretch, consider this question in French: **As-tu aller au cinéma?**, or **Did you go to the movies?**, which literally translates as Have-you to go to movies/theater? |
| "**Brevet Sans Garantie Du Gouvernement**", translated to English: "**Patented without government warranty**". |

---

### 2.1 Dataset Pelatihan (*Training Dataset*)

Sebagian besar penelitian sebelumnya melatih model bahasa pada satu domain teks tunggal, seperti artikel berita, Wikipedia, atau buku fiksi. Pendekatan para peneliti di sini justru memotivasi pembuatan dataset yang **sebesar dan se-beragam mungkin** guna mengumpulkan demonstrasi tugas bahasa dalam konteks dan domain yang tak terbatas.

Sumber teks tak terbatas yang sangat menjanjikan adalah hasil *scraping* (pengikisan data) web, seperti *Common Crawl*. Walaupun ukurannya bisa berkali-kali lipat lebih besar dari dataset saat ini, data web mentah memiliki **masalah kualitas yang fatal**. Banyak dokumen yang kontennya sama sekali tidak dapat dipahami (*unintelligible*). 

Alih-alih menggunakan seluruh Common Crawl, para peneliti menciptakan dataset *web scrape* baru yang sangat **menekankan kualitas dokumen**. 
* Sebagai titik awal yang cerdas untuk menyaring kualitas web secara otomatis (karena menyaring manual akan memakan biaya luar biasa mahal), peneliti melakukan *scrape* terhadap **semua tautan keluar (*outbound links*) dari Reddit (sebuah platform media sosial) yang menerima minimal 3 karma (*upvotes*)**. 
* Ini berfungsi sebagai **indikator heuristik** bahwa ada manusia lain yang menganggap tautan tersebut menarik, edukatif, atau setidaknya lucu.

Dataset yang dihasilkan ini diberi nama **WebText**. Dataset ini terdiri dari kumpulan teks dari **45 juta tautan** tersebut. Setelah melakukan pembersihan berbasis heuristik dan de-duplikasi, versi akhir dataset WebText ini berisi lebih dari **8 juta dokumen** dengan total ukuran teks sebesar **40 GB**. 

Satu keputusan arsitektural yang krusial: **Peneliti menghapus seluruh dokumen Wikipedia dari WebText**. Mengapa? Karena Wikipedia sudah menjadi sumber data pengujian (evaluasi) yang umum pada banyak dataset lain. Menghapus Wikipedia mencegah terjadinya tumpang tindih data (*data overlap*) yang dapat merusak validitas analisis pengujian.

---

### 2.2 Representasi Masukan (*Input Representation*)

Sebuah Pemodelan Bahasa Umum (General LM) idealnya mampu menghitung probabilitas (dan juga merangkai/membuat) untaian *string* karakter apa pun. LLM berskala besar saat ini sering menggunakan tahap pra-pemrosesan yang merusak (seperti mengubah semua menjadi huruf kecil/*lowercasing*, tokenisasi, dan token *out-of-vocabulary*/OOV), yang membatasi kemampuan model untuk memproses string bebas.

Memproses teks sebagai *sequence* berbasis **Byte UTF-8** secara elegan dapat menyelesaikan masalah ruang *string* ini. Namun, kinerja LM berbasis tingkat-Byte secara historis selalu kalah kompetitif dibandingkan LM berbasis kata (*word-level*).

Untuk menjembataninya, peneliti menggunakan **Byte Pair Encoding (BPE)**. Ini adalah jalan tengah yang praktis antara pemodelan tingkat kata dan karakter. BPE secara efektif akan membentuk token berbasis "kata utuh" (untuk sekuens simbol/kata yang sangat sering muncul) dan kembali menggunakan "karakter" (untuk sekuens/kata asing yang jarang muncul). 

Masalahnya, implementasi standar BPE sering kali beroperasi pada titik kode Unicode, BUKAN pada urutan Byte. Jika dipaksa mengakomodasi semua titik Unicode, ukuran kosakata dasar (base vocabulary) akan meledak melebihi 130.000 token. Sebaliknya, BPE tingkat-Byte hanya butuh kosakata dasar sebesar **256 ukuran** (karena representasi byte 0-255). 

Akan tetapi, jika BPE diimplementasikan mentah-mentah pada urutan Byte, hasil penggabungan kata (*merges*) menjadi sangat sub-optimal karena BPE bekerja dengan algoritma serakah (greedy). Misalnya, BPE akan membuat banyak versi untuk kata yang sama dengan tanda baca berbeda (seperti `dog`, `dog.`, `dog!`, `dog?`). Ini memboroskan slot kosakata model.

Untuk mengatasi ini, peneliti melakukan inovasi modifikasi: **Mereka mencegah BPE untuk melakukan *merging* (penggabungan) lintas kategori karakter untuk semua urutan byte**. 
Peneliti hanya memberikan satu pengecualian, yakni pada spasi, yang ternyata secara signifikan meningkatkan efisiensi kompresi memori sembari mencegah fragmentasi/pecahnya satu kata yang sama di berbagai token. 

Representasi masukan BPE tingkat-Byte ini sangat revolusioner karena memungkinkan model GPT-2 untuk dievaluasi pada dataset apapun dari seluruh dunia **tanpa memerlukan pra-pemrosesan, tokenisasi, atau modifikasi ukuran kosakata sama sekali**.

---

### 2.3 Model

Para peneliti menggunakan arsitektur berbasis **Transformer** untuk model bahasanya. Model ini secara garis besar mengikuti arsitektur **OpenAI GPT-1**, dengan sedikit modifikasi:
1. **Posisi Layer Normalization:** Normalisasi lapisan dipindahkan letaknya ke *input* dari setiap sub-blok (seperti arsitektur *pre-activation residual network*).
2. **Tambahan Layer Normalization:** Satu buah normalisasi lapisan tambahan diletakkan persis setelah blok *self-attention* paling akhir.
3. **Inisialisasi Residual:** Mereka memodifikasi inisialisasi bobot untuk mencegah akumulasi nilai pada jalur residual seiring dengan kedalaman model. Peneliti menskalakan bobot dari lapisan residual saat inisialisasi dengan faktor pembagi:
   $$ \frac{1}{\sqrt{N}} $$
   > **Penjelasan Variabel:** $N$ melambangkan jumlah total lapisan residual di dalam arsitektur model. Pembagian dengan akar kuadrat $N$ ini secara matematis menjaga stabilitas variansi propagasi sinyal ke depan dan gradien ke belakang saat melatih model yang sangat dalam (puluhan *layer*).
4. Ukuran kosa kata (*vocabulary*) diperluas menjadi **50.257**.
5. Ukuran konteks (panjang teks yang bisa dilihat model sekaligus) ditingkatkan dari 512 menjadi **1024 token**.
6. Ukuran *Batchsize* (kelompok data latih dalam satu iterasi komputasi) diperbesar menjadi **512**.

#### Tabel 2. Hiper-parameter Arsitektur untuk ke-4 Ukuran Model

| Parameter | Jumlah Lapisan (*Layers* / $N$) | Dimensi Model ($d_{model}$) |
| :--- | :---: | :---: |
| 117M (GPT-1 equivalent) | 12 | 768 |
| 345M (BERT-Large equivalent)| 24 | 1024 |
| 762M | 36 | 1280 |
| **1542M (GPT-2)** | **48** | **1600** |

*(Catatan Dosen: Amati Tabel 2. GPT-2 yang berukuran 1.542 Juta parameter (1.5 Miliar) menggunakan kedalaman 48 lapisan Transformer bertumpuk dengan lebar vektor 1600 dimensi. Ini adalah arsitektur raksasa di eranya).*

Semua model dilatih dan hiper-parameter kecepatan pembelajaran (*learning rate*) disetel secara manual untuk mendapatkan angka kebingungan/kerumitan (*perplexity*) terbaik pada sampel validasi 5% dari WebText. Hingga tulisan ini dibuat, **semua ukuran model masih bersifat *underfit*** pada WebText, yang artinya jika diberi waktu pelatihan lebih lama, performa mereka akan terus meningkat.

---

## 3. Eksperimen (*Experiments*)

*(Catatan Dosen: Mari kita bedah hasil performa di berbagai disiplin tugas tanpa ada campur tangan pelabelan manusia)*

### Deskripsi Ekstensif Gambar 1 (Zero-shot Task Performance)
Gambar 1 di dalam makalah asli menampilkan grid yang terdiri dari 4 grafik garis yang memetakan **Kapasitas Ukuran Model (Sumbu-X dalam skala Logaritmik: dari 117M hingga 1542M/1.5B)** terhadap **Performa Metrik Tugas (Sumbu-Y)** untuk empat bidang kajian (*zero-shot*):
1. **Reading Comprehension (Kiri Atas):** Sumbu-Y adalah metrik F1 (rentang 30-90) pada dataset CoQA. Kurva biru (GPT-2) terlihat meningkat secara log-linear dan konsisten seiring bertambahnya parameter model. Garis batas performa *baseline* manusia yang dipetakan sebagai garis putus-putus (*dashed line*) di angka 70+, hingga akhirnya model 1.5B menembus titik F1 = 55 (mengalahkan *baseline* sistem DrQA, dll).
2. **Translation (Kiri Bawah):** Sumbu-Y adalah BLEU skor (rentang 0-25) untuk terjemahan Bahasa Prancis-Inggris WMT-14. Kurva tren menunjukkan peningkatan tajam dari model 117M (mendekati 0) menanjak naik menjadi lebih dari 11.5 BLEU pada kapasitas 1.5B.
3. **Summarization (Kanan Atas):** Sumbu-Y adalah ROUGE-1/2/L (rentang 16-32) pada dataset CNN/DailyMail. Kurva tidak meningkat setajam komprehensi bacaan. Terdapat lompatan kecil dari model terkecil ke menengah, namun peningkatannya mulai landai dan merangkak pelan di bawah batas *baseline* model Summarization khusus (*Seq2Seq+Attn*).
4. **Question Answering (Kanan Bawah):** Sumbu-Y merepresentasikan Akurasi dalam persentase (%) pada dataset Natural Questions. Kurva menembak naik dari akurasi nyaris 0% (model kecil) dan meroket secara linear tanpa indikasi melambat hingga mendekati 6% di ujung kanan untuk GPT-2 (1.5B), mengindikasikan semakin raksasa memorinya, semakin banyak fakta ensiklopedis dunia yang dihafal dengan sendirinya.

---

### 3.1 Pemodelan Bahasa (*Language Modeling*)

Sebagai evaluasi primer, peneliti mengukur performa penguasaan pemodelan bahasa *zero-shot* menggunakan dataset umum. Karena model GPT-2 beroperasi murni pada tingkat karakter-byte tanpa pra-pemrosesan (seperti menghilangkan huruf kapital/tanda baca), model ini harus menghadapi tes yang luar biasa tidak adil. Ia diuji menebak kata *di luar distribusi* data aslinya (berbeda gaya bahasa dengan Reddit/internet mentah). 

Peneliti menggunakan **metrik Perplexity (PPL)**, atau dalam format probabilitas negatif logaritma per unit (*Bits per Character* / BPC atau *Bits per Byte* / BPB). Untuk melakukan evaluasi yang wajar pada dataset kuno yang telah kehilangan tanda bacanya (sudah di-*pre-process* dan di tokenisasi oleh pencipta awalnya dengan cara menghilangkan spasi/mengubah huruf), OpenAI GPT-2 menggunakan pen-token-balik (*invertible de-tokenizers*) untuk mengembalikan tata bahasa kalimat asalnya.

#### Tabel 3. Hasil Pemodelan Bahasa Zero-Shot

| | LAMBADA (PPL) | LAMBADA (ACC) | CBT-CN (ACC) | CBT-NE (ACC) | WikiText2 (PPL) | PTB (PPL) | enwik8 (BPB) | text8 (BPC) | WikiText103 (PPL) | 1BW (PPL) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SOTA** (Sebelumnya) | 99.8 | 59.23 | 85.7 | 82.3 | 39.14 | 46.54 | 0.99 | 1.08 | 18.3 | **21.8** |
| 117M | 35.13 | 45.99 | 87.65 | 83.4 | 29.41 | 65.85 | 1.16 | 1.17 | 37.50 | 75.20 |
| 345M | 15.60 | 55.48 | 92.35 | 87.1 | 22.76 | 47.33 | 1.01 | 1.06 | 26.37 | 55.72 |
| 762M | 10.87 | 60.12 | 93.45 | 88.0 | 19.93 | 40.31 | 0.97 | 1.02 | 22.05 | 44.575 |
| **1542M (GPT-2)**| **8.63** | **63.24** | **93.30**| **89.05**| **18.34** | **35.76** | **0.93** | **0.98** | **17.48** | 42.16 |

*(Keterangan PPL: Lebih kecil lebih baik. ACC: Lebih besar lebih baik. BPB/BPC: Lebih kecil lebih baik).*

**Kesimpulan dari Tabel 3:**
GPT-2 model raksasa (1542M) secara spektakuler mentransfer pengetahuannya dengan sangat baik, **menghancurkan rekor SOTA (State of The Art) sebelumnya pada 7 dari 8 dataset uji**. Penurunan angka Perplexity yang luar biasa tercatat pada dataset berkapasitas kecil seperti PTB (Penn Treebank) dan WikiText-2.

Satu-satunya kekalahan GPT-2 terletak pada dataset 1BW (One Billion Word Benchmark). Hal ini dapat dimaklumi karena dataset 1BW merupakan dataset yang paling merusak struktur bahasa, di mana pembuatnya mengacak posisi kalimat demi kalimat (*sentence level shuffling*), yang secara praktis membumihanguskan struktur bahasa berjangka panjang yang menjadi kekuatan utama GPT-2.

---

### 3.2 Ujian Buku Anak-Anak (*Children’s Book Test / CBT*)

CBT diciptakan untuk menguji model bahasa dalam memprediksi berbagai tipe kata secara *cloze test* (tes isian rumpang / kalimat yang ada bagian kosongnya), berfokus pada: Entitas Bernama (Named Entities) dan Kata Benda Umum (Common Nouns). Tugasnya adalah memilih 1 dari 10 kata yang tersedia untuk melengkapi kalimat bolong tersebut.

Metode *Zero-shot* yang dilakukan GPT-2: Model LM menghitung persentase probabilitas gabungan (*joint probability*) dari masing-masing 10 opsi tersebut di dalam sisa untaian kalimat. Pilihan dengan skor persentase prediksi tertinggi dipilih sebagai jawaban mesin.

### Deskripsi Ekstensif Gambar 2 (Children's Book Test)
Gambar 2 menunjukkan dua buah grafik garis. Keduanya mengukur Akurasi (Sumbu-Y, 75%-100%) berbanding Ukuran Parameter Model (Sumbu-X logaritmik, 117M ke 1542M). 
*   **Grafik Kiri (Common Nouns):** Menunjukkan bahwa garis biru model OpenAI (GPT-2) naik perlahan dari 87% menuju **93,3%**. Pada grafik ini, terdapat garis putus-putus (*dashed line*) horizontal yang melambangkan batas batas performa "Human" (Manusia) di sekitar angka 96%.
*   **Grafik Kanan (Named Entities):** Kurva biru melesat kuat dari sekitar 83% menanjak konstan membelah ke atas menuju **89,1%**. Garis batas Performa Manusia tergambar di sekitar batas 96%.
*   **Kesimpulan Akademis Gambar 2:** Kapasitas parameter secara mutlak memangkas kesenjangan (*gap*) antara performa sebuah mesin Kecerdasan Buatan (AI) yang sebelumnya hanya bernilai acak, hingga menyamai tingkat kecerdasan membaca manusia dewasa seutuhnya. Kedua grafik mencetak rekor SOTA baru.

---

### 3.3 Evaluasi LAMBADA

Dataset LAMBADA menguji kemampuan sistem AI untuk **menangkap dan memodelkan ketergantungan teks jarak jauh (*long-range dependencies*)**. Tugas model adalah memprediksi *satu buah kata terakhir* dari kalimat, di mana manusia sendiri butuh konteks bacaan minimal 50 kata sebelum kalimat tersebut agar dapat menebaknya.

GPT-2 sukses menurunkan rekor *perplexity* LAMBADA yang tadinya di angka 99.8 menjadi hanya **8.6**. Terkait Akurasi prediksi tebakan kalimat akhirnya, model ini menaikkan akurasi dari generasi model sebelumnya yang hanya mampu menjawab 19% menjadi melambung tinggi di angka **52.66%**.

Melalui analisis kesalahan (*error analysis*), peneliti menemukan fenomena unik: tebakan kata terakhir GPT-2 sebenarnya secara susunan gramatikal *sangat logis* untuk melanjutkan alur kalimat tersebut, namun seringkali ia bukanlah kata mutlak *terakhir* dari ujung kalimat itu sendiri. Untuk mengakalinya secara teknis algoritmik, tim riset menambahkan penyaring *stop-word filter* khusus yang melarang LM memuntahkan kata-kata penyambung tidak bermakna di ujung tebakannya. Trik akal-akalan linguistik ini mendongkrak skor akhir menjadi **63.24%** (Menguasai SOTA).

---

### 3.4 Winograd Schema Challenge

Winograd Schema Challenge adalah tolok ukur spesifik untuk mengevaluasi **penalaran akal sehat (*commonsense reasoning*)** AI dalam hal menangani ambiguitas kata ganti (*pronoun resolution*). Contoh: *"Piala tidak muat masuk ke koper coklat itu karena ia terlalu besar."* Pertanyaannya, apakah kata *'ia'* merujuk ke piala atau koper?

### Deskripsi Ekstensif Gambar 3 (Winograd Schema Challenge)
Gambar 3 adalah sebuah kurva plot titik (*scatter curve plot*) yang menempatkan Akurasi (Sumbu-Y, 50% hingga 75%) terhadap Skala Model (Sumbu-X). Terdapat dua garis tren yang merambat ke atas.
*   **Garis Oranye (Full Scoring):** Cara pemberian skor secara penuh di mana AI harus menebak tepat kalimat resolusi utuh. Berawal dari skor setara *random guessing* (50%) pada parameter 117M, lalu kurva merayap naik konstan melampaui garis putus-putus (*dashed line*) abu-abu, berakhir di puncaknya pada parameter 1542M (1.5B).
*   **Garis Biru (Partial Scoring):** Cara evaluasi sebagian yang memperbolehkan kelonggaran komputasi linguistik. Pola lintasannya juga persis menduplikasi lintasan linear kurva oranye.
*   **Garis Putus-putus Abu-Abu (SOTA):** Merupakan garis datar (horizontal) di level kisaran skor 63% yang menggambarkan batas rekor dunia arsitektur bahasa terbaik masa lalu (SOTA era Trinh & Le).
*   **Kesimpulan Akademis Gambar 3:** GPT-2 memecahkan rekor SOTA secara dramatis pada Winograd Challenge, mendobrak atap langit SOTA hingga mencapai akurasi **70.70%** (naik 7 persen poin tanpa melalui pelatihan khusus). Mengingat total set pertanyaan pengujian murni Winograd hanya berisi 273 skenario sempit, hasil kurva model ini memvalidasi teori "Zero-shot transfer" OpenAI.

---

### 3.5 Pemahaman Bacaan (*Reading Comprehension*)

Untuk menguji apakah model mengerti apa yang dibaca layaknya dialog antar dua insan, digunakan dataset **CoQA (Conversation Question Answering)**. Ini menguji pemahaman teks dari 7 domain dan kemampuan menjawab secara berantai (merujuk riwayat percakapan sebelumnya seperti kata *“Kenapa?”*).

Cara mengeksekusi model GPT-2 dalam mode *zero-shot* tanpa dilatih:
1. Model dikondisikan (diberi umpan teks awal/prompts) berisi teks Dokumen utama bacaan.
2. Diikuti dengan runtutan riwayat percakapan (Pertanyaan-Jawaban) masa lampau.
3. Diakhiri dengan memberikan injeksi token pemancing bertuliskan huruf `A:` (yang bermaksud Answer/Jawaban), untuk memicu model agar secara serakah memuntahkan tebakan kata lanjutannya.

Hebatnya, dengan strategi *prompting* naif ini, GPT-2 mencetak rekor F1 score = **55**. Ia mengalahkan performa 3 dari 4 sistem spesialis yang sengaja disuapi secara manual dengan data spesifik 127.000 contoh pelatihan QA (Question-Answer).
Walaupun mesin *baseline* BERT masih menang telak (di kisaran 89 F1), capaian 55 F1 milik sistem tanpa pengawasan GPT-2 membuktikan bahwa model raksasa ini cenderung berhasil beroperasi menggunakan *heuristic based retrieval* insting bahasa. Secara contoh konkrit: jika ditanya *'who' (Siapa)*, GPT-2 akan otomatis memuntahkan jawaban yang merujuk pada nama *'entitas manusia/tokoh'* di dokumen tersebut.

---

### 3.6 Peringkasan (*Summarization*)

Bagaimana GPT-2 meringkas artikel berita panjang CNN/Daily Mail tanpa pernah diajari meringkas? 

Peneliti menanamkan sebuah kata kunci *prompt* sakti berupa teks: `TL;DR:` (*Too Long; Didn't Read*) persis di akhir artikel masukannya. 

Untuk menghindari teks ringkasan yang terlalu literal terulang persis sama (degenerasi pengulangan), model menghasilkan 100 token dengan strategi acak *Top-k random sampling* ($k = 2$). Dari 100 token luaran, sistem hanya mengambil dan mengekstrak 3 kalimat pertamanya saja yang dianggap sebagai ringkasan akhir.

#### Tabel 4. Performa Peringkasan Menggunakan Metrik ROUGE F1 (CNN/Daily Mail)

| Model Evaluasi | R-1 | R-2 | R-L | R-AVG |
| :--- | :---: | :---: | :---: | :---: |
| Bottom-Up Sum (SOTA Supervised) | **41.22** | **18.68** | **38.34** | **32.75** |
| Lede-3 | 40.38 | 17.66 | 36.62 | 31.55 |
| Seq2Seq + Attn | 31.33 | 11.81 | 28.83 | 23.99 |
| **GPT-2 (Dengan Prompt TL;DR:)** | 29.34 | 8.27 | 26.58 | 21.40 |
| Random-3 (Baseline Acak) | 28.78 | 8.63 | 25.52 | 20.98 |
| GPT-2 (Tanpa hint prompt) | 21.58 | 4.03 | 19.47 | 15.03 |

**Kesimpulan dari Tabel 4:**
Meskipun hasil generasinya secara kualitatif terlihat mirip ringkasan, secara metrik statistik ROUGE, GPT-2 baru sekadar sanggup menembus angka ambang (*barely outperforms*) kemampuan acak manusia mengambil 3 kalimat random di artikel. Meski demikian, secara krusial, **penghapusan pemancing/prompt tugas (*task hint*) "TL;DR:" membuat angka performa rontok drastis turun hingga -6.4 poin**. Hal ini merupakan validasi pembuktian hipotesis peneliti: bahwa memanggil/memprovokasi perilaku spesifik-tugas bahasa secara mutlak dapat dibangkitkan secara gaib hanya menggunakan pemancing *Natural Language* "TL;DR:".

---

### 3.7 Penerjemahan (*Translation*)

Peneliti bereksperimen apakah GPT-2 diam-diam belajar menerjemahkan. Untuk merangsang GPT-2 memuntahkan kalimat translasi *zero-shot*, tim memancingnya menggunakan format penempatan (*formatting/prompting*) berurutan seperti ini:
```text
english sentence = french sentence
english sentence = french sentence
...
english sentence (kalimat yang mau dites) = 
```
Model lalu menghasilkan probabilitas huruf demi huruf menggunakan pembacaan *Greedy decoding* dan hasilnya akan diekstrak menjadi luaran (*output*).

*   **English-French (En-Fr):** Pada basis data penguji WMT-14, skor yang dicetak payah (hanya 5 BLEU).
*   **French-English (Fr-En):** Di sisi lain, GPT-2 berhasil memanfaatkan keperkasaannya dalam fondasi Bahasa Inggris untuk mengatrol nilai kemampuannya, mencetak titik tertinggi **11.5 BLEU**.

Kapasitas raihan 11.5 BLEU *Unsupervised* dari model GPT-2 ini sukses menjebol dan melampaui sistem-sistem translasi tak terselia pra-2017 milik (Artetxe dkk & Lample dkk). 
Yang membuat hasil ini **sangat mengejutkan dan tidak disangka-sangka oleh penulis sendiri** adalah: mereka secara ekskulsif menyaring membuang seluruh bahasa di luar Inggris (non-English web pages) dari WebText. Untuk menuntaskan misteri konspirasi bahasa ini, peneliti menerjunkan modul pemindai pendeteksi byte (*byte-level language detector*) kepada arsip bongkahan WebText berkapasitas 40 GB. Fakta sains mengejutkan: WebText aslinya ternyata bocor dan masih mengandung secuil teks Prancis berukuran kerdil sekitar **10 MB**, yang mana secara kuantitas kapasitasnya 500x lipat luar biasa kerdil daripada kumpulan korpus monolingual yang biasa dipakai sistem Translasi Pakar terdahulu, namun GPT-2 mampu mencaplok insting penerjemahan bahasanya.

---

### 3.8 Menjawab Pertanyaan Fakta (*Question Answering*)

Untuk menakar isi wawasan informasi dunia di jaringan otak GPT-2, model diujicobakan dalam tebakan jawaban faktual murni. Dataset yang dipakai adalah karya Universitas riset raksasa bertajuk **Natural Questions Dataset (Kwiatkowski et al., 2019)**.
Mirip di kasus translasi, AI dibekali memori konteks pengkodingan *seeded question-answer pairs*.

Secara keseluruhan, Model 1.5B (GPT-2) membalas soal trivia menjawab yang benar sekitar **4.1%**. Menariknya lagi, performa AI OpenAI tumbuh berkali lipat menanjak eksponensial searah berkat suntikan skala paramater: GPT-2 (1.5B) sukses meraup keakuratan 5,3 kali lebih tajam ketimbang versi kerdil 117M nya. Hal ini mengerucut kuat mengindikasikan jika bobot jaringan dikembangkan menuju puluhan miliar Miliar (10x lipat), tidak menutup harapan AI akan setara perpustakaan manusia sejati. Di 1% tebakan model yang paling ia yakini kemantapannya secara persentase kalibrasi internal, keakuratannya meroket menyentuh angka dewa mencapai **63.1%**.

#### Tabel 5. Pemilihan 30 Jawaban Terpercaya GPT-2 pada Natural Questions (Dipotong ke 22 Baris untuk Demonstrasi)

| Pertanyaan (Question) | Jawaban Hasil Generasi AI | Benar (Correct) | Probabilitas GPT-2 |
| :--- | :--- | :---: | :---: |
| Who wrote the book the origin of species? | Charles Darwin | ✓ | 83.4% |
| Who is the founder of the ubuntu project? | Mark Shuttleworth | ✓ | 82.0% |
| Who is the quarterback for the green bay packers? | Aaron Rodgers | ✓ | 81.1% |
| Panda is a national animal of which country? | China | ✓ | 76.8% |
| Who came up with the theory of relativity? | Albert Einstein | ✓ | 76.4% |
| When was the first star wars film released? | 1977 | ✓ | 71.4% |
| What is the most common blood type in sweden? | A | ✗ | 70.6% |
| Who took the first steps on the moon in 1969? | Neil Armstrong | ✓ | 66.8% |
| Who is the largest supermarket chain in the uk? | Tesco | ✓ | 65.3% |
| What is the meaning of shalom in english? | peace | ✓ | 64.0% |
| Largest state in the us by land mass? | California | ✗ | 59.2% |
| Green algae is an example of which type of reproduction? | parthenogenesis | ✗ | 56.5% |
| Vikram samvat calender is official in which country? | India | ✓ | 55.6% |
| Who is mostly responsible for writing the declaration of independence? | Thomas Jefferson | ✓ | 53.3% |
| What us state forms the western boundary of montana? | Montana | ✗ | 52.3% |
| Who plays ser davos in game of thrones? | Peter Dinklage | ✗ | 52.1% |
| State the process that divides one nucleus into two genetically identical nuclei? | mitosis | ✓ | 50.7% |
| Who won the most mvp awards in the nba? | Michael Jordan | ✗ | 50.2% |
| What river is associated with the city of rome? | the Tiber | ✓ | 48.6% |
| What is the name given to the common currency to the european union? | Euro | ✓ | 46.8% |
| Who proposed evolution in 1859 as the basis of biological development? | Charles Darwin | ✓ | 45.7% |
| Nuclear power plant that blew up in russia? | Chernobyl | ✓ | 45.7% |

*(Catatan Dosen: Terdapat footnote humor dari peneliti, Alec, yang mengaku awalnya merasa pandai bermain trivia sejarah/budaya secara acak, namun saat ia coba mengerjakan tes ini sendiri, ia hanya mendapat skor 14 benar, lebih sedikit dibandingkan model).*

---

## 4. Generalisasi versus Penghafalan Murni (*Generalization vs Memorization*)

Baru-baru ini di bidang *Computer Vision*, ditemukan masalah integritas akademis mendalam di mana *dataset* lazim (misalnya CIFAR-10) secara cacat menyimpan data citra uji latih (test) yang tumpang tindih nyaris identik terselip bersembunyi dalam lumbung latih/train data. Anomali kotor ini menyebabkan "Laporan Inflasi Over-Reporting". Mengingat WebText (40 GB teks) sangat raksasa ukurannya, bahaya fatal tumpang tindih data pengujian (Test set leaking into Training set) wajib diselidiki.

Untuk mengungkap ini secara metodologi matematis, peneliti membangun algoritma **Bloom filters** berisikan patahan **8-grams** token WebText (kumpulan sekuens 8 huruf gabungan). Untuk menambah daya recall (pengingat kepekaan filter string), penulis mensterilkan baris *string* membuang anomali kapital hanya meloloskan spasi tunggal. Pengujian simulasi silang dilakukan dan false positive rate dicekik keras hingga meleset ke rasio puncak 1 dibanding 100 juta string.

#### Tabel 6. Persentase Tumpang-Tindih 8-Grams antara Test Set dengan Set Latihan Berbagai Dataset Bahasa

| Dataset | PTB | WikiText-2 | enwik8 | text8 | Wikitext-103 | 1BW |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Tumpang Tindih *Dataset Train* miliknya sendiri | **2.67%** | 0.66% | **7.50%** | 2.34% | **9.09%** | **13.19%** |
| Tumpang Tindih Terhadap *WebText Train* | 0.88% | **1.63%** | 6.31% | **3.94%** | 2.42% | 3.75% |

**Kesimpulan dari Filter Data Tabel 6:**
Fakta ilmiah membuktikan bahwa test evaluasi dari model bahasa lain bersinggungan tipis dengan korpus memori raksasa latih web WebText berkisar di rerata kecil **3,2%**. Penemuan fatalnya adalah: Justru beberapa *dataset* uji milik komunitas akademik lain tumpang tindih parah/bocor *dengan set pelatihannya sendiri*, rata-ratanya membengkak nyaris dua kali lipat menjadi 5,9% tumpang tindih internal. Tragedi yang paling kronis terlihat pada basis data Evaluasi terkemuka 1BW yang secara memalukan mengalami persinggungan dengan basis latihannya sendiri sebesar **13,19%** (sangat terkontaminasi).

Secara khusus, pada ujian LAMBADA, kebocoran string dengan arsip WebText bernilai minim sekitar 1.2%. Menghapus barisan teks tumpang tindih tak murni membuahkan fakta yang melegakan: Akurasi kemahiran tes murni GPT-2 pasca filter ini jatuh hanya 0.3%, menjadi **62.9%** (tetap SOTA).

Metode lain untuk mengkonfirmasi bahwa ini BUKAN menghafal membabi buta (*memorization/overfitting*) adalah dengan memeriksa grafik performa model.

### Deskripsi Ekstensif Gambar 4 (Train vs Test Perplexity over Model Size)
Gambar 4 adalah grafik garis dua arah diagonal menurun yang menampilkan Kurva Skala (*Scaling Laws*). 
*   **Sumbu-X:** Jumlah Parameter Skala Logaritmik (117M ke 1542M).
*   **Sumbu-Y:** Metrik Perplexity/Kebingungan teks (rentang mulai tinggi 16 ke bawah ujung 8).
*   **Garis Biru Solid:** Representasi kebingungan *WebText train*.
*   **Garis Oranye Solid:** Representasi kebingungan prediksi *WebText test* (data belum pernah dihafal).
*   **Kesimpulan Akademis Gambar 4:** Dua sabuk lintasan Biru dan Oranye bergerak menukik mulus beriringan paralel satu sama lain tanpa tanda-tanda menyimpang berpisah. Bukti konvergen paralel absolut ini menandakan bahwa Jantung otak mesin kecerdasan ini tidak semata membeo mengingat buta (over-fit), melainkan mesin GPT-2 (1.5B) sejatinya masih di tahap lapar data (*underfitting*) murni, menyerap struktur tata bahasa tanpa menghafal titik mati. 

*(Catatan Dosen: Para peneliti menyebutkan GPT-2 berhasil menulis novel artikel fiksi penemuan sekelompok Unikorn yang dapat berbahasa Inggris di pegunungan Andes secara masuk akal dan nyaris sempurna. Hasil ini tersedia di Tabel 13 pada bagian lampiran asli makalah, yang menunjukkan kecanggihan generatif model di luar narasi logis manusia).*

---

## 5. Pekerjaan Terkait (*Related Work*)

Penelitian ini memiliki fondasi kokoh yang berdiri di atas bahu raksasa literatur lampau. Mayoritas penelitian sebelumnya mengukur peningkatan performa pemodelan bahasa dengan membesarkan model secara masif pada dataset yang lebih raksasa (Jozefowicz et al., 2016, dkk; Bajgar et al., 2016; Hestness et al., 2017). Eksperimen GPT-2 di atas membuktikan fenomena peningkatan skala (*scaling law*) tetap terus hidup dan merambat pada level sub-lapisan tugas-tugas hingga orde di wilayah 1+ Miliar paramater (*1B+ parameter regime*).

Literatur kuno *Skip-thought Vectors (Kiros et al., 2015)* serta penggunaan vektor terjemahan McCann merintis jejak. Hingga munculnya era emas penyesuaian halus berbasis pra-pelatihan Transformer tak berselia (*Generative pre-training*), model *Seq2seq (Ramachandran et al., 2016)* serta *BERT (Devlin et al., 2018)* merevolusi representasi transfer dua-arah universal. Penemuan yang secara inspirasional sangat dekat dengan makalah GPT-2 adalah pemodelan (Liu et al., 2018), yang tidak sengaja menelurkan temuan bahwa model bahasa raksasa Wikipedia dengan sendirinya mempelajari penerjemahan antarbangsa di dalam rahim representasinya tanpa dilatih terjemahan sama sekali.

---

## 6. Diskusi (*Discussion*)

Banyak penelitian dalam sedekade ini mendedikasikan ilmu untuk melakukan kajian belajar, membongkar pemahaman, dan evaluasi empiris kritis terhadap presentasi model metode pengawasan pakar (*supervised*) maupun tak terselia (*unsupervised*). Penemuan rekor dari jurnal ini mendongkrak status pembelajaran mesin nol langkah spesifik (Zero-shot) ke langit atas yang menjanjikan arena eksplorasi luas di masa mendatang.

Hal ini pada praktiknya memberi wawasan magis dan menjabarkan sebuah tabir fenomena masif atas sukses gemilangnya teknik pra-pelatihan pengerjaan sub-tugas turunan NLP di hilir masa lampau. Peneliti membuktikan bahwa, pada titik batas ekstrem asimtotiknya (*in the limit*), sebuah skema arsitektur tunggal tanpa embel-embel *fine tuning* pengawasan manusia (Zero-shot) dapat murni beradaptasi dan menyelesaikan semua sub-tugas NLP di kolong semesta secara fundamental.

Akan tetapi, meski menembus atap rekor SOTA F1 55 pada ranah baca komprehensif tanpa pengawasan guru, mesin linguistik raksasa ini hanya berkutat pada taraf "tingkat dasar / primitif rintisan" saat beroperasi dalam mode ringkas (Summarization). Maka, meski ini merupakan karya penemuan riset tingkat tinggi yang mencengangkan dunia, arsitektur *zero-shot performance GPT-2* (kemampuan tembakan-nolnya) belum mencapai level sempurna untuk digunakan secara komersil/industri seutuhnya (*far from use-able*). 

Keterbatasan lainnya: Pada berbagai tugas penerjemahan, QA, atau NLP esoterik lain yang tidak dibahas di sini, hasil prediksi nol-tembakan (Zero-shot) GPT-2 faktanya "tidak lebih pintar dari pelemparan koin angka acak". GPT-2 hanya menunjukan sinar magis tanda-tanda kecerdasannya (*outperform trivial baselines*) sesaat sesudah kapabilitas otak penyimpanannya dibuat meraksasa menjadi miliar-an sel saraf parameter. Pertanyaan abadi berikutnya yang akan timbul bagi ilmuwan masa depan adalah: *"Ke manakah atap plafon ujung batas kemampuan (ceiling) model GPT-2 ini jika mesin ini kemudian kita latihkan penyesuaian halus terselia (Fine-Tuning)?"* (Mengingat performa tembakan-nol (*zero-shot*)-nya saja sudah sehebat ini).

---

## 7. Kesimpulan (*Conclusion*)

Manakala arsitektur kecerdasan model bahasa berskala raksasa dilatih dan digembleng menenggak asupan korpus triliunan karakter teks bahasa yang luar biasa masif serta melintasi spektrum keragaman budaya web tanpa pandang bulu, model cerdas tersebut akan otomatis sanggup menangani lintas dimensi dominan yang mencakup tugas penyelesaian lintas dataset. 

Model bahasa **GPT-2 sanggup menundukkan skor rekor teratas dewa (State-of-The-Art) pada ujian *Zero-Shot* dalam 7 dari total 8 basis data pangkalan pengujian terberat model bahasa di planet bumi**. Keanekaragaman dimensi lintas sub-tugas generik fungsional yang mampu dikunyah dan diselesaikan lewat tembakan-nol (Zero-shot setting) secara absolut menggiring kita pada sebuah konklusi ilmiah yang tak terbantahkan:
*Mesin Bahasa Berkapasitas Ultra-tinggi (High-Capacity Model) yang dilatih murni untuk merangkai kemungkinan probabilitas penyusunan teks bahasa pada setelan variasi ekosistem korpus internet yang memadai... sungguh dengan sendirinya mulai "belajar" dan menguasai sejuta keahlian fungsional dalam ranah pemrosesan bahasa, TANPA membutuhkan setetes pun supervisi manusia.*

---
*(Catatan Dosen: Materi utama kita mengenai makalah GPT-2 berhenti sampai di sini. Pahami secara filosofis dan matematis bagaimana arsitektur tanpa modifikasi tugas ini dapat menjelma menjadi generalis yang menakjubkan. Pastikan Anda mengkaji semua metrik evaluasi yang disajikan pada tabel sebelum ujian akhir berlangsung).*