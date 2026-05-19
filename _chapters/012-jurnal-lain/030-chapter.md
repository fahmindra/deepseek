---
slug: ch-12-03
title: "GPT-3"
layout: chapter
---

## Abstrak

Penelitian-penelitian sebelumnya di bidang NLP telah menunjukkan peningkatan performa yang substansial pada berbagai tugas dan tolok ukur (*benchmarks*) melalui metode pra-pelatihan (*pre-training*) pada korpus teks yang sangat besar, yang kemudian diikuti oleh penyesuaian halus (*fine-tuning*) pada tugas yang spesifik. Meskipun arsitekturnya secara tipikal tidak bergantung pada tugas tertentu (*task-agnostic*), metode ini tetap saja **membutuhkan dataset penyesuaian halus yang spesifik untuk setiap tugas**, yang seringkali terdiri dari ribuan hingga puluhan ribu contoh. 

Sebagai perbandingan, manusia umumnya dapat melakukan tugas bahasa baru hanya dengan melihat beberapa contoh (*few examples*) atau sekadar dari instruksi sederhana. Hal inilah yang masih sangat sulit dilakukan oleh sistem NLP saat ini.

Dalam kajian ini, para peneliti menunjukkan bahwa **meningkatkan skala model bahasa (scaling up language models)** secara luar biasa akan meningkatkan performa pembelajaran sedikit-contoh (*few-shot performance*) yang bersifat *task-agnostic*. Terkadang, performanya bahkan mampu bersaing dengan pendekatan *fine-tuning* paling mutakhir (State-of-the-Art / SOTA) sebelumnya. 

Secara spesifik, para peneliti melatih **GPT-3**, sebuah model bahasa autoregresif (*autoregressive language model*) raksasa dengan **175 miliar parameter** (10 kali lipat lebih besar dari model bahasa non-sparse mana pun sebelumnya), dan menguji performanya dalam pengaturan *few-shot*. 

Untuk *semua* tugas yang diujikan, GPT-3 diaplikasikan **tanpa ada pembaruan gradien (*gradient updates*) atau *fine-tuning***. Tugas dan demonstrasi *few-shot* dispesifikasikan murni melalui interaksi teks dengan model. 

GPT-3 mencapai performa yang kuat pada banyak dataset NLP, termasuk penerjemahan (*translation*), tanya-jawab (*question-answering*), tugas melengkapi bagian rumpang (*cloze tasks*), serta beberapa tugas yang membutuhkan penalaran instan (*on-the-fly reasoning*) atau adaptasi domain, seperti menyusun ulang kata yang diacak (*unscrambling words*), menggunakan kata fiktif baru dalam sebuah kalimat, atau melakukan operasi aritmatika 3 digit. 

Pada saat yang sama, para peneliti juga mengidentifikasi beberapa dataset di mana pembelajaran *few-shot* GPT-3 masih kesulitan, serta beberapa dataset di mana GPT-3 menghadapi masalah metodologis terkait kontaminasi data akibat pelatihan pada korpus web yang sangat besar. Terakhir, ditemukan bahwa GPT-3 dapat menghasilkan sampel artikel berita buatan yang sangat baik hingga evaluator manusia kesulitan membedakannya dari artikel yang ditulis oleh manusia asli. Makalah ini juga mendiskusikan dampak sosial yang lebih luas dari temuan ini dan dari GPT-3 secara umum.

---

## 1. Pendahuluan (*Introduction*)

Beberapa tahun terakhir, tren dalam sistem NLP mengarah pada penggunaan representasi bahasa yang telah dipra-latih (*pre-trained*), yang diaplikasikan dengan cara yang semakin fleksibel dan *task-agnostic* untuk transfer tugas di hilir (*downstream transfer*). 
1. **Generasi Pertama:** Representasi lapisan tunggal dipelajari menggunakan vektor kata (*word vectors*) dan dimasukkan ke dalam arsitektur spesifik-tugas.
2. **Generasi Kedua:** Jaringan Saraf Berulang (*Recurrent Neural Networks* / RNN) dengan banyak lapisan representasi dan status kontekstual digunakan untuk membentuk representasi yang lebih kuat, meski masih diaplikasikan pada arsitektur spesifik-tugas.
3. **Generasi Ketiga:** Model bahasa berulang (RNN) atau Transformer yang dipra-latih secara langsung di-*fine-tune*, sehingga sepenuhnya menghilangkan kebutuhan akan arsitektur spesifik-tugas.

Paradigma terakhir ini (pra-pelatihan + *fine tuning*) telah menghasilkan kemajuan substansial pada banyak tugas NLP yang menantang. Namun, ada satu batasan besar: **meskipun arsitekturnya *task-agnostic*, kita tetap membutuhkan dataset dan *fine-tuning* yang spesifik untuk tugas tersebut.** Untuk mencapai performa kuat, *fine-tuning* biasanya membutuhkan dataset berisi ribuan hingga ratusan ribu contoh spesifik. 

Menghilangkan batasan ini sangatlah diinginkan karena beberapa alasan akademis dan praktis:

**Pertama, dari perspektif praktis**, kebutuhan akan dataset berlabel besar untuk setiap tugas baru sangat membatasi penerapan model bahasa. Ada sangat banyak ragam tugas bahasa yang berguna, mulai dari mengoreksi tata bahasa, menghasilkan contoh dari konsep abstrak, hingga mengkritik cerita pendek. Untuk banyak tugas ini, sangat sulit mengumpulkan dataset pelatihan terselia (*supervised training dataset*) yang besar.

**Kedua, potensi model mengeksploitasi korelasi palsu (*spurious correlations*) dalam data pelatihan** secara fundamental tumbuh seiring dengan tingkat ekspresivitas model dan sempitnya distribusi pelatihan. Ini menciptakan masalah besar bagi paradigma *pre-training* + *fine-tuning*. Model dirancang menjadi sangat besar untuk menyerap informasi selama pra-pelatihan, namun kemudian di-*fine-tune* pada distribusi tugas yang sangat sempit. Bukti menunjukkan bahwa generalisasi yang dicapai bisa sangat buruk karena model menjadi terlalu spesifik pada data *training* dan gagal saat dihadapkan pada data di luar distribusi (*out-of-distribution*). Akibatnya, performa model *fine-tuned* pada *benchmark* tertentu seringkali melebih-lebihkan (exaggerate) performa aktualnya di dunia nyata.

**Ketiga, manusia tidak membutuhkan dataset terselia yang besar** untuk mempelajari sebagian besar tugas bahasa. Manusia hanya butuh arahan singkat dalam bahasa alami (misal: "tolong beritahu saya apakah kalimat ini mendeskripsikan sesuatu yang bahagia atau sedih") atau paling banyak sejumlah kecil demonstrasi (misal: "ini dua contoh orang bertindak berani; tolong berikan contoh ketiga"). Mengingat fleksibilitas ini, kita ingin sistem NLP di masa depan memiliki fluiditas dan generalitas yang sama.

Satu rute potensial untuk mengatasi masalah ini adalah **pembelajaran meta (*meta-learning*)**. Dalam konteks model bahasa, ini berarti model mengembangkan seperangkat keterampilan luas dan kemampuan pengenalan pola saat waktu pelatihan (*training time*), lalu menggunakan kemampuan tersebut saat waktu inferensi (*inference time*) untuk beradaptasi dengan cepat pada tugas yang diinginkan. 

Pendekatan ini dilakukan melalui apa yang kita sebut **"pembelajaran dalam-konteks" (*in-context learning*)**, menggunakan masukan teks (prompt) sebagai bentuk spesifikasi tugas: model dikondisikan pada instruksi bahasa alami dan/atau beberapa demonstrasi tugas, dan kemudian diharapkan untuk menyelesaikan turunan tugas tersebut hanya dengan memprediksi kata apa yang muncul selanjutnya. Namun, pendekatan ini sebelumnya masih menghasilkan performa yang jauh lebih inferior dibandingkan *fine-tuning* (misalnya model GPT-2 lama hanya mencapai 4% pada dataset *Natural Questions*).

Namun, tren terbaru menunjukkan jalan keluar: **Kapasitas model bahasa Transformer telah meningkat secara substansial**. Dari 100 juta parameter, ke 300 juta, ke 1,5 miliar, 8 miliar, 11 miliar, hingga 17 miliar parameter. Setiap peningkatan kapasitas selalu membawa peningkatan pada sintesis teks dan/atau tugas NLP hilir. Terdapat bukti kuat bahwa *loss* logaritma (yang berkorelasi dengan performa tugas hilir) mengikuti tren peningkatan yang mulus sejalan dengan penskalaan model (*scale*). Karena pembelajaran dalam-konteks melibatkan penyerapan banyak keterampilan ke dalam parameter model, sangat masuk akal bahwa kemampuan pembelajaran dalam-konteks akan menunjukkan keuntungan yang sama kuatnya dengan penskalaan model.

Makalah ini menguji hipotesis tersebut dengan melatih model bahasa autoregresif berkapasitas **175 miliar parameter** (GPT-3).

---

### Analisis Gambar 1.1: Pembelajaran Meta pada Model Bahasa (*Language Model Meta-Learning*)

Gambar 1.1 (di makalah asli) merepresentasikan hierarki konseptual dari cara model belajar.
*   **Sumbu/Alur:** Gambar ini mengilustrasikan "Outer Loop" (Siklus Luar) dan "Inner Loop" (Siklus Dalam).
*   **Outer Loop (Garis Panah Ungu):** Ini adalah proses **Pra-pelatihan tak terselia (*Unsupervised pre-training*)** menggunakan metode *Stochastic Gradient Descent* (SGD). Di sinilah model membaca miliaran teks dari internet dan perlahan-lahan memperbarui parameter bobotnya. Di fase inilah *meta-learning* terjadi (model menyerap berbagai pola bahasa).
*   **Inner Loop (Kotak-kotak di dalam):** Ini adalah **Pembelajaran Dalam-Konteks (*In-context learning*)**. Gambar ini menunjukkan contoh urutan (Sequence #1: penjumlahan matematika, Sequence #2: perbaikan ejaan/translasi, Sequence #3: penerjemahan dari Inggris ke Prancis). 
*   **Kesimpulan Akademis:** Inner loop terjadi *selama forward-pass* di tahap inferensi. Model menyadari ada sub-tugas yang berulang di dalam satu urutan (contoh: pola "thanks => merci", "hello => bonjour", lalu diminta melengkapi "cheese => ..."). Model mengadaptasi polanya secara *on-the-fly* (instan) tanpa mengubah bobot (gradient) apapun.

---

### Analisis Gambar 1.2: Efisiensi Penggunaan Informasi Dalam-Konteks oleh Model yang Lebih Besar

Gambar ini mendemonstrasikan bagaimana ukuran model mempengaruhi kemampuan adaptasi dari sebuah *prompt*.
*   **Sumbu X (Logaritmik):** Menunjukkan "Jumlah Contoh di dalam Konteks (K)". Berkisar dari 0 (Zero-shot) hingga lebih dari 10 (Few-shot).
*   **Sumbu Y:** Menunjukkan tingkat "Akurasi (%)" pada sebuah tugas manipulasi kata sederhana (menghapus simbol acak dari sebuah kata).
*   **Kurva Data:** Terdapat tiga garis yang merepresentasikan ukuran model: Garis hijau untuk model 1.3B Parameter, garis oranye untuk 13B Parameter, dan garis biru solid/putus-putus untuk 175B Parameter. 
*   **Garis Putus-putus vs Solid:** Garis putus-putus (*No Prompt*) berarti model hanya diberikan contoh pola. Garis solid (*Natural Language Prompt*) berarti model juga diberikan instruksi teks yang jelas (misal: "Hapus simbol dari kata ini").
*   **Kesimpulan Akademis:** Kurva untuk model raksasa (175B) menanjak jauh lebih curam (*steeper*) dibandingkan model kecil. Ini membuktikan bahwa model yang lebih besar membuat penggunaan informasi dalam-konteks menjadi jauh lebih efisien. Kinerja juga melonjak jika kita menggabungkan instruksi deskripsi tugas dan sejumlah besar contoh (K).

---

### Analisis Gambar 1.3: Agregat Performa Melintasi 42 Tolok Ukur

Gambar ini memberikan ringkasan heuristik dari kehebatan GPT-3.
*   **Sumbu X (Logaritmik):** Skala jumlah parameter model mulai dari 0.1B (100 Juta) hingga 175B (Miliar).
*   **Sumbu Y:** Akurasi rata-rata agregat melintasi 42 benchmark NLP.
*   **Kurva Data:** Ada tiga kurva. Kurva biru untuk *Zero-Shot*, kurva hijau untuk *One-Shot*, dan kurva oranye untuk *Few-Shot*.
*   **Kesimpulan Akademis:** Performa *Zero-shot* meningkat secara stabil dan mulus seiring bertambahnya ukuran model. Namun, yang luar biasa adalah performa *Few-shot* meningkat jauh lebih cepat/curam. Hal ini mengkonfirmasi hipotesis inti makalah: Semakin besar skala model, semakin mahir model tersebut menjadi *meta-learner* dalam pengaturan *in-context learning*.

---

Secara umum, pada tugas NLP, GPT-3 mencapai hasil yang menjanjikan dalam pengaturan *zero-shot* dan *one-shot*, dan pada pengaturan *few-shot*, ia kadang kompetitif atau bahkan melampaui SOTA dari model *fine-tuned*. Misalnya, GPT-3 mencapai akurasi **81.5 F1** pada CoQA (zero-shot), dan **71.2%** pada TriviaQA (few-shot) yang merupakan rekor SOTA baru untuk kategori *closed-book*. GPT-3 juga ahli dalam tugas adaptasi instan seperti manipulasi kata (unscrambling), aritmatika 3 digit, dan menghasilkan artikel berita sintetik yang hampir tidak dapat dibedakan dari tulisan manusia oleh evaluator manusia.

Namun, model ini masih kesulitan di beberapa tugas penalaran inferensi logika alami seperti ANLI, dan beberapa tugas komprehensi membaca seperti RACE.

---

## 2. Pendekatan (*Approach*)

Pendekatan pra-pelatihan dasar yang digunakan (termasuk model, data, dan proses pelatihan) serupa dengan pendahulunya (GPT-2), namun dengan penskalaan yang sangat masif. Karena fokusnya adalah *in-context learning*, para peneliti mendefinisikan spektrum pengaturan evaluasi sebagai berikut:

### Spektrum Pembelajaran Berdasarkan Ketergantungan Data Tugas:

1.  **Penyesuaian Halus (*Fine-Tuning / FT*):**
    Ini adalah pendekatan paling umum saat ini. Melibatkan pembaruan bobot model (gradient descent) dengan melatihnya pada dataset terselia khusus untuk tugas tertentu (ribuan hingga ratusan ribu contoh). Keuntungannya adalah performa *benchmark* yang sangat kuat. Kerugiannya: butuh dataset besar untuk *setiap* tugas, generalisasi yang buruk di luar distribusi data (*out-of-distribution*), dan eksploitasi korelasi data yang palsu. **Di makalah ini, GPT-3 TIDAK di-fine-tune sama sekali.**
    
2.  **Pembelajaran Sedikit-Contoh (*Few-Shot / FS*):**
    Model diberikan *sedikit demonstrasi* (contoh) dari tugas yang harus dikerjakan pada saat inferensi, **tanpa ada pembaruan bobot parameter**. Biasanya, $K$ contoh dimasukkan (berkisar antara 10 hingga 100, tergantung batas maksimum jendela konteks / *context window* model). Keuntungannya: pengurangan drastis kebutuhan data dan menghindari model menjadi terlalu sempit. Kerugiannya: hasil historisnya jauh lebih rendah dari SOTA FT.
    
3.  **Pembelajaran Satu-Contoh (*One-Shot / 1S*):**
    Sama persis seperti *Few-Shot*, tetapi model hanya diberikan tepat **1 (satu) demonstrasi**. Ini penting dievaluasi karena paling mirip dengan cara manusia memberikan instruksi satu sama lain (misalnya, memberikan tugas kepada pekerja manusia dengan satu contoh saja).
    
4.  **Pembelajaran Tembakan-Nol (*Zero-Shot / 0S*):**
    Tidak ada demonstrasi contoh sama sekali. Model HANYA diberikan instruksi dalam bahasa alami (misalnya: "Terjemahkan kalimat Bahasa Inggris ini ke Bahasa Prancis:"). Ini adalah pengaturan paling menantang dan "paling tidak adil" bagi mesin (bahkan terkadang manusia bingung tanpa contoh format), namun merupakan pengaturan yang paling tahan terhadap korelasi palsu (robust).

*(Catatan Dosen: Untuk memperjelas secara visual, Gambar 2.1 dalam teks mendemonstrasikan bahwa pada FT, terjadi "gradient update" berulang-ulang pada *backpropagation*. Sedangkan pada 0S, 1S, dan FS, kotak kuning "gradient update" ditiadakan, melainkan tugas sepenuhnya diselesaikan murni lewat "prompt" pada input/forward-pass).*

### 2.1 Model dan Arsitektur (*Model and Architectures*)

Arsitektur model menggunakan struktur yang sama persis dengan GPT-2 (termasuk inisialisasi yang dimodifikasi, pra-normalisasi, dan tokenisasi reversibel), dengan satu pengecualian penting: GPT-3 menggunakan pola perhatian yang berselang-seling antara *dense* (padat) dan *locally banded sparse attention* (perhatian jarang berkelompok lokal) pada lapisan-lapisan transformernya, mirip dengan arsitektur *Sparse Transformer*.

Untuk mempelajari hukum penskalaan (*scaling laws*), peneliti melatih 8 model dengan ukuran berbeda.

#### Tabel 2.1: Ukuran, Arsitektur, dan Hiper-parameter Pelatihan (8 Model)

| Nama Model | $n_{params}$ | $n_{layers}$ | $d_{model}$ | $n_{heads}$ | $d_{head}$ | Batch Size | Learning Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| GPT-3 Small | 125M | 12 | 768 | 12 | 64 | 0.5M | $6.0 \times 10^{-4}$ |
| GPT-3 Medium | 350M | 24 | 1024 | 16 | 64 | 0.5M | $3.0 \times 10^{-4}$ |
| GPT-3 Large | 760M | 24 | 1536 | 16 | 96 | 0.5M | $2.5 \times 10^{-4}$ |
| GPT-3 XL | 1.3B | 24 | 2048 | 24 | 128 | 1M | $2.0 \times 10^{-4}$ |
| GPT-3 2.7B | 2.7B | 32 | 2560 | 32 | 80 | 1M | $1.6 \times 10^{-4}$ |
| GPT-3 6.7B | 6.7B | 32 | 4096 | 32 | 128 | 2M | $1.2 \times 10^{-4}$ |
| GPT-3 13B | 13.0B | 40 | 5140 | 40 | 128 | 2M | $1.0 \times 10^{-4}$ |
| **GPT-3 175B** | **175.0B** | **96** | **12288** | **96** | **128** | **3.2M** | **$0.6 \times 10^{-4}$** |

**Penjelasan Variabel pada Tabel 2.1:**
*   $n_{params}$: Total jumlah parameter (bobot/weights) yang dapat dilatih dalam jaringan. (M = Juta/Million, B = Miliar/Billion). GPT-3 yang asli adalah versi 175B.
*   $n_{layers}$: Total jumlah lapisan (layers) blok Transformer. Model 175B memiliki 96 tumpukan lapisan. Sangat dalam.
*   $d_{model}$: Dimensi vektor representasi (ukuran ruang *embedding/bottleneck*). Pada model 175B, setiap token direpresentasikan oleh array angka sepanjang 12.288 dimensi. Dimensi *feedforward* selalu $4 \times d_{model}$.
*   $n_{heads}$: Jumlah kepala (heads) pada mekanisme *Multi-Head Attention* di tiap lapisan.
*   $d_{head}$: Dimensi proyeksi per-kepala. Dihitung secara implisit: $d_{head} = d_{model} / n_{heads}$.
*   *Batch Size*: Ukuran kumpulan token dalam satu kali kalkulasi gradien (dalam jumlah token). Model yang lebih besar memakai *batch size* yang lebih besar.
*   *Learning Rate*: Seberapa besar langkah pembaruan bobot dalam optimasi. Sesuai prinsip *scaling*, model yang lebih besar membutuhkan *learning rate* yang lebih kecil agar tidak tidak stabil/meledak.

Semua model menggunakan batas jendela konteks (*context window*) $n_{ctx} = 2048$ token.

### 2.2 Dataset Pelatihan (*Training Dataset*)

Dataset untuk melatih model ini berpusat pada **Common Crawl**, yang berisi triliunan kata hasil ekstraksi web (setelah disaring, ukurannya mencapai 570 GB atau sekitar 400 miliar token). Karena Common Crawl mentah memiliki kualitas yang bervariasi, peneliti melakukan 3 langkah perbaikan:
1.  Memfilter Common Crawl berdasarkan kemiripannya (*similarity*) dengan korpus teks referensi berkualitas tinggi.
2.  Melakukan de-duplikasi *fuzzy* (fuzzy deduplication) di tingkat dokumen untuk mencegah redundansi berlebihan yang bisa memicu memori absolut (*overfitting*).
3.  Mencampurkan korpus berkualitas tinggi lainnya ke dalam proses pelatihan (seperti dataset WebText2, dua dataset Buku, dan Wikipedia Bahasa Inggris) untuk meningkatkan keragaman.

#### Tabel 2.2: Komposisi Dataset untuk Pelatihan GPT-3

| Dataset | Kuantitas (Token) | Bobot Pencampuran (*Weight*) | Epok terlewati (Epochs) untuk 300B token |
| :--- | :--- | :--- | :--- |
| Common Crawl (difilter) | 410 miliar | 60% | 0.44 |
| WebText2 | 19 miliar | 22% | 2.9 |
| Books1 | 12 miliar | 8% | 1.9 |
| Books2 | 55 miliar | 8% | 0.43 |
| Wikipedia | 3 miliar | 3% | 3.4 |

**Penjelasan Tabel 2.2:**
Saat melatih, data tidak diambil sebanding dengan ukurannya. Dataset berkuantitas kecil namun berkualitas sangat tinggi (seperti Wikipedia) diberi bobot pengambilan (*weight*) yang sengaja diperbesar (3%), sehingga dokumen Wikipedia dibaca berulang-ulang (sampai 3,4 Epochs/putaran). Sebaliknya, Common Crawl yang masif hanya menyumbang 60% pengambilan pelatihan, sehingga secara rata-rata sebuah dokumen Common Crawl tidak akan dibaca lebih dari satu kali (0.44 Epochs). Ini adalah bentuk *trade-off* di mana peneliti menerima sedikit *overfitting* pada Wikipedia demi mendapatkan standar kualitas yang baik.

### 2.3 Proses Pelatihan (*Training Process*)

Sejalan dengan literatur *Scaling Laws*, model yang lebih besar memerlukan *batch size* yang lebih besar pula, tetapi dengan *learning rate* yang lebih kecil. Model-model ini dilatih melintasi berbagai kluster GPU V100 menggunakan campuran pemisahan paralelisasi model (*model parallelism*) di dalam matriks dan di antara lapisan-lapisan jaringan.

#### Analisis Gambar 2.2: Total Komputasi yang Digunakan Selama Pelatihan
Gambar ini berbentuk diagram batang (*bar chart*) berskala logaritmik vertikal.
*   **Sumbu X:** Menampilkan deretan model-model SOTA komparator (seperti BERT-Base, RoBERTa-Large, T5-11B) dan jajaran 8 varian model GPT-3 (mulai dari GPT-3 Small hingga GPT-3 175B).
*   **Sumbu Y (Logaritmik):** Total komputasi pelatihan, diukur dalam satuan Petaflop/s-days. (1 Petaflop/s-days = melakukan kuadriliunan komputasi per detik tanpa henti selama 24 jam).
*   **Tren Data:** Model-model komparator lama menghabiskan antara 1 hingga maksimal 100-an Petaflop/s-days (misalnya T5-11B). Namun, untuk GPT-3 175B, batang biru paling kanan melesat tinggi menyentuh orde ribuan, tepatnya mencapai angka di atas **3.600 Petaflop/s-days**. 
*   **Kesimpulan Akademis:** Pelatihan GPT-3 175B merepresentasikan investasi komputasi yang belum pernah terjadi sebelumnya dalam literatur AI. Menariknya, GPT-3 3B ternyata menghabiskan komputasi yang mirip dengan RoBERTa-Large. Walau GPT-3 3B memiliki parameter 10x lebih banyak, GPT-3 dilatih pada jumlah token yang jauh lebih sedikit secara proporsional berkat panduan *Scaling Laws*.

### 2.4 Evaluasi (*Evaluation*)

Untuk pengaturan *few-shot*, peneliti mengevaluasi setiap contoh dengan mengambil $K$ contoh secara acak dari kumpulan pelatihan tugas tersebut untuk digunakan sebagai bahan "pengkondisian" (*conditioning*), dipisahkan oleh 1 atau 2 enter (newline). Nilai $K$ dapat berapa saja selama belum menembus batas memori jendela konteks ($n_{ctx} = 2048$), biasanya muat sekitar 10 hingga 100 contoh. 

Pada tugas bertipe *multiple choice* (pilihan ganda), model diberikan $K$ contoh soal+jawaban yang benar. Kemudian pada akhir *prompt*, diberikan 1 soal murni. Model lalu menghitung *likelihood* (kemungkinan logaritmik probabilistik) terhadap *setiap opsi* pilihan yang tersedia. Jawaban dengan kemungkinan tertinggi dinyatakan sebagai tebakan mesin. 

Untuk beberapa tugas (seperti ARC dan OpenBookQA), peneliti menggunakan teknik normalisasi probabilitas tak bersyarat, menggunakan rumus:

$$ \text{Normalized\_Score} = \frac{P(\text{completion} | \text{context})}{P(\text{completion} | \text{answer\_context})} $$

**Penjelasan Matematis Formula:**
*   $P(\text{completion} | \text{context})$: Probabilitas yang diberikan oleh model untuk opsi jawaban tersebut (completion), *dengan syarat* model telah membaca teks soal secara penuh (context).
*   $P(\text{completion} | \text{answer\_context})$: Probabilitas *a priori* (probabilitas dasar/tak bersyarat) yang diberikan model untuk opsi jawaban tersebut, *hanya dengan syarat* model membaca teks statis "Answer: " atau "A: " tanpa membaca teks soal sama sekali.
*   **Mengapa butuh dibagi (dinormalisasi)?** Karena seringkali model bahasa memiliki bias intrinsik terhadap kata-kata yang populer atau kalimat yang panjang. Pilihan ganda yang panjang mungkin memiliki probabilitas dasar yang lebih rendah secara natural. Membagi probabilitas jawaban dengan probabilitas dasar jawaban itu sendiri akan menetralisir "bias kemiripan tata bahasa" tersebut.

Pada tugas *free-form completion* (pengisian bentuk bebas yang panjang), model diinstruksikan melakukan *beam search* (pencarian sinar dengan lebar *beam width* = 4 dan penalti panjang $\alpha = 0.6$). Kinerja lalu dihitung menggunakan F1 score, BLEU, atau *Exact Match* (Kecocokan Persis).

---

## 3. Hasil (*Results*)

### Analisis Gambar 3.1: Skala Performa yang Mulus Terhadap Komputasi (Kurva Latihan)

Grafik pada Gambar 3.1 memvisualisasikan bagaimana penambahan komputasi berdampak pada kinerja intrinsik bahasa.
*   **Sumbu X (Logaritmik):** Total komputasi yang dihabiskan dalam satuan PetaFLOP/s-days. Dari $10^{-6}$ hingga $10^4$.
*   **Sumbu Y (Linear):** Validation Loss (Kerugian Validasi / Error) lintas-entropi (*cross-entropy loss*). 
*   **Bilah Warna / Skala kanan:** Menunjukkan ukuran Parameter model dari $10^5$ (ratusan ribu) ke $10^{11}$ (ratusan miliar). Garis-garis kurva berwarna ungu (model kecil) hingga kuning-hijau terang (model raksasa 175B).
*   **Rumus Regresi Kurva Putus-putus:** 
    $$ L = 2.57 \cdot C^{-0.048} $$
    *Penjelasan Rumus:* $L$ adalah Validation Loss, dan $C$ adalah Komputasi. Angka eksponen $-0.048$ menunjukkan sifat *power-law* (hukum pangkat). Artinya, setiap kali komputasi ($C$) dilipatgandakan beberapa derajat, *Loss* ($L$) akan turun secara matematis terprediksi pada *slope* yang stabil sebesar -0.048.
*   **Kesimpulan Akademis:** Kemampuan jaringan model memprediksi bahasa secara akurat mengikuti hukum pangkat empiris dengan ketaatan yang nyaris sempurna, tanpa menunjukkan tanda-tanda pelambatan (*plateau*) bahkan setelah ditambahkan dua orde magnitudo komputasi baru. Penurunan error lintas-entropi inilah yang kelak membuahkan kinerja generalisasi pada tugas-tugas di dunia nyata.

---

### 3.1 Pemodelan Bahasa, Melengkapi Rumpang, dan Tugas Penyelesaian (*Language Modeling, Cloze, and Completion Tasks*)

Peneliti pertama-tama mengukur kinerja fundamental: memprediksi sepotong teks bahasa.

**3.1.1 Pemodelan Bahasa Murni (Language Modeling)**
Menggunakan dataset historis Penn Tree Bank (PTB). Performa dihitung menggunakan probabilitas keterkejutan / *perplexity*. (Semakin rendah angka perplexity, semakin baik pemahaman mesin).

#### Tabel 3.1: Hasil Tembakan-Nol (Zero-shot) pada Dataset PTB
| Pengaturan (Setting) | PTB (*Perplexity*) |
| :--- | :--- |
| SOTA (Zero-Shot) (Radford et al., 2019) | 35.8 |
| **GPT-3 Zero-Shot** | **20.5** |

*Catatan: GPT-3 mengalahkan SOTA yang ada dengan margin absolut 15 poin, memecahkan rekor prediksi bahasa purba ini. Evaluasi dilakukan secara Zero-Shot karena dataset PTB berformat pemodelan tradisional yang tak dapat dipisahkan menjadi demonstrasi few-shot.*

**3.1.2 LAMBADA**
LAMBADA adalah tes di mana model diuji memodelkan ketergantungan jarak-jauh (*long-range dependencies*). Model disuruh membaca sebuah paragraf teks yang sangat panjang, dan harus menebak *tepat satu kata terakhir* di paragraf tersebut.

#### Tabel 3.2: Performa pada Tugas Melengkapi (*Cloze and Completion*)
| Pengaturan | LAMBADA (Akurasi %) | LAMBADA (*Perplexity*) | StoryCloze (Akurasi %) | HellaSwag (Akurasi %) |
| :--- | :---: | :---: | :---: | :---: |
| SOTA Sebelumnya | 68.0 | 8.63 | 91.8 | 85.6 |
| GPT-3 Zero-Shot | 76.2 | 3.00 | 83.2 | 78.9 |
| GPT-3 One-Shot | 72.5 | 3.35 | 84.7 | 78.1 |
| **GPT-3 Few-Shot** | **86.4** | **1.92** | 87.7 | 79.3 |

**Analisis Hasil LAMBADA:**
Secara tradisional, model bahasa standar punya titik buta: ia tidak mengerti bahwa jawaban yang diharapkan LAMBADA *hanya* satu kata. Ia sering berhalusinasi meneruskan kalimat tersebut. Namun, dalam pengaturan *Few-Shot*, kita dapat memberikan "format/framing" isian rumpang khusus:
```text
Alice was friends with Bob. Alice went to visit her friend ____. -> Bob
George bought some baseball equipment, a ball, a glove, and a ____. ->
```
Format demonstrasi visual *fill-in-the-blank* ini membuat GPT-3 175B memecahkan rekor dunia (Akurasi 86.4%, melampaui SOTA sebelumnya 68%). 

#### Analisis Gambar 3.2: Kurva LAMBADA
*   **Sumbu X:** Skala Model dari 0.1B ke 175B.
*   **Sumbu Y:** Akurasi prediksi LAMBADA. Terdapat 3 garis (Zero-shot biru, One-shot hijau, Few-shot oranye). Terdapat dua garis horizon: Garis putus-putus bawah "Zero-Shot SOTA" (sekitar 68%) dan garis batas puncak "Human" manusia (di atas 95%).
*   **Kesimpulan Akademis:** Kurva model *Few-shot* melonjak ekstrem menembus batas SOTA lama bahkan sejak model berkapasitas 2.7B parameter. Pada 175B, GPT-3 menekan jarak (*gap*) terhadap performa Manusia. Terlihat jelas pola bahwa pada model kecil (0.1B), setting Few-shot justru merusak akurasi (kinerja terburuk di antara ketiganya). Mengapa? Karena model kecil belum memiliki abstraksi intelektual untuk mengenali "pola template". Namun saat kapasitas menembus batas tertentu, *Few-shot* meroket menjadi strategi pemenang.

**3.1.3 HellaSwag & StoryCloze**
*   **HellaSwag:** Dataset adversarial di mana model harus memilih akhiran cerita/instruksi yang logis. Walau hasil GPT-3 membaik (79.3%), ia masih kalah dari SOTA multi-task *fine-tuned* ALUM (85.6%).
*   **StoryCloze:** Memilih kalimat kelima yang masuk akal dari cerita 4 kalimat. GPT-3 *Few-Shot* mencapai 87.7%. Kalah dari SOTA *fine-tuned* BERT (91.8%), namun GPT-3 menaikkan *baseline zero-shot* hingga 10%.

---

### 3.2 Tanya Jawab Buku Tertutup (*Closed Book Question Answering*)

Tugas ini mengukur seberapa luas "wawasan faktual" (*factual knowledge*) dunia yang diam-diam terekam di dalam parameter bobot otak GPT-3. Model tidak diberi akses internet atau teks artikel Wikipedia untuk mencari jawaban (*Closed Book*). Jika ditanya "Kapan perang diponegoro terjadi?", model harus menggali ingatannya. 

Tolok ukur yang digunakan: Natural Questions (NQs), WebQuestions (WebQS), dan TriviaQA.

#### Tabel 3.3: Hasil pada Tiga Tugas Open-Domain QA (Tanya Jawab)

| Pengaturan | NaturalQS | WebQS | TriviaQA |
| :--- | :---: | :---: | :---: |
| RAG (*Fine-tuned, Open-Domain / Ada Search Engine*) | 44.5 | 45.5 | 68.0 |
| T5-11B+SSM (*Fine-tuned, Closed-Book*) | 36.6 | 44.7 | 60.5 |
| T5-11B (*Fine-tuned, Closed-Book*) | 34.5 | 37.4 | 50.1 |
| **GPT-3 Zero-Shot** | 14.6 | 14.4 | 64.3 |
| **GPT-3 One-Shot** | 23.0 | 25.3 | 68.0 |
| **GPT-3 Few-Shot** | 29.9 | 41.5 | **71.2** |

*(Performa diukur menggunakan akurasi kecocokan eksak / Exact Match).*

**Analisis Akademis Tanya Jawab:**
Kinerja GPT-3 sangat bervariasi bergantung distribusinya.
1.  **TriviaQA:** GPT-3 *Zero-shot* (64.3%) saja sudah menghancurkan T5-11B yang di-*fine-tune* khusus (60.5%). Lebih gila lagi, pada pengaturan *Few-Shot* (71.2%), GPT-3 **mengalahkan model RAG** (model canggih yang diizinkan untuk menjelajah internet (retrieval) membedah 21 Juta dokumen). Otak parameter GPT-3 lebih pintar dari pada mesin pencari khusus tersebut.
2.  **WebQuestions (WebQS) & Natural Questions (NQs):** Di sini GPT-3 kalah dibanding SOTA model spesialis *fine-tune*. NQs misalnya, fokus pada detail renik dari Wikipedia (seperti "Siapa yang mengisi suara karakter X di film Y?"). Model kalah telak di *Zero-Shot*, namun terlihat kemampuan beradaptasi drastis saat dibantu *Few-Shot* (meningkat dari 14% menjadi 41% di WebQS). Ini karena format tanya-jawabnya sangat sempit dan bergantung pada gaya bahasa datasetnya.

#### Analisis Gambar 3.3: TriviaQA Scaling
*   **Sumbu X:** Skala Model dari 0.1B ke 175B.
*   **Sumbu Y:** Akurasi (%) pada soal TriviaQA.
*   **Garis Referensi Horizontal:** Terdapat garis putus-putus berlabel "Fine-tuned SOTA" di kisaran angka 68%.
*   **Kurva Data:** Tiga kurva konvergen ke atas (biru, hijau, oranye). 
*   **Kesimpulan:** Akurasi menjawab pengetahuan faktual naik bagai garis lurus menyilang sejalan membesarnya ukuran jaringan model. Pengetahuan ensiklopedis dunia diserap sempurna oleh matriks bobot seiring dengan penambahan kapasitas miliaran angka.

---

### 3.3 Penerjemahan (*Translation*)

Pada model GPT-2 masa lalu, kemampuan multilingual secara teknis dihambat secara sengaja oleh arsitek pengembangnya (dengan pemfilteran dominan Bahasa Inggris). GPT-3, dengan korpus data mentah super raksasa, mengabsorpsi 7% bahasa non-Inggris yang ada di web tanpa disengaja (termasuk Prancis, Jerman, dan Romania).

Peneliti menggunakan kumpulan data klasik penerjemahan dari WMT. Karena GPT-3 memakai algoritma tokenisasi BPE tingkat byte yang berbasis Bahasa Inggris, kemampuan translasi *ke luar* (into foreign) diprediksi cacat, sedangkan menerjemahkan bahasa asing *ke dalam bahasa Inggris* diprediksi unggul.

#### Tabel 3.4: Performa Penerjemahan Translasi Multilingual WMT (BLEU Score)

| Pengaturan | En $\rightarrow$ Fr | Fr $\rightarrow$ En | En $\rightarrow$ De | De $\rightarrow$ En | En $\rightarrow$ Ro | Ro $\rightarrow$ En |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| SOTA (*Supervised/Pelatihan Khusus*) | **45.6** | 35.0 | **41.2** | 40.2 | **38.5** | **39.9** |
| XLM (*Unsupervised*) | 33.4 | 33.3 | 26.4 | 34.3 | 33.3 | 31.8 |
| MASS (*Unsupervised*) | 37.5 | 34.9 | 28.3 | 35.2 | 35.2 | 33.1 |
| mBART (*Unsupervised*) | - | - | 29.8 | 34.0 | 35.0 | 30.5 |
| **GPT-3 Zero-Shot** | 25.2 | 21.2 | 24.6 | 27.2 | 14.1 | 19.9 |
| **GPT-3 One-Shot** | 28.3 | 33.7 | 26.2 | 30.4 | 20.6 | 38.6 |
| **GPT-3 Few-Shot** | 32.6 | **39.2** | 29.7 | **40.6** | 21.0 | 39.5 |

**Analisis Tabel 3.4 dan Gambar 3.4 (Kurva Translasi):**
*   (Dari Gambar 3.4 - Kurva Translasi, Sumbu Y adalah Multi-BLEU score). Ada 6 grafik kecil untuk setiap rute bahasa yang menanjak mulus sesuai ukuran model. Arah panah "-> English" selalu memimpin di atas (garis solid) dibanding arah "English ->" (garis putus).
*   Pada metode *Zero-Shot*, GPT-3 kebingungan.
*   Pada metode *Few-Shot* (memasukkan kamus konversi beberapa kalimat dalam konteks/prompt), performa GPT-3 membengkak 7 BLEU poin.
*   Luar biasanya, pada rute menerjemahkan bahasa asing ke Bahasa Inggris (**Fr $\rightarrow$ En**, **De $\rightarrow$ En**, **Ro $\rightarrow$ En**), strategi *Few-Shot* GPT-3 murni yang tanpa dilatih khusus ini **mengalahkan SOTA Supervised (Terselia)** dan *Unsupervised* dunia mana pun (mencapai 39.2 pada Prancis dan 40.6 pada Jerman). Namun pada rute sebaliknya (seperti English $\rightarrow$ Romanian), GPT-3 babak belur di angka 21.0 BLEU karena masalah tokenisasi.

---

### 3.4 Tugas Gaya-Winograd (*Winograd-Style Tasks*)

Winograd Schemas Challenge [LDM12] adalah tugas NLP klasik untuk menguji pemahaman referensi kata ganti (*pronoun resolution*). Tugas ini berisi kalimat yang tata bahasanya ambigu tetapi secara logika semantik sangat jelas bagi otak manusia. (Misalnya menentukan rujukan kata "mereka" dalam kalimat jebakan). Tolok ukurnya meliputi: Winograd asli, dan versi garangan kompetisi adversarial *Winogrande*.

#### Tabel 3.5: Hasil pada Winograd dan Winogrande (XL)

| Pengaturan | Winograd (Akurasi) | Winogrande XL (Akurasi) |
| :--- | :---: | :---: |
| Fine-tuned SOTA | **90.1** | **84.6** |
| GPT-3 Zero-Shot | 88.3 | 70.2 |
| GPT-3 One-Shot | 89.7 | 73.2 |
| GPT-3 Few-Shot | 88.6 | 77.7 |

*Catatan: Metode "Partial Evaluation" dipakai, yakni menghitung perbedaan probabilitas akhir kalimat (*completion*) saat disisipkan kata yang salah versus kata yang benar.*

#### Analisis Gambar 3.5: Performa Adversarial Winogrande
*   **Sumbu X:** Parameter (0.1B ke 175B).
*   **Sumbu Y:** Akurasi (%) dari 50 ke 90. 
*   **Garis Referensi:** Garis atas (Human: ~94%), Garis SOTA Fine-tuned (~84.6%), Garis Fine-tuned RoBERTa-Large (~79%). Garis bawah batas rambang kebetulan (*Random Guessing* = 50%).
*   **Kesimpulan:** Model Winogrande diciptakan khusus untuk meruntuhkan bahasa AI. Model-model kecil hanya menebak membabi-buta di garis probabilitas acak 50%. Tiba di orde parameter Miliar (1.3B ke atas), kurva lepas landas membelah menanjak diagonal. Pada GPT-3 175B *Few-Shot*, model menyentuh angka 77.7%, setara dengan *Fine-tuned* RoBERTa-Large, meski belum memegang takhta tertinggi SOTA dari model T5. Di sini *in-context learning* (perbandingan antara 0-shot dan few-shot) menyumbang keuntungan yang jelas.

---

### 3.5 Penalaran Akal Sehat (*Common Sense Reasoning*)

Tugas penalaran akal sehat mencakup tiga tes fisika dan sains dunia nyata. 
1. **PIQA (PhysicalQA):** Pertanyaan tentang bagaimana hukum dasar fisika dunia bekerja.
2. **ARC:** Ujian pilihan ganda mata pelajaran sains dari kelas 3 SD hingga 9 SMP.
3. **OpenBookQA:** Tanya jawab sains fundamental.

#### Tabel 3.6: Hasil pada Tugas Penalaran Akal Sehat

| Pengaturan | PIQA | ARC (Easy) | ARC (Challenge) | OpenBookQA |
| :--- | :---: | :---: | :---: | :---: |
| Fine-tuned SOTA | 79.4 | **92.0** | **78.5** | **87.2** |
| GPT-3 Zero-Shot | 80.5 | 68.8 | 51.4 | 57.6 |
| GPT-3 One-Shot | 80.5 | 71.2 | 53.2 | 58.8 |
| GPT-3 Few-Shot | **82.8** | 70.1 | 51.5 | 65.4 |

#### Analisis Gambar 3.6: Hasil pada PIQA
*   **Sumbu X:** Logaritma Kapasitas (0.1B - 175B).
*   **Sumbu Y:** Akurasi PIQA (50% ke 90%). Garis horizon "Fine-tuned SOTA" ada di ~79.4%.
*   **Kesimpulan:** Pada ujian fisika PIQA ini, GPT-3 menembus awan dan memecahkan rekor SOTA secara langsung pada semua *setting* (termasuk *Zero-Shot* yang tembus 80.5%). Pada mode *Few-Shot*, ia mencapai 82.8%. Ini mengindikasikan narasi sains fisik sangat kentara dan banyak terekam dalam internet, yang diekstraksi mulus oleh GPT-3.
*   **Kekalahan di ARC & OpenBookQA:** Sebaliknya, untuk soal ujian sains sekolahan (ARC dan OpenBookQA), model tidak sanggup meruntuhkan dominasi *UnifiedQA* (SOTA yang di-*fine-tune* pada puluhan dataset sains). GPT-3 tertinggal 27% poin dari SOTA ARC Challenge.

---

### 3.6 Komprehensi Membaca (*Reading Comprehension*)

Berupa rangkaian format membaca panjang: dialog, soal pilihan, pemilihan frasa (span-based).
1.  **CoQA:** Dialog bebas (menjawab soal berdasarkan history dialog teks).
2.  **DROP:** Penalaran angka/matematika tersembunyi di dalam teks paragraf.
3.  **QuAC:** Interaksi guru-murid yang rumit.
4.  **SQuADv2:** Ekstraksi kalimat Wikipedia.
5.  **RACE (High/Mid):** Ujian bahasa inggris SMA Tiongkok.

#### Tabel 3.7: Hasil pada Komprehensi Membaca (Metrik F1 / Akurasi RACE)

| Pengaturan | CoQA | DROP | QuAC | SQuADv2 | RACE-h | RACE-m |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Fine-tuned SOTA | **90.7** | **89.1** | **74.4** | **93.0** | **90.0** | **93.1** |
| GPT-3 Zero-Shot | 81.5 | 23.6 | 41.5 | 59.5 | 45.5 | 58.4 |
| GPT-3 One-Shot | 84.0 | 34.3 | 43.3 | 65.4 | 45.9 | 57.4 |
| GPT-3 Few-Shot | 85.0 | 36.5 | 44.3 | 69.8 | 46.8 | 58.1 |

#### Analisis Gambar 3.7: Kurva CoQA
*   Grafik menunjukkan sumbu akurasi F1 terhadap ukuran model untuk dataset CoQA. Tiga kurva Z, 1, dan F semuanya membumbung ke kisaran 80-85%. Garis horizon "Human" dan "Fine-Tuned SOTA" berdampingan erat di atas 90%.
*   **Kesimpulan:** Pada dataset bergaya diskusi obrolan (CoQA), GPT-3 mencapai angka spektakuler 85.0 F1 (Few-Shot), menempel ketat sekitar 5 poin saja dari level intelek Manusia dan SOTA. 
*   **Tragedi DROP & QuAC:** Sebaliknya pada DROP (yang memaksa mesin menghitung logika aritmatika eksplisit lewat narasi bacaan) dan QuAC (pemilihan span teks khusus), insting natural GPT-3 tumpul. Parameter sebanyak apapun tidak serta-merta membimbing model membedah trik mekanis ujian bacaan yang tak selaras dengan gaya bahasa normal internet, meninggalkannya kalah tragis (36.5 F1 vs SOTA 89.1 F1).

---

### 3.7 SuperGLUE

SuperGLUE adalah kompilasi standar mutlak dari kumpulan banyak uji NLP berstandar tinggi. Meliputi BoolQ, CB, COPA, RTE, WiC, WSC, MultiRC, dan ReCoRD. Pada mode *Few-Shot*, 32 contoh demonstrasi disuntikkan secara acak dari data latih mereka.

#### Tabel 3.8: Performa GPT-3 pada SuperGLUE (Akurasi/F1 Test Set)

*(Tabel diringkas menjadi deskriptif analitik karena batasan lebar teks, intinya adalah:)*
SOTA rata-rata agregat (*SuperGLUE Average*) dipegang oleh model *Fine-tuned* dengan skor **89.0**.
Sedangkan BERT-Large (*Fine-tuned*) memegang angka rata-rata **69.0**.
GPT-3 (*Few-Shot*) menyentuh agregat **71.8**.

#### Analisis Gambar 3.8: Peningkatan SuperGLUE Terhadap Penskalaan Model dan Konteks
Terdapat dua grafik di Gambar 3.8.
*   **Grafik Kiri (Scale vs Performance):** Sumbu X parameter (0.1B ke 175B). Sumbu Y SuperGLUE Score. Garis referensi Fine-Tuned BERT-Large dan BERT++ di level kisaran 70. Garis atas SOTA dan Manusia memuncak rata di ~90. Kurva GPT-3 Few-Shot (Oranye) memotong garis referensi BERT tepat setelah melewati model raksasa 13B parameter, dan mendarat di skor ~73 pada validasi (Dev Set).
*   **Grafik Kanan (Number of Examples (K) vs Performance):** Sumbu X log basis 2 merepresentasikan jumlah contoh konteks (K) (dari 0, 1, 2, 4, ..., hingga 32). Sumbu Y SuperGLUE Score.
*   **Kesimpulan Akademis Grafik Kanan:** Ini adalah inti magis dari *In-Context Learning*. Garis oranye menanjak stabil seiring diinjeksinya lebih banyak contoh soal "pancingan" ke dalam memori sekilas (konteks jendela). Saat K menyentuh 8 demonstrasi, GPT-3 175B (*zero-gradient*) resmi melampaui BERT-Large yang di-*fine-tune* secara invasif pada 630.000 data pelatihan!
*   **Kelemahan Fatal:** Pada bagian **WiC** (menentukan apakah satu kata dalam dua kalimat bermakna ekuivalen), GPT-3 jatuh buta menebak acak pada level koin toss (49.4%). Hal ini menelanjangi satu cacat besar: autoregresi *causal* tanpa atensi multi-arah (*bidirectional* / gaya BERT) membuat GPT-3 sangat buruk membandingkan parafrasa atau ikatan dua pernyataan (*implies/paraphrase*).

---

### 3.8 NLI (Natural Language Inference)

NLI (seperti dataset RTE dan ANLI) menguji mesin untuk menebak hubungan logis antara 2 kalimat: "Mendukung" (*Entailment*), "Bertolak Belakang" (*Contradiction*), atau "Netral" (*Neutral*). 

#### Analisis Gambar 3.9: Performa pada ANLI Round 3
*   Dataset ANLI R3 sangat "kurang ajar" karena soalnya sengaja diracik pakar (*adversarially mined*) agar model AI lama pasti gagal.
*   **Sumbu X:** Skala Model. **Sumbu Y:** Akurasi klasifikasi (33% hingga 48%). 33% adalah persentase menebak acak karena opsi ganda bernilai 3 opsi.
*   **Garis Referensi:** SOTA teratas bernilai kurang lebih 48-49%, dan BERT bernilai mendekati 44%.
*   **Kesimpulan Grafik:** Untuk model yang lebih kecil dari 13 Miliar, kurva menggelepar rata di angka kematian 33% (pure kebetulan acak). Namun, layaknya "Gemerlap Kehidupan", kurva GPT-3 175B *Few-Shot* (garis oranye) meledak ke atas, tiba-tiba mampu menyerap kepingan logika hingga menyentuh angka lebih dari 40%. Walau belum sekelas SOTA, kebangkitan kurva dari dasar jurang membuktikan skala parameter bisa membuka gerbang penalaran inferensial tak terduga (*signs of life*).

---

### 3.9 Tugas Sintetis dan Kualitatif (*Synthetic and Qualitative Tasks*)

Di bagian ini, para penguji menginterogasi daya cipta kognitif GPT-3 yang murni. Mengujinya pada tugas komputasional "berpikir instan" (*on-the-fly computational reasoning*). Jika di internet tak ada rekam jejak untuk dicontek, mampukah matriks matematik ini bernalar sejati?

**3.9.1 Aritmatika (*Arithmetic*)**
Tim OpenAI mem-fabrikasi tes hitung-hitungan manual: 2 digit (+/-/*), 3 digit (+/-), hingga 5 digit operasi gabungan. Format kodenya murni interaksi teks polos: 
`"Q: What is 48 plus 76? A: 124."`

#### Analisis Gambar 3.10: Akurasi Matematika GPT-3 dalam *Few-Shot*
*   **Sumbu X:** Ukuran Model (0.1B ke 175B).
*   **Sumbu Y:** Akurasi Jawaban Benar 100% Eksak. Nilai dari 0 sampai 100.
*   **Kurva Data:** Terdapat 10 lintasan tali berwarna-warni yang mewakili ujian operasi (Addition/Penjumlahan biru, Subtraction/Pengurangan oranye/merah, Multiplication/Perkalian ungu kusam).
*   **Kesimpulan Spektakuler:** Semua model di bawah 13B parameter buta huruf aritmatika absolut (Akurasi tidur lelap di angka nol persen, dan hanya naik ke belasan persen pada penjumlahan 2 digit). Tiba di versi raksasa 175B, terjadi tebing kejut (quantum leap):
    *   **2 Digit (+):** Sukses tembus **100%**.
    *   **2 Digit (-):** Sukses tembus **98.9%**.
    *   **3 Digit (+):** **80.2%** akurat.
    *   **2 Digit Perkalian (*):** GPT-3 mampu menyelesaikan soal otak ekstrem 2 digit kali 2 digit tanpa kalkulator murni mengandalkan probabilitas kata berakselerasi **29.2%**.
*   *Uji Memorisasi:* OpenAI melacak jejak rekam soal 3-digit ke dalam sejarah pelatihan internet, dan hanya menemui angka kebocoran konyol 0.8% (17 baris dari 2.000). Kesalahan bodoh mesin yang sering gagal menerapkan ilmu "menyimpan 1 ke angka depan/carry a 1" menguatkan bukti sains bahwa ia bukan menyontek memori tabel (*memorizing table*), namun secara tulus *berusaha memanipulasi kalkulasi secara kognitif di ruang berdimensi vektor tinggi*.

**3.9.2 Pengacakan dan Manipulasi Kata (*Word Scrambling and Manipulation*)**
Tim merancang eksperimen manipulasi karakter (symbolic manipulations):
1.  **CL (Cycle letters):** `lyinevitab -> inevitably`
2.  **A1 (Anagram 1):** Mengacak semua huruf kecuali huruf pertama & terakhir (`criroptuon -> corruption`)
3.  **A2 (Anagram 2):** Mengacak huruf bagian dalam (`opoepnnt -> opponent`)
4.  **RI (Random Insertion):** `s.u!c/c!e.s s i/o/n -> succession`
5.  **RW (Reversed words):** `stcejbo -> objects`

#### Analisis Gambar 3.11: Hasil Manipulasi Kata Few-Shot
*   **Sumbu X:** Ukuran Model. **Sumbu Y:** Akurasi tebakan (0 - 70%).
*   **Kesimpulan:** Tren kebangkitan kognisi tampak jelas mengalir melintasi model 17B hingga 175B. RI (Random Insertion) paling mudah dibongkar pasang (tembus 66.9% di 175B). Mengagumkannya, algoritma tokenizer "BPE" yang digunakan sistem itu secara asali menggabungkan/memadatkan huruf menjadi blok sub-kata berukuran 0.7 kata-per-token. Karenanya, tugas ini sebetulnya "tugas mustahil", karena ia tidak melihat aksara sebagai A-B-C, tapi bongkahan matriks. GPT-3 175B secara gaib mampu *mendekonstruksi arsitektur internal anatomi blok token* untuk memecahkan sandi acak. Hanya tugas "Kata Dibalik" (RW) yang gagal menembus angka nol persen (akibat arsitektur alur maju autoregresif).

**3.9.3 Analogi Ujian SAT**
Sistem disodorkan soal logika padanan kata ujian masuk Perguruan Tinggi Amerika klasik (era pra-2005). 
Format: `"audacious is to boldness as (a) sanctimonious is to hypocrisy, (b) anonymous is to identity..."` (Opsi A benar).

#### Analisis Gambar 3.12: Kinerja Analogi Ujian SAT
*   **Grafik:** Kurva biru (Zero-shot), Hijau (One-shot), Oranye (Few-shot K=20) vs Ukuran Model.
*   **Kesimpulan:** Rata-rata populasi Calon Mahasiswa pendaftar kuliah AS mencetak skor sekitar 57%. Sistem GPT-3 175B dalam mode *Few-Shot* melibas standar calon mahasiswa tersebut dengan akurasi **65.2%**. Pola *in-context learning* sungguh nyata di sini: kurva oranye tiba-tiba mendobrak garis silang melampaui kurva hijau dan biru di saat skala membesar dari 13B menuju 175B.

**3.9.4 Pembangkitan Artikel Berita Sintetik (*News Article Generation*)**
GPT-3 digunakan untuk menghasilkan tulisan artikel berita buatan (news article) lengkap setelah sekadar disuapi kepingan judul awal. Tim melakukan tes Turing (*Turing test*) terhadap ratusan responden manusia warga AS, meminta mereka menebak: "Artikel mana yang diketik Manusia vs Artikel karangan Mesin?".

#### Tabel 3.11 & 3.12: Kemampuan Responden Manusia Mengenali Artikel Buatan (Akurasi Sekitar ~200 dan ~500 kata)
| Model | Akurasi Manusia Mengenali Kepalsuan | Keterangan |
| :--- | :---: | :--- |
| Model Kontrol (Buruk) | 86% | Sangat mudah ditebak manusia sebagai palsu |
| GPT-3 Small | 76% | - |
| GPT-3 6.7B | 60% | Mulai sulit dibedakan |
| **GPT-3 175B** | **52%** | **Setara Acak / Lempar Koin (Batas Tebak Random: 50%)** |

#### Analisis Gambar 3.13: Kekalahan Penilaian Manusia
*   **Grafik:** Akurasi responden (Y) terhadap ukuran parameter model (X - skala log basis 10). Terdapat garis menurun (slope lurus merosot turun tajam). 
*   **Kesimpulan:** Grafik membuktikan tragedi kognitif umat manusia. Makin cerdas modelnya, makin lumpuh akurasi mata telanjang peserta kuis (merosot ke batas 52% dari angka kebetulan murni 50%). Fakta seram: Manusia *menghabiskan waktu membaca jauh lebih lama* pada artikel buatan GPT-3 175B, namun tetap gagal meramal apakah itu robot atau bukan. (Lihat Lampiran *Gambar 3.14/3.15* untuk teks asli artikel yang nyaris sempurna struktur diksinya).

**3.9.5 Mempelajari & Menggunakan Kata Fiktif Baru (*Learning and Using Novel Words*)**
Di ranah tes linguistik kualitatif, GPT-3 diberikan satu definisi absurd:
```text
A "whatpu" is a small, furry animal native to Tanzania. An example of a sentence that uses the word whatpu is:
```
GPT-3 secara brilian melanjutkannya dengan merangkai ejaan plural fiktif yang *grammatically sound* di ekor frasa: `"We were traveling in Africa and we saw these very cute whatpus."`
Model ini bahkan mampu merekonstruksi transfigurasi tata bahasa tenses ("screeg" -> menjadi past tense verb "screeghed").

**3.9.6 Koreksi Tata Bahasa Inggris (*Correcting English Grammar*)**
Prompt yang dijejalkan:
```text
Poor English input: I eated the purple berries.
Good English output: I ate the purple berries.
```
Setelah diberi 1-2 pancingan konteks semacam ini (*Few-Shot*), model terbukti instan mampu membuang dialek kotor (*bad grammar*) menjadi formalitas akademis ("The patient was died" -> "The patient died") persis berjejer layaknya *Fine-Tuning* purnawaktu, padahal memori model tidak pernah dimodifikasi sekalipun.

---

## 4. Mengukur dan Mencegah Menghafal Data Patokan (*Measuring and Preventing Memorization Of Benchmarks*)

Sebuah kiamat metodologis (*Methodological Concern*) paling fatal dalam pra-pelatihan model gigantik seukuran Internet adalah: **Apakah data soal ujian diam-diam tersembunyi/bocor di korpus pelatihannya?** Jika soal ujian sudah masuk di dalam ingatan latih (*test contamination*), maka pencapaian model bukan cerminan kepandaian, tetapi semata menyontek hapalan.

Sayangnya, meski telah mendesain algoritma penyaring khusus, OpenAI mendeteksi *bug* kode di menit-menit berjalannya iterasi. Beberapa kebocoran lolos. Karena biaya pelatihan server membengkak miliaran rupiah (tidak bisa sekadar tombol *Restart* ulang), mereka memutuskan menganalisis sejauh mana kotoran data merusak kemurnian klaim riset ini.

#### Analisis Gambar 4.1: Kurva Pelatihan GPT-3
*   **Grafik:** Loss vs Tokens Elapsed. Terdapat garis solid (Validation Loss) dan garis putus (Train Loss).
*   **Kesimpulan:** Walau model memiliki kapasitas absolut raksasa (175B), garis latih dan validasinya menempel tanpa membeku ke celah berongga tajam. Maknanya: Model sejati menderita under-fit (sulit menelan kapasitas seluruh dunia). Potensi penghapalan membabi-buta (*overfitting*) cukup reda.

Langkah Mitigasi: Tim OpenAI mengekstrak ulang daftar pertanyaan seluruh ujian. Jika ditemukan rangkaian frasa yang bentrok (13-gram benturan) pada set internet, contoh itu akan dibabat dan dicap kotor (Dirty). Tim menciptakan klon pengujian kedua yakni Set Soal Tersuci (Clean Version). 

#### Analisis Gambar 4.2: Evaluasi Kontaminasi Patokan
*   **Sumbu X:** Persentase kemurnian dataset uji (Data Clean/Data Asli).
*   **Sumbu Y:** Persentase perubahan performa. Jika titik merosot di atas angka negatif (-10% atau -20%), tandanya SOTA model diraih hasil mencontek. Jika diam di garis batas tengah horizon 0%, artinya model mengerjakan ujian tanpa bantuan sontekan alam bawah sadar.
*   **Kesimpulan Analisis:** Mayoritas titik memadat berkelompok statis rapi menyusuri poros 0%. Klaim prestasi model tetap sah! Hanya sebagian titik krisis minor yang mengalami turbulensi (misal: QuAC, SQuAD, DROP). Saat dibedah secara mikroskopis oleh manusia, kebetulan kontaminasinya pun hanya berkutat bocor pada bagian korpus narasi dasar artikel, *namun tidak mengandung soal tebakan apalagi rahasia kunci jawaban targetnya*.
*   Namun tim melucuti klaim murni mereka dengan menaruh label tanda bintang eksklusif (*) di tabel skor utama (seperti skor **PIQA** dan **Winograd**), menjunjung asas konservatif kehati-hatian karena mendeteksi degradasi mini saat beralih ke soal terfiltrasi murni.

---

## 5. Batasan-Batasan (*Limitations*)

Meskipun raksasa dan mutakhir, GPT-3 175B sarat akan berbagai kelemahan fundamental yang wajib dieksplorasi di riset mendatang:

1.  **Disonansi Kognitif Paragraf Panjang:** Pada kreasi tulisan ekstraksi panjang tanpa batas, GPT-3 tersandung dalam inkonsistensi berbalik-arah (*contradict themselves*). Ia mudah lupa benang merah ide yang ia tulis puluhan kata lalu, menghasilkan keping naratif yang aneh secara logis. Ia tersiksa meramal kausalitas semesta (Contoh: "*Jika keju ditaruh di dalam kulkas, apakah ia akan meleleh?*").
2.  **Cacat Arsitektur *Autoregressive*:** Peneliti menyadari kekalahan GPT-3 dalam evaluasi ANLI, WiC, dan QuAC adalah produk turunan penderitaan "arah laju bahasa". Tidak seperti jagoan BERT yang secara saksama merenungkan dua arah teks kiri-ke-kanan-kiri (Bidirectional/Denoising), arsitektur GPT-3 ditakdirkan buta masa depan (*causal left-to-right*). Mencangkok arsitektur multi-arah dengan dimensi Miliar merupakan utopia batas riset selanjutnya.
3.  **Tak Terjangkar pada Dunia Fisik (*Lack of Grounding*):** Mesin ini membeo bahasa hanya pada spektrum dua dimensi tekstual. Ia tak melihat piksel dimensi gambar (komputer visi) atau menyentuh robotika gravitasi nyata. Ia kurang intusi spasial. 
4.  **Inefisiensi Pra-pelatihan:** Model harus dicekoki triliunan abjad dalam umur inkubatornya agar sukses menelan kognisi setaraf kecerdasan balita yang hanya hidup dalam rentang puluhan hari. 
5.  **Misteri Kotak Hitam Inferensi (*Ambiguity in Few-Shot Learning*):** Peneliti menyuarakan keraguan intelektual rahasia: *Apakah model sungguh-sungguh menciptakan keahlian de novo dari ketiadaan di tahap inner-loop saat diberi format acak? Ataukah model sekadar memanggil bilik ingatan terselubung di matriks pelatihannya?* Garis batas memori atau adopsi sesungguhnya sungguh buram di model Miliar parameter.

---

## 6. Dampak Sosial yang Lebih Luas (*Broader Impacts*)

Inovasi komputasional berdarah-darah layaknya GPT-3 tak pernah sepi dari ancaman guncangan sosial di kolong medik. Diskusi sosiologis meliputi tiga tonggak rawan:

### 6.1 Penyalahgunaan Model (*Misuse*)
Kemahiran memproduksi ilusi sintesis teks di ambang dewa ini menghancurkan ambang keamanan siber. Pelaku ancaman tingkat menengah maupun tinggi (*Advanced Persistent Threats/APT*) mendapat akses injeksi generator rudal untuk: menyebarkan hoaks kampanye masif (*misinformation*), surel *phishing* jebakan bertarget presisi otomatisasi mesin, dan kecurangan manipulasi esai akademik. Kemampuan sistem bot menghasilkan ratusan teks provokatif dengan koherensi human-grade menyabotase keseimbangan organik filter internet. 

### 6.2 Representasi, Keadilan, dan Bias (Fairness & Bias)
Penyedotan membabi-buta korpus dunia maya memastikan masuknya limbah bias, prasangka sosiologis, gender, dan teologi yang tertanam permanen.

#### Analisis Model Kecondongan (Tabel 6.1): Label Gender & Jabatan Profesi
Peneliti menginjeksi matriks kalimat "The {occupation} was a". Hasil investigasi memilukan terkuak. Mesin GPT-3 condong keras merangkai jabatan Profesi tinggi seperti bankir (Banker), dewan legis (Legislator), profesor (Professor) pada konklusi frasa Pronoun **Pria (Laki-laki)**. Sebaliknya pada jabatan bidan, suster, atau ART, dominasi eksak dilontarkan pada **Wanita**.
*   **Rumus Log-Bias Sosiologi:**
    $$ \text{Bias\_Level} = \frac{1}{n_{jobs}} \sum_{jobs} \log \left( \frac{P(\text{female}|\text{Context})}{P(\text{male}|\text{Context})} \right) $$
    *(Penjelasan matematis: Semakin ekstrim selisih probabilitas pecahan opsi wanita vs laki-laki di sumbu algoritma, nilai minus eksponensial di ujung algoritma menyiratkan cacat rasis logaritma yang keji ke ranah gender)*.
*   Pada pencarian diksi pelabel sifat di *Tabel 6.1*, kata adjektif bagi maskulin disuapi istilah kuat (Bosan, Hebat, Stabil, Proteksi), sedangkan entitas feminim dipasung dalam parameter estetika dangkal merendahkan rupa tubuh ("Optimistic, Bubbly, Naughty, Gorgeous, Beautiful").

#### Analisis Ras & Agama (Tabel 6.2 & Gambar 6.1)
*   **Analisis Gambar 6.1 (Racial Sentiment):** Grafik menggambarkan sumbu X ukuran model vs Sumbu Y Sentimen Skor kata. Grafik menjatuhkan kata benda rasial Afrika Amerika (Black) di dasar kawah sentimen terdalam paling negatif secara absolut berulang kali dibanding etnis Kaukasoid maupun Ras Asia yang kokoh positif menjulang melintasi spektrum skala mesin GPT.
*   **Agama (Tabel 6.2):** GPT-3 dikelabui dengan teks "{Religion practitioners} are". Korelasi korpus mengindikasi bahwa porsi narasi Islam di pangkalan dunia maya sangat terpapar pusaran turbulensi berita. Kata pergaulan erat (Top 40) yang dimuntahkan memuat frasa menyedihkan layaknya 'violent' (kekerasan), dan 'terrorism' (terorisme).

### 6.3 Penggunaan Energi (*Energy Usage*)
Pembuatan otak sintetis model bahasa membutuhkan pasokan reaktor setara 3.600 petaflop/s-days. Ini menyedot sumber daya raksasa setara jejak emisi perusak iklim saat siklus Latih. Namun paradoks empiris tercipta di panggung efisiensi Lanjut: Pada fase inferensi (menjalankan tugas pasca-training), memuntahkan cetakan 100 lembar kertas teks GPT-3 hanya memakan ongkos daya murni senilai segelintir 0,4 kWh sen. Pengamortisasian ini melegakan jika dikompromikan dengan model arsitektur sulingan (Distilasi).

---

## 7. Pekerjaan Terkait (*Related Work*)

Pendakian pilar SOTA tidak dibangun dari ruang vakum:
*   Meningkatkan Parameter/Komputasi murni sejalan dengan jejak rekam purba LSTM bertriliun parameter (Jozefowicz dkk.), serta pergerakan eksponensial Megatron, T5, dan T-NLG skala 17 Miliar (Turing).
*   Alternatif efisiensi (campuran bobot para pakar / Mixture-of-Experts) untuk memangkas *flop-per-token* juga ditelurkan tanpa membesarkan komputasi aktif.
*   Pelopor Metodologi Lintas Siklus Inner-Loop & Pemancing Teks natural juga dieksplorasi di *Multi-task fine tuning (T5)* serta arsitektur perintis Meta-Learning MQAN (McCann et al.). Pendekatan GPT-3 mengekor radikal dengan mendemokrasikan sistem tanpa injeksi gerbang *weight updates* di medan tes, sebuah perbatasan suci yang secara naif menyatukan dua semesta *Scaling Laws* dengan *Unsupervised Transfer Learning*.

---

## 8. Kesimpulan (*Conclusion*)

Di penutup mimbar akademik ini, para periset mempersembahkan eksistensi arsitektur Transformer teramat raksasa berisi 175 miliar sirkuit bobot (parameter), **GPT-3**. Pengorbanan pemompaan rasio komputasi mesin murni mengilhami kemampuan instan *zero-shot*, *one-shot*, dan *few-shot learning* hingga meraih keunggulan *State-of-The-Art* di lintas evaluasi NLP yang keras, setara, dan terkadang membunuh taring arsitektur lama berbasis *fine-tuned*. 

Bukti empiris melacak garis tegar ramalan skalar (*scaling laws*), serta mencemaskan spekulasi dampak kerentanan bias dunia riil di saat robot ini menjajaki level tulisan setara perwajahan kemanusiaan aslinya. Walaupun model ini membawa penyakit arsitektur minus dimensi ganda (bidirectional), eksperimen maha besar ini menandai lompatan esensial lahirnya generasi kejeniusan bahasa serba-guna masa depan.

---
*(Catatan Dosen: Sampai di sini batas akhir pembahasan materi utama makalah "Language Models are Few-Shot Learners" GPT-3 ini. Pastikan Anda menguasai perbedaan konseptual antara Zero-Shot, One-Shot, dan Few-Shot tanpa fine-tuning, serta implikasi scaling laws terhadap kemampuan in-context learning. Semua pemahaman dasar NLP modern bertumpu pada pondasi yang diletakkan di penelitian ini.)*