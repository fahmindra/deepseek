---
slug: ch-11-24
title: "engram"
layout: chapter
---

## Memori Bersyarat melalui Pencarian yang Dapat Diskalakan: Sebuah Sumbu Ketersebaran Baru untuk Model Bahasa Besar

Meskipun *Mixture-of-Experts* (MoE) mampu menskalakan kapasitas model melalui komputasi bersyarat (*conditional computation*), arsitektur Transformer secara inheren kekurangan primitif bawaan (*native primitive*) untuk pencarian pengetahuan (*knowledge lookup*). Kekurangan ini memaksa Transformer untuk mensimulasikan pencarian informasi secara tidak efisien melalui proses komputasi yang berat. 

Untuk mengatasi batasan fundamental ini, kami memperkenalkan **Memori Bersyarat (*Conditional Memory*)** sebagai sumbu ketersebaran (*sparsity axis*) yang melengkapi MoE. Konsep ini diinstansiasi melalui modul **Engram**, sebuah arsitektur yang memodernisasi penyematan $N$-gram (*$N$-gram embedding*) klasik untuk memungkinkan pencarian informasi dalam waktu konstan $O(1)$. 

Dengan memformulasikan masalah **Alokasi Ketersebaran (*Sparsity Allocation*)**, kami menemukan sebuah **Hukum Penskalaan Berbentuk U (*U-shaped scaling law*)** yang mengoptimalkan pertukaran (*trade-off*) antara komputasi saraf dinamis (MoE) dan memori statis (Engram). Berpedoman pada hukum ini, kami menskalakan modul Engram hingga 27 Miliar (27B) parameter. Model ini berhasil mencapai kinerja superior dibandingkan *baseline* MoE yang memiliki jumlah parameter setara (*iso-parameter*) dan beban komputasi setara (*iso-FLOPs*).

Yang paling menonjol, sementara modul memori pada awalnya diharapkan hanya membantu tugas penarikan pengetahuan (*knowledge retrieval*) (misalnya, peningkatan skor MMLU +3.4; CMMLU +4.0), kami secara mengejutkan mengamati perolehan kinerja yang bahkan **lebih besar dalam kemampuan penalaran umum** (misalnya, BBH +5.0; ARC-Challenge +3.7) dan domain kode/matematika (HumanEval +3.0; MATH +2.4). 

Analisis mekanistik (*mechanistic analyses*) mengungkapkan bahwa Engram secara efektif membebaskan lapisan-lapisan awal (*early layers*) dari arsitektur *backbone* dari keharusan melakukan rekonstruksi memori statis. Hal ini secara efektif "memperdalam" (*deepening*) jaringan untuk berfokus pada penalaran kompleks. Lebih jauh lagi, dengan mendelegasikan ketergantungan lokal (*local dependencies*) kepada operasi pencarian (*lookups*), Engram membebaskan kapasitas mekanisme atensi (*attention capacity*) untuk berfokus pada konteks global, yang secara substansial meningkatkan penarikan konteks panjang (*long-context retrieval*) (sebagai contoh, kinerja pada Multi-Query NIAH melonjak dari 84.2 $\rightarrow$ 97.0). 

Terakhir, Engram membangun efisiensi yang sadar-infrastruktur (*infrastructure-aware efficiency*): metode pengalamatan deterministiknya (*deterministic addressing*) memungkinkan *prefetching* memori secara *runtime* (langsung saat program berjalan) dari memori *host*, sehingga nyaris tidak menimbulkan beban operasional tambahan (*negligible overhead*). Kami memproyeksikan memori bersyarat sebagai primitif pemodelan yang sangat esensial bagi model-model renggang (*sparse models*) generasi mendatang.

---

### Analisis Akademik: Gambar 1 - Kompresi pada Fox Benchmark dan OmniDocBench

Sebelum kita masuk ke dalam pendahuluan teoritis, mari kita telaah kinerja empiris dari metode ini melalui **Gambar 1**. Meskipun judul asli dari jurnal Anda mengacu pada "DeepSeek-OCR", naskah teks yang Anda berikan adalah tentang "Engram / Conditional Memory". Oleh karena gambar yang dilampirkan berfokus pada eksperimen *compression* (kompresi) dan OCR, saya akan menganalisis grafik tersebut sesuai dengan konteks yang relevan.

*(Catatan Dosen: Terdapat sedikit ketidaksesuaian antara naskah teks utama (membahas Engram/LLM Memory) dengan Gambar 1 yang disediakan (membahas DeepSeek-OCR). Namun, sesuai instruksi untuk membedah gambar secara mendetail, berikut adalah analisis akademis untuk Gambar 1).*

**Gambar 1 (a) - Kompresi pada tolok ukur Fox (*Compression on Fox benchmark*)**

*   **Deskripsi Visual:** Ini adalah sebuah diagram batang bertumpuk ganda (*grouped bar chart*) yang dilapisi dengan grafik garis putus-putus (*dotted line graph*).
*   **Sumbu X (Horizontal):** Menunjukkan **Teks Token per Halaman (Ground-Truth)**, yang dikelompokkan ke dalam beberapa rentang mulai dari 600-700 token, 700-800, hingga 1200-1300 token. Ini merepresentasikan seberapa padat teks di dalam sebuah dokumen.
*   **Sumbu Y Primer (Kiri):** Menunjukkan **Presisi (%)** dari 0% hingga 100%. Batang-batang histogram merujuk pada sumbu ini.
*   **Sumbu Y Sekunder (Kanan):** Menunjukkan **Kompresi (x)**, yaitu seberapa besar rasio pemadatan token (mulai dari 0x hingga 20x). Garis putus-putus merujuk pada sumbu ini.
*   **Elemen Data:** Terdapat dua kelompok eksperimen berdasarkan jumlah "token visi" (*vision toks*) yang dibolehkan: **64 token visi** (warna ungu muda) dan **100 token visi** (warna biru muda).
*   **Tren dan Kesimpulan Akademis:**
    1.  **Presisi Menurun Seiring Kepadatan Teks:** Saat dokumen semakin padat (bergerak ke kanan pada Sumbu X), presisi pembacaan OCR (batang diagram) secara perlahan menurun. Namun, untuk dokumen dengan 1000-1100 teks token, model yang menggunakan 100 token visi (batang biru) masih mampu mempertahankan presisi luar biasa di atas **91.5%**.
    2.  **Rasio Kompresi Melonjak:** Garis putus-putus menunjukkan bahwa saat dokumen makin padat, rasio kompresi meningkat secara linear. Pada kondisi dokumen sangat padat (1200-1300 kata) yang dipaksa diproses hanya menggunakan 64 token visi (garis ungu), rasio kompresi menyentuh angka ekstrem mendekati **20x** lipat. Meskipun pada titik ini presisi turun menjadi sekitar 59%, kemampuan mengompresi 1200 kata ke dalam 64 token adalah sebuah lompatan efisiensi yang signifikan.

**Gambar 1 (b) - Performa pada Omnidocbench**

*   **Deskripsi Visual:** Ini adalah sebuah diagram pencar (*scatter plot*) yang membandingkan berbagai model AI.
*   **Sumbu X (Horizontal):** Menunjukkan **Rata-rata Token Visi per Gambar** (*Average Vision Tokens per Image*) dalam skala logaritmik, bergerak dari 1000 hingga 7000 token.
*   **Sumbu Y (Vertikal):** Menunjukkan **Performa Keseluruhan / Jarak Edit** (*Overall Performance (Edit Distance)*). Dalam metrik jarak edit, **semakin kecil nilainya (semakin ke bawah), semakin baik** model tersebut (karena kesalahannya sedikit).
*   **Titik Data dan Anotasi:** Titik-titik merepresentasikan model yang berbeda, dikelompokkan dengan warna (misalnya, Encoder Series, DeepEncoder Series). Ada area hijau bertuliskan *"High Accuracy ED < 0.25 (1 better)"*, yang merupakan zona elit.
*   **Tren dan Kesimpulan Akademis:** Model **DeepSeek-OCR** (simbol merah) berhasil memasuki area hijau (akurasi tinggi) sembari berada di ujung paling kiri sumbu X. Artinya, DeepSeek-OCR mencapai tingkat akurasi *state-of-the-art* (SOTA) dengan **mengkonsumsi jumlah token visi yang paling minimal** dibandingkan semua model ujung-ke-ujung (*end-to-end*) lainnya (seperti Qwen2.5-VL, InternVL2, dan MinerU). Ini membuktikan dominasi efisiensi arsitektural dari model tersebut.

---

## 1. Pendahuluan

Ketersebaran (*Sparsity*) merupakan prinsip desain yang berulang kali dijumpai dalam berbagai sistem cerdas, merentang dari sirkuit saraf biologis (Lennie, 2003; Olshausen and Field, 1997) hingga Model Bahasa Besar (LLMs) modern. Saat ini, prinsip ini utamanya direalisasikan melalui arsitektur **Mixture-of-Experts (MoE)** (Dai et al., 2024; Shazeer et al., 2017), yang menskalakan kapasitas model melalui metode **komputasi bersyarat** (*conditional computation*). 

Berkat kemampuannya untuk meningkatkan ukuran model (jumlah parameter) secara drastis tanpa menyebabkan peningkatan beban komputasi yang proporsional, MoE telah menjadi standar *de facto* bagi model-model garis depan (*frontier models*) (Comanici et al., 2025; Guo et al., 2025; Team et al., 2025).

Terlepas dari kesuksesan paradigma komputasi bersyarat ini, heterogenitas (keberagaman sifat) intrinsik dari sinyal linguistik (bahasa) manusia mengindikasikan bahwa masih terdapat ruang yang sangat besar untuk **optimisasi struktural**. 

Secara spesifik, pemodelan bahasa (*language modeling*) sejatinya mencakup dua sub-tugas yang secara kualitatif sangat berbeda: 
1.  **Penalaran kompoisisional** (*compositional reasoning*).
2.  **Penarikan pengetahuan** (*knowledge retrieval*). 

Sementara penalaran kompoisisional menuntut komputasi dinamis yang mendalam, ada bagian substansial dari teks bahasa—seperti nama entitas spesifik (*named entities*) dan pola-pola kalimat baku (*formulaic patterns*)—yang bersifat lokal, statis, dan sangat berulang (*highly stereotyped*) (Constant et al., 2017; Erman, 2000). 

Efektivitas dari model klasik **$N$-gram** (Brants et al., 2007; Liu et al., 2024b; Nguyen, 2024) dalam menangkap ketergantungan lokal semacam ini membuktikan bahwa keteraturan tekstual ini sangat cocok direpresentasikan sebagai operasi pencarian (*lookups*) yang sangat murah secara komputasional. Mengingat arsitektur Transformer standar (Vaswani et al., 2017) kekurangan komponen bawaan untuk pencarian pengetahuan, LLM saat ini secara terpaksa harus **mensimulasikan penarikan informasi (retrieval) melalui komputasi paksa yang berat**.

Sebagai contoh kasus, untuk mendeteksi sebuah entitas yang terdiri dari beberapa token, model Transformer konvensional harus menghabiskan komputasi di beberapa lapisan awal (*early layers*) yang berisi jaringan atensi (*attention*) dan jaringan maju-umpan (*feed-forward networks*) (Ghandeharioun et al., 2024; Jin et al., 2025) (dapat dirujuk pada Tabel 3). Proses ini pada hakikatnya sama dengan merekonstruksi sebuah tabel pencarian statis secara manual saat *runtime*, yang mana sangat mahal dan sia-sia, karena membuang-buang "kedalaman sekuensial" (*sequential depth*) model untuk operasi sepele yang seharusnya bisa dialokasikan untuk penalaran tingkat tinggi.

Untuk menyelaraskan arsitektur model dengan dualitas linguistik ini, kami mengusulkan untuk menambahkan satu sumbu ketersebaran pelengkap: **Memori Bersyarat (*conditional memory*)**. 

*   Sementara **komputasi bersyarat** (MoE) hanya mengaktifkan sejumlah kecil parameter secara selektif untuk memproses logika dinamis (Bengio et al., 2013; Shazeer et al., 2017), 
*   **Memori bersyarat** bergantung pada operasi pencarian (lookup) yang jarang (*sparse*) untuk menarik vektor penyematan statis (*static embeddings*) untuk pengetahuan yang sudah pasti/tetap. 

Sebagai eksplorasi awal dari paradigma ini, kami meninjau ulang penyematan $N$-gram ($N$-gram embeddings) (Bojanowski et al., 2017) sebagai sebuah instansiasi kanonikal: konteks lokal dari kalimat akan bertindak sebagai "kunci" (*key*) untuk memberikan indeks pencarian (lookup) ke sebuah tabel penyematan raksasa dalam waktu yang konstan, yakni $\mathcal{O}(1)$ (Huang et al., 2025a; Pagnoni et al., 2025; Tito Svenstrup et al., 2017; Yu et al., 2025). 

Investigasi kami mengungkap suatu kejutan saintifik: mekanisme penarikan informasi statis ini dapat berfungsi sebagai komplemen (pelengkap) yang sangat ideal bagi arsitektur MoE modern—tetapi **hanya jika** ia dirancang dengan tepat. 

Dalam makalah ini, kami memperkenalkan **Engram**, sebuah modul memori bersyarat yang didasarkan pada struktur klasik $N$-gram tetapi telah dipersenjatai dengan adaptasi modern seperti: kompresi tokenizer, *multi-head hashing*, *contextualized gating* (mekanisme gerbang bersyarat konteks), dan integrasi multi-cabang (diuraikan secara mendetail pada Bagian 2).

Untuk mengukur dan menghitung sinergi antara kedua primitif (MoE dan Engram) ini, kami memformulasikan masalah **Alokasi Ketersebaran (*Sparsity Allocation*)**: *"Diberikan total anggaran parameter yang tetap, bagaimana seharusnya kapasitas tersebut didistribusikan antara pakar MoE dan memori Engram?"*

Serangkaian eksperimen kami menyingkap sebuah **hukum penskalaan berbentuk U (*U-shaped scaling law*)** yang sangat jelas. Hukum ini menegaskan bahwa bahkan mekanisme pencarian (lookup) yang sederhana sekalipun—ketika diperlakukan sebagai primitif pemodelan kelas satu—bertindak sebagai pelengkap esensial bagi komputasi saraf. 

Berpedoman pada hukum alokasi ini, kami menskalakan modul Engram menjadi model dengan **27 Miliar (27B) parameter**. Jika dibandingkan secara ketat dengan *baseline* MoE yang memiliki parameter sama persis (*iso-parameter*) dan beban komputasi sama persis (*iso-FLOPs*), Engram-27B berhasil mencapai efisiensi yang jauh lebih unggul di berbagai lintas domain. 

Secara krusial, keuntungan kinerja ini tidak hanya terbatas pada tugas padat-pengetahuan (*knowledge-intensive tasks*) (misalnya, MMLU: +3.4; CMMLU: +4.0; MMLU-Pro: +1.8), di mana kapasitas memori tambahan jelas akan menguntungkan secara intuitif. Kami justru mengamati peningkatan yang **bahkan lebih masif pada tugas penalaran umum** (misalnya, BBH: +5.0; ARC-Challenge: +3.7; DROP: +3.3) dan **domain kode/matematika** (misalnya, HumanEval: +3.0; MATH: +2.4; GSM8K: +2.2).

Analisis mekanistik menggunakan *LogitLens* (nostalgebraist, 2020) dan *Centered Kernel Alignment / CKA* (Hendrycks et al., 2021a) menyingkap asal muasal keuntungan performa ini: Engram secara langsung membebaskan (meringankan beban) jaringan *backbone* dari keharusan untuk merekonstruksi pengetahuan statis di lapisan-lapisan awal, yang secara otomatis **meningkatkan kedalaman efektif model** yang tersedia untuk melakukan penalaran kompleks. 

Lebih lanjut, dengan mendelegasikan tugas melacak ketergantungan lokal kepada modul pencarian *lookup*, Engram membebaskan kapasitas "perhatian" (*attention capacity*) model untuk berfokus secara penuh pada menangkap konteks global. Hal ini memungkinkan performa yang sangat luar biasa dalam skenario berkonteks panjang (*long-context scenarios*)—mengungguli model *baseline* secara telak pada LongPPL (Fang et al.) dan RULER (Hsieh et al.) (Sebagai contoh, pada pengujian Multi-Query NIAH skornya meroket dari 84.2 $\rightarrow$ 97.0; dan pada Variable Tracking naik dari 77.0 $\rightarrow$ 89.0).

Terakhir, kami menetapkan **efisiensi sadar-infrastruktur** sebagai prinsip desain kelas satu. Berbeda dengan sistem perutean dinamis (*dynamic routing*) pada MoE, Engram menggunakan ID (identitas) yang deterministik untuk memungkinkan teknik *prefetching* pada saat *runtime*, yang mana berhasil menumpangkan (*overlapping*) proses komunikasi data dengan proses komputasi GPU. Hasil empiris mendemonstrasikan bahwa proses memindahkan (*offloading*) tabel raksasa berparameter 100B ke memori RAM *Host* hanya menyebabkan perlambatan *overhead* yang dapat diabaikan ($< 3\%$). Hal ini membuktikan bahwa Engram secara efektif sukses "mengecoh" (membypass) keterbatasan memori VRAM GPU, memfasilitasi ekspansi parameter model secara agresif tanpa harus membeli GPU tambahan.

---

## 2. Arsitektur

### 2.1. Gambaran Umum (*Overview*)

Mari kita perhatikan arsitektur internal sistem ini dengan membedah **Gambar 1**.

### Analisis Akademis: Gambar 1 - Arsitektur Engram

*   **Deskripsi Visual:** Diagram ini mengilustrasikan anatomi dari sebuah blok Transformer yang telah dimodifikasi (di- *augmentasi*) dengan modul Engram.
*   **Aliran Utama (Kiri):** Memperlihatkan tumpukan komponen standar Transformer. Teks masukan (misal: "Only Alexander the Great could tame the horse Bucephalus.") diubah menjadi *Vocab Embedding*, melewati *Transformer Block* awal, lalu masuk ke blok yang terdiri dari `Attention` dan `MoE`.
*   **Sisi Engram (Kanan):** Ini adalah inovasi utamanya. Input dari token (misal: kata "the Great") tidak hanya masuk ke *Transformer*, tetapi juga diekstrak menjadi **2-Gram Embedding** dan **3-Gram Embedding** (menggunakan teks sebelumnya sebagai referensi pencarian / *hash*).
*   **Proses Fusi (*Fusion*):**
    1.  Vektor hasil dari pencarian *Hash* ini digabungkan (`Concat`) membentuk vektor $h$.
    2.  Vektor $h$ ini kemudian dilewatkan ke operasi linier (`Linear`) untuk menghasilkan Nilai (*Value*).
    3.  Secara bersamaan, status tersembunyi (*Input Hidden*) dari Transformer juga diekstraksi menggunakan operasi linier untuk menjadi Kueri (*Query*) dan Kunci (*Key*).
    4.  Dilakukan operasi `Scaled Dot Product` (Perkalian titik berskala) antara *Query* dari Transformer dan *Key* dari Engram untuk menghitung bobot gerbang (*gate*).
    5.  Nilai bobot ini dikalikan dengan *Value* dari Engram (melalui operasi `Conv` / Konvolusi 1D).
    6.  Hasil akhirnya disuntikkan kembali (dijumlahkan secara residual) ke dalam aliran utama *Transformer* tepat sebelum masuk ke blok `Attention`.
*   **Kesimpulan Akademis:** Modul Engram bertindak sebagai "memori eksternal" yang sangat cepat. Alih-alih menyuruh Transformer mengingat frasa statis seperti "Alexander the Great", modul Engram langsung *mencari* makna frasa tersebut di dalam tabel memori raksasa, lalu menyuntikkan maknanya kembali ke aliran data. Ini sangat menghemat beban kerja komputasi *Transformer*. Catatan penting: Modul ini HANYA diaplikasikan pada lapisan-lapisan (*layers*) tertentu saja untuk memisahkan fungsi memori dari fungsi komputasi, menjaga *input/output embedding* standar tetap tidak tersentuh.

Secara formal, diberikan urutan masukan $X = (x_1, \dots, x_T)$ dan status tersembunyi $\mathbf{H}^{(\ell)} \in \mathbb{R}^{T \times d}$ pada lapisan $\ell$, modul Engram akan memproses setiap posisi $t$ dalam dua fase fungsional: **penarikan (retrieval)** dan **penggabungan (fusion)**. 
*   Pertama (dijelaskan pada Bagian 2.2), kami mengekstraksi dan mengompresi sufiks $N$-gram untuk secara deterministik menarik vektor penyematan statis melalui metode hashing. 
*   Selanjutnya (Bagian 2.3), penyematan yang ditarik tersebut dimodulasi secara dinamis oleh status tersembunyi saat ini dan disempurnakan melalui konvolusi ringan. 
*   Terakhir, kami membahas integrasi dengan arsitektur multi-cabang (Bagian 2.4) dan desain sistemnya (Bagian 2.5).

### 2.2. Penarikan Renggang Melalui Hashed N-Grams (*Sparse Retrieval via Hashed N-grams*)

Fase pertama bertugas memetakan konteks lokal (kalimat pendek) ke dalam entri memori statis. Ini melibatkan kompresi *tokenizer* dan penarikan vektor penyematan melalui metode *hashing* yang pasti (deterministik).

**Kompresi Tokenizer (*Tokenizer Compression*).**
Model $N$-gram tradisional umumnya beroperasi langsung pada output *tokenizer*. Namun, *tokenizer* sub-kata standar (seperti BPE) sangat memprioritaskan rekonstruksi tanpa kerugian (*lossless reconstruction*). Seringkali, mereka memberikan ID token yang sama sekali berbeda untuk istilah yang maknanya persis sama (misalnya, `Apple` vs `_apple` yang diawali spasi) (Kudo and Richardson, 2018; Li et al., 2023b). 

Untuk memaksimalkan kepadatan semantik, kami mengimplementasikan sebuah lapisan proyeksi kosakata. Secara matematis, kami melakukan prapengkomputasian sebuah fungsi surjektif $\mathcal{P} : V \rightarrow V'$ yang meruntuhkan (*collapses*) ID token mentah menjadi *identifier* (pengenal) kanonikal berdasarkan kesetaraan tekstual yang dinormalisasi (menggunakan aturan NFKC (Whistler, 2025), mengubah semua huruf menjadi kecil/ *lowercasing*, dll). 

Dalam praktiknya, teknik pembersihan ini berhasil memangkas **23% ukuran kosakata efektif** untuk *tokenizer* 128k (lihat Lampiran C). Secara formal, untuk sebuah token di posisi $t$, kami memetakan ID mentahnya $x_t$ menjadi ID kanonikal $x'_t = \mathcal{P}(x_t)$ untuk membentuk sufiks $N$-gram $g_{t,n} = (x'_{t-n+1}, \dots, x'_t)$.

**Hashing Multi-Kepala (*Multi-Head Hashing*).**
Mencoba memparameterisasi ruang kombinatorial dari SEMUA kemungkinan $N$-gram secara langsung adalah hal yang sangat mustahil secara komputasional (terlalu banyak kombinasi kata). Mengikuti pendekatan dari Tito Svenstrup et al. (2017), kami mengadopsi pendekatan berbasis *hashing*. 

Untuk meminimalkan tabrakan hash (*hash collisions*), kami mempekerjakan $K$ buah kepala hash (*hash heads*) yang berbeda untuk setiap orde $N$-gram $n$. Setiap kepala $k$ akan memetakan konteks yang dikompresi tersebut ke sebuah angka indeks di dalam sebuah tabel penyematan (*embedding table*) $\mathbf{E}_{n,k}$ (yang memiliki ukuran bilangan prima $M_{n,k}$) melalui fungsi deterministik $\varphi_{n,k}$:

$$ z_{t,n,k} \triangleq \varphi_{n,k}(g_{t,n}), \quad \mathbf{e}_{t,n,k} = \mathbf{E}_{n,k}[z_{t,n,k}]. \tag{1} $$

**Penjelasan Rumus (1):**
*   $g_{t,n}$: Adalah sufiks $N$-gram yang kita miliki saat ini (potongan frasa).
*   $\varphi_{n,k}$: Adalah fungsi *hash* (Multiplicative-XOR) yang mengubah teks menjadi angka indeks (seperti alamat laci).
*   $z_{t,n,k}$: Adalah angka indeks laci hasil *hash*.
*   $\mathbf{E}_{n,k}$: Adalah lemari raksasa (Tabel Penyematan) yang berisi memori.
*   $\mathbf{e}_{t,n,k}$: Adalah isi laci (Vektor Memori) yang ditarik keluar.

Dalam penerapannya, $\varphi_{n,k}$ diimplementasikan sebagai algoritma *multiplicative-XOR hash* yang sangat ringan. Kami kemudian menyusun vektor memori final $\mathbf{e}_t \in \mathbb{R}^{d_{\text{mem}}}$ dengan cara menggabungkan (mengonkatensi / *concatenating*) seluruh penyematan yang berhasil ditarik:

$$ \mathbf{e}_t \triangleq \mathop{\Big\|}_{n=2}^N \mathop{\Big\|}_{k=1}^K \mathbf{e}_{t,n,k}. \tag{2} $$

**Penjelasan Rumus (2):**
*   $\mathbf{e}_t$: Adalah vektor memori komposit final untuk token ke-$t$.
*   $\Big\|$: Adalah operator konkatenasi (menyatukan beberapa vektor menjadi satu vektor panjang).
*   Rumus ini berarti kita menarik memori dari pola 2-gram (2 kata), 3-gram, hingga $N$-gram, dari semua kepala *hash* $K$, lalu menjahitnya menjadi satu vektor informasi yang utuh.

### 2.3. Context-aware Gating (Mekanisme Gerbang Sadar-Konteks)

Vektor penyematan $\mathbf{e}_t$ yang ditarik di atas bertindak sebagai pengetahuan dasar yang independen dari konteks (kaku). Karena sifatnya yang statis, mereka pada dasarnya kurang memiliki daya adaptasi kontekstual dan berpotensi memuat *noise* (gangguan/kesalahan) akibat terjadinya tabrakan hash (*hash collisions*) atau polisemi (satu kata beda arti, misal "bisa" ular vs "bisa" melakukan) (Haber and Poesio, 2024).

Untuk meningkatkan daya ekspresi dan memecahkan ambiguitas ini, kami menggunakan mekanisme gerbang (*gating mechanism*) sadar-konteks yang terinspirasi dari metode *Attention* (Bahdanau et al., 2015; Vaswani et al., 2017). Secara spesifik, kami memanfaatkan status tersembunyi (*hidden state*) saat ini $\mathbf{h}_t$—yang telah mengumpulkan konteks global dari lapisan-lapisan atensi sebelumnya—sebagai **Kueri dinamis (*dynamic Query*)**. Sementara itu, memori yang ditarik $\mathbf{e}_t$ akan difungsikan sebagai sumber untuk proyeksi **Kunci (*Key*)** dan **Nilai (*Value*)**:

$$ \mathbf{k}_t = \mathbf{W}_K \mathbf{e}_t, \quad \mathbf{v}_t = \mathbf{W}_V \mathbf{e}_t \tag{3} $$

**Penjelasan Rumus (3):**
*   $\mathbf{W}_K$ dan $\mathbf{W}_V$: Adalah matriks proyeksi yang dapat dipelajari (bobot yang di-*training*).
*   $\mathbf{e}_t$: Memori Engram statis.
*   Rumus ini mengubah memori kasar $\mathbf{e}_t$ menjadi Kunci $\mathbf{k}_t$ (untuk dicocokkan dengan Kueri) dan Nilai $\mathbf{v}_t$ (isi data yang sesungguhnya).

Untuk memastikan gradien matematika tetap stabil selama proses *training* (Dehghani et al., 2023), kami mengaplikasikan teknik *RMSNorm* (Zhang and Sennrich, 2019) pada Kueri dan Kunci **sebelum** menghitung nilai gerbang skalar (angka desimal penentu) $\alpha_t \in (0, 1)$:

$$ \alpha_t = \sigma \left( \frac{\text{RMSNorm}(\mathbf{h}_t)^\top \text{RMSNorm}(\mathbf{k}_t)}{\sqrt{d}} \right). \tag{4} $$

**Penjelasan Rumus (4):**
*   $\mathbf{h}_t$: Adalah Kueri (*Query*) dari konteks kalimat saat ini.
*   $\mathbf{k}_t$: Adalah Kunci (*Key*) dari memori Engram.
*   Perkalian titik transpose ($\top$) mengukur seberapa "mirip" atau "cocok" antara konteks kalimat dengan memori yang ditarik. Semakin mirip, semakin besar nilainya.
*   $\sqrt{d}$: Faktor penskalaan (*scaling factor*) untuk menjaga agar angka tidak meledak.
*   $\sigma$: Fungsi Sigmoid. Mengubah hasil perhitungan menjadi persentase probabilitas antara 0 (tolak memori) hingga 1 (terima memori sepenuhnya).
*   $\alpha_t$: Inilah nilai "Gerbang". Ia memutuskan berapa persen memori ini boleh masuk ke otak utama.

Keluaran yang telah melewati gerbang ini kemudian didefinisikan sebagai $\tilde{\mathbf{v}}_t = \alpha_t \cdot \mathbf{v}_t$. Desain matematis ini secara brilian **memaksakan keselarasan semantik**: Jika ternyata memori yang ditarik $\mathbf{e}_t$ tersebut maknanya **bertentangan** dengan konteks kalimat saat ini $\mathbf{h}_t$ (misal karena tabrakan hash), maka nilai gerbang $\alpha_t$ akan mengecil mendekati angka nol, yang secara efektif akan **meredam (membuang) *noise* tersebut**.

Sebagai sentuhan akhir, untuk memperluas area penerimaan data (receptive field) dan mendongkrak sifat non-linearitas model, kami memperkenalkan sebuah operasi **depthwise causal convolution** pendek (Gu et al., 2022; Peng et al., 2023). Biarkan $\tilde{\mathbf{V}} \in \mathbb{R}^{T \times d}$ melambangkan sekuens (rentetan) dari nilai-nilai keluaran gerbang tadi. Dengan menggunakan ukuran kernel (*kernel size*) $w$ (diatur di angka 4), tingkat dilasi (*dilation*) $\delta$ (diatur mengikuti nilai maksimal orde $N$-gram), dan fungsi aktivasi *SiLU* (Elfwing et al., 2018), hasil keluaran pamungkas $\mathbf{Y}$ dikalkulasi sebagai:

$$ \mathbf{Y} = \text{SiLU}(\text{Conv1D}(\text{RMSNorm}(\tilde{\mathbf{V}}))) + \tilde{\mathbf{V}}, \tag{5} $$

**Penjelasan Rumus (5):**
*   Ini adalah formula pelicinan sinyal. Keluaran gerbang $\tilde{\mathbf{V}}$ dinormalisasi, diproses melalui konvolusi 1D agar saling "berbaur" dengan kata-kata di sekitarnya, lalu dilewatkan ke aktivasi SiLU agar tidak linear (bisa menangani logika rumit).
*   Hasilnya ditambah kembali dengan input aslinya $\tilde{\mathbf{V}}$ (ini adalah praktik umum *residual connection* agar gradien tidak mati saat *training*).

Modul Engram ini akhirnya diintegrasikan masuk ke dalam tulang punggung model utama (*backbone*) melalui sebuah koneksi residual: $\mathbf{H}^{(\ell)} \leftarrow \mathbf{H}^{(\ell)} + \mathbf{Y}$, lalu setelahnya barulah data masuk ke proses *Attention* dan *MoE* standar. 
**Poin yang Sangat Krusial:** Modul Engram **TIDAK** disuntikkan di setiap lapisan layer. Penempatan persis lokasinya diatur berdasarkan perhitungan matematis tingkat kendala latensi sistem (*system-level latency constraints*) yang akan kita bedah di Bagian 2.5.

---

### 2.4. Integrasi dengan Arsitektur Multi-Cabang (*Multi-branch Architecture*)

Dalam penelitian ini, daripada menggunakan koneksi aliran tunggal (*single-stream*) yang konvensional (He et al., 2016), kami mengadopsi arsitektur multi-cabang tingkat lanjut sebagai tulang punggung (backbone) *default* kami. Arsitektur ini dipilih karena kapabilitas pemodelannya yang jauh lebih superior (Larsson et al., 2017; Szegedy et al., 2015; Xie et al., 2025; Zhu et al., 2025). Karakteristik utama yang mendefinisikan arsitektur ini adalah pelebaran/ekspansi dari aliran residual ke dalam $M$ cabang yang paralel, di mana aliran informasi diatur secara ritmis oleh bobot koneksi yang dapat dipelajari (*learnable connection weights*).

Meskipun modul Engram pada dasarnya tidak terikat pada topologi tertentu (*topology-agnostic*), mengadaptasinya ke dalam kerangka kerja multi-cabang ini mewajibkan adanya optimalisasi struktural yang cerdas demi menyeimbangkan efisiensi mesin dan daya ekspresi model.

Secara spesifik, kami mengimplementasikan strategi **berbagi-parameter** (*parameter-sharing strategy*): Satu buah tabel penyematan renggang (*sparse embedding table*) tunggal dan satu buah matriks proyeksi Nilai (*Value matrix*) $\mathbf{W}_V$ akan **dipakai bersama** (*shared*) melintasi seluruh $M$ cabang. Namun, di sisi lain, disiapkan $M$ buah matriks proyeksi Kunci (*Key matrix*) $\{\mathbf{W}_K^{(m)}\}_{m=1}^M$ yang **berbeda** (independen) untuk menghasilkan perilaku gerbang (*gating behaviors*) yang spesifik untuk setiap cabang. 

Untuk cabang ke-$m$ yang memiliki status tersembunyi $\mathbf{h}_t^{(m)}$, sinyal gerbang spesifik-cabang tersebut dihitung menggunakan formula:

$$ \alpha_t^{(m)} = \sigma \left( \frac{(\text{RMSNorm}(\mathbf{h}_t^{(m)}))^\top \text{RMSNorm}(\mathbf{W}_K^{(m)} \mathbf{e}_t)}{\sqrt{d}} \right). \tag{6} $$

**Penjelasan Rumus (6):**
*   Mirip dengan Persamaan (4), namun sekarang indeks $(m)$ ditambahkan. Ini berarti setiap cabang $m$ memiliki otak (*query* $\mathbf{h}_t^{(m)}$) dan pintu kunci ($\mathbf{W}_K^{(m)}$) sendiri untuk mengevaluasi apakah memori Engram $\mathbf{e}_t$ tersebut berguna untuk cabang spesifik mereka atau tidak.

Memori yang berhasil ditarik kemudian dimodulasi oleh gerbang independen ini, dan diaplikasikan kepada vektor Nilai (*Value*) yang dipakai bersama: $\mathbf{u}_t^{(m)} = \alpha_t^{(m)} \cdot (\mathbf{W}_V \mathbf{e}_t)$. 
Desain matematis yang memukau ini memungkinkan operasi proyeksi linear (yang melibatkan satu $\mathbf{W}_V$ dan $M$ buah $\mathbf{W}_K^{(m)}$ yang berbeda) untuk dilebur (*fused*) ke dalam satu buah operasi perkalian matriks *dense FP8* tunggal. Hal ini secara maksimal menguras seluruh utilisasi komputasi yang ada pada GPU modern tanpa ampun. Kecuali dinyatakan lain, seluruh eksperimen dalam makalah ini menggunakan integrasi dengan arsitektur *Manifold-Constrained Hyper-Connections* ($M = 4$) (Xie et al., 2025).

---

### Analisis Akademis: Gambar 2 - Implementasi Sistem dari Engram

Mari kita bedah arsitektur operasional sistem (rekayasa *software & hardware*) melalui **Gambar 2**.

*   **Deskripsi Visual:** Gambar ini dibagi menjadi dua skenario operasional: **(a) Saat *Training* (Pelatihan)** dan **(b) Saat *Inference* (Penggunaan/Produksi)**. Keduanya menampilkan rantai urutan (*pipeline*) blok pemrosesan data (Transformer Block).
*   **(a) Engram at training (Engram saat Pelatihan):**
    *   Terdapat blok *Vocab Embedding*, lalu data naik ke *Transformer Block*.
    *   Terdapat blok ungu bertuliskan **Engram**. Perhatikan garis putus-putus berlabel **All2All** yang melintasi dua mesin terpisah (kotak kiri dan kanan).
    *   **Kesimpulan:** Saat masa pelatihan (training), tabel memori Engram ukurannya sangat gila-gilaan (bisa ratusan miliar parameter). RAM satu GPU tidak akan sanggup menampungnya. Maka, tabel raksasa ini "dipecah-pecah" (*sharded*) dan disebar ke banyak GPU. Ketika satu GPU membutuhkan memori dari token tertentu, ia menggunakan primitif jaringan komunikasi antar-GPU tingkat tinggi bernama *All-to-All communication* untuk saling bertukar (menarik) baris memori yang aktif lintas *device*.
*   **(b) Engram at inference (Engram saat Inferensi/Produksi):**
    *   Aliran data jauh lebih sederhana. Terdapat balok silinder raksasa berwarna ungu di sebelah kanan bertuliskan **Offloaded Engram (Memory Hierarchy)**. Perhatikan panah putus-putus mengarah dari `Input IDs` ke silinder ini, lalu dari silinder masuk ke blok `Transformer Block with Engram`.
    *   **Kesimpulan Strategis:** Ini adalah trik rekayasa *hardware* yang sangat brilian. Saat model dipakai pengguna (*inference*), kita tidak butuh banyak GPU. Tabel memori Engram yang raksasa itu **dipindahkan (di-offload) ke RAM Komputer Induk (Host DRAM)**, bukan di VRAM GPU yang mahal! 
    *   *Bagaimana ini tidak bikin lambat (lag)?* Karena pencarian token (hash ID) bersifat *deterministik* (pasti), sistem sudah tahu kata apa saja yang ada di kalimat input sejak detik pertama. Maka, CPU komputer induk secara asinkron (*asynchronously*) dan diam-diam "mengambil" (*prefetch*) data memori tersebut dari RAM dan mentransfernya ke GPU melintasi kabel *PCIe*, TEPAT PADA SAAT GPU tersebut sedang sibuk menghitung lapisan Transformer pertama. Proses komunikasi lambat ini disembunyikan (*overlapped*) di balik waktu komputasi GPU, sehingga waktu tundanya (latensinya) tersembunyi (*zero latency overhead*).

### 2.5. Efisiensi Sistem: Memisahkan Komputasi dan Memori (*Decoupling Compute and Memory*)

Seperti yang divisualisasikan pada Gambar 2 di atas, penskalaan model yang di-*augmentasi* dengan memori sering kali tercekik oleh keterbatasan kapasitas RAM super-cepat GPU (HBM). Namun, mekanisme penarikan deterministik dari Engram secara natural mendukung pemisahan (dekupelisasi) antara penyimpanan parameter (memori) dengan sumber daya komputasional (otak GPU). 

*   Tidak seperti MoE, yang secara mutlak sangat bergantung pada nilai "status tersembunyi" dinamis yang baru akan diketahui saat *runtime* untuk merutekan kueri, 
*   Indeks penarikan memori Engram HANYA bergantung pada rentetan token masukan awal (*input token sequence*). Sifat yang bisa diramalkan (*predictability*) ini memberikan kemewahan untuk melakukan strategi optimisasi luar biasa.

**Kala Inferensi (*Inference*):** Sifat deterministik ini memungkinkan strategi **prefetch-and-overlap** (ambil-duluan-dan-tumpuk). Karena indeks alamat memori (hash) sudah diketahui bulat-bulat *sebelum* proses perhitungan (forward pass) dimulai, sistem dapat secara asinkron (sembunyi-sembunyi) mengambil potongan memori (*embeddings*) dari lautan memori Host (RAM CPU biasa) yang kapasitasnya raksasa melalui kabel PCIe. 

Untuk menutupi kelambatan kabel transfer komunikasi ini secara efektif, modul Engram dengan sengaja diletakkan pada lapisan (layers) tertentu agak di *tengah/belakang* di dalam *backbone*, memanfaatkan waktu komputasi dari lapisan-lapisan awal yang sedang bekerja sebagai "penyangga waktu" (*time buffer*) untuk mencegah GPU mogok menganggur menunggu data masuk (*prevent GPU stalls*). Ini adalah desain tarian yang luar biasa antara perangkat keras dan algoritma (*hardware-algorithm co-design*).

Lebih ajaibnya lagi, sifat kemunculan frasa $N$-gram (kata-kata dalam bahasa manusia) secara matematis tunduk patuh pada **Distribusi Zipfian** (Chao and Zipf, 1950; Piantadosi, 2014), yaitu: Sebagian kecil frasa (kata-kata umum seperti "dan", "adalah") memonopoli mayoritas akses pembacaan memori. 
Sifat statistik ini memotivasi terciptanya Hierarki Cache Multi-Level (*Multi-Level Cache Hierarchy*): 
*   Penyematan yang sangat sering diakses akan di-cache (disimpan) di penyimpanan paling kilat (HBM GPU atau Host DRAM).
*   Pola kata/frasa langka (*long tail of rare patterns*) yang jarang muncul akan dibuang penyimpanannya ke media berkapasitas sangat besar namun agak lambat (misalnya, Hardisk NVMe SSD).

Stratifikasi penyimpanan ini merestui modul Engram untuk membengkak ke ukuran kapasitas memori miliaran gigabyte dengan dampak pengurangan kecepatan laten aktual yang nyaris tidak terasa.

---

## 3. Hukum Penskalaan dan Alokasi Ketersebaran (*Scaling Laws and Sparsity Allocation*)

Engram, sebagai manifestasi nyata dari memori bersyarat, secara struktural merupakan pelengkap sempurna (Yin & Yang) bagi komputasi bersyarat yang disajikan oleh pakar MoE. Bagian ilmiah ini menyelidiki properti penskalaan (sifat bagaimana ia bertumbuh) dari dualitas ini, dan mencari tahu *bagaimana cara mengalokasikan kapasitas renggang ini secara optimal?* 

Dua pertanyaan penelitian yang menggerakkan eksperimen kami:
1.  **Alokasi di Bawah Kendala Terbatas:** Ketika total parameter model dan total komputasi pelatihan sudah dipatok (dikunci mati sama), berapa porsi persen kapasitas yang harus dipecah antara Pakar MoE dan memori Engram?
2.  **Rezim Memori Tak Terbatas:** Mengingat *overhead* komputasi Engram itu konstan $\mathcal{O}(1)$ dan tidak bertambah berat walau memorinya dibesarkan, jika anggaran memori kita dibebaskan (unlimited), bagaimana wujud perilaku pertumbuhannya?

### 3.1. Rasio Alokasi Optimal Antara MoE dan Engram

**Formulasi Matched-Compute.** Kami membedah *trade-off* (tarik-ulur) ini menggunakan tiga metrik parameter matematis:
*   $P_{\text{tot}}$: Total seluruh parameter yang dilatih (*trainable parameters*).
*   $P_{\text{act}}$: Total parameter yang menyala/aktif per token. Angka inilah yang mendikte biaya waktu *training* (karena FLOPs dihitung dari sini).
*   $P_{\text{sparse}} \triangleq P_{\text{tot}} - P_{\text{act}}$: Ini adalah parameter pasif (tidur), yang mewakili anggaran "gratis" yang bisa dibesarkan tanpa membebani biaya komputasi tambahan.

Dalam setiap kelas anggaran komputasi FLOPs, kami mengunci kaku nilai $P_{\text{tot}}$ dan $P_{\text{act}}$. 
*   Untuk MoE, $P_{\text{act}}$ ditentukan oleh sejumlah ahli yang dipilih (top-$k$), sementara pakar sisanya menyumbang angka ke $P_{\text{sparse}}$. 
*   Untuk Engram, berapapun besarnya tabel, ia hanya mengambil jumlah laci memori yang *tetap* per tokennya. Jadi, menaikkan/membesarkan tabel Engram akan menaikkan $P_{\text{tot}}$ **tanpa** menaikkan komputasi per-token (FLOPs).

**Rasio Alokasi.** Kami mendefinisikan rasio alokasi $\rho \in [0, 1]$ sebagai persentase pecahan anggaran $P_{\text{sparse}}$ yang diberikan murni untuk kapasitas pakar MoE:

$$ P_{\text{MoE}}^{\text{(sparse)}} = \rho P_{\text{sparse}}, \quad P_{\text{Engram}} = (1 - \rho) P_{\text{sparse}}. \tag{7} $$

**Logika Intuitif:**
*   Jika $\rho = 1$, ini adalah model murni MoE 100% tradisional.
*   Jika $\rho < 1$, kita mengurangi jumlah pakar yang bertugas (MoE dikorbankan sedikit), lalu jatah parameter yang lowong (*freed parameters*) itu disuntikkan untuk membangun laci memori Engram.

**Protokol Eksperimental.** Kami mengeksekusi tarik-ulur ini pada dua alam komputasi berbeda (Model kecil dan besar) sembari mengunci rasio kerenggangan $P_{\text{tot}} / P_{\text{act}} \approx 10$:
*   Alam $\mathcal{C} = 2 \times 10^{20}$ FLOPs: Model total $\sim 5.7\text{B}$ dan parameter aktif $= 568\text{M}$. Baseline ($\rho=1$) memiliki 106 pakar.
*   Alam $\mathcal{C} = 6 \times 10^{20}$ FLOPs: Model total $\sim 9.9\text{B}$ dan parameter aktif $= 993\text{M}$. Baseline ($\rho=1$) memiliki 99 pakar.

Dengan memvariasikan nilai $\rho$, kami membongkar-pasang arsitektur ini berulang-ulang dengan mengubah rasio jumlah Pakar vs laci Engram. Semuanya di-*train* dengan ramuan resep algoritma yang identik.

---

### Analisis Akademis: Gambar 3 - Alokasi Ketersebaran dan Penskalaan Engram

Gambar ini adalah Cawan Suci (*Holy Grail*) dari penemuan makalah ini.

*   **Gambar 3 Kiri (Alokasi Sparsitas / *Sparsity Allocation*):**
    *   *Sumbu X:* Rasio Alokasi ($\rho$) dari 40% hingga 100%. (100% berarti murni MoE tanpa Engram. Bergerak ke kiri berarti porsi MoE dikurangi, porsi Engram ditambah).
    *   *Sumbu Y:* Kerugian Validasi (*Validation Loss*). Semakin rendah (lembah), semakin jenius modelnya.
    *   *Kurva Data:* Terdapat dua lengkungan berbentuk mangkuk "U" (kurva hijau muda untuk 6e20 FLOPs, dan ungu gelap untuk 2e20 FLOPs).
    *   *Kesimpulan Akademis Mutlak:* Perhatikan ujung kanan kurva (titik 100% Pure MoE). Ketika kita mulai mengurangi porsi MoE dan memindahkan jatah parameternya ke Engram (bergerak ke kiri menuju $\sim 80\%$), nilai *Loss* menurun drastis dan mencapai dasar lembah terdalam (paling cerdas). Ini membuktikan secara saintifik bahwa memadukan MoE dan Engram (Hibrida) JAUH LEBIH SUPERIOR daripada memaksakan 100% MoE. Ada sebuah keseimbangan ekuilibrium alam (bentuk U) di rasio $\rho \approx 80\%$.
*   **Gambar 3 Kanan (Penskalaan Engram Rezim Tak Terbatas):**
    *   *Sumbu X:* Jumlah slot *Embedding* memori (skala logaritmik $10^6$ hingga $10^7$).
    *   *Sumbu Y:* Kerugian Validasi (*Validation Loss*).
    *   *Kurva Data:* Garis ungu tua (Engram) menukik turun dengan sangat mulus dan tajam ke kanan bawah, membentuk garis lurus diagonal linier yang sempurna.
    *   *Kesimpulan Akademis Mutlak:* Ini adalah manifestasi dari Hukum Pangkat (*Power Law*). Semakin rakus dan raksasa Anda membesarkan ukuran memori Engram (sumbu X ke kanan), kecerdasan model (*Loss* turun) akan terus meningkat tanpa henti, dan ajaibnya, *tanpa menambah satu tetes pun daya beban komputasi FLOPs saat ia di-training*.

**Hasil dan Analisis Kesimpulan:**
Bentuk "U" yang dikonfirmasi pada Gambar 3 (kiri) menyegel validitas sifat komplementer (saling melengkapi) struktural antara kedua mesin ini:
*   **Jika didominasi MoE ($\rho \rightarrow 100\%$):** Model AI tidak punya memori mati untuk hafalan statis, sehingga ia bodoh karena dipaksa menghitung ulang memori lewat proses matematika yang tak efisien (menderita di komputasi).
*   **Jika didominasi Engram ($\rho \rightarrow 0\%$):** Model AI jadi amnesia logika. Ia kehilangan kapasitas untuk menghitung nalar dan logika bersyarat dinamis, karena memori hafalan kaku tidak akan pernah bisa menggantikan otak komputasi untuk tugas menalar konteks. Titik temu $\sim 80\%$ adalah jalan tengah kesempurnaan.

---

## 4. Prapelatihan Skala Raksasa (*Large Scale Pre-training*)

Bersenjatakan arsitektur Engram temuan kami dan hukum rasio alokasi ($80\%$) yang terbukti secara empiris tersebut, kami merakit dan meledakkan ukuran Engram hingga mencapai skala parameter multi-miliar untuk menguji kebrutalannya di dunia nyata. 

Kami melatih (dari nol) empat model petarung: 
1.  **Dense-4B:** Model bodoh/padat konvensional (Total 4.1B parameter).
2.  **MoE-27B:** Model pintar renggang (26.7B parameter, semua jatah diisi pakar MoE).
3.  **Engram-27B:** Model revolusioner kita (26.7B parameter). Kami memangkas jumlah pakar terarah (*routed experts*) dari 72 menjadi 55 saja, lalu parameter "gratisan" hasil pemangkasan itu kami cetak menjadi memori Engram berukuran 5.7B. 
4.  **Engram-40B:** Model monster (39.5B parameter). Tulang punggung (backbone) dibiarkan sama seperti Engram-27B, namun memori Engram-nya dibengkakkan hingga ukuran 18.5B. (Memori ekstra tanpa biaya komputasi).

*Hukum besi eksperimen:* Keempat model ini disiksa dengan kurikulum data yang identik mutlak (jumlah token dan urutan materi sama persis), dan secara sadis dikunci/dipatok mati (*strictly matched*) agar parameter yang berkedip (aktif / komputasi) untuk setiap model **tidak boleh lebih dari 3.8 Miliar (3.8B)**. Pertarungan yang sangat adil secara beban komputasi.

### Tabel 1: Perbandingan Performa Prapelatihan (Puncak Klasemen)

Mari kita telaah hasil kejuaraan ini melalui lensa akurasi objektif di **Tabel 1**.

| Benchmark (Metrik) | # Shots | Dense-4B | MoE-27B | **Engram-27B** | Engram-40B |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Pengetahuan & Penalaran** | | | | | |
| MMLU (Acc.) | 5-shot| 48.6 | 57.4 | **60.4** | 60.6 |
| MMLU-Redux (Acc.) | 5-shot| 50.7 | 60.6 | **64.0** | 64.5 |
| CMMLU (Acc.) | 5-shot| 47.9 | 57.9 | **61.9** | 63.4 |
| ARC-Challenge (Acc.) | 25-shot| 59.3 | 70.1 | **73.8** | 76.4 |
| BBH (EM) | 3-shot| 42.8 | 50.9 | **55.9** | 57.5 |
| **Kode & Matematika** | | | | | |
| HumanEval (Pass@1) | 0-shot| 26.8 | 37.8 | **40.8** | 38.4 |
| GSM8K (EM) | 8-shot| 35.5 | 58.4 | **60.6** | 62.6 |
| MATH (EM) | 4-shot| 15.2 | 28.3 | **30.7** | 30.6 |

*(Model Engram-27B iso-parameter langsung dibandingkan melawan MoE-27B).*

**Evaluasi Kritis (Review Dosen):**
1.  **Hukum Penskalaan Renggang (*Sparse Scaling*) Berlaku:** Model MoE dan Engram (keduanya model renggang) secara absolut melumat dan menghancurkan model Dense-4B (angka kemenangannya selisih puluhan persen). Ini menegaskan bahwa jika kita mengunci anggaran komputasi FLOPs yang sama, model renggang selalu keluar sebagai pemenang dominan.
2.  **Supremasi Engram di Atas MoE:** Perhatikan perbandingan *apple-to-apple* antara **MoE-27B** (kolom kedua) melawan **Engram-27B** (kolom ketiga yang angkanya dicetak tebal). 
    *   Sangat menarik dan kontra-intuitif! Padahal modul memori Engram (*lookup table*) dirancang murni untuk mengingat hafalan (seperti nama kota atau rumus pasti), yang mana secara logis mendongkrak ujian *MMLU* (+3.0 poin lonjakan dari 57.4 ke 60.4).
    *   **Kejutan Saintifik Terbesar:** Kami justru mengamati lonjakan prestasi yang **JAUH LEBIH MENGGILE** pada ujian **Penalaran Umum Logika (*General Reasoning*)** dan **Matematika/Pemrograman (Code & Math)**. Di ujian *BBH* yang super sulit, skornya melesat +5.0 (dari 50.9 menembus 55.9). Di *ARC-Challenge* naik +3.7 poin. Pada soal algoritma *HumanEval* ia mencuri tambahan +3.0 poin. Dan di *GSM8K* serta *MATH* ia naik +2.2 dan +2.4 poin.
3.  **Filosofi Mengapa Ini Terjadi:** Menyisipkan primitif pencarian pengetahuan khusus (*dedicated lookup*) ini terbukti membuat otak jaringan saraf menjadi *jauh lebih efisien*. Ia membersihkan tugas-tugas hafalan remeh dari pundak jaringan *attention* utama, sehingga sisa mesin komputasi MoE bisa mengerahkan 100% konsentrasinya murni untuk membedah silogisme logika matematika yang sulit.

---

## 5. Pelatihan Konteks Panjang (*Long Context Training*)

Dengan mengekskresikan (membuang beban) pemodelan memori ketergantungan jarak dekat (hafal-hafalan kata) kepada alat *lookup* statis Engram, arsitektur ini sukses membebaskan cadangan kapasitas *Attention* (fokus) model yang berharga untuk dialihkan khusus memelototi dan mengelola benang merah **konteks jarak jauh (konteks global).** 

Dalam sub-bab krusial ini, kami membuktikan keuntungan struktural ini secara empiris di lab melalui eksperimen **Pelatihan Ekstensi Konteks Panjang** (Gao et al., 2025; Peng et al., 2024).

**Pengaturan Eksperimen.** Kami menyedot parameter model dari tahap akhir prapelatihan (titik 50k steps) dan menjejalkannya ke sesi latihan tambahan menggunakan algoritma *YaRN* (Peng et al., 2024) untuk meregangkan batas jendela konteks otak AI hingga rentang panjang sejauh **32.768 token** (membaca dokumen setebal novel mini).

### Tabel 2: Komparasi Performa Konteks-Panjang (*Long-context performance comparison*)

| Model | LongPPL (32k) | | | | RULER (32k) | | | | |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| | **Perplexity (Jarang/Bagus Jika Turun $\downarrow$)** | | | | **NIAH Accuracy (Naik $\uparrow$)** | | **Other Tasks (Naik $\uparrow$)** | | |
| | Book | Paper | Code | L-CoT | S | MK | MV | **MQ** | **VT** |
| MoE-27B (50k) | 4.38 | 2.91 | 2.49 | 14.16 | 100.0 | 88.0 | 92.7 | 84.2 | 77.0 |
| **Engram-27B (46k)** | 4.19 | 2.84 | 2.45 | 13.59 | 97.6 | 89.0 | 95.5 | **97.0** | **87.2** |
| **Engram-27B (50k)** | **4.14** | **2.82** | **2.44** | **13.41** | 99.3 | **89.3** | **96.5** | **97.0** | **89.0** |

**Ulasan Kritis (Review Dosen) atas Tabel 2:**
1.  **Daya Ingat Super (*Recall/Retrieval Power*):** Tolok ukur `RULER` adalah tes jarum-dalam-tumpukan-jerami (*Needle-in-a-Haystack/NIAH*). Pada kolom `MQ` (*Multi-Query* - mengingat banyak fakta sekaligus yang dikubur di naskah panjang) dan `VT` (*Variable Tracking* - melacak ubahan variabel rumit di sepanjang kode), model **MoE-27B** standar (baris 1) terengah-engah dan tersedak di skor **84.2** dan **77.0**. 
2.  Sekarang tatap skor **Engram-27B** yang telah dipersenjatai memori kustom (baris 2 dan 3). Skornya melesat bak roket menghancurkan atmosfer, menembus angka menakjubkan **97.0** dan **89.0** (naik belasan persen mutlak)!
3.  **Kesimpulan Epik:** Engram bertindak seperti "buku catatan saku kilat". Karena ia sudah hafal idiom dan entitas lokal pendek, mesin otaknya tidak gampang terdistraksi (lupa) ketika harus merajut benang merah argumen jarak jauh dari halaman 1 hingga halaman 100 di dalam sebuah dokumen panjang. Keunggulan arsitektural Engram sangat mutlak terbukti di sini.

---

## 6. Analisis Mekanistik Jeroan Model (*Analysis*)

Pertanyaan filosofisnya: *Apakah Engram secara fungsional setara dengan menambah kedalaman (jumlah layer) model otak AI?*

LLM di luaran sana (seperti GPT atau LLaMA) tidak memiliki pustaka primitiif "cari pengetahuan" (*knowledge lookup*). Untuk mengenali entitas rumit seperti nama tokoh sejarah `"Diana, Princess of Wales"`, LLM standar terpaksa membakar komputasi dengan menyiksanya bergulung-gulung melintasi banyak lapisan (*multiple layers of Attention and FFNs*) untuk merakit fitur-fitur kata tersebut sedikit demi sedikit, langkah demi langkah (lihat Tabel 3 di makalah asli). Ini adalah pengerjaan rodi komputasional.

Dengan menyuntikkan Engram (kemampuan *lookup* instan), kami berhipotesis bahwa Engram bagaikan mesin waktu. Ia menyelesaikan pekerjaan berat merakit fitur di tahap paling awal, membebaskan lapisan-lapisan di atasnya. 

Untuk membuktikan hipotesis sains ini, kami menghadirkan instrumen bedah saraf AI: **LogitLens** dan **Centered Kernel Alignment (CKA)**.

### Analisis Akademis: Gambar 4 - Analisis Kecepatan Konvergensi Penyelarasan Representasional

*   **Deskripsi Visual (a): Layer-wise KL Divergence via LogitLens**
    *   *Sumbu X:* Indeks Lapisan (*Layer Index*) dari otak depan (0) ke otak belakang (30).
    *   *Sumbu Y:* Jarak Divergensi KL (Semakin tinggi, artinya prediksi model masih mentah/jauh dari matang. Semakin ke bawah/mendekati 0, artinya model sudah sangat yakin dan matang akan memuntahkan kata apa).
    *   *Kurva:* Garis MoE (biru gelap) berjalan landai lambat di atas. Garis Engram (merah & cyan) **menukik jatuh terjun bebas (*steeper descent*) jauh lebih cepat dan lebih curam** di lapisan-lapisan awal (lapisan 0 hingga 15).
    *   *Kesimpulan:* Turun curamnya kurva Engram membuktikan bahwa model menyelesaikan perakitan makna kata (feature composition) **jauh lebih cepat**. Berkat contekan dari memori eksternal, Engram tidak perlu berpikir keras; ia langsung menggapai tingkat kepercayaan diri (*high-confidence*) di awal rantai saraf, melangkahi birokrasi pemrosesan hierarkis model klasik.

*   **Deskripsi Visual (b & c): Peta Panas Kemiripan (*Similarity heatmap*) dihitung oleh CKA**
    *   *Sumbu X & Y:* Membandingkan nomor lapisan Engram vs lapisan MoE standar. Garis putus-putus putih (*soft alignment curve*) melacak di lapisan nomor berapa kemiripan matematis otak mereka paling identik.
    *   *Pola Terlihat:* Garis putus-putus putih itu **melengkung membengkok ke arah KIRI ATAS (*upward shift*)**. 
    *   *Kesimpulan Saintifik:* Pergeseran ini sungguh magis. Ini berarti bahwa representasi (pemahaman) data yang berhasil dirakit oleh Engram di **Lapisan ke-5** (layer dangkal) itu wujudnya sudah sama persis, setara, dan identik tingkat kesulitannya dengan pemahaman yang baru bisa dicapai oleh MoE standar di **Lapisan ke-12** (layer dalam)! Secara harfiah, memasang Engram adalah trik *cheat code* yang menjadikan otak model tersebut secara fungsional bertindak layaknya memiliki jumlah lapisan kedalaman yang disulap jauh lebih panjang/dalam (*increasing the model's effective depth*).

---

### Analisis Akademis: Gambar 5 - Hasil Ablasi Arsitektur (Di mana letak Engram yang terbaik?)

*   **Deskripsi Visual:** 
    *   *Sumbu X:* Indeks Lapisan (*Layer Index*).
    *   *Sumbu Y:* Kerugian Validasi (*Validation Loss*, lebih bawah lebih pintar).
    *   *Garis Oranye Dotted (Atas):* Batas kebodohan MoE standar.
    *   *Kurva Biru Gelap (U-Shape):* Ini adalah hasil eksperimen *Layer Sweep* (Menyapu letak). Peneliti memindahkan modul Engram secara paksa diinjeksi ke layer 1, lalu dicoba di layer 2, lalu 3, hingga 12. 
*   **Kesimpulan Strategis dari Grafik:** 
    *   Kurva biru mencapai dasar cekungan paling dalam (paling optimal) saat Engram disuntikkan secara presisi di **Lapisan ke-2 (Layer 2)**. Begitu Engram digeser ditaruh di layer yang makin dalam (layer 10-12), performanya hancur memburuk mendekati MoE standar.
    *   *Mengapa harus di Layer 2? (Trade-off Penempatan / Placement trade-off):* Memasang Engram terlalu awal (di Layer 0 atau 1) itu buruk, karena status vektor kalimat masih mentah, belum punya "konteks kalimat global" untuk memerintah pintu gerbang (*gate*) dalam memilih memori yang tepat. Memasangnya di Layer 2 sangat pas: otak model sudah sempat mencerna "makna konteks" (*contextualized queries*) di layer 1, lalu Engram masuk memberesi urusan hafalan lokal, sebelum sisa lapisan atas dipusingkan oleh penalaran sulit.

---

### Analisis Akademis: Gambar 6 - Kinerja Model di bawah Siksaan "Ablasi Engram" (Mematikan Paksa Memori)

Untuk mengetahui apa sebenarnya isi perut memori Engram ini, para peneliti melakukan interogasi paksa di bawah stres tes ekstrem: Model disuruh mengerjakan ujian, TAPI di pertengahan jalan, **modul memori Engram-nya DIMATIKAN PAKSA (disuntik mati/ *completely suppressing the output*)**. Mari kita saksikan dampaknya pada grafik batang.

*   **Deskripsi Visual:** Diagram batang yang menelusuri Persentase Kinerja yang Bertahan (*Retained Performance %*). Angka 100% berarti model tidak terganggu, angka 20% berarti model tiba-tiba menjadi bodoh/amnesia parah.
*   **Pengelompokan Warna Batang:**
    *   Biru Muda (*Reading Comprehension*): Kemampuan membaca wacana (C3, RACE).
    *   Kuning (*Commonsense Reasoning*): Nalar akal sehat manusia (HellaSwag, PIQA).
    *   Merah Muda (*Algorithmic Reasoning*): Nalar matematika berat (BBH, GSM8K, MATH).
    *   **Krem Terang (*Factual Knowledge/Knowledge-Intensive*):** Ujian trivia hafalan sejarah/fakta (TriviaQA, PopQA).
*   **Kesimpulan Kritis Kehancuran (Catastrophic Collapse):** 
    *   Lihat batang-batang biru (Membaca/Reading). Kinerjanya bertahan sangat kuat di 81%-93% (misal di C3 skornya 93%). Artinya, mencabut memori Engram tidak merusak kemampuan AI untuk memahami teks bacaan secara nalar, karena tulang punggung jaringan *Attention* (bukan Engram) yang memang mengurus itu.
    *   Sekarang, tatap tragedi horor di batang-batang warna krem di ujung kanan (Fakta/TriviaQA). Kinerjanya **rontok, anjlok, dan hancur lebur tersisa hanya 29% (TriviaQA) hingga 44% (MMLU)!** 
    *   **Konklusi Mutlak:** Eksperimen pembedahan keji ini membuktikan tak terbantahkan secara saintifik bahwa modul Engram di dalam arsitektur AI ini benar-benar sukses dan secara eksklusif telah bertransformasi menjadi **Gudang Penyimpanan Rakus Fakta Pengetahuan (*Primary repository for parametric knowledge*)**. Sifat kognisi model telah berhasil dipisahkan!

---

## 7. Kesimpulan (*Conclusion*)

Di dalam karya penelitian ini, kami memperkenalkan dan melegitimasi konsep **memori bersyarat (*conditional memory*)** sebagai sumbu (arah) ketersebaran/sparsitas pelengkap yang fundamental untuk disandingkan mengawal kemegahan paradigma komputasi bersyarat sang penguasa industri (*Mixture-of-Experts* / MoE). 

Inovasi ini ditelurkan murni untuk menebas dan menghancurkan problem inefisiensi arsitektur Transformer yang selama ini terengah-engah berpura-pura mensimulasikan pencarian memori pengetahuan melalui jalan paksaan komputasi dinamis yang menyiksa. Kami merealisasikan (menginstansiasi) gagasan radikal ini ke dalam wujud modul **Engram**. Modul ini memodernisasi debu-debu klasik dari rumus lama (*N-gram embeddings*) dan menghidupkannya kembali untuk mendukung pencarian kamus (*lookups*) konstan $\mathcal{O}(1)$ yang instan dan terukur untuk pola-pola kata yang statis dan kaku.

Dengan membedah problem kalkulus **Alokasi Ketersebaran (*Sparsity Allocation*)**, kami menemukan sebuah fenomena fisika-AI baru: **Hukum Penskalaan Berbentuk-U**. Penemuan ini membuktikan secara deduktif bahwa menyilangkan alokasi kapasitas parameter tidur (*sparse*) secara hibrida adil ke dalam pakar MoE dan memori Engram, secara absolut dan bengis akan **melumat dan mempecundangi *baseline* arsitektur MoE murni** yang selama ini dipuja-puja. 

Diperkuat keyakinan akan hukum suci ini, kami mengembuskan napas merentangkan skala penciptaan monster parameter Engram hingga menyentuh angka raksasa 27 Miliar (27B). Sang Engram-27B ini sukses merebut takhta keunggulan kinerja lintas hamparan beragam domain industri. 

Di sisi lain, temuan yang membuat kami para ilmuwan AI paling takjub terperangah: Meskipun sejak awal desain modul saku memori ini (Engram) secara naluriah hanya ditujukan sekadar demi menolong kelancaran tarikan kuis Trivia / hafalan pengetahuan (*knowledge retrieval*), kami secara empiris malah mengobservasi luapan ledakan panen perolehan (*gains*) kemajuan intelektualitas yang **jauh melampaui batas kewajaran di ranah nalar filsafat logika umum (*general reasoning*), bahasa mesin (koding), dan kalkulus matematika murni!**

Pembedahan mekanistik kami di akhir lab membongkar rahasia trik sulap ini: Engram secara tak kasat mata telah **"memperdalam"** ruang nalar otak jaringan (*deepen the network*) karena ia memerdekakan secara instan pundak penderitaan lapisan-lapisan pertama (*early layers*) jaringan dari kerja paksa rodi membangun ulang konstruksi balok lego teks memori statis. Kemerdekaan kapasitas ini membebaskan seluruh otot komputasi kapasitas atensi difokuskan 100% matang hanya untuk menggodok konteks silogisme global dan meramu penalaran kompleks semata. 

Puncaknya (Akhirul kalam), Engram menyodorkan filosofi fatwa baru di mana keefisienan *sadar-infrastruktur perangkat keras* (*infrastructure-aware efficiency*) diposisikan sebagai pedoman desain arsitektur tingkat satu. Sistem panggil pengalamatan memori ID yang secara pasti deterministik (*deterministic addressing*) mengizinkan kita memisahkan penyimpanan (RAM) dan komputasi (GPU) ibarat membelah lautan. Ia merestui manuver *offloading*—melempar dan menitipkan tabel raksasa parameter ingatan (*massive parameter tables*) ke pangkuan memori RAM Induk CPU biasa yang murah—dengan **siksaan kelambatan waktu inferensi yang nyaris sama sekali tidak terasa di dunia nyata (*negligible inference overhead*)**. 

Kami meramalkan secara mutlak bahwa kehadiran fungsi memori bersyarat ini tak lama lagi akan berevolusi merangsek maju untuk menobatkan dirinya sebagai primitif pemodelan fondasional yang *mustahil untuk dikesampingkan* (*indispensable modeling primitive*) bagi rahim lahirnya generasi penerus model-model *sparse* AI di masa depan peradaban.

*(Tamat - Penutup Diktat Akademis).*
