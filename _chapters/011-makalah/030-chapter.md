---
slug: ch-11-03
title: "DeepSeek-Coder"
layout: chapter
---

## DeepSeek-Coder - Ketika Model Bahasa Besar Bertemu Pemrograman

Perkembangan pesat Model Bahasa Besar (*Large Language Models*/LLMs) telah merevolusi kecerdasan kode (*code intelligence*) dalam pengembangan perangkat lunak. Namun, dominasi model bersumber tertutup (*closed-source*) telah membatasi penelitian dan pengembangan yang ekstensif di kalangan akademisi dan komunitas. Untuk mengatasi hambatan ini, diperkenalkan seri **DeepSeek-Coder**, sebuah rangkaian model kode sumber terbuka (*open-source*) dengan ukuran parameter mulai dari 1,3 Miliar (1.3B) hingga 33 Miliar (33B), yang dilatih dari awal (*trained from scratch*) pada 2 triliun token. 

Model-model ini menjalani prapelatihan (*pre-training*) pada korpus kode tingkat proyek (*project-level*) berkualitas tinggi dan menggunakan tugas pengisian-bagian-yang-kosong (*fill-in-the-blank*) dengan jendela konteks (*context window*) sebesar 16K token untuk meningkatkan kemampuan pembuatan (*generation*) dan pengisian kode (*infilling*). Evaluasi ekstensif menunjukkan bahwa DeepSeek-Coder tidak hanya mencapai kinerja mutakhir (*state-of-the-art*) di antara model kode sumber terbuka di berbagai tolok ukur (*benchmarks*), tetapi juga melampaui model tertutup yang ada seperti Codex dan GPT-3.5. Selain itu, model DeepSeek-Coder dirilis di bawah lisensi permisif yang memungkinkan penelitian akademis dan penggunaan komersial tanpa batas.

---

### Analisis Gambar 1: Kinerja DeepSeek-Coder
Sebagai pengantar visual, makalah ini menyajikan **Gambar 1** yang merangkum kinerja model. 
* **Deskripsi Visual:** Gambar 1 terdiri dari dua grafik yang berdampingan. Di sebelah kiri adalah grafik radar (*radar chart*), dan di sebelah kanan adalah diagram batang (*bar chart*).
* **Grafik Radar (Kiri):** Memetakan kinerja (*Pass@1 score*) dari berbagai model pada berbagai bahasa pemrograman (Python, C++, Java, PHP, TypeScript, C#, Bash, JavaScript). Terdapat beberapa kurva melingkar:
  * *Kurva Biru Tua:* Mewakili DeepSeek-Coder-33B. Kurva ini berada di posisi paling luar di hampir semua sumbu bahasa, menunjukkan dominasi kinerja yang komprehensif.
  * *Kurva Merah/Oranye:* Mewakili StarCoder-16B.
  * *Kurva Abu-abu/Biru Muda:* Mewakili varian CodeLlama (13B dan 34B) serta CodeGeeX2-6B.
* **Diagram Batang (Kanan):** Menampilkan kinerja pada tolok ukur *LeetCode Weekly Contest*. 
  * *Sumbu X:* Menunjukkan berbagai jenis model dari kiri ke kanan (WizardCoder, CodeLlama, Phind, DeepSeek-Coder-7B, DeepSeek-Coder-33B).
  * *Sumbu Y:* Mewakili persentase akurasi *Pass@1* (%).
  * *Garis Putus-putus (*Dotted lines*):* Terdapat garis merah putus-putus di atas (GPT-4-Turbo: 41.8) dan garis hijau putus-putus di tengah (GPT-3.5-Turbo: 23.3).
  * *Kesimpulan Akademis:* Bar tertinggi di antara model sumber terbuka adalah DeepSeek-Coder-33B-Instruct (warna biru pekat) yang melampaui angka 28.9%, secara meyakinkan menembus batas kinerja GPT-3.5-Turbo dan secara signifikan mempersempit kesenjangan dengan raksasa tertutup GPT-4-Turbo.

***

## 2. Pendahuluan

Bidang pengembangan perangkat lunak telah bertransformasi secara signifikan oleh kemajuan cepat Model Bahasa Besar (LLMs). Model-model ini membawa era baru kecerdasan kode, memiliki potensi untuk mengotomatisasi banyak aspek pengkodean, mulai dari deteksi *bug* hingga pembuatan kode otomatis, sehingga meningkatkan produktivitas dan mengurangi kesalahan manusia. 

Namun, tantangan utama di lapangan adalah **kesenjangan kinerja (*performance gap*)** antara model sumber terbuka (seperti StarCoder, CodeLlama) dan model sumber tertutup (seperti Gemini, OpenAI GPT-4). Model raksasa yang tertutup seringkali tidak dapat diakses secara mendalam oleh peneliti karena sifat kepemilikannya.

Sebagai respons, seri **DeepSeek-Coder** dihadirkan. Setiap model dalam seri ini (1.3B, 6.7B, dan 33B) dilatih pada 2 triliun token yang bersumber dari 87 bahasa pemrograman. Ada dua inovasi utama dalam pendekatannya:
1. **Prapelatihan Tingkat Repositori (*Repository-level Pre-training*):** Data diatur berdasarkan repositori untuk meningkatkan pemahaman model terhadap konteks lintas-berkas (*cross-files*) di dalam satu proyek, bukan sekadar potongan berkas tunggal.
2. **Pendekatan *Fill-In-Middle* (FIM):** Selain menggunakan fungsi kerugian prediksi token berikutnya (*next token prediction loss*), model dilatih untuk mengisi kode di tengah-tengah teks (*infilling*).
3. **Konteks Panjang 16K:** Model dapat menangani panjang masukan hingga 16.000 token, memungkinkan penyelesaian skenario pengkodean yang luas dan kompleks.

Eksperimen membuktikan bahwa model basis 33B mengungguli semua model terbuka lainnya, sementara versi instruksi (Instruct 33B) mengungguli GPT-3.5 Turbo. Hebatnya, versi 7B pun mampu bersaing dengan model yang berukuran lima kali lipat lebih besar seperti CodeLlama-33B.

***

## 3. Pengumpulan Data (*Data Collection*)

Kualitas model AI sangat bergantung pada data pelatihannya. Kumpulan data DeepSeek-Coder terdiri dari:
* **87%** Kode sumber (*source code*).
* **10%** Korpus bahasa alami berbahasa Inggris terkait kode (Markdown GitHub dan StackExchange).
* **3%** Korpus bahasa alami berbahasa Mandarin (tidak terkait kode, untuk menanamkan kemampuan bahasa).

### Analisis Gambar 2: Prosedur Pembuatan Kumpulan Data
* **Deskripsi Visual:** Gambar ini adalah diagram alir (*flowchart*) horizontal yang merinci lima tahapan pipa pemrosesan data (*data pipeline*).
  1. *Data Crawling* (Ikon gurita GitHub) mengarah ke basis data.
  2. *Rule Filtering* (Ikon saringan) untuk menyaring data mentah.
  3. *Dependency Parsing* (Ikon relasi dokumen) untuk menyusun urutan berkas.
  4. *Repo-level Deduplication* (Ikon tempat sampah) untuk membuang duplikat dalam satu repositori.
  5. *Quality Screening* (Ikon lencana/plus) untuk memastikan hanya data bermutu tinggi yang masuk ke model.
* **Kesimpulan Akademis:** Diagram ini merepresentasikan standar pemrosesan big data yang wajib dilakukan dalam proyek rekayasa perangkat lunak AI, mengubah *noise* menjadi *signal* yang bersih untuk pelatihan.

### 3.1 Perayapan dan Penyaringan Data GitHub (*GitHub Data Crawling and Filtering*)
Kami mengumpulkan repositori publik GitHub yang dibuat sebelum Februari 2023 untuk 87 bahasa pemrograman. Untuk membersihkan data berkalitas rendah, aturan penyaringan yang diadopsi dari proyek StarCoder diterapkan. Proses ini memangkas data menjadi hanya **32,8%** dari ukuran aslinya. Aturan tersebut meliputi:
* Menghapus berkas dengan rata-rata panjang baris > 100 karakter atau panjang baris maksimal > 1000 karakter.
* Menghapus berkas dengan karakter alfabet kurang dari 25%.
* Menyaring berkas XML yang mengandung `<?xml version=`.
* Untuk HTML, rasio teks yang terlihat harus minimal 20% dan >= 100 karakter.
* Untuk JSON dan YAML, hanya mempertahankan berkas dengan panjang 50 hingga 5000 karakter untuk membuang berkas padat data (*data-heavy files*).

### 3.2 Penguraian Ketergantungan (*Dependency Parsing*)
Sebagian besar LLM sebelumnya dilatih pada kode tingkat berkas tunggal, mengabaikan hierarki antar-berkas (*cross-file dependencies*). Praktiknya, pemrogram bekerja dalam ekosistem *proyek*. Oleh karena itu, kita mem-*parsing* ketergantungan antar berkas dalam satu repositori dan mengurutkannya menggunakan **Topological Sort**. Tujuannya agar berkas yang menjadi *dependency* (misal: berkas fungsi dasar) muncul *lebih dulu* dalam urutan input (konteks) sebelum berkas yang memanggilnya.

Berikut adalah algoritma yang digunakan untuk analisis ketergantungan:

```python
Algorithm 1 Topological Sort for Dependency Analysis
1: procedure TOPOLOGICALSORT(files)
2:     graphs ← {}               # Inisialisasi daftar ketetanggaan kosong
3:     inDegree ← {}             # Inisialisasi kamus kosong untuk in-degrees (derajat-masuk)
4:     for each file in files do
5:         graphs[file] ← []
6:         inDegree[file] ← 0
7:     end for
8: 
9:     for each fileA in files do
10:        for each fileB in files do
11:            if HASDEPENDENCY(fileA, fileB) then     # Jika fileA bergantung pada fileB
12:                graphs[fileB].append(fileA)         # Tambahkan sisi (edge) dari B ke A
13:                inDegree[fileA] ← inDegree[fileA] + 1 # Tingkatkan in-degree dari A
14:            end if
15:        end for
16:    end for
17:
18:    subgraphs ← getDisconnectedSubgraphs(graphs)    # Identifikasi subgraf yang terputus
19:    allResults ← []
20:    for each subgraph in subgraphs do
21:        results ← []
22:        while length(results) ≠ NumberOfNodes(subgraph) do
23:            file ← argmin({inDegree[node] | node ∈ subgraph and node ∉ results})
24:            for each node in graphs[file] do
25:                inDegree[node] ← inDegree[node] - 1
26:            end for
27:            results.append(file)
28:        end while
29:        allResults.append(results)
30:    end for
31: 
32:    return allResults
33: end procedure
```
**Penjelasan Logika Algoritma Baris-demi-Baris:**
* **Baris 1-7:** Menginisialisasi graf *Directed Acyclic Graph* (DAG). `graphs` mewakili graf ketetanggaan, dan `inDegree` menghitung berapa banyak prasyarat yang harus dipenuhi oleh suatu berkas sebelum dapat diproses.
* **Baris 9-16:** Membangun graf. Jika `fileA` mengimpor/bergantung pada `fileB`, maka kita membuat panah berarah dari `fileB` ke `fileA`. Artinya, `fileB` harus dipelajari model lebih dulu. Derajat-masuk (`inDegree`) `fileA` ditambah 1.
* **Baris 18:** Proyek perangkat lunak mungkin memiliki beberapa modul independen (tidak saling memanggil). Baris ini memisahkan mereka menjadi subgraf-subgraf.
* **Baris 20-30:** Inti dari modifikasi *Topological Sort*. Alih-alih hanya mencari *node* dengan derajat-masuk persis 0 (seperti teori standar), baris 23 mencari *node* dengan derajat-masuk **paling minimum** (`argmin`). Hal ini sangat krusial karena dalam kode pemrograman di dunia nyata, sering terjadi kebuntuan ketergantungan melingkar (*circular dependencies*). Memilih nilai minimum memungkinkan algoritma untuk terus berjalan memutus siklus tersebut. Node yang terpilih dihapus dari antrean dan derajat prasyarat tetangganya dikurangi (Baris 24-26).
* Hasil akhirnya adalah urutan berkas yang saling terhubung logis, direkatkan menjadi satu sampel pelatihan panjang dengan komentar jalur berkas (*file path*) di bagian atas setiap kode.

### 3.3 Deduplikasi Tingkat Repositori (*Repo-Level Deduplication*)
Literatur (seperti Lee et al., 2022) membuktikan bahwa deduplikasi (penghapusan kode yang berulang/mirip) sangat menaikkan kinerja LLM. Berbeda dengan pendekatan klasik yang menghapus duplikasi tingkat-berkas (*file-level*), metode ini melakukan penghapusan duplikasi dekat (*near-deduplication*) pada **tingkat repositori**. Mengapa? Jika kita menghapus berkas tunggal yang mirip di dalam suatu repositori, kita akan merusak struktur hierarki *dependency* yang susah payah dibangun pada Algoritma 1. Oleh karena itu, seluruh kode dalam satu repositori digabungkan menjadi satu sampel raksasa, lalu deduplikasi dilakukan antar-repositori.

### 3.4 Penyaringan Kualitas dan Dekontaminasi (*Quality Screening and Decontamination*)
Kualitas kode diperiksa kembali menggunakan *compiler* dan heuristik (keterbacaan, kesalahan sintaks). **Tabel 1** (disajikan di bawah) merangkum ukuran korpus final yang mencapai 798 GB dengan 603 juta berkas.

**Dekontaminasi (*Decontamination*):** Sebagai akademisi, Anda harus waspada terhadap "kebocoran data" (*data leakage*), di mana model secara tidak sengaja "menghafal" jawaban soal ujian (data tes) saat pelatihan. Untuk mencegahnya, dilakukan penyaringan n-gram. Jika ada 10 gram (*10-gram string*) dari data pelatihan yang sama persis dengan yang ada di set evaluasi utama (HumanEval, MBPP, GSM8K, MATH), kode tersebut **dibuang**.

### Tabel 1: Ringkasan Data Pelatihan yang Telah Dibersihkan

| Language | Size (GB) | Files (k) | Prop. (%) | Language | Size (GB) | Files (k) | Prop. (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Ada | 0.91 | 126 | 0.11 | Literate Haskell | 0.16 | 20 | 0.02 |
| Agda | 0.26 | 59 | 0.03 | Lua | 0.82 | 138 | 0.10 |
| Alloy | 0.07 | 24 | 0.01 | Makefile | 0.92 | 460 | 0.12 |
| ... *(Beberapa bahasa disingkat demi fokus)* | | | | | | | |
| C | 28.64 | 27,111 | 3.59 | PowerShell | 0.91 | 236 | 0.11 |
| C# | 58.56 | 53,739 | 7.34 | Python | 120.68 | 75,188| 15.12|
| C++ | 90.87 | 36,006 | 11.39| Ruby | 15.01 | 18,526 | 1.88 |
| HTML | 30.05 | 14,998 | 3.77 | SQL | 15.14 | 7,009 | 1.90 |
| Java | 148.66 | 134,367| 18.63| TypeScript | 60.62 | 62,432 | 7.60 |
| JavaScript| 53.84 | 71,895 | 6.75 | YAML | 0.74 | 890 | 0.09 |
| **Total** | | | | | **797.92**| **603,173**| **100.00**|

*(Catatan Dosen: Proporsi terbesar dipegang oleh Java (18.63%), Python (15.12%), dan C++ (11.39%). Menandakan model ini sangat kuat dalam bahasa-bahasa sistem backend dan data science.)*

***

## 4. Kebijakan Pelatihan (*Training Policy*)

### 4.1 Strategi Pelatihan

#### 4.1.1 Prediksi Token Berikutnya (*Next Token Prediction*)
Objektif pertama adalah prediksi token berikutnya, standar bagi arsitektur *decoder-only*. Model mempelajari distribusi statistik kata untuk memprediksi token selanjutnya secara runut berdasarkan konteks yang diberikan.

#### 4.1.2 Pengisian-di-Tengah (*Fill-in-the-Middle* / FIM)
Objektif kedua ini revolusioner untuk pemodelan kode. Programer seringkali perlu menyisipkan kode di antara baris kode yang sudah ada. Model dilatih dengan membagi berkas menjadi tiga bagian secara acak: Prefix (Awalan), Middle (Tengah), dan Suffix (Akhiran).
Ada dua mode:
1. **PSM (Prefix-Suffix-Middle):** Disusun sebagai `Prefix, Suffix, Middle`.
2. **SPM (Suffix-Prefix-Middle):** Disusun sebagai `Suffix, Prefix, Middle`.

### Analisis Gambar 3: Efektivitas Objektif FIM
* **Deskripsi Visual:** Gambar ini menyajikan enam grafik garis kecil yang melacak kinerja selama proses pelatihan (Sumbu X: Langkah/Steps, Sumbu Y: Nilai Kinerja). Grafik membandingkan HumanEval, FIM, MBPP, dll.
* **Garis Data:** Terdapat tiga kurva mewakili tingkat implementasi FIM (0% FIM, 50% FIM, dan 100% FIM) serta mode MSP (*Masked Span Prediction*).
* **Kesimpulan Akademis:** Kurva dengan tingkat FIM 100% menghasilkan skor luar biasa pada tugas infilling, namun justru melemah di pembuatan kode generik. Mode PSM pada *rate* 50% menawarkan kinerja paling stabil dan berimbang di kedua dunia (tugas prediksi maju dan tugas penyisipan). Oleh karena itu, tingkat PSM 50% dipilih sebagai kebijakan akhir.

Dalam penerapannya, DeepSeek-Coder memisahkan kode menggunakan tiga token penanda (*sentinel tokens*). Format pelatihannya secara spesifik diatur sebagai berikut:

```markdown
<|fim_start|> Prefix <|fim_hole|> Suffix <|fim_end|> Middle <|eos_token|>
```
* **Penjelasan Struktur Prompt:**
  * `<|fim_start|>` menandakan dimulainya pemrosesan berkas.
  * *Prefix* adalah kode yang berada sebelum kursor pengguna.
  * `<|fim_hole|>` adalah titik penanda tempat kursor berada (di mana kode hilang).
  * *Suffix* adalah sisa kode di bawah kursor.
  * `<|fim_end|>` menandai bahwa model sekarang harus mulai memprediksi/mengisi bagian yang hilang (*Middle*) hingga ditutup dengan penanda akhir `<|eos_token|>`.

### 4.2 Pembuat Token (*Tokenizer*)
Model menggunakan *Byte Pair Encoding* (BPE) melalui pustaka HuggingFace Tokenizer. BPE mengelompokkan urutan byte yang sering muncul, menekan panjang sekuens. Ukuran kosakata (*vocabulary size*) ditetapkan pada 32.000 token.

### 4.3 Arsitektur Model
Dibangun berdasarkan kerangka arsitektur Transformer tipe *Decoder-only*. Fitur utamanya:
* **Rotary Position Embedding (RoPE):** Mengonversi posisi token ke dalam bentuk rotasi vektor untuk pemahaman posisi relatif yang sangat presisi.
* **Grouped-Query-Attention (GQA):** Diimplementasikan khusus untuk versi 33B (dengan ukuran grup 8), memungkinkan penghematan VRAM dan kecepatan komputasi.
* **FlashAttention v2:** Algoritma atensi memori-efisien untuk mempercepat pelatihan secara drastis.

### Tabel 2: Hiperparameter DeepSeek-Coder

| Hyperparameter | DeepSeek-Coder 1.3B | DeepSeek-Coder 6.7B | DeepSeek-Coder 33B |
| :--- | :--- | :--- | :--- |
| Hidden Activation | SwiGLU | SwiGLU | SwiGLU |
| Hidden size | 2048 | 4096 | 7168 |
| Intermediate size | 5504 | 11008 | 19200 |
| Hidden layers number | 24 | 32 | 62 |
| Attention heads number| 16 | 32 | 56 |
| Attention | Multi-head | Multi-head | Grouped-query (8)|
| Batch Size | 1024 | 2304 | 3840 |
| Max Learning Rate | 5.3e-4 | 4.2e-4 | 3.5e-4 |

### 4.4 Optimisasi
Model dilatih menggunakan pengoptimal AdamW dengan momentum $\beta_1 = 0.9$ dan $\beta_2 = 0.95$. Jadwal tingkat pembelajaran (*learning rate scheduling*) menerapkan kebijakan tiga tahap. *Learning rate* di setiap tahap akhir diskalakan turun dengan persamaan matematika berikut:

$$ LR_{new} = LR_{old} \times \sqrt{\frac{1}{10}} $$

* **Penjelasan Rumus:** Laju pembelajaran tidak dipotong secara ekstrem dan tiba-tiba, melainkan diturunkan mengikuti akar kuadrat dari sepertsepuluh kecepatan sebelumnya. Ini menjaga stabilitas model agar gradien tidak melonjak keluar batas saat parameter mulai mendekati titik optimal (konvergensi lokal).

### 4.5 Lingkungan Pelatihan (*Environments*)
Kerangka kerja HAI-LLM digunakan karena sangat ringan dan efisien. Ia merangkum paradigma paralelisme tingkat tinggi:
* **Tensor Parallelism:** Membagi satu matriks tensor ke beberapa GPU (membelah "satu neuron" raksasa).
* **ZeRO Data Parallelism:** Membagi *optimizer state*, *gradient*, dan *parameters* melintasi GPU.
* **PipeDream Pipeline Parallelism:** Membagi jaringan berlapis-lapis (layer 1-10 di GPU 1, layer 11-20 di GPU 2, dst).
Semua ini dieksekusi di atas klaster NVIDIA A100 dan H800 yang dijembatani oleh NVLink, NVSwitch, dan InfiniBand (jaringan komunikasi antar-server ultra-cepat).

### 4.6 Konteks Panjang (*Long Context*)
Untuk memroses repositori, model perlu membaca puluhan ribu baris kode sekaligus. RoPE dimodifikasi menggunakan strategi penskalaan linier (*linear scaling strategy*), mengubah frekuensi dasar (*base frequency*) dari 10.000 menjadi 100.000, serta faktor penskalaan dari 1 menjadi 4. Melalui penyesuaian analitik ini, jendela konteks direntangkan hingga 16K (16.384) token.

### 4.7 Pelatihan Halus Instruksi (*Instruction Tuning*)
Model dasar (*Base*) disempurnakan menjadi asisten yang responsif (*Instruct*) menggunakan format instruksi Alpaca. Tanda pembatas `<|EOT|>` (End of Turn) disematkan untuk memisahkan setiap dialog manusia dan AI. 

### Analisis Gambar 4: Contoh Respons Interaksi Multi-Turn
* **Deskripsi Visual:** Memperlihatkan tampilan obrolan UI tiruan antara pengguna (Q1, Q2) dan model DeepSeek-Coder-Instruct 33B (Kiri dan Kanan). Terdapat kotak terminal hitam yang merepresentasikan visualisasi *output* grafis dari kode.
* **Q1 (Kiri):** Pengguna meminta model membuat "game snake" menggunakan perpustakaan `pygame`.
  * *Respons Model:* Model menulis logika `pygame` utuh dari mendirikan kanvas, sistem kontrol *arrow keys*, logika pertumbuhan ular, hingga fungsi *game over*. Gambar terminal menunjukkan *game* ular klasik berjalan secara primitif (hanya piksel merah dan hijau).
* **Q2 (Kanan):** Pengguna meminta *follow-up*, "Tambahkan sistem skor di pojok kiri atas".
  * *Respons Model:* Alih-alih menulis ulang seluruh modul, model hanya memberikan variabel `score` baru dan baris kode `plt.show()` / *rendering text* yang harus disisipkan di antara kode sebelumnya, serta penjelasan konseptual singkat. Gambar terminal menunjukkan angka "Score: 7" tampil di atas *game*.
* **Kesimpulan Akademis:** Ini memvalidasi kecerdasan *State-Tracking* (pelacakan status). Model tidak amnesia setelah percakapan pertama selesai. Model mengerti *contextual delta*, di mana ia hanya perlu memutakhirkan kode tanpa merusak logika dasar sistem di putaran sebelumnya.

***

## 5. Hasil Eksperimen (*Experimental Results*)

Model dievaluasi dalam empat pilar pengkodean dan dibandingkan dengan tolok ukur fundamental seperti:
* **CodeGeeX2 (6B):** Model berorientasi multibahasa generasi kedua.
* **StarCoder (16B):** Dilatih pada *The Stack*, salah satu pionir berat sumber terbuka.
* **CodeLlama (7B, 13B, 34B):** Varian turunan LLaMA2 buatan Meta khusus kode.
* **code-cushman-001 (12B):** Model prarilis dari OpenAI yang mendasari GitHub Copilot.
* **GPT-3.5 dan GPT-4:** Model generatif paling mutakhir (sumber tertutup).

### 5.1 Pembuatan Kode (*Code Generation*)

Pilar ini mengevaluasi seberapa jago model memecahkan soal algoritma kompleks dari nol secara *Zero-shot*. Metrik utama adalah **Pass@1** (probabilitas model memberikan jawaban benar pada tebakan pertama).

### Tabel 3: Kinerja pada Multilingual HumanEval dan MBPP

| Model | Size | Python | C++ | Java | PHP | TS | C# | Bash | JS | Avg | MBPP |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Multilingual Base Models** |
| CodeGeeX2 | 6B | 36.0% | 29.2% | 25.9%| 23.6%| 20.8% | 29.7% | 6.3% | 24.8%| 24.5%| 36.2%|
| StarCoderBase | 16B| 31.7% | 31.1% | 28.5%| 25.4%| 34.0% | 34.8% | 8.9% | 29.8%| 28.0%| 42.8%|
| CodeLlama | 34B| 48.2% | 44.7% | 44.9%| 41.0%| 42.1% | 48.7% | 15.8%| 42.2%| 41.0%| 55.2%|
| DeepSeek-Coder | 6.7B| 49.4% | 50.3% | 43.0%| 38.5%| 49.7% | 50.0% | 28.5%| 48.4%| 44.7%| 60.6%|
| DeepSeek-Coder | 33B| **56.1%**| **58.4%**| **51.9%**| **44.1%**| **52.8%** | **51.3%** | **32.3%**| **55.3%**| **50.3%**| **66.0%**|
| **Instruction-Tuned Models** |
| GPT-3.5-Turbo | - | 76.2% | 63.4% | 69.2%| 60.9%| 69.1% | 70.8% | 42.4%| 67.1%| 64.9%| 70.8%|
| GPT-4 | - | **84.1%**| **76.4%**| **81.6%**| **77.2%**| **77.4%** | **79.1%** | **58.2%**| **78.0%**| **76.5%**| **80.0%**|
| DeepSeek-Instruct | 33B| 79.3% | 68.9% | 73.4%| 72.7%| 67.9% | 74.1% | 43.0%| 73.9%| **69.2%**| 70.0%|

**Kesimpulan Pengamatan:** Rata-rata akurasi DeepSeek-Coder 33B (50.3%) mendepak telak CodeLlama 34B (41.0%). Yang paling fenomenal, model kecil 6.7B berhasil meruntuhkan dominasi LLaMA 34B. Pada kelas model *Instruction*, DeepSeek 33B berhasil mengambil alih posisi GPT-3.5-Turbo.

#### Tolok Ukur DS-1000
Berbeda dengan algoritma abstrak, DS-1000 mengukur alur kerja ilmu data nyata (*real data science workflows*) melalui integrasi pustaka ilmiah (Matplotlib, Numpy, TensorFlow, dll.).

### Tabel 4: Kinerja pada Tugas DS-1000

| Model | Size | Matplotlib | Numpy | Pandas | Pytorch | Scipy | Scikit-Learn | Tensorflow | Avg |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| CodeLlama-Base| 34B| 50.3% | 42.7% | 23.0% | 25.0% | 28.3% | 33.9% | 40.0% | 34.3% |
| DeepSeek-Coder| 33B| **56.1%** | **49.6%**| **25.8%**| **36.8%**| **36.8%**| **40.0%** | **46.7%** | **40.2%** |

#### Tolok Ukur Kontes LeetCode
Menguji algoritma level kompetisi (ekstrem). Dievaluasi dengan instruksi tipe *Rantai Pemikiran* (*Chain-of-Thought* / CoT).

### Tabel 5: Kinerja pada Kontes LeetCode

| Model | Size | Easy (45) | Medium (91) | Hard (44) | Overall(180) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| CodeLlama-Instruct | 34B | 24.4% | 4.4% | 4.5% | 9.4% |
| GPT-3.5-Turbo | - | 46.7% | 15.4% | 15.9% | 23.3% |
| GPT-4-Turbo | - | **73.3%** | 31.9% | 25.0% | 40.6% |
| DeepSeek-Instruct | 33B | 57.8% | 22.0% | 9.1% | 27.8% |
| DeepSeek-Instruct + CoT| 33B | 53.3% | **25.3%** | **11.4%** | **28.9%** |

*Catatan Kritis Mahasiswa:* Penambahan panduan CoT ("Tulis langkah-langkah logika sebelum menulis kode") secara paradoksial menurunkan poin pada tingkat *Easy*, namun mendongkrak performa memecahkan teka-teki logika pada tingkat *Medium* dan *Hard*. Hal ini mensahihkan teori bahwa menyusun spesifikasi deskriptif terlebih dulu membantu pelacakan dependensi dalam masalah super-kompleks.

### 5.2 Penyelesaian Kode *Fill-in-the-Middle* (FIM)
Mengevaluasi kecepatan dan ketepatan auto-komplit berdasarkan satu baris sisipan.

### Tabel 6: Pendekatan FIM

| Model | Size | python | java | javascript | Mean |
| :--- | :--- | :--- | :--- | :--- | :--- |
| StarCoder | 16B | 62.0% | 73.0% | 74.0% | 69.7% |
| CodeLlama-Base| 13B | **68.3%** | 77.6% | 80.7% | 75.5% |
| DeepSeek-Coder| 7B | 66.6% | **88.1%** | 79.7% | 80.7% |
| DeepSeek-Coder| 33B | 65.4% | 86.6% | **82.5%** | **81.2%** |

### 5.3 Penyelesaian Kode Lintas-Berkas (*Cross-File Code Completion*)
Berkat algoritma *Topological Sort* (Algoritma 1) dan Prapelatihan berbasis Repo, evaluasi CrossCodeEval menghasilkan keunggulan yang mutlak di ekosistem ini.

### Tabel 7: Penyelesaian kode Lintas-Berkas (Akurasi Exact Match/EM)

| Model | Size | Python (EM) | Java (EM) | TypeScript (EM) | C# (EM) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| CodeLlama-Base + Retrieval | 7B | 13.02% | 16.41% | 12.34% | 13.19% |
| DeepSeek-Coder + Retrieval | 6.7B | **16.14%** | **17.72%** | **14.03%** | **16.23%** |
| DeepSeek + Retrieval **w/o Repo Pre-training** | 6.7B | 16.02% | 16.64% | 13.23% | 14.48% |

*(Ablasi Terbukti: Jika "Repo Pre-training" dimatikan / tidak digunakan (baris terbawah), skor akurasi lintas file Java dan C# anjlok drastis. Eksperimen ini memvalidasi teori perancangan awal).*

### 5.4 Penalaran Matematika Berbasis Program (*Program-based Math Reasoning*)
Penalaran jenis ini memanfaatkan *Program-Aided Math Reasoning* (PAL). Alih-alih meminta AI menghitung sendiri (yang mana AI sering melakukan halusinasi hitungan aritmatika), AI diminta menyusun naskah program *Python* untuk melakukan hitungan tersebut, lalu mengeksekusinya.

### Tabel 8: Tugas Penalaran Matematika via Pemrograman

| Model | Size | GSM8k | MATH | GSM-Hard | SVAMP | TabMWP | ASDiv | MAWPS | Avg |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| CodeLlama-Base| 34B | 58.2% | 21.2% | 51.8% | 70.3% | 69.8% | 70.7% | 91.8% | 62.0% |
| DeepSeek-Coder| 33B | **60.7%**| **29.1%**| **54.1%** | **71.6%** | **75.3%** | **76.7%** | **93.3%** | **65.8%** |

***

## 6. Melanjutkan Prapelatihan Dari LLM Generik (*Continue Pre-Training*)

Apakah model yang dilatih khusus kode akan menjadi autistik pada pemahaman bahasa alami manusia? Untuk bereksperimen, alih-alih dilatih dari awal, inisialisasi bobot dimulai dari parameter model generik (DeepSeek-LLM-7B) dan ditambahkan 2 triliun token campuran (kode, teks Inggris/Mandarin, matematika). Model eksperimen ini dijuluki **DeepSeek-Coder-v1.5 7B**.

### Tabel 9: Sumber Data untuk Prapelatihan DeepSeek-Coder-v1.5
| Data Source | Percentage |
| :--- | :--- |
| Source Code | 70% |
| Markdown and StackExchange | 10% |
| Natural language related to code | 7% |
| Natural language related to math | 7% |
| Bilingual (Chinese-English) natural language | 6% |

### Tabel 10: Analisis Komparatif DeepSeek-Coder-Base vs v1.5

| Models | Size | HumanEval | MBPP | GSM8K | MATH | MMLU | BBH | HellaSwag | WinoG | ARC-C |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Base | 6.7B | **44.7%** | **60.6%**| 43.2% | 19.2% | 36.6% | 44.3% | 53.8% | 57.1% | 32.5% |
| Base-v1.5 | 6.9B | 43.2% | 60.4% | **62.4%**| **24.7%**| **49.1%**| **55.2%**| **69.9%** | **63.8%**| **47.2%**|
| Instruct | 6.7B | **66.1%** | **65.4%**| 62.8% | 28.6% | 37.2% | 46.9% | 55.0% | 57.6% | 37.4% |
| Instruct-v1.5 | 6.9B | 64.1% | 64.6% | **72.6%**| **34.1%**| **49.5%**| **53.3%**| **72.2%** | **63.4%**| **48.1%**|

*Pengamatan Dosen:* Perhatikan bahwa nilai *Coding* (HumanEval/MBPP) sedikit mengalami pendarahan turun karena pencampuran data bahasa manusia. Namun imbal baliknya luar biasa masif: kecerdasan spasial manusia dan keluwesan teks (MMLU, HellaSwag, GSM8K) meroket tajam nyaris ganda. 

***

## 7. Kesimpulan

Dalam laporan teknis komprehensif ini, kita telah mengeksplorasi seri parameter ekstensif Model Bahasa Besar khusus pengkodean, DeepSeek-Coder, dalam tiga spesifikasi: 1,3B, 6,7B, dan 33B. Model-model ini disuntik dengan prapelatihan unik menggunakan pengurutan logis *topological* untuk mendeteksi *dependency* pada level repositori utuh. Penggunaan penyisipan-kode melalui teknik *Fill-in-the-Middle* bersama dengan penyesuaian matriks frekuensi berbasis *Rotary Position Embedding* memungkinkan model ini membaca sekumpulan kode sepanjang 16.384 token tanpa *crash*.

Di ranah evaluasi performa, DeepSeek-Coder-Base 33B telah resmi merajai seluruh panggung persaingan model kode di ranah sumber terbuka (*open-source*), menumbangkan CodeLlama. Dan yang lebih mengesankan dari arsitektur komputasional ini, versi mungil 6.7B mampu berakselerasi sejajar dengan raksasa 34B lainnya. 

Optimalisasi pelacakan instruksi melalui integrasi format dialog Alpaca juga meledakkan performa model instruksi (Instruct 33B) sedemikian rupa, hingga sanggup melampaui kemampuan deduksi pemrograman dari platform tertutup komersial, GPT-3.5 Turbo.

Sumbangsih pamungkas riset ini terdapat pada seri DeepSeek-Coder-v1.5. Pendekatan ini melempar konklusi baru bahwa melatih LLM untuk tugas *coding* berat sekalipun, pada ujungnya membutuhkan korpus pengetahuan bahasa alami yang komprehensif agar logika matematis dan pemahaman terhadap prompt manusia tidak bersifat kaku. Kemajuan ini menggarisbawahi komitmen para peneliti kecerdasan buatan untuk meracik agen LLM *code-focused* di masa depan yang tidak hanya jago mengetik baris sintaks, melainkan mumpuni menyerap dan menerjemahkan kehendak organik manusia.