---
slug: ch-11-04
title: "DeepSeekMath - Mendorong Batas Penalaran Matematis pada Model Bahasa Terbuka"
layout: chapter
---

## Abstrak

Penalaran matematis (*mathematical reasoning*) memberikan tantangan yang sangat signifikan bagi Model Bahasa Besar (*Large Language Models* / LLMs) karena sifatnya yang kompleks dan terstruktur. Dalam bab ini, kita akan membedah **DeepSeekMath 7B**, sebuah model yang melanjutkan tahap prapelatihan (*pre-training*) dari model DeepSeek-Coder-Base-v1.5 7B. Model ini dilatih menggunakan 120 Miliar (*120B*) token yang berkaitan dengan matematika, yang bersumber dari *Common Crawl*, dipadukan dengan data bahasa alami (*natural language*) dan kode pemrograman.

DeepSeekMath 7B berhasil mencapai skor impresif sebesar **51,7%** pada tolok ukur (*benchmark*) MATH tingkat kompetisi tanpa bergantung pada perangkat bantu eksternal (*external toolkits*) maupun teknik pemungutan suara (*voting techniques*), sebuah pencapaian yang mendekati tingkat kinerja model raksasa tertutup seperti Gemini-Ultra dan GPT-4. Melalui teknik *Self-consistency* atas 64 sampel, DeepSeekMath 7B bahkan mencapai skor **60,9%** pada MATH. 

Kemampuan penalaran matematis DeepSeekMath yang luar biasa ini dikaitkan dengan **dua faktor kunci**: 
1. Pemanfaatan potensi data web publik secara maksimal melalui alur kerja (*pipeline*) seleksi data yang direkayasa dengan sangat teliti.
2. Pengenalan **Group Relative Policy Optimization (GRPO)**, sebuah varian dari algoritma *Proximal Policy Optimization* (PPO), yang secara khusus meningkatkan kemampuan penalaran matematis sembari mengoptimalkan penggunaan memori (RAM/VRAM) secara drastis dibandingkan PPO standar.

---

### Analisis Gambar 1: Kinerja Model Sumber Terbuka pada Tolok Ukur MATH
Mari kita perhatikan secara saksama **Gambar 1** yang disajikan oleh para peneliti sebagai ringkasan pencapaian.

*   **Deskripsi Visual:** Gambar 1 adalah grafik garis (*line chart*) yang memplot akurasi Top-1 model AI sumber terbuka (*open-source*) pada tolok ukur MATH tingkat kompetisi (Hendrycks et al., 2021) seiring berjalannya waktu.
*   **Sumbu X (Horizontal):** Merepresentasikan Garis Waktu (*Date*) dari rilis model, mulai dari kuartal pertama tahun 2023 hingga awal 2024.
*   **Sumbu Y (Vertikal):** Merepresentasikan Akurasi Top-1 pada dataset MATH (*MATH Top@1 Accuracy*) dalam persentase, mulai dari 10% hingga di atas 50%.
*   **Titik Data dan Kurva:** Terdapat sebuah garis putus-putus hitam (*black dashed line*) yang menanjak ke atas, menghubungkan berbagai model terbuka dari waktu ke waktu (LLaMA1-65B, WizardMath-70B, Qwen-14B, Mistral-7B, Llemma-34B, Qwen-72B). 
*   **Puncak Kurva (Bintang Merah):** Di ujung paling kanan atas, terdapat lambang bintang merah (*red star*) yang merepresentasikan **DeepSeekMath-7B**. 
*   **Garis Referensi Horizontal:** Terdapat garis putus-putus abu-abu tebal di kisaran akurasi ~42% yang mewakili kinerja awal GPT-4 (*GPT-4 early version*), dan garis di kisaran ~53% yang mewakili GPT-4 API dan Gemini-Ultra.
*   **Kesimpulan Akademis:** Grafik ini mengilustrasikan "lompatan kuantum" dalam AI sumber terbuka. Meskipun model-model sebelumnya (seperti Qwen-72B yang ukurannya 10 kali lebih besar) hanya mentok di bawah angka 40%, DeepSeekMath-7B berhasil meroket melampaui versi awal GPT-4 dan menempel ketat dominasi raksasa sumber tertutup (Gemini-Ultra dan GPT-4 API). Ini membuktikan bahwa efisiensi algoritma (GRPO) dan kualitas data jauh lebih krusial daripada sekadar membengkakkan jumlah parameter model.

***

## 1. Pendahuluan

Model Bahasa Besar (LLM) telah merevolusi pendekatan penalaran matematis dalam Kecerdasan Buatan, memacu kemajuan signifikan baik dalam tolok ukur penalaran kuantitatif (Hendrycks et al., 2021) maupun tolok ukur penalaran geometri (Trinh et al., 2024). Lebih jauh lagi, model-model ini telah terbukti sangat membantu manusia dalam memecahkan masalah matematika yang kompleks (Tao, 2023). Namun, model-model mutakhir (*cutting-edge*) seperti GPT-4 (OpenAI, 2023) dan Gemini-Ultra (Anil et al., 2023) tidak tersedia untuk publik, dan model sumber terbuka yang ada saat ini masih tertinggal jauh dalam hal kinerja.

Dalam studi ini, para peneliti memperkenalkan **DeepSeekMath**, sebuah model bahasa spesifik domain (*domain-specific*) yang secara signifikan mengungguli kemampuan matematis model sumber terbuka lainnya dan mendekati tingkat kinerja GPT-4 pada tolok ukur akademik. 

Untuk mencapai hal ini, peneliti menciptakan **DeepSeekMath Corpus**, sebuah korpus prapelatihan berskala besar dan berkualitas tinggi yang terdiri dari 120B token matematika. Kumpulan data ini diekstraksi dari *Common Crawl* (CC) menggunakan pengklasifikasi (*classifier*) berbasis *fastText* (Joulin et al., 2016). Pada iterasi awal, pengklasifikasi dilatih menggunakan instansi dari OpenWebMath (Paster et al., 2023) sebagai contoh positif, sambil memasukkan beragam pilihan halaman web lain sebagai contoh negatif. Selanjutnya, pengklasifikasi ini digunakan untuk menambang (*mine*) instansi positif tambahan dari CC, yang kemudian disempurnakan melalui anotasi manusia. Pengklasifikasi tersebut kemudian diperbarui dengan kumpulan data yang telah ditingkatkan ini untuk memperbaiki kinerjanya. 

Hasil evaluasi menunjukkan bahwa korpus berskala besar ini memiliki kualitas yang sangat tinggi. Model dasar kita, **DeepSeekMath-Base 7B**, mencapai skor 64,2% pada GSM8K (Cobbe et al., 2021) dan 36,2% pada dataset MATH tingkat kompetisi, mengungguli Minerva 540B (Lewkowycz et al., 2022a) yang berukuran jauh lebih raksasa. Selain itu, Korpus DeepSeekMath bersifat multibahasa (*multilingual*), sehingga terjadi peningkatan yang signifikan pada tolok ukur matematika berbahasa Mandarin.

DeepSeekMath-Base diinisialisasi dari **DeepSeek-Coder-Base-v1.5 7B** (Guo et al., 2024). Para peneliti menyadari bahwa memulai pelatihan dari model yang sudah dilatih pada kode (*code training model*) adalah pilihan yang jauh lebih baik dibandingkan memulai dari LLM umum. Pelatihan matematika juga secara empiris meningkatkan kemampuan model pada tolok ukur MMLU (Hendrycks et al., 2020) dan BBH (Suzgun et al., 2022), yang mengindikasikan bahwa data matematika tidak hanya meningkatkan kemampuan kalkulasi, tetapi juga memperkuat kemampuan penalaran umum (*general reasoning capabilities*).

Setelah prapelatihan, diterapkan proses **Pelatihan Halus Instruksi Matematis (*Mathematical Instruction Tuning*)** pada DeepSeekMath-Base menggunakan data *Chain-of-Thought* (CoT) (Wei et al., 2022), *Program-of-Thought* (PoT) (Chen et al., 2022; Gao et al., 2023), dan penalaran terintegrasi perangkat bantu (*tool-integrated reasoning*) (Gou et al., 2023). Hasilnya, **DeepSeekMath-Instruct 7B** mengalahkan semua model saingan di kelas 7B dan sebanding dengan model terlatih instruksi berukuran 70B.

Lebih jauh lagi, diperkenalkan **Group Relative Policy Optimization (GRPO)**, sebuah algoritma Pembelajaran Penguatan (*Reinforcement Learning* / RL) yang merupakan varian dari *Proximal Policy Optimization* (PPO) (Schulman et al., 2017). GRPO membuang model kritikus (*critic model*) yang biasanya boros memori, dan sebagai gantinya mengestimasi *baseline* dari skor grup, yang secara drastis mengurangi kebutuhan sumber daya pelatihan. Dengan hanya menggunakan subset data instruksi berbahasa Inggris, GRPO memperoleh peningkatan substansial di atas DeepSeekMath-Instruct. Peningkatan ini mencakup tugas di dalam domain (*in-domain*) seperti GSM8K (82,9% → 88,2%) dan MATH (46,8% → 51,7%), serta tugas di luar domain (*out-of-domain*) seperti CMATH (84,6% → 88,8%).

Makalah ini juga menyediakan paradigma terpadu (*unified paradigm*) untuk memahami berbagai metode RL, seperti *Rejection Sampling Fine-Tuning* (RFT), *Direct Preference Optimization* (DPO), PPO, dan GRPO.

### 1.1 Kontribusi Penelitian

Penelitian ini membagi kontribusinya ke dalam dua area utama: Prapelatihan Matematika pada Skala Besar dan Eksplorasi/Analisis Pembelajaran Penguatan (RL).

**Prapelatihan Matematika pada Skala Besar (*Math Pre-Training at Scale*)**
*   Memberikan bukti kuat bahwa data *Common Crawl* publik mengandung informasi matematis yang sangat berharga. Melalui *pipeline* seleksi data yang teliti, tercipta **DeepSeekMath Corpus** (120B token), yang ukurannya hampir 7 kali lipat lebih besar dari yang digunakan Minerva dan 9 kali lebih besar dari OpenWebMath.
*   Model dasar **DeepSeekMath-Base 7B** membuktikan bahwa jumlah parameter bukanlah satu-satunya faktor penentu penalaran matematis. Model kecil yang dilatih pada data kualitas tinggi dapat menyaingi model raksasa (Minerva 540B).
*   Membuktikan bahwa **pelatihan kode (*code training*) sebelum pelatihan matematika** meningkatkan kemampuan pemecahan masalah matematis. Ini menjawab pertanyaan akademik jangka panjang: *apakah pelatihan kode meningkatkan kemampuan penalaran?* Jawabannya adalah ya.
*   Menemukan fakta menarik bahwa melatih model menggunakan kumpulan makalah ilmiah arXiv (yang umum dilakukan) ternyata **tidak memberikan peningkatan yang nyata** pada tolok ukur matematis yang diadopsi dalam makalah ini.

**Eksplorasi dan Analisis Pembelajaran Penguatan (*Reinforcement Learning*)**
*   Memperkenalkan **GRPO**, yang menghemat sumber daya komputasi secara signifikan dengan membuang model *critic*.
*   Mendemonstrasikan bahwa GRPO secara masif meningkatkan kinerja model yang telah diinstruksi (DeepSeekMath-Instruct), bahkan memberikan efek positif pada kinerja di luar domain.
*   Menyajikan **Paradigma Terpadu (*Unified Paradigm*)** untuk menganalisis RFT, DPO, PPO, dan GRPO.
*   Berdasarkan paradigma terpadu, mengeksplorasi alasan di balik efektivitas RL dan merangkum arah potensial untuk RL yang lebih efisien pada LLM di masa depan.

### 1.2 Ringkasan Evaluasi dan Metrik
*   **Penalaran Matematika Inggris dan Mandarin:** Menguji model dari tingkat sekolah dasar hingga perguruan tinggi (GSM8K, MATH, SAT, OCW Courses, MMLU-STEM, MGSM-zh, CMATH, Gaokao-Math).
*   **Matematika Formal:** Mengevaluasi model menggunakan tugas pembuktian teorema informal-ke-formal menggunakan *proof assistant* Isabelle pada dataset miniF2F.
*   **Pemahaman Bahasa Alami, Penalaran, dan Kode:** Dievaluasi pada MMLU, BIG-Bench Hard (BBH), HumanEval, dan MBPP.

***

## 2. Prapelatihan Matematika (*Math Pre-Training*)

### 2.1 Pengumpulan Data dan Dekontaminasi

Bagian ini menguraikan proses konstruksi **DeepSeekMath Corpus** dari *Common Crawl*. Pendekatan yang digunakan adalah *pipeline* iteratif yang dapat diadaptasi ke domain lain (misalnya, domain kode).

Sebagai mahasiswa AI, Anda harus memahami alur ini secara saksama. Mari kita analisis **Gambar 2** yang merepresentasikan arsitektur pengumpulan data ini.

### Analisis Gambar 2: Pipeline Iteratif Pengumpulan Web Matematika
*   **Deskripsi Visual:** Gambar 2 menampilkan diagram alir sirkular (berbentuk siklus tertutup) yang mendeskripsikan 4 langkah berulang untuk mengekstrak data dari internet (Common Crawl). Di tengah terdapat ikon bola dunia besar (*Deduplicated Common Crawl 40B HTML pages*).
*   **Langkah 1 (Kiri Atas): Train a FastText Model.** Menggunakan "Math Seed" (Ikon daun dengan rumus) untuk melatih model klasifikasi awal berbasis *fastText*.
*   **Langkah 2 (Kanan Atas): Recall Math-Related Webpages.** Model *fastText* yang telah dilatih digunakan untuk menarik/menemukan halaman web dari Common Crawl yang dianggap bernuansa matematika. Halaman-halaman ini dikumpulkan menjadi "Math Corpus" (Ikon pohon dengan warna-warni rumus).
*   **Langkah 3 (Kanan Bawah): Discover Math-Related Domains.** Menganalisis hasil dari Langkah 2 untuk menemukan domain asal (*base URL*) yang memiliki persentase konten matematika sangat tinggi (misal: situs forum matematika).
*   **Langkah 4 (Kiri Bawah): Annotate Math-Related URL Path.** Manusia (labeler) melakukan anotasi manual pada *path* URL dari domain yang ditemukan di Langkah 3. URL yang tervalidasi kemudian diumpankan kembali (panah ke atas) untuk memperkaya "Math Seed" awal.
*   **Kesimpulan Akademis:** Ini adalah metode *Active Learning* / *Bootstrapping* dalam kurasi dataset. Alih-alih membuat satu aturan statis, *pipeline* ini terus menerus belajar dari temuannya sendiri, memperluas cakupan deteksinya secara eksponensial di setiap iterasi, namun tetap terjaga kualitasnya melalui intervensi manual (Langkah 4).

**Detail Proses Teknis:**
Seed awal menggunakan **OpenWebMath**. Model *fastText* dilatih menggunakan 500.000 data positif dari seed ini, dan 500.000 data negatif acak dari Common Crawl. Parameter *fastText* diatur: dimensi vektor = 256, *learning rate* = 0,1, maksimum *n-gram* kata = 3, minimum kemunculan kata = 3, dan jumlah *epoch* = 3. 

Untuk memperkecil ukuran *Common Crawl*, dilakukan deduplikasi tingkat URL dan *near-deduplication*, menyisakan 40B halaman HTML. Model *fastText* lalu memberikan skor (probabilitas matematika) untuk setiap halaman. Pada iterasi pertama, 40B token teratas dipertahankan.

Untuk mengatasi kurangnya keragaman, dilakukan Langkah 3 dan 4. Domain di mana lebih dari 10% halamannya terkoleksi pada iterasi pertama diklasifikasikan sebagai *math-related* (contoh: `mathoverflow.net`). Setelah 4 iterasi perulangan, didapatkan **35,5 Juta halaman web** dengan total **120B token**. Pada iterasi keempat, 98% data telah terkumpul di iterasi ketiga, sehingga koleksi dihentikan (konvergensi data).

**Dekontaminasi (*Decontamination*):**
Untuk mencegah model berbuat "curang" dengan secara tidak sengaja menghafal set ujian (seperti GSM8K dan MATH) selama pelatihan, dilakukan filter dekontaminasi yang ketat. Semua segmen teks yang mengandung *string* **10-gram** (10 kata berurutan) yang sama persis dengan *string* apa pun dari *benchmark* evaluasi, **DIHAPUS** dari korpus pelatihan. Untuk soal yang panjangnya kurang dari 10 kata (minimal 3 kata), digunakan pencocokan persis (*exact matching*).

### 2.2 Memvalidasi Kualitas DeepSeekMath Corpus

Untuk membuktikan kualitas korpus 120B ini, peneliti melakukan eksperimen komparatif melawan korpus matematika mapan lainnya:
*   **MathPile:** 8,9B token (mayoritas makalah arXiv).
*   **OpenWebMath:** 13,6B token.
*   **Proof-Pile-2:** 51,9B token (digunakan oleh Llemma, rasio arXiv:Web:Code adalah 2:4:1).

#### 2.2.1 Pengaturan Pelatihan (*Training Setting*)
Pelatihan dilakukan pada model berukuran **1,3B parameter** menggunakan *framework* HAI-LLM. Menggunakan *optimizer* AdamW ($\beta_1=0.9, \beta_2=0.95$, *weight decay*=0.1). *Learning rate* puncak di 5,3e-4 (dicapai setelah 2.000 *warmup steps*), lalu turun bertahap ke 31,6% dan akhirnya 10,0%. *Batch size* 4M token, konteks 4K.

#### Tabel 1: Kinerja Model 1.3B pada Berbagai Korpus Matematika

Tabel ini krusial untuk membuktikan bahwa kuantitas dan kualitas data DeepSeek mengungguli *dataset* pendahulunya secara mutlak.

| Math Corpus | Size | GSM8K | MATH | OCW | SAT | MMLU STEM | CMATH | Gaokao MathCloze | Gaokao MathQA |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| No Math Training | N/A | 2.9% | 3.0% | 2.9% | 15.6% | 19.5% | 12.3% | 0.8% | 17.9% |
| MathPile | 8.9B | 2.7% | 3.3% | 2.2% | 12.5% | 15.7% | 1.2% | 0.0% | 2.8% |
| OpenWebMath | 13.6B | 11.5% | 8.9% | 3.7% | 31.3% | 29.6% | 16.8% | 0.0% | 14.2% |
| Proof-Pile-2 | 51.9B | 14.3% | 11.2% | 3.7% | 43.8% | 29.2% | 19.9% | 5.1% | 11.7% |
| **DeepSeekMath Corpus** | **120.2B** | **23.8%** | **13.6%** | **4.8%** | **56.3%** | **33.1%** | **41.5%** | **5.9%** | **23.6%** |

*(Dievaluasi menggunakan few-shot chain-of-thought prompting).*

#### 2.2.2 Hasil Evaluasi
*   **Kualitas Tinggi:** Berdasarkan Tabel 1, model yang dilatih dengan DeepSeekMath Corpus mendominasi semua 8 *benchmarks*. 
*   **Multibahasa (*Multilingual*):** Korpus ini memasukkan data bahasa Inggris dan Mandarin. Terlihat pada kolom *Chinese Benchmarks* (CMATH, Gaokao), model ini memiliki kinerja superior, sedangkan korpus lama (seperti MathPile) justru merusak kemampuan Mandarin model.

### Analisis Gambar 3: Kurva Tolok Ukur Selama Pelatihan Model 1.3B
Untuk memahami dinamika pembelajaran mesin selama proses *training*, mari kita analisis **Gambar 3**.

*   **Deskripsi Visual:** Terdapat empat grafik garis yang menunjukkan peningkatan akurasi (*Acc %*) pada 4 tolok ukur berbeda (GSM8K, MATH, CMATH, BBH) seiring dengan jumlah token yang diproses oleh model.
*   **Sumbu X (Horizontal):** Jumlah Token dalam Miliar (*Tokens (B)*), mulai dari 0 hingga 150 Miliar token.
*   **Sumbu Y (Vertikal):** Akurasi dalam persentase (*Acc %*).
*   **Kurva Data:** Terdapat 4 garis yang mewakili jenis korpus yang digunakan untuk pelatihan: Biru (MathPile), Oranye (OpenWebMath), Hijau (Proof-Pile-2), dan Merah (DeepSeekMath Corpus).
*   **Tren dan Kesimpulan Akademis:**
    *   Garis Merah (DeepSeekMath) secara konsisten berada paling atas dengan kurva yang terus menanjak tajam seiring bertambahnya token (tidak cepat stagnan/mendatar).
    *   Korpus lain yang lebih kecil (seperti OpenWebMath yang hanya 13.6B) cepat mencapai titik *plateau* (mendatar) karena model terus membaca data yang sama berulang-ulang (*multiple epochs*). 
    *   Hal ini membuktikan bahwa **Skala Besar (*Large-scale*)** dari DeepSeekMath Corpus memberikan "kurva pembelajaran yang lebih curam" dan "peningkatan yang lebih bertahan lama" pada bobot neural (*neural weights*) LLM.

### 2.3 Melatih dan Mengevaluasi DeepSeekMath-Base 7B
Model DeepSeekMath-Base 7B diinisialisasi dari **DeepSeek-Coder-Base-v1.5 7B** dan dilatih selama 500B token. Distribusi datanya: 56% DeepSeekMath Corpus, 4% AlgebraicStack, 10% arXiv, 20% kode Github, dan 10% teks bahasa alami CC (Inggris & Mandarin).

#### Tabel 2: Komparasi Model Dasar (*Base Models*) dengan Pendekatan *Step-by-Step*

Tabel ini menunjukkan supremasi model dasar berparameter 7B melawan model raksasa tertutup.

| Model | Size | GSM8K | MATH | OCW | SAT | MMLU STEM | CMATH | Gaokao MathCloze | Gaokao MathQA |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Closed-Source Base Model** |
| Minerva | 7B | 16.2% | 14.1% | 7.7% | - | 35.6% | - | - | - |
| Minerva | 62B | 52.4% | 27.6% | 12.0% | - | 53.9% | - | - | - |
| Minerva | 540B | 58.8% | 33.6% | 17.6% | - | 63.9% | - | - | - |
| **Open-Source Base Model** |
| Mistral | 7B | 40.3% | 14.3% | 9.2% | 71.9% | 51.1% | 44.9% | 5.1% | 23.4% |
| Llemma | 7B | 37.4% | 18.1% | 6.3% | 59.4% | 43.1% | 43.4% | 11.9% | 23.6% |
| Llemma | 34B | 54.0% | 25.3% | 10.3% | 71.9% | 52.9% | 56.1% | 11.9% | 26.2% |
| **DeepSeekMath-Base** | **7B** | **64.2%** | **36.2%** | **15.4%** | **84.4%** | **56.5%** | **71.7%** | **20.3%** | **35.3%** |

*(Catatan Dosen: Perhatikan bahwa DeepSeekMath-Base 7B (36.2% pada MATH) mengalahkan Minerva raksasa yang berukuran 540 Miliar parameter (33.6%). Ini adalah pukulan telak terhadap paradigma yang meyakini "lebih besar pasti lebih baik" di AI.)*

#### Tabel 3: Evaluasi Penyelesaian Masalah Matematis dengan Penggunaan Alat (*Tool Use*) dan Bukti Formal (*Formal Proving*)

Pada teknik *Program-of-Thought*, model diminta untuk menulis naskah bahasa pemrograman Python (memanfaatkan *library* `math` atau `sympy`) yang eksekusinya akan memberikan jawaban akhir matematis.

| Model | Size | GSM8K+Python | MATH+Python | miniF2F-valid | miniF2F-test |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Mistral | 7B | 48.5% | 18.2% | 18.9% | 18.0% |
| CodeLlama | 7B | 27.1% | 17.2% | 16.3% | 17.6% |
| CodeLlama | 34B | 52.7% | 23.5% | 18.5% | 18.0% |
| Llemma | 7B | 41.0% | 18.6% | 20.6% | 22.1% |
| Llemma | 34B | 64.6% | 26.3% | 21.0% | 21.3% |
| **DeepSeekMath-Base** | **7B** | **66.9%** | **31.4%** | **25.8%** | **24.6%** |

*(Catatan Dosen: Kolom miniF2F mengukur kapabilitas autoformalisasi bukti di atas perangkat Isabelle. DeepSeekMath mendominasi tugas ini.)*

#### Tabel 4: Evaluasi pada NLU, Penalaran, dan Kode

| Model | Size | MMLU | BBH | HumanEval | MBPP |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Mistral | 7B | **62.4%** | 55.7% | 28.0% | 41.4% |
| DeepSeek-Coder-Base-v1.5 (Tengah Pelatihan) | 7B | 42.9% | 42.9% | 40.2% | 52.6% |
| DeepSeek-Coder-Base-v1.5 (Selesai Pelatihan) | 7B | 49.1% | 55.2% | **43.2%** | **60.4%** |
| **DeepSeekMath-Base** | **7B** | 54.9% | **59.5%** | 40.9% | 52.6% |

*(Data ini membuktikan bahwa pelatihan matematika secara simultan menaikkan IQ logika bahasa umum model (MMLU naik dari 49,1% ke 54,9% dibandingkan model Coder), tanpa merusak kemampuan coding (*catastrophic forgetting*) secara radikal).*

***

## 3. Pelatihan Halus Terselia (*Supervised Fine-Tuning* / SFT)

### 3.1 Kurasi Data SFT
Peneliti menyusun dataset instruksi sebanyak **776.000 contoh**, yang terdiri dari:
*   **Inggris:** Soal GSM8K dan MATH yang dianotasi solusi terintegrasi alat (*tool-integrated*). Memasukkan MathInstruct dan Lila-OOD. Solusi berupa *Chain-of-Thought* (CoT) atau *Program-of-Thought* (PoT). Meliputi aljabar, probabilitas, teori bilangan, kalkulus, dan geometri.
*   **Mandarin:** Soal matematika K-12 (TK hingga SMA) mencakup 76 sub-topik.

### 3.2 Melatih dan Mengevaluasi DeepSeekMath-Instruct 7B
DeepSeekMath-Instruct 7B adalah hasil SFT pada model dasar menggunakan data 776K di atas. Parameter *training*: konteks maks 4K, dilatih 500 *steps*, *batch size* 256, *learning rate* konstan 5e-5.

#### Tabel 5: Kinerja Komprehensif Model Instruksi (Open vs Closed)

*(Karena kompleksitas data, saya merangkum analisis akademis dari tabel raksasa ini)*:
Tabel 5 membandingkan model dengan metode logika berantai (CoT) dan logika terintegrasi *tool* (PoT).
*   **Dalam Logika CoT:** DeepSeekMath-Instruct 7B mencapai **46,8%** pada MATH. Ini melibas Llama-2 70B, ChatGLM3, dan bahkan *Math-Shepherd-Mistral 7B*. Ini adalah titik di mana model *open-source* 7B mulai secara harfiah menghirup udara yang sama dengan GPT-4 (52,9%).
*   **Dalam Logika Tool-Integrated:** DeepSeekMath-Instruct 7B mencapai **57,4%** pada MATH. Dibantu eksekusi skrip Python, kinerja model melampaui batasan matematis bawaannya.

***

## 4. Pembelajaran Penguatan (*Reinforcement Learning* / RL)

Bagian ini adalah inti teoritis utama dari makalah ini. Mahasiswa AI wajib memahaminya.

### 4.1 Group Relative Policy Optimization (GRPO)

RL telah terbukti efektif meningkatkan kemampuan penalaran matematis setelah tahap SFT. Peneliti memperkenalkan **GRPO**, yang memangkas kebutuhan komputasi memori secara dramatis.

#### 4.1.1 Dari PPO Menuju GRPO

Untuk memahami GRPO, kita harus membedah arsitektur pendahulunya, yakni *Proximal Policy Optimization* (PPO).

### Analisis Gambar 4: Arsitektur PPO vs GRPO
*   **Deskripsi Visual:** Dua diagram alir (atas dan bawah) yang membandingkan alur kerja PPO dan GRPO dalam RL.
*   **PPO (Atas):** Terdapat pertanyaan $q$. Terdapat 4 model *neural network* yang dimuat di memori GPU: *Policy Model* (model utama yang dilatih), *Reference Model* (SFT awal sebagai jangkar), *Reward Model* (menilai jawaban), dan **Value Model** (memprediksi *baseline* nilai dari suatu state). Sinyal $r$ (dari Reward) dan $v$ (dari Value Model) dimasukkan ke blok fungsi *GAE* (*Generalized Advantage Estimation*) untuk menghasilkan *Advantage* $A$.
*   **GRPO (Bawah):** Sama-sama memiliki *Policy*, *Reference*, dan *Reward Model*. NAMUN, **tidak ada Value Model**. Sebagai gantinya, Policy model menghasilkan *Multiple Outputs* (Grup: $o_1, o_2, ..., o_G$). Reward model menilai tiap output ($r_1, r_2, ..., r_G$). Blok *Group Computation* membandingkan nilai-nilai tersebut satu sama lain untuk menghasilkan *Advantage* ($A_1, A_2, ..., A_G$).
*   **Kesimpulan Akademis:** PPO mewajibkan kita memuat *Value Model* (yang ukurannya seringkali sama dengan *Policy Model*), memakan VRAM GPU secara masif. GRPO menghilangkan *Value Model* secara elegan dengan menggunakan probabilitas statistik rata-rata grup dari respons yang dihasilkan, menghemat VRAM dan menyederhanakan *pipeline*.

**Formulasi Matematis:**

Dalam PPO, fungsi kerugian yang dioptimalkan (*surrogate objective*) adalah:
$$ \mathcal{J}_{PPO}(\theta) = \mathbb{E} \left[ q \sim P(q), o \sim \pi_{\theta_{old}}(o|q) \right] \frac{1}{|o|} \sum_{t=1}^{|o|} \min \left( \frac{\pi_\theta(o_t|q, o_{<t})}{\pi_{\theta_{old}}(o_t|q, o_{<t})} A_t, \text{clip}\left(\frac{\pi_\theta(o_t|q, o_{<t})}{\pi_{\theta_{old}}(o_t|q, o_{<t})}, 1-\epsilon, 1+\epsilon\right) A_t \right) \tag{1} $$

**Penjelasan Variabel Persamaan (1):**
*   $\mathbb{E}$: Ekspektasi matematis (nilai rata-rata harapan).
*   $q$: Pertanyaan (*question*).
*   $o$: Jawaban/Keluaran (*output*).
*   $\pi_\theta$ dan $\pi_{\theta_{old}}$: Model kebijakan (*policy model*) saat ini dan model kebijakan versi sebelumnya (sebelum pembaruan gradien). Ini menentukan probabilitas menghasilkan suatu token.
*   $\epsilon$: Hiperparameter *clipping* (pemotongan) yang menstabilkan pelatihan dengan mencegah model bergeser terlalu jauh dari kebijakan lama dalam satu langkah.
*   $A_t$: **Advantage** (Keuntungan). Apakah tindakan di titik $t$ lebih baik atau lebih buruk daripada estimasi dasar? Di PPO, ini dihitung menggunakan *Value Model* melalui metode GAE.

Untuk mencegah model memanipulasi *Reward Model* (mencari celah (*hacking*) agar nilainya tinggi tapi teksnya tidak masuk akal), PPO menambahkan penalti Divergensi KL (KL Penalty) di setiap token:
$$ r_t = r_\phi(q, o_{\leq t}) - \beta \log \frac{\pi_\theta(o_t|q, o_{<t})}{\pi_{ref}(o_t|q, o_{<t})} \tag{2} $$

**Penjelasan Variabel Persamaan (2):**
*   $r_\phi$: *Reward model*.
*   $\pi_{ref}$: *Reference model* (biasanya model SFT murni).
*   $\beta$: Koefisien penalti. Jika $\pi_\theta$ menyimpang terlalu jauh dari probabilitas asli $\pi_{ref}$, model akan dikenakan denda penalti logaritmik.

**Kelemahan PPO:** Membutuhkan pelatihan *Value Function* $V_\psi$ yang ukurannya membebani komputasi.

**Inovasi GRPO:** 
GRPO menghindari *Value Function*. Untuk setiap pertanyaan $q$, GRPO mengambil **Grup** keluaran $\{o_1, o_2, ..., o_G\}$ dari model lama. Objektif GRPO adalah memaksimalkan:

$$ \mathcal{J}_{GRPO}(\theta) = \mathbb{E} \left[ q \sim P(q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{old}}(o|q) \right] \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left\{ \min \left( \frac{\pi_\theta(o_{i,t}|q, o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t}|q, o_{i,<t})} \hat{A}_{i,t}, \text{clip}(\dots) \hat{A}_{i,t} \right) - \beta \mathbb{D}_{KL}[\pi_\theta || \pi_{ref}] \right\} \tag{3} $$

**Penjelasan Logis Persamaan (3):**
*   Struktur utamanya mirip PPO. Namun, fungsi ini dihitung atas serangkaian grup keluaran ($G$).
*   $\hat{A}_{i,t}$: **Relative Advantage**. Keuntungan tidak lagi dihitung oleh model eksternal, melainkan dihitung secara *relatif* terhadap skor respons lain dalam grup yang sama (menggunakan normalisasi Standar Deviasi). Ini selaras secara elegan karena *Reward Model* sendiri dilatih pada dataset komparasi manusia (memilih A lebih baik dari B).
*   Penalti KL tidak lagi dipotong per-token pada *reward*, melainkan digabungkan secara langsung pada kalkulasi kerugian akhir ($\mathbb{D}_{KL}$). Ini menyederhanakan perhitungan $A$.

Estimator Divergensi KL yang netral (*unbiased estimator*) pada GRPO dihitung sebagai:
$$ \mathbb{D}_{KL}[\pi_\theta || \pi_{ref}] = \frac{\pi_{ref}(o_{i,t}|q, o_{i,<t})}{\pi_\theta(o_{i,t}|q, o_{i,<t})} - \log \frac{\pi_{ref}(o_{i,t}|q, o_{i,<t})}{\pi_\theta(o_{i,t}|q, o_{i,<t})} - 1 \tag{4} $$
Persamaan ini secara matematis dijamin selalu bernilai positif.

#### 4.1.2 RL Pengawasan Hasil (*Outcome Supervision*) dengan GRPO
Dalam mode ini, grup $G$ keluaran dihasilkan, lalu diberi skor oleh *Reward Model*, menghasilkan skor $r = \{r_1, r_2, ..., r_G\}$. Skor ini dinormalisasi (dikurangi rata-rata, dibagi standar deviasi grup). Semua token dalam keluaran $o_i$ akan diberikan *Advantage* $\hat{A}_{i,t}$ yang persis sama dengan skor normalisasi akhir $\tilde{r}_i$. Ini disebut *Outcome Supervision* (hanya peduli hasil akhir benar atau salah).

#### 4.1.3 RL Pengawasan Proses (*Process Supervision*) dengan GRPO
Dalam matematika, proses bertahap sangat penting (seperti penilaian ujian esai matematika). Oleh karena itu, *Process Reward Model* (PRM) memberikan skor di akhir setiap langkah penalaran. Normalized reward $\tilde{r}_i$ dihitung per-langkah. *Advantage* $\hat{A}_{i,t}$ untuk sebuah token adalah jumlahan dari *normalized reward* untuk langkah saat ini dan semua langkah berikutnya hingga selesai. Ini memberikan umpan balik gradien yang jauh lebih detail.

#### 4.1.4 RL Iteratif dengan GRPO
Sebagai *Reward Model* statis, ia bisa usang seiring semakin pintarnya *Policy Model*. Algoritma 1 di makalah (meski tidak kita cantumkan sintaksnya) mengizinkan sistem untuk men-*generate* set pelatihan baru dari tebakan *Policy Model*, kemudian memperbarui *Reward Model* dengan mencampurkan 10% data historis (mencegah *catastrophic forgetting*), membentuk siklus *Replay Mechanism*.

### 4.2 Melatih dan Mengevaluasi DeepSeekMath-RL

DeepSeekMath-RL 7B dilatih dari basis Instruct 7B, menggunakan soal GSM8K dan MATH (144.0K soal). Parameter RL GRPO: $learning\_rate = 1e-6$, koefisien KL $\beta = 0.04$, *sample outputs per prompt* $G = 64$.
Berdasarkan Tabel 5 (yang disinggung sebelumnya), varian RL mencapai **88,2%** pada GSM8K dan **51,7%** pada MATH tingkat kompetisi. Kemampuan generalisasi GRPO sangat brilian, model ini meningkat pesat pada tolok ukur di luar pelatihan (seperti CMATH, naik ke **88,8%**).

***

## 5. Diskusi

### 5.1 Pelajaran dari Prapelatihan

#### 5.1.1 Pelatihan Kode Menguntungkan Penalaran Matematis
Untuk menguji hipotesis bahwa *coding* mempertajam logika AI, para peneliti merancang dua eksperimen pelatihan silang pada model 1.3B.

#### Tabel 6 & 7: Interaksi antara Pelatihan Kode dan Matematika

| Training Setting | Training Tokens (General/Code/Math) | GSM8K (w/o Tool) | MATH (w/o Tool) | GSM8K+Py (w/ Tool) | MATH+Py (w/ Tool) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Two-Stage** |
| 1: Gen -> 2: Math | 400B Gen, 150B Math | 19.1% | 14.4% | 14.3% | 6.7% |
| 1: Code -> 2: Math | 400B Code, 150B Math | **21.9%** | **15.3%** | 17.4% | 9.4% |
| **One-Stage** |
| Code & Math Mixed | 400B Code, 150B Math | 17.6% | 12.1% | **19.7%** | **13.5%** |

*   **Kesimpulan Akademis:**
    *   **Dua Tahap (*Two-Stage*):** Melatih pada kode *terlebih dahulu*, baru matematika (baris 2), memberikan akurasi jauh lebih tinggi daripada melatih teks umum lalu matematika (baris 1). Mengonfirmasi bahwa melatih model menjadi "programmer" meningkatkan abstraksi logika matematisnya.
    *   **Satu Tahap (*One-Stage*):** Mencampur langsung 400B token kode dan 150B matematika dalam *satu tungku* menurunkan kemampuan matematika murni (w/o Tool), NAMUN sangat memaksimalkan kemampuan matematika terintegrasi program Python (w/ Tool). AI berukuran 1.3B mengalami semacam "kebingungan" penyerapan kognitif jika dijejali dua keahlian abstrak secara simultan pada satu tahap, kecuali jika ia diminta memecahkan masalah matematika menggunakan koding.

#### 5.1.2 Makalah ArXiv Tampaknya Tidak Efektif
Banyak model raksasa (Minerva, Llemma) bergantung pada kumpulan *paper* ArXiv. Eksperimen di Tabel 8 & 9 (pelatihan dengan ArXiv-RedPajama dan MathPile) membuktikan bahwa menjejali model dengan makalah ilmiah mentah ArXiv **tidak memberikan lonjakan performa** yang substansial pada pemecahan soal GSM8K atau MATH, bahkan di beberapa titik merusak kemampuan (*deterioration*). Para ahli menyimpulkan bahwa web tutorial matematika (*Common Crawl* web biasa) lebih pedagogis bagi jaringan saraf buatan daripada bahasa akademis tinggi ArXiv.

### 5.2 Wawasan tentang Pembelajaran Penguatan (RL)

#### 5.2.1 Menuju Paradigma Terpadu (*Unified Paradigm*)
Para peneliti menyatukan fungsi matematika dari berbagai algoritma RL populer ke dalam **satu persamaan universal**:

$$ \nabla_\theta \mathcal{J}_{\mathcal{A}}(\theta) = \mathbb{E} \left[ (q, o) \sim \mathcal{D} \right] \left( \frac{1}{|o|} \sum_{t=1}^{|o|} \underbrace{GC_{\mathcal{A}}(q, o, t, \pi_{rf})}_{\text{Gradient Coefficient}} \nabla_\theta \log \pi_\theta(o_t | q, o_{<t}) \right) \tag{5} $$

**Bedah Anatomi Persamaan (5):**
Pembaruan parameter/gradien model dipengaruhi oleh Tiga Pilar Utama:
1.  $\mathcal{D}$ **(Data Source):** Menentukan data pelatihan apa yang dicuplik (Dari *offline dataset* atau eksplorasi model *online*).
2.  $\pi_{rf}$ **(Reward Function):** Fungsi atau Model yang menyuplai sinyal pujian/hukuman.
3.  $GC_{\mathcal{A}}$ **(Gradient Coefficient):** Manipulator matriks turunan yang mendikte seberapa ekstrem kekuatan dorongan (*reinforcement*) atau hukuman (*penalty*) pada data tertentu.

Tabel 10 (*tidak ditampilkan secara fisik, namun dijelaskan*) memetakan paradigma ini:
*   **SFT:** Sumber data statis. Reward tak ada. GC = 1 statis.
*   **RFT:** Sumber data dari SFT yang diproses. Reward berupa aturan (Benar/Salah). GC = 1 jika benar, 0 jika salah.
*   **DPO:** GC = diatur lewat preferensi marjinal (rumus khusus).
*   **PPO:** Sumber data dari Model AI Real-time (*Online*). Reward dari *Reward Model*. GC = menggunakan fungsi *Advantage* $A_t$ berbasis *Value Model*.
*   **GRPO (Kita):** Sumber data dari Model AI Real-time (*Online*). Reward dari *Reward Model*. GC = menggunakan *Relative Group Advantage* $\hat{A}_{i,t}$.

### Analisis Gambar 5 dan 6: Kinerja Berbagai Algoritma RL
*   **Gambar 5 (Kinerja Metode Berbeda):** Plot garis menunjukkan peningkatan langkah demi langkah dari RFT (Ungu), Online RFT (Hijau), GRPO+OS (Kuning), dan GRPO+PS (Biru).
    *   *Tren:* GRPO+PS secara konsisten menembus batas atas tertinggi pada GSM8K dan MATH, mengalahkan teknik statis seperti RFT konvensional. *Online Sampling* (hijau dan biru) menang mutlak terhadap *offline sampling* di fase akhir karena model mampu mengoreksi ilusinya sendiri secara real-time.
*   **Gambar 6 (Iterasi RL):**
    *   *Tren:* Grafik menunjukkan pergantian epoch iterasi GRPO (Iterasi-0 Ungu, Iterasi-1 Kuning, Iterasi-2 Hijau). Terjadi lonjakan sangat besar saat memasuki Iterasi-1, membuktikan pembaruan *Reward Model* sangat esensial mencegah model memanipulasi *reward* usang.

#### 5.2.2 Mengapa RL Bekerja dengan Sangat Baik?
Mengapa RL tiba-tiba membuat model pintar matematika? Peneliti mengevaluasi metrik *Pass@K* (Minimal 1 benar dari $K$ percobaan) dan *Maj@K* (*Majority Voting*, jawaban terbanyak yang sama dari $K$ percobaan).

### Analisis Gambar 7: Maj@K vs Pass@K
*   **Sumbu X:** Jumlah Kandidat Jawaban ($K$) logaritmik (1 hingga 64).
*   **Sumbu Y:** Akurasi dalam % (*Acc %*).
*   **Grafik:** Kurva warna Hijau (Pass@K Instruct) dan Biru (Pass@K RL) menempel tumpang tindih. Artinya, batas absolut jenius model (kemampuan menebak benar 1 dari 64 kali) **TIDAK berubah**.
    Namun, kurva Kuning (Maj@K RL) berada jauh di atas kurva Ungu (Maj@K Instruct).
*   **Kesimpulan Ilmiah:** RL pada AI sesungguhnya **TIDAK menyuntikkan kecerdasan matematis dasar yang baru**. Jika model sebelumnya benar-benar tidak bisa memecahkan soal X, RL tidak akan membuatnya tiba-tiba bisa. Yang dilakukan RL adalah mengubah *distribusi keluaran probabilitas (Output Distribution Robustness)*. RL merapikan "kabel saraf" AI sehingga ia lebih yakin dan lebih sering memuntahkan jawaban benarnya ke urutan teratas (Top-K) saat di-*generate*, menghilangkan sifat "ragu-ragu" yang sering dimiliki SFT. Ini adalah masalah *misalignment* (ketidakselarasan), dan preferensi RL meresolusi hal itu secara absolut.

#### 5.2.3 Bagaimana Mencapai RL yang Lebih Efektif?
Arah masa depan bagi komunitas riset:
1.  **Sumber Data:** Menggunakan pencarian berbasis pohon (*Tree-search methods*) untuk mengeksplorasi langkah-langkah logika alih-alih metode *sampling* biasa (*nucleus sampling*). Serta menggunakan teknik inferensi spekulatif efisien (*speculative decoding*).
2.  **Algoritma:** Menemukan arsitektur yang tahan terhadap *noise* di *Reward Model*. Dalam dataset manusia pun ada 20% salah anotasi. Diperlukan algoritma *WEAK-TO-STRONG alignment* untuk mengatasi batasan ini.
3.  **Fungsi Reward:** Membangun *Reward Model* yang bisa menjustifikasi hal-hal *out-of-distribution* dan mampu membangun *High-quality Process Reward Models* (PRM) yang menilai setiap langkah, bukan sekadar jawaban akhir.

***

## 6. Kesimpulan, Keterbatasan, dan Pekerjaan Masa Depan

Dalam bab ini, disajikan penemuan terobosan dari **DeepSeekMath**, LLM yang meruntuhkan hegemoni performa semua model sumber terbuka pada tolok ukur MATH tingkat persaingan, dan menyamai kinerja model tertutup raksasa dunia. 

DeepSeekMath diinisialisasi melalui DeepSeek-Coder-v1.5 7B dan mengalami perpanjangan prapelatihan 500B token yang melibatkan mahakarya koleksi *DeepSeekMath Corpus* berisi 120B token matematika dari web. Studi ablasi komprehensif mendemonstrasikan fenomena tak terduga: *Common Crawl* web biasa menaungi potensi data matematika jauh melampaui kumpulan makalah ilmiah mentah *arXiv*. 

Kontribusi sentral algoritmik dari riset ini adalah **Group Relative Policy Optimization (GRPO)**, evolusi efisien dari PPO. GRPO menghemat komputasi dan memori dengan mengabaikan fungsi *Value Model* eksternal dan memanfaatkan normalisasi relatif terhadap *group score*. Validasi eksperimental membuktikan bahwa meskipun model Instruct 7B sudah mencapai taraf jenius, campur tangan GRPO tetap bisa meroketkan akurasinya ke level yang sebelumnya mustahil di kelas parameter tersebut. Kerangka matematis Paradigma Terpadu (*Unified Paradigm*) yang ditawarkan di bab ini akan menjadi kompas arah penelitian RL di masa depan.

**Keterbatasan (Limitation):** 
Meskipun mendominasi kalkulasi rasional dan aritmatika, DeepSeekMath menderita kelemahan spasial kognitif di cabang **Geometri** dan **Pembuktian Teorema** jika disandingkan dengan model tertutup kelas dewa. Contohnya, pada mode *dry run*, AI tersebut mengalami kegagalan penalaran saat memanipulasi entitas visual-deskriptif seperti segitiga atau elips. Hal ini menunjukkan indikasi kuat adanya *data selection bias* (bias seleksi data) dalam pembentukan dataset, di mana data geometri kurang terserap. Selanjutnya, terkendala defisit parameter (hanya 7B), model ini juga tertinggal dari GPT-4 dalam uji coba *Few-shot capability* (kemampuan belajar dari beberapa contoh instan yang diberikan pada prom). 

Pekerjaan di masa mendatang akan difokuskan untuk memperbaiki alur seleksi prapelatihan *pipeline*, meningkatkan varians kualitas, dan mendalami eksplorasi algoritma penyelarasan RL baru demi melahirkan generasi komputasional yang menguasai matematika secara holistik.