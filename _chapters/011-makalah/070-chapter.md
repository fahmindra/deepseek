---
slug: ch-11-07
title: "DeepSeek-Prover"
layout: chapter
---

## DeepSeek-Prover: Memajukan Pembuktian Teorema dalam LLM Melalui Data Sintetis Skala Besar

Asisten pembukti (*proof assistants*) seperti **Lean** telah merevolusi verifikasi pembuktian matematis, memastikan tingkat akurasi dan keandalan yang tinggi. Meskipun Model Bahasa Besar (*Large Language Models* / LLM) menunjukkan potensi yang menjanjikan dalam penalaran matematis, kemajuan mereka dalam pembuktian teorema formal (*formal theorem proving*) masih sangat terhambat oleh kurangnya data pelatihan.

Untuk mengatasi masalah krusial ini, kami memperkenalkan sebuah pendekatan untuk menghasilkan data pembuktian Lean 4 dalam jumlah yang ekstensif, yang diturunkan dari soal-soal kompetisi matematika tingkat sekolah menengah atas hingga sarjana. Pendekatan ini melibatkan penerjemahan masalah dari bahasa alami ke dalam pernyataan formal (*formal statements*), menyaring pernyataan berkualitas rendah, dan menghasilkan bukti-bukti untuk menciptakan **data sintetis**. 

Setelah melakukan Pelatihan Halus (*fine-tuning*) pada model **DeepSeekMath 7B** menggunakan kumpulan data sintetis ini—yang mencakup **8 juta pernyataan formal beserta buktinya**—model kami mencapai akurasi pembuatan bukti utuh (*whole-proof generation accuracies*) sebesar **46,3%** dengan 64 sampel, dan **52%** secara kumulatif pada tes Lean 4 miniF2F. Angka ini secara meyakinkan melampaui garis dasar (*baseline*) GPT-4 yang berada di angka 23,0% (dengan 64 sampel) dan metode pencarian pohon berbasis pembelajaran penguatan (*tree search reinforcement learning*) yang berada di angka 41,0%.

Sebagai tambahan yang mengesankan, model kami berhasil membuktikan **5 dari 148 masalah** dalam tolok ukur *Lean 4 Formalized International Mathematical Olympiad* (FIMO), sementara GPT-4 gagal membuktikan satu pun. Hasil-hasil ini secara empiris mendemonstrasikan potensi besar pemanfaatan data sintetis berskala besar untuk meningkatkan kemampuan pembuktian teorema pada LLM. Baik kumpulan data sintetis maupun modelnya akan disediakan secara terbuka untuk memfasilitasi penelitian lebih lanjut di bidang yang menjanjikan ini.

---

## 1. Pendahuluan

Dalam matematika modern, kompleksitas pembuktian yang terus meningkat menghadirkan tantangan substansial bagi proses tinjauan sejawat (*peer review*). Kompleksitas ini tak jarang menyebabkan diterimanya bukti-bukti yang keliru, dengan kelemahan-kelemahan fatal yang sering kali baru terdeteksi setelah waktu yang cukup lama. 

Untuk mengatasi masalah pelik ini, bahasa matematika formal (*formal mathematical languages*) seperti **Lean** [De Moura et al., 2015, Moura and Ullrich, 2021], **Isabelle** [Paulson, 1994], dan **Coq** [The Coq Development Team] telah dikembangkan. Bahasa-bahasa ini memungkinkan penciptaan bukti yang dapat diverifikasi oleh komputer (*computer-verifiable proofs*) [Avigad, 2023]. Namun, merumuskan bukti formal menuntut usaha yang signifikan, keahlian khusus yang mendalam, dan bahkan menghadirkan tantangan tersendiri bagi matematikawan berpengalaman. Akibatnya, urgensi dan signifikansi dari **pembuktian teorema otomatis (*automated theorem proving*)** terus meningkat [Shulman, 2024].

Untuk mengurangi beban kerja dalam menulis bukti matematika formal, beberapa pendekatan [Polu and Sutskever, 2020, Jiang et al., 2021, Han et al., 2021, Polu et al., 2022, Lample et al., 2022, Jiang et al., 2022a, Yang et al., 2024] telah dikembangkan. Pendekatan ini utamanya berfokus pada **algoritma pencarian (*search algorithms*)** yang mengeksplorasi solusi potensial untuk teorema yang diajukan. Namun, metode-metode ini sering kali kesulitan menghadapi ruang pencarian (*search spaces*) yang terlampau luas yang dibutuhkan untuk teorema yang kompleks, menjadikannya tidak efektif untuk bukti-bukti yang lebih rumit [Loos et al., 2017].

Baru-baru ini, kemajuan dalam Model Bahasa Besar (LLM) telah memperkenalkan strategi baru, yaitu memanfaatkan model prapelatihan (*pre-trained models*) untuk memandu proses pencarian. Meskipun metode-metode baru ini [Jiang et al., 2022b, Zhao et al., 2023, Xin et al., 2023] mewakili peningkatan yang signifikan, mereka masih belum memenuhi tingkat penerapan praktis (*practical applicability*) karena **kurangnya korpus paralel (*parallel corpus*)**. Berbeda dengan bahasa pemrograman konvensional seperti Python atau Java, bahasa pembuktian formal digunakan oleh matematikawan yang jumlahnya relatif sedikit, sehingga menghasilkan kumpulan data yang terbatas. 

Kemajuan terbaru dalam **autoformalisasi (*autoformalization*)** [Wu et al., 2022] memang memungkinkan lebih banyak data yang selaras untuk disintesis guna melatih pembukti teorema otomatis berbasis LLM. Kendati demikian, kumpulan data yang dihasilkan masih terlalu kecil untuk sepenuhnya melepaskan kemampuan LLM.

Untuk mengatasi masalah krusial ini, kami mengusulkan sebuah metode untuk menghasilkan data pembuktian Lean 4 secara ekstensif dari soal-soal matematika informal. Pendekatan kami menerjemahkan soal kompetisi matematika tingkat sekolah menengah dan sarjana menjadi pernyataan formal (*formal statements*). Kami kemudian mengotomatiskan pembuatan bukti menggunakan LLM dan memverifikasi kebenaran bukti-bukti tersebut dalam lingkungan Lean 4. Tantangan utama dari metode ini adalah memastikan **skala (*scale*)** sekaligus **kualitas (*quality*)** dari data sintetis tersebut.

*   **Jaminan Kualitas (*Quality Assurance*):** Kami meningkatkan kualitas bukti yang dihasilkan melalui proses multi-langkah. Pertama, kami menyaring pernyataan yang terlalu sederhana menggunakan model penilaian kualitas dan mengecualikan pernyataan yang tidak valid melalui strategi penolakan hipotesis (*hypothesis rejection*). Kerangka kerja iteratif novel kami kemudian meningkatkan kualitas bukti dengan cara: awalnya menghasilkan pernyataan sintetis dari masalah matematika informal menggunakan LLM yang kurang terlatih (*under-trained LLM*) yang di-*fine-tune* pada data terbatas. Pernyataan-pernyataan ini digunakan untuk menghasilkan bukti yang sesuai, yang kemudian divalidasi kebenarannya menggunakan *verifier* (pemverifikasi) Lean 4. Pasangan teorema-bukti yang benar selanjutnya digunakan untuk melatih kembali model awal. Melalui beberapa iterasi, model yang dilatih pada data sintetis skala besar menjadi jauh lebih kuat dibandingkan LLM yang kurang terlatih pada awalnya, menghasilkan pasangan teorema-bukti dengan kualitas yang lebih tinggi.
*   **Jaminan Skala (*Scale Assurance*):** Untuk mempercepat proses pembuatan bukti, metode kami mengatasi tantangan ruang pencarian bukti yang luas. Penyebab utama penundaan adalah proses pada pernyataan-pernyataan yang tidak dapat dibuktikan (*unprovable statements*) yang terus diproses hingga mencapai batas waktu (*time limit*). Untuk memitigasi hal ini, kami mengusulkan untuk **membuktikan pernyataan yang dinegasikan (*negated statements*) secara paralel**. Begitu pernyataan asli atau negasinya terbukti, seluruh proses pembuktian akan segera dihentikan.

Kami menilai efektivitas metode kami dalam pembuktian teorema Lean 4 menggunakan 488 soal dari miniF2F [Zheng et al., 2021] dan 148 soal dari tolok ukur FIMO [Liu et al., 2023]. Kami menggunakan **DeepSeekMath 7B** [Shao et al., 2024], sebuah model matematika yang mutakhir (*state-of-the-art*), sebagai dasar (*base model*) kami. Hasilnya menunjukkan bahwa model kami yang dilatih secara iteratif berkinerja sangat kuat, mencapai akurasi **46,3%** dalam pembuatan bukti utuh pada tolok ukur *miniF2F-test* dengan 64 sampel, melampaui GPT-4 [Achiam et al., 2023] di angka **23,0%** dan metode *reinforcement learning* (pencarian pohon) di angka **41,0%**. Selain itu, pendekatan kami berhasil menyelesaikan **4 dari 148 masalah** dalam tolok ukur FIMO dengan 100 sampel, sementara GPT-4 sama sekali tidak menyelesaikan satu pun, dan pendekatan kami menyelesaikan **5** masalah dengan 4096 sampel. 

Eksperimen ablasi menunjukkan bahwa model secara progresif menyelesaikan lebih banyak masalah di miniF2F pada setiap iterasinya. Singkatnya, makalah kami memberikan kontribusi sebagai berikut:

*   Kami memperkenalkan sebuah **metode iteratif** untuk mensintesis 8 juta pernyataan formal, yang masing-masing disertai dengan bukti formal, yang diturunkan dari masalah matematika informal. Hasil eksperimen mendemonstrasikan bahwa metode ini secara signifikan meningkatkan skalabilitas dan kualitas data sintetis.
*   Model kami, yang dilatih pada kumpulan data sintetis ini, mencapai kinerja mutakhir (*state-of-the-art*) di berbagai tolok ukur, dengan akurasi pembuatan bukti utuh (*whole-proof*) sebesar 46,3% menggunakan 64 sampel dan 52% secara kumulatif pada tes Lean 4 miniF2F. Ini melampaui garis dasar GPT-4 pada 23,0% dengan 64 sampel dan metode *tree search reinforcement learning* pada 41,0%. Selain itu, model kami berhasil membuktikan 5 dari 148 masalah di tolok ukur *Lean 4 Formalized International Mathematical Olympiad* (FIMO), sementara GPT-4 gagal membuktikan apa pun.
*   Kami berkontribusi pada komunitas matematika dan AI dengan **membuat dan merilis secara terbuka (*open-sourcing*)** kumpulan data besar berisi bukti matematika formal berkualitas tinggi, sehingga mendorong penelitian dan pengembangan lebih lanjut dalam pembuktian teorema otomatis.

---

## 2. Latar Belakang dan Pekerjaan Terkait (*Background and Related Works*)

Pembuktian teorema otomatis (*Automated Theorem Proving* / ATP) telah menjadi area minat yang signifikan dalam penelitian kecerdasan buatan sejak awal kemunculannya [Bibel, 2013]. Upaya awal diarahkan pada kerangka logika yang lebih sederhana, yang mengarah pada pengembangan pembukti teorema orde pertama (*first-order*) yang sangat efisien seperti E [Schulz, 2002] dan Vampire [Kovács and Voronkov, 2013]. Meskipun demikian, alat-alat ini sering kali gagal dalam menangani teorema kompleks yang umumnya ditemukan pada asisten pembukti modern seperti Lean [De Moura et al., 2015], Isabelle [Paulson, 1994], dan Coq [The Coq Development Team]. 

Munculnya model pembelajaran mendalam (*deep learning*) dan teknik pencarian yang dipandu model telah menghidupkan kembali bidang ini [Bansal et al., 2019]. Pendekatan modern ini tidak hanya meningkatkan kapabilitas sistem ATP tetapi juga memperluas penerapannya dalam memecahkan masalah matematika yang lebih rumit.

**ATP dengan Model Neural (*ATP with Neural Models*).** Dengan perkembangan pembelajaran mendalam, beberapa pendekatan telah diusulkan untuk menggabungkan model neural dengan ATP [Loos et al., 2017]. Serangkaian pendekatan ATP mengadopsi algoritma pencarian pohon yang dipandu oleh model neural [Polu and Sutskever, 2020, Han et al., 2021, Polu et al., 2022, Jiang et al., 2022a, Yang et al., 2024]. Pendekatan-pendekatan ini terutama memanfaatkan teknik pembelajaran penguatan (*reinforcement learning*) untuk meningkatkan akurasi model [Kaliszyk et al., 2018, Crouse et al., 2021, Wu et al., 2021, Lample et al., 2022]. Karena ruang pencariannya sangat besar, proses pencarian memakan waktu dan sumber daya komputasi yang besar.

Serangkaian pendekatan ATP lainnya memanfaatkan kekuatan Model Bahasa Besar (LLM). Pendekatan ini biasanya melibatkan model bahasa yang di-*fine-tune* dengan data pembuktian *open-source* dan berinteraksi dengan pemverifikasi melalui program transisi *state-action* [Polu and Sutskever, 2020, Jiang et al., 2021, Han et al., 2021, Polu et al., 2022, Lample et al., 2022, Jiang et al., 2022a, Yang et al., 2024]. Proses ini secara iteratif menghasilkan langkah-langkah pembuktian dan memverifikasi kebenarannya dengan pemverifikasi formal. Ia kemudian menghasilkan langkah pembuktian berikutnya berdasarkan status pembuktian (*proof states*) yang dikembalikan oleh pemverifikasi formal. Meskipun pendekatan ini mencapai kinerja tinggi, mereka secara komputasi sangat intensif. Untuk meningkatkan efisiensi, penelitian terbaru memanfaatkan LLM untuk menghasilkan bukti formal secara lengkap dan langsung [First et al., 2023, Jiang et al., 2022b, Zhao et al., 2023, Xin et al., 2023], sehingga melewati interaksi iteratif selama pembuatan bukti.

**Autoformalisasi untuk Matematika Formal (*Autoformalization for Formal Mathematics*).** Karena ketersediaan korpus formal yang terbatas untuk pelatihan, kinerja LLM saat ini juga dibatasi. Oleh karena itu, beberapa pendekatan mengusulkan autoformalisasi [Wu et al., 2022, Jiang et al., 2022b], yang melibatkan konversi deskripsi bahasa alami ke dalam pernyataan formal yang dapat diverifikasi oleh asisten pembuktian. Beberapa penelitian telah menghasilkan kumpulan data sintetis dari bukti formal menggunakan transformasi berbasis aturan (*rule-based transformations*) dari teorema yang sudah ada [Wu et al., 2020, Wang and Deng, 2020, Xiong et al., 2023]. Meskipun efektif, metode ini dibatasi oleh ketergantungan mereka pada aturan yang telah ditentukan dan kurangnya fleksibilitas untuk aplikasi yang lebih luas. 

Metodologi terbaru mengadopsi LLM untuk menerjemahkan masalah bahasa alami ke dalam pernyataan formal [Huang et al., 2024]. Namun, dataset ini tetap lebih kecil dari yang dibutuhkan dan terbatas pada tolok ukur matematika kecil, yang hanya menghasilkan perbaikan kecil dalam hasil pelatihan untuk model bahasa. Dalam makalah ini, kami bertujuan untuk mensintesis bukti formal melalui autoformalisasi pada skala yang jauh lebih besar untuk mendongkrak kinerja *neural prover*.

---

## 3. Pendekatan (*Approach*)

Pada bagian ini, kita akan membedah secara rinci pendekatan metodologis yang diajukan dalam penelitian ini. Pendekatan ini terdiri dari empat proses kunci. Untuk memahaminya secara utuh, mari kita bedah **Gambar 1**.

### Analisis Akademik Gambar 1: Tinjauan Pendekatan (*Overview of our approach*)

*   **Deskripsi Visual:** Gambar 1 adalah diagram alir (*flowchart*) siklikal (berbentuk putaran) yang mengilustrasikan arsitektur sistem iteratif dari DeepSeek-Prover. Alur ini bergerak dari kiri ke kanan, lalu kembali ke kiri, membentuk sebuah *loop* peningkatan berkelanjutan.
*   **Komponen Utama dan Alur Kerja:**
    *   **Langkah 1: Autoformalization (Autoformalisasi):**
        *   Proses dimulai dari *Informal Math Problems* (Masalah Matematika Informal, disimbolkan dengan dokumen berisikan rumus).
        *   Masalah ini dimasukkan ke dalam modul **DS-Prover** (DeepSeek-Prover, disimbolkan dengan jaringan saraf/node).
        *   Outputnya adalah *Formal Math Statements* (Pernyataan Matematika Formal, disimbolkan dengan dokumen dengan ikon panah/koding).
    *   **Langkah 2: Model Scoring and Hypothesis Rejection (Penilaian Model dan Penolakan Hipotesis):**
        *   Pernyataan formal yang dihasilkan dievaluasi kembali oleh modul **DS-Prover**.
        *   Tahap ini bertindak sebagai filter kualitas untuk membuang pernyataan yang buruk atau salah secara logika.
        *   Outputnya adalah *High-Quality Formal Math Statements* (Pernyataan Matematika Formal Kualitas Tinggi).
    *   **Langkah 3: Statements Proving (Pembuktian Pernyataan):**
        *   Pernyataan berkualitas tinggi dimasukkan ke dalam sub-sistem pembuktian yang kompleks (kotak putus-putus berlabel "Statements Proving").
        *   Di dalam kotak ini, modul **DS-Prover** diinstruksikan untuk melakukan dua tugas secara paralel: *Proving Original Statements* (Membuktikan Pernyataan Asli) dan *Proving Negated Statements* (Membuktikan Pernyataan yang Dinegasikan/Dibalik).
        *   Hasil dari kedua pembuktian ini dikirim ke **Formal Verifier** (Pemverifikasi Formal, disimbolkan dengan logo LEAN 4 dan ikon "STOP" / Tanda Centang Hijau). Jika terbukti valid, ia akan menghasilkan *Formal Statements with Correct Proofs* (Pernyataan Formal dengan Bukti yang Benar).
    *   **Langkah 4: Fine-tuning Prover (Pelatihan Halus Pembukti):**
        *   *Formal Statements with Correct Proofs* dikumpulkan menjadi **Synthesized Data** (Data yang Disintesis, disimbolkan dengan dokumen +).
        *   Data baru ini diumpankan kembali (panah putar balik) ke modul **DS-Prover** awal untuk melakukan proses *Fine-tuning*.
    *   **Langkah 5: Repeat (Ulangi):**
        *   DS-Prover yang kini sudah lebih pintar akan mengulangi siklus ini kembali dari Langkah 1.
*   **Kesimpulan Akademis:** Diagram ini mengilustrasikan paradigma *self-improvement* (peningkatan mandiri) yang elegan. Alih-alih bergantung pada anotasi manusia yang mahal dan lambat, sistem menggunakan modelnya sendiri untuk menerjemahkan (Langkah 1), mengevaluasi (Langkah 2), dan membuktikan (Langkah 3). Penggunaan verifikator eksternal yang absolut (Lean 4) memastikan tidak ada *hallucination* (halusinasi) yang masuk ke dalam data pelatihan (Langkah 4). Melalui siklus iteratif ini, kecerdasan model didongkrak secara eksponensial.

Mari kita bahas setiap langkah tersebut secara lebih rinci di bawah ini.

### 3.1 Autoformalisasi (*Autoformalization*)

Pembuatan data pembuktian formal secara fundamental bergantung pada ketersediaan korpus pernyataan formal yang substansial. Namun dalam praktiknya, mengumpulkan koleksi pernyataan formal yang dibuat secara manual (*manually crafted*) sangatlah menantang. Untungnya, internet dipenuhi dengan masalah matematika yang diekspresikan dalam bahasa alami (*natural language*). Dengan meng-autoformalisasi masalah informal ini, kita dapat menghasilkan repositori pernyataan formal yang sangat besar.

Peneliti mengamati bahwa masalah dengan kondisi eksplisit dan tujuan yang terdefinisi dengan baik biasanya lebih mudah diformalkan dibandingkan dengan topik matematika lanjutan yang membutuhkan definisi dan konstruksi yang rumit. Akibatnya, makalah ini utamanya mengkaji masalah kompetisi tingkat sekolah menengah dan sarjana, dengan penekanan khusus pada **aljabar** dan **teori bilangan** (*number theory*), serta dalam porsi yang lebih kecil pada kombinatorika, geometri, dan statistika. 

Meskipun terlihat sederhana, masalah-masalah ini sering kali melibatkan teknik penyelesaian yang kompleks, menjadikannya kandidat yang sangat baik untuk menyusun data pembuktian. Untuk menyusun dataset, peneliti menggunakan *web scraping* (pengikisan web) dan teknik pembersihan data yang hati-hati untuk mengekstrak masalah dari sumber online yang menampilkan latihan, ujian, dan kompetisi, menghasilkan dataset sebanyak **869.659** masalah matematika bahasa alami berkualitas tinggi.

Secara spesifik, DeepSeek-Prover diinisialisasi menggunakan model **DeepSeekMath-Base 7B** [Shao et al., 2024]. Awalnya, model kesulitan mengubah masalah matematika informal menjadi pernyataan formal. Untuk mengatasinya, model DeepSeek-Prover di-*fine-tune* menggunakan dataset MMA [Jiang et al., 2023], yang berisi pernyataan formal dari `mathlib` Lean 4 yang telah diterjemahkan kembali (*back-translated*) ke deskripsi masalah bahasa alami oleh GPT-4. Model kemudian diinstruksikan untuk menerjemahkan masalah bahasa alami ini ke dalam pernyataan formal di Lean 4 menggunakan pendekatan terstruktur.

Berikut adalah format instruksi (Prompt) yang digunakan:

```markdown
Prompt:
Mathematical Problem in Natural Language:
{$informal_statement_with_answers}
Translate the problem to Lean 4 (only the core declaration):
```lean4
Response:
{$formal_statement}
```
```

**Penjelasan Prompt:**
*   `{$informal_statement_with_answers}`: Placeholder (tempat penampung) untuk variabel yang berisi teks masalah matematika asli dalam bahasa manusia beserta jawabannya.
*   Instruksi `Translate the problem to Lean 4 (only the core declaration):` secara eksplisit memerintahkan LLM untuk menuliskan kode Lean 4, namun dibatasi hanya pada "deklarasi intinya" (*core declaration*) saja, tanpa harus memberikan bukti lengkapnya pada tahap ini.
*   ```lean4 ... ``` : Memaksa model untuk mengeluarkan output dalam format blok kode (*code block*) dengan *syntax highlighting* Lean 4.
*   `{$formal_statement}`: Hasil keluaran dari model, yaitu translasi matematis murni dari soal tersebut.

### 3.2 Pemfilteran Kualitas (*Quality Filtering*)

Kualitas dari pernyataan yang diautoformalisasi pada awalnya ditemukan kurang optimal karena dua masalah utama:

**Masalah Pertama: Terlalu Sederhana.**
Banyak pernyataan formal yang dihasilkan terlalu simplistik. Untuk mengatasi hal ini, peneliti mengembangkan kriteria penilaian (*scoring criteria*) dan memberikan contoh dari `miniF2F-valid` sebagai panduan pembelajaran *few-shot* (beberapa contoh) untuk memandu model DeepSeek-Prover dalam mengevaluasi konten dan kualitas pernyataan tersebut menggunakan pendekatan *chain-of-thought* (pemikiran berantai). 

Tinjauan manual terhadap skor-skor ini mengkonfirmasi bahwa evaluasi model sangat cocok dengan intuisi dan harapan manusia. Model diinstruksikan untuk mengklasifikasikan kualitas setiap pernyataan formal ke dalam kategori: "sangat baik" (*excellent*), "baik" (*good*), "di atas rata-rata" (*above average*), "cukup" (*fair*), atau "buruk" (*poor*). Pernyataan yang dinilai sebagai "cukup" atau "buruk" selanjutnya akan **dibuang** (*excluded*).

**Masalah Kedua: Hipotesis Tidak Konsisten (Vakuum).**
Masalah kedua berkaitan dengan pernyataan formal yang, meskipun dapat dibuktikan, didasarkan pada hipotesis yang saling bertentangan (inkonsisten) yang mengarah pada kesimpulan yang hampa (*vacuous conclusions*), membuat kesimpulan tersebut menjadi tidak berarti dalam matematika.

Sebagai contoh, perhatikan pernyataan yang dihasilkan model berikut:

```lean
example (θ : ℝ) (h0 : ∀ z : ℂ, z ^ 2 = -1 ∧ z ^ 3 = -1 ∧ z ^ 6 = 1) (h1 :
  Real.tan θ = 2 * Real.sqrt 3) : θ = 5 * Real.pi / 3
```

**Penjelasan Kode:**
*   Kode di atas mencoba mendefinisikan sebuah teorema matematika.
*   Namun, perhatikan hipotesisnya: $z^2 = -1 \land z^3 = -1 \land z^6 = 1$ untuk semua bilangan kompleks $z$. Secara matematis, hal ini **jelas salah** (tidak mungkin suatu bilangan dikuadratkan hasilnya -1 sekaligus dipangkatkan tiga hasilnya juga -1).
*   Dalam logika matematika, jika premisnya salah (*False*), maka implikasi ke kesimpulan apa pun akan selalu dianggap benar (*True*) secara trivial (ini disebut kebenaran vakum / *vacuous truth*). Pembuktian seperti ini sama sekali tidak berguna.

Untuk mengeleminasi kasus-kasus seperti ini, diterapkan metode **penolakan hipotesis (*hypothesis rejection*)**. Metode ini melibatkan penggunaan DeepSeek-Prover untuk mencoba membuktikan pernyataan formal tersebut dengan menyematkan `False` sebagai kesimpulannya. Jika model berhasil membuktikan `False`, ini adalah bukti telak bahwa hipotesis awalnya tidak valid (kontradiktif), sehingga pernyataan tersebut langsung dibuang.

Contoh pembuktian *False*:

```lean
example (θ : ℝ) (h0 : ∀ z : ℂ, z ^ 2 = -1 ∧ z ^ 3 = -1 ∧ z ^ 6 = 1) (h1 :
  Real.tan θ = 2 * Real.sqrt 3) : False := by
  simpa using h0 1
```

**Penjelasan Kode:**
*   Model secara cerdas mengganti kesimpulan aslinya (tentang nilai $\theta$) menjadi `False`.
*   Taktik pembuktian `by simpa using h0 1` diaplikasikan. Ini berarti model mensubstitusikan nilai $z = 1$ ke dalam hipotesis $h0$.
*   Jika $z=1$, maka $1^2 = 1$ (bukan -1). Hipotesis langsung runtuh (menjadi *False*). Karena premis terbukti *False*, pembuktian untuk `False` menjadi sukses, yang berarti pernyataan soal aslinya cacat secara logika dan dibuang dari dataset.

Dengan menerapkan strategi ganda berupa penilaian model (*model scoring*) dan penolakan hipotesis, peneliti berhasil menyusun set yang disaring (*refined set*) sebanyak **712.073 pernyataan formal berkualitas tinggi**, memberikan fondasi yang kuat untuk sintesis bukti selanjutnya.

### 3.3 Pembuktian Pernyataan (*Statement Proving*)

Setelah membuat korpus substansial dari pernyataan formal berkualitas tinggi, model digunakan untuk **mencari bukti** dari pernyataan tersebut. 

Secara tradisional, LLM digunakan secara *brute-force* (mencoba paksa terus-menerus) untuk membuktikan teorema—mencoba berulang kali hingga bukti yang valid ditemukan atau sumber daya komputasi habis. Pendekatan ini sangat tidak efisien untuk tujuan makalah ini. 

Biasanya, LLM diterapkan pada pernyataan formal buatan manusia yang dibuat dengan cermat dan umumnya benar serta dapat dibuktikan. Namun, dalam tugas membuktikan pernyataan yang di-autoformalisasi, banyak pernyataan yang dihasilkan mungkin **salah**. Adalah tidak masuk akal mengharapkan model untuk memvalidasi proposisi yang salah di dalam sistem pembuktian yang andal mana pun. Masalah ini menjadi sangat menonjol selama autoformalisasi skala besar, di mana diamati bahwa setidaknya 20% dari pernyataan formal yang dihasilkan model (bahkan setelah pemfilteran kualitas) ternyata salah, yang akan menyebabkan pemborosan komputasi yang signifikan jika diatasi dengan *brute force*.

Untuk meminimalkan pemborosan sumber daya pada pernyataan yang tidak dapat dibuktikan (*unprovable statements*) dan meningkatkan efisiensi pencarian, peneliti memanfaatkan **simetri logis** antara sebuah pernyataan dan negasinya.

Diimplementasikan pencarian bukti ganda secara konkuren (bersamaan) untuk setiap pernyataan sintetis:
1.  Satu pencarian untuk membuktikan pernyataan asli: $\Gamma \vdash P$
2.  Pencarian lainnya untuk membuktikan negasinya (kebalikannya): $\Gamma \vdash \neg P$

**Pencarian akan langsung berhenti** segera setelah bukti valid ditemukan untuk *salah satu* dari keduanya. Hal ini secara meyakinkan membuktikan bahwa yang satunya lagi tidak dapat dibuktikan. Setiap alur pencarian bukti mencoba hingga $k$ bukti kecuali bukti valid muncul lebih awal.

Semua bukti yang tervalidasi, baik itu membuktikan teorema aslinya atau negasinya, kemudian digabungkan untuk melatih kembali (mem-*fine-tune*) DeepSeek-Prover. Pendekatan ganda ini berfungsi sebagai bentuk **augmentasi data (*data augmentation*)**, memperkaya dataset dengan proposisi dan negasinya—bahkan jika proposisi awalnya tidak diformalkan dengan benar oleh model.

### 3.4 Peningkatan Iteratif (*Iterative Enhancement*)

Karena seluruh *pipeline* sangat bergantung pada DeepSeek-Prover, meningkatkan kinerja model setelah setiap iterasi menjadi sangat krusial. Untuk mencapai hal ini, model di-*fine-tune* secara konsisten dengan data yang baru saja dihasilkan. 

Model yang telah diperbarui kemudian dimanfaatkan untuk iterasi autoformalisasi berikutnya. Wawasan utama dari proses iteratif ini adalah bahwa **model secara inkremental meningkat dalam hal kekuatan dan efektivitas** setelah setiap siklus penyempurnaan (*refinement*) dan aplikasi. Proses ini terus berlanjut hingga tidak ada lagi keuntungan yang diamati. 

Akibatnya, pasangan teorema-bukti yang dihasilkan oleh model menjadi semakin tinggi kualitasnya di setiap iterasi. Metode ini memastikan bahwa DeepSeek-Prover secara konsisten meningkatkan kemampuannya, yang pada akhirnya menghasilkan pasangan teorema-bukti superior melalui penyempurnaan yang berkelanjutan.

---

## 4. Eksperimen (*Experiments*)

### 4.1 Pengaturan Eksperimental (*Experimental Setup*)

DeepSeek-Prover dibangun di atas model **DeepSeekMath-Base 7B** [Shao et al., 2024], sebuah arsitektur transformer khusus *decoder* yang telah diprapelatihan (*pre-trained*) pada korpus yang mencakup 120 miliar token terkait matematika. 

Model ini di-*fine-tune* menggunakan pengaturan berikut:
*   *Global batch size*: 512
*   *Constant learning rate*: $1 \times 10^{-4}$
*   *Warmup steps*: 6.000 langkah menggunakan data sintetis.

Kinerja DeepSeek-Prover dievaluasi dengan membandingkannya terhadap beberapa *baseline* (garis dasar):
*   **GPT-3.5 dan GPT-4** [Achiam et al., 2023]: Model generatif mutakhir yang meskipun tidak dirancang khusus untuk pembuktian teorema, skala dan jumlah parameternya memberikan kapabilitas yang signifikan. Keduanya diinstruksikan untuk menghasilkan bukti lengkap dengan metodologi yang serupa dengan milik DeepSeek.
*   **GPT-f** [Polu and Sutskever, 2020]: Menggunakan arsitektur mirip GPT-2 dengan metode pencarian *best-first* iteratif. Metodologi ini juga digunakan oleh garis dasar lain seperti **Proof Artifact Co-Training** [Han et al., 2021], **ReProver** [Yang et al., 2024], **Llemma** [Azerbayev et al., 2023], dan **COPRA** [Thakur et al., 2023] yang menggunakan model khusus atau model umum seperti GPT-4 untuk menghasilkan *langkah-langkah pembuktian* (bukan bukti utuh sekaligus).

### 4.2 Hasil Utama (*Main Results*)

Studi ini menangani masalah matematika kompleks dalam aljabar dan teori bilangan. Evaluasi dilakukan menggunakan tolok ukur **miniF2F** [Zheng et al., 2021] dan **FIMO** [Liu et al., 2023]. Metrik yang digunakan adalah **pass@k**, yang mengindikasikan skenario di mana *setidaknya satu bukti valid berhasil ditemukan* dalam $k$ percobaan pertama yang dihasilkan oleh model.

**Hasil pada MiniF2F:** 
Tolok ukur miniF2F terdiri dari 244 masalah validasi (*miniF2F-valid*) dan 244 masalah pengujian (*miniF2F-test*), mulai dari aritmatika dasar hingga tingkat kompetisi (AIME, AMC, IMO). 

Silakan merujuk pada **Tabel 1** untuk perbandingan mendetail.

**Tabel 1: Perbandingan dengan *state-of-the-arts* pada dataset miniF2F.**

| Metode | Ukuran Model | Jumlah Percobaan (*Generation times*) | miniF2F-valid | miniF2F-test |
| :--- | :---: | :---: | :---: | :---: |
| **Metode Pencarian Pohon (*Tree Search Methods*)** | | | | |
| COPRA (GPT-4) [Thakur et al., 2023] | - | $1 \times 60$ | - | 26,6% |
| Proof Artifact Co-Training [Han et al., 2021] | 837M | $8 \times 8 \times 512$ | 29,3% | 29,2% |
| ReProver [Yang et al., 2024] | 229M | $1 \times 3751$ | - | 25,0% |
| Llemma [Azerbayev et al., 2023] | 34B | $1 \times 3200$ | - | 25,8% |
| Hypertree Proof Search [Lample et al., 2022] | 600M | kumulatif $64 \times 5000$ | 58,6% | 41,0% |
| **Metode Pembuatan Bukti Utuh (*Whole-Proof Generation Methods*)** | | | | |
| GPT-4-turbo 0409 | - | 64 | 25,4% | 23,0% |
| DeepSeekMath-Base [Shao et al., 2024] | 7B | 128 | 25,4% | 27,5% |
| **DeepSeek-Prover (Model Makalah Ini)** | **7B** | **kumulatif** | **60,2%** | **52,0%** |
| | | 1 (greedy) | - | 30,0% |
| | | 64 | - | **46,3%** |
| | | 65536 | - | 50,0% |

**Analisis Tabel 1:**
*   Tabel membandingkan dua pendekatan: Pencarian Pohon (menggenerasi baris-demi-baris) dan Pembuatan Bukti Utuh (menggenerasi seluruh blok kode pembuktian sekaligus).
*   Pada metode *Whole-Proof*, **DeepSeek-Prover (7B)** mendominasi secara absolut. Dengan hanya 64 kali percobaan generasi (*generation times*), ia mencapai akurasi **46,3%** pada *miniF2F-test*, menghancurkan GPT-4 yang hanya mencapai 23,0% pada jumlah percobaan yang sama.
*   Bahkan ketika dibandingkan dengan metode pencarian pohon terbaik yang ada (Hypertree Proof Search) yang mencapai 41,0% setelah 5000 kali iterasi komputasi yang berat, DeepSeek-Prover tetap unggul jauh dengan skor kumulatif **52,0%**.
*   Skalabilitas DeepSeek-Prover terbukti: kinerjanya meningkat seiring dengan bertambahnya jumlah tebakan (dari 30,0% pada metode *greedy* / 1 tebakan, hingga 50,0% pada 65.536 tebakan). Ini mendemonstrasikan efektivitasnya dalam menangani skenario pembuktian yang kompleks.

**Hasil pada FIMO:** 
Tolok ukur FIMO berisi 149 masalah formal yang bersumber dari *IMO shortlist* (soal-soal seleksi Olimpiade Matematika Internasional). DeepSeek-Prover **berhasil membuktikan 4 teorema** dengan 100 tebakan per teorema, sementara GPT-4 gagal membuktikan satu pun. Ketika jumlah percobaan ditingkatkan menjadi 4.096, DeepSeek-Prover berhasil membuktikan **satu teorema tambahan**. 

### 4.3 Studi Ablasi (*Ablation Studies*)

Studi ablasi dilakukan untuk mengisolasi dan membuktikan efektivitas komponen-komponen spesifik dalam arsitektur yang diajukan.

#### 4.3.1 Efektivitas Autoformalisasi Skala Besar

Untuk membuktikan apakah autoformalisasi skala besar ini benar-benar lebih baik dari dataset konvensional, dilakukan analisis komparatif (Tabel 2). 

**Tabel 2: Peningkatan *pass rates* untuk miniF2F (pass@128) pada model yang dilatih dengan berbagai data bukti formal.**

| Model / Sumber Data | #Tokens | miniF2F-valid | miniF2F-test |
| :--- | :--- | :---: | :---: |
| - (Tanpa Pelatihan Tambahan) | - | 25,4% | 27,5% |
| Mathlib (Teorema buatan manusia di Lean 4) | 0,238B | 30,3% | 31,2% |
| **Autoformalized Statements (Metode Makalah Ini)** | **3,108B** | **48,8%** | **42,6%** |

**Analisis Akademis:**
Jika model hanya dilatih menggunakan `mathlib` (kumpulan data bukti formal buatan manusia yang sangat kecil namun akurat), peningkatannya sangat marginal (hanya naik ke 31,2%). Namun, dengan menggunakan jutaan **pernyataan yang diautoformalisasi (disintesis)** oleh model sendiri, akurasi melonjak masif ke **42,6%**. Ini secara meyakinkan membuktikan bahwa kuantitas data yang besar (*large-scale*) yang dihasilkan mesin mampu mengalahkan dataset kecil buatan pakar.

#### 4.3.2 Efektivitas Penilaian Pernyataan Formal (*Formal Statements Scoring*)

Apakah sistem filter kualitas (yang membuang pernyataan "fair/poor" dan mempertahankan "excellent/good") benar-benar berfungsi?

**Tabel 3: Peningkatan *pass rates* untuk miniF2F (pass@128) pada model yang dilatih dengan skor kualitas data yang berbeda.**

| Kelas Skor (*Scored Class*) | miniF2F-valid | miniF2F-test |
| :--- | :---: | :---: |
| "excellent", "good" dan "above average" | **48,8%** | **42,6%** |
| "fair" dan "poor" | 41,4% | 38,1% |

**Analisis Akademis:**
Model yang di-*fine-tune* menggunakan data skor tinggi mengungguli model yang dilatih pada data skor rendah dengan selisih 4,5% secara absolut. Ini menegaskan utilitas (kegunaan) model LLM dalam mengevaluasi dirinya sendiri secara akurat dan menyaring *noise* / data sampah secara efektif.

#### 4.3.3 Efektivitas Peningkatan Iteratif (*Iterative Enhancement*)

Tabel 4 menunjukkan korelasi yang jelas antara jumlah iterasi sintesis data dan peningkatan kinerja pembuktian.

**Tabel 4: Peningkatan *pass rates* untuk miniF2F (pass@128) lintas iterasi pelatihan berurutan.**

| Model | miniF2F-valid | miniF2F-test |
| :--- | :---: | :---: |
| iteration 0 | 38,1% | 34,0% |
| iteration 1 | 45,1% | 39,3% |
| iteration 2 | 49,2% | 41,4% |
| iteration 3 | 54,5% | 45,1% |
| **iteration 4** | **59,4%** | **46,3%** |

**Analisis Akademis:**
Setiap kali siklus (seperti pada Gambar 1) berulang, kinerja model merangkak naik (dari 34,0% pada Iterasi 0 hingga 46,3% pada Iterasi 4). Ini membuktikan bahwa alur kerja yang dirancang dalam makalah ini tidak menemui "jalan buntu" dengan cepat, melainkan terus menyempurnakan kemampuan model dalam menangani bukti kompleks serta meningkatkan kuantitas dan kualitas data sintetis yang diproduksi.

#### 4.3.4 Efektivitas Penskalaan Data Pembuktian Sintetis

Tabel 5 menyoroti seberapa penting skala (ukuran) dataset terhadap kinerja (*scaling laws*).

**Tabel 5: Peningkatan *pass rates* untuk miniF2F (pass@128) pada model yang dilatih dengan fraksi data sintetis yang semakin membesar.**

| Ukuran (*Size*) Dataset | miniF2F-valid | miniF2F-test |
| :--- | :---: | :---: |
| 1.000 | 22,95% | 24,18% |
| 10.000 | 32,79% | 31,97% |
| 100.000 | 36,07% | 37,7% |
| 1.000.000 | 39,34% | 38,11% |
| **8.066.621** | **42,62%** | **40,16%** |

**Analisis Akademis:**
Eksperimen ini mengeksekusi pelatihan pada subset yang berukuran semakin besar (dari 1 ribu hingga 8 juta data poin). Peningkatan performa pada tolok ukur berbanding lurus (secara proporsional terhadap peningkatan eksponensial) dengan ukuran dataset. Pola ini menyoroti pentingnya dataset berskala masif untuk mendongkrak kemahiran model.

---

## 5. Studi Kasus (*Case Studies*)

Bagian ini menyajikan studi kasus untuk mendemonstrasikan aplikasi metode dalam meng-autoformalisasi teorema, menampilkan baik bukti yang sukses maupun identifikasi inkonsistensi (kesalahan logika) selama tahap Penolakan Hipotesis (*Hypothesis Rejection*).

### 5.1 Teorema yang Diautoformalisasi dengan Bukti Lengkap

**Contoh a. Masalah:** Buktikan bahwa determinan dari matriks berikut adalah nol:
$$ \begin{bmatrix} 1 & \cos(a - b) & \cos(a) \\ \cos(a - b) & 1 & \cos(b) \\ \cos(a) & \cos(b) & 1 \end{bmatrix} $$

**Teorema yang Diautoformalisasi dalam Lean (Output dari Model):**
```lean
example (a b : ℝ) :
  Matrix.det ![![1, Real.cos (a - b), Real.cos a], ![Real.cos (a - b), 1, Real.cos
  b], ![Real.cos a, Real.cos b, 1]] = 0
```
*   **Penjelasan Kode:** Pendekatan ini secara efektif dan elegan menerjemahkan ekspresi aljabar visual matriks ke dalam bahasa formal Lean. Model mendefinisikan matriks spesifik $3 \times 3$ yang bergantung pada bilangan real `a` dan `b`, dan menegaskan bahwa determinannya sama dengan nol menggunakan fungsi `Matrix.det`. Notasi `![...]` yang bersarang secara akurat merepresentasikan baris-baris matriks (seperti format *list of lists*).

### 5.2 Autoformalisasi Teorema dengan Hipotesis yang Tidak Konsisten

**Contoh b. Masalah:** Diberikan bilangan real $D$ dan kondisi bahwa untuk bilangan real tak-nol $a, b, c$, determinan dari matriks $\begin{bmatrix} a & b & c \\ 1 & 4 & 9 \\ 3 & 1 & 2 \end{bmatrix}$ sama dengan $D$, buktikan bahwa $D^2 = 154$.

**Teorema yang Diautoformalisasi dalam Lean (Versi Cacat Awal):**
```lean
example (D : ℝ) (h0 : ∀ a b c : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 →
  Matrix.det ![![a, b, c], ![1, 4, 9], ![3, 1, 2]] = D) : D ^ 2 = 154
```
*   **Analisis Cacat:** Autoformalisasi awal di atas secara keliru mengasumsikan bahwa kondisi persamaan matriks $= D$ berlaku secara **universal** untuk SEMUA ($\forall$) bilangan real tak-nol $a, b,$ dan $c$. Ini adalah kelemahan logika (salah tafsir) karena masalah tidak mengklaim penerapan universal, melainkan pada nilai $a,b,c$ *tertentu* (seharusnya eksistensial / $\exists$).

**Sistem Penolakan Hipotesis Beraksi:**
Model secara brilian mengidentifikasi inkonsistensi (kebodohan logika) dari asumsianya sendiri dan memberikan "contoh tandingan" (*counterexample*) untuk membuktikan keabsurdan hipotesis tersebut:

```lean
example (D : ℝ) (h0 : ∀ a b c : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 →
  Matrix.det ![![a, b, c], ![1, 4, 9], ![3, 1, 2]] = D) : False := by
  have h1 := h0 1 2 3
  have h2 := h0 1 4 9
  simp [Matrix.det_fin_three] at h1 h2
  linarith
```
*   **Penjelasan Kode Penolakan:** Model mengubah tujuannya menjadi membuktikan `False` (kemustahilan). Ia memasukkan dua set nilai acak: (1,2,3) menghasilkan `h1` dan (1,4,9) menghasilkan `h2`. Ia mensubstitusikan nilai ini ke hipotesis palsu `h0`. Karena determinan untuk dua matriks yang berbeda tidak mungkin menghasilkan nilai $D$ yang sama secara bersamaan, taktik `linarith` akan mendeteksi kontradiksi aritmatika, yang berarti hipotesis awal adalah absurd. Bukti untuk `False` sukses, maka **pernyataan cacat ini dibuang dari dataset**.

**Versi Teorema yang Telah Dikoreksi:**
Setelah menyadari kesalahannya, model dapat memperbaiki terjemahannya (menghilangkan $\forall$ yang salah penempatan) menjadi:
```lean
example (a b c : ℝ) (h0 : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) :
  let D := Matrix.det ![![a, b, c], ![1, 4, 9], ![3, 1, 2]];
  D ^ 2 = 154
```
Contoh ini memberikan ilustrasi meyakinkan tentang kapasitas model untuk **mengevaluasi dirinya sendiri**, memverifikasi bukti, dan secara tajam mengidentifikasi inkonsistensi hipotesis sebelum informasi sampah tersebut merusak korpus pelatihannya sendiri.

---

## Kesimpulan

Dalam penelitian yang sangat komprehensif ini, disajikan sebuah metode inovatif untuk menghasilkan data bukti sintetik dalam jumlah ekstensif yang diserap dari masalah kompetisi matematika tingkat menengah hingga sarjana. 

Melalui proses alur kerja yang menerjemahkan masalah bahasa alami ke dalam pernyataan formal, memfilter yang berkualitas rendah (menggunakan penolakan hipotesis kontradiktif), dan menggunakan pembangkitan bukti secara **iteratif dan paralel** (membuktikan argumen dan negasinya secara bersamaan), penelitian ini berhasil menciptakan **8 juta titik data bukti formal**. 

Model yang dihasilkan, **DeepSeek-Prover (7B)**, secara dramatis meningkatkan kinerjanya dalam Pembuktian Teorema Otomatis (ATP). Model ini terbukti melampaui kemampuan raksasa industri GPT-4 serta berbagai arsitektur pencarian canggih lainnya pada tolok ukur *state-of-the-art* seperti miniF2F dan FIMO. 

Dengan membuka akses (open-sourcing) terhadap himpunan data dan model yang telah dilatih, langkah ini diharapkan akan mempercepat laju penelitian global dalam otomatisasi pembuktian teorema dan memajukan perbatasan kemampuan penalaran matematika formal pada Model Bahasa Besar. Ke depannya, ekspansi keragaman ruang lingkup matematika menjadi target eksplorasi selanjutnya dalam memperkaya kapabilitas arsitektur ATP.