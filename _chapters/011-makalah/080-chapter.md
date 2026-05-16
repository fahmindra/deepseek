---
slug: ch-11-08
title: "DeepSeek-Coder-V2"
layout: chapter
---

## DeepSeek-Coder-V2: Mendobrak Batas Model Sumber Tertutup dalam Kecerdasan Kode

Pada sesi ini, kita akan membahas **DeepSeek-Coder-V2**, sebuah model bahasa pemrograman bersumber terbuka (*open-source code language model*) yang menggunakan arsitektur *Mixture-of-Experts* (MoE). Hal yang patut mendapat perhatian khusus dari model ini adalah kemampuannya mencapai kinerja yang **sebanding dengan GPT-4-Turbo** pada tugas-tugas spesifik pemrograman.

Secara teknis, DeepSeek-Coder-V2 merupakan hasil prapelatihan lanjutan (*continued pre-training*) dari titik pemeriksaan (*checkpoint*) menengah model DeepSeek-V2. Proses prapelatihan lanjutan ini melibatkan tambahan **6 Triliun (6T) token**. Melalui proses ini, DeepSeek-Coder-V2 secara substansial meningkatkan kemampuannya dalam:
1. Pemrograman (*coding*).
2. Penalaran matematis (*mathematical reasoning*).

Hebatnya, peningkatan drastis pada domain spesifik ini dicapai sembari tetap mempertahankan kinerja yang setara pada tugas-tugas bahasa umum (*general language tasks*). Dibandingkan dengan pendahulunya, DeepSeek-Coder-33B, versi V2 ini mendemonstrasikan kemajuan signifikan di berbagai aspek pemrograman, penalaran, dan kapabilitas umum.

Sebagai mahasiswa ilmu komputer, Anda harus menyoroti dua peningkatan arsitektural utama:
*   Dukungan bahasa pemrograman diperluas dari 86 menjadi **338 bahasa**.
*   Panjang konteks (*context length*) diperluas dari 16K (16.000) menjadi **128K (128.000) token**.

Dalam evaluasi tolok ukur standar (*standard benchmark evaluations*), DeepSeek-Coder-V2 berhasil mengungguli kinerja model-model raksasa bersumber tertutup (*closed-source models*) seperti GPT-4-Turbo, Claude 3 Opus, dan Gemini 1.5 Pro, khususnya pada tolok ukur pemrograman dan matematika.

---

### Analisis Akademis Gambar 1: Kinerja DeepSeek-Coder-V2 pada Tolok Ukur Pemrograman dan Matematika

Mari kita bedah secara analitis grafik utama yang disajikan dalam makalah ini (Gambar 1).

**Deskripsi Grafik:**
Gambar 1 menampilkan diagram batang ganda (*grouped bar chart*) yang membandingkan metrik akurasi dari enam model bahasa utama pada enam tolok ukur (*benchmarks*) yang berbeda.

*   **Sumbu X (Horizontal):** Menampilkan enam tolok ukur spesifik, dari kiri ke kanan:
    1.  `HumanEval`: Uji kemampuan menulis fungsi Python berdasarkan instruksi deskriptif.
    2.  `MBPP+`: Sama seperti HumanEval, namun sering diuji dengan contoh (*few-shot*).
    3.  `MATH`: Tolok ukur penyelesaian masalah matematika kompetitif tingkat tinggi.
    4.  `GSM8K`: Tolok ukur penyelesaian soal matematika tingkat sekolah dasar.
    5.  `Aider`: Tolok ukur kemampuan model dalam melakukan perbaikan dan pengeditan kode secara otomatis.
    6.  `LiveCodeBench`: Evaluasi berkelanjutan menggunakan soal-soal kontes pemrograman langsung (seperti LeetCode).
    7.  `SWE-Bench`: (Tertulis di sisi paling kanan bawah) Evaluasi pemecahan masalah (isu) GitHub di dunia nyata.

*   **Sumbu Y (Vertikal):** Merepresentasikan skor Akurasi (*Accuracy*) dalam persentase (%), dengan rentang 50 hingga 100 untuk lima tolok ukur pertama, dan rentang 0 hingga 80 untuk bagian kanan.

*   **Legenda Warna Model:**
    *   Biru Bergaris Miring: `DeepSeek-Coder-V2` (Fokus Utama Kita)
    *   Ungu Muda: `GPT-4-Turbo-0409`
    *   Hijau Tosca: `Gemini-1.5-Pro`
    *   Krem: `Claude-3-Opus`
    *   Abu-abu Kecokelatan: `Llama-3-70B`
    *   Merah Muda / *Coral*: `Codestral`

**Tren dan Kesimpulan Akademis:**
1.  **Dominasi di HumanEval dan MBPP+:** Pada pilar `HumanEval`, DeepSeek-Coder-V2 (Biru) memimpin dengan akurasi **90.2%**, melampaui GPT-4-Turbo (88.2%). Tren ini konsisten di `MBPP+` di mana DeepSeek memimpin dengan **76.2%**. Ini membuktikan supremasinya dalam menerjemahkan teks menjadi fungsi pemrograman sederhana hingga menengah.
2.  **Keunggulan dalam Logika Abstrak (MATH):** Pada pilar `MATH`, sebuah tolok ukur di mana banyak model berguguran, DeepSeek-Coder-V2 mencetak **75.7%**, yang merupakan margin tipis di atas GPT-4-Turbo (73.4%). Ini sangat mengesankan mengingat soal-soal MATH memerlukan penalaran multi-langkah yang rumit. Pada GSM8K, performanya sangat ketat dengan model tertutup di rentang 90%+.
3.  **Kapasitas Pengeditan dan Rekayasa Perangkat Lunak:** Pada pilar `Aider` (pengeditan kode), DeepSeek-Coder-V2 unggul masif dengan **73.7%**, jauh di atas Claude-3-Opus (68.4%). Pada pilar `LiveCodeBench` (kode kompetisi), DeepSeek juga memimpin dengan **43.4%**. Terakhir, di `SWE-Bench` (kemampuan menjadi *software engineer* virtual), skornya menembus **12.7%**, yang sangat kompetitif di ranah *open-source*, meskipun belum mendominasi model tertutup.
4.  **Konklusi Eksekutif:** Grafik ini secara visual dan numerik memvalidasi klaim utama makalah: bahwa batas atau kesenjangan (*barrier*) antara model sumber terbuka (seperti DeepSeek dan Llama-3) dengan model raksasa sumber tertutup (seperti GPT-4 dan Gemini) dalam tugas-tugas teknis tingkat tinggi, kini telah berhasil didobrak.

---

## 1. Pendahuluan

Komunitas sumber terbuka telah membuat langkah yang sangat signifikan dalam memajukan kecerdasan kode melalui pengembangan model-model terkemuka seperti StarCoder (Li et al., 2023b; Lozhkov et al., 2024), CodeLlama (Roziere et al., 2023), DeepSeek-Coder (Guo et al., 2024), dan Codestral (MistralAI, 2024). Model-model ini secara mantap mendekati tingkat kinerja model-model sumber tertutup, memberikan kontribusi besar pada kemajuan kecerdasan pengkodean. 

Meskipun demikian, kita harus mengakui bahwa masih terdapat **kesenjangan yang kentara (*discernible gap*)** ketika kita membandingkannya dengan model-model sumber tertutup mutakhir (*state-of-the-art* / SOTA) seperti GPT-4-Turbo (OpenAI, 2023), Claude 3 Opus (Anthropic, 2024), dan Gemini 1.5 Pro (Reid et al., 2024). Untuk menjembatani kesenjangan tersebut dan semakin mendorong pengembangan model kode sumber terbuka, diperkenalkanlah seri **DeepSeek-Coder-V2**. Model-model ini dibangun di atas fondasi **DeepSeek-V2** (DeepSeek-AI, 2024) dan selanjutnya diberikan prapelatihan lanjutan dengan korpus tambahan sebanyak 6 triliun token.

**Fase Prapelatihan (*Pre-training Phase*)**
Kumpulan data untuk DeepSeek-Coder-V2 diciptakan dengan komposisi yang cermat:
*   **60%** Kode sumber (*source code*)
*   **10%** Korpus matematika
*   **30%** Korpus bahasa alami

Khusus untuk kode sumber, kumpulan data ini terdiri dari **1.170 Miliar (1.170B)** token terkait kode yang diambil dari GitHub dan *CommonCrawl*, dengan menggunakan saluran (*pipeline*) yang sama seperti pada DeepSeekMath (Shao et al., 2024). Korpus ini diperluas dari 86 menjadi **338 bahasa pemrograman**. 

Untuk mendemonstrasikan efektivitas dari korpus kode yang baru ini, dilakukan studi ablasi (*ablation studies*) menggunakan model berparameter 1B. Hasilnya menunjukkan peningkatan akurasi sebesar 6,7% dan 9,4% pada tolok ukur HumanEval (dari 30,5% menjadi 37,2%) dan MBPP (dari 44,6% menjadi 54,0%). 

Sedangkan untuk korpus matematika, dikumpulkan 221B token terkait matematika dari *CommonCrawl* menggunakan saluran yang sama. Jumlah ini secara kasar menggandakan ukuran korpus DeepSeekMath 120B. Untuk korpus bahasa alami, sampel diambil langsung dari korpus pelatihan DeepSeek-V2. Secara total, DeepSeek-Coder-V2 terpapar pada **10,2 Triliun (10.2T) token pelatihan**, di mana 4,2 triliun berasal dari dataset DeepSeek V2 awal, sementara sisa 6 triliun tokennya berasal dari dataset DeepSeek-Coder-V2.

**Ekspansi Panjang Konteks (*Context Length Expansion*)**
Untuk mengakomodasi input kode yang jauh lebih panjang dan meningkatkan penerapannya dalam skenario rekayasa perangkat lunak dunia nyata, panjang konteks diperluas dari 16K menjadi **128K token**. Hal ini memungkinkan model untuk membaca dan memahami satu proyek direktori utuh atau log eror yang masif. Pasca prapelatihan berkelanjutan pada korpora jamak-sumber (*multi-source corpora*) ini, terbukti bahwa DeepSeek-Coder-V2 secara drastis meningkatkan kapabilitasnya dalam penalaran matematis dan pemrograman tanpa merusak kinerja pemahaman bahasa umumnya.

**Fase Penyelarasan (*Alignment Phase*)**
1.  **Pelatihan Instruksi (*Instruction Training*):** Awalnya dibangun dataset yang meliputi data kode dan matematika dari DeepSeek-Coder dan DeepSeek-Math, dipadukan dengan data instruksi umum dari DeepSeek-V2. Kumpulan data ini difungsikan untuk melatih-halus (*fine-tune*) model dasar (*base model*).
2.  **Pembelajaran Penguatan (*Reinforcement Learning* / RL):** Menggunakan algoritma **Group Relative Policy Optimization (GRPO)** untuk menyelaraskan (*align*) perilaku model dengan preferensi manusia. Data preferensi dalam domain pengkodean dikumpulkan menggunakan umpan balik kompilator (*compiler feedback*) dan kasus uji (*test cases*). Sebuah Model Penghargaan (*reward model*) dikembangkan untuk memandu pelatihan model kebijakan (*policy model*). Pendekatan ini memastikan jawaban model dioptimalkan demi kebenaran fungsional dan preferensi gaya manusia dalam tugas pemrograman. 
3.  **Pelengkap Kode (*Code Completion*):** Untuk memberdayakan model agar mendukung pengisian kode otomatis (*autocomplete*) setelah penyelarasan, digunakan pendekatan **Fill-In-Middle (FIM)** (Guo et al., 2024) selama tahap *fine-tuning* pada model dasar 16B parameter.

### 1.1 Kontribusi Penelitian (*Contributions*)

Sebagai ringkasan, kontribusi utama dalam makalah ini adalah:

*   Memperkenalkan DeepSeek-Coder-V2 dengan parameter 16B dan 236B yang dibangun pada kerangka kerja MoE (*Mixture-of-Experts*). Secara efisien, kerangka ini memiliki **parameter aktivasi hanya sebesar 2.4B dan 21B**, sehingga dengan sangat efisien menopang berbagai kebutuhan komputasi dan aplikasi. Model juga mendukung 338 bahasa pemrograman dan jendela konteks maksimum 128K token.
*   Merupakan upaya pertama dalam ranah akademis/industri untuk mengembangkan sebuah model kode sumber terbuka berskala **ratusan miliar parameter** (*hundred-billion-parameter code model*) demi memajukan bidang kecerdasan kode. Hasil eksperimen menegaskan bahwa DeepSeek-Coder-V2 236B mengalahkan model-model sumber tertutup canggih, seperti GPT-4-Turbo, Claude 3 Opus, dan Gemini 1.5 Pro, pada tugas pemrograman maupun matematika.
*   Seluruh model DeepSeek-Coder-V2 dirilis secara publik di bawah lisensi permisif, yang mengizinkan kebebasan riset akademis maupun penggunaan komersial tanpa batas.

### 1.2 Ringkasan Evaluasi dan Metrik (*Summary of Evaluations and Metrics*)

*   **Pemrograman (*Code*):** Pada evaluasi generasi kode, DeepSeek-Coder-V2 menunjukkan superioritas mutlak atas semua model sumber terbuka dan berkinerja setara dengan model tertutup terdepan. Rekor yang dicapai mencakup skor **90.2%** pada HumanEval, **76.2%** pada MBPP (menetapkan rekor SOTA baru dengan saluran evaluasi EvalPlus), dan skor **43.4%** pada LiveCodeBench (rentang pertanyaan Des. 2023 - Juni 2024). Ia juga menjadi model sumber terbuka pertama yang menembus skor 10% pada SWEBench (Jimenez et al., 2023).
*   **Matematika (*Math*):** Menyaingi raksasa seperti GPT-4o, Gemini 1.5 Pro, dan Claude 3 Opus pada tolok ukur dasar seperti GSM8K dan tolok ukur kompetisi lanjutan seperti MATH, AIME, dan Math Odyssey. Pencapaian luar biasa adalah akurasi **75.7%** di MATH (hampir identik dengan rekor SOTA GPT-4o di 76.6%). Model ini juga sukses mengalahkan model-model tertutup tersebut pada kompetisi AIME 2024.
*   **Bahasa Alami (*Natural Language*):** DeepSeek-Coder-V2 sukses menjaga performa bahasa yang sebanding dengan DeepSeek-V2 murni. Mencapai **79.2%** pada MMLU menggunakan saluran *simple-eval* OpenAI. Pada evaluasi subjektif di mana GPT-4 bertindak sebagai Juri (*judger*), model ini meraih **65.0** di arena-hard, **8.77** di MT-bench, dan **7.84** di alignbench. Skor ini jauh lebih tinggi dari model khusus kode lainnya dan sejajar dengan model bahasa umum sumber terbuka terbaik.

---

## 2. Pengumpulan Data (*Data Collection*)

Kualitas model AI generatif dikte oleh kualitas data pelatihannya. Data prapelatihan untuk DeepSeek-Coder-V2 didistribusikan sebagai **60% kode sumber, 10% korpus matematika, dan 30% korpus bahasa alami**. Karena korpus bahasa alami secara utuh diambil dari dataset DeepSeek-V2, bagian ini hanya akan membedah koleksi, pembersihan, dan pemfilteran pada data kode dan matematika.

Repositori publik yang dibuat sebelum **November 2023** dikumpulkan dari GitHub. Aturan pemfilteran awal dan deduplikasi jarak-dekat (*near-deduplication*) yang digunakan sama dengan seri DeepSeek-Coder sebelumnya (Guo et al., 2024) untuk membuang kode berulang dan berkualitas rendah. 

**Aturan Pemfilteran (*Filtering Rules*):**
1.  Membuang berkas yang panjang rata-rata per barisnya melebih 100 karakter, atau yang memiliki panjang maksimal satu baris melebihi 1000 karakter (ini sering kali berupa *minified file* atau data generat mesin, bukan kode yang diketik manusia).
2.  Menghapus berkas yang proporsi karakter alfabetnya kurang dari 25% (untuk membuang berkas biner/data angka murni).
3.  Kecuali bahasa pemrograman XSLT, seluruh berkas yang diawali dengan *string* `"<?xml version="` dalam 100 karakter pertamanya dihapus.
4.  Untuk berkas HTML, rasio teks yang terbaca manusia (*visible text*) terhadap *tag* HTML dikalkulasi. Hanya berkas dengan porsi teks terbaca minimal 20% dan tidak kurang dari 100 karakter yang dipertahankan.
5.  Untuk berkas konfigurasi data seperti JSON dan YAML, hanya berkas dengan panjang antara 50 hingga 5000 karakter yang disimpan (efektif membersihkan berkas pembuangan data masif).

Dengan mengaplikasikan pemfilteran ketat ini beserta deduplikasi, peneliti mengekstraksi **821B token kode murni** yang melingkupi 338 bahasa pemrograman, ditambah **185B token teks berbasis kode** (seperti fail Markdown dan diskusi repositori/Issues). Daftar lengkap bahasanya ada di *Appendix A*. Proses ini memanfatkan sistem tokenisasi (*tokenizer*) dari DeepSeekV2.

**Pengumpulan Data Web (Kode & Matematika):**
Untuk teks terkait pengkodean dan matematika dari *Common Crawl*, saluran yang sama seperti DeepSeekMath digunakan. Situs forum pemrograman (*StackOverflow*), portal dokumentasi pustaka (*PyTorch documentation*), dan forum matematika (*StackExchange*) digunakan sebagai **korpus benih (*seed corpus*)**. 

Korpus benih ini digunakan untuk melatih sebuah pengklasifikasi bernama **fastText model** (Joulin et al., 2016) yang bertugas mendeteksi dan mengumpulkan halaman web sejenis dari hamparan *Common Crawl*. Mengingat bahasa Asia seperti Tiongkok tidak dipisahkan oleh spasi antar kata, *Byte Pair Encoding* (BPE) *tokenizer* dari DeepSeek-V2 digunakan untuk memecah teks, yang terbukti secara signifikan mendongkrak akurasi penarikan data (*recall accuracy*) dari fastText.

Setiap domain URL yang memiliki lebih dari 10% halamannya diklasifikasikan sebagai kode/matematika oleh model, ditandai sebagai domain spesifik. Proses anotasi URL terus digulirkan, dan setiap *hyperlink* yang belum terkumpul dari URL tersebut ditambahkan ke korpus benih. Setelah **tiga iterasi** siklus pengumpulan data ini, berhasil dipanen **70 miliar token teks kode** dan **221B token teks matematika**. 

Untuk memperkaya kualitas, metode saluran yang sama dialihkan untuk menambang repositori GitHub selama dua iterasi, mengumpulkan **94B kode sumber bermutu sangat tinggi** (terutama yang dibubuhi penjelasan berbobot tinggi/ *detailed descriptions*). Hasil akhirnya adalah sebuah himpunan raksasa berukuran **1.170B token terkait pemrograman**.

---

### Tabel 1: Analisis Kinerja Model Dasar 1B antara Korpus Lama dan Korpus DeepSeek-Coder-V2

Untuk memastikan bahwa usaha melelahkan dalam membersihkan dan mengekstraksi data ini berhasil secara sains, para ilmuwan menjalankan studi ablasi: melatih model kecil 1B (*1 Billion Parameters*) dan membandingkan performanya.

| Model | Tokens | Python | C++ | Java | PHP | TS | C# | Bash | JS | Avg | MBPP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| DeepSeek-Coder-1B (Lama)| 1T | 30.5% | 28.0% | 31.7% | 23.0% | 30.8% | 31.7% | 9.5% | 28.6% | 26.7% | 44.6% |
| **DS-Coder-V2-1B (Korpus Baru)** | **1T** | **36.0%** | **34.8%** | **31.7%** | **27.3%** | **37.7%** | **34.2%** | **6.3%** | **38.5%** | **31.2%** | **49.0%** |
| DS-Coder-V2-1B (Diperpanjang)| 2T | 37.2% | 39.1% | 32.3% | 31.7% | 34.6% | 36.7% | 12.0% | 32.9% | 32.0% | 54.0% |

**Kesimpulan Akademis:**
Jika Anda melihat Baris 1 (Dataset Lama) dan Baris 2 (Dataset Baru V2) yang sama-sama dilatih selama 1 Triliun token (1T), Anda akan menyadari bahwa skor rata-rata (Avg) melompat dari 26.7% menjadi **31.2%**. Nilai MBPP melompat dari 44.6% menjadi **49.0%**. Eksperimen ini memvalidasi secara empiris bahwa kualitas pembersihan data dan strategi penyertaan multibahasa pada korpus DeepSeek-Coder-V2 secara dramatis meningkatkan daya tangkap kognitif kecerdasan buatan, melebihi sekadar pertambahan volume data belaka. Jika token ditambah ke 2T (Baris 3), efikasi kecerdasan ini akan berlanjut meningkat.

---

## 3. Kebijakan Pelatihan (*Training Policy*)

### 3.1 Strategi Pelatihan (*Training Strategy*)

Terdapat dua objektif pelatihan (*training objectives*) yang diimplementasikan pada DeepSeek-Coder-v2 berparameter 16B: 
1.  **Next-Token-Prediction** (Prediksi Token Berikutnya).
2.  **Fill-In-Middle (FIM)** (Pengisian Bagian Tengah).

Penting untuk dicatat bahwa untuk varian raksasa **236B**, hanya mekanisme *Next-Token-Prediction* yang digunakan guna menghindari instabilitas komputasional skala tinggi.

**Metode Fill-In-Middle (FIM):**
Ini adalah teknik krusial untuk melatih AI agar bisa digunakan di *IDE* (*Integrated Development Environment* seperti VSCode) sebagai fitur *autocomplete*. Model tidak hanya dilatih untuk menebak kata berikutnya yang ada di ujung sebuah teks (dari kiri ke kanan), tetapi ia diajarkan memprediksi blok teks yang terhilang di tengah sebuah dokumen.

Strategi ini memecah dokumen acak menjadi tiga urutan: **Prefix** (Awalan), **Suffix** (Akhiran), dan **Middle** (Tengah). Implementasi FIM yang diadopsi adalah struktur **PSM (Prefix, Suffix, Middle)**, yang menyusun ulang urutan rekonstruksi data teks selama pelatihan menjadi seperti kode berikut:

```markdown
<|fim_begin|> Prefix <|fim_hole|> Suffix <|fim_end|> Middle <|eos_token|>
```

**Penjelasan Prompt FIM:**
*   `<|fim_begin|>`: Tanda pengenal khusus bagi model bahwa ia sedang berada dalam mode tugas sisipan.
*   `Prefix`: Konteks awal, yaitu seluruh kode yang ada *sebelum* posisi kursor *programmer*.
*   `<|fim_hole|>`: Indikator khusus "lubang hitam". Ini adalah posisi persis di mana kursor berada (di mana kode hilang).
*   `Suffix`: Konteks ekor, yaitu seluruh baris kode yang berada *setelah* kursor. Ini penting agar model tahu fungsi apa yang akan ditutup atau dipanggil di bawahnya.
*   `<|fim_end|>`: Indikator bahwa pembacaan konteks depan dan belakang sudah selesai.
*   `Middle`: Ini adalah *Ground Truth* (kebenaran aktual), yakni teks yang seharusnya berada di antara *Prefix* dan *Suffix*. Model dipaksa menebak teks ini dan menghitung gradien kesalahan (*loss*) dari prediksinya.
*   `<|eos_token|>`: Penanda akhir sekuens (End of String).

Metode ini disuntikkan secara dinamis pada level dokumen saat fase *pre-packing* sebelum masuk ke *batch* GPU. Parameter *FIM Rate* disetel di angka **0.5** (artinya 50% data diubah formatnya menjadi FIM, dan 50% sisanya dibaca normal dari kiri ke kanan).

---

### 3.2 Arsitektur Model (*Model Architecture*)

Arsitektur yang diusung selaras dengan kerangka kerja DeepSeek-V2 (DeepSeek-AI, 2024). Pengaturan hiperparameter (*hyperparameters*) untuk versi 16B dan 236B identik dengan pengaturan pada *DeepSeek-V2-Lite* dan *DeepSeek-V2*.

Sebagai mahasiswa *machine learning*, satu hal yang harus Anda cermati dari pengalaman para peneliti ini adalah fenomena **instabilitas gradien**. Saat mencoba menggunakan teknik normalisasi matematis yang canggih (*exponential normalization technique*) pada lapisan *layer norm*, terjadi anomali fluktuasi (*spikes in gradient values*) yang menyebabkan pelatihan menjadi berantakan. Para peneliti secara logis merekonsiliasi masalah ini dengan kembali menggunakan teknik normalisasi konvensional (*conventional normalization method*, umumnya seperti RMSNorm) demi mengamankan konvergensi model triliunan token ini.

### 3.3 Hiperparameter Pelatihan (*Training Hyper-Parameters*)

Konsisten dengan metodologi aslinya, digunakan pengoptimal atau optimiser standar industri: **AdamW** (Loshchilov and Hutter, 2019).
*   Momentum Beta: $\beta_1 = 0.9$ dan $\beta_2 = 0.95$.
*   *Weight Decay* (Penyusutan Bobot): $0.1$.
*   Penjadwalan *Learning Rate* (LR): **Cosine Decay Strategy** (Strategi peluruhan Kosinus). LR dikatrol naik dari nol selama 2.000 langkah pertama (*warm-up steps*), dan kemudian LR diluruhkan secara halus bagai lengkungan fungsi kosinus hingga menyentuh angka 10% dari nilai puncaknya di akhir pelatihan.

### Tabel 2: Pengaturan Pelatihan DeepSeek-Coder-V2

| Model | DeepSeek-Coder-V2-Lite | DeepSeek-Coder-V2 |
| :--- | :---: | :---: |
| # Total Parameters (#TP) | 16B | 236B |
| # Active Parameters (#AP) | 2.4B | 21B |
| Pre-training Tokens | 4.2T + 6T | 4.2T + 6T |
| LR Scheduler | Cosine | Cosine |
| FIM | Enable | Disable |

*(Catatan Dosen: Perhatikan metrik "Active Parameters" yang luar biasa kecil berkat teknik MoE (Mixture of Experts). Model raksasa 236B secara mengejutkan hanya membakar daya kalkulasi untuk 21B parameter pada satu waktu. Kinerja maksimal, efisiensi maksimal).*

### 3.4 Perpanjangan Konteks Panjang (*Long Context Extension*)

Model dasar hanya menangkap memori pendek. Untuk memperluas wawasan kognitifnya hingga **128K** (memungkinkan AI membaca manual arsitektur setebal buku sekaligus sebelum menulis kode), digunakan manipulasi matriks yang dikenal sebagai **Yarn** (Peng et al., 2023).

Hiperparameter Yarn: skala ($s$) diatur ke 40, $\alpha$ ke 1, dan $\beta$ ke 32. Ini dilakukan dengan *fine-tuning* lanjutan bertahap:
1.  **Tahap 1:** Memperpanjang konteks ke 32K. Panjang sekuens dinaikkan, ukuran *batch* diatur ke 1152 urutan, dan dijalankan selama 1000 langkah.
2.  **Tahap 2:** Memperpanjang konteks dari 32K menjadi target akhir 128K. Ukuran *batch* disusutkan menjadi 288 urutan (agar tidak kehabisan memori VRAM GPU) selama 1000 langkah tambahan.

---

### Analisis Akademis Gambar 2: Pengujian Tekanan (Pressure Testing) Memori "Jarum di Tumpukan Jerami"

Bagaimana membuktikan bahwa AI yang diklaim mampu membaca konteks sepanjang 128.000 kata tidak benar-benar lupa tentang fakta yang ditaruh di tengah dokumen tebal tersebut? Jawabannya ada di **Gambar 2**, yang dikenal dengan uji *Needle In A Haystack (NIAH)*.

**Deskripsi Grafik (Peta Panas / *Heatmap*):**
*   **Sumbu X (Horizontal):** Panjang konteks teks, merentang dari 1K (sangat pendek) hingga 128K (sebuah novel utuh atau direktori kode raksasa).
*   **Sumbu Y (Vertikal):** *Document Depth Percent* (%), yang mengindikasikan seberapa "dalam" letak sebaris kalimat informasi ('Jarum') itu disembunyikan di dalam lautan teks ('Tumpukan Jerami'). 0% berarti kalimat ditanam di ujung atas dokumen awal. 100% berarti kalimat tersebut ditanam di kalimat penutup dokumen.
*   **Warna Latar (Z-axis):** Representasi Skala Nilai (*Score*). Hijau Murni bermakna skor absolut 10 (AI menemukan jarum secara sempurna). Merah bermakna skor 1 (AI buta total, amnesia, atau berhalusinasi).

**Kesimpulan Akademis:** 
Peta panas ini menunjukkan blok masif berwarna **Hijau Sempurna (Skor 10)** pada semua kedalaman dokumen (0% hingga 100%) dan pada semua interval peregangan (1K hingga 128K token). Ini memvalidasi secara matematis bahwa fungsi ekstrapolasi matriks RoPE menggunakan metode Yarn benar-benar berhasil, sehingga DeepSeek-Coder-V2 memiliki kemampuan "pengingatan fotografis murni" lintas seluruh spektrum kapasitas konteks yang diklaim tanpa terjadi keruntuhan memori.

---

### 3.5 Penyelarasan (*Alignment*)

Tanpa penyelarasan, model bahasa hanyalah mesin pelanjut kata autokomplet yang sangat canggih. Untuk menjadikannya entitas asisten *chatbot* (DeepSeek-Coder-V2 Chat) yang berguna, dibutuhkan dua tahap: Pelatihan Halus (*Supervised Fine-Tuning* / SFT) dan Pembelajaran Penguatan (*Reinforcement Learning* / RL).

#### 3.5.1 Pelatihan Halus Terselia (*Supervised Fine-Tuning* / SFT)

Model dilatih ulang agar format percakapannya berstruktur layaknya tanya-jawab.
*   **Campuran Data:** Dataset ini terdiri dari kumpulan yang presisi: 20 ribu data instruksi spesifik pemrograman, dan 30 ribu data instruksi spesifik matematika. Untuk memastikan AI tidak menjadi "kaku", ditambahkan sampel instruksi bahasa keseharian (*general ability*) dari model utama DeepSeek-V2.
*   **Volume:** Total ukuran data instruksi mencapai 300 Juta (300M) token. Selama SFT, model secara total akan terpapar pada 1 Miliar (1B) token pelatihan.
*   **Metode Latih:** Peluruhan LR Kosinus (*cosine schedule*), 100 *warm-up steps*, inisial *learning rate* sangat kecil yakni $5 \times 10^{-6}$ (ini krusial agar model tidak melupakan pengetahuannya selama SFT, namun cukup untuk "menekuk" gaya bicaranya).

#### 3.5.2 Pembelajaran Penguatan (*Reinforcement Learning* / RL)

SFT tidak menjamin AI akan selalu meramu kode yang dapat dikompilasi (bebas *error*). RL hadir untuk memberikan "ganjaran" (*reward*) jika kode benar, dan "hukuman" (*penalty*) jika kode bermasalah.

*   **Penyusunan Prom (*Prompts*):** Diperoleh secara teliti hingga berjumlah 40 ribu *prompt* terkait kode dan matematika. Esensinya, setiap *prompt* kode harus disertai unit pengetesan otomatis (*test cases*).
*   **Pemodelan Penghargaan (*Reward Modeling*):** Dalam kasus matematika, nilai absolut benar-salah (*ground-truth labels*) sudah mutlak. Namun dalam pembuatan kode, fungsi kompilator yang memberikan nilai 0 (gagal *test case*) atau 1 (lulus *test case*) nyatanya **menghasilkan sinyal yang terlalu berisik (noisy) dan sub-optimal** bagi algoritma RL. Hal ini disebabkan karena *test cases* sering kali tidak memiliki cakupan penuh (*full coverage*). Jika AI dibiarkan belajar langsung dari kompilator, ia akan mengeksploitasi cacat pengujian tersebut.
*   **Solusi Akademis:** Membangun *neural network* model evaluator terpisah (**Reward Model**). *Reward Model* dilatih menggunakan umpan balik dari kompilator. Kemudian, *Reward Model* inilah yang difungsikan sebagai "pemberi sinyal ganjaran yang halus" selama pelatihan kebijakan RL. 

---

### Analisis Akademis Gambar 3: Kinerja Berbagai Metode Pembelajaran Penguatan (RL)

Mari kita buktikan secara empiris mengapa pendekatan *Reward Model* lebih baik dari *Compiler Signal* mentah, lewat grafik yang disajikan pada **Gambar 3**.

**Deskripsi Grafik:**
*   Menampilkan dua panel kurva metrik. Panel kiri adalah `LeetCode-Pass@1` (tingkat lolos koding tes global). Panel kanan adalah `Leetcode-zh-Pass@1` (tes setara dalam bahasa Mandarin).
*   **Sumbu X:** Menunjukkan jumlah putaran (*Steps*) pelatihan mesin RL, mulai dari 100 hingga 600.
*   **Sumbu Y:** Menunjukkan tingkat akurasi kelulusan (`Pass@1`), dari skala kisaran 0.10 hingga 0.22 (atau 10% hingga 22%).
*   **Kurva:**
    *   **Garis Hijau Horizontal:** *SFT Model*. Ini adalah batas dasar kecerdasan model sebelum mesin RL menyala. Garis ini statis karena ia adalah titik tolak ukuran.
    *   **Garis Biru Berfluktuasi:** *Compiler Signal*. Algoritma RL dibiarkan disuapi metrik mutlak 0-1 dari *compiler code*. Terlihat performanya sangat anarkis, naik tajam, tapi sering anjlok parah ke bawah garis *baseline* hijau. Terjadi degradasi performa pada tes karena model "kebingungan" oleh sifat asimtot sinyal biner *compiler*.
    *   **Garis Oranye di Puncak:** *Reward Model Signal*. Menggunakan jaringan *Reward Model* perantara. Kurva ini berfluktuasi secara sehat dengan lintasan progres yang stabil dan terus-menerus mendominasi ruang atas grafik.

**Kesimpulan Ilmiah:** Pendelegasian evaluasi nilai melalui representasi probabilitas *Reward Model* jauh melampaui penggunaan umpan balik nilai diskrit mentah (*raw compiler signal*), memastikan *Generalization Ability* (kemampuan abstraksi umum) dari arsitektur terjamin.

*   **Algoritma RL Final:** Digunakan teknik **Group Relative Policy Optimization (GRPO)** (Shao et al., 2024). Sebagai mahasiswa AI, ingatlah bahwa salah satu kelemahan model klasik PPO (*Proximal Policy Optimization*) adalah kebutuhan untuk terus menghidupkan *Critic Model* (yang seukuran *Policy Model*) di VRAM. GRPO meruntuhkan kelemahan ini. Ia mengalkulasi matriks penalti secara lebih pintar tanpa bantuan *critic model*, sehingga memangkas biaya peranti keras secara drastis saat optimasi RL.

---

## 4. Hasil Eksperimental (*Experimental Results*)

Pada bab evaluasi ini, arsitektur DeepSeek-Coder-V2 ditandingkan head-to-head di atas ring dengan para dedengkot LLM terkini. Sebagai patokan, mari kita jabarkan para pesaingnya:

*   **CodeLlama** (7B, 13B, 34B, 70B): Keluarga *coder* keturunan Llama2 racikan Meta yang dicekoki 500-1000 Miliar token koding.
*   **StarCoder** (15B): Berbasis *The Stack* dan dilatih ekstra teliti pada 86 bahasa.
*   **StarCoder2** (3B, 7B, 15B): Berlatih pada *Stack2* supermasif (3,3 - 4,3 Triliun token) melintasi 619 bahasa program.
*   **Codestral** (22B): Karya monumental dari perusahaan *Mistral*, mengusai spesialisasi langka seperti bahasa *Swift* dan *Fortran*.
*   **Model Tertutup Generalis** (*Closed-Source Models*): Sang Penguasa Tahta yang sebenarnya (Llama3 70B, GPT-4, Claude 3 Opus, dan Gemini 1.5 Pro). Meskipun tidak dibuat spesifik hanya untuk ngoding, rasio kecerdasan kognitif umum mereka sangat superior.

### 4.1 Pembuatan Kode (*Code Generation*)

Tolok ukur de-fakto untuk tes pembuatan kode adalah **HumanEval** dan **MBPP**. Ujian ini memaksa LLM untuk memuntahkan kode algoritma yang kompleks (seperti iterasi, matriks, hingga DFS/BFS) murni dengan cara membaca paragraf instruksi manusia (*zero-shot/few-shot*). Metrik *Pass@1* artinya jawaban tebakan pertamanya harus terkompilasi lulus tanpa campur tangan koreksi sedikitpun. Model juga dieksekusi secara ketat ke pelbagai bahasa menggunakan pencarian asimtot lurus (*greedy search*).

### Tabel 3: Kinerja Multi-Bahasa pada Tolok Ukur HumanEval dan MBPP

*(Karena keterbatasan panjang teks, saya selaku dosen akan merekapitulasi matriks krusial dari Tabel 3 ke dalam argumen akademis berikut)*

**Analisis Hasil Tabel 3 (Eksperimen HumanEval Multibahasa):**
*   **Kekalahan Model Tertutup (Closed-Source):** Perhatikan model pamungkas seperti *GPT-4o-0513*. Pada Python ia mencetak 91.0%, C++ (87.0%), dan Bash (53.8%). Rata-rata kemenangannya luar biasa. *Claude-3-Opus* meraup skor rata-rata (Average) sebesar 70.8%.
*   **Dominasi DeepSeek-Coder-V2-Instruct (236B):** Jika Anda menyusuri baris DeepSeek yang di-*fine-tune* penuh ini, ia sukses mencetak poin **Python: 90.2%**, **C++: 84.8%**, **Java: 82.3%**, dan **Bash: 52.5%**. Yang paling spektakuler, model *open source* ini meraih **Skor Rata-Rata (*Average*) tertingginya yakni 75.3%**, serta rekor baru di MBPP+ sebesar **76.2%**.
*   **Signifikansi Akademis:** Angka 75.3% ini bukan sekadar statistik. Ini mengoyak kemapanan industri. DeepSeek-V2 sukses mengubur dalam-dalam semua model *open-source* sekelasnya (Codestral hanya 63.2%, Llama3 70B hanya 60.6%). Lebih brutalnya lagi, angka 75.3% milik DeepSeek **membungkam Claude-3-Opus (70.8%)** dan **GPT-4-1106 (72.5%)**, hanya tunduk 1.1% selisih margin dari Sang Raja *GPT-4o* (76.4%). DeepSeek adalah petarung *open-source* pertama yang menduduki takhta ini.

---

### Tabel 4: Kinerja Pemrograman Kompetitif pada LiveCodeBench (LCB) dan USACO

Tolok ukur konvensional (HumanEval) perlahan usang karena kode masalahnya telah terekspos di mana-mana (bocor ke data pelatihan model / *data contamination*). Para peneliti butuh tolok ukur yang **100% steril**. Digunakanlah **LiveCodeBench**.

Ujian ini menyedot masalah dari kontes seperti LeetCode, AtCoder, dan CodeForces yang diterbitkan dalam **jendela waktu baru** (Des 2023 - Jun 2024), di mana *cut-off* data pelatihannya dipastikan tidak mengetahui solusi kode tersebut.

| Model | LiveCodeBench (Overall 226) | USACO |
| :--- | :---: | :---: |
| **Model Tertutup:** | | |
| Claude-3-Opus | 34.6% | 7.8% |
| GPT-4-1106 | 37.1% | 11.1% |
| GPT-4-Turbo-0409 | **45.7%** | 12.3% |
| GPT-4o-0513 | 43.4% | **18.8%** |
| **Model Terbuka:** | | |
| Codestral (22B) | 31.0% | 4.6% |
| Llama3-Instruct (70B) | 28.7% | 3.3% |
| DS-Coder-V2-Lite-Instruct (16B) | 24.3% | 6.5% |
| **DS-Coder-V2-Instruct (236B)** | **43.4%** | **12.1%** |

**Kajian Tabel 4:**
Ini adalah wilayah Pemrograman Kompetitif (*Competitive Programming*), ranah olimpiade matematika komputasi di mana para maestro *programmer* bersaing memecahkan masalah tingkat Dewa secara instan.
Di *LiveCodeBench*, DeepSeek-Coder-V2 secara mencengangkan **mencetak skor imbang (seri) 43.4% melawan GPT-4o**. Ia hanya merunduk kalah oleh GPT-4-Turbo (45.7%). Ini membuktikan kepiawaian DeepSeek bukan semata hasil menghafal sintaks, melainkan pemahaman logika algoritmik abstrak sejati yang tidak terlihat di data masa lalunya.

---

### 4.2 Pelengkap Kode (*Code Completion*)

#### 4.2.1 Evaluasi Penyelesaian Kode Tingkat-Repositori (*Repository-Level Code Completion Evaluation*)

Kebanyakan model hanya bisa menulis 1 paragraf fungsi. Dalam dunia kerja industri, seorang programer menulis baris kode yang akan memanggil variabel dan kelas dari ratusan fail (berkas) di direktori lainnya (seperti dependensi struktur proyek di Java). Evaluasi ini diserahkan pada tolok ukur **RepoBench** (Liu et al., 2023b), yang mensimulasikan penyelesaian kode dengan lintas dependensi repositori dunia nyata.

### Tabel 5: Kinerja RepoBench (Cuplikan Analisis)

*(Merujuk representasi data pada Tabel 5 makalah)*
Ujian RepoBench melintasi jendela konteks mulai dari 2k, 4k, 8k, 12k, hingga 16k token. 

*   **Poin Krusial:** Evaluasi ini difokuskan pada varian mungilnya, yakni **DeepSeek-Coder-V2-Lite-Base (16B Parameter Total / 2.4B Aktif)**. Model kurcaci yang hanya mengaktifkan 2.4 miliar neuron otaknya ini dikompetisikan di kelas berat melawan model dasar Codestral 22B dan Llama 34B.
*   **Kesimpulan Akademis:** Varian *Lite* (mungil) secara fenomenal berhasil menyetarakan skor kompetensi level repositori di Python dan Java. Walaupun metrik tertingginya masih direngkuh Codestral 22B (wajar, karena volume otaknya 10 kali lebih bengkak / 22B aktif vs 2.4B aktif), keberadaan DeepSeek-Lite menjadikannya satu-satunya pilihan yang rasional jika Anda ingin membangun perangkat asisten koding (*AI autocompletion extension* di VSCode) yang dieksekusi secara komputasi lokal, super cepat, tanpa menggerus memori RAM komputer pengembang.

#### 4.2.2 Penyelesaian Kode Fill-in-the-Middle (FIM)

Kemampuan menebak "Apa kalimat di tengah-tengah dua blok yang terpisah" adalah syarat mutlak kecerdasan alat *autocomplete* murni.

### Tabel 6: Kinerja Model pada Tugas-Tugas FIM

| Model | # Active Params | Python | Java | JavaScript | Mean (Rata-rata) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| StarCoder (16B) | 16B | 71.5% | 82.3% | 83.0% | 80.2% |
| CodeLlama-Base (13B) | 13B | 60.7% | 74.3% | 78.5% | 73.1% |
| Codestral (22B) | 22B | 77.2% | 83.2% | 85.9% | 83.0% |
| DS-Coder-Base (33B Lama) | 33B | **80.5%** | 88.4% | 86.6% | 86.4% |
| **DS-Coder-V2-Lite-Base (16B)** | **2.4B** | 80.0% | **89.1%** | **87.2%** | **86.4%** |

**Observasi Signifikan Tabel 6:**
Perhatikan betapa efisiennya arsitektur MoE bekerja di model *DeepSeek-Coder-V2-Lite*. Meskipun ia hanya menyalakan mesin otaknya sebesar 2.4B untuk menghasilkan komputasi yang kilat, ia merengkuh skor Mean puncak sebesar **86.4%**, yang secara brutal menghajar model *dense* (Codestral) yang mengerahkan tenaga 10 kali lipatnya (22B) di skor 83.0%. Inilah manifestasi kecemerlangan metode PSM-FIM selama proses prapelatihan.

---

### 4.3 Perbaikan Kode (*Code Fixing*)

Membangun kode dari nol memang sulit, tapi melakukan revisi/perbaikan (*debugging/fixing bug*) pada ribuan baris basis kode rekayasa lunak dunia nyata adalah hal yang mematikan.

Diuji di atas **Defects4J**, **SWE-bench** (kumpulan *issues* dan *bug* otentik dari raksasa repositori GitHub, seperti kerangka kerja Apache/Python), dan **Aider** (Tolok ukur kecerdasan menyalin spesifikasi perintah refaktorisasi kompleks).

### Tabel 7: Performa Model pada Tolok Ukur Perbaikan Kode

| Model | Defects4J | SWE-Bench | Aider |
| :--- | :---: | :---: | :---: |
| **Model Tertutup:** | | | |
| GPT-4o-0513 | **26.1%** | **26.7%** | 72.9% |
| Claude-3-Opus | 25.5% | 11.7% | 68.4% |
| GPT-4-Turbo-0409| 24.3% | 18.3% | 63.9% |
| **Model Terbuka:** | | | |
| Codestral (22B) | 17.8% | 2.7% | 51.1% |
| DS-Coder-Instruct (Lama) | 11.3% | 0.0% | 54.5% |
| **DS-Coder-V2-Instruct (236B)**| 21.0% | 12.7% | **73.7%** |

**Kajian Analitik Akademis:**
1.  **Aider (Tugas Pengeditan presisi):** DeepSeek V2 memenangkan kompetisi secara absolut di angka **73.7%**. Merobohkan kedigdayaan Sang Dewa, GPT-4o (72.9%). AI ini mampu menuruti pelbagai jenis manipulasi penataan ulang blok kode secara sempurna.
2.  **SWE-Bench (Simulator *Programmer* Dunia Nyata):** Evaluasi ini bagaikan skripsi bagi mesin LLM. Ia menguji seluruh ekosistem logika nalar model. Perhatikan model *open-source* lain, skor perbaikannya hancur, bahkan versi DeepSeek lama meraih angka 0.0% (Artinya sama sekali tidak sanggup memecahkan log *bug* kompleks *enterprise* GitHub). Namun model DeepSeek-Coder-V2 **menembus dinding batas ini**, melesat hingga mengunci kelulusan 12.7%, bersandar sangat erat membayangi model triliunan dollar layaknya Claude-3-Opus (11.7%).

---

### 4.4 Pemahaman dan Penalaran Kode (*Code Understanding and Reasoning*)

Ini adalah wilayah di mana AI diinterogasi melalui ujian **CRUXEval**. Bukan memintanya menuliskan kode, melainkan AI diberikan secarik kode, dan diminta meramalkan "Apa hasil eksekusi (Output) dari blok kode Python ini?" (CRUXEval-I), ATAU sebaliknya "Jika keluaran akhirnya harus bernilai ini, tebak inputan awal (Input) yang dilempar ke fungsi kode ini" (CRUXEval-O).

*   DeepSeek-Coder-V2-Instruct menancapkan taringnya paling tinggi di domain terbuka (Skor **70.0%** pada I-COT, dan **75.1%** pada O-COT).
*   *Catatan Evaluasi Limitasi:* Kendati menjadi sang primadona, saat dihadapkan pada raksasa tertutup GPT-4o, AI kita harus bertekuk lutut akibat disparitas (kesenjangan) kompleksitas pemodelan. Model raksasa masih jauh lebih visioner dalam kalkulasi pelacakan variabel komputasi memori terbalik (mencapai skor 77.4% dan 88.7%). Batasan 21 miliar parameter aktif milik DeepSeek menabrak siling pembatas untuk merunut rekursif aljabar sejauh mesin tertutup.

---

### 4.5 Penalaran Matematis (*Mathematical Reasoning*)

Kapasitas pemecahan masalah algoritma koding berjalan seiringan dengan intelegensi kerangka berpikir matematika abstrak. Model diadu di gelanggang bergengsi Olimpiade Matematika.

### Tabel 9: Evaluasi Intelegensia Penalaran Matematis

| Model | GSM8K | MATH | AIME 2024 | Math Odyssey |
| :--- | :---: | :---: | :---: | :---: |
| Gemini 1.5 Pro | 90.8% | 67.7% | 2/30 | 45.0% |
| GPT-4-Turbo-0409 | 93.7% | 73.4% | 3/30 | 46.8% |
| GPT-4o-0513 | **95.8%** | **76.6%** | 2/30 | 53.2% |
| Llama3-Instruct (70B) | 93.0% | 50.4% | 1/30 | 27.9% |
| **DS-Coder-V2-Instruct** | 94.9% | 75.7% | **4/30** | **53.7%** |

**Kesimpulan Evaluasi Tabel 9:**
Hasil pada panggung penalaran (*reasoning*) ini bisa dikatakan *di luar nalar*. 
Dengan metode pencarian asimtot mentah (*greedy decoding* / tebakan instan), DeepSeek-Coder-V2 mendesak skor akurasi penyelesaian **MATH (tingkat komputasi aljabar universitas)** di ekuator **75.7%** dan menembus rekor **53.7%** pada soal rumit Math Odyssey.

*Puncak Pemenangan Akademis:* Tolok ukur **AIME 2024** (Kompetisi Olimpiade Matematika Undangan Amerika yang diadakan beberapa bulan lalu, jadi mustahil soal ini "hapal" masuk di *training data*). Saat model-model dunia ditantang memecahkan teka-teki gila milik AIME, GPT-4o hanya sanggup menyelesaikan 2 dari 30 soal, GPT-4-Turbo pecah rekor 3 dari 30 soal. Tapi DeepSeek? Melumat **4/30** ujian dengan tuntas! Hal ini menahbiskan kapasitas abstraksi silogisme matematisnya yang nyaris tanpa cela.

---

### 4.6 Bahasa Alami Umum (*General Natural Language*)

Apakah mesin pintar ini hanyalah kalkulator tanpa hati dan jiwa untuk mengobrol? Sama sekali tidak. Arsitekturnya diwariskan lurus dari pendahulunya, DeepSeek-V2 Chat.

Dievaluasi pada tolok ukur kesadaran manusia dan pemahaman dunia: **MMLU, HellaSwag, PIQA, TriviaQA**, hingga bahasa Tiongkok (**C-Eval, CMMLU**), serta uji wicara terbuka di **MT-Bench** dan **Arena-Hard**.

**Tabel 10 (Analisis Peringkas Kinerja Multi-Talenta):**
*   Varian mungil **DeepSeek-Coder-V2-Lite-Instruct (16B)** memiliki skor BBH yang memukau dan melonjak telak (Arena Hard dari 11.40 melesat ke 38.10) melebihi varian *Chat* regulernya, membenarkan bahwa perombakan instruksi SFT berbasis penalaran matematis menyuntikkan kemajuan masif atas struktur logika sintaks dalam membedah pertanyaan-pertanyaan debat manusia yang penuh silogisme berantai. 
*   Beralih ke varian dewa **DeepSeek-Coder-V2 Instruct (236B)**, pengadopsian data proporsi besar pada pelatihan kode, matematika, dan pemecahan silogisme menumbangkan model sebelumnya dan mencetak supremasi kognitif di **Arena-Hard (Melesat drastis ke skor 65.00)**. 

---

## 5. Kesimpulan (*Conclusion*)

Di akhir perkuliahan ini, kita dapat menarik simpulan akademis definitif berdasarkan laporan teknis yang kita telaah. Rangkaian *Large Language Models (LLMs)* **DeepSeek-Coder-V2** secara substansial merupakan **anomali dan peretas batasan kemajuan** dalam jagat komputasi terbuka (*open-source*). 

Dikembangkan melalui asimilasi prapelatihan gabungan pada model basis *Mixture-of-Experts* (DeepSeek-V2) dengan penggelontoran tak kurang dari **6 Triliun token ekstra** yang digali murni dari ekosistem tingkat kode proyek tinggi, mesin ini berevolusi melampaui kemampuan deduktif dari sang asalnya. 

Perluasan spektakuler pada kapasitas penyerapan wawasan kontekstual (dari 16.000 menjadi **128.000 karakter komputasi**) dan dukungan memori linguistik yang direntangkan merangkul **338 spesifikasi sintaks bahasa pemrograman** menegaskan taring DeepSeek sebagai sistem yang tak tertandingi di ruang terbuka. Puncaknya ditahbiskan dalam evaluasi metrik laboratorium standar dan simulator turnamen kompetitif (*LiveCodeBench*, AIME), di mana DeepSeek-Coder-V2 melabrak pintu batas yang selama ini tak tertembus, mengklaim singgasana yang sejajar (bahkan terkadang menyungkurkan takhta) dominasi mutlak ras raksasa sumber tertutup (*closed-source*) paling canggih buatan korporasi elit dunia layaknya **GPT-4 Turbo, Claude 3 Opus, dan Gemini 1.5 Pro**.

Meskipun demikian, ada sedikit kesenjangan kepatuhan kognitif pada skenario kerumitan absolut seperti menengahi sengketa jutaan baris kode *software* (evaluasi *SWEbench*), menyingkap tantangan riset futuristis: AI harus dididik untuk lebih "patuh" dan jernih merangkai wawasan instruksional dari rekayasawan perangkat lunak (*instruction-following capabilities*) demi melahirkan otomasi produksi yang paripurna.

*(Akhir dari Modul Diktat Kuliah Akademik - DeepSeek-Coder-V2)*