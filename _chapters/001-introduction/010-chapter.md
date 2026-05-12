---
slug: bab-01
title: "Pengantar tentang DeepSeek"
layout: chapter
---

# **Pengantar tentang DeepSeek**

## **Yang akan kita pelajari pada bab ini:**

*   Mengapa DeepSeek menjadi titik balik (momen perubahan besar) dalam dunia AI *open-source* (sumber terbuka yang bisa diakses siapa saja), mulai dari awal mula model V3/R1 hingga model V3.2 dan V4 yang menembus batas teknologi.
*   Gambaran besar tentang inovasi-inovasi utama yang akan kita bangun bersama, termasuk *Sparse Attention* (Perhatian Renggang), *Engram*, *mHC*, dan *On-Policy Distillation* (Penyulingan Pengetahuan).
*   Struktur buku, batasan materi, dan apa saja yang perlu Anda siapkan.

Model Bahasa Besar (*Large Language Models* atau LLM) telah mengubah dunia teknologi dalam beberapa tahun terakhir. Kita sekarang hidup di dunia di mana sistem Kecerdasan Buatan (AI) dapat melakukan percakapan, menulis kode komputer, menyusun esai, menjalankan tugas-tugas rumit secara mandiri, dan bahkan memecahkan soal olimpiade matematika dengan cara yang terasa sangat mirip dengan manusia. Namun, bagaimana jika Anda, seorang pembaca yang memiliki rasa ingin tahu tinggi terhadap teknologi, bisa membangun salah satu model AI yang sangat canggih ini dari nol?

Bagaimana jika Anda bisa memahami cara kerja bagian dalam dari sebuah LLM paling modern dengan merakitnya langkah demi langkah, menggabungkan teori dan kode pemrograman secara bersamaan? Itulah tepatnya yang ingin kami ajarkan kepada Anda melalui buku ini.

Kita akan membongkar lapisan demi lapisan dari keluarga LLM *open-source* paling canggih saat ini yang bernama **DeepSeek**. Kita akan menciptakan ulang inovasi-inovasi utamanya dari dasar. Pada akhir buku ini, Anda tidak hanya akan memahami apa yang membuat DeepSeek (termasuk versi terbarunya, yaitu V3.2 dan V4) begitu unik, tetapi Anda juga akan tahu cara menerapkan inovasi tersebut sendiri. Sepanjang perjalanan ini, Anda akan mendapatkan wawasan yang sangat berharga tentang pengembangan AI modern.

Kita akan mulai dengan memahami mengapa DeepSeek itu penting: bagaimana model ini muncul sebagai titik balik dalam dunia AI *open-source* dan mengapa kami memilihnya sebagai fokus utama buku ini. Setelah itu, kita akan melihat peta jalan dari inovasi-inovasi inti DeepSeek. Istilah-istilah seperti *Multi-Head Latent Attention*, *Mixture-of-Experts* (Campuran Para Ahli), *Conditional Memory* (Memori Bersyarat / *Engram*), *Manifold-Constrained Hyper-Connections* (*mHC*), dan Kuantisasi FP4 mungkin terdengar menakutkan bagi Anda saat ini. Namun, jangan khawatir, kami akan memperkenalkannya dengan cara yang mudah dipahami dan menjelaskan mengapa teknologi tersebut sangat penting untuk membuat AI menjadi lebih besar dan pintar.

Selanjutnya, kami akan menjelaskan struktur buku ini, apa yang bisa Anda harapkan untuk dipelajari (serta apa yang tidak akan kita bahas), untuk memastikan tujuan kami sejalan dengan tujuan Anda. Kemudian, kami akan merinci apa saja yang Anda butuhkan untuk mengikuti buku ini, mulai dari pengetahuan dasar hingga persyaratan perangkat keras (komputer) dan perangkat lunaknya. Tenang saja, Anda tidak akan membutuhkan komputer super raksasa!

Sebelum kita mulai, sedikit latar belakang: buku ini terinspirasi dari seri YouTube Vizuara yang berjudul "Build DeepSeek from Scratch" (Membangun DeepSeek dari Nol). Melalui buku ini, kami telah merangkum pengalaman kami dari dunia akademik (universitas) dan industri, serta pelajaran praktik langsung dari seri video tersebut, menjadi sebuah perjalanan terstruktur yang kini mencakup batas penelitian AI *open-source* yang paling mutakhir.

## **1.1 Mengapa DeepSeek? Titik balik dalam AI Open-Source**

Dengan begitu banyaknya model bahasa di luar sana, Anda mungkin bertanya-tanya: mengapa kita harus fokus pada DeepSeek? Apa yang membuat model ini begitu istimewa sehingga kami memutuskan untuk menulis satu buku penuh tentang cara membangunnya? Jawaban singkatnya adalah bahwa DeepSeek mewakili titik balik dalam dunia AI *open-source*. DeepSeek membuktikan untuk pertama kalinya bahwa sebuah model yang tersedia secara terbuka (gratis untuk diakses kodenya) dapat menyaingi—dan terkadang mengalahkan—kinerja model-model berbayar terbaik buatan perusahaan besar.

Mari kita mundur sejenak dan melihat keadaan AI sebelum DeepSeek muncul. Pada awal tahun 2020-an, model bahasa besar didominasi oleh beberapa raksasa teknologi dan laboratorium penelitian yang memiliki dana dan sumber daya sangat besar. Seri GPT dari OpenAI dan Gemini dari Google memimpin di posisi teratas dalam hal kemampuan, tetapi model tersebut bersifat tertutup (hanya bisa diakses dengan membayar). Komunitas *open-source* memang memiliki keberhasilan melalui model seperti LLaMA dan Qwen, namun selalu ada jarak ketertinggalan dalam hal kemampuan menalar tugas-tugas yang rumit.

DeepSeek muncul di tengah situasi ini, dan membawa konsep keterbukaan ke tingkat yang baru. Didirikan pada tahun 2023, laboratorium penelitian ini dengan cepat membuat gebrakan. Kehebatan DeepSeek pertama kali tidak bisa disangkal saat mereka merilis **DeepSeek-R1**, sebuah model penalaran yang mengejutkan komunitas AI karena berhasil menyamai model o1 milik OpenAI pada ujian-ujian sulit, seperti *American Invitational Mathematics Examination* (AIME) dan berbagai platform kompetisi *coding* (pemrograman).

Namun, DeepSeek tidak berhenti di situ. Dengan perilisan cepat **DeepSeek-V3.2** dan **DeepSeek-V4**, mereka benar-benar menghancurkan batasan efisiensi. Mereka mampu membuat model yang bisa membaca teks sangat panjang (hingga 1 juta token/kata) dan menggunakan alat-alat bantu eksternal secara mandiri. DeepSeek-V3.2 bahkan meraih performa setara medali emas di Olimpiade Informatika Internasional (IOI), sementara DeepSeek-V4 menciptakan standar baru dengan arsitektur yang sangat efisien untuk diuji pada skala yang sangat besar.

Fakta mengejutkan lainnya adalah masalah biaya: DeepSeek melatih model-model canggih ini hanya dengan sebagian kecil dari biaya yang dikeluarkan oleh perusahaan besar pembuat model berbayar. Bagi kita sebagai pelajar dan pembuat teknologi, DeepSeek adalah bahan pelajaran yang sempurna. Kesuksesan mereka berasal dari penemuan teknologi yang brilian dan dibagikan secara terbuka. Hal ini memunculkan pertanyaan-pertanyaan penting yang akan kita jawab di sepanjang buku ini:

*   Bagaimana cara DeepSeek mencapai hasil terbaik di dunia dengan kebutuhan komputer (komputasi) yang jauh lebih sedikit?
*   Bagaimana arsitektur seperti *DeepSeek Sparse Attention (DSA)* dan *Engram* (Memori Bersyarat) memecahkan masalah kemacetan pada teknologi Transformer standar?
*   Bagaimana cara Anda melatih AI agar "berpikir" terlebih dahulu sebelum ia menggunakan alat bantu atau mencari jawaban di internet?
*   Bagaimana beberapa AI kecil yang menjadi "ahli" di bidang tertentu digabungkan menjadi satu model utuh melalui proses yang disebut *On-Policy Distillation (OPD)*?

Dengan meniru dan membuat ulang bagian-bagian dari DeepSeek, kita pada dasarnya menelusuri kembali langkah-langkah penelitian AI paling maju di zaman kita. Ketika Anda membangun sesuatu dengan tangan Anda sendiri, Anda akan benar-benar menguasai ilmu tersebut, jauh melampaui apa yang bisa Anda dapatkan hanya dengan membaca makalah penelitian atau menggunakan aplikasi AI yang sudah jadi.

## **1.2 Inovasi-inovasi utama yang akan kita bangun**

Mari kita susun peta jalan arsitektur untuk membangun DeepSeek dari nol. Arsitektur (rancangan bangunan) DeepSeek didirikan di atas fondasi bernama *Transformer* yang sudah sangat dikenal. Namun, DeepSeek memperkenalkan inovasi besar untuk mengatasi masalah kinerja dan keterbatasan ukuran. Kita akan memperlakukan setiap inovasi sebagai studi kasus: pertama kita menganalisis kelemahan cara standar, lalu kita membangun teknologi penggantinya yang lebih canggih.

**1.2.1 Arsitektur (Rancangan Bangunan)**

Inti dari sebuah LLM standar bergantung pada sistem *Multi-Head Attention* yang padat dan jaringan *Feed-Forward Networks* (FFNs). DeepSeek mengganti dan meningkatkan komponen standar ini dengan sistem pengganti yang sangat efisien dan khusus:

*   **Multi-Head Latent Attention (MLA) & Sparse Attention:** Sistem perhatian standar membutuhkan ruang memori penyimpanan sementara yang sangat besar (disebut *Key-Value / KV cache*). DeepSeek memperkenalkan MLA untuk memampatkan (mengecilkan) ukuran memori ini. Pada versi V3.2 dan V4, mereka mendorong hal ini lebih jauh dengan **DeepSeek Sparse Attention (DSA)** dan pendekatan campuran seperti **Compressed Sparse Attention (CSA)** serta **Heavily Compressed Attention (HCA)**. Hal ini mengurangi beban kerja komputer lebih dari 90% dan memungkinkan AI membaca teks sepanjang 1 juta kata.
*   **Mixture-of-Experts (MoE) / Campuran Para Ahli:** Daripada menyalakan setiap bagian otak AI untuk setiap kata yang dibaca, DeepSeek hanya mengirim kata tersebut ke bagian jaringan saraf "ahli" yang spesifik. Ini sangat memperbesar kapasitas kepintaran model tanpa membuatnya menjadi lambat.
*   **Memori Bersyarat melalui Engram:** Ini adalah cara baru untuk menghemat kerja AI. Sementara MoE menangani logika yang berubah-ubah, sistem AI standar sering kali buruk dalam mengingat fakta-fakta pasti. DeepSeek memperkenalkan *Engram*, sebuah modul yang menggunakan pencarian instan untuk mengambil pengetahuan pasti, sehingga jaringan otak AI bisa fokus sepenuhnya pada pemikiran dan penalaran yang rumit.
*   **Manifold-Constrained Hyper-Connections (mHC):** Semakin dalam (semakin banyak lapisan) sebuah model AI, sambungan standarnya bisa menjadi tidak stabil saat dilatih. DeepSeek-V4 memperkenalkan mHC, yang menempatkan sambungan tersebut ke dalam sebuah ruang matematika bernama *Birkhoff polytope* untuk menjamin proses pelatihan tetap stabil meskipun model tersebut memiliki triliunan parameter (sel saraf buatan).

**1.2.2 Pelatihan**

Selain rancangan bangunan utamanya, DeepSeek juga berinovasi besar dalam hal efisiensi dan cara melatih AI:

*   **Prediksi Multi-Token (MTP):** Alih-alih hanya menebak *satu* kata berikutnya, DeepSeek menebak beberapa kata masa depan sekaligus secara bersamaan. Hal ini memberikan sinyal belajar yang lebih kuat dan membuat AI bisa bekerja jauh lebih cepat saat digunakan.
*   **Kuantisasi Ekstrem (FP8 dan FP4):** Melatih model berukuran triliunan parameter membutuhkan memori yang luar biasa besar. DeepSeek menggunakan ukuran data yang sangat kecil, yaitu *8-bit floating-point* (FP8), dan bahkan mendorong ke batas paling ekstrem yaitu 4-bit (FP4) untuk bagian pakar (MoE)-nya. Ini memotong penggunaan memori hingga setengahnya tanpa mengurangi keakuratan AI.
*   **Muon Optimizer:** Meninggalkan pengoptimal standar seperti AdamW, DeepSeek menggunakan pengoptimal bernama *Muon* dengan rumus matematika khusus untuk mencapai hasil belajar yang lebih cepat dan stabil saat dilatih menggunakan ribuan kartu grafis (GPU) yang saling terhubung.

**1.2.3 Pasca-Pelatihan (Setelah Pelatihan Dasar)**

Sebuah model AI yang baru selesai dilatih dasar hanya bisa memunculkan teks biasa; ia perlu diajari *bagaimana* cara bertindak, berpikir secara masuk akal, dan menggunakan alat. Proses pasca-pelatihan DeepSeek adalah pelajaran berharga tentang bagaimana menyelaraskan AI modern:

*   **Scalable GRPO (Group Relative Policy Optimization):** Ini adalah algoritma pembelajaran khusus (Reinforcement Learning) milik DeepSeek yang memungkinkan model untuk mengeksplorasi dan mengembangkan pola "berpikir" melalui proses coba-coba (trial-and-error). Hal ini menghilangkan kebutuhan akan data berlabel buatan manusia yang jumlahnya sangat banyak dan harganya mahal.
*   **Agentic AI & Penggunaan Alat:** DeepSeek-V3.2 memperkenalkan metode canggih untuk "Manajemen Konteks Berpikir". Model ini belajar untuk mempertahankan jalan pikirannya sendiri di dalam kepalanya, saat ia sedang berinteraksi dengan alat luar (seperti kalkulator kode atau mesin pencari web) melalui ratusan langkah kerja.
*   **On-Policy Distillation (OPD):** Pada V4, DeepSeek menyempurnakan metode pasca-pelatihan dua tahap. Pertama, mereka melatih model-model "Spesialis" yang berdiri sendiri (misalnya, AI yang ahli Matematika, AI yang ahli *Coding*). Kemudian, mereka menggunakan metode OPD untuk menyuling dan memindahkan pengetahuan dari semua model guru yang besar tersebut ke dalam satu model murid yang tunggal dan sangat efisien.

## **1.3 Struktur dan cakupan buku**

Kami telah menyusun buku ini ke dalam sebuah peta jalan yang logis dan jelas, yang terbagi menjadi delapan bab. Struktur ini bersifat bertahap, di mana setiap tahap dibangun langsung di atas pengetahuan dan kode pemrograman dari tahap sebelumnya.

**Bagian 1: Masalah Kinerja Komputer & Evolusi Perhatian (Attention)**
*   **Bab 2:** Kita mulai dengan masalah paling dasar dari LLM: *Key-Value (KV) Cache* (Memori Penyimpanan Sementara). Anda akan belajar mengapa hal ini diperlukan dan mengapa ia memakan memori komputer sangat besar.
*   **Bab 3:** Kita akan membuat kode dari terobosan sistem *Attention* milik DeepSeek. Kita akan menulis kode untuk *Multi-Head Latent Attention* (MLA), menjelajahi teknik pemahaman posisi kata (*Rotary Positional Embeddings* / RoPE), dan menerapkan sistem *Attention* yang renggang dan campuran yang sangat efisien dari versi V3.2 dan V4.

**Bagian 2: Meningkatkan Kapasitas & Arsitektur Model**
*   **Bab 4:** Kita akan menangani masalah peningkatan ukuran model. Anda akan membangun lapisan *Mixture-of-Experts* (MoE) dari nol, dan kemudian menerapkan modul **Engram** baru untuk menambahkan sistem memori bersyarat ke dalam model Anda.

**Bagian 3: Efisiensi Pelatihan Ekstrem**
*   **Bab 5:** Kita akan mendalami tujuan pelatihan dan format data. Anda akan menerapkan sistem Prediksi Multi-Token (MTP) dan menjelajahi matematika serta kode di balik kuantisasi FP8 dan FP4 (pengecilan ukuran data).
*   **Bab 6:** Kita akan melihat arsitektur skala besar yang dibutuhkan untuk membuat model berukuran triliunan parameter. Anda akan belajar tentang *Manifold-Constrained Hyper-Connections* (mHC), Pengoptimal *Muon*, dan trik infrastruktur yang dibutuhkan untuk melatih model menggunakan 32 Triliun kata (token).

**Bagian 4: Penyelarasan, Agen Berpikir, dan Penyulingan Pengetahuan**
*   **Bab 7:** Kita memasuki fase setelah pelatihan dasar. Kita akan menerapkan metode GRPO untuk proses AI belajar dari coba-coba (*Reinforcement Learning*), dan mempelajari cara melatih model agar bertindak sebagai Agen yang dapat "berpikir" dan menggunakan alat bantu luar.
*   **Bab 8:** Kita mengakhiri dengan penggabungan model. Anda akan mempelajari bagaimana DeepSeek menggunakan metode *On-Policy Distillation* (OPD) untuk menyatukan kecerdasan dari model-model guru yang sangat besar menjadi satu model yang ringkas dan mudah digunakan.

## **1.4 Apa yang akan diajarkan buku ini dan apa yang tidak**

Buku ini dirancang sebagai perjalanan praktik langsung melalui berbagai inovasi bangunan (arsitektur) yang membuat DeepSeek bisa tercipta. Kami percaya bahwa cara terbaik untuk memahami sistem-sistem yang rumit ini adalah dengan membangunnya sendiri.

**Apa yang akan Anda pelajari:**
*   Cara membangun sistem MLA dan *Sparse Attention* untuk memecahkan masalah memori komputer yang penuh.
*   Matematika dan kode pemrograman di balik sistem rute ahli (MoE) dan pencarian memori *Engram*.
*   Cara menerapkan sistem Prediksi Multi-Token agar AI dapat menjawab lebih cepat.
*   Prinsip-prinsip pengecilan ukuran data yang ekstrem (Kuantisasi FP8/FP4) dan arsitektur ukuran raksasa seperti mHC.
*   Cara menerapkan metode Belajar dari Coba-Coba (GRPO) untuk mengajari model agar bisa "berpikir" dan menggunakan alat.
*   Cara kerja metode Penyulingan Pengetahuan (OPD) untuk memampatkan model guru yang besar menjadi model murid yang efisien.

**Apa yang tidak akan dibahas dalam buku ini:**
*   Kami tidak akan memproduksi ulang data pelatihan rahasia milik DeepSeek atau mencoba melatih model berukuran 600 miliar parameter dari nol (karena itu tidak mungkin dilakukan di komputer biasa).
*   Kami tidak akan membahas secara mendalam tentang manajemen tingkat rendah dari ribuan komputer (pengaturan DevOps kluster 10.000 GPU), karena hal tersebut membutuhkan sumber daya di luar jangkauan kebanyakan pembaca.
*   Kami tidak akan membahas masalah pembuatan produk siap jual seperti membangun desain tampilan web antarmuka (UI) atau membuat penyaring moderasi untuk keamanan konten AI.

Sebagai gantinya, kami berfokus pada kejelasan dan pemahaman Anda. Setiap konsep diperkenalkan dari prinsip dasarnya. Pada akhirnya, Anda akan memahami teknik-teknik ini dengan sangat dalam, sehingga Anda dapat mengubah dan meningkatkannya untuk proyek AI Anda sendiri di masa depan.

## **1.5 Apa yang Anda butuhkan untuk mengikuti buku ini**

Untuk mengikuti buku ini, Anda membutuhkan pemahaman dasar tentang konsep *Machine Learning* (Pembelajaran Mesin), tetapi Anda tidak membutuhkan gelar doktor (PhD). Jika Anda terbiasa dengan bahasa pemrograman Python dan pernah mempelajari materi pengantar *Deep Learning* (Pembelajaran Mendalam), Anda sudah siap untuk memulai. Secara khusus, Anda harus memahami bagaimana jaringan saraf buatan belajar melalui sebuah proses bernama *backpropagation* (perambatan balik), familier dengan operasi dasar di aplikasi PyTorch, dan pernah melihat bentuk bangunan arsitektur Transformer yang standar.

Dari segi perangkat keras komputer (hardware), kami telah merancang semua praktik ini agar mudah diakses. Walaupun melatih model paling mutakhir membutuhkan komputer super raksasa, kita akan bekerja dengan versi yang diperkecil. Versi kecil ini tetap mencakup semua ide matematika pentingnya, namun tetap sanggup dijalankan.

Sebuah laptop dengan prosesor (CPU) yang cukup bagus dapat menjalankan sebagian besar contoh-contoh konsep yang kecil. Akan tetapi, sebuah komputer yang dilengkapi satu kartu grafis (GPU) dengan memori VRAM 8-24GB (seperti NVIDIA RTX 3090, 4090, atau layanan *cloud* internet yang setara) akan membuat pengalaman belajar Anda jauh lebih lancar. Hal ini memungkinkan Anda untuk bereksperimen dengan bebas menggunakan MoE, Engram, dan Kuantisasi. Kami akan menyediakan rincian spesifikasi lingkungan kerja yang lengkap, beserta pengaturan *Google Colab* yang mudah digunakan untuk setiap babnya.

Mari kita bangun sesuatu yang luar biasa bersama-sama!

## **1.6 Ringkasan**

*   **Era DeepSeek:** DeepSeek menandai momen penting dalam sejarah AI dengan merilis model-model terbuka (*open-source* seperti R1, V3.2, V4) yang mampu menyamai atau bahkan melampaui kinerja raksasa teknologi berbayar, hanya dengan sebagian kecil dari biaya mereka.
*   **Belajar Langsung dengan Praktik:** Buku ini akan membimbing Anda membangun model "DeepSeek Mini" dari nol, dengan fokus pada terobosan teknologi khusus milik mereka, bukan sekadar konsep LLM biasa.
*   **Peta Jalan Delapan Bab:** Perjalanan kita mencakup empat area utama: (1) Masalah Kinerja Komputer & Sistem Perhatian, (2) Peningkatan Kapasitas dengan MoE & Engram, (3) Efisiensi Pelatihan Ekstrem (MTP, Kuantisasi, mHC), dan (4) Pasca-Pelatihan Dasar (Belajar Coba-Coba, Agen AI, Penyulingan Ilmu).
*   **Pemberdayaan:** Dengan membangun komponen-komponen ini sendiri, Anda akan mendapatkan pengetahuan praktis langsung dari prinsip dasar arsitektur AI yang paling mutakhir. Hal ini akan memberdayakan Anda untuk menyesuaikan teknik-teknik ini demi pengembangan AI di masa depan.