---
slug: ch-11-19
title: "Insight Into DeepSeek-V3"
layout: chapter
---

## Wawasan Mendalam Mengenai DeepSeek-V3: Tantangan Penskalaan dan Refleksi Perangkat Keras untuk Arsitektur Kecerdasan Buatan

Penskalaan (perbesaran kapasitas) model bahasa besar (*Large Language Models*/LLMs) yang sangat pesat telah menyingkap keterbatasan krusial dalam arsitektur perangkat keras saat ini. Keterbatasan tersebut mencakup batasan pada kapasitas memori, efisiensi komputasional, serta lebar pita interkoneksi (*interconnection bandwidth*). Model DeepSeek-V3, yang dilatih menggunakan 2.048 unit GPU NVIDIA H800, mendemonstrasikan bagaimana perancangan bersama model yang peka terhadap perangkat keras (*hardware-aware model co-design*) dapat secara efektif mengatasi tantangan-tantangan ini. Hal ini memungkinkan pelaksanaan pelatihan dan inferensi berskala besar yang efisien secara biaya. 

Makalah ini menyajikan analisis mendalam terkait arsitektur model DeepSeek-V3 (dan saudaranya, DeepSeek-R1) beserta infrastruktur Kecerdasan Buatan (AI)-nya. Pembahasan akan menyoroti berbagai inovasi kunci, di antaranya: 
1. **Multi-head Latent Attention (MLA)** untuk meningkatkan efisiensi memori.
2. Arsitektur **Mixture of Experts (MoE)** yang mengoptimalkan pertukaran antara komputasi dan komunikasi.
3. **Pelatihan Presisi-Campuran FP8** (*FP8 mixed-precision training*) untuk mengoptimalkan potensi penuh dari kapabilitas perangkat keras.
4. Topologi Jaringan Multi-Bidang (*Multi-Plane Network Topology*) untuk meminimalkan beban jaringan tingkat klaster (*cluster-level network overhead*). 

Berangkat dari hambatan perangkat keras yang ditemui selama pengembangan DeepSeek-V3, makalah ini juga menyajikan diskusi yang lebih luas bersama para akademisi dan praktisi industri mengenai potensi arah perangkat keras di masa depan. Hal ini mencakup unit komputasi presisi-rendah yang akurat (*precise low-precision computation units*), konvergensi jaringan *scale-up* (perluasan ke atas) dan *scale-out* (perluasan ke luar), serta inovasi pada *fabric* komunikasi berlatensi rendah. Wawasan-wawasan ini menggarisbawahi peran kritis perancangan bersama (*co-design*) perangkat keras dan model dalam memenuhi eskalasi tuntutan beban kerja AI, sekaligus menawarkan cetak biru praktis untuk inovasi pada sistem AI generasi berikutnya.

---

## 1. Pendahuluan

### 1.1 Latar Belakang

Model Bahasa Besar (LLMs) telah mengalami evolusi yang sangat cepat dalam beberapa tahun terakhir. Evolusi ini didorong oleh kemajuan iteratif pada desain model, daya komputasi, dan ketersediaan data. Pada tahun 2024, model-model terobosan seperti GPT-4o, LLaMa-3, Claude 3.5 Sonnet, Grok-2, Qwen2.5, Gemini-2, serta model kita, **DeepSeek-V3**, telah memamerkan progres yang luar biasa, sehingga semakin mempersempit jarak menuju terwujudnya Kecerdasan Buatan Umum (*Artificial General Intelligence*/AGI). 

Sebagaimana ditunjukkan oleh Hukum Penskalaan (*Scaling Laws*), peningkatan ukuran model, data pelatihan, dan sumber daya komputasi berkorelasi langsung dengan peningkatan substansial pada performa model. Hal ini menegaskan peran krusial dari penskalaan (*scaling*) dalam memajukan kemampuan AI. Secara kolektif, perkembangan-perkembangan ini telah mengantarkan kita ke sebuah era di mana penskalaan ukuran model dan daya komputasi dipandang sebagai kunci untuk membuka tingkat kecerdasan yang lebih tinggi.

Perkembangan terbaru pada model-model penalaran (*reasoning models*) seperti seri o1/o3 dari OpenAI, DeepSeek-R1, Claude-3.7 Sonnet, Gemini 2.5 Pro, Seed1.5-Thinking, dan Qwen3 tidak hanya membuktikan manfaat yang diberikan oleh arsitektur berskala besar, tetapi juga urgensi untuk meningkatkan efisiensi inferensi. Hal ini sangat penting terutama dalam menangani konteks bacaan yang lebih panjang dan mencapai tingkat kedalaman penalaran yang lebih jauh. Kemajuan ini menggarisbawahi kebutuhan akan inferensi yang lebih cepat dan efisien, yang pada gilirannya akan menempatkan beban tuntutan yang terus meningkat pada sumber daya komputasional.

Untuk menjawab tantangan tersebut, para pemimpin industri seperti Alibaba, ByteDance, Google, xAI, dan Meta telah mengerahkan klaster pelatihan berukuran raksasa (*colossal training clusters*). Klaster ini memuat puluhan bahkan ratusan ribu unit GPU atau TPU. Walaupun infrastruktur masif tersebut telah memfasilitasi pengembangan model *state-of-the-art*, biayanya yang selangit menciptakan hambatan masuk (*barriers to entry*) yang sangat tinggi bagi tim riset dan organisasi yang lebih kecil. 

Meskipun terdapat hambatan tersebut, perusahaan rintisan sumber terbuka (*open-source startups*) seperti DeepSeek dan Mistral juga berlomba untuk mengembangkan model mutakhir. Secara khusus, DeepSeek telah sukses mendemonstrasikan bahwa perancangan bersama peranti lunak dan perangkat keras (*software-hardware co-design*) yang efektif dapat memfasilitasi pelatihan model besar yang efisien secara biaya. Hal ini membantu meratakan arena persaingan (*leveling the playing field*) bagi tim yang lebih kecil.

Dibangun di atas tradisi tersebut, DeepSeek-V3 merepresentasikan tonggak sejarah baru dalam hal pelatihan yang efisien secara biaya (*cost-effective training*). Hanya dengan memanfaatkan 2.048 unit GPU NVIDIA H800, DeepSeek-V3 mampu mencapai performa *state-of-the-art*. Pencapaian ini selaras dengan komitmen untuk memajukan AI melalui solusi yang praktis dan dapat diskalakan (*scalable*), seperti yang sebelumnya juga telah didemonstrasikan melalui arsitektur hemat biaya, Fire-Flyer AI-HPC. Praktik dan wawasan yang ditarik dari DeepSeek-V3 mendemonstrasikan bagaimana sumber daya perangkat keras yang ada dapat dimanfaatkan hingga ke potensi maksimalnya, memberikan pelajaran berharga bagi komunitas AI maupun Komputasi Kinerja Tinggi (*High Performance Computing*/HPC) secara luas.

### 1.2 Tujuan

Makalah ini tidak bertujuan untuk sekadar mengulang rincian arsitektural dan algoritmik dari DeepSeek-V3, yang mana telah didokumentasikan secara ekstensif dalam laporan teknis aslinya. Sebaliknya, makalah ini mengadopsi sudut pandang ganda (*dual perspective*)—yang mencakup arsitektur perangkat keras sekaligus desain model—untuk mengeksplorasi interaksi rumit di antara keduanya demi mewujudkan pelatihan dan inferensi berskala besar yang hemat biaya. Dengan memeriksa sinergi ini, kami bertujuan untuk menyajikan wawasan yang dapat langsung diterapkan (*actionable insights*) guna menskalakan LLM secara efisien tanpa perlu mengorbankan performa atau aksesibilitas.

Secara spesifik, makalah ini berfokus pada:
*   **Desain Model yang Didorong oleh Perangkat Keras (*Hardware-Driven Model Design*):** Menganalisis bagaimana fitur perangkat keras, seperti komputasi presisi rendah FP8 serta properti jaringan perluasan-ke-atas/ke-luar (*scale-up/scale-out*), mengilhami pilihan arsitektural pada DeepSeek-V3.
*   **Ketergantungan Timbal Balik Antara Perangkat Keras dan Model (*Mutual Dependencies Between Hardware and Models*):** Menginvestigasi bagaimana kapabilitas perangkat keras membentuk inovasi model, dan sebaliknya, bagaimana tuntutan LLM yang berevolusi mendorong kebutuhan akan perangkat keras generasi mendatang.
*   **Arah Masa Depan bagi Pengembangan Perangkat Keras (*Future Directions for Hardware Development*):** Menarik wawasan praktis dari DeepSeek-V3 untuk memandu perancangan bersama perangkat keras dan arsitektur model masa depan, guna membuka jalan bagi sistem AI yang tangguh sekaligus hemat biaya.

### 1.3 Struktur Makalah

Sisa dari makalah ini disusun sebagai berikut: 
*   **Bagian 2:** Mengeksplorasi prinsip desain yang mendasari arsitektur model DeepSeek-V3, dengan menyoroti inovasi penting seperti Atensi Laten Multi-Kepala (*Multi-head Latent Attention*/MLA), optimisasi Campuran-Pakar (*Mixture-of-Experts*/MoE), dan Modul Prediksi Multi-Token (*Multi-Token Prediction Module*).
*   **Bagian 3:** Mengilustrasikan bagaimana arsitektur model kita mengejar komputasi dan komunikasi dengan presisi rendah (*low-precision*).
*   **Bagian 4:** Menjabarkan optimisasi interkoneksi *scale-up* (antar GPU dalam satu *node*), membahas konvergensi *scale-up/scale-out*, serta mengeksplorasi bagaimana fitur-fitur perangkat keras memengaruhi strategi paralelisme dan seleksi pakar (*expert selection strategies*).
*   **Bagian 5:** Berfokus pada optimisasi jaringan *scale-out* (antar *node* peladen), termasuk desain bersama jaringan multi-bidang (*multi-plane network*) dan interkoneksi berlatensi rendah.
*   **Bagian 6:** Selain mengurai keterbatasan saat ini dan saran masa depan yang disebutkan di Bagian 3 hingga 5, bagian ini mengelaborasi wawasan yang lebih kritis dari DeepSeek-V3, dan mengidentifikasi arah untuk rancangan bersama perangkat keras dan model di masa depan.

---

## 2. Prinsip Desain untuk Model DeepSeek

Pengembangan DeepSeek-V3 merupakan contoh nyata dari pendekatan penskalaan LLM yang sadar akan perangkat keras (*hardware-aware approach*), di mana setiap keputusan desain diselaraskan secara hati-hati dengan kendala fisik perangkat keras (seperti memori, daya komputasi, kabel jaringan) guna mengoptimalkan performa dan efisiensi biaya.

Sebagaimana diilustrasikan pada Gambar 1 (akan dianalisis detail di bawah), DeepSeek-V3 mempekerjakan arsitektur **DeepSeekMoE** dan **Multi-head Latent Attention (MLA)** yang telah terbukti efektif pada generasi sebelumnya, DeepSeek-V2. DeepSeekMoE membuka potensi maksimal dari arsitektur MoE, sementara MLA secara drastis mengurangi konsumsi memori dengan memampatkan memori singgahan Key-Value (*KV caches*).

Selain itu, DeepSeek-V3 menginkorporasikan **pelatihan presisi-campuran FP8** (*FP8 mixed-precision training*). Ini secara signifikan menurunkan biaya komputasional dan membuat pelatihan skala besar menjadi lebih praktis tanpa mengorbankan kualitas kecerdasan model. Untuk meningkatkan kecepatan inferensi (proses model menghasilkan teks saat ditanya pengguna), DeepSeek-V3 mengintegrasikan penguraian spekulatif (*speculative decoding*) yang berakar pada Modul Prediksi Multi-Token (*Multi-Token Prediction Module*), yang pada akhirnya meningkatkan kecepatan pembuatan teks secara nyata.

Melampaui arsitektur model (*software*), kami juga mengeksplorasi infrastruktur AI (*hardware*) yang efisien biaya. Kami mendeploy topologi Jaringan Multi-Bidang (*Multi-Plane network*) dengan arsitektur dua-lapis *Fat-Tree*, sebagai pengganti topologi klasik tiga-lapis *Fat-Tree*. Hal ini terbukti mengurangi biaya jaringan tingkat klaster (*cluster networking costs*).

Inovasi-inovasi di atas bertujuan untuk mengatasi tiga tantangan fundamental dalam menskalakan LLM, yaitu: **Efisiensi Memori**, **Efektivitas Biaya**, dan **Kecepatan Inferensi**. Semuanya akan kita eksplorasi secara mendalam pada sub-bab berikut.

---

### Analisis Akademik Gambar 1: Arsitektur Dasar DeepSeek-V3

Sebelum kita masuk ke teori mendalam, mari kita pelajari anatomi sistem DeepSeek-V3 melalui diagram visual yang disediakan oleh peneliti pada **Gambar 1**.

**Deskripsi Visual Arsitektur:**
Gambar 1 merepresentasikan aliran data (*data flow*) yang sangat kompleks di dalam arsitektur DeepSeek-V3. Gambar ini dibagi menjadi beberapa modul utama, memperlihatkan integrasi inovasi: Modul Prediksi Multi-Token (MTP), Multi-Head Latent Attention (MLA), DeepSeekMoE, dan penanda presisi campuran (FP8, BF16, FP32).

*   **Sisi Kiri (Arsitektur Model Utama / *Main Model*):**
    *   Terdapat barisan **Input Tokens** ($t_1, t_2, t_3, t_4$) yang masuk ke lapisan **Embedding Layer** (beroperasi di presisi BF16).
    *   Data kemudian mengalir naik melewati tumpukan **Transformer Block $\times L$**. Perhatikan tag kecil berwarna kuning bertuliskan **FP8 Mixed Precision**. Ini berarti komputasi matriks yang sangat berat di dalam blok ini dieksekusi menggunakan format angka 8-bit (FP8) untuk menghemat ruang dan mempercepat hitungan.
    *   Data kemudian melewati tahap normalisasi **RMSNorm** (menggunakan presisi tinggi FP32 untuk mencegah *error*) dan masuk ke **Output Head** (menggunakan BF16).
    *   Hasil akhirnya dibandingkan dengan **Target Tokens** ($t_2, t_3, t_4, t_5$) untuk menghitung kerugian/ *error* (**Cross-Entropy Loss** $\mathcal{L}_{\text{Main}}$) yang juga dihitung dalam FP32.
*   **Sisi Kanan (Modul MTP - *Multi-Token Prediction*):**
    *   Di sebelah kanan Model Utama, terdapat blok-blok identik bernama **MTP Module 1, 2, dan 3**. Ini adalah rahasia kecepatan DeepSeek-V3!
    *   Daripada hanya menebak 1 kata ke depan, modul MTP ini dirancang untuk menebak kata ke-2, ke-3, dan ke-4 secara paralel.
    *   *Perhatikan panah panah hijau putus-putus bertuliskan "Shared".* Ini menunjukkan bahwa Modul MTP meminjam (*sharing*) Embedding Layer dan Output Head dari Model Utama. Ini sangat menghemat parameter!
*   **Kotak Kiri Bawah (Zoom-in Transformer Block):**
    *   Menjabarkan isi dari satu Transformer Block. Input masuk ke **RMSNorm** $\rightarrow$ **Attention** (beroperasi di FP8) $\rightarrow$ **RMSNorm** $\rightarrow$ **Feed-Forward Network** (beroperasi di FP8).
*   **Kotak Kanan Bawah (Inovasi Inti: MLA dan DeepSeekMoE):**
    *   **Multi-Head Latent Attention (MLA):** Visualisasi ini menunjukkan kompresi memori (KV Cache). Perhatikan titik masuk `Input Hidden $h_t$`. Vektor ini "diperas" menjadi satu titik kecil bernama `Latent $c_t^{KV}$` (berwarna abu-abu). Proses dekompresinya menghasilkan banyak kunci ($k$) dan nilai ($v$). Yang disimpan di memori selama inferensi (ditandai dengan kotak bergaris putus-putus biru **Cached During Inference**) hanyalah vektor laten kecil tersebut, bukan keseluruhan matriks K dan V seperti pada Transformer biasa. 
    *   **DeepSeekMoE:** Modul Jaringan Maju-Umpan (*Feed-Forward*). Perhatikan `Input Hidden $h_t'$`. Data akan masuk ke sistem perutean ganda (*Router*). Garis kuning stabil menuju ke `Shared Expert` (Pakar Bersama) yang selalu aktif untuk semua token. Garis putus-putus biru disalurkan oleh `Router` (melalui fungsi Top-K) HANYA ke beberapa `Routed Expert` (Pakar Terarah). Operasi ini direpresentasikan menggunakan panah silang `Dispatch: All-to-All` dan `Combine: All-to-All` yang memecah kalkulasi ke beberapa GPU, di mana komputasinya dikerjakan dalam FP8.

**Kesimpulan Akademis dari Gambar 1:** 
Diagram ini mendemonstrasikan sebuah mahakarya optimisasi perangkat lunak tingkat rendah (*low-level software optimization*). DeepSeek-V3 tidak hanya memperbesar ukuran matriks, tetapi menggunakan "trik bedah" matematis: 
1.  Menghitung hampir semuanya dalam 8-bit (FP8) yang sangat cepat.
2.  Menjaga presisi di simpul-simpul kritis menggunakan 32-bit (FP32) pada fungsi Normalisasi dan Loss.
3.  Memerangkap (*compressing*) memori cache (MLA) agar RAM GPU tidak cepat penuh.
4.  Menjalankan sebagian kecil otak (*sparse network*) menggunakan MoE.
5.  Menebak banyak kata sekaligus (MTP).

---

### 2.1 Efisiensi Memori (*Memory Efficiency*)

Secara umum, Model Bahasa Besar membutuhkan sumber daya memori yang kolosal, di mana permintaan memorinya meningkat lebih dari 1000% setiap tahunnya. Sebaliknya, laju pertumbuhan kapasitas memori berkecepatan tinggi (seperti HBM - *High Bandwidth Memory* pada GPU) jauh lebih lambat, tipikalnya kurang dari 50% per tahun. Walaupun strategi paralelisme multi-node (menyebarkan model ke banyak komputer) adalah solusi yang layak untuk mengatasi limitasi ini, upaya mengoptimalkan penggunaan memori pada titik sumbernya (*at the source*) tetap menjadi strategi yang krusial dan paling efektif.

#### 2.1.1 Model Presisi Rendah (*Low-Precision Models*)
Dibandingkan dengan model yang menggunakan presisi 16-bit (*bfloat16* / BF16) untuk matriks bobotnya (*weights*), pemanfaatan format 8-bit (FP8) secara signifikan mengurangi konsumsi memori hingga separuhnya. Ini secara efektif mengurangi kendala dari tantangan dinding memori AI (*AI memory wall challenge*). Pembahasan lebih rinci terkait teknik presisi rendah ini akan diuraikan pada Bagian 3: Desain yang Didorong oleh Presisi Rendah (*Low-Precision Driven Design*).

#### 2.1.2 Mengurangi Memori Singgahan (KV Cache) dengan MLA
Dalam skenario inferensi (penggunaan) LLM, permintaan dari pengguna (*user requests*) sering kali melibatkan percakapan multi-putaran (*multi-turn conversations*). Untuk menanganinya secara efisien tanpa harus membaca ulang riwayat obrolan dari awal setiap saat, konteks dari permintaan-permintaan sebelumnya disimpan ke dalam memori yang umum disebut sebagai **KV cache** (*Key-Value cache*). 

KV cache bekerja dengan menyimpan vektor Kunci (*Key*) dan Nilai (*Value*) dari token-token (kata) yang telah diproses sebelumnya. Selama setiap langkah inferensi untuk meramalkan kata baru, model hanya perlu menghitung vektor Key dan Value untuk token *saat ini*, lalu melakukan kalkulasi atensi (*attention computation*) dengan cara menggabungkannya bersama pasangan Key-Value historis yang sudah di-cache. 

Kalkulasi bertahap (*incremental computation*) ini mereduksi kompleksitas komputasional untuk menghasilkan setiap token menjadi rasio linier $O(N)$, menjadikannya efisien saat memproses urutan teks yang panjang atau input multi-putaran.

**Namun, hal ini memunculkan masalah (bottleneck) yang terikat dengan memori (*memory-bound*).** Mengapa? Karena beban komputasinya bergeser dari operasi GEMM (*General Matrix-Matrix Multiplication* - perkalian matriks dengan matriks yang sangat disukai GPU) menjadi operasi GEMV (*General Matrix-Vector Multiplication* - perkalian matriks dengan vektor tunggal). Operasi GEMV memiliki rasio komputasi-terhadap-memori (*compute-to-memory ratio*) yang sangat rendah. 
Meskipun perangkat keras modern menawarkan tenaga hitung ratusan TFLOPS, eksekusi GEMV akan sangat cepat tercekik oleh batas *memory bandwidth* (laju transfer data dari cip memori ke cip komputasi GPU). Akibatnya, akses memori menjadi hambatan utamanya.

Untuk mendobrak *bottleneck* memori ini, kami menerapkan **Multi-head Latent Attention (MLA)**. MLA memampatkan (*compresses*) representasi KV dari seluruh kepala atensi (*attention heads*) ke dalam sebuah vektor laten (*latent vector*) berukuran lebih kecil menggunakan matriks proyeksi. Matriks proyeksi ini ikut dilatih secara bersamaan dengan model secara keseluruhan. 
Saat masa inferensi (*generation*), **HANYA vektor laten kecil ini yang perlu disimpan ke dalam cache**. Ini secara dramatis menyusutkan konsumsi memori dibandingkan dengan keharusan menyimpan seluruh cache KV untuk semua kepala atensi.

Selain MLA, ada beberapa pendekatan populer lain yang telah diajukan di dunia akademis untuk mereduksi ukuran KV cache. Metode-metode berikut ini sangat bernilai dan memberikan inspirasi penting bagi kemajuan di bidang mekanisme atensi yang efisien memori:
*   **Shared KV (Atensi Kueri Terkelompok/GQA; Atensi Multi-Kueri/MQA):** Daripada mempertahankan pasangan KV yang terpisah untuk setiap kepala atensi, pada metode ini, *beberapa kepala* dipaksa berbagi satu set pasangan KV tunggal. Ini jelas menyusutkan penyimpanan KV secara signifikan. Metode yang representatif adalah Grouped-Query Attention (GQA) dan Multi-Query Attention (MQA).

---

### Analisis Akademis Tabel 1: Perbandingan Ukuran KV Cache

Untuk membuktikan efektivitas memori secara kuantitatif, mari kita bedah **Tabel 1**.

**Tabel 1: Perbandingan ukuran cache KV (presisi BF16): DeepSeek-V3 (MLA) sangat mereduksi ukuran cache KV dibandingkan dengan model lain yang menggunakan GQA.**

| Model | KV Cache Per Token | Pengganda (*Multiplier*) |
| :--- | :--- | :--- |
| **DeepSeek-V3 (MLA)** | **70.272 KB** | **1x** (Batas Dasar) |
| Qwen-2.5 72B (GQA) | 327.680 KB | 4.66x |
| LLaMA-3.1 405B (GQA) | 516.096 KB | 7.28x |

*   **Deskripsi Data:** Tabel ini membandingkan berapa Kilobyte (KB) memori RAM GPU yang ditelan oleh satu model untuk sekadar "mengingat" SATU kata/token yang baru saja ia keluarkan.
*   **Analisis Saintifik:** Model-model kelas dunia dari grup lain (Qwen dan LLaMA) menggunakan arsitektur GQA (Grouped-Query Attention) untuk menghemat memori. Namun, LLaMA-3.1 masih menuntut \~516 KB per kata. Qwen-2.5 menuntut \~327 KB per kata. 
*   Bandingkan dengan **DeepSeek-V3**. Berkat inovasi matematis dari arsitektur MLA (mengompresi data ke vektor laten), model ini HANYA membakar **\~70 KB** per token. Ini adalah pengurangan ukuran cache yang masif (LLaMA memakan memori 7 kali lipat lebih boros!).
*   **Kesimpulan Logis:** Angka ini memvalidasi secara empiris mengapa MLA jauh lebih superior dari GQA dalam hal penghematan representasi KV. Kemampuan ini membuat DeepSeek-V3 sangat ideal dan revolusioner untuk skenario yang melibatkan pemrosesan teks panjang (ratusan ribu kata) dan lingkungan perangkat keras yang terbatas secara anggaran.

---

*(Kembali ke kelanjutan metode kompresi KV lainnya di industri)*

*   **Windowed KV:** Untuk kalimat yang teramat panjang, metode ini hanya mempertahankan memori KV dari sebuah jendela berjalan (*sliding window*) kata-kata terakhir. Memori di luar jendela tersebut langsung dibuang (dianulir). Metode ini berhasil mereduksi penyimpanan, namun harus dibayar mahal dengan **rusaknya kemampuan model untuk bernalar tentang konteks panjang** (ia akan lupa apa instruksi di awal paragraf).
*   **Quantized Compression:** Memori pasangan KV dikompresi menggunakan representasi bit yang lebih rendah (*low-bit representations*) (misalnya, menjadi format 4-bit atau 2-bit), yang lebih jauh mereduksi penggunaan memori. Kuantisasi ini mampu meraih kompresi yang substansial dengan benturan (*impact*) yang sangat minim terhadap kecerdasan model.

#### 2.1.3 Arah Masa Depan dan Perspektif pada Teknik Efisien Sumber Daya
Walaupun mereduksi ukuran memori KV cache adalah metode yang menjanjikan, terdapat satu tantangan formidabel yang tersisa pada arsitektur berbasis Transformer: **Kompleksitas Kuadratik** (*quadratic complexity*). Proses *decoding* autoregresif menuntut perhitungan matematika yang kuadratik seiring bertambah panjangnya teks. Hal ini amat menyulitkan untuk konteks yang terlampau panjang. 

Upaya riset terkini, seperti **Mamba-2** dan **Lightning Attention**, menginvestigasi alternatif waktu-linear (*linear-time alternatives*), yang tidak kuadratik. Pendekatan ini menawarkan kemungkinan baru untuk menyeimbangkan antara harga komputasional dan performa kecerdasan model.

Selain itu, pendekatan lain seperti atensi renggang (*sparse attention*), yang mencari cara untuk memampatkan dan hanya mengaktivasi secara renggang Kunci (Keys) dan Nilai (Values) pada fungsi *attention*, merepresentasikan rute lain untuk menundukkan tantangan komputasi. Kami sangat menantikan progres kolaboratif dari komunitas global untuk menghasilkan terobosan di bidang ini.

---

### 2.2 Efektivitas Biaya Model MoE (*Cost-Effectiveness of MoE Models*)

Untuk mencapai komputasi yang renggang (*sparse computing*), kami telah mengembangkan **DeepSeekMoE**, sebuah arsitektur Campuran-Pakar (*Mixture of Experts*) tingkat lanjut (telah diilustrasikan di bagian kanan bawah Gambar 1 sebelumnya). Keuntungan dari model MoE terbagi menjadi dua aspek utama.

#### 2.2.1 Mereduksi Kebutuhan Komputasional untuk Pelatihan

Keunggulan terkuat dari arsitektur MoE terletak pada kemampuannya untuk **mereduksi biaya pelatihan secara radikal**. 

Secara mekanik, model ini hanya akan *mengaktifkan* sebagian kecil dari parameter "Pakar" (*experts*) saat mencerna sebuah token kata. Model MoE memungkinkan *Total Parameter* model untuk dikembangkan menjadi ukuran raksasa, *namun* beban kerja komputasional tetap dikunci pada level yang moderat. 

Sebagai contoh perbandingan absolut: **DeepSeek-V2** memiliki 236 Miliar parameter (ukuran yang cukup besar). Namun, dari 236B tersebut, **HANYA 21 Miliar (21B) parameter** yang disetrum daya komputasi dan diaktifkan untuk setiap tokennya. 
Secara serupa, **DeepSeek-V3** meledakkan ukurannya menjadi 671 Miliar parameter—hampir tiga kali lipat ukuran V2—namun tetap mempertahankan parameter yang aktif di angka moderat, yakni hanya **37 Miliar (37B)** parameter per token. 

Bandingkan dengan arsitektur Padat (*Dense Models*) klasik seperti Qwen2.5-72B dan LLaMa3.1-405B. Model berarsitektur *Dense* menuntut **seluruh (100%)** parameternya untuk menyala (aktif) selama masa pelatihan. 

### Analisis Akademis Tabel 2: Komparasi Biaya Komputasi MoE vs Dense

Mari kita lihat secara numerik berapa banyak "tenaga" (*GFLOPS*) yang dibakar untuk mengajari mesin-mesin ini.

**Tabel 2: Perbandingan biaya komputasi untuk pelatihan model MoE dan Padat (*Dense*): Biaya komputasi diukur per token, dengan asumsi panjang sekuens bacaan 4096.**

| Model | Ukuran Parameter | Biaya Pelatihan (*Training Cost*) |
| :--- | :--- | :--- |
| **DeepSeek-V2 (MoE)** | 236B | **155 GFLOPS/Token** |
| **DeepSeek-V3 (MoE)** | 671B | **250 GFLOPS/Token** |
| Qwen-72B (Dense) | 72B | 394 GFLOPS/Token |
| LLaMa-405B (Dense) | 405B | 2448 GFLOPS/Token |

*   **Deskripsi Data:** Tabel ini mencantumkan biaya komputasional (diukur dalam Giga FLoating Point Operations Per Second - GFLOPS) yang harus dikerahkan oleh GPU untuk melatih model mengevaluasi SATU token kata.
*   **Analisis Saintifik:** 
    *   Sebuah model *Dense* kecil seperti Qwen-72B (hanya 72 Miliar parameter) menuntut pembakaran **394 GFLOPS** per token. 
    *   Bandingkan dengan DeepSeek-V3. Ukurannya raksasa (671 Miliar parameter, nyaris sepuluh kali lipat Qwen), NAMUN ia hanya membakar **250 GFLOPS** per token! Ini karena ia adalah model MoE.
    *   Model *Dense* raksasa seperti LLaMa-405B? Ia memakan daya hisap mematikan hingga **2448 GFLOPS** per token. 
*   **Kesimpulan Logis:** Ini mengonfirmasi bahwa model MoE mampu mencapai kecerdasan yang sepadan—bahkan superior—dibandingkan model *dense*, namun ia menghisap sumber daya komputasi **satu orde magnitudo lebih kecil** (ratusan GFLOPS vs ribuan GFLOPS). Inilah pendefinisian efisiensi biaya yang sesungguhnya.

#### 2.2.2 Keuntungan untuk Penggunaan Pribadi dan *Deployment On-Premises*

Di masa depan di mana agen LLM personal (*personalized LLM agents*) menjadi suatu hal yang merakyat dan di mana-mana, model MoE menawarkan keunggulan ekstrem pada skenario permintaan-tunggal (*single-request scenarios*). Karena hanya sebagian kecil parameter yang "terbangun" di dalam otak model per satu panggilan (request), tuntutan memori RAM dan daya komputasi turun drastis. 

Sebagai contoh, **DeepSeek-V2** (ukuran utuh 236B parameter) hanya menyalakan 21B parameternya selama melayani (*inference*). Ini memungkinkan komputer Pribadi (PC) rumahan dengan cip SoC AI standar (yang memorinya terbatas) untuk mampu menjalankan model raksasa ini di kecepatan hampir 20 token per detik (TPS), atau bahkan berlari dua kali lipat lebih cepat. Kecepatan ini sudah sangat berlebih untuk keperluan membaca manusia secara personal. Sebagai kontras perbandingan, jika Anda memaksakan model *Dense* dengan kapasitas kecerdasan setara (misal, model Padat 70B parameter) di perangkat keras PC biasa, kecepatannya biasanya akan rontok hingga ke angka 1 digit TPS saja (sangat lambat dan *lagging*).

Hebatnya lagi, mesin inferensi sumber terbuka yang semakin populer, **KTransformers**, telah memungkinkan model komplit **DeepSeek-V3** (671B) dijalankan di sebuah komputer *server* murah menggunakan GPU kelas konsumen (*consumer GPU* seperti RTX 4090) (total harga *server* hanya sekitar $10.000), namun tetap bisa mencetak kecepatan hampir 20 TPS.

Efisiensi gila-gilaan ini menjadikan arsitektur MoE sangat cocok di-deploy secara lokal (*local deployments*) atau digunakan oleh pengguna perorangan (*single-user scenarios*), di mana anggaran untuk perangkat keras sangatlah minim. 

---

### 2.3 Peningkatan Kecepatan Inferensi (*Increasing Inference Speed*)

#### 2.3.1 Overlapping (Penumpangan) Komputasi dan Komunikasi: Memaksimalkan *Throughput*

Kecepatan inferensi mencakup dua metrik: hasil maksimum sistem secara utuh (*system-wide maximum throughput*) dan jeda respons permintaan tunggal (*single-request latency*). 

Untuk memaksimalkan jumlah antrean teks yang bisa diproses (*throughput*), model kita diarsitekturi sejak awal pelaksanaannya untuk memanfaatkan **penumpangan *micro-batch* ganda (*dual micro-batch overlap*)**. Strategi ini secara intensional menumpangkan (menjalankan bersamaan) waktu jeda komunikasi (*communication latency*) dengan waktu eksekusi komputasi (*computation*). 

Sebagaimana didemonstrasikan di sistem layanan daring (API) kita—dan didukung oleh data rekam jejak (*profiling*) publik kita—kami memisahkan (men- *decouple*) kalkulasi matematis MLA dan MoE ke dalam dua panggung (*stages*) yang berbeda. 
*   Mekanisme Paralelnya: Saat satu paket *micro-batch* data sedang menjalani perhitungan matematis (MLA/MoE) di cip GPU, paket *micro-batch* yang lain SECARA SIMULTAN menumpang di jalur jaringan untuk melakukan komunikasi antar-GPU (*dispatch communication*). 
*   Begitu pula sebaliknya, ketika *micro-batch* kedua masuk fase komputasi, *micro-batch* pertama baru masuk jalur komunikasi (*combine communication step*). 

Pendekatan jalur pipa (*pipelined approach*) ini memastikan terjadinya penumpangan (*overlap*) tanpa cacat antara komunikasi seluruh-ke-seluruh (*all-to-all communication*) jaringan dan proses kalkulasi GPU. Hal ini menjamin bahwa Unit Pemrosesan Grafis (GPU) tidak pernah *idle* (menganggur) menunggu paket data lewat jaringan, memastikan utilisasi/pemanfaatan perangkat keras 100% penuh sepanjang waktu. Selanjutnya, di level produksi, kami juga mengadopsi arsitektur pemisahan *prefill* (mencerna teks awal) dan *decode* (mengeluarkan teks jawaban), di mana kami menugaskan *request prefill* yang memiliki *batch size* raksasa dan *request decode* yang sensitif-waktu kepada grup GPU paralel yang berbeda-beda. Strategi perutean *traffic* pamungkas ini memeras *throughput* sistem hingga tetes terakhir di kondisi tekanan layanan dunia nyata.

#### 2.3.2 Batas Kecepatan Inferensi (*Inference Speed Limits*)

Seksi ini memfokuskan diri pada kecepatan pengeluaran huruf/token dari layanan LLM, yang umumnya diukur dengan satuan **Time Per Output Token (TPOT)** atau Waktu per Token Keluaran. 
TPOT adalah metrik teramat vital untuk pengalaman pengguna (*user experience*), dan ia juga berdampak brutal secara langsung terhadap responsivitas "Model Penalaran" (*reasoning models*) seperti seri o1/o3 milik OpenAI dan DeepSeek-R1. Kenapa? Karena model-model penalaran ini mengandalkan proses "bergumam panjang" (*long inference length*) di balik layar untuk merangkai logika dan menaikkan tingkat kejeniusannya. Jika mesinnya lambat mengeluarkan token (TPOT tinggi), model ini akan memakan waktu berjam-jam untuk memikirkan satu jawaban rumus fisika.

Pada model MoE, mencapai kecepatan inferensi (TPOT rendah) sangat bergantung pada kemampuan mesin untuk mengerahkan (*deploying*) parameter "pakar" secara efisien melintasi jaringan perangkat keras. 

**Analisis Kemacetan Perangkat Keras (*Hardware Bottleneck*):**
Untuk mencapai kecepatan cetak teks yang paling ekstrem, secara ideal setiap unit GPU harusnya bertugas menghitung kalkulasi untuk *satu* pakar saja (atau beberapa GPU saling patungan menghitung satu pakar). 
Namun, penggunaan **Paralelisme Pakar (*Expert Parallelism* / EP)** menuntut sistem untuk menembakkan (me- *routing*) token-token dari teks pengguna ke unit perangkat/GPU yang spesifik memiliki pakar tersebut. Proses ini mengharuskan terjadinya pertukaran data komunikasi **"semua-ke-semua" (*all-to-all communication*)** melintasi jalur kabel jaringan (*network switch*).

Sebagai konsekuensi mutlak hukum fisika komputer: **Batas atas kecepatan (*upper limit*) inferensi MoE dijepit (dibatasi) secara paksa oleh kecepatan lebar pita jaringan kabelnya (*interconnection bandwidth*)**.

*Kalkulasi Matematika Kecepatan Teoretis (Studi Kasus H800 & Jaringan IB):*
Mari kita bedah perhitungan matematis batas kecepatan ini. Bayangkan sebuah sistem peladen (server) di mana setiap perangkat GPU memegang bobot parameter untuk 1 (satu) pakar, dan sistem harus memproses sekitar 32 token dalam satu putaran waktu. Angka 32 token ini adalah angka sakti penyeimbang (*sweet spot*) antara rasio komputasi-ke-memori dengan latensi antrean komunikasi jaringan. Menetapkan batasan 32 token juga memastikan setiap GPU memproses beban kerja (*batch size*) yang sama persis selama operasi paralelisme pakar, sehingga waktu transfer jaringan dapat dihitung secara pasti.

Untuk sebuah klaster yang dihubungkan dengan kabel jaringan **CX7 400Gbps InfiniBand (IB) NICs**, waktu teoretis murni yang dibutuhkan untuk melakukan *dua* putaran komunikasi `all-to-all` (seperti dibahas di 2.3.1: *dispatch* dan *combine*) dalam Paralelisme Pakar (EP) dihitung dengan formula berikut:

$$ \text{Comm. Time} = \text{(1 Byte + 2 Bytes)} \times 32 \times 9 \times 7\text{K} \text{ / } 50\text{ GB/s} = 120.96 \, \mu\text{s} $$

**Bedah Komponen Rumus:**
*   **1 Byte + 2 Bytes:** Saat mengirim token (dispatch), dikirim dalam format angka sangat kecil FP8 (1 Byte). Saat menerima hasilnya (combine), dibaca dalam format angka yang lebih presisi BF16 (2 Bytes). Total beban data per token adalah 3 Bytes.
*   **32:** Jumlah token yang diproses.
*   **9:** Angka pengali ini mewakili arsitektur MoE kita. Setiap token dalam DeepSeek-V3 ditembakkan (disalurkan) ke 8 Pakar Terarah (*Routed Experts*) dan 1 Pakar Bersama (*Shared Expert*). Jadi ada 9 perpindahan.
*   **7K:** Merepresentasikan dimensi tersembunyi (*hidden size* $\approx 7000$) dari vektor masing-masing token.
*   **50 GB/s:** Batas atas (*throughput*) efektif dari kecepatan kabel jaringan kartu 400Gbps.
*   *(Perhatian Akademis: Kalkulasi ini adalah skenario utopia fisika. Di dunia nyata, masih ada "latency" ping-pong jaringan yang belum ditambahkan di perhitungan ini).*

Seperti yang sudah kita telusuri di Bagian 2.3.1, untuk memompa jumlah pekerjaan (*throughput*) ke level teratas, kita menggunakan strategi *dual micro-batch overlap*. Dalam skenario analisis terbaik (*best-case*) teoretis kita, kita berasumsi bahwa beban *overhead* komputasi (perhitungan GPU) berhasil disembunyikan sepenuhnya di balik proses pengiriman kabel (jaringan) yang lambat ini. Ini berarti performa maksimal dibatasi mutlak oleh *waktu transfer* (communication latency). 

Berdasarkan asumsi utopia penumpangan mikro-tumpak ganda ini, total waktu untuk memproses SATU lapisan arsitektur (satu layer) dapat diformulasikan menjadi:

$$ \text{Total Time Per Layer} = 2 \times 120.96 \, \mu\text{s} = 241.92 \, \mu\text{s} $$

Dikarenakan DeepSeek-V3 adalah sebuah "kue lapis" yang terdiri dari **61 lapisan (layers)** secara arsitektural, maka total waktu tembus (*total inference time*) untuk memuntahkan SATU token melewati seluruh lapisan adalah:

$$ \text{Total Inference Time} = 61 \times 241.92 \, \mu\text{s} = \mathbf{14.76 \, \text{ms}} $$

*   **Kesimpulan Sains Komputer:** Secara teoretis, mesin ini akan memuntahkan satu kata (*Time Per Output Token* / TPOT) setiap **14.76 milidetik**. Kecepatan ini ekuivalen dengan memberondongkan **67 token (kata) per detik**. Ini adalah batas atas kecepatan cahaya sistem tersebut!
*   *Namun, realitas selalu menyakitkan:* Di lantai produksi sesungguhnya, faktor-faktor polusi seperti hambatan paket (*communication overhead*), antrean latensi murni, saluran kabel yang tak 100% termanfaatkan, dan kelemahan *driver* *software* (inefisiensi komputasional) akan menyeret turun angka 67 token/detik ini menjadi lebih pelan.

*Skenario Utopia Masa Depan:*
Sebagai sebuah antitesis perbandingan, bayangkan jika suatu saat nanti kita menyambungkan sistem ini dengan kabel interkoneksi super-lebar layaknya arsitektur rak NVIDIA **GB200 NVL72** (yang memiliki pita searah gila-gilaan berkecepatan 900 GB/s yang mengikat 72 unit GPU). Waktu komunikasi jaringan untuk meloncati batas EP akan langsung terjun bebas menjadi:

$$ \text{Comm. Time} = \text{(1 Byte + 2 Bytes)} \times 32 \times 9 \times 7\text{K} \text{ / } \mathbf{900\text{ GB/s}} = \mathbf{6.72 \, \mu\text{s}} $$

Jika kita mengasumsikan penumpangan (overlap) super mulus antara komputasi GPU dan transmisi kabel ini terjadi, maka teoretis batas atas kelajuan TPOT kita akan menembus kecepatan supersonik **0.82 milidetik per token**. Ini ekuivalen dengan memberondongkan secara raksasa **1.200 token kata setiap detik!** (Satu buku kecil diselesaikan dalam kedipan mata).

*   *Sanggahan Ilmiah:* Tentu saja, angka 1.200 ini hanyalah kalkulator berteori (*purely theoretical*). Ia mengabaikan hukum fisika di mana GPU akan langsung kehilangan daya efisiensinya (*efficiency drop*) jika hanya disuruh mengerjakan kepingan *batch size* data yang mungil-mungil. Di lantai produksi aktual, lajunya tidak akan pernah sekencang itu. 
*   Meskipun demikian, kalkulasi hipotesis ini secara tajam memberikan demonstrasi visual (*transformative potential*) tentang betapa berkuasanya kabel interkoneksi raksasa (jaringan *scale-up* berskala *high-bandwidth*) dalam mengakselerasi pemikiran model kecerdasan buatan skala masif.

Meskipun model MoE menunjukkan skalabilitas yang sangat cantik, mengejar kecepatan inferensi tinggi *hanya* dengan sekadar membeli peranti keras tambahan yang terus membesar sangatlah mahal (*cost-prohibitive*) alias bakar uang. Oleh karena itu, kita para insinyur *software* dan pembuat algoritma HARUS ikut turun tangan dalam memeras efisiensi dari sisi algoritma inferensi.

#### 2.3.3 Modul Prediksi Multi-Token (*Multi-Token Prediction* / MTP)

Terilhami oleh kejeniusan Gloeckle et al. [36], tim DeepSeek-V3 menginjeksi sebuah kerangka kerja **Multi-Token Prediction (MTP)**. Keajaibannya? MTP bekerja membunuh dua ekor burung dengan satu batu: meningkatkan kecerdasan (*model performance*) sekaligus mendongkrak laju inferensi (*inference speed*) secara bersamaan.

Dalam paradigma inferensi autoregresif gaya lama (seperti ChatGPT versi awal), LLM hanya sanggup menebak dan memuntahkan **SATU token** pada satu siklus hitungan *decoding*. Hal ini menciptakan antrean panjang yang mencekik (hambatan sekuensial / *sequential bottlenecks*). Algoritma MTP memberantas penyakit leher botol ini dengan memberikan kekuatan super pada model untuk memprediksi **beberapa token tambahan (kandidat) sebagai bonus** dengan ongkos memori yang amat murah, lalu memverifikasi ketepatan tebakan-tebakan bonus tersebut secara paralel (bersamaan). Teknik pemangkas waktu ini menyerupai rute pintas pendekatan dekoding spekulatif (*speculative decoding*) yang mengandalkan pengerjaan draf mandiri [14, 48]. Kerangka kerja ini secara signifikan mengakselerasi inferensi tanpa mengorbankan akurasi secuil pun.

*(Merujuk kembali pada penganalisaan blok kanan atas Gambar 1 di awal bab)*:
*   Setiap modul kotak MTP beroperasi hanya memakai matriks **satu lapisan tunggal (*single layer*)**, yang menjadikannya luar biasa ringan (*lightweight*) jika dikontraskan dengan beban komputasi "Model Utama" secara keseluruhan. Modul ringan ini meramal kata-kata di masa depan, memicu verifikasi multi-kata secara paralel.
*   Meskipun beban kerja memproses banyak tebakan secara paralel ini *sedikit* menyakiti kelajuan batas *throughput* kotor sistem, sebagai ganti rugi (imbal baliknya), pendekatan ini malah berhasil secara **masif** membantai dan memenggal (*significantly improves*) durasi waktu yang ditunggu pengguna untuk melihat huruf pertama muncul (*end-to-end generation latency*).
*   Data di lapangan produksi (Dunia Nyata) membuktikan secara fenomenal bahwa modul MTP kami mencapai tingkat "Tebakan Bonus Benar" (*acceptance rate*) yang berada di kisaran fantastis **80% hingga 90%** untuk menembak tepat token posisi *kedua*!
*   Alhasil, daya dobrak TPS (*Tokens Per Second*) dalam mencetak esai teks menyembur kencang melesat hingga **1,8 kali lipat (1.8x)** lebih gesit daripada sistem tradisional yang lumpuh tanpa mesin MTP.

Nilai tambah radikal dari strategi ini: dengan memaksa mesin meramal rentetan beberapa token setiap langkahnya, MTP otomatis memperbesar bundelan "tumpukan kerja" (*inference batch size*). Di dunia *expert parallelism* (EP), ukuran bongkahan *batch size* yang gendut adalah syahwat utama (*crucial*) yang diperlukan untuk memaksimalkan intensitas hisapan komputasi dan melahap utilisasi (pemanfaatan) maksimal perangkat keras *hardware*. Inovasi-inovasi algoritmik semacam inilah yang menjadi urat nadi pendorong lahirnya layanan inferensi yang buas namun murah secara kapital di DeepSeek-V3.

#### 2.3.4 Kebutuhan Kecepatan Inferensi Tinggi untuk Model Penalaran dan Penskalaan Waktu Pengujian (*High Inference Speed for Reasoning Models and Test-Time Scaling*)

Konsep Penskalaan Waktu-Pengujian (*Test-time scaling*) pada LLM—yang dieksemplifikasi secara magis oleh seri model **o1/o3** milik OpenAI [60, 61]—telah meletupkan sebuah ledakan revolusi kapabilitas intelektual AI, khususnya dalam membedah rasionalitas penalaran matematika, pemecahan masalah algoritma (pemrograman), serta kognisi deduksi umum. Caranya sederhana namun mematikan: **AI diizinkan secara dinamis menyedot sumber daya komputasi dan "berpikir lebih lama/menggumam" selama masa eksekusi tes (*inference*)**. 

Pola paradigma model o1 ini dengan cepat menjangkiti silsilah model-model berikutnya—merangkum nama-nama besar layaknya **DeepSeek-R1** [28], **Claude-3.7 Sonnet** [9], **Gemini 2.5 Pro** [38], **Seed1.5-Thinking** [68], dan **Qwen3** [72]—yang lantas mengadopsi taktik "berpikir diam-diam di belakang layar" serupa, dan terbukti sanggup mencapai loncatan peningkatan intelegensi (skor IQ buatan) yang merusak batas kewajaran pada ujian-ujian akademis yang mengerikan.

Bagi model-model spesies penalaran mendalam ini, kecepatan muntahan mesin (token per detik) adalah **Dewa Tertinggi (*of paramount importance*)**. Kenapa?

1.  **Hambatan RL (Reinforcement Learning) di Dapur Pelatihan:** Dalam alur pipa penciptaan AI jenis ini—yang sangat bertumpu padateknologi RL purba seperti **PPO** [67], **DPO** [64], dan metode gubahan kami **GRPO** [69]—kebutuhan sistem secara mutlak menuntut kemampuan model *actor* (agen yang ditest) untuk "bermain-main" meramalkan ratusan ribu hingga jutaan contoh solusi (*samples*) dengan kecepatan yang menyamai letupan peluru dalam waktu singkat. Jika kelajuan laju mesin keluarannya (*inference throughput*) lelet, pelatihannya akan tersendat menjadi rantai macet krisis yang mencekik napas (*critical bottleneck*). Tanpa kecepatan gila, kita mustahil bisa melatih AI pemikir.
2.  **Frustasi Pengguna di Dunia Nyata:** Bagi konsumen akhir, sekuens barisan logika gumaman internal AI yang memanjang hingga berlembar-lembar halaman layar berarti meningkatkan angka jam meter "waktu tunggu pengguna" (*user wait times*). Logika kaku: Model sejenius apapun akan ditinggalkan manusia pemakai jika harus memutar kursor *loading* selama dua jam hanya untuk meresolusi persamaan diferensial. Alhasil, kecepatan lambat akan membunuh kegunaan pragmatis / *useability* komersial dari model ini di pasar korporat.

Sebagai kesimpulan akademis, mengoptimalkan daya dobrak kecepatan inferensi melalui sinergi radikal dan inovasi persilangan arsitektur peranti keras (*hardware*) dengan peranti lunak algoritmik (*software*) adalah syarat absolut (TIDAK BISA DITAWAR / *indispensable*) untuk terus membakar roda laju evolusi kemajuan pada model penalaran AI (*reasoning models*). 

Meskipun demikian, metode dan taktik efektif untuk menakselerasi mesin kecepatan membaca AI ini sekaligus meroketkan akselerasi putaran pelatihan algoritma RL masih berada dalam palung kawasan riset yang masih panas bergolak dieksplorasi (dibahas sebagai fokus wacana ke depan di Seksi 2.1.3). Kami melemparkan tantangan terbuka, menyemangati dan mengundang komunitas akademik maupun periset raksasa teknologi (*broader community*) untuk dengan antusias berkumpul, meramu, mengeksplorasi secara kolegial, serta merekayasa penemuan revolusioner mutakhir agar sanggup membongkar limitasi kebuntuan sains komputer raksasa di zaman ini.

*(Demikian akhir dari modul Diktat Kuliah Analisis Arsitektur Kinerja Tinggi AI dari makalah DeepSeek-V3 Hardware Reflections. Mahasiswa diharap menelaah dengan saksama perhitungan matematis keterbatasan bandwitdh dalam rancang bangun sistem MoE berskala masif).*
