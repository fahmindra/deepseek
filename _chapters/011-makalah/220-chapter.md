---
slug: ch-11-15
title: "DeepSeek-V3.2"
layout: chapter
---

## DeepSeek-V3.2: Mendorong Batas Kemampuan Model Bahasa Besar Sumber Terbuka

Dalam bab ini, kita akan mengkaji **DeepSeek-V3.2**, sebuah model yang berhasil menyelaraskan efisiensi komputasi tinggi dengan kinerja penalaran (*reasoning*) dan agen (*agent*) yang superior. Terobosan teknis utama dari DeepSeek-V3.2 meliputi tiga pilar berikut:

1.  **Atensi Renggang DeepSeek (*DeepSeek Sparse Attention / DSA*):** Diperkenalkan mekanisme DSA, sebuah arsitektur atensi efisien yang secara substansial mengurangi kompleksitas komputasi sembari mempertahankan kinerja model pada skenario konteks yang panjang (*long-context scenarios*).
2.  **Kerangka Kerja Pembelajaran Penguatan yang Dapat Diskalakan (*Scalable Reinforcement Learning Framework*):** Dengan menerapkan protokol Pembelajaran Penguatan (*Reinforcement Learning / RL*) yang tangguh dan menskalakan komputasi pasca-pelatihan (*post-training compute*), DeepSeek-V3.2 mampu berkinerja setara dengan GPT-5. Secara khusus, varian komputasi tinggi kita, **DeepSeek-V3.2-Speciale**, berhasil melampaui GPT-5 dan menunjukkan kemahiran penalaran yang sepadan dengan Gemini-3.0-Pro. Model ini bahkan meraih performa medali emas pada ajang *International Mathematical Olympiad* (IMO) 2025 dan *International Olympiad in Informatics* (IOI) 2025.
3.  **Saluran Sintesis Tugas Agenik Skala Besar (*Large-Scale Agentic Task Synthesis Pipeline*):** Untuk mengintegrasikan kemampuan penalaran ke dalam skenario penggunaan alat bantu (*tool-use scenarios*), dikembangkan sebuah saluran sintesis baru yang secara sistematis menghasilkan data pelatihan dalam skala besar. Metodologi ini memfasilitasi pasca-pelatihan agenik yang dapat diskalakan, menghasilkan peningkatan substansial dalam hal generalisasi dan ketangguhan mengikuti instruksi (*instruction-following robustness*) di dalam lingkungan yang interaktif dan kompleks.

---

### Analisis Akademik Gambar 1: Kinerja Tolok Ukur DeepSeek-V3.2

Sebelum kita masuk ke dalam teori yang lebih mendalam, mari kita analisis **Gambar 1** yang merangkum hasil evaluasi (benchmark) dari model ini.

*   **Deskripsi Visual:** Gambar 1 adalah sebuah diagram batang berkelompok (*grouped bar chart*) dengan sumbu Y sekunder. Sumbu X merepresentasikan berbagai tolok ukur pengujian, mulai dari AIME 2025, HMMT 2025, HLE, Codeforces, SWE Verified, Terminal Bench 2.0, $\tau^2$ Bench, hingga Tool Decathlon.
*   **Sumbu Y Primer (Kiri):** Mengukur tingkat keberhasilan atau *Accuracy / Pass@1 (%)*, mulai dari 0 hingga 100.
*   **Sumbu Y Sekunder (Kanan):** Mengukur *Codeforces Rating*, skala khusus untuk tolok ukur Codeforces, mulai dari 0 hingga 3000.
*   **Kategori (Sumbu X Bawah):** Pengujian dibagi menjadi dua blok kapabilitas besar: *Reasoning Capabilities* (Kemampuan Penalaran) dan *Agentic Capabilities* (Kemampuan Agenik).
*   **Model yang Dikomparasi (Legenda):**
    *   Biru Tua (*Dark Blue*): DeepSeek-V3.2-Speciale
    *   Biru Muda (*Light Blue*): DeepSeek-V3.2-Thinking
    *   Abu-abu (*Grey*): GPT-5-High
    *   Hijau (*Green*): Claude-4.5-Sonnet
    *   Kuning Keputihan (*Light Yellow*): Gemini-3.0-Pro
*   **Kesimpulan Akademis:** 
    *   Pada ranah *Reasoning Capabilities* (AIME, HMMT), **DeepSeek-V3.2-Speciale** (biru tua) secara konsisten mendominasi, mencapai *pass rate* hampir sempurna (96.0% di AIME dan 99.2% di HMMT). Kinerja ini bahkan melampaui GPT-5-High (abu-abu). Pada Codeforces, model ini meraih rating menembus 2701, setara dengan *Grandmaster*.
    *   Pada ranah *Agentic Capabilities* (SWE Verified, Terminal Bench, dll.), DeepSeek-V3.2-Thinking (biru muda) menunjukkan daya saing yang luar biasa, seringkali menduduki posisi pertama atau kedua, membuktikan bahwa kemampuannya menggunakan alat (*tool-use*) sudah berada di garis depan industri AI, bersaing ketat dengan Claude-4.5-Sonnet dan Gemini-3.0-Pro. *Catatan metodologi: Untuk HMMT 2025, data yang dilaporkan adalah dari kompetisi Februari agar konsisten dengan baseline. Untuk HLE, yang dilaporkan adalah subset text-only.*

---

## 1. Pendahuluan

Peluncuran model-model penalaran (DeepSeek-AI, 2025; OpenAI, 2024a) menandai momen pivotal dalam evolusi Model Bahasa Besar (LLMs), yang mengatalisasi lompatan substansial dalam kinerja keseluruhan melintasi bidang-bidang yang dapat diverifikasi (*verifiable fields*). Sejak tonggak sejarah ini, kapabilitas LLM telah maju dengan sangat pesat. 

Namun, divergensi (perbedaan) yang mencolok telah muncul dalam beberapa bulan terakhir. Di saat komunitas sumber terbuka (MiniMax, 2025; MoonShot, 2025; Qwen, 2025; ZhiPu-AI, 2025) terus membuat kemajuan, lintasan performa dari model kepemilikan sumber tertutup (*closed-source proprietary models*) (Anthropic, 2025b; DeepMind, 2025a; OpenAI, 2025) justru telah berakselerasi dengan tingkat yang jauh lebih curam. Akibatnya, alih-alih menyatu (*converging*), kesenjangan kinerja antara model tertutup dan model sumber terbuka tampaknya semakin melebar, di mana sistem eksklusif menunjukkan kapabilitas yang semakin superior pada tugas-tugas yang kompleks.

Melalui analisis yang kami lakukan, kami mengidentifikasi **tiga defisiensi kritis** yang membatasi kapabilitas model sumber terbuka dalam tugas-tugas kompleks:
1.  **Arsitektural:** Ketergantungan yang dominan pada mekanisme atensi vanilla (*vanilla attention*) (Vaswani et al., 2017) sangat membatasi efisiensi untuk urutan teks yang panjang (*long sequences*). Ketidakefisienan ini menimbulkan hambatan substansial baik untuk *deployment* (penerapan) yang dapat diskalakan maupun untuk pelatihan pasca (*post-training*) yang efektif.
2.  **Alokasi Sumber Daya:** Model sumber terbuka menderita karena investasi komputasi yang tidak memadai selama fase pasca-pelatihan, yang sangat membatasi kinerjanya pada tugas-tugas yang sulit (*hard tasks*).
3.  **Konteks Agen AI:** Model sumber terbuka menunjukkan kelambatan yang nyata dalam generalisasi dan kemampuan mengikuti instruksi dibandingkan dengan rekan tertutup mereka (EvalSys, 2025; Li et al., 2025; Luo et al., 2025), sehingga menghambat keefektifannya dalam penerapan dunia nyata (*real deployment*).

Untuk mengatasi keterbatasan-keterbatasan kritis ini, kami merancang serangkaian solusi inovatif:
1.  Pertama, kami memperkenalkan **DSA (*DeepSeek Sparse Attention*)**, sebuah mekanisme atensi yang sangat efisien yang dirancang untuk mereduksi kompleksitas komputasional secara substansial. Arsitektur ini secara efektif memecahkan hambatan efisiensi (*efficiency bottleneck*), sembari menjaga kinerja model bahkan dalam skenario konteks panjang.
2.  Kedua, kami mengembangkan **protokol Pembelajaran Penguatan (RL)** yang stabil dan dapat diskalakan yang memungkinkan ekspansi komputasional secara signifikan selama fase pasca-pelatihan. Patut dicatat, kerangka kerja ini mengalokasikan anggaran komputasi pasca-pelatihan **melebihi 10%** dari total biaya prapelatihan (*pre-training*), guna membuka (*unlock*) kapabilitas lanjutan.
3.  Ketiga, kami mengusulkan sebuah saluran (*pipeline*) baru untuk membina penalaran yang dapat digeneralisasi dalam skenario penggunaan alat bantu (*tool-use scenarios*). Awalnya, kami mengimplementasikan fase *cold-start* (mulai dingin) memanfaatkan metodologi DeepSeek-V3 (DeepSeek-AI, 2024) untuk menyatukan penalaran dan penggunaan alat di dalam satu lintasan tunggal (*single trajectories*). Selanjutnya, kami maju ke sintesis tugas agenik skala besar, di mana kami menghasilkan lebih dari 1.800 lingkungan yang berbeda dan 85.000 *prompts* kompleks. Data sintesis yang ekstensif ini menggerakkan proses RL, yang secara signifikan meningkatkan kemampuan generalisasi model dan pematuhan instruksi dalam konteks agen.

Hasilnya, **DeepSeek-V3.2** mencapai performa yang setara dengan Kimi-k2-thinking dan GPT-5 melintasi berbagai tolok ukur penalaran. Lebih jauh, DeepSeek-V3.2 secara signifikan memajukan kapabilitas agenik model terbuka, mendemonstrasikan kecakapan luar biasa pada tugas agen berekor panjang (*long-tail agent tasks*) yang diperkenalkan pada EvalSys (2025); Li et al. (2025); Luo et al. (2025). 

DeepSeek-V3.2 muncul sebagai alternatif yang sangat efisien secara biaya dalam skenario agen, mempersempit secara tajam kesenjangan kinerja antara model terbuka dan model kepemilikan garis depan (*frontier proprietary models*) sekaligus memakan biaya yang jauh lebih rendah. Yang tidak kalah penting, dengan tujuan mendobrak batas-batas model terbuka pada domain penalaran, kami melonggarkan batasan panjang (*length constraints*) untuk mengembangkan varian **DeepSeek-V3.2-Speciale**. Alhasil, DeepSeek-V3.2-Speciale mencapai tingkat keseimbangan performa (*performance parity*) dengan sistem tertutup terkemuka, Gemini-3.0-Pro (DeepMind, 2025b). Model ini bahkan menorehkan kinerja medali emas pada IOI 2025, ICPC World Final 2025, IMO 2025, dan CMO 2025.

---

## 2. Arsitektur DeepSeek-V3.2

### 2.1. Atensi Renggang DeepSeek (*DeepSeek Sparse Attention / DSA*)

DeepSeek-V3.2 sejatinya menggunakan arsitektur yang sama persis dengan DeepSeek-V3.2-Exp. Bila dibandingkan dengan DeepSeek-V3.1-Terminus (versi terakhir dari DeepSeek-V3.1), satu-satunya modifikasi arsitektur pada DeepSeek-V3.2 adalah pengenalan Atensi Renggang DeepSeek (*DeepSeek Sparse Attention / DSA*) yang dicapai melalui pelatihan lanjutan (*continued training*).

#### Purwarupa DSA (*Prototype of DSA*)
Secara konseptual, purwarupa dari DSA utamanya terdiri dari dua komponen: sebuah "pengindeks kilat" (*lightning indexer*) dan sebuah mekanisme "seleksi token berbutir halus" (*fine-grained token selection mechanism*).

1.  **Pengindeks Kilat (*Lightning Indexer*):** Modul ini menghitung skor indeks $I_{t,s}$ antara token kueri (*query token*) $\mathbf{h}_t \in \mathbb{R}^d$ dan token pendahulu (*preceding token*) $\mathbf{h}_s \in \mathbb{R}^d$. Skor ini menentukan token masa lalu mana yang relevan untuk diseleksi oleh token kueri saat ini. Formulasinya adalah sebagai berikut:

$$ I_{t,s} = \sum_{j=1}^{H^I} w_{t,j}^I \cdot \text{ReLU} \left( \mathbf{q}_{t,j}^{I} \cdot \mathbf{k}_s^{I} \right), \tag{1} $$

**Penjelasan Variabel Persamaan (1):**
*   $I_{t,s}$: Adalah skor indeks (nilai relevansi) antara token saat ini $t$ (kueri) dengan token historis $s$ (kunci). Semakin besar nilainya, semakin relevan token tersebut untuk diingat.
*   $H^I$: Merepresentasikan jumlah kepala (*heads*) pada pengindeks (*indexer*). Berbeda dengan *attention heads* biasa, ini adalah kepala khusus untuk tahap penyeleksian awal.
*   $\mathbf{q}_{t,j}^{I} \in \mathbb{R}^{d^I}$: Adalah vektor representasi kueri untuk kepala ke-$j$ pada waktu $t$. Vektor ini diturunkan (diproyeksikan) dari token kueri $\mathbf{h}_t$.
*   $\mathbf{k}_s^{I} \in \mathbb{R}^{d^I}$: Adalah vektor representasi kunci untuk token pendahulu pada waktu $s$. Vektor ini diturunkan dari token historis $\mathbf{h}_s$.
*   $w_{t,j}^I \in \mathbb{R}$: Adalah bobot skalar (*scalar weight*) yang diturunkan dari token kueri $\mathbf{h}_t$, berfungsi mengatur seberapa penting kepala indeks ke-$j$ pada langkah $t$.
*   $\text{ReLU}$: Fungsi aktivasi *Rectified Linear Unit*. Kami memilih ReLU dengan pertimbangan semata-mata untuk memaksimalkan *throughput* (kecepatan). 

Karena *lightning indexer* ini hanya memiliki jumlah kepala yang sangat sedikit dan dapat diimplementasikan menggunakan presisi data FP8 (8-bit), efisiensi komputasionalnya terbukti sangat luar biasa.

2.  **Mekanisme Seleksi Token (*Fine-grained Token Selection*):** 
Setelah skor indeks $\{I_{t,s}\}$ didapatkan untuk setiap token kueri $\mathbf{h}_t$, mekanisme ini bekerja dengan *hanya* mengambil entri kunci-nilai (*key-value entries*) $\{\mathbf{c}_s\}$ yang berkorespondensi dengan nilai top-k tertinggi dari skor indeks tersebut.

Setelah token historis diseleksi, keluaran atensi (*attention output*) $\mathbf{u}_t$ dikomputasi dengan mengaplikasikan mekanisme atensi standar di antara token kueri $\mathbf{h}_t$ dan kumpulan entri kunci-nilai yang telah diseleksi secara renggang (*sparsely selected*) $\{\mathbf{c}_s\}$:

$$ \mathbf{u}_t = \text{Attn} \left( \mathbf{h}_t, \left\{ \mathbf{c}_s \,|\, I_{t,s} \in \text{Top-k}(I_{t,:}) \right\} \right). \tag{2} $$

**Penjelasan Variabel Persamaan (2):**
*   $\mathbf{u}_t$: Adalah vektor keluaran final dari fungsi atensi untuk token $t$.
*   $\text{Attn}(\dots)$: Fungsi standar mekanisme atensi (menghitung *softmax* dari perkalian titik Q dan K, lalu dikalikan dengan V).
*   $\mathbf{h}_t$: Token kueri saat ini.
*   $\{\mathbf{c}_s \,|\, \dots\}$: Ini adalah filter *Sparsity* (Kerenggangan). Ia menginstruksikan bahwa atensi TIDAK menghitung jarak terhadap seluruh elemen masa lalu, MELAINKAN HANYA terhadap himpunan $\{\mathbf{c}_s\}$ (entri kunci-nilai laten) yang skor indeksnya ($I_{t,s}$) berhasil masuk ke dalam jajaran **Top-k** tertinggi dari seluruh skor indeks pada waktu $t$ ($I_{t,:}$).

#### Menginstansiasi DSA di Bawah MLA (*Instantiate DSA Under MLA*)
Demi mengakomodasi pelatihan lanjutan yang bertolak dari DeepSeek-V3.1-Terminus, kami menginstansiasi (menerapkan) DSA ke dalam fondasi arsitektur **Multi-head Latent Attention (MLA)** (DeepSeek-AI, 2024). 

Di tingkat kernel perangkat keras (GPU), agar komputasi menjadi efisien, setiap entri kunci-nilai *harus* dibagikan (*shared*) melintasi berbagai kueri (Yuan et al., 2025). Oleh sebab itu, kami mengimplementasikan DSA dengan bersandar pada mode **Multi-Query Attention (MQA)** (Shazeer, 2019) dari arsitektur MLA. Dalam mode MQA ini, setiap vektor laten (*latent vector* - yang merepresentasikan entri kunci-nilai dari MLA) akan digunakan secara bersama-sama (dibagikan) oleh seluruh kepala kueri (*query heads*) dari token kueri tersebut. 

*(Sebagai tambahan, arsitektur DSA yang berbasis pada MLA ini diilustrasikan pada Gambar 2 di dalam paper asli. Kami juga menyediakan implementasi open-source dari DeepSeek-V3.2 untuk mendefinisikan seluruh kerumitan teknis ini tanpa ambiguitas).*

---

### 2.1.1 Prapelatihan Lanjutan (*Continued Pre-Training*)

Kita bermula dari *base checkpoint* (titik penyimpanan) DeepSeek-V3.1-Terminus. Model ini telah kami perluas kapasitas memori panjang konteksnya (*context length*) agar mencapai angka raksasa **128K token**. Dari titik tersebut, kami melakukan Prapelatihan Lanjutan (*continued pre-training*) yang selanjutnya diikuti oleh tahapan Pasca-Pelatihan (*post-training*) untuk menetaskan DeepSeek-V3.2.

Fase prapelatihan lanjutan untuk DeepSeek-V3.2 ini dipecah ke dalam dua tahapan pelatihan. Untuk kedua tahapan tersebut, komposisi distribusi data latihnya diselaraskan sepenuhnya dengan data yang digunakan saat proses ekstensi konteks 128K pada DeepSeek-V3.1-Terminus.

#### Tahap 1: Pemanasan Padat (*Dense Warm-up Stage*)
Tahap ini adalah fase pemanasan (*warm-up*) singkat yang dikhususkan hanya untuk menginisialisasi modul *lightning indexer*. 
*   **Prosedur:** Pada tahap ini, seluruh jaringan atensi padat (*dense attention*) tetap dibiarkan berjalan normal. Kami membekukan (*freeze*) seluruh parameter model KECUALI parameter milik *lightning indexer* itu sendiri. 
*   **Objektif:** Tujuannya adalah untuk "mengajari" *indexer* ini agar *output*-nya sejalan dengan distribusi probabilitas dari fungsi atensi utama. 
*   Untuk token kueri ke-$t$, kami mengumpulkan skor atensi utama dengan menjumlahkannya di semua kepala atensi. Jumlah ini kemudian dinormalisasi dengan metode L1 sepanjang dimensi sekuens untuk menghasilkan sebuah **distribusi target** $p_{t,:} \in \mathbb{R}^t$.
*   Berdasarkan distribusi target $p_{t,:}$ ini, kami menetapkan kerugian divergensi Kullback-Leibler (*KL-divergence loss*) sebagai target objektif pelatihan dari *indexer*:

$$ \mathcal{L}^I = \sum_t \text{DKL} \left( p_{t,:} \,\big\|\, \text{Softmax}(I_{t,:}) \right). \tag{3} $$

**Penjelasan Variabel Persamaan (3):**
*   $\mathcal{L}^I$: Fungsi kerugian (*Loss function*) spesifik untuk melatih *Indexer*.
*   $\text{DKL}(\dots)$: Menghitung divergensi Kullback-Leibler, yaitu metrik matematis untuk mengukur seberapa "beda" dua buah distribusi probabilitas. Targetnya adalah mengecilkan divergensi ini ke angka 0.
*   $p_{t,:}$: Distribusi probabilitas target "ideal" yang disalin dari fungsi atensi penuh (*dense attention*).
*   $\text{Softmax}(I_{t,:})$: Distribusi probabilitas prediksi yang dihasilkan oleh *lightning indexer* berdasarkan skor indeks yang dikalkulasi pada Persamaan (1). 

Untuk tahap *warm-up* ini, kami menggunakan laju pembelajaran (*learning rate*) sebesar $10^{-3}$. *Indexer* ini dilatih dengan sangat kilat, hanya selama **1000 langkah (*steps*)**. Setiap langkah memuat 16 urutan teks, masing-masing dengan panjang 128K token. Total, fase ini mencerna **2.1 Miliar (2.1B) token**.

#### Tahap 2: Tahap Pelatihan Renggang (*Sparse Training Stage*)
Setelah *indexer* cukup "panas", kami beralih menghidupkan mekanisme "seleksi token berbutir halus". Di tahap inilah kita **mencairkan (unfreeze) dan mengoptimalkan seluruh parameter model** agar beradaptasi dengan pola renggang (*sparse pattern*) dari arsitektur DSA.

Pada tahap ini, kami tetap mempertahankan objektif untuk menyelaraskan keluaran *indexer* dengan distribusi atensi utama. NAMUN, penyelarasan ini sekarang dibatasi (dikalkulasi) HANYA pada **Himpunan Token Terseleksi ($S_t$)**:

$$ S_t = \left\{ s \,|\, I_{t,s} \in \text{Top-k}(I_{t,:}) \right\} $$

Maka, fungsi kerugiannya dimodifikasi menjadi:

$$ \mathcal{L}^I = \sum_t \text{DKL} \left( p_{t,S_t} \,\big\|\, \text{Softmax}(I_{t,S_t}) \right). \tag{4} $$

**Penjelasan Variabel Persamaan (4):**
*   Sangat mirip dengan Persamaan 3, perbedaannya hanya pada subskrip $S_t$. Kalkulasi DKL sekarang tidak lagi dihitung atas keseluruhan panjang teks ($1 \dots t$), melainkan hanya dievaluasi secara efisien pada himpunan token-token historis yang lolos seleksi filter Top-k ($S_t$). Ini menghemat komputasi secara masif selama pelatihan.

**Catatan Implementasi Penting:** Kami secara tegas memutuskan (*detach*) input *indexer* dari graf komputasi utama (*computational graph*) untuk keperluan optimasi yang terpisah.
*   Sinyal pelatihan gradien untuk memutakhirkan *indexer* MURNI hanya berasal dari kerugian $\mathcal{L}^I$.
*   Sinyal optimisasi untuk "Model Utama" murni dikalkulasi hanya berdasarkan kerugian pemodelan bahasa standar (*language modeling loss*).

Dalam tahapan pelatihan renggang ini, kami menggunakan laju pembelajaran yang jauh lebih kecil, yakni $7.3 \times 10^{-6}$. Filter diatur sedemikian rupa sehingga ia akan menyeleksi tepat **2048 token kunci-nilai (*key-value tokens*)** untuk setiap token kueri. Kami menjalankan pelatihan untuk model utama dan *indexer* ini selama 15.000 langkah. Setiap langkahnya menampung beban raksasa berupa 480 urutan teks berpanjang 128K token. Secara total, fase ini melahap **943.7 Miliar (943.7B) token**.

---

### 2.2 Evaluasi Kesetaraan (*Parity Evaluation*)

Pertanyaan terbesarnya adalah: *Apakah pemotongan komputasi menjadi "Renggang" ini merusak akurasi otak model?* 

**1. Tolok Ukur Standar (*Standard Benchmark*)**
Pada bulan September 2025, kami mengevaluasi DeepSeek-V3.2-Exp dengan merentangkan uji coba melintasi seperangkat tolok ukur yang berfokus pada kapabilitas yang beragam, dan membandingkannya secara langsung dengan versi DeepSeek-V3.1-Terminus (model atensi penuh).
*Hasilnya:* Keduanya menunjukkan performa yang identik! Walaupun DeepSeek-V3.2 Exp telah memangkas habis kompleksitas komputasi pada urutan yang panjang, **kami tidak mengamati adanya degradasi performa yang substansial** dibandingkan dengan DeepSeek-V3.1-Terminus, baik pada tugas berkonteks pendek maupun berkonteks sangat panjang.

**2. Preferensi Manusia (*Human Preference*)**
Mengingat bahwa penilaian "Preferensi Manusia" secara langsung rentan terhadap bias, kami mempekerjakan platform **ChatbotArena** sebagai kerangka evaluasi tidak langsung. ChatbotArena memungkinkan kami memperkirakan preferensi pengguna nyata (*user preferences*) terhadap model dasar yang baru ini. 
*Hasilnya:* DeepSeek-V3.1-Terminus dan DeepSeek-V3.2-Exp (yang dilatih dengan strategi *post-training* yang persis sama) memperoleh skor Elo yang sangat berdekatan (evaluasi per 10 November 2025). Kesimpulan ini menyahihkan klaim bahwa model baru mencapai performa yang **sepadan (on par)** dengan iterasi penuh pendahulunya, walaupun kini ia dimotori oleh mesin atensi renggang (*sparse attention*).

**3. Evaluasi Konteks Panjang (*Long Context Eval*)**
Beberapa entitas independen melakukan evaluasi konteks panjang terhadap DeepSeek-V3.2-Exp. 
*Hasilnya:* Pada *benchmark* representatif seperti AA-LCR, V3.2-Exp mencetak skor 4 poin lebih tinggi dari V3.1-Terminus pada mode penalaran. Pada evaluasi Fiction.liveBench, model ini juga menang secara konsisten melintasi banyak metrik. Ini adalah bukti sahih bahwa injeksi titik pemeriksaan dasar (*base checkpoint*) pada DeepSeek-V3.2-Exp sama sekali tidak mengalami kemunduran kemampuan membaca dokumen panjang.

---

### 2.3 Biaya Inferensi (*Inference Costs*)

Inilah keajaiban sesungguhnya dari algoritma ini secara matematis. Arsitektur DSA telah sukses mereduksi kompleksitas komputasi atensi dari model utama yang sebelumnya memakan biaya kuadratik $\mathcal{O}(L^2)$ menjadi biaya linier $\mathcal{O}(L \cdot k)$.

**Penjelasan Kompleksitas:**
*   $\mathcal{O}(L^2)$: Kompleksitas atensi konvensional (Padat). Jika teks bertambah panjang menjadi $L$, maka komputasi perkalian matriks akan membengkak secara kuadrat (Pangkat 2). Ini yang membuat GPU cepat panas dan kehabisan memori.
*   $\mathcal{O}(L \cdot k)$: Kompleksitas atensi DSA. Karena kita membatasi jumlah referensi sejarah pada angka konstan $k$ (dimana $k \ll L$), maka biaya komputasi hanya tumbuh secara proporsional (linier).

Memang benar, *lightning indexer* yang digunakan untuk menyeleksi top-k tersebut masih harus menyapu seluruh data dengan kompleksitas kuadratik $\mathcal{O}(L^2)$. TAPI, karena jumlah parameter pada *indexer* ini sangat minim, beban komputasinya sesungguhnya teramat ringan jika dibandingkan dengan atensi laten penuh (MLA) pada DeepSeek-V3.1-Terminus.

Digabungkan dengan implementasi kode tingkat-kernel yang telah kami optimasi, DSA memberikan akselerasi ujung-ke-ujung (*end-to-end speedup*) yang luar biasa pada skenario konteks yang panjang. *(Catatan: Gambar 3 dalam makalah asli memvisualisasikan penurunan tajam biaya operasional per token saat menggunakan DeepSeek-V3.2 di server nyata H800 dengan harga sewa \$2 per GPU jam. DSA membuktikan diri sebagai algoritma yang secara ekonomis lebih menguntungkan).* 
Sebagai catatan rekayasa: untuk tahap pembacaan teks awal yang pendek (*short-sequence prefilling*), kami mengimplementasikan mode khusus (MHA yang disamarkan / *masked MHA mode*) untuk mensimulasikan DSA, demi memeras efisiensi maksimal pada konteks-konteks pendek.

---

## 3. Pelatihan Pasca (*Post-Training*)

Setelah prapelatihan usai, kami menjalankan protokol pelatihan pasca (*post-training*) untuk menetaskan rupa final dari **DeepSeek-V3.2**. Fase ini juga mempertahankan penggunaan atensi renggang. Secara garis besar, kami mengeksekusi pipa yang sama seperti pada versi Eksperimental (V3.2-Exp), yang mencakup Distilasi Spesialis (*specialist distillation*) dan Pelatihan RL Campuran (*mixed RL training*).

### Distilasi Spesialis (*Specialist Distillation*)

Untuk setiap ranah tugas, kami awalnya menciptakan sebuah "model spesialis". Setiap spesialis ini difokuskan dan di-*fine-tune* murni berangkat dari *base checkpoint* DeepSeek-V3.2. Kerangka kerja kami mencakup enam domain spesialis secara eksklusif:
1.  Matematika (*mathematics*)
2.  Pemrograman (*programming*)
3.  Penalaran logika umum (*general logical reasoning*)
4.  Tugas agenik umum (*general agentic tasks*)
5.  Koding agenik (*agentic coding*)
6.  Pencarian agenik (*agentic search*)

Seluruh spesialis ini mendukung baik mode "berpikir" (*thinking*) maupun "langsung menjawab" (*non-thinking*). Setiap spesialis digembleng dengan siklus komputasi *Reinforcement Learning* (RL) skala besar. 

Langkah selanjutnya: Kami menyedot data jawaban berkualitas dari para spesialis ini (baik data rantai pemikiran panjang maupun respons instan). "Data spesialis" yang telah didistilasi inilah yang kemudian digunakan untuk menyuapi dan melatih *Final Checkpoint* (Model Final). Eksperimen mendemonstrasikan bahwa: model "Tugas Umum" yang dilatih menggunakan data distilasi tersebut sukses mencapai performa yang nyaris identik dengan sang model Spesialis. Sisa selisih (*gap*) performa yang kecil akhirnya lenyap tak berbekas saat model ini diikutkan pada pelatihan RL berikutnya.

### Pelatihan RL Campuran (*Mixed RL Training*)

Pada DeepSeek-V3.2, algoritma RL yang diandalkan tetaplah **Group Relative Policy Optimization (GRPO)** (DeepSeek-AI, 2025; Shao et al., 2024). 
Sesuai rancangan V3.2-Exp, kami "mencampurkan" secara ekstrem seluruh elemen—pelatihan penalaran, agenik, dan penyelarasan humanis (*human alignment*)—ke dalam SATU TAHAP RL TUNGGAL. 

Keuntungan metodologis ini: 
Ia mengelola keseimbangan performa model secara serempak di berbagai domain, dan yang terpenting: menghindari masalah "lupa ingatan yang mematikan" (*catastrophic forgetting*) yang seringkali menjadi momok kutukan pada arsitektur pelatihan multi-tahap (di mana jika AI diajari biologi hari ini, ia akan lupa matematika yang dipelajari kemarin).

*   Untuk tugas Penalaran dan Agen, kami menerapkan ganjaran berbasis hasil/aturan yang kaku (*rule-based outcome reward*), pinalti kepanjangan karakter (*length penalty*), dan pinalti konsistensi bahasa.
*   Untuk tugas Umum (sastra, ngobrol bebas), kami menggunakan *Generative Reward Model* yang dilengkapi rubrik penilaian evaluasi khusus untuk setiap *prompt*.

### DeepSeek-V3.2 dan Varian DeepSeek-V3.2-Speciale

**DeepSeek-V3.2** menyatukan seluruh kecerdasan distilasi (penalaran, agen, human-alignment). Ia ditempa dengan ribuan langkah RL untuk menggapai wujud akhirnya.

Namun, untuk mengukur seberapa jauh AI ini sanggup memikirkan sesuatu (jika tidak dibatasi panjang karakter), kami menelurkan satu mutan eksperimental yang kami beri nama **DeepSeek-V3.2-Speciale**. 
Varian *Speciale* ini dilatih secara membabi buta MURNI hanya pada data penalaran matematis dan pemrograman. Pada fase RL-nya, kami memotong drastis "pinalti kepanjangan karakter" (*reduced length penalty*), memberinya izin untuk "bergumam dan berpikir sejauh mungkin". Di atas itu, kami juga meminjam metode dataset dan *reward* dari proyek DeepSeekMath-V2 (Shao et al., 2025) guna memompa adrenalin kapabilitas pembuktian matematis formalnya (*mathematical proofs*).

*(Kita akan melihat sepak terjang monster varian Speciale ini pada Bab Evaluasi Spesifik Agenik dan Matematika nanti).*
