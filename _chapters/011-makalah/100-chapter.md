---
slug: ch-11-10
title: "DeepSeek-Prover-V1.5"
layout: chapter
---

## DeepSeek-Prover-V1.5 - Memanfaatkan Umpan Balik Asisten Pembuktian untuk Pembelajaran Penguatan dan Pencarian Pohon Monte-Carlo

Dalam bab ini, kita akan membahas **DeepSeek-Prover-V1.5**, sebuah model bahasa sumber terbuka (*open-source language model*) yang dirancang khusus untuk pembuktian teorema di dalam bahasa **Lean 4**. Versi V1.5 ini meningkatkan model pendahulunya (DeepSeek-Prover-V1) dengan mengoptimalkan dua aspek krusial: proses pelatihan (*training*) dan proses inferensi (*inference*).

Secara arsitektural, model ini diprapelatihan (*pre-trained*) berdasarkan `DeepSeekMath-Base` yang memiliki spesialisasi dalam bahasa matematika formal. Model tersebut kemudian menjalani Pelatihan Halus Terselia (*Supervised Fine-Tuning* / **SFT**) menggunakan dataset pembuktian teorema formal yang telah disempurnakan (diturunkan dari DeepSeek-Prover-V1). 

Penyempurnaan lebih lanjut (*further refinement*) dicapai melalui **Pembelajaran Penguatan dari Umpan Balik Asisten Pembuktian** (*Reinforcement Learning from Proof Assistant Feedback* / **RLPAF**). Melampaui pendekatan generasi pembuktian utuh dalam satu lintasan (*single-pass whole-proof generation*) yang digunakan pada versi V1, makalah ini mengusulkan **RMaxTS**, sebuah varian dari Pencarian Pohon Monte-Carlo (*Monte-Carlo Tree Search* / **MCTS**). RMaxTS menggunakan strategi eksplorasi yang didorong oleh penghargaan intrinsik (*intrinsic-reward-driven exploration*) untuk menghasilkan jalur-jalur pembuktian yang beragam. 

DeepSeek-Prover-V1.5 mendemonstrasikan peningkatan yang sangat signifikan dibandingkan versi V1, mencapai hasil mutakhir (*state-of-the-art*) yang baru pada himpunan pengujian (*test set*) tingkat sekolah menengah atas yaitu tolok ukur **miniF2F** (sebesar **63,5%**) dan pada tingkat sarjana yaitu tolok ukur **ProofNet** (sebesar **25,3%**).

---

### Analisis Akademik Gambar 1: Perbandingan Tingkat Kelulusan Model Pembuktian

Sebagai landasan visual sebelum kita masuk ke teori mendalam, mari kita analisis **Gambar 1** yang merangkum pencapaian model ini.

*   **Deskripsi Visual:** Gambar 1 adalah diagram batang (*bar chart*) yang terbagi menjadi dua panel komparatif. Panel sebelah kiri mengevaluasi model pada tolok ukur `miniF2F-test` (tingkat SMA), sedangkan panel sebelah kanan mengevaluasi pada tolok ukur `ProofNet` (tingkat sarjana).
*   **Sumbu X (Horizontal):** Menampilkan nama tolok ukur pengujian. Panel kiri dibagi lagi menjadi `miniF2F-test (fine-tuned)` untuk model yang telah melalui pelatihan halus, dan `miniF2F-test (pre-trained)` untuk model dasar. Panel kanan adalah untuk `ProofNet`.
*   **Sumbu Y (Vertikal):** Merepresentasikan **Tingkat Kelulusan / Akurasi (*Pass Rate %*)**, yaitu persentase teorema yang berhasil dibuktikan secara formal oleh AI. Skala untuk miniF2F berkisar antara 20% hingga 65%, sementara skala untuk ProofNet jauh lebih rendah, berkisar antara 10% hingga 26% (menunjukkan tingkat kesulitan yang jauh lebih tinggi).
*   **Legenda Warna Batang:**
    *   **Merah Bata:** `DeepSeek-Prover-V1.5` (Fokus utama makalah ini).
    *   **Kuning Tua:** `InternLM2-StepProver` (Pesaing kuat).
    *   **Cokelat / Merah Pudar:** `DeepSeek-Prover-V1` (Versi pendahulu).
    *   **Hijau Gelap:** `Hypertree Proof Search`.
    *   **Hijau Zaitun:** `GPT-f`.
    *   **Ungu Muda:** `ReProver`.
    *   **Biru / Abu-abu:** Seri `Llemma` (7B dan 34B).
*   **Kesimpulan Akademis:** Diagram batang ini dengan sangat jelas memvisualisasikan supremasi **DeepSeek-Prover-V1.5** (batang merah). Pada panel paling kiri (miniF2F-test), model V1.5 mencapai angka hampir 65% (tepatnya 63,5%), secara telak mengalahkan versi V1-nya yang berada di kisaran 50%, dan menumbangkan model-model saingan tangguh lainnya seperti InternLM2 dan metode pencarian hibrida sebelumnya. Pola dominasi ini secara identik tervalidasi di panel paling kanan untuk soal-soal tingkat sarjana (ProofNet), di mana batang merah kembali berdiri paling tinggi melampaui 25%. Ini memberikan bukti empiris bahwa arsitektur pencarian pohon yang diusulkan sangatlah sukses.

---

## 1. Pendahuluan

Kemajuan terkini dalam model bahasa besar telah secara signifikan memengaruhi penalaran matematis dan pembuktian teorema dalam ranah kecerdasan buatan. Meskipun kemajuan luar biasa terjadi dalam domain bahasa alami (*natural language*), model bahasa masih menghadapi tantangan substansial dalam pembuktian teorema formal (*formal theorem proving*). Contohnya, saat menggunakan perangkat bantu seperti **Lean** (Moura and Ullrich, 2021) dan **Isabelle** (Paulson, 1994), sistem menuntut penurunan matematis yang sangat ketat yang harus memenuhi spesifikasi formal dari sistem verifikasi tersebut. 

Bahkan model canggih layaknya GPT-4 (OpenAI, 2023) masih terseok-seok menghadapi bukti formal yang kompleks. Hal ini menggarisbawahi sifat rumit dari memadukan keahlian "koding" (*coding*) dan "matematika" secara bersamaan. Sebuah model pembuktian teorema formal tidak hanya harus memahami sintaksis dan semantik sistem formal seperti *Lean theorem prover*, tetapi juga harus menyelaraskan penalaran matematis yang abstrak dengan representasi formal yang sangat presisi.

Model bahasa dalam pembuktian teorema formal umumnya menggunakan salah satu dari dua strategi berikut:
1.  **Generasi langkah-pembuktian (*Proof-step generation*):** Metode ini memprediksi taktik (*tactic*) demi taktik berikutnya dan memverifikasinya menggunakan *formal verifier* (pemverifikasi formal) untuk mendapatkan informasi terbaru tentang "status taktik" (*tactic state*) saat ini. Ia sering kali memanfaatkan teknik pencarian pohon (*tree search*) untuk mengonstruksi bukti yang valid.
2.  **Generasi pembuktian utuh (*Whole-proof generation*):** Sebaliknya, metode ini jauh lebih efisien secara komputasi. Ia menghasilkan *seluruh* kode pembuktian sekaligus berdasarkan pernyataan teorema di awal, sehingga membutuhkan anggaran komunikasi (*communication budget*) yang lebih sedikit untuk berkoordinasi antara model pembukti dan pemverifikasi teorema formal. 

Meskipun **DeepSeek-Prover-V1** (Xin et al., 2024) telah mencapai hasil *state-of-the-art* di Lean 4 dengan menggunakan metode generasi pembuktian utuh, paradigma ini menghadirkan tantangan tersendiri. Ia membutuhkan prediksi urutan berhorizon panjang (*long-horizon sequence prediction*) tanpa akses ke "status taktik menengah" (*intermediate tactic states*), padahal taktik-taktik di masa depan sangat bergantung pada hasil yang tersembunyi tersebut. 

Dalam mode taktik Lean, bukti dikonstruksi melalui serangkaian instruksi yang mentransformasikan status bukti (*proof state*). Sifat sekuensial (berurutan) ini memunculkan risiko *compounding errors* (kesalahan yang menumpuk) (Ross et al., 2011), di mana satu kesalahan interpretasi di awal dapat berujung pada deviasi (penyimpangan) yang fatal dari jalur pembuktian yang valid. Secara lebih spesifik, model auto-regresif (*auto-regressive model*) mungkin memiliki keyakinan yang salah (*incorrect believes*) terhadap status taktik perantara saat menghasilkan kode pembuktian yang panjang.

Untuk mengintegrasikan status taktik perantara ke dalam generasi langkah-pembuktian secara mulus—sembari tetap mempertahankan kesederhanaan dan efisiensi komputasi dari generasi pembuktian utuh—kami telah mengembangkan pendekatan terpadu dalam **DeepSeek-Prover-V1.5**. 

Metode ini menggabungkan kekuatan dari teknik generasi langkah-pembuktian maupun pembuktian utuh melalui mekanisme **potong-dan-lanjutkan (*truncate-and-resume*)**. 
*   Proses ini dimulai dengan generasi pembuktian utuh standar, di mana model bahasa menyelesaikan kode pembuktian mengikuti awalan pernyataan teorema. 
*   *Lean prover* (Pemverifikasi Lean) kemudian memverifikasi kode tersebut. Jika buktinya benar dan lengkap, prosedur dihentikan. 
*   Jika kesalahan terdeteksi, kode tersebut akan **dipotong (*truncated*)** tepat pada pesan kesalahan pertama, dan kode apa pun setelahnya akan dibuang. 
*   Kode pembuktian yang berhasil dihasilkan (sebelum batas potongan tersebut) kemudian digunakan sebagai *prompt* masukan untuk menghasilkan segmen pembuktian berikutnya. 

Untuk meningkatkan keakuratan penyelesaian model yang baru, kami **menambahkan status terbaru (*latest state*) dari *Lean 4 prover*** sebagai komentar di akhir *prompt* tersebut. Yang terpenting, metode kami tidak terbatas pada sekadar "melanjutkan dari taktik terakhir yang berhasil diterapkan". Kami mengintegrasikan mekanisme potong-dan-lanjutkan ini ke dalam **Pencarian Pohon Monte-Carlo (*Monte-Carlo tree search* / MCTS)** (Coulom, 2006) di mana titik-titik pemotongan dijadwalkan oleh kebijakan pencarian pohon. 

Sebagai tambahan, kami mengusulkan algoritma eksplorasi bebas-penghargaan (*reward-free exploration algorithm*) yang inovatif untuk MCTS guna mengatasi masalah **penghargaan yang renggang (*reward sparsity*)** dalam pencarian bukti. Kami membekali agen pencarian pohon dengan motivasi intrinsik (*intrinsic motivation*), yang juga dikenal sebagai rasa ingin tahu atau *curiosity* (Schmidhuber, 2010), untuk mengeksplorasi ruang status taktik secara ekstensif. Modul-modul algoritmik ini memperluas fungsionalitas model generasi pembuktian utuh kami menjadi alat yang sangat fleksibel untuk pembuktian teorema interaktif. Alat ini dapat secara efektif memanfaatkan umpan balik dari asisten pembuktian untuk menghasilkan kandidat-kandidat solusi yang beragam.

---

### Analisis Akademik Gambar 2: Kerangka Kerja Keseluruhan (Overall Framework)

Untuk memahami secara holistik bagaimana sistem bekerja, mari kita bedah **Gambar 2**.

*   **Deskripsi Visual:** Gambar 2 adalah bagan alir operasional (*flowchart*) yang dibagi menjadi dua ranah utama: **Model Training** (Pelatihan Model) di kotak atas, dan **Model Inference** (Inferensi Model / Penggunaan Model) di kotak bawah.
*   **Alur Model Training:**
    *   Dimulai dari **Pre-training** (Prapelatihan) $\rightarrow$ berlanjut ke **Supervised Fine-tuning** (SFT) $\rightarrow$ dan diakhiri dengan **Reinforcement Learning** (RL).
    *   *Siklus RL:* Pada tahap SFT dan RL, terdapat ilustrasi interaksi. Model diberikan `prompt` (berupa potongan kode `lean4`), lalu model memberikan `response` (sisa kodenya). Saat RL, respons ini akan melalui tahapan `rollout` dan kemudian diverifikasi oleh `Lean4 prover` (Pemverifikasi Lean4). Pemverifikasi ini akan memberikan `reward` (Penghargaan) berupa nilai biner (misalnya, *Passed* atau *failed to synthesize*). Sinyal *reward* inilah yang digunakan untuk memperbaiki parameter model.
*   **Ilustrasi *Whole-proof Completion* (Sisi Kiri Bawah):**
    *   Terdapat dua contoh kode: (a) *Prompt* awal yang berakhir dengan `/- tactic state: ... -/` yang mendeskripsikan status matematika saat ini.
    *   (b) Di bawahnya, model mengisi sisa blok kode pembuktian tersebut (dari perintah `rw [h0]` hingga `norm_num`).
*   **Alur Model Inference (Kotak Kanan Bawah):**
    *   Saat digunakan di dunia nyata, model menawarkan dua jalur: **Single-pass Sampling** (Langsung tebak dari awal sampai akhir dalam sekali jalan) ATAU **Monte-Carlo Tree Search** (MCTS).
    *   **Ilustrasi MCTS (Pohon Pencabangan):** Di sebelah kiri kotak MCTS, terdapat visualisasi pohon pencarian (berwarna biru bergradasi) yang mencabang menjadi beberapa rute (node a, b, c, d, e). Setiap node merepresentasikan potongan taktik kode. 
    *   *Node yang Gagal:* Titik dengan warna merah dan tulisan "DISCARDED" menunjukkan mekanisme *Truncate* (Potong). Saat pemverifikasi Lean 4 mendeteksi *Error*, cabang itu langsung dibuang dan algoritma akan *Resume* (melanjutkan pencarian) ke cabang probabilitas lain yang belum dieksplorasi (ditunjukkan oleh tulisan *Expansion* ke node d dan e).
*   **Kesimpulan Akademis:** Diagram ini memvisualisasikan bagaimana DeepSeek-Prover-V1.5 menginkorporasikan "Umpan Balik Lingkungan yang Absolut" (Compiler Lean 4) bukan hanya sebagai hakim di akhir, melainkan sebagai penunjuk jalan aktif selama pembuatan pohon logika (MCTS) dan saat fase pelatihan kebijakan (RL).

---

### 1.1 Kontribusi Penelitian

Kami menyajikan kerangka kerja komprehensif untuk mengembangkan pembukti matematika formal berbasis model bahasa, yang mengintegrasikan beberapa komponen kunci: prapelatihan matematis skala besar, konstruksi dan augmentasi korpus matematika formal, pembelajaran penguatan secara daring (*online RL*) dari umpan balik asisten pembuktian, dan metodologi pencarian pohon untuk perencanaan jangka panjang dalam pembuktian teorema. Model prapelatihan, model pelatihan-halus terselia (SFT), dan model pembelajaran penguatan, beserta kode untuk algoritma pencarian pohon Monte-Carlo, akan dirilis secara publik untuk kepentingan penelitian dan aplikasi lebih lanjut.

*   **Prapelatihan (*Pre-Training*):** Kami meningkatkan kemampuan model dasar (*base model*) kami dalam pembuktian teorema formal dan penalaran matematis dengan melakukan prapelatihan lanjutan pada data matematika dan kode berkualitas tinggi. Fokus utama diberikan pada bahasa formal seperti Lean, Isabelle, dan Metamath.
*   **Pelatihan Halus Terselia (*Supervised Fine-Tuning*):** Kami memperbaiki kualitas dataset penyelesaian kode Lean 4 dengan mengimplementasikan dua teknik augmentasi data. Pertama, kami menggunakan DeepSeek-Coder V2 236B (Zhu et al., 2024) untuk menganotasi komentar *chain-of-thought* (pemikiran berantai) menggunakan bahasa alami di samping kode Lean 4. Hal ini bertujuan untuk menyelaraskan pembuktian teorema formal dengan logika penalaran bahasa manusia. Kedua, kami menyisipkan informasi status taktik perantara (*intermediate tactic state*) ke dalam kode pembuktian Lean 4. Hal ini krusial untuk memampukan model kami memanfaatkan umpan balik dari kompilator (*compiler*) secara efektif. Dataset yang dihasilkan kemudian digunakan untuk melakukan *fine-tune* pada model prapelatihan.
*   **Pembelajaran Penguatan (*Reinforcement Learning*):** Kami menggunakan algoritma **GRPO** (Shao et al., 2024) untuk menjalankan *reinforcement learning from proof assistant feedback* (RLPAF) pada model yang telah di-*fine-tune*. Hasil verifikasi (benar atau salah) dari *Lean prover* (Pemverifikasi Lean) berfungsi sebagai pengawasan penghargaan (*reward supervision*). Proses ini secara drastis meningkatkan keselarasan model dengan spesifikasi formal yang kaku dari sistem verifikasi tersebut.
*   **Pencarian Pohon Monte-Carlo (*Monte-Carlo Tree Search*):** Kami memajukan metode pencarian pohon dalam pembuktian teorema formal dengan memperkenalkan sebuah abstraksi baru dan algoritma pencarian yang sepadan. Mekanisme potong-dan-lanjutkan (*truncate-and-resume*) kami bertindak sebagai abstraksi *state-action* (status-tindakan), yang secara mulus mengintegrasikan proses pencarian pohon ke dalam kerangka kerja generasi pembuktian utuh. Kami menghadirkan **RMaxTS**, sebuah algoritma pencarian pohon Monte-Carlo inovatif yang memanfaatkan strategi **RMax** (Brafman and Tennenholtz, 2002) untuk mengatasi tantangan eksplorasi pada masalah pencarian bukti yang miskin penghargaan (*sparse-reward*). Dengan memberikan "penghargaan intrinsik" (rasa ingin tahu), algoritma ini mendorong agen pembukti untuk menghasilkan jalur perencanaan yang lebih beraneka ragam (*diverse planning paths*), sehingga mendorong eksplorasi ruang pembuktian secara jauh lebih ekstensif.

### 1.2 Ringkasan Evaluasi dan Metrik

*   **miniF2F:** Dalam pengaturan generasi pembuktian utuh lintasan-tunggal (*single-pass whole-proof generation*), DeepSeek-Prover-V1.5 mencapai tingkat kelulusan sebesar **60,2%** pada himpunan uji (*test set*) miniF2F. Ini menandai perbaikan yang sangat signifikan sebesar **10,2 poin persentase absolut** di atas batas 50,0% yang dicetak oleh DeepSeek-Prover-V1. Lebih dahsyat lagi, dengan memasukkan teknik pencarian pohon (MCTS), tingkat kelulusan ini terdorong naik ke rekor *state-of-the-art* yang baru sebesar **63,5%**.
*   **ProofNet:** DeepSeek-Prover-V1.5 juga menunjukkan performa tangguh dalam generasi pembuktian utuh lintasan-tunggal untuk ProofNet (soal tingkat sarjana), dengan tingkat kelulusan sebesar **21,6%** pada himpunan validasi (*validation set*) dan **23,7%** pada himpunan uji. Integrasi teknik pencarian pohon semakin meningkatkan hasil tersebut, mencapai rekor *state-of-the-art* baru sebesar **25,4%** pada himpunan validasi dan **25,3%** pada himpunan uji.

---

## 2. Pelatihan Model (*Model Training*)

### 2.1 Prapelatihan (*Pre-training*)

Untuk meningkatkan kemahiran model bahasa kami dalam menghasilkan bukti formal dan bernalar melalui bahasa matematis, kami melakukan prapelatihan lebih lanjut pada model dasar kami (Shao et al., 2024). Penyempurnaan ini melibatkan pelatihan menggunakan himpunan data berkualitas tinggi yang mencakup kode pemrograman dan konten matematis bahasa alami. Kami secara spesifik berfokus pada bahasa-bahasa formal yang diadopsi secara luas di dalam asisten pembuktian (*proof assistants*), seperti Lean, Isabelle, dan Metamath. Kami menyebut model dasar yang telah ditingkatkan ini sebagai **DeepSeek-Prover-V1.5-Base**.

### 2.2 Pelatihan Halus Terselia (*Supervised Fine-tuning*)

Di bagian ini, kami mengeksplorasi metodologi dan proses yang terlibat dalam *supervised fine-tuning* (SFT) dari DeepSeek-Prover-V1.5. Secara spesifik, kami melakukan augmentasi (penambahan data) pada dataset pembuktian dari DeepSeek-Prover-V1 dengan menyisipkan komentar penjelasan yang mendetail. Peningkatan ini bertujuan untuk memperbaiki penyelarasan (*alignment*) antara deskripsi (pemikiran) dalam bahasa alami dengan baris kode Lean 4, sehingga memfasilitasi penalaran matematis formal yang jauh lebih baik. Sebagai tambahan, kami menyisipkan informasi status taktik perantara (*intermediate tactic state information*) sebagai sebuah tugas prediksi tambahan (*auxiliary prediction task*). Hal ini dikerjakan untuk mendukung mekanisme potong-dan-lanjutkan yang krusial untuk proses Pencarian Pohon Monte-Carlo nanti. Kami menyebut model yang dihasilkan dari proses SFT ini sebagai **DeepSeek-Prover-V1.5-SFT**.

**Kurasi Data (*Data Curation*).** 
Kami mengembangkan dataset pelengkap kode Lean 4 yang komprehensif untuk *supervised fine-tuning*. Dataset ini memuat kode bukti sintetis yang diturunkan dari beraneka ragam teorema formal. Teorema-teorema ini bersumber dari berbagai proyek, seperti pustaka matematika Lean 4 standar yaitu *Mathlib4* (Mathlib Community, 2020), teorema sintetis dari DeepSeek-Prover-V1 (Xin et al., 2024) dan Lean Workbook (Ying et al., 2024), serta himpunan validasi dari tolok ukur miniF2F (Zheng et al., 2022) dan ProofNet (Azerbayev et al., 2023). 

Untuk mengaugmentasi data bukti formal, kami menggunakan proses iterasi pakar (*expert iteration process*) (Polu and Sutskever, 2020). Proses ini mencakup siklus: 
1. menghasilkan bukti menggunakan model bahasa, 
2. memverifikasi data bukti yang dihasilkan menggunakan kompilator, 
3. melatih ulang (*retraining*) model dengan data yang telah diverifikasi tersebut, 
4. menggunakan model yang telah dioptimalkan untuk kembali menghasilkan data bukti tambahan. 

Di antara setiap iterasi, kami memanfaatkan model **DeepSeek-Coder V2 236B** (Zhu et al., 2024) yang sangat cerdas untuk **menganotasikan proses berpikir (*thought process*)** tepat sebelum kode pembuktian dituliskan, dengan cara menuliskannya ke dalam format komentar. Terakhir, kami menyesuaikan format data ini agar cocok dengan mekanisme potong-dan-lanjutkan untuk Pencarian Pohon Monte-Carlo (penjelasan detail ada di Bab 3.1). Dataset bukti yang dihasilkan pada akhirnya terdiri dari **9.645 ribu sekuens**.

**Generasi Bukti Berbasis Pemikiran (*Thought-augmented Proof Generation*).** 
Pada generasi DeepSeek-Prover-V1 sebelumnya, kami mengidentifikasi sebuah celah (*gap*) pemahaman yang sangat signifikan antara "strategi penyelesaian masalah dalam bahasa alami" versus "pembuktian teorema di dalam bahasa Lean". 

Dalam bahasa alami, AI akan menghasilkan langkah-langkah deduksi yang sangat detail dan sistematis untuk menyusun bukti. Sebaliknya, saat menggunakan bahasa pemrograman Lean, AI sering kali hanya mengandalkan urutan panggilan "taktik tingkat tinggi" (*high-level tactic calls*) untuk memecahkan masalah secara *brute-force* (memaksa mesin mencari sendiri). Taktik tingkat tinggi ini (seperti `linarith` atau `ring`), meskipun sangat efektif secara kode, justru *menyembunyikan* cara kerja internal dan hasil perhitungannya. Hal ini pada akhirnya akan menghambat kemampuan model AI untuk meresolusi tujuan pembuktian yang kompleks dengan penalaran matematis yang terstruktur.

Untuk menanggulangi masalah ini, kami mengembangkan sebuah pendekatan baru. Pendekatan ini memaksa AI untuk "menuliskan penalaran bahasa alaminya terlebih dahulu" sebelum ia menuliskan kode bukti teorema (Lean). Metode ini serupa dengan teknik **Lean-STaR** (Lin et al., 2024), yang melakukan penalaran *chain-of-thought* terisolasi sebelum memulai setiap langkah pembuktian. Bedanya, metode kami mengintegrasikan penalaran ini **secara langsung sebagai komentar di dalam kode bukti itu sendiri**. 

Kami menggunakan DeepSeek-Coder V2 236B untuk meningkatkan data pada DeepSeek-Prover-V1 melalui dua cara:
1.  Menyisipkan solusi bahasa alami yang komprehensif **di awal blok bukti** (sebelum koding dimulai).
2.  Menyisipkan langkah-langkah spesifik dalam bahasa alami secara berselang-seling, tepat di atas setiap taktik Lean yang berkorespondensi dengannya. 

Melatih model dengan format data seperti ini akan memaksanya untuk mengajukan penalaran matematis yang matang di awal, serta melakukan perencanaan langkah yang detail sebelum ia memuntahkan kode taktik. Pendekatan ini berhasil menumbuhkan kebiasaan (perilaku) baru pada LLM, di mana ia sekarang menggunakan **pemikiran matematis yang halus (*delicate mathematical thinking*)** untuk memandu pembuatan kode taktiknya.

Di dalam data pelatihan, dua buah *prompt* pemandu (*guiding prompts*) yang berbeda digunakan untuk membedakan antara mode **CoT (Chain of Thought)** dan mode **non-CoT** pada tugas penyelesaian kode bukti. 

**Augmentasi Prompt dengan Informasi Status Taktik (*Prompt Augmentation with Tactic State Information*).** 
Untuk merealisasikan mekanisme "potong-dan-lanjutkan" demi Pencarian Pohon Monte-Carlo, kami harus mengekstrak informasi *tactic* dari kode yang dimuntahkan oleh model. Kami menyempurnakan program *Lean REPL (Read-Eval-Print Loop)* menggunakan modul ekstraksi data dari proyek **LeanDojo** (Yang et al., 2023). Hal ini memungkinkan kami untuk mengekstrak informasi taktik ke dalam bentuk tripel, yang mencakup: posisi setiap taktik, status taktik *sebelum* dieksekusi, dan status taktik *setelah* dieksekusi. 

Informasi ini membantu kami melacak secara pasti kode taktik mana yang memicu eror verifikasi (ini sangat penting untuk langkah *expansion* / ekspansi pada pencarian pohon, lihat Bab 3.2). Untuk setiap taktik dalam bukti formal yang valid, kami menyisipkan "status taktik" (yang dikembalikan oleh pemverifikasi Lean) ke dalam kode sebagai komentar dengan format: `/- tactic state: ... -/`. 

Selama proses pelatihan (SFT), kami hanya menggunakan token-token yang berada *setelah* komentar `/- tactic state: ` tersebut sebagai "jawaban" (respons) untuk mengkalkulasi fungsi kerugian (*loss calculation*). Sementara itu, token-token yang berada *sebelum* komentar ini akan dianggap sebagai *prompt* masukan (pertanyaan) dan tidak akan ikut serta dalam kalkulasi *training loss*.

---

**Pengaturan Pelatihan (*Training Setting*).** 
Kami melakukan pelatihan halus terselia (SFT) dengan bertolak dari model prapelatihan. Model dilatih memakan **9 Miliar (9B) token**, menggunakan ukuran tumpak (*batch size*) raksasa sebesar 2.048, dan laju pembelajaran konstan (*constant learning rate*) di angka 1e-4. Proses pelatihan ini dikawal dengan 100 langkah pemanasan (*warm-up steps*) untuk menstabilkan dinamika pembelajaran gradien. Contoh-contoh pelatihan digabungkan secara acak (*randomly concatenated*) untuk membentuk urutan kalimat, yang dibatasi hingga panjang konteks maksimum (*maximum context length*) **4.096 token**.

### 2.3 Pembelajaran Penguatan dari Umpan Balik Asisten Pembuktian (*Reinforcement Learning from Proof Assistant Feedback*)

Teknik Pembelajaran Penguatan (*Reinforcement Learning* / RL) telah dibuktikan secara empiris sangat efektif untuk mendongkrak kapabilitas penalaran matematis pada model bahasa yang telah melalui tahap SFT (Shao et al., 2024). Untuk mendorong batas kecerdasan **DeepSeek-Prover-V1.5-SFT** lebih jauh lagi, kami memasukkan sebuah fase pembelajaran penguatan, yang melahirkan entitas model final: **DeepSeek-Prover-V1.5-RL**. Fase RL ini menyerap kekuatan dari umpan balik verifikasi absolut (*verification feedback*) yang diberikan langsung oleh pemverifikasi kompilator Lean 4. Rincian proses RL ini dijabarkan di bawah:

**Prompts (Instruksi Masukan).** 
Pada tahap pembelajaran penguatan, kami menggunakan sebuah himpunan bagian (*subset*) dari pernyataan-pernyataan teorema yang ada di dataset SFT untuk dijadikan sebagai *training prompts*. Kami secara spesifik menyeleksi teorema-teorema yang pada tahap SFT memiliki tingkat kesuksesan yang "moderat" (sedang-sedang saja) dalam hal menghasilkan bukti yang benar setelah beberapa kali percobaan. 

*Kenapa memilih yang moderat?* Strategi ini menjamin bahwa model masih memiliki "ruang untuk berkembang" (belum *overfitted*), namun pada saat yang sama, ia tidak terlalu bodoh sehingga masih tetap bisa sesekali mendapatkan umpan balik yang positif. Setelah disaring, kami mempertahankan sekitar **4,5k pernyataan teorema yang unik**. Setiap teorema akan diawali dengan *prompt* pemandu yang bercorak CoT maupun non-CoT. Ini memaksa model untuk memoles insting pembuktiannya di kedua mode operasi tersebut.

**Rewards (Penghargaan).** 
Ketika kita melatih LLM menggunakan RL pada umumnya (misalnya untuk mengajarinya agar tidak toksik/berbahaya), kita biasanya melatih sebuah *Reward Model* (Jaringan saraf Penilai) yang akan bertugas memberikan sinyal umpan balik. **NAMUN**, pada kasus pembuktian teorema formal, kita memiliki keistimewaan yang luar biasa: kita bisa menggunakan verifikasi mutlak (*rigorous verification*) dari asisten pembuktian (Compiler Lean 4) sebagai hakimnya! 

Secara eksplisit, setiap bukti kode yang dihasilkan oleh AI akan langsung diganjar dengan **nilai penghargaan 1** (Reward = 1) jika Lean memverifikasinya sebagai kode yang benar tanpa *error*, dan diganjar **nilai 0** (Reward = 0) jika ada sintaksis atau logika yang meleset. Meskipun sinyal penghargaan biner (1 atau 0) ini akurasinya bersifat mutlak, masalah utamanya adalah sifatnya yang **sangat renggang (*sparse*)**. Terutama untuk teorema yang sangat sulit bagi model SFT, AI tersebut mungkin akan mendapatkan nol terus-menerus dan kebingungan harus belajar ke arah mana. Untuk memitigasi isu kerenggangan (*sparsity*) ini, seperti yang telah dijelaskan di paragraf *Prompts* di atas, kami secara taktis menyeleksi *prompt* pelatihan yang sifatnya menantang tetapi masih dalam batas probabilitas untuk bisa diselesaikan oleh model SFT.

**Reinforcement Learning Algorithm (Algoritma RL).** 
Sebagai otak optimisasinya, kami menggunakan algoritma **Group Relative Policy Optimization (GRPO)** (Shao et al., 2024). GRPO telah terbukti memiliki tingkat efektivitas dan efisiensi yang jauh lebih unggul ketimbang PPO konvensional (*Proximal Policy Optimization*; Schulman et al., 2017). Keunggulan utama GRPO adalah karena ia **mengeliminasi kebutuhan untuk melatih model kritikus tambahan (*critic model*)** (yang biasanya menghabiskan memori GPU sangat besar). 

Secara teknis, untuk setiap *prompt* teorema, GRPO akan menyuruh AI untuk memuntahkan sekumpulan kandidat bukti (*a group of candidate proofs*). Kemudian, GRPO akan mengoptimalkan bobot model berdasarkan *nilai penghargaan relatif* (*relative rewards*) dari hasil-hasil keluaran di dalam kelompok tersebut (Misal: mana jawaban yang terbaik di antara sekumpulan tebakan tersebut). Strategi pemilihan *prompt* moderat kita di awal terbukti sangat selaras dengan natur "relatif kelompok" dari GRPO, karena model berkemungkinan besar akan memuntahkan campuran antara bukti yang benar dan bukti yang cacat di antara kandidat-kandidat tebakannya. Dinamika persaingan internal ini sangat mendongkrak kualitas proses *training*.

**Pengaturan Pelatihan (*Training Setting*).** 
Pelatihan RL dilakukan dengan bertolak dari beban parameter model SFT. Model SFT ini akan berfungsi ganda sebagai (1) *Initial Model* (model yang akan dimutakhirkan) dan (2) *Reference Model* (model acuan statis untuk memberikan penalti divergensi Kullback-Leibler (KL) agar model yang sedang dilatih tidak berhalusinasi terlalu jauh dari pakem bahasa aslinya). 

Kami menggunakan laju pembelajaran konstan (*constant learning rate*) sebesar 5e-6, dan koefisien penalti KL dipatok pada angka 0,02. Untuk setiap tebakan teorema, kami memerintahkan model untuk men- *sample* (memuntahkan) **satu grup yang berisi 32 kandidat bukti**, dengan batas panjang maksimum teks dibatasi pada 2.048 token. Ukuran tumpak pelatihan (*training batch size*) secara masif dikonfigurasi pada angka 512.

---

### Analisis Akademik Gambar 3: Evolusi Kapabilitas Model Antar Fase Pelatihan

Penting bagi kita untuk memahami bagaimana setiap fase injeksi keahlian memengaruhi ketajaman model. Mari kita bedah **Gambar 3**.

*   **Deskripsi Visual:** Gambar ini menyajikan sebuah tabel perbandingan angka, dan di bawahnya terdapat dua grafik kurva area yang sangat komprehensif. Sisi kiri adalah `miniF2F-test` dan sisi kanan adalah `ProofNet-test`.
*   **Grafik Kurva Area:**
    *   **Sumbu X (Horizontal):** *Sample Budget* (Anggaran Sampel/Tebakan), bergerak dari 0 hingga 128 (Ini merepresentasikan jumlah "k" dalam metrik `Pass@k`). Semakin ke kanan, model diberi lebih banyak kesempatan menebak.
    *   **Sumbu Y (Vertikal):** `Pass@K` (Tingkat kelulusan).
    *   **Garis Kurva:** Ada empat garis yang mewakili berbagai tahap pelatihan. 
        *   Biru Muda: `SFT (non-CoT)`
        *   Biru Tua: `SFT (CoT)`
        *   Merah Muda / Oranye Pudar: `RL (non-CoT)`
        *   Merah Tua: `RL (CoT)`
    *   *Area yang Diarsir (Shaded region):* Menggambarkan deviasi standar (rentang fluktuasi variansi).
*   **Tabel "Pass@128":** Menampilkan rekapitulasi data jika model diberi 128 kali kesempatan menebak. 
*   **Kesimpulan Akademis:**
    1.  **Lompatan Kuantum SFT vs Base:** Model Dasar (*Base*) hanya menyentuh akurasi $\sim$29.7% di miniF2F. Setelah disuntik teknik SFT, akurasinya langsung melonjak jauh menembus batas $\sim$49.8% hingga 50.4%.
    2.  **Kemenangan Mode CoT:** Perhatikan bahwa garis yang berlabel (CoT / *Chain of Thought*) — yaitu Biru Tua dan Merah Tua — secara konsisten **selalu berada di atas** garis non-CoT pasangannya. Menginstruksikan model untuk "berpikir dan menalar secara tekstual" (CoT) sebelum menulis kode kompilasi *Lean 4* terbukti mutlak menaikkan tingkat keberhasilan memecahkan teorema. 
    3.  **Batas Atas RL:** Garis Merah Tua (`RL CoT`) merajai puncak kurva grafik. Pembelajaran Penguatan menggunakan kompilator Lean terbukti memoles logika pemikiran AI hingga menyentuh puncaknya di 51.6% pada miniF2F dan 18.2% pada ProofNet.

---

## 2.4 Evaluasi (*Evaluation*)

### Tolok Ukur (*Benchmarks*)

Kami mengevaluasi kinerja pembuktian teorema pada tolok ukur berikut untuk membandingkan kapasitas intelektual model secara adil setelah melalui setiap tahapan pelatihan:

*   **MiniF2F** (Zheng et al., 2022): Memfokuskan diri pada keterampilan pemecahan masalah formal untuk kompetisi dan latihan matematika di tingkat sekolah menengah atas, seperti *American Mathematics Competitions* (AMC), *American Invitational Mathematics Examination* (AIME), dan *International Mathematical Olympiad* (IMO). Penekanan terberat berada pada cabang aljabar dan teori bilangan. Tolok ukur ini mencakup 244 masalah validasi dan 244 masalah pengujian. Soal-soalnya yang awalnya dalam bahasa Lean 3, telah kami konversi secara manual ke bahasa Lean 4.9.0, merujuk pada versi repositori dari Yang (2023).
*   **ProofNet** (Azerbayev et al., 2023): Dirancang untuk mengevaluasi kapabilitas pembuktian teorema formal pada tingkat matematika program Strata-1 (Sarjana). Ia terdiri atas 185 soal validasi dan 186 soal pengujian yang diekstraksi dari buku teks perkuliahan sarjana terkemuka. Topiknya meliputi analisis riil dan kompleks, aljabar linear, aljabar abstrak, dan topologi. Mirip dengan miniF2F, kami juga telah mengkonversi manual masalah ini dari Lean 3 ke Lean 4.9.0.

### Konfigurasi Prom (*Prompting Configurations*)

Untuk setiap usaha pembuktian yang dilakukan oleh model mentah (DeepSeek-Prover-V1.5-Base), kami secara acak memungut tiga demonstrasi bukti dari himpunan validasi untuk digunakan sebagai teknik pemicu *few-shot prompting* (memberikan 3 contoh ke model). 

*   Pada miniF2F, kami menyuapkan bukti hasil tulisan tangan manusia dari Yang (2023). 
*   Pada ProofNet, kami menyuapkan contoh bukti-bukti jitu yang diproduksi oleh AI kita sendiri (DeepSeek-Prover-V1.5-RL) sebagai demonstrasi *few-shot*-nya. 

Untuk evaluasi tahap lanjut pada model V1.5-SFT dan V1.5-RL, kami menerapkan **dua jenis *prompt* pemandu yang berbeda**: 
1.  *Prompt* yang memaksa model untuk memaparkan penalaran *chain-of-thought* (CoT) sebelum mengeksekusi setiap langkah bukti.
2.  *Prompt* yang melarang CoT (non-CoT). 

*(Contoh skrip prompt ini dapat dijumpai pada Lampiran A dalam versi asli makalah).*

### Metrik (*Metric*)

Kinerja pembuktian teorema dievaluasi secara eksklusif menggunakan metrik akurasi **pass@K**. Metrik ini mengukur tingkat keberhasilan model dalam menghasilkan *setidaknya satu bukti yang benar secara mutlak* dari $K$ kali percobaan generasi (*attempts*). 

Setiap model dieksekusi di atas infrastruktur komputasi berupa sebuah GPU NVIDIA A100-40G tunggal. Kami memanfaatkan kerangka kerja vLLM (Kwon et al., 2023) untuk melayani akselerasi generasi token teks. Parameter *sampling* dikunci pada profil yang "berani mencoba": Suhu (*temperature*) = 1, nilai pemenggalan token atas (*top-p*) = 0.95, dengan limit maksimal hembusan karakter dibatasi di 2.048 token. 

Bongkahan bukti yang dimuntahkan oleh AI selanjutnya dikirim ke hakim absolut: kompilator *Lean 4 theorem prover*. Proses kompilasi dan verifikasi ini mengimpor perpustakaan raksasa *Mathlib4* (Mathlib Community, 2020) serta otomasi pencarian taktik *Aesop* (Limperg and From, 2023) untuk menyediakan postulat dan taktik pendukung bawaan. Demi mencegah *infinite loop* (kode berputar tanpa henti) dari AI, hakim *compiler* ini dibatasi waktu (*time limit/timeout*) maksimal **300 detik** (5 menit) untuk memeriksa satu soal.

### Perbandingan Lintas Tahapan Pelatihan (*Comparison across Training Stages*)

Jika Anda merujuk kembali pada analisis Gambar 3 sebelumnya (yang telah kita bedah panjang lebar), kita menyaksikan analisis komparatif dari setiap stase pelatihan. 
1.  Model dasar mentah (Base) kita sudah mencapai tingkat kelulusan yang luar biasa tinggi (menyelesaikan hampir sepertiga soal miniF2F *test* menggunakan injeksi *3-shot prompting*). 
2.  Tahap *fine-tuning* (SFT) membantai metrik dasar tersebut, memompa akurasi Pass@128 naik sebesar dua pertiga pada miniF2F dan melonjak hingga dua kali lipat pada ProofNet (soal anak kuliah). 
3.  Tahap mutakhir RL lalu melabrak batas langit dari SFT, menyempurnakan angka Pass@K di *seluruh nilai* $K$. 

Yang sangat menarik dari sudut pandang riset AI matematika: Berbeda dengan temuan di domain bahasa alami biasa, di mana pengaplikasian RL sering kali hanya menaikkan metrik di kelompok Top-K tebakan atas saja, dalam *formal theorem proving* ini, RL benar-benar **meningkatkan kapabilitas logis fundamental dari AI secara hakiki (*genuine enhancement*)**. Hal ini dibuktikan karena performa model tidak hanya naik tajam saat $K$ membesar (saat diberi anggaran sampel besar/banyak kali tebak), tetapi performanya juga sangat stabil dan solid bahkan pada anggaran sampel yang amat sempit (ketika AI hanya diberi sedikit kesempatan menebak). Asumsi ilmiah ini tervalidasi secara konklusif pada eksperimen Pencarian Pohon Monte-Carlo (MCTS) dengan limit sampel raksasa, yang akan kita kaji di Seksi 4.2.

### Perbandingan antara CoT dan non-CoT (*Comparison between CoT and non-CoT*)

Masih merujuk pada Gambar 3, kita secara empiris membuktikan bahwa mode "berpikir dan bergumam" (CoT) secara universal dan konsisten membantai performa mode "langsung ketik kode" (non-CoT) di hampir seluruh pengaturan komputasi. 

Dengan mengeksploitasi pola-pola penalaran mendalam (*enhanced theorem-proving patterns*) ini, **DeepSeek-Prover-V1.5-RL** merengkuh supremasi performa pada kedua arena gladiator, dengan akurasi rata-rata **51,6%** di miniF2F dan **18,2%** di ProofNet (skor sarjana yang belum pernah dicapai model terbuka sebelumnya). Penyatuan antara "penalaran bahasa alami logis" dan "penulisan kode program Lean 4" yang berjalan seiring (*interleaved*) dalam mode CoT ini memfasilitasi orkestrasi perancangan dan eksekusi bukti matematika formal yang tingkat akurasinya tak tertandingi.

---

## 3. Pencarian Pohon Monte-Carlo yang Berorientasi pada Eksplorasi (*Exploration-oriented Monte-Carlo Tree Search*)

Bagian ini membahas pilar arsitektur logika terakhir. Alih-alih menyuruh AI menembak jawaban sekali tembus layaknya senapan mesin (*single-pass*), kita akan mengubah AI menjadi pecatur ulung yang sanggup menerawang skenario percabangan langkah masa depan (*Tree Search*).

### 3.1 Abstraksi Pohon Tingkat-Taktik (*Tactic-level Tree Abstraction*)

Untuk mengimplementasikan metode pencarian pohon dalam kerangka kerja generasi pembuktian utuh (*whole-proof*), kami memperkenalkan "abstraksi pohon pembuktian". Abstraksi ini mendefinisikan batas-batas spesifik bagi representasi "status" (*state space*) dan "tindakan" (*action space*) dari mesin AI, menggunakan mekanisme unggulan kami: **potong-dan-lanjutkan (*truncate-and-resume*)**. 

Mengikuti secara kasar paradigma revolusioner dari Yao et al. (2023) / *Tree of Thoughts*, langkah pertama kami adalah **mendekomposisi (memecah belah) sebuah teks bukti kode yang belum selesai menjadi sebuah barisan simpul (node) pada pohon**, di mana setiap node secara eksklusif merepresentasikan satu langkah deduksi taktik (*individual proof step*). Selanjutnya, algoritma akan memanfaatkan bagian kode yang belum utuh (yang tersimpan di setiap *node*) sebagai pijakan/batu loncatan (*resume*) untuk melanjutkan sisa generasi kodenya. 

### Analisis Akademik Gambar 4: Mekanisme Potong-dan-Lanjutkan (Truncate-and-Resume) pada MCTS

Perhatikan seksama **Gambar 4** yang menampilkan interaksi elegan antara model bahasa LLM dan kompilator pencari kesalahan (*Error Compiler*).

*   **Deskripsi Operasional Visual (Kotak dan Siklus):**
    *   **(a) Tactic State Extraction:** Algoritma menyeleksi sebuah node (node bulat merah). Dari node ini, algoritma menarik ke atas untuk melihat "awalan kode pembuktian" (*proof code prefix*) yang tidak utuh. Awalan ini memuat deklarasi soal (misalnya, *Compute the sum...*) dan taktik yang sudah sukses dieksekusi di node-node leluhurnya (*ancestor nodes*). 
    *   **(b) Whole-proof Completion:** Mesin bahasa LLM kemudian membaca awalan (a) ditambah sebuah komentar ajaib `/- tactic state: ... -/`. Berbekal amunisi informasi status taktik (variabel apa saja yang ia miliki di memori saat ini), LLM akan **memuntahkan sisa blok kode taktik sampai selesai (menulis utuh)** untuk menyelesaikan bukti.
    *   **(c) Error / Discarded:** Kumpulan teks utuh (b) yang dimuntahkan LLM dilempar ke kompilator Lean 4. JIKA kompilator menemukan "ERROR" (kotak merah `Error from Lean 4: failed to synthesize...`) di baris tertentu (misalnya pada baris `linarith`), maka **pisau algoritma akan memenggal (Truncate) sisa kode apa pun di bawah garis error tersebut dan membuangnya ke tong sampah (DISCARDED)**.
    *   **(d) Expansion:** Bagian kode taktik yang lulus kompilasi (misalnya `rw [h0]`, `constructor`, `intro h`, dan `have : ...`) diselamatkan. Setiap baris taktik yang sukses tersebut diubah menjadi sebuah *node baru* di dalam pohon pencarian (direpresentasikan oleh bola-bola merah muda kecil yang membentuk lintasan cabang ke bawah).
    *   **(e) Iterasi Lanjutan:** Setelah struktur dahan pohon diperbarui, putaran MCTS berikutnya akan memilih node calon lain di titik berbeda untuk dipanjangkan (*expanded*) kembali, mengeksplorasi cabang kemungkinan lain layaknya sulur tanaman yang mencari rambatan cahaya. Proses ini terus berulang tanpa ampun hingga salah satu dahan sulur tersebut menyentuh garis akhir (solusi benar) atau peluru tebakan (anggaran sampel) AI ini ludes.

**Secara Kritis Mengapa Hal Ini Menakjubkan?**
Berbeda dengan MCTS kuno (seperti AlphaGo) yang biasanya hanya berani maju memperpanjang *satu* node per langkah (*single tactical step*), metode "Generasi Pembuktian Utuh" kita memampukan agen AI ini memuntahkan berlusin-lusin baris taktik sekaligus. Jika beruntung dan 5 baris pertama sukses divalidasi oleh kompilator sebelum akhirnya memicu *Error*, algoritma kita akan memanen kelima taktik sukses tersebut menjadi 5 rantai node turunan baru sekaligus secara instan! Ini mendongkrak laju kedalaman ekspansi pohon (*tree depth expansion*) secara eksponensial dalam sekali lintasan siklus komputasi. 

**Melanjutkan Pembangunan Bukti dari Sebuah Titik Pohon (*Resume: Proof Generation from a Tree Node*).** 
Dalam Lean 4, mengetikkan beberapa taktik *syntax* kode yang berbeda bisa membawa kita pada "titik status taktik" (kondisi variabel memori) yang sama persis (ibarat "banyak jalan menuju Roma"). Menghadapi ini, sistem kami mengelompokkan taktik-taktik kembar tersebut di dalam *node* yang sama. Ketika agen pencarian MCTS ingin membongkar (*expand*) *node* tersebut, ia cukup memilih salah satu ragam taktik secara acak untuk disematkan sebagai umpan ke dalam *prompt* model LLM. Yang terpenting, *prompt* ini selalu dipungkasi (*suffixed*) dengan informasi status taktik aktual yang dipasok oleh kompilator Lean (dalam bentuk komentar `/- tactic state...-/`). Berkat *fine-tuning* ketat yang kita bahas di Bab 2.2, model bahasa ini telah bertransformasi menjadi arsitek ulung yang tahu betul bagaimana menavigasikan langkahnya dengan dipandu oleh papan petunjuk status kompilator tersebut.

---

### 3.2 Pembuktian Teorema Interaktif via Pencarian Pohon Monte-Carlo (MCTS)

Bentuk pohon pencarian (*proof search tree*) AI kita diorkestrasikan dengan mematuhi ritme klasik standar paradigma MCTS (Coulom, 2006; Browne et al., 2012). Ritme ini menari dalam siklus empat ketukan algoritma yang tak berkesudahan: **Seleksi** (*Selection*), **Ekspansi** (*Expansion*), **Simulasi** (*Simulation*), dan **Propagasi Balik** (*Backpropagation*). Namun, dengan manuver arsitektural kita yang brilian, tahapan "Simulasi" secara praktis telah *dileburkan dan dirangkul* menyatu ke dalam tahapan "Ekspansi", karena model generasi utuh kita secara genetik sudah melakukan pelintasan laju dari awal hingga ujung akhir dokumen (rollout) secara natural. Detail anatomi algoritma ini dikuliti di bawah ini.

**Seleksi (*Selection*).** 
Fase seleksi (sering diistilahkan sebagai 'Kebijakan Pohon' / *Tree Policy*) bergerak dari akar (*root node*) dan mendaki rute ranting-ranting ke bawah guna mengendus *node* mana yang paling potensial untuk dibuka/diekspansi. Denyut jantung dari tarian algoritmik ini adalah menjaga tegangan konstan antara hasrat mengeksplorasi jalan baru (*Exploration*) dan godaan untuk memperdalam jalan lama yang tampaknya menjanjikan (*Exploitation*) (Kocsis and Szepesvári, 2006). Kebijakan pohon pada node $s$ diperintahkan untuk mengarahkan bidaknya ke "tindakan" yang memaksimalkan perhitungan kalkulus berikut dari himpunan operasi yang legal:

$$ TreePolicy(s) = \arg\max_{a \in Children(s) \cup \{\oslash\}} Q_{UCB}(s, a), \tag{1} $$

**Penjelasan Variabel Matematis (1):**
*   $a$: Merupakan pilihan aksi. Ia dapat berarti menelusuri ke bawah menuju *node* anak (*child node*) yang sudah ada (disimbolkan dengan $a \in Children(s)$), ATAU ia dapat berarti 'mengeluarkan sayap' mengekspansi *node* mandek saat ini untuk menerka langkah masa depan baru (disimbolkan dengan token unik $a = \oslash$).
*   Trik algoritma menggunakan *virtual node* (Wang et al., 2023) diterapkan di sini, menautkan sebuah node anak "imajiner" yang merepresentasikan perintah "Tolong buka (*expand*) node $s$ ini". Ini sangat sakti, karena dalam dunia generasi *Large Language Model*, lautan probabilitas variasi kode itu tak terhingga; algoritma harus selalu punya tombol untuk mencoba menerka rute *expansion* baru meskipun batas jumlah anak sebelumnya sudah banyak.
*   $Q_{UCB}(s, a)$: Merupakan Estimasi Nilai total jika aksi $a$ diambil dari node $s$. Jantung nilainya dibelah menjadi dua kompartemen penyeimbang:

$$ \forall a \in Children(s) \cup \{\oslash\}, \quad Q_{UCB}(s, a) = \underbrace{Q(s, a)}_{\text{Exploitation}} + \underbrace{UCB(s, a)}_{\text{Exploration}}, \tag{2} $$

*   $Q(s, a)$ (Eksploitasi): Nilai rekam jejak. Ini adalah rata-rata historis (penghargaan) dari kesuksesan penelusuran aksi $a$ di masa lalu. Logikanya: "Jika jalan ini sering berhasil di masa lalu, ayo hajar lagi ke sini."
*   $UCB(s, a)$ (Eksplorasi): Ini adalah bonus penasaran (*Exploration bonus*) berbasis rentang kepercayaan (*Upper Confidence Bounds* / UCB; Auer, 2002). Nilai UCB ini secara otomatis *menyusut* dan menguap jika agen AI terlalu sering memilih pasangan rute $(s, a)$ yang sama. Logikanya: "Cukup! Kau sudah terlalu sering lewat jalan ini, mari kita bonus poin bagi rute gelap di sebelah sana yang belum pernah kita lalui."

**Ekspansi (*Expansion*).** 
Segera setelah kompas Seleksi menunjuk sebuah *node*, giliran mesin LLM (generasi pembuktian) meraung hidup. Merajut untaian potongan kode taktik warisan masa lalu yang membatu di *node* tersebut, model memuntahkan sekuel balistik pelengkap kode hingga ujung halaman. Ini ekuivalen dengan ritual "Simulasi" (*Rollout*) klasik pada MCTS.

Seluruh gulungan teks kode utuh itu dilemparkan ke mulut kompilator *Lean prover* untuk dikunyah dan dinilai validitasnya. Jika layar Lean memancarkan sinyal tervalidasi selesai (*proof is complete*), maka lonceng kemenangan berbunyi; Teorema matematika telah hancur terbukti, proses MCTS diakhiri paksa secara sukses. Sebaliknya (yang mana paling sering terjadi), Lean akan memuntahkan log Eror. 

*Pada titik inilah mukjizat sistem kita bekerja*: Sistem akan melacak baris Eror yang diteriakkan Lean, mengangkat golok komputasionalnya, dan memotong putus (*truncate*) seluruh blok kode setelah baris itu. Sisa kode taktik legal yang *lulus sensor Lean* sebelum titik potong tadi dirangkai, diurai (*parsed*), dan ditancapkan sebagai juntaian renteng ranting (*node* demi *node* bercabang) ke dalam silsilah MCTS (merujuk ke penjelasan visual Gambar 4). Berbeda dengan pakem MCTS konservatif pada *game* catur (Silver et al., 2016) yang pelit dan hanya membangun *satu* keping *node* generasi setiap siklus, mesin hibrida kita sanggup menginjeksikan serangkaian juntaian jalan setapak *node* secara borongan dalam sekejap mata.

---

**Propagasi Balik (*Backpropagation*).** 
Fase penutup dari tarian iterasi pencarian pohon ini bertugas memperbarui kitab catatan nilai (*value statistics*) sepanjang lintasan pohon yang baru saja ditempuh dari pangkal akar menuju pucuk ekspansi tadi (yaitu memperbarui parameter di Persamaan 1). Mari definisikan $\tau$ sebagai daftar memori trayektori lintasan *node* yang dipilih. Algoritma akan menenggak nilai *reward* terbaru, $R(\tau)$, dan mengkalkulasi ulang matriks statistik untuk seluruh titik persinggahan di garis $\tau$ tersebut.

Sinyal *reward* absolut (ekstrinsik) disuplai utuh oleh penghakiman final kompilator Lean: $R_{extrinsic}(\tau) = 1$ jika bukti matematika mekar sempurna, dan $R_{extrinsic}(\tau) = 0$ jika gagal/tersendat. Tentu saja, model hadiah biner kaku ini berpotensi membikin AI depresi di tengah labirin matematika. Karenanya, pada Bagian 3.3, kita meramu sistem "Penghargaan Intrinsik" untuk menggairahkan "rasa penasaran" AI untuk pantang menyerah bertualang.

### 3.3 Penghargaan Intrinsik untuk Pencarian Pohon Monte-Carlo (*Intrinsic Rewards*)

Dalam rimba rimba belantara silogisme formal, sinyal petunjuk (*extrinsic rewards*) sangatlah tandus. Sang agen AI MCTS baru akan dihadiahi angka "1" jikalau teka-teki teoremanya pecah seutuhnya. Berbeda dengan *game* skor yang terus memberi poin secara inkremental, proses pencarian rute pembuktian ini bagaikan berjalan di lorong tanpa seberkas cahaya, yang menduplikasi kasus "eksplorasi mengerikan" (*hard-exploration case*) (Krishnamurthy et al., 2016) di literatur statistik RL.

Untuk membangkitkan spirit pantang menyerah, diorkestrasikan sebuah algoritma legendaris **Penghargaan Intrinsik** (*Intrinsic Rewards*) (Schmidhuber, 2010; Pathak et al., 2017). Intinya: Agen MCTS tidak lagi sekadar menjadi pemburu harta karun yang serakah (skor 1 dari kompilator), melainkan diprogram untuk menjadi *Pengelana Sains* yang akan memberikan "tepuk tangan kebanggaan pada dirinya sendiri" (Penghargaan Intrinsik) tatkala berhasil mengeksplorasi *kepingan peta taktik Lean yang sama sekali baru* di ekosistem tersebut. Makalah ini membaptis metodologi eksplorasi tanpa-hadiah (*reward-free exploration*) ini dengan tajuk **RMaxTS** (*RMax applied to Tree Search*).

**RMax yang diinjeksi ke MCTS.**
Konsep radikal RMax (Brafman and Tennenholtz, 2002) adalah melahap luasnya batas hamparan *state space* taktik. Sang agen akan "memompa" rekening *reward* pribadinya secara maksimal ketika telapak kakinya menginjak "titik node taktik (*unseen state*)" yang perawan (belum pernah dijajaki sejarah komputasinya). Pendekatan algoritmik murni ini sangat selaras dengan **ZeroRMax** (Jin et al., 2020), yang mengeksekusi kendali kemudi setir hanya dengan mengonsumsi bensin Penghargaan Intrinsik, diformulasikan absolut secara matematis: $R(\tau) = R_{intrinsic}(\tau)$.

Dan ganjaran intrinsik spesifik ini meledak pada nominal positif apabila sebuah (atau banyak) *node baru* diukir dan ditancapkan pada dahan Pohon MCTS (*search tree*):

$$ R_{intrinsic}(\tau) = \mathbb{I} [\text{setidaknya ada SATU node baru yang ditambahkan ke pohon pencarian}], \tag{3} $$

*   *$\mathbb{I}$ adalah fungsi biner yang memuntahkan 1 jika pernyataan dalam kurung siku tersebut bernilai absolut Benar.*
*   Mantra heuristik RMaxTS ini ibarat cambuk yang terus mendesak agen LLM untuk tidak puas *stuck* di zona nyaman tebakan usangnya, melainkan memacu pedal gas untuk memuntahkan varian skrip kode Lean yang mendamparkan dirinya pada garis status taktik matematika (*tactic states*) yang ekstrem, radikal, dan eksotik. Terobosan ajaibnya? Karena ada ribuan ragam cara menulis kode Lean yang pada dasarnya berakhir pada *status taktik matematika yang identik*, radar heuristik RMaxTS ini secara ajaib akan menolak memberi *reward* pada variasi tulisan koding yang membuang waktu (*redundant generation*), menaikkan drastis efisiensi tenaga kuda (*sample efficiency*) dari prosesur AI tersebut.

**Diskon UCB untuk Memerangi Sinyal Imbalan Menipu (*Discounted UCB for Non-stationary Rewards*).** 
Secara default asali, bonus UCB kompas MCTS dijabarkan dalam formula **UCB1** (Auer et al., 2002) kuno:

$$ Q_{UCB1}(s, a) = \frac{W(s, a)}{N(s, a)} + \sqrt{\frac{2 \ln \sum_{a'} N(s, a')}{N(s, a)}}, \tag{4} $$
$$ W(s, a) = \sum_{\tau \in \Gamma(s, a)} R(\tau), \tag{5} $$
$$ N(s, a) = |\Gamma(s, a)|, \tag{6} $$

*   $W(s, a)$: Akumulasi total *reward* yang sukses disedot saat lewat rute $(s, a)$.
*   $N(s, a)$: Angka kunjungan mutlak (Berapa kali rute itu dilewati).
*   *Anomali yang Menghancurkan UCB1:* Coba renungkan ini. $R_{intrinsic}$ kita itu bagai fatamorgana (*non-stationary reward signal*). Di fase awal, AI dengan gampangnya mengeksplorasi ribuan jalan perawan (dapat poin banyak). Tapi seiring Pohon MCTS yang kian gondrong dan gemuk merambah batas memori, jumlah *node* baru yang bisa diukir (*unseen tactic states*) akan menurun ekstrem secara eksponensial menuju nol. Jika kita memakai UCB1 kuno di atas, memori kejayaan *reward* masa lalu yang bombastis itu akan terus *menipu* sistem kalkulus UCB, membuat AI tetap terobsesi pada rute tua.

*   *Anti-Biotiknya: DUCB (Discounted Upper Confidence Bounds)* (Garivier and Moulines, 2011). Kita injeksikan racun amnesia terkendali bernama *Discount Factor* $\gamma \in (0, 1)$ untuk meredam inflasi nilai *reward* zaman purba:

$$ Q_{DUCB}(s, a) = \frac{W_\gamma(s, a)}{N_\gamma(s, a)} + \sqrt{\frac{2 \ln \sum_{a'} N_\gamma(s, a')}{N_\gamma(s, a)}}, \tag{7} $$
$$ W_\gamma(s, a) = \sum_{t=1}^{N(s, a)} \gamma^{N(s, a) - t} R(\tau_t), \tag{8} $$
$$ N_\gamma(s, a) = \sum_{t=0}^{N(s, a) - 1} \gamma^t, \tag{9} $$

*   Perhatikan elemen $\gamma^{N(s, a) - t}$ di Rumus 8. Konstanta peluruhan $\gamma$ ini (disetel praktis di angka **0.99**) menjamin bahwa rekaman *reward* dari percobaan yang baru saja usai (segar/ *fresh feedback*) akan diangkat martabat matematisnya dan dibobot amat berat oleh sistem evaluasi matriks, sementara sisa puing kejayaan memori masa lalu yang lapuk akan "membusuk" perlahan ke arah angka nol, sehingga tak lagi mendistorsi kompas UCB MCTS.

### 3.4 Paralelisasi Pohon Monte-Carlo (*Parallelization of Monte-Carlo Tree Search*)

Mesin MCTS ini lapar komputasi. Demi menyelaraskan sinkronisitasnya dengan kecepatan GPU, diorkestrasikan simfoni perangkat keras skala-mega (mengacu pada Chaslot et al., 2008):
*   **Paralelisasi Akar (*Root Parallelization*):** Mengerahkan **256 pelari cepat MCTS** per unit perangkat komputasi *node*. Setiap GPU memanggang *satu* arsitektur model bahasa utuh dan menelan bongkahan tumpak (*batch size*) sebanyak 512 generasi bukti. Di saat yang sama, asisten pembuktian *Lean prover* dieksekusi di ranah CPU yang dipasang berangkai dengan "ribuan" unit *core* gergaji komputasi. Setiap instruksi verifikasi pembuktian digencat dan diterminasi secara steril di dalam tabung hampa *sandbox* independen secara *asynchronous* (tidak saling tunggu). Orkestrasi brutal ini menyuntikkan bahan bakar kecepatan akselerasi drastis bagi pohon MCTS.
*   **Paralelisasi Pohon (*Tree Parallelization*):** Setiap cabang pohon MCTS diawasi oleh 32 mandor proses beralur-banyak (*thread workers*) yang melempar (*schedule*) bongkahan kalkulasi inferensi generasi koding LLM dan kalkulasi asisten pemverifikasi Lean secara selang-seling agar tidak pernah ada perangkat silikon yang menganggur.
*   **Kerugian Virtual (*Virtual Loss*):** Terakhir dan paling jenius. 32 mandor *thread* yang mencari *node* ekspansi ini rentan bentrok karena semua mendatangi *node* probabilitas tinggi yang sama secara gerombolan (*herd behavior*). Guna memecah kerumunan mandor ini agar menyebar berhamburan mendatangi sudut-sudut peta *node* lain yang terisolasi, diinjeksikanlah "Sinyal Reward Virtual" palsu senilai $R(\tau) = 0$ secara "gegabah dan sesaat" untuk mengecoh sisa *thread* yang lain. Sinyal fatamorgana hukuman penalti 0 ini dikirimkan ke batang UCB, merobohkan nilai rute probabilitas tersebut, agar mandor kedua dan seterusnya "ogah" mampir ke situ. Begitu mandor pertama menyelesaikan perhitungannya (membawa nilai kompilator faktual absolut yang murni), angka 0 gadungan ini ditarik mundur (*rollback*) lalu diperbarui total dengan honorarium *reward* emas yang sah. Manuver psikologis matematika (*exploration promote*) yang sungguh cantik.

---

### 3.5 Perbandingan Dengan Pendekatan Terdahulu (*Comparison with Existing Methods*)

Dunia keilmuan akademis yang menjerumuskan LLM untuk menjadi penakluk silogisme formal (*formal mathematics proof search*) pada hakikatnya direduksi kepada pertarungan hegemoni atas dua mahzab akbar metodologi generasi:
1.  **Mahzab Generasi Langkah-Pembuktian Multi-Lintasan (*Multi-pass proof-step generation*):** Ini adalah kubu penganut metode "setapak demi setapak" (mengukir sebuah taktik, diumpan ke Lean, disahkan, lalu LLM merenungkan taktik kedua, diumpan lagi). Ini memakan waktu selestial yang luar biasa menguras ongkos koordinasi komunikasi (karena iteratif). Pasukan di kubu ini mencakup dewa-dewa kuno seperti GPT-f, Thor, ReProver, *Hypertree Proof Search*, hingga yang termutakhir *InternLM2-StepProver*.
2.  **Mahzab Generasi Pembuktian Utuh Sekali-Jalan (*Single-pass whole-proof generation*):** Di pojok ring lain, ini adalah penganut "Semburan Maut Sekali Napas" (seperti arsitektur LLM ChatGPT standar Anda). Model disuruh merancang seluruh manuskrip kitab koding matematika sekaligus. Jika Lean kompilator memuntahkan 1 eror kecil saja, seluruh dokumen ditarik dan LLM disuruh memuntahkan kembali seluruh revisi utuh (tebak-tebakan buta). Mahzab ini ditempati oleh metode-metode seperti DSP, Subgoal-Prover, LEGO-Prover, Lyra, dan miniCTX.

**Revolusi Hibrida DeepSeek-Prover-V1.5:**
Mesin pencari pohon buatan kami (RMaxTS) mengawinkan secara ekstrem (hibridisasi) kedua paradigma radikal kutub-utara-selatan di atas. MCTS kita diaktifkan sebagai meriam *Whole-proof generation* di tembakan pertama (layaknya mahzab 2), yang disuntik senjata presisi tajam *Truncate-and-Resume* (Potong-dan-Lanjutkan) untuk melokalisir pendarahan Eror dan mengekstrak taktik yang sudah sukses secara mikro. Taktik yang sudah selamat ini secara cerdas dibedah (di-*parsed*) sebagai serpihan individual taktik layaknya di mahzab 1, lalu dijejalkan sebagai titik pijakan perpanjangan estafet *Tree Search*. Perpaduan arsitektur simfoni ini secara ajaib mengizinkan kita merebus *Satu (1)* arsitektur LLM mutlak tunggal namun mampu bertarung gila-gilaan mendominasi arena baik di arena tanding *Single-pass* (Tabel 1 dan 2 bagian atas) MUPUN di arena tanding pohon hibrida (*Tree Search*).

---

## 4. Hasil Eksperimen (*Experimental Results*)

### 4.1 Hasil Utama (*Main Results*)

Kami menghadirkan analisa arena pertarungan untuk DeepSeek-Prover-V1.5 berhadapan dengan nama-nama paling menakutkan di industri LLM AI.
*   **Model Kognitif Umum (General-purpose Models):** Raksasa sumber tertutup (Proprietary), yakni Sang Algojo, **GPT-3.5** dan **GPT-4** (OpenAI), dikondisikan (diberi kruk *in-context learning* / prompt khusus agar mahir matematika oleh agen bantuan **COPRA**). Serta para maestro model terbuka super masif seperti keluarga **Llemma** (basis korpus *Llama* dengan bobot koding raksasa).
*   **Model Spesialis Matematika Formal (Specialized Models):** Keluarga petarung garis keras MCTS lama dari zaman prasejarah **GPT-f**, ReProver, LLMStep, Lean-STaR, Hypertree Proof Search, dan si primadona Tiongkok masa kini yang ditakuti: **InternLM2-StepProver**.

**Tabel 1 (Rekapitulasi Cepat Hasil *miniF2F-test*):**
*(Saya selaku dosen akan menyarikan langsung analisa numeriknya untuk menghemat kapasitas membaca teks Anda).*

Pada gelanggang tarung `miniF2F-test` (Matematika olimpiade/SMA):
*   Di kategori pertarungan tebak instan (Single-pass Whole-proof): DeepSeek-Prover-V1.5-RL memancangkan rekor tertinggi dengan tembusan presisi **60,2%** akurasi hanya bermodalkan *sample budget* 16 x 6400 (atau \~102 ribu upaya muntahan teks acak mentah). Ini melindas secara brutal pendahulunya DeepSeek V1 (yang kandas di rawa 50,0%).
*   Di arena pertarungan kelas berat Mahzab MCTS (*Tree Search Methods*): Ketika raksasa *open source* nomor satu dunia, *InternLM2-StepProver*, membanggakan rekor *state-of-the-art* miliknya di angka 54,5% (dengan memeras energi pencarian 64 cabang MCTS x 32 turunan x 100 kedalaman komputasi). Maka, kombinasi fusi maut **DeepSeek-Prover-V1.5-RL + RMaxTS** meruntuhkan kemapanan batas cakrawala sains matematika masa kini, dan mencetak bendera rekor dunia absolut SOTA yang baru: **63,5%** kelulusan!

**Tabel 2 (Rekapitulasi Cepat Hasil *ProofNet-test*):**
Pada gelanggang tarung neraka `ProofNet-test` (Matematika abstrak tingkat Sarjana):
*   Kembali, di kelas MCTS *Tree Search*, pemegang takhta sebelumnya *InternLM2-StepProver* terhenti meratap di angka 18,1%.
*   Kombinasi dewa **DeepSeek-Prover-V1.5-RL + RMaxTS** menduduki takhta SOTA dunia yang baru dengan mencetak skor gila sebesar **25,3%** (Naik mendadak secara mutlak nyaris melampaui separuh lebih batas capaian raksasa sebelumnya).

### 4.2 Mengkaji Ulang Efektivitas Strategi Pelatihan dalam Skala Pencuplikan Besar (*Re-Examining the Effectiveness of Training Strategies on Large-scale Sampling*)

Bagian penutup eksperimen ini menyingkap misteri di balik layar.

**Peningkatan Umum Melalui Pembelajaran Penguatan (General Enhancement of Reinforcement Learning).**
Mari kita bedah kontribusi *Reinforcement Learning* (RL) di sini. Di Tabel 3 (bisa kita telaah narasinya), di komparasikan dua model klon mutlak kita: Satu model hanya dicerdas-haluskan dengan metode SFT biasa (Supervised), dan kembarannya diinjeksi teknik RL. Hasil di segala lini memvalidasi dogma empiris: Varian RL **selalu secara konstan mendominasi** dan mengungguli varian saudaranya, tidak peduli mau diterjunkan di medan perang CoT (Chain-of-thought/berpikir) MUPUN non-CoT. RL memoles tajam otot insting model dalam menenggak (*aligning*) parameter pembuktian kaku kompilator Lean 4. Ini adalah pembuktian ilmiah konkret bahwa menjejalkan sinyal umpan balik Lean Compiler (Verifikasi Sinyal Ekstrinsik mutlak) secara daring untuk menghajar gradien RL *Policy* memiliki dampak kosmik dalam meningkatkan kecerdasan AI.

**Strategi CoT (Berpikir Logis Berantai), non-CoT (Instan), dan Strategi Hibrida Campuran (Mixture Strategy).**
Model diwajibkan untuk menenggak 2 macam pil strategi:
1.  Menuntun algoritma untuk "berceloteh memaparkan pemikiran di luar kepala" dalam bahasa manusia di *Prompt* komentar Lean (CoT) sebelum naskah taktik dieksekusi. Ini memaksa LLM menjahit perencanaan wawasan arsitektur makro di kepalanya.
2.  Menuntun model membisu tanpa bacot logika naratif, yang mendesak model memanfaatkan efisiensi pemanggilan taktik tingkat-tinggi (*high-level tactics*) bawaan perpustakaan kompilator Lean murni.

Di luar dugaan, para ilmuwan mencampurkan 50-50 (*Mixture Strategy / non-CoT & CoT*) kelembagaan alokasi *Sample Budget* kepada dua kubu pemicu *prompts* radikal ini secara bergiliran. Hasil kolaboratif perpaduan hibrida ini sukses memahat loncatan raksasa tambahan *bootstrapping* di angka puncak akhir (Tabel 3 baris bawah), mencapai skor nirwana di titik **63,5%** pada `miniF2F-test`. Kolaborasi dua kutub (pemikir metodis natural dan penembak buta taktik logis kompilator) mengekstrak segala saripati jus kemahiran penalaran mesin neural tersebut.

*(Contoh cuplikan skrip aktual CoT dan non-CoT yang mengerikan dapat Anda saksikan di Apendiks A di jurnal versi asli, di mana mesin berdebat dengan metode aljabar matriks dan manipulasi trigonometri kompleks untuk mensolusi pertidaksamaan, layaknya seorang kandidat doktor ilmu matematika murni).*

---

### Analisis Akademik Gambar 5: Studi Ablasi Desain Algoritmik RMaxTS

Sebagai pakar komputasi, Anda harus selalu mempertanyakan: "Apakah seluruh komponen canggih ini benar-benar berguna, atau hanya hiasan (*gimmick*)?" Mari kita bedah buktinya di **Gambar 5**.

*   **Deskripsi Visual:** Gambar 5 menyajikan grafik kurva eksponensial (kiri) dan sebuah tabel angka (kanan). Grafik menelusuri tingkat keberhasilan (Sumbu Y, `Pass@K`) seiring bertambahnya tebakan (Sumbu X, `Sample Budget` hingga 6400).
*   **Kurva Ujian (Ablasi Moduler):**
    *   **Garis Merah Teratas:** `RMaxTS` (Formasi Penuh Sempurna kita, Sang Juara). Melampaui 58.6% akurasi.
    *   **Garis Oranye:** `RMaxTS (w/o state)` (Kita preteli/potong fitur pasokan asupan *Tactic State* Kompilator dari model MCTS kita). Akibatnya kurva ini tertinggal dan merosot jauh di bawah formasi penuh.
    *   **Garis Ungu & Biru:** `RMaxTS (UCB1)` & `UCT (w/o R_intrinsic)`. Ini adalah eksperimen untuk membedah instrumen RMaxTS kita. Jika rumus penawar ilusi (DUCB / *Discounted* UCB) yang elegan di Bab 3.3 tadi kita copot dan kita ganti lagi ke rumus uzur UCB1 (Garis ungu), kurva melorot drastis. Lebih fatal lagi, jika suntikan "Penghargaan Intrinsik" (Rasa Ingin Tahu) kita gembosi dan dimatikan total (*w/o R_intrinsic*), kurva biru tersebut babak belur di angka 58.2%.
    *   **Garis Hijau Bawah:** `w/o Tree Search`. (Single-Pass biasa, sebagai *baseline* pembanding).
*   **Kesimpulan Akademis Absolut:** Melalui demonstrasi amputasi (*ablation study*) fungsi matematika ini, kita berhasil mengesahkan bahwa seluruh inovasi *Discounted* UCB dan injeksi formula RMax (Bonus Eksplorasi) yang diusulkan oleh para ilmuwan di jurnal ini memiliki kontribusi kausal yang krusial (*critical component*) secara fungsional. Tidak ada satupun yang sekadar dekorasi akademis.

---

## 5. Kesimpulan, Keterbatasan, dan Pekerjaan Masa Depan (*Conclusion, Limitation, and Future Work*)

Dalam panggung perkuliahan penutup ini, kita mensintesiskan keseluruhan kontribusi fundamental dari makalah tersebut.

Kita menghadirkan **DeepSeek-Prover-V1.5**, silikon otak model bahasa dengan 7 miliar rajutan kognitif parameter yang dengan pongahnya merobohkan seluruh kompetitor model sumber-terbuka berskala-jagat dalam gelanggang olimpiade matematika Lean 4 *formal theorem proving*. 

Ekosistem orkestrasi perancangan *pipeline* LLM raksasa ini terinspirasi meniru cetak biru revolusioner sang Dewa Catur, **AlphaZero**. Penggunaan putaran *expert iteration* dipadukan sintesis asimilasi data (sebagai ganti permainan *self-play* di catur), di mana Oracle Kompilator bertindak mempresentasikan simulasi hukum alam (*world model*), sangatlah elegan. Dalam dimensi makro *Reinforcement Learning* tersebut, perancangan modul Pencarian Pohon (MCTS) hibrida fusi inovatif **RMaxTS** sukses memperlebar perburuan algoritma (*exploration aspect*) jauh melampaui batasan imajinasi masa lalu.

**Keterbatasan dan Eksplorasi Ekspedisi Masa Depan (*Limitations & Future Work*):**
Fokus RMaxTS masih berpusar berat memecahkan dilema "Eksplorasi" (*exploration*). NAMUN, dimensi spektrum berlawanannya: "Eksploitasi" (*exploitation*, yakni membantai pohon pencarian di titik pasti probabilitas sukses), masih sangat perawan dan kurang terjamah. 

Satu haluan masa depan yang luar biasa menggairahkan adalah membangun sosok arsitektur **Critic Model** (Model Kritikus). Sang Kritikus ini akan diutus bertugas menaksir dan menghakimi blok-blok koding pembuktian yang belum utuh (setengah jadi), lalu menebas dan memangkas dahan (*prune search branches*) yang memancarkan aura kegagalan. Model Kritikus pembuktian paruh-waktu semacam ini secara diam-diam akan mengalkulasi distribusi atribusi kredit temporer (Sutton, 1984), merekatkan dan memecah belah selisih skor utilitas langkah demi langkah dari umpan balik hasil akhir (Arjona-Medina et al., 2019). Membangun dan melatih model raksasa untuk memetakan "perencanaan wawasan jarak jauh" (*long planning paths*) dan memasok nilai penghargaan penuntun (*guidance rewards*) tetap akan menjadi tantangan saintifik "Holy Grail" dan misteri yang mengerikan (Sorg et al., 2010), namun jelas menuntut pengorbanan investigasi berdarah-darah di masa purnawaktu mendatang.

*(Demikianlah bedah makalah akademis tingkat lanjut kita pada sesi DeepSeek-Prover-V1.5. Silakan kaji setiap detail algoritma MCTS dan RL-nya, karena ini adalah tulang punggung evolusi kognisi AI modern dalam matematika murni).*
