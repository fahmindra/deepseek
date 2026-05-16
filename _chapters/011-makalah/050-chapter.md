---
slug: ch-11-05
title: "DeepSeek-VL"
layout: chapter
---

## DeepSeek-VL - Menuju Pemahaman Visi-Bahasa di Dunia Nyata

Pada bab ini, kita akan membahas **DeepSeek-VL**, sebuah Model Visi-Bahasa (*Vision-Language Model* / VLM) sumber terbuka (*open-source*) yang dirancang secara khusus untuk aplikasi pemahaman visi dan bahasa di dunia nyata. Pendekatan arsitektural dan metodologis dari model ini distrukturkan ke dalam tiga dimensi utama:

1.  **Konstruksi Data (*Data Construction*):** Berbeda dengan model yang hanya mengandalkan data akademis sempit, tim peneliti berupaya keras memastikan data pelatihan sangat beragam, dapat diskalakan (*scalable*), dan secara ekstensif mencakup skenario dunia nyata. Data ini meliputi tangkapan layar web (*web screenshots*), dokumen PDF, *Optical Character Recognition* (OCR), bagan/grafik, hingga konten berbasis pengetahuan ahli dan buku teks. Tujuannya adalah untuk merepresentasikan konteks praktis secara komprehensif. Lebih jauh, dibangun sebuah taksonomi kasus penggunaan (*use case taxonomy*) yang diturunkan dari skenario pengguna nyata untuk menyusun dataset Pelatihan Halus Instruksi (*instruction-tuning*). Proses *fine-tuning* dengan dataset ini secara substansial meningkatkan pengalaman pengguna di dunia nyata.
2.  **Arsitektur Model (*Model Architecture*):** Mempertimbangkan efisiensi komputasi, DeepSeek-VL menggabungkan **Penyandi Visi Hibrida (*Hybrid Vision Encoder*)**. Arsitektur hibrida ini secara efisien memproses gambar beresolusi tinggi ($1024 \times 1024$) dengan anggaran token (*token budget*) yang tetap, menjaga *overhead* komputasi tetap rendah. Pilihan desain ini memastikan model mampu menangkap semantik makro maupun detail mikroskopis pada berbagai tugas visual.
3.  **Strategi Pelatihan (*Training Strategy*):** Sebuah postulat krusial diajukan di sini: *Sebuah VLM yang tangguh haruslah, pertama dan terutama, memiliki kemampuan bahasa yang kuat.* Untuk memastikan tidak hilangnya kecerdasan bahasa (LLM) selama prapelatihan visi, diteliti sebuah strategi **Prapelatihan VL Gabungan (*Joint VL pretraining*)**. Pelatihan ini mengintegrasikan data teks LLM sejak awal untuk mengelola "dinamika kompetitif" antara modalitas visi dan bahasa. Dimulai dengan rasio yang lebih condong ke teks, rasio ini perlahan disesuaikan (*warmup*) untuk memfasilitasi integrasi yang seimbang.

Keluarga DeepSeek-VL (model berparameter 1.3B dan 7B) menunjukkan kapabilitas chatbot visi-bahasa yang superior. Model ini mencapai kinerja *state-of-the-art* (SOTA) atau kompetitif di berbagai tolok ukur (*benchmarks*) visi-bahasa untuk ukuran kelasnya, sekaligus mempertahankan kinerja yang kuat pada tolok ukur khusus bahasa.

---

## 2. Pendahuluan

Keberhasilan luar biasa dari Model Bahasa Besar (*Large Language Models* / LLMs) seperti GPT-4 dan Claude telah memicu permintaan akan antarmuka serbaguna yang mampu memproses berbagai modalitas selain sekadar teks. Merespons hal ini, muncul **Large Multimodal Models (LMMs)** seperti GPT-4V (OpenAI) dan Gemini (Google). Model-model ini bukan lagi sekadar pemroses teks, melainkan "asisten serbaguna" yang memahami instruksi yang melintasi domain penglihatan (*vision*) dan bahasa.

Namun, di dunia *open-source*, ada kesenjangan (*gap*) yang sangat lebar. Walaupun banyak LMM sumber terbuka mulai bermunculan dan mencapai skor bagus di ujian akademis (seperti LLaVA), mereka sering kali gagal total memberikan pengalaman yang mulus saat digunakan di skenario dunia nyata. Kesenjangan kinerja ini disebabkan oleh beberapa faktor fundamental:

*   **Fokus yang Salah pada Fase Instruksi:** Banyak solusi sumber terbuka menghabiskan terlalu banyak komputasi pada fase *instruction tuning* (pelatihan instruksi akhir) ketimbang *pretraining* (prapelatihan dasar). Padahal, untuk menanamkan pengetahuan dunia yang kaya, LMM butuh prapelatihan ekstensif pada spektrum data visi-bahasa yang sangat luas.
*   **Sindrom Dataset Akademis:** Kebiasaan mencampur berbagai dataset akademis saat pelatihan instruksi mungkin mendongkrak nilai ujian (*benchmark*), namun sering kali gagal merepresentasikan cara manusia menggunakan AI di dunia nyata.
*   **Kendala Arsitektur (Resolusi Rendah):** Kebanyakan model sebelumnya hanya "menempelkan" *vision transformer* (ViT) ke LLM pada resolusi yang sangat rendah (misal $336 \times 336$ atau $448 \times 448$). Padahal, tugas dunia nyata seperti OCR (membaca teks kecil di foto) sangat menuntut resolusi tinggi.
*   **Amensia Bahasa (*Catastrophic Forgetting*):** Saat model dilatih untuk melihat gambar, secara tak terduga, kemampuan tata bahasa dan logika teks murninya justru merosot. Diperlukan strategi pelatihan yang melestarikan kemampuan bahasa saat model belajar "melihat".

Merespons tantangan ini, DeepSeek-VL dibangun. Model ini menyeimbangkan prapelatihan yang masif, kurasi data berdasarkan taksonomi kasus nyata, desain arsitektur resolusi tinggi yang efisien, dan rasio pencampuran data yang menjaga kecerdasan bahasa aslinya.

---

## 3. Konstruksi Data (*Data Construction*)

Dataset yang besar dan beragam adalah "bahan bakar" terpenting dalam melatih LMM. Dalam ekosistem DeepSeek-VL, data dibagi menjadi dua kategori utama:
1.  **Data Prapelatihan Visi-Bahasa (*VL Pretraining Data*):** Data teks-visual skala masif dari berbagai sumber, bertujuan membangun pemahaman fundamental lintas-modal (*cross-modal*).
2.  **Data Pelatihan Halus Instruksi Terselia (*VL Supervised Fine-Tuning / SFT Data*):** Data yang jauh lebih kecil namun terkurasi sangat ketat, bertujuan mengajari model menyelesaikan perintah spesifik layaknya seorang asisten.

### 3.1 Data Prapelatihan Visi-Bahasa (*VL Pretraining Data*)

Data ini digunakan untuk memanaskan (*warmup*) Adaptor VL di Tahap 1, dan untuk melatih seluruh sistem di Tahap 2. Komposisi datanya dirangkum dalam **Tabel 1**.

**Tabel 1: Ringkasan Dataset pada Tahap Prapelatihan VL Gabungan**

| Kategori | Dataset | Rasio |
| :--- | :--- | :--- |
| **Interleaved image-text** (Teks-Gambar berselang-seling) | MMC4, Wikipedia EN & CN, Wikihow, PDF/Epub internal | 13.1% |
| **Image caption** (Keterangan Gambar) | Capsfusion, TaiSu, Detailed Caption | 11.1% |
| **Table and chart** (Tabel dan Bagan) | Chart2text, Geo170K, Ureader, Unichart, M-paper, ScienceQA, ScreenQA, SciGraphQA, Paper2figure, Widget Captioning, Screen2words, Refexp | 2.1% |
| **Web Code** (Kode Web) | Websight, Python plots scraped from GitHub | 0.4% |
| **Scene text OCR** (OCR Teks Lanskap) | ArT, MLT-17, LSVT, UberText, Coco-text, RCTW-17, ReCTS, TextOCR, OpenVINO, HierText | 1.2% |
| **Document OCR** (OCR Dokumen) | arXiv rendered markdown, Dataset OCR dokumen Mandarin internal | 2.1% |
| **Text-only corpus** (Korpus Teks Murni) | DeepSeek-LLM 2T text corpus | **70.0%** |

*(Catatan Dosen: Perhatikan bahwa 70% data yang disuapkan adalah teks murni tanpa gambar. Ini adalah strategi krusial untuk mencegah model melupakan cara bernalar menggunakan bahasa saat ia dipaksa belajar memproses gambar.)*

Penjelasan mendetail setiap kategori:
*   **Interleaved image-text:** Artikel di mana gambar dan teks bercampur (seperti artikel blog). Ini mengajari model *in-context learning* (memahami gambar berdasarkan teks yang mengapitnya).
*   **Image caption:** Pasangan gambar dan deskripsi eksplisit.
*   **Table and chart:** Melatih model bernalar menggunakan data grafis (kuantitatif).
*   **Web Code:** Mengajari model merekonstruksi kode pemrograman HTML/Python hanya dengan melihat UI (Antarmuka) atau hasil plot grafik.
*   **Document OCR:** Sangat krusial. Karena tidak ada dataset OCR Mandarin-Inggris publik yang memadai, tim mengonversi 1.4 juta makalah arXiv dari PDF kembali ke pasangan gambar-teks. Mereka juga merender (*rendering*) 860K buku bahasa Inggris dan 180K bahasa Mandarin dari *Anna's Archive* menjadi pasangan gambar dan HTML.

### 3.2 Data Pelatihan Halus Instruksi Terselia (*Supervised Fine-Tuning / SFT Data*)

Tahap 3 difokuskan pada data instruksi. Kumpulan data bersumber dari dataset SFT publik (seperti LLaVA1.6-GPT4V, ShareGPT4V) dan data internal yang dikurasi dengan ketat.

Untuk memastikan model berguna di dunia nyata, tim mengumpulkan kasus pengujian autentik dari GPT-4V dan Gemini di internet, dan mendesain sebuah **Taksonomi Kasus Penggunaan** (*Use Case Taxonomy*) yang komprehensif, seperti yang ditunjukkan pada **Tabel 3**.

**Tabel 3: Taksonomi untuk Data SFT Internal**
*(Ini adalah pemetaan terstruktur bagaimana AI harus memahami berbagai jenis pertanyaan visual dari manusia).*

| Kategori Utama | Deskripsi | Kategori Sekunder | Kategori Tersier |
| :--- | :--- | :--- | :--- |
| **Recognition** (Pengenalan) | Menguji kemampuan mendeskripsikan konten gambar dasar tanpa penalaran tingkat tinggi. | *Global Description* | Deskripsi tema, event, lokasi, emosi, gaya seni, makanan. |
| | | *Local Description* | Deskripsi objek yang ditunjuk, posisi, pengenalan orang, logo, berhitung, mata uang. |
| | | *OCR and Transcription* | Transkripsi teks cetak, tulisan tangan, format spesifik. |
| **Conversion** (Konversi) | Mengubah visual menjadi format lain (misal: UI menjadi kode). | *Image to Code* | UI ke Kode, Bagan ke Kode, Formula ke Kode. |
| | | *Image to Text* | Image ke Prompt, Ringkasan, Kreasi berbasis gambar. |
| **Analysis** (Analisis) | Membutuhkan pengetahuan domain dan logika untuk memahami gambar. | *Data Chart Analysis* | Interpretasi grafik dan tabel. |
| | | *Professional Chart Analysis* | Diagram sirkuit, *flowchart*, peta, partitur musik, denah lantai. |
| | | *Professional Image Analysis* | Gambar sensor, gambar biologis/medis. |
| | | *Encyclopedia Knowledge* | Pengetahuan seni, alam, sejarah, hiburan. |
| **Commonsense Reasoning** (Penalaran Akal Sehat) | Menguji pemahaman AI tentang "akal sehat" dunia fisik. | *Relationship Reasoning* | Hubungan antar-objek, spasial, spesies. |
| | | *Function / Environment* | Fungsi hardware, analisis status lingkungan. |
| | | *Anomaly / Humor* | Deteksi cacat/kecelakaan, memahami lelucon/meme. |
| **Logical Reasoning** (Penalaran Logis) | Membutuhkan logika kaku untuk memecahkan masalah. | *Mathematical Reasoning* | Aljabar, geometri planar, geometri solid. |
| | | *Other Logical* | Fisika, Kimia, Biologi, IQ Test. |
| **Evaluation** (Evaluasi) | Meminta model mengevaluasi gambar berdasar kriteria tertentu. | - | Evaluasi estetika, realitas, saran perbaikan. |
| **Multi-graph** (Multi-gambar) | Menganalisis sekumpulan gambar. | *Temporal Sequence* | Prediksi *event*, analisis video/perilaku. |
| | | *Multi-graph Comparison* | Mencari perbedaan, asosiasi gambar. |
| **Safety** (Keamanan) | Menilai keselamatan jawaban. | - | Pertanyaan sugestif, injeksi *prompt*, pertanyaan konterfaktual. |

---

## 4. Pendekatan dan Arsitektur (*Approach & Architecture*)

Sistem DeepSeek-VL terdiri dari tiga modul makro: Penyandi Visi Hibrida (*Hybrid Vision Encoder*), Adaptor Visi (*Vision Adaptor*), dan Model Bahasa (*Language Model*).

### 4.1 Arsitektur Model

**1. Penyandi Visi Hibrida (*Hybrid Vision Encoder*)**
Sebagian besar model hanya menggunakan SigLIP (atau CLIP) sebagai "mata". Masalahnya, SigLIP didesain untuk menangkap semantik global (makna keseluruhan gambar) pada resolusi rendah (seperti $384 \times 384$). Jika Anda menyodorkan kuitansi panjang (resolusi tinggi), SigLIP akan menjadi "buta" (*CLIP-blind pairs*) karena tidak didesain untuk membaca teks mikroskopis.

Untuk mengatasinya, DeepSeek-VL menggabungkan **dua jenis mata**:
*   **SigLIP-L:** Mengurus fitur semantik makro (resolusi $384 \times 384$).
*   **SAM-B (Segment Anything Model):** Sebuah penyandi ViTDet (Vision Transformer untuk Deteksi) yang sudah diprapelatih. SAM menerima gambar beresolusi tinggi $1024 \times 1024$ dan mengekstrak fitur *low-level* (teks kecil, ujung garis, batas objek).

*Mekanisme Pemrosesan:*
Gambar di- *resize* menjadi $1024 \times 1024$. SAM-B mengonversinya menjadi *feature map* berukuran $64 \times 64 \times 256$ (resolusi spasial $64 \times 64$, dengan $256$ *channels*). 

**2. Adaptor Visi-Bahasa (*Vision-Language Adaptor*)**
Adaptor ini berfungsi sebagai "jembatan penerjemah" dari bahasa visual ke bahasa teks. 
*   Fitur dari SAM-B ($64 \times 64 \times 256$) diinterpolasi menjadi $96 \times 96 \times 256$.
*   Fitur ini dilewatkan ke dua lapisan konvolusi (*convolutional layers*) dengan *stride* 2. Ini memperkecil dimensi spasial menjadi $24 \times 24$, tetapi memperbesar *channel* menjadi $1024$. 
*   Hasilnya di-*reshape* (diratakan) menjadi matriks berukuran $576 \times 1024$. 
*   Secara paralel, *feature map* dari SigLIP juga diproses menjadi ukuran $576 \times 1024$.
*   Kedua representasi ini **digabungkan (*concatenated*)** di sepanjang dimensinya, menghasilkan 576 "token visual" dengan dimensi 2048. 
*   Token-token ini dilewatkan melalui fungsi aktivasi GELU dan sebuah lapisan *embedding* linear akhir (*MLP*) untuk memproyeksikannya ke ruang dimensi LLM.
*   *Hasil Akhir:* Gambar resolusi raksasa $1024 \times 1024$ dipadatkan menjadi hanya **576 token**, sangat hemat biaya komputasi untuk masuk ke LLM!

**3. Model Bahasa (*Language Model*)**
Dibangun di atas arsitektur LLaMA (Pre-Norm, RMSNorm, SwiGLU). Memiliki lapisan *intermediate* FFN sebesar $\frac{8}{3} d_{model}$ dan menggunakan *Rotary Position Embedding* (RoPE). Model tersedia dalam ukuran 1.3B dan 7B parameter.

---

### 4.2 Alur Pelatihan (*Training Pipelines*)

Pelatihan dilakukan dalam tiga tahap berurutan. Mari kita analisis **Gambar 3** yang memvisualisasikan hal ini.

#### Analisis Akademik Gambar 3: Tiga Tahap Prapelatihan dan SFT
*   **Deskripsi Visual:** Gambar ini menampilkan tiga blok arsitektur berturut-turut dari kiri ke kanan. Ikon "Kepingan Salju" (Snowflake) berarti parameter dibekukan (*frozen* / tidak diperbarui). Ikon "Api" (Flame) berarti parameter dicairkan dan dilatih (*trainable*).
*   **Tahap 1 (Stage 1): Training VL Adaptor.** 
    *   *Input:* Pasangan Teks-Gambar pendek.
    *   *Proses:* SigLIP dan SAM-B dibekukan. DeepSeek LLM dibekukan. HANYA Adaptor Visi (oranye) yang dilatih (berlambang api). 
    *   *Tujuan Akademis:* Mengajari jembatan adaptor cara memproyeksikan fitur gambar murni ke dalam ruang vektor tata bahasa yang dipahami LLM.
*   **Tahap 2 (Stage 2): Joint VL Pre-training.**
    *   *Input:* Kombinasi gambar-teks yang diselang-seling (artikel panjang) PLUS urutan bahasa murni.
    *   *Proses:* LLM (biru) kini ikut DICAIRKAN (berlambang api). Penyandi Visi tetap dibekukan.
    *   *Tujuan Akademis:* Ini adalah fase terpenting di mana LLM mulai memasukkan konsep visual ke dalam kognisi linguistiknya tanpa merusak kecerdasan bahasa murninya.
*   **Tahap 3 (Stage 3): Supervised Finetuning.**
    *   *Input:* Data chat interaktif multi-modal dan chat teks murni.
    *   *Proses:* SigLIP kini ikut dicairkan. SAM-B tetap beku (karena memori GPU tidak muat).
    *   *Tujuan Akademis:* Memoles kemampuan instruksional agar model bisa berdialog seperti manusia.

---

#### Strategi Krusial di Tahap 2: Mencegah *Catastrophic Forgetting*

Mari kita dalami **Gambar 4**. Ini adalah salah satu eksperimen sains data paling penting dalam makalah ini.

#### Analisis Akademik Gambar 4: Dampak Rasio Modalitas (Gambar vs Teks)
*   **Deskripsi Visual:** Enam grafik garis (*line charts*) memonitor performa model selama Tahap 2. Sumbu X adalah langkah pelatihan (*Step*). Sumbu Y adalah tingkat akurasi (semakin tinggi semakin baik, kecuali untuk *Pile-test* PPL di mana semakin rendah semakin baik).
*   **Variabel Independen:** Warna garis mewakili proporsi pencampuran data. 
    *   Garis ungu pekat: 100% data gambar, 0% teks murni.
    *   Garis oranye paling terang: 10% data gambar, 90% teks murni.
*   **Grafik Bawah Kanan (Pile-test):** Ini adalah ujian *teks murni* (pemahaman bahasa). Perhatikan garis ungu pekat (100% gambar) melonjak tajam ke atas (yang berarti *error/perplexity* meningkat parah). Ini membuktikan bahwa jika Anda melatih LLM HANYA dengan gambar, "otak" bahasanya akan hancur (lupa cara berbahasa).
*   **Grafik Lainnya (Ujian Multi-modal):** Garis ungu pekat (100% gambar) memimpin di atas, namun garis merah (75% gambar : 25% teks) dan garis merah gelap (60% gambar : 40% teks) menempel sangat ketat di bawahnya tanpa kehilangan performa visual yang signifikan.
*   **Kesimpulan Akademis:** Terdapat hubungan kompetitif (tarik-menarik) antara melatih mata (visual) dan menjaga logika otak (bahasa). Berdasarkan kurva ini, para peneliti menetapkan **Rasio Optimal 70% Teks : 30% Gambar**. Rasio ini menjadi *sweet-spot* emas di mana model mempertahankan kepintaran bahasanya sembari sukses menyerap kejeniusan visual.

---

### 4.3 Strategi "Pemanasan" Modalitas (*Modality Warmup*)

Meskipun rasio 70:30 adalah target akhir, jika kita menyuapkan gambar secara tiba-tiba sejak Langkah 1, LLM akan "kaget" dan metrik bahasanya akan anjlok sementara (*loss spike*).

Solusinya dievaluasi pada **Gambar 9**.

#### Analisis Akademik Gambar 9: Modality Warmup
*   **Sumbu X & Y:** Langkah pelatihan vs Perplexity/Accuracy.
*   **Kurva Data:** Garis abu-abu = tanpa *warmup* (langsung rasio tetap). Garis oranye merah = menggunakan *warmup*.
*   **Kesimpulan Akademis:** Daripada langsung menggunakan rasio 70:30, peneliti memulai dengan 100% teks murni di langkah awal, lalu perlahan "membuka keran" data gambar secara bertahap hingga mencapai batas target. Hal ini terbukti menstabilkan *error rate* bahasa (grafik kiri) sejak awal pelatihan tanpa mengorbankan kualitas akhir visi (grafik tengah dan kanan).

*(Catatan Dosen: Perhatikan juga teknik **Modality Group Training**. Jika dalam satu antrean batch GPU ada data teks pendek dan data gambar resolusi tinggi, GPU akan lambat karena harus menunggu data gambar selesai diproses. Peneliti mengelompokkan data berdasarkan jenisnya: batch A khusus teks, batch B khusus gambar. Ini mempercepat efisiensi pelatihan secara masif tanpa merusak gradien matematika, dibuktikan pada **Gambar 8**).*

---

## 5. Evaluasi (*Evaluation*)

DeepSeek-VL diuji melawan LMM tertutup (Gemini Pro, GPT-4V) dan terbuka (LLaVA, Qwen-VL, CogVLM) pada berbagai ujian (MMMU, MMBench, OCRBench, dll).

### 5.1 Evaluasi Tolok Ukur Multi-Modal Publik

**Tabel 5 dan 6** merangkum keunggulan DeepSeek-VL.
*(Rangkuman Tabel Akademis)*:
*   Pada MMBench (MMB), DeepSeek-VL 7B mencetak skor **73.2**, mengalahkan semua model sumber terbuka sekelasnya (CogVLM 63.7, LLaVA-Next 67.4).
*   Yang paling fenomenal adalah **OCRBench** (kemampuan membaca teks). DeepSeek mencetak **456**, jauh meninggalkan LLaVA (331). Ini memvalidasi keputusan arsitektur menggunakan SAM-B ($1024 \times 1024$) daripada ViT biasa.

### 5.2 Evaluasi Bahasa Murni

**Tabel 7** adalah bukti nyata mengapa strategi *Joint Pretraining* (menjaga 70% data teks) itu berhasil.

| Benchmark | DeepSeek-VL 1B Chat | DeepSeek-VL 7B Chat | DeepSeek-LLM 7B Chat (Tanpa Visi) |
| :--- | :---: | :---: | :---: |
| **HellaSwag** | 56.0 | 68.4 | **68.5** |
| **MMLU** | 32.5 | **52.4** | 49.4 |
| **GSM8K** (Matematika) | 18.0 | 55.0 | **63.0** |

*Analisis:* LMM sering menjadi bodoh dalam bahasa murni setelah dilatih visi. Namun, DeepSeek-VL 7B Chat tetap memiliki HellaSwag 68.4 (identik dengan versi aslinya 68.5). Ia bahkan meningkat di MMLU (dari 49.4 menjadi 52.4). Satu-satunya metrik yang merosot kompetitif adalah ujian murni Matematika tingkat lanjut (GSM8K), menegaskan bahwa di kapasitas terbatas 7B, penalaran logis abstrak matematika terpaksa "mengalah" memberi ruang memori pada fitur visi.

---

### 5.3 Evaluasi Manusia (*Human Evaluation*)

Karena *benchmark* statis tidak selalu mencerminkan perasaan pengguna nyata, dilakukan uji coba manusia (A/B testing) pada 100 pertanyaan nyata. 

#### Analisis Akademik Gambar 6: Penilaian Manusia Langsung
*   **Sumbu X:** Kategori (Skor Total, Pengenalan, Konversi Kode, Analisis, Akal Sehat, dll).
*   **Sumbu Y:** Skor manusia (1 hingga 10).
*   **Batang:** InternLM (Biru), CogVLM (Oranye), DeepSeek-VL (Hijau), GPT-4V (Abu-abu).
*   **Kesimpulan:** Batang abu-abu (GPT-4V) mendominasi telak. Namun, di antara semua LMM terbuka, **DeepSeek-VL (Hijau)** menduduki peringkat tertinggi kedua di mayoritas skenario (terutama di bagian Konversi ke Kode, Analisis, dan Akal Sehat).

---

## 6. Visualisasi Kualitatif (Bukti Keandalan Dunia Nyata)

Makalah ini menyertakan tangkapan layar langsung (*Screenshots*) untuk membuktikan klaim mereka.

*   **Gambar 1:** DeepSeek mampu memahami diagram alir (*flowchart*) lalu menuliskan **kode Python utuh** yang merepresentasikan logika *flowchart* tersebut secara sempurna.
*   **Gambar 2:** Model ditanya "Apakah pesepeda ada di kiri atau kanan tas wanita?". Model tidak hanya menjawab "Kiri", tetapi **memberikan 4 poin alasan logika spasial spasial 3D**, mulai dari letak perspektif tas, letak pesepeda di belakang, efek gerak (*motion blur*), hingga posisi jalan.
*   **Gambar 14:** Membuktikan kekuatan OCR resolusi tinggi. DeepSeek mengonversi tabel numerik kompleks dari tangkapan layar makalah PDF menjadi format tabel Markdown secara nir-cela, sementara model *open-source* lain menyerah (*"sorry I cannot process"*).
*   **Gambar 12:** Diberikan *screenshot* *code editor* dengan satu kata yang salah. Model berhasil mencari kutu (*debugging*) visual, menunjuk baris mana indeks array-nya salah ketik dari `sequence[1]` menjadi `sequence[0]`.

---

## 7. Kesimpulan, Limitasi, dan Pekerjaan Masa Depan

Sebagai rangkuman, laporan teknis ini memperkenalkan **DeepSeek-VL**, seri LMM dengan parameter 1.3B dan 6.7B (7B). Keberhasilan arsitektur ini membantah metode lama yang hanya menempelkan proyektor visi murahan pada LLM.

**Pilar Keberhasilan DeepSeek-VL:**
1.  **Prapelatihan Visi-Bahasa Gabungan (*Joint Pretraining*):** Mencegah amnesia kognitif pada kecerdasan bahasa asli (menggunakan rasio 70% data teks).
2.  **Pemadatan Resolusi Tinggi:** Menggunakan Penyandi Visi Hibrida (SigLIP + SAM-B) yang mampu mengunci gambar raksasa $1024 \times 1024$ ke dalam 576 token tanpa merusak presisi OCR.
3.  **Strategi Kurasi SFT berbasis Taksonomi:** Memastikan asisten berguna dalam keseharian manusia, bukan sekadar "penjawab soal ujian".

Model ini di-*open-source*-kan sepenuhnya untuk memicu laju penelitian global di era AI generatif yang semakin tertutup. Di masa depan, arsitektur ini diproyeksikan akan dikawinkan dengan teknologi **Mixture of Experts (MoE)** untuk memompa jumlah parameter efektif tanpa meledakkan biaya komputasi inferensi harian.

*(Akhir dari Modul Diktat Kuliah DeepSeek-VL)*