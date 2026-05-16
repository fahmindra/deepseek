---
slug: ch-11-20
title: "DeepSeek-OCR"
layout: chapter
---

## DeepSeek-OCR: Kompresi Optik Konteks

Dalam bab ini, kita akan membahas **DeepSeek-OCR** sebagai sebuah investigasi awal ke dalam kelayakan mengompresi konteks panjang (*long contexts*) melalui pemetaan optik 2D (*optical 2D mapping*). 

Sistem DeepSeek-OCR terdiri dari dua komponen utama: 
1.  **DeepEncoder**, dan 
2.  **DeepSeek3B-MoE-A570M** yang berfungsi sebagai dekoder (*decoder*). 

Secara spesifik, **DeepEncoder** bertindak sebagai mesin inti (*core engine*). Mesin ini dirancang secara khusus untuk mempertahankan aktivasi yang rendah (*low activations*) di bawah *input* (masukan) beresolusi tinggi, namun di saat yang bersamaan mampu mencapai **rasio kompresi yang tinggi**. Tujuannya adalah untuk memastikan jumlah token visi (*vision tokens*) yang optimal dan dapat dikelola oleh memori komputasi. 

Eksperimen menunjukkan bahwa ketika jumlah token teks berada dalam batasan 10 kali lipat dari jumlah token visi (yaitu, rasio kompresi $< 10\times$), model ini dapat mencapai presisi dekode (*decoding OCR precision*) sebesar **97%**. Bahkan pada rasio kompresi yang jauh lebih ekstrem, yaitu $20\times$, akurasi OCR masih bertahan di angka sekitar **60%**. Hal ini menunjukkan potensi yang sangat menjanjikan untuk diaplikasikan pada area penelitian seperti kompresi konteks panjang historis (*historical long-context compression*) dan mekanisme kelupaan memori (*memory forgetting mechanisms*) di dalam Model Bahasa Besar (LLMs).

Lebih dari itu, DeepSeek-OCR juga mendemonstrasikan nilai kepraktisan yang tinggi. Pada tolok ukur *OmniDocBench*, model ini melampaui **GOT-OCR2.0** (256 token/halaman) hanya dengan menggunakan **100 token visi**, serta mengungguli **MinerU2.0** (rata-rata 6000+ token per halaman) sementara hanya memanfaatkan kurang dari **800 token visi**. Dalam skenario produksi industri, DeepSeek-OCR sanggup menghasilkan data pelatihan untuk LLMs/VLMs pada skala **200k+ halaman per hari** (hanya dengan menggunakan satu buah GPU A100-40G). Kode sumber dan bobot model (*model weights*) telah disediakan untuk dapat diakses secara publik.

---

### Analisis Akademis: Gambar 1 - Kinerja Rasio Kompresi DeepSeek-OCR

Sebagai pengantar visual sebelum masuk ke metodologi, mari kita analisis **Gambar 1** secara mendalam untuk memahami lanskap performa model.

**Gambar 1** terbagi menjadi dua sub-gambar, yaitu **(a)** dan **(b)**.

**Gambar 1 (a) - Kompresi pada tolok ukur Fox (*Compression on Fox benchmark*)**
*   **Deskripsi Visual:** Ini adalah sebuah diagram batang bertumpuk (*grouped bar chart*) yang dipadukan dengan grafik garis titik (*line graph*).
*   **Sumbu X (Horizontal):** Merepresentasikan jumlah rentang Token Teks dalam *Ground-Truth* (Kebenaran Dasar) per halaman dokumen yang diujikan (mulai dari rentang 600-700 token hingga 1200-1300 token).
*   **Sumbu Y Kiri (Vertikal Utama):** Merepresentasikan *Precision* (Presisi) dalam persentase (%) dari 0% hingga 100%. Sumbu ini mengukur seberapa akurat model membaca gambar. Batang-batang histogram merujuk pada sumbu ini.
*   **Sumbu Y Kanan (Vertikal Sekunder):** Merepresentasikan *Compression (x)* (Rasio Kompresi) mulai dari 0x hingga 20x. Garis putus-putus (*dotted lines*) merujuk pada sumbu ini.
*   **Elemen Data:** Terdapat dua skenario pengujian berdasarkan jumlah token visi yang diumpankan ke model, yakni **64 token visi** (warna ungu muda) dan **100 token visi** (warna biru muda).
    *   *Batang Histogram:* Menunjukkan bahwa semakin banyak token teks asli yang ada dalam satu gambar (bergerak ke kanan pada Sumbu X), Presisi model (batang) secara alamiah akan menurun secara perlahan. Namun, bahkan pada rentang ekstrem 1200-1300 token teks, presisi masih bertengger di atas 59,1% (untuk 64 token visi) dan 87,1% (untuk 100 token visi).
    *   *Garis Dotted:* Menunjukkan rasio kompresi. Terlihat jelas bahwa rasio kompresi melonjak linier ke atas. Pada 64 token visi (garis ungu putus-putus), saat dihadapkan dengan dokumen berisi 1200-1300 token teks, rasio kompresinya mendekati $20\times$. 
*   **Kesimpulan:** Gambar ini membuktikan secara empiris bahwa model mampu membaca ribuan kata dari sebuah gambar hanya dengan mengandalkan segelintir token visual, dengan tingkat degradasi keakuratan yang sangat moderat.

**Gambar 1 (b) - Performa pada OmniDocBench (*Performance on Omnidocbench*)**
*   **Deskripsi Visual:** Ini adalah diagram pencar (*scatter plot*).
*   **Sumbu X (Horizontal):** Merepresentasikan Rata-rata Token Visi per Gambar (*Average Vision Tokens per Image*), bergerak dari 1000 hingga 7000 token.
*   **Sumbu Y (Vertikal):** Merepresentasikan Performa Keseluruhan / Jarak Edit (*Overall Performance / Edit Distance*). Perhatian: Karena ini mengukur "jarak kesalahan/ *edit distance*", maka **semakin kecil nilainya (semakin ke bawah pada grafik) semakin bagus**.
*   **Elemen Data:** Berbagai model pesaing di-plot pada grafik ini. 
    *   Model *DeepSeek-OCR* (simbol bulat merah) mendominasi di area bawah (performa terbaik) *dan* di sisi kanan ekstrim (penggunaan token visi paling sedikit).
    *   Varian `DeepSeek-OCR (Tiny)` dan `GOT-OCR2.0` berada di kotak hijau "High Accuracy (ED < 0.25)". DeepSeek-OCR mencapai nilai *Edit Distance* yang setara atau lebih baik dengan jumlah token visi yang jauh di bawah 1000, memukul telak pesaing seperti seri Qwen2.5-VL, InternVL2, hingga MinerU2.0.
*   **Kesimpulan Keseluruhan Gambar 1:** DeepSeek-OCR mampu meraih performa *state-of-the-art* di antara model ujung-ke-ujung (*end-to-end models*) sembari menikmati keistimewaan menggunakan jumlah token visi yang paling sedikit di pasaran.

---

## 1. Pendahuluan

Model Bahasa Besar saat ini (*Current Large Language Models* / LLMs) dihadapkan pada tantangan komputasional yang luar biasa berat ketika memproses konten tekstual yang panjang. Hal ini disebabkan oleh sifat penskalaan kuadratik terhadap panjang urutan (*quadratic scaling with sequence length*) dari arsitektur *Transformer*. Untuk setiap penambahan kata/token baru, beban komputasi memori meningkat berlipat ganda.

Oleh karena itu, kami mengeksplorasi sebuah solusi potensial yang radikal: **memanfaatkan modalitas visual sebagai media kompresi yang efisien untuk informasi tekstual**. 

Mari kita renungkan fakta ini: sebuah gambar tunggal yang berisi teks dokumen dapat merepresentasikan informasi yang teramat kaya dengan jumlah token yang *jauh* lebih sedikit dibandingkan dengan teks digital ekuivalennya. Fakta fundamental ini mengindikasikan bahwa kompresi optik (*optical compression*) melalui penggunaan token visi dapat menghasilkan **rasio kompresi yang jauh lebih tinggi**.

Wawasan ini memotivasi kita untuk menguji ulang model visi-bahasa (*vision-language models* / VLMs) dari sudut pandang yang berpusat pada LLM (*LLM-centric perspective*). Alih-alih berfokus pada apa yang sudah biasa dilakukan manusia dengan sangat baik—yaitu Tanya Jawab Visual dasar (*Visual Question Answering* / VQA) [12, 16, 24, 32, 41]—kami berfokus pada **bagaimana penyandi visi (*vision encoders*) dapat mendongkrak efisiensi LLM dalam memproses informasi tekstual**. 

Tugas OCR (*Optical Character Recognition*), sebagai sebuah modalitas perantara yang menjembatani visi dan bahasa, menyediakan arena pengujian (*testbed*) yang ideal untuk paradigma kompresi visi-teks ini. Mengapa? Karena tugas OCR membangun sebuah pemetaan alami kompresi-dekompresi antara representasi visual dan tekstual, sembari menyajikan metrik evaluasi kuantitatif yang solid.

Oleh karena itu, kami menyajikan **DeepSeek-OCR**, sebuah VLM yang dirancang sebagai sebuah purwarupa konsep (pembuktian awal / *proof-of-concept*) untuk kompresi visi-teks yang efisien. Penelitian kami ini menyumbangkan tiga kontribusi primer:

1.  **Analisis Kuantitatif Komprehensif:** Kami menyediakan analisis kuantitatif menyeluruh tentang rasio kompresi token visi-teks. Metode kami mencapai presisi dekode OCR sebesar **96%+** pada tingkat kompresi teks $9-10\times$, sekitar **90%** pada kompresi $10-12\times$, dan sekitar **60%** pada kompresi $20\times$ pada tolok ukur Fox [21] (lihat Gambar 1(a)). Angka-angka ini mendemonstrasikan bahwa model bahasa yang ringkas (*compact*) mampu belajar untuk mendekode representasi visual yang dikompresi dengan sangat efektif. Ini mengisyaratkan bahwa LLM yang lebih raksasa dapat dengan mudah menyerap kemampuan serupa melalui desain prapelatihan yang mumpuni.
2.  **Arsitektur DeepEncoder:** Kami memperkenalkan `DeepEncoder`, sebuah arsitektur novel (baru) yang mempertahankan memori aktivasi yang rendah serta penggunaan token visi minimal *meskipun* menerima input beresolusi tinggi. Secara arsitektural, ia menghubungkan secara seri (*serially connects*) komponen *window attention* dan *global attention encoder* melalui sebuah kompresor konvolusional $16\times$. Desain ini memastikan komponen *window attention* memproses token visi dalam jumlah besar, sementara kompresor mereduksi token visi tersebut *sebelum* mereka memasuki komponen *dense global attention* yang boros memori. Hal ini sukses mencapai kompresi memori dan token yang sangat efisien.
3.  **Pengembangan DeepSeek-OCR:** Kami membangun DeepSeek-OCR berbasis pada DeepEncoder dan `DeepSeek3B-MoE` [19, 20]. Sebagaimana diilustrasikan pada Gambar 1(b), ia mencapai performa SOTA (*state-of-the-art*) dalam jajaran model ujung-ke-ujung (*end-to-end*) di *OmniDocBench*, sembari menggunakan jumlah token visi **paling sedikit**. Di luar itu, kami mempersenjatai model ini dengan kemampuan mem- *parsing* (mengurai) bagan, rumus kimia, figur geometri sederhana, hingga gambar alam untuk mendongkrak utilitas kepraktisannya. 

Dalam ringkasan, karya ini menghadirkan sebuah eksplorasi awal penggunaan modalitas visual sebagai medium kompresi yang efisien untuk pemrosesan informasi tekstual dalam ranah LLM. Melalui DeepSeek-OCR, kami membuktikan bahwa kompresi visi-teks mampu merengkuh reduksi token yang signifikan (hingga $7-20\times$) untuk tingkatan konteks historis yang berbeda-beda. Kendati saat ini masih difokuskan pada tugas OCR sebagai *proof-of-concept*, paradigma kompresi ini mendobrak probabilitas baru untuk merenungkan ulang bagaimana modalitas visi dan bahasa dapat secara sinergis disatukan demi efisiensi pemrosesan teks skala besar.

---

## 2. Pekerjaan Terkait (*Related Works*)

### 2.1 Penyandi Visi Tipikal pada Model Visi-Bahasa (VLMs)

Saat ini, VLM sumber terbuka umumnya mempekerjakan tiga varian utama dari penyandi visi (*vision encoders*). Mari kita perhatikan visualisasinya pada **Gambar 2** berikut.

### Analisis Akademis: Gambar 2 - Penyandi Visi Tipikal dalam VLM Populer

Gambar 2 memberikan diagram komparatif yang membongkar arsitektur dari tiga jenis VLM utama saat ini. Gambar ini kritis untuk dipahami karena ia memaparkan "cacat bawaan" dari masing-masing arsitektur yang berusaha diselesaikan oleh DeepSeek-OCR.

*   **1. Arsitektur Menara-Ganda (*Dual-tower architecture*) (Contoh: Vary/DeepSeekVL):**
    *   *Visualisasi Kiri:* Menampilkan dua modul yang bekerja beriringan. Satu modul `ViTDet` memproses gambar asli berskala raksasa ($1024 \times 1024$), kemudian di-*downsample* (dikompresi ukurannya). Modul `ViT` lain memproses gambar kecil ($224 \times 224$). Hasil keduanya masuk ke LLM.
    *   *Analisis:* Diperkenalkan oleh Vary [36], sistem ini memanfaatkan penyandi SAM [17] untuk menyerap gambar resolusi tinggi.
    *   *Kelemahan (Ditandai silang merah [x]):* Ia tidak mendukung *pipeline parallel* (komputasi terdistribusi antar GPU) dan prosedur prapemrosesan ganda (*two pre-processes*) membuat *deployment* (penerapan ke aplikasi nyata) menjadi sangat menyusahkan.
*   **2. Metode Berbasis Ubin (*Tile-based method*) (Contoh: Seri InternVL/DeepSeekVL2):**
    *   *Visualisasi Tengah:* Gambar raksasa (biasanya beresolusi super besar misal $4 \times 384 \times 384$) dipecah-pecah (di- *crop*) menjadi ubin-ubin (*tiles*) kecil. Setiap ubin masuk ke modul `ViT`, lalu di-*downsample*, baru masuk ke LLM.
    *   *Analisis:* Digunakan oleh InternVL2.0 [8]. Bagus untuk resolusi hiper-tinggi karena mengurangi memori aktivasi dengan memecah gambar.
    *   *Kelemahan [x]:* Resolusi *native* penyandi (kamera bawaan) umumnya sangat rendah (di bawah $512 \times 512$). Akibatnya, saat menghadapi satu gambar dokumen utuh, ia harus dipotong-potong menjadi terlampau banyak *tiles*, yang bermuara pada "ledakan" jumlah token visi (*too many vision tokens*). Visi global (gambaran utuh) juga menjadi terlalu kecil.
*   **3. Penyandian Resolusi Adaptif (*Adaptive resolution encoding*) (Contoh: Seri Qwen2(.5)/3VL):**
    *   *Visualisasi Kanan:* Menerapkan paradigma NaViT [10]. Gambar dipecah menjadi matriks kotak-kotak kecil langsung sesuai lebarnya ($w$) dan tingginya ($h$). 
    *   *Analisis:* Direpresentasikan oleh Qwen2-VL [35]. Sangat fleksibel menangani ragam resolusi.
    *   *Kelemahan [x]:* Menghasilkan jumlah token visi yang gila-gilaan (*too many vision tokens*). Karena tidak ada *down-sampling*, ia memakan *activation memory* sangat masif yang dapat menguras habis VRAM GPU. Ia juga menuntut panjang sekuens yang sangat panjang selama pelatihan dan membuat fase inferensi/jawaban (*slow inference speed*) menjadi melambat.

Kesimpulan dari Gambar 2: Ketiga paradigma ini masih belum ideal untuk melakukan "Kompresi Optik", karena pada dasarnya mereka masih menghasilkan terlalu banyak token visual untuk dicerna oleh LLM.

### 2.2 Model OCR Ujung-ke-Ujung (*End-to-end OCR Models*)

Secara historis, model *parsing* dokumen ditangani oleh saluran pipa ganda (dibutuhkan modul spesialis "Pendeketsi Kotak Teks" lalu diikuti oleh modul "Pengenal Huruf"). Dengan bangkitnya VLMs, puluhan model OCR mutakhir muncul menggunakan pendekatan *end-to-end* (satu mesin langsung keluar teks). 

Nougat [6] merintis kerangka kerja *end-to-end* untuk pembacaan makalah akademik PDF dari arXiv. GOT-OCR2.0 [38] memperlebar jangkauan dengan *parsing* gambar sintetis (bagan). Qwen-VL [35] dan InternVL [8] juga terus mendongkrak otot OCR mereka.

Akan tetapi, para akademisi ini semua meluputkan satu pertanyaan eksistensial krusial:
> *"Untuk sebuah dokumen yang memuat 1000 kata, berapa banyak token visi yang paling minimal dibutuhkan agar model dapat mendekodenya secara sempurna?"*

Pertanyaan inilah yang bernilai teramat signifikan untuk menyelidiki kebenaran dari peribahasa kuno bahwa *"Sebuah gambar bermakna seribu kata"*. Dan inilah *gap* literatur yang diserang oleh DeepSeek-OCR.

---

## 3. Metodologi (*Methodology*)

### 3.1 Arsitektur (*Architecture*)

Untuk membedah mesin utama kita, perhatikan dengan saksama **Gambar 3**.

### Analisis Akademis: Gambar 3 - Arsitektur dari DeepSeek-OCR

*   **Deskripsi Visual:** Ini adalah bagan skema jaringan saraf tiruan (*neural network architecture*) untuk DeepSeek-OCR. 
*   **Aliran Data (Kiri ke Kanan):**
    1.  **Input:** Di sisi paling kiri, sebuah gambar utuh dokumen (makalah/teks) dimasukkan ke sistem.
    2.  **Pemecahan Tambal (*Patching*):** Gambar tersebut dipotong menjadi sejumlah $n \times 16 \times 16$ kotak tambal (*patches*).
    3.  **DeepEncoder (Kotak Garis Putus-Putus Besar):** Ini adalah Jantung inovasinya. Terdiri dari 3 lapis organ:
        *   **SAM (ViTDet 80M):** Organ pertama. Menggunakan mekanisme *local attention*. Mengunyah seluruh gambar detail per detail.
        *   **Conv 16x:** Ini adalah **Katalis Kompresi**. Ia melakukan operasi *Down-sample* (Pemampatan). Dari lautan "tambalan" gambar, organ konvolusi ini menekan dan meringkas data piksel secara drastis (hingga 16 kali lipat lebih ringkas) menjadi sekumpulan *vision tokens* yang jauh lebih sedikit, namun padat makna.
        *   **CLIP ViT 300M:** Organ ketiga. Menerima token hasil pemampatan tadi dan menggunakan *global attention* (mengaitkan makna keseluruhan). Perhatikan keterangan "n/16", artinya jumlah token yang diproses di sini hanyalah 1/16 dari jumlah asli!
    4.  **Embedding Layer:** Menerjemahkan token visual menjadi angka desimal yang dapat dipahami LLM.
    5.  **Decoder (Kotak Oranye):** Adalah otak utama **DeepSeek-3B (MOE-A570M)**. Ia menerima *vision tokens* dan *Prompt* teks dari pengguna, lalu mencerna keduanya, menghasilkan **Output** berwujud teks OCR murni di ujung kanan.
*   **Kesimpulan Strategis:** Desain `DeepEncoder` secara brilian mengisolasi fungsi "membaca resolusi tajam" (di SAM) dengan fungsi "pemahaman global" (di CLIP). Jembatan kompresi 16x (*Conv 16x*) di tengahnya memastikan memori aktivasi (*low activation*) tetap terjaga kecil karena CLIP tidak perlu memproses jutaan token, melainkan hanya 1/16-nya saja.

DeepEncoder menyumbang ukuran sekitar 380 Juta (380M) parameter, sementara Dekoder memakai arsitektur *Mixture-of-Experts* (MoE) [19, 20] berukuran 3 Miliar (3B) di mana hanya sekitar 570M parameter yang teraktivasi pada saat inferensi (sangat efisien!).

### 3.2 DeepEncoder

Untuk merealisasikan kompresi optikal, kita butuh Penyandi Visi (*vision encoder*) super yang memenuhi lima syarat mahaberat:
1. Sanggup memproses resolusi ultra tinggi.
2. Memori aktivasi harus tetap rendah meski resolusi tinggi.
3. Menghasilkan jumlah token visi yang sedikit.
4. Mendukung multi-resolusi.
5. Jumlah parameter tidak boleh membengkak.

Karena tak satupun model *open-source* yang ada di dunia mampu memenuhi kelima ceklis ini (seperti dibuktikan di Bab 2.1), kami menciptakan **DeepEncoder** dari nol.

#### 3.2.1 Arsitektur dari DeepEncoder

DeepEncoder berdiri di atas dua komponen penyarian fitur:
1.  **Ekstraksi Fitur Persepsi Visual:** Didominasi oleh *window attention* (Atensi Jendela/Lokal). Kami meminjam tulang punggung **SAM-base** (dengan ukuran *patch* 16).
2.  **Ekstraksi Fitur Pengetahuan Visual:** Didominasi oleh *dense global attention* (Atensi Global Padat). Kami memakai **CLIP-large**. *Catatan penting: kami membedah dan membuang lapisan 'patch embedding' pertama dari CLIP, karena input yang masuk ke CLIP bukanlah lagi gambar mentah, melainkan token yang sudah diolah dari lapisan sebelumnya*.

Di antara dua raksasa (SAM dan CLIP) ini, kami menyuntikkan teknik dari Vary [36]: kami menyematkan modul konvolusional 2 lapis untuk melakukan **downsampling (kompresi ukuran) sebanyak $16\times$** pada token visi.
Lapisan konvolusi ini menggunakan ukuran kernel 3, *stride* 2, *padding* 1, dan mengekspansi *channel* (jalur warna) dari 256 menjadi 1024.

**Analisis Matematis Beban Memori:**
Misalkan kita mengumpankan gambar berukuran raksasa $1024 \times 1024$ piksel.
DeepEncoder (pada tahap awal SAM) akan memecahnya menjadi token *patch* sebanyak:
$\frac{1024}{16} \times \frac{1024}{16} = 4096 \text{ token patch}$.
Meskipun angkanya 4096, karena paruh pertama ini hanya menggunakan *window attention* dan ukuran parameternya kecil (80M), beban memori aktivasinya tetap aman (*acceptable*). 

Kunci magisnya terjadi sebelum data ini dikirim ke "Otak Global" (CLIP). Ke 4096 token ini dipaksa masuk ke lorong kompresi konvolusional $16\times$. 
Hasilnya? Jumlah token yang keluar menciut drastis menjadi:
$\frac{4096}{16} = \mathbf{256 \text{ token saja!}}$

Jadi, "otak berat" global (CLIP) dan model bahasa sesudahnya (LLM) hanya perlu memikirkan dan mencerna 256 token, yang secara absolut membuat beban memori keseluruhan terkendali (*controllable*).

#### 3.2.2 Dukungan Multi Resolusi (*Multiple resolution support*)

Bayangkan jika gambar input berukuran sangat aneh atau memiliki rasio memanjang. Untuk memecahkan tantangan ini, DeepEncoder dipersenjatai dengan dukungan resolusi ganda. 

### Analisis Akademis: Gambar 4 - Mode Resolusi Input DeepEncoder

Silakan cermati panel diagram pada **Gambar 4**. Ini menjelaskan kelenturan model terhadap dimensi gambar masukan.

*   **Deskripsi Visual:** Menggambarkan empat mode utama cara sistem memperlakukan gambar sebelum dikompresi.
    *   **Kiri (Mode: Tiny || Small):** Untuk gambar kecil (resolusi di bawah 512 atau 640 piksel), gambar tersebut akan di-*Resize* (dipaksa ubah ukuran ditarik/ditekan) agar pas dengan bingkai kotak.
    *   **Tengah (Mode: Base || Large):** Untuk dokumen beresolusi standar (1024 atau 1280 piksel). Untuk menjaga aspek rasio asli agar teks tidak *gepeng*, digunakan teknik *Padding* (menambahkan ruang kosong/batas berwarna di sisi gambar agar menjadi persegi tanpa merusak teks). 
    *   **Kanan (Mode: Gundam || Gundam Master):** Ini adalah senjata rahasia untuk dokumen ultra-panjang (seperti koran PDF utuh yang sangat memanjang vertikal). Dokumen raksasa ini akan dipecah-pecah (*tiled*) menjadi maksimum $n=6$ (hingga $n=9$) kotak-kotak berukuran 640x1024. 
*   **Kesimpulan Akademis:** Diagram ini membuktikan fleksibilitas sistemik. Alih-alih kaku pada satu resolusi, sistem menggunakan interpolasi dimensi dinamis untuk merekonsiliasi dokumen jenis apapun (dari kartu nama *Tiny* hingga *brochure* koran raksasa *Gundam*) tanpa mengorbankan rasio teks.

Berdasarkan penjelasan matematis, pada mode *Base* dan *Large*, gambar tidak di-resize, melainkan di-*padding* demi menjaga aspek rasio (sehingga teks tidak gepeng). Setelah di-padding, tentu ada ruang kosong (area abu-abu pada gambar 4). Token visi di area kosong ini **tidak berguna** dan memboroskan memori. Oleh karena itu, kita harus menghitung token visi *Valid* (yang benar-benar mengandung data gambar, bukan sekadar ruang kosong padding). Formula perhitungannya adalah:

$$ N_{valid} = \lceil N_{actual} \times [1 - ((\max(w, h) - \min(w, h)) / (\max(w, h)))] \rceil \tag{1} $$

**Bedah Anatomi Persamaan (1):**
*   $N_{valid}$: Jumlah target token visi yang *bermanfaat*.
*   $N_{actual}$: Jumlah token visi total secara teori (jika area padding ikut dihitung penuh).
*   $w$ dan $h$: *Width* (Lebar) dan *Height* (Tinggi) dari gambar aslinya (sebelum di-padding).
*   $\max(w, h)$: Fungsi untuk mengambil sisi terpanjang gambar.
*   $\min(w, h)$: Fungsi untuk mengambil sisi terpendek gambar.
*   $\lceil \dots \rceil$: Fungsi *Ceiling*, membulatkan angka desimal ke atas ke integer terdekat.
*   **Logika di balik rumus:** Bagian $[1 - \dots]$ pada dasarnya adalah rasio luas penampang (*aspect ratio penalty*). Ini menghitung persentase dimensi kosong yang dihasilkan dari perbedaan panjang antara sisi terpanjang dan sisi terpendek gambar. Dengan mengalikan $N_{actual}$ dengan rasio penalti ini, sistem secara cerdas akan "menganulir" token yang berada pada area *padding*, membebaskan beban komputasi LLM untuk tidak mencerna kekosongan.

**Mode Gundam (Resolusi Dinamis Lanjutan):**
Mode ini khusus untuk dokumen raksasa. Strateginya adalah pemecahan ubin (*tiling*) mengadopsi InternVL2.0 [8]. Mode Gundam merebus beberapa kotak ubin (berukuran 640x640) *ditambah* satu tampilan global beresolusi utuh (1024x1024). Karena ubin lokal diproses oleh *secondary window attention*, beban aktivasinya tertahan rendah.
Jumlah token visi yang dimuntahkan DeepEncoder pada mode Gundam dirumuskan sebagai: $n \times 100 + 256$, di mana $n$ adalah jumlah kepingan ubin.

### 3.3 Dekoder MoE (*The MoE Decoder*)

Sebagai otak bahasanya (Dekoder), digunakan **DeepSeek3B-MoE** [19, 20]. Pada fase inferensi (menjawab pengguna), model hanya memutar/menyalakan (mengaktifkan) **6 dari 64 pakar terarah (*routed experts*)** ditambah **2 pakar bersama (*shared experts*)**. Ini menyusutkan daya komputasi hingga yang bekerja hanya \~570M parameter aktif (separuh ukuran model LLaMA kecil). Arsitektur MoE raksasa berskala kecil ini sempurna untuk tugas sempit nan padat seperti OCR, karena menyerap kekuatan ekspresi model 3B dengan efisiensi kilat model 500M.

Dekoder ini bekerja "membangkitkan" kembali (*reconstructs*) barisan teks asli dari token visi laten kompresi yang diumpankan DeepEncoder. Secara matematis didefinisikan:

$$ f_{dec} : \mathbb{R}^{N \times d_{latent}} \rightarrow \mathbb{R}^{N \times d_{text}}; \quad \mathbf{\hat{X}} = f_{dec}(\mathbf{Z}) \quad \text{dimana } n \le N \tag{2} $$

**Bedah Anatomi Persamaan (2):**
*   $\mathbf{Z} \in \mathbb{R}^{N \times d_{latent}}$: Ini adalah representasi matriks dari *compressed latent (vision) tokens* hasil keluaran DeepEncoder. $N$ adalah jumlah token, $d_{latent}$ adalah dimensi vektor visi.
*   $\mathbf{\hat{X}} \in \mathbb{R}^{N \times d_{text}}$: Ini adalah matriks target teks (*reconstructed text representation*) yang dibaca oleh model, dengan dimensi panjang $d_{text}$.
*   $f_{dec}$: Fungsi dekoder neural (*non-linear mapping*). Ini merepresentasikan otak LLM itu sendiri, yang dilatih melalui optimalisasi gaya-OCR. 
*   **Signifikansi Akademis:** Persamaan sederhana ini menegaskan sebuah hipotesis sains radikal: LLM yang ukurannya dipadatkan (kompak) *dapat* secara efektif mempelajari sebuah **fungsi pemetaan non-linear (*non-linear mapping*)** untuk mendekode ulang informasi padat di dalam "gambar terkompresi" kembali menjadi "teks berurutan" melalui tahapan prapelatihan yang dimanipulasi dengan ketat.

### 3.4 Mesin Data (*Data Engine*)

Untuk mengajari model kemampuan ajaib ini, kami menyusun data set pelatihan super bervariasi:
1.  **Data OCR 1.0:** Data dokumen teks standar (PDF makalah, dsb) dengan tata letak kompleks.
2.  **Data OCR 2.0:** Tugas rumit seperti gambar *chart* (bagan), rumus kimia kompleks, dan geometri matematika.
3.  **General Vision Data:** Data gambar alam (*natural scenes*) untuk melatih pemahaman visual umum agar mesin ini tak sekadar menjadi mesin tik buta warna.

#### 3.4.1 Data OCR 1.0

Kami menghisap (*collect*) **30 Juta halaman PDF** dari internet, melingkupi hampir 100 bahasa. Dari jumlah ini, 2 Juta halaman bahasa Mandarin dan Inggris dianotasi secara *Fine* (Sangat Presisi) menggunakan model tata letak canggih (seperti PP-DocLayout) dan mesin OCR lain untuk melatih pendeteksian koordinat kotak pembatas (*bounding boxes*). 

### Analisis Akademis: Gambar 5 - Presentasi Data *Ground Truth*

Untuk membayangkan bagaimana wujud "Buku Pelajaran" untuk AI ini, tatap **Gambar 5**.

*   **Deskripsi Visual:** Ini membandingkan Gambar Dokumen Asli (Kiri) dengan Teks Label Sasaran / *Ground Truth* (Kanan).
*   **Analisis Data:** Perhatikan teks di sebelah kanan. Sebelum sebuah kalimat paragraf muncul, terdapat barisan karakter aneh seperti `<|ref|><text>|</ref><|det|>[55, 43, 130, 60]|</det>`.
*   **Kesimpulan:** Sistem pelabelan untuk model *End-to-End* OCR ini mengadopsi format **interleaved layout and text** (gabungan susunan dan teks). Model diajari tidak hanya sekadar melafalkan huruf-huruf di gambar, tetapi wajib melafalkan dan mengurutkan secara tepat "Di manakah titik koordinat ($x_1, y_1, x_2, y_2$) kotak paragraf tersebut berada di dalam gambar asli" (yang dinormalisasi menjadi skala 1000 *bins*). Ini melatih *Spatial Cognitive* (kognisi tata letak spasial) dari AI.

#### 3.4.2 Data OCR 2.0

Data OCR klasik saja tidak cukup. Generasi OCR 2.0 menuntut AI untuk membaca gambar saintifik lalu memprogram ulang gambar tersebut.

### Analisis Akademis: Gambar 6 - *Ground Truth* Bagan dan Geometri

*   **Deskripsi Visual:** Mempertunjukkan dua jenis data canggih. (a) Menampilkan gambar grafik batang berundak. Data *Ground Truth* (jawaban target)-nya adalah kode *syntax* `HTML <table>` murni yang teramat padat. (b) Menampilkan limas geometri 3D. Target teksnya adalah definisi kamus (*dictionary format*) Python yang mengkodekan garis demi garis, titik koordinat endpoint `{"line_type": ... "line_segs": ... }` dari ruang limas tersebut.
*   **Kesimpulan:** Ini membuktikan bahwa kompresi optikal (*optical compression*) dari DeepSeek-OCR didesain untuk tidak sekadar buta mengeja huruf, melainkan mengompresi dan merekonstruksi wujud matematis visual menjadi entitas pemrograman komputer murni (HTML dan Dictionary). Hal ini menjadikan model ini berpotensi membongkar informasi analitik finansial dan matematika pada LLM di masa depan.

---

### 3.5 Alur Pelatihan (*Training Pipelines*)

Pipa pelatihannya sangat ringkas, hanya melintasi dua panggung utama:
1.  **Panggung A: Melatih DeepEncoder secara Mandiri.**
    Menggunakan kerangka kerja *next token prediction* (seperti model Vary [36]). Di sini, DeepEncoder dilatih menggunakan kumpulan model bahasa kecil [15] untuk "pemanasan". Seluruh data OCR 1.0, 2.0, plus 100M data gambar umum (*LAION*) dituang ke kawah candradimuka. Di- *train* 2 epochs dengan *batch size* 1280. *Optimizer* AdamW, laju pembelajaran 5e-5 menggunakan jadwal *cosine annealing*.
2.  **Panggung B: Melatih Keseluruhan DeepSeek-OCR.**
    Di sini seluruh sistem diuji coba massal di platform `HAI-LLM` menggunakan **Pipeline Parallelism (PP)**. Model dipenggal jadi 4 kompartemen:
    *   **PP0:** Diisi oleh unit SAM dan Kompresor. Parameternya **dibekukan (frozen)** tak bisa belajar. Mereka hanya bekerja paksa mengubah piksel ke token.
    *   **PP1:** Diisi unit CLIP. Parameternya **dicairkan (unfrozen)**. CLIP akan terus belajar mengasah pemahaman fiturnya.
    *   **PP2 & PP3:** Diisi oleh Dekoder LLM raksasa (DeepSeek3B-MoE yang dipecah jadi 6 layer per unit).
    *   Menggunakan 20 mesin node (berisi total 160 GPU A100), dengan Data Parallelism (DP) di angka 40 dan global batch 640. Mesin ini berlari secepat kilat: mencerna **90 Miliar token teks per hari** dan **70 Miliar token multimodal per hari**.

---

## 4. Evaluasi (*Evaluation*)

### 4.1 Studi Kompresi Visi-Teks (*Vision-text Compression Study*)

Di awal bab, kita telah membongkar grafik empiris Gambar 1 (a) mengenai rasio kompresi. Namun, angka eksaknya perlu dikuliti dari **Tabel 2**.

### Tabel 2: Evaluasi Batas Kompresi pada Tolok Ukur Fox

| Text Tokens | Vision Tokens = 64 (Tiny Mode) | | Vision Tokens = 100 (Small Mode) | |
| :---: | :---: | :---: | :---: | :---: |
| | **Precision** | **Compression** | **Precision** | **Compression** |
| 600-700 | 96.5% | 10.5$\times$ | 98.5% | 6.7$\times$ |
| 800-900 | 83.8% | 13.2$\times$ | 96.8% | 8.5$\times$ |
| 1000-1100 | 79.3% | 16.5$\times$ | **91.5%** | **10.6$\times$** |
| 1200-1300 | 59.1% | 19.7$\times$ | 87.1% | 12.6$\times$ |

*(Tabel ini menguji batas sadis dari "seberapa jauh kita bisa memeras otak model")*

**Observasi Saintifik Akademis:**
*   Perhatikan kolom *Vision Tokens = 100*. Sebuah gambar dokumen tunggal diubah menjadi "100 butir token". 
*   Ketika gambar tersebut sejatinya memuat sekitar **1000-1100 kata teks asli** (Rasio kompresi meroket ke $\mathbf{10.6 \times}$ lipat), akurasi tebakan (Presisi) model masih kokoh perkasa di angka **91.5%**. 
*   Ini adalah pembuktian *Golden Standard*: **Kompresi $10 \times$ lipat (*lossless context compression*) adalah realitas yang bisa dicapai**. 
*   Ketika ditekan hingga rasio kompresi membengkak di atas $10\times$ (misalnya menuju 1200 kata asli / rasio $19.7\times$ pada Vision Token=64), akurasi memang tumbang ke 59.1%. Ini disebabkan tata letak teks yang memudar kabur (*blurred*) karena terlampau padat dijejalkan ke resolusi mungil $512 \times 512$ di `Tiny Mode`.

### 4.2 Performa Praktis OCR (*OCR Practical Performance*)

Apakah DeepSeek-OCR hanya macan kertas di lab uji coba? Tidak. Kami melepas liarkannya pada ujian **OmniDocBench** (ujian *parsing* dokumen brutal di dunia nyata) dan merangkum skor mautnya di **Tabel 3**.

### Tabel 3: Performa Jarak Edit di OmniDocBench

*(Catatan Dosen: Semakin rendah angka "Edit Distance" (jarak revisi), semakin dewa performanya).*

| Model (Metode) | Avg Tokens | **English (overall)** | **Chinese (overall)** |
| :--- | :---: | :---: | :---: |
| **Model Terpisah (Pipeline):** |
| Mathpix (Komersial) | - | 0.191 | 0.364 |
| MinerU-2.1.1 | - | 0.162 | 0.244 |
| **Model Ujung-ke-Ujung (End-to-End):** |
| Nougat | 2352 | 0.452 | 0.973 |
| Qwen2.5-VL-7B | 3949 | 0.316 | 0.399 |
| GOT-OCR2.0 | 256 | 0.287 | 0.411 |
| MinerU2.0 | 6790 | 0.133 | 0.238 |
| **DeepSeek-OCR (Milik Kita)**|
| **Base Mode** | 256 | 0.137 | 0.240 |
| **Gundam-M Mode** | **1853** | **0.123** | **0.157** |

**Evaluasi Kritis Kinerja Tabel 3:**
1.  Model sekelas **GOT-OCR2.0** yang memakan 256 token, hancur di angka 0.287 (Inggris) dan 0.411 (Mandarin).
2.  DeepSeek-OCR di kelas token yang sama persis (**Base Mode 256 token**) mencetak dominasi gila di angka **0.137** (Inggris) dan **0.240** (Mandarin).
3.  Lebih memalukan lagi bagi pesaing, Model Raksasa **MinerU2.0** membakar dan menelan hingga nyaris 7000 token visual per gambar. Namun, DeepSeek-OCR Mode Gundam-M hanya memerlukan **seperempat tokennya (1853 token)**, sukses mengoyak MinerU2.0 dengan meraih rekor performa tertinggi tak tertandingi **0.123** (Inggris) dan **0.157** (Mandarin). Ini menyahihkan tesis kompresi efisiensi kita.

### 4.3 Studi Kualitatif (*Qualitative Study*)

Bagian ini mendemonstrasikan visual "kekuatan lapangan" (*field capabilities*) dari DeepSeek-OCR.

#### 4.3.1 Parsing Mendalam (*Deep parsing*)

*(Dosen: Mahasiswa sekalian, silakan perhatikan secara analitis pada kumpulan Gambar Kualitatif di makalah asli).*
*   **Gambar 7 (Laporan Keuangan):** Model secara otonom merobek tabel matriks padat di dalam PDF laporan keuangan menjadi format Markdown rapi tanpa satu nilai rasio (angka desimal gila-gilaan) yang terlewat.
*   **Gambar 8 (Buku Anak):** Tak hanya teks. AI sanggup menembus kognisi gambar latar ilustrasi kartun, mendeskripsikan secara padat adegan *"Gadis muda sedang membacakan buku pada anak-anak"*, layaknya fungsi deteksi visual global tingkat dewa.
*   **Gambar 9 (Rumus Kimia):** Ketika dilempar rantai molekul poligon organik rumit berstruktur karbon heksagonal, AI memuntahkan murni sintaks rantai kimia `SMILES` yang utuh sempurna untuk mesin komputasi biologi.
*   **Gambar 10 (Geometri Sumbu):** Model menelanjangi limas trapesium tiga dimensi dengan melacak dan merender ulang setiap ujung vektor titik ($X, Y, Z$) menjadi format terstruktur (`Deep Parsing`), siap diumpan ke perangkat lunak grafik komputer (*Computer-Aided Design*).
*   **Kesimpulan:** Rangkaian pameran unjuk gigi komputasional ini melegitimasi utilitas DeepSeek-OCR tidak sekadar sebagai mesin fotokopi kata, melainkan instrumen ekstraktor data multidimensi yang ganas.

---

## 5. Diskusi (*Discussion*)

Penelitian awal (prototipikal) kita ke dalam batas kelayakan kompresi visi-teks ini membuka horizon filosofi komputasi baru. 
> "Mengapa LLM masa depan harus menghitung satu kata memori teks panjang, jika semua teks itu bisa dijepret dalam satu gambar optik mungil yang dikompres, dan disuntikkan langsung ke otak *attention* LLM?"

Terdapat sebuah prospek masa depan yang brilian. Memori pada otak AI LLM (jendela konteks yang rentan terhadap penyakit kelupaan panjang) dapat dievolusikan mengikuti arsitektur **Memori Amnesia Bertahap** mirip manusia biologis. 

### Analisis Akademik: Gambar 13 - Paradigma Mekanisme Melupakan Memori (*Forgetting Mechanism*)

*   **Deskripsi Visual:** Ini adalah bagan skema waktu (*timeline schematics*) yang membentangkan relasi ingatan AI dari "Baru Saja Terjadi" hingga "Tahun Lalu".
*   **Alur Data:**
    *   Teks yang *Baru Saja Terjadi* dibaca sebagai "Text token" (Teks Murni) konvensional dengan tingkat kejernihan ingatan "*Crystal Clear*". Otak mencernanya dengan utuh 100%.
    *   Teks dari "1 Jam" atau "1 Hari" lalu, direkam ke dalam bentuk optik (gambar) menggunakan resolusi tinggi *Gundam Mode* atau *Large Mode*. Ini mengurangi beban token (misal jadi 800 token) namun resolusi *Very Clear*.
    *   Teks dari "1 Minggu" lalu dikompresi lagi menjadi resolusi menengah *Base Mode* (256 token, resolusi menurun menjadi *Blurry/Kabur*).
    *   Teks "1 Tahun" lalu, secara drastis ditarik ciut ke resolusi terendah *Tiny Mode* (hanya 64 token) dengan ingatan samar-samar *Almost Gone*.
*   **Kesimpulan Akademis Filosopi Komputasi:** Gambar ini adalah cetak biru mutakhir. Kompresi Optik Konteks memungkinkan rekayasa "Mekanisme Melupakan Bertahap" (*Biological progressive information loss / Decay*). Semakin jauh memori itu berlalu di obrolan multi-sesi, semakin dalam ia dikompresi ke ubin resolusi mungil (sedikit token). Ia mengurangi beban kuadratik (*quadratic scaling*) komputasi tanpa pernah melupakan jejak inti secara mutlak. Resolusi spasial diubah menjadi "Dimensi Kelupaan Kognitif". Mahakarya pemodelan memori yang cemerlang.

---

## 6. Kesimpulan (*Conclusion*)

Di dalam laporan teknis ini, tim mendesain **DeepSeek-OCR** dan memvalidasi pendahuluan yang ekstrem: Kompresi Optikal Konteks adalah sebuah kelayakan absolut. Model mampu meretas (*decode*) dan membaca teks yang jumlah tokennya $10 \times$ lipat lebih banyak (lebih gemuk) dari pada jumlah token visi fisik yang menelan dayanya. 

Temuan fundamental ini niscaya akan menjadi penggerak peluncuran arsitektur mutakhir VLMs dan LLMs generasi penerus. Secara praktikal komersial, DeepSeek-OCR merupakan monster manufaktur data (*data production engine*) yang tangguh, siap mendistribusikan lautan korpus pelatih raksasa dari web murni demi mensuplai otak-otak LLM masa depan. 

Eksplorasi di ranah uji coba jarum-di-tumpukan-jerami (*needle-in-a-haystack*) pada tumpukan optik-digital hibrida ini akan menjadi target buruan di riset esok hari. Perspektif "Memaksa Modalitas Visi melayani Pemampatan Memori Abstrak Teks" telah mencabik batasan tradisi dan mendirikan sebuah panggung riset mahaluas dan revolusioner. 

*(Demikian rangkuman perkuliahan teknis dan pedagogis atas makalah DeepSeek-OCR).*
