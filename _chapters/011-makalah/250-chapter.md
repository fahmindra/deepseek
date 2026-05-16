---
slug: ch-11-25
title: "DeepSeek-OCR 2"
layout: chapter
---

## DeepSeek-OCR 2: Aliran Kausal Visual (*Visual Causal Flow*)

Dalam bab ini, kita akan membahas **DeepSeek-OCR 2** untuk menginvestigasi kelayakan dari sebuah *encoder* (penyandi) baru—yaitu **DeepEncoder V2**—yang memiliki kemampuan untuk mengatur ulang secara dinamis token-token visual (*visual tokens*) berdasarkan semantik dari sebuah gambar. 

Sebagai mahasiswa Kecerdasan Buatan, Anda harus memahami bahwa model visi-bahasa konvensional (*conventional vision-language models* / VLMs) selalu memproses token visual dalam urutan pemindaian raster yang kaku (*rigid raster-scan order*), yakni dari kiri-atas ke kanan-bawah, dengan pengodean posisi yang tetap (*fixed positional encoding*) ketika diumpankan ke dalam Model Bahasa Besar (LLMs). Namun, hal ini bertentangan dengan persepsi visual manusia. Penglihatan manusia mengikuti pola pemindaian yang fleksibel namun koheren secara semantik, yang didorong oleh struktur logis bawaan. Khususnya untuk gambar dengan tata letak (*layouts*) yang kompleks, penglihatan manusia menunjukkan pemrosesan sekuensial yang diinformasikan secara kausal (*causally-informed sequential processing*).

Terinspirasi oleh mekanisme kognitif manusia ini, **DeepEncoder V2** dirancang sedemikian rupa untuk melengkapi *encoder* dengan kemampuan penalaran kausal (*causal reasoning capabilities*). Hal ini memungkinkannya untuk mengatur ulang token visual secara cerdas *sebelum* terjadinya interpretasi konten berbasis LLM. 

Karya ini mengeksplorasi sebuah paradigma baru: *apakah pemahaman gambar 2D dapat dicapai secara efektif melalui dua struktur penalaran kausal 1D yang disusun secara kaskade (berjenjang)?* Pendekatan ini menawarkan metode arsitektural baru yang berpotensi mencapai penalaran 2D yang sejati (*genuine 2D reasoning*). Kode sumber dan bobot model (*model weights*) telah disediakan agar dapat diakses oleh publik.

---

### Analisis Akademik Gambar 1: Evolusi Arsitektur DeepEncoder

Sebelum kita melangkah lebih jauh, mari kita bedah **Gambar 1** yang memberikan representasi visual dari inovasi inti dalam makalah ini.

**Deskripsi Visual:**
Gambar 1 mengilustrasikan perbandingan arsitektur antara `DeepEncoder` (versi sebelumnya, di sebelah kiri) dan `DeepEncoder V2` (versi yang diusulkan, di sebelah kanan).

*   **Sisi Kiri (DeepEncoder):**
    *   Terdapat blok `Tokenizer` (80M 16x) di bagian atas yang memproses kotak-kotak token.
    *   Terdapat blok `CLIP` (ViT 300M) di bagian bawah.
    *   Di antara keduanya terdapat lapisan token (kotak-kotak berwarna kuning, cokelat) yang ditandai dengan panah horizontal dua arah `non-causal`. Ini berarti token-token tersebut saling memperhatikan satu sama lain secara setara tanpa ada urutan waktu/sebab-akibat (*bidirectional attention*).

*   **Sisi Kanan (DeepEncoder V2):**
    *   Blok atas tetap `Tokenizer` (80M 16x).
    *   Perubahan drastis terjadi di blok bawah: `CLIP` telah digantikan oleh `LM as Vision Encoder` (Model Bahasa sebagai Penyandi Visi), spesifiknya menggunakan arsitektur `Qwen2 500M`.
    *   Terdapat penambahan sekelompok kotak token baru di sebelah kanan (berwarna hijau, abu-abu), yang diberi label `learnable query (new reading order)` (kueri yang dapat dipelajari - urutan membaca baru).
    *   **Perhatikan arah panah di tengah:**
        *   Untuk token visual asli (kotak kuning/cokelat di kiri), panahnya adalah dua arah `non-causal` (mereka tetap bisa melihat keseluruhan gambar).
        *   Untuk token kueri baru (kotak hijau/abu-abu di kanan), panahnya menunjuk **ke kanan dan memiliki sifat `causal`**.

**Kesimpulan Akademis:**
Gambar ini menjelaskan bahwa komponen CLIP dalam `DeepEncoder` telah diganti dengan arsitektur bergaya LLM. Dengan menyesuaikan maska atensi (*attention mask*), token visual (gambar) akan menggunakan atensi dua arah (*bidirectional attention*), sedangkan kueri yang dapat dipelajari (*learnable queries*) akan mengadopsi **atensi kausal (*causal attention*)**. 

Akibatnya, setiap token kueri tersebut *dapat memperhatikan (attend to) seluruh token visual* DAN *token kueri yang mendahuluinya*. Hal ini memungkinkan terjadinya pengaturan ulang kausal secara progresif (*progressive causal reordering*) terhadap informasi visual. Model tidak lagi dipaksa membaca gambar dari kiri-atas ke kanan-bawah, melainkan membaca sesuai urutan semantik yang "dipelajari" oleh kueri kausal tersebut.

***

## 1. Pendahuluan

Sistem visual manusia sangat mirip dengan *vision encoders* berbasis transformer (Dosovitskiy, 2020; Hu et al., 2024): fiksasi foveal berfungsi layaknya token visual, yang tajam secara lokal namun tetap memiliki kesadaran secara global. Namun, tidak seperti *encoders* (penyandi) yang ada saat ini—yang secara kaku memindai token dari kiri-atas ke kanan-bawah—penglihatan manusia mengikuti aliran yang didorong oleh sebab-akibat (*causally-driven flow*) yang dipandu oleh pemahaman semantik. 

Sebagai analogi, pertimbangkan proses melacak sebuah bentuk spiral—pergerakan mata kita akan mengikuti logika bawaan di mana setiap fiksasi selanjutnya secara kausal bergantung pada titik fiksasi sebelumnya. Secara logis, token visual dalam model AI seharusnya diproses secara selektif, di mana pengurutannya sangat bergantung pada *semantik visual*, bukan sekadar koordinat spasial (letak piksel).

Wawasan ini memotivasi kita untuk mempertimbangkan kembali secara fundamental rancangan arsitektur dari model visi-bahasa (*vision-language models* / VLMs), khususnya pada komponen *encoder*. 

LLM pada dasarnya dilatih pada data berurutan 1D (satu dimensi, seperti rentetan teks), sedangkan gambar adalah struktur 2D. Meratakan (*flattening*) tambalan gambar (*image patches*) secara langsung ke dalam urutan pemindaian raster (*raster-scan order*) yang telah ditentukan sebelumnya akan memperkenalkan **bias induktif yang tidak beralasan (*unwarranted inductive bias*)** karena metode ini mengabaikan hubungan semantik. 

Untuk mengatasi masalah ini, kami menyajikan **DeepSeek-OCR 2** dengan desain *encoder* baru—**DeepEncoder V2**—untuk maju menuju pengkodean visual yang lebih mirip manusia. Mengikuti DeepSeek-OCR (Wei et al., 2025), kami mengadopsi "membaca dokumen" (*document reading*) sebagai arena pengujian eksperimental utama kami. Dokumen menghadirkan tantangan yang sangat kaya, termasuk tata letak (*layout*) pesanan yang kompleks, rumus-rumus yang rumit, dan tabel-tabel. Elemen-elemen terstruktur ini secara inheren membawa **logika visual kausal**, yang menuntut kapabilitas penalaran canggih, menjadikan OCR dokumen sebagai platform ideal untuk memvalidasi pendekatan kami.

Kontribusi utama kami mencakup tiga hal:

1.  **Pertama**, kami menyajikan DeepEncoder V2, yang menampilkan beberapa inovasi kunci: 
    *   Kami mengganti komponen CLIP (Radford et al., 2021) pada DeepEncoder (Wei et al., 2025) dengan arsitektur LLM yang ringkas (Wang et al., 2024), seperti yang diilustrasikan pada Gambar 1, untuk mencapai aliran kausal visual (*visual causal flow*).
    *   Untuk memungkinkan pemrosesan yang diparalelkan, kami memperkenalkan kueri yang dapat dipelajari (*learnable queries*) (Carion et al., 2020), yang diistilahkan sebagai **token aliran kausal (*causal flow tokens*)**, dengan token visual ditambahkan sebelumnya sebagai awalan (*prefix*). Melalui maska atensi yang disesuaikan (*customized attention mask*), token visual mempertahankan bidang reseptif global (*global receptive fields*), sementara token aliran kausal dapat memperoleh kemampuan untuk mengatur ulang token visual.
    *   Kami mempertahankan kardinalitas (jumlah elemen) yang sama antara token kausal dan token visual (termasuk redundansi seperti *padding* dan batas) untuk menyediakan kapasitas yang cukup bagi proses fiksasi-ulang (*re-fixation*).
    *   Hanya token aliran kausal—yaitu paruh kedua dari keluaran *encoder*—yang diumpankan ke dekoder LLM (Li et al., 2025), sehingga memungkinkan pemahaman visual sadar-kausal yang kaskade (*cascade causal-aware visual understanding*).
2.  **Kedua**, dengan memanfaatkan DeepEncoder V2, kami menghadirkan **DeepSeek-OCR 2**, yang mempertahankan rasio kompresi gambar dan efisiensi dekode dari DeepSeek-OCR, sekaligus mencapai peningkatan performa yang substansial. Kami membatasi token visual yang diumpankan ke LLM antara **256 hingga 1120 token**. Batas bawah (256) sesuai dengan tokenisasi DeepSeek-OCR untuk gambar $1024 \times 1024$, sedangkan batas atas (1120) sepadan dengan anggaran token visual maksimum milik Gemini-3 pro (Team, 2023). Desain ini memposisikan DeepSeek-OCR 2 bukan hanya sebagai arsitektur VLM baru untuk eksplorasi penelitian, tetapi juga sebagai alat praktis untuk menghasilkan data pelatihan berkualitas tinggi untuk prapelatihan LLM.
3.  **Ketiga**, kami memberikan validasi awal terkait penggunaan arsitektur model bahasa sebagai *vision encoders*—sebuah jalur yang menjanjikan menuju pengkodean omni-modal yang terpadu (*unified omni-modal encoding*). Kerangka kerja ini memungkinkan ekstraksi fitur dan kompresi token melintasi beragam modalitas (gambar, audio, teks (Liu and Qiu, 2025)) cukup dengan mengonfigurasi kueri *learnable* yang spesifik untuk setiap modalitas. Secara krusial, arsitektur ini secara alami mewarisi optimisasi infrastruktur tingkat lanjut dari komunitas LLM, termasuk arsitektur *Mixture-of-Experts* (MoE), mekanisme atensi yang efisien (Liu et al., 2025), dan lain sebagainya.

Ringkasnya, kami mengusulkan DeepEncoder V2 untuk DeepSeek-OCR 2, menggunakan mekanisme atensi khusus untuk memodelkan aliran visual kausal dari pembacaan dokumen secara efektif. Dibandingkan dengan *baseline* DeepSeek-OCR, DeepSeek-OCR 2 mencapai perolehan performa sebesar **3,73%** pada *OmniDocBench v1.5* (Ouyang et al., 2025) dan menghasilkan kemajuan nyata dalam logika pembacaan visual.

---

## 2. Pekerjaan Terkait (*Related Works*)

### 2.1. Kueri yang Diparalelkan dalam Dekoder (*Parallelized Queries in Decoder*)

DETR (Carion et al., 2020) memelopori integrasi arsitektur transformer ke dalam deteksi objek, secara fundamental melepaskan diri dari paradigma deteksi tradisional (Redmon and Farhadi, 2017; Ren et al., 2015). Untuk mengatasi keterbatasan efisiensi dari dekode serial (*serial decoding*) dalam blok transformer, DETR memperkenalkan kueri *learnable* yang diparalelkan yang telah diatur sebelumnya—yaitu seperangkat 100 kueri objek yang menyandikan priors (asumsi awal) objek seperti bentuk dan posisi melalui proses pelatihan. 

Kueri-kueri ini berinteraksi dengan peta fitur (*feature maps*) (He et al., 2016) melalui mekanisme **atensi-silang (*cross-attention*)**, sembari secara bersamaan terlibat dalam pertukaran informasi dua arah (*bidirectional information exchange*) di antara mereka sendiri melalui **atensi-diri (*self-attention*)**. DETR membentuk sebuah paradigma fondasional yang memungkinkan transformer untuk menangani token-token yang diparalelkan. Sejak saat itu, desain "kueri objek" telah menjadi komponen arsitektural standar *de facto* dalam metode deteksi berbasis transformer berikutnya (Radford et al., 2021; Zhu et al., 2020).

### Analisis Akademik Gambar 2: Visi Komputer dengan Kueri yang Diparalelkan

*   **Deskripsi Visual:** Gambar 2 menyandingkan dua arsitektur *computer vision* (visi komputer) populer yang memanfaatkan kueri paralel.
    *   **Kiri (DETR):** Alur data bermula dari gambar (Image) $\rightarrow$ fitur diekstraksi oleh `ResNet` $\rightarrow$ masuk ke modul `Decoder` dengan atensi `non-causal` menggunakan 100 kueri objek $\rightarrow$ hasil diteruskan ke `Detect head` via *cross-attention*.
    *   **Kanan (BLIP2):** Alur data bermula dari gambar $\rightarrow$ diekstrak oleh `CLIP` menjadi *vision token* $\rightarrow$ masuk ke modul `Q-former` dengan atensi `non-causal` menggunakan 32 kueri token $\rightarrow$ hasil dikirim ke `LLM` via *cross-attention*.
*   **Kesimpulan Akademis:** Gambar ini mendemonstrasikan bahwa konsep menggunakan sejumlah kecil kueri (seperti 32 atau 100) untuk "meringkas" informasi dari *feature map* gambar yang sangat besar bukanlah hal baru. Baik pada DETR (untuk mendeteksi objek) maupun pada BLIP2 (sebagai jembatan ke LLM), arsitektur ini mempekerjakan *self-attention* yang bersifat dua arah (*bidirectional*) di antara kueri-kuerinya. Makalah ini nantinya akan mengubah sifat *non-causal* ini menjadi *causal*.

### 2.2. Kueri yang Diparalelkan dalam Proyektor (*Parallelized Queries in Projector*)

Dalam beberapa tahun terakhir, model visi-bahasa (Chen et al., 2024b; Cui et al., 2025b; Li et al., 2023; Wei et al., 2024) telah berkembang sangat pesat, dengan arsitektur yang berkonvergensi (mengarah) ke paradigma `encoder-projector-LLM`. 

Proyektor (*projector*) berfungsi untuk menyelaraskan token-token visual dengan ruang penyematan (*embedding space*) milik LLM. Proyektor bertindak sebagai jembatan kritis yang memampukan LLM untuk memahami konten visual. **Q-former**, yang diperkenalkan pada arsitektur BLIP-2 (Li et al., 2023), merupakan contoh desain proyektor yang sangat efektif yang menggunakan kueri *learnable* untuk kompresi token visual. 

Mengadopsi arsitektur mirip BERT (Devlin et al., 2019) dan mengambil inspirasi dari kueri objek milik DETR (Carion et al., 2020), Q-former memanfaatkan **32 kueri *learnable*** yang berinteraksi dengan ratusan token visual CLIP (Radford et al., 2021) melalui atensi-silang. Representasi kueri yang telah terkompresi ini kemudian diumpankan ke dalam LLM, mencapai pemetaan yang efektif dari ruang visual ke ruang bahasa. Kesuksesan Q-former mendemonstrasikan bahwa kueri *learnable* yang diparalelkan efektif tidak hanya untuk dekode fitur dalam tugas deteksi, tetapi juga untuk kompresi token dalam penyelarasan multimodal (*multimodal alignment*).

### 2.3. Inisialisasi Multimodal Berbasis LLM (*LLM-based Multimodal Initialization*)

LLM yang dilatih menggunakan data internet berskala besar telah terbukti efektif sebagai titik awal (inisialisasi) bagi model-model multimodal. Pang et al. (Pang et al., 2023) mendemonstrasikan bahwa membekukan (*frozen*) lapisan-lapisan transformer LLM dapat meningkatkan tugas-tugas diskriminatif visual. Lebih dari itu, model-model tanpa *encoder* (*encoder-free*) atau yang memiliki *encoder* kelas bulu (*lightweight-encoder*) seperti Fuyu (Adept, 2023) dan Chameleon (Team, 2024) di ranah visi, serta VALL-E (Wang et al., 2023) di ranah percakapan/suara, lebih lanjut memvalidasi potensi pemanfaatan bobot prapelatihan LLM untuk inisialisasi multimodal.

---

## 3. Metodologi (*Methodology*)

### 3.1. Arsitektur (*Architecture*)

Mari kita bedah arsitektur makro dari sistem ini melalui **Gambar 3**.

### Analisis Akademis Gambar 3: Arsitektur DeepSeek-OCR 2

*   **Deskripsi Visual:** Gambar 3 memvisualisasikan perjalanan aliran data (*data pipeline*) dari sebuah dokumen gambar mentah hingga menjadi *output* teks OCR.
*   **Aliran Data:**
    1.  **Input:** Sebuah dokumen visual (gambar) dimasukkan.
    2.  **Pemecahan Tambalan (*Patching*):** Gambar dipecah menjadi matriks kotak-kotak berukuran $n \times 16 \times 16$.
    3.  **DeepEncoder V2:** Ini adalah jantung dari pemrosesan visual.
        *   Data masuk ke `Tokenizer` visual yang dibentuk oleh model `SAM (ViTDet 80M)` dengan atensi lokal.
        *   Hasilnya dipadatkan oleh blok `Conv 16x`, menghasilkan kompresi jumlah token (`hidden dim 896`, `low activation`). Token visual ini jumlahnya $n/16$.
        *   Token visual tersebut masuk ke **`LM as Vision Encoder (Qwen2 500M)`**. Di sinilah letak revolusi makalah ini. Alih-alih memakai CLIP, makalah ini menyuapkan token visual murni ke sebuah LLM kecil.
        *   Perhatikan **tiga blok persegi**: kotak kuning/krem mewakili *vision tokens* yang memakai maska atensi dua arah (`non-causal`). Kotak merah/ungu di belakangnya adalah token kueri baru yang dipaksa memakai maska atensi maju-searah (`causal`). Terdapat keterangan `cascade causal`.
    4.  **Decoder:** Hanya token kueri kausal (kotak abu-abu) yang diteruskan keluar dari *Encoder*. Token ini masuk ke *Decoder* raksasa `DeepSeek-3B (MOE-A570M)` bersamaan dengan `Prompt` dari pengguna.
    5.  **Output:** *Decoder* memuntahkan teks hasil OCR.

Sebagaimana diperlihatkan pada Gambar 3, DeepSeek-OCR 2 menikmati arsitektur VLM ujung-ke-ujung terpadu yang terdiri dari sebuah *encoder* dan sebuah *decoder*. 
*   *Encoder* bertugas mendiskretisasi gambar menjadi token-token visual.
*   *Decoder* menghasilkan teks keluaran yang dikondisikan pada token visual dan *prompt* teks tersebut. 

Perbedaan kuncinya terletak pada *encoder*: kami meningkatkan (memutakhirkan) DeepEncoder menjadi **DeepEncoder V2**. Versi ini mempertahankan semua kemampuan pendahulunya sembari memperkenalkan **penalaran kausal** melalui rancangan arsitektur yang baru. Kami akan mengelaborasi detail DeepSeek-OCR 2 pada bagian-bagian berikut.

### 3.2. DeepEncoder V2

*Encoder* standar (vanilla) berfungsi sebagai komponen penting yang mengekstrak dan mengompresi fitur gambar melalui mekanisme atensi, di mana setiap token memperhatikan token lainnya (dua arah), sehingga mencapai bidang reseptif gambar yang utuh yang dianalogikan sebagai penglihatan foveal dan periferal manusia. 

Namun, meratakan (*flattening*) tambalan gambar 2D ke dalam urutan 1D memaksakan bias pengurutan yang sangat kaku melalui pengodean posisi berorientasi teks (contohnya, metode RoPE (Su et al., 2021)). Hal ini sangat **berlawanan** dengan pola pembacaan visual alami, terutama ketika menjumpai tata letak non-linear dalam teks optik, formulir, dan tabel.

#### 3.2.1. Tokenizer Visi (*Vision tokenizer*)

Komponen pertama dari DeepEncoder V2 adalah *tokenizer* visi. Mengikuti DeepEncoder, kami menggunakan arsitektur yang menggabungkan model SAM-base berparameter 80M (Kirillov et al., 2023) bersama dengan dua lapisan konvolusional (Wei et al., 2024). Dimensi keluaran dari lapisan konvolusional akhir dikurangi dari 1024 (pada DeepEncoder V1) menjadi 896 agar sejajar (*align*) dengan saluran (*pipeline*) selanjutnya. 

Perlu dicatat bahwa *tokenizer* berbasis kompresi ini sifatnya **tidak wajib** dan dapat diganti dengan penyematan tambalan sederhana (*simple patch embedding*). Kami mempertahankannya karena ia berhasil mencapai kompresi token sebesar **16$\times$ lipat** (Chen et al., 2024a; Liu et al., 2025; Wei et al., 2024, 2025) melalui atensi jendela (*window attention*) dengan parameter yang minimal. Hal ini secara signifikan memangkas biaya komputasi dan memori aktivasi untuk modul atensi global di tahap selanjutnya. Lagipula, jumlah parameternya ($\sim 80\text{M}$) sebanding dengan parameter 100M yang lazim digunakan untuk penyematan input teks (*text input embeddings*) di dalam LLM.

#### 3.2.2. Model Bahasa sebagai Penyandi Visi (*Language model as vision encoder*)

Pada DeepEncoder V1, sebuah model CLIP ViT diletakkan setelah *vision tokenizer* untuk mengompresi pengetahuan visual. **DeepEncoder V2 mendesain ulang komponen ini** menjadi arsitektur bergaya LLM yang dilengkapi dengan mekanisme atensi aliran-ganda (*dual-stream attention mechanism*). 

1.  **Aliran Token Visual:** Token-token visual memanfaatkan atensi dua arah (*bidirectional attention*) guna mempertahankan kapabilitas pemodelan global seperti yang dimiliki CLIP.
2.  **Aliran Kueri Aliran Kausal (*Causal Flow Queries*):** Kueri-kueri baru yang kami perkenalkan ini menggunakan atensi kausal. Kueri yang dapat dipelajari (*learnable queries*) ini ditambahkan (disematkan) tepat *setelah* token visual sebagai sebuah sufiks. Di dalam blok ini, setiap kueri dapat memperhatikan (*attend to*) semua token visual DAN seluruh kueri yang *mendahuluinya*.

Dengan mempertahankan kardinalitas (jumlah token) yang sama antara kueri dan token visual, desain ini secara elegan memaksakan **pengurutan semantik (*semantic ordering*)** dan distilasi atas fitur visual tanpa mengubah jumlah total token. Pada tahap akhir, **hanya keluaran dari kueri kausal** yang diumpankan ke dekoder LLM.

Secara teknis, kami menginstansiasi (membangun) arsitektur ini menggunakan model **Qwen2-0.5B** (Wang et al., 2024), yang mana jumlah parameternya (500M) cukup sebanding dengan model CLIP ViT (300M) tanpa harus mendatangkan biaya *overhead* komputasi yang berlebihan. 

Arsitektur yang mengandalkan "hanya dekoder" (*decoder-only*) dengan teknik konkatenasi-awalan (*prefix-concatenation*) pada token visual ini terbukti sangat krusial: eksperimen tambahan kami yang menggunakan atensi-silang dalam struktur *encoder-decoder* bergaya mBART (Liu et al., 2020) ternyata **gagal mencapai konvergensi** (tidak bisa belajar dengan benar). Kami berhipotesis bahwa kegagalan tersebut berakar dari kurangnya interaksi antar-token visual ketika mereka diisolasi di dalam sebuah *encoder* yang terpisah. Sebaliknya, desain *prefix* (awalan) menjaga agar token-token visual tetap "hidup" dan aktif di seluruh lapisan layer, membina pertukaran informasi visual yang efektif dengan kueri kausal.

Secara nyata, arsitektur ini mendirikan pondasi penalaran kausal kaskade dua tahap (*two-stage cascade causal reasoning*): 
1.  *Encoder* mengatur ulang secara semantik token visual melalui kueri *learnable*.
2.  *Decoder* LLM melakukan penalaran autoregresif terhadap urutan token yang telah tertata tersebut. 

Berbeda dengan *encoders* standar yang secara kaku memaksakan pengurutan spasial statis lewat *positional encodings*, kueri kita yang diurutkan secara kausal ini beradaptasi secara cair terhadap keluwesan semantik visual, sekaligus secara natural menyelaraskan diri dengan pola atensi satu arah (*unidirectional*) milik LLM. Desain terobosan ini diyakini mampu menjembatani jurang pemisah antara struktur spasial 2D dan pemodelan bahasa kausal 1D.

#### 3.2.3. Kueri Aliran Kausal (*Causal flow query*)

Sebagaimana telah disinggung, jumlah token kueri kausal ini disamakan dengan jumlah token visual, yang secara matematis dihitung sebagai $\frac{W \times H}{16^2 \times 16}$, di mana $W$ dan $H$ masing-masing menunjukkan lebar dan tinggi dari gambar masukan ke *encoder*. 

Untuk menghindari kerumitan harus memelihara banyak set kueri untuk berbagai resolusi gambar yang berbeda, kami mengadopsi strategi multi-potongan (*multi-crop strategy*) yang mematok konfigurasi kueri tetap (*fixed query configurations*) pada resolusi tertentu yang sudah didefinisikan sebelumnya.

### Analisis Akademis Gambar 4: Penghitungan Token dalam Strategi Multi-Crop

Mari kita amati **Gambar 4** untuk memahami aritmetika di balik jumlah token.

*   **Deskripsi Visual:** Gambar ini menunjukkan bagaimana sebuah dokumen (halaman kertas) diproses jika ukurannya sangat besar atau kecil. Terdapat dokumen asli berukuran $W:1024 \times H:1024$. 
*   **Analisis Matematis Multi-Crop:**
    *   Sistem memecah gambar resolusi sangat tinggi menjadi **Local View** (Pandangan Lokal/potongan detail) dan **Global View** (Pandangan Global dari keseluruhan halaman namun di- *downscale*).
    *   *Global View* (Kanan): Sebuah gambar ukuran $1024 \times 1024$ akan dicerna oleh DeepEncoder V2 menjadi sejumlah token $1 \times N4 \times D$. Pada bagan tertera bahwa jumlah token dari Global View ini adalah **256 token** (diperoleh dari $\frac{1024 \times 1024}{16 \times 16 \times 16}$ dengan asumsi kernel kompresi). Ini disebut $Q2=256$.
    *   *Local View* (Kiri): Jika gambar besar itu dipecah menjadi kotak-kotak detail berukuran $768 \times 768$ (misal ada 6 kotak), maka tiap kotak akan memproduksi 144 token ($Q1=144$). Jika ada 6 kotak, maka $6 \times 144 = 864$ token.
    *   *Total:* Token Local View (864) + Token Global View (256) = **1120 token total**.
*   **Kesimpulan:** Sistem ini secara eksak membatasi memori token yang dikeluarkan. Gambar sekecil apa pun akan mengeluarkan minimal 256 token. Dan se-raksasa apa pun gambar (hingga 6 potongan lokal), sistem menguncinya di batas maksimum 1120 token. Ini memastikan arsitektur LLM di belakangnya tidak akan mati lemas kehabisan memori VRAM GPU.

Secara spesifik, pandangan global menggunakan resolusi $1024 \times 1024$, yang berkorespondensi dengan 256 penyematan kueri yang dinotasikan sebagai $\text{query}_{\text{global}}$. Potongan-potongan lokal (*local crops*) menggunakan resolusi $768 \times 768$, dengan jumlah potongan $k$ berkisar dari 0 hingga 6 (tidak ada pemotongan yang diterapkan jika kedua dimensi gambar lebih kecil dari 768). Semua pandangan lokal ini berbagi serangkaian 144 penyematan kueri terpadu yang sama, yang dinotasikan sebagai $\text{query}_{\text{local}}$. 

Akibatnya, jumlah total token visual yang diatur ulang (*reordered visual tokens*) yang diumpankan ke dalam LLM adalah $k \times 144 + 256$, dalam rentang batas $[256, 1120]$. Angka batas atas (1120) ini secara sengaja dirancang agar lebih rendah dari batas 1156 token milik DeepSeek-OCR V1 (mode Gundam) dan secara ekuivalen sepadan dengan batas anggaran token visual maksimum milik Gemini-3-Pro.

#### 3.2.4. Maska Atensi (*Attention mask*)

Untuk mengilustrasikan mekanisme atensi dari DeepEncoder V2 dengan lebih jernih, kita harus membedah matriks maska atensi di **Gambar 5**.

### Analisis Akademis Gambar 5: Arsitektur Maska Atensi DeepEncoder V2

*   **Deskripsi Visual:** Gambar ini memvisualisasikan matriks bujur sangkar yang merepresentasikan siapa-boleh-melihat-siapa dalam proses kalkulasi *Attention*. Kotak matriks ini dibagi menjadi warna hijau, merah muda, putih.
*   **Sumbu Kolom (Bawah):** Terbagi dua wilayah. Di kiri adalah `vision token` (ViT). Di kanan adalah `causal query` (LM).
*   **Sumbu Baris (Kanan, 0-5):** Menunjukkan urutan eksekusi/perhitungan per baris.
*   **Membaca Maska (Mask):**
    *   Kotak Hijau (1): Boleh diakses / diperhatikan (*can attend*).
    *   Kotak Merah (0): Diblokir / Dilarang akses (*cannot attend*).
    *   Bulatan Putih: *Self-attend* (melihat dirinya sendiri).
    *   **Blok Kiri (no-causal):** Saat memproses baris 0,1,2 (Token Visi), mereka semua saling melihat satu sama lain secara penuh. Matriksnya hijau utuh tanpa pemblokiran. Ini disebut *Bidirectional* (sama seperti arsitektur ViT/BERT standar).
    *   **Blok Kanan (causal):** Saat memproses baris 3,4,5 (Kausal Kueri), terlihat sebuah pola segitiga (*triangular mask*). Baris 3 hanya boleh melihat kolom 3. Baris 4 boleh melihat kolom 3 dan 4. Baris 5 boleh melihat kolom 3, 4, dan 5. Pola segitiga merah di pojok kanan atas menunjukkan daerah terlarang.
    *   **Penampang Lebar:** Hebatnya, baris kueri (3,4,5) ini **semuanya** kotak hijau jika melihat ke wilayah kiri (*vision token*).
*   **Kesimpulan Akademis:** Maska atensi ini diarsiteki dengan sangat licik. Region kiri (token gambar) berfungsi secara spasial (melihat keseluruhan foto sekaligus tanpa urutan). Region kanan (token kueri) berfungsi secara sekuensial temporal (menebak satu per satu layaknya LLM membaca kalimat). Dan yang terpenting, setiap token kueri yang sedang menebak diizinkan menengok ke SELURUH token gambar di sebelah kirinya untuk mencari "contekan" informasi.

Matriks maska (M) ini secara matematis diformulasikan sebagai berikut:

$$ M = \begin{bmatrix} \mathbf{1}_{m \times m} & \mathbf{0}_{m \times n} \\ \mathbf{1}_{n \times m} & \text{LowerTri}(n) \end{bmatrix}, \quad \text{di mana } n = m \tag{1} $$

**Bedah Anatomi Persamaan (1):**
*   $M$: Matriks Maska Atensi.
*   $n$: Jumlah token kueri kausal.
*   $m$: Jumlah token visual dasar (vanilla).
*   $\mathbf{1}_{m \times m}$: Kuadran kiri-atas. Matriks penuh berisi angka 1. Token visi saling melihat satu sama lain.
*   $\mathbf{0}_{m \times n}$: Kuadran kanan-atas. Matriks penuh berisi angka 0. Token visi **tidak boleh** melihat kueri (karena kueri belum terjadi di masa lalu mereka).
*   $\mathbf{1}_{n \times m}$: Kuadran kiri-bawah. Kueri diizinkan melihat semua token visi.
*   $\text{LowerTri}(n)$: Kuadran kanan-bawah. Matriks Segitiga Bawah (*Lower Triangular*). Angka 1 ada di diagonal dan di bawahnya. Angka 0 ada di atas diagonal. Ini adalah kunci dari sifat "kausalitas" (token masa depan dilarang mengintip token yang lebih jauh di masa depan).

### 3.3. Dekoder DeepSeek-MoE (*DeepSeek-MoE Decoder*)

Karena penelitian DeepSeek-OCR 2 difokuskan secara masif pada peningkatan *encoder*, kami sama sekali tidak memutakhirkan (*upgrade*) komponen *decoder*. Mengikuti prinsip desain isolasi ini, kami mempertahankan dekoder dari model DeepSeek-OCR generasi pertama, yakni struktur **MoE berparameter 3B** dengan hanya sekitar **500M parameter aktif**. 

Keputusan menggunakan model 3B DeepSeekMoE (Liu et al., 2024; Zhu et al., 2024) ini sangat taktis dan cocok untuk penelitian VLM yang *domain-centric* (terpusat pada domain sempit seperti OCR). Alasannya adalah ia mewarisi kapasitas daya ekspresi (*expressive capability*) yang tinggi dari model raksasa 3B, sembari menikmati efisiensi inferensi yang sangat ringan layaknya model mungil 500M.

Fase komputasi propagasi maju (*forward pass*) utama dari DeepSeek-OCR 2 dapat dirumuskan secara holistik sebagai:

$$ \mathbf{O} = \mathcal{D}\left( \pi_Q \left( \mathcal{T}^L( \mathcal{E}(\mathbf{I}) \oplus \mathbf{Q}_0; \mathbf{M} ) \right) \right) \tag{2} $$

**Bedah Anatomi Persamaan (2):**
*   $\mathbf{I} \in \mathbb{R}^{H \times W \times 3}$: Adalah matriks input gambar asli (dengan tinggi $H$, lebar $W$, dan 3 saluran warna RGB).
*   $\mathcal{E}$: Adalah operator *vision tokenizer* yang memetakan gambar menjadi $m$ buah token visual $\mathbf{V} \in \mathbb{R}^{m \times d}$.
*   $\mathbf{Q}_0 \in \mathbb{R}^{n \times d}$: Adalah vektor penyematan awal (*initial embeddings*) untuk kueri kausal yang dapat dipelajari (*learnable*).
*   $\oplus$: Operator Konkatenasi (penggabungan urutan dari kiri ke kanan).
*   $\mathcal{T}^L$: Merepresentasikan tumpukan blok Transformer *Encoder* sebanyak $L$ lapisan, yang telah dikonfigurasi menggunakan operasi atensi bermaska (*masked attention*).
*   $\mathbf{M} \in \{0, 1\}^{2n \times 2n}$: Adalah matriks maska atensi blok kausal yang persis didefinisikan pada Persamaan 1 di atas.
*   $\pi_Q$: Adalah operator proyektor/pemotong yang tugasnya hanya mengekstrak $n$ token paling akhir dari hasil Transformer (secara matematis, jika outputnya $\mathbf{Z} = \mathbf{X}_{m+1:m+n}$, ia hanya mengambil $\mathbf{Z}$). *Ingat, hanya token kueri yang dilempar ke LLM, token visinya dibuang*.
*   $\mathcal{D}$: Adalah bahasa *Decoder* (LLM DeepSeek-MoE).
*   $\mathbf{O} \in \mathbb{R}^{n \times |\mathcal{V}|}$: Adalah probabilitas keluaran final (Logit) untuk memprediksi kata (teks) berdasarkan kamus kosakata LLM ($|\mathcal{V}|$).

---

## 4. Pengaturan Eksperimental (*Experimental Settings*)

### 4.1. Mesin Data (*Data Engine*)

DeepSeek-OCR 2 mengunyah sumber data yang ekuivalen (sama persis) dengan DeepSeek-OCR generasi 1. Data ini meliputi:
*   Data **OCR 1.0** (teks dokumen standar)
*   Data **OCR 2.0** (bagan, rumus kimia, geometri matematika) (Chen et al., 2024; Ouyang et al., 2025; Wei et al., 2024)
*   Data visi umum (*general vision data*) (Wei et al., 2025). 

Di mana, 80% dari total adonan pakan (*training mixture*) didominasi murni oleh data OCR.

Namun, kami menginjeksi **dua modifikasi mikro**:
1.  **Strategi pencuplikan data (*sampling strategy*) yang lebih berimbang** untuk data OCR 1.0. Kami mempartisi (membagi) lembaran dokumen berdasarkan tipe kontennya (Teks murni, Formula matematika, Tabel) dengan rasio paksaan **3:1:1**.
2.  **Penyempurnaan label (*label refinement*)** untuk deteksi tata letak (*layout detection*). Kami menggabungkan kategori-kategori semantik yang bersinonim (mirip), seperti memaksa menyatukan label "figure caption" (keterangan gambar) dan "figure title" (judul gambar) menjadi satu kategori yang sama. 

Mengingat perbedaan yang teramat minim ini, kami menetapkan dengan penuh keyakinan bahwa model DeepSeek-OCR generasi pertama adalah batas dasar (*valid baseline*) yang sah dan adil untuk uji perbandingan kompetitif ini.

### 4.2. Pipa Pelatihan (*Training Pipelines*)

Kami menggembleng DeepSeek-OCR 2 melintasi tiga panggung kurikulum berjenjang:
1.  **Prapelatihan Encoder (*Encoder pretraining*)**
2.  **Peningkatan Kueri (*Query enhancement*)**
3.  **Spesialisasi Decoder (*Decoder specialization*)**

*Rasionalitas Kurikulum:*
*   **Tahap 1** menugaskan *vision tokenizer* dan *encoder* bergaya-LLM agar menguasai kapabilitas fundamental alam bawah sadar: mengekstraksi fitur piksel, memampatkan token (kompresi), dan belajar mengatur tata letak pengurutan token.
*   **Tahap 2** didesain untuk memperkokoh otot "kemampuan pengaturan ulang token" (*token reordering capability*) pada sang *encoder*, sembari di saat yang sama melatih kompresi pengetahuan visual secara masif.
*   **Tahap 3** akan membekukan mati (men- *freeze*) seluruh sirkuit parameter dari *encoder*, dan hanya menghujani siksaan optimisasi eksklusif kepada otak LLM *decoder*. Tahap ketiga inilah yang memampukan mesin untuk mengalirkan data *throughput* yang jauh lebih tinggi (karena komputasi dipangkas) di bawah batas anggaran FLOPs yang sama.

*(Catatan Dosen: Sub-bab 4.2.1 hingga 4.2.3 di dalam naskah merinci konfigurasi hyperparameter kaku seperti nilai Learning Rate, Optimizer AdamW, Batch Size, dan Topologi Pipeline Parallelism yang digelontorkan melintasi 160 unit GPU A100. Angka-angka ini spesifik untuk praktisi rekayasa / engineers yang ingin mereplikasi hasil riset ini dari nol).*

---

## 5. Evaluasi (*Evaluation*)

### 5.1. Hasil Utama (*Main Results*)

Kami menyandarkan pengujian kami kepada mahkamah ujian **OmniDocBench v1.5** (Ouyang et al., 2025) sebagai tolok ukur primer. Gelanggang pengujian ini berisi 1.355 lembar dokumen menjebak yang direntangkan melintasi 9 kategori utama (berisi campuran majalah pelangi, makalah saintifik akademis ruwet, laporan riset, dll), baik dalam aksara Mandarin maupun alfabet Inggris. 

*Kenapa OmniDocBench?* Keanekaragaman rupa sampel tes dan kriteria evaluasi yang sangat tahan banting dari *OmniDocBench* menyediakan sebuah kerangka kerja penghakiman yang sangat telak untuk memvalidasi kinerja DeepSeek-OCR 2.

**Tabel Hasil Akademik (Berdasarkan Narasi Teks):**

1.  **Kemenangan Efisiensi Resolusi:** DeepSeek-OCR 2 mencetak rekor angka kinerja maut di batas **91.09%**. Yang gila adalah, ia meraih rekor ini dengan menggunakan batas atas token visual (*V-token max*) yang **Paling Mungil di Antara Semuanya**.
2.  **Pertarungan Keturunan Langsung (V1 vs V2):** Ketika disabung secara brutal di atas ring melawan ayah kandungnya sendiri (DeepSeek-OCR V1/ *baseline*), arsitektur V2 mencetak kemajuan absolut sebesar **3,73%**. Perlu dicatat, mereka berdua disuapi pakan data *training* yang secara kualitatif sama persis. Hal ini membenarkan dan melegitimasi bahwa perombakan arsitektur *encoder* kausal yang kami temukan adalah kunci dari kecerdasan ini, bukan akibat kelicikan perbanyakan data.
3.  **Kemenangan Logika Pembacaan:** Jika melihat indikator "Jarak Edit (*Edit Distance* / ED)" secara spesifik untuk kategori "Urutan Membaca" (*Reading order* / R-order), model V2 ini menciutkan jarak eror secara agresif dari 0.085 ke 0.057. Artinya, DeepEncoder V2 dengan nyawa sadar kausalnya secara efektif dan telak terbukti mampu **menyeleksi dan merakit kepingan-kepingan token visual awal berdasarkan gravitasi informasi dari gambar itu sendiri**, bukan lagi dipaksa menelan piksel dari arah kiri-atas ke kanan-bawah secara pasif seperti robot bodoh.

### 5.2. Ruang untuk Peningkatan (*Improvement Headroom*)

Para peneliti yang baik tidak hanya pamer kemenangan, tetapi juga menelanjangi kelemahan modelnya.

Kami melakukan pembedahan forensik (*detailed performance comparison*) antara DeepSeek-OCR V1 dan V2 melintasi 9 spesies dokumen secara mikroskopik. Kesimpulannya? DeepSeek-OCR 2 masih menderita beberapa patologi (titik lemah) yang sangat nyata.
*   **Insiden Koran (*Newspapers*):** Meskipun V2 membantai V1 di nyaris seluruh medan (buku, makalah, dll), pada kategori lembaran surat kabar (*newspapers*), V2 loyo dan mencetak angka eror Jarak Edit di atas 0.13 (> 0.13 ED). 

**Diagnosis Kausal Kenapa Model Bisa Gagal:**
1.  **Batas Atas Token Visual yang Terlampau Kerdil:** Surat kabar koran cetak biasanya dipenuhi dengan lautan fonta mikroskopik (*text-super-rich*). Batas kuota atas token visi kita (1120 token) memblokir model untuk mencerna informasi ini dengan jelas. (Terapi penyembuhan di masa depan: Cukup naikkan batas *local crops*).
2.  **Kelaparan Data (*Insufficient Data*):** Kurikulum data latihan kita hanya menyelipkan sekitar 250 ribu sampel koran. Porsi gizi semungil ini mengakibatkan jaringan saraf DeepEncoder V2 menderita *under-fitting* (kurang diajari) spesifik untuk kategori koran. 

### 5.3. Kesiapan Pengerahan Praktis Industri (*Practical Readiness*)

Pertanyaan akhir dari investor korporat: *"Apakah model ini sekadar piala di laboratorium kampus, atau mesin uang yang siap dilepas di pasar industri?"*

DeepSeek-OCR pada hakikatnya bertugas mengabdi di bawah dua medan perang level pabrik (*production use cases*):
1.  Menjadi "mata dewa" untuk layanan web OCR instan yang membaca gambar bagi pengguna DeepSeek-LLMs.
2.  Menjadi *mesin penderes (pipeline)* yang diperah tenaganya untuk mengunyah dan mem- *parsing* jutaan PDF secara massal di latar belakang, demi meracik pakan data *pretraining* AI generasi esok hari.

Karena di lantai produksi alam liar kita tidak memiliki *Ground Truth* (kunci jawaban) untuk menghitung *Edit Distance*, kami mengandalkan **Tingkat Repetisi (*Repetition rate*)** sebagai metrik pengawasan patologi mutu (*quality metric*) primer. Jika mesin AI mulai berhalusinasi (ngelantur) dan mengulang-ulang kalimat yang sama ("ini adalah... ini adalah... ini adalah..."), itu artinya model tersebut rusak.

**Bukti Kesiapan:**
DeepSeek-OCR 2 mendemonstrasikan kelayakan produksi (*practical readiness*) yang luar biasa. 
*   Untuk data gambar dari interaksi log pengguna *online*, tingkat repetisi anjlok menyelam dari 6.25% turun menjadi **4.17%**. 
*   Untuk ladang produksi pengolahan data PDF (*batch processing*), tingkat repetisi dikikis dari 3.69% menyusut ke **2.88%**. 

Penyusutan tingkat halusinasi ini menyegel validitas keampuhan stabilitas arsitektur DeepSeek-OCR 2 yang baru, menjamin ia memiliki urat nadi kapasitas pemahaman visual yang murni mengakar pada logika, bukan tebak-tebakan buta statistik teks.

---

## 6. Diskusi dan Arah Masa Depan (*Discussion and Future Works*)

### 6.1. Menuju Penalaran 2D yang Sejati (*Towards Genuine 2D Reasoning*)

Arsitektur DeepSeek-OCR 2 meletakkan sebuah bongkahan batu fondasi paradigma struktural yang teramat radikal: sebuah `encoder` gaya-LLM yang dipasang kaskade (beruntun) dengan sebuah `decoder` LLM. Rantai ganda dari "Dua Sosok Penalar Kausal 1D" (*two 1D causal reasoners*) ini menyimpan potensi ledakan evolusioner menuju pemahaman penalaran 2D asli yang seutuhnya.

**Tarian Kognisi Terpisah:**
*   Sang *Encoder* menjadi jenderal yang bertanggung jawab penuh atas **logika membaca visual (*reading logic reasoning*)**. Ia dengan cerdas dan taktis merombak ulang (*reordering*) deretan susunan bata informasi visual dengan merapal mantra kueri kausalnya.
*   Sang *Decoder* kemudian berperan sebagai hakim eksekutor. Ia bertugas melaksanakan eksekusi **penalaran atas isi (*visual task reasoning*)** di atas landasan bata-bata representasi yang sudah ditata rapi secara kausal oleh si *encoder*.

Kami di DeepSeek memiliki filosofi akademis bahwa membelah dan mendekonstruksi kerumitan pemahaman 2D ke dalam dua sub-tugas kausal 1D yang saling melengkapi (ortogonal) ini merepresentasikan terobosan masif untuk melampaui batas horizon menuju Penalaran 2D Tulen (*genuine 2D reasoning*).

Memang, mendaki ke puncak keabadian ini adalah perjalanan panjang yang terjal. Sebagai contoh teoretis, jika kelak kita bermimpi ingin menyuruh model untuk sanggup "melihat ulang" (*re-examinations*) dan menyortir kembali melompat-lompat (*multi-hop reordering*) melintasi benua konten visual, kita dipastikan akan ditagih biaya komputasi untuk membentangkan token aliran kausal (*causal flow tokens*) dalam untaian panjang yang jauh lebih ekstrem ketimbang deretan token visual aslinya. 

### 6.2. Menuju Arsitektur Multimodal Asali (*Towards Native Multimodality*)

Kerangka besi DeepEncoder V2 menyajikan bukti empiris validasi tahap awal yang meyakinkan atas kelayakan arsitektur bergaya-LLM dipaksa bekerja mengekskavasi fitur visual. Namun, implikasi yang paling merinding dari arsitektur ini adalah **potensinya untuk bermutasi berevolusi menjadi satu penyandi "Omni-Modal Terpadu" (*unified omni-modal encoder*)**.

Bayangkan fiksi ilmiah ini: Satu buah mesin *Encoder* soliter tunggal—di mana fungsi atensi, jaringan FFN, dan proyeksi $W_k, W_v$-nya digunakan beramai-ramai (*shared*)—namun ia secara magis mampu menelan, membedah, dan memproses trinitas modalitas (teks, suara/suara, dan gambar) secara simultan! 

Bagaimana caranya? Cukup dengan menyematkan (*embeddings*) kueri *learnable* (yang dapat dilatih) yang bertindak spesifik sebagai kunci identitas untuk masing-masing modalitas tersebut. Enkoder sakti semacam ini berkuasa untuk **mengompresi teks berpanjang-kilometer, mengekstrak spektrum frekuensi suara manusia, dan membongkar ulang susunan teka-teki logika visual konten**, di mana semuanya dibakar di dalam satu kuali tungku ruang parameter (*parameter space*) yang identik. Perbedaan pemrosesan antarmodalitas hanyalah dibedakan oleh bobot matematis yang bersembunyi di dalam penyematan kuerinya semata. 

Mahakarya "Kompresi Optikal" (*optical compression*) dari DeepSeek-OCR merepresentasikan ayunan langkah perdana kami meraba-raba menuju era *native multimodality*. Kami berikrar akan terus tanpa henti merangsek memburu eksplorasi integrasi modalitas-modalitas indera dunia lainnya agar ditundukkan dan dilebur paksa bernaung di bawah naungan atap kerangka *encoder* bersama ini pada masa depan.

---

## 7. Kesimpulan (*Conclusion*)

Di dalam manuskrip laporan teknis mendalam ini, kami mengedepankan **DeepSeek-OCR 2**. Entitas ini merupakan sebuah pemutakhiran eskalasi kelas berat atas silsilah DeepSeek-OCR terdahulu. Ia direkayasa dengan penuh dedikasi untuk secara absolut menahan dan mengunci erat rasio kekuatan kompresi token visual di titik tertinggi, seraya secara paradoksikal meraup ledakan peningkatan performa akurasi pemahaman yang begitu bermakna.

Bahan bakar pendorong kemajuan drastis ini dinahkodai oleh **DeepEncoder V2**, mahakarya arsitektur usulan kami yang secara *implicit* (tersembunyi) mendinginkan (mendistilasi) pemahaman kausal dari fenomena optik dunia nyata. Ia merealisasikannya dengan jalan mengintegrasikan perpaduan dua aliran atensi secara hibrida: mekanisme atensi dwi-arah (*bidirectional*) yang dikawinkan silang secara mekanis dengan atensi kausal (*causal*). Fusi ini berbuah manis pada terbentuknya kecerdasan kemampuan penalaran kausal yang tertanam langsung di dalam organ *vision encoder*, yang konsekuensi logisnya, menghasilkan eskalasi tajam yang sangat kentara (*marked lifts*) di dalam silogisme logika pembacaan visual (*visual reading logic*).

Kendatipun seni membaca teks optikal, terkhususnya di ranah *parsing* dekonstruksi dokumen, memahkotai dirinya sebagai salah satu tugas visi paling menguntungkan secara industrial dan pragmatis di era raksasa LLM ini, patut diinsafi bahwa ia sejatinya hanyalah sebutir pasir di padang luas spektrum pemahaman wawasan visual alam semesta (*visual understanding landscape*). 

Menatap bentangan masa depan di garis cakrawala, kami berkomitmen untuk menajamkan dan mengadaptasi rancang bangun arsitektur ini agar sanggup bertarung menaklukkan skenario alam liar yang semakin bervariasi. Kami memburu dan menyelam lebih dalam menuju epos penciptaan kecerdasan multimodal universal yang komprehensif, totalitas, dan holistik sejati.

*(Tamat - Penutup Diktat Akademis).*
