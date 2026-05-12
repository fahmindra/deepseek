---
slug: bab-08
title: "Konsolidasi"
layout: chapter
---


# **Bab 8 Konsolidasi Pasca-Pelatihan: Dari Penyulingan Standar hingga Penyulingan Sesuai Kebijakan (On-Policy Distillation)**

## **Yang akan kita pelajari pada bab ini:**
*   Paradigma (pola pikir) Pasca-Pelatihan Dua Tahap dari DeepSeek-V4
*   Melatih AI ahli spesialis yang berdiri sendiri-sendiri (Ahli Matematika, Ahli Kode Pemrograman, Ahli Agen Mandiri)
*   Definisi baru tentang Pemindahan Ilmu (Penyulingan Pengetahuan): *On-Policy Distillation* (OPD)
*   Cara kerja rumus Kerugian KL Terbalik (*Reverse KL Loss*) dengan menggunakan banyak Guru
*   Penyulingan skor probabilitas (logit) Seluruh Kosakata dan Penjadwalan Guru yang Efisien
*   Penyulingan Standar: Memampatkan kemampuan menalar ke dalam model kecil agar bisa digunakan oleh masyarakat luas

Pada Bab 7, kita telah mempelajari cara menggunakan metode Belajar dari Coba-Coba (*Reinforcement Learning* / GRPO) untuk mengajari model dasar AI tentang bagaimana cara berpikir, memeriksa hitungan matematika, dan menjaga ingatan saat menggunakan alat bantu dari luar. Namun, sebuah tantangan raksasa muncul ketika kita mencoba melatih satu "Kecerdasan Buatan Umum" tunggal yang bisa melakukan segalanya.

Jika Anda mencampur tugas menulis cerita kreatif tingkat tinggi, tugas pembuktian rumus matematika yang kaku, dan tugas ilmu komputer (pemrograman) yang rumit ke dalam satu tahap pelatihan RL saja, nilai perbaikannya (gradien) akan saling bertabrakan. Model AI tersebut akan menderita penyakit yang disebut *lupa yang fatal (catastrophic forgetting)*. Di mana, saat kita memaksanya untuk pintar dalam aturan menulis kode komputer yang sangat kaku, kemampuannya untuk menulis percakapan manusia yang mengalir dan kreatif justru akan menurun drastis.

Lalu, bagaimana cara kita menciptakan sebuah model yang benar-benar ahli dalam *segala hal* tanpa mengorbankan kemampuan apa pun? Lebih jauh lagi, setelah kita berhasil menciptakan kecerdasan super berukuran 671 miliar parameter (sel saraf buatan) ini, bagaimana cara kita memampatkan (menyuling) pengetahuannya sehingga masyarakat umum benar-benar bisa menjalankannya di perangkat komputer biasa?

Bab ini merupakan penutup dari **Tahap 4** peta jalan kita. Kita akan menjelajahi batas paling mutakhir dari penggabungan (konsolidasi) kemampuan AI setelah masa pelatihan dasar selesai.

**Gambar 8.1 Perjalanan empat tahap kita untuk membangun model DeepSeek. Bab terakhir ini berfokus pada Konsolidasi Pasca-Pelatihan, khususnya metode Penyulingan On-Policy (OPD) dan Penyulingan Standar.**
![Gambar 8.1 Perjalanan empat tahap kita untuk membangun model DeepSeek.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/08__image001.png)

Pertama, kita akan membedah pola pikir pasca-pelatihan dua tahap yang diperkenalkan pada DeepSeek-V4. Kita akan melihat bagaimana mereka melatih model-model "Spesialis" secara terpisah, lalu menggabungkan isi otak mereka semua ke dalam satu model tunggal dan utuh menggunakan **Penyulingan On-Policy (On-Policy Distillation / OPD)**. Kita akan menyelami ilmu matematika di balik rumus *Reverse KL Loss* dan infrastruktur tingkat ekstrem yang dibutuhkan untuk mengatur triliunan parameter (sel saraf) Guru pada waktu yang bersamaan.

Terakhir, kita akan melihat **Penyulingan Standar**, untuk membuktikan bagaimana DeepSeek memindahkan kemampuan jalan pikiran `<think>` mereka yang canggih ke dalam model berukuran kecil yang padat (ukuran 1,5 Miliar hingga 70 Miliar parameter), demi meratakan akses (demokratisasi) teknologi AI terdepan bagi seluruh dunia.

---

## **8.1 Paradigma Pasca-Pelatihan Dua Tahap (DeepSeek-V4)**

Pada versi sebelumnya seperti DeepSeek-V3.2, tim peneliti menggunakan "Pelatihan RL Campuran", di mana mereka mencoba menyeimbangkan porsi data untuk kemampuan menalar, tugas agen mandiri, dan penyelarasan dengan bahasa manusia di dalam satu tahap pelatihan coba-coba (RL) saja.

Namun pada DeepSeek-V4, tim meninggalkan pendekatan campuran ini. Mereka menyadari bahwa untuk mencapai kemampuan mutlak terbaik (*state-of-the-art*) di suatu bidang ilmu tertentu, mereka harus mendorong model AI tersebut hingga ke batas maksimal murni di bidang itu saja, tanpa diganggu oleh tujuan belajar lain yang saling bertentangan. Oleh karena itu, mereka menerapkan **Paradigma (Pola Pikir) Dua Tahap**.

### **8.1.1 Tahap 1: Pelatihan Spesialis Secara Mandiri**

Pada tahap pertama, jalur pipa pasca-pelatihan dipecah belah secara total. Untuk setiap bidang ilmu target—seperti ilmu matematika, penulisan kode perangkat lunak, penggunaan alat secara mandiri (agen), dan kemampuan mematuhi perintah—sebuah model ahli (pakar) yang terpisah akan dilatih secara benar-benar mandiri sendirian.

1.  **Awalan SFT:** Model dasar pertama-tama menjalani Penyetelan Halus Terarah (*Supervised Fine-Tuning* / SFT) menggunakan data berkualitas tinggi khusus untuk bidang ilmunya guna membangun kemampuan dasar (misalnya, belajar bentuk susunan kode Python atau label tag XML untuk menggunakan alat).
2.  **GRPO Khusus Bidang Ilmu:** Selanjutnya, metode Belajar dari Coba-Coba (GRPO) diterapkan. Hal yang paling krusial di sini adalah model penilai (reward models) dan penilai aturan pastinya disesuaikan *secara eksklusif (khusus)* hanya untuk menilai kesuksesan di bidang tersebut saja. Sang ahli Matematika hanya akan menerima nilai (imbalan) berdasarkan matematika; sang ahli Agen hanya akan menerima nilai dari kemampuannya menyelesaikan ujian di lingkungannya.

Fase ini menghasilkan kumpulan ahli spesialis yang beraneka ragam. Masing-masing ahli menunjukkan kehebatan tingkat dunia di bidangnya sendiri-sendiri, namun mereka tidak memiliki kemampuan umum di bidang lain.

### **8.1.2 Tahap 2: Penggabungan Model Terpadu (Konsolidasi)**

Tahap kedua adalah fase penggabungan. Tujuannya adalah untuk menyatukan kembali model-model ahli yang wujud fisiknya terpisah ini (yang sekarang akan kita panggil sebagai model "Guru") ke dalam satu buah model dasar tunggal dan terpadu (yang kita panggil "Murid").

Metode lama yang biasa digunakan untuk melakukan ini adalah *Penggabungan Bobot* (merata-rata nilai bobot matematika dari model-model yang berbeda tersebut) atau *RL Campuran* (menggunakan model para ahli untuk memberikan skor jawaban bagi si Murid). Sayangnya, penggabungan bobot sering kali menyebabkan penurunan kecerdasan yang parah karena parameter angka-angkanya saling bertabrakan dan merusak satu sama lain. Sementara itu, metode RL campuran biasa terbukti sangat tidak stabil.

Untuk mencapai penggabungan yang sempurna tanpa cacat, DeepSeek-V4 menggunakan **Penyulingan On-Policy melibatkan banyak guru (Multi-teacher On-Policy Distillation / OPD)**.

---

## **8.2 Definisi Baru Pemindahan Ilmu (Knowledge Distillation): Penyulingan On-Policy (OPD)**

Penyulingan On-Policy (OPD) kini telah muncul sebagai standar pasca-pelatihan paling utama untuk memindahkan (menyuling) kemampuan mendalam dari banyak model ahli (pakar) ke dalam satu model tunggal yang utuh secara sangat efisien.

Dalam metode RL biasa, model AI memperbarui isi otaknya berdasarkan sebuah nilai pasti (misalnya nilai +1 untuk jawaban benar, dan -1 untuk jawaban salah). Dalam metode OPD, model si Murid memperbarui isi otaknya dengan cara melihat *distribusi probabilitas keluaran yang lengkap (seluruh pola pikir persentase tebakan kata)* dari model-model si Guru. Murid di sini benar-benar mencoba untuk meniru "proses pemikiran" yang sama persis (yang disebut skor *logits*) milik para Guru saat si Murid sedang merenungkan jawabannya sendiri.

### **8.2.1 Cara Kerja Rumus Kerugian KL Terbalik (Reverse KL Loss) dengan Banyak Guru**

Secara matematika, kita diberikan sekumpulan $N$ model guru ahli, yang kita simbolkan sebagai $\{ \pi_{E_1}, \pi_{E_2}, \dots, \pi_{E_N} \}$.

Fungsi tujuan dari metode OPD ini adalah menekan hasil perhitungan dari rumus berikut agar sekecil mungkin:

$$ \mathcal{L}_{OPD}(\theta) = \sum_{i=1}^N w_i \cdot \mathbb{D}_{KL} (\pi_\theta \parallel \pi_{E_i}) $$

Mari kita bedah rumus matematika ini:

1.  **Cara Pikir Murid ($\pi_\theta$):** Ini adalah model tunggal utuh yang saat ini sedang aktif kita latih.
2.  **Cara Pikir Guru ($\pi_{E_i}$):** Ini adalah model ahli spesialis yang otaknya sudah dikunci (tidak dilatih lagi) untuk bidang ilmu $i$.
3.  **Bobot Bidang Ilmu ($w_i$):** Ini adalah nilai pengatur yang bisa berubah-ubah yang diberikan kepada setiap ahli, ditentukan dari seberapa penting keahlian guru tersebut terhadap pertanyaan yang sedang dikerjakan saat itu. Jika pertanyaan saat itu adalah soal matematika, maka bobot guru matematika ($w_{math}$) akan sangat tinggi, sedangkan bobot guru coding ($w_{coding}$) akan menjadi nol. Logika di balik ini memastikan agar otak model murid *hanya* mencontek dan belajar dari guru spesialis yang tepat sesuai dengan konteks soal yang sedang ia kerjakan.
4.  **Perbedaan KL Terbalik / Reverse KL Divergence ($\mathbb{D}_{KL} (\pi_\theta \parallel \pi_{E_i})$):** Kita secara khusus menghitung nilai Kerugian (Loss) *Kullback-Leibler (KL) Terbalik*. Untuk menghitung ini, kita diwajibkan untuk mengambil contoh jawaban murni dari si Murid $\pi_\theta$ saja demi mempertahankan agar cara belajarnya tetap "on-policy" (sesuai dengan kebijakan pikiran murid saat ini, bukan menyontek jawaban lama).

Melalui mekanisme ini, ilmu dari bobot-bobot otak pakar yang wujud fisiknya terpisah, berhasil digabungkan ke dalam satu ruang memori yang menyatu lewat **penyelarasan di tingkat logit (tingkat proses pemikiran kata)**. Karena si Murid mencocokkan tebakannya dengan pola distribusi tebakan yang kaya dari para Guru (bukan sekadar menerima nilai +1 atau -1), metode ini sukses menghindari penurunan kepintaran yang biasanya selalu terjadi pada teknik penggabungan bobot konvensional (lama).

### **8.2.2 Penyulingan Logit Seluruh Kosakata untuk Menstabilkan Pembaruan Otak (Gradien)**

Pada masa lalu, saat industri AI mencoba menerapkan metode OPD ini, para periset biasanya menyederhanakan beban kerja komputer yang raksasa tersebut dengan cara mengubah perhitungan rumus *KL Loss* yang seharusnya mencari di seluruh kamus kosakata, menjadi hanya sekadar perkiraan nilai keuntungan pada satu kata tunggal saja. Mereka menggunakan ulang kerangka kerja RL standar dengan cara mengganti nilai keuntungannya dengan rumus perkiraan KL pada tingkat kata tunggal:

$$ \text{sg} \left[ \log \frac{\pi_{E_i}(y_t | x, y_{<t})}{\pi_\theta(y_t | x, y_{<t})} \right] $$

*(Di mana `sg` melambangkan operasi penghentian-gradien / stop-gradient).*

Meskipun pendekatan ini sangat irit energi komputer (karena Anda hanya perlu membandingkan nilai peluang dari *satu* kata yang dipilih saja, bukan seluruh kata di kamus), para periset DeepSeek menemukan bahwa hal ini menimbulkan ketidakstabilan (variansi) yang sangat besar dalam penaksiran nilai perbaikan otak (gradien), yang mana hal ini sangat sering menyebabkan proses pelatihan hancur dan gagal.

Untuk memecahkan masalah ini, DeepSeek-V4 secara kaku menerapkan **Penyulingan Logit Seluruh Kosakata (Full-Vocabulary Logit Distillation)**. Ketimbang hanya melihat nilai probabilitas (peluang) pada kata yang dipilih, perhitungan rumus ini mencari perbedaan nilai yang ada di *seluruh* sebaran distribusi 128.000 kosakata kamus yang dimiliki AI. Dengan mempertahankan seluruh bentuk distribusi logit secara utuh, hal ini memberikan nilai tebakan perbaikan (gradien) yang sangat stabil secara matematika, dan menjamin terjadinya proses penyulingan (pemindahan) ilmu pengetahuan dari para guru yang sangat sempurna, mulus, dan memiliki kesetiaan tinggi terhadap makna aslinya.

### **8.2.3 Penjadwalan Guru yang Efisien: Mengatur Triliunan Parameter Guru**

Menerapkan metode OPD Seluruh-Kosakata menggunakan lebih dari sepuluh model guru yang berbeda—di mana masing-masing guru bisa memiliki hampir satu triliun sel saraf parameter—menghadirkan mimpi buruk teknis yang nyaris kiamat bagi perangkat keras komputer. Bagaimana cara Anda memasukkan sepuluh model AI berukuran 671 Miliar parameter ke dalam memori kartu grafis (VRAM) secara bersamaan pada waktu yang sama?

Agar metode OPD Seluruh-Kosakata ini secara fisik mungkin untuk dijalankan dalam skala raksasa, DeepSeek merekayasa tiga pengoptimalan infrastruktur tingkat ekstrem:

**1. Penyimpanan Terdistribusi Terpusat (Memuat Hanya Saat Dibutuhkan / On-Demand Loading)**
Kerangka kerja ini mendukung jumlah guru yang pada dasarnya tidak terbatas. Semua bobot (ingatan) dari guru-guru ini dipindahkan dan disimpan secara permanen di sebuah sistem penyimpanan data pusat yang tersebar (yaitu sistem *3FS* milik DeepSeek). Selama proses gerak maju saat guru ingin ditanya, ingatan otak guru ini dipanggil murni hanya pada saat sedang dibutuhkan saja, menggunakan metode pemecahan parameter mirip sistem ZeRO. Ini dilakukan secara brutal untuk meringankan tekanan berat pada ruang penyimpanan (I/O) dan memori utama komputer (DRAM).

**2. Menyimpan Status Tersembunyi (Hidden State) dari Lapisan Terakhir (Menghindari Pembuatan Logit Utuh)**
Secara naif memaksa komputer untuk menciptakan wujud angka logit penuh sebanyak 128.000 kosakata untuk kumpulan ribuan soal (batch sizes) di semua guru secara bersamaan, akan memakan ratusan gigabyte memori hanya untuk ruang perhitungan sementaranya saja—bahkan jika data itu langsung dibuang ke hardisk, hal ini tetap sangat memberatkan.
DeepSeek memecahkan ini dengan cara *hanya menyimpan nilai Status Tersembunyi (Hidden States) dari lapisan paling akhir* milik para guru ke dalam memori sementara selama langkah berjalan maju. Saat tiba waktunya pelatihan, data *Hidden States* yang ukurannya jauh lebih kecil ini diambil dan dilewatkan dengan cepat melalui kepala penerjemah (prediction head) milik sang guru untuk merangkai dan membangun kembali wujud 128.000 logit utuh itu *secara instan dalam sekejap mata (on the fly)*. Teknik ini seratus persen berhasil mengelabui dan menghindari beban memori berat yang biasanya terjadi jika kita memaksakan pembuatan logit secara langsung (eksplisit).

**3. Mengurutkan Berdasarkan Nomor Urut Guru**
Untuk lebih menekan beban memori GPU akibat bongkar-pasang (keluar masuk) kepala penerjemah guru yang berukuran raksasa itu, mesin pengatur data secara aktif akan mengurutkan susunan contoh soal ujian berdasarkan nomor indeks guru siapa yang harus menilainya. Hal ini menjamin bahwa setiap kepala guru yang berbeda akan dimasukkan ke dalam memori perangkat GPU cukup *hanya satu kali* saja per setiap kelompok ujian kecil (mini-batch). Ini memastikan bahwa pada setiap detiknya (mikrodetik), maksimal hanya ada *satu* kepala guru saja yang mendiami memori VRAM.

Semua proses bongkar-pasang pemuatan data ke memori (loading/offloading) ini berjalan secara gaib di latar belakang sistem secara tidak bersamaan (asynchronous) melalui jalur jalan khusus (*CUDA streams*). Ini memastikan agar proses perhitungan hitung-menghitung inti di jalur utama tidak pernah terhenti sedetik pun akibat harus menunggu data.

---

## **8.3 Penyulingan Standar: Membuat Model AI Terbuka Menjadi Praktis Digunakan oleh Masyarakat Luas**

Meskipun metode Penyulingan On-Policy (OPD) digunakan untuk menciptakan model dasar AI raksasa yang menjadi produk andalan (seperti DeepSeek-V4-Pro), model hasil akhirnya masih terlalu raksasa untuk bisa dijalankan oleh para peneliti biasa, pembuat aplikasi (developer), dan masyarakat penghobi menggunakan perangkat komputer keras lokal (komputer rumahan).

Untuk memenuhi janji mereka dalam mendemokratisasikan AI (memberikan akses AI yang merata untuk semua orang), DeepSeek mengandalkan bentuk penyulingan ilmu kedua yang lebih tradisional: yaitu memampatkan kehebatan pola pikir dari model andalan yang raksasa tersebut ke dalam arsitektur model AI yang jauh lebih kecil dan padat.

**Konsepnya:**
Sebuah model yang luar biasa raksasa (sang "Guru") telah mempelajari kemampuan menalar dan berpikir rumit berbilang langkah melalui proses latihan Coba-Coba (RL) selama ribuan jam. Sementara itu, sebuah model kecil (sang "Murid," misalnya model berukuran 7 Miliar parameter seperti struktur *Llama* atau *Qwen*) tidak memiliki kapasitas otak yang cukup untuk sanggup mempelajari pola-pola rumit ini jika ia disuruh belajar dari nol sendiri.

Akan tetapi, jika kita memberikan pertanyaan kepada si model Guru yang raksasa itu untuk menghasilkan jutaan rekam jejak penalaran jalan pikiran berkualitas tinggi (lengkap dengan label kurung `<think>...</think>` dan pemanggilan alat XML), maka kita bisa menggunakan kumpulan data jawaban yang brilian ini sebagai bahan bacaan buku pelajaran untuk melatih si Murid yang masih kecil ini.

**Prosesnya:**
DeepSeek menyusun sebuah kumpulan data raksasa berisi pertanyaan-pertanyaan yang sangat sulit. Pertanyaan ini diberikan kepada model andalan utama mereka yaitu DeepSeek-R1 atau V4, lalu sistem menyimpan seluruh jejak pemikiran `<think>` dan jawaban akhir dari model raksasa tersebut. Selanjutnya, mereka menerapkan **Penyetelan Halus Terarah (Supervised Fine-Tuning / SFT)** pada model-model kecil bersumber terbuka (open-source) tersebut dengan menggunakan tepat kumpulan data jawaban cerdas ini.

**Hasilnya:**
Hasil dari penyulingan standar ini sungguh sangat mencengangkan. Model-model kecil tersebut—meskipun tidak pernah mengalami sendiri fase *Reinforcement Learning* (Coba-Coba) yang super mahal itu—ternyata berhasil meresapi dan meniru logika dasar serta strategi bedah masalah selangkah demi selangkah milik si guru raksasa.

Model-model mungil yang ukurannya hanya 1,5 Miliar atau 7 Miliar parameter secara ajaib belajar dan mulai secara alami memunculkan label `<think>` dari mulut mereka sendiri. Mereka juga sukses memecahkan masalah matematika dan masalah kode pemrograman komputer yang sangat rumit, yang mana di masa lalu pencapaian ini dianggap sebagai sesuatu yang mustahil untuk bisa dilakukan oleh model AI dengan ukuran sekecil itu. Dengan membuka dan membagikan model-model hasil penyulingan ini secara gratis kepada publik (*open-source*), DeepSeek memastikan bahwa kecerdasan berpikir kelas wahid (state-of-the-art) sekarang bisa dijalankan hanya dengan menggunakan laptop standar biasa. Hal ini benar-benar telah mengubah sejarah tentang apa saja yang mungkin bisa dilakukan oleh perangkat komputer rumahan di ujung jari masyarakat.

---

## **8.4 Ringkasan**

*   **Paradigma (Pola Pikir) Dua Tahap:** Untuk mencegah penyakit lupa permanen (lupa fatal) dan bentroknya nilai perbaikan (gradien), DeepSeek-V4 memecah jalur dan melatih pakar spesialis di bidang khusus secara mandiri (Ahli Matematika, Ahli Kode, Ahli Agen) menggunakan metode GRPO yang terisolasi, baru setelahnya isi otak mereka semua digabungkan jadi satu.
*   **Penyulingan On-Policy (On-Policy Distillation / OPD):** Ini adalah metode kasta tertinggi untuk menggabungkan kepintaran banyak model ahli sekaligus. Model Murid tunggal memperbarui isi otaknya dengan cara menekan nilai perbedaan rumus *Reverse KL Divergence* saat membandingkan tebakannya dengan sebaran persentase distribusi jawaban dari model-model para Guru.
*   **Penyulingan Seluruh-Kosakata (Full-Vocabulary Distillation):** Alih-alih memakai tebak-tebakan pada tingkat perkiraan satu-kata yang sangat tidak stabil, DeepSeek menghitung secara tuntas perbedaan nilai pada seluruh 128.000 kata di dalam kamusnya secara utuh. Ini menjamin proses pembaruan otak (gradien) berjalan dengan luar biasa stabil tanpa hambatan.
*   **Infrastruktur Tingkat Ekstrem:** Agar triliunan sel saraf parameter milik para Guru muat di dalam memori komputer, DeepSeek mengusir bobot otak guru tersebut dan menyimpannya di hardisk penyimpanan 3FS. Mereka *hanya* menaruh wujud ringkas (hidden states) lapisan terakhir dari guru tersebut di memori ketimbang menaruh nilai logit (tebakan seluruh kamus) yang utuh, dan mereka melakukan bongkar-pasang kepala penerjemah guru di GPU secara instan dalam sekejap mata.
*   **Penyulingan Standar:** Demi mewujudkan AI yang merakyat untuk seluruh manusia, DeepSeek mencetak jutaan riwayat rekam jejak jalan pikiran (penalaran) dari model raksasa andalan mereka. Mereka lalu menggunakan metode SFT (menirukan jawaban) untuk memindahkan pola pikir `<think>` yang persis sama tersebut ke dalam otak model kecerdasan buatan terbuka (open-source) yang ukurannya jauh lebih mungil, praktis, dan padat (ukuran 1,5 Miliar hingga 70 Miliar parameter).