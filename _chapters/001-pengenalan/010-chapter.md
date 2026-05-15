---
slug: bab-1-1
title: "Pengantar tentang DeepSeek"
layout: chapter
---

# **BAB 1: Pengantar tentang DeepSeek – Momen "Sputnik" AI dan Titik Balik Inovasi Open-Source**

## **Yang akan kita pelajari pada bab ini:**
*   Mengapa DeepSeek menjadi titik balik (momen perubahan besar) dalam dunia AI *open-source* (sumber terbuka yang bisa diakses siapa saja), mulai dari awal mula model V3/R1 hingga model V3.2 dan V4 yang menembus batas teknologi.
*   Sejarah bagaimana DeepSeek mengguncang dominasi Silicon Valley, memicu perang harga, dan meruntuhkan hegemoni perangkat keras serta perangkat lunak Amerika Serikat.
*   Gambaran besar tentang inovasi-inovasi utama yang akan kita bangun bersama, termasuk *Sparse Attention* (Perhatian Renggang), *Engram*, *mHC*, dan *On-Policy Distillation* (Penyulingan Pengetahuan).
*   Struktur buku, batasan materi, dan apa saja yang perlu Anda siapkan.

---

## **1.1 Momen "Sputnik" AI: Bagaimana DeepSeek Mengguncang Dominasi Silicon Valley**

Bayangkan situasi ini: Pada bulan Januari 2025, sebuah perusahaan yang namanya hampir tidak pernah terdengar oleh siapa pun di luar China, tiba-tiba merilis sebuah model *Artificial Intelligence* (AI). Dampaknya? Rilis ini secara brutal menghapus valuasi pasar saham Amerika Serikat hingga lebih dari satu triliun dolar (sekitar Rp 15.000 triliun) hanya dalam waktu satu hari. 

Perusahaan itu bernama **DeepSeek**. Mereka hanyalah sebuah laboratorium riset kecil di China yang didanai oleh perusahaan *hedge fund* (pengelola dana investasi). Mereka mengklaim telah berhasil melatih model AI yang mampu menyaingi model-model AI terbaik buatan Amerika, dan gokilnya, mereka hanya menghabiskan dana sekitar **$6 juta** (Rp 90-an miliar). Sebagai perbandingan, raksasa seperti OpenAI dan Anthropic bisa membakar uang sebesar itu hanya dalam hitungan hari. Rilis ini adalah langkah pembuka DeepSeek, dan apa yang terjadi setelahnya benar-benar mengubah lanskap teknologi dunia.

### **Siapa Sebenarnya DeepSeek?**
DeepSeek bukanlah raksasa teknologi konvensional seperti Google atau Meta. Perusahaan ini lahir dari *High-Flyer*, sebuah perusahaan *quantitative trading* (trading saham menggunakan algoritma matematika kompleks) yang diam-diam telah mendanai riset AI selama bertahun-tahun. Lab ini sebenarnya sudah menerbitkan beberapa jurnal riset serius, tapi dunia Barat terlalu sibuk dengan diri mereka sendiri untuk menyadarinya.

Semua berubah pada Januari 2025 ketika mereka merilis **R1**, sebuah model *reasoning* (penalaran) bersifat *open-source* yang performanya menyamai model **o1 milik OpenAI** di berbagai uji tolok ukur (*benchmarks*) utama. Hampir dalam semalam, R1 menjadi aplikasi gratis paling banyak diunduh di Amerika Serikat. 

Hal ini membuat Mark Andreessen, salah satu investor *Venture Capital* paling legendaris di Silicon Valley, menyebut kejadian ini sebagai **"Momen Sputnik AI"**. Analogi ini sangat pas. Sama seperti saat Uni Soviet meluncurkan satelit Sputnik yang bikin Amerika panik di era Perang Dingin, China baru saja membangun sesuatu yang selama ini dianggap mustahil oleh AS: model AI yang lebih murah, lebih cepat, dan lebih terbuka (*open-source*) daripada apa pun yang pernah dipersiapkan oleh Silicon Valley.

---

## **1.2 Mengapa DeepSeek? Titik Balik dalam AI Open-Source**

Melihat fenomena di atas, Anda mungkin bertanya-tanya: dengan begitu banyaknya model bahasa di luar sana, mengapa kita harus fokus pada DeepSeek? Apa yang membuat model ini begitu istimewa sehingga kami memutuskan untuk menulis satu buku penuh tentang cara membangunnya? Jawaban singkatnya adalah bahwa DeepSeek mewakili titik balik dalam dunia AI *open-source*. DeepSeek membuktikan untuk pertama kalinya bahwa sebuah model yang tersedia secara terbuka (gratis untuk diakses kodenya) dapat menyaingi—dan terkadang mengalahkan—kinerja model-model berbayar terbaik buatan perusahaan besar.

Mari kita mundur sejenak dan melihat keadaan AI sebelum DeepSeek muncul. Pada awal tahun 2020-an, model bahasa besar didominasi oleh beberapa raksasa teknologi dan laboratorium penelitian yang memiliki dana dan sumber daya sangat besar. Seri GPT dari OpenAI dan Gemini dari Google memimpin di posisi teratas dalam hal kemampuan, tetapi model tersebut bersifat tertutup (hanya bisa diakses dengan membayar). Komunitas *open-source* memang memiliki keberhasilan melalui model seperti LLaMA dan Qwen, namun selalu ada jarak ketertinggalan dalam hal kemampuan menalar tugas-tugas yang rumit.

DeepSeek muncul di tengah situasi ini, dan membawa konsep keterbukaan ke tingkat yang baru. Didirikan pada tahun 2023, laboratorium penelitian ini dengan cepat membuat gebrakan. Kehebatan DeepSeek pertama kali tidak bisa disangkal saat mereka merilis DeepSeek-R1 yang mengejutkan komunitas AI karena berhasil menyamai model o1 milik OpenAI pada ujian-ujian sulit, seperti *American Invitational Mathematics Examination* (AIME) dan berbagai platform kompetisi *coding* (pemrograman).

Namun, DeepSeek tidak berhenti di situ. Dengan perilisan cepat **DeepSeek-V3.2** dan **DeepSeek-V4**, mereka benar-benar menghancurkan batasan efisiensi. Mereka mampu membuat model yang bisa membaca teks sangat panjang (hingga 1 juta token/kata) dan menggunakan alat-alat bantu eksternal secara mandiri. DeepSeek-V3.2 bahkan meraih performa setara medali emas di Olimpiade Informatika Internasional (IOI), sementara DeepSeek-V4 menciptakan standar baru dengan arsitektur yang sangat efisien untuk diuji pada skala yang sangat besar. 

Fakta mengejutkan lainnya adalah masalah biaya: DeepSeek melatih model-model canggih ini hanya dengan sebagian kecil dari biaya yang dikeluarkan oleh perusahaan besar pembuat model berbayar. Bagi kita sebagai pelajar dan pembuat teknologi, DeepSeek adalah bahan pelajaran yang sempurna. Kesuksesan mereka berasal dari penemuan teknologi yang brilian dan dibagikan secara terbuka. Hal ini memunculkan pertanyaan-pertanyaan penting yang akan kita jawab di sepanjang buku ini:
*   Bagaimana cara DeepSeek mencapai hasil terbaik di dunia dengan kebutuhan komputer (komputasi) yang jauh lebih sedikit?
*   Bagaimana arsitektur seperti *DeepSeek Sparse Attention (DSA)* dan *Engram* (Memori Bersyarat) memecahkan masalah kemacetan pada teknologi Transformer standar?
*   Bagaimana cara Anda melatih AI agar "berpikir" terlebih dahulu sebelum ia menggunakan alat bantu atau mencari jawaban di internet?
*   Bagaimana beberapa AI kecil yang menjadi "ahli" di bidang tertentu digabungkan menjadi satu model utuh melalui proses yang disebut *On-Policy Distillation (OPD)*?

---

## **1.3 Tiga Inovasi Teknis Awal di Balik Kesuksesan DeepSeek**

Untuk paham kenapa DeepSeek bikin industri *tech* AS ketar-ketir pada peluncuran awalnya, kita harus bedah secara teknis *bagaimana* mereka membuatnya. Ancaman aslinya bukan sekadar "DeepSeek bikin AI bagus", tapi *cara* mereka membuatnya. Ada tiga inovasi utama yang menjadi fondasi awal mereka:

**1. Arsitektur *Mixture of Experts* (MoE)**
Model DeepSeek memiliki total **671 miliar parameter**—sebuah angka yang masif. Namun, untuk setiap tugas yang diberikan, model ini hanya mengaktifkan sekitar **37 miliar parameter**. 
Biar gampang bayangkannya: anggaplah ini sebuah gedung besar yang berisi penuh dengan tenaga spesialis. Daripada menyuruh semua spesialis bekerja keras memecahkan satu masalah secara bersamaan (yang boros energi), sistem ini hanya mengirimkan tugas tersebut ke spesialis yang paling relevan, sementara spesialis lainnya dibiarkan istirahat. Hasilnya? Model ini bekerja seolah-olah ia adalah AI raksasa, padahal beban komputasi (*compute*) yang berjalan di waktu yang sama sangatlah kecil. Ini sangat menghemat biaya dan daya.

**2. Pendekatan Baru dalam Mengajari AI Bernalar (*Reasoning*)**
Kebanyakan sistem AI belajar menalar melalui *unsupervised fine-tuning*. Artinya, developer menjejalkan ribuan contoh cara berpikir "langkah-demi-langkah" yang benar, lalu AI belajar meniru pola tersebut.
Tapi DeepSeek beda jalur. Mereka hampir sepenuhnya menggunakan metode **Reinforcement Learning** (pembelajaran penguatan). Mereka membiarkan AI mencoba berbagai strategi penalaran sendiri, dan baru memberikan "hadiah" (penguatan) ketika AI itu menghasilkan jawaban yang benar. Hasilnya di luar dugaan: model ini secara spontan mengembangkan kemampuan *chain of thought reasoning* (penalaran berantai) dan kemampuan untuk mendeteksi serta memperbaiki kesalahannya sendiri. AI ini belajar "cara berpikir" karena diberi *reward* saat menjawab benar, bukan karena disuapi contoh cara berpikir.

**3. Pemangkasan Biaya yang Gila-gilaan dengan Hardware "Ecek-ecek"**
Coba perhatikan tren biaya ini:
*   **DeepSeek V2**: Biaya pelatihannya hanya 1/17 dari biaya pelatihan GPT-4 Turbo.
*   **DeepSeek V3**: Biayanya turun lagi jadi 1/14 dari GPT-4.
*   **DeepSeek R1**: Biayanya cuma 1/20 dari biaya GPT-4o.

Yang paling epik? Semua pencapaian ini **TIDAK** dilakukan menggunakan *hardware* (perangkat keras) terbaik di pasaran. DeepSeek dilatih menggunakan GPU (Graphic Processing Unit) **Nvidia H800**. Untuk konteks, H800 adalah *chip* yang sengaja di-*downgrade* (diturunkan kualitasnya) oleh Nvidia khusus untuk pasar China, karena pemerintah AS melarang ekspor *chip* paling canggih mereka (H100) ke China pada tahun 2022. 
Jadi, menggunakan *hardware* "sisaan" yang dianggap pemerintah AS tidak cukup bagus untuk bikin AI canggih, DeepSeek justru berhasil mengguncang Silicon Valley.

---

## **1.4 Runtuhnya Teori Amerika, Perang Harga, dan Lahirnya DeepSeek V4**

Saat R1 meluncur, industri *tech* AS masih *denial* (menyangkal). Mereka punya teori: "China nggak akan bisa mempertahankan progres ini karena suplai *hardware* mereka sudah kita batasi dengan *chip* H800 yang lemot. Sebentar lagi mereka pasti mentok."

Teori itu hancur berantakan pada **April 2026** saat DeepSeek merilis **V4**, model paling signifikan sejak R1. Model ini hadir dalam dua versi:
1.  **V4 Pro**: Model MoE dengan **1,6 triliun parameter**, dilatih dengan 33 triliun token data. Dibuat khusus untuk *coding* tingkat dewa dan tugas *agentic* (AI yang bisa bertindak sendiri) yang kompleks.
2.  **V4 Flash**: Versi yang lebih kecil dan cepat dengan **284 miliar parameter**, dioptimalkan untuk kecepatan dan efisiensi harga.

Keduanya bersifat *open-source* di bawah lisensi MIT (tersedia di *Hugging Face*), yang artinya *developer* mana pun bisa mengunduh, memodifikasi, dan menjadikannya produk komersial. 

Secara kebetulan (atau mungkin disengaja), V4 dirilis di hari yang sama saat OpenAI merilis **GPT-5.5**. Siang itu, dua AI paling canggih di dunia rilis berbarengan, dan salah satunya bisa diakses secara gratis.
Secara performa, V4 Pro membantai semua kompetitor *open-source* di bidang matematika dan *coding*. Ia mendapat skor **120/120 (Sempurna)** di ujian matematika tersulit, Putnam 2025. Di tes *SWE-verified* (tes *coding*), ia mencetak skor **80,6%**, hanya beda sekian persen dari model *closed-source* (berbayar/tertutup) terbaik milik Amerika. DeepSeek memang jujur mengakui bahwa V4 masih tertinggal sekitar 3-6 bulan pengembangan dari AI nomor satu dunia. Tapi poinnya bukan jadi yang terbaik, melainkan performanya **cukup dekat** untuk menyaingi, namun dengan harga yang jauh lebih murah.

### **Perang Harga yang Bikin AS Pusing**
Di sinilah mimpi buruk bagi lab AS dimulai:
*   **DeepSeek V4 Pro**: $1,74 per juta *input tokens* | $3,48 per juta *output tokens*.
*   **GPT-5.5**: $5 per juta *input* | $30 per juta *output*.
*   **Claude Opus 4.7**: $5 per juta *input* | $25 per juta *output*.

Contoh kasus riil: Sebuah tugas komputasi yang memakan biaya **$35** menggunakan GPT-5.5, hanya butuh biaya lebih dari **$5** menggunakan DeepSeek. 
Untuk skala perusahaan besar (*enterprise*) yang butuh 10 juta *output token* per bulan, pakai GPT-5.5 akan menghabiskan **$300**, sementara pakai DeepSeek hanya **$35**. Selisih harga ini sangat tidak masuk akal, membuat perusahaan mana pun pasti akan sangat mudah beralih ke DeepSeek. 

Parahnya lagi bagi AS, OpenAI, Anthropic, dan Google masih harus membayar biaya pengoperasian (inferensi) menggunakan GPU Nvidia yang harganya sangat mahal dan susah turun. Sementara DeepSeek sudah memberi sinyal harga mereka akan terus turun karena Huawei mulai memproduksi massal *chip* buatan mereka sendiri.

### **"Bencana" Ekosistem: DeepSeek Beralih ke Huawei**
Ada satu bagian dari rilis V4 yang bikin Washington panik dan membuat CEO Nvidia, Jensen Huang, menyebutnya sebagai **"bencana" (catastrophic)**. 

DeepSeek V4 adalah model AI *Frontier* (kelas dewa) pertama yang berjalan secara *native* (bawaan) menggunakan *chip* **Ascend 950 buatan Huawei**, bukan lagi pakai *hardware* Nvidia! Ini adalah manuver strategis yang sangat disengaja. Menurut laporan *MIT Technology Review*, DeepSeek memberi akses awal arsitektur V4 kepada Huawei agar para *engineer* Huawei bisa mengoptimalkan *chip* Ascend khusus untuk V4. Akses yang sama **ditolak** mentah-mentah untuk diberikan kepada Nvidia atau AMD.

Kenapa ini jadi bencana buat Amerika? Karena keuntungan utama Nvidia bukan cuma jualan *chip* (GPU), tapi pada **CUDA**. 
CUDA adalah perangkat lunak/ekosistem yang dibangun Nvidia selama puluhan tahun. Saat ini, jutaan *developer* terbiasa menulis kode AI menggunakan CUDA. Sama seperti *developer* aplikasi zaman sekarang yang pasti bikin aplikasi dasar untuk platform iOS atau Android. Ketergantungan *developer* pada CUDA inilah (disebut *lock-in effect*) yang harganya jauh lebih mahal dari *chip* fisik mana pun.

Dengan membuktikan bahwa mereka bisa membuat AI canggih di ekosistem Huawei (tanpa Nvidia), DeepSeek menjadi ancaman nyata pertama terhadap dominasi CUDA. DeepSeek membuktikan bahwa perusahaan AI bisa bertahan walau harus membangun ulang sistem komputasi mereka dari nol tanpa Nvidia. 

Strategi kontrol ekspor AS yang membatasi suplai *chip* ke China ternyata gagal total. Alih-alih melambat, DeepSeek malah berhasil bikin model canggih dengan harga murah, pakai perangkat keras buatan lokal China. Meski performa *chip* Ascend belum 100% menyamai Nvidia, arah perkembangannya sudah sangat jelas.

---

## **1.5 Drama Skandal Distilasi dan Ancaman Geopolitik**

Di tengah semua ini, ada drama besar yang memanas. Pada **Februari 2026**, Anthropic menuduh DeepSeek melakukan pencurian kekayaan intelektual (industrial-scale distillation campaign) terhadap AI mereka, Claude.

Anthropic merilis angka spesifik: Ada sekitar **24.000 akun palsu** yang menghasilkan lebih dari **16 juta percakapan** dengan Claude. Tujuannya? Secara sistematis mengekstraksi dan menyalin cara bernalar (*reasoning*) yang selama ini dibangun Anthropic dengan dana miliaran dolar. DeepSeek dituduh menggunakan taktik *Hydrocluster architectures*, yaitu jaringan akun palsu yang mencampur permintaan data curian dengan permintaan wajar agar tidak terdeteksi sistem keamanan. Akun-akun ini memaksa Claude untuk menjelaskan cara berpikirnya secara *step-by-step* untuk dijadikan data pelatih bagi AI DeepSeek.

OpenAI juga melaporkan hal serupa ke Komite Khusus DPR AS untuk Urusan China. Google juga mengonfirmasi AI mereka, Gemini, diserang dengan lebih dari **100.000 prompt (perintah) terkoordinasi** dalam satu kampanye curang. Kasus ini sampai membuat Gedung Putih merilis memo resmi dan Departemen Luar Negeri AS mengirim kawat diplomatik ke negara sekutu.

*Distillation* (melatih model AI kecil dengan "mencontek" hasil jawaban model AI raksasa) sebenarnya adalah praktik normal di industri AI. Tapi tuduhan di sini adalah soal skalanya yang brutal dan penggunaan akun palsu. 

Namun, mari kita lihat ironinya. Para pengamat mencatat bahwa pada **September 2025**, Anthropic harus membayar denda ganti rugi sebesar **$1,5 miliar** akibat tuntutan ribuan penulis karena Anthropic ketahuan menyedot ratusan ribu buku bajakan tanpa izin untuk melatih AI-nya. OpenAI juga menghadapi tuntutan hukum serupa.
Jadi, poinnya sangat lucu bagi publik: Perusahaan AI Amerika membangun model mereka dengan cara "mencuri" data miliaran teks tanpa izin (yang melanggar hak cipta), tapi sekarang mereka teriak-teriak minta pemerintah memblokir perusahaan China karena diduga mencontek data dari mereka. Sebagian besar orang menganggap tuduhan AS ini murni karena mereka kalah saing secara bisnis dan politik, bukan murni soal ancaman keamanan nasional.

### **Ancaman Jangka Panjang: Penjajahan Melalui *Open-Source***
Tinggalkan sejenak soal drama pencurian data. Strategi geopolitik jangka panjang DeepSeek ada pada jalur distribusi *open-source*. Ini adalah ancaman paling mematikan.

Semua model DeepSeek dirilis gratis (Lisensi MIT), bebas dimodifikasi, dan bisa dijadikan produk jualan. Pada **Agustus 2025**, untuk pertama kalinya dalam sejarah, jumlah unduhan model *open-source* China di platform *Hugging Face* berhasil menyalip model Amerika. Di akhir 2025, model **Qwen** (buatan Alibaba, China) memiliki lebih banyak varian yang dibuat oleh *developer* di seluruh dunia dibandingkan gabungan dari model buatan Google dan Meta. 

Bahkan, **Pemerintah Singapura** sampai mengumumkan bahwa mereka membuang model *Llama* buatan Meta (AS), dan memilih membangun AI mandiri (Sovereign AI) kenegaraan mereka menggunakan *Qwen* dari China.

Logika strategi *open-source* ini sangat cerdas. Jika seorang *developer* atau negara membangun produk menggunakan arsitektur AI tertentu, mereka akan terikat pada alat, ekosistem, dan standar model tersebut. Menggantinya di kemudian hari akan sangat mahal dan merepotkan. 

Jika model *open-source* China ini menjadi fondasi *default* (bawaan) pengembangan AI di negara-negara berkembang seperti **Asia Tenggara, Afrika, Timur Tengah, dan Amerika Latin**—wilayah yang sangat peduli dengan harga murah—maka wilayah-wilayah ini secara tidak langsung akan menggunakan **standar teknologi China**. Ini adalah pengaruh geopolitik dalam skala masif yang tidak bisa diubah hanya dengan mengirim surat protes diplomatik dari Amerika.

### **Kesimpulan: Kekalahan Strategi Amerika Serikat**
Laporan **Stanford AI Index 2026** menyimpulkan sebuah kenyataan pahit: Perusahaan-perusahaan China telah berhasil menutup kesenjangan (*gap*) performa AI mereka dengan pesaing AS. 

Ini tidak berarti DeepSeek sudah 100% melampaui GPT-5.5 atau Claude Opus 4.7. Namun, jarak kualitasnya kini sudah **sangat tipis**, sehingga faktor **HARGA** dan **KEMUDAHAN AKSES** menjadi penentu utama bagi perusahaan dan *developer* di dunia nyata. Di sinilah letak ancaman substansialnya.

DeepSeek tidak harus membuat AI terbaik di dunia. Mereka hanya perlu membuat AI yang *cukup bagus* untuk digunakan bekerja, namun dijual dengan harga yang menghancurkan semua kompetitor Amerika. Hebatnya lagi, ini semua dilakukan di atas *hardware* yang kebal dari embargo Amerika Serikat, didistribusikan secara gratis ke seluruh planet ini, dengan grafik biaya yang terus *turun*, sementara biaya perusahaan AS terus *naik*.

Setiap asumsi yang dipegang teguh oleh AS—seperti keyakinan bahwa mereka unggul di suplai *chip*, unggul di modal kapital, dan aman di balik sistem AI tertutup (*closed-source*)—telah dihancurkan oleh lab kecil yang merespons setiap pembatasan dengan kecerdikan *engineering*. 
*   Blokir *chip* AS? Membuat China menemukan cara mendesain AI yang jauh lebih hemat daya.
*   Model AI Barat yang berbayar dan mahal? Dijadikan target sontekan (distilasi) dan dihancurkan lewat model distribusi gratisan.

Pola ini terlalu konsisten untuk dianggap kebetulan. Kelanjutannya sekarang bergantung pada apakah AS bisa mengejar ketertinggalan efisiensi mereka, atau apakah AI buatan China akan terus menjajah pasar negara berkembang yang sensitif harga. Namun satu hal yang pasti: asumsi industri tech bahwa gebrakan DeepSeek di **Januari 2025** hanyalah "keberuntungan sementara" atau anomali, kini terbukti salah besar. Itu adalah langkah pembuka dari sebuah invasi teknologi yang sistematis dan berkelanjutan.

---

## **1.6 Membongkar Mesinnya: Inovasi Utama yang Akan Kita Bangun**

Melihat betapa masifnya dampak geopolitik dan bisnis DeepSeek, kini kita beralih ke ranah praktis. Model Bahasa Besar (*Large Language Models* atau LLM) telah mengubah dunia teknologi. Kita sekarang hidup di dunia di mana sistem Kecerdasan Buatan (AI) dapat melakukan percakapan, menulis kode komputer, menyusun esai, menjalankan tugas-tugas rumit secara mandiri, dan bahkan memecahkan soal olimpiade matematika dengan cara yang terasa sangat mirip dengan manusia.

Namun, bagaimana jika Anda, seorang pembaca yang memiliki rasa ingin tahu tinggi terhadap teknologi, bisa membangun salah satu model AI yang sangat canggih ini dari nol? Bagaimana jika Anda bisa memahami cara kerja bagian dalam dari sebuah LLM paling modern dengan merakitnya langkah demi langkah, menggabungkan teori dan kode pemrograman secara bersamaan? Itulah tepatnya yang ingin kami ajarkan kepada Anda melalui buku ini.

Kita akan membongkar lapisan demi lapisan dari keluarga LLM *open-source* paling canggih saat ini. Kita akan menciptakan ulang inovasi-inovasi utamanya dari dasar. Pada akhir buku ini, Anda tidak hanya akan memahami apa yang membuat DeepSeek begitu unik secara teori, tetapi Anda juga akan tahu cara menerapkan inovasi tersebut sendiri. Sepanjang perjalanan ini, Anda akan mendapatkan wawasan yang sangat berharga tentang pengembangan AI modern.

Istilah-istilah seperti *Multi-Head Latent Attention*, *Mixture-of-Experts* (Campuran Para Ahli), *Conditional Memory* (Memori Bersyarat / *Engram*), *Manifold-Constrained Hyper-Connections* (*mHC*), dan Kuantisasi FP4 mungkin terdengar menakutkan bagi Anda saat ini. Namun, jangan khawatir, kami akan memperkenalkannya dengan cara yang mudah dipahami. 

Sebelum kita mulai, sedikit latar belakang: buku ini terinspirasi dari seri YouTube Vizuara yang berjudul "Build DeepSeek from Scratch" (Membangun DeepSeek dari Nol). Melalui buku ini, kami telah merangkum pengalaman kami dari dunia akademik (universitas) dan industri, serta pelajaran praktik langsung dari seri video tersebut, menjadi sebuah perjalanan terstruktur. 

Mari kita susun peta jalan arsitektur untuk membangun DeepSeek dari nol. Arsitektur (rancangan bangunan) DeepSeek didirikan di atas fondasi bernama *Transformer* yang sudah sangat dikenal. Namun, DeepSeek memperkenalkan inovasi besar untuk mengatasi masalah kinerja dan keterbatasan ukuran. Kita akan memperlakukan setiap inovasi sebagai studi kasus: pertama kita menganalisis kelemahan cara standar, lalu kita membangun teknologi penggantinya yang lebih canggih.

### **1.6.1 Arsitektur (Rancangan Bangunan)**
Inti dari sebuah LLM standar bergantung pada sistem *Multi-Head Attention* yang padat dan jaringan *Feed-Forward Networks* (FFNs). DeepSeek mengganti dan meningkatkan komponen standar ini dengan sistem pengganti yang sangat efisien dan khusus:
*   **Multi-Head Latent Attention (MLA) & Sparse Attention:** Sistem perhatian standar membutuhkan ruang memori penyimpanan sementara yang sangat besar (disebut *Key-Value / KV cache*). DeepSeek memperkenalkan MLA untuk memampatkan (mengecilkan) ukuran memori ini. Pada versi V3.2 dan V4, mereka mendorong hal ini lebih jauh dengan **DeepSeek Sparse Attention (DSA)** dan pendekatan campuran seperti **Compressed Sparse Attention (CSA)** serta **Heavily Compressed Attention (HCA)**. Hal ini mengurangi beban kerja komputer lebih dari 90% dan memungkinkan AI membaca teks sepanjang 1 juta kata.
*   **Mixture-of-Experts (MoE) / Campuran Para Ahli:** Daripada menyalakan setiap bagian otak AI untuk setiap kata yang dibaca, DeepSeek hanya mengirim kata tersebut ke bagian jaringan saraf "ahli" yang spesifik. Ini sangat memperbesar kapasitas kepintaran model tanpa membuatnya menjadi lambat.
*   **Memori Bersyarat melalui Engram:** Ini adalah cara baru untuk menghemat kerja AI. Sementara MoE menangani logika yang berubah-ubah, sistem AI standar sering kali buruk dalam mengingat fakta-fakta pasti. DeepSeek memperkenalkan *Engram*, sebuah modul yang menggunakan pencarian instan untuk mengambil pengetahuan pasti, sehingga jaringan otak AI bisa fokus sepenuhnya pada pemikiran dan penalaran yang rumit.
*   **Manifold-Constrained Hyper-Connections (mHC):** Semakin dalam (semakin banyak lapisan) sebuah model AI, sambungan standarnya bisa menjadi tidak stabil saat dilatih. DeepSeek-V4 memperkenalkan mHC, yang menempatkan sambungan tersebut ke dalam sebuah ruang matematika bernama *Birkhoff polytope* untuk menjamin proses pelatihan tetap stabil meskipun model tersebut memiliki triliunan parameter (sel saraf buatan).

### **1.6.2 Pelatihan**
Selain rancangan bangunan utamanya, DeepSeek juga berinovasi besar dalam hal efisiensi dan cara melatih AI:
*   **Prediksi Multi-Token (MTP):** Alih-alih hanya menebak *satu* kata berikutnya, DeepSeek menebak beberapa kata masa depan sekaligus secara bersamaan. Hal ini memberikan sinyal belajar yang lebih kuat dan membuat AI bisa bekerja jauh lebih cepat saat digunakan.
*   **Kuantisasi Ekstrem (FP8 dan FP4):** Melatih model berukuran triliunan parameter membutuhkan memori yang luar biasa besar. DeepSeek menggunakan ukuran data yang sangat kecil, yaitu *8-bit floating-point* (FP8), dan bahkan mendorong ke batas paling ekstrem yaitu 4-bit (FP4) untuk bagian pakar (MoE)-nya. Ini memotong penggunaan memori hingga setengahnya tanpa mengurangi keakuratan AI.
*   **Muon Optimizer:** Meninggalkan pengoptimal standar seperti AdamW, DeepSeek menggunakan pengoptimal bernama *Muon* dengan rumus matematika khusus untuk mencapai hasil belajar yang lebih cepat dan stabil saat dilatih menggunakan ribuan kartu grafis (GPU) yang saling terhubung.

### **1.6.3 Pasca-Pelatihan (Setelah Pelatihan Dasar)**
Sebuah model AI yang baru selesai dilatih dasar hanya bisa memunculkan teks biasa; ia perlu diajari *bagaimana* cara bertindak, berpikir secara masuk akal, dan menggunakan alat. Proses pasca-pelatihan DeepSeek adalah pelajaran berharga tentang bagaimana menyelaraskan AI modern:
*   **Scalable GRPO (Group Relative Policy Optimization):** Ini adalah algoritma pembelajaran khusus (Reinforcement Learning) milik DeepSeek yang memungkinkan model untuk mengeksplorasi dan mengembangkan pola "berpikir" melalui proses coba-coba (trial-and-error). Hal ini menghilangkan kebutuhan akan data berlabel buatan manusia yang jumlahnya sangat banyak dan harganya mahal.
*   **Agentic AI & Penggunaan Alat:** DeepSeek-V3.2 memperkenalkan metode canggih untuk "Manajemen Konteks Berpikir". Model ini belajar untuk mempertahankan jalan pikirannya sendiri di dalam kepalanya, saat ia sedang berinteraksi dengan alat luar (seperti kalkulator kode atau mesin pencari web) melalui ratusan langkah kerja.
*   **On-Policy Distillation (OPD):** Pada V4, DeepSeek menyempurnakan metode pasca-pelatihan dua tahap. Pertama, mereka melatih model-model "Spesialis" yang berdiri sendiri (misalnya, AI yang ahli Matematika, AI yang ahli *Coding*). Kemudian, mereka menggunakan metode OPD untuk menyuling dan memindahkan pengetahuan dari semua model guru yang besar tersebut ke dalam satu model murid yang tunggal dan sangat efisien.

---

## **1.7 Struktur dan Cakupan Buku**

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

---

## **1.8 Apa yang Akan Diajarkan Buku Ini dan Apa yang Tidak**

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

---

## **1.9 Apa yang Anda Butuhkan untuk Mengikuti Buku Ini**

Untuk mengikuti buku ini, Anda membutuhkan pemahaman dasar tentang konsep *Machine Learning* (Pembelajaran Mesin), tetapi Anda tidak membutuhkan gelar doktor (PhD). Jika Anda terbiasa dengan bahasa pemrograman Python dan pernah mempelajari materi pengantar *Deep Learning* (Pembelajaran Mendalam), Anda sudah siap untuk memulai. Secara khusus, Anda harus memahami bagaimana jaringan saraf buatan belajar melalui sebuah proses bernama *backpropagation* (perambatan balik), familier dengan operasi dasar di aplikasi PyTorch, dan pernah melihat bentuk bangunan arsitektur Transformer yang standar.

Dari segi perangkat keras komputer (hardware), kami telah merancang semua praktik ini agar mudah diakses. Walaupun melatih model paling mutakhir membutuhkan komputer super raksasa, kita akan bekerja dengan versi yang diperkecil. Versi kecil ini tetap mencakup semua ide matematika pentingnya, namun tetap sanggup dijalankan.

Sebuah laptop dengan prosesor (CPU) yang cukup bagus dapat menjalankan sebagian besar contoh-contoh konsep yang kecil. Akan tetapi, sebuah komputer yang dilengkapi satu kartu grafis (GPU) dengan memori VRAM 8-24GB (seperti NVIDIA RTX 3090, 4090, atau layanan *cloud* internet yang setara) akan membuat pengalaman belajar Anda jauh lebih lancar. Hal ini memungkinkan Anda untuk bereksperimen dengan bebas menggunakan MoE, Engram, dan Kuantisasi. Kami akan menyediakan rincian spesifikasi lingkungan kerja yang lengkap, beserta pengaturan *Google Colab* yang mudah digunakan untuk setiap babnya.

Mari kita bangun sesuatu yang luar biasa bersama-sama!

---

## **1.10 Ringkasan**

*   **Era DeepSeek:** DeepSeek menandai momen penting dalam sejarah AI dengan merilis model-model terbuka (*open-source* seperti R1, V3.2, V4) yang mampu menyamai atau bahkan melampaui kinerja raksasa teknologi berbayar, hanya dengan sebagian kecil dari biaya mereka. Hal ini memicu perang harga dan meruntuhkan strategi hegemoni teknologi Amerika Serikat.
*   **Belajar Langsung dengan Praktik:** Buku ini akan membimbing Anda membangun model "DeepSeek Mini" dari nol, dengan fokus pada terobosan teknologi khusus milik mereka, bukan sekadar konsep LLM biasa.
*   **Peta Jalan Delapan Bab:** Perjalanan kita mencakup empat area utama: (1) Masalah Kinerja Komputer & Sistem Perhatian, (2) Peningkatan Kapasitas dengan MoE & Engram, (3) Efisiensi Pelatihan Ekstrem (MTP, Kuantisasi, mHC), dan (4) Pasca-Pelatihan Dasar (Belajar Coba-Coba, Agen AI, Penyulingan Ilmu).
*   **Pemberdayaan:** Dengan membangun komponen-komponen ini sendiri, Anda akan mendapatkan pengetahuan praktis langsung dari prinsip dasar arsitektur AI yang paling mutakhir. Hal ini akan memberdayakan Anda untuk menyesuaikan teknik-teknik ini demi pengembangan AI di masa depan.