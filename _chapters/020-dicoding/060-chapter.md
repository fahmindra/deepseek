---
slug: ch-06-06
title: "Deployment Aplikasi Generative AI"
layout: chapter
---
# Deployment Aplikasi Generative AI

## Pengantar Deployment Aplikasi Generative AI

*Welcome back Coder’s* pada babak akhir sebelum bertemu pintu exit dari Kelas Pengembangan Generative AI Berbasis LLM.

Pada keempat modul sebelumnya, Anda sudah melewati fase-fase yang paling menguras otak dengan membedah anatomi mesin bahasa yang cerdas ini. Anda sudah memahami fondasi LLM, mampu mengintegrasikan data perusahaan melalui RAG, berhasil mengubah model generalis menjadi spesialis lewat fine-tuning, dan bahkan memberikan kemampuan untuk bertindak layaknya manusia menggunakan AI Agent.

Anda mungkin sedang menatap layar monitor sekarang, melihat output brilian yang dihasilkan oleh kode LLM dan merasa bangga. *It’s okay*, Anda pantas merasakannya.

Namun, sebuah mahakarya AI tidak akan memiliki nilai bisnis jika ia hanya mendekam di dalam Jupyter Notebook atau terminal localhost Anda. Ibaratnya seperti ilustrasi di bawah ini.

![AD_4nXckpnvyGzTvmiHI0k1v8KMbl6l0CtTCOx55a2dvfvhzDxOk69odgnNG7-srqfAwxkfCArr8PcbXcEJOytnlLz8Jlq6TTT8YICPnrUYDzRAbqOuis08u9XkIxBSDNxCq_iClqB_0RwF4zC93gM86I3lNNQ?key=VCE2UDhCUSjvRbWTbSQiig](https://assets.cdn.dicoding.com/original/academy/dos-f67fd3519084a847bd6793caab71948120260312134821.jpeg)

Otak super jenius yang terkurung di dalam toples kaca ini akan bungkam saat ada pertanyaan-pertanyaan mengenai bisnis.

*   Apa yang terjadi ketika aplikasi Anda tiba-tiba diakses oleh sepuluh ribu user secara bersamaan?;
*   Bagaimana jika latency response model memakan waktu terlalu lama?; atau
*   Bagaimana jika argo tagihan cloud Anda tiba-tiba meledak ratusan juta rupiah dalam semalam hanya karena arsitektur yang salah?

Semua pertanyaan tersebut punya satu benang merah, bukan? *Yup*, *user* dan uang keduanya ada di dalam satu konsep bernama bisnis. Jadi, sekali lagi selamat datang di *the endgame*, deployment.

Ketika membicarakan deploy dalam lingkup aplikasi web konvensional, itu mungkin hanya perkara memindahkan baris kode ke sebuah server dan menyambungkannya ke database. Namun, men-*deploy* aplikasi Generative AI menghadirkan "monster" yang rakus daya komputasi bernama LLM tadi.

Oleh karena itu, modul ini tidak dirancang sekadar untuk mengajari Anda cara menekan tombol *"Upload"* ke server. Kita akan belajar cara mengubah sebuah *prototype* menjadi aplikasi *enterprise-grade* yang tangguh, efisien, dan *scalable*.

Bagaimana jika dimulai dengan *blueprint*-nya terlebih dahulu agar kita mendapatkan gambaran struktur dari Aplikasi Generative AI? *Alright, let's go!* 

---

## The Blueprint of Aplikasi Gen AI

Banyak mungkin dari kita sering kali terjebak pada *euforia* kesederhanaan AI masa kini. Cukup dengan menulis beberapa baris kode Python, menempelkan API key ke dalam skrip, membuat *interface chat* sederhana, dan merasa tugas mereka sudah selesai. Memang, secara fungsional itu akan berjalan.

Namun pada skala production, aplikasi Generative AI bukanlah sekadar satu baris script Python yang memanggil model. Ia adalah sebuah ekosistem di mana berbagai komponen software harus berorkestrasi bersama.

Oleh karena itu, sebelum kita berdebat tentang *cloud provider* mana yang paling tangguh atau pusing memilih jenis infrastruktur GPU, kita wajib membedah cetak biru arsitekturnya terlebih dahulu. Bayangkan kita sedang berdiri di depan sebuah papan tulis besar, menggambar sebuah diagram blok sistem (*system design*).

### Anatomi Aplikasi Gen AI

Layaknya sebuah pabrik berteknologi tinggi, aplikasi Gen AI yang kokoh membutuhkan jalur masuk yang responsif, lapisan keamanan yang ketat, memori eksternal untuk memberikan konteks, mesin pemroses utama yang masif, hingga jalur pintas untuk menghemat biaya operasional. Setiap blok dalam diagram ini memiliki beban kerja teknis yang sangat spesifik untuk menjaga agar LLM Anda tetap aman, cepat, dan efisien.

![dos-caa0301ffd2eb03536f349366d3e094320260430105908.png](https://assets.cdn.dicoding.com/original/academy/dos-caa0301ffd2eb03536f349366d3e094320260430105908.png)

Mari kita telusuri lorong-lorong anatomi aplikasi ini satu per satu, mulai dari pintu terluarnya.

#### Frontend

Bisa dikatakan ini adalah ruang lingkup yang paling jarang kita sentuh dari pekerjaan kita sebagai Gen AI Engineer. Biasanya bahkan kita hanya menggunakan Gradio atau Streamlit untuk membuat interface sederhana. Namun, dalam level production, interface seadanya akan membawa kesan yang buruk ke pengguna karena di sinilah mereka pertama kali bersentuhan dengan kejeniusan AI Anda.

Masalahnya, merancang UI/UX untuk aplikasi Generative AI bukanlah sekadar membuat tombol yang menarik atau animasi yang *seamless*. Sebab, Anda sedang berhadapan dengan musuh utama arsitektur ini, yaitu Latensi Inferensi.

Dalam pengembangan web tradisional, kueri ke sebuah database biasanya memakan waktu hitungan milidetik, dan halaman web akan langsung memuat data tersebut secara utuh. *Yup*, kita bersama sudah paham bahwa Large Language Model tidak seringan kueri database minor seperti itu. LLM membutuhkan waktu komputasi yang intensif untuk memprediksi, memproses, dan merangkai probabilitas kata demi kata.

Sekarang coba kita ilustrasikan skenario aplikasi Generative AI Anda menggunakan arsitektur konvensional.

![dos-dfee85d33a45b535c196adb8bd70043920260504110945.gif](https://assets.cdn.dicoding.com/original/academy/dos-dfee85d33a45b535c196adb8bd70043920260504110945.gif)

*   User memasukkan prompt yang rumit untuk menulis ulang novel yang memiliki total 3.400 hingga 4.100 halaman untuk edisi bahasa Inggris.
*   User tersebut harus menatap animasi *loading* yang berulang-ulang selama 10 hingga 20 detik tanpa kepastian, menunggu seluruh rangkuman selesai dibuat di belakang layar.

Secara psikologis, di detik kelima User akan mulai gelisah. Kemudian di detik kesepuluh, User akan mengira aplikasi Anda *crash* dan menekan tombol *refresh*, yang pada akhirnya justru merusak proses dan membebani server dua kali lipat.

Di sinilah frontend engineering untuk Gen AI berubah menjadi sebuah seni memanipulasi persepsi waktu.

Jawabannya adalah teknik yang pernah kita gunakan dalam latihan fine-tuning sebelumnya, yaitu mekanisme Streaming Response. Pernahkah Anda menyadari mengapa ChatGPT, Gemini, atau Claude seolah-olah "mengetik" jawabannya secara real-time di layar? *Nah*, itu bukanlah sekadar efek visual kosmetik yang dibuat agar terlihat keren, melainkan adalah rekayasa UX untuk menutupi latensi.

Alih-alih menggunakan HTTP Request biasa, frontend aplikasi Gen AI memanfaatkan protokol koneksi persisten seperti Server-Sent Events (SSE) atau WebSockets.

Cara kerjanya kira-kira akan seperti ini.

*   **Token Perdana:** LLM di server telah memproses komputasi awalnya dan mengeluarkan satu kata atau "token" pertama. *Nah*, kecepatan lahirnya token pertama ini kita sebut sebagai *Time-to-First-Token* (TTFT).
*   **Tangkapan Instant Frontend:** Tanpa perlu menunggu kalimat tersebut utuh, lapisan frontend langsung mencegat dan menangkap token pertama tersebut detik itu juga.
*   **Rendering Milidetik:** Serpihan kata tersebut langsung di-*render* dan dimunculkan di layar perangkat pengguna.
*   **Aliran Tanpa Henti:** Proses tangkap-dan-tampilkan ini terus berulang dengan ritme yang sangat cepat hingga seluruh pemikiran LLM selesai dieksekusi.

Hasilnya, meskipun total waktu yang dibutuhkan mesin untuk menjawab tetaplah 15 detik, pengguna tidak akan merasa menunggu selama itu. Mata mereka sudah sibuk membaca kata-kata pertama yang muncul.

Sekarang sisa satu misteri yang belum terpecahkan, antara WebSockets dan SSE mana yang lebih baik? Mari kita cari tahu bersama.

![dos-fd8e8ee3b4e4ab2f6845f097f80d5ba520260430110132.gif](https://assets.cdn.dicoding.com/original/academy/dos-fd8e8ee3b4e4ab2f6845f097f80d5ba520260430110132.gif)

*   **WebSockets**
    Mengubah koneksi HTTP menjadi terowongan komunikasi dua arah (full-duplex), di mana klien dan server bisa saling melempar pesan kapan saja tanpa henti. Namun, menggunakan WebSockets untuk interaksi LLM ibarat menyewa jalan tol dua arah secara eksklusif hanya untuk dilewati satu mobil pengantar paket. User hanya bertanya sekali di awal, lalu sisa waktunya dihabiskan oleh server yang memuntahkan jawaban panjang. Mempertahankan koneksi dua arah yang persisten sangat boros memori di level server. Selain itu, mengelola WebSockets melintasi Load Balancer dan Firewall perusahaan sering kali menjadi mimpi buruk bagi tim DevOps.
*   **Server-Sent Events (SSE)**
    Inilah teknologi yang berada di balik kehalusan antarmuka ChatGPT, Claude, dan aplikasi AI enterprise lainnya. Berbeda dengan WebSockets, SSE bersifat *uni-directional*(satu arah) dengan cara kerja seperti berikut.
    *   Pengguna mengirimkan pertanyaan awal melalui HTTP Request standar.
    *   Server merespons, tapi alih-alih menutup koneksi, ia menetapkan tipe konten sebagai text/event-stream.
    *   Koneksi dipertahankan tetap terbuka (hanya searah dari server ke klien), sehingga server leluasa mengirimkan rentetan "Response update" (berupa serpihan token) layaknya air yang mengalir.
    *   Begitu token terakhir ("stop token") terkirim, server secara elegan menutup koneksinya.

Mengapa SSE menang mutlak? Karena ia jauh lebih ringan. Ia berjalan murni di atas protokol HTTP konvensional, sehingga sangat mudah di-*deploy*, bersahabat dengan firewall paling ketat sekalipun, dan secara otomatis mendukung fitur-fitur bawaan HTTP.

Kita sudah melihat bagaimana frontend bekerja keras memanipulasi persepsi waktu pengguna dengan efek streaming. Sekarang, mari kita membuka pintu tebal menuju ruang mesin utama kita, yaitu Backend.

#### Backend

Secara teknis, backend adalah lapisan perantara (middleware/proxy) yang berdiri di antara pengguna dan model AI Anda. Nah, sekarang bayangkan kita langsung membiarkan frontend berkomunikasi langsung dengan Large Language Model.

Jika menggunakan pendekatan API-Based dengan memanggil model langsung dari front-end, Anda akan mengekspos API Key rahasia di dalam kode client. Siapa pun dengan sedikit pengetahuan teknis bisa menekan klik kanan, memilih Inspect Element di browser, mencuri kunci tersebut, dan menggunakannya untuk menumpang komputasi AI secara gratis.

Di sinilah backend bekerja sebagai API Endpoint Wrapper dengan mengamankan dua hal sekaligus.

*   **Keamanan Credential**
    Backend membungkus seluruh mesin AI Anda dan menyembunyikannya dari dunia luar. Sebagai gantinya, ia mengekspos sebuah interface API yang biasanya berupa endpoint REST seperti /api/v1/chat untuk dikonsumsi oleh frontend.
*   **Perakitan Prompt secara Rahasia**
    Ketika user mengetik *"Tolong rangkum dokumen ini"*, backend akan menyisipkan System Prompt rahasia perusahaan Anda terlebih dahulu. Barulah setelah itu, “paket” prompt utuh dan kaya konteks tersebut diserahkan ke LLM. User tidak akan pernah tahu instruksi rahasia apa yang disisipkan di balik layar.

Setelah dinding pertahanan API Wrapper berdiri kokoh untuk melindungi model dari dunia luar, kita kini dihadapkan pada satu ilusi model proprietary. Saat kita berinteraksi dengan ChatGPT, model tersebut terasa sangat manusiawi karena seolah-olah mengingat setiap detail percakapan dari awal hingga akhir obrolan.

Sayangnya, base model LLM yang selama ini kita load dari Hugging Face dan digunakan untuk inference itu tidak memiliki ingatan sama sekali atau bisa kita katakan Stateless. Lalu, bagaimana cara kerja aplikasi Gen AI kelas dunia bisa memiliki ingatan?

Karena LLM terlahir stateless, tentu saja backend-lah yang akan bekerja untuk mengubah interaksi tersebut menjadi Stateful (memiliki status memori yang berkelanjutan). Di dunia industri, kita perlu menggunakan library tambahan seperti fitur Memory pada LangChain atau LlamaIndex dan menyambungkannya ke database berkecepatan tinggi seperti Redis, PostgreSQL, atau MongoDB.

*Nah*, mari kita lihat apa yang sebenarnya dilakukan oleh backend untuk membuat percakapan ini stateful.

*   **Session ID:** Saat User membuat sesi chat baru, backend membuatkan satu "buku harian" digital kosong khusus untuk sesi tersebut yang ditandai dengan ID unik.
*   **Ambil Memory:** Ketika pengguna mengetik pertanyaan baru, backend akan pergi ke database menggunakan Session ID tersebut, lalu menyalin seluruh riwayat percakapan yang pernah terjadi sebelumnya.
*   **Perakitan Prompt:** Backend kemudian menambahkan riwayat percakapan masa lalu tersebut dengan pertanyaan yang baru menjadi satu paket prompt yang akan dikirimkan ke LLM.
*   **Pencatatan Riwayat Baru:** Begitu LLM selesai menjawab, backend akan mengambil jawaban terbaru itu dan menuliskannya kembali ke "buku harian" di database agar siap digunakan untuk pertanyaan berikutnya.

*Nah*, fungsi terakhir ini yang akan membuat Dompet dan Server kita tentram, yaitu Traffic Control.

Bayangkan sebuah skenario di mana seorang kompetitor menjalankan script sederhana untuk mengirimkan 10.000 pertanyaan per detik ke aplikasi Generative AI milik Anda. Apa yang terjadi?

Jika Anda menggunakan Self-hosted LLM, antrean komputasi akan menumpuk, VRAM GPU akan mengeluarkan notifikasi yang sudah familier dengan kita, yaitu Out of Memory, dan dampaknya server akan mati total. Jika Anda menggunakan API berbayar seperti OpenAI, server Anda mungkin selamat, tapi argo tagihan Anda akan berputar gila-gilaan menembus ratusan juta rupiah dalam satu malam.

*Nah*, fungsi Throttling atau yang sering disebut Rate Limiting hadir untuk memitigasi skenario tersebut.

Backend akan mencatat setiap IP Address atau akun User yang masuk dan menetapkan batas kecepatan interaksi, misalnya maksimal 20 pertanyaan per menit per pengguna. Jika batas ini terlewati, backend akan langsung memblokir request dan mengembalikan status HTTP 429 (*Too Many Requests*).

![dos-030bce190ee03a454111924bebaa2bbf20260504111022.gif](https://assets.cdn.dicoding.com/original/academy/dos-030bce190ee03a454111924bebaa2bbf20260504111022.gif)

Request berlebih ini langsung dibuang ke tempat sampah oleh *backend* sehingga mesin LLM di belakangnya sama sekali tidak terganggu dan menjamin beberapa hal berikut.

*   **Stability:** Server tetap hidup meski trafik melonjak tiba-tiba.
*   **Fairness:** Mencegah satu pengguna memonopoli seluruh bandwidth.
*   **Security:** Menghambat serangan *brute force* (tebak password) yang mencoba ribuan kombinasi per detik.

*Alright*, sampai di sini, kita sadar bahwa backend memang tidak membuat aplikasi Anda lebih pintar, tetapi ia membuatnya bertahan hidup di kerasnya internet.

Namun, ada satu kepingan puzzle terakhir dari anatomi ini yang membedakan antara aplikasi amatir yang boros biaya dengan aplikasi enterprise yang sangat efisien. Kepingan itu adalah Caching Layer.

#### Caching Layer: Jalan Pintas Eksekusi dan Penyelamat Anggaran

Mari kita kembali ke realitas bisnis. Anggaplah Anda berhasil men-*deploy* sebuah AI Agent sebagai Customer Service internal perusahaan. Di hari pertama peluncuran, aplikasi Anda viral di kalangan karyawan. Tiba-tiba, ada 500 orang yang menanyakan pertanyaan yang persis sama: "Bagaimana prosedur klaim kacamata tahun ini?"

Jika Anda mengandalkan arsitektur naif tanpa Caching Layer, apa yang terjadi? Backend Anda akan dengan patuh mengirimkan pertanyaan tersebut ke LLM sebanyak 500 kali. Mesin probabilitas Anda akan bekerja keras, membakar ribuan komputasi matriks di dalam GPU, menyedot memori, dan memakan waktu berdetik-detik hanya untuk men-*generate* jawaban yang sama persis sebanyak 500 kali. Ini bukan sekadar tidak efisien; ini adalah pemborosan sumber daya yang brutal.

Di sinilah Caching Layer hadir sebagai pahlawan tanpa tanda jasa.

Secara teknis, komponen ini adalah sebuah sistem penyimpanan data di dalam memori utama (RAM), bukan di hard drive. Industri umumnya menggunakan teknologi *in-memory datastore* yang sangat cepat seperti Redis atau Memcached. Kecepatannya tidak diukur dalam detik, melainkan hitungan milidetik (*sub-millisecond latency*).

Logika kerjanya di dalam sistem sangatlah brilian dan egois (dalam artian positif). Ketika backend menerima sebuah prompt dari pengguna, sebelum ia repot-repot membangunkan LLM, ia akan mampir dulu ke Caching Layer. Ia akan mengecek: "Hei, apakah sebelumnya sudah ada yang bertanya dengan prompt persis seperti ini?"

*   **Skenario Cache Hit**
    Jika ternyata pertanyaannya sama persis (misalnya pertanyaan klaim kacamata tadi sudah pernah ditanyakan oleh karyawan pertama), Caching Layer akan berteriak, "Ada! Ini jawabannya!". Backend akan langsung mengambil teks jawaban tersebut dari memori dan melemparkannya kembali ke frontend. Hasilnya: Pengguna mendapatkan jawaban secara instan (0,01 detik), LLM sama sekali tidak disentuh, tidak ada antrean komputasi di GPU, dan biaya token Anda untuk 499 karyawan berikutnya adalah nol rupiah.
*   **Skenario Cache Miss**
    Jika pertanyaan tersebut benar-benar baru, barulah backend merelakannya untuk diproses oleh LLM. Namun, begitu LLM selesai memuntahkan jawaban yang susah payah di- generate tersebut, backend akan menyalinnya dan menyimpannya ke dalam Caching Layer terlebih dahulu sebelum diberikan ke pengguna. Tujuannya? Agar jika besok ada yang menanyakan hal serupa, sistem sudah siap dengan jalan pintasnya.

Dalam arsitektur enterprise, Caching Layer ini bahkan bisa di- upgrade menggunakan teknik Semantic Caching. Alih-alih mencari prompt yang sama persis kata demi kata, sistem akan mengubah prompt menjadi embedding vector dan mencari pertanyaan yang "maknanya mirip". Jadi, pertanyaan "Cara klaim kacamata?" dan "Gimana prosedur *reimburse* kacamata?" akan dianggap sama dan dilayani langsung dari *cache*.

Komponen-komponen tadi adalah kerangka tubuh dari aplikasi Generative AI Anda. Kerangka ini sudah tersusun rapi di atas kertas desain sistem Anda. Namun, kerangka raksasa ini tidak akan bernyawa tanpa kita membuat satu keputusan paling fundamental: di mana sang "otak" utama akan diletakkan?

*Nah*, itulah yang akan menjadi topik bahasan kita berikutnya.

### Dua Jalan Ninja: API-Based vs Self-Hosted LLM

Selamat datang di persimpangan jalan terbesar dalam karier Anda sebagai arsitek AI.

Di industri saat ini, menghidupkan sebuah LLM di level production bukanlah sekadar perkara menekan tombol run di terminal komputer Anda. Ada dua mazhab utama—mari kita sebut sebagai "Dua Jalan Ninja"—yang harus Anda pilih. Keputusan di persimpangan ini tidak hanya menentukan arsitektur teknis Anda, tetapi juga berdampak langsung pada anggaran bulanan, kecepatan rilis ke pasar (time-to-market), hingga seberapa nyenyak tim legal perusahaan Anda bisa tidur di malam hari.

Sebagai gambaran awal, Anda dapat memperhatikan ilustrasi berikut ini.

![dos-6179b5137b3c90fdcfdfeb11788fbbc220260504111434.png](https://assets.cdn.dicoding.com/original/academy/dos-6179b5137b3c90fdcfdfeb11788fbbc220260504111434.png)

Sekarang, mari kita bedah filosofi, keindahan, dan sisi gelap dari kedua jalan ini.

#### Jalan Ninja Pertama: Sang Aristokrat Cloud (Pendekatan API-Based)

Pendekatan ini ibarat Anda menyewa sebuah *penthouse* super mewah di pusat kota yang sudah *full-furnished*, lengkap dengan pelayan pribadi. Anda tidak perlu tahu bagaimana jaringan pipa air dipasang, atau dari mana aliran listrik berasal. Anda cukup masuk, duduk di sofa empuk, dan menikmati pemandangannya.

Dalam dunia Generative AI, pendekatan API-Based berarti Anda meminjam "otak" dari model-model komersial raksasa—seperti OpenAI (Keluarga GPT), Google (Gemini), atau Anthropic (Claude)—melalui sebuah gerbang Application Programming Interface (API). Tim Anda murni berfokus pada rekayasa prompt dan logika bisnis, sementara urusan mesin diserahkan kepada sang raksasa teknologi.

Daya tarik utama dari jalan ini adalah kecepatan absolut (*Hyper-fast Time-to-Market*). Anda tidak perlu ikut antre berbulan-bulan demi membeli GPU kelas enterprise yang harganya bisa seharga mobil sport dan ketersediaannya sering kali "gaib" di pasaran. Anda terbebas dari mimpi buruk melakukan konfigurasi driver CUDA, pusing mengatur alokasi VRAM, atau memikirkan suhu server.

Cukup dapatkan API Key, tulis beberapa baris kode di backend Anda untuk mengirim request HTTP, dan boom! Aplikasi Anda tiba-tiba memiliki kecerdasan sekelas Ph.D. Tim kecil beranggotakan dua orang pun bisa merilis aplikasi Gen AI kelas dunia dalam hitungan hari.

Selain itu, model finansialnya sangat seksi bagi startup atau proyek baru: *Pay-as-you-go*. Anda mengubah pengeluaran modal besar (Beli Server/GPU) menjadi pengeluaran operasional (Beli Token). Anda murni hanya membayar sesuai jumlah kata (token) yang dikunyah dan dimuntahkan oleh model. Jika di jam 3 pagi aplikasi Anda sepi pengunjung, biaya infrastruktur LLM Anda adalah nol rupiah.

**Sisi Gelap dan Harga Sebuah Kenyamanan**

*   **Vendor Lock-in**
    Menyewa rumah orang lain berarti Anda harus tunduk pada aturan main sang pemilik tanah. Sisi gelap pertama adalah ketergantungan absolut (*Vendor Lock-in*). Nyawa aplikasi Anda sepenuhnya berada di tangan pihak ketiga. Jika server API mereka mengalami *downtime* atau *latency* yang parah karena sedang melayani jutaan pengguna global lainnya, aplikasi Anda akan ikut lumpuh terbawa arus, dan Anda tidak bisa berbuat apa-apa selain berdoa menatap *dashboard status page* mereka.
*   **Privasi dan Keamanan Data**
    Sisi gelap yang lebih menakutkan dan sering kali menjadi pembunuh kesepakatan (dealbreaker) bagi perusahaan enterprise adalah **privasi** dan **keamanan data**. Saat Anda menggunakan pendekatan API-Based, setiap prompt, setiap dokumen yang ditarik dari Database, dan setiap pertanyaan dari user Anda akan dikirimkan melintasi internet menuju server pihak ketiga. Jika Anda sedang membangun asisten AI untuk dokter yang memproses rekam medis pasien atau AI untuk bank yang menganalisis portofolio nasabah VIP, mengirim data sensitif ini ke luar dari jaringan perusahaan adalah sebuah "dosa besar" yang akan membuat tim Compliance dan Legal Anda jantungan. Ada risiko kebocoran data dan Anda terbentur regulasi ketat soal kedaulatan data.
*   **Pedang Bermata Dua (**_**pay-as-you-go)**_
    *Pay-as-you-go* memang murah di awal. Namun, bayangkan jika aplikasi Anda meledak sukses dan digunakan oleh jutaan pengguna setiap harinya, pasti biaya token akan meroket secara linier. Yang awalnya hanya tagihan ratusan ribu rupiah per bulan, bisa membengkak menjadi miliaran rupiah tanpa Anda sadari sehingga membuat CFO berpikir keras.

Pendekatan API-Based adalah jalan pintas yang brilian untuk kecepatan, validasi ide, dan produk consumer-facing yang tidak berurusan dengan data rahasia tingkat tinggi. Namun, jika bisnis Anda menuntut privasi absolut dan kontrol penuh, Anda tidak bisa bergantung pada "otak" sewaan.

#### Jalan Ninja Kedua: Membangun Benteng Sendiri (Self-Hosted LLM)

Jika privasi data pelanggan Anda adalah harga mati, atau jika tagihan API Anda sudah mulai mencekik napas finansial perusahaan, Anda tidak punya pilihan lain selain menempuh jalan ninja yang kedua ini.

Pendekatan Self-Hosted ibarat Anda membeli sebidang tanah, menyewa arsitek, mengecor fondasi, dan membangun benteng pertahanan Anda sendiri dari nol. Di jalur ini, Anda mengambil "otak" buatan komunitas open-weights kelas dunia—seperti Llama 3 dari Meta, Mistral dari Eropa, atau Qwen—lalu membiarkannya hidup dan bernapas di dalam server atau infrastruktur cloud (VPC) yang kuncinya hanya dipegang oleh Anda.

Daya tarik paling sensual dari jalan ini adalah satu kata: Kontrol.

Bagi perusahaan berskala enterprise, perbankan, atau institusi kesehatan, Self-Hosted adalah satu-satunya jalan menuju ketenangan batin. Karena model berjalan sepenuhnya di dalam lingkungan terisolasi milik Anda sendiri, tidak ada satu pun byte data sensitif yang akan mengintip keluar tembok perusahaan. Semua prompt pengguna, data finansial, hingga rekam medis yang ditarik dari Vector Database, diproses dan dihancurkan di dalam rumah sendiri. Tidak ada data yang "disetor" ke perusahaan AI pihak ketiga. Tim legal dan compliance Anda akhirnya bisa tidur nyenyak.

Selain itu, di sinilah panggung utama bagi keringat dan air mata Anda di Modul 3. Jika Anda sudah susah payah melakukan Fine-Tuning pada sebuah model dasar agar ia berbicara persis seperti representasi brand Anda, atau agar ia menguasai ribuan jargon teknis internal pabrik Anda, Anda harus menggunakan metode Self-Hosted. Model komersial API biasanya tidak memberikan kebebasan modifikasi sedalam ini. Saat Anda melakukan Self-Hosted, Anda benar-benar memiliki otak tersebut.

Dari sisi finansial jangka panjang, Self-Hosted menukar biaya variabel yang tak tertebak dengan biaya tetap (*fixed cost*). Jika aplikasi Anda sukses besar dan melayani jutaan request per hari, tagihan server Anda akan tetap sama. Anda tidak lagi dihantui oleh argo biaya per token yang terus berputar setiap kali AI Anda memuntahkan kata.

Namun, mari bicara realita. Membangun dan menjaga benteng ini membutuhkan nyali dan keahlian rekayasa tingkat tinggi. Anda tidak lagi sekadar memanggil API; Anda sekarang berurusan dengan "besi dan silikon". Alasan tersebut membuatnya memiliki enam tantangan utama.

*   **Hardware**
    LLM sangat lapar akan Graphic Processing Unit (GPU). Anda harus menyewa atau membeli GPU kelas data center seperti NVIDIA A100 atau H100. Masalahnya, barang ini harganya sangat fantastis (bisa mencapai ratusan juta hingga miliaran rupiah per unit) dan ketersediaannya di pasaran sering kali "gaib" alias diperebutkan oleh seluruh perusahaan teknologi di dunia.
*   **Ops Burden**
    Jika server GPU Anda mati, Andalah yang harus bangun jam 3 pagi untuk memperbaikinya. Tim Anda dituntut untuk menguasai manajemen infrastruktur yang rumit. Mengelola memori VRAM dari sebuah LLM adalah seni tersendiri. Jika puluhan user bertanya secara bersamaan dan Anda salah melakukan konfigurasi, VRAM akan penuh, aplikasi Anda akan terkena penyakit mematikan bernama Out of Memory (OOM), dan server akan *crash* seketika.
*   **Engineering Overwhelm**
    Engineer tidak hanya dituntut untuk menjaga server tetap hidup, tetapi kita juga harus terus belajar teknik baru setiap hari. Hal tersebut karena aplikasi yang dianggap “terbaik” minggu lalu bisa jadi usang dalam waktu yang singkat. Tekanan untuk terus mengikuti tren dan juga beban operasional sangat berisiko membuat tim merasa burnout.
*   **Scalability Nightmare**
    Ketika kita menggunakan API, jika pengguna aplikasi melonjak dari 100 menjadi 10.000 atau lebih dalam sehari, kita hanya tinggal diam dan membayar tagihan API. So simple. Namun, pada self-host, scaling server GPU adalah situasi yang sangat mencekam. Kita tidak bisa sembarangan melakukan auto-scaling seperti pada web server biasa karena membutuhkan infrastructure yang besar.
*   **The Hidden Cost**
    *Nah*, banyak yang mengira kalau pengeluaran self-hosting berhenti setelah berhasil membeli GPU yang mahal. Kenyataannya, itu baru tiket masuknya saja lho. Kita akan disambut oleh tagihan listrik yang dinamis juga karena GPU yang menyala 24/7 membutuhkan pendingin yang menyala terus menerus. 
*   **Data Privacy Burden**
    Alasan utama banyak orang memilih self-host adalah agar data tidak bocor keluar. Ironisnya, mengamankan model LLM di server sendiri juga sangat sulit. Tanpa proteksi seperti filter dari penyedia API (misalnya guardrails dari OpenAI atau Anthropic), model self-hosted Anda sangat rentan terkena serangan Prompt Injection. Pengguna iseng bisa memanipulasi LLM Anda untuk membocorkan system prompt rahasia, mengucapkan kata-kata kasar, atau bahkan memberikan akses ke database internal jika tidak diisolasi dengan benar.

Berbeda dengan API yang biayanya nol jika tidak dipakai, di jalur Self-Hosted, entah aplikasi Anda sedang sibuk melayani 10.000 user atau sedang sepi seperti kuburan, Anda tetap harus membayar biaya sewa server GPU yang menyala 24/7 tersebut.

Memilih antara *API-Based* dan *Self-Hosted* adalah fondasi dari seluruh arsitektur Anda. Apakah Anda memilih kecepatan dan ilusi kemudahan atau berani membayar harga sebuah kemerdekaan dan privasi data?

Setelah Anda menetapkan pilihan, pertanyaan berikutnya yang harus dijawab adalah di atas tanah apa benteng ini akan dibangun?

---

## Merakit Infrastruktur: Deployment LLM Skala Produksi

Semua aplikasi entah itu berbasis web, mobile app, menggunakan AI atau tidak sama sekali, pada akhirnya hanya bisa hidup dan digunakan jika ada mesin untuk menjalankannya.

Pada modul-modul sebelumnya, “mesin” itu adalah laptop atau komputer lokal Anda. Namun, agar keajaiban yang Anda bangun ini bisa diakses oleh ratusan atau ribuan User di luar sana secara bersamaan, kita harus memindahkannya ke "mesin" yang sesungguhnya di tahap production.

Di sinilah kita memasuki fase yang paling menantang sekaligus krusial, yaitu **infrastruktur untuk deployment**.

Perlu kita pahami, saat men-*deploy* aplikasi web tradisional biasanya *step-by-step*-nya cukup *straight to the point* dengan menyewa server standar, memastikan CPU dan RAM cukup, lalu aplikasi berjalan. Selesai.

*Nah*, men-*deploy* aplikasi berbasis LLM adalah cerita yang berbeda dengan menghadirkan sebuah tantangan bernama komputasi kelas berat. Kita tidak lagi sekadar berurusan dengan CPU melainkan GPU, manajemen VRAM, latensi inferensi, dan ukuran file model yang bisa mencapai puluhan gigabyte.

Perjalanan kita dalam submodul ini akan cukup panjang. Oleh karena itu, mari kita mulai sekarang.

### Estimasi Kebutuhan VRAM untuk Production

Sebelum mempelajari lebih lanjut berbagai macam infrastruktur dan server, kita membutuhkan guideline terlebih dahulu untuk menghitung kebutuhan memori infrastruktur.

Mari kita ingat kembali sejenak apa yang sudah kita bahas di modul Fine-Tuning. Saat itu, kita belajar bahwa untuk melatih model, kita membutuhkan VRAM yang cukup banyak karena GPU harus menampung Model Weights, Gradients, Optimizer States, dan Activations.

*Nah*, kabar baiknya pada tahap Deployment atau Inference, beban kerjanya jauh lebih ringan. Kita tidak lagi melakukan training, tidak ada proses backpropagation sehingga Gradients dan Optimizer States bisa kita abaikan dari perhitungan VRAM kali ini.

*Eits*, bukan berarti kita hanya akan menyimpan model dengan rumus 2 byte setiap parameter-nya saja. Pada tahap production, ada "penyewa" baru di dalam VRAM kita yang ukurannya bisa membengkak di luar kendali jika tidak kita hitung dengan hati-hati. Hal tersebut selaras dengan ilustrasi rumus perhitungan VRAM berikut.

![dos-2a0ee0b3c213c86c92dd2cb88428611920260430162112.png](https://assets.cdn.dicoding.com/original/academy/dos-2a0ee0b3c213c86c92dd2cb88428611920260430162112.png)

*Yup*, total kebutuhan VRAM untuk menjalankan LLM di production terbagi menjadi tiga komponen utama tersebut. Mari kita bedah satu per satu.

#### Model Weights (Bobot Model)

*Nah*, ini dia "penyewa" tetap di VRAM Anda. Seperti yang sudah Anda pelajari di modul Fine-Tuning, menghitung bobot model cukup sederhana asal kita tahu presisi yang digunakan. By default, presisi yang sering digunakan untuk model saat ini adalah FP16 atau BF16 (16-bit), yang berarti setiap parameternya berukuran 2 byte.

Oleh karena kita sudah pernah menghitung ukuran model Llama 3.1 8B, akan menarik jika kita menghitung model lainnya seperti [Nemotron-Terminal-8B](http://nvidia/Nemotron-Terminal-8B).

*   Jumlah parameter: 8 miliar
*   Presisi: 16-bit (2 Byte)
*   Kebutuhan VRAM Statis: 8.000.000.000 2 bytes 16 GB

Jika Anda hanya melihat angka 16 GB ini, Anda mungkin akan berpikir, *"Ah, cukup sewa server dengan GPU tunggal berkapasitas 16GB (seperti RTX 3090) dan semuanya akan aman."* *Eits*, di sanalah jebakan deployment yang paling sering menelan korban pemula karena melupakan memori dinamisnya yang bernama KV Cache.

#### KV Cache

Seperti yang dijelaskan ilustrasi sebelumnya, KV Cache atau Key-Value Cache adalah mekanisme memori jangka pendek LLM.

Sekarang bayangkan Anda sedang membaca buku tebal. Saat sampai di halaman 10, Anda tidak perlu membaca ulang dari halaman 1 untuk memahami kalimat berikutnya, bukan? Sebab Anda sudah "mengingat" konteks dari halaman 1 tersebut.

Sayangnya, arsitektur Transformer tidak punya ingatan seperti itu *by default*. Untuk mengakalinya, kita menyimpan hasil perhitungan matriks "Key" dan "Value" dari token-token sebelumnya di dalam VRAM. *Wait*, mengapa menyimpan “Key” dan “Value” itu menjadi solusinya?

Alasan utamanya terletak pada cara mekanisme Attention bekerja, di mana setiap token baru perlu "melihat kembali" ke token sebelumnya untuk memahami konteks. Mari kita coba *breakdown* sembari melihat rumus dari Attention berikut ini.

![dos-bc423c4ee976b469cf49a0a5bc59d44820260430162236.png](https://assets.cdn.dicoding.com/original/academy/dos-bc423c4ee976b469cf49a0a5bc59d44820260430162236.png)

*   **Query (Q) bersifat dinamis:** Saat menebak kata ke-100, Q hanya berasal dari token terakhir yang berarti kata ke-99. Di sana kita ingin tahu *"Apa yang dicari oleh kata ke-99 ini dari kata-kata sebelumnya?"*.
*   **K dan V bersifat statis:** Representasi K (Key) dan V (Value) untuk kata ke-1 hingga ke-98 tidak berubah meskipun ada kata baru yang muncul.
*   **Menghindari Redundansi:** Jika kita tidak menyimpan K dan V dari kata 1-98, model terpaksa melakukan perkalian matriks linear yang sama berulang kali hanya untuk mendapatkan angka yang sebenarnya sudah pernah dihitung pada langkah sebelumnya.

Dengan menyimpan K dan V di VRAM, model cukup menghitung Q untuk token terbaru, lalu langsung melakukan operasi dot product dengan "arsip" K dan V yang sudah ada. Inilah yang membuat proses generate teks menjadi jauh lebih cepat.

*Nah*, “arsip” K dan V inilah yang kita sebut sebagai KV Cache. Namun, dari sini kita sadar satu hal, yaitu ukuran KV Cache akan membengkak seiring dengan panjangnya percakapan (Sequence Length) dan jumlah User yang mengakses bersamaan (Batch Size).

Rumus teknis untuk menghitung KV Cache tiap satu token per satu pengguna sebenarnya cukup rumit karena bergantung pada arsitektur internal model (jumlah layer, attention heads, dll). Namun, kita bisa menggunakan rumus matematis sederhana ini untuk menghitung KV Cache per token.

| *2 (Key + Value) × Layers × KV_Heads × Head_Dim × Precision (Bytes)* |
| :--- |

Secara kasar, menyimpan KV Cache untuk 1 token di Llama 3 membutuhkan sekitar 0.5 MB. Terlihat cukup besar bukan untuk satu token saja?

Tenang saja, model modern seperti Llama 3 biasanya sudah menggunakan Grouped Query Attention (GQA) yang memangkas kebutuhan memori KV Cache hingga 4x lipat tanpa mengorbankan akurasi secara signifikan. Sehingga, ukurannya menjadi jauh lebih kecil sekitar 0.13 MB (128 KB) per token untuk presisi standar FP16.

*   Jika 1 User mengirim prompt dan meminta respons dengan total panjang 4.096 token, 1 User memakan tambahan VRAM sekitar 500 MB.
*   Jika ada 10 User yang mengakses secara bersamaan dengan panjang konteks yang sama, KV Cache Anda membengkak menjadi 5 GB.

*Nah*, GPU RTX 3090 yang Anda pikir aman sebelumnya, sekarang langsung mengalami *Out of Memory* (OOM).

#### Overhead (Jatah Preman Server)

Terakhir, Anda perlu sadar bahwa GPU tidak hanya hidup untuk model AI itu sendiri. Selayaknya sebuah sistem operasi, GPU butuh ruang untuk bernapas untuk dialokasikan ke beberapa komponen lain seperti CUDA Context, memori operasi, dan berbagai fungsi low level lainnya yang sangat banyak. *Rule of thumb* yang wajib Anda pegang di industri adalah *“Selalu sisakan 10% hingga 20% VRAM (sekitar 1 - 2 GB) sebagai Buffer atau Overhead”*.

Dalam ekosistem produksi seperti vLLM, parameter gpu_memory_utilization bahkan secara default diset ke 0.90 (menyisakan 10%). Itu karena mengisi VRAM hingga 100%, hampir pasti akan mematikan proses saat ada lonjakan aktivasi sesaat.

![dos-7e892899c394372fcce1cda287d7561420260504111535.gif](https://assets.cdn.dicoding.com/original/academy/dos-7e892899c394372fcce1cda287d7561420260504111535.gif)

*Nah*, dalam ilustrasi di atas, terjadi spike dalam memori yang kita gunakan. Spike yang tiba-tiba bertambah ini karena selama proses forward pass, GPU harus menyimpan output sementara dari setiap layer sebelum dikirim ke layer berikutnya. Meskipun memori ini bersifat *transient* atau muncul dan hilang dalam milidetik, tetap saja dapat menciptakan lonjakan penggunaan memori yang tidak terhitung dalam rumus statis KV Cache.

Dari estimasi di atas, kita kini tahu bahwa menyewa satu GPU dengan ukuran yang sangat “ngepas” misalnya 16 GB pada RTX 3090 tidak akan cukup untuk production yang serius. Nantinya Anda akan dihadapkan pada dua pilihan desain arsitektur.

*   Menyewa GPU kelas enterprise dengan VRAM 40GB atau 80GB (seperti NVIDIA A100)
*   Menurunkan bobot model menggunakan teknik Quantization agar muat di GPU yang lebih murah atau bahkan membatasi jumlah paralel atau token dalam satu kali inferensi.

*Nah*, setelah kita punya angka kebutuhannya di kepala, pertanyaan berikutnya adalah *“Di mana kita akan menyewa atau membeli GPU tersebut?”*. *So*, mari kita bergeser ke pembahasan antara *Cloud-Native* vs *On-Premise*.

### Cloud-Native vs On-Premise

Kalkulator VRAM yang kita bahas sebelumnya sudah bisa disimpan sejenak. Dari sesi hitung-hitungan matematika sebelumnya, katakanlah kita sudah sampai pada satu kesimpulan pahit *"Aplikasi saya ternyata butuh minimal 80GB VRAM agar tidak crash saat diakses 50 orang bersamaan."* 

Sekarang pertanyaan yang perlu kita jawab adalah *“Dari mana kita mendapatkan infrastruktur komputasi kelas berat tersebut?”* 

Di dunia pengembangan software seperti membuat web atau mobile app biasa, perdebatan antara Cloud dan On-Premise mungkin sudah tidak menjadi topik yang *happening*. Pendekatan cloud hampir selalu menjadi pemenang mutlak karena kemudahan yang ditawarkan.

Namun, seperti yang sudah kita bahas sebelumnya, dunia Generative AI memiliki gameplay yang berbeda. Kebutuhan komputasi yang masif, kelangkaan hardware global, dan isu sensitivitas data membuat perdebatan ini hidup kembali dan jauh lebih sengit.

![dos-34eff12e242d445e7526e9bea70fde0020260430162400.gif](https://assets.cdn.dicoding.com/original/academy/dos-34eff12e242d445e7526e9bea70fde0020260430162400.gif)

Memilih infrastruktur AI bukan lagi sekadar keputusan teknis dari tim IT, melainkan keputusan bisnis strategis yang melibatkan Chief Financial Officer (CFO) hingga divisi Legal.

Di submodul ini, kita akan berdiri di persimpangan jalan dan membedah dua kutub utama infrastruktur deployment:

*   **Jalur Pertama:** Pendekatan Cloud yang merupakan rute ekspres. Anda tidak perlu memikirkan kabel atau tagihan listrik.
*   **Jalur Kedua:** Pendekatan On-Premise dengan membangun data center AI milik Anda sendiri. Terdengar repot? Memang. Namun, kita akan mengupas tuntas alasan banyak enterprise akhirnya berbelok ke jalur ini.

*Alright*, mari kita *breakdown* dengan lebih dalam kedua jalur ini.

#### Pendekatan Cloud-Native

Menggunakan layanan Cloud untuk mendeploy aplikasi AI itu ibarat kita menyewa apartemen mewah yang sudah *fully furnished*. Satu-satunya hal yang perlu Anda lakukan adalah membawa “koper” berisi kode dan model AI Anda. Selebihnya Anda dapat masuk ke dalam kamar dan semuanya sudah tersedia. Listrik menyala 24 jam, pendingin ruangan sudah diatur suhu idealnya, dan ada satpam yang menjaga di depan.

*Yup*, kelebihan utamanya sangat jelas, yaitu kecepatan dan kemudahan. Anda bisa mendapatkan infrastruktur super komputer senilai miliaran rupiah hanya dalam hitungan menit tanpa perlu mengeluarkan modal awal. Anda hanya membayar apa yang Anda gunakan per jam atau per detik.

Di ranah komputasi AI kelas berat ini, ada dua raksasa utama yang menawarkan pendekatan dan "senjata" yang berbeda. Mari kita bedah satu per satu.

##### Google Cloud Platform (GCP)

Saat seluruh dunia sedang panik dan antre panjang untuk membeli GPU buatan NVIDIA, Google bisa dibilang cukup “santai”. Hal ini karena mereka memiliki "senjata rahasia" yang dibangun sendiri bernama TPU (Tensor Processing Unit).

Seperti yang kita ketahui, GPU atau Graphics Processing Unit ini awalnya diciptakan untuk me-render grafik game sebelum akhirnya diadaptasi untuk AI. *Nah*, TPU sendiri adalah ASIC (Application-Specific Integrated Circuit) yang dirancang dari nol hanya untuk satu tugas, yaitu melakukan operasi perkalian matriks secara masif dan efisien. Mengingat Large Language Model (LLM) pada dasarnya hanyalah kumpulan operasi matematika matriks raksasa, TPU adalah mesin yang sangat sempurna.

Sekarang pertanyaannya adalah kapan harus menggunakan TPU?

Jika tumpukan *tech stack* yang Anda gunakan itu kompatibel dengan TPU, misalnya seperti dua contoh berikut ini.

*   **JAX:** Ini adalah "pasangan emas" TPU karena JAX dibangun dengan filosofi transformasi fungsi dan kompilasi XLA sejak awal, ia dapat memanfaatkan topologi interkoneksi TPU secara maksimal untuk performa yang sangat tinggi.
*   **PyTorch/XLA:** Jika menggunakan PyTorch, Anda memerlukan abstraksi XLA agar model bisa berjalan di TPU. Google dan Meta terus mengoptimalkan compiler ini untuk menutup celah performa dengan GPU.

Dalam banyak kasus, TPU menawarkan efisiensi biaya hingga 4x lebih baik daripada GPU kelas atas seperti NVIDIA H100 untuk model berbasis Transformer. Tidak berhenti disitu, TPU v7 bahkan diklaim mampu melakukan inferensi 10x lebih cepat dengan konsumsi daya 90% lebih rendah dibandingkan GPU tradisional dalam skenario tertentu.

##### AWS dengan EC2 P4/P5 Instances

Sekarang, bagaimana jika aplikasi Anda sangat bergantung pada ekosistem CUDA milik NVIDIA? *Nah*, jika ini kasusnya, menyewa instance AWS (Amazon Web Services) adalah taman bermain yang sangat sesuai untuk Anda.

*Eits*, sebelum masuk lebih dalam, apa sebenarnya instance itu?

Sederhananya, instance adalah sebuah komputer virtual atau server yang Anda sewa di dalam pusat data AWS. *Yup*, jadi itu bukan hanya seonggok GPU tanpa CPU, RAM, dan storage, melainkan satu paket unit komputer lengkap yang berjalan di atas infrastruktur cloud Amazon.

*Nah*, biasanya dalam praktik pengembangan software, kita akan memilih instance kecil dan murah yang hanya berisi CPU standar untuk menjalankan web server biasa yang biasanya disebut keluarga instance seri T atau M. Aplikasi generative AI tentu saja akan masuk ke kelas heavyweight, yaitu keluarga P-Series

*   **EC2 P4 Instances:** Komputer virtual ini ditenagai oleh kluster GPU NVIDIA A100 (GPU yang menjadi tulang punggung revolusi ChatGPT di masa-masa awal).
*   **EC2 P5 Instances:** Ini adalah "monster" generasi terbaru yang ditenagai oleh NVIDIA H100, dirancang khusus untuk melahap model-model LLM berukuran raksasa dengan kecepatan luar biasa.

Namun, ada satu fakta yang perlu Anda terima selain harga sewa instance yang cukup mahal. Mungkin Anda akan mengira menyewa server A100 di AWS sama mudahnya dengan membuat instance Ubuntu biasa, yaitu tinggal klik *“Launch”*, lalu server menyala. Nah, realita berkata sebaliknya karena saat ini sedang terjadi fenomena *GPU shortage* (kelangkaan GPU) di seluruh dunia.

Jika akun AWS Anda masih baru, kemungkinan besar Anda akan mendapati pesan error *"Insufficient Capacity"* saat mencoba menyalakan instance P4 atau P5 *On-Demand*. Kuota untuk GPU “sultan” ini sangat dibatasi. Anda harus berburu untuk mengajukan tiket peningkatan kuota ke support AWS disertai dengan argumen bisnis yang kuat tentang alasan Anda membutuhkannya, atau berkomitmen untuk menyewa dalam jangka waktu panjang (1 hingga 3 tahun) melalui skema *Reserved Instances* agar diprioritaskan.

Dinamika pasar dan industri ini membuat kita sebagai *Gen AI Engineer* tidak hanya harus ahli menulis kode, tetapi juga harus cerdas merencanakan “kapan” dan “di mana” Anda akan "memarkir" model sebelum hari peluncuran tiba.

#### Pendekatan On-Premise yang Lebih Mandiri

Seperti yang kita bahas sebelumnya, menyewa komputer virtual berisi GPU sultan di AWS atau Google Cloud memang sangat instan dan nyaman. Namun, semua kenyamanan itu tentu saja memiliki harga yang sangat tidak bersahabat.

Selain faktor harga, ada hal lain yang perlu diperhatikan. Sekarang bayangkan Anda sedang mempresentasikan sistem AI yang kepada dewan direksi sebuah bank nasional berskala besar atau fasilitas kesehatan milik pemerintah. Pertanyaan pertama mereka biasanya bukanlah seberapa akurat model AI Anda, melainkan *"Apakah data transaksi nasabah atau rekam medis pasien kita akan diproses dan mengalir ke server milik Amazon atau Google?"* Jika jawabannya *"Iya"*, deal proyek miliaran tersebut kemungkinan besar akan batal hari itu juga.

Ketika berhadapan dengan dua tembok besar ini, perusahaan tidak punya pilihan lain selain mengganti arah haluan ke arah Pendekatan On-Premise.

*Yup*, pendekatan On-Premise sederhananya adalah pendekatan di mana Anda membeli hardware fisik seperti NVIDIA DGX atau server dengan GPU H100/A100 atau bahkan GPU consumer grade seperti RTX 4090 lalu memasangnya di komputer Anda sendiri atau ruang server kantor.

Namun, bukannya membeli GPU itu juga mahal? Lantas, mengapa pendekatan On-Premise ini diklaim lebih murah? *Alright,* mari kita buka jawaban sebenarnya dengan mengenal dua istilah di dunia akuntansi korporat, yaitu OpEx dan CapEx.

*   **OpEx pada Jalur Cloud**
    Menyewa GPU A100 di AWS sebelumnya kita kategorikan sebagai OpEx atau Operational Expenditure yang maknanya adalah biaya operasional yang Anda bayarkan setiap bulan. Sifatnya fleksibel sehingga pada saat aplikasi sepi maka tagihan turun. Sebaliknya, jika aplikasi ramai, tagihan akan meroket. Masalahnya, untuk aplikasi Gen AI dengan traffic konstan 24/7, garis grafik OpEx ini akan terus naik dan lama-kelamaan memakan margin keuntungan perusahaan Anda.
*   **CapEx pada Jalur On-Premise**
    Di sisi lain, CapEx atau Capital Expenditure adalah belanja modal di awal. Anda akan mengeluarkan uang dalam jumlah raksasa di awal untuk membeli aset fisik. Misalnya, Anda membeli satu unit NVIDIA DGX (superkomputer AI *all-in-one*) atau merakit server bare-metal sendiri dengan jajaran GPU H100. Harganya bisa mencapai miliaran rupiah di awal.

Lalu, kapan CapEx menjadi lebih masuk akal?

Di sinilah letak seninya. Jika Anda menghitung proyeksi biaya sewa cloud selama 2 hingga 3 tahun untuk beban kerja tinggi yang konstan, Anda akan menemukan sebuah Titik Impas atau Break-Even Point (BEP). Sering kali, total uang yang Anda bakar untuk membayar sewa AWS selama 18 bulan ternyata sudah setara dengan harga beli fisik server tersebut.

Jika prediksi bisnis menunjukkan aplikasi Anda akan berumur panjang dengan utilitas mesin yang selalu tinggi di atas 70%, membeli hardware sendiri melalui jalur CapEx secara matematis jauh lebih murah dalam jangka panjang. Mesin itu menjadi aset perusahaan Anda, bukan sekadar tagihan bulanan.

*Alright*, pertanyaan soal harga sudah *clear*. Sekarang kita pindah ke alasan kedua, yaitu mengenai keamanan data.

Alasan kedua ini sering kali membatalkan opsi cloud sejak hari pertama, tidak peduli seberapa murah atau mahalnya biaya yang ditawarkan. Seperti yang kita bahas di awal, saat berurusan dengan data yang sifatnya sensitif, kita tidak bisa mengirimkannya melalui internet publik ke server milik pihak ketiga. Ada regulasi ketat seperti HIPAA di kesehatan, atau aturan OJK dan bahkan Bank Indonesia di sektor finansial lokal yang mengatur Data Sovereignty.

*Nah*, dalam dunia On-Premise, kita memiliki kontrol absolut baik atas model dan bahkan data yang digunakan. Anda bisa menerapkan sistem *Air-Gapped* yang memungkinkan server AI Anda secara fisik terputus dari koneksi internet luar.

Semua proses dimulai dari prompt User masuk, KV Cache terbentuk di VRAM, hingga hasil inference keluar itu terjadi di dalam ruang server milik Anda sendiri. Tidak ada satu pun byte data yang bocor ke luar dan kondisi seperti ini akan memenangkan kepercayaan klien tingkat enterprise.

*Eits*, dengan semua kelebihan di atas, bukan berarti pendekatan On Premise tidak memiliki tantangan apa pun. Justru untuk mewujudkan keamanan absolut seperti di atas, tantangan yang muncul di sini bisa dikatakan cukup *tricky*.

*   **Pengadaan Hardware**
    Tantangan pertama jatuh pada ketersediaan barang dari GPU kelas atas seperti NVIDIA A100 atau H100 yang sering kali mengalami kelangkaan global. Anda mungkin harus masuk *waiting list* berbulan-bulan hanya untuk mendapatkan perangkat kerasnya. Mungkin Anda dapat mengakalinya dengan cara menggunakan GPU consumer grade seperti NVIDIA RTX 5090 yang tentunya dengan beberapa treatment dan batasan khusus seperti menerapkan konsep *queue* atau efisiensi pada inference engine. Apa itu inference engine? Tenang kita akan membahasnya di bagian berikutnya nanti.
*   **Kebutuhan Daya Listrik yang "Brutal"**
    Satu server yang penuh dengan GPU mampu mengonsumsi listrik 5 hingga 10 kali lipat dari server web tradisional. Namun, tantangannya bukan sekadar tagihan listrik bulanan yang membengkak, tetapi kelayakan infrastruktur kelistrikan pada gedung rumah atau kantor Anda. Sering kali, perusahaan harus merombak total panel listrik mereka, menambah kapasitas UPS (*Uninterruptible Power Supply*) skala masif, hingga menyiapkan genset khusus agar proses inference atau fine-tuning tidak tiba-tiba *shutdown* karena listrik yang tidak stabil atau mati lampu.
*   **Manajemen Suhu**
    Komputasi besar pada akhirnya akan menghasilkan panas yang luar biasa besar. Ini mungkin sering Anda rasakan saat komputer atau laptop Anda sedang mengerjakan suatu tugas berat. *Nah*, AC sentral biasa di ruang server konvensional dijamin tidak akan berkutik melawan panasnya rak GPU yang sedang full load memproses token. Jika suhu dibiarkan naik, GPU secara otomatis akan mengalami thermal throttling atau kondisi di mana kecepatan clock diturunkan paksa untuk mencegah chip meleleh atau terbakar, yang ujung-ujungnya membuat respons LLM Anda jadi sangat lambat. Oleh karena itu, negara seperti Cina bahkan merendam server mereka di laut yang memiliki kapasitas penyerapan panas 3.000 kali lebih efisien daripada udara.
*   **Kebutuhan SDM Spesialis**
    Jika satu GPU error atau ada masalah pada memory bandwidth, seluruh performa aplikasi AI Anda bisa terganggu. Masalahnya, teknisi yang Anda butuhkan bukan sekadar SysAdmin konvensional. Anda butuh AI Infrastructure Engineer yang paham betul cara memantau kesehatan GPU, melakukan tuning performa, mengurus driver CUDA yang terkadang merepotkan setelah *update*, dan bisa mendiagnosis bottleneck di level hardware. Mencari, merekrut, dan mempertahankan talent dengan keahlian *niche* seperti ini adalah tantangan besar di bursa kerja saat ini.

Perdebatan antara Cloud-Native dan On-Premise akan selalu kembali pada prioritas bisnis dan kesiapan tim atau perusahaan Anda. Nah, setelah infrastrukturnya berdiri, pertanyaan lanjutannya tentu saja *“Bagaimana cara kita mengemas, memindahkan, dan menjalankan model AI yang sangat kompleks ini ke dalam?”*

---

## Containerization & Orchestration

Perlu Anda ketahui, dalam dunia Generative AI terdapat sebuah sindrom bernama *"It works on my machine"*. Sindrom seperti apa itu? Anda mungkin pernah mengalami kode berjalan sangat *seamless* dan sempurna saat diuji coba di komputer lokal, tetapi error begitu di-*deploy* ke server production. *Nah*, itu dia sindrom *"It works on my machine"*.

*Yup*, men-deploy model LLM ke tahap production itu tidak semudah menjalankan script Python di laptop lokal kita. Model LLM membawa segudang dependency yang kompleks mulai dari versi CUDA yang spesifik, library PyTorch, hingga konfigurasi sistem operasi. Nah, sedikit saja ada ketidakcocokan antara versi yang digunakan dengan environment di server, LLM Anda dijamin akan melemparkan rentetan error yang merepotkan.

Alih-alih memindahkan komponen dan dependency tersebut satu per satu, kita akan membungkus seluruh AI termasuk model, library, hingga *environment*-nya ke dalam sebuah sebuah wadah yang terstandardisasi bernama **container**. Sehingga, mau di mana pun container ini diletakkan—di server AWS, Google Cloud, atau server di kantor Anda—ia akan berjalan dengan cara yang 100% sama.

*Nah*, software paling standar di industri untuk membuat dan menjalankan container ini bernama Docker. Oleh karena itu, di bagian ini kita akan berkenalan lebih dalam dengan Docker. Tentu saja kita juga akan membahas cara menangani file image Docker LLM yang ukurannya bisa belasan hingga puluhan gigabyte. Tunggu sebentar, apa itu *file image*?

Mari kita bahas saja langsung dengan berkenalan dengan Docker terlebih dahulu.

### Berkenalan dengan Docker

Sebelum memulai perjalanan ini, ada satu pertanyaan paling fundamental saat kita ingin berkenalan dengan suatu tools. Pernahkah Anda memperhatikan logo Docker? Jika belum, coba perhatikan logonya di bawah ini.

![dos-af3111d2b9c7a7c046932f851f845df920260430161415.png](https://assets.cdn.dicoding.com/original/academy/dos-af3111d2b9c7a7c046932f851f845df920260430161415.png)

*Yup*, logonya berupa seekor paus biru lucu yang mengangkut tumpukan peti kemas atau disebut *shipping containers* di punggungnya. Logo ini bukan sekadar desain imut, melainkan representasi dari apa yang Docker lakukan.

Mari kita mundur sejenak ke industri logistik tahun 1950-an. Sebelum adanya peti kemas yang terstandardisasi, proses memuat barang ke atas kapal laut adalah sebuah kekacauan. Ada kopi di dalam karung, minyak di dalam drum, dan mesin di dalam peti kayu. Bentuknya berbeda-beda dan ukurannya bermacam-macam. Akibatnya, butuh waktu berhari-hari hanya untuk menata barang di kapal agar teratur dan dan memaksimalkan ruang.

Kehadiran peti kemas merevolusi hal itu karena tidak peduli apakah isinya karung kopi, mobil mewah, atau perangkat elektronik, semuanya dimasukkan ke dalam kotak besi dengan ukuran standar yang sama. Truk, crane pelabuhan, dan kapal kargo di seluruh dunia didesain untuk mengangkat kotak standar tersebut.

*Nah*, Docker melakukan hal yang sama persis, tetapi untuk software.

Di dunia pengembangan aplikasi, setiap aplikasi dengan *tech stack* yang berbeda-beda memiliki komponen dan dependency yang sangat bervariasi dalam “kopernya”. Nah, untuk memahami urgensi Docker, mari kita coba *breakdown* apa saja yang ada di dalam model generative AI.

*   **Versi Python:** Model Anda misalnya menggunakan Python 3.10.
*   **Library:** PyTorch, Transformers dari HuggingFace, LangChain, hingga vector database seperti Faiss atau Chroma.
*   **Hardware AI:** CUDA Toolkit

*Alright*, ini baru aplikasi generative AI, masih ada aplikasi backend berbasis Node.js, aplikasi web dengan React yang juga membawa “barang” yang berbeda-beda.

Alih-alih pusing memikirkan cara menginstal masing-masing aplikasi di server yang berbeda, Docker membungkus aplikasi Anda beserta seluruh kebutuhannya ke dalam satu "peti kemas digital" standar bernama **Container**. Setelah dibungkus, container ini bisa dijalankan di mana saja, entah itu di laptop Mac Anda, server Windows kantor, AWS, atau Google Cloud sekalipun.

Setelah Anda mendengar gambaran pentingnya docker, mungkin masih akan terselip pertanyaan seperti ini, *“Memangnya skenario buruk apa yang terjadi ketika tidak menggunakan Container?”*. *Nah*, untuk menjawabnya Anda masih ingat tentunya dengan istilah *"It works on my machine"* di awal.

Bayangkan Anda baru saja selesai membuat aplikasi chatbot RAG. Tidak hanya itu, Anda juga telah menghabiskan waktu berhari-hari melakukan fine-tuning di laptop atau PC lokal. Saat masih ada di komputer Anda, semuanya berjalan dengan sempurna menggunakan Python versi 3.10, PyTorch versi 2.1, driver CUDA NVIDIA terbaru, dan belasan library pendukung lainnya.

Setelah itu, Anda mengirimkan source code tersebut ke tim IT untuk diunggah ke server production. Satu jam kemudian, tim IT menelepon Anda dan berkata *"Bro, aplikasinya error nih. Modelnya nggak mau jalan."*. Tentu saja Anda bingung dan membalas *"Loh, nggak mungkin soalnya di laptop saya jalan kok!"*.

Apa yang sebenarnya terjadi?

Ternyata, server production di kantor Anda menggunakan OS Ubuntu versi lama dengan Python versi 3.8. Beberapa library AI yang Anda gunakan tidak kompatibel dengan Python 3.8. Lebih parahnya lagi, versi driver GPU di server tersebut berbeda dengan yang ada di laptop Anda yang membuat CUDA Toolkit yang digunakan sebelumnya *conflict*. Alhasil aplikasi Anda pun hancur lebur di production.

Di sanalah pentingnya Docker sebagai alat standardisasi. Namun, sekarang pertanyaannya berubah, *“Apa yang dilakukan Docker untuk mewujudkan standardisasi tersebut?”* Untuk menjawabnya, kita perlu berkenalan dengan ketiga komponen di dalam Docker yang bekerja secara berurutan.

*Alright*, kita akan memulainya dari sebuah resep bernama Dockerfile.

#### Dockerfile

Semuanya dimulai dari sebuah file teks sederhana yang sudah seperti *blueprint* dari Docker bernama Dockerfile. Anda bisa membayangkannya sebagai script instruksi langkah demi langkah yang sudah ditulis untuk memberi tahu Docker bagaimana cara membangun environment aplikasi Anda.

Contoh sederhananya seperti pada ilustrasi berikut ini.

![dos-fdecb02e7f8535d01e03e9fe374e30b220260430161503.png](https://assets.cdn.dicoding.com/original/academy/dos-fdecb02e7f8535d01e03e9fe374e30b220260430161503.png)

Ilustrasi di atas memberikan kita gambaran contoh sederhana dari *dockerfile* yang akan Anda buat. *Eits*, tentu saja dockerfile yang akan Anda buat nantinya tidak akan serta merta sama dengan contoh di atas karena akan menyesuaikan kebutuhan Anda. Dockerfile untuk aplikasi web biasa akan sangat berbeda dengan Dockerfile untuk menjalankan LLM.

Meskipun bentuknya berbeda, instruksi baku yang digunakan pada dasarnya mirip. *So*, mari kita pahami 5 instruksi paling fundamental dalam Dockerfile berikut ini.

*   **FROM**
    Ini wajib menjadi baris pertama di setiap Dockerfile. Perintah ini menentukan OS atau lingkungan dasar apa yang ingin kita pakai. Untuk web app biasa, mungkin kita cukup menggunakan Python dengan versi yang paling ringan seperti `FROM python:3.10-slim`. Di sisi lain, jika aplikasi Anda perlu menjalankan model LLM di atas GPU, kita harus berubah haluan ke `FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04`.
*   **WORKDIR**
    Daripada file kita berantakan di dalam sistem operasi Docker, kita membuat satu folder khusus. Misalnya, dalam contoh di atas kita menggunakan `WORKDIR /app`. Nantinya semua perintah yang kita tulis selanjutnya secara otomatis akan dilakukan di dalam folder `/app` di dalam Container, bukan di folder laptop Anda.
*   **COPY**
    Perlu diketahui bahwa Docker tidak bisa tiba-tiba membaca file di laptop Anda. Sehingga, Anda harus memerintahkannya secara eksplisit untuk menyalin file tersebut ke dalam blueprint. *Nah*, contoh instruksi `COPY . .` pada ilustrasi di atas akan meminta docker menyalin seluruh source code aplikasi AI Anda ke dalam folder `/app` tadi.
*   **RUN**
    Perintah RUN digunakan untuk mengeksekusi perintah terminal di dalam Docker saat Image sedang dibentuk. Di sinilah proses download dan instalasi terjadi, contohnya adalah `RUN apt-get update && apt-get install -y python3-pip` di atas yang digunakan untuk menginstal package manager Python.
*   **CMD**
    Jika RUN dieksekusi saat Image dibuat, CMD dieksekusi saat Container dinyalakan atau saat menjalankan `docker run` di terminal. Dapat dikatakan ini adalah garis finish dari Dockerfile. Perintah `CMD ["python", "api.py"]` di atas akan membuat Docker menjalankan script API Anda.

#### Docker Image

*Alright*, ketika sudah selesai menulis Dockerfile, Anda akan menjalankan satu perintah di terminal, yaitu `docker build`.

![dos-553750887f04ec21d9c949383e16631c20260430161302.png](https://assets.cdn.dicoding.com/original/academy/dos-553750887f04ec21d9c949383e16631c20260430161302.png)

Saat perintah ini ditekan, Docker akan mulai membaca instruksi pada dockerfile yang kita buat baris demi baris. Ia akan menjalankan setiap perintah mulai dari men-download OS dasar, menginstal Python, menginstal PyTorch, dan menyalin kode Anda. Ketika semua proses instalasi ini selesai, Docker akan memadatkan dan membekukan seluruh sistem tersebut menjadi satu file paket statis.

Hasil akhir yang beku inilah yang kita sebut sebagai Docker Image. Tunggu sebentar, apa maksudnya dibekukan?

*Alright*, kita akan menjawabnya sembari menjelaskan karakteristik yang dimiliki oleh Docker Image.

*   **Bersifat Immutable**
    Dockerfile yang dibekukan ini membuat Docker Image bersifat *immutable* atau *read-only*. Jika Anda menyadari ada typo (salah ketik) pada kode Python Anda, Anda tidak bisa membuka Docker Image tersebut lalu mengedit teksnya. Anda harus memperbaiki kodenya di laptop Anda, lalu menjalankan `docker build` lagi untuk mencetak sebuah Image versi baru. Terdengar sangat kaku, tetapi justru itulah yang membawa rasa aman. Sebab, ini akan mencegah rekan tim IT Anda dapat menghapus satu file library kecil, atau sistem server otomatis memperbarui versi driver-nya yang dapat menyebabkan conflict pada aplikasi.
*   **Arsitektur Berlapis**
    Docker Image tidak dibuat sebagai satu balok file raksasa, melainkan dibangun dari tumpukan layers. Setiap perintah di Dockerfile seperti FROM, RUN, dan COPY akan menciptakan satu layer baru.
    *   Layer 1: OS Ubuntu + CUDA Toolkit (Bisa berukuran 4 GB)
    *   Layer 2: Instalasi PyTorch & Library AI (Bisa berukuran 6 GB)
    *   Layer 3: Kode Python aplikasi Anda (Hanya 5 Megabyte)

    Saat image Docker kita berisi model AI, ukurannya bisa mencapai satuan hingga belasan gigabyte. Bayangkan jika setiap kali Anda mengedit kode Python yang ukurannya hanya 5 megabyte, Anda harus menunggu server merakit ulang file 10 GB dari awal. Berkat arsitektur lapisan ini, saat Anda hanya mengubah kode aplikasi (Layer 3) lalu melakukan build ulang, Docker tidak akan menginstal ulang Ubuntu maupun PyTorch dan hanya mengganti lapisan teratasnya.

Setelah Image belasan gigabyte ini sudah jadi di laptop atau server development kita, bagaimana cara memindahkannya ke server production?

*Nah*, Docker Image ini nantinya didistribusikan melalui tempat penyimpanan terpusat yang disebut Docker Registry yang salah satu contoh paling terkenalnya adalah Docker Hub atau Google Artifact Registry. Proses yang terjadi sebenarnya sangat sederhana dan mirip saat kita melakukan push dan pull di GitHub.

*   Anda mendorong (*push*) Image yang sudah jadi dari laptop Anda ke registry.
*   Server production Anda tinggal menarik (pull) Image tersebut dari registry.

Dengan penjelasan ini, Anda akan memiliki fondasi yang kuat ketika nanti berurusan dengan Image Docker berukuran belasan gigabyte pada bagian selanjutnya. *Eits*, tapi untuk sekarang kita akan lanjut dulu ke komponen terakhir, yaitu Docker Container.

#### Docker Container

Saat Anda mengetikkan perintah **`docker run <nama-image>`**, Docker akan mengambil Image yang beku tadi, memasukkannya ke dalam RAM server, dan menyalakannya. Detik ketika sistem itu menyala dan mulai memproses data, ia resmi disebut sebagai Container.

![dos-1f9c9408b358fbe7651510143c77726a20260430161333.png](https://assets.cdn.dicoding.com/original/academy/dos-1f9c9408b358fbe7651510143c77726a20260430161333.png)

Secara teknis, Container bukanlah sebuah mesin virtual atau komputer baru. Ia hanyalah sebuah proses aplikasi biasa di dalam server Anda. Namun, Docker membungkus proses ini dengan "dinding isolasi" yang ketat. Namun, jika Container ini sangat terisolasi, dari mana ia mendapatkan resource untuk menjalankan aplikasi?

*Nah*, selain untuk mengisolasi, dinding yang berasal dari dua fitur kernel Linux tersebutlah yang memungkinkan Container dapat mengeksekusi komputasinya. Kedua fitur tersebut adalah sebagai berikut.

*   **Namespaces:** bertugas memisahkan apa yang bisa dilihat oleh proses (seperti jaringan, ID pengguna, dan struktur file).
*   **Cgroups:** bertugas membatasi apa yang bisa digunakan oleh proses (seperti jumlah RAM, penggunaan CPU, dan bandwidth).

Hasilnya, Container yang hanya membawa software (seperti Python dan library AI) melalui Image-nya, dapat memanfaatkan beberapa hardware pada server fisik (Host) seperti CPU, RAM, dan Kernel OS.

Dalam praktiknya pada aplikasi biasa, perintah untuk menjalankan Container ini sangat singkat. Contohnya mungkin seperti di bawah ini.

```bash
docker run -d -p 8080:80 --name my-web nginx
```

Namun, men-*deploy* model LLM akan menggunakan cara yang berbeda karena menuntut konfigurasi infrastruktur yang jauh lebih spesifik. Perintah `docker run` untuk LLM biasanya digunakan untuk menjalankan inference engine seperti vLLM. *Wait*, apa itu inference engine vLLM?

Tenang saja, kita akan membahasnya di bagian lain. Untuk sekarang, mari kita bedah perintah `docker run` yang biasanya digunakan untuk LLM tersebut.

![dos-322196312ed9f6c52e58a13135ebf5fe20260430161638.png](https://assets.cdn.dicoding.com/original/academy/dos-322196312ed9f6c52e58a13135ebf5fe20260430161638.png)

*   **Volume Mapping (**`-v`**)**
    Sifat dasar Container adalah sementara, setiap kali Container di-restart, sistem harus men-*download* ulang file Model yang ukurannya raksasa tersebut dari internet. *Nah*, parameter `-v ~/.cache/huggingface:/root/.cache/huggingface` hadir untuk menyambungkan folder cache di server fisik (Host) langsung ke dalam Container yang terisolasi. Sehingga, file Model hanya perlu di-*download* sekali saja dan akan tersimpan aman di hard disk server fisik.
*   **Port Mapping (**`-p 8000:8000`**)**
    Di awal, kita tahu bahwa Container itu terisolasi dari internet. *Nah*, parameter `-p 8000:8000` inilah yang akan mengatur rute jaringan dengan berkata seperti ini, *"Setiap ada request yang masuk ke port 8000 di server fisik, langsung teruskan ke port 8000 di dalam Container."* Inilah yang membuat aplikasi frontend atau User Anda bisa berinteraksi dengan API AI di dalam Container tersebut.
*   **Mencegah Bottleneck Memori AI (**`--ipc=host`**)**
    Secara default, Docker sangat membatasi kapasitas Shared Memory (memori yang bisa dipakai bersama antar proses) untuk alasan keamanan. Namun, framework AI seperti PyTorch atau vLLM sangat rakus akan shared memory untuk memindahkan data tensor berukuran masif dengan cepat. *Nah*, parameter `--ipc=host` akan menghapus batasan tersebut dengan mengizinkan Container menggunakan kapasitas shared memory yang sama besarnya dengan mesin server utama.
*   **Docker Image**
    Di baris terakhir, `vllm/vllm-openai:latest` adalah nama Docker Image yang kita panggil, sedangkan `--model google/...` adalah perintah spesifik yang diteruskan ke dalam aplikasi vLLM untuk menentukan model mana yang harus dimuat ke memori.

Apakah sudah selesai? Sayangnya belum karena ada satu komponen yang tertinggal.

Seperti yang kita ketahui, untuk menjalankan LLM secara optimal, kita membutuhkan tenaga kuda dari GPU. Masalahnya, sistem keamanan Container secara bawaan memblokir akses ke perangkat keras eksternal seperti kartu grafis.

Jika Anda hanya menjalankan perintah di atas tanpa memedulikan GPU, vLLM akan mencoba menghitung menggunakan CPU, dan proses generate satu kalimat bisa memakan waktu berjam-jam. Di sinilah baris pertama dari perintah `docker run` di atas memainkan peran paling krusial.

![dos-5e6a02ab5ec66c4d005025d6de94076c20260430161723.gif](https://assets.cdn.dicoding.com/original/academy/dos-5e6a02ab5ec66c4d005025d6de94076c20260430161723.gif)

*Nah*, agar parameter ini tidak menghasilkan eror, kita membutuhkan teknologi penengah bernama NVIDIA Container Toolkit yang sudah harus terinstal di server fisik Anda.

Ketika Anda menambahkan parameter `--runtime nvidia --gpus all`, toolkit ini mencegat perintah tersebut dan secara eksplisit meneruskan (passthrough) akses driver GPU fisik dari server langsung ke dalam Container melewati dinding isolasi.

*Yup*, proses Python di dalam Container akhirnya bisa menggunakan seluruh kekuatan GPU yang ada di server, memungkinkan model Gen AI Anda berjalan dengan kecepatan penuh sambil tetap mempertahankan konsistensi dependensi software yang aman di dalam isolasi Container.

Sekarang pertanyaannya, apakah untuk menjalankan aplikasi Generative AI kita dapat bergantung sepenuhnya pada satu container saja? Sayangnya jawabannya adalah tidak.

### Skalabilitas dan Kebutuhan Multi-Container

Jika mengingat kembali berbagai ragam aplikasi seperti backend Node.js, interface React, hingga script Python sebelumnya, kita dapat menyadari bahwa sebuah aplikasi itu jarang sekali hanya terdiri dari satu program tunggal.

Bayangkan Anda memiliki satu server Cloud yang besar. Di dalam server tersebut, Anda tidak hanya menjalankan satu Container, melainkan menyalakan beberapa Container sekaligus dengan peran yang berbeda-beda.

![dos-674db326e37ffd4a4e7e523b66c28a3d20260430161814.png](https://assets.cdn.dicoding.com/original/academy/dos-674db326e37ffd4a4e7e523b66c28a3d20260430161814.png)

Seperti yang dapat Anda lihat pada ilustrasi tersebut, di dalam satu server kita dapat memiliki beberapa Container dari setiap anatomi aplikasi Generative AI.

*   **Container LLM Inference Engine:** Menjalankan inference engine seperti vLLM atau SGLang untuk menghasilkan output dari LLM. Nah, jika Anda perhatikan, hanya container LLM yang dapat mengakses GPU melalui NVIDIA Toolkit.
*   **Container Vector Database:** Menjalankan database seperti Chroma atau Milvus untuk menyimpan dokumen RAG Anda. Container ini tidak butuh GPU, tetapi butuh RAM yang cukup besar.
*   **Container Backend API:** Menjalankan aplikasi Python (FastAPI) atau Node.js. Ini adalah penghubung yang akan menerima pertanyaan dari User, mencari konteks ke Container Vector database, lalu menyuruh Container LLM merangkai jawabannya.
*   **Container Frontend Web:** Menjalankan website antarmuka seperti React atau Streamlit yang nantinya digunakan User mengetikkan pertanyaan.

Seperti yang kita ketahui sebelumnya, container Vector DB dan LLM Engine ini sebenarnya juga punya satu “pipa rahasia” lagi ke hard disk server untuk menyimpan data model dan dokumen agar datanya tidak hilang saat kontainer mati.

Mungkin di sini Anda sudah mulai terpikirkan sebuah pertanyaan berikut: Mengapa tidak memasukkan keempat komponen di atas ke dalam satu Container saja?

Jawabannya adalah karena untuk mengisolasi masalah.

*Yup*, jika seandainya Container yang berisi website Frontend Anda tiba-tiba mengalami *error* atau *crash*. Kerusakannya tidak akan menyebar ke Container lainnya karena Frontend yang bermasalah berada di dalam Container yang terisolasi. Jika semuanya tidak dipisah dengan Docker, satu aplikasi error bisa membuat seluruh server hang dan harus di-*restart* total.

Sekarang Anda telah sadar bahwa untuk menjalankan aplikasi generative AI dengan RAG sederhana, kita membutuhkan empat Container yang berbeda dan saling berkomunikasi melalui jaringan internal secara aman di dalam server yang sama.

Sekarang kita memiliki dua pertanyaan. Pertanyaan pertama dan paling penting adalah *“apa komponen yang berada di tengah dan dirahasiakan tersebut?”* Lalu, pertanyaan keduanya adalah *“Karena memiliki 4 container, apakah kita perlu menyalakannya satu per satu?”*.

*Nah*, kedua pertanyaan menarik tersebut akan terjawab pada materi Orkestrasi dengan Docker Compose.

### Orkestrasi dengan Docker Compose

*Yup*, mungkin sudah sedikit tertebak dari judul di atas tentang *“komponen apa yang tersembunyi”* pada ilustrasi sebelumnya. *So*, mari kita *reveal* saja melalui ilustrasi di bawah ini

![dos-b73c0efce212774c1b0d751ce5e3abed20260430161841.png](https://assets.cdn.dicoding.com/original/academy/dos-b73c0efce212774c1b0d751ce5e3abed20260430161841.png)

Sesuai dengan judul bagian ini, Docker Compose adalah *orchestrator* dari beberapa Container yang dibutuhkan aplikasi yang dibangun. Secara teknis, Docker Compose adalah *tool* bawaan dari ekosistem Docker yang berfungsi untuk mendefinisikan, menjalankan, dan mengelola aplikasi multi-container di atas satu mesin server fisik (*Single-Node*).

Namun, apakah memang sepenting itu kehadiran Docker Compose?

Pada bagian sebelumnya, kita bertanya *“apakah keempat container tersebut dijalankan satu per satu?”*. *Nah*, jika menggunakan ide satu per satu, saat perlu me-*restart* server, Anda perlu menjalankan empat perintah panjang berikut secara berurutan.

![dos-26dc6a763582e57a47fd86ea24fd8cb120260430161926.gif](https://assets.cdn.dicoding.com/original/academy/dos-26dc6a763582e57a47fd86ea24fd8cb120260430161926.gif)

Sekali lagi, instruksi `docker run` di atas hanyalah contoh untuk memberikan gambaran saja.

Perhatikan tahap 0 dalam ilustrasi di atas, sebelum menjalankan kontainer, kita diwajibkan membuat jaringan virtual manual terlebih dahulu agar antar-kontainer bisa saling mengenali namanya.

*Nah*, melalui instruksi `docker run` lengkap di atas, Anda sekarang bisa melihat dengan sangat jelas beban operasional dari pendekatan menjalankan satu per satu secara manual. Beberapa di antaranya adalah sebagai berikut.

*   Anda harus selalu ingat mengeksekusi perintah `docker network create`.
*   Anda harus menyematkan parameter `--network rag-network` di setiap perintah tanpa terkecuali.
*   Anda harus menunggu dan menatap layar terminal agar LLM selesai loading sebelum mengetik perintah Backend.
*   Jika server *restart*, kelima urutan ini termasuk network harus Anda ketik ulang satu per satu tanpa *typo* satu pun.

Sangat merepotkan, bukan? Sekarang bagaimana jika kita menggunakan Docker Compose?

*Nah*, dengan Docker Compose, kita dapat mendefinisikan dan menjalankan aplikasi multi-container menggunakan satu buah file teks berformat YAML (`docker-compose.yml`). Mari kita lihat wujud asli file `docker-compose.yml` untuk arsitektur Generative AI sederhana di bawah ini.

```yaml
version: '3.8'

services:
  # 1. Vector Database
  vector-db:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

  # 2. LLM Engine (vLLM)
  llm-engine:
    image: vllm/vllm-openai:latest
    command: --model mistralai/Mistral-7B-Instruct-v0.3
    ipc: host
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # 3. Backend API
  backend-api:
    image: my-backend-image:latest
    ports:
      - "8080:8080"
    environment:
      - DB_URL=http://vector-db:6333
      - LLM_URL=http://llm-engine:8000
    depends_on:
      - vector-db
      - llm-engine

  # 4. Frontend Web
  frontend-web:
    image: my-frontend-image:latest
    ports:
      - "3000:3000"
    environment:
      - API_URL=http://backend-api:8080
    depends_on:
      - backend-api

# Mendefinisikan volume penyimpanan persisten
volumes:
  qdrant_data:
```

Jika Anda membandingkan file YAML di atas dengan perintah `docker run` sebelumnya, akan terlihat beberapa otomatisasi dan efisiensi instruksi yang dilakukan oleh Docker Compose. Mari kita bedah satu per satu.

*   **Menghilangkan** `docker network create`
    *Yup*, Anda tidak akan menemukan definisi network manual di file ini. Sebab, secara by default, Docker Compose ini akan membuatkan satu jaringan virtual (VLAN) tertutup untuk semua services yang ada di dalam file ini. *Nah*, dengan ini seluruh container otomatis saling mengenali satu sama lain tanpa perlu kita atur satu per satu.
*   **Mengotomasi Antrian**
    Coba perhatikan blok `depends_on:` di `backend-api` dan `frontend-web`. Instruksi sederhana ini menggantikan keharusan Anda untuk menunggu prosesnya berjalan satu per satu, terutama menunggu LLM selesai loading. Compose membaca instruksi tersebut dan secara otomatis mengatur antriannya. Contohnya, *"Nyalakan Database dan LLM dulu. Setelah mereka berdua hidup, baru nyalakan Backend. Setelah Backend hidup, baru nyalakan Frontend."* 
*   **Hilangnya** `--runtime nvidia --gpus all`
    Di Docker Compose versi 3.8 ke atas, perintah GPU tersebut diterjemahkan secara rapi ke dalam blok `deploy:`. Penulisan ini jauh lebih terstruktur dan memungkinkan Anda mengatur spesifikasi hardware dengan sangat presisi. Misalnya, jika nanti Anda hanya ingin memberikan 2 GPU alih-alih semuanya, cukup ubah `count: 2`.

*Nah*, dengan menyimpan teks di atas ke dalam file bernama `docker-compose.yml`, tugas operasional Anda atau tim IT yang tadinya penuh risiko salah ketik, kini direduksi menjadi satu perintah saja di terminal.

```bash
docker compose up -d
```

*Yup*, hanya dengan menekan Enter pada perintah tersebut, Docker akan membaca seluruh arsitektur, mengunduh komponen yang belum ada, mengatur jaringan, menyambungkan GPU, mengatur volume ke hardisk, dan menghidupkan semuanya sesuai urutan.

Inilah alasan Docker Compose disebut sebagai fondasi Infrastructure as Code (IaC). *Yup*, itu karena Infrastruktur server Anda kini berbentuk teks yang bisa dibaca manusia, bisa dilacak perubahannya, dan bisa direplikasi di server mana pun di seluruh dunia dalam hitungan detik.

Namun, bagaimana cara model super raksasa yang mungkin memiliki parameter sebanyak triliunan dan mungkin tidak akan muat di dalam satu server dapat dibungkus ke dalam Docker Container?

Saat hal itu terjadi, Anda tentunya akan dipaksa menyewa server fisik kedua, ketiga, dan seterusnya. Masalahnya, Docker Compose tidak bisa menyatukan banyak server fisik. Di saat itulah Anda butuh Kubernetes untuk mendistribusikan beban (load balancing) ke banyak server sekaligus.

Namun, untuk 90% kasus startup Gen-AI di fase awal, satu server raksasa dengan Docker Compose sudah jauh dari cukup. *Nah*, terlepas dari nantinya Anda memakai Docker Compose di satu server, atau Kubernetes di 10 server, ada satu pertanyaan mendasar yang belum kita jawab.

Coba perhatikan kembali file Compose kita sebelumnya. Ada satu terminologi yang sering kita sebut dan kehadiran script Python main.py buatan kita sendiri untuk menjalankan model AI tersebut. *Yup*, itu adalah inference engine dengan layanan bernama **vLLM (image: vllm/vllm-openai:latest)**.

Pertanyaannya, mengapa kita butuh software pihak ketiga ini dan tidak me-*load* model dari HuggingFace menggunakan script Python biasa saja?

Untuk menjawabnya, mari kita ucapkan *“Selamat datang di submodul selanjutnya: Inference Engine”*.

---

## Load Model dengan Inference Engine

Di bagian sebelumnya, kita telah menyusun arsitektur production menggunakan Docker Compose. Jika Anda perhatikan dengan teliti file `docker-compose.yml`, kita tidak memanggil sebuah Docker Image yang berisi script Python buatan kita sendiri.

*Yup*, Alih-alih menulis kode Python main.py untuk memuat model, kita justru mengunduh dan menjalankan perangkat lunak pihak ketiga bernama **vLLM (image: vllm/vllm-openai:latest)**. Hal ini tentu menimbulkan pertanyaan besar mengenai alasan kita bergantung pada *software* pihak ketiga ini.

Ditambah lagi, sejak modul pertama kita sudah belajar cara memuat model langsung dari HuggingFace. Bahkan, kodenya pun sangat singkat, hanya butuh belasan baris Python untuk membuat AI kita membalas pesan.

Jawabannya ternyata ada pada istilah yang kita kenal sebelumnya saat membahas Containerization sebelumnya, yaitu *"It works on my machine".* Ternyata masalah tersebut datang menghantui saat kita menggunakan HuggingFace standard pada fase production.

### Masalah pada HuggingFace Standard

Jika menggunakan script bawaan dari pustaka transformers HuggingFace contohnya seperti fungsi `model.generate()` di dalam server API Anda, tidak diragukan lagi Anda sedang menabrak tiga tembok pembatas arsitektur yang sangat keras berikut ini.

#### Static Batching dan Masalah Padding

Di dunia nyata, request dari pengguna tidak datang secara bersamaan dengan panjang yang sama. User A mungkin hanya mengetik ingin berbasa-basi dan menyapa *"Halo"*, sementara detik berikutnya User B mungkin sedang mengerjakan tugasnya di detik akhir dan mengetik *"Buatkan esai 1000 kata"*.

*Nah*, jika menggunakan library standar Python, Anda dihadapkan pada dua pilihan yang sama-sama buruk.

*   **Opsi Tanpa Batching**
    Opsi ini akan membuat server mengeksekusi *request* satu per satu. Akibatnya, User B harus menunggu GPU selesai menjawab User A secara penuh sebelum gilirannya tiba. *Nah*, jika ada 100 antrian, User ke-100 akan mengalami *timeout*.
*   **Opsi Static Batching**
    Opsi ini menggabungkan request User A dan B ke dalam satu proses matriks GPU secara bersamaan. Namun, karena arsitektur GPU mewajibkan bentuk matriks yang simetris, sistem harus menambahkan padding tokens pada request User A agar panjangnya sama dengan Pengguna B. Akibatnya, GPU Anda menghabiskan 90% tenaga komputasinya untuk menghitung token kosong (padding) yang tidak ada artinya, hanya demi menyamakan bentuk matriks.

Lantas mana pilihan yang lebih baik?

Jawaban pragmatisnya tidak ada sama sekali yang lebih baik. Kedua pilihan tersebut sama-sama merugikan dan sulit ditoleransi untuk aplikasi Generative AI kita.

#### KV Cache Fragmentation

Anda masih ingat dengan KV Cache yang kita pelajari sebelumnya, bukan? *Yup*, saat LLM men-*generate* teks kata demi kata, ia harus mengingat konteks kata-kata sebelumnya agar tidak mengulanginya dari nol. Data ingatan jangka pendek ini disimpan di VRAM GPU dan disebut KV Cache atau Key-Value Cache.

Masalahnya, library standar HuggingFace menangani memori ini dengan sangat kaku.

Karena sistem tidak tahu secara pasti panjang token jawaban yang dihasilkan AI nantinya, sistem secara naif akan mereservasi ruang memori VRAM secara kontinu untuk kapasitas maksimal sejak detik pertama.

Kita ambil contoh GPU yang Anda gunakan telah memblokir memori VRAM sebesar 2048 token untuk seorang User yang sedang berinteraksi dengan model AI. Namun, AI ternyata hanya menjawab *"Halo, selamat pagi juga!"* yang berarti totalnya sekitar 6 token. Lantas, apa yang terjadi pada sisa ruang memori untuk 2042 token tersebut?

Ruang itu terkunci, kosong, dan tidak bisa dipakai oleh pengguna lain. Dalam ilmu komputer, ini disebut **Internal Fragmentation**.

Akibat memori yang "menguap" menjadi ruang kosong ini, sebuah server raksasa dengan GPU A100 (80GB VRAM) mungkin tiba-tiba menolak request pengguna ke-10 dan memunculkan error Out of Memory (OOM). Bukan karena memori GPU-nya benar-benar penuh oleh data, tetapi karena alokasinya sangat berantakan.

#### GIL dan Overhead

Dinding terakhir yang perlu kita lihat ada pada fondasi bahasa pemrogramannya. Seperti yang kita ketahui, library HuggingFace berjalan di atas Python dan PyTorch standar. *Nah*, Python memiliki sebuah mekanisme bawaan yang disebut Global Interpreter Lock (GIL). Mekanisme ini memastikan bahwa hanya ada satu baris kode Python yang bisa dieksekusi oleh prosesor atau CPU pada satu waktu, mencegah terjadinya *multithreading* murni.

Memang, saat melakukan komputasi paralel di GPU, python dapat melakukan *bypass* prosedur GIL ini dan membiarkan GPU tetap melakukan perhitungan secara paralel. Masalahnya, proses Tokenisasi terjadi sebelum data masuk ke GPU, dan proses ini terjadi sepenuhnya di CPU yang tercekik oleh GIL.

*Nah*, jika CPU butuh 10ms untuk melakukan Tokenisasi dan GPU hanya butuh 2 ms untuk Inference, GPU akan menganggur atau idle selama 8 ms menunggu *batch* berikutnya seperti yang ditunjukkan ilustrasi berikut.

[Gambaran terjadinya GIL, CPU 16-Core terhambat dan GPU idle]

Baiklah, Anda mungkin akan sedikit meremehkan proses Tokenisasi dan bertanya *“apakah seberat itu untuk melakukan Tokenisasi sehingga CPU butuh waktu lebih?”*.

*Yup*, tokenisasi modern bukanlah tugas yang ringan. Saat kita menekan enter untuk perintah `tokenizer.encode()`, CPU akan melakukan beberapa hal berikut.

*   **Normalisasi String:** Untuk menstandarisasi jutaan karakter atau kata agar konsisten, normalisasi dilakukan dengan beberapa cara meliputi Lower Casing dan Unicode fixing.
*   **Regex Matching:** Sebelum memecah kata, tokenizer menggunakan Regex untuk memisahkan tanda baca dan angka.
*   **Looping:** Algoritma Byte-Pair Encoding yang kita pelajari sebelumnya harus melakukan Looping pada seluruh teks untuk mencari pasangan karakter yang paling sering muncul dan menggabungkannya.

Semua itu adalah operasi CPU-Bound yang sangat membebani interpreter Python dan sangat sensitif terhadap GIL. Oleh karena itu, kita membutuhkan Inference Engine.

Inference Engine bukanlah model AI itu sendiri. Ia adalah perangkat lunak server khusus (ditulis dalam bahasa tingkat rendah seperti C++ atau CUDA) yang bertugas sebagai "manajer" antara model AI Anda dan perangkat keras GPU.

Tugas utamanya hanya satu: Memeras setiap tetes performa dari GPU Anda agar bisa melayani sebanyak mungkin pengguna secara bersamaan dengan waktu respons secepat kilat. *Nah*, kabar menariknya, kita memiliki beberapa opsi inference engine yang dapat kita pilih. Mari kita mulai eksplorasinya dari opsi yang paling ramah pemula, yaitu LMStudio.

### LMStudio

Bagi banyak orang terutama pemula, mengonfigurasi AI sering kali identik dengan layar terminal hitam yang penuh dengan teks error yang menakutkan. *Nah*, LMStudio membuang semua kerumitan itu.

![dos-e661f58f11b0dcb85fddb05b988be40220260312154142.png](https://assets.cdn.dicoding.com/original/academy/dos-e661f58f11b0dcb85fddb05b988be40220260312154142.png)

LMStudio pada dasarnya hanyalah sebuah aplikasi desktop biasa. Anda mengunduhnya, melakukan klik next-next-install layaknya menginstal Spotify atau VSCode di laptop Anda, dan selesai. LMStudio menyediakan antarmuka visual (GUI) yang sangat bersih.

![dos-48096f81cc07f5a23a21058eb54a7d0720260430162527.gif](https://assets.cdn.dicoding.com/original/academy/dos-48096f81cc07f5a23a21058eb54a7d0720260430162527.gif)

Anda bisa mencari model, mengunduhnya, dan langsung menguji prompt secara visual di kolom chat yang disediakan tanpa perlu dipusingkan dengan satu baris kode pun. Ini sangat menghemat waktu saat Anda hanya ingin tahu apakah sebuah model cukup pintar untuk membalas instruksi.

#### Sang Raja GGUF

*Yup*, LMStudio mirip seperti Ollama, mesin keduanya dibangun di atas teknologi bernama llama.cpp yang sangat teroptimasi untuk menjalankan model dengan format kuantisasi bernama GGUF.

Teknik kuantisasi tersebut yang membuatnya dapat menjalankan LLM belasan Gigabyte di laptop biasa yang tidak punya GPU. Namun, keajaiban sebenarnya ada pada fitur GPU Offloading. Jika laptop Anda memiliki RAM 16 GB tetapi GPU-nya hanya memiliki VRAM 4 GB, LMStudio akan secara otomatis membagi beban kerjanya. Ia akan meletakkan sebagian model AI di VRAM GPU agar cepat, dan sisanya dilempar ke RAM CPU biasa.

Berkat mekanisme pembagian otomatis ini, Anda bahkan bisa menjalankan model Llama-3 yang cukup besar dengan mulus di laptop Windows biasa atau MacBook.

#### Drop-in Replacement

Meskipun tampilannya seperti aplikasi chat biasa, fitur terkuat LMStudio sebenarnya tersembunyi di tab Local Server.

Hanya dengan menekan satu tombol "Start Server", LMStudio akan menyalakan API localhost di laptop Anda. Bagian "seksinya" adalah: API yang dipancarkan oleh LMStudio ini 100% kompatibel dengan standar format API milik OpenAI.

Apa artinya bagi Anda sebagai developer?

Bayangkan Anda sedang membangun kode Frontend atau Backend RAG menggunakan library resmi OpenAI di Python atau Node.js, dan kode Anda awalnya ditujukan untuk menembak API ChatGPT yang berbayar.

*Nah*, Anda cukup mengubah satu baris base URL di kode Anda menjadi `http://localhost:1234/v1`. *And Boom!* Aplikasi Anda sekarang berkomunikasi dengan model LLM gratis yang berjalan secara lokal tanpa perlu mengubah satu baris pun logika kode Anda yang lain.

Dengan segala kemudahan dan fitur canggihnya, ada satu batasan mutlak yang harus Anda tanamkan di kepala, yaitu LMStudio bukan dirancang untuk production publik skala besar.

Jika Anda memaksa men-*deploy* LMStudio ke server kantor dan membiarkan 100 User mengirimkan request secara bersamaan, LMStudio akan langsung kewalahan. Hal ini karena ia akan memproses antrean secara sekuensial atau satu per satu, membuat pengguna ke-100 harus mengantri untuk mendapatkan jawaban dari model.

### vLLM

Di dunia rekayasa AI saat ini, vLLM memegang takhta sebagai standar untuk *deployment production* di banyak company. Jika melihat sebuah perusahaan baik itu startup maupun enterprise menyewa klaster GPU mahal seperti NVIDIA A100 atau H100, Anda bisa bertaruh 90% kemungkinan mereka menjalankan vLLM di balik layar.

*Alright*, kalimat di atas mungkin terdengar sesumbar, tetapi mari kita lihat statistik berikut.

![dos-7a5db7ecc2ae85d6725e5868de87f88320260430162549.png](https://assets.cdn.dicoding.com/original/academy/dos-7a5db7ecc2ae85d6725e5868de87f88320260430162549.png)

Itu bukan grafik jumlah download, tetapi grafik siapa yang ikut berkontribusi mengembangkan vLLM. Anda dapat melihat daftar perusahaan raksasa di atas seperti AMD, Intel, dan Roblox ikut turun tangan menulis kodenya.

Apa artinya?

Perusahaan-perusahaan raksasa tersebut telah bertaruh bahwa vLLM adalah masa depan sehingga mereka rela membuang waktu engineer mahal mereka untuk berkontribusi pada proyek ini. Tentunya *willingness* tersebut berasal dari pengalaman baik mereka saat menggunakan vLLM dalam lingkungan *production*-nya.

![dos-997479f0902975b4383ba87d3ac40d3720260430194519.gif](https://assets.cdn.dicoding.com/original/academy/dos-997479f0902975b4383ba87d3ac40d3720260430194519.gif)

*   **LinkedIn:** Sejak mengadopsi vLLM, Roblox berhasil menurunkan latensi hingga 50% saat melayani 4 miliar token setiap minggu.
*   **Amazon:** Rufus dilaporkan telah melayani 250 juta pelanggan pada tahun 2025. Jumlah tersebut terus bertambah berkat penggunaan teknik continuous batching dari vLLM yang efisien.

Namun, perlu Anda ketahui bahwa karakteristik utama vLLM sangat kontras dengan LMStudio sebelumnya. Ia tidak memiliki antarmuka visual (GUI) sama sekali dan murni dijalankan dari Terminal ataupun Python class. vLLM memang tidak dirancang untuk kemudahan pemula, melainkan dirancang untuk satu tujuan, yaitu High-Throughput atau memproses sebanyak mungkin pengguna dalam waktu sesingkat-singkatnya.

Meskipun dirancang sebagai low level machine seperti itu, ia tetap ramah developer. *Yup*, jika Anda lihat di file `docker-compose.yml` sebelumnya, vLLM sudah menyediakan web server bawaan berbasis FastAPI.

Mengapa vLLM bisa begitu dominan? Jawabannya terletak pada dua "sihir" rekayasa yang langsung memecahkan masalah pendekatan naif Python yang kita bahas sebelumnya dan salah satu sihirnya sudah kita *spill* pada penjelasan di atas.

#### PagedAttention untuk Efisiensi VRAM GPU

Di materi sebelumnya, kita membahas masalah KV Cache Fragmentation. Saat LLM standar men-generate kata, ia memakan blok memori VRAM secara kaku dan berantakan. Ibarat Anda menyimpan buku berseri di sebuah rak, tetapi Anda mem-*booking* ruang kosong terlalu lebar untuk buku-buku yang belum terbit. Hal ini tentu menyebabkan banyak sekali celah di rak yang terbuang sia-sia. Hal ini selaras dengan ilustrasi di bawah ini.

![dos-8eeef36e907221e1cdf0da58186e8edd20260430193957.png](https://assets.cdn.dicoding.com/original/academy/dos-8eeef36e907221e1cdf0da58186e8edd20260430193957.png)

*Nah*, vLLM datang membawa algoritma PagedAttention yang terinspirasi dari cara sistem OS komputer mengelola virtual memory. Algoritma tersebut akan merapikan dan memecah VRAM menjadi blok-blok kecil secara dinamis untuk memberikan ruang VRAM persis ketika kata tersebut di-*generate* oleh model.

![dos-84eb1f6e5e79e5b63ee7e20c12cc0a2520260430194200.gif](https://assets.cdn.dicoding.com/original/academy/dos-84eb1f6e5e79e5b63ee7e20c12cc0a2520260430194200.gif)

*Yup*, hasilnya tidak ada 1 MB pun VRAM yang terbuang sia-sia karena seluruh ruang VRAM memang dimanfaatkan untuk menyimpan KV dari kata yang sudah di-*generate*. Dengan manajemen memori seefisien ini, vLLM mampu melipatgandakan kapasitas throughput yang kita singgung sebelumnya hingga 20x lipat dibandingkan jika Anda menggunakan script HuggingFace biasa pada perangkat keras yang sama.

#### Continuous Batching

Pada sistem konvensional bernama Static Batching, jika ada 3 User mengirim prompt bersamaan, GPU akan mengunci ketiganya dalam satu matriks. Jika User ke-1 dan ke-2 meminta jawaban pendek, sementara User ke-3 meminta esai panjang, sistem harus menahan User ke-4 masuk sampai esai User ke-3 benar-benar selesai diketik oleh AI yang membuat GPU banyak menganggur atau idle.

*Nah*, vLLM menggunakan Continuous Batching untuk menyelesaikan permasalahan pada static batching tersebut. *Alright* agar lebih mudah, mari kita gunakan ilustrasi berikut untuk menjelaskan alasannya.

![dos-6f8eb62fd4d88881ebcf4989ab2ee43a20260430162617.png](https://assets.cdn.dicoding.com/original/academy/dos-6f8eb62fd4d88881ebcf4989ab2ee43a20260430162617.png)

*   **Static Batching:** GPU harus menahan seluruh sumber daya komputasinya hingga kalimat (*sequence*) terpanjang dalam satu kelompok (*batch*) selesai diproses, sehingga banyak ruang komputasi terbuang menganggur ketika kalimat yang lebih pendek sudah selesai.
*   **Continuous Batching:** Langsung menyisipkan permintaan pengguna baru (seperti *S5*, *S6*, dan *S7*) ke dalam batch tepat pada saat sequence sebelumnya selesai di-*generate*, memastikan utilitas GPU tetap padat dan maksimal di setiap langkah waktu tanpa harus menunggu seluruh batch awal rampung.

Hebatnya lagi, endpoint API dari vLLM ini juga sengaja dibuat 100% kompatibel dengan standar API OpenAI. Artinya, sama seperti LMStudio, Anda bisa mengganti backend aplikasi dari API ChatGPT berbayar ke vLLM internal perusahaan Anda tanpa perlu merombak kode aplikasi Frontend.

### SGLang

Posisi SGLang (Structured Generation Language) di industri adalah sebagai teknologi cutting-edge yang lahir dari tim peneliti yang beririsan dengan pembuat vLLM, yaitu UC Berkeley dan Stanford. Saat ini, SGLang sering kali mengalahkan kecepatan vLLM dalam berbagai *benchmark* pengujian, menjadikannya pilihan utama bagi tim engineering yang mengejar performa ekstrem.

![dos-dc8ca6ce009a3867b11103d4ae96793b20260430162629.png](https://assets.cdn.dicoding.com/original/academy/dos-dc8ca6ce009a3867b11103d4ae96793b20260430162629.png)

Pada dasarnya, SGL juga menggunakan PagedAttention dan Continuous Batching sebagai fondasi. Hanya saja, SGLang membawa tambahan inovasi arsitektur yang sangat agresif. Berikut adalah fitur andalan yang membuat SGLang begitu cepat.

#### RadixAttention untuk Prompt Caching

Ini adalah fitur paling revolusioner dan menjadi andalan SGLang. Dalam aplikasi dunia nyata, banyak bagian dari prompt yang sebenarnya berulang-ulang. Misalnya, aplikasi Anda memiliki System Prompt sepanjang 500 kata *"Kamu adalah AI asisten toko sepatu, aturan membalasnya adalah..."*. Nah, jika ada 100 User membuka aplikasi Anda secara bersamaan, vLLM standar akan melakukan kalkulasi pada 500 kata tersebut sebanyak 100 kali secara terpisah.

SGLang menggunakan struktur data memori bernama Radix Tree yang akan "mengingat" hasil komputasi dari System Prompt tersebut di VRAM.

![dos-ad9eb3b5484745fd2cf292f67ebb1d7720260504111727.gif](https://assets.cdn.dicoding.com/original/academy/dos-ad9eb3b5484745fd2cf292f67ebb1d7720260504111727.gif)

Ilustrasi di atas menunjukkan cara SGLang mengatur memori percakapan layaknya pohon silsilah dari waktu ke waktu.

*   **Berbagi Akar (Langkah 4)**
    Perhatikan blok "*You are a helpful assistant*". Jika ada dua percakapan (User) yang berbeda di dalam server, SGLang mengenali bahwa awalan prompt mereka sama persis. AI tidak akan menyimpan kalimat dasar itu dua kali di memori. Ia menyimpannya sekali di "akar", lalu bercabang menjadi dua percakapan berbeda.
*   **Bersih-bersih Memori atau Eviction (Langkah 5, 8, dan 9)**
    Memori GPU itu terbatas. Jika memori mulai penuh, SGLang harus membuang ingatan lama (ditandai dengan kotak oranye putus-putus bertuliskan "evicted" atau dibuang). Kehebatan RadixAttention adalah ia tahu persis cabang mana yang paling jarang dipakai untuk dibuang (misalnya percakapan yang sudah selesai), sambil tetap mempertahankan "akar" instruksi utama yang sering dipakai banyak pengguna.

Jika kita analogikan penjelasannya, kurang lebih akan seperti ini. Bayangkan Anda sedang membaca buku *"Choose Your Own Adventure”*. Anda membaca Bab 1, 2, dan 3. Di akhir Bab 3, ceritanya bercabang: Anda bisa memilih masuk ke "Gua Gelap" (Opsi A) atau "Hutan Terlarang" (Opsi B).

Jika Anda memilih Opsi A dan ceritanya tamat, lalu Anda ingin mencoba Opsi B, apakah Anda akan membaca Bab 1, 2, dan 3 dari awal lagi? Tentu tidak. Otak Anda sudah "mengingat" cerita sampai Bab 3, jadi Anda tinggal melanjutkannya langsung ke Opsi B.

*   **vLLM:** Cenderung "membaca ulang" dari Bab 1 untuk setiap skenario baru, meskipun awalan teksnya persis sama.
*   **SGLang:** Mengingat "Bab 1-3" dan membagikan ingatan tersebut ke semua skenario yang membutuhkannya.

Perlu Anda ketahui, Fitur RadixAttention tersebut juga dapat dimanfaatkan pada sistem RAG yang kita pelajari sebelumnya. Saat Anda punya dokumen SOP Perusahaan sepanjang 10 halaman lalu ada 5 karyawan bertanya tentang SOP yang sama di waktu berdekatan, SGLang akan men-cache dokumen 10 halaman tersebut di GPU. SGLang hanya akan membaca dokumen itu 1 kali, tetapi bisa menjawab 5 pertanyaan berbeda darinya secara paralel dalam hitungan milidetik.

*Alright*, dari tadi kita memuja SGLang dan bahkan sempat menyebut bahwa ia memiliki throughput yang jauh lebih besar dibandingkan. Lantas apakah berarti kita harus selalu memilih SGLang ketimbang vLLM.

#### Apakah SGLang Selalu Mengalahkan vLLM?

*Nah*, ini tentunya adalah pertanyaan yang menarik.

SGLang memang menunjukkan throughput (jumlah token yang dihasilkan per detik) yang secara signifikan lebih tinggi dibandingkan vLLM pada berbagai skenario panjang input/output untuk Llama-70B.

Namun, untuk menyimpulkan bahwa semua orang harus langsung pindah ke SGLang rasanya terlalu terburu-buru. Dalam dunia deployment machine learning, memilih inference engine jarang sekali hanya didasarkan pada satu metrik.

Berikut adalah beberapa realitas teknis mengapa vLLM masih sangat populer dan migrasi tidak selalu menjadi jawaban mutlak.

*   **Ekosistem dan Komunitas:** vLLM saat ini merupakan salah satu standar industri yang paling mapan. Komunitasnya sangat masif, dokumentasinya lengkap, dan integrasinya dengan berbagai platform sangat stabil.
*   **Dukungan Model Baru:** Karena komunitasnya yang besar, setiap kali ada arsitektur model AI baru yang dirilis, vLLM biasanya menjadi yang pertama mendapatkan pembaruan untuk mendukung model tersebut.
*   **Extreme Concurrency:** Ketika traffic melonjak tinggi (misalnya 20 hingga 50+ request bersamaan), vLLM justru mencetak throughput puncak 11-15% lebih tinggi dan menjaga latensi tetap stabil. Pada SGLang, latensi awal (TTFT) sering kali melonjak tajam saat antrean bertambah.
*   **Long Context:** Jika use case Anda adalah memproses dokumen tebal secara batch (seperti merangkum puluhan PDF sekaligus), vLLM lebih tangguh. Pengujian menunjukkan vLLM bisa 2x lebih cepat dibandingkan SGLang untuk pemrosesan batch dengan konteks di atas 8.000 token.

Dari sini mungkin kita bisa membuat panduan singkat, yaitu gunakan vLLM jika Anda membangun API publik berskala besar untuk ribuan pengguna acak dengan permintaan tunggal. Gunakan SGLang jika Anda membangun asisten pintar yang butuh memori percakapan dengan histori panjang atau sistem agen AI yang kompleks.

Tahapan deployment ini sekaligus menutup seluruh rangkaian materi dalam kelas Large Language Models. Sepanjang program ini, kita telah membedah secara komprehensif siklus hidup LLM dari hulu ke hilir. Kita telah membahas arsitektur fundamental yang membentuk model bahasa, teknik rekayasa yang diperlukan untuk memanipulasi keluaran teks, hingga arsitektur penyediaan layanan (serving) di lingkungan produksi sesungguhnya.

Penguasaan dari tahap eksplorasi awal hingga tahap operasionalisasi akhir ini memberikan Anda fondasi teknis yang solid untuk merancang dan memelihara produk berbasis AI secara mandiri. Mengingat ekosistem teknologi kecerdasan buatan berkembang dengan sangat cepat, pastikan Anda terus memantau pembaruan dokumentasi dari pustaka-pustaka inti yang telah kita gunakan, serta terus menguji arsitektur model-model terbaru yang dirilis ke publik.