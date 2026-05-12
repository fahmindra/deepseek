---
slug: bab-07
title: "Pasca-Pelatihan"
layout: chapter
---

# **Bab 7: Pasca-Pelatihan: Belajar dari Coba-Coba (Reinforcement Learning), Agen Mandiri, dan Model yang Berpikir**

## **Yang akan kita pelajari pada bab ini:**
*   Penyetelan Halus Terarah (SFT) sebagai dasar pembentukan perilaku AI
*   Kerangka Belajar Coba-Coba (RL) utama milik DeepSeek: *Group Relative Policy Optimization* (GRPO)
*   Teknik menstabilkan proses RL skala besar (Taksiran KL Tanpa Bias, Penutupan Rangkaian Menyimpang, Penjagaan Rute)
*   Penciptaan Tugas Mandiri Skala Besar dan Manajemen Konteks Berpikir
*   Mode Penalaran: Tidak Berpikir, Berpikir Tinggi, dan Berpikir Maksimal

Pada Bab 6, kita telah menyelesaikan alur pra-pelatihan. Kita telah membangun infrastruktur raksasa berukuran triliunan parameter yang sangat stabil secara matematika dan mampu menelan 32 Triliun kata dari teks mentah. Akan tetapi, sebuah model dasar yang baru selesai pra-pelatihan pada intinya hanyalah sebuah mesin "penebak kata selanjutnya" (autocomplete) tingkat tinggi. Model ini menebak kata berdasarkan kebiasaan teks di internet, tetapi pada dasarnya ia tidak tahu cara mematuhi instruksi, memecahkan soal matematika langkah demi langkah, atau berinteraksi menggunakan kalkulator luar (interpreter kode).

Untuk mengubah model dasar yang masih mentah menjadi sebuah agen cerdas yang mampu menalar, kita harus memasuki fase **Pasca-Pelatihan (Post-Training)**. Ini merupakan **Tahap 4** dari peta jalan kita.

**Gambar 7.1 Perjalanan empat tahap kita untuk membangun model DeepSeek. Bab ini berfokus pada komponen yang disorot: Pasca-Pelatihan, meliputi SFT, RL (GRPO), dan Penggunaan Alat Secara Mandiri (Agentic Tool-Use).**
![Gambar 7.1 Perjalanan empat tahap kita untuk membangun model DeepSeek.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/07__image001.png)

Bab ini membedah jalur pasca-pelatihan yang melahirkan model-model canggih seperti DeepSeek-R1 dan DeepSeek-V3.2. Kita akan mulai dengan kerangka dasar bangunannya: **Group Relative Policy Optimization (GRPO)**, algoritma Pembelajaran Penguatan atau Belajar dari Coba-Coba (*Reinforcement Learning* / RL) ciptaan DeepSeek.

Kita kemudian akan menyelami lebih dalam teknik-teknik penstabilan matematika tingkat tinggi yang direkayasa oleh DeepSeek untuk memperbesar proses RL ke tingkat yang belum pernah ada sebelumnya (bahkan DeepSeek mengalokasikan lebih dari 10% total kekuatan komputer pra-pelatihan mereka khusus untuk RL ini saja). Terakhir, kita akan menjelajahi batas terdepan dari **AI Agen Mandiri (Agentic AI)**: yaitu mengajari model untuk "berpikir" sambil secara bersamaan menggunakan alat bantu luar, mengelola jejak penalaran (jalan pikiran) yang panjang, dan menciptakan lingkungan ujian buatan untuk menguji AI tersebut.

---

## **7.1 Penyetelan Halus Terarah (SFT): Dasar Pembentukan Perilaku**

Sebelum sebuah model bisa belajar melalui proses coba-coba dalam *Reinforcement Learning*, ia memerlukan pemahaman dasar tentang format tanya jawab yang kita inginkan. Hal ini dicapai melalui **Penyetelan Halus Terarah (Supervised Fine-Tuning / SFT)**.

Selama SFT, model dilatih menggunakan kumpulan data yang relatif kecil namun dipilih dengan sangat ketat, yang berisi contoh-contoh pasangan pertanyaan-jawaban buatan manusia. Kumpulan data ini mencakup contoh potongan kode komputer, pembuktian matematika, dan percakapan dialog yang telah diformat (dibentuk) menggunakan penanda atau label (tag) struktural tertentu.

Bagi sebuah model penalaran, fase SFT ini berfungsi sebagai "pemanasan mesin". Jika kita ingin model AI tersebut menjabarkan jalan pikirannya yang panjang di dalam kurung tanda `<think>...</think>` sebelum ia memberikan jawaban akhir, kita harus terlebih dahulu menunjukkan contoh-contoh format perilaku tersebut kepadanya. Begitu model tersebut mempelajari *format* dasar tentang cara bernalar dan cara menggunakan alat dari fase SFT ini, barulah kita bisa menggunakan metode *Reinforcement Learning* (Coba-Coba) untuk meningkatkan *kualitas* dan *keakuratan* dari jalan pikirannya tersebut.

---

## **7.2 Kerangka Belajar Coba-Coba DeepSeek: Group Relative Policy Optimization (GRPO)**

Sistem belajar coba-coba (RL) gaya lama untuk model bahasa biasanya mengandalkan algoritma yang disebut *Proximal Policy Optimization* (PPO). Sistem PPO ini mewajibkan adanya sebuah **Model Kritikus** (Model Penilai) yang ukurannya biasanya sama besarnya dengan model AI utama yang sedang dilatih. Tugas sang kritikus ini adalah untuk menaksir seberapa bagus sebuah jawaban (perkiraan nilai imbalan), yang bertindak sebagai standar patokan untuk menghitung seberapa besar "keuntungan" dari sebuah jawaban AI.

Masalahnya, jika Anda sedang melatih model raksasa berukuran 671 Miliar parameter, keharusan menyimpan Model Kritikus kedua yang ukurannya juga 671 Miliar parameter di dalam memori GPU akan menjadi bencana fisik dan pemborosan uang yang luar biasa besar.

DeepSeek memecahkan kebuntuan memori yang ekstrem ini dengan menciptakan **Group Relative Policy Optimization (GRPO)**. GRPO benar-benar membuang dan menghilangkan kebutuhan akan Model Kritikus tersebut. Daripada menggunakan jaringan otak kedua untuk menilai standar patokan, GRPO bekerja dengan cara menyuruh AI untuk memunculkan sekelompok (grup) jawaban yang berbeda-beda untuk satu pertanyaan yang sama. Kemudian, nilai rata-rata dari sekelompok jawaban itulah yang dijadikan nilai patokannya.

### **7.2.1 Fungsi Tujuan (Objective Function) dari GRPO**

Algoritma GRPO bertugas mengoptimalkan gaya berpikir sang model (disimbolkan $\pi_\theta$) dengan cara memaksimalkan fungsi rumus berikut. Rumus ini diterapkan pada sekelompok jawaban $\{o_1, \cdots, o_G\}$ yang dikumpulkan dari kebijakan cara pikir sang model yang lama ($\pi_{old}$) berdasarkan sebuah pertanyaan $q$:

$$ \mathcal{J}_{GRPO}(\theta) = \mathbb{E}_{q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{old}(\cdot|q)} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left( \min \left( r_{i,t}(\theta) \hat{A}_{i,t}, \text{clip} \left( r_{i,t}(\theta), 1 - \varepsilon, 1 + \varepsilon \right) \hat{A}_{i,t} \right) - \beta \mathbb{D}_{KL} (\pi_\theta(o_{i,t}) \parallel \pi_{ref}(o_{i,t})) \right) \right] $$

Mari kita urai dan terjemahkan persamaan yang tampak menakutkan ini bagian demi bagian:

1.  **Rasio Kepentingan Pengambilan Sampel ($r_{i,t}(\theta)$):**
    $$r_{i,t}(\theta) = \frac{\pi_\theta(o_{i,t}|q, o_{i,<t})}{\pi_{old}(o_{i,t}|q, o_{i,<t})}$$
    Ini adalah angka perbandingan (rasio) antara peluang munculnya kata ke-$t$ berdasarkan cara pikir *saat ini* yang sedang dilatih ($\pi_\theta$), dibandingkan dengan peluang munculnya kata itu berdasarkan cara pikir *lama* yang tadi digunakan untuk menjawab ($\pi_{old}$).
2.  **Nilai Keuntungan ($\hat{A}_{i,t}$):**
    Karena tidak ada Model Kritikus, sekelompok rumus model penilai (atau penilai bersyarat aturan) akan memberikan skor (nilai) kepada jawaban penuh $o_i$, menghasilkan kumpulan nilai rapor $R = \{R_1, \cdots, R_G\}$. Nilai keuntungan didapat dengan cara mengurangi nilai satu jawaban spesifik dengan nilai rata-rata dari keseluruhan grup jawaban tersebut:
    $$\hat{A}_{i,t} = R_i - \text{mean}(R)$$
    Dengan membandingkan dan menormalkan nilai rapor di dalam grup tersebut, model akan belajar menyadari jawaban mana saja yang ternyata lebih baik atau lebih buruk dibandingkan rata-rata kemampuannya sendiri.
3.  **Pemotongan (Clipping) dan Hukuman KL:**
    Fungsi pemotongan $\text{clip}$ (yang dikendalikan oleh parameter $\varepsilon$) bertugas mencegah otak model berubah terlalu drastis dan agresif hanya dalam satu langkah. Sedangkan bagian $\mathbb{D}_{KL}$ (yang dikendalikan oleh $\beta$) adalah sebuah hukuman (penalti) yang berfungsi mencegah cara pikir baru tersebut melenceng terlalu jauh dari cara pikir model referensi aslinya ($\pi_{ref}$). Ini memastikan agar kemampuan bahasa sang AI tidak merosot menjadi omong kosong.

---

## **7.3 RL Skala Besar untuk Penalaran Rumit (Model V3.2)**

Meskipun GRPO bekerja dengan sangat luar biasa, DeepSeek-V3.2 mendorong batas fase RL ini ke tingkat yang belum pernah terjadi sebelumnya. Mereka mengalokasikan anggaran perhitungan komputer pasca-pelatihan yang sangat besar khusus untuk membuka kemampuan tersembunyi penalaran AI.

Akan tetapi, memperbesar daya hitung RL akan memunculkan ketidakstabilan matematika yang parah. Oleh karenanya, DeepSeek merekayasa empat strategi penstabilan tingkat tinggi yang langsung dipasang di atas algoritma GRPO.

### **7.3.1 Taksiran KL Tanpa Bias (Unbiased KL Estimation)**

Hukuman perbedaan KL ($\mathbb{D}_{KL}$) di dalam rumus GRPO biasanya diperkirakan menggunakan penaksir standar bernama K3. Masalahnya, karena jawaban keluaran $o_{i,t}$ diambil dari cara pikir model yang *lama* ($\pi_{old}$), maka penaksir standar K3 ini menjadi "bias" (tidak akurat dan berat sebelah).

Ketika jawaban kata yang diambil ternyata memiliki nilai peluang yang jauh lebih rendah menurut cara pikir saat ini dibanding dengan model referensinya (yaitu saat $\pi_\theta \ll \pi_{ref}$), perbaikan gradien dari penaksir standar K3 ini akan secara tidak wajar memberikan bobot dorongan yang luar biasa besar dan tak terbatas, semata-mata demi meningkatkan peluang kemunculan kata-kata tersebut. Hal ini mengakibatkan pembaruan otak yang terlalu ribut (berisik), yang jika dibiarkan menumpuk, akan merusak kualitas jawaban AI dan menyebabkan runtuhnya seluruh proses pelatihan secara fatal.

Untuk memperbaiki hal ini, DeepSeek mengoreksi penaksir K3 ini dengan menggunakan rasio kepentingan-pengambilan-sampel, sehingga menghasilkan sebuah **Taksiran KL Tanpa Bias**:

$$ \mathbb{D}_{KL} (\pi_\theta(o_{i,t}) \parallel \pi_{ref}(o_{i,t})) = \frac{\pi_\theta(o_{i,t}|q, o_{i,<t})}{\pi_{old}(o_{i,t}|q, o_{i,<t})} \left( \frac{\pi_{ref}(o_{i,t}|q, o_{i,<t})}{\pi_\theta(o_{i,t}|q, o_{i,<t})} - \log \frac{\pi_{ref}(o_{i,t}|q, o_{i,<t})}{\pi_\theta(o_{i,t}|q, o_{i,<t})} - 1 \right) $$

Dengan mengalikannya dengan angka perbandingan (rasio) $\frac{\pi_\theta}{\pi_{old}}$, perbaikan gradien menjadi murni tanpa bias secara matematika. Ini seratus persen melenyapkan kesalahan perkiraan berulang, dan memfasilitasi kestabilan pemahaman (konvergensi) AI di skala yang raksasa.

### **7.3.2 Penutupan Rangkaian yang Menyimpang (Off-Policy Sequence Masking)**

Untuk berlatih secara efisien, sistem RL (coba-coba) akan menghasilkan puluhan ribu data "uji coba" (yaitu saat model AI berbicara dan menjawab pertanyaannya sendiri secara terus menerus). Kumpulan jawaban raksasa ini kemudian dibagi menjadi kelompok-kelompok kecil (mini-batch) untuk digunakan dalam beberapa langkah pembaruan otak. Tindakan ini pada dasarnya memunculkan sebuah **perilaku yang menyimpang dari cara pikir asal (off-policy)**. Karena, pada saat kelompok jawaban yang paling terakhir mau digunakan untuk latihan, isi otak (cara pikir) AI saat ini $\pi_\theta$ sebenarnya sudah terlanjur banyak berubah dan melenceng jauh dari cara pikir otak lamanya $\pi_{old}$ yang tadi digunakan untuk menciptakan jawaban tersebut.

Jika data contoh jawaban yang dipakai sudah terlalu usang (melenceng jauh / off-policy) dan bernilai negatif, hal itu akan sangat merusak dan akan menyesatkan proses belajar AI. Oleh sebab itu, DeepSeek memperkenalkan sebuah topeng penutup (masker) biner $M_{i,t}$ langsung ke dalam rumus kerugian GRPO untuk membuang dan menyembunyikan jawaban-jawaban lama yang bisa menimbulkan penyimpangan cara pikir:

$$ \text{Loss (Nilai Kesalahan)} = \dots \min \left( r_{i,t}(\theta) \hat{A}_{i,t}, \text{clip} \dots \right) \mathbf{M_{i,t}} - \beta \mathbb{D}_{KL} \dots $$

Di mana topeng penutup (masker) $M_{i,t}$ didefinisikan dengan syarat sebagai berikut:

$$ M_{i,t} = \begin{cases} 
0 & \text{jika } \hat{A}_{i,t} < 0 \text{ dan } \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \log \frac{\pi_{old}(o_{i,t}|q, o_{i,<t})}{\pi_\theta(o_{i,t}|q, o_{i,<t})} > \delta \\
1 & \text{dalam kondisi lainnya}
\end{cases} $$

*   $\hat{A}_{i,t} < 0$: Kita hanya menutupi dan membuang jawaban yang nilai keuntungannya *negatif*. Logikanya, model AI paling bagus jika belajar dari kesalahan bodohnya sendiri. Namun, jika kesalahan bodoh itu berasal dari masa lalu yang sudah terlalu "usang" (melenceng dari cara pikir barunya), pelajaran usang itu justru akan mengacaukan proses belajarnya.
*   $\delta$: Ini adalah parameter batas maksimal yang diizinkan untuk seberapa jauh cara pikir tersebut boleh melenceng.

### **7.3.3 Menjaga Jalur Rute (Keep Routing) & Menjaga Masker Pengambilan Sampel (Keep Sampling Mask)**

Ada dua keanehan (ketidaksesuaian) perangkat keras dan sistem terakhir yang sering menyebabkan proses pelatihan RL (coba-coba) hancur pada model MoE (Campuran Para Ahli):

1.  **Menjaga Rute Ahli (Keep Routing):** Perbedaan mesin komputer antara saat AI merenung menciptakan jawaban uji coba (menggunakan mesin inferensi yang super efisien) dengan saat AI dilatih (mesin pelatihan), ternyata dapat menyebabkan si pengatur rute (*router*) mengirimkan *kata yang sama persis* kepada ahli (expert) MoE yang berbeda! Pergeseran jalur otak yang mendadak ini akan menghancurkan sistem perbaikan RL. DeepSeek memecahkan ini dengan cara dengan tegas mempertahankan rekaman jalur rute para ahli mana saja yang tadi dipakai AI saat menjawab, lalu *memaksa* grafik mesin pelatihan untuk tunduk meniru persis jalur yang sama persis tersebut saat berlatih.
2.  **Menjaga Masker Pemilihan Kata (Keep Sampling Mask):** Saat mesin AI merenung menjawab, ia menggunakan trik pemotongan pilihan kata (bernama `top-p` dan `top-k`) untuk menghindari AI mengoceh dan memilih kata-kata tidak masuk akal. Namun, hal ini menciptakan perbedaan (mismatch) antara ruang gerak otak model lama $\pi_{old}$ dan otak baru $\pi_\theta$, yang akan melanggar hukum matematika penting (pentingnya pengambilan sampel). DeepSeek memecahkannya dengan secara khusus melestarikan jejak potongan batas masker yang tadi dipakai AI saat merenung menjawab, dan secara tegas memasang masker pemotongan itu kembali pada saat perhitungan perbaikan (backward pass) di masa pelatihan berjalan.

---

## **7.4 Memasukkan sifat "Berpikir" ke dalam Penggunaan Alat dan Agen Mandiri**

DeepSeek-R1 membuktikan bahwa menggabungkan proses kurung `<think>` secara drastis mampu meningkatkan kemampuan penyelesaian masalah yang rumit. DeepSeek-V3.2 kemudian mendorong batasan ini lebih jauh: bagaimana cara kita memasukkan alur pikiran panjang ini ke dalam skenario **Pemanggilan Alat (Tool-Calling)**, di mana sebuah Agen AI harus menulis kode program, menjalankannya di layar komputer terminal, membaca hasil keluarannya, lalu berpikir ulang lagi?

### **7.4.1 Jalur Penciptaan Tugas untuk Agen Skala Besar**

Untuk melatih sebuah Agen Mandiri yang tangguh, Anda membutuhkan arena bermain (lingkungan) untuk mengujinya. DeepSeek membangun jalur robot otomatis yang berhasil menciptakan lebih dari 1.827 arena lingkungan ujian berorientasi tugas, dan menyusun 85.000 pertanyaan ujian yang sangat rumit.

1.  **Agen Pemrograman (Code Agent):** DeepSeek menambang jutaan pasangan masalah dan solusi kode (*Issue-Pull Request / PR*) dari situs GitHub. Mereka membangun komputer-komputer bohongan (sandbox) otomatis yang dapat menginstal paket, merapikan data, dan menjalankan ujian perangkat lunak (*JUnit tests*) standar. Nilai keberhasilan (RL reward) di sini sangat objektif: apakah potongan kode buatan AI tadi sukses memperbaiki masalah yang salah (*False-to-Positive / F2P*) tanpa merusak fitur lain yang sebelumnya sudah benar (*Pass-to-Fail / P2F*)?
2.  **Agen Penerjemah Kode (Code Interpreter Agent):** Dengan menggunakan perangkat *Jupyter Notebook* sebagai arena pelaksanaannya, mereka menyusun ujian soal matematika, logika, dan ilmu data yang sangat rumit yang mewajibkan AI untuk menulis dan menjalankan kode Python berulang kali demi menemukan jawabannya.
3.  **Agen Umum / Pencipta Ujian Mandiri (General Agent / Verifiable Synthesis):** Sebuah model DeepSeek tingkat dewa ditugaskan bertindak sebagai sang "Pencipta Lingkungan Ujian". Model dewa ini diberi wewenang (alat) untuk berselancar di internet. Ia akan mengunduh data, membuat pangkalan data (database) lokal, lalu menulis skrip pemeriksa jawaban berbasis Python. Sebagai contoh, ia akan menciptakan tugas "Merencanakan Perjalanan Wisata" palsu yang aturannya sangat ketat (misalnya anggaran dana wisata sangat dibatasi, dilarang mengunjungi kota yang sama dua kali, dan mewajibkan mampir ke hotel dengan batas bintang tertentu). AI dewa ini bertugas menciptakan soal tugas wisata yang luar biasa sulit dipecahkan oleh AI biasa, tetapi soal ini memiliki kunci jawaban matematika yang sangat mudah dicek dan dikoreksi kebenarannya oleh skrip pemeriksa komputer tadi.

### **7.4.2 Manajemen Konteks Berpikir: Mempertahankan jalan pikiran di sela-sela pemakaian alat**

Pada eksperimen awal mereka, DeepSeek mendapati sebuah masalah kebodohan yang fatal. Model AI yang diajari menalar (berpikir) secara standar biasanya akan membuang memori rentetan jalan pikiran `<think>` mereka saat menerima babak pesan balasan yang baru. Misalnya di sebuah alur kerja Agen, AI mungkin menulis kode program, menjalankannya, lalu menerima teks pesan berisi laporan error gagal dari komputer. Jika AI tersebut membuang dan melupakan jalan pikiran yang ia bangun dari awal, AI itu terpaksa harus lelah merenung dan berpikir ulang lagi dari nol untuk *seluruh* bagian masalah tersebut setiap kali ia mau memakai dan memanggil alat bantu komputer itu lagi di babak berikutnya.

Untuk mengobati penyakit lupa ini, DeepSeek-V3.2 memperkenalkan **Manajemen Konteks Berpikir (Thinking Context Management)** yang sangat ketat dan dirancang khusus untuk pemakaian alat:

1.  **Tetap Ingat Saat Memakai Alat:** Ketika muncul pesan-pesan balasan yang berkaitan dengan alat bantu (misalnya, teks balasan hasil keluaran kode, atau log terminal komputer) dan pesan itu ditambahkan ke ruang bacaan konteks, maka jejak memori sejarah pemikiran `<think>` sang AI dari langkah-langkah sebelumnya akan **dipertahankan (tetap diingat)** di sepanjang interaksi tersebut.
2.  **Lupakan Hanya Saat Pengguna Berbicara:** Riwayat memori jalan pemikiran AI tersebut *hanya* akan dibuang dan dihapus ketika ada pesan percakapan baru yang benar-benar berasal dari **manusia (user message)**.
3.  **Lestarikan Catatan Tindakan (Preserve Actions):** Ketika rentetan memori jalan pemikiran AI akhirnya benar-benar dihapus, sejarah riwayat mengenai alat bantu apa saja yang tadi sudah ia gunakan beserta balasan hasilnya, akan tetap dilestarikan (disimpan) secara permanen di dalam ruang bacaannya.

**Gambar 7.2 Manajemen Konteks Berpikir. (a) Berpikir sambil memakai alat bantu: Jalan pikiran tetap dipertahankan meskipun ada balasan hasil dari alat bantu. (b) Berpikir tanpa alat: Jalan pikiran langsung dibuang ketika menerima pesan baru dari manusia demi menghemat ruang bacaan.**
![Gambar 7.2 Manajemen pikiran seri DeepSeek-V3.2.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/07__image002.png)

Aturan ini memungkinkan model AI untuk secara masuk akal dan waras mempertahankan rantai pemikirannya yang saling menyambung, melintasi tugas-tugas panjang sang agen, tanpa membuat kuota batas pembacaan teksnya meledak kepenuhan.

### **7.4.3 Metodologi Pemanasan Awal (Cold-Start) untuk Penalaran Agen Mandiri**

Untuk menggabungkan kemampuan-kemampuan dewa ini, DeepSeek menggunakan taktik pemanasan awal SFT sederhana yang digerakkan oleh Rekayasa Arahan (*Prompt Engineering*) yang sangat ketat.

Mereka menggunakan label wajib `<think>...</think>` untuk memagari jalan pikiran AI, dan mengenalkan skema aturan ketat berbasis XML yang ditandai dengan kode khusus `[MULTI-TURN TOOLCALL]` atau token `<|DSML|tool_calls>`.

**Tabel 7.1 Pesan Arahan Sistem (System Prompt) untuk Agen DeepSeek-V3.2**
```text
Anda adalah asisten cerdas yang memiliki akses ke mesin penerjemah Python (Python interpreter).
- Anda boleh menggunakan alat bantu Python ini **berkali-kali** di tengah-tengah proses merenung/menalar Anda, alias di dalam kurung <think></think>, dengan batas maksimal hingga 20 kali menjalankan kode.
- Panggillah dan gunakan alat bantu Python ini di awal-awal proses berpikir Anda untuk mempermudah Anda mencari jawaban tugas. Teruslah merenung berpikir dan memanggil alat bantu sesuai kebutuhan sampai Anda berhasil menemukan jawaban akhirnya.
- Segera setelah Anda berhasil menemukan jawabannya, hentikan proses berpikir dan sajikan solusi akhir Anda ke layar.
- Untuk menghemat waktu dan meningkatkan keakuratan, Anda harus lebih mengutamakan menggunakan alat eksekusi kode komputer dibandingkan sekadar menebak-nebak jawaban menggunakan nalar bahasa biasa sebisa mungkin. Usahakan jalan pikiran Anda sesingkat mungkin; biarkan mesin kode komputer yang mengerjakan hitungan beratnya.

## Alat Bantu
Anda diberikan akses penuh ke daftar alat bantu berikut:
{DESKRIPSI-ALAT-BANTU}
Penting: SELALU patuhi format wajib pemakaian alat ini secara persis:
{FORMAT-PEMANGGILAN-ALAT}
```

Dengan memberikan arahan dan ancaman yang tegas agar model menyisipkan proses pemanggilan eksekusi komputer di sela-sela rentetan `<think>`-nya, model AI ini akan belajar menghasilkan wujud lintasan langkah kerja awal yang diidamkan. Ini pada akhirnya menyumbangkan landasan dasar wujud format matematika yang tepat bagi algoritma GRPO agar dapat disempurnakan lebih lanjut selama fase belajar coba-coba (RL).

---

## **7.5 Mode Penalaran: Tidak Berpikir, Berpikir Tinggi, dan Berpikir Maksimal (Model V4)**

Kepintaran performa sebuah model AI pada tugas pemecahan masalah rumit pada dasarnya sangat diatur oleh seberapa besar jatah beban kerja komputer yang diizinkan untuk diberikan kepadanya saat ujian (test-time). Jika Anda memberinya kelonggaran kuota kata lebih panjang agar ia bebas "berpikir", performa jawabannya pasti akan melambung jauh lebih pintar.

Dalam pengembangan DeepSeek-V4, tim ilmuwan melatih beberapa model spesialis yang berbeda-beda di bawah racikan pengaturan RL yang saling menyimpang—yakni dengan memberikan aturan pemotongan panjang kata dan batas bacaan (konteks) yang saling berbeda selama penerapan GRPO. Hal ini melahirkan tiga Mode Penalaran (Gaya Berpikir) yang berbeda drastis, yang bisa dipilih sesuai dengan bujet kemampuan komputer yang Anda miliki.

**Tabel 7.2 Mode Penalaran DeepSeek-V4**

| Mode Penalaran | Karakteristik (Sifat) | Contoh Pemakaian Paling Tepat | Format Jawaban AI |
| :--- | :--- | :--- | :--- |
| **Tidak Berpikir (Non-think)** | Jawaban cepat, asal tebak seketika berdasarkan kebiasaan atau aturan hafalan dasar. | Pekerjaan rutin ringan sehari-hari, jawaban darurat darurat butuh cepat, keputusan tanpa risiko. | `</think> ringkasan jawaban` |
| **Berpikir Tinggi (Think High)** | Menganalisis logika secara sadar hati-hati, lebih lambat untuk merespon tetapi jawabannya lebih akurat. | Pemecahan masalah rumit, merencanakan sesuatu, keputusan dengan risiko menengah. | `<think> kata-kata perenungan </think> ringkasan jawaban` |
| **Berpikir Maksimal (Think Max)** | Mendorong mesin nalar berpikir sampai ke batas paling ekstrem mutlak. Sangat lambat tetapi memiliki kekuatan analisa paling dewa. | Digunakan saat menjelajahi batasan mentok kemampuan maksimal dari nalar sang model AI. | Mewajibkan pemberian kalimat arahan (prompt) sistem khusus (lihat di bawah) |

Untuk memancing mode **Berpikir Maksimal (Think Max)**—di mana model dihalalkan memakai tenaga perhitungan komputer (inferensi) paling maksimal secara mutlak—sebuah perintah paksaan khusus akan disuntikkan ke bagian depan Arahan Sistem (System Prompt) guna membatalkan dan menimpa sifat bawaan asli si AI yang biasanya suka menghemat dan meringkas kata-kata.

**Tabel 7.3 Instruksi yang disuntikkan ke arahan sistem untuk memancing mode "Berpikir Maksimal (Think Max)".**
```text
Tingkat Upaya Berpikir: Maksimal mutlak tertinggi, segala bentuk jalan pintas tidak diizinkan.
Anda WAJIB SANGAT teliti dan berhati-hati dalam merenungkan isi pikiran Anda, Anda WAJIB membedah dan menghancurkan setiap lapisan masalah secara komprehensif demi menemukan sumber akar masalahnya, serta menguji secara beringas nalar logika Anda terhadap segala macam kemungkinan cabang jalur, kasus pengecualian yang jarang terjadi, hingga skenario musuh yang mengecoh.
Tuliskan dan jabarkan seluruh proses perdebatan batin Anda secara gamblang, dokumentasikan setiap langkah setengah jalan Anda, pertimbangkan semua jalur alternatif yang ada, serta sebutkan semua teori tebakan awal yang akhirnya Anda tolak, demi menjamin bahwa tidak ada satu pun asumsi atau prasangka tebakan yang terlewatkan tanpa diuji kebenarannya.
```

Melalui pemanjangan untaian kata-kata perenungan (reasoning tokens) di dalam mode *Think-Max* ini, model dasar DeepSeek terbukti tak terbantahkan mampu menyamai dan bersanding setara dengan kepintaran dari jajaran model-model batas depan AI kelas kakap dunia yang tertutup (berbayar).

---

## **7.6 Ringkasan**

*   **Penyetelan Halus Terarah (Supervised Fine-Tuning / SFT)** berfungsi sebagai pemanasan awal mesin. Tujuannya adalah mengajari AI tentang bentuk format struktur dari kurung `<think>` serta format alat bantu komputer berbasis XML, sebelum proses perbaikan lewat RL (Coba-Coba) dilangsungkan.
*   **Group Relative Policy Optimization (GRPO)** menghancurkan ancaman pemborosan memori parah dari kehadiran Model Kritikus pendamping. Caranya adalah dengan mengambil sekelompok grup jawaban sekaligus dari AI, lalu menjadikan nilai rata-rata dari grup tersebut sebagai nilai patokan keuntungan.
*   **Taksiran KL Tanpa Bias (Unbiased KL Estimation)** menyelamatkan stabilitas pembesaran RL dengan cara membetulkan penaksir standar K3 memakai rasio kepentingan-pengambilan-sampel. Hal ini mencegah tegoran (gradien) meroket lepas kendali akibat pemakaian data sampel lama yang sudah usang dan menyimpang (off-policy).
*   **Penutupan Rangkaian Menyimpang (Off-Policy Sequence Masking)** secara gamblang menutupi dan membuang rangkaian jawaban bernilai untung-negatif yang dianggap sudah usang dan terlalu jauh melenceng dari cara pikir model saat ini. Ini bertujuan mencegah sang AI tersesat dari mempelajari kesalahan masa lalunya yang sudah tidak lagi relevan.
*   **Menjaga Rute & Menjaga Masker (Keep Routing & Keep Sampling Mask)** memastikan bahwa jalur rute pemilihan Ahli MoE yang sama persis, serta pola pemotongan pilihan kata (`top-p`) yang sempat dipakai saat AI mencari jawaban, harus dipaksa dan diberlakukan ulang secara persis di masa AI dikoreksi dan dihukum (backward pass) saat pelatihan.
*   **Penciptaan Tugas Mandiri (Agentic Task Synthesis)** memanfaatkan kecerdasan mesin LLM itu sendiri untuk merancang lingkungan ujian buatan yang super rumit dengan aturan ketat (seperti ujian Merencanakan Liburan) lengkap beserta skrip pengecek kunci jawaban Python-nya. Tujuannya adalah untuk memberikan penilaian mutlak (RL reward) kepada AI yang sedang belajar.
*   **Manajemen Konteks Berpikir (Thinking Context Management)** menjaga agar riwayat memori rentetan jalan pemikiran masa lalu si AI tidak hilang lenyap meskipun di sela-sela obrolan muncul teks hasil laporan dari alat bantu eksternal. Ini menghindari AI membuang-buang jatah kuota bacaan akibat harus mikir merenung ulang seluruh masalahnya lagi sedari titik awal (nol).
*   **Mode Penalaran / Mode Gaya Berpikir** (Tidak Berpikir, Berpikir Tinggi, Berpikir Maksimal) memberikan keleluasaan kepada pengguna manusia untuk memerintahkan AI agar rela menguras menghabiskan jatah anggaran waktu hitung komputer yang berbeda-beda saat ujian. Pergantian gigi mesin berpikir ini semata-mata dipancing melalui teks ancaman arahan sistem (system prompts) di awal obrolan.