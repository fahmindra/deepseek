---
slug: modul-4-6
title: modul-4-6
---
## 4.6 Pertimbangan dalam Memilih antara Training from Scratch, Fine-Tuning, atau Continued Pretraining
---

Salah satu tanggung jawab AI Engineer adalah menentukan konfigurasi LLM yang akan digunakan untuk membangun sistem yang ingin dibuat. AI Engineer seringkali dihadapkan dengan berbagai macam *constraint* seperti keperluan performa, latency, keperluan yang spesifik untuk jenis task atau domain tertentu, dan *constraint* lainnya yang perlu dipertimbangkan untuk membuat berbagai keputusan, salah satunya adalah memilih apakah perlu mengembangkan sendiri LLM yang akan digunakan. Seringkali dalam pengembangan sebuah proyek, constraint seperti *budget* dan waktu untuk *deliver* mengakibatkan perlunya membuat keputusan yang cepat dan tepat secara pragmatis. Ketika dihadapkan pada pilihan-pilihan ini, kapan seorang *AI Engineer* perlu membangun model sendiri dibandingkan menggunakan *model* yang sudah tersedia *off-the-shelf* melalui layanan API?

### 4.6.1 Memahami Kebutuhan

Sebelum memutuskan pendekatan pengembangan, AI Engineer perlu benar-benar memahami apa yang ingin dicapai dan apa saja batasan yang dimiliki. Apa tujuan akhir dari model atau sistem yang akan dikembangkan? Apakah pendekatan yang dipilih untuk memenuhi tujuan tersebut realistis jika dilihat dari ketersediaan resource? Pertanyaan-pertanyaan dasar ini menjadi pondasi penting agar keputusan yang diambil tidak berujung pada pemborosan waktu, tenaga, maupun biaya untuk solusi yang sebenarnya bisa diperoleh dengan cara yang lebih efisien.

Berbagai constraint pengembangan perlu dipahami sejak awal. Secara umum, beberapa aspek utama yang perlu diperjelas meliputi:

*   **Tujuan:** bisa berupa kebutuhan bisnis maupun target teknis yang ingin dipenuhi.
*   **Resource:** termasuk keterbatasan data, manpower, waktu, ketersediaan compute, dan terutama budget yang pada akhirnya akan mempengaruhi semua aspek lainnya.
*   **Task specificity:** seberapa unik atau umum tugas yang ingin diselesaikan? Apakah benar-benar membutuhkan solusi khusus, atau dapat ditangani oleh model yang sudah ada?

Salah satu output terpenting dari tahap ini adalah **memiliki gambaran yang jelas mengenai kriteria keberhasilan** baik itu berupa **metrik evaluasi** tertentu, bermacam *constraint* seperti batasan biaya dan kebutuhan latency, maupun kriteria lainnya yang berkaitan langsung dengan tercapainya tujuan sistem. Kriteria ini penting karena akan selalu digunakan dalam setiap tahap sebagai pertimbangan utama dalam mengambil keputusan teknis secara terarah. Dalam mengembangkan sebuah proyek LLM, hal ini harus menjadi salah satu langkah pertama sebelum melanjutkan tahapan lain dari pengembangan proyek yang dijalankan.

Dengan pemahaman ini, langkah selanjutnya dapat diambil secara lebih terarah. Setiap pilihan, mulai dari menggunakan model yang sudah ada, melakukan fine-tuning, hingga melatih model dari awal, dapat dipertimbangkan secara rasional berdasarkan kebutuhan nyata dan kapasitas yang dimiliki, bukan hanya menebak-nebak atau bahkan menjadikan keinginan yang tidak rasional dan realistis sebagai basis utama dari arah dan keputusan yang diambil dalam pengembangan model, yang berpotensi membuang *resource* yang berharga. Proses pengambilan keputusan akan didukung oleh kriteria yang jelas karena memberikan gambaran *tradeoff* antara *resource* yang perlu dihabiskan dengan potensial keuntungan yang bisa didapatkan dari memilih satu pendekatan dibanding yang lainnya.

---

### 4.6.2 Memilih LLM untuk Memenuhi Kebutuhan Pengembangan Aplikasi

Dalam mengembangkan aplikasi AI, pastinya sudah ada *use case* yang menjadi target pengembangan. Seperti yang sebelumnya telah dipelajari dalam modul 1, sebuah sistem aplikasi AI terdiri dari dua bagian besar, yaitu bagian non-LLM dari sistem (kode, database, dsb) dan LLM itu sendiri.

Apapun tujuannya, pastinya sistem yang dibangun perlu mampu memproses instruksi dan melakukan orkestrasi konteks untuk menghasilkan *prompt* yang tepat, serta memproses dan menggunakan output LLM untuk menghasilkan output akhir sistem.

Untuk LLM sendiri, ada beberapa pendekatan untuk menyediakan LLM ini agar bisa digunakan untuk membuat aplikasi AI yang akan dibahas pada bagian selanjutnya.

#### **Pendekatan 1: Melatih model dari nol**
Melatih model dari nol membutuhkan resource yang cukup signifikan dan umumnya tidak dilakukan untuk sebagian besar keperluan praktikal. Perlu diingat bahwa proses ini membutuhkan waktu yang lama untuk mengumpulkan dan menyiapkan data, serta kebutuhan *compute* yang tidak trivial untuk jangka waktu latih yang cukup panjang, mulai hitungan hari hingga berbulan-bulan. Umumnya tahap ini perlu dilakukan jika:
*   Anda adalah bagian dari lab / korporat yang ingin mengembangkan sebuah *foundation* model yang baru, atau
*   Anda ingin menguji sebuah arsitektur model yang baru untuk mendorong kemajuan riset, atau
*   Anda membutuhkan sebuah model dengan keperluan yang sangat spesifik yang kriterianya tidak bisa dipenuhi model manapun yang sudah ada.

Perlu diingatkan kembali bahwa semua keperluan ini juga mengasumsikan bahwa kebutuhan *resourcenya* bisa dipenuhi. Umumnya model bahasa modern sudah cukup *powerful* dan cukup mampu diadaptasikan ke domain lain, sehingga kecuali memang ada kebutuhan dan justifikasi yang sangat kuat untuk mengembangkan model dari nol karena kebutuhannya tidak bisa dipenuhi model manapun yang sudah ada, pada praktiknya sebagian besar orang tidak perlu melakukan proses ini.

#### **Pendekatan 2: Adaptasi lanjutan melalui Continued Pretraining**
*Continued pretraining* (CPT) dilakukan dengan mengambil model hasil *pretraining* sebagai *base model*, biasanya model yang belum melalui *post training*, lalu melanjutkan proses *pretraining* secara self supervised menggunakan corpus teks besar dari domain tertentu. Tujuan CPT adalah menghasilkan model domain yang memiliki pemahaman bahasa, konsep, dan struktur yang kuat pada suatu *domain* spesifik yang bisa menjadi basis yang kuat untuk melatih model spesifik *task* tertentu melalui SFT. Pada umumnya hasil CPT ini dilanjutkan lagi dengan *instruction tuning* dan *alignment*.

Sebagai contoh, ketika dibutuhkan model untuk menganalisis data legal, istilah yang digunakan dalam domain hukum sering kali memiliki makna dan konteks yang berbeda dibandingkan dengan penggunaan umumnya. Melalui CPT, model dapat mempelajari pola bahasa dan konsep yang khas dari domain hukum.

CPT memiliki resiko yang disebut *Catastrophic Forgetting*, yaitu menurunnya atau bahkan hilangnya kemampuan yang sudah dipelajari sebuah model *Deep Learning* (termasuk LLM) ketika model tersebut dilatih lagi menggunakan data yang baru. Misalkan *base model* LLM yang mau digunakan untuk CPT sudah memiliki pengetahuan dan pemahaman yang baik dalam bidang pertanian. Kemudian dilakukan CPT menggunakan data yang berisi pengetahuan pada bidang hukum. Ada resiko pengetahuannya mengenai bidang pertanian memburuk atau bahkan menjadi hilang setelah CPT. Perlu metode dan perhatian khusus dalam melakukan CPT untuk menghindari resiko ini, contohnya menggunakan *learning rate* yang jauh lebih kecil atau metode seperti *replay*.

Karena kebutuhan resource yang tinggi dan risiko kehilangan kemampuan umum, continued pretraining sebaiknya hanya dilakukan setelah dipastikan bahwa tujuan tidak dapat dicapai dengan fine tuning atau pengaturan instruksi dan konteks saja.

#### **Pendekatan 3: Mengadaptasikan model yang sudah ada melalui fine tuning**
Fine tuning cocok digunakan pada situasi di mana terdapat task, instruksi, atau konteks yang sering digunakan secara berulang, atau ketika use case sangat spesifik sehingga model umum kesulitan mencapai performa yang diinginkan. Pendekatan ini juga relevan ketika manajemen instruksi dan konteks menjadi terlalu rumit, misalnya karena terlalu banyak edge case atau terlalu banyak few shot example yang harus disediakan untuk menjaga performa model.

Daripada terus menyusun instruksi yang kompleks untuk menyesuaikan model general, kadang lebih masuk akal untuk melatih model agar memiliki perilaku tertentu secara langsung. Fine tuning pada dasarnya mengintegrasikan instruksi dan konteks spesifik ke dalam perilaku model, karena proses ini mengubah parameter model agar perilaku yang diinginkan menjadi bagian permanen dari representasi model.

Beberapa keuntungan bisa diperoleh melalui fine tuning. Instruksi dan konteks yang berulang kini sudah menjadi bagian dari model, sehingga context window dapat digunakan sepenuhnya untuk konteks yang lebih dinamis. Untuk task yang lebih spesifik, model yang lebih kecil namun telah di fine tune sering kali dapat mencapai performa yang sama atau bahkan lebih baik daripada model yang lebih besar karena kapasitasnya difokuskan pada domain yang sempit. Selain itu, dengan proses pelatihan yang baik, model menjadi lebih konsisten dalam memberikan hasil sesuai perilaku yang diharapkan.

Namun fine tuning juga membutuhkan investasi tambahan untuk mengumpulkan dan melabelkan data, serta menyediakan compute untuk melatih model. Tidak semua layanan API mendukung model custom, sehingga terkadang perlu membangun infrastruktur sendiri untuk serving model yang telah di fine tune. Risiko lain yang harus diperhatikan adalah kemungkinan penurunan performa pada task tertentu. Oleh karena itu, penting untuk memastikan bahwa peningkatan performa pada satu task tidak menyebabkan degradasi pada task lain yang masih relevan dengan sistem yang dikembangkan.

Fine tuning biasanya dilakukan sebagai langkah selanjutnya setelah terbukti kalau performa sistem yang dikembangkan menggunakan model yang tersedia tidak mampu memenuhi kebutuhan atau sudah semakin sulit ditingkatkan melalui perubahan *prompt* maupun arsitektur / pendekatan implementasi sistem yang digunakan.

#### **Pendekatan 4: Menggunakan model yang sudah ada melalui instruksi dan konteks**
Kemajuan teknologi di bidang akademik maupun industri telah memudahkan penggunaan LLM yang siap pakai. Berbagai penyedia layanan kini menawarkan akses API dengan biaya yang relatif terjangkau untuk model dengan kemampuan yang sudah sangat baik di berbagai jenis tugas.

Umumnya, AI Engineer dapat memulai dari tahap ini, dengan fokus utama pada pengujian untuk menilai sejauh mana LLM yang tersedia dapat digunakan untuk memenuhi tujuan dan kriteria yang telah ditetapkan. Melalui kombinasi evaluasi kualitatif dan kuantitatif, kekurangan sistem yang ada dapat diidentifikasi dengan jelas, sehingga membantu menentukan apakah diperlukan langkah lebih lanjut dalam memilih atau menyesuaikan LLM yang menjadi inti sistem.

Menggunakan LLM yang sudah tersedia juga merupakan cara yang efektif untuk mempercepat prototyping, menguji validitas ide dengan cepat, serta merilis aplikasi atau sistem awal dengan tujuan mengumpulkan feedback maupun data yang dapat dimanfaatkan untuk pengembangan selanjutnya. Dengan pendekatan ini, proses pengumpulan data dan pemahaman terhadap kebutuhan sistem dan keinginan *stakeholder* dapat dilakukan seiring berjalannya sistem yang dikembangkan.

Apabila hasil pengujian menunjukkan bahwa menggunakan LLM yang tersedia tidak berhasil memenuhi tujuan dan kriteria yang diinginkan, perlu dipastikan dulu apakah lebih masuk akal menghabiskan *resource* (waktu, manpower, budget) untuk mencoba mengembangkan sistem lebih lanjut dengan arsitektur dan manajemen konteks yang lebih baik maupun instruksi yang lebih tepat sebelum memutuskan masuk ke tahap selanjutnya.

Memulai dari sini umumnya adalah keputusan yang ideal untuk sebagian besar *use case*, dan kalaupun nanti diputuskan perlu pengembangan selanjutnya, AI Engineer sudah lebih mendapatkan gambaran mengenai tantangan, skenario penggunaan, maupun batasan-batasan dan kekurangan yang perlu ditangani melalui langkah-langkah selanjutnya.

---

### 4.6.3 Sebuah Framework Sederhana untuk Menentukan Pilihan Pengembangan Model

Tahap pertama dari memilih metode pengembangan model adalah dengan melihat tujuan dan kesediaan *budget*. Utamanya kita bisa membagi tujuan menjadi tiga, yaitu:
1. Tujuan akademis, mengembangkan arsitektur, metode, maupun menguji dataset baru.
2. Tujuan industri maupun akademis, ingin mengembangkan *foundation model* di suatu domain khusus yang belum ada.
3. Tujuan pragmatis, ingin mengembangkan sebuah aplikasi / sistem.

Untuk tahap ini, *flowchart* untuk memilih di antara bermacam pilihan tersebut adalah sebagai berikut:

![Flowchart 1](https://lms.sdmdigital.id/pluginfile.php/635391/mod_page/content/11/WhatsApp%20Image%202026-01-07%20at%2014.28.22.jpeg)
**Gambar 18. Flowchart awal untuk memilih pendekatan pengembangan model**

Kenapa hanya dua pilihan? Karena umumnya untuk sebagian besar use case, sulit diketahui apakah sudah pasti memerlukan *supervised fine tuning* maupun *continued pretraining*. Pendekatan yang paling pragmatis adalah dengan cara menentukan metrik-metrik yang kita perlu ukur yang sesuai dengan tujuan pengembangan, kemudian mengembangkan sebuah sistem menggunakan LLM yang sudah tersedia untuk mencoba memenuhi tujuan tersebut, lalu diukur metriknya.

Tahap ini penting karena berbagai hal:
*   Mendorong membangun dulu sistem yang matang yang bisa secepatnya di-*deploy*.
*   Memiliki sebuah sistem yang fungsional sejak awal bisa menjadi media untuk mengumpulkan data dan *feedback*.
*   Membangun sistem juga memudahkan pembuatan proses evaluasi yang baik dan benar.

Dalam memilih pendekatan menyediakan LLM yang digunakan, urutan yang direkomendasikan adalah sebagai berikut:
1. Menggunakan LLM yang tersedia
2. Melakukan *task-specific adaptation* (SFT)
3. Melakukan Domain Adaptation (CPT)
4. Melatih model yang baru dari awal

Melanjutkan pilihan sebelah kanan yang membuat dulu sistem menggunakan LLM siap pakai (lihat Gambar 18), ada kalanya kita perlu melakukan evaluasi secara berkala, lalu menentukan atau memikirkan apakah pendekatan ini perlu diubah, berdasarkan hasil evaluasi maupun *feedback* lainnya yang kita dapatkan.

![Flowchart 2](https://lms.sdmdigital.id/pluginfile.php/635391/mod_page/content/11/iamge36b.jpeg)
**Gambar 19. Flowchart menentukan langkah selanjutnya setelah evaluasi berkala**

Tahap selanjutnya dilakukan secara berkala ketika sistem dievaluasi. Setelah periode tertentu dari evaluasi sebelumnya atau dari awal pengembangan, kita mengukur beberapa metrik yang sudah ditentukan di awal, tujuannya adalah menentukan apakah kriteria performa yang diinginkan sudah tercapai oleh model / sistem. Apabila kriteria ini terpenuhi, berarti sudah cukup. Apabila tidak terpenuhi, perlu dipastikan dulu apakah secara *timeline*, *resource* atau melihat dari *iterasi* sebelumnya masih masuk akal untuk melakukan iterasi pada tahap sekarang. 

Salah satu kemungkinan yang terjadi adalah ternyata ditemukan melakukan pengembangan pada tahap selanjutnya ternyata tidak meningkatkan performa. Kalau hal ini terjadi, ada dua kemungkinan:
1.  **Masalahnya bukan pada model.** Mungkin ada kesalahan pada konteks atau instruksi yang diberikan ke model sebelumnya, mungkin ada *bug* pada sistem, atau kesalahan UI/UX. Solusinya adalah melakukan iterasi lagi terhadap sistem (*context engineering*, dsb).
2.  **Dataset yang digunakan kualitasnya kurang baik.** Perlu dilakukan pengecekan ulang kualitas data yang dikumpulkan, metode evaluasi yang digunakan, *split* data, dan faktor lainnya yang berhubungan dengan data.

---

### Mini Quiz
*(Silakan akses Mini Quiz interaktif pada platform LMS melalui tautan yang tersedia)*

---

> ### Refleksi/ Bahan Diskusi Pribadi
> *   Pikirkan sebuah use case LLM yang mungkin membutuhkan adaptasi. Misalnya anda ingin membuat LLM khusus untuk domain pertanian.
> *   Bagaimana anda akan merencanakan eksperimen dan evaluasi untuk melakukan pengembangan ini?
> *   Bagaimana anda akan mulai mencoba mengembangkan LLM ini? Pendekatan apa yang akan pertama anda lakukan?
> *   Bagaimana caranya mencari dan mengumpulkan data yang bisa menjadi referensi, konteks, benchmark, maupun data latih jika diperlukan?