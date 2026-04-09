---
slug: modul-1-1
title: modul-1-1
---
# 1.1. Pengantar GenAI dan AI Engineering: Proses end-to-end Pengembangan Aplikasi AI berbasiskan LLM

### **1.1.1 Definisi Singkat dari AI Engineering**

Apa itu *AI Engineering*? *AI Engineering* adalah proses membangun sebuah aplikasi atau sistem yang fungsi utamanya berasal dari kemampuan sebuah Foundation Model. Biasanya ketika orang menyebut *AI Engineering*, Foundation Model yang dimaksud adalah LLM atau jenis model *Generative AI* yang dilatih menggunakan berbagai macam jenis data dan kemudian dapat diadaptasi ke beberapa tugas yang berbeda. *AI Engineer* adalah istilah yang dipakai untuk menyebut individual yang melakukan *AI Engineering*. Untuk mempermudah penyebutan istilah, singkatnya akan disebut saja sistem yang dibangun oleh *AI Engineer* sebagai “Sistem AI” atau “Aplikasi AI”.

*AI Engineering* merupakan sebuah istilah baru yang muncul dari industri. Beberapa hal yang mungkin memiliki pengertian tertentu di dunia akademik memiliki arti yang berbeda di industri. Mari kita luruskan mengenai beberapa hal penting yang perlu dijelaskan terlebih dahulu.

### **1.1.2 Memahami istilah-istilah di sekitar AI Engineering**

*Artificial Intelligence* atau Kecerdasan Buatan merupakan suatu bidang dalam Ilmu Komputer yang bertujuan untuk mempelajari cara memberikan “kecerdasan” kepada sebuah mesin. Walaupun AI merupakan bidang yang luas yang terdiri dari banyak sub-bidang. Istilah *AI Engineering* di Industri banyak merujuk kepada membangun sistem menggunakan model AI yang masuk ke dalam famili *Generative AI*, utamanya *Large Language Models*.

*Generative AI* adalah salah satu bidang AI yang berfokus mengembangkan algoritma yang bisa memproduksi sendiri konten seperti gambar, teks, atau suara. *Large Language Model* merupakan salah satu jenis *Generative AI* pada bidang *Natural Language Processing*, sub-bidang AI yang bertujuan memberikan komputer kemampuan memproses teks dan percakapan.

Secara singkat, hubungan dari berbagai istilah ini adalah sebagai berikut:
* AI itu satu bidang keilmuan yang luas untuk mengembangkan kecerdasan buatan, LLM cuma bagian kecil dari AI.
* LLM adalah salah satu jenis *Generative Model* di domain *Natural Language Processing*. LLM adalah penerapan Deep Learning pada domain Natural Language Processing.
* AI terdiri dari banyak sub-bidang, termasuk di antaranya *Natural Language Processing* (memayungi LLM), Machine Learning dan Deep learning (juga memayungi LLM dan Generative Model lainnya).

Ilustrasi ini menggambarkan lebih jelas soal hubungan berbagai macam istilah di atas:

![Gambar 2](https://lms.sdmdigital.id/pluginfile.php/635349/mod_page/content/19/image6.jpg)
**Gambar 2. LLM merupakan perpotongan dari berbagai sub-bidang dalam Artificial Intelligence, dimana LLM merupakan irisan dari berbagai bidang seperti NLP, Generative AI, dan Deep Learning.**

LLM merupakan irisan dari NLP (kemampuan memproses bahasa), Generative AI (kemampuan menghasilkan konten baru), dan Deep Learning (menggunakan neural networks berlapis).

Sebelum hadirnya LLM, kita telah mengenal istilah *ML Engineering*. *ML Engineer* bertugas mengembangkan dan mempersiapkan model Machine Learning untuk proses *deployment* ke lingkungan *production*. LLM merupakan salah satu jenis model Machine Learning. Apa yang menyebabkan munculnya istilah baru untuk menyebut bidang yang membuat sistem menggunakan LLM? Apa yang berbeda di antara *AI Engineering* dan *ML Engineering*?

Seorang penulis bernama *Chip Huyen* melakukan survey kepada pelaku industri untuk menjawab pertanyaan ini. Dalam bukunya yang berjudul “*AI Engineering*”, dia menyatakan kalau faktor utamanya adalah sifat dari LLM itu sendiri. Cara mengadaptasikan, cara evaluasi, hingga infrastruktur yang diperlukan untuk membuat sistem berbasis LLM punya perbedaan yang signifikan dengan model ML lainnya, memerlukan disiplin ilmu dan kemampuan yang berbeda sehingga untuk mempermudah pelaku industri, maka muncullah istilah *AI Engineering*.

Sebagai analogi, seorang Web Developer maupun Android Developer pada dasarnya termasuk dalam ranah **Software Engineering**, karena keduanya berfokus pada pengembangan aplikasi. Namun, pengembangan aplikasi berbasis web dan berbasis Android memerlukan keterampilan yang berbeda, sehingga kedua istilah tersebut dibedakan dalam industri. Hal yang serupa juga terjadi antara **AI Engineering** dan **ML Engineering**. Meskipun keduanya memiliki banyak irisan, perbedaannya tetap cukup signifikan, baik dari segi keterampilan maupun pola pikir yang dibutuhkan. Oleh karena itu, muncul istilah baru, yaitu **AI Engineering**.

### **1.1.3 Perbedaan ML Engineering dengan AI Engineering**

**Tabel 1. Perbandingan antara ML Engineering dengan AI Engineering**

| Aspek | ML Engineering | AI Engineering |
| :--- | :--- | :--- |
| **Fokus utama** | Membangun, melatih, dan mengoptimalkan model berdasarkan data untuk mencapai metrik akurasi tertentu. | Mendesain sistem berbasis foundation models (misalnya LLM) dengan memanfaatkan instruksi, konteks, dan prompt, sering kali tanpa perlu melatih model dari awal. |
| **Adaptasi ke use case baru** | Membutuhkan proses training tambahan atau fine-tuning untuk adaptasi tugas baru. | Mengadaptasi via instruksi, konteks, contoh few-shot, atau fine-tuning ringan. Pada skenario tertentu masih dapat dilakukan continued pretraining. |
| **Jenis input** | Mayoritas data terstruktur atau semi-terstruktur (angka, teks, gambar). | Apa pun yang dapat direpresentasikan sebagai teks (tak terstruktur) maupun terstruktur. |
| **Sifat model** | Deterministik, yakni input sama menghasilkan hasil yang sama. | Stokastik, yakni input sama dapat memberikan hasil berbeda. |
| **Peran engineer** | Fokus pada pipeline data, arsitektur model, training, dan optimasi performa. | Fokus pada rancangan prompt, orkestra konteks, pengendalian perilaku model, evaluasi, serta integrasi model dalam sistem end-to-end. |
| **Tantangan utama** | Feature engineering, masalah training, dan optimasi performa. | Halusinasi, inkonsistensi output, serta kontrol perilaku dan kualitas respons. |
| **Evaluasi hasil** | Mengevaluasi menggunakan metrik objektif (akurasi, F1, RMSE, dsb.). | Bergantung konteks penggunaan: mulai dari metrik objektif (Perplexity, ROUGE) hingga metrik evaluasi kualitatif yang kadang memerlukan LLM lain (atau human judge) untuk mengevaluasi (LLM-as-a-judge). |

Sebelum mempelajari lebih lanjut mengenai tanggung jawab seorang AI Engineer, mari kita terlebih dahulu memahami apa yang dimaksud dengan LLM serta kemampuan apa saja yang dimilikinya sehingga dapat menjadi unsur penting dalam pengembangan sistem yang kompleks.

### **1.1.4 Mengenal LLM**

*Large Language Models* atau LLM adalah model *Machine Learning* yang dilatih dengan data berskala besar untuk melakukan *Language Modelling* dengan tujuan meniru kemampuan manusia berbahasa. *Language modeling* adalah teknik dalam NLP yang mempelajari pola bagaimana kata-kata atau rangkaian kata muncul bersama dalam sebuah bahasa. Keunikan LLM adalah selain memiliki pemodelan bahasa yang baik yang berasal dari skala data yang digunakan untuk melatihnya, LLM juga bisa dilatih untuk mengikuti instruksi. Selain itu, LLM juga bisa memahami informasi yang diberikan dalam bentuk abstrak (teks bebas, post media sosial, dsb) maupun informasi terstruktur. Pengguna LLM cukup memberikan instruksi dan informasi yang tepat untuk mendapatkan output yang diinginkan.

Karena keunikan ini, LLM menjadi sangat serbaguna karena bisa memproses informasi abstrak hanya dengan instruksi dalam bahasa manusia. Hal ini berbeda dengan pemrograman tradisional, yang memerlukan terlebih dahulu penentuan struktur informasi dan instruksi yang harus mengikuti sintaks tertentu. LLM bisa dipandang sebagai sebuah fungsi atau program yang bisa didefinisikan hanya dengan memberikan instruksi bagaimana program tersebut seharusnya bekerja disertai input yang dibutuhkan ke dalam program tersebut.

### **1.1.5 Memandang LLM secara black box**

Agar mudah dijelaskan, kita lihat terlebih dahulu dari sisi LLM yang input dan outputnya hanya berupa teks. Secara garis besar, input dan output dari sebuah LLM berbentuk teks. Input yang diberikan ke LLM akan mempengaruhi output yang dihasilkan, dan input tersebut dapat diatur atau dirancang sedemikian rupa untuk mencapai perilaku atau keluaran yang kita inginkan.

![Gambar 3](https://lms.sdmdigital.id/pluginfile.php/635349/mod_page/content/19/high%20level%20black%20box.jpg)
**Gambar 3. Gambaran high-level black box dari sebuah LLM.**

Sebuah teks bisa merepresentasikan berbagai macam informasi, mulai dari bahasa manusia hingga format informasi terstruktur seperti XML, YAML, JSON, juga syntax dari suatu bahasa pemrograman. Teks juga bisa berisi instruksi maupun informasi yang merepresentasikan konteks. Kalau input dan output dari LLM adalah teks, berarti kita bisa membuat sebuah fungsi yang input dan outputnya apapun selama input dan output ini bisa kita ubah menjadi teks. Fungsi ini juga bisa memiliki perilaku apapun, tergantung instruksi yang diberikan. Contoh: misalnya kita ingin meminta LLM membuat ringkasan dari sebuah teks berita. Maka bentuk input dan output kita akan terlihat seperti berikut:

![Gambar 4](https://lms.sdmdigital.id/pluginfile.php/635349/mod_page/content/19/ilustrasi%20penerapan%20llm.jpg)
**Gambar 4. Ilustrasi penerapan LLM untuk meringkas sebuah teks berita.**

Semua teks input perlu digabung dulu menjadi satu teks sebelum masuk LLM. Input ini yang berisi informasi dan instruksi biasanya disebut *prompt*. Selain output teks bebas, LLM juga bisa mengeluarkan output dengan format terstruktur, karena setiap format teks sebenarnya hanya teks dengan syntax tertentu. Contoh: misalnya kita ingin melakukan klasifikasi sentimen dari sebuah post media sosial, dan kita ingin agar outputnya mengikuti format tertentu, maka bentuk interaksinya dapat ditulis seperti berikut:

```json
{
  "sentimen": "positif",
  "penjelasan": "......."
}
```

Maka input dan output kita akan terlihat seperti gambar di bawah.

![Gambar 5](https://lms.sdmdigital.id/pluginfile.php/635349/mod_page/content/19/ilustrasi%20penggunaan%20llm.jpg)
**Gambar 5. Ilustrasi penggunaan LLM untuk melakukan sentiment classification dengan format output JSON.**

*(Peringatan: LLM tidak selalu menghasilkan JSON yang valid. Sistem AI Engineer harus memiliki error handling untuk kasus-kasus seperti: - Missing closing - Typo pada key name - Extra text before/after JSON)*

Seperti yang dapat dilihat, instruksi tidak hanya dapat mendeskripsikan apa fungsi yang diinginkan, tetapi juga dapat mendeskripsikan format output. Karena output ini masih berupa teks, AI Engineer biasanya perlu memproses kembali output LLM tersebut agar dapat digunakan oleh aplikasinya. Tergantung pada model yang digunakan, umumnya LLM sudah dilatih menggunakan data yang memberikan kemampuan untuk memahami dan menghasilkan output dengan format tertentu.

Walaupun memiliki kelebihan yang unik, LLM tidak selalu memberikan output yang benar. Bisa jadi format yang dihasilkan memiliki sintaks yang keliru, LLM salah menafsirkan instruksi, atau bahkan secara ekstrem LLM dapat mengarang informasi baru. Oleh karena itu, selain mengatur aspek fungsional LLM melalui instruksi dan informasi, AI Engineer juga perlu menangani kondisi-kondisi tersebut agar aplikasi dapat berjalan dengan baik.

### **1.1.6 Apa saja pekerjaan seorang AI Engineer?**

LLM terkesan serbaguna, tapi untuk menggunakannya dengan efektif, kita perlu memikirkan input teks yang tepat dan cara yang benar untuk memproses outputnya. Teks seperti apa yang perlu dimasukkan ke dalam LLM? Output seperti apa yang diinginkan dari LLM, dan nantinya digunakan seperti apa? Bagaimana cara seorang *AI Engineer* memanfaatkan model ajaib ini? Secara garis besar, seorang *AI Engineer* punya empat tanggung jawab utama sebagai berikut:

1.  **Menyiapkan input yang tepat untuk LLM**. AI Engineer memastikan LLM selalu mendapat instruksi dan konteks yang dibutuhkan agar bisa menjalankan tugasnya dengan benar. Ini termasuk membangun sistem atau infrastruktur yang bisa menyediakan input tersebut secara konsisten.
2.  **Menentukan dan mengontrol output LLM**. Tugasnya bukan cuma menentukan bentuk hasil yang diinginkan, tapi juga memastikan hasil dari LLM sesuai ekspektasi dan memenuhi standar tertentu. Termasuk di dalamnya membangun sistem evaluasi baik dari sisi performa (angka/metrik) maupun kualitas (misalnya gaya bahasa, aspek keamanan, atau memastikan output tidak mengandung konten negatif).
3.  **Mengatur konfigurasi dan pemilihan model**. AI Engineer menentukan model LLM mana yang dipakai, mengatur parameter-parameter penting, dan menilai apakah perlu dilakukan fine-tuning atau tidak agar sistem bekerja optimal.
4.  **Menjamin sistem siap dan stabil di production**. Setelah sistem dikembangkan, AI Engineer memastikan semuanya bisa dijalankan di lingkungan produksi dengan lancar dan terpantau dengan baik, termasuk infrastruktur yang mendukung pengembangan lebih lanjut sistemnya.

Bagian selanjutnya akan membahas masing-masing poin di atas secara lebih detail sebelum masuk ke pembahasan teknis di modul berikutnya.

### **1.1.7 Menyiapkan input yang tepat untuk LLM**

Input ke LLM biasanya terdiri dari instruksi disertai dengan Informasi yang relevan, biasanya disebut sebagai konteks. Setelah menentukan instruksi dan mendapatkan informasi yang dibutuhkan, semuanya akan digabung menjadi satu teks yang disebut *prompt*. Bagaimana cara *AI Engineer* menyiapkan dua hal tersebut?

#### **Instruksi**
Sebelum menentukan instruksi, seorang AI Engineer perlu memahami terlebih dahulu apa kebutuhan bisnisnya. Dari kebutuhan bisnis tersebut kemudian diturunkan menjadi masalah-masalah yang lebih kecil dan lebih terdefinisi. Setiap masalah ini kemudian dipetakan menjadi fungsi yang perlu dibuat, dan dari sana dapat diturunkan kebutuhan instruksi serta konteksnya. Proses ini sangat mirip dengan perencanaan fitur pada perangkat lunak. Perbedaannya adalah, pada kasus ini implementasi utamanya diwujudkan melalui instruksi dan konteks (berupa *prompt*), bukan melalui kode program konvensional.

Ada banyak cara untuk merancang sebuah instruksi. Bisa sesederhana memberi perintah langsung seperti “tolong terjemahkan ke Bahasa Indonesia”, atau bisa juga membutuhkan detail yang lebih lengkap: mengatur gaya bahasa, mendefinisikan peran model, menentukan format output, hingga hal yang lebih kompleks seperti mengatur cara model menyelesaikan masalah (*problem breakdown*) untuk keperluan *reasoning* atau melakukan *planning*.

Tentu instruksi saja tidak memadai. LLM juga perlu diberikan konteks, karena setiap kasus memiliki latar dan detail spesifik yang harus diketahui agar output yang dihasilkan dapat lebih akurat, relevan, dan sesuai dengan instruksi maupun ekspektasi. Dengan demikian, merancang dan menyiapkan konteks yang tepat merupakan salah satu keterampilan inti dalam praktik AI Engineering.

#### **Konteks**
Tanpa konteks, LLM tidak bisa melakukan instruksinya dengan baik atau malah mengarang asumsi lalu digunakan untuk menjalankan instruksinya. LLM punya “pengetahuan” internal sendiri yang berasal dari data latihnya, tapi setiap penggunaan pasti punya konteks sendiri yang dibutuhkan. misalnya kita ingin LLM mampu mengingat percakapan. Tanpa diberikan *chat* sebelumnya, LLM tidak akan mengerti kalau ditanya soal sesuatu dari percakapan sebelumnya.

Di sisi lain, karena konteks dapat bersumber dari mana saja, konteks tersebut perlu diformat terlebih dahulu agar sesuai dengan bentuk input LLM, yaitu teks. Misalnya ketika kita diminta membuat chatbot yang diberikan konteks berupa dokumen peraturan perusahaan. Dokumen ini dapat berbentuk PDF atau DOCX, dan perlu di-parse terlebih dahulu untuk menjadi teks sebelum dapat dimasukkan ke LLM. Oleh karena itu, tahap konversi dan normalisasi konteks juga merupakan bagian penting dalam alur kerja *AI Engineering*.

Sumber konteks yang berbeda juga memerlukan cara tersendiri untuk mencari dan memprosesnya. Konteks dalam basis data harus di-query terlebih dahulu. Konteks yang tersedia dari layanan internet harus di-request terlebih dahulu. Konteks yang disimpan sebagai berkas harus dibaca terlebih dahulu.

Terakhir, konteks yang diperlukan dapat berbeda pada setiap pemanggilan, sehingga konteks ini harus dimasukkan secara dinamis ke dalam prompt. Sistem yang dibangun *AI Engineer* tidak hanya harus mampu membaca dan memproses konteks, tetapi juga menentukan kapan suatu konteks diperlukan.

#### **Di Luar Prompt**
Selain memastikan instruksi dan konteks yang tepat diberikan lewat *prompt*, masih ada beberapa hal yang penting untuk dipastikan oleh *AI Engineer*. Prompt yang dikembangkan akan terus menerus diubah seiring dengan pengembangan aplikasi. Karena *prompt* ini berhubungan langsung dengan perilaku model, maka perlu dilakukan ***versioning*** agar setiap versi *prompt* bisa diuji secara terpisah dan dibandingkan satu sama lain. *Versioning* juga memudahkan *rollout* dan *rollback* *prompt* ketika ada versi baru yang mau dirilis atau perlu kembali ke versi sebelumnya karena yang baru ternyata bermasalah.

Belum lagi isu keamanan. LLM tidak membedakan instruksi dari siapapun; selama instruksi dan konteksnya jelas, maka instruksi tersebut akan dijalankan, meskipun instruksi itu datang dari input pengguna. Misalnya pada kasus aplikasi chatbot AI yang menyimpan data pribadi pengguna. Bayangkan ada oknum yang memahami cara kerja sistem tersebut, dan ia mengetahui bahwa aplikasi menyimpan data pribadi yang dapat diakses oleh AI. Oknum tersebut dapat memberikan instruksi melalui chat untuk mengambil data pribadi pengguna lain, dan tanpa mitigasi, LLM berpotensi mengikuti instruksi tersebut. *AI Engineer* juga harus mengantisipasi hal semacam ini dengan mengatur *policy* / *guardrails* yang akan mengawasi input untuk memastikan bahwa inputnya “aman”.

### **1.1.8 Memastikan output yang tepat keluar dari LLM**

Output dari model LLM adalah sebuah teks. Jika teks tersebut sudah memenuhi kebutuhan aplikasi, maka tidak diperlukan pemrosesan lebih lanjut. Namun dalam praktiknya, banyak aplikasi yang membutuhkan keluaran dengan struktur tertentu, misalnya format JSON agar dapat langsung dipetakan menjadi dictionary Python, atau format tabel agar dapat langsung dimasukkan ke database atau diproses modul lain. Untuk kasus seperti ini, AI Engineer perlu menambahkan lapisan pasca-pemrosesan (*post-processing*) yang bertugas memvalidasi format, mem-parse output, dan mengekstrak informasi agar hasil dari LLM dapat digunakan secara konsisten dalam sistem.

Selain itu, LLM dapat juga memberikan output yang salah, mengarang informasi, salah mengerti instruksi, memberikan output di luar yang diminta, dan bisa jadi outputnya berpotensi merugikan. AI Engineer harus memastikan output LLM valid untuk digunakan lebih lanjut. Bayangkan kembali use case sentiment analysis pada contoh sebelumnya yang ditunjukkan pada Gambar 5. Skenario kegagalan apa saja yang mungkin terjadi?

*   Output bukan dalam format JSON (format berbeda dari yang diminta).
*   JSON tidak valid (gagal di-parse sehingga tidak bisa diproses lebih lanjut).
*   Output mengandung JSON sekaligus teks lain (memerlukan pemrosesan tambahan untuk mengekstrak JSON).
*   LLM menambahkan kategori sentimen yang tidak diminta (misalnya diminta hanya positif/negatif, tetapi model menambahkan *netral*/*unknown*).
*   Struktur JSON tidak sesuai spesifikasi (misalnya nama key berbeda atau urutan atau tipe nilainya tidak konsisten).

Apa yang bisa dilakukan AI Engineer untuk mengatasi atau mengantisipasi masalah tersebut?
*   Mengubah instruksi agar lebih spesifik, misalnya secara eksplisit menentukan daftar kategori sentimen yang valid.
*   Menambahkan contoh output pada instruksi, termasuk format JSON dan kategori sentimen yang diperbolehkan.
*   Mengimplementasikan parser yang mampu mencoba mem-*parse* JSON yang tidak ideal atau sebagian benar.
*   Mengimplementasikan mekanisme *retry* dengan *feedback*, misalnya menggunakan pesan galat dari proses parsing JSON sebagai konteks tambahan untuk menghasilkan ulang (*regenerate*) output yang lebih benar.

Selain itu, sama seperti pada sisi input, output yang dihasilkan juga dapat bersifat berbahaya. Kembali ke contoh sebelumnya: misalkan meskipun sudah diberi pengamanan di sisi input, LLM masih tetap memberikan informasi pribadi pengguna lain ketika diminta. Pada kasus lain, LLM dapat mengeluarkan respons yang tidak relevan dengan topik, menggunakan bahasa yang tidak pantas, atau menyebutkan hal yang seharusnya tidak diungkapkan.

*Policy* atau *guardrails* (filter pengaman) juga perlu diterapkan pada sisi output. Berdasarkan kriteria tertentu, output yang tidak memenuhi aturan dapat difilter, kemudian sistem dapat memilih untuk meminta model melakukan *regeneration* atau mengembalikan *respons template*, tergantung pilihan desain yang paling sesuai dengan *use case*. Contohnya: ketika LLM mengembalikan kategori sentimen yang tidak valid, respons tersebut dapat langsung ditolak dan sistem meminta model menghasilkan ulang. Atau pada kasus keamanan, jika output terdeteksi berisi informasi yang tidak pantas, sistem dapat mengembalikan pesan standar seperti “permintaan ini tidak dapat dilayani”.

Dengan demikian, selain memastikan input sudah benar, *AI Engineer* juga harus memastikan output LLM melewati proses quality control sebelum dikonsumsi oleh aplikasi. Di level implementasi, ini berarti *AI Engineer* bertanggung jawab merancang filter, validator, *schema checking*, dan logika penanganan kesalahan yang memadai sehingga model tidak hanya berfungsi, tetapi juga aman, konsisten, dan dapat dipakai secara reliabel dalam sistem produksi.

![Gambar 6](https://lms.sdmdigital.id/pluginfile.php/635349/mod_page/content/19/peran%20AI%20engineer.jpg)
**Gambar 6. Ilustrasi peran AI Engineer dalam “menjaga” output dari LLM**

### **1.1.9 Menentukan konfigurasi LLM yang tepat**

Seorang AI Engineer juga memiliki tanggung jawab untuk melakukan konfigurasi LLM yang dipakai. Konfigurasi ini melibatkan dua bagian, yaitu:
*   Menentukan LLM yang dipakai, dan
*   Memilih konfigurasi yang tepat untuk LLM yang digunakan.

#### **Menentukan model LLM yang dipakai serta cara mengaksesnya**
*AI Engineer* perlu memilih antara menggunakan LLM yang disediakan melalui API oleh berbagai penyedia layanan dan melakukan *self hosting* atau mengatur sendiri infrastruktur untuk menjalankan dan menggunakan LLM, termasuk memutuskan apakah bisa menggunakan model yang sudah dilatih, atau perlu mengembangkan dulu model yang dibutuhkan melalui berbagai macam cara, seperti melatih model dari nol atau mengadaptasikan model yang sudah ada melalui fine tuning untuk sebuah task spesifik atau *continued pretraining*, yaitu melatih ulang sebuah model untuk memberikan pengetahuan lebih spesifik di suatu domain.

#### **Menentukan konfigurasi yang tepat untuk LLM yang digunakan**
LLM memiliki berbagai macam parameter untuk proses *inference*, seperti parameter untuk *sampling* (bagian dari cara LLM menghasilkan output) atau parameter lainnya seperti YaRN scaling untuk meningkatkan *context window* (jumlah total input dan output yang bisa diproses LLM). Seorang *AI Engineer* harus memahami apa saja parameter ini, efeknya, serta mampu melakukan iterasi untuk memilih nilai yang tepat untuk penggunaan yang diinginkan.

![Gambar 7](https://lms.sdmdigital.id/pluginfile.php/635349/mod_page/content/19/peran%20AI%20engineer2.jpg)
**Gambar 7. Peran AI Engineer: AI Engineer memiliki peran selayaknya seorang mekanik yang membangun sebuah mobil. Dia memiliki mesin yang tepat (model LLM), melakukan tuning dan merangkai sistemnya, lalu memonitor dan mengevaluasi kinerja dari keseluruhan sistem.**

### **1.1.10 Membangun sistem berbasis LLM**

Tanggung jawab untuk mengelola input dan output LLM diwujudkan dalam bentuk sebuah sistem yang mengelilingi LLM. Sistem ini dapat memiliki banyak komponen sesuai kebutuhannya, tetapi secara konseptual dapat dipisahkan menjadi dua bagian: komponen LLM dan komponen non-LLM. Bagian non-LLM inilah yang bertugas mengimplementasikan seluruh kebutuhan untuk mengatur input dan output LLM, termasuk melakukan validasi, filtering, transformasi, serta kontrol atas kualitas keluaran. Selain itu, bagian non-LLM juga dapat berisi *business logic* yang dibutuhkan untuk benar-benar menyelesaikan masalah bisnis yang menjadi tujuan aplikasi.

![Gambar 8](https://lms.sdmdigital.id/pluginfile.php/635349/mod_page/content/19/diagram%20high%20level.jpg)
**Gambar 8. Diagram high level sebuah sistem berbasis LLM yang terdiri dari LLM dan bagian non-LLM dari sebuah sistem.**

Bagian non-LLM dari sistem seringkali justru lebih besar daripada bagian yang secara langsung berinteraksi dengan pemanggilan LLM, dan bisa meliput bermacam hal, seperti *vector database* dan *pipeline ingestion* dokumen, grafik yang merepresentasikan sebuah *workflow*, bermacam implementasi tool yang bisa dipanggil LLM, *guardrails*, dan lain sebagainya. Bagian bagian individual ini akan dibahas pada modul selanjutnya.

### **1.1.11 Multi-modal LLM**

Sebagian LLM bisa memproses input selain teks, seperti gambar, audio, video, dan lainnya. Model ini disebut multi-modal LLM.

![Gambar 9](https://lms.sdmdigital.id/pluginfile.php/635349/mod_page/content/19/ilustrasi%20modal%20llm.jpg)
**Gambar 9. Ilustrasi multi-modal LLM.**

Apa pun jenis inputnya, tanggung jawabnya tetap sama. *AI Engineer* perlu membangun infrastruktur dan sistem agar LLM dapat menerima input, melakukan filtering, mengorkestrasi berbagai sumber data agar dapat diakses melalui sistem, serta menangani tanggung jawab operasional lainnya.

### **1.1.12 Membawa sistem ke production dan memastikan sistem berjalan dengan baik di production**

Sistem LLM pada akhirnya adalah sebuah sistem *software*. Dan sebagaimana sistem *software* lainnya, kualitasnya tidak ada artinya apabila tidak dibawa ke *production* untuk benar-benar digunakan oleh pengguna atau stakeholder. Karena itu, *AI Engineer* juga bertanggung jawab menyiapkan sistemnya agar siap di-deploy ke lingkungan *production*. Lingkungan production memiliki kebutuhan operasional yang berbeda dari sekadar prototype atau demo, misalnya: memonitor performa sistem, melakukan pembaruan sistem, memasang logging dan alert ketika terjadi kesalahan atau *crash*, mengumpulkan data kesalahan untuk dianalisis, dan lain sejenisnya.

Selain itu, *AI Engineer* juga perlu menyiapkan infrastruktur untuk menjalankan sistem di *production* (misalnya server, kontainerisasi, orkestrasi, CI/CD), dan melakukan proses *deployment* itu sendiri. Pada praktiknya, tidak semua tanggung jawab ini selalu dilakukan oleh satu orang (bisa dibantu DevOps / platform engineer), tetapi seorang *AI Engineer* perlu memahami bahwa sistem LLM yang ia bangun baru dianggap “selesai” apabila mampu beroperasi dengan stabil, aman, dan terukur di lingkungan *production*.

Setelah deployment, tanggung jawab *AI Engineer* masih belum selesai, karena sistem yang berjalan di *production* tidak selalu berjalan mulus. Setiap kali ada insiden, error, atau masalah apapun di *production*, *AI Engineer* harus melakukan diagnosa untuk memahami apa yang terjadi, membuat rencana pengembangan selanjutnya, lalu mengeksekusi rencana tersebut dan melakukan *deployment* lagi untuk versi sistem yang lebih baru.

Banyak konsep *ML Engineering* yang masih berlaku untuk *AI Engineering*, seperti mengidentifikasi *drift*, melakukan error analysis ketika terjadi error, baik itu dari segi output sistem maupun error yang terjadi saat sistem sedang berjalan, setup *feedback loop* untuk mengumpulkan data dan *insight* mengenai perilaku sistem di *production* yang bisa digunakan untuk retraining (atau dalam kasus sistem LLM, mengatur ulang *prompt*, menambahkan example, atau menambahkan rules), dan melakukan evaluasi di *production* untuk mengukur metrik-metrik performa dari sistem secara keseluruhan.

Seorang *AI Engineer* tidak hanya bertugas memastikan input yang tepat, output yang sesuai harapan, serta konfigurasi LLM yang pas, tetapi juga harus memperhatikan aspek production dalam membangun sebuah sistem AI. Dengan kata lain, tanggung jawab *AI Engineer* itu bukan hanya soal “membuat LLM bekerja”, tetapi memastikan keseluruhan sistemnya dapat dioperasikan dengan aman, stabil, dan berkelanjutan ketika digunakan secara nyata oleh pengguna. Inilah cara pandang yang komprehensif untuk memahami lingkup peran *AI Engineer* modern.

![Gambar 10](https://lms.sdmdigital.id/pluginfile.php/635349/mod_page/content/19/ilustrasi%20tanggung%20jawab%20AI%20eng.jpg)
**Gambar 10. Ilustrasi keseluruhan tanggung jawab AI Engineer**

---

### **Mini Quiz**
*(Interaktif - Tidak tersedia dalam format teks)*

---

> ***Refleksi Singkat***
>
> Pada tahun 2015, muncul paper berjudul “Hidden Technical Debt in Machine Learning Systems” yang ditulis oleh tim Machine Learning Google. Salah satu bahasan paper ini adalah bermacam-macam masalah yang unik dihadapi oleh sistem ML di *production* dan cara-cara untuk menghadapinya. Ada diagram yang cukup populer di paper tersebut yang menunjukkan bahwa pada sebuah sistem ML, meskipun fungsionalitas utamanya berasal dari sebuah algoritma ML, kode yang melibatkan ML hanya merupakan bagian kecil dari sistem secara keseluruhan.
>
> ![Gambar 11](https://lms.sdmdigital.id/pluginfile.php/635349/mod_page/content/19/ilustrasi%20kenyataan%20sistem%20ML.jpg)
> **Gambar 11. Ilustrasi kenyataan dari sebuah sistem ML**
> *Sumber Gambar: diambil dari paper “Hidden Technical Debt in Machine Learning Systems”.*
>
> Apakah menurutmu temuan di *paper* tersebut masih berlaku untuk *AI Engineering*? Atau malah justru makin berlaku? Kira kira apa perbedaan antara infrastruktur sistem ML dan sistem berbasis LLM? Coba tanyakan pada teman atau kolega yang berprofesi sebagai *ML Engineer* atau *AI Engineer*. Meskipun sudah 10 tahun sejak paper tersebut dirilis, pengetahuan mengenai MLOps masih sangat penting karena banyak konsep dasar yang masih relevan untuk membawa sistem berbasis LLM ke *production*.