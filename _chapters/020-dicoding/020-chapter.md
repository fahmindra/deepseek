---
slug: ch-06-02
title: "Large Language Model (LLM)"
layout: chapter
---
# Large Language Model (LLM)
## Pengantar Large Language Model (LLM)

Halo *Coders* dan para calon *Generative AI* Engineer idaman!

Sebelum mulai “penyambutannya” ada sebuah ramalan yang akan diberikan untuk Anda. Ramalannya adalah: “Pasti ini bukan pertama kalinya Anda mendengar kata *Generative AI* ataupun LLM”. Apakah ramalannya tepat?

*Yup*, Anda tentu sudah sangat familier dengan beberapa nama seperti ChatGPT dari OpenAI, Gemini dari Google, atau Claude dari Anthropic. Kita bahkan tidak akan melewatkan satu hari tanpa menggunakan *tools* AI yang dikembangkan oleh beberapa perusahaan berikut.

![Gambar 1](https://assets.cdn.dicoding.com/original/academy/dos-651710dfba83959b9b22aa82d9bbfb0920260304120403.jpeg) 

Seperti yang kita ketahui bersama, dunia sedang mengalami pergeseran (shifting) ke arah baru. Pergeseran ini mengharuskan kita untuk hidup berdampingan dengan teknologi AI, sebagaimana kita telah menormalkan penggunaan internet sejak tahun 1989. Jika dahulu dunia berlomba untuk menciptakan bom dan ledakan terkuat, sekarang dunia bahkan beberapa perusahaan multinasional sedang berlomba untuk membuat inovasi di bidang AI. Coba Anda simak potongan berita berikut ini.

![Gambar 2](https://assets.cdn.dicoding.com/original/academy/dos-69685b3e2644f420d64c6c252ec51e6b20260304120412.jpeg)

Pada masa ini, ilmuwan AI sudah layaknya para *superstar* di dunia sepak bola yang diperebutkan di “bursa transfer” oleh perusahaan-perusahaan besar. Mereka menjadi aset berharga yang sangat dicari karena keahlian di bidang ini dianggap sebagai salah satu sumber daya paling strategis.

Namun, apa yang dapat kita simpulkan dari fenomena di atas?

Fenomena ini menunjukkan bahwa pengetahuan tentang *Generative AI*, terutama *Large Language Model* (LLM), kini telah menjadi semacam “harta karun” bernilai luar biasa. *Nah*, pada kelas ini dan terutama di modul kali ini, kita akan menjelajahi tidak hanya di permukaan, tetapi hingga membongkar habis model-model *generative AI* yang menjadi otak dari *chatbot* populer seperti ChatGPT dan Gemini yang kita bicarakan sebelumnya.

Namun, apa saja yang akan kita bahas pada modul pertama ini? Mari kita lihat peta perjalanan berikut ini.

![Gambar 3](https://assets.cdn.dicoding.com/original/academy/dos-813ce7350393c69a2d5ad58a4195bcba20260304120424.jpeg) 

Modul ini memiliki delapan poin pembahasan atau submodul yang akan membawa kita selangkah demi selangkah semakin akrab dengan konsep Large Language Model.

Jika Anda sudah mulai bertanya apa yang dimaksud dengan Large Language Model dan bagaimana kaitannya dengan *Generative AI*, Anda berada di tempat yang tepat. Oleh karena itu, mari kita masuk ke materi berikutnya untuk mengenal LLM lebih dekat..

---

## Selangkah Lebih Dekat dengan LLM

Sebagai NLP *warrior*, ketika menghadapi sebuah permasalahan pemrosesan bahasa alami, model-model seperti LSTM, GRU, atau bahkan BERT mungkin menjadi pilihan pertama yang terlintas di pikiran. Pilihan tersebut tentu bukanlah sebuah kekeliruan, model-model itu telah menjadi tonggak penting dan masih digunakan secara luas hingga kini.

Namun, bahasa manusia itu sangatlah kompleks. Ia penuh dengan konteks, variasi makna, ambiguitas, bahkan budaya yang terus berkembang. Pemahaman tentang beberapa komponen kompleks tersebut sembari berharap model konvensional dapat menghasilkan output baru yang dapat menjawab dengan kualitasnya sama seperti manusia rasa-rasanya hanyalah mimpi di tengah hari bolong.

*Nah*, di sinilah waktu yang tepat bagi Anda untuk berkenalan dengan Large Language Model (LLM) sebagai terobosan yang mengubah peta permainan. Namun, sebelum melangkah lebih jauh, ada pertanyaan pembuka untuk Anda: apa yang terlintas di pikiran ketika mendengar istilah Large Language Model?

Mungkin sebagian akan menjawab, “model bahasa dengan ukuran besar.” Jawaban itu tidak sepenuhnya salah, tapi pertanyaan berikutnya adalah: ukuran apa yang sebenarnya besar?

Mari kita cari bersama-sama jawabannya.

### Definisi LLM

*By definition*, **Large Language Model** atau **LLM** adalah jenis model artificial intelligence yang dilatih menggunakan data teks dalam jumlah yang luar biasa besar. Melalui proses pelatihan ini, model belajar mengenali pola, struktur, dan makna dalam bahasa hingga mampu memahami dan menghasilkan teks yang menyerupai manusia.

![Gambar 4](https://assets.cdn.dicoding.com/original/academy/dos-d32f90b0bc4a8b16227aa73b9b70bdfe20260429135044.png)

Menariknya, selain dapat menjawab pertanyaan, LLM juga mampu untuk melakukan berbagai tugas bahasa yang bervariasi, seperti translation, membuat esai, dan bahkan tugas sederhana seperti analisis sentimen tanpa harus dilatih secara khusus untuk setiap tugas tersebut.

*Eits*, kita belum menjawab apa yang sebenarnya dikatakan “Besar” dalam Large Language Model. Tentu saja bukan ukuran otaknya seperti pada ilustrasi sebelumnya. Baiklah mari kita temukan jawabannya pada bagian selanjutnya.

#### Tiga pilar LLM

Hal yang “Besar” dalam LLM itu terbagi menjadi tiga yang akan kita sebut sebagai tiga pilar LLM. Dengan memahami ketiga pilar tersebut kita dapat mengungkap resep yang membuat LLM begitu kuat.

*   **Data: Triliunan Token Pengetahuan**
    Setiap LLM dibangun di atas lautan data teks yang mencakup buku, artikel, situs web, kode, dan berbagai bentuk tulisan lain. Volume data tersebut dapat mencapai triliunan token, memungkinkan model belajar tentang berbagai topik, gaya bahasa, dan konteks penggunaan kata. Semakin luas dan beragam datanya, semakin kaya pula pemahaman model terhadap bahasa dan dunia di sekitarnya.
*   **Model: Miliaran Parameter yang Belajar Sendiri**
    Pilar kedua adalah arsitektur model itu sendiri. LLM modern memiliki miliaran hingga triliunan parameter, yang berfungsi seperti “neuron” buatan. Setiap parameter menyimpan sebagian kecil dari pola dan pengetahuan yang diperoleh selama proses pelatihan. Jumlah parameter yang masif inilah yang memberi kemampuan luar biasa pada LLM untuk mengenali konteks, menyusun logika, hingga meniru gaya berbahasa manusia.
*   **Compute Power: Tenaga di Balik Otak Digital**
    Pilar ketiga adalah daya komputasi ekstrem. Melatih model sebesar ini bukan perkara sederhana sebab diperlukan ribuan GPU atau TPU yang bekerja berbulan-bulan untuk memproses data dalam skala besar. Inilah alasan mengapa hanya segelintir organisasi dengan sumber daya besar yang mampu mengembangkan LLM dari nol.

Sebelum kita melihat contoh konkret, yuk uji pemahaman Anda tentang definisi dan tiga pilar utama yang membuat sebuah model disebut LLM. 

> **Aktivitas: Fill in the Blank**
> Lengkapi kalimat berikut dengan kata yang tepat.
> LLM adalah jenis model kecerdasan buatan yang dilatih menggunakan data **(teks)** dalam jumlah yang sangat besar. Istilah “Large” pada LLM mengacu pada tiga pilar utama, yaitu **(data)**, model, dan compute power. Pilar “Model” memiliki miliaran hingga triliunan **(parameter)** yang berfungsi seperti neuron buatan untuk mengenali pola dan memahami bahasa manusia.

Sekarang, mari kita lihat bagaimana konsep tersebut diterapkan melalui contoh model Llama 3. 

Meta menyebut bahwa model Llama 3.1 405B yang merupakan versi terbesar dengan lebih dari 400 miliar parameter, dilatih secara nonstop selama 54 hari menggunakan klaster 16.384 GPU NVIDIA H100-80GB. Proses pelatihan model ini saja menghabiskan sekitar 30,84 juta jam GPU (satuan yang menghitung berapa lama **SEBUAH** chip grafis bekerja penuh).

Artinya, jika hanya satu GPU yang dipakai, dibutuhkan lebih dari 3.500 tahun untuk menyelesaikan pekerjaan yang sama. Secara keseluruhan, konsumsi energi yang dilaporkan secara resmi oleh Meta mencapai sekitar 11 GWh, setara dengan listrik yang dibutuhkan 7.000 hingga 9.000 rumah tangga di Indonesia selama satu tahun penuh.

Perlu diketahui, Llama sendiri adalah LLM open source yang bebas dimanfaatkan dan dieksplorasi siapa saja. Kita akan membahas lebih jauh soal model open source di bagian berikutnya.

Kembali ke soal Large Language Model, salah satu hal yang sering disebut-sebut adalah jumlah parameternya yang bisa mencapai ratusan miliar, seperti yang kita lihat pada Llama tadi. Lalu, sebenarnya apa itu parameter?

Sederhananya, parameter adalah semua bobot (*weights*) dan bias yang tersebar di seluruh jaringan saraf tiruan (*neural network*) sebuah model. Nilai-nilai inilah yang "dipelajari" selama proses pelatihan dan menentukan bagaimana model merespons setiap input yang diberikan.

![Gambar 5](https://assets.cdn.dicoding.com/original/academy/dos-c50fe9679af9a8c20dc7b0d26c6680c620260429135251.png)

Jika digambarkan ke dalam perhitungan sederhana maka kita akan melihatnya dari interaksi bobot dan bias.

*   Bobot (weights) antara setiap neuron (misalnya, koneksi dari 768 neuron ke 768 neuron lain berarti 768 × 768 = 589.824 parameter).
*   Bias, yaitu nilai tambahan kecil per neuron (jadi +768 lagi).

Kalau satu lapisan punya sekitar 600 ribu parameter dan modelnya punya ratusan lapisan, lalu kita kalikan semuanya, jadilah ratusan juta atau bahkan miliaran parameter.

*“Model besar seperti GPT-3 misalnya, punya 96 lapisan dengan vektor dimensi 12.288. Kalau dihitung, jumlah parameternya bisa tembus lebih dari 175 miliar.”* 

Setelah berimajinasi dengan miliaran parameter yang dimiliki LLM, sekarang pertanyaannya, bagaimana dengan model yang memiliki 100 juta parameter? Untuk menjawab pertanyaan menarik tersebut, kita perlu berkenalan dengan konsep yang disebut Small Language Model atau SLM.

### SLM vs LLM

Setelah sudah cukup akrab dengan Large Language Model atau LLM, saatnya kita juga kenalan dengan “adik” atau versi ringan dari LLM, yaitu SLM.

*Yup*, Small Language Model secara sederhana adalah LLM dengan ukuran parameter yang jauh lebih kecil. Secara aturan tidak ada kategori pasti tentang seberapa besar parameter untuk sebuah Language Model dikatakan sebagai SLM. Sumber seperti Hugging Face menyatakan model dengan jumlah parameter mulai dari 1 juta hingga 10 miliar parameter dapat dikategorikan sebagai SLM, sementara LLM adalah model dengan jumlah parameter 10 miliar ke atas.

![Gambar 6](https://assets.cdn.dicoding.com/original/academy/dos-93a28ad372436df80c7cacd29bf46ffa20260429135323.png)

Namun, “small” pada SLM tidak berarti model tersebut benar-benar kecil seperti algoritma machine learning tradisional. SLM tetaplah model canggih berbasis arsitektur transformer, hanya saja dirancang agar lebih efisien dan ringan, baik dalam kebutuhan komputasi, memori, maupun energi.

SLM biasanya memiliki kemampuan yang cukup mumpuni untuk berbagai tugas bahasa, seperti *text classification*, *summarization*, atau bahkan *generation*, tetapi tidak sebesar atau seluwes LLM dalam memahami konteks panjang atau menghasilkan teks yang kompleks.

Meski begitu, kemajuan riset terbaru menunjukkan bahwa dengan teknik pelatihan khusus pada SLM dapat memberikan hasil yang mendekati performa LLM atau bahkan “mengalahkannya”, dengan biaya jauh lebih rendah.

Inilah mengapa SLM mulai banyak diadopsi di berbagai industri, terutama untuk aplikasi *edge computing*, perangkat *mobile*, atau skenario bisnis yang membutuhkan privasi tinggi dan *latency* rendah.

*Eits*, secara tidak langsung kita sudah membocorkan beberapa kelebihan SLM ketimbang LLM. Baiklah agar tetap relevan, Anda dapat memperhatikan tabel perbandingan di bawah untuk melihat “pertarungan” antara SLM vs LLM.

| LLM | SLM |
| :--- | :--- |
| Hingga 2 triliun parameter atau bahkah lebih | 100 juta - 10 miliar parameter |
| Di-*training* dengan dataset yang besar dan beragam | Di-*training* dengan dataset kecil dan spesifik (*niche*) |
| Proses pelatihan memakan waktu berbulan-bulan | Proses pelatihan memakan waktu beberapa hari |
| Membutuhkan daya komputasi tingkat lanjut | Dapat berjalan dengan daya komputasi dasar |
| Cocok untuk tugas kompleks dan umum | Cocok untuk tugas yang spesifik |
| Biaya fine-tuning mahal | Biaya fine-tuning lebih murah |
| Latensi lebih tinggi | Latensi lebih rendah |
| Biaya token tinggi | Biaya token rendah |
| Tidak dapat berjalan di perangkat (*on-device*) atau *edge devices* | Dapat berjalan di perangkat (*on-device*) atau *edge devices* |

Dari tabel di atas kita dapat membuat beberapa kata kunci pembeda antara LLM dan SLM. Hasilnya kita dapat membuat sebuah *guideline* untuk menjawab pertanyaan yang tertera pada judul bagian selanjutnya.

#### Kapan harus menggunakan LLM dan kapan dengan SLM

Seperti sebuah statement sakral bagi seluruh AI Engineer yang kita pegang teguh, *“Model yang besar belum tentu pilihan yang terbaik”*. Hal itu juga berlaku pada *generative AI*, khususnya antara LLM dan SLM. Kita perlu mengetahui terlebih dahulu *use case* atau *problem* yang kita ingin selesaikan.

![Gambar 7](https://assets.cdn.dicoding.com/original/academy/dos-8170efbdddf5f2654a0a5dbbd49405b120260313143448.png)

Barulah dari sana kita dapat mendapatkan pencerahan dalam memecah kembimbangan antara LLM dan SLM tersebut. Mari kita bahas satu per satu aspek yang perlu kita perhatikan dalam memilih.

*   **Kompleksitas Tugas** 
    Jika membicarakan soal kompleksitas tugas, tentu pada akhirnya jumlah parameter yang akan berbicara. Untuk pekerjaan yang menuntut pemahaman bahasa mendalam, seperti penulisan konten kreatif, analisis masalah kompleks atau general purpose, LLM adalah pilihan terbaik. Sebaliknya, SLM, dengan jumlah parameter yang lebih sedikit, SLM dapat menjadi sangat efektif, bahkan sering kali lebih cepat dan efisien, untuk tugas-tugas spesifik atau domain purpose.
*   **Ketersediaan Sumber Daya** 
    Akibat dari jumlah parameter yang jauh lebih banyak, LLM membutuhkan daya komputasi yang lebih besar, misalnya GPU atau TPU kelas atas. Model seperti ini cocok bagi perusahaan besar yang memiliki sumber daya melimpah. Jika anggaran dan infrastruktur terbatas, SLM adalah alternatif yang jauh lebih efisien. Model berukuran kecil ini bisa dilatih dan dijalankan pada perangkat keras yang lebih sederhana tanpa mengorbankan performa untuk tugas-tugas general.
*   **Deployment**
    Untuk aplikasi berskala besar dengan volume data tinggi dan kebutuhan analisis lintas domain, LLM lebih unggul. Namun, di lingkungan *edge computing* atau perangkat kecil seperti ponsel, SLM jauh lebih ideal. Model ini dapat berjalan langsung di perangkat, memberikan respons cepat tanpa koneksi *cloud* yang terus-menerus

*Alright*, mari kita uji pemahaman Anda melalui sedikit selingan aktivitas di bawah ini.

> **Aktivitas: Flashcards**
> - Daya komputasi pada CPU sudah cukup untuk menjalankannya. (**Jawaban:** SLM)
> - Cocok untuk berbagai macam kebutuhan dan analisis masalah yang kompleks. (**Jawaban:** LLM)
> - Kurang ideal jika harus dijalankan di dalam *edge computing* atau *smartphone*. (**Jawaban:** LLM)
> - Model dengan parameter di bawah 10 Miliar. (**Jawaban:** SLM)

Kita telah mengenal SLM dan membahas perbandingannya dengan LLM. Namun, bagaimana cara membuat LLM yang “Large” ini menjadi versi mini ini?

#### Bagaimana Cara Mereka Membuatnya “Small”?

Mengecilkan ukuran sebuah model bahasa bukan sekadar memangkas parameter secara acak. Proses ini melibatkan serangkaian teknik cerdas yang dirancang untuk menghemat sumber daya tanpa mengorbankan terlalu banyak performa. Terdapat tiga cara yang umumnya dilakukan.

*   **Knowledge Distillation:** Melatih model yang lebih kecil menggunakan knowledge yang berasal dari model yang lebih besar (LLM).
*   **Quantization:** Mengurangi tingkat presisi dari nilai numerik yang digunakan dalam perhitungan matematis language model.
*   **Pruning:** “Memangkas” beberapa parameter yang kurang berpengaruh atau redundan dalam arsitektur neural network.
*   **Parameter-Efficient Fine-Tuning (PEFT):** Melatih sejumlah kecil parameter tambahan atau *adapters* untuk tugas spesifik.
*   **Membuatnya “kecil” dari awal:** Mengurangi jumlah layer dan dimensi tersembunyi (d<sub>model</sub>) atau mengganti komponen inti pada tansformer dengan versi yang lebih efisien

*Eits*, jangan *overthinking* terlebih dahulu dengan beberapa istilah asing dan penjelasan singkat di atas. Kita akan membedah beberapa teknik di atas pada modul-modul berikutnya. Harapannya, teknik-teknik di atas dapat memancing rasa keingintahuan Anda untuk terus mengarungi perjalanan ini bersama-sama.

Untuk menutup bagian pertama ini, mari kita akhiri dengan sebuah pertanyaan. Setelah memahami definisi LLM dan perbedaannya dengan SLM, menurut Anda, apakah ChatGPT dan Gemini yang Anda gunakan sehari-hari termasuk LLM atau SLM?

Jika Anda menjawab LLM, berarti sekarang sudah waktunya kita lanjut ke bagian selanjutnya! *Nah*, pada bagian selanjutnya, kita akan membahas istilah “*Black box*” yang sering dengan konsep LLM. Pertanyaannya, apa maksud sebenarnya istilah tersebut?

Mari kita temukan jawabannya bersama!

### LLM Bagai Black box Sebuah Pesawat

Apakah Anda pernah mendengar kata “*Black box*” sebelumnya? Kata ini sering Anda dengar saat membahas soal pesawat terbang. Si Kotak hitam ini secara definisi adalah sebuah perangkat misterius yang menyimpan semua rekaman penting tentang penerbangan.

Menariknya, istilah yang sama kini sering digunakan untuk menggambarkan LLM seperti ChatGPT ataupun Gemini. Pernahkah kalian menemui keanehan seperti pada ilustrasi di bawah?

![Gambar 8](https://assets.cdn.dicoding.com/original/academy/dos-5997d93dda70b0e18b1dafdf1122f18d20260429135533.png)

Jika pernah, tenang saja hal itu wajar. Kita menanyakan hal yang sama *“Aku lapar banget sekarang… ”* dengan kalimat yang sama pada waktu yang sangat berdekatan, LLM dapat memberikan jawaban yang sedikit atau bahkan terkadang *totally* berbeda.

Pertanyaannya, kenapa hal tersebut dapat terjadi? *Nah*, hal itulah yang menjadi pertanyaan sebagian besar orang.

Banyak orang mengagumi kemampuannya menghasilkan teks yang cerdas, menjawab pertanyaan dengan cepat, dan bahkan menulis seperti manusia. Namun, ketika ditanya “bagaimana” model itu sampai pada jawabannya, kebanyakan dari kita hanya bisa mengangkat bahu. Sama seperti blackbox pesawat, LLM bekerja di dalam ruang yang tidak sepenuhnya bisa dilihat oleh orang awam.

Sebagai pengguna atau konsumen, kita mungkin sudah paham flow yang terjadi saat menggunakan LLM. Mulai dari memberikan prompt atau perintah ke LLM, lalu LLM memproses perintah tersebut, dan pada akhirnya LLM menghasilkan jawaban berdasarkan perintah.

Langkah nomer dua inilah yang menjadi pertanyaan kita semua. Proses apa yang terjadi saat LLM yang mempunyai miliaran atau triliunan parameter dapat meniru pola bahasa manusia

Namun, sebenarnya apa yang menyebabkan LLM berperilaku seperti blackbox? Jawabannya dapat kita bedah menjadi beberapa alasan.

*   **Proses Internal yang Rumit**
    Interaksi kompleks antara miliaran parameter ini membuat pemahaman tentang cara model mencapai kesimpulan tertentu menjadi hampir tidak mungkin dipahami oleh manusia. Output yang dihasilkan adalah hasil dari kombinasi matematis yang sangat rumit, bukan berdasarkan penalaran yang bisa dijelaskan.
*   **"Halusinasi" dan ketidakakuratan**
    Sifatnya sebagai kotak hitam juga berkontribusi pada fenomena "halusinasi," di mana LLM menghasilkan informasi yang terdengar meyakinkan tetapi tidak akurat atau sepenuhnya salah. Oleh karena prosesnya tidak transparan, sulit untuk melacak alasan pasti dari kesalahan ini disebabkan oleh faktor tertentu.
*   **Kurangnya transparansi data**
    Sering kali tidak jelas data apa saja yang digunakan untuk melatih model, termasuk sumbernya, komposisinya, dan apakah data tersebut mengandung bias tertentu atau tidak. Hal ini membuat audit model untuk mendeteksi bias atau masalah etika lainnya menjadi sulit.

Jika Anda sadari, penjelasan di atas mengatakan bahwa output yang dihasilkan berasal dari kombinasi matematis. *Yup,* pada dasarnya, ini adalah proses prediksi yang dengan kata lain LLM bekerja dengan cara memprediksi kata selanjutnya. Proses prediksi ini akan dilakukan secara terus-menerus dan inilah yang sering dikenal dengan istilah autoregresif.

Namun, sebelum kita membahas lebih jauh tentang cara model “berbicara” melalui mekanisme autoregresif, ada baiknya kita memahami terlebih dahulu cara model memahami inputnya.

*So, it’s time to dive deeper, Coders!*

---

## Kita Mulai dari Pemrosesan Input

Saat membahas “Bagaimana cara model bahasa memproses input” tentu saja pembahasan kita tidak akan jauh-jauh dari **Tokenization**. *Yup*, pembahasan kita akan fokus pada dua hal tersebut.

*Eits*, Anda mungkin bertanya-tanya alasan kita kembali membahas konsep dasar pemrosesan teks yang biasanya juga kita lakukan di machine learning. Pada kelas intermediate sebelumnya, yaitu [Kelas Belajar Fundamental Deep Learning](https://www.dicoding.com/academies/185-belajar-fundamental-deep-learning), kita sudah diajak mengenal tokenisasi sampai ke implementasi menggunakan *library* NLTK.

![Gambar 9](https://assets.cdn.dicoding.com/original/academy/dos-06b0350ca0706010bcabb1179978b47920260304134511.png)

Alasan yang membawa kita perlu berkecimpung lagi dengan Tokenization adalah karena kita akan berkenalan dengan versi yang digunakan oleh LLM dari teknik tersebut.

Lagipula, tahap ini adalah salah satu fondasi dari dunia Natural Language Processing (NLP) yang akan menjadi pintu pembuka yang menarik untuk memahami cara LLM bekerja. Jadi mari kita mulai dengan tahapan paling awal, yaitu tokenisasi.

### Tokenization

Seperti yang di-mention sebelumnya, pada kelas Belajar Fundamental Deep Learning sebenarnya kita sudah mengenal konsep tokenization. Konsep dasar tersebut tetap dilestarikan dalam model canggih seperti LLM.

Namun, model-model LLM yang dilatih dengan data bahasa yang sangat banyak membuatnya memerlukan sistem yang dapat mampu menangani kosakata yang sangat luas serta efisiensi komputasi yang tinggi. Inilah pengantar untuk teknik tokenization yang lebih canggih.

Seperti yang kita ketahui, tokenization adalah proses memecah kalimat menjadi potongan-potongan kecil yang disebut token. Tidak selamanya satu token sama dengan satu kata sebab tokenisasi sendiri memiliki *levelling* nya sendiri. Berbicara mengenai level dalam tokenisasi, mari kita bahas satu per satu untuk dapat menjawab bentuk tokenisasi yang digunakan dalam LLM.

#### Token Levelling

Secara *straightforward,* kita dapat mengelompokkan level tokenization menjadi tiga level. Dua level yang paling sederhana dan sering kita temui di antaranya adalah sebagai berikut.

![Gambar 10](https://assets.cdn.dicoding.com/original/academy/dos-2841c1c4a0e3c7563617381a9db6813920260429163127.png)

*   **Character-level** 
    Pada tingkat ini, teks atau kalimat akan dipecah berdasarkan *character* atau biasa kita sebut huruf. Ilustrasi di atas dan penjelasan tersebut seharusnya sudah cukup untuk memberikan gambaran dari teknik ini. Namun, sayangnya teknik ini tidak digunakan oleh LLM. Ada dua alasan utama.
    *   **Urutan Sequence Menjadi Terlalu Panjang** 
        Seperti yang dapat Anda lihat pula pada ilustrasi di atas, kalimat pendek seperti itu pun dapat menjadi *sequence* token yang sangat panjang. LLM sendiri memiliki batas *context length*, ditambah lagi *sequence* yang terlalu panjang berisiko akan boros secara komputasi.
    *   **Kehilangan Makna Semantik**
        Memecah kata menjadi unit *character* akan membuat model bekerja ekstra keras dari awal hanya untuk mempelajari konsep kata. Hal ini seperti mencoba memahami sebuah buku dengan membacanya huruf per huruf tanpa mengelompokkannya menjadi kata terlebih dahulu yang mana ini tidak efisien.
*   **Word-level** 
    Kelemahan pada *character-level* tampaknya dapat diatasi oleh *word-level*. Namun, teknik yang memecah teks menjadi unit kata ini nyatanya memiliki kelemahan lain yang membuatnya kurang cocok untuk LLM.
    *   **Ukuran Vocabulary Raksasa** 
        Untuk bahasa yang kaya morfologi seperti bahasa Indonesia yang mana kata "makan", "memakan", "dimakan", "makanan" adalah kata berbeda, ukuran kosakata dapat meledak hingga jutaan kata. Ini membuat model sangat besar dan lambat.
    *   **Masalah Out-of-Vocabulary (OOV)** 
        Jika model bertemu kata yang tidak ada dalam kosakatanya saat *training*, ia akan menggantinya dengan token `<UNK>` (*unknown*). Akibatnya, model kehilangan semua makna dari kata tersebut.
    *   **Hubungan Morfologis** 
        Model tidak secara inheren tahu bahwa "bermain" dan "pemain" berasal dari akar kata yang sama, yaitu "main". Baginya, itu adalah dua token yang sama sekali berbeda dan tidak berhubungan.

Lantas solusi apa yang digunakan oleh LLM dalam melakukan tokenisasi agar lebih efisien? Jawabannya adalah teknik yang berada di antara *Word-level* dan *Character-level*, yaitu *Subword-level*.

![Gambar 11](https://assets.cdn.dicoding.com/original/academy/dos-508f2422de47f141e53625a52825cf2e20260429163247.png)

*Subword-level* tokenisasi dapat menyeimbangkan antara jumlah vocabulary dengan makna semantik pada setiap kata. Secara sederhana, *subword-level* akan memecah kata menjadi sub-kata. Namun, mengapa pendekatan seperti ini efektif dan lebih efisien?

Pertanyaan tersebut sebenarnya dapat dijawab melalui ilustrasi di bawah ini.

![Gambar 12](https://assets.cdn.dicoding.com/original/academy/dos-c422d996a0463f7025788818c76c72fc20260304134808.png)

Pada word-level, setiap kata disimpan sebagai entri tersendiri di dalam *vocabulary*. Artinya, kata seperti bermain, pemain, dan permainan dianggap tiga unit yang berbeda. Akibatnya, semakin banyak variasi kata yang muncul, semakin besar pula ukuran *vocab*-nya.

Sebaliknya, *subword-level* memperkenalkan konsep *reusable* pada sub-kata dengan memecahnya berdasarkan awalan (*per-*, *ber-*), kata dasar (*main*, *lari*), dan akhiran (*-an*). Dengan cara ini, model tidak perlu menyimpan ribuan kata turunan yang berbeda. Ia cukup memahami bagaimana potongan-potongan kecil ini bisa disusun ulang untuk membentuk berbagai kata baru.

Inilah alasan mengapa pendekatan *subword-level* jauh lebih efisien. Selain menghemat ukuran vocabulary, model juga menjadi lebih fleksibel dalam mengenali kata baru yang belum pernah muncul sebelumnya. Misalnya, meskipun model belum pernah melihat kata “berlarian,” ia tetap dapat menebak maknanya karena mengenali pola dari *ber-* + *lari* + *-an*. Hasilnya, jumlah token OOV dapat diminimalkan.

*Nah*, setelah kita berteori bersama sebelumnya, sekarang waktunya kita berkenalan dengan algoritma yang akan membawa kita untuk dapat mengimplementasikan tokenisasi di level *subword*.

#### Algoritma Subword-level Tokenization

Tokenisasi pada level *subword* pada dasarnya dapat dilakukan dengan tiga algoritma yang akan kita bahas di bawah ini. Perlu diketahui, beberapa algoritma berikut ini banyak digunakan oleh model modern seperti BERT dan GPT.

Tiga algoritma tersebut adalah Byte Pair Encoding (BPE), WordPiece, dan SentencePiece. Meskipun tujuan ketiganya sama, yaitu memecah teks menjadi satuan terkecil yang tetap bermakna, masing-masing memiliki cara kerja dan karakteristik yang berbeda. Mari kita bahas satu per satu.

*   **Byte Pair Encoding (BPE)** 
    Algoritma ini berangkat dari prinsip sederhana, mulai dari kumpulan karakter dasar. BPE kemudian mencari pasangan karakter atau token yang paling sering muncul secara berdekatan di seluruh korpus data, lalu menggabungkannya menjadi satu token baru. Proses ini diulang hingga terbentuk kosakata dengan ukuran tertentu. Pendekatan ini membuat BPE sangat efisien dan banyak digunakan dalam model generatif seperti GPT-2.
*   **WordPiece** 
    Secara konseptual, WordPiece adalah evolusi dari BPE. WordPiece tidak sekadar menghitung frekuensi kemunculan pasangan token. Saat memutuskan untuk menggabungkan pasangan sub-kata, ia mempertimbangkan pertanyaan kunci: *"Seberapa besar kontribusi pasangan token ini dalam meningkatkan probabilitas model untuk memprediksi sisa kata?"*. Dengan pendekatan yang lebih probabilistik ini, WordPiece cenderung menghasilkan pembagian *subword* yang lebih stabil dan sesuai dengan konteks linguistik.
*   **SentencePiece** 
    Algoritma yang dikembangkan oleh Google ini bekerja langsung dari teks mentah tanpa harus memisahkan kata berdasarkan spasi. Artinya, algoritma ini dapat digunakan pada bahasa-bahasa yang tidak memiliki pemisah kata eksplisit, seperti bahasa Jepang atau Korea. Selain itu, SentencePiece mendukung dua mode utama BPE dan *Unigram Language Model* yang memungkinkan kita menyesuaikan metode tokenisasi sesuai kebutuhan model.

Melalui proses tokenisasi, kita telah melihat proses dari bahasa manusia diuraikan menjadi potongan-potongan kecil yang dapat dipahami mesin. Tahap selanjutnya dari potongan-potongan kecil tersebut adalah pintu embedding yang akan kita bahas lebih dalam di materi terpisah sela

Kini, tiba saatnya kita mencari tahu cara potongan-potongan itu disatukan kembali melalui teknik yang disebut Autoregressive

---

## LLM Berbicara Melalui Autoregressive

Sesuai janji sebelumnya, setelah mempelajari cara LLM memproses input, sekarang kita akan menjelajah *stage* berikutnya: *“bagaimana LLM ini berbicara kembali kepada kita?”*

Sebelumnya kita setuju bahwa proses *tokenization* dan dapat diibaratkan sebagai cara LLM “mendengarkan” dan “memahami” bahasa manusia. *Nah*, tahap selanjutnya yang disebut Autoregressive adalah cara ia “menyusun kata-kata” untuk menjawab atau melanjutkan teks.

Mari kita membuka lembaran baru ini dengan “mengoprek” konsep autoregressive

### Konsep Autoregressive

Secara sederhana, autoregressive berarti “membangun sesuatu berdasarkan apa yang sudah ada sebelumnya.” Sebelumnya kita menyebut autoregressive ini sebagai cara LLM menyusun kata.

Hal ini sejalan dengan cara kerja LLM yang tidak langsung menuliskan seluruh jawabannya sekaligus. Ia justru menyusun kata demi kata, atau lebih tepatnya token demi token, dengan cara memprediksi apa yang paling mungkin muncul setelah rangkaian sebelumnya. Anda dapat memperhatikan ilustrasi di bawah sebagai gambaran dari proses Autoregressive yang terjadi pada LLM.

![Gambar 13](https://assets.cdn.dicoding.com/original/academy/dos-66e6efc4a0e853d6da09d14499d834ff20260313143742.gif)

**Quick Facts** 

*“EOS atau End Of Sequence adalah token yang memberi tahu model bahwa urutan kalimat atau teks sudah selesai, sehingga LLM dapat berhenti untuk menghasilkan kata.”* 

Agar lebih menarik, coba Anda jalankan potongan kode di bawah ini dan perhatikan output yang dihasilkan. Tenang saja, Anda tidak diharuskan mempelajari seluruh isi kode di bawah ini dan cukup jalankan saja.

```python
import random
import time
from collections import defaultdict

corpus = """
llm adalah model bahasa .
model ini bekerja secara autoregressive .
autoregressive berarti memprediksi token berikutnya .
token berikutnya bergantung pada token sebelumnya .
model bahasa ini sangat besar .
llm memprediksi kata demi kata .
setiap kata adalah token baru .
model ini belajar dari data .
"""

tokens = corpus.lower().split()
model = defaultdict(list)

for current_word, next_word in zip(tokens, tokens[1:]):
    model[current_word].append(next_word)

prompt = "llm"
max_length = 15

generated_text = [prompt]
current_word = prompt

print(f"Prompt: {prompt}")
print(f"Generated: {current_word}", end=' ', flush=True)

for _ in range(max_length - 1):
    time.sleep(0.7)
    
    if current_word not in model:
        break
        
    possible_next_words = model[current_word]
    next_word = random.choice(possible_next_words)
    generated_text.append(next_word)
    print(f"{next_word}", end=' ', flush=True)
    current_word = next_word
    
    if current_word == ".":
        break
```

Output-nya akan nampak seperti di bawah ini.

![Gambar 14](https://assets.cdn.dicoding.com/original/academy/dos-1fa4963734735082725f88b5165826ad20260502115414.gif)

*Yup, *secara “blak-blakan”, autoregressive ini adalah “*looping* prediksi versi canggih” yang dapat mempertimbangkan konteks input. *Nah*, agar rasa ingin tahu kita dapat termanjakan dengan lebih puas, mari kita *breakdown* lebih dalam *step-by-step* dalam mekanisme autoregressive.

#### Step by Step Autoregressive

Sebelumnya kita menganalogikan teknik autoregressive seperti *looping*. *Nah*, menariknya, proses ini tidak berhenti pada sekadar pengulangan prediksi.

Di balik setiap langkah “*looping*” itu, model melakukan analisis yang kompleks terhadap konteks yang sudah terbentuk. Ia menimbang hubungan makna antar token, menyesuaikan gaya bahasa, dan menjaga agar keseluruhan kalimat tetap koheren. Inilah yang membuat hasil keluaran LLM terasa begitu alami.

![Gambar 15](https://assets.cdn.dicoding.com/original/academy/dos-18ab34956dc13cebaec002fec2d9ffd820260304155933.png)

Proses ini berlangsung sangat cepat, berulang ribuan kali per detik, hingga terbentuklah satu kalimat utuh, paragraf, atau bahkan halaman penuh teks. Jika kita gali lebih dalam lagi, prosesnya dapat dibagi menjadi tiga tahapan. Anda dapat memperhatikan ilustrasi di bawah ini, kemudian membaca penjelasan ketiga tahapannya.

1.  **Menentukan Probabilitas Kata Berikutnya** 
    Setiap kali model menerima serangkaian token (kata atau potongan kata), ia akan menghitung probabilitas dari setiap token yang mungkin muncul selanjutnya. Proses ini tidak dilakukan secara acak karena model mempertimbangkan konteks, tata bahasa, dan pola makna dari teks sebelumnya. Misalnya seperti contoh di atas, setelah kata *“Autoregressive adalah…”*, kata “teknik” tentu akan mendapat probabilitas lebih tinggi dibandingkan kata “makanan”.
2.  **Menghasilkan Output Sementara** 
    Setelah model menentukan token dengan probabilitas tertinggi, token tersebut dipilih dan ditambahkan ke deretan teks yang sudah ada. Pada tahap ini, hasilnya belum final dan bisa dianggap sebagai “*draft* sementara” seperti halnya kata *“Autoregressive adalah teknik…*” pada ilustrasi di atas. Namun, inilah langkah awal yang menentukan arah kalimat berikutnya.
3.  **Melakukan Looping Dua Proses di Atas** 
    Selanjutnya, model akan mengulangi langkah pertama dan kedua, hanya saja kali ini dengan konteks yang sudah diperbarui dan juga termasuk token baru yang sebelumnya dihasilkan. Proses ini berlangsung berulang-ulang hingga model mencapai batas panjang teks yang diinginkan atau menemukan tanda untuk berhenti.

Hal yang menarik adalah, tidak ada “naskah” tetap yang disimpan di dalam model karena semua yang keluar dari “mulut” LLM adalah hasil perhitungan probabilitas dan kemungkinan kata baru muncul itu tidaklah nol.

![Gambar 16](https://assets.cdn.dicoding.com/original/academy/dos-90508f16d2cf39029b87d0b4a47592af20260313143921.gif)

Inilah bukti bahwa LLM bukan sekadar mesin penghafal data, melainkan sistem berpikir statistik yang luar biasa kompleks.

Setiap kali model memprediksi token berikutnya, ia mempertimbangkan banyak kemungkinan dan setiap kemungkinan itu memiliki peluangnya sendiri. Karena distribusi probabilitas ini selalu bergantung pada konteks percakapan yang sedang berlangsung, bahkan ketika kita mengajukan pertanyaan yang sama, jawabannya bisa berbeda setiap saat.

*Nah*, dari sini kita telah mendapatkan jawaban dari pertanyaan kita di awal bagian “LLM Bagai Blackbox Pesawat”. Berbicara soal autoregressive, konsep ini telah diadopsi untuk membuat hampir seluruh LLM, sebut saja seperti Gemini, ChatGPT, dan bahkan Llama. Seluruh LLM tersebut akan memberikan respon yang berbeda meskipun Anda menggunakan input yang sama. Hal ini sudah kita singgung sebelumnya, tetapi kita belum menemukan jawaban pastinya.

Lantas, apa sebenarnya yang membuat jawaban antar LLM, atau antar satu generasi ke generasi berikutnya, bisa berbeda-beda? Jawabannya mulai terlihat ketika kita menengok lebih dalam bagaimana LLM menentukan token output berikutnya yang ternyata dibagi menjadi beberapa strategi.

### Strategi LLM dalam Memilih Token Berikutnya

Saat kita mengetikkan sebuah input atau pertanyaan ke LLM lalu menekan tombol Enter, LLM akan mulai mengambil keputusan satu per satu untuk menentukan token apa yang akan dikeluarkan berikutnya secara bertahap.

Seperti yang telah kita bahas sebelumnya, mekanisme next token generation yang berlandaskan konsep autoregressive ini akan diulang terus-menerus, hingga pada akhirnya model menganggap bahwa jawaban tersebut sudah terbentuk secara utuh.

Konsep sederhana yang kita setujui bersama sebelumnya menjelaskan bahwa LLM akan memilih token dengan probabilitas tertinggi. Strategi ini sering dipanggil sebagai Greedy Decoding yang akan kita bahas berikut ini.

#### Greedy Decoding

Secara harfiah, "Greedy" berarti tamak atau rakus. *Nah*, dalam konteks Ilmu Komputer, algoritma greedy adalah pendekatan memecahkan masalah dengan selalu mengambil pilihan terbaik yang tersedia saat itu juga, tanpa memedulikan konsekuensi di masa depan.

Sesuai dengan konsep tersebut, teknik ini akan secara *straightforward* mengambil token dengan probabilitas tertinggi pada langkah waktu tersebut, dan langsung menjadikannya bagian permanen dari kalimat, tanpa pernah menengok ke belakang lagi. Misalnya kita punya kalimat *“My cat is”* sebagai input LLM, token berikutnya akan dihitung dengan persamaan probabilitas seperti berikut.

![Gambar 17](https://assets.cdn.dicoding.com/original/academy/dos-699f28e39b1eefc2b2efef37649a1c8120260313144334.gif)

Persamaan probabilitas pada ilustrasi tersebut menunjukkan seberapa besar kemungkinan (***P***) kata berikutnya adalah kata ***w***, ketika model sudah memiliki konteks **C** dari kata-kata sebelumnya.

Dalam Greedy Decoding, token berikutnya yang akan muncul saat kalimat inputnya *“My Cat is”* 100% adalah kata *“Kind”* yang dalam contoh ini, kata tersebut memiliki probabilitas tertinggi. Ia akan langsung mengabaikan seluruh kata-kata yang probabilitas nya lebih rendah. Terdengar sederhana dan sangat masuk akal bukan untuk dijadikan pendekatan dalam model LLM.

Sayangnya, pendekatan ini pada akhirnya sangat jarang digunakan dalam LLM modern. Hal ini dikarenakan kekurangan krusial yang dimilikinya, salah satunya adalah Locally Optimal. Apa artinya itu?

Mari gunakan kalimat sebelumnya sebagai contoh input dan perhatikan ilustrasi di bawah ini.

![Gambar 18](https://assets.cdn.dicoding.com/original/academy/dos-08483e64fbfde32ef913fcbf192e125120260502115553.gif)

Konsep Greedy decoding yang selalu memilih token dengan probabilitas tertinggi pada setiap langkah ternyata menjadi senjata makan tuan. Pada ilustrasi tersebut, Greedy Decoding akan langsung memilih Jalur atas atau kata “kind” karena pada langkah pertama terlihat lebih unggul, yakni 0.7 dibandingkan 0.5. Padahal, jika dilihat secara keseluruhan kalimat, Jalur bawah atau kata “fluppy” menghasilkan probabilitas akhir yang lebih baik, yaitu 0.45 dibandingkan 0.21.

Kondisi inilah yang disebut sebagai *Locally Optimal*, di mana model mengambil keputusan terbaik pada satu langkah tertentu, tetapi gagal menghasilkan kalimat yang optimal secara keseluruhan (*Globally Optimal*). Akibatnya, teks yang dihasilkan dengan Greedy Decoding sering terasa kaku dan, pada kalimat yang lebih panjang, berisiko kehilangan alur atau konteks pembahasan.

Kelemahan besar ini lah yang mendorong akan diselesaikan oleh pendekatan berikutnya, yaitu Beam Search.

#### Beam Search

Pendekatan yang menjadi solusi Greedy Decoding ini bekerja dengan membuat beberapa branch atau cabang secara langsung untuk mempertahankan ***k*** node paling menjanjikan pada setiap langkah. Variabel *k* ini disebut Beam Width yang umumnya, nilainya adalah 5 hingga 10.

Bayangkan Greedy Decoding adalah pelari sprint yang hanya melihat satu meter ke depan, sedangkan Beam Search adalah pemain catur yang memikirkan beberapa kemungkinan langkah sekaligus. Artinya, kemungkinan besar Beam Search akan memerlukan waktu pemrosesan yang lebih lama ketimbang Greedy Decoding demi mendapatkan kombinasi token yang optimal secara keseluruhan.

Sekarang agar lebih menarik, mari kita gunakan contoh kalimat sebelumnya pada ilustrasi berikut.

![Gambar 19](https://assets.cdn.dicoding.com/original/academy/dos-a779950f38f365f6b07eeb6125ea043c20260502115607.gif)

Pada dasarnya, proses Beam Search dilakukan dalam 3 tahapan yang akan kita breakdown satu persatu di bawah ini.

*   **Inisialisasi** 
    Seperti pendekatan sebelumnya, LLM akan melihat semua kemungkinan token yang ada. Bedanya, karena menggunakan *Beam Width* atau *k* sebesar 2, kita akan mempertahankan 2 token dengan probabilitas tertinggi pada tahap ini. Kita akan sebut saja dua jalur ini sebagai jalur A dan jalur B.
    *   **Jalur A:** *"My cat is **kind**"* (Score: 0.7)
    *   **Jalur B:** *"My cat is **fluppy**"* (Score: 0.5)
*   **Branching** 
    Sekarang, model harus memprediksi kata setelahnya untuk kedua jalur tersebut. Misalkan vocabulary kita hanya punya 10.000 kata, maka kita akan mengevaluasi 2 x 10.000 kemungkinan.
    *   Jalur A (kind)
        *   *...kind **and*** (P=0.3): Total Score A1: 0.7 x 0.3 = 0.21
        *   *...kind **to*** (P=0.2): Total Score A2: 0.7 x 0.2 = 0.14
        *   *...kind **of*** (P=0.2): Total Score A3: 0.7 x 0.2 = 0.14
    *   Jalur B (fluppy)
        *   *...fluppy **sleeping*** (P=0.9): Total Score A1: 0.5 x 0.9 = 0.45
        *   *...fluppy **as*** (P=0.1): Total Score A1: 0.5 x 0.1 = 0.05
        *   *...fluppy **and*** (P=0.4): Total Score A1: 0.5 x 0.4 = 0.20
*   **Pruning** 
    Sekarang kita punya daftar kandidat urutan kalimat beserta skor totalnya:
    *   *...fluppy sleeping* (0.45)
    *   *...kind and* (0.21)
    *   *...fluppy and* (0.20)
    *   *...kind to* (0.14)
    *   *...kind of* (0.14)
    *   *...fluppy as* (0.05)

Oleh karena **Beam Width = 2**, kita hanya simpan dua terbaik, yaitu jalur *"...fluppy sleeping"* dan *"...kind and"*. *Nah*, sementara jalur lainnya akan diabaikan, misalnya jalur *"kind of"* (Meskipun *"kind"* awalnya menang di langkah pertama, sekarang jalur ini diabaikan karena kalah secara kumulatif).

Solusi ini lah yang ditawarkan Beam Search untuk menggantikan Greedy Decoding. Ia menyadari bahwa meskipun *"fluppy"* awalnya kalah dari *"kind"*, gabungan *"fluppy sleeping"* ternyata memiliki probabilitas yang lebih besar daripada opsi apapun yang berawal dari "kind".

Lantas, apakah Beam Search ini lah yang digunakan oleh para LLM modern? Sayangnya jawabannya juga bukan.

Sebenarnya, Beam Search ternyata belum menyelesaikan seluruh kekurangan yang dimiliki oleh Greedy Decoding, yaitu hasil output yang kaku dan tidak kreatif. Sama seperti Greedy Decoding, kombinasi Token yang tercipta akan selalu sama. Saat Anda menggunakan kalimat *“My cat is”* pasti token selanjutnya akan selalu antara *“kind”* atau *“fluppy”*. Hal ini tentu membuatnya tidak cocok untuk tugas-tugas yang membutuhkan kreatifitas atau variasi jawaban seperti content writing.

Pendekatan Beam Search ini akan sangat sesuai untuk tugas-tugas yang membutuhkan akurasi tinggi dan jawaban yang eksak atau pasti seperti Machine Translation atau Summarization.

Jika Beam Search adalah tentang "Mencari Kebenaran Mutlak", maka pendekatan berikutnya adalah tentang "Mencari Variasi dan Kreativitas". Inilah yang menjadi jantung dari kemampuan LLM modern, seperti GPT, Claude, dan Gemini untuk menulis puisi, membuat kode yang unik, atau berinteraksi layaknya manusia.

#### Metode Sampling

Sebelumnya kita sepakat bahwa baik Greedy Decoding maupun Beam Search selalu memilih token dengan probabilitas tertinggi. Behaviour tersebut membuat jawaban model selalu datar, membosankan, dan repetitif. Model tidak berani mengambil risiko dan variasi token berikutnya untuk dijadikan sebagai output.

Ibarat kita berbicara dengan script pidato formal yang bentuk dan kosakatanya sudah ditentukan oleh aturan yang dibuat sebelumnya.

Sedangkan dalam kehidupan sehari-hari, kita sebagai manusia saat berbicara jarang sekali memilih kata yang paling "aman" secara statistik setiap saat. Kita sering memilih kata-kata yang sedikit tidak terduga untuk memberi warna pada percakapan.

Untuk meniru konsep ini, kita memperkenalkan *stochasticity* atau keacakan. Kita tidak lagi memilih berdasarkan argmax atau nilai tertinggi, tetapi kita mengambil sampel dari distribusi probabilitas tersebut.

Baiklah untuk memahami cara kerjanya kita perlu kembali lagi memahami cara LLM menentukan prediksi token yang akan menjadi outputnya. *Yup*, melalui perhitungan logits yang dikonversi menjadi probabilitas oleh Softmax.

![Gambar 20](https://assets.cdn.dicoding.com/original/academy/dos-c5895057f31eff4dd0036a29be4f7d5520260313144703.png)

Pendekatan Greedy dan Beam akan memanfaatkan nilai perhitungan probabilitas dari softmax ini secara apa adanya dan mendapatkan token yang menjadi output. Dalam pendekatan sampling, kita akan memodifikasi rumus softmax dengan menambahkan satu variabel bernama Temperature.

Namun, sebelum itu kita perlu membedah komponen utama dalam rumus softmax di atas.

*   *z*<sub>*i* </sub>merupakan Logits atau skor mentah dari model sebelum dijadikan probabilitas. Logits bisa berupa angka positif, negatif, atau nol.
*   *e* adalah fungsi eksponensial. Konstanta *e* (Bilangan Euler) memiliki nilai kira-kira 2,718.

Perlu Anda ketahui, fungsi eksponensial itu sangat sensitif dengan besar nilai pangkatnya. Namun, apa maksudnya? Coba perhatikan penjelasan berikut

*   2<sup>10 </sup>nilainya sangat jauh lebih besar dibanding 2<sup>2</sup>
*   2<sup>3</sup> nilainya hanya sedikit besar dari 2<sup>1</sup>.

*Nah*, disinilah Temperature akan masuk sebagai game changer. Temperature bekerja dengan cara mengubah skala (scaling) angka logits (z<sub>i</sub>) sebelum masuk ke fungsi eksponensial. Coba Anda perhatikan ilustrasi di bawah ini.

![Gambar 21](https://assets.cdn.dicoding.com/original/academy/dos-e455c95e3fa8d8f5f872ddbc97c63bdc20260313144718.png)

Saat temperature ditambahkan, distribusi probabilitas dari setiap kandidat token menjadi lebih seimbang dibandingkan normal softmax. Alhasil, token-token yang sebelumnya tidak akan diperhitungkan karena kesenjangan nilai probabilitas yang amat jauh, kini memiliki probabilitas yang lebih besar untuk muncul.

Baiklah mari kita buat contoh sederhana agar Anda dapat memahami ini sepenuhnya.

Misalnya, model kita memiliki dua kandidat output token, yaitu *“Belajar”* dengan nilai logit sebesar 10, dan token *“Tidur”* yang nilai logit-nya 5. Sekarang mari kita bagi menjadi dua skenario temperature.

*   **Skenario nilai Temperature tinggi (Misal T = 5** **)** 
    Kita bagi nilai logits dengan Temperature.
    *   **Logit *"Belajar"***: 10/5 = 2
    *   **Logit *"Tidur"***: 5/5 = 1

Sekarang kita masukan ke fungsi eksponensial: e<sup>2 </sup>≈ 7.3 dan e<sup>1 </sup>≈ 2.7.

*   **Skenario nilai Temperature rendah (Misal T = 0.5)** 
    Kita bagi nilai logits dengan Temperature.
    *   **Logit *"Belajar"***: 10/0.5 = 20
    *   **Logit *"Tidur"***: 5/0.5 = 10

Sekarang kita masukan ke fungsi eksponensial sama seperti sebelumnya: **e<sup>20 </sup>≈ 485.000.000** (nilai yang sangat besar) dan **e<sup>10 </sup>≈ 22.000** (nilai yang besar, tetapi tidak seberapa dibandingkan 485 juta). Artinya, perbandingan antara probabilitas kata “Belajar” dengan kata "Tidur" menjadi sekitar **20.000** **: 1**. Sehingga, kemungkinan munculnya kata "Belajar" mendekati 100%, sedangkan kata “Tidur” mendekati 0%.

*Nah*, dari contoh di atas kita mendapatkan kesimpulan, semakin besar nilai temperature, semakin dekat distribusi nilai pada setiap tokennya seperti ilustrasi di bawah ini. Dampaknya, pilihan output token yang dimiliki LLM akan sangat bervariasi dan tidak bisa ditebak.

![Gambar 22](https://assets.cdn.dicoding.com/original/academy/dos-6817f20775af3fd6f25c61f0d3ee21ba20260313144631.gif)

Setelah kita mendapatkan cara untuk mengatur probabilitas dari setiap token, sekarang bagaimana cara untuk memilih dan menyaring setiap token yang muncul? Metode sampling memiliki dua jawaban, Top-K dan Top-P.

Namun, sebelum masuk ke sana, kenapa token-token tersebut perlu disaring?

Sebab meskipun kita sudah menaikkan Temperature, "ekor" distribusi probabilitas yang berisi kata-kata yang sangat tidak nyambung masih memiliki peluang kecil untuk terpilih (misal 0.0001%) dan jumlahnya bisa hingga puluhan ribu kata. Oleh karena itu, dibutuhkan filter untuk memangkas kandidat yang tentunya akan lebih efisien. Sekarang mari kita bedah perbedaan antara Top-K dan Top-P.

*   **Top-K Sampling** 
    Mekanisme Top-K sangat straightforward, *"Saya hanya mau melihat K kandidat terbaik. Sisanya buang"*. Misalnya kita jika kita set K-nya adalah 5, ia hanya akan memilih 5 token dengan probabilitas tertinggi dan membuang sisanya. Namun, pendekatan ini memiliki kelemahan.
    *   **Skenario Input:** * **"Ibu kota Indonesia pada tahun 2020 adalah..."** * 
        Model sangat yakin token selanjutnya adalah Jakarta (99%). Namun, karena K=5, model dipaksa mengambil 4 token lain yang salah (misalnya Bandung, Surabaya, Medan, Bali) untuk memenuhi kuota 5. Ini meningkatkan risiko model memilih jawaban salah hanya karena harus memenuhi kuota.
    *   **Skenario Input:** * **"Di dalam hutan itu, aku melihat seekor..."** * 
        Kandidat token selanjutnya yang valid sangat bervariasi, sehingga model ragu (mungkin ada 20 hewan yang masuk akal). Namun, karena K=5, model terpaksa membuang opsi ke-6 (misal "Monyet") padahal bisa saja itu kata yang sangat valid dan kreatif.
*   **Top-P(nucleus) Sampling** 
    Metode ini diperkenalkan oleh Holtzman dkk. (2019) dan menjadi standar de facto untuk LLM modern (ChatGPT, Claude, Llama). Metode ini mengatasi kekakuan Top-K yang mana alih-alih membatasi jumlah kata, ia membatasi Total Probabilitas Kumulatif (P). Anda dapat membayangkannya seperti ini, *"Ambil sebanyak apapun kata yang diperlukan, asalkan total keyakinan kita mencapai 90%"*. Mari kita gunakan dua contoh sebelumnya.
    *   **Skenario Input:** * **"Ibu kota Indonesia pada tahun 2020 adalah..."** * 
        Kita set Top-P sebesar 95, sedangkan Model sangat yakin token selanjutnya adalah Jakarta (99%). Satu kata saja total probabilitasnya sudah di atas 95, sehingga Top-P akan langsung berhenti dan secara langsung memutuskan token “Jakarta” sebagai output.
    *   **Skenario Input:** * **"Di dalam hutan itu, aku melihat seekor..."** * 
        Oleh karena kandidat tokennya sangat bervariasi sehingga probabilitas setiap tokennya menjadi kecil-kecil (misalnya, harimau (0.21), monyet (0.19),...), Top-P mungkin butuh mengambil 20 hewan berbeda agar totalnya mencapai 0.90. Sehingga jawabannya tetap bervariasi dan kreatif.

Meskipun metode sampling ini memiliki dua pendekatan, dalam prakteknya keduanya digunakan secara bersama-sama. Urutan konfigurasinya akan seperti berikut ini.

*   **Diawali dengan Temperature:** Digunakan di awal untuk membentuk distribusi probabilitas bersama softmax sebelum masuk ke filtering Top-K dan Top-P.
*   **Baru kemudian Top-K**: Set K=40, sehingga akan memangkas puluhan ribu token yang sangat tidak berguna dan menghemat komputasi.
*   **Ditutup dengan Top-P:** Dari 40 kata yang sudah disaring Top-K, jalankan logika Nucleus (P=0.9) untuk menentukan batas potong dinamisnya.

Pada akhirnya, metode sampling seperti Temperature, Top-K, dan Top-P hanyalah “pintu keluar” dari sebuah proses autoregressive yang panjang dan kompleks. Distribusi probabilitas yang digunakan untuk memilih token berikutnya tidak pernah muncul secara instan. Ia merupakan hasil dari serangkaian mekanisme internal yang bekerja jauh sebelum tahap sampling itu sendiri.

Untuk benar benar memahaminya, kita perlu melihat apa saja komponen yang bekerja di balik layar LLM. Salah satu nama yang hampir pasti sudah tidak asing di telinga kita adalah Transformer. Namun, sebelum masuk ke sana, ada baiknya kita mundur sejenak ke “garis start”. Transformer tidak lahir begitu saja. Ia hadir sebagai jawaban atas keterbatasan pendekatan sebelumnya.

Semua berawal dari sequence model. Model inilah yang pertama kali memperkenalkan cara mesin memahami data sebagai rangkaian urutan, serta menjadi fondasi bagi konsep autoregressive yang hingga kini masih menjadi jantung dari model generatif modern.

*Eits*, sebelum berpindah ke bagian selanjutnya, ada “*medical check up*” singkat pada pemahaman Anda mengenai strategi LLM dalam melakukan autoregressive.

> **Aktivitas: Flashcards**
> - Mengeksplorasi dan menyimpan beberapa jalur kandidat kalimat terbaik secara bersamaan berdasarkan nilai "k". (**Jawaban:** Beam Search)
> - Memilih token selanjutnya secara acak dari sekumpulan kandidat probabilitas tertinggi untuk menghindari teks yang kaku. (**Jawaban:** Metode Sampling)
> - Menghasilkan output yang deterministik (pasti sama setiap kali dijalankan) jika parameternya tidak diubah. (**Jawaban:** Greedy Decoding)

### Kita Mulai dari Sequence Model

Perkembangan *Generative AI* terutama LLM tidak akan pernah lepas dari saudara tirinya, yaitu *sequence model* yang menjadi pondasi awal terbentuknya konsep *autoregressive*. Jika membicarakan soal *sequence model*, mungkin dalam benak Anda sudah dipenuhi dengan RNN, LSTM, ataupun GRU.

Namun, tahukah Anda bahwa jauh sebelum model-model *deep learning* tersebut eksis dan mendominasi, konsep sequence prediction sudah lama dirumuskan. Pembicaraan ini akan mengarahkan kita seorang matematikawan Rusia bernama Andrey Andreyevich Markov.

Pada tahun 1906, matematikawan dari negara beruang merah tersebut memperkenalkan sebuah konsep bernama Markov Chain yang digunakan untuk memodelkan kejadian yang saling bergantung secara berurutan atau yang kita kenal sekarang sebagai *sequence*.

Sebelum terlalu jauh, mari kita pahami induk dari *sequence model* ini terlebih dahulu melalui ilustrasi berikut.

![Gambar 23](https://assets.cdn.dicoding.com/original/academy/dos-63c899ad003909893033e6aac3d0d44920260313144834.png)

*   **State (keadaan):** Pada ilustrasi di atas, terdapat tiga state: *Rainy*, *Cloudy*, dan *Sunny*. Setiap state merepresentasikan kondisi sistem pada satu waktu tertentu.
*   **Transition Probability:** Angka-angka di antara state tersebut menunjukkan kemungkinan berpindah dari satu keadaan ke keadaan lain. Misalnya, jika hari ini Rainy, ada peluang 0.3 bahwa besok akan Cloudy, dan 0.5 tetap Rainy.

*Nah*, apa poin utamanya?

Markov chain menegaskan bahwa keadaan berikutnya hanya bergantung pada keadaan sekarang, bukan sejarah lengkap sebelumnya. Artinya, jika kita tahu hari ini Cloudy, maka tidak perlu untuk tahu kondisi cuaca dua hari yang lalu untuk memperkirakan besok dan cukup gunakan probabilitas transisi dari Cloudy ke state lain.

Konsep yang dibawakan Markov Chain ini yang kemudian dikenal sebagai Markov Assumption seperti yang diilustrasikan sebelumnya. Dari akar yang kuat ini, lahirlah berbagai generasi *sequence model* mulai dari *First Order Sequence, Second Order Sequence, Second Order Sequence* dengan *skips*, sampai ke RNN yang ada sekarang.

*Wait-wait*, apa itu *First Order* dan *Sequence Order*? Baiklah mari kita *breakdown* satu per satu.

#### First Order Sequence

Versi paling sederhana dari *sequence model* yang disebut juga dengan First Order Markov Model. Sesuai dengan konsep Markov Chain, model ini mengasumsikan kata berikutnya hanya bergantung pada satu kata sebelumnya. Artinya untuk memprediksi kata ke-n+1, model ini hanya melihat kata ke-n.

![Gambar 24](https://assets.cdn.dicoding.com/original/academy/dos-ab95f910a8913944ed808c1b37879c5a20260304162332.png)

Jika kita ingin menebak kata setelah "itu", model ini hanya akan melihat kata "itu". Ia akan menghitung probabilitas kata apa yang paling mungkin muncul setelah kata "itu" berdasarkan data latih (misalnya, "mengasyikkan", "menantang").

Pendekatan ini tentu saja memiliki kelebihan karena kesederhanaannya. Namun, konteks yang bisa didapatkan oleh model sangat terbatas. *Yup*, satu-satunya sumber konteks untuk memprediksi kata berikutnya hanya berasal dari satu kata sebelumnya.

Oleh karena itu, diusulkanlah pendekatan kedua berikut ini.

#### Second Order Sequence

Sesuai namanya, pendekatan ini memperluas konteks yang digunakan oleh *First Order Sequence* sebelumnya. Kali ini, selain memperhatikan satu kata sebelumnya, ia juga memperhatikan satu lagi kata sebelumnya, sehingga kini ia melihat dua kata sebelumnya.

![Gambar 25](https://assets.cdn.dicoding.com/original/academy/dos-5cbf58e4c6890ae054dc47cdc26ada6420260304162342.png)

Misalnya untuk menebak kata setelah "Belajar Transformer…", model ini akan melihat kedua kata tersebut. Ia akan menghitung probabilitas kata apa yang paling mungkin muncul setelah frasa tersebut.

Meskipun sudah diperpanjang, tetapi ternyata konteks yang model ini gunakan masih terlalu sempit. Sekarang kita memang bisa melihat konteks dua kata sebelumnya, tetapi bagaimana jika makna sebuah kata bergantung pada kata yang berada di awal kalimat? Tentu saja *Sequence Order* tetap tidak bisa mengetahuinya.

*Yup*, mari kita *jump-in* ke pendekatan selanjutnya.

#### Second Order with Skips

Sekarang agar konteksnya tidak statis, kita akan mengevolusi Second Order sebelumnya dengan sebuah mekanisme bernama *skips*. *Eits*, sebelum itu pasti Anda bertanya mengapa tidak ada Third Order, Fourth Order, dan seterusnya.

Pada dasarnya model seperti Third Order hingga seterusnya itu termasuk ke dalam keluarga besar yang disebut model N-gram atau Markov Chain Order Tinggi. Nah, kenapa kita tidak membahas keseluruhan anggota keluarga tersebut adalah karena meningkatkan order N-gram tidak memberikan solusi yang *scalable* atau fundamental terhadap keterbatasan model sekuensial awal.

Oleh karena itu, kita akan membahas sebuah mekanisme baru sebelum masuk ke RNN.

Mekanisme *skips* memungkinkan model tidak hanya melihat elemen berurutan langsung, tetapi juga melompat ke posisi yang lebih jauh. Misal kita punya kalimat seperti di bawah ini.

![Gambar 26](https://assets.cdn.dicoding.com/original/academy/dos-f4f15b84e81cbf533d06b8673f81940620260304162354.png)

*Nah*, alih-alih melihat seluruh kata sebelumnya secara berurutan, pendekatan ini melakukan sesuatu yang sedikit berbeda.

*   **Fokus pada kata sebelumnya:** Model tetap memperhatikan kata yang berada pada posisi sebelumnya untuk memprediksi kata. Misalnya, kata “Transformer” diprediksi setelah melihat kata “mempelajari model”.
*   **Melihat Jauh ke Belakang:** Kemudian, ia membentuk pasangan antara kata terakhir ini dengan setiap kata yang muncul sebelumnya dalam kalimat, mengabaikan (melompati/skipping) kata-kata di antaranya.

![Gambar 27](https://assets.cdn.dicoding.com/original/academy/dos-927aa2b95155c2cbeb9cf5ab57a48ac520260429172813.png)

Jika kita menggunakan contoh di atas, kata “Transformer” akan mempertimbangkan pasangan-pasangan seperti (Belajar, Transformer), (Sebelum, Transformer), dan mungkin (Sequence, Transformer).

Dengan mengorbankan informasi urutan presisi, *Second Order with Skips* berhasil menangkap beberapa dependensi jangka panjang melalui fitur pasangan kata yang “melompat”. Namun, meskipun “trik loncatan” ini membantu, ia tetap memiliki keterbatasan fundamental.

Ia memberitahu kita bahwa kata “*Sequence*” pernah muncul sebelum kata “Transformer”, tetapi ia tidak tahu persis di mana atau apa kata-kata yang mengelilinginya saat itu. Informasi urutan yang detail dan nuansa dari kombinasi kata-kata yang berdekatan sebagian besar hilang dalam proses “loncatan” ini. Representasi konteksnya masih terasa terfragmentasi, seperti kumpulan potongan puzzle daripada gambaran utuh.

Kita menginginkan sesuatu yang lebih baik. Sebuah model yang tidak hanya melihat pasangan kata secara terpisah, tetapi mampu membangun pemahaman secara bertahap dan membawa informasi penting dari awal kalimat hingga akhir. Hal itulah yang dibawakan oleh RNN model yang sudah kita pelajari cara dan konsep kerjanya dalam Kelas “Belajar Fundamental Deep Learning”.

Muncul pertanyaan sekarang, mengapa jika keluarga RNN (LSTM dan GRU) adalah puncak dari *sequence model* yang sangat cocok untuk melakukan autoregressive, tetapi kenapa hampir semua LLM justru menggunakan model Transformer?

Ketimbang kita bertanya-tanya sendiri, mari kita temukan jawabannya pada petualangan berikutnya.

---

## Transformer’s Here

Setelah berkelana ke Rusia bertemu dengan nenek moyang *sequence model*, yaitu Markov Chain, sekarang sudah saatnya kita membawa teknologi di belakang layar LLM ke meja diskusi. *Yup*, seperti yang Anda baca pada judul bagian ini, Transformer akan menjadi fokus pembicaraan kita kali ini.

Arsitektur yang diperkenalkan dalam *paper* legendaris berjudul “*Attention is All You Need*” oleh Ashish Vaswani bersama para peneliti lain dari Google Brain pada 2017 ini berhasil merevolusi cara model memproses kata bahkan konsep ini digunakan oleh hampir seluruh LLM yang *exist* sekarang.

![Gambar 28](https://assets.cdn.dicoding.com/original/academy/dos-62dba63d0b40837d0d87ba8bcebf47aa20260313145128.png)

**Quick Facts** 

*“Judul makalah “Attention Is All You Need” ternyata terinspirasi dari lagu legendaris The Beatles, yaitu “All You Need Is Love”. Selain itu, nama “Transformer” sendiri dipilih bukan karena alasan teknis yang rumit, melainkan karena Jakob Uszkoreit, salah satu penulis paper-nya, sekadar menyukai bunyinya!”* 

Namun, mengapa kita begitu mengagungkan arsitektur Transformer? Apa sebenarnya yang membuatnya begitu istimewa dibandingkan model *sequence* sebelumnya?

Sekarang mari kita lihat bagaimana langkah cerdas yang digunakan oleh Transformer untuk menjawab tantangan di atas.

### The Magic of Transformer

Perlu Anda ketahui, arsitektur Transformer pada dasarnya terdiri dari dua bagian, yaitu *encoder* dan *decoder*. *Nah*, konsep *encoder-decoder* tersebut sebenarnya sudah dibawakan dari sekitar tahun 2014. Mari kita mulai perjalanan mengarungi waktu secara singkat pada sejarah *encoder-decoder* ini.

#### Awal Mula Penggunaan Encoder-Decoder

Sekitar tahun 2014, para punggawa AI masih mempercayakan model recurrent seperti RNN atau LSTM sebagai *backbone* untuk merealisasikan konsep *encoder-decoder*. Namun, muncul masalah besar yang disebut "bottleneck problem".

Mari kita identifikasi melalui penjabaran sederhana cara kerja model recurrent tersebut.

![Gambar 29](https://assets.cdn.dicoding.com/original/academy/dos-45ec4365792f4bb5d1a6c40cd6d95e1920260313145144.png)

1.  Encoder (LSTM) akan "membaca" kalimat input, kata demi kata, dari awal sampai akhir.
2.  Setelah membaca kata terakhir, ia dipaksa untuk memadatkan seluruh makna dari kalimat itu (tidak peduli seberapa panjang) ke dalam satu vektor angka berukuran tetap yang sering disebut *Context Vector*.
3.  *Context Vector* ini yang kemudian diberikan kepada Decoder (LSTM).
4.  Decoder lalu harus "menulis" kalimat output hanya berbekal satu vektor rangkuman itu.

Bayangkan Anda harus merangkum seluruh novel Harry Potter ke dalam satu kalimat SMS, lalu meminta teman Anda menulis ulang novel itu hanya berdasarkan SMS dari Anda.

Mustahil, bukan? Informasi pasti akan hilang.

Inilah yang disebut "*bottleneck problem*". Vektor berukuran tetap itu menjadi "leher botol" yang menghalangi aliran informasi. Jika kalimat inputnya sangat panjang, informasi dari kata-kata di awal kalimat hampir pasti "terlupakan" atau tertimpa oleh kata-kata di akhir.

#### Terobosan Pertama: Menambahkan "Attention" ke Model Recurrent

Barulah pada tahun 2015, para peneliti mencetuskan sebuah ide yang cukup cemerlang: Bagaimana jika Decoder tidak hanya melihat “rangkuman SMS” itu saja? Bagaimana jika kita izinkan Decoder, setiap kali ia mau menulis satu kata baru, untuk “melirik kembali” dan “melihat” ke seluruh kalimat input yang asli?

Mekanisme *attention* memungkinkan model untuk dapat melakukan hal tersebut. Alih-alih satu vektor tetap, Decoder kini memiliki "akses" ke *setiap* kata di input. Ia bisa "memperhatikan" kata-kata input yang relevan saat itu.

*“Model translation yang digunakan Google Translate pada tahun 2016 adalah tumpukan dari 8 layer encoder dan decoder LSTM (total 16 layer) yang dikombinasikan dengan mekanisme attention.”* —

*Eits*, tenang saja kita akan membahas *attention* ini dalam bagian terpisah. Sebagai *clue* komponen ini yang nantinya akan menjadi “*main course*” sumber kekuatan magis Transformer.

Penambahan *attention* pada *encoder-decoder* yang menggunakan model recurrent ini terdengar solusi yang cukup meyakinkan, tetapi ada satu masalah terakhir.

#### Satu Masalah Terakhir: "Antrean Kasir"

Setelah melalui improvement di atas, pada akhirnya model-model seperti RNN/LSTM adalah model sekuensial.

Mereka harus memproses kata ke-1, sebelum bisa memproses kata ke-2, sebelum bisa memproses kata ke-3, dan seterusnya. Ini seperti antrean panjang di kasir. Anda tidak bisa melayani pelanggan ke-10 sebelum pelanggan ke-9 selesai.

Di dunia modern yang penuh dengan Graphics Processing Unit (GPU) super kuat, ini sangat tidak efisien. GPU dirancang untuk melakukan ribuan perhitungan secara paralel. GPU itu ibarat memiliki 1.000 kasir yang siap bekerja sekaligus, tapi model sekuensial hanya bisa menggunakan satu kasir dalam satu waktu. Ini tentu sangat lambat untuk dilatih.

#### The Magic’s Come

Berdasarkan perjalanan di atas, kita sadar bahwa mekanisme *attention* itu sangat powerful untuk membuat model memahami konteks dengan baik. Selain itu kita juga setuju bahwa model recurrent (RNN) itu lambat karena proses sekuensial.

*Nah*, bagaimana kita membuang RNN dan mencoba untuk membangun arsitektur baru dengan attention sebagai fondasinya. Dari situ lah terciptanya *paper* yang kita sebut di awal “*Attention Is All You Need*” dan menjadikan arsitektur bernama Transformer seperti yang terlihat pada ilustrasi di bawah ini.

![Gambar 30](https://assets.cdn.dicoding.com/original/academy/dos-cca3c3da83c28da847d814141b6b31c920260304162635.png)

Transformer menyulap cara kerja model *recurrent* yang mengandalkan urutan menjadi pemrosesan secara paralel. *Yup*, alih-alih memproses setiap token satu per satu seperti orang mengantre, Transformer memilih untuk memproses seluruh token yang masuk secara bersamaan atau paralel.

Sekilas ini tampak luar biasa, tetapi paralelisasi itu memunculkan pertanyaan yang menarik: bagaimana cara Transformer paham kata “saya” itu berada sebelum kata “nasi” dalam kalimat “saya makan nasi”?

Potensi model “gagal paham” pada suatu kalimat karena tertukar posisi antar kata sangat masuk akal terjadi pada model yang membanggakan pemrosesan secara paralelnya ini.

*Nah*, Tantangan tersebut dapat dijawab oleh Transformer melalui sumber kekuatan magisnya yang bernama *Multi-Head Attention* dan *Positional Encoding*. *Eits*, sebelum masuk ke “*main course”* tersebut, kita perlu “mengetuk pintu” pertama terlebih dahulu untuk memahami gambaran arsitektur Transformer dan cara kerjanya secara general. Baiklah, mari kita mulai dengan berkenalan dengan si kembar tetapi tidak identik, encoder dan decoder dalam Transformer.

### “Si Pemikir” dan “Si Penulis”

Sesuai dengan judul di atas, Encoder dan Decoder bisa kita ibaratkan sebagai dua *role* yang berbeda. Dari sini kita mendapatkan clue bahwa arsitektur Transformer pada dasarnya adalah sebuah sistem yang terdiri dari dua komponen yang bekerja sama.

![Gambar 31](https://assets.cdn.dicoding.com/original/academy/dos-5bac31b28d3a3036988fcbe7802733ae20260429165141.png)

Secara visual dari ilustrasi di atas, kita sepertinya hampir tidak menemukan perbedaan antara encoder dan decoder. Namun, keduanya memiliki beberapa komponen pembeda jika Anda perhatikan dan itulah yang membuat mereka punya peran yang berbeda. Mari kita ungkap satu per satu “Si Kembar” tidak identik ini.

#### Encoder

Sebagai komponen yang kita sebut sebagai “Si Pemikir”, Encoder bertugas untuk "membaca" dan "memahami" kalimat input. Bayangkan saat sedang membaca sebuah paragraf panjang, Anda tentu tidak langsung mengambil kesimpulan tentang isinya begitu saja, bukan? Anda akan mencoba memahami setiap kalimat terlebih dahulu, mencari tahu apa yang dibicarakan dan mengaitkan seluruh informasi dari awal hingga akhir.

Kira-kira seperti itulah peran Encoder dalam arsitektur Transformer.

Encoder bekerja dengan cara menyerap konteks dari seluruh input. Ia memetakan setiap kata menjadi representasi vektor, semacam bentuk numerik dari makna kata tersebut. Dari proses ini, lahirlah apa yang kita kenal sebagai Context Vector, sebuah rangkuman yang berisi “pemikiran” atau pemahaman Encoder terhadap keseluruhan kalimat.

#### Decoder

Sekarang, bayangkan Anda sudah memahami sebuah paragraf, lalu seseorang bertanya, “Bisa ceritakan ulang dengan kata-katamu sendiri?” *Nah*, di sinilah Decoder berperan.

Decoder adalah komponen yang bertugas mengubah pemahaman menjadi ucapan atau dalam konteks ini, mengubah representasi vektor dari Encoder menjadi rangkaian kata yang bisa dipahami manusia. Ia tidak menghasilkan semua kata sekaligus, melainkan satu per satu. Terdengar familier? *Yup*, Decoder inilah yang melakukan *job* bernama autoregressive.

Setiap kali Decoder menghasilkan sebuah token, token itu langsung digunakan sebagai konteks untuk memprediksi token berikutnya. Begitu seterusnya, hingga terbentuk kalimat yang utuh dan bermakna.

Baiklah sekarang Anda sudah mendapatkan gambaran besar dari *Blueprint* arsitektur Transformer. Anda juga mungkin sudah aware cukup banyak komponen yang perlu kita pelajari di dalam Transformer.

Namun, jika Anda perhatikan, ada satu komponen yang terus disebut-sebut sebagai sumber kekuatan utama Transformer dan bahkan komponen inilah yang memanifestasikan kemunculan arsitektur Transformer itu sendiri. Nama komponen tersebut bahkan dijadikan judul sekaligus slogan dalam *paper* asli Transformer di awal, tidak lain dan tidak bukan adalah **Attention Mechanism**. *So*, *it’s time to deep dive into the attention mechanism!* 

### The Main Engine: Attention Mechanism

Sebelumnya, ketika kita menelusuri evolusi arsitektur dari RNN hingga Transformer, kita sempat melihat bagaimana *attention mechanism* muncul sebagai solusi atas kelemahan mendasar model *recurrent*, yaitu masalah “kepikunan” terhadap konteks yang jauh. Attention memungkinkan model untuk melihat kembali seluruh input secara langsung, bukan hanya mengandalkan ringkasan sempit seperti context vector pada model sebelumnya.

Namun, tentu saja membaca ulang seluruh input bukan perkara ringan. Bayangkan seperti membaca satu paragraf berkali-kali hanya untuk memahami satu kata di akhirnya. Melelahkan, bukan? Oleh karena itu, *attention* berfokus pada efisiensi dalam “membaca ulang”.

*Nah*, sedari tadi kita belum mengetahui cara mengefisiensi langkah “membaca ulang” seluruh input kalimat tersebut. Sebelum itu, sebenarnya ada di mana letak *attention mechanism* dalam arsitektur Transformer?

Perhatikan ilustrasi berikut.

![Gambar 32](https://assets.cdn.dicoding.com/original/academy/dos-104254a36260b59ae8df3b35b7ca9d0020260429165843.png)

Komponen *Multi-Head Attention* pada dasarnya merupakan pengembangan dari *attention mechanism*. Kita akan membahas *Multi-Head Attention* ini lebih lanjut pada bagian terpisah. Sekarang, mari kita fokus kembali pada *attention* itu sendiri, seperti halnya attention yang berfokus pada kata-kata.

*Yup*, nama *attention* sendiri sebetulnya sudah memberikan clue besar pada kita tentang cara kerjanya dalam gambaran besar. Secara sederhana, *attention mechanism* akan membantu model untuk memberikan fokus yang lebih pada beberapa kata kunci atau yang dianggap relevan dengan kata yang sedang diproses.

Otak manusia dapat fokus pada suatu hal yang kita tentukan karena otak memberikan bobot fokus yang lebih besar ke sesuatu tersebut. Perhatikan ilustrasi di bawah.

![Gambar 33](https://assets.cdn.dicoding.com/original/academy/dos-86e5bb6b004efcf926d0ffe42c2ba3ac20260429165449.png)

Saat membaca kata “menghasilkan” pada kalimat panjang seperti pada ilustrasi di atas, otak Anda tentu perlu mengetahui, **SIAPA** atau **APA** yang “menghasilkan” performa itu?

*   Apakah “RNN” yang menghasilkan? TIDAK.
*   Apakah “arsitektur” yang menghasilkan? TIDAK juga.
*   Apakah “memproses” yang menghasilkan? Justru lebih aneh lagi.

Akhirnya, otak Anda akan terus menelusuri kata-kata sebelumnya untuk menemukan jawaban yang paling relevan, bahkan mungkin sampai pada kata yang berada di awal kalimat. Hingga pada akhirnya, jawabannya jatuh pada kata “Transformer.”

*Nah*, *attention mechanism* bekerja dengan cara yang sama.

Dengan cara itu, model dapat belajar kata kerja "menghasilkan" di sini sangat relevan dengan subjek "Transformer". Masalahnya pada ilustrasi di atas, kita mengandaikan besar fokus yang diberikan otak saat memproses suatu kata dalam bobot persentase. Sekarang jika *attention mechanism* meniru konsep itu, bagaimana proses perhitungannya dan siapa yang melakukannya?

Untuk menjawabnya, mari kita ucapkan selamat datang di kelas matematika singkat: QKV pada Attention Mechanism. *Eits*, tenang saja ini bukanlah kelas matematika rumit yang memaksa kita menghafal rumus kompleks nan abstrak selayaknya *Picasso*.

#### “Sisi Matematis” Attention Mechanism

Meskipun kita menyebutnya “Sisi Matematis”, pada dasarnya cara kerja teknis dari Attention Mechanism seperti Anda mencari file di komputer. Proses pencarian ini memanfaatkan tiga komponen yang sudah kita mention sebelumnya, yaitu Query (Q), Key (K), dan Value (V).

*   **Query (Q)** adalah "pertanyaan" yang Anda atau *user* miliki. Dalam konteks *Multi-Head Attention*, setiap kata yang sedang diproses berfungsi sebagai Query. Ia ingin mencari tahu kata-kata mana dalam kalimat yang sama yang paling relevan dengannya.
*   **Key (K)** dapat dikatakan sebagai "indeks" atau "label" dari setiap informasi yang tersedia. Setiap kata lain dalam kalimat yang sama termasuk kata Query itu sendiri juga menghasilkan Key.
*   **Value (V)** inilah yang berperan sebagai "konten" atau "informasi aktual" yang terkait dengan setiap Key. Setiap kata juga menghasilkan *Value*-nya sendiri, yang pada dasarnya adalah representasi dari makna atau informasi yang ingin diberikan jika terpilih sebagai kata yang relevan.

Baiklah setelah berkenalan dengan ketiga komponen tersebut, sekarang mari kita belajar berhitung! Agar lebih membantu Anda untuk memahami proses perhitungan di dalam *attention* ini, mari kita bedah lebih dalam lagi ilustrasi *Multi-Head Attention* sebelumnya.

![Gambar 34](https://assets.cdn.dicoding.com/original/academy/dos-ecec9c41c2454dd85f1d72e68a69eda620260429165555.png)

*Nah*, proses perhitungan bobot pada attention mechanism terjadi di dalam sebuah komponen yang disebut *Scaled Dot-Product Attention*. Proses ini terdiri dari **EMPAT** tahap utama, seperti yang terlihat pada ilustrasi di atas. Mulai dari operasi *matmul* di awal hingga melakukan operasi *matmul* kembali di akhir.

Namun, mungkin Anda memperhatikan sesuatu di gambar tersebut karena ada lima kotak, bukan empat. Mengapa seperti itu?

*Yup*, Anda benar. Tahapan kelima itu adalah Masking yang sempat kita bahas sebelumnya.

Hanya saja, masking bersifat opsional dan biasanya hanya digunakan di decoder,

tepatnya pada proses autoregressive agar model tidak “mengintip masa depan” saat menghasilkan token berikutnya.

Jadi, untuk sekarang kita akan fokus dulu pada empat tahap utama tersebut.

##### Tahap 1: Menghitung Alignment Score

Langkah pertama adalah mengukur seberapa cocok atau relevan Query saat ini (*alignment score*) dengan setiap Key dari seluruh input. Bagaimana cara termudah mengukur kesamaan antara dua vektor? Caranya adalah menggunakan perhitungan *dot-product*. Mari perhatikan ilustrasi di bawah ini.

![Gambar 35](https://assets.cdn.dicoding.com/original/academy/dos-a7c01a4404f01880df42476398cdab1b20260429165948.png)

Kita melakukan ini untuk Query tunggal kita terhadap setiap Key (K) yang ada di input. Hasilnya adalah sebuah skor mentah untuk setiap token input. Jika vektor Query dan Key memiliki arah yang mirip (selaras), hasil dot product akan tinggi, yang menandakan relevansi tinggi. Sebaliknya, jika arahnya berbeda, skornya akan rendah.

##### Tahap 2: Penskalaan (Scaling)

Setelah mendapatkan skor dari *dot-product*, kita akan membaginya dengan akar kuadrat dari dimensi vektor Key. Sedikit rumit yah? Ilustrasi di bawah akan membantu Anda.

![Gambar 36](https://assets.cdn.dicoding.com/original/academy/dos-41f18e04c0b2c40b0a813cb7046c8f5b20260429170314.png)

Mengapa proses *scaling* ini diperlukan?

Saat dimensi vektor Key menjadi besar, nilai hasil dot-product juga bisa menjadi sangat besar. Jika kita langsung memasukkan nilai yang sangat besar ini ke fungsi softmax (langkah berikutnya), gradiennya bisa menjadi sangat kecil atau mendekati nol. Ini akan "membunuh" proses belajar model. Dengan kata lain, penskalaan ini menjaga agar varians dari skor tetap stabil, membuat proses *training* lebih lancar.

##### Langkah 3: Normalisasi dengan Softmax

Skor yang sudah diskalakan tadi masih berupa angka mentah. Bisa positif, negatif, besar, atau kecil. Kita perlu mengubahnya menjadi sesuatu yang bisa diinterpretasikan sebagai bobot, misalnya angka antara 0 dan 1 yang berarti total semua bobot adalah 1. Fungsi yang sempurna untuk melakukan ini adalah *Softmax* yang sering kita jumpai pada model klasifikasi multi-class.

![Gambar 37](https://assets.cdn.dicoding.com/original/academy/dos-d10fb9ca06d329da5c97cb5ad1ffcf7120260429170349.png)

Seperti yang kita ketahui, fungsi softmax akan mengambil semua skor, lalu menonjolkan skor yang tinggi dan menekan skor yang rendah, serta memastikan hasilnya adalah sebuah distribusi probabilitas. Hasil dari softmax inilah yang kita sebut sebagai *attention weights*.

##### Langkah 4: Memberikan Bobot Pada Value

Kita sudah punya bobotnya. Sekarang apa? Kita gunakan bobot ini untuk "mengambil" informasi dari Value. Caranya adalah dengan melakukan penjumlahan terbobot (*weighted sum*) dari semua vektor *Value*.

![Gambar 38](https://assets.cdn.dicoding.com/original/academy/dos-bcdc4280babf05de645546bc2a103c2b20260429170420.png)

Vektor Value yang memiliki bobot *attention* tinggi akan memberikan kontribusi lebih besar pada hasil akhir. Sebaliknya, Value dengan bobot rendah kontribusinya akan minimal. Hasil akhir dari proses ini adalah satu vektor tunggal yang disebut Context Vector. Vektor inilah yang merangkum semua informasi relevan dari seluruh input, yang disesuaikan dengan "pertanyaan" dari Query.

*Nah*, jika seluruh keempat tahapan di atas kita satukan menjadi satu rumus, maka akan tercipta satu rumus matematis yang terpajang di *paper* “*Attention Is All You Need*”.

![Gambar 39](https://assets.cdn.dicoding.com/original/academy/dos-4907c29da9b1414f415c57bedbe5ebcc20260429170444.png)

Jika tahapan di atas dibuat menjadi analogi kira-kira akan seperti berikut.

Bayangkan Anda berada di sebuah pesta yang bising. Anda (Query) ingin memahami apa yang dikatakan teman Anda. Ada banyak suara lain di sekitar (Keys dan Values). Otak Anda secara intuitif melakukan *attention* *mechanism*.

*   Anda mencocokkan suara teman Anda (Key) dengan fokus Anda (Query). Suara teman Anda skornya tinggi.
*   Suara musik atau orang lain di sekitar Anda skornya lebih rendah.
*   Anda menerapkan "softmax" di otak, memberikan bobot hampir 100% pada suara teman Anda dan hampir 0% pada yang lain.
*   Anda akhirnya "mendengar" pesan (Value) dari teman Anda dengan jelas.

Itulah cara *attention mechanism* menghitung bobotnya. Bukan sihir atau pemberian bobot secara asal, melainkan serangkaian operasi matriks yang meniru kemampuan manusia dalam memfokuskan perhatian.

Baiklah sebelumnya Anda melihat bahwa *attention mechanism* yang digunakan pada transformer bukan *attention* biasa, melainkan versi pengembangannya, yaitu *Multi-Head Attention*. Sejauh apa perbedaannya dengan *attention* biasa? Mari kita temukan jawabannya.

#### Multi-Head Attention

Untuk dapat memahami perbedaan dan alasan penggunaan *Multi-Head Attention* ketimbang *attention* biasa, mari kita pahami tantangan yang perlu dihadapi Transformer.

Bayangkan Anda membaca kalimat kompleks. Saat Anda fokus pada satu kata, Anda mungkin memperhatikannya dalam beberapa cara sekaligus seperti berikut.

*   Hubungan sintaktisnya (kata ini adalah subjek dari kata kerja apa?)
*   Hubungan semantiknya (kata ini maknanya mirip dengan kata apa?)
*   Hubungan posisinya (kata ini berada di awal kalimat)

Jika kita hanya menggunakan satu mekanisme *attention*, ia mungkin akan "terpaksa" memilih untuk fokus hanya pada salah satu aspek tersebut, atau menghasilkan rata-rata yang kurang optimal dari ketiganya. Ini seperti meminta satu orang ahli untuk menganalisis sebuah karya seni dari segi komposisi, nilai sejarah, dan teknik kuas sekaligus. Dia bisa melakukannya, tapi hasilnya tidak akan sedalam jika kita bertanya pada tiga ahli yang berbeda.

*Nah*, Di sinilah *Multi-Head Attention* hadir sebagai solusi. Idenya sederhana saja, daripada melakukan satu perhitungan *attention*, mari kita lakukan beberapa perhitungan *attention* secara paralel, dan kemudian gabungkan hasilnya.

![Gambar 40](https://assets.cdn.dicoding.com/original/academy/dos-010db11863e16701db306ee3444ace5b20260429170512.png)

Setiap "*head*" atau "kepala" *attention* ini adalah versi independen dari *Scaled Dot-Product Attention* yang sudah kita pelajari. Bagaimana Cara Kerjanya?

Jika kita merujuk kembali ke gambar arsitektur yang Anda tunjukkan di awal, prosesnya bisa dirangkum dalam 4 langkah:

1.  **Proyeksi Linear** 
    Pintu pertama dari *Multi-Head Attention* ini bertugas memproyeksikan representasi vektor menjadi tiga representasi berbeda, yaitu Q (query), K (key), dan V (value) yang kita bahas sebelumnya. Setiap *linear layer* memiliki bobot yang berbeda untuk ketiganya, karena masing-masing representasi memiliki fungsi yang berbeda dalam menghitung *attention weights*. Jika kita memiliki 8 *head*, kita akan memiliki 8 set layer linear yang berbeda untuk Q, 8 untuk K, dan 8 untuk V. Setiap *head* mendapatkan versi Q, K, dan V-nya sendiri yang unik.
2.  **Perhitungan Bobot Attention** 
    Sekarang kita memiliki (misalnya) 8 set (Q, K, V). Kita melakukan perhitungan Scaled *Dot-Product Attention* untuk setiap *head* secara bersamaan atau paralel. Oleh karena setiap *head* memiliki bobot proyeksi yang berbeda dari langkah 1 sebelumnya, setiap *head* akan belajar untuk fokus pada aspek atau hubungan yang berbeda dari data.
3.  **Concatenate** 
    Setelah 8 *head* selesai bekerja, kita mendapatkan 8 vektor output atau context vector yang berbeda. Kita cukup menggabungkan semua vektor hasil ini menjadi satu vektor besar melalui proses *concatenate*.
4.  **Final Linear Layer** 
    Vektor besar hasil gabungan tadi kemudian dilewatkan ke satu layer linear terakhir. Tujuannya adalah untuk "mencampur" semua wawasan dari head-head yang berbeda dan mengembalikannya ke dimensi yang diharapkan oleh layer selanjutnya di dalam model.

Jika Anda perhatikan baik-baik, mungkin ada hal yang tampak sedikit janggal. Secara teori, proyeksi linear seharusnya berfungsi untuk memecah vektor masukan menjadi tiga komponen utama, yaitu Q, K, dan V. Namun, pada ilustrasi *Multi-Head Attention* sebelumnya, ketiga komponen ini justru sudah muncul sebelum tahap *Multi-Head Attention* dimulai.

*Nah*, sebenarnya hal itu terjadi karena ilustrasi yang digunakan mewakili bentuk paling umum dan fleksibel dari mekanisme *Multi-Head Attention*. Dengan pendekatan seperti itu, diagram tersebut dapat menggambarkan berbagai variasi arsitektur *attention* yang digunakan dalam model Transformer.

Artinya, dalam arsitektur Transformer tidak hanya ada satu jenis *Multi-Head Attention*. *So*, pada bagian selanjutnya kita akan berkenalan dengan seluruh variasi tersebut.

#### Mekanisme Attention dalam Transformer

Perlu Anda ketahui, dalam arsitektur Transformer versi *full*, terdapat tiga jenis *Multi-Head Attention* yang tersebar di bagian *Encoder* dan *Decoder*. Anda dapat memperhatikan ilustrasi di bawah ini untuk mengidentifikasi ketiga variasi *Multi-Head Attention* berikut.

![Gambar 41](https://assets.cdn.dicoding.com/original/academy/dos-8f5787506aa8eb466dc0f2ea077813c620260429170556.png)

Setiap jenis memiliki fungsi dan konteks penerapan yang berbeda, tergantung pada informasi apa yang akan diolah model pada tahap tersebut.

*   **Encoder Self-Attention** 
    Jenis pertama adalah Self-Attention yang berada di Encoder. Attention ini berfokus untuk memahami konteks kalimat input itu sendiri. Ia melakukannya dengan membuat setiap token dalam suatu input untuk “memperhatikan” token lain dalam *sequence* input yang sama. Pada tahap ini, Q, K, dan V semuanya berasal dari sumber yang sama, yaitu output dari layer Encoder sebelumnya.
*   **Decoder Masked Self-Attention** 
    Sekarang kita beralih ke Decoder stack, Attention ini pada dasarnya mempunyai tugas yang mirip dengan Encoder Self Attention sebelumnya. Hanya saja, ia memiliki fokus yang sedikit berbeda, yaitu untuk memahami konteks kalimat target yang sedang dibuat atau sudah dihasilkan sejauh ini, bukan lagi input awal. Selain itu, adanya penambahan mekanisme “Masked” yang akan kita bahas di bagian selanjutnya.
*   **Encoder-Decoder Attention (Cross-Attention)** 
    Satu lagi Attention yang ada di decoder dan berperan sebagai jembatan antara encoder dan Decoder. Pada Cross-Attention, Query (Q) berasal dari hasil pemrosesan di Decoder, sedangkan Key (K) dan Value (V) berasal dari output Encoder. Tujuannya adalah agar Decoder dapat “melihat kembali” konteks yang telah dipahami oleh encoder. Dengan mekanisme ini, model bisa menghubungkan informasi dari input awal dengan informasi output yang sedang dihasilkan. Mari perhatikan contoh berikut.
    *   Misalnya, model sedang menerjemahkan kalimat "*The cat sat on the sandbox*" dan sudah sampai pada kata "*kucing itu duduk di…*".
    *   Query dari Decoder saat ini pada dasarnya bertanya, "*Duduk di mana?*".
    *   Cross-Attention akan membandingkan Query ini dengan Keys dari kalimat sumber, menemukan skor tertinggi pada kata "*the sandbox*", lalu mengambil Value dari kata tersebut untuk membantu menghasilkan kata "*kotak pasir*".

Setelah berkenalan dengan seluruh variasi Attention Mechanism, kita masih terlewat penjelasan soal masking yang terdapat di bagian Decoder Self-Attention sebelumnya.

#### Mekanisme Masking dalam Attention.

Perlu diketahui, jenis mekanisme Masking yang digunakan dalam Transformer secara khusus adalah Causal Mask. Causal Mask yang ditambahkan di dalam *Decoder Self-Attention* memberikan sebuah *ability* penting yang memungkinkan model paralel seperti Transformer tetap dapat melakukan autoregressive dengan benar.

Memungkinkan teknik autoregressive secara benar? Apa maksudnya?

Baiklah untuk menjawab pertanyaan tersebut, mari kita pahami terlebih dahulu konsep masking itu sendiri. Masking adalah teknik untuk menyembunyikan atau mengabaikan bagian-bagian tertentu dari input agar tidak memengaruhi perhitungan di bagian lain.

*Nah*, Causal Mask secara khusus akan menyembunyikan token di masa depan yang bekerja dengan cara memodifikasi bobot attention sebelum fungsi softmax diterapkan.

Bagaimana cara kerjanya? Coba perhatikan ilustrasi ini.

![Gambar 42](https://assets.cdn.dicoding.com/original/academy/dos-f48a27d8d19db5bc075dd2aeb3d732a520260313150221.png)

Masking akan memproses matriks bobot attention, sehingga posisi masa depan akan diisi dengan nilai −∞. Harapannya saat masuk ke bagian softmax, bobotnya dapat menjadi nol. Sehingga cara kerja attention pada setiap token akan menjadi seperti ini.

*   Token “Setiap” hanya boleh memperhatikan dirinya sendiri (nilai 1.0).
*   Token “kata” boleh memperhatikan “Setiap” dan dirinya sendiri.
*   Token “akan” boleh memperhatikan “Setiap”, “kata”, dan “akan”.
*   Dan seterusnya.

Konsep ini dapat menjadi garansi proses Autoregressive terjadi dengan benar pada Transformer karena memitigasi dari beberapa hal berikut.

*   **Information Leakage** 
    Model diam-diam menggunakan token masa depan untuk memperkirakan token saat ini. Ini membuat pelatihan tidak realistis karena seolah-olah model sedang menebak dengan kunci jawaban terbuka.
*   **Perbedaan Distribusi Pelatihan dengan Distribusi Inferensi** 
    Saat *training*, model belajar dengan “bantuan masa depan”. Namun, ketika *inference* atau *generating* teks, model hanya punya masa lalu. Akibatnya terjadi *train-test mismatch*, dan performa turun drastis ketika benar-benar digunakan untuk menghasilkan teks.
*   **Autoregressive-nya jadi “tidak benar” secara kausalitas**
    Autoregressive meniru proses waktu, dari masa lalu ke masa depan. *Nah*, tanpa *masking*, urutan waktu dapat rusak sehingga model tidak lagi mengikuti arah kausal (*causal direction*).

Baiklah, setelah memahami berbagai jenis *Multi-Head Attention* dan proses Causal Masking, kini kita mulai bisa melihat gambaran besar dari cara kerja Transformer secara keseluruhan. Namun, *attention* hanyalah salah satu bagian dari rangkaian komponen yang saling terhubung di dalamnya.

*Nah*, sebelum kita lanjut untuk merakit keseluruhan konstruksi blok transformer, akan menarik jika Anda menguji ingatan tentang empat tahap yang terjadi di dalam *Scaled Dot-Product Attention*.

> **Aktivitas: Drag and Order**
> Urutan yang tepat:
> 1. Menghitung alignment Score pada Matmul
> 2. Melakukan penskalaan atau Scaling
> 3. Normalisasi menggunakan Softmax
> 4. Menambahkan Weight pada Value

Sekarang saatnya untuk melanjutkan langkah kita untuk merakit satu arsitektur penuh Transformer.

### Merakit Blok Transformer

Beberapa komponen yang salah satunya adalah *Multi-Head Attention* yang kita bahas secara detail sebelumnya, merupakan potongan-potongan *puzzle* yang ketika disusun akan membentuk satu kesatuan arsitektur yang kita panggil Transformer.

#### Mulai dari Pintu Masuk Transformer

Sebelum Transformer mulai bekerja, langkah pertama yang selalu dilakukan tentu saja adalah mengubah teks menjadi representasi vektor yang bisa dipahami mesin atau seperti yang kita ketahui adalah Embedding.

Seperti yang kita bahas sebelumnya, proses paralel dalam Transformer membawa tantangan dalam mengetahui urutan setiap kata. Model sekuensial seperti RNN memproses kalimat kata per kata, jadi mereka otomatis tahu urutannya. Proses paralel pada Transformer ini memang lebih cepat, tetapi itu membuatnya "buta" terhadap urutan. Bagi Transformer, kalimat "kucing makan ikan" dan "ikan makan kucing" bisa terlihat sama persis.

Untuk mengatasi ini, Transformer butuh dua jenis informasi yang kemudian digabung jadi satu.

*   Informasi Makna (Apa katanya?)
*   Informasi Posisi (Di mana letaknya?)

Oleh karena itu, Transformer memiliki dua tahapan di “Pintu Masuk” ini.

*   **Input Embedding** 
    Pada dasarnya ini adalah embedding yang sebelumnya kita pelajari dan sering ditemui. Setiap kata unik dalam data kita akan dipetakan ke sebuah representasi vektor berdimensi tinggi. Pada saat proses *training*, model akan secara otomatis menyesuaikan nilai vektor-vektor ini. Sehingga kata-kata yang maknanya mirip (seperti "belajar" dan "mengajar") akan memiliki vektor yang posisinya saling berdekatan di dalam *vector space*. Sebaliknya, kata-kata yang maknanya jauh (seperti "belajar" dan "pelabuhan") akan memiliki vektor yang berjauhan.
    ![Gambar 43](https://assets.cdn.dicoding.com/original/academy/dos-9a46998ae4feef8f06408749cbb34a6420260305091203.png)
*   **Positional Encoding** 
    Setelah kita punya vektor yang berisikan makna dari masing-masing kata, selanjutnya kita perlu "menyuntikkan" informasi urutan agar model tidak kehilangan konteks. Caranya adalah dengan membuat vektor kedua yang khusus merepresentasikan posisi sebuah kata (misalnya, vektor unik untuk posisi ke-1, posisi ke-2, dan seterusnya). Vektor posisi ini memiliki dimensi yang sama persis dengan vektor token embedding. Untuk membuat vektor posisi tadi, ada dua pendekatan populer yaitu Sinusoidal dan Learnable.
    ![Gambar 44](https://assets.cdn.dicoding.com/original/academy/dos-056429a7b440d9877fefe2981ddbfdcb20260305091223.png)
*   **Sinusoidal Positional Encoding** 
    Ini adalah pendekatan standar yang digunakan di *paper* Transformer asli. Ia menggunakan rumus matematika tetap (kombinasi fungsi sinus dan cosinus) untuk menghasilkan vektor unik bagi setiap posisi. Kelebihannya, ini cepat, tidak perlu dilatih, dan bisa menangani kalimat yang sangat panjang.
*   **Learnable Positional Encoding** 
    Pendekatan ini tidak pakai rumus. Sebaliknya, ia memperlakukan vektor posisi sebagai parameter yang bisa dilatih, mirip seperti input embedding sebelumnya. Model akan "belajar" sendiri nilai-nilai vektor terbaik untuk merepresentasikan setiap posisi berdasarkan data *training*. Ini lebih fleksibel, tapi menambah komputasi dan biasanya butuh penetapan panjang kalimat maksimum.

Dengan menggabungkan token embedding dan positional encoding, Transformer akhirnya memiliki representasi vektor yang tidak hanya memahami arti kata, tetapi juga konteks urutannya. Barulah kita bisa masuk ke “*main course*”.

#### Masuk ke Blok Utama

Blok ini bisa dianggap sebagai unit pemrosesan utama yang mengekstraksi, memperdalam, dan menyeimbangkan informasi sebelum diteruskan ke lapisan berikutnya. Satu blok Transformer terdiri dari kerja sama empat komponen inti yang salah satunya adalah *Multi-Head Attention* yang sudah kita bedah sebelumnya.

![Gambar 45](https://assets.cdn.dicoding.com/original/academy/dos-b1af18f4ebf000870fd943be8cc4a27e20260429170823.png)

*Nah*, sekarang mari kita bedah satu per satu komponen lainnya.

*   **Multi-Head Attention** 
    Mekanisme ini telah kita bahas secara mendalam sebelumnya, jadi di sini kita cukup mengingat kembali perannya secara singkat. *Multi-Head Attention* berfungsi sebagai “otak” dari Transformer. Melalui proyeksi Query, Key, dan Value, Multi-Head Attention membantu model memahami konteks kalimat secara paralel dari berbagai sudut pandang. Setelah data keluar dari *Multi-Head Attention*, ia tidak langsung dilempar ke lapisan berikutnya.
*   **Residual Connection & Normalization** 
    Setelah *Multi-Head Attention*, hasilnya tidak langsung diteruskan begitu saja. Sebaliknya, hasil tersebut diproses dalam dua langkah tambahan.
    1.  Output dari *Multi-Head Attention* akan dijumlahkan dengan input aslinya sebelum masuk *Multi-Head Attention*, inilah yang disebut *Residual Connection*. Tujuannya adalah untuk mempertahankan informasi awal agar tidak hilang selama proses transformasi.
    2.  Kedua, dilakukan Layer Normalization, yaitu proses menyeimbangkan skala nilai dalam vektor agar pembelajaran lebih stabil dan efisien.
*   **Feed-Forward Network** 
    Setiap token yang telah membawa konteks dari *attention* kemudian diproses oleh Feed-Forward Network (FFN), yang terdiri dari dua lapisan linear dengan fungsi aktivasi nonlinier seperti ReLU atau GELU di antaranya. Lapisan pertama memperluas dimensi representasi untuk menangkap pola-pola nonlinier yang lebih kompleks, sementara lapisan kedua menurunkannya kembali ke ukuran semula agar hasilnya tetap sejalan dengan mekanisme *residual connection* pada arsitektur Transformer.
*   **Residual Connection & Normalization** **(kembali)** 
    *Yup*, tahap ini mirip dengan langkah kedua. Output dari Feed-Forward Network kembali dijumlahkan dengan input aslinya melalui *residual connection*, lalu dinormalisasi. Hasil akhirnya adalah representasi vektor yang telah melalui proses pemahaman konteks dan penyempurnaan makna.

#### Berakhir di Pintu Keluar Transformer

Setelah melewati serangkaian blok Transformer yang saling bertumpuk, representasi setiap token kini telah mengandung pemahaman konteks mengenai makna kata itu sendiri hingga hubungan antar token di sekitarnya. Namun, sebelum hasil ini bisa digunakan untuk menghasilkan teks atau prediksi akhir, Transformer masih harus melalui satu tahap penting.

*   **Proyeksi Linear** 
    Tahap ini dimulai dengan linear projection, yaitu lapisan linear yang mengubah representasi internal model menjadi dimensi sebesar ukuran kosakata (vocabulary size). Proses ini ibarat mengonversi pemahaman konteks menjadi kemungkinan setiap token dalam kosakata untuk menjadi keluaran berikutnya.
*   **Softmax** 
    Selanjutnya, hasil proyeksi tersebut dilewatkan ke fungsi softmax, yang mengubah nilai-nilai logit menjadi probabilitas. Dengan cara ini, Transformer dapat menentukan token mana yang paling mungkin muncul setelah urutan sebelumnya. Token dengan probabilitas tertinggi kemudian dipilih sebagai output dan jika proses ini diulang terus menerus, terbentuklah kalimat atau teks yang lengkap.

---

## Evolusi Transformer

Baiklah, kita sudah menyelesaikan perjalanan luar biasa merakit arsitektur Transformer dari nol. Sekarang, kita sudah memegang blueprint aslinya, arsitektur lengkap dengan Encoder dan Decoder.

Namun, seperti halnya sebuah penemuan besar, *paper* "*Attention Is All You Need*" di tahun 2017 itu bukanlah akhir dari cerita Transformer. Sebaliknya, itu bisa dikatakan sebuah *Big Bang* yang memulai awal dari sebuah "alam semesta" AI baru yang berkembang dengan kecepatan luar biasa hingga sekarang.

“Si Penantang” model sekuensial ini terus berevolusi hingga sekarang. Oleh karena itu kita akan membahas perkembangan arsitektur Transformer ini melalui tiga jalan yang berbeda. *Are you ready Coders?* 

### Evolusi Arsitektur

Jika Anda membaca *paper* “*Attention Is All You Need*”, blueprint asli tersebut pada dasarnya dirancang untuk satu tugas spesifik, yaitu *translation machine* atau mesin penerjemah. Sebuah tugas *sequence-to-sequence* murni, di mana model perlu memahami satu kalimat penuh menggunakan Encoder, lalu menghasilkan kalimat baru dengan Decoder.

Namun, komunitas AI dengan cepat menyadari bahwa "mesin" *attention* ini terlalu kuat jika hanya digunakan untuk menerjemahkan. Para peneliti mulai bertanya, "Untuk tugasku, apakah aku benar-benar butuh kedua sisi Transformer?"

Pertanyaan ini melahirkan tiga "spesies" utama model Transformer yang mendominasi dunia AI saat ini.

#### Encoder-Only

Arsitektur Encoder Only hanya menggunakan bagian encoder dari Transformer asli. Fokus utamanya adalah memahami konteks secara menyeluruh dalam sebuah teks. Model seperti BERT dan turunannya dirancang untuk *bidirectional understanding*. Apa itu *bidirectional understanding*?

Artinya, karena tidak memiliki komponen Decoder yang membutuhkan masking, model-model Encoder-Only akan dapat melihat seluruh kalimat dari dua arah, baik kata-kata di sebelah kiri maupun kanan pada saat yang bersamaan. Kemampuan ini memungkinkan Encoder-Only untuk memahami konteks dan makna secara mendalam.

![Gambar 46](https://assets.cdn.dicoding.com/original/academy/dos-383129250034347fd823976da40d649d20260429170945.png)

Namun, karena tidak memiliki komponen decoder juga, model ini tidak bisa menghasilkan teks baru. Pendekatan ini sangat cocok untuk tugas-tugas pemahaman bahasa seperti klasifikasi teks, ekstraksi entitas, atau analisis sentimen.

#### Encoder-Decoder

*Nah*, tipe kedua inilah yang mempertahankan arsitektur lengkap dari Transformer asli. Encoder-Decoder inilah yang membuat komponen seperti *Cross-Attention*—yang berperan sebagai jembatan antara Encoder dan Decoder—dibutuhkan.

Secara sederhana, Encoder-Decoder model akan bekerja sangat baik jika Anda butuh pemahaman yang mendalam pada seluruh input sebelum mulai menghasilkan output. “Si Lengkap” ini pada umumnya digunakan pada tugas transformasi (mengubah teks input menjadi teks output), seperti *translation* yang disebutkan di atas dan juga merangkum atau *summarization*.

![Gambar 47](https://assets.cdn.dicoding.com/original/academy/dos-68cd4174493b8b37f4f3d7ec228bd7e520260429171115.png)

Ideologinya seperti ini, Encoder-Decoder model akan sangat dibutuhkan saat input dan output adalah dua hal yang terpisah. Encoder "membaca" seluruh input, merenungkannya, lalu menyerahkan "catatan" pemahamannya ke Decoder. Decoder kemudian mulai "menulis" dari nol sambil terus melihat "catatan" dari Encoder.

Oleh karena itu, Encoder-Decoder hampir tidak pernah digunakan oleh model-model seperti ChatGPT atau Gemini. Berbeda dengan tugas transformasi input, model-model *chatbot* yang membutuhkan percakapan secara berkelanjutan seperti itu akan membutuhkan arsitektur Decoder-Only yang akan kita bahas setelah ini.

#### Decoder-Only

Inilah yang keluarga GPT, Gemini, Llama, dan hampir semua LLM modern yang kita kenal gunakan. Berbeda 180 derajat dari BERT, arsitektur ini justru mengandalkan Decoder saja. Namun, bukan berarti Decoder-Only tidak memahami input yang diberikan hanya karena ia tidak menggunakan Encoder.

Keunggulannya justru terletak pada efisiensinya, satu komponen Decoder saja dapat menangani dua peran sekaligus, yaitu memahami input dan menghasilkan output. Seperti yang dibahas sebelumnya, *chatbot* membutuhkan percakapan yang berkelanjutan.

*Nah*, kunci utamanya dari model Decoder-Only adalah ia memperlakukan prompt (input) dan jawaban (output) sebagai satu sekuens tunggal yang berlanjut. Sehingga yang perlu ia lakukan adalah next-token prediction melalui teknik autoregressive yang kita bahas di bagian sebelumnya.

![Gambar 48](https://assets.cdn.dicoding.com/original/academy/dos-570b370451ffc59b9a010681463a33b320260429171137.png)

Kemampuan "melanjutkan token" inilah yang melahirkan keajaiban *zero-shot* dan *few-shot learning*. Anda tidak perlu melatih ulang model untuk tugas baru. Cukup berikan contoh di dalam prompt dan model akan mengerti polanya lalu melanjutkannya. Ini jauh lebih sulit dilakukan dengan arsitektur Encoder-Decoder yang kaku.

### Evolusi Skalabilitas: Membuka Kemampuan Baru

Ada alasan di balik mengapa Large Language Model itu menyandang gelar “Large” meski hanya menggunakan satu sisi Transformer, yaitu Decoder. Sebelumnya, kita mungkin mengira bahwa penggunaan model Decoder-Only semata bertujuan untuk membuat proses menjadi lebih efisien. Namun, kenyataannya tidak sesederhana itu.

Justru alasan “efisien” itulah yang menjadi dorongan para peneliti untuk menambah lebih banyak *layer*, memperbesar dimensi *embedding*, dan melatih model dengan lebih banyak data. Harapannya, performa akan meningkat secara linear atau setidaknya bisa diprediksi. Namun, yang terjadi justru sebaliknya karena kemunculan sesuatu yang lebih menarik, yaitu *emergent abilities*.

#### Fenomena Emergensi

*Emergent abilities* merupakan kemampuan yang tidak dimiliki oleh model kecil, tetapi tiba-tiba muncul secara kualitatif begitu model melewati ukuran kuantitatif tertentu, misalnya jumlah parameter (*Wei et al., 2022*).

Model-model raksasa tersebut mulai bisa berhitung, bernalar sebab-akibat, memahami instruksi kompleks, hingga menjelaskan langkah-langkah logis dalam menjawab pertanyaan. Seolah-olah, dengan bertambahnya jutaan hingga miliaran parameter, model “menemukan” cara untuk berpikir.

Namun, bukankah itu bisa dikatakan sebagai peningkatan performa yang linear? Baiklah perhatikan analogi berikut.

Bayangkan Anda mengajari anak kecil berhitung (1+1, 2+2, dan seterusnya). Anda mungkin berharap kemampuannya meningkat perlahan. Namun, setelah mencapai tingkat pemahaman tertentu, ia bisa mengerjakan soal cerita matematika kompleks yang belum pernah Anda ajarkan secara eksplisit sebelumnya.

Atau mari kita ambil contoh lain. Kita mengajarkan anak cara memegang pensil dan mencoret-coret di kertas. Pada awalnya, hasil coretannya tampak acak dan tidak bermakna. Namun, seiring waktu, anak itu bukan hanya makin pandai mencoret, melainkan mulai bisa menggambar wajah manusia dengan detail yang mengejutkan.

![Gambar 49](https://assets.cdn.dicoding.com/original/academy/dos-0dada8e7f95dbd4326bf6bf4a46ab91c20260313150410.png)

*Nah*, itulah *emergence abilities*. Jadi bukan sekadar peningkatan kuantitas, melainkan lompatan kualitas karena otak atau model telah mencapai kompleksitas tertentu. Dalam dunia LLM, beberapa contoh kasus emergence abilities yang muncul diantaranya adalah Few-shot Learning dan Reasoning.

Mari kita bahas keduanya.

*   **Few-shot Learning (GPT-3)** 
    Kemampuan untuk mempelajari tugas baru hanya dari beberapa contoh yang diberikan dalam *prompt*, tanpa perlu *fine-tuning*. Model kecil tidak bisa melakukan ini, tapi GPT-3 (175 miliar parameter) sangat mahir.
*   **Chain-of-Thought (CoT) Reasoning** 
    Kemampuan untuk memecah masalah kompleks menjadi langkah-langkah penalaran logis jika diberi contoh dalam *prompt*. Sederhananya, Alih-alih langsung menebak hasil akhir, model menuliskan proses berpikirnya secara eksplisit langkah demi langkah, sebelum sampai pada jawaban. Kemampuan ini baru muncul pada model dengan skala 100 miliar parameter atau lebih. Model yang lebih kecil, jika diberi *prompt* CoT, justru performanya bisa menurun.

![Gambar 50](https://assets.cdn.dicoding.com/original/academy/dos-658ae1a08a0bf015639c2b9dccd0cc7220260313150431.png)

*“Penelitian Google Brain (Wei et al., 2022b), menunjukkan bahwa dengan memberikan contoh penalaran bertahap di dalam prompt, model berukuran besar seperti PaLM 540B dapat meniru pola tersebut dan menghasilkan solusi yang jauh lebih akurat.”* 

Jika semakin besar model justru memunculkan kemampuan unik yang baru, pertanyaannya adalah akan sampai kapan? Peningkatan parameter model berbanding lurus dengan kebutuhan daya komputasi. Pada akhirnya tantangan skalabilitas ini akan menjadi tembok pembatas para peneliti untuk dapat mencari jalan lain.

Oleh karena itu, mari kita menuju ke tahap Evolusi Optimasi

### Evolusi Optimasi: Efisien, tetapi Tetap Hebat

Jika skalabilitas menjawab pertanyaan “bagaimana membuat model menjadi lebih pintar”, maka optimasi menjawab pertanyaan berikutnya: “bagaimana membuat kecerdasan sebesar itu tetap efisien dan bisa dijalankan di dunia nyata.”

Seiring bertambahnya ukuran model hingga ratusan miliar parameter, tantangan utamanya bukan lagi sekadar melatih model yang besar, tetapi bagaimana mengefisienkan perhitungannya tanpa mengorbankan kemampuan. Dari sinilah lahir berbagai inovasi dalam salah satu area utama, yaitu Feed Forward Network (FFN).

*“Meskipun sederhana, FFN biasanya memiliki jumlah parameter yang sangat besar bahkan hingga 2/3 dari total parameter model”* 

*Nah*, dari sinilah muncul masalah!

Selain ukurannya yang masif, FFN bersifat dense yang berarti setiap token yang masuk ke model akan diproses melalui seluruh neuron dalam lapisan FFN, tanpa mempertimbangkan apakah seluruh kapasitas tersebut benar-benar dibutuhkan.

Akibatnya, sebagian besar komputasi menjadi “mubazir” dan terbuang untuk menghitung hal-hal yang tidak selalu relevan bagi konteks token tertentu. Oleh karena itu, mari kita berkenalan dengan MoE sebagai versi efisien sekaligus pengganti FFN.

#### Mixture of Experts (Moe) “Si Pengganti” FFN

Sebagai “donatur” parameter terbesar dalam arsitektur Transformer, tentu FFN menjadi target utama apabila kita ingin mengefisiensikan ukuran LLM. Seperti yang kita bahas sebelumnya, ini terjadi karena setiap token atau input yang masuk akan selalu diproses oleh seluruh parameter di FFN tersebut. Jika kita ingin menambah "pengetahuan" model dengan memperbesar FFN (menambah parameter), beban komputasi untuk setiap token juga akan ikut membengkak secara proporsional.

Untuk mengatasi hal ini, mari kita ganti *Feed Forward Network* (FFN) tersebut dengan *Mixture of Experts* (MoE). MoE akan melakukan *Conditional Computation* untuk melakukan efisiensi pada Transformer.

Idenya sebenarnya sederhana, bagaimana jika kita punya banyak FFN kecil yang disebut *expert* lalu untuk setiap token, kita hanya mengaktifkan sebagian kecil dari *expert* tersebut yang paling relevan?

![Gambar 51](https://assets.cdn.dicoding.com/original/academy/dos-6fa4b83b2d1bb9de624e6e4d754735f120260305094504.png)

Analoginya seperti ini, bayangkan Anda sedang berada di dalam suatu kelas yang berisi para ahli dari berbagai bidang, seperti ahli sejarah, ahli kimia, hingga ahli matematika. Saat Anda bertanya persoalan matematika, apakah Anda akan bertanya ke seluruh ahli tersebut?

Tentu saja akan jauh lebih efisien serta dengan jawaban yang terpercaya jika Anda bertanya hanya pada ahli matematika. Anda tidak perlu meminta seluruh ahli di sana berpikir, cukup ahli yang memang relevan dengan persoalan yang ditanyakan. Kira-kira seperti itulah konsep MoE.

Sekarang pertanyaannya bagaimana bentuk MoE? Pada dasarnya, MoE terdiri dari dua komponen utama.

*   **Sejumlah Besar FFN "Experts"** 
    Ini adalah kumpulan FFN yang lebih kecil (misalnya 8, 64, atau bahkan ribuan *expert*). Masing-masing *expert* tersebut memiliki parameter sendiri. Secara teori, setiap *expert* bisa belajar untuk terspesialisasi dalam memproses jenis informasi atau pola tertentu.
*   **Gating Network (Router)** 
    Ini adalah jaringan saraf kecil yang bertindak sebagai "pengatur lalu lintas" atau "penentu arah". Tugasnya adalah melihat token input dan memutuskan *expert* mana yang harus dipanggil untuk memproses token tersebut.

![Gambar 52](https://assets.cdn.dicoding.com/original/academy/dos-e9460bb2f3cee8951d64f1d20f9212e820260313150551.png)

Jika hanya beberapa *exper*t yang terpilih dan akan memproses suatu token atau input, lantas apa yang dilakukan oleh *expert* yang tidak terpilih?

Sederhana, *expert* lainnya yang tidak dipilih akan “diam” atau tidak melakukan komputasi pada token tersebut. Kondisi ini biasanya disebut dengan *sparsity* atau ketersebaran.

Muncul pertanyaan lanjutan, mungkinkah ada kondisi di mana suatu *expert* itu tidak pernah terpilih sehingga tidak pernah aktif?

Jawabannya bisa saja terjadi. Kondisi pada saat Router memiliki bias atau kecenderungan untuk mengirim mayoritas besar token hanya ke satu atau beberapa *expert* favorit yang sama, alih-alih mendistribusikannya secara merata, kondisi tersebut dinamakan **Route Collapse**.

Pertanyaannya mengapa ini menjadi masalah? Terdapat beberapa alasan.

*   **Expert Tidak Belajar:** *Expert* yang jarang atau tidak pernah diaktifkan tidak mendapatkan kesempatan untuk belajar dari data dan menjadi tidak berguna.
*   **Kapasitas Model Menyusut:** Keuntungan utama MoE (kapasitas besar dari banyak *expert*) hilang jika hanya beberapa *expert* yang benar-benar digunakan. Model secara efektif berperilaku seperti model yang jauh lebih kecil.
*   **Kehilangan Spesialisasi:** Tujuan MoE adalah agar para *expert* bisa terspesialisasi. Jika semua token lari ke *expert* yang sama, spesialisasi tidak akan terjadi.

*Nah*, solusi dari tantangan tersebut adalah menggunakan suatu *loss function* yang disebut **Load Balancing Loss** pada saat melatih model Transformer dengan MoE. Load Balancing loss bekerja dengan cara memberikan "penalti" pada router jika distribusinya tidak seimbang.

Sampai disini, kita sudah memahami salah satu evolusi dari arsitektur Transformer yang telah menjadi fondasi asli bagi banyak LLM populer di luar sana. Sebelum kita lanjut ke bagian selanjutnya, ada sedikit refreshment melalui aktivitas berikut ini.

> **Aktivitas: Fill in the Blank**
> Lengkapi kalimat berikut dengan kata yang tepat.
> Mixture of Experts (MoE) adalah arsitektur neural network yang membagi sebuah model besar menjadi beberapa sub-jaringan spesialis yang disebut **(expert)**. Untuk menentukan sub-jaringan mana yang paling relevan dalam memproses suatu input teks, MoE menggunakan sebuah mekanisme pengarah yang dikenal sebagai **(router)**. Nah, ada kondisi di mana hanya sebagian pakar yang aktif sementara yang lainnya tetap diam saat melakukan inferensi ini dikenal dengan istilah **(sparsity)**.

---

## Sudah Sejauh Mana Implementasi LLM?

Dalam beberapa tahun terakhir, kemampuan LLM berkembang pesat melalui beberapa tahap evolusi. Mulai dari sekadar melengkapi teks otomatis, lalu menjadi model yang patuh instruksi, hingga bisa menggunakan *tool* eksternal, bertindak bak agen mandiri, dan terbaru muncul konsep bernama MCP. Mari kita telusuri perjalanan evolusi LLM dari awal hingga konsep paling mutakhir!

### Mulai Dari Berbicara dengan Prompt-Based Interactions

Ini adalah titik awal kita saat berinteraksi dengan LLM. Jika diibaratkan, prompt adalah cara kita “berbicara” atau memberi instruksi kepada LLM. *Yup*, seharusnya Anda sudah sangat familier dengan konsep bernama prompt ini. Seperti yang kita ketahui, kualitas respons sebagian besar bergantung pada cara kita menyusun perintah atau prompt. Cara kita berbicara dengan model menentukan seberapa baik hasil yang kita peroleh.

![Gambar 53](https://assets.cdn.dicoding.com/original/academy/dos-56e40120935ef4c0d4716027681e013e20260429173115.png)

#### Prompt Testing

Salah satu hal pertama yang biasa dilakukan dalam tahap ini adalah *prompt testing*. Di sini, pengguna mencoba berbagai cara untuk memformulasikan instruksi demi mendapatkan jawaban yang paling sesuai. Apa saja yang biasanya kita cari tahu di tahap ini?

*   **Rentang Kemampuan:** Sejauh mana ia bisa menjawab pertanyaan umum? Bisakah ia meringkas teks panjang? Intinya, kita mengujinya dengan berbagai jenis prompt untuk melihat responsnya.
*   **Gaya dan Nada:** Apakah outputnya cenderung formal, informal, kaku, atau kreatif? Bagaimana ia menangani permintaan untuk meniru gaya penulisan tertentu?
*   **Pengetahuan:** Seberapa *up-to-date* pengetahuannya? Apakah ia tahu peristiwa terkini? Seberapa dalam pengetahuannya di domain spesifik?

Berangkat dari sana, kita bisa mengoptimalkan pertanyaannya melalui teknik yang disebut *Prompt Engineering*.

#### Prompt Engineering

*Nah*, untuk mendapatkan hasil yang lebih konsisten, muncullah praktik yang disebut Prompt Engineering. Ini adalah seni dan teknik menyusun prompt agar model dapat menghasilkan keluaran yang lebih akurat dan relevan. Berikut beberapa teknik umum dalam *prompt engineering*.

*   **Few-shot Learning:** Pada dasarnya, ini seperti mengajari anak kecil dengan memberinya contoh agar ia bisa meniru cara melakukannya. *Nah*, dalam prompt, kita dapat memberikan contoh output yang kita inginkan sehingga membantu model belajar dalam konteks.
*   **Role Prompting:** Awali prompt dengan instruksi seperti "Bertindaklah sebagai seorang ahli biologi..." atau "Kamu adalah asisten penulis yang membantu..." untuk mengatur 'persona' model.
*   **Chain-of-Thought (CoT) Prompting:** Untuk tugas penalaran, minta model untuk "berpikir langkah demi langkah" atau berikan contoh yang menyertakan alur penalaran. Ini bisa secara dramatis meningkatkan kemampuannya dalam menyelesaikan masalah kompleks.

Seiring berkembangnya kemampuan model, muncul pula konsep Generative Configuration for Inference Parameters. Konsep seperti apa itu sesungguhnya?

#### Generative Configuration for Inference Parameters

Selain apa yang kita tulis di prompt, kita juga bisa mengatur bagaimana LLM menghasilkan responsnya saat *inference*. Anda tentu masih ingat dengan proses Hyperparameter Tuning, mengotak-atik konfigurasi hyperparameter dengan mengecilkan hyperparameter A atau memperbesar Hyperparameter B. *Nah*, ini adalah konsep yang mirip, hanya saja dilakukan pada proses generate jawaban dari Generative AI

Beberapa Parameter yang paling umum untuk dikonfigurasi adalah sebagai berikut.

*   **Temperature:** Parameter yang mengontrol tingkat keacakan atau *randomness*pada cara LLM menghasilkan jawaban.
    *   **Nilai Rendah (mendekati 0)** 
        Jika nilainya rendah, output yang dihasilkan akan lebih deterministik dan fokus. Model akan cenderung memilih kata-kata yang paling mungkin berikutnya. Cocok untuk tugas faktual atau tugas yang bersifat *strict*.
    *   **Nilai Tinggi (0.6 - 1.0)** 
        Sebaliknya, nilai yang tinggi akan membuat output lebih bervariasi, kreatif, dan kadang bahkan cukup “liar”. Model punya peluang lebih besar untuk memilih kata-kata yang kurang mungkin. Cocok untuk penulisan kreatif seperti cerita dan puisi, tetapi berisiko menghasilkan teks yang “ngelantur” jika terlalu tinggi.
*   **Top-k Sampling:** Membatasi pilihan kata berikutnya hanya pada k kata dengan probabilitas tertinggi. Misalnya, k=50 berarti model hanya akan mempertimbangkan 50 kata teratas sebelum memilih (atau sampling).
*   **Top-p (Nucleus) Sampling:** Metode sampling lain yang lebih dinamis. Alih-alih memilih k kata teratas, ia memilih sejumlah kata terkecil yang total probabilitas kumulatifnya mencapai ambang batas p (misal, p=0.9). Ini bisa lebih adaptif daripada top-k.
*   **Max New Tokens** atau **Max Length:** Membatasi panjang maksimal teks yang akan dihasilkan model. Penting untuk mengontrol biaya (jika menggunakan API) dan mencegah output yang terlalu bertele-tele.
*   **Repetition Penalty:** Memberi “hukuman” pada kata-kata yang muncul berulang-ulang, sehingga mendorong model untuk tidak mengulang kata atau frasa yang sama terus-menerus.

Secara keseluruhan, fase baseline ini bisa dianggap sebagai tahap “belajar berbicara” dengan LLM. Kita belum sampai pada tahap di mana model bisa beradaptasi dengan data eksternal atau menggunakan alat tambahan, tetapi kita sudah bisa melihat betapa fleksibelnya sistem ini hanya dengan memanfaatkan kekuatan bahasa alami.

### Knowledge Expansion pada LLM

Setelah memahami bagaimana prompt dan konfigurasi parameter inferensi membantu kita berinteraksi dengan LLM baseline, pertanyaan berikutnya yang muncul adalah: Bagaimana jika kita membutuhkan informasi yang lebih baru, spesifik, atau berada di luar batas waktu pelatihan model?

LLM, pada dasarnya, adalah sebuah "perpustakaan" yang pengetahuannya terhenti saat proses pelatihan terakhirnya selesai. Apapun yang terjadi setelah tanggal tersebut, atau informasi spesifik yang tidak tercakup dalam data pelatihan raksasanya, tidak akan diketahui oleh model. Di sinilah terjadi evolusi kritis pada cara kita memberikan pengetahuan atau knowledge pada LLM.

Evolusi ini terbagi menjadi dua jalur utama, yang masing-masing bertujuan mengatasi masalah "kebuntuan pengetahuan" dan potensi halusinasi (fenomena di mana LLM menghasilkan informasi yang terdengar meyakinkan tetapi faktualnya salah).

![Gambar 54](https://assets.cdn.dicoding.com/original/academy/dos-b8b715ea31eb4e1ee61958c157aa2ba320260429171441.png)

#### Fine-Tuning

Cara paling fundamental untuk memperbarui atau menyuntikkan pengetahuan baru ke dalam LLM adalah melalui Fine-Tuning. Bayangkan LLM sebagai seorang ahli yang telah lulus kuliah (pelatihan awal). Fine-Tuning adalah proses 'pendidikan lanjutan' atau spesialisasi yang intensif. Dengan kata lain kita mengajari kembali LLM pada sub-kumpulan data setelah pelatihan awalnya selesai.

Sub-kumpulan yang dimaksud adalah data yang lebih kecil dari segi jumlah, spesifik, dan berkualitas tinggi misalnya, dokumen internal perusahaan, data medis spesialis, atau gaya bahasa tertentu. Tujuannya adalah untuk menyesuaikan bobot (parameter) model secara halus, mengubah 'struktur otaknya' agar lebih memahami dan menghasilkan respons yang sesuai dengan domain atau tugas yang baru.

#### LLM + RAG (Retrieval-Augmented Generation)

Pendekatan berikutnya ini tidak akan menggunakan konsep pelatihan tambahan sehingga lebih ramah dengan komputasi, yaitu RAG (Retrieval-Augmented Generation). Jika Fine-Tuning mengubah pengetahuan internal model,RAG memberinya kemampuan untuk "membaca" data dari sumber eksternal yang up-to-date, sebelum menghasilkan jawaban.

Cara kerjanya sebenarnya cukup sederhana.

1.  Mencari dokumen yang paling relevan (*Retrieval*).
2.  Menyajikan dokumen tersebut ke LLM bersama dengan prompt asli.
3.  LLM kemudian menggunakan prompt dan tambahan konteks tersebut untuk menghasilkan jawaban (*Generation*).

Dengan cara ini kita bisa menjaga pengetahuan model tetap up-to-date dengan memperbarui dokumen yang ia “baca” melalui RAG tanpa perlu melatih ulang model.

#### LLM + RIG (Retrieval Interleaved Generation)

Sebagai evolusi lebih lanjut dari RAG, muncul pendekatan Retrieval Interleaved Generation (RIG). Berbeda dengan RAG tradisional yang melakukan pengambilan data (Retrieval) hanya di awal sebelum proses pembuatan jawaban (Generation), RIG memungkinkan proses pengambilan data dan pembuatan jawaban berjalan secara bergantian (interleaved).

Mekanismenya seperti berikut.

1.  LLM memulai proses *response generation*.
2.  Di tengah proses *generation*, model mengidentifikasi kebutuhan data eksternal yang spesifik.
3.  Secara proaktif "memanggil" sistem retrieval untuk mendapatkan data tersebut, seperti menyisipkan *placeholder* di tengah kalimat
4.  Kemudian melanjutkan generasinya dengan data yang baru diambil.

Kelebihan: Akurasi faktual yang lebih tinggi, terutama untuk pertanyaan kompleks yang membutuhkan beberapa fakta yang terperinci di berbagai titik dalam respons. Model bisa mengintervensi generasinya sendiri untuk memastikan data yang dimasukkan paling up-to-date. RIG memungkinkan model untuk "tahu kapan harus bertanya" dan mengintegrasikan data real-time secara lebih dinamis ke dalam alur kalimat.

![Gambar 55](https://assets.cdn.dicoding.com/original/academy/dos-29711b78b1aaa0dd13c7e8e001c83e7820260313151247.png)

Melalui ketiga jalur in, LLM telah bertransformasi dari sekadar "mesin prediksi teks" menjadi entitas yang mampu berinteraksi, beradaptasi, dan mengakses pengetahuan di dunia nyata secara dinamis.

### Tool Calling dan RAG: Perpanjangan Pengetahuan LLM

Walau LLM sudah pintar menjawab, mereka punya keterbatasan, yaitu pengetahuan mereka statis (hanya dari data latihan) dan lemah dalam tugas tertentu seperti matematika atau pemeriksaan fakta. Bayangkan seorang mahasiswa cerdas, tapi tak boleh membuka buku atau kalkulator, pasti sesekali ia salah atau lupa detail. Untuk mengatasi ini, tahap berikutnya dalam evolusi LLM adalah *tool calling* dan RAG (Retrieval Augmented Generation), yaitu memberi model akses untuk menggunakan alat atau sumber pengetahuan eksternal.

Dengan fitur ini, model bisa memutuskan kapan harus menggunakan kalkulator untuk perhitungan matematika, mencari informasi terbaru di internet, mencari dan mengambil dokumen yang kita miliki di database, atau memanggil API tertentu untuk tugas khusus. Salah satu terobosan populer adalah saat OpenAI meluncurkan plugins untuk ChatGPT. Dengan plugin, ChatGPT dapat mengambil tindakan di luar sekadar menghasilkan teks – misalnya memesan tiket pesawat, mengambil data cuaca terkini, atau memeriksa kebenaran sebuah fakta di basis data. Model secara otonom dapat memilih tool yang diperlukan untuk menjawab permintaan pengguna, tidak lagi terbatas pada pengetahuan internalnya saja.

Riset dari Meta AI bernama Toolformer juga menunjukkan hal serupa: LLM bisa dilatih untuk menyisipkan *tool calling*/API di tengah proses berpikirnya. Toolformer mampu memutuskan sendiri kapan dan alat apa yang dibutuhkan, misalnya menggunakan kalkulator untuk perhitungan, mesin pencari untuk mencari informasi, atau layanan penerjemah bila perlu. Dengan *tool calling*, LLM mendapat “kekuatan super” layaknya Doraemon yang punya kantong ajaib berisi berbagai alat. Ia bisa melengkapi kemampuannya – yang tadinya terbatas – dengan fungsi-fungsi eksternal sehingga jawaban atau aksi yang diberikan lebih akurat dan berguna.

### Agentic Behavior: LLM sebagai Agen Cerdas Mandiri

Tahap evolusi selanjutnya menjadikan LLM bukan hanya penjawab pasif, tapi agen yang dapat bertindak mandiri untuk mencapai tujuan kompleks. Konsep ini sering disebut **agentic behavior**. Jika *tool calling* memberi LLM alat, agentic LLM memberi mereka kemudi untuk menyetir sendiri. Artinya, model dapat merencanakan serangkaian langkah, mengambil keputusan, dan menggunakan alat secara berulang-ulang hingga tugas selesai dengan minimal intervensi manusia.

![Gambar 56](https://assets.cdn.dicoding.com/original/academy/dos-72e1bde2f5eff3ca482ab4edcf2fb10220260502120427.png)

Bayangkan Anda memiliki asisten digital yang bisa diberi target seperti "Tolong carikan ide bisnis baru dan buatkan rencana pemasarannya". Dengan kemampuan agen, LLM akan membagi tugas besar itu menjadi langkah-langkah kecil: mencari tren pasar, menganalisis kompetitor (dengan memanggil tool pencari informasi), lalu menyusun proposal strategi pemasaran. Ia bisa melakukan semuanya secara otonom, seolah-olah merencanakan dan bertindak sendiri.

Pada 2023, proyek *open-source* seperti AutoGPT dan BabyAGI menunjukkan demo kemampuan ini. Mereka membungkus LLM (contohnya GPT-4) dalam loop berulang: model diberikan sebuah target, kemudian ia akan terus membuat planning, mengeksekusi aksi (misalnya memanggil tool), mengevaluasi hasil, lalu memperbarui rencana hingga targetnya tercapai. Menariknya, AutoGPT langsung menarik perhatian luas – repositori GitHub-nya meraih lebih dari 100 ribu star dalam waktu singkat, menunjukkan betapa konsep agen mandiri ini diantisipasi banyak orang.

Dengan *agentic behavior*, LLM benar-benar berperan layaknya agen cerdas. Ibarat merekrut seorang junior yang bisa diberi tugas kompleks, kemudian ia sendiri yang akan mencari tahu langkah-langkah detailnya dan melakukannya sampai selesai. Tentu, teknologi ini masih baru dan kadang hasilnya belum sempurna. Namun, konsepnya membuka pintu ke AI yang dapat meringankan pekerjaan kita secara lebih proaktif, bukan hanya menjawab pertanyaan, tapi menjalankan aksi nyata.

### MCP: “USB-C” untuk Menghubungkan LLM dengan Dunia Luar

Setelah LLM mampu menjadi agen dan menggunakan berbagai tool, muncul tantangan berikutnya: bagaimana menghubungkan beragam model AI dengan sekian banyak alat dan sumber data secara mudah dan aman? Sebelum ini, integrasi setiap LLM dengan setiap tool sering membutuhkan penyesuaian khusus, setiap *framework* dan *project* memiliki protokol dan implementasi tool call yang berbeda. Solusi terbaru untuk masalah ini adalah konsep MCP (Model Context Protocol).

MCP adalah protokol terbuka yang distandarkan untuk menyambungkan model AI dengan sumber data dan alat-alat secara universal. Anggap MCP seperti port USB-C untuk aplikasi AI. Dengan USB-C, satu jenis kabel bisa dipakai untuk berbagai perangkat. Demikian pula dengan MCP yang menyediakan satu cara terpadu agar berbagai LLM (dari banyak vendor) dapat terhubung dengan bermacam tool eksternal tanpa perlu penyesuaian satu per satu.

![Gambar 57](https://assets.cdn.dicoding.com/original/academy/dos-4b26294d3e6bd2838da74c6b00f7482120260313151335.png)

MCP mengadopsi arsitektur *client-server*: aplikasi AI menggunakan client MCP untuk terhubung ke server MCP yang menjadi jembatan ke suatu sumber data atau layanan. Hal ini menyederhanakan integrasi; cukup implementasi protokol MCP pada aplikasi, Anda bisa menghubungkannya ke semua *tools* yang mendukung MCP. Selain itu, Anda dapat mengimplementasikan MCP pada *tools* API dan menghubungkannya ke semua client yang mendukung MCP. Jika dilihat sekilas, MCP hampir tidak ada bedanya dengan menggunakan API. Bedanya, client dan server pada MCP mengikuti spesifikasi khusus. Sehingga, kita sebagai developer server atau client dapat berasumsi bahwa client atau server lain dapat merespons dengan sesuai.

---

## Model Open-Source

Seperti yang kita bahas di awal, telinga Anda mungkin sudah sangat familiar mendengar nama-nama seperti ChatGPT, Gemini, dan Claude.Tahukah Anda, nama-nama *chatbot* services dengan LLM tersebut pada dasarnya adalah contoh dari Model Closed-Source atau Proprietary. Apa artinya?

Model proprietary atau closed-source adalah model LLM yang dikelola secara tertutup oleh perusahaan tertentu. Kita hanya bisa menggunakannya melalui API atau aplikasi resmi, tanpa akses langsung ke bobot model atau data pelatihannya. *Nah*, lawan kata dari model proprietary inilah yang kita sebut Model Open-Source. Pertanyaannya apa maksud dari “open” dalam model open-source?

Kata "open-source" di dunia LLM sebenarnya adalah sebuah spektrum, sehingga tidak semua model "open" itu sama. Saat kita berbicara "open", kita bisa mengategorikannya menjadi tiga hal.

*   **Open Weights:** Kategori ini adalah definisi yang paling umum digunakan saat ini untuk menggambarkan model *open-source*. Perusahaan merilis file model yang sudah dilatih yang dapat diunduh dan dijalankan oleh pengguna umum. Sebagian besar model yang akan kita bahas seperti Llama, Mistral, dan Gemma masuk kategori ini.
*   **Open Architecture:** Perusahaan hanya mempublikasikan *blueprint* atau "resep" modelnya, misalnya melalui paper penelitian. Kita dapat mengetahui cara membuatnya, tetapi tidak diberi "bahan jadi"-nya atau file model-nya secara eksplisit.
*   **Open Training Data:** kategori ini adalah level transparansi tertinggi karena perusahaan tidak hanya merilis arsitektur dan bobot, tetapi juga data mentah yang digunakan untuk melatih model. Tentu saja kategori ini sangat jarang ditemui untuk model SOTA (*State of The Art*). Alasannya tentu saja seperti yang kita ketahui, data adalah "rahasia dapur" yang paling berharga.

*Nah*, pada kelas ini, open-source yang akan kita gunakan dan eksplorasi kedepannya adalah jenis Open Weights Model. Lantas, apa saja model *open source* yang dapat kita gunakan?

Anda dapat memperhatikan ilustrasi di bawah ini.

![Gambar 58](https://assets.cdn.dicoding.com/original/academy/dos-98e7dae8e6a980224456e990f216230e20260313151537.png)

*Yup*, jawabannya adalah banyak sekali model open-source yang tersedia! Jika model proprietary didominasi oleh 3-4 perusahaan besar, ekosistem open-source jauh lebih beragam seperti yang dapat Anda lihat pada ilustrasi di atas. Namun, ada beberapa "keluarga" model utama yang mendominasi.

### Llama (Meta)

![Gambar 59](https://assets.cdn.dicoding.com/original/academy/dos-1cf2b84f8ccaf2b862d36bb3bad26e0420260305095711.png)

Llama adalah keluarga model yang dikembangkan oleh Meta AI (bagian dari Meta) dan sejak diluncurkan menjadi salah satu fondasi bagi ekosistem LLM “terbuka” di luar perusahaan besar yang selalu tertutup. Nama Llama pada model ini memang terinspirasi dari binatang berambut tipis dalam famili Camelidae yang berasal dari Amerika Selatan. Beberapa hal yang menarik yang dapat diperhatikan dalam keluarga Llama.

*   Meta menegaskan bahwa Llama 3 seperti varian 8B & 70B parameter dilatih pada dataset yang sangat besar, kurang lebih sebanyak 15 triliun token dengan cakupan lebih dari 30 bahasa.
*   Meta memperkenalkan fitur arsitektur seperti Grouped-Query Attention (GQA) agar inference lebih efisien.
*   Fokus kuat pada kontes panjang (context window) dan *deployability* yang luas pada berbagai hardware dan platform cloud.

Llama sering menjadi “standar de facto” dalam eksperimen open-source, menjadi benchmark atau tolok ukur kinerja yang harus dikalahkan oleh model lain. dan menjadi dasar dari banyak turunan seperti Vicuna dan Nous-Hermes.

### Gemma (Google)

![Gambar 60](https://assets.cdn.dicoding.com/original/academy/dos-94e29da330347de671bf58a115d195fc20260305095726.png)

Gemma adalah kontribusi open-source dari Google yang membawa teknologi riset skala besar Google ke komunitas open-source. Jika sebelumnya kita sudah sangat familier dengan Gemini yang merupakan versi proprietary, Gemma ini menjadi sisi open-source dari Google. Oleh karena membawa nama brand Google dan pengembangan internal yang kuat, Gemma menjadi pilihan “trusted” untuk banyak kasus pengembangan AI

*   Gemma 3 hadir dengan konteks hingga 128 000 token menjadikannya sebuah lonjakan besar untuk skenario yang membutuhkan dokumen panjang.
*   Gemma 3 juga Mendukung lebih dari 140 bahasa, dan juga hadir dengan varian “mobile atau edge” (Gemma 3n) yang ditujukan untuk perangkat ringan.
*   Dirancang agar “mudah di-adopsi” melalui integrasi pada banyak framework seperti JAX, PyTorch, TensorFlow dan *tooling developer ready*.

Gemma tidak berusaha menjadi “terbesar”, fokusnya berada pada “aksesibilitas riset dan pengembangan” sehingga menjadikannya model yang dapat dipercaya dan mudah dipelajari.

### Mistral AI

![Gambar 61](https://assets.cdn.dicoding.com/original/academy/dos-dede69aa0ce54023ddec23141930353920260305095744.png)

Perusahaan asal Paris ini terkenal karena menghasilkan model yang "lebih cerdas untuk ukurannya". Yup, fokus model yang dikembangkan ada pada “efisiensi tinggi”, artinya model yang parameter‐nya lebih kecil tetap punya performa tinggi.

*   Model seperti Mistral 7B dengan 7 miliar parameter dalam sebuah riset dikatakan dapat meg-*outperform* versi 13 B dari Llama 2 yang memiliki parameter hampir 2 kali lebih besar.
*   Menggunakan teknik arsitektur seperti Grouped-Query Attention (GQA) dan Sliding Window Attention agar konteks lebih panjang dan inference lebih cepat/efisien.
*   Model open source-nya dirilis di bawah lisensi Apache 2.0 yang dari sisi komunitas berarti sangat “ramah” untuk riset dan komersial.

Bisa dikatakan, Mistral merupakan sahabat bagi para engineer yang membutuhkan LLM versi mini atau SLM yang sebelumnya kita bahas, tetapi dengan performa yang tetap terjaga.

Saat Anda coba mengeksplorasi model-model terkenal tersebut, Anda akan bertemu dengan beberapa terminologi yang mungkin cukup asing, misalnya model base dan model instruct. Apa perbedaan keduanya? Mari kita bahas lebih dalam.

### Antara Model Base dan Instruct

Setelah mengenal berbagai keluarga besar model open-source satu hal yang sering terkadang membingungkan orang awam, mengapa satu nama model bisa punya banyak versi? Misalnya, kita menemukan Llama 3 Base dan Llama 3 Instruct, atau Mistral 7B Base dan Mistral 7B Instruct.

Sekilas namanya memang mirip, tetapi di baliknya terdapat dua filosofi yang sangat berbeda tentang cara model dilatih dan tujuan dari model tersebut. Baiklah mari kita bahas satu persatu.

#### Base Model

Sebelumnya kita sudah memahami bahwa LLM adalah model bahasa yang dilatih dengan jumlah data yang sangat masif untuk melakukan autoregressive atau next token prediction. *Nah*, Base Model ini adalah model mentah hasil pelatihan awal atau pre-training tersebut. Oleh karena ia dilatih untuk melanjutkan teks berdasarkan pola bahasa pada data latih, base model tidak “mengerti” perintah manusia dengan baik.

![Gambar 62](https://assets.cdn.dicoding.com/original/academy/dos-39cc0c3bb9ece01ee4c4b3f38004225e20260429171729.png)

Mengapa perilaku respon base model seperti pada ilustrasi di atas? Karena di internet, setelah kalimat "Bagaimana cara memasak nasi goreng?", sering kali muncul daftar resep, bukan resepnya itu sendiri.

Oleh karena itu, base model lebih cocok digunakan untuk riset, atau sebagai fondasi untuk fine-tuning dan menghasilkan instruct model yang akan kita bahas setelah ini.

#### Instruct

Instruct Model merupakan versi base model yang telah melalui proses pelatihan lanjutan, atau yang biasa disebut fine-tuning. Tahap ini menggunakan data yang berisi berbagai instruksi agar model mampu memahami dan mengikuti perintah dengan lebih baik.

Namun, sering muncul kesalahpahaman bahwa instruct model “lebih pintar” dibanding base model. Padahal, proses fine-tuning ini bukan bertujuan untuk menambah pengetahuan model, melainkan untuk melakukan penyesuaian (alignment)

Seperti yang kita bahas sebelumnya, base model pada dasarnya hanya dilatih untuk melanjutkan teks berdasarkan pola yang dipelajari dari buku, artikel, situs web, dan sumber data lainnya. Artinya, jika diberi pertanyaan, ia cenderung hanya melanjutkan kalimat tersebut, bukan benar-benar menjawabnya.

*Nah*, agar model bisa berperan sebagai asisten yang dapat menjawab pertanyaan atau melakukan perintah kita dengan baik, diperlukan proses fine-tuning berbasis instruksi. Dalam proses ini, model diajarkan cara merespons berbagai situasi.

*   Jika pengguna mengajukan pertanyaan, berikan jawaban yang jelas dan membantu.
*   Jika pengguna memberikan perintah, lakukan sesuai instruksinya dan hasilkan keluaran yang sesuai.
*   Jika pengguna meminta sesuatu yang berbahaya atau melanggar aturan, tolak dengan sopan dan aman.

![Gambar 63](https://assets.cdn.dicoding.com/original/academy/dos-b2c3b2777b8394059d3e84c7b1241b3320260429171653.png)

Oleh karena itu, instruct model terkadang juga disebut sebagai chat model, karena memang pada dasarnya kita memberitahu LLM cara berinteraksi dengan manusia yang baik dan benar. Dengan ini Anda dapat memilih dan membedakan model open-source sesuai dengan kebutuhan yang dihadapi.

*Nah*, setelah kita memahami perbedaan fungsi antara model base dan instruct, berikutnya kita akan memahami waktu yang tepat untuk menggunakan model open-source dibandingkan closed-source. *So, stay tuned Coders!* 

### Model Open-Source vs Proprietary

Memilih antara LLM open source dengan model proprietary seperti GPT-4, Claude, atau Gemini merupakan dilemma yang cukup mengganggu terutama pada engineer pemula. Pada dasarnya, keputusan ini dapat disesuaikan dengan kebutuhan atau *use case* yang sedang dikerjakan. Alasan paling umum untuk memilih model open source dapat kita bagi sebagai berikut.

*   **Kontrol Data & Privasi**
    Model open-source dapat dijalankan sendiri (on-premise). Contohnya, NASA menggunakan LLaMA di Stasiun Antariksa tanpa internet sehingga data sensitif tetap aman di perangkat lokal. Dengan model terbuka, “tidak perlu mentransfer data melalui perusahaan AI”.
*   **Biaya** 
    *Open-source* biasanya gratis (hanya butuh biaya hardware). Perusahaan kecil dapat menghindari langganan mahal dengan menjalankan model yang lebih kecil, tapi cukup kuat. Seperti kata Kevin Dewalt, “dengan open source Anda mempertahankan kontrol penuh atas data dan fleksibilitas operasional, meski butuh usaha engineering ekstra”.
*   **Kustomisasi** 
    Organisasi bisa menyesuaikan (*fine-tune*) model untuk kebutuhan khusus, memunculkan solusi unik di bidang tertentu (misalnya medis dan keuangan) tanpa batasan lisensi. Transparansi dan modifikasi ini membantu audit bias dan kesalahan model.
*   **Keterbatasan akses internet**
    Beberapa organisasi yang bekerja di wilayah dengan akses internet yang terbatas menggunakan platform seperti LM Studio, Ollama, dan Kolosal AI untuk membantu mengakses informasi dan mengerjakan pekerjaan mereka.

Sebaliknya, jika Anda butuh performa teratas tanpa repot setup, atau hanya sekadar aplikasi kecil dan *proof-of-concept* (PoC), model cloud berbayar (GPT-4, Claude) mungkin lebih praktis. Mereka biasanya memiliki throughput tinggi, optimasi, dan dukungan langsung. Namun, untuk privasi, biaya rendah, atau eksperimen, LLM open source menjadi solusi.

Jika ingin menggunakan model proprietary, kita sudah disediakan platform dengan interface yang mudah dipahami. Bahkan jika Anda ingin membuat aplikasi *chatbot* sendiri, OpenAI dan Google sudah menyediakan *tools* dan API yang dapat kalian manfaatkan.

Sekarang pertanyaannya bagaimana dengan model *Open-Source*? Mari ikuti petualangan untuk berkenalan dengan beberapa pendekatan yang dapat dilakukan untuk “bermain*” dengan LLM Open-Source*.

*Eits* sebelum itu, ada sedikit iklan sejenak melalui aktivitas di bawah ini untuk semakin mempertebal pemahaman kalian mengenai pengelompokan model LLM.

> **Aktivitas: Flashcards**
> - Model pondasi dasar yang dilatih menggunakan triliunan teks hanya untuk memprediksi kata selanjutnya, tetapi belum mahir dalam menjawab perintah pengguna. (**Jawaban:** Base Model)
> - Model AI yang bobot (weights) dan kode sumbernya tersedia secara publik, memungkinkan developer untuk mengunduh, memodifikasi, dan menjalankannya secara lokal di server mereka sendiri. (**Jawaban:** Model Open Source)
> - Model AI yang dikembangkan secara tertutup oleh perusahaan tertentu, di mana arsitektur dan bobot (weights) modelnya tidak dibagikan ke publik dan biasanya diakses melalui API berbayar. (**Jawaban:** Model Proprietary)
> - Model bahasa yang sudah melewati proses fine-tuning lanjutan agar mahir merespons prompt, mengikuti arahan, dan berdialog layaknya asisten. (**Jawaban:** Instruct Model)

### Mari Menggunakan Model Open-Source

Sebelum mulai ke praktik, akan sangat baik bagi Anda untuk familier dengan beberapa *tools* atau pendekatan yang tersedia. *Nah*, untuk Anda yang ingin mencoba model tanpa coding, terdapat dua *tools* yang sudah cukup populer.

*   **Ollama:** Sangat populer karena kemudahannya. Anda bisa mengunduh dan menjalankan berbagai model *open-source* (seperti Llama 3, Mistral, Phi-3) dengan perintah sederhana di terminal. Ollama mengurus detail backend-nya.
*   **LM Studio:** Menawarkan antarmuka grafis atau GUI yang lebih ramah pengguna. Anda bisa mencari, mengunduh, dan chatting dengan model *open-source* langsung dari aplikasi desktop, lengkap dengan pengaturan konfigurasi.

Baiklah kita akan mencoba untuk menggunakan pendekatan tanpa coding ini terlebih dahulu. Mari kita gunakan Ollama sebagai contoh.

#### Inference Model dengan Ollama

Sebelum kita bermain-main dengan LLM menggunakan Ollama, ada hal sakral yang perlu kita lakukan. Tentu saja, langkah pertama adalah menginstal Ollama itu sendiri. Anda dapat menggunakan tautan berikut ini untuk mengunduh dan menginstal Ollama di Komputer atau Laptop Anda: [Download Ollama](https://ollama.com/download/windows). Anda hanya perlu menyesuaikan dengan Operating System yang digunakan.

Jika sudah terinstal dengan baik, Anda sebenarnya akan langsung disuguhkan dengan *interface chat* yang *user friendly* dan dapat langsung digunakan untuk berinteraksi dengan model. Tampilannya kira-kira akan seperti berikut ini.

![Gambar 64](https://assets.cdn.dicoding.com/original/academy/dos-d2bd90119cf0efcaf3362e64d32d1a1a20260305100029.png)

Hanya saja, untuk experience yang lebih menantang, kita akan menggunakan jalur Command-Line Interface (CLI). Pertama yang perlu Anda lakukan adalah membuka Windows Powershell maupun Command Prompt pada laptop Windows, Terminal jika menggunakan macOS, atau Shell jika menggunakan Linux. Jika sudah terbuka, Anda dapat secara *straightforward* menjalankan perintah di bawah ini.

```bash
ollama run llama3.2:3b
```

Perintah tersebut akan langsung menjalankan model LLM bernama Llama 3.2 3B yang memiliki tiga milyar parameter. Namun, jika Anda belum pernah mengunduh model tersebut sebelumnya, maka ollama akan secara otomatis mengunduh model saat Anda menjalankan perintah di atas.

Kira-kira tampilannya akan seperti ini saat mengunduh model.

![Gambar 65](https://assets.cdn.dicoding.com/original/academy/dos-5e615bdc743f867c298f629623006ac820260305100100.png)

Proses *download* akan menggunakan internet yang tersambung ke perangkat Anda. *Nah*, model yang kalian unduh akan tersimpan langsung di penyimpanan device. *By default*, Ollama akan menyimpan model di folder cache lokal, tergantung sistem operasi device Anda.

Anda mungkin bertanya-tanya, bagaimana cara kita mengetahui model apa saja yang tersedia dan dapat kita unduh dalam Ollama. Anda dapat mengunjungi dokumentasi model Ollama melalui tautan yang disediakan: [Ollama’s Models](https://ollama.com/search).

![Gambar 66](https://assets.cdn.dicoding.com/original/academy/dos-3b4701bb89c826cf2989e893e8f5294020260305100114.png)

Di sana Anda dapat menemukan cukup banyak model LLM dan bahkan Vision Language Model yang tersedia. Pada contoh kali ini, kita akan menggunakan LLM terlebih dahulu.

*“Penting untuk diingat, gunakan model yang tidak terlalu besar, misalnya 1B hingga 3B saja karena model akan dijalankan menggunakan compute power device Anda.”*

Baiklah kembali ke CLI Ollama kita, saat proses pengunduhan model sudah selesai, tampilannya akan seperti berikut ini.

![Gambar 67](https://assets.cdn.dicoding.com/original/academy/dos-56f2a266290d5bed262cfbde9ec9d60c20260305100127.png)

Setelah itu apa yang perlu kita lakukan?

Sebetulnya pada tahap ini kita bisa langsung memasukan perintah atau prompt pada model yang sudah kita unduh tersebut. Sangat sederhana bukan? Mari kita gunakan prompt sederhana dan sedikit bermain dengan model yang sudah kita sajikan ke “atas meja” ini.

Kira-kira akan seperti ilustrasi di bawah ini saat Anda melakukan inference pada Ollama.

![Gambar 68](https://assets.cdn.dicoding.com/original/academy/dos-834db2042b4e82e119ab77bfcd53c22f20260305100416.gif)

Selain melakukan inference, Anda juga dapat sedikit mengulik dan mengenali LLM yang sedang dijalankan. Anda dapat menjalankan perintah di bawah ini untuk mendapatkan informasi mengenai LLM yang sedang berjalan.

```bash
ollama show llama3.2:3b
```

Output-nya akan seperti pada gambar di bawah ini.

![Gambar 69](https://assets.cdn.dicoding.com/original/academy/dos-6a39a3f81a009a358435a58f1082495f20260305100454.png)

Output-nya menjelaskan detail spesifikasi model LLaMA 3.2 3B, yang mencakup beberapa hal berikut.

*   **architecture:** llama
*   **parameters:** 3.2B (Besar parameter model, yaitu 3,2 miliar parameter)
*   **context length:** 131072 (Panjang konteks maksimum yang dapat diterima model dalam sekali pemrosesan)
*   **embedding length:** 3072 (Ukuran dimensi vektor representasi internal model)
*   **quantization:** Q4_K_M (Menunjukkan metode kuantisasi yang digunakan untuk mengompres ukuran model)

Jika Anda bertanya-tanya tentang quantization, tenang saja karena kita akan membahasnya pada bagian yang terpisah. Nah, selain itu ada juga beberapa informasi tambahan seperti berikut.

*   **Capabilities:** completion, tools
*   **Parameters stop:** token akhir atau batas output model
*   **License:** LLaMA 3.2 COMMUNITY LICENSE AGREEMENT
*   **Version Release Date:** September 25, 2024

Sebenarnya masih ada beberapa function atau perintah yang bisa dieksplorasi. Anda dapat mencoba beberapa perintah selain yang sudah kita lakukan pada ilustrasi di bawah ini.

| Perintah | Deskripsi |
| :--- | :--- |
| Ollama serve | Memulai Ollama di sistem lokal Anda. |
| Ollama create `<new_model>` | Membuat model baru dari model yang sudah ada untuk penyesuaian atau pelatihan. |
| Ollama pull `<model>` | Mengunduh model yang ditentukan ke sistem Anda. |
| Ollama list | Mencantumkan semua model yang sudah diunduh. |
| Ollama ps | Menampilkan model yang sedang berjalan. |
| Ollama stop `<model>` | Menghentikan model yang sedang berjalan sesuai yang ditentukan. |
| Ollama rm `<model>` | Menghapus model dari sistem sesuai yang ditentukan. |

Pada tahap ini, Anda sudah berhasil melakukan *inference* menggunakan platform yang sebenarnya bisa dikatakan *Low Code*. Sebagai Engineer dan Developer, tentu Anda tidak akan puas jika belum melakukan inference dengan jalur yang lebih *Code*. Oleh karena itu, mari kita berpindah ke jalur satunya.

#### Inference Model via Code

Selain dengan pendekatan tanpa coding, kita juga bisa menggunakan LLM dan bahkan melakukan hal yang lebih dari sekadar *inference* via Code. Pendekatan ini akan mengantarkan kita untuk berkenalan dengan Hugging Face.

Apa itu Hugging Face?

Anggap Hugging Face sebagai kombinasi antara GitHub, perpustakaan raksasa, dan taman bermain untuk AI *open-source*. Ini bukan cuma satu library, tapi sebuah platform dan komunitas besar yang menyediakan *tools* dan sumber daya penting. Kita bisa menyebut Hugging Face sebagai “Markas Besar” *LLM Open-Source*.

Kita sudah membahas Hugging Face ini secara detail di kelas Belajar Fundamental *Generative AI* sebelumnya. Jadi, saatnya kita masuk ke tahap implementasi langsung menggunakan lingkungan yang paling umum digunakan untuk eksperimen machine learning dan sudah menjadi sahabat dekat kita, yaitu Google Colab.

*First thing first*, kita perlu memastikan bahwa Google Colab yang digunakan sudah mengakses GPU T4 dengan baik. Anda dapat menjalankan kode di bawah ini untuk memeriksanya.

```bash
!nvidia-smi
```

Output yang diharapkan saat Colab Anda sudah terkoneksi dengan GPU T4 akan seperti berikut ini.

![Gambar 70](https://assets.cdn.dicoding.com/original/academy/dos-956415e42dc64b0869563fcdfc5fe99020260305101004.png)

Nampak pada output tersebut, muncul informasi tentang GPU yang digunakan, yaitu Tesla T4 dengan versi CUDA 12.4.

*Next step*, tentu saja perlu menginstal beberapa dependencies yang diperlukan. Agar semuanya smooth dan tidak terjadi error di tengah jalan, Anda dapat menggunakan versi dari beberapa dependencies yang akan kita gunakan berikut ini.

```text
accelerate==1.11.0
bitsandbytes==0.48.1
huggingface-hub==0.36.0
transformers==4.57.1
```

Sehingga, kita bisa melakukan proses instalasi sesuai dengan versi dependencies yang direkomendasikan menggunakan kode berikut ini.

```bash
!pip install accelerate==1.11.0 
!pip install bitsandbytes==0.48.1 
!pip install huggingface-hub==0.36.0 
!pip install transformers==4.57.1
```

Setelah semua selesai terinstal dengan baik, saatnya kita berinteraksi dan membuka pintu menuju model LLM melalui Hugging Face. Caranya bagaimana? Yup, kita akan login ke Hugging Face menggunakan akun Hugging Face yang sudah dimiliki.

Oleh karena itu pastikan Anda sudah memiliki akun Hugging Face. Eits sebelum login, karena Anda sudah membuat atau memiliki akun Hugging Face, ada satu langkah tambahan yang perlu dilakukan. Model LLM yang akan kita gunakan adalah Model Gemma 3 4B dari Google yang memiliki lisensi terbatas (*gated access*).

![Gambar 71](https://assets.cdn.dicoding.com/original/academy/dos-5bf5cf243610b66b94dd941c348ebf6720260305101055.png)

Oleh karena itu, Anda menyetujui Terms of Use menggunakan akun Hugging Face milik Anda terlebih dahulu di halaman model: [https://huggingface.co/google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it).

Setelah itu, Anda dapat menjalankan kode berikut untuk mengintegrasikan Hugging Face Anda dengan Google Colab.

```bash
!hf auth login
```

Setelah Anda menjalankan kode di atas, seharusnya Anda akan diminta untuk memasukan token Hugging Face dengan output seperti di bawah ini.

![Gambar 72](https://assets.cdn.dicoding.com/original/academy/dos-b8498ba1ff8871f275e6c09b46e8754a20260305101125.png)

Pertanyaannya sekarang adalah bagaimana cara mendapatkan token tersebut?

Caranya sangat sederhana dan 100% gratis. Anda hanya perlu pergi ke laman access token pada Hugging Face menggunakan akun yang sudah dibuat. Anda dapat menggunakan tautan berikut ini [Access Token Hugging Face](https://huggingface.co/settings/tokens)

Akan muncul tampilan seperti di bawah ini.

![Gambar 73](https://assets.cdn.dicoding.com/original/academy/dos-2a9091e53f54e3acdd7abd16b89925ed20260305101205.png)

*Nah*, selanjutnya Anda tinggal membuat token baru menggunakan button + Create new token ada di kanan atas. Setelah Anda tekan, tampilannya akan seperti di bawah ini.

![Gambar 74](https://assets.cdn.dicoding.com/original/academy/dos-770569da834ec8f3100058a3289b631c20260305101220.png)

Jika Anda bingung dengan *Token type*, Anda dapat memilih **Read** saja. Setelah itu Anda perlu memberikan nama token Anda dan men-*generate* token dengan menekan *button Create token*. *Nah*, jika token sudah dibuat, Anda dapat menyalin token tersebut dan kembali ke Google Colab untuk memasukan token pada field yang sudah tersedia saat menjalankan kode sebelumnya.

Tampilan setelah Anda berhasil memasukan token Hugging Face seharusnya akan seperti berikut ini.

![Gambar 75](https://assets.cdn.dicoding.com/original/academy/dos-2b5e0a797fa5666194009acda1e59ad520260305101235.png)

Jika sudah berhasil, selamat perjalanan Anda untuk berpetualang di dunia LLM resmi dimulai! Setelah beberapa tahapan setup sudah kita bereskan, saatnya masuk ke “Main Course”. Pertama tentu saja kita perlu untuk mengunduh atau memuat model Gemma 3 1B yang akan kita gunakan.

Di sini, kita tidak hanya meng-load model Gemma , tetapi beserta tokenizernya juga. Nah, ada pertanyaan menarik, apakah kita bisa menggunakan tokenizer yang berbeda dengan model yang digunakan.

Sayangnya, jawabannya adalah TIDAK bisa.

Setiap model dilatih dengan tokenizer spesifiknya sendiri. Seperti yang kita pelajari pada bagian tokenization, tokenizer menentukan "kosakata" (kata atau potongan kata apa saja yang dikenali) dan cara memecah teks. Menggunakan tokenizer dari model A untuk input ke model B akan menghasilkan angka-angka yang tidak dimengerti oleh model B, ibarat memberikan peta dengan simbol yang salah.

Kembali ke proses loading, Anda dapat menjalankan kode berikut ini.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Pilih model Gemma yang ingin digunakan
model_name = "google/gemma-3-1b-it"

# Muat tokenizer (untuk memproses input teks)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Muat model yang telah dipilih dengan konfigurasi optimal untuk GPU
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",            # Gunakan GPU jika tersedia secara otomatis
    dtype=torch.float16           # Menggunakan tipe data float16 yang disupport oleh GPU T4
)
```

Cukup banyak modul dan parameter yang asing di mata kita, mari berkenalan dengan beberapa di antaranya.

*   `AutoTokenizer`: modul dari Hugging Face yang secara otomatis memuat tokenizer sesuai dengan model yang kita pilih.
*   `AutoModelForCausalLM`: modul dari Hugging Face yang secara otomatis untuk memuat model yang digunakan untuk tugas causal language modeling. Ini adalah jenis model yang dilatih untuk tugas *next token prediction*, sehingga cocok untuk *text generation*.
*   `torch`: pustaka utama dari PyTorch yang menangani komputasi numerik dan pemrosesan GPU.
*   `device_map="auto"`: Menginstruksikan model untuk secara otomatis memilih perangkat terbaik untuk menjalankan model.
    *   Jika GPU tersedia (seperti di Colab), model akan dipindahkan ke GPU (cuda).
    *   Jika tidak, model akan dijalankan di CPU (cpu), meskipun lebih lambat.
*   `dtype=torch.float16`: Menentukan tipe data numerik yang digunakan dalam perhitungan model. Kita akan mempelajarinya pada bagian fine-tuning pada bagian berikutnya.

Baiklah, setelah Anda jalankan kode di atas, Model dan Tokenizer Gemma 3 1B akan di-Load ke environment Google Colab. Setelah prosesnya selesai, outputnya akan terlihat seperti ini.

![Gambar 76](https://assets.cdn.dicoding.com/original/academy/dos-355c86367fec4ce3e3716fb03d3ef24520260305101411.png)

Jika seluruh langkah tersebut sudah tuntas, *it’s time to play with the LLM*. Saatnya kita melakukan inference menggunakan model Gemma yang sudah di-load. Sebagai catatan Anda dapat mengganti bagian prompt dengan pertanyaan apapun untuk melihat hasil berbeda.

```python
prompt = "Jelaskan mengapa langit berwarna biru."

# Tokenisasi prompt dan kirim ke GPU
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# Jalankan inferensi dengan panjang maksimal output 500 token
output_tokens = model.generate(
    **inputs,
    max_new_tokens=500,
    do_sample=True  # Gunakan do_sample=True untuk hasil yang lebih variatif
)

# Ubah token hasil inferensi kembali menjadi teks
output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

print(output_text)
```

Sebelum melihat output dari kode di atas, kita perlu mengenali beberapa parameter yang digunakan pada kode inference tersebut.

*   `return_tensors="pt"`: Ini menandakan bahwa hasil tokenisasi dikembalikan dalam format tensor PyTorch.
*   `.to("cuda")`: digunakan untuk memindahkan tensor hasil tokenisasi ke GPU agar seluruh proses berjalan lebih cepat.
*   `**inputs`: Sintaks ** ini "membongkar" dictionary inputs yang dihasilkan tokenizer (yang berisi input_ids dan mungkin attention_mask) menjadi argumen-argumen yang dibutuhkan oleh fungsi generate().
*   `max_new_tokens=500`: Parameter ini membatasi panjang output yang akan dihasilkan oleh model. Model akan berhenti menghasilkan teks setelah menambahkan maksimal 500 token baru setelah prompt awal. Ini mencegah model menghasilkan teks yang terlalu panjang tanpa henti.
*   `do_sample=True`: Parameter ini mengaktifkan metode sampling saat model memilih kata berikutnya.
    *   Jika False, model akan menggunakan *greedy decoding* (selalu memilih kata dengan probabilitas tertinggi), yang bisa membuat hasilnya repetitif atau membosankan.
    *   Apabila True, model akan memilih kata berikutnya secara probabilistik berdasarkan distribusi probabilitas yang dihasilkannya. Ini sering kali membuat output lebih bervariasi, alami, dan kreatif.

*Nah*, hasil dari `generate()` adalah tensor angka yang merepresentasikan token-token output dan disimpan dalam output_tokens. Namun, tentu saja kita tidak bisa mendapatkan output jawabannya sekarang karena masih dalam bentuk tensor angka. Oleh karena itu, pada *step* akhir kita melakukan decoding pada `output_tokens`.

Anda dapat menjalankan kode di atas dan mendapatkan respons dari prompt yang ditulis sebelumnya. Bagaimana menarik bukan? Model yang Anda load sendiri dapat menghasilkan respons yang cukup informatif.

Jika kode sebelumnya kita melakukan inference hanya menggunakan string prompt biasa, sekarang kita akan mencoba untuk menggunakan Chat Template. Chat Template adalah format yang diekspektasikan oleh model instruct terima. Saat berurusan dengan model base yang memang fokusnya hanyalah next token prediction, pendekatan string prompt biasa seperti pada kode sebelumnya mungkin sudah optimal.

Namun saat berurusan dengan model instruct yang memang secara khusus dilatih untuk melakukan percakapan secara dua arah layaknya kita berbicara dengan assistant, Chat Template akan menunjukan kemampuannya.

*Eits*, meskipun menggunakan format yang berbeda, percakapan yang terjadi pada dasarnya tetap berupa sekuens token. Format berisi pasangan `{role, content}` yang Anda berikan ke model chat akan diubah menjadi rangkaian token, biasanya dengan tambahan token kontrol khusus seperti `<|user|>`, `<|assistant|>`, atau `<|end_of_message|>`.

Token-token kontrol ini memberitahu model tentang struktur percakapan, misalnya bagian mana yang berasal dari pengguna dan bagian mana yang merupakan respons asisten.

Ada banyak format percakapan yang mungkin digunakan, dan setiap model dapat memiliki format atau token kontrol yang berbeda-beda, meskipun mereka berasal dari model dasar (base model) yang sama dan hanya dibedakan pada tahap fine-tuning.

Baiklah kita hentikan dulu penjelasan teorinya, sekarang masuk ke cara implementasinya. Anda dapat menjalankan kode berikut ini.

```python
# Menyusun prompt dalam bentuk chat, dimana setiap member memiliki role antara: `user`, `assistant`, `system`, dan `tools`.
chat = [
  {"role": "user", "content": "Jelaskan mengapa langit berwarna biru."}
]

# Mengubah chat menjadi bentuk prompt dengan format khusus yang dimiliki oleh model
prompt = tokenizer.apply_chat_template(chat, tokenize=False)

# Tokenisasi prmpt dan dikirim ke GPU
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# Jalankan inferensi dengan panjang maksimal output 500 token
output_tokens = model.generate(**inputs, max_new_tokens=500, do_sample=True)

# Ubah token hasil inferensi kembali menjadi teks
output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

print(output_text)
```

Secara garis besar, tidak banyak yang berubah dari kode inference sebelumnya. Satu-satunya yang berbeda adalah cara Anda mendefinisikan input atau prompt untuk model.

*Nah*, selain menggunakan Chat Template, kita juga dapat menambahkan sedikit pemanis pada cara model mengeluarkan output respons. Jika Anda perhatikan, respons yang dihasilkan oleh model Gemma yang kita gunakan muncul secara tiba-tiba.

Hal ini terjadi karena model menghitung seluruh token *output* secara internal, lalu mengirimkan hasil akhirnya sekaligus.

Berbeda tentunya dengan saat kita menggunakan model Proprietary seperti ChatGPT yang menampilkan hasil secara bertahap (token per token) melalui mekanisme streaming inference. Namun, bukan berarti kita tidak bisa melakukan hal tersebut juga pada model *open-source* yang kita gunakan.

Coba Anda jalankan kode berikut ini.

```python
from transformers import TextStreamer

prompt = "Apa yang membuat air laut terasa asin?"

# Persiapkan input
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# Inisialisasi streamer untuk menampilkan teks secara bertahap
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

# Jalankan inferensi dengan streamer
_ = model.generate(
    **inputs,
    max_new_tokens=500,
    do_sample=False,
    streamer=streamer
)
```

Library Transformer dengan baik hati sudah menyediakan modul `TextStreamer` yang memungkinkan kita untuk mereplikasi mekanisme streaming output tersebut, sehingga teks bisa muncul bertahap saat model sedang melakukan inferensi.

Sekarang, kita akan membahas dua parameter yang masih asing bagi kita pada kode di atas.

*   `skip_prompt=True`: agar prompt input tidak ikut ditampilkan kembali di output (hanya jawaban model yang muncul).
*   `streamer=streamer`: memberi tahu model agar mengirim setiap token yang baru dihasilkan langsung ke `TextStreamer` yang sudah kita definisikan pada *line code* di atasnya, bukan menunggu semua selesai dulu.

Sekarang bagaimana hasil respons yang dikeluarkan? Sudah mulai mendekati cara model-model raksasa di luar sana bukan? Namun, tentu secara kualitas jawaban masih tidak bisa dibandingkan dengan model-model Proprietary karena perbedaan ukuran atau jumlah parameter.

*At least*, kita sudah mendapatkan gambaran tentang state of the art dari melakukan inference dengan LLM atau yang bisa kita sebut SLM (karena menggunakan model dengan parameter di bawah 10 milyar saat ini).

Sekarang setelah memahami model Open Source dan bahkan melakukan inference melalui dua sisi yang berbeda, saatnya kita memahami siklus kehidupan aplikasi *Generative AI* sebagai penutup dari modul pertama di kelas ini.

*One last push, Coders!* 

---

## Generative AI Project Lifecycle

Baiklah, kita sudah menjelajahi arsitektur Transformer, evolusinya yang pesat, hingga menyelami dunia model *open-source* dan cara berinteraksi dengannya. Kita sudah punya pemahaman mendalam tentang bagaimana model-model ini bekerja dan bagaimana kita bisa menjalankannya.

Namun, membangun model yang hebat di lab atau di laptop kita hanyalah separuh cerita. Perjalanan membawa “keajaiban” *Generative AI* ini dari eksperimen menjadi produk atau layanan nyata yang bisa diandalkan, diskalakan, dan terus ditingkatkan adalah cerita yang berbeda.

![Gambar 77](https://assets.cdn.dicoding.com/original/academy/dos-bad528788499f405f816f852455afa0c20260313152140.png)

Di sinilah kita memasuki ranah *Generative AI* Ops (GenAI Ops). Ini adalah evolusi dari MLOps (Machine Learning Operations) yang secara khusus disesuaikan untuk tantangan unik dalam mengelola siklus hidup model generatif, terutama LLM. GenAI Ops bukan sekadar deployment, tapi serangkaian praktik dan *tools* untuk memastikan model generatif kita bekerja secara efektif, efisien, dan bertanggung jawab di dunia nyata.

Mari kita beda satu per satu tahapan-tahapan kunci dalam Lifecycle GenAI Ops.

### Menentukan Scope

Sebelum menimbang model, data, atau metrik, penting sekali menghabiskan waktu untuk menjelaskan dengan “gamblang” titik *finish* yang ingin dicapai. Banyak proyek *Generative AI* yang “tersesat” bukan karena modelnya buruk, melainkan karena tujuan dan batasannya belum dirumuskan sejak awal. Di sinilah fondasi keberhasilan dibangun.

Scope ini bisa kita tentukan dengan memenuhi beberapa checklist yang akan kita bedah lagi satu per satu.

![Gambar 78](https://assets.cdn.dicoding.com/original/academy/dos-5c2fb1ae66f1bb3b7159873850115e9c20260313152214.png)

#### Tujuan Inti Proyek

Ini bisa kita mulai dengan menuliskan satu atau dua kalimat yang menjawab pertanyaan berikut.

*“Masalah spesifik apa yang ingin diselesaikan, untuk siapa, dan apa ukuran keberhasilannya?”* 

Contohnya seperti berikut ini.

*   Membangun asisten konversasional untuk dukungan pelanggan yang mampu menjawab FAQ seputar pengembalian barang dengan akurasi ≥ 90% dalam 3 bulan.
*   Membuat generator ringkasan meeting yang menghasilkan ringkasan 5–7 kalimat yang mencakup keputusan dan action items.

*Nah*, dari sana kita bisa berangkat ke checklist kedua.

#### Siapa Stakeholder-nya?

*Yup*, identifikasi pemangku kepentingan dan peran orang-orang yang terlibat dalam proyek ini.

*   **Pengguna akhir:** CS agent, marketing, manajer produk, dan sebagainya.
*   **Pemilik produk:** Kepala yang akan memutuskan fitur dan besar anggaran.
*   **Tim data dan ML Engineer:** pihak yang akan mengolah data dan menjalankan eksperimen.
*   **Legal atau ** * **Compliance** * **:** pihak yang bertanggung jawab pada regulasi dan privasi.

Kita juga dapat menyertakan ekspektasi tiap pemangku kepentingan sebagai titik *start* atau acuan pada saat diskusi.

#### Batasan dan Ruang Lingkup Fungsional

Penting juga untuk menentukan apa yang AKAN dan TIDAK AKAN dilakukan oleh aplikasi yang dibangun. Harapannya ini akan menjadi pembatas agar scope yang didefinisikan tidak terlalu liar karena dapat berdampak pada *manpower*, waktu pengembangan aplikasi, hingga keamanan.

Contohnya seperti berikut ini.

*   **Akan:** menjawab pertanyaan produk, mengusulkan langkah solusi teknis sederhana.
*   **Tidak akan:** memberikan saran hukum atau medis yang bersifat final; mengakses data sensitif pengguna tanpa persetujuan.

#### Data: Jenis dan Sumber

Inventarisir data yang akan digunakan oleh aplikasi *Generative AI* yang dibangun.

*   Sumber internal (transkrip Customer Service, knowledge base, dokumen produk).
*   Sumber eksternal (wikipedia, dataset publik).
*   Kebutuhan lisensi dan hak cipta.

Metrik Utama untuk Evaluasi

Definisikan beberapa metrik yang dapat digunakan untuk mengetahui scope sudah tercapai. Contoh beberapa metrik yang dapat digunakan adalah sebagai berikut.

*   Akurasi jawaban faktual yang dapat diukur oleh manusia sebagai annotator atau service LLM yang lebih canggih.
*   Tingkat kepuasan pengguna, misalnya menggunakan CSAT.
*   *Latency* atau waktu tunggu yang dibutuhkan oleh model untuk menghasilkan respons.

Penting untuk memastikan metrik terdiri dari dua sisi, yaitu sisi kuantitatif dan sisi kualitatif yang berbasis penilaian oleh manusia (quality check).

Selain itu ada beberapa checklist yang mungkin akan digunakan tergantung dari *use case* dan kebutuhan suatu perusahaan atau tim yang terlibat dalam pengembangan aplikasi *Generative AI*. Setelah scope sudah diperjelas, kita akan melangkah ke tahapan selanjutnya.

### Model Selection

Baiklah, kita akan masuk ke tahap yang cukup menarik karena kita akan mulai bersentuhan dengan LLM itu sendiri, yaitu pemilihan model. Seperti yang kita ketahui, di luar sana sudah banyak sekali variasi dan pilihan model yang dapat digunakan, bahkan platform Hugging Face menyediakan lebih dari dua juta model.

Perlu menjadi catatan, pemilihan model sebaiknya tidak dilakukan secara intuitif atau berdasarkan tren, tetapi melalui sejumlah kriteria objektif yang disepakati sejak awal. Pada bagian sebelumnya kita sudah membahas pertimbangan dalam memilih model Proprietary dengan model *open-source*. Selain kriteria tersebut, berikut kriteria utama yang biasanya dipertimbangkan.

![Gambar 79](https://assets.cdn.dicoding.com/original/academy/dos-35fa0b0ead4212c71489bf043ddd7af020260313152336.png)

#### Kecocokan Tugas (Task Fit):

Pertama tentu saja kita perlu memilih model yang sesuai dengan output yang diinginkan. Beberapa contoh tugas yang dapat dieksplorasi menggunakan LLM adalah seperti berikut ini.

*   **Text generation dan Chatbot:** Task seperti ini dapat menggunakan model seperti Gemma, LLaMA, Mistral, atau GPT-4.
*   **Code Generation:** Kasus seperti ini dapat memertimbangkan CodeLlama, StarCoder, atau CodeGemma.

#### Ukuran & Kompleksitas Model

Pembahasan soal ukuran dan kompleksitas model, akan mengarahkan kita pada LLM dan SLM kembali. LLM sebagai model besar yang memiliki lebih dari 10 milyar atau 10B parameter yang mampu memahami konteks panjang dan menghasilkan respons yang kaya, tetapi menuntut daya komputasi besar.

Sementara SLM sebagai model yang lebih kecil akan lebih hemat sumber daya dan bisa dioptimalkan di perangkat lokal. Tentu saja *best practice*-nya adalah menggunakan model sekecil mungkin yang tetap memenuhi kebutuhan kualitas.

#### Kemampuan Bahasa & Domain

Pastikan model yang digunakan memiliki dukungan bahasa dan terminologi yang sesuai dengan bidangnya. Misalnya, model berbahasa Inggris biasanya lebih unggul dalam penalaran teknis, tetapi mungkin kurang efektif untuk percakapan dalam bahasa Indonesia. Jika diperlukan, model dapat disesuaikan atau dilatih ulang menggunakan dataset lokal agar lebih relevan dengan konteks penggunaan.

#### Lisensi dan Kepatuhan (Compliance)

Pastikan lisensi model yang digunakan sesuai dengan kebijakan organisasi. Beberapa model *open-source* dengan open weight, seperti Gemma atau LLaMA, dapat digunakan secara bebas, sementara model lain mungkin memiliki batasan untuk penggunaan komersial. Selain itu, perhatikan juga aspek privasi dan pengelolaan data, terutama jika aplikasi berinteraksi dengan informasi yang bersifat sensitif.

### Model Alignment dan Adaptation

Tahap model alignment dan adaptation berfungsi untuk menyesuaikan model agar benar-benar sejalan dengan tujuan dan kebutuhan aplikasi. Proses ini melibatkan pengambilan model yang sudah dilatih sebelumnya (pre-trained model) dan mengadaptasinya agar sesuai dengan kasus penggunaan tertentu.

Penyesuaian dapat dilakukan melalui berbagai teknik yang beberapa sudah sempat kita singgung di bagian sebelumnya, seperti prompt engineering, retrieval-augmented generation (RAG), agents, fine-tuning, continuous *pre-training*, model distillation, hingga human feedback alignment.

Hanya saja, apakah sepenting itu melakukan alignment dan adaptation?

Base model seperti Gemma, LLaMA, atau Mistral telah dilatih menggunakan data umum yang sangat masif jumlahnya dari internet. Artinya, mereka memahami bahasa dan pengetahuan dunia secara luas, tetapi belum tentu memiliki beberapa kemampuan berikut ini.

*   Pengetahuan tentang konteks domain suatu organisasi, misalnya medis, hukum, atau perbankan.
*   Pemahaman gaya komunikasi yang diharapkan pengguna
*   Penyesuaian diri dengan kebijakan etika serta keamanan yang berlaku.

Proses ini bersifat iteratif, mencakup tahapan penyempurnaan dan evaluasi berkelanjutan untuk memastikan model berfungsi dengan akurat, etis, dan relevan dalam konteks yang telah ditetapkan.

### Evaluation

Setelah model selesai disesuaikan dan melalui berbagai proses development, tahap berikutnya tentu saja adalah menguji performanya melalui tahap evaluasi. Tujuan utama evaluasi di sini bukan sekadar memastikan model dapat menghasilkan teks yang terdengar “bagus,” tetapi juga untuk menilai apakah hasilnya akurat, relevan, konsisten, dan aman sesuai kebutuhan pengguna.

Dalam konteks *Generative AI* Ops, evaluasi bersifat sistematis dan berkelanjutan. Kualitas model tidak bersifat statis sehingga perlu dipantau dan disesuaikan terus-menerus sepanjang siklus hidup aplikasi agar tetap memenuhi standar performa dan etika yang diharapkan.

Namun, proses evaluasi *Generative AI* dapat dikatakan unik dan inilah yang menjadi tantangan tersendiri. Timbul satu pertanyaan, mengapa Evaluasi Model Generatif Itu dianggap unik?

Berbeda dari model prediktif tradisional atau model diskriminatif seperti klasifikasi atau regresi yang outputnya kaku atau gambarannya sudah jelas di kepala kita, keluaran dari model generatif bersifat terbuka dan subjektif. Nah, hal ini membuat output dari *Generative AI* memiliki karakteristik seperti berikut.

*   Tidak selalu ada satu jawaban yang benar.
*   Kualitas output bergantung pada konteks, gaya bahasa, dan ekspektasi pengguna.
*   Evaluasi tidak bisa hanya mengandalkan metrik numerik, tetapi juga membutuhkan penilaian dan interpretasi manusia.

Oleh karena itu, proses evaluasi *Generative AI* umumnya menggabungkan tiga pendekatan utama, yaitu Metrik Angka, Human Feedback, dan juga LLM as a Judge.

*Nah*, ketiga pendekatan tersebut yang akan kita bedah satu per satu berikut ini.

#### Metrics Evaluation

Seperti halnya saat kita mengevaluasi model diskriminatif, evaluasi metrik berfokus pada skor numerik otomatis yang dihitung berdasarkan perbandingan antara output model dan *ground truth* atau “jawaban benar”.

Tujuannya tentu adalah mendapatkan ukuran performa yang objektif dalam definisi metrik tersebut, cepat, murah, dan mudah direproduksi. Pada model diskriminatif, kita biasanya mengenal metrik seperti accuracy, F1-score, RMSE, atau MAE. Namun, dalam *Generative AI*, pendekatan evaluasinya berbeda karena yang diukur bukan sekadar benar atau salah, melainkan kualitas, relevansi, dan kemiripan makna dari keluaran model.

Berikut beberapa metrik umum yang digunakan dalam evaluasi *Generative AI. *

*   **BLEU (Bilingual Evaluation Understudy)** 
    Awalnya dikembangkan untuk tugas penerjemahan. BLEU mengukur seberapa banyak N-gram atau sekuens kata pendek dalam output model yang cocok dengan N-gram dari referensi manusia. Fokusnya ada pada presisi, seberapa tepat kata-kata yang dihasilkan dibandingkan dengan referensi. Ini seperti memeriksa apakah bahan-bahan masakan yang Anda gunakan sesuai dengan resep aslinya.
*   **ROUGE (Recall-Oriented Understudy for Gisting Evaluation)** 
    Awalnya digunakan untuk mengevaluasi hasil ringkasan. ROUGE mirip dengan BLEU, tetapi ini adalah versi yang lebih fokus ke recall. Ia akan menilai kemunculan kata-kata penting dari referensi pada output model. Terdapat beberapa varian seperti ROUGE-N yang berdasarkan N-gram dan ROUGE-L yang berdasarkan *Longest Common Subsequence* atau urutan terpanjang yang sama. Jika dianalogikan, ini seperti memeriksa apakah poin-poin utama dari teks asli sudah tercakup dalam ringkasan yang Anda buat.
*   **METEOR** 
    Bisa dikatakan metrik ini dikembangkan sebagai penyempurnaan BLEU. Berbeda dari BLEU yang hanya mencocokkan kata secara tepat, METEOR juga mempertimbangkan sinonim, bentuk kata dasar (stemming), dan urutan kata. Dengan pendekatan ini, METEOR mampu menangkap kesamaan makna meskipun kata yang digunakan tidak identik.
*   **BERTScore** 
    Menggunakan embedding kontekstual dari model seperti BERT untuk mengukur kemiripan makna antara kata-kata dalam keluaran model dan referensi. Penilaian dilakukan menggunakan cosine similarity, sehingga BERTScore lebih mampu menangkap kesamaan semantik dibanding sekadar kesamaan kata.
    Analogi: Menilai apakah makna inti dari kalimat Anda sama dengan referensi, meskipun pilihan katanya berbeda.

#### Human Feedback

Bayangkan Anda punya AI yang jago menulis puisi atau menjawab pertanyaan esai. Bagaimana cara kita tahu apakah puisi itu indah atau jawaban esainya bermanfaat dan akurat?

Metrik otomatis seperti BLEU atau ROUGE mungkin bisa memberi skor, tetapi mereka sering kali gagal menangkap nuansa, kreativitas, kebenaran faktual, atau bahkan potensi bahaya dalam teks yang dihasilkan.

*Nah*, di sinilah *Human Feedback* masuk. Idenya sebenarnya sederhana, kita meminta manusia untuk menilai kualitas output yang dihasilkan model *Generative AI*.

Sebab pada akhirnya, pengguna aplikasi *Generative AI* ini nantinya adalah manusia, sehingga sentuhan penilaian manusia menjadi “kompas” penting untuk memahami seberapa baik model kita bekerja di dunia nyata, melampaui angka-angka metrik mentah.

Sekarang pertanyaannya adalah bagaimana cara melakukannya? Terdapat beberapa pendekatan yang dapat dilakukan.

*   **Direct Rating:** Memberikan skor pada output model menggunakan skala, misal 1-5, atau jempol naik dan turun.
*   **Comparison:** Menyajikan dua atau lebih output model untuk prompt yang sama, lalu meminta evaluator memilih mana yang lebih baik.
*   **Ranking:** Memberikan beberapa output dan meminta evaluator mengurutkannya dari yang terbaik hingga terburuk.
*   **Free-form Feedback:** Meminta evaluator menuliskan komentar kualitatif tentang apa yang bagus atau kurang dari output model.
*   **Koreksi atau Edit:** Meminta evaluator untuk mengedit output model agar menjadi lebih baik sehingga dapat dijadikan ground truth untuk iterasi berikutnya.

#### LLM as a Judge

Kita sudah memahami betapa pentingnya Human Feedback dalam menilai kualitas model generatif, terutama untuk aspek-aspek subjektif yang sulit diukur dengan metrik otomatis. Namun, pendekatan berbasis manusia memiliki kendala besar, yaitu biaya yang mahal, memakan waktu yang lama, dan sulit untuk dilakukan dalam skala besar.

Lalu muncul pertanyaan: bagaimana jika kita bisa memperoleh penilaian yang seakurat dan secerdas manusia, tetapi jauh lebih cepat dan efisien?

Di sinilah konsep LLM as a Judge hadir sebagai solusi. Ide dasarnya sederhana, yaitu menggunakan Large Language Model lain yang lebih canggih, seperti GPT-5, Gemini 2.5 Pro, atau Llama 4 Behemoth, untuk bertindak sebagai “juri” yang menilai hasil keluaran dari model yang sedang kita uji.

Pada dasarnya, sistematika dan aspek yang dinilai pada pendekatan ini mirip apa yang dilakukan pada pendekatan Human Feedback sebelumnya. Hanya saja, kita mengganti manusia dengan LLM yang sangat besar dan canggih.

Pendekatan ini tentu membuat proses evaluasi menjadi jauh lebih cepat dan konsisten dibandingkan penilaian manual. Meski bukan pengganti sepenuhnya untuk human feedback, metode LLM as a Judge semakin populer karena mampu mempercepat iterasi, menurunkan biaya, dan memberikan penilaian yang mendekati kualitas manusia.

### Deployment

Setelah model melewati tahap evaluasi, proses deployment bukan sekadar “menyalakan” model di server dan membuatnya dapat diakses oleh pengguna.

Pada tingkat konseptual atau penjelasan yang sedikit rumitnya, deployment adalah langkah strategis untuk menentukan bagaimana, di mana, dan dengan kondisi apa model akan melayani *real user*.

Tahap ini mencakup lebih dari sekadar implementasi teknis. Ia menuntut keseimbangan antara kinerja, keamanan, efisiensi biaya, serta pengalaman pengguna. Dengan kata lain, deployment adalah jembatan antara eksperimen dan penerapan nyata memastikan model benar-benar siap beroperasi di lingkungan produksi yang dinamis.

Oleh karena topik ini akan sangat luas dan bersifat teknis, bagian ini hanya menyajikan gambaran umum sebagai pengantar. Pembahasan mendalam, termasuk contoh kode hingga konfigurasi infrastruktur akan kita jelajahi secara komprehensif dalam modul Deployment yang terpisah setelah modul ini.

### Continuous Improvement

*Eits*, apakah Anda berpikir pekerjaan telah selesai setelah deployment? Sayangnya belum, karena ada satu tahap terakhir yang tidak kalah penting ini. Model-model AI, terutama yang berinteraksi dengan dunia nyata, cenderung mengalami penurunan performa seiring waktu atau menghadapi kasus-kasus baru yang belum pernah dilihat sebelumnya.

Lantas apa saja yang perlu dilakukan? Mari kita “ulik” bersama.

#### Monitoring atau Mengamati Kinerja Model di Lingkungan Nyata

Monitoring bukan hanya memeriksa apakah server berjalan dengan baik, tetapi memahami perilaku model dalam konteks penggunaan sebenarnya. Terdapat tiga lapisan utama yang perlu diperhatikan.

*   **Technical Metrics** 
    Meliputi indikator performa sistem seperti waktu inferensi (latency), kapasitas permintaan per detik (throughput), tingkat pemanfaatan GPU/CPU, tingkat kesalahan (error rate), serta penggunaan memori dan penyimpanan. Alat yang umum digunakan mencakup Prometheus, Grafana, ELK Stack, dan Datadog.
*   **Model Metrics** 
    Berfokus pada kualitas keluaran model. Beberapa indikator yang lazim dipantau antara lain relevansi jawaban, konsistensi gaya bahasa, tingkat halusinasi (jawaban yang tidak faktual), pelanggaran kebijakan konten, serta tanggapan pengguna seperti penilaian atau umpan balik langsung. Alat yang sering dimanfaatkan mencakup LangFuse, Trulens, MLflow, dan dasbor evaluasi Hugging Face.
*   **Business & User Metrics** 
    Mencakup pengukuran terhadap nilai bisnis dan kepuasan pengguna, misalnya tingkat retensi, konversi, peningkatan produktivitas, serta umpan balik kualitatif. Lapisan ini membantu menghubungkan performa teknis model dengan dampaknya terhadap organisasi.

#### Continuous Feedback

Setiap interaksi dengan pengguna merupakan data berharga untuk perbaikan model. Sistem yang dirancang dengan baik akan mampu mengumpulkan masukan secara otomatis, mendeteksi anomali, dan menandai keluaran yang memerlukan peninjauan manusia. Sebagai contoh, ketika pengguna menandai jawaban model sebagai kurang tepat, sistem dapat merekam konteks pertanyaan dan hasilnya ke dalam basis data umpan balik. Informasi ini kemudian diteruskan ke tim pengembang untuk analisis lebih lanjut dan dijadikan bahan penyempurnaan model pada siklus pelatihan berikutnya. Dengan demikian, sistem tidak hanya merespons kesalahan, tetapi juga belajar darinya.

#### Continuous Improvement

Data hasil monitoring dan umpan balik menjadi dasar bagi proses penyempurnaan berkelanjutan. Pendekatan ini dapat dilakukan melalui dua jalur utama.

*   **Refinement pada Prompt dan Kebijakan**
    Meliputi pembaruan instruksi sistem, penyempurnaan filter konten, penyesuaian gaya komunikasi, serta penerapan batasan yang lebih jelas agar model beroperasi sesuai domainnya. Pendekatan ini relatif cepat dan tidak membutuhkan pelatihan ulang.
*   **Retraining dan Re-Alignment Model**
    Melibatkan penggunaan data baru untuk memperbarui atau menyesuaikan model terhadap konteks yang terus berkembang, baik melalui fine-tuning, pembaruan RAG, maupun penyesuaian kebijakan. Langkah ini biasanya dilakukan secara berkala untuk menjaga relevansi model terhadap dinamika lingkungan dan kebutuhan pengguna.

#### Governance, Etika, dan Kepatuhan

Pemantauan pasca-deployment juga mencakup tanggung jawab etika dan tata kelola. Setiap versi model, dataset, dan parameter perlu terdokumentasi dengan baik. Audit bias dilakukan secara rutin untuk memastikan keadilan, sementara privasi dan keamanan data pengguna dijaga ketat.

Selain itu, keterlibatan manusia tetap diperlukan dalam pengambilan keputusan yang bersifat sensitif, seperti pada bidang hukum, kesehatan, atau keuangan. Pendekatan ini menegaskan bahwa model tidak hanya berfungsi dengan baik secara teknis, tetapi juga beroperasi secara bertanggung jawab dan selaras dengan nilai organisasi.

Baiklah *Coders*, kita telah sampai di penghujung jalan dari perjalanan panjang Modul 1 ini yang membawa kita memahami fondasi utama Large Language Model (LLM). Perjalanan panjang kita dimulai dari konsep dasar, sejarah, arsitektur modern seperti Transformer dan Mixture of Experts, hingga cara LLM berbicara melalui *autoregressive*. Kita bahkan telah membahas implementasi lanjutan seperti *Fine-Tuning*, *RAG*, dan *Agent*.

Di bagian akhir, bahkan kita menyinggung pula bagaimana seluruh siklus ini dikelola melalui kerangka *Generative AI Ops*. *Nah*, ngomong-ngomong soal RAG, sesuai janji kita sebelumnya, kita akan mencoba untuk membedah teknik Retrieval Augmented Generation tersebut di modul berikutnya.

Hanya saja, sebelum melangkah ke topik yang lebih kompleks tersebut, kita akan “transit” sejenak melalui aktivitas berikut ini.

> **Aktivitas: Drag and Order**
> Urutan yang tepat:
> 1. Menentukan Scope
> 2. Memilih Model
> 3. Melakukan Fine-tuning
> 4. Mengevaluasi Model
> 5. Deploy Model
> 6. Continuous Improvement

*Yup*, itulah yang akan menjadi “tempat transit” kita sebelum masuk ke Modul RAG yang lebih kompleks. *Nah*, pemahaman pada modul berikutnya ini akan menjadi senjata utama untuk kita membangun sistem RAG yang *powerful*. *So, prepare yourself Coders!*