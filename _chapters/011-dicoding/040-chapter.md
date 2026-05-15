---
slug: ch-06-04
title: "Fine Tuning LLM"
layout: chapter
---

# Fine Tuning LLM

## Pengantar Fine-Tuning LLM

Selamat datang kembali *Coders*. Bagaimana perjalanan panjang kalian di dalam dunia Retrieval Augmented Generation?

Semoga energi Anda sudah terisi kembali karena kali ini kita sudah berdiri di depan pintu dunia yang baru. *Yes*, dunia berikutnya yang menjadi solusi lain untuk menambah kepintaran LLM kita dalam menjawab pertanyaan. Jika sebelumnya kita memberikan “buku” ke LLM untuk menambahkan pengetahuan yang tidak ia miliki, sekarang kita akan mengotak-atik otak LLM secara langsung.

*Wait*, jika berurusan dengan otak LLM, apakah itu artinya kita akan melakukan training model LLM? Untuk menjawab pertanyaan tersebut, mari kita perhatikan kabar menarik berikut ini.

![dos-3f58885fedda67b30c9ea87ce58481fe20260429175426.png](https://assets.cdn.dicoding.com/original/academy/dos-3f58885fedda67b30c9ea87ce58481fe20260429175426.png)

Laporan terbaru dari [The Guardian](https://www.theguardian.com/technology/2025/dec/18/2025-ai-boom-huge-co2-emissions-use-water-research-finds) dan [Euronews](https://www.euronews.com/next/2025/12/20/ai-data-centres-could-have-a-carbon-footprint-that-matches-small-european-country-new-stud#:~:text=But%20it's%20hard%20to%20measure,for%20cooling%20has%20also%20soared.) mengungkapkan bahwa pada tahun 2025, emisi karbon dari operasional sistem AI secara keseluruhan, yang mencakup tahap training sekaligus penggunaan sehari-hari diperkirakan mencapai 32,6 hingga 79,7 juta ton CO2. Angka ini setara dengan seluruh emisi karbon kota New York atau negara kecil di Eropa seperti Norwegia.

Tidak hanya itu, sebuah laporan dari [MIT Technology Review](https://www.technologyreview.com/2024/10/28/1106316/ai-e-waste/) mengenai sebuah studi baru menunjukkan bahwa volume limbah elektronik yang berkaitan dengan AI bisa meningkat secara drastis hingga jutaan ton per tahun pada dekade berikutnya. Limbah elektronik tersebut wujudnya berupa server yang sudah usang, prosesor yang diganti, atau komponen seperti GPU dan baterai yang sering dipaksa bekerja keras terutama saat fase training model.

![dos-5bce91da087eae20d3cc05c656374cb620260429173957.png](https://assets.cdn.dicoding.com/original/academy/dos-5bce91da087eae20d3cc05c656374cb620260429173957.png)

Angka-angka dan informasi di atas tentunya bukanlah data statistik biasa karena jika ini terus terjadi tentu waktu untuk berpamitan dengan bumi akan semakin dekat. Jika Anda mengaitkan seluruh informasi di atas, Anda akan menemukan sebuah benang merah bernama “Training AI Model”.

*Yup*, jika kita ingat kembali dalam modul LLM di awal, kita pernah menyebutkan bahwa untuk membangun model seperti Llama 3 dengan 400 miliar parameter, membutuhkan daya listrik yang sanggup menghidupi 20.000 rumah di Indonesia selama satu tahun penuh.

Oleh karena itu, paradigma dari perkembangan AI dan LLM ini seharusnya sedang mengarah ke efisiensi *”bagaimana cara membuat model dengan cost seminimal mungkin?”*. Cara lainnya adalah memanfaatkan model yang sudah ada, salah satunya dengan pendekatan bernama **Fine-tuning**.

Berdasarkan [laporan OpenAI](https://platform.openai.com/docs/guides/rft-use-cases?accordance=review), kolaborasi dengan para ahli pajak di Accordance menunjukkan bahwa model yang di-fine-tune memberikan peningkatan performa hingga 38.89% dalam analisis pajak dibandingkan model dasar, bahkan mencapai level penalaran setara tenaga ahli profesional.

Bahkan, data dari [Together AI](https://www.together.ai/blog/fine-tune-small-open-source-llms-outperform-closed-models) mengungkapkan bahwa model *open-source* berukuran kecil atau SLM yang di-*fine-tune* secara tepat dapat mengungguli model *proprietary* raksasa hingga 60% pada tugas spesifik, tentunya dengan biaya operasional 10 hingga 100 kali lebih murah.

Apakah Anda sudah mulai tertarik dengan kelebihan yang ditawarkan oleh Fine-tuning? Mari kita langsung berkenalan dengannya di materi berikutnya.

***

## Berkenalan dengan Fine-tuning

Saat mulai mengetuk pintu dunia ini, Anda seharusnya sudah menguasai Prompt Engineering untuk memeras performa terbaik dari model dan sudah mengimplementasikan RAG untuk menyuntikkan konteks eksternal.

Dalam dunia fine-tuning, Anda tidak lagi berbicara tentang menambahkan komponen eksternal seperti vector database atau embedding model untuk mengekspansi *knowledge* LLM. Di sini, Anda akan berurusan dengan hal yang lebih dalam, yaitu perilaku fundamental model itu sendiri.

Sebelum menyelam lebih dalam ke fine-tuning, kita perlu tahu di mana sebenarnya posisi fine tuning dalam proses pengembangan LLM?

Baiklah mari kita jawab dengan membahas fase sebuah LLM itu dibangun. Coba Anda perhatikan ilustrasi di bawah ini.

![dos-9523984f11b7feac368dfb85e487168d20260501173528.gif](https://assets.cdn.dicoding.com/original/academy/dos-9523984f11b7feac368dfb85e487168d20260501173528.gif)

Seperti yang telah kita ketahui, LLM atau SLM itu adalah sebuah model bahasa yang dilatih dengan triliunan data yang berasa dari internet atau apa pun dalam sebuah proses yang dinamakan **Pre-training**. Produk dari pre-training ini disebut dengan **Pre-trained model**.

Sementara itu*, fine tuning* adalah *treatment* lanjutan yang kita lakukan pada Pre-trained model sehingga disebut sebagai tahap Post Training atau alignment. Meskipun pada dasarnya fine-tuning—khususnya Supervised Fine-tuning—ini bukanlah satu-satunya jalan menuju roma dalam melakukan *alignment* pada pre-trained model.

*   **Supervised Fine-tuning:** LLM akan disajikan pasangan data input dan output yang ideal, lalu memaksa model untuk meminimalkan loss pada contoh-contoh tersebut.
*   **Distillation:** Dalam skenario ini, kita menggunakan model raksasa yang sangat cerdas sebagai Teacher untuk melatih model yang jauh lebih kecil sebagai Student.
*   **Reinforcement Learning:** Jika SFT mengajarkan model “cara bicara”, RL mengajarkan model “nilai dan preferensi” dengan mengevaluasinya menggunakan Reward Model.

*Nah*, dalam materi ini, kita akan memulainya terlebih dahulu dari Supervised Fine-tuning. *Yup*, secara tidak langsung kita sudah membuka “peta harta karun” dalam perjalanan di dunia Fine-tuning ini.

### Supervised Fine-tuning

Anda seharusnya sudah tidak asing dengan kosakata “Supervised”. *Yup*, Anda mungkin pertama kali bertemu dan memahami makna dari kosakata tersebut saat mempelajari Machine Learning Pemula sebelumnya.

*Nah*, idenya masih sama pada Supervised Fine-tuning ini. Di sini, kita akan “mengadaptasi” atau menyesuaikan model yang sudah dilatih sebelumnya agar dapat bekerja dengan baik pada tugas yang spesifik dengan menggunakan data berlabel.

![dos-1cb19bb9f67baeef9beb0fbc15222ca920260330130438.png](https://assets.cdn.dicoding.com/original/academy/dos-1cb19bb9f67baeef9beb0fbc15222ca920260330130438.png)

Inti dari Supervised learning pada dasarnya masih kita bawa di sini, di mana kita menggunakan data berlabel untuk diberikan ke model. Lantas apa perbedaannya? Mari kita *highlight* perbedaannya.

*   **Supervised Learning:** Melatih model “kosong” atau dari kondisi awal yang belum memiliki pengetahuan dan memahami pola apa pun, hingga akhirnya mampu mempelajari pola dan menyelesaikan suatu tugas berdasarkan data yang diberikan.
*   **Supervised Fine-tuning:** Mengadaptasi pre-trained model yang “generalis” karena dilatih oleh beragam data dalam jumlah yang besar sebelumnya dengan melakukan tweak pada parameter di dalamnya agar relevan dengan tugas atau domain spesifik.

Supervised learning adalah tahap yang dilakukan saat masa *pre-trained model* dibangun melalui *pre-training*. Sehingga, jika dibayangkan, pre-training memberikan model kemampuan untuk dapat berbicara, semantara Supervised Fine-tuning atau SFT memberikan model kemampuan berinteraksi yang sesuai kebutuhan.

*Wait*, cara berinteraksi? Bukankah Pre-trained model atau LLM base ini sudah dilatih dengan triliunan token data, mengapa ia masih tidak bisa berinteraksi?

Untuk menjawabnya, perhatikan ilustrasi berikut ini.

![dos-793b1528f884bdcdcced0ce846a30c9420260429181123.gif](https://assets.cdn.dicoding.com/original/academy/dos-793b1528f884bdcdcced0ce846a30c9420260429181123.gif)

Seperti yang Anda lihat pada ilustrasi di atas, jawaban dari Base atau Pre-trained model benar-benar sesuai dengan “DNA” yang diharapkan saat dirinya dibuat, yaitu **next token prediction**. *Yup*, ia hanya mengerti cara untuk melanjutkan kata demi kata dan melengkapi kalimat, tetapi tidak tau cara “menjawab” dengan pertanyaan dengan struktur kalimat yang baik.

Anda dapat membayangkan Base Model seperti orang jenius yang belum pernah bersosialisasi. Jika Anda mengajukan pertanyaan apa pun, ia dapat menjawabnya karena tahu segalanya. Namun, cara ia menjawab belum tentu terstruktur, jelas, atau sesuai dengan yang kita harapkan.

Satu pertanyaan terjawab muncul pertanyaan berikutnya mengenai, *“Dari mana saya harus memulai untuk melakukan supervised fine-tuning? Apakah saya ambil Base Model yang masih mentah, atau Instruct Model yang sudah dipoles?”* 

Jawaban dari pertanyaan ini akan menentukan seberapa berat effort komputasi Anda dan seberapa bagus kualitas akhir model Anda. Mari kita bedah satu per satu.

#### Skenario Fine-tune dari Base Model

Dalam skenario ini, goal utamanya adalah mengubah base model menjadi instruct model. *Nah*, terdapat dua opsi kondisi model instruct yang membuat kita perlu memilih jalur ini.

*   **Mengubah Struktur Interaksi Fundamental** 
    Contoh paling mudahnya adalah jika Anda membangun model untuk tugas non-chat. Misalkan, Anda ingin model yang inputnya adalah data JSON kotor dan *output*-nya adalah SQL Query yang valid. Di sini, gaya bahasa “sopan” dan format tanya-jawab ala Instruct Model justru menjadi gangguan atau *noise*. Anda tidak butuh model yang menjawab *“Tentu, ini SQL yang Anda minta”*, Anda hanya butuh kode SQL-nya secara langsung.
*   **Menanamkan Gaya Bahasa atau Dialek Baru yang Spesifik** 
    Biasanya terjadi saat Anda ingin membuat model yang dapat berbicara dengan bahasa tertentu, misalnya bahasa daerah atau gaya bahasa abad ke-19. Apabila kita menggunakan Instruct Model untuk kasus ini sering kali dialek tersebut tidak berhasil diadopsi seutuhnya karena model tersebut sudah bias dengan gaya bicara asisten yang umum. Ya, “melawan” bias tersebut sering kali lebih susah daripada mengajarkannya dari nol.

#### Skenario Fine-tune dari Instruct Model

Di sisi satunya ini, kita punya Instruct Model. Ini adalah model yang sudah melalui tahap SFT atau bahkan mungkin sudah melewati tahap RLHF atau Reinforcement Learning from Human Feedback. Ia sudah mengerti instruksi, sudah mengerti struktur tanya jawab seperti asisten, dan sudah tahu cara menolak permintaan berbahaya. *Wait*, RLHF itu teknik seperti apa? Tenang saja, kita akan membedahnya di bagian lain.

*Alright,* kembali lagi ke skenario Instruct Model. *Nah*, Anda harus memilih jalur ini jika tujuannya adalah dua hal berikut.

*   **Spesialisasi Domain** 
    Katakanlah Anda ingin membuat asisten medis atau legal advisor. Anda tidak perlu mengajari model tersebut cara menyapa pasien atau cara menyusun kalimat tanya-jawab karena model sudah menguasai itu. Fokus Anda hanya “menyuntikkan” knowledge medis atau hukum yang spesifik melalui fine-tuning.
*   **Efisiensi** 
    Memulai dari Instruct Model jauh lebih efisien secara data. Anda mungkin hanya butuh ratusan atau ribuan sampel data berkualitas tinggi untuk mengubah style atau menambah knowledge spesifik, dibandingkan jutaan data yang mungkin dibutuhkan jika memulai dari Base Model.

*Nah*, jadi kuncinya ada pada “seberapa jauh” target behavior Anda dari perilaku standar asisten AI. Jika dekat atau mirip, gunakan Instruct Model. Jika Anda ingin merombak total cara model merespons, kembalilah ke Base Model.

*Alright*, sekarang kita sudah memahami konsep dasar fine-tuning dan gambaran di permukaannya. Mari kita arahkan anjungan kapal selam ini untuk menukik ke bawah, kita akan mengetuk pintu implementasi supervised fine-tuning dengan berkenalan dengan pendekatan yang bisa dilakukan.

Mari kita mulai dengan pendekatan paling “purba”, yaitu Full Fine-tuning.

### Full fine-tuning

Sesuai namanya, dalam Full Fine-tuning kita benar-benar memperbarui seluruh parameter model. Jika model Anda memiliki 7 miliar parameter, itulah yang akan kita kalkulasi gradiennya dan kita update bobotnya di setiap langkah pelatihan.

Apabila konsepnya seperti itu, apakah berarti Full Fine-tuning ini dapat diartikan kita melakukan tahap pre-training tambahan pada model?

Secara konsep kasar, bisa dikatakan iya sebab di sini kita benar-benar meng-*update* seluruh parameter model. Coba Anda perhatikan ilustrasi di bawah ini untuk dapat memahaminya.

![dos-78264f49f14e1b85da82859b7e687eca20260429181618.gif](https://assets.cdn.dicoding.com/original/academy/dos-78264f49f14e1b85da82859b7e687eca20260429181618.gif)

Mari kita lihat detail kecil yang menceritakan kisah besar tersebut.

*   **Sebelum Fine-tuning:**
    Anda melihat lingkaran kuning (neuron) dengan angka tertentu, misal 0.882 di pojok kiri atas. Angka-angka ini adalah "ingatan" atau pengetahuan bawaan si model.
*   **Sesudah Fine-tuning:**
    Perhatikan angka 0.882 tadi berubah menjadi 0.132. Angka 0.122 berubah menjadi 0.875.

Intinya dalam Full fine-tuning, tidak ada layer tambahan apa pun sehingga strukturnya sama persis, tetapi “isi kepalanya” atau nilai parameternya yang berubah total. Hal ini tentunya memberikan fleksibilitas maksimal karena kita memiliki kendali penuh untuk mengubah perilaku model hingga ke akar-akarnya. Contohnya seperti yang kita lakukan di-*fine-tuning* base model, di mana kita ingin mengubah model bahasa Inggris menjadi fasih berbicara bahasa Jawa.

Namun, kebebasan ini datang dengan harga yang sangat mahal. Tantangan terbesarnya ada pada VRAM atau memori GPU. Dalam ilustrasi di atas kita hanya mereplika gambaran super mini dari sebuah model dengan 10 lingkaran kuning atau parameter, sekarang bayangkan jika lingkaran kuning itu ada 7 miliar.

Mengapa menjadi masalah? Ada dua alasan mengapa jika lingkaran kuningnya ada miliaran akan menjadi masalah.

*   **Daya Komputasi:** Untuk mengubah setiap angka dari miliaran lingkaran itu secara bersamaan, Anda membutuhkan daya komputasi (GPU) yang monster.
*   **Memori:** Membutuhkan kapasitas memori atau VRAM sekitar 4x hingga 6x lipat dari ukuran asli modelnya. Jika memori penuh, akan langsung *crash Out of Memory*.

*Yup*, saat melakukan Full-fine tuning kita butuh space memori VRAM 6x lipat dari besar model aslinya. Namun, apakah statement di atas valid dan bukan digunakan untuk menakut-nakuti kita?

Bagaimana jika kita buktikan saja “konspirasi” tersebut melalui perhitungan sederhana agar lebih menarik? *Let’s Calculate!*

#### Matematika di Balik VRAM

Statement sebelumnya mengenai ukuran model aslinya dalam Gigabytes mungkin masih menjadi miskonsepsi yang umum terjadi. Biasanya kita akan mengira model dengan 7B (Billion) parameter itu memiliki ukuran 7 GB.

Untuk membantah hal tersebut, kita perlu aware terlebih dahulu bahwa LLM itu memiliki dua versi “ukuran”, yaitu *Full Precision* dan *Half Precision*. Wait, apa itu *Precision*?

Seperti yang kita ketahui, komputer itu hanya memahami angka, semua hal yang diproses oleh komputer akan diubah menjadi angka atau nilai numerik. Nilai numerik tersebut dapat berbentuk float atau angka desimal. *Nah*, komputer menyimpan angka desimal tersebut dalam format yang disebut Precision.

![dos-f71d2276fb7bfb663a6121f0af67b61c20260429181823.png](https://assets.cdn.dicoding.com/original/academy/dos-f71d2276fb7bfb663a6121f0af67b61c20260429181823.png)

Semakin presisi atau dengan kata lain semakin banyak angka di belakang koma, semakin besar memori yang dibutuhkan. Anda dapat membayangkan precision ini seperti resolusi atau kualitas dalam video seperti 4K, 2K, hingga 360p. *Nah*, ada dua variasi opsi untuk menyimpan model.

*   **Full Precision (FP32):** Ini adalah versi 4K dari sebuah model, akurasinya sangat tinggi karena angkanya sangat detail. Sehingga, ukuran setiap angka parameternya adalah 4 bytes.
*   **Half Precision (FP16/BF16):** Sesuai namanya, ini adalah versi 2K-nya dan versi hemat dari Full Precision. Ukuran dari setiap parameternya pun terpangkas menjadi setengah lebih kecil, 2 bytes per angka.

*Eits*, Anda tidak perlu memusingkan terlebih dahulu makna lebih dalam kedua hal tersebut dan maksud dari FP atau Floating Point. Kita akan membahasnya di bagian selanjutnya PEFT. Sekarang, kita akan coba untuk melakukan peramalan ukuran model saat di-*fine-tuning.*

*Alright,* mari kita cari model SLM untuk kita jadikan “kelinci percobaan” dalam perhitungan besar memori ini. Misalnya, kita menggunakan model dari perusahaan Mark Zuckerberg, yaitu [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) atau Llama 3.1 8B versi instruct model.

![dos-bdc3f52d783692ede609523a414aca2820260429182126.gif](https://assets.cdn.dicoding.com/original/academy/dos-bdc3f52d783692ede609523a414aca2820260429182126.gif)

Seperti yang dapat Anda lihat pada ilustrasi di atas, meta-llama/Llama-3.1-8B-Instruct memiliki parameter sebanyak 8B atau 8 miliar dengan format BFP16 atau Half Precision.

Oleh karena menggunakan BFP16, artinya setiap parameter di dalam model tersebut memiliki ukuran sebesar 2 bytes. Sehingga, perhitungan ukuran model Llama-3.1-8B-Instruct tersebut akan seperti berikut.

![dos-e4496814f276bb43997f276042c57ddf20260429182305.png](https://assets.cdn.dicoding.com/original/academy/dos-e4496814f276bb43997f276042c57ddf20260429182305.png)

*Yup*, artinya untuk modelnya saja, kita membutuhkan VRAM atau memori sebesar 16 GB. Lantas, apa artinya kita membutuhkan 16 GB untuk fine-tune Llama-3.1-8B?

Jika Anda hanya ingin melakukan inference atau chatting menggunakan model tersebut, VRAM 16 GB - 24 GB seperti RTX 3090 atau 4090 sudah cukup. Namun, ceritanya berbeda saat kita ingin melakukan fine-tuning. Jika Anda hanya punya VRAM sebesar itu, bisa dipastikan saat iterasi pertama fine-tuning akan mengalami *Out of Memory*.

Mengapa demikian?

Perlu Anda ketahui bahwa saat pelatihan atau fine-tuning, memori GPU tidak hanya menyimpan bobot model. Ia juga harus menyimpan *optimizer states*, *gradients*, dan *activations* yang membuat ukurannya bisa 4x hingga 8x lipat dari ukuran model aslinya seperti yang di-*mention* sebelumnya.

*Alright*, mari kita *breakdown* dan agar lebih rapi kita akan membedah perhitungan memori setiap komponennya untuk satu parameter. Oh iya, karena sudah mengetahui bahwa ukuran setiap parameternya dalam model BFP 16 adalah 2 bytes, kita akan skip komponen model dan langsung menuju Gradients.

*   **Gradients** 
    Gradien menyimpan informasi perubahan untuk setiap parameter selama backpropagation. *Gradient*-lah menunjukkan arah dan besaran perubahan yang harus dilakukan pada setiap parameter. *Nah*, setiap 1 parameter punya 1 nilai gradient sehingga ukurannya sama dengan bobot parameter model, yaitu **2 bytes**.
*   **Optimizer States** 
    *Nah*, ini bagian *tricky*-nya. Meskipun model kita BFP16, Optimizer AdamW sebagai standar industri membutuhkan Full Precision (FP32) untuk menjaga kestabilan dan akurasinya. Kita akan membahasnya lebih dalam nantinya saat menyelami FP16. *So in total*, optimizer membutuhkan 3 data berikut untuk setiap parameternya.
    *   **FP32 Master Weights**
        Copy dari bobot model, tapi dalam resolusi tinggi sehingga setiap parameternya butuh 4 bytes.
    *   **Momentum (Mean)**
        Rata-rata pergerakan gradient “masa lalu” dan ukurannya untuk setiap parameter adalah 4 bytes.
    *   **Variance**
        Mencatat seberapa stabil atau fluktuatif perubahan gradien. Biaya dari variance ini adalah 4 bytes untuk setiap parameter model.

Berarti secara total hanya untuk Optimizer, kita butuh **12 bytes** untuk setiap parameter dalam model.

*Alright*, jika kita totalkan seluruh komponen tersebut mulai dari Parameter Model itu sendiri, Gradients dan Optimizer States maka akan butuh 16 Bytes untuk setiap parameter. Karena model kita berukuran 8 Miliar parameter, Anda dapat lihat hasilnya pada ilustrasi di bawah ini.

![dos-042cc4c228581a771196aa0707a7733b20260429182708.gif](https://assets.cdn.dicoding.com/original/academy/dos-042cc4c228581a771196aa0707a7733b20260429182708.gif)

Sebentar, sepertinya ada yang terlewat. Bukankah kita belum memasukan Activation dalam perhitungan di atas?

*Yup*, kita memang tidak memasukan perhitungan ukuran untuk Activation, sebab ukuran dari Activation itu sangat dinamis. Besar memori yang kita hitung di atas adalah Static Memory atau ukuran tetap yang tidak berubah-ubah. Besar ukuran Activation bergantung pada Batch Size dan Sequence Length atau panjang Context Window yang digunakan saat melakukan Fine-tuning.

Konfigurasi Hyperparameter seperti itu tentunya akan tergantung pada praktik di lapangan sehingga untuk saat ini kita akan menggunakan 128 GB di atas. Anggaplah ini sebagai “tiket masuk” minimum atau kita dapat berkata *“sebesar inilah ruang kosong yang harus tersedia bahkan sebelum model mulai belajar satu kalimat pun."* 

Perlu Anda ketahui, GPU paling “monster” pada akhir tahun 2025, yaitu NVIDIA Blackwell B200 itu memiliki 192GB VRAM. Jika Anda memiliki akses ke sana, kabar baik untuk Anda karena bisa memuat seluruh model Llama 3.1 8B beserta optimizer hanya dengan satu keping chip GPU saja. Kabar buruknya adalah harga GPU “monster” tersebut berkisar pada 30.000 hingga 40.000 USD atau sekitar Rp508.680.000 hingga Rp678.240.000 dengan kurs Rp16.965.

Sekarang pertanyaannya adalah, bagaimana perusahaan-perusahaan seperti OpenAI, Google, dan Anthropic mampu melatih model yang ratusan kali lebih besar dari yang kita bahas ini? Bahkan GPU paling mutakhir pun sudah menggunakan sekitar 66% kapasitas memorinya hanya untuk melatih model kecil seperti Llama 3.1 8B.

Oleh karena itulah, melakukan Full Fine-tuning pada model besar hampir mustahil dilakukan di satu GPU saja (bahkan sekelas A100 sekalipun).

Di sinilah kita membutuhkan strategi distribusi beban atau Distributed Training. Ada dua mazhab utama dalam menyiasati keterbatasan infrastruktur ini.

#### Data Parallelism

Kita mulai dengan analogi sederhana, bayangkan Anda memiliki sebuah buku tebal berisi 1 juta soal matematika (dataset) yang harus dikerjakan secepat mungkin. Jika Anda mengerjakannya sendirian (Single GPU), mungkin butuh waktu setahun untuk menyelesaikannya. Tentu ini sama sekali tidak efisien dan Anda butuh bantuan.

Solusi paling logis yang mungkin terpikirkan adalah kita menggunakan *call a friends*. Katakanlah Anda memanggil 3 teman tambahan sehingga total ada 4 orang (4 GPU) yang mengerjakan soal tersebut. *Nah*, dalam strategi Data Parallelism, yang kita lakukan bukanlah membagi-bagi otak Anda. Sebaliknya, kita melakukan replikasi penuh.

![dos-14f1cf133a71f36b760f22f0f26c4aa020260429182830.gif](https://assets.cdn.dicoding.com/original/academy/dos-14f1cf133a71f36b760f22f0f26c4aa020260429182830.gif)

Mari kita bedah satu per satu peta jalan dalam melakukan Data Parallelism ini.

*   **Replikasi** 
    Syarat pertama dan mutlak adalah setiap GPU harus memiliki VRAM yang cukup untuk memuat modelnya. Sebab, kita akan memberikan salinan Model Weights yang sama persis ke setiap GPU. Sehingga, baik itu GPU 1, GPU 2, GPU 3, dan GPU 4, semuanya memegang model Llama 3.1 yang utuh di memori mereka masing-masing.
*   **Distribusi Load “Kerja”** 
    Alih-alih satu GPU mengerjakan seluruh dataset berurutan, kita membagi “tumpukan” datanya (Micro-batches).
    *   GPU 1 mengerjakan data 1–100.
    *   GPU 2 mengerjakan data 101–200.
    *   GPU 3 mengerjakan data 201–300.
    *   GPU 4 mengerjakan data 301–400.

    Keempat GPU tersebut nantinya akan bekerja secara paralel. Oleh karena setiap GPU memegang model yang utuh, mereka tidak perlu saling “bertanya” saat sedang melakukan komputasi. GPU 1 tidak peduli apa yang sedang dikerjakan GPU 4. Mereka fokus melakukan Forward Pass dan Backward Pass.

*   **Sinkronisasi Menggunakan All-Reduce** 
    Setelah keempat GPU selesai melakukan satu epoch atau iterasi dan perhitungan Gradients, mereka tidak langsung memperbarui Model Weight mereka sendiri-sendiri. Keempat GPU tersebut akan melakukan sebuah "Rapat Kilat" yang disebut All-Reduce yang terdiri dari dua operasi matematika atau komputasi.
    *   **Reduce:** Mengambil nilai dari banyak sumber dan menguranginya menjadi satu nilai ringkas yang biasanya lewat penjumlahan atau rata-rata.
    *   **All:** Memastikan hasil ringkas tersebut dikirimkan kembali ke semua sumber.
*   **Update Secara Serentak** 
    Setelah mendapatkan nilai rata-rata gradien dari All Reduce, barulah keempat GPU memperbarui bobot model (Optimizer Step) secara bersamaan dengan nilai yang sama persis. Setelah satu iterasi selesai, GPU 1 sampai GPU 4 kembali memiliki Model Weight yang 100% identik dan siap untuk memproses tumpukan data berikutnya.

*Yup*, kekuatan utama dari Data Parallelism ada pada kecepatan dengan skalabilitas linear. Jika Anda pakai 2 GPU, kecepatannya hampir 2 kali lipat dan begitu seterusnya. Namun, strategi ini punya satu syarat mutlak yang kita *mention* sebelumnya, “Satu buah model harus muat utuh di dalam satu GPU”.

Jika model Llama 3 butuh 128GB dan di sisi lain GPU Anda hanya memiliki memori 80GB, Anda tidak bisa melakukan Data Parallelism meski ada 100 GPU identik dengan spesifikasi tersebut. Anda tidak bisa meng-*copy* Model Weight ke setiap orang, kalau “ruang penyimpanan” mereka saja tidak muat menampungnya.

Lalu, bagaimana solusinya? Di situlah kita terpaksa menggunakan strategi kedua, yaitu Model Parallelism yang akan kita bahas selanjutnya.

#### Model Parallelism

Saat ukuran model terlalu besar, tentu saja langkah satu-satunya adalah memecah model tersebut menjadi beberapa bagian agar muat dimasukan ke dalam “ruang penyimpanan”. Inilah salah satu teknik yang memungkinkan model dengan triliunan parameter dapat tercipta dengan keterbatasan memori GPU saat ini.

*Nah*, terdapat dua cara untuk memecah model menjadi potongan kecil, secara vertikal dan secara horizontal. *Yup*, seperti kita memotong “kue lapis”, tetapi tenang saja tidak akan sesederhana itu tentunya, mari kita bahas.

##### Pipeline Parallelism “Si Vertikal”

*Alright*, untuk membahas Pipeline Parallelism, Anda harus memutar sedikit mindset mengenai komputasi paralel menjadi komputasi berantai. *Yup*, karena konsep itulah yang digunakan dalam Pipeline Parallelism.

Llama 3.1 8B kita misalnya, memiliki total 32 layers transformer. Karena 4 GPU memiliki VRAM yang sangat terbatas, kita mempartisi model tersebut secara vertikal berdasarkan urutan layer-nya menjadi seperti berikut ini.

*   **GPU 1:** Memegang Layer 1 sampai 8
*   **GPU 2:** Memegang Layer 9 sampai 16
*   **GPU 3:** Memegang Layer 17 sampai 24
*   **GPU 4:** Memegang Layer 25 sampai 32

Secara fisik, data input masuk ke GPU 0, diproses, lalu hasil aktivasinya dikirim lewat kabel ke GPU 1, dan seterusnya hingga GPU 3 mengeluarkan prediksi. Kemudian, saat *backward pass* (training), gradien bergerak mundur dari GPU 3 kembali ke GPU 0.

Agar lebih terbayang di kepala kita, mari perhatikan ilustrasi di bawah ini

![dos-437a35bd412dcaf7a7d2b0dd087edbba20260429183611.gif](https://assets.cdn.dicoding.com/original/academy/dos-437a35bd412dcaf7a7d2b0dd087edbba20260429183611.gif)

Terlihat sederhana? *Wait*, mengapa kita tidak mengirimkan seluruh data langsung dan malah menggunakan *micro-batch* seperti itu?

Jika menerapkan cara “naif” seperti itu atau Batch processing biasa, kita akan menciptakan bencana efisiensi yang disebut “Pipeline Bubble”.

Bayangkan kita memasukkan satu batch besar data ke GPU 1. Saat GPU 1 sedang sibuk menghitung (Forward Pass), apa yang dilakukan GPU 2, 3, dan 4? Mereka idle atau “tidur”. Setelah GPU 1 selesai, ia melempar data ke GPU 2 dan ia bekerja. Apa yang dilakukan GPU 1? Dia tidak bisa lanjut kerja karena dia harus menunggu gradien balik dari GPU 3. Jadi GPU 1 sekarang ikut “tidur”.

Dalam skenario ini, pada satu titik waktu, hanya ada 1 GPU yang bekerja sementara 3 lainnya menganggur. Ini berarti Anda membayar 4 GPU tapi hanya mendapatkan performa setara 1 GPU.

Oleh karena itu, alih-alih mengirim satu *Global Batch* besar sekaligus, kita memecahnya menjadi paket-paket kecil yang dinamakan *Micro-batch*. Misalkan Global Batch kita 1024 data. Kita pecah menjadi 4 micro-batch masing-masing 256 data.

*   **Langkah 1:** GPU 1 memproses Micro-batch 1, lalu langsung melemparnya ke GPU 2.
*   **Langkah 2:** Saat GPU 2 sedang mengerjakan Micro-batch 1, GPU 1 tidak diam. Dia langsung menarik Micro-batch 2 untuk diproses.
*   **Langkah 3:** Sistem ini berlanjut sehingga GPU 1, 2, 3, dan 4 akhirnya bekerja secara bersamaan menangani potongan *micro-batch* yang berbeda-beda.

Strategi ini secara drastis mengurangi ukuran “Bubble” atau waktu menganggur tadi. Semakin banyak micro-batch yang kita buat, semakin padat jadwal kerja GPU, dan semakin tinggi efisiensinya.

##### Tensor Parallelism “Horizontal”

Berbeda dengan Pipeline Parallelism yang memecah model secara vertikal menjadi per layer, Tensor Parallelism membelah model secara horizontal. Artinya, setiap layer dipecah dan didistribusikan ke beberapa GPU sekaligus.

Dalam Tensor Parallelism, kita tidak menunggu giliran Model Weight selesai diproses oleh GPU sebelumnya. Kita membelah matriks Model Weight tersebut. *Nah*, ada dua strategi utama pembelahan yang terjadi di dalam blok Transformer, yaitu Column Parallelism dan Row Parallelism.

![dos-75b3d20334d6731c4e7c95f9fc98954020260429183752.gif](https://assets.cdn.dicoding.com/original/academy/dos-75b3d20334d6731c4e7c95f9fc98954020260429183752.gif)

**Column Parallelism** 

Mari kita ambil contoh pada lapisan Feed Forward Network (MLP) yang ada di dalam Transformer. Kita membelah matriks Model Weight secara vertikal atau berdasarkan kolom.

*   GPU A memegang setengah kolom pertama dari matriks Model Weight.
*   GPU B memegang setengah kolom sisanya.

Saat input X masuk, kita melakukan copy input tersebut ke kedua GPU. Setiap GPU kemudian melakukan perkalian matriks secara mandiri. Sehingga, GPU A menghasilkan setengah bagian vektor output, dan GPU B menghasilkan setengah bagian lainnya. Karena bagian mereka berbeda secara independen, kita tidak perlu melakukan penjumlahan. Kita cukup menggabungkan hasilnya di akhir.

**Row Parallelism** 

Namun, cerita berbeda terjadi pada lapisan kedua di blok MLP. Di sini kita membelah matriks bobot secara horizontal (berdasarkan baris). Untuk melakukan perkalian matriks yang valid secara matematis, kita juga harus membelah input X-nya.

*   GPU A menghitung bagian atas matriks.
*   GPU B menghitung bagian bawah matriks.

Masalah muncul di sini. Hasil keluaran dari GPU A dan GPU B hanyalah parsial. Masing-masing hanya memegang sepotong nilai yang belum lengkap. Untuk mendapatkan nilai output yang sebenarnya, kedua hasil dari GPU A dan GPU B harus dijumlahkan menggunakan teknik yang sudah kita temui sebelumnya, yaitu All-Reduce.

Sayangnya, justru operasi All-Reduce inilah yang menjadi pedang bermata dua bagi Tensor Parallelism.

Dalam operasi All-Reduce, setiap GPU harus “berbicara” dengan semua GPU lainnya untuk saling menukarkan hasil hitungan mereka masing-masing, lalu menjumlahkan dan menyinkronkan hasilnya kembali. Anda perlu ingat, ini bukan terjadi satu kali per training step, melainkan terjadi di setiap layer, pada setiap forward pass, dan setiap backward pass seperti yang kita gambarkan dalam ilustrasi Data Parallelism sebelumnya.

Jika Anda memiliki model Llama-3 dengan 80 layer, berarti ada ratusan kali sinkronisasi “*stop-and-wait*” yang terjadi dalam hitungan milidetik.

Karena intensitas komunikasi yang sangat “cerewet” ini, Tensor Parallelism sangat “haram” dilakukan antar-server yang hanya terhubung kabel LAN atau Ethernet biasa. Latensinya terlalu tinggi. GPU Anda akan menghabiskan 90% waktu untuk menunggu data lewat kabel dan hanya 10% waktu untuk menghitung.

Tensor Parallelism mewajibkan penggunaan interkoneksi super cepat seperti NVLink dan NVSwitch yang memiliki bandwidth 600 GB/s hingga 900 GB/s. Untuk memfasilitasi kecepatan "gila" ini, GPU tidak bisa dipisah-pisah dalam *casing* yang berbeda. Mereka harus duduk bersama dalam satu “meja makan” atau board yang sama.

Seluruh komponen tersebut disatukan dalam sebuah baseboard khusus. Contohnya adalah NVIDIA HGX yang mengintegrasikan GPU dan switch dalam jarak fisik yang sangat dekat, seperti yang diperlihatkan pada ilustrasi di bawah ini.

![dos-2656a3301fd882fbdb324e5c2acb1a0620260429183912.png](https://assets.cdn.dicoding.com/original/academy/dos-2656a3301fd882fbdb324e5c2acb1a0620260429183912.png)

Lihat 6 chip kecil NVSwitches dan juga NVLink Interconnect di tengah baseboard tersebut? Ia bertugas sebagai “jembatan tol” data antar 8 GPU A100 di atas tanpa harus keluar server terlebih dahulu. Inilah alasan Tensor Parallelism bisa berjalan dengan cepat dan lancar. Jika chip ini tidak ada atau bahkan diganti kabel LAN biasa, performa AI training kita akan sangat lambat karena GPU hanya akan bengong menunggu data.

Sementara NVLink Connector atau benda logam yang berdiri tegak di belakang seperti colokan (port) itu perannya akan terasa saat Anda membutuhkan lebih dari satu baseboard atau server seperti di atas. Ia yang akan menjadi pintu masuk dari “jalan tol” komunikasi antara baseboard satu dengan baseboard lainnya. Tentunya “jalan tol” yang kita gunakan bukanlah kabel LAN biasa, melainkan kabel khusus NVIDIA LinkX.

*Alright*, agar kita tidak menyimpang terlalu jauh dan membongkar terlalu dalam resep rahasia dari mesin NVIDIA tersebut, mari kita kembali fokus ke Fine-tuning LLM. Jika Anda masih penasaran dengan baseboard NVIDIA HGX tersebut, silakan gunakan tautan berikut ini: [NVIDIA HGX](https://www.nvidia.com/en-us/data-center/hgx/).

Sebelum lanjut kembali, terdapat satu aktivitas interaktif di bawah ini yang akan membantu Anda untuk me-recall konsep dari beberapa teknik Distributed Training yang sebelumnya kita pelajari.

<iframe src="https://academy-sandpack.dicoding.dev/activities/flashcards?data=eyJjYXJkcyI6W3siaWQiOiJjYXJkLTUxOC4xMDAwMDAwMDE0OTAxIiwiZnJvbnQiOiI8c3BhbiBpZD1cImRvY3MtaW50ZXJuYWwtZ3VpZC04MTBmYTM4Yy03ZmZmLTMzMWUtNGRlZS1lZGFhM2I5ZTFjZGVcIj48c3BhbiBzdHlsZT1cImZvbnQtc2l6ZTogMTFwdDsgZm9udC1mYW1pbHk6IEFyaWFsLCBzYW5zLXNlcmlmOyBiYWNrZ3JvdW5kLWNvbG9yOiB0cmFuc3BhcmVudDsgZm9udC12YXJpYW50OiBub3JtYWw7IHZlcnRpY2FsLWFsaWduOiBiYXNlbGluZTsgd2hpdGUtc3BhY2U6IHByZS13cmFwO1wiPlRla25payBkaXN0cmlidXNpIHBlbGF0aWhhbiBkaSBtYW5hIHNhbGluYW4gbW9kZWwgeWFuZyBzYW1hIHBlcnNpcyBkaWxldGFra2FuIGRpIGJlYmVyYXBhIEdQVSwgbGFsdSBzZXRpYXAgR1BVIG1lbXByb3NlcyBwb3RvbmdhbiBkYXRhICg8L3NwYW4%2BPHNwYW4gc3R5bGU9XCJmb250LXNpemU6IDExcHQ7IGZvbnQtZmFtaWx5OiBBcmlhbCwgc2Fucy1zZXJpZjsgYmFja2dyb3VuZC1jb2xvcjogdHJhbnNwYXJlbnQ7IGZvbnQtc3R5bGU6IGl0YWxpYzsgZm9udC12YXJpYW50OiBub3JtYWw7IHZlcnRpY2FsLWFsaWduOiBiYXNlbGluZTsgd2hpdGUtc3BhY2U6IHByZS13cmFwO1wiPmJhdGNoPC9zcGFuPjxzcGFuIHN0eWxlPVwiZm9udC1zaXplOiAxMXB0OyBmb250LWZhbWlseTogQXJpYWwsIHNhbnMtc2VyaWY7IGJhY2tncm91bmQtY29sb3I6IHRyYW5zcGFyZW50OyBmb250LXZhcmlhbnQ6IG5vcm1hbDsgdmVydGljYWwtYWxpZ246IGJhc2VsaW5lOyB3aGl0ZS1zcGFjZTogcHJlLXdyYXA7XCI%2BKSB5YW5nIGJlcmJlZGEgc2VjYXJhIGJlcnNhbWFhbjwvc3Bhbj48L3NwYW4%2BIiwiYmFjayI6IjxzcGFuIGlkPVwiZG9jcy1pbnRlcm5hbC1ndWlkLTBkY2IxY2JhLTdmZmYtNDgwYS03NzVjLTZlZjg3NmZjOTBmNFwiPjxzcGFuIHN0eWxlPVwiZm9udC1zaXplOiAxMXB0OyBmb250LWZhbWlseTogQXJpYWwsIHNhbnMtc2VyaWY7IGJhY2tncm91bmQtY29sb3I6IHRyYW5zcGFyZW50OyBmb250LXZhcmlhbnQ6IG5vcm1hbDsgdmVydGljYWwtYWxpZ246IGJhc2VsaW5lOyB3aGl0ZS1zcGFjZTogcHJlLXdyYXA7XCI%2BRGF0YSBQYXJhbGxlbGlzbTwvc3Bhbj48L3NwYW4%2BIn0seyJpZCI6ImNhcmQtMTI5MzQuMzAwMDAwMDAwNzQ1IiwiZnJvbnQiOiI8c3BhbiBpZD1cImRvY3MtaW50ZXJuYWwtZ3VpZC1mZjBiZjExZS03ZmZmLTYyOWQtNWNmNC1mYWQ3NGQzNWRhYjdcIj48c3BhbiBzdHlsZT1cImZvbnQtc2l6ZTogMTFwdDsgZm9udC1mYW1pbHk6IEFyaWFsLCBzYW5zLXNlcmlmOyBiYWNrZ3JvdW5kLWNvbG9yOiB0cmFuc3BhcmVudDsgZm9udC12YXJpYW50OiBub3JtYWw7IHZlcnRpY2FsLWFsaWduOiBiYXNlbGluZTsgd2hpdGUtc3BhY2U6IHByZS13cmFwO1wiPlN0cmF0ZWdpIG1lbWJhZ2kgYXJzaXRla3R1ciBtb2RlbCBzZWNhcmEgdmVydGlrYWwsIGRpIG1hbmEgbGFwaXNhbiAoPC9zcGFuPjxzcGFuIHN0eWxlPVwiZm9udC1zaXplOiAxMXB0OyBmb250LWZhbWlseTogQXJpYWwsIHNhbnMtc2VyaWY7IGJhY2tncm91bmQtY29sb3I6IHRyYW5zcGFyZW50OyBmb250LXN0eWxlOiBpdGFsaWM7IGZvbnQtdmFyaWFudDogbm9ybWFsOyB2ZXJ0aWNhbC1hbGlnbjogYmFzZWxpbmU7IHdoaXRlLXNwYWNlOiBwcmUtd3JhcDtcIj5sYXllcnM8L3NwYW4%2BPHNwYW4gc3R5bGU9XCJmb250LXNpemU6IDExcHQ7IGZvbnQtZmFtaWx5OiBBcmlhbCwgc2Fucy1zZXJpZjsgYmFja2dyb3VuZC1jb2xvcjogdHJhbnNwYXJlbnQ7IGZvbnQtdmFyaWFudDogbm9ybWFsOyB2ZXJ0aWNhbC1hbGlnbjogYmFzZWxpbmU7IHdoaXRlLXNwYWNlOiBwcmUtd3JhcDtcIj4pIGF3YWwgbW9kZWwgZGlsZXRha2thbiBkaSBHUFUgcGVydGFtYSwgbGFwaXNhbiB0ZW5nYWggZGkgR1BVIGtlZHVhLCBkYW4gc2V0ZXJ1c255YTwvc3Bhbj48L3NwYW4%2BIiwiYmFjayI6IjxzcGFuIGlkPVwiZG9jcy1pbnRlcm5hbC1ndWlkLTA0ODk3ZDQzLTdmZmYtN2JjMS05MTdiLWJkMGFmZjJmOWRkZVwiPjxzcGFuIHN0eWxlPVwiZm9udC1zaXplOiAxMXB0OyBmb250LWZhbWlseTogQXJpYWwsIHNhbnMtc2VyaWY7IGJhY2tncm91bmQtY29sb3I6IHRyYW5zcGFyZW50OyBmb250LXZhcmlhbnQ6IG5vcm1hbDsgdmVydGljYWwtYWxpZ246IGJhc2VsaW5lOyB3aGl0ZS1zcGFjZTogcHJlLXdyYXA7XCI%2BUGlwZWxpbmUgUGFyYWxsZWxpc208L3NwYW4%2BPC9zcGFuPiJ9LHsiaWQiOiJjYXJkLTI2MTExLjQwMDAwMDAwMjIzNSIsImZyb250IjoiPHNwYW4gaWQ9XCJkb2NzLWludGVybmFsLWd1aWQtZjZlOTc1YzgtN2ZmZi05YmVkLTJjZjQtM2IxYjg2ZDcxNmQzXCI%2BPHNwYW4gc3R5bGU9XCJmb250LXNpemU6IDExcHQ7IGZvbnQtZmFtaWx5OiBBcmlhbCwgc2Fucy1zZXJpZjsgYmFja2dyb3VuZC1jb2xvcjogdHJhbnNwYXJlbnQ7IGZvbnQtdmFyaWFudDogbm9ybWFsOyB2ZXJ0aWNhbC1hbGlnbjogYmFzZWxpbmU7IHdoaXRlLXNwYWNlOiBwcmUtd3JhcDtcIj5UZWtuaWsgbWVtZWNhaCBvcGVyYXNpIGtvbXB1dGFzaSBtYXRlbWF0aXMgaW5kaXZpZHVhbCAoc2VwZXJ0aSBwZXJrYWxpYW4gbWF0cmlrcyA8L3NwYW4%2BPHNwYW4gc3R5bGU9XCJmb250LXNpemU6IDExcHQ7IGZvbnQtZmFtaWx5OiBBcmlhbCwgc2Fucy1zZXJpZjsgYmFja2dyb3VuZC1jb2xvcjogdHJhbnNwYXJlbnQ7IGZvbnQtc3R5bGU6IGl0YWxpYzsgZm9udC12YXJpYW50OiBub3JtYWw7IHZlcnRpY2FsLWFsaWduOiBiYXNlbGluZTsgd2hpdGUtc3BhY2U6IHByZS13cmFwO1wiPkF0dGVudGlvbjwvc3Bhbj48c3BhbiBzdHlsZT1cImZvbnQtc2l6ZTogMTFwdDsgZm9udC1mYW1pbHk6IEFyaWFsLCBzYW5zLXNlcmlmOyBiYWNrZ3JvdW5kLWNvbG9yOiB0cmFuc3BhcmVudDsgZm9udC12YXJpYW50OiBub3JtYWw7IHZlcnRpY2FsLWFsaWduOiBiYXNlbGluZTsgd2hpdGUtc3BhY2U6IHByZS13cmFwO1wiPiBwYWRhIHNhdHUgbGFwaXNhbikgc2VjYXJhIGhvcml6b250YWwga2UgYmViZXJhcGEgR1BVIHVudHVrIGRpa2VyamFrYW4gYmVyc2FtYS1zYW1hPC9zcGFuPjwvc3Bhbj4iLCJiYWNrIjoiPHNwYW4gaWQ9XCJkb2NzLWludGVybmFsLWd1aWQtYjNjZGM4ODItN2ZmZi1iMDQyLTRkNzQtMGEyYjJmNWE0NGUxXCI%2BPHNwYW4gc3R5bGU9XCJmb250LXNpemU6IDExcHQ7IGZvbnQtZmFtaWx5OiBBcmlhbCwgc2Fucy1zZXJpZjsgYmFja2dyb3VuZC1jb2xvcjogdHJhbnNwYXJlbnQ7IGZvbnQtdmFyaWFudDogbm9ybWFsOyB2ZXJ0aWNhbC1hbGlnbjogYmFzZWxpbmU7IHdoaXRlLXNwYWNlOiBwcmUtd3JhcDtcIj5UZW5zb3IgUGFyYWxsZWxpc208L3NwYW4%2BPC9zcGFuPiJ9XSwiaW5zdHJ1Y3Rpb24iOiJLbGlrIGthcnR1IHVudHVrIG1lbGloYXQgamF3YWJhbm55YS4ifQ%3D%3D" height="522" width="100%"></iframe>

*Nah*, sampai di sini mungkin Anda berpikir seperti ini *"Wah, keren! Jadi solusinya saya harus sewa cluster GPU berisi 4x atau 8x H100 supaya bisa jalanin Model Parallelism?"* 

Tahan terlebih dahulu karena kita akan berbicara realita.

Menyewa satu H100 saja harganya sudah sangat mahal per jamnya, apalagi menyewa 8 buah sekaligus hanya untuk fine-tuning model 8B, kecuali kita bekerja di perusahaan teknologi raksasa seperti Samsung dengan budget "sultan".

![dos-02e776231edf4dad607bab41960b32d920260429184051.png](https://assets.cdn.dicoding.com/original/academy/dos-02e776231edf4dad607bab41960b32d920260429184051.png)

Bagi kebanyakan dari kita yang mungkin hanya punya akses ke Google Colab gratisan (T4 GPU 16GB) atau GPU gaming di rumah (RTX 3090 atau 4090 24GB), angka 128GB tadi adalah *Game Over*.

Namun, ceritanya akan berbeda jika kita mengubah *gameplay*-nya.

Alih-alih mengupdate seluruh 100% bobot modelnya, kita bisa "menipu" presisi matematikanya agar tidak butuh memori sebesar itu. Inilah konsep yang memungkinkan kita melatih Llama 3 di GPU "murah" sekalipun, yaitu Quantization dan PEFT atau Parameter-Efficient Fine-Tuning.

***

## Efisiensi dengan PEFT dan Quantization

Sebelumnya, kita sudah membahas cara paling “radikal” dari Fine-tuning, yaitu Full Fine-tuning. Kita bicara soal “membelah” data serta model, menyewa klaster GPU dengan NVLink, dan biaya infrastruktur yang mungkin setara dengan harga sebuah mobil mewah hanya untuk satu kali training.

Jika hanya itu satu-satunya cara untuk melakukan *alignment*, mungkin teknologi ini hanya akan dikuasai oleh segelintir perusahaan raksasa di Silicon Valley.

*Eits*, tentu saja kita tidak akan meratapi hal tersebut karena ada jalan memutar yang lebih inklusif untuk semua orang.

![dos-9dc98d58a777680f293f8acce767771220260504162753.jpeg](https://assets.cdn.dicoding.com/original/academy/dos-9dc98d58a777680f293f8acce767771220260504162753.jpeg)

Hari ini, kita memasuki babak revolusi efisiensi untuk meninggalkan pendekatan “Full” tersebut di mana seluruh parameter model “dipaksa” untuk berubah. Nah, terdapat dua pilar utama yang memungkinkan kita “menjejalkan” model raksasa ke dalam GPU consumer biasa seperti RTX 3090 atau 4090 yang sudah sangat umum ditemukan di kebanyakan laptop gaming.

*   **Quantization:** Ini adalah “seni kompresi” yang mana kita akan belajar memangkas presisi angka dari yang tadinya boros memori menjadi sangat padat.
*   **PEFT (Parameter-Efficient Fine-Tuning):** Daripada melatih ulang 7 miliar parameter yang berisiko merusak ingatan model, kita justru akan melatih lapisan tambahan yang sangat “tipis”.

Jika Anda pernah dengar sebelumnya, kombinasi keduanya inilah yang melahirkan teknik legendaris bernama QLoRA. Oleh karena itu, simpan saja dompet Anda karena kita tidak perlu menyewa A100 untuk sesi ini. Mari kita bedah cara melakukan fine-tuning level dunia dengan *resource* yang masuk akal.

***

## Quantization “Seni Kompresi”

*Alright*, komponen yang pertama akan kita bahas adalah Quantization. Perlu Anda ketahui bahwa Quantization itu bukanlah teknik yang baru “eksis” saat LLM muncul ke peradaban. Salah satu bidang yang menerapkan konsep ini salah satunya adalah **audio** atau **musik**.

Apakah Anda ingat gelombang audio WAV pada kelas Fundamental Generative AI atau lebih tepatnya di modul Audio Generation sebelumnya?

Bentuk gelombang atau sinyal WAV sangat mulus dan bergerak seperti air mengalir. Jika kita merekamnya ke dalam piringan hitam (Vinyl), kita mencoba meniru gelombang itu seakurat mungkin. Namun, saat kita mengubahnya menjadi file MP3, proses Quantization dapat dilakukan dengan mengambil sampel suara tersebut. Lalu, kita “memaksa” gelombang yang mulus tadi masuk ke dalam tingkatan tangga (steps) yang terbatas.

Agar terproyeksi di kepala Anda, coba perhatikan ilustrasi di bawah ini.

![dos-3e82612e0ca39f4b09b545f5db3ff04420260430082118.gif](https://assets.cdn.dicoding.com/original/academy/dos-3e82612e0ca39f4b09b545f5db3ff04420260430082118.gif)

Garis biru adalah sinyal asli WAV dengan presisi tinggi, sedangkan garis merah yang patah-patah adalah hasil kuantisasi. Ada informasi yang hilang di antara celah tangga tersebut. Selisih antara garis merah dan biru itulah yang kita sebut Quantization Error atau *Noise*.

*Alright*, sekarang kita sadar bahwa dunia digital adalah sekumpulan tangga diskrit dan bukan kurva mulus layaknya ombak di pantai. Nah, tentu pengetahuan tersebut harus di-*follow up* dengan satu pertanyaan, “bagaimana cara kita membangun tangga tersebut di dunia LLM?”

Pada dasarnya tidak hanya LLM, tetapi di dunia Deep Learning, kita akan berhadapan dengan angka yang perilakunya sangat ekstrem. *Wait*, apa maksud ekstrem di sini?

![dos-a91718a93933d2c70fc1d4128b42be5620260430082252.png](https://assets.cdn.dicoding.com/original/academy/dos-a91718a93933d2c70fc1d4128b42be5620260430082252.png)

Seperti yang terlihat pada ilustrasi di atas, Anda mungkin memiliki nilai Loss yang dimulai dari angka puluhan atau ratusan ribu. Namun, di sisi lain, Anda memiliki nilai Weight sebesar 0.87531 dan juga Learning Rate yang tidak kalah mikroskopis, seperti 0.000003.

*Nah*, jika kita menggunakan sistem penyimpanan angka biasa seperti Integer, model kita tidak akan pernah “pintar”. Integer sangat bagus untuk melakukan perhitungan yang exact nilainya seperti jumlah apel atau jumlah token. Namun, ia tidak mampu menangkap nuansa desimal yang nilainya sangat “halu”, misalnya Integer akan menganggap nilai 0.000003 itu adalah nol. Dalam AI, jika gradien dianggap nol, model Anda berhenti belajar alias *shutdown*.

Oleh karena itu, kita menggunakan lawan mainnya yang disebut floating point yang bisa menampung angka raksasa saat dibutuhkan, tetapi juga bisa berubah menjadi mikroskop untuk melihat angka super kecil saat diperlukan.

### Floating Point

Berbeda dengan uang di bank yang titik desimalnya statis selalu 2 angka di belakang koma, dalam Floating Point, titik desimal itu bebas “mengapung” dan diletakan di mana saja tergantung seberapa besar atau kecil angka yang ingin kita simpan.

Jika diingat kembali, sebelumnya kita sudah pernah bertemu dengan dua jenis varian dari keluarga Floating Point.

*   **FP32:** Floating point dengan ukuran 32 bit.
*   **BFP16/FP16:** Floating point dengan ukuran 16 bit.

Apa maksud dari ukuran bit ini? Kita akan membahas hal tersebut nanti. Sebelum itu, untuk memahami fleksibilitas floating point dalam menentukan letak titik desimal, kita perlu mengingat kembali pelajaran matematika di tingkat SMA, yaitu notasi ilmiah yang contohnya diperlihatkan pada ilustrasi berikut.

![dos-49c704c63111a78b2c42672c430908b420260430082333.png](https://assets.cdn.dicoding.com/original/academy/dos-49c704c63111a78b2c42672c430908b420260430082333.png)

Lantas apakah komputer atau mesin mau menerima nilai seperti ini secara mentah?

Tentu saja jawabannya, tidak. Komputer memang dapat memproses angka, tetapi ia tidak memproses angka seperti cara berpikir manusia. Pada dasarnya, komputer adalah benda mati yang dialiri listrik. *Nah*, di dalam prosesor-nya, terdapat miliaran saklar mikroskopis yang disebut Transistor.

*   Saklar ini hanya punya dua posisi, yaitu “nyala” yang diberi simbol 1 atau “mati” yang diberi simbol 0.
*   Komputer tidak mengenal angka "5". Untuk menyimpan angka 5, komputer harus menyusun saklar-saklarnya dalam urutan “nyala” - “mati” - “nyala”.

![dos-26df7bb713afbeb8e0a304477b8994d320260430082514.png](https://assets.cdn.dicoding.com/original/academy/dos-26df7bb713afbeb8e0a304477b8994d320260430082514.png)

Artinya, komputer selalu bekerja dengan cara binary, nyata atau mati, 1 atau 0. *Nah*, setiap nilai 1 atau 0 inilah yang kita anggap sebagai 1 bit seperti yang ditunjukkan ilustrasi di atas.

Sekarang pertanyaannya adalah bagaimana konsep ini bisa kita gunakan untuk membuat notasi ilmiah yang digunakan dalam Floating Point sebelumya?

Seperti yang kita pelajari sebelumnya, notasi ilmiah tersebut memiliki 3 komponen, yaitu “tanda plus atau minus”, “angka utama”, dan “pangkat”. *Nah*, standar internasional IEEE 754 telah mengatur cara memadatkan ketiga komponen notasi ilmiah tersebut dengan membagi "kavling" memori mesin menjadi tiga kompartemen vital, yaitu Sign, Exponent, dan Mantissa.

Tidak peduli apakah itu FP32, FP16, atau format lainnya, struktur anatominya selalu terdiri dari tiga bagian tersebut. Untuk memberikan gambaran untuk Anda, kita akan melihat cara kerja mesin dengan format FP16 merepresentasikan nilai desimal melalui ilustrasi berikut ini.

![dos-8f6bae8cd14e4b27751c40012d43f95d20260430082600.gif](https://assets.cdn.dicoding.com/original/academy/dos-8f6bae8cd14e4b27751c40012d43f95d20260430082600.gif)

Mari kita bedah satu per satu komponen “DNA” dari angka digital ini.

#### Sign

Bisa dikatakan bagian pertama ini adalah yang paling sederhana. Sign hanya membutuhkan 1 bit saja di ujung paling kiri yang berfungsi seperti “sakelar lampu”.

*   Jika nilainya **0**, angkanya **positif (+)**.
*   Jika nilainya **1**, angkanya **negatif (-)**.

*Yup*, hanya itu tugasnya. Ia tidak peduli seberapa besar atau kecil angkanya, ia hanya menentukan arahnya, ke kanan (positif) atau ke kiri (negatif) pada garis bilangan. Pada ilustrasi FP16 di atas, kita dapat lihat saat Sign ini berisi nilai 0, artinya angka yang akan dihasilkan adalah positif.

#### Exponent

*Nah*, inilah komponen yang sering menjadi "biang kerok" masalah numerik di AI. Mengapa bisa begitu? Anda akan mengetahui jawabannya nanti.

Sesuai namanya, Exponent bertugas menentukan skala atau magnitudo dari angka tersebut. Dapat dikatakan, ia yang menentukan seberapa jauh jangkauan angka yang dapat ditampung oleh model. Misalnya kita memiliki garis bilangan seperti pada ilustrasi di bawah ini.

![dos-fad74e54cf8a8e99d1520c9253ac448320260430082637.png](https://assets.cdn.dicoding.com/original/academy/dos-fad74e54cf8a8e99d1520c9253ac448320260430082637.png)

Dalam pelatihan model, Exponent adalah titik kunci yang menjaga agar Gradient kita tetap hidup. Jika alokasi bit untuk Exponent terlalu sedikit, rentang angkanya menjadi sempit misalnya seperti 100 dan -100 di atas. Akibatnya, jika gradien melonjak sedikit saja katakanlah 120, ia akan menabrak dinding batas atas dan menjadi Infinity (NaN). Sebaliknya, jika gradien mengecil sedikit saja, ia akan dianggap Nol (Null).

Trik paling mudah untuk memahami Exponent adalah cukup ingat bahwa Exponent inilah yang menentukan besar jangkauan.

#### Mantissa

Mantissa atau sering disebut Significand atau Fraction perannya seperti megapixel atau resolusi di kamera Anda. Jika Exponent sebelumnya menentukan *“apakah angka Anda sebesar matahari atau sekecil elektron”*, Mantissa yang menentukan *“apakah matahari itu terlihat bulat sempurna atau kotak-kotak seperti di game Minecraft”*.

Mantissa adalah tempat kita menyimpan digit-digit aktual di belakang koma. Semakin banyak bit yang kita alokasikan untuk Mantissa, semakin “halus” atau rapat jarak antar-angka yang bisa kita bedakan.

![dos-b7d0f3ffc3650c92ed0f058c2ca3ef7920260430082702.gif](https://assets.cdn.dicoding.com/original/academy/dos-b7d0f3ffc3650c92ed0f058c2ca3ef7920260430082702.gif)

Lantas apa yang terjadi jika Mantissa kita terbatas?

Keterbatasan bit Mantissa akan melahirkan sebuah konsep fatal bernama Machine Epsilon. Epsilon adalah jarak terkecil antara angka 1.0 dan angka berikutnya yang bisa dikenali komputer. Jika Anda menambahkan 1.0 dengan angka yang super kecil seperti 0.0000001, komputer akan tetap menganggap nilai tersebut 1.0

Fenomena seperti ini tentu akan sangat berbahaya saat melakukan fine-tuning. Bayangkan Anda sedang melatih model dengan Learning Rate serta Gradien yang super kecil. Saat Backpropagation, kita ingin meng-*update* bobot model. Namun, karena hasil perkalian antara Gradien dan Learning Rate terlalu “mini”, model Anda diam di tempat karena tidak ada perubahan apa pun pada bobot, padahal GPU bekerja keras. Inilah yang disebut Swallowed Update.

Jika Exponent sebelumnya adalah jangkauan,kata kunci untuk Mantissa adalah presisi atau tingkat kedetailan.

Setelah kita membedah anatomi ketiga komponen Floating Point, sekarang saatnya melihat bagaimana industri AI mengatur ulang “kue” 32-bit dan 16-bit tersebut. Pemahaman ini akan menyelamatkan training Anda dari dua mimpi buruk, yaitu *Out of Memory* (OOM) atau *Not a Number* (NaN).

### Precision Floating Point

Saat membahas fine-tuning atau bahkan RAG, kita telah mengenal kosakata asing seperti FP32 yang disebut sebagai Full-precision dan FP16 versi Half-precision. Tibalah saatnya kita benar-benar membedah satu per satu semua hal tersebut bahkan dengan bonus “Si Anak Baru”, yaitu BFP16 atau BF16.

Sebagai langkah awal dan gambaran besar perbandingan dari ketiganya, mari kita perhatikan gambar di bawah ini.

![dos-aa91df0c3beddd1f8c3c96b8c74cb5ba20260311153249.png](https://assets.cdn.dicoding.com/original/academy/dos-aa91df0c3beddd1f8c3c96b8c74cb5ba20260311153249.png)

*Alright*, mari kita mulai dari FP32 terlebih dahulu.

#### FP32

FP32 atau dikenal juga sebagai Single Precision adalah sebuah format data berukuran 32 bit yang didefinisikan dalam standar IEEE 754. *Nah*, pasti Anda penasaran tentang *“apa itu IEEE 754?”* yang sering kita sebut belakangan ini. Sederhananya itu adalah standar internasional untuk merepresentasikan bilangan desimal atau floating point dalam format biner di komputer, Anda dapat mengunjungi tautan berikut: [IEEE 754](https://www.geeksforgeeks.org/computer-organization-architecture/ieee-standard-754-floating-point-numbers/) untuk membaca secara detail.

Selama bertahun-tahun, ini adalah format default untuk semua operasi AI. Tidak percaya? Coba Anda jalankan kode di bawah ini tanpa argumen tambahan.

```python
x = torch.randn(3, 3)
print(x.dtype)
```

Kode tersebut akan membuat tensor acak dengan settingan default dan jika Anda print tipe datanya, akan keluar kata “torch.float32”. Namun, apakah Anda tidak penasaran mengapa ia disebut “Single Precision”?

Sederhananya, FP32 ini memakan satu unit memori standar 32-bit sebesar 4 byte. Kenapa bisa 4 byte? Itu karena standar baku dunia menetapkan 1 byte dapat menampung 8 bit.

![dos-c6e97006af2b2d62a8c8e3b928a9673b20260430082752.png](https://assets.cdn.dicoding.com/original/academy/dos-c6e97006af2b2d62a8c8e3b928a9673b20260430082752.png)

Ilustrasi di atas adalah 1 byte komponen exponent dengan 8 bit. Seperti yang terlihat pada diagram, setiap kotak bit memiliki nilai kelipatan dua. Jika kita menyalakan semua bitnya (11111111), kita akan mendapatkan total 255.

*Wait*, mengapa totalnya jadi 255?

Untuk membuktikan kenapa semua bit nyala (11111111) dalam 1 byte tersebut sama dengan 255, kita bisa menjumlahkan *place value* dari setiap posisi bit seperti yang ditunjukkan ilustrasi di bawah ini.

![dos-8357ace9a0318519bfa21838554b77d020260430082818.gif](https://assets.cdn.dicoding.com/original/academy/dos-8357ace9a0318519bfa21838554b77d020260430082818.gif)

*Yup*, ilustrasi di atas menunjukkan bahwa exponent memiliki jangkauan nilai pangkat dari 0 hingga 255. Jika kita *recall* kembali sebelumnya, kita belajar bahwa *exponent*-lah yang bertugas untuk menentukan besar jangkauan nilai dalam floating point. Artinya, jangkauan asli atau nilai yang dapat ditampung oleh floating point ini mulai dari 0 hingga angka yang besarnya bahkan tidak bisa lagi disebutkan tersebut.

*Nah*, mungkin Anda akan bertanya, “bagaimana cara floating point menangani pangkat negatif untuk menghasilkan pecahan jika rentangnya ada di positif seluruhnya?”

Oleh karena itu, standar IEEE 754 memperkenalkan bias dalam exponent. Mereka membuat kesepakatan bahwa angka 0 dalam komputer bukanlah “nol” yang sebenarnya. Dalam ilustrasi konversi nilai real ke floating point sebelumnya, exponent selalu dikurangi dengan bias. *Nah*, besar bias pada dasarnya tergantung dari jumlah bit dalam exponent yang digunakan.

*   **FP32:** 8 bit dalam exponent sehingga titik tengahnya di 127.
*   **FP16:** 5 bit dalam exponent sehingga titik tengahnya rendah 15.

*Nah*, karena nilai exponent selalu dikurangi dengan biasnya, dalam kasus 8 bit kita di atas, jangkauan pangkatnya berarti menjadi -126 hingga 127.

Oleh karena sudah menyinggung perihal ukuran dan memori, mari kita bedah alokasi 32 bit di dalam FP32.

![dos-629992473640564e02396602b142c84f20260430082846.png](https://assets.cdn.dicoding.com/original/academy/dos-629992473640564e02396602b142c84f20260430082846.png)

*   **Sign (1 Bit)** 
    Seperti yang sudah kita pahami, bagian sign sangat sederhana dan jelas, 0 untuk positif, 1 untuk negatif.
*   **Exponent (8 Bit)**
    Di atas sudah disinggung mengenai 8 bit exponent ini yang mana ia bisa menyimpan angka integer dari 0 sampai 255 sebagai pangkat. Hal ini membuat FP32 dapat menangani angka sekecil hingga sebesar .
*   **Mantissa (23 Bit)** 
    FP32 memberikan slot 23 bit mantissa untuk presisi. Ditambah dengan 1 Hidden Bit, total presisi efektifnya adalah 24 bit. Berkat ini, tingkat presisi dari FP32 sangat mengagumkan, ia bisa membedakan 1.000001 dan 1.000002 dengan sangat mudah.

Sampai di sini, kita mengakui bahwa FP32 adalah format "zona nyaman" bagi komputasi dengan presisi yang sangat detail dan jangkauan yang luas. Namun, di era AI modern di mana model menjadi semakin raksasa, biaya penyimpanan 4 byte untuk setiap parameter dan komputasi FP32 mulai dianggap terlalu berat.

Inilah yang memicu lahirnya inovasi format yang lebih efisien seperti FP16 dan BF16.

#### FP16

Sering disebut juga dengan Half Precision, FP16 sesuai namanya memangkas kebutuhan memori menjadi setengah dari FP32. Ia memangkas dari 32 bit menjadi 16 bit yang artinya untuk setiap parameter model weight butuh 2 byte.

Secara logika, VRAM Anda saat ini seharusnya terasa 2x lebih lega karena beban memori berkurang setengahnya jika menggunakan FP16. Namun, faktanya penghematan memori pada bagian Model Weights itu sebenarnya mitos atau ilusi semata dalam skenario training atau fine-tuning.

Mengapa bisa begitu? *Eits* tenang, kita akan membahasnya setelah perkenalan dengan FP16 telah rampung.

*Alright*, mari kita langsung saja lihat IEEE 754 membagi “kue” 16-bit yang sempit ini.

![dos-b3edb1e7d517007a4bbd8e843f390cc420260430082919.png](https://assets.cdn.dicoding.com/original/academy/dos-b3edb1e7d517007a4bbd8e843f390cc420260430082919.png)

Karena kita kehilangan 16 bit dari FP32, efisiensi perlu dilakukan.

*   **Sign (1 bit)** 
    Tetap ada. Kita masih butuh membedakan positif dan negatif.
*   **Exponent (5 bit)** 
    FP16 memangkas Exponent secara brutal dari 8 bit (di FP32) menjadi hanya 5 bit. Sehingga, rentang pangkat efektifnya menjadi sangat sempit mulai dari hingga .
*   **Mantissa (10 bit)** 
    Jatah resolusi dipotong dari 23 bit menjadi 10 bit. Ini berarti FP16 menjadi agak “rabun”. Ia hanya memiliki presisi sekitar 3 sampai 4 angka desimal. Contohnya, angka 3.14159 mungkin tersimpan dengan baik, tetapi 3.14159265 bagian “buntutnya” akan hilang atau dibulatkan.

Hasilnya jika kita bandingkan dari sisi Dynamic Range dan Precision dengan FP32, akan terlihat seperti dalam ilustrasi berikut ini.

![dos-18d052fdf534339069d95b80815c5d5b20260430083032.png](https://assets.cdn.dicoding.com/original/academy/dos-18d052fdf534339069d95b80815c5d5b20260430083032.png)

Oleh karena Exponent-nya hanya 5 bit dan Mantissa-nya hanya 10 bit, angka maksimal yang dapat ditampung oleh FP16 adalah 65.504. FP16 tidak bisa menyimpan angka di atas 65,504. Jika dalam perhitungan gradient maupun loss, hasilnya menyentuh angka 65,505 saja, FP16 akan menyerah dan mengubahnya menjadi infinity dan akan muncul sebagai NaN.

Selain itu, masalah rabun karena keterbatasan ukuran Mantissa sebelumnya akan membuat perubahan gradient atau bobot yang sangat kecil tidak akan ditangkap. Bagi FP16, angka 0.000000001 itu dianggap Nol (0). Akibatnya, *update* bobot tidak terjadi dan model Anda berhenti belajar padahal training masih berjalan. Kondisi itulah yang disebut Gradient Underflow.

Oleh karena itu, saat melatih model dengan format FP16, kita membutuhkan teknik yang disebut Mixed Precision Training atau bahasa gaulnya training dengan precision “campur-campur”. Strateginya seperti berikut ini.

*   **Master Weights (FP32):** Kita tetap menyimpan satu salinan asli bobot model dalam format FP32 yang aman. Inilah tambahan 4 byte yang kita lihat sebelumnya dalam perhitungan memori dalam bagian full fine-tuning sebelumnya.
*   **Compute (FP16):** Namun, saat melakukan perhitungan Weight, kita mengonversi Weight tersebut ke FP16 agar cepat.
*   **Loss Scaling:** Sebelum Backward Pass, kita kalikan nilai Loss dengan angka besar (misalnya x1000). Setelah perhitungan selesai, kita bagi lagi gradiennya dengan 1000 sebelum di-*update* ke Master Weights. Dengan begini, kita akan terhindar dari Underflow karena gradient atau loss yang nilainya sangat kecil sudah diskalakan menjadi jauh lebih besar.

*Nah*, di sini ada hal yang menarik. Seperti yang sudah kita hitung bersama sebelumnya, model Llama 3.1 8B dengan format FP16 membutuhkan 16 byte secara static untuk setiap parameternya. Tahukah Anda jika FP32 juga membutuhkan ukuran yang sama persis? Perhatikan ilustrasi berikut ini.

![dos-a297d6f8f7fe189d01ef6ca461c4e79920260430083100.png](https://assets.cdn.dicoding.com/original/academy/dos-a297d6f8f7fe189d01ef6ca461c4e79920260430083100.png)

Selanjutnya, kita akan menjawab pertanyaan *"Mengapa kita menggunakan FP32 untuk optimizer meskipun model yang digunakan berformat FP16?".* Jawabannya telah disebutkan sebelumnya, yaitu karena perubahan gradien—yang merupakan tugas utama optimizer—sering kali memiliki nilai yang sangat kecil. Oleh karena itu, untuk menghindari kondisi yang kita sebut underflow, optimizer menggunakan FP32 dengan presisi yang detail.

Sekarang masuk ke pertanyaan, jika FP32 dan FP16 secara static memory sama-sama berukuran 16 byte per parameter model, mengapa kita repot-repot menggunakan FP16?

Jawabannya ada di dua tempat, yaitu Activations dan Tensor Cores pada GPU.

Dalam perhitungan memori sebelumnya, kita mengatakan tidak akan menghitung activation terlebih dahulu karena ukurannya dinamis. *Yup*, dan itulah titik yang menunjukkan FP16 lebih hemat. Dalam FP32, semua matriks raksasa dari setiap layer disimpan dalam format 4 byte, sedangkan FP16 menyimpannya hanya dalam format 2 byte.

Selain memori aktivasi, alasan utama industri beralih ke FP16 atau BF16 karena GPU modern seperti A100 dengan arsitektur Ampere atau RTX 4090 telah memiliki sirkuit khusus yang disebut Tensor Cores. Sirkuit khusus ini memungkinkan dua hal berikut.

*   **Kecepatan hitung (TFLOPS):** Untuk operasi FP16 atau BF16 bisa 2 kali hingga 4 kali lebih cepat dibandingkan operasi FP32 biasa.
*   **Bandwidth:** Mengirim data 2 Byte dari memori ke chip jauh lebih cepat daripada mengirim 4 Byte.

Sehingga, meskipun "Total Static Memory"-nya sama 16 Byte, Mixed Precision yang digunakan FP16 membuat proses training dapat selesai 2 kali lebih cepat. Bagi Anda yang ingin membaca lebih detail mengenai tensor cores, kunjungi tautan berikut ini: [Tensor Cores](https://www.nvidia.com/en-eu/data-center/tensorcore/)

*Eits*, jika Anda sadar kita sudah menyinggung sedikit nama BF16. *Alright*, mari kita lanjutkan penjelasannya ke BF16.

#### BF16

Format ini lahir dari frustrasi para peneliti di Google Brain (sekarang Google DeepMind). Mereka lelah melihat training model besar mereka *crash* karena overflow di FP16, tapi mereka juga tidak mau kembali ke FP32 yang boros memori.

Mereka menyadari satu fakta psikologis tentang Neural Network *"Model AI itu tidak butuh angka yang sangat presisi di belakang koma. Mereka hanya butuh rentang angka yang luas agar gradien tidak mati."*

BF16 sebenarnya adalah hasil “operasi bedah” terhadap format FP32. Sekarang begini, saat kita mengambil FP32 (32-bit) dan ingin memotong ukurannya menjadi setengah (16-bit), bagian mana yang harus kita buang?

*   **FP16:** Memotong Exponent dan Mantissa secara berimbang. Hasilnya? Keduanya jadi “tanggung”.
*   **BF16:** Memotong habis Mantissa, tetapi mempertahankan Exponent agar tetap utuh seperti FP32.

Mari kita lihat pembagian “kue” 16-bit ala BF16.

![dos-60f4af47b37bc9d7594194e8b316960f20260430142029.png](https://assets.cdn.dicoding.com/original/academy/dos-60f4af47b37bc9d7594194e8b316960f20260430142029.png)

*   **Sign (1 Bit)** 
    Sama seperti yang lain. Tetap ada 1 bit untuk tanda positif atau negatif.
*   **Exponent (8 Bit)** 
    *Nah*, di sinilah point menariknya, BF16 memberikan 8 bit penuh untuk Exponent yang mana ini sama persis dengan jumlah bit Exponent di FP32. Dengan begini, Loss Scaling seperti yang kita lakukan sebelumnya dalam FP16 dapat kita lupakan dalam BF16.
*   **Mantissa (7 Bit)** 
    Untuk mendapatkan Exponent 8 bit, tentu saja ada harga yang perlu dibayar. Mantissa yang bertanggung jawab soal presisi hanya kebagian sisa 7 bit saja. Ini bahkan lebih sedikit daripada FP16 yang punya 10 bit.

*Nah*, karena memiliki exponent yang sama, saat kita bandingkan dengan FP32, keduanya akan memiliki dynamic range yang sama panjang.

![dos-007ea9469e01de7ab239f122b89ff93220260430083442.png](https://assets.cdn.dicoding.com/original/academy/dos-007ea9469e01de7ab239f122b89ff93220260430083442.png)

*Yup*, meski dengan 16 bit, BF16 dapat menyimpan angka yang berkisar mulai dari hingga . Namun, tentu saja kedetailan atau presisi angkanya tidak akan setara dengan FP32.

*Nah*, sekarang mari kita jawab keresahan yang pasti muncul berikut *"Jika presisinya rendah, update kecil akan dianggap tidak ada perubahan.".* Lantas mengapa Google dan NVIDIA berani bertaruh pada BF16?

Rahasianya adalah karena saat melakukan training, kita akan memanggil Master Weights dan menggunakan teknik Mixed Precision training. Selain itu, dalam Deep Learning, kita tidak pernah tahu arah gradien yang "sempurna" karena datanya terlalu banyak. Kita hanya memperkirakan arahnya berdasarkan satu batch kecil data.

Artinya, gradien kita itu sendiri sebenarnya sudah penuh dengan noise atau gangguan.

*   Apakah gradien 0.00013452 itu benar-benar akurat?
*   Atau sebenarnya nilai aslinya 0.00014000?

Kita tidak tahu secara pasti.

Para peneliti menemukan fakta bahwa menambahkan noise tambahan seperti yang terjadi dalam pembulatan BF16 justru membantu model melakukan generalisasi. Dengan memaksa angka menjadi sedikit "buram" (BF16), model dipaksa untuk belajar pola-pola besar yang penting saja, bukan detail-detail noise yang tidak berguna.

Oleh sebab itu, BF16 sering dianggap sebagai format terbaik untuk LLM modern karena ia menggabungkan keunggulan dua dunia.

*   **Stabilitas FP32:** Anda tidak perlu pusing memikirkan Loss Scaling atau Gradient Clipping yang rumit, *cause* *It just works*.
*   **Kecepatan dan Efisiensi Memori FP16:** Karena ukurannya cuma 16-bit (2 Byte), ia memakan VRAM setengah dari FP32 dan transfer datanya sangat cepat.

Inilah sebabnya model-model modern seperti Llama-3, Gemma, atau Qwen dilatih secara native menggunakan BF16.

*Nah*, sampai di sini sebenarnya perjalanan Precision Floating Point kita sudah hampir sampai di tempat pemberhentian selanjutnya. Sebelum itu ada satu Quest menarik yang dapat Anda coba. Di titik ini, kita sudah paham bahwa FP32 memiliki presisi dan range yang sangat luas dibandingkan dengan FP32. Lalu, muncul BF16 sebagai mengakomodasi range luas dengan mengorbankan presisi.

Namun, itu semua hanya teori di atas kertas. Anda dapat melakukan eksperimen secara mandiri pada notebook berikut: [Notebook Colab](https://colab.research.google.com/drive/1fH3NiwNO6X_4jW8zej9KVIjyKtoQujSh?usp=sharing).

Kita telah mencapai puncak efisiensi dalam dunia Floating Point. Namun, sadarkah Anda bahwa ketiga format di atas memiliki satu kesamaan yang menjadi “tembok pembatas” efisiensi?

*Yup*, ketiganya masih bermain di ranah 16-bit ke atas.Mungkin bagi GPU data center sekelas A100 dengan VRAM 80GB, 16-bit itu kecil. Namun, bagi kita yang ingin menjalankan model Llama-3-70B di consumer hardware atau bahkan di laptop, 16-bit itu masih terlalu gemuk.

Masalahnya, jika ingin memadatkan model lebih ekstrem lagi misalnya menjadi 4-bit, kita tidak bisa lagi menggunakan struktur Floating Point. Mengapa demikian? Itu karena jika kita hanya punya 4 bit, tidak ada ruang yang cukup untuk membagi kavling Sign, Exponent, dan Mantissa secara proporsional dan *make sense*.

Untuk menembus batas ini, kita harus melakukan Paradigm Shift yang agresif.

### Quantized Precision

Jika di bab sebelumnya kita membahas cara “licik” menjaga dynamic range dan presisi, di sini, kita akan belajar seni mengorbankan presisi demi efisiensi ekstrem tanpa membuat model menjadi “bodoh”.

Pengorbanan ini akan memaksa kita untuk tidak lagi menyimpan angka desimal yang rumit (0.123456). Sebaliknya, kita akan melakukan Discretization atau pengelompokan. Bayangkan kita memotret spektrum warna pelangi yang tak terbatas, lalu memaksanya masuk ke dalam kotak krayon dengan jumlah warna yang terbatas.

Di modul ini, kita akan membagi pembahasan menjadi dua level kompresi.

*   **Level Aman (8-Bit):** Kita akan memadatkan model menjadi seperempat ukuran aslinya. Ini adalah standar industri yang stabil.
*   **Level Ekstrem (4-Bit):** Kali ini model dipadatkan hingga seperdelapan ukuran aslinya. Kita tidak hanya akan membahas INT4 (Integer biasa), tetapi juga akan membongkar “Rahasia NF4” sebuah tipe data alien yang diciptakan khusus untuk teknik QLoRA.

Mari kita mulai bedah dari level yang paling dasar, yaitu 8-Bit Precision.

#### 8-Bit Precision (INT 8)

INT8 itu artinya kita hanya memiliki 8 bit yang secara matematika dapat diilustrasikan seperti berikut ini.

*Yup*, kita akan memaksa seluruh kekayaan informasi model yang tadinya berupa miliaran kemungkinan angka desimal di FP32 untuk masuk ke dalam 256 “slot parkir” saja. Rentang angkanya pun menjadi sangat sempit sekali dan berupa integer mulai dari -128 hingga +127.

![dos-67e290f43f2bf878435bcc3238b123b420260430083533.png](https://assets.cdn.dicoding.com/original/academy/dos-67e290f43f2bf878435bcc3238b123b420260430083533.png)

Bayangkan tantangannya, bobot asli model (FP32) yang sangat bervariasi bisa bernilai 0.1234, -5.678, 100.5, atau 0.00009. Tugas kita adalah me-*mapping* semua kemungkinan angka desimal yang tak terbatas itu ke dalam salah satu dari 256 angka bulat yang tersedia. Kita seperti sedang berada di film Mission Impossible bersama Tom Cruise.

Untuk keluar dari situasi yang *impossible* tersebut, kita akan menggunakan teknik matematika yang disebut Affine Quantization atau sering dikenal juga sebagai Asymmetric Quantization.

Proses Quantization tersebut akan melibatkan dua rumus matematis utama. *Eits* tenang, rumusnya sangat sederhana seperti yang diperlihatkan dalam ilustrasi berikut ini.

![dos-bb4da7d2fe3410b189a74cde23e6e97420260430083622.png](https://assets.cdn.dicoding.com/original/academy/dos-bb4da7d2fe3410b189a74cde23e6e97420260430083622.png)

Anda mungkin akan langsung penasaran, *“mengapa kita perlu melakukan Dequantization atau mengembalikan nilainya menjadi nilai asli setelah dikuantisasi?”*

Jawabannya sederhana, menghitung Gradien dan Bobot (Weight) menggunakan integer itu mustahil dilakukan secara akurat. Matematika training model seperti itu membutuhkan ruang gerak yang sangat luas. Sementara itu, INT8 memiliki rentang yang sangat sempit, yaitu mulai dari -128 sampai 127 atau dengan kata lain 256 titik resolusi.

Bayangkan saat proses training, kita perlu mengalikan Matriks A (10) dengan Matriks B (50) yang mana hasilnya adalah 500. *Nah*, di sinilah masalahnya, angka 500 itu jauh melampaui kapasitas INT8 yang cuma sampai 127.

Oleh karena itulah, kita wajib mengembalikannya ke “wadah” yang lebih besar seperti FP32, FP16, atau bahkan BF16 agar angkanya tidak rusak.

*Nah*, dari sini akan muncul *follow up question* kembali, *“jika perhitungan dalam training tidak bisa dilakukan dengan integer, apa manfaatnya untuk kita menggunakan INT8 ini?”*

*Yup*, memang benar saat melakukan komputasi training kita perlu Floating Point, tetapi INT dapat dimanfaatkan untuk mengefisiensi model saat berada di “ruang penyimpanan”. Apa maksudnya? Coba Anda perhatikan ilustrasi di bawah ini.

![dos-a7864e2f241e62cbcba8bb9cbff25ae820260430083646.png](https://assets.cdn.dicoding.com/original/academy/dos-a7864e2f241e62cbcba8bb9cbff25ae820260430083646.png)

Sesuai dengan ilustrasi di atas, INT 8 digunakan pada saat model berada dalam fase Storage dan Bandwidth atau Transfer. Mari kita bedah sedikit siklus hidup data dalam inferensi LLM.

*   **Storage (Hard disk/VRAM):** Saat model masih dalam kondisi idle, data disimpan sebagai INT.
*   **Memory Bandwidth:** Data masih dalam format INT saat dikirim dari VRAM ke Compute Unit (Core GPU). Sehingga, proses transfer data bisa lebih kencang karena paketnya kecil.
*   **Tepat Sebelum Komputasi:** *Core GPU* sebagai tempat komputasi terjadi memiliki pintu masuk yang disebut *register chip*. *Nah*, di dalam *register chip*, INT8 diubah (*dequantization*) sesaat menjadi FP16, BF16, atau FP32.
*   **Komputasi:** Perhitungan matematis (Perkalian dan Penjumlahan) dilakukan di “wadah” FP16, BF16, atau FP32.
*   **Re-quantization:** Setelah semua hitungan selesai dan sebelum data dikirim keluar dari core menuju VRAM sebagai memori utama, data tersebut akan di-*quantize* ulang menjadi INT8.

*Alright*, agar konsep ini semakin tertanam jauh di kepala, mari kita lakukan simulasi sederhana. Misalkan, kita memiliki nilai 0.5 dengan format Float yang ingin dikonversi menjadi INT8.

Tahap pertama yang akan kita lakukan adalah menghitung Scale dan Zero-Point-nya. Nah, sebelum dapat menghitung kedua hal tersebut, kita perlu mengetahui dulu besar range dari format Float yang digunakan ini. Oke, anggaplah format *float* yang kita gunakan ini memiliki range -1.0 hingga 3.0. Sekarang, kita memperoleh nilai Scale yang menjawab pertanyaan *"angka float 0.0 setara dengan integer nomor berapa?"*. Nilai inilah yang disebut sebagai Zero-Point.

Agar lebih menarik kita akan menggunakan ilustrasi berikut.

![dos-f438c15e52b8e42b9dcc87f333c4d52f20260430083826.gif](https://assets.cdn.dicoding.com/original/academy/dos-f438c15e52b8e42b9dcc87f333c4d52f20260430083826.gif)

Dari sini kita mendapatkan nilai *scale*-nya sekitar 0.0157 dan *zero-point*-nya adalah -64. Ini berarti jika komputer melihat integer -64, komputer akan paham bahwa itu sebenarnya adalah angka 0.0.

Barulah kita dapat mengubah angka float 0.5 menjadi integer.

![dos-47129772432b23a4623819a3d65083ed20260430083905.png](https://assets.cdn.dicoding.com/original/academy/dos-47129772432b23a4623819a3d65083ed20260430083905.png)

Berdasarkan perhitungan di atas, angka float 0.5 akan disimpan sebagai -32 dalam integer saat dilakukan Asymmetric Quantization. Setelah itu, saat kita mencoba untuk mengembalikan angkanya menjadi float melalui De-Quantization.

*Yup*, hasilnya adalah 0.5024 yang sangat dekat dengan nilai aslinya dalam float di awal, yaitu 0.5. Selisih 0.0024 itulah yang disebut Quantization Error (Noise). Quantization Error sekecil ini secara struktural masih sanggup ditoleransi oleh robustness dari arsitektur neural network tanpa melukai kecerdasan model aslinya.

Apakah kita sudah berdiri di puncak efisiensi?

Sayangnya ambisi efisiensi serasa tidak pernah puas. Alih-alih cukup dengan 256 titik resolusi pada INT8 ini, para ilmuwan break the rule dengan memangkas memori dengan hanya menggunakan 16 titik nilai. *Yup*, kita sudah tidak akan berbicara 8-Bit lagi, melainkan 4-Bit precision.

#### 4-Bit Integer (INT4)

Manusia yang tidak pernah punya rasa puas, membawa dunia efisiensi LLM ke tempat yang lebih jauh lagi. *Yup*, kali ini kita dituntut untuk menekan memori lebih ekstrem lagi dengan memotongnya menjadi separuh. Selamat datang di ranah presisi 4-bit.

Jika INT8 sebelumnya memberi kita ruang bernapas dengan 256 titik nilai diskrit, transisi menuju INT4 (4-bit Integer) adalah sebuah lompatan yang sangat ekstrem dan tidak kenal ampun. Bayangkan saja, alokasi memori 4-bit membuat kita hanya memiliki 24 atau 16 titik representasi nilai.

![dos-4e090244d16b2320124200d7ccb8385120260430083932.png](https://assets.cdn.dicoding.com/original/academy/dos-4e090244d16b2320124200d7ccb8385120260430083932.png)

*Yup*, hanya 16 titik. Rentang nilainya menyusut drastis menjadi hanya dari -8 hingga 7.

Di sinilah letak masalah arsitekturalnya. Saat kita menggunakan INT4 murni, kita masih mengadopsi pendekatan Linear Quantization seperti pada INT8. Artinya, ke-16 titik nilai ini didistribusikan dengan jarak yang sama rata (kaku) dari ujung ke ujung. Wait, memang apa masalahnya?

Coba Anda perhatikan ilustrasi di bawah ini.

![dos-ca30cd8245fc18aba1b173971adfef6220260430084021.png](https://assets.cdn.dicoding.com/original/academy/dos-ca30cd8245fc18aba1b173971adfef6220260430084021.png)

Kita memaksa rentang data yang besarnya puluhan ribu pada BFP16 misalnya seperti ilustrasi di atas, atau 255 nilai pada INT8 untuk masuk ke dalam 14-15 interval saja. Akibatnya, nilai Scale para perhitungan Quantization Integer yang kita kenal sebelumnya jadi membesar secara drastis. *Nah*, inilah yang memicu *Rounding Error* (Kesalahan Pembulatan) tingkat ekstrem. 

Rounding Error ini yang membuat nilai-nilai parameter yang secara desimal berbeda, akan dipaksa masuk ke dalam satu nilai bulat yang sama. Ini akan membuat model kehilangan sensitivitasnya.

Lalu, apa yang terjadi ketika bobot parameter LLM dipaksa masuk ke dalam 16 titik linear ini? Degradasi akurasi yang destruktif.

*Wait*, Rounding Error itu kan hanya terjadi ke beberapa nilai yang terdekat saja dan seharusnya tidak memberikan *impact* terlalu besar ke kecerdasan model. Lantas, mengapa ini berpengaruh sekali ke kecerdasan model.

Ada dua alasan teknis untuk menjawab pertanyaan tersebut.

*   **Information Loss pada Nilai Krusial**
    Di dalam arsitektur neural network, mayoritas bobot parameter (weights) berkerumun sangat padat di angka yang mendekati nol. Karena INT4 linear memiliki jarak antar-titik yang terlalu lebar dan rata, nilai-nilai bobot yang kecil namun sangat krusial ini akan dibulatkan menjadi nol secara paksa. Saat parameter berubah menjadi nol, jalur informasi di dalam network terputus.
*   **Distorsi akibat Outlier**
    Di sisi lain, neural network selalu memiliki nilai ekstrem (outlier). Karena grid 16 titik ini harus membentang untuk mencakup nilai ekstrem tersebut, resolusi di bagian tengah (tempat mayoritas data berada) menjadi sangat renggang.

![dos-4346a5c2e1e0278702bfb5e8266d8ca120260430084046.gif](https://assets.cdn.dicoding.com/original/academy/dos-4346a5c2e1e0278702bfb5e8266d8ca120260430084046.gif)

*Nah*, dari kegagalan INT4 linear inilah para peneliti sadar bahwa kita tidak bisa menggunakan pembagian titik yang rata (linear) jika distribusi data bobotnya sendiri tidak rata. Kita membutuhkan representasi data yang beradaptasi dengan posisi kerumunan data itu sendiri.

#### 4-Bit Floating Point (FP4)

FP4 hadir dengan sebuah premis yang berbeda dengan membuang aturan distribusi linear. Pada dasarnya, FP4 dihadapkan pada keterbatasan fisik yang masih sama dengan INT4, yaitu hanya memiliki alokasi 16 titik representasi. Namun, FP4 menyusun ke-16 titik tersebut menggunakan distribusi presisi non-linear.

*Yup*, sesuai namanya Floating Point, ia akan kembali menggunakan kombinasi lengkap, yaitu Sign, Exponent, dan Mantissa.

![dos-75863cef4748a70f8d48de04db3a6daa20260430084217.png](https://assets.cdn.dicoding.com/original/academy/dos-75863cef4748a70f8d48de04db3a6daa20260430084217.png)

*Wait*, kenapa hasil representasi ke FP4 terlihat seperti Integer? *Nah*, untuk menjawabnya kita perlu melihat bentuk arsitektur distribusinya.

![dos-05fbcfe6d1f0b5ce6eb2694483c80bed20260430084301.png](https://assets.cdn.dicoding.com/original/academy/dos-05fbcfe6d1f0b5ce6eb2694483c80bed20260430084301.png)

Alih-alih memberikan jarak yang sama persis antar titik nilai, algoritma tipe data FP4 secara sengaja merapatkan titik-titik representasinya di area yang mendekati angka nol. Sebaliknya, saat nilai bergerak menjauhi nol menuju angka ekstrem, jarak presisi antar-titiknya menjadi semakin renggang.

Desain asimetris ini bukanlah kebetulan, melainkan respon langsung terhadap realita statistik komputasi model machine learning.

Di dalam struktur internal Large Language Model, mayoritas nilai dari bobot parameter (*weights*) selalu berkerumun sangat padat di sekitar angka nol.

Jutaan parameter krusial yang sebelumnya "tergilas" dan dibulatkan menjadi nol pada INT4, kini mendapatkan slot memori representasinya sendiri secara akurat. Di saat yang sama, renggangnya presisi di area ekstrem tetap memungkinkan FP4 untuk mengakomodasi nilai outlier yang besar tanpa harus merusak resolusi data di area tengah.

Melalui FP4, batas memori 4-bit tidak lagi menjadi alasan mundurnya kecerdasan model. Kita berhasil membuktikan bahwa memori berkapasitas sangat rendah tetap mampu mempertahankan integritas logika LLM, asalkan struktur tipe datanya beradaptasi mengikuti letak kerumunan parameter model tersebut.

#### 4-Bit Normal Floating (NF4)

Keberhasilan FP4 dalam menyelamatkan akurasi model membuktikan satu prinsip krusial, yaitu struktur memori harus mengikuti topologi data. Namun, FP4 masih memiliki satu kelemahan terakhir. *Yup*, desain exponent dan mantissa pada FP4 pada dasarnya adalah sebuah perkiraan teknis yang dirancang oleh insinyur perangkat keras untuk menebak di mana letak data berkerumun.

Sekarang, bagaimana jika kita tidak perlu lagi menebak? Bagaimana jika kita merancang sebuah tipe data yang dapat menyesuaikan besar bobot model itu sendiri?

*Nah*, kehadiran tipe data NF4 yang diperkenalkan bersamaan dengan terobosan arsitektur QLoRA menjadi solusi dari permasalah tersebut. 

Para peneliti membuktikan bahwa ketika sebuah Large Language Model selesai dilatih hingga konvergen, nilai bobot parameternya (*weights*) tidak tersebar secara acak. Jutaan nilai desimal tersebut selalu, secara konsisten, membentuk Distribusi Normal (Gaussian Distribution) dengan nilai rata-rata berpusat tepat di angka nol (*zero-mean*).

NF4 membuang seluruh arsitektur floating point standar (Sign/Exponent/Mantissa) sebelumnya. Alih-alih menggunakan manipulasi bit biner, NF4 mendesain ulang letak ke-16 titik presisinya menggunakan kalkulasi probabilitas murni yang menghasilkan kurva distribusi normal. 

Hasilnya dapat Anda lihat pada ilustrasi di bawah ini.

![dos-43c2d36e90df4fe69ed27b12e3b05cbb20260430084326.gif](https://assets.cdn.dicoding.com/original/academy/dos-43c2d36e90df4fe69ed27b12e3b05cbb20260430084326.gif)

Jika Anda membedah ilustrasi di atas, Anda akan melihat bahwa kehebatan arsitektur NF4 ditopang oleh dua mekanisme yang bekerja secara beriringan:

*   **Posisi Slot yang Statis**
    Ke-16 titik (slot) memori pada kurva tersebut posisinya telah dikunci secara permanen berdasarkan kalkulasi kuantil dari distribusi normal baku. Artinya, sistem tidak perlu membuang tenaga komputasi untuk menghitung ulang di mana titik presisi harus diletakkan setiap saat.
*   **Representasi Nilai yang Dinamis**
    Meskipun posisi slotnya diam, jangkauan nilai asli yang diwakili oleh slot tersebut dapat memanjang atau menyusut secara dinamis. Saat memproses matriks Weight model, sistem akan mengekstrak nilai maximum dari blok matriks tersebut. Nilai maksimal inilah yang bertindak sebagai Scale Factor, memaksa rentang presisi NF4 beradaptasi secara otomatis mengikuti ukuran aktual bobot yang sedang diproses seperti yang terjadi pada ilustrasi di atas.

Hasilnya seperti yang Anda lihat di atas adalah sebuah grid memori yang secara sempurna melengkung dan melebur dengan lekukan probabilitas data. Setiap slot dari 16 titik pada NF4 memiliki probabilitas yang sama persis untuk diisi oleh bobot model. Tidak ada satu pun titik memori yang terbuang sia-sia untuk rentang nilai yang jarang muncul, dan tidak ada area kerumunan data yang menderita karena kekurangan resolusi. Pemetaan memori pada NF4 benar-benar identik dengan probabilitas letak data itu sendiri.

Melalui NF4, industri akhirnya berhasil mencapai apa yang sebelumnya dianggap mustahil: memampatkan arsitektur LLM raksasa ke dalam batasan memori 4-bit yang sangat sempit, namun secara menakjubkan mampu mempertahankan integritas logika, tingkat perplexity, dan akurasi yang nyaris tidak bisa dibedakan dengan model presisi 16-bit aslinya. Oleh mukjizatnya, kita akan memanfaatkan tipe data ini saat melakukan Quantization pada bagian latihan nantinya. Menarik bukan?

*Eits*, sebelum kesana, kita akan menutup teori Quantization ini dengan aktivitas menarik untuk memperkuat ingatan terhadap tipe datanya yang banyak sekali kerabatnya sebelum lanjut ke PEFT.

<iframe src="https://academy-sandpack.dicoding.dev/activities/flashcards?data=eyJjYXJkcyI6W3siaWQiOiJjYXJkLTI1Ni4xMDAwMDAwMDE0OTAxIiwiZnJvbnQiOiI8c3BhbiBpZD1cImRvY3MtaW50ZXJuYWwtZ3VpZC1hY2Y2N2ViYi03ZmZmLTY2NmEtMTc3ZC00NWE0MjIwYzFkMTBcIj48c3BhbiBzdHlsZT1cImZvbnQtc2l6ZTogMTFwdDsgZm9udC1mYW1pbHk6IEFyaWFsLCBzYW5zLXNlcmlmOyBiYWNrZ3JvdW5kLWNvbG9yOiB0cmFuc3BhcmVudDsgZm9udC12YXJpYW50OiBub3JtYWw7IHZlcnRpY2FsLWFsaWduOiBiYXNlbGluZTsgd2hpdGUtc3BhY2U6IHByZS13cmFwO1wiPlNlcmluZyBkaWd1bmFrYW4gYmVyc2FtYSBGUDMyIGRhbGFtIHRla25payA8L3NwYW4%2BPHNwYW4gc3R5bGU9XCJmb250LXNpemU6IDExcHQ7IGZvbnQtZmFtaWx5OiBBcmlhbCwgc2Fucy1zZXJpZjsgYmFja2dyb3VuZC1jb2xvcjogdHJhbnNwYXJlbnQ7IGZvbnQtc3R5bGU6IGl0YWxpYzsgZm9udC12YXJpYW50OiBub3JtYWw7IHZlcnRpY2FsLWFsaWduOiBiYXNlbGluZTsgd2hpdGUtc3BhY2U6IHByZS13cmFwO1wiPm1peGVkIHByZWNpc2lvbjwvc3Bhbj48c3BhbiBzdHlsZT1cImZvbnQtc2l6ZTogMTFwdDsgZm9udC1mYW1pbHk6IEFyaWFsLCBzYW5zLXNlcmlmOyBiYWNrZ3JvdW5kLWNvbG9yOiB0cmFuc3BhcmVudDsgZm9udC12YXJpYW50OiBub3JtYWw7IHZlcnRpY2FsLWFsaWduOiBiYXNlbGluZTsgd2hpdGUtc3BhY2U6IHByZS13cmFwO1wiPiB1bnR1ayBtZW1wZXJjZXBhdCBwcm9zZXMgcGVsYXRpaGFuIGRhbiBpbmZlcmVuc2kgcGFkYSBHUFUgbW9kZXJuIHRhbnBhIGtlaGlsYW5nYW4gYmFueWFrIGFrdXJhc2k8L3NwYW4%2BPC9zcGFuPiIsImJhY2siOiI8c3BhbiBpZD1cImRvY3MtaW50ZXJuYWwtZ3VpZC1mNWVlNmU5Ny03ZmZmLTg1NmYtZjc3My1kZTg2N2NkNGRiNjZcIj48c3BhbiBzdHlsZT1cImZvbnQtc2l6ZTogMTFwdDsgZm9udC1mYW1pbHk6IEFyaWFsLCBzYW5zLXNlcmlmOyBiYWNrZ3JvdW5kLWNvbG9yOiB0cmFuc3BhcmVudDsgZm9udC12YXJpYW50OiBub3JtYWw7IHZlcnRpY2FsLWFsaWduOiBiYXNlbGluZTsgd2hpdGUtc3BhY2U6IHByZS13cmFwO1wiPkZsb2F0aW5nIFBvaW50IDE2IChGUDE2KTwvc3Bhbj48L3NwYW4%2BIn0seyJpZCI6ImNhcmQtMjQzODkuMzAwMDAwMDAwNzQ1IiwiZnJvbnQiOiI8c3BhbiBpZD1cImRvY3MtaW50ZXJuYWwtZ3VpZC0zN2M0MGVkYi03ZmZmLTlkZGUtNGIxYi0zMGUxMTYyZTU0MzJcIj48c3BhbiBzdHlsZT1cImZvbnQtc2l6ZTogMTFwdDsgZm9udC1mYW1pbHk6IEFyaWFsLCBzYW5zLXNlcmlmOyBiYWNrZ3JvdW5kLWNvbG9yOiB0cmFuc3BhcmVudDsgZm9udC12YXJpYW50OiBub3JtYWw7IHZlcnRpY2FsLWFsaWduOiBiYXNlbGluZTsgd2hpdGUtc3BhY2U6IHByZS13cmFwO1wiPkZvcm1hdCAxNi1iaXQga2h1c3VzIHJhbmNhbmdhbiBHb29nbGUgeWFuZyBtZW1pbGlraSByZW50YW5nIGVrc3BvbmVuIHlhbmcgc2FtYSBwZXJzaXMgZGVuZ2FuIEZQMzIsIHNlaGluZ2dhIGphdWggbGViaWggaGVtYXQ8L3NwYW4%2BPHNwYW4gc3R5bGU9XCJmb250LXNpemU6IDExcHQ7IGZvbnQtZmFtaWx5OiBBcmlhbCwgc2Fucy1zZXJpZjsgYmFja2dyb3VuZC1jb2xvcjogdHJhbnNwYXJlbnQ7IGZvbnQtc3R5bGU6IGl0YWxpYzsgZm9udC12YXJpYW50OiBub3JtYWw7IHZlcnRpY2FsLWFsaWduOiBiYXNlbGluZTsgd2hpdGUtc3BhY2U6IHByZS13cmFwO1wiPm92ZXJmbG93PC9zcGFuPjxzcGFuIHN0eWxlPVwiZm9udC1zaXplOiAxMXB0OyBmb250LWZhbWlseTogQXJpYWwsIHNhbnMtc2VyaWY7IGJhY2tncm91bmQtY29sb3I6IHRyYW5zcGFyZW50OyBmb250LXZhcmlhbnQ6IG5vcm1hbDsgdmVydGljYWwtYWxpZ246IGJhc2VsaW5lOyB3aGl0ZS1zcGFjZTogcHJlLXdyYXA7XCI%2BIGRpYmFuZGluZ2thbiBGUDE2IGJpYXNhPC9zcGFuPjwvc3Bhbj4iLCJiYWNrIjoiPHNwYW4gaWQ9XCJkb2NzLWludGVybmFsLWd1aWQtMjgyZGI2NTYtN2ZmZi1jMzI5LWRlMTAtOTc2NzJlZDUxMTZlXCI%2BPHNwYW4gc3R5bGU9XCJmb250LXNpemU6IDExcHQ7IGZvbnQtZmFtaWx5OiBBcmlhbCwgc2Fucy1zZXJpZjsgYmFja2dyb3VuZC1jb2xvcjogdHJhbnNwYXJlbnQ7IGZvbnQtdmFyaWFudDogbm9ybWFsOyB2ZXJ0aWNhbC1hbGlnbjogYmFzZWxpbmU7IHdoaXRlLXNwYWNlOiBwcmUtd3JhcDtcIj5CcmFpbiBGbG9hdGluZyBQb2ludCAxNiAoQkYxNik8L3NwYW4%2BPC9zcGFuPin0seyJpZCI6ImNhcmQtMzI2OTMuOTAwMDAwMDAyMjM1IiwiZnJvbnQiOiI8c3BhbiBpZD1cImRvY3MtaW50ZXJuYWwtZ3VpZC00MzJkMmJhMS03ZmZmLWRlYmQtNThjYy1mM2IwNDBiMGViNTRcIj48c3BhbiBzdHlsZT1cImZvbnQtc2l6ZTogMTFwdDsgZm9udC1mYW1pbHk6IEFyaWFsLCBzYW5zLXNlcmlmOyBiYWNrZ3JvdW5kLWNvbG9yOiB0cmFuc3BhcmVudDsgZm9udC12YXJpYW50OiBub3JtYWw7IHZlcnRpY2FsLWFsaWduOiBiYXNlbGluZTsgd2hpdGUtc3BhY2U6IHByZS13cmFwO1wiPkZvcm1hdCBrdWFudGlzYXNpIGJpbGFuZ2FuIGJ1bGF0IHlhbmcgc2VjYXJhIGRyYXN0aXMgbWVuZ29tcHJlc2kgdWt1cmFuIG1vZGVsIEFJIGhpbmdnYSBzZXBlcmVtcGF0IGRhcmkgdWt1cmFuIGFzbGlueWEgZGVuZ2FuIG1lbWV0YWthbiByZW50YW5nIG5pbGFpIGRlc2ltYWwga2UgZGFsYW0gc2thbGEgYmlsYW5nYW4gYnVsYXQgOC1iaXQ8L3NwYW4%2BPC9zcGFuPiIsImJhY2siOiI8c3BhbiBpZD1cImRvY3MtaW50ZXJuYWwtZ3VpZC01ZGMxN2M4ZS03ZmZmLWY0MGUtOTU0YS03ZjQ4YTdiZTBlMjBcIj48c3BhbiBzdHlsZT1cImZvbnQtc2l6ZTogMTFwdDsgZm9udC1mYW1pbHk6IEFyaWFsLCBzYW5zLXNlcmlmOyBiYWNrZ3JvdW5kLWNvbG9yOiB0cmFuc3BhcmVudDsgZm9udC12YXJpYW50OiBub3JtYWw7IHZlcnRpY2FsLWFsaWduOiBiYXNlbGluZTsgd2hpdGUtc3BhY2U6IHByZS13cmFwO1wiPkludGVnZXIgOCAoSU5UOCk8L3NwYW4%2BPC9zcGFuPiJ9LHsiaWQiOiJjYXJkLTUxNjM5LjIwMDAwMDAwMjk4IiwiZnJvbnQiOiI8c3BhbiBpZD1cImRvY3MtaW50ZXJuYWwtZ3VpZC1lZTBmYjQ5ZC03ZmZmLTBmYmYtMjcxYy1lM2U1YzViNmQ1YjRcIj48c3BhbiBzdHlsZT1cImZvbnQtc2l6ZTogMTFwdDsgZm9udC1mYW1pbHk6IEFyaWFsLCBzYW5zLXNlcmlmOyBiYWNrZ3JvdW5kLWNvbG9yOiB0cmFuc3BhcmVudDsgZm9udC12YXJpYW50OiBub3JtYWw7IHZlcnRpY2FsLWFsaWduOiBiYXNlbGluZTsgd2hpdGUtc3BhY2U6IHByZS13cmFwO1wiPk1lcnVwYWthbiB0dWxhbmcgcHVuZ2d1bmcgZGFyaSB0ZWtuaWsgUUxvUkEsIG1lbXVuZ2tpbmthbiA8L3NwYW4%2BPHNwYW4gc3R5bGU9XCJmb250LXNpemU6IDExcHQ7IGZvbnQtZmFtaWx5OiBBcmlhbCwgc2Fucy1zZXJpZjsgYmFja2dyb3VuZC1jb2xvcjogdHJhbnNwYXJlbnQ7IGZvbnQtc3R5bGU6IGl0YWxpYzsgZm9udC12YXJpYW50OiBub3JtYWw7IHZlcnRpY2FsLWFsaWduOiBiYXNlbGluZTsgd2hpdGUtc3BhY2U6IHByZS13cmFwO1wiPmRldmVsb3Blcjwvc3Bhbj48c3BhbiBzdHlsZT1cImZvbnQtc2l6ZTogMTFwdDsgZm9udC1mYW1pbHk6IEFyaWFsLCBzYW5zLXNlcmlmOyBiYWNrZ3JvdW5kLWNvbG9yOiB0cmFuc3BhcmVudDsgZm9udC12YXJpYW50OiBub3JtYWw7IHZlcnRpY2FsLWFsaWduOiBiYXNlbGluZTsgd2hpdGUtc3BhY2U6IHByZS13cmFwO1wiPiB1bnR1ayBtZWxha3VrYW4gPC9zcGFuPjxzcGFuIHN0eWxlPVwiZm9udC1zaXplOiAxMXB0OyBmb250LWZhbWlseTogQXJpYWwsIHNhbnMtc2VyaWY7IGJhY2tncm91bmQtY29sb3I6IHRyYW5zcGFyZW50OyBmb250LXN0eWxlOiBpdGFsaWM7IGZvbnQtdmFyaWFudDogbm9ybWFsOyB2ZXJ0aWNhbC1hbGlnbjogYmFzZWxpbmU7IHdoaXRlLXNwYWNlOiBwcmUtd3JhcDtcIj5maW5lLXR1bmluZzwvc3Bhbj48c3BhbiBzdHlsZT1cImZvbnQtc2l6ZTogMTFwdDsgZm9udC1mYW1pbHk6IEFyaWFsLCBzYW5zLXNlcmlmOyBiYWNrZ3JvdW5kLWNvbG9yOiB0cmFuc3BhcmVudDsgZm9udC12YXJpYW50OiBub3JtYWw7IHZlcnRpY2FsLWFsaWduOiBiYXNlbGluZTsgd2hpdGUtc3BhY2U6IHByZS13cmFwO1wiPiBwYWRhIG1vZGVsIHJha3Nhc2EgKHNlcGVydGkgTGxhbWEgN0IgYXRhdSAxM0IpIGhhbnlhIGRlbmdhbiBtZW5nZ3VuYWthbiBzYXR1IGJ1YWggR1BVIGtlbGFzIGtvbnN1bWVuICg8L3NwYW4%2BPHNwYW4gc3R5bGU9XCJmb250LXNpemU6IDExcHQ7IGZvbnQtZmFtaWx5OiBBcmlhbCwgc2Fucy1zZXJpZjsgYmFja2dyb3VuZC1jb2xvcjogdHJhbnNwYXJlbnQ7IGZvbnQtc3R5bGU6IGl0YWxpYzsgZm9udC12YXJpYW50OiBub3JtYWw7IHZlcnRpY2FsLWFsaWduOiBiYXNlbGluZTsgd2hpdGUtc3BhY2U6IHByZS13cmFwO1wiPmNvbnN1bWVyIEdQVTwvc3Bhbj48c3BhbiBzdHlsZT1cImZvbnQtc2l6ZTogMTFwdDsgZm9udC1mYW1pbHk6IEFyaWFsLCBzYW5zLXNlcmlmOyBiYWNrZ3JvdW5kLWNvbG9yOiB0cmFuc3BhcmVudDsgZm9udC12YXJpYW50OiBub3JtYWw7IHZlcnRpY2FsLWFsaWduOiBiYXNlbGluZTsgd2hpdGUtc3BhY2U6IHByZS13cmFwO1wiPik8L3NwYW4%2BPC9zcGFuPiIsImJhY2siOiI8c3BhbiBpZD1cImRvY3MtaW50ZXJuYWwtZ3VpZC0zNDNkYTcyZi03ZmZmLTJlZjAtODA2OS0wODE0NmJiY2Y5MmRcIj48c3BhbiBzdHlsZT1cImZvbnQtc2l6ZTogMTFwdDsgZm9udC1mYW1pbHk6IEFyaWFsLCBzYW5zLXNlcmlmOyBiYWNrZ3JvdW5kLWNvbG9yOiB0cmFuc3BhcmVudDsgZm9udC12YXJpYW50OiBub3JtYWw7IHZlcnRpY2FsLWFsaWduOiBiYXNlbGluZTsgd2hpdGUtc3BhY2U6IHByZS13cmFwO1wiPk5vcm1hbCBGbG9hdCA0IChORjQpPC9zcGFuPjwvc3Bhbj4ifSx7ImlkIjoiY2FyZC05MzU4Ny4zMDAwMDAwMDA3NSIsImZyb250IjoiPHNwYW4gaWQ9XCJkb2NzLWludGVybmFsLWd1aWQtYTA5NGMxM2YtN2ZmZi1kNzE2LTg2OGMtZDk2N2JjMmJmZDExXCI%2BPHNwYW4gc3R5bGU9XCJmb250LXNpemU6IDExcHQ7IGZvbnQtZmFtaWx5OiBBcmlhbCwgc2Fucy1zZXJpZjsgYmFja2dyb3VuZC1jb2xvcjogdHJhbnNwYXJlbnQ7IGZvbnQtdmFyaWFudDogbm9ybWFsOyB2ZXJ0aWNhbC1hbGlnbjogYmFzZWxpbmU7IHdoaXRlLXNwYWNlOiBwcmUtd3JhcDtcIj5Gb3JtYXQgcHJlc2lzaSBzdGFuZGFyICg8L3NwYW4%2BPHNwYW4gc3R5bGU9XCJmb250LXNpemU6IDExcHQ7IGZvbnQtZmFtaWx5OiBBcmlhbCwgc2Fucy1zZXJpZjsgYmFja2dyb3VuZC1jb2xvcjogdHJhbnNwYXJlbnQ7IGZvbnQtc3R5bGU6IGl0YWxpYzsgZm9udC12YXJpYW50OiBub3JtYWw7IHZlcnRpY2FsLWFsaWduOiBiYXNlbGluZTsgd2hpdGUtc3BhY2U6IHByZS13cmFwO1wiPnNpbmdsZS1wcmVjaXNpb248L3NwYW4%2BPHNwYW4gc3R5bGU9XCJmb250LXNpemU6IDExcHQ7IGZvbnQtZmFtaWx5OiBBcmlhbCwgc2Fucy1zZXJpZjsgYmFja2dyb3VuZC1jb2xvcjogdHJhbnNwYXJlbnQ7IGZvbnQtdmFyaWFudDogbm9ybWFsOyB2ZXJ0aWNhbC1hbGlnbjogYmFzZWxpbmU7IHdoaXRlLXNwYWNlOiBwcmUtd3JhcDtcIj4pIHlhbmcgbWVtYmVyaWthbiB0aW5na2F0IGFrdXJhc2kgbWF0ZW1hdGlzIHRlcnRpbmdnaSBkYWxhbSBrYWxrdWxhc2kgPC9zcGFuPjxzcGFuIHN0eWxlPVwiZm9udC1zaXplOiAxMXB0OyBmb250LWZhbWlseTogQXJpYWwsIHNhbnMtc2VyaWY7IGJhY2tncm91bmQtY29sb3I6IHRyYW5zcGFyZW50OyBmb250LXN0eWxlOiBpdGFsaWM7IGZvbnQtdmFyaWFudDogbm9ybWFsOyB2ZXJ0aWNhbC1hbGlnbjogYmFzZWxpbmU7IHdoaXRlLXNwYWNlOiBwcmUtd3JhcDtcIj5uZXVyYWwgbmV0d29yazwvc3Bhbj48c3BhbiBzdHlsZT1cImZvbnQtc2l6ZTogMTFwdDsgZm9udC1mYW1pbHk6IEFyaWFsLCBzYW5zLXNlcmlmOyBiYWNrZ3JvdW5kLWNvbG9yOiB0cmFuc3BhcmVudDsgZm9udC12YXJpYW50OiBub3JtYWw7IHZlcnRpY2FsLWFsaWduOiBiYXNlbGluZTsgd2hpdGUtc3BhY2U6IHByZS13cmFwO1wiPiBrb252ZW5zaW9uYWw8L3NwYW4%2BPC9zcGFuPiIsImJhY2siOiI8c3BhbiBpZD1cImRvY3MtaW50ZXJuYWwtZ3VpZC0wMjk3OWI3Zi03ZmZmLWY0NzgtYjk2MC1lOTAzZGIzYzE1YWFcIj48c3BhbiBzdHlsZT1cImZvbnQtc2l6ZTogMTFwdDsgZm9udC1mYW1pbHk6IEFyaWFsLCBzYW5zLXNlcmlmOyBiYWNrZ3JvdW5kLWNvbG9yOiB0cmFuc3BhcmVudDsgZm9udC12YXJpYW50OiBub3JtYWw7IHZlcnRpY2FsLWFsaWduOiBiYXNlbGluZTsgd2hpdGUtc3BhY2U6IHByZS13cmFwO1wiPkZsb2F0aW5nIFBvaW50IDMyIChGUDMyKTwvc3Bhbj48L3NwYW4%2BIn1dLCJpbnN0cnVjdGlvbiI6IktsaWsga2FydHUgdW50dWsgbWVsaWhhdCBqYXdhYmFubnlhLiJ9" height="522" width="100%" allow="fullscreen"></iframe>

***

## Parameter-Efficient Fine-Tuning

Selamat datang di jalur efisiensi cabang kedua, yaitu Parameter-Efficient Fine-Tuning atau PEFT. Jika Quantization adalah seni mengecilkan ukuran parameter model, maka PEFT adalah seni mengecilkan “Beban Kerja” yang perlu dilakukan saat fine-tuning terjadi.

*Eits*, sebelum masuk lebih dalam, ada satu hal yang perlu diklarifikasi, yaitu PEFT bukanlah algoritma atau teknik tunggal. PEFT dapat dikatakan sebagai payung besar dari teknik-teknik yang ide utamanya *“melatih sebagian kecil parameter dan membekukan model utama”*.

Teknik-teknik dalam PEFT memungkinkan kita untuk mengubah perilaku model raksasa hanya dengan melatih kurang dari 1% total parameternya. Ini memungkinkan kita melakukan Fine-tuning di GPU gratisan seperti T4 Google Colab atau GPU laptop gaming sekalipun.

Lantas sebanyak apa sih teknik-teknik PEFT?

Coba Anda perhatikan diagram berikut.

![dos-e69c0de39abf3371cafdd613a5104fe120260330151456.png](https://assets.cdn.dicoding.com/original/academy/dos-e69c0de39abf3371cafdd613a5104fe120260330151456.png)

*Waw*, ternyata PEFT ini memiliki banyak sekali variasi teknik di dalamnya. *Nah*, berdasarkan cara kerjanya, ia dibagi menjadi tiga mazhab besar.

### Additive Methods

Metode ini bekerja dengan cara menyuntikkan parameter baru ke dalam model. Parameter asli model dibekukan, dan kita hanya melatih parameter tambahan tersebut. Meskipun menambah parameter, metode ini justru hemat memori saat training karena kita tidak perlu menyimpan optimizer states untuk miliaran parameter asli, cukup untuk parameter tambahan yang kecil.

Ada dua teknik populer di sini:

*   **Adapters:** Menyisipkan layer kecil (fully connected network) di antara layer-layer Transformer. (Seperti menyisipkan filter air tambahan di pipa utama).
*   **Soft Prompts**: Kita tidak mengubah model, tapi kita memanipulasi input-nya. Kita menambahkan trainable vectors di depan teks input kita agar model "tertipu" dan berperilaku sesuai keinginan kita.

Namun, metode Additive ini punya satu kekurangan krusial, yaitu waktu. Cara kerjanya yang menambahkan lapisan baru ini ibarat memasang polisi tidur di jalan tol yang membuat aliran data menjadi sedikit terhambat. Dampaknya model jadi lebih lambat saat dijalankan karena Inference Latency yang besar.

### Selective Methods

Metode ini tidak menambah apa-apa. Ia hanya memilih sebagian kecil dari parameter asli model untuk dilatih, sementara sisanya dibekukan.

Cara memilihnya bermacam-macam:

*   **Top Layer Fine-Tuning:** Hanya melatih beberapa layer terakhir (dekat output), karena layer awal biasanya hanya menangkap fitur bahasa dasar yang sudah bagus.
*   **Specific Parameter (Bias-only):** Hanya melatih komponen Bias di setiap layer, bobot utamanya (Weights) dibekukan.
*   **Sparse Updates:** Memilih neuron-neuron tertentu yang dianggap paling "aktif" atau penting. (Catatan: Teknik ini menjanjikan tapi seringkali berat secara komputasi karena komputer harus bekerja keras mencari mana neuron yang penting itu).

Metode selective memang tidak menambahkan layer baru pada model, tetapi metode ini tetap memiliki tantangan yang cukup *tricky*. Umumnya, keberhasilan metode ini bergantung pada kemampuan menemukan bagian dari model yang tepat untuk dilakukan tuning agar hasilnya maksimal.

### Reparameterization-Based Methods

Inilah kategori yang paling menarik perhatian saat ini. Metode ini memanfaatkan fakta bahwa Neural Network itu sebenarnya sangat redundan (boros). Kita bisa mengompres perubahan bobot model menggunakan representasi Low-Rank (Pangkat Rendah).

*   **LoRA (Low-Rank Adaptation):** Menggunakan dekomposisi matriks (Matrix Decomposition) untuk merepresentasikan update bobot. Alih-alih melatih matriks raksasa, kita melatih dua matriks kecil yang jika dikalikan menghasilkan matriks raksasa tersebut.
*   **Intrinsic SAID:** Menggunakan transformasi matematika (Fastfood transform) untuk representasi update yang lebih efisien lagi.

Mengapa kategori ini spesial?

Metode seperti LoRA sangat disukai karena ia menyeimbangkan segalanya: storage hemat, VRAM hemat, dan tidak menambah *latency* (kelambatan) saat model dijalankan karena hasil latihannya bisa digabung (*merge*) kembali ke model asli.

*Wait*, apakah cara yang seperti *“cheat code”* ini sungguh nyata? Mari kita cari tahu lebih lanjut di materi berikutnya.

***

## Deep Dive Low-Rank Adaptation (LoRA)

Jika Anda bertanya kepada beberapa AI Engineer atau praktisi AI hari ini *“Metode apa yang paling sering kalian gunakan untuk fine-tuning?”*, 9 dari 10 orang akan menjawab LoRA atau mungkin variannya.

*Nah*, sebelum masuk lebih dalam ke variannya, mari kita berkenalan dengan LoRA itu sendiri.

Seperti yang kita pahami, fine-tuning itu bukanlah melatih model dari awal, tetapi mengadaptasi model yang sudah pintar ke spesifik case. Ibarat Anda memiliki sebuah “Ensiklopedia” Raksasa (Pre-trained Model) yang berisi semua pengetahuan dunia. Lalu, Anda ingin menambahkan satu bab khusus tentang “Resep Masakan Padang” ke dalamnya.

Cara lama seperti Full Fine-Tuning memaksa Anda untuk menulis ulang seluruh ensiklopedia tersebut dari halaman 1 sampai selesai, hanya demi menyisipkan resep rendang. Ini sangat boros tinta, kertas, dan waktu.

Peneliti dari Microsoft dalam *paper*-nya yang berjudul *“LoRA: Low-Rank Adaptation of Large Language Models”* (Hu et al., 2021) seperti bertanya *“Bisakah kita menyisipkan catatan kecil saja tanpa merusak buku aslinya?”* 

Dari sinilah lahir LoRA. Mari kita bedah konsepnya di bawah ini.

### Memahami Hipotesis “Low-Rank”

Untuk memahami LoRA, tentu hal pertama yang perlu Anda pahami secara dalam adalah *“apa maksud dari Low-Rank?”*.

Seperti yang kita ketahui bersama, LLM maupun SLM itu memiliki miliaran hingga triliunan parameter. Alasan model-model tersebut memiliki miliaran parameter adalah agar ia bisa menjadi ahli di ribuan tugas berbeda sekaligus. Kemampuan generalis inilah yang membuat Gemini atau ChatGPT dapat menjawab berbagai topik pertanyaan dan perintah yang diberikan.

Dalam fine-tuning, tujuan kita adalah melatih model untuk melakukan tugas yang spesifik, seperti contohnya “Resep Masakan Padang”.

*Nah*, paper berjudul *"Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning"* yang ditulis Aghajanyan menemukan fakta mengejutkan berikut.

*“Ketika diajari skill baru, model tidak benar-benar menggunakan 1 Miliar neuronnya untuk berubah. Sebaliknya, perubahan yang signifikan sebenarnya hanya terjadi di jalur-jalur tertentu yang sangat spesifik.”* — Aghajanyan et al., 2020

Artinya, untuk mengubah model yang *"pintar segalanya"* menjadi *"ahli masakan Padang"*, kita tidak perlu mengubah miliaran struktur otak model tersebut secara rumit. kita cukup fokus mencari matriks tambahan yang merepresentasikan transisi keahlian tersebut.

Dengan kata lain, kita hanya perlu melatih pengetahuan barunya saja, tanpa perlu mengubah atau menghitung ulang matriks raksasa yang menampung seluruh pengetahuan umum model.

![dos-5764e6d25f401f3f948773a949fcb7a020260430084459.gif](https://assets.cdn.dicoding.com/original/academy/dos-5764e6d25f401f3f948773a949fcb7a020260430084459.gif)

Berdasarkan ilustrasi di atas, matriks yang ada di Update Knowledge jauh lebih kecil daripada di Knowledge Model. Sehingga, memperbarui nilai dalam matriks tersebut jauh lebih ringan daripada memperbarui miliaran matriks di knowledge model.

Itulah ide dari pendekatan Low-Rank.

*   **Full-Rank:** Berasumsi bahwa setiap parameter itu penting, independen, dan harus diubah secara bebas agar model bisa belajar. Saat belajar atau di-*training* pada data baru, model harus menggunakan 1 miliar parameternya untuk dihitung perubahannya.
*   **Low-Rank:** Berasumsi bahwa perubahan yang dibutuhkan untuk tugas spesifik sebenarnya sederhana dan memiliki pola sehingga tidak perlu kebebasan penuh. Dengan cara ini, kita hanya perlu meng-*update* jutaan parameter di matriks kecil untuk menghasilkan dampak yang setara dengan mengupdate miliaran parameter di model asli.

Namun, apakah seberpengaruh itu penggunaan Low-Rank? *Alright*, perhatikan ilustrasi di bawah ini.

![dos-0c1ab88d8e68acccf834d3b9498ddcb220260430084539.png](https://assets.cdn.dicoding.com/original/academy/dos-0c1ab88d8e68acccf834d3b9498ddcb220260430084539.png)

Jika Anda bertanya mengenai “r”, kita akan membahasnya setelah ini. Untuk saat ini, Anda cukup tahu terlebih dahulu jika itu adalah Rank.

Saat kita memiliki matriks 10.000 × 10.000 dan melakukan full fine-tuning dengan full rank seperti pada matriks kiri, akan ada 100 juta perhitungan delta weight yang perlu dilakukan. Apa itu delta weight? Itu merupakan selisih bobot atau dapat dikatakan besar update bobot yang terjadi pada parameter model.

Sekarang coba bandingkan dengan perhitungan matriks di sebelahnya. Jumlah perhitungan yang perlu dilakukan menjadi sekitar 5.000 kali lebih sedikit, bukan? *Yup*, dengan begini proses fine-tuning dapat terjadi dengan lebih cepat dan lebih ringan secara memori tentunya.

*Wait*, jika nantinya kita hanya perlu meng-*update* matriks parameter kecil tersebut lantas apa yang terjadi dengan parameter model yang sesungguhnya? Lalu, dari mana asalnya matriks kecil ini?

Untuk menjawabnya, kita perlu membahas *flow* yang terjadi saat model di-fine-tuning menggunakan LoRA.

### Alur Kehidupan Data dalam LoRA

Sebelum kita menjelajahi alur dari LoRA, mari kita melihat gambaran dari blueprint arsitekturnya melalui ilustrasi di bawah ini.

![dos-c6057d3cf15a5bb251c03ddd31c3193620260430084637.gif](https://assets.cdn.dicoding.com/original/academy/dos-c6057d3cf15a5bb251c03ddd31c3193620260430084637.gif)

Di sisi kiri (kotak biru), kita melihat Pre-trained Weights yang menyimpan miliaran parameter model. Perhatikan label 'Freeze' di sana, itu artinya selama pross latihan nanti, seluruh parameter atau Weight di dalam Pre-trained model tidak akan diintervensi.

Keajaibannya justru terjadi di sisi kanan (kotak oranye). Inilah modul LoRA kita. Perhatikan dua hal unik pada inisialisasinya.

*   **Matriks A (Bawah)** diisi dengan angka Random Gaussian.
*   **Matriks B (Atas)** sengaja diisi dengan angka nol (0).

Mengapa nol? Dengan membuat B bernilai nol, kita menjamin bahwa di detik pertama pelatihan, modul LoRA tidak melakukan apa-apa sehingga ia belum mengubah perilaku model asli sedikit pun.

Jadi, gambar ini menggambarkan posisi “Garis Start”. Model asli aman terjaga, sementara modul LoRA sudah terpasang dan siap untuk mulai belajar pelan-pelan dari angka nol.

*Nah*, sekarang apakah Anda penasaran mengapa untuk melakukan LoRA kita membutuhkan dua matriks yang berbeda? Jawabannya ada saat kita mengeksplorasi Down-Projection dan Up-Projection.

#### Down-Projection

Data input (x) yang diproses oleh LLM akan selalu memiliki dimensi yang sangat tinggi, umumnya berukuran 4096. Artinya, setiap kata yang masuk ke sini membawa 4.096 atribut informasi. Ada informasi tentang tata bahasa, fakta sejarah, emosi manusia, maupun konteks fisika, biologi, dan masih banyak lainnya.

Semuanya campur aduk dan inilah yang disebut sebagai High-Dimensional Input.

Seperti yang kita tekankan di awal, tujuan kita melakukan Fine-Tuning adalah tugas yang spesifik. Saat kita ingin model ini jago coding Python, apakah kita butuh informasi tentang "Biologi" atau "Fakta Sejarah Romawi"? Tentu saja tidak. Itu semua adalah *noise* yang tidak perlu untuk tugas coding.

*Nah*, hadirlah matriks A yang meringkas High-Dimensional Input tersebut menjadi matriks yang jauh lebih ringkas sebesar Rank (r). Proses tersebutlah yang disebut sebagai Down-Projection. Matriks A ini adalah sebuah matriks yang "kurus tinggi" berukuran Input Dimension × Rank. Coba Anda perhatikan ilustrasi di bawah ini

![dos-590fa046390f03e29bb484d76b26651820260430084826.gif](https://assets.cdn.dicoding.com/original/academy/dos-590fa046390f03e29bb484d76b26651820260430084826.gif)

Saat vektor input (4096 angka) menabrak Matriks A, terjadi operasi matematika perkalian matriks yang brutal. Matriks A akan mengalikan setiap atribut input dengan bobot tertentu, lalu menjumlahkannya menjadi hanya 2 angka saja.

Setelah mendapatkan "intisari murni" 2 angka ini, barulah kita butuh Matriks B (Up-Projection) untuk menerjemahkan intisari tersebut kembali ke bahasa yang dimengerti oleh otak besar model.

#### Up-Projection

Di tahap ini, kita telah memegang sebuah vektor yang sangat kecil, hanya 2 angka (Rank = 2). Ini adalah Latent Representation dari input user tadi.

Masalahnya, otak model asli (Pre-trained Weights) tidak mengerti bahasa “2 angka” ini. Otak model itu didesain untuk memproses 4096 angka sekaligus. Jika kita menyodorkan 2 angka ini langsung ke model utama, dia akan bingung, *"Ini data apa? Kok kepotong-potong?"*.

Oleh karena itu, kita harus mengembalikan bentuk fisik dari vektor 2 angka tersebut ke bentuk awalnya, yaitu 4096. Proses yang disebut Up-Projection ini akan dikerjakan oleh Matriks B. *Nah*, karena sifatnya yang berlawanan, struktur Matriks B pun mencerminkan kebalikan dari matriks pada down projection. Secara fisik, Matriks B bentuknya pendek dan lebar berukuran Rank × Output Dimension.

![dos-00cce83ba56308adbd45bff8e9f8e90820260430084927.gif](https://assets.cdn.dicoding.com/original/academy/dos-00cce83ba56308adbd45bff8e9f8e90820260430084927.gif)

Tugas matematikanya sederhana, ia mengambil vektor kecil (2 angka) tadi, lalu mengalikannya sedemikian rupa sehingga ia mekar kembali menjadi ukuran aslinya (4096 angka). *Eits*, jangan salah sangka, Up-Projection bukan sekadar "meng-*copy paste*" angka 2 kali supaya jadi banyak. Angka-angka pengali di dalamnya akan terus diperbarui selama proses training berjalan.

*Nah*, ngomong-ngomong training, sekarang kita akan mencari tahu jawaban *“bagaimana caranya sih angka-angka di dalam otak model berubah saat fine-tuning setelah kita suntikkan LoRA?”*.

### Update Weight dalam LoRA

Kita sering mendengar istilah fine-tuning, tapi sebenarnya apa yang terjadi pada angka-angka di dalam otak model tersebut? Bagaimana mungkin model yang bobot aslinya sudah kita bekukan (*freeze*) tiba-tiba bisa jadi pintar dalam tugas baru?

Jawabannya terletak pada sebuah konsep matematika sederhana, yaitu delta W (∆W) atau lebih mudahnya disebut Update Weight.

Di bagian ini, kita akan membongkar “kotak hitam” LoRA. Kita tidak lagi hanya melihat panah-panah alur data, tapi bukti fisik angkanya. Kita akan menyaksikan bagaimana dua matriks kecil (A dan B) yang kita bahas tadi, dapat memperbarui nilai dalam Pre-trained Weight model.

#### Mencari Delta W (∆W)

Jika kita mengingat kembali pengetahuan mengenai deep learning, parameter dalam model seperti weight dan bias itu akan diperbarui secara bertahap saat masuk ke dalam fase training. *Nah*, perubahan dari nilai awal ke nilai baru pada setiap langkahnya dapat kita sebut sebagai delta (∆).

Jika kita ingin lebih dalam lagi, perubahan atau delta tersebut terjadi dari perkalian matriks yang berisi dari seluruh parameter yang ada di dalam model tanpa terkecuali. *Nah*, LoRA akan lebih memilih perhitungan matriks yang jauh lebih kecil, yaitu matriks A dan matriks B yang sedari tadi kita bicarakan.

Agar konsep ini lebih mudah dipahami, mari kita hitung delta W pada sebuah pre-trained weight. Bayangkan Anda sedang melakukan PEFT dengan LoRA pada sebuah pre-trained model dengan total 16 parameter weight. Angka 16 dipilih agar perhitungan lebih sederhana untuk divisualisasikan.

![dos-ba8b76cf374b1bfcf39e4c02d6fa1e4e20260430085001.png](https://assets.cdn.dicoding.com/original/academy/dos-ba8b76cf374b1bfcf39e4c02d6fa1e4e20260430085001.png)

Seperti yang kita sudah bahas sebelumnya, LoRA akan membekukan matriks asli model dan kita akan fokus meng-*update* matriks tambahan di sampingnya yang lebih kecil. Artinya dari epoch pertama hingga epoch terakhir selesai, dua matriks inilah yang nilainya akan diperbarui oleh optimizer.

*   **Matriks A (4 × 2):** Matriks Down-Projection.
*   **Matriks B (2 × 4):** Matriks Up-Projection.

Sehingga, optimizer tidak perlu repot-repot meng-*update* nilai dalam matriks Pre-trained Model yang sangat banyak itu.

*Nah*, setelah optimizer selesai memperbarui nilai dalam matriks A dan B yang lebih kecil, kita perlu mengonversinya menjadi bentuk yang serupa dengan matriks Pre-trained Weight. Bagaimana caranya? Tentu saja dengan cara mengalikan kedua matriks kecil tersebut sehingga menghasilkan delta Weight yang kita bicarakan sebelumnya. Coba perhatikan ilustrasi berikut.

![dos-b2d5df019f5a47cb55595f188991699420260430085033.gif](https://assets.cdn.dicoding.com/original/academy/dos-b2d5df019f5a47cb55595f188991699420260430085033.gif)

Sampai di sini, kita akhirnya berhasil mendapatkan Matriks Delta Weight (∆W). Anggap saja matriks ini sebagai "lembar koreksi" yang berisi detail seberapa banyak nilai parameter harus digeser agar model menjadi pintar.

Pertanyaan terakhirnya, setelah memiliki model asli yang sedang "tidur" atau frozen beserta lembar koreksinya, bagaimana cara kita menyatukan keduanya agar model tersebut benar-benar berubah?

*Nah*, kita akan masuk ke tahap selanjutnya.

#### Penyuntikan ke Model Asli

Jawabannya ternyata sangat sederhana, yaitu cukup dengan menjumlahkannya. Kita tidak perlu membongkar matriks model yang sedang ter-*frozen* tadi karena yang perlu kita lakukan adalah menambahkan **pengetahuan lama** (W) dengan **koreksi baru** (∆W) secara langsung.

Mari kita perhatikan ilustrasi berikut ini.

![dos-fd36634745af4a926aa04ba45859113020260430085100.gif](https://assets.cdn.dicoding.com/original/academy/dos-fd36634745af4a926aa04ba45859113020260430085100.gif)

*Yup*, proses di atas akan diiterasi terus menerus hingga proses fine-tuning selesai. Matriks “W awal” merepresentasikan pengetahuan lama, sementara matriks “W baru” merepresentasikan pengetahuan yang sudah diperkaya.

Saat ini, Anda sudah memahami konsep LoRA secara mendalam. Namun, saat Anda duduk di depan “kodingan” nanti, Anda tidak bisa sekadar memerintahkan komputer *"Hei, tolong pasang LoRA!"* 

Mungkin komputer akan balik bertanya kepada Anda. *"Oke, LoRA yang mana? Seberapa kecil matriks yang ingin digunakan? Seberapa besar pengaruhnya? Bagian mana saja yang mau dipasangi LoRA?"* 

Untuk menjawab pertanyaan-pertanyaan ini, kita harus masuk ke ruang kendali. Kita akan mempelajari Hyperparameters yang menentukan “nasib” training Anda.

### Key Hyperparameter

Sebelum membedah satu per satu hyperparameter-nya, kita perlu meluruskan satu hal tentang tools yang akan dipakai.

Di luar sana, mungkin Anda mendengar tentang Library canggih seperti Unsloth yang bisa membuat proses fine-tuning secepat kilat. Namun, untuk tahap fondasi ini, kita akan menggunakan *library* versi *native*-nya, yaitu peft dari HuggingFace. Alasannya jelas, yaitu agar Anda memahami konsep mentahnya.

Oleh karena itu, dengan menguasai peft, Anda bisa menggunakan *wrapper* apa pun di masa depan. *Nah*, di dalam konfigurasi peft ini, ada 4 “Tombol Utama” yang wajib Anda pahami fungsinya. Pada implementasi kode, keempat hyperparameter tersebut nantinya akan dibungkus dalam LoRA config.

```python
from peft import get_peft_model, LoraConfig
config = LoraConfig(...,
                    ...,
                    ...,
                    ...)
model_lora = get_peft_model(model_asli, config)
```

Selanjutnya, mari kita isi bagian-bagian kosong dalam kode di atas dengan mengeksplorasinya satu per satu.

#### Rank (r)

Seharusnya, Anda sudah tidak asing dengan hyperparameter ini. Rank adalah representasi dari hyperparameter yang menentukan seberapa besar dimensi “penyempitan” di antara Matriks A dan Matriks B. Anda dapat memperhatikan ilustrasi berikut untuk mengingat kembali.

![dos-ace6fc08cc12d4c576e6d4d806a3231020260430085131.png](https://assets.cdn.dicoding.com/original/academy/dos-ace6fc08cc12d4c576e6d4d806a3231020260430085131.png)

Sekarang pertanyaannya, apa implikasi dari nilai r ini terhadap performa dan perilaku model?

Karena ukuran matriks A dan B ditentukan oleh nilai rank, rank secara langsung menentukan jumlah parameter yang dapat dilatih. Artinya nilai rank berbanding lurus dengan seberapa banyak informasi baru atau seberapa kompleks pola yang bisa dipelajari oleh adapter.

*   **Rank Rendah:** Ini akan memaksa model untuk memadatkan informasi ke dalam ruang dimensi yang sangat sempit. Sehingga, Matriks A dan B terpaksa hanya mengekstrak dan mempelajari fitur-fitur yang paling esensial dan dominan dari data Anda.
*   **Rank Tinggi:** Nah, ini memberikan model ruang kebebasan yang lebih luas. Model memiliki lebih banyak parameter untuk menangkap detail, nuansa linguistik yang rumit, atau informasi spesifik yang mungkin terlewatkan jika rank terlalu kecil.

Dalam paper asli LoRA yang dirilis oleh Hu*, et al.,* mereka menemukan sebuah fakta yang disebut Low Intrinsic Dimension. Mereka membuktikan bahwa adaptasi bobot untuk banyak tugas bahasa alami tingkat hilir (seperti terjemahan, klasifikasi, atau gaya percakapan) hanya membutuhkan rank yang sangat kecil.

Bahkan dalam beberapa eksperimen, r=2 sudah cukup untuk menyamai performa full fine-tuning. Berikut sedikit panduan bagi Anda saat mengonfigurasi peft.

*   **r = 8** **atau** **16**: Titik awal yang paling direkomendasikan untuk tugas instruct-tuning standar, seperti mengubah gaya bahasa, atau klasifikasi. Titik ini menawarkan keseimbangan terbaik antara efisiensi memori dan kualitas output.
*   **r = 32** **atau** **64**: Digunakan ketika Anda memasukkan domain pengetahuan yang benar-benar asing bagi base model. Misalnya, mengajari model bahasa Inggris murni untuk memahami tata bahasa daerah tertentu atau untuk tugas reasoning matematika yang kompleks.
*   **r > 64**: Jarang direkomendasikan kecuali Anda memiliki dataset yang masif dan melakukan tahapan Continuous Pre-Training (CPT), bukan sekadar fine-tuning biasa.

Mungkin Anda akan bertanya, mengapa nilai r-nya harus menggunakan bilangan berpangkat dua (8, 16, 32, 64)?

Alasannya ada pada Efisiensi Komputasi Tingkat Rendah (Low-Level Optimization). Secara matematis murni, algoritma LoRA sebenarnya tidak melarang Anda menggunakan angka ganjil atau tidak berpola, seperti Rank 7, 13, atau 25. LoRA akan tetap berjalan dan model tetap bisa belajar.

Namun, di dunia Machine Learning Engineering, kita tidak hanya berurusan dengan matematika di atas kertas, tetapi juga dengan perangkat keras (hardware) dan perangkat lunak (software) yang mengeksekusi matematika tersebut.

*Nah*, GPU modern seperti NVIDIA yang menjadi standar industri AI dirancang dengan arsitektur yang sangat bergantung pada struktur Pangkat Dua. Oleh karena itu, memilih deret kelipatan 8, 16, 32, dan 64 bukan karena batasan rumus LoRA, melainkan untuk “menyenangkan” Arsitektur Silikon GPU dan Compiler CUDA.

Menggunakan dimensi yang tidak mematuhi aturan perangkat keras akan menyebabkan melambatnya waktu training, penggunaan VRAM yang tidak efisien akibat padding, dan hilangnya optimalisasi tingkat perangkat keras. Anda dapat menggunakan tautan berikut jika ingin membaca lebih detail mengenai pangkat dua dalam GPU tersebut: [Matrix Multiplication](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html) dan [Optimizing GPU](https://developer.nvidia.com/blog/optimizing-gpu-performance-tensor-cores/).

Karena kita sudah mengeksplorasi Rank, kita akan melihat implementasinya dalam kode.

```python
from peft import get_peft_model, LoraConfig
config = LoraConfig(r = 8)
model_lora = get_peft_model(model_asli, config)
```

*Nice*, kita sudah mendapatkan 1 hyperparameter. Selanjutnya, mari kita lanjutkan eksplorasi ini untuk melengkapi hyperparameter-nya.

#### Alpha (∝)

Tahukah Anda bahwa LoRA ini memiliki tombol volume yang dapat digunakan untuk *“seberapa keras suaranya? Apakah ia hanya berbisik, atau ia berteriak mendominasi jalan utama?”*.

*Yup*, Alpha adalah hiperparameter yang berperan dalam menentukan seberapa besar parameter yang dalam Matriks A dan Matriks B ini memengaruhi Output atau jawaban.

*Alright*, agar lebih terbayang, mari kita ingat kembali rumus “penyuntikan” parameter baru sebelumnya dan memperhatikan ilustrasi di bawah ini.

![dos-34acd31dfbebe93ef61979c6d84cff8420260430085201.gif](https://assets.cdn.dicoding.com/original/academy/dos-34acd31dfbebe93ef61979c6d84cff8420260430085201.gif)

Seperti yang kita lihat dan sudah pelajari bersama, LoRA akan menghasilkan dua output pada setiap iterasinya, satu dari Weight asli (W), dan satu lagi dari adapter atau Delta Weight (∆W).

*   **Jika Alpha diset terlalu rendah:** Pengetahuan baru yang susah payah dipelajari di adapter akan tenggelam oleh pengetahuan raksasa dari model dasar. Model Anda seolah-olah tidak mendengar terhadap fine-tuning yang Anda lakukan.
*   **Jika Alpha diset terlalu tinggi:** Pengetahuan baru ini akan menimpa dan merusak memori asli model. Model Anda mungkin jago menjawab tugas baru, tapi mendadak lupa cara berbicara dengan tata bahasa yang benar (Catastrophic Forgetting).

Oleh karena itu, jika Anda melihat model yang "bandel" (tidak mau mengikuti gaya bahasa atau format instruksi baru yang Anda ajarkan), salah satu solusinya adalah menaikkan Alpha agar instruksi baru Anda lebih lantang mengalahkan sifat asli model tersebut.

Sisa satu pertanyaan, rasio apa yang ideal antara Alpha dan Rank?

Di awal kemunculan paper LoRA, ini sempat menjadi perdebatan. Namun, dari ribuan eksperimen yang dilakukan komunitas AI selama dua tahun terakhir, muncul sebuah konsensus praktis.

*   **Aturan 2×:** Ini yang paling umum digunakan, yaitu dengan “menyetel” nilai Alpha dua kali lipat dari nilai Rank.
*   **Aturan 1×:** Untuk kestabilan ekstra, Anda dapat mengatur nilai Alpha sama dengan nilai Rank.

Satu hal yang perlu Anda ingat, jangan pernah mengatur Alpha lebih kecil dari Rank. Ini akan menghasilkan skala di bawah 1.0, yang membuat sinyal adapter (Matriks A dan B) melemah dan proses training menjadi sangat lambat dan tidak efisien.

*Alright*, eksplorasi hyperparameter kedua berakhir di sini. So, mari kita tambahkan hyperparameter ini ke kode konfigurasi sebelumnya.

```python
from peft import get_peft_model, LoraConfig
config = LoraConfig(r = 8,
lora_alpha = 32,)
model_lora = get_peft_model(model_asli, config)
```

Nah, kita sekarang hanya punya “PR” satu hyperparameter utama lagi, yaitu Target Modules.

#### Target Modules

Sebelum memulai ke Target Modules, Anda perlu ingat bahwa LLM seperti Llama-3 dibangun dari tumpukan blok yang disebut Transformer Block atau layers.

Kita akan *recall* pelajaran di modul awal mengenai arsitektur Transformer. Seperti yang kita pelajari, setiap blok Transformer terdiri dari beberapa komponen layer, yaitu Multi-Head Self-Attention, Feed-Forward Network (FFN), Layer Normalization, dan Residual Connections.

*Nah*, tahukah Anda, saat melakukan LoRA, kita bisa memilih untuk menerapkannya di seluruh komponen dalam arsitektur Transformer dari LLM atau hanya sebagian. Keputusan "titik penanaman" ini diatur oleh parameter yang bernama Target Modules.

##### Komponen Attention

Ini adalah komponen yang membantu model untuk memahami hubungan antarkata. Di sinilah model tahu bahwa kata "Bank" dalam kalimat "Bank Mandiri" berbeda maknanya dengan "Bank Sampah".

Oleh karena itu, Anda dapat menargetkan modul Attention jika tujuan dari fine-tuning model adalah untuk mengubah perilakunya dalam memproses instruksi dan merangkai kata, misalnya gaya bahasa.

Mari kita lihat ilustrasi berikut ini.

![dos-207bcc4915cdbe4a86b4c6a1b5acb18320260330151806.png](https://assets.cdn.dicoding.com/original/academy/dos-207bcc4915cdbe4a86b4c6a1b5acb18320260330151806.png)

Seperti yang sudah dibocorkan dalam ilustrasi di atas, Attention memiliki empat modul matriks utama yang saling bekerja sama.

*   **q_proj** **(Query)**: Kata yang sedang "bertanya" atau aktif mencari konteks dari kata-kata di sekitarnya.
*   **k_proj** **(Key):** Kata-kata lain yang "menjawab" dengan memberikan petunjuk konteks untuk si Query.
*   **v_proj** **(Value):** Makna atau nilai esensial yang sebenarnya dibawa oleh kata tersebut.
*   **o_proj** **(Output):** Hasil racikan akhir yang menggabungkan pemahaman Query, Key, dan Value menjadi satu kesatuan makna yang utuh dan akurat.

Setelah data selesai diracik konteksnya di kotak Multi-Head Attention, aliran data akan bergerak naik menuju kotak biru bernama Feed-Forward.

##### Komponen Feed-Forward Network

Jika komponen Attention fokus pada memahami "konteks" antarkata, komponen Feed-Forward memiliki peran yang sama sekali berbeda. Para peneliti meyakini bahwa area ini berfungsi sebagai pusat penyimpanan informasi. Di sinilah tertanamnya sebagian besar memori, fakta, logika, dan "pengetahuan dunia" dari model.

Apabila target fine-tuning menuntut model untuk mempelajari pengetahuan teknis baru atau menyerap kumpulan fakta spesifik, Anda bisa menargetkan modul matriks yang ada di komponen ini.

![dos-3758eafd02adbfa028eb48ad0539c55620260430085325.png](https://assets.cdn.dicoding.com/original/academy/dos-3758eafd02adbfa028eb48ad0539c55620260430085325.png)

*Nah*, dalam sektor FFN ini terdapat tiga target modules yang dapat Anda gunakan.

*   **gate_proj**: Berfungsi sebagai mekanisme filter yang menyeleksi informasi mana yang relevan untuk dilewatkan dan mana yang harus ditahan.
*   **up_proj:** Memproyeksikan data ke ruang dimensi yang jauh lebih besar (upscaling). Tujuannya agar model memiliki ruang komputasi yang luas untuk memecahkan dan memproses informasi yang kompleks.
*   **down_proj**: Setelah pemrosesan di dimensi besar selesai, matriks ini bertugas memadatkan kembali data tersebut ke dimensi asalnya agar siap dikirim ke layer model selanjutnya.

Alright kita sudah membahas seluruh modules yang memiliki potensi untuk diterapkan LoRA di dalamnya. Sekarang, bagaimana cara kita mengontrol di mana LoRA akan ditanam?

Semuanya diatur lewat parameter target_modules di dalam kode PEFT.

Sebagai contoh, mari kita ambil matriks dari departemen Attention yang sudah kita bahas sebelumnya. Ketika menuliskan `target_modules = ["q_proj", "v_proj", “down_proj”]` di dalam kode, Anda pada dasarnya sedang memberikan instruksi spesifik seperti berikut ini kepada sistem.

*"Hei, tolong pasang adapter LoRA (Matriks A dan B) di kotak Linear milik Q dan V, lalu pasang juga di kotak down projection saja ya. Kotak yang lain biarkan standar pabrik!"* 

*Nah*, justru muncul pertanyaan baru. Karena kita sekarang tahu bahwa LoRA bisa dipasang di beberapa tempat saja, di mana tempat yang harus dipasangi LoRA agar untuk mendapatkan performa paling optimal?

Saat LoRA pertama kali diciptakan dalam paper aslinya, para peneliti menyarankan untuk memasang LoRA HANYA di `q_proj` dan `v_proj`. Alasannya ada di dua sisi, keseimbangan performa, dan memori.

*   **Performa:** Para peneliti beranggapan bahwa Attention merupakan komponen paling krusial untuk beradaptasi dengan instruksi baru.
*   **Memori:** *Nah*, dengan hanya menempelkan di dua titik, VRAM yang dihemat sangatlah luar biasa. Jumlah parameter yang dilatih mungkin hanya sekitar 0.1% dari total parameter model.

Namun, jika Anda ingin model melakukan sesuatu yang fundamentalnya tidak ada dalam ingatan pre-training-nya seperti bahasa yang benar-benar baru. Pendekatan hemat parameter ini kemungkinan besar tidak akan cukup kuat untuk menampung ilmu baru tersebut.

Sayangnya, dalam paper yang ditulis oleh Dettmers et al. (2023) muncul PoV baru pada pendekatan baru bernama QLoRA. *Yup*, pendekatan ini sudah pernah keluar dari mulut kita bersama.

Paper QLoRA menemukan bahwa membatasi adaptasi hanya pada modul Query dan Value tidak cukup untuk menyamai performa full finetuning (16-bit) saat menggunakan base model yang dikuantisasi (4-bit)

Oleh karena itu, lahirlah *best practice* saat ini dengan memasang adapter LoRA pada semua layer di dalam blok transformer. Pada konfigurasi target_modules, Anda akan menargetkan seluruhnya `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`.

Dengan cara ini, adapter LoRA bisa mengubah cara model bernalar sekaligus menyuntikkan fakta baru secara bersamaan.

*Nah*, berhubung eksplorasi kita dalam target_modules telah usai, mari kita lengkapi kode konfigurasi hyperparameter LoRA sebelumnya.

```python
from peft import get_peft_model, LoraConfig
config = LoraConfig(r = 8,
lora_alpha = 32,
target_modules = ["q_proj",
                  "k_proj",
                  "v_proj",
                  "o_proj",
                  "gate_proj",
                  "up_proj",
                  "down_proj"])
model_lora = get_peft_model(model_asli, config)
```

Ngomong-ngomong soal efisiensi, kita terus menyinggung nama QLoRA, tapi belum sempat membedahnya secara khusus. Daripada terus-terusan berkutat dengan teori, bagaimana kalau kita langsung berkenalan dengannya lewat ngoding?

*Yup*, tentu saja kita akan melakukan. Namun, sebelum melangkah kesana, mari kita kilas balik sejenak komponen-komponen LoRA yang telah kita pelajari melalui aktivitas interaktif berikut ini agar kita tidak tersesat saat praktik lapangan nantinya.

<iframe src="https://academy-sandpack.dicoding.dev/activities/fill-in-the-blank?data=eyJ0ZW1wbGF0ZSI6IkRhbGFtIHByb3NlcyBmaW5lLXR1bmluZyBtZW5nZ3VuYWthbiB0ZWtuaWsgTG9SQSwga2l0YSB0aWRhayBtZWxhdGloIHVsYW5nIHNlbHVydWggcGFyYW1ldGVyIG1vZGVsLCBtZWxhaW5rYW4gaGFueWEgbWVueWlzaXBrYW4gbWF0cmlrcyBhZGFwdGVyIHBhZGEgbGFwaXNhbiBhcnNpdGVrdHVyIHRlcnRlbnR1IChzZXBlcnRpIEF0dGVudGlvbiBsYXllcikgeWFuZyBzZWNhcmEgc3Blc2lmaWsga2l0YSBkZWZpbmlzaWthbiBkaSBkYWxhbSBwYXJhbWV0ZXIgezF9LiBBZGFwdGVyIGluaSBtZW1lY2FoIHBlbWJhcnVhbiBib2JvdCBtZW5qYWRpIGR1YSBsYW5na2FoIG1hdHJpa3MuIExhbmdrYWggcGVydGFtYSBhZGFsYWggbWVuZ29tcHJlc2kgZGltZW5zaSBpbnB1dCBtZW5qYWRpIHNhbmdhdCBrZWNpbCBtZWxhbHVpIHRhaGFwIHsyfS4gU2V0ZWxhaCBpdHUsIHJlcHJlc2VudGFzaSB5YW5nIHN1ZGFoIGRpa29tcHJlc2kgdGVyc2VidXQgZGlrZW1iYWxpa2FuIGxhZ2kga2UgZGltZW5zaSBhc2xpbnlhIG1lbGFsdWkgdGFoYXAgezN9LlxuXG5TZWJlcmFwYSBrZWNpbCBkaW1lbnNpIGJvdHRsZW5lY2sgZGkgYW50YXJhIGtlZHVhIG1hdHJpa3MgdGVyc2VidXQgZGlhdHVyIG9sZWggc2VidWFoIHBhcmFtZXRlciBiZXJuYW1hIHs0fSwgeWFuZyBha2FuIHNhbmdhdCBtZW5lbnR1a2FuIGp1bWxhaCBwYXJhbWV0ZXIgeWFuZyBkaWxhdGloLiBUZXJha2hpciwgdW50dWsgbWVuZW50dWthbiBzZWJlcmFwYSBrdWF0IHBlbmdhcnVoIGRhcmkgYWRhcHRlciB0ZXJzZWJ1dCwga2l0YSBtZW5nZ3VuYWthbiBwYXJhbWV0ZXIgezV9IHNlYmFnYWkgc2NhbGluZyBmYWN0b3IuXG4iLCJhbnN3ZXJzIjpbInRhcmdldCBtb2R1bGVzIiwiZG93biBwcm9qZWN0aW9uIiwidXAgcHJvamVjdGlvbiIsInJhbmsiLCJhbHBoYSJdLCJoaW50IjoiIiwic3RvcmFnZUtleSI6ImZpbGwtaW4tdGhlLWJsYW5rLTE3Nzc1Mzc5MjIxMTgiLCJpbnN0cnVjdGlvbiI6IkxlbmdrYXBpIGthbGltYXQgYmVyaWt1dCBkZW5nYW4ga2F0YSB5YW5nIHRlcGF0LiJ9" width="870" height="400" allow="fullscreen"></iframe>

*Alright*, sekarang mari kita lihat seperti apa keajaiban QLoRA saat dipraktikkan langsung di lapangan.

***

## Native Ways: Latihan Fine-tuning QLoRA

*Alright*, mari kita buka latihan ini dengan penjelasan QLoRA atau Quantized Low-Rank Adaptation. Dari nama tekniknya, kita sudah bisa berekspektasi bahwa ini merupakan gabungan dari dua pendekatan efisiensi yang sudah kita bahas sebelumnya, Quantization dan Low-Rank Adaptation (LoRA).

*Yup*, Anda sama sekali tidak salah.

Jika LoRA adalah "otak" yang membuat pelatihan menjadi efisien, dan Quantization adalah "diet" yang membuat model menjadi ramping, maka QLoRA adalah metode yang menggabungkan keduanya untuk mendobrak batasan hardware. QLoRA-lah alasan mengapa kita sekarang bisa melatih model sekelas Llama-3 atau Mistral hanya dengan menggunakan GPU gratisan di Google Colab atau laptop gaming rumahan.

Mengapa bisa begitu?

Sebab QLoRA melakukan sesuatu yang terdengar mustahil, yaitu melatih model yang kualitasnya setara dengan 16-bit, tetapi dengan konsumsi memori 4-bit.

![dos-bbc5392e5f66ae8af503a590bfb1aea020260330151914.png](https://assets.cdn.dicoding.com/original/academy/dos-bbc5392e5f66ae8af503a590bfb1aea020260330151914.png)

Namun, yang membuat QLoRA spesial bukan karena sesederhana ia menggabungkan Quantization dan LoRA. Seperti yang ditunjukan pada ilustrasi di atas, QLoRA menambahkan dua teknik optimasi tambahan sebagai garasi pelatihan yang dapat berjalan lancar pada VRAM GPU yang terbatas.

*   **Double Quantization** 
    Sistem tidak hanya mengkuantisasi bobot model, tetapi juga mengkuantisasi konstanta dari kuantisasi itu sendiri. Langkah matematis ekstra ini menghemat sekitar 0.37 bits per parameter, yang dampaknya sangat signifikan pada model berukuran puluhan miliar parameter.
*   **Paged Optimizers** 
    Menggunakan fitur Unified Memory pada sistem operasi untuk memindahkan (paging) status optimizer secara dinamis ke RAM CPU ketika memori GPU hampir mencapai batas maksimal. Hal ini mencegah terjadinya error Out-of-Memory (OOM) yang fatal saat training berlangsung.

Oleh karena bantuan dua teknik di atas, QLoRA mampu menjadi orkestrator optimasi yang menurunkan barrier to entry untuk fine-tuning LLM. Hal ini tentunya mengubah proses yang tadinya eksklusif untuk komputasi kluster berskala besar menjadi sangat aksesibel bagi praktisi di berbagai tingkatan.

*So*, karena perkenalan kita dengan QLoRA sepertinya sudah cukup, waktunya kita mempraktikkannya secara langsung di lapangan. Sebagai informasi tambahan, mungkin Anda sudah menyadari ada kata "Native" bertengger manis di judul latihan kali ini. *Yup*, itu karena perjalanan fine-tuning kita kali ini akan dimulai dengan Native Ways terlebih dahulu.

Tenang saja kita tentu tidak akan lupa dengan janji di awal untuk mengeksplorasi Unsloth sebagai pendekatan yang sudah dioptimasi lebih lanjut. Mari kita sedikit overview dua pendekatan ini.

*   **Pendekatan Native** 
    Ini adalah jalur fundamental di mana kita merakit seluruh arsitektur fine-tuning secara manual menggunakan library standar industri seperti transformers, bitsandbytes, PEFT, dan TRL. Pendekatan ini memberikan Anda kendali langsung atas setiap parameter, mulai dari mengatur konfigurasi memori saat memampatkan model ke 4-bit, hingga mendefinisikan bagaimana adaptor LoRA disuntikkan ke dalam lapisan model.
*   **Pendekatan Unsloth** 
    Ini adalah jalur efisiensi tingkat lanjut yang akan kita gunakan setelah Anda menguasai fondasi native. Unsloth pada dasarnya adalah pustaka yang menulis ulang operasi matematika yang paling berat di level GPU, seperti kalkulasi Cross Entropy Loss. Pendekatan ini adalah alat andalan untuk menghemat waktu dan biaya komputasi, yang kehebatannya baru bisa Anda apresiasi penuh setelah merasakan kerumitan di jalur native.

Apakah Anda sudah siap? Mari kita mulai dari tahap persiapan terlebih dahulu.

### Persiapan Dependencies dan GPU

Seperti biasa, perjalanan kita akan dimulai dengan mempersiapkan dependencies. Anda dapat menjalankan kode berikut untuk menginstal beberapa dependencies yang akan menemani perjalanan ini.

```python
!pip -q install transformers==4.56.2
!pip -q install datasets
!pip -q install --no-deps peft
!pip -q install --no-deps trl==0.22.2
!pip -q install --no-deps accelerate
!pip -q install --no-deps bitsandbytes
```

Jika Anda perhatikan baris kode di atas, ada satu "aturan main" baru yang sengaja kita tetapkan, yaitu penggunaan flag --no-deps. Flag ini mencegah pip menginstal ulang atau mengubah versi library dasar yang sudah ada, seperti contohnya PyTorch. Dengan begitu, kita mencegah risiko environment yang tiba-tiba rusak.

Setelah semua alat berhasil diunduh, saatnya kita panggil modul-modul yang akan digunakan dengan menjalankan kode berikut ini.

```python
import torch
import time
import torch
import os
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextStreamer
from trl import SFTTrainer, SFTConfig, setup_chat_format
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
```

Mari kita bedah secara spesifik beberapa alat di atas agar Anda tahu persis fungsi setiap alat digunakan di sini.

*   `wandb`**: Weights & Biases atau wandb adalah alat yang akan merekam dan memvisualisasikan grafik penting seperti training loss dan learning rate secara real-time di dashboard web mereka. Penggunaan modul ini sebenarnya opsional, namun sangat direkomendasikan agar kita tidak melatih model dalam keadaan "buta" dan bisa memantau proses *training*-nya.
*   `trl `**(Transformer Reinforcement Learning):**Ini adalah modul dari HuggingFace yang akan mengeksekusi training.
    *   SFTTrainer: Mesin utama yang akan menjalankan loop Supervised Fine-Tuning. Ia mengambil model, dataset, dan konfigurasi, lalu meraciknya.
    *   SFTConfig: *Blueprint* tempat kita akan mendefinisikan konfigurasi hyperparameter seperti batch size, learning rate, dan jumlah epoch.
    *   setup_chat_format: Fungsi utilitas yang sangat krusial. Fungsi ini memastikan model dan tokenizer kita memiliki pemahaman yang seragam tentang format percakapan seperti penanda user dan assistant, sehingga model tidak kebingungan saat membaca instruksi.
*   `peft `**(Parameter-Efficient Fine-Tuning)**: Inilah yang akan menangani semua hal yang berkaitan dengan LoRA.
    *   LoraConfig: *Blueprint* yang kita gunakan untuk mengatur konfigurasi adapter kita, seperti nilai *rank*, *alpha*, dan *target modules*.
    *   get_peft_model: Fungsi yang mengeksekusi cetak biru tadi. Ia membungkus model dasar kita dan menempelkan adaptor LoRA ke dalamnya.
    *   prepare_model_for_kbit_training: Fungsi ini bertugas menstabilkan model yang salah satu caranya adalah mengubah format presisi pada layer normalization. Sehingga, proses kalkulasinya tidak error atau menghasilkan angka tak terhingga (NaN) di tengah jalan. Langkah *preprocessing* ini wajib dilakukan sebelum training 4-bit dimulai.
*   `BitsAndBytesConfig`: Modul ini digunakan untuk mengatur konfigurasi Quantization dari LLM yang digunakan.
*   `os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"`: Ini adalah senjata rahasia kita untuk melawan Out of Memory (OOM). Terkadang, GPU sebenarnya masih punya sisa memori, tetapi posisinya terpencar-pencar. *Nah*, perintah ini akan meminta PyTorch untuk mengelola blok memori secara lebih fleksibel dan menyatukan pecahan-pecahan tersebut secara otomatis. Sehingga, model raksasa kita tetap bisa bernapas di ruang VRAM yang sempit.

Setelah seluruh modul, fungsi, dan perintah yang kita butuhkan telah terpanggil, saatnya kita memastikan keberadaan GPU yang tersambung dengan Google Colab. Oh iya, pastikan Anda menggunakan T4 GPU, ya.

Mari kita jalankan kode berikut untuk mendeteksi ketersediaan dan kapasitas GPU di environment.

```python
if torch.cuda.is_available():
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU Terdeteksi: {gpu_stats.name}. Max Memory = {max_memory} GB.")
else:
    print("Peringatan: GPU tidak terdeteksi.")
```

Jika *output*-nya menampilkan nama GPU (misalnya Tesla T4 atau A100), Anda sudah siap berakselerasi. *Nah*, tahapan selanjutnya sebenarnya opsional seperti yang kita mention sebelumnya. Di tahap tersebut kita mempersiapkan monitoring proses training pada dashboard Weight & Biases.

### [Opsional] Persiapan WandB untuk Report

Sebelum Anda menjalankan kode di bawah ini, Anda perlu mendapatkan token wandb miliki Anda sendiri. Caranya sangat sederhana, Anda hanya perlu pergi ke tautan [Weight & Biases](https://wandb.ai/site) dan membuat akun baru jika belum memilikinya.

Setelah semua tahapan pendaftaran selesai, Anda tinggal pergi ke menu API Keys seperti berikut ini.

![dos-8ddd0a7897b4613ee43e13c89efc2b5920260311170155.png](https://assets.cdn.dicoding.com/original/academy/dos-8ddd0a7897b4613ee43e13c89efc2b5920260311170155.png)

*Nah*, setelah masuk, Anda akan diarahkan ke tampilan berikut ini.

![dos-a869fb834ec9da536f5d4427b82c1f3f20260311170202.png](https://assets.cdn.dicoding.com/original/academy/dos-a869fb834ec9da536f5d4427b82c1f3f20260311170202.png)

Di sana, Anda dapat membuat token atau API keys sendiri. Setelah mendapatkan token-nya, Anda dapat memasukkannya pada kode di bawah ini sekaligus menjalankannya.

```python
USE_WANDB = True
WANDB_TOKEN = "TOKEN_WANDB_ANDA"

if USE_WANDB:
    os.environ["WANDB_API_KEY"] = WANDB_TOKEN
    wandb.init(project="Llama-3.1-Benchmark", name="Unsloth_Run")
    report_target = "wandb"
else:
    os.environ["WANDB_DISABLED"] = "true"
    report_target = "none"
```

Nantinya setiap informasi mengenai proses training yang akan kita lakukan ini akan tampil secara otomatis pada dashboard Wandb milik Anda.

### Konfigurasi untuk Quantization

*Alright*, kita mulai masuk ke *main course*.

Saatnya kita mempersiapkan “settingan” Quantization yang akan diterapkan ke dalam LLM yang nantinya dipanggil. Anda dapat menjalankan kode di bawah ini.

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
```

*Yup*, modul BitsAndBytesConfig yang kita kenal sebelumnya akan menjadi wadah bagi beberapa hyperparameter berikut untuk mengatur Quantization model.

*   `load_in_4bit=True`: Ini adalah saklar utama dari Quantization yang secara eksplisit memerintahkan sistem untuk memuat weights model dalam resolusi 4-bit.
*   `bnb_4bit_quant_type="nf4"`: Di sini kita memberitahu BitsAndBytes bahwa kita ingin menggunakan tipe kuantisasi NF4 atau *NormalFloat 4* yang kita pelajari juga sebelumnya.
*   `bnb_4bit_compute_dtype=torch.float16`: Masih ingat dengan proses dequantization sebelumnya? Yup, meskipun model "disimpan" dalam bentuk 4-bit di memori, kita akan mengubahnya sejenak ke format yang lebih tinggi seperti bfloat16, float16, atau float32 untuk menjaga kestabilan perhitungan matematika pada tahap komputasi. *Nah*, kali ini kita memilih untuk menggunakan float16.
*   `bnb_4bit_use_double_quant=True`: *Nah*, inilah keunikan QLoRA yang kita bahas sebelumnya. Sesuai namanya, ini adalah Double Quantization atau kuantisasi ganda.

Mungkin Anda akan bertanya seperti ini, sebelumnya kita belajar bahwa BFloat16 lebih stabil ketimbang Float16, tetapi kenapa kita menggunakan Float16 pada eksperimen ini?

Seperti yang kita ketahui, dalam eksperimen ini kita menggunakan T4 GPU. *Nah*, Jika Anda menggunakan GPU T4, AWS p3 (V100), atau kartu grafis seri GTX 10xx/RTX 20xx, hardware tersebut secara fisik tidak memiliki instruksi BFloat16. Memaksakan BF16 pada hardware ini akan membuatnya sangat lambat (emulasi software) atau error. Oleh karena itu, kita akan tetap menggunakan Float16 meskipun kita sepakat bahwa BFloat16 lebih stabil di atas kertas.

*Alright*, dengan ini *blueprint* dari Quantization kita sudah siap dan saatnya untuk mengunduh dan memuat model LLM yang sebenarnya dari Hugging Face Hub sambil menanamkan konfigurasi kompresi ini ke dalamnya.

### Panggil Model Quantization dan Tokenizer

Seperti biasa, untuk memanggil LLM dari Hugging Face, kita akan menggunakan dua komponen sekaligus, yaitu model dan tokenizer-nya.

*Yup*, jika Anda tengok sedikit kode di bawah, kita akan menggunakan Llama 3.1 dengan ukuran 8B. Tentu ini pilihan yang menarik karena kita akan membuktikan bahwa fine-tuning Llama 3.1 8B yang sebelumnya kita hitung ternyata membutuhkan sekitar 128 GB dapat diwujudkan di VRAM dengan memori 16 GB saja.

Selain itu, pemilihan model Llama 3.1 yang masih merupakan base model akan menjadi tantangan yang menarik. Seperti yang kita bahas di modul awal, base model belum dirancang sebagai instruction following model, sehingga proses fine tuning yang kita lakukan ini akan mengubahnya menjadi model instruct.

Terlebih lagi berdasarkan informasi yang tersedia di [Hugging Face](https://huggingface.co/unsloth/Llama-3.1-8B), Llama 3.1 pada dasarnya dilatih sebagai model berbahasa Inggris, sehingga menambah kompleksitas dalam proses adaptasi model ini.

![dos-2ba8045125922a83da666a122f28009620260430085449.gif](https://assets.cdn.dicoding.com/original/academy/dos-2ba8045125922a83da666a122f28009620260430085449.gif)

Oke kita cukupkan perkenalannya dengan Si Model dan mari jalankan baris kode berikut untuk mengunduh dan memuat model secara perlahan ke dalam GPU.

```python
model_name = "unsloth/Llama-3.1-8B"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    low_cpu_mem_usage=True,
    dtype=torch.float16
)
```

Alright, mari kita bedah beberapa parameter penting yang baru saja kita gunakan di atas.

*   `quantization_config=bnb_config`: *Nah*, di sinilah tempat kita memasukkan *blueprint* BitsAndBytes yang sudah kita siapkan sebelumnya dan diberi nama bnb_config. Dengan ini, saat AutomodelForCausalLM memuat model, ia akan langsung menerapkan Quantization pada model tersebut sesuai dengan konfigurasi yang sudah kita tetapkan.
*   `device_map="auto"`: Daripada mengatur secara manual, parameter ini meminta sistem untuk secara otomatis mendistribusikan layer-layer model ke perangkat komputasi yang tersedia (GPU dan CPU) dengan cara yang paling optimal.
*   `low_cpu_mem_usage=True`: Flag ini memastikan bahwa saat proses load model berlangsung, model tidak akan menghabiskan seluruh RAM CPU Anda sebelum dipindahkan ke GPU, sehingga mencegah sistem mengalami crash.
*   `dtype=torch.float16`: Di sini kita akan memaksa Non-Quantized Layers dalam model seperti Embeddings dan Normalization untuk juga menggunakan Float16.

Saat Anda menjalankan kode di atas, proses download akan terjadi dan outputnya akan terlihat seperti ini.

![dos-5d1ae7bbb2800426c5b23c3c4787554d20260311170414.png](https://assets.cdn.dicoding.com/original/academy/dos-5d1ae7bbb2800426c5b23c3c4787554d20260311170414.png)

*Nah*, kini LLM kita sudah berhasil “bersandar” dengan aman di sistem. Namun, seperti yang sebelum-sebelumnya, model ini tidak bisa langsung mencerna bahasa manusia begitu saja. Ia membutuhkan "penerjemah" yang bisa mengubah kata-kata menjadi angka. Oleh karena itu, mari kita panggil tokenizer khusus yang memang diciptakan berpasangan dengan model Llama-3.1-8B dengan menjalankan baris kode di bawah ini.

```python
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
```

Hal yang menarik di sini adalah selain kita memanggil tokenizer, kita sedikit “mengotak-atik” tokenizer agar dapat berjalan sesuai yang diharapkan. Mari kita bahas detailnya.

*   `tokenizer.pad_token = tokenizer.eos_token`: Saat melakukan training, kita biasanya memasukkan banyak kalimat sekaligus dalam satu batch. Masalahnya, panjang setiap kalimat pasti berbeda-beda, padahal GPU membutuhkan ukuran yang seragam untuk perhitungan matriks. Oleh karena itu, kita menambahkan padding ke dalam kalimat pada data kita nanti. Namun, Llama secara bawaan tidak memiliki token padding khusus sehingga kita meminjam token penanda akhir kalimat, yaitu End-Of-Sentence atau eos_token untuk melakukan tugas tersebut.
*   `tokenizer.padding_side = "right"`: Di sini kita menginstruksikan agar padding sebelumnya diletakkan di sebelah kanan yang berarti akhir kalimat. Alasannya sederhana, sebab LLM melakukan autoregressive dari kiri ke kanan. Jika ditaruh di kiri, posisi urutan kata akan berantakan dan model akan kesulitan memprediksi kata berikutnya dengan benar.

Setelah menjalankan kode di atas apakah model dan tokenizer kita sudah siap? Tentu saja belum.

Seperti yang sudah kita bahas, Llama 3.1 dan model LLM lainnya pada dasarnya hanya diajari untuk menebak kata selanjutnya atau *next-token prediction*. *Nah*, agar ia bisa berinteraksi secara luwes layaknya chatbot dengan kemampuan untuk membedakan mana instruksi dari kita manusia dan mana jawaban dari sistem, kita perlu menanamkan "tata krama percakapan" alias *chat template* ke dalamnya.

Coba Anda jalankan kode di bawah ini.

```python
model, tokenizer = setup_chat_format(model, tokenizer)
```

Satu baris kode setup_chat_format dari TRL di atas pada dasarnya melakukan banyak pekerjaan berat di belakang. Saat dijalankan, ia akan memasukkan token khusus untuk percakapan seperti penanda awal `<|im_start|>` dan akhir `<|im_end|>`, serta penanda role seperti user atau assistant ke dalam vocabulary tokenizer kita.

*Nah*, karena tokenizer kita mendapat kosakata baru berupa token-token spesial tadi, fungsi ini juga pintar merespons dengan otomatis memperbesar ukuran layer embedding pada model kita. Hal ini memastikan model tidak kebingungan atau error saat membaca tanda percakapan tersebut.

Dengan format yang terstandarisasi, model akan tahu persis kapan giliran Anda bertanya dan kapan giliran model untuk memberikan jawaban. *Nah*, karena model dan tokenizer sudah kita persiapkan dengan matang, waktunya kita bertemu dengan dataset dalam fine-tuning kali ini.

### Persiapan Dataset

Sebelumnya kita sudah kenalan dengan modelnya, sekarang saatnya kita kenalan sedikit dengan dataset-nya. [cahya/alpaca-id-cleaned](https://huggingface.co/datasets/cahya/alpaca-id-cleaned) adalah dataset instruksi bahasa Indonesia berukuran sekitar 50.000+ baris data teks yang merupakan hasil terjemahan dan pembersihan dari dataset Stanford Alpaca.

![dos-ca2f1c0f702254d123aa05b87b112ad720260311170443.png](https://assets.cdn.dicoding.com/original/academy/dos-ca2f1c0f702254d123aa05b87b112ad720260311170443.png)

Seperti yang terlihat pada gambar di atas, dataset ini menggunakan format JSON atau JSONL yang terdiri dari tiga kolom.

*   **Instruction:** Tugas atau perintah yang harus dilakukan.
*   **Input:** Konteks tambahan jika diperlukan (seringkali kosong jika instruksinya sudah jelas).
*   **Output:** Jawaban ideal yang diharapkan dari model.

Selanjutnya, karena ini adalah versi "cleaned", dataset tersebut sudah dibersihkan dan dapat langsung digunakan. Jadi, mari kita memuat datasetnya dengan menjalankan perintah di bawah ini.

```python
dataset = load_dataset("cahya/alpaca-id-cleaned", split = "train")
print(f"Jumlah data training: {len(dataset)}")
```

Agar hati lebih tenang, mari kita intip sedikit baris pertama dari data yang sudah kita load dengan kode berikut.

```python
dataset[0]
```

Hasilnya kira-kira akan terlihat seperti ini.

![dos-20375444b57f275c2b3bf2cbd184908820260311170508.png](https://assets.cdn.dicoding.com/original/academy/dos-20375444b57f275c2b3bf2cbd184908820260311170508.png)

*Yup*, bentuknya sesuai dengan informasi yang kita dapatkan saat berkenalan dengan dataset cahya/alpaca-id-cleaned sebelumnya. Lantas apakah data yang secara kasat mata ini sudah rapi bisa langsung digunakan?

Sayangnya belum.

Data teks Alpaca di atas masih berupa potongan-potongan terpisah (instruksi, input, dan output). *Nah*, agar model bisa memahaminya sebagai sebuah alur percakapan yang natural, kita harus menyatukan potongan-potongan tersebut menggunakan *format chat* yang sudah disiapkan di awal.

Untuk melakukan itu, Anda dapat menjalankan kode berikut ini.

```python
system_prompt = "Kamu adalah asisten AI yang membantu menjawab pertanyaan pengguna berdasarkan konteks yang diberikan."


def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    
    for instruction, input, output in zip(instructions, inputs, outputs):
        user_content = f"{instruction}\n\nContext:\n{input}" if input else instruction
        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output}
        ]
        text = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
        texts.append(text)
    return { "text" : texts }


dataset = dataset.map(formatting_prompts_func, batched=True)
```

Fungsi di atas mungkin terlihat agak panjang, tetapi tenang saja karena kita akan membedahnya bersama.

*   `user_content`: Tidak semua instruksi memiliki konteks tambahan atau input. Sehingga, baris kode ini akan menggabungkan instruksi dan input jika input-nya ada. Jika tidak ada, ia hanya akan menggunakan instruksinya saja.
*   `conversation`: Di sini, kita menyusun data ke dalam format *dictionary* standar yang berisi role (siapa yang berbicara) dan content (apa yang dibicarakan). Strukturnya akan selalu berurutan, yaitu mulai dari system → user (Pengguna) → assistant (LLM).
*   `tokenizer.apply_chat_template(...)`: *Nah*, ingat fungsi `setup_chat_format` yang kita jalankan sebelumnya? Di sinilah fungsi itu digunakan untuk mengubah *dictionary* tadi menjadi satu string teks panjang yang sudah disisipi token-token spesial, seperti `<|im_start|>` dan `<|im_end|>`.
    *   `tokenize=False` digunakan karena saat ini kita hanya ingin melihat hasil teks atau string, bukan token dalam bentuk angka. Sebab, SFTTrainer akan melakukan tokenisasi secara otomatis di tahap berikutnya. Ini menghindarkan data Anda ter-tokenisasi dua kali.
    *   `add_generation_prompt=False` digunakan karena ini adalah data training yang sudah lengkap dengan jawaban asistennya, bukan instruksi yang sedang menunggu model untuk men-*generate* jawaban.

Dengan data yang sudah siap ini, tibalah saatnya kita menyentuh bagian paling krusial dalam Fine-Tuning ini, yaitu LoRA.

### Konfigurasi LoRA

Sebelumnya, kita sudah menambahkan Quantization dalam fine-tuning ini. *Nah*, untuk mewujudkan QLoRA, kita akan mulai membangun konfigurasi Low-Rank Adaptation yang akan digunakan.

*Alright*, mari kita wujudkan dua matriks kecil A dan B dengan menjalankan kode berikut ini.

```python
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0,
    bias="none",
    task_type="CAUSAL_LM",
)
```

Seperti biasa kita akan membedah satu per satu hyperparameter yang ada pada “resep” LoRA di atas ini.

*   `r=8`: Ini dia tempat kita mendefinisikan ukuran Rank yang selalu kita mention saat membahas LoRA sebelumnya. Seperti yang kita pelajari sebelumnya, semakin kecil angkanya, semakin sedikit parameter yang dilatih.
*   `lora_alpha=16`: alpha atau yang kita sebut factor scale sebelumnya adalah parameter yang menentukan seberapa kuat efek dari pembelajaran di adaptor LoRA ini memengaruhi model aslinya. *Rule of thumb* yang sering dipakai adalah mengatur nilainya dua kali lipat atau setidaknya sama dengan nilai r.
*   `target_modules=["q_proj", ...]`: Nah, inilah yang menentukan di bagian mana saja matriks kecil A dan B dari LoRA ini diletakkan. Kali ini, kita memasangnya pada seluruh modul Attention Mechanism (modul Query, Key, Value, dan Output). Kita tidak memasangnya di Feed-Forward Network untuk menghindari terjadinya OOM.
*   `bias="none"`: Parameter ini menginstruksikan sistem untuk tidak melatih parameter bias. Tujuannya murni untuk menjaga agar jumlah parameter yang dilatih tetap sekecil dan seefisien mungkin.
*   `task_type="CAUSAL_LM"`: Baris ini sekadar memberi tahu library PEFT bahwa model kita memiliki tugas spesifik, yaitu memprediksi kata selanjutnya secara berurutan atau disebut Causal Language Modeling.

*Alright*, karena seluruh konfigurasi QLoRA kita sudah ready to cook, saatnya kita masuk ke fase yang paling krusial, yaitu fase training.

### Supervise Fine-tuning

*Eits*, sebelum kita menerapkan konfigurasi LoRA yang sudah dibuat ke dalam model, ada satu tahapan yang wajib dieksekusi. Karena di awal kita memuat model Llama menggunakan metode Quantization, arsitektur model tersebut perlu disesuaikan terlebih dahulu agar kompatibel dengan proses training.

Apa saja yang perlu kita lakukan untuk melakukan hal yang sepertinya kompleks tersebut? Tenang saja karena kita hanya perlu menjalankan satu baris kode di bawah ini.

```python
model = prepare_model_for_kbit_training(model)
```

Meskipun kode di atas hanya memiliki satu baris, proses yang terjadi di belakang layar cukup kompleks seperti yang kita duga di awal. Tentunya kita akan mengarungi ini bersama dengan membedahnya step by step.

*   **Penyesuaian Layer** 
    Kita sadar bahwa model yang dimuat dalam format 4-bit memiliki presisi kalkulasi yang rendah. *Nah*, kode di atas akan secara otomatis mengubah format tipe data pada komponen model yang sangat sensitif terhadap perubahan angka, seperti Layer Normalization, menjadi presisi tinggi (float32). Proses ini akan mencegah munculnya nilai eror atau NaN akibat ketidakstabilan numerik selama perhitungan gradien nantinya.
*   **Penyesuaian Output Embedding** 
    Secara default, saat model dibekukan (frozen) untuk QLoRA, semua gradien dimatikan. Fungsi ini memastikan output embedding layer tetap bisa menerima gradien. Ini sangat krusial agar model bisa "belajar" menyesuaikan distribusi token baru selama proses fine-tuning.
*   **Efisiensi Gradient Checkpointing** 
    Sebagai standar optimasi, tahap ini mempersiapkan model untuk menggunakan fitur gradient checkpointing. Alih-alih menyimpan seluruh hasil kalkulasi selama proses forward pass yang akan menghabiskan banyak VRAM GPU, teknik ini hanya menyimpan titik-titik tertentu dan menghitung ulang sisanya saat backward pass diperlukan. Hal ini menghemat penggunaan memori secara signifikan.

Dengan menjalankan fungsi tersebut, model kuantisasi Anda kini sudah stabil secara numerik dan siap untuk menerima teknik fine-tuning dengan presisi rendah. *Yup*, tahap selanjutnya adalah menentukan berbagai hyperparameter dengan SFTConfig dari modul TRL sebagai wadahnya.

```python
sft_config = SFTConfig(
    output_dir="outputs_native",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=5,
    max_steps=30,
    learning_rate=2e-4,
    lr_scheduler_type="linear",
    fp16=True,
    logging_steps=1,
    optim="paged_adamw_8bit",
    dataset_text_field="text",
    max_length=512,
    packing=True,
    report_to=report_target,
    run_name="Native_Llama3.1_Run" if USE_WANDB else None
)
```

*Waw*, hyperparameter yang sangat banyak di sana, tetapi tenang saja karena kita akan coba kupas beberapa di antaranya.

*   `per_device_train_batch_size=2`: Menetapkan jumlah sampel data yang akan diproses oleh GPU dalam satu iterasi forward dan backward pass. Nilai 2 dipilih secara spesifik untuk meminimalkan penggunaan VRAM dan mencegah terjadinya *Out of Memory* (OOM).
*   `gradient_accumulation_steps=4`: Karena batch size diatur sangat kecil (2), parameter ini menunda proses pembaruan bobot model sampai 4 langkah kalkulasi gradien selesai dilakukan. Hal ini menghasilkan pembaruan gradien yang setara dengan batch size 8 (2 dikali 4) yang membuat proses pelatihan jauh lebih stabil secara matematis tanpa memerlukan kapasitas VRAM tambahan.
*   `learning_rate=2e-4`: Menentukan besaran rasio (0.0002) pada langkah optimasi saat algoritma menyesuaikan bobot model. Nilai ini merupakan standar matematis yang terbukti efisien untuk proses fine-tuning LLM.
*   `warmup_steps=5`: Menjadwalkan peningkatan nilai learning rate secara bertahap dari angka 0 menuju target maksimal (2e-4) selama 5 langkah pertama pelatihan. Proses ini mencegah terjadinya pembaruan gradien yang terlalu ekstrem pada awal komputasi yang berpotensi menyebabkan instabilitas model.
*   `lr_scheduler_type="linear"`: Menginstruksikan sistem bahwa setelah fase *warm up* selesai, nilai learning rate akan diturunkan secara konstan (linear) seiring bertambahnya langkah pelatihan hingga mencapai titik akhir.
*   `max_steps=30`: Membatasi total langkah pembaruan bobot yang akan dieksekusi hanya sebanyak 30 langkah karena fokus kita di sini adalah demonstrasi. Jika Anda ingin hasil yang lebih maksimal, silakan tambahkan jumlah step ini. Namun, sebagai informasi untuk Anda, dengan jumlah 30 step ini saja akan memakan waktu 1 jam untuk proses fine-tuningnya.
*   `fp16=True`: Di sini, kita juga akan meminta GPU untuk menggunakan Float16 saat melakukan komputasi. Artinya, parameter yang dikuantisasi sebelumnya akan di-*dequantization* menjadi Float16.
*   `logging_steps=1`: Memerintahkan sistem untuk merekam metric pelatihan, seperti nilai training loss, pada setiap langkah pelatihan secara berurutan.
*   `optim="paged_adamw_8bit"`: Menentukan penggunaan algoritma optimasi AdamW dalam format 8-bit. Fitur "paged" merupakan mekanisme manajemen memori di mana sebagian state dari optimizer akan dipindahkan ke RAM CPU secara otomatis apabila kapasitas VRAM GPU mendekati batas maksimal, mencegah sistem dari crash.
*   `dataset_text_field="text"`: Menunjukkan nama kolom pada dataset kita (yang telah diformat pada tahap fungsi mapping sebelumnya) yang berisi teks percakapan secara utuh.
*   `max_length=512`: Membatasi jumlah maksimal token yang diproses dalam satu urutan input menjadi 512 token dan memotong (truncating) teks yang melebihi batas.
*   `packing=True`: Mengaktifkan fitur penggabungan beberapa sekuens teks pendek dari dataset ke dalam satu blok input tunggal hingga batas max_length tercapai. Ini meningkatkan efisiensi pelatihan secara signifikan karena GPU memproses rasio token aktual yang lebih tinggi dibandingkan memproses token padding yang kosong.
*   `report_to` dan `run_name`: Argumen ini mengintegrasikan sesi pelatihan dengan alat pemantauan eksternal (seperti Weights & Biases) untuk merekam visualisasi grafis dari seluruh metrik eksperimen berdasarkan variabel yang telah Anda tentukan.

*Finally*, perjalanan panjang kita dalam hyperparameter SFTConfig usai juga. Saatnya kita mendefinisikan fungsi trainer yang akan menyatukan konfigurasi SFT di atas, PEFT sebelumnya, dataset, tokenizer, dan model Llama 3.1 8B yang kita gunakan.

Anda dapat melakukan hal tersebut dengan menjalankan kode di bawah ini.

```python
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    args=sft_config,
    peft_config=peft_config,
)
```

Dengan tereksekusinya kode di atas, seluruh persiapan teknis dan *flow* data untuk proses fine-tuning telah terintegrasi secara “ciamik” pada objek trainer.

Sebelum kita benar-benar mengeksekusi proses pelatihan utama, ada satu pertanyaan menarik yang jawabannya mungkin membuat penasaran.

*“Apakah Anda tidak kepo dengan total parameter yang akan benar-benar dilatih setelah kita menerapkan QLoRA dan PEFT barusan?”* 

Untuk mengetahuinya, silakan jalankan baris kode ini untuk mencetak kalkulasi parameter pada model Anda.

```python
trainer.model.print_trainable_parameters()
```

Jika Anda mengikuti seluruh langkah-langkah sebelumnya secara 100% mirip, output-nya akan terlihat seperti berikut ini.

```python
trainable params: 6,815,744 || all params: 8,037,093,376 || trainable%: 0.0848
```

Hasil di atas menunjukkan bahwa kita akan melatih sebanyak 6,8 juta parameter yang seluruhnya berasal dari modul Attention (q_proj, k_proj, v_proj, dan o_proj), ini sesuai dengan modul yang kita targetkan sebelumnya pada LoRA. Sementara, 8,037,093,376 atau 8 miliar parameter di atas merupakan total keseluruhan parameter yang menyusun arsitektur model Llama-3.1-8B.

Artinya, kita akan melatih kurang dari 0.1% atau tepatnya 0.0848% dari total parameter Llama-3.1-8B. Jumlah yang sangat amat kecil inilah yang memungkinkan komputasi pelatihan model LLM berskala 8 miliar parameter dapat dieksekusi secara stabil tanpa memicu OOM.

*Nah*, tibalah saatnya kita menekan *button* peluncuran roket fine-tuning kali ini. Silakan untuk menjalankan kode di bawah ini untuk meluncurkan proses fine-tuning dengan *native ways* ini.

```python
trainer.train()
```

Seharusnya, jika semua langkah-langkah di atas diikuti dengan akurat, proses training akan berjalan secara mulus seperti berikut ini.

![dos-50a995b5f6e7e5acffa468d0cc1558b520260311170922.png](https://assets.cdn.dicoding.com/original/academy/dos-50a995b5f6e7e5acffa468d0cc1558b520260311170922.png)

Setelah proses komputasi pada metode trainer.train() selesai dieksekusi secara keseluruhan, sistem telah mencatat berbagai metrik pelatihan secara berurutan. Jika Anda mengaktifkan integrasi pelacakan eksperimen eksternal, ada satu tahapan penutup yang perlu dipanggil agar sinkronisasi data berjalan sempurna.

Silakan jalankan baris kode berikut untuk mengakhiri sesi monitoring tersebut.

```python
if USE_WANDB:
    wandb.finish()
```

Dengan selesainya instruksi ini, seluruh rekam jejak eksperimen fine-tuning Anda telah ditutup, divalidasi, dan diarsipkan dengan aman pada Dashboard W&B.

Setelah model kita berhasil memperbarui matriks bobotnya melalui proses pelatihan, ini adalah saat yang tepat untuk menguji langsung kapabilitasnya. Kita akan mengeksekusi proses inferensi dengan memberikan sebuah instruksi baru yang belum pernah dilihat oleh model.

### Inference

Sebelum melakukan *inference*, kita perlu menyetel LLM yang kita gunakan ke mode inference. Bagaimana caranya? Anda hanya perlu menjalankan satu bari kode sederhana berikut.

```python
model.eval()
```

Baris kode di atas secara eksplisit mengubah status arsitektur model dari mode training menjadi mode evaluation. Perintah ini akan menonaktifkan lapisan tertentu seperti *dropout* yang hanya berguna saat proses pelatihan sehingga model akan memberikan respons yang pasti dan konsisten saat inferensi.

Setelah kita memvalidasi arsitektur model pada mode evaluasi, langkah selanjutnya adalah mendefinisikan prompt dan mentransformasikannya menjadi format numerik matriks yang dapat diproses langsung oleh GPU.

```python
messages = [
    {"role": "system", "content": "Anda adalah asisten cerdas yang menjawab pertanyaan secara akurat berdasarkan fakta."},
    {"role": "user", "content": "Sebutkan 3 candi terkenal di Indonesia beserta lokasinya!"}
]


inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to("cuda")
```

Pada dasarnya apa yang kita lakukan pada kode di atas ini mirip dengan yang dilakukan pada saat formatting dataset sebelumnya. Hanya saja kali ini kita melakukan empat hal yang berbeda berikut ini.

*   `tokenize=True`: Menginstruksikan sistem untuk langsung mengonversi teks yang telah diformat tadi menjadi urutan angka (token IDs). Urutan angka inilah representasi matematis aktual yang akan dikalkulasi oleh lapisan embedding model.
*   `add_generation_prompt=True`: Parameter ini secara otomatis menambahkan token pemicu akhir seperti `<|start_header_id|>assistant<|end_header_id|>` di ujung sekuens tensor. Kehadiran token ini memberikan sinyal komputasi absolut kepada model bahwa alur input telah selesai, dan memicu model untuk memulai siklus regresi autoregressive.
*   `return_tensors="pt"`: Mengonversi format output urutan angka tersebut ke dalam bentuk matriks tensor standar PyTorch (pt) yang merupakan struktur data esensial untuk eksekusi komputasi deep learning.
*   `.to("cuda")`: Memindahkan objek tensor dari memori sistem utama (RAM CPU) secara langsung ke dalam VRAM perangkat keras grafis (GPU/CUDA). Transfer memori ini bersifat wajib agar proses komputasi inferensi dapat berjalan dengan akselerasi perangkat keras yang maksimal.

Dengan ini, input teks kita telah berhasil diubah menjadi tensor matematis yang tersimpan di memori GPU dan siap untuk diumpankan ke dalam model.

*Nah*, sebelum benar-benar men-*generate* output, kita akan menambahkan satu pemanis dalam proses inference kali ini. Coba Anda jalankan kode di bawah ini.

```python
streamer = TextStreamer(tokenizer, skip_prompt=True)
```

Seperti yang kita ketahui, LLM beroperasi secara *autoregressive*, yang berarti model menghitung dan memprediksi probabilitas token keluaran satu demi satu. Namun, jika menggunakan metode eksekusi standar, output atau response LLM tidak akan ditampilkan hingga keseluruhan sekuens teks selesai dikalkulasi sepenuhnya.

*Nah*, untuk mengatasi penundaan visual ini dan menampilkan hasil komputasi secara real-time layaknya kita menggunakan model proprietary seperti Gemini dan ChatGPT, kita mengimplementasikan modul TextStreamer dari Transformers.

*   `tokenizer`: Objek tokenizer kita diintegrasikan ke dalam `TextStreamer` sebagai argumen wajib. Ketika model menghasilkan ID token, streamer menggunakan tokenizer ini untuk langsung mengeksekusi proses decoding, yaitu mentranslasikan ID numerik tersebut kembali menjadi karakter string teks yang dapat dibaca secara instan pada setiap iterasi langkah autoregressive.
*   `skip_prompt=True`: Argumen parameter ini akan membuat streamer melakukan filtrasi pada tahap decoding. Sistem akan memotong dan tidak mencetak ulang matriks tensor dari input awal Anda (system prompt dan pertanyaan). Dengan demikian, antarmuka konsol murni hanya akan menampilkan teks baru atau response.

*Alright*, babak eksekusi akhir dari tahap inference ini adalah memasukkan variabel inputs dan streamer ke dalam metode model.generate(). Untuk melakukannya, Anda dapat menjalankan kode di bawah ini.

```python
with torch.no_grad():
  outputs = model.generate(
      input_ids=inputs,
      attention_mask=torch.ones_like(inputs), 
      max_new_tokens=256,
      streamer=streamer,
      temperature=0.4,
      top_p=0.9,
      do_sample=True,
      repetition_penalty=1.2,
      no_repeat_ngram_size=3,
      pad_token_id=tokenizer.eos_token_id,
      eos_token_id=tokenizer.eos_token_id
      )


input_length = inputs.shape[1]
response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
print(response)
```

Mari kita bedah secara komprehensif fungsi dari setiap parameter hiperparameter yang beroperasi di dalam metode model.generate tersebut.

*   `with torch.no_grad()`: *Context manager* PyTorch ini akan menonaktifkan seluruh komputasi dan pelacakan gradien matematis pada arsitektur model. Karena inferensi hanya melakukan forward pass, mematikan pelacakan gradien akan membebaskan sejumlah besar VRAM GPU dan mengakselerasi kecepatan eksekusi secara signifikan.
*   `input_ids=inputs`: Mengumpankan matriks tensor yang berisi urutan token representasi dari instruksi atau prompt Anda ke dalam lapisan embedding model.
*   `attention_mask=torch.ones_like(inputs)`: Membuat matriks tensor sekunder yang ukurannya identik dengan input_ids, di mana seluruh nilainya diisi dengan angka 1. Ini menginstruksikan mekanisme Attention model untuk memproses komputasi pada semua token input tanpa mengabaikan bagian mana pun, mencegah potensi error operasional pada format padding.
*   `max_new_tokens=256`: Model hanya diizinkan untuk memprediksi maksimal 256 token baru untuk mencegah terjadinya iterasi generasi teks yang tidak terbatas (infinite loop).
*   `streamer=streamer`: Mengintegrasikan objek TextStreamer yang telah kita buat sebelumnya.
*   `do_sample=True`: Anda masih ingat dengan berbagai cara yang digunakan oleh LLM untuk melakukan menebak token yang akan dikeluarkannya? Nah, parameter ini akan mengaktifkan mode sampling decoding ketimbang menggunakan *greedy decoding*.
*   `temperature=0.4`: Jika kita sampling decoding, tentu saja kita perlu mendefinisikan temperature yang ingin digunakan. Nilai yang rendah (0.4) akan memperbesar gap probabilitas antara token teratas dan token lainnya, memaksa model untuk menghasilkan teks yang lebih deterministik, faktual, dan presisi.
*   `top_p=0.9`: Menerapkan algoritma Nucleus Sampling. Sistem akan memotong distribusi probabilitas dan hanya mempertimbangkan kumpulan token teratas yang memiliki jumlah probabilitas kumulatif sebesar 90%.
*   `repetition_penalty=1.2`: Mengaplikasikan penalti matematis pada distribusi probabilitas untuk token-token yang sudah pernah muncul pada teks yang dihasilkan sebelumnya. Nilai > 1.0 (seperti 1.2) akan menekan probabilitas kemunculan token tersebut, mencegah model memproduksi kata yang diulang-ulang secara konstan.
*   `no_repeat_ngram_size=3`: Memberlakukan aturan restriksi mutlak yang membuat sistem tidak akan pernah mengizinkan rangkaian spesifik yang terdiri dari 3 token persis (n-gram) muncul lebih dari satu kali dalam output. Ini adalah mitigasi sekunder untuk mengatasi masalah *looping* kalimat.
*   `pad_token_id` dan `eos_token_id`: Menetapkan referensi token struktural untuk proses terminasi kalimat. Kita menggunakan eos_token_id dari tokenizer agar model memahami secara pasti parameter angka yang merepresentasikan akhir dari sebuah respons percakapan.

Dengan menjalankan kode di atas, berakhirlah babak native ini. Anda dapat melihat hasil respons yang diberikan oleh Llama-3.1-8B yang kita “paksa” untuk belajar menjawab pertanyaan dalam bahasa Indonesia

*Yup*, meskipun hasilnya kadang kurang sesuai, atleast model sekarang dapat berbicara bahasa indonesia. Hasil ini pada dasarnya adalah hal yang wajar sebab kita banyak melakukan pembatasan seperti dalam menentukan besar rank, panjang sequence, jumlah step dan bahkan target modules.

*Nah*, sekarang pertanyaannya adalah, apakah Anda sudah siap untuk “melonggarkan” seluruh batasan tersebut? Untuk mewujudkannya, kita membutuhkan pendekatan yang jauh lebih efisien dibandingkan native TRL ini. Oleh karena itu, kita akan masuk ke *Unsloth Ways*.

***

## Unsloth Ways: Latihan Fine-tuning QLoRA

Setelah babak Native Ways sebelumnya berakhir dengan baik, saatnya kita naik satu tangga untuk menuju ke pendekatan yang lebih next level, yaitu Unsloth Ways.

Jika pada metode Native kita belajar bagaimana komponen-komponen standar seperti transformers, peft, dan bitsandbytes bekerja sama, di babak ini kita akan berkenalan dengan Unsloth.

Apa itu Unsloth?

![dos-b21d1720124d3e146b294122650d592a20260430090457.png](https://assets.cdn.dicoding.com/original/academy/dos-b21d1720124d3e146b294122650d592a20260430090457.png)

Unsloth adalah library open-source yang sedang naik daun karena klaimnya yang luar biasa, yaitu fine-tuning yang 30 kali lebih cepat dan penggunaan memori 60% lebih hemat.

Klaim tersebut dapat mereka katakan dengan percaya diri karena Unsloth dirancang secara spesifik untuk mengakselerasi proses fine-tuning pada Large Language Models (LLM) dengan cara merekayasa ulang operasi komputasi matematika langsung di tingkat kernel Triton dan CUDA.

Modifikasi arsitektur tingkat rendah ini memungkinkan Unsloth untuk memangkas konsumsi VRAM GPU secara drastis dan mempercepat waktu komputasi pelatihan secara signifikan dibandingkan implementasi native konvensional. Hal ini dicapai dengan tetap mempertahankan akurasi metrik loss secara absolut.

Library ini secara otomatis menangani integrasi teknis QLoRA, manajemen gradient checkpointing, dan optimalisasi alokasi perangkat keras. Dengan demikian, proses pelatihan model berskala miliar parameter menjadi jauh lebih efisien dan stabil.

Selain itu, ada satu hal yang menarik untuk di-highlight mengenai perbedaan Unsloth dari Native Ways sebelumnya. Pada pendekatan Native sebelumnya, kita perlu memanggil AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, dan juga prepare_model_for_kbit_training.

*Nah*, Unsloth tampil dengan lebih ringkas yang mana kita cukup memanggil satu saja, yaitu FastLanguageModel.from_pretrained(). Ini dikarenakan Unsloth melakukan semua optimasi k-bit secara otomatis di latar belakang.

Penasaran dengan implementasinya? Mari kita terjun ke laboratorium untuk memulai eksperimen ini!

### Persiapan Dependencies dan GPU

Seperti biasa kita akan tetap mengawali eksperimen ini dengan mempersiapkan dependencies yang akan digunakan. Jadi, langsung saja kita jalankan kode berikut ini.

```python
import os, re
if "COLAB_" not in "".join(os.environ.keys()):
    !pip install unsloth  # Do this in local & cloud setups
else:
    import torch; v = re.match(r'[\d]{1,}\.[\d]{1,}', str(torch.__version__)).group(0)
    xformers = 'xformers==' + {'2.10':'0.0.34','2.9':'0.0.33.post1','2.8':'0.0.32.post2'}.get(v, "0.0.34")
    !pip install sentencepiece protobuf "datasets==4.3.0" "huggingface_hub>=0.34.0" hf_transfer
    !pip install --no-deps unsloth_zoo bitsandbytes accelerate {xformers} peft trl triton unsloth
!pip install transformers==4.56.2
!pip install --no-deps trl==0.22.2
```

Dalam kode di atas, terdapat beberapa library asing baru yang berbeda dengan pendekatan native sebelumnya. Jika Anda perhatikan di awal, kita membuat sebuah kondisi if-else. Kondisi tersebut digunakan untuk mendeteksi skrip dijalankan di dalam Google Colab atau tidak. Jika iya, skrip akan secara otomatis membaca versi PyTorch yang terinstal dan mencocokkannya dengan versi xformers yang paling kompatibel untuk mencegah konflik dependensi.

Oke sekarang mari kita bahas beberapa library “asing” yang diinstal pada kode di atas.

*   `unsloth` dan `unsloth_zoo`: Ini adalah modul utama yang menjadi inti optimasi kita. Library ini mengambil alih eksekusi model dan memodifikasi operasi matematika di tingkat perangkat keras GPU untuk memaksimalkan efisiensi komputasi dan meminimalisir penggunaan VRAM.
*   `xformers` dan `triton`: Keduanya adalah library esensial di balik layar Unsloth. xformers menyediakan optimasi arsitektur attention mechanism yang efisien, sedangkan triton memungkinkan eksekusi skrip komputasi tingkat rendah (GPU kernel) dengan kecepatan sangat tinggi.
*   `hf_transfer`: Modul ini diinstal untuk memaksimalkan bandwidth jaringan sehingga proses pengunduhan model berukuran besar dari server Hugging Face dapat berjalan jauh lebih cepat.

*Nah*, apakah Anda sadar bahwa kita menambahkan Flag --no-deps setiap kali melakukan instalasi beberapa library?

Flag tersebut mencegah penginstal otomatis menimpa versi library dasar, ini seperti berkata ke sistem seperti ini *“Instal package ini saja, JANGAN instal package tambahan (dependency) yang dibutuhkan olehnya.”*.

Mengapa ini penting di Unsloth?

Library seperti Unsloth membutuhkan versi spesifik dari torch, xformers, atau triton agar kernel cepatnya bisa jalan. Jika menginstal Unsloth tanpa `--no-deps`, `pip` mungkin akan mencoba meng-*update* torch Anda ke versi terbaru yang justru tidak kompatibel dengan hardware atau kernel Unsloth.

*Alright*, setelah semua terinstal dengan perfect, saatnya kita memanggil modul, fungsi, ataupun alat yang akan kita gunakan dengan menjalankan kode berikut ini.

```python
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer, SFTConfig
import torch
from datasets import load_dataset
import wandb
```

Pada implementasi Unsloth, urutan penulisan import di atas bukanlah sebuah kebetulan, melainkan aturan teknis komputasi yang sangat krusial. Dalam dunia Deep Learning, ada fenomena yang disebut dengan Library Side-Loading dan Memory Pre-allocation. Anda wajib mengimpor FastLanguageModel dan modul bawaan unsloth lainnya di baris paling atas sebelum mengimpor trl, transformers maupun torch.

Mengapa demikian? Ada beberapa alasan sebenarnya.

*   **Monkey patching** 
    Saat kita menulis from unsloth import FastLanguageModel, Unsloth melakukan sesuatu yang disebut dengan "Monkey Patching" pada library torch dan transformers. Unsloth akan mengganti fungsi-fungsi standar PyTorch yang lambat dengan kernel Triton miliknya yang jauh lebih cepat. Jika Anda meng-*import* torch atau transformers dan melakukan operasi tertentu sebelum Unsloth dipanggil, "tambalan" ini bisa gagal atau tidak teraplikasi secara sempurna.
*   **Inisialisasi CUDA yang Bersih** 
    Unsloth mengatur cara GPU mengalokasikan memori sejak awal. Dengan memanggil Unsloth di baris pertama, kita memastikan bahwa tidak ada library lain yang "mencuri" atau mengunci sebagian VRAM sebelum Unsloth sempat menerapkan optimasi hemat memorinya.
*   **Integrasi TRL** 
    SFTTrainer dari library trl diletakkan setelah Unsloth karena ia bertugas sebagai "wadah" pelatihan. TRL perlu mendeteksi model yang sudah dioptimasi oleh Unsloth agar fitur-fitur seperti *efficient gradient checkpointing* bisa berjalan otomatis tanpa konfigurasi manual yang rumit.
*   **Penanganan W&B (Weights & Biases)** 
    Meng-*import* wandb di akhir memastikan bahwa semua metadata model dan sistem yang sudah di-*patch* oleh Unsloth dapat tertangkap dengan benar saat proses logging dimulai.

Oleh karena itu, Anda harus memastikan urutan import yang dilakukan sudah sesuai dengan yang dicontohkan pada kode di atas. Sekarang, mari kita bedah beberapa modul yang kita panggil pada kode di atas.

*   `FastLanguageModel`: Ini adalah *class* utama dari Unsloth yang menggantikan AutoModelForCausalLM. Fungsi ini dirancang khusus untuk memuat arsitektur model dasar dan menginisialisasi matriks bobot secara jauh lebih cepat dibandingkan metode konvensional tersebut.
*   `get_chat_template`: Sebuah *utility* khusus dari Unsloth yang memetakan dan menerapkan chat template secara otomatis secara presisi. Fungsi ini akan meminimalisasi potensi kesalahan format pada dataset instruksi Anda.
*   `SFTTrainer` dan `SFTConfig`: Sama seperti pada eksekusi native sebelumnya, kelas dari pustaka TRL ini bertanggung jawab untuk mengatur hyperparameter pelatihan dan mengelola iterasi pembaruan bobot model.

Setelah kerangka kerja kita sudah siap, saatnya kita lanjut ke tahap berikutnya, yaitu memeriksa keberadaan GPU yang kita gunakan.

```python
if torch.cuda.is_available():
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU Detected: {gpu_stats.name}. Max Memory = {max_memory} GB.")
else:
    print("WARNING: GPU Not Detected.")
```

Kode ini masih sama dengan cara native sebelumnya. Di sini, kita hanya perlu memastikan *output*-nya menampilkan nama GPU (misalnya Tesla T4 atau A100). Sekarang, kita lanjut ke tahap yang juga masih sama dengan pendekatan sebelumnya, yaitu langkah opsional WandB.

### [Opsional] Persiapan WandB untuk Report

Sama seperti sebelumnya, jika Anda ingin menjalankan kode ini untuk membuat proses fine-tuning ini terecord ke Dashboard WandB, pastikan untuk mendapatkan Token atau API Key menggunakan akun milik sendiri. Setelah itu masukkan token tersebut dan baru jalankan kode di bawah ini.

```python
USE_WANDB = True
WANDB_TOKEN = "TOKEN_WANDB_ANDA"


if USE_WANDB:
    os.environ["WANDB_API_KEY"] = WANDB_TOKEN
    wandb.init(project="Llama-3.1-Benchmark", name="Unsloth_Run")
    report_target = "wandb"
else:
    os.environ["WANDB_DISABLED"] = "true" 
    report_target = "none"
```

<br>

### Panggil Model dan Tokenizer

*Nah*, tahap selanjutnya ini akan sedikit berbeda karena kita tidak lagi menggunakan `AutoModelForCausalLM` dan `AutoTokenizer` secara terpisah beserta konfigurasi `BitsAndBytesConfig` yang terpisah seperti pada pendekatan native sebelumnya. Sebagai gantinya, Unsloth menyederhanakan dan mengakselerasi seluruh proses pemuatan ini ke dalam satu pemanggilan fungsi tingkat tinggi yang sudah teroptimasi.

Mari jalankan kode berikut untuk memuat model dan tokenizer ke dalam memori secara bersamaan.

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.1-8B",
    max_seq_length = 2048,
    load_in_4bit = True,
    load_in_8bit = False,
    full_finetuning = False,
    dtype = None
)
```

Mari kita bedah secara spesifik konfigurasi komputasi yang beroperasi di dalam fungsi FastLanguageModel.from_pretrained tersebut.

*   `max_seq_length = 2048`: Di sini, kita mendefinisikan batas maksimal jumlah token yang dapat diproses oleh model dalam satu *context window*. *Yup,* `max_seq_length` ini tentu jauh lebih panjang ketimbang pendekatan native sebelumnya. Unsloth memiliki integrasi optimal untuk RoPE (Rotary Position Embedding) scaling sehingga sistem dapat memproses urutan teks panjang dengan konsumsi memori yang jauh lebih linier dan hemat dibandingkan implementasi standar Hugging Face.
*   `load_in_4bit = True`: Jika sebelumnya kita harus menuliskan BitsAndBytesConfig secara manual, di sini kita cukup melakukannya dengan satu baris parameter. Ini akan secara langsung menginstruksikan sistem untuk langsung memuat matriks bobot dasar model dalam format presisi terkompresi 4-bit.
*   `load_in_8bit = False` dan `full_finetuning = False`: Kita mengatur `load_in_8bit` menjadi False karena mode 4-bit telah diaktifkan. Selain itu, parameter `full_finetuning = False` secara tegas menonaktifkan kalkulasi gradien pada seluruh arsitektur utama model atau yang kita sebut sebagai full fine-tuning.
*   `dtype = None`: Ini mirip seperti deteksi otomatis pada SFTConfig sebelumnya, sistem akan memindai kemampuan perangkat keras komputasi Anda (GPU). Jika arsitektur BFloat16 didukung oleh GPU, Unsloth akan menggunakannya. Namun, jika tidak, ia akan mengalokasikannya sebagai Float16.

Setelah model dasar berhasil dimuat dengan presisi kompresi 4-bit, tahapan selanjutnya adalah standardisasi format chat yang akan dibaca oleh model Llama-3.1-8B ini. Nah, jika sebelumnya dalam pendekatan native kita menggunakan `setup_chat_format`, kali ini tentu berbeda. Mari kita gunakan `get_chat_template` dengan menjalankan kode di bawah ini.

```python
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3.1"
)
```

`get_chat_template` memang didesain oleh Unsloth untuk menggantikan proses konfigurasi manual atau penggunaan `setup_chat_format` dari pustaka eksternal. *Nah*, argumen `chat_template = "llama-3.1"` akan menginstruksikan Unsloth untuk secara otomatis mengintegrasikan skrip Jinja template resmi yang dikalibrasi khusus untuk Llama 3.1. Proses ini menyuntikkan token-token spesial dari Llama ke dalam vocabulary tokenizer, mirip seperti pendekatan native sebelumnya. Contohnya token `<|start_header_id|>`, `<|end_header_id|>`, dan `<|eot_id|>`.

*Nah*, setelah model sudah terkuantisasi dan tokenizer terstandarisasi, langkah berikutnya tentu saja adalah menanamkan Low-Rank Adaptation untuk mewujudkan QLoRA.

### Konfigurasi PEFT

Berbeda dengan eksekusi native yang membutuhkan inisialisasi kelas LoraConfig secara terpisah, Unsloth menyederhanakan dan mengoptimalkan proses integrasi ini melalui satu fungsi komprehensif, yaitu `FastLanguageModel.get_peft_model()`.

Mari jalankan kode di bawah ini.

```python
model = FastLanguageModel.get_peft_model(
    model,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    r = 16,
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing="unsloth"
)
```

Mari kita bedah secara komprehensif parameter teknis yang digunakan dalam konfigurasi optimasi Unsloth di atas.

*   `target_modules = [...]`: Nah, inilah perbedaan signifikan dibandingkan eksekusi native kita sebelumnya. Selain menargetkan modul Attention Mechanism (q_proj, k_proj, v_proj, o_proj), kita kini juga mengekspansi pemasangan matriks LoRA pada lapisan Feed-Forward Network. Seperti yang dibuktikan pada eksperimen di suatu paper yang kita jelaskan di materi sebelumnya, menargetkan seluruh lapisan linear ini secara matematis terbukti menghasilkan kualitas metrik evaluasi yang sangat mendekati Full Fine-Tuning.
*   `r = 16`: Menetapkan ukuran dimensi matriks (Rank) LoRA pada angka 16. Nilai ini memberikan kapasitas pembelajaran yang lebih besar dibandingkan rank 8 pada eksperimen native sebelumnya, memungkinkan model menangkap pola data instruksi yang lebih kompleks.
*   `lora_alpha = 16`: Menentukan faktor skala matematis untuk pembaruan bobot LoRA. Dalam konfigurasi Unsloth, menetapkan nilai lora_alpha setara dengan nilai r (rasio 1:1) merupakan standar komputasi yang sering kali menghasilkan stabilitas optimal selama proses kalkulasi loss.
*   `lora_dropout = 0`: Parameter *dropout* di sini kita nonaktifkan. Sebab, arsitektur Unsloth dirancang secara spesifik untuk memproses perhitungan matriks tanpa dropout. *Nah*, mengaktifkan dropout (nilai > 0) akan memaksa Unsloth kembali menggunakan implementasi PyTorch standar yang lebih lambat dan memakan lebih banyak VRAM.
*   `bias = "none"`: Sistem diinstruksikan untuk tidak melatih parameter bias pada arsitektur model, murni untuk menghemat alokasi memori komputasi selama backward pass.
*   `use_gradient_checkpointing="unsloth"`: Ini adalah salah satu fitur optimasi paling vital dari Unsloth. Alih-alih menggunakan implementasi gradient checkpointing bawaan PyTorch atau Hugging Face, Unsloth menggunakan algoritma checkpointing kustom miliknya. Algoritma ini memanipulasi offloading memori ke CPU dan secara drastis memangkas penggunaan VRAM GPU hingga 30% lebih rendah selama komputasi kalkulasi gradien berlangsung, memungkinkan pemrosesan *batch size* atau *sequence length* yang lebih besar.

*Yup*, jika Anda sadari, hampir seluruh parameter yang kita definisikan di atas nilainya lebih besar ketimbang pendekatan native sebelumnya. Hal ini dapat terealisasikan berkat “mukjizat” optimasi yang dibawakan Unsloth.

### Persiapan Dataset

Sebagai disclaimer, dari langkah ini hingga langkah penyelesaian monitoring WandB di akhir pada dasarnya sama persis dengan pendekatan native sebelumnya. So, kita akan *runthrough* saja langsung *step-by-step*-nya.

Mari kita jalankan kode di bawah untuk memanggil dataset yang sama dengan yang digunakan di pendekatan native sebelumnya.

```python
dataset = load_dataset("cahya/alpaca-id-cleaned", split = "train")
print(f"Jumlah data training: {len(dataset)}")
```

*Yup*, setelah data kita siap, saatnya melakukan f*ormatting* dengan fungsi yang mirip hanya saja kali ini tokenizer kita menggunakan get_chat_template dari Unsloth.

```python
system_prompt = "Kamu adalah asisten AI yang membantu menjawab pertanyaan pengguna berdasarkan konteks yang diberikan."

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []


    for instruction, input, output in zip(instructions, inputs, outputs):
        user_content = f"{instruction}\n\nContext:\n{input}" if input else instruction
        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output}
        ]
        text = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
        texts.append(text)
    return { "text" : texts }


dataset = dataset.map(formatting_prompts_func, batched=True)
```

Setelah dataset sudah terpetakan oleh format yang sesuai standar, sekarang kita akan melihat sedikit bentuk jadinya dengan menjalankan kode berikut.

```python
dataset[100]["text"]
```

*Output*-nya kira-kira akan tampak seperti ini.

![dos-86e77a07644280974ee7a0968753ed4f20260311172037.png](https://assets.cdn.dicoding.com/original/academy/dos-86e77a07644280974ee7a0968753ed4f20260311172037.png)

Model, tokenizer, hingga dataset sudah *ready*, saatnya kita masuk ke tahap *training*.

### Supervised Fine-tuning

*Alright*, seperti yang di-*mention* sebelumnya kita tidak akan merubah apa pun pada tahap ini. Jadi, Anda dapat langsung menjalankan kode berikut untuk membuat konfigurasi SFT-nya.

```python
sft_config = SFTConfig(
    output_dir="outputs_unsloth",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    gradient_checkpointing=False,
    warmup_steps=5,
    max_steps=30,
    learning_rate=2e-4,
    lr_scheduler_type="linear",
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=1,
    optim="paged_adamw_8bit",
    dataset_text_field="text",
    max_length=512,
    report_to=report_target,
    run_name="Unsloth_Llama3.1_Run" if USE_WANDB else None
)
```

Setelah konfigurasi selesai, saatnya kita membungkus semua yang sudah kita buat mulai dari model hingga SFTConfig sebelumnya ke dalam SFTTrainer.

```python
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    args=sft_config,
)
```

*Nah*, sekarang mari kita lihat berapa jumlah parameter Llama-3.1-8B yang akan kita *fine-tune* menggunakan Unlsloth. Secara diagnosis awal, jumlah parameter yang ada disini akan jauh lebih banyak dibandingkan pendekatan native sebelumnya.

*So, let’s find out* dengan menjalankan kode berikut ini.

```python
trainer.model.print_trainable_parameters()
```

*And here we go*, outputnya akan terlihat seperti ini.

```python
trainable params: 41,943,040 || all params: 8,072,204,288 || trainable%: 0.5196
```

Sesuai dugaan kita di awal, jumlah parameternya melonjak sangat drastis sekitar 41,9 juta dibandingkan pendekatan Native sebelumnya yang hanya 6,8 juta.

Peningkatan komputasi ini merupakan hasil matematis langsung dari konfigurasi Unsloth sebelumnya dengan melipatgandakan dimensi matriks menjadi `r=16` dan mengekspansi target injeksi LoRA ke seluruh lapisan FFN atau MLP (`gate_proj`, `up_proj`, `down_proj`), selain lapisan Attention.

Hasil ini tentu akan menjadi ujian yang menarik untuk arsitektur kernel Triton bawaan Unsloth dengan klaim 30 kali lebih cepat. Apakah ini fakta atau hanya gimmick?

Mari kita buktikan dengan menjalankan kode berikut ini.

```python
trainer.train()
```

<br>
*Output*-nya akan tampil log interaktif sama seperti sebelumnya dengan tambahan ilustrasi sloth di atasnya.

![dos-274aca51b6cec47e21703f3a3d6d95e120260311172050.png](https://assets.cdn.dicoding.com/original/academy/dos-274aca51b6cec47e21703f3a3d6d95e120260311172050.png)

Bagaimana hasilnya? Sangat di luar dugaan, bukan?

Kita memang sudah berekspektasi bahwa Unsloth akan jauh lebih cepat dari pendekatan native sebelumnya. Namun, dengan jumlah parameter yang sekitar 6 kali lebih banyak, Unsloth dapat menyelesaikan proses fine-tuning hanya dalam waktu 2 setengah menit atau sekitar 17 kali lebih cepat dari native sebelumnya. Ini di luar batas ekspektasi kita. Ternyata klaim mereka yang membanggakan arsitektur Triton-nya bukanlah gertakan sambal belaka.

Oh iya, jangan lupa untuk mematikan monitoring WandB jika Anda melakukannya.

```python
if USE_WANDB:
    wandb.finish()
```

*Alright*, sekarang kita akan menguji kapabilitas *text generation* dari model Unsloth secara langsung untuk melihat respons aktualnya.

### Inference

Masih sama dengan native, kita perlu mengaktifkan mode inference dari model ini. Silakan jalankan kode berikut ini.

```python
FastLanguageModel.for_inference(model)
```

*Nah*, berbeda dengan eksekusi Native sebelumnya yang hanya mengandalkan pemanggilan metode `model.eval()` standar dari ekosistem PyTorch, Unsloth menyediakan fungsi optimasi yang jauh lebih superior untuk fase pengujian ini. Mari kita bedah operasi teknis di baliknya.

*   **Akselerasi Decoding Tingkat Kernel** 
    Memanggil `FastLanguageModel.for_inference(model)` akan menginstruksikan arsitektur Unsloth untuk mengalihkan seluruh operasi matriks dari mode training ke mode decoding. Fungsi ini secara otomatis mengaktifkan kernel Triton khusus inferensi yang mampu mengeksekusi algoritma autoregressive hingga 2 kali lebih cepat dibandingkan implementasi Hugging Face standar.
*   **Manajemen Status Model** 
    Secara internal, metode ini tidak hanya mematikan fitur dropout, tetapi juga memastikan bahwa seluruh infrastruktur kalkulasi gradien dinonaktifkan sepenuhnya di tingkat alokasi memori. Hal ini menjamin bahwa model akan memberikan probabilitas output yang deterministik dan konsisten, sekaligus meminimalisasi konsumsi VRAM GPU selama proses prediksi token berlangsung.

Hanya dengan satu baris kode tersebut, model Anda kini telah dikalibrasi secara maksimal untuk menghasilkan teks dengan kecepatan dan efisiensi tertinggi. *Yup*, optimasi yang dihadirkan oleh Unsloth tidak hanya terbatas pada proses *training*, tetapi juga merambah ke fase *inference*.

*Alright*, langkah selanjutnya tentu saja adalah menyiapkan format tanya-jawab untuk model yang caranya masih sama dengan pendekatan native.

```python
messages = [
    {"role": "system", "content": "Anda adalah asisten cerdas yang menjawab pertanyaan secara akurat berdasarkan fakta."},
    {"role": "user", "content": "Sebutkan 3 candi terkenal di Indonesia beserta lokasinya!"}
]


inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to("cuda")
```

*Finally*, kita akan sampai pada garis finish dari marathon panjang fine-tuning ini dengan masuk ke tahap komputasi *inference* dengan Unsloth. Tahap komputasi ini akan memanfaatkan secara penuh kernel Triton bawaan Unsloth yang telah diaktifkan sebelumnya, mengeksekusi algoritma autoregressive dengan tingkat akselerasi komputasi yang maksimal.

Mari kita jalankan kode berikut ini untuk melihat jawaban model yang di-training dengan Unsloth dari pertanyaan *“Sebutkan 3 candi terkenal di Indonesia beserta lokasinya!”*.

```python
text_streamer = TextStreamer(tokenizer, skip_prompt=True)

_ = model.generate(
    input_ids=inputs,
    streamer=text_streamer,
    max_new_tokens=256,
    temperature=0.3,
    repetition_penalty=1.2,
    do_sample=True,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id
)
```

*Yup*, jika Anda perhatikan, kita tidak lagi menggunakan context manager ini. Hal ini terjadi karena pemanggilan fungsi `FastLanguageModel.for_inference(model)` pada langkah sebelumnya telah secara otomatis menonaktifkan seluruh pelacakan gradien di tingkat arsitektur dasar. Ini membuat penulisan kode inferensi Unsloth menjadi lebih ringkas.

Selain itu, penggunaan variabel Dummy (` _ =` ) sebagai penampung variabel adalah konvensi Python untuk menandakan bahwa kita mengeksekusi fungsi model.generate, tetapi kita mengabaikan return value berupa tensor. Karena sudah mencetak output secara langsung ke konsol menggunakan `text_streamer`, kita tidak perlu lagi menyimpan matriks tensor final tersebut dalam memori RAM sehingga alokasi memori menjadi lebih efisien.

Selebihnya, konfigurasi hyperparameter yang digunakan tidak ada yang berbeda dengan pendekatan native. Saat kode tersebut dieksekusi, *voila!*, Anda akan melihat output probabilitas teks yang diproduksi secara real-time di layar konsol, memvalidasi bahwa model Llama 3.1 Anda telah sukses mengadaptasi dataset instruksi berbahasa Indonesia melalui akselerasi komputasi Unsloth.

Berakhirnya babak Unsloth ini menutup tahapan modifikasi matriks bobot dengan Parameter-Efficient Fine-Tuning. Meskipun metode QLoRA terbukti secara matematis mampu memangkas alokasi VRAM selama fase pelatihan, arsitektur dasar model yang dieksekusi pada saat inferensi tetaplah berada pada skala 8 miliar parameter.

Lantas, apakah berarti tidak ada jalan untuk memasukkan model 8 miliar parameter ini ke dalam perangkat ringan seperti smartphone?

*Nah*, untuk mengatasi hambatan perangkat keras ini, rekayasa Machine Learning beralih ke paradigma yang dikenal sebagai **Knowledge Distillation**.

***

## Knowledge Distillation

Halo para “Pencari Efisiensi”, kita telah melalui perjalanan panjang yang luar biasa di dunia efisiensi pengembangan Generative AI. Alangkah baiknya jika kita refleksi sejenak hal-hal yang sudah kita pelajari.

*   Melalui Quantization, kita belajar cara mengompres ukuran fisik model agar muat di VRAM yang terbatas seperti INT8 dan NF4.
*   Melalui PEFT (LoRA), kita belajar cara 'mengakali' proses training agar kita tidak perlu melatih miliaran parameter, cukup melatih jalur tikus (Matriks A dan B) saja.

Jika Anda perhatikan, ada satu hal yang belum tersentuh oleh teknik efisiensi yang sudah kita eksplorasi, yaitu **kuantitas parameter** modelnya.

Sebuah model raksasa berukuran 70 miliar parameter (70B) yang meskipun sudah dikuantisasi menjadi 4-bit dan dipasangi LoRA, ia tetaplah model 70B. Saat ia digunakan untuk inference, komputer tetap harus menjalankan perhitungan matematis melewati 70 miliar koneksi tersebut.

Bagi perusahaan besar dengan server kuat, ini bukan masalah. Namun, bagaimana jika kita ingin menanamkan AI super cerdas ini ke dalam sebuah *smartphone*? Model 70B jelas terlalu berat.

Nah, keresahan ini memicu para ilmuwan AI untuk mengajukan sebuah pertanyaan yang terdengar seperti cerita dongeng fiksi ilmiah.

*"Bisakah kita mengekstrak kebijaksanaan, logika, dan kecerdasan dari otak model raksasa, lalu 'menuangkannya' ke dalam otak model yang jauh lebih kecil?"* 

Siapa sangka ternyata jawabannya, **bisa**.

Oleh karena itu, selamat datang di dunia baru yang kita sebut Knowledge Distillation.

### Konsep Knowledge Distillation

Mungkin Anda menyadari bahwa teori dan penjelasan dalam kelas ini bukanlah sekadar salinan kaku dari jurnal ilmiah, buku, atau dokumentasi teknis yang rumit.

Setiap materi yang Anda baca di sini merupakan hasil dari penyederhanaan logika dan proses transfer pemahaman melalui media yang lebih intuitif, seperti ilustrasi visual dan analogi. Tujuannya tentu saja jelas, agar teori dari teknik serumit LoRA sebelumnya dapat terserap ke dalam pikiran Anda tanpa hambatan teknis yang membosankan.

Dalam dunia AI, proses transfer pemahaman ini dikenal dengan istilah Knowledge Distillation. Mungkin Anda bertanya, *"Lho, bukannya itu sama saja dengan Transfer Learning?"*

Jawabannya sangat jelas berbeda. Coba perhatikan tabel berikut.

| | **Transfer Learning** | **Knowledge Distillation** |
|---|---|---|
| **Analogi** | **Melanjutkan** studi ke jenjang yang lebih spesifik. | Seorang Guru jenius **mengajarkan** intisari ilmunya kepada murid. |
| **Mekanisme** | Mengambil model yang sudah terlatih, lalu menyesuaikannya untuk tugas baru. | Model besar melatih model yang lebih kecil agar memiliki kemampuan serupa. |
| **Hasil Akhir** | Model tetap berukuran besar dengan keahlian baru. | Model menjadi jauh lebih ringan dan cepat, tetapi tetap cerdas. |

<br>

*Yup*, Knowledge Distillation adalah sebuah teknik kompresi model yang pertama kali dipopulerkan oleh Geoffrey Hinton, dkk., pada tahun 2015. Pendekatan kompresi yang diambil adalah dengan cara melatih sebuah model kecil untuk meniru perilaku, output, dan pola pikir dari sebuah *pre-trained model* yang jauh lebih raksasa.

Berbeda dengan teknik Quantization yang “memotong” ukuran angka biner, atau LoRA yang “mengakali” proses update bobot, Distillation adalah proses “mewariskan” ilmu beda arsitektur.

Untuk memahami bagaimana ilmu ini ditransfer, kita harus berkenalan dengan dua aktor utama di atas panggung ini.

### Konsep Teacher dan Student Model

Dalam ekosistem Knowledge Distillation, kita tidak sedang membicarakan satu model yang bekerja sendirian. Sebaliknya, kita sedang melihat sebuah proses "mentoring" digital di mana ada dua aktor utama dengan peran yang sangat kontras, tetapi saling melengkapi, yaitu Teacher dan Student Model.

![dos-7995b96cca9ee3016b6ff153b126363f20260330152634.png](https://assets.cdn.dicoding.com/original/academy/dos-7995b96cca9ee3016b6ff153b126363f20260330152634.png)

Mari kita *breakdown* kedua aktor tersebut dan simbiosis yang terjadi pada keduanya

*   **Teacher Model** 
    Pada dasarnya, ini adalah LLM berukuran raksasa entah itu model proprietary seperti Gemini dan ChatGPT, maupun model open-source seperti Llama-3 70B atau Kimi K2. Perannya dalam Knowledge Distillation sebagai *Ground Truth* yang baru dengan memberikan instruksi dan contoh jawaban yang benar. *Yup*, kita tidak lagi mengotak-atik puluhan miliar atau triliun Weights dan parameter di dalamnya.
*   **Student Model** 
    Kebalikan dari Teacher, Student Model adalah model bahasa yang ukurannya jauh lebih ringkas dan arsitekturnya lebih sederhana. Dapat dikatakan, jika Teacher Model LLM, maka Student Model itu adalah SLM (Small Language Model), misalnya Llama-3 8B, Qwen 1.5B, atau model berparameter jutaan. Nah, Student Model ini yang akan kita “ajari” untuk menghasilkan jawaban yang mendekati kualitas dari Teacher Model-nya.

Setelah melihat ilustrasi di atas dan mendengar simbiosis yang terjadi antara kedua model tersebut, Anda mungkin bertanya.

*"Kenapa repot-repot pakai Guru? Kenapa Student Model tidak dilatih saja langsung menggunakan dataset asli yang dipakai oleh Gurunya?"* 

Jika murid hanya dilatih menggunakan dataset asli, ia sekadar menghafal "kunci jawaban" mutlak yang kaku dan hitam-putih. Ia akan kehilangan kesempatan emas untuk menyerap insting, pola pikir, dan "pengalaman hidup" Teacher Model dalam melihat nuansa yang ada pada data.

*Nah*, masalahnya sekarang adalah *“Di sebelah mana tepatnya ‘Si Pengalaman Hidup’ ini disimpan di dalam Teacher Model?”*. Dengan mengetahui tempat pengetahuan ini disimpan, kita baru dapat maju untuk mewariskan ilmu ini ke Student Model yang lebih kecil.

### Membedah Anatomi Dark Knowledge

Di sub-modul ini, kita akan melakukan pembedahan terhadap Anatomi Pengetahuan.

Para peneliti AI telah memetakan bahwa 'ilmu' di dalam Neural Network terbagi menjadi tiga wujud fisik utama.

Mari kita bedah ketiga anatomi ini satu per satu berdasarkan literatur fundamental AI, lalu kita akan temukan alasan mengapa pada akhirnya industri modern hanya mengandalkan dua paradigma saja dalam praktiknya.

#### Response-based Distillation

Sering juga disebut Response-Based Knowledge, bentuk ini adalah pondasi paling klasik dari Knowledge Distillation. Pada tahap ini, pengetahuan diekstrak murni dari final output layer milik Teacher Model.

![dos-8cee3709ca5107685ad5ad2a760adbc020260430092308.png](https://assets.cdn.dicoding.com/original/academy/dos-8cee3709ca5107685ad5ad2a760adbc020260430092308.png)

Pendekatan ini menggunakan *soft label* yang dapat berupa logits atau probabilitas dari teacher model sebagai target utama untuk melatih student. Nantinya, student model akan belajar untuk mereproduksi bukan hanya jawaban yang benar, tetapi juga “tingkat keyakinan” teacher model terhadap jawaban-jawaban tersebut, dan bahkan ikut mempelajari kesalahan yang dibuat.

*Yup*, pendekatan Response-based Distillation ini adalah pendekatan yang kita lihat sebelumnya dalam ilustrasi model klasifikasi.

#### Feature-based Distillation

Terkadang, sekadar mengetahui jawaban akhir dari sang guru tidaklah cukup, terutama jika kita berhadapan dengan neural networks yang sangat dalam. *Nah*, di tahap inilah Feature-based Knowledge mengambil peran.

![dos-641266717a6dcd609dd6846226e6d52c20260430092328.png](https://assets.cdn.dicoding.com/original/academy/dos-641266717a6dcd609dd6846226e6d52c20260430092328.png)

Pendekatan ini berfokus pada informasi yang berada di Hidden Layer Teacher Model. *Nah*, informasi yang diambil dari hidden layers dalam Feature-based Distillation berupa Feature Maps. Dengan mencocokkan feature maps, Student Model bisa menyerap pola-pola spesifik yang membentuk keputusan akhir, misalnya mengenali *"oh ini ada garis lengkung"*, *"ini tekstur berbulu"*, atau *"ini pola tata bahasa tertentu"*.

#### Relation-based Distillation

Jika dua pendekatan sebelumnya berfokus pada elemen-elemen individual baik itu output akhir maupun feature maps, Relation-based Knowledge melihat pada gambaran yang lebih besar, yaitu interaksi dan hubungan antar elemen tersebut.

![dos-7e80ea47627dd3491f80f7024f80ea9320260430092354.png](https://assets.cdn.dicoding.com/original/academy/dos-7e80ea47627dd3491f80f7024f80ea9320260430092354.png)

Jika Feature-based Distillation memaksa murid untuk "memfotokopi" Feature Maps guru sama persis, Relation-based Distillation menuntut murid untuk menyalin hasil perhitungan jarak atau korelasi dari Feature Maps tersebut.

Sedikit abstrak, ya? Wajar saja karena pendekatan ini memang yang paling abstrak dibandingkan dua sebelumnya. Mari kita gunakan *use case* klasifikasi gambar sebelumnya untuk memberikan gambaran cara kerja dari Relation-based ini.

*   **Tugas Teacher:** Teacher model menerima sekumpulan gambar sekaligus misalnya gambar Kucing, Anjing, dan Burung. Kemudian ia akan menghasilkan Feature Maps untuk ketiganya.
*   **Informasi yang Ditransfer:** Sistem akan menghitung jarak vektor seperti cosine similarity dari Feature Maps tersebut. Hasilnya berupa matriks yang menyatakan *"Menurut Guru, kemiripan kucing dan anjing adalah 0.8, sedangkan kucing dan mobil adalah 0.1"*.
*   **Tugas Student:** Barulah Student Model menghasilkan Feature Maps-nya sendiri, lalu menghitung jaraknya. Student harus memastikan jarak antara kucing dan anjing di dalam otak komputasinya sama dengan jarak yang dipahami oleh guru, yaitu 0.8.

*Eits*, wujud informasi dalam Relation-based Knowledge tidak hanya terpaku pada Feature Maps, intinya ia merupakan matriks relasional yang dapat berupa hubungan antar-lapisan dalam model itu sendiri, atau bahkan interaksi antar-sampel data yang berbeda.

Melihat betapa lengkapnya ketiga anatomi di atas, rasanya ideal jika kita menggunakan ketiganya agar “Sang Murid” menjadi kloningan sempurna. Terutama pendekatan Feature dan Relation-based yang terdengar romantis karena mencoba meniru "pola pikir" Guru. Namun, industri sering kali mengabaikannya karena alasan pragmatis.

*   **Black Box dan Perbedaan Arsitektur** 
    Knowledge Distillation yang ideal sering kali mengharuskan Teacher dan Student Model memiliki arsitektur yang serupa agar fitur-fiturnya bisa diperbandingkan. Dalam industri, kita sering ingin mengecilkan model raksasa seperti Llama-3 70B ke model kecil yang arsitekturnya jauh berbeda, misalnya Qwen 1.5B. Mengejar kesamaan fitur internal menjadi sangat sulit secara matematis.
*   **Biaya Komputasi** 
    Menghitung loss dari hubungan antar-lapisan (Relation-based) membutuhkan memori dan waktu GPU yang masif. Industri lebih memilih model yang "cukup pintar" tapi selesai dilatih dalam 2 hari, daripada model "kloning sempurna" yang butuh waktu 2 minggu.

Praktik dalam industri biasanya akan mengerucut hanya pada dua paradigma praktis, yaitu Logit-based dan Synthetic Data.

### Move On ke Dua Paradigma yang Pragmatis

Jika anatomi sebelumnya berbicara tentang wujud ilmu yang ditransfer, paradigma berbicara tentang cara skenario pembelajaran itu diterapkan di dunia nyata. Dalam praktiknya, para periset dan engineer AI telah mengerucutkan berbagai teknik rumit tadi ke dalam dua jalur utama yang paling efisien, ekonomis, dan berdampak masif.

Mari kita bedah satu per satu kedua paradigma ini untuk melihat cara kerjanya. Lalu, kita lihat juga bagaimana karya-karya nyata seperti DistilBERT hingga inovasi terbaru seperti DeepSeek-R1-Distill-Llama-8B berhasil dilahirkan dari teknologi ini.

#### Logit-based Distillation

Apakah Anda ingat “Pengalaman Hidup” yang kita singgung di awal sebelumnya? Nah, di sinilah misteri nya akan diungkap. Bapak AI dunia, Geoffrey Hinton, menyebut “Pengalaman Hidup” berharga ini dengan istilah Dark Knowledge.

Untuk memahami Dark Knowledge, kita akan menggunakan ilustrasi model klasifikasi gambar berikut ini.

![dos-d7bd6d8f797e8e3d5f338372ee9befbc20260501173840.gif](https://assets.cdn.dicoding.com/original/academy/dos-d7bd6d8f797e8e3d5f338372ee9befbc20260501173840.gif)

Biasanya, saat kita melatih sebuah model AI secara konvensional, kita menggunakan apa yang disebut dengan Hard Targets. Contohnya adalah sistem klasifikasi gambar yang harus membedakan antara anjing dan kucing seperti ilustrasi di atas.

Jika inputnya adalah foto seekor anjing, label yang diberikan pada data mentah adalah mutlak, yaitu Anjing = 1, sedangkan sisanya 0. Cara belajar ini dapat dikatakan kaku sebab model hanya tahu mana yang benar dan mana yang salah tanpa memahami hubungan antar kategori tersebut.

*Wait*, apa yang dimaksud dengan hubungan antar kategori? Coba perhatikan klasifikasi gambar versi lainnya berikut ini.

![dos-5e7f060b89aff6c63af2f1080378143620260501173906.gif](https://assets.cdn.dicoding.com/original/academy/dos-5e7f060b89aff6c63af2f1080378143620260501173906.gif)

Perhatikan angka 31% pada label "Kucing" dan 2% pada label “Burung”. *Nah*, itulah yang disebut Dark Knowledge.

Large Model yang lebih pintar memang setuju bahwa foto tersebut adalah gambar “anjing”, tetapi ia memahami satu informasi yang gagal dipahami model kecil dengan berkata *"Ini memang anjing, tetapi perhatikan bulu dan bentuk telinganya, dia punya kemiripan visual dengan kucing, tetapi sama sekali tidak mirip dengan burung."*

Angka 29% tersebut memberi tahu model kecil bahwa meskipun gambar ini merupakan seekor anjing, tetapi ia memiliki fitur visual yang menyerupai kucing, entah mungkin karena bulunya yang panjang atau bentuk telinganya.

*Nah*, cara bernalar seperti itulah yang ingin kita ajarkan ke Small atau Student Model melalui Logit-based.

*Wait*, nama pendekatan ini kan Logit-based, tetapi kenapa ilustrasi di atas menggambarkannya menggunakan Probabilitas Softmax?

*Alright*, untuk mengoprek jawaban aslinya, kita recall terlebih dahulu pemahaman soal logits. Seperti yang kita ketahui logits adalah nilai mentah dari setiap kelas yang ingin diprediksi, dalam konteks LLM isinya adalah puluhan ribu kosakata yang ada di pengetahuannya.

Jika kita menggunakan ilustrasi klasifikasi gambar di atas, logits akan berbentuk seperti berikut ini.

![dos-875d04db12c569395ec0083699a2d3be20260330152705.png](https://assets.cdn.dicoding.com/original/academy/dos-875d04db12c569395ec0083699a2d3be20260330152705.png)

Biasanya, kita akan mengubah skor mentah ini menjadi probabilitas (persentase 0-100%) menggunakan fungsi matematika bernama Softmax. Sayangnya, Softmax biasa itu terlalu berisiko karena akan membuat logit yang paling tinggi menjadi mendominasi mutlak dan “menghancurkan” angka lainnya menjadi nyaris nol.

Misalnya, jika menggunakan contoh di atas maka akan seperti berikut.

*   **Burung:** Nilainya mungkin hanya sekitar 0,001%, karena nilai logit-nya sangat negatif, yang berarti model sangat yakin bahwa gambar tersebut bukan “Burung”.
*   **Kucing:** Nilainya berada di kisaran 4%, sehingga model tidak terlalu memperhatikan kelas “Kucing” ini.
*   **Anjing:** Nilainya akan mendominasi hingga sekitar 98%, karena model sangat yakin bahwa gambar tersebut adalah “Anjing”.

Dengan distribusi probabilitas yang seperti itu, model akan kehilangan informasi bahwa input gambar juga memiliki kemiripan dengan kucing. Jika ini terjadi, tentu saja kemampuan Dark Knowledge akan hilang.

*Nah*, agar Dark Knowledge tidak hilang dihancurkan oleh Softmax biasa, Geoffrey Hinton memperkenalkan sebuah variabel ajaib bernama Temperature (T). Coba Anda perhatikan perbedaan rumus matematisnya di bawah ini.

![dos-faabe67e5ef6130791ece290ae3c1eaa20260430093004.png](https://assets.cdn.dicoding.com/original/academy/dos-faabe67e5ef6130791ece290ae3c1eaa20260430093004.png)

*Yup*, jika Anda sadar, konsep penambahan Temperature dalam Softmax ini sama persis dengan yang terjadi dalam metode sampling yang kita bahas di modul LLM awal. Namun, meski matematikanya sama, tujuan penggunaannya sangat berbeda.

Jika dalam metode Sampling LLM temperature yang tinggi digunakan untuk menyuruh model memilih kata secara acak, dalam distillation, ia digunakan untuk mengungkap informasi yang tenggelam. Perhatikan contoh berikut.

*   `temperature=1` **(Normal):** Model sangat yakin.
    *   Anjing = 98%
    *   Kucing = 4%
    *   Burung = 0,001%
*   `temperature=5` **(Tinggi):** Skornya akan merata seperti ilustrasi klasifikasi gambar sebelumnya.
    *   Anjing = 67%
    *   Kucing = 31%
    *   Burung = 2%

*Nah*, dengan menaikkan temperature, probabilitas kata "Kucing" dan "Burung" tiba-tiba membesar dan bisa dilihat jelas oleh Student Model. Proses peleburan inilah yang menghasilkan apa yang kita sebut Soft Labels.

Sekarang muncul pertanyaan akhir, Logit-based Distillation memaksa kita untuk sedikit membedah otak LLM untuk mengambil logit yang dihasilkan oleh LLM.

Model-model paling cerdas di dunia umumnya adalah model proprietary seperti GPT-5, Claude atau Gemini 3 Pro. *Yup*, label model *proprietary* mengisyaratkan model ini adalah Black-Box yang tidak bisa diunduh, apalagi dibongkar Logits-nya. Kita hanya bisa berinteraksi dengan mereka melalui API, di mana kita memberi Prompt dan mereka menjawabnya.

Lalu, bagaimana cara kita mencuri kepintaran mereka jika kita tidak bisa melihat isi kepalanya? Jawabannya ada di pendekatan kedua yang sebenarnya sederhana ini.

#### Synthetic Data Distillation

Jika kita tidak bisa membaca pikiran Sang Guru, kita akan menyuruhnya menulis buku teks yang sangat tebal, lalu menyuruh Sang Murid membaca buku itu. *Nah*, Inilah esensi dari paradigma kedua, Synthetic Data Distillation atau Prompt-Based Distillation.

*Yup*, pendekatan ini sangat sederhana, alih-alih melakukan transfer matematika rumit, pendekatan ini murni berfokus pada pembuatan dataset buatan atau Synthetic Data. *Step by step*-nya akan terlihat seperti ini.

![dos-19c917273a1dfe55a250533d67dfcdd820260430093048.png](https://assets.cdn.dicoding.com/original/academy/dos-19c917273a1dfe55a250533d67dfcdd820260430093048.png)

*   **Querying:** Kita akan memberikan ribuan hingga jutaan instruksi atau prompt

yang sulit kepada Teacher Model.
*   **Generating:** Teacher Model kemudian akan menjawabnya dengan sangat detail, terstruktur, dan akurat. Kumpulan instruksi dan jawaban berkualitas tinggi ini kemudian dikumpulkan menjadi sebuah dataset raksasa.
*   **Fine-tuning:** Selanjutnya, Student Model dilatih dari nol atau di-*fine-tune* menggunakan dataset buatan tersebut.

Apakah Anda sadar sesuatu dari penjelasan di atas?

*Yup*, pada dasarnya Synthetic Data Distillation ini adalah Supervised Fine-tuning, tetapi menggunakan dataset sintetis yang berasal dari Teacher model yang jauh lebih pintar. *Eits*, tapi jangan salah sangka, industri justru menyukai pendekatan ini karena beberapa alasan berikut.

*   **Melewati Batasan Akses:** Kita tidak butuh akses ke arsitektur internal atau logit model raksasa. Selama kita bisa mendapatkan teks jawabannya, kita bisa melakukan distilasi.
*   **Fokus pada Reasoning:** Teacher model bisa diminta untuk menjabarkan langkah-langkah berpikirnya (Chain-of-Thought). Saat model Murid membaca data ini, ia secara otomatis ikut belajar bagaimana cara bernalar logis secara bertahap, bukan sekadar menebak jawaban akhir.

Jika Anda ingin mengetahui mahakarya nyata yang mendemonstrasikan kehebatan Synthetic Data Distillation, silahkan berkenalan dengan DeepSeek.

![dos-debcbd4f6bf443c5fb1633452bfe11b920260311172504.png](https://assets.cdn.dicoding.com/original/academy/dos-debcbd4f6bf443c5fb1633452bfe11b920260311172504.png)

DeepSeek memiliki model raksasa DeepSeek-R1 yang sangat jenius dalam matematika dan coding karena kemampuan reasoning. Mereka kemudian menggunakan model R1 untuk menghasilkan jutaan baris data reasoning (synthetic data). Data sintetis ini lalu dijadikan data fine-tuning oleh model Llama-3-8B yang bertindak sebagai Student Model.

Hasilnya, Llama-3-8B yang tadinya hanya model bahasa standar, tiba-tiba memiliki kemampuan penalaran logis dan matematika tingkat dewa yang nyaris menyaingi model-model raksasa berukuran puluhan kali lipat lebih besar.

*Wait*, *reasoning*? Apakah berarti kita bisa membuat model yang tidak memiliki kemampuan reasoning jadi memilikinya? *Eits*, tahan dulu. Rasa penasaran Anda akan terjawab pada modul selanjutnya.

Sekarang mari kita jawab terlebih dahulu pertanyaan berikut.

Sedari tadi kita selalu menyinggung bahwa Student model akan meniru Teacher model. Namun, sebenarnya bagaimana tepatnya cara Student model ini “meniru” Teacher model dalam training sesungguhnya?

Mari kita jawab, Coders!

### Training untuk Menyenangkan Dua “Pengawas”

Oleh karena proses training dari Synthetic Data Distillation sebelumnya hampir tidak ada yang berbeda dengan Supervised Fine Tuning pada umumnya, mari kita breakdown Logit-based saja agar menarik.

Seperti yang kita ketahui model AI itu seperti anak kecil yang belajar dari kesalahan. Ia menebak, lalu kita ukur seberapa meleset tebakannya (Loss). Semakin meleset, semakin besar "hukuman" yang ia terima, dan sistem akan mengoreksi jaringan "otak"-nya (backpropagation) agar besok tebakannya lebih baik.

Dalam Logit-based Distillation, proses belajarnya menjadi sangat unik karena sang Murid diawasi oleh dua pengawas ujian sekaligus.

![dos-8236ff14d8383a19a6fe2ebbb161397520260430093128.gif](https://assets.cdn.dicoding.com/original/academy/dos-8236ff14d8383a19a6fe2ebbb161397520260430093128.gif)

Seperti yang ditunjukan ilustrasi di atas, berikut adalah skenario cara melatih Murid.

*   **Inference Teacher** 
    Teacher Model yang sudah pintar akan melakukan prediksi pada data input dan mengeluarkan Soft Labels. Misalnya, *"Ini 85% Anjing, 14% Kucing, 1% Burung"*. Perlu diingat, parameter dalam Teacher Model tidak akan berubah sedikit pun karena ia tidak akan diikutkan dalam training. Di saat yang sama, kita juga memegang Dataset Asli (Label Dataset atau Hard Target) yang menyimpan Ground truth, *"Ini 100% Anjing, 0% Kucing, 0% Mobil"*.
*   **Forward Pass Student** 
    Kini giliran Student Model yang diminta untuk memprediksi data input pada fase forward pass. Oleh karena ia masih belajar, tebakan awalnya mungkin “ngawur”. Misalnya ia menebak *"Ini 60% Anjing, 10% Kucing, 30% Mobil"*.
*   **Hukuman Pertama: Distillation Loss** 
    Pengawas pertama akan membandingkan tebakan Student Model dengan tebakan Teacher Model. Fokus Student Model di sini adalah meniru Teacher Model.
    *   Guru: 85% Anjing, 14% Kucing, 1% Mobil.
    *   Murid: 60% Anjing, 10% Kucing, 30% Mobil.
    
    Sistem menggunakan rumus matematika yang disebut Kullback-Leibler (KL) Divergence. KL Divergence akan menghitung seberapa jauh selisih persentase Murid dari persentase Guru.
*   **Hukuman Kedua: Student Loss** 
    Pengawas kedua bertugas memastikan Murid tidak melupakan fakta dunia nyata. Ia membandingkan tebakan Murid langsung dengan kunci jawaban asli.
    *   Ground truth: 100% Anjing
    *   Murid: 60% Anjing
    
    Di sini sistem menggunakan rumus standar untuk Softmax, yaitu Cross-Entropy Loss.
*   **Eksekusi "Peniruan"** 
    Setelah perhitungan loss selesai, keduanya akan dijumlahkan. *Nah*, Biasanya para engineer memberikan "bobot" untuk setiap loss, misalnya 70% untuk Distillation Loss dan 30% sisanya untuk Student Loss pada data asli. Total nilai hukuman inilah yang kemudian digunakan untuk “mengoreksi” miliaran parameter di dalam otak Student model hingga dapat meniru pola probabilitas Teacher-nya dengan presisi.

*Alright*, dengan berakhirnya pembahasan mengenai sistem evaluasi ganda tadi, kita resmi menutup babak perjalanan Knowledge Distillation. Kita telah melihat bagaimana model "Murid" tidak hanya bisa menjadi cerdas, tetapi juga ringan dan efisien.

Namun, dalam pengembangan Large Language Model (LLM), menyusutkan ukuran model atau sekadar melatihnya memprediksi probabilitas kata berikutnya belum menyelesaikan masalah kualitas interaksi dan keamanannya secara tuntas.

Kita perlu fase alignment berikutnya untuk menyelesaikan permasalahan tersebut sebelum pada akhirnya label “Penjinak LLM” dapat kita taklukkan.

*Eits*, sebelum menjinakan LLM, tentu saja kita akan selingi terlebih dahulu dengan aktivitas interaktif di bawah ini.

<iframe src="https://academy-sandpack.dicoding.dev/activities/flashcards?data=eyJjYXJkcyI6W3siaWQiOiJjYXJkLTU2MC4zMDAwMDAwMDA3NDUxIiwiZnJvbnQiOiI8c3BhbiBpZD1cImRvY3MtaW50ZXJuYWwtZ3VpZC1mZmM2ZjdlOS03ZmZmLThkOTUtNjlkZC03ZDRlMjAzNjljZGNcIj48c3BhbiBzdHlsZT1cImZvbnQtc2l6ZTogMTFwdDsgZm9udC1mYW1pbHk6IEFyaWFsLCBzYW5zLXNlcmlmOyBiYWNrZ3JvdW5kLWNvbG9yOiB0cmFuc3BhcmVudDsgZm9udC12YXJpYW50OiBub3JtYWw7IHZlcnRpY2FsLWFsaWduOiBiYXNlbGluZTsgd2hpdGUtc3BhY2U6IHByZS13cmFwO1wiPlRla25payBwZW55dWxpbmdhbiBkaSBtYW5hIG1vZGVsIGtlY2lsIGRpbGF0aWggc2VjYXJhIG1hdGVtYXRpcyB1bnR1ayBtZW5pcnUgZGlzdHJpYnVzaSBwcm9iYWJpbGl0YXMgYXRhdSBuaWxhaSBtZW50YWggc2ViZWx1bSBoYXNpbCBha2hpciBkaXB1dHVza2FuIG9sZWggbW9kZWwgYmVzYXI8L3NwYW4%2BPC9zcGFuPiIsImJhY2siOiI8c3BhbiBpZD1cImRvY3MtaW50ZXJuYWwtZ3VpZC1mMGM4NzYyNy03ZmZmLTllOWItYjliYi1kOGJhNjA5MjNkNDlcIj48c3BhbiBzdHlsZT1cImZvbnQtc2l6ZTogMTFwdDsgZm9udC1mYW1pbHk6IEFyaWFsLCBzYW5zLXNlcmlmOyBiYWNrZ3JvdW5kLWNvbG9yOiB0cmFuc3BhcmVudDsgZm9udC12YXJpYW50OiBub3JtYWw7IHZlcnRpY2FsLWFsaWduOiBiYXNlbGluZTsgd2hpdGUtc3BhY2U6IHByZS13cmFwO1wiPkxvZ2l0IEJhc2VkIERpc3RpbGxhdGlvbjwvc3Bhbj48L3NwYW4%2BIn0seyJpZCI6ImNhcmQtMTE5ODcuMTAwMDAwMDAxNDkiLCJmcm9udCI6IjxzcGFuIGlkPVwiZG9jcy1pbnRlcm5hbC1ndWlkLWQ3YmMxMmNjLTdmZmYtYzIwYS1lNTI0LTc3OTIwNTFkNDM4OVwiPjxzcGFuIHN0eWxlPVwiZm9udC1zaXplOiAxMXB0OyBmb250LWZhbWlseTogQXJpYWwsIHNhbnMtc2VyaWY7IGJhY2tncm91bmQtY29sb3I6IHRyYW5zcGFyZW50OyBmb250LXZhcmlhbnQ6IG5vcm1hbDsgdmVydGljYWwtYWxpZ246IGJhc2VsaW5lOyB3aGl0ZS1zcGFjZTogcHJlLXdyYXA7XCI%2BTWV0b2RlIHlhbmcgbWVtdW5na2lua2FuIG1vZGVsIGtlY2lsIG1lbXBlbGFqYXJpIGRhcmsga25vd2xlZGdlLCBzZXBlcnRpIG1lbWFoYW1pIGJhaHdhIG1lc2tpcHVuIGphdWggbGViaWggbWFzdWsgYWthbCBkYXJpcGFkYSBqYXdhYmFuIEM8L3NwYW4%2BPC9zcGFuPiIsImJhY2siOiI8c3BhbiBpZD1cImRvY3MtaW50ZXJuYWwtZ3VpZC1mZTAwZGNjNy03ZmZmLWVkNzUtY2IyZS1jN2ExZGMwMTRkNDhcIj48c3BhbiBzdHlsZT1cImZvbnQtc2l6ZTogMTFwdDsgZm9udC1mYW1pbHk6IEFyaWFsLCBzYW5zLXNlcmlmOyBiYWNrZ3JvdW5kLWNvbG9yOiB0cmFuc3BhcmVudDsgZm9udC12YXJpYW50OiBub3JtYWw7IHZlcnRpY2FsLWFsaWduOiBiYXNlbGluZTsgd2hpdGUtc3BhY2U6IHByZS13cmFwO1wiPkxvZ2l0IEJhc2VkIERpc3RpbGxhdGlvbjwvc3Bhbj48L3NwYW4%2BIn0seyJpZCI6ImNhcmQtMzIwMTIuNSIsImZyb250IjoiPHNwYW4gaWQ9XCJkb2NzLWludGVybmFsLWd1aWQtMTZkNTg0MGItN2ZmZi0wMzU1LTlkOTctNzUwZDUxYmQyZmRlXCI%2BPHNwYW4gc3R5bGU9XCJmb250LXNpemU6IDExcHQ7IGZvbnQtZmFtaWx5OiBBcmlhbCwgc2Fucy1zZXJpZjsgYmFja2dyb3VuZC1jb2xvcjogdHJhbnNwYXJlbnQ7IGZvbnQtdmFyaWFudDogbm9ybWFsOyB2ZXJ0aWNhbC1hbGlnbjogYmFzZWxpbmU7IHdoaXRlLXNwYWNlOiBwcmUtd3JhcDtcIj5QZW5kZWthdGFuIGRpIG1hbmEgbW9kZWwgYmVzYXIgKHNlcGVydGkgR1BULTQpIGRpZ3VuYWthbiBzZW1hdGEtbWF0YSBzZWJhZ2FpIG1lc2luIHBlbWJ1YXQgZGF0YXNldCB1bnR1ayBtZW5naGFzaWxrYW4gcmlidWFuIHBhc2FuZyB0YW55YS1qYXdhYiBhdGF1IGluc3RydWtzaSBiZXJrdWFsaXRhcyB0aW5nZ2k8L3NwYW4%2BPC9zcGFuPiIsImJhY2siOiI8c3BhbiBpZD1cImRvY3MtaW50ZXJuYWwtZ3VpZC1mMDNkMGY4MS03ZmZmLTFiYTQtMWRiMS00NzJlNTVmMTBlMDFcIj48c3BhbiBzdHlsZT1cImZvbnQtc2l6ZTogMTFwdDsgZm9udC1mYW1pbHk6IEFyaWFsLCBzYW5zLXNlcmlmOyBiYWNrZ3JvdW5kLWNvbG9yOiB0cmFuc3BhcmVudDsgZm9udC12YXJpYW50OiBub3JtYWw7IHZlcnRpY2FsLWFsaWduOiBiYXNlbGluZTsgd2hpdGUtc3BhY2U6IHByZS13cmFwO1wiPlN5bnRoZXRpYyBEYXRhIERpc3RpbGxhdGlvbjwvc3Bhbj48L3NwYW4%2BIn0seyJpZCI6ImNhcmQtNDMzNTIuNDAwMDAwMDAyMjM1IiwiZnJvbnQiOiI8c3BhbiBpZD1cImRvY3MtaW50ZXJuYWwtZ3VpZC00OWY1NTkwZi03ZmZmLTRlZDMtOWEyMi1lMjMxMWZlY2MwOGRcIj48c3BhbiBzdHlsZT1cImZvbnQtc2l6ZTogMTFwdDsgZm9udC1mYW1pbHk6IEFyaWFsLCBzYW5zLXNlcmlmOyBiYWNrZ3JvdW5kLWNvbG9yOiB0cmFuc3BhcmVudDsgZm9udC12YXJpYW50OiBub3JtYWw7IHZlcnRpY2FsLWFsaWduOiBiYXNlbGluZTsgd2hpdGUtc3BhY2U6IHByZS13cmFwO1wiPlBlbmRla2F0YW4gZWZpc2llbiB5YW5nIGRpZ3VuYWthbiBkYWxhbSBwZW1idWF0YW4gdmFyaWFuIG1vZGVsIGtlY2lsIERlZXBTZWVrIFIxIHNlcGVydGkgdmVyc2kgTGxhbWEsIGRpIG1hbmEgbWVyZWthIG1lbnRyYW5zZmVyIGtlbWFtcHVhbiByZWFzb25pbmcgc2VjYXJhIHNlZGVyaGFuYTwvc3Bhbj48L3NwYW4%2BIiwiYmFjayI6IjxzcGFuIGlkPVwiZG9jcy1pbnRlcm5hbC1ndWlkLTQ5MWY5MTU1LTdmZmYtNmQ2YS0yMjZmLWUyYWUzZThkMzcwMlwiPjxzcGFuIHN0eWxlPVwiZm9udC1zaXplOiAxMXB0OyBmb250LWZhbWlseTogQXJpYWwsIHNhbnMtc2VyaWY7IGJhY2tncm91bmQtY29sb3I6IHRyYW5zcGFyZW50OyBmb250LXZhcmlhbnQ6IG5vcm1hbDsgdmVydGljYWwtYWxpZ246IGJhc2VsaW5lOyB3aGl0ZS1zcGFjZTogcHJlLXdyYXA7XCI%2BU3ludGhldGljIERhdGEgRGlzdGlsbGF0aW9uPC9zcGFuPjwvc3Bhbj4ifV0sImluc3RydWN0aW9uIjoiS2xpayBrYXJ0dSB1bnR1ayBtZWxpaGF0IGphd2FiYW5ueWEuIn0%3D" height="522" width="100%" allow="fullscreen"></iframe>

***

## Pasca-SFT: Alignment dan Reasoning

Selamat datang lagi, *Coders*.

Seperti statement penutup bagian sebelumnya, sebuah model yang hanya melewati tahap pre-training dan Supervised Fine-Tuning biasanya masih memiliki perilaku yang kurang baik. Artinya, model pada fase tersebut sangat rentan menghasilkan respons yang bias, tidak sopan, atau berhalusinasi menyajikan informasi yang salah.

Selain itu, ketika dihadapkan pada instruksi matematika atau logika pemrograman yang kompleks, model SFT sering kali langsung “melompat” pada kesimpulan yang keliru tanpa menyusun langkah penyelesaian yang sistematis.

*Nah*, kondisi tersebutlah yang menuntun kita pada fase ini, yaitu “penyesuaian dan penalaran”.

Sesuai dengan dua goals baru tersebut, submodul ini juga akan membedah dua pilar pengembangan LLM modern.

*   **Konsep RLHF untuk Alignment** 
    Kita akan membedah cara para AI Engineer menyelaraskan output model agar aman, bermanfaat, dan sesuai dengan ekspektasi manusia. Kita akan membahas tahapan teknisnya secara *step-by-step*, mulai dari menggunakan model SFT, melatih Reward Model, hingga melakukan fine-tuning menggunakan algoritma seperti PPO.
*   **Konsep dan Pelatihan Reasoning** 
    Setelah model selaras dengan ekspektasi dasar, kita akan membahas persoalan logika tingkat tinggi. Kita akan menelaah secara teknis apakah LLM benar-benar melakukan proses berpikir, dari mana kemampuan ini muncul, serta mengeksplorasi teknik ajaib untuk mengajari LLM “berpikir”.

Alright, mari kita mulai langkah pertama di submodul ini dengan mengupas pilar yang pertama secara detail, yaitu Konsep RLHF untuk Alignment. *Are you ready for the one last ride for this module?* 

### RLHF untuk Alignment

Setelah sebuah model melewati tahap Supervised Fine-Tuning (SFT), ia memang sudah memiliki kemampuan teknis untuk mengikuti instruksi dasar dan merangkai struktur tata bahasa yang koheren. Namun, di titik komputasi ini, model tersebut belum memiliki mekanisme evaluasi internal untuk menyaring kualitas, etika, dan keamanan dari respons yang ia hasilkan.

Kelemahan utama dari model SFT adalah model tersebut secara ketat dilatih untuk memprediksi dan mereproduksi pola dari data pelatihan yang diberikan. Jika data tersebut mengandung bias, informasi yang salah, atau kalimat toksik, model SFT akan mereproduksinya secara akurat. Selain itu, SFT tidak memiliki kapasitas untuk membedakan mana respons yang sekadar benar secara gramatikal dan mana respons yang paling optimal dan aman ketika dihadapkan pada satu instruksi yang memiliki berbagai kemungkinan jawaban.

Di sinilah konsep Alignment atau penyelarasan menjadi sebuah kebutuhan primer bagi LLM. Dalam industri AI, standar keberhasilan Alignment ini distandarkan ke dalam tiga pilar metrik utama, yang sering disingkat sebagai prinsip HHH.

*   **Helpful:** Model harus memberikan informasi yang relevan, langsung pada sasaran, dan secara efisien memecahkan masalah pengguna.
*   **Honest**: Model harus menyajikan fakta yang akurat, menyertakan ketidakpastian jika informasinya terbatas, dan meminimalisir halusinasi data.
*   **Harmless:** Model harus memiliki mekanisme penolakan terhadap instruksi yang berpotensi melanggar hukum, menyebarkan kebencian, atau membahayakan secara fisik maupun mental.

Sekarang pertanyaannya, bagaimana cara memberitahu LLM untuk mematuhi tiga aturan triple H ini saat ia berbicara?

Jawabannya ada pada teknik yang memanfaatkan “Sang Pencipta” LLM itu sendiri, yaitu Reinforcement Learning from Human Feedback (RLHF).

Sebelumnya, LLM hanya dilatih untuk menebak kata berikutnya yang secara statistik paling tepat. Namun, RLHF menggeser paradigma tersebut menjadi optimasi berbasis penghargaan (reward-based optimization). Sehingga, alih-alih sekadar meminimalkan tingkat kesalahan melalui loss function standar, kita mulai melatih model menggunakan sistem reward berupa nilai skalar atau numerik.

Apa maksudnya dari “memberikan *reward*” ke model?

![dos-be813d76b51f1b977e77ef17dd69437220260504112510.jpeg](https://assets.cdn.dicoding.com/original/academy/dos-be813d76b51f1b977e77ef17dd69437220260504112510.jpeg)

Seperti yang ditunjukan ilustrasi di atas, konsep memberikan reward ini akan dilakukan oleh sistem terpisah yang kita sebut sebagai Reward Model. *Nah*, Reward Model inilah yang secara otomatis akan mengevaluasi output teks dengan sistem penilaian seperti berikut.

*   **Nilai Tinggi:** Sistem akan memberikan skor numerik tinggi untuk respons yang selaras dengan preferensi manusia.
*   **Nilai Rendah:** Sebaliknya, sistem akan memberikan skor rendah atau negatif jika respons tersebut melanggar prinsip HHH (Helpful, Honest, Harmless), misalnya ketika model memberikan informasi yang menyesatkan atau *toxic*.

Namun, muncul pertanyaan baru dari penjelasan di atas. Jika penilaian dilakukan secara otomatis oleh Reward Model, lantas di mana letak campur tangan manusianya?

Jawabannya ada pada fase sebelum proses komputasi utama dimulai. Manusia berperan sebagai penyedia data untuk melatih Reward Model tersebut agar mengenali standar preferensi kita. Jika dijabarkan secara singkat, prosesnya akan seperti berikut ini.

1.  **Inisiasi dari LLM** 
    LLM diberikan berbagai macam prompt dan diminta menghasilkan kumpulan variasi atau sampel jawaban.
2.  **Evaluasi Jawaban oleh Manusia** 
    Sekelompok evaluator manusia membaca variasi respons tersebut, lalu secara manual memberikan peringkat (ranking) dari yang paling bagus hingga yang paling buruk. Data peringkat inilah yang merepresentasikan Human Feedback.
3.  **Bangun Reward Model** 
    Data tersebut kemudian digunakan untuk melatih neural network yang akan berperan sebagai Reward Model nantinya. Setelah tahap training selesai, Reward Model akan mahir memprediksi peringkat mana yang akan dipilih oleh manusia.

Hasilnya? Kita berhasil menciptakan sebuah sistem canggih yang mampu mengevaluasi teks dengan kacamata dan standar manusia, tetapi dengan kecepatan mesin.

Sekarang muncul sebuah masalah, jika kita hanya menyuruh model “kejar skor tertinggi”, LLM sering kali menemukan cara curang atau *reward hacking*. Misalnya, model mungkin menyadari bahwa menggunakan kata *“Tentu saja!”* selalu mendapat skor tinggi, sehingga ia akan menjawab semua pertanyaan dengan *“Tentu saja!”* tanpa memberikan isi yang relevan, atau tata bahasanya yang mungkin menjadi berantakan.

Untuk mencegah hal tersebut dan memastikan model belajar dengan stabil, kita menggunakan algoritma Reinforcement Learning khusus yang bernama PPO (Proximal Policy Optimization) yang akan kita bahas di bagian terpisah.

Untuk sekarang, bagaimana jika kita berkenalan dengan *goals* lain dari post fine-tuning yang lebih *mind blown* dibandingkan Alignment? *Alright, let’s go!*

### Reasoning: Membuat Model Bisa Berpikir

Selamat datang di arena Reasoning, fase post-training di mana LLM mulai benar-benar menunjukkan kemampuan “intelektual” yang menyerupai manusia. Jika alignment membentuk etika model, maka reasoning adalah proses mengasah ketajaman otaknya.

*Nah*, untuk memulai petualangan di dalam otak LLM ini, mari kita mulai dengan pertanyaan yang paling fundamental.

Apakah LLM benar-benar dapat berpikir seperti manusia?

Jika otak sebuah LLM dibedah hingga ke neuron unit di dalamnya, kita tidak akan menemukan ego, perenungan, dan "Aha!" momen yang biasanya kita rasakan saat menemukan ide brilian.

*Yup*, seperti yang kita ketahui, LLM adalah mesin Next-Token Prediction. Pekerjaan seumur hidupnya hanya satu, yaitu menebak patahan kata apa yang probabilitasnya paling tinggi untuk muncul selanjutnya. Ia menerima triliunan teks dari internet, mengubahnya menjadi angka-angka di dalam ruang vektor berdimensi tinggi, dan melakukan operasi perkalian matriks. Itulah mengapa para kritikus AI sering menyebut LLM sekadar "*Stochastic Parrot*" atau beo statistik karena Ia hanya meniru pola tanpa benar-benar memahami maknanya.

*Eits*, tentu cerita ini tidak akan berakhir semembosankan itu karena para ilmuwan AI menemukan plot wist yang berasal dari sebuah pertanyaan besar, *“Bagaimana mungkin sebuah mesin yang hanya dilatih menebak kata selanjutnya bisa bermain catur atau menyelesaikan puzzle yang begitu kompleks?”*.

Ternyata, sekadar menghafal pola statistik tidaklah cukup untuk mencapai akurasi absolut dalam tugas-tugas yang sangat rumit seperti itu. Penelitian terbaru menunjukkan bahwa ketika model tumbuh sampai jumlah parameter tertentu, terjadi fenomena "Emergent Abilities".

Apakah Anda terasa familier?

*Yup*, ini adalah kemampuan yang sempat kita singgung di modul 1 sebelumnya. Studi yang dilakukan peneliti [MIT](https://news.mit.edu/2024/llms-develop-own-understanding-of-reality-as-language-abilities-improve-0814) menemukan bahwa saat LLM menyelesaikan tugas yang kompleks tersebut, ia tanpa sadar juga mengembangkan simulasi internal meskipun tidak pernah dilatih untuk itu. Sederhananya ini seperti LLM memiliki kemampuan untuk membayangkan sesuatu di kepalanya.

Jadi, apakah LLM benar-benar berpikir?

Jawabannya sangat bergantung pada definisi Anda terhadap kata "berpikir".

*   **Secara Biologis:** Tidak. LLM tidak berpikir layaknya manusia. Ia tidak memiliki memori episodik tentang masa lalu, tidak punya rasa takut akan kematian, dan tidak peduli apakah jawabannya benar atau salah secara moral.
*   **Secara Komputasional:** Iya. Ia melakukan bentuk "imajinasi" di kepalanya. Ia memanipulasi konsep abstrak, mengidentifikasi pola sebab-akibat, dan menyintesis informasi baru dari data yang belum pernah ia lihat sebelumnya.

Jika kita sudah sampai pada titik sepakat bahwa LLM mampu membayangkan realitas melalui simulasi internal, pertanyaan selanjutnya adalah “untuk apa imajinasi itu digunakan?”

Di sinilah Reasoning masuk ke dalam panggung.

#### Konsep Reasoning dalam LLM

Untuk mendefinisikan apa itu reasoning (penalaran) di dalam arsitektur dingin sang mesin, kita bisa meminjam konsep psikologi klasik dari Daniel Kahneman tentang dua mode kognitif: Sistem 1 (cepat, instingtif, tanpa usaha) dan Sistem 2 (lambat, analitis, butuh fokus keras).

Secara kodratnya, LLM adalah makhluk Sistem 1. Ketika Anda bertanya, "Apa ibu kota Prancis?", ia tidak perlu repot-repot masuk ke dalam world model-nya. Ia langsung memuntahkan token "Paris" dalam hitungan milidetik karena relasi statistik kedua kata itu sudah sangat menebal. Ini murni refleks retrieval (pengambilan informasi).

Namun, coba ubah permainannya dengan teka-teki logika berlapis. Jika LLM dibiarkan menggunakan Sistem 1-nya, ia akan panik menebak angka secara acak berdasarkan pola kalimat yang mirip di internet, dan berakhir dengan halusinasi.

Di sinilah Reasoning didefinisikan secara pragmatis: Ini adalah kapasitas komputasional model untuk mengerem insting menebak cepatnya (Sistem 1), dan beralih ke mode analitis (Sistem 2).

Secara mekanis, perpindahan ke Sistem 2 ini melibatkan tiga proses krusial di dalam "kepala" sang AI:

*   **Decomposition** 
    AI yang bernalar tidak akan menelan masalah utuh-utuh. Ia mendelegasikan dan memecah instruksi raksasa yang membingungkan menjadi sub-masalah kecil yang lebih riil dan bisa dikelola satu per satu oleh simulasi internalnya.
*   **Mental Sandbox** 
    Sebelum berani mengeluarkan token jawaban akhir, model melakukan simulasi *"apa yang terjadi jika..."* di dalam ruang gagasannya. Ia mengeksplorasi berbagai cabang probabilitas. Jika satu jalur pemikiran berujung pada tembok buntu atau logika yang salah, ia tidak akan memaksakan diri.
*   **Self-Correction** 
    Inilah puncak dari reasoning. Karena model memiliki "bayangan" atau world model tentang seperti apa hasil akhir yang koheren, ia mampu memverifikasi langkahnya sendiri di tengah jalan. Ia bisa berkata, *"Tunggu, langkah perhitungan di poin ke-3 tadi sepertinya keliru,"* lalu mundur dan memperbaiki alurnya. Mekanisme inilah yang secara drastis membunuh tingkat halusinasi.

Singkatnya, reasoning dalam LLM bukanlah adalah sebuah disiplin algoritmik yang memaksa model untuk menahan godaan memberikan jawaban instan. Seluruh LLM yang memiliki label “reasoning” di dalamnya pasti akan selalu dipaksa untuk mengikuti aturan tersebut.

Namun, meskipun prinsipnya sama, cara setiap model melakukannya bisa berbeda. *Nah*, pendekatannya dibagi menjadi dua jalan seperti berikut ini.

*   **Prompt-Based Reasoning (User-Driven)** 
    Saat mengetik "Coba pikirkan langkah demi langkah", Anda sebenarnya sedang memaksa LLM standar untuk melakukan Decomposition. Namun, model ini sering kali lemah di Mental Sandbox dan Self-Correction karena mereka cenderung terus maju tanpa melihat ke belakang.
*   **Native Reasoning Models (Model-Driven)** 
    Model seperti OpenAI o1 atau DeepSeek-R1 dilatih dengan Reinforcement Learning untuk selalu melakukan ketiga tahap ini secara otomatis di "latar belakang" ( hidden thought process ) sebelum memberikan jawaban satu kata pun kepada Anda. Mereka benar-benar melakukan uji coba jalur logika dan membuang jalur yang salah di dalam "kepala" mereka.

Di sini muncul pertanyaan, bagaimana cara menjinakkan insting model yang secara bawaan tidak memiliki reasoning bisa dipaksa untuk berpikir? Tentu saja jawabannya ada pada treatment fine-tuning. *Nah*, cara paling fundamental yang dapat kita lakukan adalah dengan teknik yang juga sudah akrab sebelumnya, yaitu *Chain-of-Thought (CoT) Fine-tuning*.

*Alright* CoT yang kita bahas di atas itukan fine-tuning pada umumnya sebenarnya kan? Nah, apakah Reinforcement Learning juga bisa digunakan untuk mewujudkan hal tersebut?

*Yup*, jawabannya sangat bisa, ini juga waktu yang sangat tepat untuk menunaikan hutang kita sebelumnya untuk mengeksplorasi step-by-step RL menggunakan pendekatan paling fundamentalnya, yaitu PPO.

### Step By Step Reinforcement Learning dengan PPO

Proximal Policy Optimization (PPO) adalah algoritma Reinforcement Learning berbasis Policy Gradient yang dirancang untuk melatih Agent agar dapat mengambil keputusan optimal dalam sebuah Environment melalui proses trial-and-error.

*Yup*, banyak sekali istilah asing yang kita lihat pada penjelasan di atas. Mari kita cerna satu per satu.

Pertama soal Agent, banyak yang mungkin akan salah kaprah dan tertukar antara Agent dalam RL PPO ini dengan yang ada di sistem Agentic AI. Agent yang dimaksud di sini adalah sebuah entitas yang melakukan proses belajar melalui interaksi. Dalam konteks ini, LLM-lah yang berlaku sebagai agent.

Lalu, apa itu Environment yang dimaksud di sini?

Dalam RL, Environment dapat dikatakan sebagai “dunia” tempat Agent tadi beroperasi atau belajar. *Nah*, mari kita lihat contohnya dalam dua dunia yang berbeda ini.

*   **Dalam Game:** Environment merupakan mesin game itu sendiri seperti catur atau Super Mario yang memberi tahu kita posisi gambar atau mungkin skor .
*   **Dalam LLM:** Environment-nya bersifat simulasi atau abstrak. Sebab "Dunia" bagi si AI adalah kumpulan instruksi atau prompts yang kita sebagai manusia berikan.

Lalu, mengapa PPO juga ada di game?

*Yup*, sebelum digunakan untuk menyelaraskan LLM, PPO yang diperkenalkan oleh OpenAI pada tahun 2017 awalnya dikembangkan untuk domain Robotika dan Video Game. *Fun fact* yang berasal dari dokumentasi [OpenAI](https://openai.com/index/openai-five-defeats-dota-2-world-champions/) di bawah ini merupakan salah satu bukti nyata keberhasilan PPO dalam domain Game.

![dos-1056c7ca03da24ce5d501b6c30edfe0f20260311172848.png](https://assets.cdn.dicoding.com/original/academy/dos-1056c7ca03da24ce5d501b6c30edfe0f20260311172848.png)

OpenAI Five yang merupakan tim berisi bot AI berhasil mengalahkan juara dunia dalam game strategi kompleks Dota 2. *Eits*, OpenAI Five ini bukanlah LLM, melainkan AI yang fokus pada strategi dan aksi yang menggunakan arsitektur LSTM sebagai backbone LLM-nya.

*Nah*, sementara dalam domain LLM, OpenAI mengombinasikan SFT ditambah RLHF dengan PPO untuk menciptakan InstructGPT dan ChatGPT versi awal.

Kehadiran PPO memastikan bahwa saat LLM belajar dari feedback manusia, ia tidak berubah menjadi model yang aneh atau kehilangan kemampuan berbahasa dasarnya. Selain untuk alignment, PPO tentu saja dapat digunakan untuk membuat model memiliki kemampuan reasoning. Dalam konteks reasoning, PPO tidak akan lagi menilai jawaban, melainkan langkah-langkah.

Jadi, menjelajahi PPO akan menjadi *“stepping stones”* yang sangat fundamental. Secara flow, RL dengan PPO akan terlihat seperti pada ilustrasi di bawah ini.

![dos-6bcbcbd297d3ad2bf8d7c2a0cc01e01420260430093537.png](https://assets.cdn.dicoding.com/original/academy/dos-6bcbcbd297d3ad2bf8d7c2a0cc01e01420260430093537.png)

*Alright*, mari membedah cara PPO melatih LLM kita langkah demi langkah.

#### Langkah 1: Inisialisasi Arsitektur PPO

Sebelum proses tanya-jawab dan pemberian skor dimulai, algoritma PPO harus menyiapkan 'panggung' pelatihannya. Di fase ini, kita tidak bisa hanya membawa satu model LLM saja. PPO menggunakan arsitektur yang sangat terstruktur dan melibatkan empat komponen utama yang memiliki peran masing-masing.

*   **The Actor (Policy Model)** 
    *Nah*, ini dia “bintang utamanya”, yang merupakan LLM hasil SFT atau post fine-tuning lainnya. Di dalam istilah Reinforcement Learning, model ini bertugas menentukan “kebijakan” (policy), yaitu memilih token apa yang akan dikeluarkan selanjutnya. Nilai Weights model inilah yang akan terus diperbarui selama proses latihan.
*   **The Reference (Reference Model)** 
    Begitu latihan dimulai, sistem akan membuat satu salinan atau cloning dari model SFT yang menjadi “Actor”. Inilah yang kita sebut sebagai Reference Model. Satu hal yang membedakannya dengan “Si Actor” adalah nilai Weight dari model ini akan di-frozen sehingga tidak akan berubah. Perannya sebagai "jangkar bahasa" yang akan memberikan penalti saat Actor nanti mulai berbicara dengan pola aneh karena terlalu fokus mencari cara untuk mendapatkan skor tinggi dari reward model.
*   **The Judge (Reward Model)** 
    Ini adalah sistem juri yang sudah kita latih susah payah menggunakan data Human Feedback di tahap sebelumnya. Ia sudah memahami apa yang manusia anggap sebagai respons yang Helpful, Honest, dan Harmless. *Nah*, nilai Weight model ini juga dikunci (frozen). Tugasnya murni hanya satu, yaitu membaca teks akhir yang dihasilkan oleh Actor, lalu memberikan skor berupa angka mutlak, misalnya 5.0 untuk respons yang baik, atau -3.0 untuk respons berbahaya.
*   **The Critic (Value Model)** 
    Ini adalah senjata rahasia dari algoritma PPO yang menggunakan arsitektur Actor-Critic. Saat LLM atau Actor sedang melakukan Autoregressive, Critic bertugas memprediksi *"Dengan kata-kata yang sudah dihasilkan sejauh ini, kira-kira berapa skor akhir yang akan diberikan oleh Juri nanti?"*. Sehingga “Si Actor” tidak perlu menunggu hingga satu kalimat utuh selesai untuk tahu respon yang dihasilkannya itu arahnya sudah benar atau salah. Ia mendapat feedback seketika. Status: Sama seperti Actor, bobot model Critic ini juga diperbarui selama proses agar prediksinya semakin akurat seiring berjalannya waktu.

*Yup* seperti yang sudah kita lihat, sebelum satu kata pun dihasilkan dalam tahap RLHF, komputer kita harus menyala dan memuat setidaknya arsitektur gabungan dari keempat komponen ini.

#### Langkah 2: Proses "Rollout"

Setelah keempat model bersiap di posisinya, tahap pengumpulan data pun dimulai. Di dunia Reinforcement Learning, fase ini dikenal dengan istilah Rollout karena seperti merekam jalannya sebuah pertunjukan dari awal hingga akhir sebelum dievaluasi.

Pada dasarnya, langkah ini masih mirip dengan Reinforcement Learning. Semuanya dimulai ketika sistem memberikan sebuah instruksi atau prompt ke Actor. *Nah*, Aktor akan mulai berimprovisasi untuk merangkai token demi token sesuai dengan prinsip Autoregressive. Teks yang dihasilkan inilah yang disebut sebagai Action.

Di saat yang persis bersamaan ketika The Actor sedang merangkai kata, The Judge dan The Critic juga ikut bekerja secara diam-diam di latar belakang.

*   **Pekerjaan The Judge** 
    The Reference membaca prompt yang sama dan mengamati kata-kata yang sedang dikeluarkan oleh The Actor. Di setiap kata, The Reference akan menghitung perbedaan perhitungan berupa probabilitas antara *"apa yang diucapkan The Actor"* dengan *"apa yang seharusnya diucapkan"* oleh The Reference sebagai “jati diri” awal The Actor. Catatan perbedaan ini nantinya akan digunakan untuk menghitung penalti yang kita singgung sebelumnya melalui KL Divergence jika The Actor mulai menggunakan gaya bahasa yang tidak wajar.
*   **Pekerjaan The Critic** 
    Sementara itu, The Critic (Value Model) juga tidak tinggal diam. Saat kalimat sedang disusun, ia mencoba menerka masa depan. Berdasarkan kata-kata yang sudah keluar, The Critic menebak-nebak, "Hmm, respons seperti ini sepertinya berpotensi mendapatkan skor akhir +4.0 dari Juri nanti." Tebakan ini (nilai *v* pada diagram) terus dicatat dan diperbarui.

Begitu The Actor meletakkan tanda titik di akhir kalimat, proses Rollout untuk satu prompt dinyatakan selesai. Sistem kini memegang sebuah “paket” yang berisi beberapa hal berikut.

*   Prompt awal yang diberikan.
*   Teks respons utuh yang dihasilkan The Actor.
*   Catatan probabilitas bahasa dari The Reference.
*   Prediksi skor sementara dari The Critic.

*Nah*, ratusan hingga ribuan paket seperti ini akan dikumpulkan secara bersamaan dan diserahkan ke The Judge atau Reward Model untuk mendapatkan “vonis”.

#### Langkah 3: Evaluasi dan Kalkulasi "Advantage"

Setelah The Actor selesai memberikan jawabannya, ia akan dihadapkan pada perhitungan berlapis lapis untuk menentukan kualitas jawabannya. Mari kita cari tahu seberlapis apa perhitungan yang perlu dilakukan dengan memperhatikan ilustrasi di bawah ini.

![dos-1546da1734ae9e90a2e3e5167959dbc320260430093642.png](https://assets.cdn.dicoding.com/original/academy/dos-1546da1734ae9e90a2e3e5167959dbc320260430093642.png)

*   **Vonis dari The Judge (Reward Model)** 
    Reward Model akan menjalankan tugasnya untuk mengevaluasi seberapa Helpful, Honest, dan Harmless teks yang dihasilkan berdasarkan standar preferensi manusia yang sudah ia pelajari. Ia kemudian mengeluarkan skor mutlak misalnya +5.0 yang disimbolkan dengan huruf r. *Eits*, sebelum skor +5.0 itu diserahkan kepada The Actor sebagai hadiah, sistem PPO akan mengecek laporan dari The Reference pada perhitungan selanjutnya ini
*   **Perhitungan Penalti dari The Reference** 
    Jika laporan The Reference menunjukkan nilai KL Divergence-nya tinggi atau dengan kata lain gaya bahasa The Actor menyimpang terlalu jauh dari bahasa manusia yang natural, maka skor dari Reward Model sebelumnya akan dikurangi. Misalnya dalam contoh ini skor reward dikurangi menjadi +4.5. Inilah maksud dari rumus yang disematkan setelah penggabungan hasil dari reference dan reward sebelum masuk ke tahap GAE pada ilustrasi di atas.
*   **Kalkulasi Advantage** 
    Skor akhir yang sudah dipotong tadi tetap tidak langsung ditelan mentah-mentah oleh The Actor. *Yup*, Jika Anda perhatikan ilustrasi flow di atas, seluruh nilai akhir dikumpulkan dalam satu tahap akhir bernama GAE (Generalized Advantage Estimation). Kira-kira prosesnya akan seperti berikut ini.
    *   Sistem akan memanggil The Critic untuk bertanya *"Di awal tadi, saat The Actor baru mulai berbicara, kamu memprediksi dia akan dapat skor berapa?"*. Misalkan
    *   The Critic menjawab *"Berdasarkan penilaianku, kalimat pembukanya biasa saja, jadi aku tebak dia cuma akan dapat skor +2.0."*.

Lalu, sistem membandingkan antara nilai Reward saat ini (+4.5) dengan nilai dari The Critic (+2.0). Selisih positif inilah (+2.5) yang disebut sebagai Advantage yang dilambangkan *At*. Sebaliknya, jika realitanya lebih buruk dari ekspektasi, Advantage-nya akan bernilai negatif.

Sekarang pertanyaannya adalah mengapa kita butuh Advantage dan bukan skor mentah? Bayangkan anak jenius yang biasa dapat nilai 90, lalu di ujian ia dapat 85. Bagi dia, itu nilai yang "buruk" (Advantage negatif) dan ia harus mengevaluasi cara belajarnya. Namun, bagi anak yang biasa dapat nilai 60, skor 80 adalah prestasi luar biasa (Advantage positif). Nah, dalam LLM ilustrasi-nya secara *end-to-end* akan terlihat seperti berikut ini.

![dos-fc674ce46c1a34309ef3a78b992477fb20260504113855.gif](https://assets.cdn.dicoding.com/original/academy/dos-fc674ce46c1a34309ef3a78b992477fb20260504113855.gif)

*Yup*, bagi model yang saat masih dalam proses Autoregressive memiliki nilai tinggi tetapi menghasilkan nilai reward rendah, sistem akan mengirimkan nilai numerik ke The Actor yang bermakna *"Tenang, langkahmu tadi sebenarnya sudah bagus, hasil salah tadi cuma anomali"*. Sehingga, parameter di dalam model tidak akan berubah terlalu signifikan dan mengantisipasi model berpikir bahwa jawabannya salah total.

Sebaliknya, jika nilai reward lebih tinggi dibanding nilai sementaranya pada saat dinilai The Critic, sistem akan memintanya untuk *"Lakukanlah pola ini lebih sering"*.

#### Langkah 4: Clipping dan Policy Update

Saat sampai di tahap ini, sistem PPO bersiap menyuntikkan angka Advantage (*At*) untuk memperbarui Weight atau parameter dalam The Actor. Proses ini terlihat pada kotak biru di ilustrasi PPO di awal yang dapat kita lihat di bawah ini.

![dos-f0070d6ce7b3f6f7bd3d97b23684d90820260430094036.png](https://assets.cdn.dicoding.com/original/academy/dos-f0070d6ce7b3f6f7bd3d97b23684d90820260430094036.png)

*Eits*, tunggu dulu, sekarang bagaimana jika nilai Advantage yang dihasilkan sangat besar misalnya +31.0? LLM mungkin akan mengubah cara berpikirnya 180 derajat hanya gara-gara satu feedback bagus tersebut dan berakhir mengalami Catastrophic Forgetting.

*Nah*, PPO sudah menyiapkan pencegahannya menggunakan mekanisme yang disebut Clipping.

Konsep Clipping ini diterapkan PPO dengan menetapkan sebuah batas aman, biasanya sekitar 20%. Artinya, probabilitas sebuah kata hanya boleh dinaikkan maksimal menjadi 1.2 kali lipat, atau diturunkan maksimal menjadi 0.8 kali lipat dalam satu langkah evaluasi.

*   **Advantage Sangat Positif** 
    Jika The Actor mendapat Advantage yang sangat besar seperti pertanyaan di awal, ia akan cenderung menaikkan probabilitas kata "belajar" hingga 300% dalam sekejap. *Nah*, sebelum itu terjadi, PPO akan berkata: *"Tahan! Bagus kamu mau belajar, tapi maksimal naik 20% saja ya hari ini."*.
*   **Advantage Sangat Negatif** 
    Sebaliknya, jika Advantage-nya sangat negatif dan model panik ingin menghapus sebuah kata sepenuhnya, PPO akan memotongnya di batas bawah, yaitu 0.8.

Setelah instruksi perubahan dipastikan aman dan tidak melebihi batas "Clipping", barulah sistem mengizinkan pembaruan bobot terjadi pada neural network The Actor.

Sampai di sini, kita dapat mengatakan bahwa satu siklus pembaruan untuk satu batch kata telah berakhir. Sistem PPO akan kembali ke Langkah 2, memberikan prompt baru, membiarkan The Actor berimprovisasi, dievaluasi oleh The Judge dan The Critic, lalu diperbarui lagi secara perlahan. Loop ini diulang ribuan kali hingga The Actor benar-benar fasih menghasilkan respons yang selaras dengan prinsip *Helpful*, *Honest*, dan *Harmless*.

*Nah*, dari sini Anda sudah lengkap menyelesaikan petualangan di sistem yang cukup kompleks bernama PPO. Namun, tahukah Anda bahwa ada pendekatan RL yang memiliki ability yang sama dengan PPO tetapi dengan sistem yang lebih efisien.

*Yup*, mari kita sambut Reinforcement Learning dengan GRPO.

### Reinforcement Learning dengan GRPO

Kemegahan empat model yang saling mengawasi dan teknik yang melahirkan ChatGPT dari PPO sebelumnya memang bukanlah isapan jempol belaka.

Namun, saat sudah berbicara realita di ruang server, menjalankan empat LLM secara bersamaan bukanlah kabar baik untuk memori VRAM yang kita miliki. Itu pun jika kita punya server dengan memori VRAM yang *unlimited*. Lalu, “apa kabar” dengan perangkat pribadi yang kita miliki? Hal tersebut ibarat memasukkan empat ekor gajah Sumatra ke dalam satu kamar kos.

*Eits*, itu baru berbicara soal model. Saat berbicara soal reasoning yang kita pelajari sebelumnya, model memerlukan ruang ekstra pada memori untuk menulis langkah-langkah pemikirannya. *Yup*, tentu saja sudah tidak ada ruang ekstra lagi di “kamar kos” tersebut. Oleh karena itu, perkenalkan GRPO (Group Relative Policy Optimization) yang namanya meroket saat digunakan untuk melatih DeepSeek-R1.

![dos-21130971ab0df1468144be0d5319a3bd20260430094113.png](https://assets.cdn.dicoding.com/original/academy/dos-21130971ab0df1468144be0d5319a3bd20260430094113.png)

Lahirnya DeepSeek-R1 membuktikan bahwa model dengan reasoning tingkat tinggi yang setara dengan model proprietary berbayar dapat dibangun dengan biaya pelatihan yang jauh lebih murah dan efisiensi memori yang luar biasa.

Supaya Anda lebih yakin, perhatikan grafik perbandingan performa beberapa model yang berasal dari paper [DeepSeek](https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf) di bawah ini.

![dos-50d7dce392badd2b7c8dc9ef9bb577bb20260430094142.png](https://assets.cdn.dicoding.com/original/academy/dos-50d7dce392badd2b7c8dc9ef9bb577bb20260430094142.png)

Jika berpatokan dengan hasil di atas, kita mungkin akan membayangkan biaya yang dikeluarkan untuk membangun model OpenAI o1 dengan DeepSeek-R1 berada di angka yang sangat berdekatan. Sayangnya fakta berkata lain.

*   **DeepSeek-R1:** Menghabiskan sekitar USD 5,58 juta.
*   **OpenAI o1:** Meskipun angka pastinya rahasia, estimasi anggarannya mencapai USD 500 juta.

Ini menunjukan bahwa DeepSeek-R1 89 kali lebih murah dibandingkan OpenAI o1. *Nah*, mukjizat ini tentu tidak lepas dari peran GRPO yang membuat proses pelatihan menjadi jauh lebih efisien.

*Nah*, sekarang kita menuju pertanyaan paling fundamental sekaligus yang paling membuat penasaran, bagaimana semua efisiensi itu mungkin terjadi? Jawaban singkatnya karena GRPO membuang The Critic atau Value model dan mengganti sistem penilaiannya dengan perhitungan statistik yang jauh lebih ringan. Hal ini selaras dengan ilustrasi perbandingan PPO dan GRPO di bawah ini.

![dos-7a96e10c81b7b5a7f989e2717346d66520260430094255.png](https://assets.cdn.dicoding.com/original/academy/dos-7a96e10c81b7b5a7f989e2717346d66520260430094255.png)

Mari kita bedah mekanika yang brilian ini.

#### Langkah 1: The Multiverse of Reasoning (Generasi Output Paralel)

Coba perhatikan garis awal pada diagram PPO dan bandingkan dengan yang GRPO.

Pada PPO, sebuah prompt atau query masuk ke Policy Model atau The Actor yang kemudian mengeluarkan satu jawaban akhir. Di sisi lain, GRPO menggunakan *gameplay* yang jauh berbeda dengan memanfaatkan konsep yang seperti “arena multiverse”.

Ketika query masuk, GRPO tidak mengizinkan Policy Model untuk langsung menghasilkan satu jawaban akhir. Ia akan membuat The Actor untuk menciptakan beberapa variasi jawaban sekaligus dengan status draf. Artinya seluruh jawaban tersebut bukanlah jawaban final.

Oleh karena itu, jika Anda perhatikan pada diagram di atas, terdapat kotak-kotak berjejer *o1*, *o2*, hingga *oG* yang melambangkan output-nya itu ada lebih dari satu. Huruf "G" disana melambangkan ukuran grup atau jumlah output yang ingin dihasilkan pada GRPO. Jika dipikirkan kembali, konsep ini seperti memanifestasikan Mental Sandbox yang kita bahas di bab reasoning sebelumnya karena memaksa model untuk menghasilkan berbagai macam kemungkinan jawaban.

Namun, apa sebenarnya tujuan asli dari menciptakan jawaban yang banyak ini?

*Alright*, untuk menjawabnya mari bayangkan model sedang dihadapkan pada soal kalkulus rumit dan perlu mengeluarkan beberapa kemungkinan jawaban seperti pada ilustrasi di bawah ini. Sebagai tambahan informasi, di sini kita ingin membuat model memiliki kemampuan reasoning.

![dos-3d5712a5f5c95006810bcdaac3778df020260430094401.gif](https://assets.cdn.dicoding.com/original/academy/dos-3d5712a5f5c95006810bcdaac3778df020260430094401.gif)

Mari kita bandingkan perbedaan dari ketiga jawaban tersebut.

*   Pada jawaban pertama, model langsung menebak nilainya tanpa menghitung.
*   Pada jawaban kedua, AI ini langsung menggunakan rumus integral standar tanpa banyak penjelasan.
*   Pada jawaban ketiga, model menjelaskan konsepnya dulu, membayangkan grafik yang terbentuk, baru menghitung.

Daripada mengajarkan AI untuk mencari "satu cara yang paling benar" sejak awal, GRPO memberikan kebebasan eksplorasi dengan mengizinkan model untuk "mencoret-coret" berbagai strategi penyelesaian masalah. Langkah pertama ini memastikan bahwa AI tidak akan pernah kekurangan ide kreatif dalam memecahkan masalah berlapis.

Namun, tentu saja, dari sekian banyak variasi jawaban *o1* hingga *oG* tersebut, pasti ada yang “brilian” dan ada yang “ampas”. Bagaimana cara mengetahuinya? *Yup*, kita mungkin sudah bisa menebak entitas model yang akan bertanggung jawab untuk melakukannya. Mari kita buktikan di langkah berikutnya.

#### Langkah 2: Vonis Paralel Sang Hakim (Raw Reward Scoring)

Setelah Policy Model selesai memproduksi seluruh variasi jawaban, saatnya giliran dua entitas yang namanya sudah tidak asing bagi kita bekerja, yaitu Reference Model dan Reward Model.

Tugas keduanya masih sama persis dengan yang mereka kerjakan di PPO sebelumnya.

*   **Reward Model Tetap Menjadi Hakim Absolut:** Ia tetap membaca teks *output*, memprosesnya, dan mengeluarkan skor mentah (*r*).
*   **Reference Model:** Tetap Menghitung Penalti (KL Divergence) guna memastikan *Policy Model* tidak “menyimpang” hanya demi mendapatkan skor *r* tinggi dari Hakim.

Lantas apakah ada yang berbeda?

*Yup*, tentu saja ada. Perbedaan paling mendasar ada pada load kerja dari keduanya. Sebagai contoh, dalam PPO load dari Reward Model cukup santai karena hanya menilai satu jawaban. *Nah*, sementara dalam GRPO, ia harus me-review seluruh multiverse jawaban mulai dari *o1* hingga *oG* sekaligus dan juga mengeluarkan rentetan skor mentah *r1* hingga *rG*.

Jika menggunakan skenario yang ada pada ilustrasi di atas, mungkin Reward Model akan memberikan hasil seperti berikut.

*   *o1* dapat skor paling tinggi karena selain benar, langkah-langkahnya sangat detail dan logis yang mana ini yang kita inginkan dari model reasoning.
*   *o2* dapat skor lumayan tinggi karena jawabannya benar.
*   *o3* dapat skor negatif atau rendah karena logikanya salah dengan menganggap kurva bisa dihitung seperti segitiga dan hasilnya salah.

*Nah*, dari sini muncul pertanyaan, jika kita sudah meniadakan The Critic atau Value Model, lantas siapa yang memberi tahu sistem bahwa skor tinggi yang dihasilkan dari jawaban *o1* pantas diberikan bobot yang tinggi atau tidak?

Jawabannya ada di langkah berikutnya sekaligus yang terakhir.

#### Langkah 3: Kalkulasi Grup (The Hunger Games of Reasoning)

Dalam PPO sebelumnya kita menggunakan LLM yang setara dengan Policy Model untuk memberikan estimasi nilai dari setiap langkahnya. Nilai tersebut nantinya digunakan untuk menentukan berapa nilai ideal (Advantage) yang “dihadiahkan” ke model.

Alih-alih mencoba menebak standar nilai yang ideal, GRPO menggunakan pendekatan yang sederhana tetapi cerdik. Ia membandingkan setiap variasi jawaban dari Policy Model tadi dari mulai *o1* hingga *oG* dengan variasi jawabannya sendiri pada pertanyaan yang sama.

Cara apa yang dilakukannya untuk membandingkannya?

GRPO menggunakan perhitungan statistik sederhana untuk melakukan perbandingan dan menentukan nilai Advantage untuk setiap variasi jawaban.

![dos-b3df5fb97a38df5e48b40b07cd9b956a20260430094314.png](https://assets.cdn.dicoding.com/original/academy/dos-b3df5fb97a38df5e48b40b07cd9b956a20260430094314.png)

*Yup*, dengan rumus statistik gabungan antara rata-rata dengan standar deviasi ini, GRPO dapat mengubah skor mentah menjadi nilai Advantage tanpa bantuan The Critic. Formula di atas pada dasarnya menghasilkan aturan main yang sangat sederhana, tetapi efektif.

Sebab, jika skor variasi jawaban *oi* di atas rata-rata skor grup, maka nilai *Ai* atau Advantage-nya positif. Sebaliknya, jika di bawah rata-rata tentu nilai *Ai* menjadi negatif.

Namun, dimana letak hebatnya perhitungan ini sehingga bisa digadang-gadang menggantikan peran The Critic? Bukankah perhitungan sederhana ini tidak akan seoptimal jika kita menggunakan model?

*Eits*, tentu saja tidak. Dengan metode ini, GRPO tidak lagi peduli apakah sebuah soal matematika itu selevel anak SD atau selevel ujian masuk MIT.

*   Jika soalnya super gampang dan semua variasi jawaban AI mendapat skor 90, 92, dan 95, maka skor 90 akan dianggap buruk (karena di bawah rata-rata grup) dan diberi Advantage negatif.
*   Jika soalnya mustahil dijawab dan skor AI hancur semua di angka 10, 15, dan 20, skor 20 akan dianggap brilian (karena memimpin di grupnya) dan diberi Advantage positif.

Dengan begini ia bisa menilai secara real-time dan context-aware tanpa membutuhkan Value Model (The Critic) raksasa yang memakan jutaan dolar untuk hosting di memori server.

Langkah terakhir dari GRPO masih sama persis dengan PPO sebelumnya. Setelah mendapatkan seluruh nilai Advantage dari *A1* hingga *AG*, ia akan melakukan clipping sebelum meng-inject-nya ke dalam parameter Policy Model.

Teknik GRPO ini sekaligus menjadi makanan penutup dari petualangan kita dalam dunia reinforcement learning untuk alignment model. Namun, seperti sebelum-sebelumnya, tidak lengkap rasanya jika kita tidak melakukan sedikit aktivitas untuk memperkuat pemahaman mengenai dua teknik reinforcement learning di atas.
