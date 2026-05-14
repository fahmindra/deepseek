---
slug: ch-06-05
title: "AI Agent dan Agentic AI"
layout: chapter
---
# AI Agent dan Agentic AI

## Pengantar AI Agent dan Agentic AI

Hello, Developers! Sudah puas berkutat dengan Large Language Model (LLM)? Kalau iya, mari kita sedikit bergeser ke pembahasan yang lebih matang.

Anda pasti sudah sangat akrab dengan LLM, paham betul dengan cara kerja *token*, dan sudah menerapkan RAG (*Retrieval-Augmented Generation*) di aplikasi untuk mendapatkan konteks yang lebih presisi. Itu adalah fondasi teknis yang telah kita pelajari di modul-modul sebelumnya.

Namun, sehebat apa pun aplikasi berbasis RAG atau *chatbot* yang Anda bangun saat ini, sering kali rasanya masih ada satu kepingan yang hilang yaitu….

Jika Anda merasa sistem tersebut bersifat pasif karena hanya menunggu, menerima input, memproses teks, lalu mengembalikannya kembali, selamat! Anda berada di jalan pemahaman yang tepat.

Jika kita tilik sedikit ke belakang, saat ini kita masih menjadi pihak yang harus melakukan pekerjaan sepele dan repetitif, seperti *copy-paste* kode atau mengeksekusi secara manual saran yang diberikan AI.

Di sinilah kita memasuki pergeseran paradigma terbesar dalam pengembangan AI saat ini, yaitu era **Agentic AI**. Tren industri pun sedang bergerak drastis ke arah ini.

Laporan dari Capgemini Research Institute bertajuk [Rise of Agentic AI](https://www.capgemini.com/insights/research-library/ai-agents/) mengungkapkan fakta menarik:

*   Meskipun baru sekitar 2% organisasi yang sudah menerapkan AI Agent dalam skala penuh, sebanyak 82% perusahaan berencana untuk mengintegrasikannya dalam kurun waktu 1 hingga 3 tahun ke depan.
*   Diperkirakan dalam 12 bulan ke depan, 15% proses bisnis akan mencapai tingkat otonomi sebagian atau penuh berkat bantuan agent.
*   Capgemini memproyeksikan peluang nilai ekonomi hingga $450 miliar pada tahun 2028 melalui pertumbuhan pendapatan dan penghematan biaya yang dihasilkan oleh sistem agentic.
*   Masa depan bukan tentang AI menggantikan manusia, melainkan kolaborasi. Diprediksi pada 2028, 38% organisasi akan memiliki AI Agent sebagai rekan kerja di dalam tim mereka.

Gartner juga memperkuat hal ini dalam laporan [Top Strategic Technology Trends 2025](https://www.gartner.com/en/newsroom/press-releases/2024-10-21-gartner-identifies-the-top-10-strategic-technology-trends-for-2025), yang menempatkan Agentic AI sebagai tren utama dengan prediksi bahwa pada 2028, setidaknya 33% aplikasi software perusahaan akan memiliki kemampuan agentic.

Kalau Anda tahu, Andrew Ng, *Founder of DeepLearning.AI* mulai menyuarakan bahwa peningkatan performa AI ke depan tidak akan semata-mata datang dari model yang lebih besar (*More Parameters*), melainkan dari **Agentic Workflows**. Ini adalah fase di mana kita berhenti memperlakukan LLM sekadar sebagai mesin untuk menjawab soal, dan mulai merekayasanya menjadi mesin penyelesaian masalah yang otonom.

Perbedaan mendasar antara Generative AI yang biasa kita gunakan dengan konsep AI Agent ini terletak pada **"Agency"** atau kemampuan bertindak.

Generative AI murni beroperasi layaknya sebuah otak jenius yang terisolasi di dalam toples (*Brain in a Jar*). Ia memiliki pengetahuan luas dan logika yang tajam, tetapi ia tidak memiliki akses ke dunia luar.

Generative AI konvensional bekerja secara linear. Apa maksudnya? Ketika Anda memberikan instruksi, ia memberikan respons, dan proses berhenti di sana, terlepas dari apakah solusi tersebut benar-benar bisa dijalankan atau tidak.

![Generative AI Linear Flow](https://assets.cdn.dicoding.com/original/academy/dos-92295d0c9b76cfb7e9f08e9adacbc92920260429132332.png)

Sebaliknya, AI Agent bekerja secara iteratif. Meskipun tetap membutuhkan trigger atau instruksi awal dari kita, ia tidak langsung menyerah pada jawaban pertama. Dengan kemampuan *reasoning loop*, agent mampu mengeksekusi kode, mengobservasi hasilnya, dan jika menemukan kegagalan atau *error*, ia akan mengoreksi tindakannya sendiri hingga tujuannya tercapai.

![AI Agent Iterative Flow](https://assets.cdn.dicoding.com/original/academy/dos-cb057b7b1c44a7fbf3da0c6daa185e7320260429132346.png)

Fokus utamanya bergeser dari sekadar kreasi konten menjadi eksekusi tugas. AI Agent tidak hanya berhenti pada memberikan saran teknis, tetapi ia memiliki inisiatif untuk menggunakan alat bantu, seperti mengakses *file system*, memanggil API eksternal, atau menjelajah internet, untuk menyelesaikan permintaan Anda secara tuntas.

Sebagai developer, ini berarti peran kita pun ikut berevolusi. Kita tidak lagi sekadar menyambungkan API OpenAI atau Gemini ke antarmuka pengguna. Kita akan merancang sistem di mana AI mampu melakukan proses *looping* mandiri, seperti berpikir (*reasoning*), bertindak (*acting*), dan mengoreksi diri (*self-correction*) ketika menghadapi *error*.

Kini, batas antara *software engineering* konvensional dan *AI engineering* menjadi semakin tipis, serta jembatan penghubungnya adalah kemampuan kita dalam membangun sistem yang *agentic*.

Kalau begitu, mari kita mulai mendalami dunia agent ini!

---

## Kemunculan Konsep Agent

Dari awal belajar Generative AI, kita semua sudah terbiasa berinteraksi dengan model bahasa besar atau Large Language Model (LLM) yang sifatnya reaktif. Pengguna memberikan perintah, kemudian model LLM akan merespons. Pola ini terus berulang dan membentuk paradigma tanya-jawab yang sederhana, tapi cukup efektif juga untuk berbagai kebutuhan seperti pembuatan teks, analisis data, hingga pembuatan creative content.

Namun, sebagai pengguna, seringnya kita terus-menerus melakukan request yang lebih kompleks kepada model LLM. Seiring dengan meningkatnya kompleksitas permasalahan yang ingin diselesaikan, pendekatan ini mulai menunjukkan keterbatasan. Banyak skenario nyata yang tidak dapat diselesaikan hanya dengan satu kali respons dari model.

Sistem ternyata perlu melakukan serangkaian langkah, memanfaatkan sumber daya eksternal, menyimpan konteks dalam jangka waktu tertentu, hingga menyesuaikan tindakannya berdasarkan hasil yang diperoleh di setiap tahap.

Pada titik inilah konsep AI Agent mulai muncul sebagai evolusi dari penggunaan LLM yang bersifat pasif menuju sistem yang lebih aktif, adaptif, dan berorientasi pada penyelesaian masalah.

Di materi ini kita akan membahas latar belakang kemunculan konsep agent yang bermula dari keterbatasan Generative AI konvensional, lalu di akhir Anda akan melihat konsep yang lebih menarik tentang Agentic AI.

Pertama-tama, sebaiknya ketahui terlebih dahulu mengapa kita membutuhkan Agent?

Untuk memahami alasan di balik lahirnya konsep agent, Anda perlu melihat batasan dari pendekatan yang selama ini digunakan dalam pengembangan sistem berbasis LLM. Tiga pendekatan utama yang umum diterapkan adalah memanggil LLM secara langsung, Retrieval-Augmented Generation atau RAG, dan fine-tuning model. Mari kita bahas satu per satu.

### Batasan Large Language Model (LLM)

Pemanggilan LLM secara langsung merupakan pendekatan paling dasar dalam memanfaatkan model bahasa. Sistem mengirimkan prompt ke model, kemudian sistem menerima satu respons sebagai hasil. Di modul-modul sebelumnya, Anda juga sudah mempraktikkan langsung pendekatan ini.

![LLM Direct Call](https://assets.cdn.dicoding.com/original/academy/dos-15a75a1926154147e3ea7f9ec4571df620260429132442.png)

Pendekatan ini cocok untuk tugas-tugas yang bersifat statis dan dapat diselesaikan dalam satu langkah, seperti menjawab pertanyaan, merangkum teks, atau menghasilkan potongan kode.

Namun, dalam skenario yang lebih kompleks, pendekatan ini mulai menemui kendala. Model tidak memiliki kesadaran terhadap tujuan jangka panjang. Setiap pemanggilan bersifat terisolasi dan tidak memiliki pemahaman tentang apa yang telah dilakukan sebelumnya, kecuali jika konteks tersebut secara eksplisit dikirim ulang dalam prompt. Hal ini menyebabkan sistem sulit untuk menangani proses yang membutuhkan perencanaan bertahap, pengambilan keputusan berulang, dan evaluasi terhadap hasil di setiap langkah.

Sebagai contoh, ada sebuah sistem yang diminta untuk melakukan riset topik, mengumpulkan referensi dari berbagai sumber, menyusun ringkasan, dan kemudian menghasilkan laporan akhir. Dengan pendekatan LLM calls biasa, seluruh proses tersebut harus dirangkum dalam satu atau beberapa prompt yang panjang dan kompleks.

Sistem tidak memiliki mekanisme internal untuk memutuskan langkah apa yang harus dilakukan terlebih dahulu atau bagaimana menyesuaikan strategi jika informasi yang diperoleh tidak sesuai dengan harapan.

Keterbatasan ini menunjukkan bahwa LLM, dalam bentuk pemanggilan langsung, lebih berperan sebagai mesin respons daripada sebagai entitas yang dapat bertindak secara mandiri untuk mencapai suatu tujuan.

Selanjutnya, mari kita lihat pendekatan lain, yaitu RAG.

### Batasan Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation (RAG) dikembangkan untuk mengatasi salah satu kelemahan utama LLM, yaitu keterbatasan pengetahuan kontekstual dan risiko halusinasi. Dengan RAG, sistem dapat mengambil informasi dari sumber eksternal seperti basis data, dokumen, atau mesin pencari, kemudian menggabungkannya ke dalam prompt sebelum menghasilkan respons.

![RAG Architecture](https://assets.cdn.dicoding.com/original/academy/dos-797aaa0e47236bb0a559940391cb304320260429132510.png)

Pendekatan ini sangat efektif untuk memastikan bahwa jawaban yang dihasilkan lebih akurat dan relevan dengan konteks terkini. Namun, RAG pada dasarnya masih mempertahankan paradigma yang reaktif. Sistem tetap menunggu perintah dari pengguna, melakukan proses retrieval, dan kemudian menghasilkan satu respons sebagai hasil akhir.

RAG tidak secara inheren memiliki kemampuan untuk merencanakan rangkaian tindakan. Misalnya, sistem tidak dapat memutuskan bahwa ia perlu melakukan beberapa kali pencarian, membandingkan hasil dari berbagai sumber, lalu memilih strategi terbaik untuk menyajikan informasi. Semua alur tersebut harus ditentukan secara eksplisit oleh pengembang dalam bentuk pipeline yang kurang fleksibel.

Dengan kata lain, RAG ini memperkaya pengetahuan model, tetapi belum memberikan otonomi kepada sistem untuk mengelola proses penyelesaian tugas secara dinamis.

### Batasan Fine-Tuning

Fine-tuning merupakan pendekatan yang bertujuan untuk menyesuaikan perilaku model dengan kebutuhan spesifik melalui pelatihan tambahan menggunakan dataset khusus. Dengan cara ini, model dapat menjadi lebih andal dalam domain tertentu, seperti layanan pelanggan, analisis medis, atau pemrosesan dokumen hukum.

Meskipun efektif untuk meningkatkan kualitas respons, fine-tuning memiliki keterbatasan dalam hal fleksibilitas dan skalabilitas. Proses ini membutuhkan sumber daya komputasi yang besar, waktu yang tidak sedikit, serta keahlian khusus dalam pengelolaan data dan pelatihan model.

![Fine-Tuning Process](https://assets.cdn.dicoding.com/original/academy/dos-2fd6ed0370acb135bfca51200c0e6e8e20260429132530.png)

Selain itu, fine-tuning tidak secara langsung memberikan kemampuan kepada model untuk bertindak secara mandiri. Model yang telah di-fine-tune tetap beroperasi dalam pola yang sudah ditentukan. Ia tidak memiliki mekanisme internal untuk menentukan penggunaan alat tertentu, penyimpanan informasi, atau perubahan strategi berdasarkan hasil yang diperoleh di tengah proses.

Keterbatasan ini menunjukkan bahwa peningkatan kualitas model saja belum cukup untuk membangun sistem AI yang mampu menangani tugas kompleks dan berorientasi pada tujuan.

**Lalu apa yang bisa kita lakukan? Apakah hanya berhenti di sana?** 

Melihat berbagai keterbatasan tersebut, muncul pertanyaan mengenai arah evolusi sistem AI berbasis LLM. Jika pendekatan reaktif tidak lagi memadai, dibutuhkan paradigma baru yang memungkinkan sistem untuk tidak hanya menjawab, tetapi juga bertindak.

Perkembangan ini dapat dipahami sebagai sebuah proses yang dimulai dari Generative AI, kemudian berlanjut ke AI Agent, dan pada tahap yang lebih lanjut, muncullah Agentic AI.

Pada tahap Generative AI, sistem berfokus pada kemampuan menghasilkan konten. Interaksi bersifat satu arah, di mana pengguna menjadi pengendali utama yang menentukan apa yang harus dilakukan oleh model.

Pada tahap AI Agent, sistem mulai diberi peran yang lebih aktif. Model tidak hanya menghasilkan respons, tetapi juga dapat menggunakan alat, menyimpan memori, dan menjalankan serangkaian langkah untuk mencapai tujuan tertentu. Di sini, konsep tujuan atau goal menjadi elemen penting. Sistem dirancang untuk memahami apa yang ingin dicapai, bukan hanya apa yang harus dijawab.

Tahap berikutnya adalah Agentic AI. Ada hal penting yang harus Anda pahami, istilah *'agentic'* di sini merujuk pada sifat atau tingkat otonomi sebuah sistem, bukan semata-mata pada jumlah agen yang bekerja. Sebuah sistem dikatakan memiliki kemampuan agentik jika mampu melakukan *higher order thinking*, merencanakan tugas yang kompleks, mengevaluasi hasil kerjanya sendiri, dan beradaptasi secara dinamis tanpa intervensi manusia.

Oleh karena itu, sistem yang agentic bisa saja hanya berupa satu agen tunggal dengan pola kerja iteratif. Meskipun demikian, pada tingkatan yang paling kompleks, sifat agentik ini sering kali diwujudkan dalam bentuk Multi-Agent Systems. Dalam skenario ini, beberapa agen dengan spesialisasi masing-masing bekerja sama di bawah sebuah mekanisme orkestrasi untuk memecahkan masalah berskala besar yang tidak efisien jika hanya ditangani oleh satu entitas.

Bagian ini berfungsi sebagai jembatan konseptual antara penggunaan LLM secara konvensional dan pemahaman tentang sistem agent yang lebih maju. Perbedaan utama tidak hanya terletak pada teknologi yang digunakan, tetapi juga pada cara kita memandang peran AI dalam sebuah sistem.

Pada pendekatan Generative AI, AI diposisikan sebagai alat. Pengguna memberikan instruksi, dan AI memberikan hasil. Pada pendekatan AI Agent, AI mulai diposisikan sebagai pelaksana tugas. Sistem diberi tujuan, dan AI bertanggung jawab untuk menentukan langkah-langkah yang diperlukan untuk mencapainya.

Dalam Agentic AI, AI diposisikan sebagai bagian dari sebuah ekosistem cerdas. Bukan hanya satu entitas yang bekerja, melainkan jaringan agent yang saling berkoordinasi. Jadi, fokusnya tidak lagi hanya pada penyelesaian satu tugas, tetapi pada pengelolaan proses yang kompleks dan berkelanjutan.

Dengan memahami jembatan ini, Anda diharapkan dapat melihat bahwa konsep agent bukan sekadar tren teknologi, melainkan evolusi dari kebutuhan akan sistem AI yang lebih mandiri, adaptif, dan mampu beroperasi dalam lingkungan yang dinamis.

Pada materi ini, kita telah membahas latar belakang kemunculan konsep AI Agent melalui peninjauan terhadap keterbatasan pendekatan Generative AI konvensional. Pemanggilan LLM secara langsung, RAG, dan fine-tuning masing-masing memiliki peran penting, namun belum cukup untuk membangun sistem yang mampu bertindak secara mandiri dan berorientasi pada tujuan.

Di materi-materi selanjutnya, kita akan mempelajari arsitektur, pola desain, dan implementasi praktis AI Agent. Sampai bertemu di sana!

---

## AI Agent dengan LLM

Pada materi sebelumnya, kita telah membahas mengapa pendekatan Generative AI konvensional mulai menunjukkan keterbatasan ketika dihadapkan pada permasalahan yang kompleks dan berorientasi tujuan. Kita juga telah melihat konsep AI Agent sebenarnya sebagai respons terhadap kebutuhan akan sistem yang tidak hanya mampu menjawab, tetapi juga mampu bertindak.

Pada materi ini, kita akan memfokuskan pembahasan pada cara membangun AI Agent dengan memanfaatkan LLM sebagai komponen inti. Anda akan mempelajari definisi AI Agent dari berbagai sudut pandang dan memahami anatomi dasar yang membentuk sebuah agent modern.

### Definisi AI Agent

Untuk membangun pemahaman yang kuat, penting bagi kita untuk melihat definisi konsep AI Agent oleh para peneliti dan praktisi di bidang kecerdasan buatan. Definisi ini tidak hanya memberikan gambaran secara konseptual, tetapi juga membantu kita memahami peran dan tanggung jawab sebuah agent dalam sebuah sistem.

#### Menurut Stuart Russell dan Peter Norvig

Dalam buku “Artificial Intelligence: A Modern Approach”, Stuart Russell dan Peter Norvig mendefinisikan agent sebagai entitas yang mempersepsikan lingkungannya melalui sensor dan bertindak terhadap lingkungan tersebut melalui aktuator. Definisi ini menekankan dua elemen utama, yaitu kemampuan untuk mengamati keadaan lingkungan dan kemampuan untuk melakukan tindakan yang memengaruhi lingkungan.

![Agent Environment Interaction](https://assets.cdn.dicoding.com/original/academy/dos-b302586c960d6136eb4727c3effc731220260326103115.png)

Jika kita terapkan dalam konteks AI berbasis LLM, sensor dapat dipahami sebagai berbagai bentuk input yang diterima oleh sistem, seperti teks dari pengguna, data dari API, atau hasil pencarian dari web. Sementara itu, aktuator dapat dipahami sebagai bentuk output atau tindakan yang dihasilkan oleh agent, seperti mengirim permintaan ke layanan eksternal, menyimpan data ke basis data, atau menghasilkan respons yang ditampilkan kepada pengguna.

Namun, penting untuk Anda ingat! Dalam konteks AI Agent, aktuator tidak dipicu oleh instruksi implisit dari pengembang (*hard-coded*). Sebaliknya, LLM-lah yang secara mandiri memutuskan kapan, mengapa, dan alat (*tool*) mana yang harus digerakkan untuk mencapai tujuan yang diberikan.

Definisi ini menempatkan agent sebagai bagian aktif dari sebuah sistem, bukan sekadar komponen yang menghasilkan informasi. Agent dipandang sebagai entitas yang terus berinteraksi dengan lingkungannya untuk mencapai tujuan tertentu.

#### Menurut Praktisi Industri

Selain definisi akademik, berbagai perusahaan teknologi besar juga memberikan interpretasi mereka terhadap konsep AI Agent.

> **Definisi Praktisi Industri (Dari aktivitas interaktif):**
> *   **Google:** AI Agent sebagai sistem yang mampu menggunakan model bahasa untuk merencanakan, bernalar, dan menggunakan alat dalam rangka menyelesaikan tugas secara mandiri. Penekanan utama terletak pada kemampuan agent untuk memilih langkah-langkah yang perlu diambil, bukan hanya menghasilkan jawaban.
> *   **Amazon Web Services (AWS):** AI Agent sebagai aplikasi berbasis AI yang dapat berinteraksi dengan pengguna dan sistem lain, serta mengambil tindakan secara otomatis berdasarkan tujuan yang telah ditentukan. Definisi ini menyoroti aspek integrasi agent dengan layanan dan infrastruktur yang lebih luas.
> *   **IBM:** AI Agent sebagai komponen cerdas yang dapat mengamati konteks, mengambil keputusan, dan bertindak dalam sebuah alur kerja digital. Fokusnya berada pada peran agent sebagai pengambil keputusan dalam sistem bisnis dan operasional.

Jika kita rangkum, ketiga perspektif ini memiliki benang merah yang sama, yaitu **AI Agent** bukan hanya sistem yang menghasilkan output, melainkan sistem yang mampu memahami konteks, merencanakan tindakan, dan berinteraksi dengan lingkungan untuk mencapai tujuan tertentu.

> **Tahukah Anda?**
> 
> Mungkin Anda bertanya-tanya, "Apakah konsep AI Agent ini benar-benar baru muncul setelah ada ChatGPT, Gemini, dan kawan-kawan?"
> 
> Jawabannya adalah **tidak**. Justru, konsep Agent adalah salah satu fondasi tertua dalam sejarah kecerdasan buatan. Bedanya, dulu "Agent" itu memiliki tubuh fisik (robot), sedangkan sekarang mereka hidup di dalam server (software).
> 
> Mari kita telusuri evolusi menarik ini, dimulai dari laboratorium robotika **tahun 60-an** hingga ke laptop Anda saat ini.
> 
> Jika kita harus menunjuk satu "moyang" dari semua AI Agent modern, gelarnya jatuh kepada **Shakey the Robot**.
> 
> Dikembangkan oleh SRI International antara tahun 1966 hingga 1972, Shakey adalah robot mobil *general-purpose* pertama yang mampu bernalar atas tindakannya sendiri.
> 
> ![Shakey the Robot](https://assets.cdn.dicoding.com/original/academy/dos-0066b455ffb61675337f059ea4ebbb7d20260326103137.png)
> 
> Sumber: [https://www.sri.com/hoi/shakey-the-robot/](https://www.sri.com/hoi/shakey-the-robot/)
> 
> *   Tugasnya sederhana, seperti mendorong balok, memindahkan barang dari ruang A ke ruang B, atau menyalakan lampu.
> *   Kenapa ia spesial? Shakey tidak sekadar di program dengan instruksi yang kaku seperti robot di pabrik yang hanya mengelas di titik yang sama berulang kali. Shakey memiliki kemampuan untuk menganalisis perintah.
>     *   Jika Anda menyuruhnya: "Dorong balok di atas meja," Shakey tidak akan langsung menabrak meja.
>     *   Ia akan berpikir (Plan): "Saya harus mencari jalan landai (*ramp*) dulu, mendorongnya ke dekat meja, naik ke atas, baru mendorong baloknya."
> 
> Shakey adalah implementasi nyata pertama dari siklus Perceive, Think, Act. Ia "melihat" dengan kamera TV, "berpikir" dengan komputer raksasa (kala itu), dan "bertindak" dengan rodanya. Inilah cikal bakal logika Agentic yang kita gunakan hari ini.
> 
> ![Perceive Think Act Cycle](https://assets.cdn.dicoding.com/original/academy/dos-04221d377474e45cbea17e499ae0999f20260429132808.png)
> 
> Memasuki **tahun 80-an dan 90-an**, para peneliti mulai menyadari bahwa membangun robot fisik itu mahal, lambat, dan rentan rusak. Namun, di saat yang sama, dunia digital dan internet mulai berkembang pesat.
> 
> Fokus pun bergeser. Para ilmuwan mulai memindahkan otak agent dari tubuh robot ke dalam dunia maya. Lahirlah istilah Software Agents.
> 
> *   **Lingkungannya Berubah**
>     Jika Shakey hidup di laboratorium, Softbots hidup di dalam sistem operasi dan jaringan internet.
> *   **Tugasnya Berubah**
>     Dari memindahkan balok fisik, menjadi memindahkan data, memfilter email, atau mencari informasi di web (awal mula web crawler).
> 
> Di era inilah definisi akademis tentang Rational Agent atau AI Agent yang kita bahas sebelumnya, mulai dibakukan oleh peneliti seperti Stuart Russell dan Peter Norvig. Mereka merumuskan bahwa kecerdasan bukan hanya tentang "tahu apa", tapi tentang "bertindak rasional".

### Anatomi AI Agent

Setelah memahami definisinya, langkah berikutnya adalah mempelajari komponen utama yang membentuk sebuah AI Agent modern. Secara umum, sebuah agent terdiri dari tiga elemen besar, yaitu LLM untuk penalaran, sistem memori, dan alat atau tools. Arsitektur ini merujuk pada kerangka kerja [LLM-Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) yang dipopulerkan oleh Lilian Weng. Kerangka kerja ini sudah menjadi standar industri yang digunakan oleh berbagai framework besar seperti LangChain, AutoGPT, dan Microsoft Semantic Kernel.

![AI Agent Anatomy Diagram](https://assets.cdn.dicoding.com/original/academy/dos-8cbd99a8ff6143fa00a18b842819a75d20260429132829.png)

#### LLM sebagai Reasoning Engine

Large Language Model (LLM) berperan sebagai pusat kendali dari sebuah AI Agent. Di sinilah proses penalaran, perencanaan, dan pengambilan keputusan dilakukan. Ketika agent menerima sebuah tujuan atau tugas, LLM akan menganalisis apa yang perlu dilakukan, menentukan urutan langkah, dan memutuskan kapan harus menggunakan tools tertentu, termasuk pengelolaan dan pemanggilan memori.

![LLM Reasoning Engine](https://assets.cdn.dicoding.com/original/academy/dos-c9f39fd32570a5ed69ecf32c6de4908120260429132840.png)

Dalam konteks ini, LLM tidak hanya berfungsi sebagai generator teks, tetapi sebagai reasoning engine. Ia membantu agent menjawab pertanyaan seperti langkah apa yang harus dilakukan terlebih dahulu, informasi apa yang masih kurang, dan tindakan apa yang paling relevan untuk mencapai tujuan.

Sebagai contoh, jika agent diminta untuk membuat laporan pasar, LLM dapat memutuskan bahwa ia perlu mencari data terbaru, membandingkan beberapa sumber, menyusun ringkasan, dan baru kemudian menghasilkan laporan akhir. Semua proses tersebut direncanakan dan dikendalikan oleh LLM.

#### Memory

Agar dapat bertindak secara konsisten dan berkelanjutan, AI Agent membutuhkan sistem memori. Memori memungkinkan agent untuk mengingat informasi yang telah diperoleh sebelumnya, memahami konteks percakapan, dan menyesuaikan perilakunya berdasarkan pengalaman.

Secara umum, memori dalam AI Agent dapat dibagi menjadi beberapa jenis.

##### Short-Term Memory

Short-term memory adalah jenis memori yang digunakan oleh AI Agent untuk menyimpan informasi yang bersifat sementara dan relevan dengan konteks saat ini. Memori ini berperan menjaga kesinambungan interaksi dalam satu sesi atau satu rangkaian tugas.

Fungsi utama *short-term memory* adalah memastikan bahwa agent tidak “lupa” apa yang baru saja terjadi. Dengan memori ini, agent dapat memahami hubungan antarperintah, melacak langkah yang sedang dikerjakan, dan menjaga alur percakapan tetap konsisten.

Penting untuk dipahami bahwa *short-term memory* berbeda dengan *caching* pada proses *inference*. *Caching* biasanya bertujuan mempercepat respons untuk input yang sama, sedangkan *short-term memory* bertujuan menyediakan **konteks** bagi LLM agar dapat memahami instruksi yang berbeda-beda dalam satu alur kerja.

Karena LLM pada dasarnya bersifat *stateless* atau tidak mengingat apapun setelah satu kali proses selesai, maka *short-term memory* harus dikelola oleh sistem eksternal di sisi aplikasi. Tabel berikut menampilkan media penyimpanan *short-term memory* yang umum digunakan.

| Media Penyimpanan | Penjelasan | Kelebihan | Kekurangan |
| :--- | :--- | :--- | :--- |
| **In-Memory (RAM)** | Data disimpan dalam variabel lokal atau *state* aplikasi (misal: *array* atau *list*). | Sangat cepat (*low latency*). | Data akan hilang jika aplikasi di-*restart* atau server mati. |
| **External Cache (Redis)** | Menggunakan basis data *key-value* cepat yang terpisah dari aplikasi utama. | Persisten dalam satu sesi dan mendukung skalabilitas banyak pengguna. | Menambah kompleksitas infrastruktur dan ada sedikit *latency* jaringan. |

*Short-term memory* ini memiliki batas ukuran yang disebut **Context Window** atau kapasitas maksimal token yang bisa diproses LLM dalam satu waktu. Jika batas ini tercapai, Anda harus menerapkan strategi berikut agar memori tetap efisien.

1.  **Sliding Window**
    Hanya menyimpan sejumlah N pesan atau token terbaru. Pesan yang paling lama akan dihapus.
    *   Kelebihannya, hemat biaya dan latensi tetap stabil.
    *   Kekurangannya, agent bisa kehilangan instruksi penting yang diberikan di awal sesi.
2.  **Conversation Summary**
    Agent secara berkala meringkas riwayat percakapan lama menjadi satu paragraf pendek. Paragraf ringkasan inilah yang dikirim sebagai konteks.
    *   Kelebihannya, mampu menyimpan informasi penting dari percakapan yang sangat panjang.
    *   Kekurangannya, ada risiko detail teknis yang spesifik hilang saat proses peringkasan.

Dalam chatbot, Short-term memory memungkinkan agent untuk memahami rujukan atau kata ganti. Tanpa memori ini, saat Anda berkata "Perbaiki kode itu," agent tidak akan tahu kode mana yang Anda maksud. Dalam otomatisasi, memori ini membantu agent melacak status langkah-langkah yang sedang dikerjakan (misal: "Langkah 1 sukses, lanjut ke Langkah 2").

Nah, Keterbatasan utama short-term memory adalah kapasitasnya yang terbatas. Jika konteks yang dikirim terlalu panjang, biaya penggunaan API akan meningkat dan respons AI akan menjadi lebih lambat (*high latency*). Selain itu, pengembang harus pandai menyaring informasi mana yang relevan agar tidak terjadi "kebisingan" konteks yang bisa mengganggu penalaran LLM.

##### Long-Term Memory

Long-term memory adalah jenis memori yang dirancang untuk menyimpan informasi dalam jangka panjang dan dapat digunakan kembali di berbagai sesi dan tugas yang berbeda. Memori ini berfungsi sebagai basis pengetahuan personal atau organisasi bagi AI Agent.

Berbeda dengan *short-term memory* yang akan terhapus saat sesi berakhir, *long-term memory* menyediakan akses ke pengetahuan yang bersifat persisten. Dengan memori ini, agent dapat mengingat preferensi pengguna, kebijakan perusahaan, atau informasi penting lainnya yang relevan untuk jangka waktu panjang.

Secara teknis, *long-term memory* tidak disimpan dalam bentuk teks biasa di *database* relasional (SQL), melainkan menggunakan **Vector Database**. Anda pasti sudah familiar ya dengan istilah ini. Mari kita ingat-ingat lagi, proses ini melibatkan beberapa komponen utama, yaitu:

1.  **Embeddings**
    Informasi (teks atau dokumen) diubah menjadi rangkaian angka numerik (vektor) yang merepresentasikan makna semantiknya.
2.  **Semantic Search**
    Saat agent membutuhkan informasi, ia tidak mencari kata kunci yang sama persis (*keyword matching*), melainkan mencari vektor yang memiliki jarak paling dekat secara makna.

Nah, agar pengambilan informasi efisien dan akurat, pengembang biasanya menggunakan struktur **QMD** dalam mengelola memori jangka panjang:

*   **Query (Embeddings)**
    Representasi numerik dari data yang disimpan untuk memudahkan pencarian cepat.
*   **Metadata**
    Informasi tambahan yang ditempelkan pada dokumen, misalnya tanggal *input*, ID pengguna, atau kategori. Metadata sangat penting untuk melakukan *filtering* agar agent tidak mengambil data yang sudah kedaluwarsa.
*   **Document (Chunk)**
    Potongan teks asli yang akan dibaca oleh LLM setelah ditemukan oleh sistem *retrieval*.

*Long-term memory* juga memiliki media penyimpanan yang umum digunakan. Anda bisa melihatnya pada tabel berikut!

| Media Penyimpanan | Penjelasan | Kelebihan | Kekurangan |
| :--- | :--- | :--- | :--- |
| **Vector Database** (Pinecone, Weaviate, Milvus) | Database khusus untuk menyimpan dan mencari vektor secara cepat dalam skala besar. | Sangat cepat untuk pencarian ribuan dokumen; fitur *filtering* metadata yang kuat. | Biaya operasional biasanya lebih mahal dan butuh manajemen indeks. |
| **Local Vector Store** (ChromaDB, FAISS) | Penyimpanan vektor yang berjalan secara lokal di server atau *file system*. | Gratis, mudah di-*setup*, dan sangat cepat untuk data skala kecil-menengah. | Sulit untuk dikelola jika data sudah mencapai jutaan baris atau butuh akses multi-user. |

Dalam sistem layanan pelanggan,dengan *long-term memory*, agent dapat menyimpan preferensi pelanggan, riwayat keluhan, atau solusi yang pernah berhasil. Informasi ini dapat digunakan kembali untuk memberikan layanan yang lebih personal. Dalam sistem pembelajaran, dengan *long-term memory*, agent dapat menyimpan topik yang sudah dipelajari oleh pengguna dan menyesuaikan materi berikutnya berdasarkan progres tersebut.

Namun, memori ini juga datang dengan kekurangan, loh. Pengelolaan *long-term memory* memerlukan perhatian khusus terhadap Privasi dan Keamanan Data. Karena data bersifat permanen, pengembang harus memastikan ada mekanisme untuk menghapus data lama (*data pruning*) atau memperbarui informasi yang sudah tidak valid agar agent tidak memberikan jawaban yang menyesatkan atau melanggar privasi pengguna.

##### Procedural Memory

Procedural memory adalah jenis memori yang menyimpan pengetahuan tentang cara melakukan sesuatu. Berbeda dengan *long-term memory* yang berfokus pada fakta statis (siapa, apa, di mana), *procedural memory* berfokus pada bagaimana, yakni langkah-langkah, aturan main, dan alur kerja (*workflow*) yang harus diikuti.

Fungsi utama *procedural memory* adalah memastikan bahwa agent bertindak secara konsisten dan sesuai dengan standar yang telah ditetapkan (SOP). Dengan memori ini, agent tidak perlu menebak-nebak urutan kerja karena cukup merujuk pada panduan yang telah kita siapkan.

Dalam arsitektur LLM Agent modern, *procedural memory* disimpan melalui instruksi terstruktur. Berikut adalah media penyimpanannya:

| Media Penyimpanan | Cara Kerja | Kelebihan | Kekurangan |
| :--- | :--- | :--- | :--- |
| **System Prompt/ Instruction** | Aturan ditulis langsung ke dalam instruksi dasar agent. | Sangat kuat dan selalu diprioritaskan oleh LLM. | Terbatas oleh kuota token; sulit dikelola jika prosedurnya sangat banyak. |
| **Penyimpanan Eksternal** (JSON/YAML) | Prosedur disimpan sebagai file konfigurasi terstruktur yang dibaca saat *runtime*. | Mudah diperbarui tanpa mengubah kode program; sangat rapi. | Membutuhkan logika tambahan untuk memparsing data ke LLM. |
| **Few-Shot Examples** | Memberikan contoh pasangan "Input -> Langkah Kerja -> Output" sebagai referensi. | Agent belajar pola kerja dengan sangat cepat melalui contoh nyata. | Membutuhkan banyak token karena setiap contoh cukup panjang. |

Ketika agent menghadapi sebuah tugas, LLM akan merujuk pada prosedur yang tersimpan untuk menentukan langkah-langkah yang harus diikuti. Memori ini dapat bersifat statis maupun dinamis.

*   **Prosedur Statis** berisi aturan bisnis yang jarang berubah. Contoh: *"Setiap transaksi di atas 10 juta wajib meminta verifikasi dua langkah (2FA)."* 
*   **Prosedur Dinamis** berisi alur kerja yang bisa diperbarui berdasarkan umpan balik. Contoh: *"Jika cara A gagal dalam 3 kali percobaan, gunakan prosedur B sebagai cadangan."* 

Dalam perbankan digital, agent menggunakan *procedural memory* untuk mengikuti langkah verifikasi identitas (KYC) yang ketat sebelum memproses transaksi. Dalam sistem pelaporan, memori ini memastikan agent menggunakan format Markdown atau struktur JSON yang konsisten setiap kali menghasilkan laporan, sehingga mudah dibaca oleh sistem lain.

Sama seperti memori-memori sebelumnya, *procedural memory* juga hadir dengan sejumlah kekurangan. Jika prosedur tidak diperbarui seiring perubahan kebijakan perusahaan, agent berisiko melakukan langkah yang sudah kedaluwarsa. Sebaliknya, prosedur yang terlalu kaku dapat membatasi kreativitas LLM dalam menangani kasus-kasus unik yang tidak terdaftar di buku panduan. Strategi terbaik adalah memberikan prosedur utama yang jelas, namun tetap memberikan ruang bagi LLM untuk melakukan *reasoning* jika menghadapi situasi tertentu.

##### Episodic Memory

Episodic memory adalah jenis memori yang menyimpan catatan tentang peristiwa atau pengalaman spesifik yang pernah dialami oleh agent. Memori ini mirip dengan ingatan manusia tentang kejadian di masa lalu, seperti mengingat apa yang terjadi, kapan hal itu terjadi, dan bagaimana hasilnya.

Fungsi utama *episodic memory* adalah memberikan konteks historis yang kaya. Dengan memori ini, agent tidak hanya sekadar tahu suatu teori, tapi juga pernah mengalaminya. Agent dapat mengingat interaksi tertentu dan menyesuaikan strateginya berdasarkan keberhasilan atau kegagalan yang pernah dialami sebelumnya.

Berbeda dengan memori lainnya, *episodic memory* membutuhkan detail yang sangat spesifik. Secara teknis, setiap histori disimpan sebagai catatan yang dilengkapi dengan Metadata.

Media penyimpanan yang umum digunakan pada memori ini dapat kamu lihat pada tabel berikut:

| Media Penyimpanan | Cara Kerja | Kelebihan | Kekurangan |
| :--- | :--- | :--- | :--- |
| **Relational Database** (PostgreSQL/MongoDB) | Menyimpan log interaksi lengkap dengan kolom metadata (waktu, status, error). | Sangat akurat untuk audit trail dan mudah di filter berdasarkan waktu/user. | Tidak mendukung pencarian makna *(semantic search)* secara bawaan jika tanpa ekstensi. |
| **Vector Database dengan Metadata** | Menyimpan ringkasan kejadian dalam bentuk vektor agar bisa dicari berdasarkan kemiripan situasi. | Agent bisa menemukan "pengalaman serupa" meskipun detailnya sedikit berbeda. | Butuh pemrosesan ekstra untuk mengubah peristiwa menjadi *embedding*. |

Ketika agent menghadapi masalah, ia akan melakukan kueri ke *episodic memory*: *"Apakah saya pernah menghadapi error koneksi database seperti ini sebelumnya?"*.

Sistem akan mengembalikan episode yang paling relevan. Jika catatan menunjukkan bahwa *"Terakhir kali ini terjadi, solusi yang berhasil adalah me-restart service Docker"*, maka LLM akan memprioritaskan tindakan tersebut sebagai langkah pertama.

Penggunaan *episodic memory* pada technical support, membuat agent mengingat bahwa minggu lalu pengguna A mengalami kendala instalasi dan solusi yang berhasil adalah mengganti versi Python. Saat pengguna A bertanya lagi tentang hal serupa, agent langsung memberikan solusi tepat sasaran tanpa bertanya dari nol. *Episodic memory* digunakan juga untuk *self-correction*, agent bisa mengingat bahwa langkah sebelumnya gagal karena *timeout*, sehingga di langkah berikutnya ia secara otomatis meningkatkan parameter *delay*.

*Episodic memory* ini dapat tumbuh sangat cepat. Jika tidak dikelola, memori ini akan dipenuhi data sampah yang tidak relevan. Pengembang harus menerapkan:

1.  **Selection (Seleksi)**
    Hanya simpan histori yang memiliki nilai belajar tinggi, misalnya saat ada error atau saat tugas selesai dengan sukses.
2.  **Consolidation (Peringkasan)**
    Histori lama yang serupa diringkas menjadi satu pola pengetahuan agar ruang penyimpanan tetap efisien.

Nah, agar Anda lebih memahami perbedaan setiap jenis memori, mari kita lihat studi kasus sebuah AI Agent yang bertugas sebagai asisten pengembang software. Mari kita lihat bagaimana ia menggunakan seluruh sistem memorinya saat membantu Anda.

| Jenis Memori | Apa yang Diingat Agent? | Perannya dalam Tugas |
| :--- | :--- | :--- |
| **Short-Term** | "User baru saja meminta saya untuk deploy aplikasi ke server staging lima menit yang lalu." | Menjaga agar Agent tetap fokus pada tugas yang sedang berjalan tanpa perlu ditanya ulang. |
| **Long-Term** | "Kredensial server staging adalah IP 192.168.1.50 dan sistem operasinya adalah Ubuntu 22.04." | Mengambil data teknis yang statis dan permanen dari basis data pengetahuan perusahaan. |
| **Procedural** | "Prosedur deploy yang aman adalah:<br><br>1. Tarik kode dari Git,<br><br>2. Jalankan unit test,<br><br>3. Build Docker image,<br><br>4. Push ke server." | Memastikan Agent mengikuti standar operasional (SOP) yang sudah ditentukan, bukan asal tebak. |
| **Episodic** | "Minggu lalu saat deploy aplikasi yang sama, proses gagal karena port 8080 sedang digunakan oleh aplikasi lain." | Memberikan peringatan dini berdasarkan pengalaman masa lalu agar kesalahan yang sama tidak terulang. |

#### Tools

Tools adalah komponen yang memungkinkan agent untuk berinteraksi dengan dunia di luar LLM. Tanpa tools, agent hanya dapat berpikir dan menghasilkan teks, tetapi tidak dapat memengaruhi atau memanfaatkan sumber daya eksternal.

Beberapa jenis tools yang umum digunakan dalam AI Agent meliputi web search, image generation, retrieval, dan API interface.

![Agent Tools](https://assets.cdn.dicoding.com/original/academy/dos-4b73f13dbf24e5f040694a6a0354b17220260429132901.png)

##### Web Search

Web search adalah kemampuan agent untuk mencari dan mengambil informasi dari sumber eksternal yang berada di luar pengetahuan bawaan model. Dalam praktiknya, tool ini biasanya terhubung dengan search engine, API berita, atau indeks dokumen publik.

Fungsi utama web search adalah memastikan bahwa agent memiliki akses ke informasi yang bersifat terkini dan relevan. Model bahasa dilatih pada data historis, sehingga tidak selalu mengetahui peristiwa terbaru, perubahan kebijakan, atau data yang sering diperbarui seperti harga, jadwal, dan statistik.

Ketika agent menerima sebuah tugas, LLM akan menentukan apakah informasi yang dibutuhkan dapat dijawab dari konteks internal atau perlu dicari dari luar. Jika diperlukan, agent akan memanggil tool web search dengan kata kunci tertentu. Tool ini kemudian mengembalikan hasil berupa ringkasan, daftar tautan, atau potongan konten yang relevan. Hasil tersebut dimasukkan kembali ke dalam konteks LLM untuk diproses lebih lanjut.

Web search bergantung pada kualitas sumber eksternal. Jika sumber yang diambil tidak terpercaya, hasil yang diberikan agent juga berisiko tidak akurat. Selain itu, agent perlu dibatasi agar tidak melakukan pencarian berlebihan yang dapat meningkatkan latensi dan biaya pada layanan yang digunakan.

##### Image Generation

Image generation adalah kemampuan agent untuk menghasilkan konten visual sebagai bagian dari proses penyelesaian tugas. Tool ini biasanya terhubung dengan model generatif visual, seperti model text-to-image atau sistem desain otomatis.

Fungsi utamanya adalah memperluas bentuk output agent, tidak hanya dalam bentuk teks, tetapi juga visual. Hal ini penting dalam konteks pembelajaran, pemasaran, presentasi, dan komunikasi visual.

Ketika agent memutuskan bahwa tugas membutuhkan elemen visual, LLM akan menyusun deskripsi atau prompt visual yang sesuai. Prompt ini kemudian dikirimkan ke tool image generation. Tool tersebut menghasilkan gambar berdasarkan deskripsi yang diberikan dan mengembalikannya ke agent sebagai hasil yang dapat ditampilkan atau disimpan.

Dalam sistem pembelajaran digital, agent dapat membuat diagram atau ilustrasi untuk menjelaskan sebuah konsep teknis. Dalam tim pemasaran, agent dapat menghasilkan konsep visual awal untuk poster, banner, atau konten media sosial berdasarkan brief yang diberikan.

Kualitas gambar sangat bergantung pada kejelasan deskripsi yang dibuat oleh agent. Selain itu, penggunaan model visual biasanya membutuhkan sumber daya komputasi yang lebih besar dan waktu pemrosesan yang lebih lama dibandingkan dengan pembuatan teks.

##### Retrieval

Retrieval adalah kemampuan agent untuk mengambil informasi dari sumber internal yang terstruktur maupun tidak terstruktur, seperti basis data, dokumen perusahaan, arsip, atau sistem manajemen pengetahuan.

Fungsi utama retrieval adalah menyediakan akses ke pengetahuan spesifik yang tidak tersedia di internet atau tidak termasuk dalam data pelatihan model. Informasi ini biasanya bersifat internal, sensitif, atau sangat kontekstual terhadap sebuah organisasi atau sistem tertentu.

Agent mengirimkan permintaan ke sistem retrieval dengan representasi kebutuhan informasi, yang sering kali berupa embedding atau representasi semantik dari pertanyaan. Sistem kemudian mencari potongan data yang paling relevan dari indeks atau basis data. Hasil tersebut dikembalikan ke agent dan digunakan sebagai konteks tambahan untuk menghasilkan respons atau mengambil keputusan.

Dalam perusahaan, agent dapat digunakan untuk menjawab pertanyaan karyawan tentang kebijakan internal dengan mengambil informasi langsung dari dokumen resmi. Dalam sistem teknis, agent dapat mencari panduan penggunaan atau dokumentasi API internal ketika diminta untuk membantu pengembang.

Retrieval sangat bergantung pada kualitas data yang disimpan dan cara data tersebut diindeks. Jika dokumen tidak diperbarui atau tidak terstruktur dengan baik, agent dapat mengambil informasi yang tidak relevan atau sudah tidak berlaku.

##### API Interface

API interface adalah kemampuan agent untuk berkomunikasi secara langsung dengan sistem lain melalui antarmuka pemrograman aplikasi. Dengan tool ini, agent tidak hanya memberikan informasi, tetapi juga dapat memicu tindakan nyata dalam sebuah sistem digital.

Fungsi utamanya adalah menghubungkan agent dengan layanan eksternal dan sistem operasional, seperti sistem pembayaran, manajemen proyek, layanan cloud, atau aplikasi internal perusahaan.

Ketika agent memutuskan bahwa sebuah tindakan perlu dilakukan, LLM akan menyusun parameter dan format permintaan sesuai dengan spesifikasi API yang tersedia. Permintaan ini dikirim ke sistem target melalui API interface. Sistem kemudian memproses permintaan tersebut dan mengembalikan respons, seperti status keberhasilan, data hasil, atau pesan kesalahan. Respons ini digunakan oleh agent untuk menentukan langkah selanjutnya.

Dalam sistem pemesanan, agent dapat memanggil API untuk membuat pesanan baru, mengecek status pengiriman, atau membatalkan transaksi. Dalam manajemen proyek, agent dapat menambahkan tugas baru ke dalam papan kerja, memperbarui status pekerjaan, atau mengirim notifikasi kepada anggota tim.

API interface memerlukan pengelolaan keamanan yang ketat, seperti autentikasi dan pembatasan akses. Kesalahan dalam pemanggilan API dapat berdampak langsung pada sistem nyata, sehingga agent perlu dilengkapi dengan mekanisme validasi dan penanganan error yang baik.

Ketiga komponen utama AI Agent, yaitu LLM, memori, dan tools, tidak bekerja secara terpisah. Mereka membentuk sebuah siklus kerja yang saling bergantung. LLM menggunakan informasi dari memori untuk memahami konteks dan merencanakan tindakan. Berdasarkan rencana tersebut, agent memanfaatkan tools untuk berinteraksi dengan lingkungan. Hasil dari interaksi tersebut kemudian disimpan kembali ke dalam memory sebagai pengalaman atau pengetahuan baru.

Siklus ini memungkinkan AI Agent untuk belajar secara operasional, dalam arti meningkatkan kualitas tindakannya dari waktu ke waktu tanpa harus selalu mengandalkan pelatihan ulang model.

*Okay*, kita telah selesai mempelajari definisi AI Agent dari perspektif akademik dan industri, serta memahami bahwa agent adalah sistem yang mampu memersepsikan konteks, mengambil keputusan, dan bertindak untuk mencapai tujuan.

Kita juga telah membedah anatomi dasar AI Agent yang terdiri dari LLM sebagai mesin penalaran, memory sebagai penopang konteks dan pengalaman, dan tools sebagai sarana untuk berinteraksi dengan lingkungan. Pemahaman tentang hubungan antarkomponen ini akan menjadi fondasi penting untuk mempelajari variasi AI Agent pada materi berikutnya. Semangat!

---

## Variasi AI Agent

Pada materi sebelumnya, kita telah mempelajari bagaimana AI Agent dibangun dari tiga komponen utama. Kita juga telah memahami bahwa agent bukan hanya sekadar model yang menghasilkan teks, tetapi sebuah sistem yang mampu merencanakan dan mengeksekusi tindakan untuk mencapai tujuan tertentu.

Kali ini, kita akan membahas variasi AI Agent dari dua sudut pandang utama, yaitu dari sisi alur kerja atau workflow dan dari sisi tipe implementasi. Dengan memahami kedua aspek ini, Anda diharapkan mampu memilih pola desain dan bentuk agent yang paling sesuai dengan kebutuhan sistem yang sedang dibangun.

### Workflow

Workflow menggambarkan bagaimana sebuah agent berpikir, mengambil keputusan, dan bertindak dalam rangka menyelesaikan tugas. Setiap pola workflow memiliki karakteristik, kelebihan, dan batasan yang berbeda.

#### Chain of Thought

Chain of Thought adalah pola di mana agent memecah sebuah masalah menjadi serangkaian langkah penalaran yang lebih kecil dan terstruktur. Alih-alih langsung menghasilkan jawaban akhir, agent terlebih dahulu menelusuri proses berpikir secara bertahap.

![Chain of Thought](https://assets.cdn.dicoding.com/original/academy/dos-e135b1c64be0123f556a969c7f40e03220260326103556.png)

Pola ini membantu agent dalam menangani masalah yang kompleks, terutama yang membutuhkan analisis, perbandingan, atau perhitungan. Dengan membagi masalah menjadi langkah-langkah kecil, agent dapat mengurangi risiko kesalahan dan meningkatkan kualitas keputusan yang diambil.

Dalam implementasi praktis, Chain of Thought sering diwujudkan dalam bentuk prompt atau mekanisme internal yang mendorong LLM untuk menjelaskan alasan di balik setiap langkah sebelum sampai pada hasil akhir.

Sebagai contoh, ketika diminta untuk memilih solusi terbaik dari beberapa opsi, agent dapat terlebih dahulu mengevaluasi kelebihan dan kekurangan masing-masing opsi, lalu menarik kesimpulan berdasarkan analisis tersebut.

#### ReAct Framework

ReAct adalah singkatan dari Reasoning dan Acting. Framework ini menggabungkan proses penalaran dan tindakan dalam sebuah siklus yang berulang. Agent tidak hanya berpikir terlebih dahulu lalu bertindak, tetapi juga mengamati hasil dari tindakannya sebelum melanjutkan ke langkah berikutnya.

Siklus ini terdiri dari tiga tahap utama, yaitu thought, action, dan observation.

![ReAct Framework](https://assets.cdn.dicoding.com/original/academy/dos-e2e44f1aadf4cd546ab887ad15fab1ab20260429133017.png)

##### Thought

Tahap thought merupakan fase di mana agent melakukan proses berpikir sebelum mengambil tindakan apa pun. Pada tahap ini, LLM berperan sebagai mesin penalaran yang menganalisis tujuan, kondisi saat ini, dan informasi yang tersedia.

Peran utama tahap thought adalah membentuk pemahaman yang jelas tentang apa yang sedang dihadapi agent dan apa yang ingin dicapai. Agent mencoba menjawab pertanyaan internal, seperti tujuan akhir yang ingin dicapai, informasi yang sudah dimiliki, dan hal yang masih perlu dicari.

Agent mengidentifikasi konteks dari input yang diterima, baik itu perintah pengguna, hasil dari langkah sebelumnya, maupun data yang tersimpan di dalam memori. Agent kemudian memecah tujuan besar menjadi langkah-langkah kecil yang lebih mudah dikelola.

Selain itu, agent juga mengevaluasi opsi tindakan yang tersedia. Ia mempertimbangkan apakah perlu menggunakan tool tertentu, seperti melakukan pencarian web, mengambil data dari sistem internal, atau memanggil API untuk menjalankan sebuah proses.

Dalam tugas riset, agent pada tahap thought akan menentukan topik utama, kata kunci yang relevan, dan sumber informasi yang kemungkinan paling kredibel. Agent juga dapat memutuskan urutan pencarian, misalnya memulai dari sumber resmi sebelum beralih ke artikel atau laporan pendukung.

Tantangan utama pada tahap ini adalah risiko perencanaan yang kurang tepat. Jika agent salah memahami tujuan atau konteks, langkah berikutnya juga berpotensi menyimpang. Oleh karena itu, kualitas tahap thought sangat menentukan efektivitas keseluruhan pada siklus ReAct.

Setelah agent mengetahui apa yang harus dilakukan, proses akan berpindah ke tahap action. Mari kita lihat yang dilakukan agent di tahap ini!

##### Action

Tahap action adalah fase di mana agent mengeksekusi rencana yang telah disusun pada tahap thought. Pada tahap ini, agent berinteraksi secara langsung dengan lingkungan melalui tools atau sistem eksternal.

Peran utama tahap action adalah mengubah rencana menjadi tindakan nyata. Agent tidak lagi hanya berpikir, tetapi mulai memengaruhi lingkungan dengan memanggil layanan, mengambil data, atau menjalankan proses tertentu.

Agent memilih tool yang paling sesuai dengan kebutuhan langkah saat ini. Jika membutuhkan informasi terbaru, agent akan memanggil web search. Jika perlu menjalankan proses dalam sistem lain, agent akan menggunakan API interface.

Setelah memilih tool, agent menyusun parameter dan format permintaan yang sesuai. Permintaan ini kemudian dikirim ke sistem target, dan agent menunggu respons yang dikembalikan.

Dalam sistem pemesanan, agent pada tahap action dapat memanggil API untuk membuat pesanan baru atau mengecek status transaksi. Dalam analisis data, agent dapat menjalankan script atau fungsi untuk memproses dataset dan menghasilkan hasil sementara.

Tantangan utama dari tahap ini adalah memastikan bahwa tindakan yang diambil aman dan valid. Kesalahan dalam parameter atau pemilihan tool dapat menyebabkan kegagalan sistem atau bahkan dampak yang tidak diinginkan pada layanan eksternal. Oleh karena itu, tahap ini sering dilengkapi dengan mekanisme validasi dan pembatasan akses.

##### Observation

Tahap terakhir sebelum agent memberikan jawaban, observation, merupakan fase di mana agent mengevaluasi hasil dari tindakan yang telah dilakukan. Pada tahap ini, agent menerima umpan balik dari lingkungan dan menafsirkannya sebagai dasar untuk langkah berikutnya.

Peran utama tahap observation adalah menutup satu siklus ReAct dengan proses refleksi. Agent menilai apakah tindakan yang diambil sudah mendekatkan dirinya pada tujuan atau justru memerlukan penyesuaian strategi.

Agent membaca respons dari tool atau sistem eksternal, seperti hasil pencarian, data yang dikembalikan oleh API, atau pesan error. Informasi ini kemudian dibandingkan dengan tujuan dan rencana awal.

Jika hasilnya sesuai, agent dapat melanjutkan ke langkah berikutnya atau menyelesaikan tugas. Jika tidak, agent akan kembali ke tahap thought untuk menyusun rencana baru berdasarkan informasi yang diperoleh.

Dalam pencarian informasi, agent dapat mengevaluasi apakah hasil yang diperoleh sudah relevan dan cukup lengkap. Jika belum, agent dapat memutuskan untuk melakukan pencarian tambahan dengan kata kunci yang berbeda.

Dalam sistem otomatisasi, jika sebuah API mengembalikan pesan error, agent dapat mencatat kesalahan tersebut dan mencoba pendekatan lain, seperti memperbaiki parameter atau memilih fungsi alternatif.

Tantangan utama adalah kemampuan agent untuk menafsirkan umpan balik secara tepat. Tidak semua respons sistem bersifat eksplisit atau mudah dipahami. Agent perlu dilatih atau dirancang untuk mengenali sinyal keberhasilan, kegagalan, dan kondisi ambigu.

Itulah ketiga tahap dalam ReAct Framework. Ketiga tahap ini tentunya tidak berjalan sekali saja, melainkan membentuk sebuah siklus yang berulang. Setelah observation, agent kembali ke tahap thought dengan konteks yang diperbarui. Informasi baru dari hasil pengamatan menjadi bahan penalaran untuk menentukan langkah selanjutnya.

Sifat berulang inilah yang membuat ReAct Framework adaptif. Agent dapat menyesuaikan strategi berdasarkan kondisi nyata di lapangan, bukan hanya berdasarkan rencana awal. Hal ini menjadikan ReAct sangat efektif untuk tugas-tugas yang dinamis, tidak pasti, dan melibatkan banyak interaksi dengan lingkungan eksternal.

#### Plan and Execute Pattern

Plan and Execute adalah pola di mana agent terlebih dahulu menyusun rencana lengkap sebelum mulai mengeksekusi langkah-langkah yang telah ditentukan. Dalam pendekatan ini, proses perencanaan dan proses pelaksanaan dipisahkan secara jelas.

![Plan and Execute Pattern](https://assets.cdn.dicoding.com/original/academy/dos-6a6f59228a2057d0a35b4b8125f7008120260429133045.png)

Pada tahap perencanaan, agent mengidentifikasi tujuan akhir dan memecahnya menjadi serangkaian sub-tugas. Setelah rencana terbentuk, agent masuk ke tahap eksekusi dan menjalankan setiap langkah sesuai urutan yang telah disusun.

Pola ini cocok untuk tugas yang memiliki struktur jelas dan dapat diprediksi, seperti pembuatan laporan, pemrosesan data batch, atau alur kerja bisnis yang sudah memiliki SOP.

Kelebihan utama dari pendekatan ini adalah keteraturan dan kemudahan dalam pemantauan. Namun, pola ini cenderung kurang fleksibel jika terjadi perubahan kondisi di tengah proses.

#### Self-Correcting Agent

Self-correcting agent adalah pola di mana agent memiliki mekanisme untuk mengevaluasi hasil kerjanya sendiri dan melakukan perbaikan jika ditemukan kesalahan atau ketidaksesuaian dengan tujuan.

![Self-Correcting Agent](https://assets.cdn.dicoding.com/original/academy/dos-85ae0ef816e453f71e4fd0ed9ac0b25c20260429133059.png)

Dalam pendekatan ini, agent biasanya dilengkapi dengan kriteria evaluasi atau metrik tertentu. Setelah menghasilkan sebuah hasil, agent akan meninjau kembali output tersebut dan membandingkannya dengan kriteria yang telah ditetapkan.

Jika hasilnya belum memenuhi standar, agent akan mencoba strategi atau langkah alternatif. Pola ini membantu meningkatkan keandalan sistem, terutama dalam tugas yang membutuhkan tingkat akurasi tinggi.

Sebagai contoh, dalam sistem pembuatan kode, agent dapat menulis sebuah program, menjalankannya, lalu memperbaiki kesalahan yang muncul berdasarkan pesan error.

### Tipe Agent

Selain dari alur kerja, AI Agent juga dapat dibedakan berdasarkan bentuk dan cara mereka berinteraksi dengan sistem lain. Tipe agent ini biasanya ditentukan oleh format output dan cara mereka memanggil atau menggunakan tools.

#### JSON Agent

JSON Agent adalah tipe agent yang diinstruksikan untuk selalu menghasilkan output dalam format JSON terstruktur. Format ini memudahkan sistem lain untuk membaca dan memproses hasil dari agent secara otomatis.

Dalam praktiknya, JSON Agent sering digunakan dalam sistem yang membutuhkan integrasi antar komponen, seperti pipeline data atau aplikasi berbasis mikroservis. Dengan output yang terstruktur, informasi yang dihasilkan oleh agent dapat langsung digunakan sebagai input untuk proses berikutnya.

![JSON Agent](https://assets.cdn.dicoding.com/original/academy/dos-45631e4837edabe7ec641f2290cdab1f20260429133115.png)

Bagaimana cara kerjanya? Agent ini bekerja dengan menyisipkan instruksi format ke dalam *system prompt*. Ia harus mampu menerjemahkan penalaran internalnya ke dalam struktur kunci (*key*) dan nilai (*value*).

Lalu, apa bedanya dengan Instructor/Pydantic? Instructor/Pydantic adalah pustaka penunjang (pustaka validasi). Ia memaksa LLM untuk mengikuti skema dan melakukan *parsing* otomatis.

JSON Agent adalah identitas sistemnya. Anda bisa membangun JSON Agent *tanpa* Instructor, hanya pakai prompt manual, namun menggunakan Instructor membuat JSON Agent Anda jauh lebih stabil dan tahan terhadap error format.

Sebagai contoh, agent dapat menghasilkan JSON yang berisi daftar tugas, parameter konfigurasi, atau hasil analisis yang kemudian diproses oleh sistem lain tanpa perlu pemrosesan tambahan. Misalnya, Agent ekstraksi data yang membaca email masuk dan mengubahnya menjadi JSON berisi { "pengirim": "...", "prioritas": "...", "ringkasan": "..." } untuk dimasukkan ke database perusahaan.

#### Code Agent

Code Agent adalah tipe agent yang berfokus pada pembuatan dan eksekusi kode sebagai bagian dari proses penyelesaian tugas. Agent ini tidak hanya menghasilkan potongan kode, tetapi juga dapat menjalankannya dalam lingkungan tertentu.

Bagaimana ia bekerja? Agent ini diberikan akses ke sebuah *sandbox* atau lingkungan eksekusi kode (*runtime*). Jika ia diberikan soal matematika sulit atau pengolahan data besar, ia tidak akan menebak hasilnya dengan teks, melainkan menulis skrip, menjalankannya, dan mengambil hasil dari *standard output* kode tersebut.

![Code Agent](https://assets.cdn.dicoding.com/original/academy/dos-b674ac1147cd5f1576af0cc20b6c591920260429133128.png)

Tipe ini sangat berguna dalam konteks pengembangan perangkat lunak, analisis data, dan otomatisasi sistem. Agent dapat menulis skrip untuk mengolah data, menjalankan pengujian, atau mengonfigurasi sistem secara otomatis.

Sebagai contoh, dalam analisis data, Code Agent dapat membuat skrip untuk memuat dataset, membersihkan data, dan menghasilkan visualisasi, lalu menyajikan hasilnya kepada pengguna. Saat data analyst bertanya "Apa tren penjualan bulan lalu?", Agent akan menulis kode pandas, memuat file CSV, membuat grafik, dan memberikan kesimpulan berdasarkan hasil eksekusi kode tersebut.

#### Function-Calling Agent

Function-calling Agent adalah tipe agent yang dirancang untuk menjadi jembatan antara kemampuan penalaran LLM dengan fungsi-fungsi teknis yang telah didefinisikan dalam sistem. Ini adalah salah satu tipe agent yang paling efisien karena model AI (seperti GPT-4 atau Gemini) saat ini sudah dilatih secara khusus untuk mengenali kapan ia harus berhenti menghasilkan teks dan mulai memanggil alat bantu.

Bagaimana cara kerjanya? Ketika menerima perintah, agent tidak langsung memberikan jawaban. Ia akan menentukan fungsi mana yang paling relevan, menyusun parameter yang sesuai, dan mengembalikan instruksi panggilan tersebut ke sistem. Sistem kemudian menjalankan fungsi tersebut dan memberikan hasilnya kembali ke agent untuk diproses lebih lanjut.

Dalam praktiknya, Anda mungkin akan mendengar istilah *Tool Calling* atau *MCP*. Penting untuk memahami perbedaannya agar Anda tidak salah dalam memilih arsitektur.

*   **Function Calling**
    Ini adalah kemampuan dasar dari model itu sendiri. Model dilatih untuk menghasilkan *output* terstruktur seperti nama fungsi dan argumennya alih-alih teks naratif.
*   **Tool Calling**
    Ini adalah istilah yang sering digunakan oleh framework seperti LangChain atau OpenAI SDK. Satu tool bisa berisi satu atau lebih fungsi. Tool calling biasanya membungkus *Function Calling* dengan fitur tambahan seperti *error handling* dan *state management*.
*   **MCP (Model Context Protocol)**
    Ini adalah standar yang diperkenalkan oleh Anthropic untuk menstandarisasi bagaimana agent terhubung dengan sumber data dan *tools*. Jika *Function Calling* adalah bahasa yang digunakan, maka MCP adalah kabel USB-C universal. Dengan MCP, Anda tidak perlu menulis ulang kode pemanggilan fungsi untuk setiap model yang berbeda (Gemini, Claude, atau GPT); cukup gunakan satu server MCP yang bisa digunakan oleh berbagai model sekaligus.

![Function Calling Agent](https://assets.cdn.dicoding.com/original/academy/dos-2e14b95d3bb6213fba16b24a0fe9639a20260326103627.png)

Tipe ini sangat cocok untuk membangun sistem yang modular, di mana berbagai kemampuan disediakan dalam bentuk fungsi terpisah yang dapat dipanggil sesuai kebutuhan.

Ini adalah tipe yang paling modern dan efisien, di mana model AI (seperti GPT-4 atau Gemini) sudah dilatih secara khusus untuk mengenali kapan ia harus memanggil fungsi eksternal. Contoh Penggunaannya seperti pada asisten reservasi hotel. Agent mengenali bahwa ia perlu memanggil fungsi check_availability() ke database hotel sebelum ia bisa memberikan jawaban kepada pengguna.

Workflow dan tipe agent yang telah dijelaskan di atas tentu saja tidak berdiri sendiri. Keduanya saling melengkapi dalam membentuk perilaku dan kemampuan sebuah AI Agent. Sebagai contoh, sebuah Function-calling Agent dapat menggunakan pola ReAct untuk memutuskan kapan dan bagaimana sebuah fungsi harus dipanggil. Sebuah Code Agent dapat menggunakan pola Self-correcting untuk memperbaiki kode yang dihasilkan jika terjadi kesalahan saat eksekusi.

Dengan memahami hubungan ini, Anda dapat merancang agent yang tidak hanya sesuai secara teknis, tetapi juga efektif dalam menyelesaikan tugas yang dihadapi.

Sampai di sini, kita telah mempelajari variasi AI Agent dari sisi workflow dan tipe implementasi.

> **Latihan Fill in the Blank:**
> Dari sisi workflow, kita mengenal pola **Chain of Thought**, **ReAct Framework**, **Plan and Execute Pattern**, dan **Self-Correcting Agent**, yang masing-masing menawarkan pendekatan berbeda dalam proses penalaran dan tindakan.
> 
> Dari sisi tipe agent, kita membahas **JSON Agent**, **Code Agent**, dan **Function-Calling Agent**, yang dibedakan berdasarkan cara mereka menghasilkan output dan berinteraksi dengan sistem lain.

Pada materi selanjutnya, kita akan memastikan agent bekerja dengan baik melalui berbagai metode evaluasi agent.

---

## Agent Evaluation

Setelah memahami proses merancang dan menjalankan AI Agent, langkah berikutnya adalah memastikan bahwa agent tersebut benar-benar bekerja dengan baik di dunia nyata. Di sinilah proses evaluasi berperan penting. Evaluasi membantu kita menjawab pertanyaan sederhana, tetapi krusial, yaitu apakah agent sudah menjalankan tugasnya secara efektif, efisien, dan dapat diandalkan.

Dalam materi ini, kita akan membahas dua pendekatan utama dalam mengevaluasi AI Agent, yaitu evaluasi berbasis performa dan evaluasi berbasis metrik. Keduanya saling melengkapi dan memberikan sudut pandang yang berbeda terhadap kualitas sebuah agent.

### Performance

Evaluasi berbasis performa berfokus pada bagaimana agent bekerja dalam praktiknya. Pendekatan ini melihat hasil nyata dari penggunaan agent, bukan hanya kualitas teks atau prediksi yang dihasilkan. Dengan cara ini, kita dapat menilai apakah agent benar-benar memberikan nilai bagi pengguna dan sistem.

#### Task Completion Rate

Task completion rate adalah metrik yang digunakan untuk mengukur tingkat keberhasilan AI Agent dalam menyelesaikan tugas yang diberikan oleh pengguna dari awal hingga akhir. Metrik ini tidak hanya melihat pemberian respons oleh Agent, tetapi juga kemampuan respons tersebut dalam memenuhi tujuan yang diminta pengguna sesuai dengan kriteria yang telah ditentukan.

Dalam praktiknya, sebuah tugas dianggap selesai jika agent berhasil mencapai outcome yang diharapkan. Misalnya, pengguna meminta agent untuk "mencari tiga artikel terbaru tentang AI Agent dan merangkumnya dalam satu paragraf." Tugas tersebut dinyatakan berhasil jika agent benar-benar menemukan artikel yang relevan, memastikan informasinya aktual, dan menyajikan ringkasan yang sesuai dengan instruksi. Namun, jika salah satu langkah tersebut tidak terpenuhi, tugas tersebut dihitung sebagai gagal.

Perhitungan task completion rate biasanya dinyatakan dalam bentuk persentase. Rumus sederhananya adalah jumlah tugas yang berhasil diselesaikan dibagi dengan total tugas yang diberikan, kemudian dikalikan seratus persen. Dengan cara ini, pengembang dapat dengan mudah melihat seberapa andal agent dalam menangani berbagai jenis permintaan pengguna.

![Task Completion Rate](https://assets.cdn.dicoding.com/original/academy/dos-b7770b420f5e0103760a24934384619c20260429133236.png)

Metrik ini sangat bergantung pada definisi kriteria keberhasilan. Dalam sistem yang sederhana, kriteria bisa berupa “apakah jawaban diberikan atau tidak.” Namun, dalam sistem yang lebih kompleks, seperti agent untuk layanan pelanggan atau otomatisasi bisnis, kriteria keberhasilan bisa mencakup beberapa aspek sekaligus, seperti ketepatan informasi, kelengkapan langkah, kepatuhan terhadap aturan bisnis, dan waktu penyelesaian.

Task completion rate juga dapat dianalisis berdasarkan kategori tugas. Misalnya, pengembang dapat memisahkan antara tugas pencarian informasi, tugas pemrosesan data, dan tugas yang melibatkan eksekusi API. Dengan pendekatan ini, akan terlihat area mana yang sudah kuat dan area mana yang masih perlu ditingkatkan. Agent mungkin memiliki tingkat keberhasilan tinggi dalam menjawab pertanyaan teks, tetapi rendah dalam mengeksekusi alur kerja yang melibatkan banyak langkah dan sistem eksternal.

Dalam konteks dunia nyata, metrik ini memiliki dampak langsung terhadap pengalaman pengguna dan efisiensi operasional. Agent dengan task completion rate tinggi cenderung mengurangi beban kerja manusia, mempercepat proses, dan meningkatkan kepuasan pengguna. Sebaliknya, jika nilainya rendah, pengguna akan sering mengulang permintaan atau harus mengambil alih tugas secara manual, yang justru mengurangi nilai dari penggunaan AI Agent itu sendiri.

Oleh karena itu, task completion rate sering dijadikan metrik utama dalam proses evaluasi dan iterasi pengembangan agent. Dengan memantau perubahan nilainya dari waktu ke waktu, tim pengembang dapat menilai apakah pembaruan pada prompt, penambahan tools, atau perubahan arsitektur agent benar-benar meningkatkan kemampuan agent dalam menyelesaikan tugas secara end-to-end.

#### Latency

Latency adalah metrik yang digunakan untuk mengukur seberapa cepat AI Agent merespons permintaan pengguna atau menyelesaikan sebuah tugas dari sudut pandang waktu. Metrik ini biasanya dihitung sejak pengguna mengirimkan permintaan hingga agent memberikan respons akhir atau mengeksekusi tindakan yang diminta. Dengan kata lain, latency menggambarkan jeda waktu yang dirasakan langsung oleh pengguna saat berinteraksi dengan agent.

Dalam sistem yang sederhana, latency mungkin hanya terdiri dari waktu yang dibutuhkan model bahasa untuk memproses input dan menghasilkan output. Namun, pada AI Agent yang lebih kompleks, latency sering kali merupakan akumulasi dari beberapa tahapan. Misalnya, agent mungkin perlu memanggil layanan web search untuk mendapatkan informasi terbaru, mengambil data dari sistem internal melalui API, lalu memproses hasil tersebut dengan LLM sebelum menyajikannya kembali kepada pengguna. Setiap langkah ini menambah waktu tunggu secara keseluruhan.

Saat agent berinteraksi dengan LLM, ada tiga sub-metrik utama yang menentukan kualitas pengalaman pengguna.

*   **TTFT (Time to First Token)**
    Waktu yang dibutuhkan sejak pengguna mengirimkan permintaan hingga karakter pertama muncul di layar. Ini adalah indikator utama responsivitas.
*   **Token Generation Rate**
    Waktu rata-rata yang dibutuhkan untuk menghasilkan setiap token berikutnya. Metrik ini menentukan seberapa lancar aliran teks yang dihasilkan.
*   **TTLX (Time to Last Token)**
    Waktu total sejak input diterima hingga seluruh respons selesai dihasilkan.

Karena agent bekerja dengan siklus berpikir dan bertindak, terdapat latency tersembunyi juga di luar inferensi model.

*   **Reasoning/Planning Latency**
    Waktu yang dihabiskan agent pada tahap *thought* untuk menganalisis konteks dan merencanakan langkah sebelum mulai bertindak.
*   **Retrieval Latency**
    Waktu yang dibutuhkan sistem untuk mencari data pendukung dari *vector database* atau sumber internal (RAG).
*   **Tool Execution Latency**
    Waktu tunggu saat agent memanggil layanan eksternal (API pihak ketiga, eksekusi kode, atau *web search*). Ini sering menjadi variabel yang paling sulit dikontrol karena bergantung pada sistem di luar kendali kita.
*   **Agent Coordination Latency**
    Dalam sistem *Multi-Agent*, metrik ini mengukur waktu yang dihabiskan agen untuk berkomunikasi dan saling bertukar informasi sebelum mencapai kesimpulan akhir.

Untuk mempermudah pemahaman, Anda hanya perlu mengingat dua jenis utama latency secara umum, yaitu:

> **Flashcard Latency:**
> *   **Perceived Latency:** Waktu yang dirasakan oleh pengguna sampai mereka melihat respons pertama, seperti pesan “sedang memproses” atau token pertama (TTFT).
> *   **Total Latency:** Waktu hingga seluruh tugas benar-benar selesai, termasuk semua langkah di balik layar.

Dalam banyak aplikasi, perceived latency yang rendah sering kali sudah cukup untuk menjaga pengalaman pengguna tetap positif, meskipun total latency masih relatif tinggi.

![Latency Diagram](https://assets.cdn.dicoding.com/original/academy/dos-51646ad1a065d5f0b51f67da10f06a3d20260429133259.png)

Faktor-faktor yang memengaruhi latency sangat beragam. Dari sisi model, ukuran dan kompleksitas LLM berpengaruh besar terhadap waktu inferensi. Model yang lebih besar dan lebih canggih biasanya membutuhkan waktu lebih lama untuk memproses input. Dari sisi infrastruktur, kualitas jaringan, lokasi server, dan kapasitas komputasi juga menentukan seberapa cepat permintaan dapat diproses. Selain itu, penggunaan tools eksternal, seperti API pihak ketiga atau sistem basis data, sering menjadi sumber utama keterlambatan karena agent harus menunggu respons dari sistem lain di luar kendalinya.

Dalam proses evaluasi, latency tidak hanya diukur sebagai satu angka rata-rata. Pengembang tidak boleh hanya mengandalkan angka rata-rata (*mean*). Kita perlu melihat distribusi waktu respons untuk memahami pengalaman seluruh segmen pengguna:

*   **P50 (Median)**
    Menggambarkan pengalaman rata-rata pengguna. Setengah dari pengguna mengalami pengalaman yang lebih cepat, dan setengah lainnya lebih lambat.
*   **P95 dan P99 (Tail Latencies)**
    Menggambarkan waktu respons bagi 5% atau 1% pengguna yang mengalami keterlambatan ekstrem. Jika P99 sistem Anda adalah 10 detik, artinya 1 dari 100 pengguna mengalami kegagalan kenyamanan yang fatal.

Dari sudut pandang desain sistem, latency juga memengaruhi bagaimana alur interaksi agent dirancang. Untuk tugas-tugas yang membutuhkan waktu lama, agent dapat dibekali dengan mekanisme umpan balik sementara, seperti pemberitahuan bahwa proses sedang berlangsung atau pembagian hasil dalam beberapa tahap. Dengan cara ini, pengguna tetap merasa dilibatkan dan tidak menganggap sistem berhenti bekerja.

Pada akhirnya, tujuan utama dari memantau dan mengoptimalkan latency adalah menjaga keseimbangan antara kecepatan dan kualitas. Respons yang sangat cepat, tetapi tidak akurat akan mengurangi kepercayaan pengguna, sementara respons yang sangat akurat, tetapi terlalu lambat akan menurunkan kenyamanan penggunaan. Dengan memahami sumber-sumber latency dan dampaknya, pengembang dapat merancang AI Agent yang tidak hanya pintar, tetapi juga terasa sigap dan menyenangkan untuk digunakan.

#### Cost

Cost adalah metrik yang digunakan untuk mengukur total biaya yang dikeluarkan untuk mengoperasikan AI Agent dalam jangka waktu tertentu. Berbeda dengan metrik performa seperti task completion rate dan latency yang berfokus pada kualitas dan kecepatan, metrik ini berfokus pada aspek keberlanjutan dan efisiensi dari sisi operasional. Dengan memahami cost, pengembang dan pemangku kepentingan dapat memastikan bahwa agent yang dibangun tidak hanya berfungsi dengan baik, tetapi juga layak secara ekonomi untuk digunakan dalam skala nyata.

![Cost Evaluation](https://assets.cdn.dicoding.com/original/academy/dos-251c2119ad221ba21098eb942a168b4e20260429133312.png)

Sumber biaya dalam AI Agent biasanya berasal dari beberapa komponen utama. Yang paling umum adalah biaya penggunaan model generative AI, terutama jika menggunakan layanan berbasis API yang mengenakan tarif per jumlah token atau per permintaan. Biaya yang harus dikeluarkan akan semakin besar seiring dengan bertambahnya panjang percakapan, kompleksitas penalaran, dan frekuensi pemanggilan agent. Selain itu, pemanggilan tools eksternal seperti web search berbayar, layanan pemrosesan gambar, atau API pihak ketiga juga dapat menambah biaya secara signifikan, terutama jika agent melakukan banyak langkah dalam satu alur kerja.

Di luar layanan berbasis API, cost juga mencakup penggunaan sumber daya komputasi. Jika agent dijalankan di infrastruktur sendiri, biaya ini bisa berupa penggunaan CPU, GPU, memori, dan penyimpanan. Pada sistem berbasis cloud, hal ini sering diterjemahkan menjadi biaya per jam server, biaya transfer data, dan biaya penyimpanan log atau memori jangka panjang. Meskipun biaya ini tidak selalu terlihat secara langsung dalam satu kali interaksi, dalam penggunaan jangka panjang dan skala besar, akumulasinya bisa menjadi sangat signifikan.

Dalam proses evaluasi, cost sering dianalisis dalam bentuk metrik turunan, seperti biaya per tugas yang berhasil diselesaikan atau biaya per pengguna aktif. Pendekatan ini membantu tim memahami seberapa efisien agent dalam menghasilkan nilai. Misalnya, dua agent mungkin memiliki task completion rate yang sama. Namun, jika salah satunya membutuhkan biaya dua kali lipat untuk setiap tugas yang diselesaikan, agent tersebut dianggap kurang efisien dari sudut pandang bisnis.

Pengelolaan cost juga berkaitan erat dengan desain arsitektur agent. Pengembang dapat mengurangi biaya dengan berbagai strategi, seperti memilih model yang lebih ringan untuk tugas-tugas sederhana, membatasi panjang konteks yang dikirim ke LLM, atau menerapkan mekanisme caching untuk menyimpan hasil yang sering digunakan agar tidak perlu memanggil API berulang kali. Selain itu, agent dapat dirancang untuk lebih selektif dalam menggunakan tools, hanya memanggil layanan eksternal ketika benar-benar diperlukan.

Pada akhirnya, metrik cost membantu memastikan bahwa AI Agent dapat dioperasikan secara berkelanjutan. Dalam lingkungan bisnis, keputusan untuk mengadopsi atau memperluas penggunaan agent sering kali tidak hanya didasarkan pada seberapa pintar atau cepat agent tersebut. Faktor kelayakan biaya yang harus dikeluarkan dibandingkan dengan manfaat yang dihasilkan juga menjadi pertimbangan penting. Dengan memantau dan mengoptimalkan cost secara terus-menerus, pengembang dapat membangun agent yang tidak hanya efektif secara teknis, tetapi juga efisien secara ekonomi.

### Metric Based

Selain melihat performa secara langsung, kita juga dapat mengevaluasi agent menggunakan metrik yang lebih terstruktur dan terukur. Pendekatan ini biasanya digunakan untuk menilai kualitas keluaran agent, terutama dalam konteks teks atau konten yang dihasilkan.

#### LLM as a Judge

LLM as a Judge adalah pendekatan evaluasi di mana model bahasa tidak hanya digunakan sebagai “pelaku” yang menghasilkan respons, tetapi juga sebagai “penilai” yang mengevaluasi kualitas hasil dari AI Agent. Dalam peran ini, LLM diberi instruksi khusus untuk bertindak sebagai evaluator. LLM menilai apakah sebuah output sudah memenuhi kriteria tertentu, seperti relevansi terhadap pertanyaan, akurasi informasi, kelengkapan jawaban, serta kesesuaian dengan format atau aturan yang diminta pengguna.

Pendekatan ini biasanya diterapkan dengan cara menyusun sebuah prompt evaluasi. Prompt tersebut berisi konteks tugas, output yang dihasilkan oleh agent, dan daftar kriteria penilaian. LLM kemudian diminta untuk memberikan skor, label, atau penilaian kualitatif, misalnya dalam bentuk skala angka, kategori seperti “baik” atau “perlu perbaikan”, atau ringkasan alasan di balik penilaiannya. Dengan cara ini, proses evaluasi dapat dilakukan secara otomatis dan konsisten untuk jumlah data yang besar.

![LLM as a Judge](https://assets.cdn.dicoding.com/original/academy/dos-510a0457065c4a6b4dea50a6bee0c5eb20260429133336.png)

Salah satu kekuatan utama dari LLM as a Judge adalah fleksibilitasnya. Berbeda dengan metrik tradisional yang biasanya kaku dan hanya cocok untuk jenis tugas tertentu, LLM dapat menilai berbagai aspek yang bersifat subjektif atau kontekstual. Misalnya, dalam tugas layanan pelanggan, LLM dapat mengevaluasi apakah nada jawaban agent sudah sopan dan empatik. Dalam tugas penulisan, LLM dapat menilai apakah struktur teks sudah logis dan mudah dipahami. Hal-hal seperti ini sulit diukur hanya dengan angka atau perbandingan teks sederhana.

Pendekatan ini juga sangat membantu dalam skenario skala besar. Ketika sebuah sistem AI Agent digunakan oleh ribuan atau jutaan pengguna, melakukan evaluasi manual terhadap setiap respons menjadi tidak realistis. Dengan LLM sebagai penilai, pengembang dapat melakukan audit kualitas secara berkala, memantau tren performa, dan mendeteksi penurunan kualitas secara otomatis.

Namun, LLM as a Judge juga memiliki keterbatasan. Karena penilaiannya dilakukan oleh model, hasil evaluasi bisa dipengaruhi oleh bias yang ada dalam data pelatihan atau oleh cara prompt disusun. Dua prompt evaluasi yang sedikit berbeda dapat menghasilkan penilaian yang berbeda pula, meskipun output agent yang dinilai sama. Selain itu, LLM tidak selalu dapat memverifikasi kebenaran fakta secara absolut, terutama jika informasi yang dinilai bersifat spesifik atau membutuhkan sumber eksternal.

Untuk mengatasi hal ini, praktik terbaiknya adalah mengombinasikan LLM as a Judge dengan metode evaluasi lain. Misalnya, hasil penilaian dari LLM dapat dibandingkan dengan metrik berbasis aturan, data ground truth, atau sampel evaluasi manual oleh manusia. Dengan pendekatan ini, pengembang dapat memperoleh gambaran yang lebih seimbang antara penilaian otomatis dan validasi nyata.

Dalam konteks pengembangan berkelanjutan, LLM as a Judge sering digunakan sebagai alat bantu iterasi. Setiap kali prompt agent diperbarui, tools ditambahkan, atau arsitektur diubah, hasil baru dapat dievaluasi secara otomatis oleh LLM untuk melihat apakah kualitas meningkat atau justru menurun. Dengan demikian, LLM tidak hanya berperan sebagai mesin yang menghasilkan jawaban, tetapi juga sebagai bagian dari sistem umpan balik yang membantu meningkatkan kualitas AI Agent dari waktu ke waktu.

#### BLEU dan ROUGE

BLEU dan ROUGE adalah dua metrik berbasis teks yang digunakan untuk mengukur kualitas hasil sistem bahasa, termasuk AI Agent. Pengukuran dilakukan dengan cara membandingkan hasil AI dengan satu atau lebih teks referensi yang dianggap sebagai jawaban ideal. Kedua metrik ini banyak digunakan dalam bidang NLP karena mampu memberikan ukuran kuantitatif yang relatif objektif terhadap kesamaan antara hasil sistem dan standar yang telah ditentukan.

BLEU yang merupakan singkatan dari Bilingual Evaluation Understudy, pada dasarnya mengukur seberapa besar tumpang tindih antara teks yang dihasilkan oleh agent dan teks referensi. Pendekatan yang digunakan adalah n-gram matching, yaitu membandingkan potongan-potongan kata berurutan, seperti satu kata (unigram), dua kata (bigram), atau lebih.

![BLEU Metric](https://assets.cdn.dicoding.com/original/academy/dos-101804d96c7be4470d2f66e613e5790120260429133357.png)

Semakin banyak n-gram yang cocok antara output agent dan referensi, semakin tinggi skor BLEU yang dihasilkan. Untuk mencegah sistem menghasilkan teks yang terlalu pendek hanya demi meningkatkan kecocokan, BLEU juga menggunakan mekanisme penalti panjang, sehingga hasil yang terlalu singkat akan mendapatkan skor lebih rendah meskipun banyak kata yang cocok.

Dalam praktiknya, BLEU sering digunakan pada tugas-tugas seperti penerjemahan mesin atau pembuatan jawaban yang memiliki struktur relatif tetap. Misalnya, ketika menerjemahkan kalimat dari satu bahasa ke bahasa lain, terdapat banyak kemungkinan susunan kata yang benar, tetapi teks referensi tetap dapat dijadikan acuan untuk mengukur seberapa dekat hasil terjemahan tersebut dengan contoh yang dianggap ideal. Skor BLEU yang tinggi menunjukkan bahwa secara struktur dan pilihan kata, output agent mendekati teks referensi.

ROUGE, yang merupakan singkatan dari Recall-Oriented Understudy for Gisting Evaluation, memiliki fokus yang sedikit berbeda. Jika BLEU lebih menekankan pada presisi atau kecocokan frasa yang dihasilkan, ROUGE lebih menekankan pada seberapa banyak informasi penting dari teks referensi yang berhasil “ditangkap” oleh hasil agent. Dengan kata lain, ROUGE mengukur tingkat cakupan atau recall dari output agent terhadap referensi. Pendekatan ini sangat populer dalam tugas peringkasan teks, di mana tujuan utama bukan hanya menghasilkan kalimat yang mirip, tetapi memastikan bahwa poin-poin utama dari teks asli tetap muncul dalam ringkasan.

![ROUGE Metric](https://assets.cdn.dicoding.com/original/academy/dos-c037b1bacd03ab60738deb4d9360e1af20260429133414.png)

ROUGE memiliki beberapa varian yang umum digunakan. ROUGE-N mengukur kecocokan n-gram antara ringkasan dan referensi, mirip dengan BLEU tetapi dengan fokus pada recall.

ROUGE-L mengukur kesamaan berdasarkan urutan kata terpanjang yang sama, sehingga lebih memperhatikan struktur kalimat secara keseluruhan. Dengan berbagai varian ini, pengembang dapat memilih sudut pandang evaluasi yang paling sesuai dengan jenis tugas yang sedang dikerjakan.

Meskipun BLEU dan ROUGE memberikan angka yang jelas dan mudah dibandingkan, namun metrik ini kurang cocok untuk AI Agent.

Mengapa begitu? Dalam konteks AI Agent, kemiripan kata (leksikal) sering kali tidak berkorelasi dengan keberhasilan tugas. Inilah alasan mengapa BLEU dan ROUGE dianggap sebagai metrik sekunder bagi Agent:

*   Seorang Agent bisa saja menyelesaikan masalah dengan cara yang benar namun menggunakan gaya bahasa yang berbeda total dari teks referensi. Dalam hal ini, skor BLEU akan rendah meski Agent berhasil (misal: memanggil API yang benar).
*   BLEU dan ROUGE hanya menilai teks akhir. Mereka tidak bisa menilai apakah Agent melakukan langkah pemanggilan *tool* yang benar, apakah urutan *reasoning*-nya logis, atau apakah ia mengakses basis data yang tepat.
*   Metrik ini bekerja berdasarkan kecocokan kata secara harfiah. Jika Agent menggunakan kata "memulai" sementara referensi menggunakan "mengawali", skor akan tetap turun meskipun maknanya sama.

Meski memiliki keterbatasan, BLEU dan ROUGE tetap bisa digunakan oleh pengembang Agent dalam skenario tertentu, seperti:

*   **Agent Peringkas/Penulis**
    Jika tugas utama Agent adalah memproses dokumen menjadi teks (misalnya laporan berita atau ringkasan rapat).
*   **Benchmarking Dasar**
    Digunakan sebagai filter awal untuk memastikan output Agent tidak melantur terlalu jauh dari format standar yang diharapkan.

Oleh karena itu, dalam evaluasi AI Agent yang kompleks, BLEU dan ROUGE biasanya digunakan sebagai alat bantu, bukan satu-satunya penentu kualitas. Skor dari metrik ini sering dikombinasikan dengan metrik performa lain, seperti task completion rate dan latency, serta dengan penilaian kualitatif, baik melalui evaluasi manual oleh manusia maupun pendekatan seperti LLM as a Judge.

Dengan kombinasi ini, Anda dapat memperoleh gambaran yang lebih utuh tentang seberapa baik agent bekerja, baik dari sisi angka maupun dari sudut pandang pengalaman pengguna.

Dengan memahami berbagai pendekatan evaluasi ini, Anda dapat merancang AI Agent yang siap digunakan dalam lingkungan nyata.

---

## Latihan: Membangun AI Agent Sederhana dengan Framework ReAct

Pada modul ini, kita tidak hanya sekadar menjalankan kode, tetapi juga berperan sebagai AI engineer di dunia industri yang sedang membangun sebuah AI Research Agent sederhana.

Agent yang kita buat di latihan ini memiliki satu peran utama, yaitu mencari tren terbaru tentang topik yang diminta pengguna dan mengemasnya dalam format data yang rapi dan konsisten. Output dari agent ini tentu saja tidak akan berhenti di sini. Pada latihan selanjutnya, hasil data dari latihan ini akan menjadi bahan baku bagi agent lain untuk menyusun artikel secara utuh dengan narasi yang bisa dibaca.

Dengan kata lain, Anda sedang membangun tahap pertama dari sebuah pipeline Agentic AI dari data mentah di internet, menjadi informasi terstruktur, lalu diolah lagi menjadi konten.

Di sinilah pendekatan ReAct (Reasoning + Acting) yang telah kita pelajari menjadi sangat relevan. Kita akan melihat secara langsung bagaimana agent:

*   Berpikir (Thought): Melakukan penalaran mandiri untuk merencanakan apa yang perlu dicari.
*   Bertindak (Action): Menggunakan tools eksternal untuk mencari data.
*   Mengamati (Observation): Menilai hasil dari dunia nyata.
*   Sintesis terstruktur: Menyatukan data menjadi output yang terjamin formatnya.

Meskipun anatomi ideal sebuah AI Agent mencakup LLM (Brain), Tools, Memory, pada latihan pertama ini kita membangun sebuah Stateless Agent. Artinya, agent ini difokuskan pada kecepatan eksekusi tugas tunggal tanpa harus mengingat riwayat percakapan panjang. Komponen Memory bersifat fleksibel dan baru akan kita eksplorasi pada sistem yang lebih kompleks nanti.

Sebelum masuk ke kode, mari kita lihat dulu cerita di balik sistem yang akan dibangun. Agent yang kita buat memiliki peran sebagai AI Research Assistant. Coba Anda urutkan alur kerja yang akan kita lakukan.

> **Latihan Mengurutkan Alur Kerja:**
> 1. Menerima topik dari pengguna.
> 2. Menalar informasi apa yang dibutuhkan.
> 3. Mengakses internet untuk mencari data terbaru.
> 4. Mengevaluasi hasil pencarian.
> 5. Mengemas semua itu dalam format data yang terstruktur.

Kalau kita gambarkan sebagai alur kerja maka akan terlihat seperti ini.

![Alur Kerja Agent ReAct](https://assets.cdn.dicoding.com/original/academy/dos-9c1dc7bb220bb5f5a80a16ba8289322020260429133516.png)

Anda dapat mengakses kode lengkapnya pada [notebook](https://colab.research.google.com/drive/13cKFU8N6WUQHlcVyu6lVCa3mCmY9oFhm?usp=sharing) ini dan menjalankannya tahap demi tahap. Mari kita bedah kodenya!

### Instalasi Library: Menyiapkan Lingkungan Kerja Agent

```python
!pip install -q -U langchain-groq langchain-community duckduckgo-search pydantic ddgs
```

Pada tahap ini, kita sedang menyiapkan lingkungan kerja teknis agar agent bisa menjalankan dua hal penting, yaitu berpikir dan berinteraksi dengan dunia luar.

```python
import os
from google.colab import userdata
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from pydantic import BaseModel, Field
from typing import List
```

Setiap library yang dipasang punya peran spesifik dalam alur industri, *loh*.

*   **langchain-groq**: Menyediakan antarmuka untuk berkomunikasi dengan Brain dari agent yaitu LLM. Kita menggunakanLlama 3.3 melalui Groq API.
*   **Duckduckgo-search**: Ini adalah Tool atau tangan agent untuk mengambil informasi terbaru dari internet secara real-time.
*   **pydantic**. Alat validasi untuk menjaga agar output agent selalu mengikuti format data yang sudah disepakati secara strict.

Kalau kita lihat dari sudut pandang arsitektur sistem, bagian ini adalah fondasi. Tanpa ini, agent hanya akan menjadi komponen pasif yang bisa menghasilkan teks, tetapi tidak bisa mengumpulkan data atau menjamin kualitas outputnya.

### Konfigurasi: Menghubungkan Agent ke Layanan LLM

```python
# SETUP & KONFIGURASI
os.environ["GROQ_API_KEY"] = userdata.get('GROQ_API_KEY')
```

Baris ini menghubungkan agent kita dengan layanan LLM yang berjalan di luar sistem lokal (cloud). Dalam praktik industri, koneksi ke layanan eksternal selalu dikelola lewat environment variable atau secret manager, bukan ditulis langsung di dalam kode. Tujuannya sederhana untuk:

*   menjaga keamanan kredensial,
*   memudahkan rotasi API key, dan
*   menghindari kebocoran saat kode dibagikan atau diunggah ke repositori.

Dengan konfigurasi ini, agent sudah siap mengirim prompt ke model bahasa dan menerima respons secara real-time.

Lalu, jika Anda bertanya-tanya bagaimana cara mendapatkan API Key Groq, berikut adalah tutorialnya.

Pertama-tama, agar agent bisa terhubung ke layanan LLM Groq, Anda perlu membuat akun dan menghasilkan API Key pribadi. Prosesnya cukup sederhana dan biasanya hanya memakan waktu beberapa menit.

1.  **Buka Website Groq**
    Masuk ke halaman resmi Groq di: [https://console.groq.com](https://console.groq.com)
    Di halaman ini, Anda akan melihat dashboard untuk mengelola project dan API key.
    *   Jika belum memiliki akun, klik Sign Up dan daftar menggunakan email atau akun Google.
    *   Jika sudah punya akun, langsung klik Login.
    Setelah berhasil masuk, Anda akan diarahkan ke halaman dashboard.
2.  **Buat API Key Baru**
    Dalam dashboard:
    *   Cari menu API Keys atau Settings.
    *   Klik tombol Create API Key.
    *   Beri nama key, misalnya “colab-research-agent” agar mudah dikenali.
    Setelah itu, Groq akan menampilkan sebuah string panjang. Inilah API Key Anda. Catatan Penting! API Key ini hanya ditampilkan sekali. Simpan di tempat aman dan jangan dibagikan ke orang lain.
3.  **Simpan API Key di Google Colab**
    Agar sesuai dengan praktik industri, kita tidak menuliskan API key langsung di kode. Di Google Colab, Anda bisa menyimpannya sebagai secret:
    *   Klik ikon 'Key' Secrets di sidebar kiri Colab.
    *   Klik Add a new secret.
    *   Isi:
        *   Name: GROQ_API_KEY
        *   Value: (paste API key dari Groq)
    *   Klik Save.
    Setelah ini, baris kode sebelumnya akan otomatis mengambil key tersebut.

Jika konfigurasi sudah benar, saat Anda menjalankan agent, tidak akan muncul error terkait autentikasi, dan model akan merespons prompt seperti biasa.

Penting untuk diingat juga, Groq menyediakan akses gratis (*Free Tier*) dengan performa yang sangat cepat. Namun, terdapat batasan jumlah permintaan (*Rate Limits*). Jika Anda mendapatkan pesan *error* terkait kuota saat menjalankan kode, tunggulah beberapa saat sebelum mencoba lagi.

*Yup*, setelah mengikuti semua langkah-langkah di atas, Anda tidak hanya berhasil menghubungkan agent ke LLM, tetapi juga sudah menerapkan praktik keamanan dan manajemen kredensial ala industri. Ini adalah kebiasaan kecil yang sangat penting ketika sistem AI Anda mulai berkembang dan digunakan oleh banyak orang atau tim.

### Schema Data: Menetapkan Format Output

```python
# DEFINISI SCHEMA (Kontrak Data yang akan dipakai langsung oleh LLM)
class ResearchResult(BaseModel):
    topik: str = Field(description="Topik utama riset")
    prediksi_2026: str = Field(description="Prediksi perkembangan di tahun 2026")
    teknologi_kunci: List[str] = Field(description="Daftar minimal 3 teknologi kunci")
    status_riset: str = Field(description="Status kelengkapan riset (Lengkap/Perlu Tambahan)")
```

Di tahap ini, kita mendefinisikan kontrak data menggunakan class ResearchResult. Alasannya sederhana, agar hasil dari agent ini tidak hanya akan dibaca manusia, tetapi juga akan diproses oleh agent lain di Latihan Membuat Multi-Agent Menggunakan CrewAI untuk disusun menjadi artikel. Kalau formatnya tidak konsisten, seluruh alur setelahnya bisa gagal.

Dengan schema ini, kita memastikan bahwa setiap hasil riset selalu memiliki beberapa hal berikut.

*   Topik utama yang jelas.
*   Ringkasan prediksi sebagai inti informasi.
*   Daftar teknologi kunci sebagai poin-poin penting.
*   Status riset sebagai penanda apakah hasil sudah siap dipakai.

Pendekatan seperti ini umum digunakan dalam sistem produksi agar komponen AI bisa saling terhubung tanpa perlu menebak-nebak format data.

### ReAct Flow: Alur Kerja Agent dari Planning hingga Sintesis

Di bagian ini, kita menyusun pola kerja agent yang mencerminkan proses berpikir sistematis: merencanakan → bertindak → mengevaluasi → menyusun hasil.

Pola ini membuat alur agent lebih rapi dan memudahkan tim untuk memahami, memantau, serta mengembangkan sistem di kemudian hari. Kita akan menyimpan semua proses tersebut dalam sebuah function, yaitu `research_agent_react(topic)`. 

```python
def research_agent_react(topic):
...
```

Namun, sebelum itu, inisialisasi terlebih dahulu model LLM yang akan kita gunakan. Di sini, kita menggunakan `llama-3.3-70b-versatile`.

```python
# Inisialisasi LLM & Tool
llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
search = DuckDuckGoSearchRun()
```

Kita juga mengatur `temperature=0`. Dalam tugas ekstraksi data dan riset, kita membutuhkan respons yang deterministik (konsisten dan tidak berhalusinasi), sehingga format yang dihasilkan tetap stabil.

#### Thought: Agent Menyusun Rencana

```python
# --- THOUGHT STEP ---
# Agent menalar langkah pertama secara mandiri
reasoning_prompt = f"Tugas: Riset '{topic}'. Berikan satu kalimat pemikiran (thought) tentang informasi spesifik apa yang Anda butuhkan sebelum memberikan jawaban akhir."
thought = llm.invoke(reasoning_prompt).content
print(f"1. [THOUGHT]:\n   > {thought}")
```

Pada tahap ini, agent menggunakan LLM untuk melakukan penalaran mandiri (Though). Agent akan menganalisis tugas dan memutuskan informasi spesifik apa yang ia butuhkan sebelum memanggil tool search engine. Ini membuktikan bahwa agent memiliki inisiatif, bukan sekadar mengikuti skrip statis.

Dalam konteks industri, tahap ini berfungsi sebagai jejak penalaran. Tim dapat melihat alasan di balik setiap tindakan agent, melacak sumber kesalahan, dan memahami mengapa sebuah keputusan diambil. Hal ini menjadi semakin penting ketika sistem mulai melibatkan banyak agent atau banyak integrasi eksternal.

#### Action: Agent Mengakses Dunia Luar

```python
# --- ACTION STEP ---
print(f"\n2. [ACTION]:\n   > Memanggil DuckDuckGo Search API...")
try:
    observation = search.run(topic)
except Exception as e:
    observation = f"Gagal mengambil data: {e}"
```

Di sini, agent benar-benar berinteraksi dengan lingkungan eksternal. Ia memanggil search tool untuk mengambil data dari internet berdasarkan topik yang diberikan.

Langkah ini merepresentasikan pola integrasi yang umum di sistem produksi, seperti

*   mengakses mesin pencari,
*   mengambil data dari database perusahaan, dan
*   memanggil API pihak ketiga (DuckDuckGo).

Karena sumber eksternal tidak selalu stabil, bagian ini biasanya dilengkapi dengan penanganan error agar sistem tetap berjalan meskipun terjadi gangguan jaringan atau layanan.

#### Observation: Agent Mengevaluasi Hasil

```python
# --- OBSERVATION STEP ---
print(f"\n3. [OBSERVATION]:\n   > Berhasil mengumpulkan {len(observation)} karakter data.")
```

Setelah data diterima, agent tidak langsung memprosesnya. Ia terlebih dahulu memastikan bahwa hasil yang diperoleh ada dan layak digunakan. Dalam contoh ini, agent memvalidasi panjang karakternya.

Tahap ini mencerminkan praktik penting dalam sistem produksi, yaitu validasi input sebelum pemrosesan lanjutan. Dengan cara ini, agent dapat menghindari kesalahan berantai yang disebabkan oleh data kosong, tidak relevan, atau tidak lengkap.

#### Sintesis: Mengubah Data Menjadi Informasi Terstruktur

Pada tahap akhir, agent kembali menggunakan LLM untuk menyaring hasil pencarian dan mengubahnya menjadi JSON yang sesuai dengan schema yang sudah didefinisikan.

```python
# --- STRUCTURED SYNTHESIS STEP ---
# Menggunakan with_structured_output untuk menjamin JSON valid sesuai class ResearchResult
print(f"\n4. [FINAL THOUGHT]:\n   > Melakukan validasi dan pemformatan data ke JSON...")

structured_llm = llm.with_structured_output(ResearchResult)
final_prompt = f"Berdasarkan data berikut, buatlah laporan riset terstruktur: {observation}"

# Hasil akhir langsung berupa objek ResearchResult, bukan string teks biasa
result = structured_llm.invoke(final_prompt)
return result
```

Alih-alih melakukan manipulasi teks yang rawan gagal, kita menggunakan fitur with_structured_output(). Fitur ini mengikat LLM secara langsung ke *class* ResearchResult yang kita buat di awal. Hasil akhirnya (result) dipastikan berupa objek Python yang valid dan siap diteruskan ke agent berikutnya pada Latihan Membuat Multi-Agent Menggunakan CrewAI untuk diolah menjadi artikel tanpa perlu *parsing* manual.

### Eksekusi: Menjalankan Agent dalam Satu Alur

Bagian terakhir menyimulasikan sistem utama yang memanggil agent, misalnya backend server atau aplikasi web.

```python
# EKSEKUSI
if __name__ == "__main__":
    topik_target = "Masa depan Agentic AI di tahun 2026"
    hasil = research_agent_react(topik_target)

    print("\n" + "="*40)
    print("HASIL AKHIR AGENT (VALIDATED OBJECT)")
    print("="*40)
    print(f"Topik          : {hasil.topik}")
    print(f"Prediksi 2026  : {hasil.prediksi_2026}")
    print(f"Teknologi Kunci: {', '.join(hasil.teknologi_kunci)}")
    print(f"Status         : {hasil.status_riset}")
```

Dalam konteks latihan ini, kita mengeksekusi agent secara langsung dari notebook. Namun dalam lingkungan produksi, peran ini biasanya diambil oleh:

*   Backend service,
*   Endpoint API, atau
*   Workflow automation.

Ketika dijalankan, Anda bisa melihat bahwa hasil bukan lagi teks mentah (*string*), melainkan objek tervalidasi di mana Anda bisa memanggil atributnya secara langsung seperti hasil.topik atau hasil.prediksi_2026. Output inilah yang nantinya akan menjadi *input* utama bagi *agent* selanjutnya.

Kita akan lanjut menggunakan output dari latihan ini untuk diproses di latihan berikutnya. Sampai bertemu disana!

---

## Agentic AI “Si Multi-Agent”

Pada materi sebelumnya, kita telah membahas bagaimana sebuah AI Agent bekerja untuk memahami perintah, menggunakan tools, dan menghasilkan respons atau tindakan.

Sekarang, kita akan melangkah satu tingkat lebih jauh dengan membahas konsep Agentic AI, yaitu pendekatan di mana tidak hanya ada satu agent, tetapi sekumpulan agent yang bekerja bersama dalam sebuah sistem.

Pendekatan ini terinspirasi dari cara manusia bekerja dalam tim. Dalam sebuah organisasi, tidak semua orang mengerjakan semua hal. Setiap individu biasanya memiliki peran dan keahliannya masing-masing, lalu ada pihak yang mengoordinasikan agar semua pekerjaan tersebut berjalan selaras.

Konsep Agentic AI sebagai sistem kolaboratif ini merujuk pada kerangka kerja [Agentic Workflows](https://www.deeplearning.ai/the-batch/issue-242/) yang dipopulerkan oleh Andrew Ng, serta implementasi sistem Multi-Agent yang dikembangkan oleh tim riset Microsoft (AutoGen) dan OpenAI.

### AI Agent vs Agentic AI

Sebelum masuk lebih jauh, kita akan me-recall kembali perbedaan perbedaan antara AI Agent dan Agentic AI yang sudah sering kita mention di materi-materi sebelumnya.

AI Agent umumnya merujuk pada satu entitas agent yang mampu menerima input, melakukan penalaran, menggunakan tools, dan menghasilkan output. Agent ini dapat menyelesaikan berbagai jenis tugas, tetapi biasanya bekerja dalam ruang lingkup tugas yang spesifik dan alur yang relatif linear, misalnya merespons pertanyaan, memanggil API tertentu, atau menjalankan langkah-langkah yang telah diprogram sebelumnya.

**Agentic AI**, di sisi lain, merujuk pada sebuah pendekatan atau tingkat kemampuan sistem AI dalam bertindak sebagai agent yang lebih mandiri. **Fokus utamanya bukan pada jumlah agent, melainkan pada tingkat otonomi yang dimiliki sistem.** Dalam Agentic AI, sistem tidak hanya merespons perintah, tetapi juga mampu memahami tujuan dalam lingkup yang lebih tinggi, memecahnya menjadi sub-tujuan, merencanakan langkah-langkah yang perlu dilakukan, mengeksekusi tindakan, serta mengevaluasi dan menyesuaikan strateginya berdasarkan hasil yang diperoleh.

Dalam bentuk paling sederhana, Agentic AI dapat diwujudkan oleh satu agent tunggal yang memiliki kemampuan perencanaan dan pengambilan keputusan yang lebih dalam. Namun, dalam praktik dan arsitektur yang lebih kompleks, pendekatan ini sering diimplementasikan dalam bentuk sistem multi-agent.

![Multi-Agent Architecture](https://assets.cdn.dicoding.com/original/academy/dos-cb4077f69e3326cc927007b6413c91db20260429133618.png)

Pada pola ini, setiap agent memiliki peran atau fokus tertentu, misalnya satu agent khusus untuk pencarian informasi, satu agent untuk analisis data, dan satu agent untuk merangkai hasil akhir dalam bentuk laporan. Semua agent tersebut kemudian dikoordinasikan oleh sebuah komponen pengatur sehingga dapat bekerja bersama sebagai satu sistem yang utuh.

Melalui pembagian peran dan koordinasi, sistem Agentic AI mampu menangani tugas yang lebih besar dan kompleks dibandingkan satu agent tunggal. Oleh karena itu, multi-agent dapat dipahami sebagai salah satu bentuk implementasi Agentic AI, dengan esensi utama pada kemampuan sistem untuk bertindak secara mandiri, adaptif, dan berorientasi pada tujuan jangka panjang.

#### Mengapa Ada Agentic AI?

Konsep Agentic AI muncul sebagai respons terhadap meningkatnya kompleksitas tugas yang diharapkan dapat ditangani oleh sistem AI modern. Pada tahap awal, satu AI Agent sering kali sudah cukup untuk menjawab pertanyaan, mengambil informasi, atau menjalankan satu alur kerja sederhana. Namun, seiring bertambahnya kebutuhan pengguna, tugas yang diberikan tidak lagi berdiri sendiri, melainkan terdiri dari banyak langkah yang saling bergantung, melibatkan berbagai sumber data, serta memerlukan pengambilan keputusan di setiap tahap.

Dalam pendekatan single agent, semua tanggung jawab tersebut ditumpukan pada satu entitas. Agent harus memahami tujuan pengguna, memecahnya menjadi langkah-langkah, memilih dan memanggil tools yang tepat, memproses hasil, lalu menyusun output akhir. Ketika beban ini semakin besar, beberapa masalah mulai muncul. Proses penalaran menjadi lebih panjang dan sulit dilacak, alur kerja menjadi kurang terstruktur, dan peluang terjadinya kesalahan meningkat, terutama jika satu langkah gagal dan berdampak pada seluruh rangkaian proses.

Agentic AI hadir untuk mengatasi keterbatasan ini dengan mengadopsi prinsip pembagian peran. Alih-alih satu agent melakukan semuanya, sistem dirancang agar pekerjaan dibagi ke beberapa komponen atau agent yang masing-masing memiliki fokus tertentu. Misalnya, satu agent bertugas mengumpulkan data, agent lain menganalisis dan menafsirkan data tersebut, dan agent lain menyusun hasil dalam bentuk laporan atau rekomendasi. Dengan cara ini, setiap agent dapat dioptimalkan untuk perannya, baik dari sisi kemampuan penalaran, tools yang digunakan, maupun aturan yang diterapkan.

![Agentic AI Roles](https://assets.cdn.dicoding.com/original/academy/dos-805a8b342e785ee77af923028aa1d08020260429133641.png)

Pendekatan ini membawa keuntungan dari sisi struktur dan skalabilitas. Sistem menjadi lebih modular, artinya setiap bagian dapat dikembangkan, diuji, dan diperbaiki secara terpisah. Jika ada perubahan pada satu domain, seperti sumber data baru atau aturan bisnis yang diperbarui, pengembang hanya perlu menyesuaikan agent yang relevan tanpa harus merombak keseluruhan sistem. Hal ini membuat Agentic AI lebih fleksibel dan lebih mudah dirawat dalam jangka panjang.

Selain itu, Agentic AI juga mendukung cara kerja yang lebih adaptif. Dalam sistem multi-agent, beberapa agent dapat bekerja secara paralel untuk mengeksplorasi berbagai kemungkinan solusi. Hasil dari masing-masing agent kemudian dapat dibandingkan atau digabungkan untuk memilih pendekatan terbaik. Pola ini sangat berguna dalam tugas-tugas yang bersifat terbuka atau tidak memiliki satu jawaban benar, seperti perencanaan strategi, analisis risiko, atau pembuatan konten kreatif.

Dari sudut pandang operasional, Agentic AI juga membantu meningkatkan keandalan sistem. Jika satu agent mengalami kegagalan atau menghasilkan output yang kurang memadai, sistem masih dapat melanjutkan proses dengan mengandalkan agent lain atau menjalankan mekanisme koreksi. Dengan demikian, keseluruhan sistem menjadi lebih tahan terhadap kesalahan dibandingkan dengan pendekatan single agent yang sangat bergantung pada satu titik kegagalan.

Jadi, Agentic AI ada karena kebutuhan untuk membangun sistem AI yang mampu mengelola proses, berkolaborasi secara internal, dan bekerja secara lebih mandiri dalam mencapai tujuan yang ditetapkan.

#### Definisi Agentic AI dari Berbagai Perspektif

Sebelum kita membedah lebih dalam, perlu diingat bahwa istilah Agentic AI sering kali digunakan secara bergantian dengan AI Agent dalam percakapan umum. Namun, seperti yang telah kita bahas, Agentic AI lebih menekankan pada tingkat otonomi dan workflow yang mandiri, sedangkan AI Agent lebih merujuk pada unit entitasnya.

*   Dalam literatur penelitian, Agentic AI umumnya didefinisikan sebagai sistem AI yang menunjukkan tingkat agency, yaitu kemampuan untuk bertindak sebagai agen yang otonom dalam suatu lingkungan. Fokus utama definisi ini bukan hanya pada kemampuan menghasilkan respons, tetapi pada kemampuan sistem untuk menetapkan tujuan, merencanakan langkah-langkah untuk mencapainya, mengambil keputusan di sepanjang proses, serta menyesuaikan perilakunya berdasarkan umpan balik dari lingkungan.
*   Dalam banyak tulisan ilmiah, pendekatan ini sering diwujudkan dalam bentuk sistem multi-agent, di mana beberapa agent otonom saling berinteraksi, berkoordinasi, dan terkadang bernegosiasi untuk mencapai tujuan bersama. Setiap agent dipandang sebagai entitas yang memiliki persepsi, mekanisme penalaran, dan kebijakan tindakan sendiri, tetapi tetap beroperasi dalam kerangka aturan dan tujuan global yang sama.
    Peneliti juga menekankan aspek adaptasi dan pembelajaran berkelanjutan dalam Agentic AI. Sistem tidak hanya menjalankan rencana yang sudah ditentukan, tetapi mampu memperbarui strategi berdasarkan pengalaman sebelumnya, perubahan lingkungan, atau hasil yang tidak sesuai harapan. Dengan cara ini, Agentic AI dipandang sebagai sistem yang dinamis, yang perilakunya dapat berkembang seiring waktu, bukan sekadar mengikuti skrip statis.
*   Dari perspektif industri, definisi Agentic AI cenderung lebih berorientasi pada nilai praktis dan penerapan di dunia nyata. Perusahaan teknologi besar sering menggambarkannya sebagai arsitektur AI yang dirancang untuk menangani alur kerja end-to-end secara otomatis. Dalam pandangan ini, Agentic AI tidak hanya berfungsi sebagai antarmuka percakapan, tetapi sebagai sistem yang mampu mengelola seluruh proses bisnis atau operasional, mulai dari menerima goal, memecahnya menjadi tugas-tugas yang lebih kecil, menentukan urutan eksekusi, hingga mengoordinasikan berbagai tool untuk menyelesaikannya.

Pendekatan industri juga menyoroti pentingnya integrasi. Agentic AI diposisikan sebagai “penghubung” antara generative AI, sistem internal perusahaan, layanan pihak ketiga, dan pengguna akhir. Sistem ini harus mampu berkomunikasi dengan berbagai API, mengakses basis data, mematuhi aturan bisnis, dan memastikan bahwa setiap langkah yang diambil selaras dengan kebijakan dan tujuan organisasi. Dalam konteks ini, agent sering kali dilengkapi dengan mekanisme pengawasan, logging, dan kontrol agar setiap tindakan dapat ditelusuri dan dipertanggungjawabkan.

Perbedaan utama antara perspektif penelitian dan industri terletak pada fokusnya. Perspektif penelitian menekankan konsep teoretis seperti otonomi dan pembelajaran adaptif, sedangkan perspektif industri berfokus pada arsitektur, skalabilitas, dan dampak bisnis. Namun, keduanya sepakat bahwa Agentic AI bukan sekadar pemberi jawaban, melainkan pengelola proses dan pengambil tindakan yang berorientasi pada tujuan.

Dengan memahami kedua perspektif ini, pengembang dapat membangun Agentic AI yang tidak hanya kuat secara konsep, tetapi juga relevan dan dapat diterapkan secara nyata. Pendekatan teoretis memberikan dasar untuk merancang sistem yang adaptif dan cerdas, sementara pendekatan industri memastikan bahwa sistem tersebut dapat diintegrasikan, dioperasikan, dan memberikan nilai nyata dalam lingkungan produksi.

#### Perbedaan AI Agent dan Agentic AI

Nah, beranjak ke perbedaan AI Agent dan Agentic AI, dalam penerapan di dunia nyata, AI Agent dan Agentic AI dibedakan terutama oleh lingkup tanggung jawab, tingkat otonomi, dan kompleksitas alur kerja yang mampu mereka tangani. Perbedaan ini memengaruhi bagaimana sistem dirancang, diintegrasikan, dan digunakan dalam lingkungan produksi.

AI Agent biasanya digunakan untuk tugas-tugas yang relatif terfokus dan memiliki tujuan yang jelas serta batasan yang tegas. Contohnya adalah chatbot layanan pelanggan yang menjawab pertanyaan umum, agent pencarian informasi yang mengambil data dari satu atau dua sumber, atau agent yang membantu mengisi formulir dan memproses data sederhana.

Dalam skenario ini, alur kerja cenderung linier. Pengguna memberikan perintah, agent memprosesnya, mungkin memanggil satu atau dua tools, lalu menghasilkan output. Pengambilan keputusan agent biasanya berada dalam ruang lingkup yang sudah ditentukan oleh pengembang, seperti aturan bisnis, format jawaban, atau daftar tools yang boleh digunakan.

Agentic AI, di sisi lain, dirancang untuk menangani tujuan yang tidak selalu dapat diselesaikan dalam satu langkah atau satu jenis tindakan. Sistem ini harus mampu memecah tujuan besar menjadi serangkaian sub-tugas, menentukan urutan pengerjaan, dan menyesuaikan rencana ketika kondisi berubah atau hasil yang diperoleh tidak sesuai harapan.

Dalam sistem analisis bisnis, misalnya, prosesnya bisa dimulai dari pengumpulan data dari berbagai sumber, dilanjutkan dengan pembersihan dan analisis data, kemudian penyusunan laporan, dan diakhiri dengan pembuatan rekomendasi. Setiap tahap ini bisa ditangani oleh agent yang berbeda, tetapi semuanya berada dalam satu kerangka tujuan yang sama.

Perbedaan lainnya terletak pada cara sistem beradaptasi. AI Agent cenderung reaktif, artinya mereka merespons permintaan yang diberikan pengguna dan menyelesaikannya sesuai dengan alur yang sudah dirancang. Agentic AI bersifat lebih proaktif dan reflektif. Sistem tidak hanya menjalankan perintah, tetapi juga memantau hasil dari setiap langkah, mengevaluasi apakah tujuan sudah tercapai, dan jika belum, menentukan langkah lanjutan yang perlu dilakukan. Pola ini membuat Agentic AI lebih cocok untuk lingkungan yang dinamis, di mana informasi dan kondisi dapat berubah seiring waktu.

Dari sisi arsitektur, AI Agent biasanya lebih sederhana dan lebih mudah diimplementasikan. Sistem ini cocok untuk kebutuhan yang spesifik dan dapat dikembangkan dengan cepat. Agentic AI, sebaliknya, membutuhkan perancangan yang lebih matang, termasuk mekanisme koordinasi, pembagian tugas, memori bersama, dan pengawasan. Meskipun lebih kompleks, pendekatan ini memberikan keuntungan dalam hal skalabilitas dan fleksibilitas, karena sistem dapat diperluas dengan menambahkan agent baru tanpa harus mengubah keseluruhan struktur.

Pada akhirnya, pemilihan antara AI Agent dan Agentic AI bergantung pada kebutuhan aplikasi.

> **Latihan Fill in the Blank:**
> Jika tujuannya adalah menyelesaikan tugas-tugas yang sederhana, cepat, dan terfokus, **AI Agent** sering kali sudah memadai. Namun, jika sistem diharapkan untuk mengelola proses yang panjang, melibatkan banyak sumber, dan membutuhkan pengambilan keputusan di berbagai tahap, maka pendekatan **Agentic AI** menjadi pilihan yang lebih tepat.

Setelah mengetahui hal-hal konseptual tentang Agentic AI, mari kita mulai masuk ke pembahasan yang lebih teknikal, yaitu arsitektur Agentic AI. Let’s go!

### Arsitektur Agentic AI

Agar sistem multi-agent dapat bekerja dengan baik, dibutuhkan arsitektur yang jelas dan terstruktur. Arsitektur ini mengatur pola komunikasi antar-agent, mekanisme pembagian tugas, dan integrasi hasil.

#### Ensemble of Specialized Agents

Pada pendekatan Ensemble of Specialized Agents, sistem dirancang sebagai sebuah ekosistem yang terdiri dari beberapa agent otonom, di mana masing-masing agent memiliki peran, keahlian, dan tanggung jawab yang spesifik.

![Ensemble of Specialized Agents](https://assets.cdn.dicoding.com/original/academy/dos-ea6b5dab5069cc90d7d08326aa1e6f0020260326103728.png)

Alih-alih hanya mengandalkan satu agent serbaguna untuk menangani seluruh proses, pendekatan ini memecah beban kerja menjadi bagian-bagian yang lebih kecil dan terdefinisi dengan jelas. Setiap agent diperlakukan sebagai pakar dalam satu aspek tertentu dari keseluruhan tugas.

Spesialisasi dapat dibentuk berdasarkan fungsi maupun domain pengetahuan. Dari sisi fungsi, misalnya, terdapat agent yang berfokus pada pengumpulan informasi (search atau retrieval agent), agent yang bertugas melakukan analisis dan penalaran (analysis agent), serta agent yang menyusun dan menyajikan hasil akhir dalam bentuk laporan, ringkasan, atau rekomendasi (generation atau synthesis agent).

![Specialized Agent Roles](https://assets.cdn.dicoding.com/original/academy/dos-a82657383b4387bd63b5bd1db480e1d220260429133700.png)

Dari sisi domain, agent dapat dibedakan berdasarkan bidang keahlian, seperti keuangan, kesehatan, teknis, hukum, atau layanan pelanggan, sehingga setiap agent memahami terminologi, konteks, dan aturan yang relevan dengan bidangnya masing-masing.

Keuntungan utama dari pendekatan ini adalah optimalisasi kinerja dan kualitas hasil. Karena setiap agent hanya menangani ruang lingkup yang sempit, model, prompt, tools, dan data yang digunakan dapat disesuaikan secara khusus untuk peran tersebut. Misalnya, agent pencarian dapat diintegrasikan dengan sistem web search, database internal, atau vector retrieval untuk memastikan informasi yang dikumpulkan selalu relevan dan terkini. Sementara itu, agent analisis dapat dilengkapi dengan kemampuan pemrosesan numerik, statistik, atau logika bisnis untuk menafsirkan data dan menemukan pola atau insight yang bermakna.

Dari sudut pandang alur kerja, interaksi antara agent biasanya diatur melalui mekanisme koordinasi atau orkestrasi. Sebuah komponen pusat, sering disebut sebagai controller atau orchestrator agent, bertugas menerima tujuan tingkat tinggi dari pengguna, memecahnya menjadi sub-tugas, lalu mendistribusikannya ke agent yang paling sesuai. Setelah setiap agent menyelesaikan bagiannya, hasilnya dikumpulkan kembali dan disintesis menjadi satu output yang koheren dan terintegrasi.

Pendekatan ini juga memberikan keuntungan dalam hal skalabilitas dan pemeliharaan sistem. Ketika kebutuhan baru muncul, pengembang dapat menambahkan agent spesialis baru tanpa harus merombak keseluruhan arsitektur. Jika terjadi masalah atau penurunan performa pada satu bagian, perbaikan dapat difokuskan hanya pada agent terkait, sehingga risiko gangguan terhadap sistem secara keseluruhan menjadi lebih kecil.

Namun, pendekatan Ensemble of Specialized Agents juga membawa tantangan tersendiri. Koordinasi antar-agent harus dirancang dengan baik agar tidak terjadi konflik, redundansi, atau kehilangan konteks. Selain itu, sistem perlu memiliki mekanisme untuk memastikan konsistensi dan kualitas hasil akhir, terutama ketika informasi atau analisis yang dihasilkan oleh beberapa agent perlu digabungkan menjadi satu narasi yang utuh dan dapat dipahami oleh pengguna.

Secara keseluruhan, pendekatan ini mencerminkan cara kerja tim manusia dalam menyelesaikan masalah kompleks, di mana setiap anggota memiliki peran dan keahlian tertentu, tetapi tetap bekerja dalam satu tujuan bersama. Dengan desain yang tepat, Ensemble of Specialized Agents memungkinkan sistem Agentic AI menjadi lebih kuat, fleksibel, dan andal dalam menangani tugas-tugas berskala besar dan multidimensi.

#### Advanced Reasoning and Planning

Dalam Agentic AI, kemampuan penalaran (reasoning) dan perencanaan (planning) merupakan fondasi utama yang memungkinkan sistem tidak hanya merespons perintah, tetapi juga memahami tujuan, menyusun strategi, dan mengeksekusi langkah-langkah secara terstruktur.

Berbeda dengan sistem AI sederhana yang langsung menghasilkan jawaban dari satu input, Agentic AI harus mampu memecah tujuan besar menjadi rangkaian tindakan kecil, mengoordinasikan peran beberapa agent, serta menyesuaikan rencana ketika kondisi atau informasi baru muncul.

##### ReAct (Reasoning + Acting)

Pendekatan ReAct menggabungkan proses berpikir (reasoning) dan bertindak (acting) dalam satu siklus yang berulang. Alih-alih merencanakan seluruh langkah dari awal secara statis, agent akan

*   **Menalar**: Memahami tujuan dan menentukan langkah berikutnya.
*   **Bertindak**: Menjalankan aksi, seperti memanggil tools, mencari data, atau meminta hasil dari agent lain.
*   **Mengamati**: Mengevaluasi hasil dari tindakan tersebut.
*   **Menyesuaikan rencana**: Memperbarui strategi berdasarkan observasi terbaru.

Dalam konteks multi-agent, satu agent dapat berperan sebagai pengambil keputusan utama yang menggunakan hasil observasi dari agent lain sebagai “bahan baku” untuk langkah selanjutnya. Misalnya, agent perencana meminta agent pencarian untuk mengumpulkan data pasar, lalu menggunakan hasil tersebut untuk menentukan apakah perlu dilakukan analisis lanjutan atau langsung menyusun rekomendasi.

Pendekatan ini membuat sistem lebih adaptif dan responsif terhadap perubahan karena rencana tidak bersifat kaku, melainkan terus diperbarui seiring masuknya informasi baru.

##### Chain of Thought (CoT)

Chain of Thought berfokus pada kemampuan agent untuk memecah masalah kompleks menjadi serangkaian langkah penalaran yang lebih kecil dan logis. Setiap langkah mewakili satu bagian dari proses berpikir sehingga jalur dari input hingga output menjadi lebih transparan dan terstruktur.

Dalam sistem Agentic AI, pendekatan ini dapat diterapkan dengan cara mendistribusikan langkah-langkah tersebut ke beberapa agent. Misalnya:

*   Agent pertama bertugas memahami dan merumuskan ulang masalah.
*   Agent kedua mengumpulkan atau memverifikasi informasi yang dibutuhkan.
*   Agent ketiga melakukan analisis atau perhitungan.
*   Agent keempat menyusun hasil akhir dalam bentuk yang mudah dipahami pengguna.

Dengan pembagian seperti ini, sistem tidak hanya menjadi lebih modular, tetapi juga lebih mudah diaudit dan dievaluasi karena setiap tahap penalaran dapat ditelusuri ke agent yang bertanggung jawab.

##### Tree of Thoughts (ToT)

Tree of Thoughts memperluas konsep rantai penalaran menjadi sebuah struktur bercabang, di mana sistem tidak hanya mengikuti satu jalur solusi, tetapi mengeksplorasi beberapa kemungkinan secara paralel. Setiap cabang mewakili pendekatan atau hipotesis yang berbeda untuk menyelesaikan masalah.

![Tree of Thoughts](https://assets.cdn.dicoding.com/original/academy/dos-53e02d2d413a5428610ab2789bf4a80d20260326103751.png)

Dalam skenario multi-agent, pendekatan ini sangat alami untuk diterapkan. Beberapa agent dapat ditugaskan untuk

*   mencoba strategi A,
*   mengembangkan solusi alternatif B, atau
*   mengevaluasi risiko dan dampak dari solusi C.

Setelah semua jalur dieksplorasi, hasilnya dikumpulkan dan dibandingkan oleh agent pengoordinasi. Agent ini kemudian memilih solusi yang paling masuk akal, paling efisien, atau paling sesuai dengan kriteria yang ditentukan pengguna. Pendekatan ini sangat berguna untuk masalah yang bersifat ambigu, strategis, atau berdampak besar, seperti perencanaan bisnis, pengambilan keputusan kebijakan, atau desain sistem kompleks.

Secara keseluruhan, Advanced Reasoning and Planning memungkinkan Agentic AI untuk berfungsi lebih seperti sebuah tim yang berpikir strategis, bukan sekadar mesin penjawab. Reasoning memastikan bahwa setiap langkah memiliki dasar logis dan relevan dengan tujuan, sementara planning memastikan bahwa langkah-langkah tersebut disusun dalam urutan yang efektif dan efisien.

Dengan mengombinasikan ReAct, Chain of Thought, dan Tree of Thoughts, sistem dapat

*   bersifat adaptif (mampu menyesuaikan rencana saat kondisi berubah),
*   terstruktur (memiliki alur penalaran yang jelas), dan
*   eksploratif (mampu mempertimbangkan berbagai alternatif solusi).

Inilah yang membuat Agentic AI unggul dalam menangani tugas-tugas kompleks yang melibatkan banyak keputusan, banyak sumber informasi, dan banyak kemungkinan hasil, dibandingkan dengan sistem AI yang hanya mengandalkan satu jalur pemrosesan dari input ke output.

#### Persistent Memory

Dalam sistem Agentic AI, persistent memory berperan sebagai ingatan jangka panjang yang memungkinkan seluruh sistem belajar dan berkembang dari waktu ke waktu. Berbeda dengan konteks percakapan atau working memory yang hanya berlaku selama satu sesi atau satu tugas, persistent memory menyimpan informasi yang tetap tersedia meskipun sistem dimatikan atau tugas sudah selesai.

Persistent memory digunakan untuk menyimpan berbagai jenis informasi penting seperti berikut.

*   **Pengetahuan umum dan domain**: Fakta, aturan, atau referensi yang sering digunakan oleh agent, misalnya kebijakan perusahaan, dokumentasi teknis, atau standar operasional.
*   **Riwayat interaksi**: Catatan percakapan dengan pengguna, preferensi pengguna, atau keputusan yang pernah diambil sistem dalam konteks tertentu.
*   **Hasil kerja agent**: Output analisis, laporan, ringkasan, atau kesimpulan dari tugas sebelumnya yang masih relevan untuk tugas di masa depan.

Dengan adanya memori ini, sistem tidak perlu memulai dari nol setiap kali menerima tugas baru. Agent dapat langsung mengacu pada pengetahuan dan pengalaman yang sudah tersimpan, sehingga proses menjadi lebih cepat dan lebih konsisten.

Dalam arsitektur multi-agent, memori sering kali tidak hanya disimpan di satu tempat, tetapi dibagi atau didistribusikan. Artinya:

*   Setiap agent dapat memiliki memori lokal untuk menyimpan informasi spesifik yang berkaitan dengan tugasnya.
*   Di sisi lain, ada juga memori bersama yang dapat diakses oleh semua agent. Misalnya, dalam bentuk basis data pusat, knowledge graph, atau sistem vector database untuk pencarian semantik.

Pendekatan ini memungkinkan kolaborasi yang lebih efektif. Misalnya, jika agent pencarian menemukan data penting, hasil tersebut dapat disimpan di memori bersama, sehingga agent analisis dan agent penulis laporan dapat langsung menggunakannya tanpa perlu mencari ulang.

Secara konseptual, persistent memory biasanya dibagi menjadi beberapa lapisan:

*   **Factual Memory**: Menyimpan data dan fakta yang relatif stabil, seperti definisi, aturan, atau referensi.
*   **Contextual Memory**: Menyimpan informasi tentang situasi atau kasus tertentu, misalnya detail proyek yang sedang berjalan.
*   **Preference Memory**: Menyimpan kecenderungan atau kebiasaan pengguna, seperti format laporan yang disukai atau gaya bahasa yang sering diminta.

Dengan pemisahan ini, agent dapat mengambil jenis informasi yang paling relevan dengan kebutuhan saat itu, tanpa harus memproses seluruh memori secara sekaligus.

![Persistent Memory Layers](https://assets.cdn.dicoding.com/original/academy/dos-dfc85774528c1eccfc8d0d52e63e222f20260429133753.png)

Dalam praktiknya, persistent memory sering digunakan pada beberapa titik penting dalam siklus kerja agent.

*   **Saat perencanaan**: Agent membaca memori untuk memahami konteks dan pengalaman sebelumnya untuk menyusun rencana.
*   **Saat eksekusi**: Agent dapat menyimpan hasil sementara atau temuan penting agar bisa digunakan oleh agent lain.
*   **Saat refleksi**: Setelah tugas selesai, sistem dapat menyimpan kesimpulan atau evaluasi sebagai bahan pembelajaran di masa depan.

Proses ini membuat sistem Agentic AI bersifat lebih kontinu dan berkembang, bukan hanya reaktif terhadap satu perintah saja.

Meskipun sangat bermanfaat, persistent memory juga membawa tantangan seperti berikut.

*   **Konsistensi data**: Informasi yang tersimpan harus dijaga agar tetap akurat dan tidak saling bertentangan.
*   **Skalabilitas**: Semakin lama sistem berjalan, semakin besar memori yang harus dikelola dan dicari.
*   **Privasi dan keamanan**: Riwayat interaksi dan data pengguna harus disimpan dan diakses dengan mekanisme yang aman.

Secara keseluruhan, persistent memory membuat Agentic AI lebih mirip sebuah organisasi yang memiliki arsip dan pengalaman kolektif, bukan sekadar sekumpulan agent yang bekerja secara terpisah. Dengan memori jangka panjang yang terkelola dengan baik, sistem dapat memberikan jawaban yang lebih relevan, konsisten, dan kontekstual, serta mampu meningkatkan kualitas keputusannya seiring waktu.

#### Meta-Agents (Orchestrator)

Dalam arsitektur Agentic AI, meta-agent atau orchestrator dapat diibaratkan sebagai manajer proyek dalam sebuah tim. Jika setiap agent adalah spesialis yang mengerjakan bagian tertentu, maka orchestrator bertugas memastikan seluruh pekerjaan berjalan selaras menuju satu tujuan utama. Tanpa komponen ini, sistem multi-agent berisiko menjadi sekumpulan proses yang berjalan sendiri-sendiri tanpa arah yang jelas.

Tugas pertama orchestrator adalah memahami tujuan pengguna secara menyeluruh. Ketika pengguna menyampaikan sebuah permintaan, misalnya “buatkan laporan analisis pasar dari data penjualan dan tren media sosial,” orchestrator tidak langsung mengeksekusi tugas tersebut. Ia terlebih dahulu menerjemahkan permintaan ini menjadi tujuan yang lebih terstruktur dan dapat dikerjakan oleh beberapa agent.

Setelah itu, orchestrator melakukan dekomposisi tugas, yaitu memecah tujuan besar menjadi beberapa sub-tugas yang lebih kecil dan spesifik. Dalam contoh tadi, sub-tugas bisa berupa mengumpulkan data penjualan, mencari tren terbaru dari media sosial, menganalisis pola dan korelasi, serta menyusun laporan akhir.

Langkah berikutnya adalah penugasan agent. Orchestrator memilih agent yang paling sesuai untuk setiap sub-tugas berdasarkan spesialisasi dan kemampuan tools yang dimiliki. Agent pencarian diberi tugas mengambil data eksternal, agent analisis diberi data untuk diproses, dan agent penulisan bertugas merangkai hasil menjadi narasi yang mudah dipahami.

Dalam banyak kasus, sub-tugas tidak selalu bisa dijalankan secara paralel. Beberapa langkah bergantung pada hasil langkah sebelumnya. Orchestrator bertugas mengatur urutan eksekusi dan ketergantungan antar agent. Misalnya, agent analisis baru bisa bekerja setelah agent pencarian selesai mengumpulkan data yang diperlukan.

![Meta-Agent Orchestrator](https://assets.cdn.dicoding.com/original/academy/dos-bca79741020552ad250f2a3b14d1482220260326104148.png)

Selain itu, orchestrator juga memantau status dan kemajuan setiap agent. Jika ada agent yang gagal atau menghasilkan output yang tidak sesuai, orchestrator dapat memutuskan untuk mengulangi tugas tersebut, mengirimnya ke agent lain, atau mengubah strategi penyelesaian.

Setelah semua sub-tugas selesai, orchestrator menjalankan proses integrasi hasil. Output dari berbagai agent sering kali berbentuk potongan informasi yang terpisah-pisah, seperti tabel data, ringkasan analisis, atau catatan temuan. Orchestrator menggabungkan semua ini menjadi satu hasil akhir yang koheren, konsisten, dan sesuai dengan tujuan awal pengguna.

Pada tahap ini, orchestrator juga dapat melakukan validasi kualitas, misalnya memeriksa kelengkapan jawaban dari semua bagian, konsistensi informasi, dan kesesuaian format hasil dengan preferensi pengguna.

Selain mengatur alur kerja, meta-agent juga berperan dalam pengambilan keputusan tingkat tinggi. Contohnya adalah

*   Menentukan apakah sebuah tugas lebih efisien dikerjakan oleh satu agent atau dibagi ke beberapa agent;
*   Memilih strategi penalaran, seperti menggunakan pendekatan cepat untuk jawaban sederhana atau pendekatan lebih mendalam untuk masalah kompleks; dan
*   Mengatur penggunaan sumber daya, misalnya membatasi pemanggilan API tertentu agar biaya dan latency tetap terkendali.

Dengan peran ini, orchestrator tidak hanya mengoordinasikan, tetapi juga mengoptimalkan cara sistem bekerja secara keseluruhan.

Dalam sistem yang lebih maju, orchestrator sering terhubung erat dengan persistent memory. Ia dapat membaca pengalaman sebelumnya untuk menentukan strategi terbaik, misalnya memilih agent yang paling andal untuk jenis tugas tertentu atau menghindari pendekatan yang sebelumnya terbukti kurang efektif.

Seiring waktu, hal ini membuat sistem Agentic AI menjadi lebih adaptif dan belajar dari pengalamannya sendiri, meskipun pembelajaran ini biasanya bersifat pada level alur kerja dan strategi, bukan pada perubahan bobot model secara langsung.

Secara keseluruhan, meta-agent adalah kunci yang membuat sistem multi-agent terasa seperti sebuah tim yang terorganisir, bukan sekadar kumpulan individu yang bekerja secara terpisah. Dengan adanya orchestrator, pengguna cukup berfokus pada tujuan akhir, sementara sistem di balik layar mengatur perencanaan, kolaborasi, eksekusi, dan penyatuan hasil secara otomatis.

Inilah yang membedakan Agentic AI dari sekadar AI Agent yang berdiri sendiri. Agentic AI tidak hanya cerdas dalam menjawab, tetapi juga cerdas dalam mengelola proses kerja, mendekati cara manusia mengoordinasikan sebuah tim untuk menyelesaikan proyek yang kompleks.

---

## Mari Membuat Agent Lebih “Pintar”

Pada tahap ini, kita mulai masuk ke bagian yang lebih praktis dan strategis, yaitu bagaimana membuat AI Agent tidak hanya mampu bekerja, tetapi juga bekerja dengan lebih cerdas, efisien, dan adaptif. Dalam konteks ini, “lebih pintar” tidak selalu berarti kita harus melatih ulang model bahasa dari awal. Sebaliknya, banyak peningkatan kualitas agent justru bisa dicapai melalui cara kita merancang instruksi, memilih tools, dan mengatur alur kerja agent.

Materi ini dibagi menjadi dua pendekatan utama. Pendekatan pertama berfokus pada peningkatan kemampuan agent tanpa melakukan fine-tuning model. Pendekatan kedua membahas konsep fine-tuning khusus untuk agent, yang berbeda dari fine-tuning model bahasa pada umumnya, dengan mengacu pada pendekatan berbasis pengalaman dan memori seperti yang dijelaskan dalam paper Memento.

### Tanpa Fine-Tuning

Pendekatan ini menekankan bahwa banyak perilaku cerdas dari agent dapat dibentuk melalui desain prompt dan pengelolaan tools yang tepat. Dengan kata lain, kita memanfaatkan kemampuan bawaan LLM, lalu mengarahkannya agar bekerja sesuai dengan tujuan sistem yang kita bangun.

#### Prompt Engineering untuk Agent

Prompt engineering pada agent tidak hanya sekadar menulis instruksi agar model menghasilkan jawaban yang benar. Dalam konteks agent, prompt berperan sebagai “aturan main” yang membentuk cara agent berpikir, bertindak, dan berinteraksi dengan tools serta memori.

##### Meta-Prompt

Meta-prompt dapat dipahami sebagai fondasi identitas dan perilaku dari sebuah AI Agent. Jika prompt biasa adalah instruksi untuk satu tugas tertentu, maka meta-prompt yang mengatur bagaimana agent harus bersikap dalam semua tugas yang akan ia kerjakan. Ia tidak hanya menjawab pertanyaan, tetapi membentuk cara agent berpikir, mengambil keputusan, dan berinteraksi dengan pengguna maupun dengan sistem di sekitarnya.

Pada level paling dasar, meta-prompt menetapkan peran dan sudut pandang agent. Misalnya, agent bisa didefinisikan sebagai analis bisnis, tutor pembelajaran, atau asisten teknis. Definisi peran ini memengaruhi cara agent memilih informasi yang dianggap penting, istilah yang digunakan, serta kedalaman penjelasan yang diberikan. Seorang agent yang didefinisikan sebagai tutor, misalnya, akan cenderung memberikan penjelasan bertahap dan contoh sederhana, sementara agent yang didefinisikan sebagai analis akan lebih fokus pada ringkasan data dan insight.

Selain peran, meta-prompt juga menetapkan tujuan utama dan prioritas. Dalam sistem yang kompleks, agent sering dihadapkan pada beberapa tujuan sekaligus, seperti memberikan jawaban yang cepat, menjaga akurasi, dan meminimalkan biaya pemanggilan tools. Melalui meta-prompt, pengembang dapat menentukan mana yang harus didahulukan. Contohnya, sebuah agent layanan pelanggan mungkin diprioritaskan untuk memberikan respons cepat, sedangkan agent riset mungkin diprioritaskan untuk kedalaman dan ketepatan informasi, meskipun membutuhkan waktu lebih lama.

Meta-prompt juga berfungsi sebagai kerangka etika dan batasan operasional. Di sini, kita bisa menetapkan aturan seperti menghindari spekulasi, tidak memberikan informasi sensitif, atau selalu meminta klarifikasi jika permintaan pengguna ambigu. Dengan adanya batasan ini, agent menjadi lebih aman dan konsisten dalam berperilaku, terutama ketika digunakan dalam konteks bisnis atau pendidikan.

Dalam konteks penggunaan tools, meta-prompt berperan sebagai panduan strategi. Alih-alih hanya memberi tahu agent bahwa sebuah tool tersedia, meta-prompt menjelaskan kapan dan mengapa tool tersebut sebaiknya digunakan. Misalnya, agent dapat diarahkan untuk selalu memeriksa basis data internal terlebih dahulu sebelum melakukan web search, atau hanya memanggil API tertentu jika informasi yang dibutuhkan bersifat transaksional. Dengan panduan ini, agent tidak hanya tahu cara menggunakan tools, tetapi juga memahami konteks penggunaannya.

Aspek lain yang penting adalah gaya komunikasi dan format output. Meta-prompt dapat menetapkan apakah agent harus menggunakan bahasa formal atau santai, apakah hasil harus disajikan dalam bentuk poin, tabel, atau narasi, serta apakah perlu menyertakan ringkasan di akhir. Aturan-aturan ini membuat interaksi dengan agent terasa konsisten dan profesional, terutama ketika digunakan oleh banyak pengguna dalam satu sistem.

Dalam sistem Agentic AI yang melibatkan banyak agent, meta-prompt juga membantu menjaga keselarasan antar agent. Meskipun setiap agent memiliki spesialisasi, meta-prompt dapat menetapkan nilai dan standar bersama, seperti cara menyampaikan hasil, cara mencatat pengalaman ke dalam memori, dan cara berkoordinasi dengan orchestrator. Dengan begitu, seluruh sistem bekerja dengan “bahasa” dan prinsip yang sama.

Dengan merancang meta-prompt yang jelas dan matang, Anda dapat membentuk agent yang tidak hanya pintar dalam menyelesaikan tugas, tetapi juga konsisten, dapat dipercaya, dan selaras dengan tujuan sistem. Ini menjadikan meta-prompt sebagai salah satu elemen kunci dalam membangun AI Agent dan Agentic AI yang siap digunakan di dunia nyata.

Berikut adalah salah satu contoh penerapan Meta Prompt yang bisa Anda pelajari, berupa komponen prompt dan instruksi spesifiknya.

> **Flashcard Meta Prompt:**
> *   **Task Description:** Kamu adalah asisten peneliti medis yang bertugas mengekstraksi parameter kunci diagnosis PCOS dari teks laporan radiologi.
> *   **Domain Knowledge:** Fokus pada kriteria Morfologi Ovarium Polikistik (PCOM), yaitu jumlah folikel (ambang batas >12) dan volume ovarium (>10ml).
> *   **Solution Guidance:** Identifikasi angka spesifik untuk jumlah folikel dan volume. Bedakan antara ovarium kiri dan kanan jika tersedia.
> *   **Exception Handling:** Abaikan temuan negatif atau informasi non-spesifik seperti "massa adneksa" atau "cairan bebas" kecuali diminta.
> *   **Output Formatting:** Berikan hasil dalam format JSON: `{"ovarium": "kiri/kanan", "folikel_count": int, "volume_ml": float, "status_PCOM": bool}`.

##### Prompting untuk Reasoning

Prompting untuk reasoning berfokus pada bagaimana kita mengarahkan proses berpikir agent, bukan hanya hasil akhir yang dihasilkan. Jika meta-prompt menetapkan siapa agent dan apa perannya, maka prompting untuk reasoning mengatur bagaimana agent melangkah dari sebuah masalah menuju solusi. Pendekatan ini sangat penting ketika agent dihadapkan pada tugas yang tidak bisa diselesaikan dengan satu langkah sederhana, tetapi membutuhkan analisis, perencanaan, dan evaluasi secara berurutan.

Salah satu prinsip utama dalam prompting untuk reasoning adalah dekomposisi masalah. Agent diarahkan untuk memecah sebuah permintaan besar menjadi beberapa sub-masalah yang lebih kecil dan lebih spesifik. Misalnya, dalam tugas perencanaan proyek, agent tidak langsung membuat jadwal akhir, tetapi terlebih dahulu mengidentifikasi tujuan proyek, sumber daya yang tersedia, risiko yang mungkin muncul, dan ketergantungan antar tugas. Dengan memecah masalah seperti ini, agent dapat menghasilkan solusi yang lebih terstruktur dan mudah diikuti.

Pendekatan lain yang sering digunakan adalah perencanaan sebelum bertindak. Melalui prompt, agent diminta untuk menjelaskan rencana atau strategi yang akan ia gunakan sebelum memanggil tools atau menghasilkan output utama. Ini membantu memastikan bahwa setiap tindakan yang diambil memiliki tujuan yang jelas dan relevan dengan konteks masalah. Dalam sistem multi-agent, langkah ini juga memudahkan orchestrator atau agent lain untuk memahami alasan di balik sebuah keputusan.

Prompting untuk reasoning juga mencakup refleksi setelah bertindak. Setelah agent menggunakan sebuah tool atau menghasilkan sebuah hasil, ia diarahkan untuk mengevaluasi apakah hasil tersebut sudah sesuai dengan tujuan awal. Jika belum, agent dapat diminta untuk menyesuaikan strategi, mencari informasi tambahan, atau mencoba pendekatan lain. Pola ini mirip dengan siklus belajar manusia, di mana kita mencoba, menilai hasilnya, lalu memperbaiki langkah berikutnya.

Selain itu, prompting untuk reasoning dapat digunakan untuk mengelola banyak kriteria dan prioritas. Dalam situasi di mana agent harus menyeimbangkan beberapa faktor, seperti kecepatan, akurasi, dan biaya, prompt dapat mengarahkan agent untuk secara eksplisit mempertimbangkan setiap faktor tersebut sebelum mengambil keputusan. Dengan cara ini, hasil yang dihasilkan tidak hanya benar secara teknis, tetapi juga sesuai dengan kebutuhan sistem atau pengguna.

Dalam praktik, prompting untuk reasoning sering dipadukan dengan pola kerja seperti ReAct, Chain of Thought, atau Plan and Execute. Prompt dapat dirancang agar agent selalu melalui tahapan berpikir, bertindak, dan mengevaluasi secara berulang, sehingga proses penyelesaian masalah menjadi lebih transparan dan terkontrol.

Secara keseluruhan, prompting untuk reasoning membantu mengubah agent dari sekadar “mesin penjawab” menjadi pemecah masalah yang sistematis. Dengan arahan yang tepat, agent tidak hanya memberikan solusi, tetapi juga menunjukkan proses logis di balik solusi tersebut, sehingga hasilnya lebih dapat dipercaya, lebih mudah dipahami, dan lebih mudah dikembangkan dalam sistem yang lebih besar dan kompleks.

#### Optimasi Deskripsi dan Seleksi Tools

Optimasi deskripsi dan seleksi tools bertujuan memastikan bahwa agent tidak hanya memiliki akses ke berbagai kemampuan, tetapi juga memahami kapan dan bagaimana kemampuan tersebut harus digunakan. Dalam banyak sistem, kegagalan agent bukan disebabkan oleh keterbatasan model bahasa, melainkan karena agent salah memilih tool, menggunakan tool yang tidak relevan, atau memanggil terlalu banyak tool tanpa kebutuhan yang jelas.

Deskripsi tools berfungsi sebagai petunjuk mental bagi agent. Model bahasa membaca deskripsi ini untuk memutuskan apakah sebuah tool cocok digunakan dalam konteks tertentu. Jika deskripsi terlalu umum atau ambigu, agent akan kesulitan membedakan peran satu tool dengan tool lainnya.

![Optimasi Tools](https://assets.cdn.dicoding.com/original/academy/dos-d138acc552644ad027b1efa37a91e78320260429133841.png)

Deskripsi yang baik biasanya mencakup beberapa elemen utama, seperti:

*   Tujuan utama tool, yaitu masalah apa yang paling tepat diselesaikan oleh tool tersebut.
*   Kondisi penggunaan, yaitu kapan tool sebaiknya digunakan dan kapan sebaiknya dihindari.
*   Jenis input yang diharapkan, misalnya kata kunci, kalimat, atau format data tertentu.
*   Bentuk output yang dihasilkan, seperti teks, tabel, atau status eksekusi.

Dengan informasi ini, agent dapat membangun pemahaman yang lebih kontekstual, bukan sekadar mengetahui bahwa sebuah tool “ada,” tetapi juga memahami perannya dalam alur kerja.

Setiap agent biasanya dirancang untuk tujuan tertentu, seperti riset, analisis, atau eksekusi tindakan. Oleh karena itu, seleksi tools sebaiknya dilakukan dengan mempertimbangkan kesesuaian antara fungsi tool dan peran agent. Agent analisis, misalnya, lebih membutuhkan tool untuk pemrosesan data dan visualisasi, sementara agent layanan pelanggan lebih membutuhkan tool untuk mengakses basis data pelanggan atau sistem tiket.

Dengan menyelaraskan tools dan tujuan, agent dapat bekerja lebih fokus dan tidak tergoda untuk menggunakan tool yang sebenarnya tidak relevan dengan tugasnya.

Memberikan terlalu banyak tools dengan fungsi yang mirip dapat membingungkan agent. Misalnya, jika terdapat beberapa tool pencarian yang hasilnya serupa, agent harus “memilih” di antara opsi yang tidak memiliki perbedaan jelas. Hal ini bisa memperlambat proses reasoning dan meningkatkan risiko pemanggilan tool yang tidak efisien.

Seleksi tools yang baik berarti meminimalkan tumpang tindih dan hanya menyediakan tool yang benar-benar dibutuhkan untuk skenario yang ditargetkan. Dengan lingkungan yang lebih sederhana, agent dapat mengambil keputusan lebih cepat dan lebih konsisten.

Selain deskripsi dan jumlah, penting juga untuk mengatur urutan dan prioritas penggunaan tools. Agent dapat diarahkan, misalnya, untuk selalu memeriksa sumber internal terlebih dahulu sebelum memanggil layanan eksternal, atau untuk mencoba pendekatan yang lebih murah dan cepat sebelum menggunakan tool yang mahal dan lambat.

Strategi ini dapat dituangkan dalam meta-prompt atau aturan sistem, sehingga agent memiliki panduan yang jelas dalam menentukan langkah-langkahnya.

Optimasi tools memiliki dampak langsung pada tiga aspek utama kinerja agent. Dari sisi kualitas hasil, agent lebih mungkin menggunakan sumber yang tepat dan relevan. Dari sisi latency, jumlah pemanggilan tool yang tidak perlu dapat dikurangi, sehingga waktu respons menjadi lebih cepat. Dari sisi cost, penggunaan API atau layanan berbayar dapat dikendalikan karena agent hanya memanggilnya ketika benar-benar dibutuhkan.

Pada akhirnya, tools bukan sekadar tambahan teknis, tetapi bagian dari desain sistem agent secara keseluruhan. Cara kita mendeskripsikan, memilih, dan mengatur tools akan sangat memengaruhi perilaku agent di dunia nyata. Dengan optimasi yang baik, agent dapat bertindak lebih cerdas, lebih efisien, dan lebih selaras dengan tujuan bisnis atau kebutuhan pengguna.

Pendekatan ini membantu memastikan bahwa agent tidak hanya mampu “melakukan banyak hal,” tetapi mampu melakukan hal yang tepat pada waktu yang tepat, yang merupakan inti dari kecerdasan dalam sistem berbasis agent.

### Fine-Tuning LLM Agent

Berbeda dengan fine-tuning model bahasa pada umumnya, fine-tuning dalam konteks agent lebih menekankan pada pembelajaran berbasis pengalaman. Pendekatan ini tidak selalu mengubah bobot model, tetapi memperkaya cara agent bertindak melalui memori dan kasus-kasus yang pernah dihadapi sebelumnya.

#### Konsep Fine-Tuning LLM Agent dari Paper Memento

Paper Memento memperkenalkan gagasan bahwa agent dapat “belajar” dari pengalaman sebelumnya dengan cara menyimpan dan menggunakan kembali solusi yang pernah berhasil. Pendekatan ini sering disebut sebagai Case-Based Reasoning.

##### Case-Based Reasoning (CBR)

Case-Based Reasoning adalah pendekatan pembelajaran di mana agent belajar dari pengalaman masa lalu dengan cara menyimpan dan menggunakan kembali solusi yang pernah berhasil. Alih-alih mengandalkan aturan tetap atau model yang terus dilatih ulang, CBR memanfaatkan kumpulan kasus sebagai sumber pengetahuan praktis yang berkembang seiring waktu.

Dalam konteks agent, sebuah kasus bukan hanya catatan tentang pertanyaan dan jawaban. Kasus biasanya berupa rekaman lengkap dari satu episode pemecahan masalah yang mencakup beberapa elemen penting seperti berikut.

*   Konteks masalah, yaitu latar belakang atau kondisi saat tugas diberikan, termasuk tujuan pengguna, dan batasan yang ada.
*   Langkah-langkah penyelesaian, yaitu urutan tindakan yang diambil agent, mulai dari penalaran awal, pemanggilan tools, hingga keputusan akhir.
*   Tools dan sumber yang digunakan, misalnya API tertentu, basis data internal, atau hasil pencarian eksternal.
*   Hasil dan evaluasi, yaitu apakah solusi tersebut berhasil, serta catatan tentang kualitas hasil atau umpan balik dari pengguna.

Dengan struktur seperti ini, setiap kasus menjadi semacam cerita pengalaman yang bisa dipelajari kembali oleh agent di kemudian hari.

CBR biasanya dijalankan dalam sebuah siklus yang terdiri dari beberapa tahap. Pertama adalah retrieve, yaitu mencari kasus lama yang paling mirip dengan masalah baru. Pencarian ini sering dilakukan secara semantik sehingga agent tidak hanya mencocokkan kata kunci, tetapi juga makna dan pola situasi.

Tahap berikutnya adalah reuse, di mana agent mengambil solusi dari kasus yang relevan dan menyesuaikannya dengan kondisi saat ini. Penyesuaian ini penting karena jarang ada dua masalah yang benar-benar identik. Agent mungkin perlu mengganti beberapa parameter, menggunakan tool yang berbeda, atau menyesuaikan format output.

Setelah solusi diterapkan, agent masuk ke tahap revise, yaitu mengevaluasi apakah solusi tersebut benar-benar berhasil. Jika ada kekurangan, agent dapat memperbaiki atau menyempurnakan langkah-langkahnya.

Tahap terakhir adalah retain yang menyimpan kembali pengalaman baru sebagai kasus baru atau sebagai pembaruan dari kasus lama. Dengan cara ini, memori agent terus bertambah dan semakin kaya.

![Case-Based Reasoning Cycle](https://assets.cdn.dicoding.com/original/academy/dos-29185e88e5e49cc54d3531dd2bbb419d20260326104251.png)

Salah satu manfaat utama CBR adalah kecepatan. Untuk masalah yang memiliki pola serupa dengan pengalaman sebelumnya, agent tidak perlu melakukan reasoning dari awal. Ia cukup mengambil solusi yang sudah ada dan menyesuaikannya sehingga waktu respons menjadi lebih cepat.

Selain itu, CBR juga meningkatkan konsistensi. Jika sebuah pendekatan sudah terbukti berhasil di masa lalu, agent cenderung menggunakan pendekatan yang sama untuk kasus serupa sehingga hasil yang diberikan lebih stabil dan dapat diprediksi.

CBR juga mendukung pembelajaran berkelanjutan. Setiap interaksi baru berpotensi menjadi sumber pengetahuan baru sehingga agent dapat berkembang seiring waktu tanpa harus melalui proses pelatihan ulang model yang mahal.

Berbeda dengan sistem berbasis aturan, CBR tidak mengharuskan pengembang untuk mendefinisikan semua skenario dan solusi di awal. Sistem dapat tumbuh secara organik berdasarkan pengalaman nyata.

Dibandingkan dengan fine-tuning model, CBR lebih ringan dari sisi sumber daya. Tidak ada proses pelatihan ulang yang kompleks, tetapi tetap memberikan efek “pembelajaran” pada perilaku agent melalui memori dan strategi.

Meskipun kuat, CBR juga memiliki tantangan. Salah satunya adalah pengelolaan memori kasus. Jika terlalu banyak kasus disimpan tanpa seleksi, sistem bisa menjadi lambat atau mengambil kasus yang kurang relevan.

Tantangan lain adalah penilaian kualitas kasus. Agent perlu mekanisme untuk menentukan kasus mana yang layak disimpan, mana yang perlu diperbarui, dan mana yang sebaiknya dihapus.

Dalam sistem Agentic AI, CBR tidak hanya dimiliki oleh satu agent, tetapi bisa menjadi memori bersama yang dimanfaatkan oleh banyak agent. Dengan cara ini, pengalaman dari satu agent dapat membantu agent lain, menciptakan pembelajaran kolektif di tingkat sistem.

Secara keseluruhan, Case-Based Reasoning menjembatani konsep antara sistem yang “hanya menjalankan perintah” dan sistem yang belajar dari pengalamannya sendiri. Ini menjadikan CBR sebagai salah satu fondasi penting dalam membangun agent yang adaptif, efisien, dan semakin andal seiring waktu.

##### Perbedaan dengan Fine-Tuning LLM Biasa

Perbedaan utama antara fine-tuning LLM biasa dan pendekatan fine-tuning berbasis agent, seperti CBR, terletak pada apa yang sebenarnya dipelajari dan diubah di dalam sistem.

Pada fine-tuning LLM biasa, yang diubah adalah bobot internal model bahasa. Model dilatih ulang menggunakan dataset tambahan yang relevan dengan domain atau tugas tertentu. Setelah proses ini selesai, perilaku model secara global berubah. Artinya, setiap kali model digunakan, baik oleh satu agent maupun banyak agent, ia akan membawa “pengetahuan baru” tersebut ke semua konteks.

Sebaliknya, pada fine-tuning agent berbasis CBR, model bahasa tetap tidak berubah. Yang berkembang adalah sistem di sekeliling model, terutama memori kasus, aturan penggunaan tools, dan strategi pemecahan masalah. Agent menjadi lebih pintar bukan karena modelnya “lebih tahu,” tetapi karena ia memiliki pengalaman yang bisa dirujuk kembali.

![LLM Fine Tuning vs Agent Fine Tuning 1](https://assets.cdn.dicoding.com/original/academy/dos-8205f37b585f26f0ba51680085bee97720260429133900.png)

Fine-tuning LLM biasanya dilakukan secara offline. Pengembang mengumpulkan dataset, menyiapkan pipeline pelatihan, menjalankan proses training yang memakan waktu dan biaya, lalu mendistribusikan model baru ke sistem produksi. Proses ini bersifat diskrit dan tidak terjadi setiap saat.

Sebaliknya, pendekatan berbasis CBR mendukung pembelajaran online atau berkelanjutan. Setiap interaksi yang dianggap penting atau berhasil dapat langsung disimpan sebagai kasus baru. Dengan begitu, agent dapat belajar dari pengalaman terbaru hampir secara real-time tanpa menunggu siklus pelatihan model.

![LLM Fine Tuning vs Agent Fine Tuning 2](https://assets.cdn.dicoding.com/original/academy/dos-866c6fef94e811cf8b5ad66047dc1b1720260429133918.png)

Model yang sudah di-*fine-tune* cenderung lebih kaku dalam arti bahwa perubahan perilakunya bergantung pada dataset yang digunakan saat pelatihan. Jika lingkungan atau kebutuhan pengguna berubah, diperlukan proses fine-tuning ulang agar model tetap relevan.

Pendekatan CBR jauh lebih fleksibel. Agent dapat beradaptasi dengan cepat terhadap pola baru, preferensi pengguna, atau jenis tugas yang berbeda, cukup dengan menyimpan dan memanfaatkan kasus baru. Hal ini sangat berguna dalam sistem yang dinamis, seperti layanan pelanggan atau sistem operasional yang terus berubah.

Fine-tuning LLM memerlukan sumber daya komputasi yang besar, seperti GPU atau TPU, serta biaya operasional yang tidak kecil. Selain itu, ada biaya tambahan untuk menyimpan, mengelola, dan mendistribusikan versi model yang berbeda.

CBR, di sisi lain, lebih ringan dari sisi komputasi. Biaya utama terletak pada penyimpanan memori kasus dan mekanisme pencarian semantik. Meskipun ini tetap membutuhkan infrastruktur, skalanya biasanya lebih kecil dibandingkan dengan melatih ulang model besar.

Ketika bobot model diubah melalui fine-tuning, ada risiko perilaku model berubah di luar yang diharapkan, misalnya kehilangan kemampuan umum tertentu atau munculnya bias baru. Oleh karena itu, proses ini biasanya memerlukan pengujian dan validasi yang ketat.

Pada pendekatan berbasis CBR, perubahan perilaku lebih terlokalisasi. Dampaknya pada jenis kasus yang disimpan dan strategi yang dipilih agent. Jika terjadi kesalahan, Anda dapat memperbaiki atau menghapus kasus tertentu tanpa harus menyentuh keseluruhan sistem.

Dalam sistem multi-agent, fine-tuning LLM biasa berarti semua agent berbagi model yang sama. Setiap perubahan pada model akan memengaruhi seluruh agent secara serempak.

Sebaliknya, pendekatan CBR memungkinkan pembelajaran kolektif yang lebih granular. Agent tertentu bisa berbagi memori kasus dengan agent lain, atau bahkan memiliki memori khusus sesuai perannya. Dengan cara ini, sistem dapat berkembang sebagai sebuah ekosistem pembelajaran, bukan hanya sebagai satu model tunggal yang diubah.

Secara ringkas, fine-tuning LLM biasa mengubah "otak" model, sementara fine-tuning agent berbasis CBR memperkaya "pengalaman" dan strategi sistem. Keduanya memiliki peran berbeda namun dapat dikombinasikan untuk menyeimbangkan pengetahuan mendalam dengan adaptasi cepat.

##### AgentFly

AgentFly merupakan sebuah kerangka konseptual dan arsitektural untuk membangun agent yang tidak hanya pintar saat ini, tetapi juga menjadi semakin pintar seiring waktu. Alih-alih berfokus pada peningkatan kemampuan model bahasa di level bobot (seperti fine-tuning LLM), AgentFly menempatkan kecerdasan utamanya pada siklus pengalaman, evaluasi, dan reuse strategi yang terjadi di lapisan sistem agent.

Pada dasarnya, AgentFly memandang setiap interaksi agent dengan dunia sebagai sebuah eksperimen kecil. Ketika agent menerima tugas, ia akan

*   memahami konteks dan tujuan;
*   menyusun rencana atau strategi penyelesaian;
*   memilih dan menggunakan tools yang tersedia;
*   menghasilkan output atau tindakan; dan
*   mengevaluasi hasilnya.

Semua proses ini tidak berhenti pada tugas selesai, tetapi dicatat sebagai pengalaman yang bisa dipelajari kembali di masa depan.

Salah satu elemen kunci AgentFly adalah memori pengalaman terstruktur. Setiap kasus yang disimpan biasanya mencakup beberapa komponen berikut.

*   Konteks masalah, seperti jenis tugas, domain, batasan, dan tujuan pengguna.
*   Rencana dan langkah-langkah, cara agent memecah masalah dan urutan tindakan yang diambil.
*   Tools yang digunakan, waktu penggunaan, dan tujuannya.
*   Hasil akhir atau output yang dihasilkan dan apakah tujuan tercapai.
*   Evaluasi kualitas, seperti penilaian eksplisit (skor, feedback pengguna) atau implisit (keberhasilan atau kegagalan).

Dengan struktur ini, memori agent tidak sekadar menjadi log, tetapi berubah menjadi basis pengetahuan yang bisa ditelusuri dan dimanfaatkan kembali.

AgentFly menekankan pentingnya evaluasi hasil sebagai bagian dari siklus belajar. Evaluasi ini bisa dilakukan dengan beberapa cara, misalnya:

*   Feedback pengguna: apakah solusi dianggap membantu atau tidak.
*   Metrik otomatis: akurasi, waktu penyelesaian, jumlah iterasi, atau biaya penggunaan tool.
*   Self-reflection: agent membandingkan rencana awal dengan hasil akhir untuk menilai efektivitas strateginya.

Hasil evaluasi ini kemudian menjadi label kualitas bagi kasus yang disimpan. Dengan begitu, agent tidak hanya tahu “apa yang pernah dilakukan,” tetapi juga “apa yang terbukti berhasil dan apa yang sebaiknya dihindari.”

Ketika agent menghadapi masalah baru, AgentFly tidak langsung memulai dari nol. Ia akan:

*   Mencari kasus serupa di memori berdasarkan konteks dan tujuan.
*   Membandingkan strategi yang pernah digunakan sebelumnya.
*   Menyesuaikan solusi lama dengan kondisi baru (adaptasi, bukan sekadar menyalin).

Proses ini membuat agent semakin konsisten dan efisien. Untuk tugas yang berulang atau berpola, waktu perencanaan bisa jauh lebih singkat karena agent sudah memiliki template strategi yang terbukti efektif.

Keunggulan utama AgentFly adalah kemampuannya untuk membuat agent berkembang tanpa perlu mengubah model bahasa. Model LLM tetap berperan sebagai “mesin penalaran dan generasi bahasa,” tetapi kecerdasan adaptif muncul dari:

*   Memori kasus yang terus bertambah.
*   Strategi yang semakin matang.
*   Aturan pemilihan tools yang semakin presisi.

Dengan kata lain, AgentFly memisahkan antara kapabilitas dasar (LLM) dan pengalaman praktis (memori dan strategi agent). Hal ini membuat sistem lebih mudah dirawat, lebih murah untuk dikembangkan, dan lebih cepat beradaptasi.

Dalam aplikasi nyata, seperti layanan pelanggan, analisis data, atau perencanaan proyek, pendekatan AgentFly memungkinkan agent untuk

*   Mengingat solusi yang berhasil untuk tipe masalah tertentu;
*   Menghindari jalur penyelesaian yang terbukti tidak efektif; dan
*   Menyesuaikan gaya dan strategi sesuai preferensi pengguna atau konteks organisasi.

Seiring waktu, agent akan terlihat “lebih berpengalaman,” bukan karena modelnya dilatih ulang, tetapi karena sistemnya tumbuh melalui akumulasi pengalaman.

Secara konseptual, AgentFly berada di persimpangan antara Case-Based Reasoning (CBR) sebagai fondasi memori dan reuse solusi, dan Agentic AI modern yang menekankan otonomi, perencanaan, dan penggunaan tools.

![AgentFly Concept](https://assets.cdn.dicoding.com/original/academy/dos-0ddaa0afe15ac483ab1f5938a6939a4520260429133934.png)

Dengan menggabungkan keduanya, AgentFly mendorong lahirnya agent yang bukan hanya reaktif, tetapi reflektif dan evolutif, mampu belajar dari masa lalu untuk membuat keputusan yang lebih baik di masa depan.

Sekarang mari kita beranjak sedikit membahas arsitektur Memento. *Hmm*, mungkin Anda bertanya, apa kaitan antara AgentFly dengan Memento? Jika AgentFly adalah cara berpikir sistem yang menekankan bahwa agen harus belajar dari pengalaman, maka Arsitektur Memento adalah infrastruktur teknis untuk mewujudkannya.

Memento menyediakan komponen seperti Case Memory dan mekanisme Writing/Reading yang dibutuhkan agar konsep AgentFly dapat berjalan secara otomatis dan real-time dalam sistem produksi.

#### Arsitektur Memento

*Yup*, untuk mendukung pembelajaran berbasis pengalaman, arsitektur Memento hadir dengan memperkenalkan beberapa komponen utama yang berfokus pada pengelolaan memori.

Memento sendiri adalah sebuah memory-centric architecture untuk AI Agent yang dirancang untuk memfasilitasi pembelajaran berkelanjutan dan real-time (adaptasi berkelanjutan) tanpa memerlukan penyesuaian ulang LLM yang mahal dan berulang.

![Memento Architecture](https://assets.cdn.dicoding.com/original/academy/dos-8fbfa0a14854780307b98ac59fe8b1bf20260429133954.png)

##### Case Memory

Dalam arsitektur Memento, Case Memory berperan sebagai ingatan jangka panjang bagi agent. Jika model bahasa dapat diibaratkan sebagai otak yang berpikir dan berbicara, Case Memory adalah catatan pengalaman yang menyimpan seluruh aktivitas agent, metode pelaksanaannya, dan tingkat keberhasilannya. Komponen ini memungkinkan agent untuk belajar dari masa lalu tanpa harus mengubah bobot model bahasa itu sendiri.

![Case Memory](https://assets.cdn.dicoding.com/original/academy/dos-cfaba00480701fdad5c7a06a0e5552ff20260429134008.png)

Setiap kasus yang disimpan di case memory biasanya tidak hanya berupa teks mentah, tetapi sebuah entri terstruktur yang mencakup beberapa lapisan informasi, antara lain:

*   **Deskripsi masalah**: ringkasan tentang apa yang diminta pengguna atau tujuan utama dari tugas tersebut.
*   **Konteks**: kondisi tambahan seperti batasan waktu, domain ( bisnis, pendidikan, teknis), preferensi pengguna, atau lingkungan sistem saat tugas dijalankan.
*   **Rencana dan langkah-langkah**: bagaimana agent memecah masalah menjadi sub-tugas dan urutan tindakan yang diambil.
*   **Tools dan sumber daya**: tool apa saja yang digunakan, kapan digunakan, dan alasan pemilihannya.
*   **Hasil akhir**: output yang dihasilkan, apakah tujuan tercapai, dan dalam bentuk apa solusi disajikan.
*   **Evaluasi atau umpan balik**: penilaian kualitas, baik dari pengguna, sistem, atau hasil metrik otomatis.

Dengan struktur seperti ini, setiap kasus menjadi unit pengalaman yang kaya, bukan sekadar log aktivitas.

Salah satu ciri utama case memory dalam Memento adalah kemampuannya untuk melakukan pencarian semantik. Artinya, agent tidak mencari kasus hanya berdasarkan kata kunci yang sama, tetapi berdasarkan kemiripan makna dan konteks.

Secara konseptual, ini biasanya dilakukan dengan beberapa hal berikut.

*   Mengubah deskripsi kasus menjadi representasi vektor (embedding).
*   Menyimpan embedding tersebut dalam sistem pencarian vektor.
*   Ketika masalah baru datang, agent juga membuat embedding dari masalah tersebut.
*   Sistem kemudian mencari kasus lama yang paling dekat secara semantik.

Hasilnya, agent bisa menemukan pengalaman relevan meskipun cara pengguna merumuskan masalahnya berbeda secara bahasa.

![Semantic Search in Case Memory](https://assets.cdn.dicoding.com/original/academy/dos-9ed5270fa6165416f99485192adf45b520260429134026.png)

Case memory tidak hanya digunakan di awal, tetapi dapat terlibat di beberapa tahap siklus agent.

*   Saat perencanaan: agent mencari contoh strategi yang pernah berhasil untuk masalah serupa.
*   Saat eksekusi: agent merujuk pada langkah-langkah yang pernah diambil untuk menghindari kesalahan yang sama.
*   Saat evaluasi: agent membandingkan hasil saat ini dengan hasil dari kasus sebelumnya.

Dengan cara ini, case memory menjadi komponen aktif dalam penalaran, bukan sekadar arsip pasif.

Keuntungan besar dari case memory adalah konsistensi perilaku agent. Untuk jenis masalah yang berulang, agent cenderung

*   menggunakan pola solusi yang sama atau serupa;
*   menyajikan output dengan format yang konsisten; dan
*   memilih tools yang sudah terbukti efektif.

Selain itu, waktu penyelesaian tugas juga bisa lebih cepat karena agent tidak perlu merancang strategi dari nol setiap kali.

Dalam sistem nyata, jumlah kasus bisa bertambah sangat banyak. Oleh karena itu, case memory biasanya dilengkapi dengan mekanisme seperti berikut.

*   Pengelompokan (clustering) kasus berdasarkan domain atau tipe masalah.
*   Peringkat relevansi berdasarkan kualitas atau keberhasilan kasus sebelumnya.
*   Pembersihan memori (pruning) untuk menghapus kasus yang sudah usang atau berkualitas rendah.

Dengan pengelolaan ini, case memory tetap efisien dan tidak penuh oleh pengalaman yang kurang berguna.

Bayangkan sebuah agent layanan pelanggan. Setiap kali ia berhasil menyelesaikan keluhan tertentu, seluruh prosesnya disimpan sebagai satu kasus. Ketika keluhan serupa muncul lagi, agent dapat langsung menemukan kasus lama tersebut dan menyesuaikan solusinya. Hasilnya, respons menjadi lebih cepat, lebih tepat, dan terasa lebih “berpengalaman” bagi pengguna.

Dalam Memento, case memory menjadi jembatan antara masa lalu dan masa kini. Ia menghubungkan penalaran LLM yang bersifat generatif dengan pengalaman konkret yang sudah pernah terjadi. Dengan begitu, agent tidak hanya mengandalkan kemampuan bahasa dan logika model, tetapi juga pengetahuan praktis yang tumbuh seiring waktu.

##### Memory Writing dan Reading

Dalam arsitektur Memento, Memory Writing dan Memory Reading adalah dua mekanisme inti yang membuat agent tidak hanya “berpikir saat ini”, tetapi juga belajar dari pengalaman masa lalu dan memanfaatkannya di masa depan. Keduanya membentuk siklus pembelajaran berkelanjutan yang mirip dengan cara manusia mengingat, merefleksikan, lalu menggunakan ingatan tersebut ketika menghadapi situasi baru.

Memory writing adalah proses ketika agent memutuskan bahwa sebuah interaksi atau penyelesaian tugas layak disimpan sebagai kasus di case memory. Tujuannya bukan mencatat semua yang terjadi, tetapi menyaring pengalaman yang benar-benar bernilai.

Dalam praktiknya, proses ini biasanya melibatkan beberapa tahap.

![Memory Writing Process](https://assets.cdn.dicoding.com/original/academy/dos-2c626e3f68255b84a77fccd8986ebd5a20260429134049.png)

1.  **Case Selection**
    Tidak semua tugas perlu disimpan. Agent atau sistem evaluasi akan menentukan apakah sebuah penyelesaian dianggap:
    *   Berhasil mencapai tujuan pengguna.
    *   Menghadapi masalah yang kompleks atau tidak biasa.
    *   Menghasilkan strategi atau penggunaan tools yang baru dan efektif.
    Kasus-kasus yang rutin, gagal, atau tidak memberikan pembelajaran baru sering kali diabaikan agar memori tetap ringkas dan relevan.
2.  **Information Extraction**
    Dari seluruh interaksi, agent merangkum bagian-bagian kunci seperti berikut.
    *   Inti masalah dan konteksnya.
    *   Langkah-langkah utama yang diambil.
    *   Keputusan penting, misalnya pemilihan tools atau perubahan rencana di tengah jalan.
    *   Hasil akhir dan evaluasi kualitasnya.
    Tahap ini penting agar memori yang disimpan tidak terlalu panjang, tetapi tetap kaya makna.
3.  **Semantic Representation**
    Informasi yang sudah diringkas kemudian diubah menjadi bentuk yang mudah dicari kembali, biasanya dengan membuat embedding atau representasi vektor. Dengan cara ini, memori bisa diakses berdasarkan kemiripan makna, bukan hanya kecocokan kata.
4.  **Storage & Indexing**
    Kasus yang sudah diproses disimpan ke dalam basis data memori, lengkap dengan metadata seperti domain, tingkat keberhasilan, waktu penyimpanan, atau jenis tugas. Metadata ini membantu proses pencarian dan pemeringkatan di kemudian hari.

Secara konseptual, memory writing adalah tahap di mana agent mengubah pengalaman menjadi pengetahuan jangka panjang.

Jadi, secara konseptual, Memory Writing adalah tahap di mana *agent* mengkurasi dan mengubah pengalaman mentah menjadi pengetahuan jangka panjang yang terstruktur.

Di sisi lain, agar pengetahuan tersebut tidak hanya disimpan dan menjadi arsip pasif, sistem membutuhkan mekanisme Memory Reading. Jika *Writing* adalah cara *agent* mencatat pelajaran, maka Memory Reading adalah proses di mana *agent* menghadapi tugas baru dan mencoba mencari serta mengambil kembali pengalaman lama yang paling relevan untuk digunakan sebagai referensi.

Proses ini biasanya berjalan seperti gambar berikut.

![Memory Reading Process](https://assets.cdn.dicoding.com/original/academy/dos-e7a5afd7ee3b8dbd6eb92cecf3935fe920260429134107.png)

1.  **Understand New Problem**
    Agent merangkum permintaan pengguna menjadi representasi inti, termasuk tujuan, konteks, dan batasan.
2.  **Semantic Search in Case Memory**
    Representasi masalah baru dibandingkan dengan representasi kasus-kasus lama untuk menemukan beberapa kasus yang paling mirip secara makna dan konteks.
3.  **Rank Relevant Cases**
    Dari hasil pencarian, agent memilih kasus yang:
    *   Paling relevan dengan situasi saat ini.
    *   Memiliki tingkat keberhasilan tinggi.
    *   Berasal dari domain atau kondisi yang serupa.
4.  **Adapt Past Solutions**
    Agent tidak menyalin solusi lama secara mentah, tetapi menyesuaikannya dengan kondisi baru. Misalnya, langkah-langkah yang sama bisa digunakan, tetapi dengan tool yang berbeda atau format output yang disesuaikan dengan kebutuhan pengguna saat ini.

Dalam alur kerja agent, memory reading biasanya terjadi sebelum perencanaan atau di tengah proses penalaran, sehingga pengalaman lama benar-benar memengaruhi cara agent mengambil keputusan.

Memory reading dan writing tidak berdiri sendiri, tetapi terintegrasi dengan mekanisme penalaran seperti Chain of Thought atau ReAct. Contohnya:

*   Saat merencanakan, agent bisa membaca memori untuk melihat strategi apa yang pernah berhasil.
*   Saat bertindak, agent bisa menyesuaikan langkah berdasarkan pola dari kasus sebelumnya.
*   Setelah selesai, agent menulis kembali pengalaman baru ke memori jika dianggap bernilai.

Dengan mekanisme ini, sistem mendapatkan beberapa keuntungan besar:

*   **Adaptivitas**: agent menjadi lebih fleksibel dan kontekstual dari waktu ke waktu.
*   **Efisiensi**: tugas yang serupa dapat diselesaikan lebih cepat.
*   **Konsistensi**: kualitas dan gaya solusi lebih stabil karena mengikuti pola yang sudah terbukti.
*   **Skalabilitas pengetahuan**: pengalaman dari satu agent bisa dimanfaatkan oleh agent lain dalam sistem multi-agent melalui memori bersama.

Bayangkan seorang teknisi yang mencatat setiap perbaikan yang berhasil dalam sebuah buku. Saat menghadapi masalah baru, ia membuka buku tersebut untuk mencari kasus serupa, lalu menyesuaikan solusi lama dengan kondisi saat ini. Setelah selesai, ia menulis kembali pengalaman barunya.

Memory writing dan reading bekerja persis seperti buku catatan pengalaman ini, tetapi dalam bentuk sistem digital yang dapat diakses dan diproses oleh agent secara otomatis.

##### Online Learning

Dalam arsitektur Memento, Online Learning merujuk pada kemampuan sistem agent untuk meningkatkan kinerjanya secara berkelanjutan selama digunakan, tanpa harus melatih ulang LLM di baliknya. Fokus utamanya bukan pada perubahan bobot LLM, tetapi pada penyempurnaan memori, strategi, dan cara agent mengambil keputusan berdasarkan pengalaman baru.

Yang benar-benar dipelajari oleh agent dalam online learning adalah hal-hal di tingkat sistem seperti berikut.

*   **Pola tugas**: jenis permintaan apa yang sering muncul dan bagaimana cara paling efektif untuk menanganinya.
*   **Preferensi pengguna**: gaya bahasa, format output, atau tingkat detail yang disukai.
*   **Strategi pemecahan masalah**: urutan langkah, kombinasi tools, atau pendekatan reasoning yang paling sering berhasil.
*   **Kondisi lingkungan**: misalnya perubahan API, sumber data yang sering error, atau tool yang lebih cepat dan stabil.

Semua ini disimpan dan dikelola di dalam memori dan aturan operasional agent, bukan di dalam parameter model.

Proses online learning biasanya terjadi sebagai sebuah siklus yang terus berulang seperti yang ditunjukkan pada gambar berikut.

![Online Learning Cycle](https://assets.cdn.dicoding.com/original/academy/dos-81168f4c986c6684d1551152296c9e7320260429134147.png)

1.  **Interaksi dan Eksekusi Tugas**
    Agent menerima permintaan, melakukan reasoning, memanggil tools, dan menghasilkan output seperti biasa.
2.  **Evaluasi Hasil**
    Setelah tugas selesai, sistem melakukan evaluasi. Evaluasi ini bisa berasal dari:
    *   Umpan balik pengguna (misalnya rating, koreksi, atau komentar).
    *   Metrik otomatis seperti task completion rate, latency, atau kualitas output.
    *   Penilaian internal, misalnya LLM as a judge.
3.  **Ekstraksi Pembelajaran**
    Dari hasil evaluasi, sistem mengidentifikasi apa yang berhasil dan apa yang perlu diperbaiki. Contohnya:
    *   Strategi tertentu menghasilkan jawaban lebih cepat.
    *   Penggunaan tool tertentu sering gagal atau mahal.
    *   Format laporan tertentu lebih disukai pengguna.
4.  **Update Memori dan Strategi**
    Pembelajaran ini kemudian:
    *   disimpan sebagai kasus baru di case memory,
    *   digunakan untuk memperbarui aturan pemilihan tools, dan
    *   memengaruhi cara agent merencanakan langkah ke depan.

Siklus ini berjalan terus-menerus, sehingga setiap interaksi memperkaya “pengalaman” sistem.

Dalam Agentic AI, online learning tidak hanya terjadi pada satu agent, tetapi bisa dibagikan ke seluruh sistem melalui memori bersama atau orchestrator. Misalnya:

*   Jika agent analisis menemukan strategi baru yang lebih efisien, informasi itu bisa disimpan dan dimanfaatkan oleh agent lain.
*   Meta-agent dapat menyesuaikan cara membagi tugas berdasarkan performa agent-agent sebelumnya.

Dengan cara ini, seluruh tim agent berkembang bersama, bukan hanya satu komponen saja seperti pada fine-tuning LLM. Pada pelatihan atau fine-tuning LLM tradisional,

*   Perubahan terjadi di tingkat bobot model;
*   Prosesnya berat, mahal, dan biasanya dilakukan secara offline; dan
*   Hasilnya bersifat global dan tidak selalu spesifik ke konteks pengguna tertentu.

Namun, pada online learning berbasis memori,

*   Perubahan terjadi di tingkat perilaku sistem;
*   Prosesnya ringan dan terjadi selama sistem berjalan; dan
*   Hasilnya sangat kontekstual sesuai dengan lingkungan dan pengguna tertentu.

Beberapa manfaat utama pendekatan ini adalah sebagai berikut.

*   Adaptivitas tinggi sehingga sistem cepat menyesuaikan diri dengan perubahan kebutuhan dan kondisi.
*   Personalisasi pengalaman pengguna menjadi semakin sesuai dengan preferensi masing-masing.
*   Efisiensi biaya bisa dimaksimalkan karena tidak perlu sering melakukan retraining atau fine-tuning model.
*   Peningkatan kualitas performa agent cenderung membaik seiring waktu dan penggunaan.

Contoh nyata dari mekanisme ini dapat kita temukan pada aplikasi AI seperti Gemini atau ChatGPT. Pernahkah Anda merasa AI tersebut semakin memahami gaya bahasa atau format jawaban yang Anda sukai? Hal ini terjadi bukan karena model dasarnya dilatih ulang secara terus-menerus, melainkan karena adanya sistem Online Learning berbasis memori yang menyimpan preferensi Anda dan menggunakannya kembali sebagai referensi di masa depan.

Alright, kita telah mempelajari bahwa membuat agent lebih pintar tidak selalu harus dimulai dari sisi model. Melalui prompt engineering, optimasi tools, dan pengelolaan memori berbasis pengalaman, kita dapat membangun sistem agent yang lebih adaptif, efisien, dan konsisten.

Selanjutnya, kita akan membuat latihan multi-agent yang menjadi lanjutan dari latihan pertama.

---

## Latihan: Membuat Multi-Agent Menggunakan CrewAI

Pada latihan pertama, Anda membangun satu agent yang bertugas mencari tren dan mengemasnya menjadi data terstruktur. Di titik itu, sistem Anda sudah bisa berpikir, bertindak, dan menghasilkan sesuatu yang bisa dipakai lagi.

Sekarang, kita tidak menambah kecerdasan agent. Kita menambah cara kerjanya. Di dunia nyata, jarang ada satu sistem AI yang mengerjakan semuanya sendirian. Biasanya ada pembagian peran, seperti mencari data, menyusun laporan, dan menyiapkan hasil akhir untuk dibaca manusia atau sistem lain.

Latihan ini meniru pola itu, tapi dalam versi yang ringan dan bisa Anda pahami ujung ke ujung. Di sini, Anda akan membangun dua agent yang bekerja berurutan:

*   Agent pertama fokus pada mencari dan menyaring informasi.
*   Agent kedua fokus pada mengubah hasil riset menjadi artikel yang enak dibaca.

CrewAI akan bertindak sebagai koordinator yang memastikan mereka tidak saling tumpang tindih, tidak lompat langkah, dan benar-benar meneruskan hasil kerja satu sama lain.

### Install Library: Menyiapkan Lingkungan Agent

```python
!pip install -q -U crewai litellm langchain-groq duckduckgo-search apscheduler "pydantic[email]" langchain-community
!pip install fastapi fastapi-sso uvicorn
```

Kalau di latihan pertama kita menyiapkan agent supaya bisa berpikir dan mencari, di sini kita menyiapkan sesuatu yang lebih besar, yaitu kerangka kerja untuk tim agent.

```python
import os
from google.colab import userdata
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
```

Beberapa library di atas mungkin belum semuanya Anda pakai langsung. Itu sengaja. Di dunia industri, sistem jarang dibangun hanya untuk hari ini. Biasanya sudah disiapkan agar nanti bisa:

*   dijalankan sebagai service,
*   dihubungkan ke aplikasi lain, atau
*   ditambah agent baru tanpa harus mengubah semuanya dari nol.

Anggap saja bagian ini seperti menyiapkan ruang kerja sebelum tim masuk, meja, listrik, dan jaringan sudah siap, meskipun hari ini Anda baru menempati satu sudutnya.

### Menghubungkan Sistem ke Dunia Luar

```python
os.environ["GROQ_API_KEY"] = userdata.get('GROQ_API_KEY')
os.environ["OPENAI_API_KEY"] = "sk-dummy"
```

Sama seperti latihan pertama, agent kita tetap bergantung pada layanan di luar sistem lokal. Bedanya, sekarang kita menyiapkan sistem yang secara desain bisa berbicara dengan lebih dari satu penyedia model.

Walaupun kita hanya benar-benar memakai Groq, CrewAI mengharapkan ada konfigurasi untuk OpenAI juga. Oleh karena itu, kita beri nilai dummy agar alurnya tetap berjalan.

Ini mencerminkan situasi yang sering terjadi di sistem produksi: satu kode, banyak kemungkinan backend. Hari ini pakai A, besok bisa pindah ke B, tanpa perlu membongkar seluruh arsitektur.

### Menentukan Model LLM yang Digunakan

```python
my_llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    temperature=0.0,
    api_key=userdata.get('GROQ_API_KEY')
)
```

Di titik ini, Anda sedang menentukan satu hal penting, yaitu cara seluruh tim agent berpikir. Dengan satu objek my_llm, Anda membuat keputusan bahwa semua agent akan berbagi model gaya respons, dan tingkat kreativitas yang sama.

Temperature dibuat rendah bukan karena kita tidak ingin hasil yang menarik, tetapi karena kita ingin hasil yang stabil dan bisa diandalkan. Dalam alur kerja berbasis pipeline, konsistensi sering lebih berharga daripada variasi.

### Tool: Cara Agent Meng-explore Dunia Luar

```python
search_instance = DuckDuckGoSearchRun()
@tool("web_search")
def web_search(query: str):
    """Bermanfaat untuk mencari data real-time di internet.
    Gunakan kata kunci pencarian yang spesifik dan singkat."""
    return search_instance.run(query)
```

Tool ini adalah pembeda utama antara agent yang pintar dan agent yang berguna. Tanpa tool, agent hanya bisa mengandalkan pengetahuan yang sudah tertanam di model. Dengan tool, agent bisa:

*   mencari informasi terbaru,
*   menghadapi hasil yang tidak selalu rapi, dan
*   berurusan dengan error atau keterbatasan jaringan.

Di sinilah sistem Anda mulai terasa seperti sistem nyata, bukan sekadar demo AI.

#### Membagi Peran Agent: Siapa Melakukan Apa

##### Researcher: Mencari dan Menyaring

```python
researcher = Agent(
    role='Lead Tech Researcher',
    goal='Mencari dan merangkum 3 tren utama {topic} di tahun 2026',
    backstory='''Kamu adalah peneliti senior di biro teknologi global.
    Keahlianmu adalah menemukan informasi valid dan mengubahnya menjadi laporan poin-poin.''',
    tools=[web_search],
    llm=my_llm,
    verbose=True,
    max_iter=3
)
```

Agent ini tidak peduli bagaimana hasil akhirnya akan dibaca. Fokusnya satu: mendapatkan bahan yang layak dipakai.

Ia bertugas keluar ke internet, mengumpulkan informasi, lalu merangkum dalam bentuk yang lebih padat. Dalam tim manusia, peran ini mirip dengan peneliti atau analis yang menyiapkan brief sebelum materi itu diserahkan ke penulis.

##### Writer: Mengubah Data Menjadi Cerita

```python
writer = Agent(
    role='Tech Content Strategist',
    goal='Menyusun artikel blog edukatif berdasarkan hasil riset teknis',
    backstory='''Kamu adalah penulis blog teknologi yang ahli.
    Gaya bahasamu ramah, profesional, dan mudah dipahami, mirip gaya penulisan Dicoding.''',
    llm=my_llm,
    verbose=True
)
```

Writer tidak lagi mencari fakta. Ia menerima hasil kerja orang lain, dalam hal ini, agent Researcher. Lalu, ia mengubahnya menjadi sesuatu yang nyaman dibaca.

Di sini Anda bisa melihat dengan jelas pemisahan peran:

*   Satu agent bekerja untuk sistem.
*   Satu agent bekerja untuk manusia.

### Task: Cara Agent Saling Menunggu dan Meneruskan Hasil

```python
task_research = Task(
    description='Gunakan tool web_search untuk riset {topic}. Identifikasi 3 inovasi penting untuk tahun 2026.',
    expected_output='Laporan riset berisi 3 poin inovasi utama.',
    agent=researcher
)
 
task_write = Task(
    description='Buat artikel blog 3 paragraf berdasarkan laporan riset dari researcher.',
    expected_output='Artikel blog lengkap dalam format Markdown.',
    agent=writer
)
```

Task ini bisa Anda bayangkan sebagai catatan kerja yang ditempel di papan tim. Task pertama menghasilkan sesuatu. Task kedua tidak bisa mulai sebelum sesuatu itu ada. Dengan cara ini, alur kerja menjadi jelas, bahkan tanpa melihat ke dalam kode masing-masing agent.

### Crew: Koordinator yang Menjaga Alur Tetap Rapi

```python
my_crew = Crew(
    agents=[researcher, writer],
    tasks=[task_research, task_write],
    process=Process.hierarchical,
    manager_llm=my_llm,
    verbose=True
)
```

Di sinilah semua potongan tadi disatukan. Crew tidak mencari data. Crew tidak menulis artikel. Tugasnya hanya satu, yaitu memastikan urutan kerja dipatuhi. Dengan sequential, Crew mengatakan secara eksplisit: Jangan biarkan Writer bekerja sebelum Researcher selesai.

Konsep sederhana ini adalah dasar dari banyak sistem workflow di industri, baik itu untuk data pipeline, microservices, maupun AI orchestration.

Di sinilah semua potongan tadi disatukan dalam satu orkestrasi terpusat. Crew tidak mencari data. Crew tidak menulis artikel. Perannya bukan sebagai pelaksana teknis, melainkan sebagai pengatur strategi dan pengawas alur kerja. Dengan menggunakan Process.hierarchical, Crew tidak lagi sekadar menjalankan tugas secara berurutan seperti pada mode sequential. Sebaliknya, Crew bertindak untuk:

*   Menentukan urutan eksekusi secara dinamis,
*   Mendelegasikan tugas ke agent yang paling sesuai,
*   Mengatur koordinasi antar agent,
*   Dan memastikan setiap langkah berjalan sesuai tujuan akhir.

Dalam pendekatan ini, terdapat sebuah manager (melalui manager_llm) yang berperan seperti supervisor proyek. Ia memahami gambaran besar, memantau progres, serta memastikan output setiap agent berkontribusi terhadap tujuan akhir.

Konsep ini mencerminkan banyak sistem orkestrasi di industri, seperti:

*   Workflow engine dalam data pipeline,
*   Service orchestrator pada arsitektur microservices,
*   AI supervisor dalam sistem multi-agent modern.

### Menjalankan Sistem

```python
if __name__ == "__main__":
    print("--- [CrewAI Multi-Agent Execution Started] ---")
    try:
        # Menjalankan workflow dengan input topik tertentu
        result = my_crew.kickoff(inputs={'topic': 'Agentic AI Workflow'})
```

Di sini, Anda bukan hanya menekan tombol jalankan agent seperti sebelumnya, tetapi juga menyalakan sebuah alur kerja. Yang terjadi di belakang layar adalah percakapan antarkomponen:

*   Crew memanggil Researcher.
*   Researcher menggunakan tool dan menghasilkan ringkasan.
*   Crew meneruskan hasil itu ke Writer.
*   Writer mengubahnya menjadi artikel.

Yang Anda terima di akhir bukan sekadar teks, tapi hasil dari sebuah proses yang kolaboratif. Keren sekali, bukan?

Kalau Anda lihat latihan AI Agent sederhana dan latihan Multi-Agent ini sebagai satu rangkaian, sebenarnya Anda sedang membangun versi mini dari sistem yang sering dipakai di industri.

*   Agent pertama mengubah informasi di dunia nyata menjadi data terstruktur.
*   Agent kedua mengubah data menjadi narasi.
*   Orchestrator memastikan semuanya terjadi dalam urutan yang benar.

Menariknya lagi, jika ke depannya ingin menambah agent baru, misalnya untuk mengecek kualitas, mengedit gaya bahasa, atau mengirim hasil ke CMS, Anda tidak perlu mengubah sistem yang sudah ada. Anda hanya menambahkan satu peran baru ke dalam tim.

Kode lengkap dan output dari latihan ini dapat Anda lihat pada [notebook](https://colab.research.google.com/drive/13cKFU8N6WUQHlcVyu6lVCa3mCmY9oFhm?usp=sharing) berikut.

Selamat! Anda telah menyelesaikan semua materi dan latihan di Modul AI Agent dan Agentic AI. Selanjutnya, Anda dapat mengeksplorasi berbagai use case yang melibatkan materi ini secara mandiri. *Cheers!*