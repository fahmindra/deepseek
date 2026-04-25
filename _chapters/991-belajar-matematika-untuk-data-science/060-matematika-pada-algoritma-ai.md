---
title: Matematika pada Algoritma AI
slug: matematika-6
abstract: Menelaah kaitan antara konsep matematika dengan algoritma AI.
---

## Pengantar Matematika pada Algoritma AI

Sampai di titik ini, Anda sudah mempelajari banyak hal:

*   mempelajari struktur dan karakteristik data–baik univariat maupun multivariat–beserta ukuran pemusatan dan penyebarannya dalam materi **Data Univariat dan Multivariat**; 
*   kemudian memahami konsep dasar probabilitas hingga Teorema Bayes yang esensial untuk mengukur ketidakpastian dalam materi **Dasar-Dasar Probabilitas**; 
*   lalu belajar berbagai jenis distribusi probabilitas dan penerapannya untuk memodelkan perilaku data dalam materi **Distribusi Probabilitas**; 
*   menaklukkan **statistik inferensial**, jembatan krusial yang memungkinkan kita menarik kesimpulan valid dari sampel menuju populasi;
*   serta melakukan A/B testing untuk dapat memberikan rekomendasi strategi bisnis yang paling sesuai berdasarkan interpretasi hasil analisis data dan pengujian hipotesis dalam materi **Studi Kasus A/B Testing dengan Python**.        

Sekarang, dengan fondasi matematika yang kokoh di tangan Anda, saatnya kita melangkah lebih jauh menuju inti dari sistem cerdas yang mengubah dunia: **algoritma AI**. 

Sering kali, AI dipandang sebagai bidang yang murni komputasional atau hanya tentang "belajar" dari data secara ajaib. Namun, anggapan itu berbeda dengan kenyataan yang akan kita pelajari.

Bayangkan Anda ingin membangun sebuah sistem yang dapat mengenali wajah dari gambar, atau memahami makna dari kalimat yang diucapkan, atau bahkan memprediksi tren harga saham di masa depan. Algoritma di balik semua kemampuan luar biasa ini tidak lain adalah aplikasi canggih dari prinsip-prinsip matematika yang telah Anda pelajari. 

*   Bagaimana sistem mengukur tingkat keyakinannya terhadap suatu prediksi? Itu adalah probabilitas.
*   Bagaimana sistem menangani variasi dan ketidakpastian? Itu adalah konsep distribusi.
*   Bagaimana sebuah sistem "belajar" pola dalam data? Itu adalah proses estimasi statistik. 

Pada materi ini, kita akan membongkar "kotak hitam" AI dan melihat bahwa statistika menjadi bahasa universal yang menjelaskan dan memandu setiap operasinya. 

Kita akan secara spesifik menelaah hal berikut.

*   **Computer Vision (Metode Filtering):** Filter gambar, yang esensial untuk mendeteksi tepi atau mengurangi _noise_, secara fundamental adalah operasi statistik yang menganalisis perubahan distribusi intensitas piksel dan variansi lokal.
*   **Natural Language Processing (Metode Ekstraksi Fitur):** Representasi bahwa teks seperti TF-IDF dan Word Embeddings bukan sekadar kode angka, melainkan cerminan cerdas dari probabilitas kemunculan kata, korelasi kontekstual, serta analisis statistik frekuensi.
*   **Time Series (ARIMA):** Model prediksi deret waktu seperti ARIMA sepenuhnya dibangun di atas konsep proses stokastik, autokorelasi, dan inferensi statistik untuk memahami serta memproyeksikan pola data yang bergantung pada waktu.

Jadi, bersiaplah untuk memperdalam pemahaman Anda, tidak hanya tentang hal yang dilakukan algoritma AI, tetapi yang lebih penting: **alasan dan cara statistika untuk melakukannya**.

Ini akan memberdayakan Anda untuk tidak hanya menjadi pengguna AI, tetapi juga seorang data scientist yang dapat mengerti, berinovasi, dan bahkan mengembangkan algoritma AI dengan landasan statistik yang kuat.

## Matematika dalam Computer Vision (CV)

Kita telah menuntaskan pembahasan fundamental statistika–sebagai salah satu cabang matematika–pada materi-materi sebelumnya, mulai dari struktur data univariat dan multivariat, memahami konsep dasar probabilitas yang membentuk dasar ketidakpastian, menyelami berbagai distribusi probabilitas untuk memodelkan fenomena data, hingga menguasai statistik inferensial yang krusial dalam menarik kesimpulan dari sampel ke populasi. Semua pengetahuan ini adalah fondasi esensial kita.

Sekarang, ayo kita coba aplikasikan pemahaman statistika tersebut pada salah satu domain kunci dalam AI: **computer vision**.

![dos-e90adebb64dd69d48892857c77ac636d20250902223903.jpeg](https://assets.cdn.dicoding.com/original/academy/dos-e90adebb64dd69d48892857c77ac636d20250902223903.jpeg)

Sebuah gambar digital, yang kita lihat sebagai representasi visual, pada dasarnya adalah matriks besar berisi data numerik atau berupa kumpulan piksel dengan nilai intensitas atau warna tertentu. Data ini, layaknya data numerik lainnya, sering kali memerlukan pemrosesan untuk menghilangkan gangguan atau menonjolkan fitur penting agar dapat dianalisis secara efektif oleh mesin. Di sinilah **metode filtering** dalam pemrosesan citra memegang peranan krusial.

![dos-0e52f13598faf7dabf903ea13068e7b520250902223903.jpeg](https://assets.cdn.dicoding.com/original/academy/dos-0e52f13598faf7dabf903ea13068e7b520250902223903.jpeg)

Anda mungkin telah familier dengan efek filter dalam aplikasi pengolahan gambar. Namun, apakah Anda memahami prinsip matematika yang mendasari operasi tersebut? 

Tahukah Anda?

*   Filter mampu menghaluskan noise dan secara presisi mendeteksi tepi objek melalui mekanisme pemrosesan spasial pada intensitas piksel.
*   Setiap piksel dalam gambar dapat dimodelkan sebagai sampel acak dari suatu distribusi, sementara noise bisa direpresentasikan secara probabilistik sebagai penyimpangan dari distribusi tersebut.
*   Operasi inti seperti konvolusi adalah bentuk rata-rata tertimbang atau pengukuran terhadap perbedaan statistik lokal antar piksel.
*   Filter penghalus bekerja dengan cara mengurangi variansi lokal pada intensitas piksel, menyerupai pendekatan smoothing data dalam analisis statistik.
*   Filter deteksi tepi mengestimasi gradien atau perubahan signifikan dalam distribusi intensitas piksel dengan menggunakan konsep turunan diskret yang merepresentasikan laju perubahan secara statistik.

Tenang, kita akan bedah istilah-istilah di atas dalam materi ini. Bersiaplah untuk melihat gambar bukan sebatas representasi visual, melainkan sebagai kumpulan data yang dapat diolah dan diinterpretasi melalui lensa statistik untuk mengungkap informasi tersembunyi. 

Mari kita mulai eksplorasi.

## Matematika dalam Computer Vision (CV): Apa Itu Computer Vision?

_**Computer vision**_ **(CV)** adalah cabang dari bidang ilmu _artificial intelligence_ (AI) untuk melatih komputer agar mampu **"melihat"** serta **"memahami"** dunia dari gambar dan video sebagaimana manusia melakukannya. 

Tujuan utamanya bukan sekadar menampilkan gambar, melainkan untuk **menginterpretasikan dan mengekstraksi informasi** yang bermakna dari data visual tersebut.

Bayangkan saja kemampuan mata dan otak kita: mengenali wajah orang yang kita kenal, membaca tulisan pada papan reklame, membedakan berbagai jenis hewan, atau bahkan mengukur jarak benda dari kita. Computer vision berupaya mereplikasi kemampuan luar biasa tersebut pada lingkungan mesin.

![dos-7dec42a3a7409692ef1ba7c93a983f1420250902224104.jpeg](https://assets.cdn.dicoding.com/original/academy/dos-7dec42a3a7409692ef1ba7c93a983f1420250902224104.jpeg)

Berbeda dengan data tekstual atau numerik biasa, CV bekerja dengan **piksel**. Setiap gambar atau _frame_ video tersusun dari jutaan piksel, yakni masing-masing piksel memiliki nilai intensitas (untuk gambar hitam-putih) atau nilai warna (untuk gambar berwarna, misalnya _red-green-blue_ atau RGB). 

Dengan menganalisis pola, warna, bentuk, dan gerakan dari kumpulan piksel-piksel ini, komputer dapat "belajar" serta "memahami" hal yang sebenarnya terkandung dalam sebuah visual. Proses ini melibatkan banyak sekali perhitungan statistik untuk mengidentifikasi pola, mengukur kesamaan, dan membuat keputusan.

Implementasi computer vision mungkin terdengar rumit, tetapi faktanya, teknologi ini sudah meresap dalam berbagai aspek kehidupan kita sehari-hari, sering kali tanpa kita sadari.

*   **Pengenalan Wajah (**_**Face Recognition**_**)**  
    Ini adalah salah satu aplikasi CV yang paling populer dan banyak digunakan. Contoh implementasinya sebagai berikut.
    *   **Membuka Kunci** _**Smartphone**_**:** Anda cukup menatap layar ponsel dan dalam sepersekian detik, ponsel Anda terbuka. Sistem CV menganalisis fitur unik wajah Anda untuk memverifikasi identitas.
    *   **Verifikasi Keamanan:** Di bandara, stasiun, dan kantor; sistem CV dapat membandingkan wajah seseorang dengan database untuk memverifikasi identitas mereka.
    *   _**Tagging**_ **Otomatis pada Media Sosial:** Ketika Anda mengunggah foto ke platform media sosial, sistem secara otomatis memberikan tag nama teman-teman Anda dalam foto.
*   **Mobil Otonom (**_**Self-Driving Cars**_**)**  
    Computer vision bertindak sebagai "mata" utama bagi mobil tanpa pengemudi. Contoh implementasinya adalah berikut.
    *   **Deteksi Objek:** Kamera pada mobil terus-menerus memindai lingkungan dan CV bertugas mendeteksi objek-objek penting, seperti mobil lain, pejalan kaki, pesepeda, serta hewan.
    *   **Pengenalan Rambu Lalu Lintas dan Marka Jalan:** CV membaca rambu "STOP", "Belok Kiri", atau mengenali garis putus-putus dan garis solid di jalan.
    *   **Pelacakan Gerakan:** Sistem melacak gerakan objek lain di sekitar mobil untuk memprediksi perilakunya dan mencegah tabrakan.
*   **Filter Media Sosial (**_**Social Media Filters**_**)**  
    Pernah menambahkan telinga anjing, kacamata lucu, atau efek riasan virtual ke wajah Anda di Instagram, Snapchat, atau TikTok? Contoh implementasinya berikut.
    *   CV mendeteksi dan melacak fitur-fitur penting pada wajah Anda secara _real-time_ (misalnya, posisi mata, hidung, mulut, dan garis rahang).
    *   Setelah fitur-fitur ini terdeteksi, filter dapat menempatkan objek virtual dengan presisi tinggi, membuatnya terlihat menyatu dengan wajah Anda.

Pertanyaan menariknya, “Bagaimana computer vision bisa melakukan itu semua?”

## Matematika dalam Computer Vision (CV): Memahami Citra sebagai Data Numerik

Kita telah melihat cara computer vision memungkinkan komputer "melihat" dan "memahami" dunia visual. Namun, bagaimana sebenarnya komputer melakukan itu? 

Kuncinya terletak pada cara komputer memandang sebuah gambar: **bukan sebagai representasi visual semata, melainkan sebagai kumpulan data numerik.** Di sinilah statistika memegang peranan krusial dalam computer vision.

Setiap gambar digital yang kita lihat, dari foto dalam ponsel hingga _frame_ video, pada dasarnya adalah kumpulan angka-angka (piksel). Bayangkan sebuah gambar sebagai lembaran kotak-kotak kecil yang sangat banyak. Setiap kotak kecil ini disebut piksel dan masing-masing piksel menyimpan nilai numerik yang mewakili intensitas cahaya atau warnanya.

*   Pada **citra** _**grayscale**_ **(hitam-putih)**, setiap piksel memiliki satu nilai intensitas, biasanya dari 0 (hitam pekat) hingga 255 (putih terang). 
*   Untuk **citra berwarna**, seperti yang kita lihat sehari-hari, setiap piksel sebenarnya terdiri dari kombinasi tiga saluran warna dasar: **merah (red), hijau (green), dan biru (blue)**, sering disebut **RGB**. Masing-masing saluran ini juga memiliki nilai intensitasnya sendiri (misalnya, dari 0 hingga 255) dan kombinasi ketiganya menghasilkan spektrum warna yang luas.

![dos-771e7d7f8d6b8a68bf3d8f743c64b5fd20250902224330.jpeg](https://assets.cdn.dicoding.com/original/academy/dos-771e7d7f8d6b8a68bf3d8f743c64b5fd20250902224330.jpeg)

Karena sebuah gambar adalah kumpulan angka, semua alat dan konsep statistika yang telah kita pelajari menjadi sangat relevan. Kita dapat menggunakan statistika untuk hal-hal berikut.

*   **Menganalisis distribusi kecerahan atau warna dalam suatu area gambar**. Misalnya, dengan menghitung **rata-rata** nilai piksel di sebuah area, kita bisa mengetahui seberapa terang area tersebut secara keseluruhan. **Median** bisa memberikan gambaran kecerahan yang lebih _robust_ terhadap noise ekstrem, sementara **modus** bisa menunjukkan warna atau intensitas yang paling sering muncul.
*   **Menginterpretasi pola-pola yang tersembunyi dalam data piksel**. Apakah ada area yang sangat bervariasi (memiliki **variansi** tinggi) yang menunjukkan adanya tepi atau tekstur? Atau apakah area tersebut sangat seragam?
*   **Mengekstrak informasi penting dari data citra**. Statistik membantu kita mengidentifikasi fitur-fitur seperti tepi objek, tekstur permukaan, atau bahkan mengenali wajah, dengan mengukur dan memodelkan properti numerik dari kelompok piksel.

Dengan demikian, peran statistika sangat fundamental. Ia menyediakan kerangka kerja untuk **menganalisis, menginterpretasi, dan mengekstrak informasi bermakna** dari kumpulan data piksel, mengubahnya dari sekadar visual menjadi "pemahaman" yang dapat diolah mesin.

## Matematika dalam Computer Vision (CV): Analisis Fitur Citra Berbasis Statistik

Setelah kita memahami bahwa setiap gambar adalah sekumpulan data numerik yang terdiri dari piksel, pertanyaan berikutnya adalah “Bagaimana kita bisa mendapatkan gambaran statistik secara menyeluruh dari miliaran piksel ini?” Lalu, “Bagaimana kita bisa ‘membaca’ data ini untuk memahami karakteristik visual sebuah gambar tanpa harus memeriksa setiap piksel satu per satu?” 

Jawabannya ada pada **histogram citra**.

  

### Histogram Citra: Distribusi Intensitas Piksel

Secara fundamental, histogram citra adalah representasi grafis dari distribusi frekuensi nilai intensitas piksel dalam sebuah gambar. Bayangkan seperti sensus mini yang dilakukan pada setiap piksel: Anda menghitung jumlah piksel yang memiliki nilai intensitas 0 (hitam pekat), yang memiliki nilai 1, yang memiliki nilai 2, dan seterusnya hingga nilai maksimal 255 (putih terang). 

Histogram kemudian memvisualisasikan hasil penghitungan ini dalam bentuk grafik yang menunjukkan hubungan antara nilai intensitas (sumbu horizontal) dan jumlah piksel yang memiliki intensitas tersebut (sumbu vertikal). Dengan demikian, histogram membantu kita memahami sebaran kecerahan dalam sebuah gambar.

![dos-2888d65fff76bcd53d584947c70d815820250902225024.png](https://assets.cdn.dicoding.com/original/academy/dos-2888d65fff76bcd53d584947c70d815820250902225024.png)

![dos-c3f7584a8610abd3d8a02efda18729a620250902225052.png](https://assets.cdn.dicoding.com/original/academy/dos-c3f7584a8610abd3d8a02efda18729a620250902225052.png)

Berikut adalah penjelasan dari grafik histogram di atas.

*   **Sumbu horizontal (X-axis)** mewakili rentang nilai intensitas piksel yang mungkin muncul dalam gambar, biasanya dari 0 hingga 255. Nilai 0 di ujung kiri menunjukkan piksel yang paling gelap, sedangkan nilai 255 di ujung kanan menunjukkan piksel yang paling terang.
*   **Sumbu vertikal (Y-axis)** menunjukkan **frekuensi**, yaitu jumlah piksel yang memiliki nilai intensitas tertentu. Semakin tinggi nilai pada sumbu ini, semakin banyak piksel dalam gambar yang memiliki intensitas tersebut.

Jadi, dengan sekali pandang pada histogram, kita bisa langsung melihat "kecerahan" atau "warna" sebuah gambar terdistribusi secara statistik.

  

### Statistik Spasial: Hubungan Antar Piksel Tetangga

Setelah kita menganalisis distribusi keseluruhan piksel dalam sebuah gambar menggunakan histogram, kini saatnya masuk lebih dalam pada hubungan antar piksel secara lokal. Ingat bahwa gambar bukan hanya sekumpulan piksel terisolasi; setiap piksel memiliki "tetangga" yang memengaruhi dan dipengaruhi olehnya. 

**Statistik spasial** dalam _computer vision_ berfokus pada analisis pola dan hubungan antar piksel yang berdekatan ini. Kemampuan untuk memahami hubungan lokal antar piksel adalah kunci bagi komputer untuk mengenali bentuk, tekstur, dan tepi objek.

![dos-47d39f5686a30e93af1e935b474ca0f920250902225122.png](https://assets.cdn.dicoding.com/original/academy/dos-47d39f5686a30e93af1e935b474ca0f920250902225122.png)

Hubungan antar piksel ini biasanya dianalisis melalui jendela kecil (disebut sebagai filter) yang mengelilingi suatu piksel pusat—misalnya area 3 × 3 atau 5 × 5 piksel. Dalam jendela tersebut, nilai intensitas piksel tetangga digunakan untuk menghitung karakteristik lokal, seperti rata-rata, variansi, atau gradien. Teknik ini memungkinkan kita menangkap pola-pola mikro yang tidak terlihat hanya dari histogram global. Kita bahas lebih lanjut nanti.

Intinya, statistik spasial menjadi landasan bagi banyak operasi dalam pemrosesan citra digital, seperti _filtering_, _edge detection_, dan _feature extraction_. Misalnya, filter deteksi tepi, seperti Sobel atau Laplacian bekerja dengan mengukur perubahan intensitas antar piksel tetangga untuk menemukan batas objek. Dengan kata lain, statistik spasial memungkinkan algoritma mengenali struktur lokal yang penting bagi pemahaman visual oleh komputer.

  

### Konvolusi: Jendela Berjalan yang Membaca Pola Lokal

Inti dari banyak operasi statistik spasial dalam _computer vision_ adalah **konvolusi**. Mungkin terdengar rumit, tapi secara intuitif, konvolusi bisa dibayangkan sebagai sebuah **"jendela"** kecil (disebut _kernel_ atau _filter_) yang bergerak secara sistematis di atas seluruh gambar.

![dos-c56976227881a4caded89cf589ee638820250902225154.png](https://assets.cdn.dicoding.com/original/academy/dos-c56976227881a4caded89cf589ee638820250902225154.png)

Bayangkan seperti ini.

*   Jendela yang dimaksud ini berukuran kecil, biasanya 3 × 3 atau 5 × 5 piksel.
*   Dalam jendela tersebut ada serangkaian angka (bobot) yang telah ditentukan.
*   Jendela ini akan **bergeser** dari satu area gambar ke area lainnya, piksel demi piksel.
*   Untuk setiap posisi jendela, komputer akan **mengalikan** nilai piksel di bawah jendela dengan bobot yang sesuai dalam jendela, lalu **menjumlahkan** semua hasil perkalian tersebut.
*   Hasil penjumlahan ini kemudian menjadi **nilai piksel baru** untuk posisi tengah jendela pada gambar output.

Jadi, konvolusi pada dasarnya adalah **proses "rata-rata tertimbang"** atau **"pengukuran perbedaan statistik lokal" antar piksel tetangga**. Angka-angka dalam jendela (kernel) menentukan jenis "rata-rata" atau "perbedaan" yang sedang dihitung. Dengan mengubah angka-angka dalam kernel, kita bisa menciptakan berbagai efek _filtering_ yang menyoroti karakteristik statistik spasial tertentu dari gambar.

Setelah kita tahu bahwa komputer bisa "membaca" gambar sebagai angka, langkah selanjutnya adalah cara kita bisa memerintahkan komputer untuk melakukan sesuatu yang berguna dengan angka-angka ini, misalnya membuat gambar jadi lebih halus atau menemukan batas-batas objek di dalamnya.

Mari kita lihat dua contoh umum berikut.

*   **Filter Penghalus (**_**Smoothing Filter**_**): Menghilangkan Bintik-Bintik (Noise) dengan Efek Blur**  
    Pernah melihat foto yang tampak memiliki bintik-bintik atau noise kecil yang mengganggu? Filter penghalus adalah salah satu solusi paling umum dalam pengolahan citra untuk mengatasi masalah ini.  
      
    Filter ini bekerja dengan prinsip sederhana: **meratakan nilai-nilai piksel di area lokal** untuk menghasilkan tampilan yang lebih halus.  
      
    Dalam praktiknya, filter penghalus sering disebut juga sebagai blur filter karena memang efek visual yang dihasilkan adalah blur, yaitu gambar tampak lebih lembut dan tidak terlalu tajam. Efek ini terjadi karena **variasi ekstrem antar piksel dikurangi** sehingga transisi antar area menjadi lebih mulus.  
      
    Salah satu bentuk paling sederhana adalah **mean filter** atau **box blur**, yang menggunakan kernel berisi angka-angka seragam (misalnya semua 1). Ketika jendela filter bergerak di atas gambar, ia akan **menghitung rata-rata nilai intensitas dari piksel-piksel dalam jendela tersebut**, lalu **mengganti nilai piksel pusat dengan rata-rata itu**.  
      
    Apa artinya secara statistik? Piksel yang nilainya sangat berbeda dengan tetangganya— misalnya, terlalu terang di tengah piksel gelap—akan ditarik mendekati nilai rata-rata di sekitarnya. Hal ini efektif untuk mengurangi noise atau detail ekstrem yang tidak konsisten, mirip seperti kita menghitung rata-rata nilai ujian untuk memahami performa keseluruhan siswa, bukan hanya satu nilai ekstrem.  
      
    Hasilnya, gambar akan terlihat lebih halus (_smooth_), noise visual berkurang, serta detail halus dan tepi objek menjadi kurang tajam. Contohnya bisa Anda lihat pada perbandingan berikut: gambar asli vs. gambar setelah diterapkan filter penghalus (_blur_). Perhatikan **detail arsitektur menjadi lebih lembut** dan **tekstur kasar menjadi rata**.  
    ![dos-a8ffa3bc02e9f6481461cf13476e3c6220250902225241.png](https://assets.cdn.dicoding.com/original/academy/dos-a8ffa3bc02e9f6481461cf13476e3c6220250902225241.png)  
    
*   **Filter Deteksi Tepi: Menemukan Batasan Objek dengan Mengukur Perubahan Drastis**  
    Sekarang, bayangkan Anda ingin komputer dapat mengenali garis-garis batas antara objek dalam sebuah gambar—misalnya tepi meja, batas bangunan, atau garis pada pakaian. Di sinilah filter deteksi tepi digunakan.  
      
    Berbeda dari filter penghalus yang merata-ratakan nilai, **filter deteksi tepi dirancang untuk mencari perbedaan nilai piksel yang signifikan**. Kernel pada filter ini tidak diisi angka-angka seragam; biasanya berisi kombinasi nilai negatif dan positif untuk menyoroti perubahan intensitas antar piksel.  
      
    Ketika jendela filter ini bergerak di atas gambar, ia **tidak lagi menghitung rata-rata, tetapi mengukur seberapa besar perubahan intensitas piksel** dari satu sisi ke sisi lain. Bayangkan Anda sedang berjalan di permukaan datar, lalu tiba-tiba menghadapi tebing curam—perubahan drastis itulah yang coba dideteksi oleh filter ini.  
      
    Dalam konteks gambar, jika terjadi transisi tajam dari piksel gelap ke terang (atau sebaliknya), filter akan menghasilkan nilai intensitas tinggi, yang mengindikasikan adanya tepi. Area yang warnanya seragam atau perubahan intensitasnya halus tidak akan menghasilkan respons besar dari filter.  
    ![dos-223a29fc1a37267446856978f32287f120250902225344.png](https://assets.cdn.dicoding.com/original/academy/dos-223a29fc1a37267446856978f32287f120250902225344.png)Dengan kata lain, filter deteksi tepi bekerja seperti "sensor perubahan"—ia sangat sensitif terhadap lonjakan atau penurunan tajam dalam distribusi nilai piksel. Ini mirip dengan cara ahli statistik mencari "anomali" atau "peristiwa penting" dalam data numerik.  
      
    Filter deteksi tepi membantu komputer "melihat" struktur dan bentuk dalam gambar, menjadikannya komponen penting dalam tugas-tugas seperti segmentasi objek, pelacakan (_tracking_), dan pengenalan pola (_pattern recognition_).

  

### Ekstraksi Fitur Statistik dari Area Citra

Setelah memahami cara filter sederhana dapat digunakan untuk menghaluskan gambar atau mendeteksi tepi, kini saatnya membahas pendekatan yang lebih canggih dalam “membaca” isi gambar, yaitu **ekstraksi fitur statistik dari area citra**. Pendekatan ini tidak lagi berfokus pada piksel secara individual, tetapi **menganalisis kelompok piksel secara kolektif** untuk mengenali pola yang lebih kompleks, seperti **tekstur**.

**Tekstur** merupakan salah satu ciri visual penting yang dapat membedakan antara permukaan kasar dan halus, membedakan kayu dari kain, atau bahkan membedakan dua objek yang memiliki warna serupa. Untuk mengenali tekstur, sistem _computer vision_ menggunakan berbagai ukuran statistik—tidak hanya berdasarkan nilai rata-rata, tetapi juga memperhitungkan **penyebaran, variasi, dan keteraturan intensitas piksel** dalam suatu area.

#### Mengidentifikasi Tekstur dengan Variansi dan Standar Deviasi Lokal

Salah satu cara paling dasar untuk mengenali tekstur dalam citra adalah dengan **mengukur variansi atau standar deviasi intensitas piksel pada area-area kecil** yang bergerak melintasi gambar. Pendekatan ini mirip dengan operasi konvolusi—tapi alih-alih menghitung rata-rata, kita menilai **seberapa besar sebaran nilai piksel dari nilai tengahnya**, baik melalui variansi maupun akar kuadrat dari variansi tersebut (yaitu standar deviasi).

Bayangkan sebuah jendela kecil (misalnya 3 × 3 piksel) yang berjalan di atas gambar. Pada setiap posisi, kita menghitung seberapa bervariasi nilai-nilai piksel dalam jendela tersebut. 

Daerah dengan variansi atau standar deviasi lokal tinggi menunjukkan adanya perubahan intensitas yang tajam dan tidak merata, yang merupakan ciri khas dari tekstur kasar atau detail tinggi—seperti permukaan batu, anyaman kain, atau helai daun. 

Sebaliknya, area yang halus atau homogen—seperti langit, dinding polos, atau permukaan air—akan memiliki nilai yang rendah karena piksel-pikselnya cenderung seragam.

Untuk mempermudah pemahaman, perhatikan gambar visualisasi dari simulasi berikut.

![dos-2294177013fe09fd03e0ea996f7b9bc920250902225422.png](https://assets.cdn.dicoding.com/original/academy/dos-2294177013fe09fd03e0ea996f7b9bc920250902225422.png)

Visualisasi di atas memperlihatkan dua contoh.

*   **Citra dengan tekstur rendah** (sedikit noise) menghasilkan histogram lokal standar deviasi yang sempit dan berpusat pada nilai rendah (mean std ≈ 0.49).
*   **Citra dengan tekstur tinggi** (_noisy_) memiliki sebaran standar deviasi yang jauh lebih besar (mean std ≈ 29.38), menandakan tingkat keberagaman intensitas yang tinggi.

Dengan pendekatan ini, komputer dapat **membedakan area bertekstur dari area datar** hanya melalui analisis statistik lokal. Ini mirip dengan mengukur "keberagaman" nilai piksel di suatu wilayah—semakin tinggi keberagamannya, semakin bertekstur area tersebut.

Anda bisa mencoba simulasi di atas dengan menjalankan kode Python berikut pada Google Colab.

1.  import numpy as np
2.  import matplotlib.pyplot as plt
3.  from numpy.lib.stride\_tricks import sliding\_window\_view

5.  \# Generate two synthetic images (100x100)
6.  np.random.seed(0)
7.  \# Smooth image now with slight noise (low texture but non-zero std)
8.  img\_smooth \= np.random.normal(loc\=128, scale\=0.5, size\=(100, 100))
9.  \# Noisy image (high texture)
10.  img\_noisy  \= np.random.normal(loc\=128, scale\=30, size\=(100, 100))

12.  \# Compute local standard deviation with a 7x7 window
13.  window\_size \= 7
14.  def local\_std(img):
15.     w \= sliding\_window\_view(img, (window\_size, window\_size))
16.     return w.std(axis\=(2, 3))

18.  std\_smooth \= local\_std(img\_smooth)
19.  std\_noisy  \= local\_std(img\_noisy)

21.  \# Mean std for titles
22.  mean\_s \= std\_smooth.mean()
23.  mean\_n \= std\_noisy.mean()

25.  \# Plotting
26.  fig, axes \= plt.subplots(2, 2, figsize\=(10, 8))

28.  \# Smooth local std map and histogram
29.  axes\[0, 0\].imshow(std\_smooth, cmap\='gray')
30.  axes\[0, 0\].set\_title(f"Low Texture (Slight Noise)\\n(mean std={mean\_s:.2f})")
31.  axes\[0, 0\].axis('off')

33.  axes\[0, 1\].hist(std\_smooth.flatten(), bins\=50, color\='orange')
34.  axes\[0, 1\].set\_title("Histogram of Local Std (Low Texture)")

36.  \# Noisy local std map and histogram
37.  axes\[1, 0\].imshow(std\_noisy, cmap\='gray')
38.  axes\[1, 0\].set\_title(f"High Texture (Noisy)\\n(mean std={mean\_n:.2f})")
39.  axes\[1, 0\].axis('off')

41.  axes\[1, 1\].hist(std\_noisy.flatten(), bins\=50, color\='orange')
42.  axes\[1, 1\].set\_title("Histogram of Local Std (High Texture)")

44.  plt.tight\_layout()
45.  plt.show()

Kode di atas untuk **mengidentifikasi dan membandingkan tekstur dalam citra** menggunakan **standar deviasi lokal** sebagai ukuran statistik penyebaran intensitas piksel di area-area kecil. Mari kita bedah esensi kodenya.

##### Pembuatan Citra Sintetik

1.  img\_smooth \= np.random.normal(loc\=128, scale\=0.5, size\=(100, 100))
2.  img\_noisy  \= np.random.normal(loc\=128, scale\=30, size\=(100, 100))

*   `img_smooth`: citra dengan tekstur rendah (sedikit noise), dibuat dari distribusi normal dengan standar deviasi kecil (`scale=0.5`).
*   `img_noisy`: citra dengan tekstur tinggi (banyak noise), dibuat dari distribusi normal dengan standar deviasi besar (`scale=30`).

Kedua citra berukuran 100 × 100 piksel dan memiliki nilai intensitas pusat di sekitar 128 (abu-abu netral).

**Menghitung Standar Deviasi Lokal**

1.  window\_size \= 7
2.  def local\_std(img):
3.      w \= sliding\_window\_view(img, (window\_size, window\_size))
4.      return w.std(axis\=(2, 3))

Fungsi `local_std()` menggunakan **sliding window 7 × 7** untuk setiap piksel (tanpa _padding_). Untuk setiap jendela, kode menghitung **standar deviasi nilai-nilai piksel** dalam jendela tersebut. Ini memberi kita peta (_map_) yang menunjukkan **seberapa besar variasi lokal** di setiap wilayah gambar.

**Visualisasi Citra dan Histogram**

**![dos-2c45dd6b8d8319f9abc8044b0d03522f20250925151328.png](https://assets.cdn.dicoding.com/original/academy/dos-2c45dd6b8d8319f9abc8044b0d03522f20250925151328.png)**  

1.  axes\[0, 0\].imshow(std\_smooth, cmap\='gray')
2.  axes\[0, 1\].hist(std\_smooth.flatten(), bins\=50, color\='orange')

Gambar kiri menunjukkan **peta standar deviasi lokal** dari citra dengan tekstur rendah. Gambar kanan adalah **histogram distribusi nilai standar deviasi**—sebagian besar nilainya rendah, mencerminkan area yang seragam.

![dos-4ab53cfd18f79570301f3a1061e7a0ca20250925151441.png](https://assets.cdn.dicoding.com/original/academy/dos-4ab53cfd18f79570301f3a1061e7a0ca20250925151441.png)

1.  axes\[1, 0\].imshow(std\_noisy, cmap\='gray')
2.  axes\[1, 1\].hist(std\_noisy.flatten(), bins\=50, color\='orange')

Gambar kiri menampilkan peta standar deviasi lokal untuk citra dengan tekstur tinggi. Histogramnya tersebar lebih luas dan memiliki nilai rata-rata lebih tinggi, menandakan **keragaman piksel yang besar**, tipikal dari tekstur kasar atau noisy.

#### Mendeskripsikan Bentuk Distribusi Intensitas dengan Skewness dan Kurtosis

Untuk mendeskripsikan tekstur secara lebih rinci, kita bisa melangkah lebih jauh dari sekadar variansi dan mulai memperhatikan **bentuk distribusi intensitas piksel** dalam suatu area menggunakan ukuran statistik, seperti _skewness_ dan _kurtosis_.

**Skewness (Kemiringan Distribusi)**

Skewness mengukur seberapa asimetris distribusi nilai intensitas piksel di suatu area. Nilai skewness memberikan informasi tentang kecenderungan distribusi: apakah sebagian besar piksel berada dalam nilai rendah, tinggi, atau seimbang.

*   _**Positive skew**_ **(skew > 0)**: Distribusi condong ke kiri. Artinya, sebagian besar piksel bernilai gelap dengan sedikit piksel terang. Histogram berpuncak di sisi kiri dan memiliki ekor panjang ke kanan.
*   _**Negative skew**_ **(skew < 0)**: Distribusi condong ke kanan. Sebagian besar piksel bernilai terang dengan sedikit piksel gelap. Histogram berpuncak di kanan dan memiliki ekor ke kiri.

![dos-f21007d95b56d4020dd95c56876d6cbe20250902230036.png](https://assets.cdn.dicoding.com/original/academy/dos-f21007d95b56d4020dd95c56876d6cbe20250902230036.png)

Metode ini membantu komputer membedakan tekstur atau pola pencahayaan berdasarkan persebaran intensitas terang-gelap di permukaan suatu objek.

Untuk melakukan simulasi di atas, Anda dapat menjalankan kode berikut pada Google Colab.

1.  import numpy as np
2.  import matplotlib.pyplot as plt
3.  from scipy.stats import skew

5.  \# Generate two synthetic images (100x100)
6.  \# Positive skew: many dark pixels, few bright
7.  np.random.seed(0)
8.  image\_pos \= (np.random.beta(a\=2, b\=10, size\=(100, 100)) \* 255).astype(int)

10.  \# Negative skew: many bright pixels, few dark
11.  image\_neg \= (np.random.beta(a\=10, b\=2, size\=(100, 100)) \* 255).astype(int)

13.  \# Compute skewness
14.  skew\_pos \= skew(image\_pos.flatten())
15.  skew\_neg \= skew(image\_neg.flatten())

17.  \# Plotting
18.  fig, axes \= plt.subplots(2, 2, figsize\=(10, 8))

20.  \# Positive skew image and histogram
21.  axes\[0, 0\].imshow(image\_pos, cmap\='gray')
22.  axes\[0, 0\].set\_title(f"Positive Skew Image\\n(skew={skew\_pos:.2f})")
23.  axes\[0, 0\].axis('off')

25.  axes\[0, 1\].hist(image\_pos.flatten(), bins\=50)
26.  axes\[0, 1\].set\_title("Histogram (Positive Skew)")

28.  \# Negative skew image and histogram
29.  axes\[1, 0\].imshow(image\_neg, cmap\='gray')
30.  axes\[1, 0\].set\_title(f"Negative Skew Image\\n(skew={skew\_neg:.2f})")
31.  axes\[1, 0\].axis('off')

33.  axes\[1, 1\].hist(image\_neg.flatten(), bins\=50)
34.  axes\[1, 1\].set\_title("Histogram (Negative Skew)")

36.  plt.tight\_layout()
37.  plt.show()

Kode di atas membuat dua citra sintetis (100 × 100).

*   **Positive skew image** dihasilkan dari distribusi beta(2, 10), menunjukkan ekor panjang ke kanan.
*   **Negative skew image** menggunakan beta(10, 2) sehingga distribusi terbalik.

Setiap baris subplot menampilkan hal berikut.

1.  Citra dalam skala abu-abu.
2.  Histogram frekuensi piksel (memperlihatkan kemiringan distribusi).
3.  Nilai skewness tercetak di judul citra, memudahkan interpretasi.

**Kurtosis (Keruncingan Distribusi)**

Kurtosis mengukur “puncak” atau “kerataan” distribusi nilai piksel di area tertentu. Area dengan **kurtosis tinggi** (“_leptokurtic_”) menampilkan distribusi yang sangat runcing—banyak piksel terpusat pada nilai intensitas tertentu dengan beberapa outlier (misal area seragam dengan beberapa bintik kontras tinggi). 

Sebaliknya, area dengan **kurtosis rendah** (“_platykurtic_”) memiliki distribusi yang lebih datar, yakni saat nilai piksel tersebar merata dalam rentang nilai.

![dos-0d648313e98624bbebf9c170bbff504d20250902230406.png](https://assets.cdn.dicoding.com/original/academy/dos-0d648313e98624bbebf9c170bbff504d20250902230406.png)

Untuk melakukan simulasi di atas, Anda dapat menjalankan kode berikut pada Google Colab.

1.  import numpy as np
2.  import matplotlib.pyplot as plt
3.  from scipy.stats import kurtosis

5.  \# Generate two synthetic images (100x100)
6.  np.random.seed(1)

8.  \# High kurtosis: mostly constant with few extreme spots
9.  base \= np.full((100, 100), 128, dtype\=int)
10.  num\_spots \= 200
11.  coords \= (np.random.randint(0, 100, num\_spots), np.random.randint(0, 100, num\_spots))
12.  base\[coords\] \= np.random.choice(\[0, 255\], size\=num\_spots)

14.  \# Low kurtosis: uniform distribution across range
15.  uniform\_img \= np.random.randint(0, 256, (100, 100))

17.  \# Compute kurtosis
18.  kurt\_high \= kurtosis(base.flatten(), fisher\=True)
19.  kurt\_low \= kurtosis(uniform\_img.flatten(), fisher\=True)

21.  \# Plotting
22.  fig, axes \= plt.subplots(2, 2, figsize\=(10, 8))

24.  \# High kurtosis image and histogram
25.  axes\[0, 0\].imshow(base, cmap\='gray')
26.  axes\[0, 0\].set\_title(f"High Kurtosis Image\\n(kurtosis={kurt\_high:.2f})")
27.  axes\[0, 0\].axis('off')

29.  axes\[0, 1\].hist(base.flatten(), bins\=50)
30.  axes\[0, 1\].set\_title("Histogram (High Kurtosis)")

32.  \# Low kurtosis image and histogram
33.  axes\[1, 0\].imshow(uniform\_img, cmap\='gray')
34.  axes\[1, 0\].set\_title(f"Low Kurtosis Image\\n(kurtosis={kurt\_low:.2f})")
35.  axes\[1, 0\].axis('off')

37.  axes\[1, 1\].hist(uniform\_img.flatten(), bins\=50)
38.  axes\[1, 1\].set\_title("Histogram (Low Kurtosis)")

40.  plt.tight\_layout()
41.  plt.show()

Kode di atas menghasilkan dua citra sintetis (100 × 100).

*   **High Kurtosis Image**: mayoritas piksel bernilai 128 dengan sedikit titik ekstrem pada nilai 0 atau 255 sehingga histogramnya sangat terpusat dan “runcing” (kurtosis ≈ 47.45).
*   **Low Kurtosis Image**: piksel mengikuti distribusi uniform 0–255, menghasilkan histogram datar (kurtosis ≈ –1.20).

Dengan _skewness_ dan _kurtosis_, kita mendapatkan gambaran statistik yang lebih kaya tentang cara intensitas piksel terdistribusi dan memberikan petunjuk lebih lanjut tentang karakteristik tekstur.

#### Mengekstraksi Pola Tekstur Kompleks dengan Fitur Haralick dan Matriks GLCM

Untuk mengenali tekstur kompleks dan memiliki pola berulang, metode yang lebih canggih sering digunakan, salah satunya adalah Fitur Haralick. Konsep dasar dari pendekatan ini adalah penggunaan Gray-Level Co-occurrence Matrix (GLCM).

GLCM adalah sebuah tabel statistik yang mencatat **seberapa sering pasangan nilai intensitas piksel tertentu muncul bersama** dalam citra dengan mempertimbangkan jarak dan arah tertentu.

Misalnya, secara konseptual, kita bisa membayangkan pertanyaan seperti berikut.

*   “Berapa kali piksel dengan nilai 100 muncul tepat di sebelah kanan piksel bernilai 150?”
*   “Berapa kali piksel bernilai 50 muncul dua langkah di bawah piksel bernilai 200?”

Pertanyaan-pertanyaan ini membantu kita memahami bahwa GLCM menangkap pola kemunculan lokal antar piksel yang membentuk tekstur.

Dari GLCM ini, berbagai fitur statistik yang dikenal sebagai Fitur Haralick dapat dihitung. Ada sekitar 14 fitur standar, masing-masing merepresentasikan karakteristik berbeda dari pola tekstur. Beberapa fitur utamanya berikut.

*   **Contrast:** Mengukur perbedaan intensitas antara piksel-piksel yang berdekatan. Contoh: Tekstur kasar menghasilkan nilai kontras tinggi.
*   **Homogeneity:** Mengukur seberapa dekat nilai-nilai GLCM terhadap diagonal utama. Contoh: Tekstur halus menghasilkan nilai homogenitas tinggi.
*   **Energy/Angular Second Moment:** Mengukur tingkat keseragaman atau keteraturan dalam tekstur. Contoh: Area yang seragam memiliki energi tinggi.

Selain itu masih banyak lagi, yakni Correlation, Entropy, Dissimilarity, dll. Masing-masing menangkap aspek statistik yang berbeda dari pola distribusi piksel dalam ruang.

Fitur Haralick ini banyak digunakan dalam aplikasi nyata, seperti klasifikasi tekstur (misalnya, jenis kain, permukaan logam), deteksi anomali (pada citra medis atau satelit), dan segmentasi citra berbasis pola.

Berikut adalah contoh kode untuk melakukan simulasi dari penggunaan Fitur Haralick. Anda bisa menjalankan kode ini pada Google Colab (Anda perlu menginstal library scikit-image terlebih dahulu).

1.  import numpy as np
2.  import pandas as pd
3.  import matplotlib.pyplot as plt
4.  from skimage.feature import graycomatrix, graycoprops

6.  \# Define larger (8x8) example matrices
7.  matrix1 \= np.array(\[
8.      \[0, 0, 0, 0, 10, 10, 10, 10\],
9.      \[0, 0, 0, 0, 10, 10, 10, 10\],
10.      \[0, 0, 0, 0, 10, 10, 10, 10\],
11.      \[0, 0, 0, 0, 10, 10, 10, 10\],
12.      \[100, 100, 100, 100, 200, 200, 200, 200\],
13.      \[100, 100, 100, 100, 200, 200, 200, 200\],
14.      \[100, 100, 100, 100, 200, 200, 200, 200\],
15.      \[100, 100, 100, 100, 200, 200, 200, 200\]
16.  \], dtype\=np.uint8)

18.  matrix2 \= np.linspace(0, 255, 64, dtype\=np.uint8).reshape((8, 8))

20.  \# Compute Haralick features
21.  def compute\_haralick(mat, distances\=\[1\], angles\=\[0\]):
22.      glcm \= graycomatrix(mat, distances\=distances, angles\=angles, levels\=256, symmetric\=True, normed\=True)
23.      return {
24.          'contrast': graycoprops(glcm, 'contrast')\[0,0\],
25.          'homogeneity': graycoprops(glcm, 'homogeneity')\[0,0\],
26.          'energy': graycoprops(glcm, 'energy')\[0,0\],
27.          'correlation': graycoprops(glcm, 'correlation')\[0,0\],
28.          'dissimilarity': graycoprops(glcm, 'dissimilarity')\[0,0\]
29.      }

31.  feat1 \= compute\_haralick(matrix1)
32.  feat2 \= compute\_haralick(matrix2)

34.  \# Plot heatmaps
35.  fig, axs \= plt.subplots(1, 2, figsize\=(10, 5))
36.  axs\[0\].imshow(matrix1, interpolation\='nearest')
37.  axs\[0\].set\_title("Matrix 1 (8×8)")
38.  axs\[0\].axis('off')

40.  axs\[1\].imshow(matrix2, interpolation\='nearest')
41.  axs\[1\].set\_title("Matrix 2 (8×8)")
42.  axs\[1\].axis('off')

44.  plt.tight\_layout()
45.  plt.show()

47.  \# Compute features
48.  feat1 \= compute\_haralick(matrix1)
49.  feat2 \= compute\_haralick(matrix2)

51.  \# Print results in a table format
52.  print(f"{'Feature':<15} {'Matrix1':>10} {'Matrix2':>10}")
53.  print('-' \* 37)
54.  for key in feat1:
55.      print(f"{key:<15} {feat1\[key\]:>10.4f} {feat2\[key\]:>10.4f}")

Kode di atas digunakan untuk membandingkan dua pola tekstur dengan menggunakan **fitur Haralick**, yang dihitung dari **Gray-Level Co-occurrence Matrix (GLCM)**.

Pertama, dua buah matriks 8 × 8 dibuat seperti berikut.

*   `matrix1` terdiri dari dua blok homogen yang berbeda nilai. Empat baris pertama berisi nilai rendah (0–10) dan empat baris berikutnya berisi nilai tinggi (100–200). Matriks ini menampilkan perubahan tajam antar blok yang menyerupai tekstur kasar.
*   `matrix2` adalah gradien linier dari 0 hingga 255 yang tersebar merata. Ini merepresentasikan tekstur halus tanpa batas blok yang jelas.

Fungsi `compute_haralick()` kemudian melakukan hal berikut.

1.  Menghitung GLCM pada jarak 1 piksel dalam arah horizontal (`angle = 0`) dengan nilai piksel berada dalam rentang 0–255 (`levels=256`).
2.  Menormalisasi GLCM.
3.  Mengekstraksi **lima metrik Haralick utama**.
    *   `contrast` – perbedaan intensitas antar piksel bertetangga.
    *   `homogeneity` – seberapa dekat nilai GLCM terhadap diagonal utama.
    *   `energy` – tingkat keseragaman tekstur.
    *   `correlation` – hubungan linier antar nilai piksel.
    *   `dissimilarity` – perbedaan absolut antar nilai piksel tetangga.

Visualisasi dua matriks ditampilkan sebagai _heatmap_, sedangkan hasil perhitungan fitur ditampilkan dalam bentuk tabel agar mudah dibandingkan. Hasilnya menunjukkan hal berikut.

*   **Struktur blok yang tajam pada** **matrix1** menghasilkan kontras dan dissimilarity yang tinggi, serta homogeneity yang rendah.
*   **Gradien halus pada** **matrix2** cenderung menghasilkan nilai fitur lebih seimbang dan menggambarkan perubahan tekstur yang lebih lembut.

![dos-77c34cf04585cbe9f3add0e341a79a8e20250902230922.png](https://assets.cdn.dicoding.com/original/academy/dos-77c34cf04585cbe9f3add0e341a79a8e20250902230922.png)

Akhirnya, nilai-nilai Fitur Haralick dari masing-masing matriks dicetak dalam bentuk tabel ringkas. Dengan membandingkan hasilnya, kita dapat langsung melihat bahwa struktur blok tajam pada **matrix1** menghasilkan nilai kontras serta dissimilarity yang tinggi, sementara gradasi lembut pada **matrix2** menghasilkan fitur lebih rendah dan homogen. Hal ini menunjukkan bahwa perbedaan visual dalam pola tekstur dapat diukur secara objektif menggunakan statistik dari GLCM.

![dos-ca10605f51a0cf0cd64c2a9569f3e3c320250902230937.png](https://assets.cdn.dicoding.com/original/academy/dos-ca10605f51a0cf0cd64c2a9569f3e3c320250902230937.png)

Fitur Haralick memungkinkan _computer vision_ untuk mengenali pola tekstur yang kompleks, berulang, dan tidak teratur—sesuatu yang tidak dapat ditangkap hanya dengan statistik dasar, seperti variansi, skewness, atau histogram piksel.

Dengan pendekatan ini, komputer dapat membedakan berbagai jenis permukaan yang tampak serupa secara warna, tapi berbeda secara pola, seperti berikut.

*   **Serat kayu** yang memiliki arah dan ketebalan tertentu.
*   **Pola batu bata** yang bersifat periodik dan kontras.
*   **Tekstur kulit** yang acak, tapi khas.

Melalui pendekatan ini, kita dapat menghitung pola tekstur secara statistik dan memahami perbedaan visual dalam citra tecermin pada nilai-nilai Fitur Haralick.

Anda bisa akses kode lengkap yang dijelaskan dalam topik computer vision pada link berikut: [Modul 6 - Computer Vision.ipynb](https://colab.research.google.com/drive/1vfABQ_a0VTe2kPN_gMuNJGvPWdNtFSXl?usp=sharing)

## Matematika dalam Natural Language Processing (NLP)

Setelah memahami bahwa data numerik dianalisis secara statistik, kini saatnya kita berhadapan dengan jenis data yang lain, yang lebih kompleks, tapi juga sangat menarik, yaitu teks. 

Anda mungkin bertanya-tanya, bagaimana mungkin mesin bisa memahami bahasa manusia yang penuh makna, nuansa, dan konteks? Kita saja sebagai manusia kadang harus bertanya dua kali untuk memahami maksud seseorang. Jadi, bagaimana caranya mesin bisa "mengerti"?

![dos-eabc3987d6ddcdfd90f6314fd765fc1920250902231032.jpeg](https://assets.cdn.dicoding.com/original/academy/dos-eabc3987d6ddcdfd90f6314fd765fc1920250902231032.jpeg)

Ternyata, mesin tidak benar-benar “mengerti” seperti manusia. Namun, mereka bisa belajar dari pola dan caranya belajar adalah dengan menghitung. Statistik menjadi alat utama bagi sistem untuk memahami teks, bukan lewat arti kata dalam kamus, melainkan dari seberapa sering kata itu muncul, pada konteks apa, dan hubungannya dengan kata-kata lain.

Dalam materi ini, kita akan menyelami cara-cara sistem komputer mengekstrak fitur dari teks,  mulai yang paling sederhana, seperti menghitung kemunculan kata, hingga metode yang bisa menangkap hubungan semantik yang lebih halus, seperti TF-IDF dan Word Embeddings. 

Hal yang menarik, semua itu berpijak pada prinsip-prinsip statistik yang sudah Anda kenal.

## Matematika dalam Natural Language Processing (NLP): Pengenalan

_Natural language processing_ (NLP) adalah sebuah cabang menarik dari _artificial intelligence_ (AI) yang berfokus pada cara komputer dapat **memahami, menginterpretasi, dan bahkan menghasilkan bahasa manusia**, baik itu teks tertulis maupun ucapan. 

Tujuan utama NLP adalah menjembatani kesenjangan antara cara manusia berkomunikasi secara alami dengan cara komputer memproses informasi. Bayangkan komputer Anda bisa membaca sebuah berita, meringkasnya, lalu menjawab pertanyaan tentang isinya. Atau bisa mendengarkan perintah suara Anda, memahami maksudnya, dan melaksanakannya. 

Inilah inti dari NLP: memberikan kemampuan "berbahasa" kepada mesin. Bahasa manusia itu rumit—penuh dengan ambiguitas, metafora, dan aturan yang fleksibel. NLP berupaya mengajarkan komputer untuk mengatasi kerumitan ini sehingga memungkinkannya berinteraksi dengan manusia secara lebih intuitif dan cerdas.

![dos-a41a2080f576d8d3fbf2eb7b6a72095b20250902231145.jpeg](https://assets.cdn.dicoding.com/original/academy/dos-a41a2080f576d8d3fbf2eb7b6a72095b20250902231145.jpeg)

Sebelum sebuah sistem bisa memahami bahasa manusia, ia perlu tahu dulu cara “membaca” teks sebagai data. Namun, tidak seperti manusia yang bisa memahami makna kata berdasarkan pengalaman dan intuisi, komputer tidak punya kemampuan itu. Ia buta terhadap makna, nada, emosi, atau konteks. Hal yang bisa komputer lakukan hanyalah menghitung dan justru dari proses menghitung inilah muncul kemampuan dasar untuk “memahami” bahasa secara statistik.

Langkah awal ini penting karena dari sinilah semua proses lanjutan dibangun. Sebelum bisa berbicara tentang semantik, konteks, atau prediksi, kita harus mulai dari mengenali bahwa kata adalah bagian terkecil dari informasi dalam teks dan kemunculan kata bisa dihitung serta dijadikan dasar untuk mengenali pola.

## Matematika dalam Natural Language Processing (NLP): Visualisasi Distribusi Kata dengan WordCloud

Distribusi kata mengacu pada cara kata-kata tertentu muncul dalam teks atau koleksi dokumen. Dengan memvisualisasikan distribusi kata, kita dapat dengan mudah memahami kata-kata yang sering muncul dalam teks dan yang jarang digunakan. Salah satu cara yang paling efektif untuk menggambarkan distribusi kata dalam teks adalah melalui WordCloud.

  

### Apa itu WordCloud?

**WordCloud** adalah representasi grafis dari teks bahwa kata-kata yang lebih sering muncul dalam teks akan ditampilkan dengan ukuran lebih besar. Sebaliknya, kata-kata yang jarang muncul akan memiliki ukuran font lebih kecil. 

![dos-47b730d4bef2f405bb5142cad1c9e17b20250902231354.jpeg](https://assets.cdn.dicoding.com/original/academy/dos-47b730d4bef2f405bb5142cad1c9e17b20250902231354.jpeg)

WordCloud adalah salah satu teknik visualisasi yang sangat berguna dalam natural language processing (NLP) untuk memberikan gambaran umum tentang frekuensi kata pada sebuah teks atau kumpulan dokumen.

WordCloud sering digunakan dalam berbagai aplikasi seperti berikut.

*   **Analisis Sentimen**: Untuk memvisualisasikan kata-kata yang sering digunakan dalam teks dengan sentimen positif atau negatif.
*   **Pencarian Informasi**: Untuk mengetahui kata-kata kunci atau topik dominan dalam koleksi dokumen.
*   **Penyajian Data Teks**: Membantu dalam presentasi data teks dengan cara yang mudah dipahami dan menarik.

  

### Bagaimana WordCloud Mencerminkan Distribusi Kata?

Distribusi kata merujuk pada frekuensi kemunculan kata dalam sebuah teks atau koleksi teks. Dalam visualisasi WordCloud, distribusi kata ini digambarkan melalui ukuran font kata. Kata-kata yang sering muncul dalam teks akan memiliki font lebih besar, sementara kata-kata yang muncul lebih jarang akan tampil dengan ukuran font lebih kecil.

Berikut adalah proses visualisasi distribusi kata menggunakan WordCloud.

1.  **Menghitung frekuensi kata**: Setiap kata dalam dokumen dihitung dengan seberapa sering muncul.
2.  **Memetakan frekuensi ke ukuran font**: Kata dengan frekuensi tinggi akan diberi ukuran font besar, sedangkan kata dengan frekuensi rendah akan diberi ukuran kecil.
3.  **Menampilkan visualisasi**: WordCloud akan menampilkan kata-kata dalam bentuk gambar, yaitu kata-kata yang lebih sering muncul akan lebih dominan secara visual.

  

### Manfaat Penggunaan WordCloud

Visualisasi distribusi kata dengan **WordCloud** memiliki banyak manfaat dalam konteks NLP, terutama untuk eksplorasi data dan analisis teks, antara lain berikut.

1.  **Pengenalan Kata Kunci**  
    **WordCloud** dapat memberikan gambaran sekilas tentang kata-kata kunci dalam teks atau dokumen. Kata-kata yang muncul dengan ukuran besar adalah kata-kata yang dianggap paling penting atau dominan dalam teks tersebut.
2.  **Identifikasi Tema atau Topik Dominan**  
    Dengan menganalisis WordCloud, kita dapat dengan mudah mengidentifikasi tema atau topik yang sering dibicarakan dalam kumpulan dokumen. Misalnya, dalam koleksi artikel berita, kata-kata yang sering muncul akan memberikan petunjuk tentang topik utama dalam berita tersebut.
3.  **Memudahkan Eksplorasi Teks**  
    WordCloud adalah alat yang sangat berguna untuk eksplorasi data teks secara cepat dan mudah dipahami. Ini memungkinkan kita untuk melihat pola atau hubungan antara kata-kata yang mungkin sulit dilihat hanya dengan analisis numerik atau statistik.
4.  **Memvisualisasikan Perubahan dalam Data Teks**  
    WordCloud juga dapat digunakan untuk memvisualisasikan perubahan dalam distribusi kata dari waktu ke waktu. Misalnya, jika memiliki koleksi dokumen atau _tweet_ yang terorganisasi berdasarkan waktu, kita dapat membuat **WordCloud** untuk setiap periode waktu dan melihat kata-kata yang lebih menonjol.

  

### Membuat WordCloud dengan Python

Setelah memahami pengertian, cara kerja, dan manfaat dari penggunaan WordCloud, sekarang kita coba praktikkan pembuatan WordCloud dalam bahasa pemrograman Python. Anda bisa ikuti dengan menjalankan kode pada Google Colab.

Berikut adalah langkah-langkah untuk membuat **WordCloud** dengan Python. 

#### Langkah 1: Impor Library

Hal pertama yang perlu dilakukan adalah mengimpor library yang dibutuhkan. Untuk membuat WordCloud dengan Python, kita dapat menggunakan library **Matplotlib** dan **WordCloud**. 

1.  import matplotlib.pyplot as plt
2.  from wordcloud import WordCloud

#### Langkah 2: Menyiapkan Data Teks

Anda bisa menggunakan teks dari dokumen atau kumpulan ulasan, berita, atau sentimen. Kita akan gunakan contoh data statis berupa daftar ulasan pembelajaran matematika dari siswa seperti berikut.

1.  \# Daftar ulasan
2.  reviews \= \[
3.     "Saya suka belajar Matematika.",
4.     "Belajar matematika menyenangkan.",
5.     "Matematika itu menyenangkan.",
6.     "Belajar matematika sangat penting.",
7.     "Matematika adalah ilmu dasar."
8.  \]

Kumpulan kalimat disimpan dalam list `reviews`. Kalimat-kalimat ini merepresentasikan opini atau pernyataan pengguna tentang topik tertentu (dalam hal ini: matematika).

#### Langkah 3: Menggabungkan Kalimat Menjadi Satu String

Untuk membuat WordCloud, semua teks digabungkan menjadi satu string panjang.

1.  \# Gabungkan semua ulasan menjadi satu string untuk WordCloud
2.  combined\_reviews \= " ".join(reviews)

Metode `.join()` menggabungkan elemen list reviews dengan spasi sebagai pemisah antar kalimat.

#### Langkah 4: Membuat WordCloud 

Setelah datanya siap, sekarang saatnya kita generate WordCloud dengan kode berikut.

1.  \# Membuat WordCloud
2.  wordcloud \= WordCloud(width\=800, height\=400, background\_color\='white').generate(combined\_reviews)

Berikut detail penjelasannya.

*   `WordCloud()` adalah objek dari library `wordcloud` yang digunakan untuk menghasilkan visualisasi.
*   Parameter `width` dan `height` menentukan ukuran kanvas gambar.
*   `background_color='white'` menjadikan latar belakang Word Cloud berwarna putih.
*   `.generate(combined_reviews)` membuat visualisasi berdasarkan string gabungan yang sudah disiapkan.

#### Langkah 5: Menampilkan WordCloud

Langkah terakhir adalah bagian yang menyenangkan, yakni saatnya kita menampilkan visualisasi dari WordCloud yang tadi dibuat.

1.  \# Menampilkan WordCloud
2.  plt.figure(figsize\=(10, 5))
3.  plt.imshow(wordcloud, interpolation\='bilinear')
4.  plt.axis('off')  \# Menyembunyikan axis
5.  plt.show()

Berikut detail penjelasannya.

*   `plt.figure()` menentukan ukuran figur tampilan.
*   `plt.imshow()` menampilkan gambar WordCloud.
*   `interpolation='bilinear'` memperhalus tampilan visual.
*   `plt.axis('off')` menyembunyikan sumbu agar hasil tampak lebih bersih.
*   `plt.show()` menampilkan hasil akhir pada layar.

Inilah tampilan visualisasi dari WordCloud yang kita buat.

![dos-b0625e085b67e9884fe0e3b47c9b38bb20250902231353.jpeg](https://assets.cdn.dicoding.com/original/academy/dos-b0625e085b67e9884fe0e3b47c9b38bb20250902231353.jpeg)

Kita bisa lihat bahwa dalam contoh ini, kata **“Matematika”**, **“Belajar”**, dan **“menyenangkan”** tampil lebih besar karena sering muncul pada ulasan.

Dengan memperhitungkan frekuensi kata dalam sebuah dokumen atau kumpulan dokumen, **WordCloud** memungkinkan kita untuk melihat secara langsung kata-kata yang lebih dominan atau penting. Ini sangat berguna dalam eksplorasi data, analisis sentimen, pencarian informasi, dan aplikasi NLP lainnya.

Dengan menggunakan **WordCloud**, kita dapat mengidentifikasi kata kunci atau topik utama dalam kumpulan teks, serta melihat perubahan distribusi kata dari waktu ke waktu atau antar dokumen. Teknik visualisasi ini memungkinkan kita untuk lebih mudah memahami pola-pola dalam data teks besar.

## Matematika dalam Natural Language Processing (NLP): Representasi Frekuensi Kata (Bag-of-Words)

Dalam dunia natural language processing (NLP), salah satu tantangan terbesar adalah membuat komputer memahami bahasa manusia. 

Komputer, pada dasarnya, hanya mengerti angka. Jadi, jika ingin komputer menganalisis ulasan pelanggan, email, atau berita, kita harus mengubah teks-teks itu menjadi format numerik yang bisa diolah. 

Di sinilah teknik **Bag of Words (BoW)** berperan sebagai salah satu metode paling sederhana dan paling awal untuk melakukan "terjemahan" ini.

  

### Apa itu Bag of Words?

Bayangkan Bag of Words ini seperti sebuah tas belanja. Ketika Anda memasukkan bahan-bahan makanan ke tas, Anda mungkin tidak peduli dengan urutan memasukkannya. Hal yang penting adalah bahan dalam tas itu dan jumlah dari setiap jenis bahan yang Anda punya. 

![dos-e5fbb1f12f9465e6fcae58d2f6c4a96520250902232111.jpeg](https://assets.cdn.dicoding.com/original/academy/dos-e5fbb1f12f9465e6fcae58d2f6c4a96520250902232111.jpeg)

Konsep yang sama berlaku untuk BoW: kita hanya peduli **kata dalam sebuah dokumen teks** dan **jumlah kemunculan masing-masing kata itu**, tanpa memedulikan urutan atau konteks gramatikalnya. 

Oleh karena itu, metode ini disebut "Bag of Words" karena mirip seperti kita melihat isi tas—kita hanya peduli tentang "apa" yang ada di dalamnya, bukan "bagaimana" urutannya.

![dos-e7edec4204882b8e1de3621c3cdac3fe20250902232110.jpeg](https://assets.cdn.dicoding.com/original/academy/dos-e7edec4204882b8e1de3621c3cdac3fe20250902232110.jpeg)

Bag of Words adalah pendekatan sederhana untuk **merepresentasikan teks sebagai vektor angka** berdasarkan **frekuensi kemunculan kata** dalam kumpulan dokumen.

Dalam konteks ini, setiap **kata unik** yang muncul pada semua ulasan disebut sebagai **fitur (**_**feature**_**)**. Setiap teks diubah menjadi sebuah **vektor** yang menyatakan **jumlah setiap kata muncul** dalam ulasan tersebut.

Metode ini disebut _“bag”_ karena struktur kalimat diabaikan dan urutan kata tidak diperhatikan; yang penting hanyalah keberadaan serta frekuensi kata.

Tujuan utama BoW adalah menghasilkan fitur numerik dari dokumen teks sehingga teks tersebut dapat diaplikasikan pada berbagai algoritma _machine learning_. Algoritma seperti Naïve Bayes, _support vector machine_ (SVM), atau _decision tree_ tidak bisa langsung bekerja dengan teks mentah. Mereka butuh angka.

Dalam metode BoW, setiap kata unik yang ditemukan dalam seluruh koleksi dokumen (sering disebut _korpus_) akan dianggap sebagai **fitur** tersendiri. Kemudian, untuk setiap dokumen, kita akan menghitung frekuensi (berapa kali) setiap kata unik tersebut muncul. 

Hasilnya, setiap dokumen akan direpresentasikan sebagai sebuah **vektor** (barisan angka), yakni setiap elemen pada vektor tersebut menunjukkan frekuensi kemunculan kata tertentu dalam dokumen itu. 

![dos-31f5587a9432d79d0e242272f9ae4b1f20250902232109.jpeg](https://assets.cdn.dicoding.com/original/academy/dos-31f5587a9432d79d0e242272f9ae4b1f20250902232109.jpeg)

Metode ini sangat sering digunakan dalam tugas-tugas seperti klasifikasi teks, contohnya untuk menentukan jika sebuah ulasan bersifat positif atau negatif, atau lainnya untuk mengetahui bahwa sebuah email adalah spam atau bukan.

  

### Cara Kerja Bag of Word

Untuk memahami cara kerja metode _bag of words_, mari kita mulai dengan sebuah contoh sederhana dan relevan. Bayangkan Anda adalah seorang analis yang ingin **mengklasifikasikan ulasan siswa tentang pembelajaran matematika** dalam dua kategori: “positif” atau “negatif” berdasarkan isi teks ulasannya.

Berikut adalah beberapa contoh ulasan yang berhasil dikumpulkan.

*   **Ulasan 1:** "Saya suka belajar Matematika."
*   **Ulasan 2:** "Belajar matematika menyenangkan."
*   **Ulasan 3:** "Matematika itu menyenangkan."
*   **Ulasan 4:** "Belajar matematika sangat penting."
*   **Ulasan 5:** "Matematika adalah ilmu dasar."

Namun, agar dapat dianalisis oleh komputer—terutama oleh algoritma machine learning—**teks harus diubah dalam format numerik** karena komputer tidak memahami makna bahasa seperti manusia. Di sinilah metode **Bag of Words (BoW)** berperan.

Untuk menerapkan metode Bag of Words, kita perlu melalui beberapa tahapan penting. Mari kita uraikan langkah-langkahnya berdasarkan contoh ulasan yang sudah dikumpulkan sebelumnya.

#### Menyusun Kosakata Unik (_Vocabulary_)

Langkah pertama adalah menyusun **daftar kosakata unik** yang muncul dalam seluruh kumpulan ulasan. Disebut “unik” karena setiap kata hanya dicatat satu kali meskipun muncul pada beberapa ulasan atau berulang kali dalam satu dokumen.

Dari lima ulasan yang dimiliki, kita memperoleh **11 kata unik**, yaitu berikut.

'adalah', 'belajar', 'dasar', 'ilmu', 'itu', 'matematika', 'menyenangkan', 'penting', 'sangat', 'saya', 'suka'

Kosakata ini akan menjadi fitur-fitur (kolom) dalam tabel representasi numerik.

#### Menghitung Frekuensi Kemunculan Kata per Dokumen

Setelah memiliki kosakata, kita menghitung **jumlah setiap kata muncul dalam masing-masing ulasan**.

*   Jika sebuah kata muncul, kita beri nilai sesuai dengan frekuensinya (contohnya: 1 jika muncul satu kali).
*   Jika tidak muncul, nilainya adalah 0.

Dengan cara ini, kita memperoleh **tabel representasi dokumen**, seperti gambar berikut.

![dos-9f3b8bb46135a28fc78ef0fd27090e5520250902232109.jpeg](https://assets.cdn.dicoding.com/original/academy/dos-9f3b8bb46135a28fc78ef0fd27090e5520250902232109.jpeg)

Setiap baris mewakili satu ulasan dan setiap kolom merepresentasikan sebuah kata dari kosakata. Misalnya berikut.

Ulasan ke-1 ("Saya suka belajar matematika") direpresentasikan sebagai \[0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1\]

#### Merepresentasikan Dokumen sebagai Vektor

Dari tabel tersebut, **setiap ulasan kini telah berubah menjadi sebuah vektor angka**.  
Inilah inti dari metode Bag of Words—mengubah teks menjadi representasi numerik berdasarkan **frekuensi kata**.

Vektor-vektor ini kemudian dapat digunakan oleh algoritma machine learning, seperti **Naive Bayes**, _**logistic regression**_, atau **SVM** untuk menganalisis dan mempelajari pola dari ulasan yang bersifat positif ataupun negatif.

- - -

Metode Bag of Words adalah pendekatan yang sederhana, tapi efektif untuk mengubah teks menjadi data numerik. 

Meski tidak memperhatikan struktur atau urutan kata, metode ini mampu menangkap **informasi penting dari keberadaan dan jumlah kata**, yang cukup untuk banyak kasus klasifikasi teks dasar.

  

### Implementasi BoW dengan Python

Pada bagian sebelumnya, kita sudah belajar konsep dan cara melakukan perhitungan manual menggunakan metode Bag of Words. Agar kemampuan kita semakin baik, pada bagian ini akan disajikan contoh implementasi metode BoW menggunakan bahasa pemrograman Python. 

Untungnya, kita tidak perlu menulis kode dari nol untuk menghitung frekuensi kata dan membuat vektor. Library `scikit-learn` pada Python menyediakan alat yang sangat praktis untuk ini, yaitu `CountVectorizer`.

1.  from sklearn.feature\_extraction.text import CountVectorizer
2.  import pandas as pd

4.  \# Data ulasan
5.  reviews \= \["Saya suka belajar Matematika.",
6.             "Belajar matematika menyenangkan.",
7.             "Matematika itu menyenangkan.",
8.             "Belajar matematika sangat penting.",
9.             "Matematika adalah ilmu dasar."
10.             \]

12.  \# Membuat objek CountVectorizer
13.  vectorizer \= CountVectorizer()

15.  \# Menghitung representasi BoW
16.  X \= vectorizer.fit\_transform(reviews)

18.  \# Menampilkan hasilnya dalam bentuk array
19.  bow\_array \= X.toarray()

21.  \# Menampilkan kata-kata dalam BoW
22.  words \= vectorizer.get\_feature\_names\_out()

24.  \# Menampilkan hasil representasi BoW
25.  print("Kata-kata dalam BoW:", words)
26.  print("jumlah kata dalam bow:", len(words))
27.  print("Representasi BoW:")
28.  bow\_array

30.  \# Menampilkan hasilnya dalam bentuk array dan konversi ke DataFrame
31.  bow\_array \= X.toarray()
32.  words \= vectorizer.get\_feature\_names\_out()

34.  \# Membuat DataFrame dari BoW
35.  df\_bow \= pd.DataFrame(bow\_array, columns\=words)
36.  df\_bow

Berikut adalah penjelasan singkat dan padat tentang kode di atas.

1.  **Data ulasan**: Kumpulan lima ulasan yang akan diproses menjadi representasi **Bag of Words (BoW)**.
2.  **CountVectorizer**: Digunakan untuk mengubah teks menjadi representasi BoW dengan menghitung frekuensi kemunculan kata-kata unik dalam ulasan.
3.  **fit\_transform()**: Menghitung dan mengubah ulasan menjadi vektor BoW.
4.  **toarray()**: Mengonversi hasil BoW yang berupa matriks _sparse_ menjadi array biasa.
5.  **get\_feature\_names\_out()**: Mengambil daftar kata-kata unik yang ditemukan dalam semua ulasan.
6.  **pandas DataFrame**: Hasil BoW disimpan dalam bentuk DataFrame untuk memudahkan pembacaan serta analisis lebih lanjut dengan kolom-kolom berupa kata-kata dan baris berupa ulasan.

Oke, kini Anda sudah mengetahui cara mengimplementasikan bag-of-word dalam bahasa pemrograman Python. Luar biasa!

  

### Pertimbangan Penggunaan BoW

Ada hal yang perlu Anda perhatikan saat berinteraksi dengan BoW, yaitu ia tidak peduli pada urutan kata dalam sebuah kalimat atau dokumen. Metode ini hanya menghitung keberadaan serta frekuensi kata sehingga struktur kalimat dan hubungan antar kata diabaikan. 

Misalkan ada dua kalimat ini: "Ilham **mengejar** Rina" dan "Rina **mengejar** Ilham". Dalam konteks bahasa manusia, kedua kalimat ini memiliki arti yang sangat berbeda karena subjek dan objeknya bertukar posisi. Namun, bagi BoW, kedua kalimat ini akan direpresentasikan oleh vektor yang identik (jika hanya menghitung frekuensi kata) sebab keduanya mengandung kata-kata yang sama ("Ilham", "mengejar", "Rina") dengan frekuensi sama.

Lebih jauh lagi, metode Bag of Words memiliki keterbatasan dalam **menangkap nuansa makna** yang bergantung pada **susunan kata** atau kehadiran kata negasi. Misalnya, bandingkan kalimat berikut.

*   _"Saya suka film ini"_
*   _"Saya tidak suka film ini"_

Secara struktur, kedua kalimat memiliki kata-kata yang hampir sama. Namun, jika kata “tidak” tak ada dalam kosakata (misalnya karena dihapus saat pra-pemrosesan), Bag of Words akan menganggap kedua kalimat tersebut **sangat mirip secara numerik**, padahal maknanya justru **berlawanan secara emosional**.

Hal ini menunjukkan bahwa Bag of Words **tidak mempertimbangkan konteks, urutan kata, atau relasi antar kata** sehingga kesulitan dalam memahami hal berikut.

*   **Negasi** (“tidak suka” ≠ “suka”)
*   **Sarkasme**
*   **Perubahan makna karena struktur kalimat**

Akibatnya, representasi BoW **tidak mampu menangkap makna semantik** yang lebih dalam dari teks.

## Matematika dalam Natural Language Processing (NLP): TF-IDF: Memberi Bobot Kata

Kita telah memahami bahwa metode Bag of Words (BoW) mengubah teks menjadi vektor numerik berdasarkan frekuensi kata. BoW adalah fondasi yang baik, tetapi kita juga telah mengidentifikasi beberapa kelemahan utamanya.

1.  **Tidak Mempertimbangkan Urutan Kata:** BoW tidak menangkap konteks atau makna yang berubah karena susunan kata (misalnya, "saya suka" vs. "suka saya").
2.  **Masalah Dimensi Tinggi dan Sparsity:** Untuk kosakata yang sangat besar, vektor menjadi sangat panjang dan sebagian besar isinya nol. Ini akan memakan banyak memori dan komputasi.
3.  **Tidak Membedakan Kata Penting dari Kata Umum:** Kata-kata yang sangat sering muncul (seperti "dan", "di", "yang") dianggap sama pentingnya dengan kata-kata spesifik yang lebih informatif hanya karena frekuensinya tinggi.

Nah, untuk mengatasi ketiga kelemahan ini, hadirlah teknik yang lebih canggih, tapi tetap berbasis frekuensi: **TF-IDF (Term Frequency-Inverse Document Frequency)**. 

  

### Konsep Inti TF-IDF: Menilai Pentingnya Kata dalam Konteks

**TF-IDF** (Term Frequency-Inverse Document Frequency) adalah teknik pembobotan kata yang populer dalam pencarian informasi dan _text mining_. Tujuannya adalah menilai seberapa penting suatu kata dalam sebuah dokumen, relatif terhadap seluruh kumpulan dokumen (korpus).

TF-IDF bekerja dengan memberikan bobot numerik kepada setiap kata dalam dokumen. Semakin tinggi bobot TF-IDF sebuah kata, semakin relevan atau unik kata tersebut bagi dokumen itu dalam konteks korpus yang lebih besar. 

### Komponen TF-IDF

TF-IDF terdiri dari dua komponen utama sebagai berikut.

#### Term Frequency (TF)

TF mengukur seberapa sering sebuah kata (atau _term_) muncul dalam **satu dokumen spesifik**. Semakin sering sebuah kata muncul dalam dokumen, semakin tinggi nilai TF-nya, yang mengindikasikan bahwa kata tersebut mungkin penting untuk dokumen itu.

Jika sebuah kata muncul berkali-kali dalam sebuah artikel, kemungkinan besar artikel tersebut memang membahas topik yang terkait dengan kata itu.

Ada beberapa cara menghitung TF, tetapi yang paling sederhana adalah berikut.

![dos-df9960a16b5fec23c27b8f3e0532dca620250903133522.png](https://assets.cdn.dicoding.com/original/academy/dos-df9960a16b5fec23c27b8f3e0532dca620250903133522.png)

**Contoh:** Jika sebuah kata muncul dua kali dari total sepuluh kata dalam dokumen, TF = 2/10 = 0.2

#### Inverse Document Frequency (IDF)

IDF mengukur seberapa "langka" atau "unik" sebuah kata dalam **seluruh koleksi dokumen (korpus)**. 

Jika sebuah kata muncul pada banyak dokumen dalam korpus, nilai IDF-nya akan rendah (karena kata itu umum). Sebaliknya, jika sebuah kata hanya muncul dalam sedikit dokumen (atau hanya pada satu dokumen saja), nilai IDF-nya akan tinggi (karena kata itu khas dan informatif).

Agar lebih jelas, mari lihat contoh cara IDF bekerja dalam membedakan kata-kata umum dan kata-kata khas pada korpus dokumen perfilman. Kata-kata seperti "dan", "di", atau "yang" muncul hampir pada semua dokumen sehingga tidak efektif untuk membedakan satu dokumen dari yang lain. 

Sebaliknya, kata-kata seperti "sinematografi" atau "plot twist" mungkin jarang muncul, tetapi jika ada, biasanya sangat spesifik dan relevan terhadap topik film. IDF dirancang untuk menurunkan bobot kata-kata umum dan menaikkan bobot kata-kata yang lebih khas serta informatif.

Berikut adalah cara untuk menghitung IDF.

![dos-c226ba570de0328ebc1b2a58173cf58f20250903133612.png](https://assets.cdn.dicoding.com/original/academy/dos-c226ba570de0328ebc1b2a58173cf58f20250903133612.png)

Misalkan kita memiliki lima dokumen ulasan film dan kata yang ingin dihitung IDF-nya adalah

**t = "sinematografi"**

Lalu diketahui

*   N = 5 (jumlah total dokumen)
*   df("sinematografi") = 1 (kata “sinematografi” hanya muncul dalam 1 dokumen)

Jadi, rumusnya berikut.

![dos-74a9d56acb664cde366f88fb4b3af80920250903133649.png](https://assets.cdn.dicoding.com/original/academy/dos-74a9d56acb664cde366f88fb4b3af80920250903133649.png)

Artinya, karena kata “sinematografi” hanya muncul dalam satu dokumen, nilainya tinggi. Ini menandakan bahwa kata tersebut cukup unik dan relevan secara spesifik untuk dokumen itu.

Sebaliknya, coba kita hitung untuk kata berikut.

**t = "yang"**

*   df("yang") = 5 (muncul pada semua dokumen)  
    ![dos-54da820e95e2737dc97591209678dce020250903133713.png](https://assets.cdn.dicoding.com/original/academy/dos-54da820e95e2737dc97591209678dce020250903133713.png)

Nilainya nol karena kata “yang” terlalu umum—tidak membantu membedakan dokumen satu dengan yang lain.

Untuk menghindari pembagian dengan nol jika sebuah kata tidak muncul dalam dokumen mana pun–atau untuk memberikan bobot pada kata yang muncul dalam semua dokumen–sering kali ditambahkan +1 pada pembilang dan/atau penyebut, misalnya seperti ini.

![dos-4ac0447e815ccf76ab7e3003ccbf5f1720250903133749.png](https://assets.cdn.dicoding.com/original/academy/dos-4ac0447e815ccf76ab7e3003ccbf5f1720250903133749.png)

  

### Perhitungan TF-IDF

Setelah menghitung nilai _**term frequency**_ **(TF)** dan _**inverse document frequency**_ **(IDF)** untuk setiap kata dalam dokumen, kita memperoleh bobot akhir **TF-IDF** dengan mengalikan kedua komponen tersebut.

Persamaannya berikut.

**![dos-516d67b481bc79d81cbec85be94df06c20250903133806.png](https://assets.cdn.dicoding.com/original/academy/dos-516d67b481bc79d81cbec85be94df06c20250903133806.png)**

Sebuah kata akan memiliki bobot TF-IDF yang tinggi jika ia sering muncul dalam sebuah dokumen tertentu (TF tinggi) **dan** jarang muncul dalam dokumen-dokumen lain pada korpus (IDF tinggi). Ini adalah kombinasi yang menunjukkan relevansi dan kekhasan.

Bayangkan kita punya **lima dokumen ulasan film**. Kita ingin menghitung TF-IDF untuk kata **"sinematografi"** dalam **Dokumen 2** dengan informasi sebagai berikut.

Kata **"sinematografi"** muncul sebanyak **3 kali** pada **Dokumen 2**, yang memiliki total **100 kata**. Jadi, penghitungan TF sebagai berikut.

![dos-97f55adcbae0ff45b19243690ac0e86220250903133844.png](https://assets.cdn.dicoding.com/original/academy/dos-97f55adcbae0ff45b19243690ac0e86220250903133844.png)

Kemudian diketahui bahwa kata **"sinematografi"** hanya muncul dalam **1 dari 5 dokumen**. Jadi, penghitungan IDF sebagai berikut.

![dos-353a97227e32c45936350ca57510467320250903134018.png](https://assets.cdn.dicoding.com/original/academy/dos-353a97227e32c45936350ca57510467320250903134018.png)

Dengan demikian, hasil akhir penghitungan TF-IDF adalah berikut.

![dos-1594a54b6c631a1cf93661ea4d7a77e620250903134031.png](https://assets.cdn.dicoding.com/original/academy/dos-1594a54b6c631a1cf93661ea4d7a77e620250903134031.png)

Artinya, bobot **0.02097** menunjukkan bahwa kata _"sinematografi"_ cukup penting dalam Dokumen 2 karena cukup sering muncul dan jarang ada pada dokumen lain. 

  

### Menggunakan TF-IDF untuk Ulasan Matematika

Kali ini kita akan kembali dalam skenario ulasan matematika yang kita gunakan pada materi sebelumnya. Kita akan mengubah ulasan teks menjadi representasi angka menggunakan metode **TF-IDF** dengan langkah-langkah berikut.

#### Menyusun Kosakata Unik (Vocabulary)

Langkah pertama adalah seperti yang sudah dilakukan dalam **Bag of Words (BoW)**, yaitu menyusun kosakata unik yang muncul di seluruh kumpulan ulasan pembelajaran matematika. 

Berikut adalah beberapa contoh ulasan yang berhasil dikumpulkan.

*   **Dokumen 1:** "Saya suka belajar Matematika."
*   **Dokumen 2:** "Belajar matematika menyenangkan."
*   **Dokumen 3:** "Matematika itu menyenangkan."
*   **Dokumen 4:** "Belajar matematika sangat penting."
*   **Dokumen 5:** "Matematika adalah ilmu dasar."

Dari ulasan di atas, kosakata unik yang dihasilkan adalah sebelas kata.

'adalah', 'belajar', 'dasar', 'ilmu', 'itu', 'matematika', 'menyenangkan', 'penting', 'sangat', 'saya', 'suka'

#### Menghitung Term Frequency (TF)

**Term frequency (TF)** mengukur seberapa sering sebuah kata muncul dalam suatu dokumen (ulasan). Rumusnya adalah berikut.

![dos-249b593600fb0c5e2346487eef8c277220250903134129.png](https://assets.cdn.dicoding.com/original/academy/dos-249b593600fb0c5e2346487eef8c277220250903134129.png)

Dalam kasus ini, jumlah total kata dalam “Dokumen 1” adalah 4, yakni (\['Saya', 'suka', 'belajar', 'Matematika'\]). Kita hitung **TF** untuk setiap kata seperti berikut.

![dos-23d51065f4069fc477238b9334db547420250903134202.png](https://assets.cdn.dicoding.com/original/academy/dos-23d51065f4069fc477238b9334db547420250903134202.png)

Dengan cara yang sama kita bisa menghitung TF untuk masing-masing dokumen atau ulasan sehingga akan mendapatkan hasil berikut.

#### Menghitung Inverse Document Frequency (IDF)

**IDF** mengukur seberapa penting suatu kata dalam seluruh koleksi dokumen. Semakin jarang kata muncul dalam koleksi, semakin tinggi nilai IDF-nya. **IDF** sangat berguna untuk menurunkan bobot kata yang sering muncul dalam semua dokumen, misalnya kata-kata umum seperti "dan" atau "adalah" yang tidak memberikan banyak informasi.

![dos-755bc506a6ecd8ac30c20f952da3050d20250903134245.png](https://assets.cdn.dicoding.com/original/academy/dos-755bc506a6ecd8ac30c20f952da3050d20250903134245.png)

Pada perhitungan di atas, kata "Matematika" muncul dalam **setiap dokumen**. Jadi, kita simpulkan bahwa

*   **df("Matematika") = 5** (yaitu 5 dokumen).
*   **N = 5** (total jumlah dokumen).
*   Hasilnya, **IDF("Matematika")** = 1

Karena kata ini muncul dalam setiap dokumen, **IDF-nya menjadi 0**, yang berarti kata ini tidak memberikan banyak informasi untuk membedakan dokumen. 

Dengan cara yang sama, kita bisa menghitung IDF untuk setiap kata unik sehingga akan mendapatkan hasil berikut.

![dos-4f05431c9e9cad461cc5b19f353a292d20250903134303.png](https://assets.cdn.dicoding.com/original/academy/dos-4f05431c9e9cad461cc5b19f353a292d20250903134303.png)

#### Menghitung TF-IDF

Setelah menghitung **TF** dan **IDF**, kita dapat mengalikan keduanya untuk mendapatkan **TF-IDF**. **TF-IDF** mengukur pentingnya sebuah kata dalam dokumen dengan mempertimbangkan seberapa sering kata tersebut muncul dalam dokumen (TF) dan seberapa jarang kata tersebut muncul pada seluruh dokumen (IDF).

Rumus TF-IDF adalah berikut.

![dos-2d7205d8d2564e8cfdfe5d6a4ddbd62220250903134335.png](https://assets.cdn.dicoding.com/original/academy/dos-2d7205d8d2564e8cfdfe5d6a4ddbd62220250903134335.png)

Dari data TF dan IDF yang sudah dihitung sebelumnya, kita dapat menghitung nilai TF-IDF untuk setiap kata pada **Dokumen 1**.

![dos-11779a26c295c7d053ba8aa549a60c8f20250903134350.png](https://assets.cdn.dicoding.com/original/academy/dos-11779a26c295c7d053ba8aa549a60c8f20250903134350.png)

Dengan cara yang sama, kita dapat menghitung TF-IDF untuk setiap kata pada dokumen sehingga akan mendapatkan hasil berikut. 

![dos-25663ed469a5fd657491ca0d42bbaa7720250903134413.png](https://assets.cdn.dicoding.com/original/academy/dos-25663ed469a5fd657491ca0d42bbaa7720250903134413.png)

Dari hasil TF-IDF di atas, kita dapat melihat bahwa nilai bobot untuk kata **matematika** berbeda dalam tiap dokumennya, bergantung pada kata lain dalam dokumen tersebut. Berdasarkan data tersebut, kita dapat mendapat kesimpulan sebagai berikut. 

*   **TF** mengukur frekuensi kemunculan kata dalam satu dokumen.
*   **IDF** mengukur seberapa penting kata tersebut di seluruh koleksi dokumen.
*   **TF-IDF** adalah perkalian antara **TF** dan **IDF**, yang memberikan bobot lebih pada kata-kata penting untuk membedakan dokumen.

Ini adalah cara **manual** menghitung **TF**, **IDF**, dan **TF-IDF** yang dapat digunakan dalam tugas seperti klasifikasi teks atau pencarian informasi.

  

### Implementasi TF-IDF dengan Python

Setelah memahami konsep dasar dan perhitungan manual TF-IDF, saatnya kita lihat cara mengimplementasikannya dengan efisien menggunakan Python. 

Kode yang akan dibahas ini bertujuan untuk menghitung **TF (**_**term frequency**_**)**, **IDF (**_**inverse document frequency**_**)**, dan **TF-IDF** dari kumpulan data ulasan teks menggunakan Python, lalu menyajikan hasilnya dalam bentuk tabel menggunakan `pandas`.

#### Langkah 1: Persiapan Library dan Data

Pertama, kita perlu mengimpor beberapa library terlebih dahulu, antara lain berikut.

*   `pandas` untuk manipulasi data tabular.
*   `math` digunakan untuk perhitungan logaritma.
*   `Counter` dari collections digunakan untuk menghitung frekuensi kata.

Selain itu, kita juga menyiapkan data ulasan yang akan dipakai. Variabel reviews adalah kumpulan dokumen berupa ulasan pendek yang akan dianalisis.

1.  import pandas as pd
2.  import math
3.  from collections import Counter

5.  \# Data ulasan
6.  reviews \= \[
7.     "saya suka belajar matematika",
8.     "belajar matematika menyenangkan",
9.     "matematika itu menyenangkan",
10.     "belajar matematika sangat penting",
11.     "matematika adalah ilmu dasar"
12.  \]

#### Langkah 2: Membangun Kosakata

Kemudian, kita menggabungkan semua teks, menjadikannya huruf kecil, memecah menjadi kata, lalu membuat daftar kata unik yang telah diurutkan.

1.  \# Membuat daftar kata unik di seluruh dokumen (ubah set menjadi list dan urutkan)
2.  all\_words \= sorted(list(set(" ".join(reviews).lower().split())))

#### Langkah 3: Menghitung Term Frequency (TF)

Selanjutnya, kita menghitung proporsi kemunculan setiap kata dalam satu dokumen merujuk pada rumus ini **TF = (jumlah kata tertentu) / (total kata dalam dokumen)**.

1.  \# Menghitung TF untuk setiap dokumen
2.  def compute\_tf(doc):
3.     word\_count \= Counter(doc.lower().split())
4.     total\_words \= len(doc.split())
5.     tf \= {word: count / total\_words for word, count in word\_count.items()}
6.     return tf

#### Langkah 4: Menghitung Inverse Document Frequency (IDF)

Setelah itu, kita menghitung seberapa umum atau jarang suatu kata muncul dalam seluruh dokumen dengan rumus: **IDF = log10((N + 1) / (df + 1)) + 1**

*   **N** = jumlah dokumen
*   **df** = jumlah dokumen yang mengandung kata tersebut
*   Tambahan **+1** untuk mencegah pembagian nol atau log(0)

1.  \# Menghitung IDF untuk seluruh koleksi dokumen
2.  def compute\_idf(reviews, all\_words):
3.     N \= len(reviews)
4.     idf \= {}
5.     for word in all\_words:
6.         df \= sum(1 for review in reviews if word in review.lower())
7.         idf\[word\] \= math.log10((N + 1) / (df + 1)) + 1  \# Ditambahkan +1 untuk mencegah pembagian dengan nol
8.     return idf

#### Langkah 5: Menghitung TF-IDF

Langkah berikutnya adalah kita mengalikan nilai TF dengan IDF untuk setiap kata pada setiap dokumen. Di sini kita memberikan bobot yang menekankan kata-kata penting (unik untuk dokumen tertentu).

1.  \# Menghitung TF-IDF
2.  def compute\_tfidf(reviews, all\_words):
3.     idf \= compute\_idf(reviews, all\_words)
4.     tfidf \= \[\]
5.     for review in reviews:
6.         tf \= compute\_tf(review)
7.         tfidf.append({word: tf.get(word, 0) \* idf\[word\] for word in all\_words})
8.     return tfidf

#### Langkah 6: Mengeksekusi Perhitungan

Setelah kita membuat fungsi untuk menghitung TF, IDF, dan TF-IDF, kini saatnya mengeksekusi fungsi tersebut.

1.  tf\_results \= \[compute\_tf(review) for review in reviews\]
2.  idf\_results \= compute\_idf(reviews, all\_words)
3.  tfidf\_results \= compute\_tfidf(reviews, all\_words)

#### Langkah 7: Membuat Tabel Pandas

Untuk memudahkan dalam melihat hasil perhitungan TF, IDF, dan TF-IDF sebelumnya, kali ini kita membuat tabel data frame untuk masing-masing.

Kode ini akan menampilkan frekuensi relatif kata pada tiap dokumen.

1.  \# Membuat DataFrame untuk TF
2.  df\_tf \= pd.DataFrame(tf\_results, columns\=all\_words)
3.  df\_tf.index \= \[f"Dokumen {i+1}" for i in range(len(tf\_results))\]
4.  df\_tf \= df\_tf.fillna(0).round(2)
5.  df\_tf

Inilah hasil untuk tabel TF yang menunjukkan seberapa sering kata muncul dalam dokumen.

![dos-7d15ac5a3cb3e6166e38d18b6f8c3daf20250903135156.png](https://assets.cdn.dicoding.com/original/academy/dos-7d15ac5a3cb3e6166e38d18b6f8c3daf20250903135156.png)

Kemudian, kode berikut akan menampilkan nilai IDF dari setiap kata dalam kosakata.

1.  df\_idf \= pd.DataFrame(list(idf\_results.items()), columns\=\["Kata", "IDF"\])
2.  df\_idf \= df\_idf.round(2)
3.  df\_idf

Inilah tampilan hasil untuk tabel IDF yang menunjukkan seberapa unik kata dalam seluruh dokumen.

![dos-1d06966878d45bcb43d78717e0b077ec20250903135244.png](https://assets.cdn.dicoding.com/original/academy/dos-1d06966878d45bcb43d78717e0b077ec20250903135244.png)

Terakhir, kode berikut menampilkan bobot pentingnya kata pada tiap dokumen berdasarkan kombinasi TF dan IDF.

1.  df\_tfidf \= pd.DataFrame(tfidf\_results, columns\=all\_words)
2.  df\_tfidf.index \= \[f"Dokumen {i+1}" for i in range(len(tfidf\_results))\]
3.  df\_tfidf \= df\_tfidf.round(2)
4.  df\_tfidf

Inilah tampilan hasil untuk tabel TF-IDF yang menunjukkan bobot pentingnya kata dalam konteks dokumen.

![dos-d741297dfef2acc4a14eb6dab479c4cf20250903135330.png](https://assets.cdn.dicoding.com/original/academy/dos-d741297dfef2acc4a14eb6dab479c4cf20250903135330.png)

Perhatikan tabel TF-IDF di atas. Setiap baris mewakili satu ulasan, sementara setiap kolom merepresentasikan satu kata unik dari seluruh kumpulan ulasan (atau yang kita sebut **korpus**). Nilai yang kita lihat pada setiap sel adalah **bobot TF-IDF** untuk kata tertentu dalam ulasan spesifik.

Nilai **bobot yang lebih tinggi** mengindikasikan bahwa kata tersebut sangat **relevan dan spesifik** untuk ulasan itu. Ini biasanya berarti kata tersebut sering muncul dalam ulasan tersebut, tetapi jarang atau bahkan tidak muncul sama sekali pada ulasan lain dalam korpus. 

Menariknya, Anda akan melihat bahwa kata _"matematika"_ memiliki bobot TF-IDF yang relatif seragam dalam semua ulasan. Ini terjadi karena _"matematika"_ muncul pada setiap ulasan dalam korpus. 

Dalam filosofi TF-IDF, kata-kata yang sering muncul pada banyak dokumen biasanya memiliki nilai IDF lebih rendah karena dianggap kurang mampu membedakan satu dokumen dari yang lain. Namun karena kemunculannya tetap proporsional dan tidak terlalu dominan, nilai TF-IDF _"matematika"_ tetap memiliki bobot sedang. 

Sebaliknya, kata-kata seperti _"saya"_, _"suka"_, _"itu"_, _"sangat"_, _"penting"_, _"adalah"_, _"ilmu"_, dan _"dasar"_ bisa memiliki bobot TF-IDF lebih tinggi ketika hanya muncul pada satu atau dua ulasan, menjadikannya lebih khas untuk dokumen tempat mereka muncul.

  

### Pertimbangan Penggunaan TF-IDF

Oke, sampai saat ini kita sudah berhasil menghitung TF-IDF menggunakan Python. Ada beberapa hal yang harus Anda perhatikan saat menggunakan teknik TF-IDF ini.

Meskipun TF-IDF lebih unggul dibandingkan metode Bag of Words (BoW) karena lebih informatif dengan mengurangi bobot kata umum, tetap saja pendekatan TF-IDF ini masih memiliki beberapa keterbatasan penting yang mirip dengan BoW.

*   Pertama, **TF-IDF tidak mempertimbangkan urutan kata**. Artinya, dua kalimat dengan susunan kata berbeda, tapi makna yang sangat berbeda tetap bisa dianggap serupa. Sebagai contoh, kalimat _"Ilham mengejar Rina"_ dan _"Rina mengejar Ilham"_ akan diberikan bobot kata yang sama oleh TF-IDF meskipun maknanya jelas berbeda.
*   Kedua, **TF-IDF kesulitan menangkap konteks negasi atau ironi**. Misalnya, kalimat _"Saya suka film ini"_ dan _"Saya tidak suka film ini"_ dapat dianggap mirip jika kata _"tidak"_ diabaikan selama pra-pemrosesan, padahal maknanya berlawanan.
*   Ketiga, **TF-IDF tidak memahami makna semantik antar kata**. Kalimat seperti _"Saya ingin membeli mobil baru"_ dan _"Saya ingin membeli rumah baru"_ mungkin dianggap serupa, padahal secara makna dan konteks sangat berbeda.

Oleh karena itu, untuk menangkap konteks, makna, dan hubungan antar kata secara lebih mendalam, pendekatan lanjutan seperti **Word Embedding** lebih sesuai.

## Matematika dalam Natural Language Processing (NLP): Word Embedding

**Word Embedding** adalah teknik pada **natural language processing (NLP)** yang digunakan untuk mengonversi kata-kata dalam teks menjadi representasi numerik, biasanya berbentuk vektor berdimensi rendah. Tujuannya adalah menangkap **makna semantik** dan **hubungan antar kata** dalam sebuah korpus teks. 

Dengan **Word Embedding**, kata-kata yang memiliki makna serupa atau sering muncul dalam konteks serupa akan memiliki representasi numerik lebih mirip, memungkinkan mesin untuk "memahami" dan memproses bahasa manusia dengan lebih efektif.

Berbeda dengan teknik lama, seperti **Bag of Words (BoW)** atau **TF-IDF** yang hanya memperhitungkan frekuensi kata, **Word Embedding** berfokus pada **makna** kata dalam konteks kalimat sehingga menghasilkan representasi kata yang lebih kaya dan bermakna.

  

### Cara Kerja Word Embedding

Dalam **Word Embedding**, setiap kata dalam vocabulary direpresentasikan sebagai **vektor numerik berdimensi rendah**, biasanya terdiri dari puluhan hingga ratusan dimensi. Dimensi-dimensi ini tidak ditentukan secara manual, tetapi dipelajari secara otomatis melalui proses pelatihan algoritma yang memanfaatkan konteks kemunculan kata dalam teks.

![dos-fdb6500b17f07cfbbf0daead2f18c0fd20250903135605.jpeg](https://assets.cdn.dicoding.com/original/academy/dos-fdb6500b17f07cfbbf0daead2f18c0fd20250903135605.jpeg)

Sebagai contoh, dalam **Word Embedding** terjadi hal berikut.

*   Kata _"cat"_ dan _"kitten"_ akan memiliki vektor yang **mirip** karena keduanya memiliki makna yang berkaitan—yakni sama-sama merujuk pada hewan kucing.
*   Sebaliknya, kata _"cat"_ dan _"car"_, yang memiliki makna sangat berbeda, akan direpresentasikan oleh vektor yang **berjauhan** dalam ruang vektor.

  

### Alasan Word Embedding Lebih Unggul dari Representasi Tradisional

Berikut adalah beberapa keunggulan utama dari Word Embedding dibandingkan teknik lain, seperti Bag of Words (BoW) dan TF-IDF.

1.  **Efisien dan Padat (**_**Compact Representation**_**)**  
    Berbeda dengan BoW atau TF-IDF yang menghasilkan vektor sangat besar dan sparse (banyak nilai nol), Word Embedding merepresentasikan kata dalam vektor padat berdimensi rendah (biasanya puluhan hingga ratusan dimensi) sehingga membuat analisis teks lebih efisien secara komputasi untuk disimpan serta diproses.
2.  **Menangkap Konteks dan Makna**  
    Word Embedding dapat membedakan makna kata yang sama berdasarkan konteks penggunaannya. Misalnya, kata _"apel"_ dalam kalimat _"Saya makan apel setelah makan siang"_ akan direpresentasikan secara berbeda dibandingkan dengan _"Upacara apel di sekolah dimulai pukul tujuh pagi"_. Meskipun sama-sama menggunakan kata _"apel"_, maknanya berbeda—yang satu merujuk pada buah, sedangkan yang lain merujuk pada kegiatan formal di sekolah. Word Embedding mampu mengenali perbedaan ini dengan melihat kata-kata di sekitarnya.
3.  **Mengenali Kemiripan Makna (Sinonim)**  
    Kata-kata yang sering digunakan dalam konteks yang sama—seperti _"dokter"_ dan _"perawat"_—akan direpresentasikan dengan vektor mirip. Ini tidak dapat dicapai oleh BoW atau TF-IDF.
4.  **Menangkap Pola Semantik Lebih Dalam**  
    Word Embedding juga dapat mengenali pola semantik atau asosiasi antar kata berdasarkan konteks global. Misalnya, kata _"sekolah"_ dan _"belajar"_, _"uang"_ dan _"transaksi"_, atau “bank” dan “uang” sering muncul bersama dalam konteks serupa sehingga vektornya akan lebih mirip dibandingkan kata-kata yang tidak terkait.

  

### Jenis Word Embedding

Salah satu metode paling populer dalam membangun Word Embedding adalah **Word2Vec**. Metode ini dirancang untuk mempelajari representasi vektor dari kata-kata berdasarkan konteksnya dalam kalimat. 

Word2Vec memiliki dua arsitektur utama dengan pendekatan yang berbeda dalam memahami hubungan antar kata, yaitu **CBOW (Continuous Bag of Words)** dan **Skip-gram**. Meskipun keduanya untuk menghasilkan embedding berkualitas tinggi, cara kerja internal dan kekuatan masing-masing model memiliki perbedaan yang khas.

![dos-04ab37b295c0ba9097bfa3243537b71420250903135605.jpeg](https://assets.cdn.dicoding.com/original/academy/dos-04ab37b295c0ba9097bfa3243537b71420250903135605.jpeg)

#### CBOW (Continuous Bag of Words)

**CBOW** memprediksi kata target berdasarkan kata-kata konteks di sekitarnya. Dalam hal ini, CBOW menangkap hubungan antar kata dalam konteks tertentu, yang sangat berguna untuk memahami makna kata pada konteks kalimat.  

Pendekatan ini efektif untuk mempelajari makna kata dalam kalimat dengan asumsi bahwa kata-kata di sekitar membawa informasi penting.

Misalkan kita memiliki kalimat berikut: **"saya suka belajar matematika"**.

Jika kata **"suka"** adalah kata target yang ingin diprediksi, kata-kata **"saya"**, **"belajar"**, dan **"matematika"** menjadi kata konteks yang digunakan untuk memprediksi kata target "suka".

Supaya lebih jelas, berikut adalah cara kerja dari CBOW.

1.  **Input**: Kata-kata konteks (di sekitar kata target).
2.  **Output**: Kata target.
3.  **Jendela Konteks**: CBOW menggunakan jendela konteks yang mengelilingi kata target. Misalnya, jendela konteks adalah dua maka dua kata sebelum dan dua kata setelah kata target digunakan untuk memprediksi kata tersebut.

CBOW menggunakan kata-kata konteks untuk memprediksi kata target. Dengan kata lain, model ini mengasumsikan bahwa kata-kata yang berada dalam jendela konteks (di sekitar kata target) membawa informasi penting untuk menentukan kata yang tengah.

Dengan menggunakan **CBOW**, model mempelajari vektor representasi dari kata-kata konteks–seperti (**"Saya"**, **"belajar"**, **"Matematika"**)–dan menggunakannya untuk memprediksi kata target–misalnya (**"suka"**). Ini membuat CBOW efisien dalam pelatihan dan cocok untuk dataset besar.

#### Skip-gram

**Skip-gram** melakukan kebalikan dari CBOW: ia memprediksi kata-kata konteks berdasarkan kata target. Dengan ini, Skip-gram fokus pada mempelajari hubungan semantik antar kata dengan cara memperkirakan kata-kata di sekitar kata target. 

Ini membuatnya sangat efektif untuk menangkap kata-kata langka dan hubungan semantik yang lebih kompleks karena dapat memanfaatkan kata target guna memprediksi banyak kata konteks yang ada di sekitarnya. 

Misalkan kita memiliki kalimat yang sama: **"saya suka belajar matematika"**.

Jika kata **"suka"** adalah kata target, **Skip-gram** akan mencoba memprediksi kata-kata konteks **"saya"**, **"belajar"**, dan **"matematika"** dari kata target **"suka"**.

Skip-gram bekerja dengan cara yang berfokus pada kata target dan mencoba untuk memprediksi banyak kata konteks yang mengelilinginya.

Jelasnya, berikut adalah cara kerja dari metode skip-gram.

1.  **Input**: Kata target (misalnya, "suka").
2.  **Output**: Kata-kata konteks (misalnya, \["Saya", "belajar", "Matematika"\]).
3.  **Jendela Konteks**: Sama seperti CBOW, **Skip-gram** juga menggunakan jendela konteks untuk menentukan kata-kata yang akan diprediksi berdasarkan kata target.

Intinya, Skip-gram mempelajari representasi vektor dari kata target–misalnya (**"suka"**)–dengan memanfaatkannya untuk memprediksi kata-kata konteks yang mengelilinginya–seperti (**"Saya"**, **"belajar"**, **"Matematika"**).

#### Perbandingan CBOW dan Skip-gram

Setelah memahami cara kerja masing-masing arsitektur, kita dapat melihat kelebihan serta kekurangan dari CBOW dan Skip-gram secara berdampingan. Meskipun keduanya memiliki tujuan sama, pendekatan yang digunakan memberikan karakteristik performa dan hasil embedding berbeda.

*   CBOW lebih unggul dalam hal efisiensi pelatihan dan sangat cocok digunakan pada dataset berukuran besar. Namun, pendekatan ini kurang efektif dalam menangkap makna kata-kata yang jarang muncul.
*   Sebaliknya, Skip-gram lebih andal dalam mempelajari hubungan semantik antar kata, terutama untuk kata-kata langka. Meskipun demikian, metode ini membutuhkan waktu pelatihan yang lebih lama dibandingkan CBOW.

Kedua arsitektur ini—CBOW dan Skip-gram—merupakan fondasi utama dari Word2Vec. Keduanya memungkinkan kata-kata dikonversi menjadi representasi numerik (embedding) yang dapat dipahami dan diolah oleh mesin. Meskipun berbeda dalam cara mereka memproses kata konteks dan kata target, keduanya berperan penting untuk membangun model bahasa yang lebih canggih serta efisien.

  

### Implementasi Word Embedding dengan Python

Untuk merasakan langsung cara Word2Vec bekerja, kita akan menggunakan _library_ gensim dalam Python. `gensim` adalah _library_ NLP _open-source_ yang menyediakan implementasi `Word2Vec` yang sangat dioptimalkan dan mudah digunakan. 

Kode yang akan dibahas dalam materi ini bertujuan untuk melatih dua model Word2Vec menggunakan dua pendekatan berbeda—**CBOW (Continuous Bag of Words)** dan **Skip-gram**—dengan dataset sederhana dalam bahasa Indonesia. Setelah pelatihan, model digunakan untuk menampilkan vektor representasi kata dan kata-kata yang paling mirip dengan kata “matematika”.

#### Langkah 1: Impor Library

Pertama, kita mengimpor class `Word2Vec` dari library `gensim` yang digunakan untuk membangun dan melatih model word embedding.

\# Import library yang diperlukan
from gensim.models import Word2Vec

#### Langkah 2: Dataset Contoh

Selanjutnya, mari kita membuat dataset contoh yang terdiri dari lima kalimat sederhana yang sudah dipisahkan per kata (_tokenized_). Format input ini sesuai dengan kebutuhan gensim, yaitu list of list of tokens (_nested list_).

\# Dataset contoh
sentences = \[
   \['saya', 'suka', 'belajar', 'matematika'\],
   \['belajar', 'matematika', 'sangat', 'menyenangkan'\],
   \['matematika', 'itu', 'menyenangkan'\],
   \['belajar', 'matematika', 'penting'\],
   \['matematika', 'adalah', 'ilmu', 'dasar'\]
\]

#### Langkah 3: Pelatihan Model CBOW

Kemudian, kita gunakan kode berikut untuk melatih model menggunakan CBOW.

\# Pelatihan menggunakan CBOW
cbow\_model = Word2Vec(sentences, vector\_size=50, window=3, min\_count=1, sg=0)  # sg=0 untuk CBOW

Berikut penjelasannya.

*   `vector_size=50` → Setiap kata akan direpresentasikan dalam vektor berdimensi 50.
*   `window=3` → Ukuran jendela konteks sejauh 3 kata di kiri dan kanan target.
*   `min_count=1` → Kata dengan frekuensi 1 pun tetap dimasukkan.
*   `sg=0` → Menandakan bahwa arsitektur yang digunakan adalah CBOW.

#### Langkah 4: Pelatihan Model Skip-gram

Selanjutnya, kita gunakan kode ini untuk melatih model menggunakan skip-gram.

\# Pelatihan menggunakan Skip-gram
skipgram\_model = Word2Vec(sentences, vector\_size=50, window=3, min\_count=1, sg=1)  # sg=1 untuk Skip-gram

Untuk detail penjelasannya sama seperti CBOW, tetapi bagian `sg=1` artinya model menggunakan arsitektur Skip-gram. Skip-gram memprediksi konteks berdasarkan kata target.

#### Langkah 5: Menyimpan Model

Kedua model disimpan dalam file `.bin` agar bisa digunakan kembali tanpa perlu melatih ulang.

\# Menyimpan model untuk evaluasi lebih lanjut
cbow\_model.save("cbow\_model.bin")
skipgram\_model.save("skipgram\_model.bin")

#### Langkah 6: Melihat Vektor Representasi

Pada kode berikut, kita menampilkan vektor representasi dari kata _"matematika"_ yang telah dipelajari oleh masing-masing model.

\# Menampilkan vektor untuk kata tertentu
print("Vektor kata 'matematika' dari model CBOW:")
print(cbow\_model.wv\['matematika'\])
print("\\nVektor kata 'matematika' dari model Skip-gram:")
print(skipgram\_model.wv\['matematika'\])

#### Langkah 7: Melihat Kemiripan antara Dua Kata

Kemudian, untuk membandingkan kemiripan kata "saya" dan "belajar", kita gunakan baris kode berikut. 

\# Mengukur kesamaan antara dua kata dalam kedua model
similarity\_cbow = cbow\_model.wv.similarity('saya', 'belajar')
similarity\_skipgram = skipgram\_model.wv.similarity('saya', 'belajar')
print("\\nKesamaan antara 'saya' dan 'belajar' dalam model CBOW:")
print(similarity\_cbow)
print("\\nKesamaan antara 'saya' dan 'belajar' dalam model Skip-gram:")
print(similarity\_skipgram)

Fungsi `similarity(kata1, kata2)` dari `gensim` akan menghitung **cosine similarity** antara dua vektor kata. Nilai hasilnya berada dalam rentang **\-1 sampai 1**.

*   Semakin mendekati **1** → semakin mirip secara semantik
*   Semakin mendekati **\-1** → semakin berbeda atau tidak terkait

#### Langkah 8: Melihat Kata yang Paling Mirip

Terakhir, kita mengambil tiga kata yang paling mirip dengan _"matematika"_ berdasarkan **similaritas kosinus** dari representasi vektor. Ini digunakan untuk melihat jika model berhasil memahami hubungan semantik antar kata.

\# Menampilkan kata yang paling mirip dengan 'matematika' dalam kedua model
print("\\nKata-kata yang paling mirip dengan 'matematika' dalam model CBOW:")
print(cbow\_model.wv.most\_similar('matematika', topn=3))
print("\\nKata-kata yang paling mirip dengan 'matematika' dalam model Skip-gram:")
print(skipgram\_model.wv.most\_similar('matematika', topn=3))

Setelah menjalankan kode di atas, Anda akan melihat serangkaian _output_ yang memberikan gambaran nyata tentang cara Word Embeddings bekerja.

Pertama, Anda akan disajikan dengan **vektor kata** untuk "matematika" dari kedua model (CBOW dan Skip-gram). 

![dos-c75bfc6015339e8ccadfe46e33e6d03420250903135605.jpeg](https://assets.cdn.dicoding.com/original/academy/dos-c75bfc6015339e8ccadfe46e33e6d03420250903135605.jpeg)

Vektor-vektor ini adalah serangkaian angka (dalam kasus kita, 50 angka) yang secara matematis merepresentasikan makna kata tersebut dalam konteks dataset pelatihan. Penting untuk dicatat bahwa vektor-vektor ini akan berbeda antara model CBOW dan Skip-gram karena masing-masing arsitektur mempelajari representasi kata dengan cara yang unik berdasarkan tujuan prediksinya.

Selanjutnya, Anda akan melihat **nilai kesamaan** antara kata "saya" dan "belajar" dalam kedua model. 

![dos-6383c7c1325c13fc524f5b1bb1e56dd520250903135604.jpeg](https://assets.cdn.dicoding.com/original/academy/dos-6383c7c1325c13fc524f5b1bb1e56dd520250903135604.jpeg)

Nilai ini, yang biasanya dihitung menggunakan _cosine similarity_, mengindikasikan seberapa dekat makna kedua kata tersebut dalam ruang vektor yang dilatih. Nilai yang mendekati 1 menunjukkan kesamaan semantik yang tinggi, sementara nilai yang mendekati -1 menunjukkan ketidakmiripan signifikan. 

Nilai negatif, seperti pada gambar, menunjukkan tidak ada hubungan semantik yang terbangun, kemungkinan karena dataset terlalu kecil.

Perlu diingat bahwa dengan dataset yang sangat kecil seperti contoh kita, hasil kesamaan mungkin tidak selalu intuitif atau mencerminkan kesamaan semantik dunia nyata secara sempurna. Ini karena model memiliki konteks yang terbatas untuk belajar hubungan antar kata.

Terakhir, fungsi **most\_similar** akan menampilkan daftar kata-kata yang vektornya paling dekat dengan vektor "matematika". 

![dos-d14042b1acdd41086a4eab94b7f4f87020250903135604.jpeg](https://assets.cdn.dicoding.com/original/academy/dos-d14042b1acdd41086a4eab94b7f4f87020250903135604.jpeg)

Ini secara efektif menunjukkan kata-kata mana–menurut pemahaman model–yang memiliki makna atau konteks penggunaan paling mirip dengan "matematika" berdasarkan _dataset_ pelatihan. 

Sekali lagi, dengan dataset yang kecil, hasil ini mungkin tidak selalu sempurna, tapi tetap memberikan ilustrasi jelas tentang cara model "mempelajari" hubungan semantik antar kata dan mengelompokkan kata-kata yang terkait secara konseptual.

Meskipun dataset yang kita gunakan sangat sederhana, contoh praktis ini secara efektif mendemonstrasikan prinsip dasar penggunaan **Word2Vec** dengan arsitektur **CBOW** dan **Skip-gram** untuk menghasilkan Word Embeddings. 

Dalam aplikasi NLP dunia nyata yang lebih kompleks, model-model ini dilatih pada korpus teks yang sangat besar—sering kali terdiri dari jutaan atau bahkan miliaran kata. Hal ini memungkinkan mereka untuk menghasilkan representasi kata yang jauh lebih kaya, akurat, dan bermakna, yang kemudian menjadi dasar fundamental untuk berbagai tugas NLP tingkat lanjut, seperti terjemahan mesin, analisis sentimen, ekstraksi informasi, atau bahkan sistem rekomendasi cerdas.

## Matematika dalam Natural Language Processing (NLP): Cosine Similarity

**Cosine similarity** adalah salah satu metode paling umum guna mengukur kesamaan antara dua vektor dalam ruang vektor berdimensi tinggi, yang terutama digunakan pada **natural language processing (NLP)** untuk mengukur kemiripan antar teks atau dokumen.

Ini adalah teknik yang sangat berguna ketika kita bekerja dengan teks yang telah dikonversi menjadi representasi vektor, seperti **Bag of Words (BoW), TF-IDF,** atau **Word Embedding**. Kita sudah bersinggungan dengan cosine similarity ini pada latihan implementasi Word Embedding dengan Python sebelumnya.

  

### Cara Kerja Cosine Similarity

Pada dasarnya, **cosine similarity** mengukur seberapa dekat dua vektor dengan menghitung sudut antara keduanya. Semakin kecil sudut antara dua vektor, semakin besar kemiripannya. Jika dua vektor sejajar (sudut 0 derajat), kesamaannya sangat tinggi, dan jika mereka tegak lurus (sudut 90 derajat), tidak ada kesamaan di antara keduanya.

![dos-fd8f8551f2e9b68dc31111c0910c606d20250903145614.jpeg](https://assets.cdn.dicoding.com/original/academy/dos-fd8f8551f2e9b68dc31111c0910c606d20250903145614.jpeg)

Dalam konteks **NLP**, **cosine similarity** sering digunakan untuk membandingkan dua dokumen atau dua kalimat dan menentukan sejauh mana mereka serupa. Misalnya, dalam sistem pencarian informasi, kita ingin mengetahui seberapa mirip query yang diberikan dengan dokumen pada database. Di sinilah **cosine similarity** berperan penting.

  

### Pentingnya Cosine Similarity

Sebelum memahami lebih dalam tentang **cosine similarity**, mari kita lihat alasan ini sangat penting, khususnya dalam **NLP**.

1.  **Pencocokan Dokumen atau Kalimat**  
    **Cosine similarity** digunakan untuk mengukur kemiripan dua teks. Misalnya, dalam aplikasi pencarian dokumen atau sistem rekomendasi, kita sering kali ingin mengetahui kemiripan dua teks berdasarkan kata-kata yang ada di dalamnya. **Cosine similarity** memungkinkan kita untuk melakukan ini dengan efektif.
2.  **Tidak Terpengaruh oleh Panjang Dokumen**  
    Salah satu keunggulan dari **cosine similarity** adalah bahwa ia tidak terpengaruh oleh panjang dokumen. Artinya, meskipun dua dokumen memiliki panjang yang sangat berbeda, **cosine similarity** hanya memperhitungkan **arah** vektor (yaitu hubungan antara kata-kata dalam dokumen), bukan panjang total dokumen.
3.  **Kemampuan Mengukur Keterkaitan Semantik**  
    **Cosine similarity** dapat digunakan untuk mengukur keterkaitan semantik antara dua kalimat atau dokumen. Misalnya, dua kalimat dengan banyak kata bersama-sama dalam konteks serupa akan memiliki nilai **cosine similarity** yang tinggi.

  

### Implementasi Cosine Similarity dengan Python

Dalam pembelajaran natural language processing (NLP), memahami cara mengukur kemiripan antar dokumen adalah fondasi penting. Kali ini, kita akan membandingkan dua kalimat sederhana dalam bahasa Indonesia menggunakan tiga pendekatan berbeda: **Bag of Words (BoW)**, **TF-IDF**, dan **Word Embedding dengan Word2Vec**.

#### Langkah 1: Impor Library

Kita mulai dengan mengimpor beberapa library yang dibutuhkan, termasuk `gensim` dan `scikit-learn`.

1.  import gensim
2.  from gensim.models import Word2Vec
3.  from sklearn.feature\_extraction.text import CountVectorizer, TfidfVectorizer
4.  from sklearn.metrics.pairwise import cosine\_similarity

#### Langkah 2: Menyiapkan Dokumen

Dua kalimat pendek akan digunakan sebagai data uji sebagai berikut.

1.  \# Dua dokumen yang akan dibandingkan
2.  doc1 \= "Saya suka belajar matematika"
3.  doc2 \= "Belajar matematika sangat menyenangkan"

#### Langkah 3: Mengukur Kemiripan dengan Bag of Words (BoW)

Pendekatan ini mengubah setiap dokumen menjadi vektor berdasarkan frekuensi kata.

1.  \# 1. \*\*Bag of Words (BoW)\*\*

3.  \# Menggunakan CountVectorizer dari Scikit-learn untuk Bag of Words
4.  count\_vectorizer \= CountVectorizer()
5.  bow\_matrix \= count\_vectorizer.fit\_transform(\[doc1, doc2\])

Kemudian, cosine similarity dihitung sebagai berikut.

1.  \# Menghitung cosine similarity untuk BoW
2.  bow\_similarity \= cosine\_similarity(bow\_matrix\[0:1\], bow\_matrix\[1:2\])
3.  print(f"Cosine Similarity (BoW): {bow\_similarity\[0\]\[0\]}")

**Catatan:** BoW tidak mempertimbangkan makna atau urutan kata—hanya menghitung keberadaan dan frekuensi.

#### Langkah 4: Mengukur Kemiripan dengan TF-IDF

TF-IDF (Term Frequency–Inverse Document Frequency) memberikan bobot lebih pada kata-kata penting yang tidak sering muncul dalam semua dokumen. Kita menghitung TF-IDF dengan kode berikut.

1.  \# 2. \*\*TF-IDF\*\*

3.  \# Menggunakan TfidfVectorizer dari Scikit-learn untuk TF-IDF
4.  tfidf\_vectorizer \= TfidfVectorizer()
5.  tfidf\_matrix \= tfidf\_vectorizer.fit\_transform(\[doc1, doc2\])

7.  \# Menghitung cosine similarity untuk TF-IDF
8.  tfidf\_similarity \= cosine\_similarity(tfidf\_matrix\[0:1\], tfidf\_matrix\[1:2\])
9.  print(f"Cosine Similarity (TF-IDF): {tfidf\_similarity\[0\]\[0\]}")

**Catatan:** TF-IDF mengurangi bobot kata yang terlalu umum (seperti "belajar") dan menonjolkan kata yang khas untuk satu dokumen.

#### Langkah 5: Persiapan Dataset untuk Word2Vec

Setelah menghitung BoW dan TF-IDF, sekarang kita akan melakukan representasi semantik menggunakan Word2Vec (CBOW dan Skip-gram).

1.  Model Word2Vec memerlukan input dalam format token seperti ini.
2.  \# 3. \*\*CBOW dan Skip-gram dengan Word2Vec\*\*

4.  \# Membuat dataset kalimat (dalam bentuk list of words)
5.  sentences \= \[
6.     doc1.split(),  \# Kata-kata dalam doc1
7.     doc2.split()   \# Kata-kata dalam doc2
8.  \]

#### Langkah 6: Melatih Model Word2Vec

Pertama, kita gunakan kode ini untuk melatih model dengan teknik CBOW.

1.  \# Melatih model Word2Vec dengan CBOW (sg=0) dan Skip-gram (sg=1)
2.  \# CBOW
3.  cbow\_model \= Word2Vec(sentences, vector\_size\=50, window\=3, min\_count\=1, sg\=0)

Lanjut, sekarang giliran kita melatih model dengan teknik Skip-gram dengan baris kode berikut.

1.  \# Skip-gram
2.  skipgram\_model \= Word2Vec(sentences, vector\_size\=50, window\=3, min\_count\=1, sg\=1)

Penjelasan parameternya berikut.

*   `vector_size=50`: Panjang vektor representasi kata.
*   `window=3`: Jangkauan konteks kata.
*   `min_count=1`: Kata dengan frekuensi 1 tetap disertakan.
*   `sg=0` untuk CBOW dan `sg=1` untuk Skip-gram.

#### Langkah 7: Mendapatkan Representasi Dokumen

Kita ambil **rata-rata vektor kata** untuk setiap dokumen.

1.  \# Menghitung rata-rata vektor kata untuk kedua dokumen dalam kedua model

3.  def get\_document\_vector(model, words):
4.     \# Ambil rata-rata vektor kata untuk dokumen
5.     word\_vectors \= \[model.wv\[word\] for word in words if word in model.wv\]
6.     if len(word\_vectors) \== 0:
7.         return \[0\] \* model.vector\_size
8.     return sum(word\_vectors) / len(word\_vectors)

Lalu, dihitung untuk masing-masing model.

1.  \# Menghitung vektor untuk dokumen 1 dan dokumen 2 dengan CBOW
2.  doc1\_vector\_cbow \= get\_document\_vector(cbow\_model, doc1.split())
3.  doc2\_vector\_cbow \= get\_document\_vector(cbow\_model, doc2.split())

5.  \# Menghitung vektor untuk dokumen 1 dan dokumen 2 dengan Skip-gram
6.  doc1\_vector\_skipgram \= get\_document\_vector(skipgram\_model, doc1.split())
7.  doc2\_vector\_skipgram \= get\_document\_vector(skipgram\_model, doc2.split())

#### Langkah 8: Menghitung Kemiripan dengan Word2Vec

Dengan vektor dokumen yang didapat, kita hitung cosine similarity.

1.  \# Menghitung cosine similarity antara dokumen 1 dan 2 menggunakan CBOW dan Skip-gram
2.  cosine\_cbow \= cosine\_similarity(\[doc1\_vector\_cbow\], \[doc2\_vector\_cbow\])
3.  cosine\_skipgram \= cosine\_similarity(\[doc1\_vector\_skipgram\], \[doc2\_vector\_skipgram\])

5.  print(f"Cosine Similarity (CBOW): {cosine\_cbow\[0\]\[0\]}")
6.  print(f"Cosine Similarity (Skip-gram): {cosine\_skipgram\[0\]\[0\]}")

  

### Penutup: Interpretasi Hasil

Untuk dua kalimat berikut

*   **doc1**: _"Saya suka belajar matematika"_
*   **doc2**: _"Belajar matematika sangat menyenangkan"_

Kita telah menghitung tingkat kemiripan antara keduanya menggunakan empat pendekatan representasi teks. Setelah kode di atas dijalankan, hasilnya berikut.

![dos-8ca0139f7c40e4efb29c2fc419b5b88c20250903145614.jpeg](https://assets.cdn.dicoding.com/original/academy/dos-8ca0139f7c40e4efb29c2fc419b5b88c20250903145614.jpeg)

Berikut detail penjelasannya.

*   **BoW (0.5)**  
    Pendekatan Bag of Words menghitung kemiripan berdasarkan jumlah kata yang sama antar dokumen, tanpa memperhatikan makna atau urutan. Nilai **0.5** menunjukkan bahwa setengah dari vektor representasi kedua dokumen ini identik, yang masuk akal karena keduanya memiliki beberapa kata yang sama, seperti “belajar” dan “matematika”.
*   **TF-IDF (0.34)**  
    TF-IDF mempertimbangkan seberapa penting suatu kata dalam konteks dokumen tertentu. Nilai kemiripan **lebih rendah** dibandingkan BoW karena kata-kata umum, seperti “belajar” dan “matematika”, mungkin dianggap tidak terlalu membedakan sehingga bobotnya dikurangi.
*   **Word2Vec (0.27)**  
    Word2Vec mencoba memahami arti kata dengan mempelajari konteksnya. Baik pendekatan **CBOW** maupun **Skip-gram** menghasilkan nilai cosine similarity yang **lebih rendah**(≈ 0.27) dibandingkan BoW dan TF-IDF. Hal ini wajar karena hal berikut.
    *   Dataset terlalu kecil untuk melatih model yang benar-benar mampu memahami hubungan semantik antar kata.
    *   Kata-kata seperti “suka” dan “menyenangkan” tidak muncul dalam konteks cukup beragam untuk membentuk representasi semantik yang kuat.

Meskipun nilai similarity rendah, pendekatan ini lebih menjanjikan jika diterapkan pada korpus yang lebih besar karena mempertimbangkan makna dan konteks.

**Kesimpulan**  
BoW cocok untuk kasus sederhana. TF-IDF lebih kontekstual. Word2Vec unggul pada pemahaman makna, tapi butuh data lebih besar.

Anda bisa akses kode lengkap yang dijelaskan dalam topik natural language processing pada link berikut: [Modul 6 - Natural Language Processing.ipynb](https://colab.research.google.com/drive/1G9enf1nz6BGNKmcyGsh8zybrPJISLte-?usp=sharing)

## Matematika dalam Time Series

Setelah kita mempelajari implementasi matematika pada bidang computer vision dan natural language processing, kini kita akan beralih ke salah satu jenis data yang sangat spesifik serta memiliki karakteristik unik: **data deret waktu (time series)**. 

Berbeda dengan dataset tradisional ketika setiap observasi sering diasumsikan independen, data time series memiliki ketergantungan waktu yang intrinsik, yakni sebuah nilai pada waktu tertentu sangat mungkin dipengaruhi oleh nilai-nilai pada masa lalu. 

Pemahaman mendalam tentang struktur statistik data ini adalah kunci untuk membangun model prediksi yang akurat dan melakukan inferensi yang valid. Dalam materi ini, kita akan fokus pada cara memahami data time series sebagai proses statistik dan kemudian memformulasikan model ARIMA yang canggih.

  

### Pengertian Data Time Series

Data time series atau deret waktu adalah kumpulan observasi yang diurutkan secara kronologis. Karakteristik utamanya adalah adanya dependensi antar observasi berdasarkan urutan waktu. 

Sebagai contoh, harga saham hari ini sangat dipengaruhi oleh harga kemarin, penjualan bulan ini mungkin mengikuti pola yang sama dengan bulan yang sama tahun lalu, dan sebagainya. 

Memahami sifat fundamental ini adalah langkah pertama menuju analisis yang efektif.

  

### Konsep Proses Stokastik

Dalam konteks statistika, sebuah data time series yang kita amati di dunia nyata hanyalah sebuah realisasi tunggal dari suatu entitas yang lebih besar dan abstrak, yaitu **proses stokastik**. 

Proses stokastik dapat dibayangkan sebagai sebuah 'mesin' teoretis yang mengikuti aturan acak yang menghasilkan nilai-nilai secara terus-menerus seiring berjalannya waktu.

Setiap nilai dalam time series yang kita amati (misalnya, harga saham harian selama setahun) adalah satu dari kemungkinan tak terhingga "jalur" atau "lintasan" yang bisa dihasilkan oleh proses stokastik yang mendasarinya. 

Kunci dari proses stokastik adalah bahwa nilai pada waktu tertentu (Yt) bukan hanya variabel acak, melainkan juga **dipengaruhi oleh nilai-nilai sebelumnya** (Yt−1,Yt−2, dst.). Inilah yang membedakannya secara mendasar dari data yang bersifat independen dan terdistribusi identik, seperti yang biasa diasumsikan dalam statistik dasar.

Konsep ini secara langsung berhubungan dengan gagasan **variabel acak bersama (**_**joint random variables**_**)** dan **distribusi bersama (**_**joint distribution**_**)**. Untuk time series, kita tidak hanya tertarik pada distribusi probabilitas suatu variabel acak Yt secara terisolasi, tetapi juga pada distribusi probabilitas gabungan dari seluruh urutan variabel acak: (Y1,Y2,…,Yt). Memahami bahwa probabilitas satu nilai bergantung pada nilai sebelumnya (melalui probabilitas kondisional) adalah inti dari pemodelan proses stokastik. 

Pemahaman time series sebagai realisasi dari proses stokastik ini sangat fundamental. Ini adalah landasan teoritis yang memungkinkan kita untuk melakukan hal berikut.

*   **Memodelkan Ketergantungan Waktu:** Mengidentifikasi dan menghitung bahwa nilai di masa lalu memengaruhi nilai di masa depan.
*   **Membuat Prediksi:** Memproyeksikan nilai-nilai di masa depan berdasarkan pola historis yang diidentifikasi dari proses stokastik yang mendasarinya.
*   **Menganalisis Sifat Data:** Menentukan bahwa sebuah deret waktu memiliki sifat-sifat tertentu seperti stasioneritas (akan dibahas selanjutnya) yang sangat krusial untuk pemilihan model.

## Matematika dalam Time Series: Komponen Statistik Time Series

Meskipun time series bisa tampak sangat kompleks dan tidak teratur pada pandangan pertama, sebagian besar time series yang kita temui di dunia nyata dapat secara efektif dipecah menjadi beberapa komponen statistik yang lebih sederhana dan dapat dipahami. 

Proses ini, yang kita kenal sebagai **dekomposisi time series**, adalah langkah analitis yang sangat penting. Tujuannya adalah mengidentifikasi dan memisahkan pola-pola yang mendasari data, memisahkan bagian yang relatif dapat diprediksi dari elemen-elemen bersifat acak atau tidak dapat dijelaskan.

Secara umum, sebuah time series Yt dapat dimodelkan sebagai kombinasi (baik secara penjumlahan untuk model aditif maupun perkalian untuk model multiplikatif) dari empat komponen utama.

*   _**Trend**_ **(T****t****):** Merepresentasikan perubahan jangka panjang dalam rata-rata time series yang bisa meningkat, menurun, atau tetap datar seiring waktu (misalnya, peningkatan penjualan _smartphone_ selama dekade terakhir).
*   _**Seasonality**_ **(S****t****):** Pola yang berulang secara periodik dan konsisten dalam jangka waktu tetap, disebabkan oleh faktor-faktor musiman atau kalender yang dapat diprediksi (misalnya, peningkatan penjualan es krim di musim panas setiap tahun).
*   _**Cyclical**_ **(C****t****):** Mencerminkan fluktuasi jangka panjang yang tidak memiliki periode tetap, sering kali terkait dengan siklus bisnis atau ekonomi yang durasinya bervariasi selama beberapa tahun (misalnya, siklus _boom_ dan _bust_ ekonomi).
*   _**Residual**_**/**_**Noise**_ **(R****t** **atau ε****t****):** Ini adalah komponen yang tersisa setelah trend, seasonality, dan cyclical pattern dihilangkan. Residual ini merepresentasikan variasi yang tidak dapat dijelaskan oleh pola-pola tersebut, dan idealnya, ia bersifat acak murni atau yang kita sebut sebagai _white noise_.

![dos-35b6d76394056b169c5e22875ed609e320250903151817.jpeg](https://assets.cdn.dicoding.com/original/academy/dos-35b6d76394056b169c5e22875ed609e320250903151817.jpeg)

Dari sudut pandang statistika, komponen-komponen ini memiliki definisi yang presisi. 

*   Trend mencerminkan perubahan dalam nilai **mean** time series seiring dengan berjalannya waktu. 
*   Seasonality mengindikasikan adanya **periodisitas** atau pola berulang yang konsisten dalam mean atau variansi deret waktu pada interval waktu yang tetap. 
*   Sementara itu, **white noise** (Rt) adalah target ideal kita untuk komponen residual. Kumpulan variabel acak yang membentuk _white noise_ diasumsikan bersifat **independen dan terdistribusi identik** dengan rata-rata nol serta variansi yang konstan. Jika residual dari model kita menyerupai _white noise_, itu adalah indikasi yang sangat baik bahwa model telah berhasil menangkap semua pola secara sistematis dan terprediksi dalam data.

Penerapan dari konsep dekomposisi time series ini sangat praktis dan penting. Dengan memisahkan time series dalam bagian-bagian komponennya, kita memperoleh kemampuan untuk melakukan hal berikut.

*   **Memahami Struktur Data:** Mendapatkan wawasan jelas tentang pola-pola yang mendasari data, seperti adanya pertumbuhan jangka panjang yang dominan atau pola musiman yang kuat.
*   **Melakukan Prediksi yang Lebih Baik:** Setelah mengidentifikasi trend dan seasonality, kita dapat memproyeksikan komponen-komponen yang dapat diprediksi ini ke masa depan serta kemudian menggunakan model statistik lebih canggih untuk memodelkan residual.
*   **Mendeteksi Anomali/Outlier:** Nilai-nilai residual yang sangat besar (positif atau negatif) setelah pola trend dan seasonality dihilangkan dapat mengindikasikan anomali atau outlier dalam data.
*   **Menginformasikan Keputusan Bisnis:** Misalnya, memahami trend penjualan membantu perencanaan strategis jangka panjang, sementara memahami pola musiman penjualan sangat vital untuk manajemen inventaris dan alokasi staf. 

Pemahaman time series sebagai realisasi dari proses stokastik dengan komponen-komponen ini adalah fondasi kokoh untuk membangun model peramalan seperti ARIMA, yang akan kita bahas nanti dalam modul ini.

## Matematika dalam Time Series: Stasioneritas: Asumsi Kritis untuk Inferensi

Setelah memahami bahwa time series adalah realisasi dari proses stokastik dengan komponen, seperti trend dan seasonality, kini kita akan membahas sebuah konsep yang sangat fundamental serta krusial dalam pemodelan deret waktu: **stasioneritas**. 

Mengapa stasioneritas begitu penting? Pasalnya, sebagian besar model time series yang kuat, termasuk ARIMA, mengasumsikan bahwa data yang dianalisis sudah bersifat stasioner. Jika data tidak stasioner, prediksi yang dihasilkan model bisa menjadi tidak akurat atau bahkan tidak valid.

  

### Definisi Stasioneritas

Saat bekerja dengan data time series, salah satu langkah penting yang sering kali menjadi syarat utama sebelum membangun model adalah **membuat data menjadi stasioner**. Namun, apa sebenarnya maksud dari “stasioner”? Mengapa hal ini sangat penting?

Dalam konteks statistik, **data stasioner** berarti data tersebut **stabil secara statistik sepanjang waktu**. Artinya, rata-rata (mean), sebaran data (variance), dan hubungan antar nilai-nilai pada waktu yang berbeda (autokorelasi) tidak berubah seiring waktu.

Untuk membayangkannya lebih mudah, coba pikirkan data yang “bergetar” di sekitar nilai rata-rata tertentu, tidak naik terus, tidak juga turun drastis. Itulah karakteristik utama data stasioner. Dengan kata lain, meskipun ada fluktuasi, itu **tidak membentuk tren jangka panjang** atau **pola musiman yang kuat**.

Sebaliknya, jika data terus naik (misalnya karena pertumbuhan bisnis) atau terus menurun (seperti popularitas suatu produk yang menurun), data tersebut **tidak stasioner** karena mean-nya berubah seiring waktu.

  

### Kenapa Stasioneritas Itu Penting?

Model time series seperti **ARIMA** bekerja dengan asumsi bahwa pola-pola dalam data tidak berubah secara drastis dari waktu ke waktu. Jika data tidak stasioner, model akan kesulitan “belajar” dari masa lalu untuk memprediksi masa depan. Ini karena pola yang ditemukan di awal mungkin tidak berlaku lagi beberapa waktu kemudian.

Bayangkan Anda ingin memprediksi penjualan bulan depan dengan melihat data dua tahun terakhir. Jika tren penjualannya naik tajam setiap bulan, model akan kesulitan menemukan pola yang stabil. Namun, jika tren tersebut dihilangkan terlebih dahulu, fluktuasi yang tersisa bisa dimodelkan dengan lebih baik.

Singkatnya, **membuat data stasioner membantu model fokus pada pola yang bisa dipelajari**, bukan terganggu oleh tren besar atau pola musiman yang bisa menyesatkan prediksi.

  

### Cara Membuat Data Stasioner: Differencing

Salah satu cara paling sederhana dan efektif untuk membuat data stasioner adalah dengan menggunakan teknik yang disebut _**differencing**_. Teknik ini dilakukan dengan **menghitung selisih antara satu nilai dengan nilai sebelumnya**.

Penjualan Hari Ini    = 130 unit  

Penjualan Kemarin  = 120 unit  

Hasil Differencing    = 130 - 120 = 10

Dengan melakukan ini untuk seluruh time series, kita akan mendapatkan data baru yang disebut sebagai **first difference**. Teknik ini akan menghilangkan tren linier dari data. Jika tren masih kuat, kita bisa melakukan differencing lagi—ini disebut **second difference**.

Namun, tidak perlu langsung melakukan differencing dua kali. Umumnya, satu kali sudah cukup untuk sebagian besar kasus.

  

### Visualisasi: Cek dengan Plot

Setelah melakukan differencing, penting untuk memeriksa jika data sudah terlihat stasioner. Ini bisa dilakukan secara visual.

1.  **Plot data asli**: Biasanya terlihat naik atau turun secara umum.
2.  **Plot hasil differencing**: Harusnya garis berfluktuasi di sekitar nol, tanpa pola naik/turun jangka panjang.

Berikut adalah contoh kode sederhana untuk menunjukkan metode differencing dalam bahasa pemrograman Python.

1.  \# Re-import all necessary libraries due to code execution state reset
2.  import pandas as pd
3.  import numpy as np
4.  import matplotlib.pyplot as plt

6.  \# Langkah 1.2: Buat data deret waktu sintetis
7.  np.random.seed(42)
8.  dates \= pd.date\_range(start\='2020-01-01', periods\=100, freq\='MS')
9.  base\_sales \= np.linspace(100, 200, 100)
10.  seasonality \= 10 \* np.sin(np.linspace(0, 3 \* np.pi, 100))
11.  noise \= np.random.normal(0, 10, 100)
12.  sales\_data \= base\_sales + seasonality + noise
13.  df \= pd.DataFrame({'Sales': sales\_data}, index\=dates)

15.  \# Hitung differencing
16.  df\['Differencing'\] \= df\['Sales'\].diff()

18.  \# Plot data asli dan setelah differencing
19.  plt.figure(figsize\=(10, 5))

21.  \# Plot data asli
22.  plt.subplot(2, 1, 1)
23.  plt.plot(df\['Sales'\], label\='Data Asli', color\='blue')
24.  plt.title('Data Penjualan Asli')
25.  plt.legend()

27.  \# Plot differencing
28.  plt.subplot(2, 1, 2)
29.  plt.plot(df\['Differencing'\], label\='Differencing', color\='orange')
30.  plt.title('Setelah Differencing')
31.  plt.legend()

33.  plt.tight\_layout()
34.  plt.show()

Setelah kodenya dijalankan, ini hasilnya.

![dos-cc9f83b4b5322839aafd585181f6299720250903151958.jpeg](https://assets.cdn.dicoding.com/original/academy/dos-cc9f83b4b5322839aafd585181f6299720250903151958.jpeg)

Dengan plot ini, Anda bisa langsung melihat ketika data setelah dilakukan differencing sudah terlihat stabil dan fluktuatif di sekitar garis nol. Grafik yang awalnya bergerak secara fluktuatif akan menjadi stasioner.

Tujuannya adalah **mencapai stasioneritas, bukan menghilangkan semua variasi**. Jika setelah satu kali differencing data sudah tampak fluktuatif secara acak, itu sudah cukup. Terlalu banyak differencing bisa menghilangkan informasi penting dalam data.

Untuk memastikan lebih lanjut, Anda bisa menggunakan **uji statistik, seperti Augmented Dickey–Fuller (ADF test)**, tapi itu akan dibahas lebih detail dalam bagian selanjutnya.

Membuat data menjadi stasioner adalah langkah awal yang sangat penting dalam analisis time series. Dengan teknik sederhana seperti differencing, Anda bisa menghilangkan tren dan menjadikan data lebih “siap” untuk dimodelkan.

## Matematika dalam Time Series: Mengenal ACF & PACF

Setelah kita berhasil membuat data time series menjadi **stasioner**, langkah penting berikutnya adalah memahami bahwa data tersebut saling terkait dari waktu ke waktu. Dalam dunia analisis time series, ada dua alat bantu utama yang sangat populer untuk menganalisis hubungan ini, yaitu **ACF** _**(autocorrelation function)**_ dan **PACF** _**(partial autocorrelation function)**._

![dos-10215071629713684dbf64b1d8b7f05320250903154231.jpeg](https://assets.cdn.dicoding.com/original/academy/dos-10215071629713684dbf64b1d8b7f05320250903154231.jpeg)

Kedua fungsi ini sangat berguna karena memberikan wawasan tentang **pola keterkaitan antar waktu**, yang akan membantu kita dalam **menentukan parameter** terbaik saat membangun model ARIMA.

  

### Apa Itu Autocorrelation Function (ACF)?

**Autocorrelation function (ACF)** mengukur seberapa mirip nilai data saat ini dengan nilai-nilai pada masa lalu. Dalam konteks time series, kita menyebut tingkat keterlambatan sebagai **lag**. Misalnya berikut.

*   **Lag-1** berarti membandingkan nilai hari ini dengan nilai kemarin,
*   **Lag-2** berarti membandingkan hari ini dengan dua hari yang lalu,
*   dan seterusnya.

ACF akan menghitung **korelasi antara setiap nilai dengan versi tertundanya sendiri** untuk setiap lag yang ditentukan. Korelasi ini digambarkan dalam bentuk **plot batang**, yakni setiap batang mewakili tingkat korelasi untuk satu lag.

Contoh pertanyaan yang bisa dijawab oleh ACF sebagai berikut.

*   Apakah penjualan hari ini dipengaruhi oleh penjualan kemarin?
*   Apakah ada pola mingguan, misalnya penjualan hari ini mirip dengan penjualan seminggu yang lalu.

Jika batang pada lag tertentu sangat tinggi dan melewati batas signifikan, itu artinya ada hubungan statistik yang kuat antara nilai sekarang dengan nilai dalam lag tersebut.

  

### Apa Itu Partial Autocorrelation Function (PACF)?

Apabila ACF menghitung hubungan total antara nilai saat ini dan nilai sebelumnya, PACF berfokus pada hubungan langsung.

PACF mencoba menjawab pertanyaan berikut.

_“Apakah nilai pada lag ke-3 memengaruhi nilai hari ini setelah mempertimbangkan efek dari lag ke-1 dan lag ke-2?”_

Dengan kata lain, PACF membantu kita menyaring hubungan yang benar-benar relevan secara langsung, tanpa “gangguan” dari lag-lag sebelumnya. Ini sangat penting karena dalam dunia nyata banyak hubungan terlihat signifikan, padahal sebenarnya hanya “diturunkan” dari hubungan yang lebih pendek.

Layaknya ACF, hasil PACF juga divisualisasikan dalam bentuk plot batang, dengan garis batas untuk menandai korelasi parsial tersebut signifikan atau tidak.

  

### Hubungan ACF dan PACF dengan Model ARIMA

Kedua fungsi ini (ACF dan PACF) memiliki peran sangat penting dalam membentuk model ARIMA (AutoRegressive Integrated Moving Average) karena mereka membantu kita menentukan parameter berikut.

*   p (jumlah lag untuk komponen AR/AutoRegressive)
*   q (jumlah lag untuk komponen MA/Moving Average)

Mari kita bahas kaitannya lebih lanjut.

## Matematika dalam Time Series: Komponen Model ARIMA

Setelah memahami bahwa fungsi ACF dan PACF membantu kita mengidentifikasi pola keterkaitan dalam data time series, kini saatnya kita menggunakannya untuk membangun model prediksi yang konkret. Di sinilah model ARIMA hadir sebagai salah satu alat paling populer dalam analisis deret waktu.

Model ARIMA (AutoRegressive Integrated Moving Average) merupakan gabungan dari tiga komponen matematis yang masing-masing merepresentasikan jenis pola berbeda dalam data time series: pola ketergantungan terhadap nilai masa lalu, kebutuhan akan stasioneritas, dan pengaruh kesalahan prediksi sebelumnya.

Untuk benar-benar memahami logika di balik ARIMA, kita akan membedah masing-masing komponennya secara terpisah: dimulai dari model AR (AutoRegressive), dilanjutkan ke model MA (Moving Average), dan akhirnya menggabungkannya dalam kerangka ARIMA yang utuh.

Mari kita mulai dengan komponen pertama: Model AR (AutoRegressive).

  

### Model AR (AutoRegressive)

Model **AR** atau **AutoRegressive** adalah salah satu komponen utama dalam model ARIMA. Istilah "AutoRegressive" sendiri bisa diartikan sebagai **regresi terhadap dirinya sendiri**. Artinya, **nilai saat ini diprediksi berdasarkan nilai-nilai masa lalu dari variabel itu sendiri**.

Bayangkan Anda ingin memprediksi **jumlah pengunjung toko hari ini**. Dalam model AR, Anda tidak memerlukan data lain—cukup melihat **berapa jumlah pengunjung kemarin, dua hari lalu, tiga hari lalu**, dan seterusnya. Nilai-nilai tersebut akan digunakan untuk memperkirakan nilai hari ini.

Model AR menyatakan hal berikut.

*   Semakin besar nilai **p**, semakin banyak nilai masa lalu (lag) yang digunakan.
*   Nilai saat ini adalah kombinasi linear dari nilai masa lalu.

#### Rumus Model AR

Dalam memodelkan data time series menggunakan pendekatan AutoRegressive (AR), kita akan sering menjumpai dua bentuk rumus yang tampak mirip, tapi memiliki perbedaan penting. Keduanya merepresentasikan gagasan inti yang sama bahwa **nilai saat ini dapat diprediksi dari nilai-nilai masa lalu**, tetapi memiliki implikasi berbeda dalam praktik.

**Rumus 1: AR Tanpa Konstanta**

![dos-6a773350f8f24b1182340273533e1d3620250903155316.jpeg](https://assets.cdn.dicoding.com/original/academy/dos-6a773350f8f24b1182340273533e1d3620250903155316.jpeg)

Rumus ini menunjukkan bahwa nilai saat ini (Xt) merupakan kombinasi linear dari nilai-nilai sebelumnya ditambah gangguan acak εt. Bentuk ini **tidak memiliki konstanta** dan mengasumsikan bahwa data telah distandarkan atau telah melalui proses _differencing_ sehingga rata-ratanya (mean) mendekati nol.

Model ini sangat cocok digunakan saat kondisi berikut.

*   Data telah distasionerkan dengan differencing.
*   Kita hanya ingin memodelkan **fluktuasi atau deviasi terhadap nol**, bukan level absolut dari datanya.

**Rumus 2: AR dengan Konstanta**

![dos-d59461bcbf35f861b2587572eb683e8620250903155316.jpeg](https://assets.cdn.dicoding.com/original/academy/dos-d59461bcbf35f861b2587572eb683e8620250903155316.jpeg)

Versi ini menambahkan sebuah **konstanta** c atau sering disebut _intercept_. Komponen ini berguna untuk **menangkap rata-rata data yang tidak nol**, atau dengan kata lain, untuk memodelkan **level dasar** dari deret waktu.

Model ini lebih tepat digunakan saat kondisi berikut.

*   Data **belum di-differencing** dan masih memiliki tren atau rata-rata tetap.
*   Kita ingin memprediksi nilai absolut dari data (bukan hanya perubahannya).
*   Model harus menangkap kecenderungan nilai mendekati titik keseimbangan yang bukan nol.

#### Menentukan Nilai _p_ dengan PACF

Untuk menentukan jumlah nilai masa lalu (p) yang relevan, kita menggunakan PACF (partial autocorrelation function).

Ciri khas dari model AR adalah bahwa plot **PACF akan terputus (**_**cut-off**_**) setelah lag ke-p**. Artinya, batang-batang signifikan hanya muncul hingga lag ke-p. Setelah itu, batang akan kecil dan berada dalam batas kepercayaan (tidak signifikan lagi).

![dos-366df85e1a19d6380584328a66da497520250903155316.jpeg](https://assets.cdn.dicoding.com/original/academy/dos-366df85e1a19d6380584328a66da497520250903155316.jpeg)

Gambar di atas adalah plot **PACF (partial autocorrelation function)** yang menunjukkan cara menentukan nilai _p_ (order AR) dalam model AR.

*   Terlihat bahwa **batang signifikan hanya muncul pada lag ke-1**, lalu nilai PACF turun dan berada dalam batas kepercayaan (daerah oranye).
*   Ini menandakan bahwa **model AR(1)** adalah pilihan yang tepat untuk data ini.

Plot seperti ini berguna untuk mengidentifikasi jumlah lag (_p_) yang benar-benar relevan secara statistik dalam model ARIMA.

  

### Model MA (Moving Average)

Model **MA** atau **Moving Average** adalah komponen kedua dalam ARIMA. Berbeda dengan model AR yang menggunakan **nilai data masa lalu**, model MA menggunakan **kesalahan prediksi (error) dari masa lalu** sebagai dasar untuk memprediksi nilai sekarang.

Misalkan kemarin Anda memprediksi penjualan sebesar 100 unit, tetapi realisasinya ternyata 120 unit. Artinya Anda melakukan kesalahan prediksi (error) sebesar 20+. Nah, model MA beranggapan bahwa kesalahan ini turut memengaruhi **nilai pada hari ini**. 

Dengan kata lain,

“Hari ini bisa dipengaruhi oleh seberapa besar saya meleset kemarin.”

Model MA tidak menggunakan nilai observasi masa lalu secara langsung, tetapi menggunakan besarnya error (residual) dari prediksi sebelumnya. Model ini cocok saat fluktuasi atau noise masa lalu dianggap memengaruhi nilai saat ini.

Secara matematis, model MA dengan q lag dituliskan sebagai berikut.

![dos-61929efd4bf17655247a53a98db3fc9520250903155315.jpeg](https://assets.cdn.dicoding.com/original/academy/dos-61929efd4bf17655247a53a98db3fc9520250903155315.jpeg)

Untuk menentukan berapa banyak lag error (q) yang relevan (q), kita menggunakan **ACF (autocorrelation function)**.

Ciri khas dari model MA adalah bahwa **ACF akan terputus (**_**cut-off**_**) setelah lag ke-q**. Artinya, batang-batang yang signifikan hanya muncul hingga lag ke-q dan setelah itu menghilang.

![dos-123b42e5785ede00a7b272ff3c8be16d20250903155316.jpeg](https://assets.cdn.dicoding.com/original/academy/dos-123b42e5785ede00a7b272ff3c8be16d20250903155316.jpeg)

Gambar di atas adalah plot **ACF (autocorrelation function)** untuk data yang dihasilkan dari proses **MA(1)**.

*   Terlihat bahwa batang autokorelasi **signifikan hanya pada lag ke-1**.
*   Setelah lag ke-1, batang-batang berada dalam batas kepercayaan (daerah oranye) dan tidak signifikan secara statistik.

  

### Model ARIMA (Gabungan AR, I, MA)

Setelah memahami dasar-dasar, seperti ACF, PACF, serta konsep stasioneritas dan differencing, kini saatnya kita merangkai semuanya dalam sebuah model prediksi time series yang sangat populer, yaitu **ARIMA**.

ARIMA adalah singkatan dari **AutoRegressive Integrated Moving Average**. Meskipun namanya terdengar rumit, sebenarnya logika di balik model ini cukup mudah untuk dipahami.

Secara sederhana, ARIMA adalah model statistik yang digunakan untuk **meramalkan nilai masa depan** dari sebuah time series berdasarkan **nilai masa lalu** dan **pola kesalahan (error) yang terjadi sebelumnya**.

Model ini menggabungkan tiga komponen utama yang sudah kita pelajari sebelumnya, antara lain berikut.

1.  **AR (AutoRegressive)**: bagian ini menggunakan **nilai-nilai masa lalu** (_lag_) untuk memprediksi nilai saat ini.
2.  **I (Integrated)**: bagian ini merujuk pada **differencing**, yaitu proses mengubah data agar menjadi stasioner (stabil/tidak dipengaruhi tren).
3.  **MA (Moving Average)**: bagian ini menggunakan **kesalahan prediksi dari masa lalu** (_error_) untuk memperbaiki prediksi saat ini.

Ketiga komponen tersebut ditulis dalam bentuk model ARIMA dengan notasi berikut.

![dos-fedaebf2138093a905802c2acd27a1d620250903155315.jpeg](https://assets.cdn.dicoding.com/original/academy/dos-fedaebf2138093a905802c2acd27a1d620250903155315.jpeg)

Mari kita bahas ketiga parameter ARIMA satu per satu agar lebih jelas.

*   **p** (AutoRegressive Order) adalah jumlah lag yang digunakan dalam komponen AR. Jika p = 2, model menggunakan Yt-1 dan Yt-2 untuk memprediksi Yt. Parameter ini biasanya ditentukan dengan melihat **plot PACF**.
*   **d** (Differencing Order) menunjukkan **jumlah data perlu di-differencing** agar menjadi stasioner. Jika data menunjukkan tren naik atau turun, kita melakukan differencing untuk menghilangkan tren tersebut. Biasanya, d = 1 sudah cukup untuk membuat data menjadi stasioner. Nilai ini ditentukan dengan melihat **plot data** dan hasil dari **ADF Test**.
*   **q** (Moving Average Order) adalah jumlah lag dari kesalahan masa lalu dalam komponen MA. Contoh: Jika q = 1, model menggunakan error pada waktu t−1 untuk memperbaiki prediksi Yt. Nilai q ditentukan dengan melihat **plot ACF**.

## Matematika dalam Time Series: Membangun Model ARIMA

Membangun model ARIMA sebenarnya tidak serumit yang dibayangkan, asalkan kita mengikuti langkah-langkahnya secara sistematis. Model ini sangat berguna untuk memprediksi data time series yang memiliki pola tertentu di masa lalu.

Berikut adalah **panduan praktis enam langkah** dalam membangun model ARIMA yang dapat Anda ikuti, dari awal hingga tahap prediksi. Sebelum memulai, pastikan kita sudah memiliki sebuah data time series terlebih dahulu. Jika kita belum memiliki, jalan kode di bawah ini untuk membuat sebuah data time series yang acak.

  

### Langkah 1: Mengimpor Library dan Menyiapkan Data Time Series 

Dimulai dengan mengimpor library penting untuk analisis time series, seperti `pandas`, `numpy`, `matplotlib`, `seaborn`, serta modul dari `statsmodels` untuk uji stasioneritas dan pemodelan ARIMA.

Kemudian, kita membuat dataset sintetis penjualan mingguan selama 100 minggu dengan pola tren naik dan noise acak, lalu menyimpannya pada file CSV.

Selanjutnya, data penjualan disimulasikan menggunakan distribusi normal yang diakumulasi agar membentuk tren, kemudian dikemas dalam `DataFrame`, dibulatkan, dan disimpan sebagai file bernama `data_penjualan.csv` untuk analisis lanjutan.

1.  import pandas as pd
2.  import numpy as np
3.  import matplotlib.pyplot as plt
4.  import seaborn as sns
5.  from statsmodels.tsa.stattools import adfuller \# Untuk Uji ADF (stationarity test)
6.  from statsmodels.graphics.tsaplots import plot\_acf, plot\_pacf \# Untuk ACF/PACF plots
7.  from statsmodels.tsa.arima.model import ARIMA \# Model ARIMA

9.  \# Set random seed agar hasil tetap konsisten
10.  np.random.seed(42)

12.  \# Generate tren penjualan mingguan (misalnya dari 100 naik perlahan)
13.  minggu \= np.arange(1, 101)
14.  penjualan \= 100 + np.cumsum(np.random.normal(loc\=1.5, scale\=5, size\=100))  \# tren naik + noise

16.  \# Buat DataFrame
17.  df \= pd.DataFrame({
18.      'minggu': minggu,
19.      'penjualan': np.round(penjualan, 2)  \# dibulatkan 2 angka desimal
20.  })

22.  \# Simpan ke file CSV
23.  df.to\_csv('data\_penjualan.csv', index\=False)

25.  print("Data berhasil dibuat dan disimpan sebagai data\_penjualan.csv")

### Langkah 2: Ubah Data Menjadi Stasioner

Langkah berikutnya sebelum menerapkan ARIMA adalah memastikan bahwa data kita **sudah stasioner**. Sebagaimana yang sudah dibahas sebelumnya, data stasioner adalah data dengan rata-rata dan variansi yang konstan dari waktu ke waktu.

Bagaimana cara memeriksanya?

*   **Visual**: Plot data time series. Jika data menunjukkan tren naik/turun atau memiliki pola musiman yang jelas, kemungkinan besar data tersebut belum stasioner.
*   **Statistik**: Gunakan **augmented dickey-fuller (ADF) Test** untuk menguji stasioneritas secara statistik. Jika nilai p-value lebih kecil dari 0.05, data dapat dianggap stasioner.

Jika data belum stasioner, kita bisa melakukan **differencing**, yaitu menghitung selisih antara nilai saat ini dan nilai sebelumnya (misalnya: hari ini – kemarin). Jika sekali differencing belum cukup, kita bisa melakukannya dua kali. Nilai _d_ dalam ARIMA mengacu pada berapa kali differencing ini dilakukan. 

Untuk melakukan tahap ini, silakan lihat pada kode di bawah ini. 

1.  \# Contoh data: penjualan mingguan
2.  df \= pd.read\_csv('data\_penjualan.csv')  \# misalnya memiliki kolom 'penjualan'

4.  \# Hitung differencing
5.  df\['diff'\] \= df\['penjualan'\].diff()

7.  \# Plot sebelum dan sesudah differencing
8.  plt.figure(figsize\=(10, 5))
9.  plt.subplot(2, 1, 1)
10.  plt.plot(df\['penjualan'\], label\='Data Asli')
11.  plt.title('Data Penjualan Asli')
12.  plt.legend()

14.  plt.subplot(2, 1, 2)
15.  plt.plot(df\['diff'\], label\='Differencing', color\='orange')
16.  plt.title('Setelah Differencing')
17.  plt.legend()

19.  plt.tight\_layout()
20.  plt.show()

Kode di atas akan menampilkan dua buah grafik, yang pertama adalah visualisasi data penjualan asli, sedangkan yang kedua adalah data yang sudah dilakukan _differencing_.

![dos-f5c6b090906c733e92f57f044183418920250903160130.jpeg](https://assets.cdn.dicoding.com/original/academy/dos-f5c6b090906c733e92f57f044183418920250903160130.jpeg)

  

### Langkah 3: Analisis ACF dan PACF

Setelah data stasioner, kita perlu mengetahui **berapa banyak lag dari nilai masa lalu** dan **berapa banyak lag dari error masa lalu** yang akan digunakan dalam model.

Di sinilah **plot ACF (autocorrelation function)** dan **PACF (partial autocorrelation function)** berperan penting.

*   **ACF** membantu Anda menentukan nilai **q**, yaitu jumlah lag dari _error_ (kesalahan prediksi masa lalu) yang relevan. Jika ACF menunjukkan lonjakan besar pada lag 1 dan 2, q bisa jadi 2.
*   **PACF** membantu Anda menentukan nilai **p**, yaitu jumlah lag dari nilai observasi masa lalu yang langsung berpengaruh. Jika PACF signifikan sampai lag 1, p = 1.

Untuk melakukan tahap ini, silakan jalankan kode di bawah ini. 

1.  \# Misal data sudah di-differencing: df\['diff'\]
2.  plt.figure(figsize\=(12, 5))

4.  \# ACF plot
5.  plt.subplot(1, 2, 1)
6.  plot\_acf(df\['diff'\].dropna(), lags\=20, ax\=plt.gca())
7.  plt.title('ACF - Autokorelasi')

9.  \# PACF plot
10.  plt.subplot(1, 2, 2)
11.  plot\_pacf(df\['diff'\].dropna(), lags\=20, ax\=plt.gca(), method\='ywm')
12.  plt.title('PACF - Autokorelasi Parsial')

14.  plt.tight\_layout()
15.  plt.show()

Kode di atas digunakan untuk menampilkan visualisasi ACF dan PACF, yakni dari data tersebut kita dapat menentukan nilai p dan q yang akan digunakan pada pemodelan ARIMA nantinya. Inilah visualisasinya.

![dos-84035198659b669950af49f32b30080d20250903160130.jpeg](https://assets.cdn.dicoding.com/original/academy/dos-84035198659b669950af49f32b30080d20250903160130.jpeg)

Perhatikan grafik ACF dan PACF di atas. Batang yang **melewati batas kepercayaan** dianggap signifikan secara statistik dan menunjukkan adanya hubungan autokorelasi pada lag tersebut.  

  

### Langkah 4: Membangun Model ARIMA

Setelah menentukan nilai p, d, dan q, sekarang saatnya kita membangun model ARIMA.

1.  model \= ARIMA(df\['penjualan'\], order\=(1, 1, 1))
2.  model\_fit \= model.fit()
3.  print(model\_fit.summary())

Pada kode di atas, kita membangun model ARIMA dengan parameter `(p=1, d=1, q=1)`. Baris `model = ARIMA(...)` mendefinisikan model berdasarkan data penjualan, `model.fit()` melatih model tersebut, dan `model_fit.summary()` menampilkan ringkasan hasil estimasi, termasuk koefisien, statistik signifikan, serta informasi diagnostik lainnya. 

![dos-00159598915ebacf8f21c3b5277e2dfa20250903160130.jpeg](https://assets.cdn.dicoding.com/original/academy/dos-00159598915ebacf8f21c3b5277e2dfa20250903160130.jpeg)

Ini adalah langkah inti untuk melakukan prediksi time series.

  

### Langkah 5: Evaluasi Model

Setelah model dilatih, kita perlu memastikan bahwa model tersebut sudah cukup baik untuk digunakan. Ada dua hal utama yang perlu diperhatikan: **kualitas prediksi** dan **karakteristik residual**.

Residual adalah selisih antara nilai yang diprediksi dan nilai aktual. Idealnya, residual harus **acak**, tanpa pola yang terlihat, dan menyerupai **white noise**. Jika residual masih memiliki pola (misalnya tren), berarti model belum cukup baik.

**Uji Statistik**

*   Gunakan **AIC dan BIC** untuk membandingkan model. Semakin rendah nilainya, semakin baik model (dengan asumsi kompleksitas wajar).
*   Gunakan **uji Ljung-Box** untuk memeriksa jika residual masih mengandung autokorelasi. Jika tidak, model dianggap telah menangkap pola utama dalam data.

Jangan lupa juga untuk **memvisualisasikan hasil prediksi** dan residual agar Anda bisa melihat jika prediksi model cukup akurat serta adakah penyimpangan besar.

Silakan jalankan kode berikut.

1.  \# --- Prediksi ---
2.  df\['forecast'\] \= model\_fit.predict(start\=1, end\=len(df)-1, typ\='levels')  \# karena d=1, mulai dari indeks 1

4.  \# --- Visualisasi: Aktual vs Prediksi ---
5.  plt.figure(figsize\=(10, 4))
6.  plt.plot(df\['penjualan'\], label\='Aktual')
7.  plt.plot(df\['forecast'\], label\='Prediksi', linestyle\='--')
8.  plt.title('Aktual vs Prediksi')
9.  plt.legend()
10.  plt.tight\_layout()
11.  plt.show()

13.  \# --- Analisis Residual ---
14.  residuals \= model\_fit.resid

16.  plt.figure(figsize\=(12, 4))
17.  plt.subplot(1, 2, 1)
18.  sns.histplot(residuals, kde\=True, bins\=20)
19.  plt.title('Distribusi Residual')

21.  plt.subplot(1, 2, 2)
22.  plt.plot(residuals)
23.  plt.title('Plot Residual')
24.  plt.tight\_layout()
25.  plt.show()

27.  \# --- Uji Ljung-Box ---
28.  ljung\_result \= acorr\_ljungbox(residuals, lags\=\[10\], return\_df\=True)
29.  print("\\nHasil Uji Ljung-Box (lag=10):")
30.  print(ljung\_result)

32.  \# --- Evaluasi AIC dan BIC ---
33.  print(f"\\nAIC: {model\_fit.aic}")
34.  print(f"BIC: {model\_fit.bic}")

Setelah kode dijalankan, ia akan menampilkan visualisasi berikut.

![dos-405939acbb75c1e491e74b4a1c7c0e7620250903160131.jpeg](https://assets.cdn.dicoding.com/original/academy/dos-405939acbb75c1e491e74b4a1c7c0e7620250903160131.jpeg)

Model ARIMA yang telah dibangun menunjukkan performa cukup baik berdasarkan hasil evaluasi. Penjelasannya berikut.

*   **Plot aktual vs prediksi**: memastikan bahwa model mengikuti pola data asli. Dalam kasus ini, plot prediksi berhasil mengikuti pola data aktual secara umum, menandakan model mampu menangkap tren yang ada.
*   **Plot dan histogram residual**: mengecek residual acak (seperti white noise). Residual terlihat tersebar secara acak dan membentuk distribusi mendekati normal yang menunjukkan bahwa kesalahan prediksi tidak bersifat sistematis.
*   **Uji Ljung-Box**: menguji jika residual masih memiliki autokorelasi. Hasil uji Ljung-Box menghasilkan p-value di atas 0.05, menandakan bahwa residual tidak lagi mengandung autokorelasi signifikan dan sudah menyerupai white noise.
*   **AIC/BIC**: nilai lebih rendah = model lebih baik (dengan asumsi kompleksitas seimbang). Bisa dilihat bahwa nilai AIC dan BIC yang relatif rendah memperkuat bahwa model ini efisien dan tidak overfitting. 

Dengan demikian, model ARIMA ini dapat dianggap layak untuk digunakan dalam prediksi time series.

  

### Langkah 6: Lakukan Prediksi dan Interpretasi

Setelah yakin bahwa model sudah memadai, kita bisa menggunakannya untuk membuat **prediksi masa depan**.

Model ARIMA dapat memproyeksikan nilai beberapa langkah ke depan, misalnya prediksi tujuh hari ke depan. Hasil prediksi biasanya juga disertai dengan **interval kepercayaan**, yaitu rentang atas dan bawah dari prediksi yang mencerminkan ketidakpastian model.

Contoh kode prediksinya berikut.

1.  \# Prediksi 7 langkah ke depan
2.  forecast \= model\_fit.predict(start\=len(df), end\=len(df)+6, typ\='levels')

4.  print("Prediksi 7 hari ke depan:")
5.  for idx, value in forecast.items():
6.     print(f"Hari ke-{idx}: {value:.2f}")

Ketika kode dijalankan, kita akan mendapatkan prediksi untuk tujuh hari ke depan berdasarkan model ARIMA yang dibuat.

![dos-053b3c8ae7979bc7fa997db0f62410f720250903160130.jpeg](https://assets.cdn.dicoding.com/original/academy/dos-053b3c8ae7979bc7fa997db0f62410f720250903160130.jpeg)

Kita juga bisa menampilkan prediksi dalam bentuk grafik garis agar lebih mudah dianalisis dengan kode berikut.

1.  \# Gabungkan data historis dan prediksi untuk visualisasi
2.  minggu\_forecast \= np.arange(len(df), len(df) + len(forecast))
3.  df\_pred \= pd.Series(forecast.values, index\=minggu\_forecast)

5.  \# Plot data aktual dan prediksi
6.  plt.figure(figsize\=(10, 5))
7.  plt.plot(df\['penjualan'\], label\='Data Historis')
8.  plt.plot(df\_pred, label\='Prediksi 7 Hari ke Depan', linestyle\='--', marker\='o')
9.  plt.title('Prediksi Penjualan Menggunakan ARIMA')
10.  plt.xlabel('Minggu')
11.  plt.ylabel('Penjualan')
12.  plt.legend()
13.  plt.grid(True)
14.  plt.tight\_layout()
15.  plt.show()

Hasilnya akan menampilkan visualisasi seperti ini.

![dos-13eeddddaf7d7562d52009b67379233520250903160130.jpeg](https://assets.cdn.dicoding.com/original/academy/dos-13eeddddaf7d7562d52009b67379233520250903160130.jpeg)

Grafik di atas adalah visualisasi hasil prediksi tujuh hari ke depan menggunakan model ARIMA. Garis biru menunjukkan data historis penjualan mingguan, sementara garis putus-putus dengan titik menunjukkan hasil prediksi dari minggu ke-101 hingga 107. 

Grafik ini membantu kita melihat bahwa model memproyeksikan tren penjualan tetap stabil mendekati nilai terakhir, yang konsisten dengan karakteristik data time series yang sudah di-differencing dan dipelajari oleh model.

Anda bisa akses kode lengkap yang dijelaskan dalam topik time series pada link berikut: [Modul 6 - Time Series.ipynb](https://colab.research.google.com/drive/1pHaze2ZApmAiBM27erZcbL-o15auULCO?usp=sharing)

## Rangkuman Matematika pada Algoritma AI

Berikut adalah rangkuman untuk materi Matematika pada Algoritma AI.

  

### Matematika dalam Computer Vision

*   **Representasi Gambar sebagai Matriks**  
    Gambar digital direpresentasikan sebagai matriks piksel dengan nilai intensitas (grayscale) atau RGB (warna). Setiap piksel menjadi elemen numerik yang bisa dianalisis secara statistik dan aljabar linier.
*   **Statistik Dasar Gambar: Histogram**  
    Histogram digunakan untuk memvisualisasikan distribusi intensitas piksel dalam gambar. Membantu mengidentifikasi pencahayaan, kontras, dan potensi noise dalam gambar.
*   **Statistik Spasial: Variansi dan Korelasi Antar Piksel**  
    Menganalisis perbedaan nilai antar piksel dalam area kecil gambar untuk mendeteksi pola, tekstur, dan struktur lokal. Berguna dalam klasifikasi tekstur dan segmentasi.
*   **Operasi Konvolusi dan Kernel**  
    Konvolusi digunakan untuk mengekstrak fitur penting dengan cara menggeser kernel (filter) pada gambar dan menghitung hasil penjumlahan bobot. Aplikasi penting dalam deteksi tepi, blur, dan sharpening.
*   **Deteksi Tepi dan Gradien**  
    Menggunakan operator, seperti Sobel atau Prewitt untuk menghitung perubahan intensitas antar piksel. Konsep gradien sangat penting dalam mengenali batas objek.
*   **Ekstraksi Fitur Tekstur: GLCM dan Fitur Haralick**  
    Gray-Level Co-occurrence Matrix (GLCM) menghitung seberapa sering pasangan nilai piksel muncul berdekatan dalam arah tertentu. Fitur Haralick seperti Contrast, Entropy, dan Homogeneity membantu klasifikasi tekstur.
*   **Reduksi Dimensi dan PCA**  
    Principal component analysis (PCA) digunakan untuk mereduksi dimensi citra sambil mempertahankan informasi penting. Umum digunakan sebelum klasifikasi atau kompresi gambar.

  

### Matematika dalam Natural Language Processing

*   **Distribusi Kata & WordCloud**  
    Digunakan untuk memahami frekuensi kata dalam korpus teks. WordCloud memvisualisasikan kata-kata berdasarkan frekuensinya, berguna dalam analisis sentimen dan pencarian informasi.
*   **Representasi Frekuensi Kata: Bag-of-Words (BoW)**  
    Mengubah dokumen teks menjadi vektor berdasarkan frekuensi kata, tanpa memperhatikan urutan atau konteks. Berguna untuk model sederhana seperti klasifikasi teks dasar.
*   **Pembobotan Kata: TF-IDF (Term Frequency–Inverse Document Frequency)**  
    Memberi bobot pada kata berdasarkan seberapa sering kata muncul dalam dokumen tertentu dibandingkan korpus secara keseluruhan. Lebih informatif daripada BoW karena mengurangi pengaruh kata umum.
*   **Representasi Semantik: Word Embedding**  
    Teknik seperti CBOW dan Skip-gram (Word2Vec) digunakan untuk menangkap makna kata dalam bentuk vektor berdimensi rendah. Kata dengan makna serupa memiliki representasi vektor yang mirip.
*   **Pengukuran Kemiripan: Cosine Similarity**  
    Digunakan untuk mengukur kesamaan antar teks atau dokumen berdasarkan representasi vektor (BoW, TF-IDF, Word Embedding). Nilai kemiripan ditentukan berdasarkan sudut antar vektor.

  

### Matematika dalam Time Series

*   **Proses Stokastik dan Dependensi Waktu**  
    Data time series dipandang sebagai proses stokastik, yakni nilai saat ini bergantung pada nilai-nilai sebelumnya. Ini berbeda dengan data i.i.d. yang asumsi dasarnya saling bebas.
*   **Stasioneritas dan Differencing**  
    Stasioneritas berarti data memiliki rata-rata dan variansi yang konstan sepanjang waktu. Jika tidak stasioner (misalnya ada tren), differencing dilakukan untuk menghilangkan tren tersebut.
*   **Autokorelasi (ACF) dan Autokorelasi Parsial (PACF)**  
    ACF digunakan untuk mengukur korelasi antara nilai saat ini dan nilai-nilai sebelumnya. PACF mengukur korelasi langsung setelah mengontrol pengaruh dari lag antara. Keduanya penting dalam menentukan parameter AR dan MA.
*   **Model AR (AutoRegressive)**  
    Model AR memprediksi nilai sekarang berdasarkan kombinasi linear dari nilai-nilai sebelumnya. Dikenali dari plot PACF yang terputus pada lag ke-p.
*   **Model MA (Moving Average)**  
    Model MA menggunakan kesalahan (error) dari prediksi sebelumnya untuk memperkirakan nilai sekarang. Dikenali dari plot ACF yang terputus pada lag ke-q.
*   **Model ARIMA (Autoregressive Integrated Moving Average)**  
    Menggabungkan AR, I (differencing), dan MA dalam satu model untuk menangani tren serta autokorelasi. Ditulis sebagai ARIMA(p, d, q), yakni p = AR order, d = differencing order, q = MA order.
*   **Evaluasi Model Time Series**  
    Evaluasi dilakukan dengan mengamati residual (sisa prediksi), memastikan mereka acak (white noise), serta melihat nilai AIC, BIC, dan uji Ljung-Box untuk memvalidasi model.


---
```
This file is located at: {{ page.path }}
```
---



