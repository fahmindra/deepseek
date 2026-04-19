---
slug: mai-2
title: mai-2
---
# Bab 2: Matematika dalam Pengolahan Data

Matematika adalah fondasi kokoh yang menopang kelancaran dan ketepatan proses pengolahan data. Beragam konsep dan formula matematis menjadi alat penting dalam mengekstrak informasi berharga dari sekumpulan data yang kompleks. Fokus utama pada bab ini adalah bertujuan mengolah data menjadi matang sehingga siap dibangunkan model prediktif untuk memprediksi tren dan pola di masa yang akan datang.

## 2.1 Data dan *Encoding*

Data adalah bagian terkecil dari suatu informasi. Menurut Kamus Besar Bahasa Indonesia (KBBI), data adalah keterangan yang benar dan nyata. Dalam kata lain, data dapat direpresentasikan sebagai fakta / kenyataan. Data dapat diperoleh melalui berbagai macam cara, yaitu pengamatan, pengukuran maupun penelitian. Bentuk dari data bisa berupa teks seperti huruf dan angka, gambar, audio dan juga video. Pada bidang ilmu komputer / informatika, data dalam bentuk digital khususnya tabular seperti pada Tabel 2.1 berikut.

Tabel 2.1 adalah contoh dataset dari diabetes dengan fitur / variabel yaitu jenis kelamin, usia, BMI (*body mass index*) dan diabetes. Fitur jenis kelamin dan diabetes memiliki tipe data kategorikal, fitur usia dan BMI memiliki tipe data numerik. Perbedaan antara kategorikal dan numerik adalah jika data dengan tipe kategorikal memiliki nilai dalam bentuk kategori seperti contoh di atas, contoh lainnya adalah tinggi / rendah, besar / kecil, lebar / sempit, ya / tidak, dan lain sebagainya. Sedangkan tipe data numerik contohnya adalah data dengan nilai dalam bentuk angka / numerik.

**Tabel 2.1 Contoh Dataset**

| Jenis kelamin | usia | BMI | Diabetes |
| :--- | :--- | :--- | :--- |
| Perempuan | 30 | 22 | Tidak |
| Laki-laki | 30 | 19 | Tidak |
| Laki-laki | 41 | 22 | Tidak |
| Laki-laki | 33 | 22 | Tidak |
| Perempuan | 61 | 35 | Ya |
| Laki-laki | 54 | 29 | Ya |
| Laki-laki | 60 | 33 | Ya |
| Laki-laki | 53 | 27 | Ya |

Komputer adalah sebuah mesin yang hanya dapat memproses data dalam bentuk angka, sehingga tipe data kategorikal biasanya akan diubah menjadi tipe numerik. Jika dilihat fitur jenis kelamin pada Tabel 2.1, bagaimana cara untuk mengubahnya menjadi numerik? Cara ini dinamakan dengan *label encoding*, diantaranya adalah sebagai berikut.

### 1. One Hot Encoding

Teknik ini akan memproduksi data baru dalam bentuk biner, yaitu 0 dan 1 (Wu *et al.*, 2022). Hasil *encoding* pada fitur jenis kelamin disajikan pada Tabel 2.2 berikut.

Mengacu pada Tabel 2.2, dapat dilihat bahwa terjadi penambahan fitur sesuai dengan banyaknya kategori pada fitur jenis kelamin sebelumnya. Pada data yang bernilai laki-laki akan memiliki nilai 1 pada kolom laki-laki, sedangkan pada kolom perempuan akan bernilai 1 jika data adalah perempuan. Kelemahan dari teknik ini adalah meningkatnya data dengan nilai 0 dan bertambahnya fitur dari dataset. Dasar dari kecerdasan buatan adalah model matematis. Sehingga jika terdapat data dengan sebaran nilai 0 di dalamnya, akan sangat mempengaruhi model.

**Tabel 2.2 Contoh Dataset Dengan *One Hot Encoding***

| Laki-laki | Perempuan | usia | BMI | Diabetes |
| :--- | :--- | :--- | :--- | :--- |
| 0 | 1 | 30 | 22 | Tidak |
| 1 | 0 | 30 | 19 | Tidak |
| 1 | 0 | 41 | 22 | Tidak |
| 1 | 0 | 33 | 22 | Tidak |
| 0 | 1 | 61 | 35 | Ya |
| 1 | 0 | 54 | 29 | Ya |
| 1 | 0 | 60 | 33 | Ya |
| 1 | 0 | 53 | 27 | Ya |

### 2. Ordinal Encoding

Sedikit berbeda dengan *encoding* sebelumnya, teknik ini cenderung lebih mudah diterapkan dan tidak memproduksi fitur tambahan yang berlebih. Jika menilik pada Tabel 2.1, maka hasil *ordinal encoding*nya adalah sebagai berikut.

Berdasarkan Tabel 2.3, maka jenis kelamin perempuan akan bernilai 0 dan laki-laki bernilai 1. Kelemahan dari teknik ini adalah tidak bisa menganggap sebuah data memiliki nilai lebih tinggi / lebih rendah. Misalkan saja untuk perempuan = 0 dan laki-laki = 1. Jika pada matematika angka 1 memiliki nilai lebih besar dari 0. Sedangkan pada kenyataannya jenis kelamin itu setara. Apakah data hasil *encoding* akan bernilai biner? Jawabannya adalah tidak. Sebagai contoh jika terdapat data warna {tosca, pink, vermilion, terracota}. Hasil *encoding*nya adalah {tosca: 0; pink:1; vermilion:2; terracota:3}.

**Tabel 2.3 Contoh dataset dengan *ordinal encoding***

| Jenis kelamin | usia | BMI | Diabetes |
| :---: | :---: | :---: | :--- |
| 0 | 30 | 22 | Tidak |
| 1 | 30 | 19 | Tidak |
| 1 | 41 | 22 | Tidak |
| 1 | 33 | 22 | Tidak |
| 0 | 61 | 35 | Ya |
| 1 | 54 | 29 | Ya |
| 1 | 60 | 33 | Ya |
| 1 | 53 | 27 | Ya |

---

## 2.2 Word Embedding

Data tidak selalu dalam bentuk angka, pada beberapa kasus data dalam bentuk teks kalimat / paragraf. Karena komputer dalam komputasinya hanya mengenal angka, sehingga data dalam bentuk teks perlu dilakukan ekstraksi untuk mendapatkan esensi data dalam bentuk angka. Data angka tersebut merupakan representasi dari data berupa teks tersebut. Sebagai contoh, misalkan terdapat dataset seperti pada Tabel 2.4 berikut:

**Tabel 2.4 Contoh Dataset Berisi Kata**

| | |
| :--- | :--- |
| Dokumen 1 | Saya makan nasi |
| Dokumen 2 | Saya minum kopi |
| Dokumen 3 | Saya suka makan nasi dan suka minum kopi |

Beberapa cara untuk *word embedding* dapat dilakukan dengan 2 metode, yaitu *count vectorizer* dan TF-IDF. Berikut penjelasannya

### 1. Count Vectorizer

Proses perhitungannya didasarkan pada kemunculan kata pada dataset. (Tripathy, Agrawal and Rath, 2015). Dengan menggunakan *count vectorizer* akan memproduksi *sparse matrix*. Dengan berdasarkan pada dataset Tabel 2.4, didapatkan *corpus* seperti pada Tabel 2.5 beserta nilai *embedding*nya. *Corpus* adalah sekumpulan kata yang membentuk kalimat.

**Tabel 2.5 Dataset Hasil *Embedding* Dengan *Count Vectorizer***

| | saya | makan | nasi | minum | kopi | dan | suka |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Dokumen 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 |
| Dokumen 2 | 1 | 0 | 0 | 1 | 1 | 0 | 0 |
| Dokumen 3 | 1 | 1 | 1 | 1 | 1 | 1 | 2 |

Pada dokumen 1 terdapat kata "saya" yang kemunculannya hanya satu kali, sehingga pada Tabel 2.5 bernilai 1. Namun kata "minum" pada tidak muncul pada dokumen 1 sehingga bernilai 0. Hal ini berlaku hingga kata terakhir pada tiap dokumennya. Oleh karena itu kata "suka" pada dokumen 3 bernilai 2 karena kemunculannya terjadi sebanyak dua kali. Sehingga dapat dikatakan bahwa *count vectorizer* yang ditunjukkan oleh Tabel 2.5 adalah representasikan kemunculan kata dari seluruh *corpus* pada tiap dokumennya.

### 2. TF-IDF

TF-IDF (*Term Frequency-Inverse Document Frequency*). Metode ini banyak digunakan karena dalam mekanismenya dengan melakukan pembobotan dari suatu *term*. Kelebihan dari metode ini pada bidang kecerdasan buatan adalah dapat meningkatkan performa pada *recall* dan *precision* (Deolika, Kusrini and Luthfi, 2019). TF berarti rekuensi dari kemunculan kata pada suatu dokumen, sedangkan IDF adalah untuk melakukan pembobotan pada kata yang berbeda dari keseluruhan dokumen. Perhitungan dari TF-IDF mengacu pada persamaan (2.1) berikut:

$$tfidf(t_j, d_i) = tf(t_j, d_i) \times (\log \frac{N}{df(t_j)}) + 1$$ **(2.1)**

dengan $tf(t_j, d_i)$ adalah frekuensi kemunculan kata $t_j$ dalam satu dokumen $d_i$, $df(t_j)$ adalah banyaknya dokumen terhadap kemunculan $t_j$ dan $N$ adalah jumlah banyaknya dokumen. Berdasarkan formula di atas, rumus $IDF = (\log \frac{N}{df(t_j)}) + 1$. Pada formula terdapat penambahan 1 yang bertujuan untuk menghindari produksi nilai 0. Apabila tanpa penambahan 1, maka hasil perkalian TF dengan IDF akan banyak menghasilkan nilai 0. Mengacu pada dataset Tabel 2.4 di atas, maka nilai $N= 3$ dan nilai $tf(t_j, d_i)$ didapatkan dari tabel *count vecctorizer* di atas. Tabel akhir perhitungan TF-IDF ditunjukkan pada Tabel 2.6 berikut:

**Tabel 2.6 Dataset Hasil *Embedding* dengan TF-IDF**

| | saya | makan | nasi | minum | kopi | dan | suka |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Dokumen 1 | 1 | 1,176091 | 1,176091 | 1 | 1 | 1 | 1 |
| Dokumen 2 | 1 | 1 | 1 | 1,176091 | 1,176091 | 1 | 1 |
| Dokumen 3 | 1 | 1,176091 | 1,176091 | 1,176091 | 1,176091 | 1,477121 | 1,954243 |

$$tfidf(saya, dokumen_1) = 1 \times (\log \frac{3}{3}) + 1 = 1$$
$$tfidf(makan, dokumen_1) = 1 \times (\log \frac{3}{2}) + 1 = 1,176091$$
$$tfidf(minum, dokumen_1) = 0 \times (\log \frac{3}{2}) + 1 = 1$$
$$tfidf(saya, dokumen_2) = 1 \times (\log \frac{3}{3}) + 1 = 1$$
$$tfidf(dan, dokumen_3) = 1 \times (\log \frac{3}{1}) + 1 = 1,477121$$
$$tfidf(suka, dokumen_3) = 2 \times (\log \frac{3}{1}) + 1 = 1,954243$$

Berdasarkan Tabel 2.6, dapat diamati bahwa semakin unik kata pada *corpus* dalam sekumpulan dokumen, akan menghasilkan nilai TF-IDF yang tinggi. Berbeda dengan kata yang sering muncul atau yang jarang tidak muncul, maka nilainya akan bernilai 1. TF-IDF sangat sering digunakan sebagai dalam pembobotan karena dapat menyeimbangkan bobot antara kata yang jarang muncul dan kata yang sering muncul (Wibowo, Hasbi and Anshori, 2024).

---

## 2.3 Transformasi Data

Data yang telah dikumpulkan untuk dimasukkan ke dalam proses kecerdasan buatan tidak selalu siap untuk diolah. Pada beberapa kasus, data harus ditransformasi terlebih dahulu ke dalam bentuk standar atau normal. Kedua metode itu adalah teknik yang umum digunakan.

Sebagai contoh pada Tabel 2.7 terdapat sebuah dataset tentang harga kamar kos yang terdiri dari jumlah fasilitas dan luas kamar dalam bentuk meter persegi sebagai berikut. Data pada Tabel 2.7 memiliki rentang dan skala yang berbeda.

**Tabel 2.7 Dataset Untuk Transformasi Data**

| n fasilitas | Luas Kamar (m2) | Harga (ribu) |
| :---: | :---: | :---: |
| 4 | 4 | 450 |
| 5 | 6,25 | 500 |
| 6 | 9 | 900 |
| 7 | 6,25 | 600 |

Bagaimana jika standardisasi dan normalisasi diterapkan pada data Tabel 2.7 di atas? Berikut adalah penjelasannya.

### 1. Standardisasi

Standardisasi adalah teknik yang dapat diaplikasikan untuk mentransformasi data ke dalam bentuk yang lebih standar. Cara kerja dari metode ini adalah melakukan skala ulang pada data ke dalam bentuk distribusi normal sehingga nilai rerata 0 dan standar deviasi = 1 (Ghosh, 2022). Pada beberapa literatur lain, standardisasi juga disebut juga dengan normalisasi Z-score. Perhatikan formula yang disajikan berikut:

$$x' = \frac{x - \bar{x}}{\sigma}$$ **(2.2)**

dengan $x'$ adalah data baru hasil standardisasi, $x$ adalah data yang akan di standardisasi, $\bar{x}$ adalah rerata data pada satu kolom yang sama, dan $\sigma$ adalah standar deviasi. Jika standardisasi diterapkan pada data Tabel 2.7, maka diperoleh data baru seperti pada Tabel 2.8.

Tabel 2.8 adalah data baru hasil dari produksi menggunakan standardisasi. Jika pada tiap baris data per fitur dilakukan perhitungan rerata, maka akan didapatkan nilai 0 dan standar deviasinya akan bernilai 1.

**Tabel 2.8 Dataset Hasil Standardisasi Data**

| n fasilitas | Luas Kamar (m2) | Harga (ribu) |
| :---: | :---: | :---: |
| -1,34164 | -1,34015667 | -0,93094934 |
| -0,44721 | -0,070534562 | -0,64450339 |
| 0,447214 | 1,481225793 | 1,64706421 |
| 1,341641 | -0,070534562 | -0,07161149 |

### 2. Normalisasi

Sebuah metode yang dapat digunakan untuk menyeragamkan rentang data yang sangat berbeda adalah normalisasi. Dengan normalisasi, data yang memiliki rentang beragam dapat di transformasi ke dalam skala dengan rentang yang sama (Anshori, Haris and Teja Kusuma, 2023). Kelebihan dari normalisasi ini adalah dapat meningkatkan performa dari model kecerdasan buatan dan berpengaruh terhadap waktu komputasinya. Salah satu metode normalisasi yang umum digunakan adalah *min-max*.

$$x' = (max - min) \frac{x - minX}{maxX - minX} + min$$ **(2.3)**

dengan $x'$ adalah data hasil normalisasi, $x$ adalah data yang akan dinormalisasi, $maxX$ adalah data dengan nilai terbesar dan $minX$ adalah data dengan nilai terkecil. Sedangkan $max$ adalah rentang terbesar dan $min$ adalah rentang terkecil yang diinginkan. Jika ingin transformasi data dengan rentang 0 – 1, ikuti formulasi yang ditunjukkan pada persamaan (2.4).

$$x' = \frac{x - minX}{maxX - minX}$$ **(2.4)**

Sebagai catatan, normalisasi bekerja berdasarkan kolom yang sama karena kolom pada sebuah dataset berupa satu buah variabel independen (berdiri sendiri). Contoh implementasi dari normalisasi *min-max*, kita gunakan data dari Tabel 2.7 di atas. Tabel 2.9 berikut adalah data hasil dari implementasi normalisasi *min-max*.

**Tabel 2.9 Dataset Hasil Normalisasi Data**

| n fasilitas | Luas Kamar (m2) | Harga (ribu) |
| :---: | :---: | :---: |
| 0 | 0 | 0 |
| 0,333333 | 0,45 | 0,11111111 |
| 0,666667 | 1 | 1 |
| 1 | 0,45 | 0,33333333 |

Tabel 2.9 di atas menunjukkan hasil normalisasi *min-max*. Data yang diproduksi menghasilkan rentang yang seragam, yaitu 0 hingga 1. Walaupun data terlihat jauh berubah, tetapi normalisasi ini masih menjaga relasi dengan data aslinya (Patro and Kumar, no date) sehingga informasi yang terkandung tidak hilang.

---

## 2.4 Seleksi Fitur

Beberapa kasus pada dataset adalah terdapat fitur yang tidak ada hubungannya dalam penggalian pola tersembunyi dari data. Oleh karena itu perlu dilakukan seleksi fitur untuk menghilangkan fitur yang tidak penting tersebut. Keberadaan fitur yang bias dapat mempengaruhi kinerja dari kecerdasan buatan saat diterapkan nantinya di masa mendatang. Selain itu dengan seleksi fitur maka fitur dataset akan berkurang dan dapat mempercepat waktu komputasi sehingga tidak membutuhkan *resource* yang lebih banyak. Berikut terdapat beberapa teknik sederhana yang dapat diterapkan dalam melakukan seleksi fitur.

### 1. Correlation

Koefisien korelasi atau *correlation coefficient* adalah derajat nilai untuk mengetahui seberapa erat hubungan antar variabel / fitur (Zhou, Wang and Zhu, 2022). Pada beberapa literatur, metode ini disebut juga dengan *Pearson Correlation Coefficient* dan formulanya ditunjukkan pada persamaan berikut:

$$PCC = \frac{Cov(X, Y)}{\sigma X \cdot \sigma Y}$$ **(2.5)**

dengan $Cov(X, Y)$ adalah *covariance* dari dua buah matrik fitur dan adalah $\sigma$ standar deviasi. Kecenderungan nilai korelasi yang digunakan adalah yang mendekati nilai 1 atau -1. Hal ini dikarenakan hubungan kuat antar variabel adalah nilai PCC yang tinggi. Hubungan kuat positif adalah PCC mendekati nilai 1 dan sebaliknya jika hubungan kuat negatif adalah PCC dengan nilai mendekati -1. Berbeda jika nilai PCC mendekati nilai 0 yang menandakan bahwa fitur tersebut memiliki hubungan lemah antar variabel. Bahkan dapat dikatakan tidak memiliki hubungan jika bernilai sama dengan nol. Pada beberapa kasus, PCC yang bernilai rendah menunjukkan variabel tersebut akan dihapuskan dari dataset. Literatur lain menyebutkan bahwa *threshold* untuk PCC antara -0.5 hingga 0.5 yang dapat dihapupskan dari dataset. Sebagai contoh dengan mengacu Tabel 2.7 di atas, maka didapatkan Tabel 2.10 PCC seperti berikut:

**Tabel 2.10 Korelasi Koefisien Data Per Fitur**

| | Fasilitas | luas kamar | harga |
| :--- | :---: | :---: | :---: |
| fasilitas | 1 | | |
| Luas kamar | 0,599336 | 1 | |
| harga | 0,544436 | 0,934451 | 1 |

Tabel 2.10 di atas menunjukkan nilai korelasi antar variabelnya dan menghasilkan nilai > 0.5. Maka dari itu fitur tidak ada yang di seleksi untuk dihapus dan masih dipertahankan.

### 2. Information Gain

*Information gain* (IG) adalah metode yang secara luas digunakan untuk seleksi dan menyaring fitur. Kelebihan metode ini adalah handal diterapkan untuk mengurangi *noise* pada dataset terhadap fitur yang tidak relevan (Syafitri Hidayatul AA, Yuita Arum S, 2018). Umumnya IG diaplikasikan pada kecerdasan buatan khususnya dalam algoritma *decision tree* dalam pembuatan cabang terhadap pohon keputusannya. Berikut formula dari IG.

$$InformationGain(S, A) = Entropy (S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} Entropy (S_v)$$ **(2.6)**

$$Entropy (S) = \sum_{i}^c -p_i \log_2 p_i$$ **(2.7)**

dengan $S$ adalah banyaknya fitur, $A$ adalah kelas target, $S_v$ adalah banyaknya sampel pada tiap $v$, dan $v$ adalah nilai yang menunjukkan probabilitas untuk kelas $A$. Dalam komputasinya, IG memerlukan *entropy* yang memiliki arti informasi mengenai proporsi pembagian kelas. Nilai dari *entropy* memiliki rentang antara 0 – 1. Nilai ini memiliki makna banyaknya sampel pada kelas dataset. Jika bernilai 0 maka sampel hanya berada di satu kelas. Jika bernilai 1 maka sampel berada di masing-masing kelas (Lishania, Goejantoro and Nasution, 2019). Dapat dikatakan *entropy* adalah nilai untuk mengukur sebaran acak data.

Dalam proses seleksi fitur dengan IG dengan melakukan perankingan nilai IG dari tiap fiturnya. Nilai IG yang tinggi menandakan fitur tersebut mengandung informasi yang terikat dengan kelasnya. Berkebalikan jika nilai IG rendah mendekati 0 menandakan tidak memiliki informasi yang berhubungan dengan kelasnya. Fitur yang dipertahankan jika memiliki nilai tinggi, dan yang rendah mendekati nilai 0 akan dikecualikan dari dataset.

---

## 2.5 Reduksi Dimensi

Pada dunia nyata, data tidak selalu memiliki jumlah fitur yang sedikit. Semakin ke sini data memiliki ukuran yang sangat besar sehingga disebut dengan *big data*. Data yang berukuran besar bisa jadi memiliki baris data yang sangat banyak dan juga memiliki fitur yang teramat banyak. Semakin banyak fitur pada dataset, memiliki pengaruh terhadap dimensi data. Data dengan dimensi yang tinggi susah untuk divisualisasikan dan pada komputasinya membutuhkan sumber daya yang besar. Cara yang paling tepat untuk mengatasinya adalah dengan reduksi dimensi. Terdapat dua pendekatan yang terkenal handal untuk mengatasinya, yaitu PCA dan LDA.

### 1. Principal Component Analysis (PCA)

Metode ini sering diaplikasikan pada data dengan dimensi yang tinggi untuk mereduksi dimensinya. Dimensi data yang tinggi adalah data dengan jumlah fitur atau variabel yang sangat banyak. Selain itu kelebihan dari PCA adalah dapat mempertahankan informasi yang dikandung oleh dataset walau telah mengalami reduksi dimensi, meningkatkan visualisasi data karena dimensi data menjadi rendah, dapat mendeteksi adanya informasi penting dan pada kasus klasifikasi pada kecerdasan buatan dapat meningkatkan performanya (Zhu, Idemudia and Feng, 2019). PCA dapat dikatakan sebagai metode reduksi dimensi yang bersifat *unsupervised* karena tidak memperhatikan kelas target dari dataset.

Langkah-langkah kalkulasi dari PCA cukup sederhana dan mudah. Dimulai dengan standardisasi seluruh fitur, selanjutnya menghitung *eigenvalues* dan *eigenvectors* dengan metode *eigen value decomposition* (EVD) atau *single value decomposition* (SVD). Secara general, formula dari PCA adalah sebagai berikut:

$$Ax = \lambda x$$ **(2.8)**

dengan $\lambda$ adalah skalar. $\lambda$ di sini disebut juga dengan *eigenvalue* dari matriks $A$ dan $x$ menjadi *eigenvector* yang berkorespondensi dengan $\lambda$. Jika $x$ adalah vektor yang bernilai bukan nol, maka akan memenuhi formula (2.9) berikut:

$$(\lambda I - A)x = 0$$ **(2.9)**

dengan $I$ adalah matrik identitas. Sehingga *eigenspace* tercipta mengikuti formula (2.10):

$$E = \{x: (A - \lambda I)x\} = 0$$ **(2.10)**

Saat *eigen space* telah didapatkan, tahap reduksi dimensinya adalah dengan mengurutkan *eigenvector* oleh *eigenvalue* dari tertinggi hingga terendah. Ukuran dimensi luarannya adalah $m \times k$ dengan $k$ adalah banyaknya *principal component* yang digunakan dan $m$ adalah banyaknya baris data. Pada beberapa literatur dikatakan bahwa pemilihan *principal component* yang optimal adalah mengandung 85% varian dari data asli (Zhongwen and Huanghuang, 2017).

### 2. Linear Discriminant Analysis (LDA)

Selain PCA, terdapat metode yang pembelajarannya bersifat *supervised* yaitu *linear discriminant analysis* (LDA). Metode ini sangat memperhatikan kelas target pada proses reduksi dimensinya dan cocok untuk data dengan kasus klasifikasi / regresi karena ada target yang di prediksi. Cara kerjanya cukup sederhana yaitu dengan memproyeksikan data ke dalam ruang vektor serta meminimalkan jarak data pada kelas yang sama dan memaksimalkan jarak antar kelas yang berbeda (Xie, 2015). Sehingga data akan saling mendekat dengan kelas yang sama.

Langkah kalkulasi dari LDA ada 3, yaitu menghitung jarak dalam kelas yang sama, menghitung jarak antar kelas yang berbeda dan menciptakan data dengan ruang dimensi yang lebih rendah(Ilias *et al.*, 2016). Berbeda dengan PCA, LDA akan otomatis melakukan transformasi dataset ke dalam dimensi yang lebih rendah tanpa perlu melakukan *tuning* persentasenya.