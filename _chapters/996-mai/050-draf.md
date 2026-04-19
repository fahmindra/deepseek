---
slug: mai-5
title: mai-5
---

# Bab 5: Konsep Matematika Pada Statistik dan Probabilitas

## 5.1 Statistik dan Probabilitas

Penggunaan statistik dan probabilitas dalam kecerdasan buatan terus berkembang dengan sangat pesat, terutama didorong oleh kemajuan dalam teori statistik dan probabilitas, peningkatan dalam teknologi komputasi, dan ketersediaan data dalam jumlah besar. Dalam bab ini, akan dibahas mengenai berbagai konsep dan teknik statistik dan probabilitas yang secara umum banyak digunakan dalam kecerdasan buatan.

Statistik dan probabilitas merupakan salah satu landasan utama dalam kecerdasan buatan (*Artificial Intelligence*, AI) dan pembelajaran mesin (*machine learning*) (LeCun *et al.*, 2015). Metode statistik memungkinkan analisis data yang efektif dan validasi model, sedangkan konsep probabilitas mendasari banyak algoritma yang digunakan dalam kecerdasan buatan (Dangeti, 2017). Pemahaman yang baik tentang statistik dan probabilitas dinilai sangat penting untuk mengembangkan model kecerdasan buatan yang akurat, dapat diandalkan, dan dapat diinterpretasikan. Dalam konteks kecerdasan buatan, statistik digunakan untuk memahami data melalui analisis deskriptif dan inferensial, sementara probabilitas digunakan untuk membuat prediksi dan mengambil keputusan dalam tingkat ketidakpastian tertentu. Saat ini, banyak algoritma pembelajaran mesin, seperti regresi, klasifikasi, dan klasterisasi, mengandalkan prinsip-prinsip statistik dan probabilistik untuk mengidentifikasi pola dan membuat prediksi berdasarkan data historis (Sarker, 2021).

Berbagai metode statistik dan probabilitas banyak bergantung pada penggunaan peralatan-peralatan berbasis matematika, ketika persamaan-persamaan sering kali dibuat secara lebih umum untuk diterapkan dalam berbagai keperluan. Penggunaan konsep matematika pada statistik dan probabilitas diterapkan untuk tujuan optimasi dan menghindari penilaian secara bias. Mengingat luasnya bidang matematika yang dapat digunakan untuk keperluan statistik dan probabilitas, kecenderungan untuk terus mengembangkan algoritma yang semakin efisien dan akurat terus dilakukan. Selain sebagai “jangkar” daripada statistik dan probabilitas, matematika melalui serangkaian penerapan aljabar linier mampu menyusun algoritma yang dinilai sangat robust untuk mengukur, memprediksi, dan mengambil keputusan terhadap data dalam jumlah yang sangat banyak dan telah membantu manusia dalam berbagai keperluan, baik pada skala yang sangat teknis maupun kehidupan sehari-hari.

## 5.2 Sejarah Singkat

Penggunaan statistik dan probabilitas dalam kecerdasan buatan memiliki sejarah yang cukup panjang yang dapat ditelusuri kembali ke beberapa dekade yang lalu (Stigler, 2002). Beberapa tonggak penting dalam sejarah ini yaitu meliputi:

1. **Tahun 1950-an dan 1960-an:** Pada masa awal perkembangan kecerdasan buatan, peralatan statistika banyak digunakan, terutama dalam pengolahan data (*data processing*) dan pengenalan pola (*pattern recognition*). Model probabilistik seperti distribusi Gaussian mulai digunakan untuk memodelkan data dan membuat prediksi.
2. **Tahun 1970-an dan 1980-an:** Metode statistik dan probabilistik mulai diterapkan secara lebih luas dalam pembelajaran mesin. Algoritma regresi linear dan logistik, yang didasarkan pada prinsip-prinsip statistik, menjadi populer. Pada saat yang sama, konsep probabilistik seperti Teorema Bayes mulai diterapkan dalam algoritma pengenalan pola dan klasifikasi.
3. **Tahun 1990-an:** Pengenalan jaringan saraf (*neural network*) Bayesian dan model Markov tersembunyi (*Hidden Markov Models* - HMM) menandai kemajuan besar dalam penggunaan probabilitas dalam kecerdasan buatan dan pembelajaran mesin. Algoritma ini memungkinkan pemodelan yang lebih canggih dan akurat dari data yang kompleks, terutama dalam pengenalan suara dan teks.
4. **Tahun 2000-an dan seterusnya:** Dengan munculnya konsep *big data* serta peningkatan daya komputasi yang signifikan dengan berkembang pesatnya teknologi prosesor komputer, penggunaan statistik dan probabilitas dalam kecerdasan buatan semakin meluas. Metode seperti pengambilan keputusan berbasis probabilitas, pohon keputusan (*decision tree*), dan algoritma pembelajaran *ensemble* seperti *Random Forest* dan *Boosting*, menjadi bagian penting dari peralatan pembelajaran mesin.

## 5.3 Dasar-dasar Probabilitas dan Distribusi

Probabilitas mengukur sejauh mana suatu peristiwa kemungkinan terjadi dan menyediakan kerangka untuk memahami dan mengelola ketidakpastian dalam data (Ross, 2014). Konsep dasar probabilitas mencakup aturan-aturan dasar seperti probabilitas kondisional, hukum probabilitas total, dan teorema Bayes, yang memungkinkan kita untuk menghitung dan memprediksi kemungkinan kejadian berdasarkan informasi yang tersedia. Selain itu, distribusi probabilitas menggambarkan bagaimana probabilitas didistribusikan di seluruh kemungkinan nilai dari variabel acak, dan meliputi berbagai jenis distribusi seperti distribusi normal, binomial, dan Poisson.

Memahami dasar-dasar probabilitas dan distribusi adalah kunci untuk membangun model yang akurat dan efektif dalam berbagai aplikasi, mulai dari analisis data hingga pengembangan algoritma pembelajaran mesin. Sub-bab ini bertujuan untuk memberikan landasan yang solid dalam konsep-konsep ini, sehingga mempersiapkan pembaca untuk memahami teknik-teknik statistik yang lebih kompleks dan aplikasinya dalam kecerdasan buatan.

### 5.3.1 Definisi dan konsep probabilitas dan distribusi

Dalam suatu probabilitas, terlebih dahulu diperlukan pemahaman mengenai analisis kombinatorial, yang mana dapat membantu banyak hal dalam probabilitas, termasuk bilangan faktorial, permutasi, dan kombinasi (Sumarlin, 2023). Bilangan faktorial adalah bilangan hasil kali dari sebuah bilangan bulat positif dengan semua bilangan bulat positif yang kurang dari bilangan tersebut. Umumnya bilangan ini dinotasikan dengan tanda seru (!), misalnya 5! (dibaca “lima faktorial”), yang sama dengan $5 \times 4 \times 3 \times 2 \times 1 = 120$. Bilangan faktorial sering digunakan dalam permutasi dan kombinatorika.

Secara umum, permutasi adalah penyusunan jumlah (n) objek yang diambil sebanyak (r) objek tertentu dengan memperhatikan penyusunannya. Permutasi mengacu pada pengaturan dan susunan objek-objek dalam suatu himpunan, di mana urutan objek-objek tersebut menjadi penting. Misalnya, 3 buah bola berwarna merah, hijau, dan biru, maka berapa banyak cara yang dimungkinkan untuk mengatur bola-bola tersebut secara berbeda dalam sebuah garis? Jawabannya adalah $3 \times 2 \times 1 = 6$, karena setiap bola memiliki tiga tempat yang mungkin untuk ditempatkan pada urutan pertama, kemudian dua tempat untuk urutan kedua, dan satu tempat untuk urutan ketiga.

Sedangkan kombinasi mengacu pada pemilihan objek-objek dari suatu himpunan, ketika urutan objek-objek tersebut tidak menjadi penting. Dalam kombinasi, objek-objek dapat dipilih dalam urutan apa pun. Sebagai contoh, jika ada 4 buah bola berwarna merah, hijau, biru, dan kuning, dan ingin dipilih 2 bola dari keempat bola tersebut, berapa banyak cara yang mungkin? Jawabannya adalah 6, karena 6 adalah kemungkinan kombinasi yang berbeda: merah dan hijau, merah dan biru, merah dan kuning, hijau dan biru, hijau dan kuning, serta biru dan kuning. Rumus untuk menghitung permutasi dan kombinasi adalah sebagai berikut:

$$nPr = \frac{n!}{(n-r)!} \tag{5.1}$$

$$nCr = \frac{n!}{r!(n-r)!} \tag{5.2}$$

Dengan n adalah jumlah objek dalam himpunan, dan r adalah jumlah objek yang diatur (dalam kasus permutasi) atau dipilih (dalam kasus kombinasi).

Selanjutnya, probabilitas adalah ukuran dari kemungkinan terjadinya suatu peristiwa. Probabilitas sebuah kejadian A dilambangkan dengan P(A), dan nilai ini berada di antara 0 dan 1, yaitu:

* P(A) = 0 menunjukkan bahwa kejadian tersebut tidak mungkin terjadi.
* P(A) = 1 menunjukkan bahwa kejadian tersebut pasti terjadi.

Ruang Sampel (S) adalah himpunan semua hasil yang mungkin terjadi dari suatu percobaan, sering juga disebut dengan himpunan dari titik sampel. Sedangkan, kejadian (A) adalah himpunan bagian dari ruang sampel.

**1. Hukum Penjumlahan:** yaitu jika A dan B adalah dua kejadian yang saling eksklusif/unik, maka:
$$P(A \cup B) = P(A) + P(B) \tag{5.3}$$
Misalnya, jika sebuah dadu dilemparkan, maka probabilitas munculnya mata dadu 4 (A) dan munculnya mata dadu lebih kecil dari 3 adalah:
$P(A) = \frac{1}{6}$, $P(B) = \frac{2}{6}$
$P(A \cup B) = P(A) + P(B) = \frac{1}{6} + \frac{2}{6} = \frac{3}{6} = \frac{1}{2} = 0,5$

**2. Hukum Perkalian:** yaitu jika A dan B adalah dua kejadian yang independen, maka hukum perkaliannya yaitu $P(A \cap B) = P(A) \times P(B)$. Misalnya sebuah mata uang logam dan sebuah dadu dilemparkan satu kali secara bersamaan. Maka probabilitas munculnya sisi muka pada uang logam dan mada dadu 4 adalah:
$$P(A \cap B) = P(A) \times P(B) \tag{5.4}$$
$$P(A \cap B) = \frac{1}{2} \times \frac{1}{6} = \frac{1}{12}$$

### 5.3.2 Probabilitas bersyarat dan Teorema Bayes

Probabilitas bersyarat merupakan suatu prinsip dalam teori probabilitas, prinsip ini mengacu pada probabilitas terjadinya suatu peristiwa tertentu berdasarkan fakta bahwa peristiwa sebelumnya telah terjadi. Misalnya, probabilitas dari kejadian A terjadi dengan syarat bahwa B telah terjadi, dilambangkan sebagai P(A|B):

$$P(A|B) = \frac{A \cap B}{P(B)} \tag{5.5}$$

Sementara itu, Teorema Bayes memberikan cara untuk memperbarui probabilitas suatu hipotesis berdasarkan bukti baru, yang dilambangkan dengan persamaan:

$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)} \tag{5.6}$$

Teorema ini sangat penting dalam banyak aplikasi kecerdasan buatan, terutama dalam algoritma pengklasifikasian seperti Naive Bayes.

### 5.3.3 Distribusi diskrit

**1. Distribusi binomial**
Distribusi ini menggambarkan jumlah sukses dalam sejumlah tetap percobaan independen yang masing-masing memiliki dua kemungkinan hasil (sukses atau gagal). Jika $X$ adalah variabel acak binomial dengan $n$ percobaan dan probabilitas sukses $p$, maka:
$$P(X = k) = \binom{n}{k} p^k (1 - p)^{n-k} \tag{5.7}$$
dengan $k = 0, 1, 2, \dots, n$.

**2. Distribusi Poisson**
Distribusi ini digunakan untuk menggambarkan jumlah kejadian dalam interval waktu atau ruang tertentu yang terjadi secara independen dengan laju rata-rata $\lambda$. Jika $X$ adalah variabel acak Poisson dengan parameter $\lambda$, maka:
$$P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!} \tag{5.8}$$
dengan $k = 0, 1, 2, \dots, n$.

### 5.3.4 Distribusi kontinu

**1. Distribusi normal**
Distribusi ini sangat penting dalam statistik dan banyak digunakan karena banyak fenomena alam mengikuti distribusi ini. Distribusi normal dengan rerata $\mu$ dan standar deviasi $\sigma$ memiliki fungsi kepadatan probabilitas (*probability density function*, PDF):
$$f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}} \tag{5.9}$$

**2. Distribusi eksponensial**
Distribusi ini sering digunakan untuk memodelkan waktu antara kejadian dalam proses Poisson. Jika $X$ adalah variabel acak eksponensial dengan parameter laju $\lambda$, maka PDF-nya adalah:
$$f(x) = \lambda e^{-\lambda x} \{ -\lambda x \} \tag{5.10}$$
untuk $x \geq 0$.

### 5.3.5 Fungsi Distribusi Kumulatif (CDF) dan Fungsi Kepadatan Probabilitas (PDF)

*Cumulative Distribution Function* (CDF) dan *Probability Density Function* (PDF) adalah komponen fundamental dalam analisis probabilistik. Keduanya berperan penting dalam memodelkan dan memahami distribusi probabilitas dari suatu data. CDF memberikan informasi tentang probabilitas kumulatif hingga titik tertentu dalam data, sedangkan PDF menguraikan bagaimana probabilitas terdistribusi di sekitar nilai-nilai yang berbeda dalam suatu rentang. Dalam konteks kecerdasan buatan, CDF dan PDF digunakan dalam berbagai algoritma untuk membuat prediksi yang lebih akurat dan membantu dalam pengambilan keputusan yang didasarkan pada analisis data yang mendalam. Dengan memanfaatkan alat ini, algoritma dapat menangani ketidakpastian dan variabilitas dalam data dengan lebih efektif, yang pada akhirnya meningkatkan kinerja model kecerdasan buatan atau pembelajaran mesin.

**1. Fungsi Distribusi Kumulatif (CDF)**
CDF $F(x)$ dari variabel acak $X$ memberikan probabilitas bahwa $X$ akan lebih kecil atau sama dengan $x$:
$$F(x) = P(X \leq x) \tag{5.11}$$

**2. Fungsi Kepadatan Probabilitas (PDF)**
Untuk variabel acak kontinu, PDF $f(x)$ menggambarkan kepadatan probabilitas di sekitar nilai $x$. Integral dari PDF memberikan CDF:
$$F(x) = \int_{-\infty}^{x} f(t) dt \tag{5.12}$$

## 5.4 Statistik Deskriptif dan Inferensi Statistik

Statistik deskriptif dan inferensi statistik merupakan komponen kunci dalam analisis data. Statistik deskriptif memberikan gambaran umum mengenai data, sementara inferensi statistik memungkinkan dibuatnya generalisasi dan simpulan tentang populasi berdasarkan sampel data. Kedua metode ini esensial dalam kecerdasan buatan untuk memahami data dan membuat prediksi yang akurat.

### 5.4.1 Ukuran pemusatan dan penyebaran

Ukuran Pemusatan merujuk pada serangkaian nilai yang menggambarkan posisi tengah atau pusat dari distribusi data. Nilai-nilai ini memberikan gambaran tentang lokasi sentral ketika data cenderung berkumpul atau berfokus. Dengan memahami ukuran pemusatan, bisa diperoleh wawasan tentang bagaimana data terdistribusi di sekitar titik-titik kunci, yang sangat penting dalam berbagai analisis statistik dan aplikasinya. Ukuran-ukuran ini mencakup beberapa konsep penting, di antaranya adalah:

1. **Mean (Rerata)**, yaitu jumlah dari semua nilai dibagi dengan jumlah data. Rerata dapat dirumuskan dengan:
$$\mu = \frac{1}{n} \sum_{i=1}^{n} x_i \tag{5.13}$$

2. **Median**, yaitu nilai tengah dari data yang diurutkan. Jika jumlah data ganjil, median adalah nilai tengah. Jika jumlah data genap, median adalah rata-rata dari dua nilai tengah.

3. **Mode**: Nilai yang paling sering muncul dalam data.

Ukuran penyebaran mengukur sejauh mana data tersebar atau bervariasi dari nilai pusatnya. Ini memberikan gambaran tentang tingkat keragaman atau dispersi dalam suatu dataset, yang membantu kita memahami kontribusi data dalam dataset. Ukuran penyebaran ini penting untuk memahami variasi dalam data dan untuk melakukan analisis lebih lanjut, seperti mendeteksi *outlier* atau menilai konsistensi data di sekitar nilai pusatnya. Ukuran penyebaran mencakup beberapa metrik penting, seperti:

1. **Range**: Selisih antara nilai maksimum dan minimum.
$$Range = x_{max} - x_{min} \tag{5.14}$$

2. **Variance (Varians)**: Rata-rata dari kuadrat selisih setiap nilai dengan mean.
$$\sigma^2 = \frac{1}{n} \sum_{i=1}^{n} (x_i - \mu)^2 \tag{5.15}$$

3. **Standard Deviation (Simpangan Baku)**: Akar kuadrat dari varians.
$$\sigma = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (x_i - \mu)^2} \tag{5.16}$$

### 5.4.2 Estimasi titik dan interval

Estimasi titik dan interval adalah dua pendekatan utama dalam statistik inferensial yang digunakan untuk menyimpulkan karakteristik populasi berdasarkan data sampel. Estimasi titik memberikan perkiraan tunggal dari parameter populasi, seperti rata-rata atau proporsi, yang dianggap sebagai representasi terbaik dari nilai sebenarnya. Namun, karena pengambilan sampel melibatkan ketidakpastian, estimasi titik tidak dapat memberikan gambaran lengkap tentang akurasi estimasi tersebut. Oleh karena itu, estimasi interval digunakan untuk melengkapi estimasi titik dengan menyediakan rentang nilai yang lebih realistis, yang mencerminkan ketidakpastian inheren dalam pengukuran tersebut. Interval ini, yang dikenal sebagai interval kepercayaan, mencakup kemungkinan lokasi parameter populasi dan dilengkapi dengan tingkat kepercayaan tertentu, seperti 95% atau 99%. Kombinasi dari kedua jenis estimasi ini memungkinkan kita untuk membuat inferensi yang lebih akurat dan terpercaya tentang populasi, sekaligus mengakomodasi variabilitas yang mungkin terjadi dalam proses sampling.

1. **Estimasi Titik** adalah nilai tunggal yang digunakan untuk memperkirakan parameter populasi. Estimasi Mean Populasi: Menggunakan mean sampel ($\bar{x}$) untuk mengestimasi mean populasi ($\mu$).
$$\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x^i \tag{5.17}$$

2. **Estimasi Interval** memberikan rentang nilai yang kemungkinan besar mengandung parameter populasi. Interval Kepercayaan: Rentang nilai yang berasosiasi dengan tingkat kepercayaan tertentu, biasanya 95%.
$$CI = \bar{x} \pm z \frac{\sigma}{\sqrt{n}} \tag{5.18}$$
dengan $z$ adalah nilai $z$ dari distribusi normal standar untuk tingkat kepercayaan yang diinginkan.

### 5.4.3 Uji hipotesis

Uji hipotesis adalah metode statistik yang digunakan untuk membuat keputusan atau menarik kesimpulan tentang suatu populasi berdasarkan data sampel (Raftery *et al.*, 1995). Proses ini melibatkan pengujian dugaan awal, yang dikenal sebagai hipotesis nol ($H_0$), terhadap alternatif yang mungkin, yaitu hipotesis alternatif ($H_a$). Uji hipotesis digunakan untuk menentukan apakah ada cukup bukti dalam data sampel untuk menolak hipotesis nol demi hipotesis alternatif. Melalui pendekatan ini, dapat diukur kemungkinan bahwa hasil yang diamati terjadi secara kebetulan, dengan mempertimbangkan tingkat signifikansi tertentu. Uji hipotesis memiliki aplikasi luas dalam berbagai bidang, mulai dari penelitian ilmiah hingga pengambilan keputusan bisnis, dan merupakan alat penting dalam memastikan validitas klaim atau pernyataan yang didasarkan pada data.

1. **Hipotesis Nol ($H_0$)**: Pernyataan awal yang diasumsikan benar.
2. **Hipotesis Alternatif ($H_a$)**: Pernyataan yang akan diterima jika bukti menunjukkan $H_0$ salah.

Langkah-langkah dalam uji hipotesis:
1. Formulasi hipotesis: tentukan $H_0$ dan $H_a$.
2. Pilih tingkat signifikansi ($\alpha$): Biasanya 0,05 atau 0,01.
3. Hitung statistik uji: misalnya, uji t untuk mean.
$$t = \frac{\bar{x} - \mu_0}{s/\sqrt{n}} \tag{5.19}$$
dengan $\bar{x}$ adalah rerata yang dihipotesiskan, $s$ adalah simpangan baku sampel, dan $n$ adalah ukuran sampel.
4. Tentukan nilai kritis atau p-value:
   a. p-value: probabilitas mendapatkan hasil yang sama atau lebih ekstrem dari yang diamati, jika $H_0$ benar.
   b. Nilai kritis: nilai batas dari distribusi uji statistik untuk tingkat signifikansi yang dipilih.
5. Buat Keputusan:
   a. Jika p-value $\leq \alpha$, tolak $H_0$.
   b. Jika p-value $> \alpha$, gagal menolak $H_0$.

Contoh uji hipotesis:
1. Uji t satu sampel: menguji apakah mean sampel berbeda dari mean populasi yang diketahui.
2. Uji t dua sampel: menguji apakah mean dari dua sampel berbeda secara signifikan.
3. Uji Chi-square: menguji independensi antara dua variabel kategori atau kesesuaian distribusi sampel dengan distribusi yang diharapkan.

## 5.5 Regresi dan Korelasi

Regresi dan korelasi adalah teknik statistik penting dalam analisis data dan kecerdasan buatan. Regresi memungkinkan prediksi nilai variabel dependen berdasarkan variabel independen, sementara korelasi mengukur kekuatan dan arah hubungan antara dua variabel (Miles dan Shevlin, 2000). Evaluasi model regresi dengan metrik yang tepat membantu memastikan model yang dikembangkan akurat dan andal.

### 5.5.1 Regresi linier sederhana

Regresi linear sederhana digunakan untuk memodelkan hubungan antara dua variabel, satu variabel independen (X) dan satu variabel dependen (Y). Model ini mengasumsikan bahwa hubungan antara X dan Y dapat dijelaskan dengan garis lurus:
$$Y = \beta_0 + \beta_1 X + \epsilon \tag{5.20}$$
dengan:
* $Y$ adalah variabel dependen.
* $X$ adalah variabel independen.
* $\beta_0$ adalah intercept (titik potong sumbu-Y).
* $\beta_1$ adalah slope (kemiringan garis).
* $\epsilon$ adalah error term.

Tujuan regresi linear adalah untuk menemukan estimasi $\beta_0$ dan $\beta_1$ yang meminimalkan jumlah kuadrat error (*residual sum of squares*, RSS):
$$RSS = \sum_{i=1}^{n} (y_i - (\beta_0 + \beta_1 x_i))^2 \tag{5.21}$$
Metode kuadrat terkecil (*Least Squares*) biasanya digunakan untuk menemukan estimasi parameter ini.

### 5.5.2 Regresi linier berganda

Regresi linear berganda memperluas konsep regresi linear sederhana untuk mencakup lebih dari satu variabel independen:
$$Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_p X_p + \epsilon \tag{5.22}$$
dengan:
* $Y$ adalah variabel dependen.
* $X1, X2, \dots, Xp$ adalah variabel independen.
* $\beta_0$ adalah intercept.
* $\beta_1, \beta_2, \dots, \beta_p$ adalah koefisien regresi untuk masing-masing variabel independen.
* $\epsilon$ adalah error term.

Estimasi parameter dalam regresi linear berganda juga dilakukan menggunakan metode *Least Squares*, dengan tujuan meminimalkan RSS.

### 5.5.3 Analisis korelasi

Korelasi mengukur kekuatan dan arah hubungan linier antara dua variabel. Korelasi tidak menyiratkan sebab akibat tetapi menunjukkan adanya hubungan antara variabel.

1. **Koefisien Korelasi Pearson (r)** mengukur kekuatan dan arah hubungan linier antara dua variabel kontinu:
$$r = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^n (x_i - \bar{x})^2 \sum_{i=1}^n (y_i - \bar{y})^2}} \tag{5.23}$$
Nilai $r$ berkisar antara -1 dan 1, dengan ketentuan:
* $r = 1$ menunjukkan hubungan linier positif sempurna.
* $r = -1$ menunjukkan hubungan linier negatif sempurna.
* $r = 0$ menunjukkan tidak ada hubungan linier.

2. **Koefisien Korelasi Spearman ($\rho$)** mengukur kekuatan dan arah hubungan monotonik antara dua variabel ordinal atau tidak berdistribusi normal. Korelasi Spearman adalah korelasi Pearson yang dihitung pada peringkat data.

### 5.5.4 Pengujian model statistik

Evaluasi model statistik penting untuk menentukan seberapa baik model memprediksi atau menjelaskan data. Beberapa metrik evaluasi untuk model regresi meliputi:

1. **Koefisien Determinasi ($R^2$):** Mengukur proporsi variabilitas dalam variabel dependen yang dapat dijelaskan oleh variabel independen dalam model. Nilai $R^2$ berkisar antara 0 dan 1.
$$R^2 = 1 - \frac{RSS}{TSS} \tag{5.24}$$

2. **Adjusted $R^2$:** Mengoreksi $R^2$ untuk jumlah variabel dalam model, memberikan ukuran yang lebih adil untuk model dengan sejumlah besar variabel independen.

3. **Mean Absolute Error (MAE):** Rata-rata dari absolut nilai error.
$$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i| \tag{5.25}$$

4. **Mean Squared Error (MSE):** Rata-rata dari kuadrat nilai error.
$$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 \tag{5.26}$$

5. **Root Mean Squared Error (RMSE):** Akar kuadrat dari MSE.
$$RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2} \tag{5.27}$$

## 5.6 Teori Keputusan dan Metode Nonparametrik

Teori keputusan dan metode nonparametrik merupakan alat penting dalam analisis statistik dan kecerdasan buatan. Pohon keputusan dan algoritma Naive Bayes menyediakan cara untuk membuat keputusan berdasarkan data dan probabilitas, sementara uji nonparametrik memberikan fleksibilitas dalam analisis data yang tidak memenuhi asumsi distribusi tertentu (Berger, 2013). Analisis keputusan memungkinkan evaluasi tindakan dalam situasi ketidakpastian, membantu pengambilan keputusan yang lebih informasional.

### 5.6.1 Pohon Keputusan

Pohon keputusan adalah model prediktif yang digunakan untuk membuat keputusan berdasarkan data. Model ini membagi data ke dalam subset yang semakin kecil berdasarkan aturan keputusan yang berulang hingga mencapai keputusan akhir.

1. **Komponen Pohon Keputusan:**
   * **Akar:** Node awal yang mewakili seluruh dataset.
   * **Cabang:** Jalur dari satu node ke node berikutnya.
   * **Node Internal:** Node yang merepresentasikan tes atau keputusan berdasarkan satu atribut.
   * **Daun (*Leaf*):** Node terminal yang memberikan hasil akhir atau keputusan.

2. **Algoritma Pembentukan Pohon Keputusan:**
   * **ID3** (*Iterative Dichotomiser 3*): Menggunakan entropi dan informasi gain untuk memilih atribut yang membagi data dengan optimal.
   * **C4.5:** Perbaikan dari ID3 yang menangani atribut kontinu dan missing values.
   * **CART** (*Classification and Regression Trees*): Menghasilkan pohon keputusan biner untuk klasifikasi dan regresi.

3. **Kriteria Pemilihan Atribut:**
   * **Informasi Gain:** Mengukur pengurangan ketidakpastian atau entropi setelah membagi dataset.
   * **Gini Index:** Mengukur *impurity* dari node.
   * **Chi-Square:** Mengukur asosiasi antara atribut dan kelas.

### 5.6.2 Algoritma Naive Bayes

Algoritma Naive Bayes adalah metode klasifikasi yang sederhana namun sangat efektif, yang didasarkan pada teorema Bayes dengan asumsi independensi yang kuat antara fitur-fitur dalam data. Meskipun asumsi independensi ini jarang terpenuhi sepenuhnya dalam praktik, algoritma ini tetap menunjukkan kinerja yang baik dalam berbagai aplikasi, seperti pengenalan teks, klasifikasi email spam, dan analisis sentimen. Naive Bayes bekerja dengan menghitung probabilitas dari setiap kelas berdasarkan fitur-fitur yang diberikan, kemudian memilih kelas dengan probabilitas tertinggi sebagai prediksi. Kesederhanaan dan efisiensi dari Naive Bayes membuatnya menjadi pilihan populer dalam situasi ketika kecepatan dan interpretabilitas lebih diutamakan daripada kompleksitas model. Meskipun sederhana, Naive Bayes sering memberikan hasil yang kompetitif dibandingkan dengan algoritma klasifikasi yang lebih kompleks, terutama pada dataset yang besar dan berdimensi tinggi.

1. **Teorema Bayes:**
$$P(C|X) = \frac{P(X|C) \cdot P(C)}{P(X)} \tag{5.28}$$
dengan:
   * $P(C|X)$ adalah probabilitas kelas C pada atribut X.
   * $P(X|C)$ adalah probabilitas atribut X pada kelas C.
   * $P(C)$ adalah probabilitas awal kelas C.
   * $P(X)$ adalah probabilitas atribut X.

2. **Dalam algoritma Naive Bayes terdapat asumsi idependensi, yang dapat dihitung menggunakan persamaan:**
$$P(X|C) = \prod_{i=1}^n P(x_i|C) \tag{5.29}$$
dengan $x_i$ adalah fitur individual.

3. **Algoritma Naive Bayes memiliki beberapa keunggulan, yaitu sebagai berikut:**
   * Cepat dan efisien.
   * Cocok untuk dataset besar.
   * Menghasilkan performa yang baik meskipun asumsi independensi tidak sepenuhnya terpenuhi.

### 5.6.3 Uji Nonparametrik

Uji nonparametrik adalah metode statistik yang digunakan ketika asumsi-asumsi dasar uji parametrik, seperti distribusi normal atau homoskedastisitas, tidak terpenuhi oleh data. Uji nonparametrik tidak memerlukan asumsi tentang distribusi populasi dari mana sampel diambil, sehingga lebih fleksibel dalam penerapannya, terutama ketika berhadapan dengan data yang berskala ordinal, memiliki *outlier*, atau sampel berukuran kecil. Metode ini meliputi berbagai uji seperti uji Wilcoxon, uji Mann-Whitney, dan uji Kruskal-Wallis, yang semuanya dirancang untuk menguji hipotesis tentang median, distribusi, atau peringkat data tanpa mengandalkan parameter populasi tertentu. Karena sifatnya yang lebih umum, uji nonparametrik sering digunakan dalam analisis data yang tidak memenuhi syarat untuk uji parametrik, memberikan alternatif yang robust dan reliabel untuk menguji hipotesis dan membuat simpulan statistik dalam berbagai konteks penelitian.

1. **Uji Mann-Whitney:**
   * Digunakan untuk membandingkan dua sampel independen.
   * Hipotesis nol: Tidak ada perbedaan antara dua distribusi populasi.
   * Uji statistik:
$$U = n_1 n_2 + \frac{n_1(n_1 + 1)}{2} - R_1 \tag{5.30}$$
dengan $n_1$ dan $n_2$ adalah ukuran sampel, dan $R_1$ adalah jumlah peringkat dalam sampel pertama.

2. **Uji Kruskal-Wallis:**
   * Perluasan dari uji Mann-Whitney untuk lebih dari dua grup.
   * Hipotesis nol: Semua grup berasal dari distribusi yang sama.
   * Uji statistik:
$$H = \frac{12}{N(N + 1)} \sum_{i=1}^k \frac{R_i^2}{n_1} - 3(N + 1) \tag{5.31}$$
dengan $N$ adalah total ukuran sampel, $k$ adalah jumlah grup, $R_i$ adalah jumlah peringkat dalam grup ke-i, dan $n_i$ adalah ukuran grup ke-i.

### 5.6.4 Analisis Keputusan

Analisis keputusan adalah proses sistematis yang digunakan untuk membuat pilihan yang optimal di antara berbagai alternatif berdasarkan data dan informasi yang tersedia. Proses ini melibatkan penilaian risiko, manfaat, dan kemungkinan hasil dari setiap opsi, serta mempertimbangkan faktor-faktor ketidakpastian yang dapat mempengaruhi keputusan. Dalam analisis keputusan, berbagai alat dan teknik, seperti pohon keputusan, analisis biaya-manfaat, dan analisis sensitivitas, digunakan untuk menyusun dan mengevaluasi pilihan secara kuantitatif maupun kualitatif.

Tujuan dari analisis ini adalah untuk membantu pengambil keputusan memilih tindakan yang paling sesuai dengan tujuan yang ingin dicapai, baik itu dalam konteks bisnis, manajemen, kebijakan publik, atau bidang lainnya. Dengan pendekatan yang terstruktur, analisis keputusan memungkinkan para pengambil keputusan untuk mengevaluasi trade-off yang ada, mengidentifikasi potensi risiko, dan memilih strategi yang memaksimalkan nilai atau utilitas dalam situasi yang kompleks dan sering kali penuh ketidakpastian.

1. **Beberapa aspek terkait matriks keputusan:**
   * Merupakan tabel yang menampilkan hasil dari berbagai tindakan dalam situasi yang berbeda.
   * Setiap entri dalam matriks mewakili nilai hasil (*payoff*) untuk kombinasi tindakan dan keadaan.

2. **Beberapa aspek terkait pohon keputusan:**
   * Digunakan untuk memvisualisasikan keputusan dan hasil yang mungkin dalam bentuk pohon.
   * Setiap cabang mewakili tindakan atau keputusan, dan setiap daun mewakili hasil atau *payoff*.

3. **Beberapa aspek terkait kriteria keputusan:**
   * **Maximin:** memilih tindakan dengan nilai minimum terbesar.
   * **Maximax:** memilih tindakan dengan nilai maksimum terbesar.
   * **Hurwicz:** kombinasi dari maximin dan maximax, dengan parameter optimisme (konsep bias optimisme, yang terkait dengan evaluasi model).
   * **Laplace:** mengasumsikan semua kejadian sama kemungkinan, memilih tindakan dengan nilai ekspektasi tertinggi.
   * **Minimax Regret:** meminimalkan penyesalan maksimum dari tidak memilih tindakan terbaik.

## 5.7 Analisis Komponen Utama dan Reduksi Dimensi

*Principal Component Analysis* (PCA) dan *Linear Discriminant Analysis* (LDA) adalah teknik utama dalam reduksi dimensi, masing-masing dengan aplikasi dan keunggulan yang spesifik. PCA lebih umum digunakan untuk mereduksi dimensi data kontinu dengan mempertahankan variabilitas, sedangkan LDA lebih cocok untuk tugas klasifikasi dengan memaksimalkan separasi antar kelas (Wu, 2014). Keduanya membantu dalam menyederhanakan data dan model, meningkatkan interpretabilitas, dan memperbaiki performa analisis data dan kecerdasan buatan.

### 5.7.1 Konsep Dasar Reduksi Dimensi

Reduksi dimensi adalah proses mengurangi jumlah variabel acak yang dipertimbangkan dengan mendapatkan sekumpulan variabel utama yang lebih kecil. Ini bertujuan untuk menyederhanakan model tanpa kehilangan informasi signifikan. Beberapa manfaat dari reduksi dimensi meliputi:

* Mengurangi kompleksitas model, membuatnya lebih mudah dipahami.
* Mengurangi risiko *overfitting*.
* Mempercepat waktu pemrosesan dan analisis data.
* Menghilangkan noise dari data.

### 5.7.2 Analisis Komponen Utama (PCA)

Analisis Komponen Utama (PCA) adalah teknik statistik yang digunakan untuk mereduksi dimensi data sambil mempertahankan sebanyak mungkin variabilitas dalam dataset. PCA mengubah data ke dalam satu set baru dari variabel yang tidak berkorelasi yang disebut komponen utama.

**Langkah-langkah PCA:**
1. **Standarisasi Data:** Skala variabel sehingga masing-masing memiliki mean 0 dan varians 1.
2. **Kalkulasi Matriks Kovarians:** Menghitung kovarians antar variabel dalam data.
$$Cov(X, Y) = \frac{1}{n - 1} \sum_{i=1}^n (X_i - \bar{X})(Y_i - \bar{Y}) \tag{5.32}$$
3. **Eigen decomposition:** Menghitung eigen vektor dan eigen value dari matriks kovarians.
   * Eigen vektor: Arah dari komponen utama.
   * Eigen value: Besaran variabilitas sepanjang masing-masing eigen vektor.
4. **Pilih Komponen Utama:** Urutkan eigen vektor berdasarkan eigen value dan pilih komponen utama dengan eigen value terbesar.
5. **Transformasi Data:** Proyeksikan data asli ke ruang baru yang dibentuk oleh komponen utama yang dipilih.
**Rumus Transformasi PCA: $Z = XW$**
dengan $X$ adalah data asli, $W$ adalah matriks eigen vektor, dan $Z$ adalah data yang ditransformasikan.

### 5.7.3 Pemilihan Jumlah Komponen Utama

Menentukan jumlah komponen utama yang optimal adalah langkah kritis dalam PCA. Beberapa metode yang digunakan meliputi:
1. **Screen Plot:** *plotting eigenvalue* dari setiap komponen utama dan mencari “*elbow point*” di mana penurunan eigenvalue mulai melambat.
2. **Kaiser’s Criterion:** memilih komponen utama dengan eigenvalue lebih besar dari 1.
3. **Cumulative Explained Variance:** memilih jumlah komponen yang menjelaskan persentase tertentu dari variabilitas total (misalnya, 90-95%).

### 5.7.4 Linear Discriminant Analysis (LDA)

*Linear Discriminant Analysis* (LDA) adalah teknik reduksi dimensi yang mencari kombinasi linier dari fitur yang memaksimalkan separasi antara dua atau lebih kelas. LDA sering digunakan untuk klasifikasi dan memperbaiki performa model dengan mengurangi noise.

**Langkah-langkah LDA:**
1. **Standarisasi Data:** seperti PCA, data harus distandarisasi.
2. **Hitung Mean dan Scatter Matrix:** hitung mean dari masing-masing kelas dan scatter matrix antar kelas.
   * *Within-class scatter matrix* ($S_W$): menunjukkan penyebaran dalam kelas.
$$S_W = \sum_{i=1}^c S_i \tag{5.33}$$
dengan $S_i$ adalah scatter matrix untuk kelas ke-i.
   * *Between-class scatter matrix* ($S_B$): menunjukkan penyebaran antar kelas.
$$S_B = \sum_{i=1}^c n_i (\mu_i - \mu)(\mu_i - \mu)^T \tag{5.34}$$
dengan $n_i$ adalah jumlah sampel dalam kelas ke-i, $\mu_i$ adalah mean kelas ke-i, dan $\mu$ adalah mean keseluruhan.
3. **Eigen decomposition:** Menghitung eigen vektor dan eigen value dari matriks $S_W^{-1} S_B$.
4. **Pilih Komponen Diskriminan:** pilih eigen vektor dengan eigen value terbesar yang memaksimalkan separasi antar kelas.
5. **Transformasi Data:** proyeksikan data asli ke ruang baru yang dibentuk oleh komponen diskriminan yang dipilih.

### 5.7.5 Penggunaan Reduksi Dimensi dalam Kecerdasan Buatan

Reduksi dimensi memainkan peran penting dalam kecerdasan buatan dan machine learning. Beberapa aplikasi utamanya meliputi:
1. **Pra-pemrosesan Data:** mengurangi dimensi data sebelum melatih model kecerdasan buatan untuk meningkatkan efisiensi dan performa.
2. **Visualisasi Data:** mengurangi data ke dalam 2 atau 3 dimensi untuk memudahkan visualisasi dan interpretasi.
3. **Noise Filtering:** menghilangkan variabel yang tidak relevan atau noise dari data, yang dapat meningkatkan akurasi model.

## 5.8 Metode Monte Carlo dan Simulasi

Metode Monte Carlo adalah alat yang sangat kuat dalam analisis data dan kecerdasan buatan, memberikan cara fleksibel dan efisien untuk menangani masalah kompleks yang melibatkan ketidakpastian dan variabilitas. Monte Carlo merupakan sekelompok teknik komputasi yang menggunakan *sampling* acak untuk memperoleh solusi numerik dari masalah matematika atau fisika yang kompleks (Rubinstein dan Kroese, 2016). Metode ini dinamai berdasarkan kasino Monte Carlo di Monako, yang dikenal karena permainan peluangnya, karena elemen acak dalam teknik ini mirip dengan permainan di kasino. Simulasi Monte Carlo merupakan penerapan dari metode Monte Carlo ketika teknik ini digunakan untuk memodelkan probabilitas dari berbagai hasil dalam proses yang tidak dapat ditentukan secara deterministik, terutama ketika menghadapi ketidakpastian. Metode ini sering digunakan dalam masalah yang sulit atau tidak mungkin dipecahkan secara analitis.

### 5.8.1 Sejarah dan Aplikasi

Metode ini pertama kali dikembangkan selama Perang Dunia II oleh matematikawan seperti Stanislaw Ulam dan John von Neumann sebagai alat untuk memecahkan masalah kompleks dalam fisika nuklir. Nama “Monte Carlo” diambil dari kasino Monte Carlo di Monako, karena metode ini bergantung pada prinsip acak yang mirip dengan permainan peluang. Sejak saat itu, teknik ini telah berkembang pesat dan digunakan dalam berbagai bidang, termasuk keuangan, teknik, ilmu komputer, dan ilmu data. Dalam keuangan, Monte Carlo digunakan untuk menilai risiko portofolio dan harga opsi; dalam teknik, untuk memodelkan sistem yang kompleks dan mengoptimalkan desain; dan dalam ilmu komputer, untuk algoritma seperti simulasi *annealing* dan pencarian pohon keputusan. Aplikasi metode Monte Carlo yang luas dan kemampuannya untuk menangani ketidakpastian dan kompleksitas menjadikannya alat yang sangat berharga dalam analisis dan pengambilan keputusan berbasis data.

**Aplikasi Metode Monte Carlo:**
1. Fisika dan Kimia: memodelkan perilaku partikel, dinamika molekul.
2. Keuangan: menilai risiko dan harga opsi, simulasi portofolio.
3. Teknik: analisis keandalan, optimasi desain.
4. Ilmu Komputer: algoritma optimasi, pemrosesan citra.
5. Statistika: estimasi integral, pengujian hipotesis.

### 5.8.2 Prinsip Dasar Metode Monte Carlo

Prinsip dasar metode Monte Carlo berfokus pada penggunaan sampling acak untuk menyelesaikan masalah matematis dan statistik yang kompleks, terutama ketika metode analitik konvensional tidak memadai. Metode ini didasarkan pada konsep bahwa bahwa, dengan melakukan simulasi acak dalam jumlah besar, dapat mendekati solusi dari suatu masalah atau menghitung estimasi probabilitas dengan akurasi yang memadai. Proses ini melibatkan tiga langkah utama: (1) pembuatan sampel acak dari distribusi yang relevan, (2) penerapan fungsi atau model yang ingin dievaluasi pada sampel tersebut, dan (3) analisis hasil untuk memperoleh estimasi atau prediksi. Dengan memanfaatkan hukum bilangan besar, metode Monte Carlo memungkinkan untuk memperoleh hasil yang mendekati solusi yang tepat melalui repetisi simulasi, meskipun data yang digunakan bersifat acak dan tidak deterministik. Prinsip ini membuat Monte Carlo sangat berguna dalam berbagai aplikasi, dari perhitungan risiko finansial hingga simulasi sistem fisik yang kompleks.

**Langkah-langkah umum dalam metode Monte Carlo:**
1. Definisikan Model: Tentukan fungsi atau proses yang akan disimulasikan.
2. Generate Sampel Acak: Buat sampel acak dari distribusi yang relevan.
3. Hitung Hasil: Gunakan sampel untuk menghitung hasil.
4. Analisis Hasil: Evaluasi hasil simulasi untuk mendapatkan estimasi.

Contoh sederhana adalah memperkirakan nilai $\pi$ menggunakan titik acak dalam lingkaran yang terinskripsi dalam bujur sangkar:
1. Generate titik acak dalam bujur sangkar.
2. Hitung proporsi titik yang jatuh dalam lingkaran.
3. Gunakan proporsi ini untuk memperkirakan $\pi$.

### 5.8.3 Simulasi Monte Carlo dalam Statistik

Dalam konteks statistik, simulasi Monte Carlo memungkinkan analisis yang mendalam terhadap model statistik yang kompleks, ketika solusi analitik mungkin sulit atau tidak mungkin dicapai. Proses ini melibatkan pembuatan banyak sampel acak dari distribusi yang diharapkan, penerapan model atau fungsi statistik pada sampel tersebut, dan kemudian menganalisis hasilnya untuk memahami distribusi kemungkinan hasil dan estimasi parameter. Dengan cara ini, dapat dievaluasi ketidakpastian, memodelkan berbagai skenario, dan memperoleh interval kepercayaan yang lebih akurat. Simulasi Monte Carlo juga berguna dalam memperkirakan p-value, menganalisis distribusi hasil dari uji hipotesis, dan mengatasi masalah kompleks seperti estimasi variansi dalam model non-linier. Teknik ini sangat berharga dalam statistik karena kemampuannya untuk menangani masalah yang melibatkan ketidakpastian dan variabilitas dengan cara yang fleksibel dan dapat diterapkan pada berbagai jenis data dan model. Beberapa contoh simulasi Monte Carlo dalam statistik:

1. **Estimasi Integral:** menghitung integral yang kompleks dengan sampling acak.
$$I = \int_a^b f(x)dx \approx \frac{b-a}{n} \sum_{i=1}^n f(x_i) \tag{5.35}$$
dengan $x_i$ adalah sampel acak dalam interval [a, b]

2. **Pengujian Hipotesis:** menguji hipotesis dengan simulasi distribusi dari statistik uji.
3. **Estimasi Interval Kepercayaan:** menyediakan estimasi interval kepercayaan dengan bootstrap atau resampling.

### 5.8.4 Metode Monte Carlo Markov Chain (MCMC)

Metode Monte Carlo Markov Chain (MCMC) adalah teknik lanjutan dalam statistika dan pemodelan probabilistik yang digunakan untuk menghasilkan sampel dari distribusi probabilitas kompleks dan tidak terdistribusi dengan cara yang sulit dihitung secara langsung. MCMC menggabungkan prinsip Monte Carlo dengan proses Markov untuk memungkinkan eksplorasi sistem yang memiliki dimensi tinggi dan kompleksitas tinggi (Carlo, 2004). Teknik ini beroperasi dengan membangun rantai Markov, ketika setiap langkah bergantung pada langkah sebelumnya, dan menghasilkan sampel yang mendekati distribusi target setelah periode iterasi yang memadai. MCMC sangat berguna dalam situasi ketika distribusi posterior sulit untuk diperoleh secara analitik atau ketika data memiliki struktur yang kompleks. Dengan menerapkan algoritma seperti Metropolis-Hastings atau Gibbs Sampling, MCMC memungkinkan para peneliti untuk memodelkan dan menganalisis data dalam berbagai aplikasi, mulai dari inferensi bayesian hingga estimasi parameter dalam model statistik yang rumit. Teknik ini menjadi alat yang sangat penting dalam statistika modern karena kemampuannya untuk menangani berbagai bentuk distribusi dan memberikan solusi yang akurat dalam masalah inferensi statistik yang kompleks.

**Langkah-langkah umum dalam MCMC:**
1. Inisialisasi: Pilih titik awal.
2. Proposal: Generate titik proposal baru dari distribusi proposal.
3. *Acceptance/Rejection*: Tentukan apakah titik proposal diterima atau ditolak berdasarkan kriteria probabilitas.

**Algoritma MCMC yang populer:**
1. **Metropolis-Hastings:** Menggunakan distribusi proposal untuk menghasilkan titik baru dan memutuskan penerimaan berdasarkan rasio probabilitas.
2. **Gibbs Sampling:** Menyampel setiap variabel secara bergantian dari distribusi kondisionalnya.

### 5.8.5 Implementasi Metode Monte Carlo dalam Kecerdasan Buatan

Metode Monte Carlo telah menjadi alat yang sangat berharga dalam kecerdasan buatan (AI) karena kemampuannya untuk menangani ketidakpastian dan kompleksitas dalam model dan data. Dalam konteks AI, metode ini digunakan untuk memodelkan dan mengevaluasi berbagai skenario dengan menghasilkan sampel acak dari distribusi yang relevan dan menganalisis hasilnya. Metode Monte Carlo membantu dalam berbagai aplikasi AI, seperti pengoptimalan algoritma, estimasi nilai fungsi objektif dalam algoritma pembelajaran mesin, dan pengembangan strategi dalam pengambilan keputusan otomatis. Teknik ini memungkinkan para praktisi AI untuk mengeksplorasi ruang solusi yang luas dan menangani masalah yang melibatkan variabilitas dan ketidakpastian data dengan lebih efektif. Dengan menggunakan simulasi Monte Carlo, AI dapat memprediksi hasil yang lebih akurat, mengidentifikasi pola tersembunyi dalam data, dan mengembangkan model yang lebih robust dalam menghadapi situasi yang kompleks dan dinamis. Metode Monte Carlo digunakan dalam berbagai aplikasi kecerdasan buatan:

1. **Optimasi dan Pencarian:**
   * **Simulated Annealing:** algoritma optimasi yang menggunakan Metode Monte Carlo untuk menemukan minimum global dari fungsi objektif.
   * **Randomized Search:** pencarian hiperparameter model dengan sampling acak.
2. **Pembelajaran Penguatan (*reinforcement learning*):**
   * **Monte Carlo Tree Search (MCTS):** digunakan dalam game AI untuk memodelkan kemungkinan langkah dan memilih langkah optimal.
   * **Estimasi Nilai:** menggunakan sampel acak untuk memperkirakan nilai fungsi dalam pembelajaran penguatan.
3. **Model Generatif:**
   * **Bayesian Networks:** estimasi parameter dan inferensi dalam jaringan Bayesian dengan sampling Monte Carlo.

### 5.8.6 Contoh Penerapan: Simulasi Portofolio Keuangan

Simulasi Monte Carlo sering digunakan dalam keuangan untuk memodelkan risiko dan nilai masa depan portofolio investasi.

**Langkah-langkah simulasi portofolio:**
1. **Model Return:** tentukan model return dari aset, misalnya, distribusi normal.
2. **Generate Sampel:** generate return harian atau bulanan dari distribusi yang dipilih.
3. **Kalkulasi Nilai Portofolio:** gunakan return sampel untuk menghitung nilai portofolio di masa depan.
4. **Analisis Hasil:** Evaluasi hasil simulasi untuk mengukur risiko (VaR, CVaR) dan performa portofolio.

## 5.9 Penerapan Statistik dan Probabilitas dalam Kecerdasan Buatan

### 5.9.1 Pengantar

Penerapan statistik dan probabilitas dalam kecerdasan buatan adalah aspek penting dalam membangun model yang dapat belajar dari data dan membuat keputusan berdasarkan informasi yang tersedia. Statistik menyediakan alat untuk menganalisis data dan menarik kesimpulan, sementara probabilitas memberikan kerangka kerja untuk menangani ketidakpastian dan membuat prediksi. Penerapan statistik dan probabilitas dalam kecerdasan buatan adalah fundamental untuk mengembangkan model yang andal dan akurat. Teknik-teknik yang sudah dibahas sebelumnya membantu dalam memodelkan ketidakpastian, membuat prediksi, dan mengevaluasi kinerja model, sehingga memungkinkan pengambilan keputusan yang lebih baik berdasarkan data.

### 5.9.2 Pembelajaran Mesin (*Machine Learning*)

Pembelajaran mesin adalah cabang AI yang berfokus pada pengembangan algoritma yang memungkinkan komputer untuk belajar dari data dan membuat prediksi atau keputusan berdasarkan data. Statistik dan probabilitas memainkan peran kunci dalam banyak aspek pembelajaran mesin, termasuk:

1. **Regresi:** menggunakan metode statistik untuk memodelkan hubungan antara variabel.
2. **Klasifikasi:** menggunakan algoritma probabilistik untuk mengklasifikasikan data ke dalam kategori.
3. **Clustering:** mengelompokkan data ke dalam grup berdasarkan karakteristik yang mirip.

Beberapa metode populer dalam pembelajaran mesin yang menggunakan statistik dan probabilitas meliputi:
1. Regresi Linear dan Logistik: model statistik yang digunakan untuk prediksi dan klasifikasi.
2. k-Nearest Neighbors (KNN): algoritma non-parametrik yang menggunakan probabilitas berdasarkan kedekatan.
3. *Support Vector Machines* (SVM): menggunakan konsep statistik untuk klasifikasi dan regresi.
4. *Decision Trees* dan *Random Forests*: menggunakan probabilitas untuk membuat keputusan berdasarkan data.

### 5.9.3 Model Probabilistik

Model probabilistik digunakan untuk menangani ketidakpastian dalam data dan membuat prediksi yang lebih akurat. Beberapa model probabilistik yang umum digunakan dalam kecerdasan buatan meliputi:

1. **Naive Bayes:** algoritma klasifikasi yang didasarkan pada Teorema Bayes dengan asumsi independensi antara fitur.
$$P(C|X) = \frac{P(X|C) \cdot P(C)}{P(X)} \tag{5.36}$$
2. **Hidden Markov Models (HMMs):** model probabilistik yang digunakan untuk memodelkan proses yang menghasilkan urutan data, seperti pengenalan ucapan dan teks.
$$P(X, Z) = P(Z_1) \prod_{t=2}^T P(Z_t | Z_{t-1}) \prod_{t=1}^T P(X_t | Z_t) \tag{5.37}$$
3. **Gaussian Mixture Models (GMMs):** model yang mengasumsikan data berasal dari beberapa distribusi Gaussian.
$$P(X) = \sum_{k=1}^K \pi_k \mathcal{N}(X | \mu_k, \Sigma_k) \tag{5.38}$$
dengan $\mu_k$ adalah berat/bobot dari komponen ke-k, dan $\mathcal{N}(X | \mu_k, \Sigma_k)$ adalah distribusi Gaussian dengan rerata $\mu_k$ dan kovarians $\Sigma_k$.

### 5.9.4 Deep Learning

*Deep learning* adalah subset dari pembelajaran mesin yang berfokus pada jaringan saraf tiruan dengan banyak lapisan (*deep neural networks*). Statistik dan probabilitas digunakan dalam beberapa aspek *deep learning*, termasuk:

1. **Regularisasi:** teknik statistik untuk menghindari *overfitting* dengan menambahkan penalti pada kompleksitas model.
$$Loss_{regularized} = Loss + \lambda \cdot Regularization \ Term$$
2. **Optimisasi:** algoritma probabilistik seperti *Stochastic Gradient Descent* (SGD) digunakan untuk menemukan parameter model yang optimal.
$$\theta = \theta - \eta \nabla_\theta J(\theta) \tag{5.39}$$
dengan $\theta$ adalah parameter model, $\eta$ adalah *learning rate*, dan $J(\theta)$ adalah fungsi loss.
3. **Bayesian Neural Networks:** memperkenalkan ketidakpastian ke dalam model jaringan saraf dengan memodelkan bobot sebagai distribusi probabilistik.
$$P(W|D) = \frac{P(D|W)P(W)}{P(D)} \tag{5.40}$$

### 5.9.5 Inferensi Bayes

Inferensi Bayes adalah metode untuk memperbarui probabilitas hipotesis seiring dengan tersedianya bukti baru. Dalam AI, inferensi Bayes digunakan untuk:

1. Pembelajaran Parameter: memperkirakan distribusi parameter model dari data yang diamati.
2. *Decision Making*: membuat keputusan optimal berdasarkan distribusi probabilitas yang diperbarui.
3. Bayesian Networks: struktur grafis yang merepresentasikan hubungan probabilistik antara variabel.

### 5.9.6 Evaluasi Model

Evaluasi model adalah langkah penting dalam pembelajaran mesin dan kecerdasan buatan untuk memastikan bahwa model yang dibangun memiliki performa yang baik. Statistik dan probabilitas digunakan dalam berbagai metrik evaluasi, seperti:

1. **Akurasi:** proporsi prediksi benar dari total prediksi.
$$Akurasi = \frac{True \ Positives + True \ Negatives}{Total \ Predictions} \tag{5.41}$$
2. **Precision dan Recall:** mengukur akurasi dari prediksi positif.
$$Precision = \frac{True \ Positives}{True \ Positives + False \ Positives} \tag{5.42}$$
$$Recall = \frac{True \ Positives}{True \ Positives + False \ Negatives} \tag{5.43}$$
3. **F1 Score:** *harmonic mean* dari *precision* dan *recall*.
$$F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall} \tag{5.44}$$
4. **Area Under Curve (AUC-ROC):** Mengukur kemampuan model untuk membedakan antara kelas.
$$AUC = \int_0^1 TPR(FPR^{-1}(x))dx \tag{5.45}$$
dengan TPR adalah *True Positive Rate* dan FPR adalah *False Positive Rate*.

### 5.9.7 Studi Kasus: Penerapan dalam Analisis Sentimen

Analisis sentimen adalah aplikasi kecerdasan buatan yang menggunakan statistik dan probabilitas untuk menentukan sentimen (positif, negatif, netral) dalam teks. Langkah-langkah umumnya meliputi:

1. **Preprocessing Data:** membersihkan teks dari noise seperti *stop words* dan tokenisasi.
2. **Ekstraksi Fitur:** menggunakan metode statistik seperti *Term Frequency-Inverse Document Frequency* (TF-IDF) untuk representasi teks.
$$TF-IDF(t, d) = tf(t, d) \cdot \log \frac{N}{df(t)} \tag{5.46}$$
dengan $tf(t, d)$ adalah *frekuency term* t dalam dokumen d, N adalah total dokumen, dan $df(t)$ adalah jumlah dokumen yang mengandung term tt.
3. **Modeling:** menggunakan algoritma seperti Naive Bayes atau *Support Vector Machines* (SVM) untuk klasifikasi sentimen.
4. **Evaluasi Model:** menggunakan metrik seperti akurasi, *precision*, *recall*, dan *F1 score* untuk mengevaluasi performa model.

Pada bab ini telah dibahas konsep dasar statistik dan probabilitas serta penerapannya dalam kecerdasan buatan (AI). Dari pendahuluan konsep dasar probabilitas hingga metode canggih seperti simulasi Monte Carlo dan Metode Monte Carlo Markov Chain (MCMC), telah dieksplorasi bagaimana alat-alat ini digunakan untuk menangani ketidakpastian dan memodelkan berbagai fenomena kompleks dalam dunia kecerdasan buatan, lebih lanjut lagi memberikan prediksi. Dalam banyak hal, statistik menyediakan fondasi untuk pengambilan keputusan berbasis data, sementara probabilitas memungkinkan untuk mengukur dan menganalisis data. Teknik-teknik seperti regresi, model probabilistik, dan metode Monte Carlo menunjukkan pentingnya pendekatan statistik dalam merancang dan mengevaluasi model kecerdasan buatan atau pembelajaran mesin yang efektif. Penerapan simulasi Monte Carlo, misalnya, menyoroti bagaimana kita dapat menggunakan sampling acak untuk memahami risiko dan hasil dalam situasi yang penuh ketidakpastian. Hal ini sangat penting dalam berbagai bidang, mulai dari keuangan hingga pengembangan model pembelajaran mesin yang *robust*.

Secara keseluruhan, penggunaan statistik dan probabilitas dalam kecerdasan buatan bukan hanya tentang mengolah angka, tetapi juga tentang memahami dan mengelola kompleksitas dunia nyata. Pendekatan ini memungkinkan para praktisi kecerdasan buatan untuk membangun sistem yang lebih akurat, andal, dan responsif terhadap perubahan data serta ketidakpastian. Bab ini memberikan dasar yang kuat bagi para pembaca untuk memahami peran kritis statistik dan probabilitas dalam kemajuan teknologi kecerdasan buatan, serta mendorong aplikasi lebih lanjut dalam berbagai domain yang memanfaatkan kecerdasan buatan. Lebih lanjut lagi, pembaca disarankan untuk mempelajari terkait topik-topik tertentu pada bab ini untuk dipahami secara lebih mendalam, misalnya terkait dengan algoritma pembelajaran mesin. Beberapa pembahasan secara lebih detail dan menyeluruh tidak dapat disampaikan pada bab ini dikarenakan keterbatasan penulis dan halaman pada bab ini.