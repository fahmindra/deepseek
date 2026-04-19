---
slug: mai-3
title: mai-3
---

# Bab 3: Pemodelan Matematika untuk Algoritma Machine Learning

## 3.1 Machine Learning

*Machine Learning* adalah cabang dari kecerdasan buatan (AI) yang berfokus pada pengembangan algoritma yang memungkinkan komputer untuk belajar dari dan membuat prediksi atau keputusan berdasarkan data. *Machine Learning* digunakan di berbagai bidang, termasuk pengenalan gambar, pemrosesan bahasa alami, prediksi pasar dan lainnya.

Pemodelan matematika adalah fondasi dari algoritma *Machine Learning*. Pemodelan ini membantu dalam memahami bagaimana algoritma bekerja, mengoptimalkan kinerjanya dan mengevaluasi hasilnya. Pemahaman matematika yang mendalam memungkinkan pengembangan algoritma yang lebih efisien dan akurat.

Algoritma *Machine Learning* dapat dikelompokkan menjadi beberapa kategori, antara lain: *Supervised Learning* yaitu algoritma yang belajar dari data yang diberi label, *Unsupervised Learning* yaitu algoritma yang belajar dari data yang tidak diberi label, dan *Reinforcement Learning* yaitu algoritma yang belajar melalui interaksi dengan lingkungan.

## 3.2 Regresi

Regresi merupakan Algoritma dalam *Machine Learning* yang digunakan untuk memodelkan hubungan antara *variable dependent* (biasanya berupa nilai kontinu) dengan satu atau lebih *variable independent* (fitur atau predictor). Tujuannya untuk memprediksi atau memperkirakan nilai dari *variable dependent* berdasarkan nilai *variable independent*.

Konsep dasar regresi mencoba menemukan hubungan matematis antara *variable dependent* (Y) dan satu atau lebih *variable independent* (X). Model Regresi biasanya berbentuk persamaan: $Y = f(X) + \epsilon$, dengan $f(X)$ adalah fungsi dari *variable independent*, dan $\epsilon$ adalah error atau noise.

Regresi memiliki beberapa jenis berdasarkan hubungan antara X dan Y, antara lain: Regresi Linier, Regresi Logistik dan Regresi Polinomial.

### 3.2.1 Regresi Linier

Regresi Linier adalah regresi dengan hubungan linier antara X dan Y.

**1. Model Regresi Linier Sederhana:**
$$Y = \beta_0 + \beta_1 X + \epsilon \quad (3.1)$$
dengan Y adalah *variable dependent*, X adalah *variable independent*, $\beta_0$ adalah intercept (titik potong dengan sumbu Y), $\beta_1$ adalah slope (kemiringan garis regresi), dan $\epsilon$ adalah *error* atau residu.

**2. Model Regresi Linier Berganda:**
$$Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_p X_p + \epsilon \quad (3.2)$$
Regresi Linier Berganda bersifat Multikolinearitas ketika kondisi terjadi *variable independent* saling berkorelasi tinggi. Asumsi dalam Regresi Linier yaitu untuk validitas model Regresi Linier, ada beberapa asumsi yang harus dipenuhi yaitu hubungan antara *variable independent* dan *dependent* harus linier (Linearitas), variasi *error* harus konstan diseluruh rentang *variable independent* (Homoskedastisitas), *error* harus terdistribusi normal (Normalitas Error).

### 3.2.2 Regresi Polinomial

Regresi Polinomial adalah bentuk regresi linier ketika hubungan antara variable independent $x$ dan variable dependen $y$ dimodelkan sebagai polynomial derajat $n$. Persamaan umum regresi polynomial derajat $n$ dituliskan sebagai berikut:
$$y = \beta_0 + \beta_1 x + \beta_2 x^2 + \dots + \beta_n x^n + \epsilon \quad (3.3)$$
dengan $y$ adalah *variable dependent*, $x$ adalah *variable independent*, $\beta_0, \beta_1, \dots, \beta_n$ adalah koefisien yang diperkirakan, dan $\epsilon$ adalah kesalahan (*error*) atau *noise*.

Untuk memperkirakan koefisien metode umum yang digunakan adalah Metode Kuadrat Terkecil (*Least Square Method*). Tujuannya untuk meminimalkan jumlah kuadrat dari residual (selisih antara nilai yang diamati dan nilai yang diprediksi). Untuk Matriks desain X sistem persamaan regresi polinomial adalah sebagai berikut:

$$X = \begin{bmatrix} 1 & x_1 & x_1^2 & \dots & x_1^n \\ 1 & x_2 & x_2^2 & \dots & x_2^n \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 1 & x_m & x_m^2 & \dots & x_m^n \end{bmatrix} \quad (3.4)$$

dengan $x_1, x_2, \dots, x_m$ adalah nilai-nilai dari *variable independent*.

Koefisien $\beta$ dapat diperoleh dengan menyelesaikan persamaan normal:
$$\beta = (X^T X)^{-1} X^T y \quad (3.5)$$
dengan $X^T$ adalah transpose dari matriks desain X, dan $y$ adalah vektor dari nilai-nilai *variabel dependent*.

### 3.2.3 Regresi Logistik

Regresi Logistik adalah teknik analisis statistik yang digunakan untuk memodelkan data ketika *variable dependent* (tergantung) adalah kategori atau biner (misalnya, 0 atau 1, ya atau tidak, benar atau salah). Jenis regresi logistik ini digunakan ketika hasil dari model berupa probabilitas suatu kejadian tertentu.

Persamaan dasar regresi logistik adalah fungsi logistik atau sigmoid seperti berikut:
$$\sigma(z) = \frac{1}{1 + e^{-z}} \quad (3.6)$$
dengan $z$ adalah kombinasi linier dari variabel *independent*. Model regresi logistik memprediksi probabilitas kejadian kelas positif (misalnya, y=1). Model ini dapat dinyatakan sebagai berikut:
$$z = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n \quad (3.7)$$
dengan $\beta_0$ adalah intercept(bias), $\beta_i$ adalah koefisien untuk variabel prediktor $x_i$, sedangkan $x_i$ adalah variabel *predictor*. Parameter $\beta$ dalam regresi logistik diestimasi menggunakan metode Maksimum Likelihood Estimation (MLE) untuk mencari parameter yang memaksimalkan kemungkinan (*likelihood*) pengamatan data. Persamaan *Log-likelihood* (LL) untuk regresi logistik sebagai berikut:
$$LL(\beta) = \sum_{i=1}^N [y_i \log(\sigma(z_i)) + (1 - y_i) \log(1 - \sigma(z_i))] \quad (3.8)$$
dengan $y_i$ adalah label kelas actual (0 atau 1) untuk pengamatan ke-i. Sedangkan $\sigma(z_i)$ adalah probabilitas yang diprediksi oleh model untuk pengamatan ke-i.

## 3.3 K-Nearest Neighbors (K-NN)

*K-Nearest Neighbors* (K-NN) merupakan algoritma dalam *Machine Learning* non-parametrik untuk klasifikasi dan regresi. K-NN bekerja dengan mengidentifikasi nilai K tetangga terdekat dari sebuah titik data baru dan melakukan prediksi berdasarkan nilai rata-rata dari tetangga-tetangga tersebut. Langkah-langkah Algoritma K-NN sebagai berikut:

1.  Pemilihan Parameter K yaitu menentukan jumlah tetangga terdekat K yang akan digunakan dalam klasifikasi atau prediksi.
2.  Menghitung Jarak yaitu menghitung jarak antara sampel baru dengan titik data dalam dataset pelatihan. Rumus perhitungan jarak yang umum digunakan adalah:
    **a. Jarak Euclidean** adalah konsep dasar dalam geometri yang digunakan untuk mengukur jarak antara dua titik dalam ruang Euclidean. Jarak Euclidean sering digunakan dalam berbagai bidang termasuk bidang matematika, ilmu komputer dan statistik.
    Rumus Jarak Euclidean:
    Jarak Euclidean antara dua titik $P = (p_1, p_2, \dots, p_n)$ dan $Q = (q_1, q_2, \dots, q_n)$ dalam ruang Euclidean n-dimensi yaitu panjang dari segmen garis yang menghubungkan kedua titik tersebut. Secara matematika, jarak Euclidean didefinisikan sebagai berikut:
    $$d(P, Q) = \sqrt{(p_1 - q_1)^2 + (p_2 - q_2)^2 + \dots + (p_n - q_n)^2} \quad (3.9)$$

    **b. Jarak Manhattan** adalah ukuran jarak antar dua titik dalam ruang n-dimensi yang diukur sepanjang sumbu-sumbu koordinat pada grid persegi. Nama "Manhattan" berasal dari cara menghitung jarak di grid jalanan kota Manhattan, dimana jalan-jalan berbentuk persegi Panjang.
    Rumus Jarak Manhattan:
    Jarak Manhattan antara dua titik $P = (p_1, p_2, \dots, p_n)$ dan $Q = (q_1, q_2, \dots, q_n)$ dalam ruang Euclidean n-dimensi adalah jumlah nilai mutlak dari perbedaan koordinat-koordinat mereka. Secara matematis, jarak Manhattan didefinisikan sebagai berikut:
    $$d_{Manhattan}(P, Q) = \sum_{i=1}^n |p_i - q_i| \quad (3.10)$$

3.  Mengidentifikasi K tetangga terdekat yaitu dengan mengurutkan semua jarak yang telah dihitung dari yang terkecil hingga terbesar. Kemudian memilih K titik data dengan jarak terkecil sebagai tetangga terdekat.
4.  Untuk kasus klasifikasi dengan cara menghitung kategori dengan menentukan kelas dengan menghitung jumlah kemunculan setiap kelas di antara K tetangga terdekat. Kemudian Kelas dengan jumlah terbanyak adalah kelas yang diprediksi.
5.  Untuk kasus regresi yaitu dengan mengambil rata-rata nilai K tetangga terdekat.

### 3.3.1 Contoh Perhitungan
Diketahui dataset berikut dengan 2 fitur dan 2 kelas, terlihat pada Tabel 3.1. Dari data akan memprediksi kelas baru dengan data test Fitur1 = 3, Fitur2 = 3, dengan nilai K=3.

**Tabel 3.1 Dataset untuk KNN**

| No. | Fitur1 | Fitur2 | Kelas |
| :--- | :---: | :---: | :---: |
| 1 | 1 | 2 | A |
| 2 | 2 | 3 | A |
| 3 | 3 | 1 | B |
| 4 | 6 | 5 | B |

Langkah-langkah perhitungan sebagai berikut:
1. Hitung jarak Euclidean, jarak antar dua titik $(x_1, y_1)$ dan $(x_2, y_2)$ yaitu:
   $$Jarak = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}$$
   *   Jarak ke Data No. 1: $\sqrt{(3-1)^2 + (3-2)^2} = \sqrt{4+1} = \sqrt{5} \approx 2,24$
   *   Jarak ke Data No. 2: $\sqrt{(3-2)^2 + (3-3)^2} = \sqrt{1+0} = 1$
   *   Jarak ke Data No. 3: $\sqrt{(3-3)^2 + (3-1)^2} = \sqrt{0+4} = 2$
   *   Jarak ke Data No. 4: $\sqrt{(3-6)^2 + (3-5)^2} = \sqrt{9+4} = \sqrt{13} \approx 3,61$

2. Tentukan 3 tetangga terdekat yaitu dengan mengurutkan berdasarkan jarak pendek: Data No. 2, Data No. 3, dan Data No. 1.
3. Prediksi kelas yaitu dengan melihat kelas-kelas Data tetangga terdekat:
   *   Kelas tetangga terdekat: Data No. 2 = A, Data No. 3 = B, dan Data No. 1 = A
   *   Kelas yang paling banyak adalah kelas A
   Oleh karena itu, data baru dengan Fitur1 = 3, Fitur2 = 3, diprediksi sebagai Kelas A.

Jadi dapat ditarik kesimpulan, K-NN adalah algoritma yang mudah diimplementasikan dan dipahami, tetapi memerlukan optimalisasi pada nilai K dan pengukuran jarak yang tepat untuk kinerja terbaik.

## 3.4 Support Vector Machine (SVM)

*Support Vector Machine* (SVM) merupakan algoritma *machine learning* yang digunakan untuk klasifikasi dan regresi. SVM dikenal sebagai algoritma yang mampu menangani data berdimensi tinggi dan bekerja dengan baik pada kasus dengan jumlah sample lebih kecil dibandingkan dengan dimensi fitur. Berikut adalah teori yang mendasari SVM dan langkah-langkah utama dalam algoritma SVM.

### 3.4.1 Konsep Dasar

1.  **Hyperplane Pemisah:** SVM mencari hyperplane yang memisahkan data ke dalam dua kelas berbeda. Hyperplane ini adalah batas keputusan yang memisahkan satu kelas dari kelas lainnya dalam ruang fitur.
2.  **Margin Maksimum:** SVM berusaha menemukan hyperplane yang memaksimalkan margin, yaitu jarak antara hyperplane dan titik data terdekat dari setiap kelas. Titik data yang paling dekat ke hyperplane disebut sebagai *support vectors*.
3.  **Linear dan Non-linear SVM:**
    a.  **Linear SVM** digunakan ketika data dapat dipisahkan dengan hyperplane linear.
    b.  **Non-linear SVM** digunakan ketika data tidak dapat dipisahkan secara linear. Dalam kasus ini, SVM menggunakan teknik yang disebut *kernel trick* untuk memetakan data ke dimensi yang lebih tinggi di mana data tersebut dapat dipisahkan secara linear.

![Gambar 3.1 Konsep Hyperplane SVM](https://i.imgur.com/example_placeholder.png)
**Gambar 3.1 Konsep Hyperplane SVM**

### 3.4.2 Langkah-langkah Algoritma SVM:

1.  **Mendefinisikan Fungsi Tujuan**
    Untuk menemukan hyperplane, SVM memecahkan masalah optimisasi yang bertujuan untuk memaksimalkan margin:
    $$Minimize \quad \frac{1}{2} \|w\|^2 \quad (3.11)$$
    dengan kendala: $y_i(w.x_i + b) \geq 1$ dengan $w$ adalah vektor bobot, $x_i$ adalah vektor fitur, $y_i$ adalah label kelas, dan $b$ adalah bias.

2.  **Menggunakan Kernel Trick (Jika diperlukan)**
    Ketika data tidak dapat dipisahkan secara linier, fungsi kernel $K(x_i, x_j)$ digunakan untuk memetakan data ke ruang fitur berdimensi lebih tinggi. Beberapa kernel umum meliputi kernel linier, polinomial dan Radial Basis Function (RBF). Beberapa rumus kernel umum tersebut adalah sebagai berikut:
    a.  Linier: $K(x_i, x_j) = x_i \cdot x_j \quad (3.12)$
    b.  Polinomial: $K(x_i, x_j) = (x_i \cdot x_j + c)^d \quad (3.13)$
    c.  Gaussian (RBF): $K(x_i, x_j) = \exp(-\gamma \|x_i - x_j\|^2 \quad (3.14)$

3.  **Menentukan Support Vector**
    Solusi dari masalah optimisasi di atas memberikan vektor bobot $w$ dan bias $b$. *Support vector* adalah data yang memenuhi rumus berikut:
    $$y_i(w.x_i + b) = 1 \quad (3.15)$$

4.  **Klasifikasi Data Baru**
    Setelah model SVM dilatih, data baru dapat diklasifikasikan dengan menggunakan fungsi keputusan:
    $$f(x) = \text{sign}(w.x_i + b) \quad (3.16)$$
    Data baru $x$ di klasifikasikan berdasarkan $f(x)$ apakah hasil dari positif atau negatif. Jika sign $f(x)$ positif, maka sampe data $x$ tersebut diklasifikasikan ke dalam satu kelas (misalnya, +1) dan jika sign $f(x)$ negatif, maka sample tersebut diklasifikasikan ke kelas yang lain (misalnya, -1). Algoritma SVM adalah algoritma yang kuat dalam analisis data, khususnya Ketika menghadapi data yang sulit dipisahkan secara linier, dan memberikan hasil yang dapat diandalkan jika parameter model dipilih dengan baik.

## 3.5 Decision Tree

*Decision Tree* adalah salah satu algoritma pembelajaran mesin yang digunakan untuk tugas klasifikasi dan regresi. Algoritma ini bekerja dengan membangun model pohon keputusan berdasarkan fitur dari data untuk memprediksi nilai target. Berikut adalah teori dasar dan langkah-langkah utama dari algoritma *decision tree*, beserta contoh perhitungannya.

### 3.5.1 Konsep Dasar

Struktur *decision tree* dimulai dari node, cabang dan keputusan.

![Gambar 3.2 Struktur Decision Tree](https://i.imgur.com/example_placeholder.png)
**Gambar 3.2 Struktur Decision Tree**

1.  **Node**
    Node terdiri dari 3 level yaitu Root Node, Internal Node, dan Leaf Node. Root Node adalah node paling atas dari pohon yang mewakili seluruh dataset.
2.  **Cabang**
    Cabang adalah penghubung node dan menunjukan hasil dari pengujian pada atribut.
3.  **Keputusan**
    Keputusan diambil dengan mengikuti jalur dari rood node ke leaf node.

### 3.5.2 Langkah-langkah Algoritma Decision Tree

1.  **Pilih Atribut Terbaik**
    Untuk memilih atribut terbaik sebagai node, maka perlu menggunakan kriteria tertentu seperti Gini Impurity, Entropy, dan Information Gain.
2.  **Pembagian Dataset**
    Dataset dibangun menjadi subset berdasarkan nilai atribut yang dipilih.
3.  **Pembangunan Pohon**
    Proses ini dilakukan secara berulang dengan cara rekrusif untuk setiap subset sampai salah satu dari kondisi berikut ini terpenuhi:
    a. Semua sample dalam subset memiliki label yang sama
    b. Tidak ada atribut yang tersisa untuk dibagi
    c. Tidak ada data dalam subset.
4.  **Pohon Keputusan**
    Pohon keputusan dibentuk dengan node dan cabang dimana setiap node internal menunjukkan uji pada atribut, dan setiap leaf node menunjukkan hasil klasifikasi.

### 3.5.3 Contoh Perhitungan Decision Tree

Studi kasus sederhana, terdapat dataset Tabel 3.2 yang akan digunakan untuk mengklasifikasi apakah seseorang akan bermain tenis berdasarkan cuaca dengan menggunakan Algoritma Decision Tree. Dataset:

**Tabel 3.2 Dataset untuk Decision Tree**

| Cuaca | Suhu | Kelembaban | Berangin | Bermain Tenis |
| :--- | :--- | :--- | :--- | :--- |
| Cerah | Tinggi | Tinggi | Tidak | Tidak |
| Cerah | Tinggi | Tinggi | Ya | Tidak |
| Mendung | Tinggi | Tinggi | Tidak | Ya |
| Hujan | Sedang | Tinggi | Tidak | Ya |
| Hujan | Rendah | Normal | Tidak | Ya |
| Hujan | Rendah | Normal | Ya | Tidak |
| Mendung | Rendah | Normal | Ya | Ya |
| Cerah | Sedang | Tinggi | Tidak | Tidak |
| Cerah | Rendah | Normal | Tidak | Ya |
| Hujan | Sedang | Normal | Tidak | Ya |
| Cerah | Sedang | Normal | Ya | Ya |
| Mendung | Sedang | Tinggi | Ya | Ya |
| Mendung | Tinggi | Normal | Tidak | Ya |
| Hujan | Sedang | Tinggi | Ya | Tidak |

**Langkah-langkahnya:**

**1. Menghitung Nilai Entropy Awal**
Menghitung entropy untuk seluruh dataset dengan rumus:
$$Entropy(S) = \sum_{i=1}^n -p_i * \log_2 p_i \quad (3.17)$$

Berdasarkan Tabel 3.2, terdapat 14 record data, 9 data bermain tenis (Kelas Positif) dan 5 tidak bermain tenis (Kelas Negatif):
$$Entropy(S) = - \frac{9}{14} \log_2 \left(\frac{9}{14}\right) + - \frac{5}{14} \log_2 \left(\frac{5}{14}\right) = 0,940$$

**2. Menghitung Nilai Information Gain untuk Atribut "Cuaca":**
Membagi dataset berdasarkan nilai Cuaca (Cerah, Mendung, Hujan), kemudian menghitung entropy setiap subset.
*   **Mendung:**
    $$Entropy(S_{Mendung}) = - \frac{4}{4} \log_2 \left(\frac{4}{4}\right) = 0 \text{ (Semua Positif)}$$
*   **Cerah:**
    $$Entropy(S_{Cerah}) = - \frac{2}{5} \log_2 \left(\frac{2}{5}\right) - \frac{3}{5} \log_2 \left(\frac{3}{5}\right) = 0,971$$
*   **Hujan:**
    $$Entropy(S_{Hujan}) = - \frac{3}{5} \log_2 \left(\frac{3}{5}\right) - \frac{2}{5} \log_2 \left(\frac{2}{5}\right) = 0,971$$

*   **Menghitung Nilai Information Gain:**
    $$Gain(S, Cuaca) = Entropy(S) - \left( \frac{5}{14} \times 0,971 + \frac{4}{14} \times 0 + \frac{5}{14} \times 0,971 \right)$$
    $$Gain(S, Cuaca) = 0.940 - 0,693 = 0,247$$

**3. Pilih atribut dengan Gain Tertinggi**
Bandingkan nilai Information Gain unutk atribut lainnya yaitu Suhu, Kelembaban, Berangin, dan pilih atribut denan nilai Gain tertinggi untuk node berikutnya.

**4. Rekursif membagi dataset**
Ulangi proses diatas untuk setiap cabang menggunakan subset yang dihasilkan dari pembagian sebelumnya.

### 3.5.4 Pohon Keputusan (*Decision Tree*)
Berikut merupakan Pohon Keputusan yang terbentuk dari dataset Tabel 3.2.

![Gambar 3.3 Decision Tree berdasarkan Tabel 3.2](https://i.imgur.com/example_placeholder.png)
**Gambar 3.3 Decision Tree berdasarkan Tabel 3.2**

Pada Gambar 3.3 merupakan pohon keputusan yang diambil dengan mengikuti jalur dari root ke leaf berdasarkan nilai atribut-atribut dari dataset Tabel 3.2.

## 3.6 Jaringan Syaraf Tiruan

Jaringan Syaraf Tiruan (JST) atau *Artificial Neural Network* (ANN) adalah model komputasi yang terinspirasi oleh cara kerja otak manusia dalam memproses informasi. Gambar 3.4 merupakan ilustrasi dari jaringan syaraf biologis manusia.

![Gambar 3.4 Ilustrasi Jaringan Syaraf Biologis Manusia](https://i.imgur.com/example_placeholder.png)
**Gambar 3.4 Ilustrasi Jaringan Syaraf Biologis Manusia**
*(Sumber : javapoint.com)*

JST terdiri dari sejumlah besar unit pemrosesan sederhana yang disebut neuron, yang dihubungkan oleh bobot yang dapat disesuaikan. JST dapat belajar dari data dengan menyesuaikan bobot-bobot ini. Arsitektur Jaringan Syaraf Tiruan menggunakan beragam lapisan pemrosesan matematis untuk menginterpretasikan informasi yang diberikan. Umumnya, JST terdiri dari puluhan hingga jutaan neuron buatan yang disebut unit, yang disusun dalam beberapa lapisan atau layer. Untuk gambar arsitektur JST terlihat pada Gambar 3.5.

![Gambar 3.5 Arsitektur Jaringan Syaraf Tiruan](https://i.imgur.com/example_placeholder.png)
**Gambar 3.5 Arsitektur Jaringan Syaraf Tiruan**
*(Sumber : javapoint.com)*

JST memiliki komponen utama yaitu Neuron, Lapisan (*Layers*), Bobot (*Weights*), dan Fungsi Aktivasi. Neuron adalah Unit dasar dari JST yang memproses informasi, Lapisan (*Layers*) terdiri dari Lapisan Masukan (*Input Layer*): tempat data mentah dimasukkan ke dalam jaringan; Lapisan Tersembunyi (*Hidden Layer*): lapisan yang memproses data setelah input, bisa lebih dari satu; dan Lapisan Keluaran (*Output Layer*): hasil akhir dari pemrosesan jaringan, Bobot (*Weights*) yaitu nilai yang menghubungkan neuron satu dengan neuron yang lain ketika bobot ini akan diperbarui selama proses pembelajaran, dan Fungsi Aktivasi yaitu fungsi matematis yang menentukan keluaran dari suatu neuron berdasarkan masukan yang diterima, contohnya: sigmoid, relu, tanh.

Adapun proses pembelajaran JST sebagai berikut:
1.  **Inisialisasi Bobot:** Bobot-bobot diinisialisasi dengan nilai acak kecil.
2.  **Feedforward:** Data masukan diproses dari lapisan masukan menuju lapisan keluaran melalui lapisan tersembunyi.
3.  **Perhitungan Kesalahan (Loss):** Menghitung seberapa jauh hasil jaringan dari nilai yang diharapkan menggunakan fungsi loss.
4.  **Backpropagation:** Menyesuaikan bobot berdasarkan kesalahan yang dihitung untuk meminimalkan loss.
5.  **Pembaharuan Bobot:** Bobot diperbarui menggunakan algoritma optimasi seperti *Stochastic Gradient Descent* (SGD).

### 3.6.1 Contoh Perhitungan JST
Misalkan kita memiliki JST sederhana dengan satu neuron, satu masukan $(x)$, dan satu keluaran $(y)$. Fungsi aktivasi yang digunakan adalah fungsi linear.

**1. Inisialisasi**
Bobot awal $(w)$ diinisialisasi dengan nilai acak, misalkan $w = 0,5$. Kemudian nilai Bias $(b)$ juga diinisialisasi dengan nilai, misalkan $b = 0,1$.

**2. Feedforward**
Fungsi aktivasi linier $y = wx + b$. Misalkan, masukan $(x = 2)$, maka keluaran $(y)$ adalah
$$y = (0,5 \times 2) + 0,1 = 1,0 + 0,1 = 1,1$$

**3. Perhitungan Kesalahan**
Misalkan target keluaran $(y_{target} = 1,5)$, maka nilai kesalahan (E) adalah:
$$E = \frac{1}{2}(y_{target} - y)^2 = \frac{1}{2}(1,5 - 1,1)^2 = 0,08$$

**4. Backpropagation dan Pembaruan Bobot**
Gradien dari kesalahan terhadap bobot $(\frac{\partial E}{\partial w})$ dihitung dan digunakan untuk memperbarui bobot:
$$\frac{\partial E}{\partial w} = (y - y_{target}) \cdot x = (1,1 - 1,5) \cdot 2 = 10,8$$
Menggunakan laju pembelajaran $(\alpha = 0,01)$, bobot baru $(w')$ diperoleh:
$$w' = w - \alpha \cdot \frac{\partial E}{\partial w} = 0,5 - 0,01 \cdot (-0,8) = 0,508$$
Iterasi ini akan terus berulang sampai kesalahan (E) mencapai nilai yang diinginkan atau konvergensi tercapai. Ini adalah gambaran dasar bagaiaman JST bekerja dalam proses pembelajaran.

## 3.7 Evaluasi Model

Evaluasi model algoritma *machine learning* adalah langkah penting untuk memastikan bahwa model yang telah dibangun mampu melakukan prediksi dengan baik pada data yang belum pernah dilihat sebelumnya. Berikut adalah beberapa konsep teori dasar dan contoh perhitungan sederhana dalam evaluasi model *machine learning*:

### 3.7.1 Teori Evaluasi Model

**1. Pembagian data**
Pada proses *machine learning* menggunakan dataset, dimana dataset di bagi menjadi: Data Training, Data Validation dan Data Test.
*   **Data Training** adalah data yang digunakan untuk melatih model.
*   **Data Validation** adalah data yang digunakan untuk mengoptimalkan model dan memilih parameter terbaik.
*   **Data Test** adalah data yang digunakan untuk mengevaluasi kinerja akhir model.

**2. Metode Evaluasi**
*   **Akurasi:** Proporsi prediksi yang benar dari keseluruhan prediksi. Cocok untuk dataset yang seimbang.
*   **Precision, Recall, dan F1-Score:** Digunakan untuk mengevaluasi kinerja model pada dataset yang tidak seimbang.
*   **Precision:** Jumlah prediksi positif benar dibagi dengan jumlah total prediksi positif.
*   **Recall (Sensitivity):** Jumlah prediksi positif benar dibagi dengan jumlah total data positif yang sebenarnya.
*   **F1-Score:** *Harmonic mean* dari *precision* dan *recall*.
*   **Confusion Matrix:** Tabel yang digunakan untuk mendeskripsikan kinerja model classification, terdiri dari *True Positive* (TP), *False Positive* (FP), *True Negative* (TN), dan *False Negative* (FN).
*   **ROC-AUC Curve:** Grafik yang menunjukkan kemampuan model dalam membedakan antara kelas positif dan negatif.

**3. Cross Validation**
Metode pembagian data yang lebih robust dengan membagi dataset menjadi beberapa subset (folds) dan melatih serta menguji model pada setiap kombinasi subset. Ilustrasi pembagian data terlihat pada Gambar 3.6 di bawah ini.

![Gambar 3.6 Ilustrasi Pembagian Dataset Cross Validation](https://i.imgur.com/example_placeholder.png)
**Gambar 3.6 Ilustrasi Pembagian Dataset Cross Validation**

### 3.7.2 Contoh Perhitungan
Diketahui data hasil prediksi yaitu TP=50, FP=10, TN=30, FN=10. Hitung nilai akurasi, precision, recall dan F1-score.

**1. Hitung nilai akurasi**
$$Akurasi = \frac{TP + TN}{TP + FP + TN + FN} = \frac{50 + 30}{50 + 10 + 30 + 10} = \frac{80}{100} = 80\%$$
Akurasi dari model di atas adalah 80%.

**2. Hitung nilai precision**
$$Precision = \frac{TP}{TP + FP} = \frac{50}{50 + 10} = \frac{50}{60} \approx 0,833$$
Precision dari kedua model adalah 83,3% (ditulis 83% pada naskah asli).

**3. Hitung nilai Recall**
$$Recall = \frac{TP}{TP + FN} = \frac{50}{50 + 10} = \frac{50}{60} \approx 0,833$$
Recall dari model adalah 83,33%.

**4. Hitung nilai F1-Score**
$$F1 - Score = 2x \frac{Precision \times Recall}{Precision + Recall} = 2x \frac{0,833 \times 0,833}{0,833 + 0,833} = \frac{50}{60} \approx 0,833$$
F1-Score dari model adalah 83,33%.

Dengan menggunakan metrik-metrik di atas, kita dapat mengevaluasi model machine learning dan memahami di mana kekuatan dan kelemahan model tersebut. Dalam kasus dataset yang tidak seimbang, metrik seperti precision, recall, dan F1-score sering kali lebih informatif daripada hanya mengandalkan akurasi.

Pemodelan matematika memberikan dasar yang kuat untuk memahami dan mengimplementasikan algoritma *Machine Learning*. Setiap algoritma memiliki pendekatan unik untuk permodelan dan optimisasi, yang memungkinkan aplikasi yang luas dalam berbagai bidang. Dengan memahami matermatika dibalik Aalgoritma-algoritma *Machine Learning* dapat dikembangkan model yang lebih akurat dan efisien untuk berbagai tugas *Machine Learning*.
