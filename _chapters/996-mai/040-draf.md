---
slug: mai-4
title: mai-4
---

# Bab 4: Matematika dalam Pengembangan Model Prediktif

## 4.1 Model Prediktif

Setiap hari, kita sering kali dihadapkan pada pertanyaan-pertanyaan seperti:
* "Apakah besok pagi cuaca akan hujan?"
* "Apakah saya harus membeli rumah sekarang atau perlu menunggu?"
* "Apakah produk baru ini akan sukses di pasaran?"

Pertanyaan-pertanyaan ini mencerminkan keinginan kita untuk memprediksi kejadian di masa depan, agar kita dapat membuat keputusan yang paling bijak untuk menghadapinya. Kita biasanya membuat keputusan berdasarkan informasi yang kita peroleh. Dalam beberapa kasus, kita memiliki data yang aktual dan obyektif, seperti laporan lalu lintas pagi hari atau prakiraan cuaca (Adam, 2011). Di lain waktu, kita mengandalkan intuisi dan pengalaman, seperti "Saya harus menghindari jembatan pagi ini karena biasanya jembatan tersebut macet pada saat jam berangkat kerja", atau "Saya akan berbelanja pada saat *payday*, karena biasanya pada saat itu banyak barang yang diberi diskon". Apa pun kasusnya, kita memprediksi kejadian di masa depan berdasarkan informasi dan pengalaman yang kita miliki saat ini, dan kita mengambil keputusan berdasarkan prediksi tersebut (Allgöwer & Zheng, 2012).

Ketika informasi semakin mudah diakses melalui internet, keinginan untuk memanfaatkannya dalam pengambilan keputusan semakin meningkat. Meskipun otak manusia, baik secara sadar maupun tidak sadar, mampu mengumpulkan data dalam jumlah besar, kemampuan otak untuk memproses informasi yang relevan dan mudah didapatkan dalam jumlah besar tetap terbatas dalam menyelesaikan masalah. Untuk membantu proses pengambilan keputusan, sekarang mengandalkan *platform* seperti Google untuk menemukan informasi yang paling tepat untuk pertanyaan dan Google Maps untuk merencanakan rute perjalanan terbaik, melihat kondisi lalu lintas secara *real-time*, dan menemukan tempat-tempat menarik yang ada di sekitar.

*Platform* yang disebutkan di atas mengumpulkan informasi, menyaring data untuk mengidentifikasi pola yang relevan dengan kebutuhan kita dan memberikan jawaban. *Platform* ini tidak hanya memanfaatkan informasi yang tersedia secara luas, tetapi juga menggunakan algoritma kompleks untuk menganalisis dan memproses data secara cepat dan efisien. *Platform* ini dapat menggabungkan data historis, tren saat ini, dan preferensi individu untuk menyajikan prediksi yang lebih personal dan akurat. Dengan demikian, *platform* ini tidak hanya berfungsi sebagai alat pencarian atau navigasi semata, tetapi juga sebagai solusi yang membantu untuk mengambil keputusan di kehidupan sehari-hari.

Di dalam bidang ilmu komputer, proses pengembangan *platform* semacam ini disebut sebagai "pembelajaran mesin", "kecerdasan buatan", "pengenalan pola", "penambangan data", dan "analisis prediktif" (Sarker, 2021). Tujuannya adalah untuk membuat prediksi yang akurat berdasarkan studi kasus yang digunakan dan disebut sebagai model prediktif. Definisi model prediktif adalah proses pengembangan alat atau model matematika yang bertujuan untuk menghasilkan prediksi yang akurat (Kuhn & Johnson, 2016).

Sebagai contoh aplikasi model prediktif dalam kehidupan sehari-hari adalah Netflix dan Spotify. Netflix yang merupakan perusahaan media perfilman menggunakan pemodelan prediktif untuk memberikan rekomendasi film yang disukai pemirsanya berdasarkan pilihan film yang telah diputar atau disukai sebelumnya (Amatriain, 2013). Apabila pemirsa sering menonton film dengan genre misteri, horor, dan *thriller*, maka Netflix akan merekomendasikan film dengan genre serupa sebagai prediksi bahwa pemirsa akan menyukainya. Selain itu, terdapat pula Spotify yang dapat memberikan rekomendasi untuk membantu pendengarnya menemukan musik baru yang sesuai dengan seleranya berdasarkan riwayat mendengarkan sebelumnya (Pareek et al., 2022).

Meskipun model prediktif dapat membantu untuk mendapatkan pelayanan yang lebih baik, terkadang model tersebut dapat menghasilkan prediksi yang kurang akurat sehingga memberikan jawaban yang salah (Lantz, 2019). Misalkan saja, dalam bidang kesehatan, sebuah model prediktif digunakan untuk memprediksi kemungkinan seorang pasien menderita penyakit tertentu berdasarkan data medisnya. Jika model tersebut tidak akurat, ada beberapa kemungkinan skenario yang bisa terjadi. Misalnya, model memprediksi bahwa pasien menderita penyakit padahal sebenarnya tidak. Dampaknya pasien mungkin akan menjalani tes dan perawatan yang tidak diperlukan. Bisa juga model memprediksi bahwa pasien tidak menderita penyakit padahal sebenarnya iya. Dampaknya pasien tidak mendapatkan perawatan yang dibutuhkan tepat waktu sehingga dapat memperburuk kondisi kesehatannya.

Pada bab ini akan dibahas mengenai matematika untuk proses pengembangan model prediktif dengan menggunakan studi kasus secara langsung. Model prediktif merupakan alat penting dalam analisis data, pembelajaran mesin, dan berbagai aplikasi untuk membuat prediksi atau estimasi mengenai data masa depan berdasarkan data historis (Brink et al., 2016). Proses pengembangan model prediktif melibatkan beberapa langkah pembahasan mengenai matematika yang terlibat dalam pengembangan model prediktif akan dijelaskan langsung dengan menggunakan contoh studi kasus menggunakan algoritma *Decision Tree* dan *K-Nearest Neighbor* (K-NN).

## 4.2 Studi Kasus: Prediksi Perekrutan Karyawan

Sebuah perusahaan ingin meningkatkan efisiensi dan akurasi dalam proses perekrutan karyawannya. Setiap tahun, perusahaan menerima ribuan aplikasi untuk berbagai posisi, dan proses penyaringan serta penilaian kandidat memakan banyak waktu dan sumber daya. Oleh karena itu, perusahaan memutuskan untuk mengimplementasikan model prediktif untuk membantu dalam proses seleksi awal kandidat. Tujuan utama dari studi kasus ini adalah untuk membangun model prediktif yang dapat mengidentifikasi kandidat pelamar kerja yang paling berpotensi untuk dipekerjakan.

## 4.3 Dataset

Berdasarkan studi kasus di atas, dibutuhkan kumpulan data (*dataset*) untuk bisa membuat sistem prediksi perekrutan karyawan. Pada contoh kasus nyata di lapangan, perlu dikumpulkan data terkait studi kasus di suatu perusahaan. Misalkan, diminta data pelamar kerja dari bagian HRD selama 3 tahun terakhir agar data yang dikumpulkan valid dan *up to date*. Kemudian, data-data tersebut perlu dianalisis untuk menentukan atribut yang mempengaruhi keputusan bagian HRD untuk menerima kandidat pelamar kerja. Akan tetapi pada sub bab ini, akan digunakan sampel *dataset* untuk perekrutan karyawan.

*Dataset* ini memberikan wawasan tentang faktor-faktor yang mempengaruhi keputusan perekrutan. Setiap *record* data merepresentasikan seorang kandidat pelamar kerja dengan berbagai atribut yang dipertimbangkan selama proses perekrutan. Tujuannya adalah untuk memprediksi apakah seorang kandidat akan dipekerjakan berdasarkan atribut-atribut tersebut. Pada Tabel 4.1 dijelaskan detail atribut yang digunakan. Nomor 1 sampai 4 adalah atribut, sedangkan nomor 5 adalah kelas prediksi untuk menentukan hasil dari keputusan perekrutan (diterima atau tidak diterima).

**Tabel 4.1 Detail atribut untuk prediksi perekrutan karyawan**

| no | atribut | rentang data | tipe data |
| :--- | :--- | :--- | :--- |
| 1. | usia | 20 – 50 tahun | integer |
| 2. | skor wawancara | 0 - 100 | integer |
| 3. | skor keterampilan | 0 - 100 | integer |
| 4. | skor kepribadian | 0 - 100 | integer |
| 5. | keputusan perekrutan | Tidak diterima (0) / diterima (1) | biner |

Sebagai contoh untuk pengerjaan menggunakan algoritma prediksi di sub bab berikutnya, maka akan digunakan sampel *dataset* pada Tabel 4.2.

**Tabel 4.2 Sampel *dataset* untuk prediksi perekrutan karyawan**

| no | usia | skor wawancara | skor keterampilan | skor kepribadian | keputusan |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 1. | 25 | 70 | 78 | 91 | 1 |
| 2. | 35 | 75 | 80 | 80 | 1 |
| 3. | 25 | 40 | 40 | 60 | 0 |
| 4. | 27 | 60 | 80 | 80 | 1 |
| 5. | 35 | 75 | 80 | 80 | 1 |
| 6. | 40 | 40 | 55 | 60 | 0 |
| 7. | 40 | 75 | 60 | 50 | 1 |
| 8. | 35 | 75 | 80 | 80 | 1 |
| 9. | 27 | 40 | 55 | 60 | 0 |
| 10. | 30 | 60 | 40 | 50 | 0 |

## 4.4 Decision Tree

*Decision Tree* adalah model *supervised learning*, yang belajar dari data historis untuk membuat prediksi pada data baru yang belum diketahui hasilnya. *Decision Tree* adalah model pembelajaran mesin yang dapat digunakan untuk klasifikasi dan regresi (Suthaharan, 2016). Apabila untuk klasifikasi, *Decision Tree* digunakan untuk memprediksi kategori atau kelas dari sebuah data input. Contohnya adalah memprediksi apakah seorang kandidat akan diterima atau tidak berdasarkan atribut tertentu. Apabila untuk regresi, *Decision Tree* juga bisa digunakan untuk memprediksi nilai numerik. Misalnya, memprediksi harga rumah berdasarkan atribut seperti luas bangunan, lokasi, dan jumlah kamar.

*Decision Tree* membuat prediksi dengan membuat serangkaian keputusan berdasarkan atribut data input. Pada setiap node di dalam pohon keputusan, atribut tertentu digunakan untuk memisahkan data menjadi subset yang lebih homogen (Tsang et al., 2009). Proses ini berlanjut sampai model mencapai keputusan akhir di *leaf node*. Struktur model *Decision Tree* dapat dilihat pada Gambar 4.1 (Fatimah & Rahmawati, 2021).

![Gambar 4.1 Struktur model Decision Tree](https://i.imgur.com/example_image_decision_tree_structure.png)
**Gambar 4.1 Struktur model *Decision Tree***

*Root node* adalah node pertama dan paling atas. *Root node* merupakan titik awal ketika seluruh proses pemisahan data dimulai. *Internal node* adalah node yang berada di antara *root node* dan *leaf node*. *Internal node* digunakan untuk memisahkan data berdasarkan nilai dari atribut tertentu. *Leaf node* adalah node terakhir yang tidak memiliki cabang lebih lanjut. *Leaf node* memberikan keputusan akhir atau hasil dari jalur keputusan. Jalur keputusan adalah lintasan dari *root node* ke *leaf node*. Jalur ini terdiri dari serangkaian node dan cabang yang dilalui untuk mencapai keputusan akhir. Dalam Gambar 4.1, jalur keputusan ditunjukkan dengan garis yang menghubungkan *root node* ke *leaf node* melalui *internal node*. *Sub-tree* adalah bagian dari *Decision Tree* yang dimulai dari suatu *internal node* dan mencakup semua node di bawahnya. Untuk membuat model prediksi *Decision Tree*, digunakan beberapa rumus dan konsep penting dari teori informasi, khususnya konsep *Entropy* dan *Information Gain*. Berikut adalah rumus dan langkah-langkahnya secara detail:

### 1. Entropy (Pengukuran Ketidakpastian)
*Entropy* adalah ukuran ketidakpastian atau *impurity* dalam sebuah *dataset*. *Entropy* untuk sebuah *dataset* $S$ dengan $c$ kelas yang berbeda dihitung sebagai (Sandag & Manuke, 2020):
$$Entropy(S) = \sum_{i=1}^{c} -p_i \log_2 (p_i) \qquad (4.1)$$
dengan $p_i$ adalah proporsi elemen dalam kelas $i$ di *dataset* $S$.

### 2. Information Gain (Pengurangan Ketidakpastian)
*Information Gain* adalah pengurangan ketidakpastian yang diperoleh dengan membagi *dataset* $S$ berdasarkan atribut $A$. Informasi Gain $IG(S, A)$ dihitung sebagai (Sandag & Manuke, 2020):
$$IG(S, A) = Entropy(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} \times Entropy(S_v) \qquad (4.2)$$
dengan $Values(A)$ adalah set nilai-nilai unik dari atribut $A$, $S_v$ adalah *subset* dari $S$ di mana atribut $A$ memiliki nilai $v$, $|S|$ adalah jumlah elemen dalam *dataset* $S$, dan $S_v$ adalah jumlah elemen dalam *subset* $S_v$.

Sedangkan langkah-langkah yang harus diikuti pada saat menggunakan *Decision Tree* adalah sebagai berikut:
1. Hitunglah *Entropy* dari *dataset* awal S.
2. Hitunglah *Information Gain* untuk setiap atribut. Pada langkah ini, gunakan rumus *Information Gain* untuk menghitung pengurangan ketidakpastian.
3. Pilihlah atribut yang memiliki *Information Gain* tertinggi dan jadikan *root node* atau *node* cabang berikutnya.
4. Bagilah *dataset* berdasarkan atribut yang dipilih. Tiap *subset* akan menjadi cabang dari *node* tersebut.
5. Ulangi langkah 1-4 untuk setiap *subset* data sampai:
6. semua data dalam *subset* memiliki kelas yang sama,
7. tidak ada atribut lagi yang bisa dipecah,
8. tercapai kedalaman maksimum *tree* yang diinginkan.

Untuk lebih memahami rumus dan langkah-langkah di atas, maka disediakan contoh manual model prediksi menggunakan *Decision Tree* yang disajikan pada sub bab 4.5 di bawah ini.

## 4.5 Contoh Manual Model Prediksi Menggunakan Decision Tree

Sebagai contoh untuk perhitungan manual, akan digunakan *dataset* pada Tabel 4.2. Terdapat 10 kandidat pelamar kerja dengan 4 atribut penilaian yaitu: usia, skor wawancara, skor keterampilan, skor kepribadian. Keempat atribut ini akan menentukan kelas prediksi (keputusan perekrutan). Pada atribut skor wawancara, skor keterampilan dan skor kepribadian terdapat *threshold* untuk menentukan nilai ambang batas. Berikut ini adalah langkah-langkahnya:

### 1. Hitung Entropy Awal (S)
Total kelas Keputusan Perekrutan adalah 10 baris dengan 6 sampel positif (+) yang dinyatakan diterima dan 4 sampel negatif (-) yang dinyatakan tidak diterima, sehingga:
$$p_+ = \frac{6}{10} = 0.6$$
$$p_- = \frac{4}{10} = 0.4$$
$$Entropy(S) = -0.6 \log_2(0.6) - 0.4 \log_2(0.4) = \mathbf{0.971}$$

### 2. Hitung Entropy dan Information Gain untuk tiap atributnya

#### a. Atribut Usia
* Usia = 25, terdapat 2 kandidat yang berusia 25 tahun, 1 kandidat dinyatakan diterima sedangkan 1 kandidat lainnya tidak diterima (Total: 2; Positif: 1; Negatif: 1), sehingga:
  $$p_+ = \frac{1}{2} = 0.5$$
  $$p_- = \frac{1}{2} = 0.5$$
  $Entropy(Usia = 25) = -0.5 \log_2(0.5) - 0.5 \log_2(0.5) = 1$
* Usia = 27, terdapat 2 kandidat yang berusia 27 tahun, 1 kandidat dinyatakan diterima sedangkan 1 kandidat lainnya tidak diterima (Total: 2; Positif: 1; Negatif: 1), sehingga:
  $$p_+ = \frac{1}{2} = 0.5$$
  $$p_- = \frac{1}{2} = 0.5$$
  $Entropy(Usia = 27) = -0.5 \log_2(0.5) - 0.5 \log_2(0.5) = 1$
* Usia = 30, terdapat 1 kandidat yang berusia 30 tahun, 0 kandidat dinyatakan diterima sedangkan 1 kandidat tersebut dinyatakan tidak diterima (Total: 1; Positif: 0; Negatif: 1), sehingga:
  $$p_+ = \frac{0}{1} = 0$$
  $$p_- = \frac{1}{1} = 1$$
  $Entropy(Usia = 30) = -0 \log_2(0) - 1 \log_2(1) = 0$
* Usia = 35, terdapat 3 kandidat yang berusia 35 tahun, 3 kandidat dinyatakan diterima sedangkan 0 kandidat lainnya tidak diterima (Total: 3; Positif: 3; Negatif: 0), sehingga:
  $$p_+ = \frac{3}{3} = 1$$
  $$p_- = \frac{0}{3} = 0$$
  $Entropy(Usia = 35) = -1 \log_2(1) - 0 \log_2(0) = 0$
* Usia = 40, terdapat 2 kandidat yang berusia 40 tahun, 1 kandidat dinyatakan diterima sedangkan 1 kandidat lainnya tidak diterima (Total: 2; Positif: 1; Negatif: 1), sehingga:
  $$p_+ = \frac{1}{2} = 0.5$$
  $$p_- = \frac{1}{2} = 0.5$$
  $Entropy(Usia = 40) = -0.5 \log_2(0.5) - 0.5 \log_2(0.5) = 1$

Setelah diketahui semua nilai *Entropy* untuk atribut usia, maka langkah selanjutnya adalah mencari *Information Gain* yang ditunjukkan sebagai berikut ini:
$$Gain(S, Usia) = 0.971 - (2/10 \times 1 + 2/10 \times 1 + 1/10 \times 0 + 3/10 \times 0 + 2/10 \times 1)$$
$$Gain(S, Usia) = 0.971 - (0.2 \times 1 + 0.2 \times 1 + 0.1 \times 0 + 0.3 \times 0 + 0.2 \times 1)$$
$$Gain(S, Usia) = 0.971 - 0.6$$
$$Gain(S, Usia) = \mathbf{0.371}$$

#### b. Atribut Skor Wawancara
Skor Wawancara $\leq 60$, terdapat 5 kandidat yang memiliki skor wawancara kurang dari sama dengan 60, 1 kandidat dinyatakan diterima sedangkan 4 kandidat lainnya tidak diterima (Total: 5; Positif: 1; Negatif: 4), sehingga:
$$p_+ = \frac{1}{5} = 0.2$$
$$p_- = \frac{4}{5} = 0.8$$
$Entropy(Skor \ Wawancara \leq 60) = -0.2 \log_2(0.2) - 0.8 \log_2(0.8) = \mathbf{0.7219}$

Skor Wawancara $> 60$, terdapat 5 kandidat yang memiliki skor wawancara lebih dari 60, 5 kandidat dinyatakan diterima sedangkan 0 kandidat lainnya tidak diterima (Total: 5; Positif: 5; Negatif: 0), sehingga:
$$p_+ = \frac{5}{5} = 1$$
$$p_- = \frac{0}{5} = 0$$
$Entropy(Skor \ Wawancara > 60) = -1 \log_2(1) - 0 \log_2(0) = \mathbf{0}$

Setelah diketahui semua nilai *Entropy* untuk atribut wawancara, maka langkah selanjutnya adalah mencari *Information Gain* yang ditunjukkan sebagai berikut ini:
$$Gain(S, Skor \ Wawancara) = 0.971 - (5/10 \times 0.7219 + 5/10 \times 0)$$
$$Gain(S, Skor \ Wawancara) = 0.971 - 0.361 = \mathbf{0.61}$$

#### c. Atribut Skor Keterampilan
Skor Keterampilan $\leq 55$, terdapat 4 kandidat yang memiliki skor keterampilan kurang dari sama dengan 55, 0 kandidat dinyatakan diterima sedangkan 4 kandidat lainnya tidak diterima (Total: 4; Positif: 0; Negatif: 4), sehingga:
$$p_+ = \frac{0}{4} = 0$$
$$p_- = \frac{4}{4} = 1$$
$Entropy(Skor \ Keterampilan \leq 55) = -0 \log_2(0) - 1 \log_2(1) = \mathbf{0}$

Skor Keterampilan $> 55$, terdapat 6 kandidat yang memiliki skor keterampilan lebih dari 55, 6 kandidat dinyatakan diterima sedangkan 0 kandidat lainnya tidak diterima (Total: 6; Positif: 6; Negatif: 0), sehingga:
$$p_+ = \frac{6}{6} = 1$$
$$p_- = \frac{0}{6} = 0$$
$Entropy(Skor \ Keterampilan > 55) = -1 \log_2(1) - 0 \log_2(0) = \mathbf{0}$

Setelah diketahui semua nilai *Entropy* untuk atribut keterampilan, maka langkah selanjutnya adalah mencari *Information Gain* yang ditunjukkan sebagai berikut ini:
$$Gain(S, Skor \ Keterampilan) = 0.971 - (4/10 \times 0 + 6/10 \times 0)$$
$$Gain(S, Skor \ Keterampilan) = 0.971 - 0 = \mathbf{0.971}$$

#### d. Atribut Skor Kepribadian
Skor Kepribadian $\leq 60$, terdapat 5 kandidat yang memiliki skor kepribadian kurang dari sama dengan 60, 1 kandidat dinyatakan diterima sedangkan 4 kandidat lainnya tidak diterima (Total: 5; Positif: 1; Negatif: 4), sehingga:
$$p_+ = \frac{1}{5} = 0.2$$
$$p_- = \frac{4}{5} = 0.8$$
$Entropy(Skor \ Kepribadian \leq 60) = -0.2 \log_2(0.2) - 0.8 \log_2(0.8) = \mathbf{0.722}$

Skor Kepribadian $> 60$, terdapat 5 kandidat yang memiliki skor kepribadian lebih dari 60, 5 kandidat dinyatakan diterima sedangkan 0 kandidat lainnya tidak diterima (Total: 5; Positif: 5; Negatif: 0), sehingga:
$$p_+ = \frac{5}{5} = 1$$
$$p_- = \frac{0}{5} = 0$$
$Entropy(Skor \ Kepribadian > 60) = -1 \log_2(1) - 0 \log_2(0) = \mathbf{0}$

Setelah diketahui semua nilai *Entropy* untuk atribut kepribadian, maka langkah selanjutnya adalah mencari *Information Gain* yang ditunjukkan sebagai berikut ini:
$$Gain(S, Skor \ Kepribadian) = 0.971 - (5/10 \times 0.722 + 5/10 \times 0)$$
$$Gain(S, Skor \ Kepribadian) = 0.971 - 0.361 = \mathbf{0.61}$$

Berdasarkan perhitungan *Information Gain* yang telah dilakukan, berikut ini adalah peringkat nilai *Information Gain* dari yang terbesar ke terkecil:
* Skor keterampilan memiliki *information gain* = **0.97**
* Skor wawancara memiliki *information gain* = **0.61**
* Skor kepribadian memiliki *information gain* = **0.61**
* Usia memiliki *information gain* = **0.371**

### 3. Menggambarkan struktur pohon keputusan

![Gambar 4.2 Struktur pohon keputusan prediksi perekrutan](https://i.imgur.com/example_image_recruitment_decision_tree.png)
**Gambar 4.2 Struktur pohon keputusan prediksi perekrutan**

Atribut yang memiliki nilai *Information Gain* tertinggi akan menjadi *root node*, dan atribut dengan nilai *Information Gain* berikutnya akan menjadi *internal node*, sampai kita mencapai *leaf node* yang memberikan keputusan akhir. Gambar 4.2 menggambarkan struktur pohon keputusan yang dihasilkan. Berdasarkan Gambar 4.2, Skor keterampilan memiliki atribut yang paling penting dalam menentukan keputusan penerimaan berdasarkan *dataset* yang diberikan. Skor wawancara dan Skor Kepribadian juga berpengaruh signifikan setelah Skor Keterampilan. Dan yang terakhir adalah usia juga turut berpengaruh pada keputusan penerimaan. Berikut adalah penjelasan dari Gambar 4.2:
a. *Root Node* adalah atribut Skor Keterampilan karena memiliki nilai *Information Gain* tertinggi.
b. Cabang kiri dari *root node* adalah untuk nilai Skor Keterampilan $\leq 55$ dan langsung mengarah pada keputusan Tidak Diterima karena semua data pada cabang ini memiliki keputusan 0.
c. Cabang kanan dari *root node* adalah untuk nilai Skor Keterampilan $> 55$ dan ini dibagi lagi berdasarkan atribut Skor Wawancara.
d. Jika Skor Wawancara $\leq 60$, maka kita mengecek Skor Kepribadian:
   * Jika Skor Kepribadian $\leq 60$, keputusan adalah Tidak Diterima.
   * Jika Skor Kepribadian $> 60$, keputusan adalah Diterima.
e. Jika Skor Wawancara $> 60$, keputusan langsung adalah Diterima.

Dengan pohon keputusan ini, bisa diprediksi keputusan penerimaan kandidat pelamar kerja berdasarkan nilai dari atribut-atribut yang ada. Pohon keputusan memberikan cara yang mudah dipahami untuk menginterpretasikan data dan membuat keputusan berdasarkan informasi yang paling relevan. Pohon keputusan ini dapat digunakan sebagai model prediktif untuk mempermudah proses seleksi calon yang sesuai dengan kriteria yang telah ditentukan. Hal ini juga menunjukkan bahwa metode manual untuk menghitung *Entropy* dan *Information Gain* dapat membantu dalam memahami proses pembentukan model prediktif.

## 4.6 K-Nearest Neighbor

*K-Nearest Neighbors* (KNN) adalah salah satu metode klasifikasi yang digunakan dalam pengembangan model prediktif (Zhang, 2016). KNN termasuk dalam kategori *supervised learning* yang berarti algoritma ini memerlukan data latih yang telah diberi label untuk mempelajari pola yang ada (Ying, et al., 2021). Prinsip dasar dari KNN adalah menentukan kelas suatu data uji berdasarkan kelas mayoritas dari K tetangga terdekatnya (Shokrzade, et al., 2021). Untuk mengukur kedekatan atau kemiripan antara data uji dan data latih, digunakan jarak *Euclidean*, yang menghitung jarak linear antara dua titik dalam ruang atribut multidimensi. Berikut ini merupakan rumus *Euclidean* untuk mengukur jarak terdekat (Hu, et al., 2016):
$$d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2} \qquad (4.3)$$

Langkah pertama dalam penerapan KNN adalah menentukan nilai K, yaitu jumlah tetangga terdekat yang akan dipertimbangkan. Setelah itu, dihitung jarak *Euclidean* antara data uji dan setiap data latih. Kemudian, data latih diurutkan berdasarkan jarak tersebut, dan K data latih dengan jarak terdekat dipilih. Kelas yang paling sering muncul di antara K tetangga tersebut menjadi prediksi untuk data uji (Boateng, et al., 2020). Kelebihan utama KNN adalah kesederhanaannya dan kemampuannya untuk menangani data non-linear. Sedangkan kelemahannya termasuk sensitif terhadap nilai K yang dipilih dan dapat menjadi tidak efisien ketika menghadapi *dataset* besar (Suyanto, 2017).

## 4.7 Contoh Manual Model Prediksi Menggunakan K-NN

Dalam studi kasus pengembangan model prediktif, seperti penilaian kelayakan calon karyawan berdasarkan usia, skor wawancara, skor keterampilan, dan skor kepribadian, KNN dapat membantu memprediksi keputusan penerimaan dengan mempertimbangkan pola dari data karyawan sebelumnya. Proses ini melibatkan perhitungan jarak untuk menentukan kedekatan antara kandidat pelamar kerja dan karyawan yang sudah ada, sehingga memberikan keputusan yang didasarkan pada informasi empiris yang ada.

Penggunaan *dataset* pada Tabel 4.2 sebagai data latih untuk memberikan hasil klasifikasi pada data uji yang terdapat di Tabel 4.3 di bawah ini. Pada Tabel 4.3, nilai pada kolom keputusan masih tanda tanya (?) dikarenakan belum bisa ditentukan keputusannya (Diterima 1/Tidak Diterima 0). Untuk mengisi kolom keputusan ini maka dapat digunakan algoritma K-NN.

**Tabel 4.3 Data uji untuk prediksi perekrutan karyawan**

| usia | skor wawancara | skor keterampilan | skor kepribadian | keputusan |
| :--- | :---: | :---: | :---: | :---: |
| 31 | 65 | 75 | 80 | ? |

Berikut ini langkah-langkah menggunakan K-NN:
1. Tentukan nilai parameter K, misalnya kita pilih $K = 3$
2. Hitunglah jarak *Euclidean* untuk tiap atribut terhadap data uji

Rumus *Euclidean* sesuai *dataset*:
$$d = \sqrt{(Usia_u - Usia_i)^2 + (SkorW_u - SkorW_i)^2 + (SkorKet_u - SkorKet_i)^2 + (SkorKep_u - SkorKep_i)^2}$$

dengan $Usia_u, SkorW_u, SkorKet_u, SkorKep_u$ adalah nilai atribut dari data uji sedangkan $Usia_i, SkorW_i, SkorKet_i, SkorKep_i$ adalah nilai atribut data latih, sehingga:

* **Data latih 1**
  $d_1 = \sqrt{(31 - 25)^2 + (65 - 70)^2 + (75 - 78)^2 + (80 - 91)^2}$
  $d_1 = \sqrt{6^2 + (-5)^2 + (-3)^2 + (-11)^2}$
  $d_1 = \sqrt{36 + 25 + 9 + 121} = \sqrt{191} = \mathbf{13.82}$

* **Data latih 2**
  $d_2 = \sqrt{(31 - 35)^2 + (65 - 75)^2 + (75 - 80)^2 + (80 - 80)^2}$
  $d_2 = \sqrt{(-4)^2 + (-10)^2 + (-5)^2 + 0^2}$
  $d_2 = \sqrt{16 + 100 + 25 + 0} = \sqrt{141} = \mathbf{11.87}$

* **Data latih 3**
  $d_3 = \sqrt{(31 - 25)^2 + (65 - 40)^2 + (75 - 40)^2 + (80 - 60)^2}$
  $d_3 = \sqrt{6^2 + 25^2 + 35^2 + 20^2}$
  $d_3 = \sqrt{36 + 625 + 1225 + 400} = \sqrt{2286} = \mathbf{47.81}$

* **Data latih 4**
  $d_4 = \sqrt{(31 - 27)^2 + (65 - 60)^2 + (75 - 80)^2 + (80 - 80)^2}$
  $d_4 = \sqrt{4^2 + 5^2 + (-5)^2 + 0^2}$
  $d_4 = \sqrt{16 + 25 + 25 + 0} = \sqrt{66} = \mathbf{8.12}$

* **Data latih 5**
  $d_5 = \sqrt{(31 - 35)^2 + (65 - 75)^2 + (75 - 80)^2 + (80 - 80)^2}$
  $d_5 = \sqrt{(-4)^2 + (-10)^2 + (-5)^2 + 0^2}$
  $d_5 = \sqrt{16 + 100 + 25 + 0} = \sqrt{141} = \mathbf{11.87}$

* **Data latih 6**
  $d_6 = \sqrt{(31 - 40)^2 + (65 - 40)^2 + (75 - 55)^2 + (80 - 60)^2}$
  $d_6 = \sqrt{(-9)^2 + 25^2 + 20^2 + 20^2}$
  $d_6 = \sqrt{81 + 625 + 400 + 400} = \sqrt{1506} = \mathbf{38.81}$

* **Data latih 7**
  $d_7 = \sqrt{(31 - 40)^2 + (65 - 75)^2 + (75 - 60)^2 + (80 - 50)^2}$
  $d_7 = \sqrt{(-9)^2 + (-10)^2 + 15^2 + 30^2}$
  $d_7 = \sqrt{81 + 100 + 225 + 900} = \sqrt{1306} = \mathbf{36.14}$

* **Data latih 8**
  $d_8 = \sqrt{(31 - 35)^2 + (65 - 75)^2 + (75 - 80)^2 + (80 - 80)^2}$
  $d_8 = \sqrt{(-4)^2 + (-10)^2 + (-5)^2 + 0^2}$
  $d_8 = \sqrt{16 + 100 + 25 + 0} = \sqrt{141} = \mathbf{11.87}$

* **Data latih 9**
  $d_9 = \sqrt{(31 - 27)^2 + (65 - 40)^2 + (75 - 55)^2 + (80 - 60)^2}$
  $d_9 = \sqrt{4^2 + 25^2 + 20^2 + 20^2}$
  $d_9 = \sqrt{16 + 625 + 400 + 400} = \sqrt{1441} = \mathbf{37.93}$

* **Data latih 10**
  $d_{10} = \sqrt{(31 - 30)^2 + (65 - 60)^2 + (75 - 40)^2 + (80 - 50)^2}$
  $d_{10} = \sqrt{1^2 + 5^2 + 35^2 + 30^2}$
  $d_{10} = \sqrt{1 + 25 + 1225 + 900} = \sqrt{2151} = \mathbf{46.39}$

3. Urutkan objek terdekat berdasarkan jarak *Euclidean* terkecil ke terbesar. Tabel 4.4 menunjukkan urutan rangking berdasarkan jarak *Euclidean* yang telah dihitung pada poin 2 sebelumnya.

**Tabel 4.4 Urutan rangking jarak *Euclidean***

| data latih ke-i | *distance* | rangking | keputusan |
| :--- | :---: | :---: | :---: |
| 4 | 8.12 | 1 | 1 |
| 2 | 11.87 | 2 | 1 |
| 5 | 11.87 | 3 | 1 |
| 8 | 11.87 | 4 | 1 |
| 1 | 13.82 | 5 | 1 |
| 7 | 36.14 | 6 | 1 |
| 6 | 38.81 | 7 | 0 |
| 9 | 37.93 | 8 | 0 |
| 10 | 46.39 | 9 | 0 |
| 3 | 47.81 | 10 | 0 |

4. Pilihlah objek terdekat dengan jumlah K yang sudah ditentukan. Pada poin 1 kita sudah menentukan $K = 3$, sehingga data yang ditampilkan adalah rangking 1 sampai 3 saja sesuai yang ditunjukkan Tabel 4.5.

**Tabel 4.5 Rangking sesuai jumlah K**

| data latih ke-i | *distance* | rangking | keputusan |
| :--- | :---: | :---: | :---: |
| 4 | 8.12 | 1 | 1 |
| 2 | 11.87 | 2 | 1 |
| 5 | 11.87 | 3 | 1 |

5. Gunakan kategori kelas mayoritas untuk menentukan hasil klasifikasi. Dari tiga tetangga terdekat, semua memiliki keputusan 1 (Diterima). Oleh karena itu, keputusan untuk data uji adalah: 1 (Diterima) sesuai yang ditunjukkan Tabel 4.6.

**Tabel 4.6 Hasil klasifikasi menggunakan KNN**

| Usia | Skor Wawancara | Skor Keterampilan | Skor Kepribadian | Keputusan |
| :--- | :---: | :---: | :---: | :---: |
| 31 | 65 | 75 | 80 | 1 |

Dengan menggunakan metode *K-Nearest Neighbors* (KNN) pada *dataset* yang diberikan, dihitung jarak *Euclidean* antara data uji dan setiap data latih. Setelah mengurutkan jarak tersebut, dipilih K tetangga terdekat dan menentukan keputusan berdasarkan mayoritas dari tetangga terdekat tersebut. Keputusan untuk data uji dalam contoh ini adalah diterima (1).

Studi kasus di atas menggunakan dua algoritma pemodelan prediktif, yaitu *Decision Tree* dan *K-Nearest Neighbors* (KNN). Kedua algoritma ini digunakan untuk menganalisis dan memprediksi keputusan perekrutan karyawan berdasarkan beberapa atribut seperti usia, skor wawancara, skor keterampilan, dan skor kepribadian. Melalui perhitungan manual, telah dilihat bagaimana masing-masing algoritma berfungsi dan bagaimana matematika digunakan dalam proses tersebut. Melalui studi kasus ini, dapat dilihat bagaimana matematika memainkan peran kunci dalam algoritma pemodelan prediktif. *Decision Tree* menggunakan konsep *Entropy* dan *Information Gain* untuk membuat keputusan yang optimal di setiap node, sementara KNN menggunakan jarak *Euclidean* untuk mengukur kemiripan dan menentukan kelas berdasarkan tetangga terdekat. Pemahaman dan penerapan matematika dalam algoritma ini memungkinkan untuk dibangun model prediktif yang akurat dan dapat diandalkan, yang penting untuk pengambilan keputusan dalam berbagai domain, termasuk perekrutan karyawan.