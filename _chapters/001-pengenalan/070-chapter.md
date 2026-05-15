---
slug: bab-1-7
title: "Jia Bin Huang DeepSeek V4"
layout: chapter
---

# Arsitektur Jaringan Saraf Mendalam: Evolusi dan Optimalisasi Koneksi Residual pada Model Transformer

## 1. Pendahuluan: Jaringan Mendalam dan Koneksi Residual Standar
Kekuatan representasional dari sebuah jaringan saraf mendalam (*deep neural network*) berasal dari penumpukan lapisan-lapisan non-linear. Secara intuitif, model yang lebih dalam seharusnya dapat mempelajari representasi yang lebih kompleks, ekspresif, dan menghasilkan performa yang lebih baik. Namun, pada kenyataannya, jaringan yang dalam sangat sulit untuk dilatih. Ketika kita menumpuk terlalu banyak lapisan, model sering kali mengalami peningkatan kesalahan pelatihan (*training error*) yang berujung pada performa buruk pada data uji. 

Hal ini kontraintuitif karena secara teoritis, kita bisa membangun model yang lebih dalam dengan performa setidaknya sama dengan model dangkal, yaitu dengan menyalin lapisan dari jaringan dangkal dan menggunakan lapisan tambahan untuk melakukan pemetaan identitas (*identity mapping*). Artinya, solusi tersebut sebenarnya ada, tetapi proses optimisasinya yang sulit.

Pada arsitektur *Transformer* modern—yang memproses sekuens *embedding* token input berukuran $T \times d$ (di mana $T$ adalah panjang sekuens dan $d$ adalah dimensi *embedding*) menggunakan tumpukan lapisan *attention* dan *mixture of experts*—masalah ini diatasi menggunakan strategi **Koneksi Residual** (*Residual Connections*).

### Mengapa Koneksi Residual Bekerja?
Daripada berharap bahwa lapisan-lapisan akan langsung mempelajari pemetaan yang diinginkan, kita merancangnya untuk mempelajari pemetaan residual, yaitu selisih antara input dan output yang diinginkan. Mari kita asumsikan lapisan-lapisan jaringan ini dilambangkan sebagai $f_1, f_2$, dan seterusnya, masing-masing dengan parameter $W_l$.

Dengan koneksi residual, input untuk lapisan kedua, $h_2$, dihitung dengan menjumlahkan input lapisan pertama ($h_1$) dan output dari pemetaan lapisan pertama ($f_1(h_1)$):
$$h_2 = h_1 + f_1(h_1)$$

Koneksi *skip* (langsung) ini membawa input asli ke depan tanpa perubahan. Jika kita menjabarkannya secara rekursif untuk lapisan ketiga ($h_3$):
$$h_3 = h_2 + f_2(h_2)$$
Substitusi nilai $h_2$ dari persamaan pertama menghasilkan:
$$h_3 = h_1 + f_1(h_1) + f_2(h_2)$$

Berlanjut ke lapisan keempat ($h_4 = h_3 + f_3(h_3)$), polanya menjadi jelas. Representasi pada lapisan dalam $L$ mana pun adalah representasi pada lapisan yang lebih dangkal $l$, ditambah jumlah dari seluruh transformasi di antara keduanya:
$$h_L = h_l + \sum_{i=l}^{L-1} f_i(h_i)$$

Pemetaan identitas ini memastikan fitur dapat dipropagasi langsung dari satu unit ke unit lainnya. Saat melakukan propagasi balik (*backpropagation*), gradien yang mengalir ke lapisan awal memiliki dua komponen: satu melalui koneksi *skip* (langsung tanpa perkalian atau atenuasi), dan satu lagi melalui blok transformasi. Tidak peduli seberapa dalam jaringannya, selalu ada "jalan tol" yang bersih untuk gradien. Akibatnya, koneksi residual membantu mengurangi masalah gradien menghilang (*vanishing gradients*) dan memungkinkan pelatihan model yang sangat dalam.

---

## 2. Keterbatasan Koneksi Residual: Dilusi Pre-Norm dan Leher Botol Struktural
Meskipun efektif, koneksi residual memiliki dua tantangan utama pada *Transformer* berskala besar.

**Pertama, Masalah Dilusi *Pre-Norm* (*Pre-norm Dilution*).** 
*Transformer* modern menggunakan normalisasi *pre-layer* untuk menstabilkan pelatihan. Namun, setiap lapisan menambahkan koreksi di atas output sebelumnya. Akumulasi ini menyebabkan magnitudo representasi secara alami membesar seiring bertambahnya kedalaman. Akibatnya, kontribusi dari lapisan baru menjadi semakin menyusut (terdilusi) relatif terhadap aliran residual yang terus membesar. Lapisan yang lebih dalam akhirnya dipaksa memproduksi output dengan magnitudo yang sangat besar. Meskipun input dinormalisasi, gradien menjadi tidak seimbang: lapisan awal menerima gradien yang relatif besar, sementara lapisan dalam menerima gradien yang kecil. Ketidakseimbangan ini menyebabkan penggunaan kedalaman jaringan menjadi tidak efektif.

**Kedua, Leher Botol Struktural (*Structural Bottleneck*).**
Melihat kembali persamaan $h_l = h_{l-1} + f_{l-1}(h_{l-1})$, setiap lapisan baru hanya melihat representasi tunggal dari $h_{l-1}$. Segala informasi sebelumnya telah dikompresi menjadi satu vektor, menyebabkan informasi hilang secara ireversibel pada setiap langkah. 

Fenomena ini serupa dengan pemodelan sekuens pada *Recurrent Neural Networks* (RNN) atau LSTM, di mana *hidden state* $h_t = f_w(x_t, h_{t-1})$ menjadi kompresi dari seluruh sejarah observasi, menciptakan leher botol. Masalah pada RNN ini dipecahkan oleh mekanisme *Attention* (Perhatian), di mana setiap output secara dinamis memperhatikan seluruh input yang relevan tanpa leher botol atau kehilangan informasi. Jika kita memutar jaringan residual sebesar 90 derajat, kita melihat struktur leher botol yang sama.

---

## 3. Solusi Pertama: *Attention Residuals*
Untuk memperbaiki koneksi residual dengan cara yang sama seperti mekanisme *attention* memperbaiki RNN, muncullah gagasan ***Attention Residuals***. Ide intinya adalah membuat bobot koneksi residual—yang awalnya bernilai konstan 1—menjadi parameter yang dapat dipelajari (*learnable*) dan bergantung pada data (*data-dependent*).

Dalam koneksi residual standar: $h_2 = 1 \cdot h_1 + 1 \cdot f_1(h_1)$.
Dalam *Attention Residuals*, kita menggunakan bobot $\alpha$:
$$h_2 = \alpha_{0,2}h_1 + \alpha_{1,2}f_1(h_1)$$
Di sini, $\alpha_{0,2}$ mengontrol seberapa besar $h_1$ berkontribusi ke lapisan dua, dan $\alpha_{1,2}$ mengontrol kontribusi $f_1(h_1)$.

### Mekanisme Komputasi
Nilai $\alpha$ dihitung menggunakan mekanisme *attention*. 
1. **Query:** Setiap lapisan memiliki vektor *query* berdimensi $d$ yang dapat dilatih, misalnya $w_2$ untuk lapisan kedua.
2. **Key & Value:** Vektor *Key* dan *Value* berasal dari output lapisan-lapisan sebelumnya (misal: $h_1$ dan $f_1(h_1)$).
3. Karena magnitudo output lapisan bisa sangat bervariasi, vektor *Key* harus **dinormalisasi** sebelum dihitung produk titik (*dot product*)-nya dengan vektor *Query*.
4. Skor kecocokan dinormalisasi menggunakan fungsi *Softmax* sehingga total bobot menjumlah menjadi satu ($\sum \alpha = 1$). Output dihasilkan melalui kombinasi linear dari vektor *Value* menggunakan bobot $\alpha$ tersebut.

Secara umum, persamaannya adalah:
$$h_{l} = \sum_{i} \alpha_{i,l} v_i$$
Setiap lapisan kini menjadi campuran berbobot dari seluruh output sebelumnya. Sebagai detail implementasi yang krusial, vektor *query* yang dapat dilatih ini **diinisialisasi dengan nilai nol**. Ini menjamin bahwa pada awal pelatihan, *attention* hanya berupa rata-rata seragam yang strukturnya mirip dengan koneksi residual standar.

### Hasil dan Keunggulan
*Attention Residuals* mencapai *validation loss* yang lebih rendah dibanding *baseline*, dan dinamika pelatihan menjadi jauh lebih stabil. Penggunaan bobot yang totalnya menjumlah menjadi satu mencegah akumulasi magnitudo yang tak terbatas (mengatasi *Pre-norm dilution*). Akibatnya, magnitudo gradien menjadi lebih seimbang di seluruh lapisan.

Pendekatan di mana setiap lapisan melihat *seluruh* lapisan sebelumnya disebut **Full Attention Residuals**. Konsep ini sebenarnya bukan hal yang benar-benar baru; pendekatan memberikan akses lapisan dalam ke representasi awal telah dicoba pada arsitektur seperti *DenseNet* (konkatenasi + konvolusi), *DenseFormer* (bobot agregasi yang dipelajari), dan *DeepCrossAttention* (kombinasi dependen-dimensi). Namun, kekuatan sesungguhnya dari makalah ini terletak pada implementasi infrastruktur skalabilitasnya.

---

## 4. Optimalisasi Skala Besar: *Block Attention Residuals* dan Distribusi *Pipeline*
Pendekatan *Full Attention Residuals* mengharuskan seluruh output lapisan sebelumnya tetap hidup di memori selama proses maju (*forward pass*) dan mundur (*backward pass*), serta menuntut setiap output ditransmisikan melintasi GPU dalam pelatihan terdistribusi. Hal ini menciptakan beban memori dan komunikasi yang tidak layak untuk model raksasa. (Biasanya, sistem menyimpan sebagian kecil aktivasi dan menghitung ulang sisanya saat propagasi balik, tetapi pendekatan penuh memecah efisiensi ini).

### Kompresi dengan *Block Attention Residuals*
Alih-alih memperhatikan setiap individu lapisan secara resolusi penuh, lapisan dikelompokkan ke dalam blok (misal: blok 1 berisi lapisan 1-4, blok 2 berisi lapisan 5-8). Setiap "Ringkasan Blok" (*Block Summary*, dilambangkan dengan $B_i$) menyimpan hasil penjumlahan dari output lapisan di dalam blok tersebut. Lapisan selanjutnya hanya perlu memperhatikan ringkasan blok komplit, serta "Ringkasan Parsial" (*partial summary*, dilambangkan dengan $b_i^j$) dari blok yang sedang diproses. Akumulasi blok ini terjadi secara *online* tanpa memori ekstra.

Dengan hanya menggunakan 8 blok, *Block Attention Residuals* mampu mengembalikan hampir seluruh performa dari resolusi penuh. Model Kimi-linear 48B dengan metode ini terbukti melampaui *baseline* dalam seluruh evaluasi, terutama pada tugas penalaran multi-langkah (*multi-step reasoning*). Model mencapai kerugian (*loss*) yang sama dengan pengurangan komputasi sebesar 20% (keunggulan komputasi 1,25x). Pemetaan distribusinya menunjukkan pola diagonal yang kuat (fokus pada pendahulu langsung) serta pola non-diagonal selektif (memanfaatkan koneksi *skip*). Menariknya, eksperimen membuktikan arsitektur ini lebih memilih jaringan yang lebih dalam dan sempit; penambahan kedalaman dimanfaatkan jauh lebih efektif ketimbang pada residual standar.

### Optimalisasi Infrastruktur (Paralelisme *Pipeline*)
Untuk melatih model puluhan miliar parameter, jaringan dipotong melintasi berbagai dimensi:
*   **Data Parallel:** Membagi dimensi *batch* ke beberapa GPU.
*   **Tensor Parallel:** Membagi matriks bobot di sepanjang dimensi lebar (*width*).
*   **Pipeline Parallel:** Memotong dimensi kedalaman, di mana setiap GPU memegang grup lapisan berurutan.

Pada *Pipeline Parallelism* naif, ketika satu GPU bekerja, GPU lain menganggur, menciptakan "gelembung *pipeline*" (*pipeline bubbles*). Ini diatasi dengan pemotongan *batch* menjadi *micro-batch* (GPU langsung memproses *micro-batch* berikutnya tanpa menunggu seluruh maju selesai) dan penggunaan **Tahapan Virtual** (*Virtual Stages*), di mana satu GPU memegang grup lapisan yang terpisah-pisah (misal GPU 1 memegang lapisan 1,2 dan 9,10).

Namun, saat menggunakan *Block Attention Residuals*, memindahkan data melintasi GPU menciptakan masalah baru: ringkasan seluruh blok sebelumnya harus ikut dikirim bersama *hidden state*, meningkatkan beban komunikasi secara ekstrem. Solusinya adalah ***Cross-stage caching*** (Penyimpanan Lintas-tahap). Karena GPU tertentu (melalui siklus tahapan virtual) sebenarnya sudah menyimpan ringkasan blok dari proses sebelumnya, GPU tidak perlu meminta transfer ulang. Metode ini sukses mengurangi jumlah transfer ringkasan blok lintas-batas GPU dari 19 menjadi 11 (pengurangan beban komunikasi antar-GPU sebesar 42%).

### Efisiensi *Inference* (Inferensi)
Komputasi *attention* yang melibatkan ringkasan blok utuh dan jumlah parsial dieksekusi melalui strategi komputasi dua fase:
1. **Fase 1:** Karena vektor *query* adalah parameter yang tidak bergantung pada aktivasi saat ini, skor dan bobot *attention* untuk blok komplit dihitung menggunakan perkalian matriks *batch* (*batch matrix multiplication*).
2. **Fase 2:** *Attention* yang melibatkan jumlah parsial yang terus berkembang dihitung secara sekuensial. Kedua fase ini kemudian digabungkan secara efisien menggunakan *Online Softmax*.

---

## 5. Solusi Kedua: Ekspansi Alur dengan *Hyper-Connections*
Pendekatan alternatif lain untuk memperbaiki koneksi residual diajukan oleh ByteDance (dan diuji coba oleh DeepSeek) yang disebut **Hyper-Connections**. Jika koneksi residual standar hanya memiliki satu alur residual tunggal, *Hyper-Connections* mengekspansi fitur input sebanyak $n$ kali lipat, seolah-olah memiliki $n$ aliran (*streams*) residual paralel. 

Karena memproses lapisan sebanyak $n$ kali terlalu mahal, metodologinya dimodifikasi:
1. **Agregasi Fitur:** Input dari aliran residual (misal: $x_1, x_2, x_3, x_4$) diagregasi menjadi kombinasi berbobot. Representasi input lapisan di sini adalah matriks $X_l$ berukuran $n \times d$ (di mana $n$ adalah laju ekspansi, misalnya $n=4$). Hal ini diformulasikan melalui perkalian matriks.
2. **Pemrosesan:** Lapisan memproses fitur teragregasi ini menjadi output tunggal.
3. **Ekspansi:** Output tunggal ini diekspansi kembali menjadi $n$ aliran dengan mengalikannya menggunakan vektor bobot berukuran $n \times 1$, serta skala yang dapat dipelajari, lalu dijumlahkan kembali ke jalur residual masing-masing.
4. **Pencampuran Fitur (*Feature Mixing*):** Untuk memastikan terjadinya pertukaran informasi antar-aliran residual yang kaya, diperkenalkan transformasi linear menggunakan matriks bobot $n \times n$. Matriks ini bertindak sebagai *router* fitur, membentuk pola kombinasi linear yang fleksibel lintas aliran.

Hebatnya, pemetaan linear ini tidak diterapkan secara kaku (*static mappings*), melainkan dikondisikan secara dinamis berdasarkan input lapisan tersebut (*dynamic mappings*). Evaluasi awal *Hyper-Connections* menunjukkan konvergensi hingga 1,8 kali lebih cepat dan akurasi yang lebih tinggi dibanding *baseline*.

---

## 6. Stabilisasi Pelatihan *Hyper-Connections* (Studi Kasus DeepSeek)
Ketika tim DeepSeek mencoba mengadaptasi *Hyper-Connections* untuk pelatihan model mereka, mereka menemukan **ketidakstabilan pelatihan yang parah**.

### Akar Masalah Ketidakstabilan
Mari kita jabarkan fitur lapisan menggunakan pemetaan rekursif. Dalam residual standar, lapisan dalam mendapatkan pemetaan identitas ($h_{l} = h_{l-1} + f(...)$). Namun, pada *Hyper-Connections*, persamaan direpresentasikan dalam dua suku:
1. Suku yang merupakan fitur dari lapisan dangkal yang ditransformasi **berturut-turut** oleh matriks *feature mixing* di semua lapisan perantara.
2. Suku yang merupakan jumlahan output seluruh fungsi residual sebelumnya.

Masalahnya terletak pada suku pertama: kita memiliki pemetaan komposit (*composite mapping*), bukan identitas murni. Matriks matriks pencampuran fitur tidak dibatasi, sehingga elemen-elemennya dapat bernilai sembarang. Sebagai ilustrasi matematis satu dimensi: Jika nilai awal adalah 1 dan pemetaan perantara mensyaratkan perkalian dengan skalar $h$. Jika $h=1$, output tetap 1 (identitas). Jika $h=1,1$, maka melewati banyak lapisan, output akan mengalami peningkatan hingga 100 kali lipat (gradien meledak). Jika $h < 1$, nilai meluruh dengan cepat (gradien menghilang). Jika $h$ negatif, output akan berosilasi secara ekstrem.

### Solusi Matriks Stokastik Ganda
Untuk menstabilkan pelatihan, matriks pencampuran ini harus dijinakkan. Solusi intinya adalah memaksa matriks pencampuran fitur berukuran $n \times n$ ini menjadi **Matriks Stokastik Ganda (*Doubly Stochastic Matrix*)** atau *Birkhoff polytope*. Syarat matriks ini adalah: semua elemen bernilai positif, dan jumlah setiap baris serta setiap kolom harus sama dengan satu. Hal ini menjamin pemetaan komposit melintasi berlapis-lapis jaringan tetap menjadi stokastik ganda, mencegah masalah meledak atau menghilangnya gradien.

Bagaimana cara membuat matriks bebas menjadi stokastik ganda?
Pendekatan yang digunakan adalah **Algoritma Sinkhorn-Knopp** (atau secara akademis: memproyeksikan matriks pencampuran fitur ke *manifold* matriks stokastik ganda).
1. Seluruh elemen matriks dieksponensialkan (menggunakan fungsi *eksponensial*) untuk memastikan setiap nilai positif secara tegas dan meningkat secara monoton.
2. Menggunakan algoritma iteratif sederhana: matriks diubah skala kolomnya agar berjumlah satu, namun ini merusak jumlahan baris. Oleh karena itu, diubah pula skala barisnya agar berjumlah satu (yang kembali merusak jumlahan kolom). Reskalaan bergantian (*alternating rescaling*) ini diterapkan secara iteratif, dan hanya dalam beberapa literasi saja matriks akan sangat mendekati sifat stokastik ganda.

### Penyesuaian Tambahan dari DeepSeek
DeepSeek juga melakukan perubahan parameterisasi pada matriks lainnya dibandingkan makalah asli *Hyper-Connections*:
1. **Perubahan Fungsi Aktivasi:** Mengganti aktivasi dari *tanh* menjadi *sigmoid*. Hal ini secara tegas menghindari koefisien negatif dan membatasi (*bounding*) bobot agregasi dan ekspansi pada rentang maksimum tertentu (tidak bisa lebih besar dari 1 atau 2).
2. **Skalar Pengali ($2\times$):** Parameterisasi yang disesuaikan ini dikalikan dengan skalar 2 (di mana bobot $\alpha$ dan bias $b$ diinisialisasi mendekati nol, sehingga input ke *sigmoid* mendekati nol, menghasilkan output 0,5. Dikalikan 2 hasilnya adalah 1). Desain ini memastikan bahwa pada langkah pertama pelatihan, *Hyper-connections* berperilaku tepat sama dengan koneksi residual standar.

### Efisiensi Memori pada *Hyper-Connections*
Meski menstabilkan pelatihan, mengekspansi aliran residual menjadi $n$ (misal 4 kali lipat) menciptakan jejak memori GPU yang besar dan memperlambat pelatihan karena akses *Input/Output* (I/O) memori yang berlebihan. DeepSeek merancang tiga infrastruktur efisien untuk mengatasinya:
1. Menyusun ulang (*reorder*) normalisasi, menggabungkan (*fuse*) operasi dengan akses memori bersama, dan menulis *kernel* perangkat lunak khusus untuk mengurangi redundansi akses memori.
2. Membebaskan/menghapus aktivasi perantara setelah proses maju (*forward pass*) selesai, lalu menghitungnya ulang (*recompute*) pada proses mundur (*backward pass*) sesuai kebutuhan. Ini dipadukan dengan sinkronisasi ukuran blok dan *pipeline* untuk menghemat pemakaian memori.
3. Menjadwalkan eksekusi *pipeline* dan *kernel* secara *overlapping* (tumpang tindih) antara komputasi dan komunikasi antar-perangkat, untuk memaksimalkan utilisasi perangkat keras.

Melalui optimalisasi menyeluruh ini, *overhead* komputasi pelatihan pada laju ekspansi ($n=4$) sukses ditekan sehingga hanya meningkat sebesar 6,7%.

---

## 7. Kesimpulan
Koneksi residual standar telah menjadi landasan jaringan saraf tanpa banyak perubahan dalam waktu yang cukup lama. Upaya-upaya modern seperti **Attention Residuals** (menggunakan mekanisme perhatian untuk menimbang secara dinamis seluruh lapisan sebelumnya sekaligus mengatasi dilusi *pre-norm*) serta **Hyper-Connections** (memecah jalur residual menjadi aliran multijalur berarsitektur matriks stokastik ganda) menunjukkan masa depan inovasi arsitektur yang menjanjikan. Dengan implementasi teknik sistem tingkat lanjut seperti paralelisasi *pipeline*, *cross-stage caching*, serta optimasi I/O memori tingkat *kernel*, kompleksitas algoritmik baru ini berhasil dijalankan pada skala pelathan model raksasa secara stabil, efisien, dan menorehkan akurasi luar biasa.