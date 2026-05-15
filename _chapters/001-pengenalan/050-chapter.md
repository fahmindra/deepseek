---
slug: bab-1-5
title: "Jia Bin Huang DSA"
layout: chapter
---


# Membedah Mekanisme *DeepSeek Sparse Attention* (DSA) dan Evolusi Efisiensi *Attention* pada Model Bahasa Besar (LLM)

Pada akhir September 2025, DeepSeek merilis sebuah model eksperimental terbaru. Model mutakhir ini memperkenalkan sebuah varian mekanisme *attention* (perhatian) baru yang dikenal sebagai **DeepSeek Sparse Attention (DSR/DSA)**. Dengan memanfaatkan arsitektur ini, model tersebut mampu menyamai performa versi-versi pendahulunya, namun secara drastis mengurangi biaya komputasi, yang pada akhirnya memangkas harga API hingga 50%. 

Untuk memahami bagaimana mekanisme *attention* dapat dibuat sedemikian efisien, kita perlu membedah metode ini dari prinsip-prinsip dasarnya (*first principles*). 

## 1. Pemrosesan Bahasa Dasar: Tokenisasi dan *Embedding*
Pada tingkat abstraksi tertinggi, ketika Model Bahasa Besar (LLM) menerima masukan (*input*), ia akan menghasilkan sebuah respons keluaran (*output*). Pertanyaannya: bagaimana model tersebut memahami bahasa?

Langkah paling pertama adalah memecah teks ke dalam potongan-potongan kecil yang disebut **token**. Setiap token kemudian ditugaskan pada sebuah ID unik di dalam sebuah kamus. Proses ini dikenal sebagai **tokenisasi**. 

Namun, ID token itu sendiri tidak merepresentasikan makna semantik dari kata tersebut. Oleh karena itu, kita memetakan setiap individu token ke dalam sebuah vektor berdimensi $d$, yang disebut sebagai ***token embedding***. Di dalam ruang berdimensi $d$ ini, kata-kata yang memiliki makna semantik serupa akan memiliki representasi vektor *embedding* yang saling berdekatan. Walaupun demikian, *token embedding* ini berdiri sendiri dan tidak memiliki informasi apapun mengenai konteks kalimat di sekitarnya. Di sinilah mekanisme *attention* berperan.

## 2. Dasar Mekanisme *Self-Attention*
Mekanisme *attention* memungkinkan model untuk menentukan seberapa besar kontribusi setiap token dalam mengumpulkan informasi kontekstual dari seluruh sekuens masukan. Kita mengukur hubungan antar-token di dalam sebuah ruang fitur yang dipelajari melalui matriks *Query* dan matriks *Key*.

Matriks *Query* ($W_Q$) memiliki dimensi $d \times d_k$, di mana $d$ adalah ukuran *embedding* (sekitar 7000 pada arsitektur DeepSeek V3 atau R1), dan $d_k$ adalah dimensi dari vektor *Query*. Kita dapat menghitung vektor *Query* dengan mengalikan setiap *token embedding* dengan matriks $W_Q$. Secara matematis, perkalian antara vektor *embedding* berukuran $1 \times d$ dengan matriks berukuran $d \times d_k$ akan menghasilkan vektor *Query* berukuran $1 \times d_k$. 

Dengan cara yang serupa, kita menghitung vektor *Key* dengan mengalikan setiap *token embedding* dengan matriks *Key* ($W_K$).

Kita kemudian mengukur tingkat relevansi dari satu token ke token lainnya menggunakan perkalian titik (*dot product*) vektor antara vektor *Query* dan vektor *Key* yang telah diekstraksi. Nilai *dot product* ini merepresentasikan seberapa mirip kedua vektor tersebut.

Karena model menghasilkan responsnya satu per satu token—atau dengan kata lain, secara **autoregresif**—kita hanya mempertimbangkan relevansi antara token saat ini dengan seluruh token yang mendahuluinya. Mengingat *dot product* antara vektor *Query* dan *Key* dapat menghasilkan nilai positif maupun negatif, kita menerapkan fungsi **Softmax** untuk mengonversi skor mentah ini menjadi sebuah distribusi probabilitas yang dinormalisasi. Nilai yang dihasilkan dari proses ini dikenal sebagai **skor *attention***.

Skor *attention* mengindikasikan seberapa besar pengaruh yang harus dimiliki oleh setiap token masa lalu ketika model memperbarui representasi *embedding* dari token tertentu. 

Selanjutnya, untuk setiap token, kita menghitung vektor *Value* menggunakan matriks *Value* ($W_V$). Hal ini bertujuan untuk mengekstraksi informasi kontekstual yang dapat ditransfer ke token lain. Kita kemudian mengagregasi vektor-vektor *Value* ini menjadi sebuah representasi tunggal untuk setiap token dengan menghitung jumlah berbobot (*weighted sum*), di mana bobotnya ditentukan oleh skor *attention* tadi.

Namun, dimensi dari vektor *Value* ini jauh lebih kecil dibandingkan dimensi *embedding* awal $d$. Oleh karena itu, kita menggunakan matriks keluaran ($W_O$) untuk mengonversi vektor keluaran ini kembali ke dimensi *token embedding* yang sama. Representasi yang dihasilkan kemudian ditambahkan sebagai **vektor residual** ($\Delta x$) untuk memperbarui *token embedding* yang asli. 

Mekanisme *Self-Attention* inilah yang memungkinkan model untuk menggabungkan informasi kontekstual ke dalam setiap *embedding*, memastikan bahwa *embedding* yang telah diperbarui tersebut menangkap detail relevan dari keseluruhan sekuens. Operasi pada setiap lapisan *attention* di dalam model beroperasi dengan cara yang sama.

## 3. Representasi Matriks dari *Attention*
Menuliskan perhitungan untuk setiap vektor secara eksplisit akan sangat tidak efisien. Untungnya, kita dapat merepresentasikan komputasi ini dalam bentuk matriks yang ringkas.

Kita mendefinisikan tumpukan vektor *Query* sebagai matriks $Q$ dan tumpukan vektor *Key* sebagai matriks $K$. Hal ini memungkinkan kita untuk mengekspresikan seluruh *dot product* antar-pasangan vektor menggunakan perkalian matriks $QK^T$. Kita juga menambahkan matriks penopeng (*masking matrix*) $M$ untuk memastikan bahwa token hanya dapat "memperhatikan" token-token yang mendahuluinya dan dirinya sendiri.

Dengan bobot *attention* tersebut, kita mengekspresikan matriks keluaran $O$. Secara ringkas, mekanisme *attention* dapat dinotasikan sebagai fungsi dari matriks $Q, K$, dan $V$. *Embedding* residual ($\Delta x$) adalah hasil dari perkalian antara keluaran *attention* matriks $O$ dengan matriks keluaran $W_O$.

## 4. *Multi-Head Attention* (MHA)
Terdapat satu permasalahan mendasar: menggunakan satu pasang matriks *Query, Key*, dan *Value* saja tidak cukup untuk menangkap kerumitan hubungan antar-token. 

Untuk mengatasi hal ini, diperkenalkanlah beberapa set matriks *Query, Key*, dan *Value*, yang memungkinkan model untuk mempelajari dan merepresentasikan beragam jenis hubungan antar-token secara simultan. Hal ini dicapai dengan memperluas ukuran matriks-matriks tersebut dan membaginya ke dalam sejumlah *head* (kepala), yang disimbolkan dengan $h$. Dalam praktiknya, jumlah *head* dapat sangat besar, misalnya 128 pada DeepSeek V3 dan R1.

Kita membagi matriks *Query, Key*, dan *Value* ke dalam kelompok *head* masing-masing, dan menghitung mekanisme *attention* secara terpisah. Proses ini menghasilkan sekumpulan matriks keluaran, yaitu $O_1, O_2, \dots, O_h$. Keluaran dari berbagai *head* ini kemudian dikonkatenasi (digabungkan) dan dikalikan dengan matriks keluaran $W_O$ untuk menghasilkan *embedding* residual $\Delta x$. Metode ini dikenal sebagai ***Multi-Head Attention***, yang merupakan fondasi utama dari arsitektur *Transformer* di balik kesuksesan LLM modern.

## 5. *Key-Value Caching* (KV Cache)
Berkat matriks penopeng ($M$), model dapat menghitung *attention* untuk seluruh token secara bersamaan pada tahap awal yang disebut sebagai tahap ***prefilling***. 

Dinamika yang menarik terjadi pada tahap deksode atau pembentukan keluaran. Misalkan $x_7$ adalah token pertama dalam respons keluaran. Tujuan komputasi kini adalah menghitung keluaran *attention* $O_7$ menggunakan konteks yang disediakan oleh token-token pendahulunya. Sama seperti fase *prefilling*, kita mengekstraksi vektor *Query* dan *Key*, menghitung skor *attention* dengan *dot product* yang dinormalisasi fungsi *Softmax*, dan menghasilkan $O_7$ dengan menghitung jumlah berbobot dari vektor *Value*.

Selama proses ini, vektor *Key* dan *Value* dari token-token sebelumnya **tidak berubah sama sekali**. Hal ini berlaku di seluruh *attention heads* dan di seluruh lapisan *attention*. Menghitung ulang vektor-vektor ini setiap kali token baru diproduksi adalah sebuah pemborosan komputasi. 

Solusinya adalah menyimpan vektor *Key* dan *Value* dari token-token sebelumnya di dalam memori, dan menggunakannya kembali saat dibutuhkan. Teknik ini dikenal sebagai ***Key-Value Caching* (KV Cache)**. KV Cache sangat membantu menghindari komputasi yang tidak perlu dan mempercepat proses *decoding*.

Mengapa kita tidak menyetel singgahan (*cache*) untuk vektor *Query*? Karena sifat *masking*, nilai elemen vektor *Query* sebelumnya ditutupi dan tidak memiliki efek pada keluaran *attention* token saat ini. Karenanya, vektor *Query* masa lalu tidak perlu disimpan.

KV *Caching* sangat efektif, tetapi sangat boros memori (*memory intensive*). Sebagai referensi pada arsitektur DeepSeek V3 atau R1, total memori yang dibutuhkan untuk KV *cache* adalah:
**2 × dimensi KV × ukuran presisi × jumlah *head* × jumlah *layer* × panjang sekuens**.
Angka 2 muncul karena kita harus menyimpan baik vektor *Key* maupun *Value*. Untuk setiap vektor, kita menyimpan 128 nilai. Untuk presisi *Float16*, setiap nilai membutuhkan 2 *bytes*. Dengan 128 *heads* dan total 61 *layers*, jika kita memiliki sekuens sepanjang 32K token, memori total untuk KV *cache* akan mencapai **131 GB**.

## 6. *Multi-Query Attention* (MQA) dan *Grouped-Query Attention* (GQA)
Bagaimana cara mengurangi beban memori tersebut? Kita tidak bisa mengubah dimensi KV, presisi, atau jumlah *layer*. Namun, kita dapat menyesuaikan jumlah *head*.

Pendekatan paling sederhana adalah mengurangi jumlah *head* untuk matriks *Key* dan *Value* dari $h$ menjadi 1. Ini berarti model hanya memproduksi satu salinan vektor *Key* dan *Value* untuk setiap token pada suatu *layer*, dan vektor tersebut dibagi pakai (*shared*) di seluruh *attention heads* yang memproses vektor *Query*. Pendekatan ini disebut ***Multi-Query Attention* (MQA)**. 
Perbandingannya signifikan: jika MHA membutuhkan 4 MB memori per token, MQA hanya butuh 31 kB per token, setara dengan pengurangan memori 128 kali lipat. Sayangnya, MQA mengorbankan kemampuan model menangkap hubungan token yang kompleks, sehingga performanya menurun drastis dibandingkan MHA.

Sebagai alternatif kompromi, kita dapat mengurangi jumlah *head* KV menjadi sebuah nilai yang lebih kecil, direpresentasikan dengan $n_g$ (di mana $g$ berarti grup). Misalnya, jumlah grup ditetapkan menjadi 2. Model akan memproduksi dua pasang vektor *Key* dan *Value* ($K_1, K_2$ dan $V_1, V_2$). Vektor-vektor ini kemudian dibagi pakai oleh *attention heads* di dalam grup masing-masing. Pendekatan menengah ini disebut ***Grouped-Query Attention* (GQA)**. GQA menciptakan keseimbangan antara efisiensi memori (dari MQA) dan daya ekspresif (dari MHA). Ini adalah standar yang populer pada LLM masa kini seperti Llama (Meta), Qwen (Alibaba), dan Gemma (Google). Jika 128 *heads* dibagi ke dalam 16 grup, kita mendapatkan pengurangan memori sebesar 8 kali lipat dibanding MHA standar.

Dari perspektif matriks matematis, dalam GQA, proses menduplikasi vektor *Key* dan *Value* agar dapat dioperasikan bersama Multi-Head *Query* dapat ditulis sebagai perkalian matriks identitas berukuran $d_k \times d_k$, yang disebut matriks proyeksi-atas (*up-projection matrix*), $W_{K-up}$ dan $W_{V-up}$. Matriks *Key* dan *Value* dasar kemudian dikonkatenasi menjadi sebuah matriks yang disebut proyeksi-bawah (*down-projection*), $W_{KV-down}$. Konsep matematis ini menunjukkan bahwa GQA menggunakan faktorisasi peringkat-rendah (*low-rank factorization*) dari matriks aslinya secara terbatas.

## 7. *Multi-Head Latent Attention* (MLA)
Konsep faktorisasi *low-rank* ini dapat diekstensi dengan cara **melatih** matriks proyeksi-bawah dan proyeksi-atas secara langsung. 

Matriks $W_{KV-down}$ mengompresi *token embedding* ke dalam sebuah vektor fitur berdimensi rendah ($c_{KV}$). Matriks $W_{K-up}$ dan $W_{V-up}$ kemudian memetakan vektor yang telah terkompresi tersebut kembali menjadi vektor *Key* dan *Value* asli. Matriks proyeksi-atas yang dilatih secara dinamis ini memungkinkan model memproduksi vektor *Key* dan *Value* yang unik untuk setiap *attention head*. Ide inti inilah yang membentuk arsitektur ***Multi-Head Latent Attention* (MLA)**.

Dimensi dari vektor laten yang terkompresi ($d_c$) jauh lebih kecil dibanding ukuran MHA penuh. Pada DeepSeek V3/R1, dimensi $d_c$ bernilai 576. Hal ini menghasilkan pengurangan pemakaian memori hingga 57 kali lipat tanpa menurunkan performa (bahkan terbukti sedikit meningkatkan performa dibandingkan MHA konvensional). Selain pada matriks KV, kompresi laten (*low-rank*) ini juga dapat diterapkan pada matriks *Query* (menggunakan matriks $W_{Q-down}$ dan $W_{Q-up}$). Meskipun tidak mengurangi memori KV *cache* saat inferensi, ini menghemat memori aktivasi saat fase pelatihan.

### Trik Komputasi Inferensi pada MLA
Sekilas, proses *up-projection* (dekompresi) seolah menambah beban perkalian matriks saat inferensi. Namun, ada trik aljabar linier yang meniadakannya.

Ingat bahwa *embedding* residual $\Delta x$ adalah perkalian dari matriks $O$ dan $W_O$. Matriks $O$ adalah konkatenasi keluaran dari masing-masing matriks *head* ke-$i$. Secara matematis:
- Vektor *Query* ($q_i$) adalah hasil vektor *embedding* dikali $W_{Q-down}$, lalu dikali matriks komponen atasnya, $W_{Q-up}^i$.
- Vektor *Key* ($k_i$) adalah hasil dari laten $c_{KV}$ dikali matriks $W_{K-up}^i$. 

Karena operasi perhitungan *attention score* (sebelum *Softmax*) melibatkan perkalian (*dot product*) dari ketiga matriks tersebut, **sifat asosiatif perkalian matriks** memungkinkan kita untuk melebur (*absorb*) matriks $W_{Q-up}$ dan matriks $W_{K-up}$ ke dalam satu komputasi tunggal yang telah dikomputasi sebelumnya (*pre-computed*). Ini berarti kita **tidak perlu** secara eksplisit mendekonstruksi (menghitung ulang) vektor *Key* setiap kali inferensi.

Hal yang sama berlaku untuk vektor *Value*. Vektor $v_i$ dihitung dari $c_{KV}$ dikalikan matriks $W_{V-up}^i$. Matriks $W_{V-up}^i$ ini dapat langsung diserap ke dalam matriks *output* akhir, $W_O$. Dengan menyerap matriks proyeksi-atas ke dalam parameter komputasi *Query* dan matriks *Output*, MLA dapat dijalankan secara sangat efisien.

## 8. Konflik MLA dengan *Rotary Positional Embedding* (RoPE)
Namun, arsitektur MLA memiliki limitasi fundamental: MLA secara baku **tidak kompatibel** dengan ***Rotary Positional Embedding* (RoPE)**.

RoPE memungkinkan model menangkap relasi posisi antartoken melalui vektor matematika. Sebagai ilustrasi, asumsikan vektor $Q$ dan $K$ berdimensi dua. Untuk token "dog" pada posisi ke-1, RoPE mengaplikasikan sebuah rotasi sebesar $1\theta$. Jika kata "dog" bergeser ke posisi ke-2 (misal dalam frasa "my dog"), ia mendapat rotasi sebesar $2\theta$. Ketika dilakukan perhitungan *attention* (mencari sudut antara kedua vektor melalui *dot product*), hasil rotasi tersebut memastikan bahwa relevansi antartoken secara inheren **hanya bergantung pada posisi relatifnya**, tanpa memedulikan panjang konteks di depannya. Dalam implementasi nyatanya, vektor dibagi ke dalam sub-vektor dua dimensi, dikenai rotasi dengan frekuensi yang bervariasi, lalu dikonkatenasi kembali.

Ketidakcocokan RoPE dengan MLA terjadi pada persamaan matematisnya. Pada MLA, trik komputasi inferensinya mengandalkan peleburan matriks. Namun, ketika RoPE ditambahkan, **matriks rotasi disisipkan tepat di antara matriks proyeksi-atas *Query* dan proyeksi-atas *Key***. Gangguan matriks rotasi di tengah-tengah ekuasi ini merusak sifat asosiatif, sehingga matriks-matriks tersebut tidak dapat dilebur. Akibatnya, model dipaksa untuk menghitung ulang secara eksplisit seluruh vektor *Key* untuk seluruh token masa lalu di setiap langkah inferensi, yang sangat menggerus efisiensi komputasi.

Sebagai solusinya, DeepSeek menerapkan metode ***Decoupled RoPE*** (RoPE yang dipisahkan). DeepSeek memperkenalkan sekumpulan representasi vektor *Query* (dengan *multi-head*) dan representasi vektor *Key* (yang dibagikan/*shared*) **secara independen** hanya untuk menangani informasi posisi RoPE. Vektor RoPE ini kemudian dikonkatenasi (digabungkan) ke vektor original sebelum perhitungan utama MLA dilakukan. Praktik ini dikenal dengan ***Partial RoPE***.

## 9. *DeepSeek Sparse Attention* (DSA) dan *Lightning Indexer*
Sekalipun menggunakan MLA (seperti pada DeepSeek R1), model dihadapkan pada dilema sekuens konteks yang ekstrem. Semakin panjang token yang diproduksi, perhitungan beban *attention* antara token saat ini dan semua token sebelumnya semakin membebani sistem, yang menurunkan parameter *throughput* (token per detik).

Untuk mengatasi kelambatan ekstrem ini, DeepSeek mengusulkan ***Lightning Indexer*** (Pengindeks Kilat), yang mana merupakan fondasi dasar dari **DeepSeek Sparse Attention (DSA)**.

Konsep intinya adalah melakukan penilaian (evaluasi) cepat terhadap relevansi historis seluruh token sebelumnya, untuk secara eksklusif hanya menyeleksi sebagian kecil token yang paling relevan untuk diikutsertakan dalam komputasi *attention*. 

*Lightning indexer* menghitung skor indeks antara *query* saat ini dengan token sebelumnya dengan cara menyimulasikan perhitungan MHA: menggunakan *multi-head query* khusus dengan sebuah representasi *shared key*. Keduanya di-*dot product*-kan, melewati fungsi **ReLU** untuk membuang nilai negatif (menjadikannya nilai nol/*sparse*), dikalikan dengan bobot, lalu dijumlahkan. Karena ini melibatkan posisi, *Partial RoPE* tetap diaplikasikan.

Mengapa komputasi ekstra oleh *indexer* ini bisa meningkatkan efisiensi? Jawabannya ada pada operasi **Kuantisasi**. 
Vektor *Query* dan *Key* untuk *indexer* ini dikuantisasi (diturunkan kepresisiannya) menjadi format **8-bit**. Karena tujuan utama *indexer* hanyalah mendeteksi urutan peringkat relevansi (bukan menghitung probabilitas presisi), pendekatan kuantisasi *low-precision* ini sangat masuk akal dan berjalan sangat cepat di level perangkat keras.

### Mengatasi Kelemahan Kuantisasi dengan Transformasi Hadamard
Kuantisasi naif pada vektor *Query* dan *Key* dapat menyebabkan kesalahan akurasi (eror). Jika matriks memiliki varians yang tidak normal—beberapa titik nilai sangat tinggi (*spikes*) sementara sebagian lainnya hampir nol—kuantisasi ke 8-bit akan menghilangkan informasi rentang dinamisnya (*dynamic range*). 

Ide awalnya adalah mengalikan vektor dengan matriks ortogonal acak agar nilai ekstrem tersebut tersebar. Hal ini membuat nilai keluaran tersebar dengan baik tanpa ekstremitas. Akan tetapi, ini memerlukan operasi komputasi perkalian matriks padat (*dense matrix multiplication*) tambahan.

Pendekatan revolusioner yang digunakan DeepSeek adalah menggunakan **Transformasi Hadamard** (secara spesifik *Fast Walsh-Hadamard Transform*). Transformasi ini secara deterministik menyebarkan (*spread out*) lonjakan nilai ekstrem tersebut secara merata di seluruh koordinat matriks masukan (membentuk distribusi seragam). Karena memanfaatkan *Fast Walsh-Hadamard Transform*, komputasinya sangat efisien dan dapat menghindari rekonstruksi matriks padat, dengan secara murni hanya mengandalkan komputasi penjumlahan dan pengurangan yang sangat teroptimisasi pada tingkat *kernel* GPU. Pengujian pada 5.000 sampel menunjukkan bahwa penggunaan Transformasi Hadamard memberikan aproksimasi kuantisasi paling akurat dan tingkat deviasi standar paling rendah.

Dengan *Lightning Indexer* berbasis 8-bit kuantisasi Hadamard, algoritma DeepSeek Sparse Attention mengeksekusi sekuens panjang **2 hingga 3 kali lebih cepat** sembari mereduksi beban memori komputasi sebesar **30% hingga 40%**, tanpa adanya degradasi performa model sama sekali.

## 10. Proses Pelatihan (Training) *Lightning Indexer*
Bagaimana modul DSA yang kompleks ini dilatih? DeepSeek melakukannya dalam dua tahap utama:

**Tahap Pertama: Pemanasan Padat (*Dense Warm-up Stage*)**
Pada fase ini, parameter utama pada lapisan representasi *Multi-Head Latent Attention* dibekukan (*frozen*), dan model berfokus melatih parameter komputasi *lightning indexer*. Distribusi target diekstraksi dengan menjumlahkan skor rata-rata *attention* di setiap *heads*, yang diikuti dengan **Normalisasi L1** di sepanjang dimensi sekuens. Tujuan matematis tahap ini adalah memastikan bahwa perilaku pemetaan keluaran *indexer* selaras dengan representasi distribusi statistik pada model mekanika inti.

**Tahap Kedua: Pemilihan Token Halus (*Fine-grained Token Selection*)**
Setelah algoritma *indexer* disesuaikan, sistem menginisiasi optimasi pada seluruh *trainable parameters*. Di sinilah model mempelajari pola atensi terjarang (*sparse attention patterns*) yang sejati dari mekanisme DSA. Meskipun *indexer* diwajibkan menyimulasikan model rata-rata, tugas tersebut dibatasi eksklusif hanya untuk token yang diseleksi oleh *indexer* itu sendiri. 
Lebih jauh, algoritma menerapkan teknik *detach* terhadap masukan parameter *indexer* dari **grafik komputasi sistem** (*computational graph*). Hasilnya, *indexer* secara teknis dievaluasi dan dioptimasi secara terpisah menurut parameter kerugian (*loss*)-nya sendiri, sementara arsitektur inti model dioptimasi secara deterministik eksklusif pada parameter kerugian prokreasi linguistiknya (*language modeling loss*).

---
Dengan pemahaman dari prinsip fundamental ke arsitektur tingkat lanjut (dari MHA, MQA, GQA, MLA, hingga DSA), kita dapat melihat rekayasa matematis di balik efisiensi revolusioner model bahasa berskala besar seperti eksperimen mutakhir DeepSeek.