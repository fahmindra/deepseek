---
slug: mbg
layout: part
---
# Understanding LLMs from Scratch Using Middle School Math

**Artificial Intelligence**  
**Oleh: Rohit Patel**  
**19 Oktober 2024 | 52 min read**

---

### A self-contained, full explanation to inner workings of an LLM

Dalam artikel ini, kita akan membahas bagaimana Large Language Models (LLM) bekerja, mulai dari nol – dengan asumsi Anda hanya tahu cara menjumlahkan dan mengalikan dua angka. Artikel ini dimaksudkan untuk sepenuhnya mandiri. Kita mulai dengan membangun AI Generatif sederhana di atas kertas, lalu menelusuri semua yang kita butuhkan untuk memiliki pemahaman yang kuat tentang LLM modern dan arsitektur Transformer. Artikel ini akan membuang semua bahasa mewah dan jargon dalam ML dan merepresentasikan semuanya sebagaimana adanya: angka. Kami akan tetap menyebutkan istilah aslinya untuk menautkan pemikiran Anda saat membaca konten yang penuh jargon.

Beralih dari penjumlahan/perkalian ke model AI paling canggih saat ini tanpa mengasumsikan pengetahuan lain atau merujuk ke sumber lain berarti kita mencakup BANYAK hal. Ini BUKAN penjelasan LLM mainan – orang yang bertekad secara teoritis dapat menciptakan kembali LLM modern dari semua informasi di sini. Saya telah memotong setiap kata/baris yang tidak perlu dan karena itu artikel ini tidak benar-benar dimaksudkan untuk sekadar dibaca sekilas.

## Apa yang akan kita bahas?

1. Jaringan saraf sederhana (Simple neural network)
2. Bagaimana model-model ini dilatih?
3. Bagaimana semua ini menghasilkan bahasa?
4. Apa yang membuat LLM bekerja begitu baik?
5. Embeddings
6. Sub-word tokenizers
7. Self-attention
8. Softmax
9. Residual connections
10. Layer Normalization
11. Dropout
12. Multi-head attention
13. Positional embeddings
14. Arsitektur GPT
15. Arsitektur Transformer

Mari kita mulai.

Hal pertama yang perlu diperhatikan adalah jaringan saraf hanya dapat menerima angka sebagai input dan hanya dapat mengeluarkan angka. Tidak ada pengecualian. Seninya adalah mencari tahu bagaimana memasukkan input Anda sebagai angka, menafsirkan angka output dengan cara yang mencapai tujuan Anda. Dan terakhir, membangun jaringan saraf yang akan mengambil input yang Anda berikan dan memberi Anda output yang Anda inginkan (mengingat interpretasi yang Anda pilih untuk output ini). Mari kita telusuri bagaimana kita beralih dari menjumlahkan dan mengalikan angka ke hal-hal seperti Llama 3.1.

## Jaringan saraf sederhana (A simple neural network):

Mari kita kerjakan jaringan saraf sederhana yang dapat mengklasifikasikan suatu objek:

*   **Data objek yang tersedia:** Warna dominan (RGB) & Volume (dalam mililiter)
*   **Klasifikasi ke dalam:** Daun (Leaf) ATAU Bunga (Flower)

Berikut adalah tampilan data untuk daun dan bunga matahari:

![Image by author](https://towardsdatascience.com/wp-content/uploads/2024/10/1ymWXxVf7N4-Ps-62Cih2XQ.png)
*Image by author*

Sekarang mari kita bangun jaringan saraf yang melakukan klasifikasi ini. Kita perlu memutuskan skema interpretasi input/output. Input kita sudah berupa angka, jadi kita bisa memasukkannya langsung ke dalam jaringan. Output kita adalah dua objek, daun dan bunga, yang tidak bisa dikeluarkan oleh jaringan saraf secara langsung. Mari kita lihat beberapa skema yang bisa kita gunakan di sini:

*   Kita bisa membuat jaringan mengeluarkan satu angka. Dan jika angka tersebut positif kita katakan itu daun dan jika negatif kita katakan itu bunga.
*   ATAU, kita bisa membuat jaringan mengeluarkan dua angka. Kita menafsirkan yang pertama sebagai angka untuk daun dan yang kedua sebagai angka untuk bunga, dan kita akan mengatakan bahwa pilihannya adalah angka mana yang lebih besar.

Kedua skema tersebut memungkinkan jaringan untuk mengeluarkan angka yang dapat kita tafsirkan sebagai daun atau bunga. Mari kita pilih skema kedua di sini karena ini dapat digeneralisasi dengan baik untuk hal-hal lain yang akan kita lihat nanti. Dan inilah jaringan saraf yang melakukan klasifikasi menggunakan skema ini. Mari kita kerjakan:

![Image by author](https://towardsdatascience.com/wp-content/uploads/2024/10/19D2HQj6EBw0NC4c7YU0bWg.png)
*Image by author*

Beberapa jargon:

***Neurons/nodes***: Angka-angka di dalam lingkaran.

***Weights***: Angka-angka berwarna pada garis-garis tersebut.

***Layers***: Kumpulan neuron disebut lapisan. Anda bisa menganggap jaringan ini memiliki 3 lapisan: Lapisan input dengan 4 neuron, Lapisan tengah dengan 3 neuron, dan Lapisan output dengan 2 neuron.

Untuk menghitung prediksi/output dari jaringan ini (disebut "**forward pass**"), Anda mulai dari kiri. Kita memiliki data yang tersedia untuk neuron di lapisan Input. Untuk bergerak "maju" ke lapisan berikutnya, Anda mengalikan angka di dalam lingkaran dengan bobot (weight) untuk pasangan neuron yang sesuai dan Anda menjumlahkan semuanya. Kami mendemonstrasikan matematika lingkaran biru dan oranye di atas. Menjalankan seluruh jaringan, kita melihat bahwa angka pertama di lapisan output keluar lebih tinggi sehingga kita menafsirkannya sebagai "jaringan mengklasifikasikan nilai (RGB, Vol) ini sebagai daun". Jaringan yang terlatih dengan baik dapat mengambil berbagai input untuk (RGB, Vol) dan mengklasifikasikan objek dengan benar.

Model tersebut tidak memiliki gagasan tentang apa itu daun atau bunga, atau apa itu (RGB, Vol). Tugasnya adalah mengambil tepat 4 angka dan mengeluarkan tepat 2 angka. Adalah interpretasi kita bahwa 4 angka input tersebut adalah (RGB, Vol) dan juga keputusan kita untuk melihat angka output dan menyimpulkan bahwa jika angka pertama lebih besar maka itu adalah daun dan seterusnya. Dan terakhir, terserah kita juga untuk memilih bobot yang tepat sehingga model akan mengambil angka input kita dan memberi kita dua angka yang tepat sehingga ketika kita menafsirkannya, kita mendapatkan interpretasi yang kita inginkan.

Efek samping yang menarik dari hal ini adalah Anda dapat mengambil jaringan yang sama dan alih-alih memasukkan RGB, Vol, masukkan 4 angka lain seperti tutupan awan, kelembapan, dll. dan menafsirkan dua angka tersebut sebagai "Cerah dalam satu jam" atau "Hujan dalam satu jam" dan kemudian jika bobot Anda dikalibrasi dengan baik, Anda bisa mendapatkan jaringan yang sama persis untuk melakukan dua hal sekaligus – mengklasifikasikan daun/bunga dan memprediksi hujan dalam satu jam! Jaringan hanya memberi Anda dua angka, apakah Anda menafsirkannya sebagai klasifikasi atau prediksi atau hal lain sepenuhnya terserah Anda.

Hal-hal yang ditinggalkan demi penyederhanaan (silakan diabaikan tanpa mengorbankan pemahaman):

*   **Activation layer**: Hal penting yang hilang dari jaringan ini adalah "lapisan aktivasi". Itu adalah kata mewah untuk mengatakan bahwa kita mengambil angka di setiap lingkaran dan menerapkan fungsi non-linear padanya (**RELU** adalah fungsi umum di mana Anda hanya mengambil angka tersebut dan menyetelnya ke nol jika negatif, dan membiarkannya tidak berubah jika positif). Jadi pada dasarnya dalam kasus kita di atas, kita akan mengambil lapisan tengah dan mengganti kedua angka (-26,6 dan -47,1) dengan nol sebelum kita melanjutkan lebih jauh ke lapisan berikutnya. Tentu saja, kita harus melatih ulang bobot di sini agar jaringan berguna kembali. Tanpa lapisan aktivasi, semua penjumlahan dan perkalian dalam jaringan dapat digabungkan menjadi satu lapisan. Dalam kasus kita, Anda dapat menulis lingkaran hijau sebagai jumlah RGB secara langsung dengan beberapa bobot dan Anda tidak memerlukan lapisan tengah. Ini biasanya tidak mungkin jika kita memiliki non-linearitas di sana. Ini membantu jaringan menangani situasi yang lebih kompleks.
*   **Bias**: Jaringan biasanya juga berisi angka lain yang terkait dengan setiap node, angka ini cukup ditambahkan ke produk untuk menghitung nilai node dan angka ini disebut "**bias**". Jadi jika bias untuk node biru teratas adalah 0,25 maka nilai dalam node tersebut adalah: (32 * 0,10) + (107 * -0,29) + (56 * -0,07) + (11,2 * 0,46) **+ 0,25** = - 26,35. Kata parameter biasanya digunakan untuk merujuk pada semua angka dalam model yang bukan neuron/node.
*   **Softmax**: Kita biasanya tidak menafsirkan lapisan output secara langsung seperti yang ditunjukkan dalam model kita. Kita mengubah angka-angka tersebut menjadi probabilitas (yaitu menjadikannya agar semua angka positif dan berjumlah 1). Jika semua angka di lapisan output sudah positif, salah satu cara Anda dapat mencapainya adalah dengan membagi setiap angka dengan jumlah semua angka di lapisan output. Meskipun fungsi "softmax" biasanya digunakan yang dapat menangani angka positif dan negatif.

## Bagaimana model-model ini dilatih?

Dalam contoh di atas, kita secara ajaib memiliki bobot yang memungkinkan kita memasukkan data ke dalam model dan mendapatkan output yang baik. Namun bagaimana bobot ini ditentukan? Proses pengaturan bobot (atau "parameter") ini disebut "**melatih model**" (training the model), dan kita memerlukan beberapa data pelatihan untuk melatih model tersebut.

Katakanlah kita memiliki beberapa data di mana kita memiliki input dan kita sudah tahu apakah setiap input sesuai dengan daun atau bunga, ini adalah "**data pelatihan**" kita dan karena kita memiliki label daun/bunga untuk setiap set angka (R, G, B, Vol), ini adalah "**data berlabel**".

Begini cara kerjanya:

*   Mulailah dengan angka acak, yaitu atur setiap parameter/bobot ke angka acak.
*   Sekarang, kita tahu bahwa ketika kita memasukkan data yang sesuai dengan daun (R=32, G=107, B=56, Vol=11,2). Misalkan kita menginginkan angka yang lebih besar untuk daun di lapisan output. Katakanlah kita menginginkan angka yang sesuai dengan daun sebagai 0,8 dan angka yang sesuai dengan bunga sebagai 0,2.
*   Kita tahu angka yang kita inginkan di lapisan output, dan angka yang kita dapatkan dari parameter yang dipilih secara acak (yang berbeda dari yang kita inginkan). Jadi untuk semua neuron di lapisan output, mari kita ambil selisih antara angka yang kita inginkan dan angka yang kita miliki. Lalu jumlahkan selisihnya. Misal, jika lapisan output adalah 0,6 dan 0,4 di dua neuron, maka kita dapatkan: (0,8-0,6)=0,2 dan (0,2-0,4)= -0,2 jadi kita mendapatkan total 0,4 (mengabaikan tanda minus sebelum menjumlahkan). Kita bisa menyebut ini sebagai "**loss**" (kerugian) kita. Idealnya kita ingin loss mendekati nol, yaitu kita ingin "**meminimalkan loss**".
*   Setelah kita memiliki loss, kita dapat sedikit mengubah setiap parameter untuk melihat apakah menambah atau menguranginya akan meningkatkan loss. Ini disebut "**gradient**" (gradien) dari parameter tersebut. Kemudian kita dapat menggerakkan setiap parameter dalam jumlah kecil ke arah di mana loss turun (berlawanan dengan arah gradien). Setelah kita memindahkan semua parameter sedikit, loss seharusnya lebih rendah.
*   Terus ulangi proses ini dan Anda akan mengurangi loss, dan akhirnya memiliki serangkaian bobot/parameter yang "**terlatih**". Seluruh proses ini disebut "**gradient descent**".

Beberapa catatan:

*   Anda sering kali memiliki beberapa contoh pelatihan, jadi ketika Anda mengubah bobot sedikit untuk meminimalkan loss pada satu contoh, hal itu mungkin memperburuk loss untuk contoh lainnya. Cara menanganinya adalah dengan mendefinisikan loss sebagai loss rata-rata dari semua contoh dan kemudian mengambil gradien dari loss rata-rata tersebut. Ini mengurangi loss rata-rata pada seluruh set data pelatihan. Setiap siklus tersebut disebut "**epoch**". Kemudian Anda dapat terus mengulangi epoch sehingga menemukan bobot yang mengurangi loss rata-rata.
*   Kita sebenarnya tidak perlu "memindahkan bobot ke sana kemari" untuk menghitung gradien untuk setiap bobot – kita bisa menyimpulkannya dari rumus.

Dalam praktiknya, melatih jaringan yang dalam adalah proses yang sulit dan kompleks karena gradien dapat dengan mudah lepas kendali, menjadi nol atau tak terhingga selama pelatihan (disebut masalah "**vanishing gradient**" dan "**exploding gradient**"). Definisi sederhana tentang loss yang kita bicarakan di sini valid, tetapi jarang digunakan karena ada bentuk fungsional yang lebih baik yang bekerja dengan baik untuk tujuan tertentu. Dengan model modern yang mengandung miliaran parameter, melatih model memerlukan sumber daya komputasi yang sangat besar yang memiliki masalahnya sendiri (keterbatasan memori, paralelisasi, dll.)

## Bagaimana semua ini membantu menghasilkan bahasa?

Ingat, jaringan saraf mengambil beberapa angka, melakukan beberapa matematika berdasarkan parameter yang dilatih, dan mengeluarkan beberapa angka lainnya. Segalanya adalah tentang interpretasi dan melatih parameter. Jika kita dapat menafsirkan dua angka tersebut sebagai "daun/bunga" atau "hujan atau cerah dalam satu jam", kita juga dapat menafsirkan mereka sebagai "karakter berikutnya dalam sebuah kalimat".

Namun ada lebih dari 2 huruf dalam bahasa Inggris, jadi kita harus memperluas jumlah neuron di lapisan output menjadi, katakanlah, 26 huruf dalam bahasa Inggris (mari kita masukkan juga beberapa simbol seperti spasi, titik, dll.). Setiap neuron dapat sesuai dengan satu karakter dan kita melihat (26 atau lebih) neuron di lapisan output dan mengatakan bahwa karakter yang sesuai dengan neuron bernomor tertinggi di lapisan output adalah karakter output. Sekarang kita memiliki jaringan yang dapat mengambil beberapa input dan mengeluarkan karakter.

Bagaimana jika kita mengganti input dalam jaringan kita dengan karakter-karakter ini: "Humpty Dumpt" dan memintanya untuk mengeluarkan karakter dan menafsirkannya sebagai "Saran jaringan untuk karakter berikutnya dalam urutan yang baru saja kita masukkan". Kita mungkin bisa mengatur bobotnya dengan cukup baik agar ia mengeluarkan "y" – sehingga melengkapi "Humpty Dumpty". Kecuali untuk satu masalah, bagaimana cara kita memasukkan daftar karakter ini ke dalam jaringan? Jaringan kita hanya menerima angka!!

Salah satu solusi sederhana adalah menetapkan angka untuk setiap karakter. Katakanlah a=1, b=2 dan seterusnya. Sekarang kita bisa memasukkan "humpty dumpt" dan melatihnya untuk memberi kita "y". Jaringan kita terlihat seperti ini:

![Image by author](https://towardsdatascience.com/wp-content/uploads/2024/10/1XPyJ-V0vbPv6EDwFpk7KYQ.png)
*Image by author*

Oke, jadi sekarang kita bisa memprediksi satu karakter ke depan dengan memberikan daftar karakter ke jaringan. Kita bisa menggunakan fakta ini untuk membangun seluruh kalimat. Misalnya, setelah kita memprediksi "y", kita bisa menambahkan "y" itu ke daftar karakter yang kita miliki dan memasukkannya ke jaringan dan memintanya untuk memprediksi karakter berikutnya. Dan jika dilatih dengan baik, ia seharusnya memberi kita spasi, dan seterusnya dan seterusnya. Pada akhirnya, kita seharusnya bisa menghasilkan "Humpty Dumpty sat on a wall" secara rekursif. Kita memiliki AI Generatif. Terlebih lagi, ***sekarang kita memiliki jaringan yang mampu menghasilkan bahasa!*** Sekarang, tidak ada yang benar-benar memasukkan angka yang ditetapkan secara acak dan kita akan melihat skema yang lebih masuk akal nantinya.

Pembaca yang jeli akan menyadari bahwa kita tidak dapat benar-benar memasukkan "Humpty Dumpty" ke dalam jaringan karena cara diagramnya, ia hanya memiliki 12 neuron di lapisan input untuk setiap karakter dalam "humpty dumpt" (termasuk spasi). Jadi bagaimana kita bisa memasukkan "y" untuk pass berikutnya. Menempatkan neuron ke-13 di sana akan mengharuskan kita untuk memodifikasi seluruh jaringan, itu tidak bisa dilakukan. Solusinya sederhana, mari kita keluarkan "h" dan kirimkan 12 karakter terbaru. Jadi kita akan mengirimkan "umpty dumpty" dan jaringan akan memprediksi spasi. Kemudian kita akan memasukkan "mpty dumpty " dan ia akan menghasilkan 's' dan seterusnya. Terlihat seperti ini:

![Image by author](https://towardsdatascience.com/wp-content/uploads/2024/10/10_tSCfEAL9NIK8U95Q-6Vg.png)
*Image by author*

Kita membuang banyak informasi di baris terakhir dengan memberi model hanya " sat on the wal". Jadi apa yang dilakukan jaringan tercanggih saat ini? Kurang lebih persis seperti itu. Panjang input yang dapat kita masukkan ke dalam jaringan adalah tetap (ditentukan oleh ukuran lapisan input). Ini disebut "**context length**" (panjang konteks) – konteks yang diberikan ke jaringan untuk membuat prediksi di masa depan. Jaringan modern dapat memiliki panjang konteks yang sangat besar (beberapa ribu kata) dan itu membantu.

Hal lain yang akan diperhatikan oleh pembaca yang cermat adalah bahwa kita memiliki interpretasi yang berbeda untuk input dan output untuk huruf yang sama! Misalnya, saat memasukkan "h" kita hanya menunjukkannya dengan angka 8, tetapi pada lapisan output kita tidak meminta model untuk mengeluarkan satu angka (8 untuk "h", 9 for "i" dan seterusnya..) sebaliknya kita meminta model untuk mengeluarkan 26 angka dan kemudian kita melihat mana yang paling tinggi dan kemudian jika angka ke-8 adalah yang tertinggi kita menafsirkan output sebagai "h". Mengapa kita tidak menggunakan interpretasi yang sama dan konsisten di kedua ujungnya? Kita bisa, hanya saja dalam hal bahasa, membebaskan diri Anda untuk memilih di antara interpretasi yang berbeda memberi Anda peluang lebih baik untuk membangun model yang lebih baik. Dan kebetulan interpretasi yang paling efektif saat ini untuk input dan output adalah berbeda. Faktanya, cara kita memasukkan angka dalam model ini bukanlah cara terbaik untuk melakukannya, kita akan melihat cara yang lebih baik untuk melakukannya sebentar lagi.

## Apa yang membuat large language models bekerja begitu baik?

Menghasilkan "Humpty Dumpty sat on a wall" karakter demi karakter sangat jauh dari apa yang bisa dilakukan LLM modern. Ada sejumlah perbedaan dan inovasi yang membawa kita dari AI generatif sederhana yang kita bahas di atas ke bot yang mirip manusia. Mari kita bahas satu per satu:

## Embeddings

Ingat kita katakan bahwa cara kita memasukkan karakter ke dalam model bukanlah cara terbaik untuk melakukannya. Kita hanya memilih angka secara sembarang untuk setiap karakter. Bagaimana jika ada angka yang lebih baik yang bisa kita tetapkan yang akan memungkinkan kita untuk melatih jaringan yang lebih baik? Bagaimana kita menemukan angka-angka yang lebih baik ini? Inilah trik cerdasnya:

Saat kita melatih model di atas, cara kita melakukannya adalah dengan menggerakkan bobot dan melihat mana yang memberi kita loss terkecil pada akhirnya. Dan kemudian secara perlahan dan rekursif mengubah bobot tersebut. Pada setiap gilirannya kita akan:

*   Memasukkan input
*   Menghitung lapisan output
*   Membandingkannya dengan output yang kita inginkan dan menghitung loss rata-rata
*   Menyesuaikan bobot dan mulai lagi

Dalam proses ini, input bersifat tetap. Ini masuk akal ketika input adalah (RGB, Vol). Tetapi angka yang kita masukkan sekarang untuk a, b, c dll.. dipilih secara sembarang oleh kita. Bagaimana jika di setiap iterasi selain menggerakkan bobot sedikit, kita juga menggerakkan input dan melihat apakah kita bisa mendapatkan loss yang lebih rendah dengan menggunakan angka yang berbeda untuk mewakili "a" dan seterusnya? Kita benar-benar mengurangi loss dan membuat model menjadi lebih baik. Pada dasarnya, terapkan gradient descent tidak hanya pada bobot tetapi juga representasi angka untuk input karena itu hanyalah angka yang dipilih secara sembarang. Ini disebut "**embedding**". Ini adalah pemetaan input ke angka, dan seperti yang baru saja Anda lihat, ini perlu dilatih. Proses melatih embedding sangat mirip dengan melatih parameter. Namun, satu keuntungan besar dari hal ini adalah setelah Anda melatih embedding, Anda dapat menggunakannya di model lain jika Anda mau. Ingatlah bahwa Anda akan secara konsisten menggunakan embedding yang sama untuk mewakili satu token/karakter/kata.

Kita berbicara tentang embedding yang hanya berupa satu angka per karakter. Namun, dalam kenyataannya embedding memiliki lebih dari satu angka. Itu karena sulit untuk menangkap kekayaan konsep dengan satu angka. Jika kita melihat contoh daun dan bunga, kita memiliki empat angka untuk setiap objek. Masing-masing dari empat angka ini menyampaikan properti dan model tersebut dapat menggunakan semuanya untuk menebak objek secara efektif. Jika kita hanya memiliki satu angka, katakanlah saluran merah dari warna tersebut, mungkin akan jauh lebih sulit bagi model tersebut. Kita mencoba menangkap bahasa manusia di sini – kita akan membutuhkan lebih dari satu angka.

Jadi, alih-alih mewakili setiap karakter dengan satu angka, mungkin kita bisa mewakilinya dengan beberapa angka untuk menangkap kekayaan tersebut? Mari kita tetapkan sekumpulan angka untuk setiap karakter. Mari kita sebut kumpulan angka yang terurut sebagai "**vector**" (terurut karena setiap angka memiliki posisi, dan jika kita menukar posisi dua angka, itu akan memberi kita vektor yang berbeda). Panjang vektor hanyalah berapa banyak angka yang dikandungnya. Kita akan menetapkan vektor untuk setiap karakter. Dua pertanyaan muncul:

*   Jika kita memiliki vektor yang ditetapkan untuk setiap karakter, bukan angka, bagaimana sekarang kita memasukkan "humpty dumpt" ke jaringan? Jawabannya sederhana. Katakanlah kita menetapkan vektor berisi 10 angka untuk setiap karakter. Maka alih-alih lapisan input memiliki 12 neuron, kita akan menempatkan 120 neuron di sana karena masing-masing dari 12 karakter dalam "humpty dumpt" memiliki 10 angka untuk dimasukkan. Sekarang kita tinggal menempatkan neuron-neuron tersebut berdampingan dan kita siap berangkat.
*   Bagaimana kita menemukan vektor-vektor ini? Untungnya, kita baru saja mempelajari cara melatih angka embedding. Melatih vektor embedding tidak ada bedanya. Anda sekarang memiliki 120 input, bukan 12, tetapi yang Anda lakukan hanyalah menggerakkannya untuk melihat bagaimana Anda dapat meminimalkan loss. Dan kemudian Anda mengambil 10 yang pertama dan itulah vektor yang sesuai dengan "h" dan seterusnya.

Semua vektor embedding tentu saja harus memiliki panjang yang sama, jika tidak, kita tidak akan punya cara untuk memasukkan semua kombinasi karakter ke dalam jaringan. Misalnya "humpty dumpt" dan pada iterasi berikutnya "umpty dumpty" – dalam kedua kasus tersebut kita memasukkan 12 karakter ke dalam jaringan dan jika masing-masing dari 12 karakter tersebut tidak diwakili oleh vektor dengan panjang 10, kita tidak akan dapat memasukkannya secara andal ke dalam lapisan input sepanjang 120. Mari kita visualisasikan vektor embedding ini:

![Image by author](https://towardsdatascience.com/wp-content/uploads/2024/10/1lZOR8fNDEWHxUhLCSB-67A.png)
*Image by author*

Mari kita sebut kumpulan vektor berukuran sama yang terurut sebagai **matrix** (matriks). Matriks di atas disebut **embedding matrix**. Anda memberi tahu nomor kolom yang sesuai dengan huruf Anda dan melihat kolom tersebut dalam matriks akan memberi Anda vektor yang Anda gunakan untuk mewakili huruf tersebut. Ini dapat diterapkan secara lebih umum untuk menyematkan kumpulan hal-hal apa pun – Anda hanya perlu memiliki kolom dalam matriks ini sebanyak hal yang Anda miliki.

## Subword Tokenizers

Sejauh ini, kita telah bekerja dengan karakter sebagai blok bangunan dasar bahasa. Ini memiliki keterbatasan. Bobot jaringan saraf harus melakukan banyak kerja berat di mana mereka harus memahami urutan karakter tertentu (yaitu kata) yang muncul bersebelahan dan kemudian di sebelah kata lain. Bagaimana jika kita langsung menetapkan embedding ke kata-kata dan membuat jaringan memprediksi kata berikutnya. Jaringan tidak memahami apa pun lebih dari angka, jadi kita dapat menetapkan vektor sepanjang 10 ke masing-masing kata "humpty", "dumpty", "sat", "on" dll.. dan kemudian kita tinggal memberinya dua kata dan ia dapat memberi kita kata berikutnya. "**Token**" adalah istilah untuk satu unit yang kita sematkan dan kemudian masukkan ke model. Model kita sejauh ini menggunakan karakter sebagai token, sekarang kita mengusulkan untuk menggunakan seluruh kata sebagai token.

Menggunakan tokenisasi kata memiliki satu efek mendalam pada model kita. Ada lebih dari 180 ribu kata dalam bahasa Inggris. Menggunakan skema interpretasi output kita yang memiliki satu neuron per kemungkinan output, kita membutuhkan ratusan ribu neuron di lapisan output alih-alih 26 atau lebih. Namun, yang perlu diperhatikan adalah karena kita memperlakukan setiap kata secara terpisah, dan kita mulai dengan embedding angka acak untuk masing-masing – kata-kata yang sangat mirip (misalnya "cat" dan "cats") akan dimulai tanpa hubungan. Anda akan mengharapkan bahwa embedding untuk kedua kata tersebut harus berdekatan satu sama lain – yang pasti akan dipelajari oleh model. Tapi, bisakah kita menggunakan kemiripan yang jelas ini untuk mempermudah masalah?

Ya, kita bisa. Skema embedding yang paling umum dalam model bahasa saat ini adalah sesuatu di mana Anda memecah kata-kata menjadi sub-kata (subwords) dan kemudian menyematkannya. Dalam contoh kucing, kita akan memecah *cats* menjadi dua token "cat" dan "s". Sekarang lebih mudah bagi model untuk memahami konsep "s" yang diikuti oleh kata-kata familiar lainnya dan sebagainya. Ini juga mengurangi jumlah token yang kita butuhkan ([sentencepiece](https://github.com/google/sentencepiece) adalah tokenizer umum dengan pilihan ukuran kosakata dalam puluhan ribu vs ratusan ribu kata dalam bahasa Inggris). Tokenizer adalah sesuatu yang mengambil teks input Anda (misalnya "Humpty Dumpt") dan membaginya menjadi token dan memberi Anda angka korespondensi yang Anda butuhkan untuk mencari vektor embedding untuk token tersebut dalam matriks embedding.

Dengan embedding dan tokenisasi sub-kata, sebuah model bisa terlihat seperti ini:

![Image by author](https://towardsdatascience.com/wp-content/uploads/2024/10/1VGNZ1Zighiek1sAMdiZCaw.png)
*Image by author*

Beberapa bagian berikutnya berurusan dengan kemajuan yang lebih baru dalam pemodelan bahasa, dan yang membuat LLM sekuat sekarang. Namun, untuk memahaminya ada beberapa konsep matematika dasar yang perlu Anda ketahui:

*   Matriks dan perkalian matriks
*   Konsep umum fungsi dalam matematika
*   Menghitung pangkat angka (misal a³)
*   Mean sampel, varians, dan standar deviasi

Saya telah menambahkan ringkasan konsep-konsep ini di lampiran.

## Self Attention

Sejauh ini kita hanya melihat satu struktur jaringan saraf sederhana (disebut feedforward network), yang berisi sejumlah lapisan dan setiap lapisan terhubung sepenuhnya ke lapisan berikutnya (artinya, ada garis yang menghubungkan dua neuron di lapisan berturut-turut), dan hanya terhubung ke lapisan berikutnya (misalnya tidak ada garis antara lapisan 1 dan lapisan 3 dll.). Namun, seperti yang dapat Anda bayangkan, tidak ada yang menghentikan kita untuk menghapus atau membuat koneksi lain. Atau bahkan membuat struktur yang lebih kompleks. Mari kita jelajahi struktur yang sangat penting: self-attention.

Jika Anda melihat struktur bahasa manusia, kata berikutnya yang ingin kita prediksi akan bergantung pada semua kata sebelumnya. Namun, mereka mungkin bergantung pada beberapa kata sebelumnya ke tingkat yang lebih besar daripada yang lain. Misalnya, jika kita mencoba memprediksi kata berikutnya dalam "Damian had a secret child, a girl, and he had written in his will that all his belongings, along with the magical orb, will belong to ____". Kata di sini bisa "her" atau "him" dan itu sangat tergantung pada kata yang jauh lebih awal dalam kalimat: *girl/boy*.

Kabar baiknya adalah, model feedforward sederhana kita terhubung ke semua kata dalam konteks, sehingga ia dapat mempelajari bobot yang sesuai untuk kata-kata penting. Tapi inilah masalahnya, bobot yang menghubungkan posisi tertentu dalam model kita melalui lapisan feed forward adalah tetap (untuk setiap posisi). Jika kata penting selalu berada di posisi yang sama, ia akan mempelajari bobot dengan tepat dan kita akan baik-baik saja. Namun, kata yang relevan dengan prediksi berikutnya bisa berada di mana saja dalam sistem. Kita butuh bobot yang bergantung tidak hanya pada posisi tetapi juga pada konten di posisi tersebut. Bagaimana kita mencapai ini?

Self-attention melakukan sesuatu seperti menjumlahkan vektor embedding untuk setiap kata, tetapi alih-alih menjumlahkannya secara langsung, ia menerapkan beberapa bobot pada masing-masing kata. Jadi jika vektor embedding untuk humpty, dumpty, sat masing-masing adalah x1, x2, x3, maka ia akan mengalikan masing-masing dengan bobot (sebuah angka) sebelum menjumlahkannya. Sesuatu seperti output = 0,5 x1 + 0,25 x2 + 0,25 x3. Jika kita menulis bobotnya sebagai u1, u2, u3 sedemikian rupa sehingga output = u1x1+u2x2+u3x3 lalu bagaimana kita menemukan bobot u1, u2, u3 ini?

Idealnya, kita ingin bobot ini bergantung pada vektor yang kita jumlahkan – seperti yang kita lihat beberapa mungkin lebih penting daripada yang lain. Tapi penting bagi siapa? Bagi kata yang akan kita prediksi. Jadi kita juga ingin bobotnya bergantung pada kata yang akan kita prediksi. Nah itu masalahnya, kita tentu saja tidak tahu kata yang akan kita prediksi sebelum kita memprediksinya. Jadi, self-attention menggunakan kata yang tepat mendahului kata yang akan kita prediksi, yaitu kata terakhir dalam kalimat yang tersedia.

Bagus, jadi kita ingin bobot untuk vektor-vektor ini, dan kita ingin setiap bobot bergantung pada kata yang kita agregasikan dan kata yang tepat mendahului kata yang akan kita prediksi. Pada dasarnya, kita ingin fungsi u1 = F(x1, x3) di mana x1 adalah kata yang akan kita beri bobot dan x3 adalah kata terakhir dalam urutan yang kita miliki. Sekarang, cara mudah untuk mencapainya adalah dengan memiliki vektor untuk x1 (sebut saja k1) dan vektor terpisah untuk x3 (sebut saja q3) lalu cukup ambil dot product mereka. Ini akan memberi kita sebuah angka dan itu akan bergantung pada x1 dan x3. Bagaimana kita mendapatkan vektor k1 dan q3 ini? Kita membangun jaringan saraf satu lapisan kecil untuk beralih dari x1 ke k1. Dan kita membangun jaringan lain dari x3 ke q3 dll. Menggunakan notasi matriks kita, kita pada dasarnya menghasilkan matriks bobot Wk dan Wq sehingga k1 = Wkx1 dan q1 = Wqx1 dan seterusnya. Sekarang kita bisa mengambil dot product k1 dan q3 untuk mendapatkan skalar, jadi u1 = F(x1,x3) = Wkx1 **·** Wqx3.

Satu hal tambahan yang terjadi dalam self-attention adalah kita tidak secara langsung mengambil jumlah bobot dari vektor embedding itu sendiri. Sebaliknya, kita mengambil jumlah bobot dari beberapa "value" (nilai) dari vektor embedding tersebut, yang diperoleh dari jaringan satu lapisan kecil lainnya. Apa artinya ini mirip dengan k1 dan q1, kita sekarang juga memiliki v1 untuk kata x1 dan kita mendapatkannya melalui matriks Wv sedemikian rupa sehingga v1=Wvx1. v1 ini kemudian diagregasi. Jadi semuanya terlihat seperti ini jika kita hanya memiliki 3 kata dan kita mencoba memprediksi kata keempat:

![Self attention. Image by author](https://towardsdatascience.com/wp-content/uploads/2024/10/17AJIjt5fiRhpc0h1qJKt3g.png)
*Self attention. Image by author*

Tanda tambah mewakili penambahan sederhana dari vektor, yang menyiratkan bahwa mereka harus memiliki panjang yang sama. Satu modifikasi terakhir yang tidak ditunjukkan di sini adalah skalar u1, u2, u3 dll tidak selalu berjumlah 1. Jika kita membutuhkannya sebagai bobot, kita harus membuatnya berjumlah 1. Jadi kita akan menerapkan trik yang sudah dikenal dan menggunakan fungsi softmax.

Ini adalah self-attention. Ada juga cross-attention di mana Anda dapat memiliki q3 yang berasal dari kata terakhir, tetapi k dan v dapat berasal dari kalimat lain sama sekali. Ini misalnya berharga dalam tugas-tugas penerjemahan. Sekarang kita tahu apa itu attention.

Seluruh hal ini sekarang dapat dimasukkan ke dalam kotak dan disebut "**self attention block**". Pada dasarnya, blok self attention ini mengambil vektor embedding dan mengeluarkan satu vektor output dengan panjang berapa pun yang dipilih pengguna. Blok ini memiliki tiga parameter, Wk, Wq, Wv. Ada banyak blok seperti itu dalam literatur machine learning, dan mereka biasanya diwakili oleh kotak-kotak dalam diagram dengan namanya di atasnya. Seperti ini:

![Image by author](https://towardsdatascience.com/wp-content/uploads/2024/10/1OgcRxWyftIXN8WbQw_-QKg.png)
*Image by author*

Salah satu hal yang akan Anda perhatikan dengan self-attention adalah bahwa posisi benda-benda sejauh ini tampaknya tidak relevan. Kita menggunakan W yang sama di seluruh papan sehingga menukar Humpty dan Dumpty tidak akan benar-benar membuat perbedaan di sini. Ini berarti bahwa meskipun attention dapat mengetahui apa yang harus diperhatikan, ini tidak akan bergantung pada posisi kata. Namun, kita tahu bahwa posisi kata penting dalam bahasa Inggris dan kita mungkin dapat meningkatkan kinerja dengan memberi model gambaran tentang posisi sebuah kata.

Oleh karena itu, ketika attention digunakan, kita tidak sering memasukkan vektor embedding secara langsung ke blok self attention. Kita nanti akan melihat bagaimana "positional encoding" ditambahkan ke vektor embedding sebelum dimasukkan ke blok attention.

*Catatan untuk yang sudah berpengalaman*: Bagi Anda yang baru pertama kali membaca tentang self-attention akan menyadari bahwa kami tidak merujuk pada matriks K dan Q, atau menerapkan mask, dll. Itu karena hal-hal tersebut adalah detail implementasi yang timbul dari bagaimana model-model ini biasanya dilatih.

## Softmax

Kita berbicara singkat tentang softmax di catatan pertama. Inilah masalah yang coba dipecahkan oleh softmax: Dalam interpretasi output kita, kita memiliki neuron sebanyak opsi dari mana kita ingin jaringan memilih satu. Dan kita katakan bahwa kita akan menafsirkan pilihan jaringan sebagai neuron dengan nilai tertinggi. Kemudian kita katakan kita akan menghitung loss sebagai selisih antara nilai yang diberikan jaringan, dan nilai ideal yang kita inginkan. Tapi apa nilai ideal yang kita inginkan? Kita mengaturnya ke 0,8 dalam contoh daun/bunga. Tapi kenapa 0,8? Kenapa bukan 5, atau 10, atau 10 juta? Semakin tinggi semakin baik untuk contoh pelatihan itu. Idealnya kita ingin tak terhingga di sana! Sekarang itu akan membuat masalah tidak dapat dipecahkan – semua loss akan menjadi tak terhingga dan rencana kita untuk meminimalkan loss dengan menggerakkan parameter (ingat "gradient descent") gagal. Bagaimana kita menangani ini?

Satu hal sederhana yang bisa kita lakukan adalah membatasi nilai yang kita inginkan. Katakanlah antara 0 dan 1? Ini akan membuat semua loss menjadi terbatas, tetapi sekarang kita punya masalah tentang apa yang terjadi ketika jaringan melampaui batas (overshoot). Katakanlah ia mengeluarkan (5,1) untuk (daun,bunga) dalam satu kasus, dan (0,1) di kasus lain. Kasus pertama membuat pilihan yang tepat tetapi loss-nya lebih buruk! Oke, jadi sekarang kita butuh cara untuk juga mengubah output dari lapisan terakhir dalam rentang (0,1) sehingga mempertahankan urutannya. Kita bisa menggunakan fungsi apa pun ("**function**" dalam matematika hanyalah pemetaan satu angka ke angka lainnya) di sini untuk menyelesaikan pekerjaan tersebut. Salah satu opsi yang mungkin adalah fungsi logistik (lihat grafik di bawah) yang memetakan semua angka ke angka antara (0,1) dan mempertahankan urutannya:

![Image by author](https://towardsdatascience.com/wp-content/uploads/2024/10/1jSyo_owKB-tfTrPXV2giKg.png)
*Image by author*

Sekarang, kita memiliki angka antara 0 dan 1 untuk setiap neuron di lapisan terakhir dan kita dapat menghitung loss dengan menyetel neuron yang benar ke 1, yang lain ke 0 dan mengambil selisihnya dari apa yang diberikan jaringan kepada kita. Ini akan berhasil, tapi bisakah kita melakukannya lebih baik?

Kembali ke contoh "Humpty dumpty" kita, katakanlah kita mencoba menghasilkan dumpty karakter demi karakter dan model kita membuat kesalahan saat memprediksi "m" dalam dumpty. Alih-alih memberi kita lapisan terakhir dengan "m" sebagai nilai tertinggi, ia memberi kita "u" sebagai nilai tertinggi tetapi "m" adalah yang kedua terdekat.

Sekarang kita dapat melanjutkan dengan "duu" dan mencoba memprediksi karakter berikutnya dan seterusnya, tetapi keyakinan model akan rendah karena tidak banyak kelanjutan yang baik dari "humpty duu..". Di sisi lain, "m" adalah yang kedua terdekat, jadi kita juga bisa mencoba "m", memprediksi beberapa karakter berikutnya, dan melihat apa yang terjadi? Mungkin itu memberi kita kata yang lebih baik secara keseluruhan?

Jadi apa yang kita bicarakan di sini bukan hanya secara buta memilih nilai maksimum, tetapi mencoba beberapa. Apa cara yang baik untuk melakukannya? Ya kita harus menetapkan peluang untuk masing-masing – katakanlah kita akan memilih yang teratas dengan 50%, yang kedua dengan 25% dan seterusnya. Itu cara yang bagus. Tapi mungkin kita ingin peluangnya bergantung pada prediksi model yang mendasarinya. Jika model memprediksi nilai untuk m dan u sangat dekat satu sama lain di sini – maka mungkin peluang 50-50 untuk menjelajahi keduanya adalah ide yang bagus?

Jadi kita butuh aturan yang bagus yang mengambil semua angka ini dan mengubahnya menjadi peluang. Itulah yang dilakukan softmax. Ini adalah generalisasi dari fungsi logistik di atas tetapi dengan fitur tambahan. Jika Anda memberinya 10 angka sembarang – ia akan memberi Anda 10 output, masing-masing antara 0 dan 1 dan yang terpenting, ke-10 output tersebut berjumlah 1 sehingga kita dapat menafsirkannya sebagai peluang. Anda akan menemukan softmax sebagai lapisan terakhir di hampir setiap model bahasa.

## Residual Connections

Kita perlahan-lahan mengubah visualisasi jaringan kita seiring berjalannya bagian-bagian tersebut. Kita sekarang menggunakan kotak/blok untuk menunjukkan konsep-konsep tertentu. Notasi ini berguna dalam menunjukkan konsep yang sangat berguna dari koneksi residual (residual connections). Mari kita lihat koneksi residual yang dikombinasikan dengan blok self-attention:

![A residual connection. Image by author](https://towardsdatascience.com/wp-content/uploads/2024/10/1270MXDfslVtvmBjShHL2hQ.png)
*A residual connection. Image by author*

Perhatikan bahwa kita menempatkan "Input" dan "Output" sebagai kotak untuk membuat segalanya lebih sederhana, tetapi ini pada dasarnya masih merupakan kumpulan neuron/angka yang sama seperti yang ditunjukkan di atas.

Jadi apa yang terjadi di sini? Kita pada dasarnya mengambil output dari blok self-attention dan sebelum meneruskannya ke blok berikutnya, kita menambahkannya dengan Input aslinya. Hal pertama yang perlu diperhatikan adalah ini mengharuskan dimensi output blok self-attention sekarang harus sama dengan inputnya. Ini bukan masalah karena seperti yang kita catat, output self-attention ditentukan oleh pengguna. Tapi mengapa melakukan ini? Kita tidak akan membahas semua detailnya di sini, tetapi hal utamanya adalah ketika jaringan semakin dalam (lebih banyak lapisan antara input dan output), semakin sulit untuk melatihnya. Koneksi residual telah terbukti membantu mengatasi tantangan pelatihan ini.

## Layer Normalization

Layer normalization adalah lapisan yang cukup sederhana yang mengambil data yang masuk ke lapisan tersebut dan menormalkannya dengan mengurangkan rata-rata (mean) dan membaginya dengan standar deviasi (mungkin sedikit lebih banyak, seperti yang kita lihat di bawah). Misalnya, jika kita menerapkan layer normalization segera setelah input, ia akan mengambil semua neuron di lapisan input dan kemudian menghitung dua statistik: rata-rata dan standar deviasinya. Katakanlah rata-ratanya adalah M dan standar deviasinya adalah D maka apa yang dilakukan layer norm adalah mengambil masing-masing neuron ini dan menggantinya dengan (x-M)/D di mana x menunjukkan nilai asli neuron yang diberikan.

Sekarang bagaimana ini membantu? Ini pada dasarnya menstabilkan vektor input dan membantu pelatihan jaringan yang dalam. Salah satu kekhawatirannya adalah dengan menormalkan input, apakah kita menghapus beberapa informasi berguna dari mereka yang mungkin membantu dalam mempelajari sesuatu yang berharga tentang tujuan kita? Untuk mengatasi hal ini, lapisan layer norm memiliki parameter **scale** (skala) dan **bias**. Pada dasarnya, untuk setiap neuron Anda cukup mengalikannya dengan skalar dan kemudian menambahkan bias ke dalamnya. Nilai skalar dan bias ini adalah parameter yang dapat dilatih. Seluruhnya terlihat seperti ini:

![Layer Normalization. Image by author](https://towardsdatascience.com/wp-content/uploads/2024/10/1145xBeyaNCU3mblFQyi2_A.png)
*Layer Normalization. Image by author*

Standar deviasi adalah ukuran statistik tentang seberapa tersebar nilai-nilainya, misal, jika nilainya semuanya sama, Anda akan mengatakan standar deviasinya adalah nol. Jika, secara umum, setiap nilai sangat jauh dari rata-rata nilai yang sama ini, maka Anda akan memiliki standar deviasi yang tinggi.

## Dropout

Dropout adalah metode yang sederhana namun efektif untuk menghindari model *overfitting*. *Overfitting* adalah istilah ketika Anda melatih model pada data pelatihan Anda, dan itu berfungsi dengan baik pada dataset tersebut tetapi tidak digeneralisasi dengan baik ke contoh yang belum pernah dilihat model. Teknik yang membantu kita menghindari overfitting disebut "**regularization techniques**", dan dropout adalah salah satunya.

Konsepnya sederhana, dengan memasukkan lapisan dropout selama pelatihan, yang Anda lakukan adalah menghapus secara acak persentase tertentu dari koneksi neuron langsung di antara lapisan-lapisan di mana dropout dimasukkan. Mempertimbangkan jaringan awal kita dan memasukkan lapisan Dropout antara input dan lapisan tengah dengan tingkat dropout 50% dapat terlihat seperti ini:

![Dropout step 1](https://towardsdatascience.com/wp-content/uploads/2024/10/1j0oKuXvH7kfrIpIfXE03VA.png)

![Dropout step 2](https://towardsdatascience.com/wp-content/uploads/2024/10/1UodvtsDn5z73Cp578XOoVw.png)

![Image by author](https://towardsdatascience.com/wp-content/uploads/2024/10/1qHPSwQmV3sfvKT5pG4TEJw.png)
*Image by author*

Sekarang, ini memaksa jaringan untuk melatih dengan banyak redundansi. Intinya, Anda melatih sejumlah model yang berbeda pada saat yang sama – tetapi mereka berbagi bobot.

Sekarang untuk membuat inferensi, kita bisa mengikuti pendekatan yang sama dengan model ensemble. Kita bisa membuat beberapa prediksi menggunakan dropout dan kemudian menggabungkannya. Namun, karena itu secara komputasi intensif – dan karena model kita berbagi bobot umum – mengapa kita tidak melakukan prediksi menggunakan semua bobot saja. Ini harus memberi kita beberapa perkiraan tentang apa yang akan diberikan oleh ensemble. Caranya adalah dengan mengambil semua bobot dan mengalikannya dengan (1-p) di mana p adalah probabilitas penghapusan.

## Multi-head Attention

Ini adalah blok kunci dalam arsitektur transformer. Kita sudah melihat apa itu blok attention. Ingat bahwa output dari blok attention ditentukan oleh pengguna dan itu adalah panjang dari v. Apa itu multi-attention head pada dasarnya adalah Anda menjalankan beberapa attention head secara paralel (mereka semua mengambil input yang sama). Kemudian kita mengambil semua output mereka dan cukup menggabungkannya. Terlihat seperti ini:

![Multi-head attention. Image by author](https://towardsdatascience.com/wp-content/uploads/2024/10/1BmZd8SIDEQu7r5_R7y54kQ.png)
*Multi-head attention. Image by author*

Perhatikan bahwa panah yang mengarah dari v1 -> v1h1 adalah lapisan linear – ada matriks pada setiap panah yang bertransformasi.

Apa yang terjadi di sini adalah kita menghasilkan key, query, dan value yang sama untuk setiap head. Tetapi kemudian kita pada dasarnya menerapkan transformasi linear di atasnya (secara terpisah untuk setiap k, q, v dan secara terpisah untuk setiap head) sebelum kita menggunakan nilai k, q, v tersebut.

## Positional encoding and embedding

Kita berbicara singkat tentang motivasi untuk menggunakan positional encoding di bagian self-attention. Apa itu? Positional embedding tidak berbeda dengan embedding lainnya kecuali alih-alih menyematkan kosakata kata, kita akan menyematkan angka 1, 2, 3 dll. Jadi embedding ini adalah matriks dengan panjang yang sama dengan embedding kata, dan setiap kolom sesuai dengan sebuah angka. Itu saja yang ada di dalamnya.

## Arsitektur GPT

Mari kita bicara tentang arsitektur GPT. Inilah yang digunakan di sebagian besar model GPT (dengan variasi di antaranya). Menggunakan notasi kotak, inilah arsitektur yang terlihat di tingkat tinggi:

![The GPT Architecture. Image by author](https://towardsdatascience.com/wp-content/uploads/2024/10/0U9mQKCWiyakNVwxU.png)
*The GPT Architecture. Image by author*

Tanda + di sini berarti kedua vektor ditambahkan bersama-sama (yang berarti kedua embedding harus berukuran sama). Mari kita lihat GPT Transformer Block ini:

![GPT Block Diagram](https://towardsdatascience.com/wp-content/uploads/2024/10/1Mq4hBZcKPL9GALPSSQG9Xg.png)

Dan itu dia. Disebut "transformer" di sini karena berasal dari dan merupakan jenis transformer – yang merupakan arsitektur yang akan kita lihat di bagian selanjutnya. Mari kita rekap semua yang telah kita bahas sejauh ini yang membangun arsitektur GPT ini:

*   Kita melihat bagaimana jaringan saraf mengambil angka dan mengeluarkan angka lain dan memiliki bobot sebagai parameter yang dapat dilatih.
*   Kita dapat melampirkan interpretasi pada angka input/output ini dan memberikan makna dunia nyata ke jaringan saraf.
*   Kita dapat merantai jaringan saraf untuk membuat yang lebih besar, dan kita dapat menyebut masing-masing sebagai "blok" dan menunjukkannya dengan kotak untuk membuat diagram lebih mudah.
*   Kita mempelajari banyak jenis blok yang berbeda yang melayani tujuan yang berbeda.
*   GPT hanyalah pengaturan khusus dari blok-blok ini yang ditunjukkan di atas dengan interpretasi yang kita bahas di Bagian 1.

Sekarang, transformer GPT ini sebenarnya adalah apa yang disebut "decoder" dalam makalah transformer asli yang memperkenalkan arsitektur transformer. Mari kita lihat itu.

## Arsitektur Transformer

Ini adalah salah satu inovasi utama yang mendorong percepatan pesat dalam kemampuan model bahasa baru-baru ini. Transformer tidak hanya meningkatkan akurasi prediksi, mereka juga lebih mudah/lebih efisien daripada model sebelumnya (untuk dilatih), memungkinkan ukuran model yang lebih besar. Inilah dasar arsitektur GPT di atas.

Jika Anda melihat arsitektur GPT, Anda dapat melihat bahwa itu bagus untuk menghasilkan kata berikutnya dalam urutan. Namun, bagaimana jika Anda ingin melakukan penerjemahan. Bagaimana jika Anda memiliki kalimat dalam bahasa Jerman (misalnya "Wo wohnst du?" = "Where do you live?") dan Anda ingin menerjemahkannya ke bahasa Inggris. Bagaimana kita akan melatih model untuk melakukan ini?

Salah satu caranya adalah dengan menggabungkan kalimat bahasa Jerman di awal bahasa Inggris yang dihasilkan sejauh ini dan memasukkannya ke dalam konteks. Agar lebih mudah bagi model, kita dapat menambahkan pemisah (separator). Ini akan terlihat seperti ini di setiap langkah:

![Image by author](https://towardsdatascience.com/wp-content/uploads/2024/10/1psIz3-v2dMfI3SMsFQjRPQ.png)
*Image by author*

Ini akan berhasil, tetapi ada ruang untuk perbaikan. Transformer awalnya dibuat untuk tugas ini dan terdiri dari "encoder" dan "decoder" – yang pada dasarnya adalah dua blok terpisah. Satu blok cukup mengambil kalimat bahasa Jerman dan memberikan representasi perantara (sekali lagi, kumpulan angka) – ini disebut encoder.

Blok kedua menghasilkan kata-kata. Satu-satunya perbedaan adalah selain memberinya kata-kata yang dihasilkan sejauh ini, kita juga memberinya kalimat bahasa Jerman yang dikodekan (dari blok encoder). Jadi saat menghasilkan bahasa, konteksnya pada dasarnya adalah semua kata yang dihasilkan sejauh ini, ditambah bahasa Jermannya. Blok ini disebut decoder.

Mari kita lihat ilustrasi transformer dari makalah "Attention is all you need" dan mencoba memahaminya:

![Image from Vaswani et al. (2017)](https://towardsdatascience.com/wp-content/uploads/2024/10/1uPwJ24na4D_ECHBv1Lg5Fg.png)
*Image from Vaswani et al. (2017)*

Kumpulan blok vertikal di sebelah kiri disebut "encoder" dan yang di sebelah kanan disebut "decoder". Mari kita telusuri:

**Feed forward**: Jaringan feedforward adalah jaringan yang tidak mengandung siklus. Blok ini menggunakan struktur yang sangat mirip dengan contoh kita di bagian 1. Ia berisi dua lapisan linear, masing-masing diikuti oleh RELU dan lapisan dropout. Perlu diingat bahwa jaringan feedforward ini berlaku untuk setiap posisi secara independen.

***Cross-attention:*** Anda akan menyadari bahwa decoder memiliki multi-head attention dengan panah yang berasal dari encoder. Apa yang terjadi di sini? Ingat value, key, query dalam self-attention? Semuanya berasal dari urutan yang sama. Nah, dalam cross-attention, value dan key berasal dari output encoder, sementara query berasal dari decoder. Tidak ada yang berubah secara matematis kecuali dari mana input untuk key dan value berasal sekarang.

***Nx***: Nx di sini hanya mewakili bahwa blok ini diulang secara berantai sebanyak N kali. Jadi pada dasarnya Anda menumpuk blok tersebut secara berturut-turut. Ini adalah cara untuk membuat jaringan saraf lebih dalam.

***Add & Norm block***: Ini pada dasarnya sama seperti di bawah:

![Image by author](https://towardsdatascience.com/wp-content/uploads/2024/10/0sXvYIassJutgqw5W.png)
*Image by author*

Sekarang Anda memiliki penjelasan lengkap tentang arsitektur transformer yang dibangun dari operasi penjumlahan dan perkalian sederhana dan sepenuhnya mandiri! Anda tahu apa arti setiap baris, setiap jumlah, setiap kotak dan kata dalam hal bagaimana membangunnya dari nol. Secara teoritis, catatan ini berisi apa yang Anda butuhkan untuk mengkodekan transformer dari nol.

## Appendix (Lampiran)

### Perkalian Matriks (Matrix Multiplication)

Kita memperkenalkan vektor dan matriks di atas dalam konteks embedding. Matriks memiliki dua dimensi (jumlah baris dan kolom). Produk dari dua matriks didefinisikan sebagai:

![Image by author](https://towardsdatascience.com/wp-content/uploads/2024/10/0woel5Da5Z22EmiGx.png)
*Image by author*

Titik-titik mewakili perkalian. Sekarang mari kita lihat perhitungan neuron biru dan oranye di gambar pertama. Jika kita menulis bobot sebagai matriks dan input sebagai vektor, kita dapat menulis seluruh operasi dengan cara berikut:

![Image by author](https://towardsdatascience.com/wp-content/uploads/2024/10/0yn1TPuxw_QqnD93k.png)
*Image by author*

Jika matriks bobot disebut "W" dan input disebut "x" maka Wx adalah hasilnya.

### Standar Deviasi (Standard Deviation)

Standar deviasi adalah ukuran statistik seberapa tersebar nilai-nilainya. Rumus untuk menghitung standar deviasi untuk sekumpulan angka, a1, a2, a3…. (katakanlah N angka): kurangi rata-rata dari setiap angka, lalu kuadratkan jawabannya untuk masing-masing dari N angka. Jumlahkan semua angka ini lalu bagi dengan N. Sekarang ambil akar kuadrat dari jawabannya.

### Positional Encoding

Positional encoding hanyalah sebuah vektor dengan panjang yang sama dengan vektor embedding kata, kecuali bahwa itu bukan embedding dalam arti tidak dilatih. Kita cukup menetapkan vektor unik untuk setiap posisi, misalnya vektor yang berbeda untuk posisi 1, posisi 2, dan seterusnya. Pendekatan yang digunakan dalam makalah "Attention is all you need" menggunakan fungsi sinus dan kosinus untuk menghasilkan nilai-nilai unik yang tidak meledak angkanya.

---
*Ditulis Oleh Rohit Patel*