---
slug: ch-11-23
title: "mHC"
layout: chapter
---


## mHC: Manifold-Constrained Hyper-Connections
*(Koneksi-Hiper dengan Kendala Manifold)*

Baru-baru ini, berbagai studi yang dicontohkan oleh *Hyper-Connections* (HC) telah memperluas paradigma koneksi residual (*residual connection*) yang ada di mana-mana—yang telah mapan selama dekade terakhir—dengan memperluas lebar aliran residual (*residual stream width*) dan mendiversifikasi pola konektivitas. Meskipun pendekatan ini menghasilkan perolehan performa yang substansial, diversifikasi ini secara fundamental mengkompromikan **sifat pemetaan identitas (*identity mapping property*)** yang intrinsik pada koneksi residual. Kompromi ini menyebabkan ketidakstabilan pelatihan yang parah (*severe training instability*) dan skalabilitas yang terbatas, serta secara tambahan menimbulkan *overhead* akses memori yang mencolok.

Untuk mengatasi tantangan-tantangan ini, kami mengusulkan **Manifold-Constrained Hyper-Connections (mHC)**, sebuah kerangka kerja umum yang memproyeksikan ruang koneksi residual dari HC ke sebuah *manifold* spesifik guna **memulihkan sifat pemetaan identitas**, sembari menggabungkan optimisasi infrastruktur yang ketat untuk memastikan efisiensi komputasional. Eksperimen-eksperimen empiris mendemonstrasikan bahwa *mHC* efektif untuk pelatihan pada skala besar (*training at scale*), menawarkan peningkatan performa yang nyata dan skalabilitas yang superior. Kami mengantisipasi bahwa *mHC*, sebagai ekstensi dari HC yang fleksibel dan praktis, akan berkontribusi pada pemahaman yang lebih dalam mengenai desain arsitektur topologis dan menyarankan arah-arah yang menjanjikan bagi evolusi model-model fondasional (*foundational models*).

---

### Analisis Gambar 1: Ilustrasi Paradigma Koneksi Residual

Sebelum kita melangkah lebih jauh ke dalam perumusan matematis, mari kita telaah **Gambar 1** secara mendetail, karena gambar ini menyajikan perbandingan konseptual yang sangat krusial antara arsitektur klasik dan inovasi yang ditawarkan dalam makalah ini.

**Deskripsi Visual:**
Gambar ini terdiri dari tiga panel yang membandingkan desain struktural dari aliran data (*data flow*) antar lapisan (*layer*) dalam arsitektur jaringan saraf.

*   **(a) Residual Connection (Koneksi Residual Standar):**
    Ini adalah bentuk klasik (seperti pada ResNet). Panah abu-abu lebar dari bawah ($x_l$) mengalir lurus ke atas. Sebagian data "menyimpang" masuk ke kotak biru berlabel "Layer $\mathcal{F}$" (bisa berupa atensi atau MLP), lalu hasil dari layer $\mathcal{F}$ tersebut *dijumlahkan* kembali ($\oplus$) ke aliran utama untuk membentuk $x_{l+1}$. Sederhana, elegan, dan menstabilkan gradien.

*   **(b) Hyper-Connections (HC) (Koneksi-Hiper yang Tidak Terkendali):**
    Panel tengah menunjukkan arsitektur yang diusulkan sebelumnya (HC). Lebar aliran utama (panah abu-abu di kiri) dipecah menjadi beberapa aliran paralel. Terdapat blok-blok tambahan berwarna oranye:
    *   **Pre Mapping ($\mathcal{H}_l^{\text{pre}}$):** Memetakan informasi dari aliran multi-jalur masuk ke dalam "Layer $\mathcal{F}$" (membentuk $h_l^{\text{in}}$).
    *   **Post Mapping ($\mathcal{H}_l^{\text{post}}$):** Memetakan informasi keluaran dari "Layer $\mathcal{F}$" ($h_l^{\text{out}}$) kembali ke aliran utama.
    *   **Res Mapping ($\mathcal{H}_l^{\text{res}}$):** Mengalihkan dan mencampuradukkan informasi antar aliran paralel secara dinamis.
    **Catatan Kritis:** Karena blok-blok ini "bebas belajar" (*learnable*) tanpa batasan, total sinyal yang mengalir ke atas dapat dengan mudah membengkak (meledak) atau menyusut (menghilang).

*   **(c) Manifold-Constrained HC (mHC) (Inovasi Makalah Ini):**
    Panel kanan sangat mirip dengan panel (b), NAMUN blok-blok pemetaan sekarang berwarna **Hijau**. Blok hijau ini melambangkan penambahan fungsi kendali *Manifold* ($\mathcal{M}$).
    *   **Pre Mapping ($P_{\mathcal{M}^{\text{pre}}}(\mathcal{H}_l^{\text{pre}})$):** Matriks aslinya *diproyeksikan* (dibatasi) terlebih dahulu.
    *   **Post Mapping ($P_{\mathcal{M}^{\text{post}}}(\mathcal{H}_l^{\text{post}})$):** Dibatasi.
    *   **Res Mapping ($P_{\mathcal{M}^{\text{res}}}(\mathcal{H}_l^{\text{res}})$):** Dibatasi secara sangat spesifik menjadi matriks stokastik ganda (*doubly stochastic*).
    **Kesimpulan Akademis:** Dengan memaksakan batasan matematis pada kotak-kotak hijau ini, *mHC* menolak membiarkan sinyal membesar tanpa kontrol, sehingga memulihkan sifat konservasi energi (pemetaan identitas) dari panel (a), sembari tetap menikmati kekayaan interaksi informasi dari panel (b).

---

## 1. Pendahuluan

Arsitektur jaringan saraf dalam (*deep neural network architectures*) telah mengalami evolusi yang pesat sejak diperkenalkannya ResNets (He et al., 2016a). Sebagaimana diilustrasikan pada Gambar 1(a), struktur lapisan tunggal (*single-layer*) dari ResNet dapat dirumuskan sebagai berikut:

$$ \mathbf{x}_{l+1} = \mathbf{x}_l + \mathcal{F}(\mathbf{x}_l, \mathcal{W}_l), \tag{1} $$

**Penjelasan Rumus (1):**
*   $\mathbf{x}_l$: Merepresentasikan vektor *input* (masukan) berdimensi $C$ untuk lapisan ke-$l$.
*   $\mathbf{x}_{l+1}$: Merepresentasikan vektor *output* (keluaran) dari lapisan ke-$l$, yang sekaligus menjadi input untuk lapisan selanjutnya.
*   $\mathcal{F}$: Merepresentasikan fungsi residual (*residual function*), yang berisi operasi kompleks seperti atensi (*attention mechanism*), konvolusi, atau jaringan maju-umpan (*feed-forward networks*).
*   $\mathcal{W}_l$: Matriks bobot (*weights*) yang dapat dipelajari di dalam fungsi $\mathcal{F}$.
*   **Logika di balik rumus:** Daripada memaksa lapisan $\mathcal{F}$ untuk mempelajari pemetaan transformasi secara utuh, kita menyuruh $\mathcal{F}$ untuk hanya mempelajari "sisa" atau *residual* perubahannya saja. Input asli $\mathbf{x}_l$ "diloloskan" langsung ke output (ini yang disebut *skip connection*).

Meskipun fungsi residual $\mathcal{F}$ telah berevolusi selama dekade terakhir untuk mencakup berbagai operasi, paradigma koneksi residual telah mempertahankan bentuk aslinya. Seiring dengan kemajuan arsitektur Transformer (Vaswani et al., 2017), paradigma ini saat ini telah mengukuhkan dirinya sebagai elemen desain fundamental dalam Model Bahasa Besar (LLMs) (Brown et al., 2020; Liu et al., 2024b; Touvron et al., 2023).

Kesuksesan ini utamanya dikaitkan dengan bentuk ringkas (*concise form*) dari koneksi residual. Lebih penting lagi, riset awal (He et al., 2016b) mengungkap bahwa **sifat pemetaan identitas (*identity mapping property*)** dari koneksi residual adalah faktor utama yang mempertahankan stabilitas dan efisiensi selama pelatihan berskala besar. Dengan memperluas koneksi residual secara rekursif melintasi beberapa lapisan, Persamaan (1) menghasilkan:

$$ \mathbf{x}_L = \mathbf{x}_l + \sum_{i=l}^{L-1} \mathcal{F}(\mathbf{x}_i, \mathcal{W}_i), \tag{2} $$

**Penjelasan Rumus (2):**
*   $L$ dan $l$: Merepresentasikan lapisan yang lebih dalam ($L$) dan lapisan yang lebih dangkal ($l$).
*   **Logika di balik rumus:** Persamaan ini menunjukkan bahwa fitur pada lapisan dalam ($L$) adalah sekadar fitur dari lapisan dangkal ($l$) *ditambah* akumulasi dari semua perhitungan residual di antaranya. Sinyal asli $\mathbf{x}_l$ tidak terdistorsi (tidak dikalikan secara eksponensial), melainkan mengalir secara mulus. Istilah **pemetaan identitas (*identity mapping*)** merujuk pada komponen $\mathbf{x}_l$ itu sendiri, yang menekankan sifat bahwa sinyal dari lapisan yang lebih dangkal memetakan dirinya secara *langsung* ke lapisan yang lebih dalam tanpa modifikasi apa pun.

Baru-baru ini, studi-studi yang dicontohkan oleh *Hyper-Connections* (HC) (Zhu et al., 2024) telah memperkenalkan dimensi baru pada koneksi residual dan secara empiris mendemonstrasikan potensi performanya. Arsitektur lapisan tunggal dari HC diilustrasikan pada Gambar 1(b). Dengan memperluas lebar dari aliran residual (*residual stream*) dan meningkatkan kompleksitas koneksi, HC secara signifikan meningkatkan kompleksitas topologis tanpa mengubah *overhead* komputasi (FLOPs) dari unit-unit individunya. Secara formal, propagasi lapisan tunggal dalam HC didefinisikan sebagai:

$$ \mathbf{x}_{l+1} = \mathcal{H}_l^{\text{res}} \mathbf{x}_l + \mathcal{H}_l^{\text{post} \top} \mathcal{F}(\mathcal{H}_l^{\text{pre}} \mathbf{x}_l, \mathcal{W}_l), \tag{3} $$

**Penjelasan Rumus (3):**
*   $\mathbf{x}_l$ dan $\mathbf{x}_{l+1}$: Input dan output dari lapisan ke-$l$.
*   **Berbeda dengan Persamaan (1)**, dimensi fitur dari $\mathbf{x}_l$ dan $\mathbf{x}_{l+1}$ diperluas dari $C$ menjadi $n \times C$, di mana $n$ adalah laju ekspansi (*expansion rate*). Aliran residual kini memiliki $n$ "jalur pipa" paralel.
*   $\mathcal{H}_l^{\text{res}} \in \mathbb{R}^{n \times n}$: Matriks *learnable* (dapat dipelajari) yang memetakan dan mencampur fitur *di dalam* aliran residual itu sendiri.
*   $\mathcal{H}_l^{\text{pre}} \in \mathbb{R}^{1 \times n}$: Matriks pemetaan *learnable* yang merangkum (menggabungkan) fitur dari aliran berdimensi $nC$ ke dalam input *layer* tunggal berdimensi $C$ (untuk masuk ke fungsi $\mathcal{F}$).
*   $\mathcal{H}_l^{\text{post}} \in \mathbb{R}^{1 \times n}$: Sebaliknya, matriks ini memetakan (menyebar) output dari *layer* $\mathcal{F}$ kembali ke dalam aliran berdimensi $nC$.

Namun, seiring dengan peningkatan skala pelatihan, HC memicu risiko potensial berupa **ketidakstabilan**. Kekhawatiran utamanya adalah bahwa sifat tidak terkendali (*unconstrained nature*) dari HC **merusak sifat pemetaan identitas** saat arsitektur ini memanjang melintasi banyak lapisan. Dalam arsitektur yang terdiri dari berbagai aliran paralel, sebuah pemetaan identitas yang ideal berfungsi sebagai **mekanisme konservasi**. Hal ini memastikan bahwa intensitas rata-rata dari sinyal (*average signal intensity*) melintasi aliran tetap konstan (invarian) baik selama propagasi maju (*forward pass*) maupun propagasi mundur (*backward propagation*). Memperluas HC secara rekursif ke beberapa lapisan melalui Persamaan (3) menghasilkan:

$$ \mathbf{x}_L = \left( \prod_{i=l}^{L-1} \mathcal{H}_i^{\text{res}} \right) \mathbf{x}_l + \sum_{i=l}^{L-1} \left( \prod_{j=l}^{L-1-i} \mathcal{H}_{L-j}^{\text{res}} \right) \mathcal{H}_i^{\text{post} \top} \mathcal{F}(\mathcal{H}_i^{\text{pre}} \mathbf{x}_i, \mathcal{W}_i), \tag{4} $$

**Penjelasan Rumus (4):**
*   $L$ dan $l$: Merepresentasikan lapisan dalam dan lapisan dangkal.
*   $\prod$: Simbol perkalian berkelanjutan (perkalian matriks).
*   **Analisis Kritis:** Berbeda secara tragis dengan Persamaan (2), pemetaan komposit $\prod_{i=l}^{L-1} \mathcal{H}_i^{\text{res}}$ di dalam HC *gagal* memelihara rata-rata global (*global mean*) dari fitur. Karena matriks $\mathcal{H}^{\text{res}}$ saling dikalikan berulang kali, nilai sinyal dapat dengan mudah meledak ke titik tak terhingga (*signal amplification/exploding*) atau menyusut menjadi nol (*attenuation/vanishing*), yang mengakibatkan ketidakstabilan parah selama pelatihan skala besar. 

Pertimbangan lebih lanjut adalah bahwa, meskipun HC melestarikan efisiensi komputasional dalam hal FLOPs, **efisiensi perangkat keras mengenai biaya akses memori** untuk aliran residual yang melebar tersebut tetap tidak terselesaikan dalam desain aslinya. Faktor-faktor ini secara kolektif membatasi skalabilitas praktis dari HC dan menghambat aplikasinya dalam pelatihan skala besar.

Untuk mengatasi serangkaian tantangan tersebut, kami mengusulkan **Manifold-Constrained Hyper-Connections (mHC)**, seperti yang ditunjukkan pada Gambar 1(c). Ini adalah kerangka kerja umum yang memproyeksikan ruang koneksi residual dari HC ke sebuah *manifold* spesifik guna memulihkan sifat pemetaan identitas, sembari menggabungkan optimisasi infrastruktur yang ketat untuk memastikan efisiensi. 

Secara spesifik, *mHC* memanfaatkan algoritma **Sinkhorn-Knopp** (Sinkhorn and Knopp, 1967) untuk secara entropis memproyeksikan matriks residual $\mathcal{H}_l^{\text{res}}$ ke dalam politop Birkhoff (*Birkhoff polytope*). Operasi matematika ini secara efektif membatasi matriks koneksi residual ke dalam *manifold* yang direpresentasikan oleh **matriks stokastik ganda (*doubly stochastic matrices*)**. Karena jumlah baris dan jumlah kolom dari matriks jenis ini selalu persis sama dengan 1, maka operasi $\mathcal{H}_l^{\text{res}} \mathbf{x}_l$ bertindak semata-mata sebagai kombinasi cembung (*convex combination*) dari fitur-fitur *input*. 

Karakteristik ini memfasilitasi propagasi sinyal yang terkondisi dengan sangat baik (*well-conditioned*), di mana rata-rata fitur (energi sinyal) dipertahankan, dan *norm* (panjang vektor) dari sinyal diregularisasi secara ketat, secara efektif memitigasi risiko sinyal yang menghilang atau meledak. Lebih jauh lagi, karena sifat ketertutupan perkalian matriks (*closure of matrix multiplication*) untuk matriks stokastik ganda, pemetaan komposit $\prod_{i=l}^{L-1} \mathcal{H}_i^{\text{res}}$ tetap mempertahankan properti konservasi ini. Alhasil, *mHC* secara krusial memelihara stabilitas pemetaan identitas pada tingkat kedalaman arsitektur yang sewenang-wenang. 

Untuk memastikan efisiensi eksekusi pada GPU, kami mempekerjakan peleburan *kernel* (*kernel fusion*) dan mengembangkan *kernel* presisi campuran (*mixed precision kernels*) menggunakan *TileLang* (Wang et al., 2025). Selanjutnya, kami memitigasi jejak memori (*memory footprint*) melalui penghitungan ulang selektif (*selective recomputing*) dan dengan hati-hati melakukan tumpang-tindih komunikasi (*overlap communication*) di dalam penjadwalan *DualPipe* (Liu et al., 2024b).

Eksperimen ekstensif pada prapelatihan model bahasa mendemonstrasikan bahwa *mHC* memamerkan stabilitas dan skalabilitas yang luar biasa sembari mempertahankan keuntungan performa absolut dari HC. Indikasi dari pelatihan skala besar internal (*in-house*) kami menunjukkan bahwa *mHC* mendukung pelatihan secara stabil pada skala masif dan *hanya* menimbulkan tambahan *overhead* waktu komputasi yang sangat kecil (sekitar 6,7%) ketika laju ekspansi ($n$) ditetapkan pada 4.

---

## 2. Pekerjaan Terkait (*Related Works*)

Kemajuan arsitektur dalam pembelajaran mendalam (*deep learning*) utamanya dapat diklasifikasikan ke dalam **desain mikro (*micro-design*)** dan **desain makro (*macro-design*)**. 
*   **Desain Mikro** berkaitan dengan arsitektur *internal* dari blok komputasional; merinci bagaimana fitur-fitur data diproses melintasi dimensi spasial, temporal, dan *channel*. 
*   Sebaliknya, **Desain Makro** membentuk struktur topologis *antar-blok*; mendikte bagaimana representasi fitur itu dipropagasikan (diteruskan), dirutekan (*routed*), dan digabungkan melintasi lapisan-lapisan yang berbeda.

### 2.1 Desain Mikro (*Micro Design*)

Didorong oleh pembagian parameter (*parameter sharing*) dan invarian translasi (*translation invariance*), teknik konvolusi pada awalnya mendominasi pemrosesan sinyal terstruktur (seperti gambar). Meskipun berbagai variasi selanjutnya—seperti konvolusi terpisah kedalaman (*depthwise separable*) (Chollet, 2017) dan konvolusi berkelompok (*grouped convolutions*) (Xie et al., 2017)—mengoptimalkan efisiensi, kemunculan *Transformers* (Vaswani et al., 2017) telah menetapkan mekanisme Atensi (*Attention*) dan Jaringan Maju-Umpan (*Feed-Forward Networks* / FFNs) sebagai blok bangunan fundamental dari arsitektur modern. 

Mekanisme Atensi memfasilitasi propagasi informasi secara global (antar kata yang saling berjauhan), sedangkan FFN meningkatkan kapasitas representasi dari setiap fitur individu. Untuk menyeimbangkan performa analitis dengan tuntutan komputasional raksasa dari LLM, mekanisme atensi telah berevolusi menuju varian-varian yang efisien seperti *Multi-Query Attention* (MQA) (Shazeer, 2019), *Grouped-Query Attention* (GQA) (Ainslie et al., 2023), dan *Multi-Head Latent Attention* (MLA) (Liu et al., 2024a). Secara bersamaan, FFN telah digeneralisasi menjadi paradigma komputasi renggang (*sparse computing paradigms*) melalui *Mixture-of-Experts* (MoE) (Fedus et al., 2022; Lepikhin et al., 2020; Shazeer et al., 2017), yang memungkinkan penskalaan parameter secara masif tanpa harus meningkatkan biaya komputasional secara proporsional.

### 2.2 Desain Makro (*Macro Design*)

Desain Makro mengatur topologi global dari jaringan neural (Srivastava et al., 2015). Menyusul kesuksesan ResNet (He et al., 2016a), arsitektur lain seperti DenseNet (Huang et al., 2017) dan FractalNet (Larsson et al., 2016) bertujuan untuk meningkatkan performa dengan cara meningkatkan kompleksitas topologis melalui konektivitas yang padat dan struktur multi-jalur. *Deep Layer Aggregation* (DLA) (Yu et al., 2018) lebih lanjut mengekspansi paradigma ini dengan secara rekursif mengumpulkan fitur melintasi berbagai kedalaman dan resolusi.

Baru-baru ini, fokus desain makro telah bergeser menuju "memperluas lebar" (*expanding the width*) dari aliran residual (*residual stream*) (Chai et al., 2020; Fang et al., 2023; Heddes et al., 2025; Mak and Flanigan, 2025; Menghani et al., 2025; Pagliardini et al., 2024; Xiao et al., 2025; Xie et al., 2023; Zhu et al., 2024). 
*   **Hyper-Connections (HC)** (Zhu et al., 2024) memperkenalkan matriks-matriks yang dapat dipelajari (*learnable matrices*) untuk memodulasi kekuatan koneksi antar fitur pada berbagai kedalaman.
*   **Residual Matrix Transformer (RMT)** (Mak and Flanigan, 2025) mengganti aliran residual standar dengan memori matriks *outer-product* untuk memfasilitasi penyimpanan fitur. 
*   Serupa dengan itu, **MUDDFormer** (Xiao et al., 2025) mempekerjakan koneksi padat dinamis multi-jalur untuk mengoptimalkan aliran informasi lintas-lapisan. 

Meskipun model-model ini menjanjikan secara potensi, pendekatan ini **mencederai sifat pemetaan identitas yang melekat pada koneksi residual**, sehingga menyuntikkan penyakit ketidakstabilan dan menghalangi skalabilitas ke ukuran model raksasa. Lebih parah lagi, metode-metode tersebut memicu **overhead akses memori yang parah** dikarenakan melebarnya fitur-fitur aliran residual. 

Dibangun dengan bertumpu pada arsitektur HC, *mHC* yang kami usulkan memberikan "kekang/batasan" pada ruang koneksi residual dengan menguncinya ke sebuah *manifold* yang spesifik. Tujuannya adalah untuk mengembalikan sifat pemetaan identitas (agar model tidak meledak atau amnesia), sembari merangkum optimalisasi infrastruktur yang ketat untuk menjamin efisiensi eksekusi *hardware*. Pendekatan matematis ini sukses mendongkrak stabilitas dan skalabilitas sembari tetap mempertahankan keuntungan topologis luar biasa dari melebarnya koneksi-koneksi tersebut.

---

## 3. Pendahuluan Teknis (*Preliminary*)

Pertama-tama, kita rumuskan notasi matematis yang digunakan dalam karya ini. Dalam formulasi HC, input ke lapisan ke-$l$, yang awalnya adalah matriks $\mathbf{x}_l \in \mathbb{R}^{1 \times C}$, diekspansi (diperluas) oleh sebuah faktor perkalian $n$ untuk menyusun sebuah matriks tersembunyi (*hidden matrix*) $\mathbf{x}_l = (\mathbf{x}_{l,0}^\top, \ldots, \mathbf{x}_{l,n-1}^\top)^\top \in \mathbb{R}^{n \times C}$. Struktur ini dapat dipandang sebagai sisa aliran dengan $n$ jalur pipa paralel ($n$-stream residual). 

Operasi ekspansi ini secara efektif memperlebar diameter aliran residual. Untuk mengatur proses pembacaan (*read-out*), penulisan-masuk (*write-in*), dan pemutakhiran proses-proses di dalam aliran ini, HC memperkenalkan tiga buah matriks linier yang dapat dilatih (*learnable linear mappings*)—$\mathcal{H}_l^{\text{pre}}, \mathcal{H}_l^{\text{post}} \in \mathbb{R}^{1 \times n}$, dan $\mathcal{H}_l^{\text{res}} \in \mathbb{R}^{n \times n}$. Pemetaan-pemetaan ini memodifikasi koneksi residual standar yang ditunjukkan pada Persamaan (1), menghasilkan formulasi yang dimodifikasi pada Persamaan (3).

Dalam arsitektur HC, pemetaan yang dapat dipelajari (*learnable mappings*) terdiri dari dua bagian koefisien: 
1. Bagian yang bergantung pada input (*input-dependent*), yang disebut pemetaan dinamis (*dynamic mappings*).
2. Bagian global, yang disebut pemetaan statis (*static mappings*). 

Secara formal, HC menghitung koefisien tersebut sebagai berikut:

$$ \begin{cases} \tilde{\mathbf{x}}_l = \text{RMSNorm}(\mathbf{x}_l) \\ \mathcal{H}_l^{\text{pre}} = \alpha_l^{\text{pre}} \cdot \tanh(\boldsymbol{\theta}_l^{\text{pre}} \tilde{\mathbf{x}}_l^\top) + \mathbf{b}_l^{\text{pre}} \\ \mathcal{H}_l^{\text{post}} = \alpha_l^{\text{post}} \cdot \tanh(\boldsymbol{\theta}_l^{\text{post}} \tilde{\mathbf{x}}_l^\top) + \mathbf{b}_l^{\text{post}} \\ \mathcal{H}_l^{\text{res}} = \alpha_l^{\text{res}} \cdot \tanh(\boldsymbol{\theta}_l^{\text{res}} \tilde{\mathbf{x}}_l^\top) + \mathbf{b}_l^{\text{res}}, \end{cases} \tag{5} $$

**Penjelasan Rumus (5):**
*   $\text{RMSNorm}(\cdot)$ (Zhang and Sennrich, 2019): Diterapkan pada dimensi terakhir (untuk menormalisasi input sebelum diproses).
*   $\alpha_l^{\text{pre}}, \alpha_l^{\text{post}}, \alpha_l^{\text{res}} \in \mathbb{R}$: Adalah skalar (*gating factors*) yang dapat dipelajari dan diinisialisasi ke nilai yang sangat kecil (untuk menjaga stabilitas awal).
*   $\boldsymbol{\theta}_l^{\text{pre}}, \boldsymbol{\theta}_l^{\text{post}} \in \mathbb{R}^{1 \times C}$ dan $\boldsymbol{\theta}_l^{\text{res}} \in \mathbb{R}^{n \times C}$: Ini adalah matriks proyeksi linier (*linear projections*) yang membentuk *dynamic mappings* (ia berubah-ubah nilainya tergantung apa input $\tilde{\mathbf{x}}_l$ nya).
*   $\mathbf{b}_l^{\text{pre}}, \mathbf{b}_l^{\text{post}} \in \mathbb{R}^{1 \times n}$ dan $\mathbf{b}_l^{\text{res}} \in \mathbb{R}^{n \times n}$: Ini adalah parameter *bias* statis yang dapat dipelajari (*static mappings*).

Patut dicatat bahwa penyisipan pemetaan ini ($\mathcal{H}^{\text{pre}}, \mathcal{H}^{\text{post}}$, dan $\mathcal{H}^{\text{res}}$) **hanya membebankan tambahan komputasi (FLOPs) yang sangat dapat diabaikan (*negligible*)**. Mengapa? Karena laju ekspansinya (nilai $n$), misalnya $n=4$, ukurannya *jauh lebih kecil* ketimbang dimensi input jaringan aslinya ($C$). 

Dengan desain ini, HC secara brilian mampu memisahkan (*decouples*) kapasitas informasi di dalam aliran residual dari beban dimensi input aslinya. Karena hal ini, HC menawarkan jalan baru bagi penskalaan arsitektur LLM: kita bisa memperlebar "jalan tol" aliran residual ini sebagai metode penskalaan (scaling), melengkapi hukum penskalaan tradisional (*traditional scaling laws*) yang melulu hanya membesarkan FLOPs dan jumlah data latih (Hoffmann et al., 2022).

Meskipun HC mensyaratkan tiga buah matriks pemetaan untuk mengurus ketidakcocokan dimensi, eksperimen awal kami yang disajikan pada Tabel 1 (di bawah ini) mengindikasikan bahwa **matriks pemetaan residual $\mathcal{H}_l^{\text{res}}$ adalah penyumbang (*driver*) utama dari lonjakan perolehan performa (kecerdasan model)**. Temuan ini menegaskan betapa krusialnya proses pertukaran informasi (mengaduk aliran data) di dalam aliran residual itu sendiri.

#### Tabel 1: Studi Ablasi Komponen-Komponen HC (*Ablation Study of HC Components*)

*(Tabel ini menguji apa yang terjadi jika salah satu matriks pemetaan HC dimatikan/dinonaktifkan).*

| $\mathcal{H}^{\text{res}}$ | $\mathcal{H}^{\text{pre}}$ | $\mathcal{H}^{\text{post}}$ | Absolut Loss Gap (Jarak Kerugian Absolut) |
| :---: | :---: | :---: | :---: |
| | | | 0.0 (Baseline Murni Tanpa HC) |
| $\checkmark$ | | | $- 0.022$ |
| $\checkmark$ | $\checkmark$ | | $- 0.025$ |
| $\checkmark$ | $\checkmark$ | $\checkmark$ | **$- 0.027$** (Semua komponen aktif menghasilkan *loss error* terendah) |

*(Catatan Dosen: Saat sebuah pemetaan spesifik dinonaktifkan di tabel ini, ia diganti dengan "pemetaan tetap/pasif" (*fixed mapping*) demi konsistensi dimensi matematika. Misalnya: bobot dibagi rata $1/n$ untuk $\mathcal{H}^{\text{pre}}$, bobot selalu 1 untuk $\mathcal{H}^{\text{post}}$, dan matriks identitas murni untuk $\mathcal{H}^{\text{res}}$).*

---

### 3.1 Ketidakstabilan Numerikal (*Numerical Instability*)

Walaupun matriks pemetaan residual $\mathcal{H}_l^{\text{res}}$ bertindak sebagai instrumen ajaib dalam mendongkrak performa intelegensi, **aplikasi sekuensialnya (saat ditumpuk di banyak *layer*) memicu malapetaka ketidakstabilan numerik (*numerical stability*)**. 

Seperti yang telah didiagnosis pada Persamaan (4), ketika HC dieksekusi memanjang melewati banyak lapisan (misalnya 60 layer pada LLM modern), propagasi aliran sinyal efektif dari lapisan $l$ ke lapisan jauh $L$ akan dimodulasi oleh perkalian matriks komposit: $\prod_{i=l}^{L-1} \mathcal{H}_i^{\text{res}}$. 

Karena matriks pemetaan $\mathcal{H}_l^{\text{res}}$ yang dapat dipelajari tersebut bersifat **tidak dibatasi (*unconstrained*)** (angkanya bebas bergerak ke nilai berapa pun), maka hasil perkalian komposit ini secara tak terhindarkan akan terdeviasi (melenceng jauh) dari pemetaan identitas (skala 1). Sebagai dampak fatalnya, magnitudo sinyal (besaran angka) akan sangat rentan terhadap ledakan (*explosion*, angka miliaran) atau penyusutan gaib (*vanishing*, angka nol) selama proses propagasi maju (*forward*) dan propagasi mundur gradien (*backpropagation*). Fenomena mengerikan ini menumbangkan premis dasar dari prinsip "Pembelajaran Residual" (*residual learning*), yang absolut bergantung pada aliran sinyal yang lancar dan tak terhalang. Hal ini pada akhirnya akan mendestabilisasi (menghancurkan) proses pelatihan pada model LLM yang lebih dalam atau berskala raksasa.

Bukti-bukti empiris kami sangat mendukung analisis di atas. Mari kita lihat **Gambar 2** dan **Gambar 3**.

### Analisis Akademik Gambar 2: Ketidakstabilan Pelatihan Hyper-Connections (HC)

*   **Deskripsi Visual:** Terdapat dua grafik. (a) Menunjukkan Celah Kerugian Absolut (*Absolute Loss Gap*) pada proses *training*. (b) Menunjukkan Norma Gradien (*Gradient Norm*).
*   **Sumbu X (Horizontal):** Adalah langkah-langkah pelatihan (*Training Steps*) dari 0 hingga 50.000.
*   **Sumbu Y (Vertikal):** (a) Perbedaan nilai kerugian (*loss*), (b) Magnitudo dari vektor gradien.
*   **Kurva Data:**
    *   **Garis Biru (HC):** Mewakili arsitektur HC standar (tanpa batasan).
    *   **Garis Hitam (mHC):** Mewakili inovasi makalah ini, *mHC*.
*   **Kesimpulan Akademis:**
    *   Pada grafik (a), perhatikan garis biru (HC). Sekitar langkah ke-12.000, terjadi ledakan/lonjakan aneh ke atas (*unexpected loss surge*). Ini menunjukkan model kebingungan atau mengalami *error* konseptual (karena arsitekturnya bergejolak). Sebaliknya, garis hitam (mHC) sangat stabil dan lurus.
    *   Pada grafik (b), gradien norma dari HC (biru) sangat fluktuatif (naik turun liar seperti detak jantung tak wajar), yang menjadi biang keladi dari kegagalan pelatihan model raksasa. Kurva mHC (hitam) meluncur mulus dan tenang.

### Analisis Akademis Gambar 3: Ketidakstabilan Propagasi (Ledakan Sinyal) HC

Untuk lebih mengkuantifikasi mengapa HC mengalami anomali, para peneliti membuat **Gambar 3** yang membedah langsung isi dari perkalian matriksnya.

*   **Deskripsi Visual:** Ini adalah pengukuran *Amax Gain Magnitude* (Tingkat pembesaran maksimum).
*   **Sumbu X (Horizontal):** Indeks Lapisan / *Layer* model (1 hingga 60).
*   **Sumbu Y (Vertikal):** Pembesaran Magnitudo (Skala Logaritmik: $10^0, 10^1, 10^2, 10^3, 10^4$). Angka $10^0$ (yaitu 1) adalah nilai identitas ideal (sinyal tidak membesar/menyusut).
*   **Gambar 3 (a) - Matriks Tunggal ($\mathcal{H}_l^{\text{res}}$):** Pada satu lapisan, sinyal maju (hitam) dan sinyal mundur (biru) dari HC masih terlihat "jinak" di kisaran $10^0$ hingga $10^1$.
*   **Gambar 3 (b) - Matriks Komposit ($\prod_{i=1}^{61} \mathcal{H}_i^{\text{res}}$):** Ini adalah saat bencana terjadi! Ketika HC diterapkan melintasi semua 60 layer, perkalian matriksnya meledak. Garis biru (gradien mundur / *Backward Gradient Gain*) meroket ganas hingga menyentuh pilar $10^3$ hingga $10^4$ (puncak di angka 3000!).
*   **Kesimpulan Akademis:** Penyimpangan ekstrem (ledakan ribuan kali lipat) dari angka dasar "1" ini mengkonfirmasi secara mutlak adanya *exploding residual streams* (aliran sinyal sisa yang meledak terbakar) di dalam mesin HC standar. Model dipastikan akan hancur saat dilatih.

---

### 3.2 Overhead Sistem (Beban Arsitektural)

Meskipun kompleksitas matematis komputasi (FLOPs) dari arsitektur HC masih pada tingkat yang aman dan dapat dikendalikan, ada "hantu" biaya infrastruktur di tingkat sistem yang tidak boleh diabaikan. Secara spesifik, biaya operasi Input/Output memori (I/O) untuk membaca dan menulis data ke RAM GPU sering kali menjadi hambat-botol (*bottleneck*) pembunuh utama dalam arsitektur model modern. Situasi fatal ini di industri AI sering dijuluki sebagai **dinding memori (*memory wall*)** (Dao et al., 2022). Kendala krusial ini sayangnya sangat sering dilupakan oleh desainer algoritma AI, padahal dampaknya sangat menghancurkan efisiensi saat kode berjalan (*runtime efficiency*).

Dengan membedah arsitektur Transformer tipe *pre-norm* yang mendominasi saat ini (Vaswani et al., 2017), kami menganalisis pola kemacetan I/O yang secara bawaan menempel pada HC. 

### Tabel 2: Perbandingan Biaya Akses Memori Per Token
*(Tabel ini menghitung beban transfer data akibat melebarnya aliran residual, mengabaikan hitungan berat dari fungsi FFN atau Atensi itu sendiri).*

| Metode | Operasi Matematika | Read (Elemen Dibaca) | Write (Elemen Ditulis) |
| :--- | :--- | :---: | :---: |
| **Residual Biasa** | Penggabungan Residual ($\mathbf{x}_{l+1} = \mathbf{x}_l + ...$) | $2C$ | $C$ |
| | **Total I/O** | **$2C$** | **$C$** |
| **Hyper-Connections (HC)** | Kalkulasi $\mathcal{H}_l^{\text{pre}}, \mathcal{H}_l^{\text{post}}, \mathcal{H}_l^{\text{res}}$ | $nC$ | $n^2 + 2n$ |
| | Perhitungan $\mathcal{H}_l^{\text{pre}}$ | $nC + n$ | $C$ |
| | Perhitungan $\mathcal{H}_l^{\text{post}}$ | $C + n$ | $nC$ |
| | Perhitungan $\mathcal{H}_l^{\text{res}}$ | $nC + n^2$ | $nC$ |
| | Penggabungan Residual | $2nC$ | $nC$ |
| | **Total I/O** | **$(5n + 1)C + n^2 + 2n$**| **$(3n + 1)C + n^2 + 2n$**|

**Kajian Akademis:**
*   $C$: Dimensi vektor asli. $n$: Laju ekspansi HC (jumlah jalur paralel, misal 4).
*   Perhatikan Total I/O. Arsitektur Residual klasik (seperti LLaMA) hanya membakar memori sebanyak $2C$ Baca dan $C$ Tulis.
*   Bandingkan dengan HC. HC menggelembungkan beban Baca dan Tulis ke dalam orde $(5n+1)C$ dan $(3n+1)C$. Ini berarti lalu lintas transfer data ke memori **naik secara gila-gilaan berbanding lurus dengan nilai $n$**!
*   **Dampak:** Tuntutan rakus I/O ini (karena pita residualnya mekar menjadi $nC$) akan **membunuh (mendegradasi)** *throughput* (laju) kecepatan pelatihan secara brutal apabila kita tidak menginjeksi teknik penggabungan *kernel (fused kernels)* tingkat dewa ke dalam GPU. 
*   Selain itu, karena matriks $\mathcal{H}^{\text{pre}}, \mathcal{H}^{\text{post}}$, dan $\mathcal{H}^{\text{res}}$ memiliki parameter yang "hidup" (*learnable*), memori aktivasi perantaranya harus disimpan (*cached*) untuk proses mundur (backpropagation). Ini melipatgandakan jejak memori GPU (*GPU memory footprint*), dan memaksa *engineer* AI untuk mengerahkan jurus pengereman (*gradient checkpointing*) agar GPU tidak *Out-of-Memory* (OOM). 
*   Terakhir, saat melatih model raksasa antar-GPU (*pipeline parallelism*), membesarnya $n$ berarti data yang ditransfer via kabel jaringan NVLink antar GPU membengkak $n$-kali lipat (Qi et al., 2024), menyebabkan *bubble* (waktu tunggu kosong GPU) melebar dan membuang-buang uang.

---

## 4. Metode (*Method*)

### 4.1. Manifold-Constrained Hyper-Connections (*Koneksi-Hiper dengan Kendala Manifold*)

Mengambil inspirasi dari prinsip pemetaan identitas (He et al., 2016b), premis inti dari penemuan **mHC** adalah **memberi rantai pembatas (kendala/ *constrain*)** pada matriks pemetaan residual HC ($\mathcal{H}_l^{\text{res}}$) dan "memaksanya" masuk ke dalam sebuah kurungan matematis yang disebut *manifold* spesifik. 

Meskipun prinsip pemetaan identitas asli dari He dkk (yaitu dengan mengatur $\mathcal{H}_l^{\text{res}} = \mathbf{I}$, matriks identitas kaku) menjamin kelancaran, hal ini *secara fundamental melarang terjadinya pertukaran informasi (pengadukan data) silang-aliran* di dalam arus residual. Padahal, pertukaran informasi inilah kunci dari kecerdasan multi-aliran.

Oleh karenanya, kami mengusulkan untuk memproyeksikan matriks pemetaan residual tersebut ke dalam sebuah *manifold* yang **secara magis mempertahankan stabilitas dari aliran sinyal lintas lapisan**, namun **tetap mengizinkan terjadinya interaksi dinamis/pengadukan (mutual interaction)**. 

Untuk merealisasikan tujuan ini, kami merantai matriks $\mathcal{H}_l^{\text{res}}$ agar HANYA boleh bertingkah laku sebagai sebuah **matriks stokastik ganda (*doubly stochastic matrix*)**. Matriks ini adalah spesies matriks yang sangat eksklusif, yang hanya memiliki entri angka non-negatif (tidak ada nilai minus), dan di mana **penjumlahan angka di setiap baris (horizontal) DAN setiap kolom (vertikal) harus secara absolut sama persis dengan angka 1**.

Secara formal, biarkan $\mathcal{M}^{\text{res}}$ melambangkan *manifold* yang dibentuk oleh sekumpulan matriks stokastik ganda (dalam geometri, kawasan ini dikenal sebagai poligon/politop Birkhoff / *Birkhoff polytope*). Kami menjebak $\mathcal{H}_l^{\text{res}}$ ke dalam kurungan $P_{\mathcal{M}^{\text{res}}}(\mathcal{H}_l^{\text{res}})$, yang didefinisikan sebagai:

$$ P_{\mathcal{M}^{\text{res}}}(\mathcal{H}_l^{\text{res}}) := \left\{ \mathcal{H}_l^{\text{res}} \in \mathbb{R}^{n \times n} \mid \mathcal{H}_l^{\text{res}} \mathbf{1}_n = \mathbf{1}_n, \mathbf{1}_n^\top \mathcal{H}_l^{\text{res}} = \mathbf{1}_n^\top, \mathcal{H}_l^{\text{res}} \geqslant 0 \right\}, \tag{6} $$

**Penjelasan Rumus (6):**
*   $\mathbb{R}^{n \times n}$: Matriks bujur sangkar berukuran $n \times n$.
*   $\mathbf{1}_n$: Mewakili vektor vertikal berdimensi $n$ yang isinya angka 1 semua (cth: $[1, 1, 1]^T$).
*   Syarat $\mathcal{H}_l^{\text{res}} \mathbf{1}_n = \mathbf{1}_n$: Ini adalah bahasa matematika yang memaksa **jumlah (sum) setiap baris harus bernilai tepat 1**.
*   Syarat $\mathbf{1}_n^\top \mathcal{H}_l^{\text{res}} = \mathbf{1}_n^\top$: Memaksa **jumlah (sum) setiap kolom harus bernilai tepat 1**.
*   Syarat $\mathcal{H}_l^{\text{res}} \geqslant 0$: Mencegah adanya elemen bernilai negatif.

*Catatan Dosen:* Coba bayangkan apa yang terjadi bila laju ekspansinya cuma 1 (yakni $n=1$). Kondisi matriks stokastik ganda ini akan mengerut menjadi angka skalar absolut "1", yang secara otomatis membangkitkan kembali prinsip pemetaan identitas (ResNet) yang asli! 

Pilihan memaksa matriks menjadi stokastik ganda ini mengaruniakan beberapa mukjizat properti matematis (teoretis) yang sangat vital bagi stabilitas pelatihan model berukuran raksasa:

1.  **Pelestarian Norma (*Norm Preservation*):** Nilai norma spektral dari sebuah matriks stokastik ganda terjamin selalu kurang dari atau sama dengan 1 (yaitu, $\| \mathcal{H}_l^{\text{res}} \|_2 \le 1$). Apa maknanya? Artinya fungsi *learnable* ini secara harfiah tidak akan pernah mengekspansi (melipatgandakan) total energi data yang melewatinya. Ini **secara efektif membunuh penyakit ledakan gradien (*gradient explosion problem*)**.
2.  **Ketertutupan Komposisi (*Compositional Closure*):** Domain matematika matriks stokastik ganda memiliki sifat "tertutup" di bawah perkalian matriks. Artinya, berapapun banyak matriks tipe ini dikalikan secara berantai menyusuri miliaran *layer* model: $\prod_{i=l}^{L-1} \mathcal{H}_{L-i}^{\text{res}}$, hasil akhirnya akan TETAP berwujud sebuah matriks stokastik ganda. Properti ini menjadi segel jaminan stabilitas seutuhnya ke segenap kedalaman terdalam dari lautan parameter arsitektur model.
3.  **Interpretasi Geometris via Politop Birkhoff (*Geometric Interpretation via the Birkhoff Polytope*):** Himpunan $\mathcal{M}^{\text{res}}$ ini membentuk sebuah bangun ruang hiper yang disebut politop Birkhoff, yang sejatinya merupakan *cangkang cembung* (lambung / *convex hull*) dari sebuah "Matriks Permutasi" (Matriks tukar-posisi). Artinya, algoritma *mHC* dapat diterjemahkan secara geometris sebagai **operasi kombinasi cembung terhadap ragam permutasi pengadukan fitur**. Secara matematis, penerapan matriks ini secara repetitif akan selalu meningkatkan proses **pencampuran informasi lintas arus** (*mixing of information across streams*) secara konsisten dan terarah (monoton), bertindak sebagai mekanisme penyatuan fitur (*feature fusion*) yang paling superior dan stabil.

Sebagai tambahan kendali kemudi, kami memaksakan batasan nilai **non-negatif** (hanya angka plus) pada matriks pemetaan pintu masuk ($\mathcal{H}_l^{\text{pre}}$) dan matriks pemetaan pintu keluar ($\mathcal{H}_l^{\text{post}}$). Batasan ini melarang terjadinya "Penyitaan Sinyal" (*signal cancellation*) yang fatal, yang dapat diakibatkan oleh benturan angka matriks yang saling mengurangi. Hal ini juga dapat dipandang sebagai bentuk lain dari intervensi proyektil *manifold*.

### 4.2 Parameterisasi dan Proyeksi *Manifold*

Di sub-bab teknis ini, kami membongkar urutan matematis (*calculation process*) dari $\mathcal{H}_l^{\text{pre}}, \mathcal{H}_l^{\text{post}}$, dan $\mathcal{H}_l^{\text{res}}$ di dalam tubuh mesin *mHC*. 

Diberikan sebuah tensor input yang disembunyikan (*input hidden matrix*) berukuran $\mathbf{x}_l \in \mathbb{R}^{n \times C}$ pada lapisan ke-$l$. Kami mula-mula me- *flatten* (meratakan/memipihkan) matriks ini menjadi sebuah bentuk vektor horizontal panjang: $\vec{\mathbf{x}}_l = \text{vec}(\mathbf{x}_l) \in \mathbb{R}^{1 \times nC}$ guna memelihara informasi keseluruhan (*full context*). 

Kemudian, kami meminjam perumusan HC yang fundamental untuk menciptakan parameter "Mentah/Kotor" (yang belum difilter oleh Manifold), ditandai dengan tilde ($\sim$):

$$ \begin{cases} \vec{\mathbf{x}}'_l = \text{RMSNorm}(\vec{\mathbf{x}}_l) \\ \tilde{\mathcal{H}}_l^{\text{pre}} = \alpha_l^{\text{pre}} \cdot (\vec{\mathbf{x}}'_l \boldsymbol{\varphi}_l^{\text{pre}}) + \mathbf{b}_l^{\text{pre}} \\ \tilde{\mathcal{H}}_l^{\text{post}} = \alpha_l^{\text{post}} \cdot (\vec{\mathbf{x}}'_l \boldsymbol{\varphi}_l^{\text{post}}) + \mathbf{b}_l^{\text{post}} \\ \tilde{\mathcal{H}}_l^{\text{res}} = \alpha_l^{\text{res}} \cdot \text{mat}(\vec{\mathbf{x}}'_l \boldsymbol{\varphi}_l^{\text{res}}) + \mathbf{b}_l^{\text{res}}, \end{cases} \tag{7} $$

**Penjelasan Rumus (7):**
*   Hampir identik dengan Persamaan 5, bedanya penggunaan $\tanh$ dihilangkan demi efisiensi, dan matriks $\boldsymbol{\theta}$ kini berubah menjadi vektor pengali $\boldsymbol{\varphi}$ dengan ukuran $\boldsymbol{\varphi}_l^{\text{pre}}, \boldsymbol{\varphi}_l^{\text{post}} \in \mathbb{R}^{nC \times n}$ dan $\boldsymbol{\varphi}_l^{\text{res}} \in \mathbb{R}^{nC \times n^2}$.
*   Fungsi $\text{mat}(\cdot)$ adalah perintah untuk melipat ulang (*reshape*) vektor $1 \times n^2$ menjadi matriks persegi $n \times n$. 

Maka, untuk memperoleh matriks yang **bersih dan sudah diikat/diprojeksi ke dalam Manifold matematis yang aman**, kami mengeksekusi operasi berikut:

$$ \begin{cases} \mathcal{H}_l^{\text{pre}} = \sigma(\tilde{\mathcal{H}}_l^{\text{pre}}) \\ \mathcal{H}_l^{\text{post}} = 2\sigma(\tilde{\mathcal{H}}_l^{\text{post}}) \\ \mathcal{H}_l^{\text{res}} = \text{Sinkhorn-Knopp}(\tilde{\mathcal{H}}_l^{\text{res}}), \end{cases} \tag{8} $$

**Penjelasan Rumus (8):**
*   $\sigma(\cdot)$: Adalah fungsi aktivasi *Sigmoid* (memaksa angka yang keluar selalu berada di rentang jangkauan positif $0$ hingga $1$). Mengamankan nilai $\mathcal{H}_l^{\text{pre}}$ agar tidak bernilai minus. Angka $2\sigma$ pada $\mathcal{H}_l^{\text{post}}$ membuat nilainya berkisar antara $0$ hingga $2$ (menjaga ekuilibrium rata-rata di angka 1).
*   **Algoritma Sinkhorn-Knopp:** Ini adalah pahlawan utama kita! Operator sakti ini pertama-tama akan mengubah seluruh sel menjadi angka positif (memakai operasi eksponen), lalu secara membabi-buta dan iteratif (berulang) akan membagi nilai setiap sel (*rescaling*) sedemikian rupa sehingga: *barisnya ditotal jadi 1, kemudian kolomnya ditotal jadi 1*. Ia terus melakukan ini bergantian hingga stabil. 

Secara spesifik, bila kita memiliki titik awal (*start point*) berupa matriks positif murni $\mathbf{M}^{(0)} = \exp(\tilde{\mathcal{H}}_l^{\text{res}})$, maka algoritma penormalan berulangnya beroperasi sebagai berikut:

$$ \mathbf{M}^{(t)} = \mathcal{T}_r \left( \mathcal{T}_c \left( \mathbf{M}^{(t-1)} \right) \right), \tag{9} $$

**Penjelasan Rumus (9):**
*   $\mathcal{T}_r$ dan $\mathcal{T}_c$: Berturut-turut adalah lambang fungsi penormalan baris (*row normalization*) dan kolom (*column normalization*). 
*   Iterasi ini akan secara pasti melabuh (*converges*) membuahkan bentuk absolut dari **matriks stokastik ganda $\mathcal{H}_l^{\text{res}} = \mathbf{M}^{(t_{\text{max}})}$** ketika nilai $t_{\text{max}} \rightarrow \infty$ (tak terhingga).
*   *Trik Praktisi Kode:* Tentu kita tidak punya waktu hingga tak terhingga. Dalam arsitektur laboratorium eksekusi praktis kami, mematok angka repetisi (*loop*) di $t_{\text{max}} = 20$ putaran sudah lebih dari cukup untuk memperoleh rasio yang sangat presisi!

### 4.3 Desain Infrastruktur yang Efisien (*Efficient Infrastructure Design*)

Dalam bab ini, kami menyingkap rancang bangun mesin (infrastruktur perangkat lunak GPU) yang secara mutlak diracik khusus untuk mHC. Melalui penyiksaan uji optimisasi yang sadis, kami sukses menanamkan modul *mHC* (dengan rasio pelebaran aliran $n = 4$) ke dalam raksasa LLM dengan harga perih yang **sangat-sangat murah: penambahan lambatnya waktu *training* hanya sebesar 6,7% saja!**

#### 4.3.1 Fusi Inti (Kernel Fusion)

Mencermati bahwa pemanggilan fungsi `RMSNorm` pada modul mHC (Persamaan 7) menenggak waktu parah ketika bekerja pada status vektor *hidden state* gembrot berdimensi tinggi ($\vec{\mathbf{x}}_l \in \mathbb{R}^{1 \times nC}$), kami mengakali tatanannya (*reorder*). Operasi "pembagian-dengan-norma" kami seret agar dilakukan *setelah* (bukan sebelum) operasi "perkalian matriks".

Trik muslihat matematis ini mempertahankan validitas ekuivalensi hasil hitungan asli, NAMUN secara kilat mendongkrak laju efisiensi prosesor. Selain itu, kami menyetir (*employ*) amunisi presisi-campuran (*mixed-precision strategies*) (menggunakan tipe data 16-bit digabung 32-bit) untuk menjaring margin akurasi matematis tanpa mensabotase kelajuan (FPS) komputer. 

Dan senjata terdahsyat: Operasi gabungan. Kami melakukan peleburan berganda atas berbagai perintah matematika yang sebelumnya punya jalur akses memori berbeda-beda menjadi **"Unified Compute Kernels"** (Pabrik Komputasi Tunggal Bersatu) demi menjebol kemacetan leher botol kapasitas memori silikon RAM. 

Menimbang input dan skema persamaan di bawah (10-13), kami mengecor tiga buah cetakan C++ *kernel mHC* yang terspesialisasi tinggi, hanya untuk menyedot nilai kalkulasi akhir untuk $\mathcal{H}_l^{\text{pre}}, \mathcal{H}_l^{\text{post}}$, dan $\mathcal{H}_l^{\text{res}}$. Di dalam jantung *kernel* leburan ini, sekadar variabel 'bias' biasa (b) dan 'proyeksi linear' (φ) serta bobot pembagi 'RMSNorm' semuanya ditelan lumat disatukan menjadi $\mathbf{b}_l$ dan $\boldsymbol{\varphi}_l$. 

*(Berikut rangkuman kode algoritma kernel yang amat padat. Perhatikan bagaimana tipe data presisi rendah 16-bit (bfloat16) mengemudi vektor raksasa, sementara tipe aman 32-bit (float32) dipakai meracik bumbu pengalinya):*

$$ \boldsymbol{\varphi}_l : \text{tfloat32} \quad [nC, n^2 + 2n] \tag{10} $$
$$ \vec{\mathbf{x}}_l : \text{bfloat16} \quad [1, nC] \tag{11} $$
$$ \alpha_l^{\text{pre}}, \alpha_l^{\text{post}}, \alpha_l^{\text{res}} : \text{float32} \quad \text{Scalars} \tag{12} $$
$$ \mathbf{b}_l : \text{float32} \quad [1, n^2 + 2n] \tag{13} $$

$$ [\tilde{\mathcal{H}}_l^{\text{pre}}, \tilde{\mathcal{H}}_l^{\text{post}}, \tilde{\mathcal{H}}_l^{\text{res}}] : \text{float32} \quad = \vec{\mathbf{x}}_l \boldsymbol{\varphi}_l \tag{14} $$
$$ r : \text{float32} \quad = \| \tilde{\mathbf{x}}_l \|_2 / \sqrt{nC} \tag{15} $$

$$ [\hat{\mathcal{H}}_l^{\text{pre}}, \hat{\mathcal{H}}_l^{\text{post}}, \hat{\mathcal{H}}_l^{\text{res}}] : \text{float32} \quad = 1/r \left[ \alpha_l^{\text{pre}} \tilde{\mathcal{H}}_l^{\text{pre}}, \alpha_l^{\text{post}} \tilde{\mathcal{H}}_l^{\text{post}}, \alpha_l^{\text{res}} \tilde{\mathcal{H}}_l^{\text{res}} \right] + \mathbf{b}_l \tag{16} $$
$$ \mathcal{H}_l^{\text{pre}} : \text{float32} \quad = \sigma(\hat{\mathcal{H}}_l^{\text{pre}}) \tag{17} $$
$$ \mathcal{H}_l^{\text{post}} : \text{float32} \quad = 2\sigma(\hat{\mathcal{H}}_l^{\text{post}}) \tag{18} $$
$$ \mathcal{H}_l^{\text{res}} : \text{float32} \quad = \text{Sinkhorn-Knopp}(\hat{\mathcal{H}}_l^{\text{res}}) \tag{19} $$

**Anatomi Bedah Operasi *Kernel* Silikon:**
*   **Persamaan (14) s.d (15):** Di sini kita mem- *baking* satu *kernel* bersatu padu (*unified*) yang menyedot bacaan input dari vektor $\vec{\mathbf{x}}_l$ secara simultan dua kali. Ini menggunakan manuver catur komputasi matriks yang sangat rakus namun menghemat utilisasi memori maksimum *bandwidth*. Operasi mundur gradien (membaca sisa galat untuk mengajari jaringan) kini digabung jadi satu *kernel*. Ini menyingkirkan keharusan memanggil, memuat, dan mengisi ulang variabel $x$ secara berulang yang mengotori *cache*.
*   **Persamaan (16) s.d (18):** Instruksi manipulasi aritmatika kelas bulu (*lightweight operations*)—seperti pengali angka kecil—kini secara "oportunis dan barbar" dipaksa dirajut tergabung (*fused*) masuki ke dalam satu buah perintah komputasi lajang tunggal (*single kernel*). Dampak? *Overhead* pemanggilan ratusan baris skrip *launch kernel GPU* terpangkas mati!
*   **Persamaan (19):** Iterasi komputasi brutal Sinkhorn-Knopp diblokade diringkus masuk pada sepetak perumahan *kernel* tunggal. Dan saat menghitung koreksi kemunduran (mundur melatih *layer* belakang), kami membangun program mundur sungsang buatan sendiri (Kustom) di mana sirkuit akan memuntahkan ulang *(recomputes)* perhitungan matriks sementara (aktif) murni di selimut RAM cip-dalam (*on-chip* SRAM lokal GPU), menuntaskan operasi literasi.

Dengan bekal selundupan hasil kalkulus fusi sakti dari persandian mesin di atas, lantas diperkenalkan **DUA cetakan inti tambahan (*two additional kernels*)** sebagai algojo pamungkas. 
*   Satu pencetak memegang titah operasi fungsi masuk ($\mathcal{F}_{\text{pre}} \coloneqq \mathcal{H}_l^{\text{pre}} \mathbf{x}_l$).
*   Satu pencetak lagi memegang kendali penampungan operasi fungsi pulang/gabungan ($\mathcal{F}_{\text{post,res}} \coloneqq \mathcal{H}_l^{\text{res}} \mathbf{x}_l + \mathcal{H}_l^{\text{post} \top} \mathcal{F}(\cdot, \cdot)$).

Tarian orkestrasi peleburan ini sangat magis: penggabungan *residual merging* di baris akhir tersebut menciutkan lalu-lintas lalu lalang serapan baca GPU (dari asalnya $(3n + 1)C$ elemen menjadi melompong $(n + 1)C$ elemen saja) dan menciutkan muatan tulis (*write element*) dari $3nC$ turun menyentuh $nC$ untuk *kernel* ini! Kami merekayasa susunan kerajinan tatanan balok digital rumit (*complex calculation process*) tersebut berwadah platform mutakhir **TileLang** (Wang et al., 2025). Fasilitas ini melicinkan pipa algoritma buatan kustom untuk menyedot utuh saripati maksimum nilai batas hisap lebar jalur (*memory bandwidth*) dengan pengurasan keringat lelah ketikan insinyur paling minimalis.

#### 4.3.2. Penghitungan Ulang Paksa (Recomputing)

Paradigma residual paralelisme aliran berganda (alur pipa jalur-$n$) meneror komputasi RAM pelatihan VRAM GPU dengan membawa piala beban gembrot (*substantial memory overhead*). Menghajar teror VRAM *out-of-memory* (OOM) ini, kami mengeksekusi dogma algojo pemenggal ingatan (penghapusan aktivasi perantara).
Seketika sehabis peluru lintasan propagasi maju melesat tamat, keping memori RAM perantara (sementara) milik rahim fungsi *kernel* mHC itu **DIPANCUNG DIBUANG UTUH (*discard the intermediate activations*)**. 
Sihirnya terletak pada saat model dituntut menelaah putaran putar balik mundur mencari revisi gradien (backward pass); sirkuit mesin akan me-*restart* mereplikasi komputasinya terbang melesat seketika (*on-the-fly recomputation*), yaitu memanggil mesin operasi kernel *mHC* menari ulang murni meracik kalkulasi sesaat saja dan... melewatan blok komputasi sadis operasi murni fungsi *layer* $\mathcal{F}$ (seperti FFN). 

Implikasinya sangat indah. Demi menjahit rentetan *layer* model sepanjang blok berantai $L_r$, arsitektur silikon sistem koding kami cukup diinstruksikan menahan dan "mengingat secara mati" (simpan abadi/ *store*) HANYA entitas vektor input pertama $\mathbf{x}_{l_0}$ dari rombongan blok tadi. 
**(Menengok ke Tabel 3)**: Tabel sakral di bawah mengatalogkan inventaris mana keping aktivasi data yang dikubur diselamatkan hidup-hidup (*preserved* / ditaruh ke brankas persisten di VRAM), dan mana debu kalkulasi hantu yang lenyap diciptakan musnah seumur jagung (*transient / recomputed*).

**Tabel 3: Memori Aktivasi yang Disimpan Abadi dan yang Dikloning-Ulang (*Stored and Recomputed Intermediate Activations*)**

*(Dosen: Ingat, tabel ini menjabarkan anatomi pengiritan VRAM yang mutlak vital dalam menyusun kerangka infrastruktur arsitektur raksasa. Kolom yang menyebut $L_r$ adalah gembok batas kelompok. Simbol $\mathbf{x}_{l_0}$ dipertahankan hidup, sementara parameter turunan lainnya adalah ilusi sementara (*Transient*) yang hanya akan merongrong VRAM sekedip mata saat dibutuhkan per layer).*

| Activations (Variabel Hitungan) | $\mathbf{x}_{l_0}$ | $\mathcal{F}(\mathcal{H}_l^{\text{pre}} \mathbf{x}_l, \mathcal{W}_l)$ | $\mathbf{x}_l$ | $\mathcal{H}_l^{\text{pre}} \mathbf{x}_l$ | $\text{RMSNorm}(\mathcal{H}_l^{\text{pre}} \mathbf{x}_l)$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Size (Elements / Beban Memori)** | $nC$ | $C$ | $nC$ | $C$ | $C$ |
| **Stored Method (Sifat Ingatan)** | Setiap kelipatan $L_r$ lapisan | Setiap lapisan (Mutlak) | *Transient* (Ilusi Sementara di dalam kelompok $L_r$ lapisan) | *Transient* | *Transient* |

Oleh karena aksi kebangkitan ulang komputasi (Recomputation) sihir mHC tersebut dipaksa meledak berombongan sekaligus membalut balok raksasa (Blok $L_r$ lapisan secara terus-menerus). Atas premis seutuhnya model punya sumur $L$ kedalaman lapisan, otak model VRAM menahan napas hanya mencengkram persisten keping titik asal $\mathbf{x}_{l_0}$ terhitung sebanyak irisan paksa $\lceil \frac{L}{L_r} \rceil$ balok dalam kawah pembakaran backward. Membuntuti beban memori pendiam tersebut (*resident memory*), siluman proses re-generasi akan merasuki RAM untuk berinkarnasi numpang mampir sesaat (transient memory overhead). Berapa beban "hantu numpang" ini? Merupakan gerombolan $(n+2)C \times L_r$ kepingan unsur matriks. Inilah gembong raksasa monster penguras batasan langit kuota "Puncak Jejak Kapasitas VRAM saat Backward" (*peak memory usage*).
Dengan merapal kalkulus diferensiasi, formula optimasi di bawah dieksekusi menata rasio keseimbangan optimal untuk mengunci blok ukuran ideal $L_r^*$ sehingga kutukan membengkaknya tapak jejak memori terendah (*minimize total memory footprint*) termutlak terlunasi:

$$ L_r^* = \arg\min_{L_r} \left[ nC \times \left\lceil \frac{L}{L_r} \right\rceil + (n+2)C \times L_r \right] \approx \sqrt{ \frac{nL}{n+2} }. \tag{20} $$

Selain itu, hukum besi tatanan arsitektur raksasa Paralelisme Antar-GPU Pipa (*pipeline parallelism constraint*) mencekik ketat: **Gembok bongkahan operasi kalkulasi ulang absolut DILARANG KERAS merobek meretas batas garis lintang tapal batas tahapan transmisi server (pipeline stage boundaries).** Memelototi kaidah di mana titik rasio teoretis angka kalkulus ideal $L_r^*$ secara umum sudah terkalibrasi pas senapas seirama selaras angka besaran jumlah *layer* (lapisan) di dalam satu rak mesin potong *pipeline stage*, maka secara absolut logis kami mematok merantai keseragaman sinkronisasi palang batas blok *recomputation* tadi disamakan secara persis sejajar tembok batasan tahapan *pipeline stage* server.

#### 4.3.3. Penumpangan Tumpang Tindih Transmisi Interkom dalam DualPipe (Overlapping Communication in DualPipe)

Di panggung olimpiade pelatihan AI ukuran planet Bumi (skala super-model), arsitektur Pipa Paralel (*pipeline parallelism*) merupakan de-fakto kiblat peribadatan suci untuk menyembelih beban pemborosan memori jejak injakan monster gradient maupun penyusutan parameter raksasa VRAM. Mendalami filosofi ini, peracikan mesin kami melabuhkan pilihan bersekutu kepada ritme tata-titi protokol jadwal **DualPipe** (Liu et al., 2024b). Siasat tatanan jadwal pengiriman DualPipe merupakan pendekar mutakhir yang secara canggih mampu menimpa-tindihkan (melakukan operasi secara siluman bersamaan / *overlap*) transmisi paket-paket kolosal melintasi jaringan internet kartu kabel raksasa antar node peladen (interconnected communication traffic). Traffic tumpang tindih kilat ini meliputi peluru lalu lalang komputasi *expert parallelism* MoE dan *pipeline parallelism* murni.

NAMUN demikian! Tatkala kita menghadapkan rancangan arsitektur aliran pipa tunggal murni yang lugu (*single-stream design*) kepada kompleksitas gila ciptaan kita di jurnal ini, yakni *Sistem Aliran Arus Residual Jalur-$n$ mHC* ($n$-stream residual in mHC)... Bom waktu meledak! Konstruksi *mHC* menodongkan palak kutipan cukai utang durasi waktu tunda (latensi komunikasi antar pipa peladen / *communication latency across pipeline stages*) yang secara brutal sangat substansial mematikan.

Di samping wabah latensi jaringan tadi, setiap palang pintu gerbang ganti gardu GPU (*stage boundaries*), operasi proses bangkit dari kubur (penghitungan ulang hantu sementara / recomputation of mHC kernels) yang menimpa keseluruhan anggota regu $L_r$ *layers* itu menjatuhkan vonis hukuman penalti hambatan berstatus "Tidak Bisa Diabaikan" (*non-negligible computational overhead*).

Memerangi gerombolan biang keladi penyumbat aliran kecepatan komputasi naga silikon ini (address these bottlenecks), sang perancang kami mengimplan operasi bedah cangkok jantung memanjangkan ritme pernapasan jalur tatanan DualPipe (silakan merujuk visualisasi **Gambar 4**). 

### Analisis Akademis: Gambar 4 - Tumpang Tindih Komunikasi-Komputasi

*   **Deskripsi Visual:** Ini adalah diagram jadwal waktu eksekusi (*Gantt Chart* tingkat perangkat keras) yang memetakan aktivitas tiga jalur paralel di dalam GPU.
*   **Aliran 1 (Normal Compute Stream - Biru Muda/Kuning):** Mengerjakan tugas kalkulasi otak seperti MLP, Attention, dll.
*   **Aliran 2 (Communication Stream - Oranye/Hijau):** Mengerjakan tugas mengirim data bolak-balik melintasi kabel jaringan antar GPU (*Dispatch/Combine* dan *PP Send/Recv*).
*   **Aliran 3 (High Priority Compute Stream - Merah Bata):** Ini adalah inovasi makronya! Menjalankan operasi krusial (Recompute dan gabungan $\mathcal{F}_{\text{post,res}}$).
*   **Kesimpulan:** Dengan menambah satu "jalur tol komputasi khusus VIP" (jalur bawah), GPU dapat dengan licik dan gesit mengalihkan tugas yang sangat berat atau mendesak tanpa menyabotase aliran antrean komunikasi jaringan yang sedang menyedot data di jalur tengah.

Demi menolak keras operasi blokade palang antrean kemacetan sirkulasi nafas pipa data komputasi, operasi baris mesin eksekusi *kernel* cetak biru MLP (FFN) untuk kode $\mathcal{F}_{\text{post,res}}$ kami pecut untuk dibakar tenaganya di lorong eksklusif sirkuit tol VIP tingkat tinggi (*dedicated high-priority compute stream*). 
Makin kejam lagi strateginya: kami mutlak memberikan embargo penolakan merestui operasi sirkuit menahan posisi nafas panjang "berdiam abadi" (*persistent kernels*) untuk segala bentuk jenis komputasi urusan departemen Atensi (Attention layers). Hal ini sukses meluluhlantakkan probabilitas insiden lumpuh koma mogok GPU (*preventing extended stalls*). 
Arsitektur muslihat tingkat ini menganugerahkan ilusi sulap di mana antrean gerbong kereta komputasi layer Attention yang sangat tebal dapat diinterupsi / dibajak (preemption) kapan saja tanpa celaka, meledakkan kebebasan daya kelenturan menyusun jawdal orkestrasi kalender pengerjaan (flexible scheduling) seraya mencengkram rekor raihan utilisasi pemanfaat absolut nyaris 100% pada elemen mesin silikon gergaji (*processing units*) komputasional perangkat tersebut.

Sebagai jurus penutup kombo paripurna, operasi ritus "pembangkitan nyawa dari kubur komputasi hantu sementara" (*recomputation process*) tersebut sukses dipisahkan secara murni (*decoupled*) agar sama sekali buta dan putus hubungan batin (tidak dependen menunggu / *communication dependencies*) dari sinyal instruksi pipa lalu-lintas lintas server (*pipeline communication*). Kebuntuan dipotong karena entitas pusaka bibit input penyala mula pertama (initial activation) untuk setiap regu sirkuit GPU, yakni entitas $x_{l_0}$, toh secara kasatmata sudah abadi nongkrong terparkir membumi bercokol diam bersarang (cached locally) di piringan memori dalam cip lokal (On-Chip VRAM).

---

## 5. Eksperimen (*Experiments*)

### 5.1 Pengaturan Eksperimental (*Experimental Setup*)

Validitas dan kejantanan gagasan temuan ilmiah metode di manuskrip ini dieksekusi pembuktian empirisnya di atas arena peperangan prapelatihan model bahasa (Language Model Pre-training). Dibentangkan panggung arena tarung evaluasi komparatif berskala megah (comparative analysis) antara arsitektur dasar "Vanilla Kosong" (Baseline), model *Hyper-Connections* murni edisi lawas (HC), serta kreasi pamungkas gubahan mutakhir inovasi paten makalah tulisan ini: sang *Manifold-Constrained Hyper-Connections* (**mHC**). 

Mencuri genetik purwarupa silang DNA MoE sang jawara DeepSeek-V3 (Liu et al., 2024b), digaraplah **empat armada spesies klon mutan arsitektur LLM mandiri** guna mengakomodir berbagai tingkatan kaliber sabuk tinju evaluasi pengujian. Merujuk pengaturan pakem sakral, laju pelebaran pipanisasi arus residual ($n$ / expansion rate) dipaku pada persneling gigi **4** bagi kedua kubu petarung (HC lama maupun mHC baru kami). 

Mahkota pertempuran dipertaruhkan membidik arena fokus primer berupa sosok golem LLM dengan bobot raksasa berkekuatan otak **27 Miliar (27B) parameter**, yang disiksa mengunyah samudra kawah diet porsi pakan data raksasa rasio sebanding (proportional dataset size). Mesin juara 27B inilah yang dicanangkan didaulat menanggung mandat membopong memanggul beban gelar medali juara hasil akhir sistem tingkat-tinggi (system-level main results).

Melontarkan sayap ekspedisi penelitian agar semakin dalam menyibak tabir misteri alam semesta, dibedah juga pola hukum pembesaran tabiat kurva kalkulus otot komputasi (compute scaling behavior) dengan jalan melepas bibit unggul varian model anak-anak balita berukuran mungil 3B (3 Miliar) dan 9B (9 Miliar) parameter, dibesarkan menggunakan diet gizi pakan yang sepadan (*proportional data*). Jurus pelengkap ini mendikte kita menelanjangi dan melototi pola alur grafik performa seiring ayunan gas buang komputasi yang diinjak variatif. 

Lebih sadis lagi, demi meriset patologi murni akan "bagaimana perilaku mesin mencerna huruf demi huruf ditelan waktu" (token scaling behavior), kami menelurkan satu unit kelinci percobaan khusus model bayi 3B yang dihukum dikurung dalam sel lab untuk melumat diet baku absolut **1 Triliun Token (1T Tokens) secara konstan**. Pemetaan arsitektural detail anatomi jeroan mesin dan dosis gizi obat hiperparameter pelatihan komplit tersaji di kamar rahasia di Lampiran Apendiks A.1.

### 5.2 Hasil Utama (*Main Results*)

*(Dosen: Mari kita melihat kembali performa kestabilan. Ini berhubungan langsung dengan Gambar 5 di awal yang telah kita analisis. Ini adalah intisari perbaikan arsitektur mHC).*

Kami mulai dengan menginspeksi stabilitas pelatihan dan konvergensi dari model raksasa 27B. Seperti yang diilustrasikan dengan amat gamblang pada **Gambar 5 (a)**, inovasi perisai pengekang kurungan matematik $mHC$ kita ini dengan mutlak dan sukses membasmi menghancurkan tuntas insiden petaka penyakit bawaan "epilepsi kurva error" (*training instability*) yang menjangkiti silsilah leluhur pendahulunya si *HC* usang. mHC mencetak tinta emas kemenangan berupa penghematan pembunuhan reduksi level eror mutlak akhir (final loss reduction) sebesar **0.021** (nilai yang masif untuk LLM!) jika dibandingkan melawan Baseline asali.

Penyembuhan mukjizat kesehatan kardiovaskular model (improved stability) ini mendapat validasi palu hakim ganda dari pemeriksaan anatomi tekanan darah "norma gradien" (*gradient norm*) yang didiagnosis di **Gambar 5 (b)**. Di sana terlihat nyata dan memukau, bahwa guncangan detak jantung kurva arus mHC (garis gelap) memamerkan ketenangan nirwana batin (*significantly better behavior*) yang amat sehat dibanding si HC gila, serta sukses menjaga kelurusan profil kalem yang seirama kembaran persis layaknya Baseline konvensional.

---

### Analisis Akademik Tabel 4: Hasil Kinerja Utama (Penalaran Turunan)

Inilah ujian rapor kelulusan dari AI kita ketika disuruh turun gunung menyelesaikan tugas "berpikir dan menalar" ala manusia di dunia nyata.

**Tabel 4: Hasil Benchmark Tingkat Sistem untuk Model 27B.**

| Benchmark (Metrik) | BBH (EM) | DROP (F1) | GSM8K (EM) | HellaSwag (Acc.) | MATH (EM) | MMLU (Acc.) | PIQA (Acc.) | TriviaQA (EM) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| # Shots | 3-shot | 3-shot | 8-shot | 10-shot | 4-shot | 5-shot | 0-shot | 5-shot |
| **27B Baseline** (Tanpa Pipa Lebar) | 43.8 | 47.0 | 46.7 | 73.7 | 22.0 | 59.0 | 78.5 | 54.3 |
| **27B w/ HC** (Pipa Lebar HC Liar)| 48.9 | 51.6 | 53.2 | 74.3 | 26.4 | 63.0 | 79.9 | 56.3 |
| **27B w/ mHC** (Pipa Lebar mHC Jinak/Inovasi Kita)| **51.0** | **53.9** | **53.8** | **74.7** | **26.0** | **63.4** | **80.5** | **57.6** |

**Kesimpulan Akademis Absolut:**
*   Menghantam dan merajai seluruh panggung. Di baris terbawah (mHC kita yang dirantai oleh Poligon Birkhoff / Matriks Stokastik Ganda), model 27B kita **mencetak rekor tertinggi di mayoritas absolut (7 dari 8) ujian neraka** melawan Baseline dan HC kuno.
*   Perhatikan lompatan di BBH (Penalaran Sulit): Dari 43.8 (Baseline) loncat ke 51.0! 
*   Bahkan ketika dibandingkan dengan HC liar yang membanggakan peningkatan jalur residualnya, mHC tetap "lebih pintar dan waras" (karena ia dipandu dengan stabil, tidak meledak di tengah pelatihan). mHC mendongkrak keunggulan tambahan +2.1% di BBH dan +2.3% di DROP dibanding HC.

### 5.3 Eksperimen Penskalaan (*Scaling Experiments*)

Hukum fisika murni untuk AI: "Jika kita menggelontorkan uang (komputasi/GPU) lebih banyak, akankah model ini terus bertambah pintar secara matematis, atau justru mentok (stagnan)?"

Untuk menjawabnya, mari kita analisis **Gambar 6**.

### Analisis Akademik: Gambar 6 - Sifat Penskalaan mHC

*   **Deskripsi Visual (a): Compute Scaling Curve (Kurva Penskalaan Komputasi)**
    *   Sumbu X: FLOPs (Seberapa banyak uang/mesin GPU yang dibakar untuk kalkulasi).
    *   Sumbu Y: Absolute Loss Gap (Seberapa besar kebodohan/error berkurang).
    *   Garis Biru (mHC): Menunjukkan bahwa garisnya terus-menerus turun. Ini berarti, saat Anda membakar komputasi untuk model 3B, 9B, hingga 27B, **mHC secara stabil meraup deviden intelijensi (terus bertambah pintar tanpa henti).** Ia tidak tercekik atau "kenyang". Trajektori ini membuktikan keampuhan kokohnya pondasi mHC untuk model triliunan parameter di masa depan.
*   **Deskripsi Visual (b): Token Scaling Curve (Kurva Penskalaan Token)**
    *   Sumbu X: FLOPs/Jumlah token bacaan yang masuk ke otak mesin.
    *   Sumbu Y: Relative Loss Ratio.
    *   Grafik ini merunut rekam medis mesin 3B saat menelan 1 Triliun token. Menandakan bahwa jurang keunggulan performa antara mHC dan Baseline konsisten stabil (sejajar menjauh) sepanjang waktu proses pelatihan dari awal bulan hingga bulan depan.

---

### 5.4 Analisis Stabilitas (*Stability Analysis*)

Secara ideal matematis yang ada di awang-awang surga (lihat Persamaan 6), kami memimpikan matriks pemetaan 1 lapis mematuhi 100% rukun kendala "Stokastik Ganda Mutlak". Jika ini terwujud, maka nilai perolehan sinyal lari maju ke depan (*forward signal gain*) dan nilai perolehan balikan perlawanan gradien arus balik (*backward gradient gain*) pastilah terkunci absolut bernilai sakral tepat pada angka "1" murni bulat! 

TETAPI... kita hidup di bumi dunia nyata industri koding komputer yang kejam. 
Implementasi baris kode pragmatis dalam menyeret paksa masuk algoritma siksaan iteratif **Sinkhorn-Knopp** (Persamaan 9) secara kejam menuntut adanya palu ketok "Pembatasan Penghentian Paksa" putaran siklus pengulangan (limit the number of iterations) demi menghindari mesin GPU lemot dan agar komputasi tidak terbakar mati. 

Di laboratorium lapangan kami (Our settings), kami mematok angkat hemat memotong di angka iterasi "20 kali pemutaran" (*20 iterations*) sekadar untuk mengais sisa ampas jawaban pendekatan (*approximate solution*). 

Dampak kosmik dari pemotongan iterasi prematur ini?
Intiplah grafik diagnosis medis kardiovaskular tegangan tinggi di **Gambar 7(a)**. Gain besaran gradien perlawanan arah balik (*backward gradient gain*) mHC terekam "agak menyeleweng bandel melenceng tipis" bergeser menjauhhi garis tengah nilai mutlak 1.

Dan apabila dikompositkan/ditumpuk disiksa menembus rentetan 60 tumpukan berlapis kawah komposit (*composite case* di **Gambar 7(b)**), tingkat penyelewengan dosa eror (*deviation*) ini lambat laun merayap membesar membengkak... NAMUN secara ajaib tetap tertahan terkunci dengan batas aman absolut (*remains bounded*), hanya menjilat atap maksimal mentok menembus kubah langit angka $\approx$ **1.6**.

Sekarang, resapi keajaiban ini dalam jiwa sains komputer Anda:
Coba sandingkan, jejerkan, dan bandingkan raihan mHC ini dengan auman badai monster buas amukan amplitudo tsunami dari si "HC lama yang liar"! HC kuno yang tanpa belenggu *manifold* itu tadi mencetak rekor kiamat ledakan maksimum melesat hingga **nyaris menyentuh angka 3.000 (Tiga Ribu!)** (*lihat analisis Gambar 3(b) di atas*). 

Maka, perisai mHC kita ini terbukti secara brutal dan mutlak mampu menyunat, memenggal, membius dan **mereduksi ledakan angka kiamat tersebut dengan skala rasio menciut tiga tingkat orde magnitudo (diturunkan ribuan kali lipat)!**

Rentetan angka eksperimen pembuktian maut ini memahkotakan kesahihan supremasi absolut bahwa *mHC*, sang *Manifold-Constrained Hyper-Connections*, mendongkrak stabilitas penjalaran gelombang rambatan sinyal (propagation stability) ke kasta dimensi dewa jika disejajarkan dengan rongsokan *HC* lama, menyegel mutlak asuransi keamanan atas garansi tenangnya hempasan aliran ombak *forward signal* dan arung jeram arus balik *backward gradient*.

Untuk melengkapi galeri museum pembuktian, pelototi **Gambar 8** yang menggelar pameran lukisan abstraksi matriks (*visualizations of learnable mappings*). 
*(Analisis Kualitatif Gambar 8):* Di panel HC kuno yang baris atas, saat salah satu peta sel warna meledak jadi biru/merah gelap pekat, kotak-kotak tetangganya ikut terjangkit menular "mewabah pekat besar". Ini membuktikan adanya sel-sel kanker *"general instability"* yang meracuni menyebar luas di semua saluran pipa peredaran darah arsitektur HC. 
Sangat berkebalikan layaknya nirwana, tataplah mahakarya matriks baris bawah milik **mHC**. Sapuan warnanya yang kalem, tenang, rata, dan konsisten mendemonstrasikan secara visual bahwa mHC selalu membuahkan pemecahan perhitungan yang kokoh bersahaja dan kalem (stable results).

---

## 6. Kesimpulan dan Pandangan Masa Depan (*Conclusion and Outlook*)

Di dalam makalah orasi ilmiah ini, kita telah membongkar dan mendiagnosa sebuah anomali raksasa: Bahwa gagasan meluaskan daya regang sungai "residual stream" serta membuat variasi jalan tembus-cabang sambungan memang memicu letupan kecerdasan performa hebat seperti yang disodorkan pertama kali oleh *Hyper-Connections (HC)*. NAMUN, kemerdekaan tanpa aturan/liar (*unconstrained nature*) dari jalinan kabel serabutan tersebut melahirkan petaka mematikan: **Divergensi Sinyal** (sinyal AI menjadi pecah tak menentu). 

Bencana struktural ini mengoyak dan meruntuhkan hukum suci konservasi pelestarian nilai energi sinyal saat ia menyusup ke bawah lapisan-lapisan jurang neural, menyebabkan AI menjadi pusing, epilepsi saat masa *training* (*training instability*), dan merantai paksa kelumpuhan agar model arsitektur "Deep Network" (Jaringan Dalam) tidak mungkin dicetak dalam ukuran raksasa (*hindering scalability*).

Sebagai jawara penyelamat demi menaklukkan naga tantangan ini, kami memproklamasikan lahirnya **Manifold-Constrained Hyper-Connections (mHC)**. Ini adalah kerangka pondasi arsitektural tatanan universal makro (*generalized framework*) yang menembakkan/menyandera (proyeksi) batas ruang dimensi gerak residual connection ke atas kurungan sebuah "manifold" yang rigid dan matematis terukur.

Berangkat dengan menunggangi algoritma kereta kencana maut **Sinkhorn-Knopp**, mHC menegakkan fatwa belenggu syarat pemaksaan mutlak "Stokastik Ganda" (*doubly stochastic constraint*) kepada setiap jengkal denyut matriks pemetaan residual (*residual mappings*). Sihir mHC ini secara fisikawan merombak merubah proses rambatan propagasi sinyal menjadi sekadar **kombinasi cembung (convex combination) pencampuran adonan fitur semata**. 

Pengadilan hasil rekam jejak pengujian eksperimental empiris membuktikan secara tak bisa dibantah bahwa mHC secara efektif dan gemilang memanggil bangkit kembali "Ruh Properti Pemetaan Identitas" (*Identity Mapping Property*) dari alam kubur! Ruh tersebut membungkus pelindung imunitas stabilitas pelatihan (*stable large-scale training*) berskala-mega dengan mahkota skalabilitas yang tak terbayangkan sebelumnya jika dibandingkan memori purba HC konvensional. 
Puncaknya (Crucially), dengan digabungkan bersama manuver bedah trik peretasan optimasi tulang punggung struktur perangkat keras (infrastruktur), mahakarya **mHC ini sukes mengantar paket hantaran peningkatan pesat kecerdasan AI tanpa diiringi perampokan pembengkakan tagihan biaya komputasi tambahan (negligible computational overhead).**

*(Outlook Ilmu Pengetahuan / Pandangan Masa Depan)*
Sebagai pewaris penerus generasi dari dinasti paradigma HC, sang mHC muda ini menyingkap lebar-lebar banyak sekali pintu gerbang menjanjikan untuk lahan ekskavasi ekspedisi penelitian ilmiah akademis di hari esok. 
Kendatipun riset jurnal kami kali ini berpuas diri menggunakan kurungan "matriks stokastik ganda" (doubly stochastic matrices) demi menukar asuransi kestabilan, rupa fisik algoritma kerangka cetak biru ini sebenarnya menawari fleksibilitas mahaluas untuk diutak-atik mengakomodir penjelajahan bermacam-macam bentuk selimut batas "manifold constraints" jenis lain yang secara kustomisasi bisa dibongkar-pasang dipahat khusus demi melayani objektif pembelajaran mesin/AI yang sangat spesifik dan esoterik (*tailored to specific learning objectives*). 

Kami menyemai harapan optimis tinggi bahwa galian investigasi di masa depan menuju lorong bentuk rintangan batas kurungan geometri "Geometric Constraints" yang berbeda-beda kemungkinan bakal sanggup meletuskan beranak-pinak bermacam metodologi revolusioner mutakhir lainnya yang jauh lebih sempurna membidik titik keseimbangan kompromi tarik-ulur (*trade-off*) perkelahian di antara "plastisitas" kelenturan daya serap kecerdasan AI melawan stabilitas amukan emosi pelatihannya.

Paling ujung, kami memiliki damba semoga dentuman gema jurnal mHC ini memukul palu bedug kesadaran yang berhasil memicu dan menghidupkan meregenerasi memompa kebangkitan gairah minat kecintaan para begawan akademis sains dan insinyur komunitas perihal dunia esoterik tata letak rancang bangun Makro-Arsitektur (macro-architecture design). 
Lewat upaya penggalian tebal merasuk menukik untuk sungguh memahami secara mendalam sejauh mana pola bentuk rangka jalinan Topologi (topological structures) secara de facto sanggup mengemudikan pergerakan stir nasib keoptimalan model (optimization) dan asimilasi penalaran representasi (representation learning), *mHC* pada hakekatnya akan sudi turun tangan menolong mengentaskan jalan buntu tembok keterbatasan zaman sekarang dan berpotensi menerangi menyorotkan cahaya suar petunjuk jalan napak tilas rute-rute baru revolusioner yang akan membawa peradaban ini menuju ke arah panggung perhelatan **evolusi mutasi arsitektur Model Bahasa Fondasional generasi raksasa di masa depan (*evolution of next-generation foundational architectures*).**

*(Tamat - Akhir dari Modul Diktat Perkulihan)*
