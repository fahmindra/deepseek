---
slug: ch-11-17
title: "Inference-Time Scaling"
layout: chapter
---
## Penskalaan Waktu-Inferensi untuk Pemodelan Imbalan Generalis (*Inference-Time Scaling for Generalist Reward Modeling*)

Pembelajaran Penguatan (*Reinforcement Learning* / RL) telah diadopsi secara luas dalam fase pascapelatihan (*post-training*) untuk Model Bahasa Besar (LLMs) dalam skala yang masif. Baru-baru ini, pemberian insentif terhadap kemampuan penalaran (*reasoning capabilities*) pada LLMs yang berasal dari RL mengindikasikan bahwa metode pembelajaran yang tepat dapat memungkinkan skalabilitas waktu-inferensi (*inference-time scalability*) yang efektif. 

Tantangan utama dari penerapan RL adalah bagaimana mendapatkan sinyal imbalan (*reward signals*) yang akurat untuk LLMs di berbagai domain, yang melampaui sekadar pertanyaan yang dapat diverifikasi (*verifiable questions*) atau aturan-aturan buatan (*artificial rules*). Dalam materi ini, kita akan menyelidiki bagaimana cara meningkatkan Pemodelan Imbalan (*Reward Modeling* / RM) dengan menggunakan lebih banyak komputasi pada saat inferensi untuk kueri-kueri yang bersifat umum, atau yang kita sebut sebagai **Skalabilitas Waktu-Inferensi dari RM Generalis**.

Untuk pendekatan RM ini, kita mengadopsi Pemodelan Imbalan Generatif secara *Pointwise* (*Pointwise Generative Reward Modeling* / GRM) untuk memungkinkan fleksibilitas terhadap berbagai jenis masukan (*input*) dan membuka potensi penskalaan waktu-inferensi. Untuk metode pembelajarannya, kita akan mempelajari **Penyetelan Kritik Berprinsip Mandiri (*Self-Principled Critique Tuning* / SPCT)** guna menumbuhkan perilaku pembuatan imbalan yang dapat diskalakan pada GRMs melalui RL daring (*online RL*). Metode ini menghasilkan prinsip-prinsip secara adaptif dan memberikan kritik secara akurat, yang berujung pada penciptaan model **DeepSeek-GRM**. 

Lebih jauh lagi, untuk memastikan penskalaan waktu-inferensi berjalan efektif, kita menggunakan pengambilan sampel paralel (*parallel sampling*) untuk memperluas penggunaan komputasi, dan memperkenalkan sebuah **Meta RM** untuk memandu proses pemungutan suara (*voting*) demi performa penskalaan yang lebih baik. Secara empiris, kita akan melihat bahwa SPCT secara signifikan meningkatkan kualitas dan skalabilitas dari GRM, mengungguli metode dan model yang sudah ada di berbagai tolok ukur (*benchmarks*) RM tanpa adanya bias yang parah. Model ini juga dapat mencapai performa yang lebih baik dibandingkan dengan penskalaan pada saat waktu-pelatihan (*training-time scaling*). Meskipun demikian, DeepSeek-GRM masih menemui tantangan pada beberapa tugas spesifik, yang kita yakini dapat diatasi melalui upaya-upaya lanjutan dalam sistem imbalan generalis di masa depan.

---

## 1. Pendahuluan (*Introduction*)

Mahasiswa sekalian, kemajuan luar biasa yang terjadi pada Model Bahasa Besar (LLMs) telah mengkatalisasi pergeseran yang sangat signifikan dalam riset Kecerdasan Buatan (AI). Hal ini memungkinkan model untuk melakukan tugas-tugas yang membutuhkan pemahaman yang mendalam, kemampuan generatif, serta kemampuan pengambilan keputusan yang bernuansa. 

Baru-baru ini, Pembelajaran Penguatan (*Reinforcement Learning* / RL) yang digunakan sebagai metode pascapelatihan (*post-training*) untuk LLMs telah diadopsi secara luas pada skala besar. Pendekatan ini menghasilkan peningkatan yang luar biasa dalam hal:
1. Keselarasan nilai manusia (*human value alignment*).
2. Penalaran jangka panjang (*long-term reasoning*).
3. Adaptasi terhadap lingkungan (*environment adaptation*).

Pemodelan Imbalan (*Reward Modeling* / RM), sebagai komponen yang sangat krusial dalam RL, sangat esensial untuk menghasilkan sinyal imbalan yang akurat terhadap respons yang diberikan oleh LLM. Studi-studi terkini menunjukkan bahwa, dengan adanya imbalan yang berkualitas tinggi dan tangguh (baik pada saat pelatihan maupun pada saat inferensi), LLMs dapat mencapai performa yang sangat kuat di domain-domain yang spesifik.

> **Analisis Gambar 1: Performa Penskalaan Waktu-Inferensi pada Berbagai RM**
>
> Sebagai pengantar visual, mari kita bedah **Gambar 1** secara mendalam. Grafik ini mengilustrasikan performa berbagai model pemberian imbalan ketika kita menambahkan jumlah sampel komputasi pada saat inferensi. 
> *   **Sumbu X (Horizontal):** Merepresentasikan nilai $k$, yaitu jumlah sampel imbalan yang diekstraksi (*#sampled rewards*). Sumbu ini menggunakan skala logaritmik (titik-titiknya berada di angka 1, 2, 4, 8, 16, hingga 32). Semakin ke kanan, semakin banyak komputasi inferensi yang digunakan (pengambilan sampel yang diulang-ulang).
> *   **Sumbu Y (Vertikal):** Merepresentasikan nilai persentase performa agregat di seluruh tolok ukur (mulai dari rentang 66.5 hingga 72.5). Semakin tinggi posisinya, semakin baik kualitas dan keakuratan imbalannya.
> *   **Kurva dan Titik Data:** 
>     *   Terdapat sebuah kurva dengan garis merah tebal berlabel **DeepSeek-GRM-27B (MetaRM@k)**. Kurva ini bermula di titik sekitar 67.9 pada $k=1$, lalu melesat naik secara konstan dan tajam seiring bertambahnya $k$, hingga mencapai nilai tertinggi di sekitar 72.8 pada $k=32$. Garis merah ini adalah kontribusi utama dari makalah ini (metode *Ours*).
>     *   Di bawahnya, terdapat garis biru solid berlabel **DeepSeek-GRM-27B (Voting@k)**. Grafiknya juga naik seiring bertambahnya $k$, namun dengan kemiringan yang lebih landai, mencapai titik akhir sekitar 71.0 pada $k=32$.
>     *   Di bawah kedua garis dominan tersebut, terdapat beberapa garis putus-putus (*dashed lines*) dari model pembanding (baseline). Misalnya:
>         *   Garis cokelat muda (LLM-as-a-Judge w/ TokenProb Voting@k) yang bergerak naik perlahan namun terhenti di bawah angka 69.0.
>         *   Garis toska (CLoud-Gemma-2-27B Voting@k) yang sangat datar (*flat*), nyaris tidak ada peningkatan dari $k=1$ hingga $k=8$.
>     *   Titik-titik individual tanpa garis (seperti model Skalar Nemotron-4-340B, GPT-4o Greedy, dll) berada di sekitar nilai 70.5 hingga 71.5, merepresentasikan model statis tanpa metode penskalaan berlapis ini.
> *   **Kesimpulan Akademis:** Grafik ini membuktikan sebuah argumen sentral: Pemodelan Imbalan tradisional (seperti Skalar) tidak mampu memanfaatkan komputasi tambahan saat inferensi (kurvanya datar). Sebaliknya, dengan menggunakan **DeepSeek-GRM-27B** dipadukan dengan **Meta RM**, model mampu secara eksponensial meningkatkan keakuratan imbalannya seiring dengan bertambahnya jumlah sampel ($k$). Skalabilitas ini sangat efektif!

Namun, imbalan berkualitas tinggi pada domain spesifik tersebut saat ini umumnya didapatkan melalui:
*   Lingkungan buatan manusia dengan kondisi yang sudah jelas (seperti permainan).
*   Aturan kerajinan tangan (*hand-crafted rules*) untuk pertanyaan yang dapat diverifikasi, seperti masalah matematika atau tugas pemrograman (*coding*).

Dalam **domain umum (generalis)**, pembuatan imbalan jauh lebih menantang. Kriteria untuk memberikan imbalan menjadi sangat beragam, kompleks, dan seringkali tidak memiliki referensi eksplisit atau *ground truth* (kebenaran dasar). Oleh karena itu, Pemodelan Imbalan Generalis menjadi sangat krusial untuk meningkatkan performa LLMs pada aplikasi yang lebih luas, baik dari perspektif pascapelatihan (seperti RL dalam skala besar), maupun dari perspektif inferensi (seperti pencarian yang dipandu oleh RM). Lebih lanjut, performa RM seharusnya dapat ditingkatkan dengan menambahkan *komputasi pelatihan* dan *komputasi inferensi*.

Dalam praktiknya, timbul berbagai tantangan dalam merancang RM yang *generalis* sekaligus *dapat diskalakan secara efektif pada saat inferensi*. 
*   Syarat untuk menjadi generalis menuntut: (1) **fleksibilitas** terhadap berbagai jenis masukan (input) yang berbeda, dan (2) pembuatan **imbalan yang akurat** di berbagai domain. Paradigma ini kita sebut sebagai **Pemodelan Imbalan Generalis**. 
*   Syarat untuk mencapai skalabilitas waktu-inferensi yang efektif menuntut RM untuk: (3) menghasilkan **sinyal imbalan berkualitas lebih tinggi** seiring bertambahnya komputasi inferensi, dan (4) mempelajari perilaku yang dapat diskalakan demi tercapainya grafik peningkatan performa sebanding dengan komputasi (seperti yang terlihat pada Gambar 1).

Riset-riset sebelumnya menunjukkan beberapa paradigma untuk menghasilkan imbalan:
1.  **Skalar (*Scalar*)**
2.  **Semi-Skalar (*Semi-scalar*)**
3.  **Generatif (*Generative*)**

Selain itu, terdapat berbagai pola penilaian (*scoring patterns*), seperti:
1.  **Secara individu (*Pointwise*)**
2.  **Berpasangan (*Pairwise*)**

Pendekatan-pendekatan tersebut secara inheren menentukan tingkat fleksibilitas input (Tantangan 1) dan skalabilitas waktu-inferensi (Tantangan 3) dari RM. Sebagai contoh, RM *Pairwise* hanya mempertimbangkan preferensi relatif dari dua respons yang dipasangkan, sehingga kehilangan fleksibilitas untuk menerima input berupa satu respons tunggal atau banyak respons sekaligus. Di sisi lain, RM Skalar sangat sulit untuk menghasilkan sinyal imbalan yang beragam untuk respons yang sama, sehingga menghambat perolehan imbalan yang lebih baik melalui metode penskalaan waktu-inferensi berbasis pengambilan sampel (*sampling-based*).

Berbagai metode pembelajaran telah diusulkan untuk meningkatkan kualitas imbalan, namun sedikit sekali yang berfokus pada skalabilitas waktu-inferensi dan mempelajari interkoneksi antara perilaku generatif yang dipelajari dengan efektivitas penskalaan tersebut. Minimnya eksplorasi ini (Tantangan 2 & 4) memunculkan pertanyaan penelitian: *"Dapatkah kita mendesain sebuah metode pembelajaran yang bertujuan untuk mengaktifkan penskalaan waktu-inferensi yang efektif untuk pemodelan imbalan generalis?"*

Dalam riset ini, kita menginvestigasi berbagai pendekatan RM dan menemukan bahwa **Pemodelan Imbalan Generatif secara Pointwise (Pointwise GRM)** mampu menyatukan proses pemberian skor untuk satu, dua, atau banyak respons dalam sebuah representasi bahasa murni. Hal ini menjawab Tantangan (1). Kita juga mengeksplorasi bahwa *prinsip-prinsip (principles)* tertentu dapat memandu proses pembuatan imbalan dengan kriteria yang tepat bagi GRM, sehingga meningkatkan kualitas imbalan. Temuan ini mengisyaratkan bahwa skalabilitas waktu-inferensi dari RM dapat dicapai dengan menskalakan (memperbanyak) pembuatan *prinsip-prinsip berkualitas tinggi* dan *kritik yang akurat*.

Berdasarkan dasar tersebut, makalah ini mengusulkan sebuah metode pembelajaran baru yang disebut **Penyetelan Kritik Berprinsip Mandiri (*Self-Principled Critique Tuning* / SPCT)**. Melalui RL daring berbasis aturan (*rule-based online RL*), SPCT memungkinkan GRMs untuk belajar secara mandiri merumuskan *prinsip* dan *kritik* secara adaptif berdasarkan kueri input dan respons yang diberikan, yang akan bermuara pada hasil imbalan yang lebih baik di domain umum (menjawab Tantangan 2). Model akhir dari metode ini adalah **DeepSeek-GRM-27B** (yang dipelajari dari model dasar Gemma-2-27B).

Untuk penskalaan waktu-inferensi, komputasi diperluas dengan mengambil sampel berulang-ulang secara paralel. Melalui ini, DeepSeek-GRM menghasilkan set prinsip dan kritik yang berbeda-beda, kemudian melakukan pemungutan suara (*voting*) untuk menentukan imbalan akhir. Dengan pengambilan sampel skala besar, DeepSeek-GRM mampu melakukan penilaian yang jauh lebih akurat berdasarkan prinsip-prinsip yang beragam, menyelesaikan Tantangan (3) & (4). Sebagai tambahan, sebuah **Meta RM** juga dilatih untuk memandu proses pemungutan suara ini agar lebih optimal.

**Kontribusi Utama Makalah Ini:**
1.  Mengusulkan metode novel, **Self-Principled Critique Tuning (SPCT)**, untuk menumbuhkan penskalaan waktu-inferensi yang efektif bagi RM generalis, yang menghasilkan model **DeepSeek-GRM**. Lebih jauh, **Meta RM** diperkenalkan untuk memandu dan melampaui metode *voting* biasa.
2.  Menunjukkan secara empiris bahwa SPCT meningkatkan kualitas dan skalabilitas waktu-inferensi GRM secara drastis dibandingkan metode dan model publik yang kuat.
3.  Menerapkan jadwal pelatihan SPCT pada LLM berukuran lebih besar dan menemukan fakta akademis bahwa penskalaan di waktu-inferensi dapat mengungguli penskalaan ukuran model (parameter) di waktu-pelatihan.

---

## 2. Pendahuluan Teknis (*Preliminaries*)

### 2.1 Perbandingan Berbagai Pendekatan RM

> **Analisis Gambar 2: Paradigma Berbagai Pembuatan Imbalan dan Pola Penilaian**
>
> Mahasiswa, silakan perhatikan **Gambar 2**. Diagram arsitektural ini memetakan lanskap Pemodelan Imbalan secara sangat rapi ke dalam matriks komponen.
> 
> *   **Bagian Atas (Paradigma Pembuatan Imbalan):**
>     *   **(a) Skalar (*Scalar*):** Model menerima Kueri & Respons, dimasukkan ke dalam blok RM, dan langsung mengeluarkan nilai *Skalar* (angka).
>     *   **(b) Semi-Skalar (*Semi-Scalar*):** Blok RM mengeluarkan teks *Kritik* (*Critique*) terlebih dahulu, dan diikuti dengan nilai *Skalar*.
>     *   **(c) Generatif (*Generative*):** Blok RM murni hanya mengeluarkan teks *Kritik*. Nilai akhir (jika dibutuhkan) diekstrak dari teks tersebut.
> *   **Bagian Atas Kanan (Pola Penilaian):**
>     *   **(i) Pointwise:** Model mengevaluasi satu respons pada satu waktu, mengeluarkan *Score 1* dan *Score 2* secara independen untuk Respons 1 dan Respons 2.
>     *   **(ii) Pairwise:** Model langsung membandingkan Respons 1 dan Respons 2 secara bersamaan, dan mengeluarkan *Score Relative* (skor yang membandingkan mana yang lebih baik).
> *   **Bagian Bawah (Implementasi Metode Perwakilan):** Bagian ini menampilkan matriks kombinasi dari atas.
>     *   **Bradley-Terry [(a) + (i)]:** Model skalar dengan penilaian pointwise. Hasilnya berupa angka desimal dan selisih loss.
>     *   **PairRM [(a) + (ii)]:** Model skalar pairwise. Jika R1 > R2 maka skor $\ge 0$, sebaliknya $< 0$.
>     *   **CLoud [(b) + (i)]:** Semi-skalar pointwise. Menghasilkan *Critique 1* lalu angka, dan *Critique 2* lalu angka, diakhiri dengan Loss.
>     *   **LLM-as-a-Judge [(b)/(c) + (ii)]:** Menghasilkan kalimat perbandingan panjang "...The better resp. is [[1]]".
>     *   **Pointwise GRM [(c) + (i)]:** Model mengekstrak skor teks dari kueri independen. Teks berbunyi: "...The scores are [[8, 1]] out of 10." (Metode ini yang difokuskan pada makalah ini).
> *   **Matriks Evaluasi (Bagian Terbawah):**
>     *   **Inference Time Scalable (Bisa Diskalakan saat Inferensi):** Bradley-Terry (Silang/Tidak), PairRM (Silang/Tidak), CLoud (Centang Hijau/Ya), LLM-as-a-Judge (Centang Hijau/Ya), Pointwise GRM (Centang Hijau/Ya).
>     *   **Input Flexible (Fleksibel Terhadap Masukan):** Bradley-Terry (Centang/Ya), PairRM (Silang/Tidak), CLoud (Centang/Ya), LLM-as-a-Judge (Silang/Tidak karena harus berpasangan), **Pointwise GRM (Centang Hijau/Ya)**.
> *   **Kesimpulan Akademis:** Gambar ini merangkum bahwa HANYA **Pointwise GRM** yang memenuhi dua syarat utama kita: dapat diskalakan di waktu-inferensi dan memiliki fleksibilitas input yang tak terbatas (bisa menilai 1, 2, atau ratusan respons tanpa merombak arsitektur model).

Sebagaimana diilustrasikan dalam Gambar 2, pendekatan RM ditentukan oleh paradigma pembuatan imbalannya dan pola penilaiannya. Hal ini pada dasarnya memengaruhi skalabilitas waktu-inferensi dan fleksibilitas input.

Mari kita formulasikan paradigma ini secara matematis. 
Untuk **Paradigma Pembuatan Imbalan**, RM klasik mengadopsi pendekatan **(a) Skalar**, yang memberikan nilai numerik pada kueri dan respons. Pendekatan skalar ini dapat diperluas menjadi **(b) Semi-Skalar**, yang menghasilkan teks kritikan di samping nilai skalar. Kemudian, pendekatan **(c) Generatif** murni hanya menghasilkan imbalan dalam bentuk teks. Hubungan ketiganya dimodelkan sebagai:

$$
\mathcal{R} = 
\begin{cases} 
S & \text{(Scalar)} \\ 
(S, C) & \text{(Semi-Scalar)} \\ 
C & \text{(Generative)} 
\end{cases} 
\sim r_\theta(x, \{y_i\}_{i=1}^n)
$$

**Penjelasan Rumus:**
*   $\mathcal{R}$: Variabel yang melambangkan keseluruhan imbalan (*Reward*).
*   $x$: Merupakan Kueri masukan (*input query*).
*   $\{y_i\}_{i=1}^n$: Merupakan himpunan (*set*) dari $n$ buah respons yang diberikan untuk dievaluasi. $y_i$ adalah respons ke-$i$.
*   $r_\theta$: Adalah fungsi imbalan (biasanya sebuah *neural network* / LLM) yang diparameterisasi oleh bobot-bobot jaringan $\theta$. Simbol $\sim$ berarti sampel ditarik dari distribusi yang dimodelkan oleh $r_\theta$.
*   $S \in \mathbb{R}^m$, di mana $m \le n$: Merupakan nilai imbalan yang berbentuk skalar (angka riil).
*   $C$: Merupakan teks kritikan linguistik (*critique*).

Untuk **Pola Penilaian (*Scoring Patterns*)**, kita membedakan dua pendekatan utama. 
Yang pertama adalah **(i) Pointwise**, pendekatan ini memberikan skor secara individual ke masing-masing respons tanpa mencampurnya:

$$
\{S_i\}_{i=1}^n = f_{point}(\mathcal{R}, \{y_i\}_{i=1}^n), \quad \mathcal{R} \sim r_\theta(x, \{y_i\}_{i=1}^n), \quad S_i \in \mathbb{R}
$$

**Penjelasan Rumus:**
*   $f_{point}(\cdot, \cdot)$: Adalah fungsi pemisahan (*splitting function*). Fungsi ini memproses seluruh keluaran imbalan $\mathcal{R}$ untuk mengalokasikan nilai skalar individu $S_i$ khusus untuk respons ke-$i$. 
*   $S_i \in \mathbb{R}$: Menegaskan bahwa setiap respons mendapatkan skor berupa bilangan riil independen.

Sebagai kontras, pendekatan **(ii) Pairwise** dapat dipandang sebagai metode pemilihan "terbaik dari $n$" (*best-of-n*), yang menyeleksi sebuah respons paling unggul dari seluruh kandidat:

$$
\hat{y} = f_{pair}(\mathcal{R}, \{y_i\}_{i=1}^n), \quad \mathcal{R} \sim r_\theta(x, \{y_i\}_{i=1}^n), \quad \hat{y} \in \{y_i\}_{i=1}^n
$$

**Penjelasan Rumus:**
*   $\hat{y}$: Merupakan prediksi respons terbaik tunggal yang dipilih.
*   $f_{pair}(\cdot, \cdot)$: Adalah fungsi penyeleksi (*selection function*). Dalam mayoritas implementasi klasik, nilai yang diuji hanyalah $n = 2$.
*   *Catatan Akademis:* Walaupun metode pairwise bisa diakali untuk menangani $n > 2$ (menggunakan sistem turnamen atau algoritma kombinatorik lainnya), metode ini mustahil digunakan jika model diminta untuk menilai satu respons tunggal saja ($n=1$). Inilah letak hilangnya "fleksibilitas input".

Beberapa contoh implementasi fungsi-fungsi di atas pada metode yang sudah ada:
*   **Model Bradley-Terry:** Dilatih dengan data preferensi pairwise namun dipaksa untuk mengeluarkan skor skalar secara pointwise: $\{S_i\}_{i=1}^n = f_{point}(\mathcal{R}, \{y_i\}_{i=1}^n) = S \in \mathbb{R}^n$.
*   **PairRM:** Membandingkan dua respons ($n=2$) menggunakan fungsi tanda (*sign*) dari imbalan skalar: $\hat{y} = f_{pair}(\mathcal{R}, \{y_i\}_{i=1}^n) = y_{\lfloor \frac{1}{2}(3 - \text{sgn}(S)) \rfloor}$. (Di mana nilai $\text{sgn}(S)$ menentukan apakah indeks yang dipilih adalah 1 atau 2).
*   **LLM-as-a-Judge:** Menilai preferensi urutan antara dua pasangan respons secara tekstual: $\hat{y} = f_{pair}(\mathcal{R}, \{y_i\}_{i=1}^n) = y_{f_{extract}(C)}$. Fungsi $f_{extract}(\cdot)$ bertugas mengekstrak *indeks* respons yang dinilai terbaik dari representasi bahasa (teks) yang dihasilkan LLM.

Namun, tanpa batasan tambahan, Pemodelan Imbalan Generatif (GRM) memiliki kemampuan superior untuk menghasilkan imbalan pointwise bagi banyak respons *hanya menggunakan representasi bahasa murni*:

$$
\{S_i\}_{i=1}^n = f_{point}(\mathcal{R}, \{y_i\}_{i=1}^n) = f_{extract}(C)
$$

**Penjelasan Rumus:**
*   $f_{extract}(C)$ dalam konteks GRM bertugas mengekstrak nilai-nilai imbalan (skor) secara spesifik yang telah tertanam di dalam generasi teks/kritik $C$ untuk *setiap* respons yang diberikan.
*   Biasanya, imbalan ini bersifat diskrit. Dalam riset ini, peneliti mengalokasikan nilai bilangan bulat $S_i \in \mathbb{N}$, di mana $1 \le S_i \le 10$ sebagai konfigurasi *default*. Pendekatan inilah yang memfasilitasi skalabilitas inferensi maupun fleksibilitas input secara utuh.

### 2.2 Meningkatkan Kualitas Imbalan Menggunakan Prinsip-Prinsip (*Principles*)

RM Generalis menuntut generasi imbalan berkualitas tinggi di luar domain spesifik. Di sinilah kriteria untuk memberikan imbalan menjadi kompleks. Oleh karena itu, kita mengadopsi **prinsip-prinsip (principles)** untuk memandu pembuatan imbalan, menggantikan aturan-aturan statis buatan manusia. Dengan digunakannya prinsip-prinsip ini, persamaan untuk pembuatan imbalan oleh GRM berubah menjadi:

$$
\mathcal{R} = C \sim r_\theta(x, \{y_i\}_{i=1}^n, \{p_i\}_{i=1}^m)
$$

**Penjelasan Rumus:**
*   $\{p_i\}_{i=1}^m$: Merupakan kumpulan dari $m$ buah prinsip (*principles*) yang disediakan ke dalam model $r_\theta$ untuk memandu proses pembuatan teks kritik $C$.
*   Persamaan ini menegaskan bahwa model imbalan tidak hanya dikondisikan pada kueri $x$ dan kandidat respons $\{y_i\}$, tetapi juga dikondisikan pada pedoman kriteria normatif/evaluasi $\{p_i\}$.

Para peneliti melakukan eksperimen pendahuluan untuk menguji pengaruh prinsip yang tepat terhadap kualitas imbalan, menggunakan *subset* "Chat Hard" dari *Reward Bench* dan *subset* "IFEval" dari *PPE benchmark*.

| Metode | Chat Hard | IFEval |
| :--- | :--- | :--- |
| **GPT-4o-2024-08-06** | 76.1 | 56.0 |
| w/ Self-Gen. Principles (*Prinsip Buatan Sendiri*) | 75.9 | 55.6 |
| w/ Filtered Principles (*Prinsip Disaring*) | 77.8 | 57.5 |
| **Gemma-2-27B-it** | 59.1 | 56.1 |
| w/ Self-Gen. Principles (*Prinsip Buatan Sendiri*) | 64.0 | 55.8 |
| w/ Filtered Principles (*Prinsip Disaring*) | 68.0 | 57.3 |

**Analisis Tabel 1: Eksperimen Pendahuluan tentang Pengaruh Prinsip terhadap Kualitas Imbalan**
*   Tabel ini membandingkan skor performa (semakin tinggi semakin baik) antara GPT-4o dan Gemma-2-27B pada keadaan asali (*default*), lalu ditambahkan prinsip yang di-_generate_ sendiri (Self-Gen), dan prinsip yang difilter khusus oleh manusia (Filtered).
*   *Temuan Akademis:* Ketika model menggunakan prinsip buatan mereka sendiri secara *naif* (Self-Gen), performa cenderung staganan atau bahkan sedikit menurun (seperti pada GPT-4o yang turun dari 76.1 ke 75.9). Namun, ketika prinsip-prinsip tersebut *difilter* sedemikian rupa dengan mempertahankan proses di mana imbalan terbesar memang dialokasikan pada respons yang berlabel terbaik (Filtered), kualitasnya melonjak drastis (GPT-4o naik ke 77.8, Gemma-2 melesat ke 68.0).
*   *Kesimpulan:* (a) LLM saat ini bisa membuat banyak prinsip, tapi tidak semuanya tepat sasaran. (b) Subset prinsip yang tepat dapat sangat memandu pembuatan imbalan, membuktikan adanya potensi "kemandirian awal" (*self-bootstrapping*). Temuan dasar ini menjadi fondasi penggunaan RL Daring (*Online RL*) untuk mengoptimalkan GRM pada bagian berikutnya.

---

## 3. Penyetelan Kritik Berprinsip Mandiri (*Self-Principled Critique Tuning / SPCT*)

Terinspirasi dari hasil pendahuluan tersebut, peneliti mendesain pendekatan baru untuk Pointwise GRM agar dapat belajar menghasilkan prinsip-prinsip secara adaptif sekaligus melakukan kritikan yang efektif, yang dinamakan **Self-Principled Critique Tuning (SPCT)**.

> **Analisis Gambar 3: Ilustrasi dari SPCT (Rejective Fine-Tuning, Rule-based RL, dan Inference)**
> 
> Gambar ini merupakan visualisasi penuh dari "Nyawa" penelitian ini. Mari kita urai alirannya (dari kiri ke kanan):
> *   **Tahap 1: RFT (Rejective Fine-Tuning) sebagai Cold Start (Pelatihan Luring):** Di kiri atas, Kueri (Q) dan Respons (R) diumpankan ke model GRM. GRM akan melakukan *Sampling* untuk menghasilkan Prinsip (Misal: *Principle 1: Instruction Adherence, Principle 2: Level of Detail...*) kemudian *Extract* menjadi teks imbalan (*Response 1 is better... Final Scores: [[2, 4]]*). Karena ini adalah penolakan sampel (*rejective*), jika hasil *Final Scores* salah menebak respons terbaik dibandingkan *Ground Truth* (Kebenaran Dasar), maka sampel data itu ditolak (*Too Easy/Incorrect*). Data yang berhasil menebak masuk ke dataset RFT.
> *   **Tahap 2: RL (Rule-Based RL - Online Update):** Di bagian tengah, alirannya bergulir (*Rolling Out*). GRM sekarang secara mandiri membangkitkan Prinsip adaptif (contoh: *Principle 1: Logic Chain Correctness (35%), Principle 2: Completeness (20%)*). Dari prinsip ini, GRM membuat *Critiques* teks naratif. Hasil akhirnya kembali diekstrak menjadi skor (misal: [[2, 7]]). Skor ini langsung dibandingkan menggunakan rumus argmax(r) terhadap Aturan (Rules). Proses hitungan benar/salah ini menghasilkan sinyal Imbalan (*Reward*) positif atau negatif yang digunakan untuk memperbarui (*Update*) bobot jaringan GRM secara seketika (*online*).
> *   **Tahap 3: Inference (Penskalaan Saat Inferensi):** Pada bagian bawah, GRM siap beraksi secara riil. Diberikan sebuah Kueri. Alih-alih melakukan generasi satu kali, GRM melakukan *Parallel Sampling* (generasi secara bercabang). 
>     *   Cabang 1 menghasilkan seperangkat prinsip, kritik, dan skor teks: `1/10, 5/10`.
>     *   Cabang 2: `5/10, 6/10`.
>     *   Cabang 3: `4/10, 8/10`.
>     *   Cabang 4: `7/10, 6/10`.
>     *   Seluruh skor ini dilempar ke sebuah modul bernama **Meta RM**. Meta RM bertugas menilai apakah *Critique* tiap cabang itu berbobot atau halusinasi. Ternyata Cabang 3 disilang merah (dianggap salah). Sisanya (Cabang 1, 2, dan 4) ditambahkan/divoting bersama. Output finalnya adalah gabungan skor dari cabang-cabang berkualitas: $17/40$ dan $25/40$. Ini menghasilkan nilai dan resolusi skoring yang jauh lebih *fine-grained* (terperinci).

### 3.1 Melepaskan Prinsip dari Sekadar "Pemahaman" Menjadi "Pembuatan" (*Unpinning Principles*)

Agar model sanggup menanggung beban skala besar, kita mengubah letak prinsip: dari yang awalnya hanya instruksi pemrosesan awal (*preprocessing*) statis, kini dijadikan bagian yang di-*generate* sendiri oleh model. Hal ini dirumuskan sebagai:

$$
\{p_i\}_{i=1}^m \sim p_\theta(x, \{y_i\}_{i=1}^n), \quad \mathcal{R} = C \sim r_\theta(x, \{y_i\}_{i=1}^n, \{p_i\}_{i=1}^m)
$$

**Penjelasan Rumus:**
*   Fungsi pertama: Model pembuat prinsip $p_\theta$ memproduksi seperangkat prinsip $\{p_i\}$ berdasarkan kueri $x$ dan daftar respons kandidat $\{y_i\}$. 
*   Fungsi kedua: Pembuatan kritik dan imbalan $\mathcal{R}$ melalui model $r_\theta$ yang sekarang disuntikkan parameter prinsip yang *baru saja dibuat di fungsi pertama*.
*   Secara praktikal dalam LLM, $p_\theta$ dan $r_\theta$ tidak perlu dua model terpisah. Mereka berdua diimplementasikan melalui *language head* yang sama dalam satu LLM yang sama. Model ini mula-mula menulis *"Prinsip untuk menilai ini adalah A, B, C."*, lalu melanjutkan menulis *"Berdasarkan prinsip tersebut, respons pertama mendapatkan..."*. Ini menghasilkan keselarasan generatif yang adaptif.

### 3.2 Pembelajaran Penguatan Berbasis Aturan (*Rule-Based Reinforcement Learning*)

SPCT memadukan dua fase: Penyetelan Halus berbasis Penolakan (*Rejective Fine-Tuning*) sebagai awalan (*cold start*), dan RL daring berbasis aturan.

**Rejective Fine-Tuning (Cold Start)**
Tujuan utama tahap ini adalah melatih GRM untuk meletakkan fondasi bagaimana menulis prinsip dan kritik yang diformat dengan benar. Data diekstraksi dari model GRM pra-pelatihan (*pretrained*). Untuk setiap titik data (yang berisi kueri $x$ dan $n$ buah respons $y_i$), pembuatan prinsip dan kritik dilakukan sebanyak $N_{RFT}$ kali (*sampling times*).

Strategi penolakannya sangat jelas dan absolut: "Buang hasil trajektori jika skor akhirnya tidak sesuai dengan *ground-truth* (respons terbaik menurut anotasi), ATAU buang jika kueri tersebut ternyata tertebak benar pada seluruh percabangan $N_{RFT}$" (karena dianggap terlalu mudah untuk dipelajari model).
Secara matematis, misalkan $r_i$ melambangkan imbalan kebenaran-dasar (*ground truth*) untuk respons ke-$i$. Skor prediksi Pointwise GRM $\{S_i\}_{i=1}^n$ diklasifikasikan benar JIKA:

$$
\begin{cases} 
\forall i \neq j, \quad S_j > S_i, \quad j = \arg\max_l \{r_l\}_{l=1}^n, & \text{if } n \ge 2, \\ 
S_1 = r_1, & \text{if } n = 1.
\end{cases} 
$$

**Penjelasan Rumus:**
*   Untuk kasus komparasi lebih dari satu respons ($n \ge 2$): Evaluasi dianggap Benar jika skor $S_j$ yang diberikan oleh GRM untuk indeks $j$ (di mana $j$ adalah indeks dari respons dengan imbalan *ground truth* maksimal) nilainya **sepenuhnya lebih besar** ($>$) daripada skor $S_i$ yang diberikan untuk seluruh respons kandidat lainnya ($i \neq j$).
*   Untuk kasus penilaian respons tunggal ($n = 1$): Dievaluasi benar jika skor absolut $S_1$ yang diprediksi GRM identik ($=$) dengan skor mutlak *ground truth* $r_1$.

*Catatan terkait Hinted Sampling:* Karena model dasar sangat kesulitan membuat kritik tanpa tahu ujungnya harus ke mana, peneliti menambahkan *hint* ke dalam *prompt*: *"Respons terbaik adalah: Respons X"*. Meskipun membantu, cara ini memunculkan kecenderungan model untuk memotong jalan penjelasan (*taking shortcuts*), sehingga masuklah kebutuhan ke fase RL Daring untuk membenahinya.

**Rule-Based RL**
Pada tahap ini, GRM disempurnakan (di-*fine-tune*) melalui algoritma GRPO (*Group Relative Policy Optimization*). Saat model melakukan *rollout* (mengeksplorasi pembuatan prinsip dan kritik baru), skor prediksi di bagian akhir teksnya diekstrak dan dibandingkan dengan kebenaran-dasar. Fungsi pemetaan imbalannya dirumuskan sebagai:

$$
\hat{r}_i = 
\begin{cases} 
1, & \text{if } n \ge 2 \text{ and } \forall i' \neq j', \quad S_{j'} > S_{i'}, \quad j' = \arg\max_l \{r_l\}_{l=1}^n, \\ 
1, & \text{if } n = 1 \text{ and } S_1 = r_1, \\ 
-1, & \text{otherwise}, 
\end{cases} 
$$

**Penjelasan Rumus:**
*   $\hat{r}_i$ adalah Imbalan RL (bukan nilai GRM) yang akan dipakai untuk merotasi kalkulus gradien jaringan syaraf (*neural network*).
*   Jika logika prediksi Pointwise GRM (seperti yang telah dijelaskan pada Persamaan sebelumnya) mendeteksi indeks respons terbaik secara presisi (baik untuk multirespos $n \ge 2$ maupun tunggal $n=1$), maka model RL diberikan hadiah mutlak $\mathbf{1}$.
*   Namun jika gagal (*otherwise*), algoritma RL langsung memukulnya dengan penalti atau hukuman sebesar $\mathbf{-1}$.
Fungsi pendorong ekstrem ini memaksa model untuk tidak sekadar berhalusinasi, namun benar-benar memisahkan respons terbaik menggunakan prinsip-prinsip otentik.

---

## 4. Penskalaan Waktu-Inferensi dengan SPCT

Untuk mendorong performa lebih tinggi lagi menggunakan komputasi yang tak terbatas saat inferensi, model melakukan pengambilan sampel berganda (*sampling-based strategies*).

**Voting with Generated Rewards**
Metode klasik pemungutan suara (*voting*) diterapkan secara berbeda tergantung arsitektur dasarnya. Untuk **RM Semi-Skalar**, suara diagregasi menggunakan metode rata-rata (*averaging*):

$$
S^* = \frac{1}{k} \sum_{i=1}^k S_i, \quad \{\mathcal{R}_i = (S_i, C_i)\}_{i=1}^k \sim r_\theta(x, \{y_i\}_{i=1}^n)
$$

**Penjelasan Rumus:**
*   Imbalan akhir $S^*$ adalah rata-rata aritmatika dari seluruh output skalar individu $S_i$ yang disampel sebanyak $k$ kali. Metode ini memiliki masalah karena varians nilai skalarnya sangat terbatas (sering mengeluarkan angka yang sama persis walau dikritik berbeda).

Untuk **RM Pairwise**, *voting* dijalankan melalui fungsi seleksi mayoritas (*majority*):

$$
\hat{y}^* = \arg\max_y \sum_{i=1}^k \mathbb{I}(y = \hat{y}_i), \quad \{\mathcal{R}_i = C_i\}_{i=1}^k \sim r_\theta(x, \{y_i\}_{i=1}^n)
$$

**Penjelasan Rumus:**
*   $\hat{y}^*$: Prediksi respons pemenang secara agregat.
*   $\hat{y}_i$: Pemilihan respons tunggal tebaik pada sampel ke-$i$ (melalui fungsi $f_{pair}$).
*   $\mathbb{I}(\cdot)$: Adalah Fungsi Indikator (*Indicator function*). Fungsi ini bernilai $1$ jika pernyataan di dalamnya benar ($y$ adalah pilihan $\hat{y}_i$), dan $0$ jika salah. Singkatnya, ini menghitung frekuensi alias siapa yang mendapat perolehan suara (*voting*) terbanyak di antara $k$ sampel. Namun, kekurangannya adalah ia membuang informasi gradasi matematis jika responsnya memiliki perbedaan kuantitatif yang sangat tipis.

Berbeda dengan keduanya, proses voting untuk **Pointwise GRM** (DeepSeek-GRM) didefinisikan secara brilian dengan penjumlahan total imbalan di seluruh cabang:

$$
S_i^* = \sum_{j=1}^k S_{i,j}, \quad \{p_{i,j}\}_{i=1}^{m_j} \sim p_\theta(x, \{y_i\}_{i=1}^n), \mathcal{R}_j = C_j \sim r_\theta(x, \{y_i\}_{i=1}^n, \{p_{i,j}\}_{i=1}^{m_j}), \quad j = 1, ..., k
$$

**Penjelasan Rumus:**
*   $S_i^*$ adalah skor akumulatif final untuk satu respons kandidat spesifik ke-$i$ (di mana $i = 1, \dots, n$).
*   Skor ini diperoleh dari menjumlahkan ($\sum_{j=1}^k$) seluruh keluaran skor individu $S_{i,j}$ dari pengambilan sampel *rollout* percabangan ke-$j$ (di mana $j$ berjalan dari 1 hingga $k$).
*   Yang terpenting dari rumusan ini, karena setiap tebakan ke-$j$ menghasilkan kelompok prinsip yang unik $\{p_{i,j}\}^{m_j}$, maka skor akhir yang awalnya sempit dalam ruang diskrit kecil (contoh $\{1, ..., 10\}$) dapat memuai hingga kelipatan $k$. *Voting process actually expands the reward space by $k$ times!* Hal ini membawa model untuk mendeteksi distribusi kebenaran sebenarnya dari sebuah kumpulan respons dengan jauh lebih kaya.

**Meta Reward Modeling Guided Voting**
Pengambilan sampel yang banyak bisa jadi meloloskan kritik halusinasi (sampel jelek akibat model terbatas/acak). Maka, sebuah model khusus, **Meta RM**, dilatih secara spesifik untuk membuang suara yang jelek. Meta RM adalah model skalar pointwise sederhana yang dilatih dengan *binary cross-entropy loss* (1 untuk sampel yang kritiknya valid dan menuntun ke jawaban benar, 0 untuk sebaliknya). Pemungutan suara yang dibimbing ini berjalan sangat efektif: Dari total $k$ sampel suara, nilai final dijumlahkan HANYA berdasarkan $k_{meta} \le k$ suara teratas yang dipilih oleh Meta RM. Sisanya dibuang.

---

## 5. Hasil pada Tolok Ukur Pemodelan Imbalan (*Benchmarks*)

### 5.1 Pengaturan Eksperimen

Kita akan menguji validitas model ini di empat kumpulan data atau tolok ukur utama:
1.  **Reward Bench (RB):** Tolok ukur paling komprehensif.
2.  **PPE:** Tolok ukur untuk mendeteksi preferensi dan kebenaran faktual (*Correctness*).
3.  **RMB:** Evaluasi komprehensif di domain pembantuan manusia dan harmlessness (*safety*).
4.  **ReaLMistake:** Tolok ukur mendiagnosis *error* spesifik (hanya pada satu respons).

*Metric* standarnya adalah akurasi deteksi pilihan respons yang benar dari sebuah daftar/pasangan, dan evaluasi *ROC-AUC* untuk ReaLMistake. DeepSeek-GRM-27B diimplementasikan menggunakan sasis Gemma-2-27B, sedangkan untuk uji coba ukuran model lain dipakai rentang mulai dari V2-Lite (16B MoE) hingga V3 (671B MoE). Suhu inferensi (*Temperature*) ditetapkan pada angka $0.5$.

### 5.2 Hasil dan Analisis

**Performa pada Tolok Ukur RM**
Mari kita tinjau hasil yang disajikan pada **Tabel 2**.

| Model | RB | PPE Pref. | PPE Correct. | RMB | Keseluruhan (*Overall*) | Rata-Rata Rank ($\downarrow$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Hasil Rekaman Model Publik (Reported Results)** | | | | | | |
| Skywork-Reward-Gemma-2-27B | 94.1 | 56.6 | 56.6 | 60.2 | 66.9 | - |
| Nemotron-4-340B-Reward | 92.0 | 59.3 | 60.8 | 69.9 | 70.5 | - |
| GPT-4o | 86.7 | 67.1 | 57.6 | 73.8 | 71.3 | - |
| **Hasil Implementasi Ulang Baseline (Reproduced Results)** | | | | | | |
| LLM-as-a-Judge | 83.4 | 64.2 | 58.8 | 64.8 | 67.8 | 4.50 |
| *DeepSeek-BTRM-27B* (Skalar) | 81.7 | 68.3 | 66.7 | 57.9 | 68.6 | 3.50 |
| *CLoud-Gemma-2-27B* (Semi-Skalar)| 82.0 | 67.1 | 62.4 | 63.4 | 68.7 | 3.50 |
| DeepSeek-PairRM-27B | 87.1 | 65.8 | 64.8 | 58.2 | 69.0 | **2.75** |
| **Hasil Metode Kita Tanpa Penskalaan (Results of Our Method)** | | | | | | |
| DeepSeek-GRM-27B-RFT (Ours) | 84.5 | 64.1 | 59.6 | 67.0 | 68.8 | 4.00 |
| **DeepSeek-GRM-27B (Ours)** | 86.0 | 64.7 | 59.8 | 69.0 | **69.9** | **2.75** |
| **Hasil Penskalaan Waktu-Inferensi (Voting@32)** | | | | | | |
| DeepSeek-GRM-27B (Ours) | 88.5 | 65.3 | 60.4 | 69.7 | 71.0 | - |
| **DeepSeek-GRM-27B (MetaRM) (Ours)** | **90.4** | **67.2** | **63.2** | **70.3** | **72.8** | - |

*Analisis Tabel 2:*
*   Metode SPCT (DeepSeek-GRM-27B tanpa Meta) menunjukkan performa generalis yang tangguh dengan skor keseluruhan 69.9, secara langsung mengalahkan *baseline* di kelasnya (seperti CLoud-Gemma dan LLM-as-a-Judge). 
*   Hebatnya, saat metode ini dikenakan *penskalaan waktu inferensi* ($Voting@32$ yang dipandu *MetaRM*), skornya membengkak tajam mencapai **72.8**, menjadikannya yang terbaik mengalahkan para raksasa korporat ber-parameter raksasa sekalipun seperti Nemotron-4-340B (70.5) dan bahkan model tertutup sekelas GPT-4o (71.3).
*   *Catatan Akademis Tambahan:* Model skalar klasik seperti BTRM cenderung bias. Mereka bisa bagus dalam 1 hal (seperti angka absolut di PPE Correct) tetapi rontok dan membias di hal lain (seperti jatuh bebas di benchmark RMB: 57.9). Sebaliknya, SPCT meningkatkan kapabilitas secara general tanpa memicu bias fatal ke salah satu dominasi domain.

**Skalabilitas Waktu-Inferensi**

Mari periksa detail dinamika penskalaan dari k=1 hingga k=32 pada **Tabel 3**.

| Model | Keseluruhan (*Overall*) |
| :--- | :--- |
| **Penskalaan Waktu-Inferensi (Voting@1)** | |
| LLM-as-a-Judge | 67.0 |
| *CLoud-Gemma-2-27B* | 68.5 |
| DeepSeek-GRM-27B (Ours) | 67.9 |
| **Penskalaan Waktu-Inferensi (Voting@8)** | |
| LLM-as-a-Judge | 67.6 (+0.6) |
| *CLoud-Gemma-2-27B* | 68.8 (+0.3) |
| DeepSeek-GRM-27B (Ours) | 70.6 (+2.7) |
| DeepSeek-GRM-27B (MetaRM) (Ours) | 72.0 (+4.1) |
| **Penskalaan Waktu-Inferensi (Voting@32)** | |
| DeepSeek-GRM-27B (Ours) | 71.0 (+3.1) |
| DeepSeek-GRM-27B (MetaRM) (Ours) | **72.8 (+4.9)** |

*Analisis Tabel 3:*
*   Dengan maksimal $k=8$ sampel, DeepSeek-GRM-27B mencatatkan lonjakan paling tajam dari angka Greedy Decoding ($+2.7$), melampaui stagnansi model Semi-Skalar CLoud-Gemma yang hanya tumbuh marjinal ($+0.3$).
*   Lonjakan ini tidak terputus dan terus melebar hingga pada $k=32$, menembus $72.8$ dengan pengawal MetaRM (suatu pertumbuhan $+4.9$ yang luar biasa!). Kekuatan penskalaan tak kenal lelah ini terbukti karena ekspansi resolusi dan varians prinsip terperinci di setiap pengambilan ruang nilai imbalannya.

**Studi Ablasi (*Ablation Study*)**
Untuk menjamin tidak ada komponen yang tak berguna, peneliti mencabut/melakukan ablasi satu per satu unsur sistem pada **Tabel 4**.

| Metode | Keseluruhan (*Overall*) |
| :--- | :--- |
| **Hasil Greedy Decoding** | |
| **DeepSeek-GRM-27B** | **69.9** |
| w/o Principle Generation (*tanpa Pembangkitan Prinsip*) | 67.5 |
| w/o Rejective Sampling (*tanpa Pengambilan Sampel Penolakan*) | 68.7 |
| **Hasil Penskalaan Waktu-Inferensi (Voting@8)** | |
| **DeepSeek-GRM-27B** | **70.6** |
| w/o Principle Generation (*tanpa Pembangkitan Prinsip*) | 68.0 |

*Analisis Tabel 4:*
*   Menghilangkan fase pembentukan *Prinsip* sangat krusial; performa melorot telak dari 69.9 ke 67.5 pada status decoding biasa, dan dari 70.6 ke 68.0 saat di-*voting*. Ini adalah kesimpulan mutlak: **Pembangkitan Prinsip adalah napas (*crucial*) dari kualitas dan penskalaan waktu-inferensi DeepSeek-GRM-27B**. 
*   Tanpa Rejective Sampling (Fase RFT/Cold Start dicabut), performa juga terpengaruh. Meski begitu, model yang langsung dilatih dengan instruksi standar RL masih memberikan nilai (68.7), yang membuktikan betapa perkasanya optimisasi Online RL.

**Biaya Skala Inferensi Vs. Pelatihan**

> **Analisis Gambar 4: Grafik Perbandingan Penskalaan Saat Inferensi (a) vs Penskalaan Saat Pelatihan/Parameter (b)**
>
> Mahasiswa sekalian, ini adalah penemuan paling filosofis yang bisa merombak cara para *engineer* mengalokasikan anggaran (*budget*).
> *   **Grafik 4(a) (Inference-time scaling):** Sumbu X adalah jumlah voting (logaritmik 1 hingga 32). Sumbu Y adalah skor (dari 81.0 hingga 91.0). Terdapat garis merah tajam DeepSeek-GRM-27B (MetaRM) yang melesat menembus atap dari skor 85 menuju puncak $>90.0$ pada $k=32$. Bahkan model berparameter lebih kecil, DeepSeek-GRM-16B (garis hijau di bawah), jika diberikan kesempatan melakukan voting inferensi, grafiknya terus mendaki ke angka $\approx 85.0$.
> *   **Grafik 4(b) (Training-time scaling):** Sumbu X melambangkan membengkaknya ukuran raksasa parameter sebuah model saat pelatihan (mulai dari 8 Miliar (8B), hingga ukuran fantastis 648 Miliar (648B)). Sumbu Y adalah skor (81.0 ke 91.0). Di sini kita melihat titik model RFT menanjak dari parameter 24B ke atas 200B dengan laju yang sangat lambat. Bahkan model DeepSeek-V3 seberat raksasa 671B yang menggunakan serakah decoding biasa (garis paling kanan atas, titik "DeepSeek-V3") hanya menyentuh rentang $\approx 88.0$. Titik abu RL juga perlahan naik.
> *   **Kesimpulan Besar:** Gambar ini meneriakkan sebuah realita: "Melakukan *Voting / Penskalaan Inferensi* sebanyak 32 kali pada model 27B JAUH LEBIH BERHASIL mencapai skor 90.4, dibandingkan Anda membakar uang triliunan rupiah untuk melatih model raksasa 671B di waktu pelatihan (skornya hanya tertahan di angka $\approx 88.0$)"! Bukti ini memvalidasi efektivitas epik penskalaan di waktu inferensi dibanding sekadar membesarkan otak parameter model.

---

## 6. Pekerjaan/Riset Terkait (*Related Work*)

Sebagai akademisi, Anda harus memahami di mana posisi makalah ini dibandingkan riset terdahulu:
1.  **Pemodelan Imbalan Generatif (GRMs):** Adalah evolusi pergeseran paradigma (*paradigm shift*) mematikan model Skalar, guna merepresentasikan skor secara linguistik/tekstual untuk kekayaan justifikasi referensi. Metrik terdahulu sekelas LLM-as-a-Judge membuka pintu ini, disusul oleh model-model Semi-Skalar. Sayangnya semua masih terantuk tembok in-efisiensi (*challenges in efficiency*).
2.  **Penskalaan Waktu-Inferensi untuk LLM:** Studi kontemporer paralel di mana penskalaan agregat sampel RM dan dorongan *Chain-of-Thought (CoT)* digunakan untuk menstimulasi penalaran cerdas model (seperti pendekatan OpenAI o1 dan model R1). Riset ini melengkapi celah yang ada dengan memberikan injeksi skalabilitas ke arah perumusan imbalan universal.

---

## 7. Kesimpulan dan Pekerjaan Masa Depan (*Conclusion and Future Work*)

Sebagai penutup kuliah, mari kita rangkum capaian metodologis ini:
Kita telah diperkenalkan pada fondasi **Penyetelan Kritik Berprinsip Mandiri (*Self-Principled Critique Tuning / SPCT*)**, yang secara revolusioner membuka kunci ke arah skalabilitas waktu-inferensi untuk Pemodelan Imbalan Generalis.

Dengan pengawalan dari RL daring berbasis algoritma turunan aturan mutlak, SPCT memaksa model melepaskan prinsip-prinsip statis menuju generasi mandiri yang adaptif (*adaptive generation of principles and critiques*). Hal ini secara ekstrem melejitkan kualitas dan mendorong penskalaan lintas domain. Melalui metode ini, **DeepSeek-GRM menelan habis seluruh persaingan model publik terkuat di arena global (termasuk GPT-4o)**, dan melalui bantuan pemandu dewa Meta RM, performa DeepSeek melangit dengan peningkatan eksponensial di tahap inferensi.

Di masa depan (*future direction*), integrasi model-model imbalan generalis yang memuai pada tahapan *inference-time* ke dalam rantai (*pipelines*) RL daring penuh diprediksi akan menjadi antarmuka/poros terkuat sebagai evaluator model-model basis masa depan.

---

## Pernyataan Etika (*Ethics Statement*)

Metode SPCT yang diajukan bertujuan murni untuk efisiensi skalabilitas komputasi sistem AI. Namun, implikasi etika tidak boleh luput dari perhatian kita di bangku akademis.

Meskipun secara empiris sistem pemodelan generalis (DeepSeek-GRM) telah mereduksi bias parsial, generasi otomatis dari prinsip itu sendiri rawan untuk secara implisit *mengabadikan* atau bahkan *mengamplifikasi* racun atau bias buruk (*toxic biases*) apabila data dasar yang dicekokkan ke sistem terkontaminasi. Makalah ini berpendapat tegas bahwa pencarian lanjutan dalam mitigasi bias dan filter Meta RM harus dijadikan prioritas primer. Lebih jauh lagi, otomatisasi *Self-Principled* ini bukan dibentuk untuk mendepak peran dewan pengawas manusia (*diminish human oversight*). Sebaliknya, *framework human-in-the-loop* harus dipelihara, dan SPCT dirancang sebagai tangan perpanjangan yang kokoh untuk menskalakan peran evaluasi manusia agar tidak kewalahan di lautan data masa depan.
