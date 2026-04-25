---
slug: modul-4-3
title: modul-4-3
---
## 4.3 Tahapan melatih LLM dari awal: pretraining dan post-training (SFT, Preference Alignment)
---

### 4.3.1 Pretraining : Mengajarkan LLM Kemampuan Berbahasa

Seperti yang telah dipelajari sebelumnya pada modul 3, model bahasa modern dilatih secara *self-supervised* untuk mempelajari sendiri struktur dan konsep dalam bahasa dari data teks dengan skala yang sangat besar. Tahap ini disebut *pretraining* yang dilakukan untuk melatih kemampuan bahasa LLM serta mempelajari embedding yang tepat untuk setiap token dalam *vocabulary* secara bersamaan.

### 4.3.2 Skenario latih

*Pretraining* dilakukan secara *self-supervised* dengan melakukan *autoregressive next token prediction*, yang artinya secara berkelanjutan output dari sebuah LLM dimasukkan menjadi input untuk menghitung output selanjutnya hingga tercapai suatu kondisi berhenti tertentu atau LLM selesai mengeluarkan output.

![Ilustrasi autoregressive](https://lms.sdmdigital.id/pluginfile.php/635388/mod_page/content/47/image28.jpg)
**Gambar 7. Ilustrasi autoregressive next token prediction. Setiap input dan output merupakan token dalam sebuah vocabulary, dan output dari waktu t-1 digunakan untuk menghitung output untuk waktu t**

Data teks yang sudah disiapkan diproses dulu menjadi token individual. Selain token yang berasal dari proses *tokenization*, beberapa token spesial seperti penanda awal teks, penanda akhir teks, penanda pemisah teks, serta token lainya yang dibutuhkan untuk menandai atau mengontrol perilaku tertentu juga termasuk ke dalam token yang menjadi input LLM. Token-token spesial ini dimasukkan berdasarkan logika tertentu dan bukan merupakan bagian dari teks aslinya. Token-token ini kemudian dimasukkan ke dalam LLM secara berurutan untuk menghitung outputnya dalam bentuk distribusi peluang token selanjutnya yang dibandingkan dengan label untuk menghitung nilai *loss*.

Karena sifat alami dari teks, kita tidak memerlukan label eksplisit, karena teks sudah memiliki struktur tertentu sehingga setiap kelanjutan dari sebuah teks merupakan label untuk output yang dihitung menggunakan bagian-bagian sebelumnya dari sebuah teks.

![Proses Pretraining](https://lms.sdmdigital.id/pluginfile.php/635388/mod_page/content/47/image25.jpg)
**Gambar 8. Ilustrasi proses pretraining menggunakan teks**

Dalam ilustrasi di atas, dijelaskan skenario *pretraining* dan mengapa label tidak dibutuhkan. Misalnya kita melatih LLM untuk menebak kata selanjutnya dari kalimat “Aku suka makan bakso, mie ayam dan tahu tek”. Saat kita memproses input pertama yaitu “Aku”, kita sudah tahu kelanjutan dari kalimatnya adalah “suka”. Pada tahap selanjutnya, kita kemudian memproses lagi input kedua, yaitu “suka”. Kita tahu dari kalimatnya bahwa kata selanjutnya adalah “makan”, yang lalu kita jadikan label untuk tahap ini. Informasi ini menjadi label yang berasal dari struktur teks itu sendiri. Inilah mengapa tahap ini disebut *self-supervised*, dimana label berasal dari data informasi maupun struktur yang terkandung di dalam data itu sendiri.

### 4.3.4 Detil Proses Latih

LLM memiliki batas total token input dan output yang bisa diproses yang disebut sebagai *context window*. Selama proses pretraining, data teks dari berbagai sumber akan di-*packing* agar setiap *context window* dapat dimanfaatkan secara optimal. *Packing* ini bertujuan memaksimalkan *compute* yang dilakukan selama *forward pass*.

Biasanya, kumpulan dokumen dengan panjang bervariasi akan digabung dan dipotong menjadi segmen yang ukurannya mendekati panjang *context window* model. Jika satu dokumen berakhir sebelum kapasitas tercapai, maka dilanjutkan dengan bagian awal dokumen lain hingga penuh. Agar model tetap bisa membedakan batas antar dokumen, ditambahkan *special tokens* seperti `<bos>` (beginning of sequence) dan `<eos>` (end of sequence). Token khusus lain seperti `<pad>` digunakan hanya saat perlu menjaga keseragaman panjang pada *batch* dan tidak ikut berkontribusi terhadap perhitungan *loss*.

![Packing Sequence](https://lms.sdmdigital.id/pluginfile.php/635388/mod_page/content/47/Screenshot%202026-01-02%20at%2014.32.28.png)
**Gambar 9. Ilustrasi penggunaan token special token `<EOS>` untuk memisahkan dua dokumen yang berbeda dalam satu sequence atau baris input**

Perhitungan *loss* sendiri dilakukan hanya pada token-token yang valid (bukan *padding*), dengan menggunakan *context masking* untuk memastikan model tidak mengakses informasi dari token di masa depan. Masking ini menjaga arah kausal prediksi token selanjutnya sesuai formulasi *next token prediction* yang digunakan dalam pretraining. Selain itu perhitungan loss juga hanya dilakukan dalam level setiap dokumen, agar model tidak menggunakan konteks dari dokumen sebelumnya ketika mempelajari dokumen lain di dalam *context window*.

Pendekatan *packing* yang efisien dan penanganan *masking* yang tepat membantu memaksimalkan penggunaan komputasi, mempercepat proses latih, serta menjaga stabilitas pembelajaran terutama pada dataset besar dengan panjang dokumen yang sangat bervariasi.

### 4.3.5 Target Optimasi

Prediksi token selanjutnya dari sebuah teks dirumuskan sebagai *Maximum Log-likelihood estimation*, yang sudah umum ditemui dalam *machine learning*. Tujuan pembelajaran model dalam tahap ini adalah memahami distribusi peluang kondisional dari token selanjutnya, dikondisikan oleh token-token yang menjadi input. Dengan demikian model akan dilatih untuk semakin sering memprediksi token yang benar dengan probabilitas tertinggi. Untuk melatih kemampuan ini, digunakan *Cross Entropy Loss* yang umum digunakan pada klasifikasi.

**Formula 5**
$$L_{CrossEntropy} = -\sum_{n=1}^{N} y_n \log(p_n)$$
**Caption: Formula 5. Rumus cross entropy loss**

Dalam kasus LLM, karena hanya 1 output yang diambil dari proses sampling, nilai ini berarti hanya dihitung dengan rumus berikut:

**Formula 6**
$$L_{CrossEntropy} = -\log(p_y)$$
**Caption: Formula 6. Rumus cross entropy loss untuk 1 kelas output saja**

Dimana $p_y$ adalah probabilitas logit pada indeks kelas token yang menjadi label. Ilustrasi berikut menunjukkan perhitungan *Cross Entropy Loss* untuk satu sampel data:

![Cross Entropy Calculation](https://lms.sdmdigital.id/pluginfile.php/635388/mod_page/content/47/img32a.png)
**Gambar 10. Ilustrasi perhitungan Cross Entropy Loss**

### 4.3.6 Cara Mengukur Kinerja Latih

Kinerja latih dari proses *pretraining* umumnya diukur menggunakan nilai *loss* maupun nilai *perplexity* pada data lain yang sama sekali belum pernah “dilihat” model melalui proses *training* yang merupakan eksponensial bilangan natural dari *loss*. Selain itu juga lumrah digunakan berbagai macam *benchmark* spesifik penggunaan atau *domain* tertentu.

**Tabel 7. Beberapa dataset/benchmark spesifik domain tertentu, digunakan untuk menguji kemampuan berbahasa LLM**

| Benchmark | Deskripsi | Skenario Pengujian | Metrik yang diukur |
| :--- | :--- | :--- | :--- |
| ARC-e, ARC-c, HellaSwag, BooIQ | Benchmark untuk mengukur kemampuan reasoning *common sense* | LLM diberi pertanyaan *common sense reasoning* (pilihan ganda, jawaban yes/no) | Ketepatan jawaban, misal akurasi *zero-shot* |
| Natural Questions, TriviaQA | Benchmark untuk menguji kemampuan LLM menjawab pertanyaan secara *closed book* | LLM diberi pertanyaan tanpa diberikan akses ke informasi pembantu | Ketepatan jawaban seperti akurasi *zero-shot* dan *few shot* |
| MATH, GSM8k | Mengukur kemampuan *reasoning* matematika | Diberi soal matematika lalu jawabannya dinilai | Ketepatan jawaban |
| RACE-middle, RACE-high | Berisi soal untuk menguji *reading comprehension* | Diberikan teks lalu diberi pertanyaan mengenai teks tersebut | Ketepatan jawaban |
| HumanEval, MBPP | Menguji kemampuan LLM untuk menulis kode pemrograman | Diberikan deskripsi program, diminta menulis kode dalam bahasa Python | Validitas kode yang ditulis |

### 4.3.7 Post Training : Tuning Spesifik Domain atau Task Tertentu dan Menyelaraskan Perilaku LLM dengan Preferensi Manusia

Walaupun ditemukan kalau LLM yang telah melewati proses *pretraining* memiliki kemampuan mengikuti instruksi dan mempelajari informasi yang baik, sifat keterbukaan dari data internet memberikan beberapa resiko seperti bahasa yang *toxic* atau berbahaya. Tahap ini melibatkan data berlabel untuk melatih kemampuan tertentu. Umumnya tahap ini dibagi menjadi dua, yaitu ***Instruction Tuning*** dan ***Preference Tuning / Alignment***.

### 4.3.8 Instruction Tuning : Mengajarkan LLM cara Mengikuti Instruksi

Tahap ini umumnya memerlukan skala data yang jauh lebih kecil karena LLM sudah memiliki basis yang baik dalam memahami bahasa dari proses *pretraining*. Data diberikan dalam bentuk pasangan instruksi dan respon.

Contoh data instruksi (format JSON):
```json
{
  "instruksi": "tolong klasifikasi sentimen dari teks berikut: ...",
  "respon": "positif"
}
```

### 4.3.9 Preference Alignment : Mengajarkan LLM Agar Tidak Toxic, Harmful, dan Menyelaraskannya dengan Preferensi Manusia

Berbeda dengan *supervised fine tuning* dimana kriteria keberhasilannya jelas, etika dan moral memiliki *nuance* yang sulit diekspresikan dalam bentuk pasangan input dan output. Tahap ini memerlukan data berupa dua atau lebih respon untuk suatu pertanyaan yang kemudian diurutkan oleh annotator berdasarkan preferensi.

![Preference Ranking](https://lms.sdmdigital.id/pluginfile.php/635388/mod_page/content/47/image18.jpg)
**Gambar 11. Ilustrasi mengurutkan berbagai respon untuk input yang sama berdasarkan preferensi, contohnya kesopanan**

Dunia akademik dan industri mengenal dua metode utama: ***Proximal Policy Optimization*** (PPO) berbasis *Reinforcement Learning* (RL) dan ***Direct Preference Optimization*** (DPO) yang berbasis SFT.

![Reward Model](https://lms.sdmdigital.id/pluginfile.php/635388/mod_page/content/47/image49.jpg)
**Gambar 12. Reward model memberikan skor yang lebih tinggi untuk respon dengan tingkat preferensi yang lebih tinggi**

**Tabel 8. Perbandingan singkat antara metode preference alignment berbasis SFT dan RL**

| Aspek | Preference alignment berbasis SFT | Preference alignment berbasis RL |
| :--- | :--- | :--- |
| **Cara belajar** | Langsung mempelajari preferensi dari pasangan respon ($y^+, y^-$) | Melatih *reward model* lalu mengarahkan perilaku menggunakan RL |
| **Kompleksitas** | Sedang, sama seperti *fine-tuning* biasa | Tinggi, terdiri dari beberapa tahap |
| **Stabilitas** | Cukup stabil | Loss dan gradien bisa bervariasi liar |
| **Performa** | Cukup baik | Umumnya lebih baik (menangani variasi tinggi) |
| **Cost** | Rendah | Tinggi (komputasi lebih besar) |

### 4.3.10 Memahami Reinforcement Learning dan Penggunaannya untuk Melakukan Alignment

RL melibatkan agen (LLM) yang berinteraksi dengan lingkungan (state berupa token dalam context window) dan melakukan aksi (mengeluarkan token) untuk memaksimalkan *reward* (skor dari *reward model*).

![RL Alignment](https://lms.sdmdigital.id/pluginfile.php/635388/mod_page/content/47/image33.jpg)
**Gambar 13. Simplifikasi proses alignment LLM menggunakan algoritma RL**

### 4.3.11 Target Optimasi

**PPO: Melatih reward model**
*Reward model* dioptimasi menggunakan *pairwise preference loss*:

**Formula 7**
$$\mathcal{L}(\theta) = \log(Sigmoid(r(\mathbf{x}, \mathbf{y}_1) - r(\mathbf{x}, \mathbf{y}_2)))$$
**Caption: Formula 7. Loss function untuk pasangan respon y1 (buruk) dan y2 (baik) dari input x untuk melatih reward model**

![Sigmoid](https://lms.sdmdigital.id/pluginfile.php/635388/mod_page/content/47/image30a.png)
**Gambar 14. Fungsi sigmoid**

**PPO: Melatih policy (update parameter LLM)**
Target optimasi utama PPO:

**Formula 8**
$$\mathcal{L}(\theta) = -\mathbb{E}_{(x, y) \sim D} [U(x, y; \theta)]$$
**Caption: Formula 8. Target optimasi utama dari PPO**

**Formula 9**
![PPO Utility](https://lms.sdmdigital.id/pluginfile.php/635388/mod_page/content/47/image21.png)
**Caption: Formula 9. Rumus utility function PPO**

Intuisi dari PPO adalah menghitung rasio probabilitas model baru dibanding model lama:
![PPO Ratio](https://lms.sdmdigital.id/pluginfile.php/635388/mod_page/content/47/Screenshot%202025-12-23%20at%2015.23.21.png)

Rasio ini dikalikan dengan ***advantage function (A)***:
**Formula 10**
$$A(s_t, a_t) = r_t + \gamma V(s_{t+1}) - V(s_t)$$
**Caption: Formula 10. Definisi advantage function**

**DPO: formulasi preferensi sebagai supervised fine tuning**
DPO melakukan tuning langsung tanpa *reward model* tambahan:

**Formula 11**
![DPO Loss](https://lms.sdmdigital.id/pluginfile.php/635388/mod_page/content/47/image23.png)
**Caption: Formula 11. Loss function DPO**

Peluang keseluruhan output model dihitung sebagai penjumlahan *log-probability* dari setiap token:

**Formula 12**
$$\log \pi(y|x) = \sum_{t=1}^{T_y} \log \pi(y_t | x, y_{<t})$$
**Caption: Formula 12. Menghitung peluang sebuah model $\pi$ memberikan output y diberikan input x**

---

### Mini Quiz
*(Silakan akses kuis interaktif pada platform LMS)*

> **Refleksi Singkat**
> Mengumpulkan data merupakan proses yang sangat merepotkan. Belakangan ini banyak diterapkan penggunaan LLM lainnya untuk membantu atau bahkan membuat data sintetis.
> Menurut anda, apa potensi bahaya dari praktik ini? Apakah efektif di industri? Bagaimana cara yang baik dalam memanfaatkan LLM untuk membantu pengumpulan data?