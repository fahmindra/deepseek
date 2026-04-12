---
slug: modul-5-3
title: modul-5-3
---
# 1.3 Tahapan Pelatihan LLM: Pretraining

Setelah selesai mengumpulkan, membersihkan, dan menyiapkan data, sekarang sudah waktunya melakukan pretraining. Tahap ini merupakan tahap paling krusial dalam membangun kemampuan berbahasa LLM yang menjadi basis utama dari kemampuan LLM untuk diadaptasikan melalui tahapan-tahapan selanjutnya.

Umumnya, pretraining dilakukan atas beberapa tahap. Pertama adalah menentukan dulu cara evaluasi performa model, lalu menyiapkan infrastruktur untuk pretraining, termasuk memilih dan menyiapkan model dan dataloader. Setelah itu dilanjutkan dengan melakukan proses training dan evaluasi.

## 1.3.1 Menentukan cara evaluasi LLM

Tahap ini wajib dilakukan paling pertama, karena berkaitan langsung dengan use case dan tujuan pengembangan model, serta akan menjadi kompas pengambilan keputusan selama proses training.

**Memahami perbedaan metrik dengan benchmark**

Sebuah **metrik** adalah ukuran kuantitatif dari suatu aspek performa, misalnya akurasi, F1 Score, BLEU, atau ROUGE. Metrik selalu memerlukan konteks agar bisa dimengerti, misalnya:
* Akurasi dalam menjawab pertanyaan pilihan ganda
* F1 Score untuk klasifikasi teks
* BLEU untuk kualitas penerjemahan
* ROUGE untuk kemampuan merangkum

LLM umumnya dilatih untuk memiliki lebih dari satu kemampuan, selain kemampuan pemodelan bahasa, LLM juga diinginkan bisa mengikuti instruksi, atau bisa menulis kode, atau bisa melakukan reasoning untuk menyelesaikan masalah yang rumit. Setiap kemampuan ini perlu diukur secara terpisah secara terstandar untuk membandingkan LLM yang dilatih dengan LLM lainnya.

Di sinilah peran **benchmark**. Benchmark memberikan kerangka evaluasi standar untuk mengukur kemampuan LLM pada task spesifik secara komprehensif. Setiap benchmark terdiri dari tiga komponen utama:
1. **Dataset terstandar:** kumpulan data uji yang konsisten untuk semua model.
2. **Metrik evaluasi:** ukuran performa yang telah ditentukan / disamakan berdasarkan konvensi untuk memudahkan perbandingan (bisa satu atau kombinasi beberapa metrik).
3. **Definisi task:** konteks jelas mengenai kemampuan apa yang diuji.

Contohnya, benchmark MMLU mengukur pengetahuan pada 57 subjek berbeda menggunakan metrik akurasi, sementara HumanEval menguji kemampuan coding dengan metrik pass@k. Dengan menggunakan berbagai benchmark, kita mendapat profil lengkap tentang kekuatan dan kelemahan sebuah LLM.

Perlu diingat bahwa metrik yang digunakan pada setiap benchmark dapat berubah, karena benchmark sendiri hanya menyediakan task, konteks serta datanya saja. Umumnya penggunaan metrik disepakati oleh komunitas dan bisa berubah-ubah ketika ada metrik yang lebih efektif, misalnya BERTScore digunakan untuk menggantikan metrik seperti BLEU dan ROUGE.

Berikut adalah contoh beberapa benchmark yang umum digunakan untuk mengukur performa LLM. Tabel ini tidak exhaustive, masih banyak lagi benchmark lainnya di luar sana. Tabel ini memberikan gambaran seberapa bervariasi jenis benchmark yang digunakan untuk menilai kemampuan model.

**Tabel 5. Beberapa contoh benchmark yang umum digunakan**

| Benchmark | Apa bentuknya | Apa yang diukur | Bagaimana cara mengukurnya | Metrik yang biasanya digunakan |
| :--- | :--- | :--- | :--- | :--- |
| **MMLU** (Massive Multitask Language Understanding) | 15,908 Pertanyaan pilihan ganda pada 57 subjek berbeda (STEM, humanities, social sciences) | Pengetahuan umum dalam berbagai domain | Pertanyaan pilihan ganda, dibandingkan dengan kunci jawaban | Akurasi |
| **HumanEval** | 164 Soal pemrograman dengan test-case masing masing | Kemampuan menghasilkan kode | LLM diminta membuat sebuah fungsi untuk menjawab soal sesuai dengan template tertentu, lalu diuji pakai test case | pass@k |
| **GSM8K** | 8,500 soal cerita matematika | Kemampuan memahami dan memecahkan soal matematika | Membandingkan jawaban model dengan kunci jawaban | Akurasi |
| **HellaSwag** | 10,000+ pertanyaan lanjutan kata | Kemampuan memahami situasi sehari-hari | Diberi 4 pilihan, LLM diminta memilih 1 | Akurasi |
| **MBPP** (Mostly Basic Python Problems) | 974 Soal pemrograman Python + Test case | Kemampuan pemrograman Python | Sama seperti humaneval | pass@k |
| **ARC** (AI2 Reasoning Challenge) | 7,787 pertanyaan sains, campuran soal mudah dan sulit | Pengetahuan sains dan kemampuan menjawab pertanyaan | Pilihan ganda | Akurasi |
| **MATH** | 12,500 soal kompetisi matematika | Kemampuan matematika yang lebih lanjut | Keseluruhan proses menjawab dibandingkan dengan ground truth | Akurasi |
| **Human Eval+** | Versi lebih sulit dibanding Human Eval | Kemampuan pemrograman | Sama seperti Human Eval | pass@k |
| **MT-Bench** | 80 percakapan panjang | Kemampuan bercakap-cakap, terutama untuk percakapan yang panjang | Dinilai oleh LLM yang distandarkan (GPT-4), menggunakan skor 1-10 | Skor rata-rata |
| **AlpacaEval** | 805 pasangan instruksi dan responnya | Kemampuan mengikuti instruksi | Respon model dibandingkan dengan referensi (LLM lain, manusia) | Win rate |

Selain menggunakan *benchmark* yang tersedia, perlu disusun sendiri benchmark yang lebih spesifik pada *use case* pengembangan model. Misalnya anda melatih LLM yang bisa berbahasa jawa, artinya perlu diukur berbagai kemampuan dalam bahasa jawa:
* Bisa menjawab pertanyaan dalam bahasa jawa
* Bisa mengerti informasi dalam bahasa jawa
* Bisa menjawab pertanyaan kultural mengenai budaya jawa

Lalu bisa disusun benchmark untuk masing-masing kemampuan tersebut disertai dengan cara evaluasi serta metrik yang perlu dihitung.

## 1.3.2 Bermacam metrik yang digunakan ketika melakukan evaluasi LLM

Secara intrinsik, metrik yang dihitung untuk melakukan evaluasi LLM adalah **perplexity (PPL)** yang mengukur seberapa baik sebuah model bisa memprediksi suatu sampel. Jika diberikan sebuah teks input, PPL menghitung statistik seberapa "tidak yakin" suatu model memprediksi token-token selanjutnya.
Misalnya kita memberikan input berikut untuk diuji:

```text
<BOS>Mi ayam dan bakso adalah makanan kesukaan pak Hadi
```

Untuk penyederhanaan penjelasan, anggap saja yang menjadi output LLM adalah per kata. Realitanya perplexity dihitung dari token ke token, tapi konsepnya tetap sama.

Perplexity dihitung dengan cara mengukur tingkat confidence dari token selanjutnya pada setiap titik. Anggap saja diamati nilai-nilai confidence berikut, dimulai ketika model mengeluarkan token "Mi":

| Token | Confidence |
| :--- | :--- |
| Mi | 0.2 |
| ayam | 0.6 |
| dan | 0.7 |
| bakso | 0.8 |
| adalah | 0.6 |
| makanan | 0.5 |
| kesukaan | 0.7 |
| pak | 0.8 |
| Hadi | 0.2 |

Selanjutnya perplexity bisa dihitung dengan rumus **Perplexity** berikut:

$$\text{Perplexity} = \exp \left( -\frac{1}{N} \sum_{i=1}^{N} \log P(w_i | w_1, \dots, w_{i-1}) \right) \qquad (\text{Eq. 1})$$

Dimana:
* N merupakan total token (selain token pertama), dan
* $P(w_i|...)$ merupakan nilai confidence token yang selanjutnya diprediksi.

Menggunakan rumus tersebut, didapatkan nilai perplexity sebesar 442.88.

Selain perplexity, beberapa metrik digunakan bersamaan dengan benchmark spesifik untuk mengevaluasi kemampuan LLM:

**1. Akurasi**

Akurasi merupakan metrik yang paling umum. Akurasi membandingkan jawaban dengan sebuah ground truth, lalu menghitung persentase jawaban yang benar. Berikut rumus **Akurasi**:

$$\text{Akurasi} = \frac{\text{Jumlah jawaban benar}}{\text{Jumlah jawaban}} \qquad (\text{Eq. 2})$$

LLM umumnya memiliki output dalam bentuk teks bebas. Untuk jenis benchmark yang menginginkan output yang jelas / dalam bentuk pilihan, jawaban bisa langsung dibandingkan secara programatis. Benchmark lainnya yang menguji kebenaran secara semantik mungkin memerlukan cara lain untuk membandingkan jawaban, misalnya menggunakan LLM.
* **Digunakan pada benchmark:** MMLU, HellaSwag, ARC, GSM8K
* **Cocok untuk:** Klasifikasi, pilihan ganda, task yang jawabannya jelas (hanya 1) dan hanya bisa benar atau salah (tidak ada nilai parsial)

**2. F1 Score**

Metrik ini umumnya diukur pada klasifikasi dimana setiap prediksi hanya berupa positif (inputnya termasuk ke suatu kelas) dan negatif (inputnya tidak termasuk ke suatu kelas). F1 mengukur harmonic mean antara precision dan recall. Berikut rumus **F1-Score**:

$$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN} \qquad (\text{Eq. 2, Eq. 3})$$

$$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2TP}{2TP + FP + FN} \qquad (\text{Eq. 4})$$

Dimana:
* TP merupakan true positive, yaitu case dimana labelnya positif dan prediksi kelasnya juga positif
* TN merupakan true negative, yaitu case dimana labelnya negatif dan prediksi kelasnya juga negatif
* FP merupakan false positive, yaitu case dimana labelnya negatif tapi prediksi kelasnya positif

F1 score penting untuk kasus dimana ada class imbalance, yaitu ketika sebagian kelas memiliki jumlah yang lebih banyak dibanding yang lainnya.

Dalam kasus benchmark LLM, F1 score memiliki penggunaan yang sedikit berbeda, yaitu untuk mengukur **kebenaran parsial** sebuah jawaban. Misalnya ketika ground truth menyebut fakta A dan B, tapi jawaban hanya menyebut fakta A, F1 score akan tetap memberikan nilai parsial. Kelemahan dari menggunakan F1 score untuk menilai kebenaran parsial adalah penilaiannya bergantung exact n-gram match. Untuk jawaban yang teksnya berbeda tapi semantiknya sama, sulit ditangkap, umumnya digunakan pendekatan lain misalnya BERTScore atau langsung menggunakan LLM-as-a-judge.
* **Digunakan pada benchmark:** DROP
* **Cocok untuk:** Klasifikasi dengan class imbalance, Menilai kebenaran parsial jawaban.

**3. Pass@K**

Metrik ini mengukur seberapa baik model bisa memberikan jawaban yang sesuai dengan suatu kriteria ketika diberikan K kali percobaan. Setiap case dianggap lolos ketika setidaknya ada satu percobaan yang berhasil. Sisanya dihitung rata-rata sama seperti akurasi.

Cara menentukan apakah suatu jawaban memenuhi kriteria bisa bermacam macam, mulai dari evaluator yang deterministik hingga menggunakan LLM.

Contoh:
* Untuk menguji kemampuan coding, setiap sampel pada benchmarknya berisi deskripsi program yang harus dibuat serta test case. Setelah LLM menghasilkan kode, kode tersebut dijalankan pada test case tersebut lalu outputnya dibandingkan apakah sesuai atau tidak.
* Untuk menguji kemampuan reasoning, sebuah LLM diberikan pertanyaan yang sangat rumit, lalu cara menjawabnya dinilai oleh LLM (tidak hanya jawaban akhirnya saja) berdasarkan kriteria tertentu.

**Digunakan pada benchmark:** Human Eval, HumanEval+
**Cocok untuk:** kasus yang rumit yang masuk akal kalau perlu mencoba beberapa kali sampai berhasil (seperti coding atau reasoning).

**4. BLEU (Bilingual Evaluation Understudy)**

Metrik ini mengukur kemampuan model untuk menerjemahkan teks dengan cara membandingkan hasil terjemahannya dengan referensi lain. Untuk melakukan perbandingan, dilakukan matching pada level n-gram antara hasil terjemahan dengan ekspektasi / label terjemahan. Berikut rumus **BLEU**:

$$\text{BLEU} = BP \times \exp \left( \sum_{n=1}^{N} w_n \log p_n \right) \qquad (\text{Eq. 5})$$

Dimana:
* $p_n$ merupakan presisi dari sebuah n-gram pada c
* $w_n$ merupakan bobot untuk setiap n-gram, biasanya uniform
* BP atau *Brevity Penalty* dihitung dengan rumus:

$$\min \left( 1, e^{(1 - r/c)} \right)$$

Dimana c adalah panjang (jumlah token) kandidat dan r adalah panjang referensi.

![Contoh perhitungan BLEU](https://lms.sdmdigital.id/pluginfile.php/635415/mod_page/content/22/image60.png)
**Gambar 33. Contoh perhitungan BLEU dengan panjang n=4**

Kelemahan BLEU adalah ketergantungannya pada tokenizer yang dipilih, serta sangat bergantung pada exact match n-gram token, sehingga kurang mampu menangkap kedekatan secara makna.
* **Digunakan pada benchmark:** MT-Bench
* **Cocok untuk:** Machine Translation

**5. ROUGE (Recall-Oriented Understudy for Gisting Evaluation)**

Metrik ini bertujuan mengukur kemampuan model merangkum teks tanpa menghilangkan informasi penting pada teks asal.

Metrik ini mirip seperti BLEU, tapi arah perbandingannya berbeda. Kalau BLEU memeriksa apakah n-gram pada c ada di r, ROUGE juga memeriksa sebaliknya, mencari apakah ada n-gram di c yang termasuk dalam r.

Salah satu varian ROUGE yang digunakan adalah ROUGE-L, yang mencari Longest Common Subsequence (LCS) antara c dan r. LCS adalah jumlah token yang sama yang muncul dengan urutan yang sama di kedua teks. LCS tidak perlu contiguous atau beriringan, boleh ada skip. Lebih mudahnya diilustrasikan sebagai berikut:

```text
Text 1: Aku melihat tetanggaku memasak mi ayam di dapur bersama
Text 2: Aku melihat ibuku memasak nasi goreng di depan rumah bersama ayah

LCS: panjang dari [Aku, melihat, memasak, di, bersama] = 5
```

Rumus berikut kemudian digunakan untuk menghitung **ROUGE-L**:

$$\text{ROUGE-L} = \frac{(1 + \beta^2) \cdot \text{Precision} \cdot \text{Recall}}{\text{Recall} + \beta^2 \cdot \text{Precision}} \qquad (\text{Eq. 6})$$

Dengan detail:

$$\text{Recall} = \frac{LCS(\text{r}, \text{c})}{|\text{r}|} \qquad (\text{Eq. 7})$$

$$\text{Precision} = \frac{LCS(\text{r}, \text{c})}{|\text{c}|} \qquad (\text{Eq. 8})$$

$$\beta = \frac{\text{Precision}}{\text{Recall}} \qquad (\text{Eq. 9})$$

Dimana:
* LCS(r, c) adalah LCS antara r dan c seperti yang dijelaskan di atas.
* |r| dan |c| adalah masing masing panjang (jumlah token) dari r dan c

Perlu dipahami bahwa sama seperti BLEU, metrik ini terlalu bergantung pada pilihan tokenizer serta tidak mampu menangkap kedekatan makna, sehingga sudah jarang digunakan pada evaluasi LLM modern.
* **Cocok untuk:** Menguji kemampuan merangkum

**6. Penilaian langsung dari LLM (LLM as a judge)**

Untuk sebagian besar task evaluasi yang lebih abstrak, langsung digunakan LLM untuk memberikan skor. Skor ini bisa binary (benar atau salah, sesuai atau tidak sesuai) maupun dalam bentuk nilai dengan sebaran tertentu, misal 0-1 atau 1-10.

**7. Win rate**

Metrik ini dihitung dengan memberikan sebuah kriteria serta referensi output. Untuk setiap jawaban LLM, seorang penilai, baik manusia atau mungkin LLM, menentukan mana yang lebih “baik” antara jawaban LLM dengan referensi. Win rate kemudian diukur dengan cara menghitung rasio antara jawaban LLM yang lebih baik ini dengan seluruh test case dalam benchmark yang digunakan.
* **Cocok untuk:** Metrik yang mengukur alignment dengan suatu kriteria tertentu.

Berbagai macam metrik evaluasi ini serta detail cara menghitungnya bisa dipelajari lebih lanjut pada modul 12 (Evaluasi).
# 1.3 Tahapan Pelatihan LLM: Pretraining (2)

---

## 1.3.3 Menentukan baseline

Setelah mengetahui benchmark apa saja yang ingin digunakan dan metrik apa saja yang ingin diukur, perlu ditentukan sebuah baseline. Baseline ini bisa berasal dari berbagai sumber. Misalnya kita memutuskan untuk melatih LLM setelah gagal mencapai target performa yang diinginkan menggunakan instruksi dan konteks saja. Kita bisa menggunakan nilai performa tertinggi yang sebelumnya dicapai untuk menentukan apakah model yang kita latih lebih baik dari model yang sebelumnya sudah dipakai. Dalam skenario lain, misalnya kita ingin menguji teknik melatih baru, dataset baru, atau arsitektur yang lebih baru. Dalam kasus ini, yang kita bandingkan adalah model lainnya yang sejenis atau sepadan. Atau mungkin apabila perubahan yang kita buat membangun dari suatu model yang sudah ada, kita bisa menggunakan model tersebut dan performanya di evaluasi yang kita inginkan sebagai baseline.

**Bagaimana caranya kita tahu pasti apakah model yang kita latih bisa mencapai target yang diinginkan?**
Jawaban simpelnya adalah tidak ada yang tahu pasti, semuanya perlu eksperimen. Tidak seperti melatih model Machine Learning atau Deep Learning dengan skala yang lebih kecil, training LLM adalah proses iteratif yang membutuhkan pengawasan sepanjang prosesnya untuk mengidentifikasi masalah yang muncul dan intervensi yang tepat untuk mengatasinya, bukan hanya proses pasif yang hanya perlu ditunggu hasilnya setelah menjalankan kode training.

Salah satu miskonsepsi yang perlu diluruskan dari melatih LLM adalah proses training tidak akan berjalan secara linear sekedar menentukan tujuan -> menentukan arsitektur -> mengumpulkan data -> training -> dapatkan model hasil. Proses training sangat iteratif, terutama untuk skala sebesar LLM. Bila melihat catatan pengembangan maupun jurnal ilmiah yang mendetilkan proses melatih LLM, akan diamati bahwa terdapat proses yang iteratif yang berulang kali perlu diuji dengan melakukan suatu eksperimen, karena bisa jadi jawabannya sulit diketahui dengan pasti.

![Ilustrasi realita dari mengembangkan sebuah model dari nol](https://lms.sdmdigital.id/pluginfile.php/635416/mod_page/content/21/image40.jpg?time=1768457542182)
**Gambar 34. Ilustrasi realita dari mengembangkan sebuah model dari nol. Prosesnya tidak linear, tapi perlu iterasi**

Dalam proses di atas, eksperimen mungkin terkesan terpisah dengan proses training. Akan tetapi ini tidak hanya berlaku di awal training saja sebelum menentukan konfigurasi training yang tepat, eksperimen ini juga dilakukan di tengah training ketika ditemukan hambatan (misalnya, performa model berhenti meningkat) dan perlu menentukan keputusan selanjutnya. Keputusan ini bisa diambil dengan melakukan eksperimen terlebih dahulu.

Sebagai contoh, dalam melatih SmolLM3, tim HuggingFace melakukan banyak sekali eksperimen untuk menguji atau membandingkan beberapa pilihan dalam pengembangan model. Eksperimen ini biasanya disebut ablation. Eksperimen yang mereka lakukan meliputi:
*   Melakukan eksperimen untuk menentukan rasio mixture data yang digunakan pada setiap tahap training
*   Melakukan eksperimen untuk memilih arsitektur yang digunakan serta memvalidasi apakah pilihan-pilihan teknis dalam membuat suatu arsitektur memberikan peningkatan yang berarti
*   Mengecek strategi mixture data yang lebih efektif ketika peningkatan performa menjadi stagnan

Selain melakukan eksperimen di awal, mereka juga melakukan checkpointing secara berkala, artinya menyimpan parameter yang sudah dilatih pada titik-titik tertentu untuk dijadikan backup apabila ada sesuatu yang terjadi yang memerlukan konfigurasi ulang dan mengulang training dari titik tertentu. Salah satu contohnya adalah ketika melakukan pretraining, ketika mereka melihat performa pada suatu domain sudah stagnan atau sulit meningkat lagi, mereka menghentikan training, melakukan eksperimen untuk menentukan mixture dataset yang lebih baik, lalu melanjutkan training lagi. Hal ini mereka ulangi untuk keseluruhan 3 tahap training mereka. Salah satu mindset yang penting dalam melakukan iterasi adalah bisa identifikasi sinyal-sinyal yang bisa menunjukkan kalau pilihan yang dibuat adalah tepat, entah menggunakan metrik saja atau melakukan perbandingan dengan pilihan lainnya atau pilihan sebelumnya.

Realitanya adalah compute yang harus disiapkan dan waktu yang dihabiskan selama mengembangkan LLM tidak hanya untuk melakukan training sampai selesai mendapatkan model. Akan ditemui berbagai masalah selama pengembangan yang mungkin cara mencari solusinya adalah melakukan ablation untuk menentukan jawaban yang paling tepat secara empiris. Sebagai ilustrasi, dalam pengembangan SmolLM3, tim HuggingFace melaporkan bahwa compute budget yang mereka habiskan untuk melakukan berbagai eksperimen ini mencapai lebih dari setengah compute yang mereka butuhkan untuk training.

**Tabel 6. Detil penggunaan compute dalam melatih model SmolLM3**

| Keperluan | Jumlah GPU (unit) | Durasi (Hari) | Total Compute (GPU-hours) |
| :--- | :--- | :--- | :--- |
| Untuk Pretraining | 384 | 30 | 276,480 |
| Ablation selama pretraining | 192 | 15 | 69,120 |
| Ablation selama mid-training | 192 | 10 | 46,080 |
| Kebutuhan debugging | 384/192 | 3/4 | 46,080 |
| **Total** | **-** | **-** | **437,760** |

GPU-hours merupakan satuan untuk menggambarkan penggunaan komputasi yang diperlukan. Jika sebuah GPU bekerja 100% selama 1 jam, maka dihitung 1 GPU-hour.

---

## 1.3.4 Menyiapkan Infrastruktur untuk Training

**Menentukan arsitektur model serta jumlah datanya**

Kecuali memang tujuannya adalah menguji performa model pada suatu arsitektur yang baru, sebaiknya awali dengan memilih base architecture yang berasal dari model-model populer pada ukuran parameter yang dituju. Tahapan ini akan mempermudah proses selanjutnya, karena umumnya sudah sangat diketahui nilai-nilai hyperparameter yang bekerja untuk arsitektur tersebut, serta bisa diperkirakan kebutuhan computenya untuk dicocokkan dengan compute budget yang tersedia. Selain itu, arsitektur yang populer dan umum juga pasti sudah banyak didukung bermacam framework sehingga proses setup untuk melakukan training menjadi jauh lebih mudah. Memilih arsitektur yang sudah umum digunakan juga bisa memberikan gambaran baseline ekspektasi performa tertentu.

Kalau memang perlu ada eksperimen perubahan arsitektur, perlu menyiapkan compute dan waktu ekstra untuk keperluan ablation untuk validasi setiap pilihan perubahan arsitektur.

**Menentukan Optimizer dan Hyperparameter: batch size, learning rate, dan learning rate scheduling**

**Optimizer**
Dalam industri maupun akademik, pilihan optimizer biasanya ditentukan secara empiris saja, mengikuti mana yang umumnya bekerja dengan baik untuk berbagai kebutuhan. Umumnya optimizer yang paling banyak digunakan adalah AdamW.

AdamW merupakan sebuah upgrade dari Adam, yang merupakan salah satu famili optimizer yang adaptive dimana untuk setiap parameter, ada dua nilai yang dicatat:
*   Momentum, untuk melakukan smoothing gradien agar tidak terlalu noisy
*   Adaptive scaling, setiap parameter memiliki “learning rate” sendiri yang bisa berubah tergantung gradien-gradiennya. Semakin besar nilai nilai gradien sebelumnya, learning ratenya akan disesuaikan menjadi lebih kecil, begitupun sebaliknya

Selain itu AdamW juga melakukan weight decay atau L2 regularization yang berfungsi untuk memastikan suatu parameter tidak berubah secara ekstrim menjadi terlalu besar atau terlalu kecil. Hal ini penting untuk menjaga stabilitas training serta mencegah overfitting, menghasilkan performa yang lebih baik.

Sebagai konfigurasi awal, hyperparameter berikut banyak ditemukan berkerja dengan baik:

**Tabel 7. Parameter AdamW yang umum digunakan sebagai nilai awal**

| Parameter | Nilai |
| :--- | :--- |
| β1 | 0.9 |
| β2 | 0.95 |
| Grad norm untuk gradient clipping | 1.0 |
| Weight decay | 0.1 (LLAMA-3-405B menggunakan 0.01) |

**Learning Rate**
Untuk melatih LLM, dan umumnya model Deep Learning lainnya dengan skala besar, learning rate ukurannya tidak tetap. Learning rate divariasikan selama proses training dengan "jadwal" tertentu yang disebut sebagai learning rate scheduling.

Untuk sebagian besar training yang dilakukan untuk model Deep Learning, variasi learning rate ini terdiri atas dua tahap utama:
1.  Tahap "warmup" yaitu secara linear meningkatkan learning rate dari nol atau nilai yang sangat kecil sekali ke suatu nilai yang menjadi learning rate awal selama beberapa training step.
2.  Tahap "scheduling" yaitu bagaimana learning rate divariasikan hingga akhir training.

Ada bermacam pendekatan yang dilakukan pada scheduling, seperti menggunakan cosine decay, multi step atau warmup-stable-decay (WSD). Inti dari semua scheduling ini sama, yaitu semakin mendekati akhir training ketika parameter sudah semakin “matang”, learning rate dibuat lebih kecil agar update parameter yang dilakukan lebih berhati-hati agar tidak terlalu mengubah pemahaman yang dibangun oleh model terlalu banyak. Walaupun cosine decay menjadi salah satu pilihan utama untuk learning rate scheduling, ada kelemahan yang menjadi alasan digunakannya learning rate scheduler yang lebih sederhana. Cosine scheduling langsung melakukan decaying (istilah untuk penurunan nilai learning rate) dimulai setelah warmup. Decaying ini secara langsung bergantung pada durasi training, sehingga jika ada perubahan yang melibatkan durasi training seperti adanya tambahan compute yang tersedia atau perlu menambahkan data, mau tidak mau harus melatih dari awal.

Beberapa riset juga menunjukkan bahwa scheduler alternatif seperti WSD maupun Multi-step jika diatur sedemikian rupa bisa mencapai performa yang sama dengan cosine decay. Masing masing pendekatan tersebut, termasuk bagian warmupnya, disajikan dalam grafik berikut:

![Perbandingan beberapa pendekatan learning rate scheduler](https://lms.sdmdigital.id/pluginfile.php/635416/mod_page/content/21/image27.jpg?time=1768457599690)
**Gambar 35. Perbandingan beberapa pendekatan learning rate scheduler**

Untuk WSD dan Multi step, kapan kita harus mulai menurunkan learning rate? Beberapa heuristic telah ditemukan seiring semakin banyak digunakannya berbagai scheduler ini untuk menjawab pertanyaan tersebut, terutama agar bisa mencapai performa yang sama dengan menggunakan cosine:
*   Untuk WSD, mulai penurunan learning rate pada 10-20% terakhir dari durasi training (jumlah token).
*   Untuk Multi Step, secara empiris rasio 80/10/10 berkerja dengan baik untuk nilai awal. Artinya untuk 80% durasi training, gunakan learning rate pertama, lalu turun ke step selanjutnya selama 10% durasi training selanjutnya, lalu turun lagi ke step selanjutnya untuk sisa 10% durasi training.

Selanjutnya adalah bagaimana cara menemukan learning rate yang optimal? Jawabannya sama, dengan mengikuti nilai yang digunakan pada skenario yang sama sebagai referensi, atau tentukan dengan melakukan eksperimen. Dengan menggunakan model yang ingin dilatih, variasikan learning rate lalu lakukan training pada suatu sampel data untuk menemukan konfigurasi yang tepat dengan melihat apakah training bisa converge (loss menurun hingga mencapai suatu minimum) dan seberapa jauh nilai loss bisa turun.

**Batch Size**
Batch size penting untuk meningkatkan efisiensi dari proses training juga melakukan smoothing gradien agar bisa mencapai performa yang lebih optimal. Akan tetapi ada batas tertentu yang jika dilewati akan menurunkan efisiensi secara drastis, yang disebut sebagai critical batch size.

Sama seperti yang lainnya, adalah ide yang baik untuk menggunakan referensi dari proses training model sejenis untuk melihat batch size yang digunakan sebagai referensi untuk kemudian diadaptasikan untuk menemukan yang lebih optimal secara empiris.

**Memilih framework untuk implementasi model dan proses training**
Meskipun bisa langsung menggunakan Pytorch, LLM memiliki kompleksitas yang cukup tinggi dengan detail-detail tertentu yang perlu diperhatikan. Selain itu, usaha kolektif komunitas pengembang LLM menyediakan berbagai macam solusi yang memudahkan implementasi sebuah LLM, salah satunya adalah transformers milik HuggingFace. Pilihan ini menjadi default bagi sebagian besar pengembang LLM karena fungsionalitasnya yang cukup luas dan integrasinya yang mudah dengan berbagai macam framework untuk melakukan training maupun untuk melakukan inference. Selain itu jika ada arsitektur yang ingin digunakan untuk melatih LLM, kemungkinan besar implementasinya sudah tersedia dan bisa diakses menggunakan transformers. Walaupun transformers bisa digunakan untuk melakukan training dalam skala kecil, untuk skala yang lebih besar apalagi yang memerlukan distributed training, kita memerlukan framework training khusus untuk mempermudah implementasinya.

Berbagai framework tersedia dengan kelebihannya masing-masing untuk melakukan training. Framework seperti Megatron-LM, DeepSpeed, TorchTitan, Nanotron dan Accelerate menyediakan berbagai macam fitur seperti mixed precision training, konfigurasi scheduling yang mudah, serta utilitas lainnya yang membantu memudahkan implementasi proses training. Selain itu, umumnya framework yang disebut juga menyediakan fungsionalitas yang memudahkan implementasi distributed training.

Cara paling pragmatis untuk memilih framework adalah dengan melihat framework apa yang digunakan untuk melatih model sejenis, terlebih kalau ternyata arsitektur yang dijadikan baseline untuk pengembangan sudah menyediakan kode untuk melakukan training yang bisa langsung digunakan.

---

## 1.3.5 Konfigurasi Data Loader dan Persiapan Batch untuk Training

**Packing dan Attention Masking**

Salah satu kunci proses training yang efisien adalah memaksimalkan penggunaan GPU dengan memastikan selalu ada batch data yang siap diproses setiap kali model selesai melakukan pemrosesan satu batch data untuk satu training step. LLM dilatih menggunakan suatu panjang context window tertentu yang mengatur jumlah total token yang bisa menjadi input dan output. Saat pretraining LLM, context window ini diisi dengan teks yang ingin dipelajari.

Salah satu masalah yang mungkin terjadi adalah panjang setiap dokumen atau sampel teks individual tidak uniform atau seragam satu dengan lainnya. Hal ini bermasalah karena untuk melakukan training agar efektif dan efisien, perlu dilakukan batching, yaitu menggunakan lebih dari satu sampel data untuk menghitung gradien dan update parameter dalam satu waktu.

![Teks input LLM memiliki panjang yang beragam](https://lms.sdmdigital.id/pluginfile.php/635416/mod_page/content/21/image50.jpg?time=1768457638598)
**Gambar 36. Teks yang menjadi input LLM bisa memiliki panjang input yang beragam**

Batching membutuhkan panjang yang sama untuk semua input. Masalah ini mungkin bisa diselesaikan dengan menambah padding, yaitu token khusus untuk “memenuhi” context window agar ukurannya seragam sehingga sampel dengan ukuran yang berbeda bisa disatukan dalam sebuah batch.

![Padding sebagai solusi batching](https://lms.sdmdigital.id/pluginfile.php/635416/mod_page/content/21/image62.jpg?time=1768457683548)
**Gambar 37. Menggunakan padding sebgai solusi batching ketika panjang setiap intputnya berbeda**

Akan tetapi padding ini tidak memiliki kegunaan apapun selain “mengisi” ruang kosong pada context window. Hal ini diperburuk dengan beberapa sampel teks yang mungkin jauh lebih pendek dibandingkan context window. Dalam training, umum digunakan context window berukuran besar untuk mempersiapkan LLM agar bisa memproses input yang panjang. Sebagian data bisa lebih pendek dari context window ini.

Bayangkan untuk context window sebesar 4096 token, kita menggunakan sampel dengan panjang 2048 token, sisanya diisi dengan padding. Artinya setengah context window digunakan hanya untuk padding saja yang akan membuang waktu, memory dan compute.

![Pemborosan context window](https://lms.sdmdigital.id/pluginfile.php/635416/mod_page/content/21/image20.jpg?time=1768457725685)
**Gambar 38. Ketika teks dalam dataset jauh lebih kecil dari panjang context window, terjadi pemborosan**

Untuk memaksimalkan context window, maka dilakukan strategi pemrosesan input yang dinamakan **packing**. Packing memasukkan lebih dari satu dokumen ke context window, dipisahkan menggunakan sebuah token khusus, misalnya token penanda akhir dokumen, token penanda awal dokumen, bergantung strategi yang dipilih oleh pengembang model.

![Ilustrasi packing](https://lms.sdmdigital.id/pluginfile.php/635416/mod_page/content/21/image19.jpg?time=1768457767955)
**Gambar 39. Ilustrasi packing, lebih dari satu dokumen dimasukkan ke sekuens input yang sama, dipisahkan oleh token khusus.**

Packing membutuhkan perhatian khusus untuk mekanisme utama dari cara kerja LLM, yaitu attention. Untuk menentukan “bobot” attention pada setiap token yang berada di dalam sekuens input, mekanisme ini “memperhatikan” bagian lainnya dari input yang informasinya penting untuk dipakai menentukan output. Kita mengetahui kalau mekanisme ini pada intinya mencoba menghitung sebuah attention map yang memetakan pembobotan setiap token pada sebuah sekuens terhadap suatu token pada sekuens lainnya yang direpresentasikan sebagai sebuah matriks. Karena transformers menggunakan self-attention, maka attention map untuk setiap sequence dihitung terhadap dirinya sendiri.

Untuk LLM modern yang sebagian besar adalah autoregressive, attention ini bersifat kausal, artinya pembobotan untuk setiap token hanya bisa “memperhatikan” posisi token yang sama serta token lain pada posisi sebelumnya. Attention dapat dipaksa untuk hanya memperhatikan token-token tertentu menggunakan sebuah attention mask yang berbentuk sebuah matriks dengan dimensi yang sama dengan attention map.

Attention map direpresentasikan sebagai matriks dimana nilai pada setiap baris dan kolom memetakan bobot attention sebuah token dalam posisi yang diwakilkan indeks baris terhadap token lainnya pada posisi yang diwakilkan indeks kolom. Sebagai contoh, pada matriks berikut, kolom ke-2 dan baris ke-3 memiliki nilai 0.12. Artinya, attention untuk input "makan" mamberi pembobotan sebesar 0.12 untuk input "suka".

![Attention map](https://lms.sdmdigital.id/pluginfile.php/635416/mod_page/content/21/image49.jpg?time=1767328326915)
**Gambar 40. Attention map untuk semua token yang sedang diproses.**

Untuk causal attention, setiap token hanya memperhatikan token dalam posisi yang sama serta token-token pada posisi sebelumnya. Untuk itu dilakukan masking. Attention mask berfungsi untuk membatasi token mana saja yang boleh “diperhatikan” ketika menghitung token dalam suatu posisi. Setiap nilai pada attention mask hanya berupa true atau false. Jika nilainya false, maka nilai di posisi yang sama pada matriks attention map menjadi nol. Setelah diberikan attention mask, maka attention map di atas menjadi seperti berikut.

![Causal attention map](https://lms.sdmdigital.id/pluginfile.php/635416/mod_page/content/21/image47.jpg?time=1767328428369)
**Gambar 41. Casual attention map membatasi agar setiap posisi output hanya memperhatikan input pada posisi yang sama serta posisi-posisi sebelumnya.**

Jika kita memasukkan banyak dokumen, misalnya kita memasukkan dua dokumen dalam satu sekuens input dengan topik “sejarah peradaban manusia di Indonesia” dan dokumen selanjutnya dengan topik “resep bubur kampiun”, kita perlu “mengajarkan” LLM bahwa dua dokumen ini tidak berhubungan. Artinya ketika sedang memproses bagian dokumen kedua, attention map pada setiap posisi di dokumen kedua hanya diperbolehkan “memperhatikan” token yang berada pada posisi sebelumnya di dokumen yang sama. Lebih mudahnya bisa dilihat melalui ilustrasi berikut:

![Ilustrasi relevansi attention untuk packing](https://lms.sdmdigital.id/pluginfile.php/635416/mod_page/content/21/image4.jpg?time=1768457830297)
**Gambar 42. Ilustrasi relevansi attention untuk dokumen yang di-packing**

Bagaimana caranya kita memastikan LLM hanya memperhatikan token sebelumnya pada dokumen yang sama ketika menghitung attention for suatu sekuens input? Kita bisa mengubah attention mask agar semua attention antar dokumen bernilai nol, yang artinya tidak ada hubungan attention antar dokumen. Hal ini disebut sebagai document masking. Mengacu pada ilustrasi sebelumnya, attention mask yang causal dan juga spesifik setiap dokumen akan berbentuk sebagai berikut:

![Attention per dokumen](https://lms.sdmdigital.id/pluginfile.php/635416/mod_page/content/21/image1.jpg?time=1767328563002)
**Gambar 43. Attention untuk setiap posisi hanya dihitung pada dokumen yang sama**

Cara membacanya seperti ini: Karena dokumen dipisahkan dengan `<EOS>` pada akhir dokumen, artinya attention untuk suatu dokumen hanya bisa dihitung dari awal dokumen saja, yaitu bagian input yang muncul setelah ada token `<EOS>`.

Selain menggunakan masking, LLM juga bisa mempelajari sendiri perilaku ini, terutama karena sudah ada token spesial yang menjadi pemisah antar dokumen. Beberapa riset juga memberikan konfirmasi bahwa menggunakan document masking lebih berpengaruh pada context window yang lebih panjang. Meskipun begitu, daripada melakukan eksperimen tambahan untuk mengecek dulu apakah perlu atau tidak menggunakan document masking, tidak ada salahnya tetap digunakan. Untungnya anda tidak terlalu perlu memikirkan implementasi packing sendiri, karena bisa dilakukan menggunakan **Trainer** pada library **transformers** atau pada **SFTTrainer** pada library **trl**. Dua snippet berikut menunjukkan bagaimana packing bisa digunakan tanpa perlu repot implementasi sendiri.

Pada Trainer, secara otomatis anda bisa menggunakan packing dengan menggunakan **DataCollatorWithFlattening** sebagai collator yang digunakan untuk memanggil **Trainer**:

```python
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    DataCollatorWithFlattening,
    Trainer,
    TrainingArguments,
)


model = AutoModelForCausalLM.from_pretrained("NAMA_MODEL_YANG_DIPAKAI")


# contoh saja, yang penting mengembalikan object dataset
train_dataset = load_dataset("json", data_files="path/ke/dataset/anda")["train"]
data_collator = DataCollatorWithFlattening()


train_args = TrainingArguments(output_dir="/save/path")
trainer = Trainer(
    args=train_args,
    model=model,
    train_dataset=train_dataset,
    data_collator=data_collator,
)
trainer.train()
```

Pada **SFTTrainer**, anda bisa menggunakan packing hanya dengan memberikan nilai **True** pada parameter **padding_free** ketika inisiasi **DataCollatorForCompletionOnlyLM**:

```python
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer


dataset = load_dataset("/path/ke/dataset/anda", split="train")


model = AutoModelForCausalLM.from_pretrained(
    "MODEL_YANG_ANDA_GUNAKAN"
)
tokenizer = AutoTokenizer.from_pretrained("MODEL_YANG_ANDA_GUNAKAN")
tokenizer.pad_token = tokenizer.eos_token


# hanya contoh saja, ini fungsi untuk
# menerapkan sebuah chat template
# biasanya setiap model memiliki
# chat template sendiri
def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example["instruction"])):
        text = f"### Question: {example['instruction'][i]}\n ### Answer: {example['output'][i]}"
        output_texts.append(text)
    return output_texts




response_template = " ### Answer:"
response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[
    2:
]
collator = DataCollatorForCompletionOnlyLM(
    response_template_ids, tokenizer=tokenizer, padding_free=True
)


trainer = SFTTrainer(
    model,
    train_dataset=dataset,
    args=SFTConfig(
        output_dir="./tmp", gradient_checkpointing=True, per_device_train_batch_size=8 # hanya contoh saja
    ),
    formatting_func=formatting_prompts_func,
    data_collator=collator,
)

trainer.train()
```
# 1.3 Tahapan Pelatihan LLM: Pretraining (3)

---

## 1.3.6 Konfigurasi data loader

Setelah menetapkan strategi packing yang tepat untuk memanfaatkan context window sebaik mungkin, semua persiapan sudah dilakukan karena pipeline dari data teks menjadi batch yang terdiri dari sekuens-sekuens teks input yang sudah di-packing sudah ditentukan. Dari sini bisa dilanjutkan dengan menetapkan konfigurasi data loader.

**Bagaimana penyimpanan datanya agar efisien?**

Kita ingin meminimalkan segala pemrosesan agar ketika data di-load, data langsung bisa dipakai. Untuk itu, ada beberapa hal yang perlu dilakukan sebelum bahkan data disimpan:
*   Proses tokenization dan menambahkan special token
*   Proses packing
*   Penerapan chat template untuk SFT

Dan semua ini perlu dilakukan secara offline, alias dilakukan bukan saat training, tapi bahkan sebelum training.

Agar data tidak perlu diproses lagi, idealnya data yang disimpan sudah dalam bentuk batch berisi token yang sudah diproses dari teks dan sudah di-packing lengkap dengan special token yang diperlukan. Data yang sudah siap dikonsumsi LLM ini selanjutnya perlu disimpan pada lokasi yang tepat dengan format yang memudahkan pembacaan secara efisien.

Idealnya, data ini disimpan secara lokal pada media penyimpanan yang mendukung pembacaan cepat seperti SSD NVME langsung pada kluster-kluster komputasi yang digunakan untuk training. Untuk dataset yang ukurannya sangat besar, umumnya dipilih cloud storage seperti AWS S3 untuk menyimpan data ini. Secara berkala dataloader akan menyiapkan dan men-download sekian jumlah data yang dibutuhkan ke penyimpanan lokal untuk memastikan selalu ada data baru yang siap dikonsumsi proses training tanpa perlu menunggu waktu lama hingga tersedia.

Pilihan format untuk penyimpanan data ini juga beragam dan mungkin bergantung pada framework yang digunakan. Berikut adalah dua format yang cukup populer yang bisa menjadi pilihan untuk opsi penyimpanan dataset yang siap untuk training:

**Tabel 8. Beberapa pilihan format data siap pakai untuk di-load DataLoader**

| Format | Kelebihan | Kelemahan | Catatan |
| :--- | :--- | :--- | :--- |
| **Megatron IndexedDataset** | Format paling efisien dan cepat untuk training LLM. | Data perlu sudah diproses dahulu (tokenization, packing) | Lebih cocok untuk data yang benar-benar siap training. Bisa digunakan di luar framework megatron |
| **HuggingFace datasets** | Langsung terintegrasi dengan tokenizers, sangat fleksibel untuk bermacam penggunaan | Lebih lambat dari megatron untuk skala training yang sangat besar | Sangat baik untuk mempersiapkan data, bisa digunakan bersama format megatron |
| **Format binary custom** | - | Perlu dibuat sendiri + perlu membuat infrastruktur | - |

Banyak format lain yang bisa ditemukan, tapi kedua format di atas sudah cukup untuk sebagian besar use case, menggabungkan fleksibilitas datasets untuk pemrosesan serta kecepatan megatron untuk training serius apabila skala trainingnya cukup besar.

**Bagaimana strategi untuk menangani kebutuhan data parallelism?**
Sharding. Sharding memecah sebuah dataset menjadi beberapa bagian yang disimpan pada mesin yang berbeda. Untuk memenuhi kebutuhan data parallelism, sharding dilakukan untuk membagi dataset menjadi bagian-bagian terpisah untuk setiap worker. Jumlah worker sendiri ditambahkan sebanyak mungkin hingga diamati bahwa sudah tidak ditemukan bottleneck pada I/O. Pada industri, jumlahnya beragam. Setiap GPU bisa memiliki 2-16 worker yang didedikasikan menyediakan data untuk setiap GPU, tergantung konfigurasi dan skala dari training.

**Contoh implementasi pemrosesan offline dan dataloader**
Di bawah ini adalah contoh dua proses: yaitu pemrosesan dataset secara offline menjadi bentuk yang:
- Sudah dalam bentuk token
- Sudah di-packing
- Sudah dibagi menjadi panjang sequence input yang digunakan (diasumsikan 2048)

```python
from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader


# ===============================================
# Bagian script yang ini jalannya offline
# sebelum training
# ===============================================
# pake dataset random aja, ambil dikit buat demonstrasi
dataset = load_dataset("id_newspapers_2018", split="train[:1000]")
# bisa juga pake tokenizer yang sudah dilatih
# tinggal ganti parameternya ke path tokenizernya
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")


# Qwen3 menggunakan tokenizer qwen2, yang
# tidak ada pad_token. Mereka menyarankan
# diganti EOS aja
tokenizer.pad_token = tokenizer.eos_token
EOS = tokenizer.eos_token


# tokenization, menambahkan EOS,
# menandai start dokumen
# (untuk bikin attention mask nanti)
def tokenize(batch):
    texts = [t + EOS for t in batch["content"]]

    out = tokenizer(texts, add_special_tokens=False)
   
    # buat bikin mask dokumen
    doc_starts = []
    for ids in out["input_ids"]:
        # setiap awal dokumen, nilainya 1
        mask = [1] + [0] * (len(ids) - 1)
        doc_starts.append(mask)


    out["doc_start_mask"] = doc_starts
    return out

tokenized = dataset.map(
    tokenize,
    batched=True,
    remove_columns=dataset.column_names
)



# pake pendekatan paling sederhana buat packing
# semuanya habis tokenize diratain (flatten)
# terus langsung dipotong-potong
# mengikuti batch size
all_ids = []
all_doc_starts = []


for ids, starts in zip(tokenized["input_ids"], tokenized["doc_start_mask"]):
    all_ids.extend(ids)
    all_doc_starts.extend(starts)


all_ids = torch.tensor(all_ids, dtype=torch.long)
all_doc_starts = torch.tensor(all_doc_starts, dtype=torch.long)




# ini sequence length / panjang input yang mau dipake
# sesuaikan aja
SEQ_LEN = 2048


num_chunks = len(all_ids) // SEQ_LEN


# disini, gak diproses sequence terakhir
# yang panjangnya mungkin kurang dari SEQ_LEN
all_ids = all_ids[: num_chunks * SEQ_LEN]
all_doc_starts = all_doc_starts[: num_chunks * SEQ_LEN]


# reshape sebelum disimpan
packed_ids = all_ids.view(num_chunks, SEQ_LEN)
packed_doc_starts = all_doc_starts.view(num_chunks, SEQ_LEN)


# buat ngebatasin attention
# tapi karena gak make padding, jadi 1 semua
# ditulis aja dulu tapi
packed_attention = torch.ones_like(packed_ids, dtype=torch.long)


# simpan
# idealnya menggunakan format lain
# silahkan dieksplorasi sendiri :)
torch.save({
    "input_ids": packed_ids,
    "attention_mask": packed_attention,
    "doc_start_mask": packed_doc_starts
}, "data.pt")


print("Packed:", packed_ids.shape)


# ===============================================
# Di bawah ini idealnya di script terpisah, disimpan
# di sini buat demonstrasi aja
# ===============================================

# implementasi dataset sederhana
class PackedDataset(Dataset):
    def __init__(self, save_path="data.pt"):
        data = torch.load(save_path)
        self.ids = data["input_ids"]
        self.att = data["attention_mask"]
        self.doc = data["doc_start_mask"]

    def __len__(self):
        return self.ids.size(0)

    def __getitem__(self, idx):
        return {
            "input_ids": self.ids[idx],
            "attention_mask": self.att[idx],
            "doc_start_mask": self.doc[idx],
            "labels": self.ids[idx].clone(),
        }

dataset_packed = PackedDataset("data.pt")
# di sini bisa atur batch size dan setting lainnya
# terutama kalau perlu paralel
# tapi kalau mau bisa load paralel,
# ketika menyimpan data, dipecah, jangan 1 file saja
# batch disesuaikan saja, ini cuma contoh
loader = DataLoader(dataset_packed, batch_size=2, shuffle=True)

# ---------------------------------------------
# Check batch
# ---------------------------------------------
batch = next(iter(loader))

print("Shapes:")
print("  input_ids:", batch["input_ids"].shape)
print("  attention_mask:", batch["attention_mask"].shape)
print("  doc_start_mask:", batch["doc_start_mask"].shape)

decoded = tokenizer.batch_decode(batch["input_ids"])
print("Hasil decode:")
print(decoded[0])
print("Penanda awal dokumen (sample 50 pertama saja):")
print(batch["doc_start_mask"][0][:50])  # 50 pertama saja
```

Kode di atas sangat sederhana, dan ada banyak hal yang bisa dikembangkan.
*   Metode menyusun packing yang lebih baik agar lebih optimal, misalnya [BFD](https://www.amazon.science/blog/improving-llm-pretraining-with-better-data-organization).
*   Menyimpan format dalam bentuk lain yang lebih optimal yang lebih cepat dibaca, misalnya numpy atau format binary custom

Silahkan eksplorasi mandiri untuk mencari pengembangan yang lebih baik, sangat disarankan eksplorasi repository [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) untuk mempelajari bagaimana mereka melakukan implementasi dataloader dan pemrosesan data secara offline untuk disiapkan.
Untuk training dengan skala kecil, mungkin tidak perlu melakukan ini, tapi tetap penting untuk diketahui.

---

## 1.3.7 Evaluasi, monitoring dan checkpointing

Sebelumnya sudah dijelaskan bahwa training untuk LLM merupakan proses yang aktif dan iteratif. Proses ini memiliki tujuan untuk mengawasi dan memastikan beberapa hal berikut:
*   Model bisa mencapai convergence ke suatu titik minimum lokal maupun global. Pengertian sederhananya adalah nilai loss terus menurun hingga mencapai stagnasi di nilai yang diinginkan / sesuai ekspektasi.
*   Model mencapai tingkatan performa yang diinginkan
*   Throughput dari training bisa ditingkatkan hingga ke tingkatan yang optimal. Artinya, kita perlu mengatur konfigurasi training agar bisa memaksimalkan penggunaan hardware.

Dalam melakukan iterasi training, diperlukan beberapa fungsionalitas untuk diimplementasikan. Pertama, harus ada pipeline evaluasi untuk menghitung metrik performa model secara berkala. Kedua, diperlukan sebuah sumber informasi terpusat yang mengumpulkan semua informasi yang penting untuk diawasi selama training, termasuk metrik evaluasi. Sumber informasi ini berbentuk sistem untuk melakukan monitoring, agar memudahkan pengembang melihat kinerja training dari waktu ke waktu. Selain itu, juga diperlukan checkpointing berkala, yang sudah dijelaskan sebelumnya.

Evaluasi menghitung metrik yang akan dikumpulkan oleh sistem monitoring bersamaan dengan nilai lainnya. Kemudian, apabila diamati ada yang salah, proses training dihentikan dulu untuk diperbaiki / mengatur ulang konfigurasi model, optimizer, data, dan lainnya yang berhubungan dengan training. Setelah konfigurasi sudah ditentukan, training dilanjutkan kembali menggunakan checkpoint yang sebelumnya sudah disimpan dengan konfigurasi baru.

---

## 1.3.8 Sepanjang proses training

**Memisahkan training menjadi beberapa fase**

Sebelumnya sudah dipelajari bahwa umumnya melatih LLM dibagi menjadi beberapa fasa, bukan sekali jalan saja dengan konstan. Setiap fase ini memiliki tujuan masing masing dan umumnya memiliki mixture data yang berbeda dengan pengaturan training yang berbeda. Ada beberapa patokan yang menjadi pemisahan fase ini:

**Kemampuan general vs kemampuan spesifik**
Kemampuan berbahasa secara umum akan menjadi fokus utama tahapan awal training, sedangkan kemampuan spesifik, misalnya kemampuan multilingual, kemampuan coding, matematika, reasoning, dan kemampuan lainnya yang bisa dilatih melalui pretraining yang spesifik suatu domain tertentu akan difokuskan pada tahap selanjutnya dari training.

Pada awal training, biasanya mixture data memiliki rasio yang lebih banyak mengandung dataset untuk kemampuan berbahasa secara umum (untuk sebagian besar LLM, maksudnya web data dalam bahasa Inggris). Rasio dataset untuk domain khusus ditingkatkan pada tahap-tahap berikutnya, terutama pada fase annealing atau fase akhir pada learning rate scheduling dimana learning rate diturunkan hingga ke nilai minimalnya.

**Context Length**
Sebagian besar waktu training dilakukan menggunakan suatu context length yang rendah di bawah target, misalnya 4096, lalu secara bertahap ditingkatkan hingga akhir training. Kenapa tidak dari awal menggunakan context length maksimum yang ditargetkan? Karena mekanisme attention sangatlah computationally expensive karena meningkat secara kuadratik bersamaan dengan peningkatan context length. Selain itu, sebuah [riset](https://arxiv.org/abs/2410.02660) menemukan bahwa meningkatkan context length pada akhir training maupun saat melakukan continued pretraining dapat memberikan performa yang bagus untuk context length yang panjang.

Mungkin muncul pertanyaan. Kalau positional encoding sudah ditentukan dari awal, bagaimana caranya menambah context length secara dinamis? Jawabannya terletak pada positional encoding yang digunakan. LLM modern biasanya menggunakan RoPE (rotational positional encoding) yang memetakan setiap posisi input pada sebuah lingkaran, atau bahkan beberapa model menggunakan NoPE (no positional encoding) yang mempelajari informasi posisi ini tanpa informasi posisi eksplisit, sepenuhnya bergantung pada kemampuan model untuk mempelajari hubungan ini secara implisit. Pendekatan hybrid seperti RNoPE mulai digunakan untuk menggabungkan kelebihan RoPE dan NoPE serta saling menutupi kelemahan masing-masing.

**Kualitas dataset**
Selain dataset umum yang berasal dari internet, ada kalanya juga ditemukan dataset dengan kualitas tinggi yang baik kurasi, proses pengumpulan maupun komposisi dan kontennya sudah diatur secara hati hati untuk mengajarkan kemampuan spesifik tertentu. Selain variasi rasio dataset setiap domain dalam sebuah mixture, juga divariasikan jumlah data berkualitas tinggi yang digunakan. Tahapan awal training yang masih mempelajari kemampuan yang umum belum terlalu memerlukan dataset dengan kualitas yang lebih tinggi, baru pada saat fase selanjutnya mulai ditingkatkan rasio data berkualitas tinggi dalam setiap domain ketika training sudah berjalan dan sudah terlihat bahwa performa pada suatu domain sudah berhenti meningkat / stagnan.

---

## 1.3.9 Keputusan apa saja yang perlu diambil selama training

Selama training, diperlukan intervensi pada titik-titik tertentu yang dicapai yang diamati melalui monitoring, baik itu dari sisi metrik performa model, nilai loss atau mungkin keanehan yang diamati pada hardware utilization. Intervensi hanya perlu dilakukan ketika diamati ada yang bermasalah, misalnya loss malah meningkat, atau performa model sudah stagnan, atau mungkin teramati ada anomali melalui monitoring, misal tiba tiba terlihat throughput / kecepatan training menurun secara drastis, atau anomali lainnya misalnya pada tahap training yang sama, performa yang dicapai tidak sesuai ekspektasi bila dibandingkan dengan baseline.

Penting untuk menerapkan suatu sistem alerting yang otomatis menangkap anomali-anomali ini untuk memperingatkan para pengembang model untuk melakukan intervensi. Selain itu, melakukan evaluasi berkala hanya berdasarkan vibes atau secara manual mengamati pola-pola output LLM pada titik-titik tertentu juga bisa memberikan insight yang mungkin tidak terpikirkan sebelumnya, misalnya ternyata perlu menambah metrik yang diawasi melalui monitoring.

Training melibatkan banyak sekali komponen, mulai dari arsitektur, komponen yang berhubungan dengan training secara langsung seperti optimizer, learning rate scheduler, gradient clipping hingga infrastruktur seperti implementasi distributed training yang digunakan yang semuanya dapat menjadi point of failure, sehingga selain memastikan semua komponen ini berjalan with baik sebelum melakukan training dengan melakukan eksperimen, monitoring juga perlu bisa menangkap sinyal-sinyal yang penting untuk melakukan diagnosa sumber masalah ketika terjadi sesuatu saat training.

---

## 1.3.10 Evaluasi hasil pretraining

Evaluasi dilakukan terakhir dan umumnya bisa menggunakan pipeline yang sama yang sudah digunakan selama training. Evaluasi ini dilakukan pada test split dari dataset sebagai laporan terakhir kinerja dari sebuah model. Setelah melalui proses yang panjang dan melelahkan, harapannya adalah hasil training memenuhi ekspektasi kinerja yang diharapkan. Apabila evaluasi akhir tidak memenuhi harapan performa yang bisa dicapai, bisa dilakukan iterasi melalui checkpoint yang sebelumnya sudah disimpan.

Hasil akhir proses pretraining adalah sebuah model dan laporan performanya bila dibandingkan dengan baseline maupun model lain, misalnya yang jumlah parameternya sama atau versi sebelumnya dari model yang dikembangkan yang siap digunakan untuk proses selanjutnya, entah menjadi base model untuk melakukan continued pretraining maupun melanjutkan ke post-training.

---

## 1.3.11 Contoh kode training sederhana

Kode di bawah ini memberikan contoh bagaimana melakukan pretraining menggunakan Trainer milik Huggingface. Perlu diingat karena keterbatasan resource, sulit memberikan kode yang benar-benar menghasilkan model yang baik. Anda bisa melihat nilai loss dan perplexitynya semakin lama semakin menurun yang menandakan setidaknya proses training bisa berjalan.

Kode ini mengasumsikan ada file data.pt yang dihasilkan proses offline di bab dataloader di atas. Setiap bagian kode ini sudah dikomen untuk menjelaskan kegunaannya, sehingga bisa dipahami langsung saja dari kode tanpa perlu dijelaskan lebih lanjut.

```python
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorWithFlattening,
    Trainer,
    TrainingArguments,
)
import torch


MODEL_NAME = "Qwen/Qwen3-0.6B"


# load dataset. Datasetnya tidak punya val split, jadi kita split sendiri dari trainnya
# ini cuma contoh, idealnya perlu dishuffle dulu
train_dataset = load_dataset("id_newspapers_2018", split="train[:20000]")
val_dataset = load_dataset("id_newspapers_2018", split="train[20000:25000]")


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.model_max_length = 512  




def tokenize(batch):
    return tokenizer(
        batch["content"],
        truncation=True,
        max_length=512,
        padding="max_length",
    )


# data sudah berbentuk token. Idealnya ini dilakukan terpisah
# anda sudah ditunjukkan cara membuat dataloader terpisah
# silahkan disambungkan sendiri


train_tokenized = train_dataset.map(
    tokenize,
    batched=True,
    remove_columns=["content"],
)


val_tokenized = val_dataset.map(
    tokenize,
    batched=True,
    remove_columns=["content"],
)


collator = DataCollatorWithFlattening()


model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    attn_implementation="eager",
    torch_dtype="bfloat16",
)


# ganti ukuran context window
# tujuannya agar hemat VRAM
model.config.max_position_embeddings = 512
model.config.max_length = 512


# metrik sederhana saja, loss dan ppl
# idealnya ini dilakukan terpisah
# agar tidak mengganggu proses training
# anda juga perlu menggunakan benchmark lain
# yang lebih mencerminkan berbagai dimensi
# kemampuan model
def compute_metrics(eval_pred):
    loss = eval_pred.loss
    try:
        ppl = torch.exp(loss)
    except OverflowError:
        ppl = float("inf")
    return {"eval_loss": loss, "perplexity": ppl}




# batch size dan eval batch size perlu disesuaikan
# digunakan nilai yang rendah karena keterbatasan
# resource
args = TrainingArguments(
    output_dir="./qwen3-0.6b-pretrained",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    bf16=True,
    num_train_epochs=1,
    # agar anda bisa cepat melihat saja
    # idealnya step sebelum perlu eval
    # dan menyimpan checkpoint lebih lama
    logging_steps=10,
    save_steps=20,
    eval_steps=20,
    # untuk monitoring, misal anda punya akses ke layanan seperti wandb atau tensorboard terpisah,
    # bisa diganti jadi `tensorboard` atau `wandb`. Selengkapnya bisa langsung cek dokumentasi
    # TrainingArguments untuk melihat metode lain
    report_to="none"
)


# training
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    data_collator=collator,
    compute_metrics=compute_metrics
)


trainer.train()
```

### Mini Quiz

*(H5P Placeholder untuk Multiple Choice Quiz)*

> **Refleksi Singkat:**
>
> Walaupun tidak banyak organisasi yang berbagi secara detail bagaimana caranya mereka melatih model mereka, Huggingface memberikan detail yang lebih mendalam mengenai bagaimana mereka melatih model SmolLM3 dalam [artikel](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook) yang mereka rilis.
>
> Sebagai eksplorasi tambahan, coba anda baca artikel tersebut untuk lebih memahami bagaimana realistisnya proses training berjalan, termasuk memahami seberapa besar usaha dan resource yang diperlukan.