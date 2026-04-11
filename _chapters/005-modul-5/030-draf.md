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