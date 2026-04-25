---
slug: modul-12-1
title: modul-12-1
---
Berikut adalah isi konten yang telah dirapikan ke dalam format Markdown sesuai hirarki tanpa mengurangi isinya:

# 8.1 Tantangan dalam Evaluasi Generatif: Mengapa Akurasi Saja Tidak Cukup?

Dalam pengembangan perangkat lunak konvensional atau *Machine Learning* klasik (seperti klasifikasi gambar atau prediksi harga rumah), evaluasi adalah proses yang biner dan deterministik. Jika fungsi `add (2, 2)` menghasilkan 5, itu adalah *Bug*. Jika model klasifikasi memprediksi foto Kucing sebagai Anjing, itu adalah *Error*. Kita memiliki satu kebenaran mutlak yang disebut *Ground Truth*.

Namun, dalam dunia *Generative AI*, kita menghadapi fenomena yang disebut Ruang Output Tak Terbatas (*Infinite Output Space*). Jika pengguna bertanya: **"Jelaskan konsep fisika kuantum kepada anak berusia 5 tahun,"** tidak ada satu jawaban tunggal yang "benar".

*   Jawaban A mungkin menggunakan analogi bola yang mudah dimengerti.
*   Jawaban B mungkin menggunakan analogi air yang puitis.
*   Jawaban C mungkin sangat akurat secara teknis tetapi terlalu rumit untuk anak kecil.

Tantangan fundamental yang dihadapi oleh seorang *AI Engineer* adalah: **Subjektivitas**. Bagaimana kita mengubah penilaian kualitatif ("Jawaban ini terasa lebih pas") menjadi metrik kuantitatif ("Skor kualitas jawaban ini 8.5/10") yang dapat diukur secara otomatis pada ribuan percakapan? Di subtopik ini, kita akan membedah mengapa metrik lama kita tidak lagi relevan dan paradigma baru apa yang harus diadopsi.

## 8.1.1 Karakteristik Evaluasi: Deterministik vs. Stokastik

Perbedaan mendasar pertama terletak pada sifat kebenaran itu sendiri.

**1. Evaluasi Deterministik (Software/ML Tradisional):**
*   Bergantung pada Pencocokan String Eksak (*Exact String Match*).
*   Contoh: Dalam ekstraksi data (Modul 2.6) atau *Text-to-SQL*, output model harus presisi.
*   Jika model menghasilkan query SQL `SELEC *` (typo), maka sistem gagal total. Di sini, metrik klasik seperti Akurasi (*Accuracy*) masih berlaku.

**2. Evaluasi Stokastik (Generative AI):**
*   Bergantung pada Kesetaraan Makna Semantik (*Semantic Equivalence*).
*   Karena sifat probabilistik dari proses *sampling* (Modul 3.2), bahkan dengan *prompt* yang sama, LLM dapat menghasilkan variasi kalimat yang berbeda setiap kali dijalankan.
*   Kita tidak bisa menuntut model menghasilkan kata demi kata yang sama persis dengan referensi, karena itu akan membatasi fleksibilitas dan kreativitasnya.
*   Tantangannya adalah mengukur apakah dua kalimat yang secara leksikal (kata-kata) berbeda, sebenarnya memiliki makna yang sama.

Agar lebih mudah memahami, di bawah ini adalah ilustrasi mengenai Evaluasi.

![Visualisasi Evaluasi](https://lms.sdmdigital.id/pluginfile.php/635503/mod_page/content/7/image18.jpg?time=1766723392055)
**Gambar 1. Visualisasi Evaluasi**

Diagram ini menunjukkan pergeseran kebutuhan evaluasi. Di sisi kiri, kita bisa menggunakan fungsi `==` sederhana. Di tengah, kita mencoba menghitung kesamaan kata, namun sering gagal menangkap konteks (posisi kata "kucing" mengubah makna total). Di sisi kanan, untuk tugas kreatif, kita memerlukan "juri" yang cerdas untuk memahami nuansa bahwa sebuah penjelasan akademis dan sebuah puisi bisa jadi membahas konsep yang sama dengan gaya berbeda.

Memahami bahwa kita tidak bisa lagi mengandalkan pencocokan eksak, komunitas peneliti AI awalnya mencoba mengadaptasi metrik dari era sebelum *Deep Learning*. Mereka mencoba mengukur kualitas dengan menghitung seberapa banyak kata dalam output model yang tumpang tindih dengan jawaban kunci (*Reference*) yang dibuat manusia. Inilah era metrik berbasis N-gram.

## 8.1.2 Evolusi Metrik Berbasis Referensi (Reference-Based Metrics)

Sebelum masuk ke metrik spesifik, kita harus membedakan konteks evaluasi ini dengan apa yang telah kita pelajari di modul sebelumnya.

**Konteks: Bedanya Evaluasi di Sini dengan Evaluasi SFT/CPT?**
*   **Evaluasi saat Training (Modul 2.1 - 2.3):** Menggunakan metrik intrinsik seperti *Training Loss* atau *Perplexity*. Tujuannya adalah memantau apakah model sedang "belajar" (konvergen) secara matematis. Ini dilakukan di dalam proses *backpropagation*.
*   **Evaluasi Aplikasi (Modul 2.8):** Menggunakan metrik eksternal (seperti BLEU, ROUGE, atau Juri AI). Tujuannya adalah mengukur apakah model yang sudah jadi (*frozen*) mampu menyelesaikan tugas bisnis dengan benar sesuai standar manusia. Kita memperlakukan model sebagai "kotak hitam" (*black box*).

Jika kita memiliki kunci jawaban (*Reference* atau *Ground Truth*), berikut adalah cara kita menilainya menggunakan metrik klasik:

**1. Era N-Gram (Lexical Overlap): BLEU & ROUGE**
Metrik ini menghitung tumpang tindih kata antara output model dan referensi. Meskipun tua, mereka masih menjadi standar emas untuk tugas-tugas tertentu.

*   **BLEU (Bilingual Evaluation Understudy):**
    *   **Fokus: Presisi.** Mengukur seberapa banyak kata di output model yang benar-benar ada di referensi.
    *   **Kapan Digunakan?** Standar untuk **Translasi Mesin**. Dalam translasi, kita tidak ingin model menambah-nambahkan kata yang tidak ada di sumber, jadi presisi sangat penting.
*   **ROUGE (Recall-Oriented Understudy for Gisting Evaluation):**
    *   **Fokus: Recall (Kelengkapan).** Mengukur seberapa banyak informasi penting (kata) di referensi yang *berhasil ditangkap* oleh model.
    *   **Kapan Digunakan?** Standar untuk **Peringkasan Teks (*Summarization*)**. Dalam merangkum, yang terpenting adalah tidak ada poin kunci dari teks asli yang tertinggal.

**Kelemahan Fatal (Untuk Chatbot/Reasoning):** Kedua metrik ini buta makna. Mereka hanya melihat bentuk kata, bukan arti.
*   Referensi: "Saya suka kopi."
*   Model: "Saya benci kopi."
*   Skor BLEU: Sangat Tinggi (2/3 kata sama persis). Padahal maknanya berlawanan total.

Oleh karena itu, untuk tugas generatif yang lebih luwes (seperti *chatbot* atau penulisan kreatif), kita membutuhkan pendekatan yang memahami semantik.

**2. Era Embedding (Semantic Overlap): BERTScore**
Ini adalah evolusi penting yang memanfaatkan konsep *Embedding* (Modul 2.3). Daripada mencocokkan kata mentah ("mobil" != "kendaraan"), BERTScore mengubah kalimat Referensi dan Output Model menjadi Vektor *Embedding* kontekstual menggunakan model seperti BERT. Kemudian, ia menghitung *Cosine Similarity* (kemiripan arah vektor) antara kedua kalimat tersebut. Hasilnya: Model tahu bahwa "mobil" dan "kendaraan" memiliki vektor yang berdekatan, sehingga memberikan skor tinggi meskipun katanya berbeda. Ini jauh lebih akurat daripada BLEU untuk tugas *chatbot*.

Meskipun BERTScore adalah peningkatan masif, semua metrik di atas memiliki satu syarat mutlak: **Anda harus memiliki Kunci Jawaban (Referensi)**. Namun, dalam produksi nyata, pengguna menanyakan hal-hal baru setiap hari yang tidak ada di database kunci jawaban kita. Bagaimana kita menilai kualitas jawaban model ketika kita tidak memiliki referensi kebenaran sama sekali? Kita memasuki wilayah evaluasi tanpa referensi.

## 8.1.3 Metrik Tanpa Referensi (Reference-Less) & Tantangannya

Ketika *Ground Truth* absen, kita harus mengandalkan sinyal internal dari model itu sendiri atau properti statistik dari teks yang dihasilkan.

**1. Perplexity (PPL): Mengukur "Keyakinan", Bukan "Kebenaran"**
Perplexity adalah metrik intrinsik yang mengukur seberapa "bingung" atau "terkejut" model terhadap sebuah teks. PPL rendah berarti model sangat yakin bahwa kalimat tersebut alami dan gramatikal.
*   **Masalah (Confident Hallucination):** Model bisa sangat yakin (PPL rendah) saat berbohong. Kalimat "Bumi itu datar" bisa memiliki PPL yang sangat rendah (dianggap kalimat bagus) oleh model yang mengalami halusinasi, meskipun faktanya salah. Jadi, PPL hanya mengukur kelancaran bahasa (*fluency*), bukan kebenaran fakta.

**2. Length Bias (Bias Panjang)**
Ini adalah tantangan psikologis dalam evaluasi, baik oleh manusia maupun mesin otomatis.
*   Riset menunjukkan bahwa evaluator seringkali bias memberikan skor lebih tinggi pada jawaban yang lebih panjang dan bertele-tele, menganggapnya "lebih informatif", padahal mungkin isinya hanya pengulangan (*redundancy*).
*   Evaluasi modern harus melakukan normalisasi terhadap panjang respons agar tidak terjebak menilai "yapping" (omong kosong panjang) sebagai kualitas tinggi.

Keterbatasan metrik statistik (BLEU/BERTScore) dalam menangkap nuansa, dan ketidakmampuan metrik intrinsik (PPL) dalam memverifikasi fakta, memaksa kita mencari standar baru. Kita membutuhkan ujian yang terstandarisasi dan teruji untuk mengukur "kecerdasan" model secara objektif. Inilah yang membawa kita pada pembahasan mengenai **Benchmark Akademis**.

---

### Mini Quiz
*(Konten interaktif tersedia pada platform LMS)*

---

> **Refleksi Singkat**
>
> Kita telah membahas bahwa dalam *Generative AI*, "Kebenaran" itu cair. Bayangkan Anda sedang mempresentasikan laporan performa *chatbot* Customer Service kepada CEO. Jika Anda menggunakan metrik tradisional (*Exact Match*), skor akurasi *chatbot* Anda mungkin hanya 5% (karena *chatbot* jarang menjawab persis sama kata-per-katanya dengan skrip lama). Padahal secara kualitatif, *chatbot* itu menjawab dengan benar dan sopan. Bagaimana Anda menjelaskan fenomena "Skor Rendah tapi Kualitas Tinggi" ini kepada manajemen yang terbiasa dengan angka pasti? Apakah Anda akan tetap memaksakan metrik lama, atau berani mengajukan metrik baru (seperti *Semantic Similarity*) yang angkanya lebih abstrak?