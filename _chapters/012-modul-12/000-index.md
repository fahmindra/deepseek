---
slug: modul-12
layout: part
---

Berikut adalah isi konten yang telah dirapikan ke dalam format Markdown sesuai hirarki tanpa mengurangi isinya:

---

# Modul 8 Teknik Evaluasi Untuk Aplikasi Berbasiskan LLM

![Logo Modul 8](https://lms.sdmdigital.id/pluginfile.php/635502/mod_page/content/13/_64882c72-3761-4f81-ac43-047c5b8cdec1-removebg-preview.png)
*Visualisasi Modul 8 Teknik Evaluasi*

Selamat datang di Modul 8!

Mari kita tarik napas sejenak dan melihat kembali perjalanan epik kita di Program Intermediate ini. Kita telah bertransformasi dari sekadar *user* menjadi *architect* sistem AI yang kompleks:

### Fondasi & Adaptasi Model:

*   **Di Modul 1 Level Intermediate**, kita memahami siklus hidup pelatihan (*Pre-training* hingga *Alignment*).
*   **Di Modul 2 Level Intermediate**, kita menyuntikkan pengetahuan domain baru lewat *Continued Pretraining* (CPT).
*   **Di Modul 3 Level Intermediate**, kita mengasah keterampilan spesifik model secara efisien menggunakan *Fine-Tuning* (PEFT/LoRA).

### Arsitektur & Sistem Aplikasi:

*   **Di Modul 4 Level Intermediate**, kita belajar merancang arsitektur sistem yang *production-ready*, menyeimbangkan *latency*, biaya, dan strategi *caching*.
*   **Di Modul 5 Level Intermediate**, kita mengatasi keterbatasan pengetahuan model dengan membangun sistem RAG (*Retrieval-Augmented Generation*) yang menghubungkan LLM ke data perusahaan Anda.
*   **Di Modul 6 Level Intermediate**, kita mendisiplinkan model agar menghasilkan output terstruktur (JSON) yang aman bagi sistem *backend*.
*   **Di Modul 7 Level Intermediate**, kita menaikkan level kecerdasan dengan membangun *Agentic LLM* yang mampu berpikir (*reasoning*) dan menggunakan alat (*tool use*) secara otonom.

---

Kini, Anda memiliki sistem yang canggih: sebuah Agen RAG yang bisa mencari data, berpikir, dan memanggil API. Namun, Bos Anda datang dengan satu pertanyaan sederhana yang mematikan: **"Sistem canggih ini seberapa akurat? Apakah lebih aman dari versi kemarin? Mana buktinya?"**

Di dunia *software* biasa, kita punya *Unit Test*. Tapi di dunia *Generative AI*, di mana jawabannya cair dan kreatif, bagaimana kita mengukurnya?

*   Jika Agen RAG menjawab dengan benar tapi gayanya kasar, apakah itu sukses?
*   Jika Agen menjawab sopan tapi faktanya salah (halusinasi), bagaimana kita mendeteksinya secara otomatis pada 1.000 percakapan?

Modul ini didedikasikan untuk **Evaluasi Sistem LLM**. Kita akan bergerak dari metrik kaku (BLEU/ROUGE) menuju paradigma modern **LLM-as-a-Judge**. Kita akan belajar merancang **Golden Set**, menggunakan *framework* evaluasi otomatis **(RAGAS, DeepEval)**, dan membangun *dashboard* kualitas yang dapat dipertanggungjawabkan secara data.

---

### Refleksi Peran Krusial Dokumentasi dan Komunikasi

Sebelum kita mempelajari modul ini, alangkah baiknya kita merefleksikan peran krusial dari dokumentasi dan komunikasi ini sebagai berikut:

1.  Jika Anda mengubah *prompt* sistem chatbot Anda agar "lebih sopan", bagaimana Anda membuktikan secara data bahwa perubahan itu tidak malah membuat model jadi "lebih bodoh" dalam menjawab fakta teknis (*regression testing*)?
2.  Mengapa metrik tradisional seperti mencocokkan kata per kata (*String Matching*) seringkali memberikan nilai "0" atau "gagal" untuk jawaban LLM yang sebenarnya sangat bagus, kreatif, dan benar secara makna?
3.  Jika kita menggunakan LLM (seperti GPT-4) sebagai "Juri" untuk menilai kinerja LLM lain, bukankah itu bias?
4.  Bagaimana kita bisa mempercayai penilaian mesin terhadap mesin lain?
5.  Apa bedanya mengevaluasi model dasar (*Base Model*) dengan mengevaluasi aplikasi RAG (*Retrieval-Augmented Generation*)?
6.  Komponen mana yang sebenarnya salah jika jawaban chatbot ngawur: dokumen yang diambil (*Retriever*) atau kemampuan menjawab (*Generator*)?

Digiers, sebelum memulai pembelajaran, mari kita pahami bersama capaian yang akan kita raih. Berikut adalah tujuan pembelajaran yang akan menjadi kompas kita:

![Capaian Pembelajaran](https://lms.sdmdigital.id/pluginfile.php/635502/mod_page/content/13/Screenshot%202025-12-26%20at%2011.03.37.png)
*Tujuan Pembelajaran Modul 8*