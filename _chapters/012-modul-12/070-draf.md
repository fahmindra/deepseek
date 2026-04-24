---
slug: modul-12-7
title: modul-12-7
---
Berikut adalah isi konten yang telah dirapikan ke dalam format Markdown sesuai hirarki tanpa mengurangi isinya:

# Perancangan Framework Evaluasi Sistem RAG

**Judul Praktikum:** Strategi Audit Kualitas AI: Dari Golden Set hingga Automated Grading

**Tujuan Praktikum:** Peserta mampu merancang dataset pengujian (Golden Set) yang komprehensif, memilih metrik evaluasi yang relevan untuk sistem RAG, dan menyusun rubrik penilaian untuk LLM-as-a-Judge.

**Deskripsi Singkat Aktivitas:** Anda berperan sebagai Lead AI Engineer untuk proyek "Chatbot HRD Pintar" yang menggunakan RAG. Chatbot ini menjawab pertanyaan karyawan tentang kebijakan cuti dan asuransi. Manajemen tidak akan mengizinkan deployment sampai Anda bisa membuktikan secara data bahwa chatbot ini aman dan akurat. Tugas Anda adalah merancang protokol pengujiannya.

---

## Langkah-Langkah Praktikum

### 1. Siapkan Dokumen Laporan
Buat file dokumen: "Laporan Evaluasi Modul 2.8 - [Nama Anda]".

### 2. Desain Golden Set (Konsep 8.3)
Anda perlu membuat sampel Test Set untuk Chatbot HRD. Buatlah tabel (atau daftar) yang berisi 3 Pasang Data Uji yang mewakili kategori berikut:
*   **Kasus 1 (Happy Path):** Pertanyaan umum tentang cuti. (Tuliskan Input User & Kunci Jawaban Ideal).
*   **Kasus 2 (Edge Case):** Pertanyaan yang kompleks atau situasional (misal: cuti melahirkan bagi karyawan kontrak).
*   **Kasus 3 (Safety/Adversarial):** Pertanyaan jebakan yang mencoba memanipulasi bot (misal: minta menaikkan gaji sendiri).

### 3. Perancangan Rubrik LLM-as-a-Judge (Konsep 8.4)
Anda memutuskan menggunakan GPT-4 sebagai juri. Tuliskan rancangan System Prompt (Rubrik Penilaian) yang akan Anda berikan kepada Juri tersebut. Rubrik harus mencakup:
*   Definisi skor 1 (Buruk) hingga 5 (Sempurna).
*   Instruksi khusus untuk menangani "Halusinasi" (Hukuman berat jika jawaban bertentangan dengan dokumen kebijakan).

### 4. Analisis Metrik RAGAS (Konsep 8.6)
*   **Skenario:** Hasil evaluasi otomatis menunjukkan skor Faithfulness: 0.9 (Tinggi) tetapi Answer Relevance: 0.4 (Rendah).
*   **Analisis:** Apa arti kombinasi skor ini? Apa yang salah dengan perilaku chatbot tersebut? (Apakah dia berhalusinasi? Atau dia jujur tapi tidak menjawab pertanyaan?).
*   **Rekomendasi:** Berdasarkan analisis di atas, komponen mana yang harus diperbaiki tim developer? (Prompt Generator-nya? Atau Dokumen Retriever-nya?).

### 5. Simpan & Kumpulkan
Simpan laporan dalam format .pdf atau .docx.

---

## Tools & Resource yang Dibutuhkan

Untuk menyelesaikan tugas ini sebagai Lead AI Engineer, Anda memerlukan:
1.  **Akses Materi Modul 8:** Khususnya panduan tentang distribusi kategori Golden Set (8.3), teknik Prompting untuk Juri (8.4), dan definisi metrik RAGAS (8.6).
2.  **Aplikasi Pengolah Kata:** Microsoft Word, Google Docs, atau Notion untuk menyusun laporan strategi.
3.  **Akses ke LLM (Opsional tapi Disarankan):** Akses ke ChatGPT/Claude untuk melakukan simulasi "uji coba" rubrik penilaian Anda (apakah prompt Juri yang Anda buat benar-benar menghasilkan output JSON yang valid?).
4.  **Template Dokumen Kebijakan (Opsional):** Jika Anda ingin membuat Golden Set yang realistis, Anda bisa mencari contoh "Kebijakan Cuti Karyawan" di internet sebagai referensi Ground Truth.

---

## Contoh Output atau Hasil Akhir

Luaran yang diharapkan adalah satu file dokumen laporan (PDF atau DOCX) dengan struktur sebagai berikut:

1.  **Bab 1: Desain Golden Set**
    Tabel berisi 3 sampel data uji (Happy Path, Edge Case, Adversarial).
    *   Kolom wajib: User Input, Ground Truth (Jawaban Ideal), dan Kategori.
2.  **Bab 2: Rubrik Penilaian Otomatis**
    Copy-paste rancangan System Prompt yang akan Anda gunakan untuk LLM-as-a-Judge.
    Penjelasan singkat mengapa Anda memilih kriteria tersebut (misal: mengapa Anda menghukum berat halusinasi).
3.  **Bab 3: Analisis Diagnostik**
    Paragraf analisis mengenai skenario skor RAGAS (Faithfulness tinggi vs Relevance rendah).
    Diagnosis akar masalah (misal: "Retriever mengambil dokumen yang benar, tapi Generator gagal mengolahnya menjadi jawaban yang to-the-point").
    Rekomendasi perbaikan konkret bagi tim developer.

---

## Kriteria Keberhasilan / Penilaian Mandiri

*   **Kualitas Golden Set:** Sampel data uji mencakup variasi skenario (mudah, sulit, bahaya) dan memiliki Ground Truth yang jelas.
*   **Desain Rubrik:** Prompt penilaian jelas, objektif, dan secara eksplisit menangani kriteria kegagalan utama (halusinasi).
*   **Analisis Metrik:** Mampu menginterpretasikan makna skor RAGAS (hubungan antar metrik) dengan benar dan memberikan rekomendasi perbaikan teknis yang logis.
*   **Pemahaman Konsep:** Laporan menunjukkan pemahaman tentang perbedaan evaluasi deterministik dan generatif.

---

## Petunjuk Troubleshooting

*   **Bingung membuat Edge Case?** Pikirkan situasi "abu-abu" di kantor. Contoh: "Kalau saya sakit pas cuti, cutinya hangus gak?"
*   **Bingung analisis Faithfulness vs Relevance?**
    *   Faithfulness Tinggi: Jawaban sesuai dokumen (jujur).
    *   Relevance Rendah: Jawaban tidak nyambung dengan pertanyaan.
    *   **Kesimpulan:** Bot mungkin menjawab "Kebijakan cuti ada di pasal 5" (Jujur, tapi tidak menjawab pertanyaan "Berapa sisa cuti saya?").

**Durasi Praktik Disarankan:** 60 Menit.

---

## Aksi Lanjutan atau Refleksi (Opsional)

Setelah merancang sistem evaluasi ini, posisikan diri Anda 6 bulan ke depan. Aplikasi chatbot HRD ini telah live dan melayani 10.000 karyawan. Kebijakan perusahaan tentang "Cuti Hamil" tiba-tiba berubah total.

*   Apa yang terjadi pada Golden Set lama Anda? (Apakah masih valid?)
*   Apa yang terjadi pada skor evaluasi model jika Anda tidak memperbarui Ground Truth?
*   **Refleksi:** Bagaimana Anda akan membangun sistem "Continuous Evaluation" (CI/CD for AI) agar Test Set Anda selalu sinkron dengan dokumen kebijakan terbaru tanpa harus mengedit ribuan baris data secara manual setiap minggu?

---

[HANDS-ON: Automated Evaluation dengan RAGAS](https://github.com/Modul-LLM/Module-12-2.8-Intermediate)

---

# Penutup Modul 8 Teknik Evaluasi untuk Aplikasi Berbasiskan LLM

Kita telah sampai di penghujung Program Intermediate. Jika modul-modul sebelumnya mengajarkan kita cara "membangun" dan "melatih" otak buatan, Modul 8 mengajarkan kita cara menjadi **"Guru"** dan **"Auditor"** yang kritis bagi otak tersebut.

Kita belajar bahwa dalam dunia Generative AI, "kebenaran" tidak lagi biner (0 atau 1), melainkan sebuah spektrum kualitas yang bernuansa.

Berikut adalah intisari perjalanan kita di modul ini:

*   **Pergeseran Paradigma (8.1):** Kita meninggalkan metrik kaku berbasis pencocokan kata (*Exact Match, BLEU, ROUGE*) yang buta makna, dan beralih ke evaluasi berbasis **Semantik** dan **Model-Based** yang lebih selaras dengan persepsi manusia.
*   **Standar Pengujian (8.2 & 8.3):** Kita membedakan antara "Ujian Nasional" (Benchmark Akademis seperti MMLU/GSM8K) untuk memilih model dasar, dan "Ujian Sekolah" (**Golden Set**) yang wajib kita buat sendiri untuk menguji aplikasi spesifik bisnis kita.
*   **Otomatisasi Penilaian (8.4):** Kita menerapkan pola desain **LLM-as-a-Judge**, menggunakan model cerdas (GPT-4) untuk menilai model lain secara skala besar, lengkap dengan rubrik dan alasan (*reasoning*).
*   **Audit Sang Juri (8.5):** Kita menyadari bahwa Juri AI pun tidak sempurna. Kita mengidentifikasi bias kognitif mereka (*Verbosity, Position Bias*) dan cara memitigasinya secara statistik (*Swapping, Agreement Score*).
*   **Framework Modern (8.6):** Terakhir, kita menggunakan *tools* standar industri seperti **RAGAS** untuk membedah performa sistem RAG menjadi tiga komponen terukur (**RAG Triad**: *Faithfulness, Relevance, Precision*), memindahkan kita dari "tebak-tebakan" menjadi **data-driven engineering**.

Dengan menyelesaikan materi dalam modul ini, Anda telah berhasil mencapai tujuan pembelajaran berikut:

*   Anda kini dapat **merancang strategi evaluasi** yang komprehensif, menggabungkan metrik kuantitatif otomatis dan kualitatif manusia.
*   Anda mampu **membangun Golden Set** yang **robust**, mencakup kasus normal (*Happy Path*) hingga kasus berbahaya (*Adversarial*).
*   Anda dapat **mengimplementasikan pipeline evaluasi otomatis** menggunakan *LLM-as-a-Judge* dan RAGAS untuk mendeteksi halusinasi sebelum pengguna menemukannya.
*   Anda memiliki pemahaman untuk **mengaudit bias** pada sistem evaluasi itu sendiri, memastikan skor yang dihasilkan adil dan reliabel.

---

> **Refleksi Akhir**
>
> "You can't improve what you can't measure." (Anda tidak bisa memperbaiki apa yang tidak bisa Anda ukur).
> Kutipan manajemen klasik ini adalah mantra utama bagi seorang *AI Engineer*.
> Tanpa evaluasi yang solid, proses *Fine-Tuning* (Modul 3) atau *Prompt Engineering* (Modul 6) hanyalah perjudian.
> Anda tidak akan pernah tahu apakah perubahan yang Anda buat meningkatkan kecerdasan model atau justru membuatnya "lupa" hal lain (*regression*).
> Sekarang, Anda memiliki alat ukurnya. Pertanyaannya adalah: Apakah Anda berani menghadapi skor merah (*fail*) pada *dashboard* evaluasi Anda hari ini, demi membangun sistem yang jauh lebih tangguh dan aman untuk pengguna di masa depan?

---

# Daftar Pustaka

1.  Zheng, L., Chiang, W. L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., ... & Stoica, I. (2023). Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in neural information processing systems, 36, 46595-46623. [https://proceedings.neurips.cc/paper_files/paper/2023/file/91f18a1287b398d378ef22505bf41832-Paper-Datasets_and_Benchmarks.pdf](https://proceedings.neurips.cc/paper_files/paper/2023/file/91f18a1287b398d378ef22505bf41832-Paper-Datasets_and_Benchmarks.pdf)
2.  Es, S., James, J., Anke, L. E., & Schockaert, S. (2024, March). Ragas: Automated evaluation of retrieval augmented generation. In Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics: System Demonstrations (pp. 150-158). [https://aclanthology.org/2024.eacl-demo.16.pdf](https://aclanthology.org/2024.eacl-demo.16.pdf)
3.  Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M., Song, D., & Steinhardt, J. (2020). Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300. [https://arxiv.org/pdf/2009.03300](https://arxiv.org/pdf/2009.03300)
4.  Ragas. (n.d.). Ragas documentation – stable (Diakses 18 Desember 2025), dari [https://docs.ragas.io/en/stable/](https://docs.ragas.io/en/stable/)
5.  Lin, S., Hilton, J., & Evans, O. (2022, May). Truthfulqa: Measuring how models mimic human falsehoods. In Proceedings of the 60th annual meeting of the association for computational linguistics (volume 1: long papers) (pp. 3214-3252). [https://aclanthology.org/2022.acl-long.229.pdf](https://aclanthology.org/2022.acl-long.229.pdf)