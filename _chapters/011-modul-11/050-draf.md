---
slug: modul-11-5
title: modul-11-5
---
Berikut adalah isi konten yang telah dirapikan ke dalam format Markdown sesuai dengan hirarki aslinya:

# 7.5 Evaluasi Agent

---

Evaluasi *workflow* maupun *agent* tidak hanya berfokus pada kualitas keluaran bahasa, tetapi juga kemampuan *Agent* dalam melakukan *reasoning*, perencanaan, dan pemanfaatan *tool* secara efektif untuk mencapai tujuan yang kompleks.

Untuk menyederhanakan penyebutan, baik *workflow* maupun *agent* akan disebut sebagai *agent* dalam bagian ini, karena sudah dipelajari mengenai masalah terminologi ini di awal modul.

## Kategori Metrik Utama Evaluasi

Metrik evaluasi yang digunakan terbagi menjadi tiga kategori: **Eksekusi**, **Kualitas**, dan **Efisiensi**.

### 1. Metrik Eksekusi
Metrik ini mengukur kemampuan *Agent* untuk berhasil menyelesaikan sebuah *task*.

**Tabel 3. Metrik yang menilai kemampuan *agent* eksekusi sebuah *task***

| Metrik | Definisi |
| :--- | :--- |
| **Success Rate (tingkat keberhasilan)** | Apakah *agent* berhasil melaksanakan *task* dengan baik. Dapat berupa skor **biner** (berhasil/gagal) atau skor **parsial** (berapa persen tercapai). |
| **Tool Calling Accuracy (akurasi pemanggilan tool)** | Apakah *agent* berhasil memilih *tool* yang benar / sesuai untuk menyelesaikan suatu masalah bila dibandingkan dengan referensi. |
| **Tool Parameter Accuracy (akurasi parameter)** | Apakah *agent* dengan benar mengisi parameter yang sesuai untuk memanggil sebuah *tool*. |
| **Plan Adherence (seberapa sesuai aksi yang dilakukan dengan rencana awal)** | Menilai seberapa konsisten *Agent* mengikuti urutan aksi yang telah direncanakan sendiri. |

### 2. Metrik Kualitas
Metrik ini mengukur seberapa baik jawaban akhir sebuah *agent*. Metrik ini tidak berbeda jauh dengan kebanyakan metrik lainnya yang digunakan untuk mengukur kualitas jawaban akhir LLM.

**Tabel 4. Metrik yang menilai kualitas jawaban akhir *agent***

| Metrik | Definisi |
| :--- | :--- |
| **Akurasi jawaban** | Menentukan validitas jawaban melalui *validator* eksternal maupun dibandingkan dengan sebuah referensi. |
| **Relevansi** | Menilai apakah jawaban akhir serta langkah yang dilakukan oleh LLM koheren dan relevan dengan permintaan / deskripsi *task*. |
| **Groundedness** | Apakah jawaban akhir didasarkan pada informasi yang diberikan ke *agent*. |

### 3. Metrik Efisiensi (Bisnis dan operasional)
Metrik *non-functional* yang penting dari sisi bisnis maupun operasional, terutama untuk sistem yang berjalan di lingkungan *production*.

**Tabel 5. Metrik non fungsional yang menilai aspek efisiensi dari sebuah *agent***

| Metrik | Definisi |
| :--- | :--- |
| **Latency** | Waktu yang diperlukan *agent* untuk menyelesaikan sebuah *task*. |
| **Step Efficiency** | Dibandingkan referensi jumlah langkah yang lebih optimal, seberapa boros langkah yang diambil oleh *agent*. |
| **Cost** | Biaya dalam satuan mata uang maupun jumlah *token* yang diperlukan *agent* untuk menyelesaikan sebuah *task*. |

## Metodologi Pengukuran

Metode evaluasi utama terbagi menjadi tiga pendekatan:

1.  **Evaluasi secara programatis:**
    Output *agent* secara langsung divalidasi baik secara langsung dibandingkan dengan *ground truth* maupun diberikan ke validator eksternal.

    **Contoh:**
    *   Metrik seperti penggunaan token dan *latency* bisa langsung diukur secara programatis.
    *   Untuk *agent* yang ditugaskan menulis kode, kode buatannya bisa langsung dievaluasi menggunakan *unit test*.
    *   Jika *agent* diminta menghasilkan SQL *query* untuk mendapatkan data tertentu, bisa dinilai langsung apakah hasilnya sesuai.

    **Cocok untuk:**
    *   Output yang berstruktur / bukan berbentuk teks bebas, misalnya output berbentuk JSON dengan skema tertentu.
    *   Output yang bisa divalidasi dengan validator deterministik, misalnya kode atau SQL *query*.
    *   Menghitung metrik non-fungsional yang secara langsung bisa diukur seperti *latency* atau *cost*.

2.  **LLM-as-a-Judge**
    Pendekatan ini menggunakan LLM untuk melakukan evaluasi. Biasanya digunakan LLM yang cukup *powerful* lalu diberikan instruksi untuk memberikan penilaian dalam beberapa bentuk, entah ***binary*** (benar / salah) atau berbentuk ***skor*** dalam *range* tertentu, misalnya 0-1, 1-10, dsb.

    **Contoh:**
    *   Ketika *agent* diminta menuliskan laporan, digunakan LLM untuk membandingkan laporan buatan *agent* dengan referensi berdasarkan bermacam kriteria yang abstrak, misalnya kelengkapan topik, apakah inti laporan tersebut sama dengan referensi, dsb.
    *   LLM diminta menilai *trace* atau alur eksekusi sebuah *workflow* untuk menyelesaikan sebuah *task* untuk menentukan apakah setiap langkahnya masuk akal dan apakah hasil akhirnya relevan.

    **Cocok untuk:**
    *   Menilai kebenaran semantik maupun kriteria abstrak lainnya seperti relevansi, koherensi, dan kualitas perencanaan.
    *   Membandingkan output maupun alur eksekusi *agent* dengan referensi.

3.  **Evaluasi Manusia (Human Evaluation):**
    Metode ini dilakukan secara manual oleh evaluator manusia (biasanya *domain expert*) yang menilai jawaban akhir sebuah *agent* serta proses yang dilakukan suatu *agent* untuk menyelesaikan suatu *task*. Sama seperti yang lainnya, penilaian bisa *binary* maupun berbentuk skor.

    Selain memberi penilaian, saat memberi evaluasi, seorang penilai juga bisa menuliskan alasan / *reasoning* dibalik penilaiannya yang bisa dimanfaatkan sebagai referensi di masa depan untuk melakukan penilaian menggunakan LLM.

    **Contoh:**
    *   Pengguna aplikasi bisa menilai langsung apakah output dari *agent* menyelesaikan *task* dengan benar atau tidak menggunakan tombol ***thumbs up*** atau ***thumbs down*** pada UI aplikasi.
    *   Ketika membuat *agent* untuk asisten belanja, *domain expert* seperti asisten toko bisa menilai apakah langkah-langkah yang dilakukan *agent* adalah *valid* dan sesuai SOP.

    **Cocok untuk:**
    *   Evaluasi subjektif yang butuh perhatian khusus / sangat sensitif, misalnya evaluasi bias.
    *   Evaluasi yang sangat spesifik *domain*, misalnya jawaban akhir serta langkah yang diambil *agent* yang melakukan diagnosa medis perlu diperiksa langsung oleh tenaga medis.

### Mini Quiz
*(Konten Interaktif H5P: Multiple Choice)*

> **Refleksi Singkat**
>
> Untuk menilai validitas sebuah agent atau workflow, diperlukan informasi mengenai keseluruhan langkah yang dilakukan, bukan hanya input dan outputnya saja. Menurut anda, seperti apa idealnya kebutuhan ini dipenuhi?
>
> Misalnya anda diminta membangun sistem agentic atau sebuah workflow. Infrastruktur seperti apa yang perlu anda persiapkan agar setiap langkah yang diambil sebuah agent maupun setiap tahap eksekusi workflow yang dikembangkan bisa diamati dan dinilai?

*(Catatan: Tidak ditemukan URL gambar dalam konten asli bagian 7.5 ini untuk ditampilkan).*