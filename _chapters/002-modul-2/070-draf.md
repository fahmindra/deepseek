---
slug: modul-2-7
title: modul-2-7
---

### Analisis Konseptual: Komponen Inti dan Arsitektur LLM
---

**Tujuan Praktikum:**
Peserta mampu menganalisis dan menjelaskan fondasi teknis LLM, mencakup evolusi arsitektur (RNN vs Transformer), komponen pra-proses (Tokenization, Embeddings), dan perbedaan tiga keluarga arsitektur Transformer (Encoder, Decoder, Encoder-Decoder).

**Deskripsi Singkat Aktivitas:**
Aktivitas ini meminta Anda bertindak sebagai AI Engineer yang menjelaskan cara kerja pipeline pemrosesan LLM secara konseptual kepada rekan setim. Anda akan menganalisis contoh kalimat dan skenario untuk mengilustrasikan pemahaman Anda tentang fondasi teknis LLM, mulai dari bagaimana teks diproses hingga bagaimana arsitektur yang tepat dipilih untuk sebuah tugas.

### Langkah-Langkah Praktikum

1.  **Siapkan Dokumen Jawaban:**
    *   Buat sebuah file teks (misalnya menggunakan Notepad, Google Docs, atau editor Markdown) untuk menuliskan jawaban Anda.
    *   Beri judul seperti "Unjuk Kerja Modul 2 - [Nama Anda]".

2.  **Analisis Evolusi Arsitektur (Konsep 2.4: RNN vs Attention)**
    *   Baca kalimat [referensi](https://lms.sdmdigital.id/mod/page/view.php?id=29797): *"Pria yang mengenakan kemeja biru dan kacamata hitam, yang baru saja masuk ke ruangan di seberang aula, hari ini melaporkan bahwa..."*
    *   **a.** Tuliskan jawaban singkat (1-2 poin) yang menjelaskan mengapa model lama (RNN/LSTM) akan kesulitan memahami hubungan kontekstual antara "Pria" (subjek) dan "melaporkan" (predikat).
    *   **b.** Tuliskan jawaban singkat (1-2 poin) yang menjelaskan bagaimana Self-Attention pada Transformer (2.4.3) mengatasi kesulitan tersebut.

3.  **Analisis Pra-Proses (Konsep 2.3: Tokenization & Embeddings)**
    *   Baca kalimat [referensi](https://lms.sdmdigital.id/mod/page/view.php?id=29797): *"Implementasi BPE sangat efisien."*
    *   **a.** Tuliskan perkiraan Anda bagaimana kalimat tersebut akan dipecah menjadi subword tokens (lihat 2.3.1). Beri alasan singkat jika ada kata yang Anda pecah (misal: "Implementasi").
    *   **b.** Tuliskan penjelasan konseptual singkat tentang apa yang terjadi pada Token ID dari token-token tersebut setelah melalui lapisan Embedding dan Positional Encoding (sesuai Gambar 6).

4.  **Analisis Pemilihan Arsitektur (Konsep 2.6: 3 Keluarga)**
    *   Baca tiga skenario berikut:
        *   **(A)** Meringkas artikel berita panjang menjadi satu paragraf.
        *   **(B)** Membuat chatbot yang bisa menjawab pertanyaan secara luwes dan melanjutkan percakapan (generatif).
        *   **(C)** Menganalisis sentimen (positif/negatif) dari ribuan ulasan produk.
    *   **a.** Untuk masing-masing skenario (A, B, dan C), tuliskan arsitektur Transformer (Encoder-only, Decoder-only, atau Encoder-Decoder) yang paling cocok.
    *   **b.** Sertakan justifikasi singkat (1 kalimat) untuk setiap pilihan Anda, berdasarkan analogi "Analis", "Penulis", atau "Penerjemah" (Tabel 6).

5.  **Simpan & Kumpulkan:**
    *   Simpan file jawaban Anda dalam format .md, .txt, atau .pdf.

---

### Tools & Resource yang Dibutuhkan
*   Akses ke materi Modul 2 (terutama Subtopik 2.3, 2.4, 2.5, dan 2.6).
*   Sebuah text editor atau word processor (Notepad, Google Docs, VS Code, dll.) untuk menulis jawaban.

### Contoh Output atau Hasil Akhir
*   Satu file dokumen (.md, .txt, atau .pdf) berisi jawaban yang terstruktur sesuai nomor langkah (2a, 2b, 3a, 3b, 4a, 4b). Jawaban harus jelas, ringkas, dan menunjukkan pemahaman konsep yang dibahas di Modul 2.

### Kriteria Keberhasilan / Penilaian Mandiri
*   Mampu menjelaskan keterbatasan RNN/LSTM (bottleneck) dan keunggulan Self-Attention (koneksi langsung) menggunakan contoh kalimat.
*   Mampu memberikan perkiraan subword tokenization yang logis (misal: memecah kata berimbuhan) dan menjelaskan konsep gabungan Embeddings + Positional Encoding.
*   Mampu memilih arsitektur Transformer yang tepat (Encoder/Decoder/Encoder-Decoder) untuk skenario yang diberikan beserta justifikasinya.
*   Semua jawaban ditulis dengan jelas dan terstruktur dalam satu file dokumen.

### Petunjuk Troubleshooting
*   **Bingung menjelaskan keterbatasan RNN?** Ingat kembali pembahasan tentang Vanishing Gradient dan bottleneck memori (Subtopik 2.4). Fokus pada bagaimana informasi "Pria" harus "selamat" melewati banyak token lain untuk sampai ke "melaporkan".
*   **Kesulitan membayangkan subword tokenization?** Lihat kembali contoh di Subtopik 2.3.1 (misal: "Arsitektur" $\rightarrow$ "Arsi", "##tektur"). Pikirkan kata mana yang mungkin jarang muncul atau punya imbuhan (seperti "Implementasi").
*   **Ragu memilih arsitektur?** Ingat analogi "Si Analis/Paham" (Encoder-only/BERT), "Si Penulis" (Decoder-only/GPT), dan "Si Penerjemah" (Encoder-Decoder/T5) di Subtopik 2.6. Cocokkan dengan tujuan skenario (hanya memahami, hanya menghasilkan, atau mengubah?).

**Durasi Praktikum Disarankan:** 60 menit.

**Aksi Lanjutan atau Refleksi (Opsional):** Setelah selesai, coba pikirkan: Dari semua konsep di Modul 2 (N-gram vs RNN, Tokenization, Embeddings, Attention, 3 Arsitektur), konsep mana yang menurut Anda paling krusial untuk dipahami seorang AI Engineer saat memilih atau menggunakan sebuah LLM di dunia nyata? Mengapa?