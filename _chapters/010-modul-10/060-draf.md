
# Desain Pipeline Ekstraksi Data yang Robust

## Judul Praktikum : Strategi Ekstraksi Data Terstruktur: Dari Prompting hingga Validasi
## Tujuan Praktikum : Peserta mampu merancang strategi ekstraksi data yang tahan banting (resilient), memilih teknik prompting yang tepat (Zero-Shot vs Few-Shot), dan mendesain mekanisme validasi menggunakan Pydantic untuk mencegah data sampah masuk ke sistem.
## Deskripsi Singkat Aktivitas : 
Anda bertindak sebagai Backend Engineer yang ditugaskan mengintegrasikan LLM ke dalam sistem Legacy ERP perusahaan. Sistem ERP ini sangat kaku; jika data JSON yang masuk salah format atau salah tipe (misal string di kolom integer), sistem akan crash. Tugas Anda adalah menganalisis risiko kegagalan ekstraksi dari dokumen nyata dan merancang arsitektur pertahanan berlapis.

---

## Langkah-Langkah Praktikum

1.  **Siapkan Dokumen Jawaban:**
    *   Buat file dokumen baru: "Unjuk Kerja Modul 2.6 - [Nama Anda]".
2.  **Analisis Strategi Prompting (Konsep 2.6.2)**
    *   **Skenario:** Anda harus mengekstrak data dari "Faktur Tulisan Tangan" yang sudah di-OCR (hasil teksnya berantakan dan banyak typo). Anda ingin mengekstrak: Nama Toko, Total Bayar, dan Tanggal.
    *   **Pertanyaan:**
        *   **a.** Mengapa strategi Zero-Shot Prompting ("Ekstrak data ini ke JSON") kemungkinan besar akan gagal atau memberikan hasil buruk pada data OCR yang kotor?
        *   **b.** Buatlah rancangan Few-Shot Prompt (tuliskan 2 contoh input-output) yang mengajarkan model cara menangani typo OCR (misal: mengubah "T0tal: l00.000" menjadi `{"total": 100000}`).
3.  **Analisis Keamanan Format (Konsep 2.6.3)**
    *   **Skenario:** Manajemen khawatir LLM akan menghasilkan JSON yang tidak valid (misal: lupa tanda kutip) yang bisa mematikan API server.
    *   **Pertanyaan:**
        *   Jelaskan bagaimana fitur JSON Mode / Constrained Decoding bekerja di level logits (token probability) untuk menjamin secara matematis bahwa output model pasti memiliki sintaks JSON yang valid.
4.  **Desain Validasi Data (Konsep 2.6.4)**
    *   **Skenario:** Anda mengekstrak data pelamar kerja. Field umur harus berupa Integer. Namun, LLM sering mengekstraknya sebagai string "dua puluh lima tahun" atau "25 thn".
    *   **Pertanyaan:**
        *   Jika Anda menggunakan validasi standar `json.loads()`, data ini akan lolos (karena valid JSON string) tapi merusak database (karena bukan Int).
        *   Jelaskan bagaimana Pydantic dan fitur Type Coercion-nya menangani kasus "25 thn" atau "25" ini agar aman disimpan sebagai Integer.
5.  **Analisis Arsitektur Self-Healing (Konsep 2.6.5)**
    *   **Skenario:** Anda menggunakan library Instructor dengan `max_retries=3`.
    *   **Pertanyaan:**
        *   Gambarkan (secara deskriptif) apa yang terjadi jika pada percobaan pertama LLM lupa mengisi field wajib nomor_telepon. Apa yang dikirimkan library kembali ke LLM, dan apa yang diharapkan dari LLM pada percobaan kedua?

## Tools & Resource yang Dibutuhkan
*   Akses Materi Modul 2.6: Terutama bagian struktur Prompt dan Pydantic.
*   Editor Teks/Code: Untuk menyusun rancangan prompt dan logika validasi.
*   Dokumentasi Pydantic (Opsional).

## Contoh Output atau Hasil Akhir
Satu dokumen laporan berisi:
*   Rancangan Prompt Few-Shot yang konkret.
*   Penjelasan teknis tentang Constrained Decoding (Logit Masking).
*   Analisis kasus validasi tipe data Pydantic.
*   Deskripsi alur loop perbaikan otomatis (Self-Healing).

## Kriteria Keberhasilan / Penilaian Mandiri
1.  Mampu merancang Few-Shot Prompt yang efektif untuk menangani kesalahan data spesifik.
2.  Mampu menjelaskan secara teknis bagaimana Constrained Decoding bekerja di level logits.
3.  Mampu menguraikan mekanisme Type Coercion pada Pydantic.
4.  Mampu mendeskripsikan alur kerja logika retry.
5.  Jawaban ditulis dengan struktur jelas dan terminologi teknis yang tepat.

## Petunjuk Troubleshooting
*   **Bingung beda JSON Mode dan Pydantic?** JSON Mode menjaga Sintaks (Kurung, Koma). Pydantic menjaga Skema (Nama Field, Tipe Data).
*   **Kesulitan membuat contoh Few-Shot?** Cari pola kesalahan umum OCR (misal '0' jadi 'O') dan berikan output bersihnya.
*   **Bingung alur Self-Healing?** Error Python -> Teks Pesan Error -> Input Prompt Baru.

## Durasi Praktik Disarankan
45 – 60 Menit.

## Aksi Lanjutan atau Refleksi (Opsional)
Apakah Anda setuju bahwa Prompt Engineering saja tidak cukup untuk sistem produksi? Mengapa validasi kode seringkali lebih penting daripada "kecerdasan" model itu sendiri?

---

# Penutup Modul 6: Ekstraksi Data Terstruktur dan Validasi Output

Dalam modul ini, kita telah menelusuri transformasi peran LLM: dari seorang "penulis kreatif" menjadi "analis data" yang disiplin. Kita belajar bahwa mengintegrasikan LLM ke dalam sistem software nyata membutuhkan lebih dari sekadar prompt yang bagus; ia membutuhkan arsitektur pertahanan berlapis.

**Intisari Perjalanan Modul 6:**
*   **Tantangan Integrasi (2.6.1):** Risiko Halusinasi Skema dan ketidakpatuhan format.
*   **Strategi Prompting (2.6.2):** Hierarki kontrol dari Zero-Shot hingga Skema Eksplisit.
*   **Kontrol Deterministik (2.6.3):** JSON Mode dan Constrained Decoding di level logits.
*   **Validasi Data (2.6.4):** Pydantic sebagai gatekeeper integritas tipe data.
*   **Otomatisasi Pipeline (2.6.5):** Membangun Self-Correction Loop dengan Instructor/LangChain.

> **Refleksi Akhir:**
> Kepercayaan pada AI tidak dibangun di atas "kecerdasan" model semata, tetapi di atas "pagar pengaman" (guardrails) yang Anda bangun di sekelilingnya.

### Referensi
1. Geng, S., Josifoski, M., Peyrard, M., & West, R. (2023). Grammar-constrained decoding for structured NLP tasks without finetuning. arXiv preprint arXiv:2305.13971.
2. OpenAI. (n.d.). How to work with large language models. OpenAI Cookbook.
3. Pydantic. (n.d.). Pydantic documentation.
4. LangChain. (n.d.). LangChain Python documentation.
5. OpenAI. (n.d.). Structured outputs in multi-agent systems. OpenAI Cookbook.