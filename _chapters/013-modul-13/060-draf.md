---
slug: modul-13-6
title: modul-13-6
---
# Observability dan Monitoring Sistem LLM

---

**Tujuan Praktikum:** Peserta mampu merancang skema Structured Logging, mendesain hierarki Trace/Span yang efektif untuk debugging, serta menyusun strategi monitoring (Online vs Offline) yang efisien biaya untuk sistem LLM di production.

**Deskripsi Singkat Aktivitas:** Anda bertindak sebagai Lead DevOps/MLOps Engineer. Perusahaan baru saja meluncurkan aplikasi Chatbot RAG (Retrieval-Augmented Generation). Tim Customer Service melaporkan bahwa bot sering memberikan jawaban yang "halu" dan terkadang sangat lambat, tetapi tim teknis tidak tahu penyebab pastinya karena hanya memantau uptime. Tugas Anda adalah merancang "kacamata" (Observability) untuk melihat ke dalam sistem dan menentukan strategi pengawasannya.

---

### Langkah Persiapan

*   Buat file teks/dokumen baru: "Unjuk Kerja Modul 3.1 - [Nama Anda]".
*   Siapkan aplikasi pengolah kata atau text editor.

---

### Langkah-Langkah Praktikum

#### 1. Analisis Skenario 1: Diagnosa Masalah (Monitoring vs Observability)

*   **Skenario:** Dashboard monitoring saat ini hanya menunjukkan lampu hijau (Sistem Up) dan rata-rata latency 2 detik. Namun, user komplain jawabannya salah. Engineer junior bingung karena "tidak ada error di log server".
*   **Pertanyaan:**
    *   **a.** Jelaskan mengapa Monitoring infrastruktur biasa (CPU, Memory, Uptime) tidak cukup untuk mendiagnosa masalah kualitas jawaban LLM? (Gunakan konsep Intermediate Values).
    *   **b.** Identifikasi minimal 3 komponen/tahapan internal dari sistem RAG yang perlu diekspos nilainya agar kita tahu "mengapa" jawaban salah (apakah karena dokumen tidak ketemu, atau LLM salah baca?).

#### 2. Analisis Skenario 2: Implementasi Structured Logging

*   **Skenario:** Tim developer selama ini menggunakan `print("Error disini")` atau log teks biasa. Saat trafik tinggi, ribuan baris log tercampur dan sulit difilter. Anda mewajibkan penggunaan Structured Logging berbasis standar OTel.
*   **Pertanyaan:**
    *   **a.** Berikan contoh desain JSON Log untuk sebuah event "Retrieval Dokumen". Log tersebut harus memuat field standar OTel (Timestamp, Severity Text, Body) dan Attributes yang relevan (seperti `retrieval.top_k`, `retrieval.score`).
    *   **b.** Jelaskan bagaimana Context Propagation (TraceID & SpanID) dapat membantu Anda menghubungkan log "Request masuk dari User A" dengan log "Error di Database Vector" yang terjadi 500ms kemudian?

#### 3. Analisis Skenario 3: Desain Tracing dan Granularitas Span

*   **Skenario:** Anda perlu membagi alur proses aplikasi menjadi Spans agar bisa dilihat di dashboard seperti Langfuse. Alur aplikasinya adalah: User bertanya -> Query diperbaiki (Rewrite) -> Cari Dokumen -> Generate Jawaban.
*   **Pertanyaan:**
    *   **a.** Gambarkan/tuliskan hierarki Parent-Child Span yang ideal untuk alur di atas.
    *   **b.** Engineer Anda menyarankan menggabungkan proses "Rewrite Query" dan "Cari Dokumen" menjadi satu Span saja biar praktis. Berdasarkan prinsip "Satu Span Satu Kesalahan", mengapa saran ini buruk?

#### 4. Analisis Skenario 4: Strategi Monitoring Production (Online vs Offline)

*   **Skenario:** Manajemen ingin mengetahui tingkat "Halusinasi" dan "Biaya Token" secara real-time. Namun, budget komputasi terbatas dan Anda tidak bisa menjalankan LLM-as-a-judge (GPT-4) untuk menilai setiap chat yang masuk.
*   **Pertanyaan:**
    *   **a.** Kelompokkan metrik berikut ke dalam kategori "Mudah Dihitung (Real-time)" atau "Butuh Komputasi Rumit/Offline": Latency (TTFT), Tingkat Toksisitas Jawaban, Cost per Session, Akurasi Retrieval.
    *   **b.** Untuk metrik "Halusinasi" yang mahal dihitung, strategi apa yang Anda usulkan agar tetap bisa memantau kualitas tanpa bangkrut? (Gunakan konsep Sampling atau Proxy).

---

### Output dan Kriteria Penilaian

Satu dokumen laporan yang berisi:
1.  **Analisis perbedaan Monitoring vs Observability.**
2.  **Contoh snippet JSON log yang valid.**
3.  **Diagram pohon (tree) untuk struktur Span.**
4.  **Tabel strategi monitoring (metrik mana yang live, mana yang sampling/offline).**

---

### Kriteria Keberhasilan

*   **Diagnosis Akurat:** Mampu menjelaskan urgensi mengamati intermediate values pada sistem non-deterministik (LLM).
*   **Structured Logging:** Mampu menyusun struktur log yang memiliki konteks semantik (attributes) dan mudah diproses mesin.
*   **Desain Tracing:** Mampu merancang granularitas Span yang tepat (tidak terlalu lebar, tidak terlalu detail).
*   **Strategi Evaluasi:** Mampu membedakan strategi evaluasi Online (tanpa ground truth) dan Offline (dengan ground truth).

---

**Durasi Pengerjaan:** 60-90 Menit.