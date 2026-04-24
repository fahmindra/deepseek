---
slug: modul-11-6
title: modul-11-6
---
Berikut adalah isi konten yang telah dirapikan ke dalam format Markdown sesuai dengan hirarki aslinya:

# Desain Arsitektur: Manajemen State, Pola Workflow, dan Pemilihan Platform Orkestrasi

---

**Tujuan Praktikum:**
Peserta mampu membedakan kebutuhan penggunaan Workflow statis vs Agent dinamis, merancang manajemen State yang terstruktur, menerapkan pola orkestrasi lanjut (seperti Evaluator-Optimizer), dan memilih tools implementasi yang tepat (n8n vs LangGraph) berdasarkan kompleksitas kasus.

**Deskripsi Singkat Aktivitas:**
Anda bertindak sebagai AI Solutions Architect. Perusahaan Anda ingin mengotomatisasi beberapa proses bisnis mulai dari administrasi sederhana hingga riset kompleks. Tim developer mengajukan rancangan sistem untuk berbagai kasus tersebut. Tugas Anda adalah mengaudit rancangan mereka, mengidentifikasi kelemahan logika atau pemilihan alat, dan mengusulkan desain arsitektur yang lebih reliabel dan maintainable.

---

### Langkah-Langkah Praktikum

#### 1. Siapkan Dokumen Jawaban
*   Buat file teks/dokumen baru: "Unjuk Kerja Modul 2.7 - [Nama Anda]".

#### 2. Analisis Skenario 1: Workflow Deterministik vs Agentic (Konsep Workflow vs Agent)
*   **Skenario:** Tim HR ingin membuat bot untuk "Persetujuan Cuti". Aturannya baku: User input tanggal -> Cek sisa cuti di database -> Jika cukup, kirim email ke manajer -> Jika disetujui, update database. Developer Junior mengusulkan menggunakan ReAct Agent penuh di mana LLM bebas menentukan langkah dan memanggil tool sesuka hati.
*   **Pertanyaan:**
    *   **a.** Mengapa pendekatan *fully autonomous Agent* berisiko dan tidak efisien untuk proses yang aturannya sudah baku (deterministik) seperti ini? Jelaskan kaitannya dengan "Struktur dan Transisi".
    *   **b.** Usulkan desain *Graph-based Workflow* (FSM) sederhana untuk kasus ini. Sebutkan Node apa saja yang dibutuhkan dan apa Event pemicu transisinya.

#### 3. Analisis Skenario 2: Manajemen State Terstruktur (Konsep Manajemen State)
*   **Skenario:** Anda sedang merancang *Shopping Assistant Agent*. Saat ini, memori agent hanya berupa *raw chat history* (teks percakapan panjang). Masalah muncul ketika agent harus memvalidasi apakah user sudah menyebutkan ukuran sepatu, warna, dan alamat pengiriman sebelum melakukan checkout.
*   **Pertanyaan:**
    *   **a.** Jelaskan kelemahan menggunakan *raw chat history* untuk validasi data logis oleh sistem programatis.
    *   **b.** Rancanglah sebuah skema State Terstruktur (dalam format JSON) untuk kasus ini. Pastikan ada field untuk menyimpan data belanjaan dan status kelengkapan data.
    *   **c.** Apa keuntungan menggunakan library validasi data (seperti Pydantic) terhadap skema JSON yang Anda buat tersebut?

#### 4. Analisis Skenario 3: Pola Workflow Lanjut (Konsep Pola Evaluator-Optimizer)
*   **Skenario:** Tim Engineering ingin membuat *Coding Agent* yang otomatis memperbaiki bug pada kode lama. Saat ini, mereka menggunakan pola Linear: LLM baca kode -> LLM perbaiki kode -> Selesai. Masalahnya, seringkali kode hasil perbaikan masih mengandung error sintaks saat dijalankan.
*   **Pertanyaan:**
    *   **a.** Identifikasi pola workflow apa yang paling tepat untuk mengatasi masalah ini (pilih dari: *Parallelization*, *Orchestrator-Workers*, atau *Evaluator-Optimizer*).
    *   **b.** Gambarkan atau jelaskan alur kerja baru tersebut, khususnya bagaimana peran komponen Feedback dan Looping bekerja untuk memastikan kode bebas error.

#### 5. Analisis Skenario 4: Pemilihan Platform (Konsep n8n vs LangGraph)
*   **Skenario:** Perusahaan memiliki dua proyek prioritas:
    *   **Proyek A (News Digest):** Setiap pagi jam 8, ambil 5 berita teratas dari RSS Feed, rangkum dengan LLM, kirim ke Telegram Group.
    *   **Proyek B (Deep Research Bot):** User memberi topik riset. Bot harus mencari info, jika info kurang bot harus mencari lagi (looping), merencanakan ulang pencarian, mengkritisi hasil temuan sendiri, baru menulis laporan akhir.
*   **Pertanyaan:**
    *   **a.** Platform mana (n8n atau LangGraph) yang Anda pilih untuk Proyek A? Jelaskan alasannya dari segi kemudahan integrasi dan jenis workflow.
    *   **b.** Platform mana (n8n atau LangGraph) yang Anda pilih untuk Proyek B? Jelaskan alasannya dari segi kapabilitas *State Management* dan kontrol terhadap *Looping/Reasoning*.

---

### Tools & Resource yang Dibutuhkan
*   Akses Materi Modul 2.7: Khususnya bagian FSM (Finite State Machine), Manajemen State (JSON vs Chat History), Pola Workflow, dan Perbandingan n8n vs LangGraph.
*   Aplikasi Pengolah Kata: Microsoft Word, Google Docs, atau Text Editor.
*   Alat Visualisasi (Opsional): Kertas & Pena atau draw.io untuk membantu membayangkan alur workflow.

---

### Contoh Output atau Hasil Akhir
Satu dokumen laporan (format .pdf atau .docx) yang berisi analisis arsitektur.
*   **Format:**
    *   Analisis Skenario 1: [Jawaban a], [Jawaban b]
    *   Analisis Skenario 2: [Jawaban a], [JSON Schema], [Jawaban c]
    *   ... dst.
*   **Isi:** Jawaban harus fokus pada justifikasi teknis (mengapa memilih A bukan B) dan penerapan konsep yang ada di modul.

---

### Kriteria Keberhasilan
*   Mampu menjelaskan mengapa proses deterministik lebih baik menggunakan struktur Workflow/FSM daripada Agent otonom.
*   Mampu merancang struktur JSON untuk State yang efektif dan minim halusinasi.
*   Mampu menerapkan pola *self-correction* (Evaluator-Optimizer) untuk meningkatkan kualitas output.
*   Mampu memilih tools (*Low-code* vs *Code-first*) sesuai dengan kompleksitas logika transisi (Linear vs Cyclic).

---

### Petunjuk Troubleshooting
Gunakan panduan ini jika Anda mengalami kebuntuan:
*   **Bingung di Skenario 1?** Lihat Modul 2.7 bagian "Memahami workflow sebagai sebuah graf". Ingat keuntungan struktur eksplisit: mudah diprediksi dan di-debug. Jangan biarkan LLM menebak langkah selanjutnya jika aturannya sudah jelas.
*   **Kesulitan merancang JSON di Skenario 2?** Lihat contoh di bagian "Manajemen State". Jangan simpan "User ingin sepatu merah" (teks), tapi simpan `{"item": "sepatu", "color": "merah", "ready_to_checkout": false}`. Pikirkan data apa yang perlu dicek oleh kode if-else.
*   **Ragu memilih pola di Skenario 3?** Masalah utamanya adalah *quality control*. Pola mana yang memiliki loop "Ditolak + Feedback"? Lihat diagram "Evaluator-Optimizer" di modul.
*   **Bingung memilih n8n vs LangGraph di Skenario 4?** Kata kuncinya adalah "Looping" dan "Planning".
    *   Proyek A: Linear (Input -> Process -> Output). Banyak integrasi (RSS, Telegram). -> Arahnya ke Visual/Low-code.
    *   Proyek B: Dinamis, butuh "memutar otak" berulang kali, state berubah-ubah kompleks. -> Arahnya ke State Machine eksplisit (Code-first).

**Durasi Praktik Disarankan:** 60 – 90 Menit.

---

### Aksi Lanjutan atau Refleksi (Opsional)
Bayangkan Anda memimpin tim AI Engineering. Bagaimana Anda akan menyeimbangkan kecepatan pengembangan (menggunakan tools visual seperti n8n) dengan kebutuhan akan kontrol granular (menggunakan kode seperti LangGraph) saat sistem yang dibangun mulai berkembang dari prototype sederhana menjadi aplikasi skala besar yang kompleks? Kapan saat yang tepat untuk melakukan migrasi?

---
---

# Penutup 
## Modul 7 Agentic LLM dan Orkestrasi Berbasis Graf

Melalui modul ini, Anda sudah belajar bagaimana meningkatkan kapabilitas LLM dari sekadar pemroses teks secara pasif menjadi sebuah sistem *agentic* yang mampu berinteraksi dengan lingkungannya dan membuat keputusan sendiri dengan supervisi minimal. Anda sudah mengetahui bahwa membangun sistem *agentic* tidak hanya bergantung pada kecerdasan model, tetapi juga memerlukan perancangan struktur, manajemen memori, dan orkestrasi *tools* yang tepat agar sistem dapat mencapai tujuannya.

Dalam modul ini, kita telah mempelajari mengenai:
*   **Konsep dasar Agent** yang terdiri dari sensor untuk mengamati lingkungan dan aktuator (melalui penggunaan *tool*) untuk melakukan aksi nyata.
*   **Pentingnya manajemen State dan Memory** yang terstruktur untuk menjaga konteks penyelesaian masalah, memisahkan antara memori jangka pendek (*working memory*) dan jangka panjang (*semantic/episodic memory*).
*   **Orkestrasi workflow berbasis Graf** menggunakan konsep *Finite State Machine* (FSM) untuk menyeimbangkan fleksibilitas pengambilan keputusan LLM dengan alur proses bisnis yang deterministik.
*   **Implementasi sistem Agent** melalui dua pendekatan berbeda: pendekatan visual (*low-code*) menggunakan n8n dan pendekatan programatis (*code-first*) menggunakan LangGraph.
*   **Metodologi evaluasi** yang komprehensif untuk mengukur kinerja Agent, mencakup aspek eksekusi, kualitas jawaban, hingga efisiensi operasional.

Dengan menyelesaikan materi ini, Anda telah berhasil mencapai tujuan pembelajaran berikut:
*   Anda kini sudah mampu merancang arsitektur Agent yang menggabungkan kemampuan *reasoning* dinamis dengan struktur *workflow* yang jelas.
*   Anda kini sudah mengerti pentingnya desain *state* yang efisien agar LLM dapat memproses informasi dan mengambil keputusan tanpa terbebani konteks yang tidak relevan.
*   Anda kini sudah memahami berbagai pola orkestrasi lanjut seperti *Evaluator-Optimizer* dan *Orchestrator-Workers* untuk menangani tugas yang kompleks.
*   Anda mampu memilih *platform* atau *framework* yang paling sesuai dengan kebutuhan pengembangan sistem, apakah membutuhkan kecepatan integrasi visual atau kontrol logika yang presisi.
*   Anda mempelajari cara mengevaluasi reliabilitas sistem *agentic* secara objektif, baik melalui metode programatis, penilaian manual oleh manusia, maupun bantuan LLM lain (*LLM-as-a-Judge*).

> **Refleksi Akhir**
> 
> Pada titik ini, Anda mungkin menyadari bahwa memberikan otonomi kepada LLM membawa tantangan tersendiri terkait reliabilitas dan prediktabilitas. Sistem yang terlalu kaku sulit beradaptasi, sedangkan sistem yang terlalu bebas berisiko mengalami halusinasi atau terjebak dalam perulangan tanpa henti.
> 
> Kunci keberhasilan dalam membangun aplikasi *agentic* terletak pada keseimbangan. Sebagai seorang *AI Engineer*, tugas Anda adalah menyeimbangkan reliabilitas dan prediktabilitas dari sebuah *workflow* dengan kecerdasan LLM melalui kemampuan *reasoning* untuk menjaga agar kreativitas dan kemampuan *reasoning* LLM tetap terarah pada tujuan penyelesaian masalah. Kita tidak perlu sistem *pure agent* yang mampu benar-benar bertindak secara otonom, tapi sistem yang reliabel dan bisa diprediksi yang benar-benar membantu menyelesaikan masalah dunia nyata yang kompleks.

---

### Referensi

1.  Stuart Russell and Peter Norvig. 2009. Artificial Intelligence: A Modern Approach (3rd. ed.). Prentice Hall Press, USA. [Link](http://repo.darmajaya.ac.id/5272/1/Artificial%20Intelligence-A%20Modern%20Approach%20(3rd%20Edition)%20(%20PDFDrive%20).pdf)
2.  LangChain. LangGraph Overview Documentation. [Link](https://docs.langchain.com/oss/python/langgraph/overview)
3.  n8n. n8n Documentation. [Link](https://docs.n8n.io/)
4.  LangChain. LangGraph Workflows & Agents Documentation. [Link](https://docs.langchain.com/oss/python/langgraph/workflows-agents)
5.  Schmid, P. Agentic Patterns. [Link](https://www.philschmid.de/agentic-pattern)
6.  Anthropic. Building Effective Agents. [Link](https://www.anthropic.com/engineering/building-effective-agents)
7.  OpenAI. A Practical Guide to Building Agents (PDF). [Link](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf)
8.  Psichix. Finite State Machine – Introduction. [Link](https://psichix.github.io/emergent/decision_makers/finite_state_machine/introduction.html)