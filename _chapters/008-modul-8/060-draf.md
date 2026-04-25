---
slug: modul-8-6
title: modul-8-6
---
Berikut adalah isi modul yang telah dirapikan ke dalam format Markdown sesuai dengan hirarki dan instruksi Anda:

# Praktikum Modul 2.4: Desain Sistem dan Arsitektur Aplikasi LLM

---

**Judul Praktikum:** Perancangan Orkestrasi Prompt, Manajemen State, dan Optimasi Sistem AI

**Tujuan Praktikum:** Peserta mampu menganalisis trade-off metode akses LLM (API vs Self-hosting), merancang strategi Context Engineering dan Prompting Lanjutan (CoT, Few-Shot), mendesain arsitektur manajemen State untuk aplikasi stateless, serta mengevaluasi strategi optimasi produksi (Caching, Monitoring).

**Deskripsi Singkat Aktivitas:** Anda bertindak sebagai Lead AI Architect. Tim Anda sedang membangun "EduBot", sebuah asisten pendidikan cerdas. Proyek ini menghadapi tantangan teknis terkait biaya yang membengkak, respons yang lambat, dan model yang sering "lupa" konteks percakapan panjang. Tugas Anda adalah membedah masalah ini dan menyusun desain arsitektur solusi yang efisien dan scalable.

## Langkah-Langkah Praktikum

### 1. Siapkan Dokumen Jawaban
*   Buat file teks/dokumen baru: "Unjuk Kerja Modul 2.4 - [Nama Anda]".

### 2. Analisis Skenario 1: Strategi Akses & Infrastruktur (Konsep 2.4.1)
*   **Skenario:** EduBot saat ini menggunakan API OpenAI untuk semua fitur. Namun, manajemen mengeluh biaya operasional terlalu tinggi karena volume request harian mencapai jutaan. Di sisi lain, ada fitur "Ujian Sekolah" yang membutuhkan privasi data tingkat tinggi (tidak boleh keluar dari server internal sekolah) dan harus berjalan offline saat internet mati.
*   **Pertanyaan:**
    *   a. Untuk fitur "Ujian Sekolah", metode akses apa yang wajib dipilih: API atau Self-Hosting? Jelaskan alasannya berdasarkan faktor privasi dan ketersediaan internet.
    *   b. Jika Anda memilih self-hosting untuk fitur tersebut, sebutkan tiga komponen utama (Hardware/Software/Model) yang harus disiapkan tim infrastruktur.
    *   c. Untuk fitur umum (tanya jawab pelajaran) yang membutuhkan pengetahuan luas dan reasoning tinggi tanpa concern privasi ketat, mengapa tetap disarankan menggunakan API sebagai default?

### 3. Analisis Skenario 2: Prompt & Context Engineering (Konsep 2.4.2 & 2.4.3)
*   **Skenario:** Pengguna mengeluh EduBot sering salah menjawab soal matematika cerita yang rumit, padahal rumusnya sudah benar. Selain itu, saat diminta merangkum buku sejarah setebal 500 halaman, model mengalami error karena context window penuh.
*   **Pertanyaan:**
    *   a. Untuk memperbaiki kemampuan matematika (logika bertahap), teknik prompting apa yang harus diterapkan: Zero-shot atau Chain of Thoughts (CoT)? Berikan contoh instruksi singkatnya.
    *   b. Terkait masalah rangkuman buku 500 halaman, jelaskan mengapa kita tidak bisa sekadar memasukkan semua teks ke dalam prompt? (Hubungkan dengan isu Lost-in-the-middle atau biaya komputasi).
    *   c. Terapkan prinsip Context Engineering: Strategi apa yang lebih efisien untuk menangani dokumen besar tersebut? (Pilih antara: Kompresi/Summarization atau Retrieval/RAG, dan jelaskan mekanismenya secara singkat).

### 4. Analisis Skenario 3: Manajemen State & Orkestrasi (Konsep 2.4.4 & 2.4.5)
*   **Skenario:** EduBot dirancang untuk menjadi teman belajar jangka panjang. Namun, saat user berkata "Seperti yang aku bilang minggu lalu...", bot menjawab "Maaf, saya tidak tahu". Tim junior developer bingung karena mereka mengira LLM punya ingatan otomatis.
*   **Pertanyaan:**
    *   a. Jelaskan konsep Stateless pada LLM kepada tim junior Anda. Mengapa LLM tidak "ingat" percakapan minggu lalu?
    *   b. Anda perlu menyimpan riwayat percakapan jangka panjang. Di mana sebaiknya state (riwayat) ini disimpan agar aman dari crash aplikasi? (Memori program vs Database eksternal).
    *   c. Agar prompt tidak penuh dengan riwayat percakapan selama setahun, teknik apa yang digunakan pada Hands-on Modul 2.4 untuk memadatkan informasi lama? (Sebutkan teknik Rolling Summary).

### 5. Analisis Skenario 4: Optimasi Produksi (Latency & Cost) (Konsep 2.4.5)
*   **Skenario:** Dashboard monitoring menunjukkan banyak siswa menanyakan definisi yang sama berulang kali (misal: "Apa rumus Pitagoras?"). Hal ini memboroskan biaya token dan meningkatkan Latency. Selain itu, user merasa bot "lambat mikir" karena butuh 5 detik sebelum huruf pertama muncul.
*   **Pertanyaan:**
    *   a. Metrik latency apa yang buruk jika user merasa bot "lambat mikir" (lama memproses sebelum mulai mengetik)? (Pilih: TTFT atau TPOT).
    *   b. Untuk menghemat biaya pada pertanyaan yang berulang persis sama ("Apa rumus Pitagoras?"), jenis Caching apa yang paling tepat diterapkan? (Response Caching atau KV Caching?).
    *   c. Jika pertanyaan siswa sedikit berbeda ("Gimana cara hitung Pitagoras?") tapi maknanya sama, mekanisme caching apa yang lebih canggih yang diperlukan?

## Tools & Resource yang Dibutuhkan
*   Akses Materi Modul 2.4: Khususnya bagian Self-hosting vs API, Teknik Prompting (CoT), Manajemen State, dan bagian Caching.
*   Aplikasi Pengolah Kata: Microsoft Word, Google Docs, atau Text Editor.
*   Pemahaman Logika Pemrograman Dasar: Untuk memahami konsep LLM sebagai "fungsi" dan variabel state.

## Contoh Output atau Hasil Akhir
Satu dokumen laporan yang berisi jawaban analisis strategis.
*   **Format:**
    *   Analisis Skenario 1:
        *   a. [Jawaban]
        *   b. [Jawaban]
    *   Analisis Skenario 2: ... (dan seterusnya).
*   **Isi:** Jawaban harus langsung pada inti permasalahan (to-the-point) menggunakan istilah teknis yang dipelajari di modul (misal: Stateless, Semantic Caching, Chain of Thoughts).

## Kriteria Keberhasilan
*   Mampu memutuskan kapan harus self-hosting demi privasi/offline capability.
*   Mampu menerapkan teknik Chain of Thoughts untuk masalah logika kompleks.
*   Memahami bahwa LLM bersifat stateless dan mampu merancang penyimpanan state eksternal.
*   Mampu membedakan TTFT (responsiveness) dan TPOT (speed).
*   Mampu memilih strategi Caching yang tepat untuk efisiensi biaya.

## Petunjuk Troubleshooting
*   **Bingung Skenario 1 (API vs Self-host)?** Buka halaman 33. Cari kata kunci "Kebutuhan Privacy" dan "Internet dan Latency". Ingat, self-hosting memberikan kontrol penuh tapi butuh resource sendiri.
*   **Ragu Skenario 2 (Dokumen Besar)?** Lihat halaman 69-72 tentang Context Engineering. Ingat batasan context window dan masalah lost-in-the-middle. Solusinya adalah "memecah" atau "merangkum".
*   **Bingung Skenario 3 (Bot Lupa)?** Ingat analogi di halaman 86: LLM itu seperti orang yang amnesia setiap kali sesi berakhir. Ingatan (State) harus dicatat di buku catatan (Database) dan disodorkan kembali (Inject to Prompt) setiap kali mau ngobrol.
*   **Bingung Skenario 4 (Caching)?** Lihat halaman 112-113. Jika input teksnya persis sama -> Response Caching. Jika teks beda tapi maksud sama -> Semantic Caching. Jika ingin mempercepat inferensi token -> KV Caching.

**Durasi Praktik Disarankan:** 60 – 90 Menit.

---

> **Aksi Lanjutan atau Refleksi (Opsional)**
>
> Bayangkan Anda harus mempresentasikan anggaran infrastruktur kepada CFO (Direktur Keuangan).
>
> Bagaimana Anda menjelaskan bahwa "menginvestasikan uang untuk membangun sistem Semantic Caching dan Vector Database" (RAG) sebenarnya akan menghemat uang perusahaan dalam jangka panjang dibandingkan membiarkan LLM menjawab semua pertanyaan dari nol (menggunakan token API mahal) setiap saat?

[HANDS-ON: Petunjuk setup](https://github.com/Modul-LLM/Module-08-2.4-Intermediate)

---

# Penutup
## Modul 4 Desain Sistem dan Arsitektur Aplikasi LLM

Melalui modul ini, anda sudah belajar keseluruhan proses dari merancang sebuah aplikasi LLM. Anda sudah mengetahui bahwa merancang sistem LLM tidak terbatas pada memikirkan trik *prompting* yang tepat saja, tapi juga perlu merangkai infrastruktur yang lengkap untuk mempersenjatai LLM dengan konteks yang dibutuhkannya untuk bisa menjalankan tugasnya dengan baik.

### Dalam modul ini, kita telah mempelajari mengenai:
*   **Mempersiapkan akses ke LLM** untuk dipakai dalam sistem yang dikembangkan.
*   **Merancang prompt yang tepat** agar efektif dan memaksimalkan *context window* LLM yang sangat terbatas.
*   **Melakukan orkestrasi pemanggilan LLM** berdasarkan kebutuhan dan aristektur tertentu dengan memandangnya sama seperti sebuah fungsi yang bisa dipanggil dan disusun selayaknya program lainnya.
*   **Mempersiapkan sistem LLM yang dikembangkan** agar siap menghadapi tantangan dunia nyata setelah di-deploy.
*   **Metodologi Evaluasi RAG** menggunakan metrik terukur seperti Context Recall, Context Precision, Faithfulness, dan Answer Relevance, seringkali dengan bantuan LLM-as-a-judge.
*   **Teknik Optimasi dan Troubleshooting** seperti penggunaan Re-ranking, Query Expansion, hingga pemanfaatan metadata untuk meningkatkan akurasi dan efisiensi sistem.

### Dengan menyelesaikan materi ini, Anda telah berhasil mencapai tujuan pembelajaran berikut:
*   Anda kini sudah mampu mempersiapkan akses ke LLM, baik melalui API maupun melakukan *deployment* sendiri melalui *self-hosting*.
*   Anda kini sudah mengerti pentingnya merancang dan merangkai prompt yang baik, tidak hanya agar informasi yang dibutuhkan tersedia, tapi juga memastikan *context window* LLM yang terbatas bisa dimaksimalkan.
*   Anda kini sudah memahami batasan dari hanya *prompt* dan pentingnya orkestrasi untuk implementasi sistem yang kompleks serta pola-pola arsitektur sistem LLM yang umum digunakan.
*   Anda memahami pentingnya manajemen state sebagai bentuk penghematan *context window* dengan cara menyimpan state secara eksternal dari pemanggilan LLM dalam format yang memudahkan hanya bagian tertentu saja yang diambil.
*   Anda mempelajari aspek operasional dari mempersiapkan sebuah sistem LLM agar bisa berjalan dengan baik pada sistem *production*.

---

> **Refleksi Akhir**
>
> *Pada titik ini mungkin anda sadar kebutuhan resource yang begitu besar untuk melatih LLM. Tidak hanya menyediakan compute untuk melatih LLM, tapi juga diperlukan resource yang besar untuk pengumpulan data, anotasi data, serta butuh waktu dan compute untuk mendapatkan konfigurasi yang tepat untuk training. Belum lagi proses training perlu diawasi agar bisa mencapai tujuan.*
>
> *Untungnya, mengembangkan LLM tidak selalu perlu melatih dari nol. Pada modul-modul berikutnya, anda akan mempelajari cara yang lebih realistis untuk mendapatkan model yang sesuai dengan kebutuhan spesifik pengembangan aplikasi yang anda inginkan. Modul 2.2 dan 2.3 akan mempelajari dua pendekatan yang lebih realistis yang berfokus pada mengadaptasikan model yang sudah dilatih sebelumnya.*

---

## Referensi
1.  Prompting Guide. [https://www.promptingguide.ai/](https://www.promptingguide.ai/)
2.  LangChain. LangGraph Workflows & Agents Documentation. [https://docs.langchain.com/oss/python/langgraph/workflows-agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents)
3.  Weng, L. Prompt Engineering. 2023. [https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/)
4.  GeeksforGeeks. Difference Between Batch Processing and Real-Time Processing System. [https://www.geeksforgeeks.org/operating-systems/difference-between-batch-processing-and-real-time-processing-system/](https://www.geeksforgeeks.org/operating-systems/difference-between-batch-processing-and-real-time-processing-system/)
5.  Hugging Face. Understanding KV Caching. [https://huggingface.co/blog/not-lain/kv-caching](https://huggingface.co/blog/not-lain/kv-caching)
6.  Raschka, S. Coding the KV-Cache in LLMs. [https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms)
7.  OpenAI. API Reference Introduction. [https://platform.openai.com/docs/api-reference/introduction](https://platform.openai.com/docs/api-reference/introduction)
8.  Anthropic. Effective Context Engineering for AI Agents. [https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)