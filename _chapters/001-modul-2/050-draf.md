---
slug: modul-2-5
title: modul-2-5
---

### Perencanaan Proyek LLM dan Hands-on Pytorch
---

**Tujuan Praktikum :** Siswa mengerti bagaimana praktiknya merencanakan sebuah proyek LLM berdasarkan tanggung jawab seorang AI Engineer. Siswa juga mendapatkan pengalaman hands-on Pytorch untuk mengembangkan model sederhana.

**Deskripsi Singkat Aktivitas :** Siswa melakukan perencanaan awal / garis besar dari sebuah proyek LLM, melakukan pengamatan hands-on Pytorch dan mengerjakan latihan mengembangkan model melalui Pytorch.

### Langkah-Langkah Praktikum

1.  **Perencanaan Proyek LLM**
    *   Setelah mempelajari tanggung jawab seorang AI Engineer, mari mencoba mempraktikkan secara high level gambaran besar mengenai apa saja yang harus dilakukan ketika ingin melakukan pengembangan aplikasi berbasis LLM.
    *   Lihat sekeliling kalian, coba cari masalah apa yang ingin kalian coba selesaikan dengan LLM. Contoh:
        *   Mengembangkan analisis media sosial dengan LLM
        *   Chatbot untuk membantu mengikuti sebuah resep memasak
        *   Aplikasi untuk membuat slide presentasi secara otomatis
    *   Lalu coba jawab pertanyaan berikut:
        *   Apa yang akan menjadi input dan output LLM?
        *   Apa yang akan menjadi input dan output sistem secara keseluruhan?
        *   Apakah input perlu diproses dulu? Biasanya input tersedia dalam bentuk apa? Bagaimana cara memprosesnya kira-kira sebelum bisa diberikan ke LLM?
        *   Kira-kira output sistemnya seperti apa? Bagaimana caranya output LLM bisa diproses menjadi output sistem?
        *   LLM seperti apa kira-kira yang bisa digunakan? (Cari model free tier di Google Gemini/OpenRouter atau open source di HuggingFace. Baca model card untuk memahami kemampuannya).
        *   Penyajian sistem ini kira-kira dalam bentuk apa? (API, chatbot interaktif, atau webapp?). Coba cari tahu dulu biasanya aplikasi LLM disajikan dalam bentuk apa?
    *   Buatlah gambaran kasar (overview) dari keseluruhan sistem. Tidak perlu terlalu teknis, fokus pada pemahaman masalah dan komponen penyelesaiannya. Serta tujuannya adalah mengaplikasikan pemahaman kalian mengenai tanggung jawab seorang AI Engineer.
    *   Tidak perlu terlalu teknis juga, karena bagian paling penting sebenarnya adalah kalian memahami masalah yang ingin diselesaikan, dan tahu apa saja bagian dari masalah tersebut yang harus diselesaikan satu per satu.
    *   Gunakan template berikut sebagai format untuk menulis laporan akhir: [Template laporan unjuk kerja 1](https://docs.google.com/document/d/1DpGf2e7V1Ja_CfcQHoX3mzt88EL3YdkDVq2Xq0WMarA/edit?usp=sharing)

2.  **Hands-on Pytorch**
    *   Akses repository berikut dan clone: [[link repository]](https://github.com/Modul-LLM/Module-01-1.1-Fundamental).
    *   Set up environment python, lalu jalankan semua kode hands-on. Amati dan pahami apa yang terjadi, serta lakukan eksperimen/variasi pada kodenya.
    *   Kerjakan tugas yang ada di folder `tugas-modul-1`.

3.  **Kumpulkan Kode dan Laporan**
    *   Gunakan template berikut untuk menulis laporan: [Template Laporan Unjuk Kerja 1](https://docs.google.com/document/d/1DpGf2e7V1Ja_CfcQHoX3mzt88EL3YdkDVq2Xq0WMarA/edit?usp=sharing).
    *   Laporan harus mendetailkan:
        *   Pengamatan/catatan hands-on (apa yang diamati dan dipahami).
        *   Detail pengerjaan tugas dalam bentuk writeup, ceritakan apa yang kalian lakukan untuk melakukan implementasi dan alasannya.

### Tools & Resource yang Dibutuhkan
*   *Text Editor* untuk menulis kode (VS Code, Vim, Atom) dan untuk menulis laporan (Google Docs).
*   Akses ke Google Colab.
*   Instalasi Python (jika menjalankan eksperimen secara lokal).
*   Akses ke Modul 1.
*   Akses ke repository [[link repository]](https://github.com/Modul-LLM/Module-01-1.1-Fundamental).

### Contoh Output atau Hasil Akhir
*   Satu dokumen laporan berisi dua bagian praktikum (Perencanaan & Hands-on) serta kode hasil implementasi tugas.

### Kriteria Keberhasilan / Penilaian Mandiri
*   Siswa mampu mencoba merencanakan sebuah proyek LLM untuk menyelesaikan suatu masalah.
*   Siswa mampu menggunakan Pytorch untuk implementasi model dan proses training sederhana.

### Petunjuk Troubleshooting
*   Gunakan studi kasus pada Modul 1 sebagai [referensi](https://lms.sdmdigital.id/mod/page/view.php?id=29797) bentuk rencana proyek LLM. Fokus pada pemahaman garis besar penggunaan LLM dan peran AI Engineer.
*   Jika mengalami kesulitan Pytorch, cari informasi di dokumentasi resmi: [https://docs.pytorch.org/docs/stable/index.html](https://docs.pytorch.org/docs/stable/index.html).

**Durasi Praktikum Disarankan:** 60 menit.

**Aksi Lanjutan atau Refleksi (Opsional):** Buka [https://github.com/karpathy/nanochat](https://github.com/karpathy/nanochat). Repository ini berisi implementasi “chatgpt” mini. Coba pelajari bagaimana Pytorch digunakan untuk implementasi model dan proses training.