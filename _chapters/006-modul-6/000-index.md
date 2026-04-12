---
slug: modul-6
layout: part
---

# Modul 2 Continued Pretraining

![Logo Continued Pretraining](https://lms.sdmdigital.id/pluginfile.php/635426/mod_page/content/10/logo%20cpt%20moodle.png)
*Logo Continued Pretraining*

---

### Pendahuluan dan Konteks Kurikulum

Di Program Fundamental (Modul 1-4), kita telah membangun fondasi yang luar biasa. Kita memulai dari dasar-dasar matematika (Modul 1), membedah "anatomi" internal LLM (Modul 2: Transformer), menganalisis "fisiologi" dan perilaku model modern (Modul 3: Sampling, MoE, VLM), dan memetakan "strategi" pelatihan (Modul 4: Kapan memilih Pretrain, SFT, atau CPT).

Di Program Intermediate pada modul sebelumnya, kita telah masuk ke ranah praktis yang fokus pada **Supervised Fine-Tuning (SFT)** dan *Alignment*. Namun, penting untuk meluruskan pemahaman kita: SFT **bukanlah** tentang **mengubah** fundamental model, **melainkan** tentang **mengarahkan** model agar menghasilkan output yang sesuai dengan contoh yang diberikan dalam dataset instruksi.

Model tetap menggunakan basis pengetahuan dan kemampuan reasoning yang sudah dimilikinya dari fase *pre-training*. Peran SFT hanyalah membentuk **perilaku** seperti format jawaban, gaya penulisan, preferensi pengguna, dan pola interaksi guna mempertajam kemampuan model dalam mengikuti instruksi tertentu.

### Perbedaan Pengetahuan dan Perilaku

Namun, bagaimana jika masalahnya bukan *perilaku*, melainkan *pengetahuan*? 

Bagaimana jika model Anda adalah chatbot yang sangat sopan (misalnya model yang merupakan keluaran SFT), tetapi ia tidak tahu apa-apa tentang produk internal perusahaan Anda? Bagaimana jika model dilatih pada data hingga 2023, tetapi Anda memerlukan sehingga model tersebut memiliki pengetahuan akan peristiwa dunia tahun 2024?

Inilah celah yang akan kita isi di modul ini. CPT bukanlah tentang mengajari model *keterampilan baru*; ini tentang menginjeksi *pengetahuan baru*. Alih-alih menggunakan data Q&A berlabel, kita akan "memaksa" model untuk melanjutkan pelatihan pretraining-nya pada data mentah (misalnya, ribuan dokumen internal, artikel berita terbaru, atau jurnal medis).

Ini adalah proses yang jauh lebih berat, mahal, dan berisiko daripada SFT. Jika SFT adalah "mengajari lulusan universitas cara bekerja di perusahaan Anda", CPT adalah "mengirim mereka kembali ke universitas untuk mengambil jurusan baru".

Modul ini adalah panduan *intermediate* untuk melakukan "operasi" pengetahuan ini: kapan melakukannya, berapa biayanya, bagaimana cara menghindar dari bencana (*catastrophic forgetting*), dan teknik apa (termasuk *model merging*) yang dapat kita gunakan untuk melakukannya secara efisien.

### Refleksi dan Masalah Inti

Sebelum kita mempelajari modul ini, alangkah baiknya kita merefleksikan peran krusial dari dokumentasi dan komunikasi ini sebagai berikut:

*   Perusahaan Anda memiliki 10.000 dokumen PDF internal. Apakah Anda akan (A) mengubahnya menjadi dataset tanya-jawab dan melakukan SFT, atau (B) memasukkannya sebagai teks mentah dan melakukan CPT? Apa trade-off dari kedua pilihan tersebut?
*   Mengapa CPT (melatih dengan teks mentah) membutuhkan compute (GPU) yang jauh lebih besar dan mahal daripada SFT (melatih dengan instruksi), meskipun volume datanya mungkin sama?
*   **Masalah Inti:** Jika Anda berhasil melakukan CPT pada model Llama 3 (model umum) dengan 50GB data jurnal medis, risiko terbesar apa yang mungkin terjadi pada kemampuan model tersebut? Bagaimana Anda memastikan model "dokter" baru Anda tidak "lupa" cara berbicara bahasa Indonesia sehari-hari atau melakukan penalaran dasar?
*   Jika Anda memiliki satu model yang sangat hebat dalam coding (misal, CodeLlama) dan satu lagi yang sangat hebat dalam bahasa Indonesia (misal, model lokal), apakah satu-satunya cara menggabungkan mereka adalah dengan CPT? Mungkinkah kita "menggabungkan otak" mereka tanpa pelatihan ulang yang mahal?

### Capaian Pembelajaran

**Digiers**, sebelum memulai pembelajaran, mari kita pahami bersama capaian yang akan kita raih. Berikut adalah tujuan pembelajaran yang akan menjadi kompas kita:

![Tujuan Pembelajaran](https://lms.sdmdigital.id/pluginfile.php/635426/mod_page/content/10/Screenshot%202025-12-23%20at%2015.01.27.png)
*Tujuan Pembelajaran Modul 2 Continued Pretraining*