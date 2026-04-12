---
slug: modul-6-3
title: modul-6-3
---
# 2.3 Catastrophic Forgetting pada CPT dan Dampaknya

---

Di Subtopik 2.2, kita telah menetapkan bahwa CPT adalah proses yang sangat mahal secara *compute* (FLOPs dan VRAM). Namun, "biaya" dari CPT tidak hanya diukur dalam uang dan waktu GPU.

Ada risiko teknis tersembunyi yang jauh lebih berbahaya: Apa yang terjadi pada pengetahuan *lama* model saat kita "memaksa" memasukkan pengetahuan *baru*?

Bayangkan Anda mengambil *Base Model* seperti Llama 3 yang telah dilatih pada triliunan token dari internet sehingga dia bisa tahu sejarah, fisika, cara *coding*, dan berbicara santai. Sekarang, Anda melakukan CPT (melanjutkan pelatihannya) hanya pada 100GB data jurnal medis.

Model Anda mungkin akan menjadi seorang "ahli medis" yang brilian. Tapi apa yang terjadi jika Anda kemudian bertanya, "Siapa presiden pertama Indonesia?" atau "Tuliskan saya sebuah puisi"?

Ada kemungkinan besar model Anda akan gagal total. Ia tidak hanya belajar data medis; ia menimpa (*overwritten*) pengetahuan umumnya dengan data medis tersebut.

Fenomena inilah yang dikenal sebagai ***Catastrophic Forgetting*** (Kelupaan Katastrofik), dan ini adalah hambatan terbesar dalam penerapan CPT di dunia nyata.

---

### 2.3.1 Apa Itu Catastrophic Forgetting?

**Catastrophic Forgetting** (juga dikenal sebagai *Catastrophic Interference*) **adalah** sebuah fenomena dalam *neural network* di mana model ketika dilatih secara sekuensial pada tugas baru (Tugas B, misal: data medis), akan secara cepat dan drastis **kehilangan kinerjanya** pada tugas asli yang telah dipelajarinya (Tugas A, misal: pengetahuan umum internet).

**Analogi: Spesialisasi yang Berlebihan (Anak Kuliah yang Lupa Bahasa Ibu)**
*   **Base Model (Tugas A):** Anda adalah seorang generalis yang fasih berbahasa Indonesia dan memiliki pengetahuan umum yang luas.
*   **Proses CPT (Tugas B):** Anda memutuskan untuk mengambil S3 di bidang hukum dan pindah ke negara lain. Selama 5 tahun, Anda *hanya* membaca dan menulis dalam bahasa "Hukum Teknis" (Legal-ese). Anda tidak pernah berbicara bahasa Indonesia santai atau membaca berita umum.
*   **Hasil (*Catastrophic Forgetting*):** Setelah 5 tahun, Anda adalah seorang ahli hukum (Tugas B) yang tak tertandingi. Namun, ketika Anda mencoba berbicara bahasa Indonesia (Tugas A), Anda menjadi sangat kaku, tata bahasa Anda berantakan, dan Anda lupa fakta-fakta umum di luar hukum. Anda tidak *menambahkan* pengetahuan hukum; Anda **mengganti** pengetahuan umum Anda dengannya.

**Mengapa Ini Terjadi Secara Teknis?**
Ini terjadi karena *neural network* (termasuk Transformer) memiliki **parameter (*weights*) yang sama (*shared parameters*)** untuk semua tugas.

Pengetahuan tentang "Ibu kota Prancis" dan "definisi ibuprofen" mungkin disimpan di lapisan FFN yang sama (ingat Modul 2 Fundamental).

Ketika kita melakukan CPT hanya pada data medis, proses *training* (*backpropagation*) akan "menarik" *weights* tersebut agar optimal **hanya** untuk memprediksi teks medis. *Weights* yang sebelumnya optimal untuk pengetahuan umum akan "ditimpa" tanpa ampun.

Proses "penimpaan" (*overwriting*) *weight* yang fundamental ini tidak hanya menghilangkan satu jenis pengetahuan; dampaknya merambat ke berbagai kemampuan model yang telah terbangun (*emergent*), yang akan kita identifikasi di bagian selanjutnya.

---

### 2.3.2 Dampak Performa Model: Apa Saja yang "Hilang"?

Saat *catastrophic forgetting* terjadi, model tidak hanya lupa fakta. Kerusakannya bisa jauh lebih dalam dan memengaruhi berbagai kemampuan:

1.  **Kehilangan Pengetahuan Faktual (*Factuality Loss*):** Ini adalah yang paling jelas. Model yang di-CPT pada data hukum mungkin akan lupa bahwa langit berwarna biru atau gagal dalam soal matematika dasar. Pengetahuan faktual (yang tersimpan di lapisan FFN) adalah yang pertama kali "tertimpa".
2.  **Kehilangan Kemampuan Penalaran Umum (*Reasoning Loss*):** *Base Model* dilatih pada data yang sangat beragam, termasuk data coding, teka-teki logika, dan matematika. Keragaman data ini mengajarinya kemampuan penalaran abstrak. Jika data CPT (misal, semua data produk internal) sangat repetitif dan tidak memiliki keragaman logika, kemampuan penalaran model bisa "menumpul" atau hilang.
3.  **Kehilangan Kefasihan dan Gaya (*Fluency Loss*):** Model Anda mungkin menjadi "robot". Jika *Base Model* dilatih pada data internet yang kasual (forum, media sosial) dan Anda melakukan CPT pada 50GB data jurnal akademis yang kaku, model Anda mungkin akan "lupa" cara berbicara santai, menulis puisi, atau memahami bahasa gaul.
4.  **Kehilangan *Alignment* (Perilaku):** Ini adalah risiko besar jika Anda melakukan CPT pada model yang sudah di-SFT (misal, ChatGPT). Proses CPT yang "kasar" (menggunakan CLM) dapat **menghapus** semua *alignment* (perilaku sopan, format Q&A) yang telah diajarkan dengan susah payah selama SFT (Modul 1 Intermediate). Model Anda mungkin "kembali liar" (*revert*) menjadi base model yang hanya bisa melengkapi kalimat, bukan menjawab instruksi.

Daftar dampak ini (kehilangan fakta, nalar, dan *alignment*) terlihat sangat berbahaya. Hal ini mungkin membuat Anda bertanya: kita telah membahas SFT (di Modul 1 Intermediate) tanpa terlalu mengkhawatirkan risiko ini. Apa yang membuat CPT secara fundamental jauh lebih berisiko daripada SFT?

---

### 2.3.3 Perbandingan Risiko: Mengapa CPT Jauh Lebih Berisiko daripada SFT?

Di modul pertama, kita belajar SFT, dan kita tidak terlalu khawatir tentang *catastrophic forgetting.* Mengapa CPT jauh lebih berbahaya?

Alasannya terletak pada **Skala Data** dan **Objektif Pelatihan**.

*   **SFT (Risiko Jauh Lebih Rendah):**
    1.  **Objektif Berbeda:** SFT menggunakan *Masked Loss* (2.1.4). Ia hanya melatih *perilaku* (cara menjawab) pada token *respons.* Ia tidak terlalu agresif "menimpa" pengetahuan faktual di FFN.
    2.  **Data Kecil:** Dataset SFT sangat kecil (misal, 20.000 contoh). Jumlah *gradient update* ini terlalu kecil untuk secara drastis "menggeser" *weights* yang telah dilatih pada triliunan token.

*   **CPT (Risiko Sangat Tinggi):**
    1.  **Objektif Sama (CLM):** CPT menggunakan objektif yang sama dengan *pre-training* (memprediksi setiap token). Ini memberi sinyal pada model bahwa data baru ini adalah "kebenaran" baru yang harus menggantikan yang lama.
    2.  **Data Masif (Distribusi Bergeser):** CPT menggunakan dataset masif (puluhan/ratusan GB). Model dipaksa untuk melihat hanya data domain baru ini berulang-ulang. Ini menyebabkan **pergeseran distribusi (*distribution shift*)** yang drastis. Model berpikir bahwa "dunia" sekarang hanyalah data medis, dan ia beradaptasi dengan mengorbankan pengetahuan tentang "dunia" internet yang lama.

Risiko inilah yang membuat CPT *full-parameter* sangat tidak disukai. Di subtopik-subtopik berikutnya (2.4 - 2.6), kita akan membahas teknik-teknik khusus untuk melakukan CPT secara "aman" tanpa memicu kelupaan katastrofik ini.

---

### Mini Quiz

[Konten Interaktif: Multiple Choice Quiz]
*(Akses kuis melalui platform LMS: https://lms.sdmdigital.id/h5p/embed.php?url=https%3A%2F%2Flms.sdmdigital.id%2Fpluginfile.php%2F635429%2Fmod_page%2Fcontent%2F6%2Fmultiple-choice-186.h5p)*

---

> **Refleksi Singkat**
>
> Kita telah membahas bahwa CPT (misalnya, pada data medis) dapat "menimpa" pengetahuan umum (misalnya, data internet).
>
> *Bayangkan Anda berhasil membuat chatbot "Ahli Hukum" (setelah CPT pada data hukum). Apa risiko bisnis atau user experience (UX) yang Anda hadapi jika pengguna (klien) tiba-tiba bertanya, "Jam berapa sekarang di London?" atau pertanyaan umum lainnya di luar domain hukum?*