---
slug: modul-10-1
title: modul-10-1
---
Berikut adalah isi konten yang telah dirapikan ke dalam format Markdown sesuai dengan hirarki dan instruksi Anda:

***

## 6.1 Kasus Penggunaan dan Tantangan Ekstraksi Data Terstruktur

---

Sebagian besar data di dunia diperkirakan >80% bersifat tidak terstruktur (*unstructured*). Ini mencakup email pelanggan, dokumen kontrak PDF, transkrip rapat, artikel berita, hingga log percakapan call center.

Bagi manusia, data ini mudah dipahami. Bagi mesin (sistem software tradisional), data ini adalah "sampah" atau noise. Anda tidak bisa melakukan kueri SQL `SELECT * FROM emails WHERE sentiment = 'angry'` jika data sentimen itu masih tersembunyi dalam paragraf teks naratif.

**Structured Data Extraction** adalah proses mengubah data tidak terstruktur tersebut menjadi format terstruktur yang mematuhi skema tertentu. Dalam era LLM, tugas ini berubah dari yang tadinya membutuhkan model NLP spesifik yang rumit menjadi tugas general-purpose yang bisa dilakukan oleh satu model Foundation Model yang cerdas. Namun, fleksibilitas LLM ini datang dengan harga mahal: ketidakpastian.

Di subtopik ini, kita akan membedah di mana saja teknik ini digunakan dan rintangan teknis apa yang akan Anda hadapi saat menerapkannya di produksi.

### 6.1.1 Paradigma: Dari Teks ke Skema

Inti dari ekstraksi data adalah pemetaan (*mapping*). Kita memetakan input yang ambigu dan fleksibel (Bahasa Alami) ke output yang kaku dan deterministik (Skema Database).

![Data Tidak Terstruktur vs Data Terstruktur](https://lms.sdmdigital.id/pluginfile.php/635479/mod_page/content/9/image%20%281%29.png)
**Gambar 2. Data Tidak Terstruktur vs Data Terstruktur**

Gambar di atas mengilustrasikan fungsi inti dari LLM dalam konteks ekstraksi data. Model bertindak sebagai "mesin penstruktur" yang membaca input teks mentah (seperti pesanan pelanggan yang cair), memahami semantiknya, dan memetakan informasi tersebut ke dalam kolom-kolom data yang kaku (Skema JSON) tanpa kehilangan konteks.

Kemampuan untuk mengubah bahasa alami yang "liar" menjadi data digital yang "jinak" ini membuka pintu bagi otomatisasi di berbagai sektor industri. Di mana saja teknik transformasi ini diterapkan dalam dunia nyata?

### 6.1.2 Kasus Penggunaan Utama (Use Cases)

Di industri, ekstraksi data dengan LLM digunakan untuk mengotomatisasi proses manual yang membosankan.

1.  **Pemrosesan Dokumen Cerdas:**
    *   Contoh: Mengekstrak data dari Faktur (*Invoice*) atau Kuitansi yang formatnya berantakan.
    *   Data: Tanggal transaksi, Nama Vendor, Line Items, Total Pajak.
    *   Manfaat: Memasukkan data otomatis ke sistem akuntansi (ERP) tanpa ketik manual.
2.  **Analisis Layanan Pelanggan:**
    *   Contoh: Menganalisis tiket keluhan pelanggan yang masuk via email.
    *   Data: Kategori Masalah (misal: "Login Error"), Tingkat Urgensi (High/Medium/Low), Sentimen Pelanggan, dan ID Produk yang disebutkan.
    *   Manfaat: Routing tiket otomatis ke tim yang tepat.
3.  **Ekstraksi Informasi Medis & Legal:**
    *   Contoh: Membaca rekam medis pasien (teks narasi dokter).
    *   Data: Diagnosis (kode ICD-10), Daftar Obat, Dosis, Alergi.
    *   Manfaat: Mempercepat klaim asuransi atau analisis klinis.
4.  **Pembangunan Knowledge Graph:**
    *   Contoh: Membaca ribuan artikel berita keuangan.
    *   Data: Entitas (Perusahaan, CEO), Relasi (Akuisisi, Investasi, Tuntutan Hukum).
    *   Manfaat: Memetakan hubungan antar-perusahaan secara otomatis.

Meskipun potensi aplikasinya sangat luas dari otomatisasi keuangan hingga layanan pelanggan, implementasi di dunia nyata jarang berjalan mulus tanpa pengaman (*guardrails*). Mengandalkan model probabilistik (LLM) untuk tugas yang menuntut presisi deterministik (Database) membawa risiko kegagalan unik yang tidak ditemukan pada pemrograman konvensional. Kita harus memahami risiko-risiko ini sebelum bisa memitigasinya.

### 6.1.3 Tantangan Teknis & Risiko Kegagalan

Meskipun terdengar "ajaib", menggunakan LLM sebagai ekstraktor data membawa risiko unik yang tidak ada pada sistem software deterministik. Berikut adalah empat tantangan utamanya:

1.  **Schema & Value Hallucination (Halusinasi Skema & Nilai)**
    *   **Penjelasan:** Fenomena di mana model menghasilkan struktur data yang tampak valid dan meyakinkan, namun isinya merupakan fabrikasi (karangan) yang tidak berdasar pada dokumen sumber.
    *   **Masalah:** Model merasa "wajib" mengisi setiap kolom yang diminta dalam skema, sehingga ia cenderung mengarang data jika informasi tersebut hilang di teks asli, demi "memuaskan" permintaan pengguna.
    *   **Contoh:** Input teks tidak menyebutkan nomor telepon sama sekali, tetapi model mengisi field *phone* dengan nomor acak 08123456789 hanya agar kolomnya tidak kosong. Ini fatal untuk integritas data.
2.  **Format Non-Compliance (Ketidakpatuhan Format)**
    *   **Penjelasan:** Kegagalan model untuk menghasilkan string output yang mematuhi aturan sintaksis ketat dari format data target (seperti JSON atau XML).
    *   **Masalah:** LLM adalah model probabilistik yang memprediksi token demi token. Ada peluang kecil tapi nyata ia akan menghasilkan token yang merusak sintaks, seperti lupa menutup tanda kutip atau menambahkan koma berlebih (*trailing comma*).
    *   **Dampak:** Sistem downstream (API atau Database) akan mengalami *Parsing Exception* atau *Crash*, dan data gagal disimpan secara otomatis.
3.  **Normalization Issues (Masalah Normalisasi & Ambiguitas)**
    *   **Penjelasan:** Tantangan dalam mengubah bahasa manusia yang bersifat relatif, ambigu, dan tidak baku menjadi format data mesin yang absolut dan terstandarisasi.
    *   **Masalah:** Teks sumber sering menggunakan referensi waktu relatif (misal: "besok") atau satuan yang tidak baku, sedangkan database membutuhkan format absolut (Timestamp ISO) atau satuan standar.
    *   **Contoh:**
        *   Input: "Hubungi saya besok sore."
        *   Ekstraksi Naif: `{"time": "besok sore"}` -> Data ini tidak berguna bagi database.
        *   Kebutuhan: Model harus melakukan reasoning untuk mengubahnya menjadi `"time": "2025-11-22T15:00:00"` berdasarkan tanggal hari ini.
4.  **Missing Fields & Null Handling (Penanganan Data Kosong)**
    *   **Penjelasan:** Situasi di mana informasi yang diminta dalam skema target ternyata absen atau tidak tersedia dalam dokumen sumber.
    *   **Masalah:** Model seringkali bingung menghadapi kekosongan ini. la mungkin mencoba menebak nilai (halusinasi), atau malah mengubah struktur JSON dengan menghapus key tersebut sepenuhnya.
    *   **Risiko:** Jika key dihapus, skema database yang kaku akan menolak data tersebut (*Schema Validation Error*). Kita harus memaksa model untuk secara eksplisit menggunakan nilai standar seperti `null` atau `"N/A"`.

### Mini Quiz
*(Konten interaktif tersedia pada platform LMS)*

Di subtopik berikutnya, kita akan mulai membahas teknik Prompting untuk meminimalisir risiko-risiko tersebut.

> **Refleksi Singkat**
>
> Kita telah membahas bahwa LLM rentan terhadap Halusinasi Skema (mengarang data agar sesuai format JSON). Bayangkan Anda membangun sistem otomatisasi pembayaran faktur (*Invoice Automation*). Sistem ini membaca PDF tagihan dari vendor dan otomatis melakukan transfer bank. Jika model Anda memiliki akurasi ekstraksi 99%, itu berarti 1 dari 100 pembayaran mungkin salah jumlah atau salah tujuan transfer. Dalam konteks software engineering tradisional, error rate 1% seringkali tidak dapat diterima untuk transaksi finansial. Sebagai Al Engineer, apakah Anda akan membiarkan sistem ini berjalan fully autonomous, atau mekanisme human-in-the-loop (verifikasi manusia) seperti apa yang wajib Anda desain dalam arsitektur sistem tersebut?