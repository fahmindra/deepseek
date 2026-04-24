---
slug: modul-9-5
title: modul-9-5
---
## 5.5 Troubleshooting dan Optimasi RAG

---

Bayangkan anda telah membuat aplikasi berbasis RAG dan sudah di-deploy ke production yang berjalan untuk beberapa waktu, lalu muncul beberapa komplain dari klien:

*   **“AInya kenapa lama banget ya jawab pertanyaannya?”**
*   **“Jawaban dari AInya kok kayaknya menyesatkan ya? di dokumen seingat saya gak ada informasi itu”**
*   **“AInya kok gak bisa jawab ya pertanyaan yang gampang, padahal di dokumen ada?”**

Seiring waktu, sistem mungkin akan mengalami kesulitan menangani skala penggunaan (jumlah user, jumlah pemanggilan per hari) maupun skenario penggunaan yang beragam. Kesulitan ini bisa dilihat melalui keluhan-keluhan user yang menggunakan sistem tersebut. Seorang AI Engineer perlu melakukan maintenance dan optimasi agar sistem bisa tetap bisa memenuhi SLA (*Service level agreement*) yang dijanjikan kepada pengguna.

Tapi RAG adalah sistem yang kompleks, dan mungkin sulit diketahui bagian mana dari sistem yang perlu dioptimasi untuk keluhan atau penurunan performa tertentu. Jadi pertama perlu dilakukan troubleshooting dulu untuk mencari bottleneck sistem yang perlu dioptimasi.

### 5.5.1 Troubleshooting / debugging

Troubleshooting dilakukan dengan menggunakan metrik maupun mengamati secara langsung trace atau log dari sebuah sistem ketika terjadi sebuah kesalahan. Misalnya ketika latency terlalu tinggi, perlu dilakukan profiling waktu untuk menyelesaikan per-bagian dari setiap tahap RAG dan total waktu untuk menghasilkan jawaban, baik keseluruhan (*end to end*) maupun TTFT. Dari profiling ini bisa diidentifikasi bagian mana yang memakan waktu paling banyak untuk dilakukan optimasi bertarget.

![Ilustrasi hasil profiling](https://lms.sdmdigital.id/pluginfile.php/635471/mod_page/content/14/image14%20%281%29.png)
*Gambar 23. Ilustrasi hasil profiling*

**Untuk troubleshooting masalah akurasi, agak sedikit lebih kompleks karena akurasi akhir bisa dipengaruhi beberapa faktor, misalnya:**

*   Retrieval mengembalikan dokumen yang tidak relevan (masalah embedding, query, chunking, atau re-ranking).
*   LLM tidak menggunakan konteks yang dikembalikan retrieval dengan tepat (ada bug dalam kode untuk memasukkan konteks, atau bug saat ingestion, menggunakan konteks yang tidak relevan, dsb).

Faktor ini bisa dideteksi melalui metrik performa maupun dengan cara mengamati output-output intermediate dari RAG melalui sistem tracing dan monitoring, misalnya mengamati langsung query yang digunakan untuk melakukan pencarian, atau mengamati langsung dokumen apa saja yang dikembalikan dari pencarian, atau melihat bentuk pertanyaan user secara langsung.

Sistem tracing dan monitoring memberikan gambaran keseluruhan apa yang terjadi dalam setiap tahap dalam sebuah sistem, dan akan dipelajari secara lebih lanjut pada modul 3.1. Yang anda perlu ketahui sekarang adalah sistem seperti ini tidak hanya mencatat apa input dan output setiap kali RAG berjalan, tapi juga semua nilai perantara yang dihasilkan, seperti query, embedding, hasil retrieval, dan lain sebagainya.

#### Contoh studi kasus debugging RAG

Ada keluhan kalau performa sistem RAG kurang baik karena jawabannya salah. Seorang AI Engineer bisa melakukan debugging secara sistematis sebagai berikut:

1.  **Sebagai sanity check**, Anda bisa memeriksa apakah dokumen hasil retrieval benar-benar masuk sebagai konteks ke prompt terakhir yang digunakan menghasilkan jawaban. Anggap saja Anda bisa melihat prompt berikut diberikan ke LLM saat menghasilkan jawaban (dipotong sebagai ilustrasi):

    > ...
    > 
    > Gunakan dokumen berikut untuk menjawab pertanyaan di atas!
    > Dokumen:
    > ```
    > Dokumen 1: Berdasarkan peraturan perusahaan mengenai...
    > Dokumen 2: Setiap karyawan berhak atas .....
    > ...
    > ```
    > ...

    Ternyata benar dokumen hasil retrieval berhasil dimasukkan ke prompt sebagai konteks. Berarti masalahnya bukan di situ, kita bisa mundur lagi.

2.  **Langkah paling logis berikutnya** adalah memeriksa apakah dokumen-dokumen hasil retrieval benar-benar valid dan berhubungan dengan pertanyaan user. Misalnya user bertanya mengenai SOP perusahaan soal cuti. Setelah diperiksa, ternyata dokumen yang dihasilkan tidak tepat, malah dokumen-dokumen tidak relevan. Dari sini ada dua kemungkinan:
    *   Proses retrieval awal sudah benar, tapi ada kesalahan saat re-ranking.
    *   Kesalahannya lebih dalam lagi, perlu memeriksa retrieval.

Kemudian anda memeriksa query dan hasil retrieval awal terlebih dahulu:

**Query:**
> Peraturan mengenai Cuti Karyawan

**Hasil retrieval:**
```json
[
    {
        "document_name": "Ketentuan lembur",
        "content": "Karyawan hanya diperkenankan lembur dengan ...",
        "similarity_score": 0.23
    },
    {
        "nama dokumen": "Peraturan Jam kerja",
        "konten dokumen": "Karyawan diwajibkan memenuhi jam kerja ...",
        "similarity_score": 0.33
    },
    {
        "nama dokumen": "Ketentuan Sistem on-call",
        "konten dokumen": "Karyawan yang on-call diwajibkan untuk ...",
        "similarity_score": 0.35
    }
]
```

Ternyata dari awal retrieval sudah memberikan dokumen yang tidak relevan. Kemudian anda ingat-ingat detail mengenai retrieval yang anda lakukan:
*   Chunk maksimal 1024 token
*   Dokumen dibagi-bagi berdasarkan strukturnya
*   Model embedding yang digunakan adalah Qwen3-Embedding-0.6B
*   Digunakan fungsi cosine similarity sebagai fungsi kedekatan

Barulah anda sadar, setelah melakukan retrieval, anda mengurutkannya secara salah. Berbeda dengan L1/L2 Distance yang semakin kecil berarti semakin berdekatan, cosine similarity justru sebaliknya, semakin tinggi justru semakin berdekatan. Kemudian anda memperbaiki pengurutannya dan menguji lagi retrieval dengan query yang sama, dan didapatkan bahwa retrieval sudah mengembalikan dokumen yang sesuai, dan jawaban akhir kemudian sudah benar.

Begitulah gambaran melakukan debugging RAG. Anda perlu mengetahui alur kerja setiap proses internalnya, memetakan faktor apa yang mempengaruhi suatu proses, lalu melakukan inspeksi untuk mencari letak kesalahan.

### 5.5.2 Teknik-teknik optimasi RAG

Banyak metode yang bisa dilakukan untuk optimasi latency maupun akurasi dari sebuah sistem RAG. Berikut akan dibahas beberapa metode diurutkan dari yang paling praktikal hingga yang rumit.

#### **Optimasi Latency**

Faktor penyebab latency tinggi:
*   Database terlalu besar sehingga pencariannya lambat.
*   Konteks yang diberikan ke LLM terlalu panjang.

**Teknik optimasi:**

1.  **Build ulang index**
    *   **Level:** Retrieval
    *   **Definisi:** Melakukan update index dari awal untuk mengatasi index fragmentation.
    *   **Kapan digunakan:** Ketika jumlah chunk sangat besar (>100k) dan performa melambat.
    *   **Kompleksitas:** Sedang
    *   **Risiko:** Memakan waktu dan resource besar.

2.  **Melakukan Caching**
    *   **Level:** Retrieval
    *   **Definisi:** Menerapkan *semantic caching* untuk query yang sering muncul.
    *   **Kapan digunakan:** Ketika banyak query mirip secara makna berulang kali.
    *   **Kompleksitas:** Rendah
    *   **Risiko:** Hasil cache berpotensi "usang" jika dokumen database berubah.

3.  **Melakukan Kompresi Embedding**
    *   **Level:** Retrieval
    *   **Definisi:** Menggunakan embedding yang lebih pendek, terkuantisasi, atau *matryoshka embedding*.
    *   **Kapan digunakan:** Jika butuh speed-up tanpa penurunan kualitas signifikan.
    *   **Kompleksitas:** Rendah - Sedang
    *   **Risiko:** Berpotensi menurunkan akurasi (bisa dimitigasi dengan re-ranking).

4.  **Kompresi konteks**
    *   **Level:** Jawaban akhir
    *   **Definisi:** Melakukan summarization pada konteks atau memperkecil nilai K/ukuran chunk.
    *   **Kapan digunakan:** Ketika menghasilkan jawaban akhir menjadi bottleneck.
    *   **Kompleksitas:** Rendah
    *   **Risiko:** Menghilangkan detail penting.

#### **Optimasi akurasi RAG**

**Hal-hal fundamental yang perlu dipastikan sebelum optimasi:**
*   Sudah ada evaluasi yang jelas (metrik).
*   Konteks hasil retrieval benar-benar masuk ke prompt.
*   Model embedding yang sama digunakan untuk query dan database.
*   Proses ingestion menghasilkan representasi yang baik.
*   Tidak ada kesalahan teknis (misal: normalisasi vektor).

**Teknik optimasi akurasi:**

1.  **Optimasi chunking**
    *   **Level:** Ingestion
    *   **Definisi:** Menggunakan metode chunking yang lebih baik (tidak memotong informasi di tengah) atau memperkaya konteks dengan metadata (parent-child chunking).
    *   **Kapan digunakan:** Optimasi pertama jika akurasi buruk.
    *   **Kompleksitas:** Rendah-Sedang.

2.  **Hybrid Search**
    *   **Level:** Retrieval
    *   **Definisi:** Menggabungkan dense retrieval (semantik) dengan sparse retrieval (BM25 - kata kunci).
    *   **Kapan digunakan:** Efektif untuk Customer Support (kode barang) atau pencarian entitas spesifik.
    *   **Kompleksitas:** Sedang.

3.  **Re-ranking**
    *   **Level:** Setelah retrieval
    *   **Definisi:** Mengambil lebih dari K dokumen, lalu diurutkan kembali menggunakan model yang lebih presisi.
    *   **Kapan digunakan:** Ketika ranking awal kurang akurat.
    *   **Kompleksitas:** Sedang-Tinggi.
    *   **Risiko:** Menambah latency.

4.  **Query Expansion**
    *   **Level:** Sebelum retrieval
    *   **Definisi:** Memperluas query dengan parafrase atau membuat "dokumen bohongan" (*HyDE*) untuk di-embed.
    *   **Kapan digunakan:** Query user terlalu pendek atau ambigu.
    *   **Kompleksitas:** Sedang.
    *   **Risiko:** Risiko mendapatkan hasil tidak relevan.

5.  **Menggunakan model embedding spesifik domain**
    *   **Level:** Ingestion & Retrieval
    *   **Definisi:** Training ulang model embedding pada domain khusus (legal, medis).
    *   **Kapan digunakan:** Ketika model umum gagal menangkap makna teknis domain tertentu.
    *   **Kompleksitas:** Tinggi.

6.  **Filtering & Re-ranking berdasarkan metadata**
    *   **Level:** Retrieval
    *   **Definisi:** Membatasi pencarian berdasarkan atribut (tanggal, ID klien, kategori).
    *   **Kapan digunakan:** Sebaiknya selalu dilakukan jika metadata tersedia.
    *   **Kompleksitas:** Rendah.

7.  **Retrieval dengan hierarki**
    *   **Level:** Ingestion & Retrieval
    *   **Definisi:** Mengorganisir data dalam struktur bertingkat (parent-child).
    *   **Kapan digunakan:** Ketika konteks luas diperlukan untuk mengerti chunk kecil.
    *   **Kompleksitas:** Sedang.

8.  **Retrieval multi-step (Agentic RAG)**
    *   **Level:** Retrieval
    *   **Definisi:** LLM melakukan planning pencarian atau melakukan loop pencarian hingga info cukup.
    *   **Kapan digunakan:** Pertanyaan kompleks yang butuh menghubungkan info tersebar.
    *   **Kompleksitas:** Tinggi.

**Tabel 12. Rangkuman singkat kapan setiap metode bisa digunakan dan tradeoffs-nya**

| Teknik | Level | Kapan Digunakan (Rule of Thumb) | Kompleksitas | Tradeoffs / Risiko |
| :--- | :--- | :--- | :--- | :--- |
| Rebuild Index | Retrieval | Retrieval makin lama, DB >100k chunk. | Sedang | Waktu build lama. |
| Semantic Caching | Retrieval | Banyak query mirip / berulang. | Rendah | Cache cepat usang. |
| Embedding Compression | Retrieval | Butuh speed-up retrieval. | Rendah–Sedang | Akurasi turun jika agresif. |
| Context Compression | Jawaban Akhir | Output LLM lambat (konteks panjang). | Rendah | Risiko hilang detail penting. |
| Metadata Filtering | Retrieval | Dokumen punya atribut (tanggal, kategori). | Rendah | Perlu metadata konsisten. |
| Optimasi Chunking | Ingestion | Akurasi buruk (info terpotong). | Rendah–Sedang | Perlu metadata (parent/child). |
| Hybrid Search | Retrieval | Perlu exact matching + semantik. | Sedang | Komputasi bertambah. |
| Re-ranking | Pasca Retrieval | Ranking awal kurang presisi. | Sedang–Tinggi | Latency naik. |
| Query Expansion | Pre-retrieval | Query pendek / ambigu. | Sedang | Risiko hasil tidak relevan. |
| Domain Embedding | Ingestion | Makna domain teknis tidak tertangkap. | Tinggi | Training mahal; re-embed data. |
| Hierarchical Retrieval| Ingestion+Retr | Dokumen berlapis/kompleks. | Sedang | Pipeline makin kompleks. |
| Multi-step/Agentic | Retrieval | Konteks tersebar; butuh reasoning. | Tinggi | Latency tinggi. |

---

### Mini Quiz
*(H5P Content Placeholder - Multiple Choice)*

> #### Refleksi Singkat
> Sebelum optimasi dilakukan, telah ditekankan pentingnya memastikan tidak ada “kesalahan minor” seperti embedding tidak konsisten, atau konteks yang tidak masuk ke prompt, atau ada beberapa dokumen yang ternyata tertinggal saat ingestion. Namun dalam sistem nyata, kesalahan minor sering luput karena tidak terlihat di metrik.
> 
> Menurut Anda, bagaimana cara paling efektif untuk mendeteksi kesalahan minor ini sejak awal?