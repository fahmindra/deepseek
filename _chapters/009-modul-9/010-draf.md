---
slug: modul-9-1
title: modul-9-1
---
Berikut adalah hasil perapian konten ke dalam format Markdown sesuai dengan hirarki dan instruksi yang Anda berikan:

## 5.1 Perbedaan Continued Pretraining (CPT) dengan Fine-Tuning

---

RAG atau Retrieval Augmented Generation merupakan sebuah arsitektur dimana LLM menghasilkan jawaban dengan bantuan informasi eksternal yang didapatkan melalui proses retrieval secara dinamis. Informasi ini disimpan dalam sebuah database, berisi sumber pengetahuan yang berhubungan dengan tujuan aplikasi dikembangkan, misalnya katalog produk untuk asisten toko online, dan SOP perusahaan untuk chatbot asisten HR. Karena RAG menghasilkan jawaban berdasarkan informasi yang disimpan, validitas jawaban RAG bisa diperiksa secara lebih reliabel karena bisa langsung dilacak ke sumbernya.

![Ilustrasi RAG](https://lms.sdmdigital.id/pluginfile.php/635467/mod_page/content/11/image2.jpg)
*Gambar 1. Ilustrasi RAG*

Sebuah sistem RAG terdiri dari komponen berikut:

*   **LLM** berfungsi menentukan informasi apa yang perlu dicari serta menggunakan informasi tersebut untuk memberi jawaban.
*   **Database** yang menyimpan sumber informasi dari RAG.
*   **Retrieval** yang digunakan untuk mencari informasi yang dibutuhkan.

Selain tiga komponen utama di atas, sistem RAG memerlukan sebuah pipeline untuk memproses informasi dari bentuk mentahnya sebelum masuk ke database. Tujuannya adalah mempersiapkan informasi yang formatnya mungkin berbeda-beda menjadi bentuk yang standar dan siap digunakan dalam RAG. Proses ini disebut sebagai ingestion.

Setiap bagian ini disertai dengan alur dan detailnya masing-masing dibahas pada submodul-submodul berikutnya. Submodul 2.5.2 akan membahas lebih lanjut mengenai retrieval, sedangkan submodul 2.5.3 akan membahas lebih lanjut mengenai ingestion.

Selain itu, RAG juga memerlukan metode evaluasi khusus karena jawaban sistem RAG juga bergantung pada retrieval serta bagaimana data diproses saat melakukan ingestion. Submodul 2.5.4 akan membahas lebih lanjut mengenai evaluasi spesifik RAG.

### Mini Quiz

[Klik di sini untuk mengakses Mini Quiz (H5P Content)](https://lms.sdmdigital.id/h5p/embed.php?url=https%3A%2F%2Flms.sdmdigital.id%2Fpluginfile.php%2F635467%2Fmod_page%2Fcontent%2F11%2Fmultiple-choice-231.h5p)

> **Refleksi Singkat**
>
> RAG berfungsi dengan cara mendapatkan informasi yang sudah diproses dengan ingestion, disimpan di database, lalu diambil melalui retrieval.
>
> Jika validitas jawaban RAG bergantung pada kualitas data yang di-retrieve, menurutmu apa risiko utama yang bisa muncul jika proses ingestion atau retrieval dilakukan dengan kurang tepat, dan bagaimana hal itu dapat mempengaruhi kepercayaan pengguna terhadap sistem?