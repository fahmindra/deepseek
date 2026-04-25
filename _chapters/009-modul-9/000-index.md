---
slug: modul-9
layout: part
---
Berikut adalah isi konten yang telah dirapikan ke dalam format Markdown sesuai dengan hirarki dan instruksi Anda:

# Modul 5 Arsitektur, Implementasi, dan Evaluasi Sistem RAG

![Logo Modul 5](https://lms.sdmdigital.id/pluginfile.php/635466/mod_page/content/10/_bc8e70e6-c0c1-4514-ae4e-a91ff5a2edf5.jpeg)
*Ilustrasi Modul 5 Arsitektur, Implementasi, dan Evaluasi Sistem RAG*

---

LLM yang dilatih dari data internet bisa “menyerap” banyak pengetahuan internal, tapi pengetahuan ini sulit diverifikasi dan rawan hallucination. Selain itu, cakupannya juga terbatas pada data latih, sehingga kurang reliabel untuk domain spesifik. CPT dan SFT dari Modul 2 dan 3 bisa menambahkan pengetahuan baru, namun membutuhkan resource besar dan tidak praktis dilakukan terus-menerus, terutama jika pengetahuan domain berubah cepat dari minggu ke minggu.

Pendekatan yang lebih realistis adalah menyuntikkan konteks secara dinamis lewat prompt seperti yang dibahas di Modul 8. Tantangannya muncul ketika ukuran dan jumlah konteks sangat besar. Informasi harus dipecah menjadi kepingan yang lebih kecil, disimpan dengan format yang sesuai, dan dipilih secara tepat untuk setiap task. Inilah motivasi utama di balik teknik Retrieval-Augmented Generation (RAG).

Bagaimana caranya informasi bisa dipilih secara dinamis? Bagaimana cara menyimpannya? Bagaimana kalau informasinya tidak hanya teks saja? Pemrosesannya seperti apa?

Dalam modul ini, Digiers akan diajak memahami lebih lanjut mengenai salah satu paradigma / arsitektur yang umum digunakan untuk mengembangkan aplikasi AI yang membutuhkan pengetahuan eksternal yang intensif. Arsitektur ini disebut dengan Retrieval augmented generation atau lebih umum disebut sebagai RAG. Modul ini akan membedah konsep di balik cara kerja RAG, apa saja bagian penyusunnya, dan detail implementasi yang penting untuk dipertimbangkan agar aplikasi AI yang dikembangkan menggunakan RAG bisa berjalan secara efektif dan efisien.

Setelah mempelajari modul ini, anda diharapkan dapat:

![Target Pembelajaran](https://lms.sdmdigital.id/pluginfile.php/635466/mod_page/content/10/Screenshot%202026-01-21%20at%2013.49.28.png)
*Visualisasi Target Pembelajaran Modul 5*