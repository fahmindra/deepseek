---
slug: modul-10
layout: part
---
Berikut adalah isi konten yang telah dirapikan ke dalam format Markdown sesuai dengan hirarki dan instruksi Anda:

***

# ![Logo](https://lms.sdmdigital.id/pluginfile.php/635478/mod_page/content/14/unnamed.jpg)
# Modul 6 Ekstraksi Data Terstruktur Dan Validasi Output

---

Selamat datang di Modul 6!

Perjalanan kita dalam Program Intermediate ini telah membawa kita menelusuri siklus hidup penuh sebuah sistem Al. Di Modul 2.1, kita memulai dengan fondasi pelatihan, mempelajari bagaimana data dipersiapkan dan model dilatih dari nol (Pre-training) hingga diselaraskan (Alignment). Di Modul 2.2 dan 2.3, kita memperdalam kemampuan model tersebut, menyuntikkan pengetahuan baru lewat Continued Pretraining dan mengasah keterampilan spesifik melalui Fine-Tuning yang efisien (PEFT). Di Modul 2.4, kita belajar meletakkan model tersebut dalam arsitektur sistem yang skalabel, memperhitungkan latency, biaya, dan orkestrasi prompt. Dan di Modul 2.5, kita telah melengkapi model dengan "perpustakaan eksternal" melalui sistem RAG (Retrieval-Augmented Generation), memungkinkannya mengakses data terkini yang tidak ada dalam bobot pelatihannya.

Kini, kita memiliki model yang cerdas, terampil, efisien, dan berpengetahuan luas. Namun, saat kita mencoba mengintegrasikan "otak" hebat ini ke dalam aplikasi bisnis nyata (seperti backend web, aplikasi mobile, atau sistem otomatisasi), kita menghadapi masalah fundamental yang sering diabaikan: Inkompatibilitas Bahasa.

Sistem perangkat lunak konvensional (Database SQL, API REST, Frontend React) berbicara dalam bahasa yang kaku dan terstruktur: JSON, XML, atau CSV. Sebaliknya, betapa canggihnya LLM sekalipun, secara alami LLM adalah suatu produk probabilistik yang berbicara dalam bahasa manusia yang cair, penuh nuansa, dan tidak terstruktur.

Jika Anda meminta LLM mengekstrak data pelanggan dari sebuah email, dan ia menjawab dengan sopan: "Tentu! Berdasarkan email di atas, nama pelanggannya adalah Budi dan umurnya 25 tahun," jawaban ini tidak berguna bagi sistem backend Anda. Aplikasi Anda akan crash karena tidak bisa mem-parse kalimat tersebut ke dalam kolom database.

Modul ini adalah jembatan krusial antara Generative Al dan Software Engineering. Kita akan belajar cara "memaksa" model untuk berhenti menjadi penyair dan mulai menjadi mesin ekstraksi data yang patuh, presisi, dan deterministik. Kita akan membahas teknik prompt engineering untuk struktur, memanfaatkan fitur native seperti JSON Mode, dan yang terpenting, menerapkan lapisan validasi ketat menggunakan library Python modern seperti Pydantic untuk menjamin integritas data sebelum masuk ke sistem hilir (downstream systems).

![Format JSON](https://lms.sdmdigital.id/pluginfile.php/635478/mod_page/content/14/image.png)
**Gambar 1. Format .json**

Sebelum kita mempelajari modul ini, alangkah baiknya kita merefleksikan peran krusial dari dokumentasi dan komunikasi ini sebagai berikut:

*   Dunia perusahaan penuh dengan unstructured data (dokumen PDF kontrak, badan email, transkrip chat CS). Bagaimana cara kita mengubah tumpukan teks tak beraturan ini menjadi baris-baris rapi di tabel SQL tanpa intervensi manusia manual, namun dengan tingkat akurasi yang bisa dipertanggungjawabkan?
*   Kita tahu LLM bersifat stokastik (acak). Jika LLM melakukan kesalahan satu karakter saja dalam format JSON (misal: lupa tanda kurung kurawal penutup }), seluruh pipeline aplikasi bisa gagal. Mekanisme self-healing apa yang bisa kita pasang untuk mendeteksi dan memperbaiki kesalahan format ini secara otomatis?
*   Mengapa sekadar menulis instruksi "Keluarkan JSON ya!" di dalam prompt seringkali tidak cukup untuk model yang lebih kecil, dan bagaimana teknik Constrained Decoding dapat memaksa model untuk hanya menghasilkan token yang valid secara sintaksis?

Digiers, sebelum memulai pembelajaran, mari kita pahami bersama capaian yang akan kita raih. Berikut adalah tujuan pembelajaran yang akan menjadi kompas kita:

![Tujuan Pembelajaran](https://lms.sdmdigital.id/pluginfile.php/635478/mod_page/content/14/image%20%285%29.png)
**Gambar 2. Tujuan Pembelajaran Modul 6**