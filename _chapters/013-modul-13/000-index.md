---
slug: modul-13
layout: part
---
Berikut adalah isi konten yang telah dirapikan ke dalam format Markdown sesuai dengan hirarki aslinya:

# Pengantar Modul 1

**Untuk mendapatkan info update lainnya seputar pelatihan, peserta dapat bergabung di grup kelas telegram pada link berikut:**
[https://s.komdigi.go.id/DTA_LLM_2026](https://s.komdigi.go.id/DTA_LLM_2026)

Atau Scan Barcode berikut:
*(Gambar Barcode Diabaikan)*

---

![Banner LLM Advance](https://lms.sdmdigital.id/pluginfile.php/635521/mod_page/content/13/llm%20advance%201.png)

### Modul 3.1 Observability Sistem LLM (LLMOps)

Sebuah sistem tidak akan selalu berjalan mulus di production. Masalah seperti bug, error, lonjakan latency, atau kegagalan mendadak bisa muncul kapan saja dan langsung memengaruhi kinerja serta reliabilitas sistem. Untuk itu, sistem monitoring digunakan untuk mengawasi kondisi sistem melalui metric fungsional maupun operasional, lengkap dengan dashboard dan alerting untuk mengawasi, mendeteksi dan merespons kegagalan secara mudah dan cepat.

Berbeda dengan software tradisional, sistem berbasis LLM melibatkan proses yang tidak deterministik dengan berbagai tahap tersembunyi. Sistem RAG, misalnya, tidak hanya mengembalikan jawaban, tetapi perlu melakukan querying, retrieval, re-ranking, dan integrasi hasil pencarian untuk mendapatkan jawaban akhir. Sistem agentic melewati proses reasoning/planning, pemanggilan tool, serta transisi state yang berulang.

Setiap proses / komponen ini menerima suatu input dan menghasilkan suatu output perantara yang menjadi input untuk bagian proses lainnya. Nilai perantara ini atau disebut dengan intermediate values yang bisa memberikan insight mengenai bagaimana suatu sistem bekerja, terutama untuk melakukan inspeksi dalam proses debugging untuk mencari penyebab sistem melakukan kesalahan.

![Arsitektur Sistem LLM](https://lms.sdmdigital.id/pluginfile.php/635521/mod_page/content/13/image20.jpg)
**Gambar 1. Sistem berbasis LLM yang kompleks tersusun atas berbagai komponen, dengan nilai-nilai intermediate yang merepresentasikan proses yang dilakukan sebuah sistem.**

Pada sistem seperti ini, mengawasi metric saja tidak cukup. Ketika ada gangguan, diagnosanya memerlukan kemampuan menelusuri jejak setiap nilai perantara ini seperti keputusan, hasil pencarian, penggunaan tool, maupun transisi state. Nilai-nilai ini tidak akan muncul dengan sendirinya tanpa secara sengaja disimpan dan diberi struktur yang baik untuk memudahkan akses. Tidak hanya perlu dimonitor, sebuah sistem juga perlu didesain agar bisa diamati dengan mudah dan reliabel. Sifat ini dikenal dengan istilah observability.

Dalam mempersiapkan infrastruktur monitoring serta merancang sistem yang observable, ada beberapa pertanyaan yang perlu dijawab. Apa saja yang perlu dipikirkan dalam merancang sistem yang mudah diamati? Apa saja nilai yang perlu dimonitor? Bagaimana caranya menangkap keseluruhan alur logika dari sebuah sistem agar mudah diperiksa?

Melalui modul ini, Digiers diharapkan dapat:

![Tujuan Pembelajaran](https://lms.sdmdigital.id/pluginfile.php/635521/mod_page/content/13/image14.png)
**Gambar 2. Target kompetensi dan tujuan pembelajaran Modul 3.1**