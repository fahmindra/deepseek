---
slug: modul-7
layout: part
---
# Modul 3 Fine Tuning LLM

![Logo Fine Tuning](https://lms.sdmdigital.id/pluginfile.php/635440/mod_page/content/11/fine%20tuning%20logo.jpg)
*Logo Fine Tuning*

---

Di **Modul 2**, kita telah membahas Continued Pretraining (CPT) yaitu sebuah proses menambahkan pengetahuan baru ke dalam model yang membutuhkan data masif dan komputasi yang besar.

Namun, dalam 90% kasus di dunia industri, Anda tidak perlu membuat model Anda "lebih pintar" secara ensiklopedis. Anda hanya perlu membuatnya lebih ahli dalam satu tugas spesifik.

Misalnya, Anda membutuhkan sistem yang mampu mengekstrak data nominal dan tanggal secara presisi dari ribuan faktur vendor yang formatnya tidak baku, atau Anda memerlukan asisten virtual yang tidak hanya "bisa bicara", tetapi secara konsisten meniru gaya bahasa (tone of voice) brand perusahaan yang profesional dan empatik dalam setiap interaksi.

Guna menjembatani kesenjangan antara kemampuan umum model dasar dan tuntutan spesifik aplikasi bisnis tersebut di mana sekadar instruksi pada prompt seringkali tidak konsisten, kita menggunakan teknik Fine-Tuning.

Masalahnya: Model modern sangat besar (7 Miliar hingga 70 Miliar parameter).

Penting untuk membedakan antara tujuan dan metode. "Fine-tuning" adalah proses adaptasinya, namun metode konvensional yang digunakan disebut Full Fine-Tuning (FFT). Dalam FFT, kita memperbarui seluruh (100%) parameter model selama pelatihan. Ini berbeda dengan pendekatan modern yang lebih selektif.

Konsekuensi dari memperbarui semua parameter ini adalah kebutuhan biaya GPU yang masif dan seringkali di luar jangkauan sebagian besar engineer dan startup.

Apakah kita harus menyewa superkomputer hanya untuk melatih model agar bisa membalas chat pelanggan? Jawabannya: Tidak.

Modul ini didedikasikan untuk PEFT (Parameter-Efficient Fine-Tuning). Ini adalah kumpulan teknik revolusioner yang memungkinkan Anda melatih model raksasa pada perangkat keras yang sederhana (bahkan pada satu GPU gratis di Google Colab). Kita akan membedah rumus matematika di balik LoRA dan QLoRA, mengeksplorasi teknik adaptasi ringan lainnya, dan menggunakan tools tercepat saat ini (Unsloth) untuk melakukan eksekusi.

### Refleksi

Sebelum kita mempelajari modul ini, alangkah baiknya kita merefleksikan peran krusial dari dokumentasi dan komunikasi ini sebagai berikut:

*   Jika Anda harus mengubah perilaku model Llama3:70B (yang ukurannya 140GB), apakah masuk akal jika Anda harus menyimpan copy model sebesar 140GB untuk setiap klien yang berbeda? Adakah cara agar kita hanya menyimpan "perubahannya" saja yang berukuran beberapa Megabyte?
*   Secara matematis, ketika kita melakukan fine-tuning, apakah semua miliaran parameter di otak model benar-benar perlu berubah? Atau jangan-jangan perubahan yang krusial hanya terjadi di dimensi yang jauh lebih rendah?
*   Bagaimana kita bisa melatih model yang membutuhkan memori 48GB di atas GPU yang hanya memiliki memori 16GB tanpa crash? (Petunjuk: Kita akan memanipulasi presisi angka).

### Capaian Pembelajaran

Digiers, sebelum memulai pembelajaran, mari kita pahami bersama capaian yang akan kita raih. Berikut adalah tujuan pembelajaran yang akan menjadi kompas kita:

![Tujuan Pembelajaran](https://lms.sdmdigital.id/pluginfile.php/635440/mod_page/content/11/Screenshot%202026-01-09%20at%2016.24.38.png)
*Tujuan Pembelajaran Modul 3*