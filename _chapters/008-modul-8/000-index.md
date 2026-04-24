---
slug: modul-8
layout: part
---
![Logo Modul](https://lms.sdmdigital.id/pluginfile.php/635454/mod_page/content/9/page%205.png)
*Logo Modul 4*

# Modul 4 Desain Sistem dan Arsitektur Aplikasi LLM

Modul-modul sebelumnya pada program ini sudah mengajarkan berbagai cara mengembangkan LLM baik melalui *training* dari nol maupun melakukan *fine tuning* pada model yang sudah tersedia untuk diadaptasikan agar bisa memenuhi suatu tujuan atau kebutuhan tertentu. Dalam modul 1 program fundamental, anda sudah belajar bahwa seorang *AI Engineer* memiliki tanggung jawab untuk membuat sebuah sistem yang menggunakan sebuah LLM untuk membangun aplikasi yang bisa menyelesaikan suatu masalah.

Sistem yang dibangun perlu bisa melakukan beberapa hal, selain bisa mengatur input dan output dari dan ke LLM, sistem ini juga perlu dipastikan bisa berjalan dengan baik untuk mencapai tujuannya. Selain memilih dan mengembangkan LLM, AI Engineer juga perlu mengatur berbagai konfigurasi lain dari LLM yang termasuk konfigurasi untuk menjalankannya. Untuk mempermudah penyebutan “sebuah sistem yang berbasis LLM”, akan digunakan penyebutan **Aplikasi AI** sepanjang modul ini dan modul-modul selanjutnya.

Sebelum dilanjutkan, ada terminologi yang perlu disepakati dulu. Input yang diproses oleh aplikasi / sistem AI akan disebut sebagai **_request_**. Sedangkan outputnya akan disebut sebagai **_response_**. Sebuah LLM akan diberikan **prompt** yang berisi **instruksi** dan **konteks** untuk menjalankan suatu tugas yang akan disebut dengan istilah **task**. Terminologi ini akan digunakan agar konsisten dengan terminologi yang umum digunakan ketika membahas sebuah aplikasi / sistem yang memberikan suatu layanan.

![Skema Aplikasi LLM](https://lms.sdmdigital.id/pluginfile.php/635454/mod_page/content/9/image22.png)
**Gambar 1.** Aplikasi LLM bisa dipandang sebagai black box yang menerima request dan memberikan response.

Selanjutnya, mari kita refleksikan dulu apa saja yang sudah dipelajari pada modul 1 program fundamental:

*   Anda sudah mempelajari mengenai pengembangan LLM. Pada modul 1, sudah diajarkan bahwa sistem yang dibangun AI Engineer bertanggung jawab mengatur input dan output LLM. Apa yang sebenarnya dilakukan dalam mengatur input dan output ini?
*   Selain mengatur input dan output, AI Engineer juga bertanggung jawab mengatur konfigurasi LLM yang akan digunakan. Setelah mendapatkan sebuah LLM, baik dari pengembangan sendiri maupun menggunakan LLM yang sudah tersedia, konfigurasi seperti apa saja yang perlu dilakukan? Bagaimana bentuk konkritnya untuk mengimplementasi konfigurasi tersebut?
*   Sistem yang dibangun oleh AI Engineer tentunya selain perlu memiliki kinerja yang baik untuk menyelesaikan masalah yang dituju, juga perlu memiliki performa yang baik. Apa saja metrik yang perlu diperhatikan ketika membangun sistem berbasis LLM?

Pada modul ini, Digiers akan diajak “bertukar topi” dari **mengembangkan LLM** ke **mengembangkan sistem yang berbasis LLM**. Kalau pada modul-modul sebelumya mengajak anda merakit “mesin” utama dari sebuah sistem berbasis LLM, modul ini akan mulai mengenalkan anda untuk menjadi “montir” yang menggunakan mesin yang sudah dikembangkan menjadi sebuah “kendaraan” yang akan mengantar anda kepada suatu tujuan. Melalui modul ini, anda diharapkan dapat:

![Tujuan Pembelajaran](https://lms.sdmdigital.id/pluginfile.php/635454/mod_page/content/9/image12.png)
*Visualisasi transisi peran dari pengembang model menjadi pengembang sistem.*