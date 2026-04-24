---
slug: modul-8-5
title: modul-8-5
---

# 4.5 Prinsip Desain Aplikasi LLM

---

Dalam melakukan desain aplikasi, selain memastikan keberjalanan aplikasi tersebut dengan mengatur LLM yang digunakan, prompt, state management, serta lainnya dalam melakukan orkestrasi berbagai komponen penyusun aplikasi AI, ada beberapa constraint atau prinsip yang berbasis kebutuhan real world yang perlu diperhatikan dalam merancang sebuah aplikasi AI. Beberapa constraint ini yang disebut dengan istilah *non-functional requirements* dapat mempengaruhi beberapa pilihan, mulai dari model yang digunakan, arsitektur sistem, hingga perlunya optimasi untuk memenuhi constraint ini.

## 4.5.1 Demo vs Real World

Apa yang membedakan antara membangun aplikasi untuk demo dengan aplikasi real world?

Ketika membangun demo, concern utamanya adalah mendemonstrasikan bahwa aplikasi yang dikembangkan memiliki kapabilitas yang diharapkan. Digunakan skenario yang relatif lebih ideal serta dibuat berbagai macam asumsi agar bisa berfokus pada implementasi fungsionalitas dasar dari aplikasi. Akan tetapi begitu aplikasi ini dibawa ke tahap pengembangan yang sebenarnya, banyak faktor yang tiba-tiba harus dipikirkan:

*   Ada requirement latency agar user nyaman menggunakan aplikasinya (UX)
*   Ada keterbatasan budget sehingga cost perlu ditekan agar marjin bisnisnya masuk akal
*   Skala penggunaanya besar dan berubah-ubah, perlu bisa di-scale dengan efektif

Beberapa faktor ini mungkin tidak memiliki hubungan langsung dengan fungsionalitas aplikasi yang dikembangkan, tapi faktor-faktor ini secara tidak langsung membatasi ruang gerak pengambilan keputusan yang bisa berimbas ke pilihan teknis yang berkaitan dengan implementasi aplikasi.

## 4.5.2 Latency, Cost, Quality

Dalam membangun sebuah sistem, ada tiga pilar utama yang menjadi constraint pengembangan sebuah sistem, yaitu latency, cost dan quality.

### Latency
Latency berkaitan dengan **waktu yang dibutuhkan aplikasi untuk menjalankan suatu tugas**, misalnya waktu total untuk menghasilkan jawaban dan durasi pemrosesan data. Latency secara langsung mempengaruhi persepsi pengguna mengenai *responsiveness* dari aplikasi, yaitu ukuran seberapa "tanggap" aplikasi ini dalam menjalankan kebutuhan pengguna dilihat dari kacamata pengguna. Tentunya pengguna tidak ingin menggunakan aplikasi apabila terkesan “lambat” dan tidak tanggap.

Perlu dipahami bahwa latency bisa dipengaruhi banyak faktor dan pada praktiknya tidak selalu diukur pada sebuah proses yang berjalan secara terisolasi. Latency juga perlu diukur ketika sistem / aplikasi sedang “sibuk” mengerjakan task lainnya, karena pada praktiknya beberapa aplikasi AI dibangun untuk melayani banyak task secara paralel. Ketika sistem sedang sibuk, ada waktu “tunggu” yang menjadi faktor tambahan yang berkontribusi ke latency yang lebih tinggi. Semakin sibuk sebuah sistem, semakin lama latencynya karena setiap task perlu “mengantri” gilirannya diproses. Diperlukan arsitektur serta infrastruktur yang memadai agar “antrian” ini bisa diproses dengan lebih efektif, misalnya dengan menjalankan duplikat dari sistem untuk mendistribusikan task yang perlu dikerjakan ketika loadnya tinggi.

### Cost
Cost berkaitan dengan **biaya pengembangan aplikasi** serta **biaya operasional aplikasi** yang diperlukan untuk mengoperasikan aplikasi tersebut. Bisnis ingin mendapatkan keuntungan, sehingga sudah ada perhitungan yang menentukan batasan cost yang bisa ditoleransi untuk mendapatkan margin keuntungan yang sesuai target. Cost dapat bersumber dari LLM yang digunakan, layanan lainnya yang digunakan, biaya infrastruktur, dan biaya pengembangan.

Umumnya cost merupakan salah satu constraint utama dalam pengembangan aplikasi AI. Cost mempengaruhi banyak dimensi, mulai dari jenis LLM yang bisa digunakan, pilihan antara menggunakan layanan API atau self hosting, hingga waktu pengembangan aplikasi. Selain menjadi faktor untuk menentukan keputusan teknis maupun non-teknis yang harus diambil selama pengembangan aplikasi, cost juga menjadi salah satu faktor penentu apakah secara bisnis aplikasi yang dikembangan bisa menghasilkan keuntungan. Perlu dipastikan biaya operasionalnya tidak melebihi revenue yang didapatkan, karena tentunya akan rugi karena pada ujungnya pengembangan aplikasi AI biasanya dilakukan untuk memenuhi suatu kebutuhan bisnis.

### Quality
Quality tidak hanya mengukur seberapa **tepat** respon yang diberikan model, tapi juga **aspek reliabilitas** dan **konsistensi** sistem dalam memberikan respon. Sistem yang akurat tidak akan berguna kalau sering **down** atau **mengalami gangguan**. Pengguna juga tidak akan suka apabila sistem memberikan output yang tidak konsisten sehingga tidak reliabel untuk penggunaan yang serius.

Pada umumnya, pertimbangan mengenai ketiga pillar di atas sangat bergantung pada use case. Contohnya, untuk use case yang tidak real time seperti pemrosesan dokumen berkala, latency tidak sepenting kasus lain seperti chatbot customer support yang perlu respon cepat. Setiap domain dan use case memiliki toleransinya masing-masing terhadap latency dan quality, sehingga AI Engineer perlu memilih mana yang bisa dikorbankan mengingat constraint yang dimiliki dengan cara memilih tradeoff yang tepat.

## 4.5.3 Metrik

### Latency
Dalam konteks LLM, ada beberapa metrik latency yang perlu diukur:
*   **TTFT (Time to First Token)**
*   **TPOT (Time Per Output Token)**
*   **End-to-end latency**
*   **Latency per komponen**

**TTFT** mengukur waktu yang diperlukan LLM dari mulai memproses input sampai **mulai** menghasilkan output. Kenapa TTFT penting? Walaupun memerlukan waktu yang lebih lama untuk menghasilkan output lengkap, semakin cepat ada output yang keluar, semakin cepat bisa ditampilkan langsung secara berangsur-angsur seiring diprosesnya semua output hingga selesai. Bagi user, ini memberikan ilusi kalau sistemnya “responsif”, memberikan kesan yang sama seperti keterangan “...sedang mengetik” yang sering ditemui pada aplikasi chat seperti whatsapp.
Istilah yang digunakan untuk menyebut ilusi ini adalah *“perceived responsiveness”* yang merupakan salah satu kunci meningkatkan UX.

**TPOT** mengukur waktu rata rata untuk menghasilkan setiap token. Kalau TTFT mengukur **berapa lama hingga LLM mulai mengetik**, TPOT mengukur **kecepatan LLM “Mengetik” jawaban**. TTFT yang cepat perlu diimbangi dengan TPOT yang tidak jauh berbeda agar memberikan kesan output yang mengalir secara cepat dan stabil.

**End to end to end latency** mengukur seberapa lama **waktu yang dibutuhkan sebuah aplikasi mulai dari menerima input hingga selesai menghasilkan output**. Metrik ini sangat relevan baik untuk use case yang real time dan interaktif maupun untuk workflow otomatis yang dilakukan berkala untuk memastikan semua pemrosesan sudah selesai sebelum jadwal pemrosesan selanjutnya dimulai.

**Latency per komponen** sesuai namanya mengukur latency untuk setiap komponen yang berada di dalam sebuah aplikasi LLM. Latency ini bisa menjadi indikator yang penting apabila ada komponen tertentu yang berjalan terlalu lama di atas batas wajarnya. Selain itu apabila digabungkan dengan end to end latency, latency per komponen dapat memberikan gambaran bagian mana yang menjadi bottleneck (persentase waktunya signifikan dibandingkan dengan total waktu pemrosesan) yang bisa menjadi target optimasi.

### Cost
Untuk cost, umumnya dihitung beberapa metrik:
*   **Cost per successful task / request**
*   **Cost per session**

**Cost per successful task** mengukur cost menjalankan sistem untuk melakukan sebuah task hingga berhasil. Pada praktiknya, akan sering terjadi kegagalan sehingga sistem perlu mengulang task tersebut. Makin sering mengulang, makin besar costnya.

Di sisi lain, untuk aplikasi yang perlu menyelesaikan suatu task dalam sebuah sesi, dihitung **cost per session**. Semakin cepat aplikasi dapat menyelesaikan task atau masalah tersebut, maka semakin murah costnya, begitupun sebaliknya.

Kedua metrik ini mengukur *operational cost* dari aplikasi yang dijalankan. Masih ada faktor lain misalnya cost yang berasal dari pengembangan untuk membeli tooling yang dibutuhkan maupun membayar gaji developer selama proses pengembangan.

Menghitung dan memperkirakan cost dengan baik perlu dilakukan untuk menentukan beberapa keputusan, terutama keputusan seperti build vs buy. Maksudnya adalah ada kalanya lebih baik menggunakan layanan berbayar dari pihak ketiga dibandingkan menghabiskan waktu untuk mengembangkannya sendiri, karena selain bisa menghemat biaya pengembangan, bisa jadi secara operasional masih lebih masuk akal menggunakan layanan berbayar ini.

### Quality
Ada dua dimensi utama yang diukur dari kualitas:
*   Metrik performa untuk kinerja individual
*   Ukuran reliabilitas keseluruhan dari sebuah sistem

Kalau metrik performa mengukur seberapa baik sistem yang dikembangkan menjalankan tugasnya, reliabilitas sistem mengukur seberapa konsisten sistem dapat memberikan performa terbaiknya.

Selain mengukur akurasi dan metrik performa lainnya dari LLM seperti safety, harmlessness dan lain sebagainya yang sudah disebutkan sebelumnya, untuk reliabilitas sistem secara end to end ada beberapa metrik kualitas yang perlu diukur, beberapa di antaranya adalah:
*   Uptime
*   Konsistensi
*   Failure rate

**Uptime** menghitung persentase waktu dalam sebuah periode dimana aplikasi siap merespon, tidak sedang down atau mengalami gangguan. **Konsistensi** mengukur seberapa mampu sistem memberikan output yang sama untuk input yang sama. **Failure rate** mengukur seberapa sering aplikasi atau sistem gagal menjalankan sebuah task.

## 4.5.4 Trade-off dalam Desain Aplikasi AI

Dalam melakukan desain aplikasi, sangat tidak realistis mengharapkan kita bisa memenuhi tiga pilar di atas sekaligus, yaitu aplikasi yang sangat cepat, sangat murah dan sangat akurat dan reliabel. Pilihan ini disebut sebagai tradeoff.

### Latency vs Cost
Kedua pilar ini tidak berhubungan secara langsung, tapi juga bisa saling mempengaruhi. Misal kita ingin mencapai sebuah ukuran kualitas tertentu. Ketika menjalankan LLM yang digunakan, kita bisa memilih untuk menambah resource untuk menjalankan LLM dengan latency yang lebih rendah. Konsekuensinya adalah cost yang lebih besar. Sebaliknya, untuk menghemat cost, kita bisa memilih untuk menggunakan resource yang lebih kecil, tapi imbasnya adalah latencynya malah bertambah.

Di level sistem/aplikasi, kita bisa menggunakan hardware yang lebih baik hingga tingkatan tertentu serta menghabiskan waktu development lebih banyak untuk membuat sistem optimal yang latencynya lebih rendah, tetapi memerlukan cost extra.

### Cost vs Quality
Akurasi maupun reliabilitas sistem bisa meningkat dengan cara:
*   Menggunakan model yang lebih bagus
*   Menghabiskan uang untuk infrastruktur yang matang (overprovisioning)
*   Merancang pipeline atau workflow yang kompleks
Tapi peningkatan ini juga pasti membutuhkan cost yang lebih tinggi.

### Latency vs Quality
Kualitas yang lebih baik bisa dicapai dengan menggunakan model yang lebih bagus maupun melakukan berbagai macam pemrosesan extra untuk meningkatkan kualitas respon. Misalnya:
*   Prompt yang lebih panjang dan detil
*   Memisahkan sistem menjadi banyak langkah yang spesifik
*   Menggunakan lebih banyak sumber informasi
*   Arsitektur dan pemrosesan yang lebih advanced

Sebaagus apapun kualitas hasilnya, ada titik tertentu dimana latencynya sudah tidak bisa ditoleransi oleh user. Begitupun sebaliknya, secepat apapun modelnya, pasti ada titik dimana akurasi sudah tidak bisa ditoleransi.

## 4.5.5 Real-time vs Batch Processing

Tidak semua aplikasi AI membutuhkan interaksi cepat. Banyak skenario yang justru berjalan secara terjadwal, mengumpulkan data dalam jumlah besar, lalu memprosesnya sekaligus. Skenario ini disebut batch processing.

### Perbedaan karakteristik antara real time dan batch processing

**Real-time processing**
*   Fokus pada respon cepat (TTFT dan end-to-end latency rendah).
*   Biasanya memproses satu request saja per waktu secara online.
*   Bisa mentoleransi sedikit kesalahan jawaban, lebih prioritas kecepatan.
*   *Contoh:* customer support bot, real-time coding assistant.

**Batch processing**
*   Memproses banyak data sekaligus secara offline.
*   Latency tidak terlalu penting; compute efficiency dan akurasi malah jadi prioritas.
*   Cocok untuk workload besar yang perlu dilakukan secara berkala.
*   *Contoh:* data labeling otomatis harian, content filtering, document summarization berkala.

![Gambar 40. Ilustrasi perbedaan real-time-processing vs batch processing](https://lms.sdmdigital.id/pluginfile.php/635459/mod_page/content/18/image48.jpg)
**Gambar 40.** Ilustrasi perbedaan real-time-processing vs batch processing.

### Perbandingan Profil Engineering

| Aspek | Real Time | Batch |
| :--- | :--- | :--- |
| **Penggunaan Resource** | Random, bisa idle | Minim kondisi idle karena sekaligus memproses banyak data |
| **Throughput vs Latency** | Prioritas latency | Prioritas throughput |
| **Toleransi error** | Lebih tinggi, karena interaktif dan ada mekanisme pengawasan | Lebih rendah, karena berjalan otomatis dengan pengawasan minim |

### Menggabungkan batch dan real time processing
Banyak aplikasi yang dikembangkan menggabungkan keduanya. Bagian yang secara langsung berinteraksi dengan pengguna diimplementasi secara real time, sedangkan di balik layar dilakukan batch processing secara berkala untuk menyiapkan data konteks.

## 4.5.6 Scalability: latency vs throughput

**Scalability** adalah kemampuan sistem untuk menangani beban yang meningkat tanpa mengorbankan performa secara signifikan.

*   **Vertical scaling** (menambah kapasitas mesin): Bisa membuat sistem lebih cepat (mengurangi latency), tapi punya batas maksimal bottleneck.
*   **Horizontal scaling** (menambah deployment sistem): Menambah kapasitas pemrosesan sehingga lebih banyak task bisa diproses bersamaan. Ini meningkatkan **throughput** (jumlah input yang diproses dalam rentang waktu tertentu).

*Analogi:* Vertical scaling adalah mengganti kapak dengan gergaji mesin. Horizontal scaling adalah mengajak banyak teman menebang pohon bersama-sama.

## 4.5.7 Versioning

Versioning mempermudah pengujian perubahan atau melakukan rollback. Hal yang bisa ditrack versinya: Kode, Data, Pipeline pemrosesan, dan Prompt.

| Kategori | Tool Umum | Tujuan |
| :--- | :--- | :--- |
| **Versioning data & korpus** | DVC | Versioning dokumen sumber sebelum di-embed |
| **Versioning prompt & template** | LangFuse/LangSmith | Versioning prompt, memory, dan pola percakapan LLM |
| **Versioning Kode** | Git | Kode, prompt, dan konfigurasi |
| **Versioning logic & agent** | Git / repo internal | Strategi reasoning, definisi tools, dan rule-based logic |

## 4.5.8 Traceability + Monitoring

**Tracing** berarti menyimpan rekaman detail tentang bagaimana sebuah LLM menghasilkan jawaban (input, reasoning, tool call, guardrails). Sangat berguna untuk debugging.

**Monitoring** fokus ke performa sistem secara keseluruhan (kecepatan, biaya API, error rate, kualitas jawaban). Berguna untuk pengawasan jangka panjang di tahap production.

### Tools Tracing
| Tool | Kegunaan |
| :--- | :--- |
| LangSmith | Merekam langkah reasoning, prompt, history retrieval |
| PromptLayer | Menyimpan dan merekonstruksi prompt & pemanggilan API |
| Humanloop | Analisis percakapan, prompt, dan metrik performa |
| W&B Prompts | Mencatat hasil pemanggilan LLM & metadata-nya |
| Custom logger | Logging custom sesuai kebutuhan trace data |

### Tools Monitoring
| Tool | Kegunaan |
| :--- | :--- |
| MLFlow | Monitor performa sistem berdasarkan versi prompt/kode/data |
| W&B | Mirip seperti MLFlow |
| Prometheus + Grafana | Dashboard monitoring custom |

## 4.5.9 Caching

**Caching** adalah strategi menyimpan hasil output suatu proses agar ketika input yang sama diberikan, output bisa langsung digunakan tanpa dihitung ulang.

Tingkatan Caching:
*   **Response caching**: Menyimpan response untuk request berulang.
*   **Semantic caching**: Memetakan response untuk kelompok request yang mirip secara semantik.
*   **KV caching**: Berjalan di tingkat inferensi LLM untuk efisiensi pemrosesan.

![Gambar 41. Ilustrasi caching](https://lms.sdmdigital.id/pluginfile.php/635459/mod_page/content/18/image15.jpg)
**Gambar 41.** Ilustrasi caching.

### Semantic caching
Menggunakan representasi "fitur" (embedding) untuk membandingkan kedekatan request baru dengan yang ada di cache. Membutuhkan tuning threshold agar meminimalkan *false negative* dan *false positive*.

### KV Cache (Key-Value Cache)
Berjalan pada proses inferensi LLM dengan menyimpan nilai **Key** dan **Value** dari mekanisme attention yang sudah dihitung sebelumnya.

![Gambar 42. Ilustrasi perhitungan nilai Key, Value, dan Query](https://lms.sdmdigital.id/pluginfile.php/635459/mod_page/content/18/image41.jpg)
**Gambar 42.** Ilustrasi perhitungan nilai Key, Value, dan Query untuk menghitung attention weight.

![Gambar 43. K-V cache](https://lms.sdmdigital.id/pluginfile.php/635459/mod_page/content/18/image45.jpg)
**Gambar 43.** K-V cache digunakan untuk menyimpan nilai K-V yang sudah dihitung sebelumnya agar tidak perlu dihitung ulang saat menghitung output berikutnya.

![Gambar 44. K-V Cache comparison](https://lms.sdmdigital.id/pluginfile.php/635459/mod_page/content/18/image18.jpg)
**Gambar 44.** K-V Cache hanya menghitung nilai K-V yang belum tersimpan.

**Keuntungan menggunakan KV Cache:**
Membuat inferensi LLM jauh lebih cepat, terutama pada sekuens panjang (bisa puluhan kali lipat).

**Desain prompt untuk KV Cache:**
Letakkan bagian prompt yang statis/tidak banyak berubah di awal prompt agar nilai K-V yang sudah dihitung di awal bisa dimaksimalkan penggunaannya sebelum mencapai bagian dinamis.

---

### Mini Quiz
*(Elemen interaktif H5P di LMS)*

---

> **Refleksi Singkat**
>
> Membangun aplikasi AI di dunia nyata adalah soal mengelola tradeoff antara latency, cost, dan quality. Bayangkan beberapa skenario berikut:
> *   Asisten toko online
> *   Penyortir email untuk deteksi spam
> *   Verifikasi dokumen pribadi
> *   Bot untuk melakukan riset
>
> Dari berbagai skenario tersebut, apa yang anda perlu utamakan, dan mana yang bisa anda korbankan?