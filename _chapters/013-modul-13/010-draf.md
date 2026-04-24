---
slug: modul-13-1
title: modul-13-1
---
# 1.1 Observability Sistem LLM (LLMOps)

---

## 1.1.1 Mengenal LLMOps

Sistem software sebaik apapun tidak akan berguna apabila tidak di-deploy ke lingkungan production agar dapat digunakan oleh user. Dalam software engineering, dikenal istilah DevOps, yaitu disiplin ilmu yang menjembatani pengembangan (development) dengan pengoperasian sistem di production (operations) melalui penerapan prinsip dan penggunaan tooling untuk otomasi proses-proses manual yang terlibat. Konsep ini kemudian berkembang pada sistem machine learning (MLOps), dan lebih khusus lagi pada sistem berbasis LLM (LLMOps).

LLMOps berfokus pada memastikan sistem LLM berjalan secara stabil dan reliabel di production, serta memastikan proses pengembangan hingga deployment terintegrasi melalui serangkaian otomasi dan mekanisme feedback untuk mendukung pengembangan dan perbaikan secara berkelanjutan.

Prinsip-prinsip LLMOps terdiri atas:
* Prinsip terkait proses pengembangan (versioning, manajemen data, manajemen eksperimen, dan evaluasi)
* Prinsip terkait operasi sistem (monitoring, feedback loop, otomasi deployment)

Modul ini akan berfokus pada salah satu komponen LLMOps, yaitu monitoring yang juga disertai dengan observability yang penting untuk mengawasi keberjalanan sistem LLM di production dan melakukan diagnosa masalah untuk mengembangkan sistem.

---

## 1.1.2 Monitoring dan observability pada sistem LLM

Apa itu monitoring dan observability?

Ada beberapa ekspektasi dari sebuah sistem yang dikembangkan:
* Sistem tersebut memenuhi SLA (service level agreement) tertentu, misalnya latency dan uptime (persentase waktu sistem tidak down atau error).
* Sistem tersebut memenuhi kebutuhan performa fungsionalnya yang secara langsung berhubungan dengan kebutuhan, misalnya akurasi dan tingkat penyelesaian task.
* Sistem tersebut memenuhi kriteria non-fungsional yang tidak langsung berhubungan dengan kebutuhan, tapi tetap penting untuk memastikan kualitas interaksi yang baik, misalnya aturan berbahasa dan seberapa sering sistem mengarang jawaban (hallucination).

Monitoring bertujuan memastikan sistem LLM berjalan sesuai ekspektasi. Cara kerjanya mirip seperti ketika anda menulis kode lalu memastikan apakah output-nya benar, atau membuat unit test untuk memastikan program yang anda tulis memenuhi spesifikasi. Dalam konteks sistem LLM, berbagai metrik dihitung untuk memastikan apakah respon individualnya akurat maupun memastikan performa sistem secara keseluruhan berada dalam batas wajar. Semua metrik ini kemudian disimpan dan divisualisasi dalam sebuah dashboard, sementara bagian sistem lain yang disebut alerting akan memberi peringatan jika ada nilai yang keluar batas.

![Sistem Monitoring](https://lms.sdmdigital.id/pluginfile.php/635522/mod_page/content/18/image13.jpg)
*Gambar 2. Keseluruhan sebuah sistem monitoring dengan Metric ke-1, metric ke-2, ..., metric ke-N*

Namun, ketika ada sesuatu yang salah, hanya melihat output atau hasil tes tidak cukup. Seperti saat anda melakukan debugging: anda meletakkan print statement pada setiap tahap penting dari program yang sudah ditulis, atau memakai debugger agar bisa "melihat ke dalam" alur eksekusi dan menemukan akar masalah. Inilah peran observability: menyediakan data internal seperti logs, traces, dan metrics sehingga kita bisa memahami apa yang sebenarnya terjadi di dalam sistem dan memperbaiki masalah dengan cepat.

Sistem monitoring hanya memberikan informasi apa yang salah dengan sebuah sistem di luar batas wajar (threshold). Sistem akan membaca angka / value apakah diluar threshold / ambang batas.

Untuk mengetahui mengapa kesalahan bisa terjadi dan mengidentifikasi bagian mana dari sebuah sistem yang bertanggung jawab atas kesalahan tersebut, diperlukan observability.

Observability adalah sebuah sifat di mana cara kerja setiap komponen dalam sebuah sistem bisa diamati, dipahami, dan dianalisis untuk menjelaskan bagaimana suatu sistem memproses sebuah request hingga menghasilkan output. Observability diperlukan pada sistem LLM karena walaupun proses internalnya bersifat abstrak dan sulit diprediksi, kita dapat mengamati nilai intermediate yang berupa input dan output dari setiap tahap dalam alur sistem LLM. Dengan memastikan setiap langkah menerima input yang benar dan menghasilkan output yang benar, kita dapat menelusuri penyebab kesalahan dan memahami jalur keputusan sistem.

Contoh konkritnya:
* Proses di balik perhitungan embedding yang kemudian digunakan untuk melakukan vector search mungkin terlalu abstrak untuk dijelaskan, tetapi kita bisa langsung melihat saja apakah retrieval memberikan informasi yang dicari.
* Logika internal sebuah LLM dalam menentukan pemanggilan tool mungkin sulit dijelaskan, tapi kita bisa langsung menilai apakah tool yang dipanggil oleh LLM tepat untuk menyelesaikan suatu masalah.

![Monitoring vs Observability](https://lms.sdmdigital.id/pluginfile.php/635522/mod_page/content/18/image6.jpg)
*Gambar 3. Ilustrasi perbedaan monitoring dan observability*

Monitoring dan observability saling melengkapi satu sama lain untuk memastikan suatu sistem LLM bisa berjalan di lingkungan production dengan baik. Monitoring mendeteksi ketika terjadi suatu kesalahan, dan observability memberikan petunjuk bagaimana seorang developer dapat mengatasi masalah tersebut.

Bagaimana caranya memberikan sistem kemampuan observability?

---

## 1.1.3 Implementasi observability melalui metrics, logs, dan traces

Sebuah sistem LLM perlu dilengkapi dengan berbagai fungsionalitas khusus agar memiliki observability yang baik. Dalam industri, dikenal tiga pilar utama dari observability, yaitu metrics, logs, dan traces. Cara berpikirnya mirip seperti saat kamu melakukan debugging: kamu tidak hanya mengecek hasil akhir, tapi juga butuh jejak langkah-langkah internal agar tahu bagian mana yang bermasalah.

Setiap pilar memiliki perannya masing-masing:
* **Metrics** mengukur apakah suatu sistem atau komponennya memenuhi kriteria tertentu. Dalam sistem LLM, metrics dihitung pada tingkat operasional (contoh: latency), bisnis (contoh: jumlah token atau biaya API), fungsional (contoh: akurasi jawaban), dan kualitas (contoh: tingkat halusinasi, toxicity). Ini mirip seperti hasil unit test atau pengecekan output otomatis, memberi tahu kita apakah output dari sistem LLM benar.
* **Logs** merekam detail peristiwa (event) yang terjadi saat sistem berjalan. Misalnya dalam sistem RAG, log mencatat proses retrieval serta hasilnya. Ini setara dengan menaruh print statement di banyak titik untuk melihat nilai-nilai penting saat kode dijalankan.
* **Traces** adalah rangkaian peristiwa yang sudah direkam log, disusun secara runut sehingga kita bisa melihat alur lengkap eksekusi sistem. Ini seperti penggunaan debugger yang memperlihatkan urutan langkah program secara jelas.

![Hubungan Pilar Observability](https://lms.sdmdigital.id/pluginfile.php/635522/mod_page/content/18/image3.jpg)
*Gambar 4. Hubungan antara metrics, logs, dan traces dalam observability*

Selama sistem berjalan, observability mencatat berbagai peristiwa dalam bentuk log, lalu menyusunnya menjadi trace. Ketika sebuah metrik menunjukkan anomali, developer dapat menelusuri kembali apa yang terjadi melalui trace tersebut untuk menemukan sumber masalahnya.

### **Metrics**

Selain metrik evaluasi yang mengukur task performance fungsional (akurasi jawaban, akurasi pemilihan tool, recall pada retrieval) dan semantik (hallucination rate, faithfulness, relevance, toxicity) pada keseluruhan maupun sebagian sistem yang sudah dibahas secara intensif pada modul 12 (Evaluasi), kita juga mengenal bentuk metrik lainnya yang perlu diukur, yaitu metrik ekonomi (cost, penggunaan token) dan metrik operasional sistem (latency, throughput, error rate) yang sudah dipelajari pada modul 8 (Prinsip desain aplikasi LLM).

**Tabel 1. Berbagai jenis metrik yang diukur pada sistem LLM**

| Jenis Metrik | Tujuan | Contoh Konkrit |
| :--- | :--- | :--- |
| Operasional | Mengukur kesehatan infrastruktur sistem | Latency, throuhput |
| Bisnis / ekonomi | Mengendalikan cost | Cost per sesi, total penggunaan token |
| Performa | Memastikan kinerja penyelesaian masalah yang reliabel | Akurasi jawaban, relevansi, toxicity |

Metrik-metrik ini bisa berasal dari pengukuran langsung melalui instrumentasi deterministik, terutama metrik operasional dan ekonomi yang bisa dengan mudah diukur secara langsung, hingga metrik yang memerlukan pemrosesan teks tidak terstruktur menggunakan LLM-as-a-judge seperti kebanyakan metrik performa. Metric ini dihubungkan dengan sebuah sistem monitoring yang akan digunakan untuk mengawasi bagaimana metrik-metrik ini berubah seiring berjalannya sistem.

### **Logs**

Dalam sistem software tradisional, log mencatat sebuah nilai yang disebut artifact dalam bentuk data terstruktur. Dalam sistem LLM, artifact yang disimpan berupa data terstruktur (ex: state, request pemanggilan tool) maupun data teks dalam bentuk abstrak (ex: reasoning, hasil summarization, prompt hasil render sebuah template). Biasanya sebuah log menyimpan informasi tambahan seperti session ID dan timestamp untuk memudahkan identifikasi dan pengelompokannya. Sebagian besar bahasa pemrograman memiliki fungsional built-in untuk melakukan logging, termasuk Python.

Log merupakan inti dari observability sebuah sistem. Bisa dibilang keseluruhan proses lainnya yang dilakukan dalam mengimplementasi observability adalah bagaimana caranya menstruktur log dan merencanakan strategi logging agar nanti bisa disusun kembali menjadi trace yang digunakan untuk melakukan analisis. Contoh log paling sederhana adalah melakukan print nilai-nilai yang ingin dicatat. Dalam sistem production, log disimpan secara jangka panjang dengan cara ditulis ke sebuah file atau disimpan ke dalam sebuah database. Contoh di bawah menunjukkan log yang ditulis menggunakan library logging dari Python.

```text
INFO:root:Event: new chat Chat ID: c0cf247a36c0423c8f3e7c9dba2e9060 Content: Halo, saya ingin mengecek status pengiriman pesanan saya dengan ID 1112231233        
INFO:root:Tool check_delivery_status was called with order_id 1112231233
INFO:root:Event: new response ChatID: c0cf247a36c0423c8f3e7c9dba2e9060 Response: Tentu, pesanan Anda dengan ID 1112231233 saat ini berstatus **Dalam Pengiriman**
dan sedang disortir di DC Cakung. Perkiraan waktu tiba adalah tanggal **31-12-2025**.
INFO:root:Event: new chat Chat ID: f3a9d10fbdd84041a3e5fd60396dcfce Content: Halo, saya ingin mengecek keranjang belanja saya dengan user ID Ivan#19827
INFO:root:Tool check_shopping_cart was called with user_id Ivan#19827
INFO:root:Event: new response ChatID: f3a9d10fbdd84041a3e5fd60396dcfce Response: Halo Ivan#19827!


Berikut adalah isi keranjang belanja Anda:


*   **Sepatu Lari Babibas** (ID: 1)
    *   Warna: Merah muda cerah
    *   Ukuran: 48
*   **Kaus Kaki Performa Bortusten** (ID: 22)
    *   Warna: Kuning cerah
    *   Ukuran: XL
*   **Tali Sepatu Anti-selip Becs** (ID: 111)
    *   Warna: Cyan


Apakah ada hal lain yang bisa saya bantu?
INFO:root:Event: new chat Chat ID: 1ffb1efb84444df3b424e465abf1906c Content: Halo, apa kabar? Siapa kamu?
INFO:root:Event: new response ChatID: 1ffb1efb84444df3b424e465abf1906c Response: Halo! Saya adalah asisten belanja Anda. Saya di sini untuk membantu Anda dengan pertanyaan terkait belanja.
```

Dalam tujuan observability, log digunakan untuk mencatat sebuah event yang terjadi dalam suatu proses disertai dengan metadata yang berhubungan dengan event tersebut. Dalam modul ini, bisa diasumsikan setiap kali sebuah log disebut adalah log yang mencatat event, bukan informasi saja.

Contohnya, dalam proses retrieval pada sebuah sistem RAG, log mencatat event-event berikut:
* Query diberikan ke sistem retrieval
    * Yang dicatat: query
* Dihitung embedding query
    * Yang dicatat: embedding
    * Metadata: model embedding yang digunakan, ukuran embedding
* Vector search dilakukan
    * Yang dicatat: hasil vector search
    * Metadata: versi DB, distance function yang digunakan, konfigurasi top-K, threshold, dsb

### **Traces**

Traces merupakan komponen level tertinggi dari observability yang memberikan gambaran keseluruhan alur pemrosesan dan struktur dari cara kerja sebuah sistem. Trace dibangun dengan cara mengelompokkan event yang dicatat dalam log berdasarkan urutan dan atributnya menjadi satuan proses yang disebut sebagai span. Sebuah Trace terdiri dari banyak span yang memiliki struktur tertentu.

Setiap span terdiri atas:
* Nama atau bentuk identifikasi lainnya
* Detail mengenai waktu, misal durasi total maupun waktu awal dan akhir dari span
* Sekelompok log terstruktur yang menyusun keseluruhan peristiwa yang terjadi dalam satu span
* Atribut dari span tersebut

Kalau log memberikan informasi mengenai event individual, span menyatukan setiap peristiwa tersebut menjadi satu kesatuan proses, disertai dengan konteks apa yang diproses dan untuk tujuan apa proses tersebut terjadi.

Atribut sebuah span menyimpan metadata dari operasi yang direpresentasikan, seperti:
* Versi template dari prompt
* Informasi mengenai model LLM yang digunakan
* Nilai top-K yang digunakan, model embedding yang dipakai, metode pengukuran kedekatan, serta threshold kedekatan pada proses retrieval

![Trace Langfuse](https://lms.sdmdigital.id/pluginfile.php/635522/mod_page/content/18/image10.png)
*Gambar 5. Contoh sebuah trace eksekusi aplikasi LLM yang bisa dilihat melalui dashboard Langfuse*

Dalam sebuah trace, span bisa memiliki struktur yang nested untuk memberikan hierarki yang jelas dari setiap operasi yang berjalan dalam sebuah trace. Dalam contoh di atas, bisa dilihat bahwa ada beberapa span yang terbagi menjadi beberapa span. Misalnya span chat terbagi menjadi model, tool, dan model lagi, mengikuti alur eksekusi LLM yang berpindah dari satu proses ke yang lainnya.

Sebagai contoh, misalnya untuk sistem hybrid RAG, kita memiliki alur berikut:
* Pemanggilan RAG
    * Generate query
        * Buat prompt untuk generate query
        * Panggil LLM
    * Retrieval
        * Generate embedding
        * Vector search
        * Filtering metadata
        * Hitung distance
        * Ambil top-K
    * Re-ranking
    * Generate jawaban
        * Buat prompt akhir
        * Panggil LLM

Dimana secara umum satu pemanggilan RAG (yang menjadi span tersendiri) bisa dibagi menjadi 3 span utama, yaitu Generate query, Retrieval, dan Generate jawaban yang masing-masing dibagi lagi menjadi proses individualnya.

---

## 1.1.4 Standar untuk observability: OpenTelemetry

Banyak cara untuk menulis dan menstruktur log, menyusunnya menjadi trace, dan mengekspornya untuk kebutuhan visualisasi. Tanpa keseragaman, setiap alat observability akan memakai formatnya sendiri dan developer akan kerepotan setiap kali ingin mengganti atau menggabungkan berbagai solusi. Karena itu, industri membuat standarisasi yang menetapkan konvensi bersama sehingga semua pihak "berbicara dalam bahasa yang sama". Dengan begitu, integrasi antar-solusi menjadi jauh lebih mudah dan tidak perlu membuat format khusus setiap kali berpindah tooling.

Standar yang paling populer digunakan adalah OpenTelemetry atau disebut dengan OTel. OpenTelemetry (OTel) adalah standar industri untuk observability, menyediakan cara yang jelas dan konsisten untuk menangkap, memproses, dan mengirimkan data observasi dari seluruh sistem.

OpenTelemetry (OTel) bukan sekadar alat logging biasa. Ia adalah standar industri yang menyediakan protokol universal untuk menangkap, memproses, dan mengirimkan data (Logs, Metrics, Traces) dari satu layanan ke layanan lain tanpa terputus.

**Contoh Konkrit: "Misteri Checkout yang Gagal"**

Bayangkan Anda memiliki aplikasi E-Commerce dengan arsitektur Microservices. Saat user menekan tombol "Beli Sekarang", satu klik tersebut sebenarnya memicu reaksi berantai di belakang layar:
1. Frontend memanggil API Gateway.
2. API Gateway memanggil Inventory Service (cek stok).
3. Inventory Service memanggil Payment Service (potong saldo).
4. Payment Service menghubungi Bank Eksternal.

Tanpa OTel, jika transaksi gagal di tahap ke-3, Anda buta. Log Frontend bilang "Error 500", tapi tidak tahu kenapa.

**Bagaimana Protokol OTel Bekerja di Sini?** Di sinilah peran protokol Context Propagation (penyebaran konteks) bekerja:
1. **Penyuntikan ID (Injection):** Saat user klik "Beli", OTel di Frontend otomatis membuat Trace ID unik (misal: xyz-123).
2. **Propagasi Header (The Protocol):** Saat Frontend memanggil Inventory, Trace ID xyz-123 disisipkan ke dalam HTTP Header request tersebut.
3. **Penyambungan (Span):** Inventory Service menerima request, melihat ada ID xyz-123, dan mencatat aktivitasnya di bawah ID yang sama. Ia lalu meneruskan ID tersebut ke Payment Service.
4. **Sentralisasi:** Semua layanan mengirimkan laporan (spans) mereka ke OTel Collector menggunakan protokol OTLP (OpenTelemetry Protocol).

Hasilnya: Di dashboard monitoring (seperti Jaeger/Grafana), Anda bisa melihat satu garis lurus perjalanan ID xyz-123. Anda bisa langsung menunjuk: "Oh, transaksi gagal tepat di detik ke-2.5 saat Payment Service mencoba menghubungi Bank, karena Timeout."

Konsep intinya dimulai dari instrumentasi, yaitu proses menambahkan kemampuan bagi sistem untuk menghasilkan sinyal observasi seperti metrics, logs, dan traces (serta baggage untuk pelacakan lintas layanan). OTel juga mendefinisikan semantic conventions agar setiap sinyal mengikuti istilah dan struktur yang sama, serta protokol OTLP (Open Telemetry Protocol) untuk mengekspor sinyal tersebut ke berbagai backend observability. Semua data ini bisa dikumpulkan lewat collector OTel dan diteruskan ke platform seperti Datadog, Prometheus, Grafana, Jaeger, Tempo, Honeycomb, hingga dashboard custom.

Meskipun OTel menyediakan library untuk melakukan instrumentasi secara manual, dalam praktiknya banyak framework dan tools modern yang sudah menyertakan instrumentasi berbasis OTel secara built-in. Jadi developer tidak perlu memikirkan format khusus atau integrasi rumit. Developer hanya perlu memastikan tooling yang digunakan mengikuti standar OTel, dan seluruh ekosistem observability dapat saling bekerja sama tanpa perlu penyesuaian manual.

---

### **Mini Quiz**
*(Konten Interaktif H5P)*

> **Refleksi Singkat**
>
> Kita mengetahui sebagian besar LLM menerima, memproses, dan memberi output berupa teks yang mungkin masih mudah untuk disimpan dalam bentuk log. Bayangkan anda ingin membuat logging untuk aplikasi yang juga memproses data dalam bentuk lain, misalnya gambar atau suara. Kira-kira pendekatan apa yang sebaiknya dilakukan?