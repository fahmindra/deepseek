---
slug: modul-13-5
title: modul-13-5
---
# 1.5 Tantangan monitoring sistem LLM di production

---

Monitoring dilakukan untuk mengamati perilaku sistem dengan tujuan memberikan peringatan dini ketika diamati terjadinya penyimpangan dari batas wajarnya. Selain itu, monitoring juga penting untuk memberikan insight mengenai perilaku dari sebuah sistem sepanjang waktu. Untuk sistem LLM pada production, ada beberapa area yang menjadi fokus dari monitoring. Tabel 1 telah menunjukkan apa saja area yang perlu difokuskan untuk diamati.

---

### 1.5.1 Karakteristik evaluasi online vs offline

Istilah online dan offline yang dimaksud di sini bukan merujuk kepada koneksi internet, tapi membedakan antara evaluasi yang dilakukan secara real time saat sistem berjalan melayani live traffic (online) dan evaluasi yang dilakukan secara terpisah secara berkala yang dilakukan setelah mengumpulkan data dan mendapatkan ground truth. Perbedaan karakteristik keduanya bisa dilihat pada tabel berikut:

**Tabel 3. Perbedaan karakteristik evaluasi online dan offline**

| Aspek | Online | Offline |
| :--- | :--- | :--- |
| **Kapan dilakukan?** | Ketika sistem sedang berjalan | Dilakukan secara terpisah |
| **Apakah ada ground truth?** | Tidak tersedia | Umumnya tersedia |
| **Tujuan** | Mendeteksi anomali secepat mungkin | Pengujian performa berkala atau untuk membandingkan kualitas |
| **Kecepatan** | Mendekati real time untuk metrik teknis yang tidak perlu penilaian semantik, delayed untuk metrik yang perlu penilaian semantik | Lambat, biasanya dijadwalkan sendiri karena memang tidak butuh real-time |
| **Contoh** | Mendeteksi halusinasi, respon toxic, mendeteksi latency spike | A/B testing untuk membandingkan performa sistem menggunakan prompt baru dengan versi lamanya |

Kenapa perbedaan ini penting? Dalam sistem production, evaluasi yang dilakukan memiliki kebutuhan yang berbeda-beda. Ada yang perlu dideteksi secepat mungkin (respon toxic, halusinasi) maupun perlu diuji secara berkala untuk menentukan arah pengembangan selanjutnya (akurasi, truthfulness, groundedness, metrik custom). Perlu perhatian khusus terutama untuk metrik-metrik yang perlu dihitung dan diawasi secara online.

---

### 1.5.2 Pengelompokan metrik berdasarkan kompleksitas perhitungannya

Untuk menentukan strategi monitoring yang tepat, metrik perlu dikelompokan dulu berdasarkan kompleksitas perhitungannya. Umumnya bisa dibagi menjadi dua jenis, yaitu metrik operasional dan bisnis yang nilainya bisa dengan mudah diamati / langsung dihitung hanya menggunakan informasi dari setiap request, dan metrik semantik yang memerlukan pemrosesan ekstra untuk mendapatkan nilainya.

#### **Metrik-metrik yang mudah dihitung**

Metrik dalam kategori ini umumnya diisi oleh metrik operasional dan bisnis yang secara langsung bisa diukur, bahkan seringkali juga menjadi bagian dari respon ketika memanggil LLM menggunakan API.

**Tabel 4. Beberapa contoh metrik yang mudah dihitung nilainya**

| Jenis Metrik | Karakteristik Perhitungan | Apa yang diamati? | Strategi untuk alerting | Keterangan |
| :--- | :--- | :--- | :--- | :--- |
| **Latency (TTFT, E2EL, TPOT)** | Individual, Waktu (detik) | Individual + Aggregate (P50, P95, P99) | Real-time threshold, mengamati tren. Contoh: P95 TTFT > 2s | Selain melihat performa individual, juga bisa tracking "kesehatan" sistem dengan melihat seberapa memburuk latency ketika tingkat penggunaannya semakin tinggi |
| **Throughput (Tokens/Sec, RPS)** | Rate | Aggregate | Mengamati tren. Contoh: Drop >20% dalam 10 menit | Mengukur tingkat efisiensi pemrosesan throughput. Penting untuk perencanaan kapasitas dan skalabilitas. |
| **Cost (dolar per request, total penggunaan token)** | Individual, Moneter | Individual + Aggregate | Threshold status dan dinamis. Contoh: Single request >$5 atau Daily cost >$10K, atau tren kenaikan cost | Dihitung berdasarkan jumlah token input dan output dikalikan harga unit per token. Biaya input dan output seringkali berbeda. |

**Mengapa Mudah?**
* Tidak memerlukan analisis konten atau LLM tambahan
* Data tersedia langsung dari sistem (timestamp, token count, API response)
* Bisa dihitung sambil memproses request
* Overhead komputasi minimal

#### **Metrik yang perlu komputasi yang rumit atau dievaluasi offline**

Metrik-metrik ini melibatkan data yang umumnya tidak terstruktur yang perlu dinilai secara semantik. Umumnya metrik seperti ini dihitung menggunakan LLM (LLM-as-a-judge) atau dievaluasi secara berkala oleh pelabel manusia, terutama jika melibatkan bidang yang sensitif. Selain sulit dihitung, proses perhitungannya juga mahal sehingga pemrosesannya perlu dilakukan secara selektif, misal dengan cara melakukan sampling.

**Tabel 5. Jenis metrik yang lebih rumit perhitungannya karena subjektif dan semantik**

| Jenis Metrik | Karakteristik Perhitungan | Apa yang diamati? | Strategi untuk alerting | Keterangan |
| :--- | :--- | :--- | :--- | :--- |
| **Kebenaran (akurasi, task completion rate)** | Statistik, LLM-as-a-Judge | Batching offline, Sampling + Online untuk tool call | Trend-based, Per segmen. Contoh: Penurunan akurasi segmen | Umumnya diukur secara berkala saja batched |
| **Kualitas jawaban (groundedness, hallucination rate, toxicity score)** | Individual, statistik | Batching offline, Sampling + Online untuk yang sensitif / perlu cepat ditindak | Trend-based + Per segmen. Contoh: Tingkat toxic lebih tinggi ketika membahas topik tertentu | Penting secara berkala, tapi umumnya penting untuk dideteksi real-time bahkan kalau bisa sebelum diberikan ke user (lewat guardrails) |
| **Compliance / kepatuhan (validitas format output, refusal rate)** | Statistik / Individual | Trend-based + Per segmen; Sampling + Online untuk yang sensitif / perlu cepat ditindak | Contoh: LLM sering menolak menjawab pertanyaan mengenai sejarah yang sensitif | Tergantung seberapa krusial |
| **Konsistensi (model drift, prompt drift)** | Statistik | Offline | - | Bukan metrik standalone, diukur dengan mengamati tren perubahan metrik lain |

**Mengapa Sulit?**
* Evaluasinya perlu model atau feedback langsung dari manusia (pelabelan maupun interaktif)
* Biaya komputasi tinggi
* Wajib dilakukan secara asinkron agar tidak membebani sistem.
* Akan selalu ada lag antara ketika terjadi penurunan performa dengan ketika sistem monitoring berhasil mendeteksi penurunan tersebut karena waktu pemrosesan yang dibutuhkan untuk melakukan evaluasi.

Berbeda dengan metrik yang mudah dihitung, metrik ini perlu strategi menghadapi volume penggunaan yang besar karena mengevaluasi satu per satu setiap output akan terlalu mahal. Perlu ada strategi untuk menyampel sebagian saja output, misalnya:
* Langsung random sampling saja (berpotensi kehilangan informasi penting)
* Prioritas berdasarkan sinyal tertentu, misalnya:
    * Kategori / segmen pengguna / domain tertentu
    * Feedback langsung dari user, misalnya berdasarkan rating

---

### 1.5.3 Metrik individu vs statistik

Metrik yang penting untuk dihitung secara individu, umumnya juga penting untuk diamati agregasinya. Sedangkan tidak semua metrik yang penting dihitung dalam bentuk statistik penting juga diamati setiap individualnya.

**Kapan memonitor nilai individu saja?**
* Ketika outlier individual memiliki dampak bisnis signifikan
* Ketika masalahnya hanya terjadi pada sebagian kasus saja
* Ketika perlu ada intervensi secara real time

**Kapan memonitor statistik saja?**
* Ketika metriknya memang perlu diukur pada suatu jangka waktu tertentu. Contohnya throughput.
* Ketika yang ingin diamati adalah tren waktu ke waktu
* Ketika pengukuran individual terlalu noisy, justru lebih penting melihat tren

---

### 1.5.4 Bagaimana caranya mengatasi ketiadaan ground truth?

Monitoring dilakukan untuk mendeteksi ketika sistem melakukan kesalahan, agar bisa dicatat atau bisa memberi alert kepada developer, yang artinya dibutuhkan suatu patokan untuk menentukan apakah sesuatu yang salah terjadi. Ketika melakukan evaluasi secara offline, umumnya tersedia ground truth yang menjadi basis untuk menentukan validitas dari perilaku sistem. Dalam melakukan monitoring secara real time, data tersebut tidak tersedia, atau setidaknya belum tersedia secara langsung untuk menilai respon sistem.

Bergantung pada jenis datanya, ada beberapa cara yang bisa dilakukan untuk mengatasi perbedaan ini.

1. **Metrik yang tidak memerlukan ground truth / bisa dihitung secara langsung.** Ini adalah kasus paling sederhana yang utamanya merupakan sifat dari metrik-metrik operasional dan bisnis. Mengevaluasi nilai-nilai metrik tersebut hanya memerlukan data yang bisa langsung diamati dan dihitung berdasarkan apa yang terjadi secara real time.
    * Contoh:
        * Latency bisa langsung dihitung dengan melakukan profiling untuk mengukur waktu yang diperlukan pada tiap bagian sistem.
        * Penggunaan token bisa langsung dihitung dengan cara mengukur panjang sekuens input setelah melalui tokenizer ditambah dengan jumlah token yang dicatat saat melakukan sampling setelah selesai infilling.
2. **Melalui proxy, yaitu nilai perantara yang bisa mengindikasikan kalau ada kesalahan.** Ada dua pendekatan proxy:
    * Berdasarkan feedback dari user secara real time. Contohnya pada UI aplikasi, user bisa memberi rating sesimpel memencet tombol thumbs up jika respon aplikasi membantu, atau memencet tombol thumbs down jika responnya tidak membantu. Respon yang diberikan thumbs down bisa ditandai untuk prioritas pengecekan manual maupun otomatis.
    * Berdasarkan statistik / aturan tertentu. Contoh paling sederhana adalah menetapkan threshold berdasarkan suatu requirement atau berdasarkan running statistic untuk mengukur deviasi dari operasi yang normal.
3. **Melalui perhitungan otomatis.** Beberapa jenis metrik bisa dinilai oleh LLM menggunakan metode LLM-as-a-Judge. Metode ini cenderung mahal kalau perlu dilakukan pada seluruh respon, sehingga perlu sampling atau cara untuk memfilter respon mana yang perlu diproses. Metode ini bisa digunakan ketika suatu metrik bersifat abstrak / semantik, tapi tidak memerlukan domain knowledge yang sangat khusus / sangat spesifik, sehingga bisa diotomasi dengan LLM.
4. **Melalui human-in-the-loop.** Dengan cara menghubungkan LLM ke platform observasi, operator manusia bisa ditempatkan untuk secara langsung untuk melakukan intervensi. Tapi tidak mungkin seorang operator secara manual mengawasi keseluruhan sistem saat berjalan, sehingga idealnya, ada mekanisme filtering / alerting. Metode ini penting untuk metrik yang bisa dihitung secara otomatis, tapi sangat kritis / penting sehingga perlu second opinion dari manusia.
5. **Metrik yang tidak perlu dihitung secara real time.** Beberapa jenis metrik lebih masuk akal diawasi secara berkala dengan cara mengamati tren dari waktu ke waktu. Misalnya metrik seperti akurasi, kalaupun bisa dihitung secara real-time pasti akan berfluktuasi dari respon ke respon. Metrik seperti ini perlu dihitung secara berkala, misalnya setiap 1 bulan sekali atau 1 minggu sekali respon model disampel, dikumpulkan, lalu dilabeli untuk kemudian dievaluasi secara offline.

---

### Mini Quiz
*(Konten Interaktif H5P)*

> **Refleksi Singkat**
>
> Anda sudah melihat bagaimana metrik monitoring dapat dibagi berdasarkan kompleksitas perhitungannya dan bagaimana mengatasi ketiadaan ground truth dalam evaluasi online. Bayangkan Anda bertanggung jawab atas sistem LLM production yang melayani jutaan request per hari dengan budget komputasi yang terbatas untuk melakukan evaluasi, sehingga anda tidak bisa sembarangan menggunakan LLM as a judge untuk segalanya. Jika Anda harus merancang strategi sampling untuk metrik semantik yang mahal (seperti hallucination rate atau toxicity), bagaimana Anda akan menyeimbangkan antara kebutuhan deteksi dini dengan kendala biaya dan komputasi?
>
> Dalam merancang jawabannya, pikirkan hal-hal seperti:
> * Metrik mana yang harus diprioritaskan untuk evaluasi real-time meskipun mahal, dan mana yang cukup di-batch?
> * Bagaimana merancang sistem sampling yang tidak hanya random, tetapi juga intelligent berdasarkan sinyal-sinyal tertentu?
> * Bagaimana mengintegrasikan feedback langsung dari user sebagai proxy untuk mengurangi beban komputasi evaluasi otomatis?
> * Bagaimana menentukan threshold dan alerting yang tepat untuk menghindari false positive yang berlebihan tanpa melewatkan masalah kritis?
>
> Semua pertanyaan ini mengarah pada realitas bahwa monitoring bukan hanya soal teknis mengumpulkan metrik, tetapi juga soal trade-off strategis antara visibilitas sistem, biaya operasional, dan kecepatan respons terhadap masalah.
>
> *Sebelum optimasi dilakukan, telah ditekankan pentingnya memastikan tidak ada "kesalahan minor" seperti embedding tidak konsisten, atau konteks yang tidak masuk ke prompt, atau ada beberapa dokumen yang ternyata tertinggal saat ingestion. Namun dalam sistem nyata, kesalahan minor sering luput karena tidak terlihat di metrik. Menurut Anda, bagaimana cara paling efektif untuk mendeteksi kesalahan minor ini sejak awal?*