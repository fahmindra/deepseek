---
slug: modul-12-2
title: modul-12-2
---

Berikut adalah isi konten dari subtopik 8.2 yang telah dirapikan ke dalam format Markdown sesuai hirarki tanpa mengurangi isinya:

---

# 8.2 Tinjauan Benchmark Akademis (Kapan Menggunakannya)

Jika di 8.1 kita membahas cara menilai *output* satu per satu (metrik), di sini kita akan membahas cara menilai kemampuan total model.

Bayangkan Anda ingin merekrut karyawan. Anda tidak hanya melihat satu jawaban *interview*. Anda melihat nilai ijazah, hasil tes IQ, atau sertifikat keahlian.

Di dunia LLM, "ijazah" dan "tes IQ" ini disebut **Benchmark**.

Sebelum melangkah lebih jauh, penting untuk membedakan antara **Metrik Evaluasi** (yang kita bahas di 8.1) dengan **Benchmark**:

*   **Metrik Evaluasi** adalah **"Alat Ukur"** atau rumus (seperti BLEU, ROUGE, atau *Faithfulness Score*) yang digunakan untuk menilai kualitas **satu** jawaban spesifik.
*   **Benchmark** adalah **"Lembar Soal Ujian"** atau *dataset* terstandarisasi yang berisi ribuan pertanyaan dan kunci jawaban, yang dirancang untuk menguji kemampuan model secara menyeluruh menggunakan metrik-metrik tersebut.

Meskipun subtopik ini berfokus pada **Benchmark Akademis** (kumpulan soal publik yang dibuat oleh peneliti, seperti MMLU untuk pengetahuan umum), perlu diketahui bahwa di dunia industri juga terdapat **Benchmark Privat/Internal** (soal ujian spesifik perusahaan) dan **Crowdsourced Benchmark** (seperti *Chatbot Arena* yang berbasis *voting* manusia). Namun, untuk memilih *Base Model* yang tepat di awal proyek, **Benchmark Akademis** adalah acuan standar *de-facto* yang harus kita pahami cara membacanya.

Bagaimana kita tahu bahwa GPT-4 lebih pintar dari Llama-2? Apakah kita harus mengobrol dengan keduanya selama 100 jam? Tentu tidak.

Komunitas peneliti AI telah membuat serangkaian "Ujian Nasional" terstandarisasi. Setiap ujian dirancang untuk mengukur aspek kecerdasan tertentu: apakah model ini jago matematika? Apakah dia bisa koding? Apakah dia tahu fakta sejarah?

Kumpulan skor dari ujian-ujian inilah yang Anda lihat di **Leaderboards** (seperti *Open LLM Leaderboard* di Hugging Face).

Sebagai *AI Engineer*, Anda mungkin jarang **menjalankan** benchmark ini sendiri (karena mahal dan butuh ribuan soal). Namun, Anda **wajib bisa membacanya**. Anda harus tahu benchmark mana yang relevan dengan *use-case* bisnis Anda agar tidak salah pilih model. Jangan memilih model yang jago **MMLU** (pengetahuan umum) jika Anda butuh model untuk *Coding Python* (HumanEval).

Di subtopik ini, kita akan membedah benchmark paling populer dan memahami apa yang sebenarnya mereka ukur.

## 8.2.1 The Big Three: MMLU, GSM8K, dan HumanEval

Meskipun ada ratusan benchmark, ada tiga yang menjadi standar *de-facto* saat ini untuk mengukur "General Intelligence".

### 1. MMLU (Massive Multitask Language Understanding)
*   **Apa itu:** Tes pengetahuan umum dan akademis paling komprehensif. Terdiri dari 57 subjek, mulai dari Matematika SD, Sejarah AS, Hukum, Fisika, hingga Kedokteran.
*   **Mengukur:** **Pengetahuan Dunia (*World Knowledge*)** dan kemampuan memecahkan masalah umum.
*   **Format:** Pilihan Ganda (*Multiple Choice*).
*   **Kapan Dilihat:** Saat Anda butuh model generalis yang "tahu segalanya" (seperti ChatGPT).

### 2. GSM8K (Grade School Math 8K)
*   **Apa itu:** Kumpulan 8.500 soal cerita matematika tingkat SD.
*   **Mengukur:** **Multi-step Reasoning** (Penalaran Bertahap). Model tidak bisa sekadar menghafal jawaban; ia harus melakukan logika "A lalu B lalu C".
*   **Contoh:** "Natalia punya 5 klip. Dia memberi 2 ke adiknya. Berapa sisa klipnya?"
*   **Kapan Dilihat:** Saat aplikasi Anda membutuhkan logika yang kuat, rantai berpikir (*Chain of Thought*), atau operasi aritmatika.

### 3. HumanEval (dan MBPP)
*   **Apa itu:** Kumpulan masalah pemrograman (Python) di mana model harus melengkapi fungsi (*function completion*).
*   **Mengukur:** **Kemampuan Coding**.
*   **Kapan Dilihat:** Saat Anda membangun *Copilot* untuk programmer atau agen yang harus menulis skrip SQL/Python.

Meskipun skor tinggi pada MMLU atau GSM8K menunjukkan bahwa model memiliki kecerdasan intelektual (IQ) yang tinggi, kecerdasan saja tidak cukup untuk penerapan di dunia nyata. Sebuah model bisa sangat pintar matematika namun sekaligus sangat rasis, bias, atau suka mengarang fakta dengan penuh percaya diri. Oleh karena itu, kita memerlukan serangkaian pengujian terpisah untuk mengukur "Kecerdasan Emosional" dan integritas model.

## 8.2.2 Benchmark Keamanan, Kebenaran, dan Akal Sehat

Dalam ranah *Responsible AI*, kita tidak hanya bertanya "Apakah model **bisa** menjawab?", tetapi "Apakah model menjawab dengan **jujur** dan **aman**?". Berikut adalah benchmark standar untuk mengukur dimensi ini.

### 1. TruthfulQA: Mengukur "Imitative Falsehoods"
*   **Masalah:** LLM dilatih dari data internet. Internet penuh dengan miskonsepsi umum (misal: "Jika Anda menyentuh anak burung, induknya akan membuangnya"). Model cenderung meniru (*mimic*) kesalahan manusia ini karena statistik kemunculannya tinggi.
*   **Mekanisme Tes:** TruthfulQA berisi 817 pertanyaan yang dirancang khusus untuk memancing model agar menjawab dengan mitos atau kesalahpahaman umum tersebut.
*   **Tujuan:** Mengukur seberapa baik model bertahan pada **fakta ilmiah** melawan godaan statistik untuk mengikuti mitos populer. Model yang "jujur" akan menjawab: "Tidak ada bukti ilmiah bahwa induk burung menolak anaknya karena bau manusia," alih-alih mengiyakan mitos tersebut.

### 2. HellaSwag: Mengukur "Commonsense Reasoning" (Akal Sehat)
*   **Masalah:** Model seringkali jago menghafal fakta ensiklopedia tapi gagal memahami fisika dunia nyata yang sederhana bagi manusia.
*   **Mekanisme Tes:** Model diberikan sebuah konteks kejadian sehari-hari (misal: "Seseorang sedang menyetrika baju, lalu..."), dan diminta memilih kelanjutan kalimat yang paling masuk akal dari 4 pilihan. Pilihan salah biasanya gramatikal tapi absurd secara fisik (misal: "...lalu setrika itu berubah menjadi air").
*   **Tujuan:** Menguji apakah model benar-benar "memahami" situasi fisik atau hanya memprediksi kata berdasarkan frekuensi. HellaSwag sangat sulit bagi model lama (BERT) tapi mulai terpecahkan oleh model modern (GPT-4).

### 3. RealToxicityPrompts: Mengukur Keamanan (Safety)
*   **Tujuan:** Mengukur risiko model menghasilkan ujaran kebencian, rasisme, atau konten seksual (*toxicity*).
*   **Mekanisme:** Memberikan model potongan kalimat awal (*prompts*) yang sedikit provokatif atau ambigu, lalu melihat apakah model melanjutkannya dengan kalimat yang sopan atau justru menjadi *toxic*. Ini krusial untuk *Brand Safety* sebelum model dirilis ke publik.

Melihat deretan skor tinggi pada MMLU, TruthfulQA, dan HellaSwag di *Leaderboard* publik tentu sangat mengesankan dan membangun kepercayaan diri kita terhadap kemampuan model. Namun, di balik angka-angka fantastis tersebut, terdapat sebuah risiko integritas yang sering kali tidak terlihat oleh mata telanjang. Bagaimana kita bisa yakin bahwa model benar-benar "memahami" soal ujian tersebut, dan bukan sekadar "mengingat" jawaban yang tidak sengaja ia baca selama pelatihan?

## 8.2.3 Masalah "Data Contamination" (Kebocoran Soal)

Melihat skor tinggi pada benchmark publik seringkali menyesatkan karena adanya risiko **Data Contamination**.

**Apa itu Data Contamination?**
*Data Contamination* adalah fenomena di mana data pengujian (soal ujian seperti MMLU atau GSM8K) secara tidak sengaja **termasuk ke dalam data latihan (*training data*)** model selama proses *pre-training*.

**Mengapa Ini Bisa Terjadi? (Padahal sudah ada Split Train/Test?)**
Dalam *Machine Learning* tradisional, kita memiliki satu file CSV yang kita bagi manual (80% Train, 20% Test). Kebocoran jarang terjadi kecuali ada kesalahan manusia.
Namun, dalam pelatihan LLM, datanya adalah **seluruh internet** (triliunan token).

*   **Sifat Open Source:** Benchmark seperti MMLU atau HumanEval diunggah ke GitHub atau Hugging Face agar bisa diakses publik.
*   **Web Crawling:** Model dilatih menggunakan *crawler* yang menyedot data internet secara otomatis (seperti CommonCrawl). *Crawler* ini seringkali tidak bisa membedakan antara "halaman web biasa" dan "halaman GitHub yang berisi kunci jawaban soal MMLU".
*   **Diskusi Online:** Manusia sering mendiskusikan soal-soal sulit dari benchmark di forum seperti Reddit atau StackOverflow. Jika model dilatih pada data forum tersebut, ia secara tidak langsung "membaca" soal dan jawabannya sebelum ujian.

Akibatnya, model tersebut **sudah pernah melihat soal ujiannya**.

**Contoh:** Bayangkan soal matematika GSM8K: "Natalia punya 5 klip..."
*   **Tanpa Kontaminasi:** Model harus melakukan penalaran logika (*reasoning*) untuk menghitung jawabannya.
*   **Dengan Kontaminasi:** Model hanya melakukan **pencocokan pola ingatan (Memorization)**. Ia tidak "menghitung"; ia hanya "melengkapi kalimat" karena ia pernah melihat teks persis soal tersebut di data latihnya.

**Mengapa Ini Penting?** Ini merusak validitas evaluasi. Kita tidak bisa lagi membedakan apakah model tersebut benar-benar **cerdas (Generalization)** atau hanya **hafal mati (Memorization)**.

Oleh karena itu, jangan pernah percaya 100% pada satu angka *leaderboard*.

Solusi terbaik untuk evaluasi yang jujur adalah menggunakan soal ujian yang **belum pernah ada di internet**, yang akan kita bahas di subtopik selanjutnya.

Karena kita tidak bisa menjamin bahwa benchmark publik bersih dari kontaminasi, satu-satunya cara untuk benar-benar menguji kemampuan model pada kasus bisnis kita adalah dengan membuat "Ujian Privat" sendiri.

Kumpulan data uji rahasia inilah yang disebut **Golden Set**.

---

### Mini Quiz
*(Konten interaktif tersedia pada platform LMS)*

---

> **Refleksi Singkat**
>
> Anda sedang memilih model *Open Source* untuk membangun **Asisten Analis Keuangan** yang tugas utamanya adalah membaca laporan laba rugi dan menghitung rasio pertumbuhan. Anda melihat dua model di Leaderboard:
>
> **Model A:** Skor MMLU Tinggi (Pengetahuan Umum luas), Skor GSM8K Rendah (Logika Matematika lemah).
>
> **Model B:** Skor MMLU Sedang, Skor GSM8K Sangat Tinggi.
>
> Berdasarkan pemahaman Anda tentang apa yang diukur oleh benchmark tersebut, model mana yang akan Anda pilih sebagai kandidat utama, dan mengapa skor "Pengetahuan Umum" (MMLU) mungkin kurang relevan untuk tugas menghitung rasio keuangan?