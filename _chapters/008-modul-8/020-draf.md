---
slug: modul-8-2
title: modul-8-2
---

# 4.2 Context Engineering dan Orkestrasi Prompt Lanjutan

---

Dalam modul sebelumnya sudah dipelajari bagaimana caranya mendapatkan akses ke LLM dan bagaimana caranya menjalankan LLM tersebut untuk interaksi sederhana. Dalam industri, perilaku LLM yang diinginkan biasanya kompleks dan memerlukan data yang berbeda-beda untuk setiap pemanggilan. Selain itu tingkatan kompleksitas perilaku yang diinginkan mencakup banyak aspek selain fungsionalitas utama: gaya bahasa, format output, tools yang tersedia, cara menyelesaikan masalah, dan aspek lainnya. Setiap aspek ini perlu dipertimbangkan dalam merangkai prompt.

---

## 4.2.1 Apa saja yang Biasanya Dimasukkan ke dalam Prompt?

Utamanya sebuah prompt terdiri dari instruksi dan konteks, seperti yang telah dijelaskan sebelumnya. Tapi apa saja yang masuk ke dalam instruksi dan konteks ini?

### Instruksi
Perilaku yang diinginkan dari LLM bisa dicapai dengan memberinya instruksi. Selain instruksi yang secara langsung berkaitan dengan apa yang harus dilakukan LLM, sistem yang kompleks juga perlu mendefinisikan bagaimana LLM menjalankan instruksi tersebut. Apa saja yang umum diberikan melalui instruksi?

*   **Tugas utama / fungsionalitas utama**
    Tentunya ini adalah bagian utama dari instruksi LLM. Bagian ini menjelaskan apa yang perlu dicapai LLM, apa inputnya, dan seperti apa output yang diinginkan. Selain itu, detil tertentu misalnya edge case, cara spesifik untuk menjalankan tugas yang diberikan, juga detil penting lainnya juga perlu diberikan agar perilaku LLM semakin terarah.

*   **Format output**
    Output yang diinginkan dari LLM tidak selalu berbentuk teks bebas. Bisa jadi ada format yang diinginkan, baik struktur teksnya maupun agar output dari LLM memiliki format yang bisa diproses selanjutnya, misalnya JSON. Format ini perlu di-specify melalui instruksi.

*   **Petunjuk untuk melakukan reasoning**
    Beberapa LLM memiliki kemampuan reasoning. Singkatnya, kemampuan ini memberikan LLM ruang untuk "berpikir" dalam bentuk output intermediate yang dihasilkan LLM sebelum menghasilkan output sebenarnya. Reasoning berguna agar jawaban akhir model lebih terarah karena sifat autoregressive dari LLM itu sendiri. Hasil reasoning akan menjadi bagian dari konteks yang digunakan LLM untuk menghasilkan jawaban akhirnya. Prinsipnya mirip seperti orang yang sedang mengerjakan soal matematika akan mencoret-coret dulu perhitungannya di kertas coretan sebelum menuliskan jawaban akhirnya di lembar jawaban.
    Kalau instruksi mengenai task yang harus dilakukan mendeskripsikan apa yang harus dilakukan, bagian ini akan lebih berfokus ke bagaimana caranya mencapai hal tersebut dengan memecah masalah yang dihadapi menjadi bagian kecil yang satu per satu dipecahkan untuk mempersiapkan diri sebelum memberikan jawaban akhir.

*   **Deskripsi tools dan aturan penggunaannya**
    Beberapa LLM juga memiliki kemampuan menggunakan tool. Deskripsi mengenai tool tersebut termasuk detail mengenai input apa saja yang dibutuhkan, apa fungsi dari tool tersebut, hingga aturan penggunaan tool tersebut, seperti misalnya hanya boleh digunakan dalam skenario tertentu, perlu diberikan sebagai bagian dari instruksi.

*   **Gaya / aturan dalam merespon**
    Selain mendeskripsikan apa dan bagaimana LLM harus melakukan tugas yang diberikan, perlu juga diatur bagaimana cara LLM menyajikan jawabannya. Bagian ini mendetilkan gaya bahasa serta aturan-aturan tertentu, misalnya harus menggunakan bahasa Indonesia, harus sopan, tidak boleh menggunakan kata-kata kasar, dan aturan sejenis yang mengontrol seperti apa respon yang diberikan LLM.

*   **Safety**
    Selain mengatur fungsional, perlu juga antisipasi adanya failure case maupun penggunaan yang malicious. Bagian ini akan memberikan instruksi apa yang perlu dilakukan ketika ada sesuatu di luar skenario normal. Misalnya ketika informasi yang diberikan tidak lengkap atau tidak jelas, coba klarifikasi ke user. Contoh lainnya yang lebih ekstrim adalah jika user meminta melakukan sesuatu yang berpotensi berbahaya atau mencelakakan pihak lain, LLM diinstruksikan untuk menolak menjalankan perintah tersebut, tapi tetap dengan sopan.

### Konteks
Kalau instruksi mendeskripsikan "cara memasak", maka konteks adalah bahan makanannya. Sebuah LLM memerlukan konteks yang tepat sesuai dengan skenario penggunaan yang diharapkan agar bisa memberikan jawaban yang cocok. Ada beberapa jenis konteks yang bisa dikonsumsi LLM:

*   **Data input yang berkaitan dengan task**
    Misal LLM diminta memproses suatu input, tentunya perlu diberikan input tersebut sebagai konteks untuk diproses. Di sisi lain, bisa jadi walaupun suatu informasi atau input tidak diproses secara langsung, input ini menyediakan informasi tambahan yang berguna untuk mengubah atau mengatur pemrosesan informasi yang dilakukan.

*   **Memory / state**
    Untuk beberapa use case, misalnya chat, ada skenario tertentu yang bergantung pada apa yang sebelumnya sudah dilakukan. Masalahnya adalah LLM sifatnya stateless, setiap pemanggilan LLM tidak menyimpan informasi yang sebelumnya sudah dilakukan, sehingga informasi ini perlu disediakan dari sisi developer.
    Contoh paling sederhana adalah sejarah percakapan untuk sebuah chatbot. Ada kalanya user perlu bisa merefer ke sesuatu yang sudah sebelumnya dibahas dalam percakapan tersebut. Untuk itu, developer perlu menyimpan informasi ini dan mengupdatenya secara berkala, lalu memberikannya ke LLM ketika dibutuhkan. Informasi mengenai interaksi sebelumnya ini umumnya disebut sebagai memory.
    Di sisi lain, ada banyak use case di mana LLM perlu "ingat" kondisi terkini dari suatu proses agar bisa mengambil keputusan yang tepat. Kondisi inilah yang kita sebut state. Misalnya, kita membuat sebuah bot yang bisa menjalankan beberapa task dalam sebuah to-do list. Setelah bot menyelesaikan satu task, daftar ini harus diperbarui (task dicoret atau ditandai selesai). Versi terbaru dari to-do list inilah yang kemudian diberikan kembali ke LLM sebagai konteks untuk menentukan langkah berikutnya. Sebuah state adalah informasi tentang kondisi saat ini yang harus diketahui model untuk melanjutkan pekerjaannya, dan state perlu diberikan kembali ke dalam sebagai konteks setiap kali ada perubahan.

*   **Pengetahuan internal**
    LLM dilatih menggunakan data yang mencakup berbagai macam topik sebagai bagian dari pretraining. Pengetahuan ini menjadi bagian dari model yang dilatih dan akan digunakan secara internal oleh LLM untuk menjawab suatu pertanyaan. Walaupun bisa berguna, kadang pengetahuan internal ini bermasalah. Misalnya kita ingin membuat chatbot yang dapat menjawab pertanyaan berdasarkan suatu dokumen yang diberikan. Ketika LLM tidak menemukan informasi yang dibutuhkan, bisa jadi dia menjawab dengan menggunakan pengetahuan internalnya. Hal ini sulit dikontrol dan bisa jadi berbahaya karena selalu ada potensi terjadi halusinasi, sehingga perlu ditangani misalnya dengan cara diberi instruksi untuk menjawab "tidak tahu" kalau informasi tersebut tidak ada dalam dokumen yang diberikan.
    Di sisi lain, pengetahuan internal LLM bisa jadi mengurangi beban / kebutuhan untuk menyediakan konteks eksternal yang menyederhanakan proses pengembangan aplikasi. Akan tetapi perlu ada mekanisme yang baik untuk memastikan pengetahuan ini valid dan bisa diverifikasi.

---

## 4.2.2 Bagaimana Caranya Menyusun Prompt yang Baik?

Walaupun tidak ada aturan baku yang menentukan bagaimana prompt harus disusun, ada beberapa best practice yang umum diterapkan dalam industri. Best practices ini bukanlah aturan baku yang harus dituruti, melainkan sekedar guideline yang bisa menjadi referensi untuk menyusun prompt dengan baik.

### Tentukan Dulu Hasil yang Baik itu Seperti Apa
Iterasi adalah kunci. Daripada over-engineering memikirkan segalanya harus sempurna dari awal, lebih baik buat prompt yang ok dulu, lalu iterasi dan terus kembangkan promptnya hingga hasilnya sesuai harapan. Agar iterasinya efektif, tentu perlu patokan yang jelas apakah hasilnya menjadi lebih baik atau lebih buruk.
Beberapa jenis kriteria yang bisa diamati:
*   Format, gaya bahasa, panjang dari output
*   Apakah jawabannya tepat atau tidak, apakah konteks yang diberikan digunakan atau tidak
*   Apakah ada halusinasi atau tidak
*   Apakah ada output yang tidak seharusnya (bahasa toxic, informasi di luar pertanyaan)

### Gunakan Struktur dan Bahasa yang Mudah Dimengerti oleh LLM
Meskipun banyak LLM memiliki kemampuan multilingual yang baik, umumnya LLM banyak dilatih menggunakan dataset dalam bahasa inggris. Sehingga agar lebih efektif, sebaiknya dalam membuat prompt lebih baik menggunakan bahasa yang menjadi mayoritas dataset mereka, yaitu bahasa inggris. Kalau memang perlu membuat prompt dalam bahasa lain, perlu dilakukan benchmark / evaluasi untuk melihat apakah ada perbedaan performa yang berarti antara keduanya.
Selain itu, memberikan struktur yang jelas pada prompt akan memudahkan instruksi dan konteksnya diproses oleh LLM.

### Berikan Struktur pada Prompt
Struktur ini bisa dideskripsikan menggunakan bermacam format, bisa XML, bisa Markdown, bisa JSON. Contohnya:

```markdown
# Peran
Ambillah peran seorang analis sosial media. Kamu akan diminta melakukan analisis terhadap komentar pada media sosial mengenai suatu brand.

# Instruksi
Kamu akan diberikan list teks yang berasal dari komentar media sosial pada post suatu brand.
Dari setiap komentar ini, tentukan sentimen apa saja yang bisa diidentifikasi.
Setelah identifikasi, kelompokkan lalu tentukan sentimen keseluruhan mengenai brand tersebut.
Hanya gunakan komentar yang relevan dengan brand tersebut, misalnya kritik, keluhan, atau komentar apapun yang secara langsung ditujukan ke brand.
Jika tidak ada sama sekali komentar yang relevan, semua outputnya harus kosong dan sentimen overall harus bernilai `tidak bisa ditentukan`.

# Petunjuk Menjalankan Instruksi
Sebelum menentukan jawaban akhir, saring dulu komentar yang relevan saja, lalu kelompokkan berdasarkan setiap grupnya yaitu positif, negatif, dan netral.
Tentukan sentimen overall berdasarkan rasio antara jumlah komentar positif, negatif, dan netral.
Kalau semuanya berimbang, artinya sentimennya netral. Nilai `tidak bisa ditentukan` hanya berlaku jika
sama sekali tidak ada komentar yang relevan.
Setelah dikelompokkan, identifikasi komentar-komentar unique saja dari setiap kelompok tersebut, lalu gunakan untuk mengisi list komentar positif, negatif, dan netral pada output.

# Format output
Berikan output dalam bentuk JSON. Ikuti format berikut yang diapit oleh tag <FORMAT></FORMAT>.
Ketika memberikan output, tagnya tidak usah ikut diberikan, cukup JSONnya saja.
Petunjuk atau deskripsi mengenai nilai yang perlu dimasukkan diberikan dalam tag {{}}
<FORMAT>
{
    "sentimen_overall": {{positif/negatif/netral/tidak bisa ditentukan (pilih salah satu)}},
    "komentar_sentimen_positif": [
        {{berikan list komentar positif mengenai brand, harus unique}}
    ],
    "komentar_sentimen_negatif": [
        {{berikan list komentar negatif mengenai brand, harus unique}}
    ],
    "komentar_sentimen_netral": [
        {{berikan list komentar netral mengenai brand, harus unique}}
    ]
}
</FORMAT>

# Aturan merespon
- output hanya berupa JSON saja
- ketika inputnya tidak valid (kosong), berikan output json kosong.
- Semua output harus berupa string yang tidak diformat, plain text saja

# Input
Berikut adalah list komentar dari sosmed yang perlu diproses:

[[ ganti dengan list komentar]]
```

### Instruksi yang Jelas
Instruksi yang diberikan ke LLM harus jelas, setiap langkah atau bagaimana cara mengolah input harus dengan jelas dijabarkan. Mari coba ilustrasikan dengan eksperimen kecil:

![Gambar 22. Hasil memberikan instruksi sederhana ke LLM](https://lms.sdmdigital.id/pluginfile.php/635456/mod_page/content/16/image6.png)
**Gambar 22.** Hasil memberikan instruksi sederhana ke LLM.

![Gambar 23. Hasil setelah instruksi dibuat lebih spesifik](https://lms.sdmdigital.id/pluginfile.php/635456/mod_page/content/16/image49.png)
**Gambar 23.** Hasil setelah instruksi dibuat lebih spesifik.

![Gambar 24. Hasil setelah instruksi diubah + diberi contoh](https://lms.sdmdigital.id/pluginfile.php/635456/mod_page/content/16/image9.png)
**Gambar 24.** Hasil setelah instruksi diubah + diberi contoh.

### Konteks yang Tepat
Sebuah konteks bisa dibagi menjadi dua kategori:
1.  **Input utama yang perlu diproses LLM.** (Contoh: chat user, dokumen rangkuman, foto deskripsi).
2.  **Konteks / informasi tambahan.** (Contoh: JSON schema, dokumen RAG, aturan gaya bahasa).

---

## 4.2.3 Desain Prompt yang "Ergonomis" dan Maintainable

*   **Mudah dimengerti**: Penggunaan penamaan konsisten, menghindari logika terlalu rumit di dalam prompt.
*   **Mudah digunakan**: Menggunakan sistem templating (placeholder) untuk bagian dinamis.
*   **Modular**: Memisahkan system prompt dan user prompt.
*   **Mudah diberikan versi**: Menggunakan version control (git) atau platform khusus (MLFlow/LangFuse).

Contoh implementasi templating di Python:

```python
from jinja2 import Template
from string import Template as StringTemplate

## template python
prompt_template_string = "Terjemahkan kalimat berikut ke dalam bahasa jawa sopan: {kalimat}"
## string template python
prompt_template_string_library = StringTemplate("Terjemahkan kalimat berikut ke dalam bahasa jawa sopan: $kalimat")
## template jinja
prompt_template_jinja = Template("Terjemahkan kalimat berikut ke dalam bahasa jawa sopan: {{kalimat}}")

INPUT_STR = "Pada hari minggu kuturut ayah ke kota"

## Memasukkan input ke template prompt untuk mendapatkan prompt yang siap digunakan
prompt_string = prompt_template_string.format(kalimat=INPUT_STR)
prompt_string_library = prompt_template_string_library.substitute({"kalimat": INPUT_STR})
prompt_jinja = prompt_template_jinja.render(kalimat=INPUT_STR)

print(f"python template string: {prompt_string}")
print(f"python template string (library): {prompt_string_library}")
print(f"jinja: {prompt_jinja}")
```

---

## 4.2.4 Orkestrasi Prompt Lanjutan: Berbagai Macam Trik dan Pola Prompting

### Zero Shot Prompting
Memberikan instruksi secara langsung tanpa contoh.

![Gambar 25. Ilustrasi zero shot prompting](https://lms.sdmdigital.id/pluginfile.php/635456/mod_page/content/16/image4.jpg)
**Gambar 25.** Ilustrasi *zero shot* prompting.

### Few Shot Prompting
Memberikan beberapa contoh pasangan input-output sebelum instruksi utama.

![Gambar 26. Few shot prompting memberikan contoh cara melakukan suatu task selain instruksi dan konteks](https://lms.sdmdigital.id/pluginfile.php/635456/mod_page/content/16/image34.jpg)
**Gambar 26.** *Few shot prompting* memberikan contoh cara melakukan suatu task selain instruksi dan konteks.

### Chain of Thoughts (CoT) Prompting
Memberikan instruksi untuk memecah masalah menjadi langkah-langkah perantara.

```python
# Contoh instruksi CoT
"Sebuah toko menjual 120 buku dalam satu minggu. Penjualan meningkat 25%. Mari kita selesaikan langkah demi langkah."
```

### Self-Consistency
Menghasilkan banyak jawaban untuk input yang sama (dengan temperature > 0), lalu melakukan majority voting.

### Tree of Thoughts (ToT)
Melakukan eksplorasi ruang solusi dengan memvariasikan langkah reasoning dan mengevaluasi setiap cabang.

![Gambar 27. Tree of thoughts](https://lms.sdmdigital.id/pluginfile.php/635456/mod_page/content/16/image13.jpg)
**Gambar 27.** Tree of thoughts.

---

## 4.2.5 Mengenal Context Engineering

Memasukkan terlalu banyak konteks bisa membuat LLM "bingung" (masalah *lost-in-the-middle*) dan memakan biaya komputasi besar. **Context Engineering** adalah proses memilih dan merangkai hanya instruksi dan konteks yang tepat saja ke dalam prompt.

### Prinsip-prinsip Context Engineering:

1.  **Prinsip pertama: Selektif terhadap informasi yang dimasukkan.**
    Menentukan hanya informasi tertentu saja yang dimasukkan dalam suatu waktu secara dinamis.

2.  **Prinsip kedua: Pecah task besar yang kompleks menjadi task kecil yang lebih sederhana.**
    Menguraikan task menjadi workflow yang terdiri dari beberapa pemanggilan LLM spesifik.

![Gambar 28. Memecah task agar instruksi dan konteksnya bisa disebar untuk setiap task spesifik](https://lms.sdmdigital.id/pluginfile.php/635456/mod_page/content/16/image33.jpg)
**Gambar 28.** Memecah task agar instruksi dan konteksnya bisa disebar untuk setiap task spesifik.

3.  **Prinsip ketiga: Kompresi konteks.**
    *   **Rangkuman**: Membuat ringkasan dari sejarah percakapan.
    *   **Retrieval (RAG)**: Memecah konteks besar menjadi bagian kecil dan hanya mengambil paragraf yang relevan saja.

![Gambar 29. Ilustrasi 2 pendekatan kompresi konteks](https://lms.sdmdigital.id/pluginfile.php/635456/mod_page/content/16/image23.jpg)
**Gambar 29.** Ilustrasi 2 pendekatan kompresi konteks (Rangkuman vs Retrieval).

---

### Mini Quiz
*(Elemen interaktif H5P tersedia di LMS)*

---

### Refleksi Singkat
Setelah memahami strategi berbagai teknik untuk merancang *prompt* serta mempelajari motivasi dan aspek-aspek dari *context engineering*, coba anda lakukan eksplorasi untuk mengamati bagaimana berbagai aplikasi LLM melakukan *context engineering*. Salah satu yang paling menarik adalah bagaimana *Claude Code* bekerja. Berdasarkan [blog post ini](https://kirshatrov.com/posts/claude-code-internals), apa yang bisa anda pelajari mengenai cara mereka menstrukturkan instruksi dan konteks?