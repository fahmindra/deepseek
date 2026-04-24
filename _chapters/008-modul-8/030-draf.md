---
slug: modul-8-3
title: modul-8-3
---

# 4.3 Pola Arsitektur Aplikasi AI

---

Pada submodul sebelumnya telah disebutkan kalau aplikasi AI yang cara kerjanya cukup kompleks perlu dipecah menjadi bagian-bagian kecil. Setiap bagian ini memiliki instruksi, konteks, dan instruksinya masing-masing yang dijalankan melalui pemanggilan LLM secara terpisah. Setiap bagian ini menjadi komponen dalam suatu arsitektur yang perlu diorkestrasi agar bisa bekerjasama melakukan sebuah task yang dibutuhkan.

## 4.3.1 Memandang LLM sebagai Fungsi

Untuk memahami orkestrasi LLM secara programatis, kita perlu mengadopsi cara pandang yang lebih teknis: menganggap setiap pemanggilan LLM sebagai sebuah fungsi dalam bahasa pemrograman. Fungsi ini:

*   menerima *input* teks berbentuk apapun (teks mentah, instruksi, JSON, potongan kode, ringkasan data, dsb.)
*   mengembalikan *output* teks berbentuk apapun (teks bebas, jawaban terstruktur, reasoning chain, rencana aksi, dsb.)
*   Sudah menangani bagaimana memproses *output* menjadi format yang bisa dipakai secara programatis
*   Diprogram melalui prompt yang berperan sebagai definisi perilaku fungsi.

![Gambar 30. Memandang LLM sebagai fungsi](https://lms.sdmdigital.id/pluginfile.php/635457/mod_page/content/16/image42.jpg)
**Gambar 30.** Memandang LLM yang sudah blackbox bukan sebagai LLM, tapi sebagai sebuah modul atau fungsi dalam bahasa pemrograman.

Fungsi-fungsi ini lalu kita gunakan dalam sebuah program untuk menulis program seperti biasa. Dengan mengubah mindset **"memproses input dan output LLM yang diberikan instruksi dan konteks tertentu"** menjadi **"menulis program menggunakan fungsi yang dibuat menggunakan LLM"**, kita bisa membangun sistem yang lebih kompleks dengan cara memanfaatkan berbagai macam struktur dalam programming seperti percabangan (if else) dan *looping*.

### Contoh 1: LLM sebagai fungsi parsing
Tanpa LLM, menulis parser untuk teks tidak terstruktur mungkin akan sangat sulit dan rapuh. Dengan LLM, kita cukup memprogram perilaku fungsi lewat prompt:

**Input:** `"Pesanan: 3 bakso, 1 es teh; alamat: jl. merpati 10"`

**Prompt:** `"Ekstrak informasi pesanan dan kembalikan dalam JSON valid. Simpan informasi pesanan pada list dengan key items yang berisi objek dengan name yang menyimpan nama makanan dan qty yang menyimpan jumlah pesanan. Simpan alamat pesanan pada address."`

**Output LLM:**
```json
{
  "items": [{"name": "bakso", "qty": 3}, {"name": "es teh", "qty": 1}],
  "address": "jl. merpati 10"
}
```

Dipandang sebagai fungsi, seakan kita memiliki fungsi "canggih" yang bisa melakukan hal abstrak tanpa perlu peduli detail implementasinya:
```python
order = llm_parse(raw_text)
```

### Contoh 2: LLM sebagai fungsi klasifikasi berbasis instruksi
Tanpa model khusus, kita bisa membuat fungsi klasifikasi hanya dengan prompt:
```python
def classify_sentiment(text):
    return LLM("""
        Klasifikasikan sentiment (positive|neutral|negative).
        Jawab hanya dengan salah satu label.
        Text: {text}
    """)
```
LLM kini berperan sebagai fungsi yang memberikan label ke teks yang menjadi input, tanpa perlu model klasifikasi khusus atau sejenisnya. Fungsi ini bisa dipanggil misalnya untuk melakukan percabangan.

### Contoh 3: LLM sebagai fungsi dalam workflow dengan branching
Misalnya kita ingin workflow:
1.  Meringkas resume
2.  Menentukan apakah kandidat cocok
3.  Jika cocok buat email HR
4.  Jika tidak cocok buat email penolakan

Kita perlakukan semuanya sebagai fungsi LLM:
```python
summary = llm_summarize(resume_text)
decision = llm_decide(summary)
if decision["fit"] == True:
    email = llm_generate_accept_email(summary)
else:
    email = llm_generate_reject_email(summary)
```

Modul ini akan membahas beberapa pola arsitektur yang umum digunakan dalam aplikasi AI.

## 4.3.2 RAG (Retrieval-Augmented Generation)

RAG (retrieval-augmented generation) merupakan salah satu arsitektur yang umum digunakan untuk memberikan informasi eksternal ke sebuah LLM. RAG mengintegrasikan sebuah aplikasi AI dengan sebuah "database" berisi informasi penting yang diperlukan LLM untuk memberikan respon yang tepat untuk suatu *task*. RAG terdiri atas beberapa komponen, yaitu sebuah LLM itu sendiri, sebuah database, dan sebuah *retriever*, subsistem yang berfungsi untuk mencari informasi berdasarkan keyword atau input tertentu.

![Gambar 31. Sistem RAG](https://lms.sdmdigital.id/pluginfile.php/635457/mod_page/content/16/image3.jpg)
**Gambar 31.** Sistem RAG menyuntikkan informasi eksternal melalui *retrieval*, sebuah proses mencari informasi yang paling relevan untuk suatu *task* dalam sebuah database.

Ketika LLM memerlukan informasi untuk memberikan jawaban yang tepat, RAG akan melakukan "pencarian" dokumen yang paling sesuai berdasarkan kriteria tertentu, lalu diberikan ke LLM sebagai konteks untuk menghasilkan jawaban akhir.

RAG merupakan salah satu arsitektur yang paling populer dan paling banyak dipakai, dengan berbagai macam variasi dan teknis yang diterapkan untuk implementasinya. Detail lebih lanjut mengenai RAG akan dibahas pada modul selanjutnya yang didedikasikan untuk RAG.

## 4.3.3 Komposisi Workflow

Untuk task yang rumit dan kompleks, sama seperti membuat sebuah program, ada baiknya memecah task besar tersebut menjadi task-task dengan scope yang lebih kecil, lalu mengorkestrasi pemanggilan LLM untuk masing-masing task ini. Selain membantu manajemen context yang lebih baik, pendekatan ini juga lebih umum digunakan untuk menyusun sistem yang kompleks yang tidak bisa hanya "sekali jalan" saja, memerlukan percabangan.

Sebuah sistem yang terdiri dari orkestrasi banyak task biasanya disebut sebagai sebuah **workflow**. Contoh *workflow* yang paling sederhana adalah **workflow linear** yang membagi sebuah task yang ingin dicapai menjadi langkah-langkah perantara, dimana output setiap langkah menjadi input langkah berikutnya.

![Gambar 32. Workflow linear](https://lms.sdmdigital.id/pluginfile.php/635457/mod_page/content/16/image32.jpg)
**Gambar 32.** Orkestrasi beberapa *task* terpisah secara berurutan untuk mencapai sebuah *task* besar.

Karena diorkestrasi menggunakan pemrograman, kita bisa memanfaatkan struktur yang lebih kompleks yang tersedia dalam pemrograman. Misalnya kita bisa menggunakan **perulangan** untuk memproses lebih dari 1 output ketika LLM memberikan banyak output dalam satu waktu.

![Gambar 33. Orkestrasi looping](https://lms.sdmdigital.id/pluginfile.php/635457/mod_page/content/16/image25.jpg)
**Gambar 33.** Orkestrasi yang memanfaatkan *control flow looping*.

Selain itu, ketika pemrosesan bercabang tergantung nilai dari sebuah intermediate output, kita bisa menggunakan **logika percabangan** pada output LLM untuk menentukan langkah pemrosesan yang selanjutnya perlu dilakukan.

![Gambar 34. Orkestrasi branching](https://lms.sdmdigital.id/pluginfile.php/635457/mod_page/content/16/image31.jpg)
**Gambar 34.** Orkestrasi logika penyelesaian masalah yang dinamis dengan menggunakan percabangan.

Perlu dipahami bahwa mengintegrasikan LLM untuk diorkestrasi secara programatis membutuhkan bentuk data yang bisa dimengerti dan diproses oleh program komputer tanpa memerlukan pemrosesan ekstra untuk memahami datanya. Contohnya, ketika program melakukan percabangan, dibutuhkan nilai tertentu, baik string yang sudah ditentukan apa saja nilainya, atau boolean yang hanya bernilai true dan false, atau suatu angka. Semua nilai ini perlu bisa didapatkan dari output LLM sehingga muncul keperluan agar LLM bisa memberikan output yang terstruktur dalam format yang familiar untuk digunakan dalam pemrograman, misalnya XML atau JSON.

Modul 6 akan membahas lebih lanjut mengenai kapabilitas LLM menghasilkan data terstruktur.

## 4.3.4 Agentic Patterns

Pendekatan workflow memerlukan pengetahuan mengenai proses yang akan dilakukan untuk menciptakan alur pemrosesan disertai dengan peran setiap komponen dalam alur tersebut untuk menyelesaikan suatu task. Akan tetapi untuk masalah yang cukup dinamis, kita tidak bisa mengetahui bentuk alurnya seperti apa. Alur tersebut bahkan bisa berubah-ubah untuk sistem yang lebih kompleks / high level dimana bisa jadi setiap request membutuhkan alur yang berbeda-beda.

Kadang kita perlu bergantung pada LLM untuk "berpikir" sendiri apa saja yang perlu dilakukan untuk menyelesaikan suatu task. Pola ini biasa disebut **agentic pattern**, yaitu membiarkan LLM melakukan planning, eksplorasi dan eksekusi secara mandiri. Pola ini melibatkan memberikan LLM serangkaian "tool" yang bisa dia gunakan disertai dengan petunjuk dasar mengenai pendekatan yang perlu dia lakukan untuk memecahkan sebuah masalah.

![Gambar 35. Ruang solusi agentic](https://lms.sdmdigital.id/pluginfile.php/635457/mod_page/content/16/image26.jpg)
**Gambar 35.** Walaupun ruang solusi sudah diketahui (langkah / tool apa saja yang bisa digunakan menyelesaikan masalah), beberapa jenis domain masalah bisa memerlukan tahap penyelesaian yang berbeda untuk task yang berbeda. Di sinilah agentic pattern bisa membantu.

Untuk menyelesaikan masalah, LLM yang sudah dipersenjatai dengan tool serta petunjuk-petunjuk dasar penyelesaian masalah ini, yang biasanya disebut sebagai **Agent**, akan mengeksplorasi sendiri ruang solusi yang tersedia untuk mencapai tujuan akhirnya. Sebuah **Agent** akan berinteraksi dengan lingkungannya (tools yang tersedia) berdasarkan suatu *state* internal (memory mengenai keberjalanan suatu task) hingga mencapai suatu kondisi berhenti tertentu.

Salah satu pola arsitektur *agentic* paling sederhana adalah berbentuk loop. Sebuah LLM diberikan *tool* untuk melakukan "aksi" dalam bentuk interaksi untuk secara langsung mempengaruhi lingkungannya (misalnya baca/tulis file, membuka website) maupun mendapatkan informasi. Setelah berinteraksi dengan lingkungannya, LLM mendapatkan "feedback" yang akan menentukan aksi apa yang selanjutnya perlu dilakukan hingga dicapai kondisi berhenti tertentu.

![Gambar 36. Pola sederhana agentic](https://lms.sdmdigital.id/pluginfile.php/635457/mod_page/content/16/image37.jpg)
**Gambar 36.** Ilustrasi pola sederhana dari sebuah arsitektur *agentic*.

Salah satu pendekatan untuk implementasi agentic pattern yang populer adalah **ReAct (Reasoning and Acting)**. ReAct merupakan paradigma desain prompt untuk mengubah LLM menjadi suatu "agent" yang bisa mengamati dan berinteraksi dengan "lingkungan" nya.

ReAct membagi problem solving menjadi 2 langkah yang berulang:
*   **Reasoning:** LLM melakukan reasoning atau "berpikir" berdasarkan observasi sejauh ini, apa aksi selanjutnya.
*   **Acting:** LLM melakukan suatu "aksi" menggunakan sebuah tool.

![Gambar 37. Ilustrasi ReAct](https://lms.sdmdigital.id/pluginfile.php/635457/mod_page/content/16/image30.png)
**Gambar 37.** Ilustrasi ReAct.

Perulangan ini dilakukan hingga tercapai sebuah kondisi berhenti tertentu atau ReAct sudah melebihi batas iterasi tertentu sehingga perlu langsung diakhiri, selesai maupun tidak.

Prompt ReAct biasanya perlu disertakan dengan contoh perulangan *thought-action-observation* untuk beberapa skenario. ReAct membuat LLM bisa berpikir dan bertindak sendiri, tapi butuh perhatian lebih agar bisa efektif. Penggunaan *agentic pattern* biasanya erat hubungannya dengan tool calling, yaitu kemampuan LLM untuk memilih menggunakan suatu *tool* eksternal yang dapat memberikannya kapabilitas tertentu, mulai dari mencari informasi di internet hingga menulis dan menjalankan kode.

Modul 2.7 akan membahas lebih lanjut bagaimana sebuah Agent bekerja secara lebih mendalam serta bagaimana cara implementasinya.

## Mini Quiz

*(Konten interaktif tersedia di LMS)*

---

> **Refleksi akhir dan bridging**
>
> Ada banyak detail implementasi yang perlu dipikirkan saat merancang sebuah aplikasi menggunakan agentic patterns maupun workflow yang tidak mungkin disebutkan satu per satu. Submodul ini hanya membahas pola-pola dasar yang bisa dikembangkan lagi atau digunakan untuk menyusun arsitektur yang lebih kompleks. ReAct dan berbagai pola-pola workflow yang dijelaskan di atas hanya bagian kecil dari banyaknya variasi dan arsitektur lainnya yang lebih advanced dengan pola implementasi yang beragam, dan sangat disarankan memanfaatkan sumber-sumber internet yang ditulis oleh pelaku industri maupun akademisi untuk mempelajari bermacam pola arsitektur aplikasi secara lebih lanjut. Beberapa website berikut bisa menjadi referensi:
>
> *   [https://www.anthropic.com/engineering/building-effective-agents](https://www.anthropic.com/engineering/building-effective-agents)
> *   [https://docs.langchain.com/oss/python/langgraph/workflows-agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents)
>
> Beberapa arsitektur LLM yang kompleks membutuhkan **state**, yaitu catatan kondisi terkini dari sebuah task. Dalam workflow, setiap langkah bisa mengubah state yang kemudian dipakai langkah berikutnya. Pada *agentic patterns*, state juga penting untuk menyimpan rencana, aksi yang sudah dilakukan, hasil observasi, dan langkah selanjutnya. Karena LLM itu **stateless** (tidak pernah ingat kondisi sebelumnya), maka *state* **harus dikelola di luar LLM**. Submodul berikutnya akan membahas sifat LLM yang stateless dan berbagai cara untuk mengelola state dalam aplikasi AI.