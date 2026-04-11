---
slug: modul-4-5
title: modul-4-5
---
## 4.5 Mengadaptasikan LLM yang sudah dilatih
---

Mengembangkan LLM dari nol membutuhkan *effort* yang besar untuk proses pengumpulan dan pembersihan data serta kebutuhan *budget* yang cukup signifikan terutama untuk menyediakan komputasi. Selain itu seiring waktu berbagai macam LLM yang sudah dilatih dengan tingkat kemampuan yang cukup baik bisa dengan mudah diakses dan digunakan oleh siapapun, tersedia secara bebas pada website seperti *HuggingFace* dengan lisensi penggunaan yang permisif, bahkan untuk kebutuhan komersil.

Umumnya pendekatan yang lebih pragmatis dilakukan oleh berbagai pelaku industri maupun akademisi untuk beberapa kasus adalah mengadaptasikan LLM yang sudah dilatih sebelumnya untuk sebuah *use case* tertentu. Dalam sebuah [survey](https://arxiv.org/abs/2501.09223) terhadap pelaku industri maupun akademisi, terdapat tiga cara utama untuk mengadaptasikan LLM yang sudah dilatih:

1. Memberikan LLM instruksi dan konteks
2. Melakukan *fine tuning*
3. Melakukan *continued pretraining*

### 4.5.1 Memberikan LLM instruksi dan konteks

Metode ini merupakan metode yang paling populer dan praktis yang sangat mudah diadopsi oleh banyak pelaku industri sehingga memunculkan paradigma baru pengembangan sistem yang umum disebut sebagai *AI Engineering*. Metode ini memanfaatkan kemampuan LLM mengikut instruksi dan melakukan *in-context learning* untuk mengadaptasikannya ke *use case* tertentu dengan memberikan instruksi dan konteks yang tepat. 

Seperti yang telah dipelajari pada Modul 1, tahap ini melibatkan menciptakan sebuah sistem untuk melakukan orkestrasi baik instruksi maupun konteks dari berbagai sumber untuk merangkai *prompt* yang tepat untuk mencapai tujuan tertentu. Istilah yang lumrah digunakan dalam industri untuk proses ini adalah *context engineering*. Jenis sistem yang dikembangkan bisa beragam, mulai dari sistem sederhana yang memproses input dan output secara langsung, *Retrieval Augmented Generation* yang menambahkan cara LLM untuk mencari sendiri informasi dalam dokumen yang disimpan maupun melalui pencarian internet, hingga sistem kompleks yang terdiri atas orkestrasi lebih dari 1 LLM yang berkerja sama untuk mencapai tujuan tertentu, yang disebut *multi-agents system*.

### 4.5.2 Melakukan Continued Pretraining

Tahap ini biasanya diperlukan ketika LLM ingin digunakan pada suatu domain yang memiliki *vocabulary* tertentu yang sangat khusus atau spesifik sehingga diperlukan adaptasi lanjutan di tingkatan pemahaman LLM mengenai bahasa. Pendekatan ini umumnya menggunakan model hasil *pretraining* sebagai *base model* yang dilatih kembali menggunakan data spesifik domain yang diinginkan. 

Tidak hanya memerlukan jumlah data yang cukup banyak dan daya komputasi *compute* yang signifikan, resiko seperti *catastrophic forgetting* juga perlu diperhitungkan dan konsekuensinya perlu perhatian khusus dalam melakukan proses ini untuk memastikan hasilnya sesuai harapan. *Catasthropic forgetting* adalah ketika model kehilangan pengetahuan sebelumnya karena fokus belajar pada data baru.

### 4.5.3 Melakukan Fine Tuning

Tahap ini dilakukan secara *supervised* seperti tahap *instruction tuning* yang telah dibahas sebelumnya. Tahap ini melibatkan mengumpulkan dataset berlabel untuk lalu melakukan *supervised fine tuning*. Metode ini digunakan ketika adaptasi yang diinginkan memiliki bentuk yang bisa dengan mudah diekspresikan sebagai pasangan input-output atau berupa *task* yang cukup spesifik. 

Belakangan ini berbagai macam perkembangan mengenalkan konsep *adapter* dimana untuk mengadaptasikan LLM, kita tidak perlu mengubah bobotnya secara langsung. *Fine tuning* dilakukan untuk menghasilkan adapter yang bisa secara modular dipasang ke sebuah model untuk memberikan suatu kemampuan spesifik. Selain itu, berbagai macam pendekatan dan perkembangan di industri maupun riset memungkinkan *fine tuning* dilakukan secara efisien, dengan berbagai macam optimisasi yang menurunkan kebutuhan komputasi tinggi dan mengurangi ukuran dataset yang diperlukan untuk melakukan *fine tuning* sehingga bisa dengan mudah dilakukan secara murah atau bahkan gratis karena bisa dilakukan pada perangkat pribadi maupun layanan gratis seperti *google colab*.

### 4.5.4 Sebuah Framework Sederhana untuk Memilih Opsi Model dalam Pengembangan Sistem Berbasis AI

Berikut adalah sebuah *framework* berpikir sederhana yang bisa diikuti untuk memutuskan cara yang paling tepat untuk memilih model yang ingin kita gunakan. Untuk menggunakan *framework* ini, anda terlebih dahulu harus:

* Memahami tujuan yang ingin dicapai
* Memahami bagaimana cara mengukur ketercapaian tujuan ini dari sisi model maupun performa sistem keseluruhan
* Memahami keperluan *resource* yang dibutuhkan untuk setiap tahap yang berbeda

---

### Mini Quiz

*(Silakan akses kuis interaktif pada platform LMS melalui tautan berikut)*
[Klik untuk membuka Mini Quiz](https://lms.sdmdigital.id/h5p/embed.php?url=https%3A%2F%2Flms.sdmdigital.id%2Fpluginfile.php%2F635390%2Fmod_page%2Fcontent%2F7%2Fmultiple-choice-219.h5p)

---

> **Refleksi Singkat**
>
> Berdasarkan apa yang anda sudah ketahui soal proses melatih LLM, kebutuhan data dan *compute* nya, serta apa yang perlu dilakukan *AI Engineer* untuk mengadaptasikan LLM melalui instruksi dan konteks, bagaimana sebaiknya seorang *AI Engineer* menentukan kapan harus membangun model dari nol atau melakukan adaptasi melalui *tuning* maupun menggunakan instruksi dan konteks? Apa saja pertimbangannya?