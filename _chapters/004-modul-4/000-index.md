---
slug: modul-4
layout: part
---
### ![Ikon Modul 4](https://lms.sdmdigital.id/pluginfile.php/635385/mod_page/content/11/_6c05842c-b6bc-4934-893f-691bd4692fb6-removebg-preview.png)
**Gambar: Ikon Modul 4**

# Modul 4 Training LLM: From-Scratch, Fine-Tuning, atau Continued Pretraining?
---

Pada modul 3, kita telah mempelajari komponen penyusun sebuah LLM, prinsip kerjanya, serta konsep yang mendasari cara kerja setiap komponen tersebut. Sebagaimana model *Machine Learning* pada umumnya, sebuah LLM harus dilatih agar memiliki kapabilitas yang kita inginkan. Dalam *Machine Learning*, kita mengenali berbagai jenis paradigma pembelajaran mesin seperti *Supervised Learning*, *Semi-supervised Learning*, *Unsupervised Learning*, hingga *Reinforcement Learning*. Dalam proses *training*, ada banyak proses yang dilakukan, mulai dari persiapan data untuk kebutuhan training, penetapan fungsi yang menjadi target proses optimasi, membuat definisi *training loop*, mengatur *hyperparameter*, hingga melakukan evaluasi akhir terhadap model yang sudah dilatih menggunakan *benchmark* tertentu.

LLM sendiri memiliki perbedaan yang cukup signifikan mengingat skalanya dan kapabilitasnya yang membutuhkan perlakuan khusus untuk melatihnya dengan baik. Seperti yang akan dibahas pada modul ini, LLM memiliki beberapa tahapan dalam melatihnya, mulai dari *pretraining*, *supervised fine tuning*, hingga *preference alignment*. Selain itu, skala data dan *compute* yang dibutuhkan jauh lebih besar mengingat jumlah parameternya yang cukup banyak untuk menangkap intrikasi dari bahasa natural, seperti pola-pola kata dan kalimat, keterhubungan antara konsep, dan bagaimana konteks kultural, situasi, maupun suatu bidang ilmu dapat merubah makna dari suatu kata yang sama. Dari sini muncul beberapa pertanyaan:

*   Apa yang membedakan proses pelatihan sebuah Large Language Model (LLM) dari model Machine Learning konvensional?
*   Bagaimana karakteristik data yang digunakan dalam proses pelatihan LLM, dan mengapa kualitas serta keragamannya menjadi faktor penentu kemampuan model dalam memahami bahasa alami?
*   Mengapa pelatihan LLM umumnya dilakukan melalui beberapa tahapan seperti pretraining, fine-tuning, dan alignment sebelum dapat digunakan secara efektif dalam aplikasi dunia nyata?
*   Dalam setiap tahapan pelatihan LLM, paradigma apa yang digunakan? Seperti apa *framing* masalahnya menjadi sebuah *machine learning problem*?
*   Bagaimana caranya memaksimalkan sumber daya yang terbatas untuk melatih sebuah LLM?

Modul ini akan mengajak anda membedah satu per satu kerumitan di balik proses pelatihan sebuah LLM.

Setelah mempelajari modul ini, Digiers diharapkan mampu:

![Tujuan Pembelajaran](https://lms.sdmdigital.id/pluginfile.php/635385/mod_page/content/11/Screenshot%202025-12-05%20at%2019.58.44.png)
**Gambar: Capaian Pembelajaran Modul 4**