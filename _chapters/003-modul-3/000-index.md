---
slug: modul-3
layout: part
---

![Icon Modul](https://lms.sdmdigital.id/pluginfile.php/635373/mod_page/content/14/1.3%20Fundamental%20LLM-%20Komponen%20Inti%20dan%20Arsitektur%20LLM%20Modern.png)

### Modul 3 Fundamental LLM: Komponen Inti dan Arsitektur LLM Modern
---

Di Modul 1, kita telah memposisikan Large Language Models (LLMs) sebagai *foundation models* yang esensial dalam *pipeline* seorang AI Engineer. Di Modul 2, kita telah membedah secara mendalam *blueprint* arsitektur Transformer dan mengidentifikasi tiga "keluarga" utamanya: Si Analis (Encoder-Only/BERT), Si Penerjemah (Encoder-Decoder/T5), dan Si Penulis (Decoder-Only/GPT).

Kita sekarang tahu apa komponennya—Multi-Head Attention (MHA) untuk menangkap hubungan antar kata, Feed-Forward Network (FFN) untuk transformasi, dan Embeddings untuk representasi—dan bagaimana mereka dirakit. Sekarang di Modul 3, kita mengalihkan fokus kita. Jika Modul 2 adalah tentang arsitektur statis, Modul 3 adalah tentang perilaku dinamis.

Kita akan fokus pada keluarga arsitektur yang paling dominan saat ini: Decoder-Only (*Autoregressive*), arsitektur di balik model seperti GPT, Llama, dan Claude. Kita akan membedah mengapa desain yang "lebih sederhana" ini justru mendominasi perkembangan AI generatif.

Lebih penting lagi, kita akan menjawab pertanyaan yang mendasar: Ketika model (setelah semua perhitungan MHA dan FFN) pada langkah komputasi terakhirnya dan menghasilkan daftar probabilitas (seperti Tabel 1 di Modul 2), bagaimana ia sebenarnya memilih satu kata berikutnya? Mengapa terkadang ia terdengar faktual dan "membosankan", dan di lain waktu ia terdengar sangat "kreatif" dan bahkan "halusinasi"?

Ini adalah inti dari Proses Sampling. Memahami parameter seperti *Temperature* dan *Top-p* bukanlah sekadar "opsi", melainkan "tombol" utama yang digunakan AI Engineer untuk mengontrol perilaku model.

Lebih jauh, kita akan melihat ke bagaimana arsitektur dasar ini berevolusi: bagaimana model diadaptasi untuk memahami gambar (Multimodal), dan apakah Transformer merupakan arsitektur paling optimal, atau adakah pesaing baru dengan efisiensi dan kemampuan yang lebih baik.

> **Sebelum kita mempelajari modul ini, alangkah baiknya kita merefleksikan beberapa pertanyaan krusial sebagai berikut:**
>
> *   Jika keluarga Encoder-Only (BERT) sangat hebat dalam "memahami" teks (NLU), mengapa keluarga Decoder-Only (GPT) yang justru menjadi arsitektur dominan untuk *chatbot* AI generatif modern? Apa keunggulannya?
> *   Ketika LLM menghasilkan output, apakah ia selalu memilih kata dengan probabilitas tertinggi? Jika ya, mengapa ia bisa "kreatif"? Jika tidak, mekanisme apa yang memungkinkannya memilih kata dengan probabilitas rendah?
> *   Apakah mungkin sebuah model yang dilatih hanya pada triliunan kata atau teks tiba-tiba dapat "melihat" dan mendeskripsikan sebuah gambar? Mekanisme apa yang menjembatani domain piksel gambar ke domain makna kata?
> *   Kita tahu model menjadi lebih baik dengan menambah parameter. Proses ini sangat mahal secara komputasi. Apakah ada metode yang lebih efisien, misalnya dengan hanya mengaktifkan bagian-bagian yang relevan dari model berdasarkan query atau prompt yang diterima?

**Digiers**, sebelum memulai pembelajaran, berikut adalah capaian pembelajaran yang akan menjadi kompas kita:

![Capaian Pembelajaran](https://lms.sdmdigital.id/pluginfile.php/635373/mod_page/content/14/Screenshot%202025-12-24%20at%2014.55.11.png)