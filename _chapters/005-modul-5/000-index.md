---
slug: modul-3
layout: part
---

![Icon Modul](https://lms.sdmdigital.id/pluginfile.php/635360/mod_page/content/29/unnamed.jpg)

### Modul 2 Fundamental LLM: Dari Pemodelan Bahasa Statistik ke Arsitektur Transformer
---

Pada Modul 1, kita telah memposisikan Large Language Models (LLM) sebagai *foundation models* yang esensial dalam domain AI Engineering, serta telah mengamati kapabilitasnya yang impresif dalam generasi sebuah informasi atau pengetahuan. Sekarang, kita perlu melangkah lebih dalam untuk memahami bagaimana model-model tersebut bekerja di dalamnya, tidak hanya memperlakukannya sebagai “kotak hitam”, tetapi menelusuri mekanisme fundamental bagaimana LLM bekerja.

Sebuah LLM, pada intinya, tidak beroperasi melalui 'pemahaman' semantik seperti manusia. Sebaliknya, ia adalah sebuah sistem komputasi yang sangat canggih untuk memodelkan probabilitas sekuens teks. Pertanyaan "Bagaimana ini bekerja?" pada dasarnya adalah pertanyaan tentang *pemodelan bahasa* (Language Modeling).

Pemodelan Bahasa (*Language Modeling*), secara formal, adalah sebuah tugas untuk mengestimasi distribusi probabilitas atas sebuah sekuens token (kata atau sub-kata). Dengan kata lain, $P(w_1, w_2, ..., w_n)$, dimana $w_n$ adalah word atau token ke-n. Model yang mampu melakukan ini secara efektif (memberikan probabilitas tinggi pada teks yang "masuk akal", misal, $P(\{\text{"langit itu biru"}\}) > P(\{\text{"biru itu langit"}\})$ secara inheren telah mempelajari struktur, sintaksis, semantik, dan bahkan pengetahuan faktual dari bahasa tersebut.

Modul ini akan membahas evolusi pendekatan dalam pemodelan bahasa dari waktu ke waktu. Kita akan memulai dengan fondasi statistik klasik, yaitu model n-gram (Subtopik 2.2). Model ini, meskipun sederhana, meletakkan dasar konseptual bahwa probabilitas sebuah kata dapat diperkirakan dengan melihat n kata sebelumnya (mengikuti asumsi Markov). Kita akan menganalisis keterbatasan fundamentalnya. Masalah utamanya adalah ledakan kombinatorial dari sekuens kata yang mungkin muncul.

Keterbatasan ini mendorong pengembangan model neural sekuensial. Arsitektur seperti *Recurrent Neural Networks* (RNN) dan variannya, *Long Short-Term Memory* (LSTM) serta *Gated Recurrent Unit* (GRU), diperkenalkan. Model-model ini bekerja dengan analogi seperti seorang pembaca yang memproses teks kata demi kata. Mereka menjaga sebuah 'vektor state' (sebuah representasi numerik) yang berfungsi sebagai 'memori jangka pendek' dari apa yang telah dibaca. Vektor ini terus diperbarui saat model membaca token berikutnya.

Namun, seperti yang akan kita diskusikan (Subtopik 2.3), memori ini memiliki keterbatasan. Arsitektur rekuren ini menghadapi sebuah tantangan komputasi yang fundamental: kesulitan dalam menangani ketergantungan jangka panjang (*long-term dependencies*). Fenomena *vanishing/exploding gradients* selama pelatihan membuat 'memori' RNN menjadi tidak efektif untuk menghubungkan kata di akhir kalimat dengan kata di awal kalimat yang sangat panjang.

Titik balik revolusioner hadir dalam bentuk mekanisme *attention*. *Attention* adalah sebuah mekanisme yang memungkinkan model untuk secara dinamis "melihat kembali" dan memberi bobot kepentingan pada bagian-bagian spesifik dari input saat memproses sebuah token. *Sebagai analogi,* bayangkan seorang peneliti yang menulis sebuah rangkuman. Daripada hanya mengandalkan ingatan jangka pendeknya (seperti RNN), mekanisme *attention* memungkinkan model untuk bertindak seperti peneliti yang memiliki akses penuh ke seluruh dokumen sumber. Saat menghasilkan satu kata (misalnya, nama tokoh), model dapat 'melihat kembali' secara langsung ke bagian teks input yang paling relevan (di mana nama itu pertama kali disebut), memberinya 'bobot' kepentingan yang tinggi, dan mengabaikan sisa teks yang tidak relevan.

Puncak dari evolusi ini dan yang menjadi fokus utama modul ini, adalah arsitektur Transformer ([Vaswani et al., 2017](https://arxiv.org/html/1706.03762v7)) (Subtopik 2.4).

![Gambar 1. Arsitektur Transformer](https://lms.sdmdigital.id/pluginfile.php/635360/mod_page/content/29/Gambar1.jpg)

**Gambar 1. Arsitektur Transformer**

Transformer, yang tidak menggunakan mekanisme rekuren sepenuhnya dan hanya mengandalkan *self-attention*, telah menjadi standar *de facto* untuk hampir semua LLM modern. Kita akan membedah arsitektur ini secara metodis, mengidentifikasi komponen *Encoder, Decoder, Multi-Head Attention,* dan *Positional Encoding.*

Terakhir, kita akan menganalisis bagaimana arsitektur dasar ini berdiversifikasi menjadi tiga "keluarga" utama (Subtopik 2.5) sebagai berikut:

1.  **Encoder-Only** (misal, BERT, RoBERTa): Dirancang untuk tugas *Natural Language Understanding* (NLU) seperti klasifikasi atau *named entity recognition*, di mana pemahaman kontekstual *bidirectional* sangat krusial.
2.  **Encoder-Decoder** (misal, T5, BART, Transformer orisinal): Dirancang untuk tugas *sequence-to-sequence* (seq2seq) seperti translasi mesin atau peringkasan.
3.  **Decoder-Only** (misal, GPT, Llama): Arsitektur yang dominan saat ini untuk generasi bahasa *autoregressive*, yang akan menjadi fokus lebih dalam di Modul 3.

Memahami fondasi ini bersifat non-trivial. Bagi seorang AI Engineer, pengetahuan ini memberikan kemampuan untuk membuat keputusan arsitektural yang terinformasi, memahami *trade-off* kinerja, dan men-debug perilaku model secara efektif.

Sebelum kita mempelajari modul ini, alangkah baiknya kita merefleksikan peran krusial dari dokumentasi dan komunikasi ini sebagai berikut:

*   Secara matematis, apa yang membedakan kalimat "mahasiswa membaca buku" dengan "buku membaca mahasiswa"? Bagaimana sebuah model statistik dapat dilatih untuk "mengetahui" bahwa yang pertama jauh lebih mungkin terjadi?
*   Jika sebuah model hanya dapat "mengingat" tiga kata sebelumnya (model *trigram*), skenario bahasa kompleks apa yang hampir pasti akan gagal diproses dengan benar?
*   Pertimbangkan kalimat: "Ibu memasukkan kue ke dalam oven, tetapi kue itu hangus karena **ia** lupa waktu." Saat model memproses kata "ia", bagaimana secara komputasional ia dapat menentukan bahwa "ia" merujuk pada "Ibu" dan bukan "kue" atau "oven"?
*   Mengapa sebuah arsitektur yang sangat baik dalam *menganalisis* sentimen sebuah kalimat (seperti BERT) memiliki desain yang berbeda secara fundamental dari arsitektur yang sangat baik dalam *menulis* kelanjutan cerita (seperti GPT)?
*   **Digiers**, sebelum memulai pembelajaran, mari kita pahami bersama capaian yang akan kita raih. Berikut adalah tujuan pembelajaran yang akan menjadi kompas kita:

![Tujuan Pembelajaran](https://lms.sdmdigital.id/pluginfile.php/635360/mod_page/content/29/image%20%282%29.png)