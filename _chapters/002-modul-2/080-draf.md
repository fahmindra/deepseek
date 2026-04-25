---
slug: modul-2-8
title: modul-2-8
---

### Penutup
### Modul 2 Fundamental LLM: Dari Pemodelan Bahasa Statistik ke Arsitektur Transformer
---

Memahami fondasi teknis Large Language Models (LLMs) bukan hanya tentang memuaskan rasa ingin tahu akademis. Bagi seorang AI Engineer, ini adalah fondasi krusial untuk membuat keputusan yang terinformasi: memilih arsitektur yang tepat, memahami trade-off kinerja, dan men-debug perilaku model secara efektif.

Dalam modul ini, kita telah melakukan "bedah internal" LLM, menelusuri evolusi dari model statistik klasik hingga arsitektur neural modern:

*   Kita memulainya dengan memformulasikan masalah pemodelan bahasa secara statistik $(P(W))$ dan menganalisis solusi statistik klasik (N-gram) serta tantangan fundamentalnya (Sparsitas Data).
*   Kita membongkar dua langkah pra-proses esensial yang menjadi fondasi semua model neural: Tokenisasi (bagaimana teks dipecah menjadi ID numerik) dan Embedding (bagaimana ID tersebut diberi representasi "makna" dalam bentuk vektor).
*   Kita menelusuri evolusi model sekuensial (RNN/LSTM), memahami keterbatasannya (vanishing gradient, bottleneck), hingga menyaksikan revolusi yang dibawa oleh mekanisme Attention (Q, K, V) sebagai solusi atas ketergantungan jangka panjang.
*   Kita membedah "cetak biru" Arsitektur Transformer (Diagram 2.1), mengidentifikasi komponen inti seperti Self-Attention, Multi-Head Attention (MHA), Add & Norm, dan FFN.
*   Terakhir, kita membedakan tiga keluarga arsitektur utama: Encoder-Only (Si "Analis" seperti BERT), Decoder-Only (Si "Penulis" seperti GPT/Llama), dan Encoder-Decoder (Si "Penerjemah" seperti T5).

Dengan menyelesaikan materi dalam modul ini, Anda telah berhasil mencapai tujuan pembelajaran berikut:

*   Anda kini dapat memformulasikan masalah pemodelan bahasa statistik $(P(W))$ dan menjelaskan keterbatasan fundamental model n-gram.
*   Anda mampu menganalisis keterbatasan arsitektur rekuren (RNN/LSTM) dalam menangani ketergantungan jangka panjang.
*   Anda dapat menjelaskan evolusi konseptual dari attention standar ke self-attention dan multi-head attention sebagai mekanisme inti untuk menangkap konteks.
*   Anda dapat menguraikan komponen penyusun arsitektur Transformer orisinal (Vaswani et al., 2017), seperti MHA, FFN, dan Add & Norm.
*   Anda mampu membandingkan dan membedakan tiga keluarga arsitektur Transformer (Encoder-Only, Decoder-Only, Encoder-Decoder) dan mengasosiasikannya dengan tipe tugas yang sesuai.

> ***Refleksi Akhir***
> 
> Setelah mempelajari "mesin" di balik LLM mulai dari N-gram, ke RNN, lalu ke Transformer, apakah Anda mulai melihat bagaimana pemahaman tentang Tokenisasi, Attention, dan perbedaan 3 keluarga arsitektur Transformer ini secara langsung memengaruhi cara Anda akan memilih model yang tepat untuk sebuah tugas (misal, memilih BERT vs GPT) atau menganalisis mengapa sebuah model gagal memahami konteks yang panjang?