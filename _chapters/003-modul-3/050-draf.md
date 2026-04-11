---
slug: modul-3-5
title: modul-3-5
---
### 3.5 Pendekatan Alternatif: Model State-Space (MAMBA) dan Diffusion LLM
---

Sepanjang Modul 2 dan 3, fokus kita melakukan pembahasan eksklusif pada Transformer (di Modul 2.5 dan variannya di subtopik 2.6, 3.1, 3.3, dan 3.4.) Selama bertahun-tahun, arsitektur ini mendominasi tanpa tandingan.

Namun, Transformer memiliki **kelemahan arsitektural fundamental** yang kita bahas di subtopik 2.4.3, yaitu kompleksitas komputasi $O(N^2)$ (**kuadratik**) dari *Self-Attention*. **Artinya, jika panjang sekuens (N) ditingkatkan 2x lipat misal, 2000 token menjadi 4000 token, kebutuhan komputasi dan memori akan meningkat secara kuadrat, yaitu sekitar 4x lipat.** Ini adalah *bottleneck* besar untuk memproses konteks yang sangat panjang (seperti seluruh buku, video, atau kode repositori).

Kelemahan ini telah memicu "perlombaan" baru di dunia penelitian: Dapatkah kita merancang arsitektur baru yang memiliki kinerja setara Transformer tetapi dengan **skalabilitas linear** $O(N)$ seperti RNN? **Artinya, jika sekuens 2x lebih panjang, komputasi yang diperlukan hanya naik 2x, bukan 4x.**

Di subtopik ini, kita akan membahas dua pesaing utama yang paling menjanjikan yang mungkin akan membentuk masa depan LLM.

![Mixture of Experts](https://lms.sdmdigital.id/pluginfile.php/635377/mod_page/content/11/image48.jpg)
**Gambar 8. Mixture of Experts**

#### 3.5.1 Pesaing #1: Model State-Space (Mamba)

Arsitektur non-Transformer pertama yang secara serius menantang dominasi Transformer (bahkan mengalahkannya dalam beberapa *benchmark* konteks panjang) adalah Mamba. Mamba bukan Transformer. Ia adalah arsitektur yang didasarkan pada Model State-Space (SSM).

> **Apa itu Mamba (Arsitektur SSM)?**
> Mamba adalah arsitektur pemodelan sekuens baru yang (seperti RNN) memproses token secara sekuensial (linear, $O(N)$). Namun (tidak seperti RNN), ia menggunakan mekanisme seleksi (S6) yang canggih untuk mengontrol aliran informasi, memungkinkannya "mengingat" konteks jangka panjang secara efisien.

**Analogi: Perbandingan Tiga Pembaca**
Untuk memahami mengapa Mamba adalah terobosan, mari kita bandingkan tiga arsitektur:

1.  **RNN (di subtopik 2.4.1): Seorang pembaca yang sangat pelupa.**
    Ia membaca kata per kata $O(N)$, tetapi "memori" atau hidden state internalnya sangat terbatas dan cepat "tercuci" atau *vanishing gradient*.
2.  **Transformer (di subtopik 2.5): Seorang pembaca yang sangat teliti namun lambat.**
    Untuk memahami kata ke-100, ia berhenti dan membaca ulang 99 kata sebelumnya (*self-attention*). Untuk kata ke-101, ia berhenti dan membaca ulang 100 kata sebelumnya. Ini akurat, tetapi sangat lambat $O(N^2)$.
3.  **Mamba (di subtopik 3.5.1): Seorang pembaca super-efisien.**
    Ia membaca kata per kata $O(N)$. Ia memiliki "memori" (state) seperti RNN, tetapi memorinya dinamis dan selektif. Pada setiap kata, Mamba "memutuskan" informasi mana dari masa lalu yang harus dikompres dan disimpan, dan informasi mana yang harus dilupakan.

Kemampuan Mamba untuk **merangkum informasi yang relevan** dari konteks masa lalu ke dalam state yang ringkas **membuatnya dapat beroperasi dengan skalabilitas linear** seperti RNN, namun dengan kekuatan kontekstual yang bersaing dengan Transformer.

#### 3.5.2 Pesaing #2: Diffusion Models for Text (Diffusion LLM)

Pendekatan ini bahkan lebih radikal. Arsitektur Autoregressive (GPT) dan Mamba sama-sama bekerja dengan memprediksi kata berikutnya (Next-Token Prediction). Model Diffusion bertanya: "Bagaimana jika kita menghasilkan teks seperti kita menghasilkan gambar (di DALL-E atau Stable Diffusion)?"

> **Apa itu Model Difusi?**
> Model Difusi adalah arsitektur *non-autoregressive* yang menghasilkan teks dalam beberapa langkah iteratif. Alih-alih memprediksi "kata berikutnya", model ini dilatih untuk "memperbaiki" (*denoise*) sekuens token yang rusak (penuh [MASK]) hingga menjadi kalimat yang koheren.

*Denoise* adalah istilah teknis yang berarti merekonstruksi sinyal asli yang berarti (dalam hal ini, kalimat yang benar) dari versi yang telah diberi *noise* atau gangguan acak, seperti token [MASK]).

**Analogi: Memperjelas Teks yang Kabur (*Denoising*)**

1.  **Pelatihan (Proses Perusakan / *Noising*):**
    *   Ambil kalimat utuh: "Kucing itu duduk di atas tikar"
    *   Rusak kalimat itu secara acak: "Kucing [MASK] duduk [MASK] atas [MASK]"
    *   Latih model untuk memprediksi token asli yang di-mask.
2.  **Inferensi/Generasi (Proses Perbaikan / *Denoising*):**
    *   Kita tidak memulai dengan 1 token. Kita memulai dengan "batu" berisi kebisingan murni (misal: 10 token [MASK]).

**Ilustrasi Model Difusi**

![Alur Generasi Model Teks Difusi](https://lms.sdmdigital.id/pluginfile.php/635378/mod_page/content/11/image29.jpg?time=1766654556146)
**Gambar 10. Alur Generasi Model Teks Difusi**

**Seperti yang ditunjukkan pada Gambar 10**, alih-alih menghasilkan kata satu per satu, model *denoising* bekerja secara bertahap.
*   Pada **Iterasi 1**, ia memulai dengan *noise* penuh.
*   Pada **Iterasi 2**, ia mungkin memiliki "keyakinan rendah" dan mengisi kata-kata yang paling mungkin (...itu duduk...).
*   Pada **Iterasi 3**, ia menggunakan konteks baru itu untuk memperbaiki tebakannya lagi (Kucing itu duduk di...).

Proses ini berlanjut hingga semua [MASK] hilang dan kalimat menjadi koheren.

**Keuntungan Potensial:** Karena model ini "melihat" seluruh sekuens (meskipun dalam bentuk [MASK]) dari awal, ia berpotensi lebih baik dalam perencanaan jangka panjang dan koherensi global. Ini adalah sesuatu yang terkadang sulit dilakukan oleh model *autoregressive* (yang **secara arsitektural hanya bisa melihat token-token sebelumnya**). Ini masih merupakan area penelitian yang sangat aktif dan eksperimental.