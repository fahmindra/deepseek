---
slug: bab-1-10
title: "Arsitektur"
layout: chapter
---

# BAB 1: Anatomi dan Hiperparameter Arsitektur Model Bahasa (Language Models)

## 1.1 Pendahuluan dan Tujuan Pembelajaran
Dalam pengembangan Model Bahasa Besar (*Large Language Models* / LLMs) modern, pemahaman mengenai arsitektur dasar dan hiperparameter adalah hal yang krusial. Bab ini akan membedah secara mendalam segala hal mengenai arsitektur LLM dan hiperparameter yang sering kali dilewatkan atau dianggap sebagai "kotak hitam" (*black box*). 

Tema utama dari pembahasan ini adalah: **Cara terbaik untuk belajar adalah melalui pengalaman langsung (praktik), dan cara terbaik kedua adalah belajar dari pengalaman (dan eksperimen) orang lain.**

Fokus pembahasan dalam bab ini meliputi:
1. Rekapitulasi singkat mengenai arsitektur *Transformer* modern.
2. Identifikasi komponen-komponen inti yang menjadi standar pada mayoritas LLM berskala besar saat ini.
3. Analisis variasi umum pada arsitektur dan proses pelatihan (*training process*).

Meskipun pada tahun 2024-2025 terdapat lebih dari 19 perilisan model padat (*dense models*) baru (seperti Llama 3, Falcon, Gemma, Qwen, DeepSeek, dll.), mayoritas dari model-model tersebut memiliki kesamaan arsitektur dasar yang kuat, dengan hanya sedikit penyesuaian ( *minor architecture tweaks*). 

## 1.2 Titik Awal: Arsitektur Transformer 'Orisinal' vs. Varian Modern

Untuk memahami model modern, kita harus meninjau kembali pilihan desain pada arsitektur *Transformer* standar (sebagaimana dipublikasikan oleh Vaswani et al., 2017).

**Karakteristik Transformer Orisinal:**
*   **Posisi Normalisasi (*Norm type*):** Menggunakan *Post-norm* (Normalisasi dilakukan setelah penambahan residual) dengan fungsi `LayerNorm`.
*   **Fungsi Aktivasi (FFN):** Menggunakan fungsi aktivasi ReLU (*Rectified Linear Unit*). Rumus matematis untuk lapisan *Feed-Forward Network* (FFN) orisinal adalah:
    $$FFN(x) = \max(0, xW_1 + b_1)W_2 + b_2$$
    Di mana $x$ adalah input, $W_1$ dan $W_2$ adalah matriks bobot, serta $b_1$ dan $b_2$ adalah vektor bias.
*   **Penyematan Posisi (*Position embedding*):** Menggunakan fungsi sinus dan kosinus absolut.
    $$PE_{(pos, 2i)} = \sin(pos/10000^{2i/d_{model}})$$
    $$PE_{(pos, 2i+1)} = \cos(pos/10000^{2i/d_{model}})$$
    Di mana $pos$ adalah posisi token, $i$ adalah dimensi, dan $d_{model}$ adalah dimensi representasi model.

**Karakteristik Varian Transformer Modern (Arsitektur 'LLaMA-like'):**
Saat ini, terdapat dominasi arsitektur yang sering disebut bergaya 'LLaMA'. Jika Anda mengimplementasikan model modern, Anda akan melakukan perubahan berikut dari arsitektur orisinal:
1.  **Posisi Normalisasi:** `LayerNorm` (atau variannya) diletakkan di *depan* blok sub-lapisan (*Pre-norm*).
2.  **Penyematan Posisi:** Menggunakan *Rotary Position Embeddings* (RoPE) alih-alih sinus/kosinus absolut.
3.  **Fungsi Aktivasi:** Lapisan *Feed-Forward* menggunakan aktivasi Gated Linear Units seperti **SwiGLU**, bukan lagi ReLU.
4.  **Penghilangan Bias:** Lapisan linear (dan lapisan normalisasi) **tidak memiliki** term bias (konstanta).

Mengapa perubahan-perubahan ini dipilih? Bagian selanjutnya akan membedah justifikasi akademis dan empiris dari masing-masing modifikasi tersebut.

---

# BAB 2: Variasi Arsitektur Inti: Normalisasi dan Bias

## 2.1 Perdebatan Pre-Norm vs. Post-Norm
Satu hal yang hampir disepakati oleh seluruh peneliti AI pada tahun 2024 adalah penggunaan arsitektur **Pre-norm**. 

**Penjelasan Visual Blok Transformer (Berdasarkan Xiong et al., 2020):**
Dalam grafik visual perbandingan, pada arsitektur *Post-LN* (Post-LayerNorm), sinyal input residual ($x_l$) mengalir, ditambah dengan hasil sub-lapisan (seperti *Multi-Head Attention* atau FFN), lalu seluruh hasilnya dimasukkan ke dalam blok `LayerNorm` sebelum diteruskan ke lapisan berikutnya.
Sebaliknya, pada arsitektur *Pre-LN* (Pre-LayerNorm), sinyal residual $x_l$ mengalir lurus ke atas *tanpa gangguan*. Input menuju sub-lapisan *Attention* atau FFN diambil dari aliran utama tersebut, dilewatkan ke `LayerNorm` terlebih dahulu, diproses, lalu ditambahkan kembali ke aliran utama melalui koneksi residual. Konfigurasi ini memastikan `LayerNorm` tidak memodifikasi jalur sinyal residual utama secara langsung.

**Formulasi Matematis:**
*Transformer Post-LN:*
$$x_{l,i}^{post, 2} = x_{l,i}^{post} + x_{l,i}^{post, 1}$$
$$x_{l,i}^{post, 3} = \text{LayerNorm}(x_{l,i}^{post, 2})$$
*(Normalisasi memblokir aliran residual langsung).*

*Transformer Pre-LN:*
$$x_{l,i}^{pre, 1} = \text{LayerNorm}(x_{l,i}^{pre})$$
$$x_{l,i}^{pre, 3} = x_{l,i}^{pre} + x_{l,i}^{pre, 2}$$
*(Residual ditambahkan setelah sub-lapisan memproses data yang telah dinormalisasi).*

Hampir semua model bahasa modern menggunakan *pre-norm*, dengan BERT sebagai pengecualian historis (*post-norm*) dan OPT350M sebagai pengecualian aneh di era modern.

### Bukti Empiris Mengapa Pre-Norm Lebih Baik
Terdapat dua alasan utama berdasarkan data penelitian mengapa *Pre-norm* diunggulkan:

1.  **Mencegah Atenuasi Gradien (*Gradient Attenuation*) (Xiong, 2020):** 
    Grafik batang (*bar chart*) mengekspektasikan gradien pada lapisan FFN ($W^1$) menunjukkan bahwa pada model *Pre-LN* (batang biru), ekspektasi gradien relatif konstan melintasi lapisan 1 hingga 6. Namun, pada *Post-LN* tanpa pemanasan (*warm-up*) (batang oranye), nilai gradien melonjak tajam seiring bertambahnya lapisan, yang menyebabkan ketidakstabilan.
2.  **Mencegah Lonjakan Gradien (*Gradient Spikes*) (Salazar and Ngyuen, 2019):**
    Grafik *time-series* dari iterasi pelatihan menunjukkan bahwa *PostNorm+LayerNorm* (garis putus-putus biru) menghasilkan lonjakan *global norm* gradien yang sangat fluktuatif (spikes). Sebaliknya, *PreNorm+LayerNorm* (garis merah padat) menghasilkan pergerakan gradien yang jauh lebih stabil dan rendah.

**Kesimpulan:** Secara historis, keuntungan utama *Pre-norm* yang disorot adalah kemampuannya untuk menghilangkan kebutuhan akan tahap *warm-up* (*learning rate*). Saat ini, alasan utamanya adalah **stabilitas optimisasi** yang memungkinkan penggunaan *learning rate* (LR) yang lebih besar untuk melatih jaringan saraf berskala raksasa.

### Inovasi Baru: 'Double' Norm atau Non-residual Postnorm
Jika menempatkan `LayerNorm` di dalam aliran residual adalah hal yang buruk, mengapa tidak menambahkan *post-norm* di luar aliran residual? Beberapa model terbaru seperti **Grok, Gemma 2,** dan **Olmo 2** mengadopsi pendekatan normalisasi ganda atau menggunakan *non-residual post norm*.

## 2.2 LayerNorm vs. RMSNorm
Pada *Transformer* orisinal, `LayerNorm` digunakan untuk menormalisasi nilai rata-rata (*mean*) dan varians di seluruh dimensi $d_{model}$. Rumusnya adalah:
$$y = \frac{x - E[x]}{\sqrt{Var[x] + \epsilon}} * \gamma + \beta$$
Model historis yang menggunakan ini: GPT3/2/1, OPT, GPT-J, BLOOM.

Namun, banyak LLM modern (LLaMA-family, PaLM, Chinchilla, T5) beralih ke **RMSNorm** (*Root Mean Square Normalization*). RMSNorm **tidak** mengurangkan nilai rata-rata ($E[x]$) dan **tidak** menambahkan term bias ($\beta$).
Rumusnya menjadi lebih sederhana:
$$y = \frac{x}{\sqrt{||x||_2^2 + \epsilon}} * \gamma$$

### Mengapa RMSNorm Digunakan?
Penjelasan modern yang umum diterima adalah bahwa RMSNorm **lebih cepat (namun memiliki performa yang sama baiknya)**.
*   **Fungsi Operasi Lebih Sedikit:** Tidak perlu menghitung *mean*.
*   **Parameter Lebih Sedikit:** Tidak ada term bias yang perlu disimpan di memori.

**Namun, apakah penjelasan ini masuk akal?**
Dalam tabel analisis profil komputasi (Ivanov et al., 2023), disebutkan bahwa:
*   Kontraksi Tensor (*Matrix Multiplies*) memakan **99.80% FLOPs**.
*   Normalisasi Statistik (seperti LayerNorm/RMSNorm) hanya memakan **0.17% FLOPs**.

Jika normalisasi hanya menyumbang 0.17% dari total komputasi matematika, mengapa penyederhanaannya diklaim mempercepat model secara signifikan?

**Pelajaran Penting:** FLOPs (*Floating Point Operations Per Second*) BUKANLAH cerminan waktu eksekusi aktual (*runtime*)!
Meskipun secara FLOPs normalisasi hanya 0.17%, secara *runtime*, normalisasi memakan **25.5%** waktu komputasi! 

Hal ini dijelaskan melalui rasio FLOP-to-memory. Operasi perkalian matriks (MHA/FFN) memiliki rasio 153 (artinya banyak komputasi untuk setiap byte memori yang dipindahkan). Sebaliknya, *LayerNorm* memiliki rasio FLOP-to-memory yang sangat kecil (sekitar 3.5), yang berarti ini adalah operasi yang dibatasi oleh memori (*memory-bound*). RMSNorm menjadi sangat penting karena meminimalkan **pergerakan data (*data movement*)**, yang merupakan leher botol (*bottleneck*) terbesar di perangkat keras (GPU/TPU) modern.

Tabel eksperimen oleh Narang et al. (2020) mengonfirmasi hal ini: mengganti LayerNorm dengan RMSNorm pada arsitektur *Rezero* meningkatkan metrik Step/s (kecepatan) dari 3.26 menjadi 3.34, sekaligus sedikit memperbaiki (atau mempertahankan) metrik *loss* dan akurasi *benchmark*.

## 2.3 Penghilangan Term Bias Secara Umum
Mayoritas *Transformer* modern tidak lagi memiliki term bias pada lapisan linier mereka.
Alih-alih menggunakan FFN orisinal:
$$FFN(x) = \max(0, xW_1 + b_1)W_2 + b_2$$
Implementasi modern (jika tidak menggunakan teknik *gating*) menjadi:
$$FFN(x) = \sigma(xW_1)W_2$$
**Alasan:** Serupa dengan kasus RMSNorm, hal ini dilakukan untuk menghemat pergerakan memori dan meningkatkan stabilitas optimisasi. Keuntungan komputasi dari mengorbankan parameter bias jauh lebih besar daripada kerugian kapasitas ekspresifnya.

---

# BAB 3: Fungsi Aktivasi dan Gating

## 3.1 Fungsi Aktivasi Dasar
Selama bertahun-tahun, arsitektur jaringan saraf telah mengeksplorasi "kebun binatang" (*zoo*) fungsi aktivasi: ReLU, GeLU, Swish, ELU, GLU, GeGLU, ReGLU, SeLU, SwiGLU, LiGLU.

1.  **ReLU (*Rectified Linear Unit*):** 
    $$FF(x) = \max(0, xW_1) W_2$$
    Grafiknya datar di angka 0 untuk nilai negatif, dan linier berbanding lurus untuk nilai positif.
    *Model terkemuka:* Transformer orisinal, T5, Gopher, Chinchilla, OPT.
2.  **GeLU (*Gaussian Error Linear Unit*):**
    $$FF(x) = GELU(xW_1)W_2$$
    $$GELU(x) := x\Phi(x)$$
    Grafiknya mirip ReLU, namun memiliki kurva halus (*smooth curve*) yang sedikit menembus area negatif di sekitar nilai 0 sebelum mendatar.
    *Model terkemuka:* GPT1/2/3, GPT-J, GPT-NeoX, BLOOM.

## 3.2 Gated Activations (*GLU Variants)
Inovasi terbesar dalam lapisan maju (*feed-forward*) modern adalah penggunaan *Gated Linear Units* (GLU). GLU memodifikasi 'bagian pertama' dari lapisan FF. Alih-alih melakukan transformasi linier dan ReLU sederhana, GLU melakukan penambahan satu term linier ekstra (*entrywise*).

Transformasi konseptualnya adalah:
$$\max(0, xW_1) \rightarrow \max(0, xW_1) \otimes (xV)$$
Di mana simbol $\otimes$ melambangkan perkalian elemen-demi-elemen (*element-wise multiplication*).
Ini memberikan kita varian *gated* seperti **ReGLU** (menggunakan parameter ekstra $V$):
$$FF_{\text{ReGLU}}(x) = (\max(0, xW_1) \otimes xV) W_2$$

Varian yang paling populer hari ini adalah **GeGLU** dan **SwiGLU**:
*   **GeGLU:**
    $$FF_{\text{GEGLU}}(x, W, V, W_2) = (GELU(xW) \otimes xV)W_2$$
    *Model terkemuka:* T5 v1.1, mT5, LaMDA, Phi3, Gemma 2, 3, 4.
*   **SwiGLU** (*swish* didefinisikan sebagai $x * \text{sigmoid}(x)$):
    $$FF_{\text{SwiGLU}}(x, W, V, W_2) = (\text{Swish}_1(xW) \otimes xV)W_2$$
    *Model terkemuka:* LLaMA 1/2/3, PaLM, Mistral, OlMo, dan **hampir sebagian besar model yang dirilis setelah 2023**.

*Catatan Kritis:* Karena arsitektur *gated* memperkenalkan matriks parameter tambahan ($V$), untuk mempertahankan jumlah parameter agar tetap konsisten dengan model tradisional, dimensi *feed-forward* ($d_{ff}$) harus diperkecil menjadi sebesar $\frac{2}{3}$ dari ukuran awalnya.

### Apakah Gated Linear Units (GLU) Benar-Benar Berfungsi?
Ya, sangat konsisten. Berdasarkan studi Shazeer (2020), varian GLU (*GeGLU, SwiGLU, ReGLU*) secara persisten mengalahkan varian non-gated (*ReLU, GeLU, Swish*) dalam hal metrik rata-rata skor pada *benchmark* seperti CoLA dan SST-2. Narang et al. (2020) juga menguatkan temuan ini; *SwiGLU* menunjukkan metrik *Final Loss* yang lebih rendah (1.789 dibandingkan 1.838 milik *Vanilla Transformer*) dan performa SGLUE yang lebih tinggi (76.00 berbanding 71.66).

Meskipun *GLU tidak diwajibkan secara mutlak* untuk membuat model bekerja (GPT-3 berfungsi sangat baik tanpanya, dan Nemotron 340B bahkan menggunakan *Squared ReLU*), namun bukti empiris sangat mengarah pada perolehan performa yang konsisten dari penggunaan SwiGLU/GeGLU.

---

# BAB 4: Desain Eksekusi Lapisan (Serial vs. Parallel)

## 4.1 Lapisan Serial
Secara konvensional, blok *Transformer* dieksekusi secara **serial** (berurutan). Sinyal mengalir ke *Attention*, lalu dijumlahkan secara residual, dinormalisasi, lalu baru masuk ke *Feed-Forward Network* (MLP). Operasi tidak bisa mendahului satu sama lain.

## 4.2 Lapisan Paralel
Dapatkah kita memparalelkan blok *transformer* ini? Beberapa model (seperti GPT-J, PaLM, GPT-NeoX) memformulasikan komputasi ini agar *Attention* dan *MLP* dihitung bersamaan.

Menurut makalah GPT-J (Wang & Komatsuzaki, 2021), formulasi standar (serial) ditulis sebagai:
$$y = x + \text{MLP}(\text{LayerNorm}(x + \text{Attention}(\text{LayerNorm}(x))))$$
Sedangkan formulasi paralel direpresentasikan sebagai:
$$y = x + \text{MLP}(\text{LayerNorm}(x)) + \text{Attention}(\text{LayerNorm}(x))$$

Jika diimplementasikan dengan benar pada level perangkat keras, `LayerNorm` hanya perlu dihitung sekali lalu dibagikan (*shared*), dan perkalian matriks dari input Attention dan MLP dapat digabung (*fused*), menghasilkan kecepatan pelatihan hingga 15% lebih cepat pada skala besar tanpa degradasi kualitas. 
*Model terkini yang menggunakan ini:* Cohere Command A, Falcon 2 11B, Command R+. Namun, sebagian besar model masih menggunakan desain serial.

---

# BAB 5: Penyematan Posisi (Positional Embeddings)

*Transformer* tidak memiliki kesadaran akan urutan secara inheren (ia memproses seluruh token secara bersamaan). Oleh karena itu, diperlukan *Positional Embeddings* (PE). 

## 5.1 Evolusi Metode Posisi
1.  **Penyematan Sinus (Orisinal):** Menambahkan nilai fungsi sinus dan kosinus pada vektor input.
    $$Embed(x, i) = v_x + PE_{pos}$$
2.  **Penyematan Absolut (GPT 1/2/3):** Menambahkan vektor posisi yang dipelajari selama *training*.
    $$Embed(x, i) = v_x + u_i$$
3.  **Penyematan Relatif (T5, Chinchilla):** Menambahkan vektor/skalar khusus langsung ke dalam komputasi *attention* (bukan pada input awal).
    $$e_{ij} = \frac{x_i W^Q (x_j W^K + a_{ij}^K)^T}{\sqrt{d_z}}$$

## 5.2 Rotary Position Embeddings (RoPE)
Metode standar untuk mayoritas model pasca-2024 (termasuk LLaMA, PaLM) adalah **RoPE**.
**Landasan Teoretis:** Sebuah penyematan posisi relatif yang ideal harus berupa fungsi $f(x, i)$ sedemikian rupa sehingga hasil perkalian titik (*inner product*) dari *Query* dan *Key* hanya bergantung pada *selisih* posisi mereka (jarak relatif $i - j$):
$$\langle f(x, i), f(y, j) \rangle = g(x, y, i - j)$$

Sayangnya, metode sinus orisinal menghasilkan suku silang (*cross-terms*) yang tidak murni relatif. Metode absolut jelas tidak relatif. Dan persamaan metode relatif di atas secara teknis bukanlah murni *inner product*.

**Solusi RoPE:** Bagaimana agar *inner product* invarian (kebal) terhadap posisi absolut, namun peka terhadap jarak antar posisi?
Konsep matematikanya dipinjam dari **Bilangan Kompleks** (Geometri 2D). Rotasi sembarang sebuah vektor dalam ruang 2D akan mengubah posisinya, *tetapi sudut relatif antara dua vektor yang diputar bersamaan akan tetap konstan*. 
Oleh karena itu, RoPE bekerja dengan cara mempasangkan koordinat dari representasi fitur secara berpasangan (dalam 2D), lalu memutarnya berdasarkan posisinya di dalam sekuens. 

**Matematika RoPE (Rotasi Matriks):**
Transformasi pada vektor $x_m$ pada posisi $m$ dirumuskan dengan mengalikannya dengan matriks rotasi blok diagonal $\boldsymbol{R}_{\Theta, m}^d$:
$$f_{\{q,k\}}(x_m, m) = \boldsymbol{R}_{\Theta, m}^d \boldsymbol{W}_{\{q,k\}} x_m$$
Di mana $\boldsymbol{R}_{\Theta, m}^d$ didefinisikan sebagai blok matriks *sparse* (hanya diagonal) berisikan rotasi 2D:
$$
\begin{pmatrix}
\cos m\theta_1 & -\sin m\theta_1 & 0 & 0 & \cdots & 0 & 0 \\
\sin m\theta_1 & \cos m\theta_1 & 0 & 0 & \cdots & 0 & 0 \\
0 & 0 & \cos m\theta_2 & -\sin m\theta_2 & \cdots & 0 & 0 \\
0 & 0 & \sin m\theta_2 & \cos m\theta_2 & \cdots & 0 & 0 \\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
0 & 0 & 0 & 0 & \cdots & \cos m\theta_{d/2} & -\sin m\theta_{d/2} \\
0 & 0 & 0 & 0 & \cdots & \sin m\theta_{d/2} & \cos m\theta_{d/2} \\
\end{pmatrix}
$$
Berbeda dengan *sine embeddings* orisinal yang di-*add* (ditambah), RoPE di-*multiply* (dikalikan), sehingga ia bersifat multiplikatif, bukan aditif, dan tidak menghasilkan *cross terms*.

**Implementasi Kode RoPE (PyTorch / Pseudo-code):**
```python
query_states = self.q_proj(hidden_states)
key_states = self.k_proj(hidden_states)
value_states = self.v_proj(hidden_states)

# Standar persiapan Flash Attention (membentuk ke dimensi head)
# bsz x seq_length x num_heads x head_dim
query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

# Mengambil matriks kosinus dan sinus dari RoPE untuk ID posisi saat ini
cos, sin = self.rotary_emb(value_states, position_ids)

# Mengaplikasikan fungsi rotasi HANYA pada query dan key (Value tidak diputar)
query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

# ... Dilanjutkan dengan operasi dot-product attention multi-head konvensional
```
*Catatan Penting:* Penyematan ini diaplikasikan **setiap kali** operasi *attention* dijalankan (di setiap *layer*) untuk menegakkan invarian posisi secara menyeluruh.

---

# BAB 6: Rasio dan Hiperparameter yang Menjadi Konsensus

Meskipun terlihat bebas, perancangan model sangat terikat oleh prinsip-prinsip heuristik empiris ("*Rules of Thumb*").

## 6.1 Rasio Dimensi Model vs. Dimensi Feedforward
Hiperparameter yang paling universal adalah rasio ukuran dimensi jaringan *Feed-forward* ($d_{ff}$) terhadap dimensi model ($d_{model}$).
Secara klasik, aturan ini selalu bernilai:
$$d_{ff} = 4 \times d_{model}$$

**Pengecualian 1: Penggunaan Varian GLU**
Seperti yang dibahas sebelumnya, varian *Gated Linear Units* meningkatkan parameter dengan menambahkan satu matriks. Agar total jumlah parameter (*FLOPs budget*) tetap setara dengan model tradisional berskala sama, $d_{ff}$ harus diukur ulang (diperkecil dengan faktor $2/3$ dari rasio awal).
$$d_{ff} = \frac{8}{3} d_{model} \approx 2.66 \times d_{model}$$
*Data Pendukung:* 
- LLaMA 70B: Rasio 2.68
- Qwen 14B: Rasio 2.67
- DeepSeek 67B: Rasio 2.68
- Mistral 7B dan LLaMA-2 70B sedikit membesarkan rasio ini di angka 3.5.

**Pengecualian 2: Kasus Radikal T5**
Pada model T5 11B (Raffel et al., 2020), dimensi diatur dengan cara yang tidak lazim:
$$d_{model} = 1024$$
$$d_{ff} = 65,536$$
Ini menghasilkan pengali sebesar **64x**! Mereka menggunakan arsitektur sangat *lebar* di FFN karena efisiensi spesifik perangkat akselerator TPU. Namun, grafik hukum skala Kaplan (2020) menunjukkan bahwa *Loss Increase* (%) berbentuk huruf U (*basin*). Rasio optimal empiris (di mana *loss* paling rendah/0%) terletak di wilayah nilai pengali 1 hingga 10. Jika lebih dari 10, *loss* meroket naik secara drastis hingga 8%. T5 membuktikan rasio radikal "bisa berjalan", tetapi T5 v1.1 akhirnya merevisi rasio mereka ke angka standar 2.5 (GeGLU), membuktikan rasio 64x adalah suboptimal.

## 6.2 Konfigurasi Attention Heads
Satu prinsip dasar dari *Multi-head Self-Attention* adalah pembagian dimensi secara komputasional efisien.
Heuristik standarnya adalah:
$$\text{Head\_dim} \times \text{Num\_heads} = \text{Model\_dim}$$
Ini memungkinkan matriks berukuran $XQ \in \mathbb{R}^{n \times d}$ di-*reshape* dengan mulus menjadi $\mathbb{R}^{n \times h \times d/h}$. Sebagian besar model patuh pada rasio 1 ini (misalnya GPT-3: 96 *heads* $\times$ 128 *head dim* = 12288 *model dim*). Namun, beberapa model Google seperti PaLM memiliki rasio 1.48, menunjukkan ini adalah panduan, bukan aturan mutlak perangkat keras.

## 6.3 Rasio Aspek (Dalam vs. Lebar)
Seberapa dalam (jumlah *layer*, $n_{layer}$) atau seberapa lebar ($d_{model}$) sebuah model sebaiknya dibuat?
Metrik: $d_{model} / n_{layer}$
Rentang rasio yang diklasifikasikan sebagai *'sweet spot'* (titik optimal) berada di antara **100 hingga 200**.
*   BLOOM: 205
*   PaLM (540B): 156
*   GPT3/OPT/Mistral/Qwen/OLMo 3: 128
*   LLaMA/LLaMA2: 102

Mengapa kita tidak membuat model yang dimensinya kecil tetapi berlapis sangat dalam (misal ratusan ribu *layer*)? Tay et al. (2021) menjelaskan batas kelayakan praktis (sistem): **Jaringan yang terlalu dalam amat sangat sulit diparalelkan dan memiliki latensi eksekusi yang tinggi**. Dalam sistem arsitektur kluster multi-GPU, *pipeline parallelism* memaksa Layer 1 di GPU 1 harus menunggu komputasi dari Layer 0 di GPU 0. *Kedalaman menjadi penghalang paralelisasi sekuensial yang merugikan*, sedangkan memperbesar lebar matriks ($d_{model}$) sangat mudah didistribusikan ke ribuan alat secara simultan (melalui *Tensor Parallelism*).

## 6.4 Ukuran Kosakata (Vocabulary Size)
Berdasarkan data observasi, pemilihan ukuran jumlah representasi token bergantung murni pada tujuan model:
*   **Model Monolingual:** Cukup berukuran kecil, di kisaran **30.000 hingga 50.000**. (Contoh: GPT, LLaMA dengan ~32k).
*   **Model Multilingual / Skala Produksi:** Membutuhkan jangkauan token yang luas untuk menampung bahasa asing dan *coding*, berkisar di antara **100.000 hingga 250.000**. (Contoh: Gemma 4 - 262.144, GPT4 - 100.276).

---

# BAB 7: Regularisasi dalam Pra-Pelatihan (Pretraining)

Apakah kita memerlukan regularisasi (mencegah *overfitting*) saat melatih model dasar dari nol (*pretraining*)?
*   Argumen Teoretis yang Menentang: Korpus teks berisi triliunan token, secara teoretis melampaui jumlah kapasitas parameter model. Pelatihan SGD (*Stochastic Gradient Descent*) umumnya hanya menjalankan iterasi satu putaran penuh (*single pass/1 epoch*) sehingga nyaris mustahil bagi model untuk sekadar "menghafal" (*memorize*) data latih.

**Praktik di Lapangan:**
Model-model generasi awal (GPT-2, T5, GPT-3, OPT) masih menggunakan **Dropout = 0.1** selama proses *pretraining*. Namun, model-model mutakhir (LLaMA, PaLM, T5 v1.1) memutuskan untuk **menghilangkan Dropout secara penuh (0)**, dan kini hanya bergantung secara eksklusif pada **Weight Decay = 0.1** (kecuali arsitektur spesifik seperti Qwen yang masih mempertahankan *dropout* 0.1).

**Fungsi Sebenarnya dari Weight Decay:**
Studi komprehensif dari Andriushchenko et al. (2023) menunjukkan temuan radikal: **Weight Decay (WD) pada LLM modern tidak ditujukan untuk mencegah *overfitting*!**
*   Dalam plot matriks kerugian pelatihan (*Training Loss*) versus iterasi, terlihat bahwa model tanpa WD (garis merah $\lambda = 0.0$) pada awal konvergensi memiliki performa serupa, tetapi kurva yang memanfaatkan WD ($\lambda = 0.1$ garis hijau, $\lambda = 0.3$ garis biru) berinteraksi lebih superior dengan *Cosine Learning Rate Schedule*.
*   Di kurva plot lain, menunjukkan korelasi hampir linear antara *Validation loss* (Y) dan *Training loss* (X). Ini berarti model tidak menderita memori (*overfit*). 
Fungsi esensial dari penyusutan bobot (*regularisasi*) di sini semata-mata memodifikasi dinamika ruang optimisasi agar pencarian gradien tetap mulus.

---

# BAB 8: Trik Stabilitas Pelatihan

Selama pelatihan model bernilai jutaan dolar, fenomena yang paling ditakuti adalah ketidakstabilan di mana metrik *Loss* tiba-tiba mengalami lonjakan raksasa (disebut sebagai '*blue curve*' pada grafik perbandingan pelatihan OLMo, di mana garis loss berwarna biru mengalami ratusan paku/spikes ekstrem mencapai skala y=3.0+, dan divergensi pada norma L2 gradiennya). Lonjakan ini menghancurkan bobot yang sudah terlatih. 

Titik hancur yang paling sering bermasalah ada di satu operasi matematika: **Softmax**.
Eksekusi eksponensial dalam *softmax*, diikuti oleh pembagian (*division*) berisiko tinggi menghasilkan fenomena *Not-a-Number* (NaN) jika terjadi *overflow* (*division by zero*).

Terdapat beberapa trik untuk menjinakkan *Softmax*:

## 8.1 Trik 'Z-loss' (Output Softmax Stability)
Mengingat kembali operasi probabilitas *Softmax*:
$$\log(P(x)) = \log\left(\frac{e^{U_r(x)}}{Z(x)}\right) = U_r(x) - \log(Z(x))$$
Di mana fungsi partisi (normalizer) didefinisikan sebagai $Z(x) = \sum_{r'=1}^{|V|} e^{U_{r'}(x)}$.
Devlin (2014) memperkenalkan fungsi auxiliary *z-loss* agar normalizer $Z$ tidak "meledak". Model PaLM Google menerapkan trik ini untuk stabilitas pelatihan:
$$L = \sum_i \left[ \log(P(x_i)) - \alpha (\log(Z(x_i)) - 0)^2 \right]$$
$$L = \sum_i \left[ \log(P(x_i)) - \alpha \log^2(Z(x_i)) \right]$$
Dengan mendorong ekspektasi nilai $\log(Z)$ agar mendekati 0 dengan pengali skala $\alpha$ kecil (misal $10^{-4}$), stabilitas dipaksa meningkat tanpa merusak utilitas fungsi objektif utama. Model yang mengadopsi ini: Baichuan 2, DCLM, OLMo 2 & 3.

## 8.2 Trik 'QK Norm' (Attention Softmax Stability)
Operasi *softmax* internal di dalam MHA (saat $Q$ dikali $K$) juga rentan terhadap eksponensial ekstrem. Triknya (diadopsi dari visi komputer seperti Dehgani 2023) adalah dengan mengaplikasikan `LayerNorm` atau `RMSNorm` **secara eksplisit pada tensor representasi Query ($Q$) dan Key ($K$)** sebelum operasi perkalian titik dilakukan (*before the softmax operation*).
*Diadopsi oleh:* DCLM, OLMo2, Gemma 2, Qwen3, OLMo 3, Gemma 4. (Grafik komik lelucon mendeskripsikan teknik ini sebagai "Tambahkan lebih banyak Layer Norms!").

## 8.3 Logit Soft-Capping
Trik ketiga untuk mencegah logit *blow-up* (nilai meledak tak terbatas) adalah secara paksa "memangkas lunak" ( *soft-capping*) rentang nilai maksimum logit menggunakan fungsi tangen hiperbolik ($\tanh$).
Rumusnya:
$$\text{logits} \leftarrow \text{soft\_cap} * \tanh(\text{logits} / \text{soft\_cap})$$
Batas $\text{soft\_cap}$ biasa diatur pada nilai $50.0$ untuk layer atensi, dan $30.0$ untuk layer probabilitas final.
Kelemahan metrik ini: meskipun sangat stabil, teknik ini dapat sedikit melukai perplexity baseline arsitektur (menurut tabel performa *benchmark*, baseline 11.19 memburuk menjadi 11.24 jika *soft cap* diaktifkan tanpa *QK Norm*).

---

# BAB 9: Varian Perhatian (Attention Variants) untuk Peningkatan Efisiensi

Mayoritas arsitektur model mempertahankan desain *heads attention* tanpa modifikasi drastis. Namun, kendala raksasa muncul saat menginjak fase inferensi (teks generasi) yaitu masalah **KV Cache**.

## 9.1 Memahami Akar Permasalahan Komputasi Generasi
Mengapa inferensi mahal? Mari perhatikan operasi dasar *Multi-Head Attention* (MHA).
Perkalian melibatkan: Softmax$(X Q K^T X^T) X V = \text{Output} \in \mathbb{R}^{n \times d}$.
Variabel: $d$ (dimensi hidden), $b$ (ukuran *batch*), $n$ (panjang sekuens context), $h$ (jumlah *heads*), $k$ (dimensi *head*, $d/h$).

Untuk pelatihan sekaligus, seluruh sekuens tersedia. Namun, untuk teks generasi, hal ini tidak bisa diparalelkan. Model harus memprediksi token satu-demi-satu (*step-by-step*). Untuk menghindari komputasi ulang vektor lama berulang-ulang, *Attention* diperbarui menggunakan mekanisme penyimpanan memori yang disebut **KV Cache**. 
Masalahnya terletak pada **Intensitas Aritmetika (Arithmetic Intensity)**. 
Total memori akses dalam KV Cache inferensi berbanding lurus dengan persamaan $bn^2d + nd^2$.
Intensitas aritmetika model dirumuskan dengan $O((n/d + 1/b)^{-1})$. Suku $n/d$ sangat sulit dihilangkan. Seiring $n$ memanjang (misal memori percakapan beribu-ribu karakter), atau memuat model $d$ berukuran raksasa dengan $b$ rendah, batas kecepatan sistem tidak lagi bertumpu pada daya hitung FLOPs GPU, melainkan tercekik oleh kecepatan pergerakan transfer memori (*Memory bandwidth bound*). 

## 9.2 Solusi 1: MQA dan GQA
Bagaimana kita mengurangi lalu lintas muatan memori ini? 

*   **Multi-Query Attention (MQA):** Mengurangi jumlah dimensi fitur kunci/nilai (*Key/Value*). Pada MQA, masih terdapat sekumpulan panjang *Query Heads*, tetapi semua query tersebut secara memori dipaksa untuk 'melihat' **HANYA SATU** kepala tunggal referensi untuk matriks memori *Key* ($K$) dan matriks *Value* ($V$). Data yang ditarik dari VRAM jauh lebih kecil (hanya 1 dimensi, alih-alih diulang $h$ kali). Hasilnya, intensitas aritmetik membaik, secara dramatis membebaskan alokasi memori. 
*(Catatan: Mengakibatkan sedikit pinalti terhadap kapabilitas metrik model pada benchmark PPL/Perplexity, dari 29.9 naik menjadi 30.2 berdasar Shazeer 2019).*
*   **Grouped-Query Attention (GQA):** Titik tengah pragmatis. Alih-alih mereduksi menjadi hanya 1 kepala tunggal $K/V$, GQA memecahnya menjadi "grup" (misalnya rasio 1 matriks KV melayani 4 matriks Query). Grafik eksperimen Ainslie (2023) menunjukkan bahwa secara performa GQA memelihara akurasi layaknya model MHA penuh, namun secara kecepatan latensi sistem (ms per sampel), GQA sama cepatnya dengan arsitektur MQA.

## 9.3 Solusi 2: Sparse / Sliding Window Attention (SWA)
Memperhatikan seluruh masa lalu dalam konteks ($n$) memakan komputasi kuadratik secara alami ($n^2$). Model-model seperti GPT-3, GPT-OSS, dan Gemma 4 mengimplementasikan *Sliding Window Attention*.
Secara mekanis, alih-alih matriks persegi penuh di mana token $T_z$ melihat seluruh riwayat token dari $T_0$ hingga $T_{z-1}$, atensi dihalangi (di-*mask*) sedemikian rupa sehingga ia **hanya boleh melihat $N$-jendela token ke belakang secara lokal** (contoh: matriks *banded-diagonal* tebal). Hal ini mengorbankan kapasitas penangkapan ekspresi konteks tak terbatas namun mendongkrak durasi *runtime* pelolosan data di atas arsitektur model berskala panjang.

## 9.4 Solusi 3: Interleaved Attention (Atensi Berselang)
Trik arsitektur mutakhir (standar tahun 2024-2025) adalah pencampuran. Sebagai contoh dari arsitektur Cohere Command A: 
*   Mereka membiarkan sebagian blok beroperasi sebagai *Sliding Window Attention* (SWA) + RoPE (berfungsi membaca pola kedekatan/jarak pendek).
*   Lalu secara selektif, *setiap siklus lapisan ke-4* disisipkan menjadi model Atensi Normal/Penuh (*Full Attention*) untuk berfungsi memanen hubungan informasi struktural berjangka jauh.
*   *Implementasi Model Sejenis:* LLaMA 4, Gemma 3/4, dan OLMo 3. Desain Gemma 4 dan Qwen 3.5 turut mencampurkan berbagai metode Gating eksotis pada arsitekturnya.