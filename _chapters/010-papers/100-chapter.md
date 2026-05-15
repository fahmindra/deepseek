---
slug: ch-10-10
title: "Membedah DeepSeek-V4"
layout: chapter
---

# Membedah DeepSeek-V4 dan Pendahulunya: Mulai dari Fondasi Dasar Hingga Teknologi Tingkat Lanjut
**Panduan Praktis Memahami mHC, CSA, HCA, dan Muon Optimizer**

Oleh: James Koh, PhD (Diterjemahkan & Diadaptasi untuk Pembelajaran Mahasiswa)

Halo teman-teman mahasiswa! Tulisan ini awalnya dibuat untuk kelas *Computer Science* (CS610) semester April 2026. 

Perilisan DeepSeek-R1 pada awal 2025 sempat mengguncang dunia AI, apalagi ketika mereka merilis model *open-source* yang performanya setara dengan OpenAI-o1-mini, yang membuat saham-saham perusahaan AI raksasa anjlok. Setahun kemudian, DeepSeek merilis **V4** (April 2026) yang performanya setara dengan model-model raksasa berbayar seperti Claude-Opus-4.6 atau GPT-5.4.

Sebagai mahasiswa IT, *Data Science*, atau sekadar penggiat AI, sangat penting bagi kalian untuk paham **apa yang sebenarnya terjadi di balik layar model ini**. Kalau kalian bisa menjelaskan arsitektur ini saat wawancara kerja atau magang, kalian pasti akan sangat *stand out*. 

Kabar baiknya (sekaligus tantangannya), setiap versi baru dari DeepSeek selalu dibangun di atas fondasi yang sudah ada. Jadi, kalian nggak bisa cuma baca paper V4 saja. Di sini, kita akan bedah semuanya langkah demi langkah agar kalian paham seutuhnya.

---

## 1. Fondasi Dasar yang Pantang Kalian Lewati

Saya selalu percaya bahwa kita harus paham bagaimana sebuah *input* (masukan) diubah menjadi *output* (keluaran). Mari kita mulai dari konsep yang paling awal. Sebuah LLM (seperti DeepSeek) menerima *input* berupa teks (kumpulan karakter/huruf).

### 1.1 *Byte Pair Encoding* (BPE)

Sebelum teks bisa diubah menjadi angka/vektor, kita harus memecahnya menjadi bagian-bagian kecil bernama **token**. 

*Byte Pair Encoding* (BPE) adalah teknik tokenisasi di level *sub-word* (suku kata/potongan kata). Ini adalah jalan tengah yang sempurna: kalau kita memecah per huruf, AI akan kehilangan makna, tapi kalau kita memecah per kata utuh, AI akan kesulitan mengenali kata baru.

BPE bekerja dengan cara menggabungkan pasangan simbol/huruf yang paling sering muncul secara berulang-ulang. Kata-kata yang sering dipakai biasanya akan menjadi satu token utuh. DeepSeek-V3, misalnya, punya kamus (*vocabulary*) sebanyak 128.000 token.

Mari kita lihat contoh kodenya di Python menggunakan model DeepSeek versi 7B (agar ringan dan bisa kalian coba sendiri). Coba perhatikan kalimat: *"My father works in a financial corporation."*

```python
import torch
from transformers import AutoTokenizer, AutoModel

# Kita memuat tokenizer dari model DeepSeek
tokenizer = AutoTokenizer.from_pretrained(
    "deepseek-ai/deepseek-llm-7b-base"
)
sentence = "My father works in a financial corporation."

tokens = tokenizer.tokenize(sentence, return_tensors="pt")
token_ids = tokenizer(sentence)["input_ids"]

print("Tokens:", tokens)
print("Token IDs:", token_ids)
```

**Hasilnya:**
```text
Tokens: ['My', 'father', 'works', 'ina', 'financial', 'corpor', 'ation', '.']
Token IDs: [3673, 13446, 5774, 1695, 75293, 39656, 335, 13]
```
Perhatikan bahwa kata *'in a'* digabung jadi satu token (ID: 1695), sementara kata *'corporation'* dipecah jadi dua token: *'corpor'* dan *'ation'*. Kalian bisa mengembalikannya ke teks asli dengan perintah `tokenizer.decode()`.

### 1.2 *Embeddings*

Setelah punya token, komputer butuh representasi matematikanya. Inilah yang disebut **Embedding**. Bayangkan *embedding* sebagai vektor profil dari sebuah token—apakah dia kata benda, kata kerja, nuansanya positif/negatif, dll. Nilai ini dipelajari oleh model selama masa *training*.

Mari kita ambil vektor *embedding* dari model:

```python
model = AutoModel.from_pretrained(
    "deepseek-ai/deepseek-llm-7b-base", output_hidden_states=True
)
model.eval() # Set model ke mode evaluasi

# Ubah teks jadi tensor
token_ids = tokenizer(sentence, return_tensors="pt")["input_ids"]

with torch.no_grad():
    token_embeddings = model.embed_tokens(token_ids)
```

*Output*-nya adalah tensor dengan bentuk `[1, jumlah_token, 4096]`. Angka 4096 ini adalah dimensi ruang *embedding*-nya. Kalau kita intip 4 nilai pertamanya:

```text
token_embeddings[0, :, :4]

tensor([[-3.6621e-02, -1.0010e-01,  3.4375e-01,  3.0469e-01], # 'My'
        [ 5.1025e-02,  4.7119e-02,  1.8555e-01, -1.2695e-01], # 'father'
        ...
        [-4.1260e-02, -2.4986e-04,  3.8281e-01, -1.0620e-02]], # '.'
       dtype=torch.bfloat16)
```

**Masalahnya:** Vektor *embedding* mentah ini belum paham **konteks kalimat**. Kalau kita hitung *cosine similarity* (kemiripan arah vektor) antara kata *"works"* dan *"corpor"*:

```python
emb_1 = token_embeddings[:,2,:] # token 'works'
emb_2 = token_embeddings[:,5,:] # token 'corpor'

from torch.nn.functional import cosine_similarity
cosine_similarity(emb_1, emb_2)
# Hasilnya: tensor([0.04...])
```
Nilainya cuma **0.04** (sangat kecil). Kenapa? Karena vektor dasarnya belum dihubungkan satu sama lain. Nanti, setelah masuk ke lapisan akhir model (*last hidden state*), nilai kemiripannya naik jadi **0.707** karena model sudah paham konteks kalimatnya.

### 1.3 Memasukkan Konteks

Coba bayangkan kalimat *"Bapak bekerja di bank"*. Kata *"bank"* bisa berarti institusi keuangan, bisa juga pinggiran sungai (*river bank*). Konteksnya sangat bergantung pada kata-kata di sekitarnya (seperti kata *"bekerja"*). 

Jadi, kita butuh cara untuk memodifikasi vektor awal ($v_i$) menjadi vektor baru ($y_i$) yang sudah menyerap informasi dari tetangga-tetangganya. Ini dilakukan dengan mengalikan *input* dengan matriks bobot yang bisa dipelajari (*learnable matrix*).

Secara matematika aljabar linear, kita membuat representasi *Query* ($Q$), *Key* ($K$), dan *Value* ($V$):
$$ Q = XW_Q \quad K = XW_K \quad V = XW_V $$

Dimana $X$ adalah matriks vektor *input*, sedangkan $W_Q, W_K, W_V$ adalah matriks bobot.

### 1.4 *Positional Encodings* (Sandi Posisi)

Mekanisme jaringan saraf sebenarnya "buta" terhadap urutan kata. Padahal, kalimat *"tidak sepenuhnya"* dan *"sepenuhnya tidak"* maknanya beda jauh.

Awalnya, di paper Transformer (Vaswani, 2017), digunakan **Sinusoidal Positional Encodings** (fungsi sinus dan cosinus):
$$ PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right) $$
$$ PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right) $$

Rumus ini sekadar memberi "tanda pengenal" unik berupa angka gelombang pada setiap posisi token (`pos`), lalu nilai ini **ditambahkan** ke vektor *embedding*.

Namun, arsitektur modern seperti DeepSeek menggunakan **RoPE** (*Rotary Positional Encodings*). Alih-alih ditambah, RoPE **memutar** vektor di dalam ruang dimensinya. Bayangkan ruang dimensi dibagi berpasangan, lalu setiap pasangan diputar dengan sudut $\theta$ berdasarkan posisinya ($m$):

$$ R_{\Theta, m}^d x = \begin{pmatrix} x_1 \\ x_2 \\ \vdots \\ x_{d-1} \\ x_d \end{pmatrix} \otimes \begin{pmatrix} \cos m\theta_1 \\ \cos m\theta_1 \\ \vdots \\ \cos m\theta_{d/2} \\ \cos m\theta_{d/2} \end{pmatrix} + \begin{pmatrix} -x_2 \\ x_1 \\ \vdots \\ -x_d \\ x_{d-1} \end{pmatrix} \otimes \begin{pmatrix} \sin m\theta_1 \\ \sin m\theta_1 \\ \vdots \\ \sin m\theta_{d/2} \\ \sin m\theta_{d/2} \end{pmatrix} $$
$$ \theta_i = 10000^{-2(i-1)/d} $$

DeepSeek-V3 bahkan memperluas ini menggunakan teknik YaRN untuk bisa menangani teks yang sangat panjang.

### 1.5 Mendapatkan *Output* (Konsep *KV Cache*)

Saat LLM menjawab pertanyaan kalian, ia sebenarnya sedang menebak **satu per satu token**. Setelah menebak satu token, token itu digabungkan ke pertanyaan awal, lalu model menebak token berikutnya lagi (*autoregressive*).

Tentu ini terlihat membuang waktu karena harus menghitung ulang kalimat dari awal berkali-kali. Solusinya adalah **Key-Value Cache (KV Cache)**. Model akan "mengingat" matriks $K$ dan $V$ dari kata-kata sebelumnya, sehingga untuk kata baru, ia hanya perlu memproses perhatiannya terhadap masa lalu tanpa menghitung ulang dari nol.

---

## 2. Balok Pembangun (Di Balik DeepSeek-V3)

DeepSeek-V3 memiliki 61 *layer* Transformer. Setiap *layer* berisi komponen seperti RMSNorm, MLA (*Multi-head Latent Attention*), MoE (*Mixture of Experts*), dan *skip connections*. Mari kita bahas satu per satu.

### 2.1 *Attention* (Perhatian)

Fungsi utama *Attention* adalah memungkinkan setiap kata untuk "melihat" seberapa relevan dirinya dengan kata-kata lain di dalam kalimat. Secara matematis, rumusnya sangat elegan:

$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$

Di LLM (*decoder-only*), matriks perhatian ini diberi *causal mask* (topeng kasual), artinya sebuah token hanya boleh memperhatikan dirinya sendiri dan token-token di masa lalu, tidak boleh "mengintip" token di masa depan.

### 2.2 *Multi-Head Latent Attention* (MLA)

Satu *attention* hanya bisa melihat satu jenis hubungan. Bagaimana kalau kita butuh banyak perspektif? Misalnya kalimat: *"Ayahku bekerja di bank yang anti lembur."* Kita butuh satu "kepala" (*head*) yang melihat relasi 'Ayah' dan 'bekerja', lalu kepala lain melihat sentimen 'anti' dan 'lembur'. Di V3, terdapat **128 *attention heads***.

Lalu, apa maksud kata **Latent** di sini?
Tadi kita bahas *KV Cache*. Jika konteksnya jutaan token, memori *cache* akan membengkak luar biasa. Oleh karena itu, MLA **mengompresi** (memampatkan) data *hidden state* ($h_t$) menjadi vektor laten yang lebih kecil ($c_t^{KV}$). 

Mari kita bedah rumus matematikanya:
1. **Kompresi:** $\mathbf{c}_t^{KV} = W^{DKV}\mathbf{h}_t$ (Dimensi direduksi dari 7168 menjadi cuma 512).
2. **Dekompresi menjadi Key (K):** $\mathbf{k}_t^C = W^{UK}\mathbf{c}_t^{KV}$ (Di-ekspansi lagi jadi besar).
3. **Pemisahan Posisi (RoPE):** Posisi dihitung terpisah agar *cache* lebih bersih: $\mathbf{k}_t^R = \text{RoPE}(W^{KR}\mathbf{h}_t)$.
4. **Gabungan K:** $\mathbf{k}_{t,i} = [\mathbf{k}_{t,i}^C; \mathbf{k}_t^R]$.
5. **Dekompresi menjadi Value (V):** $\mathbf{v}_t^C = W^{UV}\mathbf{c}_t^{KV}$.

Hal yang sama persis dilakukan untuk membentuk *Query* (Q). Intinya, MLA adalah teknik jitu untuk menghemat memori (*RAM/VRAM*) secara radikal tanpa merusak kualitas informasi.

### 2.3 *Mixture-of-Experts* (MoE)

Kalau model AI itu punya miliaran parameter, butuh komputer super untuk memproses satu kata. *Mixture-of-Experts* (MoE) adalah solusinya. Bayangkan model sebagai sebuah rumah sakit. Tidak semua dokter perlu mengobati pasien yang sakit gigi, cukup dokter gigi saja.

Di DeepSeek-V3, setiap lapisan MoE punya:
*   **1 Pakar Bersama (*Shared Expert*):** Selalu aktif untuk menangani pengetahuan umum.
*   **256 Pakar Khusus (*Routed Experts*):** Bagian terpisah-pisah. Dari 256 ini, **hanya 8 pakar** yang diaktifkan (*di-routing*) untuk mengurus token tertentu.

Total parameter V3 adalah 671 Miliar, tapi berkat MoE, **hanya sekitar 5.5% parameter** yang aktif pada satu waktu. DeepSeek juga punya mekanisme *Load Balancing* (Penyeimbang Beban). Kalau sebuah *expert* jarang dipanggil *router*, performanya bakal memburuk. DeepSeek menyuntikkan angka "bias" agar pakar yang menganggur sesekali ikut terpilih.

### 2.4 *Multi-Token Prediction* (MTP)

Model biasa memprediksi 1 kata berikutnya. MTP melatih model memprediksi **beberapa kata ke depan sekaligus**. Ini memaksa model untuk "merencanakan" kalimatnya. Secara matematis, fungsinya meminimalkan *loss* (kerugian) prediksi untuk $n$ token sekaligus:

$$ L_n = - \sum_t \log P_\theta(x_{t+n:t+1} \mid x_{t:1}) $$

Secara arsitektur, modul MTP menggabungkan representasi *hidden state* masa lalu dengan *embedding* masa depan, dinormalisasi (RMSNorm), digabungkan (*concat*), dikalikan matriks linier ($M_k$), lalu dilempar ke blok Transformer untuk menebak probabilitas kosakatanya. Menariknya, fitur MTP ini cuma dipakai buat *training* biar model lebih pintar; saat sedang di-*deploy* (*inference*), modul ini dibuang agar AI lebih ringan!

---

## 3. Hal Baru di DeepSeek-V4: *Beyond Basics*

Di V4, tim DeepSeek mempertahankan arsitektur dasar V3 (MLA, MoE, MTP) tapi melakukan *upgrade* besar-besaran untuk efisiensi "jutaan konteks token".

### 3.1 *Manifold-Constrained Hyper-Connections* (mHC)

#### 3.1.1 *Hyper-Connections* (HC)
Dalam *Deep Learning*, kita memakai *Residual/Skip Connections* untuk menghindari masalah *gradient vanishing* (sinyal gradien hilang). Bentuk standarnya: $x + F(x)$. 
Namun menurut paper Zhu et al. (2025), ini memunculkan *trade-off*:
*   **Pre-Norm:** Rentan terhadap *representation collapse* (semua *layer* di ujung bentuknya jadi mirip-mirip semua, jadi *layer* ekstra tidak ada gunanya).
*   **Post-Norm:** Rentan masalah *vanishing gradient*.

Solusinya adalah *Hyper-Connections* (HC), yang mengubah *skip connection* sederhana menjadi matriks bobot ($H$) yang menjembatani hubungan antar kedalaman jaringan secara menyilang:
$$ \mathbf{H} = \begin{bmatrix} \alpha_{1,1} & \dots & \alpha_{1,n} \\ \beta_{1,1} & \dots & \beta_{1,n} \end{bmatrix} $$
$$ \begin{bmatrix} \mathbf{h}_1^{k+1} \\ \mathbf{h}_2^{k+1} \end{bmatrix} = \mathbf{H} \begin{bmatrix} \mathcal{T}(\mathbf{h}_1^k) \\ \mathbf{h}_2^k \end{bmatrix} = \begin{bmatrix} \alpha_{1,1}\mathcal{T}(\mathbf{h}_1^k) + \alpha_{1,2}\mathbf{h}_2^k \\ \beta_{1,1}\mathcal{T}(\mathbf{h}_1^k) + \beta_{1,2}\mathbf{h}_2^k \end{bmatrix} $$

Intinya: *Output* tidak cuma ditambah mentah-mentah, tapi dicampur-campur secara proporsional.

#### 3.1.2 Memasukkan *Manifold-Constrained* (mHC)
Masalahnya, kalau cuma dikali-kali saja, saat jaringannya sangat dalam, angkanya bisa meledak (*numerical instability*). Tim DeepSeek (Xie et al., 2026) memperkenalkan **mHC**. 

Mereka **membatasi/mengekang** (*constrain*) matriks campurannya ($H_l^{res}$) agar tunduk pada aturan **doubly stochastic matrix**. Rumus sakturnya:
$$ H_l^{res} \in \mathbb{R}_{+}^{n \times n}, \quad H_l^{res} \mathbf{1}_n = \mathbf{1}_n, \quad \mathbf{1}_n^T H_l^{res} = \mathbf{1}_n^T $$

Nggak usah pusing dulu, arti matematisnya sederhana:
1. Angkanya nggak boleh negatif ($\mathbb{R}_{+}$).
2. Kalau seluruh angka di satu baris dijumlahkan, hasilnya mutlak = 1.
3. Kalau seluruh angka di satu kolom dijumlahkan, hasilnya mutlak = 1.

Karena batas ini, angka-angka dalam *skip connections* tidak akan pernah tumpah ruah (*bounded* antara 0 dan 1). Perhitungannya jadi sangat stabil meskipun *layer*-nya ada 61 lapis dengan dimensi d=7168.

### 3.2 *Compressed Sparse Attention* (CSA)

Tujuan CSA adalah memangkas habis biaya komputasi saat kita mengumpan dokumen PDF tebal atau buku (jutaan token) ke AI.

#### 3.2.1 Mengingat Kembali DSA (V3.2)
Sebelum CSA, ada DSA. Konsepnya: kita pakai *Lightning Indexer* untuk menilai (*scoring*) token masa lalu mana yang relevan dengan pertanyaan saat ini. Setelah di-*score*, kita cuma ambil *"Top-K"* (misal: 1024 token terbaik). Sisa memori langsung dicuekin. Jadi *Attention* nggak perlu menghitung semua memori.

#### 3.2.2 Masuk ke CSA (V4)
CSA adalah versi *steroid* dari DSA. Di CSA, yang dinilai bukan per-token, melainkan **blok token yang sudah dikompresi**.
Secara matematis (Persamaan 9-12):
*   $C^a$ dan $C^b$ adalah hasil proyeksi dua arah aliran dari urutan input ($H$) dikalikan matriks bobot $W_{KV}$.
*   $Z^a$ dan $Z^b$ menyediakan logit (nilai mentah).
*   Kemudian $m$ buah elemen (di V4, $m=4$) dikompresi jadi satu entri memori $C^{comp}$:
    $$ C_i^{comp} = \sum_{j=1}^m \text{softmax}([Z_{i,j}^a, Z_{i,j}^b]) \odot [C_{i,j}^a + B^a, C_{i,j}^b + B^b] $$

Sama seperti DSA, ada *Lightning Indexer* (matriks $W_{DQ}, W_{IUQ}$) yang mengalikan *Query* dengan memori blok kompresi ini (Persamaan 13-17). Jika skornya tinggi (masuk *Top-K*), memori blok ini ($C_{t_{sprs}}^{comp}$) baru akan diproses lebih lanjut pakai perhitungan raksasa MQA (*Multi-Query Attention*).

Bayangkan kalau ada 1.000.000 token. Karena kompresi $m=4$, jumlahnya menyusut jadi 250.000 memori blok. Dari situ, *Indexer* menyeleksi lagi dan **hanya mengambil 1024 memori blok**. Super ringan!

### 3.3 *Heavily Compressed Attention* (HCA)

Kalau CSA mengompresi 4 token jadi 1 lalu diseleksi, **HCA mengompresi secara super ekstrem (128 token jadi 1) TAPI tidak diseleksi (tidak di-*sparse*).**

Kenapa butuh ini? HCA bertugas memberikan AI gambaran besar/pandangan mata burung (*global view*) dari sebuah teks panjang. Rumusnya (Persamaan 20-26) mirip CSA, tapi rasio kompresinya $m' = 128$.

Jadi untuk 1 juta token, HCA akan memadatkan memori menjadi cuma sekitar $1.000.000 / 128 \approx 7.800$ entri *cache*. Model kemudian membaca *semua* 7.800 entri ini tanpa ada yang dibuang.

*   Untuk mengerti detail paragraf/kalimat terbaru, DeepSeek-V4 pakai memori asli 128 token terakhir (*sliding window*).
*   Untuk detail dokumen secara spesifik, ia pakai CSA.
*   Untuk rangkuman dan pemahaman benang merah dokumen, ia pakai HCA.
*(CSA dan HCA digunakan secara selang-seling antar layer).*

### 3.4 *Muon Optimizer*

Kalian mungkin terbiasa *training* model pakai *optimizer* **SGD** atau **AdamW**. DeepSeek-V4 menggunakan **Muon**.
Namanya merupakan singkatan teknis: **M**oment**U**m **O**rthogonalized by **N**ewton-Schulz.

Intinya, *Muon* mengambil pembaruan gradien (seperti di SGD-Momentum), tapi **disejajarkan secara ortogonal** terlebih dahulu sebelum diterapkan ke bobot *neural network*. Biasanya proses mencari matriks ortogonal butuh perhitungan SVD (*Singular Value Decomposition*) yang sangat berat untuk GPU. Tapi *Muon* mengakali ini menggunakan perhitungan iteratif sederhana bernama **Newton-Schulz (NS)**.

Secara konseptual, mari bandingkan rumusnya:
**SGD (Tanpa momentum):**
$W_{baru} = W_{lama} - \text{learning\_rate} \times \nabla f(W)$

**SGD (Dengan Momentum M):**
$M_{baru} = \text{momentum} \times M_{lama} + \nabla f(W)$
$W_{baru} = W_{lama} - \text{learning\_rate} \times M_{baru}$

**Muon (Konsep dasar):**
$M_{baru} = \text{momentum} \times M_{lama} + \nabla f(W)$
$W_{baru} = W_{lama} - \text{learning\_rate} \times \textbf{Newton-Schulz}(M_{baru})$

Di Algoritma 1 paper DeepSeek-V4, langkah iterasi *Newton-Schulz* dijalankan sebanyak 10 kali (kombinasi variabel *a, b, c*) agar konvergensinya cepat dan stabil.

Menariknya, V4 **tidak sepenuhnya** meninggalkan AdamW. Untuk bagian *embedding*, ujung prediksi (*head*), mHC, dan RMSNorm, mereka tetap pakai AdamW. Sisanya baru digempur pakai Muon.

### 3.5 Infrastruktur dan *Post-Training*

Tentu saja, di dunia nyata, model semacam V4 tidak dilatih pakai GPU colok PC biasa. Dalam laporannya, DeepSeek memaparkan manipulasi *engineering* level dewa seperti *"Expert Parallelism yang menggabungkan komunikasi dan komputasi"*, menulis modifikasi kernel CUDA secara *low-level*, dll. 

Fakta menarik: jika kalian cek Apendiks A.1, ada sekitar **270 nama** (insinyur dan peneliti) hanya di bawah divisi *Research & Engineering*. Ini membuktikan skala kolaborasi masif yang diperlukan.

## Kesimpulan

Bagi kalian para mahasiswa, mungkin setahun lagi materi arsitektur ini sudah dianggap "kuno" di dunia nyata yang serba cepat. Tapi, konsep-konsep matematika fundamental, manipulasi matriks, pemahaman tentang efisiensi memori (MLA/CSA/HCA), serta dinamika *training* (mHC dan Muon) yang dibahas di atas akan **tetap menjadi dasar berharga** untuk memahami AI generasi apa pun ke depannya.

Ini adalah panduan komprehensif yang saya harap saya miliki dulu ketika pertama kali belajar. Sekarang, ilmu ini milik kalian. Selamat belajar!