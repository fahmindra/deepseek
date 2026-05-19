---
slug: part-1
layout: part
---
Berikut adalah penjelasan komprehensif, formal, dan mendetail mengenai evolusi arsitektur model bahasa DeepSeek mulai dari V1 (LLaMA-based) hingga V4, berdasarkan analisis kode sumber *library* Transformers Hugging Face yang Anda berikan. 

Penjelasan ini disusun dengan bahasa akademik yang terstruktur agar mudah dipahami oleh mahasiswa ilmu komputer atau kecerdasan buatan, dengan mengaitkan konsep teoritis langsung ke implementasi kodenya.

---

### 1. DeepSeek LLM V1 (Berbasis LLaMA)
**File Referensi:** `modeling_llama.py`

DeepSeek generasi pertama tidak memiliki arsitektur khusus yang radikal. Model ini pada dasarnya mengadopsi arsitektur **Dense Transformer standar** yang sangat mirip dengan LLaMA (Meta). 

**Karakteristik Utama dalam Kode:**
*   **Arsitektur *Dense* (Padat):** Setiap token yang masuk akan melewati seluruh parameter model. Tidak ada mekanisme *Mixture-of-Experts* (MoE). Ini terlihat dari penggunaan kelas `LlamaMLP` yang merupakan *Multilayer Perceptron* standar dengan fungsi aktivasi SwiGLU (`gate_proj`, `up_proj`, `down_proj`).
*   **Mekanisme Atensi Standar:** Menggunakan `LlamaAttention` yang melakukan proyeksi Query (Q), Key (K), dan Value (V) secara konvensional (`q_proj`, `k_proj`, `v_proj`).
*   **Rotary Positional Embedding (RoPE):** Menggunakan `LlamaRotaryEmbedding` standar untuk menyuntikkan informasi posisi token relatif ke dalam representasi Q dan K.

**Kesimpulan V1:** Ini adalah model *baseline*. Kuat dan stabil, tetapi biaya komputasi (FLOPs) meningkat secara kuadratik dan linear terhadap jumlah parameter karena setiap parameter aktif untuk setiap token.

---

### 2. DeepSeek-V2: Revolusi Efisiensi (MLA & DeepSeekMoE)
**File Referensi:** `modeling_deepseek_v2.py` & `configuration_deepseek_v2.py`

Pada V2, DeepSeek melakukan perombakan total untuk menekan biaya inferensi dan pelatihan. Ada dua inovasi utama yang terlihat jelas dalam kode: **Multi-Head Latent Attention (MLA)** dan **DeepSeekMoE**.

**A. Multi-Head Latent Attention (MLA)**
Atensi konvensional memakan memori (KV Cache) yang sangat besar. V2 menyelesaikan ini dengan mengompresi KV ke dalam representasi laten.
*   **Kode Implementasi:** Perhatikan inisialisasi `q_a_proj`, `q_b_proj`, `kv_a_proj_with_mqa`, dan `kv_b_proj` di `DeepseekV2Attention`.
*   **Cara Kerja:** Alih-alih menyimpan K dan V secara penuh, model memproyeksikan *hidden states* ke dimensi laten yang jauh lebih kecil (`kv_lora_rank = 512`). Saat komputasi atensi dibutuhkan, representasi laten ini di-*decompress* kembali menggunakan matriks K dan V.
*   **Pemisahan RoPE:** Di V2, K dan Q dipisah menjadi dua bagian: bagian *nope* (`qk_nope_head_dim` = 128) yang tidak mendapat RoPE, dan bagian *rope* (`qk_rope_head_dim` = 64) yang mendapat RoPE. Hal ini memungkinkan kompresi yang lebih efisien tanpa merusak informasi posisi temporal.

**B. Arsitektur DeepSeekMoE**
V2 beralih dari model *Dense* ke *Sparse Mixture-of-Experts* (MoE).
*   **Kode Implementasi:** Kelas `DeepseekV2Moe`.
*   **Shared & Routed Experts:** Tidak seperti MoE tradisional, V2 membagi pakar (*experts*) menjadi dua:
    1.  **Shared Experts (`n_shared_experts = 2`):** Pakar yang **selalu aktif** untuk setiap token, berfungsi menangkap pengetahuan umum/dasar.
    2.  **Routed Experts (`n_routed_experts = 64`):** Pakar spesifik, di mana *router* (`self.gate`) hanya akan memilih sejumlah kecil pakar (Top-K) untuk setiap token menggunakan metode seperti `"greedy"` atau `"group_limited_greedy"` untuk keseimbangan beban.
*   **Hybrid Dense-MoE:** Parameter `first_k_dense_replace` mengatur agar beberapa *layer* paling awal tetap menggunakan MLP *Dense* konvensional sebelum beralih ke layer MoE di layer-layer berikutnya.

---

### 3. DeepSeek-V3: Penskalaan Masif & Optimasi Logika
**File Referensi:** `modeling_deepseek_v3.py` & `configuration_deepseek_v3.py`

V3 pada dasarnya adalah arsitektur V2 yang dioptimalkan dan diskalakan ke ukuran raksasa (ratusan miliar parameter) dengan perbaikan matematis pada sistem *routing* MoE-nya.

**A. Penskalaan MoE yang Ekstrem**
*   **Kode Implementasi:** `DeepseekV3MoE`.
*   Jumlah *Routed Experts* melonjak drastis dari 64 (di V2) menjadi **256** (`n_routed_experts = 256`).
*   Setiap token kini merutekan dirinya ke **8 pakar** sekaligus (`num_experts_per_tok = 8`), bukan sekadar 1 atau 2 pakar seperti pada MoE model lain.
*   Model tetap menggunakan `first_k_dense_replace = 3`, artinya 3 layer pertama adalah *Dense layer*, sisanya MoE.

**B. Load Balancing Tanpa Auxiliary Loss (Bias Correction)**
*   **Kode Implementasi:** `e_score_correction_bias` pada kelas `DeepseekV3TopkRouter`.
*   Dalam MoE konvensional, *router* cenderung terus-menerus memilih pakar yang sama ("expert collapse"), sehingga diperlukan *auxiliary loss* (penalti kerugian tambahan) yang bisa merusak performa model. 
*   V3 menyelesaikan ini murni dengan menambahkan *bias* (`e_score_correction_bias`) pada logit *router*. Bias ini disesuaikan secara dinamis selama pelatihan untuk memaksa distribusi token yang merata tanpa memodifikasi fungsi *loss* utama secara berlebihan.

**C. Optimasi RoPE Interleave**
*   Penambahan `rope_interleave = True` dan fungsi `apply_rotary_pos_emb_interleave`. Ini adalah optimasi tingkat rendah (memori/komputasi) agar pembagian tensor untuk sin/cos RoPE menjadi lebih cepat dan efisien pada perangkat keras grafis (GPU/TPU).

---

### 4. DeepSeek-V4: Pergeseran Paradigma Radikal
**File Referensi:** `modeling_deepseek_v4.py` & `configuration_deepseek_v4.py`

V4 adalah perombakan total. Model ini dirancang khusus untuk menangani konteks yang sangat panjang (*infinite context*) dengan efisiensi ekstrem, menggunakan tiga inovasi yang sama sekali tidak ada di V1, V2, maupun V3.

**A. Tiga Mekanisme Atensi Berlapis (Bukan Lagi Murni MLA)**
V4 meninggalkan atensi MLA yang seragam di setiap layer dan menggantinya dengan jadwal atensi dinamis per-layer (`layer_types` di konfigurasi).
*   **`sliding_attention`:** Atensi lokal konvensional dengan *sliding window* (`sliding_window = 128`).
*   **`compressed_sparse_attention` (CSA):** (Kelas `DeepseekV4CSACompressor`). Layer ini mengompresi setiap $m$ token (misal: 4 token) menjadi 1 entri KV laten. CSA menggunakan **Lightning Indexer** (`DeepseekV4Indexer`) yang bertugas memilih top-k KV yang terkompresi untuk diperhatikan oleh *Query*. Ini secara drastis mengurangi matriks atensi.
*   **`heavily_compressed_attention` (HCA):** (Kelas `DeepseekV4HCACompressor`). Mengompresi token dengan rasio sangat tinggi ($m'$ = 128 token menjadi 1 entri KV laten). HCA tidak menggunakan *indexer*; model memaksa *Query* untuk memperhatikan seluruh rangkuman sejarah konteks jangka panjang secara agregat.
*   **Arsitektur V4-Pro:** Menggunakan campuran (misal: 2 layer HCA di awal untuk ringkasan global, lalu diselingi secara *interleave* antara CSA dan HCA).

**B. Hash-MoE (Bypass Neural Router di Layer Awal)**
*   **Kode Implementasi:** `DeepseekV4HashRouter`.
*   Alih-alih menggunakan *router* berbasis Jaringan Saraf (Neural Network) yang dipelajari untuk merutekan token, pada layer-layer MoE awal (`mlp_layer_types = "hash_moe"`), V4 menggunakan tabel *lookup* statis yang dibekukan (`tid2eid`). 
*   Artinya, ID token tertentu (secara deterministik/kriptografik melalui *hash*) akan selalu masuk ke pakar tertentu. Hal ini mengurangi latensi komputasi *router* secara signifikan di layer-layer bawah.

**C. Manifold-Constrained Hyper-Connections (mHC)**
Ini adalah perubahan arsitektural paling mendasar di V4 pada tingkat struktur *Transformer block*.
*   **Kode Implementasi:** `DeepseekV4HyperConnection` dan `DeepseekV4DecoderLayer`.
*   **Masalah Arsitektur Klasik:** Transformer biasa menjumlahkan sisa/residu seperti ini: `Output = Input + Sublayer(Input)`. Pada arsitektur yang sangat dalam, varians sinyal ini meledak.
*   **Solusi mHC:** V4 mengganti koneksi residual tunggal dengan **beberapa aliran stream paralel** (`hc_mult = 4`). 
*   **Proyeksi Sinkhorn-Knopp:** Untuk menggabungkan aliran ini kembali, model menggunakan iterasi Sinkhorn (`hc_sinkhorn_iters`) yang memproyeksikan bobot penggabungan ke dalam **Matriks Stokastik Ganda** (*Doubly-Stochastic Matrix*). Ini menjamin bahwa aliran sinyal melewati layer sedalam apa pun tidak akan mengalami pelebaran varians atau penyusutan sinyal, membuat pelatihan model V4 berskala triliunan parameter menjadi stabil tanpa perlu trik normalisasi rumit.
*   **Grouped Output Projection:** Di V4, karena dimensi hasil atensi raksasa (bisa mencapai 65536 dimensi), memproyeksikannya kembali ke *hidden size* sangat berat. Kelas `DeepseekV4GroupedLinear` memecah *head* menjadi grup-grup independen, memproyeksikannya ke dimensi perantara (`o_lora_rank`), baru digabungkan.

---

### Kesimpulan dan Ringkasan Evolusi

| Fitur | V1 (LLaMA Base) | V2 (Efisiensi) | V3 (Penskalaan Ekstrem) | V4 (Konteks Panjang & Stabilitas) |
| :--- | :--- | :--- | :--- | :--- |
| **Arsitektur FFN** | *Dense* MLP (SwiGLU) | DeepSeekMoE (Shared + 64 Routed) | DeepSeekMoE Raksasa (256 Routed) | Hash-MoE di awal + DeepSeekMoE di atas |
| **Mekanisme Atensi** | Standar (MHA/GQA) | MLA (Multi-Head Latent Attention) | MLA + Dukungan Native Flash Attention | Hybrid: Sliding Window + CSA + HCA |
| **Manajemen KV** | KV Cache penuh | Kompresi Laten (LoRA Rank = 512) | Kompresi Laten (Sama dengan V2) | Kompresi Temporal (Rasio 4x hingga 128x) |
| **Koneksi Residual** | Standar `x + f(x)` | Standar `x + f(x)` | Standar `x + f(x)` | **mHC** (Aliran paralel dengan matriks stokastik) |
| **Routing MoE** | Tidak Ada | *Greedy Top-K* + Auxiliary Loss | Bias Correction (Tanpa Auxiliary Loss) | *Lookup Hash Deterministic* + Top-K |

Secara akademis, Anda dapat menyimpulkan bahwa DeepSeek berevolusi dari sekadar mengimitasi model *open-source* terbaik di zamannya (V1), lalu menemukan cara mengkompresi memori (V2), menyempurnakan pelatihan skala besar (V3), hingga akhirnya mendefinisikan ulang topologi dasar Transformer untuk memproses konteks tanpa batas dengan mengubah konsep atensi temporal dan koneksi residual itu sendiri (V4).