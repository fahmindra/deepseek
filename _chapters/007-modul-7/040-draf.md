---
slug: modul-7-4
title: modul-7-4
---
# 3.4 Implementasi MoE

---

Di Modul 3.4 Fundamental, kita telah membahas teori *Mixture of Experts* (MoE): sebuah strategi untuk menskalakan model hingga triliunan parameter namun menjaga biaya inferensi tetap rendah dengan hanya mengaktifkan sebagian kecil "ahli" (experts).

Namun, bagaimana bentuk "ahli" ini dalam kode? Apakah mereka model-model terpisah?

Ternyata tidak. Dalam implementasi modern (seperti Mixtral 8x7B atau Qwen-MoE), "Experts" hanyalah **lapisan-lapisan Linear (FFN) yang disusun secara paralel** di dalam satu file model besar. Tantangan utama bagi seorang *engineer* bukanlah pada matematikanya, melainkan pada **manajemen memori**.

Meskipun hanya 2 ahli yang aktif (cepat), ke-8 ahli harus tetap dimuat di VRAM (berat). Di subtopik ini, kita akan melihat bagaimana cara memuat raksasa ini ke dalam GPU standar menggunakan teknik kuantisasi yang telah kita pelajari, dan membedah struktur kodenya.

---

## 3.4.1 Konfigurasi Arsitektur MoE (Hugging Face Config)

Dalam *library* `transformers`, MoE tidak memerlukan kelas model yang "ajaib". Ia ditangani melalui konfigurasi standar. Mari kita lihat parameter kunci yang mendefinisikan MoE.

Saat Anda melihat file `config.json` dari model MoE (misal: Mixtral), Anda akan menemukan parameter berikut yang membedakannya dari model *dense*:

1.  **`num_local_experts` (misal: 8):** Ini mendefinisikan berapa banyak "ahli" FFN yang tersedia di setiap lapisan.
2.  **`num_experts_per_tok` (misal: 2):** Ini adalah aturan untuk **Router**. Ini menentukan berapa banyak ahli yang *boleh* dipilih untuk menangani satu token (Top-2 routing).
3.  **`router_aux_loss_coef`:** Ini adalah parameter pelatihan (Load Balancing Loss) yang memastikan Router tidak pilih kasih (bias) ke satu ahli saja.

Memahami konfigurasi ini penting karena mengubah `num_experts_per_tok` secara langsung mempengaruhi *trade-off* antara kecepatan (makin sedikit makin cepat) dan akurasi (makin banyak makin pintar).

---

## 3.4.2 Tantangan Implementasi: VRAM Loading

Implementasi MoE menghadapi tantangan fisik: **Ukuran File**.

*   Model **Llama-2-7B** (Dense): Ukuran file ~14GB.
*   Model **Mixtral 8x7B** (MoE): Meskipun "aktifnya" setara model 12B, ukuran file totalnya adalah **~47GB** (karena ada 8 set bobot ahli yang disimpan).

Jika kita mencoba memuatnya secara naif (`AutoModel.from_pretrained`), kita membutuhkan GPU A6000 (48GB) atau A100 (80GB).

**Solusi Implementasi:** Kita **wajib** menggunakan teknik dari Subtopik 2.3.2: **Kuantisasi 4-bit (QLoRA/BitsAndBytes)**. Dengan 4-bit, ukuran 47GB menyusut menjadi sekitar **24-26GB**. Ini memungkinkan Mixtral dijalankan pada dua GPU consumer (2x RTX 3090/4090) atau bahkan di Google Colab Pro+ (A100).

Karena kita tidak bisa melatih Mixtral di Google Colab gratis, kita akan menggunakan model MoE yang lebih kecil dan efisien namun memiliki struktur identik: **Qwen1.5-MoE-A2.7B**.

Model ini memiliki total 14B parameter, tapi hanya 2.7B yang aktif. Ini **bisa** dimuat di Colab gratis (T4 GPU) menggunakan 4-bit loading.

Kode ini akan "membuka kap mesin" dan memperlihatkan kepada Anda di mana letak *Router* dan *Experts*-nya.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# Kita gunakan model Qwen MoE yang efisien (Total 14B, Aktif 2.7B)
# Ini muat di Colab T4 jika pakai 4-bit!
model_id = "Qwen/Qwen1.5-MoE-A2.7B"


# 1. Konfigurasi 4-bit (Wajib untuk MoE di GPU kecil)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)


print(f"Memuat model MoE: {model_id}...")
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    print("Model MoE berhasil dimuat!")
except Exception as e:
    print(f"Gagal memuat: {e}")


# 2. Membedah Struktur (Inspeksi Layer)
# Kita akan melihat isi dari satu blok Transformer untuk mencari Expert
print("\n--- Struktur Blok MoE (Layer 0) ---")


# Mengakses layer pertama
# (Nama layer bisa berbeda tiap model, di Qwen/Mistral biasanya 'model.layers')
layer_0 = model.model.layers[0]
print(layer_0)
```

**Output yang dihasilkan:**

```text
--- Struktur Blok MoE (Layer 0) ---
Qwen2MoeDecoderLayer(
  (self_attn): Qwen2MoeSdpaAttention(
    (q_proj): Linear4bit(in_features=2048, out_features=2048, bias=True)
    (k_proj): Linear4bit(in_features=2048, out_features=2048, bias=True)
    (v_proj): Linear4bit(in_features=2048, out_features=2048, bias=True)
    (o_proj): Linear4bit(in_features=2048, out_features=2048, bias=False)
    (rotary_emb): Qwen2MoeRotaryEmbedding()
  )
  (mlp): Qwen2MoeSparseMoeBlock(
    (gate): Linear4bit(in_features=2048, out_features=60, bias=False)
    (experts): ModuleList(
      (0-59): 60 x Qwen2MoeMLP(
        (gate_proj): Linear4bit(in_features=2048, out_features=1408, bias=False)
        (up_proj): Linear4bit(in_features=2048, out_features=1408, bias=False)
        (down_proj): Linear4bit(in_features=1408, out_features=2048, bias=False)
        (act_fn): SiLUActivation()
      )
    )
    (shared_expert): Qwen2MoeMLP(
      (gate_proj): Linear4bit(in_features=2048, out_features=5632, bias=False)
      (up_proj): Linear4bit(in_features=2048, out_features=5632, bias=False)
      (down_proj): Linear4bit(in_features=5632, out_features=2048, bias=False)
      (act_fn): SiLUActivation()
    )
    (shared_expert_gate): Linear4bit(in_features=2048, out_features=1, bias=False)
  )
  (input_layernorm): Qwen2MoeRMSNorm((2048,), eps=1e-06)
  (post_attention_layernorm): Qwen2MoeRMSNorm((2048,), eps=1e-06)
)
```

**Penjelasan:**

Log output tersebut mengonfirmasi bahwa kita berhasil memuat model MoE raksasa dalam format 4-bit. Mari kita terjemahkan bagian *layer* yang tidak terlihat di atas (bagian MLP) ke dalam konsep yang kita pelajari:

### 1. Sang Manajer Triase (gate)

```python
(gate): Linear4bit(in_features=2048, out_features=60, bias=False)
```

Inilah **Router**. Perhatikan `out_features=60`. Angka **60** ini bukan kebetulan. Ini berarti Router bertugas memberikan skor probabilitas kepada **60 Ahli** yang tersedia. Ia akan memilih Top-K (biasanya 2 atau 4) dari 60 opsi ini untuk setiap token.

### 2. Kabinet Ahli (experts)

```python
(experts): ModuleList(
      (0-59): 60 x Qwen2MoeMLP(
        (gate_proj): Linear4bit(in_features=2048, out_features=1408, bias=False)
        (up_proj): Linear4bit(in_features=2048, out_features=1408, bias=False)
        (down_proj): Linear4bit(in_features=1408, out_features=2048, bias=False)
        (act_fn): SiLUActivation()
      )
    )
```

Inilah bukti fisik dari konsep "Mixture of Experts". Alih-alih satu blok FFN raksasa, kita memiliki sebuah `ModuleList` yang berisi 60 blok MLP independen (bernomor 0 hingga 59). Perhatikan bahwa setiap expert juga menggunakan `Linear4bit`, membuktikan bahwa teknik QLoRA kita bekerja hingga ke bagian terdalam model.

### 3. Fitur Unik Qwen (shared_expert)

```python
(shared_expert): Qwen2MoeMLP(
      (gate_proj): Linear4bit(in_features=2048, out_features=5632, bias=False)
      (up_proj): Linear4bit(in_features=2048, out_features=5632, bias=False)
      (down_proj): Linear4bit(in_features=5632, out_features=2048, bias=False)
      (act_fn): SiLUActivation()
    )
    (shared_expert_gate): Linear4bit(in_features=2048, out_features=1, bias=False)
  )
```

Ini adalah inovasi spesifik Qwen. Selain ahli-ahli spesialis (0-59) yang dipilih-pilih, ada satu **"Ahli Umum" (Shared Expert)** yang *selalu* aktif untuk setiap token. Ini seperti memiliki "Dokter Umum" yang selalu memeriksa pasien, ditambah dengan "Dokter Spesialis" yang dipanggil oleh Router sesuai kebutuhan. Ini menjaga stabilitas output model.

> **Refleksi Singkat**
>
> Melalui kode di atas, kita melihat bahwa model MoE seperti Qwen memiliki **60 ahli** (plus shared expert) yang tersimpan di memori. Bayangkan Anda adalah AI Infrastructure Engineer. Meskipun komputasi (FLOPs) model ini ringan karena hanya mengaktifkan sedikit ahli per token, **seluruh 60 ahli tersebut harus tetap "siap sedia"** di VRAM GPU agar bisa dipanggil kapan saja oleh Router.
>
> Apa implikasi arsitektur ini terhadap strategi pemilihan hardware Anda? Apakah Anda lebih memprioritaskan GPU dengan Clock Speed (kecepatan hitung) tertinggi, atau GPU dengan VRAM Capacity (kapasitas memori) terbesar? Mengapa?