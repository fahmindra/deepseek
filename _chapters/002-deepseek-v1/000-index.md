---
slug: part-2
layout: part
---

Selamat pagi, mahasiswa sekalian. Selamat datang di mata kuliah Arsitektur *Deep Learning*. Pada sesi kali ini, kita akan membedah secara mendalam salah satu arsitektur *Large Language Model* (LLM) *open-source* yang sangat fundamental dan berkinerja tinggi, yaitu **DeepSeek LLM (v1)**.

Sebagai calon *AI Engineer* atau Peneliti Ilmu Komputer, sangat penting bagi Anda untuk tidak hanya menjadi "pengguna" *library* (seperti `transformers` dari HuggingFace), tetapi juga memahami anatomi matematika dan implementasi kode murni dari arsitektur tersebut dari awal (*from scratch*).

Berdasarkan publikasi ilmiah *"DeepSeek LLM: Scaling Open-Source Language Models with Longtermism"*, arsitektur DeepSeek sangat mengadopsi pondasi dasar dari LLaMA, namun dengan beberapa modifikasi struktural makro yang sangat spesifik. Perbedaan utamanya adalah:
1. **Ekspansi Kedalaman (Depth) pada Varian 67B:** Alih-alih melebarkan dimensi *hidden layer* secara masif, DeepSeek 67B menambah jumlah lapisan (*layers*) hingga 95 lapis. Hal ini dilakukan untuk menjaga konsistensi parameter sambil memfasilitasi optimasi komputasi *pipeline partitioning*.
2. **Grouped-Query Attention (GQA):** Varian 67B menggunakan GQA untuk menghemat penggunaan memori (VRAM) saat *inference*, sementara varian 7B tetap menggunakan *Multi-Head Attention* (MHA) standar.
3. **Penggunaan SwiGLU dengan Dimensi Spesifik:** Dimensi *intermediate* pada *Feed-Forward Network* (FFN) secara matematis diatur sebesar $\frac{8}{3} \times d_{model}$.

Mari kita mulai menulis rancang bangun arsitektur ini menggunakan PyTorch murni.

---

### 1. Konfigurasi Model (`DeepSeekConfig`)

Langkah pertama dalam mendesain sistem *deep learning* yang rapi adalah membuat kelas konfigurasi. Kelas ini berfungsi sebagai "cetak biru" yang mendefinisikan seluruh hiperparameter model. Merujuk pada **Tabel 2** di paper DeepSeek LLM, kita akan membuat fungsi pembantu (*helper methods*) untuk membangun versi 7B dan 67B.

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class DeepSeekConfig:
    """
    Kelas konfigurasi untuk mendefinisikan hiperparameter arsitektur DeepSeek LLM.
    """
    vocab_size: int = 102400          # Ukuran vocabulary (berdasarkan paper Bagian 2.1)
    hidden_size: int = 4096           # Dimensi d_model
    intermediate_size: int = 10922    # Dimensi lapisan FFN (sekitar 8/3 * hidden_size)
    num_hidden_layers: int = 30       # Jumlah blok transformer (depth)
    num_attention_heads: int = 32     # Jumlah head untuk Query
    num_key_value_heads: int = 32     # Jumlah head untuk Key dan Value (jika sama dengan attention_heads, berarti MHA)
    hidden_act: str = "silu"          # Fungsi aktivasi (Swish/SiLU untuk SwiGLU)
    max_position_embeddings: int = 4096 # Konteks maksimum sequence length
    rms_norm_eps: float = 1e-6        # Nilai epsilon untuk mencegah pembagian dengan nol pada RMSNorm
    rope_theta: float = 10000.0       # Base frekuensi untuk Rotary Position Embedding

    @classmethod
    def config_7b(cls):
        """Membuat konfigurasi untuk DeepSeek LLM 7B (Multi-Head Attention)."""
        d_model = 4096
        # Menghitung (8/3) * d_model sesuai publikasi paper
        inter_size = int((8/3) * d_model) # 10922
        return cls(
            hidden_size=d_model,
            intermediate_size=inter_size,
            num_hidden_layers=30,
            num_attention_heads=32,
            num_key_value_heads=32,       # q_heads == kv_heads -> MHA
        )

    @classmethod
    def config_67b(cls):
        """Membuat konfigurasi untuk DeepSeek LLM 67B (Grouped-Query Attention)."""
        d_model = 8192
        inter_size = int((8/3) * d_model) # 21845
        return cls(
            hidden_size=d_model,
            intermediate_size=inter_size,
            num_hidden_layers=95,         # Jauh lebih dalam dibanding LLaMA 70B
            num_attention_heads=64,
            num_key_value_heads=8,        # q_heads > kv_heads -> GQA (8 grup)
        )
```

---

### 2. Implementasi Root Mean Square Normalization (`DeepSeekRMSNorm`)

Arsitektur transformer tradisional (seperti GPT-2 atau BERT) menggunakan *Layer Normalization* yang menghitung rata-rata (*mean*) dan varians. DeepSeek menggunakan **Pre-Norm dengan RMSNorm** untuk menstabilkan gradien. RMSNorm terbukti secara matematis bahwa memusatkan mean (*mean-centering*) tidak terlalu krusial untuk kesuksesan LLM, sehingga dengan hanya membagi menggunakan *Root Mean Square*, kita bisa menghemat komputasi.

Formulasi Matematis:
$y = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}} \odot \gamma$

```python
class DeepSeekRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        # Parameter bobot yang dapat dilatih (gamma)
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        hidden_states shape: [batch_size, seq_len, hidden_size]
        """
        # Casting ke float32 untuk stabilitas numerik saat menghitung varians
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        
        # Menghitung mean dari nilai kuadrat (RMS)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        
        # Normalisasi dan kembalikan ke tipe data asal (misal bfloat16)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
```

---

### 3. Rotary Position Embedding (`DeepSeekRotaryEmbedding`)

Teks diproses secara paralel oleh LLM, sehingga model tidak mengerti urutan kata tanpa *Positional Encoding*. DeepSeek menggunakan **Rotary Position Embedding (RoPE)**. Alih-alih menambahkan vektor posisi (seperti *Absolute Positional Encoding* pada paper *Attention Is All You Need*), RoPE "memutar" representasi Query dan Key pada bidang 2D menggunakan rotasi matriks trigonometri.

```python
class DeepSeekRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 4096, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Frekuensi invers: 1 / (base ^ (2i / dim))
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int):
        """
        x shape: [batch_size, num_heads, seq_len, head_dim]
        """
        # Membuat vektor posisi: [0, 1, 2, ..., seq_len - 1]
        position_ids = torch.arange(seq_len, dtype=torch.float32, device=x.device)
        
        # Outer product antara position_ids dan inv_freq
        # shape freqs: [seq_len, dim // 2]
        freqs = torch.einsum("i,j->ij", position_ids, self.inv_freq)
        
        # Duplikasi frekuensi untuk sin dan cos agar dimensi cocok dengan head_dim
        # shape emb: [seq_len, dim]
        emb = torch.cat((freqs, freqs), dim=-1)
        
        cos = emb.cos()[None, None, :, :]  # shape: [1, 1, seq_len, dim]
        sin = emb.sin()[None, None, :, :]  # shape: [1, 1, seq_len, dim]
        
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Memutar setengah elemen pada dimensi terakhir tensor.
    Transformasi dari [x1, x2, ..., xn] menjadi [-x(n/2+1), ..., -xn, x1, ..., x(n/2)]
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """
    Mengaplikasikan RoPE pada Query dan Key.
    q, k shape: [batch_size, num_heads, seq_len, head_dim]
    """
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
```

---

### 4. SwiGLU Feed-Forward Network (`DeepSeekMLP`)

*Feed-Forward Network* (FFN) bertugas mengolah informasi spasial dari representasi kata. DeepSeek tidak menggunakan ReLU atau GELU, melainkan **SwiGLU** (kombinasi fungsi Swish/SiLU dengan Gated Linear Unit).
Sesuai paper Bagian 2.2, dimensi *intermediate* diatur sebesar $\frac{8}{3} \times d_{model}$. Berbeda dengan standar dimana dimensi intermediate biasanya diperlebar $4 \times d_{model}$.

```python
class DeepSeekMLP(nn.Module):
    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        # Gate Projection (W1 pada paper LLaMA/SwiGLU)
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        # Up Projection (W3)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        # Down Projection (W2)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        
        # Fungsi Aktivasi SiLU (x * Sigmoid(x)) identik dengan Swish tanpa parameter beta
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: [batch_size, seq_len, hidden_size]
        Formulasi SwiGLU: (SiLU(x * W1) element-wise-multiply (x * W3)) * W2
        """
        gate = self.act_fn(self.gate_proj(x)) # [batch, seq, intermediate_size]
        up = self.up_proj(x)                  # [batch, seq, intermediate_size]
        down = self.down_proj(gate * up)      # [batch, seq, hidden_size]
        return down
```

---

### 5. Attention Mechanism (`DeepSeekAttention`)

Ini adalah inti dari LLM. Kita akan mengimplementasikan logika yang dinamis. Jika `num_attention_heads == num_key_value_heads`, maka ia bertindak sebagai **Multi-Head Attention (MHA)** seperti varian 7B. Namun jika `num_key_value_heads` lebih kecil, ia secara otomatis menjadi **Grouped-Query Attention (GQA)** seperti varian 67B.

Dalam GQA, beberapa Query akan berbagi referensi *Key* dan *Value* yang sama. Oleh karena itu, *Key* dan *Value* harus diduplikasi (di-`repeat`) agar dimensinya cocok saat dilakukan perkalian matriks (*Dot Product*).

```python
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Fungsi utilitas untuk menduplikasi Key dan Value pada dimensi Head (Grouped-Query Attention).
    Berdasarkan paper, jika q_heads = 64 dan kv_heads = 8, maka n_rep = 8.
    """
    batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    
    # Expand menambah dimensi bayangan, reshape meratakan kembali dimensi head
    # shape output: [batch, num_kv_heads * n_rep, seq_len, head_dim]
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, seq_len, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)

class DeepSeekAttention(nn.Module):
    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        
        # Jumlah grup Query yang akan berbagi 1 Key/Value
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        
        # Dimensi masing-masing head
        self.head_dim = self.hidden_size // self.num_heads
        
        # Proyeksi Linear tanpa bias
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        self.rotary_emb = DeepSeekRotaryEmbedding(
            self.head_dim, 
            max_position_embeddings=config.max_position_embeddings, 
            base=config.rope_theta
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None
    ):
        """
        hidden_states shape: [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # 1. Proyeksi input menjadi Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # 2. Reshape untuk pemisahan Attention Heads
        # shape: [batch_size, seq_len, num_heads, head_dim]
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        # shape saat ini setelah transpose: [batch, num_heads, seq, head_dim]
        
        # 3. Hitung cos dan sin dari RoPE dan aplikasikan ke Q dan K
        cos, sin = self.rotary_emb(value_states, seq_len=seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # 4. GQA: Duplikasi Key dan Value (Jika MHA, num_kv_groups = 1, fungsi tidak merubah apapun)
        key_states = repeat_kv(key_states, self.num_kv_groups)
        value_states = repeat_kv(value_states, self.num_kv_groups)
        
        # 5. Scaled Dot-Product Attention: Softmax(QK^T / sqrt(d)) * V
        # Menggunakan F.scaled_dot_product_attention (FlashAttention) untuk optimasi hardware
        attn_output = F.scaled_dot_product_attention(
            query_states, 
            key_states, 
            value_states, 
            attn_mask=attention_mask, 
            is_causal=True if attention_mask is None else False
        )
        
        # 6. Mengembalikan format tensor ke bentuk semula
        attn_output = attn_output.transpose(1, 2).contiguous() # [batch, seq, num_heads, head_dim]
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size) # Flatten head
        
        # 7. Final output projection
        attn_output = self.o_proj(attn_output)
        
        return attn_output
```

---

### 6. Merangkai Jaringan: `DeepSeekDecoderLayer`, `Model`, dan `ForCausalLM`

Sekarang kita memiliki balok-balok pembangunnya (RMSNorm, RoPE, MHA/GQA, SwiGLU). Mari kita susun menjadi Blok *Transformer Decoder*, Model dasar, dan terakhir Model Bahasa generatif.

#### DeepSeekDecoderLayer
Menerapkan skema *Pre-Norm* dan *Residual Connection* (*Skip Connection*).

```python
class DeepSeekDecoderLayer(nn.Module):
    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        self.self_attn = DeepSeekAttention(config=config)
        self.mlp = DeepSeekMLP(config=config)
        
        self.input_layernorm = DeepSeekRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DeepSeekRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        # 1. Residual Block: Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)       # Pre-Norm
        hidden_states = self.self_attn(hidden_states, attention_mask=attention_mask)
        hidden_states = residual + hidden_states                  # Residual Addition
        
        # 2. Residual Block: FFN (SwiGLU)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states) # Pre-Norm
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states                     # Residual Addition
        
        return hidden_states
```

#### DeepSeekModel (Backbone Transformer)
Model dasar yang mengonversi ID Token (dari Tokenizer) ke vektor *Embedding*, meneruskannya melewati tumpukan (stack) *Decoder Layers*, dan diakhiri dengan Normalisasi akhir.

```python
class DeepSeekModel(nn.Module):
    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        self.config = config
        
        # Word Embedding Layer
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Susunan Layer Decoder (Depth model)
        self.layers = nn.ModuleList([
            DeepSeekDecoderLayer(config) for _ in range(config.num_hidden_layers)
        ])
        
        # Normalisasi akhir sebelum Output Projection
        self.norm = DeepSeekRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        input_ids shape: [batch_size, seq_len]
        """
        # Ekstraksi representasi kata
        hidden_states = self.embed_tokens(input_ids) # [batch_size, seq_len, hidden_size]
        
        # Melewati tumpukan Transformer Layers (sebanyak 30 layer untuk 7B, 95 layer untuk 67B)
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)
            
        # Terapkan Normalisasi layer terakhir
        hidden_states = self.norm(hidden_states)
        
        return hidden_states
```

#### DeepSeekForCausalLM (Model Prediksi Kata Selanjutnya)
Agar model dapat "berbicara" atau *generate text*, arsitektur *backbone* di atas harus dipasangi kepala *(head)* berwujud lapisan Linear (sering disebut LM Head). Tujuan head ini adalah memetakan `hidden_size` ke dalam ruang `vocab_size` (ukuran kamus: 102400 dimensi).

```python
class DeepSeekForCausalLM(nn.Module):
    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        self.config = config
        
        # Inisialisasi Model Inti
        self.model = DeepSeekModel(config)
        
        # Language Modeling Head (Tanpa bias)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.LongTensor, labels: torch.LongTensor = None):
        """
        Proses Causal Language Modeling. Jika `labels` diberikan, hitung CrossEntropyLoss.
        """
        # Dapatkan fitur representasi terakhir (Last Hidden State)
        # shape: [batch_size, seq_len, hidden_size]
        hidden_states = self.model(input_ids)
        
        # Proyeksikan ke ruang probabilitas vocabulary
        # shape: [batch_size, seq_len, vocab_size]
        logits = self.lm_head(hidden_states)
        
        loss = None
        # Jika fase training, kita membandingkan 'logits' dengan 'labels'
        if labels is not None:
            # Karena ini prediksi causal (next-token), prediksi digeser sejauh 1 token
            shift_logits = logits[..., :-1, :].contiguous() # Membuang prediksi pada token terakhir
            shift_labels = labels[..., 1:].contiguous()     # Membuang ground-truth pada token pertama
            
            # Meratakan (Flatten) bentuk ke format [batch * seq_len, vocab_size] untuk kalkulasi Loss
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size), 
                shift_labels.view(-1)
            )
            
        return {"loss": loss, "logits": logits}
```

---

### Penutup
Mahasiswa sekalian, apa yang baru saja kita bangun bukanlah mainan (*toy model*), melainkan basis struktural murni sekelas arsitektur produksi yang merajai papan peringkat (leaderboards) LLM.

**Kesimpulan Utama Perkuliahan Hari Ini:**
1. Arsitektur DeepSeek sangat berfokus pada kedalaman model (hingga 95 lapisan). Kedalaman model (*Depth*) terbukti efektif untuk kemampuan penalaran (*reasoning*) dibandingkan kelebaran jaringan (*Width*).
2. Mekanisme matematika RoPE memastikan bahwa pemahaman letak kata (*posisi relatif*) tetap terjaga betapapun panjang urutan konteks (hingga 4096 token untuk base pre-training mereka).
3. Pendekatan GQA menyelamatkan kita dari *bottleneck* memori. Menyimpan *Key* dan *Value Cache* untuk keseluruhan *heads* adalah mimpi buruk RAM grafis. Dengan membaginya menjadi 8 grup, kita memotong konsumsi memori *inference* hingga sebagian besarnya.