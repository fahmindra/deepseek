---
slug: part-3
layout: part
---

Selamat datang di kelas Arsitektur *Large Language Models*. Pada sesi kali ini, kita akan melakukan bedah tuntas dan implementasi dari nol (*from scratch*) model **DeepSeek-V2**. 

Sebagai peneliti AI, kita tidak boleh hanya bergantung pada abstraksi tingkat tinggi dari library eksternal. Kita harus memahami aliran tensor, operasi matematis, dan alasan di balik setiap keputusan desain arsitektur. DeepSeek-V2 memperkenalkan dua inovasi masif: **Multi-head Latent Attention (MLA)** untuk kompresi *Key-Value Cache* dan **DeepSeekMoE** untuk efisiensi komputasi *Feed-Forward Network* (FFN).

Mari kita mulai menulis kode menggunakan PyTorch murni. Pastikan Anda memperhatikan setiap penjelasan matematis dan anotasi dimensi (shape) tensor.

```python
import math
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
```

---

### Bagian 1: Konfigurasi Model (The Blueprint)

Sebelum membangun model, kita membutuhkan cetak biru (*blueprint*). Dalam DeepSeek-V2, ada beberapa *hyperparameter* unik yang mengendalikan kompresi laten dan mekanisme pakar (*experts*).

*   `q_lora_rank` ($d_c'$): Dimensi kompresi untuk *Query*.
*   `kv_lora_rank` ($d_c$): Dimensi kompresi laten gabungan untuk *Key* dan *Value*.
*   `qk_nope_head_dim`: Dimensi *head* untuk *Query/Key* yang **tidak** menggunakan *Positional Embedding* (No-PE).
*   `qk_rope_head_dim` ($d_h^R$): Dimensi *head* untuk *Query/Key* yang menggunakan *Positional Embedding* secara terpisah (Decoupled RoPE).
*   `v_head_dim` ($d_h^V$): Dimensi *head* untuk *Value*.

```python
class DeepseekV2Config:
    def __init__(
        self,
        vocab_size: int = 100000, # Mengikuti paper (100K token BBPE)
        hidden_size: int = 5120,  # Dimensi representasi internal
        intermediate_size: int = 12288,
        num_hidden_layers: int = 60,
        num_attention_heads: int = 128,
        num_key_value_heads: int = 128,
        hidden_act: str = "silu",
        max_position_embeddings: int = 4096,
        rms_norm_eps: float = 1e-6,
        
        # Hyperparameters MLA (Multi-head Latent Attention)
        q_lora_rank: Optional[int] = 1536,
        kv_lora_rank: int = 512,
        qk_nope_head_dim: int = 128,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 128,
        
        # Hyperparameters DeepSeekMoE
        first_k_dense_replace: int = 1, # Layer pertama biasanya Dense (bukan MoE)
        n_routed_experts: int = 160,    # Total pakar (experts) yang di-routing
        n_shared_experts: int = 2,      # Pakar yang selalu aktif untuk menangkap pengetahuan umum
        num_experts_per_tok: int = 6,   # Top-K pakar yang dipilih per token
        moe_intermediate_size: int = 1536, # Ukuran hidden layer per pakar
        routed_scaling_factor: float = 1.0,
        topk_method: str = "greedy",
        n_group: Optional[int] = 8,     # Untuk Device-Limited Routing
        topk_group: Optional[int] = 3,  # Batas device per token
        
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        
        self.first_k_dense_replace = first_k_dense_replace
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size
        self.routed_scaling_factor = routed_scaling_factor
        self.topk_method = topk_method
        self.n_group = n_group
        self.topk_group = topk_group
        
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
```

---

### Bagian 2: Komponen Fundamental (Norm & Positional Embedding)

**1. RMSNorm:** 
Root Mean Square Normalization mengabaikan *mean centering* pada LayerNorm klasik karena komputasi *variance* (berpusat di nol) sudah cukup untuk menstabilkan gradien dan jauh lebih efisien secara komputasi. 

**2. Decoupled Rotary Position Embedding (RoPE):**
Pada arsitektur standar, RoPE diaplikasikan pada seluruh representasi *Key* dan *Query*. Namun, DeepSeek-V2 menekan ukuran *KV Cache* menggunakan kompresi linear leten ($c_t^{KV}$). Karena sifat perkalian matriks kompresi ini, jika kita mengaplikasikan RoPE pada tahap laten, sifat pergeseran posisi (positional sensitivity) dari RoPE akan terdistorsi dan matriks dekompresi tidak dapat diserap (absorbed) saat *inference*. Solusinya (Paper Bagian 2.1.3): kita **memisahkan (decouple)** representasi yang membawa informasi posisi (sebesar `qk_rope_head_dim`) dan menambahkannya kembali setelah proyeksi laten.

```python
class DeepseekV2RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        # Mengubah ke float32 untuk stabilitas numerik saat menghitung varians
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class DeepseekV2RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 4096, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Menghitung frekuensi invers: 1 / (base ^ (2i / d))
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, num_heads, seq_len, head_dim]
        # position_ids shape: [batch_size, seq_len]
        
        # inv_freq_expanded shape: [batch_size, seq_len, dim // 2]
        inv_freq_expanded = self.inv_freq[None, None, :].expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, :, None].float()
        
        # freqs shape: [batch_size, seq_len, dim // 2]
        freqs = (position_ids_expanded @ self.inv_freq[None, :])
        
        # Membuat representasi kompleks (cos + i*sin)
        # freqs_cis shape: [batch_size, seq_len, dim // 2]
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # xq shape: [batch_size, seq_len, num_heads, head_dim]
    # Konversi ke representasi kompleks dengan memisahkan dimensi terakhir menjadi 2
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # freqs_cis shape awal: [batch_size, seq_len, dim // 2]
    # Ubah menjadi [batch_size, 1, seq_len, dim // 2] untuk penyiaran (broadcasting) ke head
    freqs_cis = freqs_cis.unsqueeze(1).transpose(1, 2).to(xq_.device)

    # Perkalian bilangan kompleks (aplikasi rotasi) lalu ubah kembali ke representasi real
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3).type_as(xq)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3).type_as(xk)
    return xq_out, xk_out
```

---

### Bagian 3: Multi-Head Latent Attention (MLA) - Inovasi Inti 1

Bagian ini adalah kunci untuk mengurangi penggunaan VRAM pada tahap *inference*. Daripada menyimpan matriks *Key* ($K$) dan *Value* ($V$) penuh (seperti pada MHA standar), kita menyimpan satu representasi terkompresi $c_t^{KV}$ (berukuran $d_c$) dan satu vektor *Decoupled Key* ($k_t^R$) yang membawa informasi posisi.

Perhatikan referensi rumus matematis (sesuai *Appendix C* paper) dalam anotasi kode di bawah.

```python
class DeepseekV2Attention(nn.Module):
    def __init__(self, config: DeepseekV2Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        
        # Scaling dot-product attention
        self.scaling = self.qk_head_dim ** -0.5

        # ---------------------------------------------------------------------
        # PROYEKSI QUERY
        # ---------------------------------------------------------------------
        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.qk_head_dim, bias=False)
        else:
            # Persamaan (37): c_t^Q = W^{DQ} h_t
            self.q_a_proj = nn.Linear(self.hidden_size, self.q_lora_rank, bias=False)
            self.q_a_layernorm = DeepseekV2RMSNorm(self.q_lora_rank)
            
            # Persamaan (38) & (39): Menghasilkan q_t^C dan komponen q_t^R
            self.q_b_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False)

        # ---------------------------------------------------------------------
        # PROYEKSI KEY-VALUE GABUNGAN (LATENT COMPRESSION)
        # ---------------------------------------------------------------------
        # Persamaan (41) dan (43): c_t^{KV} = W^{DKV} h_t  DAN  k_t^R = RoPE(...)
        # Kita proyeksikan hidden state langsung menjadi laten KV dan decoupled Key RoPE.
        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
        )
        self.kv_a_layernorm = DeepseekV2RMSNorm(self.kv_lora_rank)
        
        # Persamaan (42) & (45): Dekompresi dari c_t^{KV} menjadi k_t^C dan v_t^C
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )

        # Proyeksi Output (Persamaan 47)
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_length = hidden_states.shape[:-1]
        
        # 1. Hitung Query
        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q_c = self.q_a_proj(hidden_states)
            q_c = self.q_a_layernorm(q_c)
            q = self.q_b_proj(q_c)
            
        # Bentuk ulang Q: [batch, seq_len, num_heads, qk_head_dim]
        q = q.view(batch_size, seq_length, self.num_heads, self.qk_head_dim)
        
        # Pecah Query menjadi bagian tanpa RoPE (nope) dan dengan RoPE (pe)
        # Persamaan (40): q_{t,i} = [q_{t,i}^C ; q_{t,i}^R]
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # 2. Hitung Kompresi KV dan Decoupled Key
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        # Pisahkan latent vector (k_nope) dan decoupled positional key (k_pe)
        c_kv, k_pe = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        
        c_kv = self.kv_a_layernorm(c_kv)
        
        # Dekompresi laten menjadi Key (tanpa RoPE) dan Value
        k_v_nope = self.kv_b_proj(c_kv).view(batch_size, seq_length, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, value_states = torch.split(k_v_nope, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # Bentuk ulang k_pe untuk persiapan RoPE: [batch, 1, seq_len, qk_rope_head_dim]
        # k_pe digunakan secara global (Shared Key) di seluruh head (Multi-Query style)
        k_pe = k_pe.view(batch_size, seq_length, 1, self.qk_rope_head_dim)

        # 3. Aplikasikan RoPE pada q_pe dan k_pe
        # Persamaan (39) dan (43)
        if position_embeddings is not None:
            q_pe, k_pe = apply_rotary_emb(q_pe, k_pe, position_embeddings)

        # Expand k_pe ke semua head agar bisa digabungkan dengan k_nope
        k_pe = k_pe.expand(-1, -1, self.num_heads, -1)
        
        # 4. Gabungkan (Concatenate) kembali komponen No-PE dan PE
        # Persamaan (44): k_{t,i} = [k_{t,i}^C ; k_t^R]
        query_states = torch.cat((q_nope, q_pe), dim=-1).transpose(1, 2)  # [batch, num_heads, seq_len, qk_head_dim]
        key_states = torch.cat((k_nope, k_pe), dim=-1).transpose(1, 2)    # [batch, num_heads, seq_len, qk_head_dim]
        value_states = value_states.transpose(1, 2)                       # [batch, num_heads, seq_len, v_head_dim]

        # Catatan: Dalam praktiknya saat inference (KV Cache), kita HANYA menyimpan 'c_kv' dan 'k_pe' 
        # (sejumlah kv_lora_rank + qk_rope_head_dim dimensi). Ini menghemat memori sangat signifikan!
        
        # 5. Scaled Dot-Product Attention (Persamaan 46)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states) # [batch, num_heads, seq_len, v_head_dim]
        
        # Gabungkan dimensi head (Persamaan 47)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, -1)
        attn_output = self.o_proj(attn_output)
        
        return attn_output
```

---

### Bagian 4: DeepSeekMoE (Mixture of Experts) - Inovasi Inti 2

DeepSeekMoE memecah *experts* menjadi unit-unit yang jauh lebih kecil (*fine-grained*) dan mengisolasi sebagian dari mereka sebagai **Shared Experts**. Pakar bersama (shared) akan diaktifkan untuk setiap token, berfungsi untuk merepresentasikan pengetahuan umum. Sementara itu, *routed experts* fokus pada spesialisasi tertentu.

```python
# Act fungsi standard SwiGLU
def swiglu(x):
    gate, up = x.chunk(2, dim=-1)
    return F.silu(gate) * up

class DeepseekV2MLP(nn.Module):
    """Dense MLP, digunakan untuk layer awal dan Shared Experts."""
    def __init__(self, config: DeepseekV2Config, intermediate_size: int = None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size if intermediate_size is not None else config.intermediate_size
        
        # Proyeksi untuk Gate (Swish) dan Up (Kombinasi linear)
        self.gate_up_proj = nn.Linear(self.hidden_size, 2 * self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len, hidden_size]
        x = self.gate_up_proj(x)
        x = swiglu(x)
        return self.down_proj(x)


class DeepseekV2Experts(nn.Module):
    """Kumpulan semua pakar (Routed Experts)."""
    def __init__(self, config: DeepseekV2Config):
        super().__init__()
        self.num_experts = config.n_routed_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        
        # Kita menyimpan seluruh beban (weights) dari semua pakar dalam satu tensor 3D 
        # untuk memaksimalkan paralelisme saat komputasi.
        self.gate_up_proj = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_dim, 2 * self.intermediate_dim)
        )
        self.down_proj = nn.Parameter(
            torch.empty(self.num_experts, self.intermediate_dim, self.hidden_dim)
        )
        
        # Inisialisasi parameter (simplifikasi menggunakan normalisasi standar)
        nn.init.normal_(self.gate_up_proj, std=config.hidden_size ** -0.5)
        nn.init.normal_(self.down_proj, std=self.intermediate_dim ** -0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        # hidden_states shape: [total_tokens, hidden_dim]
        # top_k_index shape: [total_tokens, top_k]
        # top_k_weights shape: [total_tokens, top_k]
        
        final_hidden_states = torch.zeros_like(hidden_states)
        
        # expert_mask: [total_tokens, top_k, num_experts] (one-hot vector)
        expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
        # Ubah mask agar iterasi berpusat pada expert: [num_experts, top_k, total_tokens]
        expert_mask = expert_mask.permute(2, 1, 0)
        
        # Cari pakar yang setidaknya memproses 1 token
        expert_hit = torch.nonzero(expert_mask.sum(dim=(-1, -2)) > 0).squeeze(-1)

        # Iterasi pada pakar yang aktif (sparse computation)
        for expert_idx in expert_hit:
            # top_k_pos: urutan (ranking ke berapa pakar ini terpilih)
            # token_idx: indeks riil token di dalam batch yang diproses
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            
            # Ambil hidden states untuk token yang menuju pakar ini
            current_state = hidden_states[token_idx] # [num_tokens_for_expert, hidden_dim]
            
            # Komputasi SwiGLU (Persamaan standar MLP) per pakar
            gate_up = current_state @ self.gate_up_proj[expert_idx]
            current_hidden_states = swiglu(gate_up)
            current_hidden_states = current_hidden_states @ self.down_proj[expert_idx]
            
            # Kalikan dengan weight routing dari Gate
            weight = top_k_weights[token_idx, top_k_pos].unsqueeze(-1)
            current_hidden_states = current_hidden_states * weight
            
            # Agregasi hasil dari pakar kembali ke tensor output utama.
            # Menggunakan index_add_ untuk menambahkan nilai pada index spesifik secara in-place.
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states


class DeepseekV2Moe(nn.Module):
    """Lapisan Mixture of Experts Utama yang mengatur Gate (Router) dan Shared Experts."""
    def __init__(self, config: DeepseekV2Config):
        super().__init__()
        self.config = config
        self.experts = DeepseekV2Experts(config)
        
        # Linear layer untuk menghitung logit/probabilitas (Routing)
        self.gate = nn.Linear(config.hidden_size, config.n_routed_experts, bias=False)
        
        # Shared Experts (Pengetahuan Dasar yang tidak pernah di-drop)
        if config.n_shared_experts is not None and config.n_shared_experts > 0:
            shared_intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = DeepseekV2MLP(config=config, intermediate_size=shared_intermediate_size)
            
        self.routed_scaling_factor = config.routed_scaling_factor
        self.top_k = config.num_experts_per_tok
        self.topk_method = config.topk_method # "greedy" atau "group_limited_greedy"
        self.n_group = config.n_group
        self.topk_group = config.topk_group

    def route_tokens_to_experts(self, router_logits: torch.Tensor):
        # router_logits shape: [total_tokens, n_routed_experts]
        router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        
        if self.topk_method == "greedy":
            # Pendekatan konvensional: Ambil top-K pakar dengan probabilitas tertinggi
            topk_weights, topk_indices = torch.topk(router_probs, k=self.top_k, dim=-1)
            
        elif self.topk_method == "group_limited_greedy":
            # Device-Limited Routing (Bagian 2.2.2 Paper): 
            # Membatasi komunikasi agar token tidak menyebar ke terlalu banyak GPU (group).
            total_tokens = router_probs.shape[0]
            experts_per_group = self.config.n_routed_experts // self.n_group
            
            # Hitung skor tertinggi di dalam setiap grup
            group_scores = router_probs.view(total_tokens, self.n_group, experts_per_group).max(dim=-1).values
            # Pilih top-K group
            _, group_indices = torch.topk(group_scores, k=self.topk_group, dim=-1)
            
            # Buat mask untuk membatalkan probabilitas dari grup yang tidak terpilih
            group_mask = torch.zeros_like(group_scores).scatter_(1, group_indices, 1)
            score_mask = group_mask.unsqueeze(-1).expand(total_tokens, self.n_group, experts_per_group).reshape(total_tokens, -1)
            
            # Terapkan mask, paksa probabilitas grup lain menjadi 0.0
            tmp_scores = router_probs.masked_fill(~score_mask.bool(), 0.0)
            topk_weights, topk_indices = torch.topk(tmp_scores, k=self.top_k, dim=-1)
            
        # Terapkan penskalaan (scaling factor) agar nilai output pakar seimbang
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        # Flatten sequence: [batch_size * seq_len, hidden_dim]
        hidden_states_flat = hidden_states.view(-1, orig_shape[-1])
        
        # 1. Routing Logits
        router_logits = self.gate(hidden_states_flat)
        topk_indices, topk_weights = self.route_tokens_to_experts(router_logits)
        
        # 2. Komputasi Routed Experts
        routed_output = self.experts(hidden_states_flat, topk_indices, topk_weights).view(orig_shape)
        
        # 3. Komputasi Shared Experts
        shared_output = self.shared_experts(hidden_states)
        
        # Kombinasikan (Persamaan 20: h_t' = u_t + Shared_Experts(u_t) + Routed_Experts(u_t))
        # (Catatan: residual connection u_t biasanya ditambahkan di level Transformer Block, bukan di dalam MoE)
        return routed_output + shared_output
```

---

### Bagian 5: Integrasi Blok Transformer (Decoder Layer)

Di bagian ini kita menyusun lapisan *Attention* dan *FFN*. Sesuai desain DeepSeek-V2, lapisan awal bisa menggunakan *Dense MLP* konvensional sebelum beralih ke *MoE Layer*. Ini dikontrol oleh hyperparameter `first_k_dense_replace`.

```python
class DeepseekV2DecoderLayer(nn.Module):
    def __init__(self, config: DeepseekV2Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        
        self.input_layernorm = DeepseekV2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = DeepseekV2Attention(config)
        self.post_attention_layernorm = DeepseekV2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Logika penggantian: Gunakan Dense MLP untuk k lapisan pertama, MoE untuk sisanya
        if layer_idx < config.first_k_dense_replace:
            self.mlp = DeepseekV2MLP(config)
        else:
            self.mlp = DeepseekV2Moe(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Blok 1: Attention dengan Pre-LN (Residual Connection)
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        # Blok 2: FFN / MoE dengan Pre-LN (Residual Connection)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states
```

---

### Bagian 6: Arsitektur Penuh (The Final Model)

Tahap terakhir, kita membungkus keseluruhan arsitektur dari lapisan sematan kata (*embedding*) hingga kepala prediksi teks (*Language Modeling Head*).

```python
def create_causal_mask(seq_len: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Membuat mask kausal (segitiga bawah) agar token tidak melihat masa depan."""
    mask = torch.full((seq_len, seq_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    # Shape: [1, 1, seq_len, seq_len] untuk penyiaran (broadcasting) di attention head
    return mask[None, None, :, :]

class DeepseekV2Model(nn.Module):
    """Arsitektur dasar tanpa layer output akhir."""
    def __init__(self, config: DeepseekV2Config):
        super().__init__()
        self.config = config
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        
        # Membuat tumpukan layer Decoder
        self.layers = nn.ModuleList(
            [DeepseekV2DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        
        self.norm = DeepseekV2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Instansiasi RoPE yang akan diteruskan ke dalam Attention Layer
        self.rotary_emb = DeepseekV2RotaryEmbedding(
            dim=config.qk_rope_head_dim, 
            max_position_embeddings=config.max_position_embeddings
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        
        # 1. Ambil Embedding Token
        hidden_states = self.embed_tokens(input_ids) # [batch_size, seq_len, hidden_size]
        
        # 2. Persiapkan Positional Ids & Masking Kausal
        device = input_ids.device
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0) # [1, seq_len]
        
        # Hitung Frekuensi Posisi Laten (RoPE)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        
        # Buat Causal Mask [1, 1, seq_len, seq_len]
        attention_mask = create_causal_mask(seq_len, hidden_states.dtype, device)
        
        # 3. Alirkan data melewati tumpukan Decoder Layers
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings
            )
            
        # 4. Normalisasi Akhir
        hidden_states = self.norm(hidden_states)
        return hidden_states


class DeepseekV2ForCausalLM(nn.Module):
    """Model Bahasa Lengkap untuk inferensi teks / generasi (*Causal Language Modeling*)."""
    def __init__(self, config: DeepseekV2Config):
        super().__init__()
        self.config = config
        self.model = DeepseekV2Model(config)
        
        # Kepala Prediksi Token (LM Head)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        # input_ids shape: [batch_size, seq_len]
        
        # Dapatkan representasi semantik dari lapisan terakhir
        hidden_states = self.model(input_ids)
        
        # Hitung logit prediksi token selanjutnya
        # logits shape: [batch_size, seq_len, vocab_size]
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # Shift data ke kiri sebesar 1 agar model memprediksi t+1 pada langkah t
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Hitung Cross-Entropy Loss
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=self.config.pad_token_id
            )
            
        return loss, logits
```

---

### Rangkuman Akademis Penutup
Mahasiswa sekalian, implementasi PyTorch murni (*from scratch*) di atas menangkap dua terobosan paling teknis dari model DeepSeek-V2:
1.  **Multi-head Latent Attention** (Blok `DeepseekV2Attention`): Menghindari memori *O(N)* dari KV-Cache dengan memproyeksikan status laten ke dalam *bottleneck* ($c_t^{KV}$), dan menggabungkannya dengan fitur *Decoupled RoPE* ($k_t^R$) untuk menjaga informasi posisi.
2.  **DeepSeekMoE** (Blok `DeepseekV2Moe` dan `DeepseekV2Experts`): Meningkatkan efektivitas transfer gradien dengan memecah *experts* menjadi lebih banyak unit matriks berukuran kecil, menambalnya dalam komputasi linier tingkat rendah, dan secara elegan menerapkan isolasi parameter (melalui *Shared Experts*).

Dengan mengikuti kode ini, Anda kini memiliki basis yang kokoh untuk membuat varian turunan atau mengeksplorasi efisiensi lanjutan pada arsitektur LLM modern. Terus pelajari hubungan matematis yang tertanam di balik susunan kode! Silakan diskusikan bila ada bagian operasi tensor yang kurang jelas.