---
slug: part-4
layout: part
---
Selamat pagi, rekan-rekan mahasiswa sekalian. Selamat datang di mata kuliah *Advanced Deep Learning Architectures*. 

Hari ini, kita akan membedah dan merekonstruksi salah satu arsitektur *Large Language Model* (LLM) paling mutakhir saat ini: **DeepSeek-V3**. Berdasarkan laporan teknis (Technical Report) yang dirilis, model ini memiliki **671 Miliar parameter**, di mana hanya **37 Miliar parameter yang diaktifkan (activated)** per token. Hal ini dicapai melalui dua inovasi utama: **Multi-Head Latent Attention (MLA)** untuk efisiensi inferensi dan **DeepSeekMoE** dengan *Auxiliary-Loss-Free Load Balancing* untuk efisiensi pelatihan. Selain itu, mereka menggunakan objektif **Multi-Token Prediction (MTP)**.

Sebagai insinyur AI, kita tidak boleh hanya membaca paper. Kita harus bisa menulis kodenya dari nol (from scratch). Mari kita implementasikan arsitektur ini menggunakan **PyTorch murni**.

---

### 1. Pendahuluan & Konfigurasi (`DeepseekV3Config`)

Sebelum membangun model, kita mendefinisikan *hyperparameters*. Dalam DeepSeek-V3, terdapat beberapa parameter kunci:
*   `first_k_dense_replace=3`: Tiga layer pertama menggunakan arsitektur *Dense* (MLP standar). Sisa layernya (layer 4 ke atas) menggunakan *Mixture of Experts* (MoE). Mengapa? Karena layer awal bertugas mengekstraksi representasi semantik dasar yang berlaku universal untuk semua token.
*   Parameter MLA (`q_lora_rank`, `kv_lora_rank`): Menentukan dimensi kompresi (Low-Rank) untuk mengurangi memori *KV-Cache*.
*   Parameter MoE (`n_routed_experts=256`, `num_experts_per_tok=8`): Model memiliki 256 *expert*, namun hanya 8 *expert* teratas yang memproses suatu token tertentu, ditambah 1 *Shared Expert* yang selalu aktif.

```python
import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepseekV3Config:
    def __init__(self):
        # Dimensi Dasar
        self.vocab_size = 129280
        self.hidden_size = 7168
        self.num_hidden_layers = 61
        self.intermediate_size = 18432
        
        # Konfigurasi Multi-Head Latent Attention (MLA)
        self.num_attention_heads = 128
        self.num_key_value_heads = 128
        self.kv_lora_rank = 512       # Dimensi kompresi KV (Low-Rank)
        self.q_lora_rank = 1536       # Dimensi kompresi Query
        self.qk_rope_head_dim = 64    # Dimensi head untuk RoPE
        self.v_head_dim = 128         # Dimensi head untuk Value
        self.qk_nope_head_dim = 128   # Dimensi head untuk Query/Key tanpa RoPE
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        
        # Konfigurasi DeepSeekMoE
        self.first_k_dense_replace = 3 # 3 layer pertama adalah Dense
        self.moe_intermediate_size = 2048
        self.n_shared_experts = 1
        self.n_routed_experts = 256
        self.num_experts_per_tok = 8   # Top-K routing
        
        # Konfigurasi Auxiliary-Loss-Free Load Balancing
        self.n_group = 8               # Jumlah grup routing
        self.topk_group = 4            # Token diarahkan maksimal ke 4 grup
        self.routed_scaling_factor = 2.5
        self.norm_topk_prob = True
        
        # Konfigurasi Lainnya
        self.hidden_act = "silu"
        self.max_position_embeddings = 4096
        self.rms_norm_eps = 1e-6
        self.tie_word_embeddings = False
```

---

### 2. Komponen Dasar (RMSNorm & RoPE)

DeepSeek-V3 menggunakan *Root Mean Square Normalization* (RMSNorm) dan *Rotary Position Embedding* (RoPE). Untuk efisiensi, alih-alih melakukan transposisi tensor yang berat di memori, mereka menggunakan trik **RoPE Interleaving**.

```python
class DeepseekV3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # Upcast ke FP32 untuk presisi saat menghitung varians
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)

def apply_rotary_pos_emb_interleave(q, k, cos, sin):
    """
    Menggabungkan Rotary Position Embedding (RoPE) menggunakan metode interleave.
    Metode ini lebih efisien pada GPU (mengurangi overhead reshape) dibandingkan metode separuh rotasi biasa.
    """
    # q, k shape: [batch, heads, seq_len, head_dim]
    b, h, s, d = q.shape
    
    # Reshape untuk interleaving: membagi dimensi head_dim menjadi 2 bagian lalu di-transpose
    # Shape menjadi: [batch, heads, seq_len, head_dim]
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    # Tambahkan dimensi untuk broadcasting: [batch, 1, seq_len, head_dim]
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    # Rotasi setengah (memutar 90 derajat)
    q_half = torch.cat((-q[..., d // 2:], q[..., :d // 2]), dim=-1)
    k_half = torch.cat((-k[..., d // 2:], k[..., :d // 2]), dim=-1)

    q_embed = (q * cos) + (q_half * sin)
    k_embed = (k * cos) + (k_half * sin)
    return q_embed, k_embed

class DeepseekV3RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=4096, base=10000.0):
        super().__init__()
        # Menghitung frekuensi invers
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, position_ids):
        # x shape: [batch_size, seq_len, ...]
        # Mengalikan frekuensi invers dengan posisi
        freqs = (position_ids.float().unsqueeze(-1) @ self.inv_freq.unsqueeze(0)).transpose(1, 2)
        # Menggandakan frekuensi untuk cos dan sin (interleaving format)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos.to(x.dtype), sin.to(x.dtype)
```

---

### 3. Multi-Head Latent Attention (MLA)

**Teori:** Pada Transformer standar, saat melakukan generasi (inferensi), kita harus menyimpan tensor Key dan Value (*KV Cache*) dari token sebelumnya. Semakin panjang konteks, KV Cache akan menghabiskan VRAM GPU. MLA memecahkan ini melalui **Low-Rank Joint Compression**. Alih-alih menyimpan Key dan Value secara penuh, MLA memproyeksikan *hidden states* ke dalam dimensi laten kecil (`kv_lora_rank=512`), menyimpannya, lalu memproyeksikannya kembali (Up-Projection) saat dibutuhkan. Hal serupa dilakukan untuk Query.

Selain itu, RoPE dipisahkan (Decoupled RoPE) agar tidak mengganggu proses kompresi laten tersebut.

```python
class DeepseekV3Attention(nn.Module):
    def __init__(self, config: DeepseekV3Config):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        
        self.qk_nope_head_dim = config.qk_nope_head_dim # Dimensi QK tanpa RoPE
        self.qk_rope_head_dim = config.qk_rope_head_dim # Dimensi QK dengan RoPE
        self.qk_head_dim = config.qk_head_dim           # Total QK (128 + 64 = 192)
        self.v_head_dim = config.v_head_dim             # Dimensi Value (128)
        
        # 1. Kompresi Query (Low-Rank)
        self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_a_layernorm = DeepseekV3RMSNorm(config.q_lora_rank)
        self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False)
        
        # 2. Kompresi KV (Low-Rank Joint Compression) + RoPE (MQA)
        # Output dari sini akan di-cache selama inferensi
        self.kv_a_proj_with_mqa = nn.Linear(
            config.hidden_size, 
            config.kv_lora_rank + self.qk_rope_head_dim, 
            bias=False
        )
        self.kv_a_layernorm = DeepseekV3RMSNorm(config.kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            config.kv_lora_rank, 
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim), 
            bias=False
        )
        
        # 3. Output Projection
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, config.hidden_size, bias=False)
        self.scaling = self.qk_head_dim ** -0.5

    def forward(self, hidden_states, position_embeddings, attention_mask=None):
        batch_size, seq_length, _ = hidden_states.shape
        
        # --- QUERY PATH ---
        # Proyeksi down -> norm -> proyeksi up
        q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        # Reshape menjadi [batch, seq_len, heads, head_dim] -> [batch, heads, seq_len, head_dim]
        q_states = q_states.view(batch_size, seq_length, self.num_heads, self.qk_head_dim).transpose(1, 2)
        
        # Memecah Query: satu bagian untuk konten (nope), satu bagian untuk posisi (rope)
        q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        
        # --- KEY/VALUE PATH ---
        # Proyeksi kompresi KV
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        # Memecah laten KV dan Key RoPE
        k_pass_latent, k_rot = torch.split(compressed_kv, [self.config.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        
        # Dekompresi (Up-Projection) untuk mendapatkan Key (nope) dan Value
        k_pass_up = self.kv_b_proj(self.kv_a_layernorm(k_pass_latent))
        k_pass_up = k_pass_up.view(batch_size, seq_length, self.num_heads, self.qk_nope_head_dim + self.v_head_dim).transpose(1, 2)
        k_pass, value_states = torch.split(k_pass_up, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        
        # Reshape k_rot untuk di-broadcast ke semua head (MQA style)
        k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)
        
        # --- APPLY RoPE ---
        cos, sin = position_embeddings
        q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)
        k_rot = k_rot.expand(-1, self.num_heads, -1, -1)
        
        # --- GABUNGKAN KEMBALI KONTEN & POSISI ---
        query_states = torch.cat((q_pass, q_rot), dim=-1) # Shape: [batch, heads, seq_len, 192]
        key_states = torch.cat((k_pass, k_rot), dim=-1)   # Shape: [batch, heads, seq_len, 192]
        
        # --- ATTENTION COMPUTATION ---
        # Q * K^T
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
        # Softmax(Q * K^T) * V
        attn_output = torch.matmul(attn_weights, value_states) # Shape: [batch, heads, seq_len, 128]
        
        # Reshape dan Output Projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, -1)
        attn_output = self.o_proj(attn_output)
        
        return attn_output
```

---

### 4. DeepSeekMoE & *Auxiliary-Loss-Free Load Balancing*

**Teori:** Model Mixture of Experts (MoE) rawan mengalami *Routing Collapse*, di mana router hanya mengirim token ke 1 atau 2 *expert* saja, sementara *expert* lain mati. Secara tradisional, kita menggunakan "Auxiliary Loss" (penalti pada *Loss function*) untuk memaksa pembagian rata. Sayangnya, memodifikasi *Loss* utama merusak kapabilitas bahasa model. 

DeepSeek-V3 menyelesaikan ini menggunakan inovasi **Auxiliary-Loss-Free Load Balancing**. Mereka menggunakan parameter `e_score_correction_bias` yang ditambahkan pada skor logits (tanpa *gradient tracking*). Jika sebuah *expert* menerima terlalu banyak beban, *bias* ini dikurangi secara dinamis, memaksa router memilih *expert* lain. Ini murni modifikasi arsitektural saat _forward pass_, sehingga *loss function* utama tidak dikotori.

```python
class DeepseekV3MLP(nn.Module):
    """MLP Standar (SwiGLU) digunakan untuk layer awal dan Shared Expert"""
    def __init__(self, config: DeepseekV3Config, intermediate_size=None):
        super().__init__()
        self.intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        # SwiGLU: Down( SiLU(Gate(x)) * Up(x) )
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class DeepseekV3TopkRouter(nn.Module):
    """Router Dinamis untuk MoE dengan Auxiliary-Loss-Free Balancing"""
    def __init__(self, config: DeepseekV3Config):
        super().__init__()
        self.n_routed_experts = config.n_routed_experts
        # Bobot linier untuk memprediksi probabilitas setiap expert
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, config.hidden_size)))
        # Inovasi Utama: Bias untuk koreksi beban tanpa backprop pada Loss
        self.register_buffer("e_score_correction_bias", torch.zeros(self.n_routed_experts))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, hidden_states):
        # Shape: [batch * seq_len, hidden_size]
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        # Logits router: [batch * seq_len, n_routed_experts]
        router_logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32))
        return router_logits

class DeepseekV3NaiveMoe(nn.Module):
    """Koleksi Parameter Routed Experts (dimensi 3D)"""
    def __init__(self, config: DeepseekV3Config):
        super().__init__()
        self.num_experts = config.n_routed_experts
        self.intermediate_dim = config.moe_intermediate_size
        
        # Menyimpan bobot expert dalam tensor 3D untuk mempermudah indexing
        # Shape: [num_experts, 2 * intermediate_dim, hidden_size] (Gate & Up digabung)
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, config.hidden_size))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, config.hidden_size, self.intermediate_dim))
        self.act_fn = nn.SiLU()
        
        nn.init.normal_(self.gate_up_proj, std=0.02)
        nn.init.normal_(self.down_proj, std=0.02)

    def forward(self, hidden_states, top_k_index, top_k_weights):
        # hidden_states: [batch * seq_len, hidden_size]
        final_hidden_states = torch.zeros_like(hidden_states)
        
        # Buat mask one-hot untuk mengetahui expert mana yang memproses token mana
        # mask shape: [batch * seq_len, top_k, num_experts]
        expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
        expert_mask = expert_mask.permute(2, 1, 0) # [num_experts, top_k, batch * seq_len]
        
        # Cari expert mana saja yang aktif di batch ini
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            # Cari posisi top_k (ke-1 s/d ke-8) dan token index mana saja yang masuk ke expert ini
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            
            # Kumpulkan token yang sesuai (Gather)
            current_state = hidden_states[token_idx]
            
            # Operasi MLP SwiGLU pada batch token khusus untuk expert ini
            gate_up = F.linear(current_state, self.gate_up_proj[expert_idx])
            gate, up = gate_up.chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = F.linear(current_hidden_states, self.down_proj[expert_idx])
            
            # Kalikan dengan bobot probabilitas dari router
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            
            # Tambahkan kembali ke state akhir (Scatter)
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states

class DeepseekV3MoE(nn.Module):
    """Menggabungkan Router, Routed Experts, dan Shared Experts"""
    def __init__(self, config: DeepseekV3Config):
        super().__init__()
        self.config = config
        self.gate = DeepseekV3TopkRouter(config)
        self.experts = DeepseekV3NaiveMoe(config)
        
        # Shared Expert yang berukuran n_shared_experts kali lebih besar
        self.shared_experts = DeepseekV3MLP(
            config, intermediate_size=config.moe_intermediate_size * config.n_shared_experts
        )
        
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.top_k = config.num_experts_per_tok

    def route_tokens_to_experts(self, router_logits):
        router_probs = router_logits.sigmoid()
        # Menerapkan bias koreksi untuk load balancing
        router_logits_for_choice = router_probs + self.gate.e_score_correction_bias
        
        # Group-Level Routing (Membagi 256 expert menjadi 8 grup, pilih 4 grup terbaik)
        group_scores = (
            router_logits_for_choice.view(-1, self.n_group, self.config.n_routed_experts // self.n_group)
            .topk(2, dim=-1)[0].sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores).scatter_(1, group_idx, 1)
        score_mask = group_mask.unsqueeze(-1).expand(-1, self.n_group, self.config.n_routed_experts // self.n_group).reshape(-1, self.config.n_routed_experts)
        
        # Mematikan (masking) skor di luar grup terpilih
        scores_for_choice = router_logits_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
        
        # Expert-Level Routing (Pilih 8 expert teratas dari grup yang tersisa)
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        topk_weights = router_probs.gather(1, topk_indices)
        
        # Normalisasi pembobotan agar total = 1
        if self.config.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights /= denominator
            
        topk_weights = topk_weights * self.config.routed_scaling_factor
        return topk_indices, topk_weights

    def forward(self, hidden_states):
        residuals = hidden_states
        orig_shape = hidden_states.shape
        
        # 1. Routing
        router_logits = self.gate(hidden_states)
        topk_indices, topk_weights = self.route_tokens_to_experts(router_logits)
        
        # 2. Routed Experts
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        expert_output = self.experts(hidden_states, topk_indices, topk_weights).view(*orig_shape)
        
        # 3. Shared Expert (bekerja pada residual stream secara terpisah lalu digabung)
        shared_output = self.shared_experts(residuals)
        
        return expert_output + shared_output
```

---

### 5. Multi-Token Prediction (MTP) Module

**Teori:** Paper menjelaskan bahwa alih-alih melatih model untuk hanya memprediksi satu token berikutnya ($t+1$), DeepSeek-V3 melatih modul ekstra untuk memprediksi token $t+2$ (kedalaman $D=1$). Hal ini "mepadatkan" sinyal *loss function* sehingga melatih representasi data menjadi lebih efisien. Modul ini dihilangkan saat inferensi utama, namun dapat didaur-ulang (repurpose) untuk *Speculative Decoding*.

```python
class DeepseekV3MTPModule(nn.Module):
    """
    Modul untuk Multi-Token Prediction (MTP).
    Digunakan saat training untuk memprediksi token di kedalaman k (misal t+2).
    """
    def __init__(self, config: DeepseekV3Config):
        super().__init__()
        # Proyeksi linier untuk menggabungkan representasi k-1 dan embedding target t+k
        self.proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.rms_norm = DeepseekV3RMSNorm(config.hidden_size)
        
        # Satu blok Transformer sederhana (Self-Attention + MLP) khusus untuk MTP
        # Untuk simplifikasi akademik, kita instansiasi DecoderLayer biasa
        self.transformer_block = DeepseekV3DecoderLayer(config, layer_idx=999) 

    def forward(self, hidden_states_k_minus_1, target_embeddings):
        # 1. Concatenate representasi kedalaman sebelumnya dengan embedding target token
        # Shape: [batch, seq_len, hidden_size * 2]
        concat_repr = torch.cat([hidden_states_k_minus_1, target_embeddings], dim=-1)
        
        # 2. Proyeksi Linier dan Norm
        h_k = self.rms_norm(self.proj(concat_repr))
        
        # 3. Lewati block Transformer (Perhatikan: kita butuh attention_mask causal di dunia nyata)
        output = self.transformer_block(h_k, attention_mask=None, position_embeddings=None)
        
        return output
```

---

### 6. Arsitektur Keseluruhan (Model & Causal LM)

Sekarang, kita gabungkan semua bata lego (komponen) yang telah kita buat ke dalam *layer* dan kerangka model besar.

```python
class DeepseekV3DecoderLayer(nn.Module):
    def __init__(self, config: DeepseekV3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        
        self.input_layernorm = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = DeepseekV3Attention(config)
        self.post_attention_layernorm = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Inovasi Arsitektural: Ganti K-layer pertama dengan Dense MLP, sisanya MoE
        if layer_idx >= config.first_k_dense_replace:
            self.mlp = DeepseekV3MoE(config)
        else:
            self.mlp = DeepseekV3MLP(config)

    def forward(self, hidden_states, attention_mask=None, position_embeddings=None):
        # Pre-Norm + Attention + Residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_embeddings, attention_mask)
        hidden_states = residual + hidden_states

        # Pre-Norm + FFN (Dense / MoE) + Residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states

class DeepseekV3Model(nn.Module):
    """Inti dari DeepSeek-V3 Transformer"""
    def __init__(self, config: DeepseekV3Config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Instansiasi 61 Layer
        self.layers = nn.ModuleList(
            [DeepseekV3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = DeepseekV3RotaryEmbedding(config.qk_rope_head_dim)

    def forward(self, input_ids):
        # [batch, seq_len, hidden_size]
        hidden_states = self.embed_tokens(input_ids)
        
        # Membangun tensor identitas posisi untuk RoPE
        seq_length = input_ids.shape[1]
        position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        
        # Masking Causal (Causal Mask) untuk mencegah model melihat masa depan
        # Disembunyikan (diasumsikan sudah dibentuk) dalam implementasi ini
        causal_mask = None 

        for layer in self.layers:
            hidden_states = layer(
                hidden_states, 
                attention_mask=causal_mask, 
                position_embeddings=position_embeddings
            )
            
        hidden_states = self.norm(hidden_states)
        return hidden_states

class DeepseekV3ForCausalLM(nn.Module):
    """Model End-to-End dengan Language Modeling Head"""
    def __init__(self, config: DeepseekV3Config):
        super().__init__()
        self.config = config
        self.model = DeepseekV3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Opsional: Tie Weights antara embedding masuk dan keluar
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids):
        # Mendapatkan representasi laten dari token
        hidden_states = self.model(input_ids)
        
        # Memproyeksikan kembali ke distribusi probabilitas kosakata (Vocab Size)
        # Biasanya saat inferensi kita hanya butuh token terakhir
        logits = self.lm_head(hidden_states)
        
        return logits
```

### Kesimpulan
Melalui rekonstruksi di atas, rekan-rekan mahasiswa telah melihat secara matematis dan programatik bagaimana **DeepSeek-V3** mengatasi hukum batasan skala (scaling laws). MLA menekan biaya memori *Attention*, sedangkan DeepSeekMoE yang dikombinasikan dengan *Auxiliary-Loss-Free* menekan biaya komputasi *Feed-Forward Network* (FFN) tanpa merusak kecerdasan model. 

Silakan telusuri kembali setiap `shape` tensor dalam kode di atas. Memahami perubahan dimensi adalah kunci utama dalam menguasai rekayasa arsitektur LLM. Selamat belajar!