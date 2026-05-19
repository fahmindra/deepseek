---
slug: part-4
layout: part
---

Selamat datang di modul praktikum Arsitektur *Large Language Model* (LLM) Lanjut. Saya selaku dosen pengampu Anda telah menyusun modul ini secara komprehensif agar Anda dapat memahami secara mendalam arsitektur **DeepSeek-V3**. 

DeepSeek-V3 adalah model terobosan dengan pendekatan *Mixture-of-Experts* (MoE) dan *Multi-Head Latent Attention* (MLA) yang sangat efisien. Dalam modul ini, kita akan mengimplementasikan model tersebut dari nol (*from scratch*) menggunakan PyTorch. Fokus utama kita adalah memahami kompresi laten pada MLA (Persamaan 1-11) dan strategi *Auxiliary-Loss-Free Load Balancing* pada MoE (Persamaan 12-16) seperti yang tertuang dalam *DeepSeek-V3 Technical Report (arXiv:2412.19437v2)*.

Silakan pelajari, pahami, dan jalankan kode di bawah ini. Kode ini ditulis tanpa pemotongan agar Anda dapat melihat aliran tensor secara utuh.

```python
"""
Modul Praktikum DeepSeek-V3
Disusun oleh: Dosen Ahli Kecerdasan Buatan & ML Engineer
Mata Kuliah: Arsitektur Large Language Model Lanjut
Referensi Utama: DeepSeek-V3 Technical Report (arXiv:2412.19437v2)
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# 1. KONFIGURASI MODEL (DEEPSEEK-V3 CONFIG)
# ==============================================================================

@dataclass
class DeepseekV3Config:
    """
    Kelas konfigurasi yang merepresentasikan seluruh hiperparameter arsitektur DeepSeek-V3.
    Nilai default disesuaikan dengan representasi model 671B (dengan 37B parameter aktif)
    berdasarkan DeepSeek-V3 Technical Report.
    """
    # Parameter Dasar
    vocab_size: int = 129280
    hidden_size: int = 7168
    num_hidden_layers: int = 61
    num_attention_heads: int = 128
    num_key_value_heads: int = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 4096
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    
    # Konfigurasi Multi-Head Latent Attention (MLA)
    # Merujuk pada Bagian 2.1.1 (Multi-Head Latent Attention)
    kv_lora_rank: int = 512          # d_c: Dimensi kompresi Key-Value
    q_lora_rank: int = 1536          # d_c': Dimensi kompresi Query
    qk_rope_head_dim: int = 64       # Dimensi head untuk decoupled RoPE
    qk_nope_head_dim: int = 128      # Dimensi head tanpa RoPE (No-PE)
    v_head_dim: int = 128            # Dimensi head untuk Value
    rope_interleave: bool = True     # Strategi penyisipan dimensi RoPE
    
    # Konfigurasi DeepSeekMoE (Mixture of Experts)
    # Merujuk pada Bagian 2.1.2 dan implementasi HF
    moe_intermediate_size: int = 2048 # Dimensi intermediate tiap expert
    n_shared_experts: int = 1         # N_s: Jumlah shared expert (selalu aktif)
    n_routed_experts: int = 256       # N_r: Total routed experts yang tersedia
    num_experts_per_tok: int = 8      # K_r: Jumlah expert yang dipilih per token
    n_group: int = 8                  # Jumlah grup ahli (untuk Node-Limited Routing)
    topk_group: int = 4               # Jumlah grup yang dipilih per token
    routed_scaling_factor: float = 2.5 # Faktor skala untuk output routed expert
    norm_topk_prob: bool = True       # Normalisasi probabilitas top-K
    first_k_dense_replace: int = 3    # 3 Layer pertama menggunakan Dense MLP konvensional


# ==============================================================================
# 2. KOMPONEN DASAR (RMSNorm & RoPE)
# ==============================================================================

class DeepseekV3RMSNorm(nn.Module):
    """
    Root Mean Square Normalization (RMSNorm).
    Menjaga variansi aktivasi agar stabil selama pelatihan tanpa melakukan pergeseran mean.
    """
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        # Konversi ke float32 untuk stabilitas numerik saat menghitung variansi
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class DeepseekV3RotaryEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE).
    Memberikan representasi posisi relatif melalui rotasi pada ruang vektor laten.
    """
    def __init__(self, config: DeepseekV3Config):
        super().__init__()
        self.dim = config.qk_rope_head_dim
        self.max_position_embeddings = config.max_position_embeddings
        self.base = config.rope_theta
        
        # Menghitung frekuensi invers: 1 / (base ^ (2i / dim))
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # x shape: [batch_size, seq_len, num_heads, head_dim]
        t = torch.arange(seq_len, device=x.device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        # Bentuk tensor akhir: [seq_len, dim]
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Merotasi setengah dari dimensi tersembunyi (digunakan dalam RoPE)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_interleave(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Mengaplikasikan RoPE dengan metode interleaving (penyisipan).
    Mengubah bentuk dari dimensi untuk disesuaikan dengan format rotasi q dan k.
    """
    cos = cos.unsqueeze(0).unsqueeze(2) # [1, seq_len, 1, dim]
    sin = sin.unsqueeze(0).unsqueeze(2)
    
    b, s, h, d = q.shape
    q = q.view(b, s, h, d // 2, 2).transpose(4, 3).reshape(b, s, h, d)
    
    b, s, h, d = k.shape
    k = k.view(b, s, h, d // 2, 2).transpose(4, 3).reshape(b, s, h, d)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# ==============================================================================
# 3. MULTI-HEAD LATENT ATTENTION (MLA)
# ==============================================================================

class DeepseekV3Attention(nn.Module):
    """
    Multi-Head Latent Attention (MLA).
    Berdasarkan DeepSeek-V3 Technical Report Bagian 2.1.1.
    Menggunakan Low-Rank Joint Compression untuk efisiensi KV-Cache saat inferensi.
    """
    def __init__(self, config: DeepseekV3Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        
        # Dimensi spesifik MLA
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim

        # ---------------------------------------------------------
        # KOMPRESI QUERY (Persamaan 6 & 7 di Paper)
        # c_t^Q = W^{DQ} * h_t (Kompresi ke latent vector)
        # q_t^C = W^{UQ} * c_t^Q (Dekompresi)
        # ---------------------------------------------------------
        self.q_a_proj = nn.Linear(self.hidden_size, self.q_lora_rank, bias=False)
        self.q_a_layernorm = DeepseekV3RMSNorm(self.q_lora_rank)
        self.q_b_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False)

        # ---------------------------------------------------------
        # KOMPRESI KEY-VALUE (Persamaan 1, 2, 3, 5 di Paper)
        # Memisahkan kompresi latent c_t^{KV} dan Decoupled RoPE k_t^R
        # ---------------------------------------------------------
        # Menghasilkan [c_t^{KV}, k_t^R] secara bersamaan
        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size, 
            self.kv_lora_rank + self.qk_rope_head_dim, 
            bias=False
        )
        self.kv_a_layernorm = DeepseekV3RMSNorm(self.kv_lora_rank)
        # Mendekompresi c_t^{KV} menjadi [k_t^C, v_t^C]
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank, 
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim), 
            bias=False
        )

        # Proyeksi Output (Persamaan 11 di Paper)
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, self.hidden_size, bias=False)
        self.scaling = self.qk_head_dim ** -0.5

    def forward(
        self, 
        hidden_states: torch.Tensor, 
        attention_mask: Optional[torch.Tensor], 
        position_embeddings: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # 1. Menghitung Query dengan Low-Rank Compression
        # q_states menjadi representasi gabungan (No-PE dan RoPE)
        q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q_states = q_states.view(batch_size, seq_len, self.num_heads, self.qk_head_dim)
        
        # Pisahkan menjadi q_pass (No-PE) dan q_rot (RoPE)
        q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # 2. Menghitung Key dan Value dengan Low-Rank Compression
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_latent, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        
        # Dekompresi latent menjadi k_pass (No-PE) dan v_states
        k_pass_v_states = self.kv_b_proj(self.kv_a_layernorm(k_latent))
        k_pass_v_states = k_pass_v_states.view(batch_size, seq_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_pass, v_states = torch.split(k_pass_v_states, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # k_rot (Decoupled Key RoPE) di-broadcast ke seluruh head (mirip MQA)
        k_rot = k_rot.view(batch_size, seq_len, 1, self.qk_rope_head_dim)
        k_rot = k_rot.expand(-1, -1, self.num_heads, -1)

        # 3. Mengaplikasikan RoPE pada komponen ter-decouple (q_rot dan k_rot)
        cos, sin = position_embeddings
        q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)

        # 4. Penggabungan Kembali (Persamaan 4 & 9 di Paper)
        # q_t = [q_t^C; q_t^R] dan k_t = [k_t^C; k_t^R]
        query_states = torch.cat((q_pass, q_rot), dim=-1).transpose(1, 2) # [B, H, S, D]
        key_states = torch.cat((k_pass, k_rot), dim=-1).transpose(1, 2)   # [B, H, S, D]
        value_states = v_states.transpose(1, 2)                           # [B, H, S, D]

        # 5. Menghitung Attention Scores (Persamaan 10 di Paper)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_probs, value_states)

        # 6. Proyeksi Output Akhir
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.num_heads * self.v_head_dim)
        output = self.o_proj(attn_output)

        return output


# ==============================================================================
# 4. DEEPSEEK MIXTURE-OF-EXPERTS (MoE) DENGAN LOAD BALANCING
# ==============================================================================

class DeepseekV3MLP(nn.Module):
    """
    Feed-Forward Network (Dense MLP) standar menggunakan SwiGLU.
    Digunakan pada shared experts dan k-layer pertama model.
    """
    def __init__(self, config: DeepseekV3Config, intermediate_size: Optional[int] = None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size if intermediate_size else config.hidden_size * 4
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU Activation: Down(SiLU(Gate(x)) * Up(x))
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class DeepseekV3TopkRouter(nn.Module):
    """
    Router / Gating Mechanism untuk MoE.
    Mengimplementasikan Auxiliary-Loss-Free Load Balancing sesuai Bagian 2.1.2.
    """
    def __init__(self, config: DeepseekV3Config):
        super().__init__()
        self.config = config
        self.n_routed_experts = config.n_routed_experts
        
        # Vektor centroid e_i untuk setiap routed expert (Persamaan 15)
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, config.hidden_size)))
        nn.init.normal_(self.weight, std=config.initializer_range)
        
        # Bias b_i untuk load balancing tanpa auxiliary loss (Persamaan 16)
        # Secara adaptif akan diperbarui (ditambah/dikurang) selama backprop (tidak dimodelkan explisit di forward pass dasar, melainkan diatur optimizer/scheduler).
        self.register_buffer("e_score_correction_bias", torch.zeros(self.n_routed_experts))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Menghitung afinitas (dot product) antara token u_t dan centroid e_i
        hidden_states_flat = hidden_states.view(-1, self.config.hidden_size)
        router_logits = F.linear(hidden_states_flat, self.weight)
        return router_logits


class DeepseekV3NaiveMoe(nn.Module):
    """
    Implementasi batched computation untuk Routed Experts.
    Setiap token diarahkan hanya ke expert yang relevan berdasarkan probabilitas top-K.
    """
    def __init__(self, config: DeepseekV3Config):
        super().__init__()
        self.num_experts = config.n_routed_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        
        # Bobot seluruh expert digabungkan ke dalam bentuk tensor 3D untuk efisiensi
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
        
        nn.init.normal_(self.gate_up_proj, std=config.initializer_range)
        nn.init.normal_(self.down_proj, std=config.initializer_range)
        self.act_fn = nn.SiLU()

    def forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor) -> torch.Tensor:
        """
        Melakukan komputasi *sparse* di mana token hanya dihitung pada expert yang ditentukan.
        """
        final_hidden_states = torch.zeros_like(hidden_states)
        
        # Membuat mask expert yang dipilih per token
        expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
        
        for expert_idx in range(self.num_experts):
            # Mencari token mana saja yang dialokasikan ke expert_idx
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            if token_idx.numel() == 0:
                continue
                
            current_state = hidden_states[token_idx]
            
            # Operasi MLP SwiGLU untuk expert saat ini
            gate_up = F.linear(current_state, self.gate_up_proj[expert_idx])
            gate, up = gate_up.chunk(2, dim=-1)
            expert_out = self.act_fn(gate) * up
            expert_out = F.linear(expert_out, self.down_proj[expert_idx])
            
            # Mengalikan dengan bobot gating (routing weight)
            expert_out = expert_out * top_k_weights[token_idx, top_k_pos].unsqueeze(-1)
            
            # Menambahkan kembali (*scatter add*) ke tensor keluaran akhir
            final_hidden_states.index_add_(0, token_idx, expert_out.to(final_hidden_states.dtype))

        return final_hidden_states


class DeepseekV3MoE(nn.Module):
    """
    Arsitektur Mixture-of-Experts DeepSeek-V3.
    Menggabungkan hasil dari N_s Shared Experts dan K_r Routed Experts (Persamaan 12).
    Menggunakan Node-Limited/Group-Limited Routing.
    """
    def __init__(self, config: DeepseekV3Config):
        super().__init__()
        self.config = config
        self.experts = DeepseekV3NaiveMoe(config)
        self.gate = DeepseekV3TopkRouter(config)
        
        # Shared expert merupakan MLP tunggal berukuran besar (N_s dikalikan intermediate size)
        self.shared_experts = DeepseekV3MLP(
            config=config, 
            intermediate_size=config.moe_intermediate_size * config.n_shared_experts
        )
        
        self.n_routed_experts = config.n_routed_experts
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.top_k = config.num_experts_per_tok

    def route_tokens_to_experts(self, router_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Implementasi Strategi Routing berdasar Grup (Persamaan 16).
        """
        # Normalisasi nilai afinitas menggunakan fungsi Sigmoid (Persamaan 15)
        router_probs = router_logits.sigmoid()
        
        # Menambahkan bias b_i untuk auxiliary-loss-free balancing (Persamaan 16)
        router_logits_for_choice = router_probs + self.gate.e_score_correction_bias
        
        # Membagi expert ke dalam kelompok (Grup)
        # Dimensi: [jumlah_token, n_group, experts_per_group]
        experts_per_group = self.n_routed_experts // self.n_group
        group_scores = (
            router_logits_for_choice.view(-1, self.n_group, experts_per_group)
            .topk(2, dim=-1)[0] # Mengambil 2 skor teratas di tiap grup
            .sum(dim=-1)        # Menjumlahkan untuk mendapatkan representasi skor grup
        )
        
        # Memilih Top-K grup untuk dipertahankan
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores).scatter_(1, group_idx, 1)
        
        # Me-masking expert yang berada di grup yang tidak terpilih
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, experts_per_group)
            .reshape(-1, self.n_routed_experts)
        )
        scores_for_choice = router_logits_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
        
        # Memilih K_r (top_k) routed experts secara final
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        topk_weights = router_probs.gather(1, topk_indices) # Menggunakan probabilitas asli tanpa bias

        # Normalisasi afinitas (agar total bobot expert yang terpilih = 1 jika diperlukan)
        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights = topk_weights / denominator
            
        # Penskalaan bobot akhir
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, orig_shape[-1])
        
        # 1. Hitung Router/Gate
        router_logits = self.gate(hidden_states)
        topk_indices, topk_weights = self.route_tokens_to_experts(router_logits)
        
        # 2. Komputasi Routed Experts
        routed_out = self.experts(hidden_states_flat, topk_indices, topk_weights).view(orig_shape)
        
        # 3. Komputasi Shared Experts
        shared_out = self.shared_experts(hidden_states)
        
        # 4. Penggabungan H_t = Shared(U_t) + \sum Routed(U_t) (Persamaan 12)
        return routed_out + shared_out


# ==============================================================================
# 5. LAPISAN DEKODER & MODEL UTAMA
# ==============================================================================

class DeepseekV3DecoderLayer(nn.Module):
    """
    Satu lapisan penuh dari DeepSeek-V3 (RMSNorm -> MLA -> RMSNorm -> MoE/MLP).
    """
    def __init__(self, config: DeepseekV3Config, layer_idx: int):
        super().__init__()
        self.self_attn = DeepseekV3Attention(config=config)
        self.input_layernorm = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # 3 layer pertama menggunakan Dense MLP untuk menangkap representasi dasar, 
        # sisanya menggunakan arsitektur MoE untuk spesialisasi dan efisiensi.
        if layer_idx >= config.first_k_dense_replace:
            self.mlp = DeepseekV3MoE(config)
        else:
            self.mlp = DeepseekV3MLP(config)

    def forward(
        self, 
        hidden_states: torch.Tensor, 
        attention_mask: Optional[torch.Tensor], 
        position_embeddings: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        
        # Blok 1: Multi-Head Latent Attention dengan Residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states, 
            attention_mask=attention_mask, 
            position_embeddings=position_embeddings
        )
        hidden_states = residual + hidden_states

        # Blok 2: Feed-Forward (Dense/MoE) dengan Residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class DeepseekV3Model(nn.Module):
    """
    Model Inti (Base Model) DeepSeek-V3 tanpa head Language Modeling.
    Menerima token ID dan memprosesnya melalui seluruh N lapisan Dekoder.
    """
    def __init__(self, config: DeepseekV3Config):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            DeepseekV3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = DeepseekV3RotaryEmbedding(config)

    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Membuat matriks mask kausal (Autoregressive)."""
        mask = torch.tril(torch.ones((seq_len, seq_len), device=device)).view(1, 1, seq_len, seq_len)
        # Modifikasi nilai 0 menjadi -inf agar softmax bernilai 0 di token masa depan
        mask = mask.masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, 0.0)
        return mask

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Pemrosesan Token Embedding
        hidden_states = self.embed_tokens(input_ids)
        
        # Pembangkitan Mask Kausal dan Positional Embedding (RoPE)
        causal_mask = self._create_causal_mask(seq_len, device)
        position_embeddings = self.rotary_emb(hidden_states, seq_len)

        # Melewati seluruh lapisan Dekoder secara hierarkis
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings
            )

        # Normalisasi akhir sebelum Output Head
        hidden_states = self.norm(hidden_states)
        return hidden_states


class DeepseekV3ForCausalLM(nn.Module):
    """
    Model Causal Language Modeling DeepSeek-V3.
    Digunakan untuk melatih model memprediksi token berikutnya (Next-Token Prediction).
    """
    def __init__(self, config: DeepseekV3Config):
        super().__init__()
        self.config = config
        self.model = DeepseekV3Model(config)
        # Proyeksi linear ke ukuran vokabulari untuk memproduksi logits
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Mendapatkan representasi laten akhir dari model dasar
        hidden_states = self.model(input_ids)
        
        # Menghitung probabilistik logit
        logits = self.lm_head(hidden_states)

        loss = None
        # Apabila label diberikan (fase pelatihan), hitung kerugian Cross-Entropy
        if labels is not None:
            # Menggeser posisi (Shift) agar token memprediksi token di t+1
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Meratakan dimensi tensor untuk fungsi Loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        return logits, loss

# ==============================================================================
# CONTOH PENGGUNAAN (TEST INSTANTIATION)
# ==============================================================================
if __name__ == "__main__":
    # Menginisialisasi arsitektur menggunakan nilai miniatur (untuk ujicoba komputer lokal)
    # Catatan untuk Mahasiswa: Ubah nilai dimensi ini pada percobaan sesungguhnya.
    mini_config = DeepseekV3Config(
        hidden_size=256, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=4,
        kv_lora_rank=32, q_lora_rank=64, qk_rope_head_dim=16, qk_nope_head_dim=16, v_head_dim=16,
        n_routed_experts=8, num_experts_per_tok=2, n_group=2, topk_group=1, first_k_dense_replace=1
    )
    
    # Inisiasi model
    print("Membangun model DeepSeek-V3 (Miniatur)...")
    model = DeepseekV3ForCausalLM(mini_config)
    
    # Membuat tensor acak untuk input dan label (Batch=2, Sequence_Length=16)
    dummy_input_ids = torch.randint(0, mini_config.vocab_size, (2, 16))
    dummy_labels = dummy_input_ids.clone()
    
    print("Melakukan forward pass...")
    logits, loss = model(dummy_input_ids, labels=dummy_labels)
    
    print(f"Bentuk Logits: {logits.shape}") # Harapan: [2, 16, 129280]
    print(f"Nilai Kerugian (Loss): {loss.item():.4f}")
    print("Implementasi berhasil dieksekusi!")
```