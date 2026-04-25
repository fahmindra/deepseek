---
slug: modul-7-6
title: modul-7-6
---
# 3.6 Implementasi Fine-Tuning Menggunakan Unsloth

Kita sudah punya data yang bersih (3.5) dan konfigurasi hemat memori (3.2). Sekarang saatnya menekan tombol "Train".

Namun, menggunakan library standar (Hugging Face Trainer), melatih model Llama-3 8B bisa memakan waktu berjam-jam dan seringkali terkena Out-of-Memory (OOM).

Masuklah Unsloth.

Di subtopik penutup ini, kita akan membahas library yang menjadi standar de facto untuk fine-tuning cepat. Unsloth bukan metode baru, melainkan optimasi level kernel yang membuat proses pelatihan 2x-5x lebih cepat dan 60% lebih hemat memori tanpa mengorbankan akurasi.

---

## 3.6.1 Apa itu Unsloth?

Unsloth adalah library ringan yang mengoptimalkan proses training dan inference untuk model arsitektur Llama, Mistral, dan Gemma.

**Keunggulan Utama:**
1.  **Kecepatan:** Mampu melatih model 2x-5x lebih cepat dibandingkan implementasi standar Hugging Face.
2.  **Memori:** Mengurangi penggunaan VRAM hingga 60%, memungkinkan batch size yang lebih besar (misal: bisa melatih Llama-3 8B dengan context 8192 token di satu GPU gratisan T4, yang mustahil dilakukan dengan library biasa).
3.  **Akurasi:** Tidak menggunakan aproksimasi (seperti kuantisasi yang buruk); ia menghitung gradien secara eksak (mathematically equivalent) tapi dengan jalur yang lebih efisien.

**Rahasia Unsloth:**
Unsloth menulis ulang kernel GPU (program kecil yang jalan di kartu grafis) untuk operasi-operasi berat seperti Self-Attention dan MLP menggunakan bahasa OpenAI Triton.

Alih-alih menghitung banyak langkah kecil terpisah (baca memori -> hitung -> tulis memori), Unsloth menggabungkan operasi-operasi tersebut (kernel fusion) untuk meminimalkan lalu lintas data di VRAM.

---

## 3.6.2 Implementasi Kode: Training Super Cepat dengan Unsloth

Berikut adalah setup lengkap menggunakan Unsloth. Perhatikan betapa miripnya dengan kode transformers biasa, namun dengan performa yang jauh berbeda.

### 1. Instalasi
Kita mulai dengan instalasi terlebih dahulu:

```bash
# Instalasi Unsloth (Versi Colab)
# Unsloth membutuhkan instalasi spesifik tergantung versi CUDA
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes
```

Unsloth sangat bergantung pada hardware. Kita menginstal versi `[colab-new]` yang sudah memaketkan kernel yang dikompilasi untuk GPU NVIDIA T4/V100/A100 yang ada di Colab. `xformers` adalah library lain dari Meta untuk mempercepat attention.

### 2. Memuat Model
Kita mengganti `AutoModel` dengan `FastLanguageModel`. Kelas ini secara otomatis menyuntikkan optimasi memori saat model dimuat.

```python
from unsloth import FastLanguageModel
import torch


max_seq_length = 2048 # Unsloth sangat efisien menangani konteks panjang
dtype = None # None = Auto-detect (Float16 untuk T4, Bfloat16 untuk Ampere/A100)
load_in_4bit = True # Gunakan 4-bit quantization (QLoRA) bawaan Unsloth


# 1. Memuat Model
# Kita TIDAK menggunakan AutoModelForCausalLM, tapi FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit", # Versi unsloth sudah dioptimalkan
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
```

### 3. Menambahkan Adapter LoRA
Unsloth memiliki implementasi LoRA sendiri yang lebih cepat dari library `peft` standar.

```python
# 2. Menambahkan Adapter LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Rank (ukuran adapter)
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0, # PENTING: Set 0 agar lebih cepat (Unsloth optimized)
    bias = "none",
   
    # Fitur Ajaib: Gradient Checkpointing khusus Unsloth
    # Ini menghemat VRAM secara masif dengan membuang aktivasi sementara
    # dan menghitungnya ulang saat dibutuhkan.
    use_gradient_checkpointing = "unsloth",
   
    random_state = 3407,
)
```

Di sini kita mengonfigurasi LoRA. Perbedaan kuncinya adalah `use_gradient_checkpointing="unsloth"`. Fitur ini memungkinkan kita memuat batch size yang lebih besar atau konteks yang lebih panjang daripada yang biasanya muat di GPU, dengan cara mengelola memori VRAM secara cerdas di sela-sela perhitungan forward dan backward pass.

### 4. Persiapan Data dan Training
Kita menggunakan `SFTTrainer` dari library `trl` (yang kompatibel dengan Unsloth).

```python
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset


# Load dataset contoh (Alpaca)
dataset = load_dataset("yahma/alpaca-cleaned", split = "train[:500]")


# Fungsi formatting sederhana (Prompt Engineering)
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Format Alpaca standar
        text = f"""### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}""" + tokenizer.eos_token
        texts.append(text)
    return { "text" : texts, }


dataset = dataset.map(formatting_prompts_func, batched = True,)


# Setup Trainer
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
   
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60, # Demo cepat (60 langkah)
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
       
        # Optimizer khusus: adamw_8bit menghemat memori optimizer hingga 75%
        optim = "adamw_8bit",
       
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)


# Jalankan Training
print("Mulai Training dengan Unsloth...")
trainer_stats = trainer.train()
print("Selesai!")
```

Poin penting di sini adalah `optim="adamw_8bit"`. Optimizer standar (AdamW) menyimpan dua "state" (momentum & variance) untuk setiap parameter dalam 32-bit, yang sangat boros memori. `adamw_8bit` mengompres state ini menjadi 8-bit, memberikan penghematan memori VRAM yang signifikan tanpa penurunan kinerja yang berarti.

> **Refleksi Singkat**
>
> Kita telah melihat bahwa Unsloth mampu mempercepat pelatihan hingga 2x-5x lipat. Bayangkan Anda adalah seorang AI Engineer freelance. Sebelumnya, untuk melakukan fine-tuning model Llama-3 bagi klien, Anda harus menyewa GPU A100 (mahal) di cloud. Dengan Unsloth, Anda bisa melakukannya di GPU T4 (gratis di Colab atau murah di AWS).
>
> Bagaimana efisiensi ekstrem ini mengubah model bisnis Anda? Apakah ini memungkinkan Anda untuk menawarkan layanan custom fine-tuning kepada UMKM yang sebelumnya tidak mampu membayar biaya infrastruktur AI?