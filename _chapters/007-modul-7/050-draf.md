---
slug: modul-7-5
title: modul-7-5
---
# 3.5 Persiapan Data untuk Fine-Tuning: Struktur dan Format

Kita sudah membahas teknik canggih (LoRA, QLoRA) dan arsitektur kompleks (MoE). Namun, ada satu kebenaran pahit dalam Machine Learning: Model terhebat di dunia akan gagal total jika diberi makan data sampah ("Garbage In, Garbage Out").

Di subtopik ini, kita akan membahas *Data Preparation*. Ini bukan sekadar "membersihkan teks". Ini adalah seni memformat instruksi agar model mengerti perbedaan antara "User berbicara" dan "Assistant menjawab". Salah format sedikit saja (misal: lupa token EOS), model Anda akan rusak (terus mengoceh tanpa henti).

Dalam paradigma Software 2.0 (Andrej Karpathy), bobot neural network adalah kode programnya, dan dataset adalah source code-nya. Jika source code Anda berantakan, program Anda akan buggy.

Untuk Instruction Fine-Tuning (SFT), kita tidak bisa sembarangan melempar teks PDF. Kita harus menyusun data ke dalam format terstruktur yang mengajarkan model pola interaksi. Di subtopik ini, kita akan membedah dua standar industri utama (Alpaca dan ShareGPT), memahami peran krusial *Chat Templates*, dan melihat cara memvalidasi data sebelum membakar uang untuk GPU.

---

## 3.5.1 Format Standar Industri: Alpaca vs. ShareGPT

Ada dua format JSON utama yang diterima oleh sebagian besar library training (seperti Axolotl, Llama-Factory, atau Unsloth).

### 1. Format Alpaca (Single-Turn)
Ini dipopulerkan oleh Stanford Alpaca. Cocok untuk tugas instruksi sederhana (Input -> Output).

**Struktur:**
*   `instruction`: Perintah utama.
*   `input`: Konteks tambahan (opsional).
*   `output`: Jawaban yang diharapkan.

**Contoh JSON (Alpaca):**
```json
{
  "instruction": "Jelaskan apa itu LoRA.",
  "input": "",
  "output": "LoRA adalah teknik fine-tuning yang..."
}
```

### 2. Format ShareGPT (Multi-Turn / Conversational)
Ini adalah standar modern untuk melatih chatbot. Format ini menangkap sejarah percakapan.

**Struktur:** List of messages (role & content).
*   **Keunggulan:** Model belajar mempertahankan konteks antar-giliran bicara.

**Contoh JSON (ShareGPT):**
```json
{
  "conversations": [
    {"from": "human", "value": "Hai, siapa kamu?"},
    {"from": "gpt", "value": "Saya asisten AI."},
    {"from": "human", "value": "Bisa bantu coding?"},
    {"from": "gpt", "value": "Tentu, bahasa apa?"}
  ]
}
```

Pemilihan format ini bergantung pada apakah Anda ingin melatih model untuk tugas instruksi sederhana atau percakapan panjang. Berikut adalah perbedaannya dalam bentuk tabel.

**Tabel 3. Perbedaan Format Alpaca dan ShareGPT**

| Aspek | Format Alpaca | Format ShareGPT |
| :--- | :--- | :--- |
| **Tipe Interaksi** | **Single-Turn** (Satu tanya, satu jawab) | **Multi-Turn** (Percakapan timbal balik). |
| **Fokus Tugas** | *Instruction Following* (misal: meringkas, klasifikasi, ekstraksi). | *Chatbot / Assistant* (mempertahankan konteks percakapan). |
| **Struktur JSON** | Tiga *key* utama: `instruction`, `input` (opsional), `output`. | List of messages: `conversations` berisi `from` dan `value`. |
| **Contoh JSON** | `{"instruction": "Jelaskan LoRA", "input": "", "output": "LoRA adalah..."}` | `{"conversations" : [ {"from": "human", "value": "Hai"}, {"from": "gpt", "value": "Halo!"} ]}` |
| **Penggunaan** | Standar lama (Stanford Alpaca), bagus untuk tugas non-chat. | Standar modern (Vicuna, Llama-3), wajib untuk melatih asisten AI. |

---

## 3.5.2 Komponen Krusial: Chat Templates

Ini adalah bagian yang paling sering salah dipahami oleh pemula.

Model LLM (Base Model) sebenarnya tidak mengerti JSON di atas. Ia hanya mengerti **Satu String Panjang**. Kita membutuhkan *Chat Template* untuk mengubah struktur JSON menjadi string yang mengandung **Special Tokens**.

Mengapa butuh *Special Tokens*? Agar model tahu kapan giliran User selesai dan kapan giliran Assistant dimulai.

**Contoh Transformasi (Llama-3 Template):**
*   Input JSON: `{"role": "user", "content": "Hello"}`
*   Output String (Template Llama-3):
```text
<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Hello<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```

Jika Anda salah menggunakan template (misal: pakai template ChatML untuk model Llama-3), model akan menjadi bodoh karena tidak mengenali struktur percakapannya sendiri.

---

## 3.5.3 Teknik Lanjutan: Data Masking & Sequence Packing

Dalam persiapan data untuk SFT, sekadar mengubah format JSON menjadi string tidaklah cukup. Ada dua teknik manipulasi data tingkat lanjut yang wajib diterapkan untuk memastikan efisiensi pelatihan dan kebenaran perilaku model.

### 1. Data Masking (Loss Masking): "Jangan Menilai Pertanyaan"
Saat kita melatih model chatbot, input kita terdiri dari dua bagian: Instruksi/Prompt (dari User) dan Respons (dari Assistant). Secara default, *Causal Language Modeling* (CLM) akan menghitung loss (kesalahan prediksi) untuk **setiap** token dalam kalimat, dari awal hingga akhir.

Masalahnya: Jika model juga dilatih untuk memprediksi prompt user, ia akan membuang kapasitas otaknya untuk mempelajari "gaya bertanya user", bukan "gaya menjawab assistant".

Solusinya adalah **Masking**. Kita secara eksplisit memberi tahu fungsi loss untuk mengabaikan semua token yang berada di bagian instruksi.

**Mekanisme Teknis:** Dalam PyTorch/Hugging Face, token yang ingin diabaikan diberi label nilai -100.
*   **Input IDs:** [User, Apa, kabar?, Assistant, Baik]
*   **Labels:** [-100, -100, -100, -100, Baik]

**Dengan cara ini, model hanya "dihukum" jika salah memprediksi kata "Baik", dan tidak dihukum apapun terkait kata "Apa kabar". Ini memaksa model fokus 100% pada kualitas jawaban.**

### 2. Sequence Packing (Concatenation): "Anti-Pemborosan"
Masalah kedua adalah variasi panjang data.
*   Percakapan A: 100 token.
*   Percakapan B: 2000 token.
*   Context Window Model: 4096 token.

Jika kita melatih secara naif (satu percakapan per batch), Percakapan A (100 token) akan membutuhkan 3996 token padding (kosong) agar ukurannya sama dengan window model. Dampaknya adalah GPU Anda menghabiskan 95% waktunya untuk memproses angka nol (padding). Ini pemborosan listrik dan waktu yang masif.

Solusinya adalah **Sequence Packing**. Kita tidak memproses satu per satu. Kita "menjahit" beberapa percakapan pendek menjadi satu sekuens panjang hingga memenuhi batas 4096 token.
*   **Input:** [Percakapan A] [EOS] [Percakapan C] [EOS] [Awal Percakapan D]...

Dengan teknik ini, GPU selalu bekerja pada kapasitas penuh (efisiensi hampir 100%), yang bisa mempercepat waktu training hingga 3x-5x lipat dibandingkan metode standar.

---

## 3.5.4 Implementasi Kode: Formatting, Validasi & Analisis

Berikut adalah implementasi lengkap mulai dari instalasi. Kode ini menunjukkan cara mengubah data mentah menjadi format yang siap untuk SFT, serta cara melakukan validasi data (langkah yang sering dilewatkan tapi fatal).

```bash
!pip install datasets transformers matplotlib
```

Kita akan menginstall library yang dibutuhkan seperti:
*   `datasets`: Untuk manajemen data Hugging Face
*   `transformers`: Untuk tokenizer dan chat templates
*   `matplotlib`: Untuk visualisasi distribusi panjang data

**Implementasi lengkap:**

```python
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np


# 1. Load Dataset (Format Alpaca)
print("Memuat dataset...")
dataset = load_dataset("yahma/alpaca-cleaned", split="train[:1000]") # Sampel 1000 data


# 2. Load Tokenizer (Llama-3 dengan Fallback GPT-2)
model_id = "meta-llama/Meta-Llama-3-8B"


try:
    # Coba load Llama-3 (Butuh Login HF Token)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"✅ Tokenizer {model_id} berhasil dimuat.")
except:
    # Fallback ke GPT-2 jika gagal (misal karena belum login)
    print("⚠️ Gagal memuat Llama-3 (Mungkin belum login Hugging Face).")
    print("🔄 Menggunakan GPT-2 sebagai fallback untuk demo.")
   
    model_id = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
   
    # --- FIX UTAMA UNTUK GPT-2 ---
    # Kita harus mendefinisikan Chat Template manual karena GPT-2 tidak punyanya.
    # Kita gunakan format standar ChatML: <|im_start|>role\ncontent<|im_end|>\n
    tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"


# 3. Fungsi Formatting (Menerapkan Chat Template)
def format_chat_template(row):
    # Langkah A: Ubah struktur Alpaca (Input/Output) menjadi struktur Chat (User/Assistant)
    prompt = row["instruction"]
    if row["input"]: # Jika ada konteks tambahan
        prompt += "\nInput: " + row["input"]
       
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": row["output"]}
    ]
   
    # Langkah B: Terapkan template tokenizer
    # tokenize=False -> Kita ingin melihat string hasilnya dulu
    formatted_text = tokenizer.apply_chat_template(messages, tokenize=False)
   
    return {"text": formatted_text}


# Terapkan ke seluruh dataset
print("Memformat data dengan Chat Template...")
processed_dataset = dataset.map(format_chat_template)


# 4. Validasi Panjang Token (Data Profiling)
def get_token_length(row):
    return {"length": len(tokenizer(row["text"])["input_ids"])}


processed_dataset = processed_dataset.map(get_token_length)


# --- ANALISIS OUTPUT ---


# A. Inspeksi Teks (Memastikan format benar)
print("\n--- [1] Contoh Data Setelah Formatting ---")
print("Perhatikan Special Tokens yang ditambahkan otomatis:")
print("-" * 40)
print(processed_dataset[0]["text"])
print("-" * 40)


# B. Inspeksi Distribusi Panjang (Memastikan efisiensi)
lengths = processed_dataset["length"]
avg_len = np.mean(lengths)
max_len = np.max(lengths)
p95_len = np.percentile(lengths, 95)


print(f"\n--- [2] Statistik Panjang Token ---")
print(f"Rata-rata Panjang : {avg_len:.1f} token")
print(f"Panjang Maksimum  : {max_len} token")
print(f"95th Percentile   : {p95_len:.1f} token")
print("(Saran: Set 'max_seq_length' di Trainer sedikit di atas P95)")


# Visualisasi Histogram
plt.figure(figsize=(10, 6))
plt.hist(lengths, bins=30, color='skyblue', edgecolor='black')
plt.title(f"Distribusi Panjang Token (Model: {model_id})")
plt.xlabel("Jumlah Token")
plt.ylabel("Frekuensi Sampel")
plt.axvline(p95_len, color='red', linestyle='dashed', linewidth=1, label=f'95th Percentile ({p95_len:.0f})')
plt.legend()
plt.show()
```

### Bedah Logika Kode

1.  **Strategi Pemuatan Tokenizer (Blok try-except):** Bagian ini menangani skenario dunia nyata di mana kita mungkin tidak memiliki akses ke model tertutup (seperti Llama-3) tanpa token akses. Jika gagal, kita beralih ke GPT-2 dan menyuntikkan string panjang yang berisi sintaks Jinja2 sebagai `chat_template` manual agar bisa memformat percakapan layaknya model modern.
2.  **Fungsi Formatting (`format_chat_template`):** Fungsi ini menyusun ulang kolom Alpaca menjadi format standar `messages` dan menggunakan `apply_chat_template(..., tokenize=False)` untuk debugging visual.
3.  **Pemetaan Data (`dataset.map`):** Metode ini dioptimalkan untuk kecepatan dan efisiensi memori, tidak memuat seluruh data ke RAM sekaligus.
4.  **Profiling Data (Menghitung Panjang):** Kita mencari Persentil ke-95 (`np.percentile`). Jika kita bisa memotong 5% data terpanjang (*outlier*), kita bisa menghemat memori secara masif untuk 95% data lainnya.

**Output yang dihasilkan:**

```text
--- [1] Contoh Data Setelah Formatting ---
Perhatikan Special Tokens yang ditambahkan otomatis:
----------------------------------------
<|im_start|>user
Give three tips for staying healthy.<|im_end|>
<|im_start|>assistant
1. Eat a balanced and nutritious diet: Make sure your meals are inclusive of a variety of fruits and vegetables, lean protein, whole grains, and healthy fats. This helps to provide your body with the essential nutrients to function at its best and can help prevent chronic diseases.

2. Engage in regular physical activity: Exercise is crucial for maintaining strong bones, muscles, and cardiovascular health. Aim for at least 150 minutes of moderate aerobic exercise or 75 minutes of vigorous exercise each week.

3. Get enough sleep: Getting enough quality sleep is crucial for physical and mental well-being. It helps to regulate mood, improve cognitive function, and supports healthy growth and immune function. Aim for 7-9 hours of sleep each night.<|im_end|>

----------------------------------------

--- [2] Statistik Panjang Token ---
Rata-rata Panjang : 191.3 token
Panjang Maksimum  : 1069 token
95th Percentile   : 439.0 token
(Saran: Set 'max_seq_length' di Trainer sedikit di atas P95)
```

### Analisis Output

**1. Bedah Output Visual (Memahami Struktur ChatML)**
Perhatikan penggunaan **Pagar Pembatas (`<|im_start|>` & `<|im_end|>`)**. Bagi model, ini adalah *Special Tokens* yang berfungsi sebagai "pagar" yang memberi tahu model secara eksplisit di mana suara User berakhir dan di mana suara Assistant dimulai.

**2. Bedah Statistik Panjang (Keputusan Engineering Berbasis Data)**
*   **Masalah "Pencilan" (Outlier):** Panjang Maksimum adalah 1069 token, padahal Rata-rata hanya 191 token. 
*   **Keputusan:** Seorang AI Engineer yang baik akan melihat angka 95th Percentile (439). Kita bisa dengan aman men-set `max_seq_length = 512`. Kita rela memotong (*truncate*) 5% data terpanjang demi mendapatkan kecepatan pelatihan 2x-3x lipat lebih cepat dan penggunaan memori yang jauh lebih efisien.

---

### Mini Quiz
<iframe src="https://app.lumi.education/api/v1/run/SsYe8k/embed" width="1088" height="720" frameborder="0" allowfullscreen="allowfullscreen" allow="geolocation *; microphone *; camera *; midi *; encrypted-media *" style="width: 100%; height: 263px;"></iframe>

---

> **Refleksi Singkat**
>
> Kita telah membahas bahwa model Base (seperti Llama-3) sebenarnya tidak mengerti JSON; mereka hanya mengerti satu string panjang. Kita menggunakan Chat Template untuk mengubah JSON terstruktur (seperti ShareGPT) menjadi string dengan Special Tokens (misal: `<|start_header_id|>user...`).
>
> Bayangkan Anda melatih model Llama-3 menggunakan dataset percakapan yang sangat bagus, tetapi Anda lupa menerapkan Chat Template (atau menggunakan template yang salah, misal template ChatML padahal modelnya Llama). Apa yang akan terjadi pada kemampuan model tersebut saat Anda ajak chat setelah selesai training? Apakah dia akan membedakan mana suara Anda (user) dan mana suaranya sendiri (assistant), atau dia akan menganggap semuanya adalah satu teks novel yang datar?