---
slug: modul-6-4
title: modul-6-4
---
# 2.4 Teknik dan Trik Training untuk Meningkatkan Efektivitas CPT

---

Melakukan CPT bukan sekadar "melemparkan" data baru ke dalam model dan menekan tombol *train.* Jika Anda melakukan itu (misal: menggunakan *learning rate default* dan hanya data baru), Anda hampir pasti akan mengalami *catastrophic forgetting.*

CPT adalah operasi "bedah mikro". Kita harus mengubah bobot model cukup banyak agar ia mempelajari pengetahuan baru, tetapi cukup sedikit agar ia tidak merusak struktur pengetahuan lama yang berharga.

Di subtopik ini, kita akan membahas tiga pilar teknis utama untuk menjaga keseimbangan ini: strategi *Data Mixing* (pencampuran data), manajemen *Learning Rate*, dan efisiensi melalui *Sequence Packing*.

---

## 2.4.1 Strategi Data Mixing (Replay Strategy)

Teknik pertahanan utama melawan *Catastrophic Forgetting* adalah jangan pernah membiarkan model hanya melihat data baru. Apa itu sebenarnya *Data Mixing*?

***Data Mixing*** adalah teknik mencampurkan sebagian kecil data dari domain umum (*pre-training* asli) ke dalam *dataset* domain baru (CPT) selama proses pelatihan.

**Rasio Campuran:** Tidak ada aturan baku, tetapi rasio umum yang sering digunakan di industri adalah 10%-20% data lama.
*   **Dataset CPT:** 80-90% Data Baru (Misal: Jurnal Medis).
*   **Dataset Replay:** 10-20% Data Umum (Misal: Wikipedia, C4, atau dataset *pre-training* asli seperti RedPajama).

**Analogi: Menjaga Kebugaran Dasar**
Bayangkan seorang atlet lari maraton (*Base Model*) yang ingin beralih menjadi atlet angkat beban (CPT). Jika dia hanya berlatih angkat beban setiap hari, kemampuan larinya akan hilang total (ototnya kaku).

Solusinya: Dia menghabiskan 80% waktu untuk angkat beban, tapi tetap menyisihkan 20% waktu untuk *jogging* ringan. Ini memastikan tubuhnya beradaptasi dengan beban baru tanpa kehilangan stamina dasarnya.

---

## 2.4.2 Manajemen Learning Rate (LR) dan Warmup

Kesalahan pemula yang paling umum dalam CPT adalah menggunakan *learning rate* yang terlalu tinggi.

Ingat, *Base Model* Anda sudah **konvergen** (sudah selesai belajar). Bobotnya sudah berada di posisi yang sangat optimal. Jika Anda menghajarnya dengan LR tinggi, Anda akan "menendang" bobot tersebut keluar dari posisi optimalnya, menyebabkan kerusakan instan (*loss spike*).

> **Aturan Praktis untuk CPT:**
> 1.  **LR Rendah:** Gunakan *max learning rate* yang jauh lebih kecil daripada yang digunakan saat *pre-training*. Biasanya 1/10 atau 1/100 dari LR asli. (Misal: jika Pre-training LR = 3e-4, CPT LR = 3e-5).
> 2.  **Warmup Wajib:** Jangan langsung mulai dengan LR target. Gunakan **Linear Warmup** (misal, selama 100 *steps* pertama) untuk menaikkan LR dari 0 ke target secara perlahan. Ini memberi waktu bagi gradien untuk stabil sebelum perubahan besar terjadi.
> 3.  **Cosine Decay:** Turunkan LR secara perlahan kembali mendekati 0 di akhir pelatihan untuk memastikan model "mendarat" dengan mulus di titik optimal baru.

---

## 2.4.3 Efisiensi: Sequence Packing

Dataset CPT seringkali terdiri dari ribuan dokumen dengan panjang bervariasi (ada yang pendek 100 token, ada yang panjang 5000 token).

Jika kita melatih dengan *context window* 2048 token, dokumen pendek (100 token) akan membutuhkan 1948 token *padding* (`[PAD]`). Ini adalah pemborosan komputasi yang masif karena model menghabiskan waktu memproses nol. Solusinya adalah **Sequence Packing (Concatenation)**.

**Apa itu Sequence Packing?**
**Sequence Packing adalah** teknik menggabungkan beberapa dokumen pendek menjadi satu sekuens panjang hingga memenuhi context window, dipisahkan oleh token `EOS` (End of Sequence).

*   **Tanpa Packing:** `[Dokumen A] [PAD] [PAD] ...` (Boros)
*   **Dengan Packing:** `[Dokumen A] [EOS] [Dokumen B] [EOS] [Dokumen C]...` (Efisien)

Teknik ini bisa mempercepat pelatihan CPT hingga 2x-3x lipat. Nah setelah 3 teknik yang kita bahas sebelumnya, di bagian selanjutnya kita akan mengimplementasikan contoh kode dari teknik-teknik di atas.

---

## 2.4.4 Implementasi Kode Lengkap: Data Mixing & Safe CPT

Berikut adalah skrip lengkap yang mengimplementasikan strategi yang telah kita bahas:
1.  ***Data Mixing:*** Mencampur data domain baru (medis) dengan data umum (*replay*) dengan rasio tertentu.
2.  ***Safe Training:*** Menggunakan *Learning Rate* rendah dan *Cosine Schedule* untuk mencegah kerusakan bobot.

```python
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset, concatenate_datasets


# --- 1. Persiapan Data (Simulasi Strategi Data Mixing) ---


# A. Dataset Domain Baru (Target CPT - 80%)
# Anggap ini ribuan dokumen medis. Kita buat sampel kecil.
medical_texts = [
    "Pasien menunjukkan gejala hipertensi stadium 2 dengan tekanan 160/100.",
    "Terapi ACE inhibitor disarankan untuk kasus gagal jantung kongestif.",
    "Resistensi insulin adalah penanda utama diabetes tipe 2.",
    "Efek samping statin dapat mencakup nyeri otot (myalgia).",
    # ... bayangkan ada ribuan baris lagi ...
] * 100 # (Kita duplikasi agar cukup untuk demo)


# B. Dataset Umum (Replay - 20%)
# Anggap ini data Wikipedia/Internet untuk mencegah Catastrophic Forgetting.
general_texts = [
    "Ibu kota Indonesia adalah Jakarta.",
    "Matahari terbit di timur dan terbenam di barat.",
    "Python adalah bahasa pemrograman yang populer.",
    "Resep nasi goreng spesial membutuhkan kecap manis.",
    # ... bayangkan ada ribuan baris lagi ...
] * 25 # (Rasio kira-kira 1:4 atau 20%:80% dibanding medis)


# Konversi ke format Hugging Face Dataset
dataset_medical = Dataset.from_dict({"text": medical_texts})
dataset_general = Dataset.from_dict({"text": general_texts})


print(f"Jumlah Data Medis: {len(dataset_medical)}")
print(f"Jumlah Data Replay: {len(dataset_general)}")


# C. Lakukan Pencampuran (Mixing & Shuffling)
# .shuffle(seed=42) sangat PENTING agar model tidak belajar data medis dulu baru umum,
# melainkan belajar secara tercampur (interleaved).
cpt_dataset = concatenate_datasets([dataset_medical, dataset_general]).shuffle(seed=42)


print(f"Total Data CPT Siap Latih: {len(cpt_dataset)}")


# --- 2. Tokenisasi & Packing ---
model_name = "gpt2" # Model kecil untuk demo
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token


def tokenize_function(examples):
    # Di sinilah Sequence Packing biasanya diimplementasikan secara manual
    # Untuk demo sederhana, kita gunakan truncation standar
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=64)


tokenized_datasets = cpt_dataset.map(tokenize_function, batched=True, remove_columns=["text"])


# --- 3. Setup Trainer dengan "Safe Params"--- 
model = AutoModelForCausalLM.from_pretrained(model_name)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


training_args = TrainingArguments(
    output_dir="./cpt_medical_safe",
    overwrite_output_dir=True,
   
    # --- TRIK UTAMA: Learning Rate Schedule ---
    # 1. LR Rendah: 2e-5 (Jauh lebih kecil dari pre-training 3e-4)
    learning_rate=2e-5,
   
    # 2. Scheduler: Cosine (Turun perlahan agar pendaratan bobot mulus)
    lr_scheduler_type="cosine",
   
    # 3. Warmup: 10% langkah awal untuk menstabilkan gradien
    warmup_ratio=0.1,
   
    # Efisiensi
    per_device_train_batch_size=8,
    num_train_epochs=3,
    logging_steps=10,
    use_cpu=True # Ganti ke False jika ada GPU
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    data_collator=data_collator,
)


# --- 4. Eksekusi (Mock Run) ---
print("\n--- Memulai CPT dengan Data Replay & Safe LR ---")
# trainer.train() # Uncomment untuk menjalankan training nyata
print("Trainer siap. Perhatikan rasio data dan parameter LR di atas.")
```

Kode yang kita tulis di atas bukan sekadar urutan perintah, melainkan implementasi langsung dari strategi mitigasi risiko yang dibahas di teori. Berikut adalah bedah logika per baris kodenya:

1.  **Strategi Data (`concatenate_datasets` & `shuffle`)**
    *   **Rasio Data (80:20):** Kita secara manual membuat dua dataset (medical dan general). Dalam kode, kita menduplikasi data untuk mensimulasikan rasio ini. Di dunia nyata, ini krusial untuk menjaga "ingatan" model terhadap bahasa umum.
    *   **`shuffle(seed=42)`:** Ini adalah baris paling kritis. Tanpa *shuffle*, model akan belajar data medis secara berurutan, lalu data umum. Ini buruk karena gradien akan bias ke satu arah lalu berubah drastis. Dengan *shuffle*, setiap batch pelatihan (misal 8 sampel) akan berisi campuran data medis dan umum, memaksa model menyeimbangkan kedua "dunia" tersebut secara simultan.
2.  **Strategi Objektif (`DataCollatorForLanguageModeling`)**
    *   **`mlm=False`:** Parameter ini memberi tahu collator untuk tidak melakukan masking acak (seperti BERT), melainkan menyusun data untuk **Causal Language Modeling (CLM)**. Target label digeser satu posisi ke kanan. Ini memaksa model memprediksi token berikutnya, sesuai dengan arsitektur *Decoder-Only.*
3.  **Strategi Hyperparameter ("Safe CPT")**
    *   **`learning_rate=2e-5`:** Kita menggunakan nilai yang sangat kecil (biasanya 10x lebih kecil dari *pre-training*). Jika kita menggunakan LR standar (misal 1e-3 atau 3e-4), bobot model yang sudah rapi akan "meledak" keluar dari *local minima*-nya, menyebabkan kerusakan instan pada kemampuan bahasa.
    *   **`lr_scheduler_type="cosine"`:** Kurva *cosine* menurunkan LR secara perlahan mendekati 0 di akhir training. Ini memastikan model "mendarat" dengan lembut di konfigurasi bobot yang baru, mengurangi risiko *overfitting* pada data baru.
    *   **`warmup_ratio=0.1`:** Di awal CPT, gradien bisa jadi tidak stabil karena model "kaget" dengan data domain baru. *Warmup* menaikkan LR dari 0 secara perlahan, memberikan waktu bagi *optimizer* untuk beradaptasi sebelum melakukan *update* besar.

---

## Mini Quiz

[Konten Interaktif: Multiple Choice Quiz]
*(Akses kuis melalui platform LMS: https://lms.sdmdigital.id/h5p/embed.php?url=https%3A%2F%2Flms.sdmdigital.id%2Fpluginfile.php%2F635430%2Fmod_page%2Fcontent%2F13%2Fmultiple-choice-187.h5p)*

---

> **Refleksi Singkat**
>
> Kita telah membahas bahwa melakukan CPT itu ibarat "operasi bedah mikro" pada otak model. Kita harus menyuntikkan pengetahuan baru tanpa merusak yang lama.
>
> Bayangkan Anda memimpin tim AI. Anggota tim Anda mengusulkan untuk melakukan CPT pada model Llama-3 70B menggunakan 1 Terabyte data dokumen hukum internal. Dia berencana menggunakan Learning Rate standar (3e-4) dan hanya menggunakan data hukum tersebut tanpa campuran lain.
>
> Berdasarkan materi 2.2.4, bencana apa yang hampir pasti akan terjadi, dan dua instruksi spesifik apa yang akan Anda berikan padanya untuk mencegah hal tersebut?