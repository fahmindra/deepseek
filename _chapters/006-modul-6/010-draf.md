---
slug: modul-6-1
title: modul-6-1
---
# 2.1 Perbedaan Continued Pretraining (CPT) dengan Fine-Tuning

---

Pada Modul 1 (Intermediate), fokus pembahasan utama kita adalah **Supervised Fine-Tuning (SFT)** dan **Alignment**. Kita telah mempelajari metode adaptasi *Base Model* yang mana arsitekturnya dibahas di Modul 2 Fundamental untuk meningkatkan kapabilitas *instruction following*. Secara teknis, SFT bertujuan untuk menyelaraskan model agar mampu menghasilkan output yang presisi sesuai dengan format instruksi, pola percakapan (*chat*), dan preferensi pengguna yang ditargetkan.

Namun, sebagai AI Engineer, kita akan menghadapi masalah yang berbeda:
* "Model saya sangat sopan, tapi ia tidak tahu database produk internal perusahaan saya."
* "Model saya pengetahuannya terpotong di tahun 2023, tapi saya butuh dia menganalisis berita tahun 2024."

Ini bukan masalah perilaku, namun ini adalah masalah pengetahuan. Mencoba menyelesaikan masalah pengetahuan dengan alat untuk perilaku (SFT) seringkali tidak efisien dan bisa gagal. Inilah perbedaan fundamental yang harus kita pahami: SFT untuk Instruksi dan Spesialisasi, CPT untuk Pengetahuan Umum dan domain spesifik.

Di subtopik ini, kita akan membedah kedua pendekatan ini, memahami *trade-off* biayanya, dan yang terpenting, memahami "perangkap" umum yaitu mengapa CPT yang mahal seringkali merupakan pendekatan yang salah.

---

## 2.1.1 Definisi: Supervised Fine-Tuning (SFT)

**Apa itu Supervised Fine-Tuning (SFT)?**

**SFT adalah** proses melatih Base Model dengan menggunakan dataset yang lebih kecil dengan format `[instruction, output]` yang bertujuan untuk mengajarkan model mengenai keterampilan (skill) atau perilaku (behavior) baru.

**Analogi:** "Sekolah Kepatuhan" atau "Pelatihan Kerja". Kita tidak mengajari model fakta baru (dia sudah "lulus S1"). Kita mengajarinya cara menggunakan pengetahuannya untuk pekerjaan spesifik (misal: "Jika saya beri email, buatkan ringkasannya dalam 3 poin").

**Bagaimana SFT Bekerja (Secara Teknis):** SFT adalah proses yang sangat efisien karena menggunakan Masked Loss Function.
* Bayangkan data SFT kita: `Prompt: "Apa ibu kota Indonesia?" Respons: "Ibu kota Indonesia adalah Jakarta."`
* Saat pelatihan, kita menggabungkan keduanya menjadi satu sekuens.
* Kita hanya meminta model untuk belajar memprediksi token-token di bagian Respons.
* *Loss function* (sinyal "kesalahan") untuk token-token di bagian Prompt akan **dimasking (ditutup/diabaikan)**.
* **Mengapa?** Karena kita tidak peduli jika model tidak bisa "menebak" prompt kita. Kita hanya peduli ia belajar bagaimana cara menjawab (respons) ketika diberi prompt tersebut.

Karena model hanya belajar dari beberapa ribu token respons (bukan triliunan token teks mentah), SFT secara komputasi **jauh lebih murah** dan cepat daripada CPT. SFT lebih banyak melatih lapisan attention untuk belajar "format" dan lapisan output untuk belajar "kata-kata" yang harus dijawab, dan tidak terlalu "mengganggu" pengetahuan faktual yang tersimpan di lapisan FFN (Modul 2 Fundamental).

---

## 2.1.2 Definisi: Continued Pretraining (CPT)

Berbeda dengan SFT yang berfokus pada penyesuaian spesifik, CPT adalah proses adaptasi yang lebih mendasar dan berskala besar.

**Definisi: Continued Pretraining (CPT)**
CPT adalah proses mengambil Base Model yang sudah ada dan **melanjutkan fase Pre-training-nya** (proses yang sama di Modul 2 dan 4 Fundamental) pada dataset teks mentah baru yang masif.

**Analogi:** "Kuliah Lagi (Ambil S2)". Kita mengambil lulusan S1 (Base Model) dan mengirimnya kembali ke universitas untuk "menghafal" seluruh buku teks baru (misal, 100GB data jurnal medis). Tujuannya adalah **meng-injeksi pengetahuan faktual baru**.

**Bagaimana CPT Bekerja (Secara Teknis):** CPT tidak menggunakan Masked Loss Function khusus. Ia menggunakan objektif asli dari pre-training: **Causal Language Model (CLM).**
* Model diberi data mentah (misal: "...adalah lapisan neural network yang bertindak sebagai jembatan...").
* Model dilatih untuk memprediksi **SETIAP TOKEN BERIKUTNYA** dalam data tersebut. Loss dihitung pada setiap token di seluruh 100GB data baru.
* **Mengapa?** Karena kita ingin model menghafal dan menginternalisasi pola dan fakta baru dari data mentah tersebut.
* Proses ini "memaksa" semua parameter di model untuk beradaptasi. Ini secara khusus berdampak besar pada **lapisan FFN (Feed-Forward Network)** di arsitektur Transformer (Modul 2 Fundamental). Para peneliti percaya bahwa lapisan FFN inilah yang bertindak sebagai "memori" atau database (penyimpanan pengetahuan faktual) model.

Karena kita melatih setiap token pada dataset yang masif (Gigabyte atau Terabyte), CPT adalah proses yang **sangat mahal dan lambat**, hampir semahal melatih model dari nol. Lalu sebenarnya apa saja perbedaan dari CPT dan SFT? Inilah yang akan kita bahas selanjutnya.

---

## 2.1.3 Perbandingan dan Perangkap CPT

Tabel 1 di bawah ini merangkum perbedaan strategis antara kedua pendekatan.

**Tabel 1. Perbandingan Strategis SFT vs. CPT**

| Fitur | Supervised Fine-Tuning | Continued Pretraining |
| :--- | :--- | :--- |
| **Tujuan Utama** | **Mengarahkan** model agar **menghasilkan output sesuai contoh yang diberi dalam dataset instruksi** | Menambah **pengetahuan/domain spesifik** |
| **Dataset** | Berlabel (Instruksi dan Output) | Teks Mentah |
| **Fokus Data** | Kualitas Tinggi (Beberapa ribu contoh) | Kuantitas Besar (Banyak GB/TB) |
| **Objektif** | Belajar memprediksi Response | Belajar memprediksi Next Token (CLM) |
| **Analogi** | "Sekolah Kepatuhan" | "Kuliah Lagi (S2)" |
| **Contoh Kasus** | Membuat chatbot yang sopan | Membuat model paham istilah medis |
| **Biaya Compute** | Relatif Murah (Bisa di 1 GPU) | Sangat Mahal |

### Analisis Efektivitas: Mengapa CPT Seringkali Bukan Pendekatan yang Tepat

Penting bagi seorang AI Engineer untuk memahami bahwa CPT bukanlah solusi universal. Penerapan CPT pada kasus penggunaan yang salah dapat mengakibatkan inefisiensi biaya komputasi yang signifikan tanpa peningkatan kinerja yang sebanding. Seringkali, kebutuhan bisnis sebenarnya dapat diselesaikan dengan metode yang lebih efisien.

**Contoh Konkret: Studi Kasus Dokumen Internal**

Bayangkan Anda memiliki 1.000 dokumen PDF tentang Standard Operating Procedure (SOP) perusahaan. Tujuan Anda adalah membuat chatbot yang bisa menjawab pertanyaan karyawan tentang SOP tersebut secara akurat.

1.  **Pendekatan Kurang Tepat (Menggunakan CPT):** Jika Anda melakukan CPT pada data ini, Anda "memaksa" model untuk menghafal statistik kata dalam dokumen SOP.
    *   **Risiko:** Model mungkin tetap berhalusinasi saat menjawab fakta spesifik karena CPT lebih fokus pada pola bahasa daripada akurasi pengambilan fakta (*retrieval*). Selain itu, biaya pelatihannya sangat tinggi.
2.  **Pendekatan Lebih Efektif (SFT atau RAG):**
    *   **SFT:** Jika tujuannya adalah agar model paham cara menjawab pertanyaan seputar SOP, lebih efektif mengubah dokumen tersebut menjadi 5.000 pasangan Tanya-Jawab (Q&A) dan melakukan SFT. Model belajar *keterampilan* menjawab tanpa harus memproses seluruh teks mentah.
    *   **RAG (Retrieval-Augmented Generation):** Untuk akurasi fakta tertinggi, solusi terbaik seringkali bukan pelatihan (*training*), melainkan pencarian (*retrieval*). Simpan dokumen dalam database, ambil bagian yang relevan saat ada pertanyaan, dan biarkan model menjawab berdasarkan konteks tersebut. Ini jauh lebih murah dan akurat daripada CPT.

**Kesimpulan Strategis:** CPT sebaiknya hanya dipilih jika tujuannya adalah mengubah pemahaman mendasar model terhadap bahasa atau domain (misalnya: mengajarkan Bahasa Jawa ke model Inggris, atau mengajarkan sintaks bahasa pemrograman baru), bukan sekadar untuk menghafal dokumen perusahaan.

---

## 2.1.4 Implementasi Kode: CPT vs. SFT

Perbedaan konseptual terbaik terlihat jelas dalam kode. Snippet code di bawah ini menyediakan **dua skrip Python lengkap yang bisa dijalankan (misal, di Google Colab)**. Kita akan menggunakan model kecil (gpt2) untuk mendemonstrasikan perbedaan setup utama.

**Tahap 1: Instalasi Library**

```bash
pip install transformers datasets trl peft torch
```

Sebelum memulai CPT dan SFT, kita akan menginstall library yang dibutuhkan terlebih dahulu yaitu sebagai berikut:
* `transformers`: Untuk model dan trainer
* `datasets`: Untuk menangani data training
* `trl`: Untuk pengaturan SFTTrainer yang mudah
* `peft`: Untuk adaptor training model

Selanjutnya adalah Tahap 2 yaitu Implementasi CPT dan SFT. Kita akan mulai dengan CPT terlebih dahulu.

**Implementasi CPT (Continued Pretraining)**

```python
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset


# --- 1. Model & Tokenizer ---
print("Memuat base model (gpt2) untuk CPT...")
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


# Tambahkan Pad Token (GPT-2 tidak punya default pad token)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# --- 2. Data CPT (Teks Mentah) ---
# Data domain baru (Fiksi Ilmiah)
raw_text_data = [
    "Kapal luar angkasa Hyperion melompat ke warp speed.",
    "Federasi Galaksi menolak perjanjian damai dengan bangsa Xylar.",
    "Teknologi 'jump drive' baru memungkinkan perjalanan instan.",
    "Di planet Kepler-186f, lautan metana membentang luas.",
    "AI pemberontak mengambil alih koloni Mars."
]
cpt_dataset = Dataset.from_dict({"text": raw_text_data})


def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=128, padding="max_length")


tokenized_dataset = cpt_dataset.map(tokenize_function, batched=True, remove_columns=["text"])


# --- 3. Setup Trainer CPT ---
# Data Collator khusus untuk CLM
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)


training_args = TrainingArguments(
    output_dir="./results_cpt",
    overwrite_output_dir=True,
    num_train_epochs=30, # Cukup untuk overfitting data kecil ini
    per_device_train_batch_size=4,
    logging_steps=5,
    learning_rate=2e-4,
    use_cpu=False if torch.cuda.is_available() else True
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)


# --- 4. Mulai Training CPT ---
print("\n--- Memulai CPT (Melatih Pengetahuan Fiksi Ilmiah) ---")
trainer.train()
print("Training Selesai!")


# --- 5. Bukti Output (Inference) ---
print("\n--- Tes Pengetahuan Baru ---")
# Kita tes apakah model bisa melengkapi kalimat tentang 'Kapal luar angkasa'
# Sebelum training, GPT-2 mungkin bicara soal NASA. Setelah ini, harusnya soal Hyperion/Warp.
test_prompt = "Kapal luar angkasa"
inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)


output_ids = model.generate(
    **inputs,
    max_new_tokens=10,
    pad_token_id=tokenizer.eos_token_id
)
print(f"Prompt: {test_prompt}")
print(f"Output: {tokenizer.decode(output_ids[0], skip_special_tokens=True)}")
```

**Penjelasan Kode (CPT):**
Kode ini mendemonstrasikan CPT (Pengetahuan). Perhatikan tiga poin kunci teknis:
1.  **Data (`raw_text_data`):** Inputnya adalah teks mentah. Kita tidak peduli format Q&A, kita hanya ingin model "membaca" dan "menghafal" pengetahuan baru ini.
2.  **`DataCollatorForLanguageModeling`:** Ini adalah komponen krusial yang menangani persiapan batch secara otomatis. Tugas teknisnya meliputi:
    *   Padding Dinamis: Menyamakan panjang sequence dalam satu batch dengan menambahkan pad token pada data yang lebih pendek.
    *   Attention Mask: Membuat mask agar model mengabaikan pad token tersebut saat perhitungan self-attention.
    *   Label Creation: Membuat salinan `input_ids` ke dalam kolom labels. Dalam pelatihan standar, token padding pada labels akan diberi nilai -100 agar tidak dihitung dalam loss.
3.  **`mlm=False` (Causal Language Modeling):** Parameter ini menginstruksikan bahwa kita melakukan pelatihan gaya GPT (autoregressive), bukan gaya BERT.
    *   **Mekanisme Shift:** Dalam proses pelatihan ini, model menggunakan pendekatan **pergeseran label (*label shifting*)**. Input token pada posisi t digunakan untuk memprediksi label pada posisi t+1.
    *   Secara internal, model akan menggeser labels satu posisi ke kanan dibandingkan `input_ids` untuk menghitung loss prediksi token berikutnya. Inilah yang memaksa model menyerap "pengetahuan" urutan kata dari data fiksi ilmiah tersebut.

**Output dari kode CPT di atas:**
```text
--- Tes Pengetahuan Baru ---
Prompt: Kapal luar angkasa
Output: Kapal luar angkasa Hyperion melompat ke warp speed.
```

Perhatikan hasil tersebut.
*   **Sebelum CPT:** Jika kita memberi *prompt* "Kapal luar angkasa" ke model GPT-2 standar, ia kemungkinan besar akan melengkapinya dengan teks umum (misal: "...adalah kendaraan antariksa buatan NASA...").
*   **Sesudah CPT:** Model langsung melengkapinya dengan kata kunci spesifik "Hyperion" dan "warp speed".

Ini adalah bukti teknis bahwa proses *training* berhasil. Model telah menggeser distribusi probabilitasnya untuk memprioritaskan pengetahuan baru yang kita berikan dalam dataset fiksi ilmiah tersebut. Meskipun pada demo skala kecil ini model cenderung "menghafal" (*overfitting*), mekanisme yang sama inilah yang digunakan pada skala besar untuk menyuntikkan pengetahuan domain (seperti medis atau hukum) ke dalam LLM.

Selanjutnya untuk **implementasi SFT (Supervised Fine-Tuning):**

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from datasets import Dataset


# --- 1. Model & Tokenizer ---
print("\nMemuat base model (gpt2) baru yang 'polos' untuk SFT...")
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id


# --- 2. Data SFT (Format Instruksi) ---
chat_data = [
    {"role": "user", "content": "Hai, kamu siapa?"},
    {"role": "assistant", "content": "Saya adalah AI yang membantu."},
    {"role": "user", "content": "Apa ibu kota Indonesia?"},
    {"role": "assistant", "content": "Ibu kota Indonesia adalah Jakarta."},
    {"role": "user", "content": "2 + 2 = ?"},
    {"role": "assistant", "content": "Hasilnya adalah 4."},
]
sft_dataset = Dataset.from_dict({"messages": [chat_data] * 10})


# --- 3. Setup Trainer SFT ---
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=sft_dataset,
    dataset_field="messages",
    max_seq_length=128,
    args=TrainingArguments(
        output_dir="./results_sft",
        overwrite_output_dir=True,
        num_train_epochs=50,
        per_device_train_batch_size=1,
        logging_steps=10,
    ),
)


# --- 4. Mulai Training SFT ---
print("--- Memulai SFT (Melatih Perilaku Chatbot) ---")
# trainer.train() # Hapus komentar untuk menjalankan
print("Setup SFT selesai. Fokus pada 'response' (loss dimasker).")
```

**Penjelasan Kode (SFT):** Kode ini mendemonstrasikan SFT (Perilaku). Perbedaannya sangat jelas:
1.  **Data (`chat_data`):** Inputnya sangat terstruktur, mengikuti format instruksi/percakapan (Q&A).
2.  **SFTTrainer (dari trl):** Kita menggunakan *trainer* khusus. SFTTrainer secara otomatis menangani *formatting* data *chat* ini.
3.  **Masked Loss (Implisit):** SFTTrainer menerapkan strategi cerdas pada fungsi loss.
    *   Token pada bagian **User/Instruksi** akan **di-masking (diabaikan)** dari perhitungan error. Model tidak dihukum jika salah menebak pertanyaan.
    *   Token pada bagian **Assistant/Respons** akan dihitung.
    *   **Tujuannya:** Kita ingin model fokus 100% belajar **cara menjawab** (menghasilkan respons), bukan membuang kapasitas otaknya untuk menghafal pertanyaan.

Kedua *snippet* kode ini secara praktis mendemonstrasikan perbedaan inti: CPT (menggunakan `DataCollatorForLanguageModeling`) melatih **pengetahuan** pada semua token, sementara SFT (menggunakan `SFTTrainer`) melatih **perilaku** pada token respons.

Memahami perbedaan fundamental dalam setup ini adalah kunci sebelum kita membahas trade-off biaya (di 2.2) dan risiko (di 2.3) dari penerapan CPT.

---

## Mini Quiz

[Konten Interaktif: Multiple Choice Quiz]
*(Akses kuis melalui platform LMS: https://lms.sdmdigital.id/h5p/embed.php?url=https%3A%2F%2Flms.sdmdigital.id%2Fpluginfile.php%2F635427%2Fmod_page%2Fcontent%2F23%2Fmultiple-choice-184.h5p)*

---

> **Refleksi Singkat**
>
> Kita telah melihat perbedaan teknis dan konseptual antara CPT (Pengetahuan) dan SFT (Perilaku). Kita juga telah membahas "Perangkap CPT" dan alternatif yang lebih efisien seperti RAG (Retrieval-Augmented Generation).
>
> Sekarang, coba refleksikan: Bayangkan Anda memiliki 10GB dokumen medis internal perusahaan farmasi. Tujuan akhirnya adalah membuat chatbot yang dapat menjawab pertanyaan dokter tentang obat-obatan spesifik.
>
> Apakah Anda akan menggunakan CPT saja? SFT saja? Atau kombinasi CPT dan SFT? Atau RAG? Apa trade-off utama dari pilihan Anda?