---
slug: modul-6-7
title: modul-6-7
---
# 2.7 Model Merging: Pendekatan Menggabungkan Pengetahuan

---

Sepanjang modul ini, kita berasumsi bahwa satu-satunya cara untuk membuat model lebih pintar adalah dengan **melatihnya** (baik CPT maupun SFT). Pelatihan membutuhkan data, GPU mahal, dan waktu.

Tapi, bagaimana jika kita bisa menggabungkan kemampuan dua model yang berbeda **tanpa melakukan pelatihan sama sekali**?

Bayangkan Anda memiliki dua model yang berasal dari *Base Model* yang sama (misal: Llama-3-8B):

1.  **Model A:** Telah di-SFT untuk jago *Coding* Python.
2.  **Model B:** Telah di-SFT untuk jago Bahasa Indonesia formal.

Anda ingin model yang jago *Coding* DALAM Bahasa Indonesia. Cara konvensional adalah melatih ulang. Cara modern adalah ***Model Merging***.

Teknik ini memungkinkan kita mengambil bobot ($W$) dari Model A dan Model B, lalu menggabungkannya secara matematis menjadi Model C. Hasilnya seringkali mengejutkan: Model C bisa memiliki kemampuan A dan B sekaligus, tanpa pernah melihat data latih keduanya.

Di subtopik penutup ini, kita akan membahas intuisi di balik ini, teknik penggabungannya, dan cara melakukannya dalam kode.

---

## 2.7.1 Konsep: Aritmatika Vektor Tugas (*Task Arithmetic*)

Kita bisa membayangkan perubahan bobot akibat pelatihan sebagai sebuah **Vektor Tugas (*Task Vector*).**

*   $W_{base}$ = Bobot Llama 3 Asli
*   $W_{coding}$ = Bobot Llama 3 setelah belajar Coding
*   **Vektor Skill Coding** $(\tau_A) = W_{coding} - W_{base}$

Jika kita punya model Bahasa $W_{bahasa}$, kita bisa mencoba menggabungkan *skill*-nya dengan rumus sederhana:

> $W_{merged} = W_{base} + \lambda_A(\tau_A) + \lambda_B(\tau_B)$

Ini disebut **Task Arithmetic.** Kita secara harfiah "menjumlahkan" *skill coding* dan skill bahasa ke dalam model dasar.

Konsep penjumlahan vektor ini memberikan landasan teoretis yang kuat bahwa penggabungan pengetahuan itu mungkin. Namun, implementasi praktisnya tidak selalu sesederhana aritmatika dasar "A + B". Seringkali, parameter dari dua model mengalami konflik atau interferensi destruktif yang justru merusak kinerja gabungan. Masalah inilah yang mendorong para peneliti untuk mengembangkan algoritma penggabungan yang lebih canggih daripada sekadar rata-rata biasa.

---

## 2.7.2 Teknik Merging: Linear vs. SLERP vs. TIES

Ada banyak cara untuk "menjumlahkan" model. Berikut adalah evolusinya:

1.  **Linear / Spherical Interpolation (SLERP):**
    *   Ini adalah rata-rata tertimbang sederhana.
    *   Rumus: $W_{merged} = 0.5 \times W_A + 0.5 \times W_B$
    *   **Kelemahan:** Seringkali menghasilkan model yang "bingung" atau mengalami degradasi performa karena interferensi antar bobot.

2.  **TIES-Merging (*Trim, Elect Sign, & Merge*):**
    *   Teknik ini menyadari bahwa saat kita menggabungkan model, banyak parameter yang sebenarnya "bertabrakan" (interferensi).
    *   **Solusi TIES:** Ia membuang perubahan parameter yang kecil (*Trim*), menyelesaikan konflik arah (*Elect Sign*), dan baru menggabungkannya. Hasilnya jauh lebih stabil dan tajam.

3.  **DARE (*Drop And REscale*):**
    *   Teknik ekstrem yang secara acak membuang (drop) sebagian besar perubahan bobot (misal 90%) dari model *fine-tuned*, lalu men-skalakan sisanya.
    *   Ajaibnya, ini sering bekerja sangat baik untuk menggabungkan banyak model sekaligus tanpa merusak otak model.

Algoritma-algoritma di atas menawarkan berbagai strategi untuk mengatasi konflik parameter, dari yang sederhana hingga yang ekstrem. Meskipun saat ini sudah tersedia tools otomatis (seperti `mergekit`) yang menangani kompleksitas ini, sebagai engineer kita perlu memahami apa yang sebenarnya terjadi di balik layar.

Mari kita bedah implementasi low-level dari operasi "penjumlahan otak" ini secara langsung menggunakan kode Python. Meskipun ada *tools* canggih seperti mergekit (berbasis konfigurasi YAML), penting bagi *engineer* untuk memahami bahwa *merging* pada dasarnya hanyalah operasi matematika pada *tensor*.

Berikut adalah skrip Python menggunakan PyTorch untuk melakukan **Linear Merge** (rata-rata) dari dua model Llama-3.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# --- Konfigurasi ---
base_model_name = "meta-llama/Meta-Llama-3-8B"
model_a_name = "model_coding_v1"   # Anggap ini model yang sudah di-finetune coding
model_b_name = "model_chat_indo_v1" # Anggap ini model yang sudah di-finetune Indo


print("Sedang memuat model (Ini butuh RAM besar)...")
# Catatan: Di dunia nyata, kita load satu per satu untuk hemat RAM
# model_a = AutoModelForCausalLM.from_pretrained(model_a_name, torch_dtype=torch.float16)
# model_b = AutoModelForCausalLM.from_pretrained(model_b_name, torch_dtype=torch.float16)


# --- Simulasi Logika Merging (Weighted Averaging) ---
# alpha = 0.5 berarti 50% Model A + 50% Model B
alpha = 0.5


# Kita akan iterasi setiap parameter (weights)
# state_dict adalah dictionary berisi semua tensor bobot model
# merged_state_dict = {}


# for key in model_a.state_dict():
#     # Ambil tensor dari kedua model
#     tensor_a = model_a.state_dict()[key]
#     tensor_b = model_b.state_dict()[key]
   
#     # Rumus Linear Interpolation:
#     # W_new = (1 - alpha) * W_a + alpha * W_b
#     merged_tensor = (1 - alpha) * tensor_a + alpha * tensor_b
   
#     merged_state_dict[key] = merged_tensor


# --- Menyimpan Model Hasil Merge ---
# model_merged = AutoModelForCausalLM.from_config(model_a.config)
# model_merged.load_state_dict(merged_state_dict)
# model_merged.save_pretrained("./llama-3-coding-indo-merged")


print("Logika Merging Selesai.")
print("Inti kodenya adalah: merged_tensor = (0.5 * tensor_a) + (0.5 * tensor_b)")
```

**Catatan Praktis:** Di industri, kita jarang menulis loop manual seperti di atas. Kita menggunakan `mergekit`, sebuah library populer yang mengoptimalkan penggunaan RAM (memuat bobot sepotong-sepotong) dan menyediakan algoritma canggih (TIES, DARE) hanya dengan file konfigurasi YAML sederhana.

---

## Mini Quiz

[Konten Interaktif: Multiple Choice Quiz]
*(Akses kuis melalui platform LMS: https://lms.sdmdigital.id/h5p/embed.php?url=https%3A%2F%2Flms.sdmdigital.id%2Fpluginfile.php%2F635433%2Fmod_page%2Fcontent%2F21%2Fmultiple-choice-190.h5p)*

---

> **Refleksi Singkat**
>
> Model Merging memungkinkan kita menggabungkan kemampuan tanpa pelatihan.
>
> Jika Anda berhasil menggabungkan "Model Coding" dan "Model Bahasa Indonesia" menjadi satu model baru, apa keuntungan terbesar pendekatan ini dari sisi **Operational Cost (Ops)** dibandingkan dengan men-deploy dua model terpisah dan membuat sistem routing?