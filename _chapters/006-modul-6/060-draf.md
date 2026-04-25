---
slug: modul-6-6
title: modul-6-6
---
# 2.6 Pendekatan CPT + Replay untuk Mengatasi Catastrophic Forgetting

---

Di Subtopik 2.3, kita belajar bahwa musuh terbesar CPT adalah *Catastrophic Forgetting*: model menjadi ahli di domain baru (misal: medis) tapi menjadi "bodoh" di domain lama (misal: logika dasar).

Di Subtopik 2.4, kita menyinggung "Data Mixing" sebagai solusi. Di subtopik ini, kita akan membedah strategi tersebut secara mendalam. Dalam literatur *Continual Learning*, teknik ini dikenal sebagai **Experience Replay.**

Konsepnya sederhana namun krusial: Agar model tidak melupakan masa lalunya, kita harus terus-menerus "mengingatkannya" (*replay*) selama ia belajar hal baru. Tanpa *Replay*, CPT pada dasarnya adalah proses perusakan model secara perlahan.

Di sini, kita akan membahas teori distribusi di baliknya, dari mana kita mendapatkan data "masa lalu" tersebut, dan teknik teknis untuk mencampurnya dalam skala Terabyte.

---

## 2.6.1 Teori Replay dan Strategi Sumber Data

Secara statistik, catastrophic forgetting terjadi karena **Pergeseran Distribusi (*Distribution Shift*).**

*   **Pre-training:** Model belajar distribusi *P(Umum)* (Data Internet).
*   **CPT Tanpa Replay:** Kita memaksa model pindah ke distribusi *P(Medis)*. Karena *P(Medis)* sangat berbeda dari *P(Umum)*, model "bermigrasi" total dan meninggalkan area pengetahuan awalnya.
*   **CPT Dengan Replay:** Kita melatih model pada campuran *P(Medis) + α P(Umum)*. Ini memaksa model untuk menemukan "titik tengah" (irisan) di mana ia bisa memahami data medis tanpa meninggalkan area pengetahuan umum.

Dari mana kita mendapatkan data "Replay" (Masa Lalu) ini? Seorang AI Engineer biasanya dihadapkan pada dua pilihan strategis:

**1. Exact Replay (Dataset Replay - Standar Industri):**
Jika kita memiliki akses ke data latih asli model (misal: RedPajama, The Pile, atau Wikipedia), kita cukup mengambil potongan acak dari sana. Ini adalah metode terbaik karena kualitas datanya terjamin. Kita tahu persis apa yang menjaga "kecerdasan" dasar model.

**2. Generative Replay (Pseudo-Rehearsal - Alternatif Cerdas):**
Bagaimana jika kita menggunakan model *open-weights* tapi tidak punya dataset aslinya? Atau dataset aslinya tertutup (*proprietary*)? Solusinya adalah *Generative Replay*. Sebelum melakukan CPT, kita minta model (yang saat itu masih pintar) untuk men-generate ribuan teks umum (esai, kode, percakapan). Kita simpan output ini sebagai "*Synthetic Dataset*" dan menggunakannya sebagai data latihan saat CPT. Model secara efektif "belajar dari dirinya sendiri di masa lalu".

---

## 2.6.2 Teknik Implementasi Skala Besar: Interleaving

Setelah kita punya Data Baru (Domain) dan Data Lama (*Replay*), bagaimana cara menggabungkannya?

Di contoh kode 2.4.4, kita menggunakan metode sederhana: *Concatenation* (gabungkan semua lalu acak).
`Dataset_Gabungan = shuffle(Dataset_A + Dataset_B)`

Metode ini bekerja baik untuk data kecil (GB). Namun, untuk CPT skala besar (Terabyte), metode ini **gagal** karena membutuhkan RAM yang sangat besar untuk memuat indeks seluruh data sebelum mengacaknya.

Solusi untuk skala besar adalah **Interleaving (Penyisipan Streaming).**

**Definisi: Interleaving**
**Interleaving adalah** teknik membaca dari beberapa stream dataset secara bersamaan (tanpa memuat semuanya ke RAM) dan mengambil sampel dari masing-masing stream secara bergantian berdasarkan probabilitas tertentu.

*   **Stream A (Medis):** Ambil... ambil... ambil... (Probabilitas 80%)
*   **Stream B (Wiki):** ...Ambil! (Probabilitas 20%)
*   **Stream A (Medis):** Ambil... ambil...

Ini memungkinkan kita melatih pada Dataset A (1TB) dan Dataset B (1TB) secara bersamaan dengan konsumsi RAM yang konstan dan rendah.

Berikut adalah implementasi praktik terbaik (*best practice*) untuk CPT skala besar menggunakan fitur `interleave_datasets` dan mode streaming dari library Hugging Face.

```python
from datasets import load_dataset, interleave_datasets


# 1. Muat Dataset dalam mode Streaming (Hemat RAM)
# Mode streaming (streaming=True) sangat penting untuk dataset besar.
# Kita tidak mendownload semua data ke disk/RAM, tapi mengalirkannya saat dibutuhkan.


# Dataset Domain Baru (Target) - Misal: C4 (Colossal Clean Crawled Corpus)
print("Menyiapkan stream data...")
ds_domain = load_dataset("allenai/c4", "en", split="train", streaming=True)


# Dataset Replay (Pengetahuan Umum) - Misal: Wikitext
ds_replay = load_dataset("wikitext", "wikitext-103-v1", split="train", streaming=True)


# 2. Tentukan Rasio Pencampuran (Probabilitas)
# Kita ingin 80% Data Domain Baru, 20% Data Replay
# Angka ini menentukan peluang pengambilan sampel dari masing-masing stream.
probabilities = [0.8, 0.2]


# 3. Lakukan Interleaving
# Fungsi ini membuat satu dataset gabungan virtual (IterableDataset).
# Ini TIDAK menggabungkan data di memori. Ia hanya membuat 'pointer'.
cpt_dataset = interleave_datasets(
    [ds_domain, ds_replay],
    probabilities=probabilities,
    seed=42
)


# 4. Verifikasi (Mari kita lihat 5 sampel pertama)
print("--- Preview Data Campuran (Interleaved) ---")
# Karena ini streaming, kita tidak bisa pakai index [0]. Kita harus pakai iterator.
iterator = iter(cpt_dataset)


for i in range(5):
    sample = next(iterator)
    # Kita cetak 50 karakter pertama untuk melihat sumbernya
    # (Anda akan melihat campuran teks dari C4 dan Wiki)
    print(f"Sampel {i+1}: {sample['text'][:60]}...")


# Dataset 'cpt_dataset' ini siap dimasukkan ke Trainer!
# Trainer akan otomatis menarik data dari stream ini tanpa batas.
```

**Penjelasan Kode:**
*   `streaming=True`: Mengubah dataset menjadi generator (*lazy loading*). Ini memungkinkan kita memproses dataset yang ukurannya jauh lebih besar daripada RAM atau Disk kita.
*   `interleave_datasets`: Ini adalah fungsi ajaibnya. Ia bertindak sebagai "DJ" yang mencampur dua lagu (dataset) secara real-time berdasarkan ketukan (probabilitas) yang kita tentukan, memastikan model selalu mendapatkan asupan nutrisi yang seimbang antara pengetahuan baru dan lama.

---

## Mini Quiz

[Konten Interaktif: Multiple Choice Quiz]
*(Akses kuis melalui platform LMS: https://lms.sdmdigital.id/h5p/embed.php?url=https%3A%2F%2Flms.sdmdigital.id%2Fpluginfile.php%2F635432%2Fmod_page%2Fcontent%2F10%2Fmultiple-choice-189.h5p)*

---

> **Refleksi Singkat**
>
> Kita telah membahas dua cara mendapatkan data replay: **Exact Replay** (menggunakan data asli seperti Wikipedia) dan **Generative Replay** (meminta model membuat data sendiri).
>
> Bayangkan Anda bekerja di perusahaan yang menggunakan model open-source seperti Llama-3 untuk CPT data perbankan yang sangat rahasia. Anda **tidak memiliki akses** ke dataset asli yang digunakan Meta untuk melatih Llama-3.
>
> Dalam situasi ini, strategi replay apa yang paling realistis untuk Anda gunakan guna mencegah model lupa cara berbahasa Inggris yang baik, dan apa potensi risiko kualitas dari strategi tersebut?