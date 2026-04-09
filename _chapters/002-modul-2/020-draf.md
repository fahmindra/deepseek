---
slug: modul-2-2
title: modul-2-2
---

### 2.2 Pemodelan Bahasa Menggunakan N-gram dan Kneser-Ney Smoothing
---

Pada Submodul 2.1, kita telah mendefinisikan masalah inti pemodelan bahasa: bagaimana cara mengestimasi probabilitas $P(W)$ dari sebuah sekuens? Kita juga telah melihat bahwa aturan rantai (*Chain Rule*) secara teoritis memecah masalah ini menjadi $P(w_n \mid w_1, \dots, w_{n-1})$, atau "probabilitas kata berikutnya berdasarkan seluruh histori sebelumnya."

Namun, kita juga telah mengidentifikasi tantangan terbesarnya mengenai **sparsitas data**. Seperti yang diilustrasikan oleh Gambar 3, ruang dari semua kalimat yang mungkin jauh lebih masif daripada data yang bisa kita kumpulkan. Secara praktis, hampir mustahil kita menemukan konteks histori $w_1, \dots, w_{n-1}$ yang panjang dan spesifik di dalam data latih.

Untuk membuat masalah ini dapat dipecahkan secara komputasi, kita harus membuat sebuah **asumsi penyederhanaan** yang kuat. Bagaimana jika, alih-alih melihat seluruh histori, kita berasumsi bahwa probabilitas sebuah kata hanya bergantung pada $n$ kata sebelumnya yang paling baru?

Inilah ide fundamental di balik model n-gram. Model n-gram adalah pendekatan statistik klasik pertama yang sangat sukses dalam mengaproksimasi $P(W)$. Selama beberapa dekade sebelum era deep learning, model-model ini adalah teknologi andalan yang digunakan dalam aplikasi speech recognition, machine translation, dan spell checking.

Meskipun model n-gram saat ini telah banyak digantikan oleh arsitektur neural network yang lebih kuat (akan dibahas pada bagian 2.3), memahami n-gram tetap relevan karena konsep dasarnya menjadi fondasi bagi teknik pemodelan bahasa yang lebih modern. Model n-gram memberi kita fondasi penting untuk memahami konsep inti seperti **konteks, sparsitas,** dan **smoothing,** yang tetap menjadi tantangan relevan bahkan di era LLM modern.

---

#### 2.2.1 Model n-gram dan Asumsi Markov

Seperti yang telah dibahas, mengestimasi $P(w_n \mid w_1, \dots, w_{n-1})$ (probabilitas kata $w_i$ berdasarkan seluruh histori) sangat sulit karena sparsitas membuat banyak urutan sangat jarang muncul sehingga probabilitasnya mendekati nol. Solusi dari model n-gram adalah dengan menerapkan asumsi Markov.

> **Apa itu Asumsi Markov?**
> Asumsi Markov adalah sebuah asumsi penyederhanaan yang menyatakan bahwa probabilitas sebuah kejadian (kata) di masa depan hanya bergantung pada sejumlah terbatas $n$ kejadian (kata) sebelumnya yang paling baru, bukan pada seluruh histori.

Berdasarkan asumsi ini, kita mengaproksimasi probabilitas konteks penuh:

$$P(w_n \mid w_1, \dots, w_{n-1}) \approx P(w_n \mid w_{n-1})$$

Persamaan tersebut dapat dilihat di [Buku Jurafsky](https://web.stanford.edu/~jurafsky/slp3/ed3book_aug25.pdf) pada Chapter 3 halaman 40. Model yang menggunakan asumsi ini disebut **model n-gram.** Nilai $n$ menentukan seberapa banyak konteks yang kita "ingat".

**Analogi**
Model n-gram bekerja seperti seorang analis yang memiliki "memori jangka pendek" yang terbatas.
*   **Unigram $(n=1)$:** Tidak memiliki memori sama sekali. Ia hanya tahu frekuensi kata secara umum $(P(w_i))$.
*   **Bigram $(n=2)$:** Hanya mengingat 1 kata sebelumnya $(P(w_i \mid w_{i-1}))$.
*   **Trigram $(n=3)$:** Mengingat 2 kata sebelumnya $(P(w_i \mid w_{i-2}, w_{i-1}))$.

Semakin tinggi $n$, model semakin kontekstual, tetapi juga semakin rentan terhadap masalah sparsitas (karena konteks $n-1$ kata menjadi semakin spesifik dan langka).

**Tabel 3. Perbandingan model n-gram secara umum.**

| Model (Nilai n) | Asumsi | Contoh Aproksimasi untuk $P(\text{"nasi"} \mid \text{"saya"}, \text{"makan"})$ |
| :--- | :--- | :--- |
| Unigram ($n=1$) | $P(w_i)$ | $P(\text{"nasi"})$ |
| Bigram ($n=2$) | $P(w_i \mid w_{i-1})$ | $P(\text{"nasi"} \mid \text{"makan"})$ |
| Trigram ($n=3$) | $P(w_i \mid w_{i-2}, w_{i-1})$ | $P(\text{"nasi"} \mid \text{"saya"}, \text{"makan"})$ |

Tabel 3 di atas mengilustrasikan secara jelas bagaimana asumsi Markov diterapkan dalam praktiknya. Perhatikan bahwa masalah intinya selalu sama. Namun, setiap model n-gram secara drastis menyederhanakan masalah tersebut dengan "memotong" konteks yang dilihatnya:
*   **Unigram** mengabaikan semua konteks dan hanya bertanya, "Seberapa sering 'nasi' muncul secara umum?"
*   **Bigram** hanya melihat satu kata sebelumnya dan bertanya, "Seberapa sering 'nasi' muncul setelah 'makan'?"
*   **Trigram** melihat dua kata sebelumnya dan bertanya, "Seberapa sering 'nasi' muncul setelah 'saya makan'?"

Dengan mengubah masalah dari "histori tak terbatas" menjadi "konteks $n-1$ kata", kita sekarang memiliki probabilitas yang lebih sederhana.

> **Bagaimana kita menghitung probabilitas ini?**
> Kita menggunakan metode yang disebut Maximum Likelihood Estimation (MLE), yang secara sederhana berarti "menghitung frekuensi relatif" dari data latih (korpus).

Untuk model Bigram, rumusnya adalah:

$$P(w_n \mid w_{n-1}) = \frac{C(w_{n-1} w_n)}{C(w_{n-1})}$$

Keterangan rumus:
*   $C(w_{n-1}, w_n)$ adalah jumlah kemunculan sekuens bigram (misal: "makan nasi") di korpus.
*   $C(w_{n-1})$ adalah jumlah total kemunculan unigram konteks (misal: "makan") di korpus.

**Contoh untuk Perhitungan MLE:**
Misalkan kita memiliki korpus mini sbb.:
1. `<s> saya makan nasi </s>`
2. `<s> saya makan roti </s>`
3. `<s> dia makan nasi </s>`
*(Catatan: `<s>` adalah token "mulai kalimat" dan `</s>` adalah token "akhir kalimat")*

Mari kita hitung probabilitas $P(\{\text{"nasi"}\} \mid \{\text{"makan"}\})$:
*   $C(\{\text{"makan nasi"}\}) = 2$ kali muncul.
*   $C(\{\text{"makan"}\}) = 3$ kali muncul.
*   $P(\{\text{"nasi"}\} \mid \{\text{"makan"}\}) = 2/3 \approx 0.67$

Sedangkan untuk perhitungan $P(\{\text{"roti"}\} \mid \{\text{"makan"}\})$:
*   $C(\{\text{"makan roti"}\}) = 1$ kali muncul.
*   $C(\{\text{"makan"}\}) = 3$ kali muncul.
*   $P(\{\text{"roti"}\} \mid \{\text{"makan"}\}) = 1/3 \approx 0.33$

Model menyimpulkan bahwa jika seseorang 'makan', 67% kemungkinannya adalah 'nasi' dan 33% kemungkinannya adalah 'roti'. Pendekatan MLE ini logis selama n-gram pernah muncul di data latih. Namun, apa yang terjadi ketika kita menanyakan $P(\text{"ayam"} \mid \text{"makan"})$ yang tidak ada di korpus? Formula MLE akan menghasilkan probabilitas 0.

---

#### 2.2.2 Masalah Probabilitas Nol dan Add-1 (Laplace) Smoothing

Masalah Probabilitas Nol (*Zero Probability Problem*) terjadi ketika model menganggap frasa yang tidak terlihat di korpus sebagai "mustahil". Jika satu saja komponen n-gram memiliki probabilitas 0, maka probabilitas seluruh kalimat akan menjadi 0. Untuk mengatasi ini, kita memerlukan teknik **smoothing.**

> **Apa itu smoothing?**
> Smoothing adalah sekumpulan teknik statistik yang digunakan untuk "mencadangkan" dan mendistribusikan ulang sebagian kecil dari total massa probabilitas kepada n-gram yang tidak terlihat (*unseen*).

**Add-1 (Laplace) Smoothing**
Teknik ini menambahkan satu hitungan palsu ke setiap n-gram yang mungkin ada di vokabulari. Formula MLE diubah menjadi:

$$P_{Laplace}(w_n \mid w_{n-1}) = \frac{C(w_{n-1} w_n) + 1}{C(w_{n-1}) + V}$$

Di mana $V$ adalah ukuran vokabulari (jumlah total kata unik).

**Contoh Perhitungan Add-1:**
Misalkan Vokabulari kita ($V$) adalah 1000 dan menggunakan korpus sebelumnya:
*   $P(\{\text{"nasi"}\} \mid \{\text{"makan"}\}) = \frac{2 + 1}{3 + 1000} = \frac{3}{1003} \approx 0.003$
*   $P(\{\text{"ayam"}\} \mid \{\text{"makan"}\}) = \frac{0 + 1}{3 + 1000} = \frac{1}{1003} \approx 0.001$

Sekarang, n-gram yang *unseen* memiliki probabilitas non-nol. Namun, Add-1 Smoothing melakukan distorsi besar: n-gram yang terlihat ("nasi") anjlok probabilitasnya dari 67% menjadi 0.3% karena massa probabilitas habis dibagikan ke ribuan kata *unseen* lainnya.

---

#### 2.2.3 Konsep Advanced Smoothing: Kneser-Ney

Metode state-of-the-art dalam smoothing n-gram klasik adalah **Kneser-Ney Smoothing.** Ia menggunakan prinsip **absolute discounting** dan **Interpolasi** dengan n-gram yang lebih rendah.

**Intuisi Inti Kneser-Ney**
1.  **Absolute Discounting:** Mengurangi sejumlah kecil nilai absolut (misal, $d=0.7$) dari n-gram yang terlihat: $Count\_KN = Count - d$.
2.  **Distribusi Cerdas (Continuation Counts):** Memberikan probabilitas lebih tinggi pada kata yang muncul dalam konteks yang beragam. Contoh: Kata "roti" mungkin muncul setelah "makan", "beli", "bakar" (keragaman tinggi), sedangkan "Purnomo" mungkin hampir selalu hanya muncul setelah kata "Budi". KN menganggap "roti" lebih mungkin muncul di konteks baru daripada "Purnomo".

**Implementasi Menggunakan Python `dict`**
Dictionary adalah struktur data paling efisien untuk menyimpan data N-gram yang bersifat *sparse* (jarang).

```python
# Setup Data (Korpus)
bigram_counts = {
    ("makan", "nasi"): 100,
    ("makan", "roti"): 50,
    ("beli", "roti"): 30,
    ("budi", "purnomo"): 1000
}
unigram_counts = {
    "makan": 150,
    "beli": 30,
    "budi": 1000
}
d = 0.7  # Nilai diskon

# --- 1. Absolute Discounting ---
count_seen = bigram_counts.get(("makan", "nasi"), 0)
prob_seen = (count_seen - d) / unigram_counts.get("makan")
print(f"Probabilitas (Seen) P('nasi'|'makan'): {prob_seen:.4f}")

# --- 2. Continuation Counts ---
continuation_counts = {
    "nasi": 20, "roti": 50, "purnomo": 1
}
print(f"\nKeragaman Konteks 'roti': {continuation_counts['roti']}")
print(f"Keragaman Konteks 'purnomo': {continuation_counts['purnomo']}")
```

**Implementasi Menggunakan `torch.tensor`**
Hanya bisa dilakukan jika vokabulari sangat kecil karena pemborosan memori.

```python
import torch

vocab = ["makan", "nasi", "roti", "beli", "budi", "purnomo"]
word_to_idx = {word: i for i, word in enumerate(vocab)}
vocab_size = len(vocab)

bigram_counts = torch.zeros(vocab_size, vocab_size, dtype=torch.float32)
bigram_counts[word_to_idx["makan"], word_to_idx["nasi"]] = 100
bigram_counts[word_to_idx["makan"], word_to_idx["roti"]] = 50
# ... (isi count lainnya)

unigram_counts = bigram_counts.sum(dim=1)
d = 0.7
idx_makan = word_to_idx["makan"]
idx_nasi = word_to_idx["nasi"]

prob_seen = (bigram_counts[idx_makan, idx_nasi] - d) / unigram_counts[idx_makan]
print(f"Probabilitas (Seen) P('nasi'|'makan'): {prob_seen:.4f}")

continuation_counts = (bigram_counts > 0).sum(dim=0)
```

**Analisis: Mengapa `dict` Lebih Baik dari `tensor` (Untuk N-gram)**
*   **Sparsitas:** Jika $V=50,000$, tensor butuh 2.5 Miliar sel. Sebagian besar berisi nol. Ini membuang VRAM secara masif.
*   **Kesimpulan:** `dict` (Hash Map) tepat untuk N-gram karena hanya menyimpan data non-nol. `tensor` tepat untuk neural network karena *embeddings* bersifat *dense* (padat).

---

### Mini Quiz

*Silakan akses Mini Quiz melalui platform LMS:*
[Multiple Choice Quiz - Subtopik 2.2](https://lms.sdmdigital.id/h5p/embed.php?url=https%3A%2F%2Flms.sdmdigital.id%2Fpluginfile.php%2F635362%2Fmod_page%2Fcontent%2F44%2Fmultiple-choice-201.h5p)

---

> **Refleksi Singkat**
> *Kita telah melihat bahwa metode MLE (penghitungan frekuensi) akan memberikan probabilitas 0 untuk n-gram yang tidak terlihat (unseen). Jika sebuah kalimat berisi satu saja n-gram yang memiliki probabilitas 0, berapa probabilitas keseluruhan dari kalimat tersebut? Mengapa masalah "probabilitas nol" ini dianggap fatal?*